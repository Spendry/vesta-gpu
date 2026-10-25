"""
Vesta lowering: a program in GRAMMAR.md -> CUDA C source for ONE persistent
kernel that runs the whole program per launch.

    lower(text, res, axis_max, batch=1, per_token=(), outputs=(), feedback=None)
      res       : residue name -> (shape, dtype), dtype 'f32' | 'f16' | 'i32'.
                  Shapes are as the kernel sees them (batched where per-token).
                  Scalars have shape () and are read through a pointer.
      axis_max  : upper bound for axes whose extent is set at run time (e.g. s)
      batch     : B. The program is asked of B tokens at once (batch lift).
      per_token : residues that differ per token (x, the KV history)
      outputs   : named values the host reads; kept in global scratch
      feedback  : {'x': 'next', 't': 1}: after the step, emit next into x and
                  advance t, so steps run back to back with no host round trip

Mapping, one rule per form:
  law     -> a C expression per output index
  fold    -> a loop with an accumulator (step fold -> a loop over a block)
  fork    -> a C conditional
  residue -> a kernel argument; receive = read, emit = indexed store

How statements are placed (all derived from the program, none GPT-2 specific):
  tile    : a sum whose body reads a large residue along a contiguous output
            axis. A block takes 32 outputs; its 8 warps split the fold; the
            residue word is read once for every batch row (register axes).
  lane    : a fold along the contiguous index of what it reads. One warp per
            point, lanes split the fold, shuffle combine (from G3).
  thread  : one thread per point.
  private : small values (layer-norm stats, small vectors). Every block
            computes its own copy into shared memory. No grid barrier.
  group   : consecutive statements that share leading axes and only read each
            other at those axes run in one block per leading point, with block
            barriers between them instead of grid barriers.
  hoist   : a fold that does not use some output axis is lifted into its own
            statement, so it is computed once instead of once per point.
Grid barriers go only where a value crosses threads: a read of something
written since the last barrier by a different thread mapping. Elementwise
statements that read a tile/lane result at the same point run in its epilogue.
"""
import math
from dataclasses import dataclass, field
from .interp import parse

FOLD_INIT = {'sum': '0.0f', 'max': '-INFINITY', 'min': 'INFINITY'}
FOLD_STEP = {'sum': '{a} += {v};', 'max': '{a} = fmaxf({a}, {v});', 'min': '{a} = fminf({a}, {v});'}
FOLD_COMB = {'sum': '{a} + {b}', 'max': 'fmaxf({a}, {b})', 'min': 'fminf({a}, {b})'}
CFUNC = {'exp': 'expf', 'tanh': 'tanhf', 'sqrt': 'sqrtf'}
CTYPE = {'f32': 'float', 'f16': 'unsigned short', 'i32': 'int'}
NW, BLOCK, TILE = 8, 256, 32
PRIV_WORK = 1 << 15          # private if size x fold length is at most this
LANE_STORE = 8               # lane-split points per warp, stored together (8 x 4 B = one sector)
LANE_STORE_OFF = False       # switch for the sector-filling lane stores
PRIV_SIZE = 1 << 13         # ... and the value fits in 32 KB of shared memory
PRIV_BUDGET = 6144           # all private values together: 24 KB, so 4+ blocks fit per SM
STAGE_MAX = 2048             # stage a global value into shared memory only up to this size
STAGE_REG_MAX = 8192         # ... or up to this, for a register-axis input of a lane or tile site
TSH_CHUNK = 8                # tile cross-warp sums: this many accumulators per round
STAGE_PIECE_MAX = 6144       # tile k-piece staging: floats per piece (24 KB keeps 3 blocks per SM at B8)
STAGE_TILES = False          # off: measured 2.033 vs 1.98-2.08 ms at B8, no gain; the tile's slow block is its tail
GROUP_WORK = 8192            # a group statement does at most this much work per leading point
GRID_MAX = 1024              # grid reductions keep one partial per block, for up to this many blocks
PREFETCH = set()             # L2 prefetch of later tiles: 'ahead' (this step), 'wrap' (next step).
                             # Off: measured slower at batch 1 (0.847 / 0.852 / 0.853 ms vs 0.827 off)
BIG = 1 << 16                # a residue this large is never re-read per block
REG_MAX = 8                  # register axes: at most this many accumulators
LANE_MIN = 64                # lane-split a fold only if it is at least this long
KS_MAX = 2048                # split-K: work items per tile statement are capped at this

@dataclass
class Lowered:
    src: str
    params: list             # (name, dtype, emitted) in kernel argument order
    scratch: dict            # host-visible name -> (offset, axes, static shape)
    scratch_floats: int
    shared_bytes: int        # dynamic shared memory per block
    barriers_per_launch: int
    plan: list = field(default_factory=list)     # (statement, placement) for reading
    sites: list = field(default_factory=list)    # timing: the statement before each barrier site
    counters: int = 2                            # length of the bar array (barrier + split-K counters)

# ---------------------------------------------------------------- AST helpers
def nodes(e):
    if isinstance(e, tuple) and e and isinstance(e[0], str):
        yield e
        for x in e[1:]:
            if isinstance(x, list):
                for y in x: yield from nodes(y)
            else: yield from nodes(x)

def rebuild(e, f):
    """apply f bottom-up to every expression node"""
    if not (isinstance(e, tuple) and e and isinstance(e[0], str)): return e
    k = e[0]
    if k == 'idx': e = ('idx', e[1], [rebuild(i, f) for i in e[2]])
    elif k == 'fold': e = ('fold', e[1], e[2], rebuild(e[3], f))
    elif k == 'neg': e = ('neg', rebuild(e[1], f))
    elif k == 'call': e = ('call', e[1], rebuild(e[2], f))
    elif k == 'bin': e = ('bin', e[1], rebuild(e[2], f), rebuild(e[3], f))
    elif k == 'fork': e = ('fork', rebuild(e[1], f), rebuild(e[2], f), rebuild(e[3], f))
    return f(e)

def local_names(ss, out=None):
    out = set() if out is None else out
    for s in ss:
        if s[0] in ('let', 'step'): out.add(s[1])
        if s[0] == 'step': out.add(s[5]); local_names(s[6], out)
    return out

def batch_lift(stmts, B, per_token, bax='bt'):
    """'A batch is the same questions asked of several tokens at once':
    every named value and every per-token residue gains a leading token axis."""
    locs = local_names(stmts)
    def f(e):
        if e[0] == 'name' and e[1] in per_token and e[1] not in locs:
            return ('idx', e[1], [('name', bax)])
        if e[0] == 'idx' and (e[1] in locs or e[1] in per_token):
            return ('idx', e[1], [('name', bax)] + e[2])
        return e
    def st(ss):
        out = []
        for s in ss:
            k = s[0]
            if k == 'axis': out.append(('axis', s[1], rebuild(s[2], f)))
            elif k == 'recv': out.append(s)
            elif k == 'emit':
                ix = [rebuild(i, f) for i in s[2]]
                out.append(('emit', s[1], ([('name', bax)] + ix) if s[1] in per_token else ix, rebuild(s[3], f)))
            elif k == 'let': out.append(('let', s[1], [bax] + list(s[2]), rebuild(s[3], f)))
            elif k == 'step': out.append(('step', s[1], [bax] + list(s[2]), s[3], rebuild(s[4], f), s[5], st(s[6])))
            elif k == 'ret': out.append(('ret', rebuild(s[1], f)))
        return out
    return [('axis', bax, ('num', B))] + st(stmts)

# ---------------------------------------------------------------- the lowering
class Lower:
    def __init__(self, stmts, res, axis_max, outputs):
        self.res, self.axis_max, self.outputs = res, axis_max, set(outputs)
        self.ext, self.static = {}, {}
        self.steps = set()            # step variables (loop counters)
        self.emitted = set()
        self.tmp = 0
        self.inlined = []
        self.step_var, self.pending_pf = None, []
        self.barriers, self.mult = 0, 1
        self.plan = []
        self.axes_of = {}             # local -> axes
        self.alias = {}               # local -> storage name
        self.uses = {}                # local -> read count
        self.scan(stmts)
        self.stmts = self.hoist(stmts)
        self.scan_uses(self.stmts)
        self.find_aliases(self.stmts)
        self.stmts = self.inline(self.stmts)
        self.uses = {}; self.scan_uses(self.stmts)
        self.groups = {}              # id(first stmt) -> (prefix, [stmts])
        self.in_group = set()
        self.find_groups(self.stmts)
        self.private = set()
        self.find_private(self.stmts)
        self.counters = 2             # bar[0:2] is the grid barrier; split-K tile counters follow
        self.layout(self.stmts)

    # ---------------- facts about the program
    def scan(self, ss):
        for s in ss:
            if s[0] == 'axis':
                e = s[2]
                self.static[s[1]] = e[1] if e[0] == 'num' else self.axis_max[s[1]]
            if s[0] == 'emit': self.emitted.add(s[1])
            if s[0] == 'let': self.axes_of[s[1]] = tuple(s[2])
            if s[0] == 'step':
                self.steps.add(s[3]); self.axes_of[s[1]] = self.axes_of[s[5]] = tuple(s[2])
                self.scan(s[6])

    def is_axis(self, n): return n in self.static and n not in self.steps

    def free_axes(self, e):
        """axes e depends on, outside its own folds"""
        if e[0] == 'name': return {e[1]} if self.is_axis(e[1]) else (set(self.axes_of.get(e[1], ())) if e[1] in self.axes_of else set())
        if e[0] == 'fold': return self.free_axes(e[3]) - {e[2]}
        out = set()
        for x in e[1:]:
            for y in (x if isinstance(x, list) else [x]):
                if isinstance(y, tuple) and y and isinstance(y[0], str): out |= self.free_axes(y)
        return out

    def hoist(self, ss):
        """lift folds that ignore an output axis into their own statement"""
        out = []
        for s in ss:
            if s[0] == 'step':
                out.append(s[:6] + (self.hoist(s[6]),)); continue
            if s[0] != 'let': out.append(s); continue
            name, axes, e = s[1], list(s[2]), s[3]
            new = []
            def f(n, axes=axes, new=new):
                if n[0] != 'fold': return n
                fa = self.free_axes(n)
                if fa >= set(axes): return n
                hax = [a for a in axes if a in fa]
                if set(hax) != fa: return n          # depends on an axis the statement does not name
                self.tmp += 1; h = f'_h{self.tmp}'
                self.axes_of[h] = tuple(hax)
                new.append(('let', h, hax, n))
                return ('idx', h, [('name', a) for a in hax]) if hax else ('name', h)
            e2 = self.hoist_top(e, f)
            out += new + [('let', name, axes, e2)]
        return out

    def hoist_top(self, e, f):
        """apply f to outermost folds only (inner folds move with their parent)"""
        if not (isinstance(e, tuple) and e and isinstance(e[0], str)): return e
        if e[0] == 'fold': return f(e)
        k = e[0]
        if k == 'idx': return ('idx', e[1], [self.hoist_top(i, f) for i in e[2]])
        if k == 'neg': return ('neg', self.hoist_top(e[1], f))
        if k == 'call': return ('call', e[1], self.hoist_top(e[2], f))
        if k == 'bin': return ('bin', e[1], self.hoist_top(e[2], f), self.hoist_top(e[3], f))
        if k == 'fork': return ('fork', *(self.hoist_top(x, f) for x in e[1:]))
        return e

    def reads(self, e):
        """(name, index list or 'same') for every read of a local or of an
        emitted residue in e: the reads a barrier may have to order"""
        out = []
        for n in nodes(e):
            if n[0] == 'name' and n[1] in self.axes_of: out.append((n[1], 'same'))
            if n[0] == 'idx' and (n[1] in self.axes_of or n[1] in self.emitted): out.append((n[1], n[2]))
        return out

    def res_reads(self, e):
        return [n for n in nodes(e) if n[0] == 'idx' and n[1] in self.res]

    def scan_uses(self, ss):
        for s in ss:
            exprs = {'let': lambda s: [s[3]], 'emit': lambda s: s[2] + [s[3]], 'step': lambda s: [s[4]],
                     'ret': lambda s: [s[1]], 'axis': lambda s: [s[2]]}.get(s[0], lambda s: [])(s)
            for e in exprs:
                for n, _ in self.reads(e): self.uses[n] = self.uses.get(n, 0) + 1
            if s[0] == 'step': self.scan_uses(s[6])

    def find_aliases(self, ss):
        """carry <- init / carry <- ret / result <- carry need no copy when the
        value is used only there and its statement does not read the carry"""
        defs = {}
        def index(ss):
            for s in ss:
                if s[0] == 'let': defs[s[1]] = s
                if s[0] == 'step': index(s[6])
        index(ss)
        for s in ss:
            if s[0] != 'step': continue
            _, name, axes, var, init, carry, blk = s
            self.find_aliases(blk)
            ret = [b for b in blk if b[0] == 'ret'][0][1]
            for e in (init, ret):
                if e[0] == 'name' and e[1] in defs and self.uses.get(e[1]) == 1 \
                        and carry not in [n for n, _ in self.reads(defs[e[1]][3])]:
                    self.alias[e[1]] = carry
            self.alias[name] = carry

    def big_contraction(self, s, name=None):
        """a statement whose fold streams a large residue (and, given name, reads
        name inside that fold): never inline into its inner loop"""
        e = s[3] if s[0] == 'let' else s[4] if s[0] == 'step' else None
        return e is not None and any(self.big_reads(f[3]) and (name is None or any(n == name for n, _ in self.reads(f[3])))
                                     for f in self.top_folds(e))

    def inline(self, ss):
        """a fold-free named value is substituted into the statements that
        read it, instead of being stored and read back: no buffer between
        producer and consumer. Skipped for outputs, carried values, and when
        a reader is a large-residue contraction (it would recompute per step)."""
        def expand(e):          # bare local names -> explicit same-point reads
            return rebuild(e, lambda n: ('idx', n[1], [('name', a) for a in self.axes_of[n[1]]])
                           if n[0] == 'name' and n[1] in self.axes_of and self.axes_of[n[1]] else n)
        def subst(e, name, axes, body):
            def f(n):
                if n[0] == 'idx' and n[1] == name:
                    m = dict(zip(axes, n[2]))
                    return rebuild(body, lambda x: m[x[1]] if x[0] == 'name' and x[1] in m else x)
                if n[0] == 'name' and n[1] == name: return body
                return n
            return rebuild(expand(e), f)
        out = list(ss)
        i = 0
        while i < len(out):
            s = out[i]
            if s[0] == 'step':
                out[i] = s[:6] + (self.inline(s[6]),); i += 1; continue
            name = s[1] if s[0] == 'let' else None
            def reads_it(t):
                if t[0] == 'step': return any(reads_it(x) for x in [('ret', t[4])] + list(t[6]))
                exprs = {'let': [t[3]] if t[0] == 'let' else [], 'emit': t[2] + [t[3]] if t[0] == 'emit' else [],
                         'ret': [t[1]] if t[0] == 'ret' else [], 'axis': [t[2]] if t[0] == 'axis' else []}.get(t[0], [])
                return any(n == name for e in exprs for n, _ in self.reads(e))
            rs = [t for t in out[i + 1:] if reads_it(t)] if name else []
            if (name and rs and not self.top_folds(s[3]) and name not in self.outputs
                    and name not in self.alias and name not in self.alias.values()
                    and all(t[0] in ('let', 'emit') and not self.big_contraction(t) for t in rs)):
                # (also inlining into a contraction's epilogue measured slower: 0.840 vs 0.822 ms)
                body = expand(s[3]); axes = list(s[2])
                for j in range(i + 1, len(out)):
                    t = out[j]
                    if t[0] == 'let': out[j] = ('let', t[1], t[2], subst(t[3], name, axes, body))
                    elif t[0] == 'emit': out[j] = ('emit', t[1], [subst(x, name, axes, body) for x in t[2]],
                                                   subst(t[3], name, axes, body))
                del out[i]; self.inlined.append(name); continue
            i += 1
        return out

    def root(self, n):
        while n in self.alias: n = self.alias[n]
        return n

    def fold_work(self, e):
        if e[0] == 'fold': return self.static[e[2]] * self.fold_work(e[3])
        w = 1
        for x in e[1:]:
            for y in (x if isinstance(x, list) else [x]):
                if isinstance(y, tuple) and y and isinstance(y[0], str): w = max(w, self.fold_work(y))
        return w

    def big_reads(self, e):
        return [n for n in self.res_reads(e) if n[1] not in self.emitted and math.prod(self.res[n[1]][0]) > BIG]

    def find_groups(self, ss):
        i = 0
        while i < len(ss):
            s = ss[i]
            if s[0] == 'step': self.find_groups(s[6]); i += 1; continue
            best = None
            # a group: statements too big to copy per block but cheap per leading point,
            # never a tile (tiles spread better over the grid). Letting attention's
            # sc and o into a per-head group measured 1.09 ms vs 0.82: at batch 1
            # that is 12 blocks, and sc's per-thread key rows do not coalesce.
            okg = lambda t, plen: (t[0] == 'let' and not self.big_reads(t[3]) and not self.priv_ok(t)
                                   and not self.tile_ok(t) and self.point_work(t, plen) <= GROUP_WORK)
            if s[0] == 'let' and len(s[2]) >= 1:
                for plen in range(len(s[2]), 0, -1):
                    if not okg(s, plen): continue
                    P = list(s[2][:plen]); run = [s]; names = {s[1]}
                    lead = [('name', a) for a in P]
                    for t in ss[i + 1:]:
                        if t[0] != 'let' or list(t[2][:plen]) != P or not okg(t, plen): break
                        # a value made in the run may only be read at the same leading point
                        if any(n in names and ix != 'same' and list(ix[:plen]) != lead
                               for n, ix in self.reads(t[3])): break
                        run.append(t); names.add(t[1])
                    if len(run) >= 2: best = (P, run); break
            if best:
                self.groups[id(best[1][0])] = best
                for t in best[1]: self.in_group.add(id(t))
                i += len(best[1])
            else: i += 1

    def priv_ok(self, s):
        """cheap enough for every block to compute its own copy: small output,
        little work, and a small input footprint (every block reads all of it)"""
        name = s[1]
        if name in self.outputs or name in self.alias or name in self.alias.values(): return False
        size = math.prod(self.static[a] for a in s[2])
        if size > PRIV_SIZE or size * self.fold_work(s[3]) > PRIV_WORK: return False
        foot = sum(math.prod(self.static[a] for a in self.axes_of[n]) for n in {n for n, _ in self.reads(s[3])}
                   if n in self.axes_of)
        if foot > STAGE_MAX: return False
        return not any(math.prod(self.res[n[1]][0]) > BIG or n[1] in self.emitted for n in self.res_reads(s[3]))

    def tile_ok(self, s):
        """would this statement be a tile (a contraction along a contiguous output axis)"""
        tops = self.top_folds(s[3])
        return len(tops) == 1 and tops[0][1] == 'sum' and any(
            self.res[n[1]][0] and n[2][-1][0] == 'name' and n[2][-1][1] in s[2] for n in self.res_reads(tops[0][3]))

    def point_work(self, s, plen):
        return math.prod(self.static[a] for a in s[2][plen:]) * self.fold_work(s[3])

    def find_private(self, ss):
        for s in ss:
            if s[0] == 'step': self.find_private(s[6]); continue
            if s[0] == 'let' and id(s) not in self.in_group and self.priv_ok(s): self.private.add(s[1])

    def layout(self, ss):
        """global scratch for shared-across-threads values, shared memory for
        private ones (intervals over statement order, reused when dead)"""
        self.g, self.goff = {}, 0
        order, pos = [], {}
        def flat(ss, loop=None):
            for s in ss:
                if s[0] == 'step':
                    start = len(order); flat(s[6], s); end = len(order)
                    for n, (a, b) in list(pos.items()):
                        if a < start <= b < end or (a < start and b >= start): pos[n] = (a, max(b, end))
                    continue
                order.append(s)
                i = len(order) - 1
                if s[0] == 'let' and s[1] in self.private: pos.setdefault(s[1], (i, i))
                exprs = [s[3]] if s[0] == 'let' else s[2] + [s[3]] if s[0] == 'emit' else [s[1]] if s[0] == 'ret' else []
                for e in exprs:
                    for n, _ in self.reads(e):
                        n = self.root(n)
                        if n in pos: pos[n] = (pos[n][0], i)
        flat(ss)
        self.sh, used, top = {}, [], 0
        for n, (a, b) in sorted(pos.items(), key=lambda kv: kv[1][0]):
            size = math.prod(self.static[x] for x in self.axes_of[n]) or 1
            used = [u for u in used if u[1] >= a]
            off = 0
            for lo, hi_, uoff, usize in sorted(used, key=lambda u: u[2]):
                if off + size <= uoff: break
                off = max(off, uoff + usize)
            if off + size > PRIV_BUDGET:          # keep blocks per SM up: this one goes to global scratch
                self.private.discard(n); continue
            used.append((a, b, off, size)); self.sh[n] = off; top = max(top, off + size)
        self.sh_floats = self.stage_base = top
        self.staged, self.chunk = {}, {}
        for n, axes in self.axes_of.items():
            r = self.root(n)
            if r in self.private or r in self.g: continue
            self.g[r] = self.goff; self.goff += max(1, math.prod(self.static[a] for a in self.axes_of[r]))

    # ---------------- code for expressions
    def new(self, stem):
        self.tmp += 1; return f'{stem}{self.tmp}'

    def addr(self, shape, idx):
        strides, acc = [], 1
        for n in reversed(shape): strides.append(acc); acc *= n
        strides.reverse()
        return ' + '.join(f'({i}) * {s}' for i, s in zip(idx, strides)) or '0'

    def loc(self, n, idx):
        r = self.root(n); shape = tuple(self.static[a] for a in self.axes_of[r])
        a = self.addr(shape, idx)
        if r in self.staged: return f'sh_f[{self.staged[r]} + {a}]', True
        if r in self.chunk:                     # a staged k-piece: [reg flat] x [k - piece start]
            off, pce, p0, nreg = self.chunk[r]
            rf = self.addr(shape[:nreg], idx[:nreg]); kf = self.addr(shape[nreg:], idx[nreg:])
            return f'sh_f[{off} + ({rf}) * {pce} + ({kf}) - {p0}]', True
        return (f'sh_f[{self.sh[r]} + {a}]', True) if r in self.private else (f'S[{self.g[r]} + {a}]', False)

    def stage(self, ss, i, out, ind):
        """before a run of private statements: a global value that two or more
        of them read is copied into shared memory once, by the whole block"""
        run = []
        for t in ss[i:]:
            if t[0] != 'let' or self.root(t[1]) not in self.private: break
            run.append(t)
        count = {}
        for t in run:
            for n in {self.root(n) for n, _ in self.reads(t[3])}:
                if n in self.g and n in self.axes_of: count[n] = count.get(n, 0) + 1
        pick = [n for n, c in count.items() if c >= 2
                and math.prod(self.static[a] for a in self.axes_of[n]) <= STAGE_MAX]
        if not pick: return
        reads = [r for t in run for r in self.reads(t[3])]
        if self.need_barrier(reads, [], ('private',), []): self.barrier(out, ind)
        out.append(f'{ind}__syncthreads();')
        off = self.stage_base
        for n in pick:
            size = math.prod(self.static[a] for a in self.axes_of[n])
            out.append(f'{ind}#pragma unroll 4')
            out.append(f'{ind}for (int i_ = threadIdx.x; i_ < {size}; i_ += BLOCKDIM) '
                       f'sh_f[{off} + i_] = __ldcg(S + {self.g[n]} + i_);')
            self.staged[n] = off; off += size
        self.sh_floats = max(self.sh_floats, off)
        out.append(f'{ind}__syncthreads();')
        self.plan.append((','.join(pick), 'staged'))
        self.mark(out, ind, f'stage {",".join(pick)}')

    def ex(self, e, out, ind, cx):
        """C expression for e; loops it needs go to out. cx: subst, split."""
        k = e[0]
        if id(e) in cx.get('subst', {}): return cx['subst'][id(e)]
        if k == 'num':
            v = e[1]; return f'{v!r}f' if isinstance(v, float) else str(v)
        if k == 'name':
            n = e[1]
            if n in self.static: return f'ax_{n}'
            if n in self.axes_of: return self.ex(('idx', n, [('name', a) for a in self.axes_of[n]]), out, ind, cx)
            if n in self.res and self.res[n][0] == (): return f'R_{n}[0]'
            raise NameError(n)
        if k == 'neg': return f'(-{self.ex(e[1], out, ind, cx)})'
        if k == 'call': return f'{CFUNC[e[1]]}({self.ex(e[2], out, ind, cx)})'
        if k == 'bin': return f'({self.ex(e[2], out, ind, cx)} {e[1]} {self.ex(e[3], out, ind, cx)})'
        if k == 'fork':
            return f'(({self.ex(e[1], out, ind, cx)}) ? ({self.ex(e[2], out, ind, cx)}) : ({self.ex(e[3], out, ind, cx)}))'
        if k == 'idx':
            n, idx = e[1], [f'(int)({self.ex(i, out, ind, cx)})' for i in e[2]]
            if n in self.axes_of:
                ref, shared = self.loc(n, idx)
                return ref if shared else f'__ldcg({ref.replace("S[", "S + ", 1)[:-1]})'
            shape, dt = self.res[n]
            a = self.addr(shape, idx)
            raw = f'__ldcg(R_{n} + {a})' if n in self.emitted else f'R_{n}[{a}]'
            return f'h2f({raw})' if dt == 'f16' else raw
        if k == 'fold':
            op, axis, body = e[1], e[2], e[3]
            acc = self.new('acc')
            lanes = id(e) in cx.get('lanes', set())
            start, stride = ('lane', '32') if lanes else ('0', '1')
            out.append(f'{ind}float {acc} = {FOLD_INIT[op]};')
            out.append(f'{ind}#pragma unroll 8')                    # keep several loads in flight
            out.append(f'{ind}for (int ax_{axis} = {start}; ax_{axis} < {self.ext[axis]}; ax_{axis} += {stride}) {{')
            v = self.ex(body, out, ind + '    ', cx)
            out.append(ind + '    ' + FOLD_STEP[op].format(a=acc, v=v))
            out.append(f'{ind}}}')
            if lanes:
                b = f'__shfl_xor_sync(0xffffffffu, {acc}, o)'
                out.append(f'{ind}for (int o = 16; o > 0; o >>= 1) {acc} = {FOLD_COMB[op].format(a=acc, b=b)};')
            return acc
        raise ValueError(k)

    def store(self, name, axes, v):
        ref, _ = self.loc(name, [f'ax_{a}' for a in axes]); return f'{ref} = {v};'

    # ---------------- choosing a placement
    def top_folds(self, e):
        if e[0] == 'fold': return [e]
        out = []
        for x in e[1:]:
            for y in (x if isinstance(x, list) else [x]):
                if isinstance(y, tuple) and y and isinstance(y[0], str): out += self.top_folds(y)
        return out

    def names_in(self, e):
        return {n[1] for n in nodes(e) if n[0] == 'name'}

    def row_walk_node(self, fold):
        """a read whose contiguous index is the fold's axis, if the fold is long enough to split"""
        if self.static[fold[2]] < LANE_MIN: return None
        cands = [n for n in nodes(fold[3]) if n[0] == 'idx' and n[2] and n[2][-1] == ('name', fold[2])]
        # the streamed residue decides the register axes, not a small input read alongside it
        size = lambda n: math.prod(self.res[n[1]][0]) if n[1] in self.res and n[1] not in self.emitted else 0
        return max(cands, key=size) if cands else None

    def reg_axes(self, axes, node):
        used = set().union(*(self.names_in(i) for i in node[2])) if node[2] else set()
        reg = [a for a in axes if a not in used]
        return reg if 1 < math.prod(self.static[a] for a in reg) <= REG_MAX and all(
            self.ext[a] == f'({self.static[a]})' for a in reg) else []

    def placement(self, name, axes, e):
        tops = self.top_folds(e)
        if len(tops) == 1 and tops[0][1] == 'sum':
            for n in self.res_reads(tops[0][3]):
                if not self.res[n[1]][0]: continue
                lam = n[2][-1]
                if lam[0] == 'name' and lam[1] in axes:
                    reg = [a for a in self.reg_axes(axes, n) if a != lam[1]]
                    return ('tile', [tops[0]], lam[1], reg)
        npts = math.prod(self.static[a] for a in axes)
        if len(tops) == 1 and npts <= 8 and self.static[tops[0][2]] >= 16384:
            return ('gridred', tops, None, [])      # very few points, a very long fold: the whole grid
        if len(tops) == 1 and npts <= 64 and self.static[tops[0][2]] >= 4096:
            return ('block', tops, None, [])        # few points, a long fold: a whole block per point
        walk = [f for f in tops if self.row_walk_node(f)]
        if walk:
            reg = self.reg_axes(axes, self.row_walk_node(walk[0])) if len(tops) == 1 else []
            return ('lane', tops, None, reg)
        return ('thread', [], None, [])

    # ---------------- hazards and barriers
    def sig_same_point(self, reader_axes, sig, n, ix):
        r = self.root(n)
        if self.dirty.get(r) != sig or r not in self.axes_of: return False
        own = list(self.axes_of[r])
        return list(reader_axes) == own and (ix == 'same' or ix == [('name', a) for a in own])

    def need_barrier(self, reads, writes, sig, axes):
        for n, ix in reads:
            r = self.root(n)
            if r in self.private: continue
            if r in self.dirty and not self.sig_same_point(axes, sig, n, ix): return True
        for w in writes:
            if w in self.dirty and self.dirty[w] != sig: return True
            if self.readers.get(w, set()) - {sig}: return True
        return False

    def note(self, reads, writes, sig):
        for n, _ in reads:
            r = self.root(n)
            if r not in self.private: self.readers.setdefault(r, set()).add(sig)
        for w in writes: self.dirty[w] = sig

    def mark(self, out, ind, label):
        """timing: time since the previous mark, summed per site (thread 0's view)"""
        if not self.timing: return
        k = len(self.sites); self.sites.append(label)
        out.append(f'{ind}if (gtid == 0) {{ unsigned long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); '
                   f'TS[{k}] += t_ - tlast; tlast = t_; }}')

    def barrier(self, out, ind):
        out.append(f'{ind}grid_sync(bar, nblk);'); self.barriers += self.mult
        self.mark(out, ind, '  (barrier wait)')
        self.dirty, self.readers = {}, {}
        # prefetches queued by the last tile go out here: its own reads are done, DRAM is free
        for st, shift in self.pending_pf: self.prefetch(st, shift, out, ind)
        if self.pending_pf: self.mark(out, ind, '  (prefetch issue)')
        self.pending_pf = []

    # ---------------- spreading points
    def spread(self, axes, scope, unit, body_fn, out, ind):
        me, n = {('grid', 'thread'): ('gtid', 'nthr'), ('grid', 'warp'): ('gwarp', 'nwarp'),
                 ('block', 'thread'): ('(int)threadIdx.x', 'BLOCKDIM'), ('block', 'warp'): ('warp', 'NWARP')}[(scope, unit)]
        if not axes:
            out.append(f'{ind}if ({me} == 0) {{'); body_fn(out, ind + '    '); out.append(f'{ind}}}'); return
        total = ' * '.join(f'({self.ext[a]})' for a in axes)
        p = self.new('p')
        if scope == 'block': out.append(f'{ind}#pragma unroll 4')     # a few points per thread: overlap their loads
        out.append(f'{ind}for (int {p} = {me}; {p} < {total}; {p} += {n}) {{')
        out.append(f'{ind}    int r_{p} = {p};')
        for a in reversed(axes):
            out.append(f'{ind}    int ax_{a} = r_{p} % ({self.ext[a]}); r_{p} /= ({self.ext[a]});')
        body_fn(out, ind + '    ')
        out.append(f'{ind}}}')

    def spread_lanes(self, axes, RN, value_fn, store_fn, out, ind):
        """grid warps, one lane-split point each, as before; but a block's 8 warps take
        8 neighbouring points, park their values in shared memory, and 8*RN threads
        store them together. One 4 B store from one lane costs a whole 32 B sector;
        8 neighbouring values fill it. value_fn(o, i) leaves value rr in tsh[warp*RN+rr]
        (lane 0); store_fn(o, i, src) stores src at the decoded point."""
        total = ' * '.join(f'({self.ext[a]})' for a in axes)
        c = self.new('c')
        def decode(o, i, flat):
            o.append(f'{i}int r_{c} = {flat};')
            for a in reversed(axes):
                o.append(f'{i}const int ax_{a} = r_{c} % ({self.ext[a]}); r_{c} /= ({self.ext[a]});')
        out.append(f'{ind}for (int {c} = blockIdx.x * NWARP; {c} < {total}; {c} += gridDim.x * NWARP) {{   // uniform per block')
        i1 = ind + '    '
        out.append(f'{i1}if ({c} + warp < {total}) {{')
        decode(out, i1 + '    ', f'{c} + warp')
        value_fn(out, i1 + '    ')
        out.append(f'{i1}}}')
        out.append(f'{i1}__syncthreads();')
        out.append(f'{i1}if ((int)threadIdx.x < NWARP * {RN}) {{')
        out.append(f'{i1}    const int w_ = threadIdx.x % NWARP, rr = threadIdx.x / NWARP;')
        out.append(f'{i1}    if ({c} + w_ < {total}) {{')
        decode(out, i1 + '        ', f'{c} + w_')
        store_fn(out, i1 + '        ', f'tsh[w_ * {RN} + rr]')
        out.append(f'{i1}    }}')
        out.append(f'{i1}}}')
        out.append(f'{i1}__syncthreads();')
        out.append(f'{ind}}}')

    def decode_reg(self, reg, rr, out, ind):
        r = self.new('rr'); out.append(f'{ind}int {r} = {rr};')
        for a in reversed(reg):
            out.append(f'{ind}const int ax_{a} = {r} % {self.static[a]}; {r} /= {self.static[a]};')

    def simple(self, name, axes, e, scope, out, ind, points=None):
        """thread or lane placement, no register axes, inside a block or the grid.
        points: the axes to spread over, when some are already fixed (groups)"""
        tops = self.top_folds(e)
        lanes = {id(f) for f in tops} if any(self.row_walk_node(f) for f in tops) else set()
        if scope == 'block' and math.prod(self.static[a] for a in (axes if points is None else points)) > 2 * NW:
            lanes = set()            # a block has only 8 warps: with many points, one thread per point
        unit = 'warp' if lanes else 'thread'
        pts = list(axes if points is None else points)
        if lanes and scope == 'grid' and pts and not LANE_STORE_OFF:
            def value(o, i):
                v = self.ex(e, o, i, {'lanes': lanes})
                o.append(f'{i}if (lane == 0) tsh[warp] = {v};')
            self.spread_lanes(pts, 1, value, lambda o, i, src: o.append(f'{i}{self.store(name, axes, src)}'), out, ind)
            return unit
        def body(o, i):
            v = self.ex(e, o, i, {'lanes': lanes})
            guard = 'if (lane == 0) ' if lanes else ''
            o.append(f'{i}{guard}{self.store(name, axes, v)}')
        self.spread(pts, scope, unit, body, out, ind)
        return unit

    def blockred(self, name, axes, e, fold, out, ind, scope='grid', points=None):
        """one block per point (scope 'grid'), or each point in turn by this block (scope 'block'):
        all threads split the fold, then shuffle and shared combine"""
        op, kax, body = fold[1], fold[2], fold[3]
        acc, v = self.new('acc'), self.new('v')
        comb = lambda a, b: FOLD_COMB[op].format(a=a, b=b)
        pts = list(axes if points is None else points)
        total = ' * '.join(f'({self.ext[a]})' for a in pts) or '1'
        pb = self.new('pb')
        first, step = ('blockIdx.x', 'gridDim.x') if scope == 'grid' else ('0', '1')
        out.append(f'{ind}for (int {pb} = {first}; {pb} < {total}; {pb} += {step}) {{')
        i2 = ind + '    '
        out.append(f'{i2}int r_{pb} = {pb};')
        for a in reversed(pts):
            out.append(f'{i2}const int ax_{a} = r_{pb} % ({self.ext[a]}); r_{pb} /= ({self.ext[a]});')
        out.append(f'{i2}float {acc} = {FOLD_INIT[op]};')
        out.append(f'{i2}for (int ax_{kax} = threadIdx.x; ax_{kax} < {self.ext[kax]}; ax_{kax} += BLOCKDIM) {{')
        x = self.ex(body, out, i2 + '    ', {})
        out.append(f'{i2}    {FOLD_STEP[op].format(a=acc, v=x)}')
        out.append(f'{i2}}}')
        out.append(f'{i2}for (int o = 16; o > 0; o >>= 1) {acc} = {comb(acc, "__shfl_xor_sync(0xffffffffu, " + acc + ", o)")};')
        out.append(f'{i2}if (lane == 0) tsh[warp] = {acc};')
        out.append(f'{i2}__syncthreads();')
        out.append(f'{i2}if (threadIdx.x == 0) {{')
        out.append(f'{i2}    float {v} = tsh[0];')
        out.append(f'{i2}    for (int w_ = 1; w_ < NWARP; ++w_) {v} = {comb(v, "tsh[w_]")};')
        out.append(f'{i2}    {self.store(name, axes, self.ex(e, out, i2 + "    ", {"subst": {id(fold): v}}))}')
        out.append(f'{i2}}}')
        out.append(f'{i2}__syncthreads();')
        out.append(f'{ind}}}')

    def tile_geom(self, axes, fold, lam, reg, bound=()):
        """how a tile statement is cut into work items (shared by the tile and its prefetch)"""
        kaxes, kbody = [fold[2]], fold[3]
        while kbody[0] == 'fold' and kbody[1] == 'sum':
            kaxes.append(kbody[2]); kbody = kbody[3]
        vnode = next((n for n in nodes(kbody) if n[0] == 'idx' and n[1] in self.res and self.res[n[1]][0]
                      and n[2][-1] == ('name', lam)), None)
        VEC, lst = 1, self.static[lam]
        if vnode is not None and self.ext[lam] == f'({lst})':
            for cand in (4, 2):
                if lst % cand == 0 and lst >= TILE * cand and self.res[vnode[1]][0][-1] % cand == 0:
                    VEC = cand; break
        rest = [a for a in axes if a != lam and a not in reg and a not in bound]
        span = TILE * VEC
        nl = f'(({self.ext[lam]} + {span - 1}) / {span})'
        ntile_static = math.prod([self.static[a] for a in rest] + [-(-self.static[lam] // span)])
        kstatic = math.prod(self.static[a] for a in kaxes)
        return dict(kaxes=kaxes, kbody=kbody, kext=' * '.join(f'{self.ext[a]}' for a in kaxes), kstatic=kstatic,
                    vnode=vnode, VEC=VEC, rest=rest, span=span, nl=nl, lam=lam,
                    ntile=' * '.join([f'({self.ext[a]})' for a in rest] + [nl]), ntile_static=ntile_static,
                    # split-K factor, chosen at run time so the work items fill one wave of the grid
                    KCAP=max(1, min(KS_MAX // ntile_static, kstatic // (4 * NW))))

    def item_head(self, g, tl, out, ind, scope='grid'):
        """open the work-item loop of a tile and decode the item (scope 'block': every item in this block)"""
        KS = f'KS_{tl}'; i2 = ind + '    '
        if scope == 'block':
            out.append(f'{ind}const int {KS} = 1;')
            out.append(f'{ind}for (int {tl} = 0; {tl} < ({g["ntile"]}); ++{tl}) {{')
        else:
            out.append(f'{ind}const int {KS} = max(1, min({g["KCAP"]}, (int)gridDim.x / ({g["ntile"]})));')
            out.append(f'{ind}for (int {tl} = blockIdx.x; {tl} < ({g["ntile"]}) * {KS}; {tl} += gridDim.x) {{')
        out.append(f'{i2}const int ks_{tl} = {tl} % {KS}; const int tile_{tl} = {tl} / {KS};')
        out.append(f'{i2}int r_{tl} = tile_{tl}; const int lt_{tl} = r_{tl} % {g["nl"]}; r_{tl} /= {g["nl"]};')
        for a in reversed(g['rest']):
            out.append(f'{i2}const int ax_{a} = r_{tl} % ({self.ext[a]}); r_{tl} /= ({self.ext[a]});')
        out.append(f'{i2}const int l0_{tl} = (lt_{tl} * {TILE} + lane) * {g["VEC"]}; '
                   f'const bool ok_{tl} = l0_{tl} < {self.ext[g["lam"]]};')
        out.append(f'{i2}const int kn_{tl} = {g["kext"]}, kc_{tl} = (kn_{tl} + {KS} - 1) / {KS};')
        out.append(f'{i2}const int k0_{tl} = ks_{tl} * kc_{tl}, k1_{tl} = min(kn_{tl}, k0_{tl} + kc_{tl});')

    def prefetch(self, s, shift, out, ind):
        """ask L2 for the residue this block will read in a later tile statement
        (shift=1: the same statement in the next step iteration). DRAM idles
        through attention and layer norm; this puts it to work early."""
        _, name, axes, e = s
        kind, folds, lam, reg = self.placement(name, axes, e)
        if kind != 'tile': return 0
        g = self.tile_geom(axes, folds[0], lam, reg)
        vn = g['vnode']
        if vn is None or vn[1] in self.emitted: return 0
        shape, dt = self.res[vn[1]]
        per_line = max(1, 128 // (g['VEC'] * (2 if dt == 'f16' else 4)))
        out.append(f'{ind}{{')
        i1 = ind + '    '
        if shift:
            v = self.step_var
            out.append(f'{i1}const int nx_ = ax_{v} + 1;')
            out.append(f'{i1}if (nx_ < {self.ext[v]}) {{ const int ax_{v} = nx_;')
        tl = self.new('pf')
        self.item_head(g, tl, out, i1)
        i2 = i1 + '    '
        out.append(f'{i2}for (int kk = k0_{tl} + warp; kk < (ok_{tl} && (lane % {per_line}) == 0 ? k1_{tl} : 0); kk += NWARP) {{')
        out.append(f'{i2}    int rk = kk;')
        for a in reversed(g['kaxes']):
            out.append(f'{i2}    const int ax_{a} = rk % ({self.ext[a]}); rk /= ({self.ext[a]});')
        out.append(f'{i2}    const int ax_{lam} = l0_{tl};')
        ix = [f'(int)({self.ex(x, out, i2 + "    ", {})})' for x in vn[2]]
        out.append(f'{i2}    asm volatile("prefetch.global.L2 [%0];" :: "l"(R_{vn[1]} + {self.addr(shape, ix)}));')
        out.append(f'{i2}}}')
        out.append(f'{i1}}}')
        if shift: out.append(f'{i1}}}')
        out.append(f'{ind}}}')
        return math.prod(shape[-2:]) if len(shape) >= 2 else 0

    def gridred(self, name, axes, e, fold, out, ind):
        """every block reduces one chunk of the fold for every point; the last
        block to finish combines the per-block partials in block order"""
        op, kax, body = fold[1], fold[2], fold[3]
        npts = math.prod(self.static[a] for a in axes)
        comb = lambda a, b: FOLD_COMB[op].format(a=a, b=b)
        P = self.goff; self.goff += npts * GRID_MAX
        C = self.counters; self.counters += 1
        acc, v = self.new('acc'), self.new('v')
        out.append(f'{ind}{{')
        i1 = ind + '    '
        out.append(f'{i1}const int kn_ = {self.ext[kax]}, kc_ = (kn_ + gridDim.x - 1) / gridDim.x;')
        out.append(f'{i1}const int k0_ = blockIdx.x * kc_, k1_ = min(kn_, k0_ + kc_);')
        out.append(f'{i1}for (int pt_ = 0; pt_ < {npts}; ++pt_) {{')
        i2 = i1 + '    '
        out.append(f'{i2}int r_ = pt_;')
        for a in reversed(axes):
            out.append(f'{i2}const int ax_{a} = r_ % {self.static[a]}; r_ /= {self.static[a]};')
        out.append(f'{i2}float {acc} = {FOLD_INIT[op]};')
        out.append(f'{i2}for (int ax_{kax} = k0_ + threadIdx.x; ax_{kax} < k1_; ax_{kax} += BLOCKDIM) {{')
        x = self.ex(body, out, i2 + '    ', {})
        out.append(f'{i2}    {FOLD_STEP[op].format(a=acc, v=x)}')
        out.append(f'{i2}}}')
        out.append(f'{i2}for (int o = 16; o > 0; o >>= 1) {acc} = {comb(acc, "__shfl_xor_sync(0xffffffffu, " + acc + ", o)")};')
        out.append(f'{i2}if (lane == 0) tsh[warp] = {acc};')
        out.append(f'{i2}__syncthreads();')
        out.append(f'{i2}if (threadIdx.x == 0) {{ float t_ = tsh[0]; for (int w_ = 1; w_ < NWARP; ++w_) t_ = {comb("t_", "tsh[w_]")}; '
                   f'S[{P} + pt_ * {GRID_MAX} + blockIdx.x] = t_; }}')
        out.append(f'{i2}__syncthreads();')
        out.append(f'{i1}}}')
        out.append(f'{i1}__threadfence(); __syncthreads();')
        out.append(f'{i1}if (threadIdx.x == 0) last_ = atomicAdd(bar + {C}, 1u) == gridDim.x - 1;')
        out.append(f'{i1}__syncthreads();')
        out.append(f'{i1}if (last_) {{')
        out.append(f'{i1}    for (int pt_ = warp; pt_ < {npts}; pt_ += NWARP) {{')
        i3 = i1 + '        '
        out.append(f'{i3}float {v} = {FOLD_INIT[op]};')
        out.append(f'{i3}for (int b_ = lane; b_ < gridDim.x; b_ += 32) {v} = {comb(v, "__ldcg(S + " + str(P) + " + pt_ * " + str(GRID_MAX) + " + b_)")};')
        out.append(f'{i3}for (int o = 16; o > 0; o >>= 1) {v} = {comb(v, "__shfl_xor_sync(0xffffffffu, " + v + ", o)")};')
        out.append(f'{i3}if (lane == 0) {{')
        out.append(f'{i3}    int r_ = pt_;')
        for a in reversed(axes):
            out.append(f'{i3}    const int ax_{a} = r_ % {self.static[a]}; r_ /= {self.static[a]};')
        out.append(f'{i3}    {self.store(name, axes, self.ex(e, out, i3 + "    ", {"subst": {id(fold): v}}))}')
        out.append(f'{i3}}}')
        out.append(f'{i1}    }}')
        out.append(f'{i1}    if (threadIdx.x == 0) bar[{C}] = 0u;')
        out.append(f'{i1}}}')
        out.append(f'{ind}}}')

    def split(self, name, axes, e, kind, folds, lam, reg, epi, out, ind, emits=(), scope='grid', bound=()):
        """tile or lane placement with register axes and epilogue statements"""
        fold = folds[0]
        RN = math.prod(self.static[a] for a in reg) if reg else 1
        op, kax, body = fold[1], fold[2], fold[3]
        acc = self.new('acc'); v = self.new('v')
        def finish(o, i):
            o.append(f'{i}for (int rr = 0; rr < {RN}; ++rr) {{')
            self.decode_reg(reg, 'rr', o, i + '    ')
            o.append(f'{i}    float {v} = {acc}[rr];')
            val = self.ex(e, o, i + '    ', {'subst': {id(fold): v}})
            o.append(f'{i}    {self.store(name, axes, val)}')
            for (en, eax, ee) in epi:
                o.append(f'{i}    {self.store(en, eax, self.ex(ee, o, i + "    ", {}))}')
            o.append(f'{i}}}')
        def kloop(o, i, start, stride, guard):
            o.append(f'{i}float {acc}[{RN}];')
            o.append(f'{i}for (int rr = 0; rr < {RN}; ++rr) {acc}[rr] = {FOLD_INIT[op]};')
            o.append(f'{i}if ({guard}) for (int ax_{kax} = {start}; ax_{kax} < {self.ext[kax]}; ax_{kax} += {stride}) {{')
            o.append(f'{i}    #pragma unroll')
            o.append(f'{i}    for (int rr = 0; rr < {RN}; ++rr) {{')
            self.decode_reg(reg, 'rr', o, i + '        ')
            x = self.ex(body, o, i + '        ', {})
            o.append(f'{i}        {FOLD_STEP[op].format(a=acc + "[rr]", v=x)}')
            o.append(f'{i}    }}')
            o.append(f'{i}}}')
        if kind == 'lane':
            pts = [a for a in axes if a not in reg]
            def body_fn(o, i):
                kloop(o, i, 'lane', '32', 'true')
                o.append(f'{i}for (int rr = 0; rr < {RN}; ++rr) for (int o = 16; o > 0; o >>= 1) '
                         f'{acc}[rr] = {FOLD_COMB[op].format(a=acc + "[rr]", b="__shfl_xor_sync(0xffffffffu, " + acc + "[rr], o)")};')
                o.append(f'{i}if (lane == 0) {{'); finish(o, i + '    '); o.append(f'{i}}}')
            # register-axis inputs: a value the fold reads that depends only on the
            # register axes and the fold axis is the same for every point. Each block
            # copies it to shared memory once, instead of every point reading it from L2.
            ok_axes = set(reg) | {kax}
            pick = sorted({self.root(n) for n, _ in self.reads(body)
                           if self.root(n) in self.g and self.root(n) in self.axes_of
                           and set(self.axes_of[self.root(n)]) <= ok_axes} - set(self.staged)) if reg else []
            pick = [n for n in pick if math.prod(self.static[a] for a in self.axes_of[n]) <= STAGE_REG_MAX]
            if pick:
                out.append(f'{ind}__syncthreads();')
                off = self.stage_base
                for n in pick:
                    size = math.prod(self.static[a] for a in self.axes_of[n])
                    out.append(f'{ind}#pragma unroll 4')
                    out.append(f'{ind}for (int i_ = threadIdx.x; i_ < {size}; i_ += BLOCKDIM) '
                               f'sh_f[{off} + i_] = __ldcg(S + {self.g[n]} + i_);')
                    self.staged[n] = off; off += size
                self.sh_floats = max(self.sh_floats, off)
                out.append(f'{ind}__syncthreads();')
            if pts and not epi and not LANE_STORE_OFF:
                # store LANE_STORE points together (full sectors), one value per register axis
                def value(o, i):
                    kloop(o, i, 'lane', '32', 'true')
                    o.append(f'{i}for (int rr = 0; rr < {RN}; ++rr) for (int o = 16; o > 0; o >>= 1) '
                             f'{acc}[rr] = {FOLD_COMB[op].format(a=acc + "[rr]", b="__shfl_xor_sync(0xffffffffu, " + acc + "[rr], o)")};')
                    o.append(f'{i}if (lane == 0) for (int rr = 0; rr < {RN}; ++rr) {{')
                    self.decode_reg(reg, 'rr', o, i + '    ')
                    o.append(f'{i}    float {v} = {acc}[rr];')
                    o.append(f'{i}    tsh[warp * {RN} + rr] = {self.ex(e, o, i + "    ", {"subst": {id(fold): v}})};')
                    o.append(f'{i}}}')
                def store(o, i, src):
                    self.decode_reg(reg, 'rr', o, i)
                    o.append(f'{i}{self.store(name, axes, src)}')
                self.spread_lanes(pts, RN, value, store, out, ind)
            else:
                self.spread(pts, 'grid', 'warp', body_fn, out, ind)
            for n in pick: self.staged.pop(n, None)
            if pick: self.plan.append((','.join(pick), 'staged for ' + name))
            return
        # tile: a work item is TILE*VEC consecutive lam points x one chunk of
        # the fold. Warps split the chunk; nested sums are one flattened fold.
        # Each lane takes VEC adjacent points with one vector load of the
        # residue that runs along lam, so a warp reads TILE*VEC words per row.
        kaxes, kbody = [kax], body
        g = self.tile_geom(axes, fold, lam, reg, bound)
        kaxes, kbody, kext, kstatic, vnode, VEC, rest, span, nl, ntile, ntile_static, KCAP = (
            g[k] for k in ('kaxes', 'kbody', 'kext', 'kstatic', 'vnode', 'VEC', 'rest', 'span', 'nl', 'ntile',
                           'ntile_static', 'KCAP'))
        AW = RN * VEC                                    # accumulators per lane
        tl = self.new('w')
        KS = f'KS_{tl}'
        self.item_head(g, tl, out, ind, scope)
        i2 = ind + '    '
        out.append(f'{i2}float {acc}[{AW}];')
        out.append(f'{i2}for (int q_ = 0; q_ < {AW}; ++q_) {acc}[q_] = 0.0f;')
        # register-axis inputs laid out [reg..., k...]: each block copies its own k-slice
        # to shared memory, in pieces that fit STAGE_REG_MAX, instead of every lane
        # reading them from L2 for every weight word
        static_k = all(self.ext[a] == f'({self.static[a]})' for a in kaxes)
        pickc = sorted({self.root(n) for n, _ in self.reads(kbody)
                        if self.root(n) in self.g and self.root(n) not in self.staged
                        and list(self.axes_of.get(self.root(n), ())) == list(reg) + list(kaxes)}) if STAGE_TILES and reg and static_k else []
        if pickc:
            PCE = STAGE_PIECE_MAX // (RN * len(pickc))
            p0 = f'p0_{tl}'
            out.append(f'{i2}for (int {p0} = k0_{tl}; {p0} < k1_{tl}; {p0} += {PCE}) {{')
            out.append(f'{i2}  const int pn_ = min(k1_{tl} - {p0}, {PCE});')
            out.append(f'{i2}  __syncthreads();')
            off = self.stage_base
            for n in pickc:
                out.append(f'{i2}  for (int i_ = threadIdx.x; i_ < {RN} * pn_; i_ += BLOCKDIM) {{ const int r_ = i_ / pn_, q_ = i_ % pn_; '
                           f'sh_f[{off} + r_ * {PCE} + q_] = __ldcg(S + {self.g[n]} + r_ * {kstatic} + {p0} + q_); }}')
                self.chunk[n] = (off, PCE, p0, len(reg)); off += RN * PCE
            self.sh_floats = max(self.sh_floats, off)
            out.append(f'{i2}  __syncthreads();')
            klo, khi = p0, f'{p0} + pn_'
        else:
            klo, khi = f'k0_{tl}', f'k1_{tl}'
        out.append(f'{i2}#pragma unroll 4')
        out.append(f'{i2}for (int kk = {klo} + warp; kk < (ok_{tl} ? {khi} : 0); kk += NWARP) {{')
        out.append(f'{i2}    int rk = kk;')
        for a in reversed(kaxes):
            out.append(f'{i2}    const int ax_{a} = rk % ({self.ext[a]}); rk /= ({self.ext[a]});')
        sub = {}
        if VEC > 1:
            shape, dt = self.res[vnode[1]]
            vt = {('f16', 4): 'uint2', ('f16', 2): 'unsigned int', ('f32', 4): 'float4', ('f32', 2): 'float2'}[(dt, VEC)]
            vw = self.new('vw')
            out.append(f'{i2}    {vt} {vw};')
            out.append(f'{i2}    {{ const int ax_{lam} = l0_{tl};')
            ix = [f'(int)({self.ex(x, out, i2 + "      ", {})})' for x in vnode[2]]
            ptr = f'(const {vt}*)(R_{vnode[1]} + {self.addr(shape, ix)})'
            out.append(f'{i2}      {vw} = {"__ldcg(" + ptr + ")" if vnode[1] in self.emitted else "*" + ptr}; }}')
            sub = {id(vnode): f'{"hv" if dt == "f16" else "fv"}({vw}, vv)'}
        out.append(f'{i2}    #pragma unroll')
        out.append(f'{i2}    for (int vv = 0; vv < {VEC}; ++vv) {{')
        out.append(f'{i2}      const int ax_{lam} = l0_{tl} + vv;')
        out.append(f'{i2}      #pragma unroll')
        out.append(f'{i2}      for (int rr = 0; rr < {RN}; ++rr) {{')
        self.decode_reg(reg, 'rr', out, i2 + '        ')
        x = self.ex(kbody, out, i2 + '        ', {'subst': sub})
        out.append(f'{i2}        {acc}[rr * {VEC} + vv] += {x};')
        out.append(f'{i2}      }}')
        out.append(f'{i2}    }}')
        out.append(f'{i2}}}')
        if pickc:
            out.append(f'{i2}}}')                       # end of the staged piece
            for n in pickc: self.chunk.pop(n, None)
            self.plan.append((','.join(pickc), f'staged in k-pieces for {name}'))
        # cross-warp sums go CH accumulators at a time, so the shared buffer stays small
        CH = min(AW, TSH_CHUNK)
        out.append(f'{i2}#pragma unroll')
        out.append(f'{i2}for (int c_ = 0; c_ < {AW}; c_ += {CH}) {{')
        out.append(f'{i2}    #pragma unroll')
        out.append(f'{i2}    for (int q_ = 0; q_ < {CH}; ++q_) tsh[(warp * {CH} + q_) * 32 + lane] = {acc}[c_ + q_];')
        out.append(f'{i2}    __syncthreads();')
        out.append(f'{i2}    #pragma unroll')
        out.append(f'{i2}    if (warp == 0) for (int q_ = 0; q_ < {CH}; ++q_) {{ float t_ = 0.0f; '
                   f'for (int w_ = 0; w_ < NWARP; ++w_) t_ += tsh[(w_ * {CH} + q_) * 32 + lane]; {acc}[c_ + q_] = t_; }}')
        out.append(f'{i2}    __syncthreads();')
        out.append(f'{i2}}}')
        def finish_t(o, i, value=None):
            """store, epilogues and emits for each (vv, rr); with value given, for the one
            (vv, rr) already in scope, whose sum is value"""
            if value is None:
                o.append(f'{i}#pragma unroll')
                o.append(f'{i}for (int vv = 0; vv < {VEC}; ++vv) {{')
                o.append(f'{i}  for (int rr = 0; rr < {RN}; ++rr) {{')
            else:
                o.append(f'{i}{{ {{')
            o.append(f'{i}    const int ax_{lam} = l0_{tl} + vv;')
            self.decode_reg(reg, 'rr', o, i + '    ')
            o.append(f'{i}    float {v} = {acc + "[rr * " + str(VEC) + " + vv]" if value is None else value};')
            val = self.ex(e, o, i + '    ', {'subst': {id(fold): v}})
            o.append(f'{i}    {self.store(name, axes, val)}')
            for (en, eax, ee) in epi:
                o.append(f'{i}    {self.store(en, eax, self.ex(ee, o, i + "    ", {}))}')
            for (st, c, order, co, span_) in emits:
                _, rn, idx, ev = st
                o.append(f'{i}    {{ const int jj_ = ax_{lam} - ({c});')
                o.append(f'{i}      if (jj_ >= 0 && jj_ < {span_}) {{')
                for a in order:
                    o.append(f'{i}        const int ax_{a} = (jj_ / {co[a]}) % {self.static[a]};')
                ix = [f'(int)({self.ex(x, o, i + "        ", {})})' for x in idx]
                ev_c = self.ex(ev, o, i + '        ', {})
                shape, dt = self.res[rn]
                o.append(f'{i}        R_{rn}[{self.addr(shape, ix)}] = {"f2h(" + ev_c + ")" if dt == "f16" else ev_c}; }} }}')
            o.append(f'{i}  }}')
            o.append(f'{i}}}')
        if KCAP == 1 or scope == 'block':
            out.append(f'{i2}if (warp == 0 && ok_{tl}) {{'); finish_t(out, i2 + '    '); out.append(f'{i2}}}')
        else:
            # partial sums to scratch; the last block of this tile adds them in order
            P = self.goff; self.goff += ntile_static * KCAP * AW * 32
            C = self.counters; self.counters += ntile_static
            out.append(f'{i2}if (warp == 0) for (int q_ = 0; q_ < {AW}; ++q_) '
                       f'S[{P} + ((tile_{tl} * {KS} + ks_{tl}) * {AW} + q_) * 32 + lane] = {acc}[q_];')
            # (an acq_rel atomic in place of the fence measured slower: 0.839 vs 0.819 ms)
            out.append(f'{i2}__threadfence(); __syncthreads();')
            out.append(f'{i2}if (threadIdx.x == 0) last_ = atomicAdd(bar + {C} + tile_{tl}, 1u) == {KS} - 1;')
            out.append(f'{i2}__syncthreads();')
            out.append(f'{i2}if (last_) {{')
            if AW < NW:
                # few accumulators (batch 1): all warps add a share of the partials, then
                # warp 0 adds the shares and finishes. (Spreading the finish measured
                # slower here: 0.781 vs 0.770 ms at B1; only AW warps would work.)
                out.append(f'{i2}    #pragma unroll')
                out.append(f'{i2}    for (int c_ = 0; c_ < {AW}; c_ += {CH}) {{')
                out.append(f'{i2}        for (int q_ = 0; q_ < {CH}; ++q_) {{ float t_ = 0.0f;')
                out.append(f'{i2}            #pragma unroll 4')
                out.append(f'{i2}            for (int s_ = warp; s_ < {KS}; s_ += NWARP) '
                           f't_ += __ldcg(S + {P} + ((tile_{tl} * {KS} + s_) * {AW} + c_ + q_) * 32 + lane);')
                out.append(f'{i2}            tsh[(warp * {CH} + q_) * 32 + lane] = t_; }}')
                out.append(f'{i2}        __syncthreads();')
                out.append(f'{i2}        if (warp == 0) for (int q_ = 0; q_ < {CH}; ++q_) {{ float t_ = 0.0f; '
                           f'for (int w_ = 0; w_ < NWARP; ++w_) t_ += tsh[(w_ * {CH} + q_) * 32 + lane]; {acc}[c_ + q_] = t_; }}')
                out.append(f'{i2}        __syncthreads();')
                out.append(f'{i2}    }}')
                out.append(f'{i2}    if (warp == 0 && ok_{tl}) {{')
                finish_t(out, i2 + '        ')
                out.append(f'{i2}    }}')
            else:
                # the tail, spread: warp w sums and finishes accumulators w, w+8, ...
                # The add order is the one warp 0 used alone: each warp-strided share of
                # the partials in turn, then the shares in warp order. Same bits out.
                out.append(f'{i2}    if (ok_{tl}) for (int q_ = warp; q_ < {AW}; q_ += NWARP) {{')
                out.append(f'{i2}        float t_ = 0.0f;')
                out.append(f'{i2}        for (int w_ = 0; w_ < NWARP; ++w_) {{ float u_ = 0.0f;')
                out.append(f'{i2}            #pragma unroll 4')
                out.append(f'{i2}            for (int s_ = w_; s_ < {KS}; s_ += NWARP) '
                           f'u_ += __ldcg(S + {P} + ((tile_{tl} * {KS} + s_) * {AW} + q_) * 32 + lane);')
                out.append(f'{i2}            t_ += u_; }}')
                out.append(f'{i2}        const int vv = q_ % {VEC}, rr = q_ / {VEC};')
                finish_t(out, i2 + '        ', value='t_')
                out.append(f'{i2}    }}')
            out.append(f'{i2}    if (threadIdx.x == 0) bar[{C} + tile_{tl}] = 0u;')
            out.append(f'{i2}}}')
        out.append(f'{i2}__syncthreads();')
        out.append(f'{ind}}}')
        self.tile_rn = max(self.tile_rn, CH)

    # ---------------- statements
    def gen(self, ss, out, ind):
        i = 0
        while i < len(ss):
            s = ss[i]; k = s[0]
            if k == 'axis':
                self.ext[s[1]] = f'({self.ex(s[2], out, ind, {})})'; i += 1; continue
            if k in ('recv', 'ret'): i += 1; continue
            if id(s) in self.groups:
                P, run = self.groups[id(s)]
                i += len(run); self.gen_group(P, run, out, ind); self.mark(out, ind, ' '.join(self.plan[-1])); continue
            if k == 'emit':
                _, name, idx, e = s
                axes = []
                for x in idx:
                    for n in nodes(x):
                        if n[0] == 'name' and self.is_axis(n[1]) and n[1] not in axes: axes.append(n[1])
                reads = self.reads(e) + [r for x in idx for r in self.reads(x)]
                sig = ('thread', tuple(axes))
                if self.need_barrier(reads, [name], sig, axes): self.barrier(out, ind)
                def body(o, ii, name=name, idx=idx, e=e):
                    ix = [f'(int)({self.ex(x, o, ii, {})})' for x in idx]
                    v = self.ex(e, o, ii, {})
                    shape, dt = self.res[name]
                    o.append(f'{ii}R_{name}[{self.addr(shape, ix)}] = {"f2h(" + v + ")" if dt == "f16" else v};')
                self.spread(axes, 'grid', 'thread', body, out, ind)
                self.note(reads, [name], sig); self.plan.append((f'emit {name}', 'thread')); i += 1
                self.mark(out, ind, f'emit {name}'); continue
            if k == 'let':
                i = self.gen_let(ss, i, out, ind); self.mark(out, ind, ' '.join(self.plan[-1])); continue
            if k == 'step':
                self.gen_step(s, out, ind); i += 1; continue
            raise ValueError(k)

    def gen_let(self, ss, i, out, ind):
        _, name, axes, e = ss[i]
        r = self.root(name)
        if r in self.private:
            if i == 0 or ss[i - 1][0] != 'let' or self.root(ss[i - 1][1]) not in self.private:
                self.staged = {}; self.stage(ss, i, out, ind)        # start of a private run
            if i + 1 >= len(ss) or ss[i + 1][0] != 'let' or self.root(ss[i + 1][1]) not in self.private:
                end_run = True
            else: end_run = False
            reads = self.reads(e)
            if self.need_barrier(reads, [], ('private',), axes): self.barrier(out, ind)
            out.append(f'{ind}__syncthreads();')
            tops = self.top_folds(e)
            if len(tops) == 1 and math.prod(self.static[a] for a in axes) <= 4 and self.static[tops[0][2]] >= 256:
                self.blockred(name, axes, e, tops[0], out, ind, scope='block'); unit = 'block'
            else:
                unit = self.simple(name, axes, e, 'block', out, ind)
            out.append(f'{ind}__syncthreads();')
            self.note(reads, [], ('private',)); self.plan.append((name, f'private/{unit}'))
            if end_run: self.staged = {}
            return i + 1
        kind, folds, lam, reg = self.placement(name, axes, e)
        reads = self.reads(e)
        self.tmp += 1; sig = ('u', self.tmp) if kind != 'thread' else ('thread', tuple(axes))
        if self.need_barrier(reads, [r], sig, axes): self.barrier(out, ind)
        if kind == 'thread' or (kind == 'lane' and len(folds) > 1):
            unit = self.simple(name, axes, e, 'grid', out, ind)
            self.note(reads, [r], sig); self.plan.append((name, unit))
            return i + 1
        if kind == 'block':
            self.blockred(name, axes, e, folds[0], out, ind)
            self.note(reads, [r], sig); self.plan.append((name, 'block'))
            return i + 1
        if kind == 'gridred':
            self.gridred(name, axes, e, folds[0], out, ind)
            self.note(reads, [r], sig); self.plan.append((name, 'gridred'))
            return i + 1
        # epilogue: following fold-free statements over the same axes that
        # read this one only at the same point and nothing else unsynced
        epi, j, mine = [], i + 1, {r}
        while j < len(ss) and ss[j][0] == 'let' and id(ss[j]) not in self.groups and id(ss[j]) not in self.in_group:
            t = ss[j]; tr = self.root(t[1])
            if tr in self.private or list(t[2]) != list(axes) or self.top_folds(t[3]): break
            ok = True
            for n, ix in self.reads(t[3]):
                nr = self.root(n)
                if nr in mine:
                    if not (ix == 'same' or ix == [('name', a) for a in self.axes_of[nr]]): ok = False
                elif nr in self.dirty and nr not in self.private: ok = False
            if tr in self.dirty or self.readers.get(tr): ok = False
            if not ok: break
            epi.append((t[1], list(t[2]), t[3])); mine.add(tr); reads += self.reads(t[3]); j += 1
        # emits that copy a slice of this tile's output: written from its epilogue
        emits = []
        while kind == 'tile' and j < len(ss) and ss[j][0] == 'emit':
            m = self.emit_map(ss[j], r, axes, lam)
            if m is None or ss[j][1] in self.dirty or self.readers.get(ss[j][1]): break
            er = [(n, ix) for x in ss[j][2] + [ss[j][3]] for n, ix in self.reads(x)]
            if any(self.root(n) not in mine and self.root(n) in self.dirty for n, _ in er): break
            emits.append(m); mine.add(ss[j][1]); reads += er; j += 1
        if kind == 'tile' and self.step_var:
            tiles = [t for t in ss if t[0] == 'let' and self.root(t[1]) not in self.private and self.tile_ok(t)]
            if tiles and ss[i] is tiles[0] and 'ahead' in PREFETCH: self.pending_pf += [(t, 0) for t in tiles[1:]]
            if tiles and ss[i] is tiles[-1] and 'wrap' in PREFETCH: self.pending_pf += [(tiles[0], 1)]
        self.split(name, axes, e, kind, folds, lam, reg, epi, out, ind, emits)
        self.note(reads, list(mine), sig)
        self.plan.append((name, f'{kind}' + (f' reg={",".join(reg)}' if reg else '')
                          + (f' +{",".join([x[0] for x in epi] + ["emit " + m[0][1] for m in emits])}' if epi or emits else '')))
        return j

    def affine(self, e):
        """(constant, {axis: coefficient}) if e is an integer affine map of axes"""
        k = e[0]
        if k == 'num' and isinstance(e[1], int): return e[1], {}
        if k == 'name' and self.is_axis(e[1]): return 0, {e[1]: 1}
        if k == 'bin' and e[1] in '+-':
            a, b = self.affine(e[2]), self.affine(e[3])
            if a is None or b is None: return None
            sg = 1 if e[1] == '+' else -1
            co = dict(a[1])
            for x, v in b[1].items(): co[x] = co.get(x, 0) + sg * v
            return a[0] + sg * b[0], co
        if k == 'bin' and e[1] == '*':
            a, b = self.affine(e[2]), self.affine(e[3])
            if a is None or b is None: return None
            if not a[1]: return a[0] * b[0], {x: a[0] * v for x, v in b[1].items()}
            if not b[1]: return a[0] * b[0], {x: b[0] * v for x, v in a[1].items()}
        return None

    def emit_map(self, s, X, axes, lam):
        """an emit whose value reads tile output X at [same axes..., c + mixed-radix(other axes)]
        maps one-to-one onto a slice of X's lam axis: (stmt, c, axis order, coefficients, span)"""
        _, name, idx, val = s
        eaxes = []
        for x in idx:
            for n in nodes(x):
                if n[0] == 'name' and self.is_axis(n[1]) and n[1] not in eaxes: eaxes.append(n[1])
        xr = [n for n in nodes(val) if n[0] == 'idx' and self.root(n[1]) == X]
        if len(xr) != 1 or any(n[0] == 'name' and self.root(n[1]) == X for n in nodes(val)): return None
        ix, p = xr[0][2], list(axes).index(lam)
        if len(ix) != len(axes) or any(q != p and ix[q] != ('name', a) for q, a in enumerate(axes)): return None
        aff = self.affine(ix[p])
        if aff is None: return None
        c, co = aff
        fixed = set(axes) - {lam}
        if set(eaxes) != fixed | set(co) or set(co) & fixed or any(v <= 0 for v in co.values()): return None
        order = sorted(co, key=co.get, reverse=True)
        if any(self.ext[a] != f'({self.static[a]})' for a in order) or co[order[-1]] != 1: return None
        if any(co[order[k]] != co[order[k + 1]] * self.static[order[k + 1]] for k in range(len(order) - 1)): return None
        return s, c, order, co, co[order[0]] * self.static[order[0]]

    def gen_group(self, P, run, out, ind):
        reads, writes = [], set()
        names = {self.root(t[1]) for t in run}
        for t in run:
            reads += [(n, ix) for n, ix in self.reads(t[3]) if self.root(n) not in names]
            writes.add(self.root(t[1]))
        self.tmp += 1; sig = ('u', self.tmp)
        if self.need_barrier(reads, list(writes), sig, P): self.barrier(out, ind)
        total = ' * '.join(f'({self.ext[a]})' for a in P)
        g = self.new('g')
        out.append(f'{ind}for (int {g} = blockIdx.x; {g} < {total}; {g} += gridDim.x) {{')
        i2 = ind + '    '
        out.append(f'{i2}int r_{g} = {g};')
        for a in reversed(P):
            out.append(f'{i2}const int ax_{a} = r_{g} % ({self.ext[a]}); r_{g} /= ({self.ext[a]});')
        units = []
        for t in run:
            tn, tax, te = t[1], list(t[2]), t[3]
            pts = tax[len(P):]
            tops = self.top_folds(te)
            npts = math.prod(self.static[a] for a in pts)
            kind, folds, lam, reg = self.placement(tn, tax, te)
            if kind == 'tile' and lam in pts and all(a in pts for a in reg):
                self.split(tn, tax, te, 'tile', folds, lam, reg, [], out, i2, scope='block', bound=P); unit = 'tile'
            elif len(tops) == 1 and npts <= 4 and self.static[tops[0][2]] >= 256:
                self.blockred(tn, tax, te, tops[0], out, i2, scope='block', points=pts); unit = 'block'
            else:
                unit = self.simple(tn, tax, te, 'block', out, i2, points=pts)
            units.append(f'{tn}:{unit}')
            out.append(f'{i2}__syncthreads();')
            self.mark(out, i2, f'    in group: {tn} {unit}')
        out.append(f'{ind}}}')
        self.note(reads, list(writes), sig)
        self.plan.append((','.join(units), f'group[{",".join(P)}]'))

    def gen_step(self, s, out, ind):
        _, name, axes, var, init, carry, blk = s
        c = self.root(carry)
        if not (init[0] == 'name' and self.root(init[1]) == c):
            self.gen_let([('let', carry, axes, init)], 0, out, ind)
        if self.dirty: self.barrier(out, ind)
        ret = [b for b in blk if b[0] == 'ret'][0][1]
        out.append(f'{ind}for (int ax_{var} = 0; ax_{var} < {self.ext[var]}; ++ax_{var}) {{')
        m = self.mult; self.mult *= self.static[var]
        outer, self.step_var = self.step_var, var
        self.gen([b for b in blk if b[0] != 'ret'], out, ind + '    ')
        if not (ret[0] == 'name' and self.root(ret[1]) == c):
            self.gen_let([('let', carry, axes, ret)], 0, out, ind + '    ')
        if self.dirty: self.barrier(out, ind + '    ')
        self.pending_pf = []                         # nothing queued crosses out of the loop
        self.step_var = outer
        self.mult = m
        out.append(f'{ind}}}')

PRELUDE = r'''
#define INFINITY __int_as_float(0x7f800000)
#define NWARP 8
#define BLOCKDIM 256
__device__ __forceinline__ float h2f(unsigned short h) { float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }
__device__ __forceinline__ unsigned short f2h(float f) { unsigned short h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h; }
// element i of a vector load (i is a compile-time constant after unrolling)
__device__ __forceinline__ float hv(uint2 w, int i) { unsigned u = i < 2 ? w.x : w.y; return h2f((unsigned short)((i & 1) ? (u >> 16) : (u & 0xffffu))); }
__device__ __forceinline__ float hv(unsigned int w, int i) { return h2f((unsigned short)((i & 1) ? (w >> 16) : (w & 0xffffu))); }
__device__ __forceinline__ float fv(float4 w, int i) { return i == 0 ? w.x : i == 1 ? w.y : i == 2 ? w.z : w.w; }
__device__ __forceinline__ float fv(float2 w, int i) { return i == 0 ? w.x : w.y; }
// Grid barrier. Safe only under a cooperative launch: every block is resident.
// bar.sync orders the block's writes before thread 0's release-add; the
// acquire spin plus bar.sync makes every other block's writes visible after.
__device__ __forceinline__ void grid_sync(unsigned* bar, unsigned nblk) {
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned g, old, cur;
        asm volatile("ld.relaxed.gpu.u32 %0, [%1];" : "=r"(g) : "l"(bar + 1) : "memory");
        asm volatile("atom.add.acq_rel.gpu.u32 %0, [%1], 1;" : "=r"(old) : "l"(bar) : "memory");
        if (old == nblk - 1) {
            asm volatile("st.relaxed.gpu.u32 [%0], 0;" :: "l"(bar) : "memory");
            asm volatile("red.release.gpu.add.u32 [%0], 1;" :: "l"(bar + 1) : "memory");
        } else {
            do { asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(cur) : "l"(bar + 1) : "memory"); } while (cur == g);
        }
    }
    __syncthreads();
}
'''

def lower(text, res, axis_max, batch=1, per_token=(), outputs=(), feedback=None, timing=False):
    stmts = batch_lift(parse(text), batch, set(per_token))
    L = Lower(stmts, res, axis_max, outputs)
    L.dirty, L.readers, L.tile_rn, L.timing, L.sites = {}, {}, 1, timing, []
    body = []
    L.gen(L.stmts, body, '    ')
    if feedback:
        L.barrier(body, '    ')
        for dst, src in feedback.items():
            if isinstance(src, int):
                body.append(f'    if (gtid == 0) R_{dst}[0] += {src};')
            else:
                r = L.root(src); n = math.prod(L.static[a] for a in L.axes_of[r])
                body.append(f'    if (gtid < {n}) R_{dst}[gtid] = (int)__ldcg(S + {L.g[r]} + gtid);')
    params, args = [], []
    for name, (shape, dt) in res.items():
        em = name in L.emitted or (feedback and name in feedback)
        params.append((name, dt, em))
        args.append(f'{"" if em else "const "}{CTYPE[dt]}* __restrict__ R_{name}')
    args += ['float* __restrict__ S', 'unsigned* bar', 'unsigned long long* TS']
    clock0 = ('    unsigned long long tlast; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tlast));\n'
              if timing else '')
    src = (PRELUDE + 'extern "C" __global__ void __launch_bounds__(256, 4) vesta_step(' + ', '.join(args) + ') {\n'
           + clock0 +
           '    extern __shared__ float sh_f[];\n'
           f'    __shared__ float tsh[NWARP * {L.tile_rn} * 32];\n'
           '    __shared__ bool last_;\n'
           '    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;\n'
           '    const int nthr = gridDim.x * blockDim.x;\n'
           '    const unsigned nblk = gridDim.x;\n'
           '    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n'
           '    const int gwarp = gtid >> 5, nwarp = nthr >> 5;\n'
           + '\n'.join(body) + '\n}\n')
    scratch = {n: (L.g[L.root(n)], L.axes_of[n], tuple(L.static[a] for a in L.axes_of[n]))
               for n in L.axes_of if L.root(n) in L.g}
    return Lowered(src, params, scratch, L.goff, L.sh_floats * 4, L.barriers, L.plan, L.sites, L.counters)
