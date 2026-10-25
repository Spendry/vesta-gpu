"""
Vesta NumPy interpreter (V2). Runs programs written in GRAMMAR.md.

    run(text, residue) -> env
      text    : program source
      residue : dict name -> ndarray or Python scalar. Emits write into it.
      env     : dict name -> (array, axes) of every named value at the end

Four forms only. law = ('bin'|'neg'|'call'|'idx'), fold = 'fold'/step block,
fork = 'fork', residue = receive/emit statements and residue reads.
"""
import re
import numpy as np

# ------------------------------------------------------------------ lexer
TOK = re.compile(r'\s*(?:(\d+\.\d*(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+|\d+)|([A-Za-z_]\w*)|(<-|->|==|<=|>=|[\[\](),:+\-*/?<>=]))')
FOLD_OPS = {'sum': np.sum, 'max': np.max, 'min': np.min}
FUNCS = {'exp': np.exp, 'tanh': np.tanh, 'sqrt': np.sqrt}
CMP = {'==': np.equal, '<': np.less, '>': np.greater, '<=': np.less_equal, '>=': np.greater_equal}
ARITH = {'+': np.add, '-': np.subtract, '*': np.multiply, '/': np.divide}

def lex(s):
    out, i, s = [], 0, s.rstrip()
    while i < len(s):
        m = TOK.match(s, i)
        if not m or m.end() == i: raise SyntaxError(f'bad token at {s[i:]!r}')
        num, name, op = m.groups()
        out.append(('num', float(num) if any(c in num for c in '.eE') else int(num)) if num
                   else ('name', name) if name else ('op', op))
        i = m.end()
    return out

# ------------------------------------------------------------------ parser
class P:
    def __init__(self, toks): self.t, self.i = toks, 0
    def peek(self, k=0): return self.t[self.i + k] if self.i + k < len(self.t) else (None, None)
    def take(self, val=None):
        tok = self.peek()
        if val is not None and tok[1] != val: raise SyntaxError(f'expected {val!r}, got {tok[1]!r}')
        self.i += 1; return tok
    def done(self): return self.i >= len(self.t)

    def expr(self):
        if self.peek()[1] == 'fork':
            self.take(); c = self.cmp(); self.take('?'); a = self.expr(); self.take(':')
            return ('fork', c, a, self.expr())
        if self.peek()[1] == 'fold' and self.peek(1)[1] in FOLD_OPS:
            self.take(); op = self.take()[1]; ax = self.take()[1]; self.take(':')
            return ('fold', op, ax, self.expr())
        return self.cmp()
    def cmp(self):
        a = self.add()
        if self.peek()[1] in CMP: op = self.take()[1]; return ('bin', op, a, self.add())
        return a
    def add(self):
        a = self.mul()
        while self.peek()[1] in ('+', '-'): op = self.take()[1]; a = ('bin', op, a, self.mul())
        return a
    def mul(self):
        a = self.un()
        while self.peek()[1] in ('*', '/'): op = self.take()[1]; a = ('bin', op, a, self.un())
        return a
    def un(self):
        if self.peek()[1] == '-': self.take(); return ('neg', self.un())
        return self.post()
    def post(self):
        a = self.atom()
        if self.peek()[1] == '[':
            self.take(); idx = [self.expr()]
            while self.peek()[1] == ',': self.take(); idx.append(self.expr())
            self.take(']'); a = ('idx', a[1], idx)
        return a
    def atom(self):
        kind, v = self.take()
        if kind == 'num': return ('num', v)
        if v == '(':
            e = self.expr(); self.take(')'); return e
        if kind == 'name' and v in FUNCS and self.peek()[1] == '(':
            self.take('('); e = self.expr(); self.take(')'); return ('call', v, e)
        if kind == 'name': return ('name', v)
        raise SyntaxError(f'unexpected {v!r}')
    def names(self):
        out = [self.take()[1]]
        while self.peek()[1] == ',': self.take(); out.append(self.take()[1])
        return out

def parse_expr(toks):
    p = P(toks); e = p.expr()
    if not p.done(): raise SyntaxError(f'trailing {p.peek()[1]!r}')
    return e

def parse(text):
    """-> list of statements. Blocks nest via a stack."""
    root = []; stack = [root]
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.split('#', 1)[0].strip()
        if not line: continue
        try:
            toks = lex(line); p = P(toks); body = stack[-1]
            if line == 'end':
                stack.pop(); continue
            if toks[0][1] == '->':
                body.append(('ret', parse_expr(toks[1:]))); continue
            if toks[0][1] == 'axis':
                p.take(); name = p.take()[1]; p.take('=')
                body.append(('axis', name, parse_expr(toks[p.i:]))); continue
            if toks[0][1] == 'residue':
                p.take(); name = p.take()[1]
                if p.done(): body.append(('recv', name, [])); continue
                p.take('['); arrow = [t[1] for t in toks].index('<-') if ('op', '<-') in toks else None
                if arrow is None:
                    body.append(('recv', name, p.names())); continue
                idx = P(toks[p.i:arrow - 1]); ix = [idx.expr()]
                while not idx.done(): idx.take(','); ix.append(idx.expr())
                body.append(('emit', name, ix, parse_expr(toks[arrow + 1:]))); continue
            name = p.take()[1]; axes = []
            if p.peek()[1] == '[': p.take(); axes = p.names(); p.take(']')
            p.take('=')
            if p.peek()[1] == 'fold' and p.peek(1)[1] == 'step':
                p.take(); p.take(); var = p.take()[1]; p.take(':')
                rest = toks[p.i:]; k = max(i for i, t in enumerate(rest) if t[1] == 'as')
                blk = []
                body.append(('step', name, axes, var, parse_expr(rest[:k]), rest[k + 1][1], blk))
                stack.append(blk); continue
            body.append(('let', name, axes, parse_expr(toks[p.i:])))
        except SyntaxError as e:
            raise SyntaxError(f'line {n}: {raw.strip()!r}: {e}') from None
    if len(stack) != 1: raise SyntaxError('missing end')
    return root

# ------------------------------------------------------------------ evaluation
def union(*axs):
    out = []
    for ax in axs:
        for a in ax:
            if a not in out: out.append(a)
    return tuple(out)

def align(v, ax, target):
    """lay v (axes ax) out on target's axis order, size 1 where absent"""
    if not ax: return v
    v = np.asarray(v).transpose([ax.index(a) for a in target if a in ax])
    shape, k = [], 0
    for a in target:
        if a in ax: shape.append(v.shape[k]); k += 1
        else: shape.append(1)
    return v.reshape(shape)

class Interp:
    def __init__(self, residue):
        self.res, self.env, self.ext = residue, {}, {}

    def ev(self, e):
        k = e[0]
        if k == 'num': return e[1], ()
        if k == 'name':
            n = e[1]
            if n in self.env: return self.env[n]
            if n in self.ext: return np.arange(self.ext[n]), (n,)
            if n in self.res and np.ndim(self.res[n]) == 0: return self.res[n], ()
            raise NameError(n)
        if k == 'neg':
            v, ax = self.ev(e[1]); return -v, ax
        if k == 'call':
            v, ax = self.ev(e[2]); return FUNCS[e[1]](v), ax
        if k == 'bin':
            (a, aa), (b, ba) = self.ev(e[2]), self.ev(e[3]); u = union(aa, ba)
            f = ARITH.get(e[1]) or CMP[e[1]]
            return f(align(a, aa, u), align(b, ba, u)), u
        if k == 'fork':
            (c, ca), (a, aa), (b, ba) = self.ev(e[1]), self.ev(e[2]), self.ev(e[3])
            u = union(ca, aa, ba)
            return np.where(align(c, ca, u), align(a, aa, u), align(b, ba, u)), u
        if k == 'idx':
            name, idx = e[1], [self.ev(i) for i in e[2]]
            base = self.env[name][0] if name in self.env else self.res[name]
            u = union(*[ax for _, ax in idx])
            shape = tuple(self.ext[a] if a in self.ext else 1 for a in u)
            ix = tuple(np.broadcast_to(align(v, ax, u), shape) if u else v for v, ax in idx)
            return base[ix], u
        if k == 'fold':
            op, axis, body = e[1], e[2], e[3]
            if op == 'sum' and body[0] == 'bin' and body[1] == '*':
                return self.contract(axis, body[2], body[3])
            v, ax = self.ev(body)
            if axis not in ax: raise ValueError(f'fold over {axis} but body has axes {ax}')
            return FOLD_OPS[op](v, axis=ax.index(axis)), tuple(a for a in ax if a != axis)
        raise ValueError(k)

    def contract(self, axis, ea, eb):
        """fold sum of a product, as one contraction (same value, fewer temporaries)"""
        (a, aa), (b, ba) = self.ev(ea), self.ev(eb)
        u = union(aa, ba)
        if axis not in u: raise ValueError(f'fold over {axis} but body has axes {u}')
        L = {n: chr(97 + i) for i, n in enumerate(u)}
        out = tuple(n for n in u if n != axis)
        spec = f"{''.join(L[n] for n in aa)},{''.join(L[n] for n in ba)}->{''.join(L[n] for n in out)}"
        return np.einsum(spec, np.asarray(a), np.asarray(b), optimize=True), out

    def fill(self, v, ax, lhs):
        if not set(ax) <= set(lhs): raise ValueError(f'value has axes {ax}, left side only {lhs}')
        if not lhs: return v
        return np.array(np.broadcast_to(align(v, ax, lhs), tuple(self.ext[a] for a in lhs)))

    def exec(self, stmts):
        for s in stmts:
            k = s[0]
            if k == 'axis':
                self.ext[s[1]] = int(self.ev(s[2])[0])
            elif k == 'recv':
                if s[1] not in self.res: raise NameError(f'residue {s[1]} not supplied')
            elif k == 'emit':
                idx = [self.ev(i) for i in s[2]]; v, ax = self.ev(s[3])
                u = union(*[a for _, a in idx])
                if not set(ax) <= set(u): raise ValueError(f'emit value has axes {ax}, index only {u}')
                shape = tuple(self.ext[a] for a in u)
                ix = tuple(np.broadcast_to(align(i, a, u), shape) if u else i for i, a in idx)
                self.res[s[1]][ix] = np.broadcast_to(align(v, ax, u), shape) if u else v
            elif k == 'let':
                self.env[s[1]] = (self.fill(*self.ev(s[3]), s[2]), tuple(s[2]))
            elif k == 'step':
                _, name, lhs, var, init, carry, blk = s
                c = (self.fill(*self.ev(init), lhs), tuple(lhs))
                ret = [b for b in blk if b[0] == 'ret']
                if len(ret) != 1: raise SyntaxError(f'step {name} needs exactly one ->')
                for i in range(self.ext[var]):
                    self.env[var] = (i, ()); self.env[carry] = c
                    self.exec([b for b in blk if b[0] != 'ret'])
                    c = (self.fill(*self.ev(ret[0][1]), lhs), tuple(lhs))
                del self.env[var]
                self.env[name] = c

def run(text, residue):
    it = Interp(residue); it.exec(parse(text)); return it.env
