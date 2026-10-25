"""G0 -- the baseline instrument. Run this on the RTX 5070.

    python g0_bench.py                # full run on CUDA
    python g0_bench.py --device cpu --gen 8 --prompt 16   # smoke test anywhere

What it measures
  1. Peak memory bandwidth of this GPU (copy and read-only), measured, not quoted.
  2. GPT-2 small greedy decode, fp16, from a minimal reference model with a
     static KV cache (every byte it reads is known), in up to four arms:
       eager-slice  : PyTorch eager, attends to cache[:pos+1]
       eager-mask   : PyTorch eager, full static cache + mask (graph-safe shape)
       cudagraph    : eager-mask captured once as a CUDA graph, replayed
       compile+graph: torch.compile(max-autotune-no-cudagraphs) then captured
  3. Roofline: bytes the step must read (weights + KV) / measured peak BW,
     against measured step time. That ratio is the number every later gate moves.

Gate G0 (the instrument must be trustworthy before anything is compared to it):
  FAILS IF the reference model's greedy tokens differ from Hugging Face's on
  the same prompt, or if any timed arm has CV > 5% across repeats.

Requirements: PyTorch built with CUDA 12.8+ (Blackwell sm_120), transformers.
  pip install torch --index-url https://download.pytorch.org/whl/cu128
  pip install transformers
  compile arm on Windows also needs:  pip install triton-windows
"""
import argparse, json, math, os, platform, statistics, time, traceback

# triton-windows looks for its bundled TinyCC and CUDA under sysconfig's platlib.
# A user-site install (e.g. Microsoft Store Python) misses both and falls back to
# whatever gcc is on PATH. Point it at the bundled copies explicitly.
if platform.system() == 'Windows':
    import importlib.util
    _spec = importlib.util.find_spec('triton')
    if _spec is not None:
        _t = os.path.dirname(_spec.origin)
        _tcc = os.path.join(_t, 'runtime', 'tcc', 'tcc.exe')
        _cuda = os.path.join(_t, 'backends', 'nvidia')
        if os.path.exists(_tcc): os.environ.setdefault('CC', _tcc)
        if os.path.exists(os.path.join(_cuda, 'bin', 'ptxas.exe')): os.environ.setdefault('CUDA_PATH', _cuda)

import torch
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument('--device', default='cuda')
p.add_argument('--prompt', type=int, default=128)
p.add_argument('--gen', type=int, default=256)
p.add_argument('--batch', type=int, nargs='+', default=[1, 8])
p.add_argument('--repeats', type=int, default=5)
p.add_argument('--out', default='g0_results.json')
p.add_argument('--arms', nargs='+', default=None, help='run only these arms')
args = p.parse_args()
dev = torch.device(args.device)
DT = torch.float16 if dev.type == 'cuda' else torch.float32

def sync():
    if dev.type == 'cuda': torch.cuda.synchronize()

# ------------------------------------------------------------------ 1. bandwidth
def bandwidth():
    if dev.type != 'cuda': return {}
    n = 256 * 1024 * 1024 // 2                       # 512 MB of fp16 per buffer
    a = torch.empty(n, dtype=torch.float16, device=dev).normal_()
    b = torch.empty_like(a)
    out = {}
    for name, fn, mult in [('copy_GBs', lambda: b.copy_(a), 2),     # read + write
                           ('read_GBs', lambda: a.sum(), 1)]:       # read only
        for _ in range(3): fn()
        sync(); ts = []
        for _ in range(20):
            s = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - s)
        out[name] = round(mult * a.numel() * 2 / min(ts) / 1e9, 1)
    del a, b; torch.cuda.empty_cache()
    out['peak_GBs'] = max(out.values())
    return out

# ------------------------------------------------------------------ 2. model
class GPT2Ref:
    """GPT-2 small written out so every read is visible. Weights from HF."""
    def __init__(self, hf, dtype, device, tmax, batch):
        sd = {k: v.to(device=device, dtype=dtype) for k, v in hf.state_dict().items()}
        self.wte, self.wpe = sd['transformer.wte.weight'], sd['transformer.wpe.weight']
        self.L = hf.config.n_layer; self.H = hf.config.n_head; self.D = hf.config.n_embd
        self.hd = self.D // self.H; self.eps = hf.config.layer_norm_epsilon
        g = lambda l, n: sd[f'transformer.h.{l}.{n}']
        self.layers = [dict(ln1w=g(l, 'ln_1.weight'), ln1b=g(l, 'ln_1.bias'),
                            qkvw=g(l, 'attn.c_attn.weight'), qkvb=g(l, 'attn.c_attn.bias'),
                            ow=g(l, 'attn.c_proj.weight'), ob=g(l, 'attn.c_proj.bias'),
                            ln2w=g(l, 'ln_2.weight'), ln2b=g(l, 'ln_2.bias'),
                            fcw=g(l, 'mlp.c_fc.weight'), fcb=g(l, 'mlp.c_fc.bias'),
                            pw=g(l, 'mlp.c_proj.weight'), pb=g(l, 'mlp.c_proj.bias'))
                       for l in range(self.L)]
        self.lnfw, self.lnfb = sd['transformer.ln_f.weight'], sd['transformer.ln_f.bias']
        self.tmax, self.B = tmax, batch
        shape = (self.L, batch, self.H, tmax, self.hd)
        self.K = torch.zeros(shape, dtype=dtype, device=device)
        self.V = torch.zeros(shape, dtype=dtype, device=device)
        self.ar = torch.arange(tmax, device=device)
        # bytes one decode step must read, excluding the KV cache
        w = sum(t.numel() for lay in self.layers for t in lay.values())
        w += self.lnfw.numel() + self.lnfb.numel() + self.wte.numel() + self.D  # lm_head + one wpe row
        self.weight_bytes = w * self.wte.element_size()

    def step(self, tok, pos, masked):
        """tok: (B,) long. pos: 0-dim long tensor. masked: full-cache attention."""
        x = self.wte[tok] + self.wpe.index_select(0, pos.view(1))         # (B, D), no host sync
        for l, P in enumerate(self.layers):
            h = F.layer_norm(x, (self.D,), P['ln1w'], P['ln1b'], self.eps)
            q, k, v = (h @ P['qkvw'] + P['qkvb']).split(self.D, dim=-1)
            q, k, v = (t.view(self.B, self.H, 1, self.hd) for t in (q, k, v))
            self.K[l].index_copy_(2, pos.view(1), k)
            self.V[l].index_copy_(2, pos.view(1), v)
            if masked:
                m = (self.ar <= pos).view(1, 1, 1, self.tmax)
                a = F.scaled_dot_product_attention(q, self.K[l], self.V[l], attn_mask=m)
            else:
                n = int(pos) + 1
                a = F.scaled_dot_product_attention(q, self.K[l][:, :, :n], self.V[l][:, :, :n])
            x = x + a.reshape(self.B, self.D) @ P['ow'] + P['ob']
            h = F.layer_norm(x, (self.D,), P['ln2w'], P['ln2b'], self.eps)
            h = F.gelu(h @ P['fcw'] + P['fcb'], approximate='tanh')
            x = x + h @ P['pw'] + P['pb']
        x = F.layer_norm(x, (self.D,), self.lnfw, self.lnfb, self.eps)
        return x @ self.wte.T                                              # (B, vocab)

    def kv_bytes(self, n_ctx):
        return 2 * self.L * self.B * self.H * n_ctx * self.hd * self.K.element_size()

# ------------------------------------------------------------------ 3. arms
def run_arm(name, m, prompt, gen, repeats):
    """returns tokens of repeat 0 and per-repeat seconds for the gen phase"""
    pos_t = torch.zeros((), dtype=torch.long, device=dev)
    tok_t = torch.zeros(m.B, dtype=torch.long, device=dev)
    masked = name != 'eager-slice'
    fn = lambda: m.step(tok_t, pos_t, masked)
    graph = None
    if name in ('cudagraph', 'compile+graph'):
        if name == 'compile+graph':
            fn = torch.compile(fn, mode='max-autotune-no-cudagraphs', fullgraph=True)
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3): out_static = fn()
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): out_static = fn()

    def one(i):
        if graph is not None:
            graph.replay(); return out_static
        return fn()

    times, toks0 = [], None
    for r in range(repeats):
        m.K.zero_(); m.V.zero_()
        for i in range(prompt.shape[1]):                         # prefill, untimed
            tok_t.copy_(prompt[:, i]); pos_t.fill_(i); logits = one(i)
        nxt = logits.argmax(-1); toks = [nxt]          # first token comes from prefill
        sync(); t0 = time.perf_counter()
        for j in range(gen):
            tok_t.copy_(nxt); pos_t.fill_(prompt.shape[1] + j); logits = one(j)
            nxt = logits.argmax(-1); toks.append(nxt)
        sync(); times.append(time.perf_counter() - t0)
        if r == 0: toks0 = torch.stack(toks[:gen], 1).cpu()
    return toks0, times

def run_vesta(v, prompt, gen, repeats):
    """same protocol as run_arm. The step feeds its own next token (x) and
    advances t, so the timed loop is back-to-back launches plus a token copy."""
    times, toks0 = [], None
    for r in range(repeats):
        v.reset()
        for i in range(prompt.shape[1]):                         # prefill, untimed
            v.x.copy_(prompt[:, i]); v.step()
        toks = [v.x.clone()]                                     # first token comes from prefill
        sync(); t0 = time.perf_counter()
        for j in range(gen):
            v.step(); toks.append(v.x.clone())
        sync(); times.append(time.perf_counter() - t0)
        if r == 0: toks0 = torch.stack(toks[:gen], 1).long().cpu()
    return toks0, times

def main():
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    res = dict(host=platform.platform(), torch=torch.__version__,
               cuda=torch.version.cuda, args=vars(args))
    if dev.type == 'cuda':
        res['gpu'] = torch.cuda.get_device_name(0)
        res['capability'] = '.'.join(map(str, torch.cuda.get_device_capability(0)))
    try:
        import triton; res['triton'] = triton.__version__
    except Exception:
        res['triton'] = None
    res['bandwidth'] = bandwidth(); print('bandwidth', res['bandwidth'])
    peak = res['bandwidth'].get('peak_GBs')

    tok = GPT2TokenizerFast.from_pretrained('gpt2')
    hf = GPT2LMHeadModel.from_pretrained('gpt2').eval()
    text = ("The history of computing is a history of moving data. Every machine since "
            "1945 has spent most of its effort fetching numbers from memory, ") * 20
    base_ids = tok(text, return_tensors='pt').input_ids[:, :args.prompt]

    # ---- correctness: HF greedy vs reference greedy, fp32 on CPU, batch 1
    n_chk = min(32, args.gen)
    with torch.no_grad():
        hf_out = hf.generate(base_ids, max_new_tokens=n_chk, do_sample=False,
                             pad_token_id=tok.eos_token_id)[0, args.prompt:]
        ref = GPT2Ref(hf, torch.float32, torch.device('cpu'), args.prompt + n_chk, 1)
        save = dev; globals()['dev'] = torch.device('cpu')
        ref_toks, _ = run_arm('eager-slice', ref, base_ids, n_chk, 1)
        globals()['dev'] = save
    res['correct_vs_hf'] = bool(torch.equal(hf_out, ref_toks[0]))
    print('reference matches HF greedy:', res['correct_vs_hf'])

    arms = ['eager-slice', 'eager-mask'] + (['cudagraph', 'compile+graph', 'vesta'] if dev.type == 'cuda' else [])
    if args.arms: arms = [a for a in arms if a in args.arms]
    res['runs'] = []
    for B in args.batch:
        prompt = base_ids.repeat(B, 1).to(dev)
        tmax = args.prompt + args.gen
        m = GPT2Ref(hf, DT, dev, tmax, B)
        ref_tokens = None
        for arm in arms:
            row = dict(batch=B, arm=arm)
            try:
                with torch.no_grad():
                    if arm == 'vesta':
                        from vesta.gpt2 import VestaGPT2
                        v = VestaGPT2(hf, dtype='f16' if DT == torch.float16 else 'f32', batch=B, tmax=tmax)
                        row['barriers_per_step'] = v.low.barriers_per_launch
                        toks, ts = run_vesta(v, prompt, args.gen, args.repeats)
                        del v
                    else:
                        toks, ts = run_arm(arm, m, prompt, args.gen, args.repeats)
                med = statistics.median(ts)
                row['tok_per_s'] = round(B * args.gen / med, 1)
                row['ms_per_step'] = round(1e3 * med / args.gen, 4)
                row['cv'] = round(statistics.pstdev(ts) / statistics.mean(ts), 4)
                ctx = args.prompt + args.gen / 2                      # mean context
                kv = m.kv_bytes(ctx if arm in ('eager-slice', 'vesta') else tmax)
                row['bytes_per_step_MB'] = round((m.weight_bytes + kv) / 1e6, 2)
                if peak:
                    floor = (m.weight_bytes + kv) / (peak * 1e9)
                    row['roofline_fraction'] = round(floor / (med / args.gen), 3)
                if ref_tokens is None: ref_tokens = toks
                row['tokens_match_first_arm'] = bool(torch.equal(toks, ref_tokens))
            except Exception as e:
                row['error'] = f'{type(e).__name__}: {e}'.splitlines()[0][:300]
            print(row); res['runs'].append(row)
        del m
        if dev.type == 'cuda': torch.cuda.empty_cache()

    timed = [r for r in res['runs'] if 'cv' in r]
    bad_cv = [r for r in timed if r['cv'] > 0.05]
    res['GATE_G0'] = 'PASS' if res['correct_vs_hf'] and not bad_cv and timed else 'FAIL'
    res['GATE_G0_detail'] = (f"correct={res['correct_vs_hf']}, arms timed={len(timed)}, "
                             f"arms with CV>5%={len(bad_cv)}")
    print(res['GATE_G0'], res['GATE_G0_detail'])

    # ---- G4: vesta against compile+graph (and cudagraph, for the record)
    tps = {(r['batch'], r['arm']): r.get('tok_per_s') for r in res['runs']}
    g4 = {}
    for B, bar in ((1, 1.2), (8, 1.0)):
        v, cg, gr = tps.get((B, 'vesta')), tps.get((B, 'compile+graph')), tps.get((B, 'cudagraph'))
        if v and cg:
            g4[B] = dict(vesta=v, compile_graph=cg, cudagraph=gr, x_compile_graph=round(v / cg, 3),
                         x_cudagraph=round(v / gr, 3) if gr else None, bar=bar, pass_=v / cg >= bar)
    if g4:
        res['GATE_G4'] = 'PASS' if len(g4) == 2 and all(x['pass_'] for x in g4.values()) else 'FAIL'
        res['GATE_G4_detail'] = g4
        print(res['GATE_G4'], g4)
    json.dump(res, open(args.out, 'w'), indent=1)
    print('wrote', args.out)

if __name__ == '__main__':
    main()
