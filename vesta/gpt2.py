"""
gpt2.vst on the GPU: HF weights as residue, lowered to one kernel per step.

    m = VestaGPT2(hf, dtype='f16', batch=8, tmax=1024)
    m.reset(); m.x.copy_(tok); m.t.fill_(p); m.step()     # one decode step
    m.logits(), m.next()                                    # views, no sync

With feedback on, each step writes its chosen token into x and advances t,
so generation is back-to-back launches with nothing from the host between.
"""
import torch
from .lower import lower
from .launch import Kernel, TORCH_DT
from . import PROGRAMS

L, H, E, D = 12, 12, 64, 768

def gpt2_weights(hf):
    sd = hf.state_dict()
    st = lambda n: torch.stack([sd[f'transformer.h.{l}.{n}'] for l in range(L)])
    return dict(wte=sd['transformer.wte.weight'], wpe=sd['transformer.wpe.weight'],
                ln1w=st('ln_1.weight'), ln1b=st('ln_1.bias'), qkvw=st('attn.c_attn.weight'), qkvb=st('attn.c_attn.bias'),
                ow=st('attn.c_proj.weight'), ob=st('attn.c_proj.bias'), ln2w=st('ln_2.weight'), ln2b=st('ln_2.bias'),
                fcw=st('mlp.c_fc.weight'), fcb=st('mlp.c_fc.bias'), pw=st('mlp.c_proj.weight'), pb=st('mlp.c_proj.bias'),
                lnfw=sd['transformer.ln_f.weight'], lnfb=sd['transformer.ln_f.bias'])

class VestaGPT2:
    def __init__(self, hf, dtype='f16', batch=1, tmax=1024, feedback=True, program=PROGRAMS / 'gpt2.vst', timing=False, blocks_per_sm=None):
        tdt = TORCH_DT[dtype]
        self.W = {k: v.detach().to('cuda', tdt).contiguous() for k, v in gpt2_weights(hf).items()}
        self.B, self.tmax = batch, tmax
        self.keys = torch.zeros(batch, L, tmax, H, E, device='cuda', dtype=tdt)
        self.vals = torch.zeros_like(self.keys)
        self.x = torch.zeros(batch, dtype=torch.int32, device='cuda')
        self.t = torch.zeros(1, dtype=torch.int32, device='cuda')
        R = {'x': self.x, 't': self.t, **self.W, 'keys': self.keys, 'vals': self.vals}
        spec = {n: ((), 'i32') if n == 't' else (tuple(v.shape), {torch.int32: 'i32', torch.float16: 'f16',
                                                                   torch.float32: 'f32'}[v.dtype]) for n, v in R.items()}
        self.low = lower(open(program, encoding='utf-8').read(), spec, axis_max={'s': tmax}, batch=batch,
                         per_token=('x', 'keys', 'vals'), outputs=('logit', 'next'),
                         feedback={'x': 'next', 't': 1} if feedback else None, timing=timing)
        self.k = Kernel(self.low, blocks_per_sm=blocks_per_sm)
        self.k.bind(R)
        self.weight_bytes = sum(v.numel() * v.element_size() for n, v in self.W.items() if n != 'wpe') \
                            + D * self.W['wpe'].element_size()

    def reset(self):
        self.keys.zero_(); self.vals.zero_(); self.t.zero_()

    def step(self): self.k.launch()
    def logits(self): return self.k.view('logit')
    def next(self): return self.k.view('next')

    def kv_bytes(self, n_ctx):
        return 2 * L * self.B * n_ctx * D * self.keys.element_size()
