"""
Compile a Lowered program with NVRTC and run it as one cooperative launch.
No CUDA Toolkit or host compiler needed: NVRTC and the driver API come from pip
(cuda-bindings, nvidia-cuda-nvrtc-cu12). Shares PyTorch's context and stream.
"""
import ctypes
import torch
from cuda.bindings import driver as cu, nvrtc

TORCH_DT = {'f32': torch.float32, 'f16': torch.float16, 'i32': torch.int32}

def ok(r):
    err, *rest = r if isinstance(r, tuple) else (r,)
    if int(err) != 0: raise RuntimeError(f'{err}')
    return rest[0] if len(rest) == 1 else rest

def compile_cubin(src, arch):
    prog = ok(nvrtc.nvrtcCreateProgram(src.encode(), b'vesta.cu', 0, [], []))
    opts = [f'--gpu-architecture={arch}'.encode(), b'-std=c++17']
    res = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    if int(res[0]) != 0:
        n = ok(nvrtc.nvrtcGetProgramLogSize(prog)); log = b' ' * n
        nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError('NVRTC failed:\n' + log.decode(errors='replace'))
    n = ok(nvrtc.nvrtcGetCUBINSize(prog)); cubin = b' ' * n
    ok(nvrtc.nvrtcGetCUBIN(prog, cubin))
    return cubin

class Kernel:
    def __init__(self, lowered, block=256, device=0, blocks_per_sm=None):
        torch.cuda.init(); torch.zeros(1, device='cuda')           # make torch's context current
        ok(cu.cuInit(0))
        major, minor = torch.cuda.get_device_capability(device)
        self.mod = ok(cu.cuModuleLoadData(compile_cubin(lowered.src, f'sm_{major}{minor}')))
        self.fn = ok(cu.cuModuleGetFunction(self.mod, b'vesta_step'))
        self.smem = lowered.shared_bytes
        if self.smem > 0:            # static + dynamic past 48 KB needs this, even when dynamic alone is small
            ok(cu.cuFuncSetAttribute(self.fn, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                     self.smem))
        per_sm = ok(cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(self.fn, block, self.smem))
        if per_sm < 1: raise RuntimeError('kernel cannot be resident at this block size')
        if blocks_per_sm: per_sm = min(per_sm, blocks_per_sm)
        self.sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.grid, self.block = per_sm * self.sms, block
        self.lowered = lowered
        self.S = torch.zeros(lowered.scratch_floats, dtype=torch.float32, device='cuda')
        self.bar = torch.zeros(lowered.counters, dtype=torch.int32, device='cuda')
        self.TS = torch.zeros(1024, dtype=torch.int64, device='cuda')      # timing sums, if lowered with timing
        self.stream = None

    def view(self, name):
        off, _, shape = self.lowered.scratch[name]
        n = 1
        for s in shape: n *= s
        return self.S[off:off + max(1, n)].view(*shape) if shape else self.S[off]

    def bind(self, residue):
        """fix the argument list once; launches then cost one driver call"""
        vals, types = [], []
        for name, dt, _ in self.lowered.params:
            t = residue[name]
            assert t.is_cuda and t.dtype == TORCH_DT[dt] and t.is_contiguous(), (name, t.dtype, dt)
            vals.append(t.data_ptr()); types.append(ctypes.c_void_p)
        vals += [self.S.data_ptr(), self.bar.data_ptr(), self.TS.data_ptr()]; types += [ctypes.c_void_p] * 3
        self.args = (tuple(vals), tuple(types))
        self.keep = residue

    def launch(self):
        stream = cu.CUstream(torch.cuda.current_stream().cuda_stream)
        ok(cu.cuLaunchCooperativeKernel(self.fn, self.grid, 1, 1, self.block, 1, 1, self.smem, stream, self.args))

    def __call__(self, residue):
        self.bind(residue); self.launch()
