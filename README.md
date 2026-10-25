# Vesta GPU

> Part of my [portfolio](https://github.com/Spendry/portfolio).

Vesta is a small language with four forms: **law**, **fold**, **fork** and **residue**. Its compiler turns a program into **one CUDA kernel per decode step**. Written in Vesta, GPT-2 small decodes at **1,278 tokens per second** on an RTX 5070. That's 1.25 times PyTorch's `torch.compile` with CUDA graphs. Every byte it reads from GPU memory is a weight or a cached key or value, to within 5% of that floor.

## What it does

A decode step is written as 10 questions in plain terms ([QUESTIONS.md](QUESTIONS.md)). They say what is asked, never how the GPU should do it. The grammar is one page ([GRAMMAR.md](GRAMMAR.md)), and GPT-2's step is [one short program](vesta/programs/gpt2.vst).

The compiler places each statement on the GPU by rules, not by hand. Large contractions become tiles split across blocks. Small values are copied into every block. Attention gets its own grouping. The whole step then runs as a single persistent kernel, with no launches between layers and no buffers between operations.

Measured on an RTX 5070, fp16 weights, 256 generated tokens after a 128-token prompt:

| | Batch 1 | Batch 8 |
|---|---|---|
| Vesta | **1,278 tok/s** | **4,483 tok/s** |
| `torch.compile` + CUDA graph | 1,024 tok/s | 2,364 tok/s |
| PyTorch eager, CUDA graph | 899 tok/s | 6,566 tok/s |
| PyTorch eager | 128 tok/s | 962 tok/s |

Vesta's tokens match PyTorch's exactly at both batch sizes. At batch 8, plain CUDA graphs are still faster. That gap is open work.

## How to run it

This starts from a Windows PC with an NVIDIA GPU and nothing else installed. It takes about 30 minutes, mostly downloads.

**1. NVIDIA driver.** Install the newest Game Ready or Studio driver from [nvidia.com/drivers](https://www.nvidia.com/drivers). It must be version 570 or newer.

**2. Python.** Install Python 3.12 or 3.13 from [python.org/downloads](https://www.python.org/downloads/). On the installer's first screen, tick **Add python.exe to PATH**. Use the python.org installer, not the Microsoft Store version.

**3. Git.** Install [Git for Windows](https://git-scm.com/download/win) with its default options. Or skip it, and use **Code → Download ZIP** on this page, then unzip.

**4. Get the code and install it.** Open **PowerShell** and run these one at a time:

```
git clone https://github.com/Spendry/vesta-gpu
cd vesta-gpu
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .[bench]
```

The first `pip` line downloads PyTorch with CUDA, about 3 GB.

**5. Run the benchmark.**

```
python bench/g0_bench.py
```

It measures your GPU's memory bandwidth, then times five ways of running GPT-2, each five times. The first run downloads GPT-2's weights (about 500 MB). It takes 10 to 15 minutes.

**6. Read the result.** Near the end, two lines start with `PASS` or `FAIL`. The first checks that the timings were steady and that Vesta's tokens match PyTorch's. The second is the result, with the speed of each way at batch 1 and batch 8. `PASS` there means Vesta was at least 1.2 times `torch.compile` at batch 1, and at least as fast at batch 8. Everything is also saved in `g0_results.json`.

Tested on Windows 11 with an RTX 5070. Other NVIDIA GPUs from the last few generations should run it, but the numbers will differ.

## How it fits together

| Path | What |
|---|---|
| `vesta/interp.py` | The NumPy interpreter: the reference every GPU result is checked against |
| `vesta/lower.py` | The compiler: a program in, CUDA C for one persistent kernel out |
| `vesta/launch.py` | Compiles that kernel at run time with NVRTC and launches it so every block runs at once |
| `vesta/gpt2.py` | GPT-2 small from Hugging Face weights, run as one kernel per step |
| `vesta/programs/gpt2.vst` | The GPT-2 decode step, written in Vesta |
| `bench/g0_bench.py` | The benchmark above |

No CUDA Toolkit or Visual Studio is needed. The kernel compiler comes from pip.

## Credit

GPT-2 weights by OpenAI, loaded through Hugging Face [transformers](https://github.com/huggingface/transformers).

## License

Code under [MIT](LICENSE). Text (this README, GRAMMAR.md, QUESTIONS.md) under [CC BY 4.0](LICENSE-TEXT).
