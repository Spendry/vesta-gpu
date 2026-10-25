"""Vesta: a four-form language (law, fold, fork, residue) and a compiler that
lowers a program to one persistent CUDA kernel per step.

    vesta.interp   the NumPy interpreter (the reference)
    vesta.lower    program -> CUDA C for one persistent kernel
    vesta.launch   NVRTC compile and cooperative launch
    vesta.gpt2     GPT-2 small, from Hugging Face weights, on the GPU
"""
from pathlib import Path

__version__ = '0.1.0'
PROGRAMS = Path(__file__).resolve().parent / 'programs'
