#!/usr/bin/env python3
"""Build the FlashLoop adaptation of KIVI's output-parallel CUDA GEMV."""

from __future__ import annotations

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent

setup(
    name="flashloop-kivi-gemv",
    ext_modules=[
        CUDAExtension(
            name="flashloop_kivi_gemv",
            sources=[
                str(ROOT / "kivi_outer_gemv_binding.cpp"),
                str(ROOT / "kivi_outer_gemv_cuda.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
