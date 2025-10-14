# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

from setuptools import setup
import os

# Allow forcing CUDA build even if torch thinks it's not available
_force_cuda = os.environ.get("CUROPE_FORCE_CUDA", "0") == "1"

use_cuda = False
try:
    import torch
    use_cuda = torch.cuda.is_available() or _force_cuda
except Exception:
    use_cuda = False

if use_cuda:
    # Build the real CUDA extension
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    from torch import cuda as _cuda

    # compile for all possible CUDA architectures
    try:
        all_cuda_archs = _cuda.get_gencode_flags().replace('compute=', 'arch=').split()
    except Exception:
        # Fallback: a reasonable generic set; adjust if needed
        all_cuda_archs = [
            '-gencode', 'arch=compute_70,code=sm_70',
            '-gencode', 'arch=compute_75,code=sm_75',
            '-gencode', 'arch=compute_80,code=sm_80',
            '-gencode', 'arch=compute_86,code=sm_86',
        ]

    setup(
        name='curope',
        version='0.1.0',
        ext_modules=[
            CUDAExtension(
                name='curope',
                sources=[
                    "curope.cpp",
                    "kernels.cu",
                ],
                extra_compile_args=dict(
                    nvcc=['-O3', '--ptxas-options=-v', "--use_fast_math"] + all_cuda_archs,
                    cxx=['-O3'],
                ),
            )
        ],
        cmdclass={'build_ext': BuildExtension},
    )
else:
    # CPU-only install: no extension is built. Python will fall back to the slow path.
    setup(
        name='curope',
        version='0.1.0',
        description='CPU-only install; CUDA extension disabled (uses PyTorch fallback).',
        ext_modules=[],
        # No cmdclass needed; nothing to build.
    )

