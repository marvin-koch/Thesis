# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

from setuptools import setup
from torch import cuda
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# compile for all possible CUDA architectures
all_cuda_archs = cuda.get_gencode_flags().replace('compute=','arch=').split()
# alternatively, you can list cuda archs that you want, eg:
# all_cuda_archs = [
    # '-gencode', 'arch=compute_70,code=sm_70',
    # '-gencode', 'arch=compute_75,code=sm_75',
    # '-gencode', 'arch=compute_80,code=sm_80',
# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

from setuptools import setup
from torch import cuda
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os, re, subprocess

def nvcc_version():
    try:
        out = subprocess.check_output(['nvcc', '--version'], stderr=subprocess.STDOUT).decode()
        m = re.search(r'release\s+(\d+)\.(\d+)', out)
        if m:
            return int(m.group(1)), int(m.group(2))
    except Exception:
        pass
    return None

def filter_arch_pairs(pairs):
    """
    pairs = ['-gencode','arch=compute_86,code=sm_86', '-gencode','arch=compute_100,code=sm_100', ...]
    Keep only those NVCC supports. CUDA < 12.8 doesn't support sm_100+.
    """
    ver = nvcc_version()
    def supported(sm):
        if ver is None:
            return sm < 100
        major, minor = ver
        if major < 12:      # very old
            return sm <= 90
        if major == 12 and minor < 8:
            return sm <= 90
        return True

    filtered = []
    for i in range(0, len(pairs), 2):
        a, b = pairs[i], pairs[i+1]
        m = re.search(r'code=sm_(\d+)', b)
        sm = int(m.group(1)) if m else 0
        if supported(sm):
            filtered += [a, b]
    return filtered

# Respect user-provided TORCH_CUDA_ARCH_LIST if set; otherwise use PyTorch’s defaults.
arch_pairs = cuda.get_gencode_flags().replace('compute=', 'arch=').split()
# Ensure we have an even number (pairs)
if len(arch_pairs) % 2 == 0:
    arch_pairs = filter_arch_pairs(arch_pairs)

all_cuda_archs = arch_pairs

setup(
    name='curope',
    ext_modules=[
        CUDAExtension(
            name='curope',
            sources=[
                "curope.cpp",
                "kernels.cu",
            ],
            extra_compile_args=dict(
                nvcc=['-O3', '--ptxas-options=-v', '--use_fast_math'] + all_cuda_archs,
                cxx=['-O3'],
            ),
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
    # '-gencode', 'arch=compute_86,code=sm_86'
# ]

setup(
    name = 'curope',
    ext_modules = [
        CUDAExtension(
                name='curope',
                sources=[
                    "curope.cpp",
                    "kernels.cu",
                ],
                extra_compile_args = dict(
                    nvcc=['-O3','--ptxas-options=-v',"--use_fast_math"]+all_cuda_archs, 
                    cxx=['-O3'])
                )
    ],
    cmdclass = {
        'build_ext': BuildExtension
    })
