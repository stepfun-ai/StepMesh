import torch
from setuptools import setup
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import subprocess
import pathlib
import os
from pathlib import Path

def _get_cuda_bare_metal_version(cuda_dir):
    assert cuda_dir is not None, "Please ensure cuda is installed"
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                         universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return bare_metal_major, bare_metal_minor

__SRC_PATH__ = 'fserver/csrc/'
__PS_PATH__ = f'{Path.cwd()}'

if __name__ == "__main__":
    cc_flag = []
    bare_metal_major, bare_metal_minor = \
        _get_cuda_bare_metal_version(cpp_extension.CUDA_HOME)
    if int(bare_metal_major) >= 11:
        cc_flag.append('-gencode')
        cc_flag.append('arch=compute_80,code=sm_80')
        if int(bare_metal_minor) >= 8 or int(bare_metal_major) >= 12:
            cc_flag.append('-gencode')
            cc_flag.append('arch=compute_90,code=sm_90')
    
    torch_cxx11_abi = torch.compiled_with_cxx11_abi()
    setup(
        name='FServer',
        description='A Remote FFN Server Implementation for AF Disaggregation',
        author='StepFun',
        version='0.0.1.dev',
        packages=['fserver'],
        url='',
        ext_modules=[
            CUDAExtension(
                'fserver_lib',
                [
                    __SRC_PATH__ + 'ops.cc',
                ],
                
                extra_compile_args={
                    'cxx': [
                        '-O3', '-fPIC', 
                        f'-I{__PS_PATH__}/include', 
                        f'-D_GLIBCXX_USE_CXX11_ABI={str(int(torch_cxx11_abi))}',
                        '-DDMLC_USE_ZMQ',
                        '-DDMLC_USE_CUDA', 
                        '-DSTEPAF_USE_GDR',
                        '-DDMLC_USE_RDMA', 
                        '-DSTEPAF_ENABLE_TRACE',
                        '-DSTEPAF_USE_TORCH',
                    ],
                    'nvcc': ['-O3', '-gencode', 'arch=compute_70,code=sm_70', 
                                '--use_fast_math'] + cc_flag,
                },
                extra_link_args=['-lrdmacm', '-libverbs'],
                extra_objects=[f"{__PS_PATH__}/build/libps.a", f"{__PS_PATH__}/deps/lib/libzmq.a"],
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )
