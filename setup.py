import torch
from setuptools import setup
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import subprocess
import pathlib
import os
import re
import sys
from pathlib import Path

def get_version():
    version = '0.0.5.post1'
    # with open('stepkv/version.py', 'r') as fd:
    #     version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
    #                         fd.read(), re.MULTILINE).group(1)
    if len(sys.argv) >= 2:
        if sys.argv[1] == 'bdist_wheel':
            import torch
            torch_version = torch.__version__.replace("+", "")
            version = f"{version}+torch{torch_version}"
    assert version, 'Cannot find version information'
    print(version)
    return version

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

def filte_cuda_arch_and_code(cuda_dir, arch_list, code_list):
    assert len(arch_list) == len(code_list), "arch_list and code_list should have the same length"
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "--list-gpu-arch"],
                                         universal_newlines=True)
    nvcc_arch_list = raw_output.strip().split('\n')

    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "--list-gpu-code"],
                                         universal_newlines=True)
    nvcc_code_list = raw_output.strip().split('\n')

    i = 0
    while i != len(arch_list):
        if arch_list[i] not in nvcc_arch_list or code_list[i] not in nvcc_code_list:
            del arch_list[i]
            del code_list[i]
        else:
            i += 1


__SRC_PATH__ = 'fserver/csrc/'
__PS_PATH__ = f'{Path.cwd()}'

if __name__ == "__main__":
    cc_flag = []

    torch_cxx11_abi = torch.compiled_with_cxx11_abi()
    use_cuda = os.environ.get("USE_CUDA",'1')=='1'
    extra_link = ['-lrdmacm', '-libverbs']
    extra_compile_args={
            'cxx': [
                '-O3', '-fPIC', 
                f'-I{__PS_PATH__}/include', 
                f'-D_GLIBCXX_USE_CXX11_ABI={str(int(torch_cxx11_abi))}',
                '-DDMLC_USE_ZMQ',
                '-DSTEPMESH_USE_GDR',
                '-DDMLC_USE_RDMA', 
                '-DSTEPMESH_USE_TORCH',
                '-DSTEPMESH_ENABLE_TRACE',
                '-fvisibility=hidden',
                ],
                'nvcc': [],
                }
    if use_cuda:
        extra_link += ['-lcuda', '-lcudart']
        extra_compile_args['cxx'] += ['-DDMLC_USE_CUDA',]
        cuda_arch_list = ['compute_90', 'compute_80', 'compute_89', 'compute_90a']
        cuda_code_list = ['sm_90', 'sm_80', 'sm_89', 'sm_90a']
        filte_cuda_arch_and_code(cpp_extension.CUDA_HOME, cuda_arch_list, cuda_code_list)
        gencode_list = []
        for a,c in zip(cuda_arch_list, cuda_code_list):
            gencode_list.append('-gencode')
            gencode_list.append(f'arch={a},code={c}')
        extra_compile_args['nvcc'] = ['-O3'] + gencode_list + ['--use_fast_math', f'-D_GLIBCXX_USE_CXX11_ABI={str(int(torch_cxx11_abi))}'] + cc_flag
        bare_metal_major, bare_metal_minor = \
            _get_cuda_bare_metal_version(cpp_extension.CUDA_HOME)

    setup(
        name='FServer',
        description='A Remote FFN Server Implementation for AF Disaggregation',
        author='StepFun',
        version=get_version(),
        packages=['fserver'],
        url='',
        ext_modules=[
            CUDAExtension(
                'fserver_lib',
                [
                    __SRC_PATH__ + 'ops.cc',
                    __SRC_PATH__ + 'wait_kernel.cu',
                ],
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link,
                extra_objects=[f"{__PS_PATH__}/cmake_build/libaf.a", f"{__PS_PATH__}/deps/lib/libzmq.a"],
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )
