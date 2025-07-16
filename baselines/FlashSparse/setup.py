from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


setup(
    name='FlashSparse_kernel',
    # ext_modules=[module],
    ext_modules=[
       CUDAExtension(
            name='FS_SpMM', 
            sources=[
            './SpMM/src/benchmark.cpp',
            './SpMM/src/spmmKernel.cu',
            ]
         ),
       CUDAExtension(
            name='FS_SDDMM', 
            sources=[
            './SDDMM/src/benchmark.cpp',
            './SDDMM/src/sddmmKernel.cu',
            ]
         ),
        CUDAExtension(
            name='FS_Block', 
            sources=[
            './Block/example.cpp'
            ]
         ) ,
       CUDAExtension(
            name='FS_Block_gpu', 
            sources=[
            './Block_gpu/block.cpp',
            './Block_gpu/block_kernel.cu',
            ]
         ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


