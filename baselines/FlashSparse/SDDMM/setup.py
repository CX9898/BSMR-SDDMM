from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mGAT',
    ext_modules=[
       CUDAExtension(
            name='mGAT', 
            sources=[
            'mGATKernel.cu',
            'mGAT.cpp'
            ]
            # extra_compile_args=['-O3', '-fopenmp', '-mcx16'],
            # libraries=["numa", "tcmalloc_minimal"]
         ) 
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


