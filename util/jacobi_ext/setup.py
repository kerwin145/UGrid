from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="jacobi_ext",
    ext_modules=[
        CUDAExtension(
            name="jacobi_ext",
            sources=[
                "jacobi_ext.cpp",
                "jacobi_ext_cuda.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
