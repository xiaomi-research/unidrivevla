"""Builds the deformable aggregation CUDA op in place: cd unidrivevla/models/ops && python setup.py build_ext --inplace"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="deformable_aggregation_ext",
    ext_modules=[CUDAExtension("deformable_aggregation_ext",
                               ["src/deformable_aggregation.cpp", "src/deformable_aggregation_cuda.cu"])],
    cmdclass={"build_ext": BuildExtension},
)
