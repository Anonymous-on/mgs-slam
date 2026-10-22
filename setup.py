"""Build MASt3R-SLAM's native CUDA backend in this integrated source tree."""
from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).resolve().parent
setup(
    name='mgs-slam',
    version='0.1.0',
    packages=find_packages(include=('mgs_slam', 'mgs_slam.*')),
    ext_modules=[CUDAExtension(
        'mast3r_slam_backends',
        sources=[
            str(root / 'third_party/mast3r_slam/backend/src/gn.cpp'),
            str(root / 'third_party/mast3r_slam/backend/src/gn_kernels.cu'),
            str(root / 'third_party/mast3r_slam/backend/src/matching_kernels.cu'),
        ],
        include_dirs=[
            str(root / 'third_party/mast3r_slam/backend/include'),
            str(root / 'third_party/eigen'),
        ],
        extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']},
    )],
    cmdclass={'build_ext': BuildExtension},
)
