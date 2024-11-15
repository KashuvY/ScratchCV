# setup.py
from setuptools import setup, find_packages

setup(
    name="cv_lib",
    version="0.1.0",
    author="Your Name",
    description="A computer vision library built from scratch",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0,<2.0.0',
        'matplotlib>=3.4.0,<4.0.0',
        'opencv-python>=4.5.0',
        'scikit-image>=0.18.0,<0.22.0',
        'Pillow>=8.0.0'
    ],
    python_requires='>=3.8,<3.9',
)