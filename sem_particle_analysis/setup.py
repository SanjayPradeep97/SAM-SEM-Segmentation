"""
Setup script for sem_particle_analysis package
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

setup(
    name='sem_particle_analysis',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='Particle segmentation and analysis for SEM/TEM images using SAM',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/sem-particle-analysis',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Image Processing',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20.0',
        'opencv-python>=4.5.0',
        'torch>=1.10.0',
        'torchvision>=0.11.0',
        'segment-anything @ git+https://github.com/facebookresearch/segment-anything.git',
        'scikit-image>=0.18.0',
        'matplotlib>=3.3.0',
        'pandas>=1.3.0',
        'easyocr>=1.6.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
        ],
        'jupyter': [
            'jupyter>=1.0.0',
            'ipywidgets>=7.6.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'sem-analyze=sem_particle_analysis.cli:main',
        ],
    },
)
