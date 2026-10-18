#!/usr/bin/env python
"""
Setup script for Apourvoir package.

Vector Alignment of Knowledge through the Prism of Power.
"""

from setuptools import setup, find_packages
import pathlib

# Read the long description from README
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="apourvoir",
    version="0.1.0",
    description="Vector Alignment of Knowledge through the Prism of Power",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Albo-co/Apourvoir",
    author="Albo-co",
    author_email="",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Operating System :: OS Independent",
    ],
    keywords="knowledge, power, alignment, vector, mathematics, philosophy, ai",
    package_dir={"": "."},
    packages=find_packages(where="."),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.1.0",
        "safetensors>=0.4.0",
        "huggingface_hub>=0.20.0",
        "sentence-transformers>=3.0.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.900",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "notebook>=6.0.0",
        ],
        "advanced": [
            "minisom>=2.1.0",
            "faiss-cpu>=1.7.0",
            "annoy>=1.17.0",
            "plotly>=5.0.0",
            "bokeh>=2.3.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.900",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "notebook>=6.0.0",
            "minisom>=2.1.0",
            "faiss-cpu>=1.7.0",
            "annoy>=1.17.0",
            "plotly>=5.0.0",
            "bokeh>=2.3.0",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/Albo-co/Apourvoir/issues",
        "Source": "https://github.com/Albo-co/Apourvoir",
    },
)
