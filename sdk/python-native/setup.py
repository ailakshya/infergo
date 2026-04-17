"""Setup script for infergo-native Python SDK."""

from setuptools import setup, find_packages

setup(
    name="infergo-native",
    version="0.1.0",
    description="Python ctypes bindings for the infergo C inference engine",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Lakshya",
    url="https://github.com/nicewook/infergo",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: POSIX :: Linux",
    ],
    keywords="inference llm embedding vectordb ctypes native",
)
