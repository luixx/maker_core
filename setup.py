"""Setup configuration for maker-core."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="maker-core",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Python implementation of the MAKER algorithm for reliable LLM responses",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/maker-core",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.3.0",
            "pytest-asyncio>=0.24.0",
            "pytest-cov>=6.0.0",
            "black>=24.10.0",
            "mypy>=1.13.0",
            "ruff>=0.7.4",
        ],
    },
    keywords="llm ai maker reliability consensus voting decomposition",
)
