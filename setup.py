"""Setup configuration for causal discovery toolkit."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = [
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "psutil>=5.0.0"
    ]

setup(
    name="causal_discovery_toolkit",
    version="0.1.0",
    description="Automated causal inference and discovery tools for explainable AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Daniel Schmidt",
    author_email="daniel@terragonlabs.ai",
    url="https://github.com/danieleschmidt/causal-discovery-toolkit",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "full": [
            "torch>=2.0.0",
            "jax>=0.4.0",
            "wandb>=0.15.0",
            "tensorboard>=2.10.0",
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="causal inference, machine learning, explainable ai, causal discovery, DAG",
    project_urls={
        "Bug Reports": "https://github.com/danieleschmidt/causal-discovery-toolkit/issues",
        "Source": "https://github.com/danieleschmidt/causal-discovery-toolkit",
        "Documentation": "https://causal-discovery-toolkit.readthedocs.io/",
    },
    include_package_data=True,
    zip_safe=False,
)
