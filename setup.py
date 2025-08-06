from setuptools import setup, find_packages

setup(
    name="causal_discovery_toolkit",
    version="0.1.0",
    description="Automated causal inference and discovery tools for explainable AI",
    author="Daniel Schmidt",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Core dependencies will be added based on research needs
    ],
    python_requires=">=3.8",
)
