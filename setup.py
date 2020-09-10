from setuptools import setup, find_packages


setup(
    name="ecole_vs_gasse",
    version="0.1.0",
    author="Antoine Prouvost",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
