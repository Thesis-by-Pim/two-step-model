"""
This file defines to `setuptools`, the default Python packaging tool, how to package our
project and to register an entry point `cli` that runs our code.

See: https://setuptools.readthedocs.io/en/latest/
"""
from setuptools import setup  # type:ignore[import]

setup(
    name="linear_programming_setup",
    packages=["linear_programming_setup"],
    description="Barebones linear programming setup",
    install_requires=["click", "mip", "mypy"],
    include_package_data=True,
    entry_points={"console_scripts": ["cli=linear_programming_setup.cli:cli"]},
    zip_safe=False,
)

