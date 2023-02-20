"""DLzeroのセットアップ"""
from setuptools import setup

setup(
    name="DLzero",
    version="0.1.0",
    description="",
    author="",
    packages=["DLzero"],
    install_requires=["numpy", "matplotlib"],
    package_dir={"DL_zero": ["common", "dataset"]},
)
