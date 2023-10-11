# https://stackoverflow.com/questions/19048732/python-setup-py-develop-vs-install/19048754#19048754

from setuptools import setup

setup(
    name="dark",
    description="",
    version="0.1",
    # packages=find_packages(exclude=["test_*"]),
    packages=["dark"],
    package_dir={"dark": "dark"},
    install_requires = ["numpy"]
    )
