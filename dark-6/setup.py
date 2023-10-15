from setuptools import setup

setup(
    name="dark",
    description="",
    version="0.1",
    # packages=find_packages(exclude=["test_*"]),
    packages=["dark"],
    package_dir={"dark": "dark"},
    install_requires = ["numpy", "opencv-python", "numba"]
    )
