from setuptools import setup

# read the contents of your README file
from pathlib import Path

this_directory = Path(__file__).parent

setup(
    name="mellon",
    version="1.0.0",
    description="Non-parametric density estimator.",
    url="https://github.com/settylab/mellon",
    author="Setty Lab",
    author_email="msetty@fredhutch.org",
    license="GNU General Public License v3.0",
    packages=["mellon"],
    install_requires=[
        "jax",
        "jaxopt",
        "scikit-learn",
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    long_description=(this_directory / "README.rst").read_text(),
    long_description_content_type="text/x-rst",
)
