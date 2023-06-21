from setuptools import setup

# read the contents of your README file
from pathlib import Path

this_directory = Path(__file__).parent


def get_version(rel_path):
    for line in (this_directory / rel_path).read_text().splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name="mellon",
    version=get_version("mellon/__init__.py"),
    description="Non-parametric density estimator.",
    url="https://github.com/settylab/mellon",
    author="Setty Lab",
    author_email="dominik.otto@gmail.com",
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
