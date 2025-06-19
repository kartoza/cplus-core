from pathlib import Path

from setuptools import setup, find_packages
from cplus_core import __version__

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="cplus_core",
    version=__version__,
    description="Library supporting analysis of CPLUS framework.",
    long_description=long_description,
    url="https://github.com/kartoza/cplus-core",
    author="Conservation International",
    author_email="trends.earth@conservation.org",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "OSI Approved :: GNU General Public License (GPL)"
        "Programming Language :: Python :: 3.10",
    ],
    keywords="cplus plugin qgis",
    packages=find_packages(exclude=[
        "tests", "tests.*", "docker-compose.yml",
        "test_suite.py", "scripts", "scripts.*",
        "run-tests.sh"
    ]),
    package_data={"cplus_core": ["version.json", "data/**"]},
    include_package_data=True,
    install_requires=[],
    extras_require={
        "dev": [],
        "test": [],
    },
)
