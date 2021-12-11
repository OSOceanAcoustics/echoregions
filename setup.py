from setuptools import find_packages, setup

# Dependencies
with open("requirements.txt") as f:
    requirements = f.readlines()
INSTALL_REQUIRES = [t.strip() for t in requirements]

setup(
    name="echoregions",
    version="0.1",
    author="Kavin Nguyen",
    author_email="ng.kavin97@gmail.com",
    maintainer="Kavin Nguyen",
    maintainer_email="ng.kavin97@gmail.com",
    description="Parsers and functions for working with EVR and EVL files",
    url="https://github.com/OSOceanAcoustics/echoregions",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
)
