from setuptools import find_packages, setup

# Dependencies
with open("requirements.txt") as f:
    requirements = f.readlines()
INSTALL_REQUIRES = [t.strip() for t in requirements]

setup(
    name="echoregions",
    version="0.1.0",
    author="Caesar Tuguinay",
    author_email="ctuguina@uw.edu",
    maintainer="Caesar Tuguinay",
    maintainer_email="ctuguina@uw.edu",
    description="Parsers and functions for working with EVR and EVL files",
    url="https://github.com/OSOceanAcoustics/echoregions",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
)
