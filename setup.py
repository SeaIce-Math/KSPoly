from setuptools import setup, find_packages

# Read the contents of the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='KSPoly',
    version='1.0',
    packages=find_packages(),
    install_requires=requirements,
)


