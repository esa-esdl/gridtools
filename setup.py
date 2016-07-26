from setuptools import setup

from gridtools import __version__

setup(
    name="gridtools",
    version=__version__,
    license='GPL 3',
    author='Norman Fomferra',
    maintainer='Brockmann Consult GmbH',
    packages=['gridtools'],
    # *Minimum* requirements
    install_requires=['numpy', 'numba']
)
