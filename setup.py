from setuptools import setup

setup(
    name="gridtools",
    version="0.0.1",
    license='GPL 3',
    author='Norman Fomferra',
    maintainer='Brockmann Consult GmbH',
    packages=['gridtools'],
    # *Minimum* requirements
    install_requires=['numpy', 'numba']
)
