from setuptools import setup


def get_version():
    version_file = 'gridtools/version.py'
    locals = {}
    try:
        execfile(version_file, None, locals)
    except NameError:
        with open(version_file) as fp:
            exec(fp.read(), None, locals)
    return locals.get('version', '0.0.0')


# Same effect as "from ect import __version__", but avoids importing ect:
__version__ = get_version()

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
