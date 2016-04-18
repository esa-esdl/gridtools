## Common Python tools for working with numeric grids

While there exists a number of Python functions that upsample numeric images and grids
to a higher resolution, there is a lack in methods that perform grid cell
aggregation for downsampling grids to a coarser resolution. Most of the existing
aggregation methods assume an integer factor between source and target grid so that
contributions of partly overlapping grid cell areas don't exist.

This repository is a host for the Python ``gridtools.resampling`` module which
provides high-performance aggregation methods needed for the ESA CAB-LAB project
These take into account partial contribution of source grid cells to a given target
grid cell.

This repo is independent of the ``cablab-core`` repository so that it can be used
it outside the scope of the CAB-LAB project.

### Use of Numba

gridtools optionally uses **numba** to JIT-compile the resampling functions.
Although the use of numba is optional, it is strongly recommended to install it
(e.g. using Miniconda) as it speeds up computations by up to a several hundred
times (!) compared to plain Python.

To disable JIT compilation, set environment variable ``NUMBA_DISABLE_JIT``
to a non-zero value.

There is an issue in Numba that currently limits its use in certain
cases when grids are represented by numpy masked arrays, see https://github.com/numba/numba/issues/1834



