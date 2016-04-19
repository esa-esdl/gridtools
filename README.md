[![Build Status](https://travis-ci.org/CAB-LAB/gridtools.svg?branch=master)](https://travis-ci.org/CAB-LAB/gridtools)
[![codecov.io](https://codecov.io/github/CAB-LAB/gridtools/coverage.svg?branch=master)](https://codecov.io/github/CAB-LAB/gridtools?branch=master)

# gridtools - Python tools for numeric grids

While there exists a number of Python functions that upsample numeric images and grids
to a higher resolution, there is a lack in methods that perform grid cell
aggregation for downsampling grids to a coarser resolution. Most of the existing
aggregation methods assume an integer factor between source and target grid so that
contributions of partly overlapping grid cell areas don't exist.

This repo is independent of the ``cablab-core`` repository so that it can be used
it outside the scope of the CAB-LAB project.

## Modules

### Module ``gridtools.resampling``

The ``gridtools.resampling`` module which provides high-performance (geometric, 2-D)
regridding functions needed for the ESA CAB-LAB project: ``resample()``, ``upsample()``,
``downsample()``. Downsampling can take into account partial contributions of
source grid cells for a given target grid cell:

* Method ``MEAN``: Average based aggregation weighted by contribution area. Useful
   for downsampling grids whose cell values represent continuous values, e.g.
   temperatures, radiation.
* Method ``MODE``: Frequency/occurrences based aggregation weighted by contribution area. Useful
   for downsampling grids whose cell values represent classes, e.g. surface types, flags.


## Use of Numba

gridtools optionally uses [Numba](http://numba.pydata.org/) to JIT-compile the resampling functions.
Although the use of numba is optional, it is strongly recommended to install it
(e.g. using Miniconda) as it speeds up computations by up to a several hundred
times (!) compared to plain Python.

To disable JIT compilation, set environment variable ``NUMBA_DISABLE_JIT``
to a non-zero value.

There is an issue in Numba that currently limits its use in certain
cases when grids are represented by numpy masked arrays, see https://github.com/numba/numba/issues/1834



