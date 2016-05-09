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

The ``gridtools.resampling`` module provides high-performance (geometric, 2-D)
regridding functions needed for the ESA CAB-LAB project: ``resample_2d()``, ``upsample_2d()``,
``downsample_2d()``. Downsampling can take into account partial contributions of
source grid cells for a given target grid cell:

* Method ``MEAN``: Average based aggregation weighted by contribution area. Useful
   for downsampling grids whose cell values represent continuous values, e.g.
   temperatures, radiation.
* Method ``MODE``: Frequency/occurrences based aggregation weighted by contribution area. Useful
   for downsampling grids whose cell values represent classes, e.g. surface types, flags.


### Module ``gridtools.gapfilling``

The module provides functions that allow filling grid cells whose values are not *finite* by means of the
``numpy.isfinite()`` function. Two gap-filling methods are available:

* Function ``fillgaps_lowpass_2d()``: Fills cell values by averaging values of direct neighbours using a given kernel.
   This is repeated for the whole grid until all gaps are filled.
   * Pros: Simple and obviously working well for mostly isolated, single cell gaps.
   * Cons: Naive. Relatively slow, if gaps form larger connected areas. In this case gap border patterns propagate
     into gap area centers at multiples of 45 degree angles, producing strange visual artifacts, and usually an
     implausible distribution of filled values.
* Method ``fillgaps_multiscale_2d()``: Similar to ``fillgaps_lowpass_2d()`` but tries to get around its disadvantages:
     Cell values are filled in by averaging values of direct neighbours. Then the resulting grid is downsampled by a
     factor of two. If the downsampled grid still has gaps, the procedure is repeated recursively until the downsampled
     version contains no more gaps or it comprises only a single cell. Each gap-filled, downsampled grid serves as
     a source for gaps in the upsampled, 2x higher resolution grid until the original resolution is reached and all
     gaps are filled (or none).


## Use of Numba

gridtools uses [Numba](http://numba.pydata.org/) to JIT-compile the resampling functions as it speeds up
computations by up to a several hundred times (!) compared to plain Python.
It is strongly recommended to use gridtools with Numba in Miniconda or Anaconda environments.

To disable JIT compilation (e.g. for unit-level testing), set environment variable ``NUMBA_DISABLE_JIT``
to a non-zero value.

There is an issue in Numba that currently limits its use in certain
cases when grids are represented by numpy masked arrays, see https://github.com/numba/numba/issues/1834



