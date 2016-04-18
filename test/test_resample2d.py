import unittest

import numpy as np
from numpy.testing import assert_almost_equal

import gridtools.resampling as gtr


def _test_resample2d(src, out_w, out_h, ds_method, us_method, desired_out):
    actual = gtr.resample2d(np.array(src), out_w, out_h,
                            ds_method=ds_method,
                            us_method=us_method)
    assert_almost_equal(actual, np.array(desired_out))


SRC = [[0.9, 0.5, 3.0, 4.0],
       [1.1, 1.5, 1.0, 2.0],
       [4.0, 2.1, 3.0, 5.0],
       [3.0, 4.9, 3.0, 1.0]]


class Resample2dTest(unittest.TestCase):
    """
    Note: Always align this test with java/src/test/java/com/bc/perfcomp/gridtools/ResizerTest.java
    """

    def test_no_op(self):
        _test_resample2d(SRC,
                         4, 4, gtr.DS_FIRST, gtr.US_NEAREST,
                         SRC)

    def test_aggregate_w(self):
        _test_resample2d(SRC,
                         2, 4, gtr.DS_FIRST, gtr.US_NEAREST,
                         [[0.9, 3.],
                          [1.1, 1.],
                          [4., 3.],
                          [3., 3.]])

    def test_aggregate_w_aggregate_h(self):
        _test_resample2d(SRC,
                         2, 2, gtr.DS_MEAN, gtr.US_NEAREST,
                         [[1.0, 2.5],
                          [3.5, 3.0]])

    def test_aggregate_w_interpolate_h(self):
        _test_resample2d(SRC,
                         2, 8, gtr.DS_FIRST, gtr.US_NEAREST,
                         [[0.9, 3.],
                          [0.9, 3.],
                          [1.1, 1.],
                          [1.1, 1.],
                          [4., 3.],
                          [4., 3.],
                          [3., 3.],
                          [3., 3.]])

    def test_aggregate_h(self):
        _test_resample2d(SRC,
                         4, 2, gtr.DS_FIRST, gtr.US_NEAREST,
                         [[0.9, 0.5, 3., 4.],
                          [4., 2.1, 3., 5.]])

    def test_interpolate_w(self):
        _test_resample2d(SRC,
                         8, 4, gtr.DS_MEAN, gtr.US_NEAREST,
                         [[0.9, 0.9, 0.5, 0.5, 3., 3., 4., 4.],
                          [1.1, 1.1, 1.5, 1.5, 1., 1., 2., 2.],
                          [4., 4., 2.1, 2.1, 3., 3., 5., 5.],
                          [3., 3., 4.9, 4.9, 3., 3., 1., 1.]])

    def test__interpolate_w_interpolate_h(self):
        _test_resample2d(SRC,
                         8, 8, gtr.DS_MEAN, gtr.US_NEAREST,
                         [[0.9, 0.9, 0.5, 0.5, 3., 3., 4., 4.],
                          [0.9, 0.9, 0.5, 0.5, 3., 3., 4., 4.],
                          [1.1, 1.1, 1.5, 1.5, 1., 1., 2., 2.],
                          [1.1, 1.1, 1.5, 1.5, 1., 1., 2., 2.],
                          [4., 4., 2.1, 2.1, 3., 3., 5., 5.],
                          [4., 4., 2.1, 2.1, 3., 3., 5., 5.],
                          [3., 3., 4.9, 4.9, 3., 3., 1., 1.],
                          [3., 3., 4.9, 4.9, 3., 3., 1., 1.]])

    def test__interpol_w_aggregate_h(self):
        _test_resample2d(SRC,
                         8, 2, gtr.DS_MEAN, gtr.US_NEAREST,
                         [[1., 1., 1., 1., 2., 2., 3., 3.],
                          [3.5, 3.5, 3.5, 3.5, 3., 3., 3., 3.]])
