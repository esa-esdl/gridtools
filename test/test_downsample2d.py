import unittest

import numpy as np
from numpy.testing import assert_almost_equal

import gridtools.resample as gtr

NAN = np.nan


def _test_downsample2d(src, out_w, out_h, method, fill_value, desired_out):
    if not isinstance(src, (np.ndarray, np.generic)):
        src = np.array(src)
    desired_out_array = np.array(desired_out)
    actual = gtr.downsample2d(src, out_w, out_h, method=method, fill_value=fill_value)
    assert_almost_equal(actual, desired_out_array)


class Downsample2dTest(unittest.TestCase):
    """
    Note: Always align this test with java/src/test/java/com/bc/perfcomp/gridtools/ResizerTest.java
    """

    def test_no_op(self):
        _test_downsample2d([[1., 2.], [3., 4.]], 2, 2, gtr.DS_MEAN, -1., [[1., 2.], [3., 4.]])

    def test_aggregation_mean(self):
        _test_downsample2d([[0.6, 0.2, 3.4],
                            [1.4, 1.6, 1.0],
                            [4.0, 2.8, 3.0]], 2, 2, gtr.DS_MEAN, -1.,
                           [[(0.6 + 0.5 * 0.2 + 0.5 * 1.4 + 0.25 * 1.6) / (1.0 + 0.5 + 0.5 + 0.25),
                             (3.4 + 0.5 * 0.2 + 0.5 * 1.0 + 0.25 * 1.6) / (1.0 + 0.5 + 0.5 + 0.25)],
                            [(4.0 + 0.5 * 1.4 + 0.5 * 2.8 + 0.25 * 1.6) / (1.0 + 0.5 + 0.5 + 0.25),
                             (3.0 + 0.5 * 1.0 + 0.5 * 2.8 + 0.25 * 1.6) / (1.0 + 0.5 + 0.5 + 0.25)]])

        _test_downsample2d([[0.9, 0.5, 3.0, 4.0],
                            [1.1, 1.5, 1.0, 2.0],
                            [4.0, 2.1, 3.0, 5.0],
                            [3.0, 4.9, 3.0, 1.0]], 2, 2, gtr.DS_MEAN, -1.,
                           [[1.0, 2.5],
                            [3.5, 3.0]])

        _test_downsample2d([[NAN, NAN, 3.0, 4.0],
                            [NAN, NAN, 1.0, 2.0],
                            [4.0, 2.1, 3.0, 5.0],
                            [3.0, 4.9, 3.0, 1.0]], 2, 2, gtr.DS_MEAN, -1.,
                           [[-1., 2.5],
                            [3.5, 3.0]])

    def test_aggregation_mode(self):
        _test_downsample2d([[2, 4, 1],
                            [1, 2, 2],
                            [1, 1, 1]],
                           2, 2, gtr.DS_MODE, 0,
                           [[2, 1],
                            [1, 2]])

        _test_downsample2d([[3, 5, 2, 1],
                            [3, 5, 4, 3],
                            [1, 1, 3, 4],
                            [4, 1, 4, 4]],
                           2, 2, gtr.DS_MODE, 0,
                           [[3, 2],
                            [1, 4]])

        _test_downsample2d(np.ma.array([[3, 5, 2, 1],
                                        [3, 5, 4, 3],
                                        [1, 1, 3, 4],
                                        [4, 1, 4, 4]],
                                       mask=[[0, 0, 1, 1],
                                             [0, 0, 1, 1],
                                             [0, 0, 0, 0],
                                             [0, 0, 0, 0]]),
                           2, 2, gtr.DS_MODE, 9,
                           [[3, 9],
                            [1, 4]])

    def test_aggregation_first(self):
        _test_downsample2d([[0.6, 0.2, 3.4],
                            [1.4, NAN, 1.0],
                            [4.0, 2.8, 3.0]],
                           2, 2, gtr.DS_FIRST, -1.,
                           [[0.6, 0.2],
                            [1.4, 1.0]])

        _test_downsample2d([[0.9, 0.5, 3.0, 4.0],
                            [1.1, 1.5, 1.0, NAN],
                            [NAN, NAN, 3.0, 5.0],
                            [3.0, 4.9, NAN, 1.0]],
                           2, 2, gtr.DS_FIRST, -1.,
                           [[0.9, 3.0],
                            [3.0, 3.0]])

    def test_aggregation_last(self):
        _test_downsample2d([[0.6, 0.2, 3.4],
                            [1.4, NAN, 1.0],
                            [4.0, 2.8, 3.0]],
                           2, 2, gtr.DS_LAST, -1.,
                           [[1.4, 1.0],
                            [2.8, 3.0]])

        _test_downsample2d([[0.9, 0.5, 3.0, 4.0],
                            [1.1, 1.5, 1.0, NAN],
                            [NAN, 2.1, 3.0, 5.0],
                            [NAN, NAN, 4.2, NAN]],
                           2, 2, gtr.DS_LAST, -1,
                           [[1.5, 1.0],
                            [2.1, 4.2]])
