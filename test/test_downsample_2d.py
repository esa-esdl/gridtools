import unittest

import numpy as np

import gridtools.resampling as gtr

NAN = np.nan


class Downsample2dTest(unittest.TestCase):
    def _test_downsample_2d(self, src, out_w, out_h, method, fill_value, desired):
        if not isinstance(src, (np.ndarray, np.generic)):
            src = np.array(src)
        if not isinstance(desired, (np.ndarray, np.generic)):
            desired = np.array(desired)

        actual = gtr.downsample_2d(src, out_w, out_h, method=method, fill_value=fill_value)
        np.testing.assert_almost_equal(actual=desired, desired=actual)

        if isinstance(src, np.ma.MaskedArray):
            self.assertEqual(type(desired), np.ma.MaskedArray)
            np.testing.assert_equal(actual=actual.mask, desired=desired.mask)
            if not np.isfinite(fill_value):
                self.assertTrue(not np.isfinite(desired.fill_value))
            else:
                self.assertEqual(fill_value, desired.fill_value)

    def test_no_op(self):
        self._test_downsample_2d([[1., 2.], [3., 4.]], 2, 2, gtr.DS_MEAN, -1., [[1., 2.], [3., 4.]])

    def test_aggregation_mean(self):
        self._test_downsample_2d([[0.6, 0.2, 3.4],
                                  [1.4, 1.6, 1.0],
                                  [4.0, 2.8, 3.0]],
                                 2, 2, gtr.DS_MEAN, -1.,
                                 [[(0.6 + 0.5 * 0.2 + 0.5 * 1.4 + 0.25 * 1.6) / (1.0 + 0.5 + 0.5 + 0.25),
                                   (3.4 + 0.5 * 0.2 + 0.5 * 1.0 + 0.25 * 1.6) / (1.0 + 0.5 + 0.5 + 0.25)],
                                  [(4.0 + 0.5 * 1.4 + 0.5 * 2.8 + 0.25 * 1.6) / (1.0 + 0.5 + 0.5 + 0.25),
                                   (3.0 + 0.5 * 1.0 + 0.5 * 2.8 + 0.25 * 1.6) / (1.0 + 0.5 + 0.5 + 0.25)]])

        self._test_downsample_2d([[0.9, 0.5, 3.0, 4.0],
                                  [1.1, 1.5, 1.0, 2.0],
                                  [4.0, 2.1, 3.0, 5.0],
                                  [3.0, 4.9, 3.0, 1.0]],
                                 2, 2, gtr.DS_MEAN, -1.,
                                 [[1.0, 2.5],
                                  [3.5, 3.0]])

        self._test_downsample_2d([[NAN, NAN, 3.0, 4.0],
                                  [NAN, NAN, 1.0, 2.0],
                                  [4.0, 2.1, 3.0, NAN],
                                  [3.0, 4.9, NAN, 1.0]],
                                 2, 2, gtr.DS_MEAN, -1.,
                                 [[-1., 2.5],
                                  [3.5, 2.0]])

    def test_aggregation_mean_masked(self):
        self._test_downsample_2d(np.ma.array([[0.9, 0.5, 3.0, 4.0],
                                              [1.1, NAN, 1.0, 2.0],
                                              [4.0, 2.1, 3.0, 5.0],
                                              [3.0, 4.9, NAN, 1.0]],
                                             mask=[[1, 1, 0, 0],
                                                   [1, 1, 0, 0],
                                                   [0, 0, 0, 1],
                                                   [0, 0, 0, 0]]),
                                 2, 2, gtr.DS_MEAN, NAN,
                                 np.ma.array([[NAN, 2.5],
                                              [3.5, 2.0]],
                                             fill_value=NAN,
                                             mask=[[1, 0],
                                                   [0, 0]]))

    def test_aggregation_mode(self):
        self._test_downsample_2d([[2, 4, 1],
                                  [1, 2, 2],
                                  [1, 1, 1]],
                                 2, 2, gtr.DS_MODE, 0,
                                 [[2, 1],
                                  [1, 2]])

        self._test_downsample_2d([[3, 5, 2, 1],
                                  [3, 5, 4, 3],
                                  [1, 1, 3, 4],
                                  [4, 1, 4, 4]],
                                 2, 2, gtr.DS_MODE, 0,
                                 [[3, 2],
                                  [1, 4]])

    def test_aggregation_mode_masked(self):

        self._test_downsample_2d(np.ma.array([[3, 5, 2, 1],
                                              [3, 5, 4, 3],
                                              [1, 1, 3, 4],
                                              [4, 1, 4, 4]],
                                             mask=[[0, 0, 1, 1],
                                                   [0, 0, 1, 1],
                                                   [0, 0, 0, 0],
                                                   [0, 0, 0, 0]]),
                                 2, 2, gtr.DS_MODE, 9,
                                 np.ma.array([[3, 9],
                                              [1, 4]],
                                             fill_value=9,
                                             mask=[[0, 1],
                                                   [0, 0]]))

    def test_aggregation_first(self):
        self._test_downsample_2d([[0.6, 0.2, 3.4],
                                  [1.4, NAN, 1.0],
                                  [4.0, 2.8, 3.0]],
                                 2, 2, gtr.DS_FIRST, -1.,
                                 [[0.6, 0.2],
                                  [1.4, 1.0]])

        self._test_downsample_2d([[0.9, 0.5, 3.0, 4.0],
                                  [1.1, 1.5, 1.0, NAN],
                                  [NAN, NAN, 3.0, 5.0],
                                  [3.0, 4.9, NAN, 1.0]],
                                 2, 2, gtr.DS_FIRST, -1.,
                                 [[0.9, 3.0],
                                  [3.0, 3.0]])

    def test_aggregation_last(self):
        self._test_downsample_2d([[0.6, 0.2, 3.4],
                                  [1.4, NAN, 1.0],
                                  [4.0, 2.8, 3.0]],
                                 2, 2, gtr.DS_LAST, -1.,
                                 [[1.4, 1.0],
                                  [2.8, 3.0]])

        self._test_downsample_2d([[0.9, 0.5, 3.0, 4.0],
                                  [1.1, 1.5, 1.0, NAN],
                                  [NAN, 2.1, 3.0, 5.0],
                                  [NAN, NAN, 4.2, NAN]],
                                 2, 2, gtr.DS_LAST, -1,
                                 [[1.5, 1.0],
                                  [2.1, 4.2]])