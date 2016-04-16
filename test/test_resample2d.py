import unittest

import numpy as np
from numpy.testing import assert_almost_equal

import gridtools as gt
import gridtools.resample._resample2d_numba as nbimpl
import gridtools.resample._resample2d_python as pyimpl

F = np.nan


def _test_resample2d(src, out_w, out_h, desired_out):
    actual = pyimpl._resample2d(np.array(src), gt.resample.DS_MEAN, gt.resample.US_LINEAR,
                                np.zeros((out_h, out_w), dtype=np.float64))
    assert_almost_equal(actual, np.array(desired_out), err_msg='Python resample 2D impl.')

    actual = nbimpl._resample2d(np.array(src), gt.resample.DS_MEAN, gt.resample.US_LINEAR,
                               np.zeros((out_h, out_w), dtype=np.float64))
    assert_almost_equal(actual, np.array(desired_out), err_msg='Numba resample 2D impl.')

    actual = gt.resample.resample2d(np.array(src), out_w, out_h,
                                    ds_method=gt.resample.DS_MEAN,
                                    us_method=gt.resample.US_LINEAR)
    assert_almost_equal(actual, np.array(desired_out), err_msg='Gridtools resample 2D impl.')


class Resample2dTest(unittest.TestCase):
    """
    Note: Always align this test with java/src/test/java/com/bc/perfcomp/gridtools/ResizerTest.java
    """

    def test_no_op(self):
        _test_resample2d([[1., 2.], [3., 4.]],
                         2, 2,
                         [[1., 2.], [3., 4.]])

    def test_interpolation(self):
        _test_resample2d([[1., 2., 3.]],
                         5, 1,
                         [[1, 1.5, 2, 2.5, 3]]),

        _test_resample2d([[1., 2., 3.]],
                         9, 1,
                         [[1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]])

        _test_resample2d([[1., 2., 3., 4.]],
                         6, 1,
                         [[1, 1.6, 2.2, 2.8, 3.4, 4]]),

        _test_resample2d([[1., 2.],
                          [3., 4.]],
                         4, 4,
                         [[3. / 3, 4. / 3, 5. / 3, 6. / 3],
                          [5. / 3, 6. / 3, 7. / 3, 8. / 3],
                          [7. / 3, 8. / 3, 9. / 3, 10. / 3],
                          [9. / 3, 10. / 3, 11. / 3, 12. / 3]]),

        _test_resample2d([[1., 2.],
                          [3., 4.]],
                         4, 4,
                         [[3. / 3, 4. / 3, 5. / 3, 6. / 3],
                          [5. / 3, 6. / 3, 7. / 3, 8. / 3],
                          [7. / 3, 8. / 3, 9. / 3, 10. / 3],
                          [9. / 3, 10. / 3, 11. / 3, 12. / 3]])

    def test_aggregation(self):
        _test_resample2d([[0.6, 0.2, 3.4],
                          [1.4, 1.6, 1.0],
                          [4.0, 2.8, 3.0]],
                         2, 2,
                         [[(0.6 + 0.5 * 0.2 + 0.5 * 1.4 + 0.25 * 1.6) / (1.0 + 0.5 + 0.5 + 0.25),
                           (3.4 + 0.5 * 0.2 + 0.5 * 1.0 + 0.25 * 1.6) / (1.0 + 0.5 + 0.5 + 0.25)],
                          [(4.0 + 0.5 * 1.4 + 0.5 * 2.8 + 0.25 * 1.6) / (1.0 + 0.5 + 0.5 + 0.25),
                           (3.0 + 0.5 * 1.0 + 0.5 * 2.8 + 0.25 * 1.6) / (1.0 + 0.5 + 0.5 + 0.25)]])

        _test_resample2d([[0.9, 0.5, 3.0, 4.0],
                          [1.1, 1.5, 1.0, 2.0],
                          [4.0, 2.1, 3.0, 5.0],
                          [3.0, 4.9, 3.0, 1.0]],
                         2, 2,
                         [[1.0, 2.5],
                          [3.5, 3.0]])
