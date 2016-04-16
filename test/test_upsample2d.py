import unittest

import numpy as np
from numpy.testing import assert_almost_equal

import gridtools as gt
import gridtools.resample._resample2d_numba as nbimpl
import gridtools.resample._resample2d_python as pyimpl

F = np.nan


def _test_upsample2d(src, out_w, out_h, us_method, desired_out):
    actual = pyimpl._upsample2d(np.array(src), us_method,
                                np.zeros((out_h, out_w), dtype=np.float64))
    assert_almost_equal(actual, np.array(desired_out), err_msg='Python upsample 2D impl.')

    actual = nbimpl._upsample2d(np.array(src), us_method,
                                np.zeros((out_h, out_w), dtype=np.float64))
    assert_almost_equal(actual, np.array(desired_out), err_msg='Numba upsample 2D impl.')

    actual = gt.resample.upsample2d(np.array(src), out_w, out_h,
                                    us_method=us_method)
    assert_almost_equal(actual, np.array(desired_out), err_msg='Gridtools upsample 2D impl.')


class Upsample2dTest(unittest.TestCase):
    """
    Note: Always align this test with java/src/test/java/com/bc/perfcomp/gridtools/ResizerTest.java
    """

    def test_no_op(self):
        _test_upsample2d([[1., 2.], [3., 4.]],
                         2, 2, gt.resample.US_NEAREST,
                         [[1., 2.], [3., 4.]])

    def test_interpolation_linear(self):
        _test_upsample2d([[1., 2., 3.]],
                         5, 1, gt.resample.US_LINEAR,
                         [[1, 1.5, 2, 2.5, 3]]),

        _test_upsample2d([[1., 2., 3.]],
                         9, 1, gt.resample.US_LINEAR,
                         [[1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]])

        _test_upsample2d([[1., 2., 3., 4.]],
                         6, 1, gt.resample.US_LINEAR,
                         [[1, 1.6, 2.2, 2.8, 3.4, 4]]),

        _test_upsample2d([[1., 2.],
                          [3., 4.]],
                         4, 4, gt.resample.US_LINEAR,
                         [[3. / 3, 4. / 3, 5. / 3, 6. / 3],
                          [5. / 3, 6. / 3, 7. / 3, 8. / 3],
                          [7. / 3, 8. / 3, 9. / 3, 10. / 3],
                          [9. / 3, 10. / 3, 11. / 3, 12. / 3]]),

        _test_upsample2d([[1., 2.],
                          [3., 4.]],
                         4, 4, gt.resample.US_LINEAR,
                         [[3. / 3, 4. / 3, 5. / 3, 6. / 3],
                          [5. / 3, 6. / 3, 7. / 3, 8. / 3],
                          [7. / 3, 8. / 3, 9. / 3, 10. / 3],
                          [9. / 3, 10. / 3, 11. / 3, 12. / 3]])

    def test_nearest(self):
        _test_upsample2d([[1., 2., 3.]],
                         5, 1, gt.resample.US_NEAREST,
                         [[1., 1., 2., 2., 3.]]),

        _test_upsample2d([[1., 2., 3.]],
                         9, 1, gt.resample.US_NEAREST,
                         [[1., 1., 1., 2., 2., 2., 3., 3., 3.]])

        _test_upsample2d([[1., 2., 3., 4.]],
                         6, 1, gt.resample.US_NEAREST,
                         [[1., 1., 2., 3., 3., 4.]]),

        _test_upsample2d([[1., 2.],
                          [3., 4.]],
                         4, 4, gt.resample.US_NEAREST,
                         [[1., 1., 2., 2.],
                          [1., 1., 2., 2.],
                          [3., 3., 4., 4.],
                          [3., 3., 4., 4.]]),

        _test_upsample2d([[1., 2.],
                          [3., 4.]],
                         6, 4, gt.resample.US_NEAREST,
                         [[1., 1., 1., 2., 2., 2.],
                          [1., 1., 1., 2., 2., 2.],
                          [3., 3., 3., 4., 4., 4.],
                          [3., 3., 3., 4., 4., 4.]])
