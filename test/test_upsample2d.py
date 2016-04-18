import unittest

import numpy as np
from numpy.testing import assert_almost_equal

import gridtools.resample as gtr

NAN = np.nan


def _test_upsample2d(src, out_w, out_h, us_method, fill_value, desired_out):
    actual = gtr.upsample2d(np.array(src), out_w, out_h,
                            method=us_method, fill_value=fill_value)
    assert_almost_equal(actual, np.array(desired_out))


class Upsample2dTest(unittest.TestCase):
    """
    Note: Always align this test with java/src/test/java/com/bc/perfcomp/gridtools/ResizerTest.java
    """

    def test_no_op(self):
        _test_upsample2d([[1., 2.], [3., 4.]], 2, 2, gtr.US_NEAREST, -1., [[1., 2.], [3., 4.]])

    def test_interpolation_linear(self):
        _test_upsample2d([[1., 2., 3.]], 5, 1, gtr.US_LINEAR, -1., [[1, 1.5, 2, 2.5, 3]]),

        _test_upsample2d([[1., 2., 3.]], 9, 1, gtr.US_LINEAR, -1.,
                         [[1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]])

        _test_upsample2d([[1., 2., 3., 4.]], 6, 1, gtr.US_LINEAR, -1., [[1, 1.6, 2.2, 2.8, 3.4, 4]]),

        _test_upsample2d([[1., 2.],
                          [3., 4.]], 4, 4, gtr.US_LINEAR, -1., [[3. / 3, 4. / 3, 5. / 3, 6. / 3],
                                                                [5. / 3, 6. / 3, 7. / 3, 8. / 3],
                                                                [7. / 3, 8. / 3, 9. / 3, 10. / 3],
                                                                [9. / 3, 10. / 3, 11. / 3, 12. / 3]]),

        _test_upsample2d([[1., 2.],
                          [3., 4.]], 4, 4, gtr.US_LINEAR, -1., [[3. / 3, 4. / 3, 5. / 3, 6. / 3],
                                                                [5. / 3, 6. / 3, 7. / 3, 8. / 3],
                                                                [7. / 3, 8. / 3, 9. / 3, 10. / 3],
                                                                [9. / 3, 10. / 3, 11. / 3, 12. / 3]])

    def test_nearest(self):
        _test_upsample2d([[1., 2., 3.]], 5, 1, gtr.US_NEAREST, -1., [[1., 1., 2., 2., 3.]]),

        _test_upsample2d([[1., 2., 3.]], 9, 1, gtr.US_NEAREST, -1., [[1., 1., 1., 2., 2., 2., 3., 3., 3.]])

        _test_upsample2d([[1., 2., 3., 4.]], 6, 1, gtr.US_NEAREST, -1., [[1., 1., 2., 3., 3., 4.]]),

        _test_upsample2d([[1., 2.],
                          [3., 4.]], 4, 4, gtr.US_NEAREST, -1., [[1., 1., 2., 2.],
                                                                 [1., 1., 2., 2.],
                                                                 [3., 3., 4., 4.],
                                                                 [3., 3., 4., 4.]]),

        _test_upsample2d([[1., 2.],
                          [3., 4.]], 6, 4, gtr.US_NEAREST, -1., [[1., 1., 1., 2., 2., 2.],
                                                                 [1., 1., 1., 2., 2., 2.],
                                                                 [3., 3., 3., 4., 4., 4.],
                                                                 [3., 3., 3., 4., 4., 4.]])
