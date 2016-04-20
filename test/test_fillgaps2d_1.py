import unittest

import numpy as np
from numpy.testing import assert_almost_equal

import gridtools.gapfilling as gtg

GAP = np.nan


def _test_fillgaps(src, desired_out):
    src = np.array(src)
    actual = gtg.fillgaps2d(src, method=gtg.GF_AVG_NEAREST_CONTINUED)
    assert_almost_equal(actual, np.array(desired_out))


class Resample2dTest1(unittest.TestCase):
    def test_0_missing(self):
        _test_fillgaps([[1.0, 2.0],
                        [3.0, 4.0]],
                       [[1.0, 2.0],
                        [3.0, 4.0]])

    def test_1_missing(self):
        _test_fillgaps([[GAP]],
                       [[GAP]])

        _F_ = (2 + 3 + 4) / 3.
        _test_fillgaps([[GAP, 2.0],
                        [3.0, 4.0]],
                       [[_F_, 2.0],
                        [3.0, 4.0]])

        _F_ = (1 + 2 + 3) / 3.
        _test_fillgaps([[1.0, 2.0],
                        [3.0, GAP]],
                       [[1.0, 2.0],
                        [3.0, _F_]])

        _F_ = (1 + 2 + 3 + 4 + 6 + 7 + 8 + 9) / 8.
        _test_fillgaps([[1.0, 2.0, 3.0],
                        [4.0, GAP, 6.0],
                        [7.0, 8.0, 9.0]],
                       [[1.0, 2.0, 3.0],
                        [4.0, _F_, 6.0],
                        [7.0, 8.0, 9.0]])

    def test_2_missing(self):
        _test_fillgaps([[GAP, GAP]],
                       [[GAP, GAP]])

        F1_ = (2 + 3) / 2.
        F2_ = (2 + 3) / 2.
        _test_fillgaps([[GAP, 2.0],
                        [3.0, GAP]],
                       [[F1_, 2.0],
                        [3.0, F2_]])

        F1_ = (2 + 4) / 2.
        F2_ = (2 + 3 + 4 + 6 + 7 + 8 + 9) / 7.
        _test_fillgaps([[GAP, 2.0, 3.0],
                        [4.0, GAP, 6.0],
                        [7.0, 8.0, 9.0]],
                       [[F1_, 2.0, 3.0],
                        [4.0, F2_, 6.0],
                        [7.0, 8.0, 9.0]])

    def test_3_missing(self):
        _test_fillgaps([[GAP, GAP],
                        [GAP, 4.0]],
                       [[4.0, 4.0],
                        [4.0, 4.0]])

        F1_ = 2.
        F2_ = (2 + 7 + 8) / 3.
        F3_ = (2 + 3 + 6 + 7 + 8 + 9) / 6.
        _test_fillgaps([[GAP, 2.0, 3.0],
                        [GAP, GAP, 6.0],
                        [7.0, 8.0, 9.0]],
                       [[F1_, 2.0, 3.0],
                        [F2_, F3_, 6.0],
                        [7.0, 8.0, 9.0]])

    def test_4_missing(self):
        _test_fillgaps([[GAP, GAP],
                        [GAP, GAP]],
                       [[GAP, GAP],
                        [GAP, GAP]])

        F1_ = (3 + 6) / 2.
        F2_ = (7 + 8) / 2.
        F3_ = (3 + 6 + 7 + 8 + 9) / 5.
        F4_ = (F1_ + F2_ + F3_) / 3.
        _test_fillgaps([[GAP, GAP, 3.0],
                        [GAP, GAP, 6.0],
                        [7.0, 8.0, 9.0]],
                       [[F4_, F1_, 3.0],
                        [F2_, F3_, 6.0],
                        [7.0, 8.0, 9.0]])

    def test_9_missing(self):
        F1_ = (1 + 2 + 3 + 4 + 5 + 9) / 6.
        F2_ = (2 + 3 + 4) / 3.
        F3_ = (3 + 4) / 2.
        F4_ = (5 + 9 + 13) / 3.
        F5_ = (9 + 13) / 2.
        F6_ = (F1_ + F2_ + F3_ + F4_ + F5_) / 5.
        F7_ = (F2_ + F3_) / 2.
        F8_ = (F4_ + F5_) / 2.
        F9_ = (F6_ + F7_ + F8_) / 3.
        _test_fillgaps([[1.0, 2.0, 3.0, 4.0],
                        [5.0, GAP, GAP, GAP],
                        [9.0, GAP, GAP, GAP],
                        [13., GAP, GAP, GAP]],
                       [[1.0, 2.0, 3.0, 4.0],
                        [5.0, F1_, F2_, F3_],
                        [9.0, F4_, F6_, F7_],
                        [13., F5_, F8_, F9_]])
