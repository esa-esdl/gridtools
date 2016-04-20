import unittest

import numba as nb
import numpy as np


@nb.jit
def return_a_tuple(n):
    out = np.arange(n, dtype=np.float64)
    return out, n, n+4, 'A'


class NumbaTest(unittest.TestCase):
    def test_can_return_a_tuple(self):
        out, n, m, s = return_a_tuple(8)
        self.assertEqual((8,), out.shape)
        self.assertEqual(np.float64, out.dtype)
        self.assertEqual(8, n)
        self.assertEqual(8 + 4, m)
        self.assertEqual('A', s)
