import unittest

import numba as nb
import numpy as np


@nb.jit(nopython=True)
def return_a_tuple(n):
    out = np.array([n], dtype=np.float64)
    return out, n, n + 4


@nb.jit(nopython=False)
def recursive_sum(a):
    if a.size == 1:
        return a[0]
    else:
        return a[0] + recursive_sum(a[1:])


class NumbaTest(unittest.TestCase):
    def test_can_return_a_tuple(self):
        out, n, m = return_a_tuple(8)
        self.assertEqual((1,), out.shape)
        self.assertEqual(np.float64, out.dtype)
        self.assertEqual(8, n)
        self.assertEqual(8 + 4, m)

    def test_can_do_recursion(self):
        out = recursive_sum(np.array([1., 2, 3]))
        self.assertEqual(out, 6)
