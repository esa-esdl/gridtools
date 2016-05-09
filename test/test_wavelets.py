import unittest

from gridtools.wavelets import *

SQRT2 = np.sqrt(2.)
SQRT3 = np.sqrt(3.)


class WaveletsTest(unittest.TestCase):
    def test_transform_2d(self):
        class T:
            def low_low(self, a):
                return a + 100

            def hi_low(self, a):
                return a + 200

            def low_hi(self, a):
                return a + 300

            def hi_hi(self, a):
                return a + 400

        a = transform_2d(np.arange(4 * 4).reshape(4, 4), 2, T())
        np.testing.assert_almost_equal(a, np.array([
            [100, 201, 202, 203],
            [304, 405, 206, 207],
            [308, 309, 410, 411],
            [312, 313, 414, 415]
        ]))

        a = transform_2d(np.arange(4 * 8).reshape(4, 8), 2, T())
        np.testing.assert_almost_equal(a, np.array([
            [100, 101, 202, 203, 204, 205, 206, 207],
            [308, 309, 410, 411, 212, 213, 214, 215],
            [316, 317, 318, 319, 420, 421, 422, 423],
            [324, 325, 326, 327, 428, 429, 430, 431]
        ]))

    def test_get_indexes(self):
        indices = get_indices((4, 8), 2)
        self.assertEqual(len(indices), 1 + 2 * 3)
        self.assertEqual(indices[0], (slice(0, 1), slice(0, 2)))
        self.assertEqual(indices[1], (slice(0, 1), slice(2, 4)))
        self.assertEqual(indices[2], (slice(1, 2), slice(0, 2)))
        self.assertEqual(indices[3], (slice(1, 2), slice(2, 4)))
        self.assertEqual(indices[4], (slice(0, 2), slice(4, 8)))
        self.assertEqual(indices[5], (slice(2, 4), slice(0, 4)))
        self.assertEqual(indices[6], (slice(2, 4), slice(4, 8)))

    def test_get_filter_pair(self):
        a, b = get_filter_pair([1, 1])  # Haar wavelet
        np.testing.assert_almost_equal(a, np.array([1 / SQRT2, 1 / SQRT2]))
        np.testing.assert_almost_equal(b, np.array([1 / SQRT2, -1 / SQRT2]))
        np.testing.assert_almost_equal((a * b).sum(), 0.)

        d4 = 1 / SQRT2 / 4 * np.array([1 + SQRT3, 3 + SQRT3, 3 - SQRT3, 1 - SQRT3])
        a, b = get_filter_pair(d4)
        np.testing.assert_almost_equal(a, d4)
        np.testing.assert_almost_equal(b, 1 / SQRT2 / 4 * np.array([1 - SQRT3, -(3 - SQRT3), 3 + SQRT3, -(1 + SQRT3)]))
        np.testing.assert_almost_equal((a * b).sum(), 0.)

    def test_wt_analyse_1d(self):
        x = np.array([1, 0, -3, 2, 1, 0, 1, 2], dtype=np.float64)
        y = wt_analyse_1d(x, max_level=1)
        np.testing.assert_almost_equal(y, np.array([1 / SQRT2, -1 / SQRT2, 1 / SQRT2, 3 / SQRT2,
                                                    1 / SQRT2, -5 / SQRT2, 1 / SQRT2, -1 / SQRT2]))

        y = wt_analyse_1d(x, max_level=2)
        np.testing.assert_almost_equal(y, np.array([0, 2, 1, -1,
                                                    1 / SQRT2, -5 / SQRT2, 1 / SQRT2, -1 / SQRT2]))

        y = wt_analyse_1d(x, max_level=3)
        np.testing.assert_almost_equal(y, np.array([SQRT2,
                                                    -SQRT2,
                                                    1, -1,
                                                    1 / SQRT2, -5 / SQRT2, 1 / SQRT2, -1 / SQRT2]))

    def test_wt_synthesize_1d(self):
        x = np.array([1, 0, -3, 2, 1, 0, 1, 2], dtype=np.float64)

        y = wt_analyse_1d(x, max_level=1)
        x2 = wt_synthesize_1d(y, max_level=1)
        np.testing.assert_almost_equal(x2, x)

        y = wt_analyse_1d(x, max_level=2)
        x2 = wt_synthesize_1d(y, max_level=2)
        np.testing.assert_almost_equal(x2, x)

        y = wt_analyse_1d(x, max_level=3)
        x2 = wt_synthesize_1d(y, max_level=3)
        np.testing.assert_almost_equal(x2, x)

    def test_wt_analyse_2d(self):
        x = np.array([0, 0], dtype=np.float64)
        y = wt_analyse_1d(x, max_level=1)
        x2 = wt_synthesize_1d(y, max_level=1)
        np.testing.assert_almost_equal(x2, x)

        x = np.array([[0, 0],
                      [0, 0]], dtype=np.float64)
        y = wt_analyse_2d(x, max_level=1)
        x2 = wt_synthesize_2d(y, max_level=1)
        np.testing.assert_almost_equal(x2, x)

        A = 1 / SQRT2

        x = np.array([1, 1], dtype=np.float64)
        y = wt_analyse_1d(x, max_level=1)
        np.testing.assert_almost_equal(y, np.array([2 * A, 0]))
        x2 = wt_synthesize_1d(y, max_level=1)
        np.testing.assert_almost_equal(x2, x)

        x = np.array([[1, 1],
                      [1, 1]], dtype=np.float64)
        y = wt_analyse_2d(x, max_level=1)
        np.testing.assert_almost_equal(y, np.array([[A * (2 * A) + A * (2 * A), 0],
                                                    [0, 0]]))
        x2 = wt_synthesize_2d(y, max_level=1)
        np.testing.assert_almost_equal(x2, x)

        x = np.array([[1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1]], dtype=np.float64)
        y = wt_analyse_2d(x, max_level=1)
        np.testing.assert_almost_equal(y, np.array([[2, 2, 0, 0],
                                                    [2, 2, 0, 0],
                                                    [0, 0, 0, 0],
                                                    [0, 0, 0, 0]]))
        x2 = wt_synthesize_2d(y, max_level=1)
        np.testing.assert_almost_equal(x2, x)

        x = np.array([[1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1]], dtype=np.float64)
        y = wt_analyse_2d(x, max_level=2)
        np.testing.assert_almost_equal(y, np.array([[4, 0, 0, 0],
                                                    [0, 0, 0, 0],
                                                    [0, 0, 0, 0],
                                                    [0, 0, 0, 0]]))
        x2 = wt_synthesize_2d(y, max_level=2)
        np.testing.assert_almost_equal(x2, x)

        x = np.array([[1, 1, 1, 1],
                      [1, 2, 2, 1],
                      [1, 2, 2, 1],
                      [1, 1, 1, 1]], dtype=np.float64)
        y = wt_analyse_2d(x, max_level=2)
        x2 = wt_synthesize_2d(y, max_level=2)
        np.testing.assert_almost_equal(x2, x)
