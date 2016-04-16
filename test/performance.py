import timeit

import numpy as np
from gridtools.resample import *
import gridtools.resample._resample2d_python as pyimpl
import gridtools.resample._resample2d_numba as nbimpl

MAIN = 'from __main__ import np, pyimpl, nbimpl, a, out, US_LINEAR, DS_MEAN'
times = 100
N = 7

print('\nUpsampling:')
print('No\tSize\tPython\tNumba\tNumba-Gain')
s = 4
for i in range(N):
    a = np.random.rand(s, s)
    out = np.zeros((int(s * 2.5), int(s * 2.1)), dtype=np.float64)
    t1 = timeit.timeit(setup=MAIN, number=times, stmt='pyimpl._upsample2d(a, US_LINEAR, out)')
    t2 = timeit.timeit(setup=MAIN, number=times, stmt='nbimpl._upsample2d(a, US_LINEAR, out)')
    print('%d\t%d\t%f\t%f\t%f' % (i + 1, s, t1, t2, t1 / t2))
    s *= 2

print('\nDownsampling:')
print('No\tSize\tPython\tNumba\tNumba-Gain')
s = 4
for i in range(N):
    a = np.random.rand(s, s)
    out = np.zeros((int(s / 2.5), int(s / 2.1)), dtype=np.float64)
    t1 = timeit.timeit(setup=MAIN, number=times, stmt='pyimpl._downsample2d(a, DS_MEAN, out)')
    t2 = timeit.timeit(setup=MAIN, number=times, stmt='nbimpl._downsample2d(a, DS_MEAN, out)')
    print('%d\t%d\t%f\t%f\t%f' % (i + 1, s, t1, t2, t1 / t2))
    s *= 2
