import timeit

import numpy as np
import gridtools.resample as rs

MAIN = 'from __main__ import np, rs, a, out'
times = 100
N = 5

print('\nUpsampling:')
print('No\tSize\tPython\tNumba\tNumba-Gain')
s = 4
for i in range(N):
    a = np.random.rand(s, s)
    out = np.zeros((int(s * 2.5), int(s * 2.1)), dtype=np.float64)
    t1 = timeit.timeit(setup=MAIN, number=times, stmt='rs._upsample2d(a, rs.US_LINEAR, 0., out)')
    print('%d\t%d\t%f' % (i + 1, s, t1))
    s *= 2

print('\nDownsampling:')
print('No\tSize\tPython\tNumba\tNumba-Gain')
s = 4
for i in range(N):
    a = np.random.rand(s, s)
    out = np.zeros((int(s / 2.5), int(s / 2.1)), dtype=np.float64)
    t1 = timeit.timeit(setup=MAIN, number=times, stmt='rs._downsample2d(a, rs.DS_MEAN, 0., out)')
    print('%d\t%d\t%f' % (i + 1, s, t1))
    s *= 2
