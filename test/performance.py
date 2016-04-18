import timeit

import numpy as np

import gridtools.resampling as gts

MAIN = 'from __main__ import np, gts, a, out, out_shape'
times = 100
N = 8

print('\nUpsampling:')
print('No\tSize\tPython\tNumba\tNumba-Gain')
src_size = 4
for i in range(N):
    a = np.random.rand(src_size, src_size)
    out_shape = int(src_size * 2.5), int(src_size * 2.1)
    out = np.zeros(out_shape, dtype=np.float64)
    t1 = timeit.timeit(setup=MAIN, number=times, stmt='gts.upsample2d(a, out_shape[-1], out_shape[-2], out=out)')
    print('%d\t%d\t%f' % (i + 1, src_size, t1))
    src_size *= 2

print('\nDownsampling:')
print('No\tSize\tPython\tNumba\tNumba-Gain')
src_size = 4
for i in range(N):
    a = np.random.rand(src_size, src_size)
    out_shape = int(src_size / 2.5), int(src_size / 2.1)
    out = np.zeros(out_shape, dtype=np.float64)
    t1 = timeit.timeit(setup=MAIN, number=times, stmt='gts.downsample2d(a, out_shape[-1], out_shape[-2], out=out)')
    print('%d\t%d\t%f' % (i + 1, src_size, t1))
    src_size *= 2
