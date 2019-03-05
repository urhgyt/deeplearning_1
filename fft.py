import numpy
from reikna.fft import FFT
import reikna.cluda as cluda

api = cluda.cuda_api()
thr = api.Thread.create()
x = numpy.array((
    [[1, 1, 1, 0],
     [0, 1, 1, 1],
     [0, 0, 1, 1],
     [0, 0, 1, 1]
     ]), dtype=numpy.complex128)
R = x.shape[0]
L = x.shape[-1]
# print (x.real)
x = x.flatten()
NX = numpy.full((16,), 0).astype(numpy.complex128)
for i in range(0, 16):
    NX[i] = x[i]
NX = NX.reshape(4, 4)
x = thr.to_device(NX)
X = thr.array((R, L), dtype=numpy.complex128)
fft = FFT(x)
fftc = fft.compile(thr)
fftc(X, x, 0)
xfft = X.get()
print (xfft)
aa = xfft.conjugate()
# print (aa)

xx = thr.to_device(aa)
fft = FFT(xx)
fftc = fft.compile(thr)
fftc(X, xx, 0)
xifft = X.get()
xifft = xifft / (R * L)
thr.release()
print (xifft.real)