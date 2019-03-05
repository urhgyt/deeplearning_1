import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.cumath as cu
import numpy
from pycuda import gpuarray

from pycuda.compiler import SourceModule
mod = SourceModule("""
//__global__ void complex(cufftComplex *b, cufftComplex *a)
__global__ void complex(float *b, float *a)
{
  const int i = blockIdx.x*blockDim.x+threadIdx.x;
  const int j = blockIdx.y*blockDim.y+threadIdx.y;
  const int k = j*10+i;
  a[k]=b[k];
}
""")

multiply_them = mod.get_function("complex")

a = gpuarray.zeros((10,10), dtype=numpy.complex64)
b=gpuarray.ones_like(a)
c = numpy.zeros((10,10), dtype=numpy.complex64)
multiply_them(
        b, a,
        block=(10,10,1), grid=(1,2))

#b=cuda.gpuarray_to_array(gpuarray=a, order="C")
#a=numpy.angle(a)
a=a+1j
c=a.get()
c=numpy.angle(c)
#a=numpy.sin(a)
print c

