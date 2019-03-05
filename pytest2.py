import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
mod = SourceModule("""
#include<cuComplex.h>
__global__ void AHE(cuFloatComplex *a, cuFloatComplex *b,int row)
{
    int i = threadIdx.y + blockDim.y * blockIdx.y;
    int j = threadIdx.x + blockDim.x * blockIdx.x;
    const int idx = i + j*row;
    b[idx]  = a[idx] ;
}
""")
AHE = mod.get_function("AHE")
img =np.random.randn(4, 4).astype(np.complex128)
print (img)
row = np.int32(img.shape[-1])
out = img.copy()
out[:] = 0
out = np.complex128(out)
#col = np.complex128(col)
AHE(cuda.In(img),cuda.InOut(out), row, row, block=(32,32,1),grid=(1,1))
out=out+1j
out=np.angle(out)
print (out)