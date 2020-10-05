#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "cuda.h"
#include "norm.h"
}

__global__ void norm_kernel(float* x, int n, int c, int fmsz, float* mean, float* var)
{
  int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  int index_c = (i / fmsz) % c;
  if (i < n)
    x[i] = (x[i] - mean[index_c]) / var[index_c];
}

__global__ void denorm_kernel(float* x, int n, int c, int fmsz, float* mean, float* var)
{
  int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  int index_c = (i / fmsz) % c;
  if (i < n)
    x[i] = x[i] * var[index_c] + mean[index_c];
}

extern "C" void norm_fm_gpu(float* x, int c, int fmsz, float* mean, float* var)
{
  int n = c * fmsz;
  norm_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, c, fmsz, mean, var);
  check_error(cudaPeekAtLastError());
}

extern "C" void denorm_fm_gpu(float* x, int c, int fmsz, float* mean, float* var)
{
  int n = c * fmsz;
  denorm_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, c, fmsz, mean, var);
  check_error(cudaPeekAtLastError());
}

