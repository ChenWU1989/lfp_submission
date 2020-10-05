#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "lfp.h"
#include "cuda.h"
}

__device__ uint8_t fp2lfp(uint32_t src, int bit_width, int exp_width) {
  uint8_t dst;

  uint32_t src_sgn = src & 0x80000000u;
  uint32_t src_exp = src & 0x7F800000u;
  uint32_t src_mts = src & 0x007FFFFFu;

  uint8_t  dst_sgn = (uint8_t)(src_sgn >> (32 - bit_width));
  uint8_t  dst_exp;
  uint8_t  dst_mts;

  int mts_width = bit_width - 1 - exp_width;
  int lfp_exp_bias = exp_width ? (int)(1 << (exp_width - 1)) - 1 : 0;
  int lfp_exp_max = (int)(1 << exp_width) - 1;
  uint8_t lfp_mts_max = (uint8_t)((1 << mts_width) - 1);
  int dst_exp_value = ((int)(src_exp >> 23)) - 127 + lfp_exp_bias;

  if ((src & 0x7FFFFFFFu) == 0 || src_exp == 0) {                             // signed zero or denormalized number
    dst = (uint8_t)(src >> (32 - bit_width)); 
  }
  else {
    if (dst_exp_value > lfp_exp_max) {                                        // overflow to signed "max" value
      dst = ((uint8_t)(0x7F >> (8 - bit_width))) | dst_sgn;
    }
    else if (dst_exp_value <= 0) {                                            // underflow
      if (dst_exp_value < 0 - mts_width) {                                    // Mantissa shifted all the way off
                                                                              // no rounding possibility
        dst_mts = (uint8_t)0u;                                                // Set mantissa to zero
      }
      else {
        src_mts |= 0x00800000u;                                               // Add the hidden leading bit
        dst_mts = (uint8_t)(src_mts >> (24 - mts_width - dst_exp_value));
        if ((src_mts >> (23 - mts_width - dst_exp_value)) & 0x00000001u) {    // Round
          dst_mts += (uint8_t)1u;
        }
      }
      dst = dst_sgn | dst_mts;                                                // Exponent is zero for denormalized number
    }
    else {                                                                    // Normal case
      dst_exp = (uint8_t)(dst_exp_value << mts_width);
      dst_mts = (uint8_t)(src_mts >> (23 - mts_width));
      if (src_mts & (0x00400000u >> mts_width)) {                             // Round  
        if ((dst_exp_value == lfp_exp_max) && (dst_mts == lfp_mts_max))
          dst = ((uint8_t)(0x7F >> (8 - bit_width))) | dst_sgn;
        else
          dst = (dst_sgn | dst_exp | dst_mts) + (uint8_t)1u;
      }
      else {
        dst = (dst_sgn | dst_exp | dst_mts);
      }
    }
  }
  return dst;
}

__device__ uint32_t lfp2fp(uint8_t src, int bit_width, int exp_width)
{
  uint32_t dst;
 
  int mts_width = bit_width - 1 - exp_width;
  int lfp_exp_bias = exp_width ? (int)(1 << (exp_width - 1)) - 1 : 0;
  uint8_t lfp_mts_max = (uint8_t)((1 << mts_width) - 1);
  
  uint8_t src_sgn = src & (0x80u >> (8 - bit_width));
  uint8_t src_exp = src & ((0x7F >> (8 - bit_width)) - lfp_mts_max);
  uint8_t src_mts = src & lfp_mts_max;

  uint32_t dst_sgn = ((uint32_t)src_sgn) << (32 - bit_width);
  uint32_t dst_exp;
  uint32_t dst_mts;
  
  int dst_exp_value;

  if ((src & (0x7F >> (8 - bit_width))) == 0) {                               // Signed zero, return signed zero
    dst = ((uint32_t)src) << (32 - bit_width);
  }
  else {
    int e = -1;
    if (src_exp == 0) {                                                       // Denormalized number
                                                                              // Convert to normalized number in fp32
                                                                              // Find the extra to adjust exponent
      do {
        e++;
        src_mts <<= 1;
      } while ((src_mts & (0x01 << mts_width)) == 0);
    }
    else {                                                                    // Normalized number
      e = 0;
    }
    dst_exp_value = ((int)(src_exp >> mts_width)) - lfp_exp_bias + 127 - e;
    dst_exp = (uint32_t)(dst_exp_value << 23);
    dst_mts = ((uint32_t)(src_mts & lfp_mts_max)) << (23 - mts_width);
    dst = dst_sgn | dst_exp | dst_mts;
  }
  return dst;
}

__global__ void fp2lfp_kernel(uint32_t* fp32, uint8_t* lfp, int n, int bit_width, int exp_width)
{
  int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  if (i < n)
    lfp[i] = fp2lfp(fp32[i], bit_width, exp_width);
}

__global__ void lfp2fp_kernel(uint8_t* lfp, uint32_t* fp32, int n, int bit_width, int exp_width)
{
  int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  if (i < n)
    fp32[i] = lfp2fp(lfp[i], bit_width, exp_width);
}

__device__ float scale_data(float x, float scale)
{
  return x * scale;
}

__global__ void lfp_scale_kernel(float *x, int n, int exp)
{
  int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] = scale_data(x[i], powf(2, exp));
}

extern "C" void lfp_data_gpu(float *x, int n, int exp, QUANTIZE_TYPE type)
{ 
    lfp_scale_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, exp); 
    check_error(cudaPeekAtLastError());
    
    uint32_t *x_32_bin = (uint32_t*)x;
    uint8_t *x_8_bin;
    cudaMalloc(&x_8_bin, sizeof(uint8_t)*n);
    
    int bit_width;
    int exp_width;

    switch (type) {
      // fp8
      case M0E7:  {
        bit_width = 8;
        exp_width = 7;  
        break;
      }
      case M1E6:  {
        bit_width = 8;
        exp_width = 6;
        break;
      }
      case M2E5:  {
        bit_width = 8;
        exp_width = 5;  
        break;
      }
      case M3E4:  {
        bit_width = 8;
        exp_width = 4;  
        break;
      }
      case M4E3:  {
        bit_width = 8;
        exp_width = 3;  
        break;
      }
      case M5E2:  {
        bit_width = 8;
        exp_width = 2;  
        break;
      }
      case M6E1:  {
        bit_width = 8;
        exp_width = 1;  
        break;
      }
      case M7E0:  {
        bit_width = 8;
        exp_width = 0;  
        break;
      }
      // fp7
      case M0E6:  {
        bit_width = 7;
        exp_width = 6;
        break;
      }
      case M1E5:  {
        bit_width = 7;
        exp_width = 5;  
        break;
      }
      case M2E4:  {
        bit_width = 7;
        exp_width = 4;  
        break;
      }
      case M3E3:  {
        bit_width = 7;
        exp_width = 3;  
        break;
      }
      case M4E2:  {
        bit_width = 7;
        exp_width = 2;  
        break;
      }
      case M5E1:  {
        bit_width = 7;
        exp_width = 1;  
        break;
      }
      case M6E0:  {
        bit_width = 7;
        exp_width = 0;  
        break;
      }
      // fp6
      case M0E5:  {
        bit_width = 6;
        exp_width = 5;
        break;
      }
      case M1E4:  {
        bit_width = 6;
        exp_width = 4;
        break;
      }
      case M2E3:  {
        bit_width = 6;
        exp_width = 3;
        break;
      }
      case M3E2:  {
        bit_width = 6;
        exp_width = 2;
        break;
      }
      case M4E1:  {
        bit_width = 6;
        exp_width = 1;
        break;
      }
      case M5E0:  {
        bit_width = 6;
        exp_width = 0;
        break;
      }
      // fp5
      case M0E4:  {
        bit_width = 5;
        exp_width = 4;
        break;
      }
      case M1E3:  {
        bit_width = 5;
        exp_width = 3;
        break;
      }
      case M2E2:  {
        bit_width = 5;
        exp_width = 2;
        break;
      }
      case M3E1:  {
        bit_width = 5;
        exp_width = 1;
        break;
      }
      case M4E0:  {
        bit_width = 5;
        exp_width = 0;
        break;
      }
      // fp4
      case M0E3:  {
        bit_width = 4;
        exp_width = 3;
        break;
      }
      case M1E2:  {
        bit_width = 4;
        exp_width = 2;
        break;
      }
      case M2E1:  {
        bit_width = 4;
        exp_width = 1;
        break;
      }
      case M3E0:  {
        bit_width = 4;
        exp_width = 0;
        break;
      }
      default      :  {
        bit_width = 8;
        exp_width = 0;  
        break;
      }
    }
    
    fp2lfp_kernel<<<cuda_gridsize(n), BLOCK>>>(x_32_bin, x_8_bin, n, bit_width, exp_width);
    check_error(cudaPeekAtLastError());
    lfp2fp_kernel<<<cuda_gridsize(n), BLOCK>>>(x_8_bin, x_32_bin, n, bit_width, exp_width);
    check_error(cudaPeekAtLastError());
    x = (float*)x_32_bin;
    
    cudaFree(x_8_bin);
}

extern "C" void lfp_de_data_gpu(float *x, int n, int exp)
{   
    lfp_scale_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, -exp);
    check_error(cudaPeekAtLastError());
}

__global__ void lfp_normalize_kernel(float *x, float *scales, float *biases, int filters, int spatial, size_t n)
{
  int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int f = (i / spatial) % filters;
  x[i] = scales[f] * x[i] + biases[f];
}

extern "C" void lfp_normalize_gpu(float *x, float *scales, float *biases, int batch, int filters, int spatial)
{
  size_t n = batch * filters * spatial;
  lfp_normalize_kernel<<<cuda_gridsize(n), BLOCK>>>(x, scales, biases, filters, spatial, n);
  check_error(cudaPeekAtLastError());
}
