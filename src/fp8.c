/***************************************************************
The convertion between 8bit floating point and 32bit floating
point numbers refers to (1) IEEE754 floating point format;
(2) http://www.toves.org/books/float/;
(3) https://github.com/hglm/detex/blob/master/half-float.c
***************************************************************/
#include "fp8.h"
#include "cuda.h"
#include <float.h>
#include <math.h>

uint8_t fp2lfp(uint32_t src, int bit_width, int exp_width) {
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
  printf("bias = %d, exp_max = %d, lfp_mts_max = %d, dst_exp_value = %d\n", lfp_exp_bias, lfp_exp_max, lfp_mts_max, dst_exp_value);

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


uint32_t lfp2fp(uint8_t src, int bit_width, int exp_width)
{
  uint32_t dst;
 
  int mts_width = bit_width - 1 - exp_width;
  int lfp_exp_bias = exp_width ? (int)(1 << (exp_width - 1)) - 1 : 0;
  uint8_t lfp_mts_max = (uint8_t)((1 << mts_width) - 1);
  printf("bias = %d, mts_max = %x\n", lfp_exp_bias, lfp_mts_max);

  uint8_t src_sgn = src & (0x80u >> (8 - bit_width));
  uint8_t src_exp = src & ((0x7F >> (8 - bit_width)) - lfp_mts_max);
  uint8_t src_mts = src & lfp_mts_max;
  printf("sgn = %x, exp = %x, mts = %x\n", src_sgn, src_exp, src_mts);

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
	printf("dst_exp_value: %d, dst_exp: %x, dst_mts: %x\n", dst_exp_value, dst_exp, dst_mts);
    dst = dst_sgn | dst_exp | dst_mts;
  }
  return dst;
}

uint8_t fp32_2_fp8(uint32_t src, int exp_width) {
  uint8_t dst;

  uint32_t src_sgn = src & 0x80000000u;
  uint32_t src_exp = src & 0x7F800000u;
  uint32_t src_mts = src & 0x007FFFFFu;

  uint8_t  dst_sgn = (uint8_t)(src_sgn >> 24);
  uint8_t  dst_exp;
  uint8_t  dst_mts;

  int mts_width = 7 - exp_width;
  int fp8_exp_bias = exp_width ? (int)(1 << (exp_width - 1)) - 1 : 0;
  int fp8_exp_max = (int)(1 << exp_width) - 1;
  uint8_t fp8_mts_max = (uint8_t)((1 << mts_width) - 1);
  int dst_exp_value = ((int)(src_exp >> 23)) - 127 + fp8_exp_bias;

  if ((src & 0x7FFFFFFFu) == 0 || src_exp == 0) {                             // signed zero or denormalized number
    dst = (uint8_t)(src >> 24); 
  }
  else {
    if (dst_exp_value > fp8_exp_max) {                                        // overflow to signed "max" value
      dst = ((uint8_t)0x7F) | dst_sgn;
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
        if ((dst_exp_value == fp8_exp_max) && (dst_mts == fp8_mts_max))
          dst = ((uint8_t)0x7F) | dst_sgn;
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

uint8_t fp32_2_fp6(uint32_t src, int exp_width) {
  uint8_t dst;

  uint32_t src_sgn = src & 0x80000000u;
  uint32_t src_exp = src & 0x7F800000u;
  uint32_t src_mts = src & 0x007FFFFFu;

  uint8_t  dst_sgn = (uint8_t)(src_sgn >> 26);
  uint8_t  dst_exp;
  uint8_t  dst_mts;

  int mts_width = 5 - exp_width;
  int fp6_exp_bias = exp_width ? (int)(1 << (exp_width - 1)) - 1 : 0;
  int fp6_exp_max = (int)(1 << exp_width) - 1;
  uint8_t fp6_mts_max = (uint8_t)((1 << mts_width) - 1);
  int dst_exp_value = ((int)(src_exp >> 23)) - 127 + fp6_exp_bias;

  if ((src & 0x7FFFFFFFu) == 0 || src_exp == 0) {                             // signed zero or denormalized number
    dst = (uint8_t)(src >> 26); 
  }
  else {
    if (dst_exp_value > fp6_exp_max) {                                        // overflow to signed "max" value
      dst = ((uint8_t)0x1F) | dst_sgn;
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
        if ((dst_exp_value == fp6_exp_max) && (dst_mts == fp6_mts_max))
          dst = ((uint8_t)0x1F) | dst_sgn;
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

uint32_t fp8_2_fp32(uint8_t src, int exp_width)
{
  uint32_t dst;
 
  int mts_width = 7 - exp_width;
  int fp8_exp_bias = exp_width ? (int)(1 << (exp_width - 1)) - 1 : 0;
  uint8_t fp8_mts_max = (uint8_t)((1 << mts_width) - 1);
  
  uint8_t src_sgn = src & 0x80u;
  uint8_t src_exp = src & (0x7F - fp8_mts_max);
  uint8_t src_mts = src & fp8_mts_max;

  uint32_t dst_sgn = ((uint32_t)src_sgn) << 24;
  uint32_t dst_exp;
  uint32_t dst_mts;
  
  int dst_exp_value;

  if ((src & 0x7F) == 0) {                                                    // Signed zero, return signed zero
    dst = ((uint32_t)src) << 24;
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
    dst_exp_value = ((int)(src_exp >> mts_width)) - fp8_exp_bias + 127 - e;
    dst_exp = (uint32_t)(dst_exp_value << 23);
    dst_mts = ((uint32_t)(src_mts & fp8_mts_max)) << (23 - mts_width);
    dst = dst_sgn | dst_exp | dst_mts;
  }
  return dst;
}

uint32_t fp6_2_fp32(uint8_t src, int exp_width)
{
  uint32_t dst;
 
  int mts_width = 5 - exp_width;
  int fp6_exp_bias = exp_width ? (int)(1 << (exp_width - 1)) - 1 : 0;
  uint8_t fp6_mts_max = (uint8_t)((1 << mts_width) - 1);
  
  uint8_t src_sgn = src & 0x20u;
  uint8_t src_exp = src & (0x1F - fp6_mts_max);
  uint8_t src_mts = src & fp6_mts_max;

  uint32_t dst_sgn = ((uint32_t)src_sgn) << 26;
  uint32_t dst_exp;
  uint32_t dst_mts;
  
  int dst_exp_value;

  if ((src & 0x1F) == 0) {                                                    // Signed zero, return signed zero
    dst = ((uint32_t)src) << 26;
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
    dst_exp_value = ((int)(src_exp >> mts_width)) - fp6_exp_bias + 127 - e;
    dst_exp = (uint32_t)(dst_exp_value << 23);
    dst_mts = ((uint32_t)(src_mts & fp6_mts_max)) << (23 - mts_width);
    dst = dst_sgn | dst_exp | dst_mts;
  }
  return dst;
}

uint16_t fp32_2_fp16(uint32_t src) {
  uint16_t dst;

  uint32_t src_sgn = src & 0x80000000u;
  uint32_t src_exp = src & 0x7f800000u;
  uint32_t src_mts = src & 0x007fffffu;

  uint16_t dst_sgn = (uint16_t)(src_sgn >> 16);
  uint16_t dst_exp;
  uint16_t dst_mts;
  int dst_exp_value = ((int)(src_exp >> 23)) - 127 + 15;

  if ((src & 0x7FFFFFFFu) == 0 || src_exp == 0) {                             // Signed zero or denormalized number
    dst = (uint16_t)(src >> 16);  
  }
  else {
    if (dst_exp_value > 31) {                                                 // Overflow to "max" number
      dst_exp = (uint16_t)0x7C00u;
      dst_mts = (uint16_t)0x03FFu;
      dst = dst_sgn | dst_exp | dst_mts;
    }
    else if (dst_exp_value <= 0) {                                            // underflow
      if (dst_exp_value < -10) {                                              // Mantissa shifted all the way off
                                                                              // no rounding possibility
        dst_mts = (uint16_t)0u;                                               // Set mantissa to zero
      }
      else {
        src_mts |= 0x00800000u;                                               // Add the hidden leading bit
        dst_mts = (uint16_t)(src_mts >> (14 - dst_exp_value));
        if ((src_mts >> (13 - dst_exp_value)) & 0x00000001u) {                // Round
          dst_mts += (uint16_t)1u;
         }
      }
      dst = dst_sgn | dst_mts;                                                // Exponent is zero for denormalized number
    }
    else {
      dst_exp = (uint16_t)(dst_exp_value << 10);
      dst_mts = (uint16_t)(src_mts >> 13);
      if (src_mts & 0x00001000u)                                              // Round
        if ((dst_exp_value == 31) && (dst_mts == 0x03FFu))
          dst = (uint16_t)0x03FFu | dst_sgn;
        else
          dst = (dst_sgn | dst_exp | dst_mts) + (uint16_t)1u;
      else
        dst = (dst_sgn | dst_exp | dst_mts);
    }
  }
  return dst;
}

uint32_t fp16_2_fp32(uint16_t src)
{
  uint32_t dst;

  uint16_t src_sgn = src & 0x8000u;
  uint16_t src_exp = src & 0x7C00u;
  uint16_t src_mts = src & 0x03FFu;

  uint32_t dst_sgn = ((uint32_t)src_sgn) << 16;
  uint32_t dst_exp;
  uint32_t dst_mts;
  int dst_exp_value;

  if ((src & 0x7FFF) == 0) {                                                  // Signed zero
    dst = ((uint32_t)src) << 16;                                              // Return signed zero
  }
  else {
    int e = -1; 
    if (src_exp == 0) {                                                       // Denormalized number
                                                                              // Convert to normalized number in fl32
                                                                              // Find the extra to adjust exponent
      do {
        e++;
        src_mts <<= 1;
      } while ((src_mts & 0x0400) == 0);
    }
    else {
      e = 0;
    }
      dst_exp_value = ((int)(src_exp >> 10)) - 15 + 127 - e;
      dst_exp = (uint32_t)(dst_exp_value << 23);
      dst_mts = ((uint32_t)(src_mts & 0x03FFu)) << 13;
      dst = dst_sgn | dst_exp | dst_mts;
  }
  return dst;
}

float fp8_err(float *fp32, int n, int exp, QUANTIZE_TYPE type)
{
  float *fp8 = calloc(n, sizeof(float));
  memcpy(fp8, fp32, sizeof(float)*n);

  fp8_data(fp8, n, exp, type);

	float err = 0;
  float scale = powf(2, exp);
	for (int i = 0; i < n; i++)
		err += ((fp8[i] / scale - fp32[i]) * (fp8[i] / scale - fp32[i]));
	err /= n;

	free(fp8);
	fp8 = NULL;
  return err;
}

float fp8_err_rate(float *fp32, int n, int exp, QUANTIZE_TYPE type)
{
  float *fp8 = calloc(n, sizeof(float));
  memcpy(fp8, fp32, sizeof(float)*n);

  fp8_data(fp8, n, exp, type);

	float err = 0;
  float scale = powf(2, exp);
	for (int i = 0; i < n; i++)
		err += fabs(fp8[i] / scale - fp32[i]) / fabs(fp32[i]);
	err /= n;

	free(fp8);
	fp8 = NULL;
  return err;
}

int fp8_decision(float *x, int n, QUANTIZE_TYPE* type, bool multi)
{
	int exp = 0;
	float max_value = -FLT_MAX;
  float min_err = FLT_MAX;

	for (int i = 0; i < n; i++)
		if (max_value < fabs(x[i]))
			max_value = fabs(x[i]);

	int base = log2f(1 / max_value);
  
  if ( multi ) {
    QUANTIZE_TYPE quan_type;
    for (quan_type = 4; quan_type < 8; quan_type++) {
      for (int i = base - 5; i < base + 6; i++) {
        float err = fp8_err(x, n, i, quan_type);
        if (min_err > err) {
          min_err = err;
          exp = i;
          *type = quan_type;
        }
      } 
    }
  }
  else {
	  for (int i = base - 5; i < base + 6; i++) {
	  	float err = fp8_err(x, n, i, *type);
	  	if (min_err > err) {
	  		min_err = err;
	  		exp = i;
	  	}
      //printf("err @ %d = %3.12f\n", i, err);
	  }
  }
  //printf("min_err = %3.12f\n", min_err);
	return exp;
}

// Decide the uniform scale for all the output feature map, as
// we do normalization first. The process of the decision is:
// 1) Find the optimal scale of each layer;
// 2) Record the average, average - 1, and average + 1 as candidate;
// 3) Find the optimal uniform among the three, using the increment
//    of the MSE as the metric.
int fp8_oexp_decision(network *net)
{
  int* oexp_candidate = calloc(3, sizeof(int));
  int* oexp_tmp = calloc(net->n, sizeof(int));
  int i = 0;
  FILE* fp;
  
  // find the optimal scale
  for (i = 0; i < net->n; i++) {
    layer l = net->layers[i];
    if (l.type == SOFTMAX)
      break;

    oexp_tmp[i] = fp8_decision(l.norm_output, l.outputs * net->norm_batch, &net->quantize->type, true);
    oexp_candidate[0] += oexp_tmp[i];
  }
  
  // consider the uniform scale to be average, average-1 and average+1
  oexp_candidate[0] = (int)round((float)oexp_candidate[0] / (float)i);
  oexp_candidate[1] = oexp_candidate[0] - 1;
  oexp_candidate[2] = oexp_candidate[0] + 1;
  
  // find the optimal uniform scale
  float* mse_incr = calloc(3, sizeof(float));

  // FILE *f_err = fopen("oexp_err.txt", "w");
  // fprintf(f_err, "%d, %d, %d\n", oexp_candidate[0], oexp_candidate[1], oexp_candidate[2]);
  for (i = 0; i < net->n; i++) {
    layer l = net->layers[i];
    if (l.type == SOFTMAX)
      break;
    
    float err_ori = fp8_err(l.norm_output, l.outputs * net->norm_batch, oexp_tmp[i], l.fp8->type);
    float err_0   = fp8_err(l.norm_output, l.outputs * net->norm_batch, oexp_candidate[0], l.fp8->type);
    float err_1   = fp8_err(l.norm_output, l.outputs * net->norm_batch, oexp_candidate[1], l.fp8->type);
    float err_2   = fp8_err(l.norm_output, l.outputs * net->norm_batch, oexp_candidate[2], l.fp8->type);
    
    free(l.norm_output);

    mse_incr[0] += fabs(err_0 - err_ori);
    mse_incr[1] += fabs(err_1 - err_ori);
    mse_incr[2] += fabs(err_2 - err_ori);

    // for analysis
    // fprintf(f_err, "%d, %4.10f, %4.10f, %4.10f, %4.10f\n", i, err_ori, err_0, err_1, err_2);
  }
  // fclose(f_err);
  
  int optimal_oexp;
  if (mse_incr[0] <= mse_incr[1] && mse_incr[0] <= mse_incr[2])
    optimal_oexp = oexp_candidate[0];
  else if (mse_incr[1] <= mse_incr[0] && mse_incr[1] <= mse_incr[2])
    optimal_oexp = oexp_candidate[1];
  else 
    optimal_oexp = oexp_candidate[2];

  free(oexp_candidate);
  free(oexp_tmp);
  free(mse_incr);
  oexp_candidate = NULL;
  oexp_tmp = NULL;
  mse_incr = NULL;

  return optimal_oexp;
}

bool uniform_oexp(network *net)
{
  for (int i = 0; i < net->n; i++) {
    if (net->layers[i].type == SHORTCUT || net->layers[i].type == ROUTE)
      return true;
  }

  return false;
}

void fp8_network_decision(network *net)
{
  QUANTIZE_TYPE type = net->quantize->type;
  
  net->quantize->iexp = fp8_decision(net->norm_input, net->inputs * net->norm_batch, &net->quantize->type, true);
  
  // We can choose a uniform oexp for each layer, as we do normalization first, however,
  // we will still have an accuracy loss. If the network do not need a uniform oexp,
  // we will choose the optimal for each layer.
  bool uniform_or_not = uniform_oexp(net);
  int optimal_oexp = uniform_or_not ? fp8_oexp_decision(net) : 0;
  
  for (int i = 0; i < net->n; i++) {
    layer* l = net->layers + i;
    
    //printf("Decision for output feature map.\n");
    if (uniform_or_not)
      l->fp8->oexp = optimal_oexp;
    else {
      l->fp8->oexp = fp8_decision(l->norm_output, l->outputs * net->norm_batch, &l->fp8->type, true);
    }

    //printf("Decision for input feature map.\n");
    if (i == 0)
      l->fp8->iexp = net->quantize->iexp;
    else
      l->fp8->iexp = net->layers[i - 1].fp8->oexp;
    
    if (l->type == CONVOLUTIONAL || l->type == CONNECTED) {
      l->fp8->wexp = fp8_decision(l->weights, l->nweights, &l->fp8->type, true);
      if (l->batch_normalize)
        l->fp8->bn_sexp = fp8_decision(l->bn_scales, l->out_c, &l->fp8->type, true);
      else
        l->fp8->bn_sexp = 0;
    }
    else if (l->type == CROP) {
      l->fp8->wexp = fp8_decision(l->crop_scale, l->c, &l->fp8->type, true);
      l->fp8->bn_sexp = 0;
    }
    else if (l->type == SHORTCUT) {
      float* sc_coeff = calloc(l->out_c + l->c, sizeof(float));
      memcpy(sc_coeff, l->norm_sc_coeff0, sizeof(float)*l->out_c);
      memcpy(sc_coeff+l->out_c, l->norm_sc_coeff1, sizeof(float)*l->c);
      l->fp8->wexp = fp8_decision(sc_coeff, l->out_c + l->c, &l->fp8->type, true);
      l->fp8->bn_sexp = 0;

      free(sc_coeff);
    }
    else if (l->type == BATCHNORM) {
      l->fp8->wexp = 0;
      l->fp8->bn_sexp = fp8_decision(l->bn_scales, l->out_c, &l->fp8->type, true);  
    }
    else {
      l->fp8->wexp = 0;
      l->fp8->bn_sexp = 0;
    }
    
    //printf("Decision for bias and bn bias.\n");
    l->fp8->bexp = l->fp8->iexp + l->fp8->wexp;
    l->fp8->bn_bexp = l->fp8->bexp + l->fp8->bn_sexp;
    
    //printf("Decision for cutting from FL32 to FP8.\n");
    l->fp8->offset = l->fp8->bn_bexp - l->fp8->oexp;
    
    printf("Layer [ %3d ], iexp = %3d, wexp = %3d, bexp = %3d, oexp = %3d, offset = %3d, type = %3d\n",
            i, l->fp8->iexp, l->fp8->wexp, l->fp8->bexp, l->fp8->oexp, l->fp8->offset, l->fp8->type);
    if (l->type == BATCHNORM || l->batch_normalize == 1)
      printf("batch normalization: bn_sexp = %3d, bn_bexp = %3d\n", l->fp8->bn_sexp, l->fp8->bn_bexp);
  }
  
  for (int i = 0; i < net->n; i++) {
    float w_err = 0;
    switch (net->layers[i].type) {
      case CONVOLUTIONAL:
        w_err = fp8_err(net->layers[i].weights, net->layers[i].nweights, net->layers[i].fp8->wexp, net->layers[i].fp8->type);
        // w_err = fp8_err_rate(net->layers[i].weights, net->layers[i].nweights, net->layers[i].fp8->wexp, net->layers[i].fp8->type);
        break;
      case CONNECTED:
        w_err = fp8_err(net->layers[i].weights, net->layers[i].outputs * net->layers[i].inputs, net->layers[i].fp8->wexp, net->layers[i].fp8->type);
        // w_err = fp8_err_rate(net->layers[i].weights, net->layers[i].outputs * net->layers[i].inputs, net->layers[i].fp8->wexp, net->layers[i].fp8->type);
        break;
      default:
        w_err = 0;
        break;
    }
    printf("Layer [ %3d ], weights mse rate: %4.10f\n", i, w_err);
  }

  net->quantize->oexp = net->layers[net->n - 1].fp8->oexp;
}

// convert the 32-bit floating point data to 6-bit floating point data
void fp6_data(float *fp32, int n, int exp, QUANTIZE_TYPE type)
{
  float *fp32_tmp = calloc(n, sizeof(float));
  for (int i = 0; i < n; i++)
    fp32_tmp[i] = fp32[i] * powf(2, exp);

  uint32_t *fp32_bin = (uint32_t*)fp32_tmp;
  uint8_t *fp6_bin = calloc(n, sizeof(uint8_t));

  int exp_width;
  switch (type) {
    case FP6_M0E5:  exp_width = 5;  break;
    case FP6_M1E4:  exp_width = 4;  break;
    case FP6_M2E3:  exp_width = 3;  break;
    case FP6_M3E2:  exp_width = 2;  break;
    case FP6_M4E1:  exp_width = 1;  break;
    case FP6_M5E0:  exp_width = 0;  break;
    default      :  exp_width = 0;  break;
  }

  // cut data from fp32 to fp6
  for (int i = 0; i < n; i++)
    fp6_bin[i] = fp32_2_fp6(fp32_bin[i], exp_width);

  // represent fp6 with fl32, as we will use fl32 to do calculation
  for (int i = 0; i < n; i++)
    fp32_bin[i] = fp6_2_fp32(fp6_bin[i], exp_width);
  
  memcpy(fp32, fp32_bin, sizeof(float)*n);
  free(fp32_tmp);
  free(fp6_bin);
  fp32_tmp = NULL;
  fp6_bin = NULL;
}

void fp8_data(float *fp32, int n, int exp, QUANTIZE_TYPE type)
{
  float *fp32_tmp = calloc(n, sizeof(float));
  for (int i = 0; i < n; i++)
    fp32_tmp[i] = fp32[i] * powf(2, exp);

  uint32_t *fp32_bin = (uint32_t*)fp32_tmp;
  uint8_t *fp8_bin = calloc(n, sizeof(uint8_t));

  int exp_width;

  switch (type) {
    case FP8_M0E7:  exp_width = 7;  break;
    case FP8_M1E6:  exp_width = 6;  break;
    case FP8_M2E5:  exp_width = 5;  break;
    case FP8_M3E4:  exp_width = 4;  break;
    case FP8_M4E3:  exp_width = 3;  break;
    case FP8_M5E2:  exp_width = 2;  break;
    case FP8_M6E1:  exp_width = 1;  break;
    case FP8_M7E0:  exp_width = 0;  break;
    default      :  exp_width = 0;  break;
  }

  // cut data from fl32 to fp8
  for (int i = 0; i < n; i++)
    fp8_bin[i] = fp32_2_fp8(fp32_bin[i], exp_width);

  // represent fp8 with fl32, as we will use fl32 to do calculation
  for (int i = 0; i < n; i++)
    fp32_bin[i] = fp8_2_fp32(fp8_bin[i], exp_width);
  
  memcpy(fp32, fp32_bin, sizeof(float)*n);
  free(fp32_tmp);
  free(fp8_bin);
  fp32_tmp = NULL;
  fp8_bin = NULL;
}

void fp16_data(float *fp32, int n, int exp)
{
  float *fp32_tmp = calloc(n, sizeof(float));
  for (int i = 0; i < n; i++)
    fp32_tmp[i] = fp32[i] * pow(2, exp);

  uint32_t *fp32_bin = (uint32_t*)fp32_tmp;
  uint16_t *fp16_bin = calloc(n, sizeof(uint16_t));

  // cut data from fl32 to fp16
  for (int i = 0; i < n; i++)
    fp16_bin[i] = fp32_2_fp16(fp32_bin[i]);
  
  // represent fp16 with fp32
  for (int i = 0; i < n; i++)
    fp32_bin[i] = fp16_2_fp32(fp16_bin[i]);

  memcpy(fp32, fp32_bin, sizeof(float)*n);
  free(fp32_tmp);
  free(fp16_bin);
  fp32_tmp = NULL;
  fp16_bin = NULL;
}

float* merge_bn_scales(float* variance, float* scales, int filters)
{
  float *bn_scales = calloc(filters, sizeof(float));

  for (int f = 0; f < filters; f++) {
    bn_scales[f] = scales[f] / (sqrt(variance[f]) + 0.000001f);
  }

  return bn_scales;
}

float* merge_bn_biases(float* mean, float* variance, float* scales, float* biases, int filters)
{
  float *bn_biases = calloc(filters, sizeof(float));

  for (int f = 0; f < filters; f++) {
    bn_biases[f] = biases[f] - scales[f] * mean[f] / (sqrt(variance[f]) + 0.000001f);
  }

  return bn_biases;
}

void fp8_quantize(network* net, QUANTIZE_TYPE type)
{
  net->quantize->type = type;
  
  // decide the exp for weights, bias, output feature
  printf("Quantization Decision...\n");
  fp8_network_decision(net);
  printf("Done.\n");
 
  // quantize weights
  printf("Quantize weights & bias...");
  for (int i = 0; i < net->n; i++) {
    layer* l = net->layers + i;
    if (l->type == CONVOLUTIONAL || l->type == CONNECTED) {
#ifdef FP8_DEBUG
      fp8_de_data(l->weights, l->nweights, -l->fp8->wexp);
      fp8_de_data(l->biases, l->nbiases, -l->fp8->bexp);
      if (l->batch_normalize) {
        fp8_de_data(l->bn_scales, l->out_c, -l->fp8->bn_sexp);
        fp8_de_data(l->bn_biases, l->out_c, -l->fp8->bn_bexp);
      }
#else
      fp8_data(l->weights, l->nweights, l->fp8->wexp, l->fp8->type);
      fp16_data(l->biases, l->nbiases, l->fp8->bexp);
      if (l->batch_normalize) {
        fp8_data(l->bn_scales, l->out_c, l->fp8->bn_sexp, l->fp8->type);
        fp16_data(l->bn_biases, l->out_c, l->fp8->bn_bexp);
      }
#endif
    }
    else if (l->type == CROP) {       // the crop layer is considered as image preprocess
                                      // we will quantize once after all the preprocess, 
                                      // not quantize between the preprocess several times.
      fp8_de_data(l->crop_scale, l->c, -l->fp8->wexp);
      fp8_de_data(l->crop_trans, l->c, -l->fp8->bexp);
    }
    else if (l->type == SHORTCUT) {
#ifdef FP8_DEBUG
      fp8_de_data(l->norm_sc_coeff0, l->out_c, -l->fp8->wexp);
      fp8_de_data(l->norm_sc_coeff1, l->c, -l->fp8->wexp);
      fp8_de_data(l->norm_sc_bias, l->out_c, -l->fp8->bexp);
#else
      fp8_data(l->norm_sc_coeff0, l->out_c, l->fp8->wexp, l->fp8->type);
      fp8_data(l->norm_sc_coeff1, l->c, l->fp8->wexp, l->fp8->type);
      fp16_data(l->norm_sc_bias, l->out_c, l->fp8->bexp);
#endif
    }
    else if (l->type == BATCHNORM) {
#ifdef FP8_DEBUG
      fp8_de_data(l->bn_scales, l->out_c, -l->fp8->bn_sexp);
      fp8_de_data(l->bn_biases, l->out_c, -l->fp8->bn_bexp);
#else
      fp8_data(l->bn_scales, l->out_c, l->fp8->bn_sexp, l->fp8->type);
      fp16_data(l->bn_biases, l->out_c, l->fp8->bn_bexp);
#endif
    }
  }
  printf("done\n");
}

void fp8_de_data(float *x, int n, int exp)
{
  float scale = powf(2, exp);
  for (int i = 0; i < n; i++)
    x[i] /= scale;
}

void fp8_normalize_cpu(float *x, float *scales, float *biases, int batch, int filters, int spatial)
{
  int b, f, i;
  for(b = 0; b < batch; ++b){
    for(f = 0; f < filters; ++f){
      for(i = 0; i < spatial; ++i){
        int index = b*filters*spatial + f*spatial + i;
        x[index] = scales[f] * x[index] + biases[f];
      }
    }
  }
}

void save_quantize_decision(network* net, char* filename)
{
  FILE *fp = fopen(filename, "w");
  if (fp == NULL) {
    printf("Cannot open file %s.\n", filename);
    return;
  }

  printf("Saving quantization decision to %s...", filename);
  // save quantization decision
  fprintf(fp, "%d, %d\n", net->quantize->iexp, net->quantize->oexp);
  for (int i = 0; i < net->n; i++) {
    layer l = net->layers[i];
    fprintf(fp, "%3d, %3d, %3d, %3d, %3d, %3d, %3d\n", l.fp8->iexp, l.fp8->wexp, l.fp8->bexp, l.fp8->oexp, l.fp8->bn_sexp, l.fp8->bn_bexp, l.fp8->offset);
  }
  printf("done.\n");
  
  fclose(fp);
}

void load_quantize_decision(network* net, char* filename)
{
  FILE *fp = fopen(filename, "r");
  if (fp == NULL) {
    printf("Cannot open file %s.\n", filename);
    return;
  }
  
  printf("Loading quantization decision from %s ...", filename);
  fscanf(fp, "%d, %d\n", &net->quantize->iexp, &net->quantize->oexp);
  for (int i = 0; i < net->n; i++) {
    layer l = net->layers[i];
    fscanf(fp, "%d, %d, %d, %d, %d, %d, %d\n", &l.fp8->iexp, &l.fp8->wexp, &l.fp8->bexp, &l.fp8->oexp, &l.fp8->bn_sexp, &l.fp8->bn_bexp, &l.fp8->offset);
  }
  printf("done.\n");
  
  fclose(fp);
}

// Saving quantized weights into binary file.
// It includes quantization decisions, normalization parameters
// and all the quantized weights, biases.
// For each layer:
// quantization decision (iexp, wep, bexp, oexp, bn_sexp, bn_bexp, offset)
// normalization decision (mean, var)
// quantized weights, biases.
void save_quantize_weights(network* net, char* filename)
{
  FILE *fp = fopen(filename, "wb");
  if (fp == NULL) {
    printf("Cannot open file %s.\n", filename);
    return;
  }
  
  printf("Saving quantized weights to %s...", filename);
  
  int major = 0;
  int minor = 2;
  int revision = 0;
  fwrite(&major, sizeof(int), 1, fp);
  fwrite(&minor, sizeof(int), 1, fp);
  fwrite(&revision, sizeof(int), 1, fp);
  fwrite(net->seen, sizeof(size_t), 1, fp);
  
  // net quantization decision
  fwrite(&net->quantize->iexp, sizeof(int), 1, fp);
  fwrite(&net->quantize->oexp, sizeof(int), 1, fp);

  // net normalization parameters
  fwrite(net->norm_in_mean, sizeof(float), net->c, fp);
  fwrite(net->norm_in_var, sizeof(float), net->c, fp);
  fwrite(net->norm_out_mean, sizeof(float), net->layers[net->n-1].out_c, fp);
  fwrite(net->norm_out_var, sizeof(float), net->layers[net->n-1].out_c, fp);

  for (int i = 0; i < net->n; i++) {
    layer l = net->layers[i];
    
    // layer quantization decision
    fwrite(&l.fp8->iexp, sizeof(int), 1, fp);
    fwrite(&l.fp8->wexp, sizeof(int), 1, fp);
    fwrite(&l.fp8->bexp, sizeof(int), 1, fp);
    fwrite(&l.fp8->oexp, sizeof(int), 1, fp);
    fwrite(&l.fp8->bn_sexp, sizeof(int), 1, fp);
    fwrite(&l.fp8->bn_bexp, sizeof(int), 1, fp);
    fwrite(&l.fp8->offset, sizeof(int), 1, fp);

    // layer normalization parameters
    int in_num = 0;
    if (l.type == SHORTCUT)
      in_num = l.c + l.out_c;
    else
      in_num = l.c;
    
    fwrite(l.norm_in_mean, sizeof(float), in_num, fp);
    fwrite(l.norm_in_var, sizeof(float), in_num, fp);
    fwrite(l.norm_out_mean, sizeof(float), l.out_c, fp);
    fwrite(l.norm_out_var, sizeof(float), l.out_c, fp);

    // layer weights & biases
    if (l.type == CONVOLUTIONAL || l.type == CONNECTED) {
      fwrite(l.biases, sizeof(float), l.nbiases, fp);
      fwrite(l.weights, sizeof(float), l.nweights, fp);
      if (l.batch_normalize) {
        fwrite(l.bn_scales, sizeof(float), l.nbiases, fp);
        fwrite(l.bn_biases, sizeof(float), l.nbiases, fp);
      }
    }
    else if (l.type == CROP) {
      fwrite(l.crop_scale, sizeof(float), l.c, fp);
      fwrite(l.crop_trans, sizeof(float), l.c, fp);
    }
    else if (l.type == SHORTCUT) {
      fwrite(l.norm_sc_coeff0, sizeof(float), l.out_c, fp);
      fwrite(l.norm_sc_coeff1, sizeof(float), l.c, fp);
      fwrite(l.norm_sc_bias, sizeof(float), l.out_c, fp);
    }
    else if (l.type == BATCHNORM) {
      fwrite(l.bn_scales, sizeof(float), l.c, fp);
      fwrite(l.bn_biases, sizeof(float), l.c, fp);
    }
  }
  printf("done.\n");
  fclose(fp);
}

void load_quantize_weights(network* net, char* filename)
{
#ifdef GPU
  if(net->gpu_index >= 0){
    cuda_set_device(net->gpu_index);
  }
#endif

  FILE *fp = fopen(filename, "rb");
  if (fp == NULL) {
    printf("Cannot open file %s ...", filename);
    return;
  }
  
  printf("Loading weights from %s ...", filename);
  
  int major;
  int minor;
  int revision;
  fread(&major, sizeof(int), 1, fp);
  fread(&minor, sizeof(int), 1, fp);
  fread(&revision, sizeof(int), 1, fp);
  fread(net->seen, sizeof(size_t), 1, fp);
  
  // layer quantization decision
  fread(&net->quantize->iexp, sizeof(int), 1, fp);
  fread(&net->quantize->oexp, sizeof(int), 1, fp);
  
  // layer normalization parameters
  fread(net->norm_in_mean, sizeof(float), net->c, fp);
  fread(net->norm_in_var, sizeof(float), net->c, fp);
  fread(net->norm_out_mean, sizeof(float), net->layers[net->n-1].out_c, fp);
  fread(net->norm_out_var, sizeof(float), net->layers[net->n-1].out_c, fp);

#ifdef GPU
  if (gpu_index >= 0) {
    cuda_push_array(net->norm_in_mean_gpu, net->norm_in_mean, net->c);
    cuda_push_array(net->norm_in_var_gpu, net->norm_in_var, net->c);
    cuda_push_array(net->norm_out_mean_gpu, net->norm_out_mean, net->layers[net->n-1].out_c);
    cuda_push_array(net->norm_out_var_gpu, net->norm_out_var, net->layers[net->n-1].out_c);
  }
#endif

  for (int i = 0; i < net->n; i++) {
    layer l = net->layers[i];
    
    // layer quantization decision
    fread(&l.fp8->iexp, sizeof(int), 1, fp);
    fread(&l.fp8->wexp, sizeof(int), 1, fp);
    fread(&l.fp8->bexp, sizeof(int), 1, fp);
    fread(&l.fp8->oexp, sizeof(int), 1, fp);
    fread(&l.fp8->bn_sexp, sizeof(int), 1, fp);
    fread(&l.fp8->bn_bexp, sizeof(int), 1, fp);
    fread(&l.fp8->offset, sizeof(int), 1, fp);

    // layer normalization parameters
    int in_num = 0;
    if (l.type == SHORTCUT)
      in_num = l.c + l.out_c;
    else
      in_num = l.c;
    
    fread(l.norm_in_mean, sizeof(float), in_num, fp);
    fread(l.norm_in_var, sizeof(float), in_num, fp);
    fread(l.norm_out_mean, sizeof(float), l.out_c, fp);
    fread(l.norm_out_var, sizeof(float), l.out_c, fp);
    
    // layer weights & biases
    if (l.type == CONVOLUTIONAL || l.type == CONNECTED) {
      fread(l.biases, sizeof(float), l.nbiases, fp);
      fread(l.weights, sizeof(float), l.nweights, fp);
      if (l.batch_normalize) {
        fread(l.bn_scales, sizeof(float), l.nbiases, fp);
        fread(l.bn_biases, sizeof(float), l.nbiases, fp);
      }
#ifdef GPU
      if (gpu_index >= 0) {
        cuda_push_array(l.weights_gpu, l.weights, l.nweights);
        cuda_push_array(l.biases_gpu, l.biases, l.nbiases);
        if (l.batch_normalize) {
          cuda_push_array(l.bn_scales_gpu, l.bn_scales, l.nbiases);
          cuda_push_array(l.bn_biases_gpu, l.bn_biases, l.nbiases);
        }
      }
#endif
    }
    else if (l.type == CROP) {
      fread(l.crop_scale, sizeof(float), l.c, fp);
      fread(l.crop_trans, sizeof(float), l.c, fp);
#ifdef GPU
      if (gpu_index >= 0) {
        cuda_push_array(l.crop_scale_gpu, l.crop_scale, l.c);
        cuda_push_array(l.crop_trans_gpu, l.crop_trans, l.c);
      }
#endif
    }
    else if (l.type == SHORTCUT) {
      fread(l.norm_sc_coeff0, sizeof(float), l.out_c, fp);
      fread(l.norm_sc_coeff1, sizeof(float), l.c, fp);
      fread(l.norm_sc_bias, sizeof(float), l.out_c, fp);
#ifdef GPU
      if (gpu_index >= 0) {
        cuda_push_array(l.norm_sc_coeff0_gpu, l.norm_sc_coeff0, l.out_c);
        cuda_push_array(l.norm_sc_coeff1_gpu, l.norm_sc_coeff1, l.c);
        cuda_push_array(l.norm_sc_bias_gpu, l.norm_sc_bias, l.out_c);
      }
#endif
    }
    else if (l.type == BATCHNORM) {
      fread(l.bn_scales, sizeof(float), l.c, fp);
      fread(l.bn_biases, sizeof(float), l.c, fp);
#ifdef GPU
      if (gpu_index >= 0) {
        cuda_push_array(l.bn_scales_gpu, l.bn_scales, l.c);
        cuda_push_array(l.bn_biases_gpu, l.bn_biases, l.c);
      }
#endif
    }
  }

  printf("done.\n");
  fclose(fp);
}
