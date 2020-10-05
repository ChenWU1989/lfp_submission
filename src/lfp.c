#include "lfp.h"
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


float lfp_err(float *fp32, int n, int exp, QUANTIZE_TYPE type)
{
  float *lfp = calloc(n, sizeof(float));
  memcpy(lfp, fp32, sizeof(float)*n);

  lfp_data(lfp, n, exp, type);

  float err = 0;
  float scale = powf(2, exp);
  for (int i = 0; i < n; i++)
    err += ((lfp[i] / scale - fp32[i]) * (lfp[i] / scale - fp32[i]));
  err /= n;

  free(lfp);
  lfp = NULL;
  return err;
}


float lfp_err_rate(float *fp32, int n, int exp, QUANTIZE_TYPE type)
{
  float *lfp = calloc(n, sizeof(float));
  memcpy(lfp, fp32, sizeof(float)*n);

  lfp_data(lfp, n, exp, type);

  float err = 0;
  float scale = powf(2, exp);
  for (int i = 0; i < n; i++)
    err += fabs(lfp[i] / scale - fp32[i]) / fabs(fp32[i]);
  err /= n;

  free(lfp);
  lfp = NULL;
  return err;
}


int lfp_decision(float *x, int n, QUANTIZE_TYPE* type, bool multi)
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
        float err = lfp_err(x, n, i, quan_type);
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
      float err = lfp_err(x, n, i, *type);
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
int lfp_oexp_decision(network *net)
{
  int* oexp_candidate = calloc(3, sizeof(int));
  int* oexp_tmp = calloc(net->n, sizeof(int));
  int i = 0;
  
  bool multi = false;

  // find the optimal scale
  for (i = 0; i < net->n; i++) {
    layer l = net->layers[i];
    if (l.type == SOFTMAX)
      break;

    oexp_tmp[i] = lfp_decision(l.norm_output, l.outputs * net->norm_batch, &net->quantize->type, multi);
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
    
    float err_ori = lfp_err(l.norm_output, l.outputs * net->norm_batch, oexp_tmp[i], l.lfp->type);
    float err_0   = lfp_err(l.norm_output, l.outputs * net->norm_batch, oexp_candidate[0], l.lfp->type);
    float err_1   = lfp_err(l.norm_output, l.outputs * net->norm_batch, oexp_candidate[1], l.lfp->type);
    float err_2   = lfp_err(l.norm_output, l.outputs * net->norm_batch, oexp_candidate[2], l.lfp->type);
    
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

void lfp_network_decision(network *net)
{
  bool multi = false;

  net->quantize->iexp = lfp_decision(net->norm_input, net->inputs * net->norm_batch, &net->quantize->type, multi);
  
  // We can choose a uniform oexp for each layer, as we do normalization first, however,
  // we will still have an accuracy loss. If the network do not need a uniform oexp,
  // we will choose the optimal for each layer.
  bool uniform_or_not = uniform_oexp(net);
  int optimal_oexp = uniform_or_not ? lfp_oexp_decision(net) : 0;
  
  for (int i = 0; i < net->n; i++) {
    layer* l = net->layers + i;
    
    //printf("Decision for output feature map.\n");
    if (uniform_or_not)
      l->lfp->oexp = optimal_oexp;
    else {
      l->lfp->oexp = lfp_decision(l->norm_output, l->outputs * net->norm_batch, &l->lfp->type, multi);
    }

    //printf("Decision for input feature map.\n");
    if (i == 0)
      l->lfp->iexp = net->quantize->iexp;
    else
      l->lfp->iexp = net->layers[i - 1].lfp->oexp;
    
    if (l->type == CONVOLUTIONAL || l->type == CONNECTED) {
      l->lfp->wexp = lfp_decision(l->weights, l->nweights, &l->lfp->type, multi);
      if (l->batch_normalize)
        l->lfp->bn_sexp = lfp_decision(l->bn_scales, l->out_c, &l->lfp->type, multi);
      else
        l->lfp->bn_sexp = 0;
    }
    else if (l->type == CROP) {
      l->lfp->wexp = lfp_decision(l->crop_scale, l->c, &l->lfp->type, multi);
      l->lfp->bn_sexp = 0;
    }
    else if (l->type == SHORTCUT) {
      float* sc_coeff = calloc(l->out_c + l->c, sizeof(float));
      memcpy(sc_coeff, l->norm_sc_coeff0, sizeof(float)*l->out_c);
      memcpy(sc_coeff+l->out_c, l->norm_sc_coeff1, sizeof(float)*l->c);
      l->lfp->wexp = lfp_decision(sc_coeff, l->out_c + l->c, &l->lfp->type, multi);
      l->lfp->bn_sexp = 0;

      free(sc_coeff);
    }
    else if (l->type == BATCHNORM) {
      l->lfp->wexp = 0;
      l->lfp->bn_sexp = lfp_decision(l->bn_scales, l->out_c, &l->lfp->type, multi); 
    }
    else {
      l->lfp->wexp = 0;
      l->lfp->bn_sexp = 0;
    }
    
    //printf("Decision for bias and bn bias.\n");
    l->lfp->bexp = l->lfp->iexp + l->lfp->wexp;
    l->lfp->bn_bexp = l->lfp->bexp + l->lfp->bn_sexp;
    
    //printf("Decision for cutting from FL32 to LFP.\n");
    l->lfp->offset = l->lfp->bn_bexp - l->lfp->oexp;
    
    printf("Layer [ %3d ], iexp = %3d, wexp = %3d, bexp = %3d, oexp = %3d, offset = %3d, type = %3d\n",
            i, l->lfp->iexp, l->lfp->wexp, l->lfp->bexp, l->lfp->oexp, l->lfp->offset, l->lfp->type);
    if (l->type == BATCHNORM || l->batch_normalize == 1)
      printf("batch normalization: bn_sexp = %3d, bn_bexp = %3d\n", l->lfp->bn_sexp, l->lfp->bn_bexp);
  }
  
  for (int i = 0; i < net->n; i++) {
    float w_err = 0;
    switch (net->layers[i].type) {
      case CONVOLUTIONAL:
        w_err = lfp_err(net->layers[i].weights, net->layers[i].nweights, net->layers[i].lfp->wexp, net->layers[i].lfp->type);
        // w_err = lfp_err_rate(net->layers[i].weights, net->layers[i].nweights, net->layers[i].lfp->wexp, net->layers[i].lfp->type);
        break;
      case CONNECTED:
        w_err = lfp_err(net->layers[i].weights, net->layers[i].outputs * net->layers[i].inputs, net->layers[i].lfp->wexp, net->layers[i].lfp->type);
        // w_err = lfp_err_rate(net->layers[i].weights, net->layers[i].outputs * net->layers[i].inputs, net->layers[i].lfp->wexp, net->layers[i].lfp->type);
        break;
      default:
        w_err = 0;
        break;
    }
    printf("Layer [ %3d ], weights mse rate: %4.10f\n", i, w_err);
  }

  net->quantize->oexp = net->layers[net->n - 1].lfp->oexp;
}


void lfp_data(float *fp32, int n, int exp, QUANTIZE_TYPE type)
{
  float *fp32_tmp = calloc(n, sizeof(float));
  for (int i = 0; i < n; i++)
    fp32_tmp[i] = fp32[i] * powf(2, exp);

  uint32_t *fp32_bin = (uint32_t*)fp32_tmp;
  uint8_t *lfp_bin = calloc(n, sizeof(uint8_t));

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

  // cut data from fl32 to lfp
  for (int i = 0; i < n; i++)
    lfp_bin[i] = fp2lfp(fp32_bin[i], bit_width, exp_width);

  // represent lfp with fl32, as we will use fl32 to do calculation
  for (int i = 0; i < n; i++)
    fp32_bin[i] = lfp2fp(lfp_bin[i], bit_width, exp_width);
  
  memcpy(fp32, fp32_bin, sizeof(float)*n);
  free(fp32_tmp);
  free(lfp_bin);
  fp32_tmp = NULL;
  lfp_bin = NULL;
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

void lfp_quantize(network* net, QUANTIZE_TYPE type)
{
  net->quantize->type = type;
  
  // decide the exp for weights, bias, output feature
  printf("Quantization Decision...\n");
  lfp_network_decision(net);
  printf("Done.\n");
 
  // quantize weights
  printf("Quantize weights & bias...");
  for (int i = 0; i < net->n; i++) {
    layer* l = net->layers + i;
    if (l->type == CONVOLUTIONAL || l->type == CONNECTED) {
#ifdef LFP_DEBUG
      lfp_de_data(l->weights, l->nweights, -l->lfp->wexp);
      lfp_de_data(l->biases, l->nbiases, -l->lfp->bexp);
      if (l->batch_normalize) {
        lfp_de_data(l->bn_scales, l->out_c, -l->lfp->bn_sexp);
        lfp_de_data(l->bn_biases, l->out_c, -l->lfp->bn_bexp);
      }
#else
      lfp_data(l->weights, l->nweights, l->lfp->wexp, l->lfp->type);
      fp16_data(l->biases, l->nbiases, l->lfp->bexp);
      if (l->batch_normalize) {
        lfp_data(l->bn_scales, l->out_c, l->lfp->bn_sexp, l->lfp->type);
        fp16_data(l->bn_biases, l->out_c, l->lfp->bn_bexp);
      }
#endif
    }
    else if (l->type == CROP) {       // the crop layer is considered as image preprocess
                                      // we will quantize once after all the preprocess, 
                                      // not quantize between the preprocess several times.
      lfp_de_data(l->crop_scale, l->c, -l->lfp->wexp);
      lfp_de_data(l->crop_trans, l->c, -l->lfp->bexp);
    }
    else if (l->type == SHORTCUT) {
#ifdef LFP_DEBUG
      lfp_de_data(l->norm_sc_coeff0, l->out_c, -l->lfp->wexp);
      lfp_de_data(l->norm_sc_coeff1, l->c, -l->lfp->wexp);
      lfp_de_data(l->norm_sc_bias, l->out_c, -l->lfp->bexp);
#else
      lfp_data(l->norm_sc_coeff0, l->out_c, l->lfp->wexp, l->lfp->type);
      lfp_data(l->norm_sc_coeff1, l->c, l->lfp->wexp, l->lfp->type);
      fp16_data(l->norm_sc_bias, l->out_c, l->lfp->bexp);
#endif
    }
    else if (l->type == BATCHNORM) {
#ifdef LFP_DEBUG
      lfp_de_data(l->bn_scales, l->out_c, -l->lfp->bn_sexp);
      lfp_de_data(l->bn_biases, l->out_c, -l->lfp->bn_bexp);
#else
      lfp_data(l->bn_scales, l->out_c, l->lfp->bn_sexp, l->lfp->type);
      fp16_data(l->bn_biases, l->out_c, l->lfp->bn_bexp);
#endif
    }
  }
  printf("done\n");
}

void lfp_de_data(float *x, int n, int exp)
{
  float scale = powf(2, exp);
  for (int i = 0; i < n; i++)
    x[i] /= scale;
}

void lfp_normalize_cpu(float *x, float *scales, float *biases, int batch, int filters, int spatial)
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
    fprintf(fp, "%3d, %3d, %3d, %3d, %3d, %3d, %3d\n", l.lfp->iexp, l.lfp->wexp, l.lfp->bexp, l.lfp->oexp, l.lfp->bn_sexp, l.lfp->bn_bexp, l.lfp->offset);
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
    fscanf(fp, "%d, %d, %d, %d, %d, %d, %d\n", &l.lfp->iexp, &l.lfp->wexp, &l.lfp->bexp, &l.lfp->oexp, &l.lfp->bn_sexp, &l.lfp->bn_bexp, &l.lfp->offset);
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
    fwrite(&l.lfp->iexp, sizeof(int), 1, fp);
    fwrite(&l.lfp->wexp, sizeof(int), 1, fp);
    fwrite(&l.lfp->bexp, sizeof(int), 1, fp);
    fwrite(&l.lfp->oexp, sizeof(int), 1, fp);
    fwrite(&l.lfp->bn_sexp, sizeof(int), 1, fp);
    fwrite(&l.lfp->bn_bexp, sizeof(int), 1, fp);
    fwrite(&l.lfp->offset, sizeof(int), 1, fp);

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
    fread(&l.lfp->iexp, sizeof(int), 1, fp);
    fread(&l.lfp->wexp, sizeof(int), 1, fp);
    fread(&l.lfp->bexp, sizeof(int), 1, fp);
    fread(&l.lfp->oexp, sizeof(int), 1, fp);
    fread(&l.lfp->bn_sexp, sizeof(int), 1, fp);
    fread(&l.lfp->bn_bexp, sizeof(int), 1, fp);
    fread(&l.lfp->offset, sizeof(int), 1, fp);

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
