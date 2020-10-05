#ifndef FL8_H
#define FL8_H

#include "darknet.h"
#include <stdbool.h>
//#ifndef FP8_DEBUG
//#define FP8_DEBUG
//#endif

typedef unsigned int uint32_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;

void fp8_quantize(network* net, QUANTIZE_TYPE type, bool onfile);
void fp8_data(float *x, int n, int exp, QUANTIZE_TYPE type);
void fp8_de_data(float *x, int n, int exp);
void fp8_normalize_cpu(float *x, float *scales, float *biases, int batch, int filters, int spatial);

void save_quantize_decision(network* net, char* filename);
void load_quantize_decision(network* net, char* filename);
void save_quantize_weights(network* net, char* filename);
void load_quantize_weights(network* net, char* filename);

float* merge_bn_scales(float* variance, float* scales, int filters);
float* merge_bn_biases(float* mean, float* variance, float* scales, float* biases, int filters);
#ifdef GPU
void fp8_data_gpu(float *x, int n, int exp, QUANTIZE_TYPE type);
void fp8_de_data_gpu(float *x, int n, int exp);
void fp8_normalize_gpu(float *x, float *scales, float *biases, int batch, int filters, int spatial);
#endif

#endif
