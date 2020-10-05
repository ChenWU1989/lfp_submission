#ifndef NORM_H
#define NORM_H

#include "darknet.h"
#include <stdbool.h>

image norm_img(image img, float* mean, float* var);
void norm_fm(float* x, int norm_batch, int c, int fmsz, float* mean, float* var);
void denorm_fm(float* x, int c, int fmsz, float* mean, float* var);
void norm_network(network* net, bool zero_mean);
void norm_save_ofm(layer l, int n);
void save_norm_decision_bin(network* net, char* filename);
void load_norm_decision_bin(network* net, char* filename);

#ifdef GPU
void norm_fm_gpu(float* x, int c, int fmsz, float* mean, float* var);
void denorm_fm_gpu(float* x, int c, int fmsz, float* mean, float* var);
#endif

#endif
