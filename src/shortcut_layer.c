#include "shortcut_layer.h"
#include "cuda.h"
#include "blas.h"
#include "activations.h"
#include "lfp.h"
#include "norm.h"

#include <stdio.h>
#include <assert.h>

layer make_shortcut_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2)
{
    fprintf(stderr, "res  %3d                %4d x%4d x%4d   ->  %4d x%4d x%4d\n",index, w2,h2,c2, w,h,c);
    layer l = {0};
    l.type = SHORTCUT;
    l.batch = batch;
    l.w = w2;
    l.h = h2;
    l.c = c2;
    l.out_w = w;
    l.out_h = h;
    l.out_c = c;
    l.outputs = w*h*c;
    l.inputs = l.outputs;

    l.index = index;

    l.delta =  calloc(l.outputs*batch, sizeof(float));
    l.output = calloc(l.outputs*batch, sizeof(float));;
    
    l.norm_sc_bias = calloc(l.out_c, sizeof(float));
    l.norm_sc_coeff0 = calloc(l.out_c, sizeof(float));
    l.norm_sc_coeff1 = calloc(l.c, sizeof(float));

    l.lfp = calloc(1, sizeof(LFP_DECISION));

    l.norm_in_mean = calloc(l.c + l.out_c, sizeof(float));
    l.norm_in_var  = calloc(l.c + l.out_c, sizeof(float));
    l.norm_out_mean = calloc(l.out_c, sizeof(float));
    l.norm_out_var  = calloc(l.out_c, sizeof(float));
    
    l.forward = forward_shortcut_layer;
    l.backward = backward_shortcut_layer;
    #ifdef GPU
    l.forward_gpu = forward_shortcut_layer_gpu;
    l.backward_gpu = backward_shortcut_layer_gpu;

    l.delta_gpu =  cuda_make_array(l.delta, l.outputs*batch);
    l.output_gpu = cuda_make_array(l.output, l.outputs*batch);
    
    l.norm_sc_bias_gpu = cuda_make_array(l.norm_sc_bias, l.out_c);
    l.norm_sc_coeff0_gpu = cuda_make_array(l.norm_sc_coeff0, l.out_c);
    l.norm_sc_coeff1_gpu = cuda_make_array(l.norm_sc_coeff1, l.c);
    #endif
    return l;
}

void resize_shortcut_layer(layer *l, int w, int h)
{
    assert(l->w == l->out_w);
    assert(l->h == l->out_h);
    l->w = l->out_w = w;
    l->h = l->out_h = h;
    l->outputs = w*h*l->out_c;
    l->inputs = l->outputs;
    l->delta =  realloc(l->delta, l->outputs*l->batch*sizeof(float));
    l->output = realloc(l->output, l->outputs*l->batch*sizeof(float));

#ifdef GPU
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->output_gpu  = cuda_make_array(l->output, l->outputs*l->batch);
    l->delta_gpu   = cuda_make_array(l->delta,  l->outputs*l->batch);
#endif
    
}


void forward_shortcut_layer(const layer l, network net)
{
    //copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    for (int i = 0; i < l.out_c; i++) {
      for (int j = 0; j < l.out_h * l.out_w; j++) {
        int idx = i * l.out_h * l.out_w + j;
        l.output[idx] = net.input[idx] * l.norm_sc_coeff0[i] + l.norm_sc_bias[i];
      }
    }

    shortcut_cpu(l.batch, l.w, l.h, l.c, net.layers[l.index].output, l.out_w, l.out_h, l.out_c, l.norm_sc_bias, l.norm_sc_coeff1, l.output);
        
    if (!net.zero_mean && l.norm_out_var && l.norm_out_var[0]) {
      lfp_de_data(l.output, l.outputs, l.lfp->bexp);
      denorm_fm(l.output, l.out_c, l.out_h * l.out_w, l.norm_out_mean, l.norm_out_var);
    }
    activate_array(l.output, l.outputs*l.batch, l.activation);
    if (!net.zero_mean && l.norm_out_var && l.norm_out_var[0]) {
      norm_fm(l.output, 1, l.out_c, l.out_h * l.out_w, l.norm_out_mean, l.norm_out_var);
      lfp_de_data(l.output, l.outputs, -l.lfp->bexp);
    }
    
    if (l.lfp->type != FP32 ) {
#ifdef LFP_DEBUG
      lfp_de_data(l.output, l.outputs*l.batch, l.lfp->offset);
#else
      lfp_data(l.output, l.outputs*l.batch, -l.lfp->offset, l.lfp->type);
#endif
    }
}

void backward_shortcut_layer(const layer l, network net)
{
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    axpy_cpu(l.outputs*l.batch, l.alpha, l.delta, 1, net.delta, 1);
    shortcut_cpu(l.batch, l.out_w, l.out_h, l.out_c, l.delta, l.w, l.h, l.c, l.norm_sc_bias, l.norm_sc_coeff1, net.layers[l.index].delta);
}

#ifdef GPU
void forward_shortcut_layer_gpu(const layer l, network net)
{
    //copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
    sc_copy_gpu(net.input_gpu, l.output_gpu, l.batch, l.out_c, l.out_h * l.out_w, l.norm_sc_coeff0_gpu, l.norm_sc_bias_gpu);

    shortcut_gpu(l.batch, l.w, l.h, l.c, net.layers[l.index].output_gpu, l.out_w, l.out_h, l.out_c, l.norm_sc_bias_gpu, l.norm_sc_coeff1_gpu, l.output_gpu);

    if (!net.zero_mean && l.norm_out_var && l.norm_out_var[0]) {
      float *mean_gpu = cuda_make_array(l.norm_out_mean, l.out_c);
      float *var_gpu = cuda_make_array(l.norm_out_var, l.out_c);
      lfp_de_data_gpu(l.output_gpu, l.outputs, l.lfp->bexp);
      denorm_fm_gpu(l.output_gpu, l.out_c, l.out_h * l.out_w, mean_gpu, var_gpu);
      activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
      norm_fm_gpu(l.output_gpu, l.out_c, l.out_h * l.out_w, mean_gpu, var_gpu);
      lfp_de_data_gpu(l.output_gpu, l.outputs, -l.lfp->bexp);

      cuda_free(mean_gpu);
      cuda_free(var_gpu);
    }
    else
      activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
    
    if (l.lfp->type != FP32) {
#ifdef LFP_DEBUG
      lfp_de_data_gpu(l.output_gpu, l.outputs*l.batch, l.lfp->offset);
#else
      lfp_data_gpu(l.output_gpu, l.outputs*l.batch, -l.lfp->offset, l.lfp->type);
#endif
    }
}

void backward_shortcut_layer_gpu(const layer l, network net)
{
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    axpy_gpu(l.outputs*l.batch, l.alpha, l.delta_gpu, 1, net.delta_gpu, 1);
    shortcut_gpu(l.batch, l.out_w, l.out_h, l.out_c, l.delta_gpu, l.w, l.h, l.c, l.norm_sc_bias_gpu, l.norm_sc_coeff1_gpu, net.layers[l.index].delta_gpu);
}
#endif
