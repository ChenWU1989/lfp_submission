#include "layer.h"
#include "cuda.h"

#include <stdlib.h>

void free_layer(layer l)
{
    if(l.type == DROPOUT){
        if(l.rand) { 
          free(l.rand);
          l.rand = NULL;
        }
#ifdef GPU
        if(l.rand_gpu) {
          cuda_free(l.rand_gpu);
          l.rand_gpu = NULL;
        }
#endif
        return;
    }
    if(l.cweights) {
      free(l.cweights);
      l.cweights = NULL;
    }
    if(l.indexes) { 
      free(l.indexes);
      l.indexes = NULL;
    }
    if(l.input_layers) { 
      free(l.input_layers);
      l.input_layers = NULL;
    }
    if(l.input_sizes) { 
      free(l.input_sizes);
      l.input_sizes = NULL;
    }
    if(l.map) { 
      free(l.map);
      l.map = NULL;
    }
    if(l.rand) { 
      free(l.rand);
      l.rand = NULL;
    }
    if(l.cost) { 
      free(l.cost);
      l.cost = NULL;
    }
    if(l.state) { 
      free(l.state);
      l.state = NULL;
    }
    if(l.prev_state) { 
      free(l.prev_state);
      l.prev_state = NULL;
    }
    if(l.forgot_state) { 
      free(l.forgot_state);
      l.forgot_state = NULL;
    }
    if(l.forgot_delta) { 
      free(l.forgot_delta);
      l.forgot_delta = NULL;
    }
    if(l.state_delta) { 
      free(l.state_delta);
      l.state_delta = NULL;
    }
    if(l.concat) { 
      free(l.concat);
      l.concat = NULL;
    }
    if(l.concat_delta) { 
      free(l.concat_delta);
      l.concat_delta = NULL;
    }
    if(l.binary_weights) { 
      free(l.binary_weights);
      l.binary_weights = NULL;
    }
    if(l.biases) { 
      free(l.biases);
      l.biases = NULL;
    }
    if(l.bias_updates) { 
      free(l.bias_updates);
      l.bias_updates = NULL;
    }
    if(l.scales) { 
      free(l.scales);
      l.scales = NULL;
    }
    if(l.scale_updates) { 
      free(l.scale_updates);
      l.scale_updates = NULL;
    }
    if(l.weights) { 
      free(l.weights);
      l.weights = NULL;
    }
    if(l.weight_updates) { 
      free(l.weight_updates);
      l.weight_updates = NULL;
    }
    if(l.delta) { 
      free(l.delta);
      l.delta = NULL;
    }
    if(l.output) { 
      free(l.output);
      l.output = NULL;
    }
    if(l.squared) { 
      free(l.squared);
      l.squared = NULL;
    }
    if(l.norms) { 
      free(l.norms);
      l.norms = NULL;
    }
    if(l.spatial_mean) { 
      free(l.spatial_mean);
      l.spatial_mean = NULL;
    }
    if(l.mean) { 
      free(l.mean);
      l.mean = NULL;
    }
    if(l.variance) { 
      free(l.variance);
      l.variance = NULL;
    }
    if(l.mean_delta) { 
      free(l.mean_delta);
      l.mean_delta = NULL;
    }
    if(l.variance_delta) { 
      free(l.variance_delta);
      l.variance_delta = NULL;
    }
    if(l.rolling_mean) { 
      free(l.rolling_mean);
      l.rolling_mean = NULL;
    }
    if(l.rolling_variance) { 
      free(l.rolling_variance);
      l.rolling_variance = NULL;
    }
    if(l.x) { 
      free(l.x);
      l.x = NULL;
    }
    if(l.x_norm) { 
      free(l.x_norm);
      l.x_norm = NULL;
    }
    if(l.m) { 
      free(l.m);
      l.m = NULL;
    }
    if(l.v) { 
      free(l.v);
      l.v = NULL;
    }
    if(l.z_cpu) { 
      free(l.z_cpu);
      l.z_cpu = NULL;
    }
    if(l.r_cpu) { 
      free(l.r_cpu);
      l.r_cpu = NULL;
    }
    if(l.h_cpu) { 
      free(l.h_cpu);
      l.h_cpu = NULL;
    }
    if(l.binary_input) { 
      free(l.binary_input);
      l.binary_input = NULL;
    }
        
    if(l.bn_scales) {
      free(l.bn_scales);
      l.bn_scales = NULL;
    }
    if(l.bn_biases) {         
      free(l.bn_biases);
      l.bn_biases = NULL;
    }
    if(l.lfp) {
      free(l.lfp);
      l.lfp = NULL;
    }
    if(l.norm_sc_coeff0) {
      free(l.norm_sc_coeff0);
      l.norm_sc_coeff0 = NULL;
    }
    if(l.norm_sc_coeff1) {
      free(l.norm_sc_coeff1);
      l.norm_sc_coeff1 = NULL;
    }
    if(l.crop_scale) {
      free(l.crop_scale);
      l.crop_scale = NULL;
    }
    if(l.crop_trans) {
      free(l.crop_trans);
      l.crop_trans = NULL;
    }
    //if(l.norm_in_scale) {
    //  free(l.norm_in_scale);
    //  l.norm_in_scale = NULL;
    //}
    //if(l.norm_out_scale) {
    //  free(l.norm_out_scale);
    //  l.norm_out_scale = NULL;
    //}
#ifdef GPU
    if(l.indexes_gpu)           cuda_free((float *)l.indexes_gpu);
    
    if(l.bn_scales_gpu)         cuda_free(l.bn_scales_gpu);
    if(l.bn_biases_gpu)         cuda_free(l.bn_biases_gpu);

    if(l.z_gpu)                   cuda_free(l.z_gpu);
    if(l.r_gpu)                   cuda_free(l.r_gpu);
    if(l.h_gpu)                   cuda_free(l.h_gpu);
    if(l.m_gpu)                   cuda_free(l.m_gpu);
    if(l.v_gpu)                   cuda_free(l.v_gpu);
    if(l.prev_state_gpu)          cuda_free(l.prev_state_gpu);
    if(l.forgot_state_gpu)        cuda_free(l.forgot_state_gpu);
    if(l.forgot_delta_gpu)        cuda_free(l.forgot_delta_gpu);
    if(l.state_gpu)               cuda_free(l.state_gpu);
    if(l.state_delta_gpu)         cuda_free(l.state_delta_gpu);
    if(l.gate_gpu)                cuda_free(l.gate_gpu);
    if(l.gate_delta_gpu)          cuda_free(l.gate_delta_gpu);
    if(l.save_gpu)                cuda_free(l.save_gpu);
    if(l.save_delta_gpu)          cuda_free(l.save_delta_gpu);
    if(l.concat_gpu)              cuda_free(l.concat_gpu);
    if(l.concat_delta_gpu)        cuda_free(l.concat_delta_gpu);
    if(l.binary_input_gpu)        cuda_free(l.binary_input_gpu);
    if(l.binary_weights_gpu)      cuda_free(l.binary_weights_gpu);
    if(l.mean_gpu)                cuda_free(l.mean_gpu);
    if(l.variance_gpu)            cuda_free(l.variance_gpu);
    if(l.rolling_mean_gpu)        cuda_free(l.rolling_mean_gpu);
    if(l.rolling_variance_gpu)    cuda_free(l.rolling_variance_gpu);
    if(l.variance_delta_gpu)      cuda_free(l.variance_delta_gpu);
    if(l.mean_delta_gpu)          cuda_free(l.mean_delta_gpu);
    if(l.x_gpu)                   cuda_free(l.x_gpu);
    if(l.x_norm_gpu)              cuda_free(l.x_norm_gpu);
    if(l.weights_gpu)             cuda_free(l.weights_gpu);
    if(l.weight_updates_gpu)      cuda_free(l.weight_updates_gpu);
    if(l.biases_gpu)              cuda_free(l.biases_gpu);
    if(l.bias_updates_gpu)        cuda_free(l.bias_updates_gpu);
    if(l.scales_gpu)              cuda_free(l.scales_gpu);
    if(l.scale_updates_gpu)       cuda_free(l.scale_updates_gpu);
    if(l.output_gpu)              cuda_free(l.output_gpu);
    if(l.delta_gpu)               cuda_free(l.delta_gpu);
    if(l.rand_gpu)                cuda_free(l.rand_gpu);
    if(l.squared_gpu)             cuda_free(l.squared_gpu);
    if(l.norms_gpu)               cuda_free(l.norms_gpu);
#endif
}
