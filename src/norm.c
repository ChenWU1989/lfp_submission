#include "norm.h"
#include "lfp.h"

#include <float.h>
#include <math.h>

float norm_find_max(float* x, int n)
{
	float max = -FLT_MAX;
	for (int i = 0; i < n; i++) {
		if (max < fabs(x[i])) {
			max = fabs(x[i]);
		}
	}

  return max;
}

float norm_find_mean(float* x, int n)
{
  float mean = 0.0f;
  for (int i = 0; i < n; i++)
    mean += x[i];
  mean /= n;

  return mean;
}

float norm_find_mean_nonzero(float* x, int n)
{
  float mean = 0.0f;
  int cnt = 0;
  for (int i = 0; i < n; i++) {
    if (x[i] != 0) {
      mean += x[i];
      cnt++;
    }
  }
  if (cnt != 0)
    mean /= cnt;
  else
    printf("All values are zero. Cannot find the mean of non-zero values!\n");
  
  return mean;
}

float norm_find_var(float* x, int n, float mean)
{
  float var = 0.0f;
  int cnt = 1;
  for (int i = 0; i < n; i++) {
    var += ((x[i] - mean) * (x[i] - mean));
    if (x[i] != 0)
      cnt++;
  }
  var = sqrt(var / cnt);
  
  return var;
}

void norm_input_decision_per_layer(network* net, bool zero_mean)
{
  float mean = 0.0f;
  if (!zero_mean)
    mean = norm_find_mean(net->norm_input, net->inputs * net->norm_batch);  
  float var = norm_find_var(net->norm_input, net->inputs * net->norm_batch, mean);
  //float var = 1.0f;
  
  for (int i = 0; i < net->c; i++) {
    net->norm_in_mean[i] = mean;
    net->norm_in_var[i] = var ? var : 1;
  }
}

void norm_ofm_decision_per_layer(layer* l, bool zero_mean, int norm_batch)
{
  float mean = 0.0f;
  if (!zero_mean)
    mean = norm_find_mean(l->norm_output, l->outputs * norm_batch);
  float var = norm_find_var(l->norm_output, l->outputs * norm_batch , mean);
  //float var = 1.0f;

  for (int i = 0; i < l->out_c; i++) {
    l->norm_out_mean[i] = mean;
    l->norm_out_var[i] = var ? var : 1;
  }
}

// ******************************************************************************** //
// find scales for the network:
// one scale for one layer.
//
// for convolutional layers, find scale using convolutional results, ignoring
// pooling layers.
//
// All output feature maps are stored on memory
// ******************************************************************************** //
void norm_decision_per_layer_mem(network* net, bool zero_mean)
{
	printf("Network normalization decision (per layer)...");
  
  // input
  norm_input_decision_per_layer(net, zero_mean);
  
  for (int i = 0; i < net->n; i++) {
		layer* l = net->layers + i;
    
    // Decide the in mean and var
    // It equals to the out mean and var of the last layer, but also depends on
    // the layer type.
    if (i == 0) {                       // first layer, the last is input
      memcpy(l->norm_in_mean, net->norm_in_mean, sizeof(float)*l->c);
      memcpy(l->norm_in_var, net->norm_in_var, sizeof(float)*l->c);
    }
    else if (l->type == CONNECTED) {    // need to consider following convolutional layer
      for (int in_i = 0; in_i < net->layers[i - 1].out_c; in_i++) {
        for (int in_j = 0; in_j <net->layers[i - 1].out_h * net->layers[i - 1].out_w; in_j++) {
          int idx = in_i * net->layers[i - 1].out_h * net->layers[i - 1].out_w + in_j;
          l->norm_in_mean[idx] = net->layers[i - 1].norm_out_mean[in_i];
          l->norm_in_var[idx] = net->layers[i - 1].norm_out_var[in_i];
        }
      } 
    }
    else if (l->type == SHORTCUT) {     // shortcut have two input layers
      memcpy(l->norm_in_mean, net->layers[i - 1].norm_out_mean, sizeof(float)*l->out_c);
      memcpy(l->norm_in_mean + l->out_c, net->layers[l->index].norm_out_mean, sizeof(float)*l->c);
      memcpy(l->norm_in_var, net->layers[i - 1].norm_out_var, sizeof(float)*l->out_c);
      memcpy(l->norm_in_var + l->out_c, net->layers[l->index].norm_out_var, sizeof(float)*l->c);
    }
    else if (l->type == ROUTE) {
      int cur_size = l->input_sizes[0] / l->out_w / l->out_h;
      memcpy(l->norm_in_mean, net->layers[l->input_layers[0]].norm_out_mean, sizeof(float)*cur_size);
      memcpy(l->norm_in_var,  net->layers[l->input_layers[0]].norm_out_var,  sizeof(float)*cur_size);

      for (int ii = 1; ii < l->n; ii++) {
        int pre_size = l->input_sizes[ii - 1] / l->out_w / l->out_h;
        cur_size = l->input_sizes[ii] / l->out_w / l->out_h; 
        memcpy(l->norm_in_mean + pre_size, net->layers[l->input_layers[ii]].norm_out_mean, sizeof(float)*cur_size); 
        memcpy(l->norm_in_var  + pre_size, net->layers[l->input_layers[ii]].norm_out_var , sizeof(float)*cur_size); 
      }
    }
    else {
      memcpy(l->norm_in_mean, net->layers[i - 1].norm_out_mean, sizeof(float)*l->c);
      memcpy(l->norm_in_var, net->layers[i - 1].norm_out_var, sizeof(float)*l->c);
    }
    
    // Decide the out mean and var
    // It is decided by the output feature map, also depends on the layer type.
    switch (l->type) {
    case CONVOLUTIONAL:
    case CONNECTED:
    case SHORTCUT:
    case BATCHNORM: 
    case CROP: {
      norm_ofm_decision_per_layer(l, zero_mean, net->norm_batch);
      break;
    }
    case AVGPOOL:
    case MAXPOOL:
    case SOFTMAX:
    case ROUTE:
    case DROPOUT: {
      memcpy(l->norm_out_mean, l->norm_in_mean, sizeof(float)*l->out_c);
      memcpy(l->norm_out_var, l->norm_in_var, sizeof(float)*l->out_c);
      break;
    }
		default:
			printf("Sorry, we do not support this layer, layer id: %d\n", l->type);
			break;
		}

  }

  // output
	memcpy(net->norm_out_mean, net->layers[net->n - 1].norm_out_mean, sizeof(float)*net->layers[net->n-1].out_c);
	memcpy(net->norm_out_var, net->layers[net->n - 1].norm_out_var, sizeof(float)*net->layers[net->n-1].out_c);

#ifdef GPU
  if (gpu_index >= 0) {
    cuda_push_array(net->norm_in_mean_gpu, net->norm_in_mean, net->c);
    cuda_push_array(net->norm_in_var_gpu, net->norm_in_var, net->c);
    cuda_push_array(net->norm_out_mean_gpu, net->norm_out_mean, net->layers[net->n - 1].out_c);
    cuda_push_array(net->norm_out_var_gpu, net->norm_out_var, net->layers[net->n - 1].out_c);
  }
#endif

  printf("done.\n");
}

// ******************************************************************************** //
// find scales for the network:
// one scale for one layer.
//
// for convolutional layers, find scale using convolutional results, ignoring
// pooling layers.
//
// All the output feature are on the file
// ******************************************************************************** //
void norm_decision_per_layer_file(network* net, bool zero_mean)
{
	printf("Network normalization decision (per layer), ofm from file...");
  
  // input
  FILE* fp;
  net->norm_input = calloc(net->inputs * net->norm_batch, sizeof(float));
  fp = fopen("./norm/net_input.bin", "rb");
  fread(net->norm_input, sizeof(float), net->inputs * net->norm_batch, fp);
  fclose(fp);
  norm_input_decision_per_layer(net, zero_mean);
  free(net->norm_input);

  for (int i = 0; i < net->n; i++) {
		layer* l = net->layers + i;
    
    // Decide the in mean and var
    // It equals to the out mean and var of the last layer, but also depends on
    // the layer type.
    if (i == 0) {                       // first layer, the last is input
      memcpy(l->norm_in_mean, net->norm_in_mean, sizeof(float)*l->c);
      memcpy(l->norm_in_var, net->norm_in_var, sizeof(float)*l->c);
    }
    else if (l->type == CONNECTED) {    // need to consider following convolutional layer
      for (int in_i = 0; in_i < net->layers[i - 1].out_c; in_i++) {
        for (int in_j = 0; in_j <net->layers[i - 1].out_h * net->layers[i - 1].out_w; in_j++) {
          int idx = in_i * net->layers[i - 1].out_h * net->layers[i - 1].out_w + in_j;
          l->norm_in_mean[idx] = net->layers[i - 1].norm_out_mean[in_i];
          l->norm_in_var[idx] = net->layers[i - 1].norm_out_var[in_i];
        }
      } 
    }
    else if (l->type == SHORTCUT) {     // shortcut have two input layers
      memcpy(l->norm_in_mean, net->layers[i - 1].norm_out_mean, sizeof(float)*l->out_c);
      memcpy(l->norm_in_mean + l->out_c, net->layers[l->index].norm_out_mean, sizeof(float)*l->c);
      memcpy(l->norm_in_var, net->layers[i - 1].norm_out_var, sizeof(float)*l->out_c);
      memcpy(l->norm_in_var + l->out_c, net->layers[l->index].norm_out_var, sizeof(float)*l->c);
    }
    else {
      memcpy(l->norm_in_mean, net->layers[i - 1].norm_out_mean, sizeof(float)*l->c);
      memcpy(l->norm_in_var, net->layers[i - 1].norm_out_var, sizeof(float)*l->c);
    }
    
    // Decide the out mean and var
    // It is decided by the output feature map, also depends on the layer type.
    switch (l->type) {
    case CONVOLUTIONAL:
    case CONNECTED:
    case SHORTCUT:
    case BATCHNORM: 
    case CROP: {
      l->norm_output = calloc(l->outputs * net->norm_batch, sizeof(float));
      char ofm_name[50];
      snprintf(ofm_name, sizeof(ofm_name), "./norm/ofm%d.bin", i);
      fp = fopen(ofm_name, "rb");
      fread(l->norm_output, sizeof(float), l->outputs * net->norm_batch, fp);
      norm_ofm_decision_per_layer(l, zero_mean, net->norm_batch);
      free(l->norm_output);
      fclose(fp);
      break;
    }
    case AVGPOOL:
    case MAXPOOL:
    case SOFTMAX:
    case DROPOUT: {
      memcpy(l->norm_out_mean, l->norm_in_mean, sizeof(float)*l->out_c);
      memcpy(l->norm_out_var, l->norm_in_var, sizeof(float)*l->out_c);
      break;
    }
		default:
			printf("Sorry, we do not support this layer, layer id: %d\n", l->type);
			break;
		}

  }

  // output
	memcpy(net->norm_out_mean, net->layers[net->n - 1].norm_out_mean, sizeof(float)*net->layers[net->n-1].out_c);
	memcpy(net->norm_out_var, net->layers[net->n - 1].norm_out_var, sizeof(float)*net->layers[net->n-1].out_c);

#ifdef GPU
  if (gpu_index >= 0) {
    cuda_push_array(net->norm_in_mean_gpu, net->norm_in_mean, net->c);
    cuda_push_array(net->norm_in_var_gpu, net->norm_in_var, net->c);
    cuda_push_array(net->norm_out_mean_gpu, net->norm_out_mean, net->layers[net->n - 1].out_c);
    cuda_push_array(net->norm_out_var_gpu, net->norm_out_var, net->layers[net->n - 1].out_c);
  }
#endif

  printf("done.\n");
}

image norm_img(image img, float* mean, float* var)
{
	image img_out;
	img_out.w = img.w;
	img_out.h = img.h;
	img_out.c = img.c;
	img_out.data = (float*)calloc(img.w * img.h * img.c, sizeof(float));
	for (int i = 0; i < img.c; i++) {
		for (int j = 0; j < img.w * img.h; j++) {
			img_out.data[i * img.w * img.h + j] = (img.data[i * img.w * img.h + j] - mean[i]) / var[i];
		}
	}

	return img_out;
}

void norm_fm(float* x, int norm_batch, int c, int fmsz, float* mean, float* var)
{
  for (int b = 0; b < norm_batch; b++) {
    for (int i = 0; i < c; i++) {
      for (int j = 0; j < fmsz; j++) {
        int idx = b * c * fmsz + i * fmsz + j;
        x[idx] = (x[idx] - mean[i]) / var[i];
      }
    }
  }
}

void norm_crop_layer(layer* l)
{
  for (int i = 0; i < l->out_c; i++) {
    l->crop_trans[i] = (l->crop_trans[i] - l->norm_out_mean[i]) / l->norm_out_var[i];
    l->crop_scale[i] = l->crop_scale[i] / l->norm_out_var[i];
  }
#ifdef GPU
  if (gpu_index >= 0) {
    cuda_push_array(l->crop_trans_gpu, l->crop_trans, l->out_c);
    cuda_push_array(l->crop_scale_gpu, l->crop_scale, l->out_c);
  }
#endif
}

void norm_bn_layer(layer* l)
{
	for (int i = 0; i < l->out_c; i++) {
		l->bn_biases[i] = (l->bn_biases[i] - l->norm_out_mean[i]) / l->norm_out_var[i];
		l->bn_scales[i] = l->bn_scales[i] / l->norm_out_var[i];
	}
#ifdef GPU
  if (gpu_index >= 0) {
    cuda_push_array(l->bn_biases_gpu, l->bn_biases, l->out_c);
    cuda_push_array(l->bn_scales_gpu, l->bn_scales, l->out_c);
  }
#endif
}

void norm_conv_layer(layer* l)
{
	for (int i = 0; i < l->out_c; i++) {
		l->biases[i] = (l->biases[i] - l->norm_out_mean[i]) / l->norm_out_var[i];
    if (l->batch_normalize)
      l->bn_biases[i] = (l->bn_biases[i] - l->norm_out_mean[i]) / l->norm_out_var[i];
		for (int j = 0; j < l->c * l->size * l->size; j++) {
			int idx = i * l->c * l->size * l->size + j;
      l->weights[idx] = l->weights[idx] / l->norm_out_var[i];
		}
	}
#ifdef GPU
  if (gpu_index >= 0) {
    cuda_push_array(l->biases_gpu, l->biases, l->out_c);
    cuda_push_array(l->weights_gpu, l->weights, l->nweights);
    if (l->batch_normalize)
      cuda_push_array(l->bn_biases_gpu, l->bn_biases, l->out_c);
  }
#endif
}

void norm_shortcut_layer(layer* l)
{
  for (int i = 0; i < l->c; i++) {
		l->norm_sc_coeff1[i] = l->norm_sc_coeff1[i] / l->norm_out_var[i];
  }
  
	for (int i = 0; i < l->out_c; i++) {
    l->norm_sc_bias[i] = (l->norm_sc_bias[i] - l->norm_out_mean[i]) / l->norm_out_var[i];
		l->norm_sc_coeff0[i] = l->norm_sc_coeff0[i] / l->norm_out_var[i];
	}
#ifdef GPU
  if (gpu_index >= 0) {
    cuda_push_array(l->norm_sc_bias_gpu, l->norm_sc_bias, l->out_c);
    cuda_push_array(l->norm_sc_coeff1_gpu, l->norm_sc_coeff1, l->c);
    cuda_push_array(l->norm_sc_coeff0_gpu, l->norm_sc_coeff0, l->out_c);
  }
#endif
}

void denorm_fm(float* x, int c, int fmsz, float* mean, float* var)
{
  for (int i = 0; i < c; i++) {
  	for (int j = 0; j < fmsz; j++) {
      int idx = i * fmsz + j;
  		x[idx] = x[idx] * var[i] + mean[i];
  	}
  }
}

void denorm_crop_layer(layer* l)
{
  for (int i = 0; i < l->c; i++) {
    l->crop_trans[i] = l->crop_trans[i] + l->crop_scale[i] * l->norm_in_mean[i];
    l->crop_scale[i] = l->crop_scale[i] * l->norm_in_var[i];
  }
}

void denorm_bn_layer(layer* l)
{
	for (int i = 0; i < l->c; i++) {
    l->bn_biases[i] = l->bn_biases[i] + l->bn_scales[i] * l->norm_in_mean[i];
		l->bn_scales[i] = l->bn_scales[i] * l->norm_in_var[i];
	}
}

void denorm_conv_layer(layer* l)
{
  int idx = 0;
  for (int i = 0; i < l->out_c; i++) {
    float sum = 0;
    float sum_bn = 0;
    for (int j = 0; j < l->c; j++) {
      for (int k = 0; k < l->size * l->size; k++) {
        idx = i * l->c * l->size * l->size + j * l->size * l->size + k;
        sum = sum + l->weights[idx] * l->norm_in_mean[j];
        if (l->batch_normalize == 1)
          sum_bn = sum_bn + l->weights[idx] * l->bn_scales[i] * l->norm_in_mean[j];
      }
    }
    l->biases[i] = l->biases[i] + sum;
    if (l->batch_normalize == 1)
      l->bn_biases[i] = l->bn_biases[i] + sum_bn;
  }

	for (int i = 0; i < l->out_c; i++) {
		for (int j = 0; j < l->c; j++) {
			for (int k = 0; k < l->size * l->size; k++) {
				idx = i * l->c * l->size * l->size + j * l->size * l->size + k;
				l->weights[idx] = l->weights[idx] * l->norm_in_var[j];
			}
		}
	}
}

void denorm_shortcut_layer(layer* l)
{
  int minc = l->out_c > l->c ? l->c : l->out_c; 
  for (int i = 0; i < minc; i++)
    l->norm_sc_bias[i] = l->norm_sc_bias[i] + l->norm_sc_coeff0[i] * l->norm_in_mean[i] +
                                              l->norm_sc_coeff1[i] * l->norm_in_mean[l->out_c + i];
  if (minc < l->out_c) {
    for (int i = minc; i < l->out_c; i++) {
      l->norm_sc_bias[i] = l->norm_sc_bias[i] + l->norm_sc_coeff0[i] * l->norm_in_mean[i];
    }
  }

  for (int i = 0; i < l->out_c; i++)
    l->norm_sc_coeff0[i] = l->norm_sc_coeff0[i] * l->norm_in_var[i];

  for (int i = 0; i < l->c; i++)
    l->norm_sc_coeff1[i] = l->norm_sc_coeff1[i] * l->norm_in_var[i + l->out_c];
}

void save_norm_decision(network* net)
{
  FILE* fp = fopen("norm_decision.txt", "w");
	fprintf(fp, "net in: \n");
  for (int j = 0; j < net->c; j++)
    fprintf(fp, "%4.10f, %4.10f\n", net->norm_in_mean[j], net->norm_in_var[j]);
  fprintf(fp, "\n");
  
  for (int i = 0; i < net->n; i++) {
		fprintf(fp, "Layer [ %2d ]: \n", i);
    fprintf(fp, "in: \n");
    int in_num;
    if (net->layers[i].type == SHORTCUT)
      in_num = net->layers[i].c + net->layers[i].out_c;
    else
      in_num = net->layers[i].c;
    for (int j = 0; j < in_num; j++) {
      fprintf(fp, "%4.10f, %4.10f\n", net->layers[i].norm_in_mean[j], net->layers[i].norm_in_var[j]);
    }
    fprintf(fp, "out: \n");
		for (int j = 0; j < net->layers[i].out_c; j++) {
			fprintf(fp, "%4.10f, %4.10f\n", net->layers[i].norm_out_mean[j], net->layers[i].norm_out_var[j]);
    }
    fprintf(fp, "\n");
  }
  	
  fprintf(fp, "net out: \n");
  for (int j = 0; j < net->outputs; j++)
    fprintf(fp, "%4.10f, %4.10f\n", net->norm_out_mean[j], net->norm_out_var[j]);
	fclose(fp);
}

void save_norm_decision_bin(network* net, char* filename)
{
  FILE* fp = fopen(filename, "wb");
  fwrite((const void*)net->norm_in_mean, sizeof(float), net->c, fp);
  fwrite((const void*)net->norm_in_var, sizeof(float), net->c, fp);
  
  fwrite((const void*)net->norm_out_mean, sizeof(float), net->layers[net->n-1].out_c, fp);
  fwrite((const void*)net->norm_out_var, sizeof(float), net->layers[net->n-1].out_c, fp);

  for (int i = 0; i < net->n; i++) {
    int in_num = 0;
    if (net->layers[i].type == SHORTCUT)
      in_num = net->layers[i].c + net->layers[i].out_c;
    else
      in_num = net->layers[i].c;
    
    fwrite((const void*)net->layers[i].norm_in_mean, sizeof(float), in_num, fp);
    fwrite((const void*)net->layers[i].norm_in_var, sizeof(float), in_num, fp);

    fwrite((const void*)net->layers[i].norm_out_mean, sizeof(float), net->layers[i].out_c, fp);
    fwrite((const void*)net->layers[i].norm_out_var, sizeof(float), net->layers[i].out_c, fp);
  }
}

void load_norm_decision_bin(network* net, char* filename)
{
  FILE* fp = fopen(filename, "rb");

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
    int in_num = 0;
    if (net->layers[i].type == SHORTCUT)
      in_num = net->layers[i].c + net->layers[i].out_c;
    else
      in_num = net->layers[i].c;
    
    fread(net->layers[i].norm_in_mean, sizeof(float), in_num, fp);
    fread(net->layers[i].norm_in_var, sizeof(float), in_num, fp);

    fread(net->layers[i].norm_out_mean, sizeof(float), net->layers[i].out_c, fp);
    fread(net->layers[i].norm_out_var, sizeof(float), net->layers[i].out_c, fp);
  }
}

void norm_ofm(network* net)
{
  norm_fm(net->norm_input, net->norm_batch, net->c, net->h * net->w, net->norm_in_mean, net->norm_in_var);

  for (int i = 0; i < net->n; i++) {
    layer* l = net->layers + i;

    norm_fm(l->norm_output, net->norm_batch, l->out_c, l->out_h * l->out_w, l->norm_out_mean, l->norm_out_var);
  }
}

void norm_weights(network* net)
{
  for (int i = 0; i < net->n; i++) {
		layer* l = net->layers + i;

		switch (l->type) {
	  case CONVOLUTIONAL: 
    case CONNECTED: {
      denorm_conv_layer(l);
      norm_conv_layer(l);
      break;
    }
    case BATCHNORM: {
      denorm_bn_layer(l);
      norm_bn_layer(l);
      break;
    }
    case SHORTCUT: {
      denorm_shortcut_layer(l);
      norm_shortcut_layer(l);
      break;
    }
    case CROP: {
      denorm_crop_layer(l);
      norm_crop_layer(l);
      break;
    }
    case AVGPOOL:
    case MAXPOOL:
    case SOFTMAX:
    case DROPOUT:
    case ROUTE:
      break;
		default:
			printf("Sorry, we do not support this layer, layer id: %d\n", l->type);
			break;
		}
	}
}

// ******************************************************************************** //
// normalize network:
// 1) find normalization scales;
// 2) normalize output weights to get normalized outputs;
// 3) denormalize input weights to keep correct outputs;
// ******************************************************************************** //
void norm_network(network* net, bool zero_mean)
{
  norm_decision_per_layer_mem(net, zero_mean);

  norm_ofm(net);

  norm_weights(net);
}

void norm_save_ofm(layer l, int n)
{
#ifdef GPU
  if (gpu_index >= 0) {
    cuda_pull_array(l.output_gpu, l.output, l.outputs);
  }
#endif
  
  if (l.lfp)
    lfp_de_data(l.output, l.outputs, l.lfp->oexp); 

	if (l.norm_out_var != NULL && l.norm_out_var[0] != 0) {
		char ofm_name[50];
#ifdef GPU
    if (gpu_index >= 0)
      snprintf(ofm_name, sizeof(ofm_name), "norm_ofm/norm_ofm%d_gpu.bin", n);
    else
      snprintf(ofm_name, sizeof(ofm_name), "norm_ofm/norm_ofm%d.bin", n);
#else
    snprintf(ofm_name, sizeof(ofm_name), "norm_ofm/norm_ofm%d.bin", n);
#endif
    float *ofm = (float*)calloc(l.outputs, sizeof(float));
		for (int i = 0; i < l.out_c; i++) {
			for (int j = 0; j < l.out_w * l.out_h; j++) {
				int idx = i * l.out_w * l.out_h + j;
        ofm[idx] = l.output[idx] * l.norm_out_var[i] + l.norm_out_mean[i];
			}
		}
    FILE* fp = fopen(ofm_name, "wb");
    fwrite((const void*)ofm, sizeof(float), l.outputs, fp);
		fclose(fp);
		free(ofm);
    ofm = NULL;
	}
	else {
		char ofm_name[50];
    snprintf(ofm_name, sizeof(ofm_name), "norm_ofm/ofm%d.bin", n);

		FILE* fp = fopen(ofm_name, "wb");
    fwrite((const void*)l.output, sizeof(float), l.outputs, fp);
		fclose(fp);
	}
}
