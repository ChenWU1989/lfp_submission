#include "darknet.h"
#include "lfp.h"
#include "norm.h"
#include "network.h"

extern QUANTIZE_TYPE get_type(char *type_str);

QUANTIZE_TYPE get_type(char *type_str)
{
  if ( 0 == strcmp(type_str, "FP32") || 0 == strcmp(type_str, "fp32") )
    return FP32;
  // fp8
  else if ( 0 == strcmp(type_str, "M0E7") || 0 == strcmp(type_str, "m0e7") )
    return M0E7;
  else if ( 0 == strcmp(type_str, "M1E6") || 0 == strcmp(type_str, "m1e6") )
    return M1E6;
  else if ( 0 == strcmp(type_str, "M2E5") || 0 == strcmp(type_str, "m2e5") )
    return M2E5;
  else if ( 0 == strcmp(type_str, "M3E4") || 0 == strcmp(type_str, "m3e4") )
    return M3E4;
  else if ( 0 == strcmp(type_str, "M4E3") || 0 == strcmp(type_str, "m4e3") )
    return M4E3;
  else if ( 0 == strcmp(type_str, "M5E2") || 0 == strcmp(type_str, "m5e2") )
    return M5E2;
  else if ( 0 == strcmp(type_str, "M6E1") || 0 == strcmp(type_str, "m6e1") )
    return M6E1;
  else if ( 0 == strcmp(type_str, "M7E0") || 0 == strcmp(type_str, "m7e0") )
    return M7E0;
  // fp7
  else if ( 0 == strcmp(type_str, "M6E0") || 0 == strcmp(type_str, "m6e0") )
    return M6E0;
  else if ( 0 == strcmp(type_str, "M5E1") || 0 == strcmp(type_str, "m5e1") )
    return M5E1;
  else if ( 0 == strcmp(type_str, "M4E2") || 0 == strcmp(type_str, "m4e2") )
    return M4E2;
  else if ( 0 == strcmp(type_str, "M3E3") || 0 == strcmp(type_str, "m3e3") )
    return M3E3;
  else if ( 0 == strcmp(type_str, "M2E4") || 0 == strcmp(type_str, "m2e4") )
    return M2E4;
  else if ( 0 == strcmp(type_str, "M1E5") || 0 == strcmp(type_str, "m1e5") )
    return M1E5;
  else if ( 0 == strcmp(type_str, "M0E6") || 0 == strcmp(type_str, "m0e6") )
    return M0E6;
  // fp6
  else if ( 0 == strcmp(type_str, "M0E5") || 0 == strcmp(type_str, "m0e5") )
    return M0E5;
  else if ( 0 == strcmp(type_str, "M1E4") || 0 == strcmp(type_str, "m1e4") )
    return M1E4;
  else if ( 0 == strcmp(type_str, "M2E3") || 0 == strcmp(type_str, "m2e3") )
    return M2E3;
  else if ( 0 == strcmp(type_str, "M3E2") || 0 == strcmp(type_str, "m3e2") )
    return M3E2;
  else if ( 0 == strcmp(type_str, "M4E1") || 0 == strcmp(type_str, "m4e1") )
    return M4E1;
  else if ( 0 == strcmp(type_str, "M5E0") || 0 == strcmp(type_str, "m5e0") )
    return M5E0;
  // fp5
  else if ( 0 == strcmp(type_str, "M0E4") || 0 == strcmp(type_str, "m0e4") )
    return M0E4;
  else if ( 0 == strcmp(type_str, "M1E3") || 0 == strcmp(type_str, "m1e3") )
    return M1E3;
  else if ( 0 == strcmp(type_str, "M2E2") || 0 == strcmp(type_str, "m2e2") )
    return M2E2;
  else if ( 0 == strcmp(type_str, "M3E1") || 0 == strcmp(type_str, "m3e1") )
    return M3E1;
  else if ( 0 == strcmp(type_str, "M4E0") || 0 == strcmp(type_str, "m4e0") )
    return M4E0;
  // fp4
  else if ( 0 == strcmp(type_str, "M0E3") || 0 == strcmp(type_str, "m0e3") )
    return M0E3;
  else if ( 0 == strcmp(type_str, "M1E2") || 0 == strcmp(type_str, "m1e2") )
    return M1E2;
  else if ( 0 == strcmp(type_str, "M2E1") || 0 == strcmp(type_str, "m2e1") )
    return M2E1;
  else if ( 0 == strcmp(type_str, "M3E0") || 0 == strcmp(type_str, "m3e0") )
    return M3E0;
  else
    return FP32;
}

// Save all the output feature to memory, it will cost large amount of memory when batch is large
void network_batch_all_mem(network* net, char** paths, int batch)
{
  int* indexes = calloc(5, sizeof(int));
  
  net->norm_batch = batch;
  net->norm_input = calloc(batch*net->inputs, sizeof(float));
  for(int i = 0; i < batch; ++i){
    char *path = paths[i];

    // read image
#ifdef LFP_DEBUG
    image im = load_image_color("data/cat1.jpg", 0, 0);
#else
    image im = load_image_color(path, 0, 0);
#endif
    image crop = center_crop_image(im, net->w, net->h);

    // predict with 32bit floating point weights
    float *predictions = network_predict(net, crop.data);
#ifdef LFP_DEBUG
    // only for debug
    top_k(predictions, net->outputs, 5, indexes);
    for(int j = 0; j < 5; j++)
      printf("%5.2f%%: %d\n", predictions[indexes[j]]*100, indexes[j]);

    // for (int k = 0; k < net->n; k++)
    //   norm_save_ofm(net->layers[k], k);
#endif

    // save all the ofm to memory 
    // input
    memcpy(net->norm_input + i * net->inputs, crop.data, sizeof(float)*net->inputs); 

    // layers
    for (int j = 0; j < net->n; j++) {
      layer* l = net->layers + j;
      
      if (i == 0)
        l->norm_output = calloc(batch*l->outputs, sizeof(float));

#ifdef GPU
      if (gpu_index >= 0)
        cuda_pull_array(l->output_gpu, l->norm_output + i * l->outputs, l->outputs);
      else
        memcpy(l->norm_output + i * l->outputs, l->output, l->outputs * sizeof(float));
#else
        memcpy(l->norm_output + i * l->outputs, l->output, l->outputs * sizeof(float));
#endif
    }
  }
  
  free(indexes);
  indexes = NULL;
}

// Save all the output feature to file 
void network_batch_all_file(network* net, char** paths, int batch)
{
  int* indexes = calloc(5, sizeof(int));
  
  net->norm_batch = batch;
  for(int i = 0; i < batch; ++i){
    char *path = paths[i];

    // read image
#ifdef LFP_DEBUG
    image im = load_image_color("data/cat1.jpg", 0, 0);
#else
    image im = load_image_color(path, 0, 0);
#endif
    image crop = center_crop_image(im, net->w, net->h);

    // predict with 32bit floating point weights
    float *predictions = network_predict(net, crop.data);
#ifdef LFP_DEBUG
    // only for debug
    top_k(predictions, net->outputs, 5, indexes);
    for(int j = 0; j < 5; j++)
      printf("%5.2f%%: %d\n", predictions[indexes[j]]*100, indexes[j]);

    for (int k = 0; k < net->n; k++)
      norm_save_ofm(net->layers[k], k);
#endif

    // save all the ofm to file 
    // input
    FILE * fp;
    if (i == 0)
      fp = fopen("./norm/net_input.bin", "wb");
    else
      fp = fopen("./norm/net_input.bin", "ab");

    //fseek(fp, 0, SEEK_END);
    fwrite((const void*)crop.data, sizeof(float), net->inputs, fp);
    fclose(fp);

    // layers
    for (int j = 0; j < net->n; j++) {
      layer* l = net->layers + j;
      
      char ofm_name[50];
      snprintf(ofm_name, sizeof(ofm_name), "./norm/ofm%d.bin", j);
      if (i == 0)
        fp = fopen(ofm_name, "wb");
      else
        fp = fopen(ofm_name, "ab");
      
      l->norm_output = calloc(l->outputs, sizeof(float));
#ifdef GPU
      if (gpu_index >= 0)
        cuda_pull_array(l->output_gpu, l->norm_output, l->outputs);
#endif
      
      //fseek(fp, 0, SEEK_END);
      fwrite((const void*)l->norm_output, sizeof(float), l->outputs, fp);
      
      free(l->norm_output);
      fclose(fp);
    }
  }
  
  free(indexes);
  indexes = NULL;
}

void quantize_q_classifier(char *data, char *cfg, char *weight, QUANTIZE_TYPE type, char *quantize_weights, bool zero_mean, int batch)
{
    network *net = load_network(cfg, weight, 0);
    set_batch_network(net, 1);
    srand(2222222);
    
    printf("Merge batchnormalization...");
    for (int i = 0; i < net->n; i++) {
      layer* l = net->layers + i;
      if (l->batch_normalize == 1 || l->type == BATCHNORM) {
        l->bn_scales = merge_bn_scales(l->rolling_variance, l->scales, l->out_c);
        l->bn_biases = merge_bn_biases(l->rolling_mean, l->rolling_variance, l->scales, l->biases, l->out_c);
#ifdef GPU
        if (gpu_index >= 0) {
          cuda_push_array(l->bn_scales_gpu, l->bn_scales, l->out_c);
          cuda_push_array(l->bn_biases_gpu, l->bn_biases, l->out_c);
        }
#endif
      }
    }

    list *options = read_data_cfg(data);
    char *batch_list = option_find_str(options, "batch", "data/imagenet.batch.list");
    list *plist = get_paths(batch_list);
    char **paths = (char **)list_to_array(plist);
    
    // normalize network
    printf("Normalize with a batch with %d images.\n", batch);
    net->quantize->type = FP32;
    for (int ii = 0; ii < net->n; ii++) {
      net->layers[ii].lfp->type = FP32;
    }
    net->zero_mean = zero_mean;
    
    network_batch_all_mem(net, paths, batch);

    norm_network(net, net->zero_mean);

    // quantize
    printf("Quantize with the normalized network.\n");
    for (int ii = 0; ii < net->n; ii++) {
      net->layers[ii].lfp->type = type;
    }

    if ( type != FP32 ) {
      lfp_quantize(net, type);
    }

    save_quantize_weights(net, quantize_weights);
}

network *load_quantize_network(char *cfg, char *weights, int clear, QUANTIZE_TYPE type, bool zero_mean)
{
  network *net = parse_network_cfg(cfg);
  if (weights && weights[0] != 0) {
    load_quantize_weights(net, weights);
  }
  if (clear)
    (*net->seen) = 0;
   
  net->quantize->type = type;
  for (int i = 0; i < net->n; i++) {
    net->layers[i].lfp->type = type;
  }

  net->zero_mean = zero_mean;

  return net;
}

void predict_q_classifier(char *data, char *cfg, char *weights, char *img, QUANTIZE_TYPE type, bool zero_mean)
{
  // read network config and load quantization decision and quantized weights
  network *net = load_quantize_network(cfg, weights, 0, type, zero_mean);

  set_batch_network(net, 1);
  srand(2222222);
  
  // load class names
  list *options = read_data_cfg(data);

  char *name_list = option_find_str(options, "names", 0);
  if(!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
  int top = option_find_int(options, "top", 1);

  int i = 0;
  char **names = get_labels(name_list);
  clock_t time;
  int *indexes = calloc(top, sizeof(int));
  char buff[256];
  char *input = buff;
  
  // read image
  if (img){
      strncpy(input, img, 256);
  }
  else{
      printf("Enter Image Path: ");
      fflush(stdout);
      input = fgets(input, 256, stdin);
      if(!input) return;
      strtok(input, "\n");
  }
  image im = load_image_color(input, 0, 0);
  image crop = center_crop_image(im, net->w, net->h);
  image im_normed = norm_img(crop, net->norm_in_mean, net->norm_in_var); 

  // predict
  printf("Quantize input image...");
  float *X = im_normed.data;
#ifdef LFP_DEBUG
  lfp_de_data(X, im_normed.h * im_normed.w * im_normed.c, -net->quantize->iexp);
#else
  if (net->layers[0].type == CROP)
    lfp_de_data(X, im_normed.h * im_normed.w * im_normed.c, -net->quantize->iexp);
  else
    lfp_data(X, im_normed.h * im_normed.w * im_normed.c, net->quantize->iexp, type);
#endif
  printf("done.\n");
  
  time = clock();
  float *predictions = network_predict(net, X);
  if(net->hierarchy) hierarchy_predictions(predictions, net->outputs, net->hierarchy, 1, 1);
  top_k(predictions, net->outputs, top, indexes);
  printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
  for(i = 0; i < top; ++i){
      int index = indexes[i];
      printf("%5.2f%%: %s\n", predictions[index]*100, names[index]);
  }

  free_image(im);
  free_image(crop);
  free_image(im_normed);
}

void validate_q_classifier(char *data, char *cfg, char *weights, QUANTIZE_TYPE type, bool zero_mean)
{  
  // read network config and load quantization decision and quantized weights
  network *net = load_quantize_network(cfg, weights, 0, type, zero_mean);

  set_batch_network(net, 1);
  srand(time(0));
  
  int i, j;
  // load class names
  list *options = read_data_cfg(data);

  char *label_list = option_find_str(options, "labels", "data/labels.list");
  char *leaf_list = option_find_str(options, "leaves", 0);
  if(leaf_list) change_leaves(net->hierarchy, leaf_list);
  char *valid_list = option_find_str(options, "valid", "data/train.list");
  int classes = option_find_int(options, "classes", 2);
  int topk = option_find_int(options, "top", 1);

  char **labels = get_labels(label_list);
  list *plist = get_paths(valid_list);

  char **paths = (char **)list_to_array(plist);
  int m = plist->size;
  free_list(plist);

  float avg_acc = 0;
  float avg_topk = 0;
  int *indexes = calloc(topk, sizeof(int));

  // m = 1;
  for(i = 0; i < m; ++i){
    int class = -1;
    char *path = paths[i];
    for(j = 0; j < classes; ++j){
      if(strstr(path, labels[j])){
        class = j;
        break;
      }
    }
    image im = load_image_color(paths[i], 0, 0);
    image crop = center_crop_image(im, net->w, net->h);
    image im_normed = norm_img(crop, net->norm_in_mean, net->norm_in_var); 

    // calculate with quantized weights
    float *X = im_normed.data;
#ifdef LFP_DEBUG
    lfp_de_data(X, im_normed.h * im_normed.w * im_normed.c, -net->quantize->iexp);
#else
    if (net->layers[0].type == CROP)
      lfp_de_data(X, im_normed.h * im_normed.w * im_normed.c, -net->quantize->iexp);
    else
      lfp_data(X, im_normed.h * im_normed.w * im_normed.c, net->quantize->iexp, type);
#endif

    float *pred = network_predict(net, X);
    if(net->hierarchy) hierarchy_predictions(pred, net->outputs, net->hierarchy, 1, 1);

    free_image(im);
    free_image(crop);
    free_image(im_normed);
    top_k(pred, classes, topk, indexes);

    if(indexes[0] == class) avg_acc += 1;
    for(j = 0; j < topk; ++j){
      if(indexes[j] == class) avg_topk += 1;
    }

    printf("%s, %d, %d, %5.2f%%, \n", paths[i], class, indexes[0], pred[indexes[0]] * 100);
    printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    
    if (i == m - 1) {
      FILE *f = fopen("validation_result.txt", "a");
      fprintf(f, "Quantization type: %d\n", type);
      fprintf(f, "top 1: %f, top %d: %f\n", avg_acc/(i+1), topk, avg_topk/(i+1));
      fclose(f);
    }
  }
}

void run_q_classifier(int argc, char **argv)
{
    char *data = argv[3];
    char *cfg = argv[4];
    char *weights = argv[5];

    char *type_str = find_char_arg(argc, argv, "-t", "FP32");
    char *img = find_char_arg(argc, argv, "-p", "\0");
    char *quantize_weights = find_char_arg(argc, argv, "-qw", "quantize.weights");
    int zero_mean_flag = find_int_arg(argc, argv, "-zm", 1);
    int batch = find_int_arg(argc, argv, "-b", 1);
    
    printf("%d, %d\n", zero_mean_flag, batch);
    bool zero_mean = zero_mean_flag ? true : false;
    QUANTIZE_TYPE type = get_type(type_str);
    
    if ( 0 == strcmp(argv[2], "predict") ) 
      predict_q_classifier(data, cfg, weights, img, type, zero_mean);
    else if ( 0 == strcmp(argv[2], "valid") ) 
      validate_q_classifier(data, cfg, weights, type, zero_mean);
    else if ( 0 == strcmp(argv[2], "quantize") )
      quantize_q_classifier(data, cfg, weights, type, quantize_weights, zero_mean, batch);
}
