This code is based on the [darknet framework](https://pjreddie.com/darknet/), and we add 8-bit floating point quantization method and normalization method into the original framework.

# Imagenet
Please follow the [instructions](https://pjreddie.com/darknet/imagenet/) to download the imagenet datasets and the pre-trained weights.

# Command

All the original running commands are reserved. The added quantization can be run with the following commands:

```shell
./darknet q_classifier method data cfg weights
```
`method`: It has three options.  `quantize` to quantize the network. `predict` to predict with a quantized network. `valid` to do validation with the quantized network.

`data` indicates a file where stores several important information, like validation set, training set, etc.

`cfg` is the darknet configuration file of the network, while `weights` is the darknet weights file of the network.

These five commands are necessary, and we also have some parameters.

`-t` means the quantization type we can use. It includes: `fp32, m0e7, m1e6, m2e5, m3e4, m5e2, m6e1, m7e0, m0e6, m1e5, m2e4, m3e3, m4e2. m5e1, m6e0, m0e5, m1e4, m2e3, m3e2, m4e1, m5e0, m0e4, m1e3, m2e2, m3e1, m4e0, m0e3, m1e2, m2e1, m3e0`. Default: `fp32`, ignore case.

`-qw` means the location to store the quantized weights, only used when we choose the `quantize` method. Default: `./quantize.weights`.

`-b` means the batch size to use when we are doing normalization, only used when we choose the `quantize` method. Default: 1.

`-zm` means whether we assume the mean is zero or not when we are doing normalization. 1: we assume the mean is zero, 0: we calculate the mean according to the data. Default: 1.

`-p` means the location of the input image for test. Only used when we choose the `predict` method. Default: ' '.

Examples:

1. Use type `m3e4` to do quantization of VGG16 and save the quantized weights to `vgg16/m3e4.weigths`. When doing normalization, we use 1 image as input and assume the mean is zero. The command is:

	```
	./darknet q_classifier quantize cfg/imagenet1k.data cfg/vgg16.cfg weights/vgg16.weights -t m3e4 -qw vgg16/m3e4.weights -b 1 -zm 1
	```
2. Do prediction with the quantized network and weights.
	```
	./darkent q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m3e4.weights -t m3e4 -p data/cat1.jpg -zm 1
	```
3. Do validation with the quantized network and weights.
	```
	./darknet q_classifier valid cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m0e7.weights -t m3e4 -zm 1
	```

# Weights

The weights file includes all the weights, bias, and other parameters with respect to different layer types. It also includes the quantization decisions and normalization parameters.

The general formats for the weights file is:

1. The first four parameters are copied from the original darknet framework, I am not very clear about the meanings.
	```
	major, int
	minor, int
	revision, int
	seen, size_t
	```
2. The quantization decision of the input image and the network output, also the normalization parameters, followed with the following format.
	```
	iexp, float
	oexp, float
	mean of the input, float*
	standard variance of the input, float*
	mean of the output, float*
	standard variance of the output, float*
	```
3. The quantization decisions, normalization parameters, and weights of each layer are followed. They are arranged with the following format layer by layer. The meaning of `weights` and `biases` varied according to the layer types.
	- `layer_type = CONVOLUTIONAL || layer_type = CONNECTED`: the weights and biases for convolution layer and fully-connected layer. 
	- `layer_type = CROP`: `crop scale` and `crop trans` for crop layer.
	- `layer_type = SHORTCUT`: `coefficients` and `bias` for residual layer.
	- `layer_type = BATCHNORM`:`scales` and `biases` for batch normalize layer.
	```
	iexp, float
	wexp, float
	bexp, float
	oexp, float
	sexp for batch normalization, float
	bexp for batch normalization, float
	offset, float
	mean of the input, float*
	standard variance of the input, float*
	mean of the output, float*
	standard variance of the output, float*
	weights, float*
	biases, float*
	```

