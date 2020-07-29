#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <string.h>
#include "layers.h"
#include <cassert>

//activation_layer.c
layer make_activation_layer(int batch, int inputs, ACTIVATION activation)
{
	layer l = { };
	l.type = ACTIVE;

	l.inputs = inputs;
	l.outputs = inputs;
	l.batch = batch;


	l.output = (float*)calloc(batch*inputs, sizeof(float));
	l.delta = (float*)calloc(batch*inputs, sizeof(float));

	l.forward = forward_activation_layer;
	l.backward = backward_activation_layer;
#ifdef GPU
	l.forward_gpu = forward_activation_layer_gpu;
	l.backward_gpu = backward_activation_layer_gpu;

	l.output_gpu = cuda_make_array(l.output, inputs*batch);
	l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
#endif
	l.activation = activation;
	fprintf(stderr, "Activation Layer: %d inputs\n", inputs);
	return l;
}

void forward_activation_layer(layer l, network net)
{
	copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
	activate_array(l.output, l.outputs*l.batch, l.activation);
}

void backward_activation_layer(layer l, network net)
{
	gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
	copy_cpu(l.outputs*l.batch, l.delta, 1, net.delta, 1);
}

#ifdef GPU

void forward_activation_layer_gpu(layer l, network net)
{
	copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
	activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void backward_activation_layer_gpu(layer l, network net)
{
	gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
	copy_gpu(l.outputs*l.batch, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif




//activations.c
char *get_activation_string(ACTIVATION a)
{
	switch (a) {
	case LOGISTIC:
		return (char*)"logistic";
	case LOGGY:
		return (char*)"loggy";
	case RELU:
		return (char*)"relu";
	case ELU:
		return (char*)"elu";
	case SELU:
		return (char*)"selu";
	case RELIE:
		return (char*)"relie";
	case RAMP:
		return (char*)"ramp";
	case LINEAR:
		return (char*)"linear";
	case TANH:
		return (char*)"tanh";
	case PLSE:
		return (char*)"plse";
	case LEAKY:
		return (char*)"leaky";
	case STAIR:
		return (char*)"stair";
	case HARDTAN:
		return (char*)"hardtan";
	case LHTAN:
		return (char*)"lhtan";
	default:
		break;
	}
	return (char*)"relu";
}

ACTIVATION get_activation(char *s)
{
	if (strcmp(s, "logistic") == 0) return LOGISTIC;
	if (strcmp(s, "loggy") == 0) return LOGGY;
	if (strcmp(s, "relu") == 0) return RELU;
	if (strcmp(s, "elu") == 0) return ELU;
	if (strcmp(s, "selu") == 0) return SELU;
	if (strcmp(s, "relie") == 0) return RELIE;
	if (strcmp(s, "plse") == 0) return PLSE;
	if (strcmp(s, "hardtan") == 0) return HARDTAN;
	if (strcmp(s, "lhtan") == 0) return LHTAN;
	if (strcmp(s, "linear") == 0) return LINEAR;
	if (strcmp(s, "ramp") == 0) return RAMP;
	if (strcmp(s, "leaky") == 0) return LEAKY;
	if (strcmp(s, "tanh") == 0) return TANH;
	if (strcmp(s, "stair") == 0) return STAIR;
	fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
	return RELU;
}

float activate(float x, ACTIVATION a)
{
	switch (a) {
	case LINEAR:
		return linear_activate(x);
	case LOGISTIC:
		return logistic_activate(x);
	case LOGGY:
		return loggy_activate(x);
	case RELU:
		return relu_activate(x);
	case ELU:
		return elu_activate(x);
	case SELU:
		return selu_activate(x);
	case RELIE:
		return relie_activate(x);
	case RAMP:
		return ramp_activate(x);
	case LEAKY:
		return leaky_activate(x);
	case TANH:
		return tanh_activate(x);
	case PLSE:
		return plse_activate(x);
	case STAIR:
		return stair_activate(x);
	case HARDTAN:
		return hardtan_activate(x);
	case LHTAN:
		return lhtan_activate(x);
	}
	return 0;
}

void activate_array(float *x, const int n, const ACTIVATION a)
{
	int i;
	for (i = 0; i < n; ++i) {
		x[i] = activate(x[i], a);
	}
}

float gradient(float x, ACTIVATION a)
{
	switch (a) {
	case LINEAR:
		return linear_gradient(x);
	case LOGISTIC:
		return logistic_gradient(x);
	case LOGGY:
		return loggy_gradient(x);
	case RELU:
		return relu_gradient(x);
	case ELU:
		return elu_gradient(x);
	case SELU:
		return selu_gradient(x);
	case RELIE:
		return relie_gradient(x);
	case RAMP:
		return ramp_gradient(x);
	case LEAKY:
		return leaky_gradient(x);
	case TANH:
		return tanh_gradient(x);
	case PLSE:
		return plse_gradient(x);
	case STAIR:
		return stair_gradient(x);
	case HARDTAN:
		return hardtan_gradient(x);
	case LHTAN:
		return lhtan_gradient(x);
	}
	return 0;
}

void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta)
{
	int i;
	for (i = 0; i < n; ++i) {
		delta[i] *= gradient(x[i], a);
	}
}




//avgpool_layer.c
avgpool_layer make_avgpool_layer(int batch, int w, int h, int c)
{
	fprintf(stderr, "avg                     %4d x%4d x%4d   ->  %4d\n", w, h, c, c);
	avgpool_layer l = {};
	l.type = AVGPOOL;
	l.batch = batch;
	l.h = h;
	l.w = w;
	l.c = c;
	l.out_w = 1;
	l.out_h = 1;
	l.out_c = c;
	l.outputs = l.out_c;
	l.inputs = h * w*c;
	int output_size = l.outputs * batch;
	l.output = (float*)calloc(output_size, sizeof(float));
	l.delta = (float*)calloc(output_size, sizeof(float));
	l.forward = forward_avgpool_layer;
	l.backward = backward_avgpool_layer;
#ifdef GPU
	l.forward_gpu = forward_avgpool_layer_gpu;
	l.backward_gpu = backward_avgpool_layer_gpu;
	l.output_gpu = cuda_make_array(l.output, output_size);
	l.delta_gpu = cuda_make_array(l.delta, output_size);
#endif
	return l;
}

void resize_avgpool_layer(avgpool_layer *l, int w, int h)
{
	l->w = w;
	l->h = h;
	l->inputs = h * w*l->c;
}

void forward_avgpool_layer(const avgpool_layer l, network net)
{
	int b, i, k;

	for (b = 0; b < l.batch; ++b) {
		for (k = 0; k < l.c; ++k) {
			int out_index = k + b * l.c;
			l.output[out_index] = 0;
			for (i = 0; i < l.h*l.w; ++i) {
				int in_index = i + l.h*l.w*(k + b * l.c);
				l.output[out_index] += net.input[in_index];
			}
			l.output[out_index] /= l.h*l.w;
		}
	}
}

void backward_avgpool_layer(const avgpool_layer l, network net)
{
	int b, i, k;

	for (b = 0; b < l.batch; ++b) {
		for (k = 0; k < l.c; ++k) {
			int out_index = k + b * l.c;
			for (i = 0; i < l.h*l.w; ++i) {
				int in_index = i + l.h*l.w*(k + b * l.c);
				net.delta[in_index] += l.delta[out_index] / (l.h*l.w);
			}
		}
	}
}





//batchnorm_layer.c
layer make_batchnorm_layer(int batch, int w, int h, int c)
{
	fprintf(stderr, "Batch Normalization Layer: %d x %d x %d image\n", w, h, c);
	layer l = { };
	l.type = BATCHNORM;
	l.batch = batch;
	l.h = l.out_h = h;
	l.w = l.out_w = w;
	l.c = l.out_c = c;
	l.output = (float*)calloc(h * w * c * batch, sizeof(float));
	l.delta = (float*)calloc(h * w * c * batch, sizeof(float));
	l.inputs = w * h*c;
	l.outputs = l.inputs;

	l.scales = (float*)calloc(c, sizeof(float));
	l.scale_updates = (float*)calloc(c, sizeof(float));
	l.biases = (float*)calloc(c, sizeof(float));
	l.bias_updates = (float*)calloc(c, sizeof(float));
	int i;
	for (i = 0; i < c; ++i) {
		l.scales[i] = 1;
	}

	l.mean = (float*)calloc(c, sizeof(float));
	l.variance = (float*)calloc(c, sizeof(float));

	l.rolling_mean = (float*)calloc(c, sizeof(float));
	l.rolling_variance = (float*)calloc(c, sizeof(float));

	l.forward = forward_batchnorm_layer;
	l.backward = backward_batchnorm_layer;
#ifdef GPU
	l.forward_gpu = forward_batchnorm_layer_gpu;
	l.backward_gpu = backward_batchnorm_layer_gpu;

	l.output_gpu = cuda_make_array(l.output, h * w * c * batch);
	l.delta_gpu = cuda_make_array(l.delta, h * w * c * batch);

	l.biases_gpu = cuda_make_array(l.biases, c);
	l.bias_updates_gpu = cuda_make_array(l.bias_updates, c);

	l.scales_gpu = cuda_make_array(l.scales, c);
	l.scale_updates_gpu = cuda_make_array(l.scale_updates, c);

	l.mean_gpu = cuda_make_array(l.mean, c);
	l.variance_gpu = cuda_make_array(l.variance, c);

	l.rolling_mean_gpu = cuda_make_array(l.mean, c);
	l.rolling_variance_gpu = cuda_make_array(l.variance, c);

	l.mean_delta_gpu = cuda_make_array(l.mean, c);
	l.variance_delta_gpu = cuda_make_array(l.variance, c);

	l.x_gpu = cuda_make_array(l.output, l.batch*l.outputs);
	l.x_norm_gpu = cuda_make_array(l.output, l.batch*l.outputs);
#ifdef CUDNN
	cudnnCreateTensorDescriptor(&l.normTensorDesc);
	cudnnCreateTensorDescriptor(&l.dstTensorDesc);
	cudnnSetTensor4dDescriptor(l.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.out_c, l.out_h, l.out_w);
	cudnnSetTensor4dDescriptor(l.normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l.out_c, 1, 1);

#endif
#endif
	return l;
}

void backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates)
{
	int i, b, f;
	for (f = 0; f < n; ++f) {
		float sum = 0;
		for (b = 0; b < batch; ++b) {
			for (i = 0; i < size; ++i) {
				int index = i + size * (f + n * b);
				sum += delta[index] * x_norm[index];
			}
		}
		scale_updates[f] += sum;
	}
}

void mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{

	int i, j, k;
	for (i = 0; i < filters; ++i) {
		mean_delta[i] = 0;
		for (j = 0; j < batch; ++j) {
			for (k = 0; k < spatial; ++k) {
				int index = j * filters*spatial + i * spatial + k;
				mean_delta[i] += delta[index];
			}
		}
		mean_delta[i] *= float((-1. / sqrt(variance[i] + .00001f)));
	}
}
void  variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{

	int i, j, k;
	for (i = 0; i < filters; ++i) {
		variance_delta[i] = 0;
		for (j = 0; j < batch; ++j) {
			for (k = 0; k < spatial; ++k) {
				int index = j * filters*spatial + i * spatial + k;
				variance_delta[i] += delta[index] * (x[index] - mean[i]);
			}
		}
		variance_delta[i] *= (float)(-.5 * pow(variance[i] + .00001f, (float)(-3. / 2.)));
	}
}
void normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta)
{
	int f, j, k;
	for (j = 0; j < batch; ++j) {
		for (f = 0; f < filters; ++f) {
			for (k = 0; k < spatial; ++k) {
				int index = j * filters*spatial + f * spatial + k;
				delta[index] = (float)(delta[index] * 1. / (sqrt(variance[f] + .00001f)) + 
					variance_delta[f] * 2. * (x[index] - mean[f]) / (spatial * batch) + 
					mean_delta[f] / (spatial*batch));
			}
		}
	}
}

void resize_batchnorm_layer(layer *layer, int w, int h)
{
	fprintf(stderr, "Not implemented\n");
}

void forward_batchnorm_layer(layer l, network net)
{
	if (l.type == BATCHNORM) copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
	copy_cpu(l.outputs*l.batch, l.output, 1, l.x, 1);
	if (net.train) {
		mean_cpu(l.output, l.batch, l.out_c, l.out_h*l.out_w, l.mean);
		variance_cpu(l.output, l.mean, l.batch, l.out_c, l.out_h*l.out_w, l.variance);

		scal_cpu(l.out_c, (float).99, l.rolling_mean, 1);
		axpy_cpu(l.out_c, (float).01, l.mean, 1, l.rolling_mean, 1);
		scal_cpu(l.out_c, (float).99, l.rolling_variance, 1);
		axpy_cpu(l.out_c, (float).01, l.variance, 1, l.rolling_variance, 1);

		normalize_cpu(l.output, l.mean, l.variance, l.batch, l.out_c, l.out_h*l.out_w);
		copy_cpu(l.outputs*l.batch, l.output, 1, l.x_norm, 1);
	}
	else {
		normalize_cpu(l.output, l.rolling_mean, l.rolling_variance, l.batch, l.out_c, l.out_h*l.out_w);
	}
	scale_bias(l.output, l.scales, l.batch, l.out_c, l.out_h*l.out_w);
	add_bias(l.output, l.biases, l.batch, l.out_c, l.out_h*l.out_w);
}

void backward_batchnorm_layer(layer l, network net)
{
	if (!net.train) {
		l.mean = l.rolling_mean;
		l.variance = l.rolling_variance;
	}
	backward_bias(l.bias_updates, l.delta, l.batch, l.out_c, l.out_w*l.out_h);
	backward_scale_cpu(l.x_norm, l.delta, l.batch, l.out_c, l.out_w*l.out_h, l.scale_updates);

	scale_bias(l.delta, l.scales, l.batch, l.out_c, l.out_h*l.out_w);

	mean_delta_cpu(l.delta, l.variance, l.batch, l.out_c, l.out_w*l.out_h, l.mean_delta);
	variance_delta_cpu(l.x, l.delta, l.mean, l.variance, l.batch, l.out_c, l.out_w*l.out_h, l.variance_delta);
	normalize_delta_cpu(l.x, l.mean, l.variance, l.mean_delta, l.variance_delta, l.batch, l.out_c, l.out_w*l.out_h, l.delta);
	if (l.type == BATCHNORM) copy_cpu(l.outputs*l.batch, l.delta, 1, net.delta, 1);
}

#ifdef GPU

void pull_batchnorm_layer(layer l)
{
	cuda_pull_array(l.scales_gpu, l.scales, l.c);
	cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.c);
	cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.c);
}
void push_batchnorm_layer(layer l)
{
	cuda_push_array(l.scales_gpu, l.scales, l.c);
	cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.c);
	cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.c);
}

void forward_batchnorm_layer_gpu(layer l, network net)
{
	if (l.type == BATCHNORM) copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
	copy_gpu(l.outputs*l.batch, l.output_gpu, 1, l.x_gpu, 1);
	if (net.train) {
#ifdef CUDNN
		float one = 1;
		float zero = 0;
		cudnnBatchNormalizationForwardTraining(cudnn_handle(),
			CUDNN_BATCHNORM_SPATIAL,
			&one,
			&zero,
			l.dstTensorDesc,
			l.x_gpu,
			l.dstTensorDesc,
			l.output_gpu,
			l.normTensorDesc,
			l.scales_gpu,
			l.biases_gpu,
			.01,
			l.rolling_mean_gpu,
			l.rolling_variance_gpu,
			.00001,
			l.mean_gpu,
			l.variance_gpu);
#else
		fast_mean_gpu(l.output_gpu, l.batch, l.out_c, l.out_h*l.out_w, l.mean_gpu);
		fast_variance_gpu(l.output_gpu, l.mean_gpu, l.batch, l.out_c, l.out_h*l.out_w, l.variance_gpu);

		scal_gpu(l.out_c, .99, l.rolling_mean_gpu, 1);
		axpy_gpu(l.out_c, .01, l.mean_gpu, 1, l.rolling_mean_gpu, 1);
		scal_gpu(l.out_c, .99, l.rolling_variance_gpu, 1);
		axpy_gpu(l.out_c, .01, l.variance_gpu, 1, l.rolling_variance_gpu, 1);

		copy_gpu(l.outputs*l.batch, l.output_gpu, 1, l.x_gpu, 1);
		normalize_gpu(l.output_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
		copy_gpu(l.outputs*l.batch, l.output_gpu, 1, l.x_norm_gpu, 1);

		scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
		add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.out_c, l.out_w*l.out_h);
#endif
	}
	else {
		normalize_gpu(l.output_gpu, l.rolling_mean_gpu, l.rolling_variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
		scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
		add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.out_c, l.out_w*l.out_h);
	}

}

void backward_batchnorm_layer_gpu(layer l, network net)
{
	if (!net.train) {
		l.mean_gpu = l.rolling_mean_gpu;
		l.variance_gpu = l.rolling_variance_gpu;
	}
#ifdef CUDNN
	float one = 1;
	float zero = 0;
	cudnnBatchNormalizationBackward(cudnn_handle(),
		CUDNN_BATCHNORM_SPATIAL,
		&one,
		&zero,
		&one,
		&one,
		l.dstTensorDesc,
		l.x_gpu,
		l.dstTensorDesc,
		l.delta_gpu,
		l.dstTensorDesc,
		l.x_norm_gpu,
		l.normTensorDesc,
		l.scales_gpu,
		l.scale_updates_gpu,
		l.bias_updates_gpu,
		.00001,
		l.mean_gpu,
		l.variance_gpu);
	copy_gpu(l.outputs*l.batch, l.x_norm_gpu, 1, l.delta_gpu, 1);
#else
	backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.out_c, l.out_w*l.out_h);
	backward_scale_gpu(l.x_norm_gpu, l.delta_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.scale_updates_gpu);

	scale_bias_gpu(l.delta_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);

	fast_mean_delta_gpu(l.delta_gpu, l.variance_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.mean_delta_gpu);
	fast_variance_delta_gpu(l.x_gpu, l.delta_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.variance_delta_gpu);
	normalize_delta_gpu(l.x_gpu, l.mean_gpu, l.variance_gpu, l.mean_delta_gpu, l.variance_delta_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.delta_gpu);
#endif
	if (l.type == BATCHNORM) copy_gpu(l.outputs*l.batch, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif







//connected_layer.c
layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam)
{
	int i;
	layer l = {};
	l.learning_rate_scale = 1;
	l.type = CONNECTED;

	l.inputs = inputs;
	l.outputs = outputs;
	l.batch = batch;
	l.batch_normalize = batch_normalize;
	l.h = 1;
	l.w = 1;
	l.c = inputs;
	l.out_h = 1;
	l.out_w = 1;
	l.out_c = outputs;

	l.output = (float*)calloc(batch*outputs, sizeof(float));
	l.delta = (float*)calloc(batch*outputs, sizeof(float));

	l.weight_updates = (float*)calloc(inputs*outputs, sizeof(float));
	l.bias_updates = (float*)calloc(outputs, sizeof(float));

	l.weights = (float*)calloc(outputs*inputs, sizeof(float));
	l.biases = (float*)calloc(outputs, sizeof(float));

	l.forward = forward_connected_layer;
	l.backward = backward_connected_layer;
	l.update = update_connected_layer;

	//float scale = 1./sqrt(inputs);
	float scale = (float)(sqrt(2. / inputs));
	for (i = 0; i < outputs*inputs; ++i) {
		l.weights[i] = scale * rand_uniform(-1, 1);
	}

	for (i = 0; i < outputs; ++i) {
		l.biases[i] = 0;
	}

	if (adam) {
		l.m = (float*)calloc(l.inputs*l.outputs, sizeof(float));
		l.v = (float*)calloc(l.inputs*l.outputs, sizeof(float));
		l.bias_m = (float*)calloc(l.outputs, sizeof(float));
		l.scale_m = (float*)calloc(l.outputs, sizeof(float));
		l.bias_v = (float*)calloc(l.outputs, sizeof(float));
		l.scale_v = (float*)calloc(l.outputs, sizeof(float));
	}
	if (batch_normalize) {
		l.scales = (float*)calloc(outputs, sizeof(float));
		l.scale_updates = (float*)calloc(outputs, sizeof(float));
		for (i = 0; i < outputs; ++i) {
			l.scales[i] = 1;
		}

		l.mean = (float*)calloc(outputs, sizeof(float));
		l.mean_delta = (float*)calloc(outputs, sizeof(float));
		l.variance = (float*)calloc(outputs, sizeof(float));
		l.variance_delta = (float*)calloc(outputs, sizeof(float));

		l.rolling_mean = (float*)calloc(outputs, sizeof(float));
		l.rolling_variance = (float*)calloc(outputs, sizeof(float));

		l.x = (float*)calloc(batch*outputs, sizeof(float));
		l.x_norm = (float*)calloc(batch*outputs, sizeof(float));
	}

#ifdef GPU
	l.forward_gpu = forward_connected_layer_gpu;
	l.backward_gpu = backward_connected_layer_gpu;
	l.update_gpu = update_connected_layer_gpu;

	l.weights_gpu = cuda_make_array(l.weights, outputs*inputs);
	l.biases_gpu = cuda_make_array(l.biases, outputs);

	l.weight_updates_gpu = cuda_make_array(l.weight_updates, outputs*inputs);
	l.bias_updates_gpu = cuda_make_array(l.bias_updates, outputs);

	l.output_gpu = cuda_make_array(l.output, outputs*batch);
	l.delta_gpu = cuda_make_array(l.delta, outputs*batch);
	if (adam) {
		l.m_gpu = cuda_make_array(0, inputs*outputs);
		l.v_gpu = cuda_make_array(0, inputs*outputs);
		l.bias_m_gpu = cuda_make_array(0, outputs);
		l.bias_v_gpu = cuda_make_array(0, outputs);
		l.scale_m_gpu = cuda_make_array(0, outputs);
		l.scale_v_gpu = cuda_make_array(0, outputs);
	}

	if (batch_normalize) {
		l.mean_gpu = cuda_make_array(l.mean, outputs);
		l.variance_gpu = cuda_make_array(l.variance, outputs);

		l.rolling_mean_gpu = cuda_make_array(l.mean, outputs);
		l.rolling_variance_gpu = cuda_make_array(l.variance, outputs);

		l.mean_delta_gpu = cuda_make_array(l.mean, outputs);
		l.variance_delta_gpu = cuda_make_array(l.variance, outputs);

		l.scales_gpu = cuda_make_array(l.scales, outputs);
		l.scale_updates_gpu = cuda_make_array(l.scale_updates, outputs);

		l.x_gpu = cuda_make_array(l.output, l.batch*outputs);
		l.x_norm_gpu = cuda_make_array(l.output, l.batch*outputs);
#ifdef CUDNN
		cudnnCreateTensorDescriptor(&l.normTensorDesc);
		cudnnCreateTensorDescriptor(&l.dstTensorDesc);
		cudnnSetTensor4dDescriptor(l.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.out_c, l.out_h, l.out_w);
		cudnnSetTensor4dDescriptor(l.normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l.out_c, 1, 1);
#endif
	}
#endif
	l.activation = activation;
	fprintf(stderr, "connected                            %4d  ->  %4d\n", inputs, outputs);
	return l;
}

void update_connected_layer(layer l, update_args a)
{
	float learning_rate = a.learning_rate*l.learning_rate_scale;
	float momentum = a.momentum;
	float decay = a.decay;
	int batch = a.batch;
	axpy_cpu(l.outputs, learning_rate / batch, l.bias_updates, 1, l.biases, 1);
	scal_cpu(l.outputs, momentum, l.bias_updates, 1);

	if (l.batch_normalize) {
		axpy_cpu(l.outputs, learning_rate / batch, l.scale_updates, 1, l.scales, 1);
		scal_cpu(l.outputs, momentum, l.scale_updates, 1);
	}

	axpy_cpu(l.inputs*l.outputs, -decay * batch, l.weights, 1, l.weight_updates, 1);
	axpy_cpu(l.inputs*l.outputs, learning_rate / batch, l.weight_updates, 1, l.weights, 1);
	scal_cpu(l.inputs*l.outputs, momentum, l.weight_updates, 1);
}

void forward_connected_layer(layer l, network net)
{
	fill_cpu(l.outputs*l.batch, 0, l.output, 1);
	int m = l.batch;
	int k = l.inputs;
	int n = l.outputs;
	float *a = net.input;
	float *b = l.weights;
	float *c = l.output;
	gemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);
	if (l.batch_normalize) {
		forward_batchnorm_layer(l, net);
	}
	else {
		add_bias(l.output, l.biases, l.batch, l.outputs, 1);
	}
	activate_array(l.output, l.outputs*l.batch, l.activation);
}

void backward_connected_layer(layer l, network net)
{
	gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

	if (l.batch_normalize) {
		backward_batchnorm_layer(l, net);
	}
	else {
		backward_bias(l.bias_updates, l.delta, l.batch, l.outputs, 1);
	}

	int m = l.outputs;
	int k = l.batch;
	int n = l.inputs;
	float *a = l.delta;
	float *b = net.input;
	float *c = l.weight_updates;
	gemm(1, 0, m, n, k, 1, a, m, b, n, 1, c, n);

	m = l.batch;
	k = l.outputs;
	n = l.inputs;

	a = l.delta;
	b = l.weights;
	c = net.delta;

	if (c) gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
}


void denormalize_connected_layer(layer l)
{
	int i, j;
	for (i = 0; i < l.outputs; ++i) {
		float scale = (float)(l.scales[i] / sqrt(l.rolling_variance[i] + .000001));
		for (j = 0; j < l.inputs; ++j) {
			l.weights[i*l.inputs + j] *= scale;
		}
		l.biases[i] -= l.rolling_mean[i] * scale;
		l.scales[i] = 1;
		l.rolling_mean[i] = 0;
		l.rolling_variance[i] = 1;
	}
}


void statistics_connected_layer(layer l)
{
	if (l.batch_normalize) {
		printf("Scales ");
		print_statistics(l.scales, l.outputs);
		/*
		   printf("Rolling Mean ");
		   print_statistics(l.rolling_mean, l.outputs);
		   printf("Rolling Variance ");
		   print_statistics(l.rolling_variance, l.outputs);
		 */
	}
	printf("Biases ");
	print_statistics(l.biases, l.outputs);
	printf("Weights ");
	print_statistics(l.weights, l.outputs);
}

#ifdef GPU

void pull_connected_layer(layer l)
{
	cuda_pull_array(l.weights_gpu, l.weights, l.inputs*l.outputs);
	cuda_pull_array(l.biases_gpu, l.biases, l.outputs);
	cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.inputs*l.outputs);
	cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
	if (l.batch_normalize) {
		cuda_pull_array(l.scales_gpu, l.scales, l.outputs);
		cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.outputs);
		cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.outputs);
	}
}

void push_connected_layer(layer l)
{
	cuda_push_array(l.weights_gpu, l.weights, l.inputs*l.outputs);
	cuda_push_array(l.biases_gpu, l.biases, l.outputs);
	cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.inputs*l.outputs);
	cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
	if (l.batch_normalize) {
		cuda_push_array(l.scales_gpu, l.scales, l.outputs);
		cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.outputs);
		cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.outputs);
	}
}

void update_connected_layer_gpu(layer l, update_args a)
{
	float learning_rate = a.learning_rate*l.learning_rate_scale;
	float momentum = a.momentum;
	float decay = a.decay;
	int batch = a.batch;
	if (a.adam) {
		adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.inputs*l.outputs, batch, a.t);
		adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.outputs, batch, a.t);
		if (l.scales_gpu) {
			adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.outputs, batch, a.t);
		}
	}
	else {
		axpy_gpu(l.outputs, learning_rate / batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
		scal_gpu(l.outputs, momentum, l.bias_updates_gpu, 1);

		if (l.batch_normalize) {
			axpy_gpu(l.outputs, learning_rate / batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
			scal_gpu(l.outputs, momentum, l.scale_updates_gpu, 1);
		}

		axpy_gpu(l.inputs*l.outputs, -decay * batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
		axpy_gpu(l.inputs*l.outputs, learning_rate / batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
		scal_gpu(l.inputs*l.outputs, momentum, l.weight_updates_gpu, 1);
	}
}

void forward_connected_layer_gpu(layer l, network net)
{
	fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);

	int m = l.batch;
	int k = l.inputs;
	int n = l.outputs;
	float * a = net.input_gpu;
	float * b = l.weights_gpu;
	float * c = l.output_gpu;
	gemm_gpu(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);

	if (l.batch_normalize) {
		forward_batchnorm_layer_gpu(l, net);
	}
	else {
		add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.outputs, 1);
	}
	activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void backward_connected_layer_gpu(layer l, network net)
{
	constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
	gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
	if (l.batch_normalize) {
		backward_batchnorm_layer_gpu(l, net);
	}
	else {
		backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.outputs, 1);
	}

	int m = l.outputs;
	int k = l.batch;
	int n = l.inputs;
	float * a = l.delta_gpu;
	float * b = net.input_gpu;
	float * c = l.weight_updates_gpu;
	gemm_gpu(1, 0, m, n, k, 1, a, m, b, n, 1, c, n);

	m = l.batch;
	k = l.outputs;
	n = l.inputs;

	a = l.delta_gpu;
	b = l.weights_gpu;
	c = net.delta_gpu;

	if (c) gemm_gpu(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
}
#endif







//convolutional_layer.c
#ifdef AI2
#include "xnor_layer.h"
#endif

void swap_binary(convolutional_layer *l)
{
	float *swap = l->weights;
	l->weights = l->binary_weights;
	l->binary_weights = swap;

#ifdef GPU
	swap = l->weights_gpu;
	l->weights_gpu = l->binary_weights_gpu;
	l->binary_weights_gpu = swap;
#endif
}

void binarize_weights(float *weights, int n, int size, float *binary)
{
	int i, f;
	for (f = 0; f < n; ++f) {
		float mean = 0;
		for (i = 0; i < size; ++i) {
			mean += fabs(weights[f*size + i]);
		}
		mean = mean / size;
		for (i = 0; i < size; ++i) {
			binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
		}
	}
}

void binarize_cpu(float *input, int n, float *binary)
{
	int i;
	for (i = 0; i < n; ++i) {
		binary[i] = (float)((input[i] > 0) ? 1 : -1);
	}
}

void binarize_input(float *input, int n, int size, float *binary)
{
	int i, s;
	for (s = 0; s < size; ++s) {
		float mean = 0;
		for (i = 0; i < n; ++i) {
			mean += fabs(input[i*size + s]);
		}
		mean = mean / n;
		for (i = 0; i < n; ++i) {
			binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
		}
	}
}

int convolutional_out_height(convolutional_layer l)
{
	return (l.h + 2 * l.pad - l.size) / l.stride + 1;
}

int convolutional_out_width(convolutional_layer l)
{
	return (l.w + 2 * l.pad - l.size) / l.stride + 1;
}

image get_convolutional_image(convolutional_layer l)
{
	return float_to_image(l.out_w, l.out_h, l.out_c, l.output);
}

image get_convolutional_delta(convolutional_layer l)
{
	return float_to_image(l.out_w, l.out_h, l.out_c, l.delta);
}

static size_t get_workspace_size(layer l) {
#ifdef CUDNN
	if (gpu_index >= 0) {
		size_t most = 0;
		size_t s = 0;
		cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
			l.srcTensorDesc,
			l.weightDesc,
			l.convDesc,
			l.dstTensorDesc,
			l.fw_algo,
			&s);
		if (s > most) most = s;
		cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(),
			l.srcTensorDesc,
			l.ddstTensorDesc,
			l.convDesc,
			l.dweightDesc,
			l.bf_algo,
			&s);
		if (s > most) most = s;
		cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(),
			l.weightDesc,
			l.ddstTensorDesc,
			l.convDesc,
			l.dsrcTensorDesc,
			l.bd_algo,
			&s);
		if (s > most) most = s;
		return most;
	}
#endif
	return (size_t)l.out_h*l.out_w*l.size*l.size*l.c / l.groups * sizeof(float);
}

#ifdef GPU
#ifdef CUDNN
void cudnn_convolutional_setup(layer *l)
{
	cudnnSetTensor4dDescriptor(l->dsrcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w);
	cudnnSetTensor4dDescriptor(l->ddstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w);

	cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w);
	cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w);
	cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1);

	cudnnSetFilter4dDescriptor(l->dweightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c / l->groups, l->size, l->size);
	cudnnSetFilter4dDescriptor(l->weightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c / l->groups, l->size, l->size);
#if CUDNN_MAJOR >= 6
	cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
#else
	cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION);
#endif

#if CUDNN_MAJOR >= 7
	cudnnSetConvolutionGroupCount(l->convDesc, l->groups);
#else
	if (l->groups > 1) {
		error("CUDNN < 7 doesn't support groups, please upgrade!");
	}
#endif

	cudnnGetConvolutionForwardAlgorithm(cudnn_handle(),
		l->srcTensorDesc,
		l->weightDesc,
		l->convDesc,
		l->dstTensorDesc,
		CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
		2000000000,
		&l->fw_algo);
	cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle(),
		l->weightDesc,
		l->ddstTensorDesc,
		l->convDesc,
		l->dsrcTensorDesc,
		CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
		2000000000,
		&l->bd_algo);
	cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle(),
		l->srcTensorDesc,
		l->ddstTensorDesc,
		l->convDesc,
		l->dweightDesc,
		CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
		2000000000,
		&l->bf_algo);
}
#endif
#endif

//Construct The convolutional_layer
convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, 
	int groups, int size, int stride, int padding, ACTIVATION activation, 
	int batch_normalize, int binary, int xnor, int adam)
{
	int i;
	convolutional_layer l = {};
	l.type = CONVOLUTIONAL;

	//Read in the parameters
	l.groups = groups;
	l.h = h;
	l.w = w;
	l.c = c;
	l.n = n;
	l.binary = binary;
	l.xnor = xnor;
	l.batch = batch;
	l.stride = stride;
	l.size = size;
	l.pad = padding;
	l.batch_normalize = batch_normalize;

	l.weights = (float*)calloc(c / groups * n*size*size, sizeof(float));
	l.weight_updates = (float*)calloc(c / groups * n*size*size, sizeof(float));

	l.biases = (float*)calloc(n, sizeof(float));
	l.bias_updates = (float*)calloc(n, sizeof(float));

	l.nweights = c / groups * n*size*size;
	l.nbiases = n;

	// float scale = 1./sqrt(size*size*c);
	float scale = (float)(sqrt(2. / (size*size*c / l.groups)));
	//printf("convscale %f\n", scale);
	//scale = .02;
	//for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_uniform(-1, 1);
	for (i = 0; i < l.nweights; ++i) l.weights[i] = scale * rand_normal();
	int out_w = convolutional_out_width(l);
	int out_h = convolutional_out_height(l);
	l.out_h = out_h;
	l.out_w = out_w;
	l.out_c = n;
	l.outputs = l.out_h * l.out_w * l.out_c;
	l.inputs = l.w * l.h * l.c;

	l.output = (float*)calloc(l.batch*l.outputs, sizeof(float));
	l.delta = (float*)calloc(l.batch*l.outputs, sizeof(float));

	l.forward = forward_convolutional_layer;
	l.backward = backward_convolutional_layer;
	l.update = update_convolutional_layer;
	if (binary) {
		l.binary_weights = (float*)calloc(l.nweights, sizeof(float));
		l.cweights = (char*)calloc(l.nweights, sizeof(char));
		l.scales = (float*)calloc(n, sizeof(float));
	}
	if (xnor) {
		l.binary_weights = (float*)calloc(l.nweights, sizeof(float));
		l.binary_input = (float*)calloc(l.inputs*l.batch, sizeof(float));
	}

	if (batch_normalize) {
		l.scales = (float*)calloc(n, sizeof(float));
		l.scale_updates = (float*)calloc(n, sizeof(float));
		for (i = 0; i < n; ++i) {
			l.scales[i] = 1;
		}

		l.mean = (float*)calloc(n, sizeof(float));
		l.variance = (float*)calloc(n, sizeof(float));

		l.mean_delta = (float*)calloc(n, sizeof(float));
		l.variance_delta = (float*)calloc(n, sizeof(float));

		l.rolling_mean = (float*)calloc(n, sizeof(float));
		l.rolling_variance = (float*)calloc(n, sizeof(float));
		l.x = (float*)calloc(l.batch*l.outputs, sizeof(float));
		l.x_norm = (float*)calloc(l.batch*l.outputs, sizeof(float));
	}
	if (adam) {
		l.m = (float*)calloc(l.nweights, sizeof(float));
		l.v = (float*)calloc(l.nweights, sizeof(float));
		l.bias_m = (float*)calloc(n, sizeof(float));
		l.scale_m = (float*)calloc(n, sizeof(float));
		l.bias_v = (float*)calloc(n, sizeof(float));
		l.scale_v = (float*)calloc(n, sizeof(float));
	}

#ifdef GPU
	l.forward_gpu = forward_convolutional_layer_gpu;
	l.backward_gpu = backward_convolutional_layer_gpu;
	l.update_gpu = update_convolutional_layer_gpu;

	if (gpu_index >= 0) {
		if (adam) {
			l.m_gpu = cuda_make_array(l.m, l.nweights);
			l.v_gpu = cuda_make_array(l.v, l.nweights);
			l.bias_m_gpu = cuda_make_array(l.bias_m, n);
			l.bias_v_gpu = cuda_make_array(l.bias_v, n);
			l.scale_m_gpu = cuda_make_array(l.scale_m, n);
			l.scale_v_gpu = cuda_make_array(l.scale_v, n);
		}

		l.weights_gpu = cuda_make_array(l.weights, l.nweights);
		l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.nweights);

		l.biases_gpu = cuda_make_array(l.biases, n);
		l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

		l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*n);
		l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);

		if (binary) {
			l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
		}
		if (xnor) {
			l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
			l.binary_input_gpu = cuda_make_array(0, l.inputs*l.batch);
		}

		if (batch_normalize) {
			l.mean_gpu = cuda_make_array(l.mean, n);
			l.variance_gpu = cuda_make_array(l.variance, n);

			l.rolling_mean_gpu = cuda_make_array(l.mean, n);
			l.rolling_variance_gpu = cuda_make_array(l.variance, n);

			l.mean_delta_gpu = cuda_make_array(l.mean, n);
			l.variance_delta_gpu = cuda_make_array(l.variance, n);

			l.scales_gpu = cuda_make_array(l.scales, n);
			l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);

			l.x_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
			l.x_norm_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
		}
#ifdef CUDNN
		cudnnCreateTensorDescriptor(&l.normTensorDesc);
		cudnnCreateTensorDescriptor(&l.srcTensorDesc);
		cudnnCreateTensorDescriptor(&l.dstTensorDesc);
		cudnnCreateFilterDescriptor(&l.weightDesc);
		cudnnCreateTensorDescriptor(&l.dsrcTensorDesc);
		cudnnCreateTensorDescriptor(&l.ddstTensorDesc);
		cudnnCreateFilterDescriptor(&l.dweightDesc);
		cudnnCreateConvolutionDescriptor(&l.convDesc);
		cudnn_convolutional_setup(&l);
#endif
	}
#endif
	l.workspace_size = get_workspace_size(l);
	l.activation = activation;

	//Print the layer information
	fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", 
		n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c, 
		(2.0 * l.n * l.size*l.size*l.c / l.groups * l.out_h*l.out_w) / 1000000000.);

	return l;
}

void denormalize_convolutional_layer(convolutional_layer l)
{
	int i, j;
	for (i = 0; i < l.n; ++i) {
		float scale = (float)(l.scales[i] / sqrt(l.rolling_variance[i] + .00001));
		for (j = 0; j < l.c / l.groups*l.size*l.size; ++j) {
			l.weights[i*l.c / l.groups*l.size*l.size + j] *= scale;
		}
		l.biases[i] -= l.rolling_mean[i] * scale;
		l.scales[i] = 1;
		l.rolling_mean[i] = 0;
		l.rolling_variance[i] = 1;
	}
}

/*
void test_convolutional_layer()
{
	convolutional_layer l = make_convolutional_layer(1, 5, 5, 3, 2, 5, 2, 1, LEAKY, 1, 0, 0, 0);
	l.batch_normalize = 1;
	float data[] = {1,1,1,1,1,
		1,1,1,1,1,
		1,1,1,1,1,
		1,1,1,1,1,
		1,1,1,1,1,
		2,2,2,2,2,
		2,2,2,2,2,
		2,2,2,2,2,
		2,2,2,2,2,
		2,2,2,2,2,
		3,3,3,3,3,
		3,3,3,3,3,
		3,3,3,3,3,
		3,3,3,3,3,
		3,3,3,3,3};
	//net.input = data;
	//forward_convolutional_layer(l);
}
*/

void resize_convolutional_layer(convolutional_layer *l, int w, int h)
{
	l->w = w;
	l->h = h;
	int out_w = convolutional_out_width(*l);
	int out_h = convolutional_out_height(*l);

	l->out_w = out_w;
	l->out_h = out_h;

	l->outputs = l->out_h * l->out_w * l->out_c;
	l->inputs = l->w * l->h * l->c;

	l->output = (float*)realloc(l->output, l->batch*l->outputs * sizeof(float));
	l->delta = (float*)realloc(l->delta, l->batch*l->outputs * sizeof(float));
	if (l->batch_normalize) {
		l->x = (float*)realloc(l->x, l->batch*l->outputs * sizeof(float));
		l->x_norm = (float*)realloc(l->x_norm, l->batch*l->outputs * sizeof(float));
	}

#ifdef GPU
	cuda_free(l->delta_gpu);
	cuda_free(l->output_gpu);

	l->delta_gpu = cuda_make_array(l->delta, l->batch*l->outputs);
	l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);

	if (l->batch_normalize) {
		cuda_free(l->x_gpu);
		cuda_free(l->x_norm_gpu);

		l->x_gpu = cuda_make_array(l->output, l->batch*l->outputs);
		l->x_norm_gpu = cuda_make_array(l->output, l->batch*l->outputs);
	}
#ifdef CUDNN
	cudnn_convolutional_setup(l);
#endif
#endif
	l->workspace_size = get_workspace_size(*l);
}

void add_bias(float *output, float *biases, int batch, int n, int size)
{
	int i, j, b;
	for (b = 0; b < batch; ++b) {
		for (i = 0; i < n; ++i) {
			for (j = 0; j < size; ++j) {
				output[(b*n + i)*size + j] += biases[i];
			}
		}
	}
}

void scale_bias(float *output, float *scales, int batch, int n, int size)
{
	int i, j, b;
	for (b = 0; b < batch; ++b) {
		for (i = 0; i < n; ++i) {
			for (j = 0; j < size; ++j) {
				output[(b*n + i)*size + j] *= scales[i];
			}
		}
	}
}

void backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
	int i, b;
	for (b = 0; b < batch; ++b) {
		for (i = 0; i < n; ++i) {
			bias_updates[i] += sum_array(delta + size * (i + b * n), size);
		}
	}
}

void forward_convolutional_layer(convolutional_layer l, network net)
{
	int i, j;

	fill_cpu(l.outputs*l.batch, 0, l.output, 1);

	if (l.xnor) {
		binarize_weights(l.weights, l.n, l.c / l.groups*l.size*l.size, l.binary_weights);
		swap_binary(&l);
		binarize_cpu(net.input, l.c*l.h*l.w*l.batch, l.binary_input);
		net.input = l.binary_input;
	}

	int m = l.n / l.groups;
	int k = l.size*l.size*l.c / l.groups;
	int n = l.out_w*l.out_h;
	for (i = 0; i < l.batch; ++i) {
		for (j = 0; j < l.groups; ++j) {
			float *a = l.weights + j * l.nweights / l.groups;
			float *b = net.workspace;
			float *c = l.output + (i*l.groups + j)*n*m;
			float *im = net.input + (i*l.groups + j)*l.c / l.groups*l.h*l.w;

			if (l.size == 1) {
				b = im;
			}
			else {
				im2col_cpu(im, l.c / l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
			}
			gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
		}
	}

	if (l.batch_normalize) {
		forward_batchnorm_layer(l, net);
	}
	else {
		add_bias(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);
	}

	activate_array(l.output, l.outputs*l.batch, l.activation);
	if (l.binary || l.xnor) swap_binary(&l);
}

void backward_convolutional_layer(convolutional_layer l, network net)
{
	int i, j;
	int m = l.n / l.groups;
	int n = l.size*l.size*l.c / l.groups;
	int k = l.out_w*l.out_h;

	gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

	if (l.batch_normalize) {
		backward_batchnorm_layer(l, net);
	}
	else {
		backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);
	}

	for (i = 0; i < l.batch; ++i) {
		for (j = 0; j < l.groups; ++j) {
			float *a = l.delta + (i*l.groups + j)*m*k;
			float *b = net.workspace;
			float *c = l.weight_updates + j * l.nweights / l.groups;

			float *im = net.input + (i*l.groups + j)*l.c / l.groups*l.h*l.w;
			float *imd = net.delta + (i*l.groups + j)*l.c / l.groups*l.h*l.w;

			if (l.size == 1) {
				b = im;
			}
			else {
				im2col_cpu(im, l.c / l.groups, l.h, l.w,
					l.size, l.stride, l.pad, b);
			}

			gemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);

			if (net.delta) {
				a = l.weights + j * l.nweights / l.groups;
				b = l.delta + (i*l.groups + j)*m*k;
				c = net.workspace;
				if (l.size == 1) {
					c = imd;
				}

				gemm(1, 0, n, k, m, 1, a, n, b, k, 0, c, k);

				if (l.size != 1) {
					col2im_cpu(net.workspace, l.c / l.groups, l.h, l.w, l.size, l.stride, l.pad, imd);
				}
			}
		}
	}
}

void update_convolutional_layer(convolutional_layer l, update_args a)
{
	float learning_rate = a.learning_rate*l.learning_rate_scale;
	float momentum = a.momentum;
	float decay = a.decay;
	int batch = a.batch;

	axpy_cpu(l.n, learning_rate / batch, l.bias_updates, 1, l.biases, 1);
	scal_cpu(l.n, momentum, l.bias_updates, 1);

	if (l.scales) {
		axpy_cpu(l.n, learning_rate / batch, l.scale_updates, 1, l.scales, 1);
		scal_cpu(l.n, momentum, l.scale_updates, 1);
	}

	axpy_cpu(l.nweights, -decay * batch, l.weights, 1, l.weight_updates, 1);
	axpy_cpu(l.nweights, learning_rate / batch, l.weight_updates, 1, l.weights, 1);
	scal_cpu(l.nweights, momentum, l.weight_updates, 1);
}


image get_convolutional_weight(convolutional_layer l, int i)
{
	int h = l.size;
	int w = l.size;
	int c = l.c / l.groups;
	return float_to_image(w, h, c, l.weights + i * h*w*c);
}

void rgbgr_weights(convolutional_layer l)
{
	int i;
	for (i = 0; i < l.n; ++i) {
		image im = get_convolutional_weight(l, i);
		if (im.c == 3) {
			rgbgr_image(im);
		}
	}
}

void rescale_weights(convolutional_layer l, float scale, float trans)
{
	int i;
	for (i = 0; i < l.n; ++i) {
		image im = get_convolutional_weight(l, i);
		if (im.c == 3) {
			scale_image(im, scale);
			float sum = sum_array(im.data, im.w*im.h*im.c);
			l.biases[i] += sum * trans;
		}
	}
}

image *get_weights(convolutional_layer l)
{
	image *weights = (image*)calloc(l.n, sizeof(image));
	int i;
	for (i = 0; i < l.n; ++i) {
		weights[i] = copy_image(get_convolutional_weight(l, i));
		normalize_image(weights[i]);
		/*
		   char buff[256];
		   sprintf(buff, "filter%d", i);
		   save_image(weights[i], buff);
		 */
	}
	//error("hey");
	return weights;
}

image *visualize_convolutional_layer(convolutional_layer l, char *window, image *prev_weights)
{
	image *single_weights = get_weights(l);
	show_images(single_weights, l.n, window);

	image delta = get_convolutional_image(l);
	image dc = collapse_image_layers(delta, 1);
	char buff[256];
	sprintf(buff, "%s: Output", window);
	//show_image(dc, buff);
	//save_image(dc, buff);
	free_image(dc);
	return single_weights;
}


//cost_layer.c
COST_TYPE get_cost_type(char *s)
{
	if (strcmp(s, "seg") == 0) return SEG;
	if (strcmp(s, "sse") == 0) return SSE;
	if (strcmp(s, "masked") == 0) return MASKED;
	if (strcmp(s, "smooth") == 0) return SMOOTH;
	if (strcmp(s, "L1") == 0) return L1;
	if (strcmp(s, "wgan") == 0) return WGAN;
	fprintf(stderr, "Couldn't find cost type %s, going with SSE\n", s);
	return SSE;
}

char *get_cost_string(COST_TYPE a)
{
	switch (a) {
	case SEG:
		return (char*)"seg";
	case SSE:
		return (char*)"sse";
	case MASKED:
		return (char*)"masked";
	case SMOOTH:
		return (char*)"smooth";
	case L1:
		return (char*)"L1";
	case WGAN:
		return (char*)"wgan";
	}
	return (char*)"sse";
}

cost_layer make_cost_layer(int batch, int inputs, COST_TYPE cost_type, float scale)
{
	fprintf(stderr, "cost                                           %4d\n", inputs);
	cost_layer l = { };
	l.type = COST;

	l.scale = scale;
	l.batch = batch;
	l.inputs = inputs;
	l.outputs = inputs;
	l.cost_type = cost_type;
	l.delta = (float*)calloc(inputs*batch, sizeof(float));
	l.output = (float*)calloc(inputs*batch, sizeof(float));
	l.cost = (float*)calloc(1, sizeof(float));

	l.forward = forward_cost_layer;
	l.backward = backward_cost_layer;
#ifdef GPU
	l.forward_gpu = forward_cost_layer_gpu;
	l.backward_gpu = backward_cost_layer_gpu;

	l.delta_gpu = cuda_make_array(l.output, inputs*batch);
	l.output_gpu = cuda_make_array(l.delta, inputs*batch);
#endif
	return l;
}

void resize_cost_layer(cost_layer *l, int inputs)
{
	l->inputs = inputs;
	l->outputs = inputs;
	l->delta = (float*)realloc(l->delta, inputs*l->batch * sizeof(float));
	l->output = (float*)realloc(l->output, inputs*l->batch * sizeof(float));
#ifdef GPU
	cuda_free(l->delta_gpu);
	cuda_free(l->output_gpu);
	l->delta_gpu = cuda_make_array(l->delta, inputs*l->batch);
	l->output_gpu = cuda_make_array(l->output, inputs*l->batch);
#endif
}

void forward_cost_layer(cost_layer l, network net)
{
	if (!net.truth) return;
	if (l.cost_type == MASKED) {
		int i;
		for (i = 0; i < l.batch*l.inputs; ++i) {
			if (net.truth[i] == SECRET_NUM) net.input[i] = SECRET_NUM;
		}
	}
	if (l.cost_type == SMOOTH) {
		smooth_l1_cpu(l.batch*l.inputs, net.input, net.truth, l.delta, l.output);
	}
	else if (l.cost_type == L1) {
		l1_cpu(l.batch*l.inputs, net.input, net.truth, l.delta, l.output);
	}
	else {
		l2_cpu(l.batch*l.inputs, net.input, net.truth, l.delta, l.output);
	}
	l.cost[0] = sum_array(l.output, l.batch*l.inputs);
}

void backward_cost_layer(const cost_layer l, network net)
{
	axpy_cpu(l.batch*l.inputs, l.scale, l.delta, 1, net.delta, 1);
}

#ifdef GPU

void pull_cost_layer(cost_layer l)
{
	cuda_pull_array(l.delta_gpu, l.delta, l.batch*l.inputs);
}

void push_cost_layer(cost_layer l)
{
	cuda_push_array(l.delta_gpu, l.delta, l.batch*l.inputs);
}

int float_abs_compare(const void * a, const void * b)
{
	float fa = *(const float*)a;
	if (fa < 0) fa = -fa;
	float fb = *(const float*)b;
	if (fb < 0) fb = -fb;
	return (fa > fb) - (fa < fb);
}

void forward_cost_layer_gpu(cost_layer l, network net)
{
	if (!net.truth) return;
	if (l.smooth) {
		scal_gpu(l.batch*l.inputs, (1 - l.smooth), net.truth_gpu, 1);
		add_gpu(l.batch*l.inputs, l.smooth * 1. / l.inputs, net.truth_gpu, 1);
	}

	if (l.cost_type == SMOOTH) {
		smooth_l1_gpu(l.batch*l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu);
	}
	else if (l.cost_type == L1) {
		l1_gpu(l.batch*l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu);
	}
	else if (l.cost_type == WGAN) {
		wgan_gpu(l.batch*l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu);
	}
	else {
		l2_gpu(l.batch*l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu);
	}

	if (l.cost_type == SEG && l.noobject_scale != 1) {
		scale_mask_gpu(l.batch*l.inputs, l.delta_gpu, 0, net.truth_gpu, l.noobject_scale);
		scale_mask_gpu(l.batch*l.inputs, l.output_gpu, 0, net.truth_gpu, l.noobject_scale);
	}
	if (l.cost_type == MASKED) {
		mask_gpu(l.batch*l.inputs, net.delta_gpu, SECRET_NUM, net.truth_gpu, 0);
	}

	if (l.ratio) {
		cuda_pull_array(l.delta_gpu, l.delta, l.batch*l.inputs);
		qsort(l.delta, l.batch*l.inputs, sizeof(float), float_abs_compare);
		int n = (1 - l.ratio) * l.batch*l.inputs;
		float thresh = l.delta[n];
		thresh = 0;
		printf("%f\n", thresh);
		supp_gpu(l.batch*l.inputs, thresh, l.delta_gpu, 1);
	}

	if (l.thresh) {
		supp_gpu(l.batch*l.inputs, l.thresh*1. / l.inputs, l.delta_gpu, 1);
	}

	cuda_pull_array(l.output_gpu, l.output, l.batch*l.inputs);
	l.cost[0] = sum_array(l.output, l.batch*l.inputs);
}

void backward_cost_layer_gpu(const cost_layer l, network net)
{
	axpy_gpu(l.batch*l.inputs, l.scale, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif








//crnn_layer.c
static void increment_layer(layer *l, int steps)
{
	int num = l->outputs*l->batch*steps;
	l->output += num;
	l->delta += num;
	l->x += num;
	l->x_norm += num;

#ifdef GPU
	l->output_gpu += num;
	l->delta_gpu += num;
	l->x_gpu += num;
	l->x_norm_gpu += num;
#endif
}

layer make_crnn_layer(int batch, int h, int w, int c, int hidden_filters, int output_filters, int steps, ACTIVATION activation, int batch_normalize)
{
	fprintf(stderr, "CRNN Layer: %d x %d x %d image, %d filters\n", h, w, c, output_filters);
	batch = batch / steps;
	layer l = { };
	l.batch = batch;
	l.type = CRNN;
	l.steps = steps;
	l.h = h;
	l.w = w;
	l.c = c;
	l.out_h = h;
	l.out_w = w;
	l.out_c = output_filters;
	l.inputs = h * w*c;
	l.hidden = h * w * hidden_filters;
	l.outputs = l.out_h * l.out_w * l.out_c;

	l.state = (float*)calloc(l.hidden*batch*(steps + 1), sizeof(float));

	l.input_layer = (layer*)malloc(sizeof(layer));
	fprintf(stderr, "\t\t");
	*(l.input_layer) = make_convolutional_layer(batch*steps, h, w, c, hidden_filters, 1, 3, 1, 1, activation, batch_normalize, 0, 0, 0);
	l.input_layer->batch = batch;

	l.self_layer = (layer*)malloc(sizeof(layer));
	fprintf(stderr, "\t\t");
	*(l.self_layer) = make_convolutional_layer(batch*steps, h, w, hidden_filters, hidden_filters, 1, 3, 1, 1, activation, batch_normalize, 0, 0, 0);
	l.self_layer->batch = batch;

	l.output_layer = (layer*)malloc(sizeof(layer));
	fprintf(stderr, "\t\t");
	*(l.output_layer) = make_convolutional_layer(batch*steps, h, w, hidden_filters, output_filters, 1, 3, 1, 1, activation, batch_normalize, 0, 0, 0);
	l.output_layer->batch = batch;

	l.output = l.output_layer->output;
	l.delta = l.output_layer->delta;

	l.forward = forward_crnn_layer;
	l.backward = backward_crnn_layer;
	l.update = update_crnn_layer;

#ifdef GPU
	l.forward_gpu = forward_crnn_layer_gpu;
	l.backward_gpu = backward_crnn_layer_gpu;
	l.update_gpu = update_crnn_layer_gpu;

	l.state_gpu = cuda_make_array(l.state, l.hidden*batch*(steps + 1));
	l.output_gpu = l.output_layer->output_gpu;
	l.delta_gpu = l.output_layer->delta_gpu;
#endif

	return l;
}

void update_crnn_layer(layer l, update_args a)
{
	update_convolutional_layer(*(l.input_layer), a);
	update_convolutional_layer(*(l.self_layer), a);
	update_convolutional_layer(*(l.output_layer), a);
}

void forward_crnn_layer(layer l, network net)
{
	network s = net;
	s.train = net.train;
	int i;
	layer input_layer = *(l.input_layer);
	layer self_layer = *(l.self_layer);
	layer output_layer = *(l.output_layer);

	fill_cpu(l.outputs * l.batch * l.steps, 0, output_layer.delta, 1);
	fill_cpu(l.hidden * l.batch * l.steps, 0, self_layer.delta, 1);
	fill_cpu(l.hidden * l.batch * l.steps, 0, input_layer.delta, 1);
	if (net.train) fill_cpu(l.hidden * l.batch, 0, l.state, 1);

	for (i = 0; i < l.steps; ++i) {
		s.input = net.input;
		forward_convolutional_layer(input_layer, s);

		s.input = l.state;
		forward_convolutional_layer(self_layer, s);

		float *old_state = l.state;
		if (net.train) l.state += l.hidden*l.batch;
		if (l.shortcut) {
			copy_cpu(l.hidden * l.batch, old_state, 1, l.state, 1);
		}
		else {
			fill_cpu(l.hidden * l.batch, 0, l.state, 1);
		}
		axpy_cpu(l.hidden * l.batch, 1, input_layer.output, 1, l.state, 1);
		axpy_cpu(l.hidden * l.batch, 1, self_layer.output, 1, l.state, 1);

		s.input = l.state;
		forward_convolutional_layer(output_layer, s);

		net.input += l.inputs*l.batch;
		increment_layer(&input_layer, 1);
		increment_layer(&self_layer, 1);
		increment_layer(&output_layer, 1);
	}
}

void backward_crnn_layer(layer l, network net)
{
	network s = net;
	int i;
	layer input_layer = *(l.input_layer);
	layer self_layer = *(l.self_layer);
	layer output_layer = *(l.output_layer);

	increment_layer(&input_layer, l.steps - 1);
	increment_layer(&self_layer, l.steps - 1);
	increment_layer(&output_layer, l.steps - 1);

	l.state += l.hidden*l.batch*l.steps;
	for (i = l.steps - 1; i >= 0; --i) {
		copy_cpu(l.hidden * l.batch, input_layer.output, 1, l.state, 1);
		axpy_cpu(l.hidden * l.batch, 1, self_layer.output, 1, l.state, 1);

		s.input = l.state;
		s.delta = self_layer.delta;
		backward_convolutional_layer(output_layer, s);

		l.state -= l.hidden*l.batch;
		/*
		   if(i > 0){
		   copy_cpu(l.hidden * l.batch, input_layer.output - l.hidden*l.batch, 1, l.state, 1);
		   axpy_cpu(l.hidden * l.batch, 1, self_layer.output - l.hidden*l.batch, 1, l.state, 1);
		   }else{
		   fill_cpu(l.hidden * l.batch, 0, l.state, 1);
		   }
		 */

		s.input = l.state;
		s.delta = self_layer.delta - l.hidden*l.batch;
		if (i == 0) s.delta = 0;
		backward_convolutional_layer(self_layer, s);

		copy_cpu(l.hidden*l.batch, self_layer.delta, 1, input_layer.delta, 1);
		if (i > 0 && l.shortcut) axpy_cpu(l.hidden*l.batch, 1, self_layer.delta, 1, self_layer.delta - l.hidden*l.batch, 1);
		s.input = net.input + i * l.inputs*l.batch;
		if (net.delta) s.delta = net.delta + i * l.inputs*l.batch;
		else s.delta = 0;
		backward_convolutional_layer(input_layer, s);

		increment_layer(&input_layer, -1);
		increment_layer(&self_layer, -1);
		increment_layer(&output_layer, -1);
	}
}

#ifdef GPU

void pull_crnn_layer(layer l)
{
	pull_convolutional_layer(*(l.input_layer));
	pull_convolutional_layer(*(l.self_layer));
	pull_convolutional_layer(*(l.output_layer));
}

void push_crnn_layer(layer l)
{
	push_convolutional_layer(*(l.input_layer));
	push_convolutional_layer(*(l.self_layer));
	push_convolutional_layer(*(l.output_layer));
}

void update_crnn_layer_gpu(layer l, update_args a)
{
	update_convolutional_layer_gpu(*(l.input_layer), a);
	update_convolutional_layer_gpu(*(l.self_layer), a);
	update_convolutional_layer_gpu(*(l.output_layer), a);
}

void forward_crnn_layer_gpu(layer l, network net)
{
	network s = net;
	int i;
	layer input_layer = *(l.input_layer);
	layer self_layer = *(l.self_layer);
	layer output_layer = *(l.output_layer);

	fill_gpu(l.outputs * l.batch * l.steps, 0, output_layer.delta_gpu, 1);
	fill_gpu(l.hidden * l.batch * l.steps, 0, self_layer.delta_gpu, 1);
	fill_gpu(l.hidden * l.batch * l.steps, 0, input_layer.delta_gpu, 1);
	if (net.train) fill_gpu(l.hidden * l.batch, 0, l.state_gpu, 1);

	for (i = 0; i < l.steps; ++i) {
		s.input_gpu = net.input_gpu;
		forward_convolutional_layer_gpu(input_layer, s);

		s.input_gpu = l.state_gpu;
		forward_convolutional_layer_gpu(self_layer, s);

		float *old_state = l.state_gpu;
		if (net.train) l.state_gpu += l.hidden*l.batch;
		if (l.shortcut) {
			copy_gpu(l.hidden * l.batch, old_state, 1, l.state_gpu, 1);
		}
		else {
			fill_gpu(l.hidden * l.batch, 0, l.state_gpu, 1);
		}
		axpy_gpu(l.hidden * l.batch, 1, input_layer.output_gpu, 1, l.state_gpu, 1);
		axpy_gpu(l.hidden * l.batch, 1, self_layer.output_gpu, 1, l.state_gpu, 1);

		s.input_gpu = l.state_gpu;
		forward_convolutional_layer_gpu(output_layer, s);

		net.input_gpu += l.inputs*l.batch;
		increment_layer(&input_layer, 1);
		increment_layer(&self_layer, 1);
		increment_layer(&output_layer, 1);
	}
}

void backward_crnn_layer_gpu(layer l, network net)
{
	network s = net;
	s.train = net.train;
	int i;
	layer input_layer = *(l.input_layer);
	layer self_layer = *(l.self_layer);
	layer output_layer = *(l.output_layer);
	increment_layer(&input_layer, l.steps - 1);
	increment_layer(&self_layer, l.steps - 1);
	increment_layer(&output_layer, l.steps - 1);
	l.state_gpu += l.hidden*l.batch*l.steps;
	for (i = l.steps - 1; i >= 0; --i) {
		copy_gpu(l.hidden * l.batch, input_layer.output_gpu, 1, l.state_gpu, 1);
		axpy_gpu(l.hidden * l.batch, 1, self_layer.output_gpu, 1, l.state_gpu, 1);

		s.input_gpu = l.state_gpu;
		s.delta_gpu = self_layer.delta_gpu;
		backward_convolutional_layer_gpu(output_layer, s);

		l.state_gpu -= l.hidden*l.batch;

		s.input_gpu = l.state_gpu;
		s.delta_gpu = self_layer.delta_gpu - l.hidden*l.batch;
		if (i == 0) s.delta_gpu = 0;
		backward_convolutional_layer_gpu(self_layer, s);

		copy_gpu(l.hidden*l.batch, self_layer.delta_gpu, 1, input_layer.delta_gpu, 1);
		if (i > 0 && l.shortcut) axpy_gpu(l.hidden*l.batch, 1, self_layer.delta_gpu, 1, self_layer.delta_gpu - l.hidden*l.batch, 1);
		s.input_gpu = net.input_gpu + i * l.inputs*l.batch;
		if (net.delta_gpu) s.delta_gpu = net.delta_gpu + i * l.inputs*l.batch;
		else s.delta_gpu = 0;
		backward_convolutional_layer_gpu(input_layer, s);

		increment_layer(&input_layer, -1);
		increment_layer(&self_layer, -1);
		increment_layer(&output_layer, -1);
	}
}
#endif








//crop_layer.c
image get_crop_image(crop_layer l)
{
	int h = l.out_h;
	int w = l.out_w;
	int c = l.out_c;
	return float_to_image(w, h, c, l.output);
}

void backward_crop_layer(const crop_layer l, network net) {}
void backward_crop_layer_gpu(const crop_layer l, network net) {}

crop_layer make_crop_layer(int batch, int h, int w, int c, int crop_height, int crop_width, int flip, float angle, float saturation, float exposure)
{
	fprintf(stderr, "Crop Layer: %d x %d -> %d x %d x %d image\n", h, w, crop_height, crop_width, c);
	crop_layer l = { };
	l.type = CROP;
	l.batch = batch;
	l.h = h;
	l.w = w;
	l.c = c;
	l.scale = (float)crop_height / h;
	l.flip = flip;
	l.angle = angle;
	l.saturation = saturation;
	l.exposure = exposure;
	l.out_w = crop_width;
	l.out_h = crop_height;
	l.out_c = c;
	l.inputs = l.w * l.h * l.c;
	l.outputs = l.out_w * l.out_h * l.out_c;
	l.output = (float*)calloc(l.outputs*batch, sizeof(float));
	l.forward = forward_crop_layer;
	l.backward = backward_crop_layer;

#ifdef GPU
	l.forward_gpu = forward_crop_layer_gpu;
	l.backward_gpu = backward_crop_layer_gpu;
	l.output_gpu = cuda_make_array(l.output, l.outputs*batch);
	l.rand_gpu = cuda_make_array(0, l.batch * 8);
#endif
	return l;
}

void resize_crop_layer(layer *l, int w, int h)
{
	l->w = w;
	l->h = h;

	l->out_w = (int)(l->scale*w);
	l->out_h = (int)(l->scale*h);

	l->inputs = l->w * l->h * l->c;
	l->outputs = l->out_h * l->out_w * l->out_c;

	l->output = (float*)realloc(l->output, l->batch*l->outputs * sizeof(float));
#ifdef GPU
	cuda_free(l->output_gpu);
	l->output_gpu = cuda_make_array(l->output, l->outputs*l->batch);
#endif
}


void forward_crop_layer(const crop_layer l, network net)
{
	int i, j, c, b, row, col;
	int index;
	int count = 0;
	int flip = (l.flip && rand() % 2);
	int dh = rand() % (l.h - l.out_h + 1);
	int dw = rand() % (l.w - l.out_w + 1);
	float scale = 2;
	float trans = -1;
	if (l.noadjust) {
		scale = 1;
		trans = 0;
	}
	if (!net.train) {
		flip = 0;
		dh = (l.h - l.out_h) / 2;
		dw = (l.w - l.out_w) / 2;
	}
	for (b = 0; b < l.batch; ++b) {
		for (c = 0; c < l.c; ++c) {
			for (i = 0; i < l.out_h; ++i) {
				for (j = 0; j < l.out_w; ++j) {
					if (flip) {
						col = l.w - dw - j - 1;
					}
					else {
						col = j + dw;
					}
					row = i + dh;
					index = col + l.w*(row + l.h*(c + l.c*b));
					l.output[count++] = net.input[index] * scale + trans;
				}
			}
		}
	}
}











//deconvolutional_layer.c
void bilinear_init(layer l)
{
	int i, j, f;
	float center = (float)((l.size - 1) / 2.);
	for (f = 0; f < l.n; ++f) {
		for (j = 0; j < l.size; ++j) {
			for (i = 0; i < l.size; ++i) {
				float val = (1 - fabs(i - center)) * (1 - fabs(j - center));
				int c = f % l.c;
				int ind = f * l.size*l.size*l.c + c * l.size*l.size + j * l.size + i;
				l.weights[ind] = val;
			}
		}
	}
}


layer make_deconvolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int adam)
{
	int i;
	layer l = {};
	l.type = DECONVOLUTIONAL;

	l.h = h;
	l.w = w;
	l.c = c;
	l.n = n;
	l.batch = batch;
	l.stride = stride;
	l.size = size;

	l.nweights = c * n*size*size;
	l.nbiases = n;

	l.weights = (float*)calloc(c*n*size*size, sizeof(float));
	l.weight_updates = (float*)calloc(c*n*size*size, sizeof(float));

	l.biases = (float*)calloc(n, sizeof(float));
	l.bias_updates = (float*)calloc(n, sizeof(float));
	//float scale = n/(size*size*c);
	//printf("scale: %f\n", scale);
	float scale = (float).02;
	for (i = 0; i < c*n*size*size; ++i) l.weights[i] = scale * rand_normal();
	//bilinear_init(l);
	for (i = 0; i < n; ++i) {
		l.biases[i] = 0;
	}
	l.pad = padding;

	l.out_h = (l.h - 1) * l.stride + l.size - 2 * l.pad;
	l.out_w = (l.w - 1) * l.stride + l.size - 2 * l.pad;
	l.out_c = n;
	l.outputs = l.out_w * l.out_h * l.out_c;
	l.inputs = l.w * l.h * l.c;

	scal_cpu(l.nweights, (float)l.out_w*l.out_h / (l.w*l.h), l.weights, 1);

	l.output = (float*)calloc(l.batch*l.outputs, sizeof(float));
	l.delta = (float*)calloc(l.batch*l.outputs, sizeof(float));

	l.forward = forward_deconvolutional_layer;
	l.backward = backward_deconvolutional_layer;
	l.update = update_deconvolutional_layer;

	l.batch_normalize = batch_normalize;

	if (batch_normalize) {
		l.scales = (float*)calloc(n, sizeof(float));
		l.scale_updates = (float*)calloc(n, sizeof(float));
		for (i = 0; i < n; ++i) {
			l.scales[i] = 1;
		}

		l.mean = (float*)calloc(n, sizeof(float));
		l.variance = (float*)calloc(n, sizeof(float));

		l.mean_delta = (float*)calloc(n, sizeof(float));
		l.variance_delta = (float*)calloc(n, sizeof(float));

		l.rolling_mean = (float*)calloc(n, sizeof(float));
		l.rolling_variance = (float*)calloc(n, sizeof(float));
		l.x = (float*)calloc(l.batch*l.outputs, sizeof(float));
		l.x_norm = (float*)calloc(l.batch*l.outputs, sizeof(float));
	}
	if (adam) {
		l.m = (float*)calloc(c*n*size*size, sizeof(float));
		l.v = (float*)calloc(c*n*size*size, sizeof(float));
		l.bias_m = (float*)calloc(n, sizeof(float));
		l.scale_m = (float*)calloc(n, sizeof(float));
		l.bias_v = (float*)calloc(n, sizeof(float));
		l.scale_v = (float*)calloc(n, sizeof(float));
	}

#ifdef GPU
	l.forward_gpu = forward_deconvolutional_layer_gpu;
	l.backward_gpu = backward_deconvolutional_layer_gpu;
	l.update_gpu = update_deconvolutional_layer_gpu;

	if (gpu_index >= 0) {

		if (adam) {
			l.m_gpu = cuda_make_array(l.m, c*n*size*size);
			l.v_gpu = cuda_make_array(l.v, c*n*size*size);
			l.bias_m_gpu = cuda_make_array(l.bias_m, n);
			l.bias_v_gpu = cuda_make_array(l.bias_v, n);
			l.scale_m_gpu = cuda_make_array(l.scale_m, n);
			l.scale_v_gpu = cuda_make_array(l.scale_v, n);
		}
		l.weights_gpu = cuda_make_array(l.weights, c*n*size*size);
		l.weight_updates_gpu = cuda_make_array(l.weight_updates, c*n*size*size);

		l.biases_gpu = cuda_make_array(l.biases, n);
		l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

		l.delta_gpu = cuda_make_array(l.delta, l.batch*l.out_h*l.out_w*n);
		l.output_gpu = cuda_make_array(l.output, l.batch*l.out_h*l.out_w*n);

		if (batch_normalize) {
			l.mean_gpu = cuda_make_array(0, n);
			l.variance_gpu = cuda_make_array(0, n);

			l.rolling_mean_gpu = cuda_make_array(0, n);
			l.rolling_variance_gpu = cuda_make_array(0, n);

			l.mean_delta_gpu = cuda_make_array(0, n);
			l.variance_delta_gpu = cuda_make_array(0, n);

			l.scales_gpu = cuda_make_array(l.scales, n);
			l.scale_updates_gpu = cuda_make_array(0, n);

			l.x_gpu = cuda_make_array(0, l.batch*l.out_h*l.out_w*n);
			l.x_norm_gpu = cuda_make_array(0, l.batch*l.out_h*l.out_w*n);
		}
	}
#ifdef CUDNN
	cudnnCreateTensorDescriptor(&l.dstTensorDesc);
	cudnnCreateTensorDescriptor(&l.normTensorDesc);
	cudnnSetTensor4dDescriptor(l.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.out_c, l.out_h, l.out_w);
	cudnnSetTensor4dDescriptor(l.normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l.out_c, 1, 1);
#endif
#endif

	l.activation = activation;
	l.workspace_size = get_workspace_size(l);

	fprintf(stderr, "deconv%5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);

	return l;
}

void denormalize_deconvolutional_layer(layer l)
{
	int i, j;
	for (i = 0; i < l.n; ++i) {
		float scale = (float)(l.scales[i] / sqrt(l.rolling_variance[i] + .00001));
		for (j = 0; j < l.c*l.size*l.size; ++j) {
			l.weights[i*l.c*l.size*l.size + j] *= scale;
		}
		l.biases[i] -= l.rolling_mean[i] * scale;
		l.scales[i] = 1;
		l.rolling_mean[i] = 0;
		l.rolling_variance[i] = 1;
	}
}

void resize_deconvolutional_layer(layer *l, int h, int w)
{
	l->h = h;
	l->w = w;
	l->out_h = (l->h - 1) * l->stride + l->size - 2 * l->pad;
	l->out_w = (l->w - 1) * l->stride + l->size - 2 * l->pad;

	l->outputs = l->out_h * l->out_w * l->out_c;
	l->inputs = l->w * l->h * l->c;

	l->output = (float*)realloc(l->output, l->batch*l->outputs * sizeof(float));
	l->delta = (float*)realloc(l->delta, l->batch*l->outputs * sizeof(float));
	if (l->batch_normalize) {
		l->x = (float*)realloc(l->x, l->batch*l->outputs * sizeof(float));
		l->x_norm = (float*)realloc(l->x_norm, l->batch*l->outputs * sizeof(float));
	}

#ifdef GPU
	cuda_free(l->delta_gpu);
	cuda_free(l->output_gpu);

	l->delta_gpu = cuda_make_array(l->delta, l->batch*l->outputs);
	l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);

	if (l->batch_normalize) {
		cuda_free(l->x_gpu);
		cuda_free(l->x_norm_gpu);

		l->x_gpu = cuda_make_array(l->output, l->batch*l->outputs);
		l->x_norm_gpu = cuda_make_array(l->output, l->batch*l->outputs);
	}
#ifdef CUDNN
	cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w);
	cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1);
#endif
#endif
	l->workspace_size = get_workspace_size(*l);
}

void forward_deconvolutional_layer(const layer l, network net)
{
	int i;

	int m = l.size*l.size*l.n;
	int n = l.h*l.w;
	int k = l.c;

	fill_cpu(l.outputs*l.batch, 0, l.output, 1);

	for (i = 0; i < l.batch; ++i) {
		float *a = l.weights;
		float *b = net.input + i * l.c*l.h*l.w;
		float *c = net.workspace;

		gemm_cpu(1, 0, m, n, k, 1, a, m, b, n, 0, c, n);

		col2im_cpu(net.workspace, l.out_c, l.out_h, l.out_w, l.size, l.stride, l.pad, l.output + i * l.outputs);
	}
	if (l.batch_normalize) {
		forward_batchnorm_layer(l, net);
	}
	else {
		add_bias(l.output, l.biases, l.batch, l.n, l.out_w*l.out_h);
	}
	activate_array(l.output, l.batch*l.n*l.out_w*l.out_h, l.activation);
}

void backward_deconvolutional_layer(layer l, network net)
{
	int i;

	gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

	if (l.batch_normalize) {
		backward_batchnorm_layer(l, net);
	}
	else {
		backward_bias(l.bias_updates, l.delta, l.batch, l.n, l.out_w*l.out_h);
	}

	//if(net.delta) memset(net.delta, 0, l.batch*l.h*l.w*l.c*sizeof(float));

	for (i = 0; i < l.batch; ++i) {
		int m = l.c;
		int n = l.size*l.size*l.n;
		int k = l.h*l.w;

		float *a = net.input + i * m*k;
		float *b = net.workspace;
		float *c = l.weight_updates;

		im2col_cpu(l.delta + i * l.outputs, l.out_c, l.out_h, l.out_w,
			l.size, l.stride, l.pad, b);
		gemm_cpu(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);

		if (net.delta) {
			int m = l.c;
			int n = l.h*l.w;
			int k = l.size*l.size*l.n;

			float *a = l.weights;
			float *b = net.workspace;
			float *c = net.delta + i * n*m;

			gemm_cpu(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
		}
	}
}

void update_deconvolutional_layer(layer l, update_args a)
{
	float learning_rate = a.learning_rate*l.learning_rate_scale;
	float momentum = a.momentum;
	float decay = a.decay;
	int batch = a.batch;

	int size = l.size*l.size*l.c*l.n;
	axpy_cpu(l.n, learning_rate / batch, l.bias_updates, 1, l.biases, 1);
	scal_cpu(l.n, momentum, l.bias_updates, 1);

	if (l.scales) {
		axpy_cpu(l.n, learning_rate / batch, l.scale_updates, 1, l.scales, 1);
		scal_cpu(l.n, momentum, l.scale_updates, 1);
	}

	axpy_cpu(size, -decay * batch, l.weights, 1, l.weight_updates, 1);
	axpy_cpu(size, learning_rate / batch, l.weight_updates, 1, l.weights, 1);
	scal_cpu(size, momentum, l.weight_updates, 1);
}











//detection_layer.c
detection_layer make_detection_layer(int batch, int inputs, int n, int side, int classes, int coords, int rescore)
{
	detection_layer l = { };
	l.type = DETECTION;

	l.n = n;
	l.batch = batch;
	l.inputs = inputs;
	l.classes = classes;
	l.coords = coords;
	l.rescore = rescore;
	l.side = side;
	l.w = side;
	l.h = side;
	assert(side*side*((1 + l.coords)*l.n + l.classes) == inputs);
	l.cost = (float*)calloc(1, sizeof(float));
	l.outputs = l.inputs;
	l.truths = l.side*l.side*(1 + l.coords + l.classes);
	l.output = (float*)calloc(batch*l.outputs, sizeof(float));
	l.delta = (float*)calloc(batch*l.outputs, sizeof(float));

	l.forward = forward_detection_layer;
	l.backward = backward_detection_layer;
#ifdef GPU
	l.forward_gpu = forward_detection_layer_gpu;
	l.backward_gpu = backward_detection_layer_gpu;
	l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
	l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

	fprintf(stderr, "Detection Layer\n");
	srand(0);

	return l;
}

void forward_detection_layer(const detection_layer l, network net)
{
	int locations = l.side*l.side;
	int i, j;
	memcpy(l.output, net.input, l.outputs*l.batch * sizeof(float));
	//if(l.reorg) reorg(l.output, l.w*l.h, size*l.n, l.batch, 1);
	int b;
	if (l.softmax) {
		for (b = 0; b < l.batch; ++b) {
			int index = b * l.inputs;
			for (i = 0; i < locations; ++i) {
				int offset = i * l.classes;
				softmax(l.output + index + offset, l.classes, 1, 1,
					l.output + index + offset);
			}
		}
	}
	if (net.train) {
		float avg_iou = 0;
		float avg_cat = 0;
		float avg_allcat = 0;
		float avg_obj = 0;
		float avg_anyobj = 0;
		int count = 0;
		*(l.cost) = 0;
		int size = l.inputs * l.batch;
		memset(l.delta, 0, size * sizeof(float));
		for (b = 0; b < l.batch; ++b) {
			int index = b * l.inputs;
			for (i = 0; i < locations; ++i) {
				int truth_index = (b*locations + i)*(1 + l.coords + l.classes);
				int is_obj = (int)(net.truth[truth_index]);
				for (j = 0; j < l.n; ++j) {
					int p_index = index + locations * l.classes + i * l.n + j;
					l.delta[p_index] = l.noobject_scale*(0 - l.output[p_index]);
					*(l.cost) += l.noobject_scale*pow(l.output[p_index], 2);
					avg_anyobj += l.output[p_index];
				}

				int best_index = -1;
				float best_iou = 0;
				float best_rmse = 20;

				if (!is_obj) {
					continue;
				}

				int class_index = index + i * l.classes;
				for (j = 0; j < l.classes; ++j) {
					l.delta[class_index + j] = l.class_scale * (net.truth[truth_index + 1 + j] - l.output[class_index + j]);
					*(l.cost) += l.class_scale * pow(net.truth[truth_index + 1 + j] - l.output[class_index + j], 2);
					if (net.truth[truth_index + 1 + j]) avg_cat += l.output[class_index + j];
					avg_allcat += l.output[class_index + j];
				}

				box truth = float_to_box(net.truth + truth_index + 1 + l.classes, 1);
				truth.x /= l.side;
				truth.y /= l.side;

				for (j = 0; j < l.n; ++j) {
					int box_index = index + locations * (l.classes + l.n) + (i*l.n + j) * l.coords;
					box out = float_to_box(l.output + box_index, 1);
					out.x /= l.side;
					out.y /= l.side;

					if (l.sqrt) {
						out.w = out.w*out.w;
						out.h = out.h*out.h;
					}

					float iou = box_iou(out, truth);
					//iou = 0;
					float rmse = box_rmse(out, truth);
					if (best_iou > 0 || iou > 0) {
						if (iou > best_iou) {
							best_iou = iou;
							best_index = j;
						}
					}
					else {
						if (rmse < best_rmse) {
							best_rmse = rmse;
							best_index = j;
						}
					}
				}

				if (l.forced) {
					if (truth.w*truth.h < .1) {
						best_index = 1;
					}
					else {
						best_index = 0;
					}
				}
				if (l.random && *(net.seen) < 64000) {
					best_index = rand() % l.n;
				}

				int box_index = index + locations * (l.classes + l.n) + (i*l.n + best_index) * l.coords;
				int tbox_index = truth_index + 1 + l.classes;

				box out = float_to_box(l.output + box_index, 1);
				out.x /= l.side;
				out.y /= l.side;
				if (l.sqrt) {
					out.w = out.w*out.w;
					out.h = out.h*out.h;
				}
				float iou = box_iou(out, truth);

				//printf("%d,", best_index);
				int p_index = index + locations * l.classes + i * l.n + best_index;
				*(l.cost) -= l.noobject_scale * pow(l.output[p_index], 2);
				*(l.cost) += l.object_scale * pow(1 - l.output[p_index], 2);
				avg_obj += l.output[p_index];
				l.delta[p_index] = (float)(l.object_scale * (1. - l.output[p_index]));

				if (l.rescore) {
					l.delta[p_index] = l.object_scale * (iou - l.output[p_index]);
				}

				l.delta[box_index + 0] = l.coord_scale*(net.truth[tbox_index + 0] - l.output[box_index + 0]);
				l.delta[box_index + 1] = l.coord_scale*(net.truth[tbox_index + 1] - l.output[box_index + 1]);
				l.delta[box_index + 2] = l.coord_scale*(net.truth[tbox_index + 2] - l.output[box_index + 2]);
				l.delta[box_index + 3] = l.coord_scale*(net.truth[tbox_index + 3] - l.output[box_index + 3]);
				if (l.sqrt) {
					l.delta[box_index + 2] = l.coord_scale*(sqrt(net.truth[tbox_index + 2]) - l.output[box_index + 2]);
					l.delta[box_index + 3] = l.coord_scale*(sqrt(net.truth[tbox_index + 3]) - l.output[box_index + 3]);
				}

				*(l.cost) += pow(1 - iou, 2);
				avg_iou += iou;
				++count;
			}
		}

		if (0) {
			float *costs = (float*)calloc(l.batch*locations*l.n, sizeof(float));
			for (b = 0; b < l.batch; ++b) {
				int index = b * l.inputs;
				for (i = 0; i < locations; ++i) {
					for (j = 0; j < l.n; ++j) {
						int p_index = index + locations * l.classes + i * l.n + j;
						costs[b*locations*l.n + i * l.n + j] = l.delta[p_index] * l.delta[p_index];
					}
				}
			}
			int indexes[100];
			top_k(costs, l.batch*locations*l.n, 100, indexes);
			float cutoff = costs[indexes[99]];
			for (b = 0; b < l.batch; ++b) {
				int index = b * l.inputs;
				for (i = 0; i < locations; ++i) {
					for (j = 0; j < l.n; ++j) {
						int p_index = index + locations * l.classes + i * l.n + j;
						if (l.delta[p_index] * l.delta[p_index] < cutoff) l.delta[p_index] = 0;
					}
				}
			}
			free(costs);
		}


		*(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);


		printf("Detection Avg IOU: %f, Pos Cat: %f, All Cat: %f, Pos Obj: %f, Any Obj: %f, count: %d\n", avg_iou / count, avg_cat / count, avg_allcat / (count*l.classes), avg_obj / count, avg_anyobj / (l.batch*locations*l.n), count);
		//if(l.reorg) reorg(l.delta, l.w*l.h, size*l.n, l.batch, 0);
	}
}

void backward_detection_layer(const detection_layer l, network net)
{
	axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}

void get_detection_detections(layer l, int w, int h, float thresh, detection *dets)
{
	int i, j, n;
	float *predictions = l.output;
	//int per_cell = 5*num+classes;
	for (i = 0; i < l.side*l.side; ++i) {
		int row = i / l.side;
		int col = i % l.side;
		for (n = 0; n < l.n; ++n) {
			int index = i * l.n + n;
			int p_index = l.side*l.side*l.classes + i * l.n + n;
			float scale = predictions[p_index];
			int box_index = l.side*l.side*(l.classes + l.n) + (i*l.n + n) * 4;
			box b;
			b.x = (predictions[box_index + 0] + col) / l.side * w;
			b.y = (predictions[box_index + 1] + row) / l.side * h;
			b.w = pow(predictions[box_index + 2], (l.sqrt ? 2 : 1)) * w;
			b.h = pow(predictions[box_index + 3], (l.sqrt ? 2 : 1)) * h;
			dets[index].bbox = b;
			dets[index].objectness = scale;
			for (j = 0; j < l.classes; ++j) {
				int class_index = i * l.classes;
				float prob = scale * predictions[class_index + j];
				dets[index].prob[j] = (prob > thresh) ? prob : 0;
			}
		}
	}
}

#ifdef GPU

void forward_detection_layer_gpu(const detection_layer l, network net)
{
	if (!net.train) {
		copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
		return;
	}

	cuda_pull_array(net.input_gpu, net.input, l.batch*l.inputs);
	forward_detection_layer(l, net);
	cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
	cuda_push_array(l.delta_gpu, l.delta, l.batch*l.inputs);
}

void backward_detection_layer_gpu(detection_layer l, network net)
{
	axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
	//copy_gpu(l.batch*l.inputs, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif








//dropout_layer.c
dropout_layer make_dropout_layer(int batch, int inputs, float probability)
{
	dropout_layer l = {};
	l.type = DROPOUT;
	l.probability = probability;
	l.inputs = inputs;
	l.outputs = inputs;
	l.batch = batch;
	l.rand = (float*)calloc(inputs*batch, sizeof(float));
	l.scale = (float)(1. / (1. - probability));
	l.forward = forward_dropout_layer;
	l.backward = backward_dropout_layer;
#ifdef GPU
	l.forward_gpu = forward_dropout_layer_gpu;
	l.backward_gpu = backward_dropout_layer_gpu;
	l.rand_gpu = cuda_make_array(l.rand, inputs*batch);
#endif
	fprintf(stderr, "dropout       p = %.2f               %4d  ->  %4d\n", probability, inputs, inputs);
	return l;
}

void resize_dropout_layer(dropout_layer *l, int inputs)
{
	l->rand = (float*)realloc(l->rand, l->inputs*l->batch * sizeof(float));
#ifdef GPU
	cuda_free(l->rand_gpu);

	l->rand_gpu = cuda_make_array(l->rand, inputs*l->batch);
#endif
}

void forward_dropout_layer(dropout_layer l, network net)
{
	int i;
	if (!net.train) return;
	for (i = 0; i < l.batch * l.inputs; ++i) {
		float r = rand_uniform(0, 1);
		l.rand[i] = r;
		if (r < l.probability) net.input[i] = 0;
		else net.input[i] *= l.scale;
	}
}

void backward_dropout_layer(dropout_layer l, network net)
{
	int i;
	if (!net.delta) return;
	for (i = 0; i < l.batch * l.inputs; ++i) {
		float r = l.rand[i];
		if (r < l.probability) net.delta[i] = 0;
		else net.delta[i] *= l.scale;
	}
}









//gru_layer.c
//Already defined in another_layer.c
/*static void increment_layer(layer *l, int steps)
{
	int num = l->outputs*l->batch*steps;
	l->output += num;
	l->delta += num;
	l->x += num;
	l->x_norm += num;

#ifdef GPU
	l->output_gpu += num;
	l->delta_gpu += num;
	l->x_gpu += num;
	l->x_norm_gpu += num;
#endif
}*/

layer make_gru_layer(int batch, int inputs, int outputs, int steps, int batch_normalize, int adam)
{
	fprintf(stderr, "GRU Layer: %d inputs, %d outputs\n", inputs, outputs);
	batch = batch / steps;
	layer l = { };
	l.batch = batch;
	l.type = GRU;
	l.steps = steps;
	l.inputs = inputs;

	l.uz = (layer*)malloc(sizeof(layer));
	fprintf(stderr, "\t\t");
	*(l.uz) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
	l.uz->batch = batch;

	l.wz = (layer*)malloc(sizeof(layer));
	fprintf(stderr, "\t\t");
	*(l.wz) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
	l.wz->batch = batch;

	l.ur = (layer*)malloc(sizeof(layer));
	fprintf(stderr, "\t\t");
	*(l.ur) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
	l.ur->batch = batch;

	l.wr = (layer*)malloc(sizeof(layer));
	fprintf(stderr, "\t\t");
	*(l.wr) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
	l.wr->batch = batch;



	l.uh = (layer*)malloc(sizeof(layer));
	fprintf(stderr, "\t\t");
	*(l.uh) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
	l.uh->batch = batch;

	l.wh = (layer*)malloc(sizeof(layer));
	fprintf(stderr, "\t\t");
	*(l.wh) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
	l.wh->batch = batch;

	l.batch_normalize = batch_normalize;


	l.outputs = outputs;
	l.output = (float*)calloc(outputs*batch*steps, sizeof(float));
	l.delta = (float*)calloc(outputs*batch*steps, sizeof(float));
	l.state = (float*)calloc(outputs*batch, sizeof(float));
	l.prev_state = (float*)calloc(outputs*batch, sizeof(float));
	l.forgot_state = (float*)calloc(outputs*batch, sizeof(float));
	l.forgot_delta = (float*)calloc(outputs*batch, sizeof(float));

	l.r_cpu = (float*)calloc(outputs*batch, sizeof(float));
	l.z_cpu = (float*)calloc(outputs*batch, sizeof(float));
	l.h_cpu = (float*)calloc(outputs*batch, sizeof(float));

	l.forward = forward_gru_layer;
	l.backward = backward_gru_layer;
	l.update = update_gru_layer;

#ifdef GPU
	l.forward_gpu = forward_gru_layer_gpu;
	l.backward_gpu = backward_gru_layer_gpu;
	l.update_gpu = update_gru_layer_gpu;

	l.forgot_state_gpu = cuda_make_array(0, batch*outputs);
	l.forgot_delta_gpu = cuda_make_array(0, batch*outputs);
	l.prev_state_gpu = cuda_make_array(0, batch*outputs);
	l.state_gpu = cuda_make_array(0, batch*outputs);
	l.output_gpu = cuda_make_array(0, batch*outputs*steps);
	l.delta_gpu = cuda_make_array(0, batch*outputs*steps);
	l.r_gpu = cuda_make_array(0, batch*outputs);
	l.z_gpu = cuda_make_array(0, batch*outputs);
	l.h_gpu = cuda_make_array(0, batch*outputs);

#ifdef CUDNN
	cudnnSetTensor4dDescriptor(l.uz->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.uz->out_c, l.uz->out_h, l.uz->out_w);
	cudnnSetTensor4dDescriptor(l.uh->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.uh->out_c, l.uh->out_h, l.uh->out_w);
	cudnnSetTensor4dDescriptor(l.ur->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.ur->out_c, l.ur->out_h, l.ur->out_w);
	cudnnSetTensor4dDescriptor(l.wz->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.wz->out_c, l.wz->out_h, l.wz->out_w);
	cudnnSetTensor4dDescriptor(l.wh->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.wh->out_c, l.wh->out_h, l.wh->out_w);
	cudnnSetTensor4dDescriptor(l.wr->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.wr->out_c, l.wr->out_h, l.wr->out_w);
#endif
#endif

	return l;
}

void update_gru_layer(layer l, update_args a)
{
	update_connected_layer(*(l.ur), a);
	update_connected_layer(*(l.uz), a);
	update_connected_layer(*(l.uh), a);
	update_connected_layer(*(l.wr), a);
	update_connected_layer(*(l.wz), a);
	update_connected_layer(*(l.wh), a);
}

void forward_gru_layer(layer l, network net)
{
	network s = net;
	s.train = net.train;
	int i;
	layer uz = *(l.uz);
	layer ur = *(l.ur);
	layer uh = *(l.uh);

	layer wz = *(l.wz);
	layer wr = *(l.wr);
	layer wh = *(l.wh);

	fill_cpu(l.outputs * l.batch * l.steps, 0, uz.delta, 1);
	fill_cpu(l.outputs * l.batch * l.steps, 0, ur.delta, 1);
	fill_cpu(l.outputs * l.batch * l.steps, 0, uh.delta, 1);

	fill_cpu(l.outputs * l.batch * l.steps, 0, wz.delta, 1);
	fill_cpu(l.outputs * l.batch * l.steps, 0, wr.delta, 1);
	fill_cpu(l.outputs * l.batch * l.steps, 0, wh.delta, 1);
	if (net.train) {
		fill_cpu(l.outputs * l.batch * l.steps, 0, l.delta, 1);
		copy_cpu(l.outputs*l.batch, l.state, 1, l.prev_state, 1);
	}

	for (i = 0; i < l.steps; ++i) {
		s.input = l.state;
		forward_connected_layer(wz, s);
		forward_connected_layer(wr, s);

		s.input = net.input;
		forward_connected_layer(uz, s);
		forward_connected_layer(ur, s);
		forward_connected_layer(uh, s);


		copy_cpu(l.outputs*l.batch, uz.output, 1, l.z_cpu, 1);
		axpy_cpu(l.outputs*l.batch, 1, wz.output, 1, l.z_cpu, 1);

		copy_cpu(l.outputs*l.batch, ur.output, 1, l.r_cpu, 1);
		axpy_cpu(l.outputs*l.batch, 1, wr.output, 1, l.r_cpu, 1);

		activate_array(l.z_cpu, l.outputs*l.batch, LOGISTIC);
		activate_array(l.r_cpu, l.outputs*l.batch, LOGISTIC);

		copy_cpu(l.outputs*l.batch, l.state, 1, l.forgot_state, 1);
		mul_cpu(l.outputs*l.batch, l.r_cpu, 1, l.forgot_state, 1);

		s.input = l.forgot_state;
		forward_connected_layer(wh, s);

		copy_cpu(l.outputs*l.batch, uh.output, 1, l.h_cpu, 1);
		axpy_cpu(l.outputs*l.batch, 1, wh.output, 1, l.h_cpu, 1);

		if (l.tanh) {
			activate_array(l.h_cpu, l.outputs*l.batch, TANH);
		}
		else {
			activate_array(l.h_cpu, l.outputs*l.batch, LOGISTIC);
		}

		weighted_sum_cpu(l.state, l.h_cpu, l.z_cpu, l.outputs*l.batch, l.output);

		copy_cpu(l.outputs*l.batch, l.output, 1, l.state, 1);

		net.input += l.inputs*l.batch;
		l.output += l.outputs*l.batch;
		increment_layer(&uz, 1);
		increment_layer(&ur, 1);
		increment_layer(&uh, 1);

		increment_layer(&wz, 1);
		increment_layer(&wr, 1);
		increment_layer(&wh, 1);
	}
}

void backward_gru_layer(layer l, network net)
{
}

#ifdef GPU

void pull_gru_layer(layer l)
{
}

void push_gru_layer(layer l)
{
}

void update_gru_layer_gpu(layer l, update_args a)
{
	update_connected_layer_gpu(*(l.ur), a);
	update_connected_layer_gpu(*(l.uz), a);
	update_connected_layer_gpu(*(l.uh), a);
	update_connected_layer_gpu(*(l.wr), a);
	update_connected_layer_gpu(*(l.wz), a);
	update_connected_layer_gpu(*(l.wh), a);
}

void forward_gru_layer_gpu(layer l, network net)
{
	network s = { 0 };
	s.train = net.train;
	int i;
	layer uz = *(l.uz);
	layer ur = *(l.ur);
	layer uh = *(l.uh);

	layer wz = *(l.wz);
	layer wr = *(l.wr);
	layer wh = *(l.wh);

	fill_gpu(l.outputs * l.batch * l.steps, 0, uz.delta_gpu, 1);
	fill_gpu(l.outputs * l.batch * l.steps, 0, ur.delta_gpu, 1);
	fill_gpu(l.outputs * l.batch * l.steps, 0, uh.delta_gpu, 1);

	fill_gpu(l.outputs * l.batch * l.steps, 0, wz.delta_gpu, 1);
	fill_gpu(l.outputs * l.batch * l.steps, 0, wr.delta_gpu, 1);
	fill_gpu(l.outputs * l.batch * l.steps, 0, wh.delta_gpu, 1);
	if (net.train) {
		fill_gpu(l.outputs * l.batch * l.steps, 0, l.delta_gpu, 1);
		copy_gpu(l.outputs*l.batch, l.state_gpu, 1, l.prev_state_gpu, 1);
	}

	for (i = 0; i < l.steps; ++i) {
		s.input_gpu = l.state_gpu;
		forward_connected_layer_gpu(wz, s);
		forward_connected_layer_gpu(wr, s);

		s.input_gpu = net.input_gpu;
		forward_connected_layer_gpu(uz, s);
		forward_connected_layer_gpu(ur, s);
		forward_connected_layer_gpu(uh, s);

		copy_gpu(l.outputs*l.batch, uz.output_gpu, 1, l.z_gpu, 1);
		axpy_gpu(l.outputs*l.batch, 1, wz.output_gpu, 1, l.z_gpu, 1);

		copy_gpu(l.outputs*l.batch, ur.output_gpu, 1, l.r_gpu, 1);
		axpy_gpu(l.outputs*l.batch, 1, wr.output_gpu, 1, l.r_gpu, 1);

		activate_array_gpu(l.z_gpu, l.outputs*l.batch, LOGISTIC);
		activate_array_gpu(l.r_gpu, l.outputs*l.batch, LOGISTIC);

		copy_gpu(l.outputs*l.batch, l.state_gpu, 1, l.forgot_state_gpu, 1);
		mul_gpu(l.outputs*l.batch, l.r_gpu, 1, l.forgot_state_gpu, 1);

		s.input_gpu = l.forgot_state_gpu;
		forward_connected_layer_gpu(wh, s);

		copy_gpu(l.outputs*l.batch, uh.output_gpu, 1, l.h_gpu, 1);
		axpy_gpu(l.outputs*l.batch, 1, wh.output_gpu, 1, l.h_gpu, 1);

		if (l.tanh) {
			activate_array_gpu(l.h_gpu, l.outputs*l.batch, TANH);
		}
		else {
			activate_array_gpu(l.h_gpu, l.outputs*l.batch, LOGISTIC);
		}

		weighted_sum_gpu(l.state_gpu, l.h_gpu, l.z_gpu, l.outputs*l.batch, l.output_gpu);
		copy_gpu(l.outputs*l.batch, l.output_gpu, 1, l.state_gpu, 1);

		net.input_gpu += l.inputs*l.batch;
		l.output_gpu += l.outputs*l.batch;
		increment_layer(&uz, 1);
		increment_layer(&ur, 1);
		increment_layer(&uh, 1);

		increment_layer(&wz, 1);
		increment_layer(&wr, 1);
		increment_layer(&wh, 1);
	}
}

void backward_gru_layer_gpu(layer l, network net)
{
	network s = { 0 };
	s.train = net.train;
	int i;
	layer uz = *(l.uz);
	layer ur = *(l.ur);
	layer uh = *(l.uh);

	layer wz = *(l.wz);
	layer wr = *(l.wr);
	layer wh = *(l.wh);

	increment_layer(&uz, l.steps - 1);
	increment_layer(&ur, l.steps - 1);
	increment_layer(&uh, l.steps - 1);

	increment_layer(&wz, l.steps - 1);
	increment_layer(&wr, l.steps - 1);
	increment_layer(&wh, l.steps - 1);

	net.input_gpu += l.inputs*l.batch*(l.steps - 1);
	if (net.delta_gpu) net.delta_gpu += l.inputs*l.batch*(l.steps - 1);
	l.output_gpu += l.outputs*l.batch*(l.steps - 1);
	l.delta_gpu += l.outputs*l.batch*(l.steps - 1);
	float *end_state = l.output_gpu;
	for (i = l.steps - 1; i >= 0; --i) {
		if (i != 0) copy_gpu(l.outputs*l.batch, l.output_gpu - l.outputs*l.batch, 1, l.state_gpu, 1);
		else copy_gpu(l.outputs*l.batch, l.prev_state_gpu, 1, l.state_gpu, 1);
		float *prev_delta_gpu = (i == 0) ? 0 : l.delta_gpu - l.outputs*l.batch;

		copy_gpu(l.outputs*l.batch, uz.output_gpu, 1, l.z_gpu, 1);
		axpy_gpu(l.outputs*l.batch, 1, wz.output_gpu, 1, l.z_gpu, 1);

		copy_gpu(l.outputs*l.batch, ur.output_gpu, 1, l.r_gpu, 1);
		axpy_gpu(l.outputs*l.batch, 1, wr.output_gpu, 1, l.r_gpu, 1);

		activate_array_gpu(l.z_gpu, l.outputs*l.batch, LOGISTIC);
		activate_array_gpu(l.r_gpu, l.outputs*l.batch, LOGISTIC);

		copy_gpu(l.outputs*l.batch, uh.output_gpu, 1, l.h_gpu, 1);
		axpy_gpu(l.outputs*l.batch, 1, wh.output_gpu, 1, l.h_gpu, 1);

		if (l.tanh) {
			activate_array_gpu(l.h_gpu, l.outputs*l.batch, TANH);
		}
		else {
			activate_array_gpu(l.h_gpu, l.outputs*l.batch, LOGISTIC);
		}

		weighted_delta_gpu(l.state_gpu, l.h_gpu, l.z_gpu, prev_delta_gpu, uh.delta_gpu, uz.delta_gpu, l.outputs*l.batch, l.delta_gpu);

		if (l.tanh) {
			gradient_array_gpu(l.h_gpu, l.outputs*l.batch, TANH, uh.delta_gpu);
		}
		else {
			gradient_array_gpu(l.h_gpu, l.outputs*l.batch, LOGISTIC, uh.delta_gpu);
		}

		copy_gpu(l.outputs*l.batch, uh.delta_gpu, 1, wh.delta_gpu, 1);

		copy_gpu(l.outputs*l.batch, l.state_gpu, 1, l.forgot_state_gpu, 1);
		mul_gpu(l.outputs*l.batch, l.r_gpu, 1, l.forgot_state_gpu, 1);
		fill_gpu(l.outputs*l.batch, 0, l.forgot_delta_gpu, 1);

		s.input_gpu = l.forgot_state_gpu;
		s.delta_gpu = l.forgot_delta_gpu;

		backward_connected_layer_gpu(wh, s);
		if (prev_delta_gpu) mult_add_into_gpu(l.outputs*l.batch, l.forgot_delta_gpu, l.r_gpu, prev_delta_gpu);
		mult_add_into_gpu(l.outputs*l.batch, l.forgot_delta_gpu, l.state_gpu, ur.delta_gpu);

		gradient_array_gpu(l.r_gpu, l.outputs*l.batch, LOGISTIC, ur.delta_gpu);
		copy_gpu(l.outputs*l.batch, ur.delta_gpu, 1, wr.delta_gpu, 1);

		gradient_array_gpu(l.z_gpu, l.outputs*l.batch, LOGISTIC, uz.delta_gpu);
		copy_gpu(l.outputs*l.batch, uz.delta_gpu, 1, wz.delta_gpu, 1);

		s.input_gpu = l.state_gpu;
		s.delta_gpu = prev_delta_gpu;

		backward_connected_layer_gpu(wr, s);
		backward_connected_layer_gpu(wz, s);

		s.input_gpu = net.input_gpu;
		s.delta_gpu = net.delta_gpu;

		backward_connected_layer_gpu(uh, s);
		backward_connected_layer_gpu(ur, s);
		backward_connected_layer_gpu(uz, s);


		net.input_gpu -= l.inputs*l.batch;
		if (net.delta_gpu) net.delta_gpu -= l.inputs*l.batch;
		l.output_gpu -= l.outputs*l.batch;
		l.delta_gpu -= l.outputs*l.batch;
		increment_layer(&uz, -1);
		increment_layer(&ur, -1);
		increment_layer(&uh, -1);

		increment_layer(&wz, -1);
		increment_layer(&wr, -1);
		increment_layer(&wh, -1);
	}
	copy_gpu(l.outputs*l.batch, end_state, 1, l.state_gpu, 1);
}
#endif








//iseg_layer.c
layer make_iseg_layer(int batch, int w, int h, int classes, int ids)
{
	layer l = {};
	l.type = ISEG;

	l.h = h;
	l.w = w;
	l.c = classes + ids;
	l.out_w = l.w;
	l.out_h = l.h;
	l.out_c = l.c;
	l.classes = classes;
	l.batch = batch;
	l.extra = ids;
	l.cost = (float*)calloc(1, sizeof(float));
	l.outputs = h * w*l.c;
	l.inputs = l.outputs;
	l.truths = 90 * (l.w*l.h + 1);
	l.delta = (float*)calloc(batch*l.outputs, sizeof(float));
	l.output = (float*)calloc(batch*l.outputs, sizeof(float));

	l.counts = (int*)calloc(90, sizeof(int));
	l.sums = (float**)calloc(90, sizeof(float*));
	if (ids) {
		int i;
		for (i = 0; i < 90; ++i) {
			l.sums[i] = (float*)calloc(ids, sizeof(float));
		}
	}

	l.forward = forward_iseg_layer;
	l.backward = backward_iseg_layer;
#ifdef GPU
	l.forward_gpu = forward_iseg_layer_gpu;
	l.backward_gpu = backward_iseg_layer_gpu;
	l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
	l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

	fprintf(stderr, "iseg\n");
	srand(0);

	return l;
}

void resize_iseg_layer(layer *l, int w, int h)
{
	l->w = w;
	l->h = h;

	l->outputs = h * w*l->c;
	l->inputs = l->outputs;

	l->output = (float*)realloc(l->output, l->batch*l->outputs * sizeof(float));
	l->delta = (float*)realloc(l->delta, l->batch*l->outputs * sizeof(float));

#ifdef GPU
	cuda_free(l->delta_gpu);
	cuda_free(l->output_gpu);

	l->delta_gpu = cuda_make_array(l->delta, l->batch*l->outputs);
	l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);
#endif
}

void forward_iseg_layer(const layer l, network net)
{

	clock_t time = clock();
//	double time = what_time_is_it_now();
	int i, b, j, k;
	int ids = l.extra;
	memcpy(l.output, net.input, l.outputs*l.batch * sizeof(float));
	memset(l.delta, 0, l.outputs * l.batch * sizeof(float));

#ifndef GPU
	for (b = 0; b < l.batch; ++b) {
		int index = b * l.outputs;
		activate_array(l.output + index, l.classes*l.w*l.h, LOGISTIC);
	}
#endif

	for (b = 0; b < l.batch; ++b) {
		// a priori, each pixel has no class
		for (i = 0; i < l.classes; ++i) {
			for (k = 0; k < l.w*l.h; ++k) {
				int index = b * l.outputs + i * l.w*l.h + k;
				l.delta[index] = 0 - l.output[index];
			}
		}

		// a priori, embedding should be small magnitude
		for (i = 0; i < ids; ++i) {
			for (k = 0; k < l.w*l.h; ++k) {
				int index = b * l.outputs + (i + l.classes)*l.w*l.h + k;
				l.delta[index] = (float)(.1 * (0 - l.output[index]));
			}
		}


		memset(l.counts, 0, 90 * sizeof(int));
		for (i = 0; i < 90; ++i) {
			fill_cpu(ids, 0, l.sums[i], 1);

			int c = (int)(net.truth[b*l.truths + i * (l.w*l.h + 1)]);
			if (c < 0) break;
			// add up metric embeddings for each instance
			for (k = 0; k < l.w*l.h; ++k) {
				int index = b * l.outputs + c * l.w*l.h + k;
				float v = net.truth[b*l.truths + i * (l.w*l.h + 1) + 1 + k];
				if (v) {
					l.delta[index] = v - l.output[index];
					axpy_cpu(ids, 1, l.output + b * l.outputs + l.classes*l.w*l.h + k, l.w*l.h, l.sums[i], 1);
					++l.counts[i];
				}
			}
		}

		float *mse = (float*)calloc(90, sizeof(float));
		for (i = 0; i < 90; ++i) {
			int c = (int)(net.truth[b*l.truths + i * (l.w*l.h + 1)]);
			if (c < 0) break;
			for (k = 0; k < l.w*l.h; ++k) {
				float v = net.truth[b*l.truths + i * (l.w*l.h + 1) + 1 + k];
				if (v) {
					int z;
					float sum = 0;
					for (z = 0; z < ids; ++z) {
						int index = b * l.outputs + (l.classes + z)*l.w*l.h + k;
						sum += pow(l.sums[i][z] / l.counts[i] - l.output[index], 2);
					}
					mse[i] += sum;
				}
			}
			mse[i] /= l.counts[i];
		}

		// Calculate average embedding
		for (i = 0; i < 90; ++i) {
			if (!l.counts[i]) continue;
			scal_cpu(ids, 1.f / l.counts[i], l.sums[i], 1);
			if (b == 0 && net.gpu_index == 0) {
				printf("%4d, %6.3f, ", l.counts[i], mse[i]);
				for (j = 0; j < ids; ++j) {
					printf("%6.3f,", l.sums[i][j]);
				}
				printf("\n");
			}
		}
		free(mse);

		// Calculate embedding loss
		for (i = 0; i < 90; ++i) {
			if (!l.counts[i]) continue;
			for (k = 0; k < l.w*l.h; ++k) {
				float v = net.truth[b*l.truths + i * (l.w*l.h + 1) + 1 + k];
				if (v) {
					for (j = 0; j < 90; ++j) {
						if (!l.counts[j])continue;
						int z;
						for (z = 0; z < ids; ++z) {
							int index = b * l.outputs + (l.classes + z)*l.w*l.h + k;
							float diff = l.sums[j][z] - l.output[index];
							if (j == i) l.delta[index] += (float)(diff < 0 ? -.1 : .1);
							else        l.delta[index] += (float)(-(diff < 0 ? -.1 : .1));
						}
					}
				}
			}
		}

		for (i = 0; i < ids; ++i) {
			for (k = 0; k < l.w*l.h; ++k) {
				int index = b * l.outputs + (i + l.classes)*l.w*l.h + k;
				l.delta[index] *= (float).01;
			}
		}
	}

	*(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
	printf("took %lf sec\n", sec(clock() - time));
}

void backward_iseg_layer(const layer l, network net)
{
	axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}

#ifdef GPU

void forward_iseg_layer_gpu(const layer l, network net)
{
	copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
	int b;
	for (b = 0; b < l.batch; ++b) {
		activate_array_gpu(l.output_gpu + b * l.outputs, l.classes*l.w*l.h, LOGISTIC);
		//if(l.extra) activate_array_gpu(l.output_gpu + b*l.outputs + l.classes*l.w*l.h, l.extra*l.w*l.h, LOGISTIC);
	}

	cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
	forward_iseg_layer(l, net);
	cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
}

void backward_iseg_layer_gpu(const layer l, network net)
{
	int b;
	for (b = 0; b < l.batch; ++b) {
		//if(l.extra) gradient_array_gpu(l.output_gpu + b*l.outputs + l.classes*l.w*l.h, l.extra*l.w*l.h, LOGISTIC, l.delta_gpu + b*l.outputs + l.classes*l.w*l.h);
	}
	axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif









//l2norm_layer.c
layer make_l2norm_layer(int batch, int inputs)
{
	fprintf(stderr, "l2norm                                         %4d\n", inputs);
	layer l = {};
	l.type = L2NORM;
	l.batch = batch;
	l.inputs = inputs;
	l.outputs = inputs;
	l.output = (float*)calloc(inputs*batch, sizeof(float));
	l.scales = (float*)calloc(inputs*batch, sizeof(float));
	l.delta = (float*)calloc(inputs*batch, sizeof(float));

	l.forward = forward_l2norm_layer;
	l.backward = backward_l2norm_layer;
#ifdef GPU
	l.forward_gpu = forward_l2norm_layer_gpu;
	l.backward_gpu = backward_l2norm_layer_gpu;

	l.output_gpu = cuda_make_array(l.output, inputs*batch);
	l.scales_gpu = cuda_make_array(l.output, inputs*batch);
	l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
#endif
	return l;
}

void forward_l2norm_layer(const layer l, network net)
{
	copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
	l2normalize_cpu(l.output, l.scales, l.batch, l.out_c, l.out_w*l.out_h);
}

void backward_l2norm_layer(const layer l, network net)
{
	//axpy_cpu(l.inputs*l.batch, 1, l.scales, 1, l.delta, 1);
	axpy_cpu(l.inputs*l.batch, 1, l.delta, 1, net.delta, 1);
}

#ifdef GPU

void forward_l2norm_layer_gpu(const layer l, network net)
{
	copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
	l2normalize_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_w*l.out_h);
}

void backward_l2norm_layer_gpu(const layer l, network net)
{
	axpy_gpu(l.batch*l.inputs, 1, l.scales_gpu, 1, l.delta_gpu, 1);
	axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}

#endif








//local_layer.c
int local_out_height(local_layer l)
{
	int h = l.h;
	if (!l.pad) h -= l.size;
	else h -= 1;
	return h / l.stride + 1;
}

int local_out_width(local_layer l)
{
	int w = l.w;
	if (!l.pad) w -= l.size;
	else w -= 1;
	return w / l.stride + 1;
}

local_layer make_local_layer(int batch, int h, int w, int c, int n, int size, int stride, int pad, ACTIVATION activation)
{
	int i;
	local_layer l = { };
	l.type = LOCAL;

	l.h = h;
	l.w = w;
	l.c = c;
	l.n = n;
	l.batch = batch;
	l.stride = stride;
	l.size = size;
	l.pad = pad;

	int out_h = local_out_height(l);
	int out_w = local_out_width(l);
	int locations = out_h * out_w;
	l.out_h = out_h;
	l.out_w = out_w;
	l.out_c = n;
	l.outputs = l.out_h * l.out_w * l.out_c;
	l.inputs = l.w * l.h * l.c;

	l.weights = (float*)calloc(c*n*size*size*locations, sizeof(float));
	l.weight_updates = (float*)calloc(c*n*size*size*locations, sizeof(float));

	l.biases = (float*)calloc(l.outputs, sizeof(float));
	l.bias_updates = (float*)calloc(l.outputs, sizeof(float));

	// float scale = 1./sqrt(size*size*c);
	float scale = (float)(sqrt(2. / (size*size*c)));
	for (i = 0; i < c*n*size*size; ++i) l.weights[i] = scale * rand_uniform(-1, 1);

	l.output = (float*)calloc(l.batch*out_h * out_w * n, sizeof(float));
	l.delta = (float*)calloc(l.batch*out_h * out_w * n, sizeof(float));

	l.workspace_size = out_h * out_w*size*size*c;

	l.forward = forward_local_layer;
	l.backward = backward_local_layer;
	l.update = update_local_layer;

#ifdef GPU
	l.forward_gpu = forward_local_layer_gpu;
	l.backward_gpu = backward_local_layer_gpu;
	l.update_gpu = update_local_layer_gpu;

	l.weights_gpu = cuda_make_array(l.weights, c*n*size*size*locations);
	l.weight_updates_gpu = cuda_make_array(l.weight_updates, c*n*size*size*locations);

	l.biases_gpu = cuda_make_array(l.biases, l.outputs);
	l.bias_updates_gpu = cuda_make_array(l.bias_updates, l.outputs);

	l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*n);
	l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);

#endif
	l.activation = activation;

	fprintf(stderr, "Local Layer: %d x %d x %d image, %d filters -> %d x %d x %d image\n", h, w, c, n, out_h, out_w, n);

	return l;
}

void forward_local_layer(const local_layer l, network net)
{
	int out_h = local_out_height(l);
	int out_w = local_out_width(l);
	int i, j;
	int locations = out_h * out_w;

	for (i = 0; i < l.batch; ++i) {
		copy_cpu(l.outputs, l.biases, 1, l.output + i * l.outputs, 1);
	}

	for (i = 0; i < l.batch; ++i) {
		float *input = net.input + i * l.w*l.h*l.c;
		im2col_cpu(input, l.c, l.h, l.w,
			l.size, l.stride, l.pad, net.workspace);
		float *output = l.output + i * l.outputs;
		for (j = 0; j < locations; ++j) {
			float *a = l.weights + j * l.size*l.size*l.c*l.n;
			float *b = net.workspace + j;
			float *c = output + j;

			int m = l.n;
			int n = 1;
			int k = l.size*l.size*l.c;

			gemm(0, 0, m, n, k, 1, a, k, b, locations, 1, c, locations);
		}
	}
	activate_array(l.output, l.outputs*l.batch, l.activation);
}

void backward_local_layer(local_layer l, network net)
{
	int i, j;
	int locations = l.out_w*l.out_h;

	gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

	for (i = 0; i < l.batch; ++i) {
		axpy_cpu(l.outputs, 1, l.delta + i * l.outputs, 1, l.bias_updates, 1);
	}

	for (i = 0; i < l.batch; ++i) {
		float *input = net.input + i * l.w*l.h*l.c;
		im2col_cpu(input, l.c, l.h, l.w,
			l.size, l.stride, l.pad, net.workspace);

		for (j = 0; j < locations; ++j) {
			float *a = l.delta + i * l.outputs + j;
			float *b = net.workspace + j;
			float *c = l.weight_updates + j * l.size*l.size*l.c*l.n;
			int m = l.n;
			int n = l.size*l.size*l.c;
			int k = 1;

			gemm(0, 1, m, n, k, 1, a, locations, b, locations, 1, c, n);
		}

		if (net.delta) {
			for (j = 0; j < locations; ++j) {
				float *a = l.weights + j * l.size*l.size*l.c*l.n;
				float *b = l.delta + i * l.outputs + j;
				float *c = net.workspace + j;

				int m = l.size*l.size*l.c;
				int n = 1;
				int k = l.n;

				gemm(1, 0, m, n, k, 1, a, m, b, locations, 0, c, locations);
			}

			col2im_cpu(net.workspace, l.c, l.h, l.w, l.size, l.stride, l.pad, net.delta + i * l.c*l.h*l.w);
		}
	}
}

void update_local_layer(local_layer l, update_args a)
{
	float learning_rate = a.learning_rate*l.learning_rate_scale;
	float momentum = a.momentum;
	float decay = a.decay;
	int batch = a.batch;

	int locations = l.out_w*l.out_h;
	int size = l.size*l.size*l.c*l.n*locations;
	axpy_cpu(l.outputs, learning_rate / batch, l.bias_updates, 1, l.biases, 1);
	scal_cpu(l.outputs, momentum, l.bias_updates, 1);

	axpy_cpu(size, -decay * batch, l.weights, 1, l.weight_updates, 1);
	axpy_cpu(size, learning_rate / batch, l.weight_updates, 1, l.weights, 1);
	scal_cpu(size, momentum, l.weight_updates, 1);
}

#ifdef GPU

void forward_local_layer_gpu(const local_layer l, network net)
{
	int out_h = local_out_height(l);
	int out_w = local_out_width(l);
	int i, j;
	int locations = out_h * out_w;

	for (i = 0; i < l.batch; ++i) {
		copy_gpu(l.outputs, l.biases_gpu, 1, l.output_gpu + i * l.outputs, 1);
	}

	for (i = 0; i < l.batch; ++i) {
		float *input = net.input_gpu + i * l.w*l.h*l.c;
		im2col_gpu(input, l.c, l.h, l.w,
			l.size, l.stride, l.pad, net.workspace);
		float *output = l.output_gpu + i * l.outputs;
		for (j = 0; j < locations; ++j) {
			float *a = l.weights_gpu + j * l.size*l.size*l.c*l.n;
			float *b = net.workspace + j;
			float *c = output + j;

			int m = l.n;
			int n = 1;
			int k = l.size*l.size*l.c;

			gemm_gpu(0, 0, m, n, k, 1, a, k, b, locations, 1, c, locations);
		}
	}
	activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void backward_local_layer_gpu(local_layer l, network net)
{
	int i, j;
	int locations = l.out_w*l.out_h;

	gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
	for (i = 0; i < l.batch; ++i) {
		axpy_gpu(l.outputs, 1, l.delta_gpu + i * l.outputs, 1, l.bias_updates_gpu, 1);
	}

	for (i = 0; i < l.batch; ++i) {
		float *input = net.input_gpu + i * l.w*l.h*l.c;
		im2col_gpu(input, l.c, l.h, l.w,
			l.size, l.stride, l.pad, net.workspace);

		for (j = 0; j < locations; ++j) {
			float *a = l.delta_gpu + i * l.outputs + j;
			float *b = net.workspace + j;
			float *c = l.weight_updates_gpu + j * l.size*l.size*l.c*l.n;
			int m = l.n;
			int n = l.size*l.size*l.c;
			int k = 1;

			gemm_gpu(0, 1, m, n, k, 1, a, locations, b, locations, 1, c, n);
		}

		if (net.delta_gpu) {
			for (j = 0; j < locations; ++j) {
				float *a = l.weights_gpu + j * l.size*l.size*l.c*l.n;
				float *b = l.delta_gpu + i * l.outputs + j;
				float *c = net.workspace + j;

				int m = l.size*l.size*l.c;
				int n = 1;
				int k = l.n;

				gemm_gpu(1, 0, m, n, k, 1, a, m, b, locations, 0, c, locations);
			}

			col2im_gpu(net.workspace, l.c, l.h, l.w, l.size, l.stride, l.pad, net.delta_gpu + i * l.c*l.h*l.w);
		}
	}
}

void update_local_layer_gpu(local_layer l, update_args a)
{
	float learning_rate = a.learning_rate*l.learning_rate_scale;
	float momentum = a.momentum;
	float decay = a.decay;
	int batch = a.batch;

	int locations = l.out_w*l.out_h;
	int size = l.size*l.size*l.c*l.n*locations;
	axpy_gpu(l.outputs, learning_rate / batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
	scal_gpu(l.outputs, momentum, l.bias_updates_gpu, 1);

	axpy_gpu(size, -decay * batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
	axpy_gpu(size, learning_rate / batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
	scal_gpu(size, momentum, l.weight_updates_gpu, 1);
}

void pull_local_layer(local_layer l)
{
	int locations = l.out_w*l.out_h;
	int size = l.size*l.size*l.c*l.n*locations;
	cuda_pull_array(l.weights_gpu, l.weights, size);
	cuda_pull_array(l.biases_gpu, l.biases, l.outputs);
}

void push_local_layer(local_layer l)
{
	int locations = l.out_w*l.out_h;
	int size = l.size*l.size*l.c*l.n*locations;
	cuda_push_array(l.weights_gpu, l.weights, size);
	cuda_push_array(l.biases_gpu, l.biases, l.outputs);
}
#endif








//logistic_layer.c
layer make_logistic_layer(int batch, int inputs)
{
	fprintf(stderr, "logistic x entropy                             %4d\n", inputs);
	layer l = { };
	l.type = LOGXENT;
	l.batch = batch;
	l.inputs = inputs;
	l.outputs = inputs;
	l.loss = (float*)calloc(inputs*batch, sizeof(float));
	l.output = (float*)calloc(inputs*batch, sizeof(float));
	l.delta = (float*)calloc(inputs*batch, sizeof(float));
	l.cost = (float*)calloc(1, sizeof(float));

	l.forward = forward_logistic_layer;
	l.backward = backward_logistic_layer;
#ifdef GPU
	l.forward_gpu = forward_logistic_layer_gpu;
	l.backward_gpu = backward_logistic_layer_gpu;

	l.output_gpu = cuda_make_array(l.output, inputs*batch);
	l.loss_gpu = cuda_make_array(l.loss, inputs*batch);
	l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
#endif
	return l;
}

void forward_logistic_layer(const layer l, network net)
{
	copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
	activate_array(l.output, l.outputs*l.batch, LOGISTIC);
	if (net.truth) {
		logistic_x_ent_cpu(l.batch*l.inputs, l.output, net.truth, l.delta, l.loss);
		l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
	}
}

void backward_logistic_layer(const layer l, network net)
{
	axpy_cpu(l.inputs*l.batch, 1, l.delta, 1, net.delta, 1);
}

#ifdef GPU

void forward_logistic_layer_gpu(const layer l, network net)
{
	copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
	activate_array_gpu(l.output_gpu, l.outputs*l.batch, LOGISTIC);
	if (net.truth) {
		logistic_x_ent_gpu(l.batch*l.inputs, l.output_gpu, net.truth_gpu, l.delta_gpu, l.loss_gpu);
		cuda_pull_array(l.loss_gpu, l.loss, l.batch*l.inputs);
		l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
	}
}

void backward_logistic_layer_gpu(const layer l, network net)
{
	axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}

#endif










//lstm_layer.c
/*Declared in crnn_layer.c
static void increment_layer(layer *l, int steps)
{
	int num = l->outputs*l->batch*steps;
	l->output += num;
	l->delta += num;
	l->x += num;
	l->x_norm += num;

#ifdef GPU
	l->output_gpu += num;
	l->delta_gpu += num;
	l->x_gpu += num;
	l->x_norm_gpu += num;
#endif
}*/

layer make_lstm_layer(int batch, int inputs, int outputs, int steps, int batch_normalize, int adam)
{
	fprintf(stderr, "LSTM Layer: %d inputs, %d outputs\n", inputs, outputs);
	batch = batch / steps;
	layer l = {  };
	l.batch = batch;
	l.type = LSTM;
	l.steps = steps;
	l.inputs = inputs;

	l.uf = (layer*)malloc(sizeof(layer));
	fprintf(stderr, "\t\t");
	*(l.uf) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
	l.uf->batch = batch;

	l.ui = (layer*)malloc(sizeof(layer));
	fprintf(stderr, "\t\t");
	*(l.ui) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
	l.ui->batch = batch;

	l.ug = (layer*)malloc(sizeof(layer));
	fprintf(stderr, "\t\t");
	*(l.ug) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
	l.ug->batch = batch;

	l.uo = (layer*)malloc(sizeof(layer));
	fprintf(stderr, "\t\t");
	*(l.uo) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
	l.uo->batch = batch;

	l.wf = (layer*)malloc(sizeof(layer));
	fprintf(stderr, "\t\t");
	*(l.wf) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
	l.wf->batch = batch;

	l.wi = (layer*)malloc(sizeof(layer));
	fprintf(stderr, "\t\t");
	*(l.wi) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
	l.wi->batch = batch;

	l.wg = (layer*)malloc(sizeof(layer));
	fprintf(stderr, "\t\t");
	*(l.wg) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
	l.wg->batch = batch;

	l.wo = (layer*)malloc(sizeof(layer));
	fprintf(stderr, "\t\t");
	*(l.wo) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
	l.wo->batch = batch;

	l.batch_normalize = batch_normalize;
	l.outputs = outputs;

	l.output = (float*)calloc(outputs*batch*steps, sizeof(float));
	l.state = (float*)calloc(outputs*batch, sizeof(float));

	l.forward = forward_lstm_layer;
	l.update = update_lstm_layer;

	l.prev_state_cpu = (float*)calloc(batch*outputs, sizeof(float));
	l.prev_cell_cpu = (float*)calloc(batch*outputs, sizeof(float));
	l.cell_cpu = (float*)calloc(batch*outputs*steps, sizeof(float));

	l.f_cpu = (float*)calloc(batch*outputs, sizeof(float));
	l.i_cpu = (float*)calloc(batch*outputs, sizeof(float));
	l.g_cpu = (float*)calloc(batch*outputs, sizeof(float));
	l.o_cpu = (float*)calloc(batch*outputs, sizeof(float));
	l.c_cpu = (float*)calloc(batch*outputs, sizeof(float));
	l.h_cpu = (float*)calloc(batch*outputs, sizeof(float));
	l.temp_cpu = (float*)calloc(batch*outputs, sizeof(float));
	l.temp2_cpu = (float*)calloc(batch*outputs, sizeof(float));
	l.temp3_cpu = (float*)calloc(batch*outputs, sizeof(float));
	l.dc_cpu = (float*)calloc(batch*outputs, sizeof(float));
	l.dh_cpu = (float*)calloc(batch*outputs, sizeof(float));

#ifdef GPU
	l.forward_gpu = forward_lstm_layer_gpu;
	l.backward_gpu = backward_lstm_layer_gpu;
	l.update_gpu = update_lstm_layer_gpu;

	l.output_gpu = cuda_make_array(0, batch*outputs*steps);
	l.delta_gpu = cuda_make_array(0, batch*l.outputs*steps);

	l.prev_state_gpu = cuda_make_array(0, batch*outputs);
	l.prev_cell_gpu = cuda_make_array(0, batch*outputs);
	l.cell_gpu = cuda_make_array(0, batch*outputs*steps);

	l.f_gpu = cuda_make_array(0, batch*outputs);
	l.i_gpu = cuda_make_array(0, batch*outputs);
	l.g_gpu = cuda_make_array(0, batch*outputs);
	l.o_gpu = cuda_make_array(0, batch*outputs);
	l.c_gpu = cuda_make_array(0, batch*outputs);
	l.h_gpu = cuda_make_array(0, batch*outputs);
	l.temp_gpu = cuda_make_array(0, batch*outputs);
	l.temp2_gpu = cuda_make_array(0, batch*outputs);
	l.temp3_gpu = cuda_make_array(0, batch*outputs);
	l.dc_gpu = cuda_make_array(0, batch*outputs);
	l.dh_gpu = cuda_make_array(0, batch*outputs);
#ifdef CUDNN
	cudnnSetTensor4dDescriptor(l.wf->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.wf->out_c, l.wf->out_h, l.wf->out_w);
	cudnnSetTensor4dDescriptor(l.wi->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.wi->out_c, l.wi->out_h, l.wi->out_w);
	cudnnSetTensor4dDescriptor(l.wg->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.wg->out_c, l.wg->out_h, l.wg->out_w);
	cudnnSetTensor4dDescriptor(l.wo->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.wo->out_c, l.wo->out_h, l.wo->out_w);

	cudnnSetTensor4dDescriptor(l.uf->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.uf->out_c, l.uf->out_h, l.uf->out_w);
	cudnnSetTensor4dDescriptor(l.ui->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.ui->out_c, l.ui->out_h, l.ui->out_w);
	cudnnSetTensor4dDescriptor(l.ug->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.ug->out_c, l.ug->out_h, l.ug->out_w);
	cudnnSetTensor4dDescriptor(l.uo->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.uo->out_c, l.uo->out_h, l.uo->out_w);
#endif

#endif

	return l;
}

void update_lstm_layer(layer l, update_args a)
{
	update_connected_layer(*(l.wf), a);
	update_connected_layer(*(l.wi), a);
	update_connected_layer(*(l.wg), a);
	update_connected_layer(*(l.wo), a);
	update_connected_layer(*(l.uf), a);
	update_connected_layer(*(l.ui), a);
	update_connected_layer(*(l.ug), a);
	update_connected_layer(*(l.uo), a);
}

void forward_lstm_layer(layer l, network state)
{
	network s = { 0 };
	s.train = state.train;
	int i;
	layer wf = *(l.wf);
	layer wi = *(l.wi);
	layer wg = *(l.wg);
	layer wo = *(l.wo);

	layer uf = *(l.uf);
	layer ui = *(l.ui);
	layer ug = *(l.ug);
	layer uo = *(l.uo);

	fill_cpu(l.outputs * l.batch * l.steps, 0, wf.delta, 1);
	fill_cpu(l.outputs * l.batch * l.steps, 0, wi.delta, 1);
	fill_cpu(l.outputs * l.batch * l.steps, 0, wg.delta, 1);
	fill_cpu(l.outputs * l.batch * l.steps, 0, wo.delta, 1);

	fill_cpu(l.outputs * l.batch * l.steps, 0, uf.delta, 1);
	fill_cpu(l.outputs * l.batch * l.steps, 0, ui.delta, 1);
	fill_cpu(l.outputs * l.batch * l.steps, 0, ug.delta, 1);
	fill_cpu(l.outputs * l.batch * l.steps, 0, uo.delta, 1);
	if (state.train) {
		fill_cpu(l.outputs * l.batch * l.steps, 0, l.delta, 1);
	}

	for (i = 0; i < l.steps; ++i) {
		s.input = l.h_cpu;
		forward_connected_layer(wf, s);
		forward_connected_layer(wi, s);
		forward_connected_layer(wg, s);
		forward_connected_layer(wo, s);

		s.input = state.input;
		forward_connected_layer(uf, s);
		forward_connected_layer(ui, s);
		forward_connected_layer(ug, s);
		forward_connected_layer(uo, s);

		copy_cpu(l.outputs*l.batch, wf.output, 1, l.f_cpu, 1);
		axpy_cpu(l.outputs*l.batch, 1, uf.output, 1, l.f_cpu, 1);

		copy_cpu(l.outputs*l.batch, wi.output, 1, l.i_cpu, 1);
		axpy_cpu(l.outputs*l.batch, 1, ui.output, 1, l.i_cpu, 1);

		copy_cpu(l.outputs*l.batch, wg.output, 1, l.g_cpu, 1);
		axpy_cpu(l.outputs*l.batch, 1, ug.output, 1, l.g_cpu, 1);

		copy_cpu(l.outputs*l.batch, wo.output, 1, l.o_cpu, 1);
		axpy_cpu(l.outputs*l.batch, 1, uo.output, 1, l.o_cpu, 1);

		activate_array(l.f_cpu, l.outputs*l.batch, LOGISTIC);
		activate_array(l.i_cpu, l.outputs*l.batch, LOGISTIC);
		activate_array(l.g_cpu, l.outputs*l.batch, TANH);
		activate_array(l.o_cpu, l.outputs*l.batch, LOGISTIC);

		copy_cpu(l.outputs*l.batch, l.i_cpu, 1, l.temp_cpu, 1);
		mul_cpu(l.outputs*l.batch, l.g_cpu, 1, l.temp_cpu, 1);
		mul_cpu(l.outputs*l.batch, l.f_cpu, 1, l.c_cpu, 1);
		axpy_cpu(l.outputs*l.batch, 1, l.temp_cpu, 1, l.c_cpu, 1);

		copy_cpu(l.outputs*l.batch, l.c_cpu, 1, l.h_cpu, 1);
		activate_array(l.h_cpu, l.outputs*l.batch, TANH);
		mul_cpu(l.outputs*l.batch, l.o_cpu, 1, l.h_cpu, 1);

		copy_cpu(l.outputs*l.batch, l.c_cpu, 1, l.cell_cpu, 1);
		copy_cpu(l.outputs*l.batch, l.h_cpu, 1, l.output, 1);

		state.input += l.inputs*l.batch;
		l.output += l.outputs*l.batch;
		l.cell_cpu += l.outputs*l.batch;

		increment_layer(&wf, 1);
		increment_layer(&wi, 1);
		increment_layer(&wg, 1);
		increment_layer(&wo, 1);

		increment_layer(&uf, 1);
		increment_layer(&ui, 1);
		increment_layer(&ug, 1);
		increment_layer(&uo, 1);
	}
}

void backward_lstm_layer(layer l, network state)
{
	network s = { 0 };
	s.train = state.train;
	int i;
	layer wf = *(l.wf);
	layer wi = *(l.wi);
	layer wg = *(l.wg);
	layer wo = *(l.wo);

	layer uf = *(l.uf);
	layer ui = *(l.ui);
	layer ug = *(l.ug);
	layer uo = *(l.uo);

	increment_layer(&wf, l.steps - 1);
	increment_layer(&wi, l.steps - 1);
	increment_layer(&wg, l.steps - 1);
	increment_layer(&wo, l.steps - 1);

	increment_layer(&uf, l.steps - 1);
	increment_layer(&ui, l.steps - 1);
	increment_layer(&ug, l.steps - 1);
	increment_layer(&uo, l.steps - 1);

	state.input += l.inputs*l.batch*(l.steps - 1);
	if (state.delta) state.delta += l.inputs*l.batch*(l.steps - 1);

	l.output += l.outputs*l.batch*(l.steps - 1);
	l.cell_cpu += l.outputs*l.batch*(l.steps - 1);
	l.delta += l.outputs*l.batch*(l.steps - 1);

	for (i = l.steps - 1; i >= 0; --i) {
		if (i != 0) copy_cpu(l.outputs*l.batch, l.cell_cpu - l.outputs*l.batch, 1, l.prev_cell_cpu, 1);
		copy_cpu(l.outputs*l.batch, l.cell_cpu, 1, l.c_cpu, 1);
		if (i != 0) copy_cpu(l.outputs*l.batch, l.output - l.outputs*l.batch, 1, l.prev_state_cpu, 1);
		copy_cpu(l.outputs*l.batch, l.output, 1, l.h_cpu, 1);

		l.dh_cpu = (i == 0) ? 0 : l.delta - l.outputs*l.batch;

		copy_cpu(l.outputs*l.batch, wf.output, 1, l.f_cpu, 1);
		axpy_cpu(l.outputs*l.batch, 1, uf.output, 1, l.f_cpu, 1);

		copy_cpu(l.outputs*l.batch, wi.output, 1, l.i_cpu, 1);
		axpy_cpu(l.outputs*l.batch, 1, ui.output, 1, l.i_cpu, 1);

		copy_cpu(l.outputs*l.batch, wg.output, 1, l.g_cpu, 1);
		axpy_cpu(l.outputs*l.batch, 1, ug.output, 1, l.g_cpu, 1);

		copy_cpu(l.outputs*l.batch, wo.output, 1, l.o_cpu, 1);
		axpy_cpu(l.outputs*l.batch, 1, uo.output, 1, l.o_cpu, 1);

		activate_array(l.f_cpu, l.outputs*l.batch, LOGISTIC);
		activate_array(l.i_cpu, l.outputs*l.batch, LOGISTIC);
		activate_array(l.g_cpu, l.outputs*l.batch, TANH);
		activate_array(l.o_cpu, l.outputs*l.batch, LOGISTIC);

		copy_cpu(l.outputs*l.batch, l.delta, 1, l.temp3_cpu, 1);

		copy_cpu(l.outputs*l.batch, l.c_cpu, 1, l.temp_cpu, 1);
		activate_array(l.temp_cpu, l.outputs*l.batch, TANH);

		copy_cpu(l.outputs*l.batch, l.temp3_cpu, 1, l.temp2_cpu, 1);
		mul_cpu(l.outputs*l.batch, l.o_cpu, 1, l.temp2_cpu, 1);

		gradient_array(l.temp_cpu, l.outputs*l.batch, TANH, l.temp2_cpu);
		axpy_cpu(l.outputs*l.batch, 1, l.dc_cpu, 1, l.temp2_cpu, 1);

		copy_cpu(l.outputs*l.batch, l.c_cpu, 1, l.temp_cpu, 1);
		activate_array(l.temp_cpu, l.outputs*l.batch, TANH);
		mul_cpu(l.outputs*l.batch, l.temp3_cpu, 1, l.temp_cpu, 1);
		gradient_array(l.o_cpu, l.outputs*l.batch, LOGISTIC, l.temp_cpu);
		copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, wo.delta, 1);
		s.input = l.prev_state_cpu;
		s.delta = l.dh_cpu;
		backward_connected_layer(wo, s);

		copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, uo.delta, 1);
		s.input = state.input;
		s.delta = state.delta;
		backward_connected_layer(uo, s);

		copy_cpu(l.outputs*l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);
		mul_cpu(l.outputs*l.batch, l.i_cpu, 1, l.temp_cpu, 1);
		gradient_array(l.g_cpu, l.outputs*l.batch, TANH, l.temp_cpu);
		copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, wg.delta, 1);
		s.input = l.prev_state_cpu;
		s.delta = l.dh_cpu;
		backward_connected_layer(wg, s);

		copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, ug.delta, 1);
		s.input = state.input;
		s.delta = state.delta;
		backward_connected_layer(ug, s);

		copy_cpu(l.outputs*l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);
		mul_cpu(l.outputs*l.batch, l.g_cpu, 1, l.temp_cpu, 1);
		gradient_array(l.i_cpu, l.outputs*l.batch, LOGISTIC, l.temp_cpu);
		copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, wi.delta, 1);
		s.input = l.prev_state_cpu;
		s.delta = l.dh_cpu;
		backward_connected_layer(wi, s);

		copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, ui.delta, 1);
		s.input = state.input;
		s.delta = state.delta;
		backward_connected_layer(ui, s);

		copy_cpu(l.outputs*l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);
		mul_cpu(l.outputs*l.batch, l.prev_cell_cpu, 1, l.temp_cpu, 1);
		gradient_array(l.f_cpu, l.outputs*l.batch, LOGISTIC, l.temp_cpu);
		copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, wf.delta, 1);
		s.input = l.prev_state_cpu;
		s.delta = l.dh_cpu;
		backward_connected_layer(wf, s);

		copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, uf.delta, 1);
		s.input = state.input;
		s.delta = state.delta;
		backward_connected_layer(uf, s);

		copy_cpu(l.outputs*l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);
		mul_cpu(l.outputs*l.batch, l.f_cpu, 1, l.temp_cpu, 1);
		copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, l.dc_cpu, 1);

		state.input -= l.inputs*l.batch;
		if (state.delta) state.delta -= l.inputs*l.batch;
		l.output -= l.outputs*l.batch;
		l.cell_cpu -= l.outputs*l.batch;
		l.delta -= l.outputs*l.batch;

		increment_layer(&wf, -1);
		increment_layer(&wi, -1);
		increment_layer(&wg, -1);
		increment_layer(&wo, -1);

		increment_layer(&uf, -1);
		increment_layer(&ui, -1);
		increment_layer(&ug, -1);
		increment_layer(&uo, -1);
	}
}

#ifdef GPU
void update_lstm_layer_gpu(layer l, update_args a)
{
	update_connected_layer_gpu(*(l.wf), a);
	update_connected_layer_gpu(*(l.wi), a);
	update_connected_layer_gpu(*(l.wg), a);
	update_connected_layer_gpu(*(l.wo), a);
	update_connected_layer_gpu(*(l.uf), a);
	update_connected_layer_gpu(*(l.ui), a);
	update_connected_layer_gpu(*(l.ug), a);
	update_connected_layer_gpu(*(l.uo), a);
}

void forward_lstm_layer_gpu(layer l, network state)
{
	network s = { 0 };
	s.train = state.train;
	int i;
	layer wf = *(l.wf);
	layer wi = *(l.wi);
	layer wg = *(l.wg);
	layer wo = *(l.wo);

	layer uf = *(l.uf);
	layer ui = *(l.ui);
	layer ug = *(l.ug);
	layer uo = *(l.uo);

	fill_gpu(l.outputs * l.batch * l.steps, 0, wf.delta_gpu, 1);
	fill_gpu(l.outputs * l.batch * l.steps, 0, wi.delta_gpu, 1);
	fill_gpu(l.outputs * l.batch * l.steps, 0, wg.delta_gpu, 1);
	fill_gpu(l.outputs * l.batch * l.steps, 0, wo.delta_gpu, 1);

	fill_gpu(l.outputs * l.batch * l.steps, 0, uf.delta_gpu, 1);
	fill_gpu(l.outputs * l.batch * l.steps, 0, ui.delta_gpu, 1);
	fill_gpu(l.outputs * l.batch * l.steps, 0, ug.delta_gpu, 1);
	fill_gpu(l.outputs * l.batch * l.steps, 0, uo.delta_gpu, 1);
	if (state.train) {
		fill_gpu(l.outputs * l.batch * l.steps, 0, l.delta_gpu, 1);
	}

	for (i = 0; i < l.steps; ++i) {
		s.input_gpu = l.h_gpu;
		forward_connected_layer_gpu(wf, s);
		forward_connected_layer_gpu(wi, s);
		forward_connected_layer_gpu(wg, s);
		forward_connected_layer_gpu(wo, s);

		s.input_gpu = state.input_gpu;
		forward_connected_layer_gpu(uf, s);
		forward_connected_layer_gpu(ui, s);
		forward_connected_layer_gpu(ug, s);
		forward_connected_layer_gpu(uo, s);

		copy_gpu(l.outputs*l.batch, wf.output_gpu, 1, l.f_gpu, 1);
		axpy_gpu(l.outputs*l.batch, 1, uf.output_gpu, 1, l.f_gpu, 1);

		copy_gpu(l.outputs*l.batch, wi.output_gpu, 1, l.i_gpu, 1);
		axpy_gpu(l.outputs*l.batch, 1, ui.output_gpu, 1, l.i_gpu, 1);

		copy_gpu(l.outputs*l.batch, wg.output_gpu, 1, l.g_gpu, 1);
		axpy_gpu(l.outputs*l.batch, 1, ug.output_gpu, 1, l.g_gpu, 1);

		copy_gpu(l.outputs*l.batch, wo.output_gpu, 1, l.o_gpu, 1);
		axpy_gpu(l.outputs*l.batch, 1, uo.output_gpu, 1, l.o_gpu, 1);

		activate_array_gpu(l.f_gpu, l.outputs*l.batch, LOGISTIC);
		activate_array_gpu(l.i_gpu, l.outputs*l.batch, LOGISTIC);
		activate_array_gpu(l.g_gpu, l.outputs*l.batch, TANH);
		activate_array_gpu(l.o_gpu, l.outputs*l.batch, LOGISTIC);

		copy_gpu(l.outputs*l.batch, l.i_gpu, 1, l.temp_gpu, 1);
		mul_gpu(l.outputs*l.batch, l.g_gpu, 1, l.temp_gpu, 1);
		mul_gpu(l.outputs*l.batch, l.f_gpu, 1, l.c_gpu, 1);
		axpy_gpu(l.outputs*l.batch, 1, l.temp_gpu, 1, l.c_gpu, 1);

		copy_gpu(l.outputs*l.batch, l.c_gpu, 1, l.h_gpu, 1);
		activate_array_gpu(l.h_gpu, l.outputs*l.batch, TANH);
		mul_gpu(l.outputs*l.batch, l.o_gpu, 1, l.h_gpu, 1);

		copy_gpu(l.outputs*l.batch, l.c_gpu, 1, l.cell_gpu, 1);
		copy_gpu(l.outputs*l.batch, l.h_gpu, 1, l.output_gpu, 1);

		state.input_gpu += l.inputs*l.batch;
		l.output_gpu += l.outputs*l.batch;
		l.cell_gpu += l.outputs*l.batch;

		increment_layer(&wf, 1);
		increment_layer(&wi, 1);
		increment_layer(&wg, 1);
		increment_layer(&wo, 1);

		increment_layer(&uf, 1);
		increment_layer(&ui, 1);
		increment_layer(&ug, 1);
		increment_layer(&uo, 1);
	}
}

void backward_lstm_layer_gpu(layer l, network state)
{
	network s = { 0 };
	s.train = state.train;
	int i;
	layer wf = *(l.wf);
	layer wi = *(l.wi);
	layer wg = *(l.wg);
	layer wo = *(l.wo);

	layer uf = *(l.uf);
	layer ui = *(l.ui);
	layer ug = *(l.ug);
	layer uo = *(l.uo);

	increment_layer(&wf, l.steps - 1);
	increment_layer(&wi, l.steps - 1);
	increment_layer(&wg, l.steps - 1);
	increment_layer(&wo, l.steps - 1);

	increment_layer(&uf, l.steps - 1);
	increment_layer(&ui, l.steps - 1);
	increment_layer(&ug, l.steps - 1);
	increment_layer(&uo, l.steps - 1);

	state.input_gpu += l.inputs*l.batch*(l.steps - 1);
	if (state.delta_gpu) state.delta_gpu += l.inputs*l.batch*(l.steps - 1);

	l.output_gpu += l.outputs*l.batch*(l.steps - 1);
	l.cell_gpu += l.outputs*l.batch*(l.steps - 1);
	l.delta_gpu += l.outputs*l.batch*(l.steps - 1);

	for (i = l.steps - 1; i >= 0; --i) {
		if (i != 0) copy_gpu(l.outputs*l.batch, l.cell_gpu - l.outputs*l.batch, 1, l.prev_cell_gpu, 1);
		copy_gpu(l.outputs*l.batch, l.cell_gpu, 1, l.c_gpu, 1);
		if (i != 0) copy_gpu(l.outputs*l.batch, l.output_gpu - l.outputs*l.batch, 1, l.prev_state_gpu, 1);
		copy_gpu(l.outputs*l.batch, l.output_gpu, 1, l.h_gpu, 1);

		l.dh_gpu = (i == 0) ? 0 : l.delta_gpu - l.outputs*l.batch;

		copy_gpu(l.outputs*l.batch, wf.output_gpu, 1, l.f_gpu, 1);
		axpy_gpu(l.outputs*l.batch, 1, uf.output_gpu, 1, l.f_gpu, 1);

		copy_gpu(l.outputs*l.batch, wi.output_gpu, 1, l.i_gpu, 1);
		axpy_gpu(l.outputs*l.batch, 1, ui.output_gpu, 1, l.i_gpu, 1);

		copy_gpu(l.outputs*l.batch, wg.output_gpu, 1, l.g_gpu, 1);
		axpy_gpu(l.outputs*l.batch, 1, ug.output_gpu, 1, l.g_gpu, 1);

		copy_gpu(l.outputs*l.batch, wo.output_gpu, 1, l.o_gpu, 1);
		axpy_gpu(l.outputs*l.batch, 1, uo.output_gpu, 1, l.o_gpu, 1);

		activate_array_gpu(l.f_gpu, l.outputs*l.batch, LOGISTIC);
		activate_array_gpu(l.i_gpu, l.outputs*l.batch, LOGISTIC);
		activate_array_gpu(l.g_gpu, l.outputs*l.batch, TANH);
		activate_array_gpu(l.o_gpu, l.outputs*l.batch, LOGISTIC);

		copy_gpu(l.outputs*l.batch, l.delta_gpu, 1, l.temp3_gpu, 1);

		copy_gpu(l.outputs*l.batch, l.c_gpu, 1, l.temp_gpu, 1);
		activate_array_gpu(l.temp_gpu, l.outputs*l.batch, TANH);

		copy_gpu(l.outputs*l.batch, l.temp3_gpu, 1, l.temp2_gpu, 1);
		mul_gpu(l.outputs*l.batch, l.o_gpu, 1, l.temp2_gpu, 1);

		gradient_array_gpu(l.temp_gpu, l.outputs*l.batch, TANH, l.temp2_gpu);
		axpy_gpu(l.outputs*l.batch, 1, l.dc_gpu, 1, l.temp2_gpu, 1);

		copy_gpu(l.outputs*l.batch, l.c_gpu, 1, l.temp_gpu, 1);
		activate_array_gpu(l.temp_gpu, l.outputs*l.batch, TANH);
		mul_gpu(l.outputs*l.batch, l.temp3_gpu, 1, l.temp_gpu, 1);
		gradient_array_gpu(l.o_gpu, l.outputs*l.batch, LOGISTIC, l.temp_gpu);
		copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, wo.delta_gpu, 1);
		s.input_gpu = l.prev_state_gpu;
		s.delta_gpu = l.dh_gpu;
		backward_connected_layer_gpu(wo, s);

		copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, uo.delta_gpu, 1);
		s.input_gpu = state.input_gpu;
		s.delta_gpu = state.delta_gpu;
		backward_connected_layer_gpu(uo, s);

		copy_gpu(l.outputs*l.batch, l.temp2_gpu, 1, l.temp_gpu, 1);
		mul_gpu(l.outputs*l.batch, l.i_gpu, 1, l.temp_gpu, 1);
		gradient_array_gpu(l.g_gpu, l.outputs*l.batch, TANH, l.temp_gpu);
		copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, wg.delta_gpu, 1);
		s.input_gpu = l.prev_state_gpu;
		s.delta_gpu = l.dh_gpu;
		backward_connected_layer_gpu(wg, s);

		copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, ug.delta_gpu, 1);
		s.input_gpu = state.input_gpu;
		s.delta_gpu = state.delta_gpu;
		backward_connected_layer_gpu(ug, s);

		copy_gpu(l.outputs*l.batch, l.temp2_gpu, 1, l.temp_gpu, 1);
		mul_gpu(l.outputs*l.batch, l.g_gpu, 1, l.temp_gpu, 1);
		gradient_array_gpu(l.i_gpu, l.outputs*l.batch, LOGISTIC, l.temp_gpu);
		copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, wi.delta_gpu, 1);
		s.input_gpu = l.prev_state_gpu;
		s.delta_gpu = l.dh_gpu;
		backward_connected_layer_gpu(wi, s);

		copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, ui.delta_gpu, 1);
		s.input_gpu = state.input_gpu;
		s.delta_gpu = state.delta_gpu;
		backward_connected_layer_gpu(ui, s);

		copy_gpu(l.outputs*l.batch, l.temp2_gpu, 1, l.temp_gpu, 1);
		mul_gpu(l.outputs*l.batch, l.prev_cell_gpu, 1, l.temp_gpu, 1);
		gradient_array_gpu(l.f_gpu, l.outputs*l.batch, LOGISTIC, l.temp_gpu);
		copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, wf.delta_gpu, 1);
		s.input_gpu = l.prev_state_gpu;
		s.delta_gpu = l.dh_gpu;
		backward_connected_layer_gpu(wf, s);

		copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, uf.delta_gpu, 1);
		s.input_gpu = state.input_gpu;
		s.delta_gpu = state.delta_gpu;
		backward_connected_layer_gpu(uf, s);

		copy_gpu(l.outputs*l.batch, l.temp2_gpu, 1, l.temp_gpu, 1);
		mul_gpu(l.outputs*l.batch, l.f_gpu, 1, l.temp_gpu, 1);
		copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, l.dc_gpu, 1);

		state.input_gpu -= l.inputs*l.batch;
		if (state.delta_gpu) state.delta_gpu -= l.inputs*l.batch;
		l.output_gpu -= l.outputs*l.batch;
		l.cell_gpu -= l.outputs*l.batch;
		l.delta_gpu -= l.outputs*l.batch;

		increment_layer(&wf, -1);
		increment_layer(&wi, -1);
		increment_layer(&wg, -1);
		increment_layer(&wo, -1);

		increment_layer(&uf, -1);
		increment_layer(&ui, -1);
		increment_layer(&ug, -1);
		increment_layer(&uo, -1);
	}
}
#endif







//maxpool_layer.c
image get_maxpool_image(maxpool_layer l)
{
	int h = l.out_h;
	int w = l.out_w;
	int c = l.c;
	return float_to_image(w, h, c, l.output);
}

image get_maxpool_delta(maxpool_layer l)
{
	int h = l.out_h;
	int w = l.out_w;
	int c = l.c;
	return float_to_image(w, h, c, l.delta);
}

maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding)
{
	maxpool_layer l = {  };
	l.type = MAXPOOL;
	l.batch = batch;
	l.h = h;
	l.w = w;
	l.c = c;
	l.pad = padding;
	l.out_w = (w + padding - size) / stride + 1;
	l.out_h = (h + padding - size) / stride + 1;
	l.out_c = c;
	l.outputs = l.out_h * l.out_w * l.out_c;
	l.inputs = h * w*c;
	l.size = size;
	l.stride = stride;
	int output_size = l.out_h * l.out_w * l.out_c * batch;
	l.indexes = (int*)calloc(output_size, sizeof(int));
	l.output = (float*)calloc(output_size, sizeof(float));
	l.delta = (float*)calloc(output_size, sizeof(float));
	l.forward = forward_maxpool_layer;
	l.backward = backward_maxpool_layer;
#ifdef GPU
	l.forward_gpu = forward_maxpool_layer_gpu;
	l.backward_gpu = backward_maxpool_layer_gpu;
	l.indexes_gpu = cuda_make_int_array(0, output_size);
	l.output_gpu = cuda_make_array(l.output, output_size);
	l.delta_gpu = cuda_make_array(l.delta, output_size);
#endif
	fprintf(stderr, "max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
	return l;
}

void resize_maxpool_layer(maxpool_layer *l, int w, int h)
{
	l->h = h;
	l->w = w;
	l->inputs = h * w*l->c;

	l->out_w = (w + l->pad - l->size) / l->stride + 1;
	l->out_h = (h + l->pad - l->size) / l->stride + 1;
	l->outputs = l->out_w * l->out_h * l->c;
	int output_size = l->outputs * l->batch;

	l->indexes = (int*)realloc(l->indexes, output_size * sizeof(int));
	l->output = (float*)realloc(l->output, output_size * sizeof(float));
	l->delta = (float*)realloc(l->delta, output_size * sizeof(float));

#ifdef GPU
	cuda_free((float *)l->indexes_gpu);
	cuda_free(l->output_gpu);
	cuda_free(l->delta_gpu);
	l->indexes_gpu = cuda_make_int_array(0, output_size);
	l->output_gpu = cuda_make_array(l->output, output_size);
	l->delta_gpu = cuda_make_array(l->delta, output_size);
#endif
}

void forward_maxpool_layer(const maxpool_layer l, network net)
{
	int b, i, j, k, m, n;
	int w_offset = -l.pad / 2;
	int h_offset = -l.pad / 2;

	int h = l.out_h;
	int w = l.out_w;
	int c = l.c;

	for (b = 0; b < l.batch; ++b) {
		for (k = 0; k < c; ++k) {
			for (i = 0; i < h; ++i) {
				for (j = 0; j < w; ++j) {
					int out_index = j + w * (i + h * (k + c * b));
					float max = -FLT_MAX;
					int max_i = -1;
					for (n = 0; n < l.size; ++n) {
						for (m = 0; m < l.size; ++m) {
							int cur_h = h_offset + i * l.stride + n;
							int cur_w = w_offset + j * l.stride + m;
							int index = cur_w + l.w*(cur_h + l.h*(k + b * l.c));
							int valid = (cur_h >= 0 && cur_h < l.h &&
								cur_w >= 0 && cur_w < l.w);
							float val = (valid != 0) ? net.input[index] : -FLT_MAX;
							max_i = (val > max) ? index : max_i;
							max = (val > max) ? val : max;
						}
					}
					l.output[out_index] = max;
					l.indexes[out_index] = max_i;
				}
			}
		}
	}
}

void backward_maxpool_layer(const maxpool_layer l, network net)
{
	int i;
	int h = l.out_h;
	int w = l.out_w;
	int c = l.c;
	for (i = 0; i < h*w*c*l.batch; ++i) {
		int index = l.indexes[i];
		net.delta[index] += l.delta[i];
	}
}







//normalization_layer.c
layer make_normalization_layer(int batch, int w, int h, int c, int size, float alpha, float beta, float kappa)
{
	fprintf(stderr, "Local Response Normalization Layer: %d x %d x %d image, %d size\n", w, h, c, size);
	layer layer = { };
	layer.type = NORMALIZATION;
	layer.batch = batch;
	layer.h = layer.out_h = h;
	layer.w = layer.out_w = w;
	layer.c = layer.out_c = c;
	layer.kappa = kappa;
	layer.size = size;
	layer.alpha = alpha;
	layer.beta = beta;
	layer.output = (float*)calloc(h * w * c * batch, sizeof(float));
	layer.delta = (float*)calloc(h * w * c * batch, sizeof(float));
	layer.squared = (float*)calloc(h * w * c * batch, sizeof(float));
	layer.norms = (float*)calloc(h * w * c * batch, sizeof(float));
	layer.inputs = w * h*c;
	layer.outputs = layer.inputs;

	layer.forward = forward_normalization_layer;
	layer.backward = backward_normalization_layer;
#ifdef GPU
	layer.forward_gpu = forward_normalization_layer_gpu;
	layer.backward_gpu = backward_normalization_layer_gpu;

	layer.output_gpu = cuda_make_array(layer.output, h * w * c * batch);
	layer.delta_gpu = cuda_make_array(layer.delta, h * w * c * batch);
	layer.squared_gpu = cuda_make_array(layer.squared, h * w * c * batch);
	layer.norms_gpu = cuda_make_array(layer.norms, h * w * c * batch);
#endif
	return layer;
}

void resize_normalization_layer(layer *layer, int w, int h)
{
	int c = layer->c;
	int batch = layer->batch;
	layer->h = h;
	layer->w = w;
	layer->out_h = h;
	layer->out_w = w;
	layer->inputs = w * h*c;
	layer->outputs = layer->inputs;
	layer->output = (float*)realloc(layer->output, h * w * c * batch * sizeof(float));
	layer->delta = (float*)realloc(layer->delta, h * w * c * batch * sizeof(float));
	layer->squared = (float*)realloc(layer->squared, h * w * c * batch * sizeof(float));
	layer->norms = (float*)realloc(layer->norms, h * w * c * batch * sizeof(float));
#ifdef GPU
	cuda_free(layer->output_gpu);
	cuda_free(layer->delta_gpu);
	cuda_free(layer->squared_gpu);
	cuda_free(layer->norms_gpu);
	layer->output_gpu = cuda_make_array(layer->output, h * w * c * batch);
	layer->delta_gpu = cuda_make_array(layer->delta, h * w * c * batch);
	layer->squared_gpu = cuda_make_array(layer->squared, h * w * c * batch);
	layer->norms_gpu = cuda_make_array(layer->norms, h * w * c * batch);
#endif
}

void forward_normalization_layer(const layer layer, network net)
{
	int k, b;
	int w = layer.w;
	int h = layer.h;
	int c = layer.c;
	scal_cpu(w*h*c*layer.batch, 0, layer.squared, 1);

	for (b = 0; b < layer.batch; ++b) {
		float *squared = layer.squared + w * h*c*b;
		float *norms = layer.norms + w * h*c*b;
		float *input = net.input + w * h*c*b;
		pow_cpu(w*h*c, 2, input, 1, squared, 1);

		const_cpu(w*h, layer.kappa, norms, 1);
		for (k = 0; k < layer.size / 2; ++k) {
			axpy_cpu(w*h, layer.alpha, squared + w * h*k, 1, norms, 1);
		}

		for (k = 1; k < layer.c; ++k) {
			copy_cpu(w*h, norms + w * h*(k - 1), 1, norms + w * h*k, 1);
			int prev = k - ((layer.size - 1) / 2) - 1;
			int next = k + (layer.size / 2);
			if (prev >= 0)      axpy_cpu(w*h, -layer.alpha, squared + w * h*prev, 1, norms + w * h*k, 1);
			if (next < layer.c) axpy_cpu(w*h, layer.alpha, squared + w * h*next, 1, norms + w * h*k, 1);
		}
	}
	pow_cpu(w*h*c*layer.batch, -layer.beta, layer.norms, 1, layer.output, 1);
	mul_cpu(w*h*c*layer.batch, net.input, 1, layer.output, 1);
}

void backward_normalization_layer(const layer layer, network net)
{
	// TODO This is approximate ;-)
	// Also this should add in to delta instead of overwritting.

	int w = layer.w;
	int h = layer.h;
	int c = layer.c;
	pow_cpu(w*h*c*layer.batch, -layer.beta, layer.norms, 1, net.delta, 1);
	mul_cpu(w*h*c*layer.batch, layer.delta, 1, net.delta, 1);
}

#ifdef GPU
void forward_normalization_layer_gpu(const layer layer, network net)
{
	int k, b;
	int w = layer.w;
	int h = layer.h;
	int c = layer.c;
	scal_gpu(w*h*c*layer.batch, 0, layer.squared_gpu, 1);

	for (b = 0; b < layer.batch; ++b) {
		float *squared = layer.squared_gpu + w * h*c*b;
		float *norms = layer.norms_gpu + w * h*c*b;
		float *input = net.input_gpu + w * h*c*b;
		pow_gpu(w*h*c, 2, input, 1, squared, 1);

		const_gpu(w*h, layer.kappa, norms, 1);
		for (k = 0; k < layer.size / 2; ++k) {
			axpy_gpu(w*h, layer.alpha, squared + w * h*k, 1, norms, 1);
		}

		for (k = 1; k < layer.c; ++k) {
			copy_gpu(w*h, norms + w * h*(k - 1), 1, norms + w * h*k, 1);
			int prev = k - ((layer.size - 1) / 2) - 1;
			int next = k + (layer.size / 2);
			if (prev >= 0)      axpy_gpu(w*h, -layer.alpha, squared + w * h*prev, 1, norms + w * h*k, 1);
			if (next < layer.c) axpy_gpu(w*h, layer.alpha, squared + w * h*next, 1, norms + w * h*k, 1);
		}
	}
	pow_gpu(w*h*c*layer.batch, -layer.beta, layer.norms_gpu, 1, layer.output_gpu, 1);
	mul_gpu(w*h*c*layer.batch, net.input_gpu, 1, layer.output_gpu, 1);
}

void backward_normalization_layer_gpu(const layer layer, network net)
{
	// TODO This is approximate ;-)

	int w = layer.w;
	int h = layer.h;
	int c = layer.c;
	pow_gpu(w*h*c*layer.batch, -layer.beta, layer.norms_gpu, 1, net.delta_gpu, 1);
	mul_gpu(w*h*c*layer.batch, layer.delta_gpu, 1, net.delta_gpu, 1);
}
#endif







//region_layer.c
layer make_region_layer(int batch, int w, int h, int n, int classes, int coords)
{
	layer l = { };
	l.type = REGION;

	l.n = n;
	l.batch = batch;
	l.h = h;
	l.w = w;
	l.c = n * (classes + coords + 1);
	l.out_w = l.w;
	l.out_h = l.h;
	l.out_c = l.c;
	l.classes = classes;
	l.coords = coords;
	l.cost = (float*)calloc(1, sizeof(float));
	l.biases = (float*)calloc(n * 2, sizeof(float));
	l.bias_updates = (float*)calloc(n * 2, sizeof(float));
	l.outputs = h * w*n*(classes + coords + 1);
	l.inputs = l.outputs;
	l.truths = 30 * (l.coords + 1);
	l.delta = (float*)calloc(batch*l.outputs, sizeof(float));
	l.output = (float*)calloc(batch*l.outputs, sizeof(float));
	int i;
	for (i = 0; i < n * 2; ++i) {
		l.biases[i] = .5;
	}

	l.forward = forward_region_layer;
	l.backward = backward_region_layer;
#ifdef GPU
	l.forward_gpu = forward_region_layer_gpu;
	l.backward_gpu = backward_region_layer_gpu;
	l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
	l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

	fprintf(stderr, "detection\n");
	srand(0);

	return l;
}

void resize_region_layer(layer *l, int w, int h)
{
	l->w = w;
	l->h = h;

	l->outputs = h * w*l->n*(l->classes + l->coords + 1);
	l->inputs = l->outputs;

	l->output = (float*)realloc(l->output, l->batch*l->outputs * sizeof(float));
	l->delta = (float*)realloc(l->delta, l->batch*l->outputs * sizeof(float));

#ifdef GPU
	cuda_free(l->delta_gpu);
	cuda_free(l->output_gpu);

	l->delta_gpu = cuda_make_array(l->delta, l->batch*l->outputs);
	l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);
#endif
}

box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h, int stride)
{
	box b;
	b.x = (i + x[index + 0 * stride]) / w;
	b.y = (j + x[index + 1 * stride]) / h;
	b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
	b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
	return b;
}

float delta_region_box(box truth, float *x, float *biases, int n, int index, int i, int j, int w, int h, float *delta, float scale, int stride)
{
	box pred = get_region_box(x, biases, n, index, i, j, w, h, stride);
	float iou = box_iou(pred, truth);

	float tx = (truth.x*w - i);
	float ty = (truth.y*h - j);
	float tw = log(truth.w*w / biases[2 * n]);
	float th = log(truth.h*h / biases[2 * n + 1]);

	delta[index + 0 * stride] = scale * (tx - x[index + 0 * stride]);
	delta[index + 1 * stride] = scale * (ty - x[index + 1 * stride]);
	delta[index + 2 * stride] = scale * (tw - x[index + 2 * stride]);
	delta[index + 3 * stride] = scale * (th - x[index + 3 * stride]);
	return iou;
}

void delta_region_mask(float *truth, float *x, int n, int index, float *delta, int stride, int scale)
{
	int i;
	for (i = 0; i < n; ++i) {
		delta[index + i * stride] = scale * (truth[i] - x[index + i * stride]);
	}
}


void delta_region_class(float *output, float *delta, int index, int class_n, int classes, tree *hier, float scale, int stride, float *avg_cat, int tag)
{
	int i, n;
	if (hier) {
		float pred = 1;
		while (class_n >= 0) {
			pred *= output[index + stride * class_n];
			int g = hier->group[class_n];
			int offset = hier->group_offset[g];
			for (i = 0; i < hier->group_size[g]; ++i) {
				delta[index + stride * (offset + i)] = scale * (0 - output[index + stride * (offset + i)]);
			}
			delta[index + stride * class_n] = scale * (1 - output[index + stride * class_n]);

			class_n = hier->parent[class_n];
		}
		*avg_cat += pred;
	}
	else {
		if (delta[index] && tag) {
			delta[index + stride * class_n] = scale * (1 - output[index + stride * class_n]);
			return;
		}
		for (n = 0; n < classes; ++n) {
			delta[index + stride * n] = scale * (((n == class_n) ? 1 : 0) - output[index + stride * n]);
			if (n == class_n) *avg_cat += output[index + stride * n];
		}
	}
}

float logit(float x)
{
	return (float)log(x / (1. - x));
}

float tisnan(float x)
{
	return (x != x);
}

int entry_index(layer l, int batch, int location, int entry)
{
	int n = location / (l.w*l.h);
	int loc = location % (l.w*l.h);
	return batch * l.outputs + n * l.w*l.h*(l.coords + l.classes + 1) + entry * l.w*l.h + loc;
}

void forward_region_layer(const layer l, network net)
{
	int i, j, b, t, n;
	memcpy(l.output, net.input, l.outputs*l.batch * sizeof(float));

#ifndef GPU
	for (b = 0; b < l.batch; ++b) {
		for (n = 0; n < l.n; ++n) {
			int index = entry_index(l, b, n*l.w*l.h, 0);
			activate_array(l.output + index, 2 * l.w*l.h, LOGISTIC);
			index = entry_index(l, b, n*l.w*l.h, l.coords);
			if (!l.background) activate_array(l.output + index, l.w*l.h, LOGISTIC);
			index = entry_index(l, b, n*l.w*l.h, l.coords + 1);
			if (!l.softmax && !l.softmax_tree) activate_array(l.output + index, l.classes*l.w*l.h, LOGISTIC);
		}
	}
	if (l.softmax_tree) {
		int i;
		int count = l.coords + 1;
		for (i = 0; i < l.softmax_tree->groups; ++i) {
			int group_size = l.softmax_tree->group_size[i];
			softmax_cpu(net.input + count, group_size, l.batch, l.inputs, l.n*l.w*l.h, 1, l.n*l.w*l.h, l.temperature, l.output + count);
			count += group_size;
		}
	}
	else if (l.softmax) {
		int index = entry_index(l, 0, 0, l.coords + !l.background);
		softmax_cpu(net.input + index, l.classes + l.background, l.batch*l.n, l.inputs / l.n, l.w*l.h, 1, l.w*l.h, 1, l.output + index);
	}
#endif

	memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
	if (!net.train) return;
	float avg_iou = 0;
	float recall = 0;
	float avg_cat = 0;
	float avg_obj = 0;
	float avg_anyobj = 0;
	int count = 0;
	int class_count = 0;
	*(l.cost) = 0;
	for (b = 0; b < l.batch; ++b) {
		if (l.softmax_tree) {
			int onlyclass = 0;
			for (t = 0; t < 30; ++t) {
				box truth = float_to_box(net.truth + t * (l.coords + 1) + b * l.truths, 1);
				if (!truth.x) break;
				int class_n = (int)(net.truth[t*(l.coords + 1) + b * l.truths + l.coords]);
				float maxp = 0;
				int maxi = 0;
				if (truth.x > 100000 && truth.y > 100000) {
					for (n = 0; n < l.n*l.w*l.h; ++n) {
						int class_index = entry_index(l, b, n, l.coords + 1);
						int obj_index = entry_index(l, b, n, l.coords);
						float scale = l.output[obj_index];
						l.delta[obj_index] = l.noobject_scale * (0 - l.output[obj_index]);
						float p = scale * get_hierarchy_probability(l.output + class_index, l.softmax_tree, class_n, l.w*l.h);
						if (p > maxp) {
							maxp = p;
							maxi = n;
						}
					}
					int class_index = entry_index(l, b, maxi, l.coords + 1);
					int obj_index = entry_index(l, b, maxi, l.coords);
					delta_region_class(l.output, l.delta, class_index, class_n, l.classes, l.softmax_tree, l.class_scale, l.w*l.h, &avg_cat, !l.softmax);
					if (l.output[obj_index] < .3) l.delta[obj_index] = (float)(l.object_scale * (.3 - l.output[obj_index]));
					else  l.delta[obj_index] = 0;
					l.delta[obj_index] = 0;
					++class_count;
					onlyclass = 1;
					break;
				}
			}
			if (onlyclass) continue;
		}
		for (j = 0; j < l.h; ++j) {
			for (i = 0; i < l.w; ++i) {
				for (n = 0; n < l.n; ++n) {
					int box_index = entry_index(l, b, n*l.w*l.h + j * l.w + i, 0);
					box pred = get_region_box(l.output, l.biases, n, box_index, i, j, l.w, l.h, l.w*l.h);
					float best_iou = 0;
					for (t = 0; t < 30; ++t) {
						box truth = float_to_box(net.truth + t * (l.coords + 1) + b * l.truths, 1);
						if (!truth.x) break;
						float iou = box_iou(pred, truth);
						if (iou > best_iou) {
							best_iou = iou;
						}
					}
					int obj_index = entry_index(l, b, n*l.w*l.h + j * l.w + i, l.coords);
					avg_anyobj += l.output[obj_index];
					l.delta[obj_index] = l.noobject_scale * (0 - l.output[obj_index]);
					if (l.background) l.delta[obj_index] = l.noobject_scale * (1 - l.output[obj_index]);
					if (best_iou > l.thresh) {
						l.delta[obj_index] = 0;
					}

					if (*(net.seen) < 12800) {
						box truth = { 0 };
						truth.x = (float)((i + .5) / l.w);
						truth.y = (float)((j + .5) / l.h);
						truth.w = l.biases[2 * n] / l.w;
						truth.h = l.biases[2 * n + 1] / l.h;
						delta_region_box(truth, l.output, l.biases, n, box_index, i, j, l.w, l.h, l.delta, (float).01, l.w*l.h);
					}
				}
			}
		}
		for (t = 0; t < 30; ++t) {
			box truth = float_to_box(net.truth + t * (l.coords + 1) + b * l.truths, 1);

			if (!truth.x) break;
			float best_iou = 0;
			int best_n = 0;
			i = (int)(truth.x * l.w);
			j = (int)(truth.y * l.h);
			box truth_shift = truth;
			truth_shift.x = 0;
			truth_shift.y = 0;
			for (n = 0; n < l.n; ++n) {
				int box_index = entry_index(l, b, n*l.w*l.h + j * l.w + i, 0);
				box pred = get_region_box(l.output, l.biases, n, box_index, i, j, l.w, l.h, l.w*l.h);
				if (l.bias_match) {
					pred.w = l.biases[2 * n] / l.w;
					pred.h = l.biases[2 * n + 1] / l.h;
				}
				pred.x = 0;
				pred.y = 0;
				float iou = box_iou(pred, truth_shift);
				if (iou > best_iou) {
					best_iou = iou;
					best_n = n;
				}
			}

			int box_index = entry_index(l, b, best_n*l.w*l.h + j * l.w + i, 0);
			float iou = delta_region_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, l.delta, l.coord_scale *  (2 - truth.w*truth.h), l.w*l.h);
			if (l.coords > 4) {
				int mask_index = entry_index(l, b, best_n*l.w*l.h + j * l.w + i, 4);
				delta_region_mask(net.truth + t * (l.coords + 1) + b * l.truths + 5, l.output, l.coords - 4, mask_index, l.delta, l.w*l.h, (int)l.mask_scale);
			}
			if (iou > .5) recall += 1;
			avg_iou += iou;

			int obj_index = entry_index(l, b, best_n*l.w*l.h + j * l.w + i, l.coords);
			avg_obj += l.output[obj_index];
			l.delta[obj_index] = l.object_scale * (1 - l.output[obj_index]);
			if (l.rescore) {
				l.delta[obj_index] = l.object_scale * (iou - l.output[obj_index]);
			}
			if (l.background) {
				l.delta[obj_index] = l.object_scale * (0 - l.output[obj_index]);
			}

			int class_n = (int)(net.truth[t*(l.coords + 1) + b * l.truths + l.coords]);
			if (l.map) class_n = l.map[class_n];
			int class_index = entry_index(l, b, best_n*l.w*l.h + j * l.w + i, l.coords + 1);
			delta_region_class(l.output, l.delta, class_index, class_n, l.classes, l.softmax_tree, l.class_scale, l.w*l.h, &avg_cat, !l.softmax);
			++count;
			++class_count;
		}
	}
	*(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
	printf("Region Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall: %f,  count: %d\n", avg_iou / count, avg_cat / class_count, avg_obj / count, avg_anyobj / (l.w*l.h*l.n*l.batch), recall / count, count);
}

void backward_region_layer(const layer l, network net)
{
	/*
	   int b;
	   int size = l.coords + l.classes + 1;
	   for (b = 0; b < l.batch*l.n; ++b){
	   int index = (b*size + 4)*l.w*l.h;
	   gradient_array(l.output + index, l.w*l.h, LOGISTIC, l.delta + index);
	   }
	   axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
	 */
}

void correct_region_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
	int i;
	int new_w = 0;
	int new_h = 0;
	if (((float)netw / w) < ((float)neth / h)) {
		new_w = netw;
		new_h = (h * netw) / w;
	}
	else {
		new_h = neth;
		new_w = (w * neth) / h;
	}
	for (i = 0; i < n; ++i) {
		box b = dets[i].bbox;
		b.x = (float)((b.x - (netw - new_w) / 2. / netw) / ((float)new_w / netw));
		b.y = (float)((b.y - (neth - new_h) / 2. / neth) / ((float)new_h / neth));
		b.w *= (float)netw / new_w;
		b.h *= (float)neth / new_h;
		if (!relative) {
			b.x *= w;
			b.w *= w;
			b.y *= h;
			b.h *= h;
		}
		dets[i].bbox = b;
	}
}

void get_region_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets)
{
	int i, j, n, z;
	float *predictions = l.output;
	if (l.batch == 2) {
		float *flip = l.output + l.outputs;
		for (j = 0; j < l.h; ++j) {
			for (i = 0; i < l.w / 2; ++i) {
				for (n = 0; n < l.n; ++n) {
					for (z = 0; z < l.classes + l.coords + 1; ++z) {
						int i1 = z * l.w*l.h*l.n + n * l.w*l.h + j * l.w + i;
						int i2 = z * l.w*l.h*l.n + n * l.w*l.h + j * l.w + (l.w - i - 1);
						float swap = flip[i1];
						flip[i1] = flip[i2];
						flip[i2] = swap;
						if (z == 0) {
							flip[i1] = -flip[i1];
							flip[i2] = -flip[i2];
						}
					}
				}
			}
		}
		for (i = 0; i < l.outputs; ++i) {
			l.output[i] = (float)((l.output[i] + flip[i]) / 2.);
		}
	}
	for (i = 0; i < l.w*l.h; ++i) {
		int row = i / l.w;
		int col = i % l.w;
		for (n = 0; n < l.n; ++n) {
			int index = n * l.w*l.h + i;
			for (j = 0; j < l.classes; ++j) {
				dets[index].prob[j] = 0;
			}
			int obj_index = entry_index(l, 0, n*l.w*l.h + i, l.coords);
			int box_index = entry_index(l, 0, n*l.w*l.h + i, 0);
			int mask_index = entry_index(l, 0, n*l.w*l.h + i, 4);
			float scale = l.background ? 1 : predictions[obj_index];
			dets[index].bbox = get_region_box(predictions, l.biases, n, box_index, col, row, l.w, l.h, l.w*l.h);
			dets[index].objectness = scale > thresh ? scale : 0;
			if (dets[index].mask) {
				for (j = 0; j < l.coords - 4; ++j) {
					dets[index].mask[j] = l.output[mask_index + j * l.w*l.h];
				}
			}

			int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + !l.background);
			if (l.softmax_tree) {

				hierarchy_predictions(predictions + class_index, l.classes, l.softmax_tree, 0, l.w*l.h);
				if (map) {
					for (j = 0; j < 200; ++j) {
						int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + 1 + map[j]);
						float prob = scale * predictions[class_index];
						dets[index].prob[j] = (prob > thresh) ? prob : 0;
					}
				}
				else {
					int j = hierarchy_top_prediction(predictions + class_index, l.softmax_tree, tree_thresh, l.w*l.h);
					dets[index].prob[j] = (scale > thresh) ? scale : 0;
				}
			}
			else {
				if (dets[index].objectness) {
					for (j = 0; j < l.classes; ++j) {
						int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + 1 + j);
						float prob = scale * predictions[class_index];
						dets[index].prob[j] = (prob > thresh) ? prob : 0;
					}
				}
			}
		}
	}
	correct_region_boxes(dets, l.w*l.h*l.n, w, h, netw, neth, relative);
}

#ifdef GPU

void forward_region_layer_gpu(const layer l, network net)
{
	copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
	int b, n;
	for (b = 0; b < l.batch; ++b) {
		for (n = 0; n < l.n; ++n) {
			int index = entry_index(l, b, n*l.w*l.h, 0);
			activate_array_gpu(l.output_gpu + index, 2 * l.w*l.h, LOGISTIC);
			if (l.coords > 4) {
				index = entry_index(l, b, n*l.w*l.h, 4);
				activate_array_gpu(l.output_gpu + index, (l.coords - 4)*l.w*l.h, LOGISTIC);
			}
			index = entry_index(l, b, n*l.w*l.h, l.coords);
			if (!l.background) activate_array_gpu(l.output_gpu + index, l.w*l.h, LOGISTIC);
			index = entry_index(l, b, n*l.w*l.h, l.coords + 1);
			if (!l.softmax && !l.softmax_tree) activate_array_gpu(l.output_gpu + index, l.classes*l.w*l.h, LOGISTIC);
		}
	}
	if (l.softmax_tree) {
		int index = entry_index(l, 0, 0, l.coords + 1);
		softmax_tree(net.input_gpu + index, l.w*l.h, l.batch*l.n, l.inputs / l.n, 1, l.output_gpu + index, *l.softmax_tree);
	}
	else if (l.softmax) {
		int index = entry_index(l, 0, 0, l.coords + !l.background);
		softmax_gpu(net.input_gpu + index, l.classes + l.background, l.batch*l.n, l.inputs / l.n, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu + index);
	}
	if (!net.train || l.onlyforward) {
		cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
		return;
	}

	cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
	forward_region_layer(l, net);
	//cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
	if (!net.train) return;
	cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
}

void backward_region_layer_gpu(const layer l, network net)
{
	int b, n;
	for (b = 0; b < l.batch; ++b) {
		for (n = 0; n < l.n; ++n) {
			int index = entry_index(l, b, n*l.w*l.h, 0);
			gradient_array_gpu(l.output_gpu + index, 2 * l.w*l.h, LOGISTIC, l.delta_gpu + index);
			if (l.coords > 4) {
				index = entry_index(l, b, n*l.w*l.h, 4);
				gradient_array_gpu(l.output_gpu + index, (l.coords - 4)*l.w*l.h, LOGISTIC, l.delta_gpu + index);
			}
			index = entry_index(l, b, n*l.w*l.h, l.coords);
			if (!l.background) gradient_array_gpu(l.output_gpu + index, l.w*l.h, LOGISTIC, l.delta_gpu + index);
		}
	}
	axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

void zero_objectness(layer l)
{
	int i, n;
	for (i = 0; i < l.w*l.h; ++i) {
		for (n = 0; n < l.n; ++n) {
			int obj_index = entry_index(l, 0, n*l.w*l.h + i, l.coords);
			l.output[obj_index] = 0;
		}
	}
}








//reorg_layer.c
layer make_reorg_layer(int batch, int w, int h, int c, int stride, int reverse, int flatten, int extra)
{
	layer l = { };
	l.type = REORG;
	l.batch = batch;
	l.stride = stride;
	l.extra = extra;
	l.h = h;
	l.w = w;
	l.c = c;
	l.flatten = flatten;
	if (reverse) {
		l.out_w = w * stride;
		l.out_h = h * stride;
		l.out_c = c / (stride*stride);
	}
	else {
		l.out_w = w / stride;
		l.out_h = h / stride;
		l.out_c = c * (stride*stride);
	}
	l.reverse = reverse;

	l.outputs = l.out_h * l.out_w * l.out_c;
	l.inputs = h * w*c;
	if (l.extra) {
		l.out_w = l.out_h = l.out_c = 0;
		l.outputs = l.inputs + l.extra;
	}

	if (extra) {
		fprintf(stderr, "reorg              %4d   ->  %4d\n", l.inputs, l.outputs);
	}
	else {
		fprintf(stderr, "reorg              /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", stride, w, h, c, l.out_w, l.out_h, l.out_c);
	}
	int output_size = l.outputs * batch;
	l.output = (float*)calloc(output_size, sizeof(float));
	l.delta = (float*)calloc(output_size, sizeof(float));

	l.forward = forward_reorg_layer;
	l.backward = backward_reorg_layer;
#ifdef GPU
	l.forward_gpu = forward_reorg_layer_gpu;
	l.backward_gpu = backward_reorg_layer_gpu;

	l.output_gpu = cuda_make_array(l.output, output_size);
	l.delta_gpu = cuda_make_array(l.delta, output_size);
#endif
	return l;
}

void resize_reorg_layer(layer *l, int w, int h)
{
	int stride = l->stride;
	int c = l->c;

	l->h = h;
	l->w = w;

	if (l->reverse) {
		l->out_w = w * stride;
		l->out_h = h * stride;
		l->out_c = c / (stride*stride);
	}
	else {
		l->out_w = w / stride;
		l->out_h = h / stride;
		l->out_c = c * (stride*stride);
	}

	l->outputs = l->out_h * l->out_w * l->out_c;
	l->inputs = l->outputs;
	int output_size = l->outputs * l->batch;

	l->output = (float*)realloc(l->output, output_size * sizeof(float));
	l->delta = (float*)realloc(l->delta, output_size * sizeof(float));

#ifdef GPU
	cuda_free(l->output_gpu);
	cuda_free(l->delta_gpu);
	l->output_gpu = cuda_make_array(l->output, output_size);
	l->delta_gpu = cuda_make_array(l->delta, output_size);
#endif
}

void forward_reorg_layer(const layer l, network net)
{
	int i;
	if (l.flatten) {
		memcpy(l.output, net.input, l.outputs*l.batch * sizeof(float));
		if (l.reverse) {
			flatten(l.output, l.w*l.h, l.c, l.batch, 0);
		}
		else {
			flatten(l.output, l.w*l.h, l.c, l.batch, 1);
		}
	}
	else if (l.extra) {
		for (i = 0; i < l.batch; ++i) {
			copy_cpu(l.inputs, net.input + i * l.inputs, 1, l.output + i * l.outputs, 1);
		}
	}
	else if (l.reverse) {
		reorg_cpu(net.input, l.w, l.h, l.c, l.batch, l.stride, 1, l.output);
	}
	else {
		reorg_cpu(net.input, l.w, l.h, l.c, l.batch, l.stride, 0, l.output);
	}
}

void backward_reorg_layer(const layer l, network net)
{
	int i;
	if (l.flatten) {
		memcpy(net.delta, l.delta, l.outputs*l.batch * sizeof(float));
		if (l.reverse) {
			flatten(net.delta, l.w*l.h, l.c, l.batch, 1);
		}
		else {
			flatten(net.delta, l.w*l.h, l.c, l.batch, 0);
		}
	}
	else if (l.reverse) {
		reorg_cpu(l.delta, l.w, l.h, l.c, l.batch, l.stride, 0, net.delta);
	}
	else if (l.extra) {
		for (i = 0; i < l.batch; ++i) {
			copy_cpu(l.inputs, l.delta + i * l.outputs, 1, net.delta + i * l.inputs, 1);
		}
	}
	else {
		reorg_cpu(l.delta, l.w, l.h, l.c, l.batch, l.stride, 1, net.delta);
	}
}

#ifdef GPU
void forward_reorg_layer_gpu(layer l, network net)
{
	int i;
	if (l.flatten) {
		if (l.reverse) {
			flatten_gpu(net.input_gpu, l.w*l.h, l.c, l.batch, 0, l.output_gpu);
		}
		else {
			flatten_gpu(net.input_gpu, l.w*l.h, l.c, l.batch, 1, l.output_gpu);
		}
	}
	else if (l.extra) {
		for (i = 0; i < l.batch; ++i) {
			copy_gpu(l.inputs, net.input_gpu + i * l.inputs, 1, l.output_gpu + i * l.outputs, 1);
		}
	}
	else if (l.reverse) {
		reorg_gpu(net.input_gpu, l.w, l.h, l.c, l.batch, l.stride, 1, l.output_gpu);
	}
	else {
		reorg_gpu(net.input_gpu, l.w, l.h, l.c, l.batch, l.stride, 0, l.output_gpu);
	}
}

void backward_reorg_layer_gpu(layer l, network net)
{
	if (l.flatten) {
		if (l.reverse) {
			flatten_gpu(l.delta_gpu, l.w*l.h, l.c, l.batch, 1, net.delta_gpu);
		}
		else {
			flatten_gpu(l.delta_gpu, l.w*l.h, l.c, l.batch, 0, net.delta_gpu);
		}
	}
	else if (l.extra) {
		int i;
		for (i = 0; i < l.batch; ++i) {
			copy_gpu(l.inputs, l.delta_gpu + i * l.outputs, 1, net.delta_gpu + i * l.inputs, 1);
		}
	}
	else if (l.reverse) {
		reorg_gpu(l.delta_gpu, l.w, l.h, l.c, l.batch, l.stride, 0, net.delta_gpu);
	}
	else {
		reorg_gpu(l.delta_gpu, l.w, l.h, l.c, l.batch, l.stride, 1, net.delta_gpu);
	}
}
#endif








//rnn_layer.c
/*Declared in crnn_layer.c
static void increment_layer(layer *l, int steps)
{
	int num = l->outputs*l->batch*steps;
	l->output += num;
	l->delta += num;
	l->x += num;
	l->x_norm += num;

#ifdef GPU
	l->output_gpu += num;
	l->delta_gpu += num;
	l->x_gpu += num;
	l->x_norm_gpu += num;
#endif
}*/

layer make_rnn_layer(int batch, int inputs, int outputs, int steps, ACTIVATION activation, int batch_normalize, int adam)
{
	fprintf(stderr, "RNN Layer: %d inputs, %d outputs\n", inputs, outputs);
	batch = batch / steps;
	layer l = {  };
	l.batch = batch;
	l.type = RNN;
	l.steps = steps;
	l.inputs = inputs;

	l.state = (float*)calloc(batch*outputs, sizeof(float));
	l.prev_state = (float*)calloc(batch*outputs, sizeof(float));

	l.input_layer = (layer*)malloc(sizeof(layer));
	fprintf(stderr, "\t\t");
	*(l.input_layer) = make_connected_layer(batch*steps, inputs, outputs, activation, batch_normalize, adam);
	l.input_layer->batch = batch;

	l.self_layer = (layer*)malloc(sizeof(layer));
	fprintf(stderr, "\t\t");
	*(l.self_layer) = make_connected_layer(batch*steps, outputs, outputs, activation, batch_normalize, adam);
	l.self_layer->batch = batch;

	l.output_layer = (layer*)malloc(sizeof(layer));
	fprintf(stderr, "\t\t");
	*(l.output_layer) = make_connected_layer(batch*steps, outputs, outputs, activation, batch_normalize, adam);
	l.output_layer->batch = batch;

	l.outputs = outputs;
	l.output = l.output_layer->output;
	l.delta = l.output_layer->delta;

	l.forward = forward_rnn_layer;
	l.backward = backward_rnn_layer;
	l.update = update_rnn_layer;
#ifdef GPU
	l.forward_gpu = forward_rnn_layer_gpu;
	l.backward_gpu = backward_rnn_layer_gpu;
	l.update_gpu = update_rnn_layer_gpu;
	l.state_gpu = cuda_make_array(0, batch*outputs);
	l.prev_state_gpu = cuda_make_array(0, batch*outputs);
	l.output_gpu = l.output_layer->output_gpu;
	l.delta_gpu = l.output_layer->delta_gpu;
#ifdef CUDNN
	cudnnSetTensor4dDescriptor(l.input_layer->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.input_layer->out_c, l.input_layer->out_h, l.input_layer->out_w);
	cudnnSetTensor4dDescriptor(l.self_layer->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.self_layer->out_c, l.self_layer->out_h, l.self_layer->out_w);
	cudnnSetTensor4dDescriptor(l.output_layer->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.output_layer->out_c, l.output_layer->out_h, l.output_layer->out_w);
#endif
#endif

	return l;
}

void update_rnn_layer(layer l, update_args a)
{
	update_connected_layer(*(l.input_layer), a);
	update_connected_layer(*(l.self_layer), a);
	update_connected_layer(*(l.output_layer), a);
}

void forward_rnn_layer(layer l, network net)
{
	network s = net;
	s.train = net.train;
	int i;
	layer input_layer = *(l.input_layer);
	layer self_layer = *(l.self_layer);
	layer output_layer = *(l.output_layer);

	fill_cpu(l.outputs * l.batch * l.steps, 0, output_layer.delta, 1);
	fill_cpu(l.outputs * l.batch * l.steps, 0, self_layer.delta, 1);
	fill_cpu(l.outputs * l.batch * l.steps, 0, input_layer.delta, 1);
	if (net.train) fill_cpu(l.outputs * l.batch, 0, l.state, 1);

	for (i = 0; i < l.steps; ++i) {
		s.input = net.input;
		forward_connected_layer(input_layer, s);

		s.input = l.state;
		forward_connected_layer(self_layer, s);

		float *old_state = l.state;
		if (net.train) l.state += l.outputs*l.batch;
		if (l.shortcut) {
			copy_cpu(l.outputs * l.batch, old_state, 1, l.state, 1);
		}
		else {
			fill_cpu(l.outputs * l.batch, 0, l.state, 1);
		}
		axpy_cpu(l.outputs * l.batch, 1, input_layer.output, 1, l.state, 1);
		axpy_cpu(l.outputs * l.batch, 1, self_layer.output, 1, l.state, 1);

		s.input = l.state;
		forward_connected_layer(output_layer, s);

		net.input += l.inputs*l.batch;
		increment_layer(&input_layer, 1);
		increment_layer(&self_layer, 1);
		increment_layer(&output_layer, 1);
	}
}

void backward_rnn_layer(layer l, network net)
{
	network s = net;
	s.train = net.train;
	int i;
	layer input_layer = *(l.input_layer);
	layer self_layer = *(l.self_layer);
	layer output_layer = *(l.output_layer);

	increment_layer(&input_layer, l.steps - 1);
	increment_layer(&self_layer, l.steps - 1);
	increment_layer(&output_layer, l.steps - 1);

	l.state += l.outputs*l.batch*l.steps;
	for (i = l.steps - 1; i >= 0; --i) {
		copy_cpu(l.outputs * l.batch, input_layer.output, 1, l.state, 1);
		axpy_cpu(l.outputs * l.batch, 1, self_layer.output, 1, l.state, 1);

		s.input = l.state;
		s.delta = self_layer.delta;
		backward_connected_layer(output_layer, s);

		l.state -= l.outputs*l.batch;
		/*
		   if(i > 0){
		   copy_cpu(l.outputs * l.batch, input_layer.output - l.outputs*l.batch, 1, l.state, 1);
		   axpy_cpu(l.outputs * l.batch, 1, self_layer.output - l.outputs*l.batch, 1, l.state, 1);
		   }else{
		   fill_cpu(l.outputs * l.batch, 0, l.state, 1);
		   }
		 */

		s.input = l.state;
		s.delta = self_layer.delta - l.outputs*l.batch;
		if (i == 0) s.delta = 0;
		backward_connected_layer(self_layer, s);

		copy_cpu(l.outputs*l.batch, self_layer.delta, 1, input_layer.delta, 1);
		if (i > 0 && l.shortcut) axpy_cpu(l.outputs*l.batch, 1, self_layer.delta, 1, self_layer.delta - l.outputs*l.batch, 1);
		s.input = net.input + i * l.inputs*l.batch;
		if (net.delta) s.delta = net.delta + i * l.inputs*l.batch;
		else s.delta = 0;
		backward_connected_layer(input_layer, s);

		increment_layer(&input_layer, -1);
		increment_layer(&self_layer, -1);
		increment_layer(&output_layer, -1);
	}
}

#ifdef GPU

void pull_rnn_layer(layer l)
{
	pull_connected_layer(*(l.input_layer));
	pull_connected_layer(*(l.self_layer));
	pull_connected_layer(*(l.output_layer));
}

void push_rnn_layer(layer l)
{
	push_connected_layer(*(l.input_layer));
	push_connected_layer(*(l.self_layer));
	push_connected_layer(*(l.output_layer));
}

void update_rnn_layer_gpu(layer l, update_args a)
{
	update_connected_layer_gpu(*(l.input_layer), a);
	update_connected_layer_gpu(*(l.self_layer), a);
	update_connected_layer_gpu(*(l.output_layer), a);
}

void forward_rnn_layer_gpu(layer l, network net)
{
	network s = { 0 };
	s.train = net.train;
	int i;
	layer input_layer = *(l.input_layer);
	layer self_layer = *(l.self_layer);
	layer output_layer = *(l.output_layer);

	fill_gpu(l.outputs * l.batch * l.steps, 0, output_layer.delta_gpu, 1);
	fill_gpu(l.outputs * l.batch * l.steps, 0, self_layer.delta_gpu, 1);
	fill_gpu(l.outputs * l.batch * l.steps, 0, input_layer.delta_gpu, 1);

	if (net.train) {
		fill_gpu(l.outputs * l.batch * l.steps, 0, l.delta_gpu, 1);
		copy_gpu(l.outputs*l.batch, l.state_gpu, 1, l.prev_state_gpu, 1);
	}

	for (i = 0; i < l.steps; ++i) {
		s.input_gpu = net.input_gpu;
		forward_connected_layer_gpu(input_layer, s);

		s.input_gpu = l.state_gpu;
		forward_connected_layer_gpu(self_layer, s);

		fill_gpu(l.outputs * l.batch, 0, l.state_gpu, 1);
		axpy_gpu(l.outputs * l.batch, 1, input_layer.output_gpu, 1, l.state_gpu, 1);
		axpy_gpu(l.outputs * l.batch, 1, self_layer.output_gpu, 1, l.state_gpu, 1);

		s.input_gpu = l.state_gpu;
		forward_connected_layer_gpu(output_layer, s);

		net.input_gpu += l.inputs*l.batch;
		increment_layer(&input_layer, 1);
		increment_layer(&self_layer, 1);
		increment_layer(&output_layer, 1);
	}
}

void backward_rnn_layer_gpu(layer l, network net)
{
	network s = { 0 };
	s.train = net.train;
	int i;
	layer input_layer = *(l.input_layer);
	layer self_layer = *(l.self_layer);
	layer output_layer = *(l.output_layer);
	increment_layer(&input_layer, l.steps - 1);
	increment_layer(&self_layer, l.steps - 1);
	increment_layer(&output_layer, l.steps - 1);
	float *last_input = input_layer.output_gpu;
	float *last_self = self_layer.output_gpu;
	for (i = l.steps - 1; i >= 0; --i) {
		fill_gpu(l.outputs * l.batch, 0, l.state_gpu, 1);
		axpy_gpu(l.outputs * l.batch, 1, input_layer.output_gpu, 1, l.state_gpu, 1);
		axpy_gpu(l.outputs * l.batch, 1, self_layer.output_gpu, 1, l.state_gpu, 1);

		s.input_gpu = l.state_gpu;
		s.delta_gpu = self_layer.delta_gpu;
		backward_connected_layer_gpu(output_layer, s);

		if (i != 0) {
			fill_gpu(l.outputs * l.batch, 0, l.state_gpu, 1);
			axpy_gpu(l.outputs * l.batch, 1, input_layer.output_gpu - l.outputs*l.batch, 1, l.state_gpu, 1);
			axpy_gpu(l.outputs * l.batch, 1, self_layer.output_gpu - l.outputs*l.batch, 1, l.state_gpu, 1);
		}
		else {
			copy_gpu(l.outputs*l.batch, l.prev_state_gpu, 1, l.state_gpu, 1);
		}

		copy_gpu(l.outputs*l.batch, self_layer.delta_gpu, 1, input_layer.delta_gpu, 1);

		s.input_gpu = l.state_gpu;
		s.delta_gpu = (i > 0) ? self_layer.delta_gpu - l.outputs*l.batch : 0;
		if (i == 0) s.delta_gpu = 0;
		backward_connected_layer_gpu(self_layer, s);

		s.input_gpu = net.input_gpu + i * l.inputs*l.batch;
		if (net.delta_gpu) s.delta_gpu = net.delta_gpu + i * l.inputs*l.batch;
		else s.delta_gpu = 0;
		backward_connected_layer_gpu(input_layer, s);

		increment_layer(&input_layer, -1);
		increment_layer(&self_layer, -1);
		increment_layer(&output_layer, -1);
	}
	fill_gpu(l.outputs * l.batch, 0, l.state_gpu, 1);
	axpy_gpu(l.outputs * l.batch, 1, last_input, 1, l.state_gpu, 1);
	axpy_gpu(l.outputs * l.batch, 1, last_self, 1, l.state_gpu, 1);
}
#endif








//route_layer.c
route_layer make_route_layer(int batch, int n, int *input_layers, int *input_sizes)
{
	fprintf(stderr, "route ");
	route_layer l = { };
	l.type = ROUTE;
	l.batch = batch;
	l.n = n;
	l.input_layers = input_layers;
	l.input_sizes = input_sizes;
	int i;
	int outputs = 0;
	for (i = 0; i < n; ++i) {
		fprintf(stderr, " %d", input_layers[i]);
		outputs += input_sizes[i];
	}
	fprintf(stderr, "\n");
	l.outputs = outputs;
	l.inputs = outputs;
	l.delta = (float*)calloc(outputs*batch, sizeof(float));
	l.output = (float*)calloc(outputs*batch, sizeof(float));;

	l.forward = forward_route_layer;
	l.backward = backward_route_layer;
#ifdef GPU
	l.forward_gpu = forward_route_layer_gpu;
	l.backward_gpu = backward_route_layer_gpu;

	l.delta_gpu = cuda_make_array(l.delta, outputs*batch);
	l.output_gpu = cuda_make_array(l.output, outputs*batch);
#endif
	return l;
}

void resize_route_layer(route_layer *l, network *net)
{
	int i;
	layer first = net->layers[l->input_layers[0]];
	l->out_w = first.out_w;
	l->out_h = first.out_h;
	l->out_c = first.out_c;
	l->outputs = first.outputs;
	l->input_sizes[0] = first.outputs;
	for (i = 1; i < l->n; ++i) {
		int index = l->input_layers[i];
		layer next = net->layers[index];
		l->outputs += next.outputs;
		l->input_sizes[i] = next.outputs;
		if (next.out_w == first.out_w && next.out_h == first.out_h) {
			l->out_c += next.out_c;
		}
		else {
			printf("%d %d, %d %d\n", next.out_w, next.out_h, first.out_w, first.out_h);
			l->out_h = l->out_w = l->out_c = 0;
		}
	}
	l->inputs = l->outputs;
	l->delta = (float*)realloc(l->delta, l->outputs*l->batch * sizeof(float));
	l->output = (float*)realloc(l->output, l->outputs*l->batch * sizeof(float));

#ifdef GPU
	cuda_free(l->output_gpu);
	cuda_free(l->delta_gpu);
	l->output_gpu = cuda_make_array(l->output, l->outputs*l->batch);
	l->delta_gpu = cuda_make_array(l->delta, l->outputs*l->batch);
#endif

}

void forward_route_layer(const route_layer l, network net)
{
	int i, j;
	int offset = 0;
	for (i = 0; i < l.n; ++i) {
		int index = l.input_layers[i];
		float *input = net.layers[index].output;
		int input_size = l.input_sizes[i];
		for (j = 0; j < l.batch; ++j) {
			copy_cpu(input_size, input + j * input_size, 1, l.output + offset + j * l.outputs, 1);
		}
		offset += input_size;
	}
}

void backward_route_layer(const route_layer l, network net)
{
	int i, j;
	int offset = 0;
	for (i = 0; i < l.n; ++i) {
		int index = l.input_layers[i];
		float *delta = net.layers[index].delta;
		int input_size = l.input_sizes[i];
		for (j = 0; j < l.batch; ++j) {
			axpy_cpu(input_size, 1, l.delta + offset + j * l.outputs, 1, delta + j * input_size, 1);
		}
		offset += input_size;
	}
}

#ifdef GPU
void forward_route_layer_gpu(const route_layer l, network net)
{
	int i, j;
	int offset = 0;
	for (i = 0; i < l.n; ++i) {
		int index = l.input_layers[i];
		float *input = net.layers[index].output_gpu;
		int input_size = l.input_sizes[i];
		for (j = 0; j < l.batch; ++j) {
			copy_gpu(input_size, input + j * input_size, 1, l.output_gpu + offset + j * l.outputs, 1);
		}
		offset += input_size;
	}
}

void backward_route_layer_gpu(const route_layer l, network net)
{
	int i, j;
	int offset = 0;
	for (i = 0; i < l.n; ++i) {
		int index = l.input_layers[i];
		float *delta = net.layers[index].delta_gpu;
		int input_size = l.input_sizes[i];
		for (j = 0; j < l.batch; ++j) {
			axpy_gpu(input_size, 1, l.delta_gpu + offset + j * l.outputs, 1, delta + j * input_size, 1);
		}
		offset += input_size;
	}
}
#endif








//shortcut_layer.c
layer make_shortcut_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2)
{
	fprintf(stderr, "res  %3d                %4d x%4d x%4d   ->  %4d x%4d x%4d\n", index, w2, h2, c2, w, h, c);
	layer l = { };
	l.type = SHORTCUT;
	l.batch = batch;
	l.w = w2;
	l.h = h2;
	l.c = c2;
	l.out_w = w;
	l.out_h = h;
	l.out_c = c;
	l.outputs = w * h*c;
	l.inputs = l.outputs;

	l.index = index;

	l.delta = (float*)calloc(l.outputs*batch, sizeof(float));
	l.output = (float*)calloc(l.outputs*batch, sizeof(float));;

	l.forward = forward_shortcut_layer;
	l.backward = backward_shortcut_layer;
#ifdef GPU
	l.forward_gpu = forward_shortcut_layer_gpu;
	l.backward_gpu = backward_shortcut_layer_gpu;

	l.delta_gpu = cuda_make_array(l.delta, l.outputs*batch);
	l.output_gpu = cuda_make_array(l.output, l.outputs*batch);
#endif
	return l;
}

void resize_shortcut_layer(layer *l, int w, int h)
{
	assert(l->w == l->out_w);
	assert(l->h == l->out_h);
	l->w = l->out_w = w;
	l->h = l->out_h = h;
	l->outputs = w * h*l->out_c;
	l->inputs = l->outputs;
	l->delta = (float*)realloc(l->delta, l->outputs*l->batch * sizeof(float));
	l->output = (float*)realloc(l->output, l->outputs*l->batch * sizeof(float));

#ifdef GPU
	cuda_free(l->output_gpu);
	cuda_free(l->delta_gpu);
	l->output_gpu = cuda_make_array(l->output, l->outputs*l->batch);
	l->delta_gpu = cuda_make_array(l->delta, l->outputs*l->batch);
#endif

}


void forward_shortcut_layer(const layer l, network net)
{
	copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
	shortcut_cpu(l.batch, l.w, l.h, l.c, net.layers[l.index].output, l.out_w, l.out_h, l.out_c, l.alpha, l.beta, l.output);
	activate_array(l.output, l.outputs*l.batch, l.activation);
}

void backward_shortcut_layer(const layer l, network net)
{
	gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
	axpy_cpu(l.outputs*l.batch, l.alpha, l.delta, 1, net.delta, 1);
	shortcut_cpu(l.batch, l.out_w, l.out_h, l.out_c, l.delta, l.w, l.h, l.c, 1, l.beta, net.layers[l.index].delta);
}

#ifdef GPU
void forward_shortcut_layer_gpu(const layer l, network net)
{
	copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
	shortcut_gpu(l.batch, l.w, l.h, l.c, net.layers[l.index].output_gpu, l.out_w, l.out_h, l.out_c, l.alpha, l.beta, l.output_gpu);
	activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void backward_shortcut_layer_gpu(const layer l, network net)
{
	gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
	axpy_gpu(l.outputs*l.batch, l.alpha, l.delta_gpu, 1, net.delta_gpu, 1);
	shortcut_gpu(l.batch, l.out_w, l.out_h, l.out_c, l.delta_gpu, l.w, l.h, l.c, 1, l.beta, net.layers[l.index].delta_gpu);
}
#endif







//softmax_layer.c
softmax_layer make_softmax_layer(int batch, int inputs, int groups)
{
	assert(inputs%groups == 0);
	fprintf(stderr, "softmax                                        %4d\n", inputs);
	softmax_layer l = { };
	l.type = SOFTMAX;
	l.batch = batch;
	l.groups = groups;
	l.inputs = inputs;
	l.outputs = inputs;
	l.loss = (float*)calloc(inputs*batch, sizeof(float));
	l.output = (float*)calloc(inputs*batch, sizeof(float));
	l.delta = (float*)calloc(inputs*batch, sizeof(float));
	l.cost = (float*)calloc(1, sizeof(float));

	l.forward = forward_softmax_layer;
	l.backward = backward_softmax_layer;
#ifdef GPU
	l.forward_gpu = forward_softmax_layer_gpu;
	l.backward_gpu = backward_softmax_layer_gpu;

	l.output_gpu = cuda_make_array(l.output, inputs*batch);
	l.loss_gpu = cuda_make_array(l.loss, inputs*batch);
	l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
#endif
	return l;
}

void forward_softmax_layer(const softmax_layer l, network net)
{
	if (l.softmax_tree) {
		int i;
		int count = 0;
		for (i = 0; i < l.softmax_tree->groups; ++i) {
			int group_size = l.softmax_tree->group_size[i];
			softmax_cpu(net.input + count, group_size, l.batch, l.inputs, 1, 0, 1, l.temperature, l.output + count);
			count += group_size;
		}
	}
	else {
		softmax_cpu(net.input, l.inputs / l.groups, l.batch, l.inputs, l.groups, l.inputs / l.groups, 1, l.temperature, l.output);
	}

	if (net.truth && !l.noloss) {
		softmax_x_ent_cpu(l.batch*l.inputs, l.output, net.truth, l.delta, l.loss);
		l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
	}
}

void backward_softmax_layer(const softmax_layer l, network net)
{
	axpy_cpu(l.inputs*l.batch, 1, l.delta, 1, net.delta, 1);
}

#ifdef GPU

void pull_softmax_layer_output(const softmax_layer layer)
{
	cuda_pull_array(layer.output_gpu, layer.output, layer.inputs*layer.batch);
}

void forward_softmax_layer_gpu(const softmax_layer l, network net)
{
	if (l.softmax_tree) {
		softmax_tree(net.input_gpu, 1, l.batch, l.inputs, l.temperature, l.output_gpu, *l.softmax_tree);
		/*
		int i;
		int count = 0;
		for (i = 0; i < l.softmax_tree->groups; ++i) {
			int group_size = l.softmax_tree->group_size[i];
			softmax_gpu(net.input_gpu + count, group_size, l.batch, l.inputs, 1, 0, 1, l.temperature, l.output_gpu + count);
			count += group_size;
		}
		*/
	}
	else {
		if (l.spatial) {
			softmax_gpu(net.input_gpu, l.c, l.batch*l.c, l.inputs / l.c, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu);
		}
		else {
			softmax_gpu(net.input_gpu, l.inputs / l.groups, l.batch, l.inputs, l.groups, l.inputs / l.groups, 1, l.temperature, l.output_gpu);
		}
	}
	if (net.truth && !l.noloss) {
		softmax_x_ent_gpu(l.batch*l.inputs, l.output_gpu, net.truth_gpu, l.delta_gpu, l.loss_gpu);
		if (l.softmax_tree) {
			mask_gpu(l.batch*l.inputs, l.delta_gpu, SECRET_NUM, net.truth_gpu, 0);
			mask_gpu(l.batch*l.inputs, l.loss_gpu, SECRET_NUM, net.truth_gpu, 0);
		}
		cuda_pull_array(l.loss_gpu, l.loss, l.batch*l.inputs);
		l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
	}
}

void backward_softmax_layer_gpu(const softmax_layer layer, network net)
{
	axpy_gpu(layer.batch*layer.inputs, 1, layer.delta_gpu, 1, net.delta_gpu, 1);
}

#endif









//upsample_layer.c
layer make_upsample_layer(int batch, int w, int h, int c, int stride)
{
	layer l = { };
	l.type = UPSAMPLE;
	l.batch = batch;
	l.w = w;
	l.h = h;
	l.c = c;
	l.out_w = w * stride;
	l.out_h = h * stride;
	l.out_c = c;
	if (stride < 0) {
		stride = -stride;
		l.reverse = 1;
		l.out_w = w / stride;
		l.out_h = h / stride;
	}
	l.stride = stride;
	l.outputs = l.out_w*l.out_h*l.out_c;
	l.inputs = l.w*l.h*l.c;
	l.delta = (float*)calloc(l.outputs*batch, sizeof(float));
	l.output = (float*)calloc(l.outputs*batch, sizeof(float));;

	l.forward = forward_upsample_layer;
	l.backward = backward_upsample_layer;
#ifdef GPU
	l.forward_gpu = forward_upsample_layer_gpu;
	l.backward_gpu = backward_upsample_layer_gpu;

	l.delta_gpu = cuda_make_array(l.delta, l.outputs*batch);
	l.output_gpu = cuda_make_array(l.output, l.outputs*batch);
#endif
	if (l.reverse) fprintf(stderr, "downsample         %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", stride, w, h, c, l.out_w, l.out_h, l.out_c);
	else fprintf(stderr, "upsample           %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", stride, w, h, c, l.out_w, l.out_h, l.out_c);
	return l;
}

void resize_upsample_layer(layer *l, int w, int h)
{
	l->w = w;
	l->h = h;
	l->out_w = w * l->stride;
	l->out_h = h * l->stride;
	if (l->reverse) {
		l->out_w = w / l->stride;
		l->out_h = h / l->stride;
	}
	l->outputs = l->out_w*l->out_h*l->out_c;
	l->inputs = l->h*l->w*l->c;
	l->delta = (float*)realloc(l->delta, l->outputs*l->batch * sizeof(float));
	l->output = (float*)realloc(l->output, l->outputs*l->batch * sizeof(float));

#ifdef GPU
	cuda_free(l->output_gpu);
	cuda_free(l->delta_gpu);
	l->output_gpu = cuda_make_array(l->output, l->outputs*l->batch);
	l->delta_gpu = cuda_make_array(l->delta, l->outputs*l->batch);
#endif

}

void forward_upsample_layer(const layer l, network net)
{
	fill_cpu(l.outputs*l.batch, 0, l.output, 1);
	if (l.reverse) {
		upsample_cpu(l.output, l.out_w, l.out_h, l.c, l.batch, l.stride, 0, l.scale, net.input);
	}
	else {
		upsample_cpu(net.input, l.w, l.h, l.c, l.batch, l.stride, 1, l.scale, l.output);
	}
}

void backward_upsample_layer(const layer l, network net)
{
	if (l.reverse) {
		upsample_cpu(l.delta, l.out_w, l.out_h, l.c, l.batch, l.stride, 1, l.scale, net.delta);
	}
	else {
		upsample_cpu(net.delta, l.w, l.h, l.c, l.batch, l.stride, 0, l.scale, l.delta);
	}
}

#ifdef GPU
void forward_upsample_layer_gpu(const layer l, network net)
{
	fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
	if (l.reverse) {
		upsample_gpu(l.output_gpu, l.out_w, l.out_h, l.c, l.batch, l.stride, 0, l.scale, net.input_gpu);
	}
	else {
		upsample_gpu(net.input_gpu, l.w, l.h, l.c, l.batch, l.stride, 1, l.scale, l.output_gpu);
	}
}

void backward_upsample_layer_gpu(const layer l, network net)
{
	if (l.reverse) {
		upsample_gpu(l.delta_gpu, l.out_w, l.out_h, l.c, l.batch, l.stride, 1, l.scale, net.delta_gpu);
	}
	else {
		upsample_gpu(net.delta_gpu, l.w, l.h, l.c, l.batch, l.stride, 0, l.scale, l.delta_gpu);
	}
}
#endif










//yolo_layer.c
layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes)
{
	int i;
	layer l = { };
	l.type = YOLO;

	l.n = n;
	l.total = total;
	l.batch = batch;
	l.h = h;
	l.w = w;
	l.c = n * (classes + 4 + 1);
	l.out_w = l.w;
	l.out_h = l.h;
	l.out_c = l.c;
	l.classes = classes;
	l.cost = (float*)calloc(1, sizeof(float));
	l.biases = (float*)calloc(total * 2, sizeof(float));
	if (mask) l.mask = mask;
	else {
		l.mask = (int*)calloc(n, sizeof(int));
		for (i = 0; i < n; ++i) {
			l.mask[i] = i;
		}
	}
	l.bias_updates = (float*)calloc(n * 2, sizeof(float));
	l.outputs = h * w*n*(classes + 4 + 1);
	l.inputs = l.outputs;
	l.truths = 90 * (4 + 1);
	l.delta = (float*)calloc(batch*l.outputs, sizeof(float));
	l.output = (float*)calloc(batch*l.outputs, sizeof(float));
	for (i = 0; i < total * 2; ++i) {
		l.biases[i] = .5;
	}

	l.forward = forward_yolo_layer;
	l.backward = backward_yolo_layer;
#ifdef GPU
	l.forward_gpu = forward_yolo_layer_gpu;
	l.backward_gpu = backward_yolo_layer_gpu;
	l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
	l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

	fprintf(stderr, "yolo\n");
	srand(0);

	return l;
}

void resize_yolo_layer(layer *l, int w, int h)
{
	l->w = w;
	l->h = h;

	l->outputs = h * w*l->n*(l->classes + 4 + 1);
	l->inputs = l->outputs;

	l->output = (float*)realloc(l->output, l->batch*l->outputs * sizeof(float));
	l->delta = (float*)realloc(l->delta, l->batch*l->outputs * sizeof(float));

#ifdef GPU
	cuda_free(l->delta_gpu);
	cuda_free(l->output_gpu);

	l->delta_gpu = cuda_make_array(l->delta, l->batch*l->outputs);
	l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);
#endif
}

box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
	box b;
	b.x = (i + x[index + 0 * stride]) / lw;
	b.y = (j + x[index + 1 * stride]) / lh;
	b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
	b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
	return b;
}

float delta_yolo_box(box truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride)
{
	box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
	float iou = box_iou(pred, truth);

	float tx = (truth.x*lw - i);
	float ty = (truth.y*lh - j);
	float tw = log(truth.w*w / biases[2 * n]);
	float th = log(truth.h*h / biases[2 * n + 1]);

	delta[index + 0 * stride] = scale * (tx - x[index + 0 * stride]);
	delta[index + 1 * stride] = scale * (ty - x[index + 1 * stride]);
	delta[index + 2 * stride] = scale * (tw - x[index + 2 * stride]);
	delta[index + 3 * stride] = scale * (th - x[index + 3 * stride]);
	return iou;
}


void delta_yolo_class(float *output, float *delta, int index, int class_n, int classes, int stride, float *avg_cat)
{
	int n;
	if (delta[index]) {
		delta[index + stride * class_n] = 1 - output[index + stride * class_n];
		if (avg_cat) *avg_cat += output[index + stride * class_n];
		return;
	}
	for (n = 0; n < classes; ++n) {
		delta[index + stride * n] = ((n == class_n) ? 1 : 0) - output[index + stride * n];
		if (n == class_n && avg_cat) *avg_cat += output[index + stride * n];
	}
}

/*Declared in region_layer.c
static int entry_index(layer l, int batch, int location, int entry)
{
	int n = location / (l.w*l.h);
	int loc = location % (l.w*l.h);
	return batch * l.outputs + n * l.w*l.h*(4 + l.classes + 1) + entry * l.w*l.h + loc;
}*/

void forward_yolo_layer(const layer l, network net)
{
	int i, j, b, t, n;
	memcpy(l.output, net.input, l.outputs*l.batch * sizeof(float));

#ifndef GPU
	for (b = 0; b < l.batch; ++b) {
		for (n = 0; n < l.n; ++n) {
			int index = entry_index(l, b, n*l.w*l.h, 0);
			activate_array(l.output + index, 2 * l.w*l.h, LOGISTIC);
			index = entry_index(l, b, n*l.w*l.h, 4);
			activate_array(l.output + index, (1 + l.classes)*l.w*l.h, LOGISTIC);
		}
	}
#endif

	memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
	if (!net.train) return;
	float avg_iou = 0;
	float recall = 0;
	float recall75 = 0;
	float avg_cat = 0;
	float avg_obj = 0;
	float avg_anyobj = 0;
	int count = 0;
	int class_count = 0;
	*(l.cost) = 0;
	for (b = 0; b < l.batch; ++b) {
		for (j = 0; j < l.h; ++j) {
			for (i = 0; i < l.w; ++i) {
				for (n = 0; n < l.n; ++n) {
					int box_index = entry_index(l, b, n*l.w*l.h + j * l.w + i, 0);
					box pred = get_yolo_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.w*l.h);
					float best_iou = 0;
					int best_t = 0;
					for (t = 0; t < l.max_boxes; ++t) {
						box truth = float_to_box(net.truth + t * (4 + 1) + b * l.truths, 1);
						if (!truth.x) break;
						float iou = box_iou(pred, truth);
						if (iou > best_iou) {
							best_iou = iou;
							best_t = t;
						}
					}
					int obj_index = entry_index(l, b, n*l.w*l.h + j * l.w + i, 4);
					avg_anyobj += l.output[obj_index];
					l.delta[obj_index] = 0 - l.output[obj_index];
					if (best_iou > l.ignore_thresh) {
						l.delta[obj_index] = 0;
					}
					if (best_iou > l.truth_thresh) {
						l.delta[obj_index] = 1 - l.output[obj_index];

						int class_n = (int)(net.truth[best_t*(4 + 1) + b * l.truths + 4]);
						if (l.map) class_n = l.map[class_n];
						int class_index = entry_index(l, b, n*l.w*l.h + j * l.w + i, 4 + 1);
						delta_yolo_class(l.output, l.delta, class_index, class_n, l.classes, l.w*l.h, 0);
						box truth = float_to_box(net.truth + best_t * (4 + 1) + b * l.truths, 1);
						delta_yolo_box(truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2 - truth.w*truth.h), l.w*l.h);
					}
				}
			}
		}
		for (t = 0; t < l.max_boxes; ++t) {
			box truth = float_to_box(net.truth + t * (4 + 1) + b * l.truths, 1);

			if (!truth.x) break;
			float best_iou = 0;
			int best_n = 0;
			i = (int)(truth.x * l.w);
			j = (int)(truth.y * l.h);
			box truth_shift = truth;
			truth_shift.x = truth_shift.y = 0;
			for (n = 0; n < l.total; ++n) {
				box pred = { 0 };
				pred.w = l.biases[2 * n] / net.w;
				pred.h = l.biases[2 * n + 1] / net.h;
				float iou = box_iou(pred, truth_shift);
				if (iou > best_iou) {
					best_iou = iou;
					best_n = n;
				}
			}

			int mask_n = int_index(l.mask, best_n, l.n);
			if (mask_n >= 0) {
				int box_index = entry_index(l, b, mask_n*l.w*l.h + j * l.w + i, 0);
				float iou = delta_yolo_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2 - truth.w*truth.h), l.w*l.h);

				int obj_index = entry_index(l, b, mask_n*l.w*l.h + j * l.w + i, 4);
				avg_obj += l.output[obj_index];
				l.delta[obj_index] = 1 - l.output[obj_index];

				int class_n = (int)(net.truth[t*(4 + 1) + b * l.truths + 4]);
				if (l.map) class_n = l.map[class_n];
				int class_index = entry_index(l, b, mask_n*l.w*l.h + j * l.w + i, 4 + 1);
				delta_yolo_class(l.output, l.delta, class_index, class_n, l.classes, l.w*l.h, &avg_cat);

				++count;
				++class_count;
				if (iou > .5) recall += 1;
				if (iou > .75) recall75 += 1;
				avg_iou += iou;
			}
		}
	}
	*(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
	printf("Region %d Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d\n", net.index, avg_iou / count, avg_cat / class_count, avg_obj / count, avg_anyobj / (l.w*l.h*l.n*l.batch), recall / count, recall75 / count, count);
}

void backward_yolo_layer(const layer l, network net)
{
	axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}

void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
	int i;
	int new_w = 0;
	int new_h = 0;
	if (((float)netw / w) < ((float)neth / h)) {
		new_w = netw;
		new_h = (h * netw) / w;
	}
	else {
		new_h = neth;
		new_w = (w * neth) / h;
	}
	for (i = 0; i < n; ++i) {
		box b = dets[i].bbox;
		b.x = (float)((b.x - (netw - new_w) / 2. / netw) / ((float)new_w / netw));
		b.y = (float)((b.y - (neth - new_h) / 2. / neth) / ((float)new_h / neth));
		b.w *= (float)netw / new_w;
		b.h *= (float)neth / new_h;
		if (!relative) {
			b.x *= w;
			b.w *= w;
			b.y *= h;
			b.h *= h;
		}
		dets[i].bbox = b;
	}
}

int yolo_num_detections(layer l, float thresh)
{
	int i, n;
	int count = 0;
	for (i = 0; i < l.w*l.h; ++i) {
		for (n = 0; n < l.n; ++n) {
			int obj_index = entry_index(l, 0, n*l.w*l.h + i, 4);
			if (l.output[obj_index] > thresh) {
				++count;
			}
		}
	}
	return count;
}

void avg_flipped_yolo(layer l)
{
	int i, j, n, z;
	float *flip = l.output + l.outputs;
	for (j = 0; j < l.h; ++j) {
		for (i = 0; i < l.w / 2; ++i) {
			for (n = 0; n < l.n; ++n) {
				for (z = 0; z < l.classes + 4 + 1; ++z) {
					int i1 = z * l.w*l.h*l.n + n * l.w*l.h + j * l.w + i;
					int i2 = z * l.w*l.h*l.n + n * l.w*l.h + j * l.w + (l.w - i - 1);
					float swap = flip[i1];
					flip[i1] = flip[i2];
					flip[i2] = swap;
					if (z == 0) {
						flip[i1] = -flip[i1];
						flip[i2] = -flip[i2];
					}
				}
			}
		}
	}
	for (i = 0; i < l.outputs; ++i) {
		l.output[i] = (float)((l.output[i] + flip[i]) / 2.);
	}
}

int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets)
{
	int i, j, n;
	float *predictions = l.output;
	if (l.batch == 2) avg_flipped_yolo(l);
	int count = 0;
	for (i = 0; i < l.w*l.h; ++i) {
		int row = i / l.w;
		int col = i % l.w;
		for (n = 0; n < l.n; ++n) {
			int obj_index = entry_index(l, 0, n*l.w*l.h + i, 4);
			float objectness = predictions[obj_index];
			if (objectness <= thresh) continue;
			int box_index = entry_index(l, 0, n*l.w*l.h + i, 0);
			dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h);
			dets[count].objectness = objectness;
			dets[count].classes = l.classes;
			for (j = 0; j < l.classes; ++j) {
				int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
				float prob = objectness * predictions[class_index];
				dets[count].prob[j] = (prob > thresh) ? prob : 0;
			}
			++count;
		}
	}
	correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
	return count;
}

#ifdef GPU

void forward_yolo_layer_gpu(const layer l, network net)
{
	copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
	int b, n;
	for (b = 0; b < l.batch; ++b) {
		for (n = 0; n < l.n; ++n) {
			int index = entry_index(l, b, n*l.w*l.h, 0);
			activate_array_gpu(l.output_gpu + index, 2 * l.w*l.h, LOGISTIC);
			index = entry_index(l, b, n*l.w*l.h, 4);
			activate_array_gpu(l.output_gpu + index, (1 + l.classes)*l.w*l.h, LOGISTIC);
		}
	}
	if (!net.train || l.onlyforward) {
		cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
		return;
	}

	cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
	forward_yolo_layer(l, net);
	cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
}

void backward_yolo_layer_gpu(const layer l, network net)
{
	axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif









