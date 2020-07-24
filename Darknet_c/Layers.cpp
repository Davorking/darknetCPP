#include "layers.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <string>
#include <string.h>

//Activation.c
std::string get_activation_string(ACTIVATION a)
{
	switch (a) {
	case LOGISTIC:
		return "logistic";
	case LOGGY:
		return "loggy";
	case RELU:
		return "relu";
	case ELU:
		return "elu";
	case SELU:
		return "selu";
	case RELIE:
		return "relie";
	case RAMP:
		return "ramp";
	case LINEAR:
		return "linear";
	case TANH:
		return "tanh";
	case PLSE:
		return "plse";
	case LEAKY:
		return "leaky";
	case STAIR:
		return "stair";
	case HARDTAN:
		return "hardtan";
	case LHTAN:
		return "lhtan";
	default:
		break;
	}
	return "relu";
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
	l.output = new float [output_size];
	l.delta = new float [output_size];
	l.forward = forward_avgpool_layer;
	l.backward = backward_avgpool_layer;
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
	layer l = {};
	l.type = BATCHNORM;
	l.batch = batch;
	l.h = l.out_h = h;
	l.w = l.out_w = w;
	l.c = l.out_c = c;
	l.output = new float [h * w * c * batch];
	l.delta = new float[h * w * c * batch];
	l.inputs = w * h*c;
	l.outputs = l.inputs;

	l.scales = new float [c];
	l.scale_updates = new float [c];
	l.biases = new float [c];
	l.bias_updates = new float [c];
	int i;
	for (i = 0; i < c; ++i) {
		l.scales[i] = 1;
	}

	l.mean = new float [c];
	l.variance = new float [c];

	l.rolling_mean = new float [c];
	l.rolling_variance = new float [c];

//	l.forward = &forward_batchnorm_layer;
//	l.backward = &backward_batchnorm_layer;
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
		mean_delta[i] *= (-1. / sqrt(variance[i] + .00001f));
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
		variance_delta[i] *= -.5 * pow(variance[i] + .00001f, (float)(-3. / 2.));
	}
}

void normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta)
{
	int f, j, k;
	for (j = 0; j < batch; ++j) {
		for (f = 0; f < filters; ++f) {
			for (k = 0; k < spatial; ++k) {
				int index = j * filters*spatial + f * spatial + k;
				delta[index] = delta[index] * 1. / (sqrt(variance[f] + .00001f)) + variance_delta[f] * 2. * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f] / (spatial*batch);
			}
		}
	}
}

void resize_batchnorm_layer(layer *layer, int w, int h)
{
	fprintf(stderr, "Not implemented\n");
}






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

	l.output = new float [batch * outputs];
	l.delta = new float [batch * outputs];

	l.weight_updates = new float [inputs * outputs];
	l.bias_updates = new float [outputs];

	l.weights = new float [outputs * inputs];
	l.biases = new float [outputs];

//	l.forward = forward_connected_layer;
//	l.backward = backward_connected_layer;
//	l.update = update_connected_layer;

	//float scale = 1./sqrt(inputs);
	float scale = sqrt(2. / inputs);
	for (i = 0; i < outputs*inputs; ++i) {
		l.weights[i] = scale * rand_uniform(-1, 1);
	}

	for (i = 0; i < outputs; ++i) {
		l.biases[i] = 0;
	}

	if (adam) {
		l.m = new float [l.inputs * l.outputs];
		l.v = new float [l.inputs * l.outputs];
		l.bias_m = new float [l.outputs];
		l.scale_m = new float [l.outputs];
		l.bias_v = new float [l.outputs];
		l.scale_v = new float [l.outputs];
	}
	if (batch_normalize) {
		l.scales = new float [outputs];
		l.scale_updates = new float [outputs];
		for (i = 0; i < outputs; ++i) {
			l.scales[i] = 1;
		}

		l.mean = new float [outputs];
		l.mean_delta = new float [outputs];
		l.variance = new float [outputs];
		l.variance_delta = new float [outputs];

		l.rolling_mean = new float [outputs];
		l.rolling_variance = new float [outputs];

		l.x = new float [batch * outputs];
		l.x_norm = new float [batch * outputs];
	}

	l.activation = activation;
	fprintf(stderr, "connected                            %4d  ->  %4d\n", inputs, outputs);
	return l;
}

void denormalize_connected_layer(layer l)
{
	int i, j;
	for (i = 0; i < l.outputs; ++i) {
		float scale = l.scales[i] / sqrt(l.rolling_variance[i] + .000001);
		for (j = 0; j < l.inputs; ++j) {
			l.weights[i*l.inputs + j] *= scale;
		}
		l.biases[i] -= l.rolling_mean[i] * scale;
		l.scales[i] = 1;
		l.rolling_mean[i] = 0;
		l.rolling_variance[i] = 1;
	}
}







//convolutinal_layer.c
#ifdef AI2
#include "xnor_layer.h"
#endif

void swap_binary(convolutional_layer *l)
{
	float *swap = l->weights;
	l->weights = l->binary_weights;
	l->binary_weights = swap;
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
		binary[i] = (input[i] > 0) ? 1 : -1;
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

static size_t get_workspace_size(layer l) {
	return (size_t)l.out_h*l.out_w*l.size*l.size*l.c / l.groups * sizeof(float);
}

convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam)
{
	int i;
	convolutional_layer l = {};
	l.type = CONVOLUTIONAL;

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

	l.weights = new float [c/ groups * n * size * size];
	l.weight_updates = new float [c / groups * n * size * size];

	l.biases = new float [n];
	l.bias_updates = new float [n];

	l.nweights = c / groups * n*size*size;
	l.nbiases = n;

	// float scale = 1./sqrt(size*size*c);
	float scale = sqrt(2. / (size*size*c / l.groups));
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

	l.output = new float [l.batch * l.outputs];
	l.delta = new float[l.batch * l.outputs];

//	l.forward = forward_convolutional_layer;
//	l.backward = backward_convolutional_layer;
//	l.update = update_convolutional_layer;
	if (binary) {
		l.binary_weights = new float [l.nweights];
		l.cweights = new char [l.nweights];
		l.scales = new float [n];
	}
	if (xnor) {
		l.binary_weights = new float [l.nweights];
		l.binary_input = new float [l.inputs * l.batch];
	}

	if (batch_normalize) {
		l.scales = new float [n];
		l.scale_updates = new float [n];
		for (i = 0; i < n; ++i) {
			l.scales[i] = 1;
		}

		l.mean = new float[n];
		l.variance = new float[n];

		l.mean_delta = new float[n];
		l.variance_delta = new float[n];

		l.rolling_mean = new float [n];
		l.rolling_variance = new float[n];
		l.x = new float [l.batch * l.outputs];
		l.x_norm = new float [l.batch * l.outputs];
	}
	if (adam) {
		l.m = new float [l.nweights];
		l.v = new float [l.nweights];
		l.bias_m = new float [n];
		l.scale_m = new float [n];
		l.bias_v = new float [n];
		l.scale_v = new float [n];
	}
	l.workspace_size = get_workspace_size(l);
	l.activation = activation;
	fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.size*l.size*l.c / l.groups * l.out_h*l.out_w) / 1000000000.);

	return l;
}

void denormalize_convolutional_layer(convolutional_layer l)
{
	int i, j;
	for (i = 0; i < l.n; ++i) {
		float scale = l.scales[i] / sqrt(l.rolling_variance[i] + .00001);
		for (j = 0; j < l.c / l.groups*l.size*l.size; ++j) {
			l.weights[i*l.c / l.groups*l.size*l.size + j] *= scale;
		}
		l.biases[i] -= l.rolling_mean[i] * scale;
		l.scales[i] = 1;
		l.rolling_mean[i] = 0;
		l.rolling_variance[i] = 1;
	}
}

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









//maxpool_layer.c
maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding)
{
	maxpool_layer l = {};
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
	l.output = (float *)calloc(output_size, sizeof(float));
	l.delta = (float *)calloc(output_size, sizeof(float));
	l.forward = forward_maxpool_layer;
	l.backward = backward_maxpool_layer;
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







