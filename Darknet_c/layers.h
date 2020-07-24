#ifndef LAYERS_ALL_H
#define LAYERS_ALL_H
#include <iostream>
#include <string>
#include "darknet.h"
#include "math.h"

#ifndef AVGPOOL_LAYER_H
#define AVGPOOL_LAYER_H

typedef layer avgpool_layer;

avgpool_layer make_avgpool_layer(int batch, int w, int h, int c);
void resize_avgpool_layer(avgpool_layer *l, int w, int h);
void forward_avgpool_layer(const avgpool_layer l, network net);
void backward_avgpool_layer(const avgpool_layer l, network net);
#endif


#ifndef CONNECTED_LAYER_H
#define CONNECTED_LAYER_H
layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam);

void update_connected_layer(layer l, update_args a);
#endif

#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H
typedef layer convolutional_layer;


convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam);
void resize_convolutional_layer(convolutional_layer *layer, int w, int h);
void update_convolutional_layer(convolutional_layer layer, update_args a);


void binarize_weights(float *weights, int n, int size, float *binary);
void swap_binary(convolutional_layer *l);

void add_bias(float *output, float *biases, int batch, int n, int size);
void backward_bias(float *bias_updates, float *delta, int batch, int n, int size);

int convolutional_out_height(convolutional_layer layer);
int convolutional_out_width(convolutional_layer layer);
#endif


#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H
typedef layer maxpool_layer;

maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding);
void resize_maxpool_layer(maxpool_layer *l, int w, int h);
void forward_maxpool_layer(const maxpool_layer l, network net);
void backward_maxpool_layer(const maxpool_layer l, network net);
#endif


#ifndef BATCHNORM_LAYER_H
#define BATCHNORM_LAYER_H
layer make_batchnorm_layer(int batch, int w, int h, int c);
#endif


#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

ACTIVATION get_activation(char *s);

std::string get_activation_string(ACTIVATION a);
float activate(float x, ACTIVATION a);
float gradient(float x, ACTIVATION a);
void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta);
void activate_array(float *x, const int n, const ACTIVATION a);

static inline float stair_activate(float x)
{
	int n = floor(x);
	if (n % 2 == 0) return floor(x / 2.);
	else return (x - n) + floor(x / 2.);
}
static inline float hardtan_activate(float x)
{
	if (x < -1) return -1;
	if (x > 1) return 1;
	return x;
}
static inline float linear_activate(float x) { return x; }
static inline float logistic_activate(float x) { return 1. / (1. + exp(-x)); }
static inline float loggy_activate(float x) { return 2. / (1. + exp(-x)) - 1; }
static inline float relu_activate(float x) { return x * (x > 0); }
static inline float elu_activate(float x) { return (x >= 0)*x + (x < 0)*(exp(x) - 1); }
static inline float selu_activate(float x) { return (x >= 0)*1.0507*x + (x < 0)*1.0507*1.6732*(exp(x) - 1); }
static inline float relie_activate(float x) { return (x > 0) ? x : .01*x; }
static inline float ramp_activate(float x) { return x * (x > 0) + .1*x; }
static inline float leaky_activate(float x) { return (x > 0) ? x : .1*x; }
static inline float tanh_activate(float x) { return (exp(2 * x) - 1) / (exp(2 * x) + 1); }
static inline float plse_activate(float x)
{
	if (x < -4) return .01 * (x + 4);
	if (x > 4)  return .01 * (x - 4) + 1;
	return .125*x + .5;
}

static inline float lhtan_activate(float x)
{
	if (x < 0) return .001*x;
	if (x > 1) return .001*(x - 1) + 1;
	return x;
}
static inline float lhtan_gradient(float x)
{
	if (x > 0 && x < 1) return 1;
	return .001;
}

static inline float hardtan_gradient(float x)
{
	if (x > -1 && x < 1) return 1;
	return 0;
}
static inline float linear_gradient(float x) { return 1; }
static inline float logistic_gradient(float x) { return (1 - x)*x; }
static inline float loggy_gradient(float x)
{
	float y = (x + 1.) / 2.;
	return 2 * (1 - y)*y;
}
static inline float stair_gradient(float x)
{
	if (floor(x) == x) return 0;
	return 1;
}
static inline float relu_gradient(float x) { return (x > 0); }
static inline float elu_gradient(float x) { return (x >= 0) + (x < 0)*(x + 1); }
static inline float selu_gradient(float x) { return (x >= 0)*1.0507 + (x < 0)*(x + 1.0507*1.6732); }
static inline float relie_gradient(float x) { return (x > 0) ? 1 : .01; }
static inline float ramp_gradient(float x) { return (x > 0) + .1; }
static inline float leaky_gradient(float x) { return (x > 0) ? 1 : .1; }
static inline float tanh_gradient(float x) { return 1 - x * x; }
static inline float plse_gradient(float x) { return (x < 0 || x > 1) ? .01 : .125; }
#endif


#endif
