#ifndef LAYERS_ALL_H
#define LAYERS_ALL_H
#include <iostream>
#include <string>
#include <cmath>
#include "darknet.h"

typedef struct layer layer;
typedef struct network network;
typedef layer local_layer;
typedef layer convolutional_layer;
typedef layer detection_layer;
typedef layer cost_layer;
typedef layer crop_layer;
typedef layer maxpool_layer;
typedef layer avgpool_layer;
typedef layer dropout_layer;
typedef layer route_layer;
typedef layer softmax_layer;
typedef struct update_args update_args;
typedef enum ACTIVATION ACTIVATION;
typedef struct image image;
typedef enum COST_TYPE COST_TYPE;
typedef struct detection detection;
typedef struct box box;
typedef struct tree tree;







//activation_layer.h
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

layer make_activation_layer(int batch, int inputs, ACTIVATION activation);
void forward_activation_layer(layer l, network net);
void backward_activation_layer(layer l, network net);
#ifdef GPU
void forward_activation_layer_gpu(layer l, network net);
void backward_activation_layer_gpu(layer l, network net);
#endif




//activations.h
char *get_activation_string(ACTIVATION a);
ACTIVATION get_activation(char *s);
float activate(float x, ACTIVATION a);
void activate_array(float *x, const int n, const ACTIVATION a);
float gradient(float x, ACTIVATION a);
void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta);




//avgpool_layer.h
avgpool_layer make_avgpool_layer(int batch, int w, int h, int c);
void resize_avgpool_layer(avgpool_layer *l, int w, int h);
void forward_avgpool_layer(const avgpool_layer l, network net);
void backward_avgpool_layer(const avgpool_layer l, network net);




//batchnorm_layer.h
layer make_batchnorm_layer(int batch, int w, int h, int c);
void backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates);
void mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta);
void  variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta);
void normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta);
void resize_batchnorm_layer(layer *layer, int w, int h);
void forward_batchnorm_layer(layer l, network net);
void backward_batchnorm_layer(layer l, network net);
#ifdef GPU

void pull_batchnorm_layer(layer l);
void push_batchnorm_layer(layer l);
void forward_batchnorm_layer_gpu(layer l, network net);
void backward_batchnorm_layer_gpu(layer l, network net);
#endif







//connected_layer.h
layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam);
void update_connected_layer(layer l, update_args a);
void forward_connected_layer(layer l, network net);
void backward_connected_layer(layer l, network net);
void denormalize_connected_layer(layer l);
void statistics_connected_layer(layer l);
#ifdef GPU
void pull_connected_layer(layer l);
void push_connected_layer(layer l);
void update_connected_layer_gpu(layer l, update_args a);
void forward_connected_layer_gpu(layer l, network net);
void backward_connected_layer_gpu(layer l, network net);
#endif







//convolutinal_layer.h
void swap_binary(convolutional_layer *l);
void binarize_weights(float *weights, int n, int size, float *binary);
void binarize_cpu(float *input, int n, float *binary);
void binarize_input(float *input, int n, int size, float *binary);
int convolutional_out_height(convolutional_layer l);
int convolutional_out_width(convolutional_layer l);
image get_convolutional_image(convolutional_layer l);
image get_convolutional_delta(convolutional_layer l);
static size_t get_workspace_size(layer l);
#ifdef GPU
#ifdef CUDNN
void cudnn_convolutional_setup(layer *l
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

convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam);
void denormalize_convolutional_layer(convolutional_layer l);
void resize_convolutional_layer(convolutional_layer *l, int w, int h);
void add_bias(float *output, float *biases, int batch, int n, int size);
void scale_bias(float *output, float *scales, int batch, int n, int size);
void backward_bias(float *bias_updates, float *delta, int batch, int n, int size);
void forward_convolutional_layer(convolutional_layer l, network net);
void backward_convolutional_layer(convolutional_layer l, network net);
void update_convolutional_layer(convolutional_layer l, update_args a);
image get_convolutional_weight(convolutional_layer l, int i);
void rgbgr_weights(convolutional_layer l);
void rescale_weights(convolutional_layer l, float scale, float trans);
image *get_weights(convolutional_layer l);
image *visualize_convolutional_layer(convolutional_layer l, char *window, image *prev_weights);


//cost_layer.h
COST_TYPE get_cost_type(char *s);
char *get_cost_string(COST_TYPE a);
cost_layer make_cost_layer(int batch, int inputs, COST_TYPE cost_type, float scale);
void resize_cost_layer(cost_layer *l, int inputs);
void forward_cost_layer(cost_layer l, network net);
void backward_cost_layer(const cost_layer l, network net);

#ifdef GPU
void pull_cost_layer(cost_layer l);
void push_cost_layer(cost_layer l);
int float_abs_compare(const void * a, const void * b);
void forward_cost_layer_gpu(cost_layer l, network net);
void backward_cost_layer_gpu(const cost_layer l, network net);
#endif








//crnn_layer.h
static void increment_layer(layer *l, int steps);
layer make_crnn_layer(int batch, int h, int w, int c, int hidden_filters, int output_filters, int steps, ACTIVATION activation, int batch_normalize);
void update_crnn_layer(layer l, update_args a);
void forward_crnn_layer(layer l, network net);
void backward_crnn_layer(layer l, network net);
#ifdef GPU
void pull_crnn_layer(layer l);
void push_crnn_layer(layer l);
void update_crnn_layer_gpu(layer l, update_args a);
void forward_crnn_layer_gpu(layer l, network net);
void backward_crnn_layer_gpu(layer l, network net);
#endif








//crop_layer.h
image get_crop_image(crop_layer l);
void backward_crop_layer(const crop_layer l, network net);
void backward_crop_layer_gpu(const crop_layer l, network net);
crop_layer make_crop_layer(int batch, int h, int w, int c, int crop_height, int crop_width, int flip, float angle, float saturation, float exposure);
void resize_crop_layer(layer *l, int w, int h);
void forward_crop_layer(const crop_layer l, network net);











//deconvolutional_layer.h
void bilinear_init(layer l);
layer make_deconvolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int adam);
void denormalize_deconvolutional_layer(layer l);
void resize_deconvolutional_layer(layer *l, int h, int w);
void forward_deconvolutional_layer(const layer l, network net);
void backward_deconvolutional_layer(layer l, network net);
void update_deconvolutional_layer(layer l, update_args a);











//detection_layer.h
detection_layer make_detection_layer(int batch, int inputs, int n, int side, int classes, int coords, int rescore);
void forward_detection_layer(const detection_layer l, network net);
void backward_detection_layer(const detection_layer l, network net);
void get_detection_detections(layer l, int w, int h, float thresh, detection *dets);
#ifdef GPU
void forward_detection_layer_gpu(const detection_layer l, network net);
void backward_detection_layer_gpu(detection_layer l, network net);
#endif








//dropout_layer.h
dropout_layer make_dropout_layer(int batch, int inputs, float probability);
void resize_dropout_layer(dropout_layer *l, int inputs);
void forward_dropout_layer(dropout_layer l, network net);
void backward_dropout_layer(dropout_layer l, network net);







//gru_layer.h
//Already defined in another_layer.c
/*static void increment_layer(layer *l, int steps)*/

layer make_gru_layer(int batch, int inputs, int outputs, int steps, int batch_normalize, int adam);
void update_gru_layer(layer l, update_args a);
void forward_gru_layer(layer l, network net);
void backward_gru_layer(layer l, network net);
#ifdef GPU
void pull_gru_layer(layer l);
void push_gru_layer(layer l);
void update_gru_layer_gpu(layer l, update_args a);
void forward_gru_layer_gpu(layer l, network net);
void backward_gru_layer_gpu(layer l, network net);
#endif








//iseg_layer.h
layer make_iseg_layer(int batch, int w, int h, int classes, int ids);
void resize_iseg_layer(layer *l, int w, int h);
void forward_iseg_layer(const layer l, network net);
void backward_iseg_layer(const layer l, network net);
#ifdef GPU
void forward_iseg_layer_gpu(const layer l, network net);
void backward_iseg_layer_gpu(const layer l, network net);
#endif









//l2norm_layer.h
layer make_l2norm_layer(int batch, int inputs);
void forward_l2norm_layer(const layer l, network net);
void backward_l2norm_layer(const layer l, network net);
#ifdef GPU
void forward_l2norm_layer_gpu(const layer l, network net);
void backward_l2norm_layer_gpu(const layer l, network net);
#endif








//local_layer.h
int local_out_height(local_layer l);
int local_out_width(local_layer l);
local_layer make_local_layer(int batch, int h, int w, int c, int n, int size, int stride, int pad, ACTIVATION activation);
void forward_local_layer(const local_layer l, network net);
void backward_local_layer(local_layer l, network net);
void update_local_layer(local_layer l, update_args a);
#ifdef GPU
void forward_local_layer_gpu(const local_layer l, network net);
void backward_local_layer_gpu(local_layer l, network net);
void update_local_layer_gpu(local_layer l, update_args a);
void pull_local_layer(local_layer l);
void push_local_layer(local_layer l);
#endif








//logistic_layer.h
layer make_logistic_layer(int batch, int inputs);
void forward_logistic_layer(const layer l, network net);
void backward_logistic_layer(const layer l, network net);
#ifdef GPU
void forward_logistic_layer_gpu(const layer l, network net);
void backward_logistic_layer_gpu(const layer l, network net);
#endif










//lstm_layer.h
//static void increment_layer(layer *l, int steps);
layer make_lstm_layer(int batch, int inputs, int outputs, int steps, int batch_normalize, int adam);
void update_lstm_layer(layer l, update_args a);
void forward_lstm_layer(layer l, network state);
void backward_lstm_layer(layer l, network state);
#ifdef GPU
void update_lstm_layer_gpu(layer l, update_args a);
void forward_lstm_layer_gpu(layer l, network state);
void backward_lstm_layer_gpu(layer l, network state);
#endif







//maxpool_layer.h
image get_maxpool_image(maxpool_layer l);
image get_maxpool_delta(maxpool_layer l);
maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding);
void resize_maxpool_layer(maxpool_layer *l, int w, int h);
void forward_maxpool_layer(const maxpool_layer l, network net);
void backward_maxpool_layer(const maxpool_layer l, network net);







//normalization_layer.h
layer make_normalization_layer(int batch, int w, int h, int c, int size, float alpha, float beta, float kappa);
void resize_normalization_layer(layer *layer, int w, int h);
void forward_normalization_layer(const layer layer, network net);
void backward_normalization_layer(const layer layer, network net);

#ifdef GPU
void forward_normalization_layer_gpu(const layer layer, network net);
void backward_normalization_layer_gpu(const layer layer, network net);
#endif







//region_layer.h
layer make_region_layer(int batch, int w, int h, int n, int classes, int coords);
void resize_region_layer(layer *l, int w, int h);
box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h, int stride);
float delta_region_box(box truth, float *x, float *biases, int n, int index, int i, int j, int w, int h, float *delta, float scale, int stride);
void delta_region_mask(float *truth, float *x, int n, int index, float *delta, int stride, int scale);
void delta_region_class(float *output, float *delta, int index, int class_n, int classes, tree *hier, float scale, int stride, float *avg_cat, int tag);
float logit(float x);
float tisnan(float x);
int entry_index(layer l, int batch, int location, int entry);
void forward_region_layer(const layer l, network net);
void backward_region_layer(const layer l, network net);
void correct_region_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative);
void get_region_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets);
#ifdef GPU
void forward_region_layer_gpu(const layer l, network net);
void backward_region_layer_gpu(const layer l, network net);
#endif

void zero_objectness(layer l);








//reorg_layer.h
layer make_reorg_layer(int batch, int w, int h, int c, int stride, int reverse, int flatten, int extra);
void resize_reorg_layer(layer *l, int w, int h);
void forward_reorg_layer(const layer l, network net);
void backward_reorg_layer(const layer l, network net);
#ifdef GPU
void forward_reorg_layer_gpu(layer l, network net);
void backward_reorg_layer_gpu(layer l, network net);
#endif








//rnn_layer.h
//static void increment_layer(layer *l, int steps);
layer make_rnn_layer(int batch, int inputs, int outputs, int steps, ACTIVATION activation, int batch_normalize, int adam);
void update_rnn_layer(layer l, update_args a);
void forward_rnn_layer(layer l, network net);
void backward_rnn_layer(layer l, network net);

#ifdef GPU
void pull_rnn_layer(layer l);
void push_rnn_layer(layer l);
void update_rnn_layer_gpu(layer l, update_args a);
void forward_rnn_layer_gpu(layer l, network net);
void backward_rnn_layer_gpu(layer l, network net);
#endif








//route_layer.h
route_layer make_route_layer(int batch, int n, int *input_layers, int *input_sizes);
void resize_route_layer(route_layer *l, network *net);
void forward_route_layer(const route_layer l, network net);
void backward_route_layer(const route_layer l, network net);
#ifdef GPU
void forward_route_layer_gpu(const route_layer l, network net);
void backward_route_layer_gpu(const route_layer l, network net);
#endif








//shortcut_layer.h
layer make_shortcut_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2);
void resize_shortcut_layer(layer *l, int w, int h);
void forward_shortcut_layer(const layer l, network net);
void backward_shortcut_layer(const layer l, network net);
#ifdef GPU
void forward_shortcut_layer_gpu(const layer l, network net);
void backward_shortcut_layer_gpu(const layer l, network net);
#endif







//softmax_layer.h
softmax_layer make_softmax_layer(int batch, int inputs, int groups);
void forward_softmax_layer(const softmax_layer l, network net);
void backward_softmax_layer(const softmax_layer l, network net);
#ifdef GPU
void pull_softmax_layer_output(const softmax_layer layer);
void forward_softmax_layer_gpu(const softmax_layer l, network net);
void backward_softmax_layer_gpu(const softmax_layer layer, network net);
#endif









//upsample_layer.h
layer make_upsample_layer(int batch, int w, int h, int c, int stride);
void resize_upsample_layer(layer *l, int w, int h);
void forward_upsample_layer(const layer l, network net);
void backward_upsample_layer(const layer l, network net);
#ifdef GPU
void forward_upsample_layer_gpu(const layer l, network net);
void backward_upsample_layer_gpu(const layer l, network net);
#endif










//yolo_layer.h
layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes);
void resize_yolo_layer(layer *l, int w, int h);
box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride);
float delta_yolo_box(box truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride);
void delta_yolo_class(float *output, float *delta, int index, int class_n, int classes, int stride, float *avg_cat);
//static int entry_index(layer l, int batch, int location, int entry);
void forward_yolo_layer(const layer l, network net);
void backward_yolo_layer(const layer l, network net);
void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative);
int yolo_num_detections(layer l, float thresh);
void avg_flipped_yolo(layer l);
int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets);
#ifdef GPU
void forward_yolo_layer_gpu(const layer l, network net);
void backward_yolo_layer_gpu(const layer l, network net);
#endif

































//activations.h

//avgpool_layer.h

//batchnorm_layer.h

//connected_layer.h
//convolutional_layer.h
//cost_layer.h
//crnn_layer.h
//crop_layer.h
//deconvolutional_layer.h
//detectioin_layer.h
//gru_layer.h
//iseg_layer.h
//l2norm_layer.h
//local_layer.h
//logistic_layer.h
//lstm_layer.h
//#define USET
//maxpool_layer.h
//normalization_layer.h
//region_layer.h
//reorg_layer.h
//rnn_layer.h
//#define USET
//route_layer.h
//shortcut_layer.h
//softmax_layers.h
//upsample_layer.h
//yolo_layer.h





#endif
