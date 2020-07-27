#ifndef DARKNET_API
#define DARKNET_API
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <time.h>
#include <pthread.h>
#include "layers.h"
//#include <sys/time.h>




#define SECRET_NUM -1234

	//struct_enum.h
	extern int gpu_index;

	typedef struct {
		int classes;
		char **names;
	} metadata;

	typedef struct tree{
		int *leaf;
		int n;
		int *parent;
		int *child;
		int *group;
		char **name;

		int groups;
		int *group_size;
		int *group_offset;
	} tree;

/*	typedef enum {
		LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN, SELU
	} ACTIVATION;*/

	enum ACTIVATION {
		LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN, SELU
	};


	typedef enum {
		PNG, BMP, TGA, JPG
	} IMTYPE;

	typedef enum {
		MULT, ADD, SUB, DIV
	} BINARY_ACTIVATION;

	typedef enum {
		CONVOLUTIONAL,
		DECONVOLUTIONAL,
		CONNECTED,
		MAXPOOL,
		SOFTMAX,
		DETECTION,
		DROPOUT,
		CROP,
		ROUTE,
		COST,
		NORMALIZATION,
		AVGPOOL,
		LOCAL,
		SHORTCUT,
		ACTIVE,
		RNN,
		GRU,
		LSTM,
		CRNN,
		BATCHNORM,
		NETWORK,
		XNOR,
		REGION,
		YOLO,
		ISEG,
		REORG,
		UPSAMPLE,
		LOGXENT,
		L2NORM,
		BLANK
	} LAYER_TYPE;

	typedef enum COST_TYPE {
		SSE, MASKED, L1, SEG, SMOOTH, WGAN
	} COST_TYPE;

	typedef struct update_args{
		int batch;
		float learning_rate;
		float momentum;
		float decay;
		int adam;
		float B1;
		float B2;
		float eps;
		int t;
	} update_args;

	struct network;
	typedef struct network network;

	struct layer;
	typedef struct layer layer;

	struct layer {
		LAYER_TYPE type;
		ACTIVATION activation;
		COST_TYPE cost_type;
		void(*forward)   (struct layer, struct network);
		void(*backward)  (struct layer, struct network);
		void(*update)    (struct layer, update_args);
		void(*forward_gpu)   (struct layer, struct network);
		void(*backward_gpu)  (struct layer, struct network);
		void(*update_gpu)    (struct layer, update_args);
		int batch_normalize;
		int shortcut;
		int batch;
		int forced;
		int flipped;
		int inputs;
		int outputs;
		int nweights;
		int nbiases;
		int extra;
		int truths;
		int h, w, c;
		int out_h, out_w, out_c;
		int n;
		int max_boxes;
		int groups;
		int size;
		int side;
		int stride;
		int reverse;
		int flatten;
		int spatial;
		int pad;
		int sqrt;
		int flip;
		int index;
		int binary;
		int xnor;
		int steps;
		int hidden;
		int truth;
		float smooth;
		float dot;
		float angle;
		float jitter;
		float saturation;
		float exposure;
		float shift;
		float ratio;
		float learning_rate_scale;
		float clip;
		int noloss;
		int softmax;
		int classes;
		int coords;
		int background;
		int rescore;
		int objectness;
		int joint;
		int noadjust;
		int reorg;
		int log;
		int tanh;
		int *mask;
		int total;

		float alpha;
		float beta;
		float kappa;

		float coord_scale;
		float object_scale;
		float noobject_scale;
		float mask_scale;
		float class_scale;
		int bias_match;
		int random;
		float ignore_thresh;
		float truth_thresh;
		float thresh;
		float focus;
		int classfix;
		int absolute;

		int onlyforward;
		int stopbackward;
		int dontload;
		int dontsave;
		int dontloadscales;
		int numload;

		float temperature;
		float probability;
		float scale;

		char  * cweights;
		int   * indexes;
		int   * input_layers;
		int   * input_sizes;
		int   * map;
		int   * counts;
		float ** sums;
		float * rand;
		float * cost;
		float * state;
		float * prev_state;
		float * forgot_state;
		float * forgot_delta;
		float * state_delta;
		float * combine_cpu;
		float * combine_delta_cpu;

		float * concat;
		float * concat_delta;

		float * binary_weights;

		float * biases;
		float * bias_updates;

		float * scales;
		float * scale_updates;

		float * weights;
		float * weight_updates;

		float * delta;
		float * output;
		float * loss;
		float * squared;
		float * norms;

		float * spatial_mean;
		float * mean;
		float * variance;

		float * mean_delta;
		float * variance_delta;

		float * rolling_mean;
		float * rolling_variance;

		float * x;
		float * x_norm;

		float * m;
		float * v;

		float * bias_m;
		float * bias_v;
		float * scale_m;
		float * scale_v;


		float *z_cpu;
		float *r_cpu;
		float *h_cpu;
		float * prev_state_cpu;

		float *temp_cpu;
		float *temp2_cpu;
		float *temp3_cpu;

		float *dh_cpu;
		float *hh_cpu;
		float *prev_cell_cpu;
		float *cell_cpu;
		float *f_cpu;
		float *i_cpu;
		float *g_cpu;
		float *o_cpu;
		float *c_cpu;
		float *dc_cpu;

		float * binary_input;

		struct layer *input_layer;
		struct layer *self_layer;
		struct layer *output_layer;

		struct layer *reset_layer;
		struct layer *update_layer;
		struct layer *state_layer;

		struct layer *input_gate_layer;
		struct layer *state_gate_layer;
		struct layer *input_save_layer;
		struct layer *state_save_layer;
		struct layer *input_state_layer;
		struct layer *state_state_layer;

		struct layer *input_z_layer;
		struct layer *state_z_layer;

		struct layer *input_r_layer;
		struct layer *state_r_layer;

		struct layer *input_h_layer;
		struct layer *state_h_layer;

		struct layer *wz;
		struct layer *uz;
		struct layer *wr;
		struct layer *ur;
		struct layer *wh;
		struct layer *uh;
		struct layer *uo;
		struct layer *wo;
		struct layer *uf;
		struct layer *wf;
		struct layer *ui;
		struct layer *wi;
		struct layer *ug;
		struct layer *wg;

		tree *softmax_tree;

		size_t workspace_size;



#ifdef GPU
		int *indexes_gpu;

		float *z_gpu;
		float *r_gpu;
		float *h_gpu;

		float *temp_gpu;
		float *temp2_gpu;
		float *temp3_gpu;

		float *dh_gpu;
		float *hh_gpu;
		float *prev_cell_gpu;
		float *cell_gpu;
		float *f_gpu;
		float *i_gpu;
		float *g_gpu;
		float *o_gpu;
		float *c_gpu;
		float *dc_gpu;

		float *m_gpu;
		float *v_gpu;
		float *bias_m_gpu;
		float *scale_m_gpu;
		float *bias_v_gpu;
		float *scale_v_gpu;

		float * combine_gpu;
		float * combine_delta_gpu;

		float * prev_state_gpu;
		float * forgot_state_gpu;
		float * forgot_delta_gpu;
		float * state_gpu;
		float * state_delta_gpu;
		float * gate_gpu;
		float * gate_delta_gpu;
		float * save_gpu;
		float * save_delta_gpu;
		float * concat_gpu;
		float * concat_delta_gpu;

		float * binary_input_gpu;
		float * binary_weights_gpu;

		float * mean_gpu;
		float * variance_gpu;

		float * rolling_mean_gpu;
		float * rolling_variance_gpu;

		float * variance_delta_gpu;
		float * mean_delta_gpu;

		float * x_gpu;
		float * x_norm_gpu;
		float * weights_gpu;
		float * weight_updates_gpu;
		float * weight_change_gpu;

		float * biases_gpu;
		float * bias_updates_gpu;
		float * bias_change_gpu;

		float * scales_gpu;
		float * scale_updates_gpu;
		float * scale_change_gpu;

		float * output_gpu;
		float * loss_gpu;
		float * delta_gpu;
		float * rand_gpu;
		float * squared_gpu;
		float * norms_gpu;
#ifdef CUDNN
		cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
		cudnnTensorDescriptor_t dsrcTensorDesc, ddstTensorDesc;
		cudnnTensorDescriptor_t normTensorDesc;
		cudnnFilterDescriptor_t weightDesc;
		cudnnFilterDescriptor_t dweightDesc;
		cudnnConvolutionDescriptor_t convDesc;
		cudnnConvolutionFwdAlgo_t fw_algo;
		cudnnConvolutionBwdDataAlgo_t bd_algo;
		cudnnConvolutionBwdFilterAlgo_t bf_algo;
#endif
#endif
	};

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




	void free_layer(layer);

	typedef enum {
		CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
	} learning_rate_policy;

	typedef struct network {
		int n;
		int batch;
		size_t *seen;
		int *t;
		float epoch;
		int subdivisions;
		layer *layers;
		float *output;
		learning_rate_policy policy;

		float learning_rate;
		float momentum;
		float decay;
		float gamma;
		float scale;
		float power;
		int time_steps;
		int step;
		int max_batches;
		float *scales;
		int   *steps;
		int num_steps;
		int burn_in;

		int adam;
		float B1;
		float B2;
		float eps;

		int inputs;
		int outputs;
		int truths;
		int notruth;
		int h, w, c;
		int max_crop;
		int min_crop;
		float max_ratio;
		float min_ratio;
		int center;
		float angle;
		float aspect;
		float exposure;
		float saturation;
		float hue;
		int random;

		int gpu_index;
		tree *hierarchy;

		float *input;
		float *truth;
		float *delta;
		float *workspace;
		int train;
		int index;
		float *cost;
		float clip;

#ifdef GPU
		float *input_gpu;
		float *truth_gpu;
		float *delta_gpu;
		float *output_gpu;
#endif

	} network;

	typedef struct {
		int w;
		int h;
		float scale;
		float rad;
		float dx;
		float dy;
		float aspect;
	} augment_args;

	typedef struct image{
		int w;
		int h;
		int c;
		float *data;
	} image;

	typedef struct box{
		float x, y, w, h;
	} box;

	typedef struct detection {
		box bbox;
		int classes;
		float *prob;
		float *mask;
		float objectness;
		int sort_class;
	} detection;

	typedef struct matrix {
		int rows, cols;
		float **vals;
	} matrix;


	typedef struct {
		int w, h;
		matrix X;
		matrix y;
		int shallow;
		int *num_boxes;
		box **boxes;
	} data;

	typedef enum {
		CLASSIFICATION_DATA, DETECTION_DATA, CAPTCHA_DATA, REGION_DATA, IMAGE_DATA, COMPARE_DATA, WRITING_DATA, SWAG_DATA, TAG_DATA, OLD_CLASSIFICATION_DATA, STUDY_DATA, DET_DATA, SUPER_DATA, LETTERBOX_DATA, REGRESSION_DATA, SEGMENTATION_DATA, INSTANCE_DATA, ISEG_DATA
	} data_type;

	typedef struct load_args {
		int threads;
		char **paths;
		char *path;
		int n;
		int m;
		char **labels;
		int h;
		int w;
		int out_w;
		int out_h;
		int nh;
		int nw;
		int num_boxes;
		int min, max, size;
		int classes;
		int background;
		int scale;
		int center;
		int coords;
		float jitter;
		float angle;
		float aspect;
		float saturation;
		float exposure;
		float hue;
		data *d;
		image *im;
		image *resized;
		data_type type;
		tree *hierarchy;
	} load_args;

	typedef struct {
		int id;
		float x, y, w, h;
		float left, right, top, bottom;
	} box_label;

	typedef struct node {
		void *val;
		struct node *next;
		struct node *prev;
	} node;

	typedef struct list {
		int size;
		node *front;
		node *back;
	} list;

	//blas.h
#include "darknet.h"

	void reorg_cpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out);
	void flatten(float *x, int size, int layers, int batch, int forward);
	void weighted_sum_cpu(float *a, float *b, float *s, int n, float *c);
	void weighted_delta_cpu(float *a, float *b, float *s, float *da, float *db, float *ds, int n, float *dc);
	void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out);
	void mean_cpu(float *x, int batch, int filters, int spatial, float *mean);
	void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);
	void l2normalize_cpu(float *x, float *dx, int batch, int filters, int spatial);
	void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);
	void const_cpu(int N, float ALPHA, float *X, int INCX);
	void mul_cpu(int N, float *X, int INCX, float *Y, int INCY);
	void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
	void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
	void scal_cpu(int N, float ALPHA, float *X, int INCX);
	void fill_cpu(int N, float ALPHA, float *X, int INCX);
	void deinter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT);
	void inter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT);
	void copy_cpu(int N, float *X, int INCX, float *Y, int INCY);
	void mult_add_into_cpu(int N, float *X, float *Y, float *Z);
	void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error);
	void l1_cpu(int n, float *pred, float *truth, float *delta, float *error);
	void softmax_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error);
	void logistic_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error);
	void l2_cpu(int n, float *pred, float *truth, float *delta, float *error);
	float dot_cpu(int N, float *X, int INCX, float *Y, int INCY);
	void softmax(float *input, int n, float temp, int stride, float *output);
	void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output);
	void upsample_cpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out);

#ifdef GPU
#include "cuda.h"
#include "tree.h"

	void axpy_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY);
	void axpy_gpu_offset(int N, float ALPHA, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY);
	void copy_gpu(int N, float * X, int INCX, float * Y, int INCY);
	void copy_gpu_offset(int N, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY);
	void add_gpu(int N, float ALPHA, float * X, int INCX);
	void supp_gpu(int N, float ALPHA, float * X, int INCX);
	void mask_gpu(int N, float * X, float mask_num, float * mask, float val);
	void scale_mask_gpu(int N, float * X, float mask_num, float * mask, float scale);
	void const_gpu(int N, float ALPHA, float *X, int INCX);
	void pow_gpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
	void mul_gpu(int N, float *X, int INCX, float *Y, int INCY);

	void mean_gpu(float *x, int batch, int filters, int spatial, float *mean);
	void variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);
	void normalize_gpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);
	void l2normalize_gpu(float *x, float *dx, int batch, int filters, int spatial);

	void normalize_delta_gpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta);

	void fast_mean_delta_gpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta);
	void fast_variance_delta_gpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta);

	void fast_variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);
	void fast_mean_gpu(float *x, int batch, int filters, int spatial, float *mean);
	void shortcut_gpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out);
	void scale_bias_gpu(float *output, float *biases, int batch, int n, int size);
	void backward_scale_gpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates);
	void scale_bias_gpu(float *output, float *biases, int batch, int n, int size);
	void add_bias_gpu(float *output, float *biases, int batch, int n, int size);
	void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size);

	void logistic_x_ent_gpu(int n, float *pred, float *truth, float *delta, float *error);
	void softmax_x_ent_gpu(int n, float *pred, float *truth, float *delta, float *error);
	void smooth_l1_gpu(int n, float *pred, float *truth, float *delta, float *error);
	void l2_gpu(int n, float *pred, float *truth, float *delta, float *error);
	void l1_gpu(int n, float *pred, float *truth, float *delta, float *error);
	void wgan_gpu(int n, float *pred, float *truth, float *delta, float *error);
	void weighted_delta_gpu(float *a, float *b, float *s, float *da, float *db, float *ds, int num, float *dc);
	void weighted_sum_gpu(float *a, float *b, float *s, int num, float *c);
	void mult_add_into_gpu(int num, float *a, float *b, float *c);
	void inter_gpu(int NX, float *X, int NY, float *Y, int B, float *OUT);
	void deinter_gpu(int NX, float *X, int NY, float *Y, int B, float *OUT);

	void reorg_gpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out);

	void softmax_gpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output);
	void adam_update_gpu(float *w, float *d, float *m, float *v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t);
	void adam_gpu(int n, float *x, float *m, float *v, float B1, float B2, float rate, float eps, int t);

	void flatten_gpu(float *x, int spatial, int layers, int batch, int forward, float *out);
	void softmax_tree(float *input, int spatial, int batch, int stride, float temp, float *output, tree hier);
	void upsample_gpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out);
#endif

	//box.h
	typedef struct {
		float dx, dy, dw, dh;
	} dbox;

	int nms_comparator(const void *pa, const void *pb);
	void do_nms_obj(detection *dets, int total, int classes, float thresh);
	void do_nms_sort(detection *dets, int total, int classes, float thresh);
	box float_to_box(float *f, int stride);
	dbox derivative(box a, box b);
	float overlap(float x1, float w1, float x2, float w2);
	float box_intersection(box a, box b);
	float box_union(box a, box b);
	float box_iou(box a, box b);
	float box_rmse(box a, box b);
	dbox dintersect(box a, box b);
	dbox dunion(box a, box b);
	void test_dunion();
	void test_dintersect();
	void test_box();
	dbox diou(box a, box b);
	void do_nms(box *boxes, float **probs, int total, int classes, float thresh);
	box encode_box(box b, box anchor);
	box decode_box(box b, box anchor);


	//col2im.h
	void col2im_add_pixel(float *im, int height, int width, int channels,
		int row, int col, int channel, int pad, float val);
	//This one might be too, can't remember.
	void col2im_cpu(float* data_col,
		int channels, int height, int width,
		int ksize, int stride, int pad, float* data_im);
#ifdef GPU
	void col2im_gpu(float *data_col,
		int channels, int height, int width,
		int ksize, int stride, int pad, float *data_im);
#endif


	//compare.h
	void train_compare(char *cfgfile, char *weightfile);
	void validate_compare(char *filename, char *weightfile);
	typedef struct {
		network net;
		char *filename;
		int class_n;
		int classes;
		float elo;
		float *elos;
	} sortable_bbox;
	int elo_comparator(const void*a, const void *b);
	int bbox_comparator(const void *a, const void *b);
	void bbox_update(sortable_bbox *a, sortable_bbox *b, int class_n, int result);
	void bbox_fight(network net, sortable_bbox *a, sortable_bbox *b, int classes, int class_n);
	void SortMaster3000(char *filename, char *weightfile);
	void BattleRoyaleWithCheese(char *filename, char *weightfile);
	void run_compare(int argc, char **argv);


	//cuda.h
#ifdef GPU
	void cuda_set_device(int n);
	int cuda_get_device();
	void check_error(cudaError_t status);
	dim3 cuda_gridsize(size_t n);

#ifdef CUDNN
	cudnnHandle_t cudnn_handle();
#endif

	cublasHandle_t blas_handle();
	float *cuda_make_array(float *x, size_t n);
	void cuda_random(float *x_gpu, size_t n);
	float cuda_compare(float *x_gpu, float *x, size_t n, char *s);
	int *cuda_make_int_array(int *x, size_t n);
	void cuda_free(float *x_gpu);
	void cuda_push_array(float *x_gpu, float *x, size_t n);
	void cuda_pull_array(float *x_gpu, float *x, size_t n);
	float cuda_mag_array(float *x_gpu, size_t n);
#else
	void cuda_set_device(int n);
#endif


	//data.h
#define NUMCHARS 37
	static inline float distance_from_edge(int x, int max)
	{
		int dx = (max / 2) - x;
		if (dx < 0) dx = -dx;
		dx = (max / 2) + 1 - dx;
		dx *= 2;
		float dist = (float)dx / max;
		if (dist > 1) dist = 1;
		return dist;
	}

	list *get_paths(char *filename);
	char **get_random_paths(char **paths, int n, int m);
	char **find_replace_paths(char **paths, int n, char *find, char *replace);
	matrix load_image_paths_gray(char **paths, int n, int w, int h);
	matrix load_image_paths(char **paths, int n, int w, int h);
	matrix load_image_augment_paths(char **paths, int n, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure, int center);
	box_label *read_boxes(char *filename, int *n);
	void randomize_boxes(box_label *b, int n);
	void correct_boxes(box_label *boxes, int n, float dx, float dy, float sx, float sy, int flip);
	void fill_truth_swag(char *path, float *truth, int classes, int flip, float dx, float dy, float sx, float sy);
	void fill_truth_region(char *path, float *truth, int classes, int num_boxes, int flip, float dx, float dy, float sx, float sy);
	void load_rle(image im, int *rle, int n);
	void or_image(image src, image dest, int c);
	void exclusive_image(image src);
	box bound_image(image im);
	void fill_truth_iseg(char *path, int num_boxes, float *truth, int classes, int w, int h, augment_args aug, int flip, int mw, int mh);
	void fill_truth_mask(char *path, int num_boxes, float *truth, int classes, int w, int h, augment_args aug, int flip, int mw, int mh);
	void fill_truth_detection(char *path, int num_boxes, float *truth, int classes, int flip, float dx, float dy, float sx, float sy);
	void print_letters(float *pred, int n);
	void fill_truth_captcha(char *path, int n, float *truth);
	data load_data_captcha(char **paths, int n, int m, int k, int w, int h);
	data load_data_captcha_encode(char **paths, int n, int m, int w, int h);
	void fill_truth(char *path, char **labels, int k, float *truth);
	void fill_hierarchy(float *truth, int k, tree *hierarchy);
	matrix load_regression_labels_paths(char **paths, int n, int k);
	matrix load_labels_paths(char **paths, int n, char **labels, int k, tree *hierarchy);
	matrix load_tags_paths(char **paths, int n, int k);
	char **get_labels(char *filename);
	void free_data(data d);
	image get_segmentation_image(char *path, int w, int h, int classes);
	image get_segmentation_image2(char *path, int w, int h, int classes);
	data load_data_seg(int n, char **paths, int m, int w, int h, int classes, int min, int max, float angle, float aspect, float hue, float saturation, float exposure, int div);
	data load_data_iseg(int n, char **paths, int m, int w, int h, int classes, int boxes, int div, int min, int max, float angle, float aspect, float hue, float saturation, float exposure);
	data load_data_mask(int n, char **paths, int m, int w, int h, int classes, int boxes, int coords, int min, int max, float angle, float aspect, float hue, float saturation, float exposure);
	data load_data_region(int n, char **paths, int m, int w, int h, int size, int classes, float jitter, float hue, float saturation, float exposure);
	data load_data_compare(int n, char **paths, int m, int classes, int w, int h);
	data load_data_swag(char **paths, int n, int classes, float jitter);
	data load_data_detection(int n, char **paths, int m, int w, int h, int boxes, int classes, float jitter, float hue, float saturation, float exposure);
	void *load_thread(void *ptr);
	pthread_t load_data_in_thread(load_args args);
	void *load_threads(void *ptr);
	void load_data_blocking(load_args args);
	pthread_t load_data(load_args args);
	data load_data_writing(char **paths, int n, int m, int w, int h, int out_w, int out_h);
	data load_data_old(char **paths, int n, int m, char **labels, int k, int w, int h);
	data load_data_super(char **paths, int n, int m, int w, int h, int scale);
	data load_data_regression(char **paths, int n, int m, int k, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure);
	data select_data(data *orig, int *inds);
	data *tile_data(data orig, int divs, int size);
	data resize_data(data orig, int w, int h);
	data load_data_augment(char **paths, int n, int m, char **labels, int k, tree *hierarchy, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure, int center);
	data load_data_tag(char **paths, int n, int m, int k, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure);
	matrix concat_matrix(matrix m1, matrix m2);
	data concat_data(data d1, data d2);
	data concat_datas(data *d, int n);
	data load_categorical_data_csv(char *filename, int target, int k);
	data load_cifar10_data(char *filename);
	void get_random_batch(data d, int n, float *X, float *y);
	void get_next_batch(data d, int n, int offset, float *X, float *y);
	void smooth_data(data d);
	data load_all_cifar10();
	data load_go(char *filename);
	void randomize_data(data d);
	void scale_data_rows(data d, float s);
	void translate_data_rows(data d, float s);
	data copy_data(data d);
	void normalize_data_rows(data d);
	data get_data_part(data d, int part, int total);
	data get_random_data(data d, int num);
	data *split_data(data d, int part, int total);


	//demo.h
#define DEMO 1

#ifdef CV_VERSION
	static char **demo_names;
	static image **demo_alphabet;
	static int demo_classes;

	static network *net;
	static image buff[3];
	static image buff_letter[3];
	static int buff_index = 0;
	static void * cap;
	static float fps = 0;
	static float demo_thresh = 0;
	static float demo_hier = .5;
	static int running = 0;

	static int demo_frame = 3;
	static int demo_index = 0;
	static float **predictions;
	static float *avg;
	static int demo_done = 0;
	static int demo_total = 0;
	double demo_time;

	detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num);

	int size_network(network *net);
	void remember_network(network *net);
	detection *avg_predictions(network *net, int *nboxes);
	void *detect_in_thread(void *ptr);
	void *fetch_in_thread(void *ptr);
	void *display_in_thread(void *ptr);
	void *display_loop(void *ptr);
	void *detect_loop(void *ptr);
	void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen);
#endif


	//gemm.h
	void gemm_bin(int M, int N, int K, float ALPHA,
		char  *A, int lda,
		float *B, int ldb,
		float *C, int ldc);
	float *random_matrix(int rows, int cols);
	void time_random_matrix(int TA, int TB, int m, int k, int n);
	void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
		float *A, int lda,
		float *B, int ldb,
		float BETA,
		float *C, int ldc);
	void gemm_nn(int M, int N, int K, float ALPHA,
		float *A, int lda,
		float *B, int ldb,
		float *C, int ldc);
	void gemm_nt(int M, int N, int K, float ALPHA,
		float *A, int lda,
		float *B, int ldb,
		float *C, int ldc);
	void gemm_tn(int M, int N, int K, float ALPHA,
		float *A, int lda,
		float *B, int ldb,
		float *C, int ldc);
	void gemm_tt(int M, int N, int K, float ALPHA,
		float *A, int lda,
		float *B, int ldb,
		float *C, int ldc);
	void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
		float *A, int lda,
		float *B, int ldb,
		float BETA,
		float *C, int ldc);
#ifdef GPU
	void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA,
		float *A_gpu, int lda,
		float *B_gpu, int ldb,
		float BETA,
		float *C_gpu, int ldc);
	void time_gpu_random_matrix(int TA, int TB, int m, int k, int n);
	void time_gpu(int TA, int TB, int m, int k, int n);
	void test_gpu_accuracy(int TA, int TB, int m, int k, int n);
	int test_gpu_blas();
#endif


	//im2col.h
	float im2col_get_pixel(float *im, int height, int width, int channels,
		int row, int col, int channel, int pad);
	//From Berkeley Vision's Caffe!
	//https://github.com/BVLC/caffe/blob/master/LICENSE
	void im2col_cpu(float* data_im,
		int channels, int height, int width,
		int ksize, int stride, int pad, float* data_col);
#ifdef GPU
	void im2col_gpu(float *im,
		int channels, int height, int width,
		int ksize, int stride, int pad, float *data_col);

#endif


	//image.h
//#define STB_IMAGE_IMPLEMENTATION
//#include "stb_image.h"
//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#include "stb_image_write.h"

//	int windows = 0;
//	float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };


#ifdef __cplusplus
	extern "C" {
#endif
		float get_color(int c, int x, int max);
		image mask_to_rgb(image mask);
		static float get_pixel(image m, int x, int y, int c);
		static float get_pixel_extend(image m, int x, int y, int c);
		static void set_pixel(image m, int x, int y, int c, float val);
		static void add_pixel(image m, int x, int y, int c, float val);
		static float bilinear_interpolate(image im, float x, float y, int c);
		void composite_image(image source, image dest, int dx, int dy);
		image border_image(image a, int border);
		image tile_images(image a, image b, int dx);
		image get_label(image **characters, char *string, int size);
		void draw_label(image a, int r, int c, image label, const float *rgb);
		void draw_box(image a, int x1, int y1, int x2, int y2, float r, float g, float b);
		void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b);
		void draw_bbox(image a, box bbox, int w, float r, float g, float b);
		image **load_alphabet();
		void draw_detections(image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes);
		void transpose_image(image im);
		void rotate_image_cw(image im, int times);
		void flip_image(image a);
		image image_distance(image a, image b);
		void ghost_image(image source, image dest, int dx, int dy);
		void blocky_image(image im, int s);
		void censor_image(image im, int dx, int dy, int w, int h);
		void embed_image(image source, image dest, int dx, int dy);
		image collapse_image_layers(image source, int border);
		void constrain_image(image im);
		void normalize_image(image p);
		void normalize_image2(image p);
		void copy_image_into(image src, image dest);
		image copy_image(image p);
		void rgbgr_image(image im);
		int show_image(image p, const char *name, int ms);
		void save_image_options(image im, const char *name, IMTYPE f, int quality);
		void save_image(image im, const char *name);
		void show_image_layers(image p, char *name);
		void show_image_collapsed(image p, char *name);
		image make_empty_image(int w, int h, int c);
		image make_image(int w, int h, int c);
		image make_random_image(int w, int h, int c);
		image float_to_image(int w, int h, int c, float *data);
		void place_image(image im, int w, int h, int dx, int dy, image canvas);
		image center_crop_image(image im, int w, int h);
		image rotate_crop_image(image im, float rad, float s, int w, int h, float dx, float dy, float aspect);
		image rotate_image(image im, float rad);
		void fill_image(image m, float s);
		void translate_image(image m, float s);
		void scale_image(image m, float s);
		image crop_image(image im, int dx, int dy, int w, int h);
		int best_3d_shift_r(image a, image b, int min, int max);
		int best_3d_shift(image a, image b, int min, int max);
		void composite_3d(char *f1, char *f2, char *out, int delta);
		void letterbox_image_into(image im, int w, int h, image boxed);
		image letterbox_image(image im, int w, int h);
		image resize_max(image im, int max);
		image resize_min(image im, int min);
		image random_crop_image(image im, int w, int h);
		augment_args random_augment_args(image im, float angle, float aspect, int low, int high, int w, int h);
		image random_augment_image(image im, float angle, float aspect, int low, int high, int w, int h);
		float three_way_max(float a, float b, float c);
		float three_way_min(float a, float b, float c);
		void yuv_to_rgb(image im);
		void rgb_to_yuv(image im);
		void rgb_to_hsv(image im);
		void hsv_to_rgb(image im);
		void grayscale_image_3c(image im);
		image grayscale_image(image im);
		image threshold_image(image im, float thresh);
		image blend_image(image fore, image back, float alpha);
		void scale_image_channel(image im, int c, float v);
		void translate_image_channel(image im, int c, float v);
		image binarize_image(image im);
		void saturate_image(image im, float sat);
		void hue_image(image im, float hue);
		void exposure_image(image im, float sat);
		void distort_image(image im, float hue, float sat, float val);
		void random_distort_image(image im, float hue, float saturation, float exposure);
		void saturate_exposure_image(image im, float sat, float exposure);
		image resize_image(image im, int w, int h);
		void test_resize(char *filename);
		image load_image_stb(char *filename, int channels);
		image load_image(char *filename, int w, int h, int c);
		image load_image_color(char *filename, int w, int h);
		image get_image_layer(image m, int l);
		void print_image(image m);
		image collapse_images_vert(image *ims, int n);
		image collapse_images_horz(image *ims, int n);
		void show_image_normalized(image im, const char *name);
		void show_images(image *ims, int n, char *window);
		void free_image(image m);
#ifdef __cplusplus
	}
#endif



	//layer.h
	void free_layer(layer l);



	//list.h
	list *make_list();
	void *list_pop(list *l);
	void list_insert(list *l, void *val);
	void free_node(node *n);
	void free_list(list *l);
	void free_list_contents(list *l);
	void **list_to_array(list *l);


	//matrix.h
	void free_matrix(matrix m);
	float matrix_topk_accuracy(matrix truth, matrix guess, int k);
	void scale_matrix(matrix m, float scale);
	matrix resize_matrix(matrix m, int size);
	void matrix_add_matrix(matrix from, matrix to);
	matrix copy_matrix(matrix m);
	matrix make_matrix(int rows, int cols);
	matrix hold_out_matrix(matrix *m, int n);
	float *pop_column(matrix *m, int c);
	matrix csv_to_matrix(char *filename);
	void matrix_to_csv(matrix m);
	void print_matrix(matrix m);


	//network.h
	load_args get_base_args(network *net);
	network *load_network(char *cfg, char *weights, int clear);
	size_t get_current_batch(network *net);
	void reset_network_state(network *net, int b);
	void reset_rnn(network *net);
	float get_current_rate(network *net);
	char *get_layer_string(LAYER_TYPE a);
	network *make_network(int n);
	void forward_network(network *netp);
	void update_network(network *netp);
	void calc_network_cost(network *netp);
	int get_predicted_class_network(network *net);
	void backward_network(network *netp);
	float train_network_datum(network *net);
	float train_network_sgd(network *net, data d, int n);
	float train_network(network *net, data d);
	void set_temp_network(network *net, float t);
	void set_batch_network(network *net, int b);
	int resize_network(network *net, int w, int h);
	layer get_network_detection_layer(network *net);
	image get_network_image_layer(network *net, int i);
	image get_network_image(network *net);
	void visualize_network(network *net);
	void top_predictions(network *net, int k, int *index);
	float *network_predict(network *net, float *input);
	int num_detections(network *net, float thresh);
	detection *make_network_boxes(network *net, float thresh, int *num);
	void fill_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, detection *dets);
	detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num);
	void free_detections(detection *dets, int n);
	float *network_predict_image(network *net, image im);
	int network_width(network *net);
	int network_height(network *net);
	matrix network_predict_data_multi(network *net, data test, int n);
	matrix network_predict_data(network *net, data test);
	void print_network(network *net);
	void compare_networks(network *n1, network *n2, data test);
	float network_accuracy(network *net, data d);
	float *network_accuracies(network *net, data d, int n);
	layer get_network_output_layer(network *net);
	float network_accuracy_multi(network *net, data d, int n);
	void free_network(network *net);
	layer network_output_layer(network *net);
	int network_inputs(network *net);
	int network_outputs(network *net);
	float *network_output(network *net);

#ifdef GPU
	void forward_network_gpu(network *netp);
	void backward_network_gpu(network *netp);
	void update_network_gpu(network *netp);
	void harmless_update_network_gpu(network *netp);
	typedef struct {
		network *net;
		data d;
		float *err;
	} train_args;
	void *train_thread(void *ptr);
	pthread_t train_network_in_thread(network *net, data d, float *err);
	void merge_weights(layer l, layer base);
	void scale_weights(layer l, float s);
	void pull_weights(layer l);
	void push_weights(layer l);
	void distribute_weights(layer l, layer base);
	void sync_layer(network **nets, int n, int j);
	typedef struct {
		network **nets;
		int n;
		int j;
	} sync_args;

	void *sync_layer_thread(void *ptr);
	pthread_t sync_layer_in_thread(network **nets, int n, int j);
	void sync_nets(network **nets, int n, int interval);
	float train_networks(network **nets, int n, data d, int interval);
	void pull_network_output(network *net);
#endif



	//option_list.h
	typedef struct {
		char *key;
		char *val;
		int used;
	} kvp;
	list *read_data_cfg(char *filename);
	metadata get_metadata(char *file);
	int read_option(char *s, list *options);
	void option_insert(list *l, char *key, char *val);
	void option_unused(list *l);
	char *option_find(list *l, char *key);
	char *option_find_str(list *l, char *key, char *def);
	int option_find_int(list *l, char *key, int def);
	int option_find_int_quiet(list *l, char *key, int def);
	float option_find_float_quiet(list *l, char *key, float def);
	float option_find_float(list *l, char *key, float def);


	//parser.h
	typedef struct {
		char *type;
		list *options;
	}section;

	typedef struct size_params {
		int batch;
		int inputs;
		int h;
		int w;
		int c;
		int index;
		int time_steps;
		network *net;
	} size_params;

	LAYER_TYPE string_to_layer_type(char * type);
	void free_section(section *s);
	void parse_data(char *data, float *a, int n);

	local_layer parse_local(list *options, size_params params);
	layer parse_deconvolutional(list *options, size_params params);
	convolutional_layer parse_convolutional(list *options, size_params params);
	layer parse_crnn(list *options, size_params params);
	layer parse_rnn(list *options, size_params params);
	layer parse_gru(list *options, size_params params);
	layer parse_lstm(list *options, size_params params);
	layer parse_connected(list *options, size_params params);
	layer parse_softmax(list *options, size_params params);
	int *parse_yolo_mask(char *a, int *num);
	layer parse_yolo(list *options, size_params params);
	layer parse_iseg(list *options, size_params params);
	layer parse_region(list *options, size_params params);
	detection_layer parse_detection(list *options, size_params params);
	cost_layer parse_cost(list *options, size_params params);
	crop_layer parse_crop(list *options, size_params params);
	layer parse_reorg(list *options, size_params params);
	maxpool_layer parse_maxpool(list *options, size_params params);
	avgpool_layer parse_avgpool(list *options, size_params params);
	dropout_layer parse_dropout(list *options, size_params params);
	layer parse_normalization(list *options, size_params params);
	layer parse_batchnorm(list *options, size_params params);
	layer parse_shortcut(list *options, size_params params, network *net);
	layer parse_l2norm(list *options, size_params params);
	layer parse_logistic(list *options, size_params params);
	layer parse_activation(list *options, size_params params);
	layer parse_upsample(list *options, size_params params, network *net);
	route_layer parse_route(list *options, size_params params, network *net);
	learning_rate_policy get_policy(char *s);
	void parse_net_options(list *options, network *net);
	int is_network(section *s);
	network *parse_network_cfg(char *filename);
	list *read_cfg(char *filename);
	void save_convolutional_weights_binary(layer l, FILE *fp);
	void save_convolutional_weights(layer l, FILE *fp);
	void save_batchnorm_weights(layer l, FILE *fp);
	void save_connected_weights(layer l, FILE *fp);
	void save_weights_upto(network *net, char *filename, int cutoff);
	void save_weights(network *net, char *filename);
	void transpose_matrix(float *a, int rows, int cols);
	void load_connected_weights(layer l, FILE *fp, int transpose);
	void load_batchnorm_weights(layer l, FILE *fp);
	void load_convolutional_weights_binary(layer l, FILE *fp);
	void load_convolutional_weights(layer l, FILE *fp);
	void load_weights_upto(network *net, char *filename, int start, int cutoff);
	void load_weights(network *net, char *filename);

	//tree.h
	void change_leaves(tree *t, char *leaf_list);
	float get_hierarchy_probability(float *x, tree *hier, int c, int stride);
	void hierarchy_predictions(float *predictions, int n, tree *hier, int only_leaves, int stride);
	int hierarchy_top_prediction(float *predictions, tree *hier, float thresh, int stride);
	tree *read_tree(char *filename);

	//utils.h
#define TIME(a) \
    do { \
    double start = what_time_is_it_now(); \
    a; \
    printf("%s took: %f seconds\n", #a, what_time_is_it_now() - start); \
    } while (0)

#define TWO_PI 6.2831853071795864769252866f

	int *read_intlist(char *gpu_list, int *ngpus, int d);
	int *read_map(char *filename);
	void sorta_shuffle(sortable_bbox *arr, size_t n, size_t size, size_t sections);
	void shuffle(sortable_bbox *arr, size_t n, size_t size);
	int *random_index_order(int min, int max);
	void del_arg(int argc, char **argv, int index);
	int find_arg(int argc, char* argv[], char *arg);
	int find_int_arg(int argc, char **argv, char *arg, int def);
	float find_float_arg(int argc, char **argv, char *arg, float def);
	char *find_char_arg(int argc, char **argv, char *arg, char *def);
	char *basecfg(char *cfgfile);
	int alphanum_to_int(char c);
	char int_to_alphanum(int i);
	void pm(int M, int N, float *A);
	void find_replace(char *str, char *orig, char *rep, char *output);
	float sec(clock_t clocks);
	void top_k(float *a, int n, int k, int *index);
	void error(const char *s);
	unsigned char *read_file(char *filename);
	void malloc_error();
	void file_error(char *s);
	list *split_str(char *s, char delim);
	void strip(char *s);
	void strip_char(char *s, char bad);
	void free_ptrs(void **ptrs, int n);
	char *fgetl(FILE *fp);
	int read_int(int fd);
	void write_int(int fd, int n);
	int read_all_fail(int fd, char *buffer, size_t bytes);
	int write_all_fail(int fd, char *buffer, size_t bytes);
	void read_all(int fd, char *buffer, size_t bytes);
	void write_all(int fd, char *buffer, size_t bytes);
	char *copy_string(char *s);
	list *parse_csv_line(char *line);
	int count_fields(char *line);
	float *parse_fields(char *line, int n);
	float sum_array(float *a, int n);
	float mean_array(float *a, int n);
	void mean_arrays(float **a, int n, int els, float *avg);
	void print_statistics(float *a, int n);
	float variance_array(float *a, int n);
	int constrain_int(int a, int min, int max);
	float constrain(float min, float max, float a);
	float dist_array(float *a, float *b, int n, int sub);
	float mse_array(float *a, int n);
	void normalize_array(float *a, int n);
	void translate_array(float *a, int n, float s);
	float mag_array(float *a, int n);
	void scale_array(float *a, int n, float s);
	int sample_array(float *a, int n);
	int max_int_index(int *a, int n);
	int max_index(float *a, int n);
	int int_index(int *a, int val, int n);
	int rand_int(int min, int max);
	float rand_normal();
	size_t rand_size_t();
	float rand_uniform(float min, float max);
	float rand_scale(float s);
	float **one_hot_encode(float *a, int n, int k);



#endif
