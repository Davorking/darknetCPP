#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include <limits.h>
#include <time.h>
#include "darknet.h"
#include "layers.h"


//parser.c
list *read_cfg(char *filename)
{
	FILE *file = fopen(filename, "r");
	if (file == 0) file_error(filename);
	char *line;
	int nu = 0;
	list *options = make_list();
	section *current = 0;
	while ((line = fgetl(file)) != 0) {
		++nu;
		strip(line);
		switch (line[0]) {
		case '[':
			current = (section*)malloc(sizeof(section));
			list_insert(options, current);
			current->options = make_list();
			current->type = line;
			break;
		case '\0':
		case '#':
		case ';':
			free(line);
			break;
		default:
			if (!read_option(line, current->options)) {
				fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
				free(line);
			}
			break;
		}
	}
	fclose(file);
	return options;
}

LAYER_TYPE string_to_layer_type(char * type)
{

	if (strcmp(type, "[shortcut]") == 0) return SHORTCUT;
	if (strcmp(type, "[crop]") == 0) return CROP;
	if (strcmp(type, "[cost]") == 0) return COST;
	if (strcmp(type, "[detection]") == 0) return DETECTION;
	if (strcmp(type, "[region]") == 0) return REGION;
	if (strcmp(type, "[yolo]") == 0) return YOLO;
	if (strcmp(type, "[iseg]") == 0) return ISEG;
	if (strcmp(type, "[local]") == 0) return LOCAL;
	if (strcmp(type, "[conv]") == 0
		|| strcmp(type, "[convolutional]") == 0) return CONVOLUTIONAL;
	if (strcmp(type, "[deconv]") == 0
		|| strcmp(type, "[deconvolutional]") == 0) return DECONVOLUTIONAL;
	if (strcmp(type, "[activation]") == 0) return ACTIVE;
	if (strcmp(type, "[logistic]") == 0) return LOGXENT;
	if (strcmp(type, "[l2norm]") == 0) return L2NORM;
	if (strcmp(type, "[net]") == 0
		|| strcmp(type, "[network]") == 0) return NETWORK;
	if (strcmp(type, "[crnn]") == 0) return CRNN;
	if (strcmp(type, "[gru]") == 0) return GRU;
	if (strcmp(type, "[lstm]") == 0) return LSTM;
	if (strcmp(type, "[rnn]") == 0) return RNN;
	if (strcmp(type, "[conn]") == 0
		|| strcmp(type, "[connected]") == 0) return CONNECTED;
	if (strcmp(type, "[max]") == 0
		|| strcmp(type, "[maxpool]") == 0) return MAXPOOL;
	if (strcmp(type, "[reorg]") == 0) return REORG;
	if (strcmp(type, "[avg]") == 0
		|| strcmp(type, "[avgpool]") == 0) return AVGPOOL;
	if (strcmp(type, "[dropout]") == 0) return DROPOUT;
	if (strcmp(type, "[lrn]") == 0
		|| strcmp(type, "[normalization]") == 0) return NORMALIZATION;
	if (strcmp(type, "[batchnorm]") == 0) return BATCHNORM;
	if (strcmp(type, "[soft]") == 0
		|| strcmp(type, "[softmax]") == 0) return SOFTMAX;
	if (strcmp(type, "[route]") == 0) return ROUTE;
	if (strcmp(type, "[upsample]") == 0) return UPSAMPLE;
	return BLANK;
}

void parse_data(char *data, float *a, int n)
{
	int i;
	if (!data) return;
	char *curr = data;
	char *next = data;
	int done = 0;
	for (i = 0; i < n && !done; ++i) {
		while (*++next != '\0' && *next != ',');
		if (*next == '\0') done = 1;
		*next = '\0';
		sscanf(curr, "%g", &a[i]);
		curr = next + 1;
	}
}

convolutional_layer parse_convolutional(list *options, size_params params)
{
	int n = option_find_int(options, "filters", 1);
	int size = option_find_int(options, "size", 1);
	int stride = option_find_int(options, "stride", 1);
	int pad = option_find_int_quiet(options, "pad", 0);
	int padding = option_find_int_quiet(options, "padding", 0);
	int groups = option_find_int_quiet(options, "groups", 1);
	if (pad) padding = size / 2;

	char *activation_s = (char*)option_find_str(options, "activation", "logistic");
	ACTIVATION activation = get_activation(activation_s);

	int batch, h, w, c;
	h = params.h;
	w = params.w;
	c = params.c;
	batch = params.batch;
	if (!(h && w && c)) error("Layer before convolutional layer must output image.");
	int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
	int binary = option_find_int_quiet(options, "binary", 0);
	int xnor = option_find_int_quiet(options, "xnor", 0);

	convolutional_layer layer = make_convolutional_layer(batch, h, w, c, n, groups, size, stride, padding, activation, batch_normalize, binary, xnor, params.net->adam);
	layer.flipped = option_find_int_quiet(options, "flipped", 0);
	layer.dot = option_find_float_quiet(options, "dot", 0);

	return layer;
}

layer parse_connected(list *options, size_params params)
{
	int output = option_find_int(options, "output", 1);
	char *activation_s = (char*)option_find_str(options, "activation", "logistic");
	ACTIVATION activation = get_activation(activation_s);
	int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

	layer l = make_connected_layer(params.batch, params.inputs, output, activation, batch_normalize, params.net->adam);
	return l;
}

maxpool_layer parse_maxpool(list *options, size_params params)
{
	int stride = option_find_int(options, "stride", 1);
	int size = option_find_int(options, "size", stride);
	int padding = option_find_int_quiet(options, "padding", size - 1);

	int batch, h, w, c;
	h = params.h;
	w = params.w;
	c = params.c;
	batch = params.batch;
	if (!(h && w && c)) error("Layer before maxpool layer must output image.");

	maxpool_layer layer = make_maxpool_layer(batch, h, w, c, size, stride, padding);
	return layer;
}

avgpool_layer parse_avgpool(list *options, size_params params)
{
	int batch, w, h, c;
	w = params.w;
	h = params.h;
	c = params.c;
	batch = params.batch;
	if (!(h && w && c)) error("Layer before avgpool layer must output image.");

	avgpool_layer layer = make_avgpool_layer(batch, w, h, c);
	return layer;
}

learning_rate_policy get_policy(char *s)
{
	if (strcmp(s, "random") == 0) return RANDOM;
	if (strcmp(s, "poly") == 0) return POLY;
	if (strcmp(s, "constant") == 0) return CONSTANT;
	if (strcmp(s, "step") == 0) return STEP;
	if (strcmp(s, "exp") == 0) return EXP;
	if (strcmp(s, "sigmoid") == 0) return SIG;
	if (strcmp(s, "steps") == 0) return STEPS;
	fprintf(stderr, "Couldn't find policy %s, going with constant\n", s);
	return CONSTANT;
}

void parse_net_options(list *options, network *net)
{
	net->batch = option_find_int(options, "batch", 1);
	net->learning_rate = option_find_float(options, "learning_rate", .001);
	net->momentum = option_find_float(options, "momentum", .9);
	net->decay = option_find_float(options, "decay", .0001);
	int subdivs = option_find_int(options, "subdivisions", 1);
	net->time_steps = option_find_int_quiet(options, "time_steps", 1);
	net->notruth = option_find_int_quiet(options, "notruth", 0);
	net->batch /= subdivs;
	net->batch *= net->time_steps;
	net->subdivisions = subdivs;
	net->random = option_find_int_quiet(options, "random", 0);

	net->adam = option_find_int_quiet(options, "adam", 0);
	if (net->adam) {
		net->B1 = option_find_float(options, "B1", .9);
		net->B2 = option_find_float(options, "B2", .999);
		net->eps = option_find_float(options, "eps", .0000001);
	}

	net->h = option_find_int_quiet(options, "height", 0);
	net->w = option_find_int_quiet(options, "width", 0);
	net->c = option_find_int_quiet(options, "channels", 0);
	net->inputs = option_find_int_quiet(options, "inputs", net->h * net->w * net->c);
	net->max_crop = option_find_int_quiet(options, "max_crop", net->w * 2);
	net->min_crop = option_find_int_quiet(options, "min_crop", net->w);
	net->max_ratio = option_find_float_quiet(options, "max_ratio", (float)net->max_crop / net->w);
	net->min_ratio = option_find_float_quiet(options, "min_ratio", (float)net->min_crop / net->w);
	net->center = option_find_int_quiet(options, "center", 0);
	net->clip = option_find_float_quiet(options, "clip", 0);

	net->angle = option_find_float_quiet(options, "angle", 0);
	net->aspect = option_find_float_quiet(options, "aspect", 1);
	net->saturation = option_find_float_quiet(options, "saturation", 1);
	net->exposure = option_find_float_quiet(options, "exposure", 1);
	net->hue = option_find_float_quiet(options, "hue", 0);

	if (!net->inputs && !(net->h && net->w && net->c)) error("No input parameters supplied");

	char *policy_s = (char*)option_find_str(options, "policy", "constant");

	net->policy = get_policy(policy_s);

	net->burn_in = option_find_int_quiet(options, "burn_in", 0);
	net->power = option_find_float_quiet(options, "power", 4);
	if (net->policy == STEP) {
		net->step = option_find_int(options, "step", 1);
		net->scale = option_find_float(options, "scale", 1);
	}
	else if (net->policy == STEPS) {
		char *l = option_find(options, "steps");
		char *p = option_find(options, "scales");
		if (!l || !p) error("STEPS policy must have steps and scales in cfg file");

		int len = strlen(l);
		int n = 1;
		int i;
		for (i = 0; i < len; ++i) {
			if (l[i] == ',') ++n;
		}
		int *steps = (int*)calloc(n, sizeof(int));
		float *scales = (float*)calloc(n, sizeof(float));
		for (i = 0; i < n; ++i) {
			int step = atoi(l);
			float scale = atof(p);
			l = strchr(l, ',') + 1;
			p = strchr(p, ',') + 1;
			steps[i] = step;
			scales[i] = scale;
		}
		net->scales = scales;
		net->steps = steps;
		net->num_steps = n;
	}
	else if (net->policy == EXP) {
		net->gamma = option_find_float(options, "gamma", 1);
	}
	else if (net->policy == SIG) {
		net->step = option_find_int(options, "step", 1);
	}
	else if (net->policy == POLY || net->policy == RANDOM) {
	}
	net->max_batches = option_find_int(options, "max_batches", 0);
}

int is_network(section *s)
{
	return (strcmp(s->type, "[net]") == 0
		|| strcmp(s->type, "[network]") == 0);
}

void free_section(section *s)
{
	free(s->type);
	node *n = s->options->front;
	while (n) {
		kvp *pair = (kvp *)n->val;
		free(pair->key);
		free(pair);
		node *next = n->next;
		free(n);
		n = next;
	}
	free(s->options);
	free(s);
}

network *parse_network_cfg(char *filename)
{
	list *sections = read_cfg(filename);
	node *n = sections->front;
	if (!n) error("Config file has no sections");
	network *net = make_network(sections->size - 1);
	//    net->gpu_index = gpu_index;
	size_params params;



	section *s = (section *)n->val;
	list *options = s->options;
	if (!is_network(s)) error("First section must be [net] or [network]");
	parse_net_options(options, net);

	params.h = net->h;
	params.w = net->w;
	params.c = net->c;
	params.inputs = net->inputs;
	params.batch = net->batch;
	params.time_steps = net->time_steps;
	params.net = net;

	size_t workspace_size = 0;
	n = n->next;
	int count = 0;

	free_section(s);

	fprintf(stderr, "layer     filters    size              input                output\n");
	while (n) {
		params.index = count;
		fprintf(stderr, "%5d ", count);
		s = (section *)n->val;
		options = s->options;
		layer l = {};
		LAYER_TYPE lt = string_to_layer_type(s->type);
		if (lt == CONVOLUTIONAL) {
			l = parse_convolutional(options, params);
		}
		else if (lt == CONNECTED) {
			l = parse_connected(options, params);
		}
		else if (lt == MAXPOOL) {
			l = parse_maxpool(options, params);
		}
		else if (lt == AVGPOOL) {
			l = parse_avgpool(options, params);
			l.output = net->layers[count - 1].output;
			l.delta = net->layers[count - 1].delta;
		}
		else {
			fprintf(stderr, "Type not recognized: %s\n", s->type);
		}
		l.clip = net->clip;
		l.truth = option_find_int_quiet(options, "truth", 0);
		l.onlyforward = option_find_int_quiet(options, "onlyforward", 0);
		l.stopbackward = option_find_int_quiet(options, "stopbackward", 0);
		l.dontsave = option_find_int_quiet(options, "dontsave", 0);
		l.dontload = option_find_int_quiet(options, "dontload", 0);
		l.numload = option_find_int_quiet(options, "numload", 0);
		l.dontloadscales = option_find_int_quiet(options, "dontloadscales", 0);
		l.learning_rate_scale = option_find_float_quiet(options, "learning_rate", 1);
		l.smooth = option_find_float_quiet(options, "smooth", 0);
		option_unused(options);
		net->layers[count] = l;
		if (l.workspace_size > workspace_size) workspace_size = l.workspace_size;
		free_section(s);
		n = n->next;
		++count;
		if (n) {
			params.h = l.out_h;
			params.w = l.out_w;
			params.c = l.out_c;
			params.inputs = l.outputs;
		}
	}
	free_list(sections);
	layer out = get_network_output_layer(net);
	net->outputs = out.outputs;
	net->truths = out.outputs;
	if (net->layers[net->n - 1].truths) net->truths = net->layers[net->n - 1].truths;
	net->output = out.output;
	net->input = (float*)calloc(net->inputs*net->batch, sizeof(float));
	net->truth = (float*)calloc(net->truths*net->batch, sizeof(float));
	if (workspace_size) {
		//printf("%ld\n", workspace_size);

#ifdef GPU
		if (gpu_index >= 0) {
			net->workspace = cuda_make_array(0, (workspace_size - 1) / sizeof(float) + 1);
		}
		else {
			net->workspace = calloc(1, workspace_size);
		}
#else
		net->workspace = (float*)calloc(1, workspace_size);
#endif
	}
	return net;
}

void transpose_matrix(float *a, int rows, int cols)
{
	float *transpose = (float*)calloc(rows*cols, sizeof(float));
	int x, y;
	for (x = 0; x < rows; ++x) {
		for (y = 0; y < cols; ++y) {
			transpose[y*rows + x] = a[x*cols + y];
		}
	}
	memcpy(a, transpose, rows*cols * sizeof(float));
	free(transpose);
}

void load_connected_weights(layer l, FILE *fp, int transpose)
{
	fread(l.biases, sizeof(float), l.outputs, fp);
	fread(l.weights, sizeof(float), l.outputs*l.inputs, fp);
	if (transpose) {
		transpose_matrix(l.weights, l.inputs, l.outputs);
	}
	//printf("Biases: %f mean %f variance\n", mean_array(l.biases, l.outputs), variance_array(l.biases, l.outputs));
	//printf("Weights: %f mean %f variance\n", mean_array(l.weights, l.outputs*l.inputs), variance_array(l.weights, l.outputs*l.inputs));
	if (l.batch_normalize && (!l.dontloadscales)) {
		fread(l.scales, sizeof(float), l.outputs, fp);
		fread(l.rolling_mean, sizeof(float), l.outputs, fp);
		fread(l.rolling_variance, sizeof(float), l.outputs, fp);
		//printf("Scales: %f mean %f variance\n", mean_array(l.scales, l.outputs), variance_array(l.scales, l.outputs));
		//printf("rolling_mean: %f mean %f variance\n", mean_array(l.rolling_mean, l.outputs), variance_array(l.rolling_mean, l.outputs));
		//printf("rolling_variance: %f mean %f variance\n", mean_array(l.rolling_variance, l.outputs), variance_array(l.rolling_variance, l.outputs));
	}
#ifdef GPU
	if (gpu_index >= 0) {
		push_connected_layer(l);
	}
#endif
}

void load_batchnorm_weights(layer l, FILE *fp)
{
	fread(l.scales, sizeof(float), l.c, fp);
	fread(l.rolling_mean, sizeof(float), l.c, fp);
	fread(l.rolling_variance, sizeof(float), l.c, fp);
#ifdef GPU
	if (gpu_index >= 0) {
		push_batchnorm_layer(l);
	}
#endif
}

void load_convolutional_weights_binary(layer l, FILE *fp)
{
	fread(l.biases, sizeof(float), l.n, fp);
	if (l.batch_normalize && (!l.dontloadscales)) {
		fread(l.scales, sizeof(float), l.n, fp);
		fread(l.rolling_mean, sizeof(float), l.n, fp);
		fread(l.rolling_variance, sizeof(float), l.n, fp);
	}
	int size = l.c*l.size*l.size;
	int i, j, k;
	for (i = 0; i < l.n; ++i) {
		float mean = 0;
		fread(&mean, sizeof(float), 1, fp);
		for (j = 0; j < size / 8; ++j) {
			int index = i * size + j * 8;
			unsigned char c = 0;
			fread(&c, sizeof(char), 1, fp);
			for (k = 0; k < 8; ++k) {
				if (j * 8 + k >= size) break;
				l.weights[index + k] = (c & 1 << k) ? mean : -mean;
			}
		}
	}
#ifdef GPU
	if (gpu_index >= 0) {
		push_convolutional_layer(l);
	}
#endif
}

void load_convolutional_weights(layer l, FILE *fp)
{
	if (l.binary) {
		//load_convolutional_weights_binary(l, fp);
		//return;
	}
	if (l.numload) l.n = l.numload;
	int num = l.c / l.groups*l.n*l.size*l.size;
	fread(l.biases, sizeof(float), l.n, fp);
	if (l.batch_normalize && (!l.dontloadscales)) {
		fread(l.scales, sizeof(float), l.n, fp);
		fread(l.rolling_mean, sizeof(float), l.n, fp);
		fread(l.rolling_variance, sizeof(float), l.n, fp);
		if (0) {
			int i;
			for (i = 0; i < l.n; ++i) {
				printf("%g, ", l.rolling_mean[i]);
			}
			printf("\n");
			for (i = 0; i < l.n; ++i) {
				printf("%g, ", l.rolling_variance[i]);
			}
			printf("\n");
		}
		if (0) {
			fill_cpu(l.n, 0, l.rolling_mean, 1);
			fill_cpu(l.n, 0, l.rolling_variance, 1);
		}
		if (0) {
			int i;
			for (i = 0; i < l.n; ++i) {
				printf("%g, ", l.rolling_mean[i]);
			}
			printf("\n");
			for (i = 0; i < l.n; ++i) {
				printf("%g, ", l.rolling_variance[i]);
			}
			printf("\n");
		}
	}
	fread(l.weights, sizeof(float), num, fp);
	//if(l.c == 3) scal_cpu(num, 1./256, l.weights, 1);
	if (l.flipped) {
		transpose_matrix(l.weights, l.c*l.size*l.size, l.n);
	}
	//if (l.binary) binarize_weights(l.weights, l.n, l.c*l.size*l.size, l.weights);
#ifdef GPU
	if (gpu_index >= 0) {
		push_convolutional_layer(l);
	}
#endif
}

void load_weights_upto(network *net, char *filename, int start, int cutoff)
{
#ifdef GPU
	if (net->gpu_index >= 0) {
		cuda_set_device(net->gpu_index);
	}
#endif
	fprintf(stderr, "Loading weights from %s...", filename);
	fflush(stdout);
	FILE *fp = fopen(filename, "rb");
	if (!fp) file_error(filename);

	int major;
	int minor;
	int revision;
	fread(&major, sizeof(int), 1, fp);
	fread(&minor, sizeof(int), 1, fp);
	fread(&revision, sizeof(int), 1, fp);
	if ((major * 10 + minor) >= 2 && major < 1000 && minor < 1000) {
		fread(net->seen, sizeof(size_t), 1, fp);
	}
	else {
		int iseen = 0;
		fread(&iseen, sizeof(int), 1, fp);
		*net->seen = iseen;
	}
	int transpose = (major > 1000) || (minor > 1000);

	int i;
	for (i = start; i < net->n && i < cutoff; ++i) {
		layer l = net->layers[i];
		if (l.dontload) continue;
		if (l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL) {
			load_convolutional_weights(l, fp);
		}
		if (l.type == CONNECTED) {
			load_connected_weights(l, fp, transpose);
		}
		if (l.type == BATCHNORM) {
			load_batchnorm_weights(l, fp);
		}
		if (l.type == CRNN) {
			load_convolutional_weights(*(l.input_layer), fp);
			load_convolutional_weights(*(l.self_layer), fp);
			load_convolutional_weights(*(l.output_layer), fp);
		}
		if (l.type == RNN) {
			load_connected_weights(*(l.input_layer), fp, transpose);
			load_connected_weights(*(l.self_layer), fp, transpose);
			load_connected_weights(*(l.output_layer), fp, transpose);
		}
		if (l.type == LSTM) {
			load_connected_weights(*(l.wi), fp, transpose);
			load_connected_weights(*(l.wf), fp, transpose);
			load_connected_weights(*(l.wo), fp, transpose);
			load_connected_weights(*(l.wg), fp, transpose);
			load_connected_weights(*(l.ui), fp, transpose);
			load_connected_weights(*(l.uf), fp, transpose);
			load_connected_weights(*(l.uo), fp, transpose);
			load_connected_weights(*(l.ug), fp, transpose);
		}
		if (l.type == GRU) {
			if (1) {
				load_connected_weights(*(l.wz), fp, transpose);
				load_connected_weights(*(l.wr), fp, transpose);
				load_connected_weights(*(l.wh), fp, transpose);
				load_connected_weights(*(l.uz), fp, transpose);
				load_connected_weights(*(l.ur), fp, transpose);
				load_connected_weights(*(l.uh), fp, transpose);
			}
			else {
				load_connected_weights(*(l.reset_layer), fp, transpose);
				load_connected_weights(*(l.update_layer), fp, transpose);
				load_connected_weights(*(l.state_layer), fp, transpose);
			}
		}
		if (l.type == LOCAL) {
			int locations = l.out_w*l.out_h;
			int size = l.size*l.size*l.c*l.n*locations;
			fread(l.biases, sizeof(float), l.outputs, fp);
			fread(l.weights, sizeof(float), size, fp);
#ifdef GPU
			if (gpu_index >= 0) {
				push_local_layer(l);
			}
#endif
		}
	}
	fprintf(stderr, "Done!\n");
	fclose(fp);
}

void load_weights(network *net, char *filename)
{
	load_weights_upto(net, filename, 0, net->n);
}






//network.c
load_args get_base_args(network *net)
{
	load_args args = { 0 };
	args.w = net->w;
	args.h = net->h;
	args.size = net->w;

	args.min = net->min_crop;
	args.max = net->max_crop;
	args.angle = net->angle;
	args.aspect = net->aspect;
	args.exposure = net->exposure;
	args.center = net->center;
	args.saturation = net->saturation;
	args.hue = net->hue;
	return args;
}

network *load_network(char *cfg, char *weights, int clear)
{
	network *net = parse_network_cfg(cfg);
	if (weights && weights[0] != 0) {
		load_weights(net, weights);
	}
	if (clear) (*net->seen) = 0;
	return net;
}

size_t get_current_batch(network *net)
{
	size_t batch_num = (*net->seen) / (net->batch*net->subdivisions);
	return batch_num;
}

void reset_network_state(network *net, int b)
{
	int i;
	for (i = 0; i < net->n; ++i) {
#ifdef GPU
		layer l = net->layers[i];
		if (l.state_gpu) {
			fill_gpu(l.outputs, 0, l.state_gpu + l.outputs*b, 1);
		}
		if (l.h_gpu) {
			fill_gpu(l.outputs, 0, l.h_gpu + l.outputs*b, 1);
		}
#endif
	}
}

void reset_rnn(network *net)
{
	reset_network_state(net, 0);
}

float get_current_rate(network *net)
{
	size_t batch_num = get_current_batch(net);
	int i;
	float rate;
	if (batch_num < net->burn_in) return net->learning_rate * pow((float)batch_num / net->burn_in, net->power);
	switch (net->policy) {
	case CONSTANT:
		return net->learning_rate;
	case STEP:
		return net->learning_rate * pow(net->scale, batch_num / net->step);
	case STEPS:
		rate = net->learning_rate;
		for (i = 0; i < net->num_steps; ++i) {
			if (net->steps[i] > batch_num) return rate;
			rate *= net->scales[i];
		}
		return rate;
	case EXP:
		return net->learning_rate * pow(net->gamma, batch_num);
	case POLY:
		return net->learning_rate * pow(1 - (float)batch_num / net->max_batches, net->power);
	case RANDOM:
		return net->learning_rate * pow(rand_uniform(0, 1), net->power);
	case SIG:
		return net->learning_rate * (1. / (1. + exp(net->gamma*(batch_num - net->step))));
	default:
		fprintf(stderr, "Policy is weird!\n");
		return net->learning_rate;
	}
}

const char* get_layer_string(LAYER_TYPE a)
{
	switch (a) {
	case CONVOLUTIONAL:
		return "convolutional";
	case ACTIVE:
		return "activation";
	case LOCAL:
		return "local";
	case DECONVOLUTIONAL:
		return "deconvolutional";
	case CONNECTED:
		return "connected";
	case RNN:
		return "rnn";
	case GRU:
		return "gru";
	case LSTM:
		return "lstm";
	case CRNN:
		return "crnn";
	case MAXPOOL:
		return "maxpool";
	case REORG:
		return "reorg";
	case AVGPOOL:
		return "avgpool";
	case SOFTMAX:
		return "softmax";
	case DETECTION:
		return "detection";
	case REGION:
		return "region";
	case YOLO:
		return "yolo";
	case DROPOUT:
		return "dropout";
	case CROP:
		return "crop";
	case COST:
		return "cost";
	case ROUTE:
		return "route";
	case SHORTCUT:
		return "shortcut";
	case NORMALIZATION:
		return "normalization";
	case BATCHNORM:
		return "batchnorm";
	default:
		break;
	}
	return "none";
}

network *make_network(int n)
{
	network *net = (network*)calloc(1, sizeof(network));
	net->n = n;
	net->layers = (layer*)calloc(net->n, sizeof(layer));
	net->seen = (size_t*)calloc(1, sizeof(size_t));
	net->t = (int*)calloc(1, sizeof(int));
	net->cost = (float*)calloc(1, sizeof(float));
	return net;
}

void update_network(network *netp)
{
	network net = *netp;
	int i;
	update_args a = { 0 };
	a.batch = net.batch*net.subdivisions;
	a.learning_rate = get_current_rate(netp);
	a.momentum = net.momentum;
	a.decay = net.decay;
	a.adam = net.adam;
	a.B1 = net.B1;
	a.B2 = net.B2;
	a.eps = net.eps;
	++*net.t;
	a.t = *net.t;

	for (i = 0; i < net.n; ++i) {
		layer l = net.layers[i];
		if (l.update) {
			l.update(l, a);
		}
	}
}

void calc_network_cost(network *netp)
{
	network net = *netp;
	int i;
	float sum = 0;
	int count = 0;
	for (i = 0; i < net.n; ++i) {
		if (net.layers[i].cost) {
			sum += net.layers[i].cost[0];
			++count;
		}
	}
	*net.cost = sum / count;
}

int get_predicted_class_network(network *net)
{
	return max_index(net->output, net->outputs);
}

void set_temp_network(network *net, float t)
{
	int i;
	for (i = 0; i < net->n; ++i) {
		net->layers[i].temperature = t;
	}
}

void set_batch_network(network *net, int b)
{
	net->batch = b;
	int i;
	for (i = 0; i < net->n; ++i) {
		net->layers[i].batch = b;
	}
}

layer get_network_detection_layer(network *net)
{
	int i;
	for (i = 0; i < net->n; ++i) {
		if (net->layers[i].type == DETECTION) {
			return net->layers[i];
		}
	}
	fprintf(stderr, "Detection layer not found!!\n");
	layer l = {};
	return l;
}

void top_predictions(network *net, int k, int *index)
{
	top_k(net->output, net->outputs, k, index);
}

void free_detections(detection *dets, int n)
{
	int i;
	for (i = 0; i < n; ++i) {
		free(dets[i].prob);
		if (dets[i].mask) free(dets[i].mask);
	}
	free(dets);
}

int network_width(network *net) { return net->w; }
int network_height(network *net) { return net->h; }

void print_network(network *net)
{
	int i, j;
	for (i = 0; i < net->n; ++i) {
		layer l = net->layers[i];
		float *output = l.output;
		int n = l.outputs;
		float mean = mean_array(output, n);
		float vari = variance_array(output, n);
		fprintf(stderr, "Layer %d - Mean: %f, Variance: %f\n", i, mean, vari);
		if (n > 100) n = 100;
		for (j = 0; j < n; ++j) fprintf(stderr, "%f, ", output[j]);
		if (n == 100)fprintf(stderr, ".....\n");
		fprintf(stderr, "\n");
	}
}

layer get_network_output_layer(network *net)
{
	int i;
	for (i = net->n - 1; i >= 0; --i) {
		if (net->layers[i].type != COST) break;
	}
	return net->layers[i];
}

layer network_output_layer(network *net)
{
	int i;
	for (i = net->n - 1; i >= 0; --i) {
		if (net->layers[i].type != COST) break;
	}
	return net->layers[i];
}

int network_inputs(network *net)
{
	return net->layers[0].inputs;
}

int network_outputs(network *net)
{
	return network_output_layer(net).outputs;
}

float *network_output(network *net)
{
	return network_output_layer(net).output;
}












//utils.c
int *read_intlist(char *gpu_list, int *ngpus, int d)
{
	int *gpus = 0;
	if (gpu_list) {
		int len = strlen(gpu_list);
		*ngpus = 1;
		int i;
		for (i = 0; i < len; ++i) {
			if (gpu_list[i] == ',')++*ngpus;
		}
		gpus = (int*)calloc(*ngpus, sizeof(int));
		for (i = 0; i < *ngpus; ++i) {
			gpus[i] = atoi(gpu_list);
			gpu_list = strchr(gpu_list, ',') + 1;
		}
	}
	else {
		gpus = (int*)calloc(1, sizeof(int));
		*gpus = d;
		*ngpus = 1;
	}
	return gpus;
}

int *read_map(char *filename)
{
	int n = 0;
	int *map = 0;
	char *str;
	FILE *file = fopen(filename, "r");
	if (!file) file_error(filename);
	while ((str = fgetl(file))) {
		++n;
		map = (int*)realloc(map, n * sizeof(int));
		map[n - 1] = atoi(str);
	}
	return map;
}

int *random_index_order(int min, int max)
{
	int *inds = (int*)calloc(max - min, sizeof(int));
	int i;
	for (i = min; i < max; ++i) {
		inds[i] = i;
	}
	for (i = min; i < max - 1; ++i) {
		int swap = inds[i];
		int index = i + rand() % (max - i);
		inds[i] = inds[index];
		inds[index] = swap;
	}
	return inds;
}

void del_arg(int argc, char **argv, int index)
{
	int i;
	for (i = index; i < argc - 1; ++i) argv[i] = argv[i + 1];
	argv[i] = 0;
}

int find_arg(int argc, char* argv[], char *arg)
{
	int i;
	for (i = 0; i < argc; ++i) {
		if (!argv[i]) continue;
		if (0 == strcmp(argv[i], arg)) {
			del_arg(argc, argv, i);
			return 1;
		}
	}
	return 0;
}

int find_int_arg(int argc, char **argv, char *arg, int def)
{
	int i;
	for (i = 0; i < argc - 1; ++i) {
		if (!argv[i]) continue;
		if (0 == strcmp(argv[i], arg)) {
			def = atoi(argv[i + 1]);
			del_arg(argc, argv, i);
			del_arg(argc, argv, i);
			break;
		}
	}
	return def;
}

float find_float_arg(int argc, char **argv, char *arg, float def)
{
	int i;
	for (i = 0; i < argc - 1; ++i) {
		if (!argv[i]) continue;
		if (0 == strcmp(argv[i], arg)) {
			def = atof(argv[i + 1]);
			del_arg(argc, argv, i);
			del_arg(argc, argv, i);
			break;
		}
	}
	return def;
}

char *find_char_arg(int argc, char **argv, char *arg, char *def)
{
	int i;
	for (i = 0; i < argc - 1; ++i) {
		if (!argv[i]) continue;
		if (0 == strcmp(argv[i], arg)) {
			def = argv[i + 1];
			del_arg(argc, argv, i);
			del_arg(argc, argv, i);
			break;
		}
	}
	return def;
}

char *basecfg(char *cfgfile)
{
	char *c = cfgfile;
	char *next;
	while ((next = strchr(c, '/')))
	{
		c = next + 1;
	}
	c = copy_string(c);
	next = strchr(c, '.');
	if (next) *next = 0;
	return c;
}

int alphanum_to_int(char c)
{
	return (c < 58) ? c - 48 : c - 87;
}
char int_to_alphanum(int i)
{
	if (i == 36) return '.';
	return (i < 10) ? i + 48 : i + 87;
}

void pm(int M, int N, float *A)
{
	int i, j;
	for (i = 0; i < M; ++i) {
		printf("%d ", i + 1);
		for (j = 0; j < N; ++j) {
			printf("%2.4f, ", A[i*N + j]);
		}
		printf("\n");
	}
	printf("\n");
}

void find_replace(char *str, char *orig, char *rep, char *output)
{
	char buffer[4096] = { 0 };
	char *p;

	sprintf(buffer, "%s", str);
	if (!(p = strstr(buffer, orig))) {  // Is 'orig' even in 'str'?
		sprintf(output, "%s", str);
		return;
	}

	*p = '\0';

	sprintf(output, "%s%s%s", buffer, rep, p + strlen(orig));
}

float sec(clock_t clocks)
{
	return (float)clocks / CLOCKS_PER_SEC;
}

void top_k(float *a, int n, int k, int *index)
{
	int i, j;
	for (j = 0; j < k; ++j) index[j] = -1;
	for (i = 0; i < n; ++i) {
		int curr = i;
		for (j = 0; j < k; ++j) {
			if ((index[j] < 0) || a[curr] > a[index[j]]) {
				int swap = curr;
				curr = index[j];
				index[j] = swap;
			}
		}
	}
}

void error(const char *s)
{
	perror(s);
	assert(0);
	exit(-1);
}

unsigned char *read_file(char *filename)
{
	FILE *fp = fopen(filename, "rb");
	size_t size;

	fseek(fp, 0, SEEK_END);
	size = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	unsigned char *text = (unsigned char*)calloc(size + 1, sizeof(char));
	fread(text, 1, size, fp);
	fclose(fp);
	return text;
}

void malloc_error()
{
	fprintf(stderr, "Malloc error\n");
	exit(-1);
}

void file_error(char *s)
{
	fprintf(stderr, "Couldn't open file: %s\n", s);
	exit(0);
}

list *split_str(char *s, char delim)
{
	size_t i;
	size_t len = strlen(s);
	list *l = make_list();
	list_insert(l, s);
	for (i = 0; i < len; ++i) {
		if (s[i] == delim) {
			s[i] = '\0';
			list_insert(l, &(s[i + 1]));
		}
	}
	return l;
}

void strip(char *s)
{
	size_t i;
	size_t len = strlen(s);
	size_t offset = 0;
	for (i = 0; i < len; ++i) {
		char c = s[i];
		if (c == ' ' || c == '\t' || c == '\n') ++offset;
		else s[i - offset] = c;
	}
	s[len - offset] = '\0';
}

void strip_char(char *s, char bad)
{
	size_t i;
	size_t len = strlen(s);
	size_t offset = 0;
	for (i = 0; i < len; ++i) {
		char c = s[i];
		if (c == bad) ++offset;
		else s[i - offset] = c;
	}
	s[len - offset] = '\0';
}

void free_ptrs(void **ptrs, int n)
{
	int i;
	for (i = 0; i < n; ++i) free(ptrs[i]);
	free(ptrs);
}

char *fgetl(FILE *fp)
{
	if (feof(fp)) return 0;
	size_t size = 512;
	char *line = (char*)malloc(size * sizeof(char));
	if (!fgets(line, size, fp)) {
		free(line);
		return 0;
	}

	size_t curr = strlen(line);

	while ((line[curr - 1] != '\n') && !feof(fp)) {
		if (curr == size - 1) {
			size *= 2;
			line = (char*)realloc(line, size * sizeof(char));
			if (!line) {
				printf("%ld\n", size);
				malloc_error();
			}
		}
		size_t readsize = size - curr;
		if (readsize > INT_MAX) readsize = INT_MAX - 1;
		fgets(&line[curr], readsize, fp);
		curr = strlen(line);
	}
	if (line[curr - 1] == '\n') line[curr - 1] = '\0';

	return line;
}

char *copy_string(char *s)
{
	char *copy = (char*)malloc(strlen(s) + 1);
	strncpy(copy, s, strlen(s) + 1);
	return copy;
}

list *parse_csv_line(char *line)
{
	list *l = make_list();
	char *c, *p;
	int in = 0;
	for (c = line, p = line; *c != '\0'; ++c) {
		if (*c == '"') in = !in;
		else if (*c == ',' && !in) {
			*c = '\0';
			list_insert(l, copy_string(p));
			p = c + 1;
		}
	}
	list_insert(l, copy_string(p));
	return l;
}

int count_fields(char *line)
{
	int count = 0;
	int done = 0;
	char *c;
	for (c = line; !done; ++c) {
		done = (*c == '\0');
		if (*c == ',' || done) ++count;
	}
	return count;
}

float *parse_fields(char *line, int n)
{
	float *field = (float*)calloc(n, sizeof(float));
	char *c, *p, *end;
	int count = 0;
	int done = 0;
	for (c = line, p = line; !done; ++c) {
		done = (*c == '\0');
		if (*c == ',' || done) {
			*c = '\0';
			field[count] = strtod(p, &end);
			if (p == c) field[count] = nan("");
			if (end != c && (end != c - 1 || *end != '\r')) field[count] = nan(""); //DOS file formats!
			p = c + 1;
			++count;
		}
	}
	return field;
}

float sum_array(float *a, int n)
{
	int i;
	float sum = 0;
	for (i = 0; i < n; ++i) sum += a[i];
	return sum;
}

float mean_array(float *a, int n)
{
	return sum_array(a, n) / n;
}

void mean_arrays(float **a, int n, int els, float *avg)
{
	int i;
	int j;
	memset(avg, 0, els * sizeof(float));
	for (j = 0; j < n; ++j) {
		for (i = 0; i < els; ++i) {
			avg[i] += a[j][i];
		}
	}
	for (i = 0; i < els; ++i) {
		avg[i] /= n;
	}
}

void print_statistics(float *a, int n)
{
	float m = mean_array(a, n);
	float v = variance_array(a, n);
	printf("MSE: %.6f, Mean: %.6f, Variance: %.6f\n", mse_array(a, n), m, v);
}

float variance_array(float *a, int n)
{
	int i;
	float sum = 0;
	float mean = mean_array(a, n);
	for (i = 0; i < n; ++i) sum += (a[i] - mean)*(a[i] - mean);
	float variance = sum / n;
	return variance;
}

int constrain_int(int a, int min, int max)
{
	if (a < min) return min;
	if (a > max) return max;
	return a;
}

float constrain(float min, float max, float a)
{
	if (a < min) return min;
	if (a > max) return max;
	return a;
}

float dist_array(float *a, float *b, int n, int sub)
{
	int i;
	float sum = 0;
	for (i = 0; i < n; i += sub) sum += pow(a[i] - b[i], 2);
	return sqrt(sum);
}

float mse_array(float *a, int n)
{
	int i;
	float sum = 0;
	for (i = 0; i < n; ++i) sum += a[i] * a[i];
	return sqrt(sum / n);
}

void normalize_array(float *a, int n)
{
	int i;
	float mu = mean_array(a, n);
	float sigma = sqrt(variance_array(a, n));
	for (i = 0; i < n; ++i) {
		a[i] = (a[i] - mu) / sigma;
	}
	mu = mean_array(a, n);
	sigma = sqrt(variance_array(a, n));
}

void translate_array(float *a, int n, float s)
{
	int i;
	for (i = 0; i < n; ++i) {
		a[i] += s;
	}
}

float mag_array(float *a, int n)
{
	int i;
	float sum = 0;
	for (i = 0; i < n; ++i) {
		sum += a[i] * a[i];
	}
	return sqrt(sum);
}

void scale_array(float *a, int n, float s)
{
	int i;
	for (i = 0; i < n; ++i) {
		a[i] *= s;
	}
}

int sample_array(float *a, int n)
{
	float sum = sum_array(a, n);
	scale_array(a, n, 1. / sum);
	float r = rand_uniform(0, 1);
	int i;
	for (i = 0; i < n; ++i) {
		r = r - a[i];
		if (r <= 0) return i;
	}
	return n - 1;
}

int max_int_index(int *a, int n)
{
	if (n <= 0) return -1;
	int i, max_i = 0;
	int max = a[0];
	for (i = 1; i < n; ++i) {
		if (a[i] > max) {
			max = a[i];
			max_i = i;
		}
	}
	return max_i;
}

int max_index(float *a, int n)
{
	if (n <= 0) return -1;
	int i, max_i = 0;
	float max = a[0];
	for (i = 1; i < n; ++i) {
		if (a[i] > max) {
			max = a[i];
			max_i = i;
		}
	}
	return max_i;
}

int int_index(int *a, int val, int n)
{
	int i;
	for (i = 0; i < n; ++i) {
		if (a[i] == val) return i;
	}
	return -1;
}

int rand_int(int min, int max)
{
	if (max < min) {
		int s = min;
		min = max;
		max = s;
	}
	int r = (rand() % (max - min + 1)) + min;
	return r;
}

// From http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
float rand_normal()
{
	static int haveSpare = 0;
	static double rand1, rand2;

	if (haveSpare)
	{
		haveSpare = 0;
		return sqrt(rand1) * sin(rand2);
	}

	haveSpare = 1;

	rand1 = rand() / ((double)RAND_MAX);
	if (rand1 < 1e-100) rand1 = 1e-100;
	rand1 = -2 * log(rand1);
	rand2 = (rand() / ((double)RAND_MAX)) * TWO_PI;

	return sqrt(rand1) * cos(rand2);
}

/*
   float rand_normal()
   {
   int n = 12;
   int i;
   float sum= 0;
   for(i = 0; i < n; ++i) sum += (float)rand()/RAND_MAX;
   return sum-n/2.;
   }
 */

size_t rand_size_t()
{
	return  ((size_t)(rand() & 0xff) << 56) |
		((size_t)(rand() & 0xff) << 48) |
		((size_t)(rand() & 0xff) << 40) |
		((size_t)(rand() & 0xff) << 32) |
		((size_t)(rand() & 0xff) << 24) |
		((size_t)(rand() & 0xff) << 16) |
		((size_t)(rand() & 0xff) << 8) |
		((size_t)(rand() & 0xff) << 0);
}

float rand_uniform(float min, float max)
{
	if (max < min) {
		float swap = min;
		min = max;
		max = swap;
	}
	return ((float)rand() / RAND_MAX * (max - min)) + min;
}

float rand_scale(float s)
{
	float scale = rand_uniform(1, s);
	if (rand() % 2) return scale;
	return 1. / scale;
}

float **one_hot_encode(float *a, int n, int k)
{
	int i;
	float **t = (float**)calloc(n, sizeof(float*));
	for (i = 0; i < n; ++i) {
		t[i] = (float*)calloc(k, sizeof(float));
		int index = (int)a[i];
		t[i][index] = 1;
	}
	return t;
}





//option_list.c
list *read_data_cfg(char *filename)
{
	FILE *file = fopen(filename, "r");
	if (file == 0) file_error(filename);
	char *line;
	int nu = 0;
	list *options = make_list();
	while ((line = fgetl(file)) != 0) {
		++nu;
		strip(line);
		switch (line[0]) {
		case '\0':
		case '#':
		case ';':
			free(line);
			break;
		default:
			if (!read_option(line, options)) {
				fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
				free(line);
			}
			break;
		}
	}
	fclose(file);
	return options;
}

int read_option(char *s, list *options)
{
	size_t i;
	size_t len = strlen(s);
	char *val = 0;
	for (i = 0; i < len; ++i) {
		if (s[i] == '=') {
			s[i] = '\0';
			val = s + i + 1;
			break;
		}
	}
	if (i == len - 1) return 0;
	char *key = s;
	option_insert(options, key, val);
	return 1;
}

void option_insert(list *l, const char *key, const char *val)
{
	kvp *p = (kvp*)malloc(sizeof(kvp));
	p->key = (char*)key;
	p->val = (char*)val;
	p->used = 0;
	list_insert(l, p);
}

void option_unused(list *l)
{
	node *n = l->front;
	while (n) {
		kvp *p = (kvp *)n->val;
		if (!p->used) {
			fprintf(stderr, "Unused field: '%s = %s'\n", p->key, p->val);
		}
		n = n->next;
	}
}

char *option_find(list *l, const char *key)
{
	node *n = l->front;
	while (n) {
		kvp *p = (kvp *)n->val;
		if (strcmp(p->key, key) == 0) {
			p->used = 1;
			return p->val;
		}
		n = n->next;
	}
	return 0;
}

const char *option_find_str(list *l, const char *key, const char *def)
{
	char *v = option_find(l, key);
	if (v) return v;
	if (def) fprintf(stderr, "%s: Using default '%s'\n", key, def);
	return def;
}

int option_find_int(list *l, const char *key, int def)
{
	char *v = option_find(l, key);
	if (v) return atoi(v);
	fprintf(stderr, "%s: Using default '%d'\n", key, def);
	return def;
}

int option_find_int_quiet(list *l, const char *key, int def)
{
	char *v = option_find(l, key);
	if (v) return atoi(v);
	return def;
}

float option_find_float(list *l, const char *key, float def)
{
	char *v = option_find(l, key);
	if (v) return atof(v);
	fprintf(stderr, "%s: Using default '%lf'\n", key, def);
	return def;
}

float option_find_float_quiet(list *l, const char *key, float def)
{
	char *v = option_find(l, key);
	if (v) return atof(v);
	return def;
}







//list.c
list *make_list()
{
	list *l = (list*)malloc(sizeof(list));
	l->size = 0;
	l->front = 0;
	l->back = 0;
	return l;
}

/*
void transfer_node(list *s, list *d, node *n)
{
	node *prev, *next;
	prev = n->prev;
	next = n->next;
	if(prev) prev->next = next;
	if(next) next->prev = prev;
	--s->size;
	if(s->front == n) s->front = next;
	if(s->back == n) s->back = prev;
}
*/

void *list_pop(list *l) {
	if (!l->back) return 0;
	node *b = l->back;
	void *val = b->val;
	l->back = b->prev;
	if (l->back) l->back->next = 0;
	free(b);
	--l->size;

	return val;
}

void list_insert(list *l, void *val)
{
	node *t_new = (node*)malloc(sizeof(node));
	t_new->val = val;
	t_new->next = 0;

	if (!l->back) {
		l->front = t_new;
		t_new->prev = 0;
	}
	else {
		l->back->next = t_new;
		t_new->prev = l->back;
	}
	l->back = t_new;
	++l->size;
}

void free_node(node *n)
{
	node *next;
	while (n) {
		next = n->next;
		free(n);
		n = next;
	}
}

void free_list(list *l)
{
	free_node(l->front);
	free(l);
}

void free_list_contents(list *l)
{
	node *n = l->front;
	while (n) {
		free(n->val);
		n = n->next;
	}
}

void **list_to_array(list *l)
{
	void **a = (void**)calloc(l->size, sizeof(void*));
	int count = 0;
	node *n = l->front;
	while (n) {
		a[count++] = n->val;
		n = n->next;
	}
	return a;
}









//blas.c
void reorg_cpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out)
{
	int b, i, j, k;
	int out_c = c / (stride*stride);

	for (b = 0; b < batch; ++b) {
		for (k = 0; k < c; ++k) {
			for (j = 0; j < h; ++j) {
				for (i = 0; i < w; ++i) {
					int in_index = i + w * (j + h * (k + c * b));
					int c2 = k % out_c;
					int offset = k / out_c;
					int w2 = i * stride + offset % stride;
					int h2 = j * stride + offset / stride;
					int out_index = w2 + w * stride*(h2 + h * stride*(c2 + out_c * b));
					if (forward) out[out_index] = x[in_index];
					else out[in_index] = x[out_index];
				}
			}
		}
	}
}

void flatten(float *x, int size, int layers, int batch, int forward)
{
	float *swap = (float*)calloc(size*layers*batch, sizeof(float));
	int i, c, b;
	for (b = 0; b < batch; ++b) {
		for (c = 0; c < layers; ++c) {
			for (i = 0; i < size; ++i) {
				int i1 = b * layers*size + c * size + i;
				int i2 = b * layers*size + i * layers + c;
				if (forward) swap[i2] = x[i1];
				else swap[i1] = x[i2];
			}
		}
	}
	memcpy(x, swap, size*layers*batch * sizeof(float));
	free(swap);
}

void weighted_sum_cpu(float *a, float *b, float *s, int n, float *c)
{
	int i;
	for (i = 0; i < n; ++i) {
		c[i] = s[i] * a[i] + (1 - s[i])*(b ? b[i] : 0);
	}
}

void weighted_delta_cpu(float *a, float *b, float *s, float *da, float *db, float *ds, int n, float *dc)
{
	int i;
	for (i = 0; i < n; ++i) {
		if (da) da[i] += dc[i] * s[i];
		if (db) db[i] += dc[i] * (1 - s[i]);
		ds[i] += dc[i] * (a[i] - b[i]);
	}
}

void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out)
{
	int stride = w1 / w2;
	int sample = w2 / w1;
	assert(stride == h1 / h2);
	assert(sample == h2 / h1);
	if (stride < 1) stride = 1;
	if (sample < 1) sample = 1;
	int minw = (w1 < w2) ? w1 : w2;
	int minh = (h1 < h2) ? h1 : h2;
	int minc = (c1 < c2) ? c1 : c2;

	int i, j, k, b;
	for (b = 0; b < batch; ++b) {
		for (k = 0; k < minc; ++k) {
			for (j = 0; j < minh; ++j) {
				for (i = 0; i < minw; ++i) {
					int out_index = i * sample + w2 * (j*sample + h2 * (k + c2 * b));
					int add_index = i * stride + w1 * (j*stride + h1 * (k + c1 * b));
					out[out_index] = s1 * out[out_index] + s2 * add[add_index];
				}
			}
		}
	}
}

void mean_cpu(float *x, int batch, int filters, int spatial, float *mean)
{
	float scale = 1. / (batch * spatial);
	int i, j, k;
	for (i = 0; i < filters; ++i) {
		mean[i] = 0;
		for (j = 0; j < batch; ++j) {
			for (k = 0; k < spatial; ++k) {
				int index = j * filters*spatial + i * spatial + k;
				mean[i] += x[index];
			}
		}
		mean[i] *= scale;
	}
}

void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
	float scale = 1. / (batch * spatial - 1);
	int i, j, k;
	for (i = 0; i < filters; ++i) {
		variance[i] = 0;
		for (j = 0; j < batch; ++j) {
			for (k = 0; k < spatial; ++k) {
				int index = j * filters*spatial + i * spatial + k;
				variance[i] += pow((x[index] - mean[i]), 2);
			}
		}
		variance[i] *= scale;
	}
}

void l2normalize_cpu(float *x, float *dx, int batch, int filters, int spatial)
{
	int b, f, i;
	for (b = 0; b < batch; ++b) {
		for (i = 0; i < spatial; ++i) {
			float sum = 0;
			for (f = 0; f < filters; ++f) {
				int index = b * filters*spatial + f * spatial + i;
				sum += powf(x[index], 2);
			}
			sum = sqrtf(sum);
			for (f = 0; f < filters; ++f) {
				int index = b * filters*spatial + f * spatial + i;
				x[index] /= sum;
				dx[index] = (1 - x[index]) / sum;
			}
		}
	}
}

void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
	int b, f, i;
	for (b = 0; b < batch; ++b) {
		for (f = 0; f < filters; ++f) {
			for (i = 0; i < spatial; ++i) {
				int index = b * filters*spatial + f * spatial + i;
				x[index] = (x[index] - mean[f]) / (sqrt(variance[f]) + .000001f);
			}
		}
	}
}

void const_cpu(int N, float ALPHA, float *X, int INCX)
{
	int i;
	for (i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

void mul_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
	int i;
	for (i = 0; i < N; ++i) Y[i*INCY] *= X[i*INCX];
}

void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
	int i;
	for (i = 0; i < N; ++i) Y[i*INCY] = pow(X[i*INCX], ALPHA);
}

void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
	int i;
	for (i = 0; i < N; ++i) Y[i*INCY] += ALPHA * X[i*INCX];
}

void scal_cpu(int N, float ALPHA, float *X, int INCX)
{
	int i;
	for (i = 0; i < N; ++i) X[i*INCX] *= ALPHA;
}

void fill_cpu(int N, float ALPHA, float *X, int INCX)
{
	int i;
	for (i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

void deinter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
	int i, j;
	int index = 0;
	for (j = 0; j < B; ++j) {
		for (i = 0; i < NX; ++i) {
			if (X) X[j*NX + i] += OUT[index];
			++index;
		}
		for (i = 0; i < NY; ++i) {
			if (Y) Y[j*NY + i] += OUT[index];
			++index;
		}
	}
}

void inter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
	int i, j;
	int index = 0;
	for (j = 0; j < B; ++j) {
		for (i = 0; i < NX; ++i) {
			OUT[index++] = X[j*NX + i];
		}
		for (i = 0; i < NY; ++i) {
			OUT[index++] = Y[j*NY + i];
		}
	}
}

void copy_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
	int i;
	for (i = 0; i < N; ++i) Y[i*INCY] = X[i*INCX];
}

void mult_add_into_cpu(int N, float *X, float *Y, float *Z)
{
	int i;
	for (i = 0; i < N; ++i) Z[i] += X[i] * Y[i];
}

void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
	int i;
	for (i = 0; i < n; ++i) {
		float diff = truth[i] - pred[i];
		float abs_val = fabs(diff);
		if (abs_val < 1) {
			error[i] = diff * diff;
			delta[i] = diff;
		}
		else {
			error[i] = 2 * abs_val - 1;
			delta[i] = (diff < 0) ? 1 : -1;
		}
	}
}

void l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
	int i;
	for (i = 0; i < n; ++i) {
		float diff = truth[i] - pred[i];
		error[i] = fabs(diff);
		delta[i] = diff > 0 ? 1 : -1;
	}
}

void softmax_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
	int i;
	for (i = 0; i < n; ++i) {
		float t = truth[i];
		float p = pred[i];
		error[i] = (t) ? -log(p) : 0;
		delta[i] = t - p;
	}
}

void logistic_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
	int i;
	for (i = 0; i < n; ++i) {
		float t = truth[i];
		float p = pred[i];
		error[i] = -t * log(p) - (1 - t)*log(1 - p);
		delta[i] = t - p;
	}
}

void l2_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
	int i;
	for (i = 0; i < n; ++i) {
		float diff = truth[i] - pred[i];
		error[i] = diff * diff;
		delta[i] = diff;
	}
}

float dot_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
	int i;
	float dot = 0;
	for (i = 0; i < N; ++i) dot += X[i*INCX] * Y[i*INCY];
	return dot;
}

void softmax(float *input, int n, float temp, int stride, float *output)
{
	int i;
	float sum = 0;
	float largest = -FLT_MAX;
	for (i = 0; i < n; ++i) {
		if (input[i*stride] > largest) largest = input[i*stride];
	}
	for (i = 0; i < n; ++i) {
		float e = exp(input[i*stride] / temp - largest / temp);
		sum += e;
		output[i*stride] = e;
	}
	for (i = 0; i < n; ++i) {
		output[i*stride] /= sum;
	}
}

void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
	int g, b;
	for (b = 0; b < batch; ++b) {
		for (g = 0; g < groups; ++g) {
			softmax(input + b * batch_offset + g * group_offset, n, temp, stride, output + b * batch_offset + g * group_offset);
		}
	}
}

void upsample_cpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{
	int i, j, k, b;
	for (b = 0; b < batch; ++b) {
		for (k = 0; k < c; ++k) {
			for (j = 0; j < h*stride; ++j) {
				for (i = 0; i < w*stride; ++i) {
					int in_index = b * w*h*c + k * w*h + (j / stride)*w + i / stride;
					int out_index = b * w*h*c*stride*stride + k * w*h*stride*stride + j * w*stride + i;
					if (forward) out[out_index] = scale * in[in_index];
					else in[in_index] += scale * out[out_index];
				}
			}
		}
	}
}




