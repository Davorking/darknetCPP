#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <float.h>
#include <limits.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

//put cassert here to avoid 'assert not defined'
#include <cassert>


//Subsititution of <sys/time.h> for what_time_is_it_now()
#include <iostream>
#include <chrono>
#include <ctime>  

//Substitution of <unistd.h> for read(), write()
#include <io.h> 

#include "layers.h"
#include "darknet.h"


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








//box.c
int nms_comparator(const void *pa, const void *pb)
{
	detection a = *(detection *)pa;
	detection b = *(detection *)pb;
	float diff = 0;
	if (b.sort_class >= 0) {
		diff = a.prob[b.sort_class] - b.prob[b.sort_class];
	}
	else {
		diff = a.objectness - b.objectness;
	}
	if (diff < 0) return 1;
	else if (diff > 0) return -1;
	return 0;
}

void do_nms_obj(detection *dets, int total, int classes, float thresh)
{
	int i, j, k;
	k = total - 1;
	for (i = 0; i <= k; ++i) {
		if (dets[i].objectness == 0) {
			detection swap = dets[i];
			dets[i] = dets[k];
			dets[k] = swap;
			--k;
			--i;
		}
	}
	total = k + 1;

	for (i = 0; i < total; ++i) {
		dets[i].sort_class = -1;
	}

	qsort(dets, total, sizeof(detection), nms_comparator);
	for (i = 0; i < total; ++i) {
		if (dets[i].objectness == 0) continue;
		box a = dets[i].bbox;
		for (j = i + 1; j < total; ++j) {
			if (dets[j].objectness == 0) continue;
			box b = dets[j].bbox;
			if (box_iou(a, b) > thresh) {
				dets[j].objectness = 0;
				for (k = 0; k < classes; ++k) {
					dets[j].prob[k] = 0;
				}
			}
		}
	}
}

void do_nms_sort(detection *dets, int total, int classes, float thresh)
{
	int i, j, k;
	k = total - 1;
	for (i = 0; i <= k; ++i) {
		if (dets[i].objectness == 0) {
			detection swap = dets[i];
			dets[i] = dets[k];
			dets[k] = swap;
			--k;
			--i;
		}
	}
	total = k + 1;

	for (k = 0; k < classes; ++k) {
		for (i = 0; i < total; ++i) {
			dets[i].sort_class = k;
		}
		qsort(dets, total, sizeof(detection), nms_comparator);
		for (i = 0; i < total; ++i) {
			if (dets[i].prob[k] == 0) continue;
			box a = dets[i].bbox;
			for (j = i + 1; j < total; ++j) {
				box b = dets[j].bbox;
				if (box_iou(a, b) > thresh) {
					dets[j].prob[k] = 0;
				}
			}
		}
	}
}

box float_to_box(float *f, int stride)
{
	box b = { 0 };
	b.x = f[0];
	b.y = f[1 * stride];
	b.w = f[2 * stride];
	b.h = f[3 * stride];
	return b;
}

dbox derivative(box a, box b)
{
	dbox d;
	d.dx = 0;
	d.dw = 0;
	float l1 = a.x - a.w / 2;
	float l2 = b.x - b.w / 2;
	if (l1 > l2) {
		d.dx -= 1;
		d.dw += .5;
	}
	float r1 = a.x + a.w / 2;
	float r2 = b.x + b.w / 2;
	if (r1 < r2) {
		d.dx += 1;
		d.dw += .5;
	}
	if (l1 > r2) {
		d.dx = -1;
		d.dw = 0;
	}
	if (r1 < l2) {
		d.dx = 1;
		d.dw = 0;
	}

	d.dy = 0;
	d.dh = 0;
	float t1 = a.y - a.h / 2;
	float t2 = b.y - b.h / 2;
	if (t1 > t2) {
		d.dy -= 1;
		d.dh += .5;
	}
	float b1 = a.y + a.h / 2;
	float b2 = b.y + b.h / 2;
	if (b1 < b2) {
		d.dy += 1;
		d.dh += .5;
	}
	if (t1 > b2) {
		d.dy = -1;
		d.dh = 0;
	}
	if (b1 < t2) {
		d.dy = 1;
		d.dh = 0;
	}
	return d;
}

float overlap(float x1, float w1, float x2, float w2)
{
	float l1 = x1 - w1 / 2;
	float l2 = x2 - w2 / 2;
	float left = l1 > l2 ? l1 : l2;
	float r1 = x1 + w1 / 2;
	float r2 = x2 + w2 / 2;
	float right = r1 < r2 ? r1 : r2;
	return right - left;
}

float box_intersection(box a, box b)
{
	float w = overlap(a.x, a.w, b.x, b.w);
	float h = overlap(a.y, a.h, b.y, b.h);
	if (w < 0 || h < 0) return 0;
	float area = w * h;
	return area;
}

float box_union(box a, box b)
{
	float i = box_intersection(a, b);
	float u = a.w*a.h + b.w*b.h - i;
	return u;
}

float box_iou(box a, box b)
{
	return box_intersection(a, b) / box_union(a, b);
}

float box_rmse(box a, box b)
{
	return sqrt(pow(a.x - b.x, 2) +
		pow(a.y - b.y, 2) +
		pow(a.w - b.w, 2) +
		pow(a.h - b.h, 2));
}

dbox dintersect(box a, box b)
{
	float w = overlap(a.x, a.w, b.x, b.w);
	float h = overlap(a.y, a.h, b.y, b.h);
	dbox dover = derivative(a, b);
	dbox di;

	di.dw = dover.dw*h;
	di.dx = dover.dx*h;
	di.dh = dover.dh*w;
	di.dy = dover.dy*w;

	return di;
}

dbox dunion(box a, box b)
{
	dbox du;

	dbox di = dintersect(a, b);
	du.dw = a.h - di.dw;
	du.dh = a.w - di.dh;
	du.dx = -di.dx;
	du.dy = -di.dy;

	return du;
}


void test_dunion()
{
	box a = { 0, 0, 1, 1 };
	box dxa = { 0 + .0001, 0, 1, 1 };
	box dya = { 0, 0 + .0001, 1, 1 };
	box dwa = { 0, 0, 1 + .0001, 1 };
	box dha = { 0, 0, 1, 1 + .0001 };

	box b = { .5, .5, .2, .2 };
	dbox di = dunion(a, b);
	printf("Union: %f %f %f %f\n", di.dx, di.dy, di.dw, di.dh);
	float inter = box_union(a, b);
	float xinter = box_union(dxa, b);
	float yinter = box_union(dya, b);
	float winter = box_union(dwa, b);
	float hinter = box_union(dha, b);
	xinter = (xinter - inter) / (.0001);
	yinter = (yinter - inter) / (.0001);
	winter = (winter - inter) / (.0001);
	hinter = (hinter - inter) / (.0001);
	printf("Union Manual %f %f %f %f\n", xinter, yinter, winter, hinter);
}
void test_dintersect()
{
	box a = { 0, 0, 1, 1 };
	box dxa = { 0 + .0001, 0, 1, 1 };
	box dya = { 0, 0 + .0001, 1, 1 };
	box dwa = { 0, 0, 1 + .0001, 1 };
	box dha = { 0, 0, 1, 1 + .0001 };

	box b = { .5, .5, .2, .2 };
	dbox di = dintersect(a, b);
	printf("Inter: %f %f %f %f\n", di.dx, di.dy, di.dw, di.dh);
	float inter = box_intersection(a, b);
	float xinter = box_intersection(dxa, b);
	float yinter = box_intersection(dya, b);
	float winter = box_intersection(dwa, b);
	float hinter = box_intersection(dha, b);
	xinter = (xinter - inter) / (.0001);
	yinter = (yinter - inter) / (.0001);
	winter = (winter - inter) / (.0001);
	hinter = (hinter - inter) / (.0001);
	printf("Inter Manual %f %f %f %f\n", xinter, yinter, winter, hinter);
}

void test_box()
{
	test_dintersect();
	test_dunion();
	box a = { 0, 0, 1, 1 };
	box dxa = { 0 + .00001, 0, 1, 1 };
	box dya = { 0, 0 + .00001, 1, 1 };
	box dwa = { 0, 0, 1 + .00001, 1 };
	box dha = { 0, 0, 1, 1 + .00001 };

	box b = { .5, 0, .2, .2 };

	float iou = box_iou(a, b);
	iou = (1 - iou)*(1 - iou);
	printf("%f\n", iou);
	dbox d = diou(a, b);
	printf("%f %f %f %f\n", d.dx, d.dy, d.dw, d.dh);

	float xiou = box_iou(dxa, b);
	float yiou = box_iou(dya, b);
	float wiou = box_iou(dwa, b);
	float hiou = box_iou(dha, b);
	xiou = ((1 - xiou)*(1 - xiou) - iou) / (.00001);
	yiou = ((1 - yiou)*(1 - yiou) - iou) / (.00001);
	wiou = ((1 - wiou)*(1 - wiou) - iou) / (.00001);
	hiou = ((1 - hiou)*(1 - hiou) - iou) / (.00001);
	printf("manual %f %f %f %f\n", xiou, yiou, wiou, hiou);
}

dbox diou(box a, box b)
{
	float u = box_union(a, b);
	float i = box_intersection(a, b);
	dbox di = dintersect(a, b);
	dbox du = dunion(a, b);
	dbox dd = { 0,0,0,0 };

	if (i <= 0 || 1) {
		dd.dx = b.x - a.x;
		dd.dy = b.y - a.y;
		dd.dw = b.w - a.w;
		dd.dh = b.h - a.h;
		return dd;
	}

	dd.dx = 2 * pow((1 - (i / u)), 1)*(di.dx*u - du.dx*i) / (u*u);
	dd.dy = 2 * pow((1 - (i / u)), 1)*(di.dy*u - du.dy*i) / (u*u);
	dd.dw = 2 * pow((1 - (i / u)), 1)*(di.dw*u - du.dw*i) / (u*u);
	dd.dh = 2 * pow((1 - (i / u)), 1)*(di.dh*u - du.dh*i) / (u*u);
	return dd;
}

void do_nms(box *boxes, float **probs, int total, int classes, float thresh)
{
	int i, j, k;
	for (i = 0; i < total; ++i) {
		int any = 0;
		for (k = 0; k < classes; ++k) any = any || (probs[i][k] > 0);
		if (!any) {
			continue;
		}
		for (j = i + 1; j < total; ++j) {
			if (box_iou(boxes[i], boxes[j]) > thresh) {
				for (k = 0; k < classes; ++k) {
					if (probs[i][k] < probs[j][k]) probs[i][k] = 0;
					else probs[j][k] = 0;
				}
			}
		}
	}
}

box encode_box(box b, box anchor)
{
	box encode;
	encode.x = (b.x - anchor.x) / anchor.w;
	encode.y = (b.y - anchor.y) / anchor.h;
	encode.w = log2(b.w / anchor.w);
	encode.h = log2(b.h / anchor.h);
	return encode;
}

box decode_box(box b, box anchor)
{
	box decode;
	decode.x = b.x * anchor.w + anchor.x;
	decode.y = b.y * anchor.h + anchor.y;
	decode.w = pow(2., b.w) * anchor.w;
	decode.h = pow(2., b.h) * anchor.h;
	return decode;
}







//col2im.c
void col2im_add_pixel(float *im, int height, int width, int channels,
	int row, int col, int channel, int pad, float val)
{
	row -= pad;
	col -= pad;

	if (row < 0 || col < 0 ||
		row >= height || col >= width) return;
	im[col + width * (row + height * channel)] += val;
}
//This one might be too, can't remember.
void col2im_cpu(float* data_col,
	int channels, int height, int width,
	int ksize, int stride, int pad, float* data_im)
{
	int c, h, w;
	int height_col = (height + 2 * pad - ksize) / stride + 1;
	int width_col = (width + 2 * pad - ksize) / stride + 1;

	int channels_col = channels * ksize * ksize;
	for (c = 0; c < channels_col; ++c) {
		int w_offset = c % ksize;
		int h_offset = (c / ksize) % ksize;
		int c_im = c / ksize / ksize;
		for (h = 0; h < height_col; ++h) {
			for (w = 0; w < width_col; ++w) {
				int im_row = h_offset + h * stride;
				int im_col = w_offset + w * stride;
				int col_index = (c * height_col + h) * width_col + w;
				double val = data_col[col_index];
				col2im_add_pixel(data_im, height, width, channels,
					im_row, im_col, c_im, pad, val);
			}
		}
	}
}







//compare.c
void train_compare(char *cfgfile, char *weightfile)
{
	srand(time(0));
	float avg_loss = -1;
	char *base = basecfg(cfgfile);
	char *backup_directory = (char*)"/home/pjreddie/backup/";
	printf("%s\n", base);

	network net = *parse_network_cfg(cfgfile);


	if (weightfile) {
		load_weights(&net, weightfile);
	}
	printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
	int imgs = 1024;
	list *plist = get_paths((char*)"data/compare.train.list");
	char **paths = (char **)list_to_array(plist);
	int N = plist->size;
	printf("%d\n", N);
	clock_t time;
	pthread_t load_thread;
	data train;
	data buffer;

	load_args args = { 0 };
	args.w = net.w;
	args.h = net.h;
	args.paths = paths;
	args.classes = 20;
	args.n = imgs;
	args.m = N;
	args.d = &buffer;
	args.type = COMPARE_DATA;

	load_thread = load_data_in_thread(args);
	int epoch = *net.seen / N;
	int i = 0;
	while (1) {
		++i;
		time = clock();
		pthread_join(load_thread, 0);
		train = buffer;

		load_thread = load_data_in_thread(args);
		printf("Loaded: %lf seconds\n", sec(clock() - time));
		time = clock();
		float loss = train_network(&net, train);
		if (avg_loss == -1) avg_loss = loss;
		avg_loss = avg_loss * .9 + loss * .1;
		printf("%.3f: %f, %f avg, %lf seconds, %ld images\n", (float)*net.seen / N, loss, avg_loss, sec(clock() - time), *net.seen);
		free_data(train);
		if (i % 100 == 0) {
			char buff[256];
			sprintf(buff, "%s/%s_%d_minor_%d.weights", backup_directory, base, epoch, i);
			save_weights(&net, buff);
		}
		if (*net.seen / N > epoch) {
			epoch = *net.seen / N;
			i = 0;
			char buff[256];
			sprintf(buff, "%s/%s_%d.weights", backup_directory, base, epoch);
			save_weights(&net, buff);
			if (epoch % 22 == 0) net.learning_rate *= .1;
		}
	}
	pthread_join(load_thread, 0);
	free_data(buffer);
	free_network(&net);
	free_ptrs((void**)paths, plist->size);
	free_list(plist);
	free(base);
}

void validate_compare(char *filename, char *weightfile)
{
	int i = 0;
	network net = *parse_network_cfg(filename);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	srand(time(0));

	list *plist = get_paths((char*)"data/compare.val.list");
	//list *plist = get_paths("data/compare.val.old");
	char **paths = (char **)list_to_array(plist);
	int N = plist->size / 2;
	free_list(plist);

	clock_t time;
	int correct = 0;
	int total = 0;
	int splits = 10;
	int num = (i + 1)*N / splits - i * N / splits;

	data val, buffer;

	load_args args = { 0 };
	args.w = net.w;
	args.h = net.h;
	args.paths = paths;
	args.classes = 20;
	args.n = num;
	args.m = 0;
	args.d = &buffer;
	args.type = COMPARE_DATA;

	pthread_t load_thread = load_data_in_thread(args);
	for (i = 1; i <= splits; ++i) {
		time = clock();

		pthread_join(load_thread, 0);
		val = buffer;

		num = (i + 1)*N / splits - i * N / splits;
		char **part = paths + (i*N / splits);
		if (i != splits) {
			args.paths = part;
			load_thread = load_data_in_thread(args);
		}
		printf("Loaded: %d images in %lf seconds\n", val.X.rows, sec(clock() - time));

		time = clock();
		matrix pred = network_predict_data(&net, val);
		int j, k;
		for (j = 0; j < val.y.rows; ++j) {
			for (k = 0; k < 20; ++k) {
				if (val.y.vals[j][k * 2] != val.y.vals[j][k * 2 + 1]) {
					++total;
					if ((val.y.vals[j][k * 2] < val.y.vals[j][k * 2 + 1]) == (pred.vals[j][k * 2] < pred.vals[j][k * 2 + 1])) {
						++correct;
					}
				}
			}
		}
		free_matrix(pred);
		printf("%d: Acc: %f, %lf seconds, %d images\n", i, (float)correct / total, sec(clock() - time), val.X.rows);
		free_data(val);
	}
}

int total_compares = 0;
int current_class = 0;

int elo_comparator(const void*a, const void *b)
{
	sortable_bbox box1 = *(sortable_bbox*)a;
	sortable_bbox box2 = *(sortable_bbox*)b;
	if (box1.elos[current_class] == box2.elos[current_class]) return 0;
	if (box1.elos[current_class] > box2.elos[current_class]) return -1;
	return 1;
}

int bbox_comparator(const void *a, const void *b)
{
	++total_compares;
	sortable_bbox box1 = *(sortable_bbox*)a;
	sortable_bbox box2 = *(sortable_bbox*)b;
	network net = box1.net;
	int class_n = box1.class_n;

	image im1 = load_image_color(box1.filename, net.w, net.h);
	image im2 = load_image_color(box2.filename, net.w, net.h);
	float *X = (float*)calloc(net.w*net.h*net.c, sizeof(float));
	memcpy(X, im1.data, im1.w*im1.h*im1.c * sizeof(float));
	memcpy(X + im1.w*im1.h*im1.c, im2.data, im2.w*im2.h*im2.c * sizeof(float));
	float *predictions = network_predict(&net, X);

	free_image(im1);
	free_image(im2);
	free(X);
	if (predictions[class_n * 2] > predictions[class_n * 2 + 1]) {
		return 1;
	}
	return -1;
}

void bbox_update(sortable_bbox *a, sortable_bbox *b, int class_n, int result)
{
	int k = 32;
	float EA = 1. / (1 + pow(10, (b->elos[class_n] - a->elos[class_n]) / 400.));
	float EB = 1. / (1 + pow(10, (a->elos[class_n] - b->elos[class_n]) / 400.));
	float SA = result ? 1 : 0;
	float SB = result ? 0 : 1;
	a->elos[class_n] += k * (SA - EA);
	b->elos[class_n] += k * (SB - EB);
}

void bbox_fight(network net, sortable_bbox *a, sortable_bbox *b, int classes, int class_n)
{
	image im1 = load_image_color(a->filename, net.w, net.h);
	image im2 = load_image_color(b->filename, net.w, net.h);
	float *X = (float*)calloc(net.w*net.h*net.c, sizeof(float));
	memcpy(X, im1.data, im1.w*im1.h*im1.c * sizeof(float));
	memcpy(X + im1.w*im1.h*im1.c, im2.data, im2.w*im2.h*im2.c * sizeof(float));
	float *predictions = network_predict(&net, X);
	++total_compares;

	int i;
	for (i = 0; i < classes; ++i) {
		if (class_n < 0 || class_n == i) {
			int result = predictions[i * 2] > predictions[i * 2 + 1];
			bbox_update(a, b, i, result);
		}
	}

	free_image(im1);
	free_image(im2);
	free(X);
}

void SortMaster3000(char *filename, char *weightfile)
{
	int i = 0;
	network net = *parse_network_cfg(filename);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	srand(time(0));
	set_batch_network(&net, 1);

	list *plist = get_paths((char*)"data/compare.sort.list");
	//list *plist = get_paths("data/compare.val.old");
	char **paths = (char **)list_to_array(plist);
	int N = plist->size;
	free_list(plist);
	sortable_bbox *boxes = (sortable_bbox*)calloc(N, sizeof(sortable_bbox));
	printf("Sorting %d boxes...\n", N);
	for (i = 0; i < N; ++i) {
		boxes[i].filename = paths[i];
		boxes[i].net = net;
		boxes[i].class_n = 7;
		boxes[i].elo = 1500;
	}
	clock_t time = clock();
	qsort(boxes, N, sizeof(sortable_bbox), bbox_comparator);
	for (i = 0; i < N; ++i) {
		printf("%s\n", boxes[i].filename);
	}
	printf("Sorted in %d compares, %f secs\n", total_compares, sec(clock() - time));
}

void BattleRoyaleWithCheese(char *filename, char *weightfile)
{
	int classes = 20;
	int i, j;
	network net = *parse_network_cfg(filename);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	srand(time(0));
	set_batch_network(&net, 1);

	list *plist = get_paths((char*)"data/compare.sort.list");
	//list *plist = get_paths("data/compare.small.list");
	//list *plist = get_paths("data/compare.cat.list");
	//list *plist = get_paths("data/compare.val.old");
	char **paths = (char **)list_to_array(plist);
	int N = plist->size;
	int total = N;
	free_list(plist);
	sortable_bbox *boxes = (sortable_bbox*)calloc(N, sizeof(sortable_bbox));
	printf("Battling %d boxes...\n", N);
	for (i = 0; i < N; ++i) {
		boxes[i].filename = paths[i];
		boxes[i].net = net;
		boxes[i].classes = classes;
		boxes[i].elos = (float*)calloc(classes, sizeof(float));;
		for (j = 0; j < classes; ++j) {
			boxes[i].elos[j] = 1500;
		}
	}
	int round;
	clock_t time = clock();
	for (round = 1; round <= 4; ++round) {
		clock_t round_time = clock();
		printf("Round: %d\n", round);
		shuffle(boxes, N, sizeof(sortable_bbox));
		for (i = 0; i < N / 2; ++i) {
			bbox_fight(net, boxes + i * 2, boxes + i * 2 + 1, classes, -1);
		}
		printf("Round: %f secs, %d remaining\n", sec(clock() - round_time), N);
	}

	int class_n;

	for (class_n = 0; class_n < classes; ++class_n) {

		N = total;
		current_class = class_n;
		qsort(boxes, N, sizeof(sortable_bbox), elo_comparator);
		N /= 2;

		for (round = 1; round <= 100; ++round) {
			clock_t round_time = clock();
			printf("Round: %d\n", round);

			sorta_shuffle(boxes, N, sizeof(sortable_bbox), 10);
			for (i = 0; i < N / 2; ++i) {
				bbox_fight(net, boxes + i * 2, boxes + i * 2 + 1, classes, class_n);
			}
			qsort(boxes, N, sizeof(sortable_bbox), elo_comparator);
			if (round <= 20) N = (N * 9 / 10) / 2 * 2;

			printf("Round: %f secs, %d remaining\n", sec(clock() - round_time), N);
		}
		char buff[256];
		sprintf(buff, "results/battle_%d.log", class_n);
		FILE *outfp = fopen(buff, "w");
		for (i = 0; i < N; ++i) {
			fprintf(outfp, "%s %f\n", boxes[i].filename, boxes[i].elos[class_n]);
		}
		fclose(outfp);
	}
	printf("Tournament in %d compares, %f secs\n", total_compares, sec(clock() - time));
}

void run_compare(int argc, char **argv)
{
	if (argc < 4) {
		fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
		return;
	}

	char *cfg = argv[3];
	char *weights = (argc > 4) ? argv[4] : 0;
	//char *filename = (argc > 5) ? argv[5]: 0;
	if (0 == strcmp(argv[2], "train")) train_compare(cfg, weights);
	else if (0 == strcmp(argv[2], "valid")) validate_compare(cfg, weights);
	else if (0 == strcmp(argv[2], "sort")) SortMaster3000(cfg, weights);
	else if (0 == strcmp(argv[2], "battle")) BattleRoyaleWithCheese(cfg, weights);
	/*
	   else if(0==strcmp(argv[2], "train")) train_coco(cfg, weights);
	   else if(0==strcmp(argv[2], "extract")) extract_boxes(cfg, weights);
	   else if(0==strcmp(argv[2], "valid")) validate_recall(cfg, weights);
	 */
}







//cuda.c
int gpu_index = 0;

#ifdef GPU

#include "cuda.h"
#include "utils.h"
#include "blas.h"
#include <assert.h>
#include <stdlib.h>
#include <time.h>

void cuda_set_device(int n)
{
	gpu_index = n;
	cudaError_t status = cudaSetDevice(n);
	check_error(status);
}

int cuda_get_device()
{
	int n = 0;
	cudaError_t status = cudaGetDevice(&n);
	check_error(status);
	return n;
}

void check_error(cudaError_t status)
{
	//cudaDeviceSynchronize();
	cudaError_t status2 = cudaGetLastError();
	if (status != cudaSuccess)
	{
		const char *s = cudaGetErrorString(status);
		char buffer[256];
		printf("CUDA Error: %s\n", s);
		assert(0);
		snprintf(buffer, 256, "CUDA Error: %s", s);
		error(buffer);
	}
	if (status2 != cudaSuccess)
	{
		const char *s = cudaGetErrorString(status);
		char buffer[256];
		printf("CUDA Error Prev: %s\n", s);
		assert(0);
		snprintf(buffer, 256, "CUDA Error Prev: %s", s);
		error(buffer);
	}
}

dim3 cuda_gridsize(size_t n) {
	size_t k = (n - 1) / BLOCK + 1;
	size_t x = k;
	size_t y = 1;
	if (x > 65535) {
		x = ceil(sqrt(k));
		y = (n - 1) / (x*BLOCK) + 1;
	}
	dim3 d = { x, y, 1 };
	//printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
	return d;
}

#ifdef CUDNN
cudnnHandle_t cudnn_handle()
{
	static int init[16] = { 0 };
	static cudnnHandle_t handle[16];
	int i = cuda_get_device();
	if (!init[i]) {
		cudnnCreate(&handle[i]);
		init[i] = 1;
	}
	return handle[i];
}
#endif

cublasHandle_t blas_handle()
{
	static int init[16] = { 0 };
	static cublasHandle_t handle[16];
	int i = cuda_get_device();
	if (!init[i]) {
		cublasCreate(&handle[i]);
		init[i] = 1;
	}
	return handle[i];
}

float *cuda_make_array(float *x, size_t n)
{
	float *x_gpu;
	size_t size = sizeof(float)*n;
	cudaError_t status = cudaMalloc((void **)&x_gpu, size);
	check_error(status);
	if (x) {
		status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
		check_error(status);
	}
	else {
		fill_gpu(n, 0, x_gpu, 1);
	}
	if (!x_gpu) error("Cuda malloc failed\n");
	return x_gpu;
}

void cuda_random(float *x_gpu, size_t n)
{
	static curandGenerator_t gen[16];
	static int init[16] = { 0 };
	int i = cuda_get_device();
	if (!init[i]) {
		curandCreateGenerator(&gen[i], CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(gen[i], time(0));
		init[i] = 1;
	}
	curandGenerateUniform(gen[i], x_gpu, n);
	check_error(cudaPeekAtLastError());
}

float cuda_compare(float *x_gpu, float *x, size_t n, char *s)
{
	float *tmp = calloc(n, sizeof(float));
	cuda_pull_array(x_gpu, tmp, n);
	//int i;
	//for(i = 0; i < n; ++i) printf("%f %f\n", tmp[i], x[i]);
	axpy_cpu(n, -1, x, 1, tmp, 1);
	float err = dot_cpu(n, tmp, 1, tmp, 1);
	printf("Error %s: %f\n", s, sqrt(err / n));
	free(tmp);
	return err;
}

int *cuda_make_int_array(int *x, size_t n)
{
	int *x_gpu;
	size_t size = sizeof(int)*n;
	cudaError_t status = cudaMalloc((void **)&x_gpu, size);
	check_error(status);
	if (x) {
		status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
		check_error(status);
	}
	if (!x_gpu) error("Cuda malloc failed\n");
	return x_gpu;
}

void cuda_free(float *x_gpu)
{
	cudaError_t status = cudaFree(x_gpu);
	check_error(status);
}

void cuda_push_array(float *x_gpu, float *x, size_t n)
{
	size_t size = sizeof(float)*n;
	cudaError_t status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
	check_error(status);
}

void cuda_pull_array(float *x_gpu, float *x, size_t n)
{
	size_t size = sizeof(float)*n;
	cudaError_t status = cudaMemcpy(x, x_gpu, size, cudaMemcpyDeviceToHost);
	check_error(status);
}

float cuda_mag_array(float *x_gpu, size_t n)
{
	float *temp = calloc(n, sizeof(float));
	cuda_pull_array(x_gpu, temp, n);
	float m = mag_array(temp, n);
	free(temp);
	return m;
}
#else
void cuda_set_device(int n) {}

#endif







//data.c
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

list *get_paths(char *filename)
{
	char *path;
	FILE *file = fopen(filename, "r");
	if (!file) file_error(filename);
	list *lines = make_list();
	while ((path = fgetl(file))) {
		list_insert(lines, path);
	}
	fclose(file);
	return lines;
}

/*
char **get_random_paths_indexes(char **paths, int n, int m, int *indexes)
{
	char **random_paths = calloc(n, sizeof(char*));
	int i;
	pthread_mutex_lock(&mutex);
	for(i = 0; i < n; ++i){
		int index = rand()%m;
		indexes[i] = index;
		random_paths[i] = paths[index];
		if(i == 0) printf("%s\n", paths[index]);
	}
	pthread_mutex_unlock(&mutex);
	return random_paths;
}
*/

char **get_random_paths(char **paths, int n, int m)
{
	char **random_paths = (char**)calloc(n, sizeof(char*));
	int i;
	pthread_mutex_lock(&mutex);
	for (i = 0; i < n; ++i) {
		int index = rand() % m;
		random_paths[i] = paths[index];
		//if(i == 0) printf("%s\n", paths[index]);
	}
	pthread_mutex_unlock(&mutex);
	return random_paths;
}

char **find_replace_paths(char **paths, int n, char *find, char *replace)
{
	char **replace_paths = (char**)calloc(n, sizeof(char*));
	int i;
	for (i = 0; i < n; ++i) {
		char replaced[4096];
		find_replace(paths[i], find, replace, replaced);
		replace_paths[i] = copy_string(replaced);
	}
	return replace_paths;
}

matrix load_image_paths_gray(char **paths, int n, int w, int h)
{
	int i;
	matrix X;
	X.rows = n;
	X.vals = (float**)calloc(X.rows, sizeof(float*));
	X.cols = 0;

	for (i = 0; i < n; ++i) {
		image im = load_image(paths[i], w, h, 3);

		image gray = grayscale_image(im);
		free_image(im);
		im = gray;

		X.vals[i] = im.data;
		X.cols = im.h*im.w*im.c;
	}
	return X;
}

matrix load_image_paths(char **paths, int n, int w, int h)
{
	int i;
	matrix X;
	X.rows = n;
	X.vals = (float**)calloc(X.rows, sizeof(float*));
	X.cols = 0;

	for (i = 0; i < n; ++i) {
		image im = load_image_color(paths[i], w, h);
		X.vals[i] = im.data;
		X.cols = im.h*im.w*im.c;
	}
	return X;
}

matrix load_image_augment_paths(char **paths, int n, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure, int center)
{
	int i;
	matrix X;
	X.rows = n;
	X.vals = (float**)calloc(X.rows, sizeof(float*));
	X.cols = 0;

	for (i = 0; i < n; ++i) {
		image im = load_image_color(paths[i], 0, 0);
		image crop;
		if (center) {
			crop = center_crop_image(im, size, size);
		}
		else {
			crop = random_augment_image(im, angle, aspect, min, max, size, size);
		}
		int flip = rand() % 2;
		if (flip) flip_image(crop);
		random_distort_image(crop, hue, saturation, exposure);

		/*
		show_image(im, "orig");
		show_image(crop, "crop");
		cvWaitKey(0);
		*/
		//grayscale_image_3c(crop);
		free_image(im);
		X.vals[i] = crop.data;
		X.cols = crop.h*crop.w*crop.c;
	}
	return X;
}


box_label *read_boxes(char *filename, int *n)
{
	FILE *file = fopen(filename, "r");
	if (!file) file_error(filename);
	float x, y, h, w;
	int id;
	int count = 0;
	int size = 64;
	box_label *boxes = (box_label*)calloc(size, sizeof(box_label));
	while (fscanf(file, "%d %f %f %f %f", &id, &x, &y, &w, &h) == 5) {
		if (count == size) {
			size = size * 2;
			boxes = (box_label*)realloc(boxes, size * sizeof(box_label));
		}
		boxes[count].id = id;
		boxes[count].x = x;
		boxes[count].y = y;
		boxes[count].h = h;
		boxes[count].w = w;
		boxes[count].left = x - w / 2;
		boxes[count].right = x + w / 2;
		boxes[count].top = y - h / 2;
		boxes[count].bottom = y + h / 2;
		++count;
	}
	fclose(file);
	*n = count;
	return boxes;
}

void randomize_boxes(box_label *b, int n)
{
	int i;
	for (i = 0; i < n; ++i) {
		box_label swap = b[i];
		int index = rand() % n;
		b[i] = b[index];
		b[index] = swap;
	}
}

void correct_boxes(box_label *boxes, int n, float dx, float dy, float sx, float sy, int flip)
{
	int i;
	for (i = 0; i < n; ++i) {
		if (boxes[i].x == 0 && boxes[i].y == 0) {
			boxes[i].x = 999999;
			boxes[i].y = 999999;
			boxes[i].w = 999999;
			boxes[i].h = 999999;
			continue;
		}
		boxes[i].left = boxes[i].left  * sx - dx;
		boxes[i].right = boxes[i].right * sx - dx;
		boxes[i].top = boxes[i].top   * sy - dy;
		boxes[i].bottom = boxes[i].bottom* sy - dy;

		if (flip) {
			float swap = boxes[i].left;
			boxes[i].left = 1. - boxes[i].right;
			boxes[i].right = 1. - swap;
		}

		boxes[i].left = constrain(0, 1, boxes[i].left);
		boxes[i].right = constrain(0, 1, boxes[i].right);
		boxes[i].top = constrain(0, 1, boxes[i].top);
		boxes[i].bottom = constrain(0, 1, boxes[i].bottom);

		boxes[i].x = (boxes[i].left + boxes[i].right) / 2;
		boxes[i].y = (boxes[i].top + boxes[i].bottom) / 2;
		boxes[i].w = (boxes[i].right - boxes[i].left);
		boxes[i].h = (boxes[i].bottom - boxes[i].top);

		boxes[i].w = constrain(0, 1, boxes[i].w);
		boxes[i].h = constrain(0, 1, boxes[i].h);
	}
}

void fill_truth_swag(char *path, float *truth, int classes, int flip, float dx, float dy, float sx, float sy)
{
	char labelpath[4096];
	find_replace(path, (char*)"images", (char*)"labels", labelpath);
	find_replace(labelpath, (char*)"JPEGImages", (char*)"labels", labelpath);
	find_replace(labelpath, (char*)".jpg", (char*)".txt", labelpath);
	find_replace(labelpath, (char*)".JPG", (char*)".txt", labelpath);
	find_replace(labelpath, (char*)".JPEG", (char*)".txt", labelpath);

	int count = 0;
	box_label *boxes = read_boxes(labelpath, &count);
	randomize_boxes(boxes, count);
	correct_boxes(boxes, count, dx, dy, sx, sy, flip);
	float x, y, w, h;
	int id;
	int i;

	for (i = 0; i < count && i < 90; ++i) {
		x = boxes[i].x;
		y = boxes[i].y;
		w = boxes[i].w;
		h = boxes[i].h;
		id = boxes[i].id;

		if (w < .0 || h < .0) continue;

		int index = (4 + classes) * i;

		truth[index++] = x;
		truth[index++] = y;
		truth[index++] = w;
		truth[index++] = h;

		if (id < classes) truth[index + id] = 1;
	}
	free(boxes);
}

void fill_truth_region(char *path, float *truth, int classes, int num_boxes, int flip, float dx, float dy, float sx, float sy)
{
	char labelpath[4096];
	find_replace(path, (char*)"images", (char*)"labels", labelpath);
	find_replace(labelpath, (char*)"JPEGImages", (char*)"labels", labelpath);

	find_replace(labelpath, (char*)".jpg", (char*)".txt", labelpath);
	find_replace(labelpath, (char*)".png", (char*)".txt", labelpath);
	find_replace(labelpath, (char*)".JPG", (char*)".txt", labelpath);
	find_replace(labelpath, (char*)".JPEG", (char*)".txt", labelpath);
	int count = 0;
	box_label *boxes = read_boxes(labelpath, &count);
	randomize_boxes(boxes, count);
	correct_boxes(boxes, count, dx, dy, sx, sy, flip);
	float x, y, w, h;
	int id;
	int i;

	for (i = 0; i < count; ++i) {
		x = boxes[i].x;
		y = boxes[i].y;
		w = boxes[i].w;
		h = boxes[i].h;
		id = boxes[i].id;

		if (w < .005 || h < .005) continue;

		int col = (int)(x*num_boxes);
		int row = (int)(y*num_boxes);

		x = x * num_boxes - col;
		y = y * num_boxes - row;

		int index = (col + row * num_boxes)*(5 + classes);
		if (truth[index]) continue;
		truth[index++] = 1;

		if (id < classes) truth[index + id] = 1;
		index += classes;

		truth[index++] = x;
		truth[index++] = y;
		truth[index++] = w;
		truth[index++] = h;
	}
	free(boxes);
}

void load_rle(image im, int *rle, int n)
{
	int count = 0;
	int curr = 0;
	int i, j;
	for (i = 0; i < n; ++i) {
		for (j = 0; j < rle[i]; ++j) {
			im.data[count++] = curr;
		}
		curr = 1 - curr;
	}
	for (; count < im.h*im.w*im.c; ++count) {
		im.data[count] = curr;
	}
}

void or_image(image src, image dest, int c)
{
	int i;
	for (i = 0; i < src.w*src.h; ++i) {
		if (src.data[i]) dest.data[dest.w*dest.h*c + i] = 1;
	}
}

void exclusive_image(image src)
{
	int k, j, i;
	int s = src.w*src.h;
	for (k = 0; k < src.c - 1; ++k) {
		for (i = 0; i < s; ++i) {
			if (src.data[k*s + i]) {
				for (j = k + 1; j < src.c; ++j) {
					src.data[j*s + i] = 0;
				}
			}
		}
	}
}

box bound_image(image im)
{
	int x, y;
	int minx = im.w;
	int miny = im.h;
	int maxx = 0;
	int maxy = 0;
	for (y = 0; y < im.h; ++y) {
		for (x = 0; x < im.w; ++x) {
			if (im.data[y*im.w + x]) {
				minx = (x < minx) ? x : minx;
				miny = (y < miny) ? y : miny;
				maxx = (x > maxx) ? x : maxx;
				maxy = (y > maxy) ? y : maxy;
			}
		}
	}
	box b = { minx, miny, maxx - minx + 1, maxy - miny + 1 };
	//printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);
	return b;
}

void fill_truth_iseg(char *path, int num_boxes, float *truth, int classes, int w, int h, augment_args aug, int flip, int mw, int mh)
{
	char labelpath[4096];
	find_replace(path, (char*)"images", (char*)"mask", labelpath);
	find_replace(labelpath, (char*)"JPEGImages", (char*)"mask", labelpath);
	find_replace(labelpath, (char*)".jpg", (char*)".txt", labelpath);
	find_replace(labelpath, (char*)".JPG", (char*)".txt", labelpath);
	find_replace(labelpath, (char*)".JPEG", (char*)".txt", labelpath);
	FILE *file = fopen(labelpath, "r");
	if (!file) file_error(labelpath);
	char buff[32788];
	int id;
	int i = 0;
	int j;
	image part = make_image(w, h, 1);
	while ((fscanf(file, "%d %s", &id, buff) == 2) && i < num_boxes) {
		int n = 0;
		int *rle = read_intlist(buff, &n, 0);
		load_rle(part, rle, n);
		image sized = rotate_crop_image(part, aug.rad, aug.scale, aug.w, aug.h, aug.dx, aug.dy, aug.aspect);
		if (flip) flip_image(sized);

		image mask = resize_image(sized, mw, mh);
		truth[i*(mw*mh + 1)] = id;
		for (j = 0; j < mw*mh; ++j) {
			truth[i*(mw*mh + 1) + 1 + j] = mask.data[j];
		}
		++i;

		free_image(mask);
		free_image(sized);
		free(rle);
	}
	if (i < num_boxes) truth[i*(mw*mh + 1)] = -1;
	fclose(file);
	free_image(part);
}

void fill_truth_mask(char *path, int num_boxes, float *truth, int classes, int w, int h, augment_args aug, int flip, int mw, int mh)
{
	char labelpath[4096];
	find_replace(path, (char*)"images", (char*)"mask", labelpath);
	find_replace(labelpath, (char*)"JPEGImages",(char*) "mask", labelpath);
	find_replace(labelpath, (char*)".jpg", (char*)".txt", labelpath);
	find_replace(labelpath, (char*)".JPG", (char*)".txt", labelpath);
	find_replace(labelpath, (char*)".JPEG", (char*)".txt", labelpath);
	FILE *file = fopen(labelpath, "r");
	if (!file) file_error(labelpath);
	char buff[32788];
	int id;
	int i = 0;
	image part = make_image(w, h, 1);
	while ((fscanf(file, "%d %s", &id, buff) == 2) && i < num_boxes) {
		int n = 0;
		int *rle = read_intlist(buff, &n, 0);
		load_rle(part, rle, n);
		image sized = rotate_crop_image(part, aug.rad, aug.scale, aug.w, aug.h, aug.dx, aug.dy, aug.aspect);
		if (flip) flip_image(sized);
		box b = bound_image(sized);
		if (b.w > 0) {
			image crop = crop_image(sized, b.x, b.y, b.w, b.h);
			image mask = resize_image(crop, mw, mh);
			truth[i*(4 + mw * mh + 1) + 0] = (b.x + b.w / 2.) / sized.w;
			truth[i*(4 + mw * mh + 1) + 1] = (b.y + b.h / 2.) / sized.h;
			truth[i*(4 + mw * mh + 1) + 2] = b.w / sized.w;
			truth[i*(4 + mw * mh + 1) + 3] = b.h / sized.h;
			int j;
			for (j = 0; j < mw*mh; ++j) {
				truth[i*(4 + mw * mh + 1) + 4 + j] = mask.data[j];
			}
			truth[i*(4 + mw * mh + 1) + 4 + mw * mh] = id;
			free_image(crop);
			free_image(mask);
			++i;
		}
		free_image(sized);
		free(rle);
	}
	fclose(file);
	free_image(part);
}


void fill_truth_detection(char *path, int num_boxes, float *truth, int classes, int flip, float dx, float dy, float sx, float sy)
{
	char labelpath[4096];
	find_replace(path, (char*)"images", (char*)"labels", labelpath);
	find_replace(labelpath, (char*)"JPEGImages", (char*)"labels", labelpath);

	find_replace(labelpath, (char*)"raw", (char*)"labels", labelpath);
	find_replace(labelpath, (char*)".jpg", (char*)".txt", labelpath);
	find_replace(labelpath, (char*)".png", (char*)".txt", labelpath);
	find_replace(labelpath, (char*)".JPG", (char*)".txt", labelpath);
	find_replace(labelpath, (char*)".JPEG", (char*)".txt", labelpath);
	int count = 0;
	box_label *boxes = read_boxes(labelpath, &count);
	randomize_boxes(boxes, count);
	correct_boxes(boxes, count, dx, dy, sx, sy, flip);
	if (count > num_boxes) count = num_boxes;
	float x, y, w, h;
	int id;
	int i;
	int sub = 0;

	for (i = 0; i < count; ++i) {
		x = boxes[i].x;
		y = boxes[i].y;
		w = boxes[i].w;
		h = boxes[i].h;
		id = boxes[i].id;

		if ((w < .001 || h < .001)) {
			++sub;
			continue;
		}

		truth[(i - sub) * 5 + 0] = x;
		truth[(i - sub) * 5 + 1] = y;
		truth[(i - sub) * 5 + 2] = w;
		truth[(i - sub) * 5 + 3] = h;
		truth[(i - sub) * 5 + 4] = id;
	}
	free(boxes);
}

#define NUMCHARS 37

void print_letters(float *pred, int n)
{
	int i;
	for (i = 0; i < n; ++i) {
		int index = max_index(pred + i * NUMCHARS, NUMCHARS);
		printf("%c", int_to_alphanum(index));
	}
	printf("\n");
}

void fill_truth_captcha(char *path, int n, float *truth)
{
	char *begin = strrchr(path, '/');
	++begin;
	int i;
	for (i = 0; i < strlen(begin) && i < n && begin[i] != '.'; ++i) {
		int index = alphanum_to_int(begin[i]);
		if (index > 35) printf("Bad %c\n", begin[i]);
		truth[i*NUMCHARS + index] = 1;
	}
	for (; i < n; ++i) {
		truth[i*NUMCHARS + NUMCHARS - 1] = 1;
	}
}

data load_data_captcha(char **paths, int n, int m, int k, int w, int h)
{
	if (m) paths = get_random_paths(paths, n, m);
	data d = { 0 };
	d.shallow = 0;
	d.X = load_image_paths(paths, n, w, h);
	d.y = make_matrix(n, k*NUMCHARS);
	int i;
	for (i = 0; i < n; ++i) {
		fill_truth_captcha(paths[i], k, d.y.vals[i]);
	}
	if (m) free(paths);
	return d;
}

data load_data_captcha_encode(char **paths, int n, int m, int w, int h)
{
	if (m) paths = get_random_paths(paths, n, m);
	data d = { 0 };
	d.shallow = 0;
	d.X = load_image_paths(paths, n, w, h);
	d.X.cols = 17100;
	d.y = d.X;
	if (m) free(paths);
	return d;
}

void fill_truth(char *path, char **labels, int k, float *truth)
{
	int i;
	memset(truth, 0, k * sizeof(float));
	int count = 0;
	for (i = 0; i < k; ++i) {
		if (strstr(path, labels[i])) {
			truth[i] = 1;
			++count;
			//printf("%s %s %d\n", path, labels[i], i);
		}
	}
	if (count != 1 && (k != 1 || count != 0)) printf("Too many or too few labels: %d, %s\n", count, path);
}

void fill_hierarchy(float *truth, int k, tree *hierarchy)
{
	int j;
	for (j = 0; j < k; ++j) {
		if (truth[j]) {
			int parent = hierarchy->parent[j];
			while (parent >= 0) {
				truth[parent] = 1;
				parent = hierarchy->parent[parent];
			}
		}
	}
	int i;
	int count = 0;
	for (j = 0; j < hierarchy->groups; ++j) {
		//printf("%d\n", count);
		int mask = 1;
		for (i = 0; i < hierarchy->group_size[j]; ++i) {
			if (truth[count + i]) {
				mask = 0;
				break;
			}
		}
		if (mask) {
			for (i = 0; i < hierarchy->group_size[j]; ++i) {
				truth[count + i] = SECRET_NUM;
			}
		}
		count += hierarchy->group_size[j];
	}
}

matrix load_regression_labels_paths(char **paths, int n, int k)
{
	matrix y = make_matrix(n, k);
	int i, j;
	for (i = 0; i < n; ++i) {
		char labelpath[4096];
		find_replace(paths[i], (char*)"images", (char*)"labels", labelpath);
		find_replace(labelpath, (char*)"JPEGImages", (char*)"labels", labelpath);
		find_replace(labelpath, (char*)".BMP", (char*)".txt", labelpath);
		find_replace(labelpath, (char*)".JPEG", (char*)".txt", labelpath);
		find_replace(labelpath, (char*)".JPG", (char*)".txt", labelpath);
		find_replace(labelpath, (char*)".JPeG", (char*)".txt", labelpath);
		find_replace(labelpath, (char*)".Jpeg", (char*)".txt", labelpath);
		find_replace(labelpath, (char*)".PNG", (char*)".txt", labelpath);
		find_replace(labelpath, (char*)".TIF", (char*)".txt", labelpath);
		find_replace(labelpath, (char*)".bmp", (char*)".txt", labelpath);
		find_replace(labelpath, (char*)".jpeg", (char*)".txt", labelpath);
		find_replace(labelpath, (char*)".jpg", (char*)".txt", labelpath);
		find_replace(labelpath, (char*)".png", (char*)".txt", labelpath);
		find_replace(labelpath, (char*)".tif", (char*)".txt", labelpath);

		FILE *file = fopen(labelpath, "r");
		for (j = 0; j < k; ++j) {
			fscanf(file, "%f", &(y.vals[i][j]));
		}
		fclose(file);
	}
	return y;
}

matrix load_labels_paths(char **paths, int n, char **labels, int k, tree *hierarchy)
{
	matrix y = make_matrix(n, k);
	int i;
	for (i = 0; i < n && labels; ++i) {
		fill_truth(paths[i], labels, k, y.vals[i]);
		if (hierarchy) {
			fill_hierarchy(y.vals[i], k, hierarchy);
		}
	}
	return y;
}

matrix load_tags_paths(char **paths, int n, int k)
{
	matrix y = make_matrix(n, k);
	int i;
	//int count = 0;
	for (i = 0; i < n; ++i) {
		char label[4096];
		find_replace(paths[i], (char*)"images", (char*)"labels", label);
		find_replace(label, (char*)".jpg", (char*)".txt", label);
		FILE *file = fopen(label, "r");
		if (!file) continue;
		//++count;
		int tag;
		while (fscanf(file, "%d", &tag) == 1) {
			if (tag < k) {
				y.vals[i][tag] = 1;
			}
		}
		fclose(file);
	}
	//printf("%d/%d\n", count, n);
	return y;
}

char **get_labels(char *filename)
{
	list *plist = get_paths(filename);
	char **labels = (char **)list_to_array(plist);
	free_list(plist);
	return labels;
}

void free_data(data d)
{
	if (!d.shallow) {
		free_matrix(d.X);
		free_matrix(d.y);
	}
	else {
		free(d.X.vals);
		free(d.y.vals);
	}
}

image get_segmentation_image(char *path, int w, int h, int classes)
{
	char labelpath[4096];
	find_replace(path, (char*)"images", (char*)"mask", labelpath);
	find_replace(labelpath, (char*)"JPEGImages", (char*)"mask", labelpath);
	find_replace(labelpath, (char*)".jpg", (char*)".txt", labelpath);
	find_replace(labelpath, (char*)".JPG", (char*)".txt", labelpath);
	find_replace(labelpath, (char*)".JPEG", (char*)".txt", labelpath);
	image mask = make_image(w, h, classes);
	FILE *file = fopen(labelpath, "r");
	if (!file) file_error(labelpath);
	char buff[32788];
	int id;
	image part = make_image(w, h, 1);
	while (fscanf(file, "%d %s", &id, buff) == 2) {
		int n = 0;
		int *rle = read_intlist(buff, &n, 0);
		load_rle(part, rle, n);
		or_image(part, mask, id);
		free(rle);
	}
	//exclusive_image(mask);
	fclose(file);
	free_image(part);
	return mask;
}

image get_segmentation_image2(char *path, int w, int h, int classes)
{
	char labelpath[4096];
	find_replace(path, (char*)"images", (char*)"mask", labelpath);
	find_replace(labelpath, (char*)"JPEGImages", (char*)"mask", labelpath);
	find_replace(labelpath, (char*)".jpg", (char*)".txt", labelpath);
	find_replace(labelpath, (char*)".JPG", (char*)".txt", labelpath);
	find_replace(labelpath, (char*)".JPEG", (char*)".txt", labelpath);
	image mask = make_image(w, h, classes + 1);
	int i;
	for (i = 0; i < w*h; ++i) {
		mask.data[w*h*classes + i] = 1;
	}
	FILE *file = fopen(labelpath, "r");
	if (!file) file_error(labelpath);
	char buff[32788];
	int id;
	image part = make_image(w, h, 1);
	while (fscanf(file, "%d %s", &id, buff) == 2) {
		int n = 0;
		int *rle = read_intlist(buff, &n, 0);
		load_rle(part, rle, n);
		or_image(part, mask, id);
		for (i = 0; i < w*h; ++i) {
			if (part.data[i]) mask.data[w*h*classes + i] = 0;
		}
		free(rle);
	}
	//exclusive_image(mask);
	fclose(file);
	free_image(part);
	return mask;
}

data load_data_seg(int n, char **paths, int m, int w, int h, int classes, int min, int max, float angle, float aspect, float hue, float saturation, float exposure, int div)
{
	char **random_paths = get_random_paths(paths, n, m);
	int i;
	data d = { 0 };
	d.shallow = 0;

	d.X.rows = n;
	d.X.vals = (float**)calloc(d.X.rows, sizeof(float*));
	d.X.cols = h * w * 3;


	d.y.rows = n;
	d.y.cols = h * w*classes / div / div;
	d.y.vals = (float**)calloc(d.X.rows, sizeof(float*));

	for (i = 0; i < n; ++i) {
		image orig = load_image_color(random_paths[i], 0, 0);
		augment_args a = random_augment_args(orig, angle, aspect, min, max, w, h);
		image sized = rotate_crop_image(orig, a.rad, a.scale, a.w, a.h, a.dx, a.dy, a.aspect);

		int flip = rand() % 2;
		if (flip) flip_image(sized);
		random_distort_image(sized, hue, saturation, exposure);
		d.X.vals[i] = sized.data;

		image mask = get_segmentation_image(random_paths[i], orig.w, orig.h, classes);
		//image mask = make_image(orig.w, orig.h, classes+1);
		image sized_m = rotate_crop_image(mask, a.rad, a.scale / div, a.w / div, a.h / div, a.dx / div, a.dy / div, a.aspect);

		if (flip) flip_image(sized_m);
		d.y.vals[i] = sized_m.data;

		free_image(orig);
		free_image(mask);

		/*
		   image rgb = mask_to_rgb(sized_m, classes);
		   show_image(rgb, "part");
		   show_image(sized, "orig");
		   cvWaitKey(0);
		   free_image(rgb);
		 */
	}
	free(random_paths);
	return d;
}

data load_data_iseg(int n, char **paths, int m, int w, int h, int classes, int boxes, int div, int min, int max, float angle, float aspect, float hue, float saturation, float exposure)
{
	char **random_paths = get_random_paths(paths, n, m);
	int i;
	data d = { 0 };
	d.shallow = 0;

	d.X.rows = n;
	d.X.vals = (float**)calloc(d.X.rows, sizeof(float*));
	d.X.cols = h * w * 3;

	d.y = make_matrix(n, (((w / div)*(h / div)) + 1)*boxes);

	for (i = 0; i < n; ++i) {
		image orig = load_image_color(random_paths[i], 0, 0);
		augment_args a = random_augment_args(orig, angle, aspect, min, max, w, h);
		image sized = rotate_crop_image(orig, a.rad, a.scale, a.w, a.h, a.dx, a.dy, a.aspect);

		int flip = rand() % 2;
		if (flip) flip_image(sized);
		random_distort_image(sized, hue, saturation, exposure);
		d.X.vals[i] = sized.data;
		//show_image(sized, "image");

		fill_truth_iseg(random_paths[i], boxes, d.y.vals[i], classes, orig.w, orig.h, a, flip, w / div, h / div);

		free_image(orig);

		/*
		   image rgb = mask_to_rgb(sized_m, classes);
		   show_image(rgb, "part");
		   show_image(sized, "orig");
		   cvWaitKey(0);
		   free_image(rgb);
		 */
	}
	free(random_paths);
	return d;
}

data load_data_mask(int n, char **paths, int m, int w, int h, int classes, int boxes, int coords, int min, int max, float angle, float aspect, float hue, float saturation, float exposure)
{
	char **random_paths = get_random_paths(paths, n, m);
	int i;
	data d = { 0 };
	d.shallow = 0;

	d.X.rows = n;
	d.X.vals = (float**)calloc(d.X.rows, sizeof(float*));
	d.X.cols = h * w * 3;

	d.y = make_matrix(n, (coords + 1)*boxes);

	for (i = 0; i < n; ++i) {
		image orig = load_image_color(random_paths[i], 0, 0);
		augment_args a = random_augment_args(orig, angle, aspect, min, max, w, h);
		image sized = rotate_crop_image(orig, a.rad, a.scale, a.w, a.h, a.dx, a.dy, a.aspect);

		int flip = rand() % 2;
		if (flip) flip_image(sized);
		random_distort_image(sized, hue, saturation, exposure);
		d.X.vals[i] = sized.data;
		//show_image(sized, "image");

		fill_truth_mask(random_paths[i], boxes, d.y.vals[i], classes, orig.w, orig.h, a, flip, 14, 14);

		free_image(orig);

		/*
		   image rgb = mask_to_rgb(sized_m, classes);
		   show_image(rgb, "part");
		   show_image(sized, "orig");
		   cvWaitKey(0);
		   free_image(rgb);
		 */
	}
	free(random_paths);
	return d;
}

data load_data_region(int n, char **paths, int m, int w, int h, int size, int classes, float jitter, float hue, float saturation, float exposure)
{
	char **random_paths = get_random_paths(paths, n, m);
	int i;
	data d = { 0 };
	d.shallow = 0;

	d.X.rows = n;
	d.X.vals = (float**)calloc(d.X.rows, sizeof(float*));
	d.X.cols = h * w * 3;


	int k = size * size*(5 + classes);
	d.y = make_matrix(n, k);
	for (i = 0; i < n; ++i) {
		image orig = load_image_color(random_paths[i], 0, 0);

		int oh = orig.h;
		int ow = orig.w;

		int dw = (ow*jitter);
		int dh = (oh*jitter);

		int pleft = rand_uniform(-dw, dw);
		int pright = rand_uniform(-dw, dw);
		int ptop = rand_uniform(-dh, dh);
		int pbot = rand_uniform(-dh, dh);

		int swidth = ow - pleft - pright;
		int sheight = oh - ptop - pbot;

		float sx = (float)swidth / ow;
		float sy = (float)sheight / oh;

		int flip = rand() % 2;
		image cropped = crop_image(orig, pleft, ptop, swidth, sheight);

		float dx = ((float)pleft / ow) / sx;
		float dy = ((float)ptop / oh) / sy;

		image sized = resize_image(cropped, w, h);
		if (flip) flip_image(sized);
		random_distort_image(sized, hue, saturation, exposure);
		d.X.vals[i] = sized.data;

		fill_truth_region(random_paths[i], d.y.vals[i], classes, size, flip, dx, dy, 1. / sx, 1. / sy);

		free_image(orig);
		free_image(cropped);
	}
	free(random_paths);
	return d;
}

data load_data_compare(int n, char **paths, int m, int classes, int w, int h)
{
	if (m) paths = get_random_paths(paths, 2 * n, m);
	int i, j;
	data d = { 0 };
	d.shallow = 0;

	d.X.rows = n;
	d.X.vals = (float**)calloc(d.X.rows, sizeof(float*));
	d.X.cols = h * w * 6;

	int k = 2 * (classes);
	d.y = make_matrix(n, k);
	for (i = 0; i < n; ++i) {
		image im1 = load_image_color(paths[i * 2], w, h);
		image im2 = load_image_color(paths[i * 2 + 1], w, h);

		d.X.vals[i] = (float*)calloc(d.X.cols, sizeof(float));
		memcpy(d.X.vals[i], im1.data, h*w * 3 * sizeof(float));
		memcpy(d.X.vals[i] + h * w * 3, im2.data, h*w * 3 * sizeof(float));

		int id;
		float iou;

		char imlabel1[4096];
		char imlabel2[4096];
		find_replace(paths[i * 2], (char*)"imgs", (char*)"labels", imlabel1);
		find_replace(imlabel1, (char*)"jpg", (char*)"txt", imlabel1);
		FILE *fp1 = fopen(imlabel1, "r");

		while (fscanf(fp1, "%d %f", &id, &iou) == 2) {
			if (d.y.vals[i][2 * id] < iou) d.y.vals[i][2 * id] = iou;
		}

		find_replace(paths[i * 2 + 1], (char*)"imgs", (char*)"labels", imlabel2);
		find_replace(imlabel2, (char*)"jpg", (char*)"txt", imlabel2);
		FILE *fp2 = fopen(imlabel2, "r");

		while (fscanf(fp2, "%d %f", &id, &iou) == 2) {
			if (d.y.vals[i][2 * id + 1] < iou) d.y.vals[i][2 * id + 1] = iou;
		}

		for (j = 0; j < classes; ++j) {
			if (d.y.vals[i][2 * j] > .5 &&  d.y.vals[i][2 * j + 1] < .5) {
				d.y.vals[i][2 * j] = 1;
				d.y.vals[i][2 * j + 1] = 0;
			}
			else if (d.y.vals[i][2 * j] < .5 &&  d.y.vals[i][2 * j + 1] > .5) {
				d.y.vals[i][2 * j] = 0;
				d.y.vals[i][2 * j + 1] = 1;
			}
			else {
				d.y.vals[i][2 * j] = SECRET_NUM;
				d.y.vals[i][2 * j + 1] = SECRET_NUM;
			}
		}
		fclose(fp1);
		fclose(fp2);

		free_image(im1);
		free_image(im2);
	}
	if (m) free(paths);
	return d;
}

data load_data_swag(char **paths, int n, int classes, float jitter)
{
	int index = rand() % n;
	char *random_path = paths[index];

	image orig = load_image_color(random_path, 0, 0);
	int h = orig.h;
	int w = orig.w;

	data d = { 0 };
	d.shallow = 0;
	d.w = w;
	d.h = h;

	d.X.rows = 1;
	d.X.vals = (float**)calloc(d.X.rows, sizeof(float*));
	d.X.cols = h * w * 3;

	int k = (4 + classes) * 90;
	d.y = make_matrix(1, k);

	int dw = w * jitter;
	int dh = h * jitter;

	int pleft = rand_uniform(-dw, dw);
	int pright = rand_uniform(-dw, dw);
	int ptop = rand_uniform(-dh, dh);
	int pbot = rand_uniform(-dh, dh);

	int swidth = w - pleft - pright;
	int sheight = h - ptop - pbot;

	float sx = (float)swidth / w;
	float sy = (float)sheight / h;

	int flip = rand() % 2;
	image cropped = crop_image(orig, pleft, ptop, swidth, sheight);

	float dx = ((float)pleft / w) / sx;
	float dy = ((float)ptop / h) / sy;

	image sized = resize_image(cropped, w, h);
	if (flip) flip_image(sized);
	d.X.vals[0] = sized.data;

	fill_truth_swag(random_path, d.y.vals[0], classes, flip, dx, dy, 1. / sx, 1. / sy);

	free_image(orig);
	free_image(cropped);

	return d;
}

data load_data_detection(int n, char **paths, int m, int w, int h, int boxes, int classes, float jitter, float hue, float saturation, float exposure)
{
	char **random_paths = get_random_paths(paths, n, m);
	int i;
	data d = { 0 };
	d.shallow = 0;

	d.X.rows = n;
	d.X.vals = (float**)calloc(d.X.rows, sizeof(float*));
	d.X.cols = h * w * 3;

	d.y = make_matrix(n, 5 * boxes);
	for (i = 0; i < n; ++i) {
		image orig = load_image_color(random_paths[i], 0, 0);
		image sized = make_image(w, h, orig.c);
		fill_image(sized, .5);

		float dw = jitter * orig.w;
		float dh = jitter * orig.h;

		float new_ar = (orig.w + rand_uniform(-dw, dw)) / (orig.h + rand_uniform(-dh, dh));
		//float scale = rand_uniform(.25, 2);
		float scale = 1;

		float nw, nh;

		if (new_ar < 1) {
			nh = scale * h;
			nw = nh * new_ar;
		}
		else {
			nw = scale * w;
			nh = nw / new_ar;
		}

		float dx = rand_uniform(0, w - nw);
		float dy = rand_uniform(0, h - nh);

		place_image(orig, nw, nh, dx, dy, sized);

		random_distort_image(sized, hue, saturation, exposure);

		int flip = rand() % 2;
		if (flip) flip_image(sized);
		d.X.vals[i] = sized.data;


		fill_truth_detection(random_paths[i], boxes, d.y.vals[i], classes, flip, -dx / w, -dy / h, nw / w, nh / h);

		free_image(orig);
	}
	free(random_paths);
	return d;
}

void *load_thread(void *ptr)
{
	//printf("Loading data: %d\n", rand());
	load_args a = *(struct load_args*)ptr;
	if (a.exposure == 0) a.exposure = 1;
	if (a.saturation == 0) a.saturation = 1;
	if (a.aspect == 0) a.aspect = 1;

	if (a.type == OLD_CLASSIFICATION_DATA) {
		*a.d = load_data_old(a.paths, a.n, a.m, a.labels, a.classes, a.w, a.h);
	}
	else if (a.type == REGRESSION_DATA) {
		*a.d = load_data_regression(a.paths, a.n, a.m, a.classes, a.min, a.max, a.size, a.angle, a.aspect, a.hue, a.saturation, a.exposure);
	}
	else if (a.type == CLASSIFICATION_DATA) {
		*a.d = load_data_augment(a.paths, a.n, a.m, a.labels, a.classes, a.hierarchy, a.min, a.max, a.size, a.angle, a.aspect, a.hue, a.saturation, a.exposure, a.center);
	}
	else if (a.type == SUPER_DATA) {
		*a.d = load_data_super(a.paths, a.n, a.m, a.w, a.h, a.scale);
	}
	else if (a.type == WRITING_DATA) {
		*a.d = load_data_writing(a.paths, a.n, a.m, a.w, a.h, a.out_w, a.out_h);
	}
	else if (a.type == ISEG_DATA) {
		*a.d = load_data_iseg(a.n, a.paths, a.m, a.w, a.h, a.classes, a.num_boxes, a.scale, a.min, a.max, a.angle, a.aspect, a.hue, a.saturation, a.exposure);
	}
	else if (a.type == INSTANCE_DATA) {
		*a.d = load_data_mask(a.n, a.paths, a.m, a.w, a.h, a.classes, a.num_boxes, a.coords, a.min, a.max, a.angle, a.aspect, a.hue, a.saturation, a.exposure);
	}
	else if (a.type == SEGMENTATION_DATA) {
		*a.d = load_data_seg(a.n, a.paths, a.m, a.w, a.h, a.classes, a.min, a.max, a.angle, a.aspect, a.hue, a.saturation, a.exposure, a.scale);
	}
	else if (a.type == REGION_DATA) {
		*a.d = load_data_region(a.n, a.paths, a.m, a.w, a.h, a.num_boxes, a.classes, a.jitter, a.hue, a.saturation, a.exposure);
	}
	else if (a.type == DETECTION_DATA) {
		*a.d = load_data_detection(a.n, a.paths, a.m, a.w, a.h, a.num_boxes, a.classes, a.jitter, a.hue, a.saturation, a.exposure);
	}
	else if (a.type == SWAG_DATA) {
		*a.d = load_data_swag(a.paths, a.n, a.classes, a.jitter);
	}
	else if (a.type == COMPARE_DATA) {
		*a.d = load_data_compare(a.n, a.paths, a.m, a.classes, a.w, a.h);
	}
	else if (a.type == IMAGE_DATA) {
		*(a.im) = load_image_color(a.path, 0, 0);
		*(a.resized) = resize_image(*(a.im), a.w, a.h);
	}
	else if (a.type == LETTERBOX_DATA) {
		*(a.im) = load_image_color(a.path, 0, 0);
		*(a.resized) = letterbox_image(*(a.im), a.w, a.h);
	}
	else if (a.type == TAG_DATA) {
		*a.d = load_data_tag(a.paths, a.n, a.m, a.classes, a.min, a.max, a.size, a.angle, a.aspect, a.hue, a.saturation, a.exposure);
	}
	free(ptr);
	return 0;
}

pthread_t load_data_in_thread(load_args args)
{
	pthread_t thread;
	struct load_args *ptr = (load_args*)calloc(1, sizeof(struct load_args));
	*ptr = args;
	if (pthread_create(&thread, 0, load_thread, ptr)) error("Thread creation failed");
	return thread;
}

void *load_threads(void *ptr)
{
	int i;
	load_args args = *(load_args *)ptr;
	if (args.threads == 0) args.threads = 1;
	data *out = args.d;
	int total = args.n;
	free(ptr);
	data *buffers = (data*)calloc(args.threads, sizeof(data));
	pthread_t *threads = (pthread_t*)calloc(args.threads, sizeof(pthread_t));
	for (i = 0; i < args.threads; ++i) {
		args.d = buffers + i;
		args.n = (i + 1) * total / args.threads - i * total / args.threads;
		threads[i] = load_data_in_thread(args);
	}
	for (i = 0; i < args.threads; ++i) {
		pthread_join(threads[i], 0);
	}
	*out = concat_datas(buffers, args.threads);
	out->shallow = 0;
	for (i = 0; i < args.threads; ++i) {
		buffers[i].shallow = 1;
		free_data(buffers[i]);
	}
	free(buffers);
	free(threads);
	return 0;
}

void load_data_blocking(load_args args)
{
	struct load_args *ptr = (load_args*)calloc(1, sizeof(struct load_args));
	*ptr = args;
	load_thread(ptr);
}

pthread_t load_data(load_args args)
{
	pthread_t thread;
	struct load_args *ptr = (load_args*)calloc(1, sizeof(struct load_args));
	*ptr = args;
	if (pthread_create(&thread, 0, load_threads, ptr)) error("Thread creation failed");
	return thread;
}

data load_data_writing(char **paths, int n, int m, int w, int h, int out_w, int out_h)
{
	if (m) paths = get_random_paths(paths, n, m);
	char **replace_paths = find_replace_paths(paths, n, (char*)".png", (char*)"-label.png");
	data d = { 0 };
	d.shallow = 0;
	d.X = load_image_paths(paths, n, w, h);
	d.y = load_image_paths_gray(replace_paths, n, out_w, out_h);
	if (m) free(paths);
	int i;
	for (i = 0; i < n; ++i) free(replace_paths[i]);
	free(replace_paths);
	return d;
}

data load_data_old(char **paths, int n, int m, char **labels, int k, int w, int h)
{
	if (m) paths = get_random_paths(paths, n, m);
	data d = { 0 };
	d.shallow = 0;
	d.X = load_image_paths(paths, n, w, h);
	d.y = load_labels_paths(paths, n, labels, k, 0);
	if (m) free(paths);
	return d;
}

/*
   data load_data_study(char **paths, int n, int m, char **labels, int k, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure)
   {
   data d = {0};
   d.indexes = calloc(n, sizeof(int));
   if(m) paths = get_random_paths_indexes(paths, n, m, d.indexes);
   d.shallow = 0;
   d.X = load_image_augment_paths(paths, n, min, max, size, angle, aspect, hue, saturation, exposure);
   d.y = load_labels_paths(paths, n, labels, k);
   if(m) free(paths);
   return d;
   }
 */

data load_data_super(char **paths, int n, int m, int w, int h, int scale)
{
	if (m) paths = get_random_paths(paths, n, m);
	data d = { 0 };
	d.shallow = 0;

	int i;
	d.X.rows = n;
	d.X.vals = (float**)calloc(n, sizeof(float*));
	d.X.cols = w * h * 3;

	d.y.rows = n;
	d.y.vals = (float**)calloc(n, sizeof(float*));
	d.y.cols = w * scale * h*scale * 3;

	for (i = 0; i < n; ++i) {
		image im = load_image_color(paths[i], 0, 0);
		image crop = random_crop_image(im, w*scale, h*scale);
		int flip = rand() % 2;
		if (flip) flip_image(crop);
		image resize = resize_image(crop, w, h);
		d.X.vals[i] = resize.data;
		d.y.vals[i] = crop.data;
		free_image(im);
	}

	if (m) free(paths);
	return d;
}

data load_data_regression(char **paths, int n, int m, int k, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure)
{
	if (m) paths = get_random_paths(paths, n, m);
	data d = { 0 };
	d.shallow = 0;
	d.X = load_image_augment_paths(paths, n, min, max, size, angle, aspect, hue, saturation, exposure, 0);
	d.y = load_regression_labels_paths(paths, n, k);
	if (m) free(paths);
	return d;
}

data select_data(data *orig, int *inds)
{
	data d = { 0 };
	d.shallow = 1;
	d.w = orig[0].w;
	d.h = orig[0].h;

	d.X.rows = orig[0].X.rows;
	d.y.rows = orig[0].X.rows;

	d.X.cols = orig[0].X.cols;
	d.y.cols = orig[0].y.cols;

	d.X.vals = (float**)calloc(orig[0].X.rows, sizeof(float *));
	d.y.vals = (float**)calloc(orig[0].y.rows, sizeof(float *));
	int i;
	for (i = 0; i < d.X.rows; ++i) {
		d.X.vals[i] = orig[inds[i]].X.vals[i];
		d.y.vals[i] = orig[inds[i]].y.vals[i];
	}
	return d;
}

data *tile_data(data orig, int divs, int size)
{
	data *ds = (data*)calloc(divs*divs, sizeof(data));
	int i, j;
#pragma omp parallel for
	for (i = 0; i < divs*divs; ++i) {
		data d;
		d.shallow = 0;
		d.w = orig.w / divs * size;
		d.h = orig.h / divs * size;
		d.X.rows = orig.X.rows;
		d.X.cols = d.w*d.h * 3;
		d.X.vals = (float**)calloc(d.X.rows, sizeof(float*));

		d.y = copy_matrix(orig.y);
#pragma omp parallel for
		for (j = 0; j < orig.X.rows; ++j) {
			int x = (i%divs) * orig.w / divs - (d.w - orig.w / divs) / 2;
			int y = (i / divs) * orig.h / divs - (d.h - orig.h / divs) / 2;
			image im = float_to_image(orig.w, orig.h, 3, orig.X.vals[j]);
			d.X.vals[j] = crop_image(im, x, y, d.w, d.h).data;
		}
		ds[i] = d;
	}
	return ds;
}

data resize_data(data orig, int w, int h)
{
	data d = { 0 };
	d.shallow = 0;
	d.w = w;
	d.h = h;
	int i;
	d.X.rows = orig.X.rows;
	d.X.cols = w * h * 3;
	d.X.vals = (float**)calloc(d.X.rows, sizeof(float*));

	d.y = copy_matrix(orig.y);
#pragma omp parallel for
	for (i = 0; i < orig.X.rows; ++i) {
		image im = float_to_image(orig.w, orig.h, 3, orig.X.vals[i]);
		d.X.vals[i] = resize_image(im, w, h).data;
	}
	return d;
}

data load_data_augment(char **paths, int n, int m, char **labels, int k, tree *hierarchy, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure, int center)
{
	if (m) paths = get_random_paths(paths, n, m);
	data d = { 0 };
	d.shallow = 0;
	d.w = size;
	d.h = size;
	d.X = load_image_augment_paths(paths, n, min, max, size, angle, aspect, hue, saturation, exposure, center);
	d.y = load_labels_paths(paths, n, labels, k, hierarchy);
	if (m) free(paths);
	return d;
}

data load_data_tag(char **paths, int n, int m, int k, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure)
{
	if (m) paths = get_random_paths(paths, n, m);
	data d = { 0 };
	d.w = size;
	d.h = size;
	d.shallow = 0;
	d.X = load_image_augment_paths(paths, n, min, max, size, angle, aspect, hue, saturation, exposure, 0);
	d.y = load_tags_paths(paths, n, k);
	if (m) free(paths);
	return d;
}

matrix concat_matrix(matrix m1, matrix m2)
{
	int i, count = 0;
	matrix m;
	m.cols = m1.cols;
	m.rows = m1.rows + m2.rows;
	m.vals = (float**)calloc(m1.rows + m2.rows, sizeof(float*));
	for (i = 0; i < m1.rows; ++i) {
		m.vals[count++] = m1.vals[i];
	}
	for (i = 0; i < m2.rows; ++i) {
		m.vals[count++] = m2.vals[i];
	}
	return m;
}

data concat_data(data d1, data d2)
{
	data d = { 0 };
	d.shallow = 1;
	d.X = concat_matrix(d1.X, d2.X);
	d.y = concat_matrix(d1.y, d2.y);
	d.w = d1.w;
	d.h = d1.h;
	return d;
}

data concat_datas(data *d, int n)
{
	int i;
	data out = { 0 };
	for (i = 0; i < n; ++i) {
		data new_data = concat_data(d[i], out);
		free_data(out);
		out = new_data;
	}
	return out;
}

data load_categorical_data_csv(char *filename, int target, int k)
{
	data d = { 0 };
	d.shallow = 0;
	matrix X = csv_to_matrix(filename);
	float *truth_1d = pop_column(&X, target);
	float **truth = one_hot_encode(truth_1d, X.rows, k);
	matrix y;
	y.rows = X.rows;
	y.cols = k;
	y.vals = truth;
	d.X = X;
	d.y = y;
	free(truth_1d);
	return d;
}

data load_cifar10_data(char *filename)
{
	data d = { 0 };
	d.shallow = 0;
	long i, j;
	matrix X = make_matrix(10000, 3072);
	matrix y = make_matrix(10000, 10);
	d.X = X;
	d.y = y;

	FILE *fp = fopen(filename, "rb");
	if (!fp) file_error(filename);
	for (i = 0; i < 10000; ++i) {
		unsigned char bytes[3073];
		fread(bytes, 1, 3073, fp);
		int class_n = bytes[0];
		y.vals[i][class_n] = 1;
		for (j = 0; j < X.cols; ++j) {
			X.vals[i][j] = (double)bytes[j + 1];
		}
	}
	scale_data_rows(d, 1. / 255);
	//normalize_data_rows(d);
	fclose(fp);
	return d;
}

void get_random_batch(data d, int n, float *X, float *y)
{
	int j;
	for (j = 0; j < n; ++j) {
		int index = rand() % d.X.rows;
		memcpy(X + j * d.X.cols, d.X.vals[index], d.X.cols * sizeof(float));
		memcpy(y + j * d.y.cols, d.y.vals[index], d.y.cols * sizeof(float));
	}
}

void get_next_batch(data d, int n, int offset, float *X, float *y)
{
	int j;
	for (j = 0; j < n; ++j) {
		int index = offset + j;
		memcpy(X + j * d.X.cols, d.X.vals[index], d.X.cols * sizeof(float));
		if (y) memcpy(y + j * d.y.cols, d.y.vals[index], d.y.cols * sizeof(float));
	}
}

void smooth_data(data d)
{
	int i, j;
	float scale = 1. / d.y.cols;
	float eps = .1;
	for (i = 0; i < d.y.rows; ++i) {
		for (j = 0; j < d.y.cols; ++j) {
			d.y.vals[i][j] = eps * scale + (1 - eps) * d.y.vals[i][j];
		}
	}
}

data load_all_cifar10()
{
	data d = { 0 };
	d.shallow = 0;
	int i, j, b;
	matrix X = make_matrix(50000, 3072);
	matrix y = make_matrix(50000, 10);
	d.X = X;
	d.y = y;


	for (b = 0; b < 5; ++b) {
		char buff[256];
		sprintf(buff, "data/cifar/cifar-10-batches-bin/data_batch_%d.bin", b + 1);
		FILE *fp = fopen(buff, "rb");
		if (!fp) file_error(buff);
		for (i = 0; i < 10000; ++i) {
			unsigned char bytes[3073];
			fread(bytes, 1, 3073, fp);
			int class_n = bytes[0];
			y.vals[i + b * 10000][class_n] = 1;
			for (j = 0; j < X.cols; ++j) {
				X.vals[i + b * 10000][j] = (double)bytes[j + 1];
			}
		}
		fclose(fp);
	}
	//normalize_data_rows(d);
	scale_data_rows(d, 1. / 255);
	smooth_data(d);
	return d;
}

data load_go(char *filename)
{
	FILE *fp = fopen(filename, "rb");
	matrix X = make_matrix(3363059, 361);
	matrix y = make_matrix(3363059, 361);
	int row, col;

	if (!fp) file_error(filename);
	char *label;
	int count = 0;
	while ((label = fgetl(fp))) {
		int i;
		if (count == X.rows) {
			X = resize_matrix(X, count * 2);
			y = resize_matrix(y, count * 2);
		}
		sscanf(label, "%d %d", &row, &col);
		char *board = fgetl(fp);

		int index = row * 19 + col;
		y.vals[count][index] = 1;

		for (i = 0; i < 19 * 19; ++i) {
			float val = 0;
			if (board[i] == '1') val = 1;
			else if (board[i] == '2') val = -1;
			X.vals[count][i] = val;
		}
		++count;
		free(label);
		free(board);
	}
	X = resize_matrix(X, count);
	y = resize_matrix(y, count);

	data d = { 0 };
	d.shallow = 0;
	d.X = X;
	d.y = y;


	fclose(fp);

	return d;
}


void randomize_data(data d)
{
	int i;
	for (i = d.X.rows - 1; i > 0; --i) {
		int index = rand() % i;
		float *swap = d.X.vals[index];
		d.X.vals[index] = d.X.vals[i];
		d.X.vals[i] = swap;

		swap = d.y.vals[index];
		d.y.vals[index] = d.y.vals[i];
		d.y.vals[i] = swap;
	}
}

void scale_data_rows(data d, float s)
{
	int i;
	for (i = 0; i < d.X.rows; ++i) {
		scale_array(d.X.vals[i], d.X.cols, s);
	}
}

void translate_data_rows(data d, float s)
{
	int i;
	for (i = 0; i < d.X.rows; ++i) {
		translate_array(d.X.vals[i], d.X.cols, s);
	}
}

data copy_data(data d)
{
	data c = { 0 };
	c.w = d.w;
	c.h = d.h;
	c.shallow = 0;
	c.num_boxes = d.num_boxes;
	c.boxes = d.boxes;
	c.X = copy_matrix(d.X);
	c.y = copy_matrix(d.y);
	return c;
}

void normalize_data_rows(data d)
{
	int i;
	for (i = 0; i < d.X.rows; ++i) {
		normalize_array(d.X.vals[i], d.X.cols);
	}
}

data get_data_part(data d, int part, int total)
{
	data p = { 0 };
	p.shallow = 1;
	p.X.rows = d.X.rows * (part + 1) / total - d.X.rows * part / total;
	p.y.rows = d.y.rows * (part + 1) / total - d.y.rows * part / total;
	p.X.cols = d.X.cols;
	p.y.cols = d.y.cols;
	p.X.vals = d.X.vals + d.X.rows * part / total;
	p.y.vals = d.y.vals + d.y.rows * part / total;
	return p;
}

data get_random_data(data d, int num)
{
	data r = { 0 };
	r.shallow = 1;

	r.X.rows = num;
	r.y.rows = num;

	r.X.cols = d.X.cols;
	r.y.cols = d.y.cols;

	r.X.vals = (float**)calloc(num, sizeof(float *));
	r.y.vals = (float**)calloc(num, sizeof(float *));

	int i;
	for (i = 0; i < num; ++i) {
		int index = rand() % d.X.rows;
		r.X.vals[i] = d.X.vals[index];
		r.y.vals[i] = d.y.vals[index];
	}
	return r;
}

data *split_data(data d, int part, int total)
{
	data *split = (data*)calloc(2, sizeof(data));
	int i;
	int start = part * d.X.rows / total;
	int end = (part + 1)*d.X.rows / total;
	data train;
	data test;
	train.shallow = test.shallow = 1;

	test.X.rows = test.y.rows = end - start;
	train.X.rows = train.y.rows = d.X.rows - (end - start);
	train.X.cols = test.X.cols = d.X.cols;
	train.y.cols = test.y.cols = d.y.cols;

	train.X.vals = (float**)calloc(train.X.rows, sizeof(float*));
	test.X.vals = (float**)calloc(test.X.rows, sizeof(float*));
	train.y.vals = (float**)calloc(train.y.rows, sizeof(float*));
	test.y.vals = (float**)calloc(test.y.rows, sizeof(float*));

	for (i = 0; i < start; ++i) {
		train.X.vals[i] = d.X.vals[i];
		train.y.vals[i] = d.y.vals[i];
	}
	for (i = start; i < end; ++i) {
		test.X.vals[i - start] = d.X.vals[i];
		test.y.vals[i - start] = d.y.vals[i];
	}
	for (i = end; i < d.X.rows; ++i) {
		train.X.vals[i - (end - start)] = d.X.vals[i];
		train.y.vals[i - (end - start)] = d.y.vals[i];
	}
	split[0] = train;
	split[1] = test;
	return split;
}






//image_opencv.cpp
#ifdef CV_VERSION
namespace cv
{

//	extern "C" {

		IplImage *image_to_ipl(image im)
		{
			int x, y, c;
			IplImage *disp = cvCreateImage(cvSize(im.w, im.h), IPL_DEPTH_8U, im.c);
			int step = disp->widthStep;
			for (y = 0; y < im.h; ++y) {
				for (x = 0; x < im.w; ++x) {
					for (c = 0; c < im.c; ++c) {
						float val = im.data[c*im.h*im.w + y * im.w + x];
						disp->imageData[y*step + x * im.c + c] = (unsigned char)(val * 255);
					}
				}
			}
			return disp;
		}

		image ipl_to_image(IplImage* src)
		{
			int h = src->height;
			int w = src->width;
			int c = src->nChannels;
			image im = make_image(w, h, c);
			unsigned char *data = (unsigned char *)src->imageData;
			int step = src->widthStep;
			int i, j, k;

			for (i = 0; i < h; ++i) {
				for (k = 0; k < c; ++k) {
					for (j = 0; j < w; ++j) {
						im.data[k*w*h + i * w + j] = data[i*step + j * c + k] / 255.;
					}
				}
			}
			return im;
		}

		Mat image_to_mat(image im)
		{
			image copy = copy_image(im);
			constrain_image(copy);
			if (im.c == 3) rgbgr_image(copy);

			IplImage *ipl = image_to_ipl(copy);
			Mat m = cvarrToMat(ipl, true);
			cvReleaseImage(&ipl);
			free_image(copy);
			return m;
		}

		image mat_to_image(Mat m)
		{
			IplImage ipl = m;
			image im = ipl_to_image(&ipl);
			rgbgr_image(im);
			return im;
		}

		void *open_video_stream(const char *f, int c, int w, int h, int fps)
		{
			VideoCapture *cap;
			if (f) cap = new VideoCapture(f);
			else cap = new VideoCapture(c);
			if (!cap->isOpened()) return 0;
			if (w) cap->set(CV_CAP_PROP_FRAME_WIDTH, w);
			if (h) cap->set(CV_CAP_PROP_FRAME_HEIGHT, w);
			if (fps) cap->set(CV_CAP_PROP_FPS, w);
			return (void *)cap;
		}

		image get_image_from_stream(void *p)
		{
			VideoCapture *cap = (VideoCapture *)p;
			Mat m;
			*cap >> m;
			if (m.empty()) return make_empty_image(0, 0, 0);
			return mat_to_image(m);
		}

		image load_image_cv(char *filename, int channels)
		{
			int flag = -1;
			if (channels == 0) flag = -1;
			else if (channels == 1) flag = 0;
			else if (channels == 3) flag = 1;
			else {
				fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);
			}
			Mat m;
			m = imread(filename, flag);
			if (!m.data) {
				fprintf(stderr, "Cannot load image \"%s\"\n", filename);
				char buff[256];
				sprintf(buff, "echo %s >> bad.list", filename);
				system(buff);
				return make_image(10, 10, 3);
				//exit(0);
			}
			image im = mat_to_image(m);
			return im;
		}

		int show_image_cv(image im, const char* name, int ms)
		{
			Mat m = image_to_mat(im);
			imshow(name, m);
			int c = waitKey(ms);
			if (c != -1) c = c % 256;
			return c;
		}

		void make_window(char *name, int w, int h, int fullscreen)
		{
			namedWindow(name, WINDOW_NORMAL);
			if (fullscreen) {
				setWindowProperty(name, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
			}
			else {
				resizeWindow(name, w, h);
				if (strcmp(name, "Demo") == 0) moveWindow(name, 0, 0);
			}
		}

//	}
}
#endif





//demo.c
#define DEMO 1

#ifdef CV_VERSION
detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num);

int size_network(network *net)
{
	int i;
	int count = 0;
	for (i = 0; i < net->n; ++i) {
		layer l = net->layers[i];
		if (l.type == YOLO || l.type == REGION || l.type == DETECTION) {
			count += l.outputs;
		}
	}
	return count;
}

void remember_network(network *net)
{
	int i;
	int count = 0;
	for (i = 0; i < net->n; ++i) {
		layer l = net->layers[i];
		if (l.type == YOLO || l.type == REGION || l.type == DETECTION) {
			memcpy(predictions[demo_index] + count, net->layers[i].output, sizeof(float) * l.outputs);
			count += l.outputs;
		}
	}
}

detection *avg_predictions(network *net, int *nboxes)
{
	int i, j;
	int count = 0;
	fill_cpu(demo_total, 0, avg, 1);
	for (j = 0; j < demo_frame; ++j) {
		axpy_cpu(demo_total, 1. / demo_frame, predictions[j], 1, avg, 1);
	}
	for (i = 0; i < net->n; ++i) {
		layer l = net->layers[i];
		if (l.type == YOLO || l.type == REGION || l.type == DETECTION) {
			memcpy(l.output, avg + count, sizeof(float) * l.outputs);
			count += l.outputs;
		}
	}
	detection *dets = get_network_boxes(net, buff[0].w, buff[0].h, demo_thresh, demo_hier, 0, 1, nboxes);
	return dets;
}

void *detect_in_thread(void *ptr)
{
	running = 1;
	float nms = .4;

	layer l = net->layers[net->n - 1];
	float *X = buff_letter[(buff_index + 2) % 3].data;
	network_predict(net, X);

	/*
	   if(l.type == DETECTION){
	   get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
	   } else */
	remember_network(net);
	detection *dets = 0;
	int nboxes = 0;
	dets = avg_predictions(net, &nboxes);


	/*
	   int i,j;
	   box zero = {0};
	   int classes = l.classes;
	   for(i = 0; i < demo_detections; ++i){
	   avg[i].objectness = 0;
	   avg[i].bbox = zero;
	   memset(avg[i].prob, 0, classes*sizeof(float));
	   for(j = 0; j < demo_frame; ++j){
	   axpy_cpu(classes, 1./demo_frame, dets[j][i].prob, 1, avg[i].prob, 1);
	   avg[i].objectness += dets[j][i].objectness * 1./demo_frame;
	   avg[i].bbox.x += dets[j][i].bbox.x * 1./demo_frame;
	   avg[i].bbox.y += dets[j][i].bbox.y * 1./demo_frame;
	   avg[i].bbox.w += dets[j][i].bbox.w * 1./demo_frame;
	   avg[i].bbox.h += dets[j][i].bbox.h * 1./demo_frame;
	   }
	//copy_cpu(classes, dets[0][i].prob, 1, avg[i].prob, 1);
	//avg[i].objectness = dets[0][i].objectness;
	}
	 */

	if (nms > 0) do_nms_obj(dets, nboxes, l.classes, nms);

	printf("\033[2J");
	printf("\033[1;1H");
	printf("\nFPS:%.1f\n", fps);
	printf("Objects:\n\n");
	image display = buff[(buff_index + 2) % 3];
	draw_detections(display, dets, nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes);
	free_detections(dets, nboxes);

	demo_index = (demo_index + 1) % demo_frame;
	running = 0;
	return 0;
}

void *fetch_in_thread(void *ptr)
{
	free_image(buff[buff_index]);
	buff[buff_index] = cv::get_image_from_stream(cap);
	if (buff[buff_index].data == 0) {
		demo_done = 1;
		return 0;
	}
	letterbox_image_into(buff[buff_index], net->w, net->h, buff_letter[buff_index]);
	return 0;
}

void *display_in_thread(void *ptr)
{
	int c = show_image(buff[(buff_index + 1) % 3], "Demo", 1);
	if (c != -1) c = c % 256;
	if (c == 27) {
		demo_done = 1;
		return 0;
	}
	else if (c == 82) {
		demo_thresh += .02;
	}
	else if (c == 84) {
		demo_thresh -= .02;
		if (demo_thresh <= .02) demo_thresh = .02;
	}
	else if (c == 83) {
		demo_hier += .02;
	}
	else if (c == 81) {
		demo_hier -= .02;
		if (demo_hier <= .0) demo_hier = .0;
	}
	return 0;
}

void *display_loop(void *ptr)
{
	while (1) {
		display_in_thread(0);
	}
}

void *detect_loop(void *ptr)
{
	while (1) {
		detect_in_thread(0);
	}
}

void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
{
	//demo_frame = avg_frames;
	image **alphabet = load_alphabet();
	demo_names = names;
	demo_alphabet = alphabet;
	demo_classes = classes;
	demo_thresh = thresh;
	demo_hier = hier;
	printf("Demo\n");
	net = load_network(cfgfile, weightfile, 0);
	set_batch_network(net, 1);
	pthread_t detect_thread;
	pthread_t fetch_thread;

	srand(2222222);

	int i;
	demo_total = size_network(net);
	predictions = (float**)calloc(demo_frame, sizeof(float*));
	for (i = 0; i < demo_frame; ++i) {
		predictions[i] = (float*)calloc(demo_total, sizeof(float));
	}
	avg = (float*)calloc(demo_total, sizeof(float));

	if (filename) {
		printf("video file: %s\n", filename);
		cap = cv::open_video_stream(filename, 0, 0, 0, 0);
	}
	else {
		cap = cv::open_video_stream(0, cam_index, w, h, frames);
	}

	if (!cap) error("Couldn't connect to webcam.\n");

	buff[0] = cv::get_image_from_stream(cap);
	buff[1] = copy_image(buff[0]);
	buff[2] = copy_image(buff[0]);
	buff_letter[0] = letterbox_image(buff[0], net->w, net->h);
	buff_letter[1] = letterbox_image(buff[0], net->w, net->h);
	buff_letter[2] = letterbox_image(buff[0], net->w, net->h);

	int count = 0;
	if (!prefix) {
		cv::make_window((char*)"Demo", 1352, 1013, fullscreen);
	}

	auto start = std::chrono::system_clock::now();


	while (!demo_done) {
		buff_index = (buff_index + 1) % 3;
		if (pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
		if (pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");
		if (!prefix) {
			auto end = std::chrono::system_clock::now();
			std::chrono::duration<double> elapsed_seconds = end - start;
			fps = 1. / elapsed_seconds.count();

			start = std::chrono::system_clock::now();
			display_in_thread(0);
		}
		else {
			char name[256];
			sprintf(name, "%s_%08d", prefix, count);
			save_image(buff[(buff_index + 1) % 3], name);
		}
		pthread_join(fetch_thread, 0);
		pthread_join(detect_thread, 0);
		++count;
	}
}

/*
   void demo_compare(char *cfg1, char *weight1, char *cfg2, char *weight2, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
   {
   demo_frame = avg_frames;
   predictions = calloc(demo_frame, sizeof(float*));
   image **alphabet = load_alphabet();
   demo_names = names;
   demo_alphabet = alphabet;
   demo_classes = classes;
   demo_thresh = thresh;
   demo_hier = hier;
   printf("Demo\n");
   net = load_network(cfg1, weight1, 0);
   set_batch_network(net, 1);
   pthread_t detect_thread;
   pthread_t fetch_thread;

   srand(2222222);

   if(filename){
   printf("video file: %s\n", filename);
   cap = cvCaptureFromFile(filename);
   }else{
   cap = cvCaptureFromCAM(cam_index);

   if(w){
   cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
   }
   if(h){
   cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
   }
   if(frames){
   cvSetCaptureProperty(cap, CV_CAP_PROP_FPS, frames);
   }
   }

   if(!cap) error("Couldn't connect to webcam.\n");

   layer l = net->layers[net->n-1];
   demo_detections = l.n*l.w*l.h;
   int j;

   avg = (float *) calloc(l.outputs, sizeof(float));
   for(j = 0; j < demo_frame; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));

   boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
   probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
   for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes+1, sizeof(float));

   buff[0] = get_image_from_stream(cap);
   buff[1] = copy_image(buff[0]);
   buff[2] = copy_image(buff[0]);
   buff_letter[0] = letterbox_image(buff[0], net->w, net->h);
   buff_letter[1] = letterbox_image(buff[0], net->w, net->h);
   buff_letter[2] = letterbox_image(buff[0], net->w, net->h);
   ipl = cvCreateImage(cvSize(buff[0].w,buff[0].h), IPL_DEPTH_8U, buff[0].c);

   int count = 0;
   if(!prefix){
   cvNamedWindow("Demo", CV_WINDOW_NORMAL);
   if(fullscreen){
   cvSetWindowProperty("Demo", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
   } else {
   cvMoveWindow("Demo", 0, 0);
   cvResizeWindow("Demo", 1352, 1013);
   }
   }

   demo_time = what_time_is_it_now();

   while(!demo_done){
buff_index = (buff_index + 1) %3;
if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");
if(!prefix){
	fps = 1./(what_time_is_it_now() - demo_time);
	demo_time = what_time_is_it_now();
	display_in_thread(0);
}else{
	char name[256];
	sprintf(name, "%s_%08d", prefix, count);
	save_image(buff[(buff_index + 1)%3], name);
}
pthread_join(fetch_thread, 0);
pthread_join(detect_thread, 0);
++count;
}
}
*/
#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg, float hier, int w, int h, int frames, int fullscreen)
{
	fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif








//gemm.c
void gemm_bin(int M, int N, int K, float ALPHA,
	char  *A, int lda,
	float *B, int ldb,
	float *C, int ldc)
{
	int i, j, k;
	for (i = 0; i < M; ++i) {
		for (k = 0; k < K; ++k) {
			char A_PART = A[i*lda + k];
			if (A_PART) {
				for (j = 0; j < N; ++j) {
					C[i*ldc + j] += B[k*ldb + j];
				}
			}
			else {
				for (j = 0; j < N; ++j) {
					C[i*ldc + j] -= B[k*ldb + j];
				}
			}
		}
	}
}

float *random_matrix(int rows, int cols)
{
	int i;
	float *m = (float*)calloc(rows*cols, sizeof(float));
	for (i = 0; i < rows*cols; ++i) {
		m[i] = (float)rand() / RAND_MAX;
	}
	return m;
}

void time_random_matrix(int TA, int TB, int m, int k, int n)
{
	float *a;
	if (!TA) a = random_matrix(m, k);
	else a = random_matrix(k, m);
	int lda = (!TA) ? k : m;
	float *b;
	if (!TB) b = random_matrix(k, n);
	else b = random_matrix(n, k);
	int ldb = (!TB) ? n : k;

	float *c = random_matrix(m, n);
	int i;
	clock_t start = clock(), end;
	for (i = 0; i < 10; ++i) {
		gemm_cpu(TA, TB, m, n, k, 1, a, lda, b, ldb, 1, c, n);
	}
	end = clock();
	printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf ms\n", m, k, k, n, TA, TB, (float)(end - start) / CLOCKS_PER_SEC);
	free(a);
	free(b);
	free(c);
}


void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
	float *A, int lda,
	float *B, int ldb,
	float BETA,
	float *C, int ldc)
{
	gemm_cpu(TA, TB, M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
}

void gemm_nn(int M, int N, int K, float ALPHA,
	float *A, int lda,
	float *B, int ldb,
	float *C, int ldc)
{
	int i, j, k;
#pragma omp parallel for
	for (i = 0; i < M; ++i) {
		for (k = 0; k < K; ++k) {
			register float A_PART = ALPHA * A[i*lda + k];
			for (j = 0; j < N; ++j) {
				C[i*ldc + j] += A_PART * B[k*ldb + j];
			}
		}
	}
}

void gemm_nt(int M, int N, int K, float ALPHA,
	float *A, int lda,
	float *B, int ldb,
	float *C, int ldc)
{
	int i, j, k;
#pragma omp parallel for
	for (i = 0; i < M; ++i) {
		for (j = 0; j < N; ++j) {
			register float sum = 0;
			for (k = 0; k < K; ++k) {
				sum += ALPHA * A[i*lda + k] * B[j*ldb + k];
			}
			C[i*ldc + j] += sum;
		}
	}
}

void gemm_tn(int M, int N, int K, float ALPHA,
	float *A, int lda,
	float *B, int ldb,
	float *C, int ldc)
{
	int i, j, k;
#pragma omp parallel for
	for (i = 0; i < M; ++i) {
		for (k = 0; k < K; ++k) {
			register float A_PART = ALPHA * A[k*lda + i];
			for (j = 0; j < N; ++j) {
				C[i*ldc + j] += A_PART * B[k*ldb + j];
			}
		}
	}
}

void gemm_tt(int M, int N, int K, float ALPHA,
	float *A, int lda,
	float *B, int ldb,
	float *C, int ldc)
{
	int i, j, k;
#pragma omp parallel for
	for (i = 0; i < M; ++i) {
		for (j = 0; j < N; ++j) {
			register float sum = 0;
			for (k = 0; k < K; ++k) {
				sum += ALPHA * A[i + k * lda] * B[k + j * ldb];
			}
			C[i*ldc + j] += sum;
		}
	}
}


void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
	float *A, int lda,
	float *B, int ldb,
	float BETA,
	float *C, int ldc)
{
	//printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
	int i, j;
	for (i = 0; i < M; ++i) {
		for (j = 0; j < N; ++j) {
			C[i*ldc + j] *= BETA;
		}
	}
	if (!TA && !TB)
		gemm_nn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
	else if (TA && !TB)
		gemm_tn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
	else if (!TA && TB)
		gemm_nt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
	else
		gemm_tt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
}

#ifdef GPU

#include <math.h>

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA,
	float *A_gpu, int lda,
	float *B_gpu, int ldb,
	float BETA,
	float *C_gpu, int ldc)
{
	cublasHandle_t handle = blas_handle();
	cudaError_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
		(TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
	check_error(status);
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void time_gpu_random_matrix(int TA, int TB, int m, int k, int n)
{
	float *a;
	if (!TA) a = random_matrix(m, k);
	else a = random_matrix(k, m);
	int lda = (!TA) ? k : m;
	float *b;
	if (!TB) b = random_matrix(k, n);
	else b = random_matrix(n, k);
	int ldb = (!TB) ? n : k;

	float *c = random_matrix(m, n);
	int i;
	clock_t start = clock(), end;
	for (i = 0; i < 32; ++i) {
		gemm_gpu(TA, TB, m, n, k, 1, a, lda, b, ldb, 1, c, n);
	}
	end = clock();
	printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s\n", m, k, k, n, TA, TB, (float)(end - start) / CLOCKS_PER_SEC);
	free(a);
	free(b);
	free(c);
}

void time_gpu(int TA, int TB, int m, int k, int n)
{
	int iter = 10;
	float *a = random_matrix(m, k);
	float *b = random_matrix(k, n);

	int lda = (!TA) ? k : m;
	int ldb = (!TB) ? n : k;

	float *c = random_matrix(m, n);

	float *a_cl = cuda_make_array(a, m*k);
	float *b_cl = cuda_make_array(b, k*n);
	float *c_cl = cuda_make_array(c, m*n);

	int i;
	clock_t start = clock(), end;
	for (i = 0; i < iter; ++i) {
		gemm_gpu(TA, TB, m, n, k, 1, a_cl, lda, b_cl, ldb, 1, c_cl, n);
		cudaThreadSynchronize();
	}
	double flop = ((double)m)*n*(2.*k + 2.)*iter;
	double gflop = flop / pow(10., 9);
	end = clock();
	double seconds = sec(end - start);
	printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s, %lf GFLOPS\n", m, k, k, n, TA, TB, seconds, gflop / seconds);
	cuda_free(a_cl);
	cuda_free(b_cl);
	cuda_free(c_cl);
	free(a);
	free(b);
	free(c);
}


void test_gpu_accuracy(int TA, int TB, int m, int k, int n)
{
	srand(0);
	float *a;
	if (!TA) a = random_matrix(m, k);
	else a = random_matrix(k, m);
	int lda = (!TA) ? k : m;
	float *b;
	if (!TB) b = random_matrix(k, n);
	else b = random_matrix(n, k);
	int ldb = (!TB) ? n : k;

	float *c = random_matrix(m, n);
	float *c_gpu = random_matrix(m, n);
	memset(c, 0, m*n * sizeof(float));
	memset(c_gpu, 0, m*n * sizeof(float));
	int i;
	//pm(m,k,b);
	gemm_gpu(TA, TB, m, n, k, 1, a, lda, b, ldb, 1, c_gpu, n);
	//printf("GPU\n");
	//pm(m, n, c_gpu);

	gemm_cpu(TA, TB, m, n, k, 1, a, lda, b, ldb, 1, c, n);
	//printf("\n\nCPU\n");
	//pm(m, n, c);
	double sse = 0;
	for (i = 0; i < m*n; ++i) {
		//printf("%f %f\n", c[i], c_gpu[i]);
		sse += pow(c[i] - c_gpu[i], 2);
	}
	printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %g SSE\n", m, k, k, n, TA, TB, sse / (m*n));
	free(a);
	free(b);
	free(c);
	free(c_gpu);
}

int test_gpu_blas()
{
	/*
	   test_gpu_accuracy(0,0,10,576,75);

	   test_gpu_accuracy(0,0,17,10,10);
	   test_gpu_accuracy(1,0,17,10,10);
	   test_gpu_accuracy(0,1,17,10,10);
	   test_gpu_accuracy(1,1,17,10,10);

	   test_gpu_accuracy(0,0,1000,10,100);
	   test_gpu_accuracy(1,0,1000,10,100);
	   test_gpu_accuracy(0,1,1000,10,100);
	   test_gpu_accuracy(1,1,1000,10,100);

	   test_gpu_accuracy(0,0,10,10,10);

	   time_gpu(0,0,64,2916,363);
	   time_gpu(0,0,64,2916,363);
	   time_gpu(0,0,64,2916,363);
	   time_gpu(0,0,192,729,1600);
	   time_gpu(0,0,384,196,1728);
	   time_gpu(0,0,256,196,3456);
	   time_gpu(0,0,256,196,2304);
	   time_gpu(0,0,128,4096,12544);
	   time_gpu(0,0,128,4096,4096);
	 */
	time_gpu(0, 0, 64, 75, 12544);
	time_gpu(0, 0, 64, 75, 12544);
	time_gpu(0, 0, 64, 75, 12544);
	time_gpu(0, 0, 64, 576, 12544);
	time_gpu(0, 0, 256, 2304, 784);
	time_gpu(1, 1, 2304, 256, 784);
	time_gpu(0, 0, 512, 4608, 196);
	time_gpu(1, 1, 4608, 512, 196);

	return 0;
}
#endif








//im2col.c
float im2col_get_pixel(float *im, int height, int width, int channels,
	int row, int col, int channel, int pad)
{
	row -= pad;
	col -= pad;

	if (row < 0 || col < 0 ||
		row >= height || col >= width) return 0;
	return im[col + width * (row + height * channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(float* data_im,
	int channels, int height, int width,
	int ksize, int stride, int pad, float* data_col)
{
	int c, h, w;
	int height_col = (height + 2 * pad - ksize) / stride + 1;
	int width_col = (width + 2 * pad - ksize) / stride + 1;

	int channels_col = channels * ksize * ksize;
	for (c = 0; c < channels_col; ++c) {
		int w_offset = c % ksize;
		int h_offset = (c / ksize) % ksize;
		int c_im = c / ksize / ksize;
		for (h = 0; h < height_col; ++h) {
			for (w = 0; w < width_col; ++w) {
				int im_row = h_offset + h * stride;
				int im_col = w_offset + w * stride;
				int col_index = (c * height_col + h) * width_col + w;
				data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
					im_row, im_col, c_im, pad);
			}
		}
	}
}







//image.c
float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };
float get_color(int c, int x, int max)
{
	float ratio = ((float)x / max) * 5;
	int i = floor(ratio);
	int j = ceil(ratio);
	ratio -= i;
	float r = (1 - ratio) * colors[i][c] + ratio * colors[j][c];
	//printf("%f\n", r);
	return r;
}

image mask_to_rgb(image mask)
{
	int n = mask.c;
	image im = make_image(mask.w, mask.h, 3);
	int i, j;
	for (j = 0; j < n; ++j) {
		int offset = j * 123457 % n;
		float red = get_color(2, offset, n);
		float green = get_color(1, offset, n);
		float blue = get_color(0, offset, n);
		for (i = 0; i < im.w*im.h; ++i) {
			im.data[i + 0 * im.w*im.h] += mask.data[j*im.h*im.w + i] * red;
			im.data[i + 1 * im.w*im.h] += mask.data[j*im.h*im.w + i] * green;
			im.data[i + 2 * im.w*im.h] += mask.data[j*im.h*im.w + i] * blue;
		}
	}
	return im;
}

static float get_pixel(image m, int x, int y, int c)
{
	assert(x < m.w && y < m.h && c < m.c);
	return m.data[c*m.h*m.w + y * m.w + x];
}
static float get_pixel_extend(image m, int x, int y, int c)
{
	if (x < 0 || x >= m.w || y < 0 || y >= m.h) return 0;
	/*
	if(x < 0) x = 0;
	if(x >= m.w) x = m.w-1;
	if(y < 0) y = 0;
	if(y >= m.h) y = m.h-1;
	*/
	if (c < 0 || c >= m.c) return 0;
	return get_pixel(m, x, y, c);
}
static void set_pixel(image m, int x, int y, int c, float val)
{
	if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
	assert(x < m.w && y < m.h && c < m.c);
	m.data[c*m.h*m.w + y * m.w + x] = val;
}
static void add_pixel(image m, int x, int y, int c, float val)
{
	assert(x < m.w && y < m.h && c < m.c);
	m.data[c*m.h*m.w + y * m.w + x] += val;
}

static float bilinear_interpolate(image im, float x, float y, int c)
{
	int ix = (int)floorf(x);
	int iy = (int)floorf(y);

	float dx = x - ix;
	float dy = y - iy;

	float val = (1 - dy) * (1 - dx) * get_pixel_extend(im, ix, iy, c) +
		dy * (1 - dx) * get_pixel_extend(im, ix, iy + 1, c) +
		(1 - dy) *   dx   * get_pixel_extend(im, ix + 1, iy, c) +
		dy * dx   * get_pixel_extend(im, ix + 1, iy + 1, c);
	return val;
}


void composite_image(image source, image dest, int dx, int dy)
{
	int x, y, k;
	for (k = 0; k < source.c; ++k) {
		for (y = 0; y < source.h; ++y) {
			for (x = 0; x < source.w; ++x) {
				float val = get_pixel(source, x, y, k);
				float val2 = get_pixel_extend(dest, dx + x, dy + y, k);
				set_pixel(dest, dx + x, dy + y, k, val * val2);
			}
		}
	}
}

image border_image(image a, int border)
{
	image b = make_image(a.w + 2 * border, a.h + 2 * border, a.c);
	int x, y, k;
	for (k = 0; k < b.c; ++k) {
		for (y = 0; y < b.h; ++y) {
			for (x = 0; x < b.w; ++x) {
				float val = get_pixel_extend(a, x - border, y - border, k);
				if (x - border < 0 || x - border >= a.w || y - border < 0 || y - border >= a.h) val = 1;
				set_pixel(b, x, y, k, val);
			}
		}
	}
	return b;
}

image tile_images(image a, image b, int dx)
{
	if (a.w == 0) return copy_image(b);
	image c = make_image(a.w + b.w + dx, (a.h > b.h) ? a.h : b.h, (a.c > b.c) ? a.c : b.c);
	fill_cpu(c.w*c.h*c.c, 1, c.data, 1);
	embed_image(a, c, 0, 0);
	composite_image(b, c, a.w + dx, 0);
	return c;
}

image get_label(image **characters, char *string, int size)
{
	size = size / 10;
	if (size > 7) size = 7;
	image label = make_empty_image(0, 0, 0);
	while (*string) {
		image l = characters[size][(int)*string];
		image n = tile_images(label, l, -size - 1 + (size + 1) / 2);
		free_image(label);
		label = n;
		++string;
	}
	image b = border_image(label, label.h*.25);
	free_image(label);
	return b;
}

void draw_label(image a, int r, int c, image label, const float *rgb)
{
	int w = label.w;
	int h = label.h;
	if (r - h >= 0) r = r - h;

	int i, j, k;
	for (j = 0; j < h && j + r < a.h; ++j) {
		for (i = 0; i < w && i + c < a.w; ++i) {
			for (k = 0; k < label.c; ++k) {
				float val = get_pixel(label, i, j, k);
				set_pixel(a, i + c, j + r, k, rgb[k] * val);
			}
		}
	}
}

void draw_box(image a, int x1, int y1, int x2, int y2, float r, float g, float b)
{
	//normalize_image(a);
	int i;
	if (x1 < 0) x1 = 0;
	if (x1 >= a.w) x1 = a.w - 1;
	if (x2 < 0) x2 = 0;
	if (x2 >= a.w) x2 = a.w - 1;

	if (y1 < 0) y1 = 0;
	if (y1 >= a.h) y1 = a.h - 1;
	if (y2 < 0) y2 = 0;
	if (y2 >= a.h) y2 = a.h - 1;

	for (i = x1; i <= x2; ++i) {
		a.data[i + y1 * a.w + 0 * a.w*a.h] = r;
		a.data[i + y2 * a.w + 0 * a.w*a.h] = r;

		a.data[i + y1 * a.w + 1 * a.w*a.h] = g;
		a.data[i + y2 * a.w + 1 * a.w*a.h] = g;

		a.data[i + y1 * a.w + 2 * a.w*a.h] = b;
		a.data[i + y2 * a.w + 2 * a.w*a.h] = b;
	}
	for (i = y1; i <= y2; ++i) {
		a.data[x1 + i * a.w + 0 * a.w*a.h] = r;
		a.data[x2 + i * a.w + 0 * a.w*a.h] = r;

		a.data[x1 + i * a.w + 1 * a.w*a.h] = g;
		a.data[x2 + i * a.w + 1 * a.w*a.h] = g;

		a.data[x1 + i * a.w + 2 * a.w*a.h] = b;
		a.data[x2 + i * a.w + 2 * a.w*a.h] = b;
	}
}

void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b)
{
	int i;
	for (i = 0; i < w; ++i) {
		draw_box(a, x1 + i, y1 + i, x2 - i, y2 - i, r, g, b);
	}
}

void draw_bbox(image a, box bbox, int w, float r, float g, float b)
{
	int left = (bbox.x - bbox.w / 2)*a.w;
	int right = (bbox.x + bbox.w / 2)*a.w;
	int top = (bbox.y - bbox.h / 2)*a.h;
	int bot = (bbox.y + bbox.h / 2)*a.h;

	int i;
	for (i = 0; i < w; ++i) {
		draw_box(a, left + i, top + i, right - i, bot - i, r, g, b);
	}
}

image **load_alphabet()
{
	int i, j;
	const int nsize = 8;
	image **alphabets = (image**)calloc(nsize, sizeof(image));
	for (j = 0; j < nsize; ++j) {
		alphabets[j] = (image*)calloc(128, sizeof(image));
		for (i = 32; i < 127; ++i) {
			char buff[256];
//			sprintf(buff, "data/labels/%d_%d.png", i, j);

			sprintf(buff, "C:\\Users\\ASUS\\Desktop\\temp_data\\darknet\\DarkNet\\darknet\\data\\labels\\%d_%d.png", i, j);


			alphabets[j][i] = load_image_color(buff, 0, 0);
		}
	}
	return alphabets;
}

void draw_detections(image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes)
{
	int i, j;

	for (i = 0; i < num; ++i) {
		char labelstr[4096] = { 0 };
		int class_n = -1;
		for (j = 0; j < classes; ++j) {
			if (dets[i].prob[j] > thresh) {
				if (class_n < 0) {
					strcat(labelstr, names[j]);
					class_n = j;
				}
				else {
					strcat(labelstr, ", ");
					strcat(labelstr, names[j]);
				}
				printf("%s: %.0f%%\n", names[j], dets[i].prob[j] * 100);
			}
		}
		if (class_n >= 0) {
			int width = im.h * .006;

			/*
			   if(0){
			   width = pow(prob, 1./2.)*10+1;
			   alphabet = 0;
			   }
			 */

			 //printf("%d %s: %.0f%%\n", i, names[class], prob*100);
			int offset = class_n * 123457 % classes;
			float red = get_color(2, offset, classes);
			float green = get_color(1, offset, classes);
			float blue = get_color(0, offset, classes);
			float rgb[3];

			//width = prob*20+2;

			rgb[0] = red;
			rgb[1] = green;
			rgb[2] = blue;
			box b = dets[i].bbox;
			//printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

			int left = (b.x - b.w / 2.)*im.w;
			int right = (b.x + b.w / 2.)*im.w;
			int top = (b.y - b.h / 2.)*im.h;
			int bot = (b.y + b.h / 2.)*im.h;

			if (left < 0) left = 0;
			if (right > im.w - 1) right = im.w - 1;
			if (top < 0) top = 0;
			if (bot > im.h - 1) bot = im.h - 1;

			draw_box_width(im, left, top, right, bot, width, red, green, blue);
			if (alphabet) {
				image label = get_label(alphabet, labelstr, (im.h*.03));
				draw_label(im, top + width, left, label, rgb);
				free_image(label);
			}
			if (dets[i].mask) {
				image mask = float_to_image(14, 14, 1, dets[i].mask);
				image resized_mask = resize_image(mask, b.w*im.w, b.h*im.h);
				image tmask = threshold_image(resized_mask, .5);
				embed_image(tmask, im, left, top);
				free_image(mask);
				free_image(resized_mask);
				free_image(tmask);
			}
		}
	}
}

void transpose_image(image im)
{
	assert(im.w == im.h);
	int n, m;
	int c;
	for (c = 0; c < im.c; ++c) {
		for (n = 0; n < im.w - 1; ++n) {
			for (m = n + 1; m < im.w; ++m) {
				float swap = im.data[m + im.w*(n + im.h*c)];
				im.data[m + im.w*(n + im.h*c)] = im.data[n + im.w*(m + im.h*c)];
				im.data[n + im.w*(m + im.h*c)] = swap;
			}
		}
	}
}

void rotate_image_cw(image im, int times)
{
	assert(im.w == im.h);
	times = (times + 400) % 4;
	int i, x, y, c;
	int n = im.w;
	for (i = 0; i < times; ++i) {
		for (c = 0; c < im.c; ++c) {
			for (x = 0; x < n / 2; ++x) {
				for (y = 0; y < (n - 1) / 2 + 1; ++y) {
					float temp = im.data[y + im.w*(x + im.h*c)];
					im.data[y + im.w*(x + im.h*c)] = im.data[n - 1 - x + im.w*(y + im.h*c)];
					im.data[n - 1 - x + im.w*(y + im.h*c)] = im.data[n - 1 - y + im.w*(n - 1 - x + im.h*c)];
					im.data[n - 1 - y + im.w*(n - 1 - x + im.h*c)] = im.data[x + im.w*(n - 1 - y + im.h*c)];
					im.data[x + im.w*(n - 1 - y + im.h*c)] = temp;
				}
			}
		}
	}
}

void flip_image(image a)
{
	int i, j, k;
	for (k = 0; k < a.c; ++k) {
		for (i = 0; i < a.h; ++i) {
			for (j = 0; j < a.w / 2; ++j) {
				int index = j + a.w*(i + a.h*(k));
				int flip = (a.w - j - 1) + a.w*(i + a.h*(k));
				float swap = a.data[flip];
				a.data[flip] = a.data[index];
				a.data[index] = swap;
			}
		}
	}
}

image image_distance(image a, image b)
{
	int i, j;
	image dist = make_image(a.w, a.h, 1);
	for (i = 0; i < a.c; ++i) {
		for (j = 0; j < a.h*a.w; ++j) {
			dist.data[j] += pow(a.data[i*a.h*a.w + j] - b.data[i*a.h*a.w + j], 2);
		}
	}
	for (j = 0; j < a.h*a.w; ++j) {
		dist.data[j] = sqrt(dist.data[j]);
	}
	return dist;
}

void ghost_image(image source, image dest, int dx, int dy)
{
	int x, y, k;
	float max_dist = sqrt((-source.w / 2. + .5)*(-source.w / 2. + .5));
	for (k = 0; k < source.c; ++k) {
		for (y = 0; y < source.h; ++y) {
			for (x = 0; x < source.w; ++x) {
				float dist = sqrt((x - source.w / 2. + .5)*(x - source.w / 2. + .5) + (y - source.h / 2. + .5)*(y - source.h / 2. + .5));
				float alpha = (1 - dist / max_dist);
				if (alpha < 0) alpha = 0;
				float v1 = get_pixel(source, x, y, k);
				float v2 = get_pixel(dest, dx + x, dy + y, k);
				float val = alpha * v1 + (1 - alpha)*v2;
				set_pixel(dest, dx + x, dy + y, k, val);
			}
		}
	}
}

void blocky_image(image im, int s)
{
	int i, j, k;
	for (k = 0; k < im.c; ++k) {
		for (j = 0; j < im.h; ++j) {
			for (i = 0; i < im.w; ++i) {
				im.data[i + im.w*(j + im.h*k)] = im.data[i / s * s + im.w*(j / s * s + im.h*k)];
			}
		}
	}
}

void censor_image(image im, int dx, int dy, int w, int h)
{
	int i, j, k;
	int s = 32;
	if (dx < 0) dx = 0;
	if (dy < 0) dy = 0;

	for (k = 0; k < im.c; ++k) {
		for (j = dy; j < dy + h && j < im.h; ++j) {
			for (i = dx; i < dx + w && i < im.w; ++i) {
				im.data[i + im.w*(j + im.h*k)] = im.data[i / s * s + im.w*(j / s * s + im.h*k)];
				//im.data[i + j*im.w + k*im.w*im.h] = 0;
			}
		}
	}
}

void embed_image(image source, image dest, int dx, int dy)
{
	int x, y, k;
	for (k = 0; k < source.c; ++k) {
		for (y = 0; y < source.h; ++y) {
			for (x = 0; x < source.w; ++x) {
				float val = get_pixel(source, x, y, k);
				set_pixel(dest, dx + x, dy + y, k, val);
			}
		}
	}
}

image collapse_image_layers(image source, int border)
{
	int h = source.h;
	h = (h + border)*source.c - border;
	image dest = make_image(source.w, h, 1);
	int i;
	for (i = 0; i < source.c; ++i) {
		image layer = get_image_layer(source, i);
		int h_offset = i * (source.h + border);
		embed_image(layer, dest, 0, h_offset);
		free_image(layer);
	}
	return dest;
}

void constrain_image(image im)
{
	int i;
	for (i = 0; i < im.w*im.h*im.c; ++i) {
		if (im.data[i] < 0) im.data[i] = 0;
		if (im.data[i] > 1) im.data[i] = 1;
	}
}

void normalize_image(image p)
{
	int i;
	float min = 9999999;
	float max = -999999;

	for (i = 0; i < p.h*p.w*p.c; ++i) {
		float v = p.data[i];
		if (v < min) min = v;
		if (v > max) max = v;
	}
	if (max - min < .000000001) {
		min = 0;
		max = 1;
	}
	for (i = 0; i < p.c*p.w*p.h; ++i) {
		p.data[i] = (p.data[i] - min) / (max - min);
	}
}

void normalize_image2(image p)
{
	float *min = (float*)calloc(p.c, sizeof(float));
	float *max = (float*)calloc(p.c, sizeof(float));
	int i, j;
	for (i = 0; i < p.c; ++i) min[i] = max[i] = p.data[i*p.h*p.w];

	for (j = 0; j < p.c; ++j) {
		for (i = 0; i < p.h*p.w; ++i) {
			float v = p.data[i + j * p.h*p.w];
			if (v < min[j]) min[j] = v;
			if (v > max[j]) max[j] = v;
		}
	}
	for (i = 0; i < p.c; ++i) {
		if (max[i] - min[i] < .000000001) {
			min[i] = 0;
			max[i] = 1;
		}
	}
	for (j = 0; j < p.c; ++j) {
		for (i = 0; i < p.w*p.h; ++i) {
			p.data[i + j * p.h*p.w] = (p.data[i + j * p.h*p.w] - min[j]) / (max[j] - min[j]);
		}
	}
	free(min);
	free(max);
}

void copy_image_into(image src, image dest)
{
	memcpy(dest.data, src.data, src.h*src.w*src.c * sizeof(float));
}

image copy_image(image p)
{
	image copy = p;
	copy.data = (float*)calloc(p.h*p.w*p.c, sizeof(float));
	memcpy(copy.data, p.data, p.h*p.w*p.c * sizeof(float));
	return copy;
}

void rgbgr_image(image im)
{
	int i;
	for (i = 0; i < im.w*im.h; ++i) {
		float swap = im.data[i];
		im.data[i] = im.data[i + im.w*im.h * 2];
		im.data[i + im.w*im.h * 2] = swap;
	}
}

int show_image(image p, const char *name, int ms)
{
#ifdef CV_VERSION
	int c = cv::show_image_cv(p, name, ms);
	return c;
#else
	fprintf(stderr, "Not compiled with OpenCV, saving to %s.png instead\n", name);
	save_image(p, name);
	return -1;
#endif
}

void save_image_options(image im, const char *name, IMTYPE f, int quality)
{
	char buff[256];
	//sprintf(buff, "%s (%d)", name, windows);
	if (f == PNG)       sprintf(buff, "%s.png", name);
	else if (f == BMP) sprintf(buff, "%s.bmp", name);
	else if (f == TGA) sprintf(buff, "%s.tga", name);
	else if (f == JPG) sprintf(buff, "%s.jpg", name);
	else               sprintf(buff, "%s.png", name);
	unsigned char *data = (unsigned char*)calloc(im.w*im.h*im.c, sizeof(char));
	int i, k;
	for (k = 0; k < im.c; ++k) {
		for (i = 0; i < im.w*im.h; ++i) {
			data[i*im.c + k] = (unsigned char)(255 * im.data[i + k * im.w*im.h]);
		}
	}
	int success = 0;
	if (f == PNG)       success = stbi_write_png(buff, im.w, im.h, im.c, data, im.w*im.c);
	else if (f == BMP) success = stbi_write_bmp(buff, im.w, im.h, im.c, data);
	else if (f == TGA) success = stbi_write_tga(buff, im.w, im.h, im.c, data);
	else if (f == JPG) success = stbi_write_jpg(buff, im.w, im.h, im.c, data, quality);
	free(data);
	if (!success) fprintf(stderr, "Failed to write image %s\n", buff);
}

void save_image(image im, const char *name)
{
	save_image_options(im, name, JPG, 80);
}

void show_image_layers(image p, char *name)
{
	int i;
	char buff[256];
	for (i = 0; i < p.c; ++i) {
		sprintf(buff, "%s - Layer %d", name, i);
		image layer = get_image_layer(p, i);
		show_image(layer, buff, 1);
		free_image(layer);
	}
}

void show_image_collapsed(image p, char *name)
{
	image c = collapse_image_layers(p, 1);
	show_image(c, name, 1);
	free_image(c);
}

image make_empty_image(int w, int h, int c)
{
	image out;
	out.data = 0;
	out.h = h;
	out.w = w;
	out.c = c;
	return out;
}

image make_image(int w, int h, int c)
{
	image out = make_empty_image(w, h, c);
	out.data = (float*)calloc(h*w*c, sizeof(float));
	return out;
}

image make_random_image(int w, int h, int c)
{
	image out = make_empty_image(w, h, c);
	out.data = (float*)calloc(h*w*c, sizeof(float));
	int i;
	for (i = 0; i < w*h*c; ++i) {
		out.data[i] = (rand_normal() * .25) + .5;
	}
	return out;
}

image float_to_image(int w, int h, int c, float *data)
{
	image out = make_empty_image(w, h, c);
	out.data = data;
	return out;
}

void place_image(image im, int w, int h, int dx, int dy, image canvas)
{
	int x, y, c;
	for (c = 0; c < im.c; ++c) {
		for (y = 0; y < h; ++y) {
			for (x = 0; x < w; ++x) {
				float rx = ((float)x / w) * im.w;
				float ry = ((float)y / h) * im.h;
				float val = bilinear_interpolate(im, rx, ry, c);
				set_pixel(canvas, x + dx, y + dy, c, val);
			}
		}
	}
}

image center_crop_image(image im, int w, int h)
{
	int m = (im.w < im.h) ? im.w : im.h;
	image c = crop_image(im, (im.w - m) / 2, (im.h - m) / 2, m, m);
	image r = resize_image(c, w, h);
	free_image(c);
	return r;
}

image rotate_crop_image(image im, float rad, float s, int w, int h, float dx, float dy, float aspect)
{
	int x, y, c;
	float cx = im.w / 2.;
	float cy = im.h / 2.;
	image rot = make_image(w, h, im.c);
	for (c = 0; c < im.c; ++c) {
		for (y = 0; y < h; ++y) {
			for (x = 0; x < w; ++x) {
				float rx = cos(rad)*((x - w / 2.) / s * aspect + dx / s * aspect) - sin(rad)*((y - h / 2.) / s + dy / s) + cx;
				float ry = sin(rad)*((x - w / 2.) / s * aspect + dx / s * aspect) + cos(rad)*((y - h / 2.) / s + dy / s) + cy;
				float val = bilinear_interpolate(im, rx, ry, c);
				set_pixel(rot, x, y, c, val);
			}
		}
	}
	return rot;
}

image rotate_image(image im, float rad)
{
	int x, y, c;
	float cx = im.w / 2.;
	float cy = im.h / 2.;
	image rot = make_image(im.w, im.h, im.c);
	for (c = 0; c < im.c; ++c) {
		for (y = 0; y < im.h; ++y) {
			for (x = 0; x < im.w; ++x) {
				float rx = cos(rad)*(x - cx) - sin(rad)*(y - cy) + cx;
				float ry = sin(rad)*(x - cx) + cos(rad)*(y - cy) + cy;
				float val = bilinear_interpolate(im, rx, ry, c);
				set_pixel(rot, x, y, c, val);
			}
		}
	}
	return rot;
}

void fill_image(image m, float s)
{
	int i;
	for (i = 0; i < m.h*m.w*m.c; ++i) m.data[i] = s;
}

void translate_image(image m, float s)
{
	int i;
	for (i = 0; i < m.h*m.w*m.c; ++i) m.data[i] += s;
}

void scale_image(image m, float s)
{
	int i;
	for (i = 0; i < m.h*m.w*m.c; ++i) m.data[i] *= s;
}

image crop_image(image im, int dx, int dy, int w, int h)
{
	image cropped = make_image(w, h, im.c);
	int i, j, k;
	for (k = 0; k < im.c; ++k) {
		for (j = 0; j < h; ++j) {
			for (i = 0; i < w; ++i) {
				int r = j + dy;
				int c = i + dx;
				float val = 0;
				r = constrain_int(r, 0, im.h - 1);
				c = constrain_int(c, 0, im.w - 1);
				val = get_pixel(im, c, r, k);
				set_pixel(cropped, i, j, k, val);
			}
		}
	}
	return cropped;
}

int best_3d_shift_r(image a, image b, int min, int max)
{
	if (min == max) return min;
	int mid = floor((min + max) / 2.);
	image c1 = crop_image(b, 0, mid, b.w, b.h);
	image c2 = crop_image(b, 0, mid + 1, b.w, b.h);
	float d1 = dist_array(c1.data, a.data, a.w*a.h*a.c, 10);
	float d2 = dist_array(c2.data, a.data, a.w*a.h*a.c, 10);
	free_image(c1);
	free_image(c2);
	if (d1 < d2) return best_3d_shift_r(a, b, min, mid);
	else return best_3d_shift_r(a, b, mid + 1, max);
}

int best_3d_shift(image a, image b, int min, int max)
{
	int i;
	int best = 0;
	float best_distance = FLT_MAX;
	for (i = min; i <= max; i += 2) {
		image c = crop_image(b, 0, i, b.w, b.h);
		float d = dist_array(c.data, a.data, a.w*a.h*a.c, 100);
		if (d < best_distance) {
			best_distance = d;
			best = i;
		}
		printf("%d %f\n", i, d);
		free_image(c);
	}
	return best;
}

void composite_3d(char *f1, char *f2, char *out, int delta)
{
	if (!out) out = (char*)"out";
	image a = load_image(f1, 0, 0, 0);
	image b = load_image(f2, 0, 0, 0);
	int shift = best_3d_shift_r(a, b, -a.h / 100, a.h / 100);

	image c1 = crop_image(b, 10, shift, b.w, b.h);
	float d1 = dist_array(c1.data, a.data, a.w*a.h*a.c, 100);
	image c2 = crop_image(b, -10, shift, b.w, b.h);
	float d2 = dist_array(c2.data, a.data, a.w*a.h*a.c, 100);

	if (d2 < d1 && 0) {
		image swap = a;
		a = b;
		b = swap;
		shift = -shift;
		printf("swapped, %d\n", shift);
	}
	else {
		printf("%d\n", shift);
	}

	image c = crop_image(b, delta, shift, a.w, a.h);
	int i;
	for (i = 0; i < c.w*c.h; ++i) {
		c.data[i] = a.data[i];
	}
	save_image(c, out);
}

void letterbox_image_into(image im, int w, int h, image boxed)
{
	int new_w = im.w;
	int new_h = im.h;
	if (((float)w / im.w) < ((float)h / im.h)) {
		new_w = w;
		new_h = (im.h * w) / im.w;
	}
	else {
		new_h = h;
		new_w = (im.w * h) / im.h;
	}
	image resized = resize_image(im, new_w, new_h);
	embed_image(resized, boxed, (w - new_w) / 2, (h - new_h) / 2);
	free_image(resized);
}

image letterbox_image(image im, int w, int h)
{
	int new_w = im.w;
	int new_h = im.h;
	if (((float)w / im.w) < ((float)h / im.h)) {
		new_w = w;
		new_h = (im.h * w) / im.w;
	}
	else {
		new_h = h;
		new_w = (im.w * h) / im.h;
	}
	image resized = resize_image(im, new_w, new_h);
	image boxed = make_image(w, h, im.c);
	fill_image(boxed, .5);
	//int i;
	//for(i = 0; i < boxed.w*boxed.h*boxed.c; ++i) boxed.data[i] = 0;
	embed_image(resized, boxed, (w - new_w) / 2, (h - new_h) / 2);
	free_image(resized);
	return boxed;
}

image resize_max(image im, int max)
{
	int w = im.w;
	int h = im.h;
	if (w > h) {
		h = (h * max) / w;
		w = max;
	}
	else {
		w = (w * max) / h;
		h = max;
	}
	if (w == im.w && h == im.h) return im;
	image resized = resize_image(im, w, h);
	return resized;
}

image resize_min(image im, int min)
{
	int w = im.w;
	int h = im.h;
	if (w < h) {
		h = (h * min) / w;
		w = min;
	}
	else {
		w = (w * min) / h;
		h = min;
	}
	if (w == im.w && h == im.h) return im;
	image resized = resize_image(im, w, h);
	return resized;
}

image random_crop_image(image im, int w, int h)
{
	int dx = rand_int(0, im.w - w);
	int dy = rand_int(0, im.h - h);
	image crop = crop_image(im, dx, dy, w, h);
	return crop;
}

augment_args random_augment_args(image im, float angle, float aspect, int low, int high, int w, int h)
{
	augment_args a = { 0 };
	aspect = rand_scale(aspect);
	int r = rand_int(low, high);
	int min = (im.h < im.w*aspect) ? im.h : im.w*aspect;
	float scale = (float)r / min;

	float rad = rand_uniform(-angle, angle) * TWO_PI / 360.;

	float dx = (im.w*scale / aspect - w) / 2.;
	float dy = (im.h*scale - w) / 2.;
	//if(dx < 0) dx = 0;
	//if(dy < 0) dy = 0;
	dx = rand_uniform(-dx, dx);
	dy = rand_uniform(-dy, dy);

	a.rad = rad;
	a.scale = scale;
	a.w = w;
	a.h = h;
	a.dx = dx;
	a.dy = dy;
	a.aspect = aspect;
	return a;
}

image random_augment_image(image im, float angle, float aspect, int low, int high, int w, int h)
{
	augment_args a = random_augment_args(im, angle, aspect, low, high, w, h);
	image crop = rotate_crop_image(im, a.rad, a.scale, a.w, a.h, a.dx, a.dy, a.aspect);
	return crop;
}

float three_way_max(float a, float b, float c)
{
	return (a > b) ? ((a > c) ? a : c) : ((b > c) ? b : c);
}

float three_way_min(float a, float b, float c)
{
	return (a < b) ? ((a < c) ? a : c) : ((b < c) ? b : c);
}

void yuv_to_rgb(image im)
{
	assert(im.c == 3);
	int i, j;
	float r, g, b;
	float y, u, v;
	for (j = 0; j < im.h; ++j) {
		for (i = 0; i < im.w; ++i) {
			y = get_pixel(im, i, j, 0);
			u = get_pixel(im, i, j, 1);
			v = get_pixel(im, i, j, 2);

			r = y + 1.13983*v;
			g = y + -.39465*u + -.58060*v;
			b = y + 2.03211*u;

			set_pixel(im, i, j, 0, r);
			set_pixel(im, i, j, 1, g);
			set_pixel(im, i, j, 2, b);
		}
	}
}

void rgb_to_yuv(image im)
{
	assert(im.c == 3);
	int i, j;
	float r, g, b;
	float y, u, v;
	for (j = 0; j < im.h; ++j) {
		for (i = 0; i < im.w; ++i) {
			r = get_pixel(im, i, j, 0);
			g = get_pixel(im, i, j, 1);
			b = get_pixel(im, i, j, 2);

			y = .299*r + .587*g + .114*b;
			u = -.14713*r + -.28886*g + .436*b;
			v = .615*r + -.51499*g + -.10001*b;

			set_pixel(im, i, j, 0, y);
			set_pixel(im, i, j, 1, u);
			set_pixel(im, i, j, 2, v);
		}
	}
}

// http://www.cs.rit.edu/~ncs/color/t_convert.html
void rgb_to_hsv(image im)
{
	assert(im.c == 3);
	int i, j;
	float r, g, b;
	float h, s, v;
	for (j = 0; j < im.h; ++j) {
		for (i = 0; i < im.w; ++i) {
			r = get_pixel(im, i, j, 0);
			g = get_pixel(im, i, j, 1);
			b = get_pixel(im, i, j, 2);
			float max = three_way_max(r, g, b);
			float min = three_way_min(r, g, b);
			float delta = max - min;
			v = max;
			if (max == 0) {
				s = 0;
				h = 0;
			}
			else {
				s = delta / max;
				if (r == max) {
					h = (g - b) / delta;
				}
				else if (g == max) {
					h = 2 + (b - r) / delta;
				}
				else {
					h = 4 + (r - g) / delta;
				}
				if (h < 0) h += 6;
				h = h / 6.;
			}
			set_pixel(im, i, j, 0, h);
			set_pixel(im, i, j, 1, s);
			set_pixel(im, i, j, 2, v);
		}
	}
}

void hsv_to_rgb(image im)
{
	assert(im.c == 3);
	int i, j;
	float r, g, b;
	float h, s, v;
	float f, p, q, t;
	for (j = 0; j < im.h; ++j) {
		for (i = 0; i < im.w; ++i) {
			h = 6 * get_pixel(im, i, j, 0);
			s = get_pixel(im, i, j, 1);
			v = get_pixel(im, i, j, 2);
			if (s == 0) {
				r = g = b = v;
			}
			else {
				int index = floor(h);
				f = h - index;
				p = v * (1 - s);
				q = v * (1 - s * f);
				t = v * (1 - s * (1 - f));
				if (index == 0) {
					r = v; g = t; b = p;
				}
				else if (index == 1) {
					r = q; g = v; b = p;
				}
				else if (index == 2) {
					r = p; g = v; b = t;
				}
				else if (index == 3) {
					r = p; g = q; b = v;
				}
				else if (index == 4) {
					r = t; g = p; b = v;
				}
				else {
					r = v; g = p; b = q;
				}
			}
			set_pixel(im, i, j, 0, r);
			set_pixel(im, i, j, 1, g);
			set_pixel(im, i, j, 2, b);
		}
	}
}

void grayscale_image_3c(image im)
{
	assert(im.c == 3);
	int i, j, k;
	float scale[] = { 0.299, 0.587, 0.114 };
	for (j = 0; j < im.h; ++j) {
		for (i = 0; i < im.w; ++i) {
			float val = 0;
			for (k = 0; k < 3; ++k) {
				val += scale[k] * get_pixel(im, i, j, k);
			}
			im.data[0 * im.h*im.w + im.w*j + i] = val;
			im.data[1 * im.h*im.w + im.w*j + i] = val;
			im.data[2 * im.h*im.w + im.w*j + i] = val;
		}
	}
}

image grayscale_image(image im)
{
	assert(im.c == 3);
	int i, j, k;
	image gray = make_image(im.w, im.h, 1);
	float scale[] = { 0.299, 0.587, 0.114 };
	for (k = 0; k < im.c; ++k) {
		for (j = 0; j < im.h; ++j) {
			for (i = 0; i < im.w; ++i) {
				gray.data[i + im.w*j] += scale[k] * get_pixel(im, i, j, k);
			}
		}
	}
	return gray;
}

image threshold_image(image im, float thresh)
{
	int i;
	image t = make_image(im.w, im.h, im.c);
	for (i = 0; i < im.w*im.h*im.c; ++i) {
		t.data[i] = im.data[i] > thresh ? 1 : 0;
	}
	return t;
}

image blend_image(image fore, image back, float alpha)
{
	assert(fore.w == back.w && fore.h == back.h && fore.c == back.c);
	image blend = make_image(fore.w, fore.h, fore.c);
	int i, j, k;
	for (k = 0; k < fore.c; ++k) {
		for (j = 0; j < fore.h; ++j) {
			for (i = 0; i < fore.w; ++i) {
				float val = alpha * get_pixel(fore, i, j, k) +
					(1 - alpha)* get_pixel(back, i, j, k);
				set_pixel(blend, i, j, k, val);
			}
		}
	}
	return blend;
}

void scale_image_channel(image im, int c, float v)
{
	int i, j;
	for (j = 0; j < im.h; ++j) {
		for (i = 0; i < im.w; ++i) {
			float pix = get_pixel(im, i, j, c);
			pix = pix * v;
			set_pixel(im, i, j, c, pix);
		}
	}
}

void translate_image_channel(image im, int c, float v)
{
	int i, j;
	for (j = 0; j < im.h; ++j) {
		for (i = 0; i < im.w; ++i) {
			float pix = get_pixel(im, i, j, c);
			pix = pix + v;
			set_pixel(im, i, j, c, pix);
		}
	}
}

image binarize_image(image im)
{
	image c = copy_image(im);
	int i;
	for (i = 0; i < im.w * im.h * im.c; ++i) {
		if (c.data[i] > .5) c.data[i] = 1;
		else c.data[i] = 0;
	}
	return c;
}

void saturate_image(image im, float sat)
{
	rgb_to_hsv(im);
	scale_image_channel(im, 1, sat);
	hsv_to_rgb(im);
	constrain_image(im);
}

void hue_image(image im, float hue)
{
	rgb_to_hsv(im);
	int i;
	for (i = 0; i < im.w*im.h; ++i) {
		im.data[i] = im.data[i] + hue;
		if (im.data[i] > 1) im.data[i] -= 1;
		if (im.data[i] < 0) im.data[i] += 1;
	}
	hsv_to_rgb(im);
	constrain_image(im);
}

void exposure_image(image im, float sat)
{
	rgb_to_hsv(im);
	scale_image_channel(im, 2, sat);
	hsv_to_rgb(im);
	constrain_image(im);
}

void distort_image(image im, float hue, float sat, float val)
{
	rgb_to_hsv(im);
	scale_image_channel(im, 1, sat);
	scale_image_channel(im, 2, val);
	int i;
	for (i = 0; i < im.w*im.h; ++i) {
		im.data[i] = im.data[i] + hue;
		if (im.data[i] > 1) im.data[i] -= 1;
		if (im.data[i] < 0) im.data[i] += 1;
	}
	hsv_to_rgb(im);
	constrain_image(im);
}

void random_distort_image(image im, float hue, float saturation, float exposure)
{
	float dhue = rand_uniform(-hue, hue);
	float dsat = rand_scale(saturation);
	float dexp = rand_scale(exposure);
	distort_image(im, dhue, dsat, dexp);
}

void saturate_exposure_image(image im, float sat, float exposure)
{
	rgb_to_hsv(im);
	scale_image_channel(im, 1, sat);
	scale_image_channel(im, 2, exposure);
	hsv_to_rgb(im);
	constrain_image(im);
}

image resize_image(image im, int w, int h)
{
	image resized = make_image(w, h, im.c);
	image part = make_image(w, im.h, im.c);
	int r, c, k;
	float w_scale = (float)(im.w - 1) / (w - 1);
	float h_scale = (float)(im.h - 1) / (h - 1);
	for (k = 0; k < im.c; ++k) {
		for (r = 0; r < im.h; ++r) {
			for (c = 0; c < w; ++c) {
				float val = 0;
				if (c == w - 1 || im.w == 1) {
					val = get_pixel(im, im.w - 1, r, k);
				}
				else {
					float sx = c * w_scale;
					int ix = (int)sx;
					float dx = sx - ix;
					val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix + 1, r, k);
				}
				set_pixel(part, c, r, k, val);
			}
		}
	}
	for (k = 0; k < im.c; ++k) {
		for (r = 0; r < h; ++r) {
			float sy = r * h_scale;
			int iy = (int)sy;
			float dy = sy - iy;
			for (c = 0; c < w; ++c) {
				float val = (1 - dy) * get_pixel(part, c, iy, k);
				set_pixel(resized, c, r, k, val);
			}
			if (r == h - 1 || im.h == 1) continue;
			for (c = 0; c < w; ++c) {
				float val = dy * get_pixel(part, c, iy + 1, k);
				add_pixel(resized, c, r, k, val);
			}
		}
	}

	free_image(part);
	return resized;
}


void test_resize(char *filename)
{
	image im = load_image(filename, 0, 0, 3);
	float mag = mag_array(im.data, im.w*im.h*im.c);
	printf("L2 Norm: %f\n", mag);
	image gray = grayscale_image(im);

	image c1 = copy_image(im);
	image c2 = copy_image(im);
	image c3 = copy_image(im);
	image c4 = copy_image(im);
	distort_image(c1, .1, 1.5, 1.5);
	distort_image(c2, -.1, .66666, .66666);
	distort_image(c3, .1, 1.5, .66666);
	distort_image(c4, .1, .66666, 1.5);


	show_image(im, "Original", 1);
	show_image(gray, "Gray", 1);
	show_image(c1, "C1", 1);
	show_image(c2, "C2", 1);
	show_image(c3, "C3", 1);
	show_image(c4, "C4", 1);
#ifdef CV_VERSION
	while (1) {
		image aug = random_augment_image(im, 0, .75, 320, 448, 320, 320);
		show_image(aug, "aug", 1);
		free_image(aug);


		float exposure = 1.15;
		float saturation = 1.15;
		float hue = .05;

		image c = copy_image(im);

		float dexp = rand_scale(exposure);
		float dsat = rand_scale(saturation);
		float dhue = rand_uniform(-hue, hue);

		distort_image(c, dhue, dsat, dexp);
		show_image(c, "rand", 1);
		printf("%f %f %f\n", dhue, dsat, dexp);
		free_image(c);
	}
#endif
}


image load_image_stb(char *filename, int channels)
{
	int w, h, c;
	unsigned char *data = stbi_load(filename, &w, &h, &c, channels);
	if (!data) {
		fprintf(stderr, "Cannot load image \"%s\"\nSTB Reason: %s\n", filename, stbi_failure_reason());
		exit(0);
	}
	if (channels) c = channels;
	int i, j, k;
	image im = make_image(w, h, c);
	for (k = 0; k < c; ++k) {
		for (j = 0; j < h; ++j) {
			for (i = 0; i < w; ++i) {
				int dst_index = i + w * j + w * h*k;
				int src_index = k + c * i + c * w*j;
				im.data[dst_index] = (float)data[src_index] / 255.;
			}
		}
	}
	free(data);
	return im;
}

image load_image(char *filename, int w, int h, int c)
{
#ifdef CV_VERSION
	image out = cv::load_image_cv(filename, c);
#else
	image out = load_image_stb(filename, c);
#endif

	if ((h && w) && (h != out.h || w != out.w)) {
		image resized = resize_image(out, w, h);
		free_image(out);
		out = resized;
	}
	return out;
}

image load_image_color(char *filename, int w, int h)
{
	return load_image(filename, w, h, 3);
}

image get_image_layer(image m, int l)
{
	image out = make_image(m.w, m.h, 1);
	int i;
	for (i = 0; i < m.h*m.w; ++i) {
		out.data[i] = m.data[i + l * m.h*m.w];
	}
	return out;
}
void print_image(image m)
{
	int i, j, k;
	for (i = 0; i < m.c; ++i) {
		for (j = 0; j < m.h; ++j) {
			for (k = 0; k < m.w; ++k) {
				printf("%.2lf, ", m.data[i*m.h*m.w + j * m.w + k]);
				if (k > 30) break;
			}
			printf("\n");
			if (j > 30) break;
		}
		printf("\n");
	}
	printf("\n");
}

image collapse_images_vert(image *ims, int n)
{
	int color = 1;
	int border = 1;
	int h, w, c;
	w = ims[0].w;
	h = (ims[0].h + border) * n - border;
	c = ims[0].c;
	if (c != 3 || !color) {
		w = (w + border)*c - border;
		c = 1;
	}

	image filters = make_image(w, h, c);
	int i, j;
	for (i = 0; i < n; ++i) {
		int h_offset = i * (ims[0].h + border);
		image copy = copy_image(ims[i]);
		//normalize_image(copy);
		if (c == 3 && color) {
			embed_image(copy, filters, 0, h_offset);
		}
		else {
			for (j = 0; j < copy.c; ++j) {
				int w_offset = j * (ims[0].w + border);
				image layer = get_image_layer(copy, j);
				embed_image(layer, filters, w_offset, h_offset);
				free_image(layer);
			}
		}
		free_image(copy);
	}
	return filters;
}

image collapse_images_horz(image *ims, int n)
{
	int color = 1;
	int border = 1;
	int h, w, c;
	int size = ims[0].h;
	h = size;
	w = (ims[0].w + border) * n - border;
	c = ims[0].c;
	if (c != 3 || !color) {
		h = (h + border)*c - border;
		c = 1;
	}

	image filters = make_image(w, h, c);
	int i, j;
	for (i = 0; i < n; ++i) {
		int w_offset = i * (size + border);
		image copy = copy_image(ims[i]);
		//normalize_image(copy);
		if (c == 3 && color) {
			embed_image(copy, filters, w_offset, 0);
		}
		else {
			for (j = 0; j < copy.c; ++j) {
				int h_offset = j * (size + border);
				image layer = get_image_layer(copy, j);
				embed_image(layer, filters, w_offset, h_offset);
				free_image(layer);
			}
		}
		free_image(copy);
	}
	return filters;
}

void show_image_normalized(image im, const char *name)
{
	image c = copy_image(im);
	normalize_image(c);
	show_image(c, name, 1);
	free_image(c);
}

void show_images(image *ims, int n, char *window)
{
	image m = collapse_images_vert(ims, n);
	/*
	   int w = 448;
	   int h = ((float)m.h/m.w) * 448;
	   if(h > 896){
	   h = 896;
	   w = ((float)m.w/m.h) * 896;
	   }
	   image sized = resize_image(m, w, h);
	 */
	normalize_image(m);
	save_image(m, window);
	show_image(m, window, 1);
	free_image(m);
}

void free_image(image m)
{
	if (m.data) {
		free(m.data);
	}
}














//layer.c
void free_layer(layer l)
{
	if (l.type == DROPOUT) {
		if (l.rand)           free(l.rand);
#ifdef GPU
		if (l.rand_gpu)             cuda_free(l.rand_gpu);
#endif
		return;
	}
	if (l.cweights)           free(l.cweights);
	if (l.indexes)            free(l.indexes);
	if (l.input_layers)       free(l.input_layers);
	if (l.input_sizes)        free(l.input_sizes);
	if (l.map)                free(l.map);
	if (l.rand)               free(l.rand);
	if (l.cost)               free(l.cost);
	if (l.state)              free(l.state);
	if (l.prev_state)         free(l.prev_state);
	if (l.forgot_state)       free(l.forgot_state);
	if (l.forgot_delta)       free(l.forgot_delta);
	if (l.state_delta)        free(l.state_delta);
	if (l.concat)             free(l.concat);
	if (l.concat_delta)       free(l.concat_delta);
	if (l.binary_weights)     free(l.binary_weights);
	if (l.biases)             free(l.biases);
	if (l.bias_updates)       free(l.bias_updates);
	if (l.scales)             free(l.scales);
	if (l.scale_updates)      free(l.scale_updates);
	if (l.weights)            free(l.weights);
	if (l.weight_updates)     free(l.weight_updates);
	if (l.delta)              free(l.delta);
	if (l.output)             free(l.output);
	if (l.squared)            free(l.squared);
	if (l.norms)              free(l.norms);
	if (l.spatial_mean)       free(l.spatial_mean);
	if (l.mean)               free(l.mean);
	if (l.variance)           free(l.variance);
	if (l.mean_delta)         free(l.mean_delta);
	if (l.variance_delta)     free(l.variance_delta);
	if (l.rolling_mean)       free(l.rolling_mean);
	if (l.rolling_variance)   free(l.rolling_variance);
	if (l.x)                  free(l.x);
	if (l.x_norm)             free(l.x_norm);
	if (l.m)                  free(l.m);
	if (l.v)                  free(l.v);
	if (l.z_cpu)              free(l.z_cpu);
	if (l.r_cpu)              free(l.r_cpu);
	if (l.h_cpu)              free(l.h_cpu);
	if (l.binary_input)       free(l.binary_input);

#ifdef GPU
	if (l.indexes_gpu)           cuda_free((float *)l.indexes_gpu);

	if (l.z_gpu)                   cuda_free(l.z_gpu);
	if (l.r_gpu)                   cuda_free(l.r_gpu);
	if (l.h_gpu)                   cuda_free(l.h_gpu);
	if (l.m_gpu)                   cuda_free(l.m_gpu);
	if (l.v_gpu)                   cuda_free(l.v_gpu);
	if (l.prev_state_gpu)          cuda_free(l.prev_state_gpu);
	if (l.forgot_state_gpu)        cuda_free(l.forgot_state_gpu);
	if (l.forgot_delta_gpu)        cuda_free(l.forgot_delta_gpu);
	if (l.state_gpu)               cuda_free(l.state_gpu);
	if (l.state_delta_gpu)         cuda_free(l.state_delta_gpu);
	if (l.gate_gpu)                cuda_free(l.gate_gpu);
	if (l.gate_delta_gpu)          cuda_free(l.gate_delta_gpu);
	if (l.save_gpu)                cuda_free(l.save_gpu);
	if (l.save_delta_gpu)          cuda_free(l.save_delta_gpu);
	if (l.concat_gpu)              cuda_free(l.concat_gpu);
	if (l.concat_delta_gpu)        cuda_free(l.concat_delta_gpu);
	if (l.binary_input_gpu)        cuda_free(l.binary_input_gpu);
	if (l.binary_weights_gpu)      cuda_free(l.binary_weights_gpu);
	if (l.mean_gpu)                cuda_free(l.mean_gpu);
	if (l.variance_gpu)            cuda_free(l.variance_gpu);
	if (l.rolling_mean_gpu)        cuda_free(l.rolling_mean_gpu);
	if (l.rolling_variance_gpu)    cuda_free(l.rolling_variance_gpu);
	if (l.variance_delta_gpu)      cuda_free(l.variance_delta_gpu);
	if (l.mean_delta_gpu)          cuda_free(l.mean_delta_gpu);
	if (l.x_gpu)                   cuda_free(l.x_gpu);
	if (l.x_norm_gpu)              cuda_free(l.x_norm_gpu);
	if (l.weights_gpu)             cuda_free(l.weights_gpu);
	if (l.weight_updates_gpu)      cuda_free(l.weight_updates_gpu);
	if (l.biases_gpu)              cuda_free(l.biases_gpu);
	if (l.bias_updates_gpu)        cuda_free(l.bias_updates_gpu);
	if (l.scales_gpu)              cuda_free(l.scales_gpu);
	if (l.scale_updates_gpu)       cuda_free(l.scale_updates_gpu);
	if (l.output_gpu)              cuda_free(l.output_gpu);
	if (l.delta_gpu)               cuda_free(l.delta_gpu);
	if (l.rand_gpu)                cuda_free(l.rand_gpu);
	if (l.squared_gpu)             cuda_free(l.squared_gpu);
	if (l.norms_gpu)               cuda_free(l.norms_gpu);
#endif
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
	node *new_node = (node*)malloc(sizeof(node));
	new_node->val = val;
	new_node->next = 0;

	if (!l->back) {
		l->front = new_node;
		new_node->prev = 0;
	}
	else {
		l->back->next = new_node;
		new_node->prev = l->back;
	}
	l->back = new_node;
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






//matrix.c
void free_matrix(matrix m)
{
	int i;
	for (i = 0; i < m.rows; ++i) free(m.vals[i]);
	free(m.vals);
}

float matrix_topk_accuracy(matrix truth, matrix guess, int k)
{
	int *indexes = (int*)calloc(k, sizeof(int));
	int n = truth.cols;
	int i, j;
	int correct = 0;
	for (i = 0; i < truth.rows; ++i) {
		top_k(guess.vals[i], n, k, indexes);
		for (j = 0; j < k; ++j) {
			int class_n = indexes[j];
			if (truth.vals[i][class_n]) {
				++correct;
				break;
			}
		}
	}
	free(indexes);
	return (float)correct / truth.rows;
}

void scale_matrix(matrix m, float scale)
{
	int i, j;
	for (i = 0; i < m.rows; ++i) {
		for (j = 0; j < m.cols; ++j) {
			m.vals[i][j] *= scale;
		}
	}
}

matrix resize_matrix(matrix m, int size)
{
	int i;
	if (m.rows == size) return m;
	if (m.rows < size) {
		m.vals = (float**)realloc(m.vals, size * sizeof(float*));
		for (i = m.rows; i < size; ++i) {
			m.vals[i] = (float*)calloc(m.cols, sizeof(float));
		}
	}
	else if (m.rows > size) {
		for (i = size; i < m.rows; ++i) {
			free(m.vals[i]);
		}
		m.vals = (float**)realloc(m.vals, size * sizeof(float*));
	}
	m.rows = size;
	return m;
}

void matrix_add_matrix(matrix from, matrix to)
{
	assert(from.rows == to.rows && from.cols == to.cols);
	int i, j;
	for (i = 0; i < from.rows; ++i) {
		for (j = 0; j < from.cols; ++j) {
			to.vals[i][j] += from.vals[i][j];
		}
	}
}

matrix copy_matrix(matrix m)
{
	matrix c = { 0 };
	c.rows = m.rows;
	c.cols = m.cols;
	c.vals = (float**)calloc(c.rows, sizeof(float *));
	int i;
	for (i = 0; i < c.rows; ++i) {
		c.vals[i] = (float*)calloc(c.cols, sizeof(float));
		copy_cpu(c.cols, m.vals[i], 1, c.vals[i], 1);
	}
	return c;
}

matrix make_matrix(int rows, int cols)
{
	int i;
	matrix m;
	m.rows = rows;
	m.cols = cols;
	m.vals = (float**)calloc(m.rows, sizeof(float *));
	for (i = 0; i < m.rows; ++i) {
		m.vals[i] = (float*)calloc(m.cols, sizeof(float));
	}
	return m;
}

matrix hold_out_matrix(matrix *m, int n)
{
	int i;
	matrix h;
	h.rows = n;
	h.cols = m->cols;
	h.vals = (float**)calloc(h.rows, sizeof(float *));
	for (i = 0; i < n; ++i) {
		int index = rand() % m->rows;
		h.vals[i] = m->vals[index];
		m->vals[index] = m->vals[--(m->rows)];
	}
	return h;
}

float *pop_column(matrix *m, int c)
{
	float *col = (float*)calloc(m->rows, sizeof(float));
	int i, j;
	for (i = 0; i < m->rows; ++i) {
		col[i] = m->vals[i][c];
		for (j = c; j < m->cols - 1; ++j) {
			m->vals[i][j] = m->vals[i][j + 1];
		}
	}
	--m->cols;
	return col;
}

matrix csv_to_matrix(char *filename)
{
	FILE *fp = fopen(filename, "r");
	if (!fp) file_error(filename);

	matrix m;
	m.cols = -1;

	char *line;

	int n = 0;
	int size = 1024;
	m.vals = (float**)calloc(size, sizeof(float*));
	while ((line = fgetl(fp))) {
		if (m.cols == -1) m.cols = count_fields(line);
		if (n == size) {
			size *= 2;
			m.vals = (float**)realloc(m.vals, size * sizeof(float*));
		}
		m.vals[n] = parse_fields(line, m.cols);
		free(line);
		++n;
	}
	m.vals = (float**)realloc(m.vals, n * sizeof(float*));
	m.rows = n;
	return m;
}

void matrix_to_csv(matrix m)
{
	int i, j;

	for (i = 0; i < m.rows; ++i) {
		for (j = 0; j < m.cols; ++j) {
			if (j > 0) printf(",");
			printf("%.17g", m.vals[i][j]);
		}
		printf("\n");
	}
}

void print_matrix(matrix m)
{
	int i, j;
	printf("%d X %d Matrix:\n", m.rows, m.cols);
	printf(" __");
	for (j = 0; j < 16 * m.cols - 1; ++j) printf(" ");
	printf("__ \n");

	printf("|  ");
	for (j = 0; j < 16 * m.cols - 1; ++j) printf(" ");
	printf("  |\n");

	for (i = 0; i < m.rows; ++i) {
		printf("|  ");
		for (j = 0; j < m.cols; ++j) {
			printf("%15.7f ", m.vals[i][j]);
		}
		printf(" |\n");
	}
	printf("|__");
	for (j = 0; j < 16 * m.cols - 1; ++j) printf(" ");
	printf("__|\n");
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

char *get_layer_string(LAYER_TYPE a)
{
	switch (a) {
	case CONVOLUTIONAL:
		return (char*)"convolutional";
	case ACTIVE:
		return (char*)"activation";
	case LOCAL:
		return (char*)"local";
	case DECONVOLUTIONAL:
		return (char*)"deconvolutional";
	case CONNECTED:
		return (char*)"connected";
	case RNN:
		return (char*)"rnn";
	case GRU:
		return (char*)"gru";
	case LSTM:
		return (char*)"lstm";
	case CRNN:
		return (char*)"crnn";
	case MAXPOOL:
		return (char*)"maxpool";
	case REORG:
		return (char*)"reorg";
	case AVGPOOL:
		return (char*)"avgpool";
	case SOFTMAX:
		return (char*)"softmax";
	case DETECTION:
		return (char*)"detection";
	case REGION:
		return (char*)"region";
	case YOLO:
		return (char*)"yolo";
	case DROPOUT:
		return (char*)"dropout";
	case CROP:
		return (char*)"crop";
	case COST:
		return (char*)"cost";
	case ROUTE:
		return (char*)"route";
	case SHORTCUT:
		return (char*)"shortcut";
	case NORMALIZATION:
		return (char*)"normalization";
	case BATCHNORM:
		return (char*)"batchnorm";
	default:
		break;
	}
	return (char*)"none";
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

void forward_network(network *netp)
{
#ifdef GPU
	if (netp->gpu_index >= 0) {
		forward_network_gpu(netp);
		return;
	}
#endif
	network net = *netp;
	int i;
	for (i = 0; i < net.n; ++i) {
		net.index = i;
		layer l = net.layers[i];
		if (l.delta) {
			fill_cpu(l.outputs * l.batch, 0, l.delta, 1);
		}
		l.forward(l, net);
		net.input = l.output;
		if (l.truth) {
			net.truth = l.output;
		}
	}
	calc_network_cost(netp);
}

void update_network(network *netp)
{
#ifdef GPU
	if (netp->gpu_index >= 0) {
		update_network_gpu(netp);
		return;
	}
#endif
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

void backward_network(network *netp)
{
#ifdef GPU
	if (netp->gpu_index >= 0) {
		backward_network_gpu(netp);
		return;
	}
#endif
	network net = *netp;
	int i;
	network orig = net;
	for (i = net.n - 1; i >= 0; --i) {
		layer l = net.layers[i];
		if (l.stopbackward) break;
		if (i == 0) {
			net = orig;
		}
		else {
			layer prev = net.layers[i - 1];
			net.input = prev.output;
			net.delta = prev.delta;
		}
		net.index = i;
		l.backward(l, net);
	}
}

float train_network_datum(network *net)
{
	*net->seen += net->batch;
	net->train = 1;
	forward_network(net);
	backward_network(net);
	float error = *net->cost;
	if (((*net->seen) / net->batch) % net->subdivisions == 0) update_network(net);
	return error;
}

float train_network_sgd(network *net, data d, int n)
{
	int batch = net->batch;

	int i;
	float sum = 0;
	for (i = 0; i < n; ++i) {
		get_random_batch(d, batch, net->input, net->truth);
		float err = train_network_datum(net);
		sum += err;
	}
	return (float)sum / (n*batch);
}

float train_network(network *net, data d)
{
	assert(d.X.rows % net->batch == 0);
	int batch = net->batch;
	int n = d.X.rows / batch;

	int i;
	float sum = 0;
	for (i = 0; i < n; ++i) {
		get_next_batch(d, batch, i*batch, net->input, net->truth);
		float err = train_network_datum(net);
		sum += err;
	}
	return (float)sum / (n*batch);
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
#ifdef CUDNN
		if (net->layers[i].type == CONVOLUTIONAL) {
			cudnn_convolutional_setup(net->layers + i);
		}
		if (net->layers[i].type == DECONVOLUTIONAL) {
			layer *l = net->layers + i;
			cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, l->out_h, l->out_w);
			cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1);
		}
#endif
	}
}

int resize_network(network *net, int w, int h)
{
#ifdef GPU
	cuda_set_device(net->gpu_index);
	cuda_free(net->workspace);
#endif
	int i;
	//if(w == net->w && h == net->h) return 0;
	net->w = w;
	net->h = h;
	int inputs = 0;
	size_t workspace_size = 0;
	//fprintf(stderr, "Resizing to %d x %d...\n", w, h);
	//fflush(stderr);
	for (i = 0; i < net->n; ++i) {
		layer l = net->layers[i];
		if (l.type == CONVOLUTIONAL) {
			resize_convolutional_layer(&l, w, h);
		}
		else if (l.type == CROP) {
			resize_crop_layer(&l, w, h);
		}
		else if (l.type == MAXPOOL) {
			resize_maxpool_layer(&l, w, h);
		}
		else if (l.type == REGION) {
			resize_region_layer(&l, w, h);
		}
		else if (l.type == YOLO) {
			resize_yolo_layer(&l, w, h);
		}
		else if (l.type == ROUTE) {
			resize_route_layer(&l, net);
		}
		else if (l.type == SHORTCUT) {
			resize_shortcut_layer(&l, w, h);
		}
		else if (l.type == UPSAMPLE) {
			resize_upsample_layer(&l, w, h);
		}
		else if (l.type == REORG) {
			resize_reorg_layer(&l, w, h);
		}
		else if (l.type == AVGPOOL) {
			resize_avgpool_layer(&l, w, h);
		}
		else if (l.type == NORMALIZATION) {
			resize_normalization_layer(&l, w, h);
		}
		else if (l.type == COST) {
			resize_cost_layer(&l, inputs);
		}
		else {
			error("Cannot resize this type of layer");
		}
		if (l.workspace_size > workspace_size) workspace_size = l.workspace_size;
		if (l.workspace_size > 2000000000) assert(0);
		inputs = l.outputs;
		net->layers[i] = l;
		w = l.out_w;
		h = l.out_h;
		if (l.type == AVGPOOL) break;
	}
	layer out = get_network_output_layer(net);
	net->inputs = net->layers[0].inputs;
	net->outputs = out.outputs;
	net->truths = out.outputs;
	if (net->layers[net->n - 1].truths) net->truths = net->layers[net->n - 1].truths;
	net->output = out.output;
	free(net->input);
	free(net->truth);
	net->input = (float*)calloc(net->inputs*net->batch, sizeof(float));
	net->truth = (float*)calloc(net->truths*net->batch, sizeof(float));
#ifdef GPU
	if (gpu_index >= 0) {
		cuda_free(net->input_gpu);
		cuda_free(net->truth_gpu);
		net->input_gpu = cuda_make_array(net->input, net->inputs*net->batch);
		net->truth_gpu = cuda_make_array(net->truth, net->truths*net->batch);
		if (workspace_size) {
			net->workspace = cuda_make_array(0, (workspace_size - 1) / sizeof(float) + 1);
		}
	}
	else {
		free(net->workspace);
		net->workspace = calloc(1, workspace_size);
	}
#else
	free(net->workspace);
	net->workspace = (float*)calloc(1, workspace_size);
#endif
	//fprintf(stderr, " Done!\n");
	return 0;
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
	layer l = { };
	return l;
}

image get_network_image_layer(network *net, int i)
{
	layer l = net->layers[i];
#ifdef GPU
	//cuda_pull_array(l.output_gpu, l.output, l.outputs);
#endif
	if (l.out_w && l.out_h && l.out_c) {
		return float_to_image(l.out_w, l.out_h, l.out_c, l.output);
	}
	image def = { 0 };
	return def;
}

image get_network_image(network *net)
{
	int i;
	for (i = net->n - 1; i >= 0; --i) {
		image m = get_network_image_layer(net, i);
		if (m.h != 0) return m;
	}
	image def = { 0 };
	return def;
}

void visualize_network(network *net)
{
	image *prev = 0;
	int i;
	char buff[256];
	for (i = 0; i < net->n; ++i) {
		sprintf(buff, "Layer %d", i);
		layer l = net->layers[i];
		if (l.type == CONVOLUTIONAL) {
			prev = visualize_convolutional_layer(l, buff, prev);
		}
	}
}

void top_predictions(network *net, int k, int *index)
{
	top_k(net->output, net->outputs, k, index);
}


float *network_predict(network *net, float *input)
{
	network orig = *net;
	net->input = input;
	net->truth = 0;
	net->train = 0;
	net->delta = 0;
	forward_network(net);
	float *out = net->output;
	*net = orig;
	return out;
}

int num_detections(network *net, float thresh)
{
	int i;
	int s = 0;
	for (i = 0; i < net->n; ++i) {
		layer l = net->layers[i];
		if (l.type == YOLO) {
			s += yolo_num_detections(l, thresh);
		}
		if (l.type == DETECTION || l.type == REGION) {
			s += l.w*l.h*l.n;
		}
	}
	return s;
}

detection *make_network_boxes(network *net, float thresh, int *num)
{
	layer l = net->layers[net->n - 1];
	int i;
	int nboxes = num_detections(net, thresh);
	if (num) *num = nboxes;
	detection *dets = (detection*)calloc(nboxes, sizeof(detection));
	for (i = 0; i < nboxes; ++i) {
		dets[i].prob = (float*)calloc(l.classes, sizeof(float));
		if (l.coords > 4) {
			dets[i].mask = (float*)calloc(l.coords - 4, sizeof(float));
		}
	}
	return dets;
}

void fill_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, detection *dets)
{
	int j;
	for (j = 0; j < net->n; ++j) {
		layer l = net->layers[j];
		if (l.type == YOLO) {
			int count = get_yolo_detections(l, w, h, net->w, net->h, thresh, map, relative, dets);
			dets += count;
		}
		if (l.type == REGION) {
			get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets);
			dets += l.w*l.h*l.n;
		}
		if (l.type == DETECTION) {
			get_detection_detections(l, w, h, thresh, dets);
			dets += l.w*l.h*l.n;
		}
	}
}

detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num)
{
	detection *dets = make_network_boxes(net, thresh, num);
	fill_network_boxes(net, w, h, thresh, hier, map, relative, dets);
	return dets;
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

float *network_predict_image(network *net, image im)
{
	image imr = letterbox_image(im, net->w, net->h);
	set_batch_network(net, 1);
	float *p = network_predict(net, imr.data);
	free_image(imr);
	return p;
}

int network_width(network *net) { return net->w; }
int network_height(network *net) { return net->h; }

matrix network_predict_data_multi(network *net, data test, int n)
{
	int i, j, b, m;
	int k = net->outputs;
	matrix pred = make_matrix(test.X.rows, k);
	float *X = (float*)calloc(net->batch*test.X.rows, sizeof(float));
	for (i = 0; i < test.X.rows; i += net->batch) {
		for (b = 0; b < net->batch; ++b) {
			if (i + b == test.X.rows) break;
			memcpy(X + b * test.X.cols, test.X.vals[i + b], test.X.cols * sizeof(float));
		}
		for (m = 0; m < n; ++m) {
			float *out = network_predict(net, X);
			for (b = 0; b < net->batch; ++b) {
				if (i + b == test.X.rows) break;
				for (j = 0; j < k; ++j) {
					pred.vals[i + b][j] += out[j + b * k] / n;
				}
			}
		}
	}
	free(X);
	return pred;
}

matrix network_predict_data(network *net, data test)
{
	int i, j, b;
	int k = net->outputs;
	matrix pred = make_matrix(test.X.rows, k);
	float *X = (float*)calloc(net->batch*test.X.cols, sizeof(float));
	for (i = 0; i < test.X.rows; i += net->batch) {
		for (b = 0; b < net->batch; ++b) {
			if (i + b == test.X.rows) break;
			memcpy(X + b * test.X.cols, test.X.vals[i + b], test.X.cols * sizeof(float));
		}
		float *out = network_predict(net, X);
		for (b = 0; b < net->batch; ++b) {
			if (i + b == test.X.rows) break;
			for (j = 0; j < k; ++j) {
				pred.vals[i + b][j] = out[j + b * k];
			}
		}
	}
	free(X);
	return pred;
}

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

void compare_networks(network *n1, network *n2, data test)
{
	matrix g1 = network_predict_data(n1, test);
	matrix g2 = network_predict_data(n2, test);
	int i;
	int a, b, c, d;
	a = b = c = d = 0;
	for (i = 0; i < g1.rows; ++i) {
		int truth = max_index(test.y.vals[i], test.y.cols);
		int p1 = max_index(g1.vals[i], g1.cols);
		int p2 = max_index(g2.vals[i], g2.cols);
		if (p1 == truth) {
			if (p2 == truth) ++d;
			else ++c;
		}
		else {
			if (p2 == truth) ++b;
			else ++a;
		}
	}
	printf("%5d %5d\n%5d %5d\n", a, b, c, d);
	float num = pow((abs(b - c) - 1.), 2.);
	float den = b + c;
	printf("%f\n", num / den);
}

float network_accuracy(network *net, data d)
{
	matrix guess = network_predict_data(net, d);
	float acc = matrix_topk_accuracy(d.y, guess, 1);
	free_matrix(guess);
	return acc;
}

float *network_accuracies(network *net, data d, int n)
{
	static float acc[2];
	matrix guess = network_predict_data(net, d);
	acc[0] = matrix_topk_accuracy(d.y, guess, 1);
	acc[1] = matrix_topk_accuracy(d.y, guess, n);
	free_matrix(guess);
	return acc;
}

layer get_network_output_layer(network *net)
{
	int i;
	for (i = net->n - 1; i >= 0; --i) {
		if (net->layers[i].type != COST) break;
	}
	return net->layers[i];
}

float network_accuracy_multi(network *net, data d, int n)
{
	matrix guess = network_predict_data_multi(net, d, n);
	float acc = matrix_topk_accuracy(d.y, guess, 1);
	free_matrix(guess);
	return acc;
}

void free_network(network *net)
{
	int i;
	for (i = 0; i < net->n; ++i) {
		free_layer(net->layers[i]);
	}
	free(net->layers);
	if (net->input) free(net->input);
	if (net->truth) free(net->truth);
#ifdef GPU
	if (net->input_gpu) cuda_free(net->input_gpu);
	if (net->truth_gpu) cuda_free(net->truth_gpu);
#endif
	free(net);
}

// Some day...
// ^ What the hell is this comment for?


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

#ifdef GPU

void forward_network_gpu(network *netp)
{
	network net = *netp;
	cuda_set_device(net.gpu_index);
	cuda_push_array(net.input_gpu, net.input, net.inputs*net.batch);
	if (net.truth) {
		cuda_push_array(net.truth_gpu, net.truth, net.truths*net.batch);
	}

	int i;
	for (i = 0; i < net.n; ++i) {
		net.index = i;
		layer l = net.layers[i];
		if (l.delta_gpu) {
			fill_gpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
		}
		l.forward_gpu(l, net);
		net.input_gpu = l.output_gpu;
		net.input = l.output;
		if (l.truth) {
			net.truth_gpu = l.output_gpu;
			net.truth = l.output;
		}
	}
	pull_network_output(netp);
	calc_network_cost(netp);
}

void backward_network_gpu(network *netp)
{
	int i;
	network net = *netp;
	network orig = net;
	cuda_set_device(net.gpu_index);
	for (i = net.n - 1; i >= 0; --i) {
		layer l = net.layers[i];
		if (l.stopbackward) break;
		if (i == 0) {
			net = orig;
		}
		else {
			layer prev = net.layers[i - 1];
			net.input = prev.output;
			net.delta = prev.delta;
			net.input_gpu = prev.output_gpu;
			net.delta_gpu = prev.delta_gpu;
		}
		net.index = i;
		l.backward_gpu(l, net);
	}
}

void update_network_gpu(network *netp)
{
	network net = *netp;
	cuda_set_device(net.gpu_index);
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
	a.t = (*net.t);

	for (i = 0; i < net.n; ++i) {
		layer l = net.layers[i];
		if (l.update_gpu) {
			l.update_gpu(l, a);
		}
	}
}

void harmless_update_network_gpu(network *netp)
{
	network net = *netp;
	cuda_set_device(net.gpu_index);
	int i;
	for (i = 0; i < net.n; ++i) {
		layer l = net.layers[i];
		if (l.weight_updates_gpu) fill_gpu(l.nweights, 0, l.weight_updates_gpu, 1);
		if (l.bias_updates_gpu) fill_gpu(l.nbiases, 0, l.bias_updates_gpu, 1);
		if (l.scale_updates_gpu) fill_gpu(l.nbiases, 0, l.scale_updates_gpu, 1);
	}
}

typedef struct {
	network *net;
	data d;
	float *err;
} train_args;

void *train_thread(void *ptr)
{
	train_args args = *(train_args*)ptr;
	free(ptr);
	cuda_set_device(args.net->gpu_index);
	*args.err = train_network(args.net, args.d);
	return 0;
}

pthread_t train_network_in_thread(network *net, data d, float *err)
{
	pthread_t thread;
	train_args *ptr = (train_args *)calloc(1, sizeof(train_args));
	ptr->net = net;
	ptr->d = d;
	ptr->err = err;
	if (pthread_create(&thread, 0, train_thread, ptr)) error("Thread creation failed");
	return thread;
}

void merge_weights(layer l, layer base)
{
	if (l.type == CONVOLUTIONAL) {
		axpy_cpu(l.n, 1, l.bias_updates, 1, base.biases, 1);
		axpy_cpu(l.nweights, 1, l.weight_updates, 1, base.weights, 1);
		if (l.scales) {
			axpy_cpu(l.n, 1, l.scale_updates, 1, base.scales, 1);
		}
	}
	else if (l.type == CONNECTED) {
		axpy_cpu(l.outputs, 1, l.bias_updates, 1, base.biases, 1);
		axpy_cpu(l.outputs*l.inputs, 1, l.weight_updates, 1, base.weights, 1);
	}
}

void scale_weights(layer l, float s)
{
	if (l.type == CONVOLUTIONAL) {
		scal_cpu(l.n, s, l.biases, 1);
		scal_cpu(l.nweights, s, l.weights, 1);
		if (l.scales) {
			scal_cpu(l.n, s, l.scales, 1);
		}
	}
	else if (l.type == CONNECTED) {
		scal_cpu(l.outputs, s, l.biases, 1);
		scal_cpu(l.outputs*l.inputs, s, l.weights, 1);
	}
}


void pull_weights(layer l)
{
	if (l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL) {
		cuda_pull_array(l.biases_gpu, l.bias_updates, l.n);
		cuda_pull_array(l.weights_gpu, l.weight_updates, l.nweights);
		if (l.scales) cuda_pull_array(l.scales_gpu, l.scale_updates, l.n);
	}
	else if (l.type == CONNECTED) {
		cuda_pull_array(l.biases_gpu, l.bias_updates, l.outputs);
		cuda_pull_array(l.weights_gpu, l.weight_updates, l.outputs*l.inputs);
	}
}

void push_weights(layer l)
{
	if (l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL) {
		cuda_push_array(l.biases_gpu, l.biases, l.n);
		cuda_push_array(l.weights_gpu, l.weights, l.nweights);
		if (l.scales) cuda_push_array(l.scales_gpu, l.scales, l.n);
	}
	else if (l.type == CONNECTED) {
		cuda_push_array(l.biases_gpu, l.biases, l.outputs);
		cuda_push_array(l.weights_gpu, l.weights, l.outputs*l.inputs);
	}
}

void distribute_weights(layer l, layer base)
{
	if (l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL) {
		cuda_push_array(l.biases_gpu, base.biases, l.n);
		cuda_push_array(l.weights_gpu, base.weights, l.nweights);
		if (base.scales) cuda_push_array(l.scales_gpu, base.scales, l.n);
	}
	else if (l.type == CONNECTED) {
		cuda_push_array(l.biases_gpu, base.biases, l.outputs);
		cuda_push_array(l.weights_gpu, base.weights, l.outputs*l.inputs);
	}
}


/*

   void pull_updates(layer l)
   {
   if(l.type == CONVOLUTIONAL){
   cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
   cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
   if(l.scale_updates) cuda_pull_array(l.scale_updates_gpu, l.scale_updates, l.n);
   } else if(l.type == CONNECTED){
   cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
   cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.outputs*l.inputs);
   }
   }

   void push_updates(layer l)
   {
   if(l.type == CONVOLUTIONAL){
   cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
   cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
   if(l.scale_updates) cuda_push_array(l.scale_updates_gpu, l.scale_updates, l.n);
   } else if(l.type == CONNECTED){
   cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
   cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.outputs*l.inputs);
   }
   }

   void update_layer(layer l, network net)
   {
   int update_batch = net.batch*net.subdivisions;
   float rate = get_current_rate(net);
   l.t = get_current_batch(net);
   if(l.update_gpu){
   l.update_gpu(l, update_batch, rate*l.learning_rate_scale, net.momentum, net.decay);
   }
   }
   void merge_updates(layer l, layer base)
   {
   if (l.type == CONVOLUTIONAL) {
   axpy_cpu(l.n, 1, l.bias_updates, 1, base.bias_updates, 1);
   axpy_cpu(l.nweights, 1, l.weight_updates, 1, base.weight_updates, 1);
   if (l.scale_updates) {
   axpy_cpu(l.n, 1, l.scale_updates, 1, base.scale_updates, 1);
   }
   } else if(l.type == CONNECTED) {
   axpy_cpu(l.outputs, 1, l.bias_updates, 1, base.bias_updates, 1);
   axpy_cpu(l.outputs*l.inputs, 1, l.weight_updates, 1, base.weight_updates, 1);
   }
   }

   void distribute_updates(layer l, layer base)
   {
   if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
   cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.n);
   cuda_push_array(l.weight_updates_gpu, base.weight_updates, l.nweights);
   if(base.scale_updates) cuda_push_array(l.scale_updates_gpu, base.scale_updates, l.n);
   } else if(l.type == CONNECTED){
   cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.outputs);
   cuda_push_array(l.weight_updates_gpu, base.weight_updates, l.outputs*l.inputs);
   }
   }
 */

 /*
	void sync_layer(network *nets, int n, int j)
	{
	int i;
	network net = nets[0];
	layer base = net.layers[j];
	scale_weights(base, 0);
	for (i = 0; i < n; ++i) {
	cuda_set_device(nets[i].gpu_index);
	layer l = nets[i].layers[j];
	pull_weights(l);
	merge_weights(l, base);
	}
	scale_weights(base, 1./n);
	for (i = 0; i < n; ++i) {
	cuda_set_device(nets[i].gpu_index);
	layer l = nets[i].layers[j];
	distribute_weights(l, base);
	}
	}
  */

void sync_layer(network **nets, int n, int j)
{
	int i;
	network *net = nets[0];
	layer base = net->layers[j];
	scale_weights(base, 0);
	for (i = 0; i < n; ++i) {
		cuda_set_device(nets[i]->gpu_index);
		layer l = nets[i]->layers[j];
		pull_weights(l);
		merge_weights(l, base);
	}
	scale_weights(base, 1. / n);
	for (i = 0; i < n; ++i) {
		cuda_set_device(nets[i]->gpu_index);
		layer l = nets[i]->layers[j];
		distribute_weights(l, base);
	}
}

typedef struct {
	network **nets;
	int n;
	int j;
} sync_args;

void *sync_layer_thread(void *ptr)
{
	sync_args args = *(sync_args*)ptr;
	sync_layer(args.nets, args.n, args.j);
	free(ptr);
	return 0;
}

pthread_t sync_layer_in_thread(network **nets, int n, int j)
{
	pthread_t thread;
	sync_args *ptr = (sync_args *)calloc(1, sizeof(sync_args));
	ptr->nets = nets;
	ptr->n = n;
	ptr->j = j;
	if (pthread_create(&thread, 0, sync_layer_thread, ptr)) error("Thread creation failed");
	return thread;
}

void sync_nets(network **nets, int n, int interval)
{
	int j;
	int layers = nets[0]->n;
	pthread_t *threads = (pthread_t *)calloc(layers, sizeof(pthread_t));

	*(nets[0]->seen) += interval * (n - 1) * nets[0]->batch * nets[0]->subdivisions;
	for (j = 0; j < n; ++j) {
		*(nets[j]->seen) = *(nets[0]->seen);
	}
	for (j = 0; j < layers; ++j) {
		threads[j] = sync_layer_in_thread(nets, n, j);
	}
	for (j = 0; j < layers; ++j) {
		pthread_join(threads[j], 0);
	}
	free(threads);
}

float train_networks(network **nets, int n, data d, int interval)
{
	int i;
	int batch = nets[0]->batch;
	int subdivisions = nets[0]->subdivisions;
	assert(batch * subdivisions * n == d.X.rows);
	pthread_t *threads = (pthread_t *)calloc(n, sizeof(pthread_t));
	float *errors = (float *)calloc(n, sizeof(float));

	float sum = 0;
	for (i = 0; i < n; ++i) {
		data p = get_data_part(d, i, n);
		threads[i] = train_network_in_thread(nets[i], p, errors + i);
	}
	for (i = 0; i < n; ++i) {
		pthread_join(threads[i], 0);
		//printf("%f\n", errors[i]);
		sum += errors[i];
	}
	//cudaDeviceSynchronize();
	if (get_current_batch(nets[0]) % interval == 0) {
		printf("Syncing... ");
		fflush(stdout);
		sync_nets(nets, n, interval);
		printf("Done!\n");
	}
	//cudaDeviceSynchronize();
	free(threads);
	free(errors);
	return (float)sum / (n);
}

void pull_network_output(network *net)
{
	layer l = get_network_output_layer(net);
	cuda_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
}

#endif






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

metadata get_metadata(char *file)
{
	metadata m = { 0 };
	list *options = read_data_cfg(file);

	char *name_list = option_find_str(options, (char*)"names", 0);
	if (!name_list) name_list = option_find_str(options, (char*)"labels", 0);
	if (!name_list) {
		fprintf(stderr, "No names or labels found\n");
	}
	else {
		m.names = get_labels(name_list);
	}
	m.classes = option_find_int(options, (char*)"classes", 2);
	free_list(options);
	return m;
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

void option_insert(list *l, char *key, char *val)
{
	kvp *p = (kvp*)malloc(sizeof(kvp));
	p->key = key;
	p->val = val;
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

char *option_find(list *l, char *key)
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
char *option_find_str(list *l, char *key, char *def)
{
	char *v = option_find(l, key);
	if (v) return v;
	if (def) fprintf(stderr, "%s: Using default '%s'\n", key, def);
	return def;
}

int option_find_int(list *l, char *key, int def)
{
	char *v = option_find(l, key);
	if (v) return atoi(v);
	fprintf(stderr, "%s: Using default '%d'\n", key, def);
	return def;
}

int option_find_int_quiet(list *l, char *key, int def)
{
	char *v = option_find(l, key);
	if (v) return atoi(v);
	return def;
}

float option_find_float_quiet(list *l, char *key, float def)
{
	char *v = option_find(l, key);
	if (v) return atof(v);
	return def;
}

float option_find_float(list *l, char *key, float def)
{
	char *v = option_find(l, key);
	if (v) return atof(v);
	fprintf(stderr, "%s: Using default '%lf'\n", key, def);
	return def;
}







//parser.c
list *read_cfg(char *filename);

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

local_layer parse_local(list *options, size_params params)
{
	int n = option_find_int(options, (char*)"filters", 1);
	int size = option_find_int(options, (char*)"size", 1);
	int stride = option_find_int(options, (char*)"stride", 1);
	int pad = option_find_int(options, (char*)"pad", 0);
	char *activation_s = option_find_str(options, (char*)"activation", (char*)"logistic");
	ACTIVATION activation = get_activation(activation_s);

	int batch, h, w, c;
	h = params.h;
	w = params.w;
	c = params.c;
	batch = params.batch;
	if (!(h && w && c)) error("Layer before local layer must output image.");

	local_layer layer = make_local_layer(batch, h, w, c, n, size, stride, pad, activation);

	return layer;
}

layer parse_deconvolutional(list *options, size_params params)
{
	int n = option_find_int(options, (char*)"filters", 1);
	int size = option_find_int(options, (char*)"size", 1);
	int stride = option_find_int(options, (char*)"stride", 1);

	char *activation_s = option_find_str(options, (char*)"activation", (char*)"logistic");
	ACTIVATION activation = get_activation(activation_s);

	int batch, h, w, c;
	h = params.h;
	w = params.w;
	c = params.c;
	batch = params.batch;
	if (!(h && w && c)) error("Layer before deconvolutional layer must output image.");
	int batch_normalize = option_find_int_quiet(options, (char*)"batch_normalize", 0);
	int pad = option_find_int_quiet(options, (char*)"pad", 0);
	int padding = option_find_int_quiet(options, (char*)"padding", 0);
	if (pad) padding = size / 2;

	layer l = make_deconvolutional_layer(batch, h, w, c, n, size, stride, padding, activation, batch_normalize, params.net->adam);

	return l;
}

convolutional_layer parse_convolutional(list *options, size_params params)
{
	int n = option_find_int(options, (char*)"filters", 1);
	int size = option_find_int(options, (char*)"size", 1);
	int stride = option_find_int(options, (char*)"stride", 1);
	int pad = option_find_int_quiet(options, (char*)"pad", 0);
	int padding = option_find_int_quiet(options, (char*)"padding", 0);
	int groups = option_find_int_quiet(options, (char*)"groups", 1);
	if (pad) padding = size / 2;

	char *activation_s = option_find_str(options, (char*)"activation", (char*)"logistic");
	ACTIVATION activation = get_activation(activation_s);

	int batch, h, w, c;
	h = params.h;
	w = params.w;
	c = params.c;
	batch = params.batch;
	if (!(h && w && c)) error("Layer before convolutional layer must output image.");
	int batch_normalize = option_find_int_quiet(options, (char*)"batch_normalize", 0);
	int binary = option_find_int_quiet(options, (char*)"binary", 0);
	int xnor = option_find_int_quiet(options, (char*)"xnor", 0);

	convolutional_layer layer = make_convolutional_layer(batch, h, w, c, n, groups, size, stride, padding, activation, batch_normalize, binary, xnor, params.net->adam);
	layer.flipped = option_find_int_quiet(options, (char*)"flipped", 0);
	layer.dot = option_find_float_quiet(options, (char*)"dot", 0);

	return layer;
}

layer parse_crnn(list *options, size_params params)
{
	int output_filters = option_find_int(options, (char*)"output_filters", 1);
	int hidden_filters = option_find_int(options, (char*)"hidden_filters", 1);
	char *activation_s = option_find_str(options, (char*)"activation", (char*)"logistic");
	ACTIVATION activation = get_activation(activation_s);
	int batch_normalize = option_find_int_quiet(options, (char*)"batch_normalize", 0);

	layer l = make_crnn_layer(params.batch, params.w, params.h, params.c, hidden_filters, output_filters, params.time_steps, activation, batch_normalize);

	l.shortcut = option_find_int_quiet(options, (char*)"shortcut", 0);

	return l;
}

layer parse_rnn(list *options, size_params params)
{
	int output = option_find_int(options, (char*)"output", 1);
	char *activation_s = option_find_str(options, (char*)"activation", (char*)"logistic");
	ACTIVATION activation = get_activation(activation_s);
	int batch_normalize = option_find_int_quiet(options, (char*)"batch_normalize", 0);

	layer l = make_rnn_layer(params.batch, params.inputs, output, params.time_steps, activation, batch_normalize, params.net->adam);

	l.shortcut = option_find_int_quiet(options, (char*)"shortcut", 0);

	return l;
}

layer parse_gru(list *options, size_params params)
{
	int output = option_find_int(options, (char*)"output", 1);
	int batch_normalize = option_find_int_quiet(options, (char*)"batch_normalize", 0);

	layer l = make_gru_layer(params.batch, params.inputs, output, params.time_steps, batch_normalize, params.net->adam);
	l.tanh = option_find_int_quiet(options, (char*)"tanh", 0);

	return l;
}

layer parse_lstm(list *options, size_params params)
{
	int output = option_find_int(options, (char*)"output", 1);
	int batch_normalize = option_find_int_quiet(options, (char*)"batch_normalize", 0);

	layer l = make_lstm_layer(params.batch, params.inputs, output, params.time_steps, batch_normalize, params.net->adam);

	return l;
}

layer parse_connected(list *options, size_params params)
{
	int output = option_find_int(options, (char*)"output", 1);
	char *activation_s = option_find_str(options, (char*)"activation", (char*)"logistic");
	ACTIVATION activation = get_activation(activation_s);
	int batch_normalize = option_find_int_quiet(options, (char*)"batch_normalize", 0);

	layer l = make_connected_layer(params.batch, params.inputs, output, activation, batch_normalize, params.net->adam);
	return l;
}

layer parse_softmax(list *options, size_params params)
{
	int groups = option_find_int_quiet(options, (char*)"groups", 1);
	layer l = make_softmax_layer(params.batch, params.inputs, groups);
	l.temperature = option_find_float_quiet(options, (char*)"temperature", 1);
	char *tree_file = option_find_str(options, (char*)"tree", 0);
	if (tree_file) l.softmax_tree = read_tree(tree_file);
	l.w = params.w;
	l.h = params.h;
	l.c = params.c;
	l.spatial = option_find_float_quiet(options, (char*)"spatial", 0);
	l.noloss = option_find_int_quiet(options, (char*)"noloss", 0);
	return l;
}

int *parse_yolo_mask(char *a, int *num)
{
	int *mask = 0;
	if (a) {
		int len = strlen(a);
		int n = 1;
		int i;
		for (i = 0; i < len; ++i) {
			if (a[i] == ',') ++n;
		}
		mask = (int*)calloc(n, sizeof(int));
		for (i = 0; i < n; ++i) {
			int val = atoi(a);
			mask[i] = val;
			a = strchr(a, ',') + 1;
		}
		*num = n;
	}
	return mask;
}

layer parse_yolo(list *options, size_params params)
{
	int classes = option_find_int(options, (char*)"classes", 20);
	int total = option_find_int(options, (char*)"num", 1);
	int num = total;

	char *a = option_find_str(options, (char*)"mask", 0);
	int *mask = parse_yolo_mask(a, &num);
	layer l = make_yolo_layer(params.batch, params.w, params.h, num, total, mask, classes);
	assert(l.outputs == params.inputs);

	l.max_boxes = option_find_int_quiet(options, (char*)"max", 90);
	l.jitter = option_find_float(options, (char*)"jitter", .2);

	l.ignore_thresh = option_find_float(options, (char*)"ignore_thresh", .5);
	l.truth_thresh = option_find_float(options, (char*)"truth_thresh", 1);
	l.random = option_find_int_quiet(options, (char*)"random", 0);

	char *map_file = option_find_str(options, (char*)"map", 0);
	if (map_file) l.map = read_map(map_file);

	a = option_find_str(options, (char*)"anchors", 0);
	if (a) {
		int len = strlen(a);
		int n = 1;
		int i;
		for (i = 0; i < len; ++i) {
			if (a[i] == ',') ++n;
		}
		for (i = 0; i < n; ++i) {
			float bias = atof(a);
			l.biases[i] = bias;
			a = strchr(a, ',') + 1;
		}
	}
	return l;
}

layer parse_iseg(list *options, size_params params)
{
	int classes = option_find_int(options, (char*)"classes", 20);
	int ids = option_find_int(options, (char*)"ids", 32);
	layer l = make_iseg_layer(params.batch, params.w, params.h, classes, ids);
	assert(l.outputs == params.inputs);
	return l;
}

layer parse_region(list *options, size_params params)
{
	int coords = option_find_int(options, (char*)"coords", 4);
	int classes = option_find_int(options, (char*)"classes", 20);
	int num = option_find_int(options, (char*)"num", 1);

	layer l = make_region_layer(params.batch, params.w, params.h, num, classes, coords);
	assert(l.outputs == params.inputs);

	l.log = option_find_int_quiet(options, (char*)"log", 0);
	l.sqrt = option_find_int_quiet(options, (char*)"sqrt", 0);

	l.softmax = option_find_int(options, (char*)"softmax", 0);
	l.background = option_find_int_quiet(options, (char*)"background", 0);
	l.max_boxes = option_find_int_quiet(options, (char*)"max", 30);
	l.jitter = option_find_float(options, (char*)"jitter", .2);
	l.rescore = option_find_int_quiet(options, (char*)"rescore", 0);

	l.thresh = option_find_float(options, (char*)"thresh", .5);
	l.classfix = option_find_int_quiet(options, (char*)"classfix", 0);
	l.absolute = option_find_int_quiet(options, (char*)"absolute", 0);
	l.random = option_find_int_quiet(options, (char*)"random", 0);

	l.coord_scale = option_find_float(options, (char*)"coord_scale", 1);
	l.object_scale = option_find_float(options, (char*)"object_scale", 1);
	l.noobject_scale = option_find_float(options, (char*)"noobject_scale", 1);
	l.mask_scale = option_find_float(options, (char*)"mask_scale", 1);
	l.class_scale = option_find_float(options, (char*)"class_scale", 1);
	l.bias_match = option_find_int_quiet(options, (char*)"bias_match", 0);

	char *tree_file = option_find_str(options, (char*)"tree", 0);
	if (tree_file) l.softmax_tree = read_tree(tree_file);
	char *map_file = option_find_str(options, (char*)"map", 0);
	if (map_file) l.map = read_map(map_file);

	char *a = option_find_str(options, (char*)"anchors", 0);
	if (a) {
		int len = strlen(a);
		int n = 1;
		int i;
		for (i = 0; i < len; ++i) {
			if (a[i] == ',') ++n;
		}
		for (i = 0; i < n; ++i) {
			float bias = atof(a);
			l.biases[i] = bias;
			a = strchr(a, ',') + 1;
		}
	}
	return l;
}

detection_layer parse_detection(list *options, size_params params)
{
	int coords = option_find_int(options, (char*)"coords", 1);
	int classes = option_find_int(options, (char*)"classes", 1);
	int rescore = option_find_int(options, (char*)"rescore", 0);
	int num = option_find_int(options, (char*)"num", 1);
	int side = option_find_int(options, (char*)"side", 7);
	detection_layer layer = make_detection_layer(params.batch, params.inputs, num, side, classes, coords, rescore);

	layer.softmax = option_find_int(options, (char*)"softmax", 0);
	layer.sqrt = option_find_int(options, (char*)"sqrt", 0);

	layer.max_boxes = option_find_int_quiet(options, (char*)"max", 90);
	layer.coord_scale = option_find_float(options, (char*)"coord_scale", 1);
	layer.forced = option_find_int(options, (char*)"forced", 0);
	layer.object_scale = option_find_float(options, (char*)"object_scale", 1);
	layer.noobject_scale = option_find_float(options, (char*)"noobject_scale", 1);
	layer.class_scale = option_find_float(options, (char*)"class_scale", 1);
	layer.jitter = option_find_float(options, (char*)"jitter", .2);
	layer.random = option_find_int_quiet(options, (char*)"random", 0);
	layer.reorg = option_find_int_quiet(options, (char*)"reorg", 0);
	return layer;
}

cost_layer parse_cost(list *options, size_params params)
{
	char *type_s = option_find_str(options, (char*)"type", (char*)"sse");
	COST_TYPE type = get_cost_type(type_s);
	float scale = option_find_float_quiet(options, (char*)"scale", 1);
	cost_layer layer = make_cost_layer(params.batch, params.inputs, type, scale);
	layer.ratio = option_find_float_quiet(options, (char*)"ratio", 0);
	layer.noobject_scale = option_find_float_quiet(options, (char*)"noobj", 1);
	layer.thresh = option_find_float_quiet(options, (char*)"thresh", 0);
	return layer;
}

crop_layer parse_crop(list *options, size_params params)
{
	int crop_height = option_find_int(options, (char*)"crop_height", 1);
	int crop_width = option_find_int(options, (char*)"crop_width", 1);
	int flip = option_find_int(options, (char*)"flip", 0);
	float angle = option_find_float(options, (char*)"angle", 0);
	float saturation = option_find_float(options, (char*)"saturation", 1);
	float exposure = option_find_float(options, (char*)"exposure", 1);

	int batch, h, w, c;
	h = params.h;
	w = params.w;
	c = params.c;
	batch = params.batch;
	if (!(h && w && c)) error("Layer before crop layer must output image.");

	int noadjust = option_find_int_quiet(options, (char*)"noadjust", 0);

	crop_layer l = make_crop_layer(batch, h, w, c, crop_height, crop_width, flip, angle, saturation, exposure);
	l.shift = option_find_float(options, (char*)"shift", 0);
	l.noadjust = noadjust;
	return l;
}

layer parse_reorg(list *options, size_params params)
{
	int stride = option_find_int(options, (char*)"stride", 1);
	int reverse = option_find_int_quiet(options, (char*)"reverse", 0);
	int flatten = option_find_int_quiet(options, (char*)"flatten", 0);
	int extra = option_find_int_quiet(options, (char*)"extra", 0);

	int batch, h, w, c;
	h = params.h;
	w = params.w;
	c = params.c;
	batch = params.batch;
	if (!(h && w && c)) error("Layer before reorg layer must output image.");

	layer layer = make_reorg_layer(batch, w, h, c, stride, reverse, flatten, extra);
	return layer;
}

maxpool_layer parse_maxpool(list *options, size_params params)
{
	int stride = option_find_int(options, (char*)"stride", 1);
	int size = option_find_int(options, (char*)"size", stride);
	int padding = option_find_int_quiet(options, (char*)"padding", size - 1);

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

dropout_layer parse_dropout(list *options, size_params params)
{
	float probability = option_find_float(options, (char*)"probability", .5);
	dropout_layer layer = make_dropout_layer(params.batch, params.inputs, probability);
	layer.out_w = params.w;
	layer.out_h = params.h;
	layer.out_c = params.c;
	return layer;
}

layer parse_normalization(list *options, size_params params)
{
	float alpha = option_find_float(options, (char*)"alpha", .0001);
	float beta = option_find_float(options, (char*)"beta", .75);
	float kappa = option_find_float(options, (char*)"kappa", 1);
	int size = option_find_int(options, (char*)"size", 5);
	layer l = make_normalization_layer(params.batch, params.w, params.h, params.c, size, alpha, beta, kappa);
	return l;
}

layer parse_batchnorm(list *options, size_params params)
{
	layer l = make_batchnorm_layer(params.batch, params.w, params.h, params.c);
	return l;
}

layer parse_shortcut(list *options, size_params params, network *net)
{
	char *l = option_find(options, (char*)"from");
	int index = atoi(l);
	if (index < 0) index = params.index + index;

	int batch = params.batch;
	layer from = net->layers[index];

	layer s = make_shortcut_layer(batch, index, params.w, params.h, params.c, from.out_w, from.out_h, from.out_c);

	char *activation_s = option_find_str(options, (char*)"activation", (char*)"linear");
	ACTIVATION activation = get_activation(activation_s);
	s.activation = activation;
	s.alpha = option_find_float_quiet(options, (char*)"alpha", 1);
	s.beta = option_find_float_quiet(options, (char*)"beta", 1);
	return s;
}

layer parse_l2norm(list *options, size_params params)
{
	layer l = make_l2norm_layer(params.batch, params.inputs);
	l.h = l.out_h = params.h;
	l.w = l.out_w = params.w;
	l.c = l.out_c = params.c;
	return l;
}

layer parse_logistic(list *options, size_params params)
{
	layer l = make_logistic_layer(params.batch, params.inputs);
	l.h = l.out_h = params.h;
	l.w = l.out_w = params.w;
	l.c = l.out_c = params.c;
	return l;
}

layer parse_activation(list *options, size_params params)
{
	char *activation_s = option_find_str(options, (char*)"activation", (char*)"linear");
	ACTIVATION activation = get_activation(activation_s);

	layer l = make_activation_layer(params.batch, params.inputs, activation);

	l.h = l.out_h = params.h;
	l.w = l.out_w = params.w;
	l.c = l.out_c = params.c;

	return l;
}

layer parse_upsample(list *options, size_params params, network *net)
{

	int stride = option_find_int(options, (char*)"stride", 2);
	layer l = make_upsample_layer(params.batch, params.w, params.h, params.c, stride);
	l.scale = option_find_float_quiet(options, (char*)"scale", 1);
	return l;
}

route_layer parse_route(list *options, size_params params, network *net)
{
	char *l = option_find(options, (char*)"layers");
	int len = strlen(l);
	if (!l) error("Route Layer must specify input layers");
	int n = 1;
	int i;
	for (i = 0; i < len; ++i) {
		if (l[i] == ',') ++n;
	}

	int *layers = (int*)calloc(n, sizeof(int));
	int *sizes = (int*)calloc(n, sizeof(int));
	for (i = 0; i < n; ++i) {
		int index = atoi(l);
		l = strchr(l, ',') + 1;
		if (index < 0) index = params.index + index;
		layers[i] = index;
		sizes[i] = net->layers[index].outputs;
	}
	int batch = params.batch;

	route_layer layer = make_route_layer(batch, n, layers, sizes);

	convolutional_layer first = net->layers[layers[0]];
	layer.out_w = first.out_w;
	layer.out_h = first.out_h;
	layer.out_c = first.out_c;
	for (i = 1; i < n; ++i) {
		int index = layers[i];
		convolutional_layer next = net->layers[index];
		if (next.out_w == first.out_w && next.out_h == first.out_h) {
			layer.out_c += next.out_c;
		}
		else {
			layer.out_h = layer.out_w = layer.out_c = 0;
		}
	}

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
	net->batch = option_find_int(options, (char*)"batch", 1);
	net->learning_rate = option_find_float(options, (char*)"learning_rate", .001);
	net->momentum = option_find_float(options, (char*)"momentum", .9);
	net->decay = option_find_float(options, (char*)"decay", .0001);
	int subdivs = option_find_int(options, (char*)"subdivisions", 1);
	net->time_steps = option_find_int_quiet(options, (char*)"time_steps", 1);
	net->notruth = option_find_int_quiet(options, (char*)"notruth", 0);
	net->batch /= subdivs;
	net->batch *= net->time_steps;
	net->subdivisions = subdivs;
	net->random = option_find_int_quiet(options, (char*)"random", 0);

	net->adam = option_find_int_quiet(options, (char*)"adam", 0);
	if (net->adam) {
		net->B1 = option_find_float(options, (char*)"B1", .9);
		net->B2 = option_find_float(options, (char*)"B2", .999);
		net->eps = option_find_float(options, (char*)"eps", .0000001);
	}

	net->h = option_find_int_quiet(options, (char*)"height", 0);
	net->w = option_find_int_quiet(options, (char*)"width", 0);
	net->c = option_find_int_quiet(options, (char*)"channels", 0);
	net->inputs = option_find_int_quiet(options, (char*)"inputs", net->h * net->w * net->c);
	net->max_crop = option_find_int_quiet(options, (char*)"max_crop", net->w * 2);
	net->min_crop = option_find_int_quiet(options, (char*)"min_crop", net->w);
	net->max_ratio = option_find_float_quiet(options, (char*)"max_ratio", (float)net->max_crop / net->w);
	net->min_ratio = option_find_float_quiet(options, (char*)"min_ratio", (float)net->min_crop / net->w);
	net->center = option_find_int_quiet(options, (char*)"center", 0);
	net->clip = option_find_float_quiet(options, (char*)"clip", 0);

	net->angle = option_find_float_quiet(options, (char*)"angle", 0);
	net->aspect = option_find_float_quiet(options, (char*)"aspect", 1);
	net->saturation = option_find_float_quiet(options, (char*)"saturation", 1);
	net->exposure = option_find_float_quiet(options, (char*)"exposure", 1);
	net->hue = option_find_float_quiet(options, (char*)"hue", 0);

	if (!net->inputs && !(net->h && net->w && net->c)) error("No input parameters supplied");

	char *policy_s = option_find_str(options, (char*)"policy", (char*)"constant");
	net->policy = get_policy(policy_s);
	net->burn_in = option_find_int_quiet(options, (char*)"burn_in", 0);
	net->power = option_find_float_quiet(options, (char*)"power", 4);
	if (net->policy == STEP) {
		net->step = option_find_int(options, (char*)"step", 1);
		net->scale = option_find_float(options, (char*)"scale", 1);
	}
	else if (net->policy == STEPS) {
		char *l = option_find(options, (char*)"steps");
		char *p = option_find(options, (char*)"scales");
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
		net->gamma = option_find_float(options, (char*)"gamma", 1);
	}
	else if (net->policy == SIG) {
		net->gamma = option_find_float(options, (char*)"gamma", 1);
		net->step = option_find_int(options, (char*)"step", 1);
	}
	else if (net->policy == POLY || net->policy == RANDOM) {
	}
	net->max_batches = option_find_int(options, (char*)"max_batches", 0);
}

int is_network(section *s)
{
	return (strcmp(s->type, "[net]") == 0
		|| strcmp(s->type, "[network]") == 0);
}

network *parse_network_cfg(char *filename)
{
	list *sections = read_cfg(filename);
	node *n = sections->front;
	if (!n) error("Config file has no sections");
	network *net = make_network(sections->size - 1);
	net->gpu_index = gpu_index;
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
		else if (lt == DECONVOLUTIONAL) {
			l = parse_deconvolutional(options, params);
		}
		else if (lt == LOCAL) {
			l = parse_local(options, params);
		}
		else if (lt == ACTIVE) {
			l = parse_activation(options, params);
		}
		else if (lt == LOGXENT) {
			l = parse_logistic(options, params);
		}
		else if (lt == L2NORM) {
			l = parse_l2norm(options, params);
		}
		else if (lt == RNN) {
			l = parse_rnn(options, params);
		}
		else if (lt == GRU) {
			l = parse_gru(options, params);
		}
		else if (lt == LSTM) {
			l = parse_lstm(options, params);
		}
		else if (lt == CRNN) {
			l = parse_crnn(options, params);
		}
		else if (lt == CONNECTED) {
			l = parse_connected(options, params);
		}
		else if (lt == CROP) {
			l = parse_crop(options, params);
		}
		else if (lt == COST) {
			l = parse_cost(options, params);
		}
		else if (lt == REGION) {
			l = parse_region(options, params);
		}
		else if (lt == YOLO) {
			l = parse_yolo(options, params);
		}
		else if (lt == ISEG) {
			l = parse_iseg(options, params);
		}
		else if (lt == DETECTION) {
			l = parse_detection(options, params);
		}
		else if (lt == SOFTMAX) {
			l = parse_softmax(options, params);
			net->hierarchy = l.softmax_tree;
		}
		else if (lt == NORMALIZATION) {
			l = parse_normalization(options, params);
		}
		else if (lt == BATCHNORM) {
			l = parse_batchnorm(options, params);
		}
		else if (lt == MAXPOOL) {
			l = parse_maxpool(options, params);
		}
		else if (lt == REORG) {
			l = parse_reorg(options, params);
		}
		else if (lt == AVGPOOL) {
			l = parse_avgpool(options, params);
		}
		else if (lt == ROUTE) {
			l = parse_route(options, params, net);
		}
		else if (lt == UPSAMPLE) {
			l = parse_upsample(options, params, net);
		}
		else if (lt == SHORTCUT) {
			l = parse_shortcut(options, params, net);
		}
		else if (lt == DROPOUT) {
			l = parse_dropout(options, params);
			l.output = net->layers[count - 1].output;
			l.delta = net->layers[count - 1].delta;
#ifdef GPU
			l.output_gpu = net->layers[count - 1].output_gpu;
			l.delta_gpu = net->layers[count - 1].delta_gpu;
#endif
		}
		else {
			fprintf(stderr, "Type not recognized: %s\n", s->type);
		}
		l.clip = net->clip;
		l.truth = option_find_int_quiet(options, (char*)"truth", 0);
		l.onlyforward = option_find_int_quiet(options, (char*)"onlyforward", 0);
		l.stopbackward = option_find_int_quiet(options, (char*)"stopbackward", 0);
		l.dontsave = option_find_int_quiet(options, (char*)"dontsave", 0);
		l.dontload = option_find_int_quiet(options, (char*)"dontload", 0);
		l.numload = option_find_int_quiet(options, (char*)"numload", 0);
		l.dontloadscales = option_find_int_quiet(options, (char*)"dontloadscales", 0);
		l.learning_rate_scale = option_find_float_quiet(options, (char*)"learning_rate", 1);
		l.smooth = option_find_float_quiet(options, (char*)"smooth", 0);
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
#ifdef GPU
	net->output_gpu = out.output_gpu;
	net->input_gpu = cuda_make_array(net->input, net->inputs*net->batch);
	net->truth_gpu = cuda_make_array(net->truth, net->truths*net->batch);
#endif
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

void save_convolutional_weights_binary(layer l, FILE *fp)
{
#ifdef GPU
	if (gpu_index >= 0) {
		pull_convolutional_layer(l);
	}
#endif
	binarize_weights(l.weights, l.n, l.c*l.size*l.size, l.binary_weights);
	int size = l.c*l.size*l.size;
	int i, j, k;
	fwrite(l.biases, sizeof(float), l.n, fp);
	if (l.batch_normalize) {
		fwrite(l.scales, sizeof(float), l.n, fp);
		fwrite(l.rolling_mean, sizeof(float), l.n, fp);
		fwrite(l.rolling_variance, sizeof(float), l.n, fp);
	}
	for (i = 0; i < l.n; ++i) {
		float mean = l.binary_weights[i*size];
		if (mean < 0) mean = -mean;
		fwrite(&mean, sizeof(float), 1, fp);
		for (j = 0; j < size / 8; ++j) {
			int index = i * size + j * 8;
			unsigned char c = 0;
			for (k = 0; k < 8; ++k) {
				if (j * 8 + k >= size) break;
				if (l.binary_weights[index + k] > 0) c = (c | 1 << k);
			}
			fwrite(&c, sizeof(char), 1, fp);
		}
	}
}

void save_convolutional_weights(layer l, FILE *fp)
{
	if (l.binary) {
		//save_convolutional_weights_binary(l, fp);
		//return;
	}
#ifdef GPU
	if (gpu_index >= 0) {
		pull_convolutional_layer(l);
	}
#endif
	int num = l.nweights;
	fwrite(l.biases, sizeof(float), l.n, fp);
	if (l.batch_normalize) {
		fwrite(l.scales, sizeof(float), l.n, fp);
		fwrite(l.rolling_mean, sizeof(float), l.n, fp);
		fwrite(l.rolling_variance, sizeof(float), l.n, fp);
	}
	fwrite(l.weights, sizeof(float), num, fp);
}

void save_batchnorm_weights(layer l, FILE *fp)
{
#ifdef GPU
	if (gpu_index >= 0) {
		pull_batchnorm_layer(l);
	}
#endif
	fwrite(l.scales, sizeof(float), l.c, fp);
	fwrite(l.rolling_mean, sizeof(float), l.c, fp);
	fwrite(l.rolling_variance, sizeof(float), l.c, fp);
}

void save_connected_weights(layer l, FILE *fp)
{
#ifdef GPU
	if (gpu_index >= 0) {
		pull_connected_layer(l);
	}
#endif
	fwrite(l.biases, sizeof(float), l.outputs, fp);
	fwrite(l.weights, sizeof(float), l.outputs*l.inputs, fp);
	if (l.batch_normalize) {
		fwrite(l.scales, sizeof(float), l.outputs, fp);
		fwrite(l.rolling_mean, sizeof(float), l.outputs, fp);
		fwrite(l.rolling_variance, sizeof(float), l.outputs, fp);
	}
}

void save_weights_upto(network *net, char *filename, int cutoff)
{
#ifdef GPU
	if (net->gpu_index >= 0) {
		cuda_set_device(net->gpu_index);
	}
#endif
	fprintf(stderr, "Saving weights to %s\n", filename);
	FILE *fp = fopen(filename, "wb");
	if (!fp) file_error(filename);

	int major = 0;
	int minor = 2;
	int revision = 0;
	fwrite(&major, sizeof(int), 1, fp);
	fwrite(&minor, sizeof(int), 1, fp);
	fwrite(&revision, sizeof(int), 1, fp);
	fwrite(net->seen, sizeof(size_t), 1, fp);

	int i;
	for (i = 0; i < net->n && i < cutoff; ++i) {
		layer l = net->layers[i];
		if (l.dontsave) continue;
		if (l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL) {
			save_convolutional_weights(l, fp);
		} if (l.type == CONNECTED) {
			save_connected_weights(l, fp);
		} if (l.type == BATCHNORM) {
			save_batchnorm_weights(l, fp);
		} if (l.type == RNN) {
			save_connected_weights(*(l.input_layer), fp);
			save_connected_weights(*(l.self_layer), fp);
			save_connected_weights(*(l.output_layer), fp);
		} if (l.type == LSTM) {
			save_connected_weights(*(l.wi), fp);
			save_connected_weights(*(l.wf), fp);
			save_connected_weights(*(l.wo), fp);
			save_connected_weights(*(l.wg), fp);
			save_connected_weights(*(l.ui), fp);
			save_connected_weights(*(l.uf), fp);
			save_connected_weights(*(l.uo), fp);
			save_connected_weights(*(l.ug), fp);
		} if (l.type == GRU) {
			if (1) {
				save_connected_weights(*(l.wz), fp);
				save_connected_weights(*(l.wr), fp);
				save_connected_weights(*(l.wh), fp);
				save_connected_weights(*(l.uz), fp);
				save_connected_weights(*(l.ur), fp);
				save_connected_weights(*(l.uh), fp);
			}
			else {
				save_connected_weights(*(l.reset_layer), fp);
				save_connected_weights(*(l.update_layer), fp);
				save_connected_weights(*(l.state_layer), fp);
			}
		}  if (l.type == CRNN) {
			save_convolutional_weights(*(l.input_layer), fp);
			save_convolutional_weights(*(l.self_layer), fp);
			save_convolutional_weights(*(l.output_layer), fp);
		} if (l.type == LOCAL) {
#ifdef GPU
			if (gpu_index >= 0) {
				pull_local_layer(l);
			}
#endif
			int locations = l.out_w*l.out_h;
			int size = l.size*l.size*l.c*l.n*locations;
			fwrite(l.biases, sizeof(float), l.outputs, fp);
			fwrite(l.weights, sizeof(float), size, fp);
		}
	}
	fclose(fp);
}
void save_weights(network *net, char *filename)
{
	save_weights_upto(net, filename, net->n);
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

//	float* t_biases = new float [l.n];
//	t_biases = l.biases;

//	std::cout << "l.n: " << l.n << std::endl;

//	for (int i = 0; i < l.n; i++)
//	{
//		std::cout << *(t_biases + i) << " ";
//		if (i % 10 == 9)
//		{
//			std::cout << std::endl;
//		}
//
//	}
//	std::cout << std::endl;
//	std::cout << std::endl;


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
		//	std::cout << "Layer Name: " << "conv " << (i + 1) << std::endl;
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








//tree.c
void change_leaves(tree *t, char *leaf_list)
{
	list *llist = get_paths(leaf_list);
	char **leaves = (char **)list_to_array(llist);
	int n = llist->size;
	int i, j;
	int found = 0;
	for (i = 0; i < t->n; ++i) {
		t->leaf[i] = 0;
		for (j = 0; j < n; ++j) {
			if (0 == strcmp(t->name[i], leaves[j])) {
				t->leaf[i] = 1;
				++found;
				break;
			}
		}
	}
	fprintf(stderr, "Found %d leaves.\n", found);
}

float get_hierarchy_probability(float *x, tree *hier, int c, int stride)
{
	float p = 1;
	while (c >= 0) {
		p = p * x[c*stride];
		c = hier->parent[c];
	}
	return p;
}

void hierarchy_predictions(float *predictions, int n, tree *hier, int only_leaves, int stride)
{
	int j;
	for (j = 0; j < n; ++j) {
		int parent = hier->parent[j];
		if (parent >= 0) {
			predictions[j*stride] *= predictions[parent*stride];
		}
	}
	if (only_leaves) {
		for (j = 0; j < n; ++j) {
			if (!hier->leaf[j]) predictions[j*stride] = 0;
		}
	}
}

int hierarchy_top_prediction(float *predictions, tree *hier, float thresh, int stride)
{
	float p = 1;
	int group = 0;
	int i;
	while (1) {
		float max = 0;
		int max_i = 0;

		for (i = 0; i < hier->group_size[group]; ++i) {
			int index = i + hier->group_offset[group];
			float val = predictions[(i + hier->group_offset[group])*stride];
			if (val > max) {
				max_i = index;
				max = val;
			}
		}
		if (p*max > thresh) {
			p = p * max;
			group = hier->child[max_i];
			if (hier->child[max_i] < 0) return max_i;
		}
		else if (group == 0) {
			return max_i;
		}
		else {
			return hier->parent[hier->group_offset[group]];
		}
	}
	return 0;
}

tree *read_tree(char *filename)
{
	tree t = { 0 };
	FILE *fp = fopen(filename, "r");

	char *line;
	int last_parent = -1;
	int group_size = 0;
	int groups = 0;
	int n = 0;
	while ((line = fgetl(fp)) != 0) {
		char *id = (char*)calloc(256, sizeof(char));
		int parent = -1;
		sscanf(line, "%s %d", id, &parent);
		t.parent = (int*)realloc(t.parent, (n + 1) * sizeof(int));
		t.parent[n] = parent;

		t.child = (int*)realloc(t.child, (n + 1) * sizeof(int));
		t.child[n] = -1;

		t.name = (char**)realloc(t.name, (n + 1) * sizeof(char *));
		t.name[n] = id;
		if (parent != last_parent) {
			++groups;
			t.group_offset = (int*)realloc(t.group_offset, groups * sizeof(int));
			t.group_offset[groups - 1] = n - group_size;
			t.group_size = (int*)realloc(t.group_size, groups * sizeof(int));
			t.group_size[groups - 1] = group_size;
			group_size = 0;
			last_parent = parent;
		}
		t.group = (int*)realloc(t.group, (n + 1) * sizeof(int));
		t.group[n] = groups;
		if (parent >= 0) {
			t.child[parent] = groups;
		}
		++n;
		++group_size;
	}
	++groups;
	t.group_offset = (int*)realloc(t.group_offset, groups * sizeof(int));
	t.group_offset[groups - 1] = n - group_size;
	t.group_size = (int*)realloc(t.group_size, groups * sizeof(int));
	t.group_size[groups - 1] = group_size;
	t.n = n;
	t.groups = groups;
	t.leaf = (int*)calloc(n, sizeof(int));
	int i;
	for (i = 0; i < n; ++i) t.leaf[i] = 1;
	for (i = 0; i < n; ++i) if (t.parent[i] >= 0) t.leaf[t.parent[i]] = 0;

	fclose(fp);
	tree *tree_ptr = (tree*)calloc(1, sizeof(tree));
	*tree_ptr = t;
	//error(0);
	return tree_ptr;
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

void sorta_shuffle(sortable_bbox *arr, size_t n, size_t size, size_t sections)
{
	size_t i;
	for (i = 0; i < sections; ++i) {
		size_t start = n * i / sections;
		size_t end = n * (i + 1) / sections;
		size_t num = end - start;
		shuffle(arr + (start*size), num, size);
	}
}

void shuffle(sortable_bbox *arr, size_t n, size_t size)
{
	size_t i;
	void *swp = calloc(1, size);
	for (i = 0; i < n - 1; ++i) {
		size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
		memcpy(swp, arr + (j*size), size);
		memcpy(arr + (j*size), arr + (i*size), size);
		memcpy(arr + (i*size), swp, size);
	}
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

int read_int(int fd)
{
	int n = 0;
	int next = _read(fd, &n, sizeof(int));
	if (next <= 0) return -1;
	return n;
}

void write_int(int fd, int n)
{
	int next = _write(fd, &n, sizeof(int));
	if (next <= 0) error("read failed");
}

int read_all_fail(int fd, char *buffer, size_t bytes)
{
	size_t n = 0;
	while (n < bytes) {
		int next = _read(fd, buffer + n, bytes - n);
		if (next <= 0) return 1;
		n += next;
	}
	return 0;
}

int write_all_fail(int fd, char *buffer, size_t bytes)
{
	size_t n = 0;
	while (n < bytes) {
		size_t next = _write(fd, buffer + n, bytes - n);
		if (next <= 0) return 1;
		n += next;
	}
	return 0;
}

void read_all(int fd, char *buffer, size_t bytes)
{
	size_t n = 0;
	while (n < bytes) {
		int next = _read(fd, buffer + n, bytes - n);
		if (next <= 0) error("read failed");
		n += next;
	}
}

void write_all(int fd, char *buffer, size_t bytes)
{
	size_t n = 0;
	while (n < bytes) {
		size_t next = _write(fd, buffer + n, bytes - n);
		if (next <= 0) error("write failed");
		n += next;
	}
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






