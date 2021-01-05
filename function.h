#pragma once
#include "function.h"
#include "Matrix.h"
#include <iostream>
#include<string.h>
#include<math.h>
#include<opencv2/opencv.hpp>
#include"opencv2/core/simd_intrinsics.hpp"

using namespace cv;
using namespace std;



void ConvertToFloat(Mat& input, Matrix& output) {
	int channels = input.channels();
	int rows = input.rows;
	int cols = input.cols;
	int n = input.total() * channels;
	float* out = new float[n];

	for (size_t nrow = 0; nrow < rows; nrow++)
	{
		uchar* data = input.ptr<uchar>(nrow);
		int col = cols * channels;
		for (size_t ncol = 0; ncol < col; ncol++)
		{
			out[nrow * col + ncol] = float(data[ncol]) / 255;
		}
		// delete data;
	}

	for (int i = 0; i < n; i += 3) {
		swap(out[i], out[i + 2]);
	}
	output.rows = input.rows;
	output.cols = input.cols;
	output.channels = input.channels();
	output.data = out;
	output.count_matrix = (int*)malloc(sizeof(int));
	(*output.count_matrix) = 1;
	return;

}

float* AddPadding(Matrix& input, int padding) {
	//here we can use new variables to reduce the time we multiply
	int new_cols = input.cols + 2 * padding;
	int new_rows = input.rows + 2 * padding;
	float* ans = new float[new_rows * new_cols * input.channels];
	//  memset(ans, -1, new_rows * new_cols * input.channels * sizeof(float));

	for (int i = 0; i < padding; i++) {
		for (int j = 0; j < new_cols * input.channels; j++) {
			ans[i * new_cols * input.channels + j] = 0;
			ans[(i + input.rows + padding) * new_cols * input.channels + j] = 0;
		}
	}
	for (int i = padding; i < input.rows + padding; i++) {
		for (int j = 0; j < padding * input.channels; j++) {
			ans[i * new_cols * input.channels + j] = 0;
			ans[(i * new_cols + padding + input.cols) * input.channels + j] = 0;
		}

		for (int j = padding * input.channels; j < padding * input.channels + input.rows * input.channels; j++) {
			ans[i * new_cols * input.channels + j] = input.data[(i - padding) * input.cols * input.channels + j - padding * input.channels];
		}

	}

	return ans;
}

inline float Multiply(float* data, float* ker, int length) {
	float ans = 0;
	if (CV_SIMD512 == 1 || CV_SIMD256 == 1 || CV_SIMD128 == 1) {
		v_float32 v_sum = cv::vx_setzero_f32();
		for (int i = 0; i < length - 4; i += 4) {
			v_float32x4 va = v_load(data + i);
			v_float32x4 vb = v_load(ker + i);
			v_sum += va	* vb;
		}
		ans = v_reduce_sum(v_sum);
		for (int i = length - 4; i < length; i++) {
			ans += data[i] * ker[i];
		}
	}
	else {
		for (int i = 0; i < length; i++) {
			ans += data[i] * ker[i];
		}
	}
	return ans;
}

//inline float Multiply(float* data, float* ker, int length) {
//	float ans = 0;
//	for (int i = 0; i < length; i++) {
//		ans += data[i] * ker[i];
//	}
//	return ans;
//}


float* PickMatrix(float* data, int upleft_x, int upleft_y, int data_rows, int data_cols, int rows, int cols, int channels) {
	float* ans = new float[rows * cols * channels];
	for (int i = upleft_x; i < upleft_x + rows; i++) {
		for (int j = upleft_y * channels; j < (upleft_y + cols) * channels; j++) {
			ans[(i - upleft_x) * cols * channels + j - upleft_y * channels] = data[i * data_cols * channels + j];
		}
	}
	return ans;
}



void COV(Matrix& input, Matrix* kernel, int padding, int stride, Matrix& output, int channels, float* bias) {

	float* data = input.data;
	int data_rows = input.rows + 2 * padding;
	int data_cols = input.cols + 2 * padding;
	int data_channels = input.channels;
	int kernel_rows = kernel[0].rows;
	int kernel_cols = kernel[0].cols;
	int ans_rows = (data_rows - kernel_rows) / stride + 1;
	int ans_cols = (data_cols - kernel_cols) / stride + 1;
	float* ans = new float[ans_cols * ans_rows * channels];
	//memset(ans, -1, ans_cols * ans_rows * channels * sizeof(float));
	if (padding > 0) {
		data = AddPadding(input, padding);
	}
	for (int i = 0; i < ans_rows; i++) {
		for (int j = 0; j < ans_cols; j++) {
			float* pick = PickMatrix(data, i * stride, j * stride, data_rows, data_cols, kernel_rows, kernel_cols, data_channels);
			for (int k = 0; k < channels; k++) {
				float result = Multiply(pick, kernel[k].data, kernel_cols * kernel_rows * data_channels);
				ans[i * ans_cols * channels + j * channels + k] = result + bias[k];
			}
		}
	}
	output.channels = channels;
	output.rows = ans_rows;
	output.cols = ans_cols;
	output.data = ans;
	output.count_matrix = (int*)malloc(sizeof(int));
	(*output.count_matrix) = 1;
	return;
}



void ReLU(Matrix& input, Matrix& output) {
	output.rows = input.rows;
	output.cols = input.cols;
	output.channels = input.channels;
	int size = input.rows * input.cols * input.channels;
	if (output.data == NULL)output.data = new float[size];
	for (int i = 0; i < size; i++) {
		if (input.data[i] > 0) {
			output.data[i] = input.data[i];
		}
		else {
			output.data[i] = 0;
		}
	}
	output.count_matrix = (int*)malloc(sizeof(int));
	(*output.count_matrix) = 1;
	return;
}

float MaxPoolingOneLayer(float* data, int upleft_x, int upleft_y, int data_rows, int data_cols, int rows, int cols, int channels, int layer) {
	float ans = -1;
	int location;
	for (int i = upleft_x; i < upleft_x + rows; i++) {
		for (int j = upleft_y; j < (upleft_y + cols); j++) {
			location = i * data_cols * channels + j * channels + layer;
			if (ans < data[location])ans = data[location];
		}
	}
	return ans;
}

void MaxPooling(Matrix& input, Matrix& output, int pooling_rows, int pooling_cols, int stride) {
	float* data = input.data;
	int data_rows = input.rows;
	int data_cols = input.cols;
	int data_channels = input.channels;
	int ans_rows = (data_rows - pooling_rows) / stride + 1;
	int ans_cols = (data_cols - pooling_cols) / stride + 1;
	float* ans = new float[ans_cols * ans_rows * data_channels];
	for (int i = 0; i < ans_rows; i++) {
		for (int j = 0; j < ans_cols; j++) {
			for (int k = 0; k < data_channels; k++) {
				ans[i * ans_cols * data_channels + j * data_channels + k] = MaxPoolingOneLayer(data, i * stride, j * stride, data_rows, data_cols, pooling_rows, pooling_cols, data_channels, k);
			}
		}
	}
	output.channels = data_channels;
	output.rows = ans_rows;
	output.cols = ans_cols;
	output.data = ans;
	output.count_matrix = (int*)malloc(sizeof(int));
	(*output.count_matrix) = 1;
	return;
}

void RGBRGBToRRGGBB(float* input, float* output, int channels, int size) {
	int count = 0;
	for (int i = 0; i < channels; i++) {
		for (int j = 0; j < size; j++) {
			output[count] = input[i + j * channels];
			count++;
		}
	}
}

inline void Gemm(Matrix& input, float* output, float* weight, float* bias) {
	/*float result_1 = Multiply(input.data, weight, input.rows * input.cols * input.channels);
	float result_2 = Multiply(input.data, weight + input.rows * input.cols * input.channels, input.rows * input.cols * input.channels);*/
	float* tmp = new float[input.channels * input.rows * input.cols];
	RGBRGBToRRGGBB(input.data, tmp, input.channels, input.rows * input.cols);
	float result_1 = Multiply(tmp, weight, input.rows * input.cols * input.channels);
	float result_2 = Multiply(tmp, weight + input.rows * input.cols * input.channels, input.rows * input.cols * input.channels);
	output[0] = result_1 + bias[0];
	output[1] = result_2 + bias[1];
	float e_1 = exp(output[0]);
	float e_2 = exp(output[1]);
	output[0] = e_1 / (e_1 + e_2);
	output[1] = e_2 / (e_1 + e_2);
	return;
}