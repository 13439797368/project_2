#pragma once
#include "Matrix.h"
#include <iostream>
#include<iomanip>
using namespace std;

struct Matrix{
	int cols;
	int rows;
	int channels;
	int* count_matrix = NULL;
	float* data = NULL;

	Matrix clone(Matrix& m)
	{
		Matrix temp(m.rows, m.cols,m.channels,NULL);
		cout << "adress: " << &temp << " is a clone of adress" << &m << endl;
		for (int i = 0; i < m.rows * m.cols*m.channels; i++) {
			temp.data[i] = m.data[i];
		}
		return temp;
	}

	void operator=(const Matrix& m)
	{
		//cout << "operation = is been used" << endl;
		this->rows = m.rows;
		this->cols = m.cols;
		this->channels = m.channels;
		this->data = m.data;
		this->count_matrix = m.count_matrix;
		(*this->count_matrix)++;
		return;
	}


	Matrix(int r, int c, int cha, float* d) {
		cols = c;
		rows = r;
		channels = cha;
		data = new float[r * c * cha];
		for (int i = 0; i < r * c * cha; i++) {
			data[i] = d[i];
		}
		this->count_matrix = (int*)malloc(sizeof(int));
		(*this->count_matrix) = 1;
	}
	Matrix(){}
	~Matrix() {
		//cout << "Matrix " << this << " with data "<<this->data<< " is distroyed  ";
		if (data != NULL) {
			if (*count_matrix == 1) {
				//cout << "  data is distroyed" << endl;
				delete[] data;
				delete count_matrix;
			}
			else if (*count_matrix > 1) {
				//cout << "  data is not distroyed" << endl;
				(*count_matrix)--;
			}
		}
	}

	void Print(){

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols*channels; j++) {
				cout << setiosflags(ios::left) << setw(9) << data[i * cols*channels + j] << " ";
			}
			cout << endl;
		}
		return;
	}
};
