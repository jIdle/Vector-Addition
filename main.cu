/*
 * Kobe Davis
 * Prof. Karavan
 * CS 405
 * 19 April 2019
 *
 * Assignment 1: Vector Addition
*/

#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

__host__   void errCatch(cudaError_t err);
__global__ void vecAddKernel(int* A, int* B, int* C, int size);

int main()
{
	int len = 1024;
	int* d_A, * d_B, * d_C;
	int* h_A = new int[len];
	int* h_B = new int[len];
	int* h_C = new int[len];

	for (int i = 0; i < len; ++i) {
		h_A[i] = h_B[i] = 1;
		h_C[i] = 0;
	}

	int size = len * sizeof(int);
	errCatch(cudaMalloc((void**)& d_A, size));
	errCatch(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
	errCatch(cudaMalloc((void**)& d_B, size));
	errCatch(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
	errCatch(cudaMalloc((void**)& d_C, size));

	vecAddKernel<<< len / 256, 256 >>>(d_A, d_B, d_C, len);
	errCatch(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

	int sum = 0;
	cout << "Sum of Resultant Vector: ";
	for (int i = 0; i < len; ++i)
		sum += h_C[i];
	cout << sum << endl;

	errCatch(cudaFree(d_A));
	errCatch(cudaFree(d_B));
	errCatch(cudaFree(d_C));
	delete[] h_A;
	delete[] h_B;
	delete[] h_C;

	return 0;
}

__global__
void vecAddKernel(int* A, int* B, int* C, int size) {
	int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	if (idx < size)
		C[idx] = A[idx] + B[idx];
}

void errCatch(cudaError_t err) {
	if (err != cudaSuccess) {
		cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
		exit(EXIT_FAILURE);
	}
}