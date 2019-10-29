
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "work.h"

cudaError_t Calculate(int n, int k, double *weights, int *arr);
cudaError_t Save_Array_Points_And_Weights(point_t *pointsArray, int n, double *weights, int k);
cudaError_t copy_w(double *weights, int k);
cudaError_t free_All(void);

point_t *gpu_pointsArray;
double *gpu_weights;
int *dev_arr;


__global__  void GpuCudaArr (point_t* gpu_pointsArray, double* gpu_weights, int* dev_arr, int k) {
	int idInsideBlock = threadIdx.x;
	int idBlock = blockIdx.x;
	int myLocation = NUM_OF_THERADS * idBlock + idInsideBlock;
	int i;
	double sum = 0;
	for (i = 0; i < k; i++) 
		sum += (gpu_pointsArray[myLocation].inputs[i] * gpu_weights[i]);
	if (sum >= 0) 
		dev_arr[myLocation] = 1;
	else
		dev_arr[myLocation] = -1;

}


//save the point array and Weights without free:
cudaError_t Save_Array_Points_And_Weights(point_t *pointsArray, int n, double *weights, int k) {
	gpu_pointsArray = 0;
	gpu_weights = 0;
	dev_arr = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	// Allocate GPU buffers:
	cudaStatus = cudaMalloc((void**)&gpu_pointsArray, n * sizeof(point_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&gpu_weights, k * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_arr, n * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(gpu_pointsArray, pointsArray, n * sizeof(point_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, " cudaDeviceSynchronize returned error code %d after launching Calculate_Bigger_Than_Zero!\n", cudaStatus);
		goto Error;
	}

Error:
	return cudaStatus;
}



//after tarining:
cudaError_t  copy_w(double *weights, int k) {
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(gpu_weights, weights, k * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Calculate_Bigger_Than_Zero!\n", cudaStatus);
		goto Error;
	}

Error:
	return cudaStatus;
}


cudaError_t free_All(void) {
	cudaError_t cudaStatus;
//	cudaStatus = cudaSetDevice(0);
	cudaFree(gpu_pointsArray);
	cudaFree(gpu_weights);
	cudaFree(dev_arr);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching freecuda!\n", cudaStatus);
	}
	return cudaStatus;
}

cudaError_t Calculate(int n,int k, double *weights, int *arr) {
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(gpu_weights, weights, k * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with thread.
	int numBlocks = (n/ NUM_OF_THERADS);
	GpuCudaArr <<<numBlocks, NUM_OF_THERADS >>>(gpu_pointsArray,gpu_weights, dev_arr, k);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Calculate launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Calculate_Bigger_Than_Zero!\n", cudaStatus);
		goto Error;
	}

	// Copy output from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(arr, dev_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("cudaStatus failed! %s \n", cudaGetErrorString(cudaStatus));
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	return cudaStatus;

}
