#include "perceptron.h"


void initWeights(int k, double* weights) {
	int i;
	//init the vector to zero 
#pragma omp parallel for
	for (i = 0; i < k; i++)
		weights[i] = 0;
}

int sign(double num) {
	if (num >= 0.0) 
		return 1;
	else 
		return -1;
}


double f(double* w, point_t point) {
	int i;
	double sum = 0;
	for (i = 0; i < point.k; i++) {
		sum += (point.inputs[i] * w[i]);
	}
	//return (double)sign(sum);
	return sum;
}


void train(int k, point_t point, double* w, double alpha, int error) {
	int i;
//#pragma omp parallel for
	for (i = 0; i < k; i++)
		w[i] += alpha*(double)error*point.inputs[i];
}

output_t calculatePerceptron(int n, point_t* pointsArray, int k,
	double* alphaInit, double* alphasForP, double* q, double* qc, int limit, int* arr)
{

	output_t o;
	int i, j, nMis = 0;
	double error;
	*q = *qc + 1;
	o.k = k;
	cudaError_t cudaStatus;
	double fAterSign = 0.0;

	while (*q > *qc && alphasForP[0] < alphasForP[1]) {
		initWeights(k, o.w);
		for (j = 0; j < limit; j++) {
			for (i = 0; i < n; i++) {
				fAterSign = sign(f(o.w, pointsArray[i]));
				error = pointsArray[i].answer - fAterSign;
				if (error != 0) {
					train(k, pointsArray[i], o.w, alphasForP[0], sign(error));
					//train(k, pointsArray[i], o.w, alphasForP[0], fAterSign);
					break;
				}
			}
		}
		//CUDA: find the nMiss:
		cudaStatus = Calculate(n, o.k, o.w, arr);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "failed!");
		}
#pragma omp parallel for reduction(+: nMis)
		for (i = 0; i < n; i++) {
			if (pointsArray[i].answer != arr[i])
				nMis += 1;
		}
		*q = (double)nMis / (double)n;
		alphasForP[0] += *alphaInit;
	}
	alphasForP[0] = alphasForP[0] - *alphaInit;
	o.a = alphasForP[0];
	o.q = *q;
	if (o.q > *qc) {
		o.a=1; 
	}
	return o;
}
