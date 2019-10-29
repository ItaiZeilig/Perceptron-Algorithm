#pragma once
#include "work.h"
void initWeights(int k, double* weights);
double f(double* w, point_t point);
int sign(double num);
void train(int k, point_t point, double* w, double alpha, int error);
output_t calculatePerceptron(int n, point_t* pointsArray, int k, double* alphaInit
	, double* alphasForP, double* q, double* qc, int limit, int* arr);