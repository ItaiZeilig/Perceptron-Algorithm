#pragma once

#include <omp.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
//including cuda:
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define OUTPUT_FILE_NAME "C:/Users/cudauser/desktop/outPut.txt"

#define MIN_SIZE_POINTS 100000
#define MAX_SIZE_POINTS 500000
#define LIMIT 1000
#define MAX_DIM 20
#define MAX_LIMIT 1000
#define NUM_OF_THERADS 1000
#define ALPHA_FOR_P_SIZE 2

struct Point
{
	double inputs[MAX_DIM];
	int answer;
	int k;
}typedef point_t;

struct OutPut 
{
	double w [MAX_DIM];
	double a,q;
	int k;
 }typedef output_t;


void printPoint(const point_t point);
void printPointsArray(int n, point_t* pointsArray);
void printOutPut(const output_t o);
void printOutPutsArray(int numOfProc, output_t* oArr);
int foundAMin(int numOfProc, output_t* oArr);
void saveToOutPutFile(output_t o);
void freeAll(int* arr, point_t* pointsArray, cudaError_t cudaStatus);


// CUDA:
cudaError_t Save_Array_Points_And_Weights(point_t *pointsArray, int n, double *weights, int k);
cudaError_t  copy_w (double *weights, int k);
cudaError_t free_All(void);
cudaError_t Calculate(int n,int k, double *weights,int *arr);