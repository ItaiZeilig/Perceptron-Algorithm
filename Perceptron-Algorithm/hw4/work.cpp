
#define _CRT_SECURE_NO_WARNINGS
#include "work.h"

void printPoint(const point_t point) {
	int i;
	for (i = 0; i < point.k - 1; i++)
		printf("%f ", (point.inputs[i]));
	printf("%d ", point.answer);
	printf("\n");
	fflush(stdout);
}

void printPointsArray(int n, point_t* pointsArray) {
	int j;
	for (j = 0; j < n; j++)
		printPoint(pointsArray[j]);
	fflush(stdout);

}

void printOutPut(const output_t o) {
	int i;
	for (i = 0; i < o.k; i++) {
		printf("w[%d] = %f \n", i, o.w[i]);
	}
	printf("q finael = %f \n", o.q);
	fflush(stdout);
	printf("alfa finael= %f \n",o.a);
	fflush(stdout);
}
void printOutPutsArray(int numOfProc, output_t* oArr) {
	int j;
	for (j = 0; j < numOfProc; j++)
		printOutPut(oArr[j]);
	fflush(stdout);
}

int foundAMin(int numOfProc, output_t* oArr) {
	int index=0,i;
	double aMin = 0.0;
#pragma omp parallel for
	for (i = 0; i < numOfProc; i++) {
		if (oArr[i].a < aMin) {
			aMin = oArr[i].a;
			index = i;
		}
	}
	return index;
}

void saveToOutPutFile( output_t o) {
	int i;
	FILE* f = fopen(OUTPUT_FILE_NAME, "w");
	if (f == NULL)
	{
		printf("Failed opening the file. Exiting!\n");
		return;
	}
	if (o.a == 1) {
		fputs("Alpha is not found ", f);
	}
	else {
		fputs("Alpha minimum = ", f);
		fprintf(f, "%lf,", o.a);
		fputs(" q = ", f);
		fprintf(f, "%lf \n", o.q);
		for (i = 0; i < o.k; i++)
			fprintf(f, "%lf \n", o.w[i]);;
	}

	fclose(f);
}

void freeAll(int* arr, point_t* pointsArray, cudaError_t cudaStatus) {
	free(pointsArray);
	//CUDA:
	free(arr);
	cudaStatus = free_All();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed free cuda!");
	}
}

