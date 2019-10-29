
#define _CRT_SECURE_NO_WARNINGS
#include "readFromFile.h"

point_t* readFromFile(char* fileName, int* n, int* k, double* alpha, double* alphaMax, int* limit, double* qc) {
	point_t* pointsArray;
	point_t onePoint;
	FILE* f = fopen(fileName, "r");
	*n = 0;
	int i, j = 0;
	// check if open file succeeded..
	if (f == NULL) {
		printf("Failed opening the file. Exiting!\n");
		return NULL;
	}
	else { //check the input
		fscanf(f, "%d", n);

		//check the size n:
		if (*n > MAX_SIZE_POINTS || *n < MIN_SIZE_POINTS) {
			printf("Please input number of points between %d - %d .\n", MIN_SIZE_POINTS, MAX_SIZE_POINTS);
			return NULL;
		}
		//check the size k:
		fscanf(f, "%d", k);
		if (*k > MAX_DIM) {
			printf("Please input dim no more then %d .\n", MAX_DIM);
			return NULL;
		}
		*k = *k + 1; //bias
		onePoint.k = *k;
		fscanf(f, "%lf", alpha);
		fscanf(f, "%lf", alphaMax);
		if (int(*alphaMax / *alpha) > 100) {
			printf("Alpha can't be more then 100 iterations .\n");
			return NULL;
		}
		//check the limit:
		fscanf(f, "%d", limit);
		if (*limit > MAX_LIMIT) {
			printf("Please input limit no more then %d .\n", MAX_LIMIT);
			return NULL;
		}
		fscanf(f, "%lf", qc);
		// allocating the points array
		pointsArray = (point_t*)calloc(*n, sizeof(point_t));

		printf("n: %d , dim:%d, a:%f, amax:%f, limit:%d, qc:%f \n", *n,
			*k, *alpha, *alphaMax, *limit, *qc);
		fflush(stdout);

		if (pointsArray == NULL) {
			printf("Failed allocating memory! \n");
			return NULL;
		}

		//read all points deatils:
		for (i = 0; i < *n; i++) {
			// allocating the inputs array
			for (j = 0; j < (*k) - 1; j++) {
				fscanf(f, "%lf", &onePoint.inputs[j]);
			}
			onePoint.inputs[*k - 1] = 1; //bias
			fscanf(f, "%d", &onePoint.answer);
			pointsArray[i] = onePoint;
		}

		fclose(f);
	}
	return pointsArray;

}