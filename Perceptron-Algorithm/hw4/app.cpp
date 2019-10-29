#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "readFromFile.h"
#include "perceptron.h"
#include "mpi.h"
#include <mpi.h>
#include <omp.h>
//#include "work.h"

void masterSendPointsAndRangeAlpha(double range,int numOfProc, double alphaMaxInit, double
	alphaInit, point_t* pointsArray, int n, MPI_Datatype* PointMPIType);
void createOutPutMPIType(MPI_Datatype* OutPutMPIType);
void createPointMPIType(MPI_Datatype* PointMPIType);


int main(int argc, char *argv[]){
	//mpi:
	MPI_Datatype PointMPIType;
	MPI_Datatype OutPutMPIType;
	MPI_Status status;
	point_t* pointsArray;
	output_t* outPutsArray;
	output_t outPut;
	int myrank, n, root = 0, numOfProc, k = 0, limit, i, nMis = 0, flag = 1;
	double alphaInit, alphaMaxInit, qc, q=0,range=0, alphasForP[ALPHA_FOR_P_SIZE];

	//CUDA:
	int* arr;
	cudaError_t cudaStatus;

	//time:
	double t1 = 0, t2 = 0;

	//mpi:
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &numOfProc);
	createPointMPIType(&PointMPIType);
	createOutPutMPIType(&OutPutMPIType);

	if (myrank == root) {
		t1 = MPI_Wtime();
		//The master read the input:
		pointsArray = readFromFile(INPUT_FILE_NAME, &n, &k, &alphaInit, &alphaMaxInit, &limit, &qc);
		if (pointsArray == NULL) {
			printf("failed .\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
			fflush(stdout);
		}
		range = ((int)(alphaMaxInit * 10) / numOfProc)*alphaInit;
		double temp = alphaMaxInit / alphaInit;
		if (temp < numOfProc) {
			printf("Please run with less then %f proccess \n", temp);
			free(pointsArray);
			MPI_Abort(MPI_COMM_WORLD, 1);
			fflush(stdout);
		}
		//each p return his result to this array (by MPI_Gather and MPI_Scatter)
		outPutsArray = (output_t*)calloc(numOfProc, sizeof(output_t));
	}

	MPI_Bcast(&k, 1, MPI_INT, root, MPI_COMM_WORLD);
	MPI_Bcast(&limit, 1, MPI_INT, root, MPI_COMM_WORLD);
	MPI_Bcast(&alphaInit, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);
	MPI_Bcast(&qc, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);

	//Master send the size of points to proccess
	if (myrank == 0) {
#pragma omp parallel for
		for (i=1 ; i< numOfProc ; i++)
			MPI_Send(&n, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
	}
	//Each process gets the size of the array and calloc it:
	else {
		MPI_Recv(&n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		pointsArray = (point_t*)calloc(n, sizeof(point_t));
	}

	//CUDA: Each process keeps the same array in cuda:
	cudaStatus = Save_Array_Points_And_Weights(pointsArray, n,outPut.w,k);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed!");
	}
	arr = (int*)calloc(n, sizeof(int));

	/*master send the alpha range + the array of points: 
	alphaForp[0] = alpha min for p
	alphaforp[1] = alpha max for p */
	if (myrank == 0) {
		masterSendPointsAndRangeAlpha(range,numOfProc,alphaMaxInit, alphaInit
			,pointsArray,n, &PointMPIType);
		alphasForP[0] = alphaInit;
		alphasForP[1] = alphaInit + range;
		}
	else {
		MPI_Recv(pointsArray, n, PointMPIType, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(alphasForP, 2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
	}

	MPI_Scatter(outPutsArray, 1, OutPutMPIType, &outPut, 1, OutPutMPIType, root, MPI_COMM_WORLD);
	outPut = calculatePerceptron(n, pointsArray, k, &alphaInit, alphasForP, &q, &qc, limit, arr);
	MPI_Gather(&outPut, 1, OutPutMPIType, outPutsArray, 1, OutPutMPIType, root, MPI_COMM_WORLD);
	freeAll(arr, pointsArray, cudaStatus);

	if (myrank == 0) {
		/*Master finds the minimun alpha and write to a file 
		If the master not find minimum alpha he will write it */
		int index = foundAMin(numOfProc, outPutsArray);
		saveToOutPutFile(outPutsArray[index]);
		t2 = MPI_Wtime();
		printf("Elapsed time is %f\n", t2 - t1);
		fflush(stdout);
		if (outPutsArray[index].a == 1) 
			printf("Alpha is not found ");
		else 
			printOutPut(outPutsArray[index]);
		free(outPutsArray);
	}

	MPI_Finalize();

}


void createPointMPIType(MPI_Datatype* PointMPIType) {
	MPI_Datatype type[3] = { MPI_DOUBLE ,MPI_INT, MPI_INT };
	int blocklen[3] = { 20, 1, 1 };
	MPI_Aint disp[3];
	point_t p;
	// Create MPI user data type for point
	disp[0] = (char *)&p.inputs - (char *)&p;
	disp[1] = (char *)&p.answer - (char *)&p;
	disp[2] = (char *)&p.k - (char *)&p;

	MPI_Type_create_struct(3, blocklen, disp, type, PointMPIType);
	MPI_Type_commit(PointMPIType);
}


void createOutPutMPIType(MPI_Datatype* OutPutMPIType) {
	MPI_Datatype type[4] = { MPI_DOUBLE ,MPI_DOUBLE, MPI_DOUBLE, MPI_INT };
	int blocklen[4] = { 20, 1, 1 ,1 };
	MPI_Aint disp[4];
	output_t o;
	// Create MPI user data type for output
#pragma omp parallel
	{
		disp[0] = (char *)&o.w - (char *)&o;
		disp[1] = (char *)&o.a - (char *)&o;
		disp[2] = (char *)&o.q - (char *)&o;
		disp[3] = (char *)&o.k - (char *)&o;
	}

	MPI_Type_create_struct(4, blocklen, disp, type, OutPutMPIType);
	MPI_Type_commit(OutPutMPIType);
}

void masterSendPointsAndRangeAlpha(double range, int numOfProc, double alphaMaxInit, double
	alphaInit, point_t* pointsArray, int n, MPI_Datatype* PointMPIType) {
	int i = 0;
	double alphasForP[ALPHA_FOR_P_SIZE];
	for (i = 1, alphasForP[0] = alphaInit, alphasForP[1] = alphaInit + range
		; i < numOfProc; i++) {
		alphasForP[0] = alphasForP[0] + range;
		alphasForP[1] = alphasForP[1] + range;
		while (alphasForP[1] > alphaMaxInit) {
			alphasForP[1] = alphasForP[1] - alphaInit;
		}
		MPI_Send(pointsArray, n, *PointMPIType, i, 0, MPI_COMM_WORLD);
		MPI_Send(alphasForP, ALPHA_FOR_P_SIZE, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
	}
}