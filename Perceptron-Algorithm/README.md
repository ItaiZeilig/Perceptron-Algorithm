# Parallel-implementation-of-Binary-Classification
Parallel implementation of Binary Classification In Course Parallel and Distributed Computation .
I used three different parallel technologies for it - MPI, OpenMP, Cuda.

## Getting Started


### Prerequisites
 
 * MPICH2 
 * make sure that your run environment supports Cuda.
 * [Download the NVIDIA CUDA Toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#download-cuda-software) 

### Problem Definition
Given a set of N points in K-dimensional space. Each point X is marked as belonging to set A or B.
Implement a Simplified Binary Classification algorithm to find a Linear Classifier. 
The result depends on the maximum iteration allowed and a value of the chosen parameter alpha. 
The purpose of the project is to define a minimal value of alpha
 that leads to the Classifier with acceptable value of Quality of Classifier.

 ![](https://cdn1.imggmi.com/uploads/2019/10/3/6e302cd2c1a45fea60730899843a6ca6-full.png)

## Sequential Implementation of Simplified Binary Classification algorithm

1.	Set alpha = alpha0
2.	Choose initial value of all components of the vector of weights W to be equal to zero.
3.	Cycle through all given points Xi in the order as it is defined in the input file
4.	For each point Xi define a sign of discriminant function f(Xi) = WT Xi. If the values of vector W is chosen properly then all points belonging to set A will have positive value of f(Xi) and all points belonging to set B will have negative value of f(Xi). The first point P that does not satisfies this criterion will cause to stop the check and immediate redefinition of the vector W:
W = W + [alpha*sign(f(P))] P
5.	 Loop through stages 3-4 till one of following satisfies:
a.	All given points are classified properly
b.	The number maximum iterations LIMIT is reached

6.	Find Nmis - the number of points that are wrongly classified, meaning that the value of f(Xi) is not correct for those points. Calculate a Quality of Classifier q according the formula q = Nmis / N

7.	Check if the Quality of Classifier is reached (q is less than a given value QC). 
8.	Stop if q < QC.
9.	Increment the value of alpha:    alpha= alpha + alpha0.    Stop if  alpha> alphaMAX
10.	Loop through stages 2-9

## Input data and Output Result of the project

*	N - number of points
*	K – number of coordinates of points
*	Coordinates of all points with attached value: 1 for those that belong to set A and -1 for the points that belong to set B.
*	alpha0 – increment value of alpha0 alphamax – maximum value of alpha
*	LIMIT – the maximum number of iterations. 
*	QC – Quality of Classifier to be reached 

## Input File format
  The first line of the file contains   N    K    alpha0   alphaMax LIMIT   QC.  
Next lines are coordinates of all points, one per line, with attached value 1 or -1.
