#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cufft.h>
#include <cuComplex.h>


static const int WORK_SIZE = 10;

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

int main(void) {
	int *d = NULL;
	int i;
	float2 idata[WORK_SIZE];
	float2 odata[WORK_SIZE];

	for (i = 0; i < WORK_SIZE; i++){
		idata[i].x = i;
		idata[i].y = 0;
	}

		cufftHandle plan;
		cufftComplex *data;
		cudaMalloc((void**)&data, sizeof(float2)*WORK_SIZE);
		cudaMemcpy(data,idata,sizeof(float2)*WORK_SIZE,cudaMemcpyHostToDevice);
		cufftPlan1d(&plan, WORK_SIZE, CUFFT_C2C,1);
		cufftExecC2C(plan, data, data, CUFFT_FORWARD);

		cudaDeviceSynchronize();



	CUDA_CHECK_RETURN(cudaMemcpy(odata, data, sizeof(float2)*WORK_SIZE, cudaMemcpyDeviceToHost));

	for (i = 0; i < WORK_SIZE; i++)
		printf("Input value: %f, device output: %f\n", idata[i].x, cuCabsf(odata[i]));

	CUDA_CHECK_RETURN(cudaFree((int*) d));
	CUDA_CHECK_RETURN(cudaDeviceReset());
	cudaFree(data);
	cufftDestroy(plan);

	return 0;
}
