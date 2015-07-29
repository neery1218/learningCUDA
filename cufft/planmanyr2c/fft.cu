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
	float *d = NULL;
	int i;
	float idata[WORK_SIZE];
	float2 *odata_d=NULL;
	float2 *odata_h=(float2*)malloc((WORK_SIZE/2 +1) * sizeof(float2));

	for (i = 0; i < WORK_SIZE; i++){
		idata[i] = i;
	}
		cufftReal a;
		cufftHandle plan;
		float *data;
		cudaMalloc((void**)&data, sizeof(float)*WORK_SIZE);
		cudaMalloc((void**)&odata_d, sizeof(float2)*(WORK_SIZE/2+1));
		cudaMemcpy(data,idata,sizeof(float)*WORK_SIZE,cudaMemcpyHostToDevice);
		cufftPlan1d(&plan, WORK_SIZE, CUFFT_R2C,1);
		//cufftPlanMany(cufftHandle *plan, int rank, int *n, int *inembed, int istride, int idist, int *onembed, int ostride, int odist, cufftType type, int batch);

		int rank = 1;
		int n[1];
		int inembed[1];
		int istride=1;
		int idist=5;
		int onembed[1];
		int ostride = 1;
		int odist = 5;
		int batch =2;

		n[0]=WORK_SIZE/2;

		inembed[0]=WORK_SIZE;

		onembed[0]=WORK_SIZE/2 + 1;


		//cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batch);

		cufftExecR2C(plan, data, odata_d);

		cudaDeviceSynchronize();



	CUDA_CHECK_RETURN(cudaMemcpy(odata_h, odata_d, sizeof(float2)*(WORK_SIZE/2+1), cudaMemcpyDeviceToHost));

	for (i = 0; i < WORK_SIZE/2 + 1; i++)
		printf("%f\n",cuCabsf(odata_h[i]));

	CUDA_CHECK_RETURN(cudaFree((int*) d));
	CUDA_CHECK_RETURN(cudaDeviceReset());
	cudaFree(data);
	cudaFree(odata_d);
	free(odata_h);
	free(idata);
	cufftDestroy(plan);

	return 0;
}

