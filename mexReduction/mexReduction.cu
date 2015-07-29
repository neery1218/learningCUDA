#include <cuda.h>
#include <cuda_runtime.h>
#include <mex.h>
#include <cufft.h>
#include <cuComplex.h>
#include <math.h>
__constant__ int size; 
__global__ void sum (double *array){
	extern __shared__ double shared_data[];

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	shared_data[tid] = array[tid];

	__syncthreads(); 

	for (unsigned int s=blockDim.x/2; s>0; s/=2) {
		if (tid < s) {
			shared_data[tid] += shared_data[tid + s];
		}
		__syncthreads();
	}
	array[tid] = shared_data[tid]; 
	//array[tid] = 7; 
}
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{              
    int size;              
	size = mxGetN(prhs[0]);

	double *x_h = mxGetPr(prhs[0]);
	double *x_d; 
      
    /* check for proper number of arguments */
    	if(nrhs!=1) {
        	mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","Two inputs required.");
    	}

    	cudaMalloc(&x_d, sizeof(double) * size);
	cudaMemcpy(x_d, x_h, sizeof(double)*size, cudaMemcpyHostToDevice);

	sum<<<1,8,sizeof(double)*8>>>(x_d);

	cudaDeviceSynchronize(); 
	plhs[0] = mxCreateDoubleMatrix(1,(mwSize)size,mxREAL);

	cudaMemcpy(mxGetPr(plhs[0]), x_d, sizeof(double)*size, cudaMemcpyDeviceToHost);

	//free(x_h);
	cudaFree(x_d); 


    

   
}    
        
