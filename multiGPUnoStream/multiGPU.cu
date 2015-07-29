#include <cuda.h>
#include <cuda_runtime.h>
#include <mex.h>
#include <cufft.h>
#include <cuComplex.h>
#include <math.h>
__constant__ int size; 
__global__ void sum (double *array){
	array[blockDim.x * blockIdx.x + threadIdx.x]*=2; 
}
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{              
    int size;              
	size = mxGetN(prhs[0]);

	double *x_h = mxGetPr(prhs[0]);
	double *x_d_1;
	double *x_d_2;  
      
    /* check for proper number of arguments */
    	if(nrhs!=1) {
        	mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","Two inputs required.");
    	}
	int num_devices;
	cudaGetDeviceCount(&num_devices);
	mexPrintf("num devices: %d",num_devices); 
	int offset = 4; 

	cudaSetDevice(0);
	cudaMalloc(&x_d_1, sizeof(double) *offset);
	cudaMemcpy(x_d_1, x_h, sizeof(double)*offset, cudaMemcpyHostToDevice);

	cudaSetDevice(1); 
    	cudaMalloc(&x_d_2, sizeof(double) * (size-offset));
	cudaMemcpy(x_d_2, x_h+offset, sizeof(double)*(size-offset), cudaMemcpyHostToDevice);

	cudaSetDevice(0); 
	sum<<<1,offset>>>(x_d_1);

	cudaSetDevice(1);
	sum<<<1,size-offset>>>(x_d_2); 

	plhs[0] = mxCreateDoubleMatrix(1,(mwSize)size,mxREAL);
	cudaMemcpy(mxGetPr(plhs[0]), x_d_1, sizeof(double)*offset, cudaMemcpyDeviceToHost);

	cudaMemcpy(mxGetPr(plhs[0])+offset, x_d_2, sizeof(double)*(size-offset), cudaMemcpyDeviceToHost);

	//free(x_h);
	cudaFree(x_d_1);
	cudaFree(x_d_2);  


    

   
}    
        
