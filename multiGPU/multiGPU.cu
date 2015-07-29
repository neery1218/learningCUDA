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
	int offset = 3; 
	cudaStream_t stream1, stream2;

	 cudaStreamCreate(&stream1);
  	 cudaStreamCreate(&stream2);
 
	cudaSetDevice(0);
	cudaMalloc(&x_d_1, sizeof(double) *offset);
	cudaMemcpyAsync(x_d_1, x_h, sizeof(double)*offset, cudaMemcpyHostToDevice,stream1);

	cudaSetDevice(1); 
    	cudaMalloc(&x_d_2, sizeof(double) * (size-offset));
	cudaMemcpyAsync(x_d_2, x_h+offset, sizeof(double)*(size-offset), cudaMemcpyHostToDevice,stream2);

	cudaSetDevice(0); 
	sum<<<1,offset,0,stream1>>>(x_d_1);

	cudaSetDevice(1);
	sum<<<1,size-offset,0,stream2>>>(x_d_2); 

	plhs[0] = mxCreateDoubleMatrix(1,(mwSize)size,mxREAL);
	cudaStreamSynchronize(stream1);
	cudaMemcpyAsync(mxGetPr(plhs[0]), x_d_1, sizeof(double)*offset, cudaMemcpyDeviceToHost,stream1);

	
	cudaStreamSynchronize(stream2);
	cudaMemcpyAsync(mxGetPr(plhs[0])+offset, x_d_2, sizeof(double)*(size-offset), cudaMemcpyDeviceToHost,stream2);

	cudaStreamSynchronize(stream1);
	cudaStreamSynchronize(stream2); 
	double *outMatrix = mxGetPr(plhs[0]); 

	//free(x_h);
	cudaFree(x_d_1);
	cudaFree(x_d_2);  


    

   
}    
        
