#include <cuda.h>
#include <cuda_runtime.h>
#include <mex.h>
#include <cufft.h>
#include <cuComplex.h>
#include <math.h>
__constant__ int size; 

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{              
    int size;              
	size = mxGetN(prhs[0]);

	float2* idata = (float2*)malloc(size*sizeof(float2)); 
	float2 *odata_h=(float2*)malloc(size * sizeof(float2));
      
    /* check for proper number of arguments */
    if(nrhs!=1) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","Two inputs required.");
    }
    //if(nlhs!=1) {
      // mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs","One output required.");
    //}
   

    /* create a pointer to the real data in the input matrix  */
    double  *tempIn = mxGetPr(prhs[0]);

	for (int i = 0; i < size; i++)
	{
		idata[i].x = (float)tempIn[i];
		idata[i].y = 0.0f; 
	}

    /* get dimensions of the input matrix */
    	
	
	cufftHandle plan;
	float2 *data;
	cudaMalloc((void**)&data, sizeof(float2)*size);
	cudaMemcpy(data,idata,sizeof(float2)*size,cudaMemcpyHostToDevice);
	cufftPlan1d(&plan, size, CUFFT_C2C,1);
		
	cufftExecC2C(plan, data, data, CUFFT_FORWARD);

	cudaDeviceSynchronize();
	mexPrintf("size: %d",size); 
	

	cudaMemcpy(odata_h, data, sizeof(float2)*size, cudaMemcpyDeviceToHost);

	plhs[0] = mxCreateDoubleMatrix(1,(mwSize)size,mxREAL);
	double *tempOut = mxGetPr(plhs[0]);

	for (int i = 0; i < size; i++){
		tempOut[i]=cuCabsf(odata_h[i]); 
	}
	//cudaMemcpy(mxGetPr(plhs[0]), odata_d, (sizeof(double) * size), cudaMemcpyDeviceToHost);

	cudaFree(data);
	cufftDestroy(plan); 
	


    

   
}    
        
