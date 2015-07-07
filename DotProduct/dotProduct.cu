#define THREADS_PER_BLOCK 3
#define TOTAL_BLOCKS 1

__global__ void dot_product (int* a, int*b, int*c)
{
     __shared__ int multiplicationStorage [THREADS_PER_BLOCK]; 

     multiplicationStorage[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];

	__syncthreads(); 

     if (threadIdx.x == 0){
	//compute sum
	int tempSum = 0; 
	for (int i = 0; i < THREADS_PER_BLOCK; i++){
		tempSum+=multiplicationStorage[i];
	}
	*c = tempSum; 
	//atomicAdd(c,tempSum); 
     }
	
}
