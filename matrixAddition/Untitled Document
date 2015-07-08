#define THREADS_PER_BLOCK 3
#define TOTAL_BLOCKS 1
//make sure numbers above  match the matlab script

__device__ int blockSums [TOTAL_BLOCKS]; 
__constant__ int VECTOR_SIZE; 

__global__ void dot_product (int* a, int*b, int*c)
{
     __shared__ int multiplicationStorage [THREADS_PER_BLOCK]; 
	
     if (threadIdx.x < VECTOR_SIZE)	
     multiplicationStorage[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];

	__syncthreads(); 

     if (threadIdx.x == 0){
	//compute sum
	int tempSum = 0; 
	for (int i = 0; i < VECTOR_SIZE; i++){
		tempSum+=multiplicationStorage[i];
	}
	blockSums[blockIdx.x]=tempSum; 
	__syncthreads(); 

	if (blockIdx.x==0)
	   for (int i = 0; i < TOTAL_BLOCKS; i++)	
		*c+=blockSums[i]; 
	
	//atomicAdd(c,tempSum); 
     }
	
}
