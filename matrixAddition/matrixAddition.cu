//make sure numbers above  match the matlab script

__constant__ int num_row;
__constant__ int num_col; 

__global__ void matrix_addition (int* a, int* b, int* c)//each block calculates a row
{
	

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	//c[x * num_col + y] = a[x * num_col + y] + b[x * num_col + y];
	c[y * num_row + x] = a[y * num_row + x] + b[y * num_row + x];

	
	
}
