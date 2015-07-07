__global__ void cu_square(int* a)
{
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
        a[idx]*=a[idx];
}
