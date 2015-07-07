function [c] = dotProduct(a,b)

%square every element in array
gpuDevice(1); 

k = parallel.gpu.CUDAKernel('dotProduct.ptx','dotProduct.cu');

%set object properties 
k.GridSize = [1 1 1];
k.ThreadBlockSize = [3 1 1];

dotProduct = 0; 

setConstantMemory(k,'VECTOR_SIZE',int32(k.ThreadBlockSize(1))); 

[~,~,result] = feval(k,a,b,dotProduct);
c = gather(result); 


end
