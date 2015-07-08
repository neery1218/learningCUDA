function [c,t_matrixAddition] = matrixAddition(a,b)

%square every element in array
gpuDevice(1); 

k = parallel.gpu.CUDAKernel('matrixAddition.ptx','matrixAddition.cu');

%set object properties 

% automatic grid and numThread resizing

k.GridSize = [1 1 1];
k.ThreadBlockSize = [size(a,1) size(a,2) 1];

summed=zeros(size(a,1),size(a,2)); 

setConstantMemory(k,'num_row',int32(size(a,1))); 
setConstantMemory(k,'num_col',int32(size(a,2))); 
tic;
[~,~,result] = feval(k,a,b,summed);
c = gather(result); 
t_matrixAddition = toc; 


end
