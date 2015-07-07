function [transformed,sizeOfTransformed] = matlabDriverScript(x)

%square every element in array
gpuDevice(1); 

k = parallel.gpu.CUDAKernel('square.ptx','square.cu');

%set object properties 
k.GridSize = [3 1 1];
k.ThreadBlockSize = [3 1 1];

result = feval(k,x);
transformed = gather(result); 
sizeOfTransformed = size(transformed); 


end
