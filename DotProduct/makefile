all: dotProduct.ptx

dotProduct.ptx: dotProduct.cu
	nvcc -arch=sm_11 -ptx dotProduct.cu

clean:
	rm -f dotProduct.ptx
