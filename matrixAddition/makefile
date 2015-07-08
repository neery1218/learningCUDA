all: matrixAddition.ptx

matrixAddition.ptx: matrixAddition.cu
	nvcc -arch=sm_11 -ptx matrixAddition.cu

clean:
	rm -f matrixAddition.ptx
