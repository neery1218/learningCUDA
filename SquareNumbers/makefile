all: square.ptx

square.ptx: square.cu
	nvcc -ptx square.cu

clean:
	rm -f square.ptx
