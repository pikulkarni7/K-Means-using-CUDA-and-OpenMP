CC = g++
NV = nvcc

default: XRay_omp

XRay_omp: XRay_omp.c
	${CC} -fopenmp -o $@ XRay_omp.c

XRay_cuda: $(SRC)
	${NV} -o $@ XRay_cuda.cu

clean:
	-rm -f XRay_omp
	-rm -f XRay_cuda