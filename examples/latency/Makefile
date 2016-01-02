include ../../config.mk

all:
ifeq ($(OPENCL),yes)
	PATH=$(PATH):../../bin; ../../bin/swan --opencl latency.kh latency.cu 
	$(CC) -o latency.ocl latency.c -I. -I../../include -L../../lib -lswan_ocl $(OPENCL_FLAGS)
	$(CC) -o opencl-reference opencl-reference.c $(OPENCL_FLAGS)
endif
ifeq ($(CUDA),yes)
	PATH=$(PATH):../../bin; ../../bin/swan --cuda latency.kh latency.cu 
	$(CC) -o latency.nv latency.c -I. -I../../include -L../../lib -lswan_cuda $(CUDA_FLAGS)
	nvcc   -o cuda-reference cuda-reference.cu
endif

clean:
	rm -f latency.nv latency.ocl __swan* _swan*

