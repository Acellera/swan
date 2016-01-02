include ../../config.mk

all:
ifeq ($(CUDA),yes)
	PATH=$(PATH):../../bin; ../../bin/swan --cuda reduce.kh reduce.cu 
	$(CC) -O3 -ffast-math -o reduce.nv reduce.c -I. -I../../include -L../../lib -lswan_cuda $(CUDA_FLAGS)
endif
ifeq ($(OPENCL),yes)
	PATH=$(PATH):../../bin; ../../bin/swan --opencl reduce.kh reduce.cu 
	$(CC) -o reduce.ocl reduce.c -I. -I../../include -L../../lib -lswan_ocl $(OPENCL_FLAGS)
endif


clean:
	rm -f reduce.nv reduce.ocl __swan* _swan*

