include ../../config.mk

all:
ifeq ($(OPENCL),yes)
	PATH=$(PATH):../../bin; ../../bin/swan --opencl vecadd.kh vecadd.cu 
	$(CC) -o vecadd.ocl vecadd.c -I. -I../../include -L../../lib -lswan_ocl $(OPENCL_FLAGS)
endif
ifeq ($(CUDA),yes)
	PATH=$(PATH):../../bin; ../../bin/swan --cuda vecadd.kh vecadd.cu 
	$(CC) -o vecadd.nv vecadd.c -I. -I../../include -L../../lib -lswan_cuda $(CUDA_FLAGS)
endif

clean:
	rm -f vecadd.nv vecadd.ocl __swan* _swan*

