include ../../config.mk

all:
ifeq ($(OPENCL),yes)
	$(CC) -g -o info.ocl info.c -I. -I../../include -L../../lib -lswan_ocl $(OPENCL_FLAGS)
endif
ifeq ($(CUDA),yes)
	$(CC) -g -o info.nv info.c -I. -I../../include -L../../lib -lswan_cuda $(CUDA_FLAGS)
endif

clean:
	rm -f info.nv info.ocl __swan* _swan*

