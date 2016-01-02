include ../../config.mk

all:
ifeq ($(OPENCL),yes)
	$(CC) -o copytest.ocl copytest.c -I. -I../../include -L../../lib -lswan_ocl $(OPENCL_FLAGS)
endif
ifeq ($(CUDA),yes)
	$(CC) -o copytest.nv copytest.c -I. -I../../include -L../../lib -lswan_cuda $(CUDA_FLAGS)
endif

clean:
	rm -f copytest.nv copytest.ocl __swan* _swan*

