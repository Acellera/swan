include ../../config.mk

all:
ifeq ($(OPENCL),yes)
	PATH=$(PATH):../../bin; ../../bin/swan --opencl gridblock.kh gridblock.cu 
	$(CC) -o gridblock.ocl gridblock.c -I. -I../../include -L../../lib -lswan_ocl $(OPENCL_FLAGS)
endif
ifeq ($(CUDA),yes)
	PATH=$(PATH):../../bin; ../../bin/swan --cuda gridblock.kh gridblock.cu 
	$(CC) -o gridblock.nv gridblock.c -I. -I../../include -L../../lib -lswan_cuda $(CUDA_FLAGS)
endif

clean:
	rm -f gridblock.nv gridblock.ocl __swan* _swan*

