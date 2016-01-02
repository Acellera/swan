include ../../config.mk

all:
ifeq ($(OPENCL),yes)
	PATH=$(PATH):../../bin; ../../bin/swan --opencl shmem.kh shmem.cu 
	$(CC) -o shmem.ocl shmem.c -I. -I../../include -L../../lib -lswan_ocl $(OPENCL_FLAGS)
endif
ifeq ($(CUDA),yes)
	PATH=$(PATH):../../bin; ../../bin/swan --cuda shmem.kh shmem.cu 
	$(CC) -o shmem.nv shmem.c -I. -I../../include -L../../lib -lswan_cuda $(CUDA_FLAGS)
endif

clean:
	rm -f shmem.nv shmem.ocl __swan* _swan*

