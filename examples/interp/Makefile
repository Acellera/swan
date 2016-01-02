include ../../config.mk

all:
ifeq ($(CUDA),yes)
	PATH=$(PATH):../../bin; ../../bin/swan --cuda interp.kh interp.cu 
	$(CC) -o interp.nv interp.c -I. -I../../include -L../../lib -lswan_cuda $(CUDA_FLAGS)
endif
ifeq ($(OPENCL),yes)
	PATH=$(PATH):../../bin; ../../bin/swan --opencl interp.kh interp.cu 
	$(CC) -o interp.ocl interp.c -I. -I../../include -L../../lib -lswan_ocl $(OPENCL_FLAGS)
#	echo ""
#	echo "Interpolation example does not work for OpenCL (yet)"
#	echo ""
endif


clean:
	rm -f interp.nv interp.ocl __swan* _swan*

