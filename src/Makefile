include ../config.mk

all: ocl-tools lib-cuda lib-opencl 

ocl-tools:
	$(CC) -g -o ../bin/swan-to-asc to-asc.c
ifeq ($(OPENCL),yes)
	$(CC) -g -o ../bin/swan-oclc oclc.c $(OPENCL_FLAGS)
endif

lib-opencl:  fastfill
ifeq ($(OPENCL),yes)
ifeq ($(DSO),yes)
	$(CC) -g --shared -fPIC -o ../lib/libswan_ocl.so  swanlib_ocl.c deviceQueryOcl.c $(OPENCL_FLAGS) -I../include -lpthread
endif
	$(CC) -g -c  deviceQueryOcl.c $(OPENCL_FLAGS) -I../include -lpthread
	$(CC) -g -c  swanlib_ocl.c $(OPENCL_FLAGS) -I../include -lpthread
	ar cr ../lib/libswan_ocl.a deviceQueryOcl.o swanlib_ocl.o 
#	$(CC) -g -o ../lib/libswan_ocl.a -c  swanlib_ocl.c deviceQueryOcl.c $(OPENCL_FLAGS) -I../include -lpthread
endif

lib-cuda: fastfill header
ifeq ($(CUDA),yes)
ifeq ($(DSO),yes)
	$(CC) -g --shared -fPIC -o ../lib/libswan_cuda.so  swanlib_nv.c $(CUDA_FLAGS) -I../include -lpthread
endif
	$(CC) -g -o ../lib/libswan_cuda.a -c swanlib_nv.c  $(CUDA_FLAGS) -I../include -lpthread
endif

fastfill:
ifeq ($(CUDA),yes)
	PATH="$(PATH)":../bin; ../bin/swan --cuda fastfill_nv.kh fastfill.cu
endif
ifeq ($(OPENCL),yes)
	PATH="$(PATH)":../bin; ../bin/swan --opencl fastfill_ocl.kh fastfill.cu
endif


clean:
	rm -f fastfill*kh
	rm -f ../bin/swan-oclc ../bin/swan-to-asc
	rm -f ../lib/*.a -f ../lib/*.so


header:
	nvcc --version |grep release | awk '{print $$5}' | sed "s/,//g" | sed "s/\./ /g" | awk '{print "#define CUDA_MAJOR "$$1"\n#define CUDA_MINOR "$$2"\n"}' > cuda_version.h
