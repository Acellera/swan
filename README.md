SWAN Version 0.1 
12 Mar 2010
(c) M J Harvey, Imperial College London  <m.j.harvey@imperial.ac.uk>


INTRODUCTION
------------

Swan is a simple tool for aiding a port from CUDA to OpenCL. 
It has the following benefits:

* Reduce reliance on nvcc for CUDA code by replacing the CUDA runtime extensions
* Automatic code generation of entry point functions for kernels, to replace <<<>>> syntax
* Source translation of CUDA kernels to OpenCL.
* Common API for both CUDA and OpenCL targets.
  

BUILDING
--------

* In the top level directory (tld), edit config.mk and set the paths appropriately
  OpenCL and CUDA libraries may be selectively enabled with "OPENCL=yes|no" and
	"CUDA=yes|no"

* Set LD_LIBRARY_PATH to point to libOpenCL and other libs.
* Set PATH to point to (tld)/bin

* To build library and examples:
  $ cd (tld)
	$ make

	If all went well, you will have lib/libswan_(cuda|ocl).(so|a) 
	and CUDA and/or OpenCL versions of each example (.nv and .ocl suffices)  

* Build just the examples (having already built the library):
	$ cd (tld)
	$ make examples

EXAMPLES
--------

Each example will build in OpenCL and CUDA variants, with suffices .ocl and .nv respectively.

* examples/vecadd
  A simple vector addition

* examples/vecadd_global
	A simple vector addition showing use of __device__/__constant__ global data

* examples/shmem
	Shows use of dynamic shared memory

* examples/interp
  1D Texturing, with arrays and linear memory (CUDA only)


PORTING EXISTING CUDA CODE
--------------------------

How to use in your own code:

* Put the kernels in a separate file, e.g. kernel.cu

* Include the compiled code in the host source:
	#include "kernel.kh"

* Replace CUDA-style kernel launches with calls to the entry point
  functions.  The formal definitions of these functions are
  sytematically derived from the kernel source. For example, a kernel:

	__global__ void func1( float4 *in, int *out, int N );

  will have the following entry points:

	void k_func1( block_config_t grid, block_config_t block, int shmem, float4 *in, int *out, int N );
	
  and

	void k_func1_async( block_config_t grid, block_config_t block, int shmem, float4 *in, int *out, int N );

  grid, block and shmem have the same definitions as in the CUDA launch syntax 
  <<< grid, block, shmem >>>. block_config_t has 3 elements .x,.y,.z, like a dim3.

* Replace cuda API calls with Swan equivalents (See swan_api.h)


GENERATING CUDA / OPENCL EXECUTABLES
------------------------------------

* For cross-compiling CUDA kernels to OpenCL:
	swan --opencl kernel.kh kernel.cu

* For compiling CUDA kernels to CUDA:
	swan --cuda kernel.kh kernel.cu

* For compiling OpenCL kernels to OpenCL:
  swan --opencl-direct kernel.kh kernel.cu

  This will produce a source code header file that contains a binary object
	containing the compiled kernel source and a set of functions for calling the 
	kernels. 

* Compile 

* Link against the appropriate swan library (libswan_ocl or libswan_cuda).
