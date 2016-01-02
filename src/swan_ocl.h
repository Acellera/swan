#ifndef __SWAN_OCL_H
#define __SWAN_OCL_H 1

#include "CL/cl.h"


//#define __align__(n)  __attribute__((aligned(n)))
#define __global__
#define __device__
#define __host__

// quiet an unsatisfied dependency in cufft
typedef int cudaStream_t;


#endif
