#ifndef __SWAN_H 
#define __SWAN_H 1

#include <stdio.h>

#ifdef _WINDOWS
#define __thread __declspec(thread)
#endif


#ifdef OPENCL
#ifndef SWAN_CONST_T
struct __swan_const_t {
	char *name;
	size_t size;
	void *ptr;
};
#define SWAN_CONST_T
#endif
#endif

#ifdef OPENCL

	#include "swan_ocl.h"
// suppress the laoding of the cuda headers when cufft.h is included
	#define __DRIVER_TYPES_H__
//	#define __VECTOR_TYPES_H__

#else
	#include "vector_types.h"
	#include "cuda.h"
	#include "cuda_runtime.h"
	#include "cuda_runtime_api.h"
#endif

#undef CUT_CHECK_ERROR

#ifdef __cplusplus 
extern "C" {
#endif



#ifdef __cplusplus 
}
#endif

#include "swan_nv_types.h"

#include "swan_api.h"

#endif

