#ifndef SWANLIB_NV_H
#define SWANLIB_NV_H 1

#include "driver_functions.h"
#include "driver_types.h"




struct swanlib_state_t {
	int init;
	CUdevice    device;
	CUcontext   context;
	CUmodule   *mods;
	char      **mod_names;
	CUfunction *funcs;
	CUstream stream;
	char      **func_names;
	int         num_mods;
	int         num_funcs;

	int target_device;
	unsigned long bytes_allocated;
	int device_version;
	int multiProcessorCount;
	int checkpointing;
	int debug;
};


#  define CU_SAFE_CALL_NO_SYNC( call ) do {                                 \
    CUresult err = call;                                                    \
    if( CUDA_SUCCESS != err) {                                              \
        fprintf(stderr, "SWAN : FATAL : Cuda driver error %x in file '%s' in line %i.\n",  \
                err, __FILE__, __LINE__ );                                  \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)


#endif
