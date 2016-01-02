// Swan (c) 2010 M J Harvey  m.j.harvey@ic.ac.uk
// License: GPL v 2

#include <assert.h>
#include <sys/time.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "fastfill_ocl.kh"

#include "swan.h"
#include "swan_ocl.h"
#include "swan_types.h"

//#include "swan/swan_ocl.h"
//#include "swan/swan_nv_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include "swan_api.h"


#define LOG_IN double tt = -compat_gettime();
#define LOG_OUT(A) if(state.flog!=NULL) fprintf(state.flog, "[%s] [%f]\n", A, tt+compat_gettime() );


static double compat_gettime( void ) {
  struct timeval now;
  gettimeofday(&now, NULL);

 return now.tv_sec + (now.tv_usec / 1000000.0);
}


void *swanMalloc( size_t len ) ;
void swanMemcpyHtoD( void *ptr_h, void *ptr_dp, size_t len ) ;
void *swanMallocImage2D( void *ptr, int width, size_t typesize ) ;

cl_context_properties * stupid_icd_thing(void) ;

struct swanlib_state_t {
  int init;
  cl_device_id*    devices;
	int num_devices;
	int device;
	cl_command_queue cq;
  cl_context  context;
  cl_program *programs;
  char      **program_names;
  cl_kernel  *kernels;
  char      **kernel_names;
  int         num_programs;
  int         num_kernels;
	int					max_local_group_size;

	int target_device; 
	int actual_device;

	int target_device_number;
	int actual_device_number;
	void * dummy_mem;
	void * dummy_image;
	FILE *flog;
	int checkpointing;
	int num_compute_elements;
	char *device_name;
	int cpu ;
};

static __thread struct swanlib_state_t state = {
	0, NULL, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL, 0, 0, 0, 0, 0, 0 ,0, 0, 0
};


#define CHECK_ERROR_RET  if(status != CL_SUCCESS) {  \
  fprintf( stderr, "SWAN OCL: FATAL : Error at %s:%d : %d\n", __FILE__, __LINE__, status );\
assert(0);\
	exit(-40);\
}


#define DEBLOCK(swan_type,type) \
      case swan_type: \
        {\
				status = clSetKernelArg( k, argc, sizeof(type), ptr ); CHECK_ERROR_RET;\
				}\
      break;

#define DEBLOCK_PTR(swan_type) \
      case swan_type: \
        {\
				if(   ptr == NULL  || *((int*)ptr)==0  ) {\
					ptr = &( state.dummy_mem); \
				}\
				status = clSetKernelArg( k, argc, sizeof(void*), ptr ); CHECK_ERROR_RET;\
				}\
      break;

#define DEBLOCK_IMAGE(swan_type) \
      case swan_type: \
        {\
				if(   ptr == NULL  || *((int*)ptr)==0  ) {\
					ptr = &( state.dummy_image); \
				}\
				status = clSetKernelArg( k, argc, sizeof(void*), ptr ); CHECK_ERROR_RET;\
				}\
      break;




static inline void error( char *e ) {
  fprintf( stderr, "SWAN OCL: FATAL : Error at %s:%d : %s\n", __FILE__, __LINE__, e );\
  exit(-40);
}


void  swanInit( void ) {

  if( !state.init ) {

		// Force image support in ATI SDK 2.01
		setenv( "GPU_IMAGES_SUPPORT", "1", 1 );

    // try initialising
        if(getenv("CUDA_PROFILE") != 0 || getenv("SWAN_PROFILE") != 0 ) {
                state.flog = fopen("swan_profile.log", "w" );
        }
        else {
                state.flog = NULL;
        }

    cl_int status = 0;
    size_t deviceListSize;

	cl_context_properties *cprop = stupid_icd_thing();

//state.target_device = SWAN_TARGET_GPU;

//		if(getenv( "SWAN_CPU_TARGET" ) != NULL ) {
//		switch( state.target_device ) {
//		case SWAN_TARGET_GPU:	
		if( getenv( "SWAN_CPU_TARGET" ) != NULL ) {
      printf("# SWAN OCL :  SWAN_CPU_TARGET set. Forcing CPU target\n");
      state.context = clCreateContextFromType(cprop, CL_DEVICE_TYPE_CPU, NULL, NULL, &status);
			state.cpu = 1;
		}
		else {
	 	  state.context = clCreateContextFromType(cprop, CL_DEVICE_TYPE_GPU, NULL, NULL, &status);

    	if( status != CL_SUCCESS ) {
      	printf("# SWAN OCL :  No GPU available. Trying to fall back to CPU device\n");
	      state.context = clCreateContextFromType(cprop, CL_DEVICE_TYPE_CPU, NULL, NULL, &status);
				state.cpu = 1;
  	  }
		}
//			printf("SWAN :: GPU Target\n" );
//			break;
//		case SWAN_TARGET_GPU:	
//  	  state.context = clCreateContextFromType(cprop, CL_DEVICE_TYPE_GPU, NULL, NULL, &status);
//			printf("SWAN :: CPU Target\n" );
//			break;
//		default:
//			error("Unknown device type" );
//		}


    CHECK_ERROR_RET;

		state.actual_device = state.target_device;

    status = clGetContextInfo( state.context, CL_CONTEXT_DEVICES, 0, NULL, &deviceListSize);

    CHECK_ERROR_RET;

    if( deviceListSize == 0 ) {
      error( "No devices found" );
    }

    state.num_devices = deviceListSize;
    state.devices = (cl_device_id *)malloc(deviceListSize);

  /* Now, get the device list data */
    status = clGetContextInfo( state.context, CL_CONTEXT_DEVICES, deviceListSize, state.devices, NULL);

    CHECK_ERROR_RET;

	 	if( state.target_device_number >= state.num_devices ) {
			error( "Invalid device number specified" );
		}

		state.actual_device_number = state.target_device_number;

		printf("OPENCL :: Using device %d of %d\n", state.actual_device_number, state.num_devices );

   	state.cq = clCreateCommandQueue( state.context, state.devices[ state.actual_device_number ], 0, &status);

    CHECK_ERROR_RET;

    state.kernels      = NULL;
    state.kernel_names = NULL;
    state.num_kernels  = 0;

    state.programs      = NULL;
    state.program_names = NULL;
    state.num_programs  = 0;

    state.init = 1;

//		printf("SWAN :: OpenCL initialised\n");

		state.max_local_group_size=256;

		state.dummy_mem = swanMalloc(1);
		void *ptr = malloc( sizeof(float4) );

if ( ! state.cpu ) {
		state.dummy_image = swanMallocImage2D( ptr, 1, sizeof(float4) );
}

		clGetDeviceInfo(  state.devices[ state.actual_device_number ], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(int), &(state.num_compute_elements), NULL );

		size_t len;
		clGetDeviceInfo( state.devices[ state.actual_device_number ], CL_DEVICE_NAME, 0, NULL, &len );
		state.device_name = (char*) malloc( len + 1 );
		memset( state.device_name, 0, len+1 );
		clGetDeviceInfo( state.devices[ state.actual_device_number ], CL_DEVICE_NAME, len, (state.device_name), NULL );
  }

	
//  printf("Selected device [%s] \n", openclParameter( CL_DEVICE_NAME ) );
  return;
}



int swanGetDeviceCount( void ) {
	swanInit();
	return state.num_devices;
}

void swanLoadProgramFromSource( const char *module, const unsigned char *ptx, size_t len, int target ) {
	state.target_device = target;
	swanInit();

//fprintf(stderr, "swanLoadProgram [%s]\n", module );

//	if( state.target_device != state.actual_device ) {
//		error("Cannot mix CPU and GPU target code\n" );
//	}

//printf("SWAN :: swanLoadProgramFromSource for module [%s]\n", module );

	int i;
	cl_uint num_kernels;
  cl_int status;
  cl_int s2;
	cl_program program;

//	if( state.actual_device == SWAN_TARGET_GPU ) {
//		printf("Loading kernel from [%s]\n", ptx );
//	}
#ifdef OPENCL_FROM_SOURCE
	printf("WARNING: This version of SWAN is building from source \n%s\n", module);
  program = clCreateProgramWithSource( state.context, 1, (const char**) &ptx, &len, &status );
#else
  program = clCreateProgramWithBinary( state.context, 1, state.devices, &len, (const unsigned char**) &ptx, &s2, &status );
#endif
  CHECK_ERROR_RET;
//fprintf(stderr, "\tBuildProgram in\n" );
char *args=getenv("SWAN_COMPILER_ARGS");
printf("Compiling with args [%s]\n", args );
  status = clBuildProgram(program, 1, state.devices, args , NULL, NULL);
//fprintf(stderr, "\tBuildProgram out\n" );

//	if( status != CL_SUCCESS ) {
		// get the error and print it
		clGetProgramBuildInfo( program, state.devices[state.actual_device_number ], CL_PROGRAM_BUILD_LOG, 0, NULL, &len );
if( len >0) {
		char *ptr = (char*) malloc(len+1);
		clGetProgramBuildInfo( program, state.devices[state.actual_device_number ] ,CL_PROGRAM_BUILD_LOG, len, ptr, NULL );
		ptr[len]='\0';
		printf("LOG :\n %s\n", ptr );
		free(ptr);
}
//	}

//fprintf(stderr, "\tCreateKernels in\n" );
  CHECK_ERROR_RET;
  status = clCreateKernelsInProgram( program, 0, NULL, &num_kernels );
  CHECK_ERROR_RET;
//fprintf(stderr, "\tCreateKernels out\n" );
//  cl_kernel * kernels =  (cl_kernel*) malloc(  sizeof(cl_kernel) * ( num_kernels ));

//	printf(" There are %d kernels\n", num_kernels );

	state.programs = (cl_program*) realloc( state.programs, (state.num_programs + 1) * sizeof(cl_program*) );
	state.programs[ state.num_programs ] = program;
	state.num_programs++;

	state.kernels = (cl_kernel*) realloc( state.kernels, (state.num_kernels + num_kernels) * sizeof(cl_kernel*) );
	state.kernel_names = (char**) realloc( state.kernel_names, (state.num_kernels + num_kernels) * sizeof(char**) );

  status = clCreateKernelsInProgram(program, num_kernels, state.kernels + state.num_kernels  , NULL );
  CHECK_ERROR_RET;

// FIXME TODO not storing kernels by program, so name conflicts between programs will be a problem...


for( i=0; i < num_kernels; i++ ) {
    size_t len;
    cl_kernel k = state.kernels[ state.num_kernels + i ];

    clGetKernelInfo( k, CL_KERNEL_FUNCTION_NAME, 0, NULL, &len );
    char *buf = (char*) malloc( len+1 );
    clGetKernelInfo( k, CL_KERNEL_FUNCTION_NAME, len, buf, NULL );
    buf[len]='\0';
//		printf("KERNEL [%s]\n", buf );
		state.kernel_names[ state.num_kernels + i ] = (char*) malloc( len );
		strcpy( state.kernel_names[ state.num_kernels + i ], buf );
}
	state.num_kernels += num_kernels;

//fprintf(stderr, "swanLoadProgram done\n" );

}



void swanSynchronize() {

	swanInit();
	clFinish( state.cq );



}



static void __swanRunKernel(  const char *a,  block_config_t grid , block_config_t block, size_t shmem_bytes ,  int flags, void *ptrs[], int *types ) {
	swanInit();
	cl_int status;
	int i;
	//
//	printf("SWAN:: swanRunKernel %s \n", a);

//	if( block.x > state.max_local_group_size ) {
//printf("SWAN: swanRunKernel: Rounding the local group size down to %d\n", state.max_local_group_size );
//		block.x = state.max_local_group_size;
		
//	}

	// First try to find the kernel program
	cl_kernel k = NULL;
	for( i=0; i < state.num_kernels; i++ ) {
		if( !strcmp( state.kernel_names[i], a ) ) {
			k = state.kernels[i];
			break;
		}
	}
	if( k== NULL ) {
		error("Can't find kernel");
	}

	int nargs = 0;
  clGetKernelInfo( k, CL_KERNEL_NUM_ARGS, sizeof(cl_uint) , &nargs, NULL );

	// k now contains the kernel. Queue up the arguments and run it.

	int argc = 0;
	size_t gt[3], bt[3];
	void *ptr;
	int type;


	type = types[argc];


	while( type != SWAN_END ) {
		void *ptr = ptrs[argc];
		switch( type ) {
			DEBLOCK_PTR( SWAN_PTR );
			DEBLOCK_IMAGE( SWAN_IMAGE );
			DEBLOCK( SWAN_int, int   );
			DEBLOCK( SWAN_int4,   int4 );
			DEBLOCK( SWAN_int2,   int2 );
			DEBLOCK( SWAN_float2, float2 );
			DEBLOCK( SWAN_float4, float4 );
			DEBLOCK( SWAN_uint4,  uint4 );
			DEBLOCK( SWAN_uint,  uint );
			DEBLOCK( SWAN_uint2,  uint2 );
			DEBLOCK( SWAN_float,  float );
			default:
				printf("Argument type %d\n", type );
				error("Unknown type");
			break;
		}
		argc++;
		type = types[argc];
	}


	// now add the dummy argument for shmem

	if( flags == 1 ) {
		if( shmem_bytes == 0 ) { shmem_bytes=1; }	// Fix for ATI

		status = clSetKernelArg( k, argc, shmem_bytes, NULL );
		CHECK_ERROR_RET;

		argc++;
	}

	if( argc != nargs ) {
		printf("%d!=%d\n", argc, nargs );
		error("Not the right # of arguments");
	}

	bt[0] = block.x;
	bt[1] = block.y;
	bt[2] = block.z;
	gt[0] = bt[0] * grid.x;
	gt[1] = bt[1] * grid.y;
	gt[2] = bt[2] * grid.z;

//	cl_event event;

//	printf( "Arguments enqueued\n" );
//	fflush(stdout);
//	fflush(stderr);

        double tt = 0.;
        if(state.flog) {
                tt -= compat_gettime();
        }
  status = clEnqueueNDRangeKernel ( state.cq, k, 3, 0, gt, bt, 0, NULL,  NULL ) ; //&event );

	if( status != CL_SUCCESS ) {
		printf("Failure launching kernel [%s]\n", a );
		printf("Launch dimensions Global (%d,%d,%d) : grid (%d,%d,%d) block (%d,%d,%d)\n", gt[0], gt[1], gt[2], grid.x, grid.y, grid.z, block.x, block.y, block.z );
	}
	CHECK_ERROR_RET;

//	status = clWaitForEvents( 1, &event );

	//CHECK_ERROR_RET;

//	clReleaseEvent( event );

        if( state.flog ) {
                tt += compat_gettime();
                fprintf(state.flog, "[%s] [%f]\n", a, tt );
								fflush( state.flog );
        }

}


void swanRunKernel (  const char *a,  block_config_t grid , block_config_t block, size_t shmem_bytes ,  int flags, void *ptrs[], int *types ) {

#ifdef USE_CANARY
	{
	block_config_t gridc, blockc;
	gridc.x = gridc.y = gridc.z = blockc.x = blockc.y = blockc.z = 1;

  __swan_try_init();

	int param0 = 1;

 int types[2];
  void *ptrs[2];
  ptrs[0]  = (void*)&(param0);
  types[0] = SWAN_int;
  ptrs[1]  = NULL;
  types[1] = SWAN_END;
  __swanRunKernel ( "canary", gridc, blockc, 0, 1, ptrs, types);
	}

#endif

	__swanRunKernel( a, grid, block, shmem_bytes, flags, ptrs, types );

#ifdef USE_CANARY
//	k_canary( gridc, blockc, 0, 1 );
	{
	block_config_t gridc, blockc;
	gridc.x = gridc.y = gridc.z = blockc.x = blockc.y = blockc.z = 1;

  __swan_try_init();


	int param0 = 1;
 int types[2];
  void *ptrs[2];
  ptrs[0]  = (void*)&(param0);
  types[0] = SWAN_int;
  ptrs[1]  = NULL;
  types[1] = SWAN_END;

  __swanRunKernel ( "canary", gridc, blockc, 0, 1, ptrs, types);
	}


#endif
}

void swanRunKernelAsync(  const char *a,  block_config_t grid , block_config_t block, size_t shmem_bytes , int flags, void *ptrs[], int *types ) {
	swanRunKernel( a, grid, block, shmem_bytes, flags, ptrs, types );
}

void *swanMalloc( size_t len ) {
	swanInit();
  cl_int status;

	if( len == 0 ) {  len = 1; } // the runtime barfs on a zero length allocation, and we can't just use a NULL, either

  cl_mem out = clCreateBuffer( state.context, CL_MEM_READ_WRITE, len, NULL, &status);

	if( status != CL_SUCCESS ) {
		printf("SWAN:: Failed to allocate a R/W buffer of %lu bytes\n", len );
	}
  CHECK_ERROR_RET;

	// MJH likes his memory tidy
	swanMemset( (void*) out, 0, len );
	return (void*) out;
}

void *swanMallocReadOnly( size_t len, void *ptr ) {
	int i;
	swanInit();
  cl_int status;

	if( len == 0 ) {  len = 1; } // the runtime barfs on a zero length allocation, and we can't just use a NULL, either

//for(i=0; i < len; i++ ) {
//	printf("%x ",((char*) ptr)[i] );
//}
//printf("\n\n");

  cl_mem out = clCreateBuffer( state.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, len, ptr, &status);

	if( status != CL_SUCCESS ) {
		printf("SWAN:: Failed to allocate a R/O buffer of %lu bytes\n", len );
	}
  CHECK_ERROR_RET;

	return (void*) out;
}


void *swanMallocImage2D( void *ptr, int width, size_t typesize ) {

	void *ptr2;
	int i;
	swanInit();
	cl_int status;

	cl_image_format f;
	f.image_channel_data_type = CL_FLOAT;
	f.image_channel_order = CL_RGBA;

	
	ptr2 = (void*) malloc( width * sizeof(float4) );

	switch( typesize/sizeof(float) ) {
	case 4: 
		memcpy( ptr2, ptr, sizeof(float4) * width );
	break;
		case 2: 
			for(i=0; i < width; i++ ) {
				uint4* pd = (uint4*) ptr2;
				uint2 * ps = (uint2*) ptr;
				pd[i].x = ps[i].x; 
				pd[i].y = ps[i].y; 
				pd[i].z = 0;
				pd[i].w = 0;
			}
	break;
		case 1: 
		ptr2 = (void*) malloc( width * sizeof(float4) );
		for(i=0; i < width; i++ ) {
				uint4* pd = (uint4*) ptr2;
				uint * ps = (uint*) ptr;
				pd[i].x = ps[i]; 
				pd[i].y = 0;
				pd[i].z = 0;
				pd[i].w = 0;
			}
	break;
	default:
		printf("SWAN:: swanMallocImage2D :: Invalid typesize\n");
		exit(-1);
	}

//	printf("Allocation is %d bytes\n", width * typesize );
//printf("Width =%d\n", width);
//for(int i=0; i < width; i++ ) {
//	float4 *pp = ((float4*)ptr2) + i;
//	printf("X %d %f\t%f\t%f\t%f\n", i, pp->x, pp->y, pp->z, pp->w );
//}

	cl_mem out = clCreateImage2D( state.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &f, width, 1, NULL, ptr2, &status );

	if( status != CL_SUCCESS ) {
		printf("SWAN:: Failed to allocate an Image2D (%dx1) %lu bytes : error = %d\n",width,  width* typesize, status );
		exit(-40);
	}

	free(ptr2);

	return (void*) out;
}




void *swanMallocPitch( size_t *pitch_in_bytes, size_t width_in_bytes, size_t height ) {
	swanInit();
//	printf("TODO:: %s %d\n", __FILE__, __LINE__ );

	size_t len = width_in_bytes;
	int PITCH = 256;
	if( len % PITCH ) {
		len = (1+(len / PITCH )) * PITCH; // round up to multiple of PITCH
	}
	void *ptr = swanMalloc( height * len );
	*pitch_in_bytes = len;
	return ptr;
}

void swanMemset( void *ptr, unsigned char byte, size_t len ) {
	swanInit();

	if( !len ) {return; }


  if( (0 == len % 16) && byte==0 ) {
      block_config_t grid, block;
      int threads;
        threads = 256;
    
      swanDecompose( &grid, &block, len/(sizeof(uint4)), threads );
    

        k_swan_fast_fill( grid, block, 0, (uint4*) ptr, len/(sizeof(uint4) )  );
  }
	else if( (0 == len % 4 ) && byte==0 ) {
      block_config_t grid, block;
      int threads;
        threads = 256;
    
      swanDecompose( &grid, &block, len/(sizeof(uint)), threads );
    

        k_swan_fast_fill_word( grid, block, 0, (uint*) ptr, len/(sizeof(uint) )  );
	}
	else {

	LOG_IN
printf("SLOW FILL %d\n", len );
//	printf("TODO:: %s %d\n", __FILE__, __LINE__ );
		unsigned char *pp = (unsigned char*) malloc( len );
		memset( pp, byte, len );
		swanMemcpyHtoD( pp, ptr, len );
		free(pp);
        LOG_OUT( "swanMemset(slow)" )
	}

}

void *swanMallocHost( size_t len ) {
	swanInit();
	printf("TODO:: swanMallocHost\n" );
	return (malloc(len));
}
void swanFree( void *ptrd ) {
	swanInit();
	cl_mem ptr = (cl_mem) ptrd;
	clReleaseMemObject( ptr );
}
void swanFreeHost( void *ptr ) {
	swanInit();
	printf("TODO:: swanFreeHost\n" );
	free(ptr);
}

void swanMemcpyHtoD( void *ptr_h, void *ptr_dp, size_t len ) {
	if(!len) return;

	swanInit();
	LOG_IN
  cl_uint status;
  cl_event event;
	cl_mem ptr_d = (cl_mem) ptr_dp;
  status = clEnqueueWriteBuffer ( state.cq, ptr_d , 1, 0, len, ptr_h, 0, NULL, &event );

 if(status != CL_SUCCESS) { assert(0); }
  CHECK_ERROR_RET;
  status = clWaitForEvents(1, &event);
  CHECK_ERROR_RET;
        LOG_OUT( "swanMemcpyHtoD" )

}

void swanMemcpyDtoH( void *ptr_dp, void *ptr_h, size_t len ) {
	if(!len) return;
	swanInit();
	LOG_IN
  cl_uint status;
  cl_event event;
	cl_mem ptr_d = (cl_mem) ptr_dp;

  status = clEnqueueReadBuffer ( state.cq, ptr_d , 1, 0, len, ptr_h, 0, NULL, &event );
  CHECK_ERROR_RET;
  status = clWaitForEvents(1, &event);
  CHECK_ERROR_RET;
        LOG_OUT( "swanMemcpyDtoH" )

}

void swanMemcpyDtoD( void *ptr_dsrc, void *ptr_ddest, size_t len ) {
	if(!len) return;
	swanInit();
	LOG_IN
//	printf("TODO:: %s %d\n", __FILE__, __LINE__ );
//	error("swanMEmcpyDtoD" );
//	printf(" swanMemcpyDtoD slow\n" );
	void *tmp = malloc( len );
	swanMemcpyDtoH( ptr_dsrc, tmp, len );
	swanMemcpyHtoD( tmp, ptr_ddest, len );
	free(tmp);
        LOG_OUT( "swanMemcpyHtoD"  )

}



void swanBindToTexture1DEx( const char *modname, const char *texname, int width, void *ptr, size_t typesize, int flags ) {
	swanInit();
	printf("TODO::  swanBindToTexture1DEx : [%s][%s], %s %d\n",  modname, texname, __FILE__, __LINE__ );
//	error("swanBindToTexture1DEx" );
}
void swanMakeTexture2DEx( const char *modname, const char *texname, int width, int height, void *ptr, size_t typesize, int flags ) {
	swanInit();
	printf("TODO:: swanMakeTexture2DEx [%s][%s]  %s %d\n", modname, texname, __FILE__, __LINE__ );
//	error("swanBindToTexture2DEx" );
}

void swanMakeTexture1DEx( const char *modname, const char *texname, int width,  void *ptr, size_t typesize, int flags ) {
	swanInit();
	printf("TODO:: swanMakeTexture1DEx [%s][%s] %s %d\n", modname, texname, __FILE__, __LINE__ );
	//error("swanMakeTexture1Dex" );
}



void swanDecompose( block_config_t *grid, block_config_t *block, int thread_count, int threads_per_block ) 
{
  grid->x = grid->y = grid->z  = 1;
  block->x= block->y= block->z = 1;

#if 1
	if( threads_per_block > state.max_local_group_size ) {
		printf("SWAN: swanRunKernel: Rounding the local group size down to %d\n", state.max_local_group_size );
		threads_per_block = state.max_local_group_size;
	}
#endif

  block->x = threads_per_block;
  grid->x  = thread_count / threads_per_block;

  if( (grid->x * threads_per_block) < thread_count ) { grid->x ++; }
}


void swanBindToConstantEx( struct __swan_const_t *consts, const char *constname, size_t len , void *ptr ) {
	struct __swan_const_t *t = consts;
	while(t->name !=NULL ) {
		if(!strcmp( t->name, constname ) ) {
			if( len > t->size ) {
				error("swanBindToConstantEx: lengths do not match" );
			}
			else if (len < t->size ) {
				fprintf(stderr, "SWAN: swanBindToConstantEx: length is short [%s] %d vs %d\n ", constname, len, t->size );
			}
		
			if( t->ptr != NULL ) {	
				swanFree( t->ptr );
			}
			t->ptr = swanMallocReadOnly( len, ptr );
			
	
//			swanMemcpyHtoD( ptr, t->ptr, len );
			return;
		}
		t++;
	}
	error("swanBindToConstantEx: constant not found");
}

void swanSetDeviceNumber( int devtype ) {
	if( state.init ) { 
		error( "sanSetTargetDevice(): OCL is already initialised" );
	}
	state.target_device_number = devtype;
}


void swanSetTargetDevice( int devtype ) {
//	if( devtype != SWAN_TARGET_GPU && devtype != SWAN_TARGET_GPU ) {
//		error("swanSetTargetDevice(): Device must be either SWAN_TARGET_GPU or SWAN_TARGET_GPU" );
//	}
//	if( state.init ) { 
//		error( "sanSetTargetDevice(): OCL is already initialised" );
//	}
//	state.target_device = devtype;
}


cl_context_properties * stupid_icd_thing(void) {


    /*
 *  *      * Have a look at the available platforms and pick either
 *   *           * the AMD one if available or a reasonable default.
 *    *                */




    cl_uint numPlatforms;
    cl_platform_id platform = NULL;
    cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);
     CHECK_ERROR_RET;

    if (0 < numPlatforms)
    {
	unsigned int i;
        cl_platform_id* platforms =  (cl_platform_id*) malloc( sizeof(cl_platform_id) * numPlatforms );
        status = clGetPlatformIDs(numPlatforms, platforms, NULL);
                CHECK_ERROR_RET;

        for (i = 0; i < numPlatforms; ++i)
        {
            char pbuf[100];
            status = clGetPlatformInfo(platforms[i],
                                       CL_PLATFORM_VENDOR,
                                       sizeof(pbuf),
                                       pbuf,
                                       NULL);


                        CHECK_ERROR_RET;

            platform = platforms[i];
            if (!strcmp(pbuf, "Advanced Micro Devices, Inc."))
            {
                break;
            }
        }
        free( platforms );
    }


    /*
 *  *      * If we could find our platform, use it. Otherwise pass a NULL and get wh
 *  atever the
 *   *           * implementation thinks we should be using.
 *
 *
 */


    cl_context_properties cps[3] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform,
        0
    };
    /* Use NULL for backward compatibility */

    /* Use NULL for backward compatibility */
        if( platform == NULL ) { return NULL; }

        cl_context_properties *ptr = (cl_context_properties*) malloc( sizeof(cl_context_properties ) * 3 );
	int i;
        for(i=0; i<3;i++ ) {
                ptr[i] = cps[i];
        }
        return ptr;



}
int swanGetNumberOfComputeElements( void ) {
	swanInit();	
	return state.num_compute_elements;
}
//char  swanDeviceName( void ) {
//	swanInit();	
//	return state.device_name;
//}




int swanDeviceVersion( void ) {
	return SWAN_DEVICE_CUDA_13;
}

int swanMaxThreadCount( void ) { 
	return 256; // The most ATI can support
}

void swanFinalize(void) {
	int i;
	if( state.init ) {
		fprintf( stderr, "SWAN: swanfinalize():\n" );
		cl_int status ;

		for(i=0; i < state.num_kernels; i++ ){	
			status = clReleaseKernel( state.kernels[i] );
		}
		for(i=0; i < state.num_programs; i++ ){	
			status = clReleaseProgram( state.programs[i] );
		}


		status =clReleaseCommandQueue( state.cq );
		CHECK_ERROR_RET;
		status =clReleaseContext( state.context );
		CHECK_ERROR_RET;
	}
}

size_t swanMemAvailable( void ) {
	fprintf(stderr, "SWAN: swanMemAvailable() TODO\n" );
	return 2147483648lu;
}

#ifdef DEV_EXTENSIONS

#ifndef _WINDOWS

#include "threading.c"

#endif

#include "checkpoint.c"
#endif


#ifdef __cplusplus
}
#endif

const char *swanGetVersion(void) { return "$Rev: 1505 $"; }
