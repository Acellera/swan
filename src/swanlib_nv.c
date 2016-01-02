// Swan (c) 2010 M J Harvey  m.j.harvey@ic.ac.uk
// License: GPL v 2

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <assert.h>
#ifndef _WINDOWS
#include <sys/time.h>
#include <unistd.h>
#include <pthread.h>
#endif
#include "swan_api.h"

#include "cuda_version.h"

#ifndef CUDA_MAJOR
	#error "CUDA_MAJOR / CUDA_MINOR not set. Check cuda_version.h was generated."
#endif

#ifdef _WINDOWS
typedef unsigned int uint;
#endif

//#define MALLOC_DEBUG
#include "cuda.h"

#include "swanlib_nv.h"
#include "swan.h"

#define __SWAN_NO_LOAD_TYPES
#include "fastfill_nv.kh"
//#include "cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ALIGN_UP(offset, alignment) { (offset) = ((offset) + (alignment) - 1) & ~ ((alignment)-1);}

#if (CUDA_MAJOR==3 && CUDA_MINOR >=2 ) || CUDA_MAJOR >=4

	#define PTR_TO_CUDEVPTR( a ) ((CUdeviceptr*) a )
#else
//	#define PTR_TO_CUDEVPTR( a ) ( * ((CUdeviceptr*)(void*)(&(a))))
	#define PTR_TO_CUDEVPTR( a )  ((CUdeviceptr)((size_t)a)) 
#endif
//((CUdeviceptr)((size_t)a)) 

#define  CU_SAFE_CALL (a) {\
	CUresult err = a;\
	if( a != CUDA_ERROR ) { error( "swanBindToTexture" ); }\
}


#define DEBLOCK(swan_type,type,OFFSET) \
			case swan_type: \
				{ALIGN_UP( offset, (OFFSET));\
				cuParamSetv( f, offset, ptr, sizeof(type) );\
/*printf("ARGS: %d %x %d\n", swan_type, ptr, offset );*/\
				offset += sizeof(type); }\
			break;

/*
  int init;
  CUdevice    device;
  CUcontext   context;
  CUmodule   *mods;
  char      **mod_names;
  CUfunction *funcs;
  char      **func_names;
  int         num_mods;
  int         num_funcs;

  int target_device;
*/

static __thread struct swanlib_state_t state ={ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

static  void try_init( void );
static  void error( char * );

void swanInit(void) { 
	try_init(); 
}

void swanSetDeviceNumber( int n ) {
	if( state.init ) {
		error("Already initialised" );
	}
	state.target_device = n;
}

void swanLoadProgramFromSource(  const char *module, const unsigned  char *ptx, size_t len , int devtype ) {
	int i=0;
	CUresult err;
	try_init();
	// let's see whether this module is already loaded
	for( i=0; i < state.num_mods; i++ ) {
		if( !strcmp( state.mod_names[i], module ) ) {
			return; // already loaded
		}
	} 	

	if( ptx == NULL || len == 0 ) {
		fprintf ( stderr, "SWAN : Module load failure [%s]. No source \n", module );
		error( "Module source invalid" );
	}

	i = state.num_mods;
	state.num_mods++;
	state.mods         = (CUmodule*) realloc( state.mods, state.num_mods * sizeof(CUmodule) );
	state.mod_names    = (char**) realloc( state.mod_names, state.num_mods * sizeof(char*) );
	state.mod_names[i] = (char*) malloc( strlen( module ) + 1 );
	strcpy( state.mod_names[i], module );

	// now load the PTX into a module
	err = cuModuleLoadData( &state.mods[i], ptx );
	if( err != CUDA_SUCCESS ) {
		fprintf ( stderr, "SWAN : Module load result [%s] [%d]\n", module, err );
		error( "Module load failed\n" );
	}
	
}

void swanRunKernel( const char *kernel,  block_config_t grid , block_config_t block, size_t shmem, int flags, void *ptrs[], int *types  ) {
	CUresult  err;
	swanRunKernelAsync( kernel, grid, block, shmem, flags, ptrs, types );

 if(state.debug) {
	err =cuCtxSynchronize();
 }

//	if( err != CUDA_SUCCESS ) {
//		fprintf( stderr , "SWAN : FATAL : Failure executing kernel sync [%s] [%d]\n", kernel, err );
//	assert(0);
//		exit(-99);
//	}

}


#ifdef DEV_EXTENSIONS
double swanTime( void ) {
	struct timeval tv;
	gettimeofday( &tv, NULL );
	return ( tv.tv_sec  + tv.tv_usec/1000000.0 );
}
#endif


void swanRunKernelAsync( const char *kernel,  block_config_t grid , block_config_t block, size_t shmem, int flags, void *ptrs[], int *types  ) {
	// find the kernel

	if( !grid.x || !grid.y || !grid.z || !block.x || !block.y || !block.z ) { return; } // suppress launch of kernel if any of the launch dims are 0

	CUfunction f = NULL;
	int i;
	int offset = 0;
	CUresult err;

	int type;
	int idx=0;
	try_init();
	for( i=0; i < state.num_funcs; i++ ) {
		if( !strcmp( state.func_names[i], kernel ) ) {
			f = state.funcs[i];
			break;
		}
	}

	if( f == NULL ) {
		for( i=0; i < state.num_mods; i++ ) {
			cuModuleGetFunction( &f, state.mods[i], kernel );
			if( f!= NULL ) { 
				// found a kernel. store it for future use
				int j = state.num_funcs;
				state.num_funcs++;
				state.funcs      = (CUfunction*) realloc( state.funcs, sizeof(CUfunction) * state.num_funcs );
				state.funcs[j]   = f;
				state.func_names = (char**)      realloc( state.func_names, sizeof(char*) * state.num_funcs );
				state.func_names[j] = (char*) malloc( strlen(kernel) + 1 );
				strcpy( state.func_names[j], kernel );
				break; 
			}
		}
	}

	if( f== NULL ) {
		fprintf(stderr, "Error running kernel [%s] : \n", kernel );
		error( "No kernel found" );
	}

	if( grid.z != 1 ) {
		printf("Kernel [%s] launched with (%d %d %d)(%d %d %d)\n", kernel, grid.x, grid.y, grid.z, block.x, block.y, block.z );
		error( "grid.z needs to be 1" );
	}

//printf("Running kernel [%s]\n", kernel );

	type = types[idx];
	while( type != SWAN_END ) {
		void *ptr = ptrs[idx];
		switch( type ) {
//			DEBLOCK( SWAN_uchar, uchar,  1 );
			DEBLOCK( SWAN_uchar2, uchar2,  2 );
			DEBLOCK( SWAN_uchar3, uchar3,  1 );
			DEBLOCK( SWAN_uchar4, uchar4,  4 );
			DEBLOCK( SWAN_char , int,  1 );
//			DEBLOCK( SWAN_char1 , char1,  1 );
			DEBLOCK( SWAN_char2 , char2,  2 );
			DEBLOCK( SWAN_char3 , char3,  1 );
			DEBLOCK( SWAN_char4 , char4,  4 );
			DEBLOCK( SWAN_int, int,  4 );
//			DEBLOCK( SWAN_int1, int1,  4 );
			DEBLOCK( SWAN_int2, int2,  8 );
			DEBLOCK( SWAN_int3, int3,  4 );
			DEBLOCK( SWAN_int4, int4,  16 );
//			DEBLOCK( SWAN_float, double,  4 );
//			DEBLOCK( SWAN_float1, float1,  4 );
			DEBLOCK( SWAN_float2, float2,  8 );
			DEBLOCK( SWAN_float3, float3,  4 );
			DEBLOCK( SWAN_float4, float4,  16 );

			DEBLOCK( SWAN_uint, uint,  4 );
			DEBLOCK( SWAN_uint2, uint2,  8 );
			DEBLOCK( SWAN_uint3, uint3,  4 );
			DEBLOCK( SWAN_uint4, uint4,  16 );
			DEBLOCK( SWAN_float, float,  4 );


//#define DEBLOCK(swan_type,type,OFFSET) 
#if ( CUDA_MAJOR == 3 && CUDA_MINOR >= 2 ) || CUDA_MAJOR >= 4
			case SWAN_PTR: 
				{
//printf("PTR as NATIVE\n");
				ALIGN_UP( offset, (sizeof(void*)));
				cuParamSetv( f, offset, ptr, sizeof(void*) );
				offset += sizeof(void*); }
			break;
#else
			case SWAN_PTR: 
				{
//printf("PTR as INT\n");
				ALIGN_UP( offset, (sizeof(int)));
				cuParamSetv( f, offset, ptr, sizeof(int) );
				offset += sizeof(int); }
			break;
#endif



			default:
        printf("%d\n", type );
				error("Parameter type not handled\n");


		}
		idx++;
		type = types[idx];
	}

//printf("Launching kernel [%s] [%X]  with (%d %d %d) (%d %d %d)\n", kernel, f, grid.x, grid.y, grid.z, block.x, block.y, block.z );
//printf(" TOTAL OFFSET %d\n", offset );
	CU_SAFE_CALL_NO_SYNC( cuParamSetSize( f, offset ) );
	CU_SAFE_CALL_NO_SYNC( cuFuncSetBlockShape( f, block.x, block.y, block.z ) );
	CU_SAFE_CALL_NO_SYNC( cuFuncSetSharedSize( f, shmem ) );
#if (CUDA_MAJOR ==3 && CUDA_MINOR >=1 ) || CUDA_MAJOR>=4
	cuFuncSetCacheConfig( f, CU_FUNC_CACHE_PREFER_SHARED ); // This seems to be better in every case for acemd
#endif

	err = cuLaunchGridAsync( f, grid.x, grid.y, NULL ) ; //state.stream ) ;

	if( err != CUDA_SUCCESS ) {
		fprintf( stderr , "SWAN : FATAL : Failure executing kernel [%s] [%d] [%d,%d,%d][%d,%d,%d]\n", kernel, err, grid.x ,grid.y, grid.z, block.x, block.y, block.z );
	assert(0);
		exit(-99);
	}

//printf("Kernel completed\n" );
}


static void try_init( void ) {
  int deviceCount = 0;                                                    
	int syncflag;
	char *sync;
  CUresult err = cuInit(0);
	CUresult status;
  struct cudaDeviceProp deviceProp;
	if( state.init  ) { return; } // already initialised
  state.device = 0;                                                            

  if (CUDA_SUCCESS == err)                                                 
  CU_SAFE_CALL_NO_SYNC(cuDeviceGetCount(&deviceCount));                
  if (deviceCount == 0) {  error( "No device found" ); }


	if( state.target_device >= deviceCount ) {
		error( "Invalid device requested" );
	}



        CU_SAFE_CALL_NO_SYNC(cuDeviceGet(&(state.device), state.target_device));        
#ifdef USE_BOINC
	syncflag=0x4;
#else
	syncflag=0x0;
#endif

	sync = getenv("SWAN_SYNC" );
	if( sync != NULL ) {
		syncflag = atoi( sync );
		fprintf(stderr, "SWAN: Using synchronization method %d\n", syncflag );
	}

	if( getenv("SWAN_PROFILE") || getenv("CUDA_PROFILE") ) {
		state.debug = 1;
	}
	if( state.debug ) {
		printf("SWAN: Built for CUDA version %d.%d\n", CUDA_MAJOR, CUDA_MINOR );
	}

#ifdef DEV_EXTENSIONS
	syncflag |= CU_CTX_MAP_HOST;
#endif

#ifdef USE_FIXED_DEVICE
printf( "********************************************** OVERRIDING DEVICE ALLOCATION\n");
  status = cuCtxCreate( &(state.context), syncflag ,  0 ); // state.device + state.target_device );
  if ( CUDA_SUCCESS != status ) error ( "Unable to create context\n" );
#else
  status = cuCtxCreate( &(state.context), syncflag , state.target_device );


  if ( CUDA_SUCCESS != status ) {
   printf("SWAN: Failed to get requested device (compute exclusive mode). Trying any..\n" );
    int count;
    cuDeviceGetCount( &count );
    int i=0;

    while ( (i < count) &&  (status != CUDA_SUCCESS) ) {
      state.target_device = i;
      CU_SAFE_CALL_NO_SYNC(cuDeviceGet(&(state.device), state.target_device));
      status = cuCtxCreate( &(state.context), syncflag , state.target_device );
			i++;
    }
	}

#endif

  if ( CUDA_SUCCESS != status ) error ( "Unable to create context\n" );

	state.mods         = NULL;
	state.mod_names    = NULL;
	state.num_mods     = 0;
	state.funcs        = NULL;
	state.func_names   = NULL;
	state.num_funcs    = 0;

	state.init = 1;

  cudaGetDeviceProperties(&deviceProp, state.device);

	cuStreamCreate( &state.stream, 0 );
	state.device_version = (deviceProp.major * 100 + deviceProp.minor * 10);
	state.multiProcessorCount = deviceProp.multiProcessorCount;

}

static void error( char *e ) {
	fprintf( stderr, "SWAN: FATAL : %s\n", e );
	exit(-40);
}


void swanMemset( void *ptr, unsigned char b, size_t len ) {
	CUresult  err;
	if( len == 0 ) { return; }

	if( (0 == len % 16) && b==0 ) {
			block_config_t grid, block;
			int threads;
//			if( __CUDA_ARCH__ == 110 ) {
				threads = 128;
//			} else {
//				threads = CUDA_THREAD_MAX;
//			}

			swanDecompose( &grid, &block, len/(sizeof(uint4)), threads );

			if( grid.x < 65535 ) {
				k_swan_fast_fill( grid, block, 0, (uint4*) ptr, len/(sizeof(uint4) )  );
			}
			else {
				// the region to be zeroed is too large for the simple-minded fill kernel 
				// fall-back to something dumb
				err = cuMemsetD32( PTR_TO_CUDEVPTR(ptr), 0, len/4 );
			}
			return;
	}
	else if( 0 == len % 4 ) {
//			printf("SWAN: Warning: swanMemset using D32\n");
			unsigned word = 0;
			word = b;
			word <<=8;
			word |= b;
			word = word | (word<<16);
			err = cuMemsetD32( PTR_TO_CUDEVPTR(ptr), word, len/4 );
	}
	else {
		printf("SWAN: Warning: swanMemset using D8\n");
		err = cuMemsetD8( PTR_TO_CUDEVPTR(ptr), b, len );
	}
	if ( err != CUDA_SUCCESS ) {
		error("swanMemset failed\n" );
	}
}
void *swanMalloc( size_t len ) {
	void *ptr;
	CUdeviceptr dptr;
	CUresult err;
	try_init();
	if( len == 0 ) {
//		printf("SWAN: WARNING - swnaMalloc() called with 0\n");
		return NULL;
	}

	err = cuMemAlloc( (CUdeviceptr*) &dptr, len );
	
	ptr = (void*)dptr;

	if ( err != CUDA_SUCCESS ) {
		printf("Attempted to allocate %lu bytes (%lu already allocated)\n", len, state.bytes_allocated );
	abort();
		error("swanMalloc failed\n" );
	}
	state.bytes_allocated += len;


	// MJH likes his memory clean
	swanMemset( ptr, 0, len );
	return ptr;
}


void *swanMallocHost( size_t len ) {
	CUresult err;
	void *ptr;
	try_init();

#ifdef DEV_EXTENSIONS
	err= cuMemHostAlloc( &ptr, len, CU_MEMHOSTALLOC_PORTABLE  ); //| CU_MEMHOSTALLOC_DEVICEMAP  ); //| CU_MEMHOSTALLOC_WRITECOMBINED );
#else
	err = cuMemAllocHost( &ptr, len );
#endif

	if ( err != CUDA_SUCCESS ) {
		fprintf( stderr, "swanMallocHost error: %d\n", err );
		error("swanMallocHost failed\n" );
	}
//	printf("MallocHost %p\n", ptr );
	memset( ptr, 0, len );
	return ptr;
}

void *swanDevicePtrForHostPtr( void *ptr ) {
	CUdeviceptr ptrd;	
	CUresult err = cuMemHostGetDevicePointer( &ptrd, (CUdeviceptr) ptr, 0 );
	if ( err != CUDA_SUCCESS ) {
		error("swanMallocHost failed\n" );
	}
	return (void*) ptrd;
	
}


void swanFree( void *ptr ) {
#ifdef MALLOC_DEBUG
	printf("SWAN Free %p\n", ptr );
#endif
	CUresult err = cuMemFree(  PTR_TO_CUDEVPTR(ptr) );
	if ( err != CUDA_SUCCESS ) {
		error("swanFree failed\n" );
	}
}

void swanFreeHost( void *ptr ) {
	//printf("FreeHost %p\n", ptr );
	CUresult err = cuMemFreeHost( ptr );
	if ( err != CUDA_SUCCESS ) {
		error("swanFreeHost failed\n" );
	}
}

void swanSynchronize( void ) {
	CUresult err =cuCtxSynchronize();

	if ( err != CUDA_SUCCESS ) {
		error("swanSynchronize failed\n" );
	}
	if( state.debug ) {
		printf("# swanSynchronize()\n");
	}
}

void swanMemcpyHtoD( void *p_h, void *p_d, size_t len ) {
	CUresult err = cuMemcpyHtoD( PTR_TO_CUDEVPTR(p_d), p_h, len );
	if ( err != CUDA_SUCCESS ) {
		error("swanMemcpyHtoD failed\n" );
	}
}

void swanMemcpyDtoH( void *p_d, void *p_h, size_t len ) {
	CUresult	err=CUDA_SUCCESS; //cuCtxSynchronize();
	if ( err != CUDA_SUCCESS ) {
		error("swanMemcpyDtoH sync failed\n" );
	}

	err = cuMemcpyDtoH(  p_h, PTR_TO_CUDEVPTR(p_d), len );
	if ( err != CUDA_SUCCESS ) {
		error("swanMemcpyDtoH failed\n" );
	}
}

void swanMemcpyDtoD( void *psrc, void *pdest, size_t len ) {
	CUresult err = cuMemcpyDtoD(  PTR_TO_CUDEVPTR( pdest ), PTR_TO_CUDEVPTR(psrc), len );
	
//	err=cuCtxSynchronize();
//	if ( err != CUDA_SUCCESS ) {
//		error("swanMemcpyDtoD sync failed\n" );
//	}
	if ( err != CUDA_SUCCESS ) {
		error("swanMemcpyDtoD failed\n" );
	}

}


void *swanMallocPitch( size_t *pitch_in_bytes, size_t width_in_bytes, size_t height ) {
/*
	void *ptr;
	ptr = swanMalloc( width_in_bytes * height );
	*pitch_in_bytes = width_in_bytes;
	return ptr;
*/
CUdeviceptr  dptr;
	CUresult err;
	void *ptr;

	if( width_in_bytes == 0 || height == 0 ) {
//		printf("SWAN: WARNING: swanMAllocPitch called with 0 argument\n" );
		return NULL;
	}

	err = cuMemAllocPitch( &dptr,  (size_t*) pitch_in_bytes,  (size_t) width_in_bytes, height, sizeof(float4) );
	if ( err != CUDA_SUCCESS ) {
		error("swanMallocPitch failed\n" );
	}

	ptr = (void*)(size_t) dptr;
	return (void*) ptr;

}


static CUmodule swanGetModule( const char *modname ) {
	int i;
	for( i=0; i < state.num_mods; i++ ) {
		if( !strcmp( state.mod_names[i], modname  ) ) {
			return state.mods[i];
		}
	} 	
	error( "swanGetModule: module not found" );
}


void swanBindToTexture1DEx(  const char *modname, const char *texname, size_t width, void *ptr, size_t typesize, int flags ) {
	CUresult err;
    CUtexref cu_texref;
	int mode, channels;

		// get the module
		CUmodule mod  = swanGetModule( modname );

		// get the texture
    err = cuModuleGetTexRef(&cu_texref, mod, texname );
		if( err != CUDA_SUCCESS) { error( "swanBindToTexture1D failed -- texture not found" ); }


		// now bind
	 err = cuTexRefSetAddress( NULL, cu_texref,  PTR_TO_CUDEVPTR(ptr), width * typesize );
	if( err != CUDA_SUCCESS) { 
			printf("EEERRR = %d\n", err );
		error( "swanBindToTexture1D failed -- bind failed" ); 
	}

// does not work for linear memory
/*
	if( (flags & TEXTURE_INTERPOLATE) == TEXTURE_INTERPOLATE ) {
		err = cuTexRefSetFilterMode( cu_texref, CU_TR_FILTER_MODE_LINEAR );
	}
	else {
		err = cuTexRefSetFilterMode( cu_texref, CU_TR_FILTER_MODE_POINT );
	}
		if( err != CUDA_SUCCESS) { error( "swanBindToTexture1D failed -- setfiltermode failed" ); }
*/

	mode = flags & TEXTURE_TYPE_MASK;
	channels = typesize / sizeof(float);
	switch( mode ) {
		case TEXTURE_FLOAT:
		err = cuTexRefSetFormat( cu_texref, CU_AD_FORMAT_FLOAT, channels );
		break;
		case TEXTURE_INT:
		err = cuTexRefSetFormat( cu_texref, CU_AD_FORMAT_SIGNED_INT32, channels );
		break;
		case TEXTURE_UINT:
		err = cuTexRefSetFormat( cu_texref, CU_AD_FORMAT_UNSIGNED_INT32, channels );
		break;
		default:
			error( "swanBinToTexture1D failed -- invalid format" );
	}

	if( err != CUDA_SUCCESS) {	
			error( "swanBinToTexture1D failed -- setformat failed" );
	}


	return;

}

void swanMakeTexture1DEx(  const char *modname, const char *texname, size_t width,  void *ptr, size_t typesize, int flags ) {
	int err;
		// get the texture
    CUtexref cu_texref;
	int mode, channels;
		CUarray array;
  CUDA_MEMCPY2D copyParam;
   CUDA_ARRAY_DESCRIPTOR p;

		// get the module
		CUmodule mod  = swanGetModule( modname );

    err = cuModuleGetTexRef(&cu_texref, mod, texname );
		if( err != CUDA_SUCCESS) { error( "swanMakeTexture1D failed -- texture not found" ); }

		p.Width = width;
		p.Height= 1;
	mode = flags & TEXTURE_TYPE_MASK;
	channels = typesize / sizeof(float);
	switch( mode ) {
		case TEXTURE_FLOAT:
		p.Format = CU_AD_FORMAT_FLOAT;
		p.NumChannels = channels;
		break;
		case TEXTURE_INT:
		p.Format = CU_AD_FORMAT_SIGNED_INT32;
		p.NumChannels = channels;
		break;
		case TEXTURE_UINT:
		p.Format = CU_AD_FORMAT_UNSIGNED_INT32;
		p.NumChannels = channels;
		break;
		default:
			error( "swanMakeTexture1D failed -- invalid format" );
	}


	  err = cuArrayCreate(  &array	, &p);
		if( err != CUDA_SUCCESS) { error( "swanMakeTexture1D failed -- array create failed" ); }

  memset(&copyParam, 0, sizeof(copyParam));
  copyParam.dstMemoryType = CU_MEMORYTYPE_ARRAY;
  copyParam.dstArray = array;
  copyParam.srcMemoryType = CU_MEMORYTYPE_HOST;
  copyParam.srcHost = ptr;
  copyParam.srcPitch = width * sizeof(float);
  copyParam.WidthInBytes = copyParam.srcPitch;
  copyParam.Height = 1;
  // err = cuMemcpy2D(&copyParam);


	err = cuMemcpyHtoA( array, 0, ptr,  typesize  * width );
	if( err != CUDA_SUCCESS) { error( "swanMakeTexture1D failed -- memcpy failed" ); }
 
	err = cuTexRefSetArray ( cu_texref, array, CU_TRSA_OVERRIDE_FORMAT );
	if( err != CUDA_SUCCESS) { error( "swanMakeTexture1D failed -- setarray failed" ); }


	if( (flags & TEXTURE_INTERPOLATE) == TEXTURE_INTERPOLATE ) {
		err = cuTexRefSetFilterMode( cu_texref, CU_TR_FILTER_MODE_LINEAR );
	}
	else {
		err = cuTexRefSetFilterMode( cu_texref, CU_TR_FILTER_MODE_POINT );
	}
		if( err != CUDA_SUCCESS) { error( "swanBindToTexture1D failed -- setfiltermode failed" ); }

	if(  (flags & TEXTURE_NORMALISE ) == TEXTURE_NORMALISE ) {
		err  = cuTexRefSetFlags(cu_texref, CU_TRSF_NORMALIZED_COORDINATES);
    err |= cuTexRefSetAddressMode(cu_texref, 0, CU_TR_ADDRESS_MODE_CLAMP);
    err |= cuTexRefSetAddressMode(cu_texref, 1, CU_TR_ADDRESS_MODE_CLAMP);
		if( err != CUDA_SUCCESS) { error( "swanBindToTexture1D failed -- setflags 1 failed" ); }
	}

		err = cuTexRefSetFormat( cu_texref, CU_AD_FORMAT_FLOAT, channels );
		if( err != CUDA_SUCCESS) { error( "swanBindToTexture1D failed -- setformat failed" ); }

//printf("TEX BIND DONE\n");
}

void swanMakeTexture2DEx(  const char *modname, const char *texname, size_t width, size_t height, void *ptr, size_t typesize, int flags ) {
	error("swanMakeTexture2DEx : not implemented" );
}

void swanBindToConstantEx( const char *modname, const  char *constname, size_t len , void *ptr ) {
	CUmodule mod = swanGetModule( modname );
	CUdeviceptr p;
	size_t lenr;
	int err = cuModuleGetGlobal( &p, &lenr, mod, constname );

	if( err != CUDA_SUCCESS ) { error ("swanBindToConstant failed -- no such name" ); }

	if( len != lenr ) {
		if( err != CUDA_SUCCESS ) { error ("swanBindToConstant failed -- size wrong" ); }
	}
	err = cuMemcpyHtoD( p, ptr, len );
	if( err != CUDA_SUCCESS ) { error ("swanBindToConstant failed -- copy failed" ); }
}


void swanDecompose( block_config_t *grid, block_config_t *block, int thread_count, int threads_per_block ) {
  grid->x = grid->y = grid->z  = 1;
  block->x= block->y= block->z = 1;

  block->x = threads_per_block;
  grid->x  = thread_count / threads_per_block;

  if( (grid->x * threads_per_block) < thread_count ) { grid->x ++; }

//printf(" Decompose %d threads (%d per block) : %d blocks \n", thread_count, threads_per_block, grid->x );
}

void swanSetTargetDevice( int i ) {
}


int swanGetNumberOfComputeElements( void ) {
	try_init();
//	struct cudaDeviceProp prop;
//	cudaGetDeviceProperties( &prop, state.device );
	return state.multiProcessorCount;
}


// end extern c

int swanDeviceVersion( void ) {
	try_init();

	return state.device_version;

}

size_t swanMemAvailable( void ) {
	size_t free, total;
	try_init();

	cuMemGetInfo( &free, &total );
	
	return free;
}

int swanMaxThreadCount( void ) {
	switch( swanDeviceVersion() ) {
		case SWAN_DEVICE_CUDA_10:
		case SWAN_DEVICE_CUDA_11:
			return 384;
		case SWAN_DEVICE_CUDA_12:
		case SWAN_DEVICE_CUDA_13:
		case SWAN_DEVICE_CUDA_20:
			return 512;
	}
}

void swanFinalize(void) {
}

#include "deviceQuery.c"

#ifdef DEV_EXTENSIONS
 
#ifndef _WINDOWS
	#include "threading.c"
#endif

#include "checkpoint.c"

#endif


const char *swanGetVersion(void) { return "$Rev: 1542 $"; }

#ifdef __cplusplus
}
#endif
