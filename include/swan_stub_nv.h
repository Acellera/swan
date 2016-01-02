#ifdef  __PTX_INCLUDED
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include "swan_api.h"

#ifndef __SWAN_NO_LOAD_TYPES
#include "swan_types.h"
#endif

#include "swan_nv_types.h"

#ifdef _WINDOWS
#define __thread __declspec(thread)
#endif


#ifdef __cplusplus
extern "C" {
#endif

void swanBindToTexture1DEx( const char *modname,const  char *texname, size_t width, void *ptr, size_t typesize, int flags );
void swanMakeTexture2DEx( const char *modname,const  char *texname, size_t width, size_t height, void *ptr, size_t typesize, int flags );
void swanMakeTexture1DEx( const char *modname,const  char *texname, size_t width,  void *ptr, size_t typesize, int flags );
void swanBindToConstantEx( const char *modname, const  char *constname, size_t len , void *ptr );

void swanLoadProgramFromSource( const char *module, const unsigned char *ptx, size_t len, int target );
void swanRunKernel(  const char *a,  block_config_t grid , block_config_t block, size_t shmem_bytes , int flags, void *ptrs[], int *lens  );
void swanRunKernelAsync(  const char *a,  block_config_t grid , block_config_t block, size_t shmem_bytes , int flags,   void *ptrs[], int *lens );

static __thread int  __swan_initialised = 0;

static void __swan_try_init( void ) {
  if ( !__swan_initialised ) {
		const unsigned char *ptr = NULL;
		int len = 0;

		swanInit();

		switch( swanDeviceVersion() ) {
			case SWAN_DEVICE_CUDA_23:
			case SWAN_DEVICE_CUDA_22:
			case SWAN_DEVICE_CUDA_21:
			case  SWAN_DEVICE_CUDA_20: ptr = __swan_program_source_20; len = sizeof( __swan_program_source_20 ); break;

// Note the fall-through. This is important.
			case  SWAN_DEVICE_CUDA_13: ptr = __swan_program_source_13; len = sizeof( __swan_program_source_13 ); 
			case  SWAN_DEVICE_CUDA_12: if( len == 0 ) { ptr = __swan_program_source_12; len = sizeof( __swan_program_source_12 ); }
			case  SWAN_DEVICE_CUDA_11: if( len == 0 ) { ptr = __swan_program_source_11; len = sizeof( __swan_program_source_11 ); }
			default:
			case  SWAN_DEVICE_CUDA_10: if( len == 0 ) { ptr = __swan_program_source_10; len = sizeof( __swan_program_source_10 ); }
		}	

    swanLoadProgramFromSource( __SWAN_MODULE_NAME__  , ptr, len , 0 );
    __swan_initialised = 1;

  }
}

static void swanBindToGlobal( char *constname, size_t l, void *ptr ) {
	__swan_try_init();

	swanBindToConstantEx( __SWAN_MODULE_NAME__, constname, l, ptr );
}

static void swanBindToTexture1D(const  char *texname, int width, void *ptr, size_t typesize, int flags ) {
	__swan_try_init();
	swanBindToTexture1DEx( __SWAN_MODULE_NAME__, texname, width, ptr, typesize, flags );
	
}
static void swanMakeTexture1D(  char *texname, int width, void *ptr, size_t typesize, int flags ) {
	__swan_try_init();
	swanMakeTexture1DEx( __SWAN_MODULE_NAME__, texname, width, ptr, typesize, flags );
}

static void swanMakeTexture2D(  const char *texname, int width, int height, void *ptr, size_t typesize, int flags ) {
	__swan_try_init();
	swanMakeTexture2DEx( __SWAN_MODULE_NAME__, texname, width, height, ptr, typesize, flags );
}


#ifdef __cplusplus
}
#endif
#endif

