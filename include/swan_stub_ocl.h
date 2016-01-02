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

void swanBindToTexture1DEx( const char *modname,const  char *texname, int width, void *ptr, size_t typesize, int flags );
void swanMakeTexture2DEx( const char *modname,const  char *texname, int width, int height, void *ptr, size_t typesize, int flags );
void swanMakeTexture1DEx( const char *modname,const  char *texname, int width,  void *ptr, size_t typesize, int flags );
void swanBindToConstantEx( struct __swan_const_t *consts,const   char *constname, size_t len , void *ptr );

void* swanMallocReadOnly( size_t len, void *ptr );
void swanLoadProgramFromSource( const char *module, const unsigned char *ptx, size_t len, int target );
void swanRunKernel(  const char *a,  block_config_t grid , block_config_t block, size_t shmem_bytes , int flags, void *ptrs[], int *lens  );
void swanRunKernelAsync(  const char *a,  block_config_t grid , block_config_t block, size_t shmem_bytes , int flags,   void *ptrs[], int *lens );

static __thread int  __swan_initialised = 0;

static void __swan_try_init( void ) {
  if ( !__swan_initialised ) {

    swanLoadProgramFromSource( __SWAN_MODULE_NAME__  , __swan_program_source, sizeof( __swan_program_source ) , 0 );
    __swan_initialised = 1;
		{
			int i=0;
			while( __swan_textures[i] != NULL ) {
				__swan_texture_ptrs[i] = NULL; 
				__swan_texture_images[i] = NULL; 
				__swan_texture_lens[i] = 0; 
				i++;
			}
		}

		// For opencl, we malloc the constant/ module-local device-size things now
//printf("__swan_try_init::: THERE ARE  BUFFERS TO ALLOCATE\n"  );
		int i=0;
    struct __swan_const_t *t = __swan_const;
		while( t->name != NULL ) {
			t->ptr = NULL;
//printf("[%s][%d] [%x]\n", t->name, t->size, t->ptr );
			t++;
		}

  }
}

static void swanBindToGlobal( char *constname, size_t l, void *ptr ) {
	__swan_try_init();

	swanBindToConstantEx( __swan_const, constname, l, ptr );
}

static void swanBindToTexture1D(const  char *texname, int width, void *ptr, size_t typesize, int flags ) {
	__swan_try_init();
	int i = 0;
	int found = 0 ;
	while( __swan_textures[i] != NULL ) {
		if( !strcmp( __swan_textures[i], texname )  ) {
			__swan_texture_ptrs  [i] = ptr; 
			__swan_texture_lens  [i] = width; 
			found = 1;
		}
		i++;
	}
	if( !found ) {
	  fprintf( stderr, "SWAN OCL: FATAL : Texture [%s] not found\n", texname );
  	exit(-40);\
	}

}

void *swanMallocImage2D( void *ptr, int width, size_t typesize ) ;

static void swanMakeTexture1D(  char *texname, int width, void *ptr, size_t typesize, int flags ) {
	__swan_try_init();
	void * ptrd = swanMallocImage2D( ptr, width , typesize );

	void *ptrh = swanMalloc( width * typesize );
	swanMemcpyHtoD( ptr, ptrh, width * typesize );

	int i = 0;
	int found = 0 ;
	while( __swan_textures[i] != NULL ) {
		if( !strcmp( __swan_textures[i], texname ) ) {
			__swan_texture_images[i] = ptrd; 
			__swan_texture_lens  [i] = width; 
			__swan_texture_ptrs  [i] = ptrh; 
			found = 1;
		}
		i++;
	}
	if( !found ) {
	  fprintf( stderr, "SWAN OCL: FATAL : Texture [%s] not found\n", texname );
  	exit(-40);\
	}


}


static void swanMakeTexture2D(  const char *texname, int width, int height, void *ptr, size_t typesize, int flags ) {
	__swan_try_init();
	swanMakeTexture2DEx( __SWAN_MODULE_NAME__, texname, width, height, ptr, typesize, flags );
}


#ifdef __cplusplus
}
#endif
#endif

