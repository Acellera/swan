#ifndef __SWAN_API_H
#define __SWAN_API_H 1

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>

typedef struct {
	int x; int y; int z;
} block_config_t;

void swanLoadProgramFromSource( const char *, const unsigned char *, size_t, int );


typedef long swanThread_t;

void swanEnableCheckpointing( void );

void swanInit(void); /// Not necessary to call this
void swanFinalize(void);

swanThread_t * swanInitThread( void*(*func)(void*), void *arg, int *devices, int ndev );
void swanWaitForThreads( swanThread_t *thr, int n );

int swanGetDeviceCount( void );

void swanSetDeviceNumber( int );
//void swanSetTargetDevice( int );
void *swanDevicePtrForHostPtr( void *ptr );


void swanSynchronize( void );

size_t swanMemAvailable( void );
void *swanMalloc( size_t len );
void *swanMallocPitch( size_t *pitch_in_bytes, size_t width_in_bytes, size_t height );
void swanMemset( void *ptr, unsigned char byte, size_t len );
void *swanMallocHost( size_t len );
void *swanMallocHostShared( const  char *tag, size_t len );
double swanTime(void);
void swanFree( void *ptr );
void swanFreeHost( void *ptr );
void swanMemcpyHtoD( void *ptr_h, void *ptr_d, size_t len );
void swanMemcpyDtoH( void *ptr_d, void *ptr_h, size_t len );
void swanMemcpyDtoD( void *ptr_dsrc, void *ptr_ddest, size_t len );
int swanGetNumberOfComputeElements( void );
int swanEnumerateDevices( FILE *fout );
void swanDeviceName( void );
int swanDeviceVersion( void );
void swanDecompose( block_config_t *grid, block_config_t *block, int thread_count, int threads_per_block ) ;
int swanMaxThreadCount( void );

const char *swanGetVersion(void);

int swanThreadBarrier( void ) ;
int swanThreadIndex( void ) ;
int swanThreadCount( void ) ;
void swanWaitForThreads( swanThread_t *thr, int n ) ;


#ifdef __cplusplus
}
#endif


#define TEXTURE_FLOAT 0x00001
#define TEXTURE_INT   0x00002
#define TEXTURE_UINT  0x00004

#define TEXTURE_TYPE_MASK 0x0000F

#define TEXTURE_NORMALISE   0x00010
#define TEXTURE_INTERPOLATE 0x00020

#define SWAN_DEVICE_BASIC  100
#define SWAN_DEVICE_OPENCL_10 130
#define SWAN_DEVICE_CUDA_10 100
#define SWAN_DEVICE_CUDA_11 110
#define SWAN_DEVICE_CUDA_12 120
#define SWAN_DEVICE_CUDA_13 130
#define SWAN_DEVICE_CUDA_20 200
#define SWAN_DEVICE_CUDA_21 210
#define SWAN_DEVICE_CUDA_22 220
#define SWAN_DEVICE_CUDA_23 230
#endif
