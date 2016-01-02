#include "swan_api.h"
#include "swan_types.h"

#include "mpi.h"

#include "reduce.kh"

#include <stdio.h>
#include <math.h>

void *mainfunc(void * );

#define NG 2
#define ITER 10000
#define N 29184

int main( int argc, char *argv ) {
	int i, j;

	MPI_Init( NULL, NULL );
	int devices[NG];

	for( i=0; i < NG; i++ ) {
		devices[i] = i;
	}

	printf("Making threads\n");

	swanThread_t *thr ;
	thr = swanInitThread( mainfunc, NULL, devices, NG );

	swanWaitForThreads( thr, NG );
}


void *mainfunc( void * arg ) {

	fprintf( stderr, "mainfunc\n");

	int i,j;
	float4 *ptrh = (float4*) swanMallocHostShared( "sharedbuffer", N * sizeof(float4) * swanThreadCount() );

	float4 *ptrd = swanDevicePtrForHostPtr( ptrh );

	float4 *ptrd_local = (float4*) swanMalloc( N * sizeof(float4) );

	int offset = N * swanThreadIndex();
	int offset_bytes = offset * sizeof(float4);


	printf( "Thread ID %d/%d : Is parallel %d: \n", swanThreadIndex(), swanThreadCount(), swanIsParallel() );

	swanThreadBarrier();


	block_config_t block, grid;

	swanDecompose( &grid, &block, N, 512 );

//	k_fill( grid, block, 0, ptrd_local, i, ptrd, offset, N );
//	swanThreadBarrier();
//	k_sum_2( grid, block, 0, ptrd_local, ptrd, 0, offset, N );

	double t = -swanTime();

	for( i=0; i < ITER; i++ ) {
		k_fill_async( grid, block, 0, ptrd_local, 1., ptrd, offset, N );
		swanThreadBarrier();
		k_sum_2_async( grid, block, 0, ptrd_local, ptrd, 0, N, N );
	}


	swanThreadBarrier();

	t+= swanTime();

	fprintf( stderr, "%d : %f\n", swanThreadIndex(), t/ITER );
	fflush(stdout);


	float4 *h1 = (float4*) malloc( N * sizeof(float4 ) );

	swanMemcpyDtoH( ptrd_local, h1,  N * sizeof(float4 ) );
#if 0
	for( j=0; j<swanThreadCount();j++ ) {
		if( swanThreadIndex() == j ) {
			for(i=0;i<N; i++ ) {
				printf( "%d : %d : %f\n", j, i, h1[i].x );
			}
		}
		swanThreadBarrier();
	}
#endif
	return NULL;
}
