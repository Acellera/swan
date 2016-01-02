#include "swan_api.h"
#include "swan_types.h"

#include "vecadd.kh"

#include <stdio.h>
#include <math.h>

void *mainfunc(void * );

#define NG 2
int main( int argc, char *argv ) {
	int i;

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

	int N = 30000;
	int i;
	float4 *hs1 = (float4*) swanMallocHostShared( "shared1", N * sizeof(float4) );
	float4 *hs2 = (float4*) swanMallocHostShared( "shared2", N * sizeof(float4) );

	float4 *ptr_h1;
	float4 *ptr_h2;

printf("%d hs1 %p\n", swanThreadIndex(), hs1 );
printf("%d hs2 %p\n", swanThreadIndex(), hs2 );

	switch( swanThreadIndex() ) {
		case 0:
			ptr_h1 = hs1;
			ptr_h2 = hs2;
			break;
		case 1:
			ptr_h1 = hs2;
			ptr_h2 = hs1;
			break;
		default:
		printf("Fail\n");
		exit(0);
	}

	float4 * ptrd_1 = (float4*) swanMalloc( N * sizeof(float4) );
	float4 * ptrd_2 = (float4*) swanMalloc( N * sizeof(float4) );


	printf( "Thread ID %d/%d : Is parallel %d: \n", swanThreadIndex(), swanThreadCount(), swanIsParallel() );

	swanThreadBarrier();


	float4 *pp = (float4*) malloc( sizeof(float4) * N );
	for( i=0; i < N; i++ ) {
		pp[i].x = swanThreadIndex()+1;
		pp[i].y = swanThreadIndex()+1;
		pp[i].z = swanThreadIndex()+1;
		pp[i].w = swanThreadIndex()+1;
	}
	swanMemcpyHtoD( pp, ptrd_1, sizeof(float4) * N );


	double t = -swanTime();

	for( i=0; i < 10000; i++ ) {
		swanMemcpyDtoH( ptrd_1, ptr_h1, sizeof(float4) * N );

		swanThreadBarrier();
		swanMemcpyHtoD( ptr_h2, ptrd_2, sizeof(float4) * N );

		block_config_t block, grid;
		swanDecompose( &grid, &block, N, 256 );
		k_sum( grid, block, 0, ptrd_1, ptrd_2, N );

	}


	swanThreadBarrier();

	t+= swanTime();

	fprintf( stderr, "%d : %e\n", swanThreadIndex(), t/10000. );
	fflush(stdout);

	swanMemcpyDtoH( ptrd_1, pp, sizeof(float4) * N );

	return NULL;
/*
	int j;
	for(j=0; j < swanThreadCount(); j++ ) {
		if( j== swanThreadIndex() ) {
			for( i= 0; i <N; i++ ) {
				printf("%d %d : %f %f %f %f\n", j, i, pp[i].x, pp[i].y, pp[i].z, pp[i].w );
		
			}
		}
		swanThreadBarrier();
	}

	return NULL;
*/
}
