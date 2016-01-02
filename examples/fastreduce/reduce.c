#include "swan_api.h"
#include "swan_types.h"


#include "reduce.kh"

#include <stdio.h>
#include <math.h>

#include "xmmintrin.h"
void *mainfunc(void * );

#define NG 4
#define ITER 10000
#define N 29184

int main( int argc, char *argv ) {
	int i, j;

	int devices[NG];

	for( i=0; i < NG; i++ ) {
		devices[i] = 0;
	}

	printf("Making threads\n");

	swanThread_t *thr ;
	thr = swanInitThread( mainfunc, NULL, devices, NG );

	swanWaitForThreads( thr, NG );
}


void *mainfunc( void * arg ) {

	fprintf( stderr, "mainfunc\n");

	int i,j, ix;
	float4 *ptrh1 = (float4*) swanMallocHostShared( "sharedbuffer1", (16+N) * sizeof(float4) );
	float4 *ptrh2 = (float4*) swanMallocHostShared( "sharedbuffer2", (16+N) * sizeof(float4) );
	float4 *ptrh3 = (float4*) swanMallocHostShared( "sharedbuffer3", (16+N) * sizeof(float4) );
	float4 *ptrh4 = (float4*) swanMallocHostShared( "sharedbuffer4", (16+N) * sizeof(float4) );

	float4 *ptrh;


//	float4 *ptrd = swanDevicePtrForHostPtr( ptrh );

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
		switch( swanThreadIndex() ) {
			case 0:
				ptrh = ptrh1;
				break;
			case 1:
				ptrh = ptrh2;
				break;
			case 2:
				ptrh = ptrh3;
				break;
			case 3:
				ptrh = ptrh4;
				break;
			default:
				exit(9);
		}

printf("%p\n", ptrh );

	double t = -swanTime();

	for( i=0; i < ITER; i++ ) {

//		swanMemcpyDtoH( ptrd_local, ptrh, N * sizeof(float4) );
		

		swanThreadBarrier();

	 {
			// SSE summation
			__m128 *p1 = (__m128*) ptrh1;
			__m128 *p2 = (__m128*) ptrh2;
			__m128 *p3 = (__m128*) ptrh3;
			__m128 *p4 = (__m128*) ptrh4;

			int min = N/swanThreadCount() * swanThreadIndex();
			int max = N/swanThreadCount() * (1+swanThreadIndex()) - 16;

			__m128 *px = p2;
			for( ix=min; ix< max; ix+=16 ) {
				j=ix;
				p1[j+0]  =  _mm_add_ps( p1[j+0], px[j+0] );
				p1[j+1]  =  _mm_add_ps( p1[j+1], px[j+1] );
				p1[j+2]  =  _mm_add_ps( p1[j+2], px[j+2] );
				p1[j+3]  =  _mm_add_ps( p1[j+3], px[j+3] );
				p1[j+4]  =  _mm_add_ps( p1[j+4], px[j+4] );
				p1[j+5]  =  _mm_add_ps( p1[j+5], px[j+5] );
				p1[j+6]  =  _mm_add_ps( p1[j+6], px[j+6] );
				p1[j+7]  =  _mm_add_ps( p1[j+7], px[j+7] );
				p1[j+8]  =  _mm_add_ps( p1[j+8], px[j+8] );
				p1[j+9]  =  _mm_add_ps( p1[j+9], px[j+9] );
				p1[j+10]  = _mm_add_ps( p1[j+10], px[j+10] );
				p1[j+11]  = _mm_add_ps( p1[j+11], px[j+11] );
				p1[j+12]  = _mm_add_ps( p1[j+12], px[j+12] );
				p1[j+13]  =  _mm_add_ps( p1[j+13], px[j+13] );
				p1[j+14]  =  _mm_add_ps( p1[j+14], px[j+14] );
				p1[j+15]  =  _mm_add_ps( p1[j+15], px[j+15] );
			}
			px = p3;
			for( ix=min; ix< max; ix+=16 ) {
				j=ix;
				p1[j+0]  =  _mm_add_ps( p1[j+0], px[j+0] );
				p1[j+1]  =  _mm_add_ps( p1[j+1], px[j+1] );
				p1[j+2]  =  _mm_add_ps( p1[j+2], px[j+2] );
				p1[j+3]  =  _mm_add_ps( p1[j+3], px[j+3] );
				p1[j+4]  =  _mm_add_ps( p1[j+4], px[j+4] );
				p1[j+5]  =  _mm_add_ps( p1[j+5], px[j+5] );
				p1[j+6]  =  _mm_add_ps( p1[j+6], px[j+6] );
				p1[j+7]  =  _mm_add_ps( p1[j+7], px[j+7] );
				p1[j+8]  =  _mm_add_ps( p1[j+8], px[j+8] );
				p1[j+9]  =  _mm_add_ps( p1[j+9], px[j+9] );
				p1[j+10]  = _mm_add_ps( p1[j+10], px[j+10] );
				p1[j+11]  = _mm_add_ps( p1[j+11], px[j+11] );
				p1[j+12]  = _mm_add_ps( p1[j+12], px[j+12] );
				p1[j+13]  =  _mm_add_ps( p1[j+13], px[j+13] );
				p1[j+14]  =  _mm_add_ps( p1[j+14], px[j+14] );
				p1[j+15]  =  _mm_add_ps( p1[j+15], px[j+15] );
			}
			px = p4;
			for( ix=min; ix< max; ix+=16 ) {
				j=ix;
				p1[j+0]  =  _mm_add_ps( p1[j+0], px[j+0] );
				p1[j+1]  =  _mm_add_ps( p1[j+1], px[j+1] );
				p1[j+2]  =  _mm_add_ps( p1[j+2], px[j+2] );
				p1[j+3]  =  _mm_add_ps( p1[j+3], px[j+3] );
				p1[j+4]  =  _mm_add_ps( p1[j+4], px[j+4] );
				p1[j+5]  =  _mm_add_ps( p1[j+5], px[j+5] );
				p1[j+6]  =  _mm_add_ps( p1[j+6], px[j+6] );
				p1[j+7]  =  _mm_add_ps( p1[j+7], px[j+7] );
				p1[j+8]  =  _mm_add_ps( p1[j+8], px[j+8] );
				p1[j+9]  =  _mm_add_ps( p1[j+9], px[j+9] );
				p1[j+10]  = _mm_add_ps( p1[j+10], px[j+10] );
				p1[j+11]  = _mm_add_ps( p1[j+11], px[j+11] );
				p1[j+12]  = _mm_add_ps( p1[j+12], px[j+12] );
				p1[j+13]  =  _mm_add_ps( p1[j+13], px[j+13] );
				p1[j+14]  =  _mm_add_ps( p1[j+14], px[j+14] );
				p1[j+15]  =  _mm_add_ps( p1[j+15], px[j+15] );
			}

		}

		swanThreadBarrier();
//		swanMemcpyHtoD( ptrh1, ptrd_local, N * sizeof(float4) );

//		k_sum_2_async( grid, block, 0, ptrd_local, ptrd, 0, N, N );
	}


	swanThreadBarrier();

	t+= swanTime();

	fprintf( stderr, "%d : %f\n", swanThreadIndex(), t/ITER );
	fflush(stdout);


	float4 *h1 = (float4*) malloc( N * sizeof(float4 ) );

//	swanMemcpyDtoH( ptrd_local, h1,  N * sizeof(float4 ) );
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
