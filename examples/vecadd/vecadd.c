#include "swan_api.h"
#include "swan_types.h"

#include "vecadd.kh"

#include <stdio.h>
#include <math.h>

int main( int argc, char *argv ) {
	int N = 1000;
	int i;
	float * ptrh_in1 = (float*) swanMallocHost( N * sizeof(float) );
	float * ptrh_in2 = (float*) swanMallocHost( N * sizeof(float) );
	float * ptrh_out = (float*) swanMallocHost( N * sizeof(float) );

	float * ptrd_in1 = (float*) swanMalloc( N * sizeof(float) );
	float * ptrd_in2 = (float*) swanMalloc( N * sizeof(float) );
	float * ptrd_out = (float*) swanMalloc( N * sizeof(float) );

	for( i=0; i < N; i++ ) {
		ptrh_in1[ i ] = cos( (float)i/N );
		ptrh_in2[ i ] = sin( (float)i/N );
	}

	block_config_t grid;
	block_config_t block;

	swanDecompose( &grid, &block, N, 128 );

	swanMemcpyHtoD( ptrh_in1, ptrd_in1, N * sizeof(float));
	swanMemcpyHtoD( ptrh_in2, ptrd_in2, N * sizeof(float));

	k_vecadd( grid, block, 0, ptrd_in1, ptrd_in2, ptrd_out, N );

	swanMemcpyDtoH( ptrd_out, ptrh_out, N * sizeof(float));

	float max = 0.;
	for( i=0; i < N; i++ ) {
		float delta =  fabs( ptrh_out[i] - ( ptrh_in1[ i ] + ptrh_in2[ i ] ) );
		if( delta > max ) { max = delta; }
	}
	if( max > 1.e-6 ) {
		printf("FAILED: Delta = %e\n", max );
	}
	else {
		printf("PASSED: Delta = %e\n", max );
	}
}
