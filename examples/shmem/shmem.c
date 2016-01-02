#include "shmem.kh"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char**argv ) {
	int N = 32;
	int *ptrd;
	int *ptrd2;
	int *ptrh;
	int i;
	block_config_t grid, block;

	ptrd = (int*) swanMalloc( N * sizeof(int) );
	ptrd2= (int*) swanMalloc( N * sizeof(int) );
	ptrh = (int*) swanMallocHost( N * sizeof(int) );

	for( i=0; i < N; i++ ) {	
		ptrh[i] = i;
	}

	swanMemcpyHtoD( ptrh, ptrd, sizeof(int) * N );
	swanDecompose( &grid, &block, N, N );
	k_shmem( grid, block, sizeof(int) * N, ptrd, ptrd2, N );

	swanMemcpyDtoH( ptrd2, ptrh, sizeof(int) * N );

	for( i=0; i < N; i++ ) {	
		if( ptrh[i] != N-i-1 ) {
			printf( "FAILED: %d\t%d\n", i, ptrh[i] );
			exit(0);
		}
	}
	printf("SUCCESS\n");	

	return 0;
}
