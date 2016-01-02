#include "swan_api.h"
#include "swan_types.h"

#define N  10000

int main( void ) {
	int *pd1 = (int*) swanMalloc( sizeof(int) * N ) ;
	int *pd2 = (int*) swanMalloc( sizeof(int) * N ) ;

	int *ph1 = (int*) malloc( sizeof(int) * N );
	int *ph2 = (int*) malloc( sizeof(int) * N );

	int i;

	for(i=0;i<N;i++) {
		ph1[i]=i;
	}
	swanMemcpyHtoD( ph1, pd1, sizeof(int) * N );
	swanMemcpyDtoD( pd1, pd2, sizeof(int) * N );
	swanMemset( pd1, 0, sizeof(int) * N );
	swanMemcpyDtoH( pd2, ph2, sizeof(int) * N );

	for(i=0;i<N;i++) {
		if(ph2[i] != i ) { 
			printf("FAILED: element %d = %d\n", i, ph2[i] );
			exit(-1);
		}
	}
	printf("SUCCESS\n");

}

