#include <stdio.h>
#include <sys/time.h>
#include "latency.kh"

#define N 100000
double compat_gettime( void ) {
  struct timeval now;
  gettimeofday(&now, NULL);

 return now.tv_sec + (now.tv_usec / 1000000.0);
}



int main(void) {
	block_config_t grid, block;
	int i;

	grid.x = grid.y = grid.z =1;
	block.x = block.y = block.z = 1;


	k_vector( grid, block, 0, NULL );	
	swanSynchronize();

	double tt =- compat_gettime();
	for(i=0; i< N; i++ ) {
		k_vector( grid, block, 0, NULL );	
	}
	swanSynchronize();
	tt += compat_gettime();

	printf("Time %f us (nosync)\n", tt/N *1000000. );

	tt =- compat_gettime();
	for(i=0; i< N; i++ ) {
		k_vector( grid, block, 0, NULL );	
		swanSynchronize();
	}
		swanSynchronize();
	tt += compat_gettime();

	printf("Time %f us (sync)\n", tt/N *1000000. );

	return 1;
}
