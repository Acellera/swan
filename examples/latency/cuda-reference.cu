#include <stdio.h>
#include <sys/time.h>

__global__ void kk( float4 *ptr ) {
}

#define N 100000
double compat_gettime( void ) {
  struct timeval now;
  gettimeofday(&now, NULL);

 return now.tv_sec + (now.tv_usec / 1000000.0);
}



int main(void) {

	kk<<<1,1,0>>>(NULL);
	cudaThreadSynchronize();

	double tt =- compat_gettime();
	for(int i=0; i< N; i++ ) {
	}
	cudaThreadSynchronize();
	tt += compat_gettime();

	printf("Time %f us (nosync)\n", tt/N *1000000. );


	tt =- compat_gettime();

	for(int i=0; i< N; i++ ) {
		kk<<<1,1,0>>>( NULL );
	}
	cudaThreadSynchronize();
	tt += compat_gettime();

	printf("Time %f us (nosync)\n", tt/N *1000000. );

	tt =- compat_gettime();
	for(int i=0; i< N; i++ ) {
		kk<<<1,1,0>>>( NULL );
		cudaThreadSynchronize();
	}
	cudaThreadSynchronize();
	tt += compat_gettime();

	printf("Time %f us (sync)\n", tt/N *1000000. );

	return 1;
}
