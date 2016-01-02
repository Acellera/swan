__global__ void shmem ( int *in, int *out, int N ) {
	extern __shared__ int buf[];

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if( idx < N ) {
		buf[ idx ] = in[ idx ];
	}
	__syncthreads();

	if ( idx < N/2 ) {
		int tmp = buf[  N - idx - 1];
		buf[ N - idx - 1 ] = buf [ idx ];
		buf[ idx ] = tmp;
	}
	__syncthreads();

	if( idx < N ) {
		out[ idx ] = buf[ idx ];
	}
}
