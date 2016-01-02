__global__ void vecadd( float *in1, float *in2, float *out, int N ) {
	int idx= blockDim.x * blockIdx.x + threadIdx.x;
	if( idx < N ) {
		out[idx] =in1[idx] + in2[idx];
	}
}
