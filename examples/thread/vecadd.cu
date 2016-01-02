__global__ void vecadd( float *in1, float *in2, float *out, int N ) {
	int idx= blockDim.x * blockIdx.x + threadIdx.x;
	if( idx < N ) {
		out[idx] =in1[idx] + in2[idx];
	}
}

__global__ void sum( float4 *a, float4 *b, int N ) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if( idx < N ) {
		float4 t1 = a[idx];
		float4 t2 = b[idx];
		t1.x +=  t2.x;
		t1.y +=  t2.y;
		t1.z +=  t2.z;
		t1.w +=  t2.w;
		a[idx] = t1;
	}
}

