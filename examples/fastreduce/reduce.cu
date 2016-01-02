__global__ void fill( float4 *localbuf, float val, float4* ptr, int offset, int N ) {
	int idx= blockDim.x * blockIdx.x + threadIdx.x;
	if( idx < N ) {
		float4 t = localbuf[ idx ];
		t.x += val;
		t.y += val;
		t.z += val;
		t.w += val;

		ptr[ offset + idx ] = t;
	}
}

__global__ void sum_2(  float4 *localbuf, float4 *ptrd, int offset_0, int offset_1, int N ) {
	int idx= blockDim.x * blockIdx.x + threadIdx.x;
	if( idx < N ) {
		float4 t1 = ptrd[ offset_0 + idx ];
		float4 t2 = ptrd[ offset_1 + idx ];
		t1.x += t2.x;
		t1.y += t2.y;
		t1.z += t2.z;
		t1.w += t2.w;

		localbuf[ idx ] = t1;
	}
}

