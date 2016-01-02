texture<float, 1, cudaReadModeElementType> tex_sin;
texture<float, 1, cudaReadModeElementType> tex_sin2;
texture<float, 1, cudaReadModeElementType> tex_sin3;


__global__ void interp( float* out, float *out2, int N ) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if( idx < N ) {
		// interpolated texture lookup
		out[idx] = tex1D( tex_sin3 , (float) (idx) / N   );
		out2[idx] = tex1D( tex_sin, (float) (idx) / N   );

//		float a1 = tex1Dfetch( tex_sin2, idx/4 );

//		out2[idx] = a1;
	}
}

