__global__ void gridblock( uint *ptr ) {
	int index = (blockIdx.z * (gridDim.x * gridDim.y ) + blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y * blockDim.z );
	index += threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x + 1;
	atomicAdd( ptr, index );
}

