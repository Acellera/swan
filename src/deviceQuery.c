
#ifdef OPENCL
int swanDeviceVersion( void ) {
	printf("swanDeviceVersion  :: TODO\n" );
	return 0;
}

void swanDeviceName(void) { 
	printf("swanDeviceName :: TODO\n" );
	
}

int  swanEnumerateDevices( void ) {
	printf("swanEnumerateDevices :: TODO -- pretending there's a device\n" );
	return 0;	
}

#else


void swanDeviceName(void) { 
    int dev;
  struct cudaDeviceProp deviceProp;
    cudaGetDevice( &dev );
    cudaGetDeviceProperties(&deviceProp, dev);
  printf("%s\n", deviceProp.name );
}

int swanGetDeviceCount(void) {
	int deviceCount;
    cudaGetDeviceCount(&deviceCount);
	return deviceCount;
}
int  swanEnumerateDevices( FILE *fout ) {


    int deviceCount=0;
    int dev;
		CUresult err;
    err = cudaGetDeviceCount(&deviceCount);

		if( (err != CUDA_SUCCESS)   ||  (deviceCount>32) ) {
			error("Unable to enumerate devices");

		}
    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
        fprintf( fout, "# There is no device supporting CUDA\n");
    for (dev = 0; dev < deviceCount; ++dev) {
        struct cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        fprintf( fout, "# \n# Device %d: \"%s\"\n", dev, deviceProp.name);
        fprintf( fout, "#  CUDA Capability Major revision number:         %d\n", deviceProp.major);
        fprintf( fout, "#  CUDA Capability Minor revision number:         %d\n", deviceProp.minor);
        fprintf( fout, "#  Total amount of global memory:                 %u bytes\n", deviceProp.totalGlobalMem);
//    #if CUDART_VERSION >= 2000
        fprintf( fout, "#  Number of multiprocessors:                     %d\n", deviceProp.multiProcessorCount);
        fprintf( fout, "#  Number of cores:                               %d\n", 8 * deviceProp.multiProcessorCount);
//    #endif
        fprintf( fout, "#  Total amount of constant memory:               %u bytes\n", deviceProp.totalConstMem); 
        fprintf( fout, "#  Total amount of shared memory per block:       %u bytes\n", deviceProp.sharedMemPerBlock);
        fprintf( fout, "#  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
        fprintf( fout, "#  Warp size:                                     %d\n", deviceProp.warpSize);
        fprintf( fout, "#  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
        fprintf( fout, "#  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        fprintf( fout, "#  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        fprintf( fout, "#  Maximum memory pitch:                          %u bytes\n", deviceProp.memPitch);
        fprintf( fout, "#  Texture alignment:                             %u bytes\n", deviceProp.textureAlignment);
        fprintf( fout, "#  Clock rate:                                    %.2f GHz\n", deviceProp.clockRate * 1e-6f);
//    #if CUDART_VERSION >= 2000
        fprintf( fout, "#  Concurrent copy and execution:                 %s\n", deviceProp.deviceOverlap ? "Yes" : "No");
//    #endif
//    #if CUDART_VERSION >= 2020
        fprintf( fout, "#  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
        fprintf( fout, "#  Integrated:                                    %s\n", deviceProp.integrated ? "Yes" : "No");
        fprintf( fout, "#  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
        fprintf( fout, "#  Compute mode:                                  %s\n", deviceProp.computeMode == cudaComputeModeDefault ?
			                                                            "Default (multiple host threads can use this device simultaneously)" :
		                                                                deviceProp.computeMode == cudaComputeModeExclusive ?
																		"Exclusive (only one host thread at a time can use this device)" :
		                                                                deviceProp.computeMode == cudaComputeModeProhibited ?
																		"Prohibited (no host thread can use this device)" :
																		"Unknown");
//    #endif

        if (dev == 0) {
			// This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
            if (deviceProp.major == 9999 && deviceProp.minor == 9999)
                fprintf( stderr, "# There is no device supporting CUDA.\n");
            else if (deviceCount == 1)
                fprintf( stderr, "# There is 1 device supporting CUDA\n");
            else
                fprintf( stderr, "# There are %d devices supporting CUDA\n", deviceCount);
        }

#ifdef USE_BOINC
        fprintf(stderr,"# Device %d: \"%s\"\n", dev, deviceProp.name);//GIANNI
        fprintf(stderr,"# Clock rate: %.2f GHz\n", deviceProp.clockRate * 1e-6f);
        fprintf(stderr,"# Total amount of global memory:                 %d bytes\n",
               deviceProp.totalGlobalMem);
    #if CUDART_VERSION >= 2000
        fprintf( stderr, "# Number of multiprocessors:                     %d\n",
               deviceProp.multiProcessorCount);
        fprintf( stderr, "# Number of cores:                               %d\n",
               8 * deviceProp.multiProcessorCount);
    #endif
#endif

}

	return 0;
}



#endif
