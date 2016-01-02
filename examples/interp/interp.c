#include "interp.kh"


#include "swan_api.h"

#include <stdlib.h>

#include <math.h>
#include <stdio.h>

int main( int argc, char **argv ) {
	int N = 64;
	int i;
	block_config_t grid, block;
	float *ref = (float*) malloc( sizeof(float) * N );
	float *outd  = (float*) swanMalloc( sizeof(float) * N * 4 );	 	
	float *outd2 = (float*) swanMalloc( sizeof(float) * N * 4 );	 	
	float *ind   = (float*) swanMalloc( sizeof(float) * N );	 	
	float *outh  = (float*) swanMallocHost( sizeof(float) * N * 4 );	 	
	float *outh2 = (float*) swanMallocHost( sizeof(float) * N * 4 );	 	

	for( i=0; i < N; i++ ){
		ref[i] =sin( M_PI * ((float)i+0.5) / N );
	}

	// Use swanMakeTexture1D to make a texture using an Array object
	// this will allow normalised addressing and interpolation
	swanMakeTexture1D( "tex_sin", N, ref, sizeof(float), TEXTURE_FLOAT  | TEXTURE_INTERPOLATE | TEXTURE_NORMALISE );
	swanMakeTexture1D( "tex_sin3", N, ref, sizeof(float), TEXTURE_FLOAT  | TEXTURE_INTERPOLATE | TEXTURE_NORMALISE );

	// Also bind a texture to a linear array
	swanMemcpyHtoD( ref, ind, sizeof(float) * N );
	swanBindToTexture1D( "tex_sin2", N, ind, sizeof(float), TEXTURE_FLOAT );

//	swanDecompose( &grid, &block, N * 4, N );
grid.x = grid.y = grid.z = 1;
block.y = block.z = 1;
block.x = N*4;
	k_interp( grid, block, 0, outd, outd2, N*4 );

	swanMemcpyDtoH( outd , outh  ,N*4 * sizeof(float) );	
	swanMemcpyDtoH( outd2, outh2 ,N*4 * sizeof(float) );	

	printf("Input\t\tOut texture\tOut linear\tCalculated\n");
	for( i=0; i < N * 4; i++ ){
		float calc = sin (M_PI * ((float)i) / (N*4) );
		if( (i-2)%4 == 0 ) {
			printf("%f\t%f\t%f\t%f\n",		ref[i/4], outh[i], outh2[i], calc );
		}
		else {
			printf("\t\t%f\t%f\t%f\n",		outh[i], outh2[i], calc );
		}
	}

}

