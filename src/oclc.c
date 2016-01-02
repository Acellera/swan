#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "CL/cl.h" //opencl.h"

cl_context_properties * stupid_icd_thing(void);
#define CHECK_ERROR if( status != CL_SUCCESS ) { printf("ERROR at %d : %d\n", __LINE__, status); exit(-9); }

static char* load( char *filename );

int main( int argc, char **argv ) {

	setenv( "GPU_IMAGES_SUPPORT", "1",  1);

	int gpu_build = 1;
	int initial_arg = 1;

	char *infile;
	char *outfile;
		char *src;

	if( argc < 2 ) {
			printf("Syntax [--gpu|--cpu] %s outfile infile [compilation arguments]\n", argv[0] );
		exit(0);
	}


	if( argv[ initial_arg][0] == '-' ) {
			if( !strcmp( argv[initial_arg], "--gpu" ) ) {
			gpu_build =1;
			initial_arg++;
			}
			else if( !strcmp( argv[initial_arg], "--cpu" ) ) {
			gpu_build =0;
			initial_arg++;
			}
			else {
				printf("Unrecognised argument [%s]\n" , argv[initial_arg] );
				exit(0);
			}
	}

	outfile = argv[ initial_arg++ ];
	infile  = argv[ initial_arg++ ];


	int i=0; int j=0;
	cl_int status;
	cl_device_id *devices;
	size_t deviceListSize;
  size_t len;

	cl_context context;
	cl_context_properties *cprops = stupid_icd_thing();
	if(!cprops) { CHECK_ERROR("Didn't get a list of context properties\n");}


	if( gpu_build ) {
	  context = clCreateContextFromType(cprops, CL_DEVICE_TYPE_GPU, NULL, NULL, &status);
	}
	else {
	  context = clCreateContextFromType(cprops, CL_DEVICE_TYPE_CPU, NULL, NULL, &status);
	}
	CHECK_ERROR;

  	status = clGetContextInfo( context, CL_CONTEXT_DEVICES, 0, NULL, &deviceListSize);
	CHECK_ERROR;
  if( deviceListSize == 0 ) { exit(-8); }
  devices = (cl_device_id *)malloc(deviceListSize);
  status = clGetContextInfo( context, CL_CONTEXT_DEVICES, deviceListSize, devices, NULL);
	CHECK_ERROR;
  status = clGetContextInfo( context, CL_CONTEXT_DEVICES, 0, NULL, &deviceListSize);
	CHECK_ERROR;
  cl_command_queue commandQueue = clCreateCommandQueue( context, devices[0], 0, &status);


	char *buildargs= NULL ;
	buildargs = (char*) malloc( 1 );
	buildargs[0]='\0';
	for( i=initial_arg; i<argc; i++ ) {
		if( i< argc-1 ) { 
		if( !strcmp ( "-D", argv[i] ) || !strcmp( "-I", argv[i]) ) {
			buildargs =(char*) realloc( buildargs, strlen( buildargs) + strlen( argv[i]  ) +2 );
			strcat( buildargs, argv[i] );
			strcat( buildargs, " " );
			strcat( buildargs, "\0" );
			buildargs =(char*) realloc( buildargs, strlen( buildargs) + strlen( argv[i+1]  )+2 );
			strcat( buildargs, argv[i+1] );
			strcat( buildargs, " " );
			strcat( buildargs, "\0" );
			continue;
		}
		
		}
		if( argv[i][0]=='-' ) {
			buildargs =(char*) realloc( buildargs, strlen( buildargs) + strlen( argv[i]  )+2 );
			strcat( buildargs, argv[i] );
			strcat( buildargs, " " );
			strcat( buildargs, "\0" );
		}
		
		
	}

//printf("INFILE  [%s]\n", infile );
//printf("OUTFILE [%s]\n",  outfile );
//printf("ARGS [%s]\n", buildargs );

	char *ptr;


		ptr = src = load( infile );
		if( ptr == NULL ) {
			printf("Error loading file\n"); exit(-1);
		}

	  size_t strsize[1];
  	strsize[0] = strlen( ptr );
	  cl_program program = clCreateProgramWithSource( context, 1,  (const char**)&ptr, strsize, &status);
		CHECK_ERROR;

	  status = clBuildProgram(program, 1, devices, buildargs, NULL, NULL);

    	// get and print the build log
	    clGetProgramBuildInfo( program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &len );
  	  ptr =  (char*) malloc( len+1 );
	    clGetProgramBuildInfo( program, devices[0], CL_PROGRAM_BUILD_LOG, len, ptr, NULL );
  	  ptr[len]='\0';

	  if( status != CL_SUCCESS ) {
	    printf("BUILD FAILED. Results follow:\n" );
			printf("%s\n", ptr );
    	free(ptr);
		printf("--\n");
			exit(-1);
	  }
		if( strlen(ptr) )  {
	    printf("BUILD WARNINGS: Results follow:\n" );
			printf("%s\n", ptr );
		printf("--\n");
		}
		printf("Compilation successful\n" );

		// now get the binary object
		status = clGetProgramInfo( program, CL_PROGRAM_BINARY_SIZES, 0, NULL, &len );
  	size_t *sptr =  (size_t*) malloc( sizeof(size_t) *len );
		memset( sptr, 0, sizeof(size_t)*len );
		status = clGetProgramInfo( program, CL_PROGRAM_BINARY_SIZES, len, sptr, NULL );


		unsigned char *bin = (unsigned char*) malloc( sptr[0] );

		status = clGetProgramInfo( program, CL_PROGRAM_BINARIES, 0, NULL, &len );
		CHECK_ERROR;
  	ptr =  (char*) malloc( len+1 );
		memset( ptr, 0, len );
		status = clGetProgramInfo( program, CL_PROGRAM_BINARIES, len, &bin,  NULL );
		CHECK_ERROR;

		FILE *fout = fopen( outfile , "w" );

	size_t outlen;
	char *outptr;

//for(int j=0; j < sptr[0]; j++ ) {
//	fprintf( stderr, "%c", bin[j] );
//}
fprintf(stderr, "\n" );
#ifdef OPENCL_FROM_SOURCE
		printf("WARNING: This version of SWAN is embedding source\n"); 
		outptr = src;
		outlen = strlen(src);
#else
	outptr = bin;
	outlen = sptr[0];	

#endif
//printf("[%s]\n", bin );
		fprintf( fout, "{\n" );
		for( i=0; i < outlen; i++ ) {
			char c = outptr[i];
//			if( c == '"' ) { fprintf( fout, "\"" );}
//			else if( c == '\n' ) { fprintf( fout, "\\n\\\n" ); }
			fprintf ( fout, "0x%02x", (unsigned char) outptr[i] );
			if( i<outlen-1 ) { fprintf( fout, ", "); }
			if( (i+1)%16==0) { fprintf( fout, "\n"); }
// 			else fprintf( fout, "%c", c );
		}
		fprintf( fout, "\n};\n\n" );
//	fprintf( fout, "static int __initialised       = 0;\n" );

	cl_int s2;
	len = sptr[0];
  cl_program p2  = clCreateProgramWithBinary( context, 1, devices, &len, (const unsigned char**) &bin, &s2, &status );
  CHECK_ERROR;
  status = clBuildProgram(p2, 1, devices, NULL, NULL, NULL);
  CHECK_ERROR;
	printf("Test build succeeded\n");



	fclose(fout);
	return 0;
}


static char* load( char *filename ) {
  FILE *fin = fopen(filename ,"r" );
  if( fin == NULL ) { return NULL; }
  char *string = NULL;
  int len =0;
  while(!feof(fin) ) {
    len++;
    string = (char*) realloc( string, len );
    string[len-1] = fgetc( fin );
  }
  string[len-1]='\0';
  fclose(fin);
  return string;
}

cl_context_properties * stupid_icd_thing(void) {


    /*
 *      * Have a look at the available platforms and pick either
 *           * the AMD one if available or a reasonable default.
 *                */

	unsigned int i;
    cl_uint numPlatforms;
    cl_platform_id platform = NULL;
    cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);
	CHECK_ERROR;

    if (0 < numPlatforms) 
    {
        cl_platform_id* platforms = (cl_platform_id*) malloc( sizeof(cl_platform_id) * numPlatforms );
        status = clGetPlatformIDs(numPlatforms, platforms, NULL);
		CHECK_ERROR;

        for (i = 0; i < numPlatforms; ++i) 
        {
            char pbuf[100];
            status = clGetPlatformInfo(platforms[i],
                                       CL_PLATFORM_VENDOR,
                                       sizeof(pbuf),
                                       pbuf,
                                       NULL);


			CHECK_ERROR;

            platform = platforms[i];
            if (!strcmp(pbuf, "Advanced Micro Devices, Inc.")) 
            {
                break;
            }
        }
        free( platforms );
    }


    /*
 *      * If we could find our platform, use it. Otherwise pass a NULL and get whatever the
 *           * implementation thinks we should be using.
 *                */


    cl_context_properties cps[3] = 
    {
        CL_CONTEXT_PLATFORM, 
        (cl_context_properties)platform, 
        0
    };
    /* Use NULL for backward compatibility */
	if( platform == NULL ) { return NULL; }
	
	cl_context_properties *ptr = (cl_context_properties*) malloc( sizeof(cl_context_properties ) * 3 );

	for(i=0; i<3;i++ ) {
		ptr[i] = cps[i];
	}
	return ptr;

}

