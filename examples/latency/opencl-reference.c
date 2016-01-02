#include <stdio.h>
#include "CL/cl.h"
#include <string.h>

#define N 100000
double compat_gettime( void ) {
  struct timeval now;
  gettimeofday(&now, NULL);

 return now.tv_sec + (now.tv_usec / 1000000.0);
}



typedef struct { float x,y,z,w; } float4;
#define IMLEN 10
#define OUTLEN  30

#define CHECK_ERROR_RET  if(status != CL_SUCCESS) {  \
  fprintf( stderr, "SWAN OCL: FATAL : Error at %s:%d : %d\n", __FILE__, __LINE__, status );\
  exit(-40);\
}

cl_context_properties * stupid_icd_thing(void) ;

const char *program_source = "\
\
__kernel void nada_de_nada( __global float4 *ptr ) {}\
";



cl_context_properties * stupid_icd_thing(void) {


    /*
 *  *      * Have a look at the available platforms and pick either
 *   *           * the AMD one if available or a reasonable default.
 *    *                */




    cl_uint numPlatforms;
    cl_platform_id platform = NULL;
    cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);
     CHECK_ERROR_RET;

    if (0 < numPlatforms)
    {
  unsigned int i;
        cl_platform_id* platforms =  (cl_platform_id*) malloc( sizeof(cl_platform_id) * numPlatforms );
        status = clGetPlatformIDs(numPlatforms, platforms, NULL);
                CHECK_ERROR_RET;

        for (i = 0; i < numPlatforms; ++i)
        {
            char pbuf[100];
            status = clGetPlatformInfo(platforms[i],
                                       CL_PLATFORM_VENDOR,
                                       sizeof(pbuf),
                                       pbuf,
                                       NULL);


                        CHECK_ERROR_RET;

            platform = platforms[i];
            if (!strcmp(pbuf, "Advanced Micro Devices, Inc."))
            {
                break;
            }
        }
        free( platforms );
    }


    /*
 *  *      * If we could find our platform, use it. Otherwise pass a NULL and get wh
 *  atever the
 *   *           * implementation thinks we should be using.
 *
 *
*/


    cl_context_properties cps[3] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform,
        0
    };
    /* Use NULL for backward compatibility */

    /* Use NULL for backward compatibility */
        if( platform == NULL ) { return NULL; }

        cl_context_properties *ptr = (cl_context_properties*) malloc( sizeof(cl_context_properties ) * 3 );
  int i;
        for(i=0; i<3;i++ ) {
                ptr[i] = cps[i];
        }
        return ptr;



}

void main(void) {
	int i;
	cl_int status = 0;
  cl_command_queue cq;
  cl_context  context;
  cl_program *programs = NULL;
	cl_device_id *devices;
	cl_program program;
	int num_kernels =0;
	int num_programs = 0;
	cl_kernel* kernels = NULL;
  char      **kernel_names = NULL;

printf(" Runs a null-body, zero-argument kernel and times the per-launch cost\n");
printf(" Reference times on my test system (OpenCL 2.2, Radeon 5850,  Centos 5.5 64bit) shown in parens\n");

	size_t deviceListSize;


  cl_context_properties *cprop = stupid_icd_thing();

  context = clCreateContextFromType(cprop, CL_DEVICE_TYPE_GPU, NULL, NULL, &status);
  CHECK_ERROR_RET;

  clGetContextInfo( context, CL_CONTEXT_DEVICES, 0, NULL, &deviceListSize);
  CHECK_ERROR_RET;
	devices = (cl_device_id *)malloc(deviceListSize);
  CHECK_ERROR_RET;
  clGetContextInfo( context, CL_CONTEXT_DEVICES, deviceListSize, devices, NULL);
  CHECK_ERROR_RET;

  cq      = clCreateCommandQueue( context, devices[ 0 ], 0, &status);
  CHECK_ERROR_RET;

	size_t len = strlen(program_source);
  program = clCreateProgramWithSource( 
			context, 
			1, 
			(const char**) &program_source, 
			&len,
			&status 
		);
  CHECK_ERROR_RET;
  status = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    // get the error and print it
    clGetProgramBuildInfo( program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &len );
    char *ptr = (char*) malloc(len+1);
    clGetProgramBuildInfo( program, devices[0] ,CL_PROGRAM_BUILD_LOG, len, ptr, NULL );
    ptr[len]='\0';
    printf("LOG :\n %s\n", ptr );

	int nn;
  status = clCreateKernelsInProgram( program, 0, NULL, &nn );
  CHECK_ERROR_RET;

printf("There are %d kernels\n", nn );
  kernels = (cl_kernel*) realloc( kernels, (nn) * sizeof(cl_kernel*) );
  status = clCreateKernelsInProgram(program, nn, kernels  , NULL );
  CHECK_ERROR_RET;




	// now make an image
	cl_mem image;
	{
  cl_image_format f;
  f.image_channel_data_type = CL_FLOAT;
  f.image_channel_order = CL_RGBA;

	float4 *ptr = (float4*) malloc( sizeof(float4) * IMLEN );

	for(i=0; i < IMLEN; i++ ) {
		ptr[i].x = (float)i  ;
		ptr[i].y = (float)i  ;
		ptr[i].z = (float)i  ;
		ptr[i].w = (float)i  ;
	}

	int width = IMLEN;

 	image = clCreateImage2D( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &f, width, 1, 0, ptr, &status );
  CHECK_ERROR_RET;
	}



	// Still going.....

	// Make output buffer for kernel
	cl_mem out1;
  out1 = clCreateBuffer( context, CL_MEM_READ_WRITE, OUTLEN * sizeof(float4), NULL, &status);
  CHECK_ERROR_RET;

	cl_mem out2;
  out2 = clCreateBuffer( context, CL_MEM_READ_WRITE, OUTLEN * sizeof(float4), NULL, &status);
  CHECK_ERROR_RET;

	// run kernel

	{
	size_t bt[3], gt[3];
		char *name = "nada_de_nada";
	 cl_kernel k = kernels[0];

  int argc = 0;
	int ii = OUTLEN;

  bt[0] = 1;
  bt[1] = 1;
  bt[2] = 1;
  gt[0] = bt[0];
  gt[1] = bt[1];
  gt[2] = bt[2];

	// run the kernel once, let it compile
	cl_event event;
	status=clSetKernelArg( k, 0, sizeof(cl_mem), &out1 );
  CHECK_ERROR_RET;
  status = clEnqueueNDRangeKernel ( cq, k, 3, 0, gt, bt, 0, NULL,  &event );
  CHECK_ERROR_RET;
  status = clWaitForEvents( 1, &event );
  CHECK_ERROR_RET;

	int i=0;
	double tt = - compat_gettime();
	for( i=0; i< N; i++ ) {
		status=clSetKernelArg( k, 0, sizeof(cl_mem), &out1 );
  	CHECK_ERROR_RET;
	  status = clEnqueueNDRangeKernel ( cq, k, 3, 0, gt, bt, 0, NULL, NULL ); // &event );
  CHECK_ERROR_RET;
	}

	clFinish( cq );

	tt += compat_gettime();
	printf("Time per iter (no wait): %f usec    (40us +/- 3%)\n", tt/N *1000000. );
  
 tt = - compat_gettime();
	for( i=0; i< N; i++ ) {
		status =clSetKernelArg( k, 0, sizeof(cl_mem), &out1 );
  	CHECK_ERROR_RET;
	  status = clEnqueueNDRangeKernel ( cq, k, 3, 0, gt, bt, 0, NULL, NULL ) ; //&event );
  	CHECK_ERROR_RET;
		clFinish( cq );
//  	status = clWaitForEvents( 1, &event );
//	  CHECK_ERROR_RET;
	}

	clFinish( cq );
	tt += compat_gettime();

	printf("Time per iter (wait)   : %f usec   (71us +/- 3%)\n", tt/N *1000000. );





	}


}

