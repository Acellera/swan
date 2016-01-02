#include "CL/cl.h"
#include <stdio.h>

static char * getpinfo( cl_platform_id p, cl_platform_info param ) {
	size_t size;
	clGetPlatformInfo( p, param, 0, NULL, &size );
	char *ptr = (char*) malloc(size);
	clGetPlatformInfo( p, param, size, ptr, NULL );
	return ptr;
}

inline void checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        fprintf(stderr, "#ERROR: %d : %s\n", err, name );
        exit(1);
    }
}


int  swanEnumerateDevices( FILE *fout ) {
    cl_int err;

    // Plaform info
    cl_uint numPlatforms;
    cl_platform_id platform = NULL;
    cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);
		int i;

    if (0 < numPlatforms) {
        cl_platform_id* platforms =  (cl_platform_id*) malloc( sizeof(cl_platform_id) * numPlatforms );
        status = clGetPlatformIDs(numPlatforms, platforms, NULL);


    // Iteratate over platforms
    fprintf( stdout ,  "#Number of platforms:\t\t\t\t%d \n" , numPlatforms );;

				for( i=0; i < numPlatforms; i++ ) {
					cl_platform_id p = platforms[i];
						
        	fprintf( stdout ,  "#  Plaform Profile:\t\t\t\t%s \n", getpinfo( p, CL_PLATFORM_PROFILE ));    
      	  fprintf( stdout ,  "#  Plaform Version:\t\t\t\t%s \n", getpinfo( p, CL_PLATFORM_VERSION ));    
    	    fprintf( stdout ,  "#  Plaform Name:\t\t\t\t\t%s \n", getpinfo( p , CL_PLATFORM_NAME ) );     
  	      fprintf( stdout ,  "#  Plaform Vendor:\t\t\t\t%s \n", getpinfo( p, CL_PLATFORM_VENDOR ) );   
	        fprintf( stdout ,  "#  Plaform Extensions:\t\t\t%s \n", getpinfo( p, CL_PLATFORM_EXTENSIONS ) ); 

        }

//        fprintf( stdout ,  "#  Plaform Name:\t\t\t\t\t%s \n", gpinfo( p, CL_PLATFORM_NAME ) );
				}
#if 0 
        cl::vector<cl::Device> devices;
        (*p).getDevices(CL_DEVICE_TYPE_ALL, &devices);
    
        fprintf( stdout ,  "#Number of devices:\t\t\t\t%s \n" , devices.size() );;
        for (cl::vector<cl::Device>::iterator i = devices.begin(); 
             i != devices.end(); 
             ++i) {
            
            fprintf( stdout ,  "#  Device Type:\t\t\t\t\t%s \n" ;
            cl_device_type dtype = (*i).getInfo<CL_DEVICE_TYPE>();
            switch (dtype) {
            case CL_DEVICE_TYPE_ACCELERATOR:
                fprintf( stdout ,  "#CL_DEVICE_TYPE_ACCRLERATOR" );;
                break;
            case CL_DEVICE_TYPE_CPU:
                fprintf( stdout ,  "#CL_DEVICE_TYPE_CPU" );;
                break;
            case CL_DEVICE_TYPE_DEFAULT:
                fprintf( stdout ,  "#CL_DEVICE_TYPE_DEFAULT" );;
                break;
            case CL_DEVICE_TYPE_GPU:
                fprintf( stdout ,  "#CL_DEVICE_TYPE_GPU" );;
                break;
            }

            fprintf( stdout ,  "#  Device ID:\t\t\t\t\t%s \n" 
                      , (*i).getInfo<CL_DEVICE_VENDOR_ID>() 
                      );;
            
            fprintf( stdout ,  "#  Max compute units:\t\t\t\t%s \n" 
                      , (*i).getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() 
                      );;
            
            fprintf( stdout ,  "#  Max work items dimensions:\t\t\t%s \n" 
                      , (*i).getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() 
                      );;
            
            cl::vector< ::size_t> witems = 
                (*i).getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
            for (unsigned int x = 0; 
                 x < (*i).getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>(); 
                 x++) {
                fprintf( stdout ,  "#    Max work items[" 
                          , x , "#]:\t\t\t\t%s \n" 
                          , witems[x] 
                          );;
            }

            fprintf( stdout ,  "#  Max work group size:\t\t\t\t%s \n" 
                      , (*i).getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() 
                      );;
            
            fprintf( stdout ,  "#  Preferred vector width char:\t\t\t%s \n" 
                      , (*i).getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR>() 
                      );;

            fprintf( stdout ,  "#  Preferred vector width short:\t\t\t%s \n" 
                      , (*i).getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT>() 
                      );;
            
            fprintf( stdout ,  "#  Preferred vector width int:\t\t\t%s \n" 
                      , (*i).getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT>() 
                      );;
            
            fprintf( stdout ,  "#  Preferred vector width long:\t\t\t%s \n" 
                      , (*i).getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG>() 
                      );;
            
            fprintf( stdout ,  "#  Preferred vector width float:\t\t\t%s \n" 
                      , (*i).getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT>() 
                      );;
            
            fprintf( stdout ,  "#  Preferred vector width double:\t\t%s \n" 
                      , (*i).getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE>() 
                      );;
            
            fprintf( stdout ,  "#  Max clock frequency:\t\t\t\t%s \n" 
                      , (*i).getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() 
                      , "#Mhz"
                      );;
            
            fprintf( stdout ,  "#  Address bits:\t\t\t\t\t%s \n" 
                      , (*i).getInfo<CL_DEVICE_ADDRESS_BITS>() 
                      );;        
            
            fprintf( stdout ,  "#  Max memory allocation:\t\t\t%s \n" 
                      , (*i).getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() 
                      );;        
            
            fprintf( stdout ,  "#  Image support:\t\t\t\t%s \n" 
                      , ((*i).getInfo<CL_DEVICE_IMAGE_SUPPORT>() ? "Yes" : "No")
                      );;        
            
            if ((*i).getInfo<CL_DEVICE_IMAGE_SUPPORT>()) {
                fprintf( stdout ,  "#  Max number of images read arguments:\t%s \n" 
                          , (*i).getInfo<CL_DEVICE_MAX_READ_IMAGE_ARGS>()
                          );;        

                fprintf( stdout ,  "#  Max number of images write arguments:\t%s \n" 
                          , (*i).getInfo<CL_DEVICE_MAX_WRITE_IMAGE_ARGS>()
                          );;        
                
                fprintf( stdout ,  "#  Max image 2D width:\t\t\t%s \n" 
                          , (*i).getInfo<CL_DEVICE_IMAGE2D_MAX_WIDTH>()
                          );;        

                fprintf( stdout ,  "#  Max image 2D height:\t\t\t%s \n" 
                          , (*i).getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>()
                          );;        
                
                fprintf( stdout ,  "#  Max image 3D width:\t\t\t%s \n" 
                          , (*i).getInfo<CL_DEVICE_IMAGE3D_MAX_WIDTH>()
                          );;        

                fprintf( stdout ,  "#  Max image 3D height:\t%s \n" 
                          , (*i).getInfo<CL_DEVICE_IMAGE3D_MAX_HEIGHT>()
                          );;        
                
                fprintf( stdout ,  "#  Max image 3D depth:\t\t\t%s \n" 
                          , (*i).getInfo<CL_DEVICE_IMAGE3D_MAX_DEPTH>()
                          );;        

                fprintf( stdout ,  "#  Max samplers within kernel:\t\t%s \n" 
                          , (*i).getInfo<CL_DEVICE_MAX_SAMPLERS>()
                          );;        
            }

            fprintf( stdout ,  "#  Max size of kernel argument:\t\t\t%s \n" 
                      , (*i).getInfo<CL_DEVICE_MAX_PARAMETER_SIZE>()
                      );;        
            
            fprintf( stdout ,  "#  Alignment (bits) of base address:\t\t%s \n" 
                      , (*i).getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>()
                      );;        
            
            fprintf( stdout ,  "#  Minimum alignment (bytes) for any datatype:\t%s \n" 
                      , (*i).getInfo<CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE>()
                      );;        

            fprintf( stdout ,  "#  Single precision floating point capability" );;
            fprintf( stdout ,  "#    Denorms:\t\t\t\t\t%s \n" 
                      , ((*i).getInfo<CL_DEVICE_SINGLE_FP_CONFIG>() & 
                          CL_FP_DENORM ? "Yes" : "No")
                      );;
            fprintf( stdout ,  "#    Quiet NaNs:\t\t\t\t\t%s \n" 
                      , ((*i).getInfo<CL_DEVICE_SINGLE_FP_CONFIG>() & 
                          CL_FP_INF_NAN ? "Yes" : "No")
                      );;
            fprintf( stdout ,  "#    Round to nearest even:\t\t\t%s \n" 
                      , ((*i).getInfo<CL_DEVICE_SINGLE_FP_CONFIG>() &  
                          CL_FP_ROUND_TO_NEAREST ? "Yes" : "No")
                      );;
            fprintf( stdout ,  "#    Round to zero:\t\t\t\t%s \n" 
                      , ((*i).getInfo<CL_DEVICE_SINGLE_FP_CONFIG>() &  
                          CL_FP_ROUND_TO_ZERO ? "Yes" : "No")
                      );;
            fprintf( stdout ,  "#    Round to +ve and infinity:\t\t\t%s \n" 
                      , ((*i).getInfo<CL_DEVICE_SINGLE_FP_CONFIG>() &  
                          CL_FP_ROUND_TO_INF ? "Yes" : "No")
                      );;
            fprintf( stdout ,  "#    IEEE754-2008 fused multiply-add:\t\t%s \n" 
                      , ((*i).getInfo<CL_DEVICE_SINGLE_FP_CONFIG>() &  
                          CL_FP_FMA ? "Yes" : "No")
                      );;

            fprintf( stdout ,  "#  Cache type:\t\t\t\t\t%s \n" ;
            switch ((*i).getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_TYPE>()) {
            case CL_NONE:
                fprintf( stdout ,  "#None" );;
                break;
            case CL_READ_ONLY_CACHE:
                fprintf( stdout ,  "#Read only" );;
                break;
            case CL_READ_WRITE_CACHE:
                fprintf( stdout ,  "#Read/Write" );;
                break;
            }
            
            fprintf( stdout ,  "#  Cache line size:\t\t\t\t%s \n" 
                      , (*i).getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>()
                      );;
            
            fprintf( stdout ,  "#  Cache size:\t\t\t\t\t%s \n" 
                      , (*i).getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>()
                      );;
            
            fprintf( stdout ,  "#  Global memory size:\t\t\t\t%s \n" 
                      , (*i).getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()
                      );;
            
            fprintf( stdout ,  "#  Constant buffer size:\t\t\t\t%s \n" 
                      , (*i).getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>()
                      );;
            
            fprintf( stdout ,  "#  Max number of constant args:\t\t\t%s \n" 
                      , (*i).getInfo<CL_DEVICE_MAX_CONSTANT_ARGS>()
                      );;

            fprintf( stdout ,  "#  Local memory type:\t\t\t\t%s \n" ;
            switch ((*i).getInfo<CL_DEVICE_LOCAL_MEM_TYPE>()) {
            case CL_LOCAL:
                fprintf( stdout ,  "#Scratchpad" );;
                break;
            case CL_GLOBAL:
                fprintf( stdout ,  "#Global" );;
                break;
            }
            

            fprintf( stdout ,  "#  Local memory size:\t\t\t\t%s \n" 
                      , (*i).getInfo<CL_DEVICE_LOCAL_MEM_SIZE>()
                      );;
            
            fprintf( stdout ,  "#  Profiling timer resolution:\t\t\t%s \n" 
                      , (*i).getInfo<CL_DEVICE_PROFILING_TIMER_RESOLUTION>() 
                      );;
            
            fprintf( stdout ,  "#  Device endianess:\t\t\t\t%s \n" 
                      , ((*i).getInfo<CL_DEVICE_ENDIAN_LITTLE>() ? "Little" : "Big") 
                      );;
            
            fprintf( stdout ,  "#  Available:\t\t\t\t\t%s \n" 
                      , ((*i).getInfo<CL_DEVICE_AVAILABLE>() ? "Yes" : "No")
                      );;
     
            fprintf( stdout ,  "#  Compiler available:\t\t\t\t%s \n" 
                      , ((*i).getInfo<CL_DEVICE_COMPILER_AVAILABLE>() ? "Yes" : "No")
                      );;
            
            fprintf( stdout ,  "#  Execution capabilities:\t\t\t\t%s \n" );;
            fprintf( stdout ,  "#    Execute OpenCL kernels:\t\t\t%s \n" 
                      , ((*i).getInfo<CL_DEVICE_EXECUTION_CAPABILITIES>() & 
                          CL_EXEC_KERNEL ? "Yes" : "No")
                      );;
            fprintf( stdout ,  "#    Execute native function:\t\t\t%s \n" 
                      , ((*i).getInfo<CL_DEVICE_EXECUTION_CAPABILITIES>() & 
                          CL_EXEC_NATIVE_KERNEL ? "Yes" : "No")
                      );;
            
            fprintf( stdout ,  "#  Queue properties:\t\t\t\t%s \n" );;
            fprintf( stdout ,  "#    Out-of-Order:\t\t\t\t%s \n" 
                      , ((*i).getInfo<CL_DEVICE_QUEUE_PROPERTIES>() & 
                          CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE ? "Yes" : "No")
                      );;
            fprintf( stdout ,  "#    Profiling :\t\t\t\t\t%s \n" 
                      , ((*i).getInfo<CL_DEVICE_QUEUE_PROPERTIES>() & 
                          CL_QUEUE_PROFILING_ENABLE ? "Yes" : "No")
                      );;
            
            
            fprintf( stdout ,  "#  Platform ID:\t\t\t\t\t%s \n" 
                  , (*i).getInfo<CL_DEVICE_PLATFORM>()
                      );;
            
            fprintf( stdout ,  "#  Name:\t\t\t\t\t\t%s \n" 
                      , (*i).getInfo<CL_DEVICE_NAME>().c_str()
                      );;
            
            fprintf( stdout ,  "#  Vendor:\t\t\t\t\t%s \n" 
                      , (*i).getInfo<CL_DEVICE_VENDOR>().c_str()
                      );;
            
            fprintf( stdout ,  "#  Driver version:\t\t\t\t%s \n" 
                      , (*i).getInfo<CL_DRIVER_VERSION>().c_str()
                      );;
            
            fprintf( stdout ,  "#  Profile:\t\t\t\t\t%s \n" 
                      , (*i).getInfo<CL_DEVICE_PROFILE>().c_str()
                      );;
            
            fprintf( stdout ,  "#  Version:\t\t\t\t\t%s \n" 
                      , (*i).getInfo<CL_DEVICE_VERSION>().c_str()
                      );;

            fprintf( stdout ,  "#  Extensions:\t\t\t\t\t%s \n" 
                      , (*i).getInfo<CL_DEVICE_EXTENSIONS>().c_str()
                      );;
        }
        fprintf( stdout ,  std::endl );;
    }
#endif
    return 0;
}
