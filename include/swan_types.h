#ifndef __SWAN_TYPES 
#define __SWAN_TYPES 1

#ifndef __VECTOR_TYPES_H__

typedef unsigned int uint;

typedef struct double2 {
  double x; double y;
} double2;

typedef struct {
  int x; int y; int z;
} dim3;

typedef struct {
  int x; int y;
} int2;

typedef struct {
  int x; int y; int z;
} int3;

typedef struct {
  int x; int y; int z; int w;
} int4;



typedef struct {
  unsigned int x; unsigned int y; 
} uint2;

typedef struct {
  unsigned int x; unsigned int y; unsigned int z;
} uint3;

typedef struct {
  unsigned int x; unsigned int y; unsigned int z; unsigned int w;
} uint4;



typedef struct {
  float x; float y; 
} float2;

typedef struct {
  float x; float y; float z;
} float3;

typedef struct {
  float x; float y; float z; float w;
} float4;

typedef struct {
  double x; double y; double z; double w;
} double4;

static double4 make_double4( double a, double b, double c, double d ) {
  double4 f;
  f.x = a;
  f.y = b;
  f.z = c;
  f.w = d;
  return f;

}


static float4 make_float4( float a, float b, float c, float d ) {
  float4 f;
  f.x = a;
  f.y = b;
  f.z = c;
  f.w = d;
  return f;

}

static float3 make_float3( float a, float b, float c ) {
  float3 f;
  f.x = a;
  f.y = b;
  f.z = c;
  return f;
}

static float2 make_float2( float a, float b  ) {
  float2 f;
  f.x = a;
  f.y = b;
  return f;
}

static uint2 make_uint2( uint a, uint b  ) {
  uint2 f;
  f.x = a;
  f.y = b;
  return f;
}


#define __global__
#define __device__
#define __constant__
#define __host__

#define __VECTOR_TYPES_H__ 1
#endif

#endif
