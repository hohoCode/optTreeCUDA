#ifndef STRUCT_PLUS_H_GUARD
#define STRUCT_PLUS_H_GUARD

#include<stdlib.h>

typedef struct StructPlus StructPlus;
typedef unsigned int uint;
typedef unsigned short u16;
typedef float f32;
typedef unsigned int u32;
typedef u16 sdt_length;
typedef unsigned short sdt_slope;

typedef struct StructPlus StructPlus;

struct StructPlus {
  StructPlus* right;
  StructPlus* left;
  unsigned long id;
  int fid;
  float threshold;
};

typedef struct StructSimple StructSimple;

struct StructSimple{
  int left;
  int right;
  int fid;
  float threshold;
};

#endif
