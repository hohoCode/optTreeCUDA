INCDIR = -I.
DBG    = -g
OPT    = -arch=sm_20 -O3 --compiler-options -fpermissive -use_fast_math
CPP    = nvcc
CFLAGS = $(OPT) $(INCDIR)
LINK   = -lm 

.cpp.o:
	$(CPP) $(CFLAGS) -c $< -o $@

all: struct_cuda struct_v1 ##struct_v7: v7 requires multiple GPUs

struct_cuda: StructCUDA.cuda.cpp 
	$(CPP) $(CFLAGS) $(LINK) -o out/struct_cuda_v5 src/StructCUDA.cuda.cpp src/StructCUDA.v5.streams.cu

struct_v1: StructCUDA.v6.cpp
	$(CPP) $(CFLAGS) $(LINK) -o out/struct_cuda_v6 src/StructCUDA.v6.cpp src/StructCUDA.v6.floatstream.cu
	
struct_v7: StructCUDA.v7.cpp 
	$(CPP) $(CFLAGS) $(LINK) -o out/struct_cuda_v7 src/StructCUDA.v7.cpp src/StructCUDA.v7.multiG.cu
	
clean:
	rm -rf out/struct_cuda_v*

