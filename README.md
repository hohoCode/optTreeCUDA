optTreeCUDA
===========

Code Introduction:

1. There are three versions of CUDA codes for OptTree project: CUDA stream version, CUDA compression version, and CUDA Multiple GPU version.

2. CUDA stream version consists of files:
StructCUDA.cuda.cpp + StructCUDA.h + StuctCUDA.v5.streams.cu

3. CUDA compression version consists of files:
StructCUDA.v6.cpp + StructCUDA.h + StuctCUDA.v6.floatstream.cu

4. The .cpp files are used to load the trees and features. They are similar to the sequential initilization codes, except that the trees are flatten into a linear array for better CUDA support. And also there are some compression codes added.

5. The .cu files are the real CUDA codes, including the device memory allocation/dellocation, host/device data transfer, device computation codes.


===========

How to RUN:

1. Create a directory with name "out".

2. If you are running the codes on UMIACS GPU servers, please specify the lib address:
export LD_LIBRARY_PATH=/usr/local/lib64

3. Type command: make (please note this makefile works with Fermi GPUs (C20XX), if you want to run on Tesla GPUs (C10XX), please simply remove the flag  -arch=sm_20)

4. You will see a bunch of warnings after make during the compilation, please ignore that. (Due to old graphics driver and others.)

5. Executable files - struct_cuda_v5, struct_cuda_v6 - will be generated inside out directory if everything is successful

6. Prepare the input trees and features - One tree is attached in the repo. From mq-2006 dataset, one ensemble tree (I can also sent the prepared LTR data if needed)

7. Cd into "out" directory, and run commands like: (same as the sequential codes, tree file first and feature file second)

./struct_cuda_v5 ../ensemble-3-1.xml.tree.end /scratch0/MQ2006/Fold1/test.txt


==========
