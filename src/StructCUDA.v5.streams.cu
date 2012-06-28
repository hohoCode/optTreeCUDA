//////////////////////////////Author: Hua He
//////////////////////////////OptTree CUDA Version 5 - Stream
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "StructCUDA.h"
#include "device_launch_parameters.h"

////Please make sure NumberOfThreads is a power of 2, for multithreads reduction purpose.
#define NumberOfThreads 64
////Please make sure the number of CUDA streams is an even number and >= 2, for better stream overlapping purpose.
#define STREAM_COUNT 32

extern "C" void kernel_wrapper (float* a, StructSimple* b, int* c, int d, int e, int f, int g);

__device__ float getLeafLoop(StructSimple* root, float* featureS) {
	StructSimple* item = root;
	while(item->left!=0||item->right!=0){
		float cc = *(featureS + item->fid);

		if( cc <= item->threshold) {
			if(item->left==0){
				break;
			}
			item = root + item->left;
		} else {
			if(item->right==0){
				break;
			}
			item = root + item->right;
		}
	}	
	return item->threshold;
}

__global__ void scoreAccumulator(StructSimple* tree, float* feature, int* nodeSizes, int numberOfInstances, int nbTrees, int numberOfFeatures, float* output){
	///The total thread number is numberOfInstances*nbTrees. Many threads will run CONCURRENTLY...
	int treeNo = blockIdx.y; 
	int cacheIndex = threadIdx.y * blockDim.x + threadIdx.x; 
	int instanceNo = blockIdx.x * blockDim.x + threadIdx.x; 
	__shared__ float out[NumberOfThreads];
	out[cacheIndex] = 0;
	__syncthreads();

	if(treeNo < nbTrees && instanceNo < numberOfInstances){
		int fStart = (numberOfFeatures)*instanceNo;
		int tStart = *(nodeSizes+treeNo);
		//////////////////////// Call the tree traversal code
		out[cacheIndex] = getLeafLoop(tree+tStart, feature+fStart);			
	}
	__syncthreads();
	int i = NumberOfThreads/2;
	while (i != 0) {
		if (cacheIndex < i)
			out[cacheIndex] += out[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}
	if(threadIdx.x == 0&&threadIdx.y==0){
		////////////////////Per block
		output[treeNo * gridDim.x + blockIdx.x] = out[0]; //Correct
	}
}


void kernel_wrapper(float* feature, StructSimple* tree, int* nodeSizes, int numberOfInstances, int nbTrees, int numberOfFeatures, int totalNodes){
	//////////////////CUDA Varibles Definition
	int* nodeSizes_cuda;
	StructSimple* tree_cuda;

	float *output[STREAM_COUNT];
	float *feature_stream[STREAM_COUNT];
	float *output_stream[STREAM_COUNT];
	cudaStream_t stream[STREAM_COUNT];

	int OringinalNumberOfInstances = numberOfInstances;
	numberOfInstances = numberOfInstances/STREAM_COUNT;
	int SIZE = numberOfInstances * numberOfFeatures;
	int endSlot = OringinalNumberOfInstances - numberOfInstances * STREAM_COUNT + numberOfInstances;
	int endSize= endSlot * numberOfFeatures;

	/////////////////Timer
	cudaEvent_t start_event, stop_event;
	float time;
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);
	cudaEventRecord(start_event, 0);
	
	/////////////////CUDA Dimension Definision
	dim3 dimBlock(NumberOfThreads, 1);
	int xx = (numberOfInstances + dimBlock.x - 1)/ dimBlock.x;
	int endxx = (endSlot + dimBlock.x - 1)/ dimBlock.x;
	int yy = (nbTrees + dimBlock.y -1)/ dimBlock.y;
	dim3 dimGrid1(endxx, yy);
	dim3 dimGrid(xx, yy);

	//////////////////CUDA Variable Initilization
	int ii = 0;
	for( ii =0; ii<STREAM_COUNT; ++ii ) {			
		cudaStreamCreate(&stream[ii]);
		if(ii == STREAM_COUNT-1){	
			cudaHostAlloc((void **) &output[ii], sizeof(float)* endxx * yy,  cudaHostAllocDefault);
			cudaMalloc((void**)&feature_stream[ii], sizeof(float) * endSize);
			cudaMalloc((void**)&output_stream[ii], sizeof(float) * endxx * yy);		
		}else{
			cudaHostAlloc((void **) &output[ii], sizeof(float)* xx * yy,  cudaHostAllocDefault);
			cudaMalloc((void**)&feature_stream[ii], sizeof(float) * SIZE);
			cudaMalloc((void**)&output_stream[ii], sizeof(float) * xx * yy);		
		}			
	}
	cudaMalloc((void**)&tree_cuda, sizeof(StructSimple) * totalNodes);
	cudaMalloc((void**)&nodeSizes_cuda, sizeof(int) * nbTrees);

	//////////////////////////////////////////
	////////////Memory Move
	//////////////////////////////////////////
	cudaMemcpy(tree_cuda, tree, sizeof(StructSimple) * totalNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(nodeSizes_cuda, nodeSizes, sizeof(int) * nbTrees, cudaMemcpyHostToDevice);	  
	ii = 0;

	///////////////////////////////CUDA WORK Start here 
	///////////////////////////////Using CUDA Stream ver 5
	///////////////////////////////Several streams working concurrently
	for( ii =0; ii<STREAM_COUNT; ii+=2 ){
		int next = ii+1;
		cudaMemcpyAsync(feature_stream[ii], feature+ii*SIZE, sizeof(float)*SIZE, cudaMemcpyHostToDevice, stream[ii]);		
		//////////////////////////The if condition is to deal with the final cuda block special situation.
		if(next == STREAM_COUNT-1){				
			cudaMemcpyAsync(feature_stream[next], feature+next*SIZE, sizeof(float)*endSize, cudaMemcpyHostToDevice, stream[next]);	
		}else{
			cudaMemcpyAsync(feature_stream[next], feature+next*SIZE, sizeof(float)*SIZE, cudaMemcpyHostToDevice, stream[next]);	
		}

		///////////////////////////REAL Work here - call real kernel functions
		scoreAccumulator<<<dimGrid, dimBlock, 0, stream[ii]>>>(tree_cuda, feature_stream[ii], nodeSizes_cuda, numberOfInstances, nbTrees, numberOfFeatures, output_stream[ii]);
		if(next == STREAM_COUNT-1){								
			scoreAccumulator<<<dimGrid1, dimBlock, 0, stream[next]>>>(tree_cuda, feature_stream[next], nodeSizes_cuda, endSlot, nbTrees, numberOfFeatures, output_stream[next]);
		}else{
			scoreAccumulator<<<dimGrid, dimBlock, 0, stream[next]>>>(tree_cuda, feature_stream[next], nodeSizes_cuda, numberOfInstances, nbTrees, numberOfFeatures, output_stream[next]);
		}

		//////////////////////////Error Checking
		cudaError_t error = cudaGetLastError();
		if(error != cudaSuccess)
		{
			// print the CUDA error message and exit if any
			printf("CUDA error second: %s at Iteration: %d\n", cudaGetErrorString(error), ii);
			exit(-1);
		}
		
		cudaMemcpyAsync(output[ii], output_stream[ii], sizeof(float) * xx * yy, cudaMemcpyDeviceToHost, stream[ii]);			  
		if(next == STREAM_COUNT-1){								
			cudaMemcpyAsync(output[next], output_stream[next], sizeof(float) * endxx * yy, cudaMemcpyDeviceToHost, stream[next]);
		}else{
			cudaMemcpyAsync(output[next], output_stream[next], sizeof(float) * xx * yy, cudaMemcpyDeviceToHost, stream[next]);
		}
	}
	/////////////////////////////////Waiting the work to be done in aync manner
	for( ii =0; ii<STREAM_COUNT; ++ii){
		cudaStreamSynchronize( stream[ii] );
	}
	cudaThreadSynchronize();

	/////////////////////////////////Get the CUDA results out from device, get final sum
	double sum =0;
	int tindex = 0;
	for(ii=0; ii < STREAM_COUNT; ii++){
		if(ii == STREAM_COUNT-1){		
			for(tindex = 0; tindex < endxx * yy; tindex++) {
				sum += output[ii][tindex];				
			}	
		}else{
			for(tindex = 0; tindex < xx * yy; tindex++) {
				sum += output[ii][tindex];				
			}
		}
	}
	////////////////////////////////Timer Stop
	cudaEventRecord(stop_event, 0);
	cudaEventSynchronize(stop_event);
	cudaEventElapsedTime(&time, start_event, stop_event);	

	float timeperinstance = time*1000000/(float)OringinalNumberOfInstances ;
	printf ("Total Time is %f ns, and Time/each instance: %f ns\n", time*1000000, timeperinstance);
	printf("Final Score is %.2f\n", sum);

	////////////////////////////////CUDA Memory Deallocation
	for( ii =0; ii<STREAM_COUNT; ++ii){
		cudaFree(feature_stream[ii]); 
		cudaFree(output_stream[ii]); 
		cudaStreamDestroy(stream[ii]);
		cudaFreeHost(output[ii]);
	} 
	cudaFree(tree_cuda);   
	cudaFree(nodeSizes_cuda);   
}
