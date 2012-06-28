#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "StructCUDA.h"

extern "C" void kernel_wrapper (float* a, StructSimple* b, int* c, int d, int e, int f, int g);

StructPlus* createNodes(int size) {
	StructPlus* tree = (StructPlus*) calloc(size, sizeof(StructPlus));
	return tree;
}

void destroyTree(StructPlus* tree) {
	free(tree);
	tree = 0;
}

void setRoot(StructPlus* tree, unsigned int id, int featureId, float threshold) {
	tree[0].id = id;
	tree[0].fid = featureId;
	tree[0].threshold = threshold;
	tree[0].left = 0;
	tree[0].right = 0;
}

int addNode(StructPlus* tree, unsigned int pindex, unsigned int id,
	int leftChild, int featureId, float threshold) {
		int index = 2*pindex + 2 - (leftChild != 0);
		tree[index].id = id;
		tree[index].fid = featureId;
		tree[index].threshold = threshold;
		tree[index].left = 0;
		tree[index].right = 0;
		if(leftChild) {
			tree[pindex].left = &tree[index];
		} else {
			tree[pindex].right = &tree[index];
		}
		return index;
}

int main(int argc, char** args) {
	/*if(argc < 3) {
		return -1;
	}

	char* configFile = args[1];
	char* featureFile = args[2];
	*/
	char* configFile = "ensemble-3-1.xml.tree.end";
	char* featureFile = "test.txt";

	////////////////////////////////////////////
	// build DecisionTree
	////////////////////////////////////////////
	FILE *fp = fopen(configFile, "r");
	int nbTrees;
	fscanf(fp, "%d", &nbTrees);

	int totalNodes = 0;
	int* nodeSizes;
	cudaHostAlloc((void **) &nodeSizes, sizeof(int)*nbTrees, cudaHostAllocDefault);
	//int* nodeSizes = (int*) malloc(nbTrees * sizeof(int));
	StructPlus** trees = (StructPlus**) malloc(nbTrees * sizeof(StructPlus*));
	printf("Starting Tree Reading....\n");
	int tindex = 0;
	for(tindex = 0; tindex < nbTrees; tindex++) {
		int treeSize;
		fscanf(fp, "%d", &treeSize);
		int internalSize = pow(2.0, treeSize) - 1;
		int fullSize = 2* pow(2.0, treeSize) - 1;
		nodeSizes[tindex] = fullSize;
		totalNodes += fullSize;
		int* pointers = (int*) malloc(internalSize * sizeof(int));
		trees[tindex] = createNodes(fullSize);

		char text[20];
		int line = 0;
		for(line = 0; line < internalSize; line++) pointers[line] = -1;
		fscanf(fp, "%s", text);
		while(strcmp(text, "end") != 0) {
			int id;
			fscanf(fp, "%d", &id);

			if(strcmp(text, "root") == 0) {
				int fid;
				float threshold;
				fscanf(fp, "%d %f", &fid, &threshold);
				setRoot(trees[tindex], id, fid, threshold);
				pointers[id] = 0;
			} else if(strcmp(text, "node") == 0) {
				int fid;
				int pid;
				float threshold;
				int leftChild = 0;
				fscanf(fp, "%d %d %d %f", &pid, &fid, &leftChild, &threshold);
				if(pointers[pid] >= 0 && trees[tindex][pointers[pid]].fid >= 0) {
					pointers[id] = addNode(trees[tindex], pointers[pid], id, leftChild, fid, threshold);
				}
			} else if(strcmp(text, "leaf") == 0) {
				int pid;
				int leftChild = 0;
				float value;
				fscanf(fp, "%d %d %f", &pid, &leftChild, &value);
				if(pointers[pid] >= 0 && trees[tindex][pointers[pid]].fid >= 0) {
					addNode(trees[tindex], pointers[pid], id, leftChild, -1, value);
				}
			}
			fscanf(fp, "%s", text);
		}
		free(pointers);
	}
	fclose(fp);		

	// Pack all trees into a single array, thus avoiding two-D arrays.
	printf("Starting Rearrange the Tree....\n");
	//StructSimple* all_nodes = (StructSimple*) malloc(totalNodes * sizeof(StructSimple));
	StructSimple* all_nodes = NULL;
	cudaHostAlloc((void **) &all_nodes, sizeof(StructSimple)*totalNodes, cudaHostAllocDefault);
	int newIndex = 0;

	for(tindex = 0; tindex < nbTrees; tindex++) {
		int nsize = nodeSizes[tindex];
		nodeSizes[tindex] = newIndex;
		int telement;
		//printf("Size of the tree is %d\n", nsize);
		for(telement = 0; telement < nsize; telement++) {
			printf("tindex %d telement %d - FID %d Threshold %f\n",tindex, telement, trees[tindex][telement].fid,trees[tindex][telement].threshold);
			if(telement == 0){
				all_nodes[newIndex].fid = abs(trees[tindex][telement].fid);
				all_nodes[newIndex].threshold = trees[tindex][telement].threshold;
				all_nodes[newIndex].leaf = (!trees[tindex][telement].left && !trees[tindex][telement].right)?'y':'n';
			}else if(trees[tindex][telement].fid && trees[tindex][telement].id){
				all_nodes[newIndex].fid = trees[tindex][telement].fid;
				all_nodes[newIndex].threshold = trees[tindex][telement].threshold;
				all_nodes[newIndex].leaf = (!trees[tindex][telement].left && !trees[tindex][telement].right)?'y':'n';
			}else{
				all_nodes[newIndex].fid = NULL;
				all_nodes[newIndex].threshold = NULL;
				all_nodes[newIndex].leaf = NULL;
			}
			//printf("---fid=%d, threshold=%f, left=%d, right=%d\n", trees[tindex][telement].fid, trees[tindex][telement].threshold, trees[tindex][telement].left, trees[tindex][telement].right);
			//printf("fid=%d, threshold=%f, leaf=%c\n", all_nodes[newIndex].fid, all_nodes[newIndex].threshold, all_nodes[newIndex].leaf);
			newIndex++;
		}		
	}

	///////////////////////////////////////////////////////////
	///////////FEATURES FILES READING//////////////////////////////
	//////////////////////////////////////////////////////////
	printf("Reading Feature File....\n");
	int numberOfFeatures = 0;
	int numberOfInstances = 0;
	fp = fopen(featureFile, "r");
	fscanf(fp, "%d %d", &numberOfInstances, &numberOfFeatures);

	///New Code On Feature Array
	float* features = NULL;
	cudaHostAlloc((void **) &features, sizeof(float)*numberOfFeatures * numberOfInstances,  cudaHostAllocDefault);
	//float* features = (float*) malloc(numberOfFeatures * numberOfInstances * sizeof(float));
	float fvalue;
	int fIndex = 0, iIndex = 0;
	int ignore;
	char text[20];
	for(iIndex = 0; iIndex < numberOfInstances; iIndex++) {
		fscanf(fp, "%d %[^:]:%d", &ignore, text, &ignore);
		for(fIndex = 0; fIndex < numberOfFeatures; fIndex++) {
			fscanf(fp, "%[^:]:%f", text, &fvalue);
			features[iIndex*numberOfFeatures+fIndex] = fvalue;	
		}
	}

	///////////////////////////////////////////////
	/////////////TIMER
	//////////////////////////////////////////////
	float time;
	cudaEvent_t start_event, stop_event;
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);
	cudaEventRecord(start_event, 0);

	///////////////////KERNEL////////////////////////
	kernel_wrapper(features, all_nodes, nodeSizes, numberOfInstances, nbTrees, numberOfFeatures, totalNodes);
	//////////////////////////////////////////////////
	
	cudaEventRecord(stop_event, 0);
	cudaEventSynchronize(stop_event);
	cudaEventElapsedTime(&time, start_event, stop_event);
	float timeperinstance = time*1000000/(float)numberOfInstances;
	printf ("Outside Total Time is %f ns, and Time/each instance: %f ns\n", time*1000000, timeperinstance);
	
	cudaFreeHost(nodeSizes);
	cudaFreeHost(all_nodes);
	cudaFreeHost(features);
	free(trees);
	fclose(fp);
	return 0;
}
