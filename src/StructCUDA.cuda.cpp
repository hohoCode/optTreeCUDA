#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "StructCUDA.h"

extern "C" void kernel_wrapper (float* a, StructSimple* b, int* c, int d, int e, int f, int g);

StructPlus* createNode(unsigned long id, int fid, float threshold) {
	StructPlus* node = (StructPlus*) malloc(sizeof(StructPlus));
	node->id = id;
	node->fid = fid;
	node->threshold = threshold;
	node->left = 0;
	node->right = 0;
	return node;
}

void destroyTree(StructPlus* node) {
	if(node->right != 0) {
		destroyTree(node->right);
		free(node->right);
		node->right = 0;
	}
	if(node->left != 0) {
		destroyTree(node->left);
		free(node->left);
		node->left = 0;
	}
}

StructPlus* addNode(StructPlus* node, unsigned long id, int leftChild, int featureId, float threshold) {
	if(leftChild) {
		node->left = createNode(id, featureId, threshold);
		return node->left;
	} else {
		node->right = createNode(id, featureId, threshold);
		return node->right;
	}
}

long compressNodes(StructSimple* array, StructPlus* old, int index) {
  array[index].fid = old->fid;
  array[index].threshold = old->threshold;

  if(old->right || old->left) {
    int pindex = index;
    index = compressNodes(array, old->left, index + 1);
    array[pindex].left = pindex + 1;

    array[pindex].right = index + 1;
    index = compressNodes(array, old->right, index + 1);
  } else {
    array[index].right = 0;
    array[index].left = 0;
  }
  return index;
}

StructSimple* compress(StructPlus* root, int size) {
  StructSimple* tree = (StructSimple*) calloc(size, sizeof(StructSimple));
  long index = compressNodes(tree, root, 0);
  return tree;
}

int main(int argc, char** args) {	
	if(argc < 3) {
		return -1;
	}

	char* configFile = args[1];
	char* featureFile = args[2];
	
	////////////////////////////////////////////
	// Read DecisionTree
	////////////////////////////////////////////
	
	FILE *fp = fopen(configFile, "r");
	int nbTrees;
	fscanf(fp, "%d", &nbTrees);
	int totalNodes = 0;
	int* nodeSizes = (int*) malloc(nbTrees * sizeof(int));
	StructPlus** trees = (StructPlus**) malloc(nbTrees * sizeof(StructPlus*));
	StructSimple** treeArray = (StructSimple**) malloc(nbTrees * sizeof(StructSimple*));

	int tindex = 0;
	for(tindex = 0; tindex < nbTrees; tindex++) {
		long treeSize;
		fscanf(fp, "%ld", &treeSize);
		long internalSize = pow(2.0, treeSize) - 1;	
		StructPlus** pointers = (StructPlus**) malloc(internalSize * sizeof(StructPlus*));

		char text[20];
		long line = 0;
		for(line = 0; line < internalSize; line++) pointers[line] = 0;
		fscanf(fp, "%s", text);
		int realTreeSize = 0;
		while(strcmp(text, "end") != 0) {
			long id;
			fscanf(fp, "%ld", &id);

			if(strcmp(text, "root") == 0) {
				int fid;
				float threshold;
				fscanf(fp, "%d %f", &fid, &threshold);
				trees[tindex] = createNode(id, fid, threshold);
				pointers[id] = trees[tindex];
				realTreeSize ++;
			} else if(strcmp(text, "node") == 0) {
				int fid;
				long pid;
				float threshold;
				int leftChild = 0;
				fscanf(fp, "%ld %d %d %f", &pid, &fid, &leftChild, &threshold);
				if((pointers[pid] != 0) && pointers[pid]->fid >= 0) {
					realTreeSize ++;
					pointers[id] = addNode(pointers[pid], id, leftChild, fid, threshold);
				}
			} else if(strcmp(text, "leaf") == 0) {
				long pid;
				int leftChild = 0;
				float value;
				fscanf(fp, "%ld %d %f", &pid, &leftChild, &value);
				if((pointers[pid] != 0) && pointers[pid]->fid >= 0) {
					realTreeSize ++;
					addNode(pointers[pid], id, leftChild, 0, value);
				}
			}
			fscanf(fp, "%s", text);
		}
		free(pointers);
		///Convert Pointer Clusters to An Array
		treeArray[tindex] = compress(trees[tindex], realTreeSize);
		nodeSizes[tindex] = realTreeSize;
		totalNodes += realTreeSize;
	}

	fclose(fp);

	// Pack all trees into a single array.
	StructSimple* all_nodes = NULL;
	cudaHostAlloc((void **) &all_nodes, sizeof(StructSimple)*totalNodes, cudaHostAllocDefault);
	int newIndex = 0;
	for(tindex = 0; tindex < nbTrees; tindex++) {
		int nsize = nodeSizes[tindex];
		nodeSizes[tindex] = newIndex;		
		for(int telement = 0; telement < nsize; telement++) {
			all_nodes[newIndex].fid = abs(treeArray[tindex][telement].fid);
			all_nodes[newIndex].threshold = treeArray[tindex][telement].threshold;
			all_nodes[newIndex].left = treeArray[tindex][telement].left;
			all_nodes[newIndex].right = treeArray[tindex][telement].right;			
			newIndex++;
		}
		destroyTree(trees[tindex]);
	}

	///////////////////////////////////////////////////////////
	///////////FEATURES FILES READING//////////////////////////////
	//////////////////////////////////////////////////////////
	int numberOfFeatures = 0;
	int numberOfInstances = 0;
	fp = fopen(featureFile, "r");
	fscanf(fp, "%d %d", &numberOfInstances, &numberOfFeatures);

	///New Code On Feature Array
	float* features = NULL;
	cudaHostAlloc((void **) &features, sizeof(float)*numberOfFeatures * numberOfInstances, cudaHostAllocWriteCombined);
	
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

	///////////////////KERNEL////////////////////////
	kernel_wrapper(features, all_nodes, nodeSizes, numberOfInstances, nbTrees, numberOfFeatures, totalNodes);
	//////////////////////////////////////////////////
	
	cudaFreeHost(all_nodes);
	cudaFreeHost(features);
	free(treeArray);
	fclose(fp);
	return 0;
}

