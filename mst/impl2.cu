#include <vector>
#include <iostream>
#include <algorithm>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "graph.h"
#include "parse_graph.hpp"
#include "limits.h"

/****** START UTIL METHODS ******/
bool edgeSrcComparator(edge a, edge b){ return (a.src < b.src); }
bool edgeDestComparator(edge a, edge b){ return (a.dest < b.dest); }

void swap(void **a, void **b){
    void *tmp = *a;
    *a = *b;
    *b = tmp;
}

int readCudaInt(int *i){
    int tmp;
    cudaMemcpy(&tmp, i, sizeof(int), cudaMemcpyDeviceToHost);
    
    return tmp;
}

void printEdges(std::vector<edge> edges){
    for(edge e : edges){
        printf("src: %d, dst: %d, weight: %d\n", e.src, e.dest, e.weight);
    }
}

int getNumVertices(std::vector<edge> edges){
    int max = -1;
    for(edge e : edges){
        int tmp = std::max(e.src, e.dest);
        max = std::max(max, tmp);
    }

    return max + 1;
}

void writeAnswer(int *output, int len){
    FILE *fp = fopen("output.txt", "w");
    for(int i = 0; i < len; i++){
        fprintf(fp, "%d:\t%d\n", i, output[i]);
    }
    fclose(fp);
}

__global__ void cudaInitIntArray(int *a, int len, int val){
    int totalThreads = gridDim.x * blockDim.x;
    int totalWarps = (totalThreads % 32 == 0) ?  totalThreads / 32 : totalThreads / 32 + 1;
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int warpId = threadId / 32;
    int laneId = threadId % 32;
    int load = (len % totalWarps == 0) ? len / totalWarps : len / totalWarps + 1;
    int beg = load * warpId;
    int end = (len < beg + load) ? len : beg + load;
    beg = beg + laneId;

    for(int i = beg; i < end; i += 32){
        a[i] = val;
    }
}

struct vertex{
  unsigned int start;
  unsigned int len;
};

/****** END UTIL METHODS ******/

void mst(std::vector<edge> * edgesPtr, int blockSize, int blockNum){
    setTime();
    
    // get edge list
    std::vector<edge> edgeVector = *edgesPtr;
    std::sort(edgeVector.begin(), edgeVector.end(), edgeSrcComparator);
    edge *edges = edgeVector.data();
    int elen = edgeVector.size();

    // get vertex list
    vertex *vertices;
    int numVertices = getNumVertices(edgeVector);
    int vlen = numVertices;
    vertices = (vertex*)malloc(numVertices * sizeof(vertex));
    int prevSrc = -1;
    int curVertex, start, len = 0;
    for(edge e : *edgesPtr){
      len++;
      if(prevSrc == -1){
        prevSrc = e.src;
        continue;
      }

      if(prevSrc != e.src){
        vertices[curVertex].start = start;
        vertices[curVertex].len = len;
        
        start += len;
        curVertex++;
        len = 0;
      }
    }
    vertices[curVertex].start = start;
    vertices[curVertex].len = len;

    for(int i = 0; i < vlen; i++){
        printf("v%d: start = %d len = %d %d\n", i, vertices[i].start, vertices[i].len, vlen);
    }
    for(int i = 0; i < elen; i++){
        printf("e%d: src = %d dest = %d weight = %d\n", i, edges[i].src, edges[i].dest, edges[i].weight);
    }

    cudaDeviceProp props; cudaGetDeviceProperties(&props, 0);
    printf("The total computation kernel time on GPU %s is %f milli-seconds\n", props.name, getTime());
}
