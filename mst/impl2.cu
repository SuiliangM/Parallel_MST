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

    // vertices will be numbered starting from 0 or 1
    bool verticesStartFromZero = false;

    for(edge e : edges){
        int tmp = std::max(e.src, e.dest);
        max = std::max(max, tmp);

        if(e.src == 0){
            verticesStartFromZero = true;
        }
    }

    return verticesStartFromZero ? max + 1 : max;
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
  int start;
  unsigned int len;
  int successor;
};

/****** END UTIL METHODS ******/

bool done(vertex *v, int vlen){
    vertex tmp;
    cudaMemcpy(&tmp, v, sizeof(vertex), cudaMemcpyDeviceToHost);
    int successor = tmp.successor;
    for(int i = 1; i < vlen; i++){
        cudaMemcpy(&tmp, &v[i], sizeof(vertex), cudaMemcpyDeviceToHost);
        if(successor != tmp.successor){
            return false;
        }
    }

    return true;
}

__global__ void
findMins(vertex *v, edge *e, int *inMst, int vlen, int elen){
    int totalThreads = gridDim.x * blockDim.x;
    int totalWarps = (totalThreads % 32 == 0) ?  totalThreads / 32 : totalThreads / 32 + 1;
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int warpId = threadId / 32;
    int laneId = threadId % 32;
    int load = (vlen % totalWarps == 0) ? vlen / totalWarps : vlen / totalWarps + 1;
    int beg = load * warpId;
    int end = (vlen < beg + load) ? vlen : beg + load;
    beg = beg + laneId;

    for(int i = beg; i < end; i += 32){
        vertex cur = v[i];
        int min = INT_MAX;
        int minIndex = elen;
        for(int j = cur.start; j < (cur.start + cur.len); j++){
            int w = e[j].weight;
            cur.successor = ((min <= w) * cur.successor) + ((w < min) * e[j].dest);
            min = ((min <= w) * min) + ((w < min) * w);
            minIndex = ((min <= w) * minIndex) + ((w < min) * j);
        }
        inMst[minIndex] = 1;
        e[minIndex].weight = INT_MAX;
    }
}

__global__ void
setSuccessors(vertex *v, int vlen){
    int totalThreads = gridDim.x * blockDim.x;
    int totalWarps = (totalThreads % 32 == 0) ?  totalThreads / 32 : totalThreads / 32 + 1;
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int warpId = threadId / 32;
    int laneId = threadId % 32;
    int load = (vlen % totalWarps == 0) ? vlen / totalWarps : vlen / totalWarps + 1;
    int beg = load * warpId;
    int end = (vlen < beg + load) ? vlen : beg + load;
    beg = beg + laneId;

    for(int i = beg; i < end; i += 32){
        vertex cur = v[i];
        while(v[cur.successor].successor != cur.successor){
            cur.successor = v[cur.successor].successor;
        }
    }
}

void mst(std::vector<edge> * edgesPtr, int blockSize, int blockNum){
    setTime();
    
    // get edge list
    std::vector<edge> edgeVector = *edgesPtr;
    std::sort(edgeVector.begin(), edgeVector.end(), edgeSrcComparator);
    edge *edges = edgeVector.data();
    int elen = edgeVector.size();

    // get vertex list
    vertex *vertices;
    int vlen = getNumVertices(edgeVector);
    vertices = (vertex*)malloc(vlen * sizeof(vertex));
    for(int i = 0; i < vlen; i++){
        vertices[i].start = INT_MAX;
        vertices[i].successor = i;
    }
    int prevSrc = -1;
    int curVertex = 0;
    int start = 0;
    int len = 0;
    for(edge e : *edgesPtr){
      if(prevSrc == -1){
        curVertex = e.src;
        prevSrc = e.src;
      }

      if(prevSrc != e.src){
        vertices[curVertex].start = start;
        vertices[curVertex].len = len;
        
        start += len;
        curVertex = e.src;
        prevSrc = e.src;
        len = 0;
      }

      len++;
    }
    vertices[curVertex].start = start;
    vertices[curVertex].len = len;

    for(int i = 0; i < vlen; i++){
        printf("v%d: start = %d len = %d %d\n", i, vertices[i].start, vertices[i].len, vlen);
    }
    for(int i = 0; i < elen; i++){
        printf("e%d: src = %d dest = %d weight = %d\n", i, edges[i].src, edges[i].dest, edges[i].weight);
    }

    int *inMst;
    cudaMalloc((void**)&inMst, sizeof(int) * elen);
    cudaInitIntArray<<<blockNum, blockSize>>>(inMst, elen, 0);

    edge *e; 
    cudaMalloc((void**)&e, sizeof(edge) * elen);
    cudaMemcpy(e, edgeVector.data(), elen * sizeof(edge), cudaMemcpyHostToDevice);

    vertex *v;
    cudaMalloc((void**)&v, sizeof(vertex) * vlen);
    cudaMemcpy(v, vertices, vlen * sizeof(vertex), cudaMemcpyHostToDevice);

    while(!done(v, vlen)){
        findMins<<<blockSize, blockNum>>>(v, e, inMst, vlen, elen);
        setSuccessors<<<blockSize, blockNum>>>(v, vlen);
    }

    for(int i = 0; i < elen; i++){
        if(readCudaInt(&inMst[i]) == 1){
            edge tmp = edges[i];
            printf("IN MST: %d\t%d\n", tmp.src, tmp.dest);
        }
    }

    cudaDeviceProp props; cudaGetDeviceProperties(&props, 0);
    printf("The total computation kernel time on GPU %s is %f milli-seconds\n", props.name, getTime());
}
