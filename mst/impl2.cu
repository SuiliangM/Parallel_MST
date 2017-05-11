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
        int minv = INT_MAX;
        for(int j = cur.start; j < (cur.start + cur.len); j++){
            int w = e[j].weight;
            int tmpMin = min; int tmpInd = minIndex;
            int dest = e[j].dest;
            min = ( ((min < w) || ((min == w) && (minv < dest))) * min) + ( (((w == min) && (dest < minv)) || (w < min)) * w);
            minIndex = ( ((tmpMin < w) || ((tmpMin == w) && (minv < dest))) * minIndex) + ( (((w == tmpMin) && (dest < minv)) || (w < tmpMin)) * j);
            minv = ( ((tmpMin < w) || ((tmpMin == w) && (minv < dest))) * minv) + ( (((w == tmpMin) && (dest < minv)) || (w < tmpMin)) * dest);
            //printf("minv: %d, cursuc: %d\n", minv, cur.successor);

            //minIndex = ((tmpMin <= w) * minIndex) + (((w < tmpMin) && (j < minIndex)) * j);
            //printf("min: %d\tnew: %d\t%d\n", tmpMin, w, (int)(((w == tmpMin) && (dest < minv)) || (w < tmpMin)));
        }
        inMst[minIndex] = 1;
        //printf("min index: %d", minIndex);
        e[minIndex].weight = INT_MAX;
        v[i].successor = ((minv != INT_MAX) * minv) + ((minv == INT_MAX) * cur.successor);
        printf("cur: %d, suc: %d\n", i, v[i].successor);
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
        while(v[v[i].successor].successor != v[i].successor){
            v[i].successor = v[v[i].successor].successor;
        }
    }
}

__global__ void
fixSuccessors(vertex *v, int vlen){
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
        bool shouldChange = (v[cur.successor].successor == i) && (i < cur.successor);
        printf("me: %d, suc: %d, sucsuc: %d\n", i, cur.successor, v[cur.successor].successor);
        v[i].successor = ((int)shouldChange * i) + (((int)!shouldChange) * cur.successor);
        //printf("cur: %d, suc: %d\n", i, cur.successor);
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
    for(int i = 0; i < edgeVector.size(); i++){
        edge e = edgeVector[i];
      if(prevSrc == -1){
        curVertex = e.src;
        prevSrc = e.src;
      }

      if(prevSrc != e.src){
        vertices[curVertex].start = start;
        //printf("cur: %d, start: %d", curVertex, vertices[curVertex].start);
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

    int stop = 0;
    while(!done(v, vlen)){
        findMins<<<blockSize, blockNum>>>(v, e, inMst, vlen, elen);

        /*for(int i = 0; i < elen; i++){
            printf("HERE %d:\t%d\n", i, readCudaInt(&inMst[i]));
        }*/
        for(int i = 0; i < vlen; i++){
            vertex tmp;
            cudaMemcpy(&tmp, &v[i], sizeof(vertex), cudaMemcpyDeviceToHost);
            printf("HERE %d:\t%d\n", i, tmp.successor);
        }
        fixSuccessors<<<blockSize, blockNum>>>(v, vlen);
        cudaDeviceSynchronize();
        setSuccessors<<<blockSize, blockNum>>>(v, vlen);
        for(int i = 0; i < vlen; i++){
            vertex tmp;
            cudaMemcpy(&tmp, &v[i], sizeof(vertex), cudaMemcpyDeviceToHost);
            printf("HERE %d:\t%d\n", i, tmp.successor);
        }

        if(stop >= 2){
            break;
        }
        stop++;

        for(int i = 0; i < elen; i++){
            if(readCudaInt(&inMst[i]) == 1){
                edge tmp = edges[i];
                printf("IN MST: %d\t%d\n", tmp.src, tmp.dest);
            }
        }
    }

    cudaDeviceProp props; cudaGetDeviceProperties(&props, 0);
    printf("The total computation kernel time on GPU %s is %f milli-seconds\n", props.name, getTime());
}
