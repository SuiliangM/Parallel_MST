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

/** vertex structure
  * vertexID refers to the ID of this vertex
  * start variable refers to the first vertex that is 
  * adjacent to this vertex. 
  * len variable refers to the degree of this vertex
  */

struct Vertex{
  unsigned int vertexID;	
  unsigned int start;
  unsigned int len;
};

/** Component structure
  * It maintains a list of the IDs of the vertices 
  * in this component.
  * It keeps an unsigned int which is the ID of the vertex that is a representative 
  * of this component. Size is the number of 
  * vertices contained in the component 
  */

struct Component{
	unsigned int *vertex_list;
	unsigned int representative;
	int size;
};

__global__ initialize_components(const Vertex* vertex_list, const int numVertices, const Component* components, const int* component_size){
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	int thread_num = blockDim.x * gridDim.x;
	/* the number of total iterations needed for every thread  */
	
	int iter = numVertices % thread_num? numVertices/thread_num + 1 : size/thread_num; 
	
	int i;
	
	for(i = 0; i < iter; i++){
		
		/*the data_id for every thread to write */
		
		int data_id = thread_id + i * thread_num; 
		
		if(data_id < size){
			Vertex v = vertex_list[data_id]; // the vertex that the thread is working on
			components[data_id].vertex_list[0] = v.vertexID;
			components[data_id].representative = v.vertexID;
			components[data_id].size = 1;
		}
	}
}

/****** END UTIL METHODS ******/

void mst(std::vector<edge> * edgesPtr, int blockSize, int blockNum){
    setTime();
    
    // get edge list
    std::vector<edge> edgeVector = *edgesPtr;
    std::sort(edgeVector.begin(), edgeVector.end(), edgeSrcComparator);
    edge *edges = edgeVector.data();
    int elen = edgeVector.size();

    // get vertex list
    Vertex *vertices;
    int numVertices = getNumVertices(edgeVector);
    int vlen = numVertices;
    vertices = (Vertex*)malloc(numVertices * sizeof(Vertex));
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
	
	Vertex* d_vertex_list;
	Component* d_component_list;
	int* component_size; //the total number of components
	
	cudaMalloc((void **)&d_vertex_list, vlen * sizeof(Vertex));
	cudaMalloc((void **)&d_component_list, vlen * sizeof(Component));
	cudaMalloc((void **)&component_size, sizeof(int));
	
	/* copy data from CPU to GPU */

	cudaMemcpy(d_vertex_list, vertices, vlen * sizeof(Vertex), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_component_list, 0, vlen * sizeof(Component), cudaMemcpyDeviceToHost);
	cudaMemset(component_size, vlen, sizeof(int));
	
	/* allocate space for the inner vertex_list of every component object */
	
	for(int i = 0; i < component_size; i++){
		cudaMalloc((void **)&(d_component_list[i].vertex_list), vlen * sizeof(int));
		cudaMemset(d_component_list[i].vertex_list, vlen, sizeof(int));
	}
	
	/* initialize_components initializes the vertices such that every vertex forms a single component */
	
	initialize_components<<<blockNum, blockSize>>>(d_vertex_list, numVertices, d_component_list, component_size); 

	cudaDeviceSynchronize(); // wait until all the threads finish their jobs


    cudaDeviceProp props; cudaGetDeviceProperties(&props, 0);
    printf("The total computation kernel time on GPU %s is %f milli-seconds\n", props.name, getTime());
}