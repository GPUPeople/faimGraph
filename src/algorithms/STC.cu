//------------------------------------------------------------------------------
// STC.cu
//
// Masterproject/-thesis aimGraph
//
// Authors: Martin Winter, 1130688
//------------------------------------------------------------------------------
//
#include <iostream>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "STC.h"
#include "MemoryManager.h"

//------------------------------------------------------------------------------
//
__device__ void conditionalWarpReduceIP(volatile int32_t* sharedData,int blockSize,int dataLength)
{
  if(blockSize >= dataLength)
  {
    if(threadIdx.x < (dataLength/2))
    {
      sharedData[threadIdx.x] += sharedData[threadIdx.x+(dataLength/2)];
    }
    __syncthreads();
  }
}

//------------------------------------------------------------------------------
//
__device__ void warpReduceIP(int32_t* __restrict__ outDataPtr, volatile int32_t* __restrict__ sharedData, int blockSize)
{
  conditionalWarpReduceIP(sharedData,blockSize,64);
  conditionalWarpReduceIP(sharedData,blockSize,32);
  conditionalWarpReduceIP(sharedData,blockSize,16);
  conditionalWarpReduceIP(sharedData,blockSize,8);
  conditionalWarpReduceIP(sharedData,blockSize,4);
  if(threadIdx.x == 0)
  {
    *outDataPtr= sharedData[0] + sharedData[1];
  }
  __syncthreads();
}

//------------------------------------------------------------------------------
//
__device__ void conditionalReduceIP(volatile int32_t* __restrict__ sharedData,int blockSize,int dataLength)
{
	if(blockSize >= dataLength)
  {
		if(threadIdx.x < (dataLength/2))
		{
      sharedData[threadIdx.x] += sharedData[threadIdx.x+(dataLength/2)];
    }
		__syncthreads();
	}
	if((blockSize < dataLength) && (blockSize > (dataLength/2)))
  {
		if(threadIdx.x+(dataLength/2) < blockSize)
    {
			sharedData[threadIdx.x] += sharedData[threadIdx.x+(dataLength/2)];
		}
		__syncthreads();
	}
}

//------------------------------------------------------------------------------
//
__device__ void blockReduceIP(int32_t* __restrict__ outGlobalDataPtr, volatile int32_t* __restrict__ sharedData, int blockSize){
  __syncthreads();
  conditionalReduceIP(sharedData,blockSize,1024);
  conditionalReduceIP(sharedData,blockSize,512);
  conditionalReduceIP(sharedData,blockSize,256);
  conditionalReduceIP(sharedData,blockSize,128);

  warpReduceIP(outGlobalDataPtr, sharedData, blockSize);
  __syncthreads();
}

//------------------------------------------------------------------------------
//
__device__ void initializeIP(const int32_t diag_id, const int32_t u_len, int32_t v_len,
    int32_t* const __restrict__ u_min, int32_t* const __restrict__ u_max,
    int32_t* const __restrict__ v_min, int32_t* const __restrict__ v_max,
    int* const __restrict__ found)
{
	if (diag_id == 0)
  {
		*u_min=*u_max=*v_min=*v_max=0;
		*found=1;
	}
	else if (diag_id < u_len)
  {
		*u_min=0; *u_max=diag_id;
		*v_max=diag_id;*v_min=0;
	}
	else if (diag_id < v_len)
  {
		*u_min=0; *u_max=u_len;
		*v_max=diag_id;*v_min=diag_id-u_len;
	}
	else
  {
		*u_min=diag_id-v_len; *u_max=u_len;
		*v_min=diag_id-u_len; *v_max=v_len;
	}
}

//------------------------------------------------------------------------------
//
__device__ void workPerThreadIP(const int32_t uLength, const int32_t vLength, 
	const int threadsPerIntersection, const int threadId,
    int * const __restrict__ outWorkPerThread, int * const __restrict__ outDiagonalId)
{
  int totalWork = uLength + vLength;
  int remainderWork = totalWork % threadsPerIntersection;
  int workPerThread = totalWork / threadsPerIntersection;

  int longDiagonals  = (threadId > remainderWork) ? remainderWork : threadId;
  int shortDiagonals = (threadId > remainderWork) ? (threadId - remainderWork) : 0;

  *outDiagonalId = ((workPerThread + 1) * longDiagonals) + (workPerThread * shortDiagonals);
  *outWorkPerThread = workPerThread + (threadId < remainderWork);
}

//------------------------------------------------------------------------------
//
template <typename EdgeDataType>
__device__ void bSearchIP(unsigned int found, const int32_t diagonalId,
  AdjacencyIterator<EdgeDataType>& uNodes, AdjacencyIterator<EdgeDataType>& vNodes,
    int32_t const * const __restrict__ uLength, 
    int32_t * const __restrict__ outUMin, int32_t * const __restrict__ outUMax,
    int32_t * const __restrict__ outVMin, int32_t * const __restrict__ outVMax,    
    int32_t * const __restrict__ outUCurr,
    int32_t * const __restrict__ outVCurr, memory_t* memory, const int page_size, int start_index, vertex_t edges_per_block)
{
  int32_t length;
	
	while(!found) 
  {
	    *outUCurr = (*outUMin + *outUMax)>>1;
	    *outVCurr = diagonalId - *outUCurr;
	    if(*outVCurr >= *outVMax)
      {
			  length = *outUMax - *outUMin;
        if(length == 1)
        {
          found = 1;
          continue;
        }
	    }

	    unsigned int comp1 = uNodes.at(*outUCurr, memory, page_size, start_index, edges_per_block) > vNodes.at(*outVCurr-1, memory, page_size, start_index, edges_per_block);
	    unsigned int comp2 = uNodes.at(*outUCurr-1, memory, page_size, start_index, edges_per_block) > vNodes.at(*outVCurr, memory, page_size, start_index, edges_per_block);
	    if(comp1 && !comp2)
      {
			  found = 1;
	    }
	    else if(comp1)
      {
	      *outVMin = *outVCurr;
	      *outUMax = *outUCurr;
	    }
	    else
      {
	      *outVMax = *outVCurr;
	      *outUMin = *outUCurr;
	    }
  	}

	if((*outVCurr >= *outVMax) && (length == 1) && (*outVCurr > 0) &&
	(*outUCurr > 0) && (*outUCurr < (*uLength - 1)))
  {
		unsigned int comp1 = uNodes.at(*outUCurr, memory, page_size, start_index, edges_per_block) > vNodes.at(*outVCurr - 1, memory, page_size, start_index, edges_per_block);
		unsigned int comp2 = uNodes.at(*outUCurr - 1, memory, page_size, start_index, edges_per_block) > vNodes.at(*outVCurr, memory, page_size, start_index, edges_per_block);
		if(!comp1 && !comp2)
    {
      (*outUCurr)++; 
      (*outVCurr)--;
    }
	}
}

//------------------------------------------------------------------------------
//
template <typename EdgeDataType>
__device__ int fixStartPointIP(const int32_t uLength, const int32_t vLength,
    int32_t * const __restrict__ uCurr, int32_t * const __restrict__ vCurr,
    AdjacencyIterator<EdgeDataType>& uNodes, AdjacencyIterator<EdgeDataType>& vNodes, memory_t* memory, const int page_size, int start_index, vertex_t edges_per_block)
{
	unsigned int uBigger = (*uCurr > 0) && (*vCurr < vLength) && (uNodes.at(*uCurr-1, memory, page_size, start_index, edges_per_block) == vNodes.at(*vCurr, memory, page_size, start_index, edges_per_block));
	unsigned int vBigger = (*vCurr > 0) && (*uCurr < uLength) && (vNodes.at(*vCurr-1, memory, page_size, start_index, edges_per_block) == uNodes.at(*uCurr, memory, page_size, start_index, edges_per_block));
	*uCurr += vBigger;
	*vCurr += uBigger;
	return (uBigger + vBigger);
}

//------------------------------------------------------------------------------
//
template <typename EdgeDataType>
__device__ void intersectPath(const int32_t uLength, const int32_t vLength,
    AdjacencyIterator<EdgeDataType>& uNodes, AdjacencyIterator<EdgeDataType>& vNodes,
    int32_t * const __restrict__ uCurr, int32_t * const __restrict__ vCurr,
    int * const __restrict__ workIndex, int * const __restrict__ workPerThread,
    int * const __restrict__ triangles, int found, memory_t* memory, const int page_size, int start_index, vertex_t edges_per_block)
{
  if((*uCurr < uLength) && (*vCurr < vLength))
  {
    int comp;
    while(*workIndex < *workPerThread)
    {
      comp = uNodes.at(*uCurr, memory, page_size, start_index, edges_per_block) - vNodes.at(*vCurr, memory, page_size, start_index, edges_per_block);
      *triangles += (comp == 0);
      *uCurr += (comp <= 0);
      *vCurr += (comp >= 0);
      *workIndex += (comp == 0) + 1;

      if((*vCurr == vLength) || (*uCurr == uLength))
      {
        break;
      }
    }
    *triangles -= ((comp == 0) && (*workIndex > *workPerThread) && (found));
  }
}

//------------------------------------------------------------------------------
//
template <typename EdgeDataType>
__device__ int32_t singleIntersection(int32_t u, AdjacencyIterator<EdgeDataType>& u_nodes, int32_t u_len,
    int32_t v, AdjacencyIterator<EdgeDataType>& v_nodes, int32_t v_len, int threads_per_block,
    volatile int32_t* __restrict__ firstFound, int tId, memory_t* memory, const int page_size, int start_index, vertex_t edges_per_block)
{
	// Partitioning the work to the multiple thread of a single GPU processor. 
  // The threads should get a near equal number of the elements to intersect 
  // - this number will be off by 1.
	int work_per_thread, diag_id;
	workPerThreadIP(u_len, v_len, threads_per_block, tId, &work_per_thread, &diag_id);
	int triangles = 0;
	int work_index = 0, found = 0;
	int32_t u_min, u_max, v_min, v_max, u_curr, v_curr;

	firstFound[tId] = 0;

	if(work_per_thread > 0)
  {
		// For the binary search, we are figuring out the initial poT of search.
		initializeIP(diag_id, u_len, v_len, &u_min, &u_max, &v_min, &v_max, &found);
    u_curr = 0; v_curr = 0;
	  
    bSearchIP(found, diag_id, u_nodes, v_nodes, &u_len, &u_min, &u_max, &v_min, &v_max, &u_curr, &v_curr, memory, page_size, start_index, edges_per_block);

    int sum = fixStartPointIP(u_len, v_len, &u_curr, &v_curr, u_nodes, v_nodes, memory, page_size, start_index, edges_per_block);
    work_index += sum;
	  if(tId > 0)
    {
	    firstFound[tId-1] = sum;
    }
	  triangles += sum;
	  intersectPath(u_len, v_len, u_nodes, v_nodes, &u_curr, &v_curr, &work_index, &work_per_thread, &triangles, firstFound[tId], memory, page_size, start_index, edges_per_block);
	}
	return triangles;
}

//------------------------------------------------------------------------------
//
__device__ void workPerBlockIP(const int32_t numVertices,
                               int32_t* const __restrict__ outMpStart,
                               int32_t* const __restrict__ outMpEnd, 
                               int blockSize)
{
	int32_t verticesPerMp = numVertices / gridDim.x;
	int32_t remainderBlocks = numVertices % gridDim.x;
	int32_t extraVertexBlocks = (blockIdx.x > remainderBlocks) ? remainderBlocks : blockIdx.x;
	int32_t regularVertexBlocks = (blockIdx.x > remainderBlocks) ? (blockIdx.x - remainderBlocks) : 0;

	int32_t mpStart = ((verticesPerMp + 1) * extraVertexBlocks) + (verticesPerMp * regularVertexBlocks);
	*outMpStart = mpStart;
	*outMpEnd = mpStart + verticesPerMp + (blockIdx.x < remainderBlocks);
}

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType>
__global__ void d_balancedTriangleCount(MemoryManager* memory_manager,
                                        memory_t* memory,
                                        int32_t* const __restrict__ triangles,
                                        const int number_vertices,
                                        const int index_shift,
                                        const int page_size,
                                        const int threads_per_block,
                                        const int number_blocks, 
                                        const int shifter)
{
  VertexDataType* vertices = reinterpret_cast<VertexDataType*>(memory);

  int32_t this_multiproc_start, this_multipproc_stop;
  const int blockSize = blockDim.x;

  // Partition the work to the multiple threads on a MP, each thread should receive a near equal amount of work
  workPerBlockIP(number_vertices, &this_multiproc_start, &this_multipproc_stop, blockSize);

  __shared__ int32_t  s_triangles[1024];
	__shared__ int32_t firstFound[1024];

	int32_t adjacency_offset = threadIdx.x >> shifter;
	int32_t* firstFoundPos = firstFound + (adjacency_offset << shifter);
  vertex_t edges_per_page = memory_manager->edges_per_page;
  int start_index = memory_manager->start_index;


  for (int32_t src = this_multiproc_start; src < this_multipproc_stop; src++)
	{
    int32_t triangle_count = 0;
    AdjacencyIterator<EdgeDataType> src_adjacency_iterator(pageAccess<EdgeDataType>(memory, vertices[src].mem_index, page_size, start_index));
    VertexDataType src_vertex = vertices[src];

    for(int k = adjacency_offset; k < src_vertex.neighbours; k += number_blocks)
		{
      int dest = src_adjacency_iterator.getDestinationAt(k);
      VertexDataType dst_vertex = vertices[dest];

      // Check if calculation is necessary
			if((src == dest) || (dst_vertex.neighbours < 2) || (src_vertex.neighbours < 2))
				continue;

      // Check which vertex has the smaller adjacency
      int32_t small, large, small_len, large_len;
      if (src_vertex.neighbours < dst_vertex.neighbours)
      {
        small = src;
        large = dest;
        small_len = src_vertex.neighbours;
        large_len = dst_vertex.neighbours;
      }
      else
      {
        small = dest;
        large = src;
        small_len = dst_vertex.neighbours;
        large_len = src_vertex.neighbours;
      }
      
      AdjacencyIterator<EdgeDataType> small_adjacency_iterator (pageAccess<EdgeDataType>(memory, vertices[small].mem_index, page_size, start_index));
      AdjacencyIterator<EdgeDataType> large_adjacency_iterator (pageAccess<EdgeDataType>(memory, vertices[large].mem_index, page_size, start_index));

      // Intersect both adjacencies
      triangle_count += singleIntersection(small, 
                                           small_adjacency_iterator,
                                           small_len,
				                                   large,
                                           large_adjacency_iterator,
                                           large_len,
				                                   threads_per_block,
                                           firstFoundPos,
                                           threadIdx.x % threads_per_block,
                                           memory,
                                           page_size,
                                           start_index,
                                           edges_per_page);
    }
    s_triangles[threadIdx.x] = triangle_count;
		blockReduceIP(&triangles[src],s_triangles,blockSize);
  }

  return;
}







template <typename VertexDataType, typename EdgeDataType>
std::unique_ptr<int32_t> workBalancedSTC(const std::unique_ptr<MemoryManager>& memory_manager,
                                         const int threads_per_block, 
                                         const int number_blocks, 
                                         const int shifter, 
                                         const int thread_blocks, 
                                         const int blockdim)
{
  std::unique_ptr<int32_t> triangles (new int32_t[memory_manager->number_vertices]);
  // Setup memory
  TemporaryMemoryAccessHeap temp_memory_dispenser(memory_manager.get(), memory_manager->next_free_vertex_index, sizeof(VertexDataType));
  int32_t* d_triangles = temp_memory_dispenser.getTemporaryMemory<int32_t>(memory_manager->next_free_vertex_index);
  // int32_t* d_triangle_count = temp_memory_dispenser.getTemporaryMemory<int32_t>(memory_manager->next_free_vertex_index);
  // int32_t triangle_count;
  // static int iteration_counter = 0;


  d_balancedTriangleCount<VertexDataType, EdgeDataType> <<<thread_blocks, blockdim>>>(reinterpret_cast<MemoryManager*>(memory_manager->d_memory),
                                                                                      memory_manager->d_data, 
                                                                                      d_triangles,
                                                                                      memory_manager->number_vertices,
                                                                                      memory_manager->index_shift,
                                                                                      memory_manager->page_size,
                                                                                      threads_per_block,
                                                                                      number_blocks,
                                                                                      shifter);
  
  // if (iteration_counter % 36 == 0)
  // {
  //   // Prefix scan on d_triangles to get number of triangles
  //   thrust::device_ptr<int32_t> th_triangles(d_triangles);
  //   thrust::device_ptr<int32_t> th_triangle_count(d_triangle_count);
  //   thrust::inclusive_scan(th_triangles, th_triangles + memory_manager->number_vertices, th_triangle_count);


  //   // Copy result back to host
  //   HANDLE_ERROR(cudaMemcpy(triangles.get(),
  //                           d_triangles,
  //                           sizeof(int32_t) * memory_manager->number_vertices,
  //                           cudaMemcpyDeviceToHost));

  //   // Copy number of triangles back
  //   HANDLE_ERROR(cudaMemcpy(&triangle_count,
  //                           d_triangle_count + (memory_manager->number_vertices - 1),
  //                           sizeof(int32_t),
  //                           cudaMemcpyDeviceToHost));

  //   std::cout << "Number of triangles in graph: " << triangle_count << std::endl;
  // }
  // ++iteration_counter;

  return std::move(triangles);
}

template std::unique_ptr<int32_t> workBalancedSTC <VertexData, EdgeData> (const std::unique_ptr<MemoryManager>& memory_manager, const int threads_per_block, const int number_blocks, const int shifter, const int thread_blocks, const int blockdim);
template std::unique_ptr<int32_t> workBalancedSTC <VertexDataWeight, EdgeDataWeight> (const std::unique_ptr<MemoryManager>& memory_manager, const int threads_per_block, const int number_blocks, const int shifter, const int thread_blocks, const int blockdim);
template std::unique_ptr<int32_t> workBalancedSTC <VertexDataSemantic, EdgeDataSemantic> (const std::unique_ptr<MemoryManager>& memory_manager, const int threads_per_block, const int number_blocks, const int shifter, const int thread_blocks, const int blockdim);
template std::unique_ptr<int32_t> workBalancedSTC <VertexData, EdgeDataSOA> (const std::unique_ptr<MemoryManager>& memory_manager, const int threads_per_block, const int number_blocks, const int shifter, const int thread_blocks, const int blockdim);
template std::unique_ptr<int32_t> workBalancedSTC <VertexDataWeight, EdgeDataWeightSOA> (const std::unique_ptr<MemoryManager>& memory_manager, const int threads_per_block, const int number_blocks, const int shifter, const int thread_blocks, const int blockdim);
template std::unique_ptr<int32_t> workBalancedSTC <VertexDataSemantic, EdgeDataSemanticSOA> (const std::unique_ptr<MemoryManager>& memory_manager, const int threads_per_block, const int number_blocks, const int shifter, const int thread_blocks, const int blockdim);