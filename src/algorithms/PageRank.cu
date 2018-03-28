//------------------------------------------------------------------------------
// PageRank.cu
//
// Masterthesis aimGraph
//
// Authors: Martin Winter, 1130688
//------------------------------------------------------------------------------
//
#include <thrust/device_vector.h>

#include "MemoryManager.h"
#include "PageRank.h"

//------------------------------------------------------------------------------
// Device funtionality
//------------------------------------------------------------------------------
//

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType>
__global__ void d_algPageRankNaive(MemoryManager* memory_manager,
                                   memory_t* memory,
                                   int page_size,
                                   float* page_rank,
                                   float* next_page_rank)
{
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid >= memory_manager->next_free_vertex_index)
    return;

  // PageRank
  VertexDataType* vertices = (VertexDataType*)memory;

  if (vertices[tid].host_identifier == DELETIONMARKER)
    return;

  AdjacencyIterator<EdgeDataType> adjacency_iterator(pageAccess<EdgeDataType>(memory, vertices[tid].mem_index, page_size, memory_manager->start_index));

  float page_factor = page_rank[tid] / vertices[tid].neighbours;

  for (int i = 0; i < vertices[tid].neighbours; ++i)
  {
    atomicAdd(&next_page_rank[adjacency_iterator.getDestination()], page_factor);
    adjacency_iterator.advanceIterator(i, memory_manager->edges_per_page, memory, page_size, memory_manager->start_index);
  }

  return;
}

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType>
__global__ void d_algPageRankBalanced(MemoryManager* memory_manager,
                                      memory_t* memory,
                                      int page_size,
                                      float* page_rank,
                                      float* next_page_rank,
                                      vertex_t* vertex_index,
                                      vertex_t* page_per_vertex_index,
                                      int page_count)
{
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid >= page_count)
    return;

  // PageRank
  VertexDataType* vertices = (VertexDataType*)memory;
  vertex_t index = vertex_index[tid];
  vertex_t page_index = page_per_vertex_index[tid];
  vertex_t edges_per_page = memory_manager->edges_per_page;
  vertex_t neighbours = vertices[index].neighbours;

  if (vertices[index].host_identifier == DELETIONMARKER)
    return;

  AdjacencyIterator<EdgeDataType> adjacency_iterator(pageAccess<EdgeDataType>(memory, vertices[index].mem_index, page_size, memory_manager->start_index));
  for (int i = page_index; i > 0; --i)
  {
    adjacency_iterator.blockTraversalAbsolute(edges_per_page, memory, page_size, memory_manager->start_index);
  }
  // Now every thread points to its unique page in memory

  int iterations;
  if ((neighbours) < ((page_index + 1) * edges_per_page))
  {
    iterations = (neighbours) % edges_per_page;
  }
  else
  {
    iterations = edges_per_page;
  }

  float page_factor = page_rank[index] / neighbours;

  for (int i = 0; i < iterations; ++i)
  {
    atomicAdd(&next_page_rank[adjacency_iterator.getDestination()], page_factor);
    adjacency_iterator.advanceIterator(i, memory_manager->edges_per_page, memory, page_size, memory_manager->start_index);
  }

  return;
}

//------------------------------------------------------------------------------
//
__global__ void d_applyPageRank(MemoryManager* memory_manager,
                                float* page_rank,
                                float* next_page_rank,
                                float* absolute_difference,
                                float dampening_factor)
{
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid >= memory_manager->next_free_vertex_index)
    return;

  float abs_diff{0.0f};
  if (dampening_factor <= 0)
  {
    // Use standard formula: PR = sum(PR(x)/N(x))
    absolute_difference[tid] = page_rank[tid] - next_page_rank[tid];
    page_rank[tid] = next_page_rank[tid];
  }
  else
  {
    // Use formula with dampening: PR = (1 - damp)/N +  d*(sum(PR(x)/N(x)))
    abs_diff = page_rank[tid];
    page_rank[tid] = ((1.0f - dampening_factor) / (memory_manager->number_vertices)) + (dampening_factor * next_page_rank[tid]);
    absolute_difference[tid] = abs_diff - page_rank[tid];
  }
  return;
}


//------------------------------------------------------------------------------
// Host funtionality
//------------------------------------------------------------------------------
//

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType>
float PageRank<VertexDataType, EdgeDataType>::algPageRankNaive(const std::unique_ptr<MemoryManager>& memory_manager)
{
  float absDiff = 0.0f;
  int block_size = 256;
  int grid_size = (memory_manager->next_free_vertex_index / block_size) + 1;

  HANDLE_ERROR(cudaMemset(d_next_page_rank,
                          0.0f, 
                          sizeof(float) * memory_manager->next_free_vertex_index));

  d_algPageRankNaive <VertexDataType, EdgeDataType> << <grid_size, block_size >> > ((MemoryManager*)memory_manager->d_memory,
                                                                                     memory_manager->d_data, 
                                                                                     memory_manager->page_size,
                                                                                     d_page_rank,
                                                                                     d_next_page_rank);

  // Now we have to set the pagerank
  d_applyPageRank << < grid_size, block_size >> > ((MemoryManager*)memory_manager->d_memory,
                                                    d_page_rank,
                                                    d_next_page_rank,
                                                    d_absolute_difference,
                                                    dampening_factor);


  thrust::device_ptr<float> th_abs_diff(d_absolute_difference);
  thrust::device_ptr<float> th_diff_sum(d_diff_sum);
	thrust::inclusive_scan(th_abs_diff, th_abs_diff + memory_manager->next_free_vertex_index, th_diff_sum);

  // Copy result back to host
  HANDLE_ERROR(cudaMemcpy(&absDiff,
                          d_diff_sum + (memory_manager->next_free_vertex_index - 1),
                          sizeof(float),
                          cudaMemcpyDeviceToHost));

  return absDiff;
}

template float PageRank<VertexData, EdgeData>::algPageRankNaive (const std::unique_ptr<MemoryManager>& memory_manager);
template float PageRank<VertexDataWeight, EdgeDataWeight>::algPageRankNaive (const std::unique_ptr<MemoryManager>& memory_manager);
template float PageRank<VertexDataSemantic, EdgeDataSemantic>::algPageRankNaive (const std::unique_ptr<MemoryManager>& memory_manager);
template float PageRank<VertexData, EdgeDataSOA>::algPageRankNaive (const std::unique_ptr<MemoryManager>& memory_manager);
template float PageRank<VertexDataWeight, EdgeDataWeightSOA>::algPageRankNaive (const std::unique_ptr<MemoryManager>& memory_manager);
template float PageRank<VertexDataSemantic, EdgeDataSemanticSOA>::algPageRankNaive (const std::unique_ptr<MemoryManager>& memory_manager);

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType>
float PageRank<VertexDataType, EdgeDataType>::algPageRankBalanced(const std::unique_ptr<MemoryManager>& memory_manager)
{
  float absDiff = 0.0f;
  int block_size = 256;
  int grid_size = (memory_manager->next_free_vertex_index / block_size) + 1;

  HANDLE_ERROR(cudaMemset(d_next_page_rank,
                          0.0f,
                          sizeof(float) * memory_manager->next_free_vertex_index));

  int number_pages = memory_manager->numberPagesInMemory<VertexDataType>(d_page_count, d_accumulated_page_count);

  if (d_vertex_index == nullptr)
  {
    TemporaryMemoryAccessHeap temp_memory_dispenser(memory_manager.get(), reinterpret_cast<memory_t*>(d_page_count + memory_manager->next_free_vertex_index + 1));
    d_vertex_index = temp_memory_dispenser.getTemporaryMemory<vertex_t>(number_pages);
    d_page_per_vertex_index = temp_memory_dispenser.getTemporaryMemory<vertex_t>(number_pages);
  } 

  memory_manager->workBalanceCalculation(d_accumulated_page_count, number_pages, d_vertex_index, d_page_per_vertex_index);

  grid_size = (number_pages / block_size) + 1;
  d_algPageRankBalanced <VertexDataType, EdgeDataType> << <grid_size, block_size >> > ((MemoryManager*)memory_manager->d_memory,
                                                                                        memory_manager->d_data,
                                                                                        memory_manager->page_size,
                                                                                        d_page_rank,
                                                                                        d_next_page_rank,
                                                                                        d_vertex_index,
                                                                                        d_page_per_vertex_index,
                                                                                        number_pages);

  grid_size = (memory_manager->next_free_vertex_index / block_size) + 1;

  // Now we have to set the pagerank
  d_applyPageRank << < grid_size, block_size >> > ((MemoryManager*)memory_manager->d_memory,
                                                    d_page_rank,
                                                    d_next_page_rank,
                                                    d_absolute_difference,
                                                    dampening_factor);


  thrust::device_ptr<float> th_abs_diff(d_absolute_difference);
  thrust::device_ptr<float> th_diff_sum(d_diff_sum);
  thrust::inclusive_scan(th_abs_diff, th_abs_diff + memory_manager->next_free_vertex_index, th_diff_sum);

  // Copy result back to host
  HANDLE_ERROR(cudaMemcpy(&absDiff,
                          d_diff_sum + (memory_manager->next_free_vertex_index - 1),
                          sizeof(float),
                          cudaMemcpyDeviceToHost));

  return absDiff;
}

template float PageRank<VertexData, EdgeData>::algPageRankBalanced (const std::unique_ptr<MemoryManager>& memory_manager);
template float PageRank<VertexDataWeight, EdgeDataWeight>::algPageRankBalanced (const std::unique_ptr<MemoryManager>& memory_manager);
template float PageRank<VertexDataSemantic, EdgeDataSemantic>::algPageRankBalanced (const std::unique_ptr<MemoryManager>& memory_manager);
template float PageRank<VertexData, EdgeDataSOA>::algPageRankBalanced (const std::unique_ptr<MemoryManager>& memory_manager);
template float PageRank<VertexDataWeight, EdgeDataWeightSOA>::algPageRankBalanced (const std::unique_ptr<MemoryManager>& memory_manager);
template float PageRank<VertexDataSemantic, EdgeDataSemanticSOA>::algPageRankBalanced (const std::unique_ptr<MemoryManager>& memory_manager);