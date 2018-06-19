//------------------------------------------------------------------------------
// EdgeUtility.cu
//
// faimGraph
//
//------------------------------------------------------------------------------
//
#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <thrust/device_vector.h>
#include <cstddef>

#include "faimGraph.h"
#include "EdgeUpdate.h"
#include "MemoryManager.h"
#include "ConfigurationParser.h"

//------------------------------------------------------------------------------
// Device funtionality
//------------------------------------------------------------------------------
//

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType, typename UpdateDataType>
__global__ void d_realisticEdgeDeletionUpdate(MemoryManager* memory_manager,
                                              memory_t* memory,
                                              int page_size,
                                              UpdateDataType* edge_update_data,
                                              int batch_size)
{
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid >= batch_size)
    return;

  // Gather pointers
  VertexDataType* vertices = (VertexDataType*)memory;
  vertex_t edges_per_page = memory_manager->edges_per_page;

  // Get data per update
  index_t edge_src = edge_update_data[tid].source;
  int neighbours = vertices[edge_src].neighbours;

  AdjacencyIterator<EdgeDataType> adjacency_iterator(pageAccess<EdgeDataType>(memory, vertices[edge_src].mem_index, page_size, memory_manager->start_index));

  // If there are no neighbours
  if (neighbours == 0)
  {
	  edge_update_data[tid].update.destination = INVALID_INDEX;
    return;
  }

  // "Random edge"
  unsigned int random_edge_index = tid % neighbours;
  //int random_edge_index = 0;
  edge_update_data[tid].update.destination = INVALID_INDEX;
  for (int i = 0; i <= random_edge_index; ++i)
  {
    // We have found the index, get value and break
    if (i == random_edge_index)
    {
      if(adjacency_iterator.getDestination() != DELETIONMARKER)
		  edge_update_data[tid].update.destination = adjacency_iterator.getDestination();
      break;
    }
    adjacency_iterator.advanceIterator(i, edges_per_page, memory, page_size, memory_manager->start_index);
  }

  return;
}

//------------------------------------------------------------------------------
// Counts the number of updates per src index
template <typename UpdateDataType>
__global__ void d_updateInstanceCounter(UpdateDataType* edge_update_data,
                                        index_t* edge_src_counter,
                                        int number_vertices,
                                        int batch_size)
{
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid >= batch_size)
    return;

  index_t edge_src = edge_update_data[tid].source;

  if (edge_src >= number_vertices)
    return;

  atomicAdd(&(edge_src_counter[edge_src]), 1);

  return;
}

//------------------------------------------------------------------------------
// Sets up the corresponding indices, such that each thread finds all updates per src
template <typename UpdateDataType>
__global__ void d_updateSetIndices(UpdateDataType* edge_update_data,
                                   index_t* edge_src_counter,
                                   index_t* update_src_offsets,
                                   index_t* update_src_indices,
                                   int number_vertices,
                                   int batch_size)
{
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid >= batch_size)
    return;

  index_t edge_src = edge_update_data[tid].source;

  if (edge_src >= number_vertices)
    return;

  update_src_indices[update_src_offsets[edge_src] + (atomicSub(&(edge_src_counter[edge_src]), 1) - 1)] = tid;

  return;
}

//------------------------------------------------------------------------------
//
template <typename UpdateDataType>
__global__ void d_duplicateCheckingInSortedBatch(UpdateDataType* edge_update_data,
                                                 int batch_size)
{
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid >= batch_size)
    return;    

  UpdateDataType edge_update = edge_update_data[tid];
  UpdateDataType compare_element;
  while (tid < batch_size)
  {
    compare_element = edge_update_data[++tid];
    if ((edge_update.source == compare_element.source) && (edge_update.update.destination == compare_element.update.destination))
    {
      atomicExch(&(edge_update_data[tid].update.destination), DELETIONMARKER);
    }
    else
      break;
  }
  return;
}

#define MULTIPLICATOR 4
//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType, typename UpdateDataType>
__global__ void d_duplicateCheckingInSortedBatch2Graph(MemoryManager* memory_manager,
                                                       memory_t* memory,
                                                       int page_size,
                                                       UpdateDataType* edge_update_data,
                                                       int batch_size,
                                                       index_t* update_src_offsets,
                                                       index_t* deletion_helper)
{
  int warpID = threadIdx.x / WARPSIZE;
  int wid = (blockIdx.x * MULTIPLICATOR) + warpID;
  int threadID = threadIdx.x - (warpID * WARPSIZE);
  vertex_t edges_per_page = memory_manager->edges_per_page;
  // Outside threads per block (because of indexing structure we use 31 threads)
  if ((threadID >= edges_per_page) || (wid >= memory_manager->next_free_vertex_index))
    return;

  index_t update_index = update_src_offsets[wid];
  int number_updates = update_src_offsets[wid + 1] - update_src_offsets[wid];

  if (number_updates == 0)
    return;  

  VertexDataType* vertices = (VertexDataType*)memory;
  __shared__ VertexDataType vertex[MULTIPLICATOR];
  __shared__ AdjacencyIterator<EdgeDataType> adjacency_iterator[MULTIPLICATOR];

  // Retrieve vertex
  __syncwarp();
  if (SINGLE_THREAD_MULTI)
  {
    vertex[warpID] = vertices[wid];
    adjacency_iterator[warpID].setIterator(pageAccess<EdgeDataType>(memory, vertex[warpID].mem_index, page_size, memory_manager->start_index));
  }
  __syncwarp();

  // Iterate over adjacency and check duplicates batch - graph
  int round = 0;
  while (round < vertex[warpID].capacity)
  {
    if (threadID == 0 && round != 0)
    {
      adjacency_iterator[warpID] += edges_per_page;
      pointerHandlingTraverse(adjacency_iterator[warpID].getIterator(), memory, page_size, edges_per_page, memory_manager->start_index);
    }
    __syncwarp();

    // Check full adjacency against edge_update_data
    if (threadID + round < vertex[warpID].neighbours)
    {
      d_binarySearch(edge_update_data, adjacency_iterator[warpID].getDestinationAt(threadID), update_index, number_updates, deletion_helper);
    }
    
    round += (edges_per_page);
    __syncwarp();
  }

  return;
}

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType, typename UpdateDataType>
__global__ void d_duplicateCheckingInSortedBatch2Graph_Updatebased(MemoryManager* memory_manager,
																						 memory_t* memory,
																						 int page_size,
																						 UpdateDataType* edge_update_data,
																						 int batch_size,
																						 index_t* update_src_offsets)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= batch_size)
		return;

	vertex_t edges_per_page = memory_manager->edges_per_page;
	UpdateDataType edge_update = edge_update_data[tid];
	VertexDataType* vertices = (VertexDataType*)memory;
	VertexDataType vertex = vertices[edge_update.source];
	AdjacencyIterator<EdgeDataType> adjacency_iterator(pageAccess<EdgeDataType>(memory, vertex.mem_index, page_size, memory_manager->start_index));
	for(int i = 0; i < vertex.neighbours; ++i)
	{
		if (adjacency_iterator.getDestination() == edge_update.update.destination)
		{
			edge_update_data[tid].update.destination = DELETIONMARKER;
			return;
		}
		adjacency_iterator.advanceIterator(i, edges_per_page, memory, page_size, memory_manager->start_index);
	}

	return;
}

//------------------------------------------------------------------------------
// Detailed discussion at: 
// https://bitbucket.org/gpupeople/gpustreaminggraphs/issues/9/sorted-duplicate-checking-problem
template <typename UpdateDataType>
__global__ void d_duplicateCheckingIntegrateDeletions(UpdateDataType* edge_update_data,
                                                      int batch_size,
                                                      index_t* deletion_helper)
{
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid >= batch_size)
    return;

  if (deletion_helper[tid] == DELETIONMARKER)
  {
    // Special case, if duplicates are both duplicates within batch AND graph
    if (edge_update_data[tid].update.destination == DELETIONMARKER)
    {
      // We have duplicates within batch, that are at the same time duplicates in graph
      // But binary search did not find the first element, need to delete this
      do
      {
        --tid;
      } while (edge_update_data[tid].update.destination == DELETIONMARKER);
      edge_update_data[tid].update.destination = DELETIONMARKER;
    }
    else
    {
      edge_update_data[tid].update.destination = DELETIONMARKER;
    }
  }

  return;
}

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType>
__global__ void d_compaction(MemoryManager* memory_manager,
                             memory_t* memory,
                             int page_size)
{
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid >= memory_manager->next_free_vertex_index)
    return;

  // Retrieve vertex
  VertexDataType* vertices = (VertexDataType*)memory;
  VertexDataType vertex = vertices[tid];
  vertex_t edges_per_page = memory_manager->edges_per_page;
  index_t edge_block_index{ INVALID_INDEX };

  if (vertex.host_identifier == DELETIONMARKER)
  {
    // This is a deleted vertex
    return;
  }

  AdjacencyIterator<EdgeDataType> adjacency_iterator(pageAccess<EdgeDataType>(memory, vertex.mem_index, page_size, memory_manager->start_index));
  int compaction_counter{ 0 };

  // Count Deletionmarkers
  for (int i = 0; i < vertex.neighbours; ++i)
  {
    if (adjacency_iterator.getDestination() == DELETIONMARKER)
    {
      ++compaction_counter;
    }
    adjacency_iterator.advanceIterator(i, edges_per_page, memory, page_size, memory_manager->start_index);
  }

  if (compaction_counter == 0)
  {
    // Adjacency is fine, no need for compaction
    return;
  }

  // Otherwise we need to perform compaction here
  adjacency_iterator.setIterator(pageAccess<EdgeDataType>(memory, vertex.mem_index, page_size, memory_manager->start_index));
  AdjacencyIterator<EdgeDataType> compaction_iterator(adjacency_iterator);
  vertex_t compaction_index = vertex.neighbours - compaction_counter;
  compaction_iterator.advanceIteratorToIndex(edges_per_page, memory, page_size, memory_manager->start_index, edge_block_index, compaction_index);
  int compaction_counter_tmp{ compaction_counter };

  // Adjacency_iterator points now to the first element and compaction_iterator to the first possible element for compaction
  edge_block_index = INVALID_INDEX;
  for (int i = 0; i < vertex.neighbours; ++i)
  {
    if (adjacency_iterator.getDestination() == DELETIONMARKER)
    {
      // We want to compact here
      // First move the iterator to a valid position
      while (compaction_iterator.getDestinationAt(compaction_index) == DELETIONMARKER)
      {
        --compaction_counter_tmp;
        if (compaction_counter_tmp <= 0)
        {
          break;
        }
        ++compaction_index;
        compaction_iterator.advanceIteratorToIndex(edges_per_page, memory, page_size, memory_manager->start_index, edge_block_index, compaction_index);
        if (edge_block_index != INVALID_INDEX)
        {
          memory_manager->d_page_queue.enqueue(edge_block_index);
          vertices[tid].capacity -= edges_per_page;
          edge_block_index = INVALID_INDEX;
        }
      }
      if (compaction_counter_tmp <= 0)
      {
        break;
      }
      adjacency_iterator.setDestination(compaction_iterator.getDestinationAt(compaction_index));
      compaction_iterator.setDestinationAt(compaction_index, DELETIONMARKER);
      --compaction_counter_tmp;
      if (compaction_counter_tmp <= 0)
      {
        break;
      }

      ++compaction_index;
      compaction_iterator.advanceIteratorToIndex(edges_per_page, memory, page_size, memory_manager->start_index, edge_block_index, compaction_index);
      if (edge_block_index != INVALID_INDEX)
      {
        memory_manager->d_page_queue.enqueue(edge_block_index);
        vertices[tid].capacity -= edges_per_page;
        edge_block_index = INVALID_INDEX;
      }
    }
    adjacency_iterator.advanceIterator(i, edges_per_page, memory, page_size, memory_manager->start_index);
  }

  // Setup new neighbours count
  vertices[tid].neighbours -= compaction_counter;

  return;
}

//------------------------------------------------------------------------------
// 
template <typename VertexDataType, typename EdgeDataType>
__global__ void d_verifySortOrder(MemoryManager* memory_manager,
                                  memory_t* memory,
                                  int page_size,
                                  SortOrder sort_order)
{
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid >= memory_manager->next_free_vertex_index)
    return;

  // Retrieve vertex
  VertexDataType* vertices = (VertexDataType*)memory;
  vertex_t edges_per_page = memory_manager->edges_per_page;

  if (vertices[tid].host_identifier == DELETIONMARKER || vertices[tid].neighbours <= 1)
  {
    // This is a deleted vertex or we only have one edge
    return;
  }

  AdjacencyIterator<EdgeDataType> adjacency_iterator(pageAccess<EdgeDataType>(memory, vertices[tid].mem_index, page_size, memory_manager->start_index));
  AdjacencyIterator<EdgeDataType> comparison_iterator(pageAccess<EdgeDataType>(memory, vertices[tid].mem_index, page_size, memory_manager->start_index));
  ++comparison_iterator;

  for (int i = 0, j = 1; i < (vertices[tid].neighbours - 1); ++i, ++j)
  {
    if (sort_order == SortOrder::ASCENDING)
    {
      if (adjacency_iterator.getDestination() > comparison_iterator.getDestination())
      {
        printf("Adjacency sorting not correct!\n");
      }
    }
    else
    {
      if (adjacency_iterator.getDestination() < comparison_iterator.getDestination())
      {
        printf("Adjacency sorting not correct!\n");
      }
    }
    adjacency_iterator.advanceIterator(i, edges_per_page, memory, page_size, memory_manager->start_index);
    comparison_iterator.advanceIterator(j, edges_per_page, memory, page_size, memory_manager->start_index);
  }

  return;
}

//------------------------------------------------------------------------------
// 
template <typename VertexDataType, typename EdgeDataType>
__global__ void d_sortAdjacency(MemoryManager* memory_manager,
                                memory_t* memory,
                                int page_size,
                                SortOrder sort_order)
{
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid >= memory_manager->next_free_vertex_index)
    return;

  // Retrieve vertex
  VertexDataType* vertices = (VertexDataType*)memory;
  vertex_t edges_per_page = memory_manager->edges_per_page;

  if (vertices[tid].host_identifier == DELETIONMARKER || vertices[tid].neighbours <= 1)
  {
    // This is a deleted vertex
    return;
  }

  AdjacencyIterator<EdgeDataType> adjacency_iterator(pageAccess<EdgeDataType>(memory, vertices[tid].mem_index, page_size, memory_manager->start_index));
  AdjacencyIterator<EdgeDataType> sort_iterator;
  AdjacencyIterator<EdgeDataType> search_iterator;

  switch (sort_order)
  {
  case SortOrder::ASCENDING:
  {
    for (int i = 0; i < (vertices[tid].neighbours - 1); ++i)
    {
      sort_iterator.setIterator(adjacency_iterator);
      vertex_t min_element = adjacency_iterator.getDestination();
      bool need_to_swap = false;
      sort_iterator.advanceIterator(i, edges_per_page, memory, page_size, memory_manager->start_index);
      for (int j = i + 1; j < vertices[tid].neighbours; ++j)
      {
        if (sort_iterator.getDestination() < min_element)
        {
          search_iterator.setIterator(sort_iterator);
          min_element = sort_iterator.getDestination();
          need_to_swap = true;
        }
        sort_iterator.advanceIterator(j, edges_per_page, memory, page_size, memory_manager->start_index);
      }

      // Swap data and advance in list
      if (need_to_swap)
        swap(adjacency_iterator.getIterator(), search_iterator.getIterator(), edges_per_page);

      adjacency_iterator.advanceIterator(i, edges_per_page, memory, page_size, memory_manager->start_index);
    }
    break;
  }
  case SortOrder::DESCENDING:
  {
    for (int i = 0; i < (vertices[tid].neighbours - 1); ++i)
    {
      sort_iterator.setIterator(adjacency_iterator);
      vertex_t max_element = adjacency_iterator.getDestination();
      bool need_to_swap = false;
      sort_iterator.advanceIterator(i, edges_per_page, memory, page_size, memory_manager->start_index);
      for (int j = i + 1; j < vertices[tid].neighbours; ++j)
      {
        if (sort_iterator.getDestination() > max_element)
        {
          search_iterator.setIterator(sort_iterator);
          max_element = sort_iterator.getDestination();
          need_to_swap = true;
        }
        sort_iterator.advanceIterator(j, edges_per_page, memory, page_size, memory_manager->start_index);
      }

      // Swap data and advance in list
      if (need_to_swap)
        swap(adjacency_iterator.getIterator(), search_iterator.getIterator(), edges_per_page);

      adjacency_iterator.advanceIterator(i, edges_per_page, memory, page_size, memory_manager->start_index);
    }
    break;
  }
  }

  return;
}

//------------------------------------------------------------------------------
// 
template <typename VertexDataType, typename EdgeDataType>
__global__ void d_sortAdjacencyWarpSized(MemoryManager* memory_manager,
                                         memory_t* memory,
                                         int page_size,
                                         int max_block_number)
{
  int wid = blockIdx.x*blockDim.x;
  if (wid >= memory_manager->next_free_vertex_index)
    return;

  return;
}

//------------------------------------------------------------------------------
// 
template <typename VertexDataType, typename EdgeDataType>
__global__ void d_testUndirectedness(MemoryManager* memory_manager,
                                memory_t* memory,
                                int page_size)
{
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid >= memory_manager->next_free_vertex_index)
    return;

  // Retrieve vertex
  VertexDataType* vertices = (VertexDataType*)memory;
  vertex_t edges_per_page = memory_manager->edges_per_page;

  if (vertices[tid].host_identifier == DELETIONMARKER)
  {
    // This is a deleted vertex
    printf("Deleted Vertex at %d\n", tid);
    return;
  }

  AdjacencyIterator<EdgeDataType> adjacency_iterator(pageAccess<EdgeDataType>(memory, vertices[tid].mem_index, page_size, memory_manager->start_index));
  AdjacencyIterator<EdgeDataType> search_iterator;

  // Test undirectedness
  for(int i = 0; i < vertices[tid].neighbours; ++i)
  {
    vertex_t value = adjacency_iterator.getDestination();
    search_iterator.setIterator(pageAccess<EdgeDataType>(memory, vertices[value].mem_index, page_size, memory_manager->start_index));
    bool back_edge_found = false;
    for(int j = 0; j < vertices[value].neighbours; ++j)
    {
      if(search_iterator.getDestination() == tid)
      {
        back_edge_found = true;
      }
      search_iterator.advanceIterator(j, edges_per_page, memory, page_size, memory_manager->start_index);
    }
    if(!back_edge_found)
    {
      printf("Test Undirectedness failed for vertex: %d\n", tid);
    }
    adjacency_iterator.advanceIterator(i, edges_per_page, memory, page_size, memory_manager->start_index);
  } 
  
  return;
}

//------------------------------------------------------------------------------
// 
template <typename VertexDataType, typename EdgeDataType>
__global__ void d_testSelfLoops(MemoryManager* memory_manager,
                                memory_t* memory,
                                int page_size)
{
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid >= memory_manager->next_free_vertex_index)
    return;

  // Retrieve vertex
  VertexDataType* vertices = (VertexDataType*)memory;
  vertex_t edges_per_page = memory_manager->edges_per_page;

  if (vertices[tid].host_identifier == DELETIONMARKER)
  {
    // This is a deleted vertex
    printf("Deleted Vertex at %d\n", tid);
    return;
  }

  AdjacencyIterator<EdgeDataType> adjacency_iterator(pageAccess<EdgeDataType>(memory, vertices[tid].mem_index, page_size, memory_manager->start_index));

  // Test self loops
  for(int i = 0; i < vertices[tid].neighbours; ++i)
  {
    if(adjacency_iterator.getDestination() == tid)
    {
      printf("Self Loop detected for vertex: %d\n", tid);
    }
    adjacency_iterator.advanceIterator(i, edges_per_page, memory, page_size, memory_manager->start_index);
  } 
  
  return;
}

//------------------------------------------------------------------------------
// 
template <typename VertexDataType, typename EdgeDataType>
__global__ void d_resetAllocationStatus(MemoryManager* memory_manager,
                                        memory_t* memory,
                                        int page_size,
                                        vertex_t number_vertices,
                                        vertex_t vertex_offset)
{
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid >= number_vertices)
    return;

  VertexDataType* vertices = (VertexDataType*)memory;
  vertices[vertex_offset + tid].neighbours = 0;
  vertices[vertex_offset + tid].capacity = memory_manager->edges_per_page;
  
  return;
}

//------------------------------------------------------------------------------
// 
template <typename VertexDataType, typename EdgeDataType>
__global__ void d_testDuplicates(MemoryManager* memory_manager,
                                memory_t* memory,
                                int page_size)
{
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid >= memory_manager->next_free_vertex_index)
    return;

  // Retrieve vertex
  VertexDataType* vertices = (VertexDataType*)memory;
  vertex_t edges_per_page = memory_manager->edges_per_page;

  if (vertices[tid].host_identifier == DELETIONMARKER)
  {
    // This is a deleted vertex
    printf("Deleted Vertex at %d\n", tid);
    return;
  }

  AdjacencyIterator<EdgeDataType> adjacency_iterator(pageAccess<EdgeDataType>(memory, vertices[tid].mem_index, page_size, memory_manager->start_index));
  AdjacencyIterator<EdgeDataType> search_iterator;

  // Test duplicates
  for(int i = 0; i < vertices[tid].neighbours - 1; ++i)
  {
    search_iterator.setIterator(adjacency_iterator);
    search_iterator.advanceIterator(i, edges_per_page, memory, page_size, memory_manager->start_index);
    for(int j = i + 1; j < vertices[tid].neighbours; ++j)
    {
      if(adjacency_iterator.getDestination() == search_iterator.getDestination())
      {
        printf("Duplicate found in adjacency for vertex: %d\n", tid);
      }
      search_iterator.advanceIterator(j, edges_per_page, memory, page_size, memory_manager->start_index);
    }
    adjacency_iterator.advanceIterator(i, edges_per_page, memory, page_size, memory_manager->start_index);
  } 
  
  return;
}

//------------------------------------------------------------------------------
// Host funtionality
//------------------------------------------------------------------------------
//

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType, typename UpdateDataType>
std::unique_ptr<EdgeUpdatePreProcessing<UpdateDataType>> EdgeUpdateManager<VertexDataType, EdgeDataType, UpdateDataType>::edgeUpdatePreprocessing(std::unique_ptr<MemoryManager>& memory_manager,
                                                                                                                                                  const std::shared_ptr<Config>& config)
{
  std::unique_ptr<EdgeUpdatePreProcessing<UpdateDataType>> preprocessed = std::make_unique<EdgeUpdatePreProcessing<UpdateDataType>>(static_cast<uint32_t>(memory_manager->next_free_vertex_index),
                                                                                                                             static_cast<vertex_t>(updates->edge_update.size()),
                                                                                                                             memory_manager,
																																										static_cast<size_t>(sizeof(VertexDataType)));
  int batch_size = updates->edge_update.size();
  int block_size = config->testruns_.at(config->testrun_index_)->params->init_launch_block_size_;
  int grid_size = (batch_size / block_size) + 1;

  HANDLE_ERROR(cudaMemset(preprocessed->d_edge_src_counter,
                          0,
                          sizeof(index_t) * memory_manager->next_free_vertex_index));

  // Count instances of edge updates
  d_updateInstanceCounter<UpdateDataType> << < grid_size, block_size >> > (updates->d_edge_update,
                                                                           preprocessed->d_edge_src_counter,
                                                                           memory_manager->next_free_vertex_index,
                                                                           batch_size);

  // Prefix Sum Scan on d_edge_src_counter to get number of updates per vertex
  thrust::device_ptr<index_t> th_edge_src_counter(preprocessed->d_edge_src_counter);
  thrust::device_ptr<index_t> th_update_src_offsets(preprocessed->d_update_src_offsets);
  thrust::exclusive_scan(th_edge_src_counter, th_edge_src_counter + memory_manager->next_free_vertex_index + 1, th_update_src_offsets);

  return std::move(preprocessed);
}

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType>
std::unique_ptr<EdgeUpdatePreProcessing<EdgeDataUpdate>> EdgeQueryManager<VertexDataType, EdgeDataType>::edgeQueryPreprocessing(std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config)
{
  std::unique_ptr<EdgeUpdatePreProcessing<EdgeDataUpdate>> preprocessed = std::make_unique<EdgeUpdatePreProcessing<EdgeDataUpdate>>(static_cast<uint32_t>(memory_manager->next_free_vertex_index),
                                                                                                                                    static_cast<vertex_t>(queries->edge_update.size()),
                                                                                                                                    memory_manager,
																																												sizeof(VertexDataType));
  int batch_size = queries->edge_update.size();
  int block_size = config->testruns_.at(config->testrun_index_)->params->init_launch_block_size_;
  int grid_size = (batch_size / block_size) + 1;

  HANDLE_ERROR(cudaMemset(preprocessed->d_edge_src_counter,
                          0,
                          sizeof(index_t) * memory_manager->next_free_vertex_index));

  // Count instances of edge updates
  d_updateInstanceCounter<EdgeDataUpdate> << < grid_size, block_size >> > (queries->d_edge_update,
                                                                          preprocessed->d_edge_src_counter,
                                                                          memory_manager->next_free_vertex_index,
                                                                          batch_size);

  // Prefix Sum Scan on d_edge_src_counter to get number of updates per vertex
  thrust::device_ptr<index_t> th_edge_src_counter(preprocessed->d_edge_src_counter);
  thrust::device_ptr<index_t> th_update_src_offsets(preprocessed->d_update_src_offsets);
  thrust::exclusive_scan(th_edge_src_counter, th_edge_src_counter + memory_manager->next_free_vertex_index + 1, th_update_src_offsets);

  return std::move(preprocessed);
}

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType, typename UpdateDataType>
void EdgeUpdateManager<VertexDataType, EdgeDataType, UpdateDataType>::edgeUpdateDuplicateChecking(std::unique_ptr<MemoryManager>& memory_manager,
                                                                                                 const std::shared_ptr<Config>& config,
                                                                                                 const std::unique_ptr<EdgeUpdatePreProcessing<UpdateDataType>>& preprocessed)
{
  int batch_size = updates->edge_update.size();
  int block_size = 256; // config->testruns_.at(config->testrun_index_)->params->init_launch_block_size_;
  int grid_size = (memory_manager->next_free_vertex_index / block_size) + 1;

  // Duplicate Checking in Graph (sorted updates)
#ifdef UPDATE_BASED_DUPLICATE_CHECKING
  block_size = 256;
  grid_size = (batch_size / block_size) + 1;
  d_duplicateCheckingInSortedBatch2Graph_Updatebased<VertexDataType, EdgeDataType, UpdateDataType> << < grid_size, block_size >> >((MemoryManager*)memory_manager->d_memory,
																																												memory_manager->d_data,
																																												memory_manager->page_size,
																																												updates->d_edge_update,
																																												batch_size,
																																												preprocessed->d_update_src_offsets);
#else
  TemporaryMemoryAccessHeap temp_memory_dispenser(memory_manager.get(), memory_manager->next_free_vertex_index, sizeof(VertexData));
  temp_memory_dispenser.getTemporaryMemory<UpdateDataType>(batch_size);
  temp_memory_dispenser.getTemporaryMemory<index_t>(memory_manager->next_free_vertex_index + 1);
  temp_memory_dispenser.getTemporaryMemory<index_t>(memory_manager->next_free_vertex_index + 1);
  index_t* d_deletion_helper = temp_memory_dispenser.getTemporaryMemory<index_t>(batch_size);
  HANDLE_ERROR(cudaMemset(d_deletion_helper,
	  0,
	  sizeof(index_t) * batch_size));
  block_size = WARPSIZE * MULTIPLICATOR;
  grid_size = (memory_manager->next_free_vertex_index / MULTIPLICATOR) + 1;
  d_duplicateCheckingInSortedBatch2Graph<VertexDataType, EdgeDataType, UpdateDataType> << < grid_size, block_size >> >((MemoryManager*)memory_manager->d_memory,
																																								memory_manager->d_data,
																																								memory_manager->page_size,
																																								updates->d_edge_update,
																																								batch_size,
																																								preprocessed->d_update_src_offsets,
																																								d_deletion_helper);
#endif
  // }
  

  // Duplicate Checking in Batch (sorted updates)
  block_size = 256;
  grid_size = (batch_size / block_size) + 1;
  d_duplicateCheckingInSortedBatch<UpdateDataType> << < grid_size, block_size >> >(updates->d_edge_update,
                                                                                   batch_size);
  cudaDeviceSynchronize();

#ifndef UPDATE_BASED_DUPLICATE_CHECKING
  d_duplicateCheckingIntegrateDeletions<UpdateDataType> << < grid_size, block_size >> >(updates->d_edge_update,
                                                                                        batch_size,
                                                                                        d_deletion_helper);
#endif

  cudaDeviceSynchronize();

  return;
}

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType, typename UpdateDataType>
std::unique_ptr<EdgeUpdateBatch<UpdateDataType>> EdgeUpdateManager<VertexDataType, EdgeDataType, UpdateDataType>::generateEdgeUpdates(const std::unique_ptr<MemoryManager>& memory_manager,
                                                                     vertex_t batch_size,
                                                                     unsigned int seed,
                                                                     unsigned int range,
                                                                     unsigned int offset)
{
  std::unique_ptr<EdgeUpdateBatch<UpdateDataType>> edge_update(std::make_unique<EdgeUpdateBatch<UpdateDataType>>());
  int block_size = KERNEL_LAUNCH_BLOCK_SIZE;
  int grid_size = (batch_size / block_size) + 1;

  // Generate random start vertices
  srand(seed + 1);
  // edge_update->edge_update.reserve(batch_size);
  for (unsigned int i = 0; i < batch_size/2; ++i)
  {
    UpdateDataType edge_update_data;
    vertex_t intermediate = rand() % ((range && (range < memory_manager->next_free_vertex_index)) ? range : memory_manager->next_free_vertex_index);
    vertex_t source;
    if(offset + intermediate < memory_manager->next_free_vertex_index)
      source = offset + intermediate;
    else
      source = intermediate;
    edge_update_data.source = source;
    edge_update->edge_update.push_back(edge_update_data);
  }
  for (unsigned int i = batch_size/2; i < batch_size; ++i)
  {
    UpdateDataType edge_update_data;
    vertex_t intermediate = rand() % (memory_manager->next_free_vertex_index);
    vertex_t source;
    if(offset + intermediate < memory_manager->next_free_vertex_index)
      source = offset + intermediate;
    else
      source = intermediate;
    edge_update_data.source = source;
    edge_update->edge_update.push_back(edge_update_data);
  }

  // Allocate memory on device heap for edge updates
  TemporaryMemoryAccessHeap temp_memory_dispenser(memory_manager.get(), memory_manager->next_free_vertex_index, sizeof(VertexDataType));
  edge_update->d_edge_update = temp_memory_dispenser.getTemporaryMemory<UpdateDataType>(batch_size);

  HANDLE_ERROR(cudaMemcpy(edge_update->d_edge_update,
                          edge_update->edge_update.data(),
                          sizeof(UpdateDataType) * batch_size,
                          cudaMemcpyHostToDevice));

  d_realisticEdgeDeletionUpdate<VertexDataType, EdgeDataType, UpdateDataType> << < grid_size, block_size >> >((MemoryManager*)memory_manager->d_memory,
                                                                                                              memory_manager->d_data,
                                                                                                              memory_manager->page_size,
                                                                                                              edge_update->d_edge_update,
                                                                                                              batch_size);

  cudaDeviceSynchronize();

  HANDLE_ERROR(cudaMemcpy(edge_update->edge_update.data(),
                          edge_update->d_edge_update,
                          sizeof(UpdateDataType) * batch_size,
                          cudaMemcpyDeviceToHost));

  // Write data to file to verify
  static int counter = 0;
#ifdef DEBUG_VERBOSE_OUPUT
  std::string filename = std::string("../tests/Verification/VerifyDelete");
  filename += std::to_string(counter) + std::string(".txt");
  std::ofstream file(filename);
  if (file.is_open())
  {
    for (int i = 0; i < batch_size; ++i)
    {
      file << edge_update->edge_update.at(i).source << " ";
      file << edge_update->edge_update.at(i).update.destination << "\n";
    }
  }
#endif
  ++counter;

  return std::move(edge_update);
}

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType, typename UpdateDataType>
template <typename VertexUpdateType>
std::unique_ptr<EdgeUpdateBatch<UpdateDataType>> EdgeUpdateManager<VertexDataType, EdgeDataType, UpdateDataType>::generateEdgeUpdatesConcurrent<VertexUpdateType>(std::unique_ptr<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>>& faimGraph,
                                                                                                                                                                  const std::unique_ptr<MemoryManager>& memory_manager,
                                                                                                                                                                  vertex_t batch_size,
                                                                                                                                                                  unsigned int seed,
                                                                                                                                                                  unsigned int range,
                                                                                                                                                                  unsigned int offset)
{
  std::unique_ptr<EdgeUpdateBatch<UpdateDataType>> edge_update (std::make_unique<EdgeUpdateBatch<UpdateDataType>>());

  // Get original graph
  auto verify_graph = faimGraph->verifyGraphStructure (const_cast<std::unique_ptr<MemoryManager>&>(memory_manager));

  // Generate random edge updates
  srand(seed + 1);
  
  for(unsigned int i = 0; i < batch_size; ++i)
  {
	  UpdateDataType edge_update_data;
	  edge_update_data.source = offset + rand() % ((range) ? range : memory_manager->next_free_vertex_index);
    vertex_t index = verify_graph->h_offset[edge_update_data.source];
    vertex_t neighbours;
    if(edge_update_data.source == memory_manager->next_free_vertex_index - 1)
    {
      neighbours = (verify_graph->number_edges) - index;
    }
    else 
    {
      neighbours = verify_graph->h_offset[edge_update_data.source + 1] - index;
    } 
    bool value_okay = false;
    while(!value_okay)
    {
	    edge_update_data.update.destination = rand() % memory_manager->next_free_vertex_index;
      value_okay = true;
      for(unsigned int j = index; j < index + neighbours; ++j)
      {
        if(verify_graph->h_adjacency[j] == edge_update_data.update.destination)
        {
          value_okay = false;
        }
      }
    }     
    edge_update->edge_update.push_back(edge_update_data);
  }

  return std::move(edge_update);
}

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType>
void MemoryManager::compaction(const std::shared_ptr<Config>& config)
{
  int batch_size = next_free_vertex_index;
  int block_size = config->testruns_.at(config->testrun_index_)->params->insert_launch_block_size_;
  int grid_size = (batch_size / block_size) + 1;

  d_compaction<VertexDataType, EdgeDataType> << <grid_size, block_size >> > ((MemoryManager*)d_memory,
                                                                              d_data,
                                                                              page_size);
  return;
}

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType>
void MemoryManager::sortAdjacency(const std::shared_ptr<Config>& config, SortOrder sort_order)
{
  int batch_size = next_free_vertex_index;
  int block_size = config->testruns_.at(config->testrun_index_)->params->insert_launch_block_size_;
  int grid_size = (batch_size / block_size) + 1;

  d_sortAdjacency<VertexDataType, EdgeDataType> << <grid_size, block_size >> > ((MemoryManager*)d_memory,
                                                                                d_data,
                                                                                page_size,
                                                                                sort_order);

  cudaDeviceSynchronize();
  
  return;
}

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType>
void MemoryManager::testUndirectedness(const std::shared_ptr<Config>& config)
{
  int batch_size = next_free_vertex_index;
  int block_size = config->testruns_.at(config->testrun_index_)->params->insert_launch_block_size_;
  int grid_size = (batch_size / block_size) + 1;

  d_testUndirectedness<VertexDataType, EdgeDataType> << <grid_size, block_size >> > ((MemoryManager*)d_memory,
                                                                                      d_data,
                                                                                      page_size);

  cudaDeviceSynchronize();  
  return;
}

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType>
void MemoryManager::testSelfLoops(const std::shared_ptr<Config>& config)
{
  int batch_size = next_free_vertex_index;
  int block_size = config->testruns_.at(config->testrun_index_)->params->insert_launch_block_size_;
  int grid_size = (batch_size / block_size) + 1;

  d_testSelfLoops<VertexDataType, EdgeDataType> << <grid_size, block_size >> > ((MemoryManager*)d_memory,
                                                                              d_data,
                                                                              page_size);

  cudaDeviceSynchronize();  
  return;
}

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType>
void MemoryManager::resetAllocationStatus(const std::shared_ptr<Config>& config, vertex_t number_vertices, vertex_t vertex_offset)
{
  int block_size = config->testruns_.at(config->testrun_index_)->params->insert_launch_block_size_;
  int grid_size = (number_vertices / block_size) + 1;

  d_resetAllocationStatus<VertexDataType, EdgeDataType> << <grid_size, block_size >> > ((MemoryManager*)d_memory,
                                                                                        d_data,
                                                                                        page_size,
                                                                                        number_vertices,
                                                                                        vertex_offset);

  cudaDeviceSynchronize();
  return;
}

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType>
void MemoryManager::testDuplicates(const std::shared_ptr<Config>& config)
{
  int batch_size = next_free_vertex_index;
  int block_size = config->testruns_.at(config->testrun_index_)->params->insert_launch_block_size_;
  int grid_size = (batch_size / block_size) + 1;

  d_testDuplicates<VertexDataType, EdgeDataType> << <grid_size, block_size >> > ((MemoryManager*)d_memory,
                                                                                d_data,
                                                                                page_size);

  cudaDeviceSynchronize();  
  return;
}
