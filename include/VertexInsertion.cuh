//------------------------------------------------------------------------------
// VertexInsertion.cu
//
// faimGraph
//
//------------------------------------------------------------------------------
//
#pragma once
#include <iostream>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include "VertexUpdate.h"
#include "MemoryManager.h"
#include "ConfigurationParser.h"
#include "faimGraph.h"
#include "EdgeUpdate.h"

namespace faimGraphVertexInsertion
{
  //------------------------------------------------------------------------------
  // Device funtionality
  //------------------------------------------------------------------------------
  //

  //------------------------------------------------------------------------------
  // Detailed discussion at: 
  // https://bitbucket.org/gpupeople/gpustreaminggraphs/issues/9/sorted-duplicate-checking-problem
  template <typename VertexUpdateType>
  __global__ void d_updateIntegrateDeletions(VertexUpdateType* vertex_update_data,
                                            int batch_size,
                                            index_t* deletion_helper)
  {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid >= batch_size)
      return;

    if (deletion_helper[tid] == DELETIONMARKER)
    {
      // Special case, if duplicates are both duplicates within batch AND graph
      if (vertex_update_data[tid].identifier == DELETIONMARKER)
      {
        // We have duplicates within batch, that are at the same time duplicates in graph
        // But binary search did not find the first element, need to delete this
        do
        {
          --tid;
        } while (vertex_update_data[tid].identifier == DELETIONMARKER);
        vertex_update_data[tid].identifier = DELETIONMARKER;
      }
      else
      {
        vertex_update_data[tid].identifier = DELETIONMARKER;
      }
    }

    return;
  }

  //------------------------------------------------------------------------------
  //
  template <typename VertexUpdateType>
  __global__ void d_duplicateInBatchCheckingSorted(VertexUpdateType* vertex_update_data,
                                                  int batch_size)
  {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid >= batch_size)
      return;

    index_t host_identifier = vertex_update_data[tid].identifier;
    if (host_identifier != DELETIONMARKER)
    {
      while (host_identifier == vertex_update_data[tid + 1].identifier && tid < batch_size - 1)
      {
        vertex_update_data[tid + 1].identifier = DELETIONMARKER;
        ++tid;
      }
    }

    return;
  }

  //------------------------------------------------------------------------------
  //
  template <typename VertexDataType, typename VertexUpdateType>
  __global__ void d_duplicateInGraphCheckingSorted(VertexUpdateType* vertex_update_data,
                                                  int batch_size,
                                                  MemoryManager* memory_manager,
                                                  memory_t* memory,
                                                  index_t* deletion_helper)
  {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid >= memory_manager->next_free_vertex_index)
      return;

    VertexDataType* vertices = (VertexDataType*)memory;
    index_t vertex_ID = vertices[tid].host_identifier;

    // Do a binary search
    d_binarySearch(vertex_update_data, vertex_ID, batch_size, deletion_helper);

    return;
  }

  //------------------------------------------------------------------------------
  //
  template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType>
  __global__ void d_vertexInsertion(MemoryManager* memory_manager,
                                    memory_t* memory,
                                    int number_vertices,
                                    int page_size,
                                    int batch_size,
                                    index_t* device_mapping,
                                    index_t* device_mapping_update,
                                    VertexUpdateType* vertex_update_data,
                                    int device_mapping_size)
  {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid >= batch_size)
      return;

    // First let's see if we got a valid update
    VertexUpdateType vertex_update = vertex_update_data[tid];

    if (vertex_update.identifier == DELETIONMARKER)
    {
      // We got a duplicate, let's return
      device_mapping_update[tid] = DELETIONMARKER;
      return;
    }

    // We need an index, first ask the queue for deleted indices, otherwise take a new one
    index_t device_index{ DELETIONMARKER };
    if (memory_manager->d_vertex_queue.dequeueAlternating(device_index) == FALSE)
    {
      device_index = atomicAdd(&(memory_manager->next_free_vertex_index), 1);
    }

    // Now we have an index, now we just need an empty block as well
    index_t page_index{ DELETIONMARKER };
    if (memory_manager->d_page_queue.dequeue(page_index) == FALSE)
    {
      page_index = atomicAdd(&(memory_manager->next_free_page), 1);
    }

    // Clean page at hand
    AdjacencyIterator<EdgeDataType> adjacency_iterator(pageAccess<EdgeDataType>(memory, page_index, page_size, memory_manager->start_index));
    adjacency_iterator.cleanPageInclusive(memory_manager->edges_per_page);

    // Set all the stuff up and write to global memory
    VertexDataType vertex;
    setupVertex(vertex, vertex_update, page_index, memory_manager->edges_per_page);
    VertexDataType* vertices = (VertexDataType*)(memory);
    vertices[device_index] = vertex;

    // Get mapping back to host
    device_mapping[device_index] = vertex_update.identifier;
    device_mapping_update[tid] = device_index;

    // Increase number_vertices counter
    atomicAdd(&(memory_manager->number_vertices), 1);

    return;
  }

  //------------------------------------------------------------------------------
  //
  template <typename VertexUpdateType>
  __global__ void d_duplicateInBatchChecking(VertexUpdateType* vertex_update_data,
                                            int batch_size)
  {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid >= batch_size)
      return;

    // Perform duplicate checking within a batch
    for (int i = tid + 1; i < batch_size; ++i)
    {
      if (vertex_update_data[tid].identifier == vertex_update_data[i].identifier)
      {
        atomicExch(&(vertex_update_data[i].identifier), DELETIONMARKER);
      }
    }

    return;
  }

  //------------------------------------------------------------------------------
  // TODO: Make this more efficient using shared memory at least
  //
  template <typename VertexDataType, typename VertexUpdateType>
  __global__ void d_duplicateInGraphChecking(VertexUpdateType* vertex_update_data,
                                            int batch_size,
                                            MemoryManager* memory_manager,
                                            memory_t* memory,
                                            bool iterateUpdate)
  {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (iterateUpdate)
    {
      if (tid >= memory_manager->next_free_vertex_index)
        return;

      VertexDataType* vertices = (VertexDataType*)memory;
      index_t vertex_ID = vertices[tid].host_identifier;
      if (vertex_ID == DELETIONMARKER)
        return;

      // Perform duplicate checking graph-batch  
      for (int i = 0; i < batch_size; ++i)
      {
        if (vertex_update_data[i].identifier == vertex_ID)
        {
          atomicExch(&(vertex_update_data[i].identifier), DELETIONMARKER);
          return;
        }
      }
    }
    else
    {
      if (tid >= batch_size)
        return;

      VertexDataType* vertices = (VertexDataType*)memory;
      index_t update_ID = vertex_update_data[tid].identifier;
      if (update_ID == DELETIONMARKER)
        return;

      // Perform duplicate checking graph-batch  
      for (int i = 0; i < memory_manager->next_free_vertex_index; ++i)
      {
        if (vertices[i].host_identifier == update_ID)
        {
          vertex_update_data[tid].identifier = DELETIONMARKER;
          return;
        }
      }
    }

    return;
  }
}

//------------------------------------------------------------------------------
// Host funtionality
//------------------------------------------------------------------------------
//

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename VertexUpdateType>
template <typename EdgeDataType>
void VertexUpdateManager<VertexDataType, VertexUpdateType>::deviceVertexInsertion<EdgeDataType>(std::unique_ptr<MemoryManager>& memory_manager,
                                                                                               const std::shared_ptr<Config>& config,
                                                                                               VertexMapper<index_t, index_t>& mapper,
                                                                                               bool duplicate_checking)
{
  int batch_size = vertex_insertion_updates->vertex_data.size();
  int block_size = config->testruns_.at(config->testrun_index_)->params->insert_launch_block_size_;
  int grid_size = (batch_size / block_size) + 1;

  cudaEvent_t ce_start, ce_stop;

  ScopedMemoryAccessHelper scoped_mem_access_counter(memory_manager.get(), sizeof(VertexUpdateType) *  batch_size);
  TemporaryMemoryAccessStack temp_memory_dispenser(memory_manager.get(), reinterpret_cast<memory_t*>(vertex_insertion_updates->d_vertex_data));

  // Copy Updates to the device
  HANDLE_ERROR(cudaMemcpy(vertex_insertion_updates->d_vertex_data,
    vertex_insertion_updates->vertex_data.data(),
    sizeof(VertexUpdateType) * batch_size,
    cudaMemcpyHostToDevice));

  // Do we need duplicate Checking in the beginning
  if (duplicate_checking)
  {
    if (config->testruns_.at(config->testrun_index_)->params->sorting_)
    {
      index_t* d_deletion_helper = temp_memory_dispenser.getTemporaryMemory<index_t>(batch_size);
      HANDLE_ERROR(cudaMemset(d_deletion_helper,
        0,
        sizeof(index_t) * batch_size));

      thrust::device_ptr<index_t> th_vertex_updates((index_t*)(vertex_insertion_updates->d_vertex_data));
      thrust::sort(th_vertex_updates, th_vertex_updates + batch_size);

      // Check Duplicates within graph-batch
      start_clock(ce_start, ce_stop);
      grid_size = (memory_manager->next_free_vertex_index / block_size) + 1;
      faimGraphVertexInsertion::d_duplicateInGraphCheckingSorted<VertexDataType, VertexUpdateType> << < grid_size, block_size >> > (vertex_insertion_updates->d_vertex_data,
                                                                                                          batch_size,
                                                                                                          (MemoryManager*)memory_manager->d_memory,
                                                                                                          memory_manager->d_data,
                                                                                                          d_deletion_helper);
      time_dup_in_graph += end_clock(ce_start, ce_stop);
      grid_size = (batch_size / block_size) + 1;
      start_clock(ce_start, ce_stop);
      // Check Duplicates within the batch
      faimGraphVertexInsertion::d_duplicateInBatchCheckingSorted <VertexUpdateType> << < grid_size, block_size >> > (vertex_insertion_updates->d_vertex_data,
                                                                                           batch_size);
      time_dup_in_batch += end_clock(ce_start, ce_stop);

      faimGraphVertexInsertion::d_updateIntegrateDeletions<VertexUpdateType> << < grid_size, block_size >> >(vertex_insertion_updates->d_vertex_data,
                                                                                   batch_size,
                                                                                   d_deletion_helper);
    }
    else
    {
      start_clock(ce_start, ce_stop);
      // Check Duplicates within the batch
      faimGraphVertexInsertion::d_duplicateInBatchChecking <VertexUpdateType> << < grid_size, block_size >> > (vertex_insertion_updates->d_vertex_data,
                                                                                     batch_size);
      time_dup_in_batch += end_clock(ce_start, ce_stop);

      // Check Duplicates within graph-batch
      start_clock(ce_start, ce_stop);
      grid_size = (memory_manager->next_free_vertex_index / block_size) + 1;
      faimGraphVertexInsertion::d_duplicateInGraphChecking<VertexDataType, VertexUpdateType> << < grid_size, block_size >> > (vertex_insertion_updates->d_vertex_data,
                                                                                                    batch_size,
                                                                                                    (MemoryManager*)memory_manager->d_memory,
                                                                                                    memory_manager->d_data,
                                                                                                    true);

      time_dup_in_graph += end_clock(ce_start, ce_stop);
      grid_size = (batch_size / block_size) + 1;
    }
  }

  start_clock(ce_start, ce_stop);
  // The following implementations assumes that no duplicates are present
  faimGraphVertexInsertion::d_vertexInsertion<VertexDataType, VertexUpdateType, EdgeDataType> << < grid_size, block_size >> > ((MemoryManager*)memory_manager->d_memory,
                                                                                                      memory_manager->d_data,
                                                                                                      memory_manager->number_vertices,
                                                                                                      memory_manager->page_size,
                                                                                                      batch_size,
                                                                                                      mapper.d_device_mapping,
                                                                                                      mapper.d_device_mapping_update,
                                                                                                      vertex_insertion_updates->d_vertex_data,
                                                                                                      memory_manager->next_free_vertex_index + batch_size);

  time_insertion += end_clock(ce_start, ce_stop);

  updateMemoryManagerHost(memory_manager);

  //std::cout << "MappingSize: " << memory_manager->next_free_vertex_index << " and MappingUpdateSize: " << updates->vertex_data.size() << std::endl;
  mapper.h_device_mapping.resize(memory_manager->next_free_vertex_index);

  //Copy Updates back from the device
  HANDLE_ERROR(cudaMemcpy(mapper.h_device_mapping.data(),
                          mapper.d_device_mapping,
                          sizeof(index_t) * memory_manager->next_free_vertex_index,
                          cudaMemcpyDeviceToHost));

  HANDLE_ERROR(cudaMemcpy(mapper.h_device_mapping_update.data(),
                          mapper.d_device_mapping_update,
                          sizeof(index_t) * vertex_insertion_updates->vertex_data.size(),
                          cudaMemcpyDeviceToHost));

  if (config->testruns_.at(config->testrun_index_)->params->sorting_)
  {
    HANDLE_ERROR(cudaMemcpy(vertex_insertion_updates->vertex_data.data(),
                            vertex_insertion_updates->d_vertex_data,
                            sizeof(VertexUpdateType) * batch_size,
                            cudaMemcpyDeviceToHost));
  }

  return;
}

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename EdgeUpdateType>
void faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, EdgeUpdateType>::vertexInsertion(VertexMapper<index_t, index_t>& mapper)
{
  vertex_update_manager->template deviceVertexInsertion<EdgeDataType>(memory_manager, config, mapper, true);
}

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename VertexUpdateType>
void VertexUpdateManager<VertexDataType, VertexUpdateType>::duplicateInBatchChecking(const std::shared_ptr<Config>& config)
{
  int batch_size = vertex_insertion_updates->vertex_data.size();
  int block_size = config->testruns_.at(config->testrun_index_)->params->insert_launch_block_size_;
  int grid_size = (batch_size / block_size) + 1;

  // Check Duplicates within the batch
  faimGraphVertexInsertion::d_duplicateInBatchChecking <VertexUpdateType> << < grid_size, block_size >> > (vertex_insertion_updates->d_vertex_data,
                                                                                 batch_size);

  return;
}

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename VertexUpdateType>
void VertexUpdateManager<VertexDataType, VertexUpdateType>::duplicateInGraphChecking(const std::unique_ptr<MemoryManager>& memory_manager, 
                                                                                     const std::shared_ptr<Config>& config)
{
  int batch_size = vertex_insertion_updates->vertex_data.size();
  int block_size = config->testruns_.at(config->testrun_index_)->params->insert_launch_block_size_;
  int grid_size = (batch_size / block_size) + 1;


  // Check Duplicates within graph-batch
  faimGraphVertexInsertion::d_duplicateInGraphChecking <VertexDataType, VertexUpdateType> << < grid_size, block_size >> > (vertex_insertion_updates->d_vertex_data,
                                                                                                 batch_size,
                                                                                                 (MemoryManager*)memory_manager->d_memory,
                                                                                                 memory_manager->d_data,
                                                                                                 false);
  return;
}
