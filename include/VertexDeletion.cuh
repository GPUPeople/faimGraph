//------------------------------------------------------------------------------
// VertexDeletion.cu
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


namespace faimGraphVertexDeletion
{
	//------------------------------------------------------------------------------
	// Device funtionality
	//------------------------------------------------------------------------------
	//

	//------------------------------------------------------------------------------
	// TODO: Exhibits problems for bad scheduling, rewrite needed as detailed in
	// issue: https://bitbucket.org/gpupeople/gpustreaminggraphs/issues/5/d_hostidtodeviceid-probably-needs-a
	template <typename VertexDataType, typename VertexUpdateType>
	__global__ void d_hostIDToDeviceID(VertexUpdateType* vertex_update_data,
									int batch_size,
									MemoryManager* memory_manager,
									memory_t* memory)
	{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= memory_manager->next_free_vertex_index)
		return;

	// Translate host identifier to device identifier, vertex_update_data contains host identifier
	VertexDataType* vertices = (VertexDataType*)memory;
	index_t vertex_host_ID = vertices[tid].host_identifier;

	if (vertex_host_ID == DELETIONMARKER)
	{
		return;
	}

	for (int i = 0; i < batch_size; ++i)
	{
		if (vertex_update_data[i].identifier == vertex_host_ID)
		{
		atomicExch(&(vertex_update_data[i].identifier), tid);
		}
	}
	return;
	}

	//------------------------------------------------------------------------------
	//
	template <typename VertexDataType, typename EdgeDataType>
	__global__ void d_deleteVertexMentionsSorted(MemoryManager* memory_manager,
												memory_t* memory,
												int page_size,
												int batch_size,
												VertexUpdate* vertex_update_data)
	{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= memory_manager->next_free_vertex_index)
		return;

	// Updates are sorted
	VertexDataType* vertices = (VertexDataType*)memory;
	VertexDataType vertex = vertices[tid];

	if (vertex.host_identifier == DELETIONMARKER)
	{
		// There is no valid vertex here anymore
		return;
	}

	// Let's search the adjacency for hits
	AdjacencyIterator<EdgeDataType> adjacency_iterator(pageAccess<EdgeDataType>(memory, vertex.mem_index, page_size, memory_manager->start_index));

	for (int i = 0; i < vertex.neighbours; ++i)
	{
		vertex_t adj_dest = adjacency_iterator.getDestination();
		if (adj_dest == DELETIONMARKER)
		continue;

		// Check if this edge was deleted through its vertex
		d_binarySearch(vertex_update_data, adj_dest, batch_size, adjacency_iterator);

		adjacency_iterator.advanceIterator(i, memory_manager->edges_per_page, memory, page_size, memory_manager->start_index);
	}

	return;
	}

	//------------------------------------------------------------------------------
	//
	template <typename VertexDataType, typename EdgeDataType>
	__global__ void d_deleteVertexMentions(MemoryManager* memory_manager,
										memory_t* memory,
										int page_size,
										int batch_size,
										VertexUpdate* vertex_update_data)
	{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= memory_manager->next_free_vertex_index)
		return;

	VertexDataType* vertices = (VertexDataType*)memory;
	VertexDataType vertex = vertices[tid];

	if (vertex.host_identifier == DELETIONMARKER)
	{
		// There is no valid vertex here anymore
		return;
	}

	// Let's search the adjacency for hits
	AdjacencyIterator<EdgeDataType> adjacency_iterator(pageAccess<EdgeDataType>(memory, vertex.mem_index, page_size, memory_manager->start_index));

	for (int i = 0; i < vertex.neighbours; ++i)
	{
		vertex_t adj_dest = adjacency_iterator.getDestination();
		if (adj_dest == DELETIONMARKER)
		continue;

		// Check if this edge was deleted through its vertex
		for (int j = 0; j < batch_size; ++j)
		{
		if (adj_dest == vertex_update_data[j].identifier)
		{
			adjacency_iterator.setDestination(DELETIONMARKER);
			break;
		}
		}
		adjacency_iterator.advanceIterator(i, memory_manager->edges_per_page, memory, page_size, memory_manager->start_index);
	}

	return;
	}

	//------------------------------------------------------------------------------
	//
	template <typename VertexDataType, typename EdgeDataType>
	__global__ void d_vertexDeletion(MemoryManager* memory_manager,
									memory_t* memory,
									int number_vertices,
									int page_size,
									int batch_size,
									index_t* device_mapping,
									index_t* device_mapping_update,
									VertexUpdate* vertex_update_data,
									GraphDirectionality graph_directionality)
	{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= batch_size)
		return;

	// First retrieve the vertex to delete
	//########################################################################################################################
	// We assume at this point that we get a device identifier (as the host has the corresponding mapping)
	// Should this NOT be the case, we need a kernel that does the translation ahead of this call (d_hostIDToDeviceID)
	//########################################################################################################################
	VertexUpdate vertex_update = vertex_update_data[tid];
	VertexDataType* vertices = (VertexDataType*)(memory);
	VertexDataType vertex;

	// Is it a valid index?
	if (vertex_update.identifier >= memory_manager->next_free_vertex_index)
	{
		// We are out of range, the index is not valid
		device_mapping_update[tid] = DELETIONMARKER;
		return;
	}

	// Since we could get duplicates in the batch, let's make sure that only one thread actually deletes the vertex using Atomics
	index_t host_identifier{ 0 };
	if ((host_identifier = atomicExch(&(vertices[vertex_update.identifier].host_identifier), DELETIONMARKER)) == DELETIONMARKER)
	{
		// Another thread is doing the work, we can return
		device_mapping_update[tid] = DELETIONMARKER;
		return;
	}

	// Now we should be the only thread modifying this particular vertex,
	// the only thing left is to return the vertex to the index queue,
	// and return all its pages to the queue as well
	vertex_t edges_per_page = memory_manager->edges_per_page;
	vertex = vertices[vertex_update.identifier];
	index_t page_index = vertex.mem_index;
	AdjacencyIterator<EdgeDataType> adjacency_iterator(pageAccess<EdgeDataType>(memory, page_index, page_size, memory_manager->start_index));

	// Return all blocks to the queue
	if (graph_directionality == GraphDirectionality::DIRECTED)
	{
		while (vertex.capacity > 0)
		{
		memory_manager->d_page_queue.enqueue(page_index);
		adjacency_iterator.blockTraversalAbsolute(edges_per_page, memory, page_size, memory_manager->start_index, page_index);
		vertex.capacity -= edges_per_page;
		}
	}
	else
	{
		// We can delete all the edges in other adjacencies right here right here
		AdjacencyIterator<EdgeDataType> deletion_iterator;
		while (vertex.capacity > 0)
		{
		memory_manager->d_page_queue.enqueue(page_index);
		for (int i = 0; i < edges_per_page; ++i)
		{
			vertex_t adj_destination = adjacency_iterator.getDestinationAt(i);
			if (adj_destination == DELETIONMARKER)
			continue;

			// We got a valid edge, delete it now
			deletion_iterator.setIterator(pageAccess<EdgeDataType>(memory, vertices[adj_destination].mem_index, page_size, memory_manager->start_index));
			for (int j = 0; j < vertices[adj_destination].neighbours; ++j)
			{
			vertex_t running_vertex = deletion_iterator.getDestination();
			if (running_vertex == vertex_update.identifier)
			{
				deletion_iterator.setDestination(DELETIONMARKER);
				break;
			}
			deletion_iterator.advanceIterator(j, edges_per_page, memory, page_size, memory_manager->start_index);
			}
		}
		adjacency_iterator.blockTraversalAbsolute(edges_per_page, memory, page_size, memory_manager->start_index, page_index);
		vertex.capacity -= edges_per_page;
		}
	}

	// Last but not least, return the vertex index to the queue, the rest is dealt with in other kernels
	memory_manager->d_vertex_queue.enqueueAlternating(vertex_update.identifier);

	// Delete the mapping
	device_mapping[vertex_update.identifier] = DELETIONMARKER;
	device_mapping_update[tid] = host_identifier;
	// Increase number_vertices counter
	atomicSub(&(memory_manager->number_vertices), 1);

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
void VertexUpdateManager<VertexDataType, VertexUpdateType>::deviceVertexDeletion<EdgeDataType>(std::unique_ptr<MemoryManager>& memory_manager,
                                                                                                const std::shared_ptr<Config>& config,
                                                                                                VertexMapper<index_t, index_t>& mapper)
{
  int batch_size = vertex_deletion_updates->vertex_data.size();
  int block_size = config->testruns_.at(config->testrun_index_)->params->insert_launch_block_size_;
  int grid_size = (batch_size / block_size) + 1;

  cudaEvent_t ce_start, ce_stop;

  ScopedMemoryAccessHelper scoped_mem_access_counter(memory_manager.get(), sizeof(VertexUpdate) *  batch_size);

  // Copy Updates to the device
  HANDLE_ERROR(cudaMemcpy(vertex_deletion_updates->d_vertex_data,
                          vertex_deletion_updates->vertex_data.data(),
                          sizeof(VertexUpdate) * batch_size,
                          cudaMemcpyHostToDevice));

  if (config->testruns_.at(config->testrun_index_)->params->sorting_)
  {
    thrust::device_ptr<index_t> th_vertex_updates((index_t*)(vertex_deletion_updates->d_vertex_data));
    thrust::sort(th_vertex_updates, th_vertex_updates + batch_size);
  }

  start_clock(ce_start, ce_stop);
  faimGraphVertexDeletion::d_vertexDeletion <VertexDataType, EdgeDataType> << < grid_size, block_size >> > ((MemoryManager*)memory_manager->d_memory,
																											memory_manager->d_data,
																											memory_manager->number_vertices,
																											memory_manager->page_size,
																											batch_size,
																											mapper.d_device_mapping,
																											mapper.d_device_mapping_update,
																											vertex_deletion_updates->d_vertex_data,
																											memory_manager->graph_directionality);

  time_deletion += end_clock(ce_start, ce_stop);
  // Now go over the whole graph
  grid_size = (memory_manager->next_free_vertex_index / block_size) + 1;

  start_clock(ce_start, ce_stop);
  if (memory_manager->graph_directionality == GraphDirectionality::DIRECTED)
  {
    if (config->testruns_.at(config->testrun_index_)->params->sorting_)
    {
      // We need to delete all other mentions of the updates
      faimGraphVertexDeletion::d_deleteVertexMentionsSorted<VertexDataType, EdgeDataType> << < grid_size, block_size >> >((MemoryManager*)memory_manager->d_memory,
																														memory_manager->d_data,
																														memory_manager->page_size,
																														batch_size,
																														vertex_deletion_updates->d_vertex_data);
    }
    else
    {
      // We need to delete all other mentions of the updates
      faimGraphVertexDeletion::d_deleteVertexMentions<VertexDataType, EdgeDataType> << < grid_size, block_size >> >((MemoryManager*)memory_manager->d_memory,
																													memory_manager->d_data,
																													memory_manager->page_size,
																													batch_size,
																													vertex_deletion_updates->d_vertex_data);
    }
  }
  time_vertex_mentions += end_clock(ce_start, ce_stop);
  // We need to compact the graph here still
  start_clock(ce_start, ce_stop);
  memory_manager->compaction<VertexDataType, EdgeDataType>(config);
  time_compaction += end_clock(ce_start, ce_stop);

  updateMemoryManagerHost(memory_manager);

  // Copy Updates back from the device
  HANDLE_ERROR(cudaMemcpy(mapper.h_device_mapping.data(),
                          mapper.d_device_mapping,
                          sizeof(index_t) * mapper.h_device_mapping.size(),
                          cudaMemcpyDeviceToHost));

  HANDLE_ERROR(cudaMemcpy(mapper.h_device_mapping_update.data(),
                          mapper.d_device_mapping_update,
                          sizeof(index_t) * mapper.h_device_mapping_update.size(),
                          cudaMemcpyDeviceToHost));

  return;
}

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename EdgeUpdateType>
void faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, EdgeUpdateType>::vertexDeletion(VertexMapper<index_t, index_t>& mapper)
{
  vertex_update_manager->template deviceVertexDeletion<EdgeDataType>(memory_manager, config, mapper);
}
