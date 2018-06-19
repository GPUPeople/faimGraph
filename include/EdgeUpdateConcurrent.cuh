//------------------------------------------------------------------------------
// EdgeUpdate.cu
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

#include "EdgeUpdate.h"
#include "MemoryManager.h"
#include "ConfigurationParser.h"

namespace faimGraphEdgeUpdateConcurrent
{
	//------------------------------------------------------------------------------
	// Device funtionality
	//------------------------------------------------------------------------------
	//

	//------------------------------------------------------------------------------
	//
	template <typename VertexDataType, typename EdgeDataType, typename UpdateDataType>
	__global__ void d_edge_update_concurrent(MemoryManager* memory_manager,
											memory_t* memory,
											int page_size,
											UpdateDataType* insert_edge_dst_data,
											int insert_batch_size,
											int insert_grid_size,
											UpdateDataType* delete_edge_dst_data,
											int delete_batch_size,
											int delete_grid_size)
	{
	int tid;

	// Gather pointer
	volatile VertexDataType* vertices = (VertexDataType*)memory;
	vertex_t edges_per_page = memory_manager->edges_per_page;
	AdjacencyIterator<EdgeDataType> adjacency_iterator;
	AdjacencyIterator<EdgeDataType> search_iterator;
	bool leave_loop = false;

	// Insertion or Deletion
	if(blockIdx.x < insert_grid_size)
	{
		//------------------------------------------------------------------------------
		// Insertion
		//------------------------------------------------------------------------------
		tid = threadIdx.x + blockIdx.x*blockDim.x;
		if(tid >= insert_batch_size)
		return;

		// Get edge data
		UpdateDataType edge_update = insert_edge_dst_data[tid];

		if (edge_update.source >= memory_manager->next_free_vertex_index || edge_update.update.destination >= memory_manager->next_free_vertex_index)
		{
		return;
		}

		adjacency_iterator.setIterator(pageAccess<EdgeDataType>(memory, vertices[edge_update.source].mem_index, page_size, memory_manager->start_index));

		bool edge_inserted = false;
		bool duplicate_found = false;
		while (!leave_loop)
		{
		if (atomicExch((int*)&(vertices[edge_update.source].locking), LOCK) == UNLOCK)
		{
			int neighbours = vertices[edge_update.source].neighbours;
			int capacity = vertices[edge_update.source].capacity;

			for (int i = 0; i < capacity; ++i)
			{
			vertex_t adj_dest = adjacency_iterator.getDestination();

			// Duplicate Check
			if (adj_dest == edge_update.update.destination)
			{
				duplicate_found = true;
				break;
			}

			// Insert Check
			if (!edge_inserted && ((i == neighbours)))
			{
				search_iterator.setIterator(adjacency_iterator);
				edge_inserted = true;
				// Here we have checked all and taken the last element, we can break
				break;
			}

			if (!edge_inserted && ((adj_dest == DELETIONMARKER)))
			{
				search_iterator.setIterator(adjacency_iterator);
				edge_inserted = true;
			}

			adjacency_iterator.advanceIteratorEndCheck(i, edges_per_page, memory, page_size, memory_manager->start_index, capacity);
			}

			// If there was no space and no duplicate has been found
			if (!edge_inserted && !duplicate_found)
			{
			// Set index to next block and then reset adjacency list and insert edge
			index_t edge_block_index;
			index_t* edge_block_index_ptr = adjacency_iterator.getPageIndexPtr(edges_per_page);
	#ifdef QUEUING
			if (memory_manager->d_page_queue.dequeue(edge_block_index))
			{
				// We got something from the queue
				*edge_block_index_ptr = edge_block_index;
			}
			else
			{
	#endif
				// Queue is currently empty
				*edge_block_index_ptr = atomicAdd(&(memory_manager->next_free_page), 1);
	#ifdef ACCESS_METRICS
				atomicAdd(&(memory_manager->access_counter), 1);
	#endif
	#ifdef QUEUING
			}
	#endif

			adjacency_iterator.setIterator(pageAccess<EdgeDataType>(memory, *edge_block_index_ptr, page_size, memory_manager->start_index));
			updateAdjacency(adjacency_iterator.getIterator(), edge_update, edges_per_page);

			vertices[edge_update.source].neighbours += 1;
			vertices[edge_update.source].capacity += edges_per_page;

			// In the standard deletion model we need Deletionmarkers on the new block
			++adjacency_iterator;
			for (int i = 1; i < edges_per_page; ++i)
			{
				setDeletionMarker(adjacency_iterator.getIterator(), edges_per_page);
				++adjacency_iterator;
			}

			__threadfence();

			memory_manager->free_memory -= memory_manager->page_size;
			}
			else if (!duplicate_found)
			{
			// Finally insert edge if no duplicate was found
			updateAdjacency(search_iterator.getIterator(), edge_update, edges_per_page);
			vertices[edge_update.source].neighbours += 1;

			__threadfence();
			}

			leave_loop = true;
			atomicExch((int*)&(vertices[edge_update.source].locking), UNLOCK);
		}
		}
	}
	else
	{
		//------------------------------------------------------------------------------
		// Deletion
		//------------------------------------------------------------------------------
		tid = threadIdx.x + (blockIdx.x - insert_grid_size)*blockDim.x;
		if(tid >= delete_batch_size)
		return;

		// Get edge data
		UpdateDataType edge_update = delete_edge_dst_data[tid];

		if (edge_update.source >= memory_manager->next_free_vertex_index || edge_update.update.destination >= memory_manager->next_free_vertex_index)
		{
		return;
		}

		adjacency_iterator.setIterator(pageAccess<EdgeDataType>(memory, vertices[edge_update.source].mem_index, page_size, memory_manager->start_index));
		search_iterator.setIterator(adjacency_iterator);
		
		while (!leave_loop)
		{
		if (atomicExch((int*)&(vertices[edge_update.source].locking), LOCK) == UNLOCK)
		{
			int neighbours = vertices[edge_update.source].neighbours;
			if (neighbours > 0)
			{
			vertex_t shuffle_index = neighbours - 1;
			index_t edge_block_index = INVALID_INDEX;
			for (int i = 0; i < neighbours; ++i)
			{
				vertex_t adj_dest = adjacency_iterator.getDestination();
				if (adj_dest == edge_update.update.destination)
				{
				// Move to the last item in the adjacency
				search_iterator.advanceIteratorToIndex(edges_per_page, memory, page_size, memory_manager->start_index, edge_block_index, shuffle_index);

				// Copy last value onto the value that should be deleted and delete last value
				adjacency_iterator.setDestination(search_iterator.getDestinationAt(shuffle_index));
				search_iterator.setDestinationAt(shuffle_index, DELETIONMARKER);
				vertices[edge_update.source].neighbours -= 1;

				// Can we return a block to the queue?
				if ((shuffle_index) == 0)
				{
					// We can return this block to the queue if it's not INVALID (we always want to have one remaining block)
					if (edge_block_index != INVALID_INDEX)
					{
	#ifdef QUEUING
					memory_manager->d_page_queue.enqueue(edge_block_index);
					vertices[edge_update.source].capacity -= edges_per_page;
	#endif
					}
				}

				__threadfence();

				break;
				}
				adjacency_iterator.advanceIteratorDeletionCompaction(i, edges_per_page, memory, page_size, memory_manager->start_index, edge_block_index, search_iterator, shuffle_index);
			}
			}

			leave_loop = true;
			atomicExch((int*)&(vertices[edge_update.source].locking), UNLOCK);
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
template <typename VertexDataType, typename EdgeDataType, typename UpdateDataType>
void EdgeUpdateManager<VertexDataType, EdgeDataType, UpdateDataType>::deviceEdgeUpdateConcurrentStream(cudaStream_t& insertion_stream,
                                      cudaStream_t& deletion_stream,
                                      std::unique_ptr<MemoryManager>& memory_manager,
                                      const std::shared_ptr<Config>& config)
{
  int batch_size = updates_insertion->edge_update.size();
  int block_size = config->testruns_.at(config->testrun_index_)->params->insert_launch_block_size_;
  int grid_size = (batch_size / block_size) + 1;

  // Reserve space on device
  TemporaryMemoryAccessHeap temp_memory_dispenser(memory_manager.get(), memory_manager->next_free_vertex_index, sizeof(VertexDataType));
  updates_insertion->d_edge_update = temp_memory_dispenser.getTemporaryMemory<UpdateDataType>(updates_insertion->edge_update.size());
  updates_deletion->d_edge_update = temp_memory_dispenser.getTemporaryMemory<UpdateDataType>(updates_deletion->edge_update.size());

  // Async memcpy to device
  HANDLE_ERROR(cudaMemcpyAsync(updates_insertion->d_edge_update, 
                               updates_insertion->raw_edge_update, 
                               updates_insertion->edge_update.size() * sizeof(UpdateDataType), 
                               cudaMemcpyHostToDevice, 
                               insertion_stream));
                               
  HANDLE_ERROR(cudaMemcpyAsync(updates_deletion->d_edge_update,
                               updates_deletion->raw_edge_update,
                               updates_deletion->edge_update.size() * sizeof(UpdateDataType),
                               cudaMemcpyHostToDevice,
                               deletion_stream));

  ScopedMemoryAccessHelper scoped_mem_access_counter(memory_manager.get(), (sizeof(vertex_t) * batch_size * 2) + (sizeof(UpdateDataType) *  batch_size * 2));

  w_edgeInsertion(insertion_stream, updates_insertion, memory_manager, batch_size, block_size, grid_size);
  w_edgeDeletion(deletion_stream, updates_deletion, memory_manager, config, batch_size, block_size, grid_size);

  HANDLE_ERROR(cudaStreamSynchronize(insertion_stream));
  HANDLE_ERROR(cudaStreamSynchronize(deletion_stream));

  return;
}

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType, typename UpdateDataType>
void EdgeUpdateManager<VertexDataType, EdgeDataType, UpdateDataType>::deviceEdgeUpdateConcurrent(std::unique_ptr<MemoryManager>& memory_manager,
                                                                                                 const std::shared_ptr<Config>& config)
{
  int insert_batch_size = updates_insertion->edge_update.size();
  int delete_batch_size = updates_deletion->edge_update.size();
  int block_size = config->testruns_.at(config->testrun_index_)->params->insert_launch_block_size_;
  int insert_grid_size = (insert_batch_size / block_size) + 1;
  int delete_grid_size = (delete_batch_size / block_size) + 1;

  // Copy updates to device
  TemporaryMemoryAccessHeap temp_memory_dispenser(memory_manager.get(), memory_manager->next_free_vertex_index, sizeof(VertexDataType));
  updates_insertion->d_edge_update = temp_memory_dispenser.getTemporaryMemory<UpdateDataType>(insert_batch_size);
  updates_deletion->d_edge_update = temp_memory_dispenser.getTemporaryMemory<UpdateDataType>(delete_batch_size);
  
  HANDLE_ERROR(cudaMemcpy(updates_insertion->d_edge_update,
                          updates_insertion->edge_update.data(),
                          sizeof(UpdateDataType) * insert_batch_size,
                          cudaMemcpyHostToDevice));

  HANDLE_ERROR(cudaMemcpy(updates_deletion->d_edge_update,
                          updates_deletion->edge_update.data(),
                          sizeof(UpdateDataType) * delete_batch_size,
                          cudaMemcpyHostToDevice));

	faimGraphEdgeUpdateConcurrent::d_edge_update_concurrent<VertexDataType, EdgeDataType, UpdateDataType> << < insert_grid_size + delete_grid_size, block_size >> >((MemoryManager*)memory_manager->d_memory,
                                                                                                                        memory_manager->d_data,
                                                                                                                        memory_manager->page_size,
                                                                                                                        updates_insertion->d_edge_update,
                                                                                                                        insert_batch_size,
                                                                                                                        insert_grid_size,
                                                                                                                        updates_deletion->d_edge_update,
                                                                                                                        delete_batch_size,
                                                                                                                        delete_grid_size);

  updateMemoryManagerHost(memory_manager);

  return;
}
