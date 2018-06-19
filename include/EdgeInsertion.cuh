//------------------------------------------------------------------------------
// EdgeInsertion.cu
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
#include <thrust/sort.h>
#include <cstddef>

#include "EdgeUpdate.h"
#include "faimGraph.h"
#include "MemoryManager.h"
#include "ConfigurationParser.h"

namespace faimGraphEdgeInsertion
{
	//------------------------------------------------------------------------------
	// Device funtionality
	//------------------------------------------------------------------------------
	//

	//------------------------------------------------------------------------------
	//
	template <typename VertexDataType, typename EdgeDataType, typename UpdateDataType>
	__global__ void d_edgeInsertionVertexCentric(MemoryManager* memory_manager,
											memory_t* memory,
											int page_size,
											UpdateDataType* edge_update_data,
											int batch_size,
											index_t* update_src_offsets)
	{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= memory_manager->next_free_vertex_index)
		return;

	index_t index_offset = update_src_offsets[tid];
	index_t number_updates = update_src_offsets[tid + 1] - index_offset;

	if (number_updates == 0)
		return;

	// Now just threads that actually work on updates should be left, tid corresponds to the src vertex that is being modified
	// Gather pointer
	VertexDataType* vertices = (VertexDataType*)memory;

	vertex_t edges_per_page = memory_manager->edges_per_page;
	int neighbours = vertices[tid].neighbours;
	int capacity = vertices[tid].capacity;

	AdjacencyIterator<EdgeDataType> adjacency_iterator(pageAccess<EdgeDataType>(memory, vertices[tid].mem_index, page_size, memory_manager->start_index));

	index_t fresh_position = neighbours;
	adjacency_iterator.advanceIteratorToIndex(edges_per_page, memory, page_size, memory_manager->start_index, fresh_position, neighbours, capacity);

	while (true)
	{
		while (neighbours < capacity && number_updates > 0)
		{
		if(edge_update_data[index_offset + (number_updates - 1)].update.destination == DELETIONMARKER)
		{
			--number_updates;
			continue;
		}
		
		updateAdjacency(adjacency_iterator.getIterator(), edge_update_data[index_offset + (number_updates - 1)], edges_per_page);
		++adjacency_iterator;
		--number_updates;
		++neighbours;
		}
		if (number_updates == 0)
		{
		// Then we are done
		vertices[tid].neighbours = neighbours;
		vertices[tid].capacity = capacity;
	#ifdef CLEAN_PAGE
		while (neighbours < capacity)
		{
			// Set the rest of the new block to DELETIONMARKERS
			setDeletionMarker(adjacency_iterator.getIterator(), edges_per_page);
			++adjacency_iterator;
			++neighbours;
		}
	#endif
		break;
		}
		else
		{
		// We need to get a new page and start all over again
		// Set index to next block and then reset adjacency list
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
		capacity += edges_per_page;
		}
	}

	return;
	}

	//------------------------------------------------------------------------------
	//
	template <typename VertexDataType, typename EdgeDataType, typename UpdateDataType>
	__global__ void d_edgeInsertionVertexCentricSorted(MemoryManager* memory_manager,
													memory_t* memory,
													int page_size,
													UpdateDataType* edge_update_data,
													int batch_size,
													index_t* update_src_offsets)
	{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= memory_manager->next_free_vertex_index)
		return;

	index_t index_offset = update_src_offsets[tid];
	index_t number_updates = update_src_offsets[tid + 1] - index_offset;

	if (number_updates == 0)
		return;

	// Now just threads that actually work on updates should be left, tid corresponds to the src vertex that is being modified
	// Gather pointer
	VertexDataType vertex = ((VertexDataType*)memory)[tid];
	UpdateDataType swap_element;
	edge_update_data += index_offset;
	vertex_t edges_per_page = memory_manager->edges_per_page;
	index_t internal_index;

	AdjacencyIterator<EdgeDataType> adjacency_iterator(pageAccess<EdgeDataType>(memory, vertex.mem_index, page_size, memory_manager->start_index));

	for (int i = 0; i < vertex.neighbours; ++i)
	{
		internal_index = 0;
		if (adjacency_iterator.getDestination() >= edge_update_data->update.destination)
		{
		if (adjacency_iterator.getDestination() == edge_update_data->update.destination)
		{
			// Duplicate to graph, skip these (may be multiple if both duplicate to graph and in batch)
			while (number_updates > 0 && edge_update_data->update.destination == adjacency_iterator.getDestination())
			{
			++edge_update_data;
			--number_updates;
			}
			// We are done
			if (number_updates == 0)
			{
			return;
			}
		}
		else
		{
			// Check duplicates in batch
			while (number_updates > 1 && edge_update_data[0].update.destination == edge_update_data[1].update.destination)
			{
			++edge_update_data;
			--number_updates;
			}

			// Edge Update data should be placed
			swapIntoLocal(adjacency_iterator.getIterator(), swap_element, edges_per_page);
			updateAdjacency(adjacency_iterator.getIterator(), edge_update_data, edges_per_page);

			// Find correct position in update vector to place swap element
			while ((internal_index + 1) < number_updates && edge_update_data[internal_index + 1].update.destination < swap_element.update.destination)
			{
			edge_update_data[internal_index] = edge_update_data[internal_index + 1];
			++internal_index;
			}
			edge_update_data[internal_index] = swap_element;       
		}
		}
		adjacency_iterator.advanceIteratorEndCheck(i, edges_per_page, memory, page_size, memory_manager->start_index, vertex.capacity);
	}

	// Now the edge_update_data should hold all elements (in a sorted fashion) that still need to be inserted (including duplicates in batch, excluding duplicates to graph
	while (true)
	{
		while (vertex.neighbours < vertex.capacity && number_updates > 0)
		{
		if ((number_updates > 1) && (edge_update_data[0].update.destination == edge_update_data[1].update.destination))
		{
			--number_updates;
			++edge_update_data;
			continue;
		}
		updateAdjacency(adjacency_iterator.getIterator(), *edge_update_data, edges_per_page);
		++edge_update_data;
		++adjacency_iterator;
		--number_updates;
		++(vertex.neighbours);
		}
		if (number_updates == 0)
		{
		// Then we are done
		((VertexDataType*)memory)[tid].neighbours = vertex.neighbours;
		((VertexDataType*)memory)[tid].capacity = vertex.capacity;
	#ifdef CLEAN_PAGE
		while (vertex.neighbours < vertex.capacity)
		{
			// Set the rest of the new block to DELETIONMARKERS
			setDeletionMarker(adjacency_iterator.getIterator(), edges_per_page);
			++adjacency_iterator;
			++(vertex.neighbours);
		}
	#endif
		break;
		}
		else
		{
		// We need to get a new page and start all over again
		// Set index to next block and then reset adjacency list
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
		vertex.capacity += edges_per_page;
		}
	}

	return;
	}

	constexpr int MULTIPLICATOR {4};
	//------------------------------------------------------------------------------
	//
	template <typename VertexDataType, typename EdgeDataType, typename UpdateDataType>
	__global__ void d_edgeInsertionWarpSized(MemoryManager* memory_manager,
											memory_t* memory,
											int page_size,
											UpdateDataType* edge_update_data,
											int batch_size,
											int kernel_block_size)
	{
	int warpID = threadIdx.x / WARPSIZE;
	int wid = (blockIdx.x * MULTIPLICATOR) + warpID;  
	int threadID = threadIdx.x - (warpID * WARPSIZE);
	vertex_t edges_per_page = memory_manager->edges_per_page;
	// Outside threads per block (because of indexing structure we use 31 threads)
	if ((threadID >= edges_per_page) || (wid >= batch_size))
		return;

	volatile VertexDataType* vertices = (VertexDataType*)memory;

	// Shared variables per block to determine index
	__shared__ index_t insertion_index[MULTIPLICATOR];
	__shared__ AdjacencyIterator<EdgeDataType> adjacency_iterator[MULTIPLICATOR], insert_iterator[MULTIPLICATOR];
	__shared__ UpdateDataType edge_update[MULTIPLICATOR];
	__shared__ bool duplicate_found[MULTIPLICATOR];
	__shared__ int neighbours[MULTIPLICATOR], capacity[MULTIPLICATOR];

	// Perform individual updates
	if (SINGLE_THREAD_MULTI)
	{
		edge_update[warpID] = edge_update_data[wid];
		insertion_index[warpID] = INVALID_INDEX;
		adjacency_iterator[warpID].setIterator(pageAccess<EdgeDataType>(memory, vertices[edge_update[warpID].source].mem_index, page_size, memory_manager->start_index));
		insert_iterator[warpID].setIterator(nullptr);
		duplicate_found[warpID] = false;
		// LOCKING
		while (atomicCAS((int*)&(vertices[edge_update[warpID].source].locking), UNLOCK, LOCK) != UNLOCK)__threadfence();
		// LOCKING
		neighbours[warpID] = vertices[edge_update[warpID].source].neighbours;
		capacity[warpID] = vertices[edge_update[warpID].source].capacity;
		// ################ SYNC ################
		__syncwarp();
		// ################ SYNC ################
	}
	else
	{
		// ################ SYNC ################
		__syncwarp();
		// ################ SYNC ################
	}

	//------------------------------------------------------------------------------
	// TODO: Currently no support for adding/deleting new vertices!!!!!
	//------------------------------------------------------------------------------
	//
	if (edge_update[warpID].source >= memory_manager->next_free_vertex_index || edge_update[warpID].update.destination >= memory_manager->next_free_vertex_index)
	{
		if (SINGLE_THREAD_MULTI)
		{
		atomicExch((int*)&(vertices[edge_update[warpID].source].locking), UNLOCK);
		}
		// ################ SYNC ################
		__syncwarp();
		// ################ SYNC ################

		return;
	}

	// Search for a space for that edge, this loop uses the whole warp in parallel
	index_t i = threadID;
	int round = 0;
	while (round < capacity[warpID])
	{
		// Check if we found a match
		vertex_t adj = adjacency_iterator[warpID].getDestinationAt(i);
		if (adj == edge_update[warpID].update.destination)
		{
		duplicate_found[warpID] = true;
		}
		// ################ SYNC ################
		__syncwarp();
		// ################ SYNC ################
		if (duplicate_found[warpID])
		{
		break;
		}

		if ((adj == DELETIONMARKER || (i + round) >= neighbours[warpID]) && insert_iterator[warpID].getIterator() == nullptr)
		{
		// Just save the index, as the adjacency will be moved along accordingly
		index_t old_val = atomicMin(&(insertion_index[warpID]), i);
		if (old_val == INVALID_INDEX)
		{
			// Save the insert_list corresponding to the index
			insert_iterator[warpID].setIterator(adjacency_iterator[warpID]);
		}
		}
		round += (edges_per_page);
		// ################ SYNC ################
		__syncwarp();
		// ################ SYNC ################
		if (SINGLE_THREAD_MULTI && round < capacity[warpID])
		{
		// First move adjacency to the last element = index of next block
		adjacency_iterator[warpID].blockTraversalAbsolute(edges_per_page, memory, page_size, memory_manager->start_index);
		}
		// Sync so that everythread has the correct adjacencylist
		// ################ SYNC ################
		__syncwarp();
		// ################ SYNC ################
	}

	// After this loop, we have 3 cases
	// 1.) We have a valid index that we can use
	// 2.) We have a duplicate
	// 3.) We need more space

	if (SINGLE_THREAD_MULTI)
	{
		if (insertion_index[warpID] != INVALID_INDEX && !duplicate_found[warpID])
		{
		// Handle case 1.)
		insert_iterator[warpID] += insertion_index[warpID];
		updateAdjacency(insert_iterator[warpID].getIterator(), edge_update[warpID], edges_per_page);

		vertices[edge_update[warpID].source].neighbours += 1;

		__threadfence();

		}
		else if (!duplicate_found[warpID])
		{
		// Handle case 3.)
		// Move adjacency over last element to "index"
		adjacency_iterator[warpID] += edges_per_page;

		index_t edge_block_index;
		index_t* edge_block_index_ptr = adjacency_iterator[warpID].getPageIndexPtr(edges_per_page);
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

		adjacency_iterator[warpID].setIterator(pageAccess<EdgeDataType>(memory, *edge_block_index_ptr, page_size, memory_manager->start_index));
		updateAdjacency(adjacency_iterator[warpID].getIterator(), edge_update[warpID], edges_per_page);
		vertices[edge_update[warpID].source].neighbours += 1;
		vertices[edge_update[warpID].source].capacity += edges_per_page;

		// In the standard deletion model we need Deletionmarkers on the new block
		adjacency_iterator[warpID].cleanPageExclusive(edges_per_page);

		__threadfence();

		memory_manager->free_memory -= memory_manager->page_size;
		}
		// Case 2.) will automatically end up here
		atomicExch((int*)&(vertices[edge_update[warpID].source].locking), UNLOCK);

	}
	// ################ SYNC ################
	__syncwarp();
	// ################ SYNC ################    

	return;
	}


	//------------------------------------------------------------------------------
	//
	template <typename VertexDataType, typename EdgeDataType, typename UpdateDataType>
	__global__ void d_edgeInsertion(MemoryManager* memory_manager,
									memory_t* memory,
									int page_size,
									UpdateDataType* edge_update_data,
									int batch_size)
	{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= batch_size)
		return;

	// Gather pointer
	volatile VertexDataType* vertices = (VertexDataType*)memory;
	vertex_t edges_per_page = memory_manager->edges_per_page;

	// Get edge data
	UpdateDataType edge_update = edge_update_data[tid];

	if (edge_update.source >= memory_manager->next_free_vertex_index || edge_update.update.destination >= memory_manager->next_free_vertex_index)
	{
		return;
	}

	AdjacencyIterator<EdgeDataType> adjacency_iterator(pageAccess<EdgeDataType>(memory, vertices[edge_update.source].mem_index, page_size, memory_manager->start_index));
	AdjacencyIterator<EdgeDataType> insert_iterator;

	bool leave_loop_vertex = false;
	bool edge_inserted = false;
	bool duplicate_found = false;
	while (!leave_loop_vertex)
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
			insert_iterator.setIterator(adjacency_iterator);
			edge_inserted = true;
			// Here we have checked all and taken the last element, we can break
			break;
			}

			if (!edge_inserted && ((adj_dest == DELETIONMARKER)))
			{
			insert_iterator.setIterator(adjacency_iterator);
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
			updateAdjacency(insert_iterator.getIterator(), edge_update, edges_per_page);
			vertices[edge_update.source].neighbours += 1;

			__threadfence();
		}

		leave_loop_vertex = true;
		atomicExch((int*)&(vertices[edge_update.source].locking), UNLOCK);
		}
	}

	return;
	}


	//------------------------------------------------------------------------------
	//
	template <typename VertexDataType, typename EdgeDataType, typename UpdateDataType>
	__global__ void d_edgeInsertionTest(MemoryManager* memory_manager,
										memory_t* memory,
										int page_size,
										UpdateDataType* edge_update_data,
										int batch_size)
	{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= batch_size)
		return;

	// Gather pointer
	volatile VertexDataType* vertices = (VertexDataType*)memory;
	vertex_t edges_per_page = memory_manager->edges_per_page;

	// Get edge data
	UpdateDataType edge_update = edge_update_data[tid];

	if (edge_update.source >= memory_manager->next_free_vertex_index || edge_update.update.destination >= memory_manager->next_free_vertex_index)
	{
		return;
	}

	AdjacencyIterator<EdgeDataType> adjacency_iterator(pageAccess<EdgeDataType>(memory, vertices[edge_update.source].mem_index, page_size, memory_manager->start_index));
	AdjacencyIterator<EdgeDataType> insert_iterator;

	//EdgeBlock<EdgeDataType> edge_block_test = *((EdgeBlock<EdgeDataType>*)(adjacency_iterator.getIterator()));

	bool leave_loop_vertex = false;
	bool edge_inserted = false;
	bool duplicate_found = false;
	while (!leave_loop_vertex)
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
			insert_iterator.setIterator(adjacency_iterator);
			edge_inserted = true;
			// Here we have checked all and taken the last element, we can break
			break;
			}

			if (!edge_inserted && ((adj_dest == DELETIONMARKER)))
			{
			insert_iterator.setIterator(adjacency_iterator);
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
			updateAdjacency(insert_iterator.getIterator(), edge_update, edges_per_page);
			vertices[edge_update.source].neighbours += 1;

			__threadfence();
		}

		leave_loop_vertex = true;
		atomicExch((int*)&(vertices[edge_update.source].locking), UNLOCK);
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
void EdgeUpdateManager<VertexDataType, EdgeDataType, UpdateDataType>::deviceEdgeInsertion(std::unique_ptr<MemoryManager>& memory_manager,
                                                                                          const std::shared_ptr<Config>& config)
{
  int batch_size = updates->edge_update.size();
  int block_size = config->testruns_.at(config->testrun_index_)->params->insert_launch_block_size_;
  int grid_size = (batch_size / block_size) + 1;

  ScopedMemoryAccessHelper scoped_mem_access_counter(memory_manager.get(), sizeof(UpdateDataType) *  batch_size);

  // Copy updates to device
  TemporaryMemoryAccessHeap temp_memory_dispenser(memory_manager.get(), memory_manager->next_free_vertex_index, sizeof(VertexDataType));
  updates->d_edge_update = temp_memory_dispenser.getTemporaryMemory<UpdateDataType>(batch_size);
  HANDLE_ERROR(cudaMemcpy(updates->d_edge_update,
                          updates->edge_update.data(),
                          sizeof(UpdateDataType) * batch_size,
                          cudaMemcpyHostToDevice));

  ConfigurationParameters::DeletionVariant deletion_variant = config->testruns_.at(config->testrun_index_)->params->deletion_variant_;

  // Insert Edges using the standard approach ( 1 thread / 1 update)
  if (config->testruns_.at(config->testrun_index_)->params->update_variant_ == ConfigurationParameters::UpdateVariant::STANDARD)
  {
    faimGraphEdgeInsertion::d_edgeInsertion<VertexDataType, EdgeDataType, UpdateDataType> << < grid_size, block_size >> > ((MemoryManager*)memory_manager->d_memory,
                                                                                                    memory_manager->d_data,
                                                                                                    memory_manager->page_size,
                                                                                                    updates->d_edge_update,
                                                                                                    batch_size);
  }
  // Insert Edges using the warpsized approach ( 1 warp / 1 update)
  else if (config->testruns_.at(config->testrun_index_)->params->update_variant_ == ConfigurationParameters::UpdateVariant::WARPSIZED)
  {
    block_size = WARPSIZE * faimGraphEdgeInsertion::MULTIPLICATOR;
    grid_size = (batch_size / faimGraphEdgeInsertion::MULTIPLICATOR) + 1;
    faimGraphEdgeInsertion::d_edgeInsertionWarpSized<VertexDataType, EdgeDataType, UpdateDataType> << < grid_size, block_size >> > ((MemoryManager*)memory_manager->d_memory,
                                                                                                            memory_manager->d_data,
                                                                                                            memory_manager->page_size,
                                                                                                            updates->d_edge_update,
                                                                                                            batch_size,
                                                                                                            block_size);
  }
  // Insert Edges using the vertex centric approach ( 1 thread / 1 vertex)
  else if (config->testruns_.at(config->testrun_index_)->params->update_variant_ == ConfigurationParameters::UpdateVariant::VERTEXCENTRIC)
  {
    grid_size = (memory_manager->next_free_vertex_index / block_size) + 1;

    thrust::device_ptr<UpdateDataType> th_edge_updates((updates->d_edge_update));
    thrust::sort(th_edge_updates, th_edge_updates + batch_size);    
 
    auto preprocessed = edgeUpdatePreprocessing(memory_manager, config);

    edgeUpdateDuplicateChecking(memory_manager, config, preprocessed);

    faimGraphEdgeInsertion::d_edgeInsertionVertexCentric<VertexDataType, EdgeDataType, UpdateDataType> << < grid_size, block_size >> > ((MemoryManager*)memory_manager->d_memory,
                                                                                                               memory_manager->d_data,
                                                                                                               memory_manager->page_size,
                                                                                                               updates->d_edge_update,
                                                                                                               batch_size,
                                                                                                               preprocessed->d_update_src_offsets);
  }
  // Insert Edges using the vertex centric approach in sorted fashion ( 1 thread / 1 vertex)
  else if (config->testruns_.at(config->testrun_index_)->params->update_variant_ == ConfigurationParameters::UpdateVariant::VERTEXCENTRICSORTED)
  {
    grid_size = (memory_manager->next_free_vertex_index / block_size) + 1;

    thrust::device_ptr<UpdateDataType> th_edge_updates((updates->d_edge_update));
    thrust::sort(th_edge_updates, th_edge_updates + batch_size);

    auto preprocessed = edgeUpdatePreprocessing(memory_manager, config);

    faimGraphEdgeInsertion::d_edgeInsertionVertexCentricSorted<VertexDataType, EdgeDataType, UpdateDataType> << < grid_size, block_size >> > ((MemoryManager*)memory_manager->d_memory,
                                                                                                                      memory_manager->d_data,
                                                                                                                      memory_manager->page_size,
                                                                                                                      updates->d_edge_update,
                                                                                                                      batch_size,
                                                                                                                      preprocessed->d_update_src_offsets);
  }

  updateMemoryManagerHost(memory_manager);

  return;
}


//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename EdgeUpdateType>
void faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, EdgeUpdateType>::edgeInsertion()
{
  edge_update_manager->deviceEdgeInsertion(memory_manager, config);
}

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType, typename UpdateDataType>
void EdgeUpdateManager<VertexDataType, EdgeDataType, UpdateDataType>::w_edgeInsertion(cudaStream_t& stream,
                     const std::unique_ptr<EdgeUpdateBatch<UpdateDataType>>& updates_insertion,
                     std::unique_ptr<MemoryManager>& memory_manager,
                     int batch_size,
                     int block_size,
                     int grid_size)
{
	faimGraphEdgeInsertion::d_edgeInsertion<VertexDataType, EdgeDataType, UpdateDataType> << < grid_size, block_size, 0, stream >> > ((MemoryManager*)memory_manager->d_memory,
                                                                                                            memory_manager->d_data,
                                                                                                            memory_manager->page_size,
                                                                                                            updates_insertion->d_edge_update,
                                                                                                            batch_size);

  updateMemoryManagerHost(memory_manager);
}

