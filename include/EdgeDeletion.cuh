//------------------------------------------------------------------------------
// EdgeDeletion.cu
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

#include "EdgeUpdate.h"
#include "faimGraph.h"
#include "MemoryManager.h"
#include "ConfigurationParser.h"

namespace faimGraphEdgeDeletion
{
	//------------------------------------------------------------------------------
	// Device funtionality
	//------------------------------------------------------------------------------
	//

	//------------------------------------------------------------------------------
	//
	template <typename VertexDataType, typename EdgeDataType, typename UpdateDataType>
	__global__ void d_edgeDeletionVertexCentric(MemoryManager* memory_manager,
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
	vertex_t neighbours = vertices[tid].neighbours;

	if (neighbours == 0)
		return;

	int capacity = vertices[tid].capacity;
	AdjacencyIterator<EdgeDataType> adjacency_iterator(pageAccess<EdgeDataType>(memory, vertices[tid].mem_index, page_size, memory_manager->start_index));
	AdjacencyIterator<EdgeDataType> compaction_iterator(nullptr);
	index_t compaction_index = 0, edge_block_index = INVALID_INDEX, potential_edge_block_index = INVALID_INDEX;

	index_t actual_updates = number_updates;
	// Iterate over adjacency to find edges to delete
	for (int i = 0; i < neighbours; ++i)
	{
		vertex_t adj_dest = adjacency_iterator.getDestination();
		if (d_binarySearch(edge_update_data, adj_dest, index_offset, actual_updates, adjacency_iterator))
		{
		--number_updates;
		// We can delete an element
		if (compaction_iterator.getIterator() == nullptr)
		{
			// Remember the first position in need of an update
			compaction_iterator.setIterator(adjacency_iterator);
			compaction_index = i;
			if (((i) % (edges_per_page)) == 0)
			{
			potential_edge_block_index = edge_block_index;
			}
		}
		}

		// If all edges are gone, we can break
		if (number_updates == 0)
		{
		break;
		}

		adjacency_iterator.advanceIteratorEndCheck(i, edges_per_page, memory, page_size, memory_manager->start_index, capacity, edge_block_index);
	}

	// Now edges are gone, but we still need compaction
	index_t processed_updates = actual_updates - number_updates;

	if (processed_updates == 0)
	{
		// No updates were deleted
		return;
	}
	else if (compaction_index == (neighbours - processed_updates) &&
		((compaction_index) % (edges_per_page)) == 0 &&
		potential_edge_block_index != INVALID_INDEX)
	{
		// This is one hell of a case xD
		/* Explanation: This case deals with one edge case, to be precise: If the first item we want to delete is also
		the first item on a new block and we want to delete all elements after this item
		In this case, the other cases would not free the first block as they won't see it as part of their algorithm
		*/
	#ifdef QUEUING
		if (memory_manager->d_page_queue.enqueue(potential_edge_block_index))
		{
		vertices[tid].capacity -= edges_per_page;
		}    
		else
		{
		atomicOr(&(memory_manager->error_code), static_cast<unsigned int>(ErrorCode::PAGE_QUEUE_FULL));
		}
	#endif
	}

	number_updates = processed_updates;

	// Go to the element (neighbours - processed_updates)
	adjacency_iterator.setIterator(compaction_iterator);
	for (int i = compaction_index; i < (neighbours - processed_updates); ++i)
	{
		if (adjacency_iterator.advanceIteratorEndCheckBoolReturn(i, edges_per_page, memory, page_size, memory_manager->start_index, capacity, edge_block_index))
		{
		if (i == (neighbours - processed_updates - 1))
		{
			// Seemingly complicated stuff, but here is the reasoning
			// We have to query this here additionally, because if the first element we want to use for compaction
			// is also the first element in a block, this means that we will free this block for sure
			// But after the traversal, we don't have access to the index any longer, hence we need to do it here
			// So if the last step of the traversal also requires switching to a new block, this new block will be
			// freed later for sure and we can/have to do it here!


			// We can return the next block
	#ifdef QUEUING
			if (memory_manager->d_page_queue.enqueue(edge_block_index))
			{
			vertices[tid].capacity -= edges_per_page;
			}       
			else
			{
			atomicOr(&(memory_manager->error_code), static_cast<unsigned int>(ErrorCode::PAGE_QUEUE_FULL));
			}
	#endif
		}
		}
	}

	while (true)
	{
		while (processed_updates > 0)
		{
		// Move compaction list to next item that should be deleted
		for (int i = compaction_index; i < neighbours && compaction_iterator.getDestination() != DELETIONMARKER; ++i)
		{
			++compaction_index;
			compaction_iterator.advanceIteratorEndCheck(i, edges_per_page, memory, page_size, memory_manager->start_index, capacity);
		}

		// Move adjacency to next item that we can copy to the deletion position
		for (int i = (neighbours - processed_updates); i < neighbours && adjacency_iterator.getDestination() == DELETIONMARKER; ++i)
		{
			--processed_updates;
			if (adjacency_iterator.advanceIteratorEndCheckBoolReturn(i, edges_per_page, memory, page_size, memory_manager->start_index, capacity, edge_block_index))
			{
			// We can return the next block
	#ifdef QUEUING
			if (memory_manager->d_page_queue.enqueue(edge_block_index))
			{
				vertices[tid].capacity -= edges_per_page;
			}  
			else
			{
				atomicOr(&(memory_manager->error_code), static_cast<unsigned int>(ErrorCode::PAGE_QUEUE_FULL));
			}
	#endif
			}
		}

		if (processed_updates == 0)
		{
			break;
		}

		compaction_iterator.setDestination(adjacency_iterator);
		adjacency_iterator.setDestination(DELETIONMARKER);
		}

		if (processed_updates == 0)
		{
		// Then we are done
		vertices[tid].neighbours -= number_updates;
		break;
		}
	}

	return;
	}

	//------------------------------------------------------------------------------
	// This update mode relies on sorted update data as well as sorted adjacency data
	// Compaction should retain the sorted order of the given data
	// Duplicates within batch should be handled correctly
	//
	template <typename VertexDataType, typename EdgeDataType, typename UpdateDataType>
	__global__ void d_edgeDeletionVertexCentricSorted(MemoryManager* memory_manager,
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
	index_t actual_updates = 0;

	if (number_updates == 0)
		return;

	// Now just threads that actually work on updates should be left, tid corresponds to the src vertex that is being modified
	// Gather pointer
	VertexDataType* vertices = (VertexDataType*)memory;
	edge_update_data += index_offset;
	vertex_t edges_per_page = memory_manager->edges_per_page;
	vertex_t neighbours = vertices[tid].neighbours;
	int j = 0;

	if (neighbours == 0)
		return;

	AdjacencyIterator<EdgeDataType> adjacency_iterator(pageAccess<EdgeDataType>(memory, vertices[tid].mem_index, page_size, memory_manager->start_index));
	AdjacencyIterator<EdgeDataType> compaction_iterator(nullptr);

	for (int i = 0; i < neighbours; ++i)
	{
		// Go over adjacency
		if (adjacency_iterator.getDestination() >= edge_update_data->update.destination)
		{
		while (adjacency_iterator.getDestination() > edge_update_data->update.destination && number_updates > 0)
		{
			++edge_update_data;
			--number_updates;
		}
		if (number_updates == 0)
		{
			// Last update, just finish compaction
			--edge_update_data;
			edge_update_data->update.destination = DELETIONMARKER;
		}

		if (adjacency_iterator.getDestination() == edge_update_data->update.destination)
		{
			if (compaction_iterator.isNotValid())
			{
			// Set compaction iterator after first deletion and remember index
			compaction_iterator.setIterator(adjacency_iterator);
			j = i;
			// If the first time is also the last time, we need to delete this element here
			compaction_iterator.setDestination(DELETIONMARKER);
			}

			// Advance the update status
			if (number_updates > 0)
			{
			// Edge update handled
			++edge_update_data;
			--number_updates;
			++actual_updates;
			}
			if (number_updates == 0)
			{
			// Last update, just finish compaction
			--edge_update_data;
			edge_update_data->update.destination = DELETIONMARKER;
			}
			// This element should be deleted, move iterator over and start loop again
			adjacency_iterator.advanceIterator(i, edges_per_page, memory, page_size, memory_manager->start_index);
			continue;
		}
		}
		
		if (compaction_iterator.isValid() && i < neighbours)
		{
		compaction_iterator.setDestination(adjacency_iterator);
		compaction_iterator.advanceIterator(j, edges_per_page, memory, page_size, memory_manager->start_index);
		++j;
		}
		
		adjacency_iterator.advanceIterator(i, edges_per_page, memory, page_size, memory_manager->start_index);
	}

	// Go to beginning of currently last page and check if we need to return some pages
	vertex_t positions_available = vertices[tid].capacity - (neighbours - actual_updates);
	compaction_iterator -= (j % edges_per_page);
	while (positions_available > edges_per_page)
	{
		// We can return pages to the memory manager
		compaction_iterator.blockTraversalAbsolute(edges_per_page, memory, page_size, memory_manager->start_index, index_offset);
		if (!memory_manager->d_page_queue.enqueue(index_offset))
		{
		atomicOr(&(memory_manager->error_code), static_cast<unsigned int>(ErrorCode::PAGE_QUEUE_FULL));
		break;
		}
		positions_available -= edges_per_page;
		vertices[tid].capacity -= edges_per_page;
	}

	// Write configuration changes to memory
	vertices[tid].neighbours -= actual_updates;

	return;
	}


	constexpr int MULTIPLICATOR {4};
	//------------------------------------------------------------------------------
	//
	template <typename VertexDataType, typename EdgeDataType, typename UpdateDataType>
	__global__ void d_edgeDeletionWarpSized(MemoryManager* memory_manager,
											memory_t* memory,
											int page_size,
											UpdateDataType* edge_update_data,
											int batch_size,
											ConfigurationParameters::DeletionVariant DELETION_VARIANT)
	{
	int warpID = threadIdx.x / WARPSIZE;
	int wid = (blockIdx.x * MULTIPLICATOR) + warpID;  
	int threadID = threadIdx.x - (warpID * WARPSIZE);
	vertex_t edges_per_page = memory_manager->edges_per_page;
	// Outside threads per block (because of indexing structure we use 31 threads)
	if ((threadID >= edges_per_page) || (wid >= batch_size))
		return;

	// Gather pointer
	volatile VertexDataType* vertices = (VertexDataType*)memory;

	// Shared variables per block to determine index
	__shared__ index_t deletion_index[MULTIPLICATOR];
	__shared__ AdjacencyIterator<EdgeDataType> adjacency_iterator[MULTIPLICATOR], search_iterator[MULTIPLICATOR];
	__shared__ index_t edge_block_index[MULTIPLICATOR];
	__shared__ vertex_t shuffle_index[MULTIPLICATOR];
	__shared__ UpdateDataType edge_update[MULTIPLICATOR];

	__syncwarp();
	if (SINGLE_THREAD_MULTI)
	{
		edge_update[warpID] = edge_update_data[wid];
		deletion_index[warpID] = INVALID_INDEX;
		adjacency_iterator[warpID].setIterator(pageAccess<EdgeDataType>(memory, vertices[edge_update[warpID].source].mem_index, page_size, memory_manager->start_index));
		search_iterator[warpID].setIterator(adjacency_iterator[warpID]);
		edge_block_index[warpID] = INVALID_INDEX;
		if (DELETION_VARIANT == ConfigurationParameters::DeletionVariant::COMPACTION)
		{
		while (atomicCAS((int*)&(vertices[edge_update[warpID].source].locking), UNLOCK, LOCK) != UNLOCK)__threadfence();
		}
	}
	__syncwarp();


	//------------------------------------------------------------------------------
	// TODO: Currently no support for adding/deleting new vertices!!!!!
	//------------------------------------------------------------------------------
	//
	if (edge_update[warpID].source >= memory_manager->next_free_vertex_index || edge_update[warpID].update.destination >= memory_manager->next_free_vertex_index)
	{
		if (DELETION_VARIANT == ConfigurationParameters::DeletionVariant::COMPACTION)
		{
		__syncwarp();
		if (SINGLE_THREAD_MULTI)
		{
			atomicExch((int*)&(vertices[edge_update[warpID].source].locking), UNLOCK);
		}
		__syncwarp();
		}

		return;
	}

	int neighbours = vertices[edge_update[warpID].source].neighbours;
	if (neighbours <= 0)
	{
		if (DELETION_VARIANT == ConfigurationParameters::DeletionVariant::COMPACTION)
		{
		__syncwarp();
		if (SINGLE_THREAD_MULTI)
		{
			atomicExch((int*)&(vertices[edge_update[warpID].source].locking), UNLOCK);
		}
		__syncwarp();
		}
		return;
	}
	int capacity = vertices[edge_update[warpID].source].capacity;
	if (SINGLE_THREAD_MULTI)
	{
		shuffle_index[warpID] = neighbours - 1;
	}
	__syncwarp();

	// Search edge that should be deleted, this loop uses the whole warp in parallel
	int round = 0;
	while (round < capacity && deletion_index[warpID] == INVALID_INDEX)
	{
		if (threadID == 0 && round != 0)
		{
		// First move adjacency to the last element = index of next block

		adjacency_iterator[warpID] += edges_per_page;
		edge_block_index[warpID] = adjacency_iterator[warpID].getPageIndex(edges_per_page);
		pointerHandlingTraverse(adjacency_iterator[warpID].getIterator(), memory, page_size, edges_per_page, memory_manager->start_index);
		search_iterator[warpID].setIterator(adjacency_iterator[warpID]);
		shuffle_index[warpID] -= edges_per_page;
		}
		// Sync so that everythread has the correct adjacencylist
		__syncwarp();

		// Check if we found a match
		if (adjacency_iterator[warpID].getDestinationAt(threadID) == edge_update[warpID].update.destination)
		{
		// Just save the index, as the adjacency will be moved along accordingly
		deletion_index[warpID] = threadID;
		}
		round += (edges_per_page);

		__syncwarp();
	}

	// If deletion_index is still -1, Nothing to do, edge does not exist
	if (deletion_index[warpID] == INVALID_INDEX)
	{
		if (DELETION_VARIANT == ConfigurationParameters::DeletionVariant::COMPACTION)
		{
		__syncwarp();
		if (SINGLE_THREAD_MULTI)
		{
			atomicExch((int*)&(vertices[edge_update[warpID].source].locking), UNLOCK);
		}
		__syncwarp();
		}
		return;
	}

	// So we have found the corresponding edge, now let's delete it
	__syncwarp();
	if (SINGLE_THREAD_MULTI)
	{
		//##############################################################################################################
		if (DELETION_VARIANT == ConfigurationParameters::DeletionVariant::COMPACTION)
		{
		//##############################################################################################################
		// Find value at index neighbours
		// Move to the last item in the adjacency
		search_iterator[warpID].advanceIteratorToIndex(edges_per_page, memory, page_size, memory_manager->start_index, edge_block_index[warpID], shuffle_index[warpID]);


		// Copy last value onto the value that should be deleted and delete last value
		adjacency_iterator[warpID].setDestinationAt(deletion_index[warpID], search_iterator[warpID].getDestinationAt(shuffle_index[warpID]));
		search_iterator[warpID].setDestinationAt(shuffle_index[warpID], DELETIONMARKER);
		vertices[edge_update[warpID].source].neighbours -= 1;

		// Can we return a block to the queue?
		if ((shuffle_index[warpID]) == 0)
		{
			// We can return this block to the queue if it's not INVALID (we always want to have one remaining block)
			if (edge_block_index[warpID] != INVALID_INDEX)
			{
	#ifdef QUEUING
			if (memory_manager->d_page_queue.enqueue(edge_block_index[warpID]))
			{
				vertices[edge_update[warpID].source].capacity -= edges_per_page;
			}
			else
			{
				atomicOr(&(memory_manager->error_code), static_cast<unsigned int>(ErrorCode::PAGE_QUEUE_FULL));
			}
	#endif
			}
		}

		__threadfence();

		atomicExch((int*)&(vertices[edge_update[warpID].source].locking), UNLOCK);
		//##############################################################################################################
		}
		//##############################################################################################################
		//##############################################################################################################
		else
		{
		//##############################################################################################################
		vertex_t retVal = atomicCAS(adjacency_iterator[warpID].getDestinationPtrAt(deletion_index[warpID]), edge_update[warpID].update.destination, DELETIONMARKER);
		if (retVal == edge_update[warpID].update.destination)
		{
			atomicSub((vertex_t*)&(vertices[edge_update[warpID].source].neighbours), 1);
		}
		//##############################################################################################################
		}
		//##############################################################################################################       
	}
	__syncwarp();

	return;
	}


	//------------------------------------------------------------------------------
	//
	template <typename VertexDataType, typename EdgeDataType, typename UpdateDataType>
	__global__ void d_edgeDeletion(MemoryManager* memory_manager,
								memory_t* memory,
								int page_size,
								UpdateDataType* edge_update_data,
								int batch_size,
								ConfigurationParameters::DeletionVariant DELETION_VARIANT)
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
	AdjacencyIterator<EdgeDataType> search_iterator(adjacency_iterator);

	if (DELETION_VARIANT == ConfigurationParameters::DeletionVariant::STANDARD)
	{
		//##############################################################################################################
		int capacity = vertices[edge_update.source].capacity;
		for (int i = 0; i < capacity; ++i)
		{
		if (adjacency_iterator.getDestination() == edge_update.update.destination)
		{
			vertex_t retVal = atomicExch(adjacency_iterator.getDestinationPtr(), DELETIONMARKER);
			if (retVal != DELETIONMARKER)
			{
			atomicSub((vertex_t*)&(vertices[edge_update.source].neighbours), 1);
			}
			break;
		}
		adjacency_iterator.advanceIterator(i, edges_per_page, memory, page_size, memory_manager->start_index);
		}

		//##############################################################################################################  
	}
	//##############################################################################################################

	//##############################################################################################################
	else if (DELETION_VARIANT == ConfigurationParameters::DeletionVariant::COMPACTION)
	{
		//##############################################################################################################
		bool leave_loop = false;
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
					if (memory_manager->d_page_queue.enqueue(edge_block_index))
					{
						vertices[edge_update.source].capacity -= edges_per_page;
					}
					else
					{
						atomicOr(&(memory_manager->error_code), static_cast<unsigned int>(ErrorCode::PAGE_QUEUE_FULL));
					}
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
		//##############################################################################################################  
	}
	//##############################################################################################################

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
void EdgeUpdateManager<VertexDataType, EdgeDataType, UpdateDataType>::deviceEdgeDeletion(std::unique_ptr<MemoryManager>& memory_manager,
                                                                                        const std::shared_ptr<Config>& config)
{
  int batch_size = updates->edge_update.size();
  int block_size = config->testruns_.at(config->testrun_index_)->params->delete_launch_block_size_;
  int grid_size = (batch_size / block_size) + 1;
  ScopedMemoryAccessHelper scoped_mem_access_counter(memory_manager.get(), sizeof(UpdateDataType) *  batch_size);

  // Copy updates to device if necessary
  TemporaryMemoryAccessHeap temp_memory_dispenser(memory_manager.get(), memory_manager->next_free_vertex_index, sizeof(VertexDataType));
  updates->d_edge_update = temp_memory_dispenser.getTemporaryMemory<UpdateDataType>(batch_size);
  HANDLE_ERROR(cudaMemcpy(updates->d_edge_update,
                          updates->edge_update.data(),
                          sizeof(UpdateDataType) * batch_size,
                          cudaMemcpyHostToDevice));
 

  ConfigurationParameters::DeletionVariant deletion_variant = config->testruns_.at(config->testrun_index_)->params->deletion_variant_;
  // Delete Edges using the standard approach ( 1 thread / 1 update)
  if (config->testruns_.at(config->testrun_index_)->params->update_variant_ == ConfigurationParameters::UpdateVariant::STANDARD)
  {
    faimGraphEdgeDeletion::d_edgeDeletion<VertexDataType, EdgeDataType, UpdateDataType> << < grid_size, block_size >> > ((MemoryManager*)memory_manager->d_memory,
                                                                                                  memory_manager->d_data,
                                                                                                  memory_manager->page_size,
                                                                                                  updates->d_edge_update,
                                                                                                  batch_size,
                                                                                                  deletion_variant);
  }
  // Delete Edges using the warpsized approach ( 1 warp / 1 update)
  else if (config->testruns_.at(config->testrun_index_)->params->update_variant_ == ConfigurationParameters::UpdateVariant::WARPSIZED)
  {
    block_size = WARPSIZE * faimGraphEdgeDeletion::MULTIPLICATOR;
    grid_size = (batch_size / faimGraphEdgeDeletion::MULTIPLICATOR) + 1;
    faimGraphEdgeDeletion::d_edgeDeletionWarpSized<VertexDataType, EdgeDataType, UpdateDataType> << < grid_size, block_size >> > ((MemoryManager*)memory_manager->d_memory,
                                                                                                            memory_manager->d_data,
                                                                                                            memory_manager->page_size,
                                                                                                            updates->d_edge_update,
                                                                                                            batch_size,
                                                                                                            deletion_variant);
  }
  // Deletes edges where each threads handles all updates to one src vertex ( 1 thread / 1 vertex)
  else if (config->testruns_.at(config->testrun_index_)->params->update_variant_ == ConfigurationParameters::UpdateVariant::VERTEXCENTRIC)
  {
    grid_size = (memory_manager->next_free_vertex_index / block_size) + 1;
 
     thrust::device_ptr<UpdateDataType> th_edge_updates((UpdateDataType*)(updates->d_edge_update));
     thrust::sort(th_edge_updates, th_edge_updates + batch_size);
   
    auto preprocessed = edgeUpdatePreprocessing(memory_manager, config);

    faimGraphEdgeDeletion::d_edgeDeletionVertexCentric<VertexDataType, EdgeDataType, UpdateDataType> << < grid_size, block_size >> > ((MemoryManager*)memory_manager->d_memory,
                                                                                                               memory_manager->d_data,
                                                                                                               memory_manager->page_size,
                                                                                                               updates->d_edge_update,
                                                                                                               batch_size,
                                                                                                               preprocessed->d_update_src_offsets);
  }
  // Deletes edges where each threads handles all updates to one src vertex in sorted fashion ( 1 thread / 1 vertex)
  else if (config->testruns_.at(config->testrun_index_)->params->update_variant_ == ConfigurationParameters::UpdateVariant::VERTEXCENTRICSORTED)
  {
    grid_size = (memory_manager->next_free_vertex_index / block_size) + 1;

    thrust::device_ptr<UpdateDataType> th_edge_updates((UpdateDataType*)(updates->d_edge_update));
    thrust::sort(th_edge_updates, th_edge_updates + batch_size);

    auto preprocessed = edgeUpdatePreprocessing(memory_manager, config);

    faimGraphEdgeDeletion::d_edgeDeletionVertexCentricSorted<VertexDataType, EdgeDataType, UpdateDataType> << < grid_size, block_size >> > ((MemoryManager*)memory_manager->d_memory,
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
void faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, EdgeUpdateType>::edgeDeletion()
{
  edge_update_manager->deviceEdgeDeletion(memory_manager, config);
}

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType, typename UpdateDataType>
void EdgeUpdateManager<VertexDataType, EdgeDataType, UpdateDataType>::w_edgeDeletion(cudaStream_t& stream,
                    const std::unique_ptr<EdgeUpdateBatch<UpdateDataType>>& updates_deletion,
                    std::unique_ptr<MemoryManager>& memory_manager,
                    const std::shared_ptr<Config>& config,
                    int batch_size,
                    int block_size,
                    int grid_size)
{
	faimGraphEdgeDeletion::d_edgeDeletion<VertexDataType, EdgeDataType, UpdateDataType> << < grid_size, block_size, 0, stream >> > ((MemoryManager*)memory_manager->d_memory,
                                                                                                            memory_manager->d_data,
                                                                                                            memory_manager->page_size,
                                                                                                            updates_deletion->d_edge_update,
                                                                                                            batch_size,
                                                                                                            config->testruns_.at(config->testrun_index_)->params->deletion_variant_);
  updateMemoryManagerHost(memory_manager);
}
