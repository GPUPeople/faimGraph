//------------------------------------------------------------------------------
// STC.cuh
//
// faimGraph
//
//------------------------------------------------------------------------------
//
#pragma once
#include <iostream>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "STC.h"
#include "MemoryManager.h"
#include "EdgeUpdate.h"

//------------------------------------------------------------------------------
// Device funtionality
//------------------------------------------------------------------------------
//

namespace faimGraphSTC
{
	//------------------------------------------------------------------------------
	//
	template <typename VertexDataType, typename EdgeDataType>
	__global__ void d_StaticTriangleCounting(MemoryManager* memory_manager,
											memory_t* memory,
											uint32_t* triangles,
											int number_vertices,
											int page_size)
	{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
		if (tid >= number_vertices)
			return;
		

	VertexDataType* vertices = (VertexDataType*)memory;
	vertex_t edges_per_page = memory_manager->edges_per_page;

	// Retrieve data
	vertex_t neighbours = vertices[tid].neighbours;
	AdjacencyIterator<EdgeDataType> adjacency_iterator(pageAccess<EdgeDataType>(memory, vertices[tid].mem_index, page_size, memory_manager->start_index));
	AdjacencyIterator<EdgeDataType> iterator_adjacency, running_adjacency_list, search_iterator;

	// Iterate over neighbours
	for(int i = 0; i < (neighbours - 1); ++i)
	{        
		// Retrieve each vertex index and for vertex index i, go over vertices i+1 to capacity
		// and check in every adjacency list of those vertices, if vertex i is included
		// Then we found a triangle            
		iterator_adjacency.setIterator(adjacency_iterator);
		vertex_t compare_value = adjacency_iterator.getDestination();
		if(compare_value > tid)
		{
		adjacency_iterator.advanceIterator(i, edges_per_page, memory, page_size, memory_manager->start_index);
		break;
		}
		
		// Loop over all capacity - i indices and test their adjacency list if i is in there
		iterator_adjacency.advanceIterator(i, edges_per_page, memory, page_size, memory_manager->start_index);
		for(int j = i + 1; j < neighbours; ++j)
		{
		vertex_t running_index = iterator_adjacency.getDestination();
		
		if(running_index > tid)
		{
			iterator_adjacency.advanceIterator(j, edges_per_page, memory, page_size, memory_manager->start_index);
			break;
		}
		running_adjacency_list.setIterator(pageAccess<EdgeDataType>(memory, vertices[running_index].mem_index, page_size, memory_manager->start_index));
		// Now iterate over individual adjacency list and search for the neighbour_index
		vertex_t running_neighbours = vertices[running_index].neighbours;   
		while(true)
		{
			search_iterator.setIterator(running_adjacency_list);
			if (running_neighbours > edges_per_page)
			{
			running_adjacency_list.blockTraversalAbsolute(edges_per_page, memory, page_size, memory_manager->start_index);
			if(running_adjacency_list.getDestination() <= compare_value)
				running_neighbours -= edges_per_page;
			else
				break;
			}
			else
			{
			break;
			}    
		}
		if(running_neighbours > edges_per_page)
		{
			running_neighbours = edges_per_page;
		}

		if(d_binarySearchOnPage(search_iterator, compare_value, running_neighbours))
		{
			atomicAdd(&triangles[tid], 1);
			atomicAdd(&triangles[compare_value], 1);
			atomicAdd(&triangles[running_index], 1);
		}

		iterator_adjacency.advanceIterator(j, edges_per_page, memory, page_size, memory_manager->start_index);
		}
		adjacency_iterator.advanceIterator(i, edges_per_page, memory, page_size, memory_manager->start_index);
	}

	return;
	}

	//------------------------------------------------------------------------------
	//
	template <typename VertexDataType, typename EdgeDataType>
	__global__ void d_StaticTriangleCountingBalanced(MemoryManager* memory_manager,
												memory_t* memory,
												vertex_t* triangles,
												int page_size,
												vertex_t* vertex_index,
												vertex_t* page_per_vertex_index,
												int page_count)
	{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid >= page_count)
			return;
	
	vertex_t index = vertex_index[tid];
	vertex_t page_index = page_per_vertex_index[tid];
	VertexDataType* vertices = (VertexDataType*)memory;  
	vertex_t edges_per_page = memory_manager->edges_per_page;

	vertex_t neighbours = vertices[index].neighbours;
	AdjacencyIterator<EdgeDataType> adjacency_iterator(pageAccess<EdgeDataType>(memory, vertices[index].mem_index, page_size, memory_manager->start_index));
	AdjacencyIterator<EdgeDataType> iterator_adjacency, running_adjacency_list, search_iterator;
	for(int i = page_index; i > 0; --i)
	{
		adjacency_iterator.blockTraversalAbsolute(edges_per_page, memory, page_size, memory_manager->start_index);
	}

	// Now every thread points to its unique page in memory
	int iterations;
	if ((neighbours - 1) < ((page_index + 1) * edges_per_page))
	{
		iterations = (neighbours - 1) % edges_per_page;
	}
	else
	{
		iterations = edges_per_page;
	}

	// Iterate over neighbours
	for (int i = 0; i < iterations; ++i)
	{
		iterator_adjacency.setIterator(adjacency_iterator);
		vertex_t compare_value = adjacency_iterator.getDestination();
		if(compare_value > index)
		{
		adjacency_iterator.advanceIterator(i, edges_per_page, memory, page_size, memory_manager->start_index);
		break;
		}

		iterator_adjacency.advanceIterator(i, edges_per_page, memory, page_size, memory_manager->start_index);
		for (int j = (page_index * edges_per_page) + i + 1; j < neighbours; ++j)
		{
		vertex_t running_index = iterator_adjacency.getDestination();
		if(running_index > index)
		{
			iterator_adjacency.advanceIterator(j, edges_per_page, memory, page_size, memory_manager->start_index);
			break;
		}

		running_adjacency_list.setIterator(pageAccess<EdgeDataType>(memory, vertices[running_index].mem_index, page_size, memory_manager->start_index));
			
		// Now iterate over individual adjacency list and search for the neighbour_index
		vertex_t running_neighbours = vertices[running_index].neighbours;  
		while(true)
		{
			search_iterator.setIterator(running_adjacency_list);
			if (running_neighbours > edges_per_page)
			{
			running_adjacency_list.blockTraversalAbsolute(edges_per_page, memory, page_size, memory_manager->start_index);
			if(running_adjacency_list.getDestination() <= compare_value)
				running_neighbours -= edges_per_page;
			else
				break;
			}
			else
			{
			break;
			}    
		}
		if(running_neighbours > edges_per_page)
		{
			running_neighbours = edges_per_page;
		}

		if(d_binarySearchOnPage(search_iterator, compare_value, running_neighbours))
		{        
			atomicAdd(&triangles[index], 1);
			atomicAdd(&triangles[compare_value], 1);
			atomicAdd(&triangles[running_index], 1);
		}   

		iterator_adjacency.advanceIterator(j, edges_per_page, memory, page_size, memory_manager->start_index);
		}
		++adjacency_iterator;
	}
	
	return;
	}


	constexpr int MULTIPLICATOR {8};
	//------------------------------------------------------------------------------
	//
	template <typename VertexDataType, typename EdgeDataType>
	__global__ void d_StaticTriangleCountingWarpSized(MemoryManager* memory_manager,
													memory_t* memory,
													vertex_t* triangles,
													int page_size)
	{
	int warpID = threadIdx.x / WARPSIZE;
	int wid = (blockIdx.x * MULTIPLICATOR) + warpID;  
	int threadID = threadIdx.x - (warpID * WARPSIZE);
	if ((wid >= memory_manager->next_free_vertex_index) || (threadID >= memory_manager->edges_per_page))
		return;

	VertexDataType* vertices = (VertexDataType*)memory;
	__shared__ vertex_t neighbours[MULTIPLICATOR], edges_per_page[MULTIPLICATOR], running_index[MULTIPLICATOR], compare_value[MULTIPLICATOR];
	__shared__ AdjacencyIterator<EdgeDataType> adjacency_iterator[MULTIPLICATOR], iterator_adjacency[MULTIPLICATOR], running_adjacency_list[MULTIPLICATOR];
	__shared__ bool triangle_found[MULTIPLICATOR], continue_loop[MULTIPLICATOR];  

	if (SINGLE_THREAD_MULTI)
	{
		edges_per_page[warpID] = memory_manager->edges_per_page;
		neighbours[warpID] = vertices[wid].neighbours;
		adjacency_iterator[warpID].setIterator(pageAccess<EdgeDataType>(memory, vertices[wid].mem_index, page_size, memory_manager->start_index));
		triangle_found[warpID] = false;
		continue_loop[warpID] = false;
	} 
	__syncwarp();

	// Iterate over neighbours
	for (int i = 0; i < neighbours[warpID] - 1; ++i)
	{
		if (SINGLE_THREAD_MULTI)
		{
		continue_loop[warpID] = false;
		iterator_adjacency[warpID].setIterator(adjacency_iterator[warpID]);
		compare_value[warpID] = adjacency_iterator[warpID].getDestination();
		if (compare_value[warpID] < wid)
		{
			continue_loop[warpID] = true;
			adjacency_iterator[warpID].advanceIterator(i, edges_per_page[warpID], memory, page_size, memory_manager->start_index);
		}
		else
		{
			iterator_adjacency[warpID].advanceIterator(i, edges_per_page[warpID], memory, page_size, memory_manager->start_index);
		}      
		}
		__syncwarp();
		// If value smaller than vertex index, continue
		if (continue_loop[warpID])
		continue;

		for (int j = i + 1; j < neighbours[warpID]; ++j)
		{
		if (SINGLE_THREAD_MULTI)
		{
			continue_loop[warpID] = false;
			running_index[warpID] = iterator_adjacency[warpID].getDestination();
			if (running_index[warpID] < wid)
			{
			continue_loop[warpID] = true;
			iterator_adjacency[warpID].advanceIterator(j, edges_per_page[warpID], memory, page_size, memory_manager->start_index);
			}
			else
			{
			running_adjacency_list[warpID].setIterator(pageAccess<EdgeDataType>(memory, vertices[running_index[warpID]].mem_index, page_size, memory_manager->start_index));
			}        
		}
		__syncwarp();
		// If value smaller than vertex index, continue
		if (continue_loop[warpID])
			continue;

		// Now iterate over individual adjacency list and search for the neighbour_index
		for (int k = threadID; (k < vertices[running_index[warpID]].neighbours) && !(triangle_found[warpID]); k += edges_per_page[warpID])
		{
			if (running_adjacency_list[warpID].getDestinationAt(threadID) >= compare_value[warpID])
			{
			if (running_adjacency_list[warpID].getDestinationAt(threadID) == compare_value[warpID])
			{
				atomicAdd(&triangles[wid], 1);
				atomicAdd(&triangles[compare_value[warpID]], 1);
				atomicAdd(&triangles[running_index[warpID]], 1);
			}
			triangle_found[warpID] = true;                
			}
			__syncwarp();

			// Check if we found something
			if (triangle_found[warpID])
			{
			break;
			}

			if (SINGLE_THREAD_MULTI)
			{
			running_adjacency_list[warpID].blockTraversalAbsolute(edges_per_page[warpID], memory, page_size, memory_manager->start_index);
			}
			__syncwarp();
		}

		if (SINGLE_THREAD_MULTI)
		{
			iterator_adjacency[warpID].advanceIterator(j, edges_per_page[warpID], memory, page_size, memory_manager->start_index);
			triangle_found[warpID] = false; 
		}
		__syncwarp();

		}
		if (SINGLE_THREAD_MULTI)
		{
		adjacency_iterator[warpID].advanceIterator(i, edges_per_page[warpID], memory, page_size, memory_manager->start_index);
		}
		__syncwarp();

	}

	return;
	}

	//------------------------------------------------------------------------------
	//
	template <typename VertexDataType, typename EdgeDataType>
	__global__ void d_StaticTriangleCountingWarpSizedBalanced(MemoryManager* memory_manager,
														memory_t* memory,
														vertex_t* triangles,
														int page_size,
														vertex_t* vertex_index,
														vertex_t* page_per_vertex_index,
														int page_count)
	{
	int warpID = threadIdx.x / WARPSIZE;
	int pageID = (blockIdx.x * MULTIPLICATOR) + warpID;  
	int threadID = threadIdx.x - (warpID * WARPSIZE);
	if ((pageID >= page_count) || (threadID >= memory_manager->edges_per_page))
		return;

	VertexDataType* vertices = (VertexDataType*)memory;

	__shared__ vertex_t neighbours[MULTIPLICATOR], edges_per_page[MULTIPLICATOR], running_index[MULTIPLICATOR], compare_value[MULTIPLICATOR], index[MULTIPLICATOR], page_index[MULTIPLICATOR];
	__shared__ AdjacencyIterator<EdgeDataType> adjacency_iterator[MULTIPLICATOR], iterator_adjacency[MULTIPLICATOR], running_adjacency_list[MULTIPLICATOR];
	__shared__ int triangleCount[MULTIPLICATOR], iterations[MULTIPLICATOR];
	__shared__ bool triangle_found[MULTIPLICATOR];

	if (SINGLE_THREAD_MULTI)
	{
		index[warpID] = vertex_index[pageID];
		page_index[warpID] = page_per_vertex_index[pageID];
		triangleCount[warpID] = 0;
		edges_per_page[warpID] = memory_manager->edges_per_page;
		neighbours[warpID] = vertices[index[warpID]].neighbours;
		adjacency_iterator[warpID].setIterator(pageAccess<EdgeDataType>(memory, vertices[index[warpID]].mem_index, page_size, memory_manager->start_index));
		triangle_found[warpID] = false;
		for(int i = page_index[warpID]; i > 0; --i)
		{
		adjacency_iterator[warpID].blockTraversalAbsolute(edges_per_page[warpID], memory, page_size, memory_manager->start_index);
		}
		if ((neighbours[warpID] - 1) < ((page_index[warpID] + 1) * edges_per_page[warpID]))
		{
		iterations[warpID] = (neighbours[warpID] - 1) % edges_per_page[warpID];
		}
		else
		{
		iterations[warpID] = edges_per_page[warpID];
		}
	} 
	__syncwarp();

	// Iterate over neighbours
	for (int i = 0; i < iterations[warpID]; ++i)
	{
		if (SINGLE_THREAD_MULTI)
		{
		iterator_adjacency[warpID].setIterator(adjacency_iterator[warpID]);
		compare_value[warpID] = adjacency_iterator[warpID].getDestination();
		iterator_adjacency[warpID].advanceIterator(i, edges_per_page[warpID], memory, page_size, memory_manager->start_index);
		}
		__syncwarp();

		for (int j = (page_index[warpID] * edges_per_page[warpID]) + i + 1; j < neighbours[warpID]; ++j)
		{
		if (SINGLE_THREAD_MULTI)
		{
			running_index[warpID] = iterator_adjacency[warpID].getDestination();
			running_adjacency_list[warpID].setIterator(pageAccess<EdgeDataType>(memory, vertices[running_index[warpID]].mem_index, page_size, memory_manager->start_index));
		}
		__syncwarp();

		// Now iterate over individual adjacency list and search for the neighbour_index
		for (int k = threadID; (k < vertices[running_index[warpID]].neighbours) && !(triangle_found[warpID]); k += edges_per_page[warpID])
		{
			if (running_adjacency_list[warpID].getDestinationAt(threadID) >= compare_value[warpID])
			{
			if (running_adjacency_list[warpID].getDestinationAt(threadID) == compare_value[warpID])
			{
				(triangleCount[warpID])++;
			}            
			triangle_found[warpID] = true;
			}
			__syncwarp();

			// Check if we found something
			if (triangle_found[warpID])
			{
			break;
			}

			if (SINGLE_THREAD_MULTI)
			{
			running_adjacency_list[warpID].blockTraversalAbsolute(edges_per_page[warpID], memory, page_size, memory_manager->start_index);
			}
			__syncwarp();
		}

		if (SINGLE_THREAD_MULTI)
		{
			iterator_adjacency[warpID].advanceIterator(j, edges_per_page[warpID], memory, page_size, memory_manager->start_index);
			triangle_found[warpID] = false; 
		}
		__syncwarp();

		}
		if (SINGLE_THREAD_MULTI)
		{
		++(adjacency_iterator[warpID]);
		}
		__syncwarp();

	}

	// Save triangle count to array
	if (triangleCount[warpID] > 0 && SINGLE_THREAD_MULTI)
		atomicAdd(&triangles[index[warpID]], triangleCount[warpID]);

	return;
	}
}



//------------------------------------------------------------------------------
// Host funtionality
//------------------------------------------------------------------------------
//

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType>
uint32_t STC<VertexDataType, EdgeDataType>::StaticTriangleCounting(const std::unique_ptr<MemoryManager>& memory_manager, bool global_TC_count)
{
  int block_size;
  int grid_size;
  uint32_t triangle_count;

  // Launch triangle counting algorithm
  if (variant == STCVariant::NAIVE)
  {
    block_size = KERNEL_LAUNCH_BLOCK_SIZE_STANDARD;
    grid_size = (memory_manager->next_free_vertex_index / block_size) + 1;
    HANDLE_ERROR(cudaMemset(d_triangles,
                            0,
                            sizeof(uint32_t) * memory_manager->next_free_vertex_index));
    faimGraphSTC::d_StaticTriangleCounting<VertexDataType, EdgeDataType> << <grid_size, block_size >> > ((MemoryManager*)memory_manager->d_memory,
                                                                                            memory_manager->d_data,
                                                                                            d_triangles,
                                                                                            memory_manager->next_free_vertex_index,
                                                                                            memory_manager->page_size);
  }
  else if (variant == STCVariant::BALANCED)
  {
    HANDLE_ERROR(cudaMemset(d_triangles,
                            0,
                            sizeof(uint32_t) * memory_manager->next_free_vertex_index));
    int number_pages = memory_manager->numberPagesInMemory<VertexDataType>(d_page_count, d_accumulated_page_count);
    //std::cout << "Number of Pages: " << number_pages << std::endl;
    if (d_vertex_index == nullptr)
    {
      TemporaryMemoryAccessHeap temp_memory_dispenser(memory_manager.get(), reinterpret_cast<memory_t*>(d_page_count + memory_manager->next_free_vertex_index + 1));
      d_vertex_index = temp_memory_dispenser.getTemporaryMemory<vertex_t>(number_pages);
      d_page_per_vertex_index = temp_memory_dispenser.getTemporaryMemory<vertex_t>(number_pages);
    }

    memory_manager->workBalanceCalculation(d_accumulated_page_count, number_pages, d_vertex_index, d_page_per_vertex_index);

    block_size = KERNEL_LAUNCH_BLOCK_SIZE_STANDARD;
    grid_size = (number_pages / block_size) + 1;
    faimGraphSTC::d_StaticTriangleCountingBalanced<VertexDataType, EdgeDataType> << <grid_size, block_size >> >((MemoryManager*)memory_manager->d_memory,
                                                                                                memory_manager->d_data,
                                                                                                d_triangles,
                                                                                                memory_manager->page_size,
                                                                                                d_vertex_index,
                                                                                                d_page_per_vertex_index,
                                                                                                number_pages);
  }
  else if(variant == STCVariant::WARPSIZED)
  {
    block_size = WARPSIZE * faimGraphSTC::MULTIPLICATOR;
    grid_size = (memory_manager->next_free_vertex_index / faimGraphSTC::MULTIPLICATOR) + 1;
    faimGraphSTC::d_StaticTriangleCountingWarpSized<VertexDataType, EdgeDataType> << <grid_size, block_size >> > ((MemoryManager*)memory_manager->d_memory,
                                                                                                memory_manager->d_data,
                                                                                                d_triangles,
                                                                                                memory_manager->page_size);
  }
  else if(variant == STCVariant::WARPSIZEDBALANCED)
  {
    HANDLE_ERROR(cudaMemset(d_triangles,
                            0,
                            sizeof(uint32_t) * memory_manager->next_free_vertex_index));
    int number_pages = memory_manager->numberPagesInMemory<VertexDataType>(d_page_count, d_accumulated_page_count);
    //std::cout << "Number of Pages: " << number_pages << std::endl;
    if (d_vertex_index == nullptr)
    {
      TemporaryMemoryAccessHeap temp_memory_dispenser(memory_manager.get(), reinterpret_cast<memory_t*>(d_page_count + memory_manager->next_free_vertex_index + 1));
      d_vertex_index = temp_memory_dispenser.getTemporaryMemory<vertex_t>(number_pages);
      d_page_per_vertex_index = temp_memory_dispenser.getTemporaryMemory<vertex_t>(number_pages);
    }

    memory_manager->workBalanceCalculation(d_accumulated_page_count, number_pages, d_vertex_index, d_page_per_vertex_index);

    block_size = WARPSIZE * faimGraphSTC::MULTIPLICATOR;
    grid_size = (number_pages / faimGraphSTC::MULTIPLICATOR) + 1;
    faimGraphSTC::d_StaticTriangleCountingWarpSizedBalanced<VertexDataType, EdgeDataType> << <grid_size, block_size >> >((MemoryManager*)memory_manager->d_memory,
                                                                                                        memory_manager->d_data,
                                                                                                        d_triangles,
                                                                                                        memory_manager->page_size,
                                                                                                        d_vertex_index,
                                                                                                        d_page_per_vertex_index,
                                                                                                        number_pages);
  }

  if(global_TC_count)
  {
    // Prefix scan on d_triangles to get number of triangles
    thrust::device_ptr<uint32_t> th_triangles(d_triangles);
    thrust::device_ptr<uint32_t> th_triangle_count(d_triangle_count);
    thrust::inclusive_scan(th_triangles, th_triangles + memory_manager->next_free_vertex_index, th_triangle_count);

    // Copy result back to host
    HANDLE_ERROR (cudaMemcpy(triangles.get(),
                            d_triangles,
                            sizeof(uint32_t) * memory_manager->number_vertices,
                            cudaMemcpyDeviceToHost));

    // Copy number of triangles back
    HANDLE_ERROR (cudaMemcpy(&triangle_count,
                            d_triangle_count + (memory_manager->number_vertices - 1),
                            sizeof(uint32_t),
                            cudaMemcpyDeviceToHost));

    // // Report back number of triangles
    std::cout << "Triangle count is " << triangle_count << std::endl;

    return triangle_count;
  }
  else
  {
    return 0;
  }
}
