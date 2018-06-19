//------------------------------------------------------------------------------
// MemoryManager.cu
//
// faimGraph
//
//------------------------------------------------------------------------------
//
#pragma once
#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <thrust/device_vector.h>

#include "faimGraph.h"
#include "MemoryManager.h"
#include "GraphParser.h"
#include "ConfigurationParser.h"

namespace faimGraphMemoryManager
{
	//############################################################################################################################################################
	// Device funtionality
	//############################################################################################################################################################

	//------------------------------------------------------------------------------
	//
	template <typename VertexDataType>
	__global__ void d_reportEdges(MemoryManager* memory_manager, memory_t* memory, vertex_t* neighbours)
	{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= memory_manager->next_free_vertex_index)
		return;
	
	VertexDataType* vertices = (VertexDataType*)memory;
	if (vertices[tid].host_identifier != DELETIONMARKER)
		neighbours[tid] = vertices[tid].neighbours;
	else
		neighbours[tid] = 0;

	return;
	}

	//------------------------------------------------------------------------------
	//
	template <typename VertexDataType>
	__global__ void d_reportEdges(MemoryManager* memory_manager, memory_t* memory, vertex_t* neighbours, vertex_t vertex_offset, vertex_t number_vertices)
	{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= number_vertices)
		return;

	VertexDataType* vertices = (VertexDataType*)memory;
	if (vertices[vertex_offset + tid].host_identifier != DELETIONMARKER)
		neighbours[tid] = vertices[vertex_offset + tid].neighbours;
	else
		neighbours[tid] = 0;

	return;
	}

	//------------------------------------------------------------------------------
	//
	template <typename VertexDataType>
	__global__ void d_reportPages(MemoryManager* memory_manager, memory_t* memory, vertex_t* pages)
	{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= memory_manager->next_free_vertex_index)
		return;

	VertexDataType* vertices = (VertexDataType*)memory;
	if (vertices[tid].host_identifier != DELETIONMARKER)
		pages[tid] = vertices[tid].capacity / memory_manager->edges_per_page;
	else
		pages[tid] = 0;

	return;
	}

	//------------------------------------------------------------------------------
	// accumulated_page_count is size + 1
	//
	__global__ void d_workBalanceCalculation(MemoryManager* memory_manager, vertex_t* accumulated_page_count, vertex_t page_count, vertex_t* vertex_indices, vertex_t* page_per_vertex_indices)
	{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= memory_manager->next_free_vertex_index)
		return;

	vertex_t offset = accumulated_page_count[tid];
	vertex_t pages_per_vertex = accumulated_page_count[tid + 1] - offset;


	for(int i = 0; i < pages_per_vertex; ++i)
	{
		vertex_indices[offset + i] = tid;
		page_per_vertex_indices[offset + i] = i;
	}

	return;
	}
}


//############################################################################################################################################################
// Host funtionality
//############################################################################################################################################################

//------------------------------------------------------------------------------
//
void updateMemoryManagerHost(std::unique_ptr<MemoryManager>& memory_manager)
{
  HANDLE_ERROR(cudaMemcpy(memory_manager.get(),
                          memory_manager->d_memory,
                          sizeof(MemoryManager),
                          cudaMemcpyDeviceToHost));
  return;
}

//------------------------------------------------------------------------------
//
void updateMemoryManagerDevice(std::unique_ptr<MemoryManager>& memory_manager)
{
  HANDLE_ERROR(cudaMemcpy(memory_manager->d_memory,
                          memory_manager.get(),
                          sizeof(MemoryManager),
                          cudaMemcpyHostToDevice));
  return;
}

//------------------------------------------------------------------------------
//
template <typename VertexDataType>
size_t MemoryManager::numberEdgesInMemory(vertex_t* d_neighbours_count, bool return_count)
{
  int block_size = 256;
  int grid_size = (next_free_vertex_index / block_size) + 1;

  faimGraphMemoryManager::d_reportEdges<VertexDataType> <<< grid_size, block_size >>> ((MemoryManager*)d_memory,d_data, d_neighbours_count);

  if (return_count)
  {
    vertex_t accumulated_edge_count;
    thrust::device_ptr<vertex_t> th_neighbours_count(d_neighbours_count);
    thrust::exclusive_scan(th_neighbours_count, th_neighbours_count + (next_free_vertex_index + 1), th_neighbours_count);

    HANDLE_ERROR(cudaMemcpy(&accumulated_edge_count,
                            d_neighbours_count + (next_free_vertex_index),
                            sizeof(vertex_t),
                            cudaMemcpyDeviceToHost));

    return accumulated_edge_count;
  }
  return 0;
}

//------------------------------------------------------------------------------
//
template <typename VertexDataType>
size_t MemoryManager::numberEdgesInMemory(vertex_t* d_neighbours_count, vertex_t vertex_offset, vertex_t number_vertices, bool return_count)
{
  int block_size = 256;
  int grid_size = (number_vertices / block_size) + 1;

  faimGraphMemoryManager::d_reportEdges<VertexDataType> << < grid_size, block_size >> > ((MemoryManager*)d_memory, d_data, d_neighbours_count, vertex_offset, number_vertices);

  if (return_count)
  {
    vertex_t accumulated_edge_count;
    thrust::device_ptr<vertex_t> th_neighbours_count(d_neighbours_count);
    thrust::exclusive_scan(th_neighbours_count, th_neighbours_count + (number_vertices + 1), th_neighbours_count);

    HANDLE_ERROR(cudaMemcpy(&accumulated_edge_count,
                            d_neighbours_count + (number_vertices),
                            sizeof(vertex_t),
                            cudaMemcpyDeviceToHost));

    return accumulated_edge_count;
  }

  return 0;
}

//------------------------------------------------------------------------------
//
template <typename VertexDataType>
void MemoryManager::numberPagesInMemory(vertex_t* d_page_count)
{
  int block_size = 256;
  int grid_size = (next_free_vertex_index / block_size) + 1;

  faimGraphMemoryManager::d_reportPages<VertexDataType> << < grid_size, block_size >> > ((MemoryManager*)d_memory, d_data, d_page_count);
}

//------------------------------------------------------------------------------
//
template <typename VertexDataType>
size_t MemoryManager::numberPagesInMemory(vertex_t* d_page_count, vertex_t* d_accumulated_page_count)
{
  int block_size = 256;
  int grid_size = (next_free_vertex_index / block_size) + 1;

  faimGraphMemoryManager::d_reportPages<VertexDataType> << < grid_size, block_size >> > ((MemoryManager*)d_memory, d_data, d_page_count);

  vertex_t  accumulated_page_count;
	thrust::device_ptr<vertex_t> th_page_count(d_page_count);
  if (d_accumulated_page_count != nullptr)
  {
    thrust::device_ptr<vertex_t> th_page_requirements(d_accumulated_page_count);
    thrust::exclusive_scan(th_page_count, th_page_count + (next_free_vertex_index + 1), th_page_requirements);

    HANDLE_ERROR(cudaMemcpy(&accumulated_page_count,
      d_accumulated_page_count + (next_free_vertex_index),
      sizeof(vertex_t),
      cudaMemcpyDeviceToHost));
  }
  else
  {
    thrust::exclusive_scan(th_page_count, th_page_count + (next_free_vertex_index + 1), th_page_count);

    HANDLE_ERROR(cudaMemcpy(&accumulated_page_count,
      d_page_count + (next_free_vertex_index),
      sizeof(vertex_t),
      cudaMemcpyDeviceToHost));
  }
	
  number_pages = accumulated_page_count;
  return accumulated_page_count;
}

//------------------------------------------------------------------------------
//
void MemoryManager::workBalanceCalculation(vertex_t* d_accumulated_page_count, vertex_t page_count, vertex_t* d_vertex_indices, vertex_t* d_page_per_vertex_indices)
{
  int block_size = 256;
  int grid_size = (next_free_vertex_index / block_size) + 1;

  faimGraphMemoryManager::d_workBalanceCalculation << < grid_size, block_size >> > ((MemoryManager*)d_memory, d_accumulated_page_count, page_count, d_vertex_indices, d_page_per_vertex_indices);
  return;
}
