//------------------------------------------------------------------------------
// SpMV.cu
//
// faimGraph
//
//------------------------------------------------------------------------------
//
#include <memory>
#include <thrust/device_vector.h>

#include "SpMV.h"
#include "ConfigurationParser.h"
#include "MemoryManager.h"
#include "Definitions.h"

//------------------------------------------------------------------------------
// Device funtionality
//------------------------------------------------------------------------------
//

#define MULTIPLICATOR 4

//------------------------------------------------------------------------------
//
__global__ void d_SpMVMultiplication(MemoryManager* memory_manager,
									memory_t* memory,
									int page_size,
									matrix_t* input_vector,
									matrix_t* result_vector)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= memory_manager->next_free_vertex_index)
		return;

	// Gather data access
	VertexData* vertices = (VertexData*)memory;
  VertexData vertex = vertices[tid];
	vertex_t edges_per_page = memory_manager->edges_per_page;
  matrix_t vector_element{ 0 };
  AdjacencyIterator<EdgeDataMatrix> adjacency_iterator(pageAccess<EdgeDataMatrix>(memory, vertices[tid].mem_index, page_size, memory_manager->start_index));
  
  EdgeDataMatrix local_element;
  for (int i = 0; i < vertex.neighbours; ++i)
  {
    local_element = adjacency_iterator.getElement();
    vector_element += local_element.matrix_value * input_vector[local_element.destination];
    adjacency_iterator.advanceIterator(i, edges_per_page, memory, page_size, memory_manager->start_index);
  }

  result_vector[tid] = vector_element;

  return;
}

//------------------------------------------------------------------------------
//
__global__ void d_SpMVMultiplication_Warpsized(MemoryManager* memory_manager,
								memory_t* memory,
								int page_size,
								matrix_t* input_vector,
								matrix_t* result_vector)
{
  int warpID = threadIdx.x / WARPSIZE;
  int wid = (blockIdx.x * MULTIPLICATOR) + warpID;
  vertex_t threadID = threadIdx.x - (warpID * WARPSIZE);
  vertex_t edges_per_page = memory_manager->edges_per_page;
  // Outside threads per block (because of indexing structure we use 31 threads)
  if ((threadID >= edges_per_page) || (wid >= memory_manager->next_free_vertex_index))
    return;

  // Gather data access
  volatile VertexData* vertices = (VertexData*)memory;

  // Shared variables per block to determine index
  __shared__ AdjacencyIterator < EdgeDataMatrix > adjacency_iterator[MULTIPLICATOR];
  __shared__ vertex_t neighbours[MULTIPLICATOR];
  __shared__ matrix_t vector_element[MULTIPLICATOR];

  if (SINGLE_THREAD_MULTI)
  {
    adjacency_iterator[warpID].setIterator(pageAccess<EdgeDataMatrix>(memory, vertices[wid].mem_index, page_size, memory_manager->start_index));
    neighbours[warpID] = vertices[wid].neighbours;
    vector_element[warpID] = 0;
  }
  __syncwarp();

  // Multiplication
  EdgeDataMatrix local_element;
  while (threadID < neighbours[warpID])
  {
    local_element = adjacency_iterator[warpID].getElementAt(threadID);
	  atomicAdd(&vector_element[warpID], (local_element.matrix_value * input_vector[local_element.destination]));
    if (SINGLE_THREAD_MULTI)
    {
      if (neighbours[warpID] > edges_per_page)
      {
        adjacency_iterator[warpID].blockTraversalAbsolute(edges_per_page, memory, page_size, memory_manager->start_index);
		    neighbours[warpID] -= edges_per_page;
      }
	    else
	    {
		    neighbours[warpID] = 0;
	    }
    }
    __syncwarp();
  }

  // Write to result vector
  if (SINGLE_THREAD_MULTI)
  {
    result_vector[wid] = vector_element[warpID];
  }
  __syncwarp();

  return;
}

//------------------------------------------------------------------------------
//
__global__ void d_occurenceCounter(MemoryManager* memory_manager,
                                   memory_t* memory,
                                   int page_size,
                                   vertex_t* occurence_counter)
{
  int warpID = threadIdx.x / WARPSIZE;
  int wid = (blockIdx.x * MULTIPLICATOR) + warpID;
  vertex_t threadID = threadIdx.x - (warpID * WARPSIZE);
  vertex_t edges_per_page = memory_manager->edges_per_page;
  // Outside threads per block (because of indexing structure we use 31 threads)
  if ((threadID >= edges_per_page) || (wid >= memory_manager->next_free_vertex_index))
    return;

  // Gather data access
  volatile VertexData* vertices = (VertexData*)memory;

  // Shared variables per block to determine index
  __shared__ AdjacencyIterator < EdgeDataMatrix > adjacency_iterator[MULTIPLICATOR];
  __shared__ vertex_t neighbours[MULTIPLICATOR];

  if (SINGLE_THREAD_MULTI)
  {
    adjacency_iterator[warpID].setIterator(pageAccess<EdgeDataMatrix>(memory, vertices[wid].mem_index, page_size, memory_manager->start_index));
    neighbours[warpID] = vertices[wid].neighbours;
  }
  __syncwarp();

  while (threadID < neighbours[warpID])
  {
    atomicAdd(&occurence_counter[adjacency_iterator[warpID].getDestinationAt(threadID)], 1);
    if (SINGLE_THREAD_MULTI)
    {
      if (neighbours[warpID] > edges_per_page)
      {
        adjacency_iterator[warpID].blockTraversalAbsolute(edges_per_page, memory, page_size, memory_manager->start_index);
        neighbours[warpID] -= edges_per_page;
      }
      else
      {
        neighbours[warpID] = 0;
      }
    }
    __syncwarp();
  }

  return;
}

//------------------------------------------------------------------------------
//
__global__ void d_writeTransposeCSR(MemoryManager* memory_manager,
                                    memory_t* memory,
                                    int page_size,
                                    vertex_t* offset,
                                    vertex_t* adjacency,
                                    matrix_t* matrix_values,
                                    vertex_t* position_helper)
{
  int warpID = threadIdx.x / WARPSIZE;
  int wid = (blockIdx.x * MULTIPLICATOR) + warpID;
  vertex_t threadID = threadIdx.x - (warpID * WARPSIZE);
  vertex_t edges_per_page = memory_manager->edges_per_page;
  // Outside threads per block (because of indexing structure we use 31 threads)
  if ((threadID >= edges_per_page) || (wid >= memory_manager->next_free_vertex_index))
    return;

  // Gather data access
  volatile VertexData* vertices = (VertexData*)memory;

  // Shared variables per block to determine index
  __shared__ AdjacencyIterator < EdgeDataMatrix > adjacency_iterator[MULTIPLICATOR];
  __shared__ vertex_t neighbours[MULTIPLICATOR];

  if (SINGLE_THREAD_MULTI)
  {
    adjacency_iterator[warpID].setIterator(pageAccess<EdgeDataMatrix>(memory, vertices[wid].mem_index, page_size, memory_manager->start_index));
    neighbours[warpID] = vertices[wid].neighbours;
  }
  __syncwarp();

  EdgeDataMatrix local_element;
  index_t insert_position;
  while (threadID < neighbours[warpID])
  {
    local_element = adjacency_iterator[warpID].getElementAt(threadID);
    insert_position = offset[local_element.destination];
    insert_position += atomicAdd(&(position_helper[local_element.destination]), 1);
    adjacency[insert_position] = wid;
    matrix_values[insert_position] = local_element.matrix_value;

    if (SINGLE_THREAD_MULTI)
    {
      if (neighbours[warpID] > edges_per_page)
      {
        adjacency_iterator[warpID].blockTraversalAbsolute(edges_per_page, memory, page_size, memory_manager->start_index);
        neighbours[warpID] -= edges_per_page;
      }
      else
      {
        neighbours[warpID] = 0;
      }
    }
    __syncwarp();
  }

  return;
}


//------------------------------------------------------------------------------
// Host funtionality
//------------------------------------------------------------------------------
//

//------------------------------------------------------------------------------
//
void SpMVManager::deviceSpMV(std::unique_ptr<MemoryManager>& memory_manager, 
                             const std::shared_ptr<Config>& config)
{
  int batch_size = matrix_rows;
  int block_size = WARPSIZE * MULTIPLICATOR;
  int grid_size = (batch_size / MULTIPLICATOR) + 1;
  bool warpsized = true;

  // Copy vector to device
  TemporaryMemoryAccessHeap temp_memory_dispenser(memory_manager.get(), memory_manager->next_free_vertex_index, sizeof(VertexData));
  d_input_vector = temp_memory_dispenser.getTemporaryMemory<matrix_t>(matrix_columns);
  d_result_vector = temp_memory_dispenser.getTemporaryMemory<matrix_t>(matrix_rows);
  HANDLE_ERROR(cudaMemcpy(d_input_vector,
               input_vector.get(),
               sizeof(matrix_t) * matrix_columns,
               cudaMemcpyHostToDevice));



  // Perform multiplication
  if (warpsized)
  {
    block_size = WARPSIZE * MULTIPLICATOR;
    grid_size = (batch_size / MULTIPLICATOR) + 1;
    d_SpMVMultiplication_Warpsized << < grid_size, block_size >> > ((MemoryManager*)memory_manager->d_memory,
                                                                    memory_manager->d_data,
                                                                    memory_manager->page_size,
                                                                    d_input_vector,
                                                                    d_result_vector);
  }
  else
  {
    block_size = config->testruns_.at(config->testrun_index_)->params->insert_launch_block_size_;
    grid_size = (batch_size / block_size) + 1;
    d_SpMVMultiplication << < grid_size, block_size >> > ((MemoryManager*)memory_manager->d_memory,
                                                            memory_manager->d_data,
                                                            memory_manager->page_size,
                                                            d_input_vector,
                                                            d_result_vector);
  }
  

  // Copy result to host
  HANDLE_ERROR(cudaMemcpy(result_vector.get(),
               d_result_vector,
               sizeof(matrix_t) * matrix_rows,
               cudaMemcpyDeviceToHost));

  return;
}

//------------------------------------------------------------------------------
//
template <typename EdgeDataType>
void SpMVManager::transposeaim2CSR2aim(std::unique_ptr<faimGraph<VertexData, VertexUpdate, EdgeDataType, EdgeDataUpdate>>& faimGraph,
                                       const std::shared_ptr<Config>& config)
{
  int batch_size = matrix_rows;
  int block_size = WARPSIZE * MULTIPLICATOR;
  int grid_size = (batch_size / MULTIPLICATOR) + 1;

  // Get number of edges in memory  
  std::unique_ptr<CSRMatrixData> csr_matrix_data = std::make_unique<CSRMatrixData>();
  TemporaryMemoryAccessHeap temp_memory_dispenser(faimGraph->memory_manager.get(), matrix_columns, sizeof(VertexData));
  vertex_t* d_occurence_counter = temp_memory_dispenser.getTemporaryMemory<vertex_t>(matrix_columns + 1);
  csr_matrix_data->d_offset = temp_memory_dispenser.getTemporaryMemory<vertex_t>(matrix_columns + 1);
  csr_matrix_data->d_neighbours = temp_memory_dispenser.getTemporaryMemory<vertex_t>(matrix_columns);
  csr_matrix_data->d_capacity = temp_memory_dispenser.getTemporaryMemory<vertex_t>(matrix_columns);
  csr_matrix_data->d_block_requirements = temp_memory_dispenser.getTemporaryMemory<vertex_t>(matrix_columns);
  csr_matrix_data->d_mem_requirements = temp_memory_dispenser.getTemporaryMemory<vertex_t>(matrix_columns);
  HANDLE_ERROR(cudaMemset(d_occurence_counter,
                          0,
                          sizeof(vertex_t) * (matrix_columns + 1)));
  // Count how often each vertex is referenced
  d_occurenceCounter << < grid_size, block_size >> > ((MemoryManager*)(faimGraph->memory_manager->d_memory),
																	  faimGraph->memory_manager->d_data,
																	  faimGraph->memory_manager->page_size,
																	  d_occurence_counter);

  thrust::device_ptr<vertex_t> th_occurence_counter(d_occurence_counter);
  thrust::device_ptr<vertex_t> th_offset(csr_matrix_data->d_offset);
  thrust::exclusive_scan(th_occurence_counter, th_occurence_counter + (matrix_columns + 1), th_offset);
  HANDLE_ERROR(cudaMemcpy(&(csr_matrix_data->edge_count),
                          csr_matrix_data->d_offset + matrix_columns,
                          sizeof(vertex_t),
                          cudaMemcpyDeviceToHost));

  // Allocate memory for adjacency and matrix data
  csr_matrix_data->d_adjacency = temp_memory_dispenser.getTemporaryMemory<vertex_t>(csr_matrix_data->edge_count);
  csr_matrix_data->d_matrix_values = temp_memory_dispenser.getTemporaryMemory<matrix_t>(csr_matrix_data->edge_count);
  HANDLE_ERROR(cudaMemset(d_occurence_counter,
                          0,
                          sizeof(vertex_t) * (matrix_columns + 1)));
  
  // Switch row/column for transpose
  csr_matrix_data->matrix_rows = matrix_columns;
  csr_matrix_data->matrix_columns = matrix_rows;

  // Write transpose CSR
  d_writeTransposeCSR << < grid_size, block_size >> > ((MemoryManager*)(faimGraph->memory_manager->d_memory),
																			faimGraph->memory_manager->d_data,
																			faimGraph->memory_manager->page_size,
																			csr_matrix_data->d_offset,
																			csr_matrix_data->d_adjacency,
																			csr_matrix_data->d_matrix_values,
																			d_occurence_counter);

  // Reset aimGraph
  faimGraph->memory_manager->resetFaimGraph(csr_matrix_data->matrix_rows, csr_matrix_data->matrix_columns);

  // Now we have the transpose of the matrix in CSR format (offset|adjacency|matrix_values)
  faimGraph->initializefaimGraphMatrix(csr_matrix_data);

  return;
}

template void SpMVManager::transposeaim2CSR2aim<EdgeDataMatrix>(std::unique_ptr<faimGraph<VertexData, VertexUpdate, EdgeDataMatrix, EdgeDataUpdate>>& faimGraph, const std::shared_ptr<Config>& config);
template void SpMVManager::transposeaim2CSR2aim<EdgeDataMatrixSOA>(std::unique_ptr<faimGraph<VertexData, VertexUpdate, EdgeDataMatrixSOA, EdgeDataUpdate>>& faimGraph, const std::shared_ptr<Config>& config);

//------------------------------------------------------------------------------
//
void SpMVManager::transpose(std::unique_ptr<MemoryManager>& memory_manager, 
                            const std::shared_ptr<Config>& config)
{
  return;
}

//------------------------------------------------------------------------------
//
void SpMVManager::generateRandomVector()
{
  srand(1);
  for (vertex_t i = 0; i < matrix_columns; ++i)
  {
    /*input_vector[i] = static_cast<matrix_t>(rand());*/
	input_vector[i] = 1;
  }
}