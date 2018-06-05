//------------------------------------------------------------------------------
// faimGraph.cu
//
// faimGraph
//
//------------------------------------------------------------------------------
//

#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <thrust/device_vector.h>

#include "faimGraph.h"
#include "MemoryManager.h"
#include "GraphParser.h"
#include "ConfigurationParser.h"

//############################################################################################################################################################
// Device funtionality
//############################################################################################################################################################

//------------------------------------------------------------------------------
// Initialisation functionality
//------------------------------------------------------------------------------
//

//------------------------------------------------------------------------------
//
__global__ void d_calculateMemoryRequirements(MemoryManager* memory_manager,
                                              vertex_t* offset,
                                              vertex_t* neighbours,
                                              vertex_t* capacity,
                                              vertex_t* block_requirements,
                                              int number_vertices, 
                                              int page_size)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
	  if (tid >= number_vertices)
		  return;
    
    // Calculate memory requirements
    vertex_t vertices_in_adjacency = offset[tid + 1] - offset[tid];
    // We need space for all the vertices in EdgeData and also need an edgeblock index per block
    vertex_t number_blocks = (vertex_t)ceil((float)(vertices_in_adjacency * MEMORYOVERALLOCATION) / (float)(memory_manager->edges_per_page));
    // If we have no edges initially, we still want an empty block
    if(number_blocks == 0)
    {
        number_blocks = 1;
    }

    vertex_t max_neighbours = (number_blocks * memory_manager->edges_per_page);

    neighbours[tid] = vertices_in_adjacency;
    capacity[tid] = max_neighbours;
    block_requirements[tid] = number_blocks;     

    return;
}

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType>
__global__ void d_setupFaimGraph(MemoryManager* memory_manager,
                                memory_t* memory,
                                vertex_t* adjacency,
                                vertex_t* offset,
                                vertex_t* neighbours,
                                vertex_t* capacity,
                                vertex_t* block_requirements,
                                vertex_t* mem_offsets,
                                int number_vertices, 
                                int index_shift,
                                int page_size,
                                ConfigurationParameters::PageLinkage page_linkage)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    
    // Initialise queuing approach
    memory_manager->d_page_queue.init();
    memory_manager->d_vertex_queue.init();
    
	  if (tid >= number_vertices)
		  return;

    // Setup memory
    VertexDataType* vertices = (VertexDataType*)memory;
    VertexDataType vertex; 

    vertex.locking = UNLOCK;
    vertex_t number_neighbours = neighbours[tid];
    vertex_t number_capacity = capacity[tid];
    vertex_t edges_per_page = memory_manager->edges_per_page;
    vertex.neighbours = number_neighbours;
    vertex.capacity = number_capacity;
    vertex.host_identifier = tid;

    // Setup Edge data and mem_index
    int block_index = mem_offsets[tid];
    vertex.mem_index = block_index;

    // Set vertex management data in memory
    vertices[tid] = vertex;

    AdjacencyIterator<EdgeDataType> adjacency_iterator(pageAccess<EdgeDataType>(memory, block_index, page_size, memory_manager->start_index));

    // Setup next free block
    if(tid == (number_vertices - 1))
    {
        // The last vertex sets the next free block
        memory_manager->next_free_page = block_index + block_requirements[number_vertices - 1];

        // Decrease free memory at initialization
        memory_manager->free_memory -= (page_size * (block_index + block_requirements[number_vertices - 1]));
    }

    // Write EdgeData
    int offset_index = offset[tid];
    for(int i = 0; i < number_neighbours; ++i)
    {
      if(page_linkage == ConfigurationParameters::PageLinkage::SINGLE)
        adjacency_iterator.adjacencySetup(i, edges_per_page, memory, page_size, memory_manager->start_index, adjacency, offset_index, block_index);
      else
        adjacency_iterator.adjacencySetupDoubleLinked(i, edges_per_page, memory, page_size, memory_manager->start_index, adjacency, offset_index, block_index);
    }

    // Set the rest to deletionmarker
    for(int i = number_neighbours; i < number_capacity; ++i)
    {
        setDeletionMarker(adjacency_iterator.getIterator(), edges_per_page);
        ++adjacency_iterator;
    }    

    return;   
}

//------------------------------------------------------------------------------
//
template <typename EdgeDataType>
__global__ void d_setupFaimGraphMatrix(MemoryManager* memory_manager,
                                      memory_t* memory,
                                      vertex_t* adjacency,
                                      matrix_t* matrix_values,
                                      vertex_t* offset,
                                      vertex_t* neighbours,
                                      vertex_t* capacity,
                                      vertex_t* block_requirements,
                                      vertex_t* mem_offsets,
                                      int number_vertices,
                                      int index_shift,
                                      int page_size,
                                      ConfigurationParameters::PageLinkage page_linkage,
                                      unsigned int vertex_offset = 0,
                                      unsigned int first_valid_page_index = 0)
{
  int tid = threadIdx.x + blockIdx.x*blockDim.x;

  // Initialise queuing approach
  if (vertex_offset == 0)
  {
    memory_manager->d_page_queue.init();
    memory_manager->d_vertex_queue.init();
  }  

  if (tid >= number_vertices)
    return;

  // Setup memory
  VertexData* vertices = (VertexData*)memory;
  VertexData vertex;

  vertex.locking = UNLOCK;
  vertex_t number_neighbours = neighbours[tid];
  vertex_t number_capacity = capacity[tid];
  vertex_t edges_per_page = memory_manager->edges_per_page;
  vertex.neighbours = number_neighbours;
  vertex.capacity = number_capacity;
  vertex.host_identifier = tid;

  // Setup Edge data and mem_index
  int block_index = first_valid_page_index + mem_offsets[tid];
  vertex.mem_index = block_index;

  // Set vertex management data in memory
  vertices[tid + vertex_offset] = vertex;

  AdjacencyIterator<EdgeDataType> adjacency_iterator(pageAccess<EdgeDataType>(memory, block_index, page_size, memory_manager->start_index));

  // Setup next free block
  if (tid == (number_vertices - 1))
  {
    // The last vertex sets the next free block
    memory_manager->next_free_page = first_valid_page_index + block_index + block_requirements[number_vertices - 1];

    // Decrease free memory at initialization
    memory_manager->free_memory -= (page_size * (block_index + block_requirements[number_vertices - 1]));
  }

  // Write EdgeData
  int offset_index = offset[tid];
  for (int i = 0; i < number_neighbours; ++i)
  {
    if (page_linkage == ConfigurationParameters::PageLinkage::SINGLE)
      adjacency_iterator.adjacencySetup(i, edges_per_page, memory, page_size, memory_manager->start_index, adjacency, matrix_values, offset_index, block_index);
    else
      adjacency_iterator.adjacencySetupDoubleLinked(i, edges_per_page, memory, page_size, memory_manager->start_index, adjacency, matrix_values, offset_index, block_index);
  }

  // Set the rest to deletionmarker
  for (int i = number_neighbours; i < number_capacity; ++i)
  {
    setDeletionMarker(adjacency_iterator.getIterator(), edges_per_page);
    ++adjacency_iterator;
  }

  return;
}

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType>
__global__ void d_faimGraphToCSR(MemoryManager* memory_manager,
                                    memory_t* memory,
                                    vertex_t* adjacency,
                                    vertex_t* offset,
                                    int index_shift,
                                    int page_size)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
	  if (tid >= memory_manager->next_free_vertex_index)
		  return;

    VertexDataType* vertices = (VertexDataType*)memory;

    // Deleted vertices
    if (vertices[tid].host_identifier == DELETIONMARKER)
    {
      return;
    }

    // Retrieve data
    vertex_t edge_data_index = vertices[tid].mem_index;
    vertex_t number_neighbours = vertices[tid].neighbours;
    vertex_t edges_per_page = memory_manager->edges_per_page;
    AdjacencyIterator<EdgeDataType> adjacency_iterator(pageAccess<EdgeDataType>(memory, edge_data_index, page_size, memory_manager->start_index));
    

    // Write EdgeData
    int offset_index = offset[tid];
    for(int i = 0, j = 0; j < number_neighbours; ++i)
    {
        // Normal case
        vertex_t adj_dest = adjacency_iterator.getDestination();
        if(adj_dest != DELETIONMARKER)
        {
            adjacency[offset_index + j] = adj_dest;
            j += 1;
        }  
        adjacency_iterator.advanceIterator(i, edges_per_page, memory, page_size, memory_manager->start_index);
    }

    return;
}

//------------------------------------------------------------------------------
//
template <typename EdgeDataType>
__global__ void d_faimGraphMatrixToCSR(MemoryManager* memory_manager,
                                      memory_t* memory,
                                      vertex_t* adjacency,
                                      matrix_t* matrix_values,
                                      vertex_t* offset,
                                      int index_shift,
                                      int page_size,
                                      vertex_t vertex_offset,
                                      vertex_t number_vertices)
{
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid >= number_vertices)
    return;

  VertexData* vertices = (VertexData*)memory;

  // Deleted vertices
  if (vertices[vertex_offset + tid].host_identifier == DELETIONMARKER)
  {
    return;
  }

  // Retrieve data
  vertex_t edge_data_index = vertices[vertex_offset + tid].mem_index;
  vertex_t number_neighbours = vertices[vertex_offset + tid].neighbours;
  vertex_t edges_per_page = memory_manager->edges_per_page;
  AdjacencyIterator<EdgeDataMatrix> adjacency_iterator(pageAccess<EdgeDataMatrix>(memory, edge_data_index, page_size, memory_manager->start_index));


  // Write EdgeData
  int offset_index = offset[tid];
  for (int i = 0, j = 0; j < number_neighbours; ++i)
  {
    // Normal case
    EdgeDataMatrix adj_dest = adjacency_iterator.getElement();
    if (adj_dest.destination != DELETIONMARKER)
    {
      adjacency[offset_index + j] = adj_dest.destination;
      matrix_values[offset_index + j] = adj_dest.matrix_value;
      j += 1;
    }
    adjacency_iterator.advanceIterator(i, edges_per_page, memory, page_size, memory_manager->start_index);
  }

  return;
}

//------------------------------------------------------------------------------
//
__global__ void d_CompareGraphs(vertex_t* adjacency_prover,
                                vertex_t* adjacency_verifier,
                                vertex_t* offset_prover,
                                vertex_t* offset_verifier,
                                int number_vertices,
                                int* equal)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
	  if (tid >= number_vertices)
		return;

    // Compare offset
    if(offset_prover[tid] != offset_verifier[tid])
    {
        printf("Offset-Round: %d | Prover: %d | Verifier: %d\n",tid, offset_prover[tid], offset_verifier[tid]);
        *equal = 0;
        return;
    }
        
    // Compare adjacency
    int offset = offset_verifier[tid];
    int neighbours = offset_verifier[tid + 1] - offset;
    for(int j = 0; j < neighbours; ++j)
    {
        int found_match = 0;
        for(int k = 0; k < neighbours; ++k)
        {
            if(adjacency_prover[offset + k] == adjacency_verifier[offset + j])
            {
                found_match = 1;
                break;
            }                    
        }
        if(found_match == 0)
        {
            printf("Vertex-Index: %d\n",tid);
            if(tid != number_vertices - 1)
            {
                printf("[DEVICE] Neighbours: %d\n", offset_prover[tid + 1] - offset_prover[tid]);
            }
            printf("[DEVICE]Prover-List:\n");            
            for (int l = 0; l < neighbours; ++l)
            {
              printf(" %d",adjacency_prover[offset + l]);
            }
            if(tid != number_vertices - 1)
            {
                printf("\n[HOST] Neighbours: %d\n", offset_verifier[tid + 1] - offset_verifier[tid]);
            }
            printf("[HOST]Verifier-List:\n");
            for (int l = 0; l < neighbours; ++l)
            {
              printf(" %d",adjacency_verifier[offset + l]);
            }
            *equal = 0;
            return;
        }                
    }
    return;
}


//############################################################################################################################################################
// Host funtionality
//############################################################################################################################################################

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename EdgeUpdateType>
void faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, EdgeUpdateType>::initializeMemory(std::unique_ptr<GraphParser>& graph_parser)
{
  // First setup memory manager
  memory_manager->initialize(config);

  // Set data in memory manager
  int number_of_vertices = graph_parser->getNumberOfVertices();

  // Setup csr data and calculate launch params
  std::unique_ptr<CSRData> csr_data (new CSRData(graph_parser, memory_manager));
  int block_size = config->testruns_.at(config->testrun_index_)->params->init_launch_block_size_;  
  int grid_size = (number_of_vertices / block_size) + 1;

  // Push memory manager information to device
  updateMemoryManagerDevice(memory_manager);

  // Calculate memory requirements
  d_calculateMemoryRequirements<<< grid_size, block_size >>>((MemoryManager*)memory_manager->d_memory,
                                                              csr_data->d_offset,
                                                              csr_data->d_neighbours,
                                                              csr_data->d_capacity,
                                                              csr_data->d_block_requirements,
                                                              number_of_vertices, 
                                                              memory_manager->page_size);

  // Prefix scan on d_block_requirements to get correct memory offsets
	thrust::device_ptr<vertex_t> th_block_requirements(csr_data->d_block_requirements);
	thrust::device_ptr<vertex_t> th_mem_requirements(csr_data->d_mem_requirements);
	thrust::exclusive_scan(th_block_requirements, th_block_requirements + number_of_vertices, th_mem_requirements);

  // Setup GPU Streaming memory
  d_setupFaimGraph<VertexDataType, EdgeDataType> <<< grid_size, block_size >>> ((MemoryManager*)memory_manager->d_memory,
                                                                              memory_manager->d_data,
                                                                              csr_data->d_adjacency,
                                                                              csr_data->d_offset,
                                                                              csr_data->d_neighbours,
                                                                              csr_data->d_capacity,
                                                                              csr_data->d_block_requirements,
                                                                              csr_data->d_mem_requirements,
                                                                              number_of_vertices,
                                                                              memory_manager->index_shift,
                                                                              memory_manager->page_size,
                                                                              memory_manager->page_linkage);

  // Push memory manager information back to host
  size_t mem_before_device_update = memory_manager->free_memory;
  updateMemoryManagerHost(memory_manager);

  // Vertex management data allocated
  memory_manager->decreaseAvailableMemory(sizeof(VertexDataType) * number_of_vertices);

  return;
}

template void faimGraph<VertexData, VertexUpdate, EdgeData, EdgeDataUpdate>::initializeMemory(std::unique_ptr<GraphParser>& graph_parser);
template void faimGraph<VertexData, VertexUpdate, EdgeDataMatrix, EdgeDataUpdate>::initializeMemory(std::unique_ptr<GraphParser>& graph_parser);
template void faimGraph<VertexDataWeight, VertexUpdateWeight, EdgeDataWeight, EdgeDataWeightUpdate>::initializeMemory(std::unique_ptr<GraphParser>& graph_parser);
template void faimGraph<VertexDataSemantic, VertexUpdateSemantic, EdgeDataSemantic, EdgeDataSemanticUpdate>::initializeMemory(std::unique_ptr<GraphParser>& graph_parser);
template void faimGraph<VertexData, VertexUpdate, EdgeDataSOA, EdgeDataUpdate>::initializeMemory(std::unique_ptr<GraphParser>& graph_parser);
template void faimGraph<VertexData, VertexUpdate, EdgeDataMatrixSOA, EdgeDataUpdate>::initializeMemory(std::unique_ptr<GraphParser>& graph_parser);
template void faimGraph<VertexDataWeight, VertexUpdateWeight, EdgeDataWeightSOA, EdgeDataWeightUpdate>::initializeMemory(std::unique_ptr<GraphParser>& graph_parser);
template void faimGraph<VertexDataSemantic, VertexUpdateSemantic, EdgeDataSemanticSOA, EdgeDataSemanticUpdate>::initializeMemory(std::unique_ptr<GraphParser>& graph_parser);


//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename EdgeUpdateType>
void faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, EdgeUpdateType>::initializefaimGraphMatrix(std::unique_ptr<CSRMatrixData>& csr_matrix_data)
{
  // First setup memory manager
  memory_manager->initialize(config);

  int block_size = config->testruns_.at(config->testrun_index_)->params->init_launch_block_size_;
  int grid_size = (csr_matrix_data->matrix_rows / block_size) + 1;

  // Push memory manager information to device
  updateMemoryManagerDevice(memory_manager);

  // Calculate memory requirements
  d_calculateMemoryRequirements << < grid_size, block_size >> >((MemoryManager*)memory_manager->d_memory,
                                                                csr_matrix_data->d_offset,
                                                                csr_matrix_data->d_neighbours,
                                                                csr_matrix_data->d_capacity,
                                                                csr_matrix_data->d_block_requirements,
                                                                csr_matrix_data->matrix_rows,
                                                                memory_manager->page_size);

  // Prefix scan on d_block_requirements to get correct memory offsets
  thrust::device_ptr<vertex_t> th_block_requirements(csr_matrix_data->d_block_requirements);
  thrust::device_ptr<vertex_t> th_mem_requirements(csr_matrix_data->d_mem_requirements);
  thrust::exclusive_scan(th_block_requirements, th_block_requirements + csr_matrix_data->matrix_rows, th_mem_requirements);

  // Setup GPU Streaming memory
  d_setupFaimGraphMatrix <EdgeDataType> << < grid_size, block_size >> > ((MemoryManager*)memory_manager->d_memory,
                                                        memory_manager->d_data,
                                                        csr_matrix_data->d_adjacency,
                                                        csr_matrix_data->d_matrix_values,
                                                        csr_matrix_data->d_offset,
                                                        csr_matrix_data->d_neighbours,
                                                        csr_matrix_data->d_capacity,
                                                        csr_matrix_data->d_block_requirements,
                                                        csr_matrix_data->d_mem_requirements,
                                                        csr_matrix_data->matrix_rows,
                                                        memory_manager->index_shift,
                                                        memory_manager->page_size,
                                                        memory_manager->page_linkage);

  // Push memory manager information back to host
  size_t mem_before_device_update = memory_manager->free_memory;
  updateMemoryManagerHost(memory_manager);

  // Vertex management data allocated
  memory_manager->decreaseAvailableMemory(sizeof(VertexData) * csr_matrix_data->matrix_rows);

  return;
}
template void faimGraph<VertexData, VertexUpdate, EdgeDataMatrix, EdgeDataUpdate>::initializefaimGraphMatrix(std::unique_ptr<CSRMatrixData>& csr_matrix_data);
template void faimGraph<VertexData, VertexUpdate, EdgeDataMatrixSOA, EdgeDataUpdate>::initializefaimGraphMatrix(std::unique_ptr<CSRMatrixData>& csr_matrix_data);

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename EdgeUpdateType>
void faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, EdgeUpdateType>::initializefaimGraphMatrix(std::unique_ptr<GraphParser>& graph_parser, unsigned int vertex_offset)
{
  // First setup memory manager
  memory_manager->initialize(config);

  int number_of_vertices = graph_parser->getNumberOfVertices();

  // Setup csr data and calculate launch params
  std::unique_ptr<CSRData> csr_data(new CSRData(graph_parser, memory_manager, vertex_offset));
  int block_size = config->testruns_.at(config->testrun_index_)->params->init_launch_block_size_;
  int grid_size = (number_of_vertices / block_size) + 1;

  // Push memory manager information to device
  updateMemoryManagerDevice(memory_manager);

  // Calculate memory requirements
  d_calculateMemoryRequirements << < grid_size, block_size >> >((MemoryManager*)memory_manager->d_memory,
                                                                csr_data->d_offset,
                                                                csr_data->d_neighbours,
                                                                csr_data->d_capacity,
                                                                csr_data->d_block_requirements,
                                                                number_of_vertices,
                                                                memory_manager->page_size);

  // Prefix scan on d_block_requirements to get correct memory offsets
  thrust::device_ptr<vertex_t> th_block_requirements(csr_data->d_block_requirements);
  thrust::device_ptr<vertex_t> th_mem_requirements(csr_data->d_mem_requirements);
  thrust::exclusive_scan(th_block_requirements, th_block_requirements + number_of_vertices, th_mem_requirements);

  // Setup GPU Streaming memory
  d_setupFaimGraphMatrix <EdgeDataType> << < grid_size, block_size >> > ((MemoryManager*)memory_manager->d_memory,
                                                                        memory_manager->d_data,
                                                                        csr_data->d_adjacency,
                                                                        csr_data->d_matrix_values,
                                                                        csr_data->d_offset,
                                                                        csr_data->d_neighbours,
                                                                        csr_data->d_capacity,
                                                                        csr_data->d_block_requirements,
                                                                        csr_data->d_mem_requirements,
                                                                        number_of_vertices,
                                                                        memory_manager->index_shift,
                                                                        memory_manager->page_size,
                                                                        memory_manager->page_linkage,
                                                                        vertex_offset,
                                                                        memory_manager->next_free_page);

  // Push memory manager information back to host
  size_t mem_before_device_update = memory_manager->free_memory;
  updateMemoryManagerHost(memory_manager);

  // Vertex management data allocated
  memory_manager->decreaseAvailableMemory(sizeof(VertexDataType) * number_of_vertices);

  return;
}

template void faimGraph<VertexData, VertexUpdate, EdgeDataMatrix, EdgeDataUpdate>::initializefaimGraphMatrix(std::unique_ptr<GraphParser>& graph_parser, unsigned int vertex_offset);
template void faimGraph<VertexData, VertexUpdate, EdgeDataMatrixSOA, EdgeDataUpdate>::initializefaimGraphMatrix(std::unique_ptr<GraphParser>& graph_parser, unsigned int vertex_offset);

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename EdgeUpdateType>
void faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, EdgeUpdateType>::initializefaimGraphEmptyMatrix(unsigned int number_rows, unsigned int vertex_offset)
{
  // First setup memory manager
  memory_manager->initialize(config);

  int number_of_vertices = number_rows;

  // Setup csr data and calculate launch params
  std::unique_ptr<CSRData> csr_data(new CSRData(memory_manager, number_rows, vertex_offset));
  int block_size = config->testruns_.at(config->testrun_index_)->params->init_launch_block_size_;
  int grid_size = (number_of_vertices / block_size) + 1;

  // Push memory manager information to device
  updateMemoryManagerDevice(memory_manager);  

  // Calculate memory requirements
  d_calculateMemoryRequirements << < grid_size, block_size >> >((MemoryManager*)memory_manager->d_memory,
                                                                csr_data->d_offset,
                                                                csr_data->d_neighbours,
                                                                csr_data->d_capacity,
                                                                csr_data->d_block_requirements,
                                                                number_of_vertices,
                                                                memory_manager->page_size);

  cudaDeviceSynchronize();

  // Prefix scan on d_block_requirements to get correct memory offsets
  thrust::device_ptr<vertex_t> th_block_requirements(csr_data->d_block_requirements);
  thrust::device_ptr<vertex_t> th_mem_requirements(csr_data->d_mem_requirements);
  thrust::exclusive_scan(th_block_requirements, th_block_requirements + number_of_vertices, th_mem_requirements);

  // Setup GPU Streaming memory
  d_setupFaimGraphMatrix <EdgeDataType> << < grid_size, block_size >> > ((MemoryManager*)memory_manager->d_memory,
                                                                        memory_manager->d_data,
                                                                        csr_data->d_adjacency,
                                                                        csr_data->d_matrix_values,
                                                                        csr_data->d_offset,
                                                                        csr_data->d_neighbours,
                                                                        csr_data->d_capacity,
                                                                        csr_data->d_block_requirements,
                                                                        csr_data->d_mem_requirements,
                                                                        number_of_vertices,
                                                                        memory_manager->index_shift,
                                                                        memory_manager->page_size,
                                                                        memory_manager->page_linkage,
                                                                        vertex_offset,
                                                                        memory_manager->next_free_page);

  cudaDeviceSynchronize();

  // Push memory manager information back to host
  size_t mem_before_device_update = memory_manager->free_memory;
  updateMemoryManagerHost(memory_manager);

  // Vertex management data allocated
  memory_manager->decreaseAvailableMemory(sizeof(VertexDataType) * number_of_vertices);

  return;
}

template void faimGraph<VertexData, VertexUpdate, EdgeDataMatrix, EdgeDataUpdate>::initializefaimGraphEmptyMatrix(unsigned int number_rows, unsigned int vertex_offset);
template void faimGraph<VertexData, VertexUpdate, EdgeDataMatrixSOA, EdgeDataUpdate>::initializefaimGraphEmptyMatrix(unsigned int number_rows, unsigned int vertex_offset);


//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename EdgeUpdateType>
std::unique_ptr<aimGraphCSR> faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, EdgeUpdateType>::verifyGraphStructure(std::unique_ptr<MemoryManager>& memory_manager)
{
    int block_size = KERNEL_LAUNCH_BLOCK_SIZE;
    int grid_size = (memory_manager->next_free_vertex_index / block_size) + 1;
    std::unique_ptr<aimGraphCSR> verifyGraph(new aimGraphCSR(memory_manager));
    vertex_t number_adjacency;

    memory_manager->numberEdgesInMemory<VertexDataType>(verifyGraph->d_mem_requirement);
    

     // Prefix scan on d_mem_requirement to get correct memory offsets
	  thrust::device_ptr<vertex_t> th_mem_requirements(verifyGraph->d_mem_requirement);
    thrust::device_ptr<vertex_t> th_offset(verifyGraph->d_offset);
	  thrust::exclusive_scan(th_mem_requirements, th_mem_requirements + memory_manager->next_free_vertex_index, th_offset);

    // Copy offsets to host
    HANDLE_ERROR(cudaMemcpy(verifyGraph->h_offset,
                            verifyGraph->d_offset,
                            sizeof(vertex_t) * memory_manager->next_free_vertex_index,
                            cudaMemcpyDeviceToHost));
    

    // Copy neighbors of last element to calculate memory requirements
    HANDLE_ERROR(cudaMemcpy(&number_adjacency,
                            verifyGraph->d_mem_requirement + (memory_manager->next_free_vertex_index - 1),
                            sizeof(vertex_t),
                            cudaMemcpyDeviceToHost));

    vertex_t accumulated_offset = verifyGraph->h_offset[(memory_manager->next_free_vertex_index - 1)];
    number_adjacency += accumulated_offset;
    verifyGraph->number_edges = number_adjacency;

    // Allocate memory for adjacency
    verifyGraph->scoped_mem_access_counter.alterSize(sizeof(vertex_t) * number_adjacency);
    
    verifyGraph->h_adjacency = (vertex_t*) malloc(sizeof(vertex_t) * number_adjacency);

    d_faimGraphToCSR<VertexDataType, EdgeDataType> <<< grid_size, block_size >>> ((MemoryManager*)memory_manager->d_memory,
                                                                                  memory_manager->d_data,
                                                                                  verifyGraph->d_adjacency,
                                                                                  verifyGraph->d_offset,
                                                                                  memory_manager->index_shift,
                                                                                  memory_manager->page_size);

    // Copy adjacency to host
    HANDLE_ERROR(cudaMemcpy(verifyGraph->h_adjacency,
                            verifyGraph->d_adjacency,
                            sizeof(vertex_t) * number_adjacency,
                            cudaMemcpyDeviceToHost));


    // Print adjacency at certain index
    //index_t index = 888;
    //index_t start_index = verifyGraph->h_offset[index];
    //index_t end_index = verifyGraph->h_offset[index + 1];
    //std::cout << "\n############################################################\n";
    //std::cout << "Print adjacency for index " << index << " with " << (end_index - start_index) << " neighbours" << std::endl;
    //for(size_t i = start_index, j = 0; i < end_index; ++i, ++j)
    //{
    //  std::cout << verifyGraph->h_adjacency[i] << " | ";
    //  if(((j) % (memory_manager->edges_per_page)) == (memory_manager->edges_per_page - 1))
    //  {
    //    std::cout << std::endl;
    //  }
    //}
    //std::cout << std::endl;
    //std::cout << "############################################################\n\n";

    return std::move(verifyGraph);
}

template std::unique_ptr<aimGraphCSR> faimGraph <VertexData, VertexUpdate, EdgeData, EdgeDataUpdate>::verifyGraphStructure (std::unique_ptr<MemoryManager>& memory_manager);
template std::unique_ptr<aimGraphCSR> faimGraph < VertexData, VertexUpdate, EdgeDataMatrix , EdgeDataUpdate > ::verifyGraphStructure(std::unique_ptr<MemoryManager>& memory_manager);
template std::unique_ptr<aimGraphCSR> faimGraph <VertexDataWeight, VertexUpdateWeight, EdgeDataWeight, EdgeDataWeightUpdate>::verifyGraphStructure (std::unique_ptr<MemoryManager>& memory_manager);
template std::unique_ptr<aimGraphCSR> faimGraph <VertexDataSemantic, VertexUpdateSemantic, EdgeDataSemantic, EdgeDataSemanticUpdate>::verifyGraphStructure (std::unique_ptr<MemoryManager>& memory_manager);
template std::unique_ptr<aimGraphCSR> faimGraph <VertexData, VertexUpdate, EdgeDataSOA, EdgeDataUpdate>::verifyGraphStructure (std::unique_ptr<MemoryManager>& memory_manager);
template std::unique_ptr<aimGraphCSR> faimGraph <VertexData, VertexUpdate, EdgeDataMatrixSOA, EdgeDataUpdate>::verifyGraphStructure(std::unique_ptr<MemoryManager>& memory_manager);
template std::unique_ptr<aimGraphCSR> faimGraph <VertexDataWeight, VertexUpdateWeight, EdgeDataWeightSOA, EdgeDataWeightUpdate>::verifyGraphStructure (std::unique_ptr<MemoryManager>& memory_manager);
template std::unique_ptr<aimGraphCSR> faimGraph <VertexDataSemantic, VertexUpdateSemantic, EdgeDataSemanticSOA, EdgeDataSemanticUpdate>::verifyGraphStructure (std::unique_ptr<MemoryManager>& memory_manager);

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename EdgeUpdateType>
std::unique_ptr<aimGraphCSR> faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, EdgeUpdateType>::verifyMatrixStructure(std::unique_ptr<MemoryManager>& memory_manager, vertex_t vertex_offset, vertex_t number_vertices)
{
  int block_size = KERNEL_LAUNCH_BLOCK_SIZE;
  int grid_size = (number_vertices / block_size) + 1;
  std::unique_ptr<aimGraphCSR> verifyGraph(new aimGraphCSR(memory_manager, vertex_offset, number_vertices));
  vertex_t number_adjacency;

  memory_manager->numberEdgesInMemory<VertexDataType>(verifyGraph->d_mem_requirement, vertex_offset, number_vertices);


  // Prefix scan on d_mem_requirement to get correct memory offsets
  thrust::device_ptr<vertex_t> th_mem_requirements(verifyGraph->d_mem_requirement);
  thrust::device_ptr<vertex_t> th_offset(verifyGraph->d_offset);
  thrust::exclusive_scan(th_mem_requirements, th_mem_requirements + number_vertices, th_offset);

  // Copy offsets to host
  HANDLE_ERROR(cudaMemcpy(verifyGraph->h_offset,
                          verifyGraph->d_offset,
                          sizeof(vertex_t) * number_vertices,
                          cudaMemcpyDeviceToHost));


  // Copy neighbors of last element to calculate memory requirements
  HANDLE_ERROR(cudaMemcpy(&number_adjacency,
                          verifyGraph->d_mem_requirement + (number_vertices - 1),
                          sizeof(vertex_t),
                          cudaMemcpyDeviceToHost));

  vertex_t accumulated_offset = verifyGraph->h_offset[(number_vertices - 1)];
  number_adjacency += accumulated_offset;
  verifyGraph->number_edges = number_adjacency;

  // Allocate memory for adjacency
  verifyGraph->scoped_mem_access_counter.alterSize(sizeof(vertex_t) * number_adjacency);
  verifyGraph->h_adjacency = (vertex_t*)malloc(sizeof(vertex_t) * number_adjacency);
  verifyGraph->h_matrix_values = (matrix_t*)malloc(sizeof(matrix_t) * number_adjacency);
  verifyGraph->d_matrix_values = static_cast<matrix_t*>(verifyGraph->d_adjacency + (sizeof(vertex_t) * number_adjacency));

  d_faimGraphMatrixToCSR<EdgeDataType> << < grid_size, block_size >> > ((MemoryManager*)memory_manager->d_memory,
                                                                                        memory_manager->d_data,
                                                                                        verifyGraph->d_adjacency,
                                                                                        verifyGraph->d_matrix_values,
                                                                                        verifyGraph->d_offset,
                                                                                        memory_manager->index_shift,
                                                                                        memory_manager->page_size,
                                                                                        vertex_offset,
                                                                                        number_vertices);

  // Copy adjacency and matrix values to host
  HANDLE_ERROR(cudaMemcpy(verifyGraph->h_adjacency,
                          verifyGraph->d_adjacency,
                          sizeof(vertex_t) * number_adjacency,
                          cudaMemcpyDeviceToHost));

  HANDLE_ERROR(cudaMemcpy(verifyGraph->h_matrix_values,
                          verifyGraph->d_matrix_values,
                          sizeof(matrix_t) * number_adjacency,
                          cudaMemcpyDeviceToHost));


  // Print adjacency at certain index
  //index_t index = 888;
  //index_t start_index = verifyGraph->h_offset[index];
  //index_t end_index = verifyGraph->h_offset[index + 1];
  //std::cout << "\n############################################################\n";
  //std::cout << "Print adjacency for index " << index << " with " << (end_index - start_index) << " neighbours" << std::endl;
  //for(size_t i = start_index, j = 0; i < end_index; ++i, ++j)
  //{
  //  std::cout << verifyGraph->h_adjacency[i] << " | ";
  //  if(((j) % (memory_manager->edges_per_page)) == (memory_manager->edges_per_page - 1))
  //  {
  //    std::cout << std::endl;
  //  }
  //}
  //std::cout << std::endl;
  //std::cout << "############################################################\n\n";

  return std::move(verifyGraph);
}

template std::unique_ptr<aimGraphCSR> faimGraph < VertexData, VertexUpdate, EdgeDataMatrix, EdgeDataUpdate > ::verifyMatrixStructure(std::unique_ptr<MemoryManager>& memory_manager, vertex_t vertex_offset, vertex_t number_vertices);
template std::unique_ptr<aimGraphCSR> faimGraph <VertexData, VertexUpdate, EdgeDataMatrixSOA, EdgeDataUpdate>::verifyMatrixStructure(std::unique_ptr<MemoryManager>& memory_manager, vertex_t vertex_offset, vertex_t number_vertices);

//------------------------------------------------------------------------------
//
bool h_CompareGraphs(vertex_t* adjacency_prover,
                     vertex_t* adjacency_verifier,
                     vertex_t* offset_prover,
                     vertex_t* offset_verifier,
                     int number_vertices,
                     int number_edges,
                     bool duplicate_check)
{
  for(int tid = 0; tid < number_vertices; ++tid)
  {
    // Compare offset
    int offset_prover_tid, offset_verifier_tid;
    if(tid != number_vertices - 1)
    {
      offset_prover_tid = offset_prover[tid + 1];
      offset_verifier_tid = offset_verifier[tid + 1];
    }
    else
    {
      offset_prover_tid = number_edges;
      offset_verifier_tid = offset_verifier[tid + 1];
    }

    if(offset_prover_tid != offset_verifier_tid)
    {
      std::cout << "[Offset] Src-Vertex: " << tid << " | [Device]faimGraph: " << offset_prover_tid << " | [Host]Host-Graph: " << offset_verifier_tid << std::endl;
    }

    int offset;
    int neighbours;

    if(offset_verifier_tid >= offset_prover_tid)
    {
      offset = offset_verifier[tid];
      neighbours = offset_verifier_tid - offset;
      for(int j = 0; j < neighbours; ++j)
      {
        bool found_match = false;
        vertex_t search_item = adjacency_verifier[offset + j];
        for(int k = 0; k < neighbours; ++k)
        {
          if(adjacency_prover[offset + k] == search_item)
          {
            found_match = true;
            break;
          }                    
        }
        if(found_match == false)
        {
          std::cout << "Host-Graph >= faimGraph" << std::endl;
          std::cout << "[Adjacency] Src-Vertex: " << tid  << " and search_item: " << search_item << std::endl;
          if(tid != number_vertices - 1)
          {
            std::cout << "\n[DEVICE] Neighbours: " << offset_prover_tid - offset_prover[tid] << std::endl;
          }
          
          std::cout << "[DEVICE]faimGraph-List:\n";          
          for (int l = 0; l < neighbours; ++l)
          {
            std::cout << adjacency_prover[offset + l] << " | ";
          }
          if(tid != number_vertices - 1)
          {
              std::cout << "\n[HOST] Neighbours: " << offset_verifier_tid - offset_verifier[tid] << std::endl;
          }

          std::cout << "[HOST]Host-Graph-List:\n";
          for (int l = 0; l < neighbours; ++l)
          {
            std::cout << adjacency_verifier[offset + l] << " | ";
          }
          std::cout << std::endl;  
          return false;
        }                
      }
      if(duplicate_check)
      {
        // Duplicate check
        offset = offset_prover[tid];
        neighbours = offset_prover_tid - offset;
        for(int j = 0; j < neighbours - 1; ++j)
        {
          for(int k = j + 1; k < neighbours; ++k)
          {
            if(adjacency_prover[offset + j] == adjacency_prover[offset + k])
            {
              std::cout << "DUPLICATE: " << adjacency_prover[offset + j] << std::endl;
              return false;
            }
          }
        }
      }
    }
    else
    {
      offset = offset_prover[tid];
      neighbours = offset_prover_tid - offset;
      for(int j = 0; j < neighbours; ++j)
      {
        bool found_match = false;
        vertex_t search_item = adjacency_prover[offset + j];
        for(int k = 0; k < neighbours; ++k)
        {
          if(adjacency_verifier[offset + k] == search_item)
          {
            found_match = true;
            break;
          }                    
        }
        if(found_match == false)
        {
          std::cout << "faimGraph > Host-Graph" << std::endl;
          std::cout << "[Adjacency] Src-Vertex: " << tid  << " and search_item: " << search_item << std::endl;
          
          std::cout << "\n[DEVICE] Neighbours: " << offset_prover_tid - offset_prover[tid] << std::endl;
          
          std::cout << "[DEVICE]faimGraph-List:\n";          
          for (int l = 0; l < neighbours; ++l)
          {
            std::cout << adjacency_prover[offset + l] << " | ";
          }

          std::cout << "\n[HOST] Neighbours: " << offset_verifier_tid - offset_verifier[tid] << std::endl;

          std::cout << "[HOST]Host-Graph-List:\n";
          for (int l = 0; l < neighbours; ++l)
          {
            std::cout << adjacency_verifier[offset + l] << " | ";
          }
          std::cout << std::endl;  
          return false;
        }                
      }
      if(duplicate_check)
      {
        // Duplicate check
        offset = offset_prover[tid];
        neighbours = offset_prover_tid - offset;
        for(int j = 0; j < neighbours - 1; ++j)
        {
          for(int k = j + 1; k < neighbours; ++k)
          {
            if(adjacency_prover[offset + j] == adjacency_prover[offset + k])
            {
              std::cout << "DUPLICATE: " << adjacency_prover[offset + j] << std::endl;
              return false;
            }
          }
        }
      }
    }
  }
  return true;
}

//------------------------------------------------------------------------------
//
bool h_CompareGraphs(vertex_t* adjacency_prover,
                      vertex_t* adjacency_verifier,
                      vertex_t* offset_prover,
                      vertex_t* offset_verifier,
                      matrix_t* matrix_values_prover,
                      matrix_t* matrix_values_verifier,
                      int number_vertices,
                      int number_edges,
                      bool duplicate_check)
{
  for (int tid = 0; tid < number_vertices; ++tid)
  {
    // Compare offset
    int offset_prover_tid, offset_verifier_tid;
    if (tid != number_vertices - 1)
    {
      offset_prover_tid = offset_prover[tid + 1];
      offset_verifier_tid = offset_verifier[tid + 1];
    }
    else
    {
      offset_prover_tid = number_edges;
      offset_verifier_tid = offset_verifier[tid + 1];
    }

    if (offset_prover_tid != offset_verifier_tid)
    {
      std::cout << "[Offset] Src-Vertex: " << tid << " | [Device]faimGraph: " << offset_prover_tid << " | [Host]Host-Graph: " << offset_verifier_tid << std::endl;
    }

    int offset;
    int neighbours;

    if (offset_verifier_tid >= offset_prover_tid)
    {
      offset = offset_verifier[tid];
      neighbours = offset_verifier_tid - offset;
      for (int j = 0; j < neighbours; ++j)
      {
        bool found_match = false;
        vertex_t search_item = adjacency_verifier[offset + j];
        matrix_t search_value = matrix_values_verifier[offset + j];
        for (int k = 0; k < neighbours; ++k)
        {
          if (adjacency_prover[offset + k] == search_item && matrix_values_prover[offset + k] == search_value)
          {
            found_match = true;
            break;
          }
        }
        if (found_match == false)
        {
          std::cout << "Host-Graph >= faimGraph" << std::endl;
          std::cout << "[Adjacency] Src-Vertex: " << tid << " and search_item: " << search_item << " and search_value: " << search_value << std::endl;
          if (tid != number_vertices - 1)
          {
            std::cout << "\n[DEVICE] Neighbours: " << offset_prover_tid - offset_prover[tid] << std::endl;
          }

          std::cout << "[DEVICE]faimGraph-List:\n";
          for (int l = 0; l < neighbours; ++l)
          {
            std::cout << "(" <<  adjacency_prover[offset + l] << " | " << matrix_values_prover[offset + l] << ")" << " | ";
          }
          if (tid != number_vertices - 1)
          {
            std::cout << "\n[HOST] Neighbours: " << offset_verifier_tid - offset_verifier[tid] << std::endl;
          }

          std::cout << "[HOST]Host-Graph-List:\n";
          for (int l = 0; l < neighbours; ++l)
          {
            std::cout << "(" << adjacency_verifier[offset + l] << " | " << matrix_values_verifier[offset + l] << ")" << " | ";
          }
          std::cout << std::endl;
          return false;
        }
      }
    }
    else
    {
      offset = offset_prover[tid];
      neighbours = offset_prover_tid - offset;
      for (int j = 0; j < neighbours; ++j)
      {
        bool found_match = false;
        vertex_t search_item = adjacency_prover[offset + j];
        matrix_t search_value = matrix_values_prover[offset + j];
        for (int k = 0; k < neighbours; ++k)
        {
          if (adjacency_verifier[offset + k] == search_item && matrix_values_verifier[offset + k] == search_value)
          {
            found_match = true;
            break;
          }
        }
        if (found_match == false)
        {
          std::cout << "faimGraph > Host-Graph" << std::endl;
          std::cout << "[Adjacency] Src-Vertex: " << tid << " and search_item: " << search_item << std::endl;

          std::cout << "\n[DEVICE] Neighbours: " << offset_prover_tid - offset_prover[tid] << std::endl;

          std::cout << "[DEVICE]faimGraph-List:\n";
          for (int l = 0; l < neighbours; ++l)
          {
            std::cout << "(" << adjacency_prover[offset + l] << " | " << matrix_values_prover[offset + l] << ")" << " | ";
          }

          std::cout << "\n[HOST] Neighbours: " << offset_verifier_tid - offset_verifier[tid] << std::endl;

          std::cout << "[HOST]Host-Graph-List:\n";
          for (int l = 0; l < neighbours; ++l)
          {
            std::cout << "(" << adjacency_verifier[offset + l] << " | " << matrix_values_verifier[offset + l] << ")" << " | ";
          }
          std::cout << std::endl;
          return false;
        }
      }
    }
  }
  return true;
}

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename EdgeUpdateType>
bool faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, EdgeUpdateType>::compareGraphs(std::unique_ptr<GraphParser>& graph_parser,
                   std::unique_ptr<aimGraphCSR>& verify_graph,
                   bool duplicate_check)
{
    int number_of_vertices = memory_manager->next_free_vertex_index;
    const auto prover_offset = verify_graph->h_offset;
    const auto prover_adjacency = verify_graph->h_adjacency;
    const auto verifier_offset = graph_parser->getOffset().data();
    const auto verifier_adjacency = graph_parser->getAdjacency().data();

    return h_CompareGraphs(prover_adjacency, verifier_adjacency, prover_offset, verifier_offset, number_of_vertices, verify_graph->number_edges, duplicate_check);
}

template bool faimGraph<VertexData, VertexUpdate, EdgeData, EdgeDataUpdate>::compareGraphs(std::unique_ptr<GraphParser>& graph_parser, std::unique_ptr<aimGraphCSR>& verify_graph, bool duplicate_check);
template bool faimGraph<VertexData, VertexUpdate, EdgeDataMatrix, EdgeDataUpdate>::compareGraphs(std::unique_ptr<GraphParser>& graph_parser, std::unique_ptr<aimGraphCSR>& verify_graph, bool duplicate_check);
template bool faimGraph<VertexDataWeight, VertexUpdateWeight, EdgeDataWeight, EdgeDataWeightUpdate>::compareGraphs(std::unique_ptr<GraphParser>& graph_parser, std::unique_ptr<aimGraphCSR>& verify_graph, bool duplicate_check);
template bool faimGraph<VertexDataSemantic, VertexUpdateSemantic, EdgeDataSemantic, EdgeDataSemanticUpdate>::compareGraphs(std::unique_ptr<GraphParser>& graph_parser, std::unique_ptr<aimGraphCSR>& verify_graph, bool duplicate_check);
template bool faimGraph<VertexData, VertexUpdate, EdgeDataSOA, EdgeDataUpdate>::compareGraphs(std::unique_ptr<GraphParser>& graph_parser, std::unique_ptr<aimGraphCSR>& verify_graph, bool duplicate_check);
template bool faimGraph<VertexData, VertexUpdate, EdgeDataMatrixSOA, EdgeDataUpdate>::compareGraphs(std::unique_ptr<GraphParser>& graph_parser, std::unique_ptr<aimGraphCSR>& verify_graph, bool duplicate_check);
template bool faimGraph<VertexDataWeight, VertexUpdateWeight, EdgeDataWeightSOA, EdgeDataWeightUpdate>::compareGraphs(std::unique_ptr<GraphParser>& graph_parser, std::unique_ptr<aimGraphCSR>& verify_graph, bool duplicate_check);
template bool faimGraph<VertexDataSemantic, VertexUpdateSemantic, EdgeDataSemanticSOA, EdgeDataSemanticUpdate>::compareGraphs(std::unique_ptr<GraphParser>& graph_parser, std::unique_ptr<aimGraphCSR>& verify_graph, bool duplicate_check);


//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename EdgeUpdateType>
bool faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, EdgeUpdateType>::compareGraphs(std::unique_ptr<CSRMatrix>& csr_matrix,
                                                                                            std::unique_ptr<aimGraphCSR>& verify_graph,
                                                                                            bool duplicate_check)
{
  int number_of_vertices = memory_manager->next_free_vertex_index;
  const auto prover_offset = verify_graph->h_offset;
  const auto prover_adjacency = verify_graph->h_adjacency;
  const auto prover_matrix_values = verify_graph->h_matrix_values;
  const auto verifier_offset = csr_matrix->offset.data();
  const auto verifier_adjacency = csr_matrix->adjacency.data();
  const auto verifier_matrix_values = csr_matrix->matrix_values.data();

  return h_CompareGraphs(prover_adjacency, verifier_adjacency, prover_offset, verifier_offset, prover_matrix_values, verifier_matrix_values, number_of_vertices, verify_graph->number_edges, duplicate_check);
}

template bool faimGraph<VertexData, VertexUpdate, EdgeDataMatrix, EdgeDataUpdate>::compareGraphs(std::unique_ptr<CSRMatrix>& csr_matrix, std::unique_ptr<aimGraphCSR>& verify_graph, bool duplicate_check);
template bool faimGraph<VertexData, VertexUpdate, EdgeDataMatrixSOA, EdgeDataUpdate>::compareGraphs(std::unique_ptr<CSRMatrix>& csr_matrix, std::unique_ptr<aimGraphCSR>& verify_graph, bool duplicate_check);