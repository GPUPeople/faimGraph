//------------------------------------------------------------------------------
// SpMM.cu
//
// faimGraph
//
//------------------------------------------------------------------------------
//

#include "SpMM.h"
#include "faimGraph.h"


//------------------------------------------------------------------------------
// Device funtionality
//------------------------------------------------------------------------------
//

//------------------------------------------------------------------------------
//
template <typename EdgeDataType>
__global__ void d_SpMMMultiplication(MemoryManager* memory_manager,
                                    memory_t* memory,
                                    int page_size,
                                    SpMMManager* spmm_manager)
{
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid >= spmm_manager->input_A_rows)
    return;  

  EdgeDataMatrix update;
  VertexData* vertices = (VertexData*)memory;
  VertexData input_vertex_A = vertices[tid];
  VertexData input_vertex_B;
  VertexData output_vertex = vertices[spmm_manager->output_offset + tid];
  vertex_t edges_per_page = memory_manager->edges_per_page;
  bool value_updated{ false };

  AdjacencyIterator<EdgeDataMatrix> adjacency_iterator(pageAccess<EdgeDataMatrix>(memory, input_vertex_A.mem_index, page_size, memory_manager->start_index));
  AdjacencyIterator<EdgeDataMatrix> matrix_iterator;
  AdjacencyIterator<EdgeDataMatrix> output_iterator;

  for (int i = 0; i < input_vertex_A.neighbours; ++i)
  {
    EdgeDataMatrix element_A = adjacency_iterator.getElement();
    /*if (tid == 66327)
    {
      printf("DEVICE: Dest: %u Val: %u\n", element_A.destination, element_A.matrix_value);
    }*/
    input_vertex_B = vertices[spmm_manager->input_B_offset + element_A.destination];
    matrix_iterator.setIterator(pageAccess<EdgeDataMatrix>(memory, input_vertex_B.mem_index, page_size, memory_manager->start_index));
    for (int j = 0; j < input_vertex_B.neighbours; ++j)
    {
      EdgeDataMatrix element_B = matrix_iterator.getElement();
      /*if (tid == 66327)
      {
        printf("DEVICE INNER: Dest: %u Val: %u\n", element_B.destination, element_B.matrix_value);
      }*/
      update.destination = element_B.destination;
      update.matrix_value = element_A.matrix_value * element_B.matrix_value;
      
      // #########################################################################
      // Update output matrix
      // #########################################################################
      output_iterator.setIterator(pageAccess<EdgeDataMatrix>(memory, output_vertex.mem_index, page_size, memory_manager->start_index));

      for (int i = 0; i < output_vertex.neighbours; ++i)
      {
        vertex_t adj_dest = output_iterator.getDestination();

        // Duplicate - Add value
        if (adj_dest == update.destination)
        {
          output_iterator.getElementPtr()->matrix_value += update.matrix_value;
          value_updated = true;
          break;
        }

        output_iterator.advanceIteratorEndCheck(i, edges_per_page, memory, page_size, memory_manager->start_index, output_vertex.capacity);
      }


      if (output_vertex.neighbours == output_vertex.capacity && !value_updated)
      {
        // If there was no space
        // Set index to next block and then reset adjacency list and insert edge
        index_t edge_block_index;
        index_t* edge_block_index_ptr = output_iterator.getPageIndexPtr(edges_per_page);

        if (memory_manager->d_page_queue.dequeue(edge_block_index))
        {
          // We got something from the queue
          *edge_block_index_ptr = edge_block_index;
        }
        else
        {
          // Queue is currently empty
          *edge_block_index_ptr = atomicAdd(&(memory_manager->next_free_page), 1);
        }

        output_iterator.setIterator(pageAccess<EdgeDataMatrix>(memory, *edge_block_index_ptr, page_size, memory_manager->start_index));
        updateAdjacency(output_iterator.getIterator(), update, edges_per_page);

        output_vertex.neighbours += 1;
        output_vertex.capacity += edges_per_page;
      }
      else if (!value_updated)
      {
        // There was space, insert edge
        updateAdjacency(output_iterator.getIterator(), update, edges_per_page);
        output_vertex.neighbours += 1;
      }
      value_updated = false;
      // #########################################################################
      // Update output matrix
      // #########################################################################

      matrix_iterator.advanceIterator(j, edges_per_page, memory, page_size, memory_manager->start_index);
    }
    adjacency_iterator.advanceIterator(i, edges_per_page, memory, page_size, memory_manager->start_index);
  }  

  vertices[spmm_manager->output_offset + tid] = output_vertex;

  return;
}

//------------------------------------------------------------------------------
//
template <typename EdgeDataType>
__global__ void d_SpMMMultiplication(MemoryManager* input_matrix_A,
                                      memory_t* input_matrix_A_memory,
                                      MemoryManager* input_matrix_B,
                                      memory_t* input_matrix_B_memory,
                                      MemoryManager* output_matrix,
                                      memory_t* output_matrix_memory,
                                      int page_size,
                                      SpMMManager* spmm_manager)
{
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid >= spmm_manager->input_A_rows)
    return;

  EdgeDataMatrix update;
  VertexData input_vertex_A = ((VertexData*)input_matrix_A_memory)[tid];
  VertexData input_vertex_B;
  VertexData output_vertex = ((VertexData*)output_matrix_memory)[tid];
  vertex_t edges_per_page = input_matrix_A->edges_per_page;
  bool value_updated{ false };

  AdjacencyIterator<EdgeDataMatrix> adjacency_iterator(pageAccess<EdgeDataMatrix>(input_matrix_A_memory, input_vertex_A.mem_index, page_size, input_matrix_A->start_index));
  AdjacencyIterator<EdgeDataMatrix> matrix_iterator;
  AdjacencyIterator<EdgeDataMatrix> output_iterator;

  for (int i = 0; i < input_vertex_A.neighbours; ++i)
  {
    EdgeDataMatrix element_A = adjacency_iterator.getElement();
    input_vertex_B = ((VertexData*)input_matrix_B_memory)[element_A.destination];
    matrix_iterator.setIterator(pageAccess<EdgeDataMatrix>(input_matrix_B_memory, input_vertex_B.mem_index, page_size, input_matrix_B->start_index));
    for (int j = 0; j < input_vertex_B.neighbours; ++j)
    {
      EdgeDataMatrix element_B = matrix_iterator.getElement();
      update.destination = element_B.destination;
      update.matrix_value = element_A.matrix_value * element_B.matrix_value;

      // #########################################################################
      // Update output matrix
      // #########################################################################
      output_iterator.setIterator(pageAccess<EdgeDataMatrix>(output_matrix_memory, output_vertex.mem_index, page_size, output_matrix->start_index));

      for (int i = 0; i < output_vertex.neighbours; ++i)
      {
        vertex_t adj_dest = output_iterator.getDestination();

        // Duplicate - Add value
        if (adj_dest == update.destination)
        {
          output_iterator.getElementPtr()->matrix_value += update.matrix_value;
          value_updated = true;
          break;
        }

        output_iterator.advanceIteratorEndCheck(i, edges_per_page, output_matrix_memory, page_size, output_matrix->start_index, output_vertex.capacity);
      }


      if (output_vertex.neighbours == output_vertex.capacity && !value_updated)
      {
        // If there was no space
        // Set index to next block and then reset adjacency list and insert edge
        index_t edge_block_index;
        index_t* edge_block_index_ptr = output_iterator.getPageIndexPtr(edges_per_page);

        if (output_matrix->d_page_queue.dequeue(edge_block_index))
        {
          // We got something from the queue
          *edge_block_index_ptr = edge_block_index;
        }
        else
        {
          // Queue is currently empty
          *edge_block_index_ptr = atomicAdd(&(output_matrix->next_free_page), 1);
        }

        output_iterator.setIterator(pageAccess<EdgeDataMatrix>(output_matrix_memory, *edge_block_index_ptr, page_size, output_matrix->start_index));
        updateAdjacency(output_iterator.getIterator(), update, edges_per_page);

        output_vertex.neighbours += 1;
        output_vertex.capacity += edges_per_page;
      }
      else if (!value_updated)
      {
        // There was space, insert edge
        updateAdjacency(output_iterator.getIterator(), update, edges_per_page);
        output_vertex.neighbours += 1;
      }
      value_updated = false;
      // #########################################################################
      // Update output matrix
      // #########################################################################

      matrix_iterator.advanceIterator(j, edges_per_page, input_matrix_B_memory, page_size, input_matrix_B->start_index);
    }
    adjacency_iterator.advanceIterator(i, edges_per_page, input_matrix_A_memory, page_size, input_matrix_A->start_index);
  }

  ((VertexData*)output_matrix_memory)[tid] = output_vertex;

  return;
}

#define MULTIPLICATOR 4
#define EDGES_PER_PAGE 15
//#define VERSION1

//------------------------------------------------------------------------------
// Parallelism on the first and third level
template <typename EdgeDataType>
__global__ void d_SpMMMultiplicationWarpsized(MemoryManager* memory_manager,
                                              memory_t* memory,
                                              int page_size,
                                              SpMMManager* spmm_manager)
{
  int warpID = threadIdx.x / WARPSIZE;
  int wid = (blockIdx.x * MULTIPLICATOR) + warpID;
  vertex_t threadID = threadIdx.x - (warpID * WARPSIZE);
  vertex_t edges_per_page = memory_manager->edges_per_page;
  // Outside threads per block (because of indexing structure we use 31 threads)
  if ((threadID >= edges_per_page) || (wid >= spmm_manager->input_A_rows))
    return;

  // Per Warp
  __shared__ VertexData input_vertex_A[MULTIPLICATOR], input_vertex_B[MULTIPLICATOR], output_vertex[MULTIPLICATOR];
  __shared__ EdgeDataMatrix update[MULTIPLICATOR][EDGES_PER_PAGE];
  __shared__ AdjacencyIterator<EdgeDataMatrix> adjacency_iterator[MULTIPLICATOR], matrix_iterator[MULTIPLICATOR], output_iterator[MULTIPLICATOR];
  __shared__ EdgeDataMatrix element_A[MULTIPLICATOR];
  __shared__ unsigned int insert_position[MULTIPLICATOR];

  // Per Thread
  EdgeDataMatrix element_B;
  int round{0};


  if (SINGLE_THREAD_MULTI)
  {
    input_vertex_A[warpID] = ((VertexData*)memory)[wid];
    output_vertex[warpID] = ((VertexData*)memory)[spmm_manager->output_offset + wid];
    adjacency_iterator[warpID].setIterator(pageAccess<EdgeDataMatrix>(memory, input_vertex_A[warpID].mem_index, page_size, memory_manager->start_index));
    output_iterator[warpID].setIterator(pageAccess<EdgeDataMatrix>(memory, output_vertex[warpID].mem_index, page_size, memory_manager->start_index));
  }
  __syncwarp();

  
  for (int i = 0; i < input_vertex_A[warpID].neighbours; ++i)
  {
    if (SINGLE_THREAD_MULTI)
    {
      element_A[warpID] = adjacency_iterator[warpID].getElement();
      input_vertex_B[warpID] = ((VertexData*)memory)[spmm_manager->input_B_offset + element_A[warpID].destination];
      matrix_iterator[warpID].setIterator(pageAccess<EdgeDataMatrix>(memory, input_vertex_B[warpID].mem_index, page_size, memory_manager->start_index));
      //if (wid == 2252)
      //{
      //  printf("Check out input vertex: %u with value: %u with neighbours: %u and capacity: %u\n", element_A[warpID].destination, element_A[warpID].matrix_value, input_vertex_B[warpID].neighbours, input_vertex_B[warpID].capacity);
      //  printf("##############################################\n");
      //}
    }
    __syncwarp();    
    
    round = 0;
    while (round < input_vertex_B[warpID].capacity)
    {
      element_B = matrix_iterator[warpID].getElementAt(threadID);
      if (round + threadID < input_vertex_B[warpID].neighbours)
      {
        update[warpID][threadID].destination = element_B.destination;
        update[warpID][threadID].matrix_value = element_A[warpID].matrix_value * element_B.matrix_value;
        //if (wid == 2252)
        //{
        //  printf("Generate update for %u with value %u in round %d\n", update[warpID][threadID].destination, update[warpID][threadID].matrix_value, round);
        //}
      }
      else
      {
        update[warpID][threadID].destination = DELETIONMARKER;
      }
      
      if (SINGLE_THREAD_MULTI)
      {
        insert_position[warpID] = output_vertex[warpID].neighbours;
        output_iterator[warpID].setIterator(pageAccess<EdgeDataMatrix>(memory, output_vertex[warpID].mem_index, page_size, memory_manager->start_index));
      }
      __syncwarp();

      // #########################################################################
      // Update output matrix
      // #########################################################################

      for (int j = 0; j < output_vertex[warpID].neighbours; ++j)
      {
        vertex_t adj_dest = output_iterator[warpID].getDestination();
        
        // Duplicate - Fuse values to graph
        if (adj_dest == update[warpID][threadID].destination)
        {
          atomicAdd(&((output_iterator[warpID].getElementPtr())->matrix_value), update[warpID][threadID].matrix_value);
          update[warpID][threadID].destination = DELETIONMARKER;
          //if (wid == 222)
          //{
          //  printf("Fuse to graph at destination: %u with value: %u\n", adj_dest, update[warpID][threadID].matrix_value);
          //}
        }
        if (SINGLE_THREAD_MULTI)
        {
          //if (wid == 222)
          //{
          //  printf("Examining destination: %u\n", adj_dest);
          //}
          output_iterator[warpID].advanceIteratorEndCheck(j, edges_per_page, memory, page_size, memory_manager->start_index, output_vertex[warpID].capacity);
        }
        __syncwarp();
      }

      // Duplicate fusion in batch
      vertex_t position{ DELETIONMARKER };
      for (unsigned int check = threadID; check > 0 && update[warpID][threadID].destination != DELETIONMARKER; --check)
      {
        if (update[warpID][check - 1].destination == update[warpID][threadID].destination)
        {
          position = check - 1;
        }
      }
      if (position != DELETIONMARKER)
      {
        //if (wid == 222)
        //{
        //  printf("Fuse duplicate in batch | Update at: %u with update at: %u\n", update[warpID][position].destination, update[warpID][threadID].destination);
        //}
        atomicAdd(&(update[warpID][position].matrix_value), update[warpID][threadID].matrix_value);
        update[warpID][threadID].destination = DELETIONMARKER;
      }
      __syncwarp();

# ifdef VERSION1
      // Single threaded inside
      if (SINGLE_THREAD_MULTI)
      {
        // Insert remaining updates into graph
        for (int k = output_vertex[warpID].neighbours, l = 0; l < EDGES_PER_PAGE; k++)
        {
          // Get first valid update
          while (update[warpID][l].destination == DELETIONMARKER && l < EDGES_PER_PAGE)
          {
            ++l;
          }
          if (l == EDGES_PER_PAGE)
          {
            break;
          }

          if (k < output_vertex[warpID].capacity)
          {
            // Insert updates
            updateAdjacency(output_iterator[warpID].getIterator(), update[warpID][l], edges_per_page);
            output_vertex[warpID].neighbours += 1;
            ++l;
          }
          else
          {
            // Get new page
            index_t edge_block_index;
            index_t* edge_block_index_ptr = output_iterator[warpID].getPageIndexPtr(edges_per_page);

            if (memory_manager->d_page_queue.dequeue(edge_block_index))
            {
              // We got something from the queue
              *edge_block_index_ptr = edge_block_index;
            }
            else
            {
              // Queue is currently empty
              *edge_block_index_ptr = atomicAdd(&(memory_manager->next_free_page), 1);
            }
            output_iterator[warpID].setIterator(pageAccess<EdgeDataMatrix>(memory, *edge_block_index_ptr, page_size, memory_manager->start_index));
            updateAdjacency(output_iterator[warpID].getIterator(), update[warpID][l], edges_per_page);
            output_vertex[warpID].neighbours += 1;
            output_vertex[warpID].capacity += edges_per_page;
            ++l;
          }
          output_iterator[warpID].advanceIterator(k, edges_per_page, memory, page_size, memory_manager->start_index);
        }
      }

#else
      // Parallel inside as well
      if (update[warpID][threadID].destination != DELETIONMARKER)
      {
        vertex_t pos = atomicAdd(&(insert_position[warpID]), 1);
        vertex_t neighbours = output_vertex[warpID].neighbours;
        if (pos < output_vertex[warpID].capacity)
        {
          updateAdjacency(output_iterator[warpID].getIteratorAt(pos - neighbours), update[warpID][threadID], edges_per_page);
          atomicAdd(&(output_vertex[warpID].neighbours), 1);
        }
        __syncwarp();
        // Remaining updates need new page
        if (pos >= output_vertex[warpID].capacity)
        {
          if (pos == output_vertex[warpID].capacity)
          {
            // Get new page
            index_t edge_block_index;
            output_iterator[warpID] += output_vertex[warpID].capacity - neighbours;
            index_t* edge_block_index_ptr = output_iterator[warpID].getPageIndexPtr(edges_per_page);

            if (memory_manager->d_page_queue.dequeue(edge_block_index))
            {
              // We got something from the queue
              *edge_block_index_ptr = edge_block_index;
            }
            else
            {
              // Queue is currently empty
              *edge_block_index_ptr = atomicAdd(&(memory_manager->next_free_page), 1);
            }
            output_iterator[warpID].setIterator(pageAccess<EdgeDataMatrix>(memory, *edge_block_index_ptr, page_size, memory_manager->start_index));
            atomicAdd(&(output_vertex[warpID].capacity), edges_per_page);
          }
          __syncwarp();
          updateAdjacency(output_iterator[warpID].getIteratorAt(pos % edges_per_page), update[warpID][threadID], edges_per_page);
          atomicAdd(&(output_vertex[warpID].neighbours), 1);
        }
        else
        {
          __syncwarp();
        }
        
      }
      else
      {
        __syncwarp();
        __syncwarp();
      }

#endif

      // #########################################################################
      // Update output matrix
      // #########################################################################


      round += (edges_per_page);
      // ################ SYNC ################
      __syncwarp();
      // ################ SYNC ################
      if (SINGLE_THREAD_MULTI && round < input_vertex_B[warpID].capacity)
      {
        // First move adjacency to the last element = index of next block
        matrix_iterator[warpID].blockTraversalAbsolute(edges_per_page, memory, page_size, memory_manager->start_index);
      }
      //if (SINGLE_THREAD_MULTI && wid == 222)
      //{
      //  printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
      //}
      // Sync so that everythread has the correct adjacencylist
      // ################ SYNC ################
      __syncwarp();
      // ################ SYNC ################
      
    }

    // Page Traversal in Input matrix A
    if (SINGLE_THREAD_MULTI)
    {
      adjacency_iterator[warpID].advanceIterator(i, edges_per_page, memory, page_size, memory_manager->start_index);
      //if (wid == 2252)
      //{
      //  printf("Round: %d with neighbours: %u\n", i, output_vertex[warpID].neighbours);
      //  printf("##############################################\n");
      //}
    }
    __syncwarp();
  }
  
  if (SINGLE_THREAD_MULTI)
  {
    ((VertexData*)memory)[spmm_manager->output_offset + wid] = output_vertex[warpID];
  }
  __syncwarp();
  

  return;
}

//------------------------------------------------------------------------------
// Host funtionality
//------------------------------------------------------------------------------
//

//------------------------------------------------------------------------------
//
template <typename EdgeDataType>
void SpMMManager::initializeFaimGraphMatrix(std::unique_ptr<faimGraph<VertexData, VertexUpdate, EdgeDataType, EdgeDataUpdate>>& faimGraph, std::unique_ptr<GraphParser>& graph_parser, const std::shared_ptr<Config>& config)
{
  // Setup first matrix
	faimGraph->initializefaimGraphMatrix(graph_parser, input_A_offset);
  cudaDeviceSynchronize();

  // Setup second matrix
  faimGraph->initializefaimGraphMatrix(graph_parser, input_B_offset);
  cudaDeviceSynchronize();

  // Setup output matrix
  faimGraph->initializefaimGraphEmptyMatrix(graph_parser->getNumberOfVertices(), output_offset);

  cudaDeviceSynchronize();
  // Update current settings
  updateMemoryManagerHost(faimGraph->memory_manager);
  next_free_page_after_init = faimGraph->memory_manager->next_free_page;

  return;
}

template void SpMMManager::initializeFaimGraphMatrix<EdgeDataMatrix>(std::unique_ptr<faimGraph<VertexData, VertexUpdate, EdgeDataMatrix, EdgeDataUpdate>>& faimGraph, std::unique_ptr<GraphParser>& graph_parser, const std::shared_ptr<Config>& config);
template void SpMMManager::initializeFaimGraphMatrix<EdgeDataMatrixSOA>(std::unique_ptr<faimGraph<VertexData, VertexUpdate, EdgeDataMatrixSOA, EdgeDataUpdate>>& faimGraph, std::unique_ptr<GraphParser>& graph_parser, const std::shared_ptr<Config>& config);

//------------------------------------------------------------------------------
//
template <typename EdgeDataType>
void SpMMManager::spmmMultiplication(std::unique_ptr<faimGraph<VertexData, VertexUpdate, EdgeDataType, EdgeDataUpdate>>& faimGraph, const std::shared_ptr<Config>& config)
{
  bool warpSized = true;
  int block_size;
  int grid_size;
  TemporaryMemoryAccessHeap temp_memory_dispenser(faimGraph->memory_manager.get(), next_free_vertex, sizeof(VertexData));
  auto d_spmm_manager = temp_memory_dispenser.getTemporaryMemory<SpMMManager>(1);
  auto d_mem_requirement = temp_memory_dispenser.getTemporaryMemory<vertex_t>(output_rows + 1);

  HANDLE_ERROR(cudaMemcpy(d_spmm_manager,
                          this,
                          sizeof(SpMMManager),
                          cudaMemcpyHostToDevice));

  if (!warpSized)
  {
    block_size = 256;
    grid_size = (input_A_rows / block_size) + 1;
    d_SpMMMultiplication<EdgeDataType> << <grid_size, block_size >> > ((MemoryManager*)faimGraph->memory_manager->d_memory,
																								faimGraph->memory_manager->d_data,
																								faimGraph->memory_manager->page_size,
																								d_spmm_manager);
  }
  else
  {
    block_size = WARPSIZE * MULTIPLICATOR;
    grid_size = (input_A_rows / MULTIPLICATOR) + 1;
    d_SpMMMultiplicationWarpsized<EdgeDataType> << <grid_size, block_size >> > ((MemoryManager*)faimGraph->memory_manager->d_memory,
																											faimGraph->memory_manager->d_data,
																											faimGraph->memory_manager->page_size,
																											d_spmm_manager);
  }

  

  std::cout << "Edges after Multiplication: " << faimGraph->memory_manager->template numberEdgesInMemory<VertexData>(d_mem_requirement, output_offset, output_rows, true) << std::endl;

  return;
}

template void SpMMManager::spmmMultiplication<EdgeDataMatrix>(std::unique_ptr<faimGraph<VertexData, VertexUpdate, EdgeDataMatrix, EdgeDataUpdate>>& faimGraph, const std::shared_ptr<Config>& config);
template void SpMMManager::spmmMultiplication<EdgeDataMatrixSOA>(std::unique_ptr<faimGraph<VertexData, VertexUpdate, EdgeDataMatrixSOA, EdgeDataUpdate>>& faimGraph, const std::shared_ptr<Config>& config);

//------------------------------------------------------------------------------
//
template <typename EdgeDataType>
void SpMMManager::spmmMultiplication(std::unique_ptr<faimGraph<VertexData, VertexUpdate, EdgeDataType, EdgeDataUpdate>>& input_matrix_A, 
                                    std::unique_ptr<faimGraph<VertexData, VertexUpdate, EdgeDataType, EdgeDataUpdate>>& input_matrix_B, 
                                    std::unique_ptr<faimGraph<VertexData, VertexUpdate, EdgeDataType, EdgeDataUpdate>>& output_matrix, 
                                    const std::shared_ptr<Config>& config)
{
  bool warpSized = false;
  int block_size;
  int grid_size;
  TemporaryMemoryAccessHeap temp_memory_dispenser(input_matrix_A->memory_manager.get(), next_free_vertex, sizeof(VertexData));
  auto d_spmm_manager = temp_memory_dispenser.getTemporaryMemory<SpMMManager>(1);
  auto d_mem_requirement = temp_memory_dispenser.getTemporaryMemory<vertex_t>(output_rows + 1);

  HANDLE_ERROR(cudaMemcpy(d_spmm_manager,
    this,
    sizeof(SpMMManager),
    cudaMemcpyHostToDevice));

  if (!warpSized)
  {
    block_size = 256;
    grid_size = (input_A_rows / block_size) + 1;
    d_SpMMMultiplication<EdgeDataType> << <grid_size, block_size >> > ((MemoryManager*)input_matrix_A->memory_manager->d_memory,
      input_matrix_A->memory_manager->d_data,
      (MemoryManager*)input_matrix_B->memory_manager->d_memory,
      input_matrix_B->memory_manager->d_data,
      (MemoryManager*)output_matrix->memory_manager->d_memory,
      output_matrix->memory_manager->d_data,
      input_matrix_A->memory_manager->page_size,
      d_spmm_manager);
  }
  else
  {
    //block_size = WARPSIZE * MULTIPLICATOR;
    //grid_size = (input_A_rows / MULTIPLICATOR) + 1;
    //d_SpMMMultiplicationWarpsized<EdgeDataType> << <grid_size, block_size >> > ((MemoryManager*)input_matrix_A->memory_manager->d_memory,
    //  input_matrix_A->memory_manager->d_data,
    //  input_matrix_A->memory_manager->page_size,
    //  d_spmm_manager);
  }



  std::cout << "Edges after Multiplication: " << output_matrix->memory_manager->template numberEdgesInMemory<VertexData>(d_mem_requirement, true) << std::endl;

  return;
}

template void SpMMManager::spmmMultiplication<EdgeDataMatrix>(std::unique_ptr<faimGraph<VertexData, VertexUpdate, EdgeDataMatrix, EdgeDataUpdate>>& input_matrix_A, std::unique_ptr<faimGraph<VertexData, VertexUpdate, EdgeDataMatrix, EdgeDataUpdate>>& input_matrix_B, std::unique_ptr<faimGraph<VertexData, VertexUpdate, EdgeDataMatrix, EdgeDataUpdate>>& output_matrix, const std::shared_ptr<Config>& config);
template void SpMMManager::spmmMultiplication<EdgeDataMatrixSOA>(std::unique_ptr<faimGraph<VertexData, VertexUpdate, EdgeDataMatrixSOA, EdgeDataUpdate>>& input_matrix_A, std::unique_ptr<faimGraph<VertexData, VertexUpdate, EdgeDataMatrixSOA, EdgeDataUpdate>>& input_matrix_B, std::unique_ptr<faimGraph<VertexData, VertexUpdate, EdgeDataMatrixSOA, EdgeDataUpdate>>& output_matrix, const std::shared_ptr<Config>& config);

//------------------------------------------------------------------------------
//
template <typename EdgeDataType>
void SpMMManager::resetResultMatrix(std::unique_ptr<faimGraph<VertexData, VertexUpdate, EdgeDataType, EdgeDataUpdate>>& faimGraph, const std::shared_ptr<Config>& config, bool tripleAimGraph)
{
  /*
  We have to reset the new page count -> simplest way to dismiss all new allocated pages
  And reset the neighbours count as well as the capacity for all vertices in the output
  */
  updateMemoryManagerHost(faimGraph->memory_manager);

  if (tripleAimGraph)
  {
	  faimGraph->memory_manager->next_free_page = 0;

	  faimGraph->memory_manager->template resetAllocationStatus<VertexData, EdgeDataType>(config, faimGraph->memory_manager->next_free_vertex_index, 0);
  }
  else
  {
	  faimGraph->memory_manager->next_free_page = next_free_page_after_init;

	  faimGraph->memory_manager->template resetAllocationStatus<VertexData, EdgeDataType>(config, output_rows, output_offset);
  }  

  updateMemoryManagerDevice(faimGraph->memory_manager);

  return;
}

template void SpMMManager::resetResultMatrix<EdgeDataMatrix>(std::unique_ptr<faimGraph<VertexData, VertexUpdate, EdgeDataMatrix, EdgeDataUpdate>>& faimGraph, const std::shared_ptr<Config>& config, bool tripleAimGraph);
template void SpMMManager::resetResultMatrix<EdgeDataMatrixSOA>(std::unique_ptr<faimGraph<VertexData, VertexUpdate, EdgeDataMatrixSOA, EdgeDataUpdate>>& faimGraph, const std::shared_ptr<Config>& config, bool tripleAimGraph);