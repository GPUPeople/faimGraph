//------------------------------------------------------------------------------
// EdgeQuery.cu
//
// faimGraph
//
//------------------------------------------------------------------------------
//

#include <iostream>
#include <fstream>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "faimGraph.h"
#include "EdgeUpdate.h"
#include "MemoryManager.h"
#include "ConfigurationParser.h"

//------------------------------------------------------------------------------
// Device funtionality
//------------------------------------------------------------------------------
//

//------------------------------------------------------------------------------
// 
template <typename VertexDataType, typename EdgeDataType>
__global__ void d_resolveQueries(MemoryManager* memory_manager,
                                memory_t* memory,
                                int page_size,
                                EdgeDataUpdate* queries,
                                int batch_size,
                                bool* query_results)
{
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid >= batch_size)
    return;

  VertexDataType* vertices = (VertexDataType*)memory;
  vertex_t edges_per_page = memory_manager->edges_per_page;
  
  // Retrieve query
  EdgeDataUpdate query = queries[tid];
  VertexDataType vertex = vertices[query.source];
  bool query_result = false;

  if (vertex.host_identifier == DELETIONMARKER)
  {
    // This is a deleted vertex
    printf("Deleted Vertex at %d\n", query.source);
    return;
  }

  AdjacencyIterator<EdgeDataType> adjacency_iterator(pageAccess<EdgeDataType>(memory, vertex.mem_index, page_size, memory_manager->start_index));

  // Search query
  for (int i = 0; i < vertex.neighbours; ++i)
  {
    if (adjacency_iterator.getDestination() == query.update.destination)
    {
      query_result = true;
      break;
    }      
    adjacency_iterator.advanceIterator(i, edges_per_page, memory, page_size, memory_manager->start_index);
  }

  // Return query result
  query_results[tid] = query_result;

  return;
}

#define MULTIPLICATOR 4

//------------------------------------------------------------------------------
// 
template <typename VertexDataType, typename EdgeDataType>
__global__ void d_resolveQueriesWarpsized(MemoryManager* memory_manager,
                                          memory_t* memory,
                                          int page_size,
                                          EdgeDataUpdate* queries,
                                          int batch_size,
                                          bool* query_results)
{
  int warpID = threadIdx.x / WARPSIZE;
  int wid = (blockIdx.x * MULTIPLICATOR) + warpID;
  int threadID = threadIdx.x - (warpID * WARPSIZE);
  vertex_t edges_per_page = memory_manager->edges_per_page;
  // Outside threads per block (because of indexing structure we use 31 threads)
  if ((threadID >= edges_per_page) || (wid >= batch_size))
    return;

  VertexDataType* vertices = (VertexDataType*)memory;

  __shared__ AdjacencyIterator<EdgeDataType> adjacency_iterator[MULTIPLICATOR];
  __shared__ EdgeDataUpdate query[MULTIPLICATOR];
  __shared__ bool query_result[MULTIPLICATOR];
  __shared__ VertexDataType vertex[MULTIPLICATOR];

  // Retrieve query
  if (SINGLE_THREAD_MULTI)
  {
    query[warpID] = queries[wid];
    vertex[warpID] = vertices[query[warpID].source];
    query_result[warpID] = false;
    adjacency_iterator[warpID].setIterator(pageAccess<EdgeDataType>(memory, vertex[warpID].mem_index, page_size, memory_manager->start_index));
  }
  __syncwarp();

  if (vertex[warpID].host_identifier == DELETIONMARKER)
  {
    // This is a deleted vertex
    return;
  }  

  // Search query
  while (vertex[warpID].capacity >= edges_per_page)
  {
    if (adjacency_iterator[warpID].getDestinationAt(threadID) == query[warpID].update.destination)
    {
      query_result[warpID] = true;
    }
    __syncwarp();
    
    if(query_result[warpID])
    {
      break;
    }

    if (SINGLE_THREAD_MULTI)
    {
      vertex[warpID].capacity -= edges_per_page;
      if (vertex[warpID].capacity > edges_per_page)
      {
        adjacency_iterator[warpID].blockTraversalAbsolute(edges_per_page, memory, page_size, memory_manager->start_index);
      }        
    }
    __syncwarp();
  }

  // Return query result
  if (SINGLE_THREAD_MULTI)
  {
    query_results[wid] = query_result[warpID];
  }
  __syncwarp();

  return;
}

////------------------------------------------------------------------------------
//// 
//template <typename VertexDataType, typename EdgeDataType>
//__global__ void d_resolveQueriesWarpsizedtest(MemoryManager* memory_manager,
//                                              memory_t* memory,
//                                              int page_size,
//                                              EdgeDataUpdate* queries,
//                                              int batch_size,
//                                              bool* query_results)
//{
//  int warpID = threadIdx.x / WARPSIZE;
//  int wid = (blockIdx.x * MULTIPLICATOR) + warpID;
//  int threadID = threadIdx.x - (warpID * WARPSIZE);
//  vertex_t edges_per_page = memory_manager->edges_per_page;
//  // Outside threads per block (because of indexing structure we use 31 threads)
//  if ((threadID > edges_per_page) || (wid >= batch_size))
//    return;
//
//  VertexDataType* vertices = (VertexDataType*)memory;
//
//  __shared__ AdjacencyIterator<EdgeDataType> adjacency_iterator[MULTIPLICATOR];
//  __shared__ EdgeDataUpdate query[MULTIPLICATOR];
//  __shared__ bool query_result[MULTIPLICATOR];
//  __shared__ VertexDataType vertex[MULTIPLICATOR];
//  __shared__ index_t page_index[MULTIPLICATOR];
//
//  // Retrieve query
//  if (SINGLE_THREAD_MULTI)
//  {
//    query[warpID] = queries[wid];
//    vertex[warpID] = vertices[query[warpID].source];
//    query_result[warpID] = false;
//    adjacency_iterator[warpID].setIterator(pageAccess<EdgeDataType>(memory, vertex[warpID].mem_index, page_size, memory_manager->start_index));
//  }
//  __syncwarp();
//
//  if (vertex[warpID].host_identifier == DELETIONMARKER)
//  {
//    // This is a deleted vertex
//    return;
//  }
//
//  // Search query
//  vertex_t destination;
//  while (vertex[warpID].capacity >= edges_per_page)
//  {
//    destination = adjacency_iterator[warpID].getDestinationAt(threadID);
//    if (SINGLE_THREAD_INDEX)
//    {
//      page_index[warpID] = destination;
//    }
//    else
//    {
//      if (destination == query[warpID].update.destination)
//      {
//        query_result[warpID] = true;
//      }
//    }
//    __syncwarp();
//
//    if (query_result[warpID])
//    {
//      break;
//    }
//
//    if (SINGLE_THREAD_MULTI)
//    {
//      vertex[warpID].capacity -= edges_per_page;
//      if (vertex[warpID].capacity > edges_per_page)
//      {
//        adjacency_iterator[warpID].blockTraversalAbsolutePageIndex(edges_per_page, memory, page_size, memory_manager->start_index, page_index[warpID]);
//      }
//    }
//    __syncwarp();
//  }
//
//  // Return query result
//  if (SINGLE_THREAD_MULTI)
//  {
//    query_results[wid] = query_result[warpID];
//  }
//  __syncwarp();
//
//  return;
//}

//------------------------------------------------------------------------------
// 
template <typename VertexDataType, typename EdgeDataType>
__global__ void d_resolveQueriesVertexCentric(MemoryManager* memory_manager,
                                              memory_t* memory,
                                              int page_size,
                                              EdgeDataUpdate* queries,
                                              int batch_size,
                                              bool* query_results,
                                              index_t* update_src_offsets)
{
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid >= memory_manager->next_free_vertex_index)
    return;

  index_t index_offset = update_src_offsets[tid];
  index_t number_queries = update_src_offsets[tid + 1] - index_offset;

  if (number_queries == 0)
    return;

  // Now just threads that actually work on updates should be left, tid corresponds to the src vertex that is being modified
  // Gather pointer
  VertexDataType* vertices = (VertexDataType*)memory;
  vertex_t edges_per_page = memory_manager->edges_per_page;
  vertex_t neighbours = vertices[tid].neighbours;

  if (neighbours == 0)
    return;

  AdjacencyIterator<EdgeDataType> adjacency_iterator(pageAccess<EdgeDataType>(memory, vertices[tid].mem_index, page_size, memory_manager->start_index));

  for (int i = 0; i < neighbours; ++i)
  {
    vertex_t adj_dest = adjacency_iterator.getDestination();
    index_t search_index = d_binaryQuerySearch(queries, adj_dest, index_offset, number_queries, adjacency_iterator);
    if (search_index != DELETIONMARKER)
    {
      query_results[search_index] = true;
    }
    adjacency_iterator.advanceIterator(i, edges_per_page, memory, page_size, memory_manager->start_index);
  }

  return;
}


//------------------------------------------------------------------------------
// Host funtionality
//------------------------------------------------------------------------------
//

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType>
void EdgeQueryManager<VertexDataType, EdgeDataType>::deviceQuery(std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config)
{
  int batch_size = queries->edge_update.size();
  int block_size = 256;
  int grid_size = (batch_size / block_size) + 1;

  // Get temporary memory for queries and results on device
  TemporaryMemoryAccessHeap temp_memory_dispenser(memory_manager.get(), memory_manager->next_free_vertex_index, sizeof(VertexDataType));
  queries->d_edge_update = temp_memory_dispenser.getTemporaryMemory<EdgeDataUpdate>(batch_size);
  d_query_results = temp_memory_dispenser.getTemporaryMemory<bool>(batch_size);

  // Copy queries to device
  HANDLE_ERROR(cudaMemcpy(queries->d_edge_update,
                          queries->edge_update.data(),
                          sizeof(EdgeDataUpdate) * queries->edge_update.size(),
                          cudaMemcpyHostToDevice));

  if (this->config == QueryKernelConfig::STANDARD)
  {
    d_resolveQueries<VertexDataType, EdgeDataType> << <grid_size, block_size >> > ((MemoryManager*)memory_manager->d_memory,
                                                                                    memory_manager->d_data,
                                                                                    memory_manager->page_size,
                                                                                    queries->d_edge_update,
                                                                                    batch_size,
                                                                                    d_query_results);
  }
  else if (this->config == QueryKernelConfig::WARPSIZED)
  {
    block_size = WARPSIZE * MULTIPLICATOR;
    grid_size = (batch_size / MULTIPLICATOR) + 1;
    d_resolveQueriesWarpsized<VertexDataType, EdgeDataType> << <grid_size, block_size >> > ((MemoryManager*)memory_manager->d_memory,
                                                                                            memory_manager->d_data,
                                                                                            memory_manager->page_size,
                                                                                            queries->d_edge_update,
                                                                                            batch_size,
                                                                                            d_query_results);
  }
  else if (this->config == QueryKernelConfig::VERTEXCENTRIC)
  {
    block_size = 256;
    grid_size = (memory_manager->next_free_vertex_index / block_size) + 1;

    HANDLE_ERROR(cudaMemset(d_query_results,
                            false,
                            sizeof(bool) * queries->edge_update.size()));

    thrust::device_ptr<EdgeDataUpdate> th_edge_updates((EdgeDataUpdate*)(queries->d_edge_update));
    thrust::sort(th_edge_updates, th_edge_updates + batch_size);

    auto preprocessed = edgeQueryPreprocessing(memory_manager, config);

    d_resolveQueriesVertexCentric<VertexDataType, EdgeDataType> << <grid_size, block_size >> > ((MemoryManager*)memory_manager->d_memory,
                                                                                                memory_manager->d_data,
                                                                                                memory_manager->page_size,
                                                                                                queries->d_edge_update,
                                                                                                batch_size,
                                                                                                d_query_results,
                                                                                                preprocessed->d_update_src_offsets);
  }

  // Copy query results back
  HANDLE_ERROR(cudaMemcpy(query_results.get(),
                          d_query_results,
                          sizeof(bool) * queries->edge_update.size(),
                          cudaMemcpyDeviceToHost));
  return;
}

template void EdgeQueryManager<VertexData, EdgeData>::deviceQuery(std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config);
template void EdgeQueryManager<VertexDataWeight, EdgeDataWeight>::deviceQuery(std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config);
template void EdgeQueryManager<VertexDataSemantic, EdgeDataSemantic>::deviceQuery(std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config);
template void EdgeQueryManager<VertexData, EdgeDataSOA>::deviceQuery(std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config);
template void EdgeQueryManager<VertexDataWeight, EdgeDataWeightSOA>::deviceQuery(std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config);
template void EdgeQueryManager<VertexDataSemantic, EdgeDataSemanticSOA>::deviceQuery(std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config);

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType>
void EdgeQueryManager<VertexDataType, EdgeDataType>::generateQueries(vertex_t number_vertices, vertex_t batch_size, unsigned int seed, unsigned int range, unsigned int offset)
{
	std::unique_ptr<EdgeUpdateBatch<EdgeDataUpdate>> query(std::make_unique<EdgeUpdateBatch<EdgeDataUpdate>>());

	// Generate random edge updates
	srand(seed + 100);
	for (unsigned int i = 0; i < batch_size; ++i)
	{
    EdgeDataUpdate edge_update_data;
    vertex_t intermediate = rand() % ((range && (range < number_vertices)) ? range : number_vertices);
    vertex_t source;
    if(offset + intermediate < number_vertices)
      source = offset + intermediate;
    else
      source = intermediate;
		edge_update_data.source = source;
		edge_update_data.update.destination = rand() % number_vertices;
		query->edge_update.push_back(edge_update_data);
	}

  receiveQueries(std::move(query));
  query_results = std::make_unique<bool[]>(batch_size);

	return;
}

template void EdgeQueryManager<VertexData, EdgeData>::generateQueries(vertex_t number_vertices, vertex_t batch_size, unsigned int seed, unsigned int range, unsigned int offset);
template void EdgeQueryManager<VertexDataWeight, EdgeDataWeight>::generateQueries(vertex_t number_vertices, vertex_t batch_size, unsigned int seed, unsigned int range, unsigned int offset);
template void EdgeQueryManager<VertexDataSemantic, EdgeDataSemantic>::generateQueries(vertex_t number_vertices, vertex_t batch_size, unsigned int seed, unsigned int range, unsigned int offset);
template void EdgeQueryManager<VertexData, EdgeDataSOA>::generateQueries(vertex_t number_vertices, vertex_t batch_size, unsigned int seed, unsigned int range, unsigned int offset);
template void EdgeQueryManager<VertexDataWeight, EdgeDataWeightSOA>::generateQueries(vertex_t number_vertices, vertex_t batch_size, unsigned int seed, unsigned int range, unsigned int offset);
template void EdgeQueryManager<VertexDataSemantic, EdgeDataSemanticSOA>::generateQueries(vertex_t number_vertices, vertex_t batch_size, unsigned int seed, unsigned int range, unsigned int offset);