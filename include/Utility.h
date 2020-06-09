//------------------------------------------------------------------------------
// Utility.h
//
// faimGraph
//
//------------------------------------------------------------------------------
//

#pragma once

#include "Definitions.h"
#include "MemoryLayout.h"
#include <stdio.h>

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void queryAndPrintDeviceProperties();
//------------------------------------------------------------------------------
void inline start_clock(cudaEvent_t &start, cudaEvent_t &end)
{
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&end));
    HANDLE_ERROR(cudaEventRecord(start,0));
}
//------------------------------------------------------------------------------
float inline end_clock(cudaEvent_t &start, cudaEvent_t &end)
{
    float time;
    HANDLE_ERROR(cudaEventRecord(end,0));
    HANDLE_ERROR(cudaEventSynchronize(end));
    HANDLE_ERROR(cudaEventElapsedTime(&time,start,end));
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(end));

    // Returns ms
    return time;
}

#ifdef __INTELLISENSE__
void __syncthreads();
void __syncwarp();
#endif

//------------------------------------------------------------------------------
//
template <typename EdgeDataType, typename IndexType, typename BlockType>
__forceinline__ __host__ __device__ EdgeDataType* pageAccess(memory_t* memory, IndexType page_index, BlockType page_size, uint64_t start_index)
{
  return (EdgeDataType*)&memory[(start_index - page_index) * page_size];
}

//------------------------------------------------------------------------------
// Set Adjacency
//------------------------------------------------------------------------------
//

//------------------------------------------------------------------------------
//
__forceinline__ __device__ void setAdjacency(EdgeData* edge_data, vertex_t* adjacency, int index, vertex_t edges_per_page)
{
    edge_data->destination = adjacency[index];
}

__forceinline__ __device__ void setAdjacency(EdgeDataWeight* edge_data, vertex_t* adjacency, int index, vertex_t edges_per_page)
{
    edge_data->destination = adjacency[index];
    edge_data->weight = 0;
}

__forceinline__ __device__ void setAdjacency(EdgeDataMatrix* edge_data, vertex_t* adjacency, int index, vertex_t edges_per_page)
{
  edge_data->destination = adjacency[index];
  edge_data->matrix_value = 1;
}

__forceinline__ __device__ void setAdjacency(EdgeDataMatrix* edge_data, vertex_t* adjacency, matrix_t* matrix_value, int index, vertex_t edges_per_page)
{
  edge_data->destination = adjacency[index];
  edge_data->matrix_value = matrix_value[index];
}

__forceinline__ __device__ void setAdjacency(EdgeDataSemantic* edge_data, vertex_t* adjacency, int index, vertex_t edges_per_page)
{
    edge_data->destination = adjacency[index];
    edge_data->weight = 0;
    edge_data->type = 0;
    edge_data->timestamp_1 = 0;
    edge_data->timestamp_2 = 0;
}

__forceinline__ __device__ void setAdjacency(EdgeDataSOA* edge_data, vertex_t* adjacency, int index, vertex_t edges_per_page)
{
  edge_data->destination = adjacency[index];
}

__forceinline__ __device__ void setAdjacency(EdgeDataWeightSOA* edge_data, vertex_t* adjacency, int index, vertex_t edges_per_page)
{
  edge_data->destination = adjacency[index];
  (edge_data + (edges_per_page * SOA_OFFSET_WEIGHT))->destination = 0;
}

__forceinline__ __device__ void setAdjacency(EdgeDataMatrixSOA* edge_data, vertex_t* adjacency, int index, vertex_t edges_per_page)
{
  edge_data->destination = adjacency[index];
  (edge_data + (edges_per_page * SOA_OFFSET_MATRIX))->destination = 0;
}

__forceinline__ __device__ void setAdjacency(EdgeDataMatrixSOA* edge_data, vertex_t* adjacency, matrix_t* matrix_value, int index, vertex_t edges_per_page)
{
  edge_data->destination = adjacency[index];
  (edge_data + (edges_per_page * SOA_OFFSET_MATRIX))->destination = matrix_value[index];
}

__forceinline__ __device__ void setAdjacency(EdgeDataSemanticSOA* edge_data, vertex_t* adjacency, int index, vertex_t edges_per_page)
{
  edge_data->destination = adjacency[index];
  (edge_data + (edges_per_page * SOA_OFFSET_WEIGHT))->destination = 0;
  (edge_data + (edges_per_page * SOA_OFFSET_TYPE))->destination = 0;
  (edge_data + (edges_per_page * SOA_OFFSET_TIMESTAMP1))->destination = 0;
  (edge_data + (edges_per_page * SOA_OFFSET_TIMESTAMP2))->destination = 0;
}

//------------------------------------------------------------------------------
// Update Adjacency
//------------------------------------------------------------------------------
//

//------------------------------------------------------------------------------
//
template <typename EdgeDataType>
__forceinline__ __device__ void updateAdjacency(EdgeDataType* edge_data, EdgeDataType& update_data, vertex_t edges_per_page)
{
  *edge_data = update_data;
}

template <typename EdgeDataType>
__forceinline__ __device__ void updateAdjacency(EdgeDataType* edge_data, EdgeDataType* update_data, vertex_t edges_per_page)
{
  *edge_data = *update_data;
}

template <typename EdgeDataType, typename UpdateDataType>
__forceinline__ __device__ void updateAdjacency(EdgeDataType* edge_data, UpdateDataType& update_data, vertex_t edges_per_page)
{
	*edge_data = update_data.update;
}

template <typename EdgeDataType, typename UpdateDataType>
__forceinline__ __device__ void updateAdjacency(EdgeDataType* edge_data, UpdateDataType* update_data, vertex_t edges_per_page)
{
	*edge_data = update_data->update;
}

//------------------------------------------------------------------------------
//
template <>
__forceinline__ __device__ void updateAdjacency<EdgeDataSOA, EdgeData>(EdgeDataSOA* edge_data, EdgeData& update_data, vertex_t edges_per_page)
{
  (edge_data)->destination = update_data.destination;
}

template <>
__forceinline__ __device__ void updateAdjacency<EdgeDataSOA, EdgeDataSOA>(EdgeDataSOA* edge_data, EdgeDataSOA* update_data, vertex_t edges_per_page)
{
  (edge_data)->destination = update_data->destination;
}

template <>
__forceinline__ __device__ void updateAdjacency<EdgeDataSOA, EdgeDataUpdate>(EdgeDataSOA* edge_data, EdgeDataUpdate& update_data, vertex_t edges_per_page)
{
	(edge_data)->destination = update_data.update.destination;
}

template <>
__forceinline__ __device__ void updateAdjacency<EdgeDataSOA, EdgeDataUpdate>(EdgeDataSOA* edge_data, EdgeDataUpdate* update_data, vertex_t edges_per_page)
{
	(edge_data)->destination = update_data->update.destination;
}

//------------------------------------------------------------------------------
//
template <>
__forceinline__ __device__ void updateAdjacency<EdgeDataWeightSOA, EdgeDataWeight>(EdgeDataWeightSOA* edge_data, EdgeDataWeight& update_data, vertex_t edges_per_page)
{
  (edge_data)->destination = update_data.destination;
  (edge_data + (edges_per_page * SOA_OFFSET_WEIGHT))->destination = update_data.weight;
}

template <>
__forceinline__ __device__ void updateAdjacency<EdgeDataWeightSOA, EdgeDataWeightSOA>(EdgeDataWeightSOA* edge_data, EdgeDataWeightSOA* update_data, vertex_t edges_per_page)
{
  (edge_data)->destination = update_data->destination;
  (edge_data + (edges_per_page * SOA_OFFSET_WEIGHT))->destination = (update_data + (edges_per_page * SOA_OFFSET_WEIGHT))->destination;
}

template <>
__forceinline__ __device__ void updateAdjacency<EdgeDataWeightSOA, EdgeDataWeightUpdate>(EdgeDataWeightSOA* edge_data, EdgeDataWeightUpdate& update_data, vertex_t edges_per_page)
{
	(edge_data)->destination = update_data.update.destination;
	(edge_data + (edges_per_page * SOA_OFFSET_WEIGHT))->destination = update_data.update.weight;
}

template <>
__forceinline__ __device__ void updateAdjacency<EdgeDataWeightSOA, EdgeDataWeightUpdate>(EdgeDataWeightSOA* edge_data, EdgeDataWeightUpdate* update_data, vertex_t edges_per_page)
{
	(edge_data)->destination = update_data->update.destination;
	(edge_data + (edges_per_page * SOA_OFFSET_WEIGHT))->destination = update_data->update.weight;
}

//------------------------------------------------------------------------------
//
template <>
__forceinline__ __device__ void updateAdjacency<EdgeDataMatrixSOA, EdgeDataMatrix>(EdgeDataMatrixSOA* edge_data, EdgeDataMatrix& update_data, vertex_t edges_per_page)
{
  (edge_data)->destination = update_data.destination;
  (edge_data + (edges_per_page * SOA_OFFSET_WEIGHT))->destination = update_data.matrix_value;
}

template <>
__forceinline__ __device__ void updateAdjacency<EdgeDataMatrixSOA, EdgeDataMatrixSOA>(EdgeDataMatrixSOA* edge_data, EdgeDataMatrixSOA* update_data, vertex_t edges_per_page)
{
  (edge_data)->destination = update_data->destination;
  (edge_data + (edges_per_page * SOA_OFFSET_WEIGHT))->destination = (update_data + (edges_per_page * SOA_OFFSET_WEIGHT))->destination;
}

template <>
__forceinline__ __device__ void updateAdjacency<EdgeDataMatrixSOA, EdgeDataMatrixUpdate>(EdgeDataMatrixSOA* edge_data, EdgeDataMatrixUpdate& update_data, vertex_t edges_per_page)
{
  (edge_data)->destination = update_data.update.destination;
  (edge_data + (edges_per_page * SOA_OFFSET_WEIGHT))->destination = update_data.update.matrix_value;
}

template <>
__forceinline__ __device__ void updateAdjacency<EdgeDataMatrixSOA, EdgeDataMatrixUpdate>(EdgeDataMatrixSOA* edge_data, EdgeDataMatrixUpdate* update_data, vertex_t edges_per_page)
{
  (edge_data)->destination = update_data->update.destination;
  (edge_data + (edges_per_page * SOA_OFFSET_WEIGHT))->destination = update_data->update.matrix_value;
}

//------------------------------------------------------------------------------
//
template <>
__forceinline__ __device__ void updateAdjacency<EdgeDataSemanticSOA, EdgeDataSemantic>(EdgeDataSemanticSOA* edge_data, EdgeDataSemantic& update_data, vertex_t edges_per_page)
{
  (edge_data)->destination = update_data.destination;
  (edge_data + (edges_per_page * SOA_OFFSET_WEIGHT))->destination = update_data.weight;
  (edge_data + (edges_per_page * SOA_OFFSET_TYPE))->destination = update_data.type;
  (edge_data + (edges_per_page * SOA_OFFSET_TIMESTAMP1))->destination = update_data.timestamp_1;
  (edge_data + (edges_per_page * SOA_OFFSET_TIMESTAMP2))->destination = update_data.timestamp_2;
}

template <>
__forceinline__ __device__ void updateAdjacency<EdgeDataSemanticSOA, EdgeDataSemanticSOA>(EdgeDataSemanticSOA* edge_data, EdgeDataSemanticSOA* update_data, vertex_t edges_per_page)
{
  (edge_data)->destination = update_data->destination;
  (edge_data + (edges_per_page * SOA_OFFSET_WEIGHT))->destination = (update_data + (edges_per_page * SOA_OFFSET_WEIGHT))->destination;
  (edge_data + (edges_per_page * SOA_OFFSET_TYPE))->destination = (update_data + (edges_per_page * SOA_OFFSET_TYPE))->destination;
  (edge_data + (edges_per_page * SOA_OFFSET_TIMESTAMP1))->destination = (update_data + (edges_per_page * SOA_OFFSET_TIMESTAMP1))->destination;
  (edge_data + (edges_per_page * SOA_OFFSET_TIMESTAMP2))->destination = (update_data + (edges_per_page * SOA_OFFSET_TIMESTAMP2))->destination;
}

template <>
__forceinline__ __device__ void updateAdjacency<EdgeDataSemanticSOA, EdgeDataSemanticUpdate>(EdgeDataSemanticSOA* edge_data, EdgeDataSemanticUpdate& update_data, vertex_t edges_per_page)
{
	(edge_data)->destination = update_data.update.destination;
	(edge_data + (edges_per_page * SOA_OFFSET_WEIGHT))->destination = update_data.update.weight;
	(edge_data + (edges_per_page * SOA_OFFSET_TYPE))->destination = update_data.update.type;
	(edge_data + (edges_per_page * SOA_OFFSET_TIMESTAMP1))->destination = update_data.update.timestamp_1;
	(edge_data + (edges_per_page * SOA_OFFSET_TIMESTAMP2))->destination = update_data.update.timestamp_2;
}

template <>
__forceinline__ __device__ void updateAdjacency<EdgeDataSemanticSOA, EdgeDataSemanticUpdate>(EdgeDataSemanticSOA* edge_data, EdgeDataSemanticUpdate* update_data, vertex_t edges_per_page)
{
	(edge_data)->destination = update_data->update.destination;
	(edge_data + (edges_per_page * SOA_OFFSET_WEIGHT))->destination = update_data->update.weight;
	(edge_data + (edges_per_page * SOA_OFFSET_TYPE))->destination = update_data->update.type;
	(edge_data + (edges_per_page * SOA_OFFSET_TIMESTAMP1))->destination = update_data->update.timestamp_1;
	(edge_data + (edges_per_page * SOA_OFFSET_TIMESTAMP2))->destination = update_data->update.timestamp_2;
}

//------------------------------------------------------------------------------
// Swap into local memory
//------------------------------------------------------------------------------
//

template <typename UpdateDataType>
__forceinline__ __device__ void swapIntoLocal(UpdateDataType* edge_data, UpdateDataType& local_element, vertex_t& edges_per_page)
{
  local_element = *edge_data;
}

template <typename EdgeDataType, typename UpdateDataType>
__forceinline__ __device__ void swapIntoLocal(EdgeDataType* edge_data, UpdateDataType& local_element, vertex_t& edges_per_page)
{
  local_element.update = *edge_data;
}

template <>
__forceinline__ __device__ void swapIntoLocal<EdgeDataSOA, EdgeDataUpdate>(EdgeDataSOA* edge_data, EdgeDataUpdate& local_element, vertex_t& edges_per_page)
{
  local_element.update.destination = edge_data->destination;
}

template <>
__forceinline__ __device__ void swapIntoLocal<EdgeDataWeightSOA, EdgeDataWeightUpdate>(EdgeDataWeightSOA* edge_data, EdgeDataWeightUpdate& local_element, vertex_t& edges_per_page)
{
  local_element.update.destination = edge_data->destination;
  local_element.update.weight = (edge_data + (edges_per_page * SOA_OFFSET_WEIGHT))->destination;
}

template <>
__forceinline__ __device__ void swapIntoLocal<EdgeDataMatrixSOA, EdgeDataMatrixUpdate>(EdgeDataMatrixSOA* edge_data, EdgeDataMatrixUpdate& local_element, vertex_t& edges_per_page)
{
  local_element.update.destination = edge_data->destination;
  local_element.update.matrix_value = (edge_data + (edges_per_page * SOA_OFFSET_MATRIX))->destination;
}

template <>
__forceinline__ __device__ void swapIntoLocal<EdgeDataSemanticSOA, EdgeDataSemanticUpdate>(EdgeDataSemanticSOA* edge_data, EdgeDataSemanticUpdate& local_element, vertex_t& edges_per_page)
{
  local_element.update.destination = edge_data->destination;
  local_element.update.weight = (edge_data + (edges_per_page * SOA_OFFSET_WEIGHT))->destination;
  local_element.update.type = (edge_data + (edges_per_page * SOA_OFFSET_TYPE))->destination;
  local_element.update.timestamp_1 = (edge_data + (edges_per_page * SOA_OFFSET_TIMESTAMP1))->destination;
  local_element.update.timestamp_2 = (edge_data + (edges_per_page * SOA_OFFSET_TIMESTAMP2))->destination;
}


//------------------------------------------------------------------------------
// Set DeletionMarker
//------------------------------------------------------------------------------
//

//------------------------------------------------------------------------------
//
template <typename EdgeDataType>
__forceinline__ __device__ void setDeletionMarker(EdgeDataType* edge_data, vertex_t edges_per_page)
{
  edge_data->destination = DELETIONMARKER;
}

//------------------------------------------------------------------------------
// Set VertexData
//------------------------------------------------------------------------------
//

__forceinline__ __device__ void setupVertex(VertexData& vertex, VertexUpdate& update, index_t page_index, vertex_t edges_per_page)
{
  vertex.locking = UNLOCK;
  vertex.mem_index = page_index;
  vertex.neighbours = 0;
  vertex.capacity = edges_per_page;
  vertex.host_identifier = update.identifier;
}

__forceinline__ __device__ void setupVertex(VertexDataWeight& vertex, VertexUpdateWeight& update, index_t page_index, vertex_t edges_per_page)
{
  vertex.locking = UNLOCK;
  vertex.mem_index = page_index;
  vertex.neighbours = 0;
  vertex.capacity = edges_per_page;
  vertex.host_identifier = update.identifier;
  vertex.weight = update.weight;
}

__forceinline__ __device__ void setupVertex(VertexDataSemantic& vertex, VertexUpdateSemantic& update, index_t page_index, vertex_t edges_per_page)
{
  vertex.locking = UNLOCK;
  vertex.mem_index = page_index;
  vertex.neighbours = 0;
  vertex.capacity = edges_per_page;
  vertex.host_identifier = update.identifier;
  vertex.weight = update.weight;
  vertex.type = update.type;
}


//------------------------------------------------------------------------------
// Template specification for different EdgeDataTypes
//

template <typename EdgeDataType>
__forceinline__ __device__ void pointerHandlingSetup(EdgeDataType*& adjacency_list, memory_t* memory, vertex_t& block_index, int page_size, vertex_t edges_per_page, uint64_t start_index)
{
  *((index_t*)adjacency_list) = ++block_index;
  adjacency_list = pageAccess<EdgeDataType>(memory, block_index, page_size, start_index);
}

template <typename EdgeDataType>
__forceinline__ __device__ void pointerHandlingSetupDoubleLinked(EdgeDataType*& adjacency_list, memory_t* memory, vertex_t& block_index, int page_size, vertex_t edges_per_page, uint64_t start_index)
{
  *(((index_t*)adjacency_list) + 1) = block_index++;
  *((index_t*)adjacency_list) = block_index;
  adjacency_list = pageAccess<EdgeDataType>(memory, block_index, page_size, start_index);
}

template <>
__forceinline__ __device__ void pointerHandlingSetup<EdgeDataWeightSOA>(EdgeDataWeightSOA*& adjacency_list, memory_t* edgedata_start_index, vertex_t& block_index, int page_size, vertex_t edges_per_page, uint64_t start_index)
{
  *((index_t*)((adjacency_list) + (edges_per_page * SOA_OFFSET_WEIGHT))) = ++block_index;
  adjacency_list = pageAccess<EdgeDataWeightSOA>(edgedata_start_index, block_index, page_size, start_index);
}

template <>
__forceinline__ __device__ void pointerHandlingSetupDoubleLinked<EdgeDataWeightSOA>(EdgeDataWeightSOA*& adjacency_list, memory_t* edgedata_start_index, vertex_t& block_index, int page_size, vertex_t edges_per_page, uint64_t start_index)
{
  *(((index_t*)((adjacency_list)+(edges_per_page * SOA_OFFSET_WEIGHT))) + 1) = block_index++;
  *((index_t*)((adjacency_list)+(edges_per_page * SOA_OFFSET_WEIGHT))) = block_index;
  adjacency_list = pageAccess<EdgeDataWeightSOA>(edgedata_start_index, block_index, page_size, start_index);
}

template <>
__forceinline__ __device__ void pointerHandlingSetup<EdgeDataMatrixSOA>(EdgeDataMatrixSOA*& adjacency_list, memory_t* edgedata_start_index, vertex_t& block_index, int page_size, vertex_t edges_per_page, uint64_t start_index)
{
  *((index_t*)((adjacency_list)+(edges_per_page * SOA_OFFSET_MATRIX))) = ++block_index;
  adjacency_list = pageAccess<EdgeDataMatrixSOA>(edgedata_start_index, block_index, page_size, start_index);
}

template <>
__forceinline__ __device__ void pointerHandlingSetupDoubleLinked<EdgeDataMatrixSOA>(EdgeDataMatrixSOA*& adjacency_list, memory_t* edgedata_start_index, vertex_t& block_index, int page_size, vertex_t edges_per_page, uint64_t start_index)
{
  *(((index_t*)((adjacency_list)+(edges_per_page * SOA_OFFSET_MATRIX))) + 1) = block_index++;
  *((index_t*)((adjacency_list)+(edges_per_page * SOA_OFFSET_MATRIX))) = block_index;
  adjacency_list = pageAccess<EdgeDataMatrixSOA>(edgedata_start_index, block_index, page_size, start_index);
}

template <>
__forceinline__ __device__ void pointerHandlingSetup<EdgeDataSemanticSOA>(EdgeDataSemanticSOA*& adjacency_list, memory_t* edgedata_start_index, vertex_t& block_index, int page_size, vertex_t edges_per_page, uint64_t start_index)
{
  *((index_t*)((adjacency_list)+ (edges_per_page * SOA_OFFSET_TIMESTAMP2))) = ++block_index;
  adjacency_list = pageAccess<EdgeDataSemanticSOA>(edgedata_start_index, block_index, page_size, start_index);
}

template <>
__forceinline__ __device__ void pointerHandlingSetupDoubleLinked<EdgeDataSemanticSOA>(EdgeDataSemanticSOA*& adjacency_list, memory_t* edgedata_start_index, vertex_t& block_index, int page_size, vertex_t edges_per_page, uint64_t start_index)
{
  *(((index_t*)((adjacency_list)+(edges_per_page * SOA_OFFSET_TIMESTAMP2))) + 1) = block_index++;
  *((index_t*)((adjacency_list)+(edges_per_page * SOA_OFFSET_TIMESTAMP2))) = block_index;
  adjacency_list = pageAccess<EdgeDataSemanticSOA>(edgedata_start_index, block_index, page_size, start_index);
}

//------------------------------------------------------------------------------
// Traversal
//
template <typename EdgeDataType>
__forceinline__ __device__ __host__ void pointerHandlingTraverse(EdgeDataType*& adjacency_list, memory_t* memory, int page_size, vertex_t edges_per_page, uint64_t& start_index)
{
  adjacency_list = pageAccess<EdgeDataType>(memory, *((index_t*)adjacency_list), page_size, start_index);
}

template <>
__forceinline__ __device__ __host__ void pointerHandlingTraverse<EdgeDataWeightSOA>(EdgeDataWeightSOA*& adjacency_list, memory_t* edgedata_start_index, int page_size, vertex_t edges_per_page, uint64_t& start_index)
{
  adjacency_list = pageAccess<EdgeDataWeightSOA>(edgedata_start_index, *((index_t*)((adjacency_list) + (edges_per_page * (SOA_OFFSET_WEIGHT)))), page_size, start_index);
}

template <>
__forceinline__ __device__ __host__ void pointerHandlingTraverse<EdgeDataMatrixSOA>(EdgeDataMatrixSOA*& adjacency_list, memory_t* edgedata_start_index, int page_size, vertex_t edges_per_page, uint64_t& start_index)
{
  adjacency_list = pageAccess<EdgeDataMatrixSOA>(edgedata_start_index, *((index_t*)((adjacency_list)+(edges_per_page * (SOA_OFFSET_MATRIX)))), page_size, start_index);
}

template <>
__forceinline__ __device__ __host__ void pointerHandlingTraverse<EdgeDataSemanticSOA>(EdgeDataSemanticSOA*& adjacency_list, memory_t* edgedata_start_index, int page_size, vertex_t edges_per_page, uint64_t& start_index)
{
  adjacency_list = pageAccess<EdgeDataSemanticSOA>(edgedata_start_index, *((index_t*)((adjacency_list) + (edges_per_page * (SOA_OFFSET_TIMESTAMP2)))), page_size, start_index);
}

//------------------------------------------------------------------------------
// Traversal with given page index
//
template <typename EdgeDataType>
__forceinline__ __device__ void pointerHandlingTraverse(EdgeDataType*& adjacency_list, memory_t* memory, int page_size, vertex_t edges_per_page, uint64_t& start_index, index_t& page_index)
{
  adjacency_list = pageAccess<EdgeDataType>(memory, page_index, page_size, start_index);
}

//------------------------------------------------------------------------------
// Get PageIndex when traversal is finished
//

template <typename EdgeDataType>
__forceinline__ __device__ index_t* getEdgeBlockIndex(EdgeDataType* adjacency, int edges_per_page)
{
  return (index_t*)adjacency;
}

template <>
__forceinline__ __device__ index_t* getEdgeBlockIndex<EdgeDataWeightSOA>(EdgeDataWeightSOA* adjacency, int edges_per_page)
{
    return (index_t*)(adjacency + (edges_per_page * (SOA_OFFSET_WEIGHT)));
}

template <>
__forceinline__ __device__ index_t* getEdgeBlockIndex<EdgeDataMatrixSOA>(EdgeDataMatrixSOA* adjacency, int edges_per_page)
{
  return (index_t*)(adjacency + (edges_per_page * (SOA_OFFSET_MATRIX)));
}

template <>
__forceinline__ __device__ index_t* getEdgeBlockIndex<EdgeDataSemanticSOA>(EdgeDataSemanticSOA* adjacency, int edges_per_page)
{
    return (index_t*)(adjacency + (edges_per_page * (SOA_OFFSET_TIMESTAMP2)));
}

//------------------------------------------------------------------------------
// Get PageIndex when pointer is on page start
//

template <typename EdgeDataType>
__forceinline__ __device__ index_t* getEdgeBlockIndexAbsolute(EdgeDataType* adjacency, int edges_per_page)
{
  return (index_t*)(adjacency + edges_per_page);
}

template <>
__forceinline__ __device__ index_t* getEdgeBlockIndexAbsolute<EdgeDataWeightSOA>(EdgeDataWeightSOA* adjacency, int edges_per_page)
{
  return (index_t*)((adjacency + edges_per_page) + (edges_per_page * (SOA_OFFSET_WEIGHT)));
}

template <>
__forceinline__ __device__ index_t* getEdgeBlockIndexAbsolute<EdgeDataMatrixSOA>(EdgeDataMatrixSOA* adjacency, int edges_per_page)
{
  return (index_t*)((adjacency + edges_per_page) + (edges_per_page * (SOA_OFFSET_MATRIX)));
}

template <>
__forceinline__ __device__ index_t* getEdgeBlockIndexAbsolute<EdgeDataSemanticSOA>(EdgeDataSemanticSOA* adjacency, int edges_per_page)
{
  return (index_t*)((adjacency + edges_per_page) + (edges_per_page * (SOA_OFFSET_TIMESTAMP2)));
}

//------------------------------------------------------------------------------
// Iterator class
//------------------------------------------------------------------------------
//
template <typename EdgeDataType>
class AdjacencyIterator
{
public:
  __device__ AdjacencyIterator(EdgeDataType* it): iterator{it}{}
  __device__ AdjacencyIterator(const AdjacencyIterator<EdgeDataType>& it): iterator{it.iterator}{}
  __device__ AdjacencyIterator(){}

  __forceinline__ __device__ void setIterator(AdjacencyIterator<EdgeDataType>& it) { iterator = it.iterator; }

  __forceinline__ __device__ void setIterator(EdgeDataType* it) { iterator = it; }

  __forceinline__ __device__ bool isValid() { return iterator != nullptr; }

  __forceinline__ __device__ bool isNotValid() { return iterator == nullptr; }

  __forceinline__ __device__ EdgeDataType*& getIterator() { return iterator; }

  __forceinline__ __device__ EdgeDataType* getIteratorAt(index_t index) { return iterator + index; }

  __forceinline__ __device__ vertex_t getDestination() { return iterator->destination; }

  __forceinline__ __device__ EdgeDataType getElement() { return *iterator; }

  __forceinline__ __device__ EdgeDataType getElementAt(index_t index) { return iterator[index]; }

  __forceinline__ __device__ EdgeDataType* getElementPtr() { return iterator; }

  __forceinline__ __device__ EdgeDataType* getElementPtrAt(index_t index) { return &iterator[index]; }

  __forceinline__ __device__ vertex_t* getDestinationPtr() { return &(iterator->destination); }

  __forceinline__ __device__ vertex_t getDestinationAt(index_t index) { return iterator[index].destination; }

  __forceinline__ __device__ vertex_t* getDestinationPtrAt(index_t index) { return &(iterator[index].destination); }

  __forceinline__ __device__ index_t* getPageIndexPtr(vertex_t& edges_per_page) { return getEdgeBlockIndex(iterator, edges_per_page); }

  __forceinline__ __device__ index_t getPageIndex(vertex_t& edges_per_page) { return *getEdgeBlockIndex(iterator, edges_per_page); }

  __forceinline__ __device__ index_t* getPageIndexPtrAbsolute(vertex_t& edges_per_page) { return getEdgeBlockIndexAbsolute(iterator, edges_per_page); }

  __forceinline__ __device__ index_t getPageIndexAbsolute(vertex_t& edges_per_page) { return *getEdgeBlockIndexAbsolute(iterator, edges_per_page); }

  __forceinline__ __device__ void setDestination(vertex_t value) { iterator->destination = value; }

  __forceinline__ __device__ void setDestinationAt(index_t index, vertex_t value) { iterator[index].destination = value; }

  __forceinline__ __device__ void setDestination(AdjacencyIterator<EdgeDataType>& it) { iterator->destination = it.iterator->destination; }

  __forceinline__ __device__ AdjacencyIterator& operator++() { ++iterator; return *this;}

  __forceinline__ __device__ AdjacencyIterator operator++(int) {AdjacencyIterator result(*this); ++iterator; return result;}

  __forceinline__ __device__ AdjacencyIterator& operator+=(int edges_per_page) { iterator += edges_per_page; return *this; }

  __forceinline__ __device__ AdjacencyIterator& operator-=(int edges_per_page) { iterator -= edges_per_page; return *this; }

  __forceinline__ __device__ vertex_t operator[](int index)
  {
      return iterator[index].destination;
  }

  __forceinline__ __device__ vertex_t at(int index, memory_t*& memory, int page_size, uint64_t start_index, vertex_t edges_per_page)
  {
    if (index <= edges_per_page)
    {
      return iterator[index].destination;
    }
    else
    {
      // We need traversal
      EdgeDataType* tmp_iterator = iterator;
      while (index > edges_per_page)
      {
        tmp_iterator += edges_per_page;
        pointerHandlingTraverse(tmp_iterator, memory, page_size, edges_per_page, start_index);
        index -= edges_per_page;
      }
      return tmp_iterator[index].destination;
    }
  }

  __forceinline__ __device__ void adjacencySetup(int& loop_index, vertex_t& edges_per_page, memory_t*& memory, int& page_size, uint64_t& start_index, vertex_t* adjacency, int offset_index, vertex_t& block_index)
  {
    setAdjacency(iterator, adjacency, offset_index + loop_index, edges_per_page);
    ++iterator;
    if (((loop_index) % (edges_per_page)) == (edges_per_page - 1))
    {
      pointerHandlingSetup(iterator, memory, block_index, page_size, edges_per_page, start_index);
    }
  }

  __forceinline__ __device__ void adjacencySetup(int& loop_index, vertex_t& edges_per_page, memory_t*& memory, int& page_size, uint64_t& start_index, vertex_t* adjacency, matrix_t* matrix_values, int offset_index, vertex_t& block_index)
  {
    setAdjacency(iterator, adjacency, matrix_values, offset_index + loop_index, edges_per_page);
    ++iterator;
    if (((loop_index) % (edges_per_page)) == (edges_per_page - 1))
    {
      pointerHandlingSetup(iterator, memory, block_index, page_size, edges_per_page, start_index);
    }
  }

  __forceinline__ __device__ void adjacencySetupDoubleLinked(int& loop_index, vertex_t& edges_per_page, memory_t*& memory, int& page_size, uint64_t& start_index, vertex_t* adjacency, int offset_index, vertex_t& block_index)
  {
    setAdjacency(iterator, adjacency, offset_index + loop_index, edges_per_page);
    ++iterator;
    if (((loop_index) % (edges_per_page)) == (edges_per_page - 1))
    {
      pointerHandlingSetupDoubleLinked(iterator, memory, block_index, page_size, edges_per_page, start_index);
    }
  }

  __forceinline__ __device__ void adjacencySetupDoubleLinked(int& loop_index, vertex_t& edges_per_page, memory_t*& memory, int& page_size, uint64_t& start_index, vertex_t* adjacency, matrix_t* matrix_values, int offset_index, vertex_t& block_index)
  {
    setAdjacency(iterator, adjacency, matrix_values, offset_index + loop_index, edges_per_page);
    ++iterator;
    if (((loop_index) % (edges_per_page)) == (edges_per_page - 1))
    {
      pointerHandlingSetupDoubleLinked(iterator, memory, block_index, page_size, edges_per_page, start_index);
    }
  }

  __forceinline__ __device__ void advanceIterator(int& loop_index, vertex_t& edges_per_page, memory_t*& memory, int& page_size, uint64_t& start_index)
  {
    ++iterator;
    if (((loop_index) % (edges_per_page)) == (edges_per_page - 1))
    {
      // Edgeblock is full, place pointer to next block, reset adjacency_list pointer to next block
      pointerHandlingTraverse(iterator, memory, page_size, edges_per_page, start_index);
    }
  }

  __forceinline__ __device__ void blockTraversalAbsolute(vertex_t& edges_per_page, memory_t*& memory, int& page_size, uint64_t& start_index)
  {
    iterator += edges_per_page;
    pointerHandlingTraverse(iterator, memory, page_size, edges_per_page, start_index);
  }

  __forceinline__ __device__ void blockTraversalAbsolutePageIndex(vertex_t& edges_per_page, memory_t*& memory, int& page_size, uint64_t& start_index, index_t& page_index)
  {
    pointerHandlingTraverse(iterator, memory, page_size, edges_per_page, start_index, page_index);
  }

  __forceinline__ __device__ void blockTraversalAbsolute(vertex_t& edges_per_page, memory_t*& memory, int& page_size, uint64_t& start_index, index_t& page_index)
  {
    iterator += edges_per_page;
    page_index = getPageIndex(edges_per_page);
    pointerHandlingTraverse(iterator, memory, page_size, edges_per_page, start_index);
  }

  __forceinline__ __device__ void cleanPageExclusive(vertex_t& edges_per_page)
  {
    ++iterator;
    for (int i = 1; i < edges_per_page; ++i)
    {
      setDeletionMarker(iterator, edges_per_page);
      ++iterator;
    }
  }

  __forceinline__ __device__ void cleanPageInclusive(vertex_t& edges_per_page)
  {
    for (int i = 0; i < edges_per_page; ++i)
    {
      setDeletionMarker(iterator, edges_per_page);
      ++iterator;
    }
  }

  template <typename PageIndexDataType, typename CapacityDataType>
  __forceinline__ __device__ void advanceIteratorEndCheck(int& loop_index, vertex_t& edges_per_page, memory_t*& memory, PageIndexDataType& page_size, uint64_t& start_index, CapacityDataType& capacity)
  {
    ++iterator;
    if (((loop_index) % (edges_per_page)) == (edges_per_page - 1) && loop_index != (capacity - 1))
    {
      // Edgeblock is full, place pointer to next block, reset adjacency_list pointer to next block
      pointerHandlingTraverse(iterator, memory, page_size, edges_per_page, start_index);
    }
  }

  __forceinline__ __device__ bool advanceIteratorEndCheckBoolReturn(int& loop_index, vertex_t& edges_per_page, memory_t*& memory, int& page_size, uint64_t& start_index, int& capacity, index_t& edge_block_index)
  {
    ++iterator;
    if (((loop_index) % (edges_per_page)) == (edges_per_page - 1) && loop_index != (capacity - 1))
    {
      // Edgeblock is full, place pointer to next block, reset adjacency_list pointer to next block
      edge_block_index = getPageIndex(edges_per_page);
      pointerHandlingTraverse(iterator, memory, page_size, edges_per_page, start_index);
      return true;
    }
    return false;
  }

  __forceinline__ __device__ void advanceIteratorEndCheck(int& loop_index, vertex_t& edges_per_page, memory_t*& memory, int& page_size, uint64_t& start_index, int& capacity, index_t& edge_block_index)
  {
    ++iterator;
    if (((loop_index) % (edges_per_page)) == (edges_per_page - 1) && loop_index != (capacity - 1))
    {
      // Edgeblock is full, place pointer to next block, reset adjacency_list pointer to next block
      edge_block_index = getPageIndex(edges_per_page);
      pointerHandlingTraverse(iterator, memory, page_size, edges_per_page, start_index);
    }
  }

  __forceinline__ __device__ void advanceIteratorDeletionCompaction(int& loop_index, vertex_t& edges_per_page, memory_t*& memory, int& page_size, uint64_t& start_index, index_t& edge_block_index, AdjacencyIterator<EdgeDataType>& search_list, vertex_t& shuffle_index)  {
    ++iterator;
    if (((loop_index) % (edges_per_page)) == (edges_per_page - 1))
    {
      // Edgeblock is full, place pointer to next block, reset adjacency_list pointer to next block
      edge_block_index = *(getEdgeBlockIndex(iterator, edges_per_page));
      pointerHandlingTraverse(iterator, memory, page_size, edges_per_page, start_index);
      search_list.setIterator(iterator);
      shuffle_index -= edges_per_page;
    }
  }

  __forceinline__ __device__ void advanceIteratorToIndex(vertex_t& edges_per_page, memory_t*& memory, int& page_size, uint64_t& start_index, index_t& edge_block_index, vertex_t& shuffle_index)
  {
    while (shuffle_index >= edges_per_page)
    {
      iterator += edges_per_page;
      edge_block_index = getPageIndex(edges_per_page);
      pointerHandlingTraverse(iterator, memory, page_size, edges_per_page, start_index);
      shuffle_index -= edges_per_page;
    }
  }

  __forceinline__ __device__ void advanceIteratorToIndex(vertex_t& edges_per_page, memory_t*& memory, int& page_size, uint64_t& start_index, vertex_t& shuffle_index, int& neighbours, int& capacity)
  {
    while (shuffle_index > edges_per_page)
    {
      iterator += edges_per_page;
      pointerHandlingTraverse(iterator, memory, page_size, edges_per_page, start_index);
      shuffle_index -= edges_per_page;
    }
    if(shuffle_index == edges_per_page && neighbours < capacity)
    {
      iterator += edges_per_page;
      pointerHandlingTraverse(iterator, memory, page_size, edges_per_page, start_index);
    }
    else
    {
      iterator += shuffle_index;
    }    
  }

protected:
  EdgeDataType* iterator;
};

//------------------------------------------------------------------------------
//
template <typename EdgeDataType>
__device__ void swap(EdgeDataType* a, EdgeDataType* b, vertex_t edges_per_page)
{
  EdgeDataType tmp;
  updateAdjacency(&tmp, a, edges_per_page);
  updateAdjacency(a, b, edges_per_page);
  updateAdjacency(b, &tmp, edges_per_page);
}

//------------------------------------------------------------------------------
// ReUseIterator class
//------------------------------------------------------------------------------
//
template <typename EdgeDataType>
class ReuseAdjacencyIterator : public AdjacencyIterator<EdgeDataType>
{
public:
  using AdjacencyIterator<EdgeDataType>::AdjacencyIterator; // Inherit constructor

  __device__ ReuseAdjacencyIterator(EdgeDataType* it) : AdjacencyIterator<EdgeDataType>(it), tmp_iterator {it} {}
  __device__ ReuseAdjacencyIterator(const ReuseAdjacencyIterator<EdgeDataType>& it) : AdjacencyIterator<EdgeDataType>(it), tmp_iterator{ it.iterator } {}
  __device__ ReuseAdjacencyIterator() {}

  __forceinline__ __device__ vertex_t at(int index, memory_t*& memory, int page_size, uint64_t start_index, vertex_t edges_per_page)
  {
    if (index <= edges_per_page)
    {
      return AdjacencyIterator<EdgeDataType>::iterator[index].destination;
    }
    else
    {
      // We need traversal
      tmp_iterator = AdjacencyIterator<EdgeDataType>::iterator;
      while (index > edges_per_page)
      {
        tmp_iterator += edges_per_page;
        pointerHandlingTraverse(tmp_iterator, memory, page_size, edges_per_page, start_index);
        index -= edges_per_page;
      }
      return tmp_iterator[index].destination;
    }
  }

private:
  EdgeDataType* tmp_iterator;
};

//------------------------------------------------------------------------------
//
template <typename VertexUpdateType>
__forceinline__ __device__ void d_binarySearch(VertexUpdateType* vertex_update_data, index_t search_element, int batch_size, index_t* deletion_helper)
{
  int lower_bound = 0;
  int upper_bound = batch_size - 1;
  index_t search_index;
  while (lower_bound <= upper_bound)
  {
    search_index = lower_bound + ((upper_bound - lower_bound) / 2);
    index_t update = vertex_update_data[search_index].identifier;

    // First check if we get a hit
    if (update == search_element)
    {
      // We have a duplicate, let's mark it for deletion and then finish
      deletion_helper[search_index] = DELETIONMARKER;
      break;
    }
    else if (update < search_element)
    {
      lower_bound = search_index + 1;
    }
    else
    {
      upper_bound = search_index - 1;
    }
  }
  return;
}

//------------------------------------------------------------------------------
//
template <typename VertexUpdateType, typename EdgeDataType>
__forceinline__ __device__ void d_binarySearch(VertexUpdateType* vertex_update_data, index_t search_element, int batch_size, AdjacencyIterator<EdgeDataType>& adjacency_iterator)
{
  int lower_bound = 0;
  int upper_bound = batch_size - 1;
  index_t search_index;
  while (lower_bound <= upper_bound)
  {
    search_index = lower_bound + ((upper_bound - lower_bound) / 2);
    index_t update = vertex_update_data[search_index].identifier;

    // First check if we get a hit
    if (update == search_element)
    {
      // We have a duplicate, let's mark it for deletion and then finish
      adjacency_iterator.setDestination(DELETIONMARKER);
      break;
    }
    else if (update < search_element)
    {
      lower_bound = search_index + 1;
    }
    else
    {
      upper_bound = search_index - 1;
    }
  }
  return;
}

//------------------------------------------------------------------------------
//
template <typename UpdateDataType>
__forceinline__ __device__ void d_binarySearch(UpdateDataType* edge_update_data, index_t search_element, index_t start_index, int number_updates, index_t* deletion_helper)
{
  int lower_bound = start_index;
  int upper_bound = start_index + (number_updates - 1);
  index_t search_index;
  while (lower_bound <= upper_bound)
  {
    search_index = lower_bound + ((upper_bound - lower_bound) / 2);
    index_t update = edge_update_data[search_index].update.destination;

    // First check if we get a hit
    if (update == search_element)
    {
      // We have a duplicate, let's mark it for deletion and then finish
      deletion_helper[search_index] = DELETIONMARKER;
      break;
    }
    else if (update < search_element)
    {
      lower_bound = search_index + 1;
    }
    else
    {
      upper_bound = search_index - 1;
    }
  }
  return;
}

//------------------------------------------------------------------------------
//
template <typename UpdateDataType, typename EdgeDataType>
__forceinline__ __device__ bool d_binarySearch(UpdateDataType* edge_update_data, index_t search_element, index_t start_index, int number_updates, AdjacencyIterator<EdgeDataType>& adjacency_iterator)
{
  int lower_bound = start_index;
  int upper_bound = start_index + (number_updates - 1);
  index_t search_index;
  while (lower_bound <= upper_bound)
  {
    search_index = lower_bound + ((upper_bound - lower_bound) / 2);
    index_t update = edge_update_data[search_index].update.destination;

    // First check if we get a hit
    if (update == search_element)
    {
      // We have a duplicate, let's mark it for deletion and then finish
      adjacency_iterator.setDestination(DELETIONMARKER);
      return true;
    }
    else if (update < search_element)
    {
      lower_bound = search_index + 1;
    }
    else
    {
      upper_bound = search_index - 1;
    }
  }
  return false;
}

//------------------------------------------------------------------------------
//
template <typename UpdateDataType, typename EdgeDataType>
__forceinline__ __device__ index_t d_binaryQuerySearch(UpdateDataType* edge_update_data, index_t search_element, index_t start_index, int number_updates, AdjacencyIterator<EdgeDataType>& adjacency_iterator)
{
  int lower_bound = start_index;
  int upper_bound = start_index + (number_updates - 1);
  index_t search_index;
  while (lower_bound <= upper_bound)
  {
    search_index = lower_bound + ((upper_bound - lower_bound) / 2);
    index_t update = edge_update_data[search_index].update.destination;

    // First check if we get a hit
    if (update == search_element)
    {
      // Search query found an element
      return search_index;
    }
    else if (update < search_element)
    {
      lower_bound = search_index + 1;
    }
    else
    {
      upper_bound = search_index - 1;
    }
  }
  return DELETIONMARKER;
}

//------------------------------------------------------------------------------
//
template <typename EdgeDataType>
__forceinline__ __device__ bool d_binarySearchOnPage(AdjacencyIterator<EdgeDataType>& adjacency_iterator, index_t search_element, int number_elements_to_check)
{
  int lower_bound = 0;
  int upper_bound = (number_elements_to_check - 1);
  index_t search_index;
  while (lower_bound <= upper_bound)
  {
    search_index = lower_bound + ((upper_bound - lower_bound) / 2);
    index_t element = adjacency_iterator.getDestinationAt(search_index);

    // First check if we get a hit
    if (element == search_element)
    {
      // We have a duplicate, let's mark it for deletion and then finish
      return true;
    }
    else if (element < search_element)
    {
      lower_bound = search_index + 1;
    }
    else
    {
      upper_bound = search_index - 1;
    }
  }
  return false;
}
