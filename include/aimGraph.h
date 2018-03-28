//------------------------------------------------------------------------------
// GPUStreaming.h
//
// Masterproject/-thesis GPU Streaming Graphs
//
// Authors: Martin Winter, 1130688
//------------------------------------------------------------------------------
//

#ifndef GPUSTREAMING_H
#define GPUSTREAMING_H

#include "Utility.h"
#include "EdgeUpdate.h"
#include "VertexUpdate.h"
#include "MemoryManager.h"
#include "ConfigurationParser.h"

#define MEMORYOVERALLOCATION 1.0f

// Forward declaration
class GraphParser;

template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename EdgeUpdateType>
class aimGraph
{
public:
  aimGraph(std::shared_ptr<Config> configuration, std::unique_ptr<GraphParser>& graph_parser) : 
    config{ std::move(configuration) },
    memory_manager{ std::make_unique<MemoryManager>(static_cast<uint64_t>(GIGABYTE * config->device_mem_size_), config, graph_parser) },
    edge_update_manager{ std::make_unique<EdgeUpdateManager<VertexDataType, EdgeDataType, EdgeUpdateType>>() }, 
    vertex_update_manager{std::make_unique<VertexUpdateManager<VertexDataType, VertexUpdateType>>()}{}
  
  // Setup
  void initializeMemory(std::unique_ptr<GraphParser>& graph_parser);
  void initializeaimGraphMatrix(std::unique_ptr<CSRMatrixData>& csr_matrix_data);
  void initializeaimGraphMatrix(std::unique_ptr<GraphParser>& graph_parser, unsigned int vertex_offset = 0);
  void initializeaimGraphEmptyMatrix(unsigned int number_rows, unsigned int vertex_offset = 0);

  // Updates
  void edgeInsertion();
  void edgeDeletion();
  void vertexInsertion(VertexMapper<index_t, index_t>& mapper);
  void vertexDeletion(VertexMapper<index_t, index_t>& mapper);

  // Verification
  std::unique_ptr<aimGraphCSR> verifyGraphStructure(std::unique_ptr<MemoryManager>& memory_manager);
  std::unique_ptr<aimGraphCSR> verifyMatrixStructure(std::unique_ptr<MemoryManager>& memory_manager, vertex_t vertex_offset = 0, vertex_t number_vertices = 0);
  bool compareGraphs(std::unique_ptr<GraphParser>& graph_parser, std::unique_ptr<aimGraphCSR>& verify_graph, bool duplicate_check);
  bool compareGraphs(std::unique_ptr<CSRMatrix>& csr_matrix, std::unique_ptr<aimGraphCSR>& verify_graph, bool duplicate_check);

public:
  std::shared_ptr<Config> config;
  std::unique_ptr<MemoryManager> memory_manager;
  std::unique_ptr<EdgeUpdateManager<VertexDataType, EdgeDataType, EdgeUpdateType>> edge_update_manager;
  std::unique_ptr<VertexUpdateManager<VertexDataType, VertexUpdateType>> vertex_update_manager;
};


#endif