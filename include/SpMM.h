//------------------------------------------------------------------------------
// SpMM.h
//
// faimGraph
//
//------------------------------------------------------------------------------
//

#pragma once

#include <memory>

#include "Utility.h"
#include "MemoryManager.h"
#include "faimGraph.h"
#include "GraphParser.h"

class SpMMManager
{
public:
  SpMMManager(vertex_t in_A_rows, vertex_t in_A_columns, vertex_t in_B_rows, vertex_t in_B_columns) :
    input_A_rows{ in_A_rows },
    input_A_columns{ in_A_columns },
    input_B_rows{ in_B_rows },
    input_B_columns{ in_B_columns },
    output_rows{ in_A_rows },
    output_columns{ in_B_columns },
    input_B_offset{ in_A_rows },
    output_offset{in_A_rows + in_B_rows},
    next_free_vertex{ in_A_rows + in_B_rows + in_A_rows }
  {}

  template <typename EdgeDataType>
  void initializeFaimGraphMatrix(std::unique_ptr<faimGraph<VertexData, VertexUpdate, EdgeDataType, EdgeDataUpdate>>& faimgraph, std::unique_ptr<GraphParser>& graph_parser, const std::shared_ptr<Config>& config);

  template <typename EdgeDataType>
  void spmmMultiplication(std::unique_ptr<faimGraph<VertexData, VertexUpdate, EdgeDataType, EdgeDataUpdate>>& faimGraph, const std::shared_ptr<Config>& config);

  template <typename EdgeDataType>
  void spmmMultiplication(std::unique_ptr<faimGraph<VertexData, VertexUpdate, EdgeDataType, EdgeDataUpdate>>& input_matrix_A, std::unique_ptr<faimGraph<VertexData, VertexUpdate, EdgeDataType, EdgeDataUpdate>>& input_matrix_B, std::unique_ptr<faimGraph<VertexData, VertexUpdate, EdgeDataType, EdgeDataUpdate>>& output_matrix, const std::shared_ptr<Config>& config);

  template <typename EdgeDataType>
  void resetResultMatrix(std::unique_ptr<faimGraph<VertexData, VertexUpdate, EdgeDataType, EdgeDataUpdate>>& faimGraph, const std::shared_ptr<Config>& config, bool tripleAimGraph = false);


  // Matrix dimensions
  vertex_t input_A_rows;
  vertex_t input_B_rows;
  vertex_t output_rows;
  vertex_t input_A_columns;
  vertex_t input_B_columns;
  vertex_t output_columns;

  // Graph Vertex Offsets
  vertex_t input_A_offset{ 0 };
  vertex_t input_B_offset;
  vertex_t output_offset;
  vertex_t next_free_vertex;
  vertex_t next_free_page_after_init;
};

