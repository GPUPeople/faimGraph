//------------------------------------------------------------------------------
// SpMV.h
//
// Masterthesis aimGraph
//
// Authors: Martin Winter, 1130688
//------------------------------------------------------------------------------
//
#ifndef SPMV_AIMGRAPH_H
#define SPMV_AIMGRAPH_H

#include <memory>

#include "Utility.h"
#include "MemoryManager.h"
#include "aimGraph.h"



class SpMVManager
{
public:
  SpMVManager(vertex_t number_matrix_rows, vertex_t number_matrix_columns) :
    matrix_rows{ number_matrix_rows },
    matrix_columns{ number_matrix_columns },
    input_vector{ std::make_unique<matrix_t[]>(matrix_columns) },
    result_vector { std::make_unique<matrix_t[]>(matrix_columns) }
  {}
  void generateRandomVector();
  void receiveInputVector(std::unique_ptr<matrix_t[]> input) { input_vector = std::move(input); };
  void deviceSpMV(std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config);
  template <typename EdgeDataType> 
  void transposeaim2CSR2aim(std::unique_ptr<aimGraph<VertexData, VertexUpdate, EdgeDataType, EdgeDataUpdate>>& aimgraph, const std::shared_ptr<Config>& config);
  void transpose(std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config);

private:
  vertex_t matrix_rows;
  vertex_t matrix_columns;
  std::unique_ptr<matrix_t[]> input_vector;
  std::unique_ptr<matrix_t[]> result_vector;
  matrix_t* d_input_vector;
  matrix_t* d_result_vector;
};

#endif