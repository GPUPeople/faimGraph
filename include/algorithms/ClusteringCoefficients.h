//------------------------------------------------------------------------------
// ClusteringCoefficients.h
//
// faimGraph
//
//------------------------------------------------------------------------------
//

#pragma once

#include  "STC.h"

template <typename VertexDataType, typename EdgeDataType>
class ClusteringCoefficients
{
public:
  ClusteringCoefficients(std::unique_ptr<MemoryManager>& memory_manager, STCVariant stc_variant = STCVariant::NAIVE):
    stc{ std::make_unique<STC<VertexDataType, EdgeDataType>>(memory_manager, stc_variant) },
    clustering_coefficients { std::make_unique<float[]>(memory_manager->next_free_vertex_index) }
  {
    // Start right after triangle count from stc
    TemporaryMemoryAccessHeap temp_memory_dispenser(memory_manager.get(), reinterpret_cast<memory_t*>(stc->d_triangles + memory_manager->next_free_vertex_index));
    d_clustering_coefficients = temp_memory_dispenser.getTemporaryMemory<float>(memory_manager->next_free_vertex_index);
    d_clustering_coefficients_count = temp_memory_dispenser.getTemporaryMemory<float>(memory_manager->next_free_vertex_index);
  }

  //! Performs a clustering coefficient computation on aimGraph, 4 different STC implementations available
  float computeClusteringCoefficients(std::unique_ptr<MemoryManager>& memory_manager, bool global_CC_count = false);

// Members on device
  float* d_clustering_coefficients{ nullptr };
  float* d_clustering_coefficients_count{ nullptr };

// Members on host
  std::unique_ptr<STC<VertexDataType, EdgeDataType>> stc;
  std::unique_ptr<float[]> clustering_coefficients;
};
