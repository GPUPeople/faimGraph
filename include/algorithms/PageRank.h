//------------------------------------------------------------------------------
// PageRank.h
//
// faimGraph
//
//------------------------------------------------------------------------------
//

#pragma once

#include "Utility.h"
#include "MemoryManager.h"

enum class PageRankVariant
{
  NAIVE,
  BALANCED
};

template <typename VertexDataType, typename EdgeDataType>
class PageRank
{
public:
  PageRank(std::unique_ptr<MemoryManager>& memory_manager, PageRankVariant pr_variant):
    variant{ pr_variant }
  {
    TemporaryMemoryAccessHeap temp_memory_dispenser(memory_manager.get(), memory_manager->next_free_vertex_index, sizeof(VertexDataType));
    d_page_rank = temp_memory_dispenser.getTemporaryMemory<float>(memory_manager->next_free_vertex_index);
    d_next_page_rank = temp_memory_dispenser.getTemporaryMemory<float>(memory_manager->next_free_vertex_index);
    d_absolute_difference = temp_memory_dispenser.getTemporaryMemory<float>(memory_manager->next_free_vertex_index);
    d_diff_sum = temp_memory_dispenser.getTemporaryMemory<float>(memory_manager->next_free_vertex_index);
    if (pr_variant == PageRankVariant::BALANCED)
    {
      d_accumulated_page_count = temp_memory_dispenser.getTemporaryMemory<vertex_t>(memory_manager->next_free_vertex_index + 1);
      d_page_count = temp_memory_dispenser.getTemporaryMemory<vertex_t>(memory_manager->next_free_vertex_index + 1);
    }
  }

  void initializePageRankVector(float initial_value, uint32_t number_values)
  {
    HANDLE_ERROR(cudaMemset(d_page_rank,
      initial_value,
      sizeof(float) * number_values));
    HANDLE_ERROR(cudaMemset(d_next_page_rank,
                        0.0f, 
                        sizeof(float) * number_values));
  }

  //! Performs PageRank computation on aimGraph, naive implementation
  float algPageRankNaive(const std::unique_ptr<MemoryManager>& memory_manager);
  //! Performs PageRank computation on aimGraph, page-balanced implementation
  float algPageRankBalanced(const std::unique_ptr<MemoryManager>& memory_manager);

// Member on device
  float* d_page_rank { nullptr };
  float* d_next_page_rank{ nullptr };
  float* d_absolute_difference{ nullptr };
  float* d_diff_sum{ nullptr };
  vertex_t* d_accumulated_page_count{ nullptr };
  vertex_t* d_page_count{ nullptr };
  vertex_t* d_vertex_index{ nullptr };
  vertex_t* d_page_per_vertex_index{ nullptr };

//Member on host
  float dampening_factor{ 0.85f };
  PageRankVariant variant{ PageRankVariant::NAIVE };
};
