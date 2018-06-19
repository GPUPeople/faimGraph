//------------------------------------------------------------------------------
// StaticTriangleCounting.h
//
// faimGraph
//
//------------------------------------------------------------------------------
//
//!  Triangle Counting implementations
/*!
  Holds 4 different STC implementations that can be toggled by passing the corresponding STCVariant
*/

#pragma once

#include "Utility.h"
#include "MemoryManager.h"


/*! \class STCVariant
    \brief Choose between 4 different STC implementations
*/
enum class STCVariant
{
  NAIVE,
  BALANCED,
  WARPSIZED,
  WARPSIZEDBALANCED
};

template <typename VertexDataType, typename EdgeDataType>
class STC
{
public:
  STC(std::unique_ptr<MemoryManager>& memory_manager, STCVariant stc_variant = STCVariant::NAIVE) : 
    triangles{std::make_unique<uint32_t[]>(memory_manager->next_free_vertex_index)}, 
    variant{ stc_variant }
  {
    TemporaryMemoryAccessHeap temp_memory_dispenser(memory_manager.get(), memory_manager->next_free_vertex_index, sizeof(VertexDataType));
    d_triangles = temp_memory_dispenser.getTemporaryMemory<uint32_t>(memory_manager->next_free_vertex_index);
    d_triangle_count = temp_memory_dispenser.getTemporaryMemory<uint32_t>(memory_manager->next_free_vertex_index);
    if (stc_variant == STCVariant::BALANCED || stc_variant == STCVariant::WARPSIZEDBALANCED)
    {
      d_accumulated_page_count = temp_memory_dispenser.getTemporaryMemory<vertex_t>(memory_manager->next_free_vertex_index + 1);
      d_page_count = temp_memory_dispenser.getTemporaryMemory<vertex_t>(memory_manager->next_free_vertex_index + 1);
    }
  }

  //! Performs triangle counting on aimGraph, 4 different implementations available
  uint32_t StaticTriangleCounting(const std::unique_ptr<MemoryManager>& memory_manager, bool global_TC_count = false);
  uint32_t host_StaticTriangleCounting(std::unique_ptr<GraphParser>& graph_parser);

// Data on device
  uint32_t* d_triangles{nullptr};
  uint32_t* d_triangle_count{ nullptr };
  vertex_t* d_accumulated_page_count{ nullptr };
  vertex_t* d_page_count{ nullptr };
  vertex_t* d_vertex_index{ nullptr };
  vertex_t* d_page_per_vertex_index{ nullptr };

// Data on host
  std::unique_ptr<uint32_t[]> triangles;
  STCVariant variant;
};
