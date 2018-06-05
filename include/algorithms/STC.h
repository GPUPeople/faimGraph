//------------------------------------------------------------------------------
// STC.h
//
// faimGraph
//
//------------------------------------------------------------------------------
//

#pragma once

#include "Utility.h"

// Forward declaration
class MemoryManager;

//! Performs PageRank computation on aimGraph using the same implementation as cuSTINGER
template <typename VertexDataType, typename EdgeDataType>
std::unique_ptr<int32_t> workBalancedSTC(const std::unique_ptr<MemoryManager>& memory_manager,
                                          const int threads_per_block, 
                                          const int number_blocks, 
                                          const int shifter, 
                                          const int thread_blocks, 
                                          const int blockdim);

