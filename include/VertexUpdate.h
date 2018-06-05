//------------------------------------------------------------------------------
// VertexUpdate.h
//
// faimGraph
//
//------------------------------------------------------------------------------
//

#pragma once

#include "VertexMapper.h"

// Forward declaration
class GraphParser;
class MemoryManager;
class VerifyGraphCSR;
class Config;

/*! \class VertexUpdateBatch
\brief Templatised class to hold a batch of vertex updates
*/
template <typename VertexUpdateType>
class VertexUpdateBatch
{
public:
  // Host data
  std::vector<VertexUpdateType> vertex_data;

  // Device data
  VertexUpdateType* d_vertex_data;  
};

enum class VertexUpdateVersion
{
  INSERTION,
  DELETION
};

/*! \class VertexUpdateManager
\brief Templatised class used for vertex updates
*/
template <typename VertexDataType, typename VertexUpdateType>
class VertexUpdateManager
{
public:
  // Updates on device
  template <typename EdgeDataType> void deviceVertexInsertion(std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config, VertexMapper<index_t, index_t>& mapper, bool duplicate_checking);
  template <typename EdgeDataType> void deviceVertexDeletion(std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config, VertexMapper<index_t, index_t>& mapper);

  // Updates on host
  void hostVertexInsertion(std::unique_ptr<GraphParser>& parser, VertexMapper<index_t, index_t>& mapper);
  void hostVertexDeletion(std::unique_ptr<GraphParser>& parser, VertexMapper<index_t, index_t>& mapper);

  // Receive and generate updates
  void receiveVertexInsertionUpdates(std::unique_ptr<VertexUpdateBatch<VertexUpdateType>> update) { vertex_insertion_updates = std::move(update); }
  void receiveVertexDeletionUpdates(std::unique_ptr<VertexUpdateBatch<VertexUpdate>> update) { vertex_deletion_updates = std::move(update); }
  void generateVertexInsertUpdates(vertex_t batch_size, unsigned int seed);
  void generateVertexDeleteUpdates(VertexMapper<index_t, index_t>& mapper, vertex_t batch_size, unsigned int seed, unsigned int highest_index);

  // Duplicate Checking on device
  void duplicateInBatchChecking(const std::shared_ptr<Config>& config);
  void duplicateInGraphChecking(const std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config);
  // Duplicate Checking on host
  void hostDuplicateCheckInBatch(VertexMapper<index_t, index_t>& mapper);
  void hostDuplicateCheckInGraph(VertexMapper<index_t, index_t>& mapper);

  // Setup memory and integrate changes on device in host mapper
  void setupMemory(const std::unique_ptr<MemoryManager>& memory_manager, VertexMapper<index_t, index_t>& mapper, VertexUpdateVersion version);
  void integrateInsertionChanges(VertexMapper<index_t, index_t>& mapper);
  void integrateDeletionChanges(VertexMapper<index_t, index_t>& mapper);


  float time_dup_in_batch {0.0f};
  float time_dup_in_graph {0.0f};
  float time_insertion {0.0f};
  float time_vertex_mentions {0.0f};
  float time_compaction {0.0f};
  float time_deletion {0.0f};

private:
  //
  std::unique_ptr<VertexUpdateBatch<VertexUpdateType>> vertex_insertion_updates;
  std::unique_ptr<VertexUpdateBatch<VertexUpdate>> vertex_deletion_updates;  
};
