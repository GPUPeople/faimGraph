//------------------------------------------------------------------------------
// VertexUpdate.cpp
//
// faimGraph
//
//------------------------------------------------------------------------------
//
#include <iostream>

#include "VertexUpdate.h"
#include "MemoryManager.h"
#include "GraphParser.h"

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename VertexUpdateType>
void VertexUpdateManager<VertexDataType, VertexUpdateType>::hostDuplicateCheckInBatch(VertexMapper<index_t, index_t>& mapper)
{
  for (int i = 0; i < vertex_insertion_updates->vertex_data.size() - 1; ++i)
  {
    if (vertex_insertion_updates->vertex_data.at(i).identifier == DELETIONMARKER)
      continue;

    for (int j = i + 1; j < vertex_insertion_updates->vertex_data.size(); ++j)
    {
      if (vertex_insertion_updates->vertex_data.at(j).identifier == DELETIONMARKER)
        continue;

      if (vertex_insertion_updates->vertex_data.at(i).identifier == vertex_insertion_updates->vertex_data.at(j).identifier)
      {
        if (mapper.h_device_mapping_update[j] != DELETIONMARKER)
        {
          std::cout << "For update: " << j << " which is " << vertex_insertion_updates->vertex_data.at(j).identifier << " the device DID NOT delete a duplicate in the batch" << std::endl;
          //exit(-1);
        }
        vertex_insertion_updates->vertex_data[j].identifier = DELETIONMARKER;
      }
    }
  }
  return;
}

template void VertexUpdateManager<VertexData, VertexUpdate>::hostDuplicateCheckInBatch(VertexMapper<index_t, index_t>& mapper);
template void VertexUpdateManager<VertexDataWeight, VertexUpdateWeight>::hostDuplicateCheckInBatch(VertexMapper<index_t, index_t>& mapper);
template void VertexUpdateManager<VertexDataSemantic, VertexUpdateSemantic>::hostDuplicateCheckInBatch(VertexMapper<index_t, index_t>& mapper);


//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename VertexUpdateType>
void VertexUpdateManager<VertexDataType, VertexUpdateType>::hostDuplicateCheckInGraph(VertexMapper<index_t, index_t>& mapper)
{
  const auto& map = mapper.h_map_indentifier_to_index;
  for (int i = 0; i < vertex_insertion_updates->vertex_data.size(); ++i)
  {
    if (map.find(vertex_insertion_updates->vertex_data.at(i).identifier) != map.end())
    {
      if (mapper.h_device_mapping_update[i] != DELETIONMARKER)
      {
        std::cout << "For update: " << i << " which is " << vertex_insertion_updates->vertex_data.at(i).identifier << " the device DID NOT delete a duplicate in the graph" << std::endl;
        //exit(-1);
      }
      vertex_insertion_updates->vertex_data[i].identifier = DELETIONMARKER;
    }
  }
  return;
}

template void VertexUpdateManager<VertexData, VertexUpdate>::hostDuplicateCheckInGraph(VertexMapper<index_t, index_t>& mapper);
template void VertexUpdateManager<VertexDataWeight, VertexUpdateWeight>::hostDuplicateCheckInGraph(VertexMapper<index_t, index_t>& mapper);
template void VertexUpdateManager<VertexDataSemantic, VertexUpdateSemantic>::hostDuplicateCheckInGraph(VertexMapper<index_t, index_t>& mapper);

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename VertexUpdateType>
void VertexUpdateManager<VertexDataType, VertexUpdateType>::hostVertexInsertion(std::unique_ptr<GraphParser>& parser,
                                                                                VertexMapper<index_t, index_t>& mapper)
{
  int current_number_vertices = parser->getOffset().size();
  auto& offset = parser->getOffset();
  int test = 0;
  
  
  // First, perform a duplicate check in batch
  hostDuplicateCheckInBatch(mapper);

  // Second, perform a duplicate check in graph
  hostDuplicateCheckInGraph(mapper);
  
  // Now let's look at what we should be doing
  for (int i = 0; i < vertex_insertion_updates->vertex_data.size(); ++i)
  {
    if (vertex_insertion_updates->vertex_data.at(i).identifier != DELETIONMARKER)
    {
      
      // We want to insert this vertex
      if(mapper.h_device_mapping_update.at(i) >= static_cast<vertex_t>(current_number_vertices - 1))
      {
        // Add it to the back, it's a new position
        offset.push_back(offset.back());
        ++test;
        
      }
      else
      {
        // Simply check if there was a deleted vertex there
        auto begin_iter = offset.begin() + mapper.h_device_mapping_update.at(i);
        if (*begin_iter != *(begin_iter + 1))
        {
          std::cout << "For update: " << i << " which is " << vertex_insertion_updates->vertex_data.at(i).identifier << " there is no free vertex there!" << std::endl;
        }
      }
    }
  }

  //std::cout << "PushbackCounter: " << test << std::endl;
  return;
}

template void VertexUpdateManager<VertexData, VertexUpdate>::hostVertexInsertion(std::unique_ptr<GraphParser>& parser, VertexMapper<index_t, index_t>& mapper);
template void VertexUpdateManager<VertexDataWeight, VertexUpdateWeight>::hostVertexInsertion(std::unique_ptr<GraphParser>& parser, VertexMapper<index_t, index_t>& mapper);
template void VertexUpdateManager<VertexDataSemantic, VertexUpdateSemantic>::hostVertexInsertion(std::unique_ptr<GraphParser>& parser, VertexMapper<index_t, index_t>& mapper);

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename VertexUpdateType>
void VertexUpdateManager<VertexDataType, VertexUpdateType>::hostVertexDeletion(std::unique_ptr<GraphParser>& parser,
                                                                               VertexMapper<index_t, index_t>& mapper)
{
  int current_number_vertices = parser->getOffset().size();
  auto& offset = parser->getOffset();
  auto& adjacency = parser->getAdjacency();
  const auto& map = mapper.h_map_indentifier_to_index;

  

  // std::cout << "VertexDeletion" << std::endl;

  // for(int i = 0; i < 10; ++i)
  // {
  //   std::cout << "AdjacencySize: " << offset.at(i + 1) - offset.at(i) << std::endl;
  // }
  // std::cout << "############################" << std::endl;

  for (int i = 0; i < vertex_deletion_updates->vertex_data.size(); ++i)
  {
    const auto& update = vertex_deletion_updates->vertex_data.at(i);
    if (update.identifier >= static_cast<index_t>(current_number_vertices - 1))
    {
      // Check that we got back a DELETIONMARKER in the mapping
      if (mapper.h_device_mapping_update.at(i) != DELETIONMARKER)
      {
        std::cout << "For update: " << i << " which is " << vertex_deletion_updates->vertex_data.at(i).identifier << " the update was out of range but mapping did not detect!" << std::endl;
      }
    }
    else
    {
      // Delete adjacency

      auto begin_iter = adjacency.begin() + offset.at(update.identifier);
      auto end_iter = adjacency.begin() + offset.at(update.identifier + 1);
      auto adjacency_size = offset.at(update.identifier + 1) - offset.at(update.identifier);

      // Update offset list
      if (adjacency_size > 0)
      {
        for (auto i = update.identifier + 1; i < offset.size(); ++i)
        {
          offset[i] -= adjacency_size;
        }
        adjacency.erase(begin_iter, begin_iter + adjacency_size);
      }            
    }
  }

  // Delete mentions
  int subtract_offset = 0;
  bool mention_deleted = false;
  for(int i = 0; i < offset.size() - 1; ++i)
  {
    // Delete all the mentions
    auto begin_iter = adjacency.begin() + offset.at(i);
    auto size_adjacency = offset.at(i + 1) - offset.at(i) - subtract_offset;
    while(size_adjacency > 0)
    {
      for(const auto& update : vertex_deletion_updates->vertex_data)
      {
        if(update.identifier == *begin_iter)
        {
          begin_iter = adjacency.erase(begin_iter);
          ++subtract_offset;
          mention_deleted = true;
          break;
        }
      }
      if (!mention_deleted)
        ++begin_iter;
      --size_adjacency;
      mention_deleted = false;
    }
    offset[i + 1] -= subtract_offset;
  }

  return;
}

template void VertexUpdateManager<VertexData, VertexUpdate>::hostVertexDeletion(std::unique_ptr<GraphParser>& parser, VertexMapper<index_t, index_t>& mapper);
template void VertexUpdateManager<VertexDataWeight, VertexUpdateWeight>::hostVertexDeletion(std::unique_ptr<GraphParser>& parser, VertexMapper<index_t, index_t>& mapper);
template void VertexUpdateManager<VertexDataSemantic, VertexUpdateSemantic>::hostVertexDeletion(std::unique_ptr<GraphParser>& parser, VertexMapper<index_t, index_t>& mapper);

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename VertexUpdateType>
void VertexUpdateManager<VertexDataType, VertexUpdateType>::generateVertexInsertUpdates(vertex_t batch_size, unsigned int seed)
{
  std::unique_ptr<VertexUpdateBatch<VertexUpdateType>> vertex_update(std::make_unique<VertexUpdateBatch<VertexUpdateType>>());

  // Generate random edge updates
  srand(seed + 1);

  for (vertex_t i = 0; i < batch_size; ++i)
  {
    VertexUpdateType update;
    update.identifier = rand();
    vertex_update->vertex_data.push_back(update);
  }

  receiveVertexInsertionUpdates(std::move(vertex_update));
  return;
}

template void VertexUpdateManager<VertexData, VertexUpdate>::generateVertexInsertUpdates(vertex_t batch_size, unsigned int seed);
template void VertexUpdateManager<VertexDataWeight, VertexUpdateWeight>::generateVertexInsertUpdates(vertex_t batch_size, unsigned int seed);
template void VertexUpdateManager<VertexDataSemantic, VertexUpdateSemantic>::generateVertexInsertUpdates(vertex_t batch_size, unsigned int seed);

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename VertexUpdateType>
void VertexUpdateManager<VertexDataType, VertexUpdateType>::generateVertexDeleteUpdates(VertexMapper<index_t, index_t>& mapper,
                                                                                        vertex_t batch_size,
                                                                                        unsigned int seed,
                                                                                        unsigned int highest_index)
{
  std::unique_ptr<VertexUpdateBatch<VertexUpdate>> vertex_update(std::make_unique<VertexUpdateBatch<VertexUpdate>>());

  srand(seed + 1);

  for (vertex_t i = 0; i < batch_size; ++i)
  {
    VertexUpdate update;

    int index = rand() % highest_index;

    update.identifier = index;

    vertex_update->vertex_data.push_back(update);
  }

  receiveVertexDeletionUpdates(std::move(vertex_update));
  return;
}

template void VertexUpdateManager<VertexData, VertexUpdate>::generateVertexDeleteUpdates(VertexMapper<index_t, index_t>& mapper, vertex_t batch_size, unsigned int seed, unsigned int highest_index);
template void VertexUpdateManager<VertexDataWeight, VertexUpdateWeight>::generateVertexDeleteUpdates(VertexMapper<index_t, index_t>& mapper, vertex_t batch_size, unsigned int seed, unsigned int highest_index);
template void VertexUpdateManager<VertexDataSemantic, VertexUpdateSemantic>::generateVertexDeleteUpdates(VertexMapper<index_t, index_t>& mapper, vertex_t batch_size, unsigned int seed, unsigned int highest_index);


//------------------------------------------------------------------------------
//
template <typename HostDataType, typename DeviceDataType>
void VertexMapper<HostDataType, DeviceDataType>::initialMapperSetup(const std::unique_ptr<MemoryManager>& memory_manager, int batch_size)
{
  // Setup initial memory
  for (vertex_t i = 0; i < memory_manager->number_vertices; ++i)
  {
    h_device_mapping.push_back(i);
    h_map_indentifier_to_index.insert(std::pair<index_t, index_t>(i, i));
  }

  h_device_mapping_update.resize(batch_size);
  return;
}

template void VertexMapper<index_t, index_t>::initialMapperSetup(const std::unique_ptr<MemoryManager>& memory_manager, int batch_size);

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename VertexUpdateType>
void VertexUpdateManager<VertexDataType, VertexUpdateType>::integrateInsertionChanges(VertexMapper<index_t, index_t>& mapper)
{
  for (int i = 0; i < vertex_insertion_updates->vertex_data.size(); ++i)
  {
    if (mapper.h_device_mapping_update.at(i) != DELETIONMARKER)
    {
      mapper.insertTuple(vertex_insertion_updates->vertex_data.at(i).identifier, mapper.h_device_mapping_update.at(i));
    }
  }
}

template void VertexUpdateManager<VertexData, VertexUpdate>::integrateInsertionChanges(VertexMapper<index_t, index_t>& mapper);
template void VertexUpdateManager<VertexDataWeight, VertexUpdateWeight>::integrateInsertionChanges(VertexMapper<index_t, index_t>& mapper);
template void VertexUpdateManager<VertexDataSemantic, VertexUpdateSemantic>::integrateInsertionChanges(VertexMapper<index_t, index_t>& mapper);

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename VertexUpdateType>
void VertexUpdateManager<VertexDataType, VertexUpdateType>::integrateDeletionChanges(VertexMapper<index_t, index_t>& mapper)
{
  for (int i = 0; i < vertex_deletion_updates->vertex_data.size(); ++i)
  {
    if (mapper.h_device_mapping_update.at(i) != DELETIONMARKER)
    {
      mapper.deleteTuple(mapper.h_device_mapping_update.at(i));
    }
  }
}

template void VertexUpdateManager<VertexData, VertexUpdate>::integrateDeletionChanges(VertexMapper<index_t, index_t>& mapper);
template void VertexUpdateManager<VertexDataWeight, VertexUpdateWeight>::integrateDeletionChanges(VertexMapper<index_t, index_t>& mapper);
template void VertexUpdateManager<VertexDataSemantic, VertexUpdateSemantic>::integrateDeletionChanges(VertexMapper<index_t, index_t>& mapper);

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename VertexUpdateType>
void VertexUpdateManager<VertexDataType, VertexUpdateType>::setupMemory(const std::unique_ptr<MemoryManager>& memory_manager, VertexMapper<index_t, index_t>& mapper, VertexUpdateVersion version)
{
  // Place initial pointers in device memory
  TemporaryMemoryAccessStack temp_memory_dispenser(memory_manager.get());
  int vertex_data_size;
  if (version == VertexUpdateVersion::INSERTION)
  {
    vertex_data_size = vertex_insertion_updates->vertex_data.size();
  }
  else
  {
    vertex_data_size = vertex_deletion_updates->vertex_data.size();
  }
  
  mapper.d_device_mapping = temp_memory_dispenser.getTemporaryMemory<index_t>(memory_manager->next_free_vertex_index + vertex_data_size);
  mapper.d_device_mapping_update = temp_memory_dispenser.getTemporaryMemory<index_t>(vertex_data_size);

  if (version == VertexUpdateVersion::INSERTION)
  {
    vertex_insertion_updates->d_vertex_data = temp_memory_dispenser.getTemporaryMemory<VertexUpdateType>(vertex_data_size);
  }
  else
  {
    vertex_deletion_updates->d_vertex_data = temp_memory_dispenser.getTemporaryMemory<VertexUpdate>(vertex_data_size);
  }  
}

template void VertexUpdateManager<VertexData, VertexUpdate>::setupMemory(const std::unique_ptr<MemoryManager>& memory_manager, VertexMapper<index_t, index_t>& mapper, VertexUpdateVersion version);
template void VertexUpdateManager<VertexDataWeight, VertexUpdateWeight>::setupMemory(const std::unique_ptr<MemoryManager>& memory_manager, VertexMapper<index_t, index_t>& mapper, VertexUpdateVersion version);
template void VertexUpdateManager<VertexDataSemantic, VertexUpdateSemantic>::setupMemory(const std::unique_ptr<MemoryManager>& memory_manager, VertexMapper<index_t, index_t>& mapper, VertexUpdateVersion version);