//------------------------------------------------------------------------------
// VertexMapper.h
//
// faimGraph
//
//------------------------------------------------------------------------------
//

#pragma once

#include <iostream>
#include <vector>
#include <map>

#include "Utility.h"

class MemoryManager;

template <typename HostDataType, typename DeviceDataType>
class VertexMapper
{
public:
  // Convenience functionality
  inline DeviceDataType getIndexAt(index_t key) { return h_map_indentifier_to_index.at(key); }
  inline size_t getSize() { return h_map_indentifier_to_index.size(); }
  inline void insertTuple(HostDataType identifier, DeviceDataType index) 
  { 
    auto test = h_map_indentifier_to_index.insert(std::pair<HostDataType, DeviceDataType>(identifier, index));
    if(test.second == false)
    {
      std::cout << "Insert duplicate " << identifier << " which maps to " << index << std::endl;
    }
  }
  inline void deleteTuple(HostDataType identifier) 
  {
    h_map_indentifier_to_index.erase(identifier); 
  }

  // Setup
  void initialMapperSetup(const std::unique_ptr<MemoryManager>& memory_manager, int batch_size);

//private:
  //------------------------------------------------------------------------------
  // Host Data
  //------------------------------------------------------------------------------
  std::map<HostDataType, DeviceDataType> h_map_indentifier_to_index;
  std::vector<HostDataType> h_device_mapping; /*!< Holds the host-device identifiers */
  std::vector<DeviceDataType> h_device_mapping_update; /*!< Holds the host-device identifiers */

  //------------------------------------------------------------------------------
  // Device Data
  //------------------------------------------------------------------------------
  HostDataType* d_device_mapping{ nullptr }; /*!< Holds the host-device identifiers */
  DeviceDataType* d_device_mapping_update{ nullptr }; /*!< Holds the new device identifiers set in the update */

  //------------------------------------------------------------------------------
  // Global Data
  //------------------------------------------------------------------------------
  index_t mapping_size{0}; /*!< Holds the size of the mapping arrays */
};
