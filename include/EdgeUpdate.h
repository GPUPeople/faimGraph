//------------------------------------------------------------------------------
// EdgeUpdate.h
//
// faimGraph
//
//------------------------------------------------------------------------------
//

#pragma once

#include "Utility.h"
#include "MemoryManager.h"

#define QUEUING
#define CLEAN_PAGE

enum class QueryKernelConfig
{
  STANDARD,
  WARPSIZED,
  VERTEXCENTRIC
};

// Forward declaration
class GraphParser;
class Config;
template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename EdgeUpdateType> class faimGraph;

template <typename EdgeDataType>
class EdgeBlock
{
  EdgeDataType edgeblock[15];
};

/*! \class EdgeUpdateBatch
\brief Templatised class to hold a batch of edge updates
*/
template <typename UpdateDataType>
class EdgeUpdateBatch
{
  public:
    // Host side
    std::vector<UpdateDataType> edge_update;
    UpdateDataType* raw_edge_update;

    // Device side
    UpdateDataType* d_edge_update;
};


/*! \class EdgeUpdatePreProcessing
\brief Templatised class used for preprocessing of edge updates
*/
template <typename UpdateDataType>
class EdgeUpdatePreProcessing
{
public:
	
	EdgeUpdatePreProcessing(vertex_t number_vertices, vertex_t batch_size, std::unique_ptr<MemoryManager>& memory_manager, size_t sizeofVertexData)
	{
		TemporaryMemoryAccessHeap temp_memory_dispenser(memory_manager.get(), number_vertices, sizeofVertexData);
		temp_memory_dispenser.getTemporaryMemory<UpdateDataType>(batch_size); // Move after update data

		// Now let's set the member pointers
		d_edge_src_counter = temp_memory_dispenser.getTemporaryMemory<index_t>(number_vertices + 1);
		d_update_src_offsets = temp_memory_dispenser.getTemporaryMemory<index_t>(number_vertices + 1);
	}
  
  index_t* d_edge_src_counter;
  index_t* d_update_src_offsets;
};

enum class EdgeUpdateVersion
{
  GENERAL,
  INSERTION,
  DELETION
};

enum class EdgeUpdateMechanism
{
  SEQUENTIAL,
  CONCURRENT
};

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType, typename EdgeUpdateType>
class EdgeUpdateManager
{
public:
  EdgeUpdateManager() : update_type{ EdgeUpdateMechanism::SEQUENTIAL } {}

  // Sequential Update Functionality on device
  void deviceEdgeInsertion(std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config);
  void deviceEdgeDeletion(std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config);

  // Concurrent Update Functionality on device
  void deviceEdgeUpdateConcurrentStream(cudaStream_t& insertion_stream, cudaStream_t& deletion_stream, std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config);
  void deviceEdgeUpdateConcurrent(std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config);

  // Sequential Update Functionality on host
  void hostEdgeInsertion(const std::unique_ptr<GraphParser>& parser);
  void hostEdgeDeletion(const std::unique_ptr<GraphParser>& parser);

  // Generate Edge Update Data
  std::unique_ptr<EdgeUpdateBatch<EdgeUpdateType>> generateEdgeUpdates(vertex_t number_vertices, vertex_t batch_size, unsigned int seed, unsigned int range = 0, unsigned int offset = 0);
  std::unique_ptr<EdgeUpdateBatch<EdgeUpdateType>> generateEdgeUpdates(const std::unique_ptr<MemoryManager>& memory_manager, vertex_t batch_size, unsigned int seed, unsigned int range = 0, unsigned int offset = 0);
  template <typename VertexUpdateType>
  std::unique_ptr<EdgeUpdateBatch<EdgeUpdateType>> generateEdgeUpdatesConcurrent(std::unique_ptr<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, EdgeUpdateType>>& faimGraph, const std::unique_ptr<MemoryManager>& memory_manager, vertex_t batch_size, unsigned int seed, unsigned int range = 0, unsigned int offset = 0);

  void receiveEdgeUpdates(std::unique_ptr<EdgeUpdateBatch<EdgeUpdateType>> updates, EdgeUpdateVersion type);
  void hostCudaAllocConcurrentUpdates();
  void hostCudaFreeConcurrentUpdates();

  // Edge Update Processing
  std::unique_ptr<EdgeUpdatePreProcessing<EdgeUpdateType>> edgeUpdatePreprocessing(std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config);
  void edgeUpdateDuplicateChecking(std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config, const std::unique_ptr<EdgeUpdatePreProcessing<EdgeUpdateType>>& preprocessed);

  // Write/Read Update to/from file
  void writeEdgeUpdatesToFile(vertex_t number_vertices, vertex_t batch_size, const std::string& filename);
  void writeEdgeUpdatesToFile(const std::unique_ptr<EdgeUpdateBatch<EdgeUpdateType>>& edges, vertex_t batch_size, const std::string& filename);
  std::unique_ptr<EdgeUpdateBatch<EdgeUpdateType>> readEdgeUpdatesFromFile(const std::string& filename);
  void writeGraphsToFile(const std::unique_ptr<aimGraphCSR>& verify_graph, const std::unique_ptr<GraphParser>& graph_parser, const std::string& filename);

  // General
  void setUpdateType(EdgeUpdateMechanism type) { update_type = type; }
  EdgeUpdateMechanism getUpdateType() { return update_type; }

private:
  // Interface for calling update kernel explicitely
  void w_edgeInsertion(cudaStream_t& stream, const std::unique_ptr<EdgeUpdateBatch<EdgeUpdateType>>& updates_insertion, std::unique_ptr<MemoryManager>& memory_manager, int batch_size, int block_size, int grid_size);
  void w_edgeDeletion(cudaStream_t& stream, const std::unique_ptr<EdgeUpdateBatch<EdgeUpdateType>>& updates_deletion, std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config, int batch_size, int block_size, int grid_size);

  std::unique_ptr<EdgeUpdateBatch<EdgeUpdateType>> updates;
  std::unique_ptr<EdgeUpdateBatch<EdgeUpdateType>> updates_insertion;
  std::unique_ptr<EdgeUpdateBatch<EdgeUpdateType>> updates_deletion;
  EdgeUpdateMechanism update_type;
};

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType>
class EdgeQueryManager
{
public:
  void deviceQuery(std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config);
  void generateQueries(vertex_t number_vertices, vertex_t batch_size, unsigned int seed, unsigned int range = 0, unsigned int offset = 0);
  void receiveQueries(std::unique_ptr<EdgeUpdateBatch<EdgeDataUpdate>> adjacency_queries)
  {
    queries = std::move(adjacency_queries);
  }
  std::unique_ptr<EdgeUpdatePreProcessing<EdgeDataUpdate>> edgeQueryPreprocessing(std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& configuration);

private:
  std::unique_ptr<EdgeUpdateBatch<EdgeDataUpdate>> queries;
  std::unique_ptr<bool[]> query_results;
  bool* d_query_results;
  QueryKernelConfig config{ QueryKernelConfig::VERTEXCENTRIC };
};
