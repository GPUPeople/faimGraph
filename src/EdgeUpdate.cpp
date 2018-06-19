//------------------------------------------------------------------------------
// EdgeUpdate.cpp
//
// faimGraph
//
//------------------------------------------------------------------------------
//
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <time.h>

#include "EdgeUpdate.h"
#include "GraphParser.h"
#include "MemoryManager.h"

template class EdgeUpdateBatch<EdgeDataUpdate>;
template class EdgeUpdateBatch<EdgeDataWeightUpdate>;
template class EdgeUpdateBatch<EdgeDataSemanticUpdate>;

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType, typename EdgeUpdateType>
std::unique_ptr<EdgeUpdateBatch<EdgeUpdateType>> EdgeUpdateManager<VertexDataType, EdgeDataType, EdgeUpdateType>::generateEdgeUpdates(vertex_t number_vertices, vertex_t batch_size, unsigned int seed, unsigned int range, unsigned int offset)
{
	std::unique_ptr<EdgeUpdateBatch<EdgeUpdateType>> edge_update(std::make_unique<EdgeUpdateBatch<EdgeUpdateType>>());
	// Generate random edge updates
	srand(seed + 1);
	for (vertex_t i = 0; i < batch_size/2; ++i)
	{
    EdgeUpdateType edge_update_data;
    vertex_t intermediate = rand() % ((range && (range < number_vertices)) ? range : number_vertices);
    vertex_t source;
    if(offset + intermediate < number_vertices)
      source = offset + intermediate;
    else
      source = intermediate;
		edge_update_data.source = source;
		edge_update_data.update.destination = rand() % number_vertices;
		edge_update->edge_update.push_back(edge_update_data);
	}

  for (vertex_t i = batch_size/2; i < batch_size; ++i)
	{
    EdgeUpdateType edge_update_data;
    vertex_t intermediate = rand() % (number_vertices);
    vertex_t source;
    if(offset + intermediate < number_vertices)
      source = offset + intermediate;
    else
      source = intermediate;
		edge_update_data.source = source;
		edge_update_data.update.destination = rand() % number_vertices;
		edge_update->edge_update.push_back(edge_update_data);
	}

  /*for (auto const& update : edge_update->edge_update)
  {
	  if (update.source == 218)
		  printf("Generate Update %u | %u\n", update.source, update.update.destination);
  }*/

	// Write data to file to verify
	static int counter = 0;
#ifdef DEBUG_VERBOSE_OUPUT
	std::string filename = std::string("../tests/Verification/VerifyInsert");
	filename += std::to_string(counter) + std::string(".txt");
	std::ofstream file(filename);
	if (file.is_open())
	{
		for (int i = 0; i < batch_size; ++i)
		{
			file << edge_update->edge_update.at(i).source << " ";
			file << edge_update->edge_update.at(i).update.destination << "\n";
		}
	}
#endif
	++counter;

	return std::move(edge_update);
}

template std::unique_ptr<EdgeUpdateBatch<EdgeDataUpdate>> EdgeUpdateManager<VertexData, EdgeData, EdgeDataUpdate>::generateEdgeUpdates(vertex_t number_vertices, vertex_t batch_size, unsigned int seed, unsigned int range, unsigned int offset);
template std::unique_ptr<EdgeUpdateBatch<EdgeDataWeightUpdate>> EdgeUpdateManager<VertexDataWeight, EdgeDataWeight, EdgeDataWeightUpdate>::generateEdgeUpdates(vertex_t number_vertices, vertex_t batch_size, unsigned int seed, unsigned int range, unsigned int offset);
template std::unique_ptr<EdgeUpdateBatch<EdgeDataSemanticUpdate>> EdgeUpdateManager<VertexDataSemantic, EdgeDataSemantic, EdgeDataSemanticUpdate>::generateEdgeUpdates(vertex_t number_vertices, vertex_t batch_size, unsigned int seed, unsigned int range, unsigned int offset);
template std::unique_ptr<EdgeUpdateBatch<EdgeDataUpdate>> EdgeUpdateManager<VertexData, EdgeDataSOA, EdgeDataUpdate>::generateEdgeUpdates(vertex_t number_vertices, vertex_t batch_size, unsigned int seed, unsigned int range, unsigned int offset);
template std::unique_ptr<EdgeUpdateBatch<EdgeDataWeightUpdate>> EdgeUpdateManager<VertexDataWeight, EdgeDataWeightSOA, EdgeDataWeightUpdate>::generateEdgeUpdates(vertex_t number_vertices, vertex_t batch_size, unsigned int seed, unsigned int range, unsigned int offset);
template std::unique_ptr<EdgeUpdateBatch<EdgeDataSemanticUpdate>> EdgeUpdateManager<VertexDataSemantic, EdgeDataSemanticSOA, EdgeDataSemanticUpdate>::generateEdgeUpdates(vertex_t number_vertices, vertex_t batch_size, unsigned int seed, unsigned int range, unsigned int offset);

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType, typename EdgeUpdateType>
void EdgeUpdateManager<VertexDataType, EdgeDataType, EdgeUpdateType>::receiveEdgeUpdates(std::unique_ptr<EdgeUpdateBatch<EdgeUpdateType>> edge_updates, EdgeUpdateVersion type)
{
  if (type == EdgeUpdateVersion::GENERAL)
  {
    updates = std::move(edge_updates);
  }
  else if (type == EdgeUpdateVersion::INSERTION)
  {
    updates_insertion = std::move(edge_updates);
  }
  else if (type == EdgeUpdateVersion::DELETION)
  {
    updates_deletion = std::move(edge_updates);
  }
}

template void EdgeUpdateManager<VertexData, EdgeData, EdgeDataUpdate>::receiveEdgeUpdates(std::unique_ptr<EdgeUpdateBatch<EdgeDataUpdate>> edge_updates, EdgeUpdateVersion type);
template void EdgeUpdateManager<VertexDataWeight, EdgeDataWeight, EdgeDataWeightUpdate>::receiveEdgeUpdates(std::unique_ptr<EdgeUpdateBatch<EdgeDataWeightUpdate>> edge_updates, EdgeUpdateVersion type);
template void EdgeUpdateManager<VertexDataSemantic, EdgeDataSemantic, EdgeDataSemanticUpdate>::receiveEdgeUpdates(std::unique_ptr<EdgeUpdateBatch<EdgeDataSemanticUpdate>> edge_updates, EdgeUpdateVersion type);
template void EdgeUpdateManager<VertexData, EdgeDataSOA, EdgeDataUpdate>::receiveEdgeUpdates(std::unique_ptr<EdgeUpdateBatch<EdgeDataUpdate>> edge_updates, EdgeUpdateVersion type);
template void EdgeUpdateManager<VertexDataWeight, EdgeDataWeightSOA, EdgeDataWeightUpdate>::receiveEdgeUpdates(std::unique_ptr<EdgeUpdateBatch<EdgeDataWeightUpdate>> edge_updates, EdgeUpdateVersion type);
template void EdgeUpdateManager<VertexDataSemantic, EdgeDataSemanticSOA, EdgeDataSemanticUpdate>::receiveEdgeUpdates(std::unique_ptr<EdgeUpdateBatch<EdgeDataSemanticUpdate>> edge_updates, EdgeUpdateVersion type);

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType, typename EdgeUpdateType>
void EdgeUpdateManager<VertexDataType, EdgeDataType, EdgeUpdateType>::hostCudaAllocConcurrentUpdates()
{
  HANDLE_ERROR(cudaHostAlloc((void **)&(updates_insertion->raw_edge_update), updates_insertion->edge_update.size() * sizeof(EdgeUpdateType), cudaHostAllocDefault));
  HANDLE_ERROR(cudaHostAlloc((void **)&(updates_deletion->raw_edge_update), updates_deletion->edge_update.size() * sizeof(EdgeUpdateType), cudaHostAllocDefault));
}

template void EdgeUpdateManager<VertexData, EdgeData, EdgeDataUpdate>::hostCudaAllocConcurrentUpdates();
template void EdgeUpdateManager<VertexDataWeight, EdgeDataWeight, EdgeDataWeightUpdate>::hostCudaAllocConcurrentUpdates();
template void EdgeUpdateManager<VertexDataSemantic, EdgeDataSemantic, EdgeDataSemanticUpdate>::hostCudaAllocConcurrentUpdates();
template void EdgeUpdateManager<VertexData, EdgeDataSOA, EdgeDataUpdate>::hostCudaAllocConcurrentUpdates();
template void EdgeUpdateManager<VertexDataWeight, EdgeDataWeightSOA, EdgeDataWeightUpdate>::hostCudaAllocConcurrentUpdates();
template void EdgeUpdateManager<VertexDataSemantic, EdgeDataSemanticSOA, EdgeDataSemanticUpdate>::hostCudaAllocConcurrentUpdates();

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType, typename EdgeUpdateType>
void EdgeUpdateManager<VertexDataType, EdgeDataType, EdgeUpdateType>::hostCudaFreeConcurrentUpdates()
{
  HANDLE_ERROR(cudaFreeHost(updates_insertion->raw_edge_update));
  HANDLE_ERROR(cudaFreeHost(updates_deletion->raw_edge_update));
}

template void EdgeUpdateManager<VertexData, EdgeData, EdgeDataUpdate>::hostCudaFreeConcurrentUpdates();
template void EdgeUpdateManager<VertexDataWeight, EdgeDataWeight, EdgeDataWeightUpdate>::hostCudaFreeConcurrentUpdates();
template void EdgeUpdateManager<VertexDataSemantic, EdgeDataSemantic, EdgeDataSemanticUpdate>::hostCudaFreeConcurrentUpdates();
template void EdgeUpdateManager<VertexData, EdgeDataSOA, EdgeDataUpdate>::hostCudaFreeConcurrentUpdates();
template void EdgeUpdateManager<VertexDataWeight, EdgeDataWeightSOA, EdgeDataWeightUpdate>::hostCudaFreeConcurrentUpdates();
template void EdgeUpdateManager<VertexDataSemantic, EdgeDataSemanticSOA, EdgeDataSemanticUpdate>::hostCudaFreeConcurrentUpdates();

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType, typename EdgeUpdateType>
void EdgeUpdateManager<VertexDataType, EdgeDataType, EdgeUpdateType>::writeEdgeUpdatesToFile(vertex_t number_vertices, vertex_t batch_size, const std::string& filename)
{
  std::ofstream file(filename);
  srand(static_cast<unsigned int>(time(NULL)));

  if(file.is_open())
  {
    for(vertex_t i = 0; i < batch_size; ++i)
    {
      vertex_t edge_src = rand() % number_vertices;
      vertex_t edge_dst = rand() % number_vertices;
      file << edge_src << " " << edge_dst << std::endl;
    }
  }
  file.close();
  return;
}

template void EdgeUpdateManager<VertexData, EdgeData, EdgeDataUpdate>::writeEdgeUpdatesToFile(vertex_t number_vertices, vertex_t batch_size, const std::string& filename);
template void EdgeUpdateManager<VertexDataWeight, EdgeDataWeight, EdgeDataWeightUpdate>::writeEdgeUpdatesToFile(vertex_t number_vertices, vertex_t batch_size, const std::string& filename);
template void EdgeUpdateManager<VertexDataSemantic, EdgeDataSemantic, EdgeDataSemanticUpdate>::writeEdgeUpdatesToFile(vertex_t number_vertices, vertex_t batch_size, const std::string& filename);
template void EdgeUpdateManager<VertexData, EdgeDataSOA, EdgeDataUpdate>::writeEdgeUpdatesToFile(vertex_t number_vertices, vertex_t batch_size, const std::string& filename);
template void EdgeUpdateManager<VertexDataWeight, EdgeDataWeightSOA, EdgeDataWeightUpdate>::writeEdgeUpdatesToFile(vertex_t number_vertices, vertex_t batch_size, const std::string& filename);
template void EdgeUpdateManager<VertexDataSemantic, EdgeDataSemanticSOA, EdgeDataSemanticUpdate>::writeEdgeUpdatesToFile(vertex_t number_vertices, vertex_t batch_size, const std::string& filename);

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType, typename EdgeUpdateType>
void EdgeUpdateManager<VertexDataType, EdgeDataType, EdgeUpdateType>::writeEdgeUpdatesToFile(const std::unique_ptr<EdgeUpdateBatch<EdgeUpdateType>>& edges, vertex_t batch_size, const std::string& filename)
{
  std::ofstream file(filename);

  if(file.is_open())
  {
    for(vertex_t i = 0; i < batch_size; ++i)
    {
      vertex_t edge_src = edges->edge_update.at(i).source;
      vertex_t edge_dst = edges->edge_update.at(i).update.destination;
      file << "|" << edge_src << " " << edge_dst << std::endl;
    }
  }
  return;
}

template void EdgeUpdateManager<VertexData, EdgeData, EdgeDataUpdate>::writeEdgeUpdatesToFile(const std::unique_ptr<EdgeUpdateBatch<EdgeDataUpdate>>& edges, vertex_t batch_size, const std::string& filename);
template void EdgeUpdateManager<VertexDataWeight, EdgeDataWeight, EdgeDataWeightUpdate>::writeEdgeUpdatesToFile(const std::unique_ptr<EdgeUpdateBatch<EdgeDataWeightUpdate>>& edges, vertex_t batch_size, const std::string& filename);
template void EdgeUpdateManager<VertexDataSemantic, EdgeDataSemantic, EdgeDataSemanticUpdate>::writeEdgeUpdatesToFile(const std::unique_ptr<EdgeUpdateBatch<EdgeDataSemanticUpdate>>& edges, vertex_t batch_size, const std::string& filename);
template void EdgeUpdateManager<VertexData, EdgeDataSOA, EdgeDataUpdate>::writeEdgeUpdatesToFile(const std::unique_ptr<EdgeUpdateBatch<EdgeDataUpdate>>& edges, vertex_t batch_size, const std::string& filename);
template void EdgeUpdateManager<VertexDataWeight, EdgeDataWeightSOA, EdgeDataWeightUpdate>::writeEdgeUpdatesToFile(const std::unique_ptr<EdgeUpdateBatch<EdgeDataWeightUpdate>>& edges, vertex_t batch_size, const std::string& filename);
template void EdgeUpdateManager<VertexDataSemantic, EdgeDataSemanticSOA, EdgeDataSemanticUpdate>::writeEdgeUpdatesToFile(const std::unique_ptr<EdgeUpdateBatch<EdgeDataSemanticUpdate>>& edges, vertex_t batch_size, const std::string& filename);

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType, typename EdgeUpdateType>
std::unique_ptr<EdgeUpdateBatch<EdgeUpdateType>> EdgeUpdateManager<VertexDataType, EdgeDataType, EdgeUpdateType>::readEdgeUpdatesFromFile(const std::string& filename)
{
  std::unique_ptr<EdgeUpdateBatch<EdgeUpdateType>> edge_update (new EdgeUpdateBatch<EdgeUpdateType>());
  std::ifstream graph_file(filename);
  std::string line;

  while(std::getline(graph_file, line))
  {
    EdgeUpdateType edge_update_data;
    std::istringstream istream(line);

    istream >> edge_update_data.source;
    istream >> edge_update_data.update.destination;
    edge_update->edge_update.push_back(edge_update_data);
  }

  return std::move(edge_update);
}

template std::unique_ptr<EdgeUpdateBatch<EdgeDataUpdate>> EdgeUpdateManager<VertexData, EdgeData, EdgeDataUpdate>::readEdgeUpdatesFromFile(const std::string& filename);
template std::unique_ptr<EdgeUpdateBatch<EdgeDataWeightUpdate>> EdgeUpdateManager<VertexDataWeight, EdgeDataWeight, EdgeDataWeightUpdate>::readEdgeUpdatesFromFile(const std::string& filename);
template std::unique_ptr<EdgeUpdateBatch<EdgeDataSemanticUpdate>> EdgeUpdateManager<VertexDataSemantic, EdgeDataSemantic, EdgeDataSemanticUpdate>::readEdgeUpdatesFromFile(const std::string& filename);
template std::unique_ptr<EdgeUpdateBatch<EdgeDataUpdate>> EdgeUpdateManager<VertexData, EdgeDataSOA, EdgeDataUpdate>::readEdgeUpdatesFromFile(const std::string& filename);
template std::unique_ptr<EdgeUpdateBatch<EdgeDataWeightUpdate>> EdgeUpdateManager<VertexDataWeight, EdgeDataWeightSOA, EdgeDataWeightUpdate>::readEdgeUpdatesFromFile(const std::string& filename);
template std::unique_ptr<EdgeUpdateBatch<EdgeDataSemanticUpdate>> EdgeUpdateManager<VertexDataSemantic, EdgeDataSemanticSOA, EdgeDataSemanticUpdate>::readEdgeUpdatesFromFile(const std::string& filename);

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType, typename EdgeUpdateType>
void EdgeUpdateManager<VertexDataType, EdgeDataType, EdgeUpdateType>::hostEdgeInsertion(const std::unique_ptr<GraphParser>& parser)
{
  auto& adjacency = parser->getAdjacency();
  auto& offset = parser->getOffset();
  auto number_vertices = parser->getNumberOfVertices();
  int number_updates;
  if (updates)
  {
    number_updates = updates->edge_update.size();
  }
  else
  {
    number_updates = updates_insertion->edge_update.size();
  }

  // Go over all updates
  for(int i = 0; i < number_updates; ++i)
  {
    // Set iterator to begin()
    auto iter = adjacency.begin();
    vertex_t edge_src, edge_dst;

    if (updates)
    {
      edge_src = updates->edge_update.at(i).source;
      edge_dst = updates->edge_update.at(i).update.destination;
    }
    else
    {
      edge_src = updates_insertion->edge_update.at(i).source;
      edge_dst = updates_insertion->edge_update.at(i).update.destination;
    }    

    //------------------------------------------------------------------------------
    // TODO: Currently no support for adding new vertices!!!!!
    //------------------------------------------------------------------------------
    //
    if(edge_src >= number_vertices || edge_dst >= number_vertices)
    {
      continue;
    }

    // Calculate iterator positions
    auto begin_iter = iter + offset.at(edge_src);
    auto end_iter = iter + offset.at(edge_src + 1);
    
    // Search item
    auto pos = std::find(begin_iter, end_iter, edge_dst);
    if(pos != end_iter)
    {
      // Edge already present     
      continue;
    }
    else
    {
      // Insert edge
      adjacency.insert(pos, edge_dst);
      
      // Update offset list (on the host this is number_vertices + 1 in size)
      for(auto i = edge_src + 1; i < (number_vertices + 1); ++i)
      {
        offset[i] += 1;
      }
    }

  }
  return;
}

template void EdgeUpdateManager<VertexData, EdgeData, EdgeDataUpdate>::hostEdgeInsertion(const std::unique_ptr<GraphParser>& parser);
template void EdgeUpdateManager<VertexDataWeight, EdgeDataWeight, EdgeDataWeightUpdate>::hostEdgeInsertion(const std::unique_ptr<GraphParser>& parser);
template void EdgeUpdateManager<VertexDataSemantic, EdgeDataSemantic, EdgeDataSemanticUpdate>::hostEdgeInsertion(const std::unique_ptr<GraphParser>& parser);
template void EdgeUpdateManager<VertexData, EdgeDataSOA, EdgeDataUpdate>::hostEdgeInsertion(const std::unique_ptr<GraphParser>& parser);
template void EdgeUpdateManager<VertexDataWeight, EdgeDataWeightSOA, EdgeDataWeightUpdate>::hostEdgeInsertion(const std::unique_ptr<GraphParser>& parser);
template void EdgeUpdateManager<VertexDataSemantic, EdgeDataSemanticSOA, EdgeDataSemanticUpdate>::hostEdgeInsertion(const std::unique_ptr<GraphParser>& parser);

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType, typename EdgeUpdateType>
void EdgeUpdateManager<VertexDataType, EdgeDataType, EdgeUpdateType>::hostEdgeDeletion(const std::unique_ptr<GraphParser>& parser)
{
  auto& adjacency = parser->getAdjacency();
  auto& offset = parser->getOffset();
  auto number_vertices = parser->getNumberOfVertices();
  int number_updates;
  if (updates)
  {
    number_updates = updates->edge_update.size();
  }
  else
  {
    number_updates = updates_insertion->edge_update.size();
  }

  // Go over all updates
  for(int i = 0; i < number_updates; ++i)
  {
    // Set iterator to begin()
    auto iter = adjacency.begin();
    vertex_t edge_src, edge_dst;

    if (updates)
    {
      edge_src = updates->edge_update.at(i).source;
      edge_dst = updates->edge_update.at(i).update.destination;
    }
    else
    {
      edge_src = updates_insertion->edge_update.at(i).source;
      edge_dst = updates_insertion->edge_update.at(i).update.destination;
    }

    // Check if valid vertices are given
    if(edge_src >= number_vertices || edge_dst >= number_vertices)
    {
      continue;
    }

    // Calculate iterator positions
    auto begin_iter = iter + offset.at(edge_src);
    auto end_iter = iter + offset.at(edge_src + 1);

    // Search item
    auto pos = std::find(begin_iter, end_iter, edge_dst);
    if(pos != end_iter)
    {
      // Found edge, will be deleted now
      adjacency.erase(pos); 
      
      // Update offset list (on the host this is number_vertices + 1 in size)
      for(auto i = edge_src + 1; i < (number_vertices + 1); ++i)
      {
        offset[i] -= 1;
      }
    }
    else
    {
      // Edge not present
      continue;
    }
  }
  return;
}

template void EdgeUpdateManager<VertexData, EdgeData, EdgeDataUpdate>::hostEdgeDeletion(const std::unique_ptr<GraphParser>& parser);
template void EdgeUpdateManager<VertexDataWeight, EdgeDataWeight, EdgeDataWeightUpdate>::hostEdgeDeletion(const std::unique_ptr<GraphParser>& parser);
template void EdgeUpdateManager<VertexDataSemantic, EdgeDataSemantic, EdgeDataSemanticUpdate>::hostEdgeDeletion(const std::unique_ptr<GraphParser>& parser);
template void EdgeUpdateManager<VertexData, EdgeDataSOA, EdgeDataUpdate>::hostEdgeDeletion(const std::unique_ptr<GraphParser>& parser);
template void EdgeUpdateManager<VertexDataWeight, EdgeDataWeightSOA, EdgeDataWeightUpdate>::hostEdgeDeletion(const std::unique_ptr<GraphParser>& parser);
template void EdgeUpdateManager<VertexDataSemantic, EdgeDataSemanticSOA, EdgeDataSemanticUpdate>::hostEdgeDeletion(const std::unique_ptr<GraphParser>& parser);

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType, typename EdgeUpdateType>
void EdgeUpdateManager<VertexDataType, EdgeDataType, EdgeUpdateType>::writeGraphsToFile(const std::unique_ptr<aimGraphCSR>& verify_graph,
                                                                                        const std::unique_ptr<GraphParser>& graph_parser,
                                                                                        const std::string& filename)
{
  static int counter = 0;
  std::string prover_filename = filename;// + "_prover" + std::to_string(counter) + ".txt";
  //std::string verifier_filename = filename + "_verifier" + std::to_string(counter) + ".txt";
  int number_vertices = graph_parser->getNumberOfVertices();
  ++counter;

  // Start with prover graph
  std::ofstream prover_file(prover_filename);

  // Write number of vertices and edges
  prover_file << verify_graph->number_vertices << " " << verify_graph->number_edges << std::endl;

  for(index_t i = 0; i < verify_graph->number_vertices; ++i)
  {
    int offset = verify_graph->h_offset[i];
    int neighbours = ((i == (verify_graph->number_vertices - 1)) 
                      ? (verify_graph->number_edges) 
                      : (verify_graph->h_offset[i + 1]) ) 
                      - offset;
    for(int j = 0; j < neighbours; ++j)
    {
      // Graph format uses 1-n, we have 0 - (n-1), hence now add 1
      prover_file << verify_graph->h_adjacency[offset + j] + 1;
      if(j < (neighbours - 1))
        prover_file << " ";
    }
    prover_file << "\n";
  }

  prover_file.close();

  // End with verifier graph
  // std::ofstream verifier_file(verifier_filename);

  // // Write number of vertices and edges
  // verifier_file << number_vertices << " " << graph_parser->getAdjacency().size()  << std::endl;

  // for(int i = 0; i < number_vertices; ++i)
  // {
  //   int offset = graph_parser->getOffset().at(i);
  //   int neighbours = graph_parser->getOffset().at(i+1) - offset;
  //   for(int j = 0; j < neighbours; ++j)
  //   {
  //     // Graph format uses 1-n, we have 0 - (n-1), hence now add 1
  //     verifier_file << graph_parser->getAdjacency().at(offset + j) + 1;
  //     if(j < (neighbours - 1))
  //       verifier_file << " ";
  //   }
  //   verifier_file << "\n";
  // }

  // verifier_file.close();

  // std::cout << "Writing files is done" << std::endl;

  return;
}

template void EdgeUpdateManager<VertexData, EdgeData, EdgeDataUpdate>::writeGraphsToFile(const std::unique_ptr<aimGraphCSR>& verify_graph, const std::unique_ptr<GraphParser>& graph_parser, const std::string& filename);
template void EdgeUpdateManager<VertexDataWeight, EdgeDataWeight, EdgeDataWeightUpdate>::writeGraphsToFile(const std::unique_ptr<aimGraphCSR>& verify_graph, const std::unique_ptr<GraphParser>& graph_parser, const std::string& filename);
template void EdgeUpdateManager<VertexDataSemantic, EdgeDataSemantic, EdgeDataSemanticUpdate>::writeGraphsToFile(const std::unique_ptr<aimGraphCSR>& verify_graph, const std::unique_ptr<GraphParser>& graph_parser, const std::string& filename);
template void EdgeUpdateManager<VertexData, EdgeDataSOA, EdgeDataUpdate>::writeGraphsToFile(const std::unique_ptr<aimGraphCSR>& verify_graph, const std::unique_ptr<GraphParser>& graph_parser, const std::string& filename);
template void EdgeUpdateManager<VertexDataWeight, EdgeDataWeightSOA, EdgeDataWeightUpdate>::writeGraphsToFile(const std::unique_ptr<aimGraphCSR>& verify_graph, const std::unique_ptr<GraphParser>& graph_parser, const std::string& filename);
template void EdgeUpdateManager<VertexDataSemantic, EdgeDataSemanticSOA, EdgeDataSemanticUpdate>::writeGraphsToFile(const std::unique_ptr<aimGraphCSR>& verify_graph, const std::unique_ptr<GraphParser>& graph_parser, const std::string& filename);