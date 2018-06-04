//------------------------------------------------------------------------------
// MemoryManager.cpp
//
// Masterproject/-thesis aimGraph
//
// Authors: Martin Winter, 1130688
//------------------------------------------------------------------------------
//
#include <iostream>
#include <math.h>

#include "MemoryManager.h"
#include "GraphParser.h"
#include "ConfigurationParser.h"
#include "MemoryLayout.h"

//------------------------------------------------------------------------------
//
MemoryManager::MemoryManager(uint64_t memory_size, const std::shared_ptr<Config>& config, std::unique_ptr<GraphParser>& graph_parser) :
  total_memory{ memory_size },
  free_memory{ memory_size },
  edgeblock_lock{ UNLOCK },
  next_free_page{ 0 },
  next_free_vertex_index{ graph_parser->getNumberOfVertices() },
  access_counter{0},
  page_size{ config->testruns_.at(config->testrun_index_)->params->page_size_ },
  number_vertices{ graph_parser->getNumberOfVertices() },
  number_edges{ static_cast<vertex_t>(graph_parser->getAdjacency().size()) },
  page_linkage{ config->testruns_.at(config->testrun_index_)->params->page_linkage_}
{}

//------------------------------------------------------------------------------
//
void MemoryManager::initialize(const std::shared_ptr<Config>& config)
{
  if (initialized)
    return;

  // Allocate memory
  HANDLE_ERROR(cudaMalloc((void **)&d_memory, total_memory));

  // Point stack pointer to end of device memory
  d_stack_pointer = d_memory;
  d_stack_pointer += total_memory;

  // We want cacheline aligned memory
  int mem_offset = MEMMANOFFSET * static_cast<int>(ceil(static_cast<float>(sizeof(MemoryManager)) / static_cast<float>(MEMMANOFFSET)));
  if (mem_offset > MEMMANOFFSET)
  {
    std::cout << "Re-Evaluate size constraints" << std::endl;
    exit(-1);
  }

  // Set Indexshift
  int cachelines_per_vertex_attribute = (int)ceil((float)(number_vertices * sizeof(vertex_t)) / (float)(CACHELINESIZE));
  index_shift = (cachelines_per_vertex_attribute * CACHELINESIZE);

  // Place data pointer after memory_manager and decrement the available memory
  d_data = d_memory + mem_offset;
  decreaseAvailableMemory(mem_offset);

  // Initialisation
  setGraphMode(config);
  setEdgesPerBlock();
  setQueueSizeAndPosition(config->testruns_.at(config->testrun_index_)->params->queuesize_);

  start_index = static_cast<uint64_t>((total_memory - (config->testruns_.at(config->testrun_index_)->params->queuesize_ * sizeof(index_t) * 2) - config->testruns_.at(config->testrun_index_)->params->stacksize_ - MEMMANOFFSET) / page_size) - 1;

  if (config->testruns_.at(config->testrun_index_)->params->directionality_ == ConfigurationParameters::GraphDirectionality::DIRECTED)
  {
    graph_directionality = GraphDirectionality::DIRECTED;
  }
  else
  {
    graph_directionality = GraphDirectionality::UNDIRECTED;
  }
  initialized = true;
}

//------------------------------------------------------------------------------
//
void MemoryManager::setGraphMode(const std::shared_ptr<Config>& config)
{
  graph_mode = config->testruns_.at(config->testrun_index_)->params->graph_mode_;
}

//------------------------------------------------------------------------------
//
void MemoryManager::queryErrorCode()
{
  if ((error_code & static_cast<unsigned int>(ErrorCode::PAGE_QUEUE_FULL)))
  {
    std::cout << "Page Queue is full" << std::endl;
  }
}

//------------------------------------------------------------------------------
//
void MemoryManager::resetAimGraph(vertex_t number_vertices, vertex_t number_edges)
{
  next_free_page = 0;
  next_free_vertex_index = number_vertices;
  access_counter = 0;
  this->number_vertices = number_vertices;
  // Reset page queues
  d_page_queue.resetQueue();
  d_vertex_queue.resetQueue();
}

//------------------------------------------------------------------------------
//
void MemoryManager::setQueueSizeAndPosition(int size)
{
  // Setup both queues and stack pointer
  memory_t* tmp_ptr = d_memory + total_memory;
  d_page_queue.queue_ = reinterpret_cast<index_t*>(tmp_ptr);
  int maxNumberBlocks = size;
  d_page_queue.queue_ -= maxNumberBlocks;
  d_page_queue.size_ = maxNumberBlocks;
  d_vertex_queue.queue_ = d_page_queue.queue_ - maxNumberBlocks;
  d_vertex_queue.size_ = maxNumberBlocks;

  d_stack_pointer = reinterpret_cast<memory_t*>(d_vertex_queue.queue_);
}

//------------------------------------------------------------------------------
//
void MemoryManager::setEdgesPerBlock()
{
  int number_of_indices = 1;
  if (page_linkage == ConfigurationParameters::PageLinkage::DOUBLE)
  {
    number_of_indices = 2;
  }

  if(graph_mode == ConfigurationParameters::GraphMode::SEMANTIC)
  {
    edges_per_page = (page_size - (number_of_indices * sizeof(index_t))) / sizeof(EdgeDataSemantic);
  }
  else if(graph_mode == ConfigurationParameters::GraphMode::WEIGHT)
  {
    edges_per_page = (page_size - (number_of_indices * sizeof(index_t))) / sizeof(EdgeDataWeight);
  }
  else if (graph_mode == ConfigurationParameters::GraphMode::MATRIX)
  {
	  edges_per_page = (page_size - (number_of_indices * sizeof(index_t))) / sizeof(EdgeDataMatrix);
  }
  else
  {
    edges_per_page = (page_size - (number_of_indices * sizeof(index_t))) / sizeof(EdgeData);
  }
}


//------------------------------------------------------------------------------
//
void MemoryManager::estimateInitialStorageRequirements(vertex_t numberVertices, vertex_t numberEdges, int batchsize, int size_of_edgedata)
{
  int csr_data_size = ceil(static_cast<float>(sizeof(vertex_t) * (numberEdges + (5 * numberVertices))) / 1000000);
  int verify_graph_size = ceil(static_cast<float>(sizeof(vertex_t) * (numberEdges + (2 * numberVertices))) / 1000000);
  int edge_update_batch_size = ceil(static_cast<float>(batchsize * (4 + size_of_edgedata)) / 1000000);
  int edge_update_preprocessing_size = ceil(static_cast<float>(4 * ((2 * numberVertices) + batchsize)) / 1000000);

  // std::cout << "CSR:" << csr_data_size << " mB" << std::endl;
  // std::cout << "Verify:" << verify_graph_size << " mB" << std::endl;
  // std::cout << "Update:" << edge_update_batch_size << " mB" << std::endl;
  // std::cout << "Preprocessing:" << edge_update_preprocessing_size << " mB" << std::endl;

  // std::cout << "---------------------------------------" << std::endl;
  std::cout << "Maximal Stacksize required: " << edge_update_batch_size + edge_update_preprocessing_size << " mB" << std::endl;
  if(csr_data_size > verify_graph_size)
    std::cout << "Maximal StaticSize required: " << csr_data_size << " mB" << std::endl;
  else
    std::cout << "Maximal StaticSize required: " << verify_graph_size << " mB" << std::endl;
}

//------------------------------------------------------------------------------
//
void MemoryManager::printEssentials(const std::string& text)
{
  std::cout << std::endl;
  std::cout << "----- Memory Manager Essentials | " << text <<" -----" << std::endl;
  std::cout << "Number Vertices: " << number_vertices << std::endl;
  std::cout << "Max Number Vertices: " << next_free_vertex_index << std::endl;
  std::cout << "Pages Used: " << next_free_page - d_page_queue.count_ << " / " << number_pages << std::endl;
  std::cout << "PageQueue Fill-Level:   " << static_cast<int>(100 * (static_cast<float>(d_page_queue.count_) / d_page_queue.size_)) << " %" << std::endl;
  std::cout << "VertexQueue Fill-Level: " << static_cast<int>(100 * (static_cast<float>(d_vertex_queue.count_) / d_vertex_queue.size_)) << " %" << std::endl;
  std::cout << std::endl;
}

//------------------------------------------------------------------------------
//
CSRData::CSRData(const std::unique_ptr<GraphParser>& graph_parser,
                 std::unique_ptr<MemoryManager>& memory_manager,
                unsigned int vertex_offset):
  scoped_mem_access_counter{memory_manager.get(), sizeof(vertex_t) * (
                                                  graph_parser->getAdjacency().size() +
                                                  graph_parser->getOffset().size() +
                                                  (4 * graph_parser->getNumberOfVertices()))}
{
  TemporaryMemoryAccessHeap temp_memory_dispenser(memory_manager.get(), vertex_offset + graph_parser->getNumberOfVertices(), sizeof(VertexData));

  d_offset = temp_memory_dispenser.getTemporaryMemory<vertex_t>(graph_parser->getOffset().size());
  d_adjacency = temp_memory_dispenser.getTemporaryMemory<vertex_t>(graph_parser->getAdjacency().size());
  if (graph_parser->isGraphMatrix())
  {
    d_matrix_values = temp_memory_dispenser.getTemporaryMemory<matrix_t>(graph_parser->getMatrixValues().size());
  }
 
  d_neighbours = temp_memory_dispenser.getTemporaryMemory<vertex_t>(graph_parser->getNumberOfVertices());
  d_capacity = temp_memory_dispenser.getTemporaryMemory<vertex_t>(graph_parser->getNumberOfVertices());
  d_block_requirements = temp_memory_dispenser.getTemporaryMemory<vertex_t>(graph_parser->getNumberOfVertices());
  d_mem_requirements = temp_memory_dispenser.getTemporaryMemory<vertex_t>(graph_parser->getNumberOfVertices());

  // Copy adjacency/offset list to device
  HANDLE_ERROR(cudaMemcpy(d_adjacency, graph_parser->getAdjacency().data(),
                          sizeof(vertex_t) * graph_parser->getAdjacency().size(),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_offset, graph_parser->getOffset().data(),
                          sizeof(vertex_t) * graph_parser->getOffset().size(),
                          cudaMemcpyHostToDevice));
  if (graph_parser->isGraphMatrix())
  {
    HANDLE_ERROR(cudaMemcpy(d_matrix_values, graph_parser->getMatrixValues().data(),
                            sizeof(matrix_t) * graph_parser->getMatrixValues().size(),
                            cudaMemcpyHostToDevice));
  }
}

//------------------------------------------------------------------------------
//
CSRData::CSRData(std::unique_ptr<MemoryManager>& memory_manager, unsigned int number_rows, unsigned int vertex_offset):
  scoped_mem_access_counter{ memory_manager.get(), sizeof(vertex_t) * (
    number_rows + 1 +
    (4 * number_rows)) }
{
  TemporaryMemoryAccessHeap temp_memory_dispenser(memory_manager.get(), vertex_offset + number_rows, sizeof(VertexData));
  d_offset = temp_memory_dispenser.getTemporaryMemory<vertex_t>(number_rows + 1);
  d_neighbours = temp_memory_dispenser.getTemporaryMemory<vertex_t>(number_rows);
  d_capacity = temp_memory_dispenser.getTemporaryMemory<vertex_t>(number_rows);
  d_block_requirements = temp_memory_dispenser.getTemporaryMemory<vertex_t>(number_rows);
  d_mem_requirements = temp_memory_dispenser.getTemporaryMemory<vertex_t>(number_rows);

  HANDLE_ERROR(cudaMemset(d_offset,
                          0,
                          sizeof(vertex_t) * (number_rows + 1)));
}

//------------------------------------------------------------------------------
//
aimGraphCSR::aimGraphCSR(std::unique_ptr<MemoryManager>& memory_manager):
number_vertices{memory_manager->next_free_vertex_index},
scoped_mem_access_counter{ memory_manager.get(), sizeof(vertex_t) * (2 * memory_manager->next_free_vertex_index) }
{
  TemporaryMemoryAccessHeap temp_memory_dispenser(memory_manager.get(), memory_manager->next_free_vertex_index, sizeof(VertexData));

  d_offset = temp_memory_dispenser.getTemporaryMemory<vertex_t>(memory_manager->next_free_vertex_index);
  d_mem_requirement = temp_memory_dispenser.getTemporaryMemory<vertex_t>(memory_manager->next_free_vertex_index);
  d_adjacency = temp_memory_dispenser.getTemporaryMemory<vertex_t>(0);
    
  h_offset = (vertex_t*) malloc(sizeof(vertex_t) * memory_manager->next_free_vertex_index);
}

//------------------------------------------------------------------------------
//
aimGraphCSR::aimGraphCSR(std::unique_ptr<MemoryManager>& memory_manager, vertex_t vertex_offset, vertex_t number_vertices) :
  number_vertices{ number_vertices },
  scoped_mem_access_counter{ memory_manager.get(), sizeof(vertex_t) * (2 * number_vertices) }
{
  TemporaryMemoryAccessHeap temp_memory_dispenser(memory_manager.get(), vertex_offset + number_vertices, sizeof(VertexData));

  d_offset = temp_memory_dispenser.getTemporaryMemory<vertex_t>(number_vertices);
  d_mem_requirement = temp_memory_dispenser.getTemporaryMemory<vertex_t>(number_vertices);
  d_adjacency = temp_memory_dispenser.getTemporaryMemory<vertex_t>(0);
  d_matrix_values = temp_memory_dispenser.getTemporaryMemory<matrix_t>(0);

  h_offset = (vertex_t*)malloc(sizeof(vertex_t) * memory_manager->next_free_vertex_index);
}