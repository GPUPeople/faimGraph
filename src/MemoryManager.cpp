//------------------------------------------------------------------------------
// MemoryManager.cpp
//
// faimGraph
//
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
template <typename VertexDataType, typename EdgeDataType>
void MemoryManager::initialize(const std::shared_ptr<Config>& config)
{
  if (initialized)
    return;

  // Initialisation
  setGraphMode(config);
  setEdgesPerBlock();
  estimateStorageRequirements<VertexDataType, EdgeDataType>(config);
  setQueueSizeAndPosition(config->testruns_.at(config->testrun_index_)->params->queuesize_);

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

template void MemoryManager::initialize<VertexData, EdgeData>(const std::shared_ptr<Config>& config);
template void MemoryManager::initialize<VertexData, EdgeDataMatrix>(const std::shared_ptr<Config>& config);
template void MemoryManager::initialize<VertexDataWeight, EdgeDataWeight>(const std::shared_ptr<Config>& config);
template void MemoryManager::initialize<VertexDataSemantic, EdgeDataSemantic>(const std::shared_ptr<Config>& config);
template void MemoryManager::initialize<VertexData, EdgeDataSOA>(const std::shared_ptr<Config>& config);
template void MemoryManager::initialize<VertexData, EdgeDataMatrixSOA>(const std::shared_ptr<Config>& config);
template void MemoryManager::initialize<VertexDataWeight, EdgeDataWeightSOA>(const std::shared_ptr<Config>& config);
template void MemoryManager::initialize<VertexDataSemantic, EdgeDataSemanticSOA>(const std::shared_ptr<Config>& config);

template<typename EdgeDataType>
uint64_t estimateAdditionalMemoryRequirements(const std::shared_ptr<Config>& config, vertex_t number_vertices, vertex_t number_edges)
{
	uint64_t requirements{ 0 }, update_requirements{ 0 }, csr_requirements{ 0 };

	// Calculate Update Requirements
	update_requirements = MAXIMAL_BATCH_SIZE * EdgeDataType::sizeOfUpdateData();
	if (config->testruns_.at(config->testrun_index_)->params->update_variant_ == ConfigurationParameters::UpdateVariant::VERTEXCENTRIC ||
		config->testruns_.at(config->testrun_index_)->params->update_variant_ == ConfigurationParameters::UpdateVariant::VERTEXCENTRICSORTED)
	{
		// EdgeUpdate Preprocessing
		update_requirements += (number_vertices + 1) * 2 * sizeof(index_t);
	}

	//// Calculate CSR Requirements
	//csr_requirements = (sizeof(vertex_t) * number_vertices) +		// offset
	//						(sizeof(vertex_t) * number_edges) +				// adjacency
	//						(sizeof(vertex_t) * number_vertices) +			// d_neighbours
	//						(sizeof(vertex_t) * (number_vertices + 1)); 	// d_block_requirements


	requirements = (update_requirements > csr_requirements) ? update_requirements : csr_requirements;
	//printf("Update Requirements: %u MB | CSR Requirements: %u MB\n", update_requirements / (1024*1024), csr_requirements / (1024 * 1024));
	return requirements;
}

#define ESTIMATE_MEMORY

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType>
void MemoryManager::estimateStorageRequirements(const std::shared_ptr<Config>& config)
{
	// We want cacheline aligned memory
	int mem_offset = MEMMANOFFSET * static_cast<int>(ceil(static_cast<float>(sizeof(MemoryManager)) / static_cast<float>(MEMMANOFFSET)));
	if (mem_offset > MEMMANOFFSET)
	{
		std::cout << "Re-Evaluate size constraints" << std::endl;
		exit(-1);
	}

#ifdef ESTIMATE_MEMORY
	uint64_t additional_space_requirements = estimateAdditionalMemoryRequirements<EdgeDataType>(config, number_vertices, number_edges);
	float page_flux_factor = 1.5f;
	uint64_t page_estimation_per_vertex = (ceil(static_cast<float>(number_edges) / static_cast<float>(number_vertices * edges_per_page)) * number_vertices) * page_flux_factor;
	//printf("Page Estimation per vertex: %u %u %u %u\n", page_estimation_per_vertex, number_edges, number_vertices, edges_per_page);
	uint64_t size_estimation = static_cast<uint64_t>(mem_offset) + 
		static_cast<uint64_t>(sizeof(VertexData) * number_vertices) +
		static_cast<uint64_t>(page_estimation_per_vertex * page_size) +
		static_cast<uint64_t>((config->testruns_.at(config->testrun_index_)->params->queuesize_ * sizeof(index_t) * 2)) +
		static_cast<uint64_t>(config->testruns_.at(config->testrun_index_)->params->stacksize_) +
		additional_space_requirements; 
	total_memory = size_estimation;

	// printf("Size Estimation : %lu MB  ---  Pages: %lu MB | Vertices: %lu MB | Helper: %lu MB | AdditionalSpace: %lu MB\n", 
	// 	size_estimation / (1024 * 1024),
	// 	static_cast<uint64_t>(page_estimation_per_vertex * page_size) / (1024 * 1024),
	// 	static_cast<uint64_t>(sizeof(VertexData) * number_vertices) / (1024*1024),
	// 	(static_cast<uint64_t>((config->testruns_.at(config->testrun_index_)->params->queuesize_ * sizeof(index_t) * 2)) +
	// 	static_cast<uint64_t>(config->testruns_.at(config->testrun_index_)->params->stacksize_)) / (1024*1024),
	// 	additional_space_requirements / (1024*1024));
	
#endif

	// Allocate memory
	if (d_memory == nullptr)
	{
		HANDLE_ERROR(cudaMalloc((void **)&d_memory, total_memory));
	}

	// Point stack pointer to end of device memory
	d_stack_pointer = d_memory;
	d_stack_pointer += total_memory;

	// Place data pointer after memory_manager and decrement the available memory
	d_data = d_memory + mem_offset;
	decreaseAvailableMemory(mem_offset);

	start_index = static_cast<uint64_t>((total_memory - (config->testruns_.at(config->testrun_index_)->params->queuesize_ * sizeof(index_t) * 2) - config->testruns_.at(config->testrun_index_)->params->stacksize_ - mem_offset) / page_size) - 1;
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
void MemoryManager::resetFaimGraph(vertex_t number_vertices, vertex_t number_edges)
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
  d_page_queue.queue_ = reinterpret_cast<index_t*>(d_stack_pointer);
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
