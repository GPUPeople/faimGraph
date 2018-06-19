/*!/------------------------------------------------------------------------------
 * reInitTC.cpp
 *
 * faimGraph
 *
 *------------------------------------------------------------------------------
*/

//------------------------------------------------------------------------------
// Library Includes
//
#include <iostream>
#include <iomanip>
#include <string>

//------------------------------------------------------------------------------
// Includes
//
#include "MemoryManager.h"
#include "GraphParser.h"
#include "faimGraph.h"
#include "EdgeUpdate.h"
#include "VertexUpdate.h"
#include "ConfigurationParser.h"
#include "CSVWriter.h"
#include "CSR.h"

void printCUDAStats(const std::string& init_string)
{
	// show memory usage of GPU
	size_t free_byte;
	size_t total_byte;
	cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;

	if ( cudaSuccess != cuda_status ){
			printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
			exit(1);
	}

	double free_db = (double)free_byte ;
	double total_db = (double)total_byte ;
	double used_db = total_db - free_db ;
	std::cout << init_string << " | ";
	printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n", used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}

template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename UpdateDataType>
void testrunImplementation(const std::shared_ptr<Config>& config, const std::unique_ptr<Testruns>& testrun);

template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename UpdateDataType>
void verification(std::unique_ptr<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>>& faimGraph, const std::string& outputstring, std::unique_ptr<GraphParser>& parser, const std::unique_ptr<Testruns>& testrun, int round, int updateround, bool duplicate_check);

int main(int argc, char *argv[])
{
	if(argc != 2)
	{
		std::cout << "Usage: ./reInitTC <configuration-file>" << std::endl;
		return RET_ERROR;
	}
  std::cout << "########## reInit Demo ##########" << std::endl;

	// Query device properties
	//queryAndPrintDeviceProperties();
  ConfigurationParser config_parser(argv[1]);
  std::cout << "Parse Configuration File" << std::endl;
  auto config = config_parser.parseConfiguration();
  // Print the configuration information
  printConfigurationInformation(config);

  cudaDeviceProp prop;
  cudaSetDevice(config->deviceID_);
  HANDLE_ERROR(cudaGetDeviceProperties(&prop, config->deviceID_));
  std::cout << "Used GPU Name: " << prop.name << " with ID: " << config->deviceID_ << std::endl;
  printCUDAStats("Init: ");

  // Perform testruns
  for (const auto& testrun : config->testruns_)
  {
    if (testrun->params->memory_layout_ == ConfigurationParameters::MemoryLayout::AOS)
    {
      if (testrun->params->graph_mode_ == ConfigurationParameters::GraphMode::SIMPLE)
      {
        testrunImplementation <VertexData, VertexUpdate, EdgeData, EdgeDataUpdate>(config, testrun);
      }
      else if (testrun->params->graph_mode_ == ConfigurationParameters::GraphMode::WEIGHT)
      {
        testrunImplementation <VertexDataWeight, VertexUpdateWeight, EdgeDataWeight, EdgeDataWeightUpdate>(config, testrun);
      }
      else if (testrun->params->graph_mode_ == ConfigurationParameters::GraphMode::SEMANTIC)
      {
        testrunImplementation <VertexDataSemantic, VertexUpdateSemantic, EdgeDataSemantic, EdgeDataSemanticUpdate>(config, testrun);
      }
      else
      {
        std::cout << "Error parsing graph mode configuration" << std::endl;
      }
    }
    else if (testrun->params->memory_layout_ == ConfigurationParameters::MemoryLayout::SOA)
    {
      if (testrun->params->graph_mode_ == ConfigurationParameters::GraphMode::SIMPLE)
      {
        testrunImplementation <VertexData, VertexUpdate, EdgeDataSOA, EdgeDataUpdate>(config, testrun);
      }
      else if (testrun->params->graph_mode_ == ConfigurationParameters::GraphMode::WEIGHT)
      {
        testrunImplementation <VertexDataWeight, VertexUpdateWeight, EdgeDataWeightSOA, EdgeDataWeightUpdate>(config, testrun);
      }
      else if (testrun->params->graph_mode_ == ConfigurationParameters::GraphMode::SEMANTIC)
      {
        testrunImplementation <VertexDataSemantic, VertexUpdateSemantic, EdgeDataSemanticSOA, EdgeDataSemanticUpdate>(config, testrun);
      }
      else
      {
        std::cout << "Error parsing graph mode configuration" << std::endl;
      }
    }
    else
    {
      std::cout << "Error parsing memory layout configuration" << std::endl;
    }
    
  }  

	return 0;
}

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename UpdateDataType>
void testrunImplementation(const std::shared_ptr<Config>& config, const std::unique_ptr<Testruns>& testrun)
{
  // Timing
  cudaEvent_t ce_start, ce_stop;
  float time_diff;
  std::vector<PerformanceData> perfData;

  // Global Properties
  bool realisticDeletion = false;
  bool gpuVerification = true;
  bool writeToFile = false;
  bool duplicate_check = true;

	for (auto batchsize : testrun->batchsizes)
	{
		std::cout << "Batchsize: " << batchsize << std::endl;
		for (const auto& graph : testrun->graphs)
		{
			// Timing information
			float time_elapsed_init = 0;
			float time_elapsed_reinit = 0;
			int warmup_rounds = 0;

			//Setup graph parser and read in file
			std::unique_ptr<GraphParser> parser(new GraphParser(graph));
			if (!parser->parseGraph())
			{
				std::cout << "Error while parsing graph" << std::endl;
				return;
			}

			float device_mem_size = config->device_mem_size_;

			for (int i = 0; i < testrun->params->rounds_ + warmup_rounds; i++)
			{
				//------------------------------------------------------------------------------
				// Initialization phase
				//------------------------------------------------------------------------------
				//
				
				printf("Round: %d\n", i + 1);

				printCUDAStats("Before Init");
				config->device_mem_size_ = device_mem_size;

				std::unique_ptr<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>> faimGraph(std::make_unique<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>>(config, parser));
				
				start_clock(ce_start, ce_stop);
				faimGraph->initializeMemory(parser);
				time_elapsed_init += end_clock(ce_start, ce_stop);

				//printCUDAStats("After Init");

				for (int j = 0; j < testrun->params->update_rounds_; ++j)
				{
					// Test reinitialization
					/*faimGraph->config->device_mem_size_ *= 1.10;*/

					start_clock(ce_start, ce_stop);
					auto return_csr = faimGraph->reinitializeFaimGraph(1.05f);
					float timing = end_clock(ce_start, ce_stop);
					time_elapsed_reinit += timing;

					printf("New Size: %lu MB - %f ms\n", (faimGraph->memory_manager->total_memory) / (1024*1024), timing);

					//printCUDAStats("After Re-Init");

					//verification(faimGraph, "Verify ReInitialization Round", parser, testrun, i, 0, duplicate_check);
				}

				// Let's retrieve a fresh graph
				parser->getFreshGraph();
			}
			std::cout << "Time elapsed during Initialization:   ";
			std::cout << std::setw(10) << time_elapsed_init / static_cast<float>(testrun->params->rounds_) << " ms" << std::endl;

			std::cout << "Time elapsed during Re-Initialization:   ";
			std::cout << std::setw(10) << time_elapsed_reinit / static_cast<float>(testrun->params->rounds_ * testrun->params->update_rounds_) << " ms" << std::endl;
		}
	}

	// Increment the testrun index
	config->testrun_index_++;
}

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename UpdateDataType>
void verification(std::unique_ptr<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>>& faimGraph,
                  const std::string& outputstring,
                  std::unique_ptr<GraphParser>& parser,
                  const std::unique_ptr<Testruns>& testrun,
                  int round, 
                  int updateround,
                  bool duplicate_check)
{
	std::cout << "############ " << outputstring << " " << (round * testrun->params->rounds_) + updateround << " ############" << std::endl;
	std::unique_ptr<aimGraphCSR> verify_graph = faimGraph->verifyGraphStructure (faimGraph->memory_manager);

	// Compare graph structures
	if (!faimGraph->compareGraphs(parser, verify_graph, duplicate_check))
	{
	std::cout << "########## Graphs are NOT the same ##########" << std::endl;
	exit(-1);
	}
}
