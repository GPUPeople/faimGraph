/*!/------------------------------------------------------------------------------
* queryTC.cpp
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

template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename UpdateDataType>
void testrunImplementation(const std::shared_ptr<Config>& config, const std::unique_ptr<Testruns>& testrun);

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    std::cout << "Usage: ./queryTC <configuration-file>" << std::endl;
    return RET_ERROR;
  }
  std::cout << "########## faimGraph Demo ##########" << std::endl;

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
  std::cout << "Used GPU Name: " << prop.name << std::endl;

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
  bool realisticDeletion = true;
  bool gpuVerification = true;
  bool writeToFile = false;
  bool duplicate_check = true;
  unsigned int range = 1000;
  unsigned int offset = 0;

  int vertex_batch_size = 100;
  int max_adjacency_size = 10;

  for (auto batchsize : testrun->batchsizes)
  {
    std::cout << "Batchsize: " << batchsize << std::endl;
    for (const auto& graph : testrun->graphs)
    {
      // Timing information
      float time_elapsed_init = 0;
      float time_elapsed_query = 0;

      //Setup graph parser and read in file
      std::unique_ptr<GraphParser> parser(new GraphParser(graph));
      if (!parser->parseGraph())
      {
        std::cout << "Error while parsing graph" << std::endl;
        return;
      }

      for (int i = 0; i < testrun->params->rounds_; i++, offset += range)
      {
        //------------------------------------------------------------------------------
        // Initialization phase
        //------------------------------------------------------------------------------
        //

        //std::cout << "Round: " << i + 1 << std::endl;

        std::unique_ptr<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>> faimGraph(std::make_unique<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>>(config, parser));
        std::unique_ptr<EdgeQueryManager<VertexDataType, EdgeDataType>> query_manager(std::make_unique<EdgeQueryManager<VertexDataType, EdgeDataType>>());

        start_clock(ce_start, ce_stop);

		  faimGraph->initializeMemory(parser);

        time_diff = end_clock(ce_start, ce_stop);
        time_elapsed_init += time_diff;

        for (int j = 0; j < testrun->params->update_rounds_; j++)
        {
          // std::cout << "Update-Round: " << j + 1 << std::endl;
          query_manager->generateQueries(parser->getNumberOfVertices(), batchsize, (i * testrun->params->rounds_) + j);

          //------------------------------------------------------------------------------
          // Start Timer for Queries
          //------------------------------------------------------------------------------
          //
          start_clock(ce_start, ce_stop);

          query_manager->deviceQuery(faimGraph->memory_manager, config);

          time_diff = end_clock(ce_start, ce_stop);
          time_elapsed_query += time_diff;
        }
      }

      // std::cout << "Time elapsed during initialization:   ";
      // std::cout << std::setw(10) << time_elapsed_init / static_cast<float>(testrun->params->rounds_) << " ms" << std::endl;

      std::cout << "Time elapsed during edge insertion:   ";
      std::cout << std::setw(10) << time_elapsed_query / static_cast<float>(testrun->params->rounds_ * testrun->params->update_rounds_) << " ms" << std::endl;
    }
    std::cout << "################################################################" << std::endl;
    std::cout << "################################################################" << std::endl;
    std::cout << "################################################################" << std::endl;
  }
  // Increment the testrun index
  config->testrun_index_++;
}
