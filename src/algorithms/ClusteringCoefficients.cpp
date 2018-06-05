//------------------------------------------------------------------------------
// ClusteringCoefficients.cpp
//
// faimGraph
//
//------------------------------------------------------------------------------
//

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
#include "ConfigurationParser.h"
#include "CSVWriter.h"
#include "ClusteringCoefficients.h"

template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename UpdateDataType>
void testrunImplementation(const std::shared_ptr<Config>& config, const std::unique_ptr<Testruns>& testrun);

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    std::cout << "Usage: ./STCfaimGraph <configuration-file>" << std::endl;
    return RET_ERROR;
  }
  std::cout << "########## faimGraph Static Triangle Counting ##########" << std::endl;
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

  for (const auto& graph : testrun->graphs)
  {
    float time_elapsed_init = 0;
    float time_elapsed_clustering_coefficients = 0;
    int iteration_counter = 0;
    float minTime = 10e9, timing = 0;

    //Setup graph parser and read in file
    std::unique_ptr<GraphParser> parser(new GraphParser(graph));
    if (!parser->parseGraph())
    {
      std::cout << "Error while parsing graph" << std::endl;
      return;
    }

    //------------------------------------------------------------------------------
    // Initialization phase
    //------------------------------------------------------------------------------
    //

    for (int i = 0; i < testrun->params->rounds_; i++)
    {
      std::cout << "Round: " << i + 1 << std::endl;

      start_clock(ce_start, ce_stop);

      std::unique_ptr<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>> faimGraph(std::make_unique<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>>(config, parser));
		faimGraph->initializeMemory(parser);

      time_diff = end_clock(ce_start, ce_stop);
      time_elapsed_init += time_diff;

      //------------------------------------------------------------------------------
      // Clustering Coefficient computation phase
      //------------------------------------------------------------------------------
      //
      std::unique_ptr<ClusteringCoefficients<VertexDataType, EdgeDataType>> cc(std::make_unique<ClusteringCoefficients<VertexDataType, EdgeDataType>>(faimGraph->memory_manager, STCVariant::BALANCED));
      start_clock(ce_start, ce_stop);

      auto clustering_coefficient = cc->computeClusteringCoefficients(faimGraph->memory_manager);

      time_elapsed_clustering_coefficients += end_clock(ce_start, ce_stop); 

    }
    std::cout << "Time elapsed during initialization:          ";
    std::cout << std::setw(10) << time_elapsed_init / static_cast<float>(testrun->params->rounds_) << " ms" << std::endl;

    std::cout << "Time elapsed during calculation of clustering coefficients: ";
    std::cout << std::setw(10) << time_elapsed_clustering_coefficients / static_cast<float>(testrun->params->rounds_) << " ms" << std::endl;
  }
}