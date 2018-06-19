//------------------------------------------------------------------------------
// PR.cu
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
#include "PageRank.h"

template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename UpdateDataType>
void testrunImplementation(const std::shared_ptr<Config>& config, const std::unique_ptr<Testruns>& testrun);

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    std::cout << "Usage: ./PR <configuration-file>" << std::endl;
    return RET_ERROR;
  }
  std::cout << "########## faimGraph PageRank ##########" << std::endl;
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
    float time_elapsed_pagerank_naive = 0;
    float time_elapsed_pagerank_balanced = 0;
    int iterations = 20;

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
    start_clock(ce_start, ce_stop);

    std::unique_ptr<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>> faimGraph(std::make_unique<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>>(config, parser));
	 faimGraph->initializeMemory(parser);

    time_diff = end_clock(ce_start, ce_stop);
    time_elapsed_init += time_diff;
    
    for (int i = 0; i < testrun->params->rounds_; i++)
    {
      //std::cout << "Round: " << i + 1 << std::endl;
      //------------------------------------------------------------------------------
      // PageRank Naive phase
      //------------------------------------------------------------------------------
      //
      std::unique_ptr<PageRank<VertexDataType, EdgeDataType>> pr(std::make_unique<PageRank<VertexDataType, EdgeDataType>>(faimGraph->memory_manager, PageRankVariant::NAIVE));

      start_clock(ce_start, ce_stop);

      // Set up initial page rank, according to Wikipedia this is set to 0.25
      pr->initializePageRankVector(0.25f, faimGraph->memory_manager->next_free_vertex_index);

      for(int i = 0; i < iterations; ++i)
        auto rank_naive = pr->algPageRankNaive(faimGraph->memory_manager);

      time_elapsed_pagerank_naive += end_clock(ce_start, ce_stop);

      //------------------------------------------------------------------------------
      // PageRank Balanced phase
      //------------------------------------------------------------------------------
      //

      pr = std::make_unique<PageRank<VertexDataType, EdgeDataType>>(faimGraph->memory_manager, PageRankVariant::BALANCED);

      start_clock(ce_start, ce_stop);
      // Set up initial page rank, according to Wikipedia this is set to 0.25
      pr->initializePageRankVector(0.25f, faimGraph->memory_manager->next_free_vertex_index);
      
      for(int i = 0; i < iterations; ++i)
        auto rank_balanced = pr->algPageRankBalanced(faimGraph->memory_manager);

      time_elapsed_pagerank_balanced += end_clock(ce_start, ce_stop);
    }

    // std::cout << "Time elapsed during initialization:          ";
    // std::cout << std::setw(10) << time_elapsed_init / static_cast<float>(testrun->params->rounds_) << " ms" << std::endl;

    std::cout << "Time elapsed during page rank naive:          ";
    std::cout << std::setw(10) << time_elapsed_pagerank_naive / static_cast<float>(testrun->params->rounds_ * iterations) << " ms" << std::endl;

    std::cout << "Time elapsed during page rank balanced:          ";
    std::cout << std::setw(10) << time_elapsed_pagerank_balanced / static_cast<float>(testrun->params->rounds_* iterations) << " ms" << std::endl;
  }
  return;
}