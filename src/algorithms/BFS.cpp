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

#include "BFS.h"

template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename UpdateDataType>
void testrunImplementation(const std::shared_ptr<Config>& config, const std::unique_ptr<Testruns>& testrun);

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    std::cout << "Usage: ./BFS <configuration-file>" << std::endl;
    return RET_ERROR;
  }
  std::cout << "########## faimGraph BFS ##########" << std::endl;
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
    float time_elapsed_init {0};
    float time_elapsed_bfs_basic {0};
    float time_elapsed_bfs_dp {0};
    float time_elapsed_bfs_pre {0};
    float time_elapsed_bfs_class {0};

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

    IndividualTimings basic_timing, dynamic_timing, preprocessing_timing, classification_timing;
    
    for (int i = 0; i < testrun->params->rounds_; i++)
    {
      std::cout << "Round: " << i + 1 << std::endl;
      //------------------------------------------------------------------------------
      // BFS phase
      //------------------------------------------------------------------------------
      //
      BFS<VertexDataType, EdgeDataType> bfs (faimGraph->memory_manager);

      start_clock(ce_start, ce_stop);

      bfs.algBFSBasic(faimGraph->memory_manager, basic_timing);

      time_elapsed_bfs_basic += end_clock(ce_start, ce_stop);

      start_clock(ce_start, ce_stop);

      bfs.algBFSDynamicParalellism(faimGraph->memory_manager, dynamic_timing);

      time_elapsed_bfs_dp += end_clock(ce_start, ce_stop);
      
      start_clock(ce_start, ce_stop);

      bfs.algBFSPreprocessing(faimGraph->memory_manager, preprocessing_timing);

      time_elapsed_bfs_pre += end_clock(ce_start, ce_stop);

      start_clock(ce_start, ce_stop);

      bfs.algBFSClassification(faimGraph->memory_manager, classification_timing);

      time_elapsed_bfs_class += end_clock(ce_start, ce_stop);

    }

    // std::cout << "Time elapsed during initialization:          ";
    // std::cout << std::setw(10) << time_elapsed_init / static_cast<float>(testrun->params->rounds_) << " ms" << std::endl;

    basic_timing /= static_cast<float>(testrun->params->rounds_);
    dynamic_timing /= static_cast<float>(testrun->params->rounds_);
    preprocessing_timing /= static_cast<float>(testrun->params->rounds_);
    classification_timing /= static_cast<float>(testrun->params->rounds_);

    std::cout << "Time elapsed during BFS Basic:          ";
    std::cout << std::setw(10) << time_elapsed_bfs_basic / static_cast<float>(testrun->params->rounds_) << " ms";
    std::cout << " | alloc: " << basic_timing.overall_alloc << " | kernel: " << basic_timing.overall_kernel << " | cpy: " << basic_timing.overall_cpy << std::endl;

    std::cout << "Time elapsed during BFS Dynamic Parallelism:          ";
    std::cout << std::setw(10) << time_elapsed_bfs_dp / static_cast<float>(testrun->params->rounds_) << " ms";
    std::cout << " | alloc: " << dynamic_timing.overall_alloc << " | kernel: " << dynamic_timing.overall_kernel << " | cpy: " << dynamic_timing.overall_cpy << std::endl;

    std::cout << "Time elapsed during BFS PreProcessing:          ";
    std::cout << std::setw(10) << time_elapsed_bfs_pre / static_cast<float>(testrun->params->rounds_) << " ms";
    std::cout << " | alloc: " << preprocessing_timing.overall_alloc << " | kernel: " << preprocessing_timing.overall_kernel << " | cpy: " << preprocessing_timing.overall_cpy << std::endl;

    std::cout << "Time elapsed during BFS Classification:          ";
    std::cout << std::setw(10) << time_elapsed_bfs_class / static_cast<float>(testrun->params->rounds_) << " ms";
    std::cout << " | alloc: " << classification_timing.overall_alloc << " | kernel: " << classification_timing.overall_kernel << " | cpy: " << classification_timing.overall_cpy << std::endl;
  }
  return;
}