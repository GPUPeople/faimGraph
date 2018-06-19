/*!/------------------------------------------------------------------------------
 * concurrentTC.cpp
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

template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename UpdateDataType>
void verification(std::unique_ptr<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>>& aimGraph, std::unique_ptr<EdgeUpdateManager<VertexDataType, EdgeDataType, UpdateDataType>>& edge_update_manager, const std::string& outputstring, std::unique_ptr<MemoryManager>& memory_manager, std::unique_ptr<GraphParser>& parser, const std::unique_ptr<Testruns>& testrun, int round, int updateround, bool gpuVerification, bool insertion, bool duplicate_check);

int main(int argc, char *argv[])
{
	if(argc != 2)
	{
		std::cout << "Usage: ./concurrentTC <configuration-file>" << std::endl;
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
  bool duplicate_check = false;

  int vertex_batch_size = 100;
  int max_adjacency_size = 10;

  for (auto batchsize : testrun->batchsizes)
  {
    for (const auto& graph : testrun->graphs)
    {
      // Timing information
      float time_elapsed_init = 0;
      float time_elapsed_edgeupdate = 0;
      bool streamVariant = false;

      //Setup graph parser and read in file
      std::unique_ptr<GraphParser> parser(new GraphParser(graph));
      if (!parser->parseGraph())
      {
        std::cout << "Error while parsing graph" << std::endl;
        return;
      }

      for (int i = 0; i < testrun->params->rounds_; i++)
      {
        //------------------------------------------------------------------------------
        // Initialization phase
        //------------------------------------------------------------------------------
        //

        std::cout << "Round: " << i + 1 << " with batchsize " << batchsize << std::endl;
        
        start_clock(ce_start, ce_stop);        

        std::unique_ptr<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>> faimGraph(std::make_unique<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>>(config, parser));
		  faimGraph->initializeMemory(parser);
        time_diff = end_clock(ce_start, ce_stop);
        time_elapsed_init += time_diff;

        if(streamVariant)
        {
          //------------------------------------------------------------------------------
          // Stream approach with multiple kernels
          //------------------------------------------------------------------------------
          //
          // Setup two streams
          cudaStream_t insertion_stream;
          cudaStream_t deletion_stream;
          HANDLE_ERROR(cudaStreamCreate(&insertion_stream));
          HANDLE_ERROR(cudaStreamCreate(&deletion_stream));

          for (int j = 0; j < testrun->params->update_rounds_; j++)
          {
            std::cout << "Update-Round: " << j + 1 << std::endl;

            // Generate updates
            auto edge_insertion_updates = faimGraph->edge_update_manager->template generateEdgeUpdatesConcurrent<VertexUpdateType>(faimGraph, faimGraph->memory_manager, batchsize, (i * testrun->params->rounds_) + j);
            auto edge_deletion_updates = faimGraph->edge_update_manager->generateEdgeUpdates(faimGraph->memory_manager, batchsize, (i * testrun->params->rounds_) + j);
				faimGraph->edge_update_manager->receiveEdgeUpdates(std::move(edge_insertion_updates), EdgeUpdateVersion::INSERTION);
				faimGraph->edge_update_manager->receiveEdgeUpdates(std::move(edge_deletion_updates), EdgeUpdateVersion::DELETION);

            // We need page-locked memory for a-sync memcpy
				faimGraph->edge_update_manager->hostCudaAllocConcurrentUpdates();

            start_clock(ce_start, ce_stop);
            
				faimGraph->edge_update_manager->deviceEdgeUpdateConcurrentStream(insertion_stream, deletion_stream, faimGraph->memory_manager, faimGraph->config);
            
            time_diff = end_clock(ce_start, ce_stop);
            time_elapsed_edgeupdate += time_diff;

            if (testrun->params->verification_)
            {
              verification <VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>(faimGraph, faimGraph->edge_update_manager, "Verify Round", faimGraph->memory_manager, parser, testrun, i, j, gpuVerification, true, duplicate_check);
            }

            // Free page-locked memory again
				faimGraph->edge_update_manager->hostCudaFreeConcurrentUpdates();
          }

          // Clean up streams
          HANDLE_ERROR(cudaStreamDestroy(insertion_stream));
          HANDLE_ERROR(cudaStreamDestroy(deletion_stream));
        }
        else
        {
          //------------------------------------------------------------------------------
          // Concurrent updates in 1 kernel
          //------------------------------------------------------------------------------
          //
          for (int j = 0; j < testrun->params->update_rounds_; j++)
          {
            //std::cout << "Update-Round: " << j + 1 << std::endl;
            // Generate updates
            auto edge_insertion_updates = faimGraph->edge_update_manager->template generateEdgeUpdatesConcurrent<VertexUpdateType>(faimGraph, faimGraph->memory_manager, batchsize, (i * testrun->params->rounds_) + j);
            auto edge_deletion_updates = faimGraph->edge_update_manager->generateEdgeUpdates(faimGraph->memory_manager, batchsize, (i * testrun->params->rounds_) + j);
				faimGraph->edge_update_manager->receiveEdgeUpdates(std::move(edge_insertion_updates), EdgeUpdateVersion::INSERTION);
				faimGraph->edge_update_manager->receiveEdgeUpdates(std::move(edge_deletion_updates), EdgeUpdateVersion::DELETION);

            start_clock(ce_start, ce_stop);
            
				faimGraph->edge_update_manager->deviceEdgeUpdateConcurrent(faimGraph->memory_manager, faimGraph->config);
            
            time_diff = end_clock(ce_start, ce_stop);
            time_elapsed_edgeupdate += time_diff;

            if (testrun->params->verification_)
            {
              verification <VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>(faimGraph, faimGraph->edge_update_manager, "Verify Round", faimGraph->memory_manager, parser, testrun, i, j, gpuVerification, true, duplicate_check);
            }
          }
        }
        
        // Let's retrieve a fresh graph
        parser->getFreshGraph();
      }

      PerformanceData perf_data(time_elapsed_init / static_cast<float>(testrun->params->rounds_),
        time_elapsed_edgeupdate / static_cast<float>(testrun->params->rounds_ * testrun->params->update_rounds_),
        0.0f);
      perfData.push_back(perf_data);

      // Write Performance Output to Console
      if (testrun->params->performance_output_ == ConfigurationParameters::PerformanceOutput::STDOUT)
      {
        std::cout << "Time elapsed during initialization:   ";
        std::cout << std::setw(10) << perf_data.init_time << " ms" << std::endl;

        std::cout << "Time elapsed during updates:   ";
        std::cout << std::setw(10) << perf_data.insert_time << " ms" << std::endl;
      }
    }
  }
  // Write Performance Output into CSV-File
  if (testrun->params->performance_output_ == ConfigurationParameters::PerformanceOutput::CSV)
  {
    CSVWriter performancewriter;
    performancewriter.writePerformanceMetric("Testrun", config, perfData, 0);
  }
  // Increment the testrun index
  config->testrun_index_++;
}

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename UpdateDataType>
void verification(std::unique_ptr<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>>& faimGraph,
  std::unique_ptr<EdgeUpdateManager<VertexDataType, EdgeDataType, UpdateDataType>>& edge_update_manager,
                  const std::string& outputstring, 
                  std::unique_ptr<MemoryManager>& memory_manager,
                  std::unique_ptr<GraphParser>& parser,
                  const std::unique_ptr<Testruns>& testrun,
                  int round, 
                  int updateround,
                  bool gpuVerification,
                  bool insertion,
                  bool duplicate_check)
{
  std::cout << "############ " << outputstring << " " << (round * testrun->params->rounds_) + updateround << " ############" << std::endl;
  std::unique_ptr<aimGraphCSR> verify_graph = faimGraph->verifyGraphStructure (memory_manager);
  // Update host graph
  edge_update_manager->hostEdgeInsertion(parser);
  edge_update_manager->hostEdgeDeletion(parser);

  std::string filename;
  if(((round * testrun->params->rounds_) + updateround) < 10)
  {
    filename = "../tests/Verification/graphverification" + outputstring + "0" + std::to_string((round * testrun->params->rounds_) + updateround) + ".txt";
  }
  else
  {
    filename = "../tests/Verification/graphverification" + outputstring + std::to_string((round * testrun->params->rounds_) + updateround) + ".txt";
  }
  edge_update_manager->writeGraphsToFile(verify_graph, parser, filename);
  

  // Compare graph structures
  if (gpuVerification)
  {
    if (!faimGraph->compareGraphs(parser, verify_graph, duplicate_check))
    {
      std::cout << "########## Graphs are NOT the same ##########" << std::endl;
      exit(-1);
    }
  }
  else
  {
    edge_update_manager->writeGraphsToFile(verify_graph, parser, "../tests/Verification/graphverification");
  }
}
