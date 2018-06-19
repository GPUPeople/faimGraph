/*!/------------------------------------------------------------------------------
 * Main.cpp
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
void verification(std::unique_ptr<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>>& faimGraph, std::unique_ptr<EdgeUpdateManager<VertexDataType, EdgeDataType, UpdateDataType>>& edge_update_manager, const std::string& outputstring, std::unique_ptr<MemoryManager>& memory_manager, std::unique_ptr<GraphParser>& parser, const std::unique_ptr<Testruns>& testrun, int round, int updateround, bool gpuVerification, bool insertion, bool duplicate_check);

int main(int argc, char *argv[])
{
	if(argc != 2)
	{
		std::cout << "Usage: ./mainfaimGraph <configuration-file>" << std::endl;
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
  unsigned int range = 0;
  unsigned int offset = 0;

  int vertex_batch_size = 100;
  int max_adjacency_size = 10;

  for (auto batchsize : testrun->batchsizes)
  {
	  if (batchsize > MAXIMAL_BATCH_SIZE)
		  return;
    std::cout << "Batchsize: " << batchsize << std::endl;
    for (const auto& graph : testrun->graphs)
    {
      // Timing information
      float time_elapsed_init = 0;
      float time_elapsed_edgeinsertion = 0;
      float time_elapsed_edgedeletion = 0; 
      int warmup_rounds = 0;

      //Setup graph parser and read in file
      std::unique_ptr<GraphParser> parser(new GraphParser(graph));
      if (!parser->parseGraph())
      {
        std::cout << "Error while parsing graph" << std::endl;
        return;
      }

      for (int i = 0; i < testrun->params->rounds_ + warmup_rounds; i++, offset += range)
      {
        //------------------------------------------------------------------------------
        // Initialization phase
        //------------------------------------------------------------------------------
        //

        std::cout << "Round: " << i + 1 << std::endl;
        
        std::unique_ptr<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>> faimGraph(std::make_unique<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>>(config, parser));
        //HANDLE_ERROR(cudaMalloc((void **)&(aimGraph->memory_manager->d_memory), static_cast<uint64_t>(GIGABYTE * config->device_mem_size_)));
        //printCUDAStats("Allocation: ");
        
        

        start_clock(ce_start, ce_stop); 
        
		  faimGraph->initializeMemory(parser);
        
		  time_diff = end_clock(ce_start, ce_stop);
        if(i >= warmup_rounds)
          time_elapsed_init += time_diff;        

        //aimGraph->memory_manager->template sortAdjacency<VertexDataType, EdgeDataType>(config, SortOrder::ASCENDING);

        // if(i == 0)
        //   aimGraph->memory_manager->estimateInitialStorageRequirements(parser->getNumberOfVertices(), parser->getNumberOfEdges(), batchsize, sizeof(UpdateDataType));

        if (testrun->params->verification_)
        {
          std::unique_ptr<aimGraphCSR> verify_graph = faimGraph->verifyGraphStructure(faimGraph->memory_manager);
          if(writeToFile)
          {
				 faimGraph->edge_update_manager->writeGraphsToFile(verify_graph, parser, "../tests/Verification/StartGraph.txt");
          }
        }

        for (int j = 0; j < testrun->params->update_rounds_; j++)
        {
          // std::cout << "Update-Round: " << j + 1 << std::endl;
          //------------------------------------------------------------------------------
          // Edge Insertion phase
          //------------------------------------------------------------------------------
          //
          //auto edge_updates = aimGraph->edge_update_manager->generateEdgeUpdates(parser->getNumberOfVertices(), batchsize, (i * testrun->params->rounds_) + j);
          auto edge_updates = faimGraph->edge_update_manager->generateEdgeUpdates(parser->getNumberOfVertices(), batchsize, (i * testrun->params->rounds_) + j, range, offset);
			 faimGraph->edge_update_manager->receiveEdgeUpdates(std::move(edge_updates), EdgeUpdateVersion::GENERAL);

          //------------------------------------------------------------------------------
          // Start Timer for Edge Insertion
          //------------------------------------------------------------------------------
          //
          start_clock(ce_start, ce_stop);
          
			 faimGraph->edgeInsertion();
          
          time_diff = end_clock(ce_start, ce_stop);
          if(i >= warmup_rounds)
            time_elapsed_edgeinsertion += time_diff;

          if (testrun->params->verification_)
          {
            if(writeToFile)
            {
              std::string filename;
              if(((i * testrun->params->rounds_) + j) < 10)
              {
                filename = "../tests/Verification/edgeupdatesinsert0" + std::to_string((i * testrun->params->rounds_) + j) + ".txt";
              }
              else
              {
                filename = "../tests/Verification/edgeupdatesinsert" + std::to_string((i * testrun->params->rounds_) + j) + ".txt";
              }
				  faimGraph->edge_update_manager->writeEdgeUpdatesToFile(edge_updates, batchsize, filename);
            }

            verification <VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>(faimGraph, faimGraph->edge_update_manager, "Verify Insertion Round", faimGraph->memory_manager, parser, testrun, i, j, gpuVerification, true, duplicate_check);
          }            

          //------------------------------------------------------------------------------
          // Edge Deletion phase
          //------------------------------------------------------------------------------
          //
          std::unique_ptr<EdgeUpdateBatch<UpdateDataType>> realistic_edge_updates;
          if(realisticDeletion)
          {
            // Generate Edge deletion updates randomly from graph data
            //realistic_edge_updates = aimGraph->edge_update_manager->generateEdgeUpdates(aimGraph->memory_manager, batchsize, (i * testrun->params->rounds_) + j);
            realistic_edge_updates = faimGraph->edge_update_manager->generateEdgeUpdates(faimGraph->memory_manager, batchsize, (i * testrun->params->rounds_) + j, range, offset);
				faimGraph->edge_update_manager->receiveEdgeUpdates(std::move(realistic_edge_updates), EdgeUpdateVersion::GENERAL);
          }
          
          start_clock(ce_start, ce_stop);
          
			 faimGraph->edgeDeletion();
          
          time_diff = end_clock(ce_start, ce_stop);
          if(i >= warmup_rounds)
            time_elapsed_edgedeletion += time_diff;

          //------------------------------------------------------------------------------
          // Verification phase
          //------------------------------------------------------------------------------
          //
          if (testrun->params->verification_)
          {
            if (realisticDeletion)
            {
              if(writeToFile)
              {
                std::string filename;
                if(((i * testrun->params->rounds_) + j) < 10)
                {
                  filename = "../tests/Verification/edgeupdatesdelete0" + std::to_string((i * testrun->params->rounds_) + j) + ".txt";
                }
                else
                {
                  filename = "../tests/Verification/edgeupdatesdelete" + std::to_string((i * testrun->params->rounds_) + j) + ".txt";
                }
					 faimGraph->edge_update_manager->writeEdgeUpdatesToFile(realistic_edge_updates, batchsize, filename);
              }

              verification <VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>(faimGraph, faimGraph->edge_update_manager, "Verify Deletion Round", faimGraph->memory_manager, parser, testrun, i, j, gpuVerification, false, duplicate_check);
            }
            else
            {
              verification <VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>(faimGraph, faimGraph->edge_update_manager, "Verify Deletion Round", faimGraph->memory_manager, parser, testrun, i, j, gpuVerification, false, duplicate_check);
            }
          }
        }

        // Let's retrieve a fresh graph
        parser->getFreshGraph();
      }

      PerformanceData perf_data(time_elapsed_init / static_cast<float>(testrun->params->rounds_),
        time_elapsed_edgeinsertion / static_cast<float>(testrun->params->rounds_ * testrun->params->update_rounds_),
        time_elapsed_edgedeletion / static_cast<float>(testrun->params->rounds_ * testrun->params->update_rounds_));
      perfData.push_back(perf_data);

      // Write Performance Output to Console
      if (testrun->params->performance_output_ == ConfigurationParameters::PerformanceOutput::STDOUT)
      {
        std::cout << "Time elapsed during initialization:   ";
        std::cout << std::setw(10) << perf_data.init_time << " ms" << std::endl;

        std::cout << "Time elapsed during edge insertion:   ";
        std::cout << std::setw(10) << perf_data.insert_time << " ms" << std::endl;

        std::cout << "Time elapsed during edge deletion:    ";
        std::cout << std::setw(10) << perf_data.delete_time << " ms" << std::endl;
      }
    }
    std::cout << "################################################################" << std::endl;
    std::cout << "################################################################" << std::endl;
    std::cout << "################################################################" << std::endl;
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
  if (insertion)
  {
    edge_update_manager->hostEdgeInsertion(parser);
  }
  else
  {
    edge_update_manager->hostEdgeDeletion(parser);
  }

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
