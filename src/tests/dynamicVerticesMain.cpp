/*!/------------------------------------------------------------------------------
 * dynamicVerticesMain.cpp
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
void verification(std::unique_ptr<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>>& aimGraph, std::unique_ptr<EdgeUpdateManager<VertexDataType, EdgeDataType, UpdateDataType>>& edge_update_manager, const std::string& outputstring, std::unique_ptr<MemoryManager>& memory_manager, std::unique_ptr<GraphParser>& parser, const std::unique_ptr<Testruns>& testrun, const std::unique_ptr<EdgeUpdateBatch<UpdateDataType>>& edge_updates, int round, int updateround, bool gpuVerification, bool insertion, bool duplicate_check);

template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename UpdateDataType>
void verificationInsertion(std::unique_ptr<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>>& aimGraph, std::unique_ptr<EdgeUpdateManager<VertexDataType, EdgeDataType, UpdateDataType>>& edge_update_manager, std::unique_ptr<VertexUpdateManager<VertexDataType, VertexUpdateType>>& vertex_update_manager, const std::string& outputstring, std::unique_ptr<MemoryManager>& memory_manager, std::unique_ptr<GraphParser>& parser, const std::unique_ptr<Testruns>& testrun, int round, int updateround, bool gpuVerification, bool duplicate_check, VertexMapper<index_t, index_t>& mapper);

template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename UpdateDataType>
void verificationDeletion(std::unique_ptr<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>>& aimGraph, std::unique_ptr<EdgeUpdateManager<VertexDataType, EdgeDataType, UpdateDataType>>& edge_update_manager, std::unique_ptr<VertexUpdateManager<VertexDataType, VertexUpdateType>>& vertex_update_manager, const std::string& outputstring, std::unique_ptr<MemoryManager>& memory_manager, std::unique_ptr<GraphParser>& parser, const std::unique_ptr<Testruns>& testrun, int round, int updateround, bool gpuVerification, bool duplicate_check, VertexMapper<index_t, index_t>& mapper);

int main(int argc, char *argv[])
{
	if(argc != 2)
	{
		std::cout << "Usage: ./mainaimGraph <configuration-file>" << std::endl;
		return RET_ERROR;
	}
  std::cout << "########## aimGraph Demo ##########" << std::endl;

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
  bool duplicateChecking = true;
  bool gpuVerification = true;
  bool writeToFile = false;
  bool duplicate_check = false;

  for (auto batchsize : testrun->batchsizes)
  {

    for (const auto& graph : testrun->graphs)
    {
      // Timing information
      float time_elapsed_init = 0;
      float time_elapsed_vertex_insertion = 0;
      float time_elapsed_vertex_deletion = 0; 

      //Setup graph parser and read in file
      std::unique_ptr<GraphParser> parser(new GraphParser(graph));
      if (!parser->parseGraph())
      {
        std::cout << "Error while parsing graph" << std::endl;
        return;
      }

      std::cout << "Batchsize: " << batchsize << std::endl;

      for (int i = 0; i < testrun->params->rounds_; i++)
      {
        //------------------------------------------------------------------------------
        // Initialization phase
        //------------------------------------------------------------------------------
        //
        // Setup initial Vertex Mapper
        VertexMapper<index_t, index_t> mapper;

        //std::cout << "Round: " << i + 1 << std::endl;
        
        start_clock(ce_start, ce_stop);        

        std::unique_ptr<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>> faimGraph(std::make_unique<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>>(config, parser));
		  faimGraph->initializeMemory(parser);
        time_diff = end_clock(ce_start, ce_stop);
        time_elapsed_init += time_diff;

        // memory_manager->printEssentials("Init");
        // Setup mapper and reserve some memory
        mapper.initialMapperSetup(faimGraph->memory_manager, batchsize);

        for (int j = 0; j < testrun->params->update_rounds_; j++)
        {
          //std::cout << "Update-Round: " << j + 1 << std::endl;

          //------------------------------------------------------------------------------
          // Vertex Insertion phase
          //------------------------------------------------------------------------------
          //

			  faimGraph->vertex_update_manager->generateVertexInsertUpdates(batchsize, (i * testrun->params->rounds_) + j);
			  faimGraph->vertex_update_manager->setupMemory(faimGraph->memory_manager, mapper, VertexUpdateVersion::INSERTION);

          start_clock(ce_start, ce_stop);

			 faimGraph->vertexInsertion(mapper);

          time_diff = end_clock(ce_start, ce_stop);
          time_elapsed_vertex_insertion += time_diff; 

          // memory_manager->printEssentials("Insertion");

          if (testrun->params->verification_)
          {
            verificationInsertion <VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>(faimGraph, faimGraph->edge_update_manager, faimGraph->vertex_update_manager, "Verify Insertion Round", faimGraph->memory_manager, parser, testrun, i, j, gpuVerification, duplicate_check, mapper);
          }

			 faimGraph->vertex_update_manager->integrateInsertionChanges(mapper);

          //------------------------------------------------------------------------------
          // Vertex Deletion phase
          //------------------------------------------------------------------------------
          //

			 faimGraph->vertex_update_manager->generateVertexDeleteUpdates(mapper, batchsize, (i * testrun->params->rounds_) + j, faimGraph->memory_manager->next_free_vertex_index);
			 faimGraph->vertex_update_manager->setupMemory(faimGraph->memory_manager, mapper, VertexUpdateVersion::DELETION);

          start_clock(ce_start, ce_stop);

			 faimGraph->vertexDeletion(mapper);

          time_diff = end_clock(ce_start, ce_stop);
          time_elapsed_vertex_deletion += time_diff;

          // memory_manager->printEssentials("Deletion");

          if (testrun->params->verification_)
          {
            verificationDeletion <VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>(faimGraph, faimGraph->edge_update_manager, faimGraph->vertex_update_manager, "Verify Deletion Round", faimGraph->memory_manager, parser, testrun, i, j, gpuVerification, duplicate_check, mapper);
          }

			 faimGraph->vertex_update_manager->integrateDeletionChanges(mapper);
        }

        // if(i == testrun->params->rounds_ - 1)
        // {
        //   std::cout << "####################################################################" << std::endl;
        //   std::cout << "Insertion: " << aimGraph->vertex_update_manager->time_insertion / static_cast<float>(testrun->params->update_rounds_) << " | Dup_in_batch: " << aimGraph->vertex_update_manager->time_dup_in_batch / static_cast<float>(testrun->params->update_rounds_) << " | Dup_in_graph: " << aimGraph->vertex_update_manager->time_dup_in_graph / static_cast<float>(testrun->params->update_rounds_) << std::endl;
        //   std::cout << "####################################################################" << std::endl;
        //   std::cout << "Deletion: " << aimGraph->vertex_update_manager->time_deletion / static_cast<float>(testrun->params->update_rounds_) << " | Vertex Mentions: " << aimGraph->vertex_update_manager->time_vertex_mentions / static_cast<float>(testrun->params->update_rounds_) << " | Compaction: " << aimGraph->vertex_update_manager->time_compaction / static_cast<float>(testrun->params->update_rounds_) << std::endl;
        //   std::cout << "####################################################################" << std::endl;
        // }

        // Let's retrieve a fresh graph
        parser->getFreshGraph();
      }
      PerformanceData perf_data(time_elapsed_init / static_cast<float>(testrun->params->rounds_),
        time_elapsed_vertex_insertion / static_cast<float>(testrun->params->rounds_ * testrun->params->update_rounds_),
        time_elapsed_vertex_deletion / static_cast<float>(testrun->params->rounds_ * testrun->params->update_rounds_));
      perfData.push_back(perf_data);

      std::cout << "Time elapsed during initialization:   ";
      std::cout << std::setw(10) << perf_data.init_time << " ms" << std::endl;

      std::cout << "Time elapsed during vertex insertion:   ";
      std::cout << std::setw(10) << perf_data.insert_time << " ms" << std::endl;

      std::cout << "Time elapsed during vertex deletion:    ";
      std::cout << std::setw(10) << perf_data.delete_time << " ms" << std::endl;
    }
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
                  const std::unique_ptr<EdgeUpdateBatch<UpdateDataType>>& edge_updates,
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
    edge_update_manager->hostEdgeInsertion(edge_updates, parser);
  }
  else
  {
    edge_update_manager->hostEdgeDeletion(edge_updates, parser);
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
    if (!faimGraph->compareGraphs(parser, verify_graph, memory_manager, duplicate_check))
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

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename UpdateDataType>
void verificationInsertion(std::unique_ptr<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>>& faimGraph,
  std::unique_ptr<EdgeUpdateManager<VertexDataType, EdgeDataType, UpdateDataType>>& edge_update_manager,
  std::unique_ptr<VertexUpdateManager<VertexDataType, VertexUpdateType>>& vertex_update_manager,
  const std::string& outputstring,
  std::unique_ptr<MemoryManager>& memory_manager,
  std::unique_ptr<GraphParser>& parser,
  const std::unique_ptr<Testruns>& testrun,
  int round,
  int updateround,
  bool gpuVerification,
  bool duplicate_check,
  VertexMapper<index_t, index_t>& mapper)
{
  std::cout << "############ " << outputstring << " " << (round * testrun->params->rounds_) + updateround << " ############" << std::endl;
  std::unique_ptr<aimGraphCSR> verify_graph = faimGraph->verifyGraphStructure (memory_manager);
  
  // Update host graph
  vertex_update_manager->hostVertexInsertion(parser, mapper);

  std::string filename;
  if (((round * testrun->params->rounds_) + updateround) < 10)
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

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename UpdateDataType>
void verificationDeletion(std::unique_ptr<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>>& faimGraph,
  std::unique_ptr<EdgeUpdateManager<VertexDataType, EdgeDataType, UpdateDataType>>& edge_update_manager,
  std::unique_ptr<VertexUpdateManager<VertexDataType, VertexUpdateType>>& vertex_update_manager,
  const std::string& outputstring,
  std::unique_ptr<MemoryManager>& memory_manager,
  std::unique_ptr<GraphParser>& parser,
  const std::unique_ptr<Testruns>& testrun,
  int round,
  int updateround,
  bool gpuVerification,
  bool duplicate_check,
  VertexMapper<index_t, index_t>& mapper)
{
  std::cout << "############ " << outputstring << " " << (round * testrun->params->rounds_) + updateround << " ############" << std::endl;
  std::unique_ptr<aimGraphCSR> verify_graph = faimGraph->verifyGraphStructure (memory_manager);
  
  // Update host graph
  vertex_update_manager->hostVertexDeletion(parser, mapper);

  std::string filename;
  if (((round * testrun->params->rounds_) + updateround) < 10)
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
