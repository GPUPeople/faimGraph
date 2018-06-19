//------------------------------------------------------------------------------
// ContinuousTC.cpp
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
#include "EdgeUpdate.h"
#include "ConfigurationParser.h"
#include "CSVWriter.h"

template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename UpdateDataType>
void testrunImplementationUniform(const std::shared_ptr<Config>& config, const std::unique_ptr<Testruns>& testrun);

template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename UpdateDataType>
void testrunImplementationSweep(const std::shared_ptr<Config>& config, const std::unique_ptr<Testruns>& testrun);

template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename UpdateDataType>
void testrunImplementationRandom(const std::shared_ptr<Config>& config, const std::unique_ptr<Testruns>& testrun);

//#define TESTRUN_UNIFORM
#define TESTRUN_SWEEP

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    std::cout << "Usage: ./continuousTC <configuration-file>" << std::endl;
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
        #ifdef TESTRUN_UNIFORM
        testrunImplementationUniform <VertexData, VertexUpdate, EdgeData, EdgeDataUpdate>(config, testrun);
        #elif defined TESTRUN_SWEEP
        testrunImplementationSweep <VertexData, VertexUpdate, EdgeData, EdgeDataUpdate>(config, testrun);
        #else
        testrunImplementationRandom <VertexData, VertexUpdate, EdgeData, EdgeDataUpdate>(config, testrun);
        #endif
      }
      else if (testrun->params->graph_mode_ == ConfigurationParameters::GraphMode::WEIGHT)
      {
        #ifdef TESTRUN_UNIFORM
        testrunImplementationUniform <VertexDataWeight, VertexUpdateWeight, EdgeDataWeight, EdgeDataWeightUpdate>(config, testrun);
        #elif defined TESTRUN_SWEEP
        testrunImplementationSweep <VertexDataWeight, VertexUpdateWeight, EdgeDataWeight, EdgeDataWeightUpdate>(config, testrun);
        #else
        testrunImplementationRandom <VertexDataWeight, VertexUpdateWeight, EdgeDataWeight, EdgeDataWeightUpdate>(config, testrun);
        #endif
      }
      else if (testrun->params->graph_mode_ == ConfigurationParameters::GraphMode::SEMANTIC)
      {
        #ifdef TESTRUN_UNIFORM
        testrunImplementationUniform <VertexDataSemantic, VertexUpdateSemantic, EdgeDataSemantic, EdgeDataSemanticUpdate>(config, testrun);
        #elif defined TESTRUN_SWEEP
        testrunImplementationSweep <VertexDataSemantic, VertexUpdateSemantic, EdgeDataSemantic, EdgeDataSemanticUpdate>(config, testrun);
        #else
        testrunImplementationRandom <VertexDataSemantic, VertexUpdateSemantic, EdgeDataSemantic, EdgeDataSemanticUpdate>(config, testrun);
        #endif
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
        #ifdef TESTRUN_UNIFORM
        testrunImplementationUniform <VertexData, VertexUpdate, EdgeDataSOA, EdgeDataUpdate>(config, testrun);
        #elif defined TESTRUN_SWEEP
        testrunImplementationSweep <VertexData, VertexUpdate, EdgeDataSOA, EdgeDataUpdate>(config, testrun);
        #else
        testrunImplementationRandom <VertexData, VertexUpdate, EdgeDataSOA, EdgeDataUpdate>(config, testrun);
        #endif
      }
      else if (testrun->params->graph_mode_ == ConfigurationParameters::GraphMode::WEIGHT)
      {
        #ifdef TESTRUN_UNIFORM
        testrunImplementationUniform <VertexDataWeight, VertexUpdateWeight, EdgeDataWeightSOA, EdgeDataWeightUpdate>(config, testrun);
        #elif defined TESTRUN_SWEEP
        testrunImplementationSweep <VertexDataWeight, VertexUpdateWeight, EdgeDataWeightSOA, EdgeDataWeightUpdate>(config, testrun);
        #else
        testrunImplementationRandom <VertexDataWeight, VertexUpdateWeight, EdgeDataWeightSOA, EdgeDataWeightUpdate>(config, testrun);
        #endif
      }
      else if (testrun->params->graph_mode_ == ConfigurationParameters::GraphMode::SEMANTIC)
      {
        #ifdef TESTRUN_UNIFORM
        testrunImplementationUniform <VertexDataSemantic, VertexUpdateSemantic, EdgeDataSemanticSOA, EdgeDataSemanticUpdate>(config, testrun);
        #elif defined TESTRUN_SWEEP
        testrunImplementationSweep <VertexDataSemantic, VertexUpdateSemantic, EdgeDataSemanticSOA, EdgeDataSemanticUpdate>(config, testrun);
        #else
        testrunImplementationRandom <VertexDataSemantic, VertexUpdateSemantic, EdgeDataSemanticSOA, EdgeDataSemanticUpdate>(config, testrun);
        #endif
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
void testrunImplementationUniform(const std::shared_ptr<Config>& config, const std::unique_ptr<Testruns>& testrun)
{
  std::cout << "Uniform Testrun" << std::endl;
  // Timing
  cudaEvent_t ce_start, ce_stop;
  float time_diff;
  std::vector<PerformanceData> perfData;

  // Global Properties
  bool realisticDeletion = true;
  bool gpuVerification = true;
  bool duplicate_check = true;

  for (auto batchsize : testrun->batchsizes)
  {
    for (const auto& graph : testrun->graphs)
    {
      // Timing information
      float time_elapsed_init = 0;
      float time_elapsed_edgeinsertion = 0;
      float time_elapsed_edgedeletion = 0;

      std::string file_name = "memory/" + graph.substr(10, graph.length() - 10);
#ifdef QUEUING
      file_name += "_queuing";
#endif
      std::cout << "Filename is: " << file_name << std::endl;
      /*CSVWriter csv_writer(file_name);*/

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

        std::cout << "Round: " << i + 1 << std::endl;

        start_clock(ce_start, ce_stop);

        std::unique_ptr<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>> aimGraph(std::make_unique<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>>(config, parser));
        aimGraph->initializeMemory(parser);

        time_diff = end_clock(ce_start, ce_stop);
        time_elapsed_init += time_diff;


        /*csv_writer.writePageAllocationHeader(aimGraph->config, aimGraph->memory_manager->next_free_vertex_index);*/

        for (int j = 0; j < testrun->params->update_rounds_; j++)
        {
          /*std::vector<vertex_t> pages_in_memory(aimGraph->memory_manager->next_free_vertex_index);
          HANDLE_ERROR(cudaMemcpy(pages_in_memory.data(),
                                  d_pages_in_memory,
                                  sizeof(vertex_t) * aimGraph->memory_manager->next_free_vertex_index,
                                  cudaMemcpyDeviceToHost));

          csv_writer.writePageAllocationLine(aimGraph->config, pages_in_memory, aimGraph->memory_manager->next_free_page - aimGraph->memory_manager->d_page_queue.count_, aimGraph->memory_manager->next_free_vertex_index);*/

          //std::cout << "Update-Round: " << j + 1 << std::endl;
          
          //------------------------------------------------------------------------------
          // Edge Insertion phase
          //------------------------------------------------------------------------------
          //
          auto edge_updates = aimGraph->edge_update_manager->generateEdgeUpdates(parser->getNumberOfVertices(), batchsize, (i * testrun->params->rounds_) + j, 0, 0);
          aimGraph->edge_update_manager->receiveEdgeUpdates(std::move(edge_updates), EdgeUpdateVersion::GENERAL);

          start_clock(ce_start, ce_stop);

          aimGraph->edgeInsertion();

          time_diff = end_clock(ce_start, ce_stop);
          time_elapsed_edgeinsertion += time_diff;

          //------------------------------------------------------------------------------
          // Edge Deletion phase
          //------------------------------------------------------------------------------
          //
          std::unique_ptr<EdgeUpdateBatch<UpdateDataType>> realistic_edge_updates;
          if (realisticDeletion)
          {
            // Generate Edge deletion updates randomly from graph data
            realistic_edge_updates = aimGraph->edge_update_manager->generateEdgeUpdates(aimGraph->memory_manager, batchsize, (i * testrun->params->rounds_) + j);
            aimGraph->edge_update_manager->receiveEdgeUpdates(std::move(realistic_edge_updates), EdgeUpdateVersion::GENERAL);
          }

          start_clock(ce_start, ce_stop);

          aimGraph->edgeDeletion();

          time_diff = end_clock(ce_start, ce_stop);
          time_elapsed_edgedeletion += time_diff;
        }

        // Get stats
        TemporaryMemoryAccessHeap temp_memory_dispenser(aimGraph->memory_manager.get(), aimGraph->memory_manager->next_free_vertex_index, sizeof(VertexDataType));
        vertex_t* d_pages_in_memory = temp_memory_dispenser.getTemporaryMemory<vertex_t>(aimGraph->memory_manager->next_free_vertex_index + 1);
        vertex_t* d_accumulated_pages_in_memory = temp_memory_dispenser.getTemporaryMemory<vertex_t>(aimGraph->memory_manager->next_free_vertex_index + 1);
        auto pages_in_memory = aimGraph->memory_manager->template numberPagesInMemory<VertexDataType>(d_pages_in_memory, d_accumulated_pages_in_memory);
        double vertex_mem_used = static_cast<double>(aimGraph->memory_manager->next_free_vertex_index * sizeof(VertexDataType)) / (MEGABYTE);
        double page_mem_used = (static_cast<double>(pages_in_memory * aimGraph->memory_manager->page_size) / (MEGABYTE));
        std::cout << "Vertex Memory used: " << vertex_mem_used << std::endl;
        std::cout << "Edge Memory used: " << page_mem_used << std::endl;
        std::cout << "Total Memory used: " << vertex_mem_used + page_mem_used << std::endl;
        std::cout << "#################################################################################" << std::endl;

        // Let's retrieve a fresh graph
        parser->getFreshGraph();
      }
    }
  }
  // Increment the testrun index
  config->testrun_index_++;
}

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename UpdateDataType>
void testrunImplementationSweep(const std::shared_ptr<Config>& config, const std::unique_ptr<Testruns>& testrun)
{
  std::cout << "Sweep Testrun" << std::endl;
  // Timing
  cudaEvent_t ce_start, ce_stop;
  float time_diff;
  std::vector<PerformanceData> perfData;

  // Global Properties
  bool realisticDeletion = false;
  bool gpuVerification = true;
  bool duplicate_check = true;
  int offset = 0;
  int range = 0;

  for (auto batchsize : testrun->batchsizes)
  {
    for (const auto& graph : testrun->graphs)
    {
      // Timing information
      float time_elapsed_init = 0;
      float time_elapsed_edgeinsertion = 0;
      float time_elapsed_edgedeletion = 0;

      std::string file_name = "memory/" + graph.substr(10, graph.length() - 10);
#ifdef QUEUING
      file_name += "_queuing";
#endif
      std::cout << "Filename is: " << file_name << std::endl;
      /*CSVWriter csv_writer(file_name);*/

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

        std::cout << "Round: " << i + 1 << std::endl;

        start_clock(ce_start, ce_stop);

        std::unique_ptr<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>> faimGraph(std::make_unique<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>>(config, parser));
		  faimGraph->initializeMemory(parser);

        time_diff = end_clock(ce_start, ce_stop);
        time_elapsed_init += time_diff;

        //range = aimGraph->memory_manager->next_free_vertex_index / (testrun->params->update_rounds_ + 10);
        range = 100;
        std::cout << "Range is: " << range << std::endl;
        std::cout << "Highest Page is: " << faimGraph->memory_manager->start_index << std::endl;
        TemporaryMemoryAccessStack temp_memory_dispenser(faimGraph->memory_manager.get(), faimGraph->memory_manager->d_stack_pointer);
        vertex_t* d_pages_in_memory = temp_memory_dispenser.getTemporaryMemory<vertex_t>(faimGraph->memory_manager->next_free_vertex_index + 1);
        vertex_t* d_accumulated_pages_in_memory = temp_memory_dispenser.getTemporaryMemory<vertex_t>(faimGraph->memory_manager->next_free_vertex_index + 1);
        auto pages_in_memory = faimGraph->memory_manager->template numberPagesInMemory<VertexDataType>(d_pages_in_memory, d_accumulated_pages_in_memory);

        /*csv_writer.writePageAllocationHeader(aimGraph->config, aimGraph->memory_manager->next_free_vertex_index);*/


        for (int j = 0; j < faimGraph->memory_manager->next_free_vertex_index / range && pages_in_memory < faimGraph->memory_manager->start_index; j++, offset += range)
        {
          /*std::vector<vertex_t> pages_in_memory(aimGraph->memory_manager->next_free_vertex_index);
          HANDLE_ERROR(cudaMemcpy(pages_in_memory.data(),
                                  d_pages_in_memory,
                                  sizeof(vertex_t) * aimGraph->memory_manager->next_free_vertex_index,
                                  cudaMemcpyDeviceToHost));

          csv_writer.writePageAllocationLine(aimGraph->config, pages_in_memory, aimGraph->memory_manager->next_free_page - aimGraph->memory_manager->d_page_queue.count_, aimGraph->memory_manager->next_free_vertex_index);*/

          std::cout << "Update-Round: " << j + 1 << std::endl;
          
          //------------------------------------------------------------------------------
          // Edge Insertion phase
          //------------------------------------------------------------------------------
          //
          auto edge_updates = faimGraph->edge_update_manager->generateEdgeUpdates(parser->getNumberOfVertices(), batchsize, (i * testrun->params->rounds_) + j, range, offset);
			 faimGraph->edge_update_manager->receiveEdgeUpdates(std::move(edge_updates), EdgeUpdateVersion::GENERAL);

          start_clock(ce_start, ce_stop);

			 faimGraph->edgeInsertion();

          time_diff = end_clock(ce_start, ce_stop);
          time_elapsed_edgeinsertion += time_diff;

          //------------------------------------------------------------------------------
          // Edge Deletion phase
          //------------------------------------------------------------------------------
          //
          std::unique_ptr<EdgeUpdateBatch<UpdateDataType>> realistic_edge_updates;
          if (realisticDeletion)
          {
            // Generate Edge deletion updates randomly from graph data
            realistic_edge_updates = faimGraph->edge_update_manager->generateEdgeUpdates(faimGraph->memory_manager, batchsize, (i * testrun->params->rounds_) + j, 0, 0);
				faimGraph->edge_update_manager->receiveEdgeUpdates(std::move(realistic_edge_updates), EdgeUpdateVersion::GENERAL);
          }

          // Get stats
          pages_in_memory = faimGraph->memory_manager->template numberPagesInMemory<VertexDataType>(d_pages_in_memory, d_accumulated_pages_in_memory);
          double vertex_mem_used = static_cast<double>(faimGraph->memory_manager->next_free_vertex_index * sizeof(VertexDataType)) / (MEGABYTE);
          double page_mem_used = (static_cast<double>(pages_in_memory * faimGraph->memory_manager->page_size) / (MEGABYTE));
          std::cout << "Total Memory used: " << vertex_mem_used + page_mem_used << " and pages available: " << faimGraph->memory_manager->start_index - pages_in_memory <<std::endl;
          std::cout << "#################################################################################" << std::endl;

          start_clock(ce_start, ce_stop);
          
			 faimGraph->edgeDeletion();

          time_diff = end_clock(ce_start, ce_stop);
          time_elapsed_edgedeletion += time_diff;
        }

        

        // Let's retrieve a fresh graph
        parser->getFreshGraph();
      }

    }
  }
  // Increment the testrun index
  config->testrun_index_++;
}

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename UpdateDataType>
void testrunImplementationRandom(const std::shared_ptr<Config>& config, const std::unique_ptr<Testruns>& testrun)
{
  std::cout << "Random Testrun" << std::endl;

  // Global Properties
  bool realisticDeletion = true;
  bool gpuVerification = true;
  bool duplicate_check = true;

  for (auto batchsize : testrun->batchsizes)
  {
    for (const auto& graph : testrun->graphs)
    {
      std::string file_name = "memory/" + graph.substr(10, graph.length() - 10);
#ifdef QUEUING
      file_name += "_queuing";
#endif
      std::cout << "Filename is: " << file_name << std::endl;
      /*CSVWriter csv_writer(file_name);*/

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

        std::cout << "Round: " << i + 1 << std::endl;


        std::unique_ptr<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>> faimGraph(std::make_unique<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>>(config, parser));
		  faimGraph->initializeMemory(parser);

        /*csv_writer.writePageAllocationHeader(aimGraph->config, aimGraph->memory_manager->next_free_vertex_index);*/

        for (int j = 0; j < testrun->params->update_rounds_; j++)
        {
          srand(j + 1);
          // Get stats
          /*std::vector<vertex_t> pages_in_memory(aimGraph->memory_manager->next_free_vertex_index);
          HANDLE_ERROR(cudaMemcpy(pages_in_memory.data(),
                                  d_pages_in_memory,
                                  sizeof(vertex_t) * aimGraph->memory_manager->next_free_vertex_index,
                                  cudaMemcpyDeviceToHost));

          csv_writer.writePageAllocationLine(aimGraph->config, pages_in_memory, aimGraph->memory_manager->next_free_page - aimGraph->memory_manager->d_page_queue.count_, aimGraph->memory_manager->next_free_vertex_index);*/

          //std::cout << "Update-Round: " << j + 1 << std::endl;

          if(rand() % 2)
          {
            // Insertion
            //std::cout << "Insertion" << std::endl;
            auto edge_updates = faimGraph->edge_update_manager->generateEdgeUpdates(parser->getNumberOfVertices(), batchsize, (i * testrun->params->rounds_) + j, 0, 0);
				faimGraph->edge_update_manager->receiveEdgeUpdates(std::move(edge_updates), EdgeUpdateVersion::GENERAL);
				faimGraph->edgeInsertion ();
          }
          else
          {
            // Deletion
            //std::cout << "Deletion" << std::endl;
            auto edge_updates = faimGraph->edge_update_manager->generateEdgeUpdates(faimGraph->memory_manager, batchsize, (i * testrun->params->rounds_) + j, 0, 0);
				faimGraph->edge_update_manager->receiveEdgeUpdates(std::move(edge_updates), EdgeUpdateVersion::GENERAL);
				faimGraph->edgeDeletion();
          }
        }
        TemporaryMemoryAccessHeap temp_memory_dispenser(faimGraph->memory_manager.get(), faimGraph->memory_manager->next_free_vertex_index, sizeof(VertexDataType));
        vertex_t* d_pages_in_memory = temp_memory_dispenser.getTemporaryMemory<vertex_t>(faimGraph->memory_manager->next_free_vertex_index + 1);
        vertex_t* d_accumulated_pages_in_memory = temp_memory_dispenser.getTemporaryMemory<vertex_t>(faimGraph->memory_manager->next_free_vertex_index + 1);
        auto pages_in_memory = faimGraph->memory_manager->template numberPagesInMemory<VertexDataType>(d_pages_in_memory, d_accumulated_pages_in_memory);
        double vertex_mem_used = static_cast<double>(faimGraph->memory_manager->next_free_vertex_index * sizeof(VertexDataType)) / (MEGABYTE);
        double page_mem_used = (static_cast<double>(pages_in_memory * faimGraph->memory_manager->page_size) / (MEGABYTE));
        std::cout << "Vertex Memory used: " << vertex_mem_used << std::endl;
        std::cout << "Edge Memory used: " << page_mem_used << std::endl;
        std::cout << "Total Memory used: " << vertex_mem_used + page_mem_used << std::endl;
        std::cout << "#################################################################################" << std::endl;

        // Let's retrieve a fresh graph
        parser->getFreshGraph();
      }
    }
  }
  // Increment the testrun index
  config->testrun_index_++;
}