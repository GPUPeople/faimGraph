 //------------------------------------------------------------------------------
// StaticTriangleCounting.cpp
//
// faimGraph
//
//------------------------------------------------------------------------------

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
#include "STC.h"

int arrayBlocks[]={16000};
int arrayBlockSize[]={32,64,96,128,192,256};
int arrayThreadPerIntersection[]={1,2,4,8,16,32};
int arrayThreadShift[]={0,1,2,3,4,5};

template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename UpdateDataType>
void testrunImplementation(const std::shared_ptr<Config>& config, const std::unique_ptr<Testruns>& testrun);

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    std::cout << "Usage: ./STC <configuration-file>" << std::endl;
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
    float time_elapsed_trianglecounting = 0;
    float time_elapsed_trianglecounting_balanced = 0;
    float time_elapsed_sorting = 0;
    int iteration_counter = 0;
    float minTime=10e9, timing = 0;

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
      //std::cout << "Round: " << i + 1 << std::endl;

      start_clock(ce_start, ce_stop);

      std::unique_ptr<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>> faimGraph(std::make_unique<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>>(config, parser));
		faimGraph->initializeMemory(parser);

      time_diff = end_clock(ce_start, ce_stop);
      time_elapsed_init += time_diff;

		faimGraph->memory_manager->template sortAdjacency<VertexDataType, EdgeDataType>(config, SortOrder::ASCENDING);
      // aimGraph->memory_manager->template testUndirectedness<VertexDataType, EdgeDataType>(config);
      // aimGraph->memory_manager->template testSelfLoops<VertexDataType, EdgeDataType>(config);
      // aimGraph->memory_manager->template testDuplicates<VertexDataType, EdgeDataType>(config);

      //------------------------------------------------------------------------------
      // Triangle counting phase
      //------------------------------------------------------------------------------
      //
      std::unique_ptr<STC<VertexDataType, EdgeDataType>> stc(std::make_unique<STC<VertexDataType, EdgeDataType>>(faimGraph->memory_manager, STCVariant::BALANCED));
      //stc->host_StaticTriangleCounting(parser);
      start_clock(ce_start, ce_stop);

      auto triangle_count = stc->StaticTriangleCounting(faimGraph->memory_manager, false);

      time_elapsed_trianglecounting += end_clock(ce_start, ce_stop);   
      
      // static int counter = 0;
      // std::string filename = std::string("../tests/Verification/DeviceSTC");
      // filename += std::to_string(counter) + std::string(".txt");
      // std::ofstream file(filename);
      // if (file.is_open())
      // {
      //   for (int i = 0; i < aimGraph->memory_manager->next_free_vertex_index; ++i)
      //   {
      //     file << stc->triangles[i] << "\n";
      //   }
      // }
      // ++counter;

      //------------------------------------------------------------------------------
      // Sorting test
      //------------------------------------------------------------------------------
      //
    }
    // std::cout << "Time elapsed during initialization:          ";
    // std::cout << std::setw(10) << time_elapsed_init / static_cast<float>(testrun->params->rounds_) << " ms" << std::endl;

    std::cout << "Time elapsed during naive triangle counting: ";
    std::cout << std::setw(10) << time_elapsed_trianglecounting / static_cast<float>(testrun->params->rounds_) << " ms" << std::endl;
    std::cout << std::endl;

    // std::cout << "Time elapsed during sorting: ";
    // std::cout << std::setw(10) << time_elapsed_sorting / static_cast<float>(testrun->params->rounds_) << " ms" << std::endl;
  }
}

bool searchTriangle(std::unique_ptr<GraphParser>& parser, uint32_t start_index, uint32_t first_value, uint32_t second_value)
{
  auto& adjacency = parser->getAdjacency();
  auto& offset = parser->getOffset();
  auto number_vertices = parser->getNumberOfVertices();

  auto begin_iter = adjacency.begin() + offset.at(start_index);
  auto end_iter = adjacency.begin() + offset.at(start_index + 1);
  bool first_found = false;
  bool second_found = false;
  while(begin_iter != end_iter)
  {
    if(*begin_iter == first_value)
    {
      first_found = true;
    }
    else if (*begin_iter == second_value)
    {
      second_found = true;
    }
    if(first_found && second_found)
    {
      return true;
    }
    ++begin_iter;
  }
  return false;
}

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType>
uint32_t STC<VertexDataType, EdgeDataType>::host_StaticTriangleCounting(std::unique_ptr<GraphParser>& parser)
{
  uint32_t triangle_count = 0;
  auto& adjacency = parser->getAdjacency();
  auto& offset = parser->getOffset();
  auto number_vertices = parser->getNumberOfVertices();
  std::vector<uint32_t> triangle_count_per_vertex(number_vertices);
  std::fill(triangle_count_per_vertex.begin(), triangle_count_per_vertex.end(), 0);

  //------------------------------------------------------------------------------
  // STANDARD METHOD
  //------------------------------------------------------------------------------

  // for(int i = 0; i < number_vertices; ++i)
  // {
  //   auto begin_iter = adjacency.begin() + offset.at(i);
  //   auto end_iter = adjacency.begin() + offset.at(i + 1);
  //   while(begin_iter != end_iter)
  //   {
  //     // Go over adjacency
  //     // Get value of first element
  //     auto first_value = *begin_iter;
  //     // Setup iterator on next element
  //     auto adjacency_iter = ++begin_iter;
  //     while(adjacency_iter != end_iter)
  //     {
  //       // Go over adjacency and for each element search for the back edge
  //       auto begin_adjacency_iter = adjacency.begin() + offset.at(*adjacency_iter);
  //       auto end_adjacency_iter = adjacency.begin() + offset.at(*adjacency_iter + 1);
  //       while(begin_adjacency_iter != end_adjacency_iter)
  //       {
  //         // Search for the back edge
  //         if(*begin_adjacency_iter == first_value)
  //         {
  //           triangle_count++;
  //           triangle_count_per_vertex.at(i) += 1;
  //           break;
  //         }
  //         ++begin_adjacency_iter;
  //       }
  //       ++adjacency_iter;
  //     }
  //   }
  // }

  //------------------------------------------------------------------------------
  // Smallest index METHOD
  //------------------------------------------------------------------------------

   for(int i = 0; i < number_vertices; ++i)
   {
     auto begin_iter = adjacency.begin() + offset.at(i);
     auto end_iter = adjacency.begin() + offset.at(i + 1);
     while(begin_iter != end_iter)
     {
       // Go over adjacency
       // Get value of first element
       auto first_value = *begin_iter;
       if(first_value < i)
       {
         ++begin_iter;
         continue;
       }
       // Setup iterator on next element
       auto adjacency_iter = ++begin_iter;
       while(adjacency_iter != end_iter)
       {
         auto second_value = *adjacency_iter;
         if(second_value < i)
         {
           ++adjacency_iter;
           continue;
         }
         // Go over adjacency and for each element search for the back edge
         auto begin_adjacency_iter = adjacency.begin() + offset.at(*adjacency_iter);
         auto end_adjacency_iter = adjacency.begin() + offset.at(*adjacency_iter + 1);
         while(begin_adjacency_iter != end_adjacency_iter)
         {
           // Search for the back edge
           if(*begin_adjacency_iter == first_value)
           {
             triangle_count += 3;
             triangle_count_per_vertex.at(i) += 1;
             triangle_count_per_vertex.at(first_value) += 1;
             triangle_count_per_vertex.at(second_value) += 1;
            //  if(i == 6977 || first_value == 6977 || second_value == 6977)
            //  {
            //    std::cout << "Triangle: < " << i << ", " << first_value << ", " << second_value << " >" << std::endl;
            //  }
             break;
           }
           ++begin_adjacency_iter;
         }
         ++adjacency_iter;
       }
     }
   }

  //------------------------------------------------------------------------------
  // Smallest index verification METHOD
  //------------------------------------------------------------------------------

  //for(int i = 0; i < number_vertices; ++i)
  //{
  //  auto begin_iter = adjacency.begin() + offset.at(i);
  //  auto end_iter = adjacency.begin() + offset.at(i + 1);
  //  while(begin_iter != end_iter)
  //  {
  //    // Go over adjacency
  //    // Get value of first element
  //    auto first_value = *begin_iter;
  //    if(first_value < i)
  //    {
  //      ++begin_iter;
  //      continue;
  //    }
  //    // Setup iterator on next element
  //    auto adjacency_iter = ++begin_iter;
  //    while(adjacency_iter != end_iter)
  //    {
  //      auto second_value = *adjacency_iter;
  //      if(second_value < i)
  //      {
  //        ++adjacency_iter;
  //        continue;
  //      }
  //      // Go over adjacency and for each element search for the back edge
  //      auto begin_adjacency_iter = adjacency.begin() + offset.at(*adjacency_iter);
  //      auto end_adjacency_iter = adjacency.begin() + offset.at(*adjacency_iter + 1);
  //      while(begin_adjacency_iter != end_adjacency_iter)
  //      {
  //        // Search for the back edge
  //        if(*begin_adjacency_iter == first_value)
  //        {
  //          triangle_count += 3;
  //          triangle_count_per_vertex.at(i) += 1;
  //          triangle_count_per_vertex.at(first_value) += 1;
  //          triangle_count_per_vertex.at(second_value) += 1;
  //          if(not searchTriangle(parser, first_value, i, second_value))
  //          {
  //            std::cout << "Triangle should be " << i << " | " << first_value << " | " << second_value << " , did not find: " << first_value << " | " << i << " | " << second_value << std::endl;
  //          }
  //          if(not searchTriangle(parser, second_value, i, first_value))
  //          {
  //            std::cout << "Triangle should be " << i << " | " << first_value << " | " << second_value << " , did not find: " << second_value << " | " << i << " | " << first_value << std::endl;
  //          }
  //          break;
  //        }
  //        ++begin_adjacency_iter;
  //      }
  //      ++adjacency_iter;
  //    }
  //  }
  //}

  // Write data to file to verify
  static int counter = 0;
  std::string filename = std::string("../tests/Verification/VerifySTC");
  filename += std::to_string(counter) + std::string(".txt");
  std::ofstream file(filename);
  if (file.is_open())
  {
    for (const auto& element : triangle_count_per_vertex)
    {
      file << element << "\n";
    }
  }
  ++counter;

  std::cout << "Triangle count is " << triangle_count << std::endl;
  return 0;
}