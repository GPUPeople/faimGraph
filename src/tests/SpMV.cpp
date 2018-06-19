//------------------------------------------------------------------------------
// SpMV.cpp
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
#include <fstream>
#include <sstream>
#include <algorithm>

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
#include "SpMV.h"
#include "SpMM.h"

template <typename VertexDataType, typename EdgeDataType>
void testrunImplementationMV(const std::shared_ptr<Config>& config, const std::unique_ptr<Testruns>& testrun);

template <typename VertexDataType, typename EdgeDataType>
void testrunImplementationMM(const std::shared_ptr<Config>& config, const std::unique_ptr<Testruns>& testrun);

template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename UpdateDataType>
void verification(std::unique_ptr<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>>& aimGraph, const std::string& outputstring, std::unique_ptr<GraphParser>& parser, const std::unique_ptr<Testruns>& testrun, int round, int updateround, bool duplicate_check);

template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename UpdateDataType>
void verificationMMMultiplication(std::unique_ptr<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>>& aimGraph, const std::string& outputstring, std::unique_ptr<GraphParser>& parser, const std::unique_ptr<Testruns>& testrun, int round, int updateround, bool duplicate_check, std::unique_ptr<SpMMManager>& spmm);


//#define SPMM

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    std::cout << "Usage: ./spmv <configuration-file>" << std::endl;
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
#ifdef SPMM
      testrunImplementationMV <VertexData, EdgeDataMatrix>(config, testrun);
#else
      testrunImplementationMM <VertexData, EdgeDataMatrix>(config, testrun);
#endif
    }
    else if (testrun->params->memory_layout_ == ConfigurationParameters::MemoryLayout::SOA)
    {
#ifdef SPMM
      testrunImplementationMV <VertexData, EdgeDataMatrixSOA>(config, testrun);
#else
      testrunImplementationMM <VertexData, EdgeDataMatrixSOA>(config, testrun);
#endif
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
template <typename VertexDataType, typename EdgeDataType>
void testrunImplementationMV(const std::shared_ptr<Config>& config, const std::unique_ptr<Testruns>& testrun)
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
      float time_elapsed_multiplication = 0;
      float time_elapsed_transposeCSR = 0;
      float time_elapsed_transpose = 0;

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

        std::unique_ptr<faimGraph<VertexDataType, VertexUpdate, EdgeDataType, EdgeDataUpdate>> faimGraph(std::make_unique<faimGraph<VertexDataType, VertexUpdate, EdgeDataType, EdgeDataUpdate>>(config, parser));
        std::unique_ptr<SpMVManager> spmv(std::make_unique<SpMVManager>(faimGraph->memory_manager->next_free_vertex_index, faimGraph->memory_manager->next_free_vertex_index));

        start_clock(ce_start, ce_stop);

		  faimGraph->initializeMemory(parser);

        time_diff = end_clock(ce_start, ce_stop);
        time_elapsed_init += time_diff;

        for (int j = 0; j < testrun->params->update_rounds_; j++)
        {
          //std::cout << "Update-Round: " << j + 1 << std::endl;

          //------------------------------------------------------------------------------
          // Start Timer for Multiplication
          //------------------------------------------------------------------------------
          //
		      /*spmv->generateRandomVector();

          start_clock(ce_start, ce_stop);

          spmv->deviceSpMV(aimGraph->memory_manager, config);

          time_diff = end_clock(ce_start, ce_stop);
          time_elapsed_multiplication += time_diff;*/

          //------------------------------------------------------------------------------
          // Start Timer for Transpose CSR
          //------------------------------------------------------------------------------
          //
          start_clock(ce_start, ce_stop);

          spmv->template transposeaim2CSR2aim<EdgeDataType>(faimGraph, config);

          time_diff = end_clock(ce_start, ce_stop);
          time_elapsed_transposeCSR += time_diff;

          if (testrun->params->verification_)
          {
            // Transpose back and check if everything is still the same
            spmv->template transposeaim2CSR2aim<EdgeDataType>(faimGraph, config);

            verification<VertexDataType, VertexUpdate, EdgeDataType, EdgeDataUpdate>(faimGraph, "Test Transpose ", parser, testrun, i, j, false);
          }

          //------------------------------------------------------------------------------
          // Start Timer for Transpose
          //------------------------------------------------------------------------------
          //
          //start_clock(ce_start, ce_stop);

          //spmv->transpose(aimGraph->memory_manager, config);

          //time_diff = end_clock(ce_start, ce_stop);
          //time_elapsed_transpose += time_diff;
        }
      }

      // std::cout << "Time elapsed during initialization:   ";
      // std::cout << std::setw(10) << time_elapsed_init / static_cast<float>(testrun->params->rounds_) << " ms" << std::endl;

      //std::cout << "Time elapsed during MV multiplication:   ";
      //std::cout << std::setw(10) << time_elapsed_multiplication / static_cast<float>(testrun->params->rounds_ * testrun->params->update_rounds_) << " ms" << std::endl;

      std::cout << "Time elapsed during transpose CSR:   ";
      std::cout << std::setw(10) << time_elapsed_transposeCSR / static_cast<float>(testrun->params->rounds_ * testrun->params->update_rounds_) << " ms" << std::endl;

      //std::cout << "Time elapsed during transpose:   ";
      //std::cout << std::setw(10) << time_elapsed_transpose / static_cast<float>(testrun->params->rounds_ * testrun->params->update_rounds_) << " ms" << std::endl;
    }
    std::cout << "################################################################" << std::endl;
    std::cout << "################################################################" << std::endl;
    std::cout << "################################################################" << std::endl;
  }
  // Increment the testrun index
  config->testrun_index_++;
}

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType>
void testrunImplementationMM(const std::shared_ptr<Config>& config, const std::unique_ptr<Testruns>& testrun)
{
  // Timing
  cudaEvent_t ce_start, ce_stop;
  float time_diff;
  std::vector<PerformanceData> perfData;

  // Global Properties
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
      float time_elapsed_multiplication = 0;

      //Setup graph parser and read in file
      std::unique_ptr<GraphParser> parser(new GraphParser(graph, true));
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

        std::unique_ptr<faimGraph<VertexDataType, VertexUpdate, EdgeDataType, EdgeDataUpdate>> faimGraph(std::make_unique<faimGraph<VertexDataType, VertexUpdate, EdgeDataType, EdgeDataUpdate>>(config, parser));
        std::unique_ptr<SpMMManager> spmm(std::make_unique<SpMMManager>(faimGraph->memory_manager->next_free_vertex_index, faimGraph->memory_manager->next_free_vertex_index, faimGraph->memory_manager->next_free_vertex_index, faimGraph->memory_manager->next_free_vertex_index));

        start_clock(ce_start, ce_stop);

        spmm->template initializeFaimGraphMatrix<EdgeDataType>(faimGraph, parser, config);

        time_diff = end_clock(ce_start, ce_stop);
        time_elapsed_init += time_diff;

        for (int j = 0; j < testrun->params->update_rounds_; j++)
        {
          //std::cout << "Update-Round: " << j + 1 << std::endl;

          //------------------------------------------------------------------------------
          // Start Timer for Multiplication MM
          //------------------------------------------------------------------------------
          //
          start_clock(ce_start, ce_stop);

          spmm->template spmmMultiplication<EdgeDataType>(faimGraph, config);

          time_diff = end_clock(ce_start, ce_stop);
          time_elapsed_multiplication += time_diff;

          /*std::cout << "Time elapsed during MM multiplication:   ";
          std::cout << std::setw(10) << time_diff << " ms" << std::endl;*/

          //hostMatrixMultiplyWriteToFile(parser, spmm->output_rows, spmm->output_columns, graph + ".matrix");

          if (testrun->params->verification_)
          {
            verificationMMMultiplication<VertexDataType, VertexUpdate, EdgeDataType, EdgeDataUpdate>(faimGraph, "Test Multiplication ", parser, testrun, i, j, false, spmm, graph + ".matrix");
          }

          spmm->template resetResultMatrix <EdgeDataType>(faimGraph, config);
        }
      }

       std::cout << "Time elapsed during initialization:   ";
       std::cout << std::setw(10) << time_elapsed_init / static_cast<float>(testrun->params->rounds_) << " ms" << std::endl;

      std::cout << "Time elapsed during MM multiplication:   ";
      std::cout << std::setw(10) << time_elapsed_multiplication / static_cast<float>(testrun->params->rounds_ * testrun->params->update_rounds_) << " ms" << std::endl;
    }
    std::cout << "################################################################" << std::endl;
    std::cout << "################################################################" << std::endl;
    std::cout << "################################################################" << std::endl;
  }
  // Increment the testrun index
  config->testrun_index_++;
}

//------------------------------------------------------------------------------
//
//template <typename VertexDataType, typename EdgeDataType>
//void testrunImplementationMM(const std::shared_ptr<Config>& config, const std::unique_ptr<Testruns>& testrun)
//{
//  // Timing
//  cudaEvent_t ce_start, ce_stop;
//  float time_diff;
//  std::vector<PerformanceData> perfData;
//
//  // Global Properties
//  bool gpuVerification = true;
//  bool writeToFile = false;
//  bool duplicate_check = true;
//
//  for (auto batchsize : testrun->batchsizes)
//  {
//    std::cout << "Batchsize: " << batchsize << std::endl;
//    for (const auto& graph : testrun->graphs)
//    {
//      // Timing information
//      float time_elapsed_init = 0;
//      float time_elapsed_multiplication = 0;
//
//      //Setup graph parser and read in file
//      std::unique_ptr<GraphParser> parser(new GraphParser(graph, true));
//      if (!parser->parseGraph())
//      {
//        std::cout << "Error while parsing graph" << std::endl;
//        return;
//      }
//
//      for (int i = 0; i < testrun->params->rounds_; i++)
//      {
//        //------------------------------------------------------------------------------
//        // Initialization phase
//        //------------------------------------------------------------------------------
//        //
//        std::cout << "Round: " << i + 1 << std::endl;
//
//        config->device_mem_size_ /= 3;
//
//        std::unique_ptr<aimGraph<VertexDataType, VertexUpdate, EdgeDataType, EdgeDataUpdate>> input_matrix_A(std::make_unique<aimGraph<VertexDataType, VertexUpdate, EdgeDataType, EdgeDataUpdate>>(config, parser));
//        std::unique_ptr<aimGraph<VertexDataType, VertexUpdate, EdgeDataType, EdgeDataUpdate>> input_matrix_B(std::make_unique<aimGraph<VertexDataType, VertexUpdate, EdgeDataType, EdgeDataUpdate>>(config, parser));
//        std::unique_ptr<aimGraph<VertexDataType, VertexUpdate, EdgeDataType, EdgeDataUpdate>> output_matrix(std::make_unique<aimGraph<VertexDataType, VertexUpdate, EdgeDataType, EdgeDataUpdate>>(config, parser));
//        std::unique_ptr<SpMMManager> spmm(std::make_unique<SpMMManager>(input_matrix_A->memory_manager->next_free_vertex_index, input_matrix_A->memory_manager->next_free_vertex_index, input_matrix_A->memory_manager->next_free_vertex_index, input_matrix_A->memory_manager->next_free_vertex_index));
//
//        start_clock(ce_start, ce_stop);
//
//        input_matrix_A->initializeaimGraphMatrix(parser);
//        input_matrix_B->initializeaimGraphMatrix(parser);
//        output_matrix->initializeaimGraphEmptyMatrix(parser->getNumberOfVertices());
//
//        time_diff = end_clock(ce_start, ce_stop);
//        time_elapsed_init += time_diff;
//
//        for (int j = 0; j < testrun->params->update_rounds_; j++)
//        {
//          //std::cout << "Update-Round: " << j + 1 << std::endl;
//
//          //------------------------------------------------------------------------------
//          // Start Timer for Multiplication MM
//          //------------------------------------------------------------------------------
//          //
//          start_clock(ce_start, ce_stop);
//
//          spmm->template spmmMultiplication<EdgeDataType>(input_matrix_A, input_matrix_B, output_matrix, config);
//
//          time_diff = end_clock(ce_start, ce_stop);
//          time_elapsed_multiplication += time_diff;
//
//          std::cout << "Time elapsed during MM multiplication:   ";
//          std::cout << std::setw(10) << time_diff << " ms" << std::endl;
//
//          if (testrun->params->verification_)
//          {
//            verificationMMMultiplication<VertexDataType, VertexUpdate, EdgeDataType, EdgeDataUpdate>(output_matrix, "Test Multiplication ", parser, testrun, i, j, false, spmm);
//          }
//
//          spmm->template resetResultMatrix <EdgeDataType>(output_matrix, config, true);
//        }
//      }
//
//      std::cout << "Time elapsed during initialization:   ";
//      std::cout << std::setw(10) << time_elapsed_init / static_cast<float>(testrun->params->rounds_) << " ms" << std::endl;
//
//      std::cout << "Time elapsed during MM multiplication:   ";
//      std::cout << std::setw(10) << time_elapsed_multiplication / static_cast<float>(testrun->params->rounds_ * testrun->params->update_rounds_) << " ms" << std::endl;
//    }
//    std::cout << "################################################################" << std::endl;
//    std::cout << "################################################################" << std::endl;
//    std::cout << "################################################################" << std::endl;
//  }
//  // Increment the testrun index
//  config->testrun_index_++;
//}

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename UpdateDataType>
void verification(std::unique_ptr<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>>& aimGraph,
                  const std::string& outputstring,
                  std::unique_ptr<GraphParser>& parser,
                  const std::unique_ptr<Testruns>& testrun,
                  int round,
                  int updateround,
                  bool duplicate_check)
{
  std::cout << "############ " << outputstring << " " << (round * testrun->params->rounds_) + updateround << " ############" << std::endl;
  std::unique_ptr<aimGraphCSR> verify_graph = aimGraph->verifyGraphStructure(aimGraph->memory_manager);

  // Compare graph structures
  if (!aimGraph->compareGraphs(parser, verify_graph, duplicate_check))
  {
    std::cout << "########## Graphs are NOT the same ##########" << std::endl;
    exit(-1);
  }
}

std::unique_ptr<CSRMatrix> hostMatrixMultiply(std::unique_ptr<GraphParser>& parser, vertex_t number_rows, vertex_t number_columns)
{
  std::unique_ptr<CSRMatrix> csr_matrix(std::make_unique<CSRMatrix>());

  auto const& offset = parser->getOffset();
  auto const& adjacency = parser->getAdjacency();
  auto const& matrix_values = parser->getMatrixValues();

  //for (int i = 0; i < offset[1282] - offset[1281]; ++i)
  //{
  //  std::cout << "Dest: " << adjacency[offset[1281] + i] << " Val: " << matrix_values[offset[1281] + i] << std::endl;
  //}
  //std::cout << "###########" << std::endl;
  //for (int j = 0; j < offset[1281] - offset[1280]; ++j)
  //{
  //  std::cout << "Dest: " << adjacency[offset[1280] + j] << " Val: " << matrix_values[offset[1280] + j] << std::endl;
  //}
  //std::cout << "###########" << std::endl;
  //for (int j = 0; j < offset[1283] - offset[1282]; ++j)
  //{
  //  std::cout << "Dest: " << adjacency[offset[1282] + j] << " Val: " << matrix_values[offset[1282] + j] << std::endl;
  //}
  //std::cout << "###########" << std::endl;
  //for (int j = 0; j < offset[107873] - offset[107872]; ++j)
  //{
  //  std::cout << "Dest: " << adjacency[offset[107872] + j] << " Val: " << matrix_values[offset[107872] + j] << std::endl;
  //}

  csr_matrix->offset.resize(offset.size(), 0);

  for (unsigned int i = 0; i < offset.size() - 1; ++i)
  {
    vertex_t neighbours = offset[i + 1] - offset[i];
    auto A_adjacency_iterator = adjacency.begin() + offset[i];
    auto A_matrix_value_iterator = matrix_values.begin() + offset[i];

    for (unsigned int j = 0; j < neighbours; ++j)
    {
      vertex_t A_destination = A_adjacency_iterator[j];
      matrix_t A_value = A_matrix_value_iterator[j];
      vertex_t B_neighbours = offset[A_destination + 1] - offset[A_destination];
      auto B_adjacency_iterator = adjacency.begin() + offset[A_destination];
      auto B_matrix_value_iterator = matrix_values.begin() + offset[A_destination];

      /*if (i == 66327)
      {
        std::cout << "A_Destination: " << A_destination << " and value: " << A_value << std::endl;
      }*/

      for (unsigned int k = 0; k < B_neighbours; ++k)
      {
        vertex_t B_destination = B_adjacency_iterator[k];
        matrix_t B_value = B_matrix_value_iterator[k];
        /*if (i == 66327)
        {
          std::cout << "B_Destination: " << B_destination << " and value: " << B_value << " at position: " << offset[A_destination] + k <<  std::endl;
        }*/
        B_value *= A_value;      

        // Check if value is already here or not
        auto begin_iter = csr_matrix->adjacency.begin() + csr_matrix->offset[i];
        auto end_iter = csr_matrix->adjacency.begin() + csr_matrix->offset[i + 1];

        // Search if item is present      
        auto pos = std::find(begin_iter, end_iter, B_destination);
        if (pos != end_iter)
        {
          // Edge already present, add new factor
          vertex_t position = pos - csr_matrix->adjacency.begin();
          csr_matrix->matrix_values[position] += B_value;
        }
        else
        {
          // Insert edge
          auto matrix_iter = csr_matrix->matrix_values.begin() + (pos - csr_matrix->adjacency.begin());
          csr_matrix->adjacency.insert(pos, B_destination);
          csr_matrix->matrix_values.insert(matrix_iter, B_value);
          for (auto l = i + 1; l < (csr_matrix->offset.size()); ++l)
          {
            csr_matrix->offset[l] += 1;
          }
        }
      }
    }
    if(i % 10000 == 0)
      std::cout << "HOST: Handled " << i << " vertices" << std::endl;
  }

  std::cout << "HOST: Number edges: " << csr_matrix->adjacency.size() << std::endl;

  return std::move(csr_matrix);
}

void hostMatrixMultiplyWriteToFile(std::unique_ptr<GraphParser>& parser, vertex_t number_rows, vertex_t number_columns, const std::string& filename)
{
  std::unique_ptr<CSRMatrix> csr_matrix(std::make_unique<CSRMatrix>());

  std::cout << "Write to file with name: " << filename << std::endl;

  auto const& offset = parser->getOffset();
  auto const& adjacency = parser->getAdjacency();
  auto const& matrix_values = parser->getMatrixValues();

  csr_matrix->offset.resize(offset.size(), 0);

  for (unsigned int i = 0; i < offset.size() - 1; ++i)
  {
    vertex_t neighbours = offset[i + 1] - offset[i];
    auto A_adjacency_iterator = adjacency.begin() + offset[i];
    auto A_matrix_value_iterator = matrix_values.begin() + offset[i];

    for (unsigned int j = 0; j < neighbours; ++j)
    {
      vertex_t A_destination = A_adjacency_iterator[j];
      matrix_t A_value = A_matrix_value_iterator[j];
      vertex_t B_neighbours = offset[A_destination + 1] - offset[A_destination];
      auto B_adjacency_iterator = adjacency.begin() + offset[A_destination];
      auto B_matrix_value_iterator = matrix_values.begin() + offset[A_destination];

      for (unsigned int k = 0; k < B_neighbours; ++k)
      {
        vertex_t B_destination = B_adjacency_iterator[k];
        matrix_t B_value = B_matrix_value_iterator[k];
        B_value *= A_value;

        // Check if value is already here or not
        auto begin_iter = csr_matrix->adjacency.begin() + csr_matrix->offset[i];
        auto end_iter = csr_matrix->adjacency.begin() + csr_matrix->offset[i + 1];

        // Search if item is present      
        auto pos = std::find(begin_iter, end_iter, B_destination);
        if (pos != end_iter)
        {
          // Edge already present, add new factor
          vertex_t position = pos - csr_matrix->adjacency.begin();
          csr_matrix->matrix_values[position] += B_value;
        }
        else
        {
          // Insert edge
          auto matrix_iter = csr_matrix->matrix_values.begin() + (pos - csr_matrix->adjacency.begin());
          csr_matrix->adjacency.insert(pos, B_destination);
          csr_matrix->matrix_values.insert(matrix_iter, B_value);
          for (auto l = i + 1; l < (csr_matrix->offset.size()); ++l)
          {
            csr_matrix->offset[l] += 1;
          }
        }
      }
    }
    if (i % 10 == 0)
      std::cout << "HOST: Handled " << i << " vertices" << std::endl;
  }

  std::cout << "HOST: Number edges: " << csr_matrix->adjacency.size() << std::endl;

  // Write to File
  std::ofstream matrix_file(filename);

  // Write number of vertices and edges
  matrix_file << number_rows << " " << csr_matrix->adjacency.size() << std::endl;

  for (unsigned int i = 0; i < offset.size() - 1; ++i)
  {
    int offset = csr_matrix->offset[i];
    int neighbours = csr_matrix->offset[i + 1] - csr_matrix->offset[i];
    for (int j = 0; j < neighbours; ++j)
    {
      matrix_file << csr_matrix->adjacency[offset + j] << " ";
      matrix_file << csr_matrix->matrix_values[offset + j];
      if (j < (neighbours - 1))
        matrix_file << " ";
    }
    matrix_file << "\n"; 
  }
  matrix_file.close();


  return;
}

std::unique_ptr<CSRMatrix> hostReadMatrixMultiplyFromFile(const std::string& filename)
{
  std::unique_ptr<CSRMatrix> csr_matrix(std::make_unique<CSRMatrix>());
  std::ifstream graph_file(filename);
  std::string line;
  vertex_t number_vertices;
  vertex_t number_edges;
  vertex_t offset{ 0 };

  // Parse in number vertices and number edges
  std::getline(graph_file, line);
  std::istringstream istream(line);
  istream >> number_vertices;
  istream >> number_edges;

  std::cout << "Comparsion-Matrix | #v: " << number_vertices << " | #e: " << number_edges << std::endl;

  while (std::getline(graph_file, line))
  {
    vertex_t adjacency;
    matrix_t matrix_value;
    csr_matrix->offset.push_back(offset);

    std::istringstream istream(line);
    while (!istream.eof())
    {
      istream >> adjacency;
      istream >> matrix_value;
      ++offset;

      csr_matrix->adjacency.push_back(adjacency);
      csr_matrix->matrix_values.push_back(matrix_value);
    }
  }
  csr_matrix->offset.push_back(offset);

  graph_file.close();

  return std::move(csr_matrix);
}

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename VertexUpdateType, typename EdgeDataType, typename UpdateDataType>
void verificationMMMultiplication(std::unique_ptr<faimGraph<VertexDataType, VertexUpdateType, EdgeDataType, UpdateDataType>>& aimGraph,
                                  const std::string& outputstring,
                                  std::unique_ptr<GraphParser>& parser,
                                  const std::unique_ptr<Testruns>& testrun,
                                  int round,
                                  int updateround,
                                  bool duplicate_check,
                                  std::unique_ptr<SpMMManager>& spmm,
                                  const std::string& matrix_file)
{
  std::cout << "############ " << outputstring << " " << (round * testrun->params->rounds_) + updateround << " ############" << std::endl;
  std::unique_ptr<aimGraphCSR> verify_graph = aimGraph->verifyMatrixStructure(aimGraph->memory_manager, spmm->output_offset, spmm->output_rows);

  //auto csrmatrix = hostMatrixMultiply(parser, spmm->output_rows, spmm->output_columns);
  std::cout << "Parsing result matrix" << std::endl;
  auto csrmatrix = hostReadMatrixMultiplyFromFile(matrix_file);
  std::cout << "Verification in progress" << std::endl;
  // Compare graph structures
  if (!aimGraph->compareGraphs(csrmatrix, verify_graph, duplicate_check))
  {
    std::cout << "########## Graphs are NOT the same ##########" << std::endl;
    exit(-1);
  }
}

