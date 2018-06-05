//------------------------------------------------------------------------------
// CSVWriter.h
//
// faimGraph
//
//------------------------------------------------------------------------------
//

#include <iostream>
#include <time.h>

#include "CSVWriter.h"
#include "ConfigurationParser.h"

//------------------------------------------------------------------------------
//
void CSVWriter::writePerformanceMetric(const std::string& name, 
                                       const std::shared_ptr<Config>& config,
                                       const std::vector<PerformanceData>& perfdata,
                                       int testrunindex)
{
  time_t rawtime;
  struct tm* timeinfo;
  char buffer[80];
  time(&rawtime);
  timeinfo = localtime(&rawtime);
  strftime(buffer, 80, "%F_%H-%M-%S", timeinfo);
  std::string time(buffer);
  size_t index = 0;

  output_file_stream_.open("../tests/PerformanceData/" + time + ".csv");
  if (!output_file_stream_.is_open())
  {
    std::cout << "Could not write file!" << std::endl;
  }

  output_file_stream_ << name.c_str() << "\n";
  output_file_stream_ << "\n";
  output_file_stream_ << "Initialisationtime" << "\n";
  output_file_stream_ << ";";
  for (const auto& graphs : config->testruns_.at(testrunindex)->graphs)
  {
    output_file_stream_ << graphs.c_str() << ";";
  }
  output_file_stream_ << "\n";
  for (auto batchsize : config->testruns_.at(testrunindex)->batchsizes)
  {
    output_file_stream_ << "Batchsize: " << batchsize << ";";
    for (const auto& graphs : config->testruns_.at(testrunindex)->graphs)
    {
      const PerformanceData& perf = perfdata.at(index);
      output_file_stream_ << perf.init_time << ";";
      index++;
    }
    output_file_stream_ << "\n";
  }
  index = 0;

  output_file_stream_ << "\n";
  output_file_stream_ << "Insertiontime" << "\n";
  output_file_stream_ << ";";
  for (const auto& graphs : config->testruns_.at(testrunindex)->graphs)
  {
    output_file_stream_ << graphs.c_str() << ";";
  }
  output_file_stream_ << "\n";
  for (auto batchsize : config->testruns_.at(testrunindex)->batchsizes)
  {
    output_file_stream_ << "Batchsize: " << batchsize << ";";
    for (const auto& graphs : config->testruns_.at(testrunindex)->graphs)
    {
      const PerformanceData& perf = perfdata.at(index);
      output_file_stream_ << perf.insert_time << ";";
      index++;
    }
    output_file_stream_ << "\n";
  }
  index = 0;

  output_file_stream_ << "\n";
  output_file_stream_ << "Deletiontime" << "\n";
  output_file_stream_ << ";";
  for (const auto& graphs : config->testruns_.at(testrunindex)->graphs)
  {
    output_file_stream_ << graphs.c_str() << ";";
  }
  output_file_stream_ << "\n";
  for (auto batchsize : config->testruns_.at(testrunindex)->batchsizes)
  {
    output_file_stream_ << "Batchsize: " << batchsize << ";";
    for (const auto& graphs : config->testruns_.at(testrunindex)->graphs)
    {
      const PerformanceData& perf = perfdata.at(index);
      output_file_stream_ << perf.delete_time << ";";
      index++;
    }
    output_file_stream_ << "\n";
  }
  index = 0;
}

//------------------------------------------------------------------------------
//
void CSVWriter::writePageAllocationHeader(const std::shared_ptr<Config>& config,
                                        int number_vertices)
{
  int factor = 25;
  int range = number_vertices / factor;
  int start = 0;
  vertex_t page_number = 0;

  // Write header data
  output_file_stream_ << ";";
  for (int i = 0; i < factor; ++i)
  {
    output_file_stream_ << "Vertices " << i * range << " - " << ((i + 1) * range) << ";";
  }
  output_file_stream_ << "Pages used in total: " << "\n";
}

//------------------------------------------------------------------------------
//
void CSVWriter::writePageAllocationLine(const std::shared_ptr<Config>& config,
                                        std::vector<vertex_t>& pages_in_memory,
                                        vertex_t pages_used,
                                        int number_vertices)
{
  static int round_number = 0;
  int factor = 25;
  int range = number_vertices / factor;
  int start = 0;
  vertex_t page_number = 0;

  output_file_stream_ << "Round: " << round_number << ";";

  for (int i = 0; i < factor; ++i)
  {
    for (int j = 0; j < range; ++j)
    {
      page_number += pages_in_memory.at(start + j);
    }
    output_file_stream_ << page_number << ";";
    start += range;
    page_number = 0;
  }
  output_file_stream_ << pages_used << "\n";
  ++round_number;
  return;
}

//------------------------------------------------------------------------------
//
PerformanceData::PerformanceData(float init, float insert, float del):
  init_time{init}, insert_time{insert}, delete_time{del}
{}
