//------------------------------------------------------------------------------
// CSVWriter.h
//
// faimGraph
//
//------------------------------------------------------------------------------
//

#pragma once

#include <fstream>

#include "Utility.h"

// Forward declaration
class Config;
class PerformanceData;

class CSVWriter
{
public:
  CSVWriter(){}
  CSVWriter(const std::string& name) 
  { 
    output_file_stream_.open("../tests/PerformanceData/" + name + ".csv"); 
    if (!output_file_stream_.is_open())
    {
      std::cout << "Could not write file!" << std::endl;
    }
    else
    {
      std::cout << "Opened file" << std::endl;
    }    
  }
  ~CSVWriter()
  {
    if (output_file_stream_.is_open())
      output_file_stream_.close();
  }

  void writePerformanceMetric(const std::string& name,
                              const std::shared_ptr<Config>& config,
                              const std::vector<PerformanceData>& perfdata,
                              int testrunindex);

  void writePageAllocationHeader(const std::shared_ptr<Config>& config,
                                 int number_vertices);

  void writePageAllocationLine(const std::shared_ptr<Config>& config, 
                               std::vector<vertex_t>& pages_in_memory,
                               vertex_t pages_used,
                               int number_vertices);
  
private:
  std::ofstream output_file_stream_;
};

class PerformanceData
{
public:
  PerformanceData(float init, float insert, float del);
  float init_time;
  float insert_time;
  float delete_time;
};
