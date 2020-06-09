//------------------------------------------------------------------------------
// ConfigurationParser.h
//
// faimGraph
//
//------------------------------------------------------------------------------
//

#pragma once

#include <string>
#include <memory>

#include "Utility.h"

class Testruns;
class Config;
enum class GraphDirectionality;

class ConfigurationParser
{
public:
  ConfigurationParser(const std::string& filename);
  std::shared_ptr<Config> parseConfiguration();

private:
  std::string filename_;
  // Available Tags
  const std::string TAG_FILE_{"aimGraph"};
  const std::string TAG_GLOBAL{ "global" };
  const std::string TAG_TESTRUN{ "testrun" };
  const std::string TAG_CONFIG{ "configuration" };
  const std::string TAG_GRAPHS{ "graphs" };
  const std::string TAG_GRAPHS_FILENAME{ "filename" };
  const std::string TAG_BATCHES{ "batches" };
  const std::string TAG_BATCH_SIZE{ "batchsize" };

  // Global Tags
  const std::string TAG_MEM_SIZE{ "devicememsize" };
  const std::string TAG_DEV_ID{ "deviceID" };

  // Configuration Tags
  const std::string TAG_MEMORY_LAYOUT{ "memorylayout" };
  const std::string TAG_GRAPH_MODE{ "graphmode" };
  const std::string TAG_UPDATE_VARIANT{ "updatevariant" };
  const std::string TAG_ROUNDS{ "rounds" };
  const std::string TAG_UPDATE_ROUNDS{ "updaterounds" };
  const std::string TAG_PAGE_SIZE{ "pagesize" };
  const std::string TAG_INIT_LAUNCH{ "initlaunchblocksize" };
  const std::string TAG_INSERT_LAUNCH{ "insertlaunchblocksize" };
  const std::string TAG_DELETE_LAUNCH{ "deletelaunchblocksize" };
  const std::string TAG_VERIFICATION{ "verification" };
  const std::string TAG_PERFORMANCE_DATA{ "performanceoutput" };
  const std::string TAG_MEMORY_OVERALLOCATION{ "memoryoverallocation" };
  const std::string TAG_DELETION_VARIATION{ "deletionvariation" };
  const std::string TAG_DELETION_COMPACTION{ "deletioncompaction" };
  const std::string TAG_STACKSIZE{ "stacksize" };
  const std::string TAG_QUEUESIZE{ "queuesize" };
  const std::string TAG_DIRECTIONALITY{ "directionality" };
  const std::string TAG_SORTING{ "sorting" };
  const std::string TAG_PAGE_LINKAGE{ "pagelinkage" };
};

class Config
{
public:
  float device_mem_size_;
  int deviceID_;
  std::vector<std::unique_ptr<Testruns>> testruns_;
  int testrun_index_{0};
};

class ConfigurationParameters
{
public:
  enum class UpdateVariant
  {
    STANDARD,
    WARPSIZED,
    VERTEXCENTRIC,
    VERTEXCENTRICSORTED
  };

  enum class PerformanceOutput
  {
    STDOUT,
    CSV
  };

  enum class DeletionVariant
  {
    STANDARD,
    COMPACTION
  };

  enum class MemoryLayout
  {
    AOS,
    SOA
  };

  enum class GraphMode
  {
    SIMPLE,
    WEIGHT,
    SEMANTIC,
	MATRIX
  };

  enum class GraphDirectionality
  {
    DIRECTED,
    UNDIRECTED
  };

  enum class PageLinkage
  {
    SINGLE,
    DOUBLE
  };

  MemoryLayout memory_layout_;
  GraphMode graph_mode_;
  UpdateVariant update_variant_;
  PerformanceOutput performance_output_;
  DeletionVariant deletion_variant_;
  GraphDirectionality directionality_;
  PageLinkage page_linkage_;
  int rounds_;
  int update_rounds_;
  uint32_t page_size_;
  int init_launch_block_size_;
  int insert_launch_block_size_;
  int delete_launch_block_size_;
  bool verification_;
  float memory_overallocation_factor_;
  uint32_t stacksize_;
  uint32_t queuesize_;
  bool sorting_;
};

class Testruns
{
public:
  std::unique_ptr<ConfigurationParameters> params;
  std::vector<std::string> graphs;
  std::vector<int> batchsizes;
};


void printConfigurationInformation(const std::shared_ptr<Config>& testruns);
