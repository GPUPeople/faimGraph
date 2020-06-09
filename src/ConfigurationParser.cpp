//------------------------------------------------------------------------------
// ConfigurationParser.cpp
//
// faimGraph
//
//------------------------------------------------------------------------------
//

// Library Includes 
#include <iostream>
#include <fstream>
#include <sstream>

// Project Includes
#include "ConfigurationParser.h"

//------------------------------------------------------------------------------
//
ConfigurationParser::ConfigurationParser(const std::string& filename):
  filename_{filename}
{}

//------------------------------------------------------------------------------
//
std::shared_ptr<Config> ConfigurationParser::parseConfiguration()
{
  auto config = std::make_shared<Config>();
  std::vector<std::unique_ptr<Testruns>> testruns;

  std::ifstream input_file_stream;
  input_file_stream.open(filename_);
  if (!input_file_stream.is_open())
  {
    std::cout << "Could not read file!" << std::endl;
    exit(-1);
  }

  // Read in file
  std::stringstream file_content_stream;
  std::string line;
  for (int i = 0; getline(input_file_stream, line); i++)
  {
    file_content_stream << line;
  }

  std::string file_content = file_content_stream.str();

  // Parse configuration file
  // There may be multiple testruns specified per configuration file 
  std::size_t start = 0;
  std::size_t end = 0;
  std::vector<std::string> testrun_strings;
  size_t sizeoftag = TAG_GLOBAL.length() + 3;
  if ((start = file_content.find("<" + TAG_GLOBAL + ">")) != std::string::npos &&
    (end = file_content.find("</" + TAG_GLOBAL + ">")) != std::string::npos)
  {
    std::string global_settings = file_content.substr(start, (end + sizeoftag) - start);
    file_content.erase(start, (end + sizeoftag) - start);

    /* #########################################################
    # Device Memory
    ######################################################### */
    if ((start = global_settings.find("<" + TAG_MEM_SIZE + ">")) != std::string::npos &&
      (end = global_settings.find("</" + TAG_MEM_SIZE + ">")) != std::string::npos)
    {
      sizeoftag = TAG_MEM_SIZE.length() + 2;
      std::string param = global_settings.substr(start + sizeoftag, end - (start + sizeoftag));
      config->device_mem_size_ = std::stof(param);
      //std::cout << testrun->params->device_mem_size_ << std::endl;
    }

    /* #########################################################
    # Device ID
    ######################################################### */
    if ((start = global_settings.find("<" + TAG_DEV_ID + ">")) != std::string::npos &&
      (end = global_settings.find("</" + TAG_DEV_ID + ">")) != std::string::npos)
    {
      sizeoftag = TAG_DEV_ID.length() + 2;
      std::string param = global_settings.substr(start + sizeoftag, end - (start + sizeoftag));
      config->deviceID_ = std::stoi(param);
      //std::cout << testrun->params->device_mem_size_ << std::endl;
    }
  }


  // </ + > = 3 + length(tag)
  sizeoftag = TAG_TESTRUN.length() + 3;
  while ((start = file_content.find("<" + TAG_TESTRUN + ">")) != std::string::npos &&
    (end = file_content.find("</" + TAG_TESTRUN + ">")) != std::string::npos)
  {
    // Split into individual testruns
    testrun_strings.push_back(file_content.substr(start, (end + sizeoftag) - start));
    file_content.erase(start, (end + sizeoftag) - start);
  }

  int i = 1;
  for (std::string& test : testrun_strings)
  {
    //std::cout << "################################  " << i++ << " testrun ################################" << std::endl;
    // First set all values to default values, overwrite values defined 
    auto testrun = std::make_unique<Testruns>();
    testrun->params = std::make_unique<ConfigurationParameters>();
    testrun->params->rounds_ = 10;
    testrun->params->update_rounds_ = 10;
    testrun->params->page_size_ = EDGEBLOCKSIZE;
    testrun->params->init_launch_block_size_ = KERNEL_LAUNCH_BLOCK_SIZE;
    testrun->params->insert_launch_block_size_ = KERNEL_LAUNCH_BLOCK_SIZE_INSERT;
    testrun->params->delete_launch_block_size_ = KERNEL_LAUNCH_BLOCK_SIZE_DELETE;
    testrun->params->verification_ = false;
    testrun->params->update_variant_ = ConfigurationParameters::UpdateVariant::STANDARD;
    testrun->params->deletion_variant_ = ConfigurationParameters::DeletionVariant::STANDARD;
    testrun->params->performance_output_ = ConfigurationParameters::PerformanceOutput::CSV;
    testrun->params->memory_overallocation_factor_ = 1.0f;
    testrun->params->memory_layout_ = ConfigurationParameters::MemoryLayout::AOS;
    testrun->params->graph_mode_ = ConfigurationParameters::GraphMode::SIMPLE;
    testrun->params->stacksize_ = 200000000;
    testrun->params->queuesize_ = 2000000;
    testrun->params->directionality_ = ConfigurationParameters::GraphDirectionality::UNDIRECTED;
    testrun->params->sorting_ = false;
    testrun->params->page_linkage_ = ConfigurationParameters::PageLinkage::SINGLE;

    // Parse available params

    /* #########################################################
    # Memory Layout
    ######################################################### */
    if ((start = test.find("<" + TAG_MEMORY_LAYOUT + ">")) != std::string::npos &&
      (end = test.find("</" + TAG_MEMORY_LAYOUT + ">")) != std::string::npos)
    {
      sizeoftag = TAG_MEMORY_LAYOUT.length() + 2;
      std::string param = test.substr(start + sizeoftag, end - (start + sizeoftag));
      if (param.compare("AOS") == 0)
      {
        testrun->params->memory_layout_ = ConfigurationParameters::MemoryLayout::AOS;
      }
      else if (param.compare("SOA") == 0)
      {
        testrun->params->memory_layout_ = ConfigurationParameters::MemoryLayout::SOA;
      }
      else
      {
        std::cout << "Could not parse memory layout, set it to standard = AOS\n";
        testrun->params->memory_layout_ = ConfigurationParameters::MemoryLayout::AOS;
      }
    }

    /* #########################################################
    # Graph Mode
    ######################################################### */
    if ((start = test.find("<" + TAG_GRAPH_MODE + ">")) != std::string::npos &&
      (end = test.find("</" + TAG_GRAPH_MODE + ">")) != std::string::npos)
    {
      sizeoftag = TAG_GRAPH_MODE.length() + 2;
      std::string param = test.substr(start + sizeoftag, end - (start + sizeoftag));
      if (param.compare("simple") == 0)
      {
        testrun->params->graph_mode_ = ConfigurationParameters::GraphMode::SIMPLE;
      }
      else if (param.compare("weight") == 0)
      {
        testrun->params->graph_mode_ = ConfigurationParameters::GraphMode::WEIGHT;
      }
      else if (param.compare("semantic") == 0)
      {
        testrun->params->graph_mode_ = ConfigurationParameters::GraphMode::SEMANTIC;
      }
	  else if (param.compare("matrix") == 0)
	  {
		  testrun->params->graph_mode_ = ConfigurationParameters::GraphMode::MATRIX;
	  }
      else
      {
        std::cout << "Could not parse graph mode, set it to standard = simple\n";
        testrun->params->graph_mode_ = ConfigurationParameters::GraphMode::SIMPLE;
      }
    }

    /* #########################################################
    # Update Variant
    ######################################################### */
    if ((start = test.find("<" + TAG_UPDATE_VARIANT + ">")) != std::string::npos &&
      (end = test.find("</" + TAG_UPDATE_VARIANT + ">")) != std::string::npos)
    {
      sizeoftag = TAG_UPDATE_VARIANT.length() + 2;
      std::string param = test.substr(start + sizeoftag, end - (start + sizeoftag));
      if (param.compare("standard") == 0)
      {
        testrun->params->update_variant_ = ConfigurationParameters::UpdateVariant::STANDARD;
      }
      else if (param.compare("warpsized") == 0)
      {
        testrun->params->update_variant_ = ConfigurationParameters::UpdateVariant::WARPSIZED;
      }
      else if (param.compare("vertexcentric") == 0)
      {
        testrun->params->update_variant_ = ConfigurationParameters::UpdateVariant::VERTEXCENTRIC;
      }
      else if (param.compare("vertexcentricsorted") == 0)
      {
        testrun->params->update_variant_ = ConfigurationParameters::UpdateVariant::VERTEXCENTRICSORTED;
      }
      else
      {
        std::cout << "Could not parse Update variant, set it to standard\n";
        testrun->params->update_variant_ = ConfigurationParameters::UpdateVariant::STANDARD;
      }
    }

    /* #########################################################
    # Rounds
    ######################################################### */
    if((start = test.find("<" + TAG_ROUNDS + ">")) != std::string::npos &&
    (end = test.find("</" + TAG_ROUNDS + ">")) != std::string::npos)
    {
      sizeoftag = TAG_ROUNDS.length() + 2;
      std::string param = test.substr(start + sizeoftag, end - (start + sizeoftag));
      testrun->params->rounds_ = std::stoi(param);
    }

    /* #########################################################
    # Update Rounds
    ######################################################### */
    if((start = test.find("<" + TAG_UPDATE_ROUNDS + ">")) != std::string::npos &&
    (end = test.find("</" + TAG_UPDATE_ROUNDS + ">")) != std::string::npos)
    {
      sizeoftag = TAG_UPDATE_ROUNDS.length() + 2;
      std::string param = test.substr(start + sizeoftag, end - (start + sizeoftag));
      testrun->params->update_rounds_ = std::stoi(param);
    }

    /* #########################################################
    # Page Size
    ######################################################### */
    if((start = test.find("<" + TAG_PAGE_SIZE + ">")) != std::string::npos &&
    (end = test.find("</" + TAG_PAGE_SIZE + ">")) != std::string::npos)
    {
      sizeoftag = TAG_PAGE_SIZE.length() + 2;
      std::string param = test.substr(start + sizeoftag, end - (start + sizeoftag));

      int blocksize = std::stoi(param);
      if (testrun->params->update_variant_ == ConfigurationParameters::UpdateVariant::WARPSIZED)
      {
        if (blocksize != BLOCK_SIZE_WARP_SIZED_KERNEL_LAUNCH)
        {
          blocksize = BLOCK_SIZE_WARP_SIZED_KERNEL_LAUNCH;
          std::cout << "Invalid BlockSize Config\n";
        }
      }

      testrun->params->page_size_ = blocksize;
    }
    else
    {
      std::cout << "############# No page size specified, will use 64 Bytes!" << std::endl;
    }

    /* #########################################################
    # Stacksize
    ######################################################### */
    if((start = test.find("<" + TAG_STACKSIZE + ">")) != std::string::npos &&
    (end = test.find("</" + TAG_STACKSIZE + ">")) != std::string::npos)
    {
      sizeoftag = TAG_STACKSIZE.length() + 2;
      std::string param = test.substr(start + sizeoftag, end - (start + sizeoftag));
      testrun->params->stacksize_ = std::stoi(param);
    }

    /* #########################################################
    # Queuesize
    ######################################################### */
    if((start = test.find("<" + TAG_QUEUESIZE + ">")) != std::string::npos &&
    (end = test.find("</" + TAG_QUEUESIZE + ">")) != std::string::npos)
    {
      sizeoftag = TAG_QUEUESIZE.length() + 2;
      std::string param = test.substr(start + sizeoftag, end - (start + sizeoftag));
      testrun->params->queuesize_ = std::stoi(param);
    }

    /* #########################################################
    # Init Kernel Launch Block Size
    ######################################################### */
    if((start = test.find("<" + TAG_INIT_LAUNCH + ">")) != std::string::npos &&
    (end = test.find("</" + TAG_INIT_LAUNCH + ">")) != std::string::npos)
    {
      sizeoftag = TAG_INIT_LAUNCH.length() + 2;
      std::string param = test.substr(start + sizeoftag, end - (start + sizeoftag));
      testrun->params->init_launch_block_size_ = std::stoi(param);
    }

    /* #########################################################
    # Insertion Kernel Launch Block Size
    ######################################################### */
    if((start = test.find("<" + TAG_INSERT_LAUNCH + ">")) != std::string::npos &&
    (end = test.find("</" + TAG_INSERT_LAUNCH + ">")) != std::string::npos)
    {
      sizeoftag = TAG_INSERT_LAUNCH.length() + 2;
      std::string param = test.substr(start + sizeoftag, end - (start + sizeoftag));
      int init_launch = std::stoi(param);
      if (testrun->params->update_variant_ == ConfigurationParameters::UpdateVariant::WARPSIZED)
      {
        if (init_launch != KERNEL_LAUNCH_BLOCK_SIZE_WARP_SIZED)
        {
          init_launch = KERNEL_LAUNCH_BLOCK_SIZE_WARP_SIZED;
          std::cout << "Invalid Insert Launch Config\n";
        }          
      }
      testrun->params->insert_launch_block_size_ = init_launch;
    }

    /* #########################################################
    # Deletion Kernel Launch Block Size
    ######################################################### */
    if((start = test.find("<" + TAG_DELETE_LAUNCH + ">")) != std::string::npos &&
    (end = test.find("</" + TAG_DELETE_LAUNCH + ">")) != std::string::npos)
    {
      sizeoftag = TAG_DELETE_LAUNCH.length() + 2;
      std::string param = test.substr(start + sizeoftag, end - (start + sizeoftag));
      int init_launch = std::stoi(param);
      if (testrun->params->update_variant_ == ConfigurationParameters::UpdateVariant::WARPSIZED)
      {
        if (init_launch != KERNEL_LAUNCH_BLOCK_SIZE_WARP_SIZED)
        {
          init_launch = KERNEL_LAUNCH_BLOCK_SIZE_WARP_SIZED;
          std::cout << "Invalid Delete Launch Config\n";
        }
      }
      testrun->params->delete_launch_block_size_ = init_launch;
    }

    /* #########################################################
    # Deletion Variant
    ######################################################### */
    if ((start = test.find("<" + TAG_DELETION_VARIATION + ">")) != std::string::npos &&
      (end = test.find("</" + TAG_DELETION_VARIATION + ">")) != std::string::npos)
    {
      sizeoftag = TAG_DELETION_VARIATION.length() + 2;
      std::string param = test.substr(start + sizeoftag, end - (start + sizeoftag));

      if (param.compare("standard") == 0)
      {
        testrun->params->deletion_variant_ = ConfigurationParameters::DeletionVariant::STANDARD;
        if (testrun->params->update_variant_ == ConfigurationParameters::UpdateVariant::VERTEXCENTRIC || testrun->params->update_variant_ == ConfigurationParameters::UpdateVariant::VERTEXCENTRICSORTED)
        {
          testrun->params->deletion_variant_ = ConfigurationParameters::DeletionVariant::COMPACTION;
          std::cout << "Invalid Deletion Variant for specialised setup, set it to compaction.\n";
        }
      }
      else if (param.compare("compaction") == 0)
      {
        testrun->params->deletion_variant_ = ConfigurationParameters::DeletionVariant::COMPACTION;
      }
      else
      {
        testrun->params->deletion_variant_ = ConfigurationParameters::DeletionVariant::STANDARD;
        std::cout << "Invalid Deletion Variant, set it to standard\n";
      }
    }

    /* #########################################################
    # Verification enabled
    ######################################################### */
    if ((start = test.find("<" + TAG_VERIFICATION + ">")) != std::string::npos &&
      (end = test.find("</" + TAG_VERIFICATION + ">")) != std::string::npos)
    {
      sizeoftag = TAG_VERIFICATION.length() + 2;
      std::string param = test.substr(start + sizeoftag, end - (start + sizeoftag));
      if (param.compare("true") == 0)
        testrun->params->verification_ = true;
      else
        testrun->params->verification_ = false;
    }

    /* #########################################################
    # Sorting enabled
    ######################################################### */
    if ((start = test.find("<" + TAG_SORTING + ">")) != std::string::npos &&
      (end = test.find("</" + TAG_SORTING + ">")) != std::string::npos)
    {
      sizeoftag = TAG_SORTING.length() + 2;
      std::string param = test.substr(start + sizeoftag, end - (start + sizeoftag));
      if (param.compare("true") == 0)
        testrun->params->sorting_ = true;
      else
        testrun->params->sorting_ = false;
    }

    /* #########################################################
    # Memory Overallocation Factor
    ######################################################### */
    if ((start = test.find("<" + TAG_MEMORY_OVERALLOCATION + ">")) != std::string::npos &&
      (end = test.find("</" + TAG_MEMORY_OVERALLOCATION + ">")) != std::string::npos)
    {
      sizeoftag = TAG_MEMORY_OVERALLOCATION.length() + 2;
      std::string param = test.substr(start + sizeoftag, end - (start + sizeoftag));
      testrun->params->memory_overallocation_factor_ = std::stof(param);
    }

    /* #########################################################
    # Performance Output
    ######################################################### */
    if ((start = test.find("<" + TAG_PERFORMANCE_DATA + ">")) != std::string::npos &&
      (end = test.find("</" + TAG_PERFORMANCE_DATA + ">")) != std::string::npos)
    {
      sizeoftag = TAG_PERFORMANCE_DATA.length() + 2;
      std::string param = test.substr(start + sizeoftag, end - (start + sizeoftag));
      if (param.compare("stdout") == 0)
      {
        testrun->params->performance_output_ = ConfigurationParameters::PerformanceOutput::STDOUT;
      }
      else if (param.compare("csv") == 0)
      {
        testrun->params->performance_output_ = ConfigurationParameters::PerformanceOutput::CSV;
      }
      else
      {
        std::cout << "Could not parse performance output, set it to stdout\n";
        testrun->params->performance_output_ = ConfigurationParameters::PerformanceOutput::STDOUT;
      }
    }

    /* #########################################################
    # Directionality
    ######################################################### */
    if ((start = test.find("<" + TAG_DIRECTIONALITY + ">")) != std::string::npos &&
      (end = test.find("</" + TAG_DIRECTIONALITY + ">")) != std::string::npos)
    {
      sizeoftag = TAG_DIRECTIONALITY.length() + 2;
      std::string param = test.substr(start + sizeoftag, end - (start + sizeoftag));
      if (param.compare("directed") == 0)
      {
        testrun->params->directionality_ = ConfigurationParameters::GraphDirectionality::DIRECTED;
      }
      else if (param.compare("undirected") == 0)
      {
        testrun->params->directionality_ = ConfigurationParameters::GraphDirectionality::UNDIRECTED;
      }
      else
      {
        std::cout << "Could not parse directionality, set it to undirected\n";
        testrun->params->directionality_ = ConfigurationParameters::GraphDirectionality::UNDIRECTED;
      }
    }

    /* #########################################################
    # Page Linkage
    ######################################################### */
    if ((start = test.find("<" + TAG_PAGE_LINKAGE + ">")) != std::string::npos &&
      (end = test.find("</" + TAG_PAGE_LINKAGE + ">")) != std::string::npos)
    {
      sizeoftag = TAG_PAGE_LINKAGE.length() + 2;
      std::string param = test.substr(start + sizeoftag, end - (start + sizeoftag));
      if (param.compare("single") == 0)
      {
        testrun->params->page_linkage_ = ConfigurationParameters::PageLinkage::SINGLE;
      }
      else if (param.compare("double") == 0)
      {
        testrun->params->page_linkage_ = ConfigurationParameters::PageLinkage::DOUBLE;
      }
      else
      {
        std::cout << "Could not parse page linkage, set it to single linked.\n";
        testrun->params->page_linkage_ = ConfigurationParameters::PageLinkage::SINGLE;
      }
    }


    /* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    % Graphs to be tested
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */
    if ((start = test.find("<" + TAG_GRAPHS + ">")) != std::string::npos &&
      (end = test.find("</" + TAG_GRAPHS + ">")) != std::string::npos)
    {
      // Now extract each graph to be used later
      sizeoftag = TAG_GRAPHS_FILENAME.length() + 2;
      while ((start = test.find("<" + TAG_GRAPHS_FILENAME + ">")) != std::string::npos &&
        (end = test.find("</" + TAG_GRAPHS_FILENAME + ">")) != std::string::npos)
      {
        // Split into individual testruns
        testrun->graphs.push_back(test.substr(start + sizeoftag, end - (start + sizeoftag)));
        test.erase(start, (end + sizeoftag + 1) - start);
      }
    }
    else
    {
      std::cout << "No graphs were specified, abort!" << std::endl;
      exit(0);
    }

    /* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    % Batchsizes to be tested
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */
    if ((start = test.find("<" + TAG_BATCHES + ">")) != std::string::npos &&
      (end = test.find("</" + TAG_BATCHES + ">")) != std::string::npos)
    {
      // Now extract each graph to be used later
      sizeoftag = TAG_BATCH_SIZE.length() + 2;
      while ((start = test.find("<" + TAG_BATCH_SIZE + ">")) != std::string::npos &&
        (end = test.find("</" + TAG_BATCH_SIZE + ">")) != std::string::npos)
      {
        // Split into individual testruns
        testrun->batchsizes.push_back(stoi(test.substr(start + sizeoftag, end - (start + sizeoftag))));
        test.erase(start, (end + sizeoftag + 1) - start);
      }
    }
    else
    {
      std::cout << "No batchsizes were specified!" << std::endl;
    }

    // Put each individual testrun into the vector
    testruns.push_back(std::move(testrun));
  }  

  std::cout << "Successfully parsed test configuration" << std::endl;
  config->testruns_= std::move(testruns);
  return std::move(config);
}

//------------------------------------------------------------------------------
//
void printConfigurationInformation(const std::shared_ptr<Config>& config)
{
  std::cout << "\n" << "############## Launch-Configuration START ##############" << "\n" << std::endl;
  std::cout << "~~~~~~~~~~~~~~~ Global Settings ~~~~~~~~~~~~~~~" << std::endl;
  std::cout << "DeviceID: " << config->deviceID_ << std::endl;
  std::cout << "Device-MemSize: " << config->device_mem_size_  << " GB"<< std::endl;
  int i = 1;
  for (const auto& testrun : config->testruns_)
  {
    std::cout << "--------------- Testrun " << i++ << " ---------------" << std::endl;
    if (testrun->params->memory_layout_ == ConfigurationParameters::MemoryLayout::AOS)
    {
      std::cout << "Memory Layout is " << "AOS (Array of Structures)\n";
    }
    else if (testrun->params->memory_layout_ == ConfigurationParameters::MemoryLayout::SOA)
    {
      std::cout << "Memory Layout is " << "SOA (Structure of Arrays)\n";
    }

    if (testrun->params->graph_mode_ == ConfigurationParameters::GraphMode::SIMPLE)
    {
      std::cout << "Graph Mode is " << "SIMPLE\n";
    }
    else if (testrun->params->graph_mode_ == ConfigurationParameters::GraphMode::WEIGHT)
    {
      std::cout << "Graph Mode is " << "WEIGHT\n";
    }
    else if (testrun->params->graph_mode_ == ConfigurationParameters::GraphMode::SEMANTIC)
    {
      std::cout << "Graph Mode is " << "SEMANTIC\n";
    }

    if (testrun->params->update_variant_ == ConfigurationParameters::UpdateVariant::STANDARD)
    {
      std::cout << "Update Variant is " << "STANDARD\n";
    }
    else if (testrun->params->update_variant_ == ConfigurationParameters::UpdateVariant::WARPSIZED)
    {
      std::cout << "Update Variant is " << "WARPSIZED\n";
    }
    else if (testrun->params->update_variant_ == ConfigurationParameters::UpdateVariant::VERTEXCENTRIC)
    {
      std::cout << "Update Variant is " << "VERTEX - CENTRIC\n";
    }
    else if (testrun->params->update_variant_ == ConfigurationParameters::UpdateVariant::VERTEXCENTRICSORTED)
    {
      std::cout << "Update Variant is " << "VERTEX - CENTRIC - SORTED\n";
    }
    std::cout  << "Page Size: " << testrun->params->page_size_ << std::endl;
    std::cout << "Rounds: " << testrun->params->rounds_ << " | Update-Rounds: " 
      << testrun->params->update_rounds_ << std::endl;
    std::cout << "Init-Launch-Block-Size: " << testrun->params->init_launch_block_size_ << std::endl;
    std::cout << "Insert-Launch-Block-Size: " << testrun->params->insert_launch_block_size_ << std::endl;
    std::cout << "Delete-Launch-Block-Size: " << testrun->params->delete_launch_block_size_ << std::endl;
    if (testrun->params->verification_)
      std::cout << "Verification: TRUE" << std::endl;
    else
      std::cout << "Verification: FALSE" << std::endl;

    if (testrun->params->sorting_)
      std::cout << "Sorting: TRUE" << std::endl;
    else
      std::cout << "Sorting: FALSE" << std::endl;

    if (testrun->params->performance_output_ == ConfigurationParameters::PerformanceOutput::STDOUT)
    {
      std::cout << "Performance Data is displayed on " << "STDOUT\n";
    }
    else if (testrun->params->performance_output_ == ConfigurationParameters::PerformanceOutput::CSV)
    {
      std::cout << "Performance Data is displayed on " << "CSV sheet.\n";
    }
    if (testrun->params->deletion_variant_ == ConfigurationParameters::DeletionVariant::STANDARD)
    {
      std::cout << "Deletion Variant is" << " STANDARD\n";
    }
    else if (testrun->params->deletion_variant_ == ConfigurationParameters::DeletionVariant::COMPACTION)
    {
      std::cout << "Deletion Variant is" << " COMPACTION\n";
    }
    std::cout << "Memory-Overallocation-Factor is " << testrun->params->memory_overallocation_factor_ << std::endl;
    std::cout << "StackSize is " << testrun->params->stacksize_ << std::endl;
    std::cout << "QueueSize is " << testrun->params->queuesize_ << std::endl;

    if (testrun->params->directionality_ == ConfigurationParameters::GraphDirectionality::DIRECTED)
    {
      std::cout << "Graph Directionality is " << "DIRECTED\n";
    }
    else
    {
      std::cout << "Graph Directionality is " << "UNDIRECTED\n";
    }

    std::cout << "+++++++++++++++ Graphs to launch" << " +++++++++++++++" << std::endl;
    for (const auto& graph : testrun->graphs)
    {
      std::cout << graph << std::endl;
    }
    std::cout << "+++++++++++++++ Batchsizes to test" << " +++++++++++++++" << std::endl;
    for (auto size : testrun->batchsizes)
    {
      std::cout << size << std::endl;
    }
  }
  std::cout << "\n" << "############## Launch-Configuration END   ##############" << "\n" << std::endl;
}
