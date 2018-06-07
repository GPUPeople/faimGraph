//------------------------------------------------------------------------------
// GraphParser.cpp
//
// faimGraph
//
//------------------------------------------------------------------------------
//
#include <fstream>
#include <sstream>
#include <iostream>

#include "GraphParser.h"
#include "CSR.h"
#include "COO.h"
#include "Vector.h"
#include "dCSR.h"

//------------------------------------------------------------------------------
bool GraphParser::parseGraph(bool generateGraph)
{
    std::cout << "parseGraph called with " + filename_ << std::endl;

    if(generateGraph)
        return generateGraphSynthetical();

    // First check for valid graph format
    if(!checkGraphFormat())
        return false;    
    
    if (format_ == GraphFormat::DIMACS)
      return parseDIMACSGraph();
    else if (format_ == GraphFormat::MM)
      return parseMMGraph();
    
}

//------------------------------------------------------------------------------
//
bool GraphParser::generateGraphSynthetical()
{
    srand(100);
    number_vertices = 80*1000*1000;
    vertex_t max_number_edges_per_adjacency = 10;
    vertex_t number_edges_per_adjacency;
    vertex_t vertex_index = 0;
    for(unsigned int i = 0; i < number_vertices; ++i)
    {
        offset_.push_back(vertex_index);
        number_edges_per_adjacency = (rand() % max_number_edges_per_adjacency) + 1;
        for(unsigned int j = 0; j < number_edges_per_adjacency; ++j)
        {
            adjacency_.push_back(rand() % number_vertices);
            ++vertex_index;
        }
    }
    offset_.push_back(vertex_index);
    number_edges = adjacency_.size();
    std::cout << "#v: " << number_vertices << " and #e: " << number_edges << std::endl;
    getFreshGraph();
    return true;
}

//------------------------------------------------------------------------------
//
bool GraphParser::parseDIMACSGraph()
{
    // Open file and iterate over it line by line
    std::ifstream graph_file(filename_);
    std::string line;
    vertex_t index;
    vertex_t vertex_index = 0;
    highest_edge = 0;

    if (!graph_file.is_open())
    {
      std::cout << "File does not exist" << std::endl;
      return false;
    }

    /* Graph starts with #vertices #edges
    *  after that always the adjacency list of each vertex
    */

    // Overstep comments and parse #v and #e 
    while(std::getline(graph_file, line))
    {
        std::istringstream istream(line);
        if(istream >> number_vertices)
        {
            // found first non-comment, we got #v and #e
            istream >> number_edges;
            break;            
        }
    }

    // Parse adjacency list
    while(std::getline(graph_file, line))
    {
        offset_.push_back(vertex_index);
        std::istringstream istream(line);
        while(istream >> index)
        {
            // Graph format uses 1-n, we would like to have 0 - (n-1)
            adjacency_.push_back(index - 1);
            ++vertex_index;
            if (index > highest_edge)
            {
              highest_edge = index;
            }              
        }
    }
    // Also include the offset for the #v+1 element (needed for a calculation later)
    offset_.push_back(vertex_index);
    number_edges = adjacency_.size();
    std::cout << "#v: " << number_vertices << " and #e: " << number_edges << " and highest edge: " << highest_edge <<std::endl;
	  //std::cout << "End parsing Graph!" << std::endl;
    if (isMatrix)
    {
      generateMatrixValues();
    }
    getFreshGraph();
    return true;
}

//------------------------------------------------------------------------------
//
//#define READ_REAL_MATRIX_VALUES
bool GraphParser::parseMMGraph()
{
	CSR<float> csr_mat;
	std::string csr_name = filename_ + ".csr";
	try
	{
		std::cout << "trying to load csr file \"" << csr_name << "\"\n";
		csr_mat = loadCSR<float>(csr_name.c_str());
		std::cout << "succesfully loaded: \"" << csr_name << "\"\n";
	}
	catch (std::exception& ex)
	{
		std::cout << "could not load csr file:\n\t" << ex.what() << "\n";
		try
		{
			std::cout << "trying to load mtx file \"" << filename_ << "\"\n";
			COO<float> coo_mat = loadMTX<float>(filename_.c_str());
			convert(csr_mat, coo_mat);
			std::cout << "succesfully loaded and converted: \"" << csr_name << "\"\n";
		}
		catch (std::exception& ex)
		{
			std::cout << ex.what() << std::endl;
			return -1;
		}
		try
		{
			std::cout << "write csr file for future use\n";
			storeCSR(csr_mat, csr_name.c_str());
		}
		catch (std::exception& ex)
		{
			std::cout << ex.what() << std::endl;
		}
	}

	// Transfer data to graph parser
	number_edges = csr_mat.nnz;
	number_vertices = csr_mat.rows;
	offset_.resize(number_vertices + 1);
	adjacency_.resize(number_edges);
	matrix_values_.resize(number_edges);
	memcpy(&offset_[0], csr_mat.row_offsets.get(), sizeof(vertex_t) * (number_vertices + 1));
	memcpy(&adjacency_[0], csr_mat.col_ids.get(), sizeof(vertex_t) * (number_edges));
	memcpy(&matrix_values_[0], csr_mat.data.get(), sizeof(float) * (number_edges));

	std::cout << "#v: " << number_vertices << " and #e: " << number_edges << std::endl;

	getFreshGraph();
	return true;
}

//------------------------------------------------------------------------------
//
void GraphParser::generateMatrixValues()
{
  srand(1);
  for (int i = 0; i < adjacency_.size(); ++i)
  {
    matrix_t rand_val = rand() % 10;
    matrix_values_.push_back(rand_val);
  }
}

//------------------------------------------------------------------------------
//
void GraphParser::getFreshGraph()
{
    adjacency_modifiable_ = adjacency_;
    offset_modifiable_ = offset_;
    matrix_values_modifiable_ = matrix_values_;
}

//------------------------------------------------------------------------------
//
bool GraphParser::checkGraphFormat()
{
    if(filename_.find(".graph") != std::string::npos)
    {
        format_ = GraphFormat::DIMACS;
        //std::cout << "Graph format is DIMACS!" << std::endl;
        return true;
    }
    else if(filename_.find(".txt") != std::string::npos)
    {
        format_ = GraphFormat::SNAP;
        std::cout << "Graph format SNAP is currently not supported!"<< std::endl;
        return false;
    }
    else if(filename_.find(".mtx") != std::string::npos)
    {
        format_ = GraphFormat::MM;
        /*std::cout << "Graph format MM is currently not supported!" << std::endl;*/
        return true;
    }
    else if(filename_.find(".kron") != std::string::npos)
    {
        format_ = GraphFormat::RMAT;
        std::cout << "Graph format RMAT is currently not supported!"<< std::endl;
        return false;
    }
    else
    {
        format_ = GraphFormat::UNKNOWN;
        std::cout << "Invalid Format" << std::endl;
        return false;
    }
}

//------------------------------------------------------------------------------
//
void GraphParser::printAdjacencyAtIndex(index_t index)
{
  index_t start_index = offset_.at(index);
  index_t end_index = offset_.at(index + 1);
  std::cout << "Print adjacency for index " << index << std::endl;
  for(size_t i = start_index; i < end_index; ++i)
  {
    std::cout << adjacency_.at(i) << " | ";
  }
  std::cout << std::endl;
}