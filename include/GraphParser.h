//------------------------------------------------------------------------------
// GraphParser.h
//
// faimGraph
//
//------------------------------------------------------------------------------
//

#pragma once

#include <string>
#include <memory>

#include "Definitions.h"

class GraphParser
{
  public:
    enum class GraphFormat
    {
      DIMACS,
      SNAP,
      MM,
      RMAT,
      UNKNOWN
    };
  public:
    explicit GraphParser(const std::string& filename, bool is_matrix = false): 
      filename_{ filename }, format_{ GraphFormat::UNKNOWN }, isMatrix{is_matrix} {}
    ~GraphParser(){}

    // Parses graph from file
    bool parseGraph(bool generateGraph = false);
    void getFreshGraph();
    bool parseDIMACSGraph();
    bool parseMMGraph();
    bool checkGraphFormat();
    bool generateGraphSynthetical();
    void generateMatrixValues();

    // Verification
    void printAdjacencyAtIndex(index_t index);

    // Getter & Setter
    vertex_t getNumberOfVertices() const {return number_vertices;}
    vertex_t getNumberOfEdges() const {return number_edges;}
    AdjacencyList_t& getAdjacency() {return adjacency_modifiable_;}
    OffsetList_t& getOffset() {return offset_modifiable_;}
    MatrixList_t& getMatrixValues() { return matrix_values_modifiable_; }
    bool isGraphMatrix() { return isMatrix; }
    std::vector<index_t>& getIndexQueue() { return index_queue; }
    
  private:
    std::string filename_;
    AdjacencyList_t adjacency_;
    OffsetList_t offset_;
    MatrixList_t matrix_values_;
    AdjacencyList_t adjacency_modifiable_;
    OffsetList_t offset_modifiable_;
    MatrixList_t matrix_values_modifiable_;
    vertex_t number_vertices;
    vertex_t number_edges;
    vertex_t highest_edge;
    GraphFormat format_;
    bool isMatrix{ false };

    std::vector<index_t> index_queue;
};
