//------------------------------------------------------------------------------
// MemoryLayout.h
//
// faimGraph
//
//------------------------------------------------------------------------------
//

#pragma once

//------------------------------------------------------------------------------
// EdgeData Variants for simple, with weight or semantic graphs AOS
//------------------------------------------------------------------------------
//

struct EdgeDataUpdate;

struct EdgeData
{
  vertex_t destination;
  friend __host__ __device__ bool operator<(const EdgeData &lhs, const EdgeData &rhs) { return (lhs.destination < rhs.destination); }
  static size_t sizeOfUpdateData() { return sizeof(vertex_t) + sizeof(vertex_t); }
};

struct EdgeDataWeight
{
  vertex_t destination;
  vertex_t weight;
  friend __host__ __device__ bool operator<(const EdgeDataWeight &lhs, const EdgeDataWeight &rhs) { return (lhs.destination < rhs.destination); };
  static size_t sizeOfUpdateData() { return sizeof(vertex_t) + sizeof(vertex_t) + sizeof(vertex_t); }
};

struct EdgeDataSemantic
{
  vertex_t destination;
  vertex_t weight;
  vertex_t type;
  vertex_t timestamp_1;
  vertex_t timestamp_2;
  friend __host__ __device__ bool operator<(const EdgeDataSemantic &lhs, const EdgeDataSemantic &rhs) { return (lhs.destination < rhs.destination); };
  static size_t sizeOfUpdateData() { return sizeof(vertex_t) + sizeof(vertex_t) + sizeof(vertex_t) + sizeof(vertex_t) + sizeof(vertex_t) + sizeof(vertex_t); }
};

struct EdgeDataMatrix
{
  vertex_t destination;
  matrix_t matrix_value;
  friend __host__ __device__ bool operator<(const EdgeDataMatrix &lhs, const EdgeDataMatrix &rhs) { return (lhs.destination < rhs.destination); };
  static size_t sizeOfUpdateData() { return sizeof(vertex_t) + sizeof(vertex_t) + sizeof(matrix_t); }
};


//------------------------------------------------------------------------------
// EdgeData Variants for simple, with weight or semantic graphs SOA
//------------------------------------------------------------------------------
//

struct EdgeDataSOA
{
	vertex_t destination;
	static size_t sizeOfUpdateData() { return sizeof(vertex_t) + sizeof(vertex_t); }
};

struct EdgeDataWeightSOA
{
	vertex_t destination;
	static size_t sizeOfUpdateData() { return sizeof(vertex_t) + sizeof(vertex_t) + sizeof(vertex_t); }
};

struct EdgeDataSemanticSOA
{
	vertex_t destination;
	static size_t sizeOfUpdateData() { return sizeof(vertex_t) + sizeof(vertex_t) + sizeof(vertex_t) + sizeof(vertex_t) + sizeof(vertex_t) + sizeof(vertex_t); }
};

struct EdgeDataMatrixSOA
{
  vertex_t destination;
  static size_t sizeOfUpdateData() { return sizeof(vertex_t) + sizeof(vertex_t) + sizeof(matrix_t); }
};

//------------------------------------------------------------------------------
// EdgeUpdate Variants for simple, with weight or semantic graphs
//------------------------------------------------------------------------------
//

struct EdgeDataUpdate
{
	vertex_t source;
	EdgeData update;
  friend __host__ __device__ bool operator<(const EdgeDataUpdate &lhs, const EdgeDataUpdate &rhs) 
  { 
    return ((lhs.source < rhs.source) || (lhs.source == rhs.source && (lhs.update < rhs.update))); 
  }
};


struct EdgeDataWeightUpdate
{
	vertex_t source;
	EdgeDataWeight update;
  friend __host__ __device__ bool operator<(const EdgeDataWeightUpdate &lhs, const EdgeDataWeightUpdate &rhs) 
  { 
    return ((lhs.source < rhs.source) || (lhs.source == rhs.source && (lhs.update < rhs.update))); 
  };
};


struct EdgeDataSemanticUpdate
{
	vertex_t source;
	EdgeDataSemantic update;
  friend __host__ __device__ bool operator<(const EdgeDataSemanticUpdate &lhs, const EdgeDataSemanticUpdate &rhs) 
  { 
    return ((lhs.source < rhs.source) || (lhs.source == rhs.source && (lhs.update < rhs.update))); 
  };
};

struct EdgeDataMatrixUpdate
{
  vertex_t source;
  EdgeDataMatrix update;
  friend __host__ __device__ bool operator<(const EdgeDataMatrixUpdate &lhs, const EdgeDataMatrixUpdate &rhs)
  {
    return ((lhs.source < rhs.source) || (lhs.source == rhs.source && (lhs.update < rhs.update)));
  };
};

//------------------------------------------------------------------------------
// VertexData Variants for simple, with weight or semantic graphs
//------------------------------------------------------------------------------
//

typedef struct VertexData
{
  int locking;
  vertex_t mem_index;
  vertex_t neighbours;
  vertex_t capacity;
  index_t host_identifier;
}VertexData;

typedef struct VertexDataWeight
{
  int locking;
  vertex_t mem_index;
  vertex_t neighbours;
  vertex_t capacity;
  index_t host_identifier;
  vertex_t weight;
}VertexDataWeight;

typedef struct VertexDataSemantic
{
  int locking;
  vertex_t mem_index;
  vertex_t neighbours;
  vertex_t capacity;
  index_t host_identifier;
  vertex_t weight;
  vertex_t type;
}VertexDataSemantic;

//------------------------------------------------------------------------------
// VertexUpdate Variants for simple, with weight or semantic graphs
//------------------------------------------------------------------------------
//

typedef struct VertexUpdate
{
  index_t identifier;
}VertexUpdate;

__forceinline__ __host__ __device__ bool operator<(const VertexUpdate &lhs, const VertexUpdate &rhs) { return (lhs.identifier < rhs.identifier); };

typedef struct VertexUpdateWeight
{
  index_t identifier;
  vertex_t weight;
}VertexUpdateWeight;

__forceinline__ __host__ __device__ bool operator<(const VertexUpdateWeight &lhs, const VertexUpdateWeight &rhs) { return (lhs.identifier < rhs.identifier); };

typedef struct VertexUpdateSemantic
{
  index_t identifier;
  vertex_t weight;
  vertex_t type;
}VertexUpdateSemantic;

__forceinline__ __host__ __device__ bool operator<(const VertexUpdateSemantic &lhs, const VertexUpdateSemantic &rhs) { return (lhs.identifier < rhs.identifier); };


//------------------------------------------------------------------------------
// CSR Structure for matrices
//------------------------------------------------------------------------------
//
class CSRMatrix
{
public:
  OffsetList_t offset;
  AdjacencyList_t adjacency;
  MatrixList_t matrix_values;
};
