#include "PageRank.cuh"
#include "STC.cuh"
#include "ClusteringCoefficients.cuh"
#include "faimGraph.cuh"
#include "EdgeInsertion.cuh"
#include "EdgeDeletion.cuh"
#include "EdgeQuery.cuh"
#include "MemoryManager.cuh"
#include "VertexInsertion.cuh"
#include "VertexDeletion.cuh"
#include "EdgeUpdateConcurrent.cuh"
#include "EdgeUtility.cuh"

//------------------------------------------------------------------------------
// faimGraph
//------------------------------------------------------------------------------
#define FAIMGRAPH(VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE) \
template void faimGraph<VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE>::initializeMemory(std::unique_ptr<GraphParser>& graph_parser); \
template CSR<float> faimGraph <VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE>::reinitializeFaimGraph(float overallocation_factor); \
template std::unique_ptr<aimGraphCSR> faimGraph <VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE>::verifyGraphStructure (std::unique_ptr<MemoryManager>& memory_manager); \
template void faimGraph<VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE>::initializeMemory(vertex_t* d_offset, vertex_t* d_adjacency, int number_vertices); \
template bool faimGraph<VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE>::compareGraphs(std::unique_ptr<GraphParser>& graph_parser, std::unique_ptr<aimGraphCSR>& verify_graph, bool duplicate_check);

#define FAIMGRAPHMATRIX(VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE) \
template void faimGraph<VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE>::initializefaimGraphMatrix(std::unique_ptr<GraphParser>& graph_parser, unsigned int vertex_offset); \
template void faimGraph<VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE>::initializefaimGraphEmptyMatrix(unsigned int number_rows, unsigned int vertex_offset); \
template void faimGraph<VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE>::initializefaimGraphMatrix(std::unique_ptr<CSRMatrixData>& csr_matrix_data); \
template std::unique_ptr<aimGraphCSR> faimGraph < VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE > ::verifyMatrixStructure(std::unique_ptr<MemoryManager>& memory_manager, vertex_t vertex_offset, vertex_t number_vertices); \
template bool faimGraph<VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE>::compareGraphs(std::unique_ptr<CSRMatrix>& csr_matrix, std::unique_ptr<aimGraphCSR>& verify_graph, bool duplicate_check);

#define FAIMGRAPHMEMMAN(VERTEXDATATYPE) \
template size_t MemoryManager::numberEdgesInMemory<VERTEXDATATYPE>(vertex_t* d_neighbours_count, bool return_count); \
template size_t MemoryManager::numberEdgesInMemory<VERTEXDATATYPE>(vertex_t* d_neighbours_count, vertex_t vertex_offset, vertex_t number_vertices, bool return_count); \
template void MemoryManager::numberPagesInMemory<VERTEXDATATYPE>(vertex_t* d_page_count); \
template size_t MemoryManager::numberPagesInMemory<VERTEXDATATYPE>(vertex_t* d_page_count, vertex_t* d_accumulated_page_count);

//------------------------------------------------------------------------------
// faimGraph Edge Updates
//------------------------------------------------------------------------------
#define FAIMGRAPHEDGEUTILITY(VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE) \
template std::unique_ptr<EdgeUpdatePreProcessing<EDGEUPATETYPE>> EdgeUpdateManager<VERTEXDATATYPE, EDGEDATATYPE, EDGEUPATETYPE>::edgeUpdatePreprocessing(std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config); \
template std::unique_ptr<EdgeUpdatePreProcessing<EdgeDataUpdate>> EdgeQueryManager<VERTEXDATATYPE, EDGEDATATYPE>::edgeQueryPreprocessing(std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config); \
template void EdgeUpdateManager<VERTEXDATATYPE, EDGEDATATYPE, EDGEUPATETYPE>::edgeUpdateDuplicateChecking(std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config, const std::unique_ptr<EdgeUpdatePreProcessing<EDGEUPATETYPE>>& preprocessed); \
template std::unique_ptr<EdgeUpdateBatch<EDGEUPATETYPE>> EdgeUpdateManager<VERTEXDATATYPE, EDGEDATATYPE, EDGEUPATETYPE>::generateEdgeUpdates (const std::unique_ptr<MemoryManager>& memory_manager, vertex_t batch_size, unsigned int seed, unsigned int range, unsigned int offset); \
template std::unique_ptr<EdgeUpdateBatch<EDGEUPATETYPE>> EdgeUpdateManager<VERTEXDATATYPE, EDGEDATATYPE, EDGEUPATETYPE>::generateEdgeUpdatesConcurrent <VERTEXUPDATETYPE>(std::unique_ptr<faimGraph<VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE>>& faimGraph, const std::unique_ptr<MemoryManager>& memory_manager, vertex_t batch_size, unsigned int seed, unsigned int range, unsigned int offset); \
template void MemoryManager::compaction<VERTEXDATATYPE, EDGEDATATYPE>(const std::shared_ptr<Config>& config); \
template void MemoryManager::sortAdjacency<VERTEXDATATYPE, EDGEDATATYPE>(const std::shared_ptr<Config>& config, SortOrder sort_order); \
template void MemoryManager::testUndirectedness<VERTEXDATATYPE, EDGEDATATYPE>(const std::shared_ptr<Config>& config); \
template void MemoryManager::testSelfLoops<VERTEXDATATYPE, EDGEDATATYPE>(const std::shared_ptr<Config>& config); \
template void MemoryManager::resetAllocationStatus<VERTEXDATATYPE, EDGEDATATYPE>(const std::shared_ptr<Config>& config, vertex_t number_vertices, vertex_t vertex_offset); \
template void MemoryManager::testDuplicates<VERTEXDATATYPE, EDGEDATATYPE>(const std::shared_ptr<Config>& config);

#define FAIMGRAPHEDGEUTILITYMATRIX(VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE) \
template void MemoryManager::resetAllocationStatus<VERTEXDATATYPE, EDGEDATATYPE>(const std::shared_ptr<Config>& config, vertex_t number_vertices, vertex_t vertex_offset);

//------------------------------------------------------------------------------
// faimGraph Edge Updates
//------------------------------------------------------------------------------
#define FAIMGRAPHEDGEUPDATES(VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE) \
template void EdgeUpdateManager<VERTEXDATATYPE, EDGEDATATYPE, EDGEUPATETYPE>::w_edgeInsertion(cudaStream_t& stream, const std::unique_ptr<EdgeUpdateBatch<EDGEUPATETYPE>>& updates_insertion, std::unique_ptr<MemoryManager>& memory_manager, int batch_size, int block_size, int grid_size); \
template void EdgeUpdateManager<VERTEXDATATYPE, EDGEDATATYPE, EDGEUPATETYPE>::w_edgeDeletion(cudaStream_t& stream, const std::unique_ptr<EdgeUpdateBatch<EDGEUPATETYPE>>& updates_deletion, std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config, int batch_size, int block_size, int grid_size); \
template void faimGraph<VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE>::edgeInsertion(); \
template void faimGraph<VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE>::edgeDeletion(); \
template void EdgeUpdateManager<VERTEXDATATYPE, EDGEDATATYPE, EDGEUPATETYPE>::deviceEdgeInsertion (std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config); \
template void EdgeUpdateManager<VERTEXDATATYPE, EDGEDATATYPE, EDGEUPATETYPE>::deviceEdgeDeletion (std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config); \
template void EdgeUpdateManager<VERTEXDATATYPE, EDGEDATATYPE, EDGEUPATETYPE>::deviceEdgeUpdateConcurrentStream (cudaStream_t& insertion_stream, cudaStream_t& deletion_stream, std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config); \
template void EdgeUpdateManager<VERTEXDATATYPE, EDGEDATATYPE, EDGEUPATETYPE>::deviceEdgeUpdateConcurrent (std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config);

//------------------------------------------------------------------------------
// faimGraph Vertex Updates
//------------------------------------------------------------------------------
#define FAIMGRAPHVERTEXUPDATESALL(VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE) \
template void VertexUpdateManager<VERTEXDATATYPE, VERTEXUPDATETYPE>::deviceVertexInsertion<EDGEDATATYPE>(std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config, VertexMapper<index_t, index_t>& mapper, bool duplicate_checking); \
template void faimGraph<VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE>::vertexInsertion(VertexMapper<index_t, index_t>& mapper); \
template void VertexUpdateManager<VERTEXDATATYPE, VERTEXUPDATETYPE>::deviceVertexDeletion<EDGEDATATYPE>(std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config, VertexMapper<index_t, index_t>& mapper); \
template void faimGraph<VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE>::vertexDeletion(VertexMapper<index_t, index_t>& mapper);

#define FAIMGRAPHVERTEXUPDATES(VERTEXDATATYPE, VERTEXUPDATETYPE) \
template void VertexUpdateManager<VERTEXDATATYPE, VERTEXUPDATETYPE>::duplicateInBatchChecking(const std::shared_ptr<Config>& config); \
template void VertexUpdateManager<VERTEXDATATYPE, VERTEXUPDATETYPE>::duplicateInGraphChecking(const std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config);

//------------------------------------------------------------------------------
// faimGraph Edge Queries
//------------------------------------------------------------------------------
#define FAIMGRAPHEDGEQUERIES(VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE) \
template void EdgeQueryManager<VERTEXDATATYPE, EDGEDATATYPE>::deviceQuery(std::unique_ptr<MemoryManager>& memory_manager, const std::shared_ptr<Config>& config); \
template void EdgeQueryManager<VERTEXDATATYPE, EDGEDATATYPE>::generateQueries(vertex_t number_vertices, vertex_t batch_size, unsigned int seed, unsigned int range, unsigned int offset);

//------------------------------------------------------------------------------
// PAGERANK
//------------------------------------------------------------------------------
#define PAGERANK(VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE) \
template float PageRank<VERTEXDATATYPE, EDGEDATATYPE>::algPageRankNaive (const std::unique_ptr<MemoryManager>& memory_manager); \
template float PageRank<VERTEXDATATYPE, EDGEDATATYPE>::algPageRankBalanced (const std::unique_ptr<MemoryManager>& memory_manager);

//------------------------------------------------------------------------------
// STC
//------------------------------------------------------------------------------
#define STC(VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE) \
template uint32_t STC<VERTEXDATATYPE, EDGEDATATYPE>::StaticTriangleCounting (const std::unique_ptr<MemoryManager>& memory_manager, bool global_TC_count);

//------------------------------------------------------------------------------
// Clustering Coefficients
//------------------------------------------------------------------------------
#define CC(VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE) \
template float ClusteringCoefficients<VERTEXDATATYPE, EDGEDATATYPE>::computeClusteringCoefficients(std::unique_ptr<MemoryManager>& memory_manager, bool global_CC_count);

//------------------------------------------------------------------------------
// Instantiation helper
//------------------------------------------------------------------------------
#define INSTANTIATE_TEMPLATES(VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE) \
PAGERANK(VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE) \
STC(VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE) \
CC(VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE) \
FAIMGRAPH(VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE) \
FAIMGRAPHEDGEUPDATES(VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE) \
FAIMGRAPHEDGEQUERIES(VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE) \
FAIMGRAPHVERTEXUPDATESALL(VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE) \
FAIMGRAPHEDGEUTILITY(VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE)

#define INSTANTIATE_MATRIX_TEMPLATES(VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE) \
FAIMGRAPHMATRIX(VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE) \
FAIMGRAPHEDGEUTILITYMATRIX(VERTEXDATATYPE, VERTEXUPDATETYPE, EDGEDATATYPE, EDGEUPATETYPE)

#define INSTANTIATE_VERTEX_TEMPLATES(VERTEXDATATYPE, VERTEXUPDATETYPE) \
FAIMGRAPHMEMMAN(VERTEXDATATYPE) \
FAIMGRAPHVERTEXUPDATES(VERTEXDATATYPE, VERTEXUPDATETYPE)

//------------------------------------------------------------------------------
// Explicit Instantiations
//------------------------------------------------------------------------------
INSTANTIATE_TEMPLATES(VertexData, VertexUpdate, EdgeData, EdgeDataUpdate)
INSTANTIATE_TEMPLATES(VertexDataWeight, VertexUpdateWeight, EdgeDataWeight, EdgeDataWeightUpdate)
INSTANTIATE_TEMPLATES(VertexDataSemantic, VertexUpdateSemantic, EdgeDataSemantic, EdgeDataSemanticUpdate)
INSTANTIATE_TEMPLATES(VertexData, VertexUpdate, EdgeDataSOA, EdgeDataUpdate)
INSTANTIATE_TEMPLATES(VertexDataWeight, VertexUpdateWeight, EdgeDataWeightSOA, EdgeDataWeightUpdate)
INSTANTIATE_TEMPLATES(VertexDataSemantic, VertexUpdateSemantic, EdgeDataSemanticSOA, EdgeDataSemanticUpdate)
INSTANTIATE_MATRIX_TEMPLATES(VertexData, VertexUpdate, EdgeDataMatrix, EdgeDataUpdate)
INSTANTIATE_MATRIX_TEMPLATES(VertexData, VertexUpdate, EdgeDataMatrixSOA, EdgeDataUpdate)
INSTANTIATE_VERTEX_TEMPLATES(VertexData, VertexUpdate)
INSTANTIATE_VERTEX_TEMPLATES(VertexDataWeight, VertexUpdateWeight)
INSTANTIATE_VERTEX_TEMPLATES(VertexDataSemantic, VertexUpdateSemantic)