//------------------------------------------------------------------------------
// ClusteringCoefficients.cu
//
// aimGraph
//
// Authors: Martin Winter, martin.winter@icg.tugraz.at
//------------------------------------------------------------------------------
//
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "ClusteringCoefficients.h"
#include "MemoryManager.h"
#include "EdgeUpdate.h"

//------------------------------------------------------------------------------
// Device funtionality
//------------------------------------------------------------------------------
//
//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType>
__global__ void d_clusteringCoefficients(MemoryManager* memory_manager,
                                        memory_t* memory,
                                        uint32_t* triangles,
                                        float* clustering_coefficients,
                                        int number_vertices,
                                        int page_size)
{
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid >= number_vertices)
    return;

  VertexDataType* vertices = (VertexDataType*)memory;
  VertexDataType vertex = vertices[tid];

  clustering_coefficients[tid] = static_cast<float>(triangles[tid]) / (vertex.neighbours * (vertex.neighbours - 1));

  return;
}


//------------------------------------------------------------------------------
// Host funtionality
//------------------------------------------------------------------------------
//

//------------------------------------------------------------------------------
//
template <typename VertexDataType, typename EdgeDataType>
float ClusteringCoefficients<VertexDataType, EdgeDataType>::computeClusteringCoefficients(std::unique_ptr<MemoryManager>& memory_manager, bool global_CC_count)
{
  int block_size = KERNEL_LAUNCH_BLOCK_SIZE_STANDARD;
  int grid_size = (memory_manager->next_free_vertex_index / block_size) + 1;
  float clustering_coefficient = 0.0f;
  
  // Compute triangle count
  stc->StaticTriangleCounting(memory_manager, false);

  // Compute clustering coefficients
  d_clusteringCoefficients<VertexDataType, EdgeDataType> << <grid_size, block_size >> > ((MemoryManager*)memory_manager->d_memory,
                                                                                          memory_manager->d_data,
                                                                                          stc->d_triangles,
                                                                                          d_clustering_coefficients,
                                                                                          memory_manager->next_free_vertex_index,
                                                                                          memory_manager->page_size);

  if (global_CC_count)
  {
    // Prefix scan on d_triangles to get number of triangles
    thrust::device_ptr<float> th_clustering_coefficients(d_clustering_coefficients);
    thrust::device_ptr<float> th_clustering_coefficients_count(d_clustering_coefficients_count);
    thrust::inclusive_scan(th_clustering_coefficients, th_clustering_coefficients + memory_manager->next_free_vertex_index, th_clustering_coefficients_count);

    // Copy result back to host
    HANDLE_ERROR(cudaMemcpy(clustering_coefficients.get(),
                            d_clustering_coefficients,
                            sizeof(float) * memory_manager->next_free_vertex_index,
                            cudaMemcpyDeviceToHost));

    // Copy number of triangles back
    HANDLE_ERROR(cudaMemcpy(&clustering_coefficient,
                            d_clustering_coefficients + (memory_manager->next_free_vertex_index - 1),
                            sizeof(float),
                            cudaMemcpyDeviceToHost));
  }
  
  return clustering_coefficient;
}

template float ClusteringCoefficients<VertexData, EdgeData>::computeClusteringCoefficients(std::unique_ptr<MemoryManager>& memory_manager, bool global_CC_count);
template float ClusteringCoefficients<VertexDataWeight, EdgeDataWeight>::computeClusteringCoefficients(std::unique_ptr<MemoryManager>& memory_manager, bool global_CC_count);
template float ClusteringCoefficients<VertexDataSemantic, EdgeDataSemantic>::computeClusteringCoefficients(std::unique_ptr<MemoryManager>& memory_manager, bool global_CC_count);
template float ClusteringCoefficients<VertexData, EdgeDataSOA>::computeClusteringCoefficients(std::unique_ptr<MemoryManager>& memory_manager, bool global_CC_count);
template float ClusteringCoefficients<VertexDataWeight, EdgeDataWeightSOA>::computeClusteringCoefficients(std::unique_ptr<MemoryManager>& memory_manager, bool global_CC_count);
template float ClusteringCoefficients<VertexDataSemantic, EdgeDataSemanticSOA>::computeClusteringCoefficients(std::unique_ptr<MemoryManager>& memory_manager, bool global_CC_count);