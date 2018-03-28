cmake_minimum_required(VERSION 3.2)
project(aimgraph)

if (UNIX)
set(CUDA_HOST_COMPILER /usr/bin/gcc-6)
endif (UNIX)

FIND_PACKAGE(CUDA REQUIRED)

INCLUDE(FindCUDA)

option(CUDA_BUILD_CC20 "Build with compute capability 2.0 support" FALSE)
option(CUDA_BUILD_CC21 "Build with compute capability 2.1 support" FALSE)
option(CUDA_BUILD_CC30 "Build with compute capability 3.0 support" FALSE)
option(CUDA_BUILD_CC35 "Build with compute capability 3.5 support" FALSE)
option(CUDA_BUILD_CC50 "Build with compute capability 5.0 support" FALSE)
option(CUDA_BUILD_CC52 "Build with compute capability 5.2 support" FALSE)
option(CUDA_BUILD_CC61 "Build with compute capability 6.1 support" TRUE)
option(CUDA_BUILD_INFO "Build with kernel statistics and line numbers" TRUE)
option(CUDA_BUILD_DEBUG "Build with kernel debug" FALSE)

INCLUDE_DIRECTORIES(/usr/local/cuda/include)
INCLUDE_DIRECTORIES(include)
INCLUDE_DIRECTORIES(include/algorithms)
   
if(WIN32)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
SET(CUDA_PROPAGATE_HOST_FLAGS ON)
else()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
# SET(CUDA_PROPAGATE_HOST_FLAGS ON)
endif()

# set(CMAKE_BUILD_TYPE Debug)
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -lineinfo -G -DTHRUST_DEBUG")

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4464 /wd4514 /wd4820 /wd4668 /wd4574 /wd4571 /wd4324 /wd4710 /wd4711 /wd4365 /wd4515 /wd4201 /wd4267 /wd5027 /wd4626")
endif()

LIST(APPEND CUDA_NCCC_FLAGS --compiler-options)
if (CUDA_BUILD_CC20)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode=arch=compute_20,code=sm_20;")
endif ()
if (CUDA_BUILD_CC21)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode=arch=compute_20,code=sm_21;")
endif ()
if (CUDA_BUILD_CC30)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode=arch=compute_30,code=sm_30;")
endif ()
if(CUDA_BUILD_CC35)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode=arch=compute_35,code=sm_35;")
endif ()
if (CUDA_BUILD_CC50)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode=arch=compute_50,code=sm_50;")
endif ()
if (CUDA_BUILD_CC52)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode=arch=compute_52,code=sm_52;")
endif ()
if (CUDA_BUILD_CC61)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode=arch=compute_61,code=sm_61;")
endif ()


LIST(APPEND CUDA_NVCC_FLAGS "-DVERBOSE;")
LIST(APPEND CUDA_NVCC_FLAGS "-Xcompiler -Wall -D_FORCE_INLINES; --expt-extended-lambda;-use_fast_math")

if (CUDA_BUILD_INFO)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-keep;--ptxas-options=-v;-lineinfo")
endif ()

if (CUDA_BUILD_DEBUG)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-G")
endif ()

cuda_add_library(aimgraph
          			src/ConfigurationParser.cpp
					src/CSVWriter.cpp
					src/MemoryManager.cpp
					src/MemoryManager.cu
					src/GraphParser.cpp
					src/Utility.cpp
					src/Utility.cu
					src/aimGraph.cpp
					src/aimGraph.cu
					src/EdgeUpdate.cpp
					src/EdgeUpdate.cu
					src/EdgeQuery.cu
					src/EdgeInsertion.cu
					src/EdgeDeletion.cu
					src/EdgeUtility.cu
					src/VertexInsertion.cu
					src/VertexDeletion.cu
					src/VertexUtility.cu
					src/VertexUpdate.cpp
					src/SpMV.cu
					src/SpMM.cu
					src/algorithms/StaticTriangleCounting.cu
					src/algorithms/STC.cu
					src/algorithms/PageRank.cu
					src/algorithms/ClusteringCoefficients.cu
					src/algorithms/BetweennessCentrality.cu
					src/algorithms/ConnectedComponents.cu
        )

cuda_add_executable(mainaimGraph 
          src/main.cpp
           )

cuda_add_executable(STCaimGraph 
          src/algorithms/StaticTriangleCounting.cpp
           )

cuda_add_executable(ClusteringCoefficients 
          src/algorithms/ClusteringCoefficients.cpp
           )

cuda_add_executable(ConnectedComponents 
          src/algorithms/ConnectedComponents.cpp
           )

cuda_add_executable(BetweennessCentrality	 
          src/algorithms/BetweennessCentrality.cpp
           )

cuda_add_executable(PageRankaimGraph 
          src/algorithms/PageRank.cpp
           )

cuda_add_executable(continuousTCaimGraph 
          src/tests/ContinuousTC.cpp
           )

cuda_add_executable(dynamicVerticesMain 
          src/tests/dynamicVerticesMain.cpp
           )

cuda_add_executable(concurrentTCaimGraph 
          src/tests/concurrentTC.cpp
           )

cuda_add_executable(queryTCaimGraph 
          src/tests/queryTC.cpp
           )

cuda_add_executable(spmvaimGraph 
          src/tests/SpMV.cpp
           )


TARGET_LINK_LIBRARIES(mainaimGraph aimgraph ${CUDA_cudart_static_LIBRARY})
TARGET_LINK_LIBRARIES(STCaimGraph aimgraph  ${CUDA_cudart_static_LIBRARY})
TARGET_LINK_LIBRARIES(ClusteringCoefficients aimgraph  ${CUDA_cudart_static_LIBRARY})
TARGET_LINK_LIBRARIES(ConnectedComponents aimgraph  ${CUDA_cudart_static_LIBRARY})
TARGET_LINK_LIBRARIES(BetweennessCentrality aimgraph  ${CUDA_cudart_static_LIBRARY})
TARGET_LINK_LIBRARIES(PageRankaimGraph aimgraph  ${CUDA_cudart_static_LIBRARY})
TARGET_LINK_LIBRARIES(continuousTCaimGraph aimgraph  ${CUDA_cudart_static_LIBRARY})
TARGET_LINK_LIBRARIES(dynamicVerticesMain aimgraph  ${CUDA_cudart_static_LIBRARY})
TARGET_LINK_LIBRARIES(concurrentTCaimGraph aimgraph  ${CUDA_cudart_static_LIBRARY})
TARGET_LINK_LIBRARIES(queryTCaimGraph aimgraph  ${CUDA_cudart_static_LIBRARY})
TARGET_LINK_LIBRARIES(spmvaimGraph aimgraph  ${CUDA_cudart_static_LIBRARY})