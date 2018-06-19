cmake_minimum_required(VERSION 3.2)
project(faimgraph)

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

SET(FAIMGRAPHHEADERS
	include/ConfigurationParser.h
	include/COO.h 
	include/CSR.h 
	include/CSVWriter.h 
	include/dCSR.h 
	include/Definitions.h 
	include/EdgeUpdate.h 
	include/EdgeInsertion.cuh 
	include/EdgeDeletion.cuh 
	include/EdgeQuery.cuh 
	include/EdgeUtility.cuh 
	include/EdgeUpdateConcurrent.cuh 
	include/faimGraph.h 
	include/faimGraph.cuh 
	include/GraphParser.h 
	include/MemoryLayout.h 
	include/MemoryManager.h 
	include/MemoryManager.cuh 
	include/Queue.h 
	include/SpMM.h 
	include/SpMV.h 
	include/Utility.h 
	include/Vector.h 
	include/VertexInsertion.cuh 
	include/VertexDeletion.cuh 
	include/VertexMapper.h 
	include/VertexUpdate.h 
)

cuda_add_library(faimgraph
					src/COO.cpp
					src/CSR.cpp
					src/dCSR.cpp
          			src/ConfigurationParser.cpp
					src/CSVWriter.cpp
					src/MemoryManager.cpp
					src/GraphParser.cpp
					src/Utility.cpp
					src/faimGraph.cpp
					src/EdgeUpdate.cpp
					src/VertexUpdate.cpp
					src/SpMV.cu
					src/SpMM.cu
					src/Instantiations.cu
					${FAIMGRAPHHEADERS}
        )

cuda_add_executable(mainfaimGraph 
          src/main.cpp
           )

cuda_add_executable(reInitTC 
          src/tests/reInitTC.cpp
           )

cuda_add_executable(STC 
          src/algorithms/STC.cpp
		  include/algorithms/STC.h
		  include/algorithms/STC.cuh
           )

cuda_add_executable(CCoeff
          src/algorithms/ClusteringCoefficients.cpp
		  include/algorithms/ClusteringCoefficients.h 
		  include/algorithms/ClusteringCoefficients.cuh 
           )

cuda_add_executable(CComp
          src/algorithms/ConnectedComponents.cpp
		  include/algorithms/ConnectedComponents.h 
           )

cuda_add_executable(BC 
          src/algorithms/BetweennessCentrality.cpp
		  include/algorithms/BetweennessCentrality.h 
           )

cuda_add_executable(PR
          src/algorithms/PR.cpp
		  include/algorithms/PageRank.h 
		  include/algorithms/PageRank.cuh 
           )

cuda_add_executable(continuousTC
          src/tests/ContinuousTC.cpp
           )

cuda_add_executable(dynamicVerticesMain 
          src/tests/dynamicVerticesMain.cpp
           )

cuda_add_executable(concurrentTC
          src/tests/concurrentTC.cpp
           )

cuda_add_executable(queryTC
          src/tests/queryTC.cpp
           )

cuda_add_executable(spmv 
          src/tests/SpMV.cpp
           )


TARGET_LINK_LIBRARIES(mainfaimGraph faimgraph ${CUDA_cudart_static_LIBRARY})
TARGET_LINK_LIBRARIES(STC faimgraph  ${CUDA_cudart_static_LIBRARY})
TARGET_LINK_LIBRARIES(CCoeff faimgraph  ${CUDA_cudart_static_LIBRARY})
TARGET_LINK_LIBRARIES(CComp faimgraph  ${CUDA_cudart_static_LIBRARY})
TARGET_LINK_LIBRARIES(BC faimgraph  ${CUDA_cudart_static_LIBRARY})
TARGET_LINK_LIBRARIES(PR faimgraph  ${CUDA_cudart_static_LIBRARY})
TARGET_LINK_LIBRARIES(continuousTC faimgraph  ${CUDA_cudart_static_LIBRARY})
TARGET_LINK_LIBRARIES(dynamicVerticesMain faimgraph  ${CUDA_cudart_static_LIBRARY})
TARGET_LINK_LIBRARIES(concurrentTC faimgraph  ${CUDA_cudart_static_LIBRARY})
TARGET_LINK_LIBRARIES(queryTC faimgraph  ${CUDA_cudart_static_LIBRARY})
TARGET_LINK_LIBRARIES(spmv faimgraph  ${CUDA_cudart_static_LIBRARY})
TARGET_LINK_LIBRARIES(reInitTC faimgraph  ${CUDA_cudart_static_LIBRARY})
