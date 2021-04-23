cmake_minimum_required(VERSION 3.16)
project(faimgraph LANGUAGES CXX CUDA)


##########################################################################
# Include Directories
include_directories(include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(include)
include_directories(include/algorithms)

option(CUDA_BUILD_CC20 "Build with compute capability 2.0 support" FALSE)
option(CUDA_BUILD_CC21 "Build with compute capability 2.1 support" FALSE)
option(CUDA_BUILD_CC30 "Build with compute capability 3.0 support" FALSE)
option(CUDA_BUILD_CC35 "Build with compute capability 3.5 support" FALSE)
option(CUDA_BUILD_CC50 "Build with compute capability 5.0 support" FALSE)
option(CUDA_BUILD_CC52 "Build with compute capability 5.2 support" FALSE)
option(CUDA_BUILD_CC61 "Build with compute capability 6.1 support" FALSE)
option(CUDA_BUILD_CC70_SYNC "Build with compute capability 7.0 support - SYNC" FALSE)
option(CUDA_BUILD_CC70_ASYNC "Build with compute capability 7.0 support - ASYNC" TRUE)
option(CUDA_BUILD_CC75 "Build with compute capability 7.5 support" FALSE)
option(CUDA_BUILD_INFO "Build with kernel statistics and line numbers" TRUE)
option(CUDA_BUILD_DEBUG "Build with kernel debug" FALSE)

##########################################################################
# CUDA Flags
if (CUDA_BUILD_CC50)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_50,code=sm_50")
endif ()
if (CUDA_BUILD_CC52)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_52,code=sm_52")
endif ()
if (CUDA_BUILD_CC61)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_61,code=sm_61")
endif ()
if (CUDA_BUILD_CC70_SYNC)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_61,code=sm_70")
endif ()
if (CUDA_BUILD_CC70_ASYNC)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_70,code=sm_70")
endif ()
if (CUDA_BUILD_CC75)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_75,code=sm_75")
endif ()


string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler -Wall -D_FORCE_INLINES -DVERBOSE --expt-extended-lambda -use_fast_math --expt-relaxed-constexpr")

if (CUDA_BUILD_INFO)
	string(APPEND CMAKE_CUDA_FLAGS " -keep --ptxas-options=-v -lineinfo")
endif ()

if (CUDA_BUILD_DEBUG)
	string(APPEND CMAKE_CUDA_FLAGS " -G")
endif ()

##########################################################################
# CXX Flags
if(WIN32)
set(CUDA_PROPAGATE_HOST_FLAGS ON)
if(MSVC)
  string(APPEND CMAKE_CXX_FLAGS "/wd4464 /wd4514 /wd4820 /wd4668 /wd4574 /wd4571 /wd4324 /wd4710 /wd4711 /wd4365 /wd4515 /wd4201 /wd4267 /wd5027 /wd4626")
endif()
else()
set(CUDA_PROPAGATE_HOST_FLAGS ON)
string(APPEND CMAKE_CXX_FLAGS "-std=c++17")
endif()

##########################################################################
# FaimGraph Headers
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

##########################################################################
# FaimGraph Library
add_library(faimgraph
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

##########################################################################
# Executable
add_executable(main 
          src/main.cpp
           )

add_executable(reInitTC 
          src/tests/reInitTC.cpp
           )

add_executable(STC 
          src/algorithms/STC.cpp
		  include/algorithms/STC.h
		  include/algorithms/STC.cuh
           )

add_executable(BFS 
          src/algorithms/BFS.cpp
		  		include/algorithms/BFS.h
		  		src/algorithms/BFS.cu
					 )
					 
# Enable Device launches for DP
set_property(TARGET BFS PROPERTY CUDA_SEPARABLE_COMPILATION ON)

add_executable(CCoeff
          src/algorithms/ClusteringCoefficients.cpp
		  include/algorithms/ClusteringCoefficients.h 
		  include/algorithms/ClusteringCoefficients.cuh 
           )

add_executable(CComp
          src/algorithms/ConnectedComponents.cpp
		  include/algorithms/ConnectedComponents.h 
           )

add_executable(BC 
          src/algorithms/BetweennessCentrality.cpp
		  include/algorithms/BetweennessCentrality.h 
           )

add_executable(PR
          src/algorithms/PR.cpp
		  include/algorithms/PageRank.h 
		  include/algorithms/PageRank.cuh 
           )

add_executable(continuousTC
          src/tests/ContinuousTC.cpp
           )

add_executable(dynamicVertices 
          src/tests/dynamicVerticesMain.cpp
           )

add_executable(concurrentTC
          src/tests/concurrentTC.cpp
           )

add_executable(queryTC
          src/tests/queryTC.cpp
           )

add_executable(spmv 
          src/tests/SpMV.cpp
           )

##########################################################################
# Link libraries
target_link_libraries(main faimgraph)
target_link_libraries(STC faimgraph)
if(WIN32)
	target_link_libraries(BFS faimgraph cudadevrt.lib)
else()
	target_link_libraries(BFS faimgraph)
endif()
target_link_libraries(CCoeff faimgraph)
target_link_libraries(CComp faimgraph)
target_link_libraries(BC faimgraph)
target_link_libraries(PR faimgraph)
target_link_libraries(continuousTC faimgraph)
target_link_libraries(dynamicVertices faimgraph)
target_link_libraries(concurrentTC faimgraph)
target_link_libraries(queryTC faimgraph)
target_link_libraries(spmv faimgraph)
target_link_libraries(reInitTC faimgraph)
