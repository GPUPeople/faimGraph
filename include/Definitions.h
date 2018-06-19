//------------------------------------------------------------------------------
// Definitions.h
//
// faimGraph
//
//------------------------------------------------------------------------------
//

#pragma once

#include <typeinfo>
#include <memory>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

//------------------------------------------------------------------------------
// Datatype definitions
//------------------------------------------------------------------------------
using vertex_t = uint32_t;
using index_t = vertex_t;
using memory_t = int8_t;
using matrix_t = uint32_t;
using OffsetList_t = std::vector<vertex_t>;
using AdjacencyList_t = std::vector<vertex_t>;
using MatrixList_t = std::vector<matrix_t>;

//------------------------------------------------------------------------------
// Number definitions
//------------------------------------------------------------------------------
#define RET_ERROR -1
#define MEGABYTE (1024 * 1024)
#define GIGABYTE (MEGABYTE * 1024)
#define CACHELINESIZE 128
#define WARPSIZE 32
#define MEMMANOFFSET (2 * CACHELINESIZE)

#define SINGLE_THREAD (threadIdx.x == 0)
#define SINGLE_THREAD_MULTI (threadID == 0)
#define SINGLE_THREAD_INDEX (threadID == edges_per_block)
#define SINGLE_THREAD_INDEX_WARPID (threadID == edges_per_block[warpID])
#define LINEAR_THREAD_ID (threadIdx.x + blockIdx.x*blockDim.x)

//------------------------------------------------------------------------------
// Memory layout definitions
//------------------------------------------------------------------------------
#define LOCKING_INDEX_MULTIPLIER 0
#define MEM_INDEX_MULTIPLIER (LOCKING_INDEX_MULTIPLIER + 1)
#define NEIGHBOURS_INDEX_MULTIPLIER (MEM_INDEX_MULTIPLIER + 1)
#define CAPACITY_INDEX_MULTIPLIER (NEIGHBOURS_INDEX_MULTIPLIER + 1)
#define WEIGHT_INDEX_MULTIPLIER (CAPACITY_INDEX_MULTIPLIER + 1)
#define TYPE_INDEX_MULTIPLIER (WEIGHT_INDEX_MULTIPLIER + 1)
//#define EDGEDATA_INDEX_MULTIPLIER (TYPE_INDEX_MULTIPLIER + 1)

//------------------------------------------------------------------------------
// SOA (Structure of Arrays) offset definitions
//------------------------------------------------------------------------------
//
#define SOA_OFFSET_WEIGHT 1
#define SOA_OFFSET_TYPE (SOA_OFFSET_WEIGHT + 1)
#define SOA_OFFSET_TIMESTAMP1 (SOA_OFFSET_TYPE + 1)
#define SOA_OFFSET_TIMESTAMP2 (SOA_OFFSET_TIMESTAMP1 + 1)
#define SOA_OFFSET_MATRIX 1


//------------------------------------------------------------------------------
// Marker definitions
//------------------------------------------------------------------------------
#define DELETIONMARKER UINT32_MAX
#define INVALID_INDEX (DELETIONMARKER - 1)
#define EMPTYMARKER (DELETIONMARKER - 2)
#define INVALID_FLOAT FLT_MAX

//------------------------------------------------------------------------------
// Locking Definitions
//------------------------------------------------------------------------------
#define UNLOCK 0
#define LOCK 1

#define FALSE 0
#define TRUE 1

#define UPDATE_BASED_DUPLICATE_CHECKING
constexpr int MAXIMAL_BATCH_SIZE = 1000000;

//------------------------------------------------------------------------------
// Kernel Params
//------------------------------------------------------------------------------
#define KERNEL_LAUNCH_BLOCK_SIZE 32
#define KERNEL_LAUNCH_BLOCK_SIZE_WARP_SIZED 32
#define KERNEL_LAUNCH_BLOCK_SIZE_STANDARD 256
#define BLOCK_SIZE_WARP_SIZED_KERNEL_LAUNCH 128

//------------------------------------------------------------------------------
// Switches
//------------------------------------------------------------------------------
//#define WARP_SIZE_KERNEL_LAUNCH

//------------------------------------------------------------------------------
// Launch Definitions
//------------------------------------------------------------------------------
#ifdef WARP_SIZE_KERNEL_LAUNCH
const int32_t EDGEBLOCKSIZE = 128;
#else
const int32_t EDGEBLOCKSIZE = 64;
#endif

#ifdef WARP_SIZE_KERNEL_LAUNCH
#define KERNEL_LAUNCH_BLOCK_SIZE_INSERT 32
#else
#define KERNEL_LAUNCH_BLOCK_SIZE_INSERT 256
#endif

#ifdef WARP_SIZE_KERNEL_LAUNCH
#define KERNEL_LAUNCH_BLOCK_SIZE_DELETE 32
#else
#define KERNEL_LAUNCH_BLOCK_SIZE_DELETE 256
#endif

//------------------------------------------------------------------------------
// Debug Flag
//------------------------------------------------------------------------------
//#define DEBUG_VERBOSE_OUPUT
//#define ACCESS_METRICS
