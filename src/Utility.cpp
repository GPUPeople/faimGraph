//------------------------------------------------------------------------------
// Utility.cpp
//
// faimGraph
//
//------------------------------------------------------------------------------
//
#include <iostream>

#include "Utility.h"
#include "Definitions.h"

void queryAndPrintDeviceProperties()
{
    cudaDeviceProp prop;
    int count;
    HANDLE_ERROR(cudaGetDeviceCount(&count));
    for(int i = 0; i < count; ++i)
    {
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
        std::cout << "  -- General Information for device " << i << " ---  " << std::endl;
        std::cout << "Name: " << prop.name << std::endl;
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  -- Memory Information for device ---  " << std::endl;
        std::cout << "Total global memory: " << prop.totalGlobalMem << std::endl;
        std::cout << "Total constant memory: " << prop.totalConstMem << std::endl;
        std::cout << "  -- MP Information for device ---  " << std::endl;
        std::cout << "Multiprocessor count " << prop.multiProcessorCount << std::endl;
        std::cout << "Shared mem per MP " << prop.sharedMemPerBlock << std::endl;
        std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Max thread dimensions: (" << prop.maxThreadsDim[0];
        std::cout << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "Max thread dimensions: (" << prop.maxGridSize[0];
        std::cout << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;

        size_t free_mem, total_mem;
        HANDLE_ERROR(cudaMemGetInfo(&free_mem, &total_mem));
        std::cout << "Total Memory: " << total_mem << std::endl;
        std::cout << "Free Memory: " << free_mem << std::endl;
    }
}