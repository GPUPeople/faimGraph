//------------------------------------------------------------------------------
// Utility.cu
//
// faimGraph
//
//------------------------------------------------------------------------------
//

#include "Utility.h"

//------------------------------------------------------------------------------
void start_clock(cudaEvent_t &start, cudaEvent_t &end)
{
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&end));
    HANDLE_ERROR(cudaEventRecord(start,0));
}
//------------------------------------------------------------------------------
float end_clock(cudaEvent_t &start, cudaEvent_t &end)
{
    float time;
    HANDLE_ERROR(cudaEventRecord(end,0));
    HANDLE_ERROR(cudaEventSynchronize(end));
    HANDLE_ERROR(cudaEventElapsedTime(&time,start,end));
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(end));

    // Returns ms
    return time;
}