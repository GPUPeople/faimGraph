//------------------------------------------------------------------------------
// Queue.h
//
// faimGraph
//
//------------------------------------------------------------------------------
//

#pragma once

#include <cuda_runtime_api.h>
#include <cuda.h>

class IndexQueue
{
public:
  __forceinline__ __device__ void init();
  
  __forceinline__ __device__ bool enqueue(index_t i);
  __forceinline__ __device__ void enqueueAlternating(index_t i);
  
  __forceinline__ __device__ int dequeue(index_t& element);
  __forceinline__ __device__ int dequeueAlternating(index_t& element);

  void resetQueue();

  index_t* queue_;
  int count_{ 0 };
  unsigned int front_{ 0 };
  unsigned int back_{ 0 };
  int size_{ 0 };
};

__forceinline__ __host__ void IndexQueue::resetQueue()
{
  count_ = 0;
  front_ = 0;
  back_ = 0;
}

__forceinline__ __device__ void IndexQueue::init()
{
  #ifdef __CUDA_ARCH__
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size_; i += blockDim.x * gridDim.x)
  {
    queue_[i] = DELETIONMARKER;
  }
  #endif
}

__forceinline__ __device__ bool IndexQueue::enqueue(index_t i)
{
  #ifdef __CUDA_ARCH__
  int fill = atomicAdd(&count_, 1);
  if (fill < static_cast<int>(size_))
  {
    //we have to wait in case there is still something in the spot
    // note: as the filllevel could be increased by this thread, we are certain that the spot will become available
    unsigned int pos = atomicAdd(&back_, 1) % size_;
    while (atomicCAS(queue_ + pos, DELETIONMARKER, i) != DELETIONMARKER)
      __threadfence();
    return true;
  }
  else
  {
    //__trap(); //no space to enqueue -> fail
    return false;
  }
  #endif
}

__forceinline__ __device__ void IndexQueue::enqueueAlternating(index_t i)
{
#ifdef __CUDA_ARCH__
  int fill = atomicAdd(&count_, 1);
  if (fill < static_cast<int>(size_))
  {
    //we have to wait in case there is still something in the spot
    // note: as the filllevel could be increased by this thread, we are certain that the spot will become available
    unsigned int pos = atomicAdd(&back_, 1) % size_;
    index_t test = atomicExch(queue_ + pos, i);
  }
  else
  {
    __trap(); //no space to enqueue -> fail
  }
#endif
}

__forceinline__ __device__ int IndexQueue::dequeue(index_t& element)
{
  //for more efficiency potentially:
  //if(ldg(&count) <= 0)
  //   return 0;  

  #ifdef __CUDA_ARCH__      
  int readable = atomicSub(&count_, 1);
  if (readable <= 0)
  {
    //dequeue not working is a common case
    atomicAdd(&count_, 1);
    return FALSE;
  }
  unsigned int pos = atomicAdd(&front_, 1) % size_;
  while ((element = atomicExch(queue_ + pos, DELETIONMARKER)) == DELETIONMARKER)
    __threadfence();
  return TRUE;
  #else
  return FALSE;
  #endif
}

__forceinline__ __device__ int IndexQueue::dequeueAlternating(index_t& element)
{
#ifdef __CUDA_ARCH__      
  int readable = atomicSub(&count_, 1);
  if (readable <= 0)
  {
    //dequeue not working is a common case
    atomicAdd(&count_, 1);
    return FALSE;
  }
  unsigned int pos = atomicAdd(&front_, 1) % size_;
  element = atomicExch(queue_ + pos, DELETIONMARKER);
  return TRUE;
#else
  return FALSE;
#endif
}
