#include <numeric>
#include "MemoryManager.h"
#include "BFS.h"

#include "cub/cub.cuh"

#ifdef __INTELLISENSE__
unsigned int atomicCAS(unsigned int* address, unsigned int compare, unsigned int val);
unsigned int atomicAdd(unsigned int* address, unsigned int val);
void __syncthreads();
#endif

namespace faimGraphBFS
{
	const vertex_t NOT_VISITIED = std::numeric_limits<vertex_t>::max();

	struct dFrontierQueue
	{
	public:
		unsigned int *size;
		vertex_t *nodes;

		dFrontierQueue(unsigned int capacity)
		{
			cudaMalloc((void**)&nodes, sizeof(vertex_t) * capacity);
			cudaMalloc((void**)&size, sizeof(unsigned int));
			cudaMemset(size, 0, sizeof(unsigned int));
		}

		void Free()
		{
			if (nodes)
				cudaFree(nodes);
			if (size)
				cudaFree(size);
			nodes = nullptr;
			size = nullptr;
		}

		__device__
			void Reset()
		{
			*size = 0;
		}

		__device__
			unsigned int Allocate(unsigned int n_nodes)
		{
			return atomicAdd(size, n_nodes);
		}
	};

	template <typename VertexDataType, typename EdgeDataType>
	__global__ void d_BFSIteration(MemoryManager *memory_manager, memory_t *memory, int page_size, int *found_new_nodes, vertex_t *frontier, int iteration)
	{
		unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

		if (tid < memory_manager->number_vertices && frontier[tid] == iteration)
		{
			VertexDataType *vertices = (VertexDataType*)memory;

			AdjacencyIterator<EdgeDataType> adjacency_iterator(pageAccess<EdgeDataType>(memory, vertices[tid].mem_index, page_size, memory_manager->start_index));
			for (int i = 0; i < vertices[tid].neighbours; i++)
			{
				vertex_t next_node = adjacency_iterator.getDestination();
				if (atomicCAS(frontier + next_node, NOT_VISITIED, iteration + 1) == NOT_VISITIED)
				{
					*found_new_nodes = 1;
				}
				adjacency_iterator.advanceIterator(i, memory_manager->edges_per_page, memory, page_size, memory_manager->start_index);
			}
		}
	}

    template <typename VertexDataType, typename EdgeDataType, size_t THREADS_PER_BLOCK>
	__global__ void d_bfsBasic(MemoryManager* memory_manager, memory_t* memory, int page_size, int *found_new_nodes, vertex_t *frontier, vertex_t start_node)
	{
		frontier[start_node] = 0;

		int iteration = 0;
		do
		{
			*found_new_nodes = 0;
			d_BFSIteration <VertexDataType, EdgeDataType> << <(memory_manager->number_vertices + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >
				(memory_manager, memory, page_size, found_new_nodes, frontier, iteration);
			iteration++;
			cudaDeviceSynchronize();
		} while (*found_new_nodes);
	}

	// Explores the given node with a single thread
	template <typename VertexDataType, typename EdgeDataType>
	__device__ void d_ExploreEdges(MemoryManager *memory_manager, memory_t *memory, int page_size, vertex_t *frontiers,
		vertex_t *thread_frontier, int &n_frontier_nodes, vertex_t node, int iteration)
	{
		VertexDataType *vertices = (VertexDataType*)memory;
		VertexDataType &current_node = vertices[node];
		AdjacencyIterator<EdgeDataType> adjacency_iterator(pageAccess<EdgeDataType>(memory, current_node.mem_index, page_size, memory_manager->start_index));
		auto n_edges = current_node.neighbours;

		for (int i = 0; i < n_edges; i++)
		{
			vertex_t next_node = adjacency_iterator.getDestination();
			if (atomicCAS(frontiers + next_node, NOT_VISITIED, iteration + 1) == NOT_VISITIED)
			{
				thread_frontier[n_frontier_nodes++] = next_node;
			}
			adjacency_iterator.advanceIterator(i, memory_manager->edges_per_page, memory, page_size, memory_manager->start_index);
		}
	}

	// Explores the adjacency of the given node with all calling threads in parallel. n_threads should equal the number of calling threads
	template<typename VertexDataType, typename EdgeDataType>
	__device__ void d_ExploreEdges(MemoryManager *memory_manager, memory_t *memory, int page_size, vertex_t *frontiers,
		vertex_t *thread_frontier, int &n_frontier_nodes, vertex_t node, vertex_t start, vertex_t n_threads, int iteration)
	{
		VertexDataType *vertices = (VertexDataType*)memory;
		VertexDataType &current_node = vertices[node];
		AdjacencyIterator<EdgeDataType> adjacency_iterator(pageAccess<EdgeDataType>(memory, current_node.mem_index, page_size, memory_manager->start_index));
		auto n_edges = current_node.neighbours;
		auto edges_per_page = memory_manager->edges_per_page;

		vertex_t page_index = 0;
		auto edge = start;
		while (edge < n_edges)
		{
			auto target_page_index = edge / edges_per_page;
			auto page_offset = edge % edges_per_page; // Offset on target page

			// Do page traversals until target page
			for (; page_index < target_page_index; page_index++)
				adjacency_iterator.blockTraversalAbsolute(edges_per_page, memory, page_size, memory_manager->start_index);


			auto const next_node = adjacency_iterator.getDestinationAt(page_offset);
			if (atomicCAS(frontiers + next_node, NOT_VISITIED, iteration + 1) == NOT_VISITIED)
			{
				thread_frontier[n_frontier_nodes] = next_node;
				n_frontier_nodes++;
			}

			edge += n_threads;
		} 
	}
	
	// Fills a frontier queue based on individual thread frontiers
	// Has to be called by the entire block
	template<size_t THREADS_PER_BLOCK>
	__device__ void d_FillFrontierQueue(dFrontierQueue &queue, vertex_t *thread_frontier, int n_frontier_nodes)
	{
		typedef cub::BlockScan<unsigned int, THREADS_PER_BLOCK> BlockScan;
		__shared__ typename BlockScan::TempStorage temp_storage;

		// Get per-thread offset in queue
		unsigned int thread_offset;
		BlockScan(temp_storage).ExclusiveSum(n_frontier_nodes, thread_offset);
		__syncthreads();

		// Get per-block offset in queue. Last thread knows total size, so it does the allocation
		__shared__ unsigned int block_offset;
		if (threadIdx.x == THREADS_PER_BLOCK - 1)
		{
			unsigned int total_size = thread_offset + n_frontier_nodes;
			if (total_size == 0)
				block_offset = std::numeric_limits<unsigned int>::max();
			else
				block_offset = queue.Allocate(total_size);
		}
		__syncthreads();

		// If we didn't discover any new nodes we don't have anything to copy to the queue
		if (block_offset == std::numeric_limits<unsigned int>::max())
			return;

		// Lastly, copy all discovered nodes to the queue on a per-thread basis
		thread_offset += block_offset;
		for (int i = 0; i < n_frontier_nodes; i++)
			queue.nodes[thread_offset + i] = thread_frontier[i];
	}


	template<typename VertexDataType, typename EdgeDataType, size_t EDGES_PER_THREAD, size_t THREADS_PER_BLOCK>
	__global__ void d_ExploreEdges_kernel(MemoryManager *memory_manager, memory_t *memory, int page_size, vertex_t *frontiers,
			vertex_t node, dFrontierQueue newFrontierQueue, int iteration)
	{
		unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
		unsigned int n_threads = blockDim.x * gridDim.x;

		vertex_t thread_frontier[EDGES_PER_THREAD];
		int n_frontier_nodes = 0;

		d_ExploreEdges<VertexDataType, EdgeDataType>(memory_manager, memory, page_size, frontiers,
			thread_frontier, n_frontier_nodes, node, tid, n_threads, iteration);
		__syncthreads();
		d_FillFrontierQueue<THREADS_PER_BLOCK>(newFrontierQueue, thread_frontier, n_frontier_nodes);
	}

	template<typename VertexDataType, typename EdgeDataType, size_t EDGES_PER_THREAD, size_t THREADS_PER_BLOCK>
	__global__ void d_bfsDynamicParalellismIteration(MemoryManager *memory_manager, memory_t *memory, int page_size, vertex_t *frontiers,
		dFrontierQueue newFrontierQueue, dFrontierQueue oldFrontierQueue, int iteration)
	{
		int const edges_per_block = THREADS_PER_BLOCK * EDGES_PER_THREAD;

		unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;

		vertex_t thread_frontier[EDGES_PER_THREAD];
		int n_frontier_nodes = 0;

		VertexDataType *vertices = (VertexDataType*)memory;

		if (id < *oldFrontierQueue.size)
		{
			auto const current_node = oldFrontierQueue.nodes[id];
			
			vertex_t n_edges = vertices[current_node].neighbours;
			AdjacencyIterator<EdgeDataType> adjacency_iterator(pageAccess<EdgeDataType>(memory, vertices[current_node].mem_index, page_size, memory_manager->start_index));

			if (n_edges <= EDGES_PER_THREAD)
			{
				d_ExploreEdges<VertexDataType, EdgeDataType>(memory_manager, memory, page_size, frontiers,
					thread_frontier, n_frontier_nodes, current_node, iteration);
			}
			else
			{
				//printf("Using sub-kernel with %d blocks for %d edges\n", (n_edges + edges_per_block - 1) / edges_per_block, n_edges);
				d_ExploreEdges_kernel<VertexDataType, EdgeDataType, EDGES_PER_THREAD, THREADS_PER_BLOCK> << <(n_edges + edges_per_block - 1) / edges_per_block, THREADS_PER_BLOCK >> >
					(memory_manager, memory, page_size, frontiers, current_node, newFrontierQueue, iteration);
			}
		}
		__syncthreads();

		d_FillFrontierQueue<THREADS_PER_BLOCK>(newFrontierQueue, thread_frontier, n_frontier_nodes);
	}

	template<typename VertexDataType, typename EdgeDataType, size_t EDGES_PER_THREAD, size_t THREADS_PER_BLOCK>
	__global__ void d_bfsDynamicParalellism(MemoryManager *memory_manager, memory_t *memory, int page_size, vertex_t *frontiers,
		dFrontierQueue newFrontierQueue, dFrontierQueue oldFrontierQueue, vertex_t start_node)
	{
		frontiers[start_node] = 0;
		newFrontierQueue.Allocate(1);
		newFrontierQueue.nodes[0] = start_node;

		int iteration = 0;
		do
		{
			auto temp = oldFrontierQueue;
			oldFrontierQueue = newFrontierQueue;
			newFrontierQueue = temp;
			newFrontierQueue.Reset();

			d_bfsDynamicParalellismIteration<VertexDataType, EdgeDataType, EDGES_PER_THREAD, THREADS_PER_BLOCK>
				<< <(*oldFrontierQueue.size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >
				(memory_manager, memory, page_size, frontiers, newFrontierQueue, oldFrontierQueue, iteration);
			iteration++;
			cudaDeviceSynchronize();
		} while (*newFrontierQueue.size > 0);
	}

	template<typename VertexDataType, size_t THREADS_PER_BLOCK, size_t NODES_PER_THREAD, size_t EDGES_PER_THREAD>
	__global__ void d_ClassifyNodes_kernel(MemoryManager *memory_manager, memory_t *memory, dFrontierQueue rawFrontier,
		dFrontierQueue smallNodesFrontier, dFrontierQueue mediumNodesFrontier, dFrontierQueue largeNodesFrontier)
	{
		unsigned int const tid = threadIdx.x + blockIdx.x * blockDim.x;
		unsigned int stride = blockDim.x * gridDim.x;

		VertexDataType *vertices = (VertexDataType*)memory;

		vertex_t smallThreadFrontier[NODES_PER_THREAD];
		vertex_t mediumThreadFrontier[NODES_PER_THREAD];
		vertex_t largeThreadFrontier[NODES_PER_THREAD];
		int n_small = 0;
		int n_medium = 0;
		int n_large = 0;

		for (unsigned int i = tid, end = *rawFrontier.size; i < end; i += stride)
		{
			auto const node = rawFrontier.nodes[i];
			auto const n_edges = vertices[node].neighbours;

			if (n_edges <= EDGES_PER_THREAD)
			{
				//printf("Classifying node %u with %u edges as small\n", node, n_edges);
				smallThreadFrontier[n_small++] = node;
			}
			else if (n_edges <= EDGES_PER_THREAD * WARPSIZE)
			{
				//printf("Classifying node %u with %u edges as medium\n", node, n_edges);
				mediumThreadFrontier[n_medium++] = node;
			}
			else
			{
				//printf("Classifying node %u with %u edges as large\n", node, n_edges);
				largeThreadFrontier[n_large++] = node;
			}
		}

		__syncthreads();
		d_FillFrontierQueue<THREADS_PER_BLOCK>(smallNodesFrontier, smallThreadFrontier, n_small);
		d_FillFrontierQueue<THREADS_PER_BLOCK>(mediumNodesFrontier, mediumThreadFrontier, n_medium);
		d_FillFrontierQueue<THREADS_PER_BLOCK>(largeNodesFrontier, largeThreadFrontier, n_large);
	}

	template<typename VertexDataType, typename EdgeDataType, size_t EDGES_PER_THREAD, size_t THREADS_PER_BLOCK>
	__global__ void d_ExploreFrontier_kernel(MemoryManager *memory_manager, memory_t *memory, int page_size, vertex_t *frontiers,
		dFrontierQueue frontier, dFrontierQueue newFrontier, vertex_t threads_per_node, unsigned int iteration)
	{
		unsigned int const tid = threadIdx.x + blockIdx.x * blockDim.x;

		vertex_t offset = tid % threads_per_node;

		vertex_t thread_frontier[EDGES_PER_THREAD];
		int n_frontier_nodes = 0;

		unsigned int node_index = tid / threads_per_node;

		if (node_index < *frontier.size)
		{
			auto const node = frontier.nodes[node_index];

			//printf("Thread %u exploring node %u\n", tid, node);

			d_ExploreEdges<VertexDataType, EdgeDataType>
				(memory_manager, memory, page_size, frontiers, thread_frontier, n_frontier_nodes, node, offset, threads_per_node, iteration);
		}

		__syncthreads();

		d_FillFrontierQueue<THREADS_PER_BLOCK>(newFrontier, thread_frontier, n_frontier_nodes);
	}

	// Gets the size of the largest node in the queue and stores it in current_max_node_size
	template <typename VertexDataType, size_t THREADS_PER_BLOCK, size_t ITEMS_PER_THREAD>
	__global__
		void d_GetMaxNodeSize_kernel(dFrontierQueue queue, memory_t *memory, vertex_t *current_max_node_size)
	{
		using BlockReduce = cub::BlockReduce<unsigned int, THREADS_PER_BLOCK>;
		__shared__ typename BlockReduce::TempStorage temp_storage;

		int tid = threadIdx.x;
		int start = tid * ITEMS_PER_THREAD;

		VertexDataType *vertices = (VertexDataType*)memory;

		vertex_t node_sizes[ITEMS_PER_THREAD];
		for (int i = 0; i < ITEMS_PER_THREAD; i++)
		{
			vertex_t node_index = start + i;
			if (node_index > *queue.size)
			{
				// Don't forget to set unused entries to 0
				for (; i < ITEMS_PER_THREAD; i++)
					node_sizes[i] = 0;
				break;
			}

			vertex_t node = queue.nodes[node_index];
			node_sizes[i] = vertices[node].neighbours;
		}

		vertex_t max_size = BlockReduce(temp_storage).Reduce(node_sizes, cub::Max());
		if (tid == 0)
		{
			*current_max_node_size = max_size;
		}
	}

	template<typename VertexDataType, typename EdgeDataType, size_t EDGES_PER_THREAD, size_t THREADS_PER_BLOCK>
	__global__ void d_BFSPreprocessing(MemoryManager *memory_manager, memory_t *memory, int page_size, vertex_t *frontiers,
		dFrontierQueue rawFrontier,	dFrontierQueue smallNodesFrontier, dFrontierQueue mediumNodesFrontier, 
		dFrontierQueue largeNodesFrontier, dFrontierQueue hugeNodesFrontier, vertex_t *current_max_node_size, vertex_t start_node)
	{
		size_t const EDGES_PER_BLOCK = THREADS_PER_BLOCK * EDGES_PER_THREAD;

		frontiers[start_node] = 0;
		rawFrontier.Allocate(1);
		rawFrontier.nodes[0] = start_node;

		unsigned int iteration = 0;
		while (*rawFrontier.size > 0)
		{
			smallNodesFrontier.Reset();
			mediumNodesFrontier.Reset();
			largeNodesFrontier.Reset();
			//printf("Iteration %u, %u nodes\n", iteration, *rawFrontier.size);
			d_ClassifyNodes_kernel<VertexDataType, THREADS_PER_BLOCK, EDGES_PER_THREAD, EDGES_PER_THREAD>
				<< <(*rawFrontier.size + EDGES_PER_BLOCK - 1) / EDGES_PER_BLOCK, THREADS_PER_BLOCK >> >
				(memory_manager, memory, rawFrontier, smallNodesFrontier, mediumNodesFrontier, largeNodesFrontier);
			cudaDeviceSynchronize();

			//printf("Queue sizes: %u, %u, %u\n", *smallNodesFrontier.size, *mediumNodesFrontier.size, *largeNodesFrontier.size);

			rawFrontier.Reset();

			if (*hugeNodesFrontier.size > 0)
			{
				if (*hugeNodesFrontier.size <= THREADS_PER_BLOCK)
					d_GetMaxNodeSize_kernel<VertexDataType, THREADS_PER_BLOCK, 1> << <1, THREADS_PER_BLOCK >> > (hugeNodesFrontier, memory, current_max_node_size);
				else if (*hugeNodesFrontier.size <= THREADS_PER_BLOCK * 4)
					d_GetMaxNodeSize_kernel<VertexDataType, THREADS_PER_BLOCK, 4> << <1, THREADS_PER_BLOCK >> > (hugeNodesFrontier, memory, current_max_node_size);
				else if (*hugeNodesFrontier.size <= THREADS_PER_BLOCK * 16)
					d_GetMaxNodeSize_kernel<VertexDataType, THREADS_PER_BLOCK, 16> << <1, THREADS_PER_BLOCK >> > (hugeNodesFrontier, memory, current_max_node_size);
				else if (*hugeNodesFrontier.size <= THREADS_PER_BLOCK * 64)
					d_GetMaxNodeSize_kernel<VertexDataType, THREADS_PER_BLOCK, 64> << <1, THREADS_PER_BLOCK >> > (hugeNodesFrontier, memory, current_max_node_size);
				else
					d_GetMaxNodeSize_kernel<VertexDataType, THREADS_PER_BLOCK, 128> << <1, THREADS_PER_BLOCK >> > (hugeNodesFrontier, memory, current_max_node_size);
				cudaDeviceSynchronize();

				vertex_t n_blocks = (*current_max_node_size + EDGES_PER_BLOCK - 1) / EDGES_PER_BLOCK;
				d_ExploreFrontier_kernel<VertexDataType, EdgeDataType, EDGES_PER_THREAD, THREADS_PER_BLOCK>
					<< <n_blocks * *hugeNodesFrontier.size, THREADS_PER_BLOCK >> >
					(memory_manager, memory, page_size, frontiers, hugeNodesFrontier, rawFrontier, n_blocks * THREADS_PER_BLOCK, iteration);
			}

			if (*smallNodesFrontier.size > 0)
			{
				d_ExploreFrontier_kernel<VertexDataType, EdgeDataType, EDGES_PER_THREAD, THREADS_PER_BLOCK>
					<< <(*smallNodesFrontier.size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >
					(memory_manager, memory, page_size, frontiers, smallNodesFrontier, rawFrontier, 1, iteration);
			}
			if (*mediumNodesFrontier.size > 0)
			{
				d_ExploreFrontier_kernel<VertexDataType, EdgeDataType, EDGES_PER_THREAD, THREADS_PER_BLOCK>
					<< <((*mediumNodesFrontier.size * WARPSIZE) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >
					(memory_manager, memory, page_size, frontiers, mediumNodesFrontier, rawFrontier, WARPSIZE, iteration);
			}
			if (*largeNodesFrontier.size > 0)
			{
				d_ExploreFrontier_kernel<VertexDataType, EdgeDataType, EDGES_PER_THREAD, THREADS_PER_BLOCK>
					<< <*largeNodesFrontier.size, THREADS_PER_BLOCK >> >
					(memory_manager, memory, page_size, frontiers, largeNodesFrontier, rawFrontier, THREADS_PER_BLOCK, iteration);
			}

			cudaDeviceSynchronize();
			iteration++;
		}
	}

	// Explores the adjacency of the given node with all calling threads in parallel. n_threads should equal the number of calling threads
	template<typename VertexDataType, typename EdgeDataType, size_t EDGES_PER_THREAD, size_t THREADS_PER_BLOCK>
	__device__ void d_ExploreEdgesClassification(MemoryManager *memory_manager, memory_t *memory, int page_size, vertex_t *frontiers,
		vertex_t *smallThreadFrontier, int &n_small, bool &foundSmallNode,
		vertex_t *mediumThreadFrontier, int &n_medium, bool &foundMediumNode,
		vertex_t *largeThreadFrontier, int &n_large, bool &foundLargeNode,
		vertex_t *hugeThreadFrontier, int &n_huge, bool &foundHugeNode,
		vertex_t node, vertex_t start, vertex_t n_threads, int iteration)
	{
		VertexDataType *vertices = (VertexDataType*)memory;
		VertexDataType &current_node = vertices[node];
		AdjacencyIterator<EdgeDataType> adjacency_iterator(pageAccess<EdgeDataType>(memory, current_node.mem_index, page_size, memory_manager->start_index));
		auto n_edges = current_node.neighbours;
		auto edges_per_page = memory_manager->edges_per_page;

		vertex_t page_index = 0;
		auto edge = start;
		while (edge < n_edges)
		{
			auto target_page_index = edge / edges_per_page;
			auto page_offset = edge % edges_per_page; // Offset on target page

													  // Do page traversals until target page
			for (; page_index < target_page_index; page_index++)
				adjacency_iterator.blockTraversalAbsolute(edges_per_page, memory, page_size, memory_manager->start_index);


			auto const next_node = adjacency_iterator.getDestinationAt(page_offset);
			if (atomicCAS(frontiers + next_node, NOT_VISITIED, iteration + 1) == NOT_VISITIED)
			{
				vertex_t const n_edges = vertices[next_node].neighbours;

				if (n_edges <= EDGES_PER_THREAD)
				{
					//printf("Classifying node %u with %u edges as small\n", next_node, n_edges);
					smallThreadFrontier[n_small++] = next_node;
					foundSmallNode = true;
				}
				else if (n_edges <= EDGES_PER_THREAD * 32)
				{
					//printf("Classifying node %u with %u edges as medium\n", next_node, n_edges);
					mediumThreadFrontier[n_medium++] = next_node;
					foundMediumNode = true;
				}
				else if (n_edges <= EDGES_PER_THREAD * THREADS_PER_BLOCK)
				{
					//printf("Classifying node %u with %u edges as large\n", next_node, n_edges);
					largeThreadFrontier[n_large++] = next_node;
					foundLargeNode = true;
				}
				else
				{
					hugeThreadFrontier[n_huge++] = next_node;
					foundHugeNode = true;
				}
			}

			edge += n_threads;
		}
	}

	// // Checks a single edge for its discovered status and classifies it if necessary
	// template<size_t EDGES_PER_THREAD, size_t THREADS_PER_BLOCK>
	// __device__
	// 	void d_CheckAndClassifyEdge(unsigned int edge, unsigned int *row_offsets, unsigned int *col_ids, unsigned int *frontiers, unsigned int iteration,
	// 		unsigned int smallThreadFrontier[], int &n_small, bool &foundSmallNode,
	// 		unsigned int mediumThreadFrontier[], int &n_medium, bool &foundMediumNode,
	// 		unsigned int largeThreadFrontier[], int &n_large, bool &foundLargeNode,
	// 		unsigned int hugeThreadFrontier[], int &n_huge, bool &foundHugeNode)
	// {
	// 	unsigned int next_node = col_ids[edge];

	// 	if (atomicCAS(frontiers + next_node, NO_DEPTH, iteration + 1) == NO_DEPTH)
	// 	{
	// 		unsigned int const n_edges = row_offsets[next_node + 1] - row_offsets[next_node];

	// 		if (n_edges <= EDGES_PER_THREAD)
	// 		{
	// 			//printf("Classifying node %u with %u edges as small\n", next_node, n_edges);
	// 			smallThreadFrontier[n_small++] = next_node;
	// 			foundSmallNode = true;
	// 		}
	// 		else if (n_edges <= EDGES_PER_THREAD * 32)
	// 		{
	// 			//printf("Classifying node %u with %u edges as medium\n", next_node, n_edges);
	// 			mediumThreadFrontier[n_medium++] = next_node;
	// 			foundMediumNode = true;
	// 		}
	// 		else if (n_edges <= EDGES_PER_BLOCK)
	// 		{
	// 			//printf("Classifying node %u with %u edges as large\n", next_node, n_edges);
	// 			largeThreadFrontier[n_large++] = next_node;
	// 			foundLargeNode = true;
	// 		}
	// 		else
	// 		{
	// 			hugeThreadFrontier[n_huge++] = next_node;
	// 			foundHugeNode = true;
	// 		}
	// 	}
	// }

	// threads_per_node should divide the number of threads evenly
	template<typename VertexDataType, typename EdgeDataType, size_t EDGES_PER_THREAD, size_t THREADS_PER_BLOCK>
	__global__
		void d_ExploreFrontierClassification_kernel(dFrontierQueue frontier, MemoryManager *memory_manager, memory_t *memory, int page_size, vertex_t *frontiers,
			dFrontierQueue newSmallNodesFrontier, dFrontierQueue newMediumNodesFrontier,
			dFrontierQueue newLargeNodesFrontier, dFrontierQueue newHugeNodesFrontier,
			unsigned int threads_per_node, unsigned int iteration)
	{
		unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

		unsigned int offset = tid % threads_per_node;

		vertex_t smallThreadFrontier[EDGES_PER_THREAD];
		vertex_t mediumThreadFrontier[EDGES_PER_THREAD];
		vertex_t largeThreadFrontier[EDGES_PER_THREAD];
		vertex_t hugeThreadFrontier[EDGES_PER_THREAD];
		int n_small = 0;
		int n_medium = 0;
		int n_large = 0;
		int n_huge = 0;
		__shared__ bool foundSmallNodes;
		__shared__ bool foundMediumNodes;
		__shared__ bool foundLargeNodes;
		__shared__ bool foundHugeNodes;

		if (threadIdx.x == 0)
		{
			foundSmallNodes = false;
			foundMediumNodes = false;
			foundLargeNodes = false;
			foundHugeNodes = false;
		}

		__syncthreads();

		unsigned int node_index = tid / threads_per_node;

		if (node_index < *frontier.size)
		{
			vertex_t node = frontier.nodes[node_index];
			d_ExploreEdgesClassification<VertexDataType, EdgeDataType, EDGES_PER_THREAD, THREADS_PER_BLOCK>(memory_manager, memory, page_size, frontiers,
				smallThreadFrontier, n_small, foundSmallNodes,
				mediumThreadFrontier, n_medium, foundMediumNodes,
				largeThreadFrontier, n_large, foundLargeNodes,
				hugeThreadFrontier, n_huge, foundHugeNodes,
				node, offset, threads_per_node, iteration);
		}

		__syncthreads();
		if (foundSmallNodes)
			d_FillFrontierQueue<THREADS_PER_BLOCK>(newSmallNodesFrontier, smallThreadFrontier, n_small);
		if (foundMediumNodes)
			d_FillFrontierQueue<THREADS_PER_BLOCK>(newMediumNodesFrontier, mediumThreadFrontier, n_medium);
		if (foundLargeNodes)
			d_FillFrontierQueue<THREADS_PER_BLOCK>(newLargeNodesFrontier, largeThreadFrontier, n_large);
		if (foundHugeNodes)
			d_FillFrontierQueue<THREADS_PER_BLOCK>(newHugeNodesFrontier, hugeThreadFrontier, n_huge);
	}

	// Explores all nodes in the frontier by having each thread explore a bit of each node
	template<typename VertexDataType, typename EdgeDataType, size_t EDGES_PER_THREAD, size_t THREADS_PER_BLOCK>
	__global__
		void d_ExploreHugeNodesFrontierClassification_kernel(dFrontierQueue frontier, MemoryManager *memory_manager, memory_t *memory, int page_size, vertex_t *frontiers,
			dFrontierQueue newSmallNodesFrontier, dFrontierQueue newMediumNodesFrontier,
			dFrontierQueue newLargeNodesFrontier, dFrontierQueue newHugeNodesFrontier,
			unsigned int threads_per_node, unsigned int iteration)
	{
		unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

		unsigned int offset = tid % threads_per_node;

		vertex_t smallThreadFrontier[EDGES_PER_THREAD];
		vertex_t mediumThreadFrontier[EDGES_PER_THREAD];
		vertex_t largeThreadFrontier[EDGES_PER_THREAD];
		vertex_t hugeThreadFrontier[EDGES_PER_THREAD];
		int n_small;
		int n_medium;
		int n_large;
		int n_huge;
		__shared__ bool foundSmallNodes;
		__shared__ bool foundMediumNodes;
		__shared__ bool foundLargeNodes;
		__shared__ bool foundHugeNodes;

		for (unsigned int node_index = 0; node_index < *frontier.size; node_index++)
		{
			n_small = 0;
			n_medium = 0;
			n_large = 0;
			n_huge = 0;
			if (threadIdx.x == 0)
			{
				foundSmallNodes = false;
				foundMediumNodes = false;
				foundLargeNodes = false;
				foundHugeNodes = false;
			}
			__syncthreads();

			vertex_t node = frontier.nodes[node_index];

			d_ExploreEdgesClassification<VertexDataType, EdgeDataType, EDGES_PER_THREAD, THREADS_PER_BLOCK>(memory_manager, memory, page_size, frontiers,
				smallThreadFrontier, n_small, foundSmallNodes,
				mediumThreadFrontier, n_medium, foundMediumNodes,
				largeThreadFrontier, n_large, foundLargeNodes,
				hugeThreadFrontier, n_huge, foundHugeNodes,
				node, offset, threads_per_node, iteration);

			__syncthreads();
			if (foundSmallNodes)
				d_FillFrontierQueue<THREADS_PER_BLOCK>(newSmallNodesFrontier, smallThreadFrontier, n_small);
			if (foundMediumNodes)
				d_FillFrontierQueue<THREADS_PER_BLOCK>(newMediumNodesFrontier, mediumThreadFrontier, n_medium);
			if (foundLargeNodes)
				d_FillFrontierQueue<THREADS_PER_BLOCK>(newLargeNodesFrontier, largeThreadFrontier, n_large);
			if (foundHugeNodes)
				d_FillFrontierQueue<THREADS_PER_BLOCK>(newHugeNodesFrontier, hugeThreadFrontier, n_huge);
		}
	}

	__device__
		void d_SwapQueues(dFrontierQueue &queue1, dFrontierQueue &queue2)
	{
		dFrontierQueue temp = queue1;
		queue1 = queue2;
		queue2 = temp;
	}

	template<typename VertexDataType, typename EdgeDataType, size_t EDGES_PER_THREAD, size_t THREADS_PER_BLOCK>
	__global__
		void d_bfsClassification(MemoryManager *memory_manager, memory_t *memory, int page_size, vertex_t *frontiers,
			dFrontierQueue newSmallNodesFrontier, dFrontierQueue newMediumNodesFrontier,
			dFrontierQueue newLargeNodesFrontier, dFrontierQueue newHugeNodesFrontier, 
			dFrontierQueue oldSmallNodesFrontier, dFrontierQueue oldMediumNodesFrontier,
			dFrontierQueue oldLargeNodesFrontier, dFrontierQueue oldHugeNodesFrontier, 
			unsigned int *current_max_node_size, vertex_t starting_node)
	{
		VertexDataType *vertices = (VertexDataType*)memory;

		size_t const EDGES_PER_BLOCK = EDGES_PER_THREAD * THREADS_PER_BLOCK;

		unsigned int n_edges = vertices[starting_node].neighbours;
		if (n_edges <= EDGES_PER_THREAD)
		{
			newSmallNodesFrontier.Allocate(1);
			newSmallNodesFrontier.nodes[0] = starting_node;
		}
		else if (n_edges <= EDGES_PER_THREAD * 32)
		{
			newMediumNodesFrontier.Allocate(1);
			newMediumNodesFrontier.nodes[0] = starting_node;
		}
		else if (n_edges <= EDGES_PER_BLOCK)
		{
			newLargeNodesFrontier.Allocate(1);
			newLargeNodesFrontier.nodes[0] = starting_node;
		}
		else
		{
			newHugeNodesFrontier.Allocate(1);
			newHugeNodesFrontier.nodes[0] = starting_node;
		}

		unsigned int iteration = 0;

		do
		{
			//printf("iteration %u\n", iteration);
			d_SwapQueues(newSmallNodesFrontier, oldSmallNodesFrontier);
			newSmallNodesFrontier.Reset();
			d_SwapQueues(newMediumNodesFrontier, oldMediumNodesFrontier);
			newMediumNodesFrontier.Reset();
			d_SwapQueues(newLargeNodesFrontier, oldLargeNodesFrontier);
			newLargeNodesFrontier.Reset();
			d_SwapQueues(newHugeNodesFrontier, oldHugeNodesFrontier);
			newHugeNodesFrontier.Reset();

			//printf("Queue sizes: %u, %u, %u, %u\n", *data.oldSmallNodesFrontier.size, *data.oldMediumNodesFrontier.size, *data.oldLargeNodesFrontier.size, *data.oldHugeNodesFrontier.size);
			if (*oldHugeNodesFrontier.size > 0)
			{
				if (*oldHugeNodesFrontier.size <= THREADS_PER_BLOCK)
					d_GetMaxNodeSize_kernel<VertexDataType, THREADS_PER_BLOCK, 1> << <1, THREADS_PER_BLOCK >> > (oldHugeNodesFrontier, memory, current_max_node_size);
				else if (*oldHugeNodesFrontier.size <= THREADS_PER_BLOCK * 4)
					d_GetMaxNodeSize_kernel<VertexDataType, THREADS_PER_BLOCK, 4> << <1, THREADS_PER_BLOCK >> > (oldHugeNodesFrontier, memory, current_max_node_size);
				else if (*oldHugeNodesFrontier.size <= THREADS_PER_BLOCK * 16)
					d_GetMaxNodeSize_kernel<VertexDataType, THREADS_PER_BLOCK, 16> << <1, THREADS_PER_BLOCK >> > (oldHugeNodesFrontier, memory, current_max_node_size);
				else if (*oldHugeNodesFrontier.size <= THREADS_PER_BLOCK * 64)
					d_GetMaxNodeSize_kernel<VertexDataType, THREADS_PER_BLOCK, 64> << <1, THREADS_PER_BLOCK >> > (oldHugeNodesFrontier, memory, current_max_node_size);
				else
					d_GetMaxNodeSize_kernel<VertexDataType, THREADS_PER_BLOCK, 128> << <1, THREADS_PER_BLOCK >> > (oldHugeNodesFrontier, memory, current_max_node_size);
				cudaDeviceSynchronize();

				unsigned int n_blocks = (*current_max_node_size + EDGES_PER_BLOCK - 1) / EDGES_PER_BLOCK;
				d_ExploreHugeNodesFrontierClassification_kernel<VertexDataType, EdgeDataType, EDGES_PER_THREAD, THREADS_PER_BLOCK> << <n_blocks, THREADS_PER_BLOCK >> > (
					oldHugeNodesFrontier, memory_manager, memory, page_size, frontiers,
					newSmallNodesFrontier, newMediumNodesFrontier, newLargeNodesFrontier, newHugeNodesFrontier,
					n_blocks * THREADS_PER_BLOCK, iteration);
			}

			if (*oldSmallNodesFrontier.size > 0)
			{
				d_ExploreFrontierClassification_kernel<VertexDataType, EdgeDataType, EDGES_PER_THREAD, THREADS_PER_BLOCK>
					<< <(*oldSmallNodesFrontier.size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >
					(oldSmallNodesFrontier, memory_manager, memory, page_size, frontiers, 
					newSmallNodesFrontier, newMediumNodesFrontier, newLargeNodesFrontier, newHugeNodesFrontier, 
					1, iteration);
			}
			if (*oldMediumNodesFrontier.size > 0)
			{
				d_ExploreFrontierClassification_kernel<VertexDataType, EdgeDataType, EDGES_PER_THREAD, THREADS_PER_BLOCK>
					<< <((*oldMediumNodesFrontier.size * 32) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >
					(oldMediumNodesFrontier, memory_manager, memory, page_size, frontiers,
					newSmallNodesFrontier, newMediumNodesFrontier, newLargeNodesFrontier, newHugeNodesFrontier,
					32, iteration);
			}
			if (*oldLargeNodesFrontier.size > 0)
			{
				d_ExploreFrontierClassification_kernel<VertexDataType, EdgeDataType, EDGES_PER_THREAD, THREADS_PER_BLOCK>
					<< <*oldLargeNodesFrontier.size, THREADS_PER_BLOCK >> >
					(oldLargeNodesFrontier, memory_manager, memory, page_size, frontiers,
					newSmallNodesFrontier, newMediumNodesFrontier, newLargeNodesFrontier, newHugeNodesFrontier,
					THREADS_PER_BLOCK, iteration);
			}

			cudaDeviceSynchronize();
			iteration++;
			//printf("Queue sizes: %u, %u, %u, %u\n", *data.newSmallNodesFrontier.size, *data.newMediumNodesFrontier.size, *data.newLargeNodesFrontier.size, *data.newHugeNodesFrontier.size);

		} while (*newSmallNodesFrontier.size > 0
			|| *newMediumNodesFrontier.size > 0
			|| *newLargeNodesFrontier.size > 0);
	}
}

template <typename VertexDataType, typename EdgeDataType>
std::vector<vertex_t> BFS<VertexDataType, EdgeDataType>::algBFSBasic(const std::unique_ptr<MemoryManager>& memory_manager, IndividualTimings& timing, vertex_t start_vertex)
{
	cudaEvent_t start_allocation, end_allocation, start_kernel, end_kernel, start_copy, end_copy;

	int *dev_found_new_nodes;
	vertex_t *dev_frontier;

	start_clock(start_allocation, end_allocation);
	cudaMalloc((void**)&dev_found_new_nodes, sizeof(int));
	cudaMalloc((void**)&dev_frontier, sizeof(vertex_t) * memory_manager->number_vertices);

	cudaMemset(dev_frontier, faimGraphBFS::NOT_VISITIED, sizeof(vertex_t) * memory_manager->number_vertices);
	float allocation_time = end_clock(start_allocation, end_allocation);

	start_clock(start_kernel, end_kernel);
	faimGraphBFS::d_bfsBasic <VertexDataType, EdgeDataType, 256> <<<1, 1>>>
		((MemoryManager*)memory_manager->d_memory, memory_manager->d_data, memory_manager->page_size, dev_found_new_nodes, dev_frontier, start_vertex);
	float kernel_time = end_clock(start_kernel, end_kernel);

	start_clock(start_copy, end_copy);
	std::vector<vertex_t> result;
	result.reserve(memory_manager->number_vertices);
	cudaMemcpy(&result[0], dev_frontier, sizeof(vertex_t) * memory_manager->number_vertices, cudaMemcpyDeviceToHost);
	float copy_time = end_clock(start_copy, end_copy);

	cudaFree(dev_found_new_nodes);
	cudaFree(dev_frontier);

	//printf("algBFSBasic done with kernel time: %fms, allocation time %fms, and result copy time %fms\n", kernel_time, allocation_time, copy_time);

	timing.overall_alloc += allocation_time;
	timing.overall_kernel += kernel_time;
	timing.overall_cpy += copy_time;

	return result;
}

template <typename VertexDataType, typename EdgeDataType>
std::vector<vertex_t> BFS<VertexDataType, EdgeDataType>::algBFSDynamicParalellism(const std::unique_ptr<MemoryManager>& memory_manager, IndividualTimings& timing, vertex_t start_vertex)
{
	cudaEvent_t start_allocation, end_allocation, start_kernel, end_kernel, start_copy, end_copy;

	size_t launch_limit;
	cudaDeviceGetLimit(&launch_limit, cudaLimitDevRuntimePendingLaunchCount);
	cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 32768);

	vertex_t *dev_frontier;

	start_clock(start_allocation, end_allocation);
	faimGraphBFS::dFrontierQueue newFrontierQueue(memory_manager->number_vertices);
	faimGraphBFS::dFrontierQueue oldFrontierQueue(memory_manager->number_vertices);

	cudaMalloc((void**)&dev_frontier, sizeof(vertex_t) * memory_manager->number_vertices);

	cudaMemset(dev_frontier, faimGraphBFS::NOT_VISITIED, sizeof(vertex_t) * memory_manager->number_vertices);
	float allocation_time = end_clock(start_allocation, end_allocation);

	start_clock(start_kernel, end_kernel);
	faimGraphBFS::d_bfsDynamicParalellism <VertexDataType, EdgeDataType, 64, 256> << <1, 1 >> >
		((MemoryManager*)memory_manager->d_memory, memory_manager->d_data, memory_manager->page_size, dev_frontier,
			newFrontierQueue, oldFrontierQueue, start_vertex);
	float kernel_time = end_clock(start_kernel, end_kernel);

	start_clock(start_copy, end_copy);
	std::vector<vertex_t> result;
	result.reserve(memory_manager->number_vertices);
	cudaMemcpy(&result[0], dev_frontier, sizeof(vertex_t) * memory_manager->number_vertices, cudaMemcpyDeviceToHost);
	float copy_time = end_clock(start_copy, end_copy);

	newFrontierQueue.Free();
	oldFrontierQueue.Free();
	
	cudaFree(dev_frontier);

	cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, launch_limit);

	//printf("algBFSDynamicParalellism done with kernel time: %fms, allocation time %fms, and result copy time %fms\n", kernel_time, allocation_time, copy_time);

	timing.overall_alloc += allocation_time;
	timing.overall_kernel += kernel_time;
	timing.overall_cpy += copy_time;

	return result;
}

template <typename VertexDataType, typename EdgeDataType>
std::vector<vertex_t> BFS<VertexDataType, EdgeDataType>::algBFSPreprocessing(const std::unique_ptr<MemoryManager>& memory_manager, IndividualTimings& timing, vertex_t start_vertex)
{
	cudaEvent_t start_allocation, end_allocation, start_kernel, end_kernel, start_copy, end_copy;

	vertex_t *dev_frontier;
	unsigned int *current_max_node_size; // Used for huge frontier handling

	start_clock(start_allocation, end_allocation);
	faimGraphBFS::dFrontierQueue rawFrontierQueue(memory_manager->number_vertices);
	faimGraphBFS::dFrontierQueue smallNodesQueue(memory_manager->number_vertices);
	faimGraphBFS::dFrontierQueue mediumNodesQueue(memory_manager->number_vertices);
	faimGraphBFS::dFrontierQueue largeNodesQueue(memory_manager->number_vertices);
	faimGraphBFS::dFrontierQueue hugeNodesQueue(memory_manager->number_vertices);

	cudaMalloc((void**)&dev_frontier, sizeof(vertex_t) * memory_manager->number_vertices);
	cudaMalloc((void**)&current_max_node_size, sizeof(unsigned int));

	cudaMemset(dev_frontier, faimGraphBFS::NOT_VISITIED, sizeof(vertex_t) * memory_manager->number_vertices);
	float allocation_time = end_clock(start_allocation, end_allocation);

	start_clock(start_kernel, end_kernel);
	faimGraphBFS::d_BFSPreprocessing<VertexDataType, EdgeDataType, 4, 128> << <1, 1 >> >
		((MemoryManager*)memory_manager->d_memory, memory_manager->d_data, memory_manager->page_size, dev_frontier,
			rawFrontierQueue, smallNodesQueue, mediumNodesQueue, largeNodesQueue, hugeNodesQueue, current_max_node_size, start_vertex);
	float kernel_time = end_clock(start_kernel, end_kernel);

	start_clock(start_copy, end_copy);
	std::vector<vertex_t> result;
	result.reserve(memory_manager->number_vertices);
	cudaMemcpy(&result[0], dev_frontier, sizeof(vertex_t) * memory_manager->number_vertices, cudaMemcpyDeviceToHost);
	float copy_time = end_clock(start_copy, end_copy);

	rawFrontierQueue.Free();
	smallNodesQueue.Free();
	mediumNodesQueue.Free();
	largeNodesQueue.Free();
	hugeNodesQueue.Free();

	cudaFree(dev_frontier);
	cudaFree(current_max_node_size);

	//printf("algBFSPreprocessing done with kernel time: %fms, allocation time %fms\n", kernel_time, allocation_time);

	timing.overall_alloc += allocation_time;
	timing.overall_kernel += kernel_time;
	timing.overall_cpy += copy_time;

	return result;
}

template <typename VertexDataType, typename EdgeDataType>
std::vector<vertex_t> BFS<VertexDataType, EdgeDataType>::algBFSClassification(const std::unique_ptr<MemoryManager>& memory_manager, IndividualTimings& timing, vertex_t start_vertex)
{
	cudaEvent_t start_allocation, end_allocation, start_kernel, end_kernel, start_copy, end_copy;

	vertex_t *dev_frontier;
	unsigned int *current_max_node_size; // Used for huge frontier handling

	start_clock(start_allocation, end_allocation);
	faimGraphBFS::dFrontierQueue newSmallNodesQueue(memory_manager->number_vertices);
	faimGraphBFS::dFrontierQueue newMediumNodesQueue(memory_manager->number_vertices);
	faimGraphBFS::dFrontierQueue newLargeNodesQueue(memory_manager->number_vertices);
	faimGraphBFS::dFrontierQueue newHugeNodesQueue(memory_manager->number_vertices);
	faimGraphBFS::dFrontierQueue oldSmallNodesQueue(memory_manager->number_vertices);
	faimGraphBFS::dFrontierQueue oldMediumNodesQueue(memory_manager->number_vertices);
	faimGraphBFS::dFrontierQueue oldLargeNodesQueue(memory_manager->number_vertices);
	faimGraphBFS::dFrontierQueue oldHugeNodesQueue(memory_manager->number_vertices);

	cudaMalloc((void**)&dev_frontier, sizeof(vertex_t) * memory_manager->number_vertices);
	cudaMalloc((void**)&current_max_node_size, sizeof(unsigned int));

	cudaMemset(dev_frontier, faimGraphBFS::NOT_VISITIED, sizeof(vertex_t) * memory_manager->number_vertices);
	float allocation_time = end_clock(start_allocation, end_allocation);

	start_clock(start_kernel, end_kernel);
	faimGraphBFS::d_bfsClassification<VertexDataType, EdgeDataType, 16, 256> << <1, 1 >> >
		((MemoryManager*)memory_manager->d_memory, memory_manager->d_data, memory_manager->page_size, dev_frontier,
			newSmallNodesQueue, newMediumNodesQueue, newLargeNodesQueue, newHugeNodesQueue,
			oldSmallNodesQueue, oldMediumNodesQueue, oldLargeNodesQueue, oldHugeNodesQueue,
			current_max_node_size, start_vertex);
	float kernel_time = end_clock(start_kernel, end_kernel);

	start_clock(start_copy, end_copy);
	std::vector<vertex_t> result;
	result.reserve(memory_manager->number_vertices);
	cudaMemcpy(&result[0], dev_frontier, sizeof(vertex_t) * memory_manager->number_vertices, cudaMemcpyDeviceToHost);
	float copy_time = end_clock(start_copy, end_copy);

	oldSmallNodesQueue.Free();
	oldMediumNodesQueue.Free();
	oldLargeNodesQueue.Free();
	oldHugeNodesQueue.Free();
	newSmallNodesQueue.Free();
	newMediumNodesQueue.Free();
	newLargeNodesQueue.Free();
	newHugeNodesQueue.Free();

	cudaFree(dev_frontier);
	cudaFree(current_max_node_size);

	//printf("algBFSClassification done with kernel time: %fms, allocation time %fms, and result copy time %fms\n", kernel_time, allocation_time, copy_time);

	timing.overall_alloc += allocation_time;
	timing.overall_kernel += kernel_time;
	timing.overall_cpy += copy_time;

	return result;
}

template std::vector<vertex_t> BFS<VertexData, EdgeData>::algBFSBasic(const std::unique_ptr<MemoryManager>& memory_manager, IndividualTimings& timing, vertex_t start_vertex);
template std::vector<vertex_t> BFS<VertexDataWeight, EdgeDataWeight>::algBFSBasic(const std::unique_ptr<MemoryManager>& memory_manager, IndividualTimings& timing,vertex_t start_vertex);
template std::vector<vertex_t> BFS<VertexDataSemantic, EdgeDataSemantic>::algBFSBasic(const std::unique_ptr<MemoryManager>& memory_manager, IndividualTimings& timing,vertex_t start_vertex);
template std::vector<vertex_t> BFS<VertexData, EdgeDataSOA>::algBFSBasic(const std::unique_ptr<MemoryManager>& memory_manager, IndividualTimings& timing,vertex_t start_vertex);
template std::vector<vertex_t> BFS<VertexDataWeight, EdgeDataWeightSOA>::algBFSBasic(const std::unique_ptr<MemoryManager>& memory_manager, IndividualTimings& timing,vertex_t start_vertex);
template std::vector<vertex_t> BFS<VertexDataSemantic, EdgeDataSemanticSOA>::algBFSBasic(const std::unique_ptr<MemoryManager>& memory_manager, IndividualTimings& timing,vertex_t start_vertex);

template std::vector<vertex_t> BFS<VertexData, EdgeData>::algBFSDynamicParalellism(const std::unique_ptr<MemoryManager>& memory_manager, IndividualTimings& timing,vertex_t start_vertex);
template std::vector<vertex_t> BFS<VertexDataWeight, EdgeDataWeight>::algBFSDynamicParalellism(const std::unique_ptr<MemoryManager>& memory_manager, IndividualTimings& timing,vertex_t start_vertex);
template std::vector<vertex_t> BFS<VertexDataSemantic, EdgeDataSemantic>::algBFSDynamicParalellism(const std::unique_ptr<MemoryManager>& memory_manager, IndividualTimings& timing,vertex_t start_vertex);
template std::vector<vertex_t> BFS<VertexData, EdgeDataSOA>::algBFSDynamicParalellism(const std::unique_ptr<MemoryManager>& memory_manager, IndividualTimings& timing,vertex_t start_vertex);
template std::vector<vertex_t> BFS<VertexDataWeight, EdgeDataWeightSOA>::algBFSDynamicParalellism(const std::unique_ptr<MemoryManager>& memory_manager, IndividualTimings& timing,vertex_t start_vertex);
template std::vector<vertex_t> BFS<VertexDataSemantic, EdgeDataSemanticSOA>::algBFSDynamicParalellism(const std::unique_ptr<MemoryManager>& memory_manager, IndividualTimings& timing,vertex_t start_vertex);

template std::vector<vertex_t> BFS<VertexData, EdgeData>::algBFSPreprocessing(const std::unique_ptr<MemoryManager>& memory_manager, IndividualTimings& timing,vertex_t start_vertex);
template std::vector<vertex_t> BFS<VertexDataWeight, EdgeDataWeight>::algBFSPreprocessing(const std::unique_ptr<MemoryManager>& memory_manager, IndividualTimings& timing,vertex_t start_vertex);
template std::vector<vertex_t> BFS<VertexDataSemantic, EdgeDataSemantic>::algBFSPreprocessing(const std::unique_ptr<MemoryManager>& memory_manager, IndividualTimings& timing,vertex_t start_vertex);
template std::vector<vertex_t> BFS<VertexData, EdgeDataSOA>::algBFSPreprocessing(const std::unique_ptr<MemoryManager>& memory_manager, IndividualTimings& timing,vertex_t start_vertex);
template std::vector<vertex_t> BFS<VertexDataWeight, EdgeDataWeightSOA>::algBFSPreprocessing(const std::unique_ptr<MemoryManager>& memory_manager, IndividualTimings& timing,vertex_t start_vertex);
template std::vector<vertex_t> BFS<VertexDataSemantic, EdgeDataSemanticSOA>::algBFSPreprocessing(const std::unique_ptr<MemoryManager>& memory_manager, IndividualTimings& timing,vertex_t start_vertex);

template std::vector<vertex_t> BFS<VertexData, EdgeData>::algBFSClassification(const std::unique_ptr<MemoryManager>& memory_manager, IndividualTimings& timing,vertex_t start_vertex);
template std::vector<vertex_t> BFS<VertexDataWeight, EdgeDataWeight>::algBFSClassification(const std::unique_ptr<MemoryManager>& memory_manager, IndividualTimings& timing,vertex_t start_vertex);
template std::vector<vertex_t> BFS<VertexDataSemantic, EdgeDataSemantic>::algBFSClassification(const std::unique_ptr<MemoryManager>& memory_manager, IndividualTimings& timing,vertex_t start_vertex);
template std::vector<vertex_t> BFS<VertexData, EdgeDataSOA>::algBFSClassification(const std::unique_ptr<MemoryManager>& memory_manager, IndividualTimings& timing,vertex_t start_vertex);
template std::vector<vertex_t> BFS<VertexDataWeight, EdgeDataWeightSOA>::algBFSClassification(const std::unique_ptr<MemoryManager>& memory_manager, IndividualTimings& timing,vertex_t start_vertex);
template std::vector<vertex_t> BFS<VertexDataSemantic, EdgeDataSemanticSOA>::algBFSClassification(const std::unique_ptr<MemoryManager>& memory_manager, IndividualTimings& timing,vertex_t start_vertex);
