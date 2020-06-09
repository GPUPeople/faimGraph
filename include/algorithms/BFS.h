#pragma once

#include "Utility.h"
#include "MemoryManager.h"

struct IndividualTimings
{
	float overall_alloc{0};
	float overall_kernel{0};
	float overall_cpy{0};

	IndividualTimings& operator/=(const float dividend) // compound assignment (does not need to be a member,
  {                           // but often is, to modify the private members)
    /* addition of rhs to *this takes place here */
	overall_alloc /= dividend;
	overall_kernel /= dividend;
	overall_cpy /= dividend;
    return *this; // return the result by reference
  }
};

template <typename VertexDataType, typename EdgeDataType>
class BFS
{
public:
	BFS(std::unique_ptr<MemoryManager>& memory_manager)
	{

	}

	// Execute BFS on 
	std::vector<vertex_t> algBFSBasic(const std::unique_ptr<MemoryManager>& memory_manager, IndividualTimings& timing, vertex_t start_vertex = 0);
	std::vector<vertex_t> algBFSDynamicParalellism(const std::unique_ptr<MemoryManager>& memory_manager, IndividualTimings& timing, vertex_t start_vertex = 0);
	std::vector<vertex_t> algBFSPreprocessing(const std::unique_ptr<MemoryManager>& memory_manager, IndividualTimings& timing, vertex_t start_vertex = 0);
	std::vector<vertex_t> algBFSClassification(const std::unique_ptr<MemoryManager>& memory_manager, IndividualTimings& timing, vertex_t start_vertex = 0);

private:
	vertex_t* some_memory{nullptr};
};