//------------------------------------------------------------------------------
// MemoryManager.h
//
// faimGraph
//
//------------------------------------------------------------------------------
//
//!  Device Memory Manager 
/*!
  Holds the necessary classes for memory management on the device
*/

#pragma once

#include "Utility.h"
#include "ConfigurationParser.h"
#include "Queue.h"

// Forward declaration
class GraphParser;

enum class GraphDirectionality
{
  DIRECTED,
  UNDIRECTED
};

enum class TemporaryMemoryArea
{
  STACK,
  HEAP
};

enum class SortOrder
{
  ASCENDING,
  DESCENDING
};

enum class ErrorCode : unsigned int
{
  NO_ERROR = 0x00000000,
  PAGE_QUEUE_FULL = 0x00000001,
  VERTEX_QUEUE_FULL = 0x00000002,
  OUT_OF_MEMORY = 0x00000004,
  UNKNOWN_ERROR = 0xFFFFFFFF
};

//------------------------------------------------------------------------------
//
/*! \class MemoryManager
    \brief Device Memory Manager holding management data
*/
class MemoryManager
{
  public:
    MemoryManager(uint64_t memory_size, const std::shared_ptr<Config>& config, std::unique_ptr<GraphParser>& graph_parser);
    ~MemoryManager()
    {
      HANDLE_ERROR(cudaFree(d_memory));
    }

    // Global device memory
    memory_t* d_memory; /*!< Holds a pointer to the beginning of the device memory */
    memory_t* d_data;  /*!< Points to the data segment (after memory manager) */
    memory_t* d_stack_pointer; /*!< Holds a pointer to the end of the device memory */
    
    uint64_t start_index; /*!< Holds index offset needed to locate the first page in memory */

    // Queues for pages and vertices
    IndexQueue d_page_queue; /*!< Holds empty page indices */
    IndexQueue d_vertex_queue; /*!< Holds empty vertex indices */

    int edgeblock_lock; /*!< Can be used to control access to the memory manager */
    vertex_t number_vertices; /*!< Holds the number of vertices in use */
    vertex_t number_edges;  /*!< Holds the number of edges in use */
    vertex_t number_pages;  /*!< Holds the number of pages in use */
    vertex_t edges_per_page; /*!< How many edges can fit within a page */

    // Memory related
    uint64_t total_memory; /*!< How much memory is allocated in general */
    uint64_t free_memory; /*!< How much memory is free at the moment */
    
    // Memory-Layout
    uint32_t index_shift; /*!< Index Shift is used together with the multiplier to find the start of individual memory segments */
    uint32_t page_size; /*!< Size of a page in Bytes */
    uint32_t next_free_page; /*!< Indicates the index of the next free block */
    uint32_t next_free_vertex_index; /*!< Indicates the index of the next free vertex position */
    uint32_t access_counter; /*!< Used for testing, indicates how often the memory manager was used to increase space */

    ConfigurationParameters::GraphMode graph_mode; /*!< Holds the current graph_mode */
    GraphDirectionality graph_directionality; /*!< Indicates if the graph is directed/undirected */
    ConfigurationParameters::PageLinkage page_linkage; /*!< Indicates if the graph structure is single or double linked */

    bool initialized{ false };
    unsigned int error_code{ static_cast<unsigned int>(ErrorCode::NO_ERROR) };

    //#############################################################################################################################################################################################
    // Memory Manager functionality
    //#############################################################################################################################################################################################

    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // Initialization functionality
    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    void initialize(const std::shared_ptr<Config>& config);

    //! Sets number of edges per block
    /*!
      This function can be used to set the size of edge blocks (how many edges per block), computes
      the given number depending on the graph mode and the edge block size
    */
    void setEdgesPerBlock();

    //! Sets the size of the queue and sets the initial pointer
    void setQueueSizeAndPosition(int size);

    //! Sets graph meta mode
    void setGraphMode(const std::shared_ptr<Config>& config);

    //! Reset aimGraph
    void resetFaimGraph(vertex_t number_vertices, vertex_t number_edges);

    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // General use
    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    //! Increases the free memory counter
    inline void increaseAvailableMemory(size_t sizeInBytes) { free_memory += sizeInBytes; };

    //! Decreases the free memory counter
    inline void decreaseAvailableMemory(size_t sizeInBytes) { free_memory -= sizeInBytes; };

    //! Reports back the number of edges in memory
    template <typename VertexDataType> size_t numberEdgesInMemory(vertex_t* d_neighbours_count, bool return_count = false);
    template <typename VertexDataType> size_t numberEdgesInMemory(vertex_t* d_neighbours_count, vertex_t vertex_offset, vertex_t number_vertices, bool return_count = false);

    //! Reports back an array with the number of pages in memory
    template <typename VertexDataType> void numberPagesInMemory(vertex_t* d_page_count);

    //! Reports back the number of pages in memory and the accumulated pages
    template <typename VertexDataType> size_t numberPagesInMemory(vertex_t* d_page_count, vertex_t* d_accumulated_page_count);

    //! Workbalance according to page layout
    void workBalanceCalculation(vertex_t* d_accumulated_page_count, vertex_t page_count, vertex_t* d_vertex_indices, vertex_t* d_page_per_vertex_indices);

    //! Reports back the total capacity of edges in memory
    size_t numberEdgesTotalCapacity();

    //! Prints out the number of edges and current capacity
    void printStats(const std::string& pre_text);

    //! Prints the number of edges on the device
    void printNeighboursCount();

    //! Prints the number of edges possible without reallocation
    void printCapacityCount();

    //! Prints an estimation of the initial storage requirements based on number of vertices, edges, update batchsize and the size of individual edgedata
    void estimateInitialStorageRequirements(vertex_t numberVertices, vertex_t numberEdges, int batchsize, int size_of_edgedata);

    //! Prints general stats on the system including number vertices, edges, page fill level...
    void printEssentials(const std::string& text);

    //! Query error code
    void queryErrorCode();

    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // Utility functionality
    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    //! Perform compaction on the adjacency data
    template <typename VertexDataType, typename EdgeDataType>
    void compaction(const std::shared_ptr<Config>& config);

    //! Sort the adjacency data
    template <typename VertexDataType, typename EdgeDataType>
    void sortAdjacency(const std::shared_ptr<Config>& config, SortOrder sort_order);

    //! Test undirected-ness
    template <typename VertexDataType, typename EdgeDataType>
    void testUndirectedness(const std::shared_ptr<Config>& config);

    //! Test self-loops
    template <typename VertexDataType, typename EdgeDataType>
    void testSelfLoops(const std::shared_ptr<Config>& config);

    //! Test duplicates in graph
    template <typename VertexDataType, typename EdgeDataType>
    void testDuplicates(const std::shared_ptr<Config>& config);

    //! Reset the allocation status for certain vertices
    template <typename VertexDataType, typename EdgeDataType>
    void resetAllocationStatus(const std::shared_ptr<Config>& config, vertex_t number_vertices, vertex_t vertex_offset);
};

//------------------------------------------------------------------------------
// Copy Memory Manager data from device to host
//
void updateMemoryManagerHost(std::unique_ptr<MemoryManager>& memory_manager);

//------------------------------------------------------------------------------
// Copy Memory Manager data from host to device
//
void updateMemoryManagerDevice(std::unique_ptr<MemoryManager>& memory_manager);


/*! \class TemporaryMemoryAccess
\brief Convenience class for Temporary Memory Access
*/
class TemporaryMemoryAccess
{

};


/*! \class TemporaryMemoryAccessStack
\brief Grant access to temporary memory in stack area
*/
class TemporaryMemoryAccessStack : public TemporaryMemoryAccess
{
public:
  //! Setup Memory access directly from the stack pointer
  explicit TemporaryMemoryAccessStack(MemoryManager* memory_manager) :
    temp_memory_location{ memory_manager->d_stack_pointer },
    memory_manager_{ memory_manager },
    size_{ 0 } {}

  //! Setup Memory access from a given pointer
  explicit TemporaryMemoryAccessStack(MemoryManager* memory_manager, memory_t* mem_location) :
    temp_memory_location{ mem_location },
    memory_manager_ { memory_manager},
    size_{ 0 } {}

  //! Retrieve a chunk of temporary memory from the fixed size stack region
  template <typename DataType>
  DataType* getTemporaryMemory(size_t size_in_items)
  {
    temp_memory_location -= sizeof(DataType) * size_in_items;
    return reinterpret_cast<DataType*>(temp_memory_location);
  }

private:
  memory_t* temp_memory_location;
  size_t size_; /*!< Size of memory allocated in this scope */
  MemoryManager* memory_manager_; /*!< Raw pointer to memory manager to alter free memory size */
};


/*! \class TemporaryMemoryAccessHeap
\brief Grant access to temporary memory in heap area
*/
class TemporaryMemoryAccessHeap : public TemporaryMemoryAccess
{
public:
  TemporaryMemoryAccessHeap(MemoryManager* memory_manager, vertex_t number_vertex_indices, vertex_t size_of_vertex_data) :
    temp_memory_location{ memory_manager->d_data + (size_of_vertex_data * number_vertex_indices) },
    memory_manager_{ memory_manager }, 
    size_{ 0 } {}

  TemporaryMemoryAccessHeap(MemoryManager* memory_manager, memory_t* current_position) :
    temp_memory_location{ current_position },
    memory_manager_{ memory_manager },
    size_{ 0 } {}

  //! Destructor automatically increases size again
  ~TemporaryMemoryAccessHeap()
  {
    memory_manager_->increaseAvailableMemory(size_);
  }

  template <typename DataType>
  DataType* getTemporaryMemory(size_t size_in_items)
  {
    memory_t* ret_val = temp_memory_location;
    temp_memory_location += sizeof(DataType) * size_in_items;
    size_ -= sizeof(DataType) * size_in_items;
    memory_manager_->decreaseAvailableMemory(sizeof(DataType) * size_in_items);
    return reinterpret_cast<DataType*>(ret_val);
  }

private:
  memory_t* temp_memory_location;
  size_t size_; /*!< Size of memory allocated in this scope */
  MemoryManager* memory_manager_; /*!< Raw pointer to memory manager to alter free memory size */
};

/*! \class TemporaryMemoryAccessHeap
\brief Grant access to temporary memory in heap area
*/
class TemporaryMemoryAccessHeapTop : public TemporaryMemoryAccess
{
public:
  TemporaryMemoryAccessHeapTop(MemoryManager* memory_manager, vertex_t memory_offset_in_Bytes = 0) :
    temp_memory_location{ pageAccess<memory_t>(memory_manager->d_data, memory_manager->next_free_page, memory_manager->page_size, memory_manager->start_index) - memory_offset_in_Bytes },
    memory_manager_{ memory_manager }, 
    size_{ 0 } {}

  TemporaryMemoryAccessHeapTop(MemoryManager* memory_manager, memory_t* current_position) :
    temp_memory_location{ current_position },
    memory_manager_{ memory_manager },
    size_{ 0 } {}

  //! Destructor automatically increases size again
  ~TemporaryMemoryAccessHeapTop()
  {
    memory_manager_->increaseAvailableMemory(size_);
  }

  template <typename DataType>
  DataType* getTemporaryMemory(size_t size_in_items)
  {
    temp_memory_location -= sizeof(DataType) * size_in_items;
    size_ += sizeof(DataType) * size_in_items;
    memory_manager_->decreaseAvailableMemory(sizeof(DataType) * size_in_items);
    return reinterpret_cast<DataType*>(temp_memory_location);
  }

private:
  memory_t* temp_memory_location;
  size_t size_; /*!< Size of memory allocated in this scope */
  MemoryManager* memory_manager_; /*!< Raw pointer to memory manager to alter free memory size */
};


/*! \class ScopedMemoryAccessHelper
\brief Convenience class for memory management

Convencience class that decrements the available memory stats upon construction
and increments again when going out of scope
Can/should be used for stack management if memory is restriced to a certain scope
*/
class ScopedMemoryAccessHelper
{
public:
  //! Constructor taking a raw MemoryManager pointer and the size of the allocated Memory
  /*!
  Upon construction, the available memory is decremented
  */
  ScopedMemoryAccessHelper(MemoryManager* mem_man, size_t size) :
    memory_manager_{ mem_man },
    size_{ size }
  {
    memory_manager_->decreaseAvailableMemory(size_);
  }

  //! Destructor automatically increases size again
  ~ScopedMemoryAccessHelper()
  {
    memory_manager_->increaseAvailableMemory(size_);
  }

  //! Can be used to alter the size of the available memory
  void alterSize(size_t size)
  {
    size_ += size;
    memory_manager_->decreaseAvailableMemory(size);
  }

private:
  size_t size_; /*!< Size of memory allocated in this scope */
  MemoryManager* memory_manager_; /*!< Raw pointer to memory manager to alter free memory size */
};



//------------------------------------------------------------------------------
// RAII funtionality for CSR data
//------------------------------------------------------------------------------
//
class CSRData
{
  public:
    explicit CSRData(const std::unique_ptr<GraphParser>& graph_parser,
                     std::unique_ptr<MemoryManager>& memory_manager,
      unsigned int vertex_offset = 0);
    explicit CSRData(std::unique_ptr<MemoryManager>& memory_manager, 
      unsigned int number_rows, 
      unsigned int vertex_offset = 0);
    ~CSRData()
    {
      // With more sophisticated memory management,
      // reset stack pointer here!
    }

  // CSR data structure holding the initial graph
    vertex_t* d_adjacency;
    vertex_t* d_offset;
    matrix_t* d_matrix_values{nullptr};
    vertex_t* d_neighbours;
    vertex_t* d_capacity;
    vertex_t* d_block_requirements;
    vertex_t* d_mem_requirements;

    // Used to deal with temporary memory management count
    ScopedMemoryAccessHelper scoped_mem_access_counter;
};

//------------------------------------------------------------------------------
// RAII funtionality for CSR data
//------------------------------------------------------------------------------
//
class CSRMatrixData
{
public:
  CSRMatrixData() {};

  // CSR data structure holding the initial graph
  vertex_t* d_adjacency;
  vertex_t* d_offset;
  matrix_t* d_matrix_values;
  vertex_t* d_neighbours;
  vertex_t* d_capacity;
  vertex_t* d_block_requirements;
  vertex_t* d_mem_requirements;

  vertex_t matrix_rows;
  vertex_t matrix_columns;
  vertex_t edge_count;
};

//------------------------------------------------------------------------------
// RAII funtionality for CSR data
//------------------------------------------------------------------------------
//
class aimGraphCSR
{
  public:
    aimGraphCSR(std::unique_ptr<MemoryManager>& memory_manager);
    aimGraphCSR(std::unique_ptr<MemoryManager>& memory_manager, vertex_t vertex_offset, vertex_t number_vertices);
    ~aimGraphCSR()
    {
      free(h_adjacency);
      free(h_offset);
      if(h_matrix_values != nullptr)
        free(h_matrix_values);
    }

  // CSR data structure holding the GPU graph
    vertex_t* d_adjacency;
    matrix_t* d_matrix_values{nullptr};
    vertex_t* d_offset;
    vertex_t* d_mem_requirement;

    vertex_t* h_adjacency;
    matrix_t* h_matrix_values{ nullptr };
    vertex_t* h_offset;

    index_t number_vertices;
    index_t number_edges;

    // Used to deal with temporary memory management count
    ScopedMemoryAccessHelper scoped_mem_access_counter;
};
