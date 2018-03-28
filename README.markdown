# aimGraph

The following repository holds data and source code which is part of the masterproject/thesis of Martin Winter


## General Information

This code base represents aimGraph (Autonomous, Independent Management of Dynamic Graphs on GPUs)

`IMPORTANT NOTE`:
Since this is currently in development as the on-going master thesis of Martin Winter, this represents a snapshot as of 18.09.2017 and not a completely finished project, this means

- Queuing approach already implemented
- Fully Dynamic also implemented (but not tested and validated yet)
- Graph Directionality not fully implemented (in undirected mode, currently both edges (src-dst and dst-src) have
  to be inserted separately, will be automatic later on)
- Memory Layout quite different from paper, but should work in similar fashion and produce the same results
- STC not fully done yet

The framework currently can handle graphs provided in CSR format as found here:
http://www.cc.gatech.edu/dimacs10/

Graphs in other formats would need to be converted to CSR format first.

The framework can be configured using an xml-style configuration file (Description below).

`IMPORTANT NOTE`:
Since the memory management is maintained entirely by the framework, it is very important to choose the memory requirements (memory size, stacksize, queuesize) correctly as it is possible to overwrite graph data by incorrectly choosing the stack size for example


## Setup

Requirements:
- CUDA 9.1 
- gcc-6 or MVSC (VS17) (c++14 support required)
- CMake 3.2 or higher

Linux:
To setup and build the project, just run setup.sh

Windows:
Use CMake to setup project and build using Visual Studio 2017


## Running project

To run the basic edge update testcase similar to cuSTINGER testcase, run:
./mainaimGraph configuration.xml


## Configuration

Each configuration file contains exactly one global section and at least ONE testrun section, but there can be multiple testruns specified.

global
	deviceID
		Which device should be used to run aimGraph (e.g. 0)
	devicememsize
		How much memory is allocated by aimGraph (in GB, e.g. 1.75)
testrun
	configuration
		memorylayout
			Defines if AOS (Array of Structures) or SOA (Structure of Arrays) should be used
				Standard version uses AOS
		graphmode
			Sets up which graphmode is used
				simple - weight - semantic
		updatevariant
			Can be currently "standard", "warpsized" or "specialised"
				Defines how the update should occur (1 thread or 1 warp per update)
				Warpsized approach requires a blocksize of 128 and update kernel launch size of 32
				Standard approach should be flexible
		rounds
			Number of rounds for this testrun, the graphstructure will be initialized and torn down this much
		updaterounds
			Number of individual update rounds within a round, for each update round batch size of edges is inserted and removed again (can be either the same updates or "realistic" updates, this can be changed in main.cpp via parameter "realisticDeletion")
		edgeblocksize
			Size of edgeblocks that are given out by the memory manager in Bytes
				For warpsized approach has to be 128
		initlaunchblocksize
			Kernel launch size for the Initialization (e.g. 32)
		insertlaunchblocksize
			Kernel launch size for the Insertion (e.g. 256)
				For warpsized approach has to be 32
		deletelaunchblocksize
			Kernel launch size for the Deletion (e.g. 256)
				For warpsized approach has to be 32
		deletionvariation
			Which Deletion variation should be used, can be currently "standard" or "compaction" (needed for queuing)
				Which are available depends on the updatevariant
		verification
			If verification should be performed to check if graph structure on the device matches graph on host (VERY SLOW)
		performanceoutput
			Currently either "csv" or "stdout", changes where the performance output is written to
		memoryoverallocation
			Is the overallocation factor that determines how much more memory is allocated per adjacency (should be 1.0, was just introduced for testing purposes)
		stacksize
			Stacksize in Bytes
		queuesize
			Queuesize in number of items per queue
		directionality
			undirected - vs directed
				Currently not fully implemented
	graphs
		filename
			Can be given multiple times for multiple graphs
	batches
		batchsize
			Can be given multiple times for multiple batchsizes


Performs testruns as:

```plain
for (auto batchsize : testrun->batchsizes)
{
	for (const auto& graph : testrun->graphs)
	{
		// Run update procedure on graph with batchsize
		
		for(rounds)
		{
			// Setup aimGraph

			for(updaterounds)
			{
				// Insert Edgeupdatebatch

				// Delete Edgeupdatebatch
			}
		}
	}
}
```