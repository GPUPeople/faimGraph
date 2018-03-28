# faimGraph

The following repository holds data and source code which is part of the masterproject/thesis of Martin Winter


## General Information

This code base represents faimGraph: High Performance Management of Fully-dynamic Graphs under tight Memory Constraints on the GPU

`IMPORTANT NOTE`:
As this is still a research project, the current state is still prone to misconfiguration, hence it is possible to provide the framework with invalid or "bad" values without a warning(e.g. setting the queue size to 0, hence loosing access to all returned indices). 
The provided configuration files should provide a guideline on how to set up the project to perform as intended, if questions do arise, it would be highly appreciated to seek contact with the authors for clarification.

The framework currently can handle graphs provided in CSR format as found here:
http://www.cc.gatech.edu/dimacs10/

Graphs in other formats would need to be converted to CSR format first.

The framework can be configured using an xml-style configuration file (Description below).

## Setup

Requirements:
- CUDA 9.1 
- gcc-6 or MVSC (VS17) (c++14 support required)
- CMake 3.2 or higher

Linux:
To setup and build the project, just run setup.sh

Windows:
Use CMake to setup project and build using for example Visual Studio 2017


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
			Can be currently "standard", "warpsized", "vertexcentric" or "vertexcentricsorted"
				Defines how the update should occur (1 thread or 1 warp per update) or based on affected vertices (+ respecting sort order)
		rounds
			Number of rounds for this testrun, the graphstructure will be initialized and torn down this much
		updaterounds
			Number of individual update rounds within a round, for each update round batch size of edges is inserted and removed again (can be either the same updates or "realistic" updates, this can be changed in main.cpp via parameter "realisticDeletion")
		pagesize
			Size of a page used for edge data
		initlaunchblocksize
			Kernel launch size for the Initialization (e.g. 32)
		insertlaunchblocksize
			Kernel launch size for the Insertion (e.g. 256)
				For warpsized approach has to be 32
		deletelaunchblocksize
			Kernel launch size for the Deletion (e.g. 256)
				For warpsized approach has to be 32
		deletionvariation
			To utilize efficient memory management, "compaction" has to be enabled, only the standard update approach can still perform "standard" deletion (aka no compaction)
		verification
			If verification should be performed to check if graph structure on the device matches graph on host (VERY SLOW - unoptimized on CPU)
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