//------------------------------------------------------------------------------
// CSR.h
//
// faimGraph
//
//------------------------------------------------------------------------------
//

#pragma once

#include <memory>
#include <algorithm>
#include <math.h>
#include <cstring>

template<typename T>
struct COO;

template<typename T>
struct DenseVector;

template<typename T>
struct CSR
{
	size_t rows, cols, nnz;

	std::unique_ptr<T[]> data;
	std::unique_ptr<unsigned int[]> row_offsets;
	std::unique_ptr<unsigned int[]> col_ids;

	CSR() : rows(0), cols(0), nnz(0) { }
	void alloc(size_t rows, size_t cols, size_t nnz, bool allocData=true);
};


template<typename T>
CSR<T> loadCSR(const char* file);
template<typename T>
void storeCSR(const CSR<T>& mat, const char* file);

template<typename T>
void spmv(DenseVector<T>& res, const CSR<T>& m, const DenseVector<T>& v, bool transpose = false);

template<typename T>
void convert(CSR<T>& res, const COO<T>& coo);
