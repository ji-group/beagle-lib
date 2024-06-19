/*
 *  BeagleGPUActionImpl.h
 *  BEAGLE
 *
 * Copyright 2024 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * BEAGLE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * BEAGLE is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with BEAGLE.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @author Xiang Ji
 * @author Benjamin Redelings
 * @author Marc Suchard
 */


#ifndef BEAGLE_BEAGLEGPUACTIONIMPL_H
#define BEAGLE_BEAGLEGPUACTIONIMPL_H

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/GPU/BeagleGPUImpl.h"
#include <vector>
#include <random>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMV
#include <cublas_v2.h>

using std::vector;
using std::tuple;
using Eigen::MatrixXi;

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)


template <typename T>
double normP1(const T& matrix) {
    return (Eigen::RowVectorXd::Ones(matrix.rows()) * matrix.cwiseAbs()).maxCoeff();
}

template <typename T>
tuple<double,int> ArgNormP1(const T& matrix)
{
    int x=-1;
    double v = matrix.colwise().template lpNorm<1>().maxCoeff(&x);
    return {v,x};
}

template <typename T>
auto normPInf(const T& matrix) {
    return matrix.template lpNorm<Eigen::Infinity>();
}

template <typename Real>
Real normPInf(Real* matrix, int nRows, int nCols, cublasHandle_t cublasH) {
    int index;
    Real result;
    if constexpr (std::is_same<Real, float>::value) {
        CUBLAS_CHECK(cublasIsamax(cublasH, nRows * nCols, matrix, 1, &index));
    } else {
        CUBLAS_CHECK(cublasIdamax(cublasH, nRows * nCols, matrix, 1, &index));
    }
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaMemcpy(&result, matrix + index - 1, sizeof(Real), cudaMemcpyDeviceToHost))
    return std::abs(result);
}

template <typename Real>
using SpMatrix = Eigen::SparseMatrix<Real, Eigen::StorageOptions::RowMajor>;

template <typename Real>
using Triplet = Eigen::Triplet<Real>;

template <typename Real>
using DnMatrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;

template <typename Real>
using DnVector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

template <typename T> cusparseIndexType_t IndexType;
template <> cusparseIndexType_t IndexType<int32_t> = CUSPARSE_INDEX_32I;
template <> cusparseIndexType_t IndexType<int64_t> = CUSPARSE_INDEX_64I;

template <typename T> cudaDataType DataType;
template <> cudaDataType DataType<float> = CUDA_R_32F;
template <> cudaDataType DataType<double> = CUDA_R_64F;

template <typename Real>
struct DnMatrixDevice
{
    cublasHandle_t cublasHandle = nullptr;
    // Lets assume any cusparseHandle_t comes from a sparse matrix.
    cusparseDnMatDescr_t descr = nullptr;
    Real* ptr = nullptr;
    size_t size1 = 0;
    size_t size2 = 0;
    cusparseOrder_t order = CUSPARSE_ORDER_COL;
    size_t size() const {return size1*size2;}
    size_t byte_size() const {return size()*sizeof(Real);}

    // Disallow copying -- only one object can "own" the descriptor.
    DnMatrixDevice<Real>& operator=(const DnMatrixDevice<Real>&) = delete;
    // Allow moving.
    DnMatrixDevice<Real>& operator=(DnMatrixDevice<Real>&& D) noexcept
    {
	std::swap(cublasHandle, D.cublasHandle);
	std::swap(descr, D.descr);
	std::swap(ptr, D.ptr);
	std::swap(size1, D.size1);
	std::swap(size2, D.size2);
	std::swap(order, D.order);
	return *this;
    }

    DnMatrixDevice<Real>& operator+=(const DnMatrixDevice<Real>& D)
    {
	assert(D.size1 == size1);
	assert(D.size2 == size2);
	assert(D.cublasHandle == cublasHandle);
	Real one = 1;
	if constexpr (std::is_same<Real, float>::value) {
	    cublasSaxpy(cublasHandle, size(), &one, D.ptr, 1, ptr, 1);
	} else {
	    cublasDaxpy(cublasHandle, size(), &one, D.ptr, 1, ptr, 1);
	}

	return *this;
    }

    // Disallow copying.
    DnMatrixDevice(const DnMatrixDevice<Real>&) = delete;
    // Allow moving.
    DnMatrixDevice(DnMatrixDevice<Real>&& D) noexcept
    {
	operator=(std::move(D));
    }

    DnMatrixDevice() = default;
    DnMatrixDevice(cublasHandle_t cb, Real* p, size_t s1, size_t s2, cusparseOrder_t o = CUSPARSE_ORDER_COL)
	:cublasHandle(cb), ptr(p), size1(s1), size2(s2), order(o)
    {
	auto status = (order == CUSPARSE_ORDER_COL)
	    ? cusparseCreateDnMat(&descr, size1, size2, size1, ptr, DataType<Real>, CUSPARSE_ORDER_COL)
	    : cusparseCreateDnMat(&descr, size1, size2, size2, ptr, DataType<Real>, CUSPARSE_ORDER_ROW);

	if (status != CUSPARSE_STATUS_SUCCESS)
	{
	    std::cerr<<"cusparseCreateDnMat( ) failed!";
	    std::exit(1);
	}
    }

    ~DnMatrixDevice()
     {
	 // This class does not own the memory, so does not call cudaFree.
	 if (descr)
	 {
	     cusparseDestroyDnMat(descr);
	     descr = nullptr;
	 }
     }
};

enum class sparseFormat { none, csr, csc };

template <typename Real>
struct SpMatrixDevice
{
    cusparseHandle_t cusparseHandle = nullptr;
    cusparseSpMatDescr_t descr = nullptr;
    int size1 = 0; // rows
    int size2 = 0; // columns
    int num_non_zeros = 0;
    Real* values = nullptr;
    int* inner = nullptr;
    int* offsets = nullptr;
    sparseFormat format = sparseFormat::none;

    int outer_dim_size() const
    {
	// With CSR, we return the number of row.
	if (format == sparseFormat::csr)
	    return size1;
	// With CSC, we return the number of columns;
	else if (format == sparseFormat::csc)
	    return size2;
	else
	    std::abort();
    }

    // Disallow copying -- only one object can "own" the descriptor.
    SpMatrixDevice<Real>& operator=(const SpMatrixDevice<Real>&) = delete;
    // Allow moving.
    SpMatrixDevice<Real>& operator=(SpMatrixDevice<Real>&& D) noexcept
    {
	std::swap(cusparseHandle, D.cusparseHandle);
	std::swap(descr, D.descr);
	std::swap(size1, D.size1);
	std::swap(size2, D.size2);
	std::swap(num_non_zeros, D.num_non_zeros);
	std::swap(values, D.values);
	std::swap(inner, D.inner);
	std::swap(offsets, D.offsets);
	std::swap(format, D.format);
	return *this;
    }

    // Disallow copying.
    SpMatrixDevice(const SpMatrixDevice<Real>&) = delete;
    // Allow moving.
    SpMatrixDevice(SpMatrixDevice<Real>&& D) noexcept
    {
	operator=(std::move(D));
    }

    SpMatrixDevice() = default;
    SpMatrixDevice(cusparseHandle_t h, int s1, int s2, int n, Real* v, int* c, int* o, sparseFormat f)
	:cusparseHandle(h), size1(s1), size2(s2), num_non_zeros(n), values(v), inner(c), offsets(o), format(f)
    {
	cusparseStatus_t status;
	if (format == sparseFormat::csc)
	{
	    status = cusparseCreateCsc(&descr, s1, s2, num_non_zeros, offsets, inner, values,
				       IndexType<int>, IndexType<int>, CUSPARSE_INDEX_BASE_ZERO, DataType<Real>);
	}
	else
	{
	    status = cusparseCreateCsr(&descr, s1, s2, num_non_zeros, offsets, inner, values,
				       IndexType<int>, IndexType<int>, CUSPARSE_INDEX_BASE_ZERO, DataType<Real>);
	}

	if (status != CUSPARSE_STATUS_SUCCESS)
	{
	    std::cerr<<"cusparseCreateSpMat( ) failed!";
	    std::exit(1);
	}
    }

    ~SpMatrixDevice()
    {
	if (descr)
	{
	    cusparseDestroySpMat(descr);
	    descr = nullptr;
	}
    }
};


template <typename Real>
auto normPInf(const DnMatrixDevice<Real>& M)
{
    return normPInf(M.ptr, M.size1, M.size2, M.cublasHandle);
}


//template <typename Real>
//using MapType = DnMatrix<Real>;

namespace beagle {
namespace gpu {

#ifdef CUDA
	namespace cuda {
#else
	namespace opencl {
#endif

BEAGLE_GPU_TEMPLATE
class BeagleGPUActionImpl : public BeagleGPUImpl<BEAGLE_GPU_GENERIC>
{
public:
    const char* getName();

    long long getFlags();

    int getInstanceDetails(BeagleInstanceDetails* retunInfo);

    int createInstance(int tipCount,
		       int partialsBufferCount,
		       int compactBufferCount,
		       int stateCount,
		       int patternCount,
		       int eigenDecompositionCount,
		       int matrixCount,
		       int categoryCount,
		       int scaleBufferCount,
		       int globalResourceNumber,
		       int pluginResourceNumber,
		       long long preferenceFlags,
		       long long requirementFlags);

    int setTipStates(int tipIndex, const int* inStates);

    int setTipPartials(int tipIndex,
                       const Real* inPartials);

    int setPartials(int bufferIndex,
                    const Real* inPartials);

    int setSparseMatrix(int matrixIndex,
                        const int* rowIndices,
                        const int* colIndices,
                        const Real* values,
                        int numNonZeros);

    int updatePartials(const int* operations,
                       int operationCount,
                       int cumulativeScalingIndex);

protected:

    int kPartialsCacheOffset;
    std::vector<SpMatrix<Real>> hInstantaneousMatrices;
    std::vector<SpMatrix<Real>> hBs;
    SpMatrix<Real> hIdentity;
    std::vector<Real> hMuBs;
    std::vector<Real> hB1Norms;
    const int mMax = 55;
    std::vector<std::vector<Real>> hds;
    std::map<int, double> thetaConstants = {
            //The first 30 values are from table A.3 of  Computing Matrix Functions.
            // For double precision, tol = 2^(-53)
            // TODO: maybe calculate this
            {1, 2.29E-16},
            {2, 2.58E-8},
            {3, 1.39E-5},
            {4, 3.40E-4},
            {5, 2.40E-3},
            {6, 9.07E-3},
            {7, 2.38E-2},
            {8, 5.00E-2},
            {9, 8.96E-2},
            {10, 1.44E-1},
            {11, 2.14E-1},
            {12, 3.00E-1},
            {13, 4.00E-1},
            {14, 5.14E-1},
            {15, 6.41E-1},
            {16, 7.81E-1},
            {17, 9.31E-1},
            {18, 1.09},
            {19, 1.26},
            {20, 1.44},
            {21, 1.62},
            {22, 1.82},
            {23, 2.01},
            {24, 2.22},
            {25, 2.43},
            {26, 2.64},
            {27, 2.86},
            {28, 3.08},
            {29, 3.31},
            {30, 3.54},
            //The rest are from table 3.1 of Computing the Action of the Matrix Exponential.
            {35, 4.7},
            {40, 6.0},
            {45, 7.2},
            {50, 8.5},
            {55, 9.9},
    };

//    std::vector<cusparseSpMatDescr_t> dInstantaneousMatrices;
    std::vector<DnMatrixDevice<Real>> dPartialsWrapper;
    std::vector<DnMatrixDevice<Real>> dFLeft;
    std::vector<DnMatrixDevice<Real>> dFRight;
    std::vector<DnMatrixDevice<Real>> dIntegrationTmpLeft;
    std::vector<DnMatrixDevice<Real>> dIntegrationTmpRight;
    std::vector<SpMatrixDevice<Real>> dAs;
    std::vector<size_t> integrationLeftBufferSize;
    std::vector<void*> dIntegrationLeftBuffer;
    std::vector<size_t> integrationRightBufferSize;
    std::vector<void*> dIntegrationRightBuffer;

    std::vector<int *> dBsCsrOffsetsCache;
    std::vector<int *> dBsCsrColumnsCache;
    std::vector<Real*> dBsCsrValuesCache;
    std::vector<Real*> dACscValuesCache;
    std::vector<int> currentCacheNNZs;

    cublasHandle_t cublasHandle;
    cusparseHandle_t cusparseHandle;

    std::vector<int> hEigenMaps;
    std::vector<Real> hEdgeMultipliers;

    std::vector<tuple<int, int>> msCache;
    std::vector<Real> etaCache;
    std::vector<Real> c1Cache;
    std::vector<Real> c2Cache;
    std::vector<Real> alphaCache;
    std::vector<Real> integrationMultipliers;


    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kTipCount;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kPartialsBufferCount;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kStateCount;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kEigenDecompCount;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kCategoryCount;


    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kPaddedStateCount;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::gpu;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kPaddedPatternCount;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kFlags;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kMatrixSize;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kernels;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dScalingFactors;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dPartials;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dPartialsOrigin;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::hCategoryRates;



    int upPartials(bool byPartition,
		   const int *operations,
		   int operationCount,
		   int cumulativeScalingIndex);

    int upPrePartials(bool byPartition,
		      const int *operations,
		      int operationCount,
		      int cumulativeScalingIndex);

    ~BeagleGPUActionImpl();

private:
    char* getInstanceName();

    int updateTransitionMatrices(int eigenIndex,
				 const int* probabilityIndices,
				 const int* firstDerivativeIndices,
				 const int* secondDerivativeIndices,
				 const Real* edgeLengths,
				 int count);
    double getPMax() const;

    int getPartialIndex(int nodeIndex, int categoryIndex);

    int getPartialCacheIndex(int nodeIndex, int categoryIndex);

    void calcPartialsPartials(int destPIndex,
                              int partials1Index,
                              int edgeIndex1,
                              int partials2Index,
                              int edgeIndex2);

    int simpleAction2(int destPIndex, int partialsIndex, int edgeIndex, int category, int matrixIndex, bool left, bool transpose);

    int simpleAction3(int partialsIndex1, int edgeIndex1,
                      int partialsIndex2, int edgeIndex2);

    int cacheAMatrices(int edgeIndex1, int edgeIndex2, bool transpose);

    std::tuple<int,int> getStatistics2(double t, int nCol, double edgeMultiplier,
                                       int eigenIndex) const;

    double getDValue(int p, int eigenIndex) const;

    int PrintfDeviceVector(Real* dPtr, int length, double checkValue, int *signal, Real r);
    int PrintfDeviceVector(int* dPtr, int length, double checkValue, int *signal, Real r);
    int PrintfDeviceVector(cusparseDnMatDescr_t dPtr, int length, double checkValue, int *signal, Real r);

};

BEAGLE_GPU_TEMPLATE
class BeagleGPUActionImplFactory : public BeagleGPUImplFactory<BEAGLE_GPU_GENERIC> {
    virtual BeagleImpl* createImpl(int tipCount,
                                   int partialsBufferCount,
                                   int compactBufferCount,
                                   int stateCount,
                                   int patternCount,
                                   int eigenBufferCount,
                                   int matrixBufferCount,
                                   int categoryCount,
                                   int scaleBufferCount,
                                   int resourceNumber,
                                   int pluginResourceNumber,
                                   long long preferenceFlags,
                                   long long requirementFlags,
                                   int* errorCode);

    virtual const char* getName();
    virtual long long getFlags();
};

} // namspace device
}	// namespace gpu
}	// namespace beagle

#include "libhmsbeagle/GPU/BeagleGPUActionImpl.hpp"

#endif

