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
double normPInf(const T& matrix) {
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

    int setSparseMatrix(int matrixIndex,
                        const int* rowIndices,
                        const int* colIndices,
                        const Real* values,
                        int numNonZeros);

    int updatePartials(const int* operations,
                       int operationCount,
                       int cumulativeScalingIndex);

protected:

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
    std::vector<cusparseDnMatDescr_t> dPartialsWrapper;
    std::vector<cusparseDnMatDescr_t> dFLeft;
    std::vector<cusparseDnMatDescr_t> dFRight;
    std::vector<cusparseDnMatDescr_t> dIntegrationTmpLeft;
    std::vector<cusparseDnMatDescr_t> dIntegrationTmpRight;
    std::vector<cusparseSpMatDescr_t> dAs;
    std::vector<Real*> dPartialCache;
    std::vector<Real*> dFLeftCache;
    std::vector<Real*> dFRightCache;
    std::vector<Real*> dIntegrationTmpLeftCache;
    std::vector<Real*> dIntegrationTmpRightCache;
    std::vector<size_t> integrationLeftBufferSize;
    std::vector<size_t> integrationLeftStoredBufferSize;
    std::vector<void*> dIntegrationLeftBuffer;
    std::vector<size_t> integrationRightBufferSize;
    std::vector<size_t> integrationRightStoredBufferSize;
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

    int simpleAction3(int destPIndex1, int partialsIndex1, int edgeIndex1, int matrixIndex1,
                      int destPIndex2, int partialsIndex2, int edgeIndex2, int matrixIndex2,
                      bool transpose);

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

