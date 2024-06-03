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

using std::vector;
using std::tuple;
using Eigen::MatrixXi;

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


typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;
typedef Eigen::Map<MatrixXd> MapType;
typedef Eigen::SparseMatrix<double> SpMatrix;
typedef Eigen::Triplet<double> Triplet;

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

    int setTipPartials(int tipIndex, const double* inPartials);

    int setSparseMatrix(int matrixIndex,
                        const int* rowIndices,
                        const int* colIndices,
                        const double* values,
                        int numNonZeros);

    int setPatternWeights(const double* inPatternWeights);

    int setStateFrequencies(int stateFrequenciesIndex,
                            const double* inStateFrequencies);

    int setCategoryWeights(int categoryWeightsIndex,
                           const double* inCategoryWeights);

    int getPartials(int bufferIndex,
                    int scaleIndex,
                    double* outPartials);
protected:

    std::vector<SpMatrix> hInstantaneousMatrices;
    SpMatrix hIdentity;
    std::vector<SpMatrix> hBs;
    std::vector<double> hMuBs;
    std::vector<double> hB1Norms;
    const int mMax = 55;
    std::vector<std::vector<double>> ds;
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

    std::vector<cusparseSpMatDescr_t> dInstantaneousMatrices;
    std::vector<cusparseDnMatDescr_t> dPartials;
    std::vector<Real*> dPartialCache;
    Real **dFrequenciesCache, **dWeightsCache;
    std::vector<cusparseDnVecDescr_t> dFrequencies;
    std::vector<cusparseDnVecDescr_t> dWeights;
    int *dMatrixCsrOffsetsCache, *dMatrixCsrColumnsCache;
    Real *dMatrixCsrValuesCache;
    int currentCacheNNZ;
    Real *dPatternWeightsCache;
    cusparseDnVecDescr_t dPatternWeights;

    std::vector<int> hEigenMaps;
    std::vector<double> hEdgeMultipliers;



    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kInitialized;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kTipCount;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kPartialsBufferCount;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kCompactBufferCount;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kStateCount;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kPatternCount;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kEigenDecompCount;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kMatrixCount;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kCategoryCount;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kScaleBufferCount;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kExtraMatrixCount;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kPartitionCount;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kMaxPartitionCount;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kPartitionsInitialised;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kPatternsReordered;

    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::resourceNumber;

    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kTipPartialsBufferCount;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kBufferCount;

    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kInternalPartialsBufferCount;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kPaddedStateCount;

    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::gpu;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kDeviceType;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kDeviceCode;

    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kPaddedPatternCount;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kResultPaddedPatterns;

    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kFlags;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kScaleBufferSize;

    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kSumSitesBlockSize;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kSumSitesBlockCount;

    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kPartialsSize;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kMatrixSize;

    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kEigenValuesSize;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kLastCompactBufferIndex;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kLastTipPartialsBufferIndex;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kernels;

    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::hWeightsCache;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::hFrequenciesCache;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::hPartialsCache;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::hStatesCache;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::hLogLikelihoodsCache;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::hMatrixCache;
//    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dEvec;
//    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dIevc;
//    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dEigenValues;
//    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dWeights;
//    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dFrequencies;

    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dMatrices;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kIndexOffsetMat;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dMatricesOrigin;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dScalingFactors;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dScalingFactorsMaster;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::hRescalingTrigger;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dRescalingTrigger;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kEvecOffset;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kEvalOffset;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kWeightsOffset;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kFrequenciesOffset;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dIntegrationTmp;
//    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dPatternWeights;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dSumLogLikelihood;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dPartialsTmp;

    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kDerivBuffersInitialised;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kMultipleDerivativesLength;

//    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dPartials;
//    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dPartialsOrigin;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::hPartialsOffsets;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kIndexOffsetPat;

    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dStatesOrigin;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dStates;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::hStatesOffsets;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kIndexOffsetStates;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dCompactBuffers;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dTipPartialsBuffers;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::hStreamIndices;

    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dBranchLengths;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dDistanceQueue;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::hDistanceQueue;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dPtrQueue;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::hPtrQueue;

    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dDerivativeQueue;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::hDerivativeQueue;

    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kSitesPerIntegrateBlock;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kSitesPerBlock;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kNumPatternBlocks;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kPaddedPartitionBlocks;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kMaxPaddedPartitionBlocks;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kPaddedPartitionIntegrateBlocks;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kMaxPaddedPartitionIntegrateBlocks;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kUsingMultiGrid;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::hPartitionOffsets;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::hCategoryRates;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::hPatternWeightsCache;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dMaxScalingFactors;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dIndexMaxScalingFactors;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dAccumulatedScalingFactors;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kUsingAutoTranspose;

    void  allocateMultiGridBuffers();


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
				 const double* edgeLengths,
				 int count);
    double getPMax() const;

    int getPartialIndex(int nodeIndex, int categoryIndex);
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

