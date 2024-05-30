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
#include <Eigen/Sparse>
#include <Eigen/Dense>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;
typedef Eigen::Map<MatrixXd> MapType;
typedef Eigen::SparseMatrix<double> SpMatrix;

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

protected:

    std::vector<SpMatrix> hInstantaneousMatrices;

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
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dEvec;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dIevc;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dEigenValues;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dWeights;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dFrequencies;

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
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dPatternWeights;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dSumLogLikelihood;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dPartialsTmp;

    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kDerivBuffersInitialised;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::kMultipleDerivativesLength;

    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dPartials;
    using BeagleGPUImpl<BEAGLE_GPU_GENERIC>::dPartialsOrigin;
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

    int setSparseMatrix(int matrixIndex,
                        const int* rowIndices,
                        const int* colIndices,
                        const double* values,
                        int numNonZeros);

    int updateTransitionMatrices(int eigenIndex,
				 const int* probabilityIndices,
				 const int* firstDerivativeIndices,
				 const int* secondDerivativeIndices,
				 const double* edgeLengths,
				 int count);
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

