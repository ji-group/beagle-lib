
/*
 *  BeagleGPUImpl.cpp
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
#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#ifndef BEAGLE_BEAGLEGPUACTIONIMPL_HPP
#define BEAGLE_BEAGLEGPUACTIONIMPL_HPP

namespace beagle {
namespace gpu {

#ifdef CUDA
    namespace cuda {
#else
    namespace opencl {
#endif

BEAGLE_GPU_TEMPLATE
const char* BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::getName()
{
    return BeagleGPUActionImplFactory<BEAGLE_GPU_GENERIC>::getName();
}


BEAGLE_GPU_TEMPLATE
long long BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::getFlags()
{
    auto flags = BeagleGPUImpl<BEAGLE_GPU_GENERIC>::getFlags();

    flags |= BEAGLE_FLAG_COMPUTATION_ACTION;

    return flags;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::createInstance(int tipCount,
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
                                  long long requirementFlags)
{

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUActionImpl::createInstance\n");
#endif

    kInitialized = 0;

    kTipCount = tipCount;
    kPartialsBufferCount = partialsBufferCount;
    kCompactBufferCount = 0; // XJ: ignore compact buffer count because of using all partials for tip states
    kStateCount = stateCount;
    kEigenDecompCount = eigenDecompositionCount;
    kMatrixCount = matrixCount;
    kCategoryCount = categoryCount;
    kScaleBufferCount = scaleBufferCount;

    kPartitionCount = 1; // XJ: seems related to stream setup
    kMaxPartitionCount = kPartitionCount;
    kPartitionsInitialised = false;
    kPatternsReordered = false;

    resourceNumber = globalResourceNumber;

    kTipPartialsBufferCount = kTipCount; // use Partials for all tip states
    kBufferCount = kPartialsBufferCount;

    kInternalPartialsBufferCount = kBufferCount - kTipCount;

    //TODO: check if pad state is necessary
    kPaddedStateCount = kStateCount;

    gpu = new GPUInterface();

    gpu->Initialize();

    int numDevices = 0;
    numDevices = gpu->GetDeviceCount();
    if (numDevices == 0) {
        fprintf(stderr, "Error: No GPU devices\n");
        return BEAGLE_ERROR_NO_RESOURCE;
    }
    if (pluginResourceNumber > numDevices) {
        fprintf(stderr,"Error: Trying to initialize device # %d (which does not exist)\n",resourceNumber);
        return BEAGLE_ERROR_NO_RESOURCE;
    }

    //TODO: check if pad patterns is necessary, ignored for now
    bool CPUImpl = false;

    kDeviceType = gpu->GetDeviceTypeFlag(pluginResourceNumber);
    kDeviceCode = gpu->GetDeviceImplementationCode(pluginResourceNumber);

    kPatternCount = patternCount;
    kPaddedPatternCount = kPatternCount;
    kScaleBufferSize = kPaddedPatternCount;

    kFlags = 0;

    if (preferenceFlags & BEAGLE_FLAG_SCALING_AUTO || requirementFlags & BEAGLE_FLAG_SCALING_AUTO) {
        kFlags |= BEAGLE_FLAG_SCALING_AUTO;
        kFlags |= BEAGLE_FLAG_SCALERS_LOG;
        kScaleBufferCount = kInternalPartialsBufferCount;
        kScaleBufferSize *= kCategoryCount;
    } else if (preferenceFlags & BEAGLE_FLAG_SCALING_ALWAYS || requirementFlags & BEAGLE_FLAG_SCALING_ALWAYS) {
        kFlags |= BEAGLE_FLAG_SCALING_ALWAYS;
        kFlags |= BEAGLE_FLAG_SCALERS_LOG;
        kScaleBufferCount = kInternalPartialsBufferCount + 1; // +1 for temp buffer used by edgelikelihood
    } else if (preferenceFlags & BEAGLE_FLAG_SCALING_DYNAMIC || requirementFlags & BEAGLE_FLAG_SCALING_DYNAMIC) {
        kFlags |= BEAGLE_FLAG_SCALING_DYNAMIC;
        kFlags |= BEAGLE_FLAG_SCALERS_RAW;
    } else if (preferenceFlags & BEAGLE_FLAG_SCALERS_LOG || requirementFlags & BEAGLE_FLAG_SCALERS_LOG) {
        kFlags |= BEAGLE_FLAG_SCALING_MANUAL;
        kFlags |= BEAGLE_FLAG_SCALERS_LOG;
    } else {
        kFlags |= BEAGLE_FLAG_SCALING_MANUAL;
        kFlags |= BEAGLE_FLAG_SCALERS_RAW;
    }
    //TODO: remember to implement scaling

//    if (preferenceFlags & BEAGLE_FLAG_EIGEN_COMPLEX || requirementFlags & BEAGLE_FLAG_EIGEN_COMPLEX) {
//        kFlags |= BEAGLE_FLAG_EIGEN_COMPLEX;
//    } else {
//        kFlags |= BEAGLE_FLAG_EIGEN_REAL;
//    }

    if (requirementFlags & BEAGLE_FLAG_INVEVEC_TRANSPOSED || preferenceFlags & BEAGLE_FLAG_INVEVEC_TRANSPOSED)
        kFlags |= BEAGLE_FLAG_INVEVEC_TRANSPOSED;
    else
        kFlags |= BEAGLE_FLAG_INVEVEC_STANDARD;


    // TODO: this chunk of control is not checked
    if (kDeviceCode == BEAGLE_OPENCL_DEVICE_APPLE_CPU)
        kFlags |= BEAGLE_FLAG_PARALLELOPS_STREAMS;
    else if (requirementFlags & BEAGLE_FLAG_PARALLELOPS_STREAMS || preferenceFlags & BEAGLE_FLAG_PARALLELOPS_STREAMS)
        kFlags |= BEAGLE_FLAG_PARALLELOPS_STREAMS;
    else if (requirementFlags & BEAGLE_FLAG_PARALLELOPS_GRID || preferenceFlags & BEAGLE_FLAG_PARALLELOPS_GRID)
        kFlags |= BEAGLE_FLAG_PARALLELOPS_GRID;

    if (preferenceFlags & BEAGLE_FLAG_COMPUTATION_ASYNCH || requirementFlags & BEAGLE_FLAG_COMPUTATION_ASYNCH) {
        kFlags |= BEAGLE_FLAG_COMPUTATION_ASYNCH;
    } else {
        kFlags |= BEAGLE_FLAG_COMPUTATION_SYNCH;
    }

    if (preferenceFlags & BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO || requirementFlags & BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO) {
        kFlags |= BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO;
    } else {
        kFlags |= BEAGLE_FLAG_PREORDER_TRANSPOSE_MANUAL;
    }

    Real r = 0;
    modifyFlagsForPrecision(&kFlags, r);

    //TODO: sum block size ignored
    kPartialsSize = kPaddedPatternCount * kPaddedStateCount * kCategoryCount;
    kMatrixSize = kPaddedStateCount * kPaddedStateCount;

    kLastCompactBufferIndex = -1;
    kLastTipPartialsBufferIndex = -1;

    //TODO: we probably don't use the kernel resource anyway, separate it?
    gpu->SetDevice(pluginResourceNumber, kPaddedStateCount, kCategoryCount,
                   kPaddedPatternCount, kPatternCount, kTipCount, kFlags);

    int ptrQueueLength = kMatrixCount * kCategoryCount * 3 * 3; // first '3' for derivatives, last '3' is for 3 ops for uTMWMM
    if (kInternalPartialsBufferCount > ptrQueueLength)
        ptrQueueLength = kInternalPartialsBufferCount;

    // TODO: not sure if need kernels
//    kernels = new KernelLauncher(gpu);

    hWeightsCache = (Real*) gpu->CallocHost(kCategoryCount, sizeof(Real));
    hPatternWeightsCache = (Real*) gpu->CallocHost(kPatternCount, sizeof(Real));
    hFrequenciesCache = (Real*) gpu->CallocHost(kPaddedStateCount, sizeof(Real));
    hPartialsCache = (Real*) gpu->CallocHost(kPartialsSize, sizeof(Real));

    int hMatrixCacheSize = kMatrixSize * kCategoryCount * BEAGLE_CACHED_MATRICES_COUNT;  //TODO: use Eigen csr representation?
    hLogLikelihoodsCache = (Real*) gpu->MallocHost(kPatternCount * sizeof(Real));
    hMatrixCache = (Real*) gpu->CallocHost(hMatrixCacheSize, sizeof(Real));
    hInstantaneousMatrices.resize(kEigenDecompCount);
    for (int i = 0; i < kEigenDecompCount; i++) {
        hInstantaneousMatrices[i] = SpMatrix(kPaddedStateCount, kPaddedStateCount);
    }
    dInstantaneousMatrices = (cusparseSpMatDescr_t *) calloc(sizeof(cusparseSpMatDescr_t), kEigenDecompCount);
    for (int i = 0; i < kEigenDecompCount; i++) {
        dInstantaneousMatrices[i] = NULL;
    }
    CHECK_CUDA(cudaMalloc((void**) &dMatrixCsrOffsetsCache, (kPaddedStateCount + 1) * sizeof(int)))
    dMatrixCsrColumnsCache = NULL;
    currentCacheNNZ = 0;

    dPartials = (cusparseDnMatDescr_t *) calloc(sizeof(cusparseDnMatDescr_t), kPartialsBufferCount * kCategoryCount);
    for (int i = 0; i < kPartialsBufferCount * kCategoryCount; i++) {
        dPartials[i] = NULL;
    }

    dFrequenciesCache = (Real**) gpu->MallocHost(sizeof(Real*) * kEigenDecompCount);
    dWeightsCache = (Real**) gpu->MallocHost(sizeof(Real*) * kEigenDecompCount);;
    dWeights = (cusparseDnVecDescr_t *) calloc(sizeof(cusparseDnVecDescr_t), kEigenDecompCount);
    dFrequencies = (cusparseDnVecDescr_t *) calloc(sizeof(cusparseDnVecDescr_t), kEigenDecompCount);
    for (int i = 0; i < kEigenDecompCount; i++) {
        dFrequenciesCache[i] = NULL;
        dWeightsCache[i] = NULL;
        dWeights[i] = NULL;
        dFrequencies[i] = NULL;
    }

    hCategoryRates = (double**) calloc(sizeof(double*),kEigenDecompCount); // Keep in double-precision
    hCategoryRates[0] = (double*) gpu->MallocHost(sizeof(double) * kCategoryCount);
    checkHostMemory(hCategoryRates[0]);

//    dMatrices = (GPUPtr*) malloc(sizeof(GPUPtr) * kMatrixCount);
    CHECK_CUDA(cudaMalloc((void**) &dPatternWeightsCache, kPatternCount * sizeof(Real)))






    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::getInstanceDetails(BeagleInstanceDetails* returnInfo) {
    BeagleGPUImpl<BEAGLE_GPU_GENERIC>::getInstanceDetails(returnInfo);
    returnInfo->implName = getInstanceName();
    return BEAGLE_SUCCESS;
}

#ifdef CUDA
template<>
char* BeagleGPUActionImpl<double>::getInstanceName() {
    return (char*) "Action-CUDA-Double";
}

template<>
char* BeagleGPUActionImpl<float>::getInstanceName() {
    return (char*) "Action-CUDA-Single";
}
#elif defined(FW_OPENCL)
            template<>
char* BeagleGPUActionImpl<double>::getInstanceName() {
    return (char*) "Action-OpenCL-Double";
}

template<>
char* BeagleGPUActionImpl<float>::getInstanceName() {
    return (char*) "Action-OpenCL-Single";
}
#endif

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::setPatternWeights(const double* inPatternWeights) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUActionImpl::setPatternWeights\n");
#endif

//#ifdef DOUBLE_PRECISION
//  const double* tmpWeights = inPatternWeights;
//#else
//  Real* tmpWeights = hPatternWeightsCache;
//  MEMCNV(hPatternWeightsCache, inPatternWeights, kPatternCount, Real);
//#endif
//    const Real* tmpWeights = beagleCastIfNecessary(inPatternWeights, hPatternWeightsCache, kPatternCount);
    CHECK_CUDA(cudaMemcpy(dPatternWeightsCache, inPatternWeights, kPatternCount * sizeof(Real), cudaMemcpyHostToDevice))

    CHECK_CUSPARSE(cusparseCreateDnVec(dPatternWeights, kPatternCount, dPatternWeightsCache, CUDA_R_64F))

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUActionImpl::setPatternWeights\n");
#endif

    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::setStateFrequencies(int stateFrequenciesIndex,
                                                           const double* inStateFrequencies) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUActionImpl::setStateFrequencies\n");
#endif

    if (stateFrequenciesIndex < 0 || stateFrequenciesIndex >= kEigenDecompCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;

//#ifdef DOUBLE_PRECISION
//    memcpy(hFrequenciesCache, inStateFrequencies, kStateCount * sizeof(Real));
//#else
//    MEMCNV(hFrequenciesCache, inStateFrequencies, kStateCount, Real);
//#endif
    beagleMemCpy(hFrequenciesCache, inStateFrequencies, kStateCount);

    if (dFrequenciesCache[stateFrequenciesIndex] == NULL) {
        CHECK_CUDA(cudaMalloc((void**) &dFrequenciesCache[stateFrequenciesIndex], sizeof(Real) * kPaddedStateCount))
    }
    CHECK_CUDA(cudaMemcpy(dFrequenciesCache[stateFrequenciesIndex], hFrequenciesCache, kStateCount, cudaMemcpyHostToDevice))
    if (dFrequencies[stateFrequenciesIndex] == NULL) {
        CHECK_CUSPARSE(cusparseCreateDnVec(&dFrequencies[stateFrequenciesIndex], kPaddedStateCount, dFrequenciesCache[stateFrequenciesIndex], CUDA_R_64F))
    } else {
        CHECK_CUSPARSE(cusparseDnVecSetValues(dFrequencies[stateFrequenciesIndex], dFrequenciesCache[stateFrequenciesIndex]))
    }

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUActionImpl::setStateFrequencies\n");
#endif

    return BEAGLE_SUCCESS;
}


BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::setTipStates(int tipIndex, const int* inStates)
{
    std::cerr<<"\nBEAGLE: When using action-based likelihood computations, setTipStates( ) is not allowed.\n";
    std::cerr<<"        Use setTipPartials( ) instead.\n\n";

    // There does not appear to be a simple method of throwing C++ exceptions into Java through the JNI.
    // However, throwing this exception makes Java print a stack trace that shows where the setTipStates( )
    //   call is coming from.
    throw std::runtime_error("This message will not be seen");

    std::abort();
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::setTipPartials(int tipIndex, const double* inPartials) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUActionImpl::setTipPartials\n");
#endif

    if (tipIndex < 0 || tipIndex >= kTipCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;

    const double* inPartialsOffset = inPartials;
    Real* tmpRealPartialsOffset = hPartialsCache;
    for (int i = 0; i < kPatternCount; i++) {
//#ifdef DOUBLE_PRECISION
//        memcpy(tmpRealPartialsOffset, inPartialsOffset, sizeof(Real) * kStateCount);
//#else
//        MEMCNV(tmpRealPartialsOffset, inPartialsOffset, kStateCount, Real);
//#endif
        beagleMemCpy(tmpRealPartialsOffset, inPartialsOffset, kStateCount);
        tmpRealPartialsOffset += kPaddedStateCount;
        inPartialsOffset += kStateCount;
    }

//    int partialsLength = kPaddedPatternCount * kPaddedStateCount;
//    for (int i = 1; i < kCategoryCount; i++) {
//        memcpy(hPartialsCache + i * partialsLength, hPartialsCache, partialsLength * sizeof(Real));
//    }

    if (tipIndex < kTipCount) {
        if (dPartials[tipIndex] == NULL) {
            for (int i = 0; i < kCategoryCount; i++) {
                CHECK_CUSPARSE(cusparseCreateDnMat(&dPartials[kPartialsBufferCount * i + tipIndex], kPaddedStateCount, kPaddedPatternCount, kPaddedStateCount, hPartialsCache,
                                                   CUDA_R_64F, CUSPARSE_ORDER_COL))
            }
        }
    }
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUActionImpl::setTipPartials\n");
#endif

    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::upPartials(bool byPartition,
							const int *operations,
							int operationCount,
							int cumulativeScalingIndex)
{
    return BeagleGPUImpl<BEAGLE_GPU_GENERIC>::upPartials(byPartition, operations, operationCount, cumulativeScalingIndex);
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::upPrePartials(bool byPartition,
							   const int *operations,
							   int operationCount,
							   int cumulativeScalingIndex)
{
    return BeagleGPUImpl<BEAGLE_GPU_GENERIC>::upPrePartials(byPartition, operations, operationCount, cumulativeScalingIndex);
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::setSparseMatrix(int matrixIndex,
                                                             const int* rowIndices,
                                                             const int* colIndices,
                                                             const double* values,
                                                             int numNonZeros)
{
    std::vector<Triplet> tripletList;
    for (int i = 0; i < numNonZeros; i++) {
        tripletList.push_back(Triplet(rowIndices[i], colIndices[i], values[i]));
    }
    hInstantaneousMatrices[matrixIndex].setFromTriplets(tripletList.begin(), tripletList.end());

    const int currentNNZ = hInstantaneousMatrices[matrixIndex].nonZeros();
    const int paddedNNZ = currentNNZ + 16 - currentNNZ%16;
    if (currentCacheNNZ < paddedNNZ) {
        currentCacheNNZ = paddedNNZ;
        CHECK_CUDA(cudaMalloc((void**) &dMatrixCsrColumnsCache, currentCacheNNZ * sizeof(int)))
        CHECK_CUDA(cudaMalloc((void**) &dMatrixCsrValuesCache, currentCacheNNZ * sizeof(Real)))
    }

    CHECK_CUDA(cudaMemcpy(dMatrixCsrOffsetsCache, hInstantaneousMatrices[matrixIndex].outerIndexPtr(), sizeof(int) * (kPaddedStateCount + 1), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dMatrixCsrColumnsCache, hInstantaneousMatrices[matrixIndex].innerIndexPtr(), sizeof(int) * currentNNZ, cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dMatrixCsrValuesCache, hInstantaneousMatrices[matrixIndex].valuePtr(), sizeof(Real) * currentNNZ, cudaMemcpyHostToDevice))

    CHECK_CUSPARSE(cusparseCreateCsr(&dInstantaneousMatrices[matrixIndex], kPaddedStateCount, kPaddedStateCount, currentNNZ,
                                     dMatrixCsrOffsetsCache, dMatrixCsrColumnsCache, dMatrixCsrValuesCache,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F))


//#ifdef BEAGLE_DEBUG_FLOW
//    std::cerr<<"Setting host matrix: "<<matrixIndex<<std::endl<<hInstantaneousMatrices[matrixIndex]<<std::endl
//    <<std::endl<<"Setting device matrix: " << matrixIndex << std::endl << dInstantaneousMatrices[matrixIndex]<<std::endl;
//#endif
    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::updateTransitionMatrices(int eigenIndex,
								      const int* probabilityIndices,
								      const int* firstDerivativeIndices,
								      const int* secondDerivativeIndices,
								      const double* edgeLengths,
								      int count)
{
    return BEAGLE_ERROR_NO_IMPLEMENTATION;
}

///-------------------------------- Factory -------------------------------------///	
BEAGLE_GPU_TEMPLATE
BeagleImpl*  BeagleGPUActionImplFactory<BEAGLE_GPU_GENERIC>::createImpl(int tipCount,
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
                                              int* errorCode) {
    BeagleImpl * impl = new BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>();
    try {
        *errorCode =
            impl->createInstance(tipCount, partialsBufferCount, compactBufferCount, stateCount,
                                 patternCount, eigenBufferCount, matrixBufferCount,
                                 categoryCount,scaleBufferCount, resourceNumber, pluginResourceNumber, preferenceFlags, requirementFlags);
        if (*errorCode == BEAGLE_SUCCESS) {
            return impl;
        }
        delete impl;
        return NULL;
    }
    catch(...)
    {
        delete impl;
        *errorCode = BEAGLE_ERROR_GENERAL;
        throw;
    }
    delete impl;
    *errorCode = BEAGLE_ERROR_GENERAL;
    return NULL;    
}

#ifdef CUDA
template<>
const char* BeagleGPUActionImplFactory<double>::getName() {
    return "GPU-DP-CUDA-Action";
}

template<>
const char* BeagleGPUActionImplFactory<float>::getName() {
    return "GPU-SP-CUDA-Action";
}
#elif defined(FW_OPENCL)
template<>
const char* BeagleGPUActionImplFactory<double>::getName() {
    return "DP-OpenCL-Action";

}
template<>
const char* BeagleGPUActionImplFactory<float>::getName() {
    return "SP-OpenCL-Action";
}
#endif

BEAGLE_GPU_TEMPLATE
long long BeagleGPUActionImplFactory<BEAGLE_GPU_GENERIC>::getFlags() {
    return BeagleGPUImplFactory<BEAGLE_GPU_GENERIC>::getFlags() | BEAGLE_FLAG_COMPUTATION_ACTION;
}

} // end of device namespace
} // end of gpu namespace
} // end of beagle namespace


#endif
