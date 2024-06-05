
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

#include <Eigen/Eigenvalues>

// Duplicated from CPU code
std::independent_bits_engine<std::mt19937_64,1,unsigned short> engine;

bool random_bool()
{
    return engine();
}
double random_plus_minus_1_func(double x)
{
    if (random_bool())
        return 1;
    else
        return -1;
}

// Algorithm 2.4 from Higham and Tisseur (2000), A BLOCK ALGORITHM FOR MATRIX 1-NORM ESTIMATION,
//    WITH AN APPLICATION TO 1-NORM PSEUDOSPECTRA.
// See OneNormEst in https://eprints.maths.manchester.ac.uk/2195/1/thesis-main.pdf
//    This seems to have a bug where it checks if columns in S are parallel to EVERY column of S_old.
// See also https://github.com/gnu-octave/octave/blob/default/scripts/linear-algebra/normest1.m
// See dlacn1.f
template <typename Real>
Real normest1(const SpMatrix<Real>& A, int p, int t=2, int itmax=5)
{
    assert(p >= 0);
    assert(t != 0); // negative means t = n
    assert(itmax >= 1);

    if (p == 0) return 1.0;

    // A is (n,n);
    assert(A.rows() == A.cols());
    int n = A.cols();

    // Handle t too large
    t = std::min(n,t);

    // Interpret negative t as t == n
    if (t < 0) t = n;

    // Defer to normP1 if p=1 and n is small or we want an exact answer.
    if (p == 1 and (n <= 4 or t == n))
        return normP1(A);

    // (0) Choose starting matrix X that is (n,t) with columns of unit 1-norm.
    DnMatrix<Real> X(n,t);
    // We choose the first column to be all 1s.
    X.col(0).setOnes();
    // The other columns have randomly chosen {-1,+1} entries.
    X = X.unaryExpr( &random_plus_minus_1_func );
    // Divide by n so that the norm of each column is 1.
    X /= n;

    // 3.
    std::vector<bool> ind_hist(n,0);
    std::vector<int> indices(n,0);
    int ind_best = -1;
    Real est_old = 0;
    DnMatrix<Real> S = DnMatrix<Real>::Ones(n,t);
    DnMatrix<Real> S_old = DnMatrix<Real>::Ones(n,t);
    MatrixXi prodS(t,t);
    DnMatrix<Real> Y(n,t);
    DnMatrix<Real> Z(n,t);
    DnVector<Real> h(n);

    for(int k=1; k<=itmax; k++)
    {
        // std::cerr<<"iter "<<k<<"\n";
        Y = A*X; // Y is (n,n) * (n,t) = (n,t)
        for(int i=1;i<p;i++)
            Y = A*Y;

        auto [est, j] = ArgNormP1(Y);

        if (est > est_old or k == 2)
        {
            // Note that j is in [0,t-1], but indices[j] is in [0,n-1].
            ind_best = indices[j];
            // w = Y.col(ind_best);
        }
        // std::cerr<<"  est = "<<est<<"  (est_old = "<<est_old<<")\n";
        assert(ind_best < n);

        // (1) of Algorithm 2.4
        if (est < est_old and k >= 2)
        {
            // std::cerr<<"  The new estimate ("<<est<<") is smaller than the old estimate ("<<est_old<<")\n";
            return est_old;
        }

        est_old = est;

        assert(est >= est_old);

        // S = sign(Y), 0.0 -> 1.0
        S = Y.unaryExpr([](const Real& x) {return (x>=0) ? 1.0 : -1.0 ;});

        // prodS is (t,t)
        prodS = (S_old.transpose() * S).matrix().cwiseAbs().template cast<int>() ;

        // (2) If each columns in S is parallel to SOME column of S_old
        if (prodS.colwise().maxCoeff().sum() == n * t and k >= 2)
        {
            // std::cerr<<"  All columns of S parallel to S_old\n";
            return est;
        }

        if (t > 1)
        {
            // If S(j) is parallel to S_old(i), replace S(j) with random column
            for(int j=0;j<S.cols();j++)
            {
                for(int i=0;i<S_old.cols();i++)
                    if (prodS(i,j) == n)
                    {
                        // std::cerr<<"  S.col("<<j<<") parallel to S_old.col("<<i<<")\n";
                        S.col(j) = S.col(j).unaryExpr( &random_plus_minus_1_func );
                        break;
                    }
            }

            // If S(j) is parallel to S(i) for i<j, replace S(j) with random column
            prodS = (S.transpose() * S).matrix().template cast<int>() ;
            for(int i=0;i<S.cols();i++)
                for(int j=i+1;j<S.cols();j++)
                    if (prodS(i,j) == n)
                    {
                        // std::cerr<<"  S.col("<<j<<") parallel to S.col("<<i<<")\n";
                        S.col(j) = S.col(j).unaryExpr( &random_plus_minus_1_func );
                        break;
                    }
        }

        // (3) of Algorithm 2.4
        Z = A.transpose() * S; // (t,n) * (n,t) -> (t,t)

        h = Z.cwiseAbs().rowwise().maxCoeff();

        // (4) of Algorithm 2.4
        if (k >= 2 and h.maxCoeff() == h[ind_best])
        {
            // std::cerr<<"  The best column ("<<ind_best<<") is not new\n";

            // According to Algorithm 2.4, we should exit here.

            // However, continuing until we find a different reason to exit
            // seems to providegreater accuracy.

            // return est;
        }

        indices.resize(n);
        for(int i=0;i<n;i++)
            indices[i] = i;

        // reorder idx so that the highest values of h[indices[i]] come first.
        std::sort(indices.begin(), indices.end(), [&](int i,int j) {return h[i] > h[j];});

        // (5) of Algorithm 2.4
        int n_found = 0;
        for(int i=0;i<t;i++)
            if (ind_hist[indices[i]])
                n_found++;

        if (n_found == t)
        {
            assert(k >= 2);
            // std::cerr<<"  All columns were found in the column history.\n";
            return est;
        }

        // find the first t indices that are not in ind_hist
        int l=0;
        for(int i=0;i<indices.size() and l < t;i++)
        {
            if (not ind_hist[indices[i]])
            {
                indices[l] = indices[i];
                l++;
            }
        }
        indices.resize( std::min(l,t) );
        assert(not indices.empty());

        int tmax = std::min<int>(t, indices.size());

        X = DnMatrix<Real>::Zero(n, tmax);
        for(int j=0; j < tmax; j++)
            X(indices[j], j) = 1; // X(:,j) = e(indices[j])

        for(int i: indices)
            ind_hist[i] = true;

        S_old = S;
    }

    return est_old;
}


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

template <typename T> cusparseIndexType_t IndexType;
template <> cusparseIndexType_t IndexType<int32_t> = CUSPARSE_INDEX_32I;
template <> cusparseIndexType_t IndexType<int64_t> = CUSPARSE_INDEX_64I;

template <typename T> cudaDataType DataType;
template <> cudaDataType DataType<float> = CUDA_R_32F;
template <> cudaDataType DataType<double> = CUDA_R_64F;

template <typename T>
T* cudaDeviceNew(int n)
{
    T* result;
    auto status = cudaMalloc((void**)&result, n*sizeof(T));
    if (status != cudaSuccess)
	throw std::runtime_error("cudaMalloc: failed!");
    return result;
}

template <typename T>
void MemcpyHostToDevice(T* dptr, const T* hptr, int n)
{
    auto status = cudaMemcpy(dptr, hptr, n*sizeof(T), cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
	throw std::runtime_error("cudaMemcpy(Host->Device): failed!");
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

    int status = BeagleGPUImpl<Real>::createInstance(tipCount,
						     partialsBufferCount,
						     compactBufferCount,
						     stateCount,
						     patternCount,
						     eigenDecompositionCount,
						     matrixCount,
						     categoryCount,
						     scaleBufferCount,
						     globalResourceNumber,
						     pluginResourceNumber,
						     preferenceFlags,
						     requirementFlags);

    if (status != BEAGLE_SUCCESS) return status;
/*
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
    hIdentity = SpMatrix<Real>(kPaddedStateCount, kPaddedStateCount);
    hIdentity.setIdentity();
*/
    hInstantaneousMatrices.resize(kEigenDecompCount);
    for (int i = 0; i < kEigenDecompCount; i++) {
        hInstantaneousMatrices[i] = SpMatrix<Real>(kPaddedStateCount, kPaddedStateCount);
    }
/*
    hBs.resize(kEigenDecompCount);
    hMuBs.resize(kEigenDecompCount);
    hB1Norms.resize(kEigenDecompCount);
    ds.resize(kEigenDecompCount);
    dInstantaneousMatrices = std::vector<cusparseSpMatDescr_t>(kEigenDecompCount, nullptr);

    dMatrixCsrOffsetsCache = cudaDeviceNew<int>(kPaddedStateCount + 1);
    dMatrixCsrColumnsCache = NULL;
    currentCacheNNZ = 0;

    dPartials = std::vector<cusparseDnMatDescr_t>(kPartialsBufferCount * kCategoryCount, nullptr);
    dPartialCache = std::vector<Real*>(kPartialsBufferCount * kCategoryCount, nullptr);

    dFrequenciesCache = (Real**) gpu->MallocHost(sizeof(Real*) * kEigenDecompCount);
    dWeightsCache = (Real**) gpu->MallocHost(sizeof(Real*) * kEigenDecompCount);;
    dWeights = std::vector<cusparseDnVecDescr_t>(kEigenDecompCount, nullptr);
    dFrequencies = std::vector<cusparseDnVecDescr_t>(kEigenDecompCount, nullptr);
    for (int i = 0; i < kEigenDecompCount; i++) {
        dFrequenciesCache[i] = NULL;
        dWeightsCache[i] = NULL;
    }

    hEigenMaps.resize(kPartialsBufferCount);
    hEdgeMultipliers.resize(kPartialsBufferCount * kCategoryCount);

    hCategoryRates = (double**) calloc(sizeof(double*),kEigenDecompCount); // Keep in double-precision
    hCategoryRates[0] = (double*) gpu->MallocHost(sizeof(double) * kCategoryCount);
    checkHostMemory(hCategoryRates[0]);

//    dMatrices = (GPUPtr*) malloc(sizeof(GPUPtr) * kMatrixCount);
    dPatternWeightsCache = cudaDeviceNew<Real>(kPatternCount);
    dPatternWeights = NULL;
*/

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
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::setPatternWeights(const Real* inPatternWeights) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUActionImpl::setPatternWeights\n");
#endif

    return BeagleGPUImpl<Real>::setPatternWeights(inPatternWeights);

/*
//#ifdef DOUBLE_PRECISION
//  const double* tmpWeights = inPatternWeights;
//#else
//  Real* tmpWeights = hPatternWeightsCache;
//  MEMCNV(hPatternWeightsCache, inPatternWeights, kPatternCount, Real);
//#endif
//    const Real* tmpWeights = beagleCastIfNecessary(inPatternWeights, hPatternWeightsCache, kPatternCount);
    MemcpyHostToDevice(dPatternWeightsCache, inPatternWeights, kPatternCount);

    if (dPatternWeights == NULL) {
        CHECK_CUSPARSE(cusparseCreateDnVec(&dPatternWeights, kPatternCount, dPatternWeightsCache, DataType<Real>))
    } else {
        CHECK_CUSPARSE(cusparseDnVecSetValues(dPatternWeights, dPatternWeightsCache))
    }

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUActionImpl::setPatternWeights\n");
#endif

    return BEAGLE_SUCCESS;
*/
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::setStateFrequencies(int stateFrequenciesIndex,
								 const Real* inStateFrequencies) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUActionImpl::setStateFrequencies\n");
#endif

    return BeagleGPUImpl<Real>::setStateFrequencies(stateFrequenciesIndex, inStateFrequencies);

/*
    if (stateFrequenciesIndex < 0 || stateFrequenciesIndex >= kEigenDecompCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;

    beagleMemCpy(hFrequenciesCache, inStateFrequencies, kStateCount);

    if (dFrequenciesCache[stateFrequenciesIndex] == NULL) {
        dFrequenciesCache[stateFrequenciesIndex] = cudaDeviceNew<Real>(kPaddedStateCount);
    }
    MemcpyHostToDevice(dFrequenciesCache[stateFrequenciesIndex], hFrequenciesCache, kStateCount);
    if (dFrequencies[stateFrequenciesIndex] == NULL) {
        CHECK_CUSPARSE(cusparseCreateDnVec(&dFrequencies[stateFrequenciesIndex], kPaddedStateCount, dFrequenciesCache[stateFrequenciesIndex], DataType<Real>))
    } else {
        CHECK_CUSPARSE(cusparseDnVecSetValues(dFrequencies[stateFrequenciesIndex], dFrequenciesCache[stateFrequenciesIndex]))
    }

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUActionImpl::setStateFrequencies\n");
#endif

    return BEAGLE_SUCCESS;
*/
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::setCategoryWeights(int categoryWeightsIndex,
                                                          const Real* inCategoryWeights) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUActionImpl::setCategoryWeights\n");
#endif

    return BeagleGPUImpl<Real>::setCategoryWeights(categoryWeightsIndex, inCategoryWeights);

/*
    if (categoryWeightsIndex < 0 || categoryWeightsIndex >= kEigenDecompCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;

    beagleMemCpy(hWeightsCache, inCategoryWeights, kCategoryCount);

    if (dWeightsCache[categoryWeightsIndex] == NULL) {
        dWeightsCache[categoryWeightsIndex] = cudaDeviceNew<Real>(kCategoryCount);
    }
    MemcpyHostToDevice(dWeightsCache[categoryWeightsIndex], hWeightsCache, kCategoryCount);

    if (dWeights[categoryWeightsIndex] == NULL) {
        CHECK_CUSPARSE(cusparseCreateDnVec(&dWeights[categoryWeightsIndex], kCategoryCount, dWeightsCache[categoryWeightsIndex], DataType<Real>))
    } else {
        CHECK_CUSPARSE(cusparseDnVecSetValues(dWeights[categoryWeightsIndex], dWeightsCache[categoryWeightsIndex]))
    }

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUActionImpl::setCategoryWeights\n");
#endif

    return BEAGLE_SUCCESS;
*/
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
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::setTipPartials(int tipIndex, const Real* inPartials) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUActionImpl::setTipPartials\n");
#endif

    return BeagleGPUImpl<Real>::setTipPartials(tipIndex, inPartials);

/*
    if (tipIndex < 0 || tipIndex >= kTipCount)
        return BEAGLE_ERROR_OUT_OF_RANGE;

    const Real* inPartialsOffset = inPartials;
    Real* tmpRealPartialsOffset = hPartialsCache;
    for (int i = 0; i < kPatternCount; i++) {
        beagleMemCpy(tmpRealPartialsOffset, inPartialsOffset, kStateCount);
        tmpRealPartialsOffset += kPaddedStateCount;
        inPartialsOffset += kStateCount;
    }
    for (int i = 0; i < kCategoryCount; i++) {
        if (dPartialCache[getPartialIndex(tipIndex, i)] == NULL) {
            dPartialCache[getPartialIndex(tipIndex, i)] = cudaDeviceNew<Real>(kPaddedStateCount * kPaddedPatternCount);
        }
        MemcpyHostToDevice(dPartialCache[getPartialIndex(tipIndex, i)], hPartialsCache, kPaddedStateCount * kPaddedPatternCount);
    }



//    int partialsLength = kPaddedPatternCount * kPaddedStateCount;
//    for (int i = 1; i < kCategoryCount; i++) {
//        memcpy(hPartialsCache + i * partialsLength, hPartialsCache, partialsLength * sizeof(Real));
//    }

    if (tipIndex < kTipCount) {
        for (int i = 0; i < kCategoryCount; i++) {
            const int partialIndex = getPartialIndex(tipIndex, i);
            if (dPartials[partialIndex] == NULL) {
                CHECK_CUSPARSE(cusparseCreateDnMat(&dPartials[partialIndex], kPaddedStateCount, kPaddedPatternCount, kPaddedStateCount, dPartialCache[partialIndex],
                                                   DataType<Real>, CUSPARSE_ORDER_COL))
            } else {
                CHECK_CUSPARSE(cusparseDnMatSetValues(dPartials[partialIndex], dPartialCache[partialIndex]))
            }
        }
    }
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUActionImpl::setTipPartials\n");
#endif

    return BEAGLE_SUCCESS;
*/
}


BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::getPartials(int bufferIndex,
                                                   int scaleIndex,
                                                   Real* outPartials) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUActionImpl::getPartials\n");
#endif

    int status = BeagleGPUImpl<Real>::getPartials(bufferIndex, scaleIndex, outPartials);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUActionImpl::getPartials\n");
#endif

    return status;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::getPartialIndex(int nodeIndex, int categoryIndex) {
    return kPartialsBufferCount * categoryIndex + nodeIndex;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::upPartials(bool byPartition,
							const int *operations,
							int operationCount,
							int cumulativeScalingIndex)
{
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUImpl::upPartials\n");
#endif

    if (byPartition)
    {
	std::cerr<<"GPU-Action: upPartials( ): byPartition is true, but must be false!";
	throw std::runtime_error("This will not be seen by java");
    }

    GPUPtr cumulativeScalingBuffer = 0;
    if (cumulativeScalingIndex != BEAGLE_OP_NONE)
        cumulativeScalingBuffer = dScalingFactors[cumulativeScalingIndex];

    int numOps = BEAGLE_OP_COUNT;

    int lastStreamIndex = 0;

    int anyRescale = BEAGLE_OP_NONE;

    int streamIndex = -1;
    int waitIndex = -1;
    if (anyRescale == 1 && kPartitionsInitialised) {
        gpu->SynchronizeDevice();
        for (int i = 0; i < kBufferCount * kPartitionCount; i++) {
            hStreamIndices[i] = -1;
        }
    }

    for (int op = 0; op < operationCount; op++) {
        const int parIndex = operations[op * numOps];
        const int writeScalingIndex = operations[op * numOps + 1];
        const int readScalingIndex = operations[op * numOps + 2];
        const int child1Index = operations[op * numOps + 3];
        const int child1TransMatIndex = operations[op * numOps + 4];
        const int child2Index = operations[op * numOps + 5];
        const int child2TransMatIndex = operations[op * numOps + 6];
        int currentPartition = 0;

        if (anyRescale == 1 && kPartitionsInitialised) {
            int pOffset = currentPartition * kBufferCount;
            waitIndex = hStreamIndices[child2Index + pOffset];
            if (hStreamIndices[child1Index + pOffset] != -1) {
                hStreamIndices[parIndex + pOffset] = hStreamIndices[child1Index + pOffset];
            } else if (hStreamIndices[child2Index + pOffset] != -1) {
                hStreamIndices[parIndex + pOffset] = hStreamIndices[child2Index + pOffset];
                waitIndex = hStreamIndices[child1Index + pOffset];
            } else {
                hStreamIndices[parIndex + pOffset] = lastStreamIndex++;
            }
            streamIndex = hStreamIndices[parIndex + pOffset];
        }
	
        GPUPtr matrices1 = dMatrices[child1TransMatIndex];
        GPUPtr matrices2 = dMatrices[child2TransMatIndex];

        GPUPtr partials1 = dPartials[child1Index];
        GPUPtr partials2 = dPartials[child2Index];

        GPUPtr partials3 = dPartials[parIndex];

        GPUPtr tipStates1 = dStates[child1Index];
        GPUPtr tipStates2 = dStates[child2Index];

        int rescale = BEAGLE_OP_NONE;
        GPUPtr scalingFactors = (GPUPtr)NULL;

        if (kFlags & BEAGLE_FLAG_SCALING_AUTO) {
            int sIndex = parIndex - kTipCount;

            if (tipStates1 == 0 && tipStates2 == 0) {
                rescale = 2;
                scalingFactors = dScalingFactors[sIndex];
            }
        } else if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS) {
            rescale = 1;
            scalingFactors = dScalingFactors[parIndex - kTipCount];
        } else if ((kFlags & BEAGLE_FLAG_SCALING_MANUAL) && writeScalingIndex >= 0) {
            rescale = 1;
            scalingFactors = dScalingFactors[writeScalingIndex];
        } else if ((kFlags & BEAGLE_FLAG_SCALING_MANUAL) && readScalingIndex >= 0) {
            rescale = 0;
            scalingFactors = dScalingFactors[readScalingIndex];
        }

        int startPattern = 0;
        int endPattern = 0;


	if (kFlags & BEAGLE_FLAG_SCALING_DYNAMIC)
	{
	    std::cerr<<"GPU-Action: upPartials( ): boldly ignoring BEAGLE_FLAG_SCALING_DYNAMIC!";
	}

	kernels->PartialsPartialsPruningDynamicScaling(partials1, partials2, partials3,
						       matrices1, matrices2, scalingFactors,
						       cumulativeScalingBuffer,
						       startPattern, endPattern,
						       kPaddedPatternCount, kCategoryCount,
						       rescale,
						       streamIndex, waitIndex);

/*
        We are doing this, if rescale == 1:

            gpu->LaunchKernelConcurrent(fPartialsPartialsByPatternBlockCoherent,
                                        bgPeelingBlock, bgPeelingGrid,
                                        streamIndex, waitIndex,
                                        5, 6,
                                        partials1, partials2, partials3, matrices1, matrices2,
                                        kPaddedPatternCount);

	    kernels->RescalePartials(partials3, scalingFactors, cumulativeScalingBuffer,
				     kPaddedPatternCount, kCategoryCount, 0, streamIndex, -1);
*/

        if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS) {
            int parScalingIndex = parIndex - kTipCount;
            int child1ScalingIndex = child1Index - kTipCount;
            int child2ScalingIndex = child2Index - kTipCount;
            if (child1ScalingIndex >= 0 && child2ScalingIndex >= 0) {
                int scalingIndices[2] = {child1ScalingIndex, child2ScalingIndex};
                BeagleGPUImpl<Real>::accumulateScaleFactors(scalingIndices, 2, parScalingIndex);
            } else if (child1ScalingIndex >= 0) {
                int scalingIndices[1] = {child1ScalingIndex};
                BeagleGPUImpl<Real>::accumulateScaleFactors(scalingIndices, 1, parScalingIndex);
            } else if (child2ScalingIndex >= 0) {
                int scalingIndices[1] = {child2ScalingIndex};
                BeagleGPUImpl<Real>::accumulateScaleFactors(scalingIndices, 1, parScalingIndex);
            }
        }
    } //end for loop over operationCount


    if (anyRescale == 1 && kPartitionsInitialised) {
        gpu->SynchronizeDevice();
    }

#ifdef BEAGLE_DEBUG_SYNCH
    gpu->SynchronizeHost();
#endif

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUImpl::upPartials\n");
#endif

    return BEAGLE_SUCCESS;
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
                                                             const Real* values,
                                                             int numNonZeros)
{
#ifdef BEAGLE_DEBUG_FLOW
    std::cerr<<"\tEntering BeagleGPUActionImpl::setSparseMatrix\n";
#endif

    std::vector<Triplet<Real>> tripletList;
    for (int i = 0; i < numNonZeros; i++) {
        tripletList.push_back(Triplet<Real>(rowIndices[i], colIndices[i], values[i]));
    }
    hInstantaneousMatrices[matrixIndex].setFromTriplets(tripletList.begin(), tripletList.end());

    DnMatrix<Real> tmp = hInstantaneousMatrices[matrixIndex];
    Eigen::EigenSolver<DnMatrix<Real>> es(tmp);

    DnMatrix<Real> V = es.pseudoEigenvectors();
    DnMatrix<Real> inverseV = es.pseudoEigenvectors().inverse();

    std::vector<Real> eigenVectors(kStateCount*kStateCount, 0);
    std::vector<Real> inverseEigenVectors(kStateCount*kStateCount, 0);
    std::vector<Real> eigenValues(2*kStateCount, 0);

#ifdef EIGEN_DEBUG_FLOW
    std::cerr<< "The eigenvalues of Matrix are:" << std::endl << es.pseudoEigenvalueMatrix().diagonal() << std::endl;
    std::cerr<< "The matrix of eigenvectors, V, is:" << std::endl << es.pseudoEigenvectors() << std::endl << std::endl;
    std::cerr<< "The matrix of inverse eigenvectors, invV, is:" << std::endl << inverseV << std::endl << std::endl;
#endif
    bool complexPair = false;
    for (int i = 0; i < kStateCount; i++) {
	eigenValues[i] = es.pseudoEigenvalueMatrix().col(i)[i];
	if (complexPair) {
	    eigenValues[kStateCount + i] = es.pseudoEigenvalueMatrix().col(i - 1)[i];
	    complexPair = false;
	} else {
	    double nextValue = i < kStateCount - 1 ? es.pseudoEigenvalueMatrix().col(i + 1)[i] : 0;
	    if (nextValue != 0) {
		complexPair = true;
		eigenValues[kStateCount + i] = nextValue;
	    } else {
		eigenValues[kStateCount + i] = 0;
	    }
	}
	for (int j = 0; j < kStateCount; j++) {
	    eigenVectors[kStateCount * i + j] = V.col(j)[i];
	    inverseEigenVectors[kStateCount * i + j] = inverseV.col(j)[i];
	}
#ifdef EIGEN_DEBUG_FLOW
	std::cerr<< "The " << i <<"th eigenvalue :" << std::endl << eigenValues[i] << "," << eigenValues[i + kStateCount] << std::endl;
#endif

    }

    BeagleGPUImpl<Real>::setEigenDecomposition(matrixIndex, eigenVectors.data(), inverseEigenVectors.data(), eigenValues.data());

/*
    const int currentNNZ = hInstantaneousMatrices[matrixIndex].nonZeros();
    const int paddedNNZ = currentNNZ + 16 - currentNNZ%16;
    if (currentCacheNNZ < paddedNNZ) {
        currentCacheNNZ = paddedNNZ;
        dMatrixCsrColumnsCache = cudaDeviceNew<int>(currentCacheNNZ);
        dMatrixCsrValuesCache = cudaDeviceNew<Real>(currentCacheNNZ);
    }

    MemcpyHostToDevice(dMatrixCsrOffsetsCache, hInstantaneousMatrices[matrixIndex].outerIndexPtr(), kPaddedStateCount + 1);
    MemcpyHostToDevice(dMatrixCsrColumnsCache, hInstantaneousMatrices[matrixIndex].innerIndexPtr(), currentNNZ);
    MemcpyHostToDevice(dMatrixCsrValuesCache, hInstantaneousMatrices[matrixIndex].valuePtr(), currentNNZ);

    CHECK_CUSPARSE(cusparseCreateCsr(&dInstantaneousMatrices[matrixIndex], kPaddedStateCount, kPaddedStateCount, currentNNZ,
                                     dMatrixCsrOffsetsCache, dMatrixCsrColumnsCache, dMatrixCsrValuesCache,
                                     IndexType<int>, IndexType<int>,
                                     CUSPARSE_INDEX_BASE_ZERO, DataType<Real>))
    //TODO: use cusparse function for diagonal sum?
    Real mu_B = 0.0;
    for (int i = 0; i < kStateCount; i++) {
        mu_B += hInstantaneousMatrices[matrixIndex].coeff(i, i);
    }
    mu_B /= (Real) kStateCount;

    hMuBs[matrixIndex] = mu_B;
    hBs[matrixIndex] = hInstantaneousMatrices[matrixIndex] - mu_B * hIdentity;
    hB1Norms[matrixIndex] = normP1(hBs[matrixIndex]);

    ds[matrixIndex].clear();

    int pMax = getPMax();
    for(int p=0;p <= pMax+1; p++)
    {
        int t = 5;
        Real approx_norm = normest1( hBs[matrixIndex], p, t);

        // equation 3.7 in Al-Mohy and Higham
        ds[matrixIndex].push_back( pow( approx_norm, 1.0/Real(p) ) );
    }

//#ifdef BEAGLE_DEBUG_FLOW
//    std::cerr<<"Setting host matrix: "<<matrixIndex<<std::endl<<hInstantaneousMatrices[matrixIndex]<<std::endl
//    <<std::endl<<"Setting device matrix: " << matrixIndex << std::endl << dInstantaneousMatrices[matrixIndex]<<std::endl;
//#endif

*/
#ifdef BEAGLE_DEBUG_FLOW
    std::cerr<<"\tLeaving BeagleGPUActionImpl::setSparseMatrix\n";
#endif
    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
double BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::getPMax() const
{
    return floor(0.5 + 0.5 * sqrt(5.0 + 4.0 * mMax));
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::updateTransitionMatrices(int eigenIndex,
								      const int* probabilityIndices,
								      const int* firstDerivativeIndices,
								      const int* secondDerivativeIndices,
								      const Real* edgeLengths,
								      int count)
{
    return BeagleGPUImpl<Real>::updateTransitionMatrices(eigenIndex,
							 probabilityIndices,
							 firstDerivativeIndices,
							 secondDerivativeIndices,
							 edgeLengths,
							 count);
/*
    for (int i = 0; i < count; i++) {
        const int nodeIndex = probabilityIndices[i];
        hEigenMaps[nodeIndex] = eigenIndex;

        for (int category = 0; category < kCategoryCount; category++) {
            const double categoryRate = hCategoryRates[0][category]; // XJ: because rate categories are only set for first eigen index
            hEdgeMultipliers[nodeIndex * kCategoryCount + category] = edgeLengths[i] * categoryRate;
        }
    }
    // TODO: check if need to copy it from host to device afterwards
*/
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
