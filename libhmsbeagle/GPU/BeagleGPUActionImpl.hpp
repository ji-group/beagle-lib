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

// Create an io-manipulator so that we can write std::cerr<<byRow(D)<<"\n";
template <typename Real>
class byRowDense
{
    const DnMatrixDevice<Real>& D;
    bool multiline = false;
public:

    friend std::ostream& operator<<(std::ostream& o, const byRowDense<Real>& pr)
    {
        auto& D = pr.D;
	bool multiline = pr.multiline;

        auto hPtr = MemcpyDeviceToHostVector(D.ptr, D.size());

        if (not multiline) o<<"Rows[ ";
        for(int row=0;row<D.size1;row++)
        {
            if (not multiline) o<<"[ ";
            for(int col=0;col<D.size2;col++)
            {
                if (D.order == CUSPARSE_ORDER_COL)
                    o<<hPtr[col*D.size1 + row]<<", ";
                else
                    o<<hPtr[row*D.size2 + col]<<", ";
            }
	    if (not multiline)
		o<<"] ";
	    else if (row+1 != D.size1)
		o<<"\n";
        }
        if (not multiline) o<<"]";

	if (D.order == CUSPARSE_ORDER_COL)
	    o<<" (column-major)";
	else
	    o<<" (row-major)";

	if (multiline) o<<"\n";
        return o;
    }

    // Initialize the forwarding struct from the matrix that we want to print.
    byRowDense(const DnMatrixDevice<Real>& d, bool m=false):D(d),multiline(m) {}
};

// Create an io-manipulator so that we can write std::cerr<<byCol(D)<<"\n";
template <typename Real>
class byColDense
{
    const DnMatrixDevice<Real>& D;
public:

    friend std::ostream& operator<<(std::ostream& o, const byColDense<Real>& pr)
    {
	auto& D = pr.D;

        auto hPtr = MemcpyDeviceToHostVector(D.ptr, D.size());

        o<<"Cols[ ";
        for(int col=0;col<D.size2;col++)
        {
            o<<"[ ";
            for(int row=0;row<D.size1;row++)
            {
                if (D.order == CUSPARSE_ORDER_COL)
                    o<<hPtr[col*D.size1 + row]<<", ";
                else
                    o<<hPtr[row*D.size2 + col]<<", ";
            }
            o<<"] ";
        }
        o<<"]";

	if (D.order == CUSPARSE_ORDER_COL)
	    o<<" (column-major)";
	else
	    o<<" (row-major)";

        return o;
    }

    // Initialize the forwarding struct from the matrix that we want to print.
    byColDense(const DnMatrixDevice<Real>& d):D(d) {}
};

std::ostream& operator<<(std::ostream& o, sparseFormat f)
{
    if (f == sparseFormat::csr)
	o<<"csr";
    else if (f == sparseFormat::csc)
	o<<"csc";
    else
	o<<"sparseFormat=unknown";
    return o;
}
	
// Create an io-manipulator so that we can write std::cerr<<byRow(D)<<"\n";
template <typename Real>
class byRowSparse
{
    const SpMatrixDevice<Real>& S;
    bool multiline = false;
public:

    friend std::ostream& operator<<(std::ostream& o, const byRowSparse<Real>& pr)
    {
        auto& S = pr.S;
	bool multiline = pr.multiline;

	auto values = MemcpyDeviceToHostVector(S.values, S.num_non_zeros);
	auto inner = MemcpyDeviceToHostVector(S.inner, S.num_non_zeros);
	auto offsets = MemcpyDeviceToHostVector(S.offsets, S.outer_dim_size() + 1);
	
	std::vector< std::vector<Real> > D(S.size1, std::vector<Real>(S.size2, 0));
        for(int o=0; o< S.outer_dim_size();o++)
        {
            for(int index=offsets[o];index<offsets[o+1];index++)
            {
                if (S.format == sparseFormat::csr)
		    D[o][inner[index]] = values[index];
		else
		    D[inner[index]][o] = values[index];
            }
        }

        if (not multiline) o<<"Row[ ";
	for(int row=0;row<D.size();row++)
	{
            if (not multiline) o<<"[ ";
	    for(int col=0;col<D[row].size();col++)
	    {
		o<<D[row][col]<<", ";
            }
            if (not multiline)
		o<<"] ";
	    else if (row+1 != D.size())
		o<<"\n";
        }
        if (not multiline) o<<"]";

	o<<" ("<<S.format<<")";
	if (multiline) o<<"\n";

        return o;
    }

    // Initialize the forwarding struct from the matrix that we want to print.
    byRowSparse(const SpMatrixDevice<Real>& s, bool m = false):S(s),multiline(m) {}
};

template <typename Real>
byRowDense<Real> byRow(const DnMatrixDevice<Real>& D, bool m = false)
{
    return byRowDense<Real>(D, m);
}

template <typename Real>
byRowSparse<Real> byRow(const SpMatrixDevice<Real>& D, bool m = false)
{
    return byRowSparse<Real>(D, m);
}

template <typename Real>
byColDense<Real> byCol(const DnMatrixDevice<Real>& D)
{
    return byColDense<Real>(D);
}

template <typename Real>
std::ostream& operator<<(std::ostream& o, const DnMatrixDevice<Real>& D)
{
    return o<<byRow(D, true);
}

template <typename Real>
std::ostream& operator<<(std::ostream& o, const SpMatrixDevice<Real>& D)
{
    return o<<byRow(D, true);
}


template <typename Real>
struct asDeviceVec
{
    const Real* data;
    int length;

    asDeviceVec(const Real* d, int l):data(d),length(l) { }
};

template <typename Real>
std::ostream& operator<<(std::ostream& o, const asDeviceVec<Real>& D)
{
    auto H = MemcpyDeviceToHostVector(D.data, D.length);

    o<<"[ ";
    for(auto& h: H)
	o<<h<<" ";
    o<<"]";

    return o;
}

template <typename T>
std::ostream& operator<<(std::ostream& o, const std::vector<T>& H)
{
    o<<"[ ";
    for(auto& h: H)
	o<<h<<" ";
    o<<"]";

    return o;
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

    BeagleGPUImpl<Real>::createInstance(tipCount, partialsBufferCount * 2, 0, stateCount, patternCount,
                                        eigenDecompositionCount, matrixCount, categoryCount, scaleBufferCount, globalResourceNumber, pluginResourceNumber,
                                        preferenceFlags, requirementFlags);
    kPartialsCacheOffset = partialsBufferCount;
    hIdentity = SpMatrix<Real>(kPaddedStateCount, kPaddedStateCount);
    hIdentity.setIdentity();
    hInstantaneousMatrices.resize(kEigenDecompCount);
    hBs.resize(kEigenDecompCount);
    for (int i = 0; i < kEigenDecompCount; i++) {
        hInstantaneousMatrices[i] = SpMatrix<Real>(kPaddedStateCount, kPaddedStateCount);
        hBs[i] = SpMatrix<Real>(kPaddedStateCount, kPaddedStateCount);
    }
    hMuBs.resize(kEigenDecompCount);
    hB1Norms.resize(kEigenDecompCount);
    hds.resize(kEigenDecompCount);

    CUBLAS_CHECK(cublasCreate(&cublasHandle));  //TODO: destroyer: CUBLAS_CHECK(cublasDestroy(cublasHandle));
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

    dBsCsrOffsetsCache.resize(kEigenDecompCount);
    dBsCsrColumnsCache.resize(kEigenDecompCount);
    dBsCsrValuesCache.resize(kEigenDecompCount);
    dACscValuesCache.resize(kEigenDecompCount * kCategoryCount * 2);
    dAs = std::vector<SpMatrixDevice<Real>>(kEigenDecompCount * kCategoryCount * 2);
    currentCacheNNZs = std::vector<int>(kEigenDecompCount, kPaddedStateCount);
    for (int i = 0; i < kEigenDecompCount; i++) {
        dBsCsrOffsetsCache[i] = cudaDeviceNew<int>(kPaddedStateCount + 1);
        dBsCsrColumnsCache[i] = cudaDeviceNew<int>(currentCacheNNZs[i]);
        dBsCsrValuesCache[i] = cudaDeviceNew<Real>(currentCacheNNZs[i]);
        for (int j = 0; j < kCategoryCount; j++) {
            dACscValuesCache[i * kCategoryCount * 2 + 2 * j] = cudaDeviceNew<Real>(currentCacheNNZs[i]);
            dACscValuesCache[i * kCategoryCount * 2 + 2 * j + 1] = cudaDeviceNew<Real>(currentCacheNNZs[i]);
        }
    }


    dPartialsWrapper.resize(kPartialsBufferCount * kCategoryCount);
    for (int i = 0; i < kPartialsBufferCount; i++) {
        for (int category = 0; category < kCategoryCount; category++) {
	    auto ptr = (Real *) gpu->CreateSubPointer(dPartialsOrigin, sizeof(Real) * kPaddedStateCount * kPaddedPatternCount * (kCategoryCount * i + category), sizeof(Real) * kPaddedStateCount * kPaddedPatternCount);
	    auto mat = DnMatrixDevice<Real>(cublasHandle, ptr, kPaddedStateCount, kPaddedPatternCount);
            dPartialsWrapper[i * kCategoryCount + category] = std::move(mat);
        }
    }

    Real* dLeftCachePtr = cudaDeviceNew<Real>(kPaddedStateCount * kPaddedPatternCount * kCategoryCount);
    Real* dRightCachePtr = cudaDeviceNew<Real>(kPaddedStateCount * kPaddedPatternCount * kCategoryCount);

    Real* dIntegrationTmpLeftCachePtr = cudaDeviceNew<Real>(kPaddedStateCount * kPaddedPatternCount * kCategoryCount);
    Real* dIntegrationTmpRightCachePtr = cudaDeviceNew<Real>(kPaddedStateCount * kPaddedPatternCount * kCategoryCount);

    integrationLeftBufferSize = std::vector<size_t>(kCategoryCount, kPaddedStateCount * kPaddedPatternCount);
    dIntegrationLeftBuffer = std::vector<void*>(kCategoryCount, nullptr);
    integrationRightBufferSize = std::vector<size_t>(kCategoryCount, kPaddedStateCount * kPaddedPatternCount);
    dIntegrationRightBuffer = std::vector<void*>(kCategoryCount, nullptr);

    dFLeft.clear();

    dFRight.clear();

    dIntegrationTmpLeft.clear();

    dIntegrationTmpRight.clear();

    for (int category = 0; category < kCategoryCount; category++)
    {
	size_t offset = kPaddedStateCount * kPaddedPatternCount * category;

        dFLeft.emplace_back(cublasHandle, dLeftCachePtr + offset, kPaddedStateCount, kPaddedPatternCount);
        dFRight.emplace_back(cublasHandle, dRightCachePtr + offset, kPaddedStateCount, kPaddedPatternCount);

	dIntegrationTmpLeft.emplace_back(cublasHandle, dIntegrationTmpLeftCachePtr + offset,  kPaddedStateCount, kPaddedPatternCount);
	dIntegrationTmpRight.emplace_back(cublasHandle, dIntegrationTmpRightCachePtr + offset,  kPaddedStateCount, kPaddedPatternCount);
    }

    for (int category = 0; category < kCategoryCount; category++) {

        CHECK_CUDA(cudaMalloc(&dIntegrationLeftBuffer[category], integrationLeftBufferSize[category]))
        CHECK_CUDA(cudaMalloc(&dIntegrationRightBuffer[category], integrationRightBufferSize[category]))
    }

    hEigenMaps.resize(kPartialsCacheOffset);
    hEdgeMultipliers.resize(kPartialsCacheOffset * kCategoryCount);

    msCache = std::vector<tuple<int, int>>(2 * kCategoryCount, tuple<int, int>{0, 0});
    etaCache = std::vector<Real>(2 * kCategoryCount, 0);
    c1Cache.resize(2 * kCategoryCount);
    c2Cache.resize(2 * kCategoryCount);
    alphaCache.resize(2 * kCategoryCount);
    integrationMultipliers = std::vector<Real>(kCategoryCount * 2, 1);

    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::~BeagleGPUActionImpl()
{

    for (int i = 0; i < kCategoryCount; i++) {
        cudaFree(dIntegrationLeftBuffer[i]);
        cudaFree(dIntegrationRightBuffer[i]);
    }

    // The memory for each matrix group was allocation in a single block.
    cudaFree(dFLeft[0].ptr);
    cudaFree(dFRight[0].ptr);
    cudaFree(dIntegrationTmpLeft[0].ptr);
    cudaFree(dIntegrationTmpRight[0].ptr);

    for (int i = 0; i < kEigenDecompCount; i++) {
        cudaFree(dBsCsrOffsetsCache[i]);
        cudaFree(dBsCsrColumnsCache[i]);
        cudaFree(dBsCsrValuesCache[i]);
        for (int j = 0; j < kCategoryCount * 2; j++) {
            cudaFree(dACscValuesCache[i * kCategoryCount * 2 + j]);
//            cusparseDestroySpMat(dAs[i * kCategoryCount * 2 + i]);
        }
    }
    cublasDestroy(cublasHandle);
    cusparseDestroy(cusparseHandle);
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

    BeagleGPUImpl<Real>::setTipPartials(tipIndex, inPartials);

    for (int category = 0; category < kCategoryCount; category++) {
        dPartialsWrapper[getPartialIndex(tipIndex, category)].ptr = (Real*) dPartials[tipIndex] + kPaddedStateCount * kPaddedPatternCount * category;
    }


#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUActionImpl::setTipPartials\n");
#endif

    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::setPartials(int tipIndex, const Real* inPartials) {
#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUActionImpl::setPartials\n");
#endif

    BeagleGPUImpl<Real>::setPartials(tipIndex, inPartials);
    for (int category = 0; category < kCategoryCount; category++) {
        dPartialsWrapper[getPartialIndex(tipIndex, category)].ptr = (Real*) dPartials[tipIndex] + kPaddedStateCount * kPaddedPatternCount * category;
    }


#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUActionImpl::setPartials\n");
#endif

    return BEAGLE_SUCCESS;
}



BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::updatePartials(const int* operations,
                                                      int operationCount,
                                                      int cumulativeScalingIndex) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUActionImpl::updatePartials\n");
#endif

    bool byPartition = false;
    int returnCode = upPartials(byPartition,
                                operations,
                                operationCount,
                                cumulativeScalingIndex);

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUActionImpl::updatePartials\n");
#endif

    return returnCode;
}

BEAGLE_GPU_TEMPLATE
void  BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::rescalePartials(Real* partials, Real* scalingFactors, Real* cumulativeScalingBuffer, int streamIndex)
{

//            kernels->RescalePartials(partials3, scalingFactors, cumulativeScalingBuffer,
//                                     kPaddedPatternCount, kCategoryCount, 0, streamIndex, -1);

    auto hostPartials = MemcpyDeviceToHostVector(partials, kPaddedStateCount * kPaddedPatternCount * kCategoryCount);
    auto hostScalingFactors = MemcpyDeviceToHostVector(scalingFactors, kPaddedPatternCount);
    std::vector<Real> hostCumulativeScalingBuffer;
    if (cumulativeScalingBuffer)
        hostCumulativeScalingBuffer = MemcpyDeviceToHostVector(cumulativeScalingBuffer, kPaddedPatternCount);


    bool scalers_log = (kFlags & BEAGLE_FLAG_SCALERS_LOG)?true:false;
    for(int pattern = 0; pattern < kPatternCount;pattern++)
    {
        // FIND_MAX_PARTIALS_X_CPU();
        int deltaPartialsByState = pattern * kPaddedStateCount;
        REAL max = 0;
        for(int m = 0; m < kCategoryCount; m++)
        {
            int deltaPartialsByCategory = m * kPaddedStateCount * kPaddedPatternCount;
            int deltaPartials = deltaPartialsByCategory + deltaPartialsByState;
            for(int i = 0; i < kPaddedStateCount; i++) {
                REAL iPartial = hostPartials[deltaPartials + i];
                if (iPartial > max)
                    max = iPartial;
            }
        }

        if (max == 0)
        {
            max = 1.0;
            if (scalers_log)
                hostScalingFactors[pattern] = 0;
            else
                hostScalingFactors[pattern] = 1;
        }
        else
        {
            if (scalers_log)
            {
                REAL logMax = log(max);
                hostScalingFactors[pattern] = logMax;
                if (cumulativeScalingBuffer != 0)
                    hostCumulativeScalingBuffer[pattern] += logMax;
            }
            else
            {
                hostScalingFactors[pattern] = max;
                if (cumulativeScalingBuffer != 0)
                    hostCumulativeScalingBuffer[pattern] += log(max);
            }
        }

        // SCALE_PARTIALS_X_CPU();
        for(int m = 0; m < kCategoryCount; m++)
        {
            int deltaPartialsByCategory = m * kPaddedStateCount * kPaddedPatternCount;
            int deltaPartials = deltaPartialsByCategory + deltaPartialsByState;
            for(int i = 0; i < kPaddedStateCount; i++) {
                hostPartials[deltaPartials + i] /= max;
            }
        }
    }

    MemcpyHostToDevice( partials, hostPartials.data(), kPaddedStateCount * kPaddedPatternCount * kCategoryCount );
    MemcpyHostToDevice( scalingFactors, hostScalingFactors.data(), kPatternCount );
    if (cumulativeScalingBuffer)
        MemcpyHostToDevice( cumulativeScalingBuffer, hostCumulativeScalingBuffer.data(), kPatternCount );

//          std::cerr<<"rescaled partials (kernel) = "<<asDeviceVec((Real*)partials3, kPaddedStateCount * kPaddedPatternCount * kCategoryCount)<<"\n";
//          std::cerr<<"rescaled partials (CPU)    = "<<hostPartials<<"\n";
//          std::cerr<<"scaling factors (kernel) = "<<asDeviceVec((Real*)scalingFactors, kPatternCount)<<"\n";
//          std::cerr<<"scaling factors (CPU)    = "<<hostScalingFactors<<"\n";
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::getPartialIndex(int nodeIndex, int categoryIndex) {
    return nodeIndex * kCategoryCount + categoryIndex;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::getPartialCacheIndex(int nodeIndex, int categoryIndex) {
    return kPartialsCacheOffset * kCategoryCount + getPartialIndex(nodeIndex, categoryIndex);
}

BEAGLE_GPU_TEMPLATE
DnMatrixDevice<Real>& BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::getPartialsWrapper(int nodeIndex, int categoryIndex) {
    return dPartialsWrapper[getPartialIndex(nodeIndex, categoryIndex)];
}

BEAGLE_GPU_TEMPLATE
DnMatrixDevice<Real>& BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::getPartialsCacheWrapper(int nodeIndex, int categoryIndex) {
    return dPartialsWrapper[getPartialCacheIndex(nodeIndex, categoryIndex)];
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::upPartials(bool byPartition,
							const int *operations,
							int operationCount,
							int cumulativeScalingIndex)
{

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUActionImpl::upPartials\n");
#endif

    Real* cumulativeScalingBuffer = 0;
    if (cumulativeScalingIndex != BEAGLE_OP_NONE)
        cumulativeScalingBuffer = (Real*)dScalingFactors[cumulativeScalingIndex];

    int streamIndex = -1;
    int waitIndex = -1;

    for (int op = 0; op < operationCount; op++) {
        const int numOps = BEAGLE_OP_COUNT;

        const int destinationPartialIndex = operations[op * numOps];
        const int writeScalingIndex = operations[op * numOps + 1];
        const int readScalingIndex = operations[op * numOps + 2];
        const int firstChildPartialIndex = operations[op * numOps + 3];
        const int firstChildSubstitutionMatrixIndex = operations[op * numOps + 4];
        const int secondChildPartialIndex = operations[op * numOps + 5];
        const int secondChildSubstitutionMatrixIndex = operations[op * numOps + 6];

        int rescale = BEAGLE_OP_NONE;
        Real* scalingFactors = nullptr;

        if (kFlags & BEAGLE_FLAG_SCALING_AUTO) {
            int sIndex = destinationPartialIndex - kTipCount;

	    rescale = 2;
	    scalingFactors = (Real*)dScalingFactors[sIndex];
        } else if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS) {
            rescale = 1;
            scalingFactors = (Real*)dScalingFactors[destinationPartialIndex - kTipCount];
        } else if ((kFlags & BEAGLE_FLAG_SCALING_MANUAL) && writeScalingIndex >= 0) {
            rescale = 1;
            scalingFactors = (Real*)dScalingFactors[writeScalingIndex];
        } else if ((kFlags & BEAGLE_FLAG_SCALING_MANUAL) && readScalingIndex >= 0) {
            rescale = 0;
            scalingFactors = (Real*)dScalingFactors[readScalingIndex];
        }

        calcPartialsPartials(destinationPartialIndex, firstChildPartialIndex, firstChildSubstitutionMatrixIndex,
                             secondChildPartialIndex, secondChildSubstitutionMatrixIndex);


        if (rescale == 1)
        {
            Real* partials3 = (Real*)dPartials[destinationPartialIndex];

	    rescalePartials(partials3, scalingFactors, cumulativeScalingBuffer, streamIndex);
        }

        if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS) {
            int parScalingIndex = destinationPartialIndex - kTipCount;
            int child1ScalingIndex = firstChildPartialIndex - kTipCount;
            int child2ScalingIndex = secondChildPartialIndex - kTipCount;
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
    }

#ifdef BEAGLE_DEBUG_SYNCH
    gpu->SynchronizeHost();
#endif

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUActionImpl::upPartials\n");
#endif

    return BEAGLE_SUCCESS;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::PrintfDeviceVector(int* dPtr,
                                                                int length,
                                                                double checkValue,
                                                                int *signal,
                                                                Real r) {
    int* hPtr = (int*) malloc(sizeof(int) * length);

//    MemcpyDeviceToHost(hPtr, dPtr, sizeof(Real) * length);
    CHECK_CUDA(cudaMemcpy(hPtr, dPtr, sizeof(int) * length, cudaMemcpyDeviceToHost))
    printfInt(hPtr, length);

    if (checkValue != -1) {
        double sum = 0;
        for(int i=0; i<length; i++) {
            sum += hPtr[i];
            if( (hPtr[i] > checkValue) && (hPtr[i]-checkValue > 1.0E-4)) {
                fprintf(stderr,"Check value exception!  (%d) %2.5d > %2.5e (diff = %2.5e)\n",
                        i,hPtr[i],checkValue, (hPtr[i]-checkValue));
                if( signal != 0 )
                    *signal = 1;
            }
            if (hPtr[i] != hPtr[i]) {
                fprintf(stderr,"NaN found!\n");
                if( signal != 0 )
                    *signal = 1;
            }
        }
        if (sum == 0) {
            fprintf(stderr,"Zero-sum vector!\n");
            if( signal != 0 )
                *signal = 1;
        }
    }
    free(hPtr);
    return 0;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::PrintfDeviceVector(Real* dPtr,
                                                                 int length,
                                                                 double checkValue,
                                                                 int *signal,
                                                                 Real r) {
    Real* hPtr = (Real*) malloc(sizeof(Real) * length);

//    MemcpyDeviceToHost(hPtr, dPtr, sizeof(Real) * length);
    CHECK_CUDA(cudaMemcpy(hPtr, dPtr, sizeof(Real) * length, cudaMemcpyDeviceToHost))
    printfVector(hPtr, length);

    if (checkValue != -1) {
        double sum = 0;
        for(int i=0; i<length; i++) {
            sum += hPtr[i];
            if( (hPtr[i] > checkValue) && (hPtr[i]-checkValue > 1.0E-4)) {
                fprintf(stderr,"Check value exception!  (%d) %2.5e > %2.5e (diff = %2.5e)\n",
                        i,hPtr[i],checkValue, (hPtr[i]-checkValue));
                if( signal != 0 )
                    *signal = 1;
            }
            if (hPtr[i] != hPtr[i]) {
                fprintf(stderr,"NaN found!\n");
                if( signal != 0 )
                    *signal = 1;
            }
        }
        if (sum == 0) {
            fprintf(stderr,"Zero-sum vector!\n");
            if( signal != 0 )
                *signal = 1;
        }
    }
    free(hPtr);
    return 0;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::PrintfDeviceVector(cusparseDnMatDescr_t dPtr,
                                                                int length,
                                                                double checkValue,
                                                                int *signal,
                                                                Real r) {
    Real* hPtr = (Real*) malloc(sizeof(Real) * length);

//    MemcpyDeviceToHost(hPtr, dPtr, sizeof(Real) * length);
//    CHECK_CUDA(cudaMemcpy(hPtr, dPtr, sizeof(Real) * length, cudaMemcpyDeviceToHost))
    CHECK_CUSPARSE(cusparseConstDnMatGetValues(dPtr, reinterpret_cast<const void **>(hPtr)))
    printfVector(hPtr, length);

    if (checkValue != -1) {
        double sum = 0;
        for(int i=0; i<length; i++) {
            sum += hPtr[i];
            if( (hPtr[i] > checkValue) && (hPtr[i]-checkValue > 1.0E-4)) {
                fprintf(stderr,"Check value exception!  (%d) %2.5e > %2.5e (diff = %2.5e)\n",
                        i,hPtr[i],checkValue, (hPtr[i]-checkValue));
                if( signal != 0 )
                    *signal = 1;
            }
            if (hPtr[i] != hPtr[i]) {
                fprintf(stderr,"NaN found!\n");
                if( signal != 0 )
                    *signal = 1;
            }
        }
        if (sum == 0) {
            fprintf(stderr,"Zero-sum vector!\n");
            if( signal != 0 )
                *signal = 1;
        }
    }
    free(hPtr);
}

BEAGLE_GPU_TEMPLATE
void BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::calcPartialsPartials(int destPIndex,
                                                                   int partials1Index,
                                                                   int edgeIndex1,
                                                                   int partials2Index,
                                                                   int edgeIndex2) {

    cacheAMatrices(edgeIndex1, edgeIndex2, false);

    for (int category = 0; category < kCategoryCount; category++)
    {
        const int partial1Index = getPartialIndex(partials1Index, category);
        const int partial2Index = getPartialIndex(partials2Index, category);

        const int partial1CacheIndex = getPartialCacheIndex(partials1Index, category);
        const int partial2CacheIndex = getPartialCacheIndex(partials2Index, category);

        const int matrixIndex1 = hEigenMaps[edgeIndex1] * kCategoryCount * 2 + category;
        const int matrixIndex2 = hEigenMaps[edgeIndex2] * kCategoryCount * 2 + kCategoryCount + category;

        simpleAction2(getPartialsCacheWrapper(partials1Index, category), getPartialsWrapper(partials1Index, category),
                      edgeIndex1, category, matrixIndex1, true, false);

        simpleAction2(getPartialsCacheWrapper(partials2Index, category), getPartialsWrapper(partials2Index, category),
                      edgeIndex2, category, matrixIndex2, false, false);
    }

//    simpleAction3(partials1Index, edgeIndex1, partials2Index, edgeIndex2);
    cudaDeviceSynchronize();

    for (int category = 0; category < kCategoryCount; category++)
    {
        const int destPartialIndex = getPartialIndex(destPIndex, category);

        const int partial1CacheIndex = getPartialCacheIndex(partials1Index, category);
        const int partial2CacheIndex = getPartialCacheIndex(partials2Index, category);

	// element-wise multiply
        if constexpr (std::is_same<Real, float>::value) {
            CUBLAS_CHECK(cublasSdgmm(cublasHandle, CUBLAS_SIDE_LEFT, kPaddedStateCount * kPaddedPatternCount, 1, dPartialsWrapper[partial1CacheIndex].ptr, kPaddedStateCount * kPaddedPatternCount,
                                     dPartialsWrapper[partial2CacheIndex].ptr, 1, dPartialsWrapper[destPartialIndex].ptr, kPaddedStateCount * kPaddedPatternCount));

        } else {
            CUBLAS_CHECK(cublasDdgmm(cublasHandle, CUBLAS_SIDE_LEFT, kPaddedStateCount * kPaddedPatternCount, 1, dPartialsWrapper[partial1CacheIndex].ptr, kPaddedStateCount * kPaddedPatternCount,
                                     dPartialsWrapper[partial2CacheIndex].ptr, 1, dPartialsWrapper[destPartialIndex].ptr, kPaddedStateCount * kPaddedPatternCount));

        }
//#ifdef BEAGLE_DEBUG_FLOW
//        std::cerr<<"Checking p_parent = p_1 * p_2, parent index = "<<destPartialIndex<<" chil1 index = " << partial1CacheIndex<< " child2 index = "<<partial2CacheIndex<<std::endl;
//        std::cerr<<"p1 = "<<std::endl;
//        PrintfDeviceVector(dPartialCache[partial1CacheIndex], kPaddedStateCount * kPaddedPatternCount, -1, 0, 0);
//        std::cerr<<"p2 = "<<std::endl;
//        PrintfDeviceVector(dPartialCache[partial2CacheIndex], kPaddedStateCount * kPaddedPatternCount, -1, 0, 0);
//        std::cerr<<"p_parent = "<<std::endl;
//        PrintfDeviceVector(dPartialCache[destPartialIndex], kPaddedStateCount * kPaddedPatternCount, -1, 0, 0);
//#endif
    }
    cudaDeviceSynchronize();

}

BEAGLE_GPU_TEMPLATE
double BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::getPMax() const
{
    return floor(0.5 + 0.5 * sqrt(5.0 + 4.0 * mMax));
}

BEAGLE_GPU_TEMPLATE
double BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::getDValue(int p, int eigenIndex) const
{
    assert(p >= 0 and p < hds[eigenIndex].size());

    return hds[eigenIndex][p];
}

std::ostream& showScalingInfo(std::ostream& o, std::uint64_t kFlags, const int* cumulativeScaleIndices, int kScaleBufferSize)
{
    int scale = 1;
    if (kFlags & BEAGLE_FLAG_SCALING_AUTO)
	;
    else if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS)
	;
    else if (cumulativeScaleIndices[0] != BEAGLE_OP_NONE)
	;
    else
	scale = 0;

    o<<"scale = "<<scale
     <<"  scaling_auto = "<<bool(kFlags & BEAGLE_FLAG_SCALING_AUTO)
     <<"  scaling always = "<<bool(kFlags & BEAGLE_FLAG_SCALING_ALWAYS)
     <<"  scaling_dynamic = "<<bool(kFlags & BEAGLE_FLAG_SCALING_DYNAMIC)
     <<"  scalers = "<<bool(kFlags & BEAGLE_FLAG_SCALERS_LOG)
     <<"  scalers_raw = "<<bool(kFlags & BEAGLE_FLAG_SCALERS_RAW)
     <<"  cumulative_scale_indices = "<<cumulativeScaleIndices[0]
     <<"  kScaleBufferSize = "<<kScaleBufferSize
     <<"\n";

    return o;
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::calculateRootLogLikelihoods(const int* bufferIndices,
                                                                         const int* categoryWeightsIndices,
                                                                         const int* stateFrequenciesIndices,
                                                                         const int* cumulativeScaleIndices,
                                                                         int count,
                                                                         double* outSumLogLikelihood) {

#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tEntering BeagleGPUActionImpl::calculateRootLogLikelihoods\n");
#endif

    int returnCode = BEAGLE_SUCCESS;

    const int rootNodeIndex = bufferIndices[0];
    const int categoryWeightsIndex = categoryWeightsIndices[0];
    const int stateFrequenciesIndex = stateFrequenciesIndices[0];

    GPUPtr dCumulativeScalingFactor;
    bool scale = 1;
    if (kFlags & BEAGLE_FLAG_SCALING_AUTO)
        dCumulativeScalingFactor = dAccumulatedScalingFactors;
    else if (kFlags & BEAGLE_FLAG_SCALING_ALWAYS)
        dCumulativeScalingFactor = dScalingFactors[bufferIndices[0] - kTipCount];
    else if (cumulativeScaleIndices[0] != BEAGLE_OP_NONE)
        dCumulativeScalingFactor = dScalingFactors[cumulativeScaleIndices[0]];
    else
        scale = 0;

#ifdef BEAGLE_DEBUG_VALUES
    std::cerr<<"root partials = "<<asDeviceVec((Real*)dPartials[rootNodeIndex], kPaddedPatternCount * kPaddedStateCount * kCategoryCount)<<"\n";
#endif

    auto hRootPartials = MemcpyDeviceToHostVector((Real*)dPartials[rootNodeIndex], kPaddedPatternCount * kPaddedStateCount * kCategoryCount);
    auto hStateFrequencies = MemcpyDeviceToHostVector((Real*)dFrequencies[stateFrequenciesIndex], kStateCount);
    auto hCategoryWeights = MemcpyDeviceToHostVector((Real*)dWeights[categoryWeightsIndex], kCategoryCount);
    auto hPatternWeights = MemcpyDeviceToHostVector((Real*)dPatternWeights, kPatternCount);
    std::vector<Real> hScalingFactors;
    if (kFlags & BEAGLE_FLAG_SCALING_AUTO)
    {
	// scaling factor per pattern*category -- see BeagleGPUImpl<>::createInstance
	// scaling factor is char?? -- see BeagleGPUImpl<>::createInstance
	// scaling factor is int??  -- see KernelIntegrateLikelihoodsAutoScaling
	// See kernelIntegrateLikelihoodsAutoScaling in KernelsX.cu
	std::cerr<<"BeagleGPUActionImpl< >::calculateRootLogLikelihoods -- FLAG_SCALING_AUTO not implemented!";
	std::abort();
    }
    else if (scale)
	hScalingFactors = MemcpyDeviceToHostVector((Real*)dCumulativeScalingFactor, kScaleBufferSize);

    std::vector<Real> hColumnProbs(kPatternCount);

//    showScalingInfo(std::cerr, kFlags, cumulativeScaleIndices, kScaleBufferSize);
//    std::cerr<<"root partials (h) = "<<hRootPartials<<"\n";
//    std::cerr<<"state frequencies (h) = "<<hStateFrequencies<<"\n";
//    std::cerr<<"category weights (h) = "<<hCategoryWeights<<"\n";
//    std::cerr<<"scaling factors (h) = "<<hScalingFactors<<"\n";

    // We want to sum dRootPartials(category, pattern, state) * dWeights[category] * dFrequencies[state,category]
    //    over (category,state).
    // dRootPartials(c,p,s) = dRootPartials[c*kPatternCount*kPaddedStateCount + p*kPaddedStateCount + s]
    for(int pattern = 0; pattern < kPatternCount; pattern++)
    {
	double Pr = 0;
	for(int category = 0; category < kCategoryCount; category++)
	{
	    double tmp = 0;
	    for(int state = 0; state < kStateCount; state++)
	    {
		tmp += hRootPartials[state + pattern*kPaddedStateCount + category*kPaddedStateCount*kPaddedPatternCount] * hStateFrequencies[state];
	    }

	    Pr += tmp * hCategoryWeights[category];
	}
	hColumnProbs[pattern] = Pr;
    }

    for(auto& Pr: hColumnProbs)
	Pr = log(Pr);

    for(int pattern = 0; pattern < kPatternCount; pattern++)
    {
	if (scale)
	    hColumnProbs[pattern] += hScalingFactors[pattern];
    }

    MemcpyHostToDevice((Real*)dIntegrationTmp, (Real*)hColumnProbs.data(), kPatternCount);

/*
    std::cerr<<"site probs (d1) = "<<asDeviceVec((Real*)dIntegrationTmp, kPatternCount)<<"\n";
*/

    // Step 1. Sum over states: pi * partials -> TMP1
    // pi = 1 x kStateCount       partials = kStateCount * kPatternCount * kCategoryCount      TMP = 1 x (kPatternCount * kCategoryCount)
    // DnMatrixDevice<Real> PI(cublasHandle, (Real*)dFrequencies[stateFrequenciesIndex], 1, kPaddedStateCount); // grouped by column
    // DnMatrixDevice<Real> PARTIALS(cublasHandle, dIntegrationTmp, kPaddedStateCount, kPaddedPatternCount * kCategoryCount); // grouped by column

    // ***STARTHERE*** PROBLEM - We need some memory here to store TMP1!
    // Alternatively we MODIFY partials(cat,pat,state) by multiplying by weight(cat) and freq(state), and then just sum it.
    // cublas is heavily optimized, so writing our own kernel might not be so easy: https://siboehm.com/articles/22/CUDA-MMM

    // Step 2. Sum over category weights: TMP1 * cats -> columnProbs
    // TMP = kPatternCount x kCategoryCount     cats = kCategoryCount x 1     columnProbs = kPatternCount x 1

    // Step 3. Take the log of all the column probabilities.

    // Step 4. Take the sum of the logged column probabilities.

    // Step 5. If cumulativeScalingBuffer, take the sum of the cumulativeScalingBuffer also.

    /*
    if (scale) {
        // if SCALING_AUTO -> See kernelIntegrateLikelihoodsAutoScaling (in KernelsX.cu)
        // otherwise       -> see kernelIntegrateLikelihoodsFixedScale  (in KernelsX.cu)
        kernels->IntegrateLikelihoodsDynamicScaling(dIntegrationTmp, dPartials[rootNodeIndex],
                                                    dWeights[categoryWeightsIndex],
                                                    dFrequencies[stateFrequenciesIndex],
                                                    dCumulativeScalingFactor,
                                                    kPaddedPatternCount,
                                                    kCategoryCount);
    } else {
        // See kernelIntegrateLikelihoods in KernelsX.cu
        kernels->IntegrateLikelihoods(dIntegrationTmp, dPartials[rootNodeIndex],
                                      dWeights[categoryWeightsIndex],
                                      dFrequencies[stateFrequenciesIndex],
                                      kPaddedPatternCount, kCategoryCount);
    }
    */

#ifdef BEAGLE_DEBUG_VALUES
    std::cerr<<"before pattern weights = "<<asDeviceVec((Real*)dIntegrationTmp, kPatternCount)<<"\n";
#endif

    // Take the dot product of the pattern log-likelihoods and the pattern weights.
    // The output (dSumLogLikelihood) needs to be a device pointer.
    dotProduct((Real*)dSumLogLikelihood, cublasHandle, kPatternCount, (Real*)dIntegrationTmp, (Real*)dPatternWeights);

//    std::cerr<<"logLikelihood (kernel) = "<<MemcpyDeviceToHostVector((Real*)dSumLogLikelihood,1)[0]<<"   logLikelihood (ours) = "<<OurResult<<"\n";

    if (kFlags & BEAGLE_FLAG_COMPUTATION_SYNCH) {
        gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumLogLikelihood, sizeof(Real) * kSumSitesBlockCount);

        *outSumLogLikelihood = 0.0;
        for (int i = 0; i < kSumSitesBlockCount; i++) {
            if (hLogLikelihoodsCache[i] != hLogLikelihoodsCache[i])
                returnCode = BEAGLE_ERROR_FLOATING_POINT;

            *outSumLogLikelihood += hLogLikelihoodsCache[i];
        }
    }

#ifdef BEAGLE_DEBUG_VALUES
    Real r = 0;
    fprintf(stderr, "parent = \n");
    gpu->PrintfDeviceVector(dIntegrationTmp, kPatternCount, r);
#endif


#ifdef BEAGLE_DEBUG_FLOW
    fprintf(stderr, "\tLeaving  BeagleGPUActionImpl::calculateRootLogLikelihoods\n");
#endif

    return returnCode;
}


BEAGLE_GPU_TEMPLATE
std::tuple<int,int> BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::getStatistics2(double t, int nCol,
                                                              double edgeMultiplier,
                                                              int eigenIndex) const {
    assert( t >= 0 );
    assert( nCol >= 0);
    assert( edgeMultiplier >= 0 );
    assert( eigenIndex >= 0);

    if (t * hB1Norms[eigenIndex] == 0.0)
        return {0, 1};

    int bestM = INT_MAX;
    double bestS = INT_MAX;  // Not all the values of s can fit in a 32-bit int.

    const double theta = thetaConstants.at(mMax);
    const double pMax = getPMax();
    // pMax is the largest positive integer such that p*(p-1) <= mMax + 1

    const bool conditionFragment313 = hB1Norms[eigenIndex] * edgeMultiplier <= 2.0 * theta / ((double) nCol * mMax) * pMax * (pMax + 3);
    // using l = 1 as in equation 3.13
    if (conditionFragment313) {
        for (auto& [thisM, thetaM]: thetaConstants) {
            const double thisS = ceil(hB1Norms[eigenIndex] * edgeMultiplier / thetaM);
            if (bestM == INT_MAX || ((double) thisM) * thisS < bestM * bestS) {
                bestS = thisS;
                bestM = thisM;
            }
        }
    } else {
        for (int p = 2; p < pMax; p++) {
            for (int thisM = p * (p - 1) - 1; thisM < mMax + 1; thisM++) {
                auto it = thetaConstants.find(thisM);
                if (it != thetaConstants.end()) {
                    // equation 3.7 in Al-Mohy and Higham
                    const double dValueP = getDValue(p, eigenIndex);
                    const double dValuePPlusOne = getDValue(p + 1, eigenIndex);
                    const double alpha = std::max(dValueP, dValuePPlusOne) * edgeMultiplier;
                    // part of equation 3.10
                    const double thisS = ceil(alpha / thetaConstants.at(thisM));
                    if (bestM == INT_MAX || ((double) thisM) * thisS < bestM * bestS) {
                        bestS = thisS;
                        bestM = thisM;
                    }
                }
            }
        }
    }
    bestS = std::max(std::min<double>(bestS, INT_MAX), 1.0);
    assert( bestS >= 1 );
    assert( bestS <= INT_MAX );

    int m = bestM;
    int s = (int) bestS;

    assert(m >= 0);
    assert(s >= 1);

    return {m,s};
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::cacheAMatrices(int edgeIndex1, int edgeIndex2, bool transpose)
{
    for (int category = 0; category < kCategoryCount; category++) {
        const int matrixIndex1 = hEigenMaps[edgeIndex1] * kCategoryCount * 2 + category;
        const int edgeMultiplierIndex1 = edgeIndex1 * kCategoryCount + category;
        const Real edgeMultiplier1 = hEdgeMultipliers[edgeMultiplierIndex1];

        const int matrixIndex2 = hEigenMaps[edgeIndex2] * kCategoryCount * 2 + kCategoryCount + category;
        const int edgeMultiplierIndex2 = edgeIndex2 * kCategoryCount + category;
        const Real edgeMultiplier2 = hEdgeMultipliers[edgeMultiplierIndex2];

        MemcpyDeviceToDevice(dACscValuesCache[matrixIndex1], dBsCsrValuesCache[hEigenMaps[edgeIndex1]], currentCacheNNZs[hEigenMaps[edgeIndex1]]);
        MemcpyDeviceToDevice(dACscValuesCache[matrixIndex2], dBsCsrValuesCache[hEigenMaps[edgeIndex2]], currentCacheNNZs[hEigenMaps[edgeIndex2]]);
	auto format = transpose ? sparseFormat::csc : sparseFormat::csr;

	dAs[matrixIndex1] = SpMatrixDevice<Real>(cublasHandle, cusparseHandle,
						 kPaddedStateCount, kPaddedStateCount,
						 currentCacheNNZs[hEigenMaps[edgeIndex1]],
						 dACscValuesCache[matrixIndex1],
						 dBsCsrColumnsCache[hEigenMaps[edgeIndex1]],
						 dBsCsrOffsetsCache[hEigenMaps[edgeIndex1]],
						 format);

	dAs[matrixIndex2] = SpMatrixDevice<Real>(cublasHandle, cusparseHandle,
						 kPaddedStateCount, kPaddedStateCount,
						 currentCacheNNZs[hEigenMaps[edgeIndex2]],
						 dACscValuesCache[matrixIndex2],
						 dBsCsrColumnsCache[hEigenMaps[edgeIndex2]],
						 dBsCsrOffsetsCache[hEigenMaps[edgeIndex2]],
						 format);

	dAs[matrixIndex1] *= edgeMultiplier1; //Check if this is asynchronous with the following
	dAs[matrixIndex2] *= edgeMultiplier2;

#ifdef BEAGLE_DEBUG_FLOW
        std::cerr<<"category = "<<category<<std::endl;
        std::cerr<<"matrixIndex1 = "<<matrixIndex1<<std::endl;
        std::cerr<<"edgeIndex1 = "<<edgeIndex1<<std::endl;
        std::cerr<<"edgeMultiplierIndex1 = "<<edgeMultiplierIndex1<<std::endl;
        std::cerr<<"edgeMultiplier1 = "<<edgeMultiplier1<<std::endl;
        std::cerr<<"hEigenMaps[edgeIndex1] = "<<hEigenMaps[edgeIndex1]<<std::endl;
	std::cerr<<"dAs["<<matrixIndex1<<"] = \n"<<dAs[matrixIndex1]<<"\n";
#endif
    }
    cudaDeviceSynchronize();
    return BEAGLE_SUCCESS;
}

template <typename Real>
int spMM(cusparseHandle_t handle, cusparseDnMatDescr_t C, Real alpha, cusparseSpMatDescr_t A, cusparseDnMatDescr_t B, Real beta, void*& buffer, size_t& buffersize)
{
    size_t new_buffersize;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, A, B, &beta, C, DataType<Real>,
                                           CUSPARSE_SPMM_ALG_DEFAULT, &new_buffersize));

    if(new_buffersize > buffersize)
    {
        CHECK_CUDA(cudaFree(buffer));
        CHECK_CUDA(cudaMalloc(&buffer, buffersize));
        buffersize = new_buffersize;
    }

    // integrationTmp = alpha * A * destP
    CHECK_CUSPARSE(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, A, B, &beta, C, DataType<Real>,
                                CUSPARSE_SPMM_ALG_DEFAULT, buffer)); //row-major layout provides higher performance (?)
    return 0;
}

template <typename Real>
int spMM(DnMatrixDevice<Real>& C, Real alpha, const SpMatrixDevice<Real>& A, const DnMatrixDevice<Real>& B, Real beta, void*& buffer, size_t& buffersize)
{
    return spMM<Real>(A.cusparseHandle, C.descr, alpha, A.descr, B.descr, beta, buffer, buffersize);
}

BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::simpleAction2(DnMatrixDevice<Real>& destP, const DnMatrixDevice<Real>& inPartials, int edgeIndex, int category, int matrixIndex, bool left, bool transpose) {
    const double tol = pow(2.0, -53.0);
    const double t = 1.0;
    const int nCol = kPaddedStateCount * kPaddedPatternCount;
    const int edgeMultiplierIndex = edgeIndex * kCategoryCount + category;

    const double edgeMultiplier = hEdgeMultipliers[edgeMultiplierIndex];

    auto [m, s] = getStatistics2(t, nCol, edgeMultiplier, hEigenMaps[edgeIndex]);


#ifdef BEAGLE_DEBUG_FLOW
    std::cerr << "\n\nsimpleAction2: m = " << m << "  s = " << s << " t = " << t << " nCol = " << nCol << " edgeMultiplier = " << edgeMultiplier << std::endl;
    std::cerr << "  edgeIndex = " << edgeIndex << " category = " << category << " matrixIndex = "<<matrixIndex << std::endl;
#endif

//    SpMatrix<Real> A = hBs[hEigenMaps[edgeIndex]] * edgeMultiplier;
    auto& A = dAs[matrixIndex];
    vector<DnMatrixDevice<Real>>* FF;
    vector<DnMatrixDevice<Real>>* integrationTmpPtr;
    std::vector<void*> integrationBuffer;
    std::vector<size_t> integrationBufferSize;
    if (left) {
        FF = &dFLeft;
        integrationTmpPtr = &dIntegrationTmpLeft;
        integrationBuffer = dIntegrationLeftBuffer;
        integrationBufferSize = integrationLeftBufferSize;
    } else {
        FF = &dFRight;
        integrationTmpPtr = &dIntegrationTmpRight;
        integrationBuffer = dIntegrationRightBuffer;
        integrationBufferSize = integrationRightBufferSize;
    }
    auto& F = (*FF)[category];
    auto& integrationTmp = (*integrationTmpPtr)[category];

//#ifdef BEAGLE_DEBUG_FLOW
//    std::cerr<<"Before destP = partials operation, destPCache:\n"<<std::endl;
//    std::cerr<<byCol(inPartials)<<"\n";
//#endif

//    destP = partials;
    destP.copyFrom( inPartials );

//#ifdef BEAGLE_DEBUG_FLOW
//    std::cerr<<"destP = partials:\n";
//    std::cerr<<"   from: "<<byCol(inPartials)<<"\n";
//    std::cerr<<"   to:   "<<byCol(destP)<<"\n";
//#endif

//    MatrixXd F(kStateCount, nCol);
//    F = destP;
    F.copyFrom( inPartials );

//#ifdef BEAGLE_DEBUG_FLOW
//    std::cerr<<"F = partials operation, FCache:\n"<<F<<std::endl;
//#endif
    cudaDeviceSynchronize();

    const Real eta = exp(t * hMuBs[hEigenMaps[edgeIndex]] * edgeMultiplier / (Real) s);

//#ifdef BEAGLE_DEBUG_FLOW
//    std::cerr<<"eta = "<<eta<<std::endl;
//#endif
    const Real zero = 0;
    const Real one = 1;
    for (int i = 0; i < s; i++) {
//#ifdef BEAGLE_DEBUG_FLOW
//        std::cerr<<"dPartials:\n"<<inPartials<<std::endl;
//#endif

        Real c1 = normPInf(inPartials);

        for (int j = 1; j < m + 1; j++) {
//#ifdef BEAGLE_DEBUG_FLOW
//            std::cerr<<"j/m = "<<j<<"/"<<m<<", alpha = "<<alpha<<std::endl;
//#endif

//            integrationTmp = t / (s * j) * A * destP;
            spMM<Real>(integrationTmp, t / ((Real) s * j), A, destP, 0, integrationBuffer[category], integrationBufferSize[category]);

//#ifdef BEAGLE_DEBUG_FLOW
//            std::cerr<<"edge multiplier = "<< hEdgeMultipliers[edgeIndex * kCategoryCount + category]<<"\nB ="<<std::endl;
//            PrintfDeviceVector(dBsCsrValuesCache[hEigenMaps[edgeIndex]], currentCacheNNZs[hEigenMaps[edgeIndex]], -1, 0, 0);
//            std::cerr<<"A =\n"<<A<<"\n";
//            std::cerr<<"AP * alpha = "<<byCol(integrationTmp)<<"\n";
//#endif

            destP.copyFrom( integrationTmp );

//#ifdef BEAGLE_DEBUG_FLOW
//            std::cerr<<"P = IntegrationTmp = "<<byCol(destP)<<std::endl;
//#endif

	    F += destP;

            Real c2 = normPInf(destP);
//#ifdef BEAGLE_DEBUG_FLOW
//            std::cerr<<"F += destP, c1 = " <<c1 <<"  c2 = " <<c2 <<byCol(F)<<std::endl;
//#endif

            if (c1 + c2 <= tol * normPInf(F)) {
                break;
            }
            c1 = c2;
        }

	F *= eta;

//#ifdef BEAGLE_DEBUG_FLOW
//        std::cerr<<"F *= eta (eta ="<<eta<<"): "<<F<<"\n";
//#endif

//        destP = F;
        destP.copyFrom ( F );

#ifdef BEAGLE_DEBUG_FLOW
        std::cerr<<"i = "<<i<<" destP = F:\n"<<"  "<<byCol(destP)<<"\n";
#endif
    }

    return BEAGLE_SUCCESS;
}


BEAGLE_GPU_TEMPLATE
int BeagleGPUActionImpl<BEAGLE_GPU_GENERIC>::simpleAction3(int partialsIndex1, int edgeIndex1,
                                                           int partialsIndex2, int edgeIndex2) {
    const double tol = pow(2.0, -53.0);
    const double t = 1.0;
    const int nCol = kPaddedStateCount * kPaddedPatternCount;

    int m_max = 0, s_max = 0;
    for (int category = 0; category < kCategoryCount; category++) {
        const int edgeMultiplierIndex1 = edgeIndex1 * kCategoryCount + category;
        const double edgeMultiplier1 = hEdgeMultipliers[edgeMultiplierIndex1];
        msCache[category] = getStatistics2(t, nCol, edgeMultiplier1, hEigenMaps[edgeIndex1]);
        auto [m1, s1] = msCache[category];
        etaCache[category] = exp(t * hMuBs[hEigenMaps[edgeIndex1]] * edgeMultiplier1 / (Real) s1);
        if (m_max < m1) m_max = m1;
        if (s_max < s1) s_max = s1;

        const int edgeMultiplierIndex2 = edgeIndex2 * kCategoryCount + category;
        const double edgeMultiplier2 = hEdgeMultipliers[edgeMultiplierIndex2];
        msCache[kCategoryCount + category] = getStatistics2(t, nCol, edgeMultiplier2, hEigenMaps[edgeIndex2]);
        auto [m2, s2] = msCache[kCategoryCount + category];
        etaCache[kCategoryCount + category] = exp(t * hMuBs[hEigenMaps[edgeIndex2]] * edgeMultiplier2 / (Real) s2);
        if (m_max < m2) m_max = m2;
        if (s_max < s2) s_max = s2;
    }


    //    destP = partials;
    CHECK_CUDA(cudaMemcpyAsync(dPartialsWrapper[getPartialCacheIndex(partialsIndex1, 0)].ptr, dPartialsWrapper[getPartialIndex(partialsIndex1, 0)].ptr, sizeof(Real) * kPaddedStateCount * kPaddedPatternCount * kCategoryCount, cudaMemcpyDeviceToDevice))
    CHECK_CUDA(cudaMemcpyAsync(dPartialsWrapper[getPartialCacheIndex(partialsIndex2, 0)].ptr, dPartialsWrapper[getPartialIndex(partialsIndex2, 0)].ptr, sizeof(Real) * kPaddedStateCount * kPaddedPatternCount * kCategoryCount, cudaMemcpyDeviceToDevice))
    //    F = destP;
    CHECK_CUDA(cudaMemcpyAsync(dFLeft[0].ptr, dPartialsWrapper[getPartialIndex(partialsIndex1, 0)].ptr, sizeof(Real) * kPaddedStateCount * kPaddedPatternCount * kCategoryCount, cudaMemcpyDeviceToDevice))
    CHECK_CUDA(cudaMemcpyAsync(dFRight[0].ptr, dPartialsWrapper[getPartialIndex(partialsIndex2, 0)].ptr, sizeof(Real) * kPaddedStateCount * kPaddedPatternCount * kCategoryCount, cudaMemcpyDeviceToDevice))

    cudaDeviceSynchronize();

#ifdef BEAGLE_DEBUG_FLOW
    std::cerr<<"P:" <<std::endl;
    PrintfDeviceVector(dPartialsWrapper[getPartialIndex(partialsIndex1, 0)].ptr, kPaddedStateCount * kPaddedPatternCount * kCategoryCount, -1, 0, 0);
    std::cerr<<"destP:" <<std::endl;
    PrintfDeviceVector(dPartialsWrapper[getPartialCacheIndex(partialsIndex1, 0)].ptr, kPaddedStateCount * kPaddedPatternCount * kCategoryCount, -1, 0, 0);
    std::cerr<<"F:" <<std::endl;
    PrintfDeviceVector(dFLeft[0].ptr, kPaddedStateCount * kPaddedPatternCount * kCategoryCount, -1, 0, 0);
    std::cerr<<"ms:"<<std::endl;
    for (int i = 0; i < kCategoryCount; i++) {
        std::cerr<<"("<<std::get<0>(msCache[i])<<", "<<std::get<1>(msCache[i])<<"), ";
        std::cerr<<"("<<std::get<0>(msCache[i + kCategoryCount])<<", "<<std::get<1>(msCache[i  + kCategoryCount])<<"), ";
    }
#endif

    const Real zero = 0;
    const Real one = 1;
    for (int i = 0; i < s_max; i++) {

        for (int category = 0; category < kCategoryCount; category++) {
            const Real c1Left = normPInf(dPartialsWrapper[getPartialIndex(partialsIndex1, category)]);
            c1Cache[category] = c1Left;
            const Real c1Right = normPInf(dPartialsWrapper[getPartialIndex(partialsIndex2, category)]);
            c1Cache[kCategoryCount + category] = c1Right;

            etaCache[category] = (i < std::get<1>(msCache[category])) * (etaCache[category] - 1) + 1;
            etaCache[kCategoryCount + category] = (i < std::get<1>(msCache[kCategoryCount + category])) * (etaCache[kCategoryCount + category] - 1) + 1;
        }
        std::fill(integrationMultipliers.begin(), integrationMultipliers.end(), 1.0);

#ifdef BEAGLE_DEBUG_FLOW
        std::cerr<<"s/s_max = "<<i<<"/"<<s_max<<", eta = ";
        for (int i = 0; i < 2 * kCategoryCount; i++) {
            std::cerr<<etaCache[i]<<", ";
        }
        std::cerr<<std::endl;
#endif

        for (int j = 1; j < m_max + 1; j++) {

            for (int category = 0; category < kCategoryCount; category++) {
                alphaCache[category] = t / ((Real) std::get<1>(msCache[category]) * j) * integrationMultipliers[category] * (j < std::get<0>(msCache[category]) && i < std::get<1>(msCache[category]));
                alphaCache[kCategoryCount + category] = t / ((Real) std::get<1>(msCache[kCategoryCount + category]) * j) * integrationMultipliers[kCategoryCount + category] * (j < std::get<0>(msCache[kCategoryCount + category]) && i < std::get<1>(msCache[kCategoryCount + category]));
            }

#ifdef BEAGLE_DEBUG_FLOW
            std::cerr<<"\nj/m_max = "<<j<<"/"<<m_max<<", alpha = ";
            for (int i = 0; i < 2 * kCategoryCount; i++) {
                std::cerr<<alphaCache[i]<<", ";
            }
            std::cerr<<std::endl;
#endif

            for (int category = 0; category < kCategoryCount; category++) {

                spMM<Real>(dIntegrationTmpLeft[category], alphaCache[category], dAs[hEigenMaps[edgeIndex1] * kCategoryCount * 2 + category], dPartialsWrapper[getPartialCacheIndex(partialsIndex1, category)], 0, dIntegrationLeftBuffer[category], integrationLeftBufferSize[category]);
                spMM<Real>(dIntegrationTmpRight[category], alphaCache[kCategoryCount + category], dAs[hEigenMaps[edgeIndex2] * kCategoryCount * 2 + kCategoryCount + category], dPartialsWrapper[getPartialCacheIndex(partialsIndex2, category)], 0, dIntegrationRightBuffer[category], integrationRightBufferSize[category]);

            }
            cudaDeviceSynchronize();

#ifdef BEAGLE_DEBUG_FLOW
            std::cerr<<"integrationTmp = alpha * A * destP = " <<std::endl;
            PrintfDeviceVector(dIntegrationTmpLeft[0].ptr, kPaddedStateCount * kPaddedPatternCount * kCategoryCount, -1, 0, 0);
            PrintfDeviceVector(dIntegrationTmpRight[0].ptr, kPaddedStateCount * kPaddedPatternCount * kCategoryCount, -1, 0, 0);
#endif

            // destP = IntegrationTmp
            CHECK_CUDA(cudaMemcpyAsync(dPartialsWrapper[getPartialCacheIndex(partialsIndex1, 0)].ptr, dIntegrationTmpLeft[0].ptr, sizeof(Real) * kPaddedStateCount * kPaddedPatternCount * kCategoryCount, cudaMemcpyDeviceToDevice))
            CHECK_CUDA(cudaMemcpyAsync(dPartialsWrapper[getPartialCacheIndex(partialsIndex2, 0)].ptr, dIntegrationTmpRight[0].ptr, sizeof(Real) * kPaddedStateCount * kPaddedPatternCount * kCategoryCount, cudaMemcpyDeviceToDevice))

//            F += destP = IntegrationTmp;
            if constexpr (std::is_same<Real, float>::value) {
                CUBLAS_CHECK(cublasSaxpy(cublasHandle, kPaddedStateCount * kPaddedPatternCount * kCategoryCount, &one, dIntegrationTmpLeft[0].ptr, 1, dFLeft[0].ptr, 1));
                CUBLAS_CHECK(cublasSaxpy(cublasHandle, kPaddedStateCount * kPaddedPatternCount * kCategoryCount, &one, dIntegrationTmpRight[0].ptr, 1, dFRight[0].ptr, 1));
            } else {
                CUBLAS_CHECK(cublasDaxpy(cublasHandle, kPaddedStateCount * kPaddedPatternCount * kCategoryCount, &one, dIntegrationTmpLeft[0].ptr, 1, dFLeft[0].ptr, 1));
                CUBLAS_CHECK(cublasDaxpy(cublasHandle, kPaddedStateCount * kPaddedPatternCount * kCategoryCount, &one, dIntegrationTmpRight[0].ptr, 1, dFRight[0].ptr, 1));
            }

            cudaDeviceSynchronize();

            for (int category = 0; category < kCategoryCount; category++) {
                c2Cache[category] = normPInf(dIntegrationTmpLeft[category]);
                c2Cache[kCategoryCount + category] = normPInf(dIntegrationTmpRight[category]);
                if (c1Cache[category] + c2Cache[category] < tol * normPInf(dFLeft[category])) {
                    integrationMultipliers[category] = 0;
                }
                if (c1Cache[kCategoryCount + category] + c2Cache[kCategoryCount + category] < tol * normPInf(dFRight[category])) {
                    integrationMultipliers[kCategoryCount + category] = 0;
                }

                c1Cache[category] = c2Cache[category];
                c1Cache[kCategoryCount + category] = c2Cache[kCategoryCount + category];
            }

#ifdef BEAGLE_DEBUG_FLOW
            std::cerr<<"dFLeftCache = " <<std::endl;
            PrintfDeviceVector(dFLeft[0].ptr, kPaddedStateCount * kPaddedPatternCount * kCategoryCount, -1, 0, 0);
            std::cerr<<"dFRightCache = " <<std::endl;
            PrintfDeviceVector(dFRight[0].ptr, kPaddedStateCount * kPaddedPatternCount * kCategoryCount, -1, 0, 0);
            std::cerr<<"c2Cache = ";
            for (int i = 0; i < 2 * kCategoryCount; i++) {
                std::cerr<<c2Cache[i]<<", ";
            }
            std::cerr<<std::endl;
#endif

            if ( std::all_of(integrationMultipliers.begin(), integrationMultipliers.end(), [](Real i) { return i==0; }) ) {
                break;
            }

        }
//        F *= eta;
        for (int category = 0; category < kCategoryCount; category++) {
            if constexpr (std::is_same<Real, float>::value) {
                CUBLAS_CHECK(cublasSscal(cublasHandle, kPaddedStateCount * kPaddedPatternCount, &etaCache[category], dFLeft[category].ptr, 1));
                CUBLAS_CHECK(cublasSscal(cublasHandle, kPaddedStateCount * kPaddedPatternCount, &etaCache[kCategoryCount + category], dFRight[category].ptr, 1));
            } else {
                CUBLAS_CHECK(cublasDscal(cublasHandle, kPaddedStateCount * kPaddedPatternCount, &etaCache[category], dFLeft[category].ptr, 1));
                CUBLAS_CHECK(cublasDscal(cublasHandle, kPaddedStateCount * kPaddedPatternCount, &etaCache[kCategoryCount + category], dFRight[category].ptr, 1));
            }
        }
        cudaDeviceSynchronize();
//        destP = F;
        CHECK_CUDA(cudaMemcpyAsync(dPartialsWrapper[getPartialCacheIndex(partialsIndex1, 0)].ptr, dFLeft[0].ptr, sizeof(Real) * kPaddedStateCount * kPaddedPatternCount * kCategoryCount, cudaMemcpyDeviceToDevice))
        CHECK_CUDA(cudaMemcpyAsync(dPartialsWrapper[getPartialCacheIndex(partialsIndex2, 0)].ptr, dFRight[0].ptr, sizeof(Real) * kPaddedStateCount * kPaddedPatternCount * kCategoryCount, cudaMemcpyDeviceToDevice))
        cudaDeviceSynchronize();
    }

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
    std::vector<Triplet<Real>> tripletList;
    for (int i = 0; i < numNonZeros; i++) {
        tripletList.push_back(Triplet<Real>(rowIndices[i], colIndices[i], values[i]));
    }
    hInstantaneousMatrices[matrixIndex].setFromTriplets(tripletList.begin(), tripletList.end());

#ifdef BEAGLE_DEBUG_FLOW

    std::cerr<<"CPU instantaneous matrix "<<matrixIndex<<" =\n"<<hInstantaneousMatrices[matrixIndex]<<std::endl;
    std::cerr<<"outerIndexPtr = \n";
    for (int i = 0; i < kPaddedStateCount + 1; i++) {
        std::cerr<<hInstantaneousMatrices[matrixIndex].outerIndexPtr()[i]<<", ";
    }
    std::cerr<<"\ninnerIndexPtr = \n";
    for (int i = 0; i < hInstantaneousMatrices[matrixIndex].nonZeros(); i++) {
        std::cerr<<hInstantaneousMatrices[matrixIndex].innerIndexPtr()[i]<<", ";
    }
    std::cerr<<"\nvaluePtr = \n";
    for (int i = 0; i < hInstantaneousMatrices[matrixIndex].nonZeros(); i++) {
        std::cerr<<hInstantaneousMatrices[matrixIndex].valuePtr()[i]<<", ";
    }

#endif

    //TODO: use cusparse function for diagonal sum?
    Real mu_B = 0.0;
    for (int i = 0; i < kStateCount; i++) {
        mu_B += hInstantaneousMatrices[matrixIndex].coeff(i, i);
    }
    mu_B /= (Real) kStateCount;

    hMuBs[matrixIndex] = mu_B;
    hBs[matrixIndex] = hInstantaneousMatrices[matrixIndex] - mu_B * hIdentity;
    hB1Norms[matrixIndex] = normP1(hBs[matrixIndex]);

    const int currentNNZ = hBs[matrixIndex].nonZeros();
    if (currentCacheNNZs[matrixIndex] != currentNNZ) {
        currentCacheNNZs[matrixIndex] = currentNNZ;
        dBsCsrColumnsCache[matrixIndex] = cudaDeviceNew<int>(currentNNZ);
        dBsCsrValuesCache[matrixIndex] = cudaDeviceNew<Real>(currentNNZ);
        for (int category = 0; category < kCategoryCount; category++) {
            dACscValuesCache[matrixIndex * kCategoryCount * 2 + category] = cudaDeviceNew<Real>(currentNNZ);
            dACscValuesCache[matrixIndex * kCategoryCount * 2 + kCategoryCount + category] = cudaDeviceNew<Real>(currentNNZ);
        }
    }

    MemcpyHostToDevice(dBsCsrOffsetsCache[matrixIndex], hBs[matrixIndex].outerIndexPtr(), kPaddedStateCount + 1);
    MemcpyHostToDevice(dBsCsrColumnsCache[matrixIndex], hBs[matrixIndex].innerIndexPtr(), currentNNZ);
    MemcpyHostToDevice(dBsCsrValuesCache[matrixIndex], hBs[matrixIndex].valuePtr(), currentNNZ);

#ifdef BEAGLE_DEBUG_FLOW

    std::cerr<<"\ndBsCsrOffsetsCache for "<< matrixIndex <<"="<<std::endl;
    PrintfDeviceVector(dBsCsrOffsetsCache[matrixIndex], kPaddedStateCount + 1, -1, 0, 0);

    std::cerr<<"dBsCsrColumnsCache for "<< matrixIndex <<"="<<std::endl;
    PrintfDeviceVector(dBsCsrColumnsCache[matrixIndex], currentNNZ, -1, 0, 0);

    std::cerr<<"currentNNZ ="<< currentNNZ <<std::endl;

    std::cerr<<"dBsCsrValuesCache for "<< matrixIndex <<"="<<std::endl;
    PrintfDeviceVector(dBsCsrValuesCache[matrixIndex], currentNNZ, -1, 0, 0);
#endif


    hds[matrixIndex].clear();

    int pMax = getPMax();
    for(int p=0;p <= pMax+1; p++)
    {
        int t = 5;
        Real approx_norm = normest1( hBs[matrixIndex], p, t);

        // equation 3.7 in Al-Mohy and Higham
        hds[matrixIndex].push_back(pow(approx_norm, 1.0 / Real(p) ) );
    }

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
								      const Real* edgeLengths,
								      int count)
{
    for (int i = 0; i < count; i++) {
        const int nodeIndex = probabilityIndices[i];
        hEigenMaps[nodeIndex] = eigenIndex;

        for (int category = 0; category < kCategoryCount; category++) {
            const double categoryRate = hCategoryRates[0][category]; // XJ: because rate categories are only set for first eigen index
            hEdgeMultipliers[nodeIndex * kCategoryCount + category] = edgeLengths[i] * categoryRate;
        }
    }
    // TODO: check if need to copy it from host to device afterwards
    return BEAGLE_SUCCESS;
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
