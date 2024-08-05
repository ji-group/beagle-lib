#include "cuda_ops.h"

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

// QUESTION: Why can't we do the transform in-place?
void cuda_log_vector(double* v, int length)
{
    thrust::device_ptr<double> vdptr = thrust::device_pointer_cast<double>(v);

    // QUESTION: How slow is allocating the device memory here?
    thrust::device_vector<double> V(vdptr, vdptr + length);

    thrust::transform(V.begin(), V.end(), vdptr, [] __device__ (double x) {return log(x);});
}

void cuda_log_vector(float* v, int length)
{
    thrust::device_ptr<float> vdptr = thrust::device_pointer_cast<float>(v);

    // QUESTION: How slow is allocating the device memory here?
    thrust::device_vector<float> V(vdptr, vdptr + length);

    thrust::transform(V.begin(), V.end(), vdptr, [] __device__ (float x) {return log(x);});
}

// FIXME: It would be nice to merge the code for the <float> and <double> versions of
//        rescalePartialsDevice, but this was somehow causing the program to crash.

void rescalePartialsDevice(float* partials, float* scalingFactors, float* cumulativeScalingBuffer,
			   int nStates, int nPatterns, int nCategories)
{
    using namespace thrust::placeholders;

    // 1. Surprisingly, cuBLAS has no operations that reduce (sum,maximize,minimize,etc.) rows or columns.
    //    It can only reduce an entire dense matrix to a single value.

    // 2. thrust::reduce_by_key is able to reduce regions of a vector down to MULTIPLE values.
    //    We assign each value a "key" that decides which group it is in.
    //    Adjacent values with the same key end up in the same group.
    //    However, non-adjacent values with the same key end up in different groups.
    //    thrust::reduce_by_key performs the reduction operation on each group.

    // 3. Thrust can maximize over states (adjacent memory_ and categories (non-adjacent) in one operation
    //    by changing the order in which entries of the input matrix are visited.  All the entries in the
    //    same group need to be visited sequentially.  We can do this using a permutation iterator.

    // Implementation notes:
    // * the library MatX (https://github.com/NVIDIA/MatX) offers much nicer syntax, but thrust is older
    //   and more stable (in 2024)

    // The size of a partials buffer
    size_t partials_size = nStates * nPatterns * nCategories;


    // OK, so we can convert the input index i=[0...partials_size] to (p,c,s) as follows:
    //   s = i % nStates
    //   c = (i / nStates) % nCategories
    //   p = (i / nStates * nCategories)

    // The linear index into the input matrix is:
    //   j = s + nStates *p + (nStates*nPatterns)*c

    // A list of 00000..111111..222222........(P-1)(P-1)(P-1)(P-1) that groups values by state and categories.
    // There should be nPatterns groups.
    auto in_keys_start = thrust::make_transform_iterator(thrust::make_counting_iterator((int) 0), (_1 / (nStates * nCategories)));

    // We need to compute j as a function of i:
    //   j(i) = (i % nStates) + nStates*(i/(nStates * nCategories)) + (nStates*nPatterns)*((i/nStates) % nCategories)
    auto in_values_start = thrust::make_permutation_iterator(
	thrust::device_pointer_cast<float>(partials),
	thrust::make_transform_iterator( thrust::make_counting_iterator((int)0),
					 (_1 % nStates) + nStates*(_1/(nStates*nCategories)) + nStates*nPatterns*((_1/nStates)%nCategories))
	);

    thrust::reduce_by_key(
	// add execution policy thrust::cuda::par_nosync?
	in_keys_start,                                 // key indices start (group by pattern)
	in_keys_start + partials_size,                 // key indices end
	thrust::device_pointer_cast<float>(partials), // values to reduce (category, pattern, state)
	thrust::make_discard_iterator(),               // key values out
	thrust::device_pointer_cast<float>(cumulativeScalingBuffer), // reduced values out
	thrust::equal_to<int>(),                       // compare keys operation
	thrust::maximum<float>()                      // reduction operation
    );
}

void rescalePartialsDevice(double* partials, double* scalingFactors, double* cumulativeScalingBuffer,
			   int nStates, int nPatterns, int nCategories)
{
    using namespace thrust::placeholders;

    // 1. Surprisingly, cuBLAS has no operations that reduce (sum,maximize,minimize,etc.) rows or columns.
    //    It can only reduce an entire dense matrix to a single value.

    // 2. thrust::reduce_by_key is able to reduce regions of a vector down to MULTIPLE values.
    //    We assign each value a "key" that decides which group it is in.
    //    Adjacent values with the same key end up in the same group.
    //    However, non-adjacent values with the same key end up in different groups.
    //    thrust::reduce_by_key performs the reduction operation on each group.

    // 3. Thrust can maximize over states (adjacent memory_ and categories (non-adjacent) in one operation
    //    by changing the order in which entries of the input matrix are visited.  All the entries in the
    //    same group need to be visited sequentially.  We can do this using a permutation iterator.

    // Implementation notes:
    // * the library MatX (https://github.com/NVIDIA/MatX) offers much nicer syntax, but thrust is older
    //   and more stable (in 2024)

    // The size of a partials buffer
    size_t partials_size = nStates * nPatterns * nCategories;


    // OK, so we can convert the input index i=[0...partials_size] to (p,c,s) as follows:
    //   s = i % nStates
    //   c = (i / nStates) % nCategories
    //   p = (i / nStates * nCategories)

    // The linear index into the input matrix is:
    //   j = s + nStates *p + (nStates*nPatterns)*c

    // A list of 00000..111111..222222........(P-1)(P-1)(P-1)(P-1) that groups values by state and categories.
    // There should be nPatterns groups.
    auto in_keys_start = thrust::make_transform_iterator(thrust::make_counting_iterator((int) 0), (_1 / (nStates * nCategories)));

    // We need to compute j as a function of i:
    //   j(i) = (i % nStates) + nStates*(i/(nStates * nCategories)) + (nStates*nPatterns)*((i/nStates) % nCategories)
    auto in_values_start = thrust::make_permutation_iterator(
	thrust::device_pointer_cast<double>(partials),
	thrust::make_transform_iterator( thrust::make_counting_iterator((int)0),
					 (_1 % nStates) + nStates*(_1/(nStates*nCategories)) + nStates*nPatterns*((_1/nStates)%nCategories))
	);

    thrust::reduce_by_key(
	// add execution policy thrust::cuda::par_nosync?
	in_keys_start,                                 // key indices start (group by pattern)
	in_keys_start + partials_size,                 // key indices end
	thrust::device_pointer_cast<double>(partials), // values to reduce (category, pattern, state)
	thrust::make_discard_iterator(),               // key values out
	thrust::device_pointer_cast<double>(cumulativeScalingBuffer), // reduced values out
	thrust::equal_to<int>(),                       // compare keys operation
	thrust::maximum<double>()                      // reduction operation
    );
}

void  rescalePartials2(int kPaddedPatternCount, float* partials, float* scalingFactors, float* cumulativeScalingBuffer, int streamIndex)
{
    auto start_pattern = thrust::make_counting_iterator<int>(0);
    auto rescale_pattern = [] __device__ (int pattern) {};
    thrust::for_each(start_pattern, start_pattern + kPaddedPatternCount, rescale_pattern);
}

void  rescalePartials2(int kPaddedPatternCount, double* partials, double* scalingFactors, double* cumulativeScalingBuffer, int streamIndex)
{
    auto start_pattern = thrust::make_counting_iterator<int>(0);
    auto rescale_pattern = [] __device__ (int pattern) {};
    thrust::for_each(start_pattern, start_pattern + kPaddedPatternCount, rescale_pattern);
}
