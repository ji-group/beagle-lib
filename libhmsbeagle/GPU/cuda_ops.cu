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

    // 3. After maximizing over states, we then need to maximize over categories.

    // Implementation notes:
    // * ideally we could maximize over states (adjacent memory) and categories (non-adjacent) in one operation.
    // * can thrust maximize over states and categories in one operation?
    // * the library MatX (https://github.com/NVIDIA/MatX) offers much nicer syntax, but
    //   thrust is more older and more stable (in 2024)

    // The size of a partials buffer
    size_t partials_size = nStates * nPatterns * nCategories;

    // When we maximize over states, we will get this many values.
    thrust::device_vector<float> max_tmp(nPatterns * nCategories);

    // A list of 00000..111111..222222........00000... that groups values by pattern.
    auto in_keys_start = thrust::make_transform_iterator(thrust::make_counting_iterator((int) 0), (_1 / nStates) % nPatterns);

    // maximize over states for each category.
    thrust::reduce_by_key(
	in_keys_start,                                 // key indices start (group by pattern)
	in_keys_start + partials_size,                 // key indices end
	thrust::device_pointer_cast<float>(partials),  // values to reduce (category, pattern, state)
	thrust::make_discard_iterator(),               // key values out
	&max_tmp[0],                                   // reduced values out
	thrust::equal_to<int>(),                       // compare keys operation
	thrust::maximum<float>()                       // reduction operation
    );

    // Now we have maxima per (category,pattern) in max_tmp.
    // We want to go from (category,pattern) -> pattern.
    // 01234...(P-1)01234..(P-1)01234...(P-1).
    // We need to maximize over the non-adjacent values for each category.
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

    // 3. After maximizing over states, we then need to maximize over categories.

    // Implementation notes:
    // * ideally we could maximize over states (adjacent memory) and categories (non-adjacent) in one operation.
    // * can thrust maximize over states and categories in one operation?
    // * the library MatX (https://github.com/NVIDIA/MatX) offers much nicer syntax, but
    //   thrust is more older and more stable (in 2024)

    // The size of a partials buffer
    size_t partials_size = nStates * nPatterns * nCategories;

    // When we maximize over states, we will get this many values.
    thrust::device_vector<double> max_tmp(nPatterns * nCategories);

    // A list of 00000..111111..222222........00000... that groups values by pattern.
    auto in_keys_start = thrust::make_transform_iterator(thrust::make_counting_iterator((int) 0), (_1 / nStates) % nPatterns);

    // maximize over states for each category.
    thrust::reduce_by_key(
	// add execution policy thrust::cuda::par_nosync?
	in_keys_start,                                 // key indices start (group by pattern)
	in_keys_start + partials_size,                 // key indices end
	thrust::device_pointer_cast<double>(partials), // values to reduce (category, pattern, state)
	thrust::make_discard_iterator(),               // key values out
	&max_tmp[0],                                   // reduced values out
	thrust::equal_to<int>(),                       // compare keys operation
	thrust::maximum<double>()                      // reduction operation
    );

    // Now we have maxima per (category,pattern) in max_tmp.
    // We want to go from (category,pattern) -> pattern.
    // 01234...(P-1)01234..(P-1)01234...(P-1).
    // We need to maximize over the non-adjacent values for each category.

    // Actually, we can probably do this in one operation without allocating a temporary.
    // We need to construct a permutation iterator that reorders the input values so that all entries
    //   for the same category are adjacent.
    // Then we can order the keys so that we have groups of size nStates * nCategories.
    // auto in_keys_start = thrust::make_transform_iterator(thrust::make_counting_iterator((int) 0), (_1 / (nStates*nCategories)) );
}

