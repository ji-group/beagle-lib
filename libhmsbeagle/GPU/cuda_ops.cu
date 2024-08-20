#include "cuda_ops.h"

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/async/for_each.h>

void cuda_log_vector(double* v, int length)
{
    thrust::device_ptr<double> vdptr = thrust::device_pointer_cast<double>(v);

    // In-place update is accomplished by making the output iterator the same the starting input iterator.
    thrust::transform(vdptr, vdptr + length, vdptr, [] __device__ (double x) {return log(x);});
}

void cuda_log_vector(float* v, int length)
{
    thrust::device_ptr<float> vdptr = thrust::device_pointer_cast<float>(v);

    // In-place update is accomplished by making the output iterator the same the starting input iterator.
    thrust::transform(vdptr, vdptr + length, vdptr, [] __device__ (float x) {return log(x);});
}

template <typename REAL>
struct rescalePartialsDeviceOp
{
    REAL* partials;
    REAL* scalingFactors;

    int kPaddedStateCount;
    int kPaddedPatternCount;
    int kCategoryCount;

    __host__ __device__ void operator()(int pattern) const
    {
	// FIND_MAX_PARTIALS_X_CPU();
        int deltaPartialsByState = pattern * kPaddedStateCount;
	auto& max = scalingFactors[pattern];

        // SCALE_PARTIALS_X_CPU();
        for(int m = 0; m < kCategoryCount; m++)
        {
            int deltaPartialsByCategory = m * kPaddedStateCount * kPaddedPatternCount;
            int deltaPartials = deltaPartialsByCategory + deltaPartialsByState;
            for(int i = 0; i < kPaddedStateCount; i++) {
                partials[deltaPartials + i] /= max;
            }
        }
    }

    rescalePartialsDeviceOp(REAL* r1, REAL* r2, int i1, int i2, int i3)
	:partials(r1),
	 scalingFactors(r2),
	 kPaddedStateCount(i1),
	 kPaddedPatternCount(i2),
	 kCategoryCount(i3)
	{
	}

};

// FIXME: It would be nice to merge the code for the <float> and <double> versions of
//        rescalePartialsDevice, but this was somehow causing the program to crash.

void justMaximize(float* partials, float* scalingFactors,
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
	in_values_start,                               // values to reduce (category, pattern, state)
	thrust::make_discard_iterator(),               // key values out
	thrust::device_pointer_cast<float>(scalingFactors), // reduced values out
	thrust::equal_to<int>(),                       // compare keys operation
	thrust::maximum<float>()                      // reduction operation
    );
}

void justMaximize(double* partials, double* scalingFactors,
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
	in_values_start,                               // values to reduce (category, pattern, state)
	thrust::make_discard_iterator(),               // key values out
	thrust::device_pointer_cast<double>(scalingFactors), // reduced values out
	thrust::equal_to<int>(),                       // compare keys operation
	thrust::maximum<double>()                      // reduction operation
    );
}

void  rescalePartials2(bool scalers_log, int kCategoryCount, int kPaddedPatternCount, int kPaddedStateCount,
                       float* partials, float* scalingFactors, float* cumulativeScalingBuffer, int streamIndex)
{
    using namespace thrust::placeholders;

//    thrust::device_ptr<float> partials2 = thrust::device_pointer_cast<float>(partials);
    thrust::device_ptr<float> scalingFactors2 = thrust::device_pointer_cast<float>(scalingFactors);

    // 1. Find maximum partial likelihood -> scalingFactors[pattern]
    justMaximize(partials, scalingFactors, kPaddedStateCount, kPaddedPatternCount, kCategoryCount);

    // 2. Transform scalingfactors[pattern] -> 1 if it equals 0.
    thrust::transform(scalingFactors2, scalingFactors2 + kPaddedPatternCount, scalingFactors2, [] __device__ (float x) { return (x == 0) ? 1.0 : x;});

    // 3. Rescale each pattern by scalingFactors[pattern]
    auto start = thrust::make_counting_iterator<int>(0);
    auto end = start + kPaddedPatternCount;

    thrust::for_each(start, end,
                     rescalePartialsDeviceOp<float>(partials, scalingFactors,
                                                    kPaddedStateCount, kPaddedPatternCount, kCategoryCount));

//    std::cerr<<"scalers_log = "<<scalers_log<<"   cumulativeScalingBuffer = "<<cumulativeScalingBuffer<<"\n";

    // 4. Transform by log if (scalers_log)
    if (scalers_log)
        thrust::transform(scalingFactors2, scalingFactors2 + kPaddedPatternCount, // in
                          scalingFactors2,                                        // out
                          [] __device__ (float x) { return log(x); }             // transformation
                         );

    // 5. Add to cumulativeScalingBuffer
    if (cumulativeScalingBuffer)
    {
        thrust::device_ptr<float> cumulative2 = thrust::device_pointer_cast<float>(cumulativeScalingBuffer);

        if (scalers_log)
        {
            thrust::transform(cumulative2, cumulative2 + kPaddedPatternCount, // in1
                              scalingFactors2,                                // in2
                              cumulative2,                                    // out
                              thrust::plus<float>()                          // operation;
                             );
        }
        else
        {
            auto logScalingFactors2 = thrust::make_transform_iterator( scalingFactors2,
                                                                       [] __host__ __device__ (float x) { return log(x); });

            thrust::transform(cumulative2, cumulative2 + kPaddedPatternCount, // in1
                              logScalingFactors2,                             // in2
                              cumulative2,                                    // out
                              thrust::plus<float>()                          // operation;
                             );
        }
    }
}

void  rescalePartials2(bool scalers_log, int kCategoryCount, int kPaddedPatternCount, int kPaddedStateCount,
                       double* partials, double* scalingFactors, double* cumulativeScalingBuffer, int streamIndex)
{
    using namespace thrust::placeholders;

//    thrust::device_ptr<double> partials2 = thrust::device_pointer_cast<double>(partials);
    thrust::device_ptr<double> scalingFactors2 = thrust::device_pointer_cast<double>(scalingFactors);

    // 1. Find maximum partial likelihood -> scalingFactors[pattern]
    justMaximize(partials, scalingFactors, kPaddedStateCount, kPaddedPatternCount, kCategoryCount);

    // 2. Transform scalingfactors[pattern] -> 1 if it equals 0.
    thrust::transform(scalingFactors2, scalingFactors2 + kPaddedPatternCount, scalingFactors2, [] __device__ (double x) { return (x == 0) ? 1.0 : x;});

    // 3. Rescale each pattern by scalingFactors[pattern]
    auto start = thrust::make_counting_iterator<int>(0);
    auto end = start + kPaddedPatternCount;

    thrust::for_each(start, end,
                     rescalePartialsDeviceOp<double>(partials, scalingFactors,
                                                     kPaddedStateCount, kPaddedPatternCount, kCategoryCount));

//    std::cerr<<"scalers_log = "<<scalers_log<<"   cumulativeScalingBuffer = "<<cumulativeScalingBuffer<<"\n";

    // 4. Transform by log if (scalers_log)
    if (scalers_log)
        thrust::transform(scalingFactors2, scalingFactors2 + kPaddedPatternCount, // in
                          scalingFactors2,                                        // out
                          [] __device__ (double x) { return log(x); }             // transformation
                         );

    // 5. Add to cumulativeScalingBuffer
    if (cumulativeScalingBuffer)
    {
        thrust::device_ptr<double> cumulative2 = thrust::device_pointer_cast<double>(cumulativeScalingBuffer);

        if (scalers_log)
        {
            thrust::transform(cumulative2, cumulative2 + kPaddedPatternCount, // in1
                              scalingFactors2,                                // in2
                              cumulative2,                                    // out
                              thrust::plus<double>()                          // operation;
                             );
        }
        else
        {
            auto logScalingFactors2 = thrust::make_transform_iterator( scalingFactors2,
                                                                       [] __host__ __device__ (double x) { return log(x); });

            thrust::transform(cumulative2, cumulative2 + kPaddedPatternCount, // in1
                              logScalingFactors2,                             // in2
                              cumulative2,                                    // out
                              thrust::plus<double>()                          // operation;
                             );
        }
    }
}

