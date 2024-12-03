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
#include <thrust/random.h>

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

void cuda_sign_vector(double* v, int n, int t)
{
    thrust::device_ptr<double> vdptr = thrust::device_pointer_cast<double>(v);

    // In-place update is accomplished by making the output iterator the same the starting input iterator.
    thrust::transform(vdptr, vdptr + n*t, vdptr, [n] __device__ (double x) -> double {return (x<0)?-1.0/n:1.0/n;});
}

void cuda_sign_vector(float* v, int n, int t)
{
    thrust::device_ptr<float> vdptr = thrust::device_pointer_cast<float>(v);

    // In-place update is accomplished by making the output iterator the same the starting input iterator.
    thrust::transform(vdptr, vdptr + n*t, vdptr, [n] __device__ (float x) ->float {return (x<0)?-1.0/n:1.0/n;});
}

double cuda_max_abs(double* values, int length)
{
    thrust::device_ptr<double> values_ptr = thrust::device_pointer_cast<double>(values);

    auto in_ptr = thrust::transform_iterator(values_ptr, [] __host__ __device__ (double x) {return std::abs(x);});

    // Don't use `*(thrust::max_element(....))` -- it is slower than thrust::reduce.

    // thrust::reduce is similar in speed to cublasI{s,d}amax.
    return thrust::reduce(in_ptr, in_ptr + length, 0.0, thrust::maximum<double>());
}

// This seems to be slow because it inserts cudaStreamSynchronize()
float cuda_max_abs(float* values, int length)
{
    thrust::device_ptr<float> values_ptr = thrust::device_pointer_cast<float>(values);

    auto in_ptr = thrust::transform_iterator(values_ptr, [] __host__ __device__ (float x) {return std::abs(x);});

    // Don't use `*(thrust::max_element(....))` -- it is slower than thrust::reduce.

    // thrust::reduce is similar in speed to cublasI{s,d}amax.
    return thrust::reduce(in_ptr, in_ptr + length, 0.0, thrust::maximum<float>());
}

float cuda_max_l1_norm(float* values, int n, int t, float* buffer_)
{
    using namespace thrust::placeholders;


    // 1. First sum the absolute values in each column and place the results into buffer
    auto in_keys_start = thrust::make_transform_iterator(thrust::make_counting_iterator((int)0), (_1 / n));

    auto in_values_start = thrust::transform_iterator(thrust::device_pointer_cast(values),
                                                      [] __host__ __device__ (float x) {return std::abs(x);} );

    auto buffer = thrust::device_pointer_cast(buffer_);

    thrust::reduce_by_key(in_keys_start, in_keys_start + n*t,    // key indices (group by column)
                          in_values_start,                       // values to reduce (with abs applied)
                          thrust::make_discard_iterator(),       // key values out
                          buffer,                                // reduced values out
                          thrust::equal_to<int>()                // compare-keys operation
        );                                                       // summation is the default operation.

    // 2. Second maximize over the column sums and return the highest.
    return thrust::reduce(buffer, buffer+t, 0.0, thrust::maximum<float>());
}

double cuda_max_l1_norm(double* values, int n, int t, double* buffer_)
{
    using namespace thrust::placeholders;


    // 1. First sum the absolute values in each column and place the results into buffer
    auto in_keys_start = thrust::make_transform_iterator(thrust::make_counting_iterator((int)0), (_1 / n));

    auto in_values_start = thrust::transform_iterator(thrust::device_pointer_cast(values),
                                                      [] __host__ __device__ (double x) {return std::abs(x);} );

    auto buffer = thrust::device_pointer_cast(buffer_);

    thrust::reduce_by_key(in_keys_start, in_keys_start + n*t,    // key indices (group by column)
                          in_values_start,                       // values to reduce (with abs applied)
                          thrust::make_discard_iterator(),       // key values out
                          buffer,                                // reduced values out
                          thrust::equal_to<int>()                // compare-keys operation
        );                                                       // summation is the default operation.

    // 2. Second maximize over the column sums and return the highest.
    return thrust::reduce(buffer, buffer+t, 0.0, thrust::maximum<double>());
}

void cuda_rowwise_max_abs(float* values_ptr, int n, int t, float* out_ptr)
{
    using namespace thrust::placeholders;
    //    We assume that the matrix has dimensions (n,t) and is column-major.

    // 1. First sum the absolute values in each column and place the results into buffer
    //    Using _1 % n here should group by row, yielding 0 1 2 3 ...(n-1) 0 1 2 3 ... (n-1) ...
    auto in_keys_start = thrust::make_transform_iterator(thrust::make_counting_iterator((int)0), (_1 % n));

    auto in_values_start = thrust::transform_iterator(thrust::device_pointer_cast(values_ptr),
                                                      [] __host__ __device__ (float x) {return std::abs(x);} );

    auto out = thrust::device_pointer_cast(out_ptr);

    thrust::reduce_by_key(in_keys_start, in_keys_start + n*t,    // key indices (group by column)
                          in_values_start,                       // values to reduce (with abs applied)
                          thrust::make_discard_iterator(),       // key values out
                          out,                                   // reduced values out
                          thrust::equal_to<int>(),               // compare-keys operation
                          thrust::maximum<float>()                  // summation is the default operation.
        );
}

void cuda_rowwise_max_abs(double* values_ptr, int n, int t, double* out_ptr)
{
    using namespace thrust::placeholders;
    //    We assume that the matrix has dimensions (n,t) and is column-major.

    // 1. First sum the absolute values in each column and place the results into buffer
    //    Using _1 % n here should group by row, yielding 0 1 2 3 ...(n-1) 0 1 2 3 ... (n-1) ...
    auto in_keys_start = thrust::make_transform_iterator(thrust::make_counting_iterator((int)0), (_1 % n));

    auto in_values_start = thrust::transform_iterator(thrust::device_pointer_cast(values_ptr),
                                                      [] __host__ __device__ (double x) {return std::abs(x);} );

    auto out = thrust::device_pointer_cast(out_ptr);

    thrust::reduce_by_key(in_keys_start, in_keys_start + n*t,    // key indices (group by column)
                          in_values_start,                       // values to reduce (with abs applied)
                          thrust::make_discard_iterator(),       // key values out
                          out,                                   // reduced values out
                          thrust::equal_to<int>(),               // compare-keys operation
                          thrust::maximum<double>()                  // summation is the default operation.
        );
}

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

template <typename T>
struct Prod3
{
    __host__ __device__ T operator()(thrust::tuple<T,T,T> t)
    {
	return thrust::get<0>(t) * thrust::get<1>(t) * thrust::get<2>(t);
    }
};

void sumRootLikelihoods(float* siteProbs, // OUT
			float* likelihoods, float* weights, float* frequencies, // INT
			int nStates, int nPatterns, int nCategories)
{
    using namespace thrust::placeholders;

    // The size of a partials buffer
    size_t partials_size = nStates * nPatterns * nCategories;


    // OK, so we can convert the input index i=[0...partials_size] to (p,c,s) as follows:
    //   s = i % nStates
    //   c = (i / nStates) % nCategories
    //   p = (i / nStates * nCategories)

    // 1. A list of 00000..111111..222222........(P-1)(P-1)(P-1)(P-1) that groups values by state and categories.
    // There should be nPatterns groups.
    auto in_keys_start = thrust::make_transform_iterator(thrust::make_counting_iterator((int) 0), (_1 / (nStates * nCategories)));

    // 2. The linear index into the likelihood matrix is:
    //   j = s + nStates *p + (nStates*nPatterns)*c
    //
    // We need to compute j as a function of i:
    //   j(i) = (i % nStates) + nStates*(i/(nStates * nCategories)) + (nStates*nPatterns)*((i/nStates) % nCategories)
    auto in_lks_start = thrust::make_permutation_iterator(
	thrust::device_pointer_cast<float>(likelihoods),
	thrust::make_transform_iterator( thrust::make_counting_iterator((int)0),
					 (_1 % nStates) + nStates*(_1/(nStates*nCategories)) + nStates*nPatterns*((_1/nStates)%nCategories))
	);

    // 3. The linear index into the weights matrix is just:
    //   k = c
    // Therefore
    //   k(i) = (i / nStates) % nCategories

    auto in_weights_start =  thrust::make_permutation_iterator(
	thrust::device_pointer_cast<float>(weights),
	thrust::make_transform_iterator( thrust::make_counting_iterator((int)0),
					 (_1/nStates)%nCategories)
	);

    // 4. The linear index into the states matrix is:
    //   l = s
    // Therefore
    //   l(i) = (i % nStates)

    auto in_frequencies_start =  thrust::make_permutation_iterator(
	thrust::device_pointer_cast<float>(frequencies),
	thrust::make_transform_iterator( thrust::make_counting_iterator((int)0),
					 _1 % nStates)
	);


    // 5. Make the sequence of likelihood * categoryWeight * stateFrequency
    auto tuples = thrust::make_zip_iterator(thrust::make_tuple(in_lks_start, in_weights_start, in_frequencies_start));

    auto in_values_start = thrust::make_transform_iterator(tuples, Prod3<float>());

    // 6. Sum over state and category for each pattern.
    thrust::reduce_by_key(
	// add execution policy thrust::cuda::par_nosync?
	in_keys_start,                                  // key indices start (group by pattern)
	in_keys_start + partials_size,                  // key indices end
	in_values_start,                                // values to reduce (category, pattern, state)
	thrust::make_discard_iterator(),                // key values out
	thrust::device_pointer_cast<float>(siteProbs), // reduced values out
	thrust::equal_to<int>()                         // compare keys operation
	                                                // Default operation is (+)
    );
}


void sumRootLikelihoods(double* siteProbs, // OUT
			double* likelihoods, double* weights, double* frequencies, // INT
			int nStates, int nPatterns, int nCategories)
{
    using namespace thrust::placeholders;

    // The size of a partials buffer
    size_t partials_size = nStates * nPatterns * nCategories;


    // OK, so we can convert the input index i=[0...partials_size] to (p,c,s) as follows:
    //   s = i % nStates
    //   c = (i / nStates) % nCategories
    //   p = (i / nStates * nCategories)

    // 1. A list of 00000..111111..222222........(P-1)(P-1)(P-1)(P-1) that groups values by state and categories.
    // There should be nPatterns groups.
    auto in_keys_start = thrust::make_transform_iterator(thrust::make_counting_iterator((int) 0), (_1 / (nStates * nCategories)));

    // 2. The linear index into the likelihood matrix is:
    //   j = s + nStates *p + (nStates*nPatterns)*c
    //
    // We need to compute j as a function of i:
    //   j(i) = (i % nStates) + nStates*(i/(nStates * nCategories)) + (nStates*nPatterns)*((i/nStates) % nCategories)
    auto in_lks_start = thrust::make_permutation_iterator(
	thrust::device_pointer_cast<double>(likelihoods),
	thrust::make_transform_iterator( thrust::make_counting_iterator((int)0),
					 (_1 % nStates) + nStates*(_1/(nStates*nCategories)) + nStates*nPatterns*((_1/nStates)%nCategories))
	);

    // 3. The linear index into the weights matrix is just:
    //   k = c
    // Therefore
    //   k(i) = (i / nStates) % nCategories

    auto in_weights_start =  thrust::make_permutation_iterator(
	thrust::device_pointer_cast<double>(weights),
	thrust::make_transform_iterator( thrust::make_counting_iterator((int)0),
					 (_1/nStates)%nCategories)
	);

    // 4. The linear index into the states matrix is:
    //   l = s
    // Therefore
    //   l(i) = (i % nStates)

    auto in_frequencies_start =  thrust::make_permutation_iterator(
	thrust::device_pointer_cast<double>(frequencies),
	thrust::make_transform_iterator( thrust::make_counting_iterator((int)0),
					 _1 % nStates)
	);


    // 5. Make the sequence of likelihood * categoryWeight * stateFrequency
    auto tuples = thrust::make_zip_iterator(thrust::make_tuple(in_lks_start, in_weights_start, in_frequencies_start));

    auto in_values_start = thrust::make_transform_iterator(tuples, Prod3<double>());

    // 6. Sum over state and category for each pattern.
    thrust::reduce_by_key(
	// add execution policy thrust::cuda::par_nosync?
	in_keys_start,                                  // key indices start (group by pattern)
	in_keys_start + partials_size,                  // key indices end
	in_values_start,                                // values to reduce (category, pattern, state)
	thrust::make_discard_iterator(),                // key values out
	thrust::device_pointer_cast<double>(siteProbs), // reduced values out
	thrust::equal_to<int>()                         // compare keys operation
	                                                // Default operation is (+)
    );
}


void  rescalePartials2(bool scalers_log, int kCategoryCount, int kPaddedPatternCount, int kPaddedStateCount,
                       float* partials, float* scalingFactors, float* cumulativeScalingBuffer, int streamIndex)
{
    using namespace thrust::placeholders;

    size_t partialsSize = kCategoryCount * kPaddedPatternCount * kPaddedStateCount;
    thrust::device_ptr<float> partials2 = thrust::device_pointer_cast<float>(partials);
    thrust::device_ptr<float> scalingFactors2 = thrust::device_pointer_cast<float>(scalingFactors);

    // 1. Find maximize partial likelihood -> scalingFactors[pattern]
    justMaximize(partials, scalingFactors, kPaddedStateCount, kPaddedPatternCount, kCategoryCount);

    // 2. Transform scalingfactors[pattern] -> 1 if it equals 0.
    thrust::transform(scalingFactors2, scalingFactors2 + kPaddedPatternCount, scalingFactors2, [] __device__ (float x) { return (x == 0) ? 1.0 : x;});

    // 3. Rescale each pattern by scalingFactors[pattern(index)]

    // pattern =  (i/kPaddedStateCount) % kPaddedPatternCount

    // iter_max computes the scaling factor as a function of the index into the partials buffer.
    auto iter_max = thrust::make_permutation_iterator(
	                scalingFactors2,
		        thrust::make_transform_iterator(
			    thrust::make_counting_iterator<int>(0),
			    (_1/kPaddedStateCount) % kPaddedPatternCount
			)
	            );


    thrust::transform(partials2, partials2 + partialsSize, // in1 = partials[i]
		      iter_max,                            // in2 = scalingFactors[pattern(i)]
		      partials2,                           // out
		      thrust::divides<float>()             // operation
	             );

//    std::cerr<<"scalers_log = "<<scalers_log<<"   cumulativeScalingBuffer = "<<cumulativeScalingBuffer<<"\n";

    // 4. Transform by log if (scalers_log)
    if (scalers_log)
        thrust::transform(scalingFactors2, scalingFactors2 + kPaddedPatternCount, // in
                          scalingFactors2,                                        // out
                          [] __device__ (float x) { return log(x); }              // transformation
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
                              thrust::plus<float>()                           // operation;
                             );
        }
        else
        {
            auto logScalingFactors2 = thrust::make_transform_iterator( scalingFactors2,
                                                                       [] __host__ __device__ (float x) { return log(x); });

            thrust::transform(cumulative2, cumulative2 + kPaddedPatternCount, // in1
                              logScalingFactors2,                             // in2
                              cumulative2,                                    // out
                              thrust::plus<float>()                           // operation;
                             );
        }
    }
}

void  rescalePartials2(bool scalers_log, int kCategoryCount, int kPaddedPatternCount, int kPaddedStateCount,
                       double* partials, double* scalingFactors, double* cumulativeScalingBuffer, int streamIndex)
{
    using namespace thrust::placeholders;

    size_t partialsSize = kCategoryCount * kPaddedPatternCount * kPaddedStateCount;
    thrust::device_ptr<double> partials2 = thrust::device_pointer_cast<double>(partials);
    thrust::device_ptr<double> scalingFactors2 = thrust::device_pointer_cast<double>(scalingFactors);

    // 1. Find maximize partial likelihood -> scalingFactors[pattern]
    justMaximize(partials, scalingFactors, kPaddedStateCount, kPaddedPatternCount, kCategoryCount);

    // 2. Transform scalingfactors[pattern] -> 1 if it equals 0.
    thrust::transform(scalingFactors2, scalingFactors2 + kPaddedPatternCount, scalingFactors2, [] __device__ (double x) { return (x == 0) ? 1.0 : x;});

    // 3. Rescale each pattern by scalingFactors[pattern(index)]

    // pattern =  (i/kPaddedStateCount) % kPaddedPatternCount

    // iter_max computes the scaling factor as a function of the index into the partials buffer.
    auto iter_max = thrust::make_permutation_iterator(
	                scalingFactors2,
		        thrust::make_transform_iterator(
			    thrust::make_counting_iterator<int>(0),
			    (_1/kPaddedStateCount) % kPaddedPatternCount
			)
	            );


    thrust::transform(partials2, partials2 + partialsSize, // in1 = partials[i]
		      iter_max,                            // in2 = scalingFactors[pattern(i)]
		      partials2,                           // out
		      thrust::divides<double>()            // operation
	             );

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

void initialize_norm_x_matrix(float* data, int n, int m)
{
    auto out_ptr = thrust::device_pointer_cast(data);
    auto indices = thrust::counting_iterator<unsigned int>(0);
    auto initialize = [n,m] __host__ __device__ (int i)
    {
        thrust::default_random_engine rng;
        thrust::uniform_real_distribution<float> dist(0, 1);

        if (i<n) return 1.0/n;
        rng.discard(i);
        if (dist(rng) > 0.5)
            return 1.0/n;
        else
            return -1.0/n;
    };

    thrust::transform(indices, indices + n*m, // in
                      out_ptr,                // out
                      initialize);
}

void initialize_norm_x_matrix(double* data, int n, int m)
{
    auto out_ptr = thrust::device_pointer_cast(data);
    auto indices = thrust::counting_iterator<unsigned int>(0);
    auto initialize = [n,m] __host__ __device__ (int i)
    {
        thrust::default_random_engine rng;
        thrust::uniform_real_distribution<float> dist(0, 1);

        if (i<n) return 1.0/n;
        rng.discard(i);
        if (dist(rng) > 0.5)
            return 1.0/n;
        else
            return -1.0/n;
    };

    thrust::transform(indices, indices + n*m, // in
                      out_ptr,                // out
                      initialize);
}
