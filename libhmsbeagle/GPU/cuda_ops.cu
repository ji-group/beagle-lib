#include "cuda_ops.h"
#include <thrust/device_vector.h>

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

