#include "cuda_ops.h"
#include <thrust/device_vector.h>

void cuda_log_vector(double* v, int length)
{
    thrust::device_ptr<double> vdptr = thrust::device_pointer_cast<double>(v);

    thrust::device_vector<double> V(vdptr, vdptr + length);

    thrust::device_vector<double> Vout = V;

    thrust::transform(V.begin(), V.end(), Vout.begin(), [] __device__ (double x) {return log(x);});

    double* Vout_ptr = thrust::raw_pointer_cast(&Vout[0]);
    
    MemcpyDeviceToDevice<double>(v, Vout_ptr, length);
}

void cuda_log_vector(float* v, int length)
{
    thrust::device_ptr<float> vdptr = thrust::device_pointer_cast<float>(v);

    thrust::device_vector<float> V(vdptr, vdptr + length);

    thrust::device_vector<float> Vout = V;

    thrust::transform(V.begin(), V.end(), Vout.begin(), [] __device__ (float x) {return log(x);});

    float* Vout_ptr = thrust::raw_pointer_cast(&Vout[0]);
    
    MemcpyDeviceToDevice<float>(v, Vout_ptr, length);
}

