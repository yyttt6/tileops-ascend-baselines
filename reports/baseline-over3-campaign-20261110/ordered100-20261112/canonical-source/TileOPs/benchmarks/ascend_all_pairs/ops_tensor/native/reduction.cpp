#define main allpairs_unused_reduction_main
#include "include/tensor_api/examples/01_simd_cpp_api/07_tensor_api/experimental/reg_vector_compute/reduction_compute/reduction_compute.asc"
#undef main

extern "C" int allpairs_opstensor_reduction(float* x, float* out, int64_t rows, void* stream)
{
    if (!x || !out || rows <= 0) return 1;
    for (int64_t row = 0; row < rows; ++row)
        reduction_compute<<<1, 0, stream>>>(x + row * input_length, out + row * output_length);
    return 0;
}
