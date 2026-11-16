#define main allpairs_unused_trunc_main
#include "include/tensor_api/examples/01_simd_cpp_api/07_tensor_api/experimental/reg_vector_compute/trunc/trunc.asc"
#undef main

extern "C" int allpairs_opstensor_trunc(float* x, float* out, int64_t count, void* stream)
{
    if (!x || !out || count <= 0 || count % total_length) return 1;
    for (int64_t offset = 0; offset < count; offset += total_length)
        trunc<float><<<1, 0, stream>>>(reinterpret_cast<uint8_t*>(x + offset),
                                      reinterpret_cast<uint8_t*>(out + offset), row, col);
    return 0;
}
