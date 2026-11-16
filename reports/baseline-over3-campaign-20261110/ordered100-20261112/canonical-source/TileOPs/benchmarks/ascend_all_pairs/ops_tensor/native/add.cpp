#define main allpairs_unused_add_main
#include "include/tensor_api/examples/01_simd_cpp_api/07_tensor_api/experimental/reg_vector_compute/add/add.asc"
#undef main

extern "C" int allpairs_opstensor_add(float* a, float* b, float* out, int64_t count, void* stream)
{
    if (!a || !b || !out || count <= 0 || count % total_length) return 1;
    for (int64_t offset = 0; offset < count; offset += total_length)
        add<<<1, 0, stream>>>(a + offset, b + offset, out + offset);
    return 0;
}
