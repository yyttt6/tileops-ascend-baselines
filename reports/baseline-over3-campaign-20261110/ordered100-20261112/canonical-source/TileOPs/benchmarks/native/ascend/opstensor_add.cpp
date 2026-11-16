// Include the complete original kernel. Only replace the demo main symbol.
#define main classic_original_add_demo_main
#include "/root/dyq/workspace/ascend_baselines/ops-tensor/include/tensor_api/examples/01_simd_cpp_api/07_tensor_api/experimental/reg_vector_compute/add/add.asc"
#undef main

extern "C" int classic_opstensor_add(float* a, float* b, float* out,
                                    int64_t count, void* stream) {
    if (count <= 0 || count % total_length != 0) return 1;
    // Host-side batching of the original64-element invocation. The kernel,
    // its one-block launch, synchronization and vector schedule are unchanged.
    for (int64_t offset = 0; offset < count; offset += total_length) {
        add<<<1, 0, stream>>>(a + offset, b + offset, out + offset);
    }
    return 0;
}
