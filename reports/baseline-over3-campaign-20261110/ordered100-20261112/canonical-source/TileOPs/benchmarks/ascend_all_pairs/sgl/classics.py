"""Lazy wrappers around already reviewed representative routes."""
def add(case):
    from benchmarks.ascend_add_cases import sgl_add_cases
    return sgl_add_cases(case)
def amax(case):
    from benchmarks.ascend_reduction_cases import sgl_amax_cases
    return sgl_amax_cases(case)
def rms(case):
    from benchmarks.ascend_sgl_large_rms import sgl_large_rms_cases
    return sgl_large_rms_cases(case)
def gemm(case):
    from benchmarks.ascend_sgl_gemv_cases import sgl_gemv_cases
    return sgl_gemv_cases(case)
def conv2d(case):
    from benchmarks.ascend_bias_coordinate_conv import sgl_bias_coordinate_cases
    return sgl_bias_coordinate_cases(case)
def rope(case):
    from benchmarks.ascend_rope_cases import sgl_rope_cases
    return sgl_rope_cases(case)
def delta(case):
    from benchmarks.ascend_sgl_delta_cases import sgl_delta_cases
    return sgl_delta_cases(case)
def gqa(case):
    # The FP8 route uses an old build directory outside our isolated workspace;
    # it is numerically quarantined and deliberately omitted from this registry.
    from benchmarks.ascend_baselines import UnsupportedCase
    if str(case.inputs[0].dtype).endswith('float8_e4m3fn'):
        raise UnsupportedCase('Original full-FP8 path has a preserved numerical quarantine; use coverage evidence.')
    from benchmarks.ascend_rope_attention_cases import sgl_rope_gqa_cases
    return sgl_rope_gqa_cases(case)
