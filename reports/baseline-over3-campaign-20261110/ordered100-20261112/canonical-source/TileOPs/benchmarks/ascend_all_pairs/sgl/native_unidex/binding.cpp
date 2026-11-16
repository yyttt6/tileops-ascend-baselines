#include <ATen/ATen.h>
#include <torch/library.h>

namespace sglang { namespace npu_kernel {
void unidex_copy(const at::Tensor &, at::Tensor &, const at::Tensor &,
                 const at::Tensor &, const at::Tensor &, int64_t, int64_t,
                 int64_t, int64_t, int64_t, c10::optional<int64_t>,
                 c10::optional<int64_t>);
}}

TORCH_LIBRARY(allpairs_sgl_unidex, m) {
    m.def("copy(Tensor src, Tensor(a!) dst, Tensor src_index, Tensor dst_index, "
          "Tensor valid_mask, int src_rows, int dst_rows, int block_bytes, "
          "int max_copy, int block_dim=8, int? src_ptr=None, int? dst_ptr=None) -> ()");
}
TORCH_LIBRARY_IMPL(allpairs_sgl_unidex, PrivateUse1, m) {
    m.impl("copy", TORCH_FN(sglang::npu_kernel::unidex_copy));
}
