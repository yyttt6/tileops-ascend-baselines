// Register the existing exported SGL host entry. No device code is compiled here.
#include <torch/library.h>
#include "sparse_attention_score.h"

TORCH_LIBRARY(classic_sgl_fp8, m) {
  m.def("sparse_attention_score(Tensor query, Tensor key, Tensor value, Tensor select_idx, "
        "Tensor block_table, Tensor? select_num_idx=None, Tensor? q_dequant_scale=None, "
        "Tensor? k_dequant_scale=None, Tensor? v_dequant_scale=None, "
        "Tensor? actual_seq_lengths=None, Tensor? actual_seq_lengths_kv=None, "
        "int num_key_value_heads=1, float scale_value=1.0, int block_size=128, "
        "int top_k=16, int inner_precise=0) -> Tensor");
}

TORCH_LIBRARY_IMPL(classic_sgl_fp8, PrivateUse1, m) {
  m.impl("sparse_attention_score", TORCH_FN(sglang::npu_kernel::sparse_attention_score));
}
