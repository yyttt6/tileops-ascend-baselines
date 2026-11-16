// Deployment metadata only. The original ACLNN frontend allocates y as
// [x.dim(0), topK] FP32 and indices as the same shape INT32 (moe_fused_topk.cpp).
// The original graph package lacks this inference registration. No numerical
// kernel, tiling, schedule, input tensor, or output tensor data is changed here.
#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus TransformerTopKShape(gert::InferShapeContext* ctx) {
  const auto* x = ctx->GetInputShape(0);
  const auto* attrs = ctx->GetAttrs();
  auto* y = ctx->GetOutputShape(0);
  auto* ids = ctx->GetOutputShape(1);
  if (!x || !attrs || !y || !ids || x->GetDimNum() != 2) return ge::GRAPH_FAILED;
  const auto* k = attrs->GetAttrPointer<int64_t>(3);
  if (!k || *k <= 0) return ge::GRAPH_FAILED;
  *y = *x;
  y->SetDim(1, *k);
  *ids = *y;
  return ge::GRAPH_SUCCESS;
}
static ge::graphStatus TransformerTopKDtype(gert::InferDataTypeContext* ctx) {
  if (!ctx) return ge::GRAPH_FAILED;
  ctx->SetOutputDataType(0, ge::DT_FLOAT);
  ctx->SetOutputDataType(1, ge::DT_INT32);
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(MoeFusedTopk).InferShape(TransformerTopKShape).InferDataType(TransformerTopKDtype);
}  // namespace ops
