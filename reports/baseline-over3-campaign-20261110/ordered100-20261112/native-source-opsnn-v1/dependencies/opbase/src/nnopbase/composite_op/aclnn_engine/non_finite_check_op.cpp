/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "non_finite_check_op.h"

#include <unordered_map>
#include <string>

namespace op {
namespace internal {

std::unordered_map<std::string, op::DataType> NonFiniteCheckOpKernel::dtypeMap_ = {
    {"float32", op::DataType::DT_FLOAT}, {"float16", op::DataType::DT_FLOAT16}, {"bfloat16", op::DataType::DT_BF16}};

thread_local std::unordered_map<op::DataType, OpKernelBin*> NonFiniteCheckOpKernel::dtype2KernelBin_;

#if defined(NNOPBASE_UT) || defined(NNOPBASE_ST)
aclnnStatus NonFiniteCheckOp::RunNonfiniteCheckOp([[maybe_unused]] NonFiniteCheckOpContext& nonFiniteCheckOpCtx,
                                                  bool& dump)
{
    dump = true;
    return ACLNN_SUCCESS;
}
#endif

} // namespace internal
} // namespace op