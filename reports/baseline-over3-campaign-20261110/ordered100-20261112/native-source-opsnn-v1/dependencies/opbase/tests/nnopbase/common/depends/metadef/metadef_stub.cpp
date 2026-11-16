/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/any_value.h"
#include "op_impl_register_v2_impl.h"
#include "register/op_impl_registry.h"
#include "register/op_impl_registry_base.h"
#include "register/op_check.h"
#include "external/graph/ascend_string.h"
#include "debug/ge_log.h"
#include "common/util/mem_utils.h"
#include "register/op_impl_kernel_registry.h"
#include "register/op_impl_registry_holder_manager.h"
#include "register/op_impl_space_registry.h"
#include "register/kernel_registry.h"

#include "register/op_info_record_registry.h"

namespace aclnnOpInfoRecord {
static OpInfoRecordRegister* Instance()
{
    static OpInfoRecordRegister instance;
    return &instance;
}
void OpInfoRecordRegister::ExeOptInfoStat(const gert::TilingContext* ctx, const OpCompilerOption& opt,
                                          const OpKernelInfo* kernelInfo) const
{
    return;
}
} // namespace aclnnOpInfoRecord
