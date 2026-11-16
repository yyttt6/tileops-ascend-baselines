/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bridge_dfx.h"
#include "platform/soc_spec.h"
#include "dump/adump_pub.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

namespace op {
namespace internal {
bool IsDumpEnable() { return Adx::AdumpGetDumpSwitch(Adx::DumpType::OPERATOR); }

bool IsExceptionDumpEnable()
{
#if defined(NNOPBASE_UT) || defined(NNOPBASE_ST)
    bool isExceptionEnable = Adx::AdumpGetDumpSwitch(Adx::DumpType::EXCEPTION);
#else
    static bool isExceptionEnable = Adx::AdumpGetDumpSwitch(Adx::DumpType::EXCEPTION);
#endif
    OP_LOGI("Exception dump = %d", isExceptionEnable);
    return isExceptionEnable;
}

bool IsArgExceptionDumpEnable()
{
    static bool isArgExceptionEnable = Adx::AdumpGetDumpSwitch(Adx::DumpType::ARGS_EXCEPTION);
    OP_LOGI("Arg exception dump = %d", isArgExceptionEnable);
    return isArgExceptionEnable;
}

bool IsOverflowDumpEnable()
{
    static bool isOverflowEnable = Adx::AdumpGetDumpSwitch(Adx::DumpType::OP_OVERFLOW);
    OP_LOGI("Overflow dump = %d", isOverflowEnable);
    return isOverflowEnable;
}

bool IsNeedL0ExceptionDump()
{
    static NpuArch npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    static bool isNeed = ((npuArch == NpuArch::DAV_1001 || npuArch == NpuArch::DAV_3002) && IsArgExceptionDumpEnable());
    OP_LOGI("npu arch: %u, is L0 exception dump need: %d", npuArch, isNeed);
    return isNeed;
}

} // namespace internal
} // namespace op
