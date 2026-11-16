/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_feature_internal.h"

#include <cstdint>
#include <mutex>
#include <vector>

#include "acl/acl_rt.h"

#include "aclnn/acl_meta.h"
#include "opdev/platform.h"
#include "opdev/op_log.h"
#include "opdev/op_errno.h"

namespace op {
namespace internal {

namespace {
// 与 rts 的 aclrtAddrRange 二进制布局一致，弱符号调用可直接用 PcieAddrRange* 接收
struct PcieAddrRange {
    void* startAddr{nullptr};
    void* endAddr{nullptr};
};

bool g_pcieThroughEnabled{false};
std::vector<PcieAddrRange> g_pcieAddrRanges;
std::once_flag g_pcieThroughOnceFlag;
} // namespace

aclnnStatus InitPcieThroughInfo()
{
    static aclnnStatus initRet = ACLNN_SUCCESS;
    std::call_once(g_pcieThroughOnceFlag, [&initRet]() {
        NpuArch npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
        OP_LOGI("NPU arch is %d", static_cast<int32_t>(npuArch));

        int32_t deviceId = 0;
        auto aclRet = aclrtGetDevice(&deviceId);
        if (aclRet != ACL_SUCCESS) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "aclrtGetDevice failed, return %d", static_cast<int32_t>(aclRet));
            initRet = ACLNN_ERR_RUNTIME_ERROR;
            return;
        }

        // 遗留项：HD connect 判断（aclrtGetDeviceInfo 兼容性问题）与地址段获取（接口未提供）暂注释

        // 框架阶段：赋初始值用于性能验证，真实地址段待弱符号接口接入后填充
        // 选用 0x1~0x2 这种不存在的极小地址段，保证真实 tensor 地址不会命中
        g_pcieAddrRanges.resize(1);
        g_pcieAddrRanges[0].startAddr = reinterpret_cast<void*>(0x1);
        g_pcieAddrRanges[0].endAddr = reinterpret_cast<void*>(0x2);
        g_pcieThroughEnabled = true;
    });
    return initRet;
}

bool IsPcieThroughEnabled() { return g_pcieThroughEnabled; }

bool IsTensorAddrInPcieRange(const void* const addr)
{
    uintptr_t addrVal = reinterpret_cast<uintptr_t>(addr);              // NOLINT
    for (const auto& range : g_pcieAddrRanges) {
        uintptr_t start = reinterpret_cast<uintptr_t>(range.startAddr); // NOLINT
        uintptr_t end = reinterpret_cast<uintptr_t>(range.endAddr);     // NOLINT
        if (addrVal >= start && addrVal <= end) {
            OP_LOGI("Tensor addr %p is in PCIe range, startAddr %p, endAddr %p", addr, range.startAddr, range.endAddr);
            return true;
        }
    }
    return false;
}

} // namespace internal
} // namespace op
