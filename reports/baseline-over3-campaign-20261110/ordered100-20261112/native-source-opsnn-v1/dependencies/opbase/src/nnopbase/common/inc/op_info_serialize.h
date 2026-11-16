/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __OP_INFO_SERIALIZE_H__
#define __OP_INFO_SERIALIZE_H__
#include <string>
#include "exe_graph/runtime/tiling_context.h"

namespace aclnnOpInfoRecord {
struct OpCompilerOption {
    explicit OpCompilerOption(const std::string& impl_mode_v, uint8_t deterministic_v = true)
        : impl_mode(impl_mode_v), deterministic(deterministic_v)
    {}
    explicit OpCompilerOption(const char* impl_mode_v, uint8_t deterministic_v = true)
        : impl_mode(impl_mode_v), deterministic(deterministic_v)
    {}
    std::string impl_mode;
    uint8_t deterministic;
};

struct OpKernelInfo {
    explicit OpKernelInfo(const std::string& bin_info_v, int8_t bin_type_v)
        : bin_info(bin_info_v), bin_type(bin_type_v), isMc2(false)
    {}
    explicit OpKernelInfo(const char* bin_info_v, int8_t bin_type_v)
        : bin_info(bin_info_v), bin_type(bin_type_v), isMc2(false)
    {}
    std::string bin_info;
    int8_t bin_type;
    bool isMc2;
};

/*
 * @brief: Extract Op Info from TilingContext and complete serialization.
 * @param [in] ctx: TilingContext
 * @return int32_t: 0 == SUCCESS, others is FAILED
 */
int32_t OpInfoSerialize(const gert::TilingContext* ctx, const aclnnOpInfoRecord::OpCompilerOption& opt,
                        const aclnnOpInfoRecord::OpKernelInfo* kernelInfo = nullptr);

int32_t OpInfoDump(void);

} // namespace aclnnOpInfoRecord

#endif