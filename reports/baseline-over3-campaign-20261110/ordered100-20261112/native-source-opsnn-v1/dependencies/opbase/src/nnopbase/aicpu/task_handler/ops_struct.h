/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_COMMON_INC_OPDEV_AICPU_OPS_STRUCT_H_
#define OP_API_COMMON_INC_OPDEV_AICPU_OPS_STRUCT_H_

#include <string>
#include <vector>

namespace op {
namespace internal {
struct OpFullInfo {
    std::string opKernelLib;  // which opsKernelStore
    std::string kernelSo;     // kernel so
    std::string functionName; // function name
    bool userDefined;         // user defined
};

struct OpInfoDesc {
    std::string opType; // op Type
    OpFullInfo opInfo;  // config information that the op
};

struct OpInfoDescs {
    std::vector<OpInfoDesc> opInfos; // op info
};
} // namespace internal
} // namespace op
#endif