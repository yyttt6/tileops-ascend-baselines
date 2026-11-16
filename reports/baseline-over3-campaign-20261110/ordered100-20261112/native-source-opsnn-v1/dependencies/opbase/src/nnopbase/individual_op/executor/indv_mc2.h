/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INDV_MC2_H_
#define INDV_MC2_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "acl/acl_rt.h"
#include "hccl/hccl_types.h"
#include "individual_op_api.h"
#include "runtime/runtime/kernel.h"
#include "utils/indv_types.h"

static constexpr size_t NNOPBASE_AICPU_PARAM_LEN = 32U;
static const std::string NNOPBASE_MC2_AICPU_SUFFIX = "Mc2AicpuKernel";
static constexpr NnopbaseUChar NNOPBASE_MC2_AICPU_SO_NAME[NNOPBASE_AICPU_PARAM_LEN] = {"libccl_kernel.so"};
static constexpr NnopbaseUChar NNOPBASE_MC2_AICPU_KERNEL_NAME[NNOPBASE_AICPU_PARAM_LEN] = {"RunAicpuKfcSrvLaunch"};
static constexpr NnopbaseUChar NNOPBASE_MC2_SERVER_SO_NAME[NNOPBASE_AICPU_PARAM_LEN] = {"libmc2_server.so"};
static constexpr NnopbaseUChar NNOPBASE_MC2_SERVER_KERNEL_NAME[NNOPBASE_AICPU_PARAM_LEN] = {"Mc2ServerKernel"};
static constexpr uint8_t NNOPBASE_MC2_NOTIFY_COUNT = 2U;
static constexpr uint16_t NNOPBASE_HCCL_DEFAULT_TIME = 1836U;
static constexpr uint32_t NNOPBASE_HCCL_ALG_MAX_NUM = 8U;
static constexpr uint32_t NNOPBASE_CCU_INVALID_OP_TYPE = static_cast<uint32_t>(HCCL_CMD_INVALID);

struct NnopbaseMc2Execution {
    bool enabled = false;
    bool fallback = false;
    NnopbaseHcclServerType serverType = NNOPBASE_HCCL_SERVER_TYPE_END;
    std::vector<HcclComm> commHandles;
    std::vector<void*> contextAddrs;
    rtAicpuArgsEx_t aicpuArgs{};
    rtFusionArgsEx_t fusionArgs{};
    std::vector<aclrtStream> aicpuStreams;
    std::vector<uint64_t> aicpuThreads;
    std::vector<std::pair<aclrtStream, aclrtStream>> aicpuNotifies;

    void ClearRuntimeState()
    {
        enabled = false;
        fallback = false;
        serverType = NNOPBASE_HCCL_SERVER_TYPE_END;
        commHandles.clear();
        contextAddrs.clear();
        aicpuStreams.clear();
        aicpuThreads.clear();
        aicpuNotifies.clear();
    }

    void Reset()
    {
        ClearRuntimeState();
        aicpuArgs = {};
        fusionArgs = {};
    }
};

struct NnopbaseCcuAlgInfo {
    uint64_t offset;
    uint64_t opParam;
};

struct NnopbaseCcuOpResCtx {
    uint64_t version;
    uint64_t workSpace;
    uint64_t workSpaceSize;
    uint64_t rankId;
    uint64_t rankSize;
    NnopbaseCcuAlgInfo algInfo[NNOPBASE_HCCL_ALG_MAX_NUM];
    uint64_t xnAddr;
    uint64_t ckeAddr;
    uint64_t sprAddr;
    uint64_t res[NNOPBASE_HCCL_ALG_MAX_NUM];
    uint64_t resCtx;
    uint32_t opType[NNOPBASE_HCCL_ALG_MAX_NUM];
    uint32_t algorithmType[NNOPBASE_HCCL_ALG_MAX_NUM];
    bool isKfc[NNOPBASE_HCCL_ALG_MAX_NUM];
};

#endif // INDV_MC2_H_
