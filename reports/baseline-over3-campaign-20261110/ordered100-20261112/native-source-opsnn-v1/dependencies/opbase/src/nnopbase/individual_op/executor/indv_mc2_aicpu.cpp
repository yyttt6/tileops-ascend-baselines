/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "indv_mc2_aicpu.h"
#include "utils/thread_var_container.h"
#include "runtime/runtime/rt_model.h"
#include "utils/indv_soc.h"
#include <cstddef>

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

namespace {
size_t NnopbaseAlignToEightBytes(const size_t size)
{
    return ((size + NNOPBASE_SEVENS_BYTES) / NNOPBASE_EIGHT_BYTES) * NNOPBASE_EIGHT_BYTES;
}

aclnnStatus NnopbaseCheckMC2ParamBuffer(const NnopbaseExecutor* const executor,
                                        const NnopbaseExecutorArgsAddr* const argsAddr, const size_t requiredSize)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor->args);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(argsAddr);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(argsAddr->hostInputData);

    const uintptr_t argsBufStart = reinterpret_cast<uintptr_t>(executor->args->argsBuf.data());
    const uintptr_t argsBufEnd = argsBufStart + executor->args->argsBuf.size();
    const uintptr_t hostInputData = reinterpret_cast<uintptr_t>(argsAddr->hostInputData);
    CHECK_COND((hostInputData >= argsBufStart) && (hostInputData <= argsBufEnd), ACLNN_ERR_PARAM_INVALID,
               "MC2 hostInputData is out of argsBuf, argsBuf start is %p, argsBuf size is %zu, hostInputData is %p.",
               executor->args->argsBuf.data(), executor->args->argsBuf.size(), argsAddr->hostInputData);

    const size_t remainingSize = argsBufEnd - hostInputData;
    CHECK_COND(remainingSize >= requiredSize, ACLNN_ERR_PARAM_INVALID,
               "MC2 argsBuf remaining size is insufficient, remaining size is %zu, required size is %zu.",
               remainingSize, requiredSize);
    return OK;
}

bool NnopbaseReadCcuOpType(const void* opResCtx, uint32_t* opType)
{
    if ((opResCtx == nullptr) || (opType == nullptr)) {
        return false;
    }

    *opType = NNOPBASE_CCU_INVALID_OP_TYPE;
    const auto opTypeAddr = op::internal::PtrCastTo<const uint8_t>(opResCtx) + offsetof(NnopbaseCcuOpResCtx, opType);
    const aclError ret = aclrtMemcpy(opType, sizeof(uint32_t), opTypeAddr, sizeof(uint32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        OP_LOGW("Read ccu opType failed, opResCtx is %p, ret is %d.", opResCtx, ret);
        return false;
    }
    return *opType != NNOPBASE_CCU_INVALID_OP_TYPE;
}

bool NnopbaseIsCcuSplitLaunchOp(const NnopbaseExecutor* executor)
{
    if ((executor == nullptr) || executor->mc2.contextAddrs.empty() || executor->mc2.commHandles.empty()) {
        return false;
    }

    if ((executor->mc2.contextAddrs[0] == nullptr) || (executor->mc2.commHandles[0] == nullptr)) {
        OP_LOGI("Skip ccu split launch, contextAddr[0] is %p, hcomHandle[0] is %p.", executor->mc2.contextAddrs[0],
                executor->mc2.commHandles[0]);
        return false;
    }

    if (executor->mc2.fallback) {
        OP_LOGI("Skip ccu split launch, mc2 falling back to hccl module.");
        return false;
    }
    return true;
}

aclnnStatus DoHcclAllocComResourceByTiling(NnopbaseExecutor* executor, HcclComm comm, void* stream, void* tilingData,
                                           void** commCtx)
{
    const bool useA5Mc2Client = nnopbase::IndvSoc::GetInstance().NnopbaseUseA5Mc2Client(executor->mc2.serverType);
    OP_LOGI("Nnopbase MC2 alloc resource route, sType[%d], useA5Mc2Client[%d].", executor->mc2.serverType,
            useA5Mc2Client);
    if (useA5Mc2Client) {
        using AllocResult = nnopbase::IndvMc2ClientWrapper::HcclAllocComResourceResult;
        const AllocResult result = nnopbase::IndvMc2ClientWrapper::GetInstance().HcclAllocComResourceByTiling(
            comm, stream, tilingData, commCtx);
        if (result != AllocResult::ALGORITHM_NOT_SUPPORTED) {
            return result == AllocResult::SUCCESS ? OK : ACLNN_ERR_INNER;
        }
        executor->mc2.fallback = true;
        OP_LOGI("The mc2_client module does not support the current algorithm; falling back to the HCCL module.");
    }
    return nnopbase::IndvHcclWrapper::GetInstance().HcclAllocComResourceByTiling(comm, stream, tilingData, commCtx);
}
} // namespace

aclnnStatus NnopbaseGetHcomResource(NnopbaseExecutor* executor, aclrtStream const stream)
{
    executor->mc2.contextAddrs.clear();
    executor->mc2.aicpuStreams.clear();
    executor->mc2.aicpuThreads.clear();
    executor->mc2.aicpuNotifies.clear();
    for (HcclComm commHandle : executor->mc2.commHandles) {
        void* contextAddr = nullptr;
        if (commHandle == nullptr) {
            executor->mc2.contextAddrs.push_back(contextAddr);
            executor->mc2.aicpuStreams.push_back(nullptr);
            executor->mc2.aicpuThreads.push_back(0U);
            executor->mc2.aicpuNotifies.push_back(std::make_pair(nullptr, nullptr));
            continue;
        }
        if (!executor->hasTiling) {
            // 静态mc2算子场景，用json读取的静态tilingData去激活Hccl通信
            NNOPBASE_ASSERT_OK_RETVAL(DoHcclAllocComResourceByTiling(
                executor, commHandle, stream, executor->args->tilingInfo.staticTilingData.data(), &contextAddr));
        } else {
            NNOPBASE_ASSERT_OK_RETVAL(DoHcclAllocComResourceByTiling(
                executor, commHandle, stream,
                (op::internal::PtrCastTo<NnopbaseTilingData>(executor->args->tilingInfo.tilingData))->GetData(),
                &contextAddr));
        }
        executor->mc2.contextAddrs.push_back(contextAddr);
        aclrtStream aicpuStream = nullptr;
        aclrtStream notify[NNOPBASE_MC2_NOTIFY_COUNT] = {};
        if (!nnopbase::IndvSoc::GetInstance().NnopbaseEnableCcuLaunch(executor->mc2.serverType)) {
            if (nnopbase::IndvSoc::GetInstance().SupportMc2FusionLaunch()) {
                uint64_t threadHandle = 0U;
                NNOPBASE_ASSERT_OK_RETVAL(
                    nnopbase::IndvHcclWrapper::GetInstance().HcclGetUnfoldThread(commHandle, &threadHandle));
                executor->mc2.aicpuThreads.push_back(threadHandle);
                NNOPBASE_ASSERT_OK_RETVAL(nnopbase::IndvHcclWrapper::GetInstance().HcclStreamAcquireWithThread(
                    commHandle, threadHandle, &aicpuStream));
            } else {
                NNOPBASE_ASSERT_OK_RETVAL(nnopbase::IndvHcclWrapper::GetInstance().HcclGetAicpuOpStreamAndNotify(
                    commHandle, &aicpuStream, NNOPBASE_MC2_NOTIFY_COUNT, notify));
                executor->mc2.aicpuNotifies.push_back(std::make_pair(notify[0], notify[1]));
            }
            executor->mc2.aicpuStreams.push_back(aicpuStream);
        }
        OP_LOGI(
            "Executor is %p, stream is %p, commHandle is %p, get contextAddr is %p, get aicpuStream is %p, notify[0] "
            "is %p, notify[1] is %p.",
            executor, stream, commHandle, contextAddr, aicpuStream, notify[0], notify[1]);
    }
    return OK;
}

aclnnStatus NnopbaseExecutorGetMc2Num(NnopbaseExecutor* const executor, aclrtStream const stream,
                                      NnopbaseExecutorArgsAddr* argsAddr, uint32_t* mc2Num)
{
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseGetHcomResource(executor, stream));
    argsAddr->hcclDesc->version = 1U;
    argsAddr->hcclDesc->groupNum = static_cast<uint64_t>(executor->mc2.contextAddrs.size());
    if (nnopbase::IndvSoc::GetInstance().NnopbaseEnableCcuLaunch(executor->mc2.serverType)) {
        *mc2Num = argsAddr->hcclDesc->groupNum;
        executor->mc2.fusionArgs.isNoNeedH2DCopy = 0U;
        executor->mc2.fusionArgs.hostInputInfoPtr = nullptr;
        executor->mc2.fusionArgs.hostInputInfoNum = 0U;
        executor->mc2.fusionArgs.aicpuNum = 0U;
    } else {
        *mc2Num = argsAddr->hcclDesc->groupNum + 1U; // 1 is NnopbaseHcclCommParamDesc
        executor->mc2.aicpuArgs.kernelOffsetInfoPtr = nullptr;
        executor->mc2.aicpuArgs.kernelOffsetInfoNum = 0;
        executor->mc2.aicpuArgs.isNoNeedH2DCopy = false;
        executor->mc2.aicpuArgs.hostInputInfoPtr = nullptr;
        executor->mc2.aicpuArgs.hostInputInfoNum = 0U;
    }
    return OK;
}

aclnnStatus NnopbaseAddCapture(aclrtStream stream, std::vector<aclrtStream> aicpuStreams)
{
    aclmdlRICaptureStatus status;
    aclmdlRI captureMdl;
    if ((aclmdlRICaptureGetInfo(stream, &status, &captureMdl) == ACL_SUCCESS) &&
        (status == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE)) {
        for (auto& stm : aicpuStreams) {
            if (stm != nullptr) {
                NNOPBASE_ASSERT_RTOK_RETVAL(rtStreamAddToModel(stm, captureMdl));
            }
        }
    }
    return OK;
}

aclnnStatus NnopbaseLaunchKFCTask(NnopbaseExecutor* const executor, aclrtStream stream)
{
    OP_LOGI("Launch kernel by KFC mode.");
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddCapture(stream, executor->mc2.aicpuStreams));
    if (executor->mc2.aicpuStreams[0] != nullptr) {
        CHECK_COND(
            aclrtWaitAndResetNotify(executor->mc2.aicpuNotifies[0].first, executor->mc2.aicpuStreams[0], UINT32_MAX) ==
                ACL_SUCCESS,
            ACLNN_ERR_RUNTIME_ERROR,
            "Failed to call aclrtWaitAndResetNotify, executor is %p, aicpuStream is %p, stream is %p, Notify is %p.",
            executor, executor->mc2.aicpuStreams[0], stream, executor->mc2.aicpuNotifies[0].first);
        NNOPBASE_ASSERT_RTOK_RETVAL(aclrtRecordNotify(executor->mc2.aicpuNotifies[0].first, stream));
    }
    for (size_t i = 1U; i < executor->mc2.aicpuStreams.size(); i++) {
        if ((executor->mc2.aicpuStreams[0] != nullptr) && (executor->mc2.aicpuNotifies[i].first != nullptr)) {
            NNOPBASE_ASSERT_RTOK_RETVAL(aclrtWaitAndResetNotify(executor->mc2.aicpuNotifies[i].first,
                                                                executor->mc2.aicpuStreams[0], UINT32_MAX));
        }
        // aicpuStream和aicpuNotify是hccl同时创建的，aicpuStream不是null，aicpuNotify也不应该是null
        if (executor->mc2.aicpuStreams[i] != nullptr) {
            NNOPBASE_ASSERT_RTOK_RETVAL(
                aclrtRecordNotify(executor->mc2.aicpuNotifies[i].first, executor->mc2.aicpuStreams[i]));
        }
    }
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAicpuKernelLaunch(executor));
    for (size_t i = 1U; i < executor->mc2.aicpuStreams.size(); i++) {
        if (executor->mc2.aicpuStreams[i] != nullptr) {
            NNOPBASE_ASSERT_RTOK_RETVAL(aclrtWaitAndResetNotify(executor->mc2.aicpuNotifies[i].second,
                                                                executor->mc2.aicpuStreams[i], UINT32_MAX));
        }
        if ((executor->mc2.aicpuStreams[0] != nullptr) && (executor->mc2.aicpuNotifies[i].second != nullptr)) {
            NNOPBASE_ASSERT_RTOK_RETVAL(
                aclrtRecordNotify(executor->mc2.aicpuNotifies[i].second, executor->mc2.aicpuStreams[0]));
        }
    }
    OP_LOGI("Launch kernel by KFC mode successfully.");
    return OK;
}

aclnnStatus NnopbaseLaunchKFCTaskA5(NnopbaseExecutor* const executor, aclrtStream stream)
{
    OP_LOGI("Launch kernel by A5 KFC mode.");

    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddCapture(stream, executor->mc2.aicpuStreams));
    if (executor->mc2.aicpuStreams[0] != nullptr) {
        const uint64_t unfoldThread = executor->mc2.aicpuThreads[0];
        HcclComm comm = executor->mc2.commHandles[0];
        auto& wrapper = nnopbase::IndvHcclWrapper::GetInstance();

        // unfold线程在获取展开流时已保存，主流线程由stream现取
        uint64_t mainThread = 0U;
        uint32_t notifyNum = 61U;
        NNOPBASE_ASSERT_OK_RETVAL(wrapper.HcclThreadAcquireWithStream(comm, stream, notifyNum, &mainThread));

        // 主流线程record进unfold线程的notify槽位，unfold线程等待自身槽位，等价于原跨流同步语义
        NNOPBASE_ASSERT_OK_RETVAL(wrapper.HcommThreadNotifyWaitOnThread(unfoldThread, 0, UINT32_MAX));
        NNOPBASE_ASSERT_OK_RETVAL(wrapper.HcommThreadNotifyRecordOnThread(mainThread, unfoldThread, 0));
    }
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAicpuKernelLaunch(executor));
    OP_LOGI("Launch kernel by A5 KFC mode successfully.");
    return OK;
}

void NnopbaseCopyDavidMC2ParamDesc(NnopbaseExecutor* executor, NnopbaseExecutorArgsAddr* argsAddr)
{
    const NnopbaseUChar* const args = op::internal::PtrCastTo<NnopbaseUChar>(executor->mc2.fusionArgs.args);
    const uint16_t kfcArgsFmtOffset = static_cast<uint16_t>(argsAddr->hostInputData - args);
    executor->mc2.fusionArgs.aicpuArgs[0].kfcArgsFmtOffset = kfcArgsFmtOffset / sizeof(void*);
    OP_LOGI("KfcArgsFmtOffset is %u.", kfcArgsFmtOffset);
    for (size_t i = 0U; i < sizeof(NnopbaseHcclCommParamDesc); i++) {
        argsAddr->hostInputData = NnopbaseAppend1Byte(argsAddr->hostInputData,
                                                      (op::internal::PtrCastTo<NnopbaseUChar>(argsAddr->hcclDesc))[i]);
    }
    OP_LOGI("GroupNum is %u, hasffts is %u, tilingOff is %u, isDyn is %lu.", argsAddr->hcclDesc->groupNum,
            argsAddr->hcclDesc->hasFfts, argsAddr->hcclDesc->tilingOff, argsAddr->hcclDesc->isDyn);
    // 3 is soname/kernelname/opname
    executor->mc2.fusionArgs.argsSize = kfcArgsFmtOffset + sizeof(NnopbaseHcclCommParamDesc) + NNOPBASE_PARAM_EXT_LEN +
                                        static_cast<uint32_t>(sizeof(rtHostInputInfo_t));
}

void NnopbaseCopyMC2ParamDesc(NnopbaseExecutor* executor, NnopbaseExecutorArgsAddr* argsAddr)
{
    NnopbaseUChar* descAddr = op::internal::PtrCastTo<NnopbaseUChar>(executor->mc2.aicpuArgs.args);
    for (size_t i = 0U; i < sizeof(NnopbaseHcclCommParamDesc); i++) {
        descAddr = NnopbaseAppend1Byte(descAddr, (op::internal::PtrCastTo<NnopbaseUChar>(argsAddr->hcclDesc))[i]);
    }

    OP_LOGI("GroupNum is %u, hasffts is %u, tilingOff is %u, isDyn is %lu.", argsAddr->hcclDesc->groupNum,
            argsAddr->hcclDesc->hasFfts, argsAddr->hcclDesc->tilingOff, argsAddr->hcclDesc->isDyn);

    // 3 is soname/kernelname/opname
    executor->mc2.aicpuArgs.argsSize = executor->argsExt.argsSize + sizeof(NnopbaseHcclCommParamDesc) +
                                       NNOPBASE_AICPU_PARAM_LEN * 3U;
}

aclnnStatus NnopbasePrepareMC2Params(NnopbaseExecutor* executor, NnopbaseExecutorArgsAddr* argsAddr)
{
    NnopbaseUChar* args = nullptr;
    if (nnopbase::IndvSoc::GetInstance().NnopbaseEnableCcuLaunch(executor->mc2.serverType)) {
        args = (NnopbaseUChar*)executor->mc2.fusionArgs.args;
        executor->mc2.fusionArgs.aicpuArgs[0].soNameAddrOffset = static_cast<uint16_t>(argsAddr->hostInputData - args);
    } else {
        args = (NnopbaseUChar*)executor->mc2.aicpuArgs.args;
        executor->mc2.aicpuArgs.soNameAddrOffset = static_cast<uint32_t>(argsAddr->hostInputData - args);
    }

    const bool useA5Mc2Client = nnopbase::IndvSoc::GetInstance().NnopbaseUseA5Mc2Client(executor->mc2.serverType);
    const NnopbaseUChar* pSoName = useA5Mc2Client ? NNOPBASE_MC2_SERVER_SO_NAME : NNOPBASE_MC2_AICPU_SO_NAME;
    const NnopbaseUChar* pKernelName =
        useA5Mc2Client ? NNOPBASE_MC2_SERVER_KERNEL_NAME : NNOPBASE_MC2_AICPU_KERNEL_NAME;
    const std::string opName = std::string(executor->opType) + NNOPBASE_MC2_AICPU_SUFFIX;
    const bool enableCcuLaunch = nnopbase::IndvSoc::GetInstance().NnopbaseEnableCcuLaunch(executor->mc2.serverType);
    const size_t opNameDataLen = enableCcuLaunch ? NnopbaseAlignToEightBytes(opName.length()) : opName.length();
    size_t requiredSize = NNOPBASE_AICPU_PARAM_LEN * 2U + opNameDataLen;
    if (enableCcuLaunch) {
        requiredSize += sizeof(NnopbaseHcclCommParamDesc);
    }
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseCheckMC2ParamBuffer(executor, argsAddr, requiredSize));

    const NnopbaseUChar* const soNameAddr = argsAddr->hostInputData;
    for (size_t i = 0U; i < NNOPBASE_AICPU_PARAM_LEN; i++) {
        argsAddr->hostInputData = NnopbaseAppend1Byte(argsAddr->hostInputData, pSoName[i]);
    }

    if (nnopbase::IndvSoc::GetInstance().NnopbaseEnableCcuLaunch(executor->mc2.serverType)) {
        executor->mc2.fusionArgs.aicpuArgs[0].kernelNameAddrOffset = static_cast<uint16_t>(argsAddr->hostInputData -
                                                                                           args);
    } else {
        executor->mc2.aicpuArgs.kernelNameAddrOffset = static_cast<uint16_t>(argsAddr->hostInputData - args);
    }

    const NnopbaseUChar* const aicpuKernelNameAddr = argsAddr->hostInputData;
    for (size_t i = 0U; i < NNOPBASE_AICPU_PARAM_LEN; i++) {
        argsAddr->hostInputData = NnopbaseAppend1Byte(argsAddr->hostInputData, pKernelName[i]);
    }

    const NnopbaseUChar* const apiNameAddr = argsAddr->hostInputData;
    const size_t remainingSize = reinterpret_cast<uintptr_t>(executor->args->argsBuf.data()) +
                                 executor->args->argsBuf.size() - reinterpret_cast<uintptr_t>(argsAddr->hostInputData);
    auto ret = memcpy_s(argsAddr->hostInputData, remainingSize, opName.data(), opName.length());
    if (ret != EOK) {
        OP_LOGE(ACLNN_ERR_INNER, "Failed to execute memcpy_s for aicpu opName, opName is %s, length %zu, ret is %d.",
                opName.c_str(), opName.length(), ret);
        return ACLNN_ERR_PARAM_INVALID;
    }
    OP_LOGI("MC2 params are soName[%s], apiName[%.*s], aicpuKernelName[%s].", reinterpret_cast<const char*>(soNameAddr),
            static_cast<int32_t>(opName.length()), reinterpret_cast<const char*>(apiNameAddr),
            reinterpret_cast<const char*>(aicpuKernelNameAddr));

    if (enableCcuLaunch) {
        argsAddr->hostInputData += opNameDataLen;
        NnopbaseCopyDavidMC2ParamDesc(executor, argsAddr);
    } else {
        argsAddr->hostInputData += opName.length();
        NnopbaseCopyMC2ParamDesc(executor, argsAddr);
    }
    return OK;
}

static aclnnStatus NnopbaseGetAicpuTimeout(uint32_t* time)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(time);
    CHECK_COND(aclrtGetOpExecuteTimeout(time) == ACL_RT_SUCCESS, ACLNN_ERR_RUNTIME_ERROR, "Failed to get op timeout.");
    const NnopbaseChar* hcclTimeoutEnv = nullptr;
    MM_SYS_GET_ENV(MM_ENV_HCCL_EXEC_TIMEOUT, hcclTimeoutEnv);
    uint32_t hcclTimeout = NNOPBASE_HCCL_DEFAULT_TIME;
    if (hcclTimeoutEnv != nullptr) {
        hcclTimeout = std::atoi(hcclTimeoutEnv);
    }
    *time += hcclTimeout;
    OP_LOGI("HcclTimeout is %u, time is %u.", hcclTimeout, *time);
    return OK;
}

aclnnStatus NnopbaseAicpuKernelLaunch(NnopbaseExecutor* const executor)
{
    uint32_t time = 0;
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseGetAicpuTimeout(&time));
    executor->mc2.aicpuArgs.timeout = time;
    const uint64_t launchBeginTime = NnopbaseMsprofSysTime();
    const std::string opType = std::string(executor->opType) + NNOPBASE_MC2_AICPU_SUFFIX;
    const uint32_t numBlocks = executor->args->tilingInfo.aicpuNumBlocks == 0U ?
                                   1U :
                                   executor->args->tilingInfo.aicpuNumBlocks;
    NNOPBASE_ASSERT_RTOK_RETVAL(
        rtAicpuKernelLaunchExWithArgs(KERNEL_TYPE_AICPU_KFC, &opType[0], numBlocks, &executor->mc2.aicpuArgs, nullptr,
                                      executor->mc2.aicpuStreams[0], RT_KERNEL_USE_SPECIAL_TIMEOUT));
    OP_LOGI("%s launch successfully, numBlocks is %u.", opType.c_str(), numBlocks);

    NnopbaseInnerReportLaunchInfo(launchBeginTime, executor->dfx.aicpuItemId);
    NnopbaseReportAicpuAdditionInfo(launchBeginTime + 1, &opType[0]);
    NnopbaseReportCacheOpInfo(executor, 0U, MSPROF_GE_TASK_TYPE_AI_CPU, executor->mc2.aicpuStreams[0]);
    return OK;
}

aclnnStatus NnopbaseFusionKernelLaunch(NnopbaseExecutor* const executor, aclrtStream const stream)
{
    OP_LOGI("Launch kernel by fusion mode.");
    uint32_t numBlocks = executor->args->tilingInfo.numBlocks;
    uint64_t tilingKey = executor->args->tilingInfo.tilingKey;
    OP_LOGI("numBlocks is %u, tilingKey is %lu.", numBlocks, tilingKey);

    rtLaunchAttribute_t launchAttr[1];
    launchAttr[0].id = RT_LAUNCH_ATTRIBUTE_BLOCKDIM;
    launchAttr[0].value.blockDim = numBlocks;
    rtLaunchConfig_t launchCfg = {launchAttr, 1U};
    rtAicoreFusionInfo_t aicoreInfo = {executor->args->binInfo->ccuBinHandle, tilingKey, &launchCfg, nullptr};
    rtFunsionTaskInfo_t fusionTaskInfo = {};
    rtCcuTaskGroup_t ccuTaskGroup = {};
    NNOPBASE_ASSERT_RTOK_RETVAL(nnopbase::IndvHcclWrapper::GetInstance().HcclGetCcuTaskInfo(
        executor->mc2.commHandles[0],
        (op::internal::PtrCastTo<NnopbaseTilingData>(executor->args->tilingInfo.tilingData))->GetData(),
        &ccuTaskGroup));
    fusionTaskInfo.subTask[0].type = RT_FUSION_CCU;
    fusionTaskInfo.subTask[0].task.ccuInfo = ccuTaskGroup;
    fusionTaskInfo.subTaskNum = 2U;
    fusionTaskInfo.subTask[1].type = RT_FUSION_AICORE;
    fusionTaskInfo.subTask[1].task.aicoreInfo = aicoreInfo;
    OP_LOGI("SubTaskNum is %u. subTask[0].type is %d, subTask[1].type is %d.", fusionTaskInfo.subTaskNum,
            fusionTaskInfo.subTask[0].type, fusionTaskInfo.subTask[1].type);

    const uint64_t launchBeginTime = NnopbaseMsprofSysTime();
    NNOPBASE_ASSERT_RTOK_RETVAL(rtFusionLaunch(&fusionTaskInfo, stream, &executor->mc2.fusionArgs));
    NnopbaseExecutorReportProfiling(executor, numBlocks, MSPROF_GE_TASK_TYPE_FUSION, launchBeginTime, stream);
    OP_LOGI("Op %s fusion launch successfully.", executor->opType);
    return OK;
}

aclnnStatus NnopbaseCcuKernelLaunch(NnopbaseExecutor* const executor, rtStream_t const stream)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);

    // commHandles，contextAddrs已经由上层调用确保非空
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor->mc2.commHandles[0]);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor->mc2.contextAddrs[0]);

    NNOPBASE_ASSERT_OK_RETVAL(nnopbase::IndvMc2ClientWrapper::GetInstance().CcuKernelLaunch(
        executor->mc2.commHandles[0], executor->mc2.contextAddrs[0]));

    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseExecutorKernelLaunch(executor, stream));

    OP_LOGI("Op %s ccu split launch successfully.", executor->opType);
    return OK;
}

static aclnnStatus NnopbaseMC2KernelMTE(NnopbaseExecutor* executor, aclrtStream stream)
{
    return NnopbaseExecutorKernelLaunch(executor, stream);
}

static aclnnStatus NnopbaseMC2KernelCCU(NnopbaseExecutor* const executor, aclrtStream stream)
{
    if (NnopbaseIsCcuSplitLaunchOp(executor)) {
        return NnopbaseCcuKernelLaunch(executor, stream);
    }
    return NnopbaseFusionKernelLaunch(executor, stream);
}

static aclnnStatus NnopbaseMC2KernelAicpu(NnopbaseExecutor* executor, aclrtStream stream)
{
    if (nnopbase::IndvSoc::GetInstance().SupportMc2FusionLaunch()) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseLaunchKFCTaskA5(executor, stream));
    } else {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseLaunchKFCTask(executor, stream));
    }
    return NnopbaseExecutorKernelLaunch(executor, stream);
}

aclnnStatus NnopbaseMC2KernelLaunch(NnopbaseExecutor* executor, aclrtStream stream)
{
    OP_LOGI("HcclServerType is %d.", executor->mc2.serverType);
    if (NnopbaseSkipKernelLaunch(executor)) {
        OP_LOGI("Output of MC2 operator is empty and ZeroEleOutputLaunch is not enabled, skipping launch.");
        return OK;
    }
    if (executor->mc2.serverType == NNOPBASE_HCCL_SERVER_TYPE_END) { // default
        if (nnopbase::IndvSoc::GetInstance().NnopbaseEnableCcuLaunch(executor->mc2.serverType)) {
            return NnopbaseMC2KernelCCU(executor, stream);
        } else {
            return NnopbaseMC2KernelAicpu(executor, stream);
        }
    } else {
        switch (executor->mc2.serverType) {
            case NNOPBASE_HCCL_SERVER_TYPE_MTE:
                return NnopbaseMC2KernelMTE(executor, stream);
            case NNOPBASE_HCCL_SERVER_TYPE_CCU:
                return NnopbaseMC2KernelCCU(executor, stream);
            default:
                return NnopbaseMC2KernelAicpu(executor, stream);
        }
    }
}

#ifdef __cplusplus
}
#endif
