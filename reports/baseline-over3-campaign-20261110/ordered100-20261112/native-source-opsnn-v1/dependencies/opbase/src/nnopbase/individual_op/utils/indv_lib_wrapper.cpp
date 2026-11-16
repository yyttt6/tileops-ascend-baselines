/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "indv_lib_wrapper.h"
#include "securec.h"
#include "utils/indv_debug_assert.h"

namespace nnopbase {
NnopBaseLoadSo::~NnopBaseLoadSo() = default;

aclnnStatus NnopBaseLoadSo::openSo(const char* loadSoPath)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(loadSoPath);

    OP_LOGI("Start loading dynamic library %s.", loadSoPath);
    soHandle_ = mmDlopen(loadSoPath, MMPA_RTLD_LAZY);
    if (soHandle_ == nullptr) {
        OP_LOGW("Failed to open "
                "dynamic library %s. dlopen error:%s.",
                loadSoPath, mmDlerror());
        return ACLNN_ERR_PARAM_INVALID;
    }
    OP_LOGI("Load dynamic library %s successfully.", loadSoPath);

    soPath_ = loadSoPath;
    return OK;
}

void NnopBaseLoadSo::closeSo()
{
    if (soHandle_ != nullptr) {
        mmDlclose(soHandle_);
        soHandle_ = nullptr;
        soPath_ = "";
    }
}

ApiWrapper& ApiWrapper::GetInstance()
{
    static ApiWrapper instance;
    return instance;
}

void* ApiWrapper::GetFunc(const char* funcName)
{
    const std::lock_guard<std::mutex> lk(mutex_);
    if (!hasInit_) {
        if (openSo("libopapi_math.so") != OK) {
            std::string
                errMsg = "Failed to call the AutoContiguous API. Check whether the ops_math operator package"
                         " is installed and whether libopapi_math.so is under environment variable LD_LIBRARY_PATH";
            OP_LOGE_FOR_EXECUTION_ERROR_WITHOUT_SOLUTION(errMsg.c_str());
            return nullptr;
        }
        hasInit_ = true;
    }
    if (funcName == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Function name is nullptr!");
        return nullptr;
    }
    const auto& it = functions_.find(funcName);
    if (it != functions_.end()) {
        return it->second;
    }
    void* const func = LoadFunction<void*>(funcName);
    if (func != nullptr) {
        functions_[funcName] = func;
        return func;
    }
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Failed to get function %s!", funcName);
    return nullptr;
}

aclnnStatus ApiWrapper::LoadFunctions() { return OK; }

ApiWrapper::ApiWrapper() = default;

ApiWrapper::~ApiWrapper() { closeSo(); }

IndvMc2ClientWrapper& IndvMc2ClientWrapper::GetInstance(void)
{
    static IndvMc2ClientWrapper instance;
    return instance;
}

aclnnStatus IndvMc2ClientWrapper::IndvMc2ClientWrapperInit(const char* loadSoPath)
{
    hcclAllocComResourceByTilingHandle = nullptr;
    ccuKernelLaunchHandle = nullptr;
    closeSo();
    aclnnStatus ret = openSo(loadSoPath);
    if (ret != OK) {
        OP_LOGW("Failed to load dynamic library %s, retVal = %d.", loadSoPath, ret);
        return ret;
    }
    NNOPBASE_ASSERT_OK_RETVAL(LoadFunctions());
    return OK;
}

IndvMc2ClientWrapper::HcclAllocComResourceResult IndvMc2ClientWrapper::HcclAllocComResourceByTiling(
    HcclComm comm, void* stream, void* TilingData, void** commContext)
{
    static constexpr uint32_t HCCL_ALGORITHM_NOT_SUPPORTED = 1042U;
    if (hcclAllocComResourceByTilingHandle == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER, "HcclAllocComResourceByTiling function of the mc2_client module is not loaded.");
        return HcclAllocComResourceResult::FAILED;
    }

    const HcclResult ret = hcclAllocComResourceByTilingHandle(comm, stream, TilingData, commContext);
    if (static_cast<uint32_t>(ret) == HCCL_ALGORITHM_NOT_SUPPORTED) {
        return HcclAllocComResourceResult::ALGORITHM_NOT_SUPPORTED;
    }
    if (ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "Failed to invoke the HcclAllocComResourceByTiling "
                "function of the mc2_client module, ret = %d, comm = %p.",
                ret, comm);
        return HcclAllocComResourceResult::FAILED;
    }

    return HcclAllocComResourceResult::SUCCESS;
}

aclnnStatus IndvMc2ClientWrapper::CcuKernelLaunch(HcclComm comm, void* opResCtx)
{
    static constexpr uint32_t KERNEL_LAUNCH_SUCCESS = 0U;
    NNOPBASE_ASSERT_NOTNULL_RETVAL(ccuKernelLaunchHandle);

    uint32_t ret = ccuKernelLaunchHandle(comm, opResCtx);
    if (ret != KERNEL_LAUNCH_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "Failed to invoke the CcuKernelLaunch "
                "function of the mc2_client module, ret = %u, comm = %p, opResCtx = %p.",
                ret, comm, opResCtx);
        return ACLNN_ERR_INNER;
    }

    return OK;
}

IndvMc2ClientWrapper::IndvMc2ClientWrapper(void) {}

IndvMc2ClientWrapper::~IndvMc2ClientWrapper()
{
    hcclAllocComResourceByTilingHandle = nullptr;
    ccuKernelLaunchHandle = nullptr;
    closeSo();
}

aclnnStatus IndvMc2ClientWrapper::LoadFunctions()
{
    hcclAllocComResourceByTilingHandle = LoadFunction<HcclAllocComResourceByTilingFunc>("HcclAllocComResourceByTiling");
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclAllocComResourceByTilingHandle);
    ccuKernelLaunchHandle = LoadFunction<CcuKernelLaunchFunc>("CcuKernelLaunch");
    NNOPBASE_ASSERT_NOTNULL_RETVAL(ccuKernelLaunchHandle);
    return OK;
}

IndvHcclWrapper& IndvHcclWrapper::GetInstance(void)
{
    static IndvHcclWrapper instance;
    return instance;
}

aclnnStatus IndvHcclWrapper::IndvHcclWrapperInit(const char* loadSoPath, const bool needLoadDavidHcclApi)
{
    hcclAllocComResourceByTilingHandle = nullptr;
    hcclGetAicpuOpStreamAndNotifyHandle = nullptr;
    hcomGetCommHandleByGroupHandle = nullptr;
    hcclGetRankIdHandle = nullptr;
    hcclGetCcuTaskInfoHandle = nullptr;

    // 静态Mc2开启场景重构Topo结构落盘Json用
    hcclGetRankSizeHandle = nullptr;
    hcclRankGraphGetLayersHandle = nullptr;
    hcclRankGraphGetRankSizeByLayerHandle = nullptr;
    hcclRankGraphGetTopoTypeByLayerHandle = nullptr;
    hcclGetHcclBufferHandle = nullptr;
    hcclGetCommNameHandle = nullptr;
    hcclEngineCtxGetHandle = nullptr;
    hcclThreadResGetInfoHandle = nullptr;
    hcclThreadAcquireWithStreamHandle = nullptr;
    hcommThreadNotifyRecordOnThreadHandle = nullptr;
    hcommThreadNotifyWaitOnThreadHandle = nullptr;
    hcclThreadAcquireWithConfigHandle = nullptr;
    hcclEngineCtxCreateHandle = nullptr;
    hcclGetNotifyNumInThreadHandle = nullptr;
    closeSo();

    NNOPBASE_ASSERT_OK_RETVAL(openSo(loadSoPath));
    NNOPBASE_ASSERT_OK_RETVAL(LoadFunctions());
    if (needLoadDavidHcclApi) {
        NNOPBASE_ASSERT_OK_RETVAL(LoadDavidHcclFunctions());
    }
    return OK;
}

aclnnStatus IndvHcclWrapper::HcclAllocComResourceByTiling(HcclComm comm, void* stream, void* TilingData,
                                                          void** commContext)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclAllocComResourceByTilingHandle);

    HcclResult ret = hcclAllocComResourceByTilingHandle(comm, stream, TilingData, commContext);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "Failed to invoke the HcclAllocComResourceByTiling "
                "function of the hccl module, ret = %d, comm = %p.",
                ret, comm);
        return ACLNN_ERR_INNER;
    }

    return OK;
}

aclnnStatus IndvHcclWrapper::HcclGetAicpuOpStreamAndNotify(HcclComm comm, aclrtStream* opStream, uint8_t notifyCnt,
                                                           void** aicpuNotify)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclGetAicpuOpStreamAndNotifyHandle);
    HcclResult ret = hcclGetAicpuOpStreamAndNotifyHandle(comm, opStream, notifyCnt, aicpuNotify);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "Failed to invoke the HcclGetAicpuOpStreamAndNotify "
                "function of the hccl module, ret = %d, comm = %p, notifyCnt = %u.",
                ret, comm, notifyCnt);
        return ACLNN_ERR_INNER;
    }

    return OK;
}

aclnnStatus IndvHcclWrapper::HcomGetCommHandleByGroup(const char* group, HcclComm* commHandle)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcomGetCommHandleByGroupHandle);

    HcclResult ret = hcomGetCommHandleByGroupHandle(group, commHandle);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "Failed to invoke the HcomGetCommHandleByGroup "
                "function of the hccl module, ret = %d, group = %s.",
                ret, group);
        return ACLNN_ERR_INNER;
    }

    return OK;
}

aclnnStatus IndvHcclWrapper::HcclGetRankId(HcclComm comm, uint32_t* rankSize)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclGetRankIdHandle);

    HcclResult ret = hcclGetRankIdHandle(comm, rankSize);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "Failed to invoke the HcclGetRankId "
                "function of the hccl module, ret = %u, comm = %p.",
                ret, comm);
        return ACLNN_ERR_INNER;
    }

    return OK;
}

aclnnStatus IndvHcclWrapper::HcclGetCcuTaskInfo(HcclComm comm, void* fusionArgs, void* ccuTaskGroup)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclGetCcuTaskInfoHandle);

    HcclResult ret = hcclGetCcuTaskInfoHandle(comm, fusionArgs, ccuTaskGroup);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "Failed to invoke the HcclGetCcuTaskInfo "
                "function of the hccl module, ret = %u, comm = %p.",
                ret, comm);
        return ACLNN_ERR_INNER;
    }

    return OK;
}

aclnnStatus IndvHcclWrapper::HcclRankGraphGetLayers(HcclComm comm, uint32_t** netLayers, uint32_t* netLayerNum)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclRankGraphGetLayersHandle);

    HcclResult ret = hcclRankGraphGetLayersHandle(comm, netLayers, netLayerNum);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "Failed to invoke the HcclRankGraphGetLayers "
                "function of the hccl module, ret = %u, comm = %p.",
                ret, comm);
        return ACLNN_ERR_INNER;
    }

    return OK;
}

aclnnStatus IndvHcclWrapper::HcclRankGraphGetRankSizeByLayer(HcclComm comm, uint32_t netLayer, uint32_t* rankNum)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclRankGraphGetRankSizeByLayerHandle);

    HcclResult ret = hcclRankGraphGetRankSizeByLayerHandle(comm, netLayer, rankNum);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "Failed to invoke the HcclRankGraphGetRankSizeByLayer "
                "function of the hccl module, ret = %u, comm = %p.",
                ret, comm);
        return ACLNN_ERR_INNER;
    }

    return OK;
}

aclnnStatus IndvHcclWrapper::HcclRankGraphGetTopoTypeByLayer(HcclComm comm, uint32_t netLayer, uint32_t* topoType)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclRankGraphGetTopoTypeByLayerHandle);

    HcclResult ret = hcclRankGraphGetTopoTypeByLayerHandle(comm, netLayer, topoType);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "Failed to invoke the HcclRankGraphGetTopoTypeByLayer "
                "function of the hccl module, ret = %u, comm = %p.",
                ret, comm);
        return ACLNN_ERR_INNER;
    }

    return OK;
}

aclnnStatus IndvHcclWrapper::HcclGetHcclBuffer(HcclComm comm, void** buffer, uint64_t* size)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclGetHcclBufferHandle);

    HcclResult ret = hcclGetHcclBufferHandle(comm, buffer, size);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "Failed to invoke the HcclGetHcclBuffer "
                "function of the hccl module, ret = %u, comm = %p.",
                ret, comm);
        return ACLNN_ERR_INNER;
    }

    return OK;
}

aclnnStatus IndvHcclWrapper::HcclGetRankSize(HcclComm comm, uint32_t* rankSize)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclGetRankSizeHandle);

    HcclResult ret = hcclGetRankSizeHandle(comm, rankSize);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "Failed to invoke the HcclGetRankId "
                "function of the hccl module, ret = %u, comm = %p.",
                ret, comm);
        return ACLNN_ERR_INNER;
    }

    return OK;
}

aclnnStatus IndvHcclWrapper::HcclGetCommName(HcclComm comm, char* commName)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclGetCommNameHandle);

    HcclResult ret = hcclGetCommNameHandle(comm, commName);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "Failed to invoke the HcclGetCommName "
                "function of the hccl module, ret = %u, comm = %p.",
                ret, comm);
        return ACLNN_ERR_INNER;
    }

    return OK;
}

aclnnStatus IndvHcclWrapper::HcclGetUnfoldThread(HcclComm comm, uint64_t* threadHandle)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(threadHandle);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclEngineCtxGetHandle);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclGetNotifyNumInThreadHandle);

    // 与hccl侧CommEngine::COMM_ENGINE_CPU_TS保持一致
    constexpr int32_t COMM_ENGINE_CPU_TS = 1;
    constexpr uint32_t COMM_NAME_MAX_LENGTH = 128;
    constexpr uint32_t UNFOLD_TAG_MAX_LENGTH = COMM_NAME_MAX_LENGTH + 16;

    char commName[COMM_NAME_MAX_LENGTH] = {};
    NNOPBASE_ASSERT_OK_RETVAL(HcclGetCommName(comm, commName));

    // 与hccl侧SaveUnfoldThreadInfo约定的"%s_unfold"保持一致
    char unfoldCtxTag[UNFOLD_TAG_MAX_LENGTH] = {};
    int len = snprintf_s(unfoldCtxTag, sizeof(unfoldCtxTag), sizeof(unfoldCtxTag) - 1, "%s_unfold", commName);
    if (len <= 0) {
        OP_LOGE(ACLNN_ERR_INNER, "Failed to fill unfoldCtxTag, comm = %p, commName = %s.", comm, commName);
        return ACLNN_ERR_INNER;
    }

    void* ctx = nullptr;
    uint64_t size = sizeof(uint64_t);
    HcclResult ret = hcclEngineCtxGetHandle(comm, unfoldCtxTag, COMM_ENGINE_CPU_TS, &ctx, &size);
    if (ret == HCCL_SUCCESS && ctx != nullptr) {
        *threadHandle = *(static_cast<uint64_t*>(ctx));
        uint32_t notifyNum = 0;
        HcclResult notifyRet = hcclGetNotifyNumInThreadHandle(comm, *threadHandle, COMM_ENGINE_CPU_TS, &notifyNum);
        OP_LOGI("HcclGetUnfoldThread success, comm = %p, unfoldCtxTag = %s, threadHandle = %lu, notifyNum = %u, "
                "notifyRet = %d.",
                comm, unfoldCtxTag, *threadHandle, notifyNum, notifyRet);
        // notify数查询成功且不为0，说明展开流线程已带有A5同步所需的notify槽位，直接复用
        if (notifyRet == HCCL_SUCCESS && notifyNum != 0U) {
            return OK;
        }
        // 否则(查询失败，或notifyNum为0，即展开流线程由hccl侧以notifyNumPerThread=0创建)，缺少A5同步所需的notify槽位。
        // V2通信域下，以相同的(engine=COMM_ENGINE_CPU, type=THREAD_TYPE_TS)再次调用HcclThreadAcquireWithConfig，
        // hccl侧会命中该原线程并原地补齐notify(SupplementNotify)，返回的仍是同一个threadHandle，不会新建线程。
        // 因此这里只需按notifyNumPerThread=2补notify，并把(不变的)handle等值回写已存在的ctx即可，无需重新落盘。
        OP_LOGI("Unfold thread has no usable notify (notifyNum = %u, notifyRet = %d), supplement notify on the "
                "original thread, comm = %p, unfoldCtxTag = %s, threadHandle = %lu.",
                notifyNum, notifyRet, comm, unfoldCtxTag, *threadHandle);
        NNOPBASE_ASSERT_OK_RETVAL(HcclAcquireUnfoldThread(comm, threadHandle));
        *(static_cast<uint64_t*>(ctx)) = *threadHandle;
        OP_LOGI("HcclGetUnfoldThread supplement notify success, comm = %p, unfoldCtxTag = %s, threadHandle = %lu.",
                comm, unfoldCtxTag, *threadHandle);
        return OK;
    }

    // 当前上下文中尚无展开流线程，参考hccl侧HcclGetAicpuThread申请一个并落盘到通信引擎上下文
    OP_LOGI("Unfold thread not found in ctx, acquire a new one, comm = %p, unfoldCtxTag = %s, ret = %u, ctx = %p.",
            comm, unfoldCtxTag, ret, ctx);
    NNOPBASE_ASSERT_OK_RETVAL(HcclAcquireUnfoldThread(comm, threadHandle));
    NNOPBASE_ASSERT_OK_RETVAL(HcclSaveUnfoldThread(comm, unfoldCtxTag, *threadHandle));
    OP_LOGI("HcclGetUnfoldThread acquire success, comm = %p, unfoldCtxTag = %s, threadHandle = %lu.", comm,
            unfoldCtxTag, *threadHandle);
    return OK;
}

bool IndvHcclWrapper::NnopbaseThreadConfigInit(NnopbaseThreadConfig* config, uint32_t num)
{
    if (config == nullptr) {
        return false;
    }
    for (uint32_t i = 0U; i < num; i++) {
        if (memset_s(&config[i], sizeof(NnopbaseThreadConfig), 0, sizeof(NnopbaseThreadConfig)) != EOK) {
            return false;
        }
        config[i].header.version = NNOPBASE_THREAD_CONFIG_VERSION;
        config[i].header.magicWord = NNOPBASE_THREAD_CONFIG_MAGIC_WORD;
        config[i].header.size = static_cast<uint32_t>(sizeof(NnopbaseThreadConfig));
        config[i].header.reserved = 0U;
        config[i].notifyNumPerThread = 0U;
    }
    return true;
}

aclnnStatus IndvHcclWrapper::HcclAcquireUnfoldThread(HcclComm comm, uint64_t* threadHandle)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(threadHandle);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclThreadAcquireWithConfigHandle);

    // 与hccl侧CommEngine::COMM_ENGINE_CPU保持一致
    constexpr int32_t COMM_ENGINE_CPU = 0;
    constexpr uint32_t UNFOLD_THREAD_NUM = 1;

    NnopbaseThreadConfig unfoldThreadConfig;
    if (!NnopbaseThreadConfigInit(&unfoldThreadConfig, UNFOLD_THREAD_NUM)) {
        OP_LOGE(ACLNN_ERR_INNER, "Failed to init ThreadConfig, comm = %p.", comm);
        return ACLNN_ERR_INNER;
    }
    unfoldThreadConfig.notifyNumPerThread = 2U;
    HcclResult ret = hcclThreadAcquireWithConfigHandle(comm, COMM_ENGINE_CPU, UNFOLD_THREAD_NUM,
                                                       NNOPBASE_THREAD_TYPE_TS, &unfoldThreadConfig, threadHandle);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "Failed to invoke the HcclThreadAcquireWithConfig "
                "function of the hccl module, ret = %d, comm = %p.",
                ret, comm);
        return ACLNN_ERR_INNER;
    }

    return OK;
}

aclnnStatus IndvHcclWrapper::HcclSaveUnfoldThread(HcclComm comm, const char* unfoldCtxTag, uint64_t threadHandle)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(unfoldCtxTag);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclEngineCtxCreateHandle);

    // 与hccl侧CommEngine::COMM_ENGINE_CPU_TS保持一致
    constexpr int32_t COMM_ENGINE_CPU_TS = 1;

    void* ctx = nullptr;
    HcclResult ret = hcclEngineCtxCreateHandle(comm, unfoldCtxTag, COMM_ENGINE_CPU_TS, sizeof(uint64_t), &ctx);
    if (ret != HCCL_SUCCESS || ctx == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER,
                "Failed to invoke the HcclEngineCtxCreate "
                "function of the hccl module, ret = %d, comm = %p, unfoldCtxTag = %s, ctx = %p.",
                ret, comm, unfoldCtxTag, ctx);
        return ACLNN_ERR_INNER;
    }
    *(static_cast<uint64_t*>(ctx)) = threadHandle;
    OP_LOGI("HcclSaveUnfoldThread success, comm = %p, unfoldCtxTag = %s, ctx = %p, threadHandle = %lu.", comm,
            unfoldCtxTag, ctx, threadHandle);
    return OK;
}

aclnnStatus IndvHcclWrapper::HcclStreamAcquireWithThread(HcclComm comm, uint64_t threadHandle, aclrtStream* stream)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(stream);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclThreadResGetInfoHandle);

    constexpr int32_t THREAD_RES_TYPE_STREAM = 0;
    void* unfoldStream = nullptr;
    HcclResult ret = hcclThreadResGetInfoHandle(comm, threadHandle, THREAD_RES_TYPE_STREAM, sizeof(void*),
                                                &unfoldStream);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "Failed to invoke the HcclStreamAcquireWithThread "
                "function of the hccl module, ret = %u, threadHandle = %lu.",
                ret, threadHandle);
        return ACLNN_ERR_INNER;
    }
    *stream = static_cast<aclrtStream>(unfoldStream);
    OP_LOGI("HcclStreamAcquireWithThread success, stream = %p.", *stream);
    return OK;
}

aclnnStatus IndvHcclWrapper::HcclThreadAcquireWithStream(HcclComm comm, aclrtStream stream, uint32_t notifyNum,
                                                         uint64_t* thread)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(thread);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclThreadAcquireWithStreamHandle);

    // 与hccl侧CommEngine::COMM_ENGINE_CPU_TS保持一致
    constexpr int32_t COMM_ENGINE_CPU_TS = 1;
    HcclResult ret = hcclThreadAcquireWithStreamHandle(comm, COMM_ENGINE_CPU_TS, stream, notifyNum, thread);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "Failed to invoke the HcclThreadAcquireWithStream "
                "function of the hccl module, ret = %d, comm = %p, stream = %p, notifyNum = %u.",
                ret, comm, stream, notifyNum);
        return ACLNN_ERR_INNER;
    }

    return OK;
}

aclnnStatus IndvHcclWrapper::HcommThreadNotifyRecordOnThread(uint64_t thread, uint64_t dstThread, uint32_t dstNotifyIdx)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcommThreadNotifyRecordOnThreadHandle);

    int32_t ret = hcommThreadNotifyRecordOnThreadHandle(thread, dstThread, dstNotifyIdx);
    if (ret != 0) {
        OP_LOGE(ACLNN_ERR_INNER,
                "Failed to invoke the HcommThreadNotifyRecordOnThread "
                "function of the hcomm module, ret = %d, thread = %lu, dstThread = %lu, dstNotifyIdx = %u.",
                ret, thread, dstThread, dstNotifyIdx);
        return ACLNN_ERR_INNER;
    }

    return OK;
}

aclnnStatus IndvHcclWrapper::HcommThreadNotifyWaitOnThread(uint64_t thread, uint32_t notifyIdx, uint32_t timeOut)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcommThreadNotifyWaitOnThreadHandle);

    int32_t ret = hcommThreadNotifyWaitOnThreadHandle(thread, notifyIdx, timeOut);
    if (ret != 0) {
        OP_LOGE(ACLNN_ERR_INNER,
                "Failed to invoke the HcommThreadNotifyWaitOnThread "
                "function of the hcomm module, ret = %d, thread = %lu, notifyIdx = %u, timeOut = %u.",
                ret, thread, notifyIdx, timeOut);
        return ACLNN_ERR_INNER;
    }

    return OK;
}

IndvHcclWrapper::IndvHcclWrapper(void) {}

IndvHcclWrapper::~IndvHcclWrapper()
{
    hcclAllocComResourceByTilingHandle = nullptr;
    hcclGetAicpuOpStreamAndNotifyHandle = nullptr;
    hcomGetCommHandleByGroupHandle = nullptr;
    hcclGetCcuTaskInfoHandle = nullptr;
    hcclGetRankIdHandle = nullptr;

    // 静态Mc2开启场景重构Topo结构落盘Json用
    hcclGetRankSizeHandle = nullptr;
    hcclRankGraphGetLayersHandle = nullptr;
    hcclRankGraphGetRankSizeByLayerHandle = nullptr;
    hcclRankGraphGetTopoTypeByLayerHandle = nullptr;
    hcclGetHcclBufferHandle = nullptr;
    hcclGetCommNameHandle = nullptr;
    hcclEngineCtxGetHandle = nullptr;
    hcclThreadResGetInfoHandle = nullptr;
    hcclThreadAcquireWithStreamHandle = nullptr;
    hcommThreadNotifyRecordOnThreadHandle = nullptr;
    hcommThreadNotifyWaitOnThreadHandle = nullptr;
    hcclThreadAcquireWithConfigHandle = nullptr;
    hcclEngineCtxCreateHandle = nullptr;
    hcclGetNotifyNumInThreadHandle = nullptr;
    closeSo();
}

aclnnStatus IndvHcclWrapper::LoadFunctions()
{
    hcclAllocComResourceByTilingHandle = LoadFunction<HcclAllocComResourceByTilingFunc>("HcclAllocComResourceByTiling");
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclAllocComResourceByTilingHandle);
    hcclGetAicpuOpStreamAndNotifyHandle = LoadFunction<HcclGetAicpuOpStreamAndNotifyFunc>(
        "HcclGetAicpuOpStreamAndNotify");
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclGetAicpuOpStreamAndNotifyHandle);
    hcomGetCommHandleByGroupHandle = LoadFunction<HcomGetCommHandleByGroupFunc>("HcomGetCommHandleByGroup");
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcomGetCommHandleByGroupHandle);
    hcclGetRankIdHandle = LoadFunction<HcclGetRankIdFunc>("HcclGetRankId");
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclGetRankIdHandle);

    // 静态Mc2开启场景重构Topo结构落盘Json用
    hcclGetRankSizeHandle = LoadFunction<HcclGetRankSizeFunc>("HcclGetRankSize");
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclGetRankSizeHandle);
    hcclRankGraphGetLayersHandle = LoadFunction<HcclRankGraphGetLayersFunc>("HcclRankGraphGetLayers");
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclRankGraphGetLayersHandle);
    hcclRankGraphGetRankSizeByLayerHandle = LoadFunction<HcclRankGraphGetRankSizeByLayerFunc>(
        "HcclRankGraphGetRankSizeByLayer");
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclRankGraphGetRankSizeByLayerHandle);
    hcclRankGraphGetTopoTypeByLayerHandle = LoadFunction<HcclRankGraphGetTopoTypeByLayerFunc>(
        "HcclRankGraphGetTopoTypeByLayer");
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclRankGraphGetTopoTypeByLayerHandle);
    hcclGetHcclBufferHandle = LoadFunction<HcclGetHcclBufferFunc>("HcclGetHcclBuffer");
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclGetHcclBufferHandle);
    hcclGetCommNameHandle = LoadFunction<HcclGetCommNameFunc>("HcclGetCommName");
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclGetCommNameHandle);
    return OK;
}

aclnnStatus IndvHcclWrapper::LoadDavidHcclFunctions()
{
    hcclGetCcuTaskInfoHandle = LoadFunction<HcclGetCcuTaskInfoFunc>("HcclGetCcuTaskInfo");
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclGetCcuTaskInfoHandle);
    // Mc2融合下发场景，opbase侧自行实现GetUnfoldThreadInfo所需的hccl底层接口
    hcclEngineCtxGetHandle = LoadFunction<HcclEngineCtxGetFunc>("HcclEngineCtxGet");
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclEngineCtxGetHandle);
    hcclThreadResGetInfoHandle = LoadFunction<HcclThreadResGetInfoFunc>("HcclThreadResGetInfo");
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclThreadResGetInfoHandle);
    // A5融合下发场景，KFC任务与aicpu server之间用thread级notify原语做同步
    hcclThreadAcquireWithStreamHandle = LoadFunction<HcclThreadAcquireWithStreamFunc>("HcclThreadAcquireWithStream");
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclThreadAcquireWithStreamHandle);
    hcommThreadNotifyRecordOnThreadHandle = LoadFunction<HcommThreadNotifyRecordOnThreadFunc>(
        "HcommThreadNotifyRecordOnThread");
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcommThreadNotifyRecordOnThreadHandle);
    hcommThreadNotifyWaitOnThreadHandle = LoadFunction<HcommThreadNotifyWaitOnThreadFunc>(
        "HcommThreadNotifyWaitOnThread");
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcommThreadNotifyWaitOnThreadHandle);
    // A5融合下发场景，当前无展开流线程时用于申请CPU_TS线程并保存到通信引擎上下文
    hcclThreadAcquireWithConfigHandle = LoadFunction<HcclThreadAcquireWithConfigFunc>("HcclThreadAcquireWithConfig");
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclThreadAcquireWithConfigHandle);
    hcclEngineCtxCreateHandle = LoadFunction<HcclEngineCtxCreateFunc>("HcclEngineCtxCreate");
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclEngineCtxCreateHandle);
    hcclGetNotifyNumInThreadHandle = LoadFunction<HcclGetNotifyNumInThreadFunc>("HcclGetNotifyNumInThread");
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclGetNotifyNumInThreadHandle);
    return OK;
}

NnopbaseSoLoader& NnopbaseSoLoader::GetInstance()
{
    static NnopbaseSoLoader instance;
    return instance;
}

void* NnopbaseSoLoader::FindFunction(const std::vector<std::string>& soPaths, const std::string& funcName)
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& soPath : soPaths) {
        if (loadedLibraries.find(soPath) != loadedLibraries.end()) {
            if (functions_[soPath].find(funcName) != functions_[soPath].end()) {
                OP_LOGI("Get func %s from %s successfully.", funcName.c_str(), soPath.c_str());
                return functions_[soPath][funcName];
            } else {
                void* func_ptr = mmDlsym(loadedLibraries[soPath], funcName.c_str());
                if (func_ptr) {
                    functions_[soPath][funcName] = func_ptr;
                    OP_LOGW("Get funcName %s successfully", funcName.c_str());
                    return func_ptr;
                }
                OP_LOGW("Failed to get function %s from %s!", funcName.c_str(), soPath.c_str());
            }
        } else {
            void* handle = mmDlopen(soPath.c_str(), RTLD_LAZY);
            if (!handle) {
                OP_LOGW("Failed to open %s, dlopen error:%s.", soPath.c_str(), mmDlerror());
                continue;
            }
            loadedLibraries[soPath] = handle;
            OP_LOGW("Open %s successfully", soPath.c_str());

            void* func_ptr = mmDlsym(loadedLibraries[soPath], funcName.c_str());
            if (func_ptr) {
                functions_[soPath][funcName] = func_ptr;
                OP_LOGW("Get funcName %s successfully", funcName.c_str());
                return func_ptr;
            }
        }
    }
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Failed to get function %s!", funcName.c_str());
    return nullptr;
}

void NnopbaseSoLoader::closeSo()
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& pair : loadedLibraries) {
        mmDlclose(pair.second);
    }
    loadedLibraries.clear();
    functions_.clear();
}

NnopbaseSoLoader::NnopbaseSoLoader() = default;

NnopbaseSoLoader::~NnopbaseSoLoader() { closeSo(); }

IndvRtsWrapper* IndvRtsWrapper::GetInstance(void)
{
    static IndvRtsWrapper instance;
    static std::once_flag initFlag;
    static bool initSuccess = false;
    std::call_once(initFlag, [&]() { initSuccess = (instance.IndvRtsWrapperInit() == OK); });
    return initSuccess ? &instance : nullptr;
}

aclnnStatus IndvRtsWrapper::AclrtBinarySetExceptionCallback(void* binHandle, void* callback, void* userData)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(rtsRegisterExceptionCallbackHandle);
    auto ret = rtsRegisterExceptionCallbackHandle(binHandle, reinterpret_cast<customOpExceptionCallback>(callback),
                                                  userData);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "Failed to invoke the aclrtBinarySetExceptionCallback "
                "function of the acl_rt module, ret = %d, callback = %p.",
                ret, callback);
        return ACLNN_ERR_INNER;
    }
    return OK;
}

IndvRtsWrapper::IndvRtsWrapper(){};

IndvRtsWrapper::~IndvRtsWrapper()
{
    rtsRegisterExceptionCallbackHandle = nullptr;
    closeSo();
}

aclnnStatus IndvRtsWrapper::IndvRtsWrapperInit()
{
    rtsRegisterExceptionCallbackHandle = nullptr;
    closeSo();
    const char* ascendHomePath = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_HOME_PATH, ascendHomePath);
    OP_CHECK(ascendHomePath != nullptr,
             OP_LOGE_FOR_CONFIG_ERROR_INVALID_ENVIRONMENT_VARIABLE("Loading libacl_rt.so", "ASCEND_HOME_PATH"),
             return ACLNN_ERR_INNER_NULLPTR);
    std::string rtsLibraryPath = std::string(ascendHomePath) + "/lib64/libacl_rt.so";
    NNOPBASE_ASSERT_OK_RETVAL(openSo(rtsLibraryPath.c_str()));
    NNOPBASE_ASSERT_OK_RETVAL(LoadFunctions());
    return OK;
}

aclnnStatus IndvRtsWrapper::LoadFunctions()
{
    rtsRegisterExceptionCallbackHandle = LoadFunction<NnopbaseRegisterExceptionCallbackFunc>(
        "aclrtBinarySetExceptionCallback");
    NNOPBASE_ASSERT_NOTNULL_RETVAL(rtsRegisterExceptionCallbackHandle);
    return OK;
}
} // namespace nnopbase
