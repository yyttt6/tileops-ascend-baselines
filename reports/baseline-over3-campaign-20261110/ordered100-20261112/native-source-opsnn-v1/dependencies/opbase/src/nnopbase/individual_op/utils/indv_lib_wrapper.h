/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INDV_LIB_WRAPPER_H_
#define INDV_LIB_WRAPPER_H_

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "acl/acl_rt.h"
#include "aclnn/acl_meta.h"
#include "hccl/base.h"
#include "mmpa/mmpa_api.h"
#include "nnopbase_error_msg.h"

namespace nnopbase {
class NnopBaseLoadSo {
public:
    virtual ~NnopBaseLoadSo();

    aclnnStatus openSo(const char* loadSoPath);
    void closeSo();

    template <typename T>
    T LoadFunction(const char* const functionName)
    {
        OP_LOGI("Start loading function %s in %s.", functionName, soPath_.c_str());
        T function = reinterpret_cast<T>(mmDlsym(soHandle_, functionName));
        if (function == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Failed to load function %s in %s. "
                    "dlsym error:%s.",
                    functionName, soPath_.c_str(), mmDlerror());
            return nullptr;
        }
        OP_LOGI("Load function %s in %s successfully.", functionName, soPath_.c_str());

        return function;
    }

    // 待子类实现加载所需的符号表
    virtual aclnnStatus LoadFunctions() = 0;

private:
    void* soHandle_ = nullptr;
    std::string soPath_ = "";
};

class ApiWrapper : public NnopBaseLoadSo {
public:
    static ApiWrapper& GetInstance();
    void* GetFunc(const char* funcName);
    aclnnStatus LoadFunctions() override;

private:
    ApiWrapper();
    ~ApiWrapper() override;

    bool hasInit_ = false;
    std::mutex mutex_;
    std::unordered_map<std::string, void*> functions_;
};

class IndvMc2ClientWrapper : public NnopBaseLoadSo {
public:
    // Function-specific result contract for mc2_client resource allocation. It must not be propagated as aclnnStatus.
    enum class HcclAllocComResourceResult {
        SUCCESS,
        ALGORITHM_NOT_SUPPORTED,
        FAILED,
    };

    static IndvMc2ClientWrapper& GetInstance(void);

    // 调用端必须保证不多线程调用
    aclnnStatus IndvMc2ClientWrapperInit(const char* loadSoPath);
    HcclAllocComResourceResult HcclAllocComResourceByTiling(HcclComm comm, void* stream, void* TilingData,
                                                           void** commContext);
    aclnnStatus CcuKernelLaunch(HcclComm comm, void* opResCtx);

private:
    using HcclAllocComResourceByTilingFunc = HcclResult (*)(HcclComm, void*, void*, void**);
    // CcuKernelLaunch is resolved dynamically. Keep only its stable return-value ABI here to avoid a CCU header
    // dependency in opbase.
    using CcuKernelLaunchFunc = uint32_t (*)(HcclComm, void*);

    IndvMc2ClientWrapper(void);
    ~IndvMc2ClientWrapper() override;

    aclnnStatus LoadFunctions() override;

    HcclAllocComResourceByTilingFunc hcclAllocComResourceByTilingHandle = nullptr;
    CcuKernelLaunchFunc ccuKernelLaunchHandle = nullptr;
};

class IndvHcclWrapper : public NnopBaseLoadSo {
public:
    static IndvHcclWrapper& GetInstance(void);

    // 调用端必须保证不多线程调用
    aclnnStatus IndvHcclWrapperInit(const char* loadSoPath, const bool needLoadDavidHcclApi);
    aclnnStatus HcclAllocComResourceByTiling(HcclComm comm, void* stream, void* TilingData, void** commContext);
    aclnnStatus HcclGetAicpuOpStreamAndNotify(HcclComm comm, aclrtStream* opStream, uint8_t notifyCnt,
                                              void** aicpuNotify);
    aclnnStatus HcomGetCommHandleByGroup(const char* group, HcclComm* commHandle);
    aclnnStatus HcclGetRankId(HcclComm comm, uint32_t* rankSize);
    aclnnStatus HcclGetCcuTaskInfo(HcclComm comm, void* fusionArgs, void* ccuTaskGroup);
    aclnnStatus HcclRankGraphGetLayers(HcclComm comm, uint32_t** netLayers, uint32_t* netLayerNum);
    aclnnStatus HcclRankGraphGetRankSizeByLayer(HcclComm comm, uint32_t netLayer, uint32_t* rankNum);
    aclnnStatus HcclRankGraphGetTopoTypeByLayer(HcclComm comm, uint32_t netLayer, uint32_t* topoType);
    aclnnStatus HcclGetHcclBuffer(HcclComm comm, void** buffer, uint64_t* size);
    aclnnStatus HcclGetRankSize(HcclComm comm, uint32_t* rankSize);
    aclnnStatus HcclGetCommName(HcclComm comm, char* commName);
    aclnnStatus HcclGetUnfoldThread(HcclComm comm, uint64_t* threadHandle);
    aclnnStatus HcclAcquireUnfoldThread(HcclComm comm, uint64_t* threadHandle);
    aclnnStatus HcclSaveUnfoldThread(HcclComm comm, const char* unfoldCtxTag, uint64_t threadHandle);
    aclnnStatus HcclStreamAcquireWithThread(HcclComm comm, uint64_t threadHandle, aclrtStream* stream);
    aclnnStatus HcclThreadAcquireWithStream(HcclComm comm, aclrtStream stream, uint32_t notifyNum, uint64_t* thread);
    aclnnStatus HcommThreadNotifyRecordOnThread(uint64_t thread, uint64_t dstThread, uint32_t dstNotifyIdx);
    aclnnStatus HcommThreadNotifyWaitOnThread(uint64_t thread, uint32_t notifyIdx, uint32_t timeOut);

private:
    using HcclAllocComResourceByTilingFunc = HcclResult (*)(HcclComm, void*, void*, void**);
    using HcclGetAicpuOpStreamAndNotifyFunc = HcclResult (*)(HcclComm, aclrtStream*, uint8_t, void**);
    using HcomGetCommHandleByGroupFunc = HcclResult (*)(const char*, HcclComm*);
    using HcclGetCcuTaskInfoFunc = HcclResult (*)(HcclComm, void*, void*);
    using HcclGetRankIdFunc = HcclResult (*)(HcclComm, uint32_t*);

    // 静态Mc2开启场景重构Topo结构落盘Json用
    using HcclGetRankSizeFunc = HcclResult (*)(HcclComm, uint32_t*);
    using HcclRankGraphGetLayersFunc = HcclResult (*)(HcclComm, uint32_t**, uint32_t*);
    using HcclRankGraphGetRankSizeByLayerFunc = HcclResult (*)(HcclComm, uint32_t, uint32_t*);
    using HcclRankGraphGetTopoTypeByLayerFunc = HcclResult (*)(HcclComm, uint32_t, uint32_t*);
    using HcclGetHcclBufferFunc = HcclResult (*)(HcclComm, void**, uint64_t*);
    using HcclGetCommNameFunc = HcclResult (*)(HcclComm, char*);
    using HcclEngineCtxGetFunc = HcclResult (*)(HcclComm, const char*, int32_t, void**, uint64_t*);
    using HcclThreadResGetInfoFunc = HcclResult (*)(HcclComm, uint64_t, int32_t, uint32_t, void**);
    using HcclThreadAcquireWithStreamFunc = HcclResult (*)(HcclComm, int32_t, aclrtStream, uint32_t, uint64_t*);
    using HcommThreadNotifyRecordOnThreadFunc = int32_t (*)(uint64_t, uint64_t, uint32_t);
    using HcommThreadNotifyWaitOnThreadFunc = int32_t (*)(uint64_t, uint32_t, uint32_t);
    // 与hcomm侧hcomm_res_defs.h的ThreadType::THREAD_TYPE_TS保持一致
    static constexpr int32_t NNOPBASE_THREAD_TYPE_TS = 0;
    // 与hcomm侧hcomm_res_defs.h的CommAbiHeader/ThreadConfig二进制布局保持一致
    // ThreadConfig带magicWord校验，申请线程前必须经NnopbaseThreadConfigInit初始化，否则hccl侧会报magicWord mismatch
    static constexpr uint32_t NNOPBASE_THREAD_CONFIG_MAGIC_WORD = 0x0f0f0f2f;
    static constexpr uint32_t NNOPBASE_THREAD_CONFIG_VERSION = 1U;
    struct NnopbaseCommAbiHeader {
        uint32_t version;
        uint32_t magicWord;
        uint32_t size;
        uint32_t reserved;
    };
    struct NnopbaseThreadConfig {
        NnopbaseCommAbiHeader header;
        uint16_t notifyNumPerThread;
        uint8_t reserved[14];
    };
    static bool NnopbaseThreadConfigInit(NnopbaseThreadConfig* config, uint32_t num);
    using HcclThreadAcquireWithConfigFunc = HcclResult (*)(HcclComm, int32_t, uint32_t, int32_t,
                                                           const NnopbaseThreadConfig*, uint64_t*);
    using HcclEngineCtxCreateFunc = HcclResult (*)(HcclComm, const char*, int32_t, uint64_t, void**);
    using HcclGetNotifyNumInThreadFunc = HcclResult (*)(HcclComm, uint64_t, int32_t, uint32_t*);

    IndvHcclWrapper(void);
    ~IndvHcclWrapper() override;

    aclnnStatus LoadFunctions() override;
    aclnnStatus LoadDavidHcclFunctions();

    HcclAllocComResourceByTilingFunc hcclAllocComResourceByTilingHandle = nullptr;
    HcclGetAicpuOpStreamAndNotifyFunc hcclGetAicpuOpStreamAndNotifyHandle = nullptr;
    HcomGetCommHandleByGroupFunc hcomGetCommHandleByGroupHandle = nullptr;
    HcclGetCcuTaskInfoFunc hcclGetCcuTaskInfoHandle = nullptr;
    HcclGetRankIdFunc hcclGetRankIdHandle = nullptr;

    // 静态Mc2开启场景重构Topo结构落盘Json用
    HcclGetRankSizeFunc hcclGetRankSizeHandle = nullptr;
    HcclRankGraphGetLayersFunc hcclRankGraphGetLayersHandle = nullptr;
    HcclRankGraphGetRankSizeByLayerFunc hcclRankGraphGetRankSizeByLayerHandle = nullptr;
    HcclRankGraphGetTopoTypeByLayerFunc hcclRankGraphGetTopoTypeByLayerHandle = nullptr;
    HcclGetHcclBufferFunc hcclGetHcclBufferHandle = nullptr;
    HcclGetCommNameFunc hcclGetCommNameHandle = nullptr;
    HcclEngineCtxGetFunc hcclEngineCtxGetHandle = nullptr;
    HcclThreadResGetInfoFunc hcclThreadResGetInfoHandle = nullptr;
    HcclThreadAcquireWithStreamFunc hcclThreadAcquireWithStreamHandle = nullptr;
    HcommThreadNotifyRecordOnThreadFunc hcommThreadNotifyRecordOnThreadHandle = nullptr;
    HcommThreadNotifyWaitOnThreadFunc hcommThreadNotifyWaitOnThreadHandle = nullptr;
    HcclThreadAcquireWithConfigFunc hcclThreadAcquireWithConfigHandle = nullptr;
    HcclEngineCtxCreateFunc hcclEngineCtxCreateHandle = nullptr;
    HcclGetNotifyNumInThreadFunc hcclGetNotifyNumInThreadHandle = nullptr;
};

class NnopbaseSoLoader {
public:
    static NnopbaseSoLoader& GetInstance();
    void* FindFunction(const std::vector<std::string>& soPaths, const std::string& funcName);
    void closeSo();

private:
    NnopbaseSoLoader();
    ~NnopbaseSoLoader();

    std::unordered_map<std::string, std::unordered_map<std::string, void*>> functions_;
    std::unordered_map<std::string, void*> loadedLibraries;
    std::mutex mutex_;
};

class IndvRtsWrapper : public NnopBaseLoadSo {
public:
    static IndvRtsWrapper* GetInstance(void);
    aclnnStatus AclrtBinarySetExceptionCallback(void* binHandle, void* callback, void* userData = nullptr);

private:
    using customOpExceptionCallback = void (*)(aclrtExceptionInfo*, void*);
    using NnopbaseRegisterExceptionCallbackFunc = aclError (*)(void*, customOpExceptionCallback, void*);

    IndvRtsWrapper();
    ~IndvRtsWrapper() override;

    aclnnStatus IndvRtsWrapperInit();
    aclnnStatus LoadFunctions() override;

    NnopbaseRegisterExceptionCallbackFunc rtsRegisterExceptionCallbackHandle = nullptr;
};
} // namespace nnopbase

#endif // INDV_LIB_WRAPPER_H_
