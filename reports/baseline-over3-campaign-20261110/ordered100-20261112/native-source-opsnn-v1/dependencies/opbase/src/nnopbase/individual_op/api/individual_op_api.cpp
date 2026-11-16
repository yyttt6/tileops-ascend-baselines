/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "individual_op_api.h"
#include <mutex>
#include <thread>
#include <atomic>
#include <algorithm>
#include <sstream>
#include "utils/indv_base.h"
#include "utils/indv_debug_assert.h"
#include "utils/indv_lib_wrapper.h"
#include "utils/thread_var_container.h"
#include "utils/indv_soc.h"
#include "executor/indv_executor.h"
#include "executor/indv_collector.h"
#include "executor/indv_tilingcontext_builder.h"
#include "executor/indv_bininfo.h"
#include "executor/indv_args_pool.h"
#include "executor/indv_cache_key_builder.h"
#include "profiling/prof_api.h"
#include "aclnn/aclnn_base.h"
#include "thread_local_context.h"
#include "opdev/platform.h"
#include "op_dfx_internal.h"
#include "nnopbase_error_msg.h"

void NnopbaseOpLogE(const aclnnStatus code, const NnopbaseChar* const expr) { OP_LOGE(code, "Check %s failed", expr); }
using namespace op::internal;

namespace {
constexpr size_t MAX_PRINT_NUM = 128U;
#define NNOPBASE_LOG_GUARD(level)                          \
    if (unlikely(CheckLogLevelInner(OP_ID, level) != 1)) { \
        return;                                            \
    }

static inline std::string NnopbaseBuildShapeStridesStr(const aclTensor* const tensor)
{
    const auto& shape = tensor->GetViewShape();
    const auto& strides = tensor->GetViewStrides();
    std::stringstream ss;
    ss << "shape=[";
    for (size_t i = 0U; i < shape.GetDimNum(); i++) {
        if (i > 0)
            ss << ",";
        ss << shape.GetDim(i);
    }
    ss << "], strides=[";
    for (size_t i = 0U; i < strides.size(); i++) {
        if (i > 0)
            ss << ",";
        ss << strides[i];
    }
    ss << "]";
    return ss.str();
}

static inline void NnopbaseLogTensorInfo(const char* const paramName, const aclTensor* const tensor,
                                         const uint32_t index)
{
    NNOPBASE_LOG_GUARD(OP_LOG_INFO);
    if (tensor == nullptr) {
        OP_LOGI("[DFX] %s[%u] addr=nullptr.", paramName, index);
        return;
    }
    OP_LOGI("[DFX] %s[%u] dtype=%s, format=%s, placement=%d, addr=%p, %s, offset=%lld.", paramName, index,
            op::ToString(tensor->GetDataType()).GetString(), op::ToString(tensor->GetStorageFormat()).GetString(),
            static_cast<int32_t>(tensor->GetPlacement()), tensor->GetData(),
            NnopbaseBuildShapeStridesStr(tensor).c_str(), static_cast<int64_t>(tensor->GetViewOffset()));
}

static inline void NnopbaseLogTensorListInfo(const char* const paramName, const aclTensorList* const tensorList,
                                             const uint32_t index)
{
    NNOPBASE_LOG_GUARD(OP_LOG_INFO);
    if (tensorList == nullptr) {
        OP_LOGI("[DFX] %s[%u] addr=nullptr.", paramName, index);
        return;
    }
    OP_LOGI("[DFX] %s[%u] dtype=aclTensorList, size=%llu, addr=%p.", paramName, index, tensorList->Size(),
            static_cast<const void*>(tensorList));
    for (uint64_t i = 0; i < tensorList->Size(); i++) {
        if ((*tensorList)[i] != nullptr) {
            const auto& t = (*tensorList)[i];
            OP_LOGI("[DFX] %s[%u][%llu] dtype=%s, format=%s, placement=%d, addr=%p, %s, offset=%lld.", paramName, index,
                    i, op::ToString(t->GetDataType()).GetString(), op::ToString(t->GetStorageFormat()).GetString(),
                    static_cast<int32_t>(t->GetPlacement()), t->GetData(), NnopbaseBuildShapeStridesStr(t).c_str(),
                    static_cast<int64_t>(t->GetViewOffset()));
        }
    }
}

static inline void NnopbaseLogScalarInfo(const char* const paramName, const aclScalar* const scalar,
                                         const uint32_t index)
{
    NNOPBASE_LOG_GUARD(OP_LOG_INFO);
    if (scalar == nullptr) {
        OP_LOGI("[DFX] %s[%u] addr=nullptr.", paramName, index);
        return;
    }
    OP_LOGI("[DFX] %s[%u] info=%s, size=%llu, addr=%p.", paramName, index, scalar->ToString().GetString(),
            static_cast<uint64_t>(scalar->Size()), static_cast<const void*>(scalar));
}

static inline void NnopbaseLogScalarListInfo(const char* const paramName, const aclScalarList* const scalarList,
                                             const uint32_t index)
{
    NNOPBASE_LOG_GUARD(OP_LOG_INFO);
    if (scalarList == nullptr) {
        OP_LOGI("[DFX] %s[%u] addr=nullptr.", paramName, index);
        return;
    }
    OP_LOGI("[DFX] %s[%u] dtype=ScalarList, size=%llu, addr=%p.", paramName, index, scalarList->Size(),
            static_cast<const void*>(scalarList));
    const uint64_t totalSize = scalarList->Size();
    for (uint64_t i = 0U; i < totalSize; i++) {
        if (i >= static_cast<uint64_t>(MAX_PRINT_NUM)) {
            OP_LOGI("[DFX] %s[%u][%llu]...", paramName, index, i);
            break;
        }
        const aclScalar* const scalar = (*scalarList)[i];
        if (scalar != nullptr) {
            OP_LOGI("[DFX] %s[%u][%llu] info=%s, size=%llu, addr=%p.", paramName, index, i,
                    scalar->ToString().GetString(), static_cast<uint64_t>(scalar->Size()),
                    static_cast<const void*>(scalar));
        }
    }
}

template <typename T>
static inline void NnopbaseLogArrayInfo(const char* const typeName, const char* const paramName, const T* const array,
                                        const size_t index)
{
    NNOPBASE_LOG_GUARD(OP_LOG_INFO);
    if (array == nullptr) {
        OP_LOGI("[DFX] %s[%zu] dtype=%s, addr=nullptr.", paramName, index, typeName);
        return;
    }
    const size_t size = array->Size();
    const void* data = array->GetData();
    std::stringstream ss;
    ss << "[";
    if (data != nullptr) {
        for (size_t i = 0U; i < std::min(size, MAX_PRINT_NUM); i++) {
            if (i > 0)
                ss << ",";
            if constexpr (std::is_same<T, aclIntArray>::value) {
                ss << static_cast<const int64_t*>(data)[i];
            } else if constexpr (std::is_same<T, aclBoolArray>::value) {
                ss << static_cast<const bool*>(data)[i];
            } else if constexpr (std::is_same<T, aclFloatArray>::value) {
                ss << static_cast<const float*>(data)[i];
            }
        }
        if (size > MAX_PRINT_NUM)
            ss << ",...";
    } else {
        ss << "<null data>";
    }
    ss << "]";
    OP_LOGI("[DFX] %s[%zu] dtype=%s, size=%zu, addr=%p, values=%s.", paramName, index, typeName, size, data,
            ss.str().c_str());
}

static inline void NnopbaseLogAttrValueInfo(const NnopbaseAttrDtype dtype, const void* const addr, const size_t attrLen,
                                            const size_t index)
{
    NNOPBASE_LOG_GUARD(OP_LOG_INFO);
    const std::string typeNameStr = nnopbase::ToStr(dtype);
    const char* typeName = typeNameStr.c_str();
    if (addr == nullptr) {
        OP_LOGI("[DFX] AddAttr[%zu] dtype=%s, addr=nullptr.", index, typeName);
        return;
    }
    switch (dtype) {
        case kNnopbaseInt:
            OP_LOGI("[DFX] AddAttr[%zu] dtype=%s, value=%lld, attrLen=%zu, addr=%p.", index, typeName,
                    *op::internal::PtrCastTo<const int64_t>(addr), attrLen, addr);
            break;
        case kNnopbaseFloat:
            OP_LOGI("[DFX] AddAttr[%zu] dtype=%s, value=%f, attrLen=%zu, addr=%p.", index, typeName,
                    *op::internal::PtrCastTo<const float>(addr), attrLen, addr);
            break;
        case kNnopbaseBool:
            OP_LOGI("[DFX] AddAttr[%zu] dtype=%s, value=%s, attrLen=%zu, addr=%p.", index, typeName,
                    *op::internal::PtrCastTo<const bool>(addr) ? "1" : "0", attrLen, addr);
            break;
        case kNnopbaseString:
            OP_LOGI("[DFX] AddAttr[%zu] dtype=%s, value=\"%s\", attrLen=%zu, addr=%p.", index, typeName,
                    op::internal::PtrCastTo<const char>(addr), attrLen, addr);
            break;
        default:
            OP_LOGI("[DFX] AddAttr[%zu] dtype=Unknown(%d), attrLen=%zu, addr=%p.", index, static_cast<int32_t>(dtype),
                    attrLen, addr);
            break;
    }
}

static inline void NnopbaseLogRawArrayAttrInfo(const NnopbaseAttrDtype dtype, const void* const addr, const size_t len,
                                               const size_t index)
{
    NNOPBASE_LOG_GUARD(OP_LOG_INFO);
    const std::string typeNameStr = nnopbase::ToStr(dtype) + "Array";
    const char* typeName = typeNameStr.c_str();
    if (addr == nullptr || len == 0U) {
        OP_LOGI("[DFX] AddAttr[%zu] dtype=%s, len=%zu, addr=%p.", index, typeName, len, addr);
        return;
    }
    std::stringstream ss;
    ss << "[";
    const size_t printLen = std::min(len, MAX_PRINT_NUM);
    for (size_t i = 0U; i < printLen; i++) {
        if (i > 0)
            ss << ",";
        switch (dtype) {
            case kNnopbaseInt:
                ss << op::internal::PtrCastTo<const int64_t>(addr)[i];
                break;
            case kNnopbaseFloat:
                ss << op::internal::PtrCastTo<const float>(addr)[i];
                break;
            case kNnopbaseBool:
                ss << (op::internal::PtrCastTo<const bool>(addr)[i] ? "1" : "0");
                break;
            default:
                break;
        }
    }
    if (len > MAX_PRINT_NUM)
        ss << ",...";
    ss << "]";
    OP_LOGI("[DFX] AddAttr[%zu] dtype=%s, values=%s, arraySize=%zu, addr=%p.", index, typeName, ss.str().c_str(), len,
            addr);
}
} // namespace

#ifdef __cplusplus
extern "C" {
#endif

NnopbaseExecutorSpaceSet g_nnopbaseSpaceSet;
static std::mutex g_nnopbaseInitMtx;
constexpr const char* LIBMC2_CLIENT_SO = "libmc2_client.so";
constexpr const char* LIBHCCL_SO = "libhccl.so";
constexpr uint32_t MAX_SOC_NAME_LIST_LEN = 256U;

aclnnStatus NnopbaseInit()
{
    static bool isInit = false;
    if (isInit) {
        return OK;
    }
    OP_LOGI("NnopbaseInit Start.");
    if (gBinCollector == nullptr) {
        auto binCollector = std::make_unique<NnopbaseBinCollector>();
        NNOPBASE_ASSERT_NOTNULL_RETVAL(binCollector);
        RecordNnopbaseInitTime(binCollector.get(), NnopbaseCollectorTimeIdx::kCollectorInitStart);
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseCollectorInit(binCollector.get()));
        RecordNnopbaseInitTime(binCollector.get(), NnopbaseCollectorTimeIdx::kCollectorInitEnd);
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseCollectorWork(binCollector.get()));
        gBinCollector = binCollector.release();
    }
    NnopbaseExecutorSpaceSetInit(&g_nnopbaseSpaceSet);
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseExecutorSetGlobalConfig());
    PrintNnopbaseInitTimeStampInfo();
    isInit = true;
    OP_LOGI("NnopbaseInit end.");
    return OK;
}

void NnopbaseReloadStaticBinJsonInfos(const char* basePath)
{
    OP_LOGI("NnopbaseReloadStaticBinJsonInfos start.");
    NnopbaseExecutorClearSet(&g_nnopbaseSpaceSet);
    std::string staticKernelBasePath = "";
    if (basePath != nullptr) {
        staticKernelBasePath = std::string(basePath);
    }
    nnopbase::utils::ThreadVarContainer::SetStaticKernelBasePathInThread(staticKernelBasePath);
    auto ret = NnopbaseRefreshStaticKernelInfos(gBinCollector);
    if (ret != OK) {
        OP_LOGW("Failed to reload static kernel information.");
    } else {
        OP_LOGI("Reload static kernel information successfully.");
    }
    nnopbase::ArgsPool::GetInstance().Finalize();
    OP_LOGI("NnopbaseReloadStaticBinJsonInfos end.");
}

aclnnStatus NnopbaseCreateExecutorSpace(void** space)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(space);
    const std::lock_guard<std::mutex> lock(g_nnopbaseInitMtx);
    NNOPBASE_ASSERT_RTOK_RETVAL(NnopbaseInit());
    if (*space == nullptr) {
        NnopbaseExecutorSpace* executorSpace = new (std::nothrow) NnopbaseExecutorSpace;
        NNOPBASE_ASSERT_NOTNULL_RETVAL(executorSpace);
        *space = executorSpace;
        NnopbaseExecutorAddSpaceToSet(&g_nnopbaseSpaceSet, executorSpace);
    }
    return OK;
}

void* NnopbaseGetExecutor(void* space, const NnopbaseChar* opType, NnopbaseChar* inputsDesc, uint32_t inputNum,
                          NnopbaseChar* outputsDesc, uint32_t outputNum, NnopbaseChar* attrsDesc, uint32_t attrsNum)
{
    op::internal::GetThreadLocalContext().logInfo_.l2ApiName = opType;
    NnopbaseExecutorSpace* executorSpace = (NnopbaseExecutorSpace*)space;
    if (executorSpace == nullptr) {
        return nullptr;
    }
    const std::lock_guard<std::mutex> lock(executorSpace->spaceMtx);
    // 同一个opType可以对应多个executor
    for (auto executor : executorSpace->executors) {
        if ((executor != nullptr) && (!executor->isWork)) {
            executor->isWork = true;
            executor->poolIndex = op::internal::GetThreadLocalContext().poolIndex_;
            OP_LOGI("Get op %s space %p executor addr %p.", opType, space, executor);
            RecordNnopbaseTime(executor, NnopbaseTimeIdx::kAfterCreateExecutor);
            // 留出optype的偏移量后续生成key的时候用
            executor->ownArgs.keyLen = strlen(executor->opType);
            executor->ownArgs.remainKeyLen = NNOPBASE_MAX_ARGS_KEY_LEN - strlen(executor->opType);
            executor->ownArgs.inputKey.resize(NNOPBASE_MAX_ARGS_KEY_LEN);
            return executor;
        }
    }
    NnopbaseExecutor* executor = new (std::nothrow) NnopbaseExecutor;
    if (executor != nullptr) {
        aclnnStatus ret = NnopbaseExecutorInit(executor,
                                               {inputsDesc, inputNum, outputsDesc, outputNum, attrsDesc, attrsNum});
        if (ret == OK) {
            NnopbaseExecutorSetCollector(executor, gBinCollector);
            executor->space = (NnopbaseExecutorSpace*)space;
            ret = NnopbaseExecutorSetRegInfo(executor, opType);
        }
        if (ret != OK) {
            NnopbaseExecutorDeInit(executor);
            delete executor;
            executor = nullptr;
        }
    }
    if (executor != nullptr) {
        executor->isWork = true;
        executor->poolIndex = op::internal::GetThreadLocalContext().poolIndex_;
        executorSpace->executors.push_back(executor);
        RecordNnopbaseTime(executor, NnopbaseTimeIdx::kAfterCreateExecutor);
        // 留出optype的偏移量后续生成key的时候用
        executor->ownArgs.keyLen = strlen(executor->opType);
        executor->ownArgs.remainKeyLen = NNOPBASE_MAX_ARGS_KEY_LEN - strlen(executor->opType);
    }
    OP_LOGI("Get op %s space %p executor addr %p.", opType, space, executor);
    return executor;
}

aclnnStatus NnopbaseRunForWorkspace(void* executor, uint64_t* workspaceLen)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(workspaceLen);
    return NnopbaseExecutorRunForWorkspace((NnopbaseExecutor*)executor, workspaceLen);
}

aclnnStatus NnopbaseRunWithWorkspace(void* executor, aclrtStream stream, void* workspace, uint64_t workspaceLen)
{
    if (executor == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_NULL_POINTER("Calling the second-phase API of aclnn", "executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (workspaceLen > 0U && workspace == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_NULL_POINTER("Calling the second-phase API of aclnn", "workspace");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    NnopbaseExecutor* nnopExecutor = PtrCastTo<NnopbaseExecutor>(executor);
    RecordNnopbaseTime(nnopExecutor, NnopbaseTimeIdx::kRunWithWsStart);
    op::internal::GetThreadLocalContext().logInfo_.l2ApiName = nnopExecutor->opType;
    OP_LOGI("Run op %s with workspace len %lu bytes, executor addr %p, executor workspace len %lu bytes.",
            nnopExecutor->opType, workspaceLen, nnopExecutor, nnopExecutor->workspaces.length);
    if (workspaceLen < nnopExecutor->workspaces.length) {
        std::string workspaceLenStr = std::to_string(workspaceLen);
        std::string reason = "The passed workspace size " + workspaceLenStr +
                             " does not meet the minimum workspace size " +
                             std::to_string(nnopExecutor->workspaces.length) + " actually required by the operator";
        OP_LOGE_FOR_INVALID_ARGUMENT_WITHOUT_SOLUTION(workspaceLenStr.c_str(), "workspaceLen", reason.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return NnopbaseExecutorRunWithWorkspace(nnopExecutor, stream, workspace, workspaceLen);
}
aclnnStatus NnopbaseAddInput(void* executor, const aclTensor* tensor, const uint32_t index)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    NnopbaseLogTensorInfo("AddInput", tensor, index);
    NnopbaseExecutor* nnopExecutor = PtrCastTo<NnopbaseExecutor>(executor);
    const auto tensors = &nnopExecutor->ownArgs.inputs;
    if (NnopbaseIsV2CacheKeyEnabled(nnopExecutor)) {
        tensors->paramDescs.instances[index].tensor = tensor;
        Indv::CacheKeyBuilder::AppendTensor(&nnopExecutor->ownArgs, tensor);
        return OK;
    } else {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseExecutorUpdateTensorsIndex(tensors, index));
        return NnopbaseExecutorAddTensor(nnopExecutor, tensor, index, true, false);
    }
}

aclnnStatus NnopbaseAddIgnoreContinuesInput(void* executor, const aclTensor* tensor, const uint32_t index)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    NnopbaseLogTensorInfo("AddIgnoreContInput", tensor, index);
    NnopbaseExecutor* nnopExecutor = PtrCastTo<NnopbaseExecutor>(executor);
    const auto tensors = &nnopExecutor->ownArgs.inputs;
    if (NnopbaseIsV2CacheKeyEnabled(nnopExecutor)) {
        tensors->paramDescs.instances[index].ignoreCont = true;
        tensors->paramDescs.instances[index].tensor = tensor;
        Indv::CacheKeyBuilder::AppendTensor(&nnopExecutor->ownArgs, tensor);
        return OK;
    } else {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseExecutorUpdateTensorsIndex(tensors, index));
        return NnopbaseExecutorAddTensor(nnopExecutor, tensor, index, true, true);
    }
}

aclnnStatus NnopbaseAddIntArrayInput(void* executor, const aclIntArray* array, const uint32_t index)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    NnopbaseLogArrayInfo("IntArray", "AddValueDependInput", array, index);
    NnopbaseExecutor* nnopExecutor = PtrCastTo<NnopbaseExecutor>(executor);
    const auto tensors = &nnopExecutor->ownArgs.inputs;
    if (NnopbaseIsV2CacheKeyEnabled(nnopExecutor)) {
        tensors->paramDescs.instances[index].intArray = array;
        auto args = &nnopExecutor->ownArgs;
        if (array != nullptr) {
            Indv::CacheKeyBuilder::AppendValueDependTensor(args, array->GetData(), array->Size(),
                                                           array->Size() * sizeof(int64_t), ge::DataType::DT_INT64);
        } else {
            Indv::CacheKeyBuilder::AppendPlaceHolder(args,
                                                     PtrCastTo<NnopbaseUChar>(args->inputKey.data() + args->keyLen));
        }
        return OK;
    } else {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseExecutorUpdateTensorsIndex(tensors, index));
        return nnopbase::NnopbaseExecutorAddArrayInput<aclIntArray>(tensors, array, index);
    }
}

aclnnStatus NnopbaseAddBoolArrayInput(void* executor, const aclBoolArray* array, const uint32_t index)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    NnopbaseLogArrayInfo("BoolArray", "AddValueDependInput", array, index);
    NnopbaseExecutor* nnopExecutor = PtrCastTo<NnopbaseExecutor>(executor);
    const auto tensors = &nnopExecutor->ownArgs.inputs;
    if (NnopbaseIsV2CacheKeyEnabled(nnopExecutor)) {
        tensors->paramDescs.instances[index].boolArray = array;
        auto args = &nnopExecutor->ownArgs;
        if (array != nullptr) {
            Indv::CacheKeyBuilder::AppendValueDependTensor(args, array->GetData(), array->Size(),
                                                           array->Size() * sizeof(bool), ge::DataType::DT_BOOL);
        } else {
            Indv::CacheKeyBuilder::AppendPlaceHolder(args,
                                                     PtrCastTo<NnopbaseUChar>(args->inputKey.data() + args->keyLen));
        }
        return OK;
    } else {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseExecutorUpdateTensorsIndex(tensors, index));
        return nnopbase::NnopbaseExecutorAddArrayInput<aclBoolArray>(tensors, array, index);
    }
}

aclnnStatus NnopbaseAddFloatArrayInput(void* executor, const aclFloatArray* array, const uint32_t index)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    NnopbaseLogArrayInfo("FloatArray", "AddValueDependInput", array, index);
    NnopbaseExecutor* nnopExecutor = PtrCastTo<NnopbaseExecutor>(executor);
    const auto tensors = &nnopExecutor->ownArgs.inputs;
    if (NnopbaseIsV2CacheKeyEnabled(nnopExecutor)) {
        tensors->paramDescs.instances[index].floatArray = array;
        auto args = &nnopExecutor->ownArgs;
        if (array != nullptr) {
            Indv::CacheKeyBuilder::AppendValueDependTensor(args, array->GetData(), array->Size(),
                                                           array->Size() * sizeof(float), ge::DataType::DT_FLOAT);
        } else {
            Indv::CacheKeyBuilder::AppendPlaceHolder(args,
                                                     PtrCastTo<NnopbaseUChar>(args->inputKey.data() + args->keyLen));
        }
        return OK;
    } else {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseExecutorUpdateTensorsIndex(tensors, index));
        return nnopbase::NnopbaseExecutorAddArrayInput<aclFloatArray>(tensors, array, index);
    }
}

aclnnStatus NnopbaseAddDynamicInput(void* executor, const aclTensorList* tensorList, const uint32_t index)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    NnopbaseLogTensorListInfo("AddDynamicInput", tensorList, index);
    NnopbaseExecutor* nnopExecutor = PtrCastTo<NnopbaseExecutor>(executor);
    const auto tensors = &nnopExecutor->ownArgs.inputs;
    if (NnopbaseIsV2CacheKeyEnabled(nnopExecutor)) {
        tensors->paramDescs.instances[index].tensorList = tensorList;
        Indv::CacheKeyBuilder::AppendTensorList(&nnopExecutor->ownArgs, tensorList);
        return OK;
    } else {
        return NnopbaseExecutorAddDynamicTensors(nnopExecutor, tensorList, index, true);
    }
}

aclnnStatus NnopbaseAddIgnoreContiguousDynamicInput(void* executor, const aclTensorList* tensorList,
                                                    const uint32_t index)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    NnopbaseLogTensorListInfo("AddIgnoreContDynInput", tensorList, index);
    NnopbaseExecutor* nnopExecutor = PtrCastTo<NnopbaseExecutor>(executor);
    const auto tensors = &nnopExecutor->ownArgs.inputs;
    if (NnopbaseIsV2CacheKeyEnabled(nnopExecutor)) {
        tensors->paramDescs.instances[index].ignoreCont = true;
        tensors->paramDescs.instances[index].tensorList = tensorList;
        Indv::CacheKeyBuilder::AppendTensorList(&nnopExecutor->ownArgs, tensorList);
        return OK;
    } else {
        return NnopbaseExecutorAddDynamicTensors(nnopExecutor, tensorList, index, true, true);
    }
}

aclnnStatus NnopbaseAddOutput(void* executor, const aclTensor* tensor, const uint32_t index)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    NnopbaseLogTensorInfo("AddOutput", tensor, index);
    NnopbaseExecutor* nnopExecutor = PtrCastTo<NnopbaseExecutor>(executor);
    const auto tensors = &nnopExecutor->ownArgs.outputs;
    if (NnopbaseIsV2CacheKeyEnabled(nnopExecutor)) {
        tensors->paramDescs.instances[index].tensor = tensor;
        auto args = &nnopExecutor->ownArgs;
        // 输出和属性中间要用占位符隔开
        if (index == 0) {
            Indv::CacheKeyBuilder::AppendPlaceHolder(args,
                                                     PtrCastTo<NnopbaseUChar>(args->inputKey.data() + args->keyLen));
        }
        Indv::CacheKeyBuilder::AppendTensor(args, tensor);
        return OK;
    } else {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseExecutorUpdateTensorsIndex(tensors, index));
        return NnopbaseExecutorAddTensor(nnopExecutor, tensor, index, false, false);
    }
}

aclnnStatus NnopbaseAddOutputShapeDependTensor(void* executor, aclTensor* tensor, const uint32_t index)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    NnopbaseTensors* tensors = &(((NnopbaseExecutor*)executor)->ownArgs.outputs);
    tensors->outPutShapeSize += sizeof(uint64_t) * 9U; // 9 表示1个count和8个dim
    tensors->outPutShapeMap[index] = tensor;           // 覆盖刷新tensor地址
    OP_LOGI("[DFX] AddShapeDependOutputTensor[%u] addr=%p.", index, tensor);
    return NnopbaseAddOutput(executor, tensor, index);
}

aclnnStatus NnopbaseAddDynamicOutput(void* executor, const aclTensorList* tensorList, const uint32_t index)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    NnopbaseLogTensorListInfo("AddDynamicOutput", tensorList, index);
    NnopbaseExecutor* nnopExecutor = PtrCastTo<NnopbaseExecutor>(executor);
    const auto tensors = &nnopExecutor->ownArgs.outputs;
    if (NnopbaseIsV2CacheKeyEnabled(nnopExecutor)) {
        tensors->paramDescs.instances[index].tensorList = tensorList;
        auto args = &nnopExecutor->ownArgs;
        if (index == 0) {
            Indv::CacheKeyBuilder::AppendPlaceHolder(args,
                                                     PtrCastTo<NnopbaseUChar>(args->inputKey.data() + args->keyLen));
        }
        Indv::CacheKeyBuilder::AppendTensorList(args, tensorList);
        return OK;
    } else {
        return NnopbaseExecutorAddDynamicTensors(nnopExecutor, tensorList, index, false);
    }
}

aclnnStatus NnopbaseAddParamName(void* executor, const uint32_t index, const char* name, const bool isInput)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(name);

    NnopbaseTensors* tensors = nullptr;
    if (isInput) {
        tensors = &(((NnopbaseExecutor*)executor)->ownArgs.inputs);
    } else {
        tensors = &(((NnopbaseExecutor*)executor)->ownArgs.outputs);
    }
    NNOPBASE_ASSERT_INDEX_OUT_OF_RANGE(index, "index", tensors->paramDescs.count);
    tensors->paramDescs.instances[index].name = name;
    return OK;
}

aclnnStatus NnopbaseAddAttr(void* executor, void* attrAddr, const size_t attrLen, const size_t index)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(attrAddr);
    OP_LOGI("[DFX] AddAttr[%zu] dtype=Unknown, addr=%p, attrLen=%zu.", index, attrAddr, attrLen);
    return NnopbaseExecutorAddAttr((NnopbaseExecutor*)executor, attrAddr, attrLen, index, 0U, kNnopbaseAttrEnd);
}

aclnnStatus NnopbaseAddAttrWithDtype(void* executor, void* attrAddr, const size_t attrLen, const size_t index,
                                     const NnopbaseAttrDtype dtype)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(attrAddr);
    NnopbaseLogAttrValueInfo(dtype, attrAddr, attrLen, index);
    return NnopbaseExecutorAddAttr((NnopbaseExecutor*)executor, attrAddr, attrLen, index, 0U, dtype);
}

aclnnStatus NnopbaseAddIntArrayAttr(void* executor, const aclIntArray* array, const size_t index)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(array);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    NnopbaseLogArrayInfo("IntArray", "AddAttr", array, index);
    return NnopbaseExecutorAddAttr(PtrCastTo<NnopbaseExecutor>(executor), (PtrCastTo<const void>(array->GetData())),
                                   array->Size() * sizeof(int64_t), index, sizeof(int64_t), kNnopbaseInt);
}

aclnnStatus NnopbaseAddBoolArrayAttr(void* executor, const aclBoolArray* array, const size_t index)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(array);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    NnopbaseLogArrayInfo("BoolArray", "AddAttr", array, index);
    return NnopbaseExecutorAddAttr(PtrCastTo<NnopbaseExecutor>(executor), PtrCastTo<const void>(array->GetData()),
                                   array->Size() * sizeof(bool), index, sizeof(bool), kNnopbaseBool);
}

aclnnStatus NnopbaseAddFloatArrayAttr(void* executor, const aclFloatArray* array, const size_t index)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(array);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    NnopbaseLogArrayInfo("FloatArray", "AddAttr", array, index);
    return NnopbaseExecutorAddAttr(PtrCastTo<NnopbaseExecutor>(executor), PtrCastTo<const void>(array->GetData()),
                                   array->Size() * sizeof(float), index, sizeof(float), kNnopbaseFloat);
}

aclnnStatus NnopbaseAddArrayAttr(void* executor, void* array, const size_t len, const size_t elementSize,
                                 const size_t index)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(array);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    OP_LOGI("[DFX] AddAttr[%zu] dtype=Unknown, len=%zu, elementSize=%zu.", index, len, elementSize);
    return NnopbaseExecutorAddAttr((NnopbaseExecutor*)executor, array, len * elementSize, index, elementSize,
                                   kNnopbaseAttrEnd);
}

aclnnStatus NnopbaseAddArrayAttrWithDtype(void* executor, void* array, const size_t len, const size_t elementSize,
                                          const size_t index, const NnopbaseAttrDtype dtype)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(array);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    NnopbaseLogRawArrayAttrInfo(dtype, array, len, index);
    return NnopbaseExecutorAddAttr((NnopbaseExecutor*)executor, array, len * elementSize, index, elementSize, dtype);
}

aclnnStatus NnopbaseGetAttrAddr(void* executor, const size_t index, void** attrAddr, size_t* attrLen)
{
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(executor, ACLNN_ERR_PARAM_NULLPTR);
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(attrAddr, ACLNN_ERR_PARAM_NULLPTR);
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(attrLen, ACLNN_ERR_PARAM_NULLPTR);
    aclnnStatus ret;
    NnopbaseAttrAddr* attr = nullptr;
    ret = NnopbaseExecutorGetAttr((NnopbaseExecutor*)executor, index, &attr);
    if (attr != nullptr) {
        *attrAddr = const_cast<void*>(attr->addr);
        *attrLen = attr->size;
    }
    return ret;
}

void NnopbaseSetUserHandle(void* executor, void* handle)
{
    if (executor == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_NULL_POINTER("NnopbaseSetUserHandle", "executor");
        return;
    }
    ((NnopbaseExecutor*)executor)->userHandle = handle;
}

void* NnopbaseGetUserHandle(void* executor)
{
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(executor, nullptr);
    return ((NnopbaseExecutor*)executor)->userHandle;
}

void NnopbaseGetTilingData(void* executor, void** tilingData, uint64_t* dataLen)
{
    NNOPBASE_ASSERT_NOTNULL(executor);
    NNOPBASE_ASSERT_NOTNULL(tilingData);
    NNOPBASE_ASSERT_NOTNULL(dataLen);
    NnopbaseExecutor* nnopExecutor = PtrCastTo<NnopbaseExecutor>(executor);
    if (nnopExecutor->hasTiling && (PtrCastTo<NnopbaseTilingData>(nnopExecutor->args) != nullptr)) {
        auto tilingDataPtr = PtrCastTo<NnopbaseTilingData>(nnopExecutor->args->tilingInfo.tilingData);
        if (tilingDataPtr != nullptr) {
            *tilingData = PtrCastTo<void>(tilingDataPtr->GetData());
            *dataLen = static_cast<uint64_t>(tilingDataPtr->GetDataSize());
            return;
        }
    }

    *tilingData = nullptr;
    *dataLen = 0U;
}

void NnopbaseGetInputTensorAddr(void* executor, size_t index, void** addr)
{
    NNOPBASE_ASSERT_NOTNULL(executor);
    NNOPBASE_ASSERT_NOTNULL(addr);
    auto tensors = &(((NnopbaseExecutor*)executor)->args->inputs);
    if (index >= tensors->arrayLen) {
        OP_LOGE_FOR_INVALID_ARGUMENT_INDEX_OUT_OF_RANGE(std::to_string(index).c_str(), "index",
                                                        std::to_string(tensors->arrayLen).c_str());
        return;
    }
    *addr = tensors->extTensors[index].rt2Tensor.GetAddr();
}

void NnopbaseGetOutputTensorAddr(void* executor, size_t index, void** addr)
{
    NNOPBASE_ASSERT_NOTNULL(executor);
    NNOPBASE_ASSERT_NOTNULL(addr);
    auto tensors = &(((NnopbaseExecutor*)executor)->args->outputs);
    if (index >= tensors->arrayLen) {
        OP_LOGE_FOR_INVALID_ARGUMENT_INDEX_OUT_OF_RANGE(std::to_string(index).c_str(), "index",
                                                        std::to_string(tensors->arrayLen).c_str());
        return;
    }
    *addr = tensors->extTensors[index].rt2Tensor.GetAddr();
}

aclnnStatus NnopbaseGetInputTensorCount(const void* const executor, const size_t irIndex, uint32_t* const count)
{
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(executor, ACLNN_ERR_PARAM_NULLPTR);
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(count, ACLNN_ERR_PARAM_NULLPTR);

    const auto& tensors = (PtrCastTo<NnopbaseExecutor>(executor))->args->inputs;
    NNOPBASE_ASSERT_INDEX_OUT_OF_RANGE(irIndex, "irIndex", tensors.paramDescs.count);
    *count = tensors.paramDescs.instances[irIndex].num;
    return OK;
}

aclnnStatus NnopbaseGetOutputTensorCount(const void* const executor, const size_t irIndex, uint32_t* const count)
{
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(executor, ACLNN_ERR_PARAM_NULLPTR);
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(count, ACLNN_ERR_PARAM_NULLPTR);
    const auto& tensors = (PtrCastTo<NnopbaseExecutor>(executor))->args->outputs;
    NNOPBASE_ASSERT_INDEX_OUT_OF_RANGE(irIndex, "irIndex", tensors.paramDescs.count);
    *count = tensors.paramDescs.instances[irIndex].num;
    return OK;
}

static aclnnStatus NnopbaseGetDynamicTensorAddrs(const NnopbaseTensors& tensors, const size_t irIndex,
                                                 void** const addrs, const size_t addrSize, uint32_t* const count)
{
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(addrs, ACLNN_ERR_PARAM_NULLPTR);
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(count, ACLNN_ERR_PARAM_NULLPTR);
    NNOPBASE_ASSERT_INDEX_OUT_OF_RANGE(irIndex, "irIndex", tensors.paramDescs.count);
    CHECK_COND(tensors.paramDescs.instances[irIndex].isDynamic, ACLNN_ERR_PARAM_INVALID,
               "Failed to get tensor address, tensors[%zu] must be dynamic, while isDynamic = %d.", irIndex,
               tensors.paramDescs.instances[irIndex].isDynamic);

    uint32_t num = tensors.paramDescs.instances[irIndex].num;
    OP_LOGI("NnopbaseGetDynamicTensorAddrs: irIndex = %zu, "
            "addrSize = %zu bytes, tensors num = %u.",
            irIndex, addrSize, num);
    if (addrSize < num) {
        num = static_cast<uint32_t>(addrSize);
    }

    const uint32_t startIndex = tensors.paramDescs.instances[irIndex].startIndex;
    for (uint32_t idx = 0U; idx < num; idx++) {
        addrs[idx] = tensors.extTensors[startIndex + idx].rt2Tensor.GetTensorData().GetAddr();
    }

    *count = num;
    return OK;
}

aclnnStatus NnopbaseGetDynamicInputTensorAddrs(const void* const executor, const size_t irIndex, void** const addrs,
                                               const size_t addrSize, uint32_t* const count)
{
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(executor, ACLNN_ERR_PARAM_NULLPTR);
    return NnopbaseGetDynamicTensorAddrs((PtrCastTo<NnopbaseExecutor>(executor))->args->inputs, irIndex, addrs,
                                         addrSize, count);
}

aclnnStatus NnopbaseGetDynamicOutputTensorAddrs(const void* const executor, const size_t irIndex, void** const addrs,
                                                const size_t addrSize, uint32_t* const count)
{
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(executor, ACLNN_ERR_PARAM_NULLPTR);
    return NnopbaseGetDynamicTensorAddrs((PtrCastTo<NnopbaseExecutor>(executor))->args->outputs, irIndex, addrs,
                                         addrSize, count);
}

aclnnStatus NnopbaseSetInputTensorAddr(void* executor, const size_t index, const void* const addr)
{
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(executor, ACLNN_ERR_PARAM_NULLPTR);

    NnopbaseExecutor* nnopExecutor = PtrCastTo<NnopbaseExecutor>(executor);
    auto& tensors = nnopExecutor->args->inputs;
    NNOPBASE_ASSERT_INDEX_OUT_OF_RANGE(index, "index", tensors.num);
    if (tensors.extTensors[index].isNull) {
        std::string errMsg = "The " + std::to_string(index) + "th input in operator " + nnopExecutor->opType +
                             " is a null pointer. The address cannot be updated through the "
                             "aclSetTensorAddr/aclSetInputTensorAddr API";
        OP_LOGE_FOR_EXECUTION_ERROR_WITHOUT_SOLUTION(errMsg.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    NNOPBASE_ASSERT_OK_RETVAL(tensors.extTensors[index].rt2Tensor.MutableTensorData().SetAddr(addr, nullptr));
    return NnopbaseSetRefTensorAddr(nnopExecutor, index, addr, tensors.paramDescs.instances,
                                    nnopExecutor->args->outputs.paramDescs.instances);
}

aclnnStatus NnopbaseSetOutputTensorAddr(void* executor, const size_t index, const void* const addr)
{
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(executor, ACLNN_ERR_PARAM_NULLPTR);

    NnopbaseExecutor* nnopExecutor = PtrCastTo<NnopbaseExecutor>(executor);
    auto& tensors = nnopExecutor->args->outputs;
    NNOPBASE_ASSERT_INDEX_OUT_OF_RANGE(index, "index", tensors.num);
    if (tensors.extTensors[index].isNull) {
        std::string errMsg = "The " + std::to_string(index) + "th output in operator " + nnopExecutor->opType +
                             " is a null pointer. The address cannot be updated through the "
                             "aclSetTensorAddr/aclSetOutputTensorAddr API";
        OP_LOGE_FOR_EXECUTION_ERROR_WITHOUT_SOLUTION(errMsg.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    NNOPBASE_ASSERT_OK_RETVAL(tensors.extTensors[index].rt2Tensor.MutableTensorData().SetAddr(addr, nullptr));
    return OK;
}

aclnnStatus NnopbaseSetDynamicInputTensorAddr(void* executor, size_t irIndex, const size_t relativeIndex,
                                              const void* const addr)
{
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(executor, ACLNN_ERR_PARAM_NULLPTR);

    NnopbaseExecutor* nnopExecutor = PtrCastTo<NnopbaseExecutor>(executor);
    auto& tensors = nnopExecutor->args->inputs;
    NNOPBASE_ASSERT_INDEX_OUT_OF_RANGE(irIndex, "irIndex", tensors.paramDescs.count);
    NNOPBASE_ASSERT_INDEX_OUT_OF_RANGE(relativeIndex, "relativeIndex", tensors.paramDescs.instances[irIndex].num);

    const size_t index = static_cast<size_t>(tensors.paramDescs.instances[irIndex].startIndex) + relativeIndex;
    if (tensors.extTensors[index].isNull) {
        std::string
            errMsg = "The " + std::to_string(irIndex) + "th tensor in the " + std::to_string(relativeIndex) +
                     "th input tensorList in operator " + nnopExecutor->opType +
                     " is a null pointer. The address cannot be updated through"
                     " the aclSetDynamicInputTensorAddr/AclSetDynamicInputTensorAddr/aclSetDynamicTensorAddr API";
        OP_LOGE_FOR_EXECUTION_ERROR_WITHOUT_SOLUTION(errMsg.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    NNOPBASE_ASSERT_OK_RETVAL(tensors.extTensors[index].rt2Tensor.MutableTensorData().SetAddr(addr, nullptr));
    auto& inInstances = tensors.paramDescs.instances;
    if (inInstances[irIndex].refIdx != -1) {
        NnopbaseSetDynamicOutputTensorAddr(executor, inInstances[irIndex].refIdx, relativeIndex, addr);
    }
    return OK;
}

aclnnStatus NnopbaseSetDynamicOutputTensorAddr(void* executor, size_t irIndex, const size_t relativeIndex,
                                               const void* const addr)
{
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(executor, ACLNN_ERR_PARAM_NULLPTR);

    NnopbaseExecutor* nnopExecutor = PtrCastTo<NnopbaseExecutor>(executor);
    auto& tensors = nnopExecutor->args->outputs;
    NNOPBASE_ASSERT_INDEX_OUT_OF_RANGE(irIndex, "irIndex", tensors.paramDescs.count);
    NNOPBASE_ASSERT_INDEX_OUT_OF_RANGE(relativeIndex, "relativeIndex", tensors.paramDescs.instances[irIndex].num);
    const size_t index = static_cast<size_t>(tensors.paramDescs.instances[irIndex].startIndex) + relativeIndex;
    if (tensors.extTensors[index].isNull) {
        std::string
            errMsg = "The " + std::to_string(irIndex) + "th tensor in the " + std::to_string(relativeIndex) +
                     "th output tensorList in operator " + nnopExecutor->opType +
                     " is a null pointer. The address cannot be updated through"
                     " the aclSetDynamicOutputTensorAddr/AclSetDynamicOutputTensorAddr/aclSetDynamicTensorAddr API";
        OP_LOGE_FOR_EXECUTION_ERROR_WITHOUT_SOLUTION(errMsg.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    NNOPBASE_ASSERT_OK_RETVAL(tensors.extTensors[index].rt2Tensor.MutableTensorData().SetAddr(addr, nullptr));
    return OK;
}

aclnnStatus NnopbaseDisableOptionalInput(void* executor, const size_t irIndex)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    NnopbaseExecutor* nnopExecutor = PtrCastTo<NnopbaseExecutor>(executor);
    auto& tensors = nnopExecutor->args->inputs;
    NNOPBASE_ASSERT_INDEX_OUT_OF_RANGE(irIndex, "irIndex", tensors.paramDescs.count);
    tensors.paramDescs.instances[irIndex].isDisable = true;
    tensors.paramDescs.activeInputCount--;
    return OK;
}

uint64_t NnopbaseMsprofSysTime()
{
    if (op::internal::opProfilingSwitch.reportFlag || op::internal::opProfilingSwitch.kernelLaunchFlag ||
        op::internal::opProfilingSwitch.additionInfoFlag || op::internal::opProfilingSwitch.level2ProfilingFlag ||
        op::internal::opProfilingSwitch.timeStampFlag) {
        return MsprofSysCycleTime();
    }
    return 0U;
}

aclnnStatus NnopbaseAddTilingId(void* executor, NnopbaseDfxId* tilingId)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(tilingId);
    NnopbaseExecutorAddTilingId((NnopbaseExecutor*)executor, tilingId);
    return OK;
}

aclnnStatus NnopbaseAddSupportList(void* executor, OpSupportList* list, uint32_t* socSupportList,
                                   size_t socSupportListLen)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(list);
    NnopbaseExecutorAddSupportList(PtrCastTo<NnopbaseExecutor>(executor)->opCompatibility, list, socSupportList,
                                   socSupportListLen);
    return OK;
}

void NnopbaseAddOpTypeId(void* executor, const uint32_t opTypeId)
{
    NNOPBASE_ASSERT_NOTNULL(executor);
    NnopbaseExecutorAddOpTypeId((NnopbaseExecutor*)executor, opTypeId);
}

uint32_t NnopbaseGetSocEnum() { return nnopbase::IndvSoc::GetInstance().GetSocEnum(); }

const char* NnopbaseGetSocName()
{
    thread_local std::string socName = nnopbase::IndvSoc::GetInstance().GetCurSocVersion();
    return socName.c_str();
}

static void NnopbaseConvertSocNamesToEnums(const nnopbase::IndvSoc& indvSoc, const char* const* socNameList,
                                           size_t socNameListLen, uint32_t* enumList)
{
    for (size_t i = 0U; i < socNameListLen; i++) {
        enumList[i] = indvSoc.GetSocEnumByName(socNameList[i]);
        if (enumList[i] == nnopbase::SOC_VERSION_INVALID) {
            OP_LOGW("NnopbaseAddSocNameList: soc name [%s] is not supported, converted to SOC_VERSION_INVALID.",
                    socNameList[i]);
        }
    }
}

static aclnnStatus NnopbaseCheckCurrentSocMatch(const char* const* socNameList, size_t socNameListLen)
{
    const std::string& curSocVersion = nnopbase::IndvSoc::GetInstance().GetCurSocVersion();
    for (size_t i = 0U; i < socNameListLen; i++) {
        if (socNameList[i] != nullptr && curSocVersion == socNameList[i]) {
            return OK;
        }
    }
    std::string
        errMsg = "SoC version " + curSocVersion +
                 " verification failed. This SoC is not configured through the AddConfig API of the OpDef class";
    OP_LOGE_FOR_EXECUTION_ERROR_WITHOUT_SOLUTION(errMsg.c_str());
    return ACLNN_ERR_PARAM_INVALID;
}

aclnnStatus NnopbaseAddSocNameList(void* executor, OpSupportList* list, const char* const* socNameList,
                                   size_t socNameListLen)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(list);
    if (socNameList == nullptr || socNameListLen == 0U) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "socNameList is nullptr or socNameListLen is 0.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    NNOPBASE_ASSERT_RTOK_RETVAL(NnopbaseCheckCurrentSocMatch(socNameList, socNameListLen));
    NnopbaseExecutor* nnopExecutor = PtrCastTo<NnopbaseExecutor>(executor);
    auto& opCompatibility = nnopExecutor->opCompatibility;
    if (opCompatibility.supportedSocList != nullptr &&
        opCompatibility.supportedSocCount == static_cast<uint32_t>(socNameListLen) &&
        opCompatibility.ownsSupportedSocList) {
        NnopbaseExecutorAddSupportList(opCompatibility, list, opCompatibility.supportedSocList, socNameListLen, true);
        return OK;
    }
    const auto& indvSoc = nnopbase::IndvSoc::GetInstance();
    if (socNameListLen > MAX_SOC_NAME_LIST_LEN) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "socNameListLen %zu is too large, max length is %zu.", socNameListLen,
                MAX_SOC_NAME_LIST_LEN);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (opCompatibility.ownsSupportedSocList) {
        delete[] opCompatibility.supportedSocList;
    }
    opCompatibility.supportedSocList = nullptr;
    opCompatibility.ownsSupportedSocList = false;
    auto* enumList = new (std::nothrow) uint32_t[socNameListLen];
    NNOPBASE_ASSERT_NOTNULL_RETVAL(enumList);
    for (size_t i = 0U; i < socNameListLen; i++) {
        if (socNameList[i] == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "socNameList[%zu] is nullptr.", i);
            delete[] enumList;
            return ACLNN_ERR_PARAM_NULLPTR;
        }
    }
    NnopbaseConvertSocNamesToEnums(indvSoc, socNameList, socNameListLen, enumList);
    NnopbaseExecutorAddSupportList(opCompatibility, list, enumList, socNameListLen, true);
    return OK;
}

aclnnStatus NnopbaseGetUnContiguousTensors(void* executor, const aclTensorList** inTensors)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(inTensors);
    NnopbaseExecutorGetUnContiguousTensors((NnopbaseExecutor*)executor, inTensors);
    return OK;
}

aclnnStatus NnopbaseSetUnContExecutor(void* executor, aclOpExecutor* inExe, const size_t inWsSize)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    NnopbaseExecutorSetUnContExecutor((NnopbaseExecutor*)executor, inExe, inWsSize);
    return OK;
}

aclnnStatus NnopbaseGetUnContExecutor(void* executor, aclOpExecutor** inExe, size_t* inWsSize)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(inExe);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(inWsSize);
    NnopbaseExecutorGetUnContExecutor((NnopbaseExecutor*)executor, inExe, inWsSize);
    return OK;
}

aclnnStatus NnopbaseGetRefUnContiguousTensors(void* executor, const aclTensorList** unContTensors,
                                              const aclTensorList** contTensors)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(unContTensors);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(contTensors);
    NnopbaseExecutorGetRefUnContiguousTensors((NnopbaseExecutor*)executor, unContTensors, contTensors);
    return OK;
}

aclnnStatus NnopbaseSetRef(void* executor, const size_t inputIrIdx, const size_t outputIrIdx)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    return NnopbaseExecutorSetRef((NnopbaseExecutor*)executor, inputIrIdx, outputIrIdx);
}

aclnnStatus NnopbaseSetViewCopyExecutor(void* executor, aclOpExecutor* exe)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    NnopbaseExecutorSetViewCopyExecutor((NnopbaseExecutor*)executor, exe);
    return OK;
}

aclnnStatus NnopbaseGetViewCopyExecutor(void* executor, aclOpExecutor** exe)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(exe);
    NnopbaseExecutorGetViewCopyExecutor((NnopbaseExecutor*)executor, exe);
    return OK;
}

aclnnStatus NnopbaseReleaseRefContiguousTensors(void* executor, const aclTensorList** tensors)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(tensors);
    NnopbaseExecutorReleaseRefContiguousTensors((NnopbaseExecutor*)executor, tensors);
    return OK;
}

aclnnStatus NnopbaseAddScalarInput(void* executor, const aclScalar* scalar, const uint32_t index,
                                   const int32_t srcIndex, const ge::DataType dtype)
{
    OP_LOGD("NnopbaseAddScalarInput Start.");
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    NnopbaseLogScalarInfo("AddScalarInput", scalar, index);
    NnopbaseExecutor* nnopExecutor = PtrCastTo<NnopbaseExecutor>(executor);
    const auto tensors = &nnopExecutor->ownArgs.inputs;
    if (NnopbaseIsV2CacheKeyEnabled(nnopExecutor)) {
        tensors->paramDescs.instances[index].scalar = scalar;
        tensors->paramDescs.instances[index].scalarDtype = dtype;
        tensors->paramDescs.instances[index].srcIndex = srcIndex;
        Indv::CacheKeyBuilder::AppendScalar(&nnopExecutor->ownArgs, scalar, index, srcIndex, dtype);
        return OK;
    } else {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseExecutorUpdateTensorsIndex(tensors, index));
        return NnopbaseExecutorAddScalarInput(tensors, scalar, index, srcIndex, dtype);
    }
}

aclnnStatus NnopbaseAddScalarListInput(void* executor, const aclScalarList* scalarList, const uint32_t index,
                                       const int32_t srcIndex, const ge::DataType dtype)
{
    OP_LOGD("NnopbaseAddScalarListInput Start.");
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    NnopbaseLogScalarListInfo("AddScalarListInput", scalarList, index);
    NnopbaseExecutor* nnopExecutor = PtrCastTo<NnopbaseExecutor>(executor);
    const auto tensors = &nnopExecutor->ownArgs.inputs;
    if (NnopbaseIsV2CacheKeyEnabled(nnopExecutor)) {
        tensors->paramDescs.instances[index].scalarList = scalarList;
        tensors->paramDescs.instances[index].scalarDtype = dtype;
        tensors->paramDescs.instances[index].srcIndex = srcIndex;
        Indv::CacheKeyBuilder::AppendScalarList(&nnopExecutor->ownArgs, scalarList, index, srcIndex, dtype);
        return OK;
    } else {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseExecutorUpdateTensorsIndex(tensors, index));
        return NnopbaseExecutorAddScalarListInput(tensors, scalarList, index, srcIndex, dtype);
    }
}

const NnopbaseChar* NnopbaseFindStaticKernel(const aclTensor* tensors[], const NnopbaseAttrAddr* attrs[],
                                             const int64_t valueDepend[],
                                             const NnopbaseStaticTensorNumInfo* const tensorNumInfo,
                                             const NnopbaseStaticRuntimeInfo* const staticRuntimeInfo)
{
    {
        const std::lock_guard<std::mutex> lock(g_nnopbaseInitMtx);
        if (NnopbaseInit() != OK) {
            return nullptr;
        }
    }
    thread_local static NnopbaseUChar verbose[NNOPBASE_MAX_STATICKEY_LEN];
    NnopbaseUChar* verKey1 = verbose;
    NnopbaseRegInfoKey regInfoKey;
    regInfoKey.opType = staticRuntimeInfo->opType;
    regInfoKey.hashKey = static_cast<uint64_t>(
        NnopbaseHashBinary(PtrCastTo<NnopbaseUChar>(regInfoKey.opType.c_str()), regInfoKey.opType.size()) %
        NNOPBASE_NORM_MAX_BIN_BUCKETS);
    OP_LOGI("OpType is %s, hashkey is %lu.", regInfoKey.opType.c_str(), regInfoKey.hashKey);

    verKey1 = NnopbaseCollectorGenStaticKey(verKey1, &regInfoKey, tensorNumInfo, tensors, attrs,
                                            staticRuntimeInfo->implMode, staticRuntimeInfo->deterMode, valueDepend,
                                            true);

    StaticKernelPlatformInfo platformInfo{{staticRuntimeInfo->aicNum, staticRuntimeInfo->aivNum},
                                          static_cast<int8_t>(staticRuntimeInfo->deterMode)};
    auto simplifiedKey = NnopbaseCollectorGetStaticKernelBin(regInfoKey.opType.c_str(), regInfoKey.hashKey, verbose,
                                                             uint32_t(verKey1 - verbose), &platformInfo);
    if (simplifiedKey != nullptr) {
        return simplifiedKey;
    } else {
        OP_LOGW("Cannot find static kernel bin with stride information, trying to find without stride again.");
        NnopbaseUChar* verKey2 = verbose;
        verKey2 = NnopbaseCollectorGenStaticKey(verKey2, &regInfoKey, tensorNumInfo, tensors, attrs,
                                                staticRuntimeInfo->implMode, staticRuntimeInfo->deterMode, valueDepend,
                                                false);
        return NnopbaseCollectorGetStaticKernelBin(regInfoKey.opType.c_str(), regInfoKey.hashKey, verbose,
                                                   uint32_t(verKey2 - verbose), &platformInfo);
    }
}

aclnnStatus NnopbaseGetStreamAndEvent(const aclrtStream stream, aclrtStream* subStream, aclrtEvent* evtA,
                                      aclrtEvent* evtB, std::shared_ptr<std::mutex>& streamLckPtr)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(subStream);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(evtA);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(evtB);
    return NnopbaseExecutorGetStreamAndEvent(stream, subStream, evtA, evtB, streamLckPtr);
}

// mc2
aclnnStatus NnopbaseSetMc2(void* const executor)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    ((NnopbaseExecutor*)executor)->mc2.enabled = true;
    bool needLoadDavidHcclApi = nnopbase::IndvSoc::GetInstance().SupportMc2FusionLaunch();
    if (needLoadDavidHcclApi) {
        const static aclnnStatus retVal = nnopbase::IndvMc2ClientWrapper::GetInstance().IndvMc2ClientWrapperInit(
            LIBMC2_CLIENT_SO);
        if (retVal != OK) {
            OP_LOGW("Failed to load libmc2_client.so, ret = %d.", retVal);
        }
    }
    const static aclnnStatus ret = nnopbase::IndvHcclWrapper::GetInstance().IndvHcclWrapperInit(LIBHCCL_SO,
                                                                                                needLoadDavidHcclApi);
    CHECK_COND((ret == OK), ret, "Failed to load hccl module libhccl.so, ret = %d.", ret);
    return OK;
}

aclnnStatus NnopbaseSetHcomGroup(void* const executor, char* const group)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    CHECK_COND(((NnopbaseExecutor*)executor)->mc2.enabled, ACLNN_ERR_PARAM_INVALID,
               "Failed to set Hcom group for %s, group is %p.", ((NnopbaseExecutor*)executor)->opType, group);
    if ((group == nullptr) || strcmp(group, "") == 0) {
        ((NnopbaseExecutor*)executor)->mc2.commHandles.push_back(nullptr);
    } else {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseSetMc2Tiling(((NnopbaseExecutor*)executor), group));
    }
    return OK;
}

void* NnopbaseGetApiFunc(const char* funcName) { return nnopbase::ApiWrapper::GetInstance().GetFunc(funcName); }

aclnnStatus NnopbaseSetFormatMatchMode(void* executor, const uint32_t mode)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    ((NnopbaseExecutor*)executor)->formatCheckOption = static_cast<NnopbaseFormatCheckOption>(mode);
    return OK;
}

void NnopbaseSetHcclServerType(void* executor, NnopbaseHcclServerType sType)
{
    NNOPBASE_ASSERT_NOTNULL(executor);
    NnopbaseExecutorSetHcclServerType((NnopbaseExecutor*)executor, sType);
}

aclnnStatus NnopbaseSetHcclServerTypeBySocName(void* executor, NnopbaseHcclServerType* hcclServerTypeList,
                                               const char* const* socNameList, size_t socNameListLen)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(hcclServerTypeList);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(socNameList);
    NnopbaseExecutor* nnopExecutor = PtrCastTo<NnopbaseExecutor>(executor);
    const std::string& curSocVersion = nnopbase::IndvSoc::GetInstance().GetCurSocVersion();
    if (!nnopbase::IndvSoc::GetInstance().SupportCurrentSoc()) {
        OP_LOGW("HcclServerType API does not support this SoC version %s.", curSocVersion.c_str());
        return OK;
    }
    for (size_t i = 0U; i < socNameListLen; i++) {
        if (socNameList[i] != nullptr && curSocVersion == socNameList[i]) {
            NnopbaseExecutorSetHcclServerType(nnopExecutor, hcclServerTypeList[i]);
            return OK;
        }
    }
    std::string
        errMsg = "SoC version " + curSocVersion +
                 " verification failed. This SoC is not configured through the HcclServerType API of the OpDef class";
    OP_LOGE_FOR_EXECUTION_ERROR_WITHOUT_SOLUTION(errMsg.c_str());
    return ACLNN_ERR_PARAM_INVALID;
}

void NnopbaseSetHcclServerTypeList(void* executor, NnopbaseHcclServerType* hcclServerTypeList,
                                   const uint32_t* socSupportList, size_t socSupportListLen)
{
    NNOPBASE_ASSERT_NOTNULL(executor);
    NNOPBASE_ASSERT_NOTNULL(hcclServerTypeList);
    NNOPBASE_ASSERT_NOTNULL(socSupportList);
    NnopbaseExecutor* nnopExecutor = PtrCastTo<NnopbaseExecutor>(executor);
    const auto supportSocMap = nnopbase::IndvSoc::GetInstance().GetSocTypeMap();
    const std::string socVersion = nnopbase::IndvSoc::GetInstance().GetCurSocVersion();
    const auto& iter = supportSocMap.find(socVersion);
    if (iter == supportSocMap.cend()) {
        return;
    }
    for (size_t i = 0U; i < socSupportListLen; i++) {
        if (socSupportList[i] == iter->second) {
            NnopbaseExecutorSetHcclServerType(nnopExecutor, hcclServerTypeList[i]);
            return;
        }
    }
}

void NnopbaseSetZeroEleOutputLaunchFlag(void* executor)
{
    NNOPBASE_ASSERT_NOTNULL(executor);
    ((NnopbaseExecutor*)executor)->isZeroEleOutputLaunch = true;
}

void NnopbaseSetParamCheckMode(void* executor, const uint32_t mode)
{
    NNOPBASE_ASSERT_NOTNULL(executor);
    NnopbaseExecutor* nnopExecutor = PtrCastTo<NnopbaseExecutor>(executor);
    if (static_cast<uint32_t>(NnopbaseParamCheckMode::kCheckEnd) > mode) {
        nnopExecutor->paramCheckMode = static_cast<NnopbaseParamCheckMode>(mode);
    } else {
        OP_LOGW("Failed to set paramCheckMode. Value of mode is %u, which must be between [0, %u).", mode,
                static_cast<uint32_t>(NnopbaseParamCheckMode::kCheckEnd));
    }
}

void NnopbaseSetMatchArgsFlag(void* executor)
{
    NNOPBASE_ASSERT_NOTNULL(executor);
    ((NnopbaseExecutor*)executor)->matchArgsV2 = true;
}

bool NnopbaseMatchArgs(void* executor, uint64_t* workspaceLen)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(workspaceLen);
    NnopbaseExecutor* nnopExecutor = PtrCastTo<NnopbaseExecutor>(executor);
    RecordNnopbaseTime(nnopExecutor, NnopbaseTimeIdx::kGetWsStart);
    nnopbase::NnopbaseGetCoreNum(&nnopExecutor->coreNum.aicNum, &nnopExecutor->coreNum.aivNum);
    nnopExecutor->deterministicLevel = static_cast<uint8_t>(nnopbase::GetGlobalDeterministic());
    nnopExecutor->deterministic = nnopExecutor->deterministicLevel == 0U ? false : true;
    NnopbaseUpdatePlatformInfo(nnopExecutor);
    if ((!g_nnopbaseSysCfgParams.enableArgsCache) || op::internal::GetOpProfilingRecordArgFlag()) {
        nnopExecutor->ownArgs.enableCache = false;
        OP_LOGI("Op %s does not enable match args cache.", nnopExecutor->opType);
        RecordNnopbaseTime(nnopExecutor, NnopbaseTimeIdx::kMatchCacheStart);
        RecordNnopbaseTime(nnopExecutor, NnopbaseTimeIdx::kMatchCacheEnd);
        return false;
    }
    uint8_t* key = nnopExecutor->ownArgs.inputKey.data();
    key = PtrCastTo<NnopbaseUChar>(
        NnopbaseAppendBinary(key, strlen(nnopExecutor->opType), nnopExecutor->opType, strlen(nnopExecutor->opType)));
    key = Indv::CacheKeyBuilder::AppendCoreNum(&nnopExecutor->ownArgs, &nnopExecutor->coreNum);
    uint32_t mc2RankId = nnopbase::utils::ThreadVarContainer::GetCurMc2RankIdInThread();
    key = Indv::CacheKeyBuilder::AppendMc2RankId(&nnopExecutor->ownArgs, &mc2RankId);
    key = Indv::CacheKeyBuilder::AppendDeterministicLevel(&nnopExecutor->ownArgs, &(nnopExecutor->deterministicLevel));
    nnopExecutor->ownArgs.seed = NnopbaseHashBinary(PtrCastTo<NnopbaseUChar>(nnopExecutor->ownArgs.inputKey.data()),
                                                    nnopExecutor->ownArgs.keyLen);
    if (nnopbase::ArgsPool::GetInstance().MatchArgs(nnopExecutor)) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseUpdateInputAddr(nnopExecutor));
        NNOPBASE_ASSERT_OK_RETVAL(
            NnopbaseUpdateOutputAddr(&nnopExecutor->args->outputs, &nnopExecutor->ownArgs.outputs));
        NnopbaseCheckHasContiguous(nnopExecutor);
        if (!nnopExecutor->args->inputs.unContiguousTensors.tensors.empty()) {
            OP_LOGI("Executor addr %p, op %s has %zu uncontiguous inputs.", nnopExecutor, nnopExecutor->opType,
                    nnopExecutor->args->inputs.unContiguousTensors.tensors.size());
            return false;
        }
        NnopbaseSetCachedInfo(nnopExecutor);
        *workspaceLen = nnopExecutor->workspaces.length;
        OP_LOGI("Executor addr %p, op %s match args cache successfully, workspaceLen is %lu.", nnopExecutor,
                nnopExecutor->opType, *workspaceLen);
        NnopbaseExecutorCopyCacheAttr(nnopExecutor);
        return true;
    }
    OP_LOGI("Executor addr %p, op %s does not match args cache.", nnopExecutor, nnopExecutor->opType);
    CHECK_COND(NnopbaseAddIoTensors(nnopExecutor) == OK, false, "Failed to add input&output tensors for operator %s.",
               nnopExecutor->opType);
    return false;
}

bool NnopbaseSupportTensorV2() { return true; }

#ifdef __cplusplus
}
#endif
