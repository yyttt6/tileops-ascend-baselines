/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "indv_executor.h"
#include <string>
#include <mutex>
#include "indv_args_pool.h"
#include "utils/thread_var_container.h"
#include "utils/indv_soc.h"
#include "indv_tilingcontext_builder.h"
#include "graph/ge_error_codes.h"
#include "thread_local_context.h"
#include "opdev/format_utils.h"
#include "op_dfx_internal.h"
#include "runtime/runtime/rt_ffts.h"
#include "exe_graph/runtime/tensor_data.h"
#include "mmpa/mmpa_api.h"
#include "indv_mc2_aicpu.h"
#include "opdev/data_type_utils.h"
#include "kernel_utils.h"
#include "bridge_dfx.h"
#include "acl/acl_rt.h"
#ifndef PRODUCT_SIDE_IS_DEVICE
#include "version/runtime_version.h"
#include "version/metadef_version.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

NnopbaseSysGlobalParams g_nnopbaseSysCfgParams = {nullptr, false, 0U, true, false, "", false};
static std::mutex g_nnopbaseRegisterMtx;
#define CUSTOM_EXCEPTION_CALL_BACK_SUPPORT_VER 90000000
namespace {
constexpr uint32_t NNOPBASE_MAX_TENSOR_NUM = 50U;
constexpr int32_t NNOPBASE_MODULE_TYPE_AICORE = 4;
constexpr int32_t NNOPBASE_MODULE_TYPE_VECTOR_CORE = 7;
static constexpr size_t NNOPBASE_GERT_SHAPE_MAX_DIMS = 8U;
std::vector<aclrtLaunchKernelAttr> CreateRtsLaunchCfgAttrs(uint8_t scheMode, uint32_t dynUbufSize, bool is195x)
{
    std::vector<aclrtLaunchKernelAttr> attrs;
    aclrtLaunchKernelAttr schemModeAttr = {
        .id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE,
        .value = {.schemMode = scheMode},
    };
    attrs.push_back(schemModeAttr);
    if (is195x) {
        aclrtLaunchKernelAttr engineTypeAttr = {
            .id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE,
            .value = {.engineType = ACL_RT_ENGINE_TYPE_AIC},
        };
        attrs.push_back(engineTypeAttr);
        aclrtLaunchKernelAttr numBlocksOffsetAttr = {
            .id = ACL_RT_LAUNCH_KERNEL_ATTR_BLOCKDIM_OFFSET,
            .value = {.blockDimOffset = 0},
        };
        attrs.push_back(numBlocksOffsetAttr);
    } else {
        aclrtLaunchKernelAttr simtModeAttr = {
            .id = static_cast<aclrtLaunchKernelAttrId>(INDV_LAUNCH_KERNEL_RTS_DYN_UBUF_SIZE),
            .value = {.localMemorySize = dynUbufSize},
        };
        attrs.push_back(simtModeAttr);
    }
    return attrs;
}

aclnnStatus RegisterCustomizedCallback(NnopbaseExecutor* const executor, NnopbaseBinInfo* const binInfo)
{
#ifndef PRODUCT_SIDE_IS_DEVICE
#if defined(RUNTIME_VERSION_NUM) && defined(METADEF_VERSION_NUM) &&    \
    (RUNTIME_VERSION_NUM >= CUSTOM_EXCEPTION_CALL_BACK_SUPPORT_VER) && \
    (METADEF_VERSION_NUM >= CUSTOM_EXCEPTION_CALL_BACK_SUPPORT_VER)
    int32_t metadefVerNum = -1;
    int32_t runtimeVerNum = -1;
    auto aclRetForMetadef = aclsysGetVersionNum("metadef", &metadefVerNum);
    auto aclRetForRts = aclsysGetVersionNum("runtime", &runtimeVerNum);
    if (aclRetForMetadef == ACL_SUCCESS && aclRetForRts == ACL_SUCCESS &&
        metadefVerNum >= CUSTOM_EXCEPTION_CALL_BACK_SUPPORT_VER &&
        runtimeVerNum >= CUSTOM_EXCEPTION_CALL_BACK_SUPPORT_VER) {
        auto& registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
        NNOPBASE_ASSERT_NOTNULL_RETVAL(registry);
        auto opImpl = registry->GetOpImpl(executor->opType);
        if (opImpl != nullptr && opImpl->exception_func != nullptr) {
            OP_LOGI("Start to register customized exception dump parse function for op: %s.", executor->opType);
            auto inst = nnopbase::IndvRtsWrapper::GetInstance();
            if (inst == nullptr) {
                return ACLNN_ERR_INNER;
            }
            NNOPBASE_ASSERT_OK_RETVAL(inst->AclrtBinarySetExceptionCallback(
                binInfo->binHandle, op::internal::PtrCastTo<void>(opImpl->exception_func), nullptr));
            OP_LOGI("Register customized exception dump parse function for op: %s successfully.", executor->opType);
        }
    }
#endif
#endif
    return OK;
}

void ResizeExecutorArgsBuf(NnopbaseExecutor* executor)
{
    OP_LOGI("Start resizing %s args buffer.", executor->opType);
    const uint32_t ioNum = executor->args->inputs.num + executor->args->outputs.num;
    size_t argsLen = NNOPBASE_MAX_ARGS_BUF_LEN;
    const size_t tilingDataSize = executor->args->binInfo->opParaSize == 0U ? NNOPBASE_MAX_TILING_DATA_LEN :
                                                                              executor->args->binInfo->opParaSize;
    // 3k包括2k的输入输出、oom、dynamic输入的shape信息和rtHostInputInfo_t结构体
    const size_t argSize = executor->args->inputs.hostInputSize + tilingDataSize + NNOPBASE_HOST_DATA_LEN;
    OP_LOGI("Tiling data size is %zu, argSize is %zu, hostInputSize is %zu.", tilingDataSize, argSize,
            executor->args->inputs.hostInputSize);
    if ((ioNum > NNOPBASE_MAX_TENSOR_NUM) || (argSize > argsLen) || (tilingDataSize > NNOPBASE_MAX_TILING_DATA_LEN)) {
        argsLen = NnopbaseCalcArgsSize(executor, tilingDataSize);
    }
    if (argsLen > executor->args->argsBuf.size()) {
        executor->args->argsBuf.resize(argsLen);
    }
    OP_LOGI("Finish resizing %s args buffer.", executor->opType);
}

bool GenStaticKeyAndFindStaticBin(NnopbaseExecutor* executor, bool usingStride)
{
    if (NnopbaseExecutorGenStaticKey(executor, usingStride) != OK) {
        return false;
    }
    // 已在缓存匹配流程调用控核接口，保证executor的coreNum/determisiticLevel不为非法值
    StaticKernelPlatformInfo platformInfo{NnopbaseCoreNum{executor->coreNum.aicNum, executor->coreNum.aivNum},
                                          static_cast<int8_t>(executor->deterministicLevel)};
    auto binInfo = NnopbaseCollectorFindBinInfo(executor->regInfo, executor->binInfoKey.hashKey,
                                                &(executor->binInfoKey.verbose[0U]), executor->binInfoKey.len,
                                                &platformInfo);
    if ((binInfo != nullptr) && (NnopbaseKernelRegister(executor, binInfo) == OK)) {
        executor->args->binInfo = binInfo;
        executor->hasTiling = false;
        return true;
    }
    return false;
}

void UpdateStaticKernelTilingInfo(NnopbaseExecutor* executor)
{
    OP_LOGI("Start updating %s tilingInfo.", executor->opType);
    auto staticBinInfo = executor->args->binInfo;
    if (staticBinInfo->extraKernelDesc == nullptr || staticBinInfo->extraKernelDesc->runInfo == nullptr) {
        return;
    }
    auto& staticKernelRunInfo = staticBinInfo->extraKernelDesc->runInfo;
    ResizeExecutorArgsBuf(executor);
    auto& tilingInfo = executor->args->tilingInfo;
    tilingInfo.tilingKey = staticKernelRunInfo->tilingKey;
    tilingInfo.numBlocks = staticKernelRunInfo->numBlocks;
    tilingInfo.scheMode = static_cast<uint8_t>(staticKernelRunInfo->scheduleMode);
    tilingInfo.aicpuNumBlocks = staticKernelRunInfo->aicpuNumBlocks;
    tilingInfo.needAtomic = staticKernelRunInfo->clearAtomic;
    tilingInfo.staticTilingData = staticKernelRunInfo->tilingData;
    OP_LOGI("Updated static tilingInfo: opType %s, aicpuNumBlocks %u, numBlocks %u,"
            " scheduleMode %u, tilingKey %llu, clearAtomic %d, staticTilingDataSize %zu.",
            executor->opType, tilingInfo.aicpuNumBlocks, tilingInfo.numBlocks, tilingInfo.scheMode,
            tilingInfo.tilingKey, tilingInfo.needAtomic, tilingInfo.staticTilingData.size());
}
} // namespace

void NnopbaseExecutorClearSet(NnopbaseExecutorSpaceSet* set)
{
    while (!__sync_bool_compare_and_swap(&set->isVisit, false, true))
        ;
    for (NnopbaseExecutorSpace* space : set->spaces) {
        NnopbaseExecutorClearSpace(space);
    }
    set->isVisit = false;
}

aclnnStatus NnopbaseExecutorGetAttr(NnopbaseExecutor* executor, const size_t index, NnopbaseAttrAddr** attr)
{
    NnopbaseAttrs* opAttrs = &executor->attrs;
    if (index >= opAttrs->num) {
        OP_LOGE_FOR_INVALID_ARGUMENT_INDEX_OUT_OF_RANGE(std::to_string(index).c_str(), "index",
                                                        std::to_string(opAttrs->num).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    *attr = &(opAttrs->attrs[index].addr);
    return OK;
}

aclnnStatus NnopbaseExecutorSetRef(NnopbaseExecutor* executor, const size_t inputIrIdx, const size_t outputIrIdx)
{
    auto& inInstances = executor->ownArgs.inputs.paramDescs.instances;
    auto& outInstances = executor->ownArgs.outputs.paramDescs.instances;
    CHECK_COND((inputIrIdx < inInstances.size()), ACLNN_ERR_PARAM_INVALID,
               "Get input ir index %zu is out of input num %zu.", inputIrIdx, inInstances.size());
    CHECK_COND((outputIrIdx < outInstances.size()), ACLNN_ERR_PARAM_INVALID,
               "Get output ir index %zu is out of output num %zu.", outputIrIdx, outInstances.size());
    OP_LOGD("Op %s set ref input idx %zu, output index %zu.", executor->opType, inputIrIdx, outputIrIdx);
    inInstances[inputIrIdx].refIdx = outputIrIdx;
    outInstances[outputIrIdx].refIdx = inputIrIdx;
    return OK;
}

static aclnnStatus NnopbaseExecutorPrintErrMsg(const NnopbaseExecutor* const executor,
                                               const std::vector<size_t>& inIdxs)
{
    std::string reason = "Parameter ";
    std::vector<std::string> paramNames;
    // inIdxs存的是实际的index，需要转换成irIndex，打印对应的name
    auto& inInstances = executor->args->inputs.paramDescs.instances;
    uint32_t repeatIndex = 0U;
    for (uint32_t irIndex = 0U; irIndex < executor->args->inputs.paramDescs.count; irIndex++) {
        const uint32_t startIndex = inInstances[irIndex].startIndex;
        for (size_t idx : inIdxs) {
            if (executor->args->inputs.paramDescs.instances[irIndex].isDynamic) {
                if ((idx >= startIndex) && (idx < startIndex + inInstances[irIndex].num) && (repeatIndex != irIndex)) {
                    paramNames.push_back(std::string(inInstances[irIndex].name));
                    repeatIndex = irIndex;
                }
            } else {
                if (idx == startIndex) {
                    paramNames.push_back(std::string(inInstances[irIndex].name));
                }
            }
        }
    }
    for (size_t i = 0U; i < paramNames.size(); ++i) {
        reason += paramNames[i];
        if (i != paramNames.size() - 1U) {
            reason += ", ";
        }
    }
    if (paramNames.size() > 1U) {
        reason += " of the operator are non-contiguous tensors.";
    } else {
        reason += " of the operator is a non-contiguous tensor.";
    }
    reason += " A processing policy must be configured through "
              "the AutoContiguous() or IgnoreContiguous() API of the OpDef class";
    OP_LOGE_FOR_EXECUTION_ERROR_WITHOUT_SOLUTION(reason.c_str());
    return ACLNN_ERR_INNER;
}

static inline void NnopbaseAddTensorsInfo(const NnopbaseTensors& tensors, std::string& ioInfo)
{
    const auto& extTensors = tensors.extTensors;
    const auto& paramInstance = tensors.paramDescs.instances;
    size_t j = 0U;
    for (uint32_t i = 0; i < tensors.paramDescs.count; ++i) {
        if (paramInstance[i].name != nullptr) {
            ioInfo += std::string(paramInstance[i].name);
        }
        // 对于未传入的可选输入占位
        if (extTensors[j].isNull) {
            ioInfo += "(NULL, NULL)";
            j += 1U;
            ioInfo += (i != tensors.paramDescs.count - 1U) ? ", " : "";
            continue;
        }

        // 动态输入打印第一个输入的dtype和format
        if (paramInstance[i].isDynamic) {
            const size_t startIndex = paramInstance[i].startIndex;
            ioInfo += std::string("(") + op::ToString(extTensors[startIndex].rt2Tensor.GetDataType()).GetString() +
                      std::string(", ");
            ioInfo += op::ToString(extTensors[startIndex].rt2Tensor.GetStorageFormat()).GetString() + std::string(")");
            j += paramInstance[i].num;
        } else {
            ioInfo += std::string("(") + op::ToString(extTensors[j].rt2Tensor.GetDataType()).GetString() +
                      std::string(", ");
            ioInfo += op::ToString(extTensors[j].rt2Tensor.GetStorageFormat()).GetString() + std::string(")");
            j += 1U;
        }
        ioInfo += (i != tensors.paramDescs.count - 1U) ? ", " : "";
    }
}

static inline aclnnStatus NnopbaseGetSupportInfo(const NnopbaseOpCompatibility& opCompatibility,
                                                 OpSocSupportInfo& supportInfo)
{
    CHECK_COND((opCompatibility.paramSupportList != nullptr), ACLNN_ERR_INNER_FIND_KERNEL_ERROR,
               "SupportList is nullptr!");
    const auto socTypeMap = nnopbase::IndvSoc::GetInstance().GetSocTypeMap();
    const auto socVersion = nnopbase::IndvSoc::GetInstance().GetCurSocVersion();
    const auto& iter = socTypeMap.find(socVersion);
    if (iter != socTypeMap.cend()) {
        for (size_t i = 0U; i < opCompatibility.supportedSocCount; i++) {
            if (opCompatibility.supportedSocList[i] == iter->second) {
                CHECK_COND((i < opCompatibility.paramSupportList->num), ACLNN_ERR_INNER_FIND_KERNEL_ERROR,
                           "Index[%zu] is out of SupportList num[%zu]!", i, opCompatibility.paramSupportList->num);
                supportInfo = opCompatibility.paramSupportList->socSupportInfo[i];
                return OK;
            }
        }
    }

    return ACLNN_ERR_INNER_FIND_KERNEL_ERROR;
}

// 异常场景，返回错误码
static inline aclnnStatus PrintExceptionIoParamInfo(const NnopbaseTensors& inputs, const NnopbaseTensors& outputs,
                                                    const OpSocSupportInfo& supportList)
{
    std::string ioInfo = std::string("Supported dtype and format list is:");
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s", ioInfo.c_str());

    // print support info
    for (size_t i = 0U; i < supportList.num; i++) {
        std::string info = std::string("SupportInfo[") + std::to_string(i) + "]: ";
        const auto& supportInfo = supportList.supportInfo[i];
        for (size_t j = 0U; j < supportInfo.inputsNum; ++j) {
            if (inputs.paramDescs.instances[j].name != nullptr) {
                info += std::string(inputs.paramDescs.instances[j].name);
            }
            info += std::string("(") + op::ToString(supportInfo.inputsDesc[j].dtype).GetString() + std::string(", ");
            info += op::ToString(supportInfo.inputsDesc[j].format).GetString() + std::string("), ");
        }
        for (size_t j = 0U; j < supportInfo.outputsNum; ++j) {
            if (outputs.paramDescs.instances[j].name != nullptr) {
                info += std::string(outputs.paramDescs.instances[j].name);
            }
            info += std::string("(") + op::ToString(supportInfo.outputsDesc[j].dtype).GetString() + std::string(", ");
            info += op::ToString(supportInfo.outputsDesc[j].format).GetString() + std::string(")");
            info += (j < supportInfo.outputsNum - 1U) ? ", " : "";
        }
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s", info.c_str());
    }
    return ACLNN_ERR_PARAM_INVALID;
}

static inline bool Compare(const TensorDesc& paramDesc, const GertTensor& tensor)
{
    return paramDesc.dtype == tensor.GetDataType() && paramDesc.format == tensor.GetStorageFormat();
}

static inline bool DynamicIoCheck(const NnopbaseTensors& tensors, const TensorDesc* ioDesc, const size_t index)
{
    // dynamic输入只匹配第一个tensor的dtype和format
    const size_t startIndex = tensors.paramDescs.instances[index].startIndex;
    if (!Compare(ioDesc[index], tensors.extTensors[startIndex].rt2Tensor)) {
        return false;
    }
    return true;
}

static inline bool RequiredIoCheck(const NnopbaseTensors& tensors, const TensorDesc* ioDesc, const size_t index)
{
    const size_t startIndex = static_cast<size_t>(tensors.paramDescs.instances[index].startIndex);
    if (tensors.extTensors[startIndex].isRequired) {
        if (!Compare(ioDesc[index], tensors.extTensors[startIndex].rt2Tensor)) {
            return false;
        }
    }
    return true;
}

static inline bool OptionalIoCheck(const NnopbaseTensors& tensors, const TensorDesc* ioDesc, const size_t index)
{
    const size_t startIndex = static_cast<size_t>(tensors.paramDescs.instances[index].startIndex);
    if (tensors.extTensors[startIndex].isNull) {
        return true;
    }
    if (tensors.extTensors[startIndex].isOptional) {
        if (!Compare(ioDesc[index], tensors.extTensors[startIndex].rt2Tensor)) {
            return false;
        }
    }
    return true;
}

static inline bool IoParamCheck(const NnopbaseTensors& tensors, const TensorDesc* ioDesc, const size_t num,
                                NnopbaseParamCheckMode mode, bool isInput)
{
    bool optionalCheckInputFlag = (mode == NnopbaseParamCheckMode::kCheckOptionalInput) ||
                                  (mode == NnopbaseParamCheckMode::kCheckAllIo);
    bool optionalCheckOutputFlag = (mode == NnopbaseParamCheckMode::kCheckOptionalOutput) ||
                                   (mode == NnopbaseParamCheckMode::kCheckAllIo);
    for (size_t j = 0U; j < num; j++) {
        if (tensors.paramDescs.instances[j].isDynamic && !DynamicIoCheck(tensors, ioDesc, j)) {
            return false;
        } else if (optionalCheckInputFlag && isInput &&
                   (!(RequiredIoCheck(tensors, ioDesc, j)) || !(OptionalIoCheck(tensors, ioDesc, j)))) {
            return false;
        } else if (optionalCheckOutputFlag && !isInput &&
                   (!(RequiredIoCheck(tensors, ioDesc, j)) || !(OptionalIoCheck(tensors, ioDesc, j)))) {
            return false;
        } else if (!RequiredIoCheck(tensors, ioDesc, j)) {
            return false;
        }
    }
    return true;
}

static inline bool IsParamMatchSupportInfo(const NnopbaseTensors& inputs, const NnopbaseTensors& outputs,
                                           const OpSocSupportInfo& supportList, NnopbaseParamCheckMode mode)
{
    const size_t num = supportList.num;
    for (size_t i = 0U; i < num; i++) {
        if (IoParamCheck(inputs, supportList.supportInfo[i].inputsDesc, supportList.supportInfo[i].inputsNum, mode,
                         true) &&
            IoParamCheck(outputs, supportList.supportInfo[i].outputsDesc, supportList.supportInfo[i].outputsNum, mode,
                         false)) {
            return true;
        }
    }
    return false;
}

aclnnStatus NnopbaseSetUnContiguousExecutorRepeatable(NnopbaseExecutor* executor)
{
    aclOpExecutor* inExe = executor->inUnContExe;
    if (inExe != nullptr) {
        if (inExe->GetMagicNumber() != K_EXECUTOR_MAGIC_NUMBER) {
            OP_LOGE(ACLNN_ERR_INNER,
                    "Magic number is %lu, this executor for Contiguous operator does not support setting repeatable.",
                    inExe->GetMagicNumber());
            return ACLNN_ERR_INNER;
        }
        auto ret = inExe->SetRepeatable();
        CHECK_COND(ret == OK, ACLNN_ERR_INNER, "Failed to set executor for Contiguous operator repeatable.");
        aclOpExecutor* viewCopyExe = executor->viewCopyExe;
        if (viewCopyExe != nullptr) {
            if (viewCopyExe->GetMagicNumber() == K_EXECUTOR_MAGIC_NUMBER) {
                return viewCopyExe->SetRepeatable();
            }
            OP_LOGE(ACLNN_ERR_INNER,
                    "Magic number is %lu, this executor for ViewCopy operator does not support setting repeatable.",
                    viewCopyExe->GetMagicNumber());
            return ACLNN_ERR_INNER;
        }
        return ret;
    }

    return OK;
}

aclnnStatus NnopbaseSetRepeatable(void* executor)
{
    NNOPBASE_ASSERT_NULLPTR_WITH_RETURN(executor, ACLNN_ERR_PARAM_NULLPTR);
    op::internal::GetThreadLocalContext().logInfo_.l2ApiName = (static_cast<NnopbaseExecutor*>(executor))->opType;
    (static_cast<NnopbaseExecutor*>(executor))->repeatFlag = true;
    aclnnStatus ret = OK;
    NnopbaseExecutorFixCache(static_cast<NnopbaseExecutor*>(executor));
    if ((static_cast<NnopbaseExecutor*>(executor))->inUnContExe != nullptr) {
        ret = NnopbaseSetUnContiguousExecutorRepeatable(static_cast<NnopbaseExecutor*>(executor));
    }
    return ret;
}

aclnnStatus NnopbaseResetAuxOpExecutor(NnopbaseExecutor* executor)
{
    aclOpExecutor* inExe = executor->inUnContExe;
    if (inExe != nullptr) {
        if (inExe->GetMagicNumber() != K_EXECUTOR_MAGIC_NUMBER) {
            OP_LOGE(ACLNN_ERR_INNER,
                    "Magic number is %lu, this executor of Contiguous operator does not support resetting.",
                    inExe->GetMagicNumber());
            return ACLNN_ERR_INNER;
        }
        delete inExe;
        executor->inUnContExe = nullptr;
        aclOpExecutor* viewCopyExe = executor->viewCopyExe;
        if ((viewCopyExe != nullptr) && (viewCopyExe->GetMagicNumber() == K_EXECUTOR_MAGIC_NUMBER)) {
            delete viewCopyExe;
            executor->viewCopyExe = nullptr;
        }
        return OK;
    }

    return OK;
}

aclnnStatus NnopbaseResetExecutor(void* executor)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    op::internal::GetThreadLocalContext().logInfo_.l2ApiName = (static_cast<NnopbaseExecutor*>(executor))->opType;
    aclnnStatus ret = OK;
    if ((static_cast<NnopbaseExecutor*>(executor))->inUnContExe != nullptr) {
        ret = NnopbaseResetAuxOpExecutor(static_cast<NnopbaseExecutor*>(executor));
    }
    NnopbaseExecutorClear(static_cast<NnopbaseExecutor*>(executor));
    return ret;
}

bool NnopbaseIsInput(const void* executor, const size_t index, size_t* tensorIndex)
{
    CHECK_COND(executor != nullptr, false, "Executor is nullptr.");
    CHECK_COND(tensorIndex != nullptr, false, "TensorIndex is nullptr.");
    const auto& tensors = (op::internal::PtrCastTo<const NnopbaseExecutor>(executor))->args->inputs;
    size_t realIndex = index;
    NnopbaseGetRealIndex(tensors.paramDescs, &realIndex);
    OP_LOGI("Index is %zu, realindex is %zu.", index, realIndex);
    if (realIndex < tensors.num) {
        *tensorIndex = realIndex;
        return true;
    } else {
        *tensorIndex = realIndex - tensors.num;
        OP_LOGI("Index is %zu, output tensorIndex is %zu.", index, *tensorIndex);
        return false;
    }
}

bool NnopbaseDynamicIsInput(const void* executor, const size_t irIndex, size_t* tensorIrIndex)
{
    CHECK_COND(executor != nullptr, false, "Executor is nullptr.");
    CHECK_COND(tensorIrIndex != nullptr, false, "TensorIrIndex is nullptr.");
    const auto& inputParamDescs = (op::internal::PtrCastTo<const NnopbaseExecutor>(executor))->args->inputs.paramDescs;
    *tensorIrIndex = irIndex;
    if (irIndex < inputParamDescs.activeInputCount) {
        NnopbaseGetRealIndex(inputParamDescs, tensorIrIndex);
        OP_LOGI("IrIndex is %zu, activeInputCount is %u, input tensorIrIndex is %zu.", irIndex,
                inputParamDescs.activeInputCount, *tensorIrIndex);
        return true;
    } else {
        *tensorIrIndex = irIndex - inputParamDescs.activeInputCount;
        OP_LOGI("IrIndex is %zu, activeInputCount is %u, output tensorIrIndex is %zu.", irIndex,
                inputParamDescs.activeInputCount, *tensorIrIndex);
        return false;
    }
}

static inline void PrintAttr(NnopbaseExecutor* executor)
{
    // 存在attr才打此行日志
    if (executor->attrs.num > 0U) {
        OP_LOGI("[PrintAttr]Op %s attrs is %s", executor->opType, nnopbase::ToStr(executor->attrs).c_str());
    }
}

static aclnnStatus NnopbaseExecutorUpdateAddr(NnopbaseExecutor* executor, void* const workspace,
                                              const uint64_t workspaceLen)
{
    const auto& inIdxs = executor->args->inputs.unContiguousTensors.idxs;
    const auto& offsets = executor->args->inputs.unContiguousTensors.workspaceOffsets;
    OP_LOGI("Executor addr is %p, inIdxs.size() is %zu, offsets.size() is %zu", executor, inIdxs.size(),
            offsets.size());
    if (!inIdxs.empty()) {
        if (offsets.size() != inIdxs.size()) {
            return NnopbaseExecutorPrintErrMsg(executor, inIdxs);
        }
        const auto inWorkspace = op::internal::PtrCastTo<NnopbaseChar>(workspace) + workspaceLen;
        for (size_t i = 0U; i < inIdxs.size(); ++i) {
            const auto addr = inWorkspace + offsets[i];
            const size_t idx = inIdxs[i];
            auto& rt2Tensor = executor->args->inputs.extTensors[idx].rt2Tensor;
            OP_LOGI("Op %s update input[%zu], addr is %p.", executor->opType, idx, addr);
            NNOPBASE_ASSERT_OK_RETVAL(rt2Tensor.MutableTensorData().SetAddr(addr, nullptr));
        }
    }

    const auto& refIdxs = executor->args->inputs.unContiguousTensors.refIdxs;
    if (!refIdxs.empty()) {
        for (size_t i = 0U; i < refIdxs.size(); ++i) {
            const size_t idx = refIdxs[i];
            // 获取input信息
            size_t inIrIndex = 0U;
            size_t relativeIndex = 0U; // ref类对应output的相对index相等
            NnopbaseGetIrIndex(executor->args->inputs.paramDescs, idx, inIrIndex, relativeIndex);
            const auto& inInstance = executor->args->inputs.paramDescs.instances[inIrIndex];
            const size_t inStartIndex = inInstance.startIndex;
            auto& inRt2Tensor = executor->args->inputs.extTensors[inStartIndex + relativeIndex].rt2Tensor;

            // 获取output信息
            const int32_t outIrIdx = inInstance.refIdx;
            const auto& outInstance = executor->args->outputs.paramDescs.instances[outIrIdx];
            const size_t outStartIndex = outInstance.startIndex;
            auto& outRt2Tensor = executor->args->outputs.extTensors[outStartIndex + relativeIndex].rt2Tensor;

            // 刷新output地址
            OP_LOGI("Op %s update ref output[%zu], addr is %p.", executor->opType, outStartIndex + relativeIndex,
                    inRt2Tensor.GetAddr());
            NNOPBASE_ASSERT_OK_RETVAL(outRt2Tensor.MutableTensorData().SetAddr(inRt2Tensor.GetAddr(), nullptr));
            // refIdxs, refContTensors, refUnContTensors 的 size 是一样的
            executor->args->inputs.unContiguousTensors.refContTensors[i]->SetStorageAddr(inRt2Tensor.GetAddr());
        }
    }
    return OK;
}

aclnnStatus NnopbaseExecutorPrepareParamsExt(NnopbaseExecutor* const executor, aclrtStream const stream)
{
    if (NnopbaseSkipKernelLaunch(executor)) {
        return OK;
    }
    NnopbaseHcclCommParamDesc paramDesc = {0, 0, 0, 0, 0};
    NnopbaseExecutorArgsAddr argsAddr = {nullptr, nullptr, nullptr, nullptr, &paramDesc};

    uint32_t mc2Num = 0U;
    if (executor->mc2.enabled) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseExecutorGetMc2Num(executor, stream, &argsAddr, &mc2Num));
    }

    // workspace num为0时，args中需要占位
    const uint32_t workspaceNum = executor->workspaces.num == 0U ? 1U : static_cast<uint32_t>(executor->workspaces.num);
    // input, output, workspaces, 3 for tiling, overflow, ctrlAddr
    uint32_t num = executor->args->inputs.paramDescs.count + executor->args->outputs.paramDescs.count + workspaceNum +
                   3U + mc2Num;
    if (executor->args->outputs.outPutShapeSize != 0U) {
        num++; // for outPutShapeSize
    }
    auto tilingData = op::internal::PtrCastTo<NnopbaseTilingData>(executor->args->tilingInfo.tilingData);
    void* args = nullptr;
    if (executor->hasTiling) {
        args = op::internal::PtrCastTo<NnopbaseUChar>(tilingData->GetData()) - num * sizeof(void*);
        const size_t tilingDataSize = tilingData->GetDataSize();
        OP_LOGI("TilingDataSize is %zu bytes.", tilingDataSize);

        const size_t alignTilingDataSize = ((tilingDataSize % 8U) != 0) ? (tilingDataSize / 8U + 1U) * 8U :
                                                                          tilingDataSize; // 8byte对齐
        argsAddr.ptr = executor->args->binInfo->oomFlag ?
                           op::internal::PtrCastTo<NnopbaseUChar>(tilingData->GetData()) + alignTilingDataSize :
                           op::internal::PtrCastTo<NnopbaseUChar>(tilingData->GetData()) + tilingDataSize;

        if (executor->args->binInfo->oomFlag || op::internal::IsArgExceptionDumpEnable()) {
            NNOPBASE_ASSERT_OK_RETVAL(NnopbaseExecutorArgsGetDfxInfo(executor, &argsAddr, workspaceNum));
        }
    } else {
        // MC2算子没有静态场景
        args = static_cast<void*>(executor->args->argsBuf.data());
        argsAddr.ptr = op::internal::PtrCastTo<NnopbaseUChar>(executor->args->argsBuf.data()) + num * sizeof(void*);
    }

    if ((executor->args->inputs.hostInputNum > 0) || executor->args->inputs.hasDynamic ||
        executor->args->outputs.hasDynamic || (executor->mc2.enabled && executor->hasTiling)) {
        executor->argsExt.hostInputInfoNum = executor->args->inputs.hostInputNum +
                                             static_cast<uint16_t>(executor->args->inputs.dynamicNum +
                                                                   executor->args->outputs.dynamicNum);
        OP_LOGD("hostInputInfoNum is %u", executor->argsExt.hostInputInfoNum);
        if (executor->mc2.enabled) {
            if (nnopbase::IndvSoc::GetInstance().NnopbaseEnableCcuLaunch(executor->mc2.serverType)) {
                executor->mc2.fusionArgs.hostInputInfoNum = executor->argsExt.hostInputInfoNum + 1U;
                argsAddr.hostInputData = argsAddr.ptr +
                                         executor->mc2.fusionArgs.hostInputInfoNum * sizeof(aclrtPlaceHolderInfo);
                // mc2算子会占用一个rtHostInputInfo_t存储tilingdata信息
                argsAddr.hostInputInfo = op::internal::PtrCastTo<aclrtPlaceHolderInfo>(argsAddr.ptr +
                                                                                       sizeof(aclrtPlaceHolderInfo));
                executor->mc2.fusionArgs.hostInputInfoPtr = op::internal::PtrCastTo<rtHostInputInfo_t>(argsAddr.ptr);
            } else {
                // 1 is aicpu tilingdata hostinfo
                executor->mc2.aicpuArgs.hostInputInfoNum = executor->argsExt.hostInputInfoNum + 1;
                argsAddr.hostInputData = argsAddr.ptr + (executor->mc2.aicpuArgs.hostInputInfoNum +
                                                         executor->argsExt.hostInputInfoNum) *
                                                            sizeof(aclrtPlaceHolderInfo);
                argsAddr.aicpuHostInputInfo = op::internal::PtrCastTo<aclrtPlaceHolderInfo>(
                    argsAddr.ptr + executor->argsExt.hostInputInfoNum * sizeof(aclrtPlaceHolderInfo));
                argsAddr.hostInputInfo = op::internal::PtrCastTo<aclrtPlaceHolderInfo>(argsAddr.ptr);
                executor->mc2.aicpuArgs.hostInputInfoPtr = op::internal::PtrCastTo<rtHostInputInfo_t>(
                    argsAddr.aicpuHostInputInfo);
                executor->argsExt.hostInputInfoPtr = argsAddr.hostInputInfo;
            }
        } else {
            argsAddr.hostInputData = argsAddr.ptr + executor->argsExt.hostInputInfoNum * sizeof(aclrtPlaceHolderInfo);
            argsAddr.hostInputInfo = op::internal::PtrCastTo<aclrtPlaceHolderInfo>(argsAddr.ptr);
            executor->argsExt.hostInputInfoPtr = argsAddr.hostInputInfo;
        }
    } else {
        executor->argsExt.hostInputInfoPtr = nullptr;
        executor->argsExt.hostInputInfoNum = 0U;
        argsAddr.hostInputData = argsAddr.ptr;
    }
    void** addr = nullptr;
    if (executor->mc2.enabled) {
        if (nnopbase::IndvSoc::GetInstance().NnopbaseEnableCcuLaunch(executor->mc2.serverType)) {
            addr = (void**)args;
            executor->mc2.fusionArgs.args = args;
            executor->argsExt.args = args;
        } else {
            addr = op::internal::PtrCastTo<void*>(op::internal::PtrCastTo<NnopbaseUChar>(args) + sizeof(void*));
            executor->mc2.aicpuArgs.args = args;
            executor->argsExt.args = op::internal::PtrCastTo<void>(addr);
        }
    } else {
        addr = (void**)args;
        executor->argsExt.args = args;
    }

    if (nnopbase::IndvSoc::GetInstance().UseCoreTypeMagic() && (executor->args->binInfo->coreType == kMix)) {
        void* ctrlAddr = nullptr;
        NNOPBASE_ASSERT_RTOK_RETVAL(
            aclrtGetHardwareSyncAddr(&ctrlAddr)); // 获取核间同步地址配合AscendC核间同步接口完成多核任务协调
        addr[0] = ctrlAddr;
        addr++;
        argsAddr.hcclDesc->hasFfts = 1U;
    }
    if (executor->mc2.enabled) {
        for (size_t i = 0U; i < executor->mc2.contextAddrs.size(); i++) {
            *addr = executor->mc2.contextAddrs[i];
            addr++;
        }
    }
    /* inputs address. */
    addr = NnopbaseExecutorPrepareInputsParamsExt(executor, addr, &argsAddr);

    /* outputs address. */
    addr = NnopbaseExecutorPrepareOutputsParamsExt(executor, addr, &argsAddr);

    if (executor->args->outputs.outPutShapeSize != 0U) {
        *addr = op::internal::PtrCastTo<uint8_t>(executor->workspaces.workspaces[0]) +
                (executor->workspaces.length - executor->args->outputs.outPutShapeSize);
        executor->args->outputs.outPutShapeArgsOffset = (op::internal::PtrCastTo<NnopbaseUChar>(addr) -
                                                         op::internal::PtrCastTo<NnopbaseUChar>(
                                                             executor->argsExt.args)) /
                                                        sizeof(void*);
        OP_LOGI("OutputShape addr is %p, outPutShapeArgsOffset is %zu.", *addr,
                executor->args->outputs.outPutShapeArgsOffset);
        addr++;
    }

    /* workspace address. */
    if (executor->workspaces.num == 0U) {
        *addr = nullptr;
        addr++;
    } else {
        for (size_t i = 0U; i < executor->workspaces.num; i++) {
            *addr = executor->workspaces.workspaces[i];
            executor->workspaces.workspaceArgsOffset[i] = static_cast<uint32_t>(
                op::internal::PtrCastTo<NnopbaseUChar>(addr) -
                op::internal::PtrCastTo<NnopbaseUChar>(executor->argsExt.args));
            addr++;
        }
    }
    /* tiling data addr. */
    if (executor->hasTiling) {
        *addr = tilingData->GetData();
        executor->argsExt.tilingAddrOffset = static_cast<uint32_t>(
            op::internal::PtrCastTo<NnopbaseUChar>(addr) -
            op::internal::PtrCastTo<NnopbaseUChar>(executor->argsExt.args));
        executor->argsExt.tilingDataOffset = static_cast<uint32_t>(
            op::internal::PtrCastTo<NnopbaseUChar>(tilingData->GetData()) -
            op::internal::PtrCastTo<NnopbaseUChar>(executor->argsExt.args));
        if (executor->mc2.enabled) {
            if (nnopbase::IndvSoc::GetInstance().NnopbaseEnableCcuLaunch(executor->mc2.serverType)) {
                executor->mc2.fusionArgs.hostInputInfoPtr->addrOffset = op::internal::PtrCastTo<NnopbaseUChar>(addr) -
                                                                        op::internal::PtrCastTo<NnopbaseUChar>(
                                                                            executor->mc2.fusionArgs.args);
                executor->mc2.fusionArgs.hostInputInfoPtr->dataOffset = op::internal::PtrCastTo<NnopbaseUChar>(
                                                                            tilingData->GetData()) -
                                                                        op::internal::PtrCastTo<NnopbaseUChar>(
                                                                            executor->mc2.fusionArgs.args);
                // tilingOff设置的是tilingDataPtr首地址和args首地址间的偏移，用来解析args获取tilingdata
                argsAddr.hcclDesc->tilingOff = executor->mc2.fusionArgs.hostInputInfoPtr->addrOffset / sizeof(void*);
            } else {
                argsAddr.aicpuHostInputInfo->addrOffset = static_cast<uint32_t>(
                    op::internal::PtrCastTo<NnopbaseUChar>(addr) -
                    op::internal::PtrCastTo<NnopbaseUChar>(executor->mc2.aicpuArgs.args));
                argsAddr.aicpuHostInputInfo->dataOffset = static_cast<uint32_t>(
                    op::internal::PtrCastTo<NnopbaseUChar>(tilingData->GetData()) -
                    op::internal::PtrCastTo<NnopbaseUChar>(executor->mc2.aicpuArgs.args));
                // tilingOff设置的是tilingdata ptr地址和args首地址间的偏移，算子在kernel中用来解析args获取tilingdata
                argsAddr.hcclDesc->tilingOff = argsAddr.aicpuHostInputInfo->addrOffset / sizeof(void*);
            }
        }
        addr++;
    } else {
        executor->argsExt.tilingAddrOffset = 0U;
        executor->argsExt.tilingDataOffset = 0U;
    }
    executor->argsExt.hasTiling = executor->hasTiling;
    /* overflow addr. */
    *addr = g_nnopbaseSysCfgParams.overflowAddr;
    /* tiling data has been encode in tiling func. */
    /* host input data and data have been encoded. */
    executor->argsExt.argsSize = static_cast<uint32_t>(argsAddr.ptr -
                                                       op::internal::PtrCastTo<NnopbaseUChar>(executor->argsExt.args)) +
                                 NNOPBASE_PARAM_EXT_LEN + static_cast<uint32_t>(sizeof(aclrtPlaceHolderInfo));
    if (executor->mc2.enabled) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbasePrepareMC2Params(executor, &argsAddr));
    }

    CHECK_COND((argsAddr.hostInputData <= &executor->args->argsBuf.back()), ACLNN_ERR_PARAM_INVALID,
               "args is out of memory. argsBuffer addr is %p, args addr is %p", &executor->args->argsBuf.back(),
               argsAddr.hostInputData);
    return OK;
}

aclnnStatus NnopbaseKernelRegister(NnopbaseExecutor* executor, NnopbaseBinInfo* binInfo)
{
    const std::lock_guard<std::mutex> lock(g_nnopbaseRegisterMtx);
    if (binInfo->hasReg) {
        return OK;
    }

    if (binInfo->bin == nullptr &&
        NnopbaseReadBinFile(binInfo->binPath.c_str(), &binInfo->bin, &binInfo->binLen) != OK) {
        std::string
            errMsg = "1.The file does not exist. 2.The file is damaged. 3.The user does not have the read permission";
        OP_LOGE_FOR_FILE_OPERATION_ERROR_PARSE(binInfo->binPath.c_str(), errMsg.c_str());
        return ACLNN_ERR_INNER_OP_FILE_INVALID;
    }

    const std::string socVersion = nnopbase::IndvSoc::GetInstance().GetCurSocVersion();
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseReadKernelJsonFile(binInfo, executor->collector->oppPath, socVersion));
    if (executor->mc2.enabled && nnopbase::IndvSoc::GetInstance().SupportMc2FusionLaunch()) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseMC2DynamicKernelRegister(executor->collector->useCoreTypeMagic, binInfo));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAclrtBinaryLoad(executor->collector->useCoreTypeMagic, binInfo));
    } else {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAclrtBinaryLoad(executor->collector->useCoreTypeMagic, binInfo));
    }
    NNOPBASE_ASSERT_OK_RETVAL(RegisterCustomizedCallback(executor, binInfo));
    if (binInfo->initValues.size() > 0U) {
        NNOPBASE_ASSERT_OK_RETVAL(SetTensorDataSizeToInitValue(executor));
        // 构造TilingParseContext，执行tilingParse函数
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseBuildAndRunMemsetTilingParse(executor->args->binInfo->memsetInfo));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseMemsetTilingContextInit(executor->args->binInfo->initValues,
                                                                  executor->args->binInfo->memsetInfo,
                                                                  executor->args->outputs.paramDescs));
    }
    binInfo->hasReg = true;
    return OK;
}

bool NnopbaseExecutorGetStaticBinInfo(NnopbaseExecutor* const executor)
{
    if (!GenStaticKeyAndFindStaticBin(executor, true)) {
        OP_LOGW("Cannot find static kernel bin with stride information, trying to find without stride again.");
        return GenStaticKeyAndFindStaticBin(executor, false);
    }
    return true;
}

static void NnopbaseExecutorGetWorkspaceSizes(NnopbaseExecutor* executor, uint64_t* workspaceLen)
{
    if (NnopbaseSkipKernelLaunch(executor)) {
        executor->workspaces.num = 0U;
        *workspaceLen = 0U;
        executor->workspaces.length = 0U;
        executor->args->outputs.outPutShapeSize = 0U;
        OP_LOGI("Op %s all outputs are empty tensors.", executor->opType);
        return;
    }
    const auto workspacesSizes = NnopbaseGetWorkspacesSizesFromArgs(executor->args);
    executor->workspaces.num = workspacesSizes->GetSize();
    *workspaceLen = 0U;
    for (size_t i = 0U; i < executor->workspaces.num; i++) {
        *workspaceLen += workspacesSizes->GetData()[i];
    }

    if ((executor->args->binInfo->dfxInfo.isPrintEnable) || (executor->args->binInfo->dfxInfo.isAssertEnable) ||
        (executor->args->binInfo->dfxInfo.isTimeStampEnable)) {
        *workspaceLen += executor->args->binInfo->debugBufSize;
        // 若算子本身的workspace num为0，需要增加为1
        executor->workspaces.num = (executor->workspaces.num == 0U) ? 1U : executor->workspaces.num;
    }
    if (executor->args->outputs.outPutShapeSize != 0U) {
        *workspaceLen += executor->args->outputs.outPutShapeSize;
    }
    executor->workspaces.length = *workspaceLen;
    OP_LOGI("Op[%s] workspace num is %zu, workspaceLen is %lu bytes, bin path is %s.", executor->opType,
            executor->workspaces.num, executor->workspaces.length, executor->args->binInfo->binPath.c_str());
}

static aclnnStatus NnopbaseExecutorDoTiling(NnopbaseExecutor* executor)
{
    if (NnopbaseSkipKernelLaunch(executor)) {
        RecordNnopbaseTime(executor, NnopbaseTimeIdx::kTilingStart);
        RecordNnopbaseTime(executor, NnopbaseTimeIdx::kTilingEnd);
        return OK;
    }
    // init tilingdata & workspace
    auto& tilingInfo = executor->args->tilingInfo;
    auto workspacesSizes = op::internal::PtrCastTo<gert::ContinuousVector>(tilingInfo.workspacesSizes);
    auto tilingData = op::internal::PtrCastTo<NnopbaseTilingData>(tilingInfo.tilingData);
    workspacesSizes->Init(NNOPBASE_NORM_MAX_WORKSPACE_NUMS);
    const size_t tilingDataSize = executor->args->binInfo->opParaSize == 0U ? NNOPBASE_MAX_TILING_DATA_LEN :
                                                                              executor->args->binInfo->opParaSize;
    tilingData->Init(tilingDataSize, &(executor->args->argsBuf[executor->args->tilingDataOffset]));

    // build context
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseTilingContextBuild(executor));

    // do tiling
    RecordNnopbaseTime(executor, NnopbaseTimeIdx::kTilingStart);
    const auto tilingBeginTime = NnopbaseMsprofSysTime();
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor->regInfo);

    auto tiling = executor->regInfo->tiling;
    if (tiling == nullptr) {
        OP_LOGE_FOR_EXECUTION_TILING_ERROR("The tiling function does not exist");
        return ACLNN_ERR_INNER_TILING_ERROR;
    }
    ge::graphStatus ret = executor->regInfo->tiling(
        op::internal::PtrCastTo<gert::TilingContext>(executor->tiling.contextExt.context));
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_EXECUTION_TILING_ERROR("Failed to execute tiling function");
        return ACLNN_ERR_INNER_TILING_ERROR;
    }
    if (executor->dfx.tilingId != nullptr) {
        NnopbaseReportApiInfo(tilingBeginTime, *executor->dfx.tilingId);
    }
    RecordNnopbaseTime(executor, NnopbaseTimeIdx::kTilingEnd);

    // get tiling output
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor->tiling.tilingKey);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor->tiling.numBlocks);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor->tiling.scheMode);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor->tiling.needAtomic);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor->tiling.dynUbufSize);
    tilingInfo.tilingKey = *(executor->tiling.tilingKey);
    tilingInfo.numBlocks = *(executor->tiling.numBlocks);
    tilingInfo.scheMode = *(executor->tiling.scheMode);
    tilingInfo.needAtomic = *(executor->tiling.needAtomic);
    tilingInfo.aicpuNumBlocks = *(executor->tiling.aicpuNumBlocks);
    tilingInfo.dynUbufSize = *(executor->tiling.dynUbufSize);
    OP_LOGI(
        "Tiling outputs of %s are [tilingKey: %llu, numBlocks: %u, scheMode: %u, needAtomic: %u, aicpuNumBlocks: %u, "
        "dynBufSize: %u].",
        executor->opType, tilingInfo.tilingKey, tilingInfo.numBlocks, tilingInfo.scheMode, tilingInfo.needAtomic,
        tilingInfo.aicpuNumBlocks, tilingInfo.dynUbufSize);
    return OK;
}

static aclnnStatus NnopbaseExecutorCopyAttr(NnopbaseExecutor* executor)
{
    NnopbaseAttrs& attrs = executor->attrs;
    executor->args->attrsData.resize(attrs.totalDataLen);
    size_t offset = 0;
    for (size_t i = 0U; i < attrs.num; i++) {
        CHECK_COND((memcpy_s((op::internal::PtrCastTo<uint8_t>(executor->args->attrsData.data()) + offset),
                             attrs.attrs[i].addr.size, attrs.attrs[i].addr.addr, attrs.attrs[i].addr.size) == EOK),
                   ACLNN_ERR_PARAM_INVALID,
                   "Failed to execute memcpy_s for attr[%zu] info, src is %p, dst is %p, size is %zu.", i,
                   op::internal::PtrCastTo<uint8_t>(executor->args->attrsData.data()) + offset,
                   attrs.attrs[i].addr.addr, attrs.attrs[i].addr.size);
        // 开启L2 profiling时，保存attr，保证传给算子tiling的attr和上报的attr一致
        attrs.attrs[i].addr.addr = op::internal::PtrCastTo<void>(executor->args->attrsData.data() + offset);
        offset += attrs.attrs[i].addr.size;
    }
    OP_LOGI("Copy attr successfully.");
    return OK;
}

aclnnStatus NnopbaseExecutorTilingAndUpdateBinInfo(NnopbaseExecutor* executor)
{
    // create args
    NNOPBASE_ASSERT_OK_RETVAL(nnopbase::ArgsPool::GetInstance().CreateArgs(executor));
    executor->tiling.contextExt.hasBuilt = false;
    if ((executor->paramCheckMode != NnopbaseParamCheckMode::kCheckEnd) &&
        (CheckSocVersionAndParam(executor, executor->paramCheckMode) != OK)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (executor->attrs.num > 0) {
        // 不能判断level2ProfilingFlag为true的时候再拷贝，可能存在第一次没开profiling，第二次打开profiling又匹配到缓存的场景
        // 二阶段才上报profiling，attr可能会失效，需要拷贝保存
        NnopbaseExecutorCopyAttr(executor);
    }

    // find static bin
    if (executor->regInfo->hasStaticShapeBin && NnopbaseExecutorGetStaticBinInfo(executor)) {
        OP_LOGI("Op[%s] has found static bin.", executor->opType);
        UpdateStaticKernelTilingInfo(executor);
        RecordNnopbaseTime(executor, NnopbaseTimeIdx::kFindStaticEnd);
        RecordNnopbaseTime(executor, NnopbaseTimeIdx::kFindBinEnd);
        RecordNnopbaseTime(executor, NnopbaseTimeIdx::kTilingStart);
        RecordNnopbaseTime(executor, NnopbaseTimeIdx::kTilingEnd);
        return OK;
    }
    RecordNnopbaseTime(executor, NnopbaseTimeIdx::kFindStaticEnd);

    // find dynamic bin info.
    if (!NnopbaseExecutorGetDynamicBinInfo(executor)) {
        aclnnStatus ret = CheckSocVersionAndParam(executor, NnopbaseParamCheckMode::kCheckRequiredIo);
        if (ret != ACLNN_ERR_PARAM_INVALID) {
            std::string errMsg = "Cannot find kernel file for operator " + std::string(executor->opType) +
                                 ". The corresponding simplifiedKey" +
                                 " in binary_info_config.json file is missing, please reinstall operator package";
            OP_LOGE_FOR_EXECUTION_ERROR_WITHOUT_SOLUTION(errMsg.c_str());
            return ACLNN_ERR_INNER_FIND_KERNEL_ERROR;
        }
        return ACLNN_ERR_PARAM_INVALID;
    }

    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseKernelRegister(executor, executor->args->binInfo));
    RecordNnopbaseTime(executor, NnopbaseTimeIdx::kFindBinEnd);

    ResizeExecutorArgsBuf(executor);

    // do tiling
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseExecutorDoTiling(executor));
    if (op::internal::IsArgExceptionDumpEnable() || executor->args->binInfo->oomFlag) {
        NnopbaseExecutorPrepareDfxInfo(executor);
    }
    return OK;
}

aclnnStatus CheckSocVersionAndParam(NnopbaseExecutor* executor, NnopbaseParamCheckMode mode)
{
    OpSocSupportInfo supportList{};
    if (NnopbaseGetSupportInfo(executor->opCompatibility, supportList) != OK) {
        std::string
            errMsg = "SoC version " + nnopbase::IndvSoc::GetInstance().GetCurSocVersion() +
                     " verification failed. This SoC is not configured through the AddConfig API of the OpDef class";
        OP_LOGE_FOR_EXECUTION_ERROR_WITHOUT_SOLUTION(errMsg.c_str());
        return ACLNN_ERR_INNER_FIND_KERNEL_ERROR;
    }
    if (IsParamMatchSupportInfo(executor->args->inputs, executor->args->outputs, supportList, mode)) {
        return OK;
    } else {
        std::string
            errMsg = "The dtype or format of the actual input or output parameter of the operator "
                     "is inconsistent with that defined in the operator prototype OpDef, details of parameter are ";
        NnopbaseAddTensorsInfo(executor->args->inputs, errMsg);
        errMsg += ", ";
        NnopbaseAddTensorsInfo(executor->args->outputs, errMsg);
        OP_LOGE_FOR_EXECUTION_ERROR_WITHOUT_SOLUTION(errMsg.c_str());
        return PrintExceptionIoParamInfo(executor->args->inputs, executor->args->outputs, supportList);
    }
}

static void NnopbaseExecutorPrintIo(NnopbaseExecutor* executor)
{
    for (size_t i = 0U; i < executor->args->inputs.num; i++) {
        const auto& tensor = executor->args->inputs.extTensors[i];
        if (tensor.isNull) {
            OP_LOGI("[PrintIo]Op %s optional input[%zu] is null", executor->opType, i);
            continue;
        }
        OP_LOGI("[PrintIo]Op %s input[%zu] Get StorageFormat %s, dataType %s, shape %s, addr is %p", executor->opType,
                i, op::ToString(tensor.rt2Tensor.GetStorageFormat()).GetString(),
                op::ToString(tensor.rt2Tensor.GetDataType()).GetString(),
                op::ToString(tensor.rt2Tensor.GetStorageShape()).GetString(), tensor.rt2Tensor.GetAddr());
    }
    for (size_t i = 0U; i < executor->args->outputs.num; i++) {
        const auto& tensor = executor->args->outputs.extTensors[i];
        if (tensor.isNull) {
            OP_LOGI("[PrintIo]Op %s optional output[%zu] is null", executor->opType, i);
            continue;
        }
        OP_LOGI("[PrintIo]Op %s output[%zu] Get StorageFormat %s, dataType %s, shape %s, addr is %p", executor->opType,
                i, op::ToString(tensor.rt2Tensor.GetStorageFormat()).GetString(),
                op::ToString(tensor.rt2Tensor.GetDataType()).GetString(),
                op::ToString(tensor.rt2Tensor.GetStorageShape()).GetString(), tensor.rt2Tensor.GetAddr());
    }
}

void NnopbaseExecutorCopyCacheAttr(NnopbaseExecutor* executor)
{
    if (op::internal::opProfilingSwitch.level2ProfilingFlag) {
        size_t offset = 0;
        for (size_t i = 0U; i < executor->attrs.num; i++) {
            executor->attrs.attrs[i].addr.addr = op::internal::PtrCastTo<void>(executor->args->attrsData.data() +
                                                                               offset);
            offset += executor->attrs.attrs[i].addr.size;
        }
    }
}

static bool NnopbaseExecutorCachedArgs(NnopbaseExecutor* executor)
{
    // 新流程有三种场景会走到这里：匹配到缓存但是非连续、args被占用导致没匹配到缓存和没匹配到缓存
    // 匹配到缓存但是非连续不用再执行tiling
    if (executor->isCachedArgs) {
        if ((executor->args->binInfo->initValues.size() != 0U)) {
            executor->hasMemset = executor->args->binInfo->isStaticShape ? true : executor->args->tilingInfo.needAtomic;
        }
        NnopbaseExecutorCopyCacheAttr(executor);
        return true;
    }
    if (nnopbase::ArgsPool::GetInstance().MatchArgs(executor)) {
        // 新流程args被占用导致第一次未匹配到缓存，此时匹配到缓存需要更新args上的io地址
        // 老流程add io时tensor是添加到ownArgs上，此时也需要更新args上的io地址
        UpdateArgsIoAddr(&executor->args->inputs, &executor->ownArgs.inputs);
        UpdateArgsIoAddr(&executor->args->outputs, &executor->ownArgs.outputs);
        UpdateArgsUncontiguousTensor(executor->args->inputs.unContiguousTensors,
                                     executor->ownArgs.inputs.unContiguousTensors);
        if ((executor->args->binInfo->initValues.size() != 0U)) {
            executor->hasMemset = executor->args->binInfo->isStaticShape ? true : executor->args->tilingInfo.needAtomic;
        }
        NnopbaseExecutorCopyCacheAttr(executor);
        return true;
    }
    return false;
}

aclnnStatus NnopbaseExecutorMatchCache(NnopbaseExecutor* executor)
{
    // set empty tensor
    if ((executor->ownArgs.outputs.paramDescs.count != 0U) &&
        (executor->ownArgs.outputs.paramDescs.emptyNum == executor->ownArgs.outputs.paramDescs.count)) {
        executor->isOutEmpty = true;
    }
    if (NnopbaseSkipKernelLaunch(executor)) {
        executor->hasTiling = false;
    }

    // 更新控核信息
    if (!executor->matchArgsV2) {
        nnopbase::NnopbaseGetCoreNum(&executor->coreNum.aicNum, &executor->coreNum.aivNum);
        NnopbaseUpdatePlatformInfo(executor);
    }

    // 开启dump不开启缓存匹配
    if ((!g_nnopbaseSysCfgParams.enableArgsCache) || op::internal::GetOpProfilingRecordArgFlag()) {
        executor->ownArgs.enableCache = false;
    } else if (NnopbaseExecutorCachedArgs(executor)) {
        return OK;
    }

    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseExecutorTilingAndUpdateBinInfo(executor));
    if ((executor->args->binInfo->initValues.size() != 0U)) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseExecutorInsertMemsetOp(executor));
    }

    return OK;
}

static void NnopbaseExecutorSaveArgs(NnopbaseExecutor* executor)
{
    executor->args->workspaceNum = executor->workspaces.num;
    executor->args->workspaceLen = executor->workspaces.length;
    executor->args->hasTiling = executor->hasTiling;
    executor->args->isOutEmpty = executor->isOutEmpty;
    executor->args->hasMemset = executor->hasMemset;
}

aclnnStatus NnopbaseExecutorRunForWorkspace(NnopbaseExecutor* executor, uint64_t* workspaceLen)
{
    OP_LOGI("Run op %s for workspace, executor addr %p.", executor->opType, executor);
    op::internal::GetThreadLocalContext().logInfo_.l2ApiName = executor->opType;
    executor->poolIndex = op::internal::GetThreadLocalContext().poolIndex_;
    PrintAttr(executor);
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseExecutorMatchCache(executor));
    NnopbaseExecutorGetWorkspaceSizes(executor, workspaceLen);
    NnopbaseExecutorSaveArgs(executor);
    OP_LOGI("Finish RunForWorkspace for op %s, executor addr %p.", executor->opType, executor);
    return OK;
}

static inline aclnnStatus NnopbaseDumpNodeInfo(NnopbaseExecutor* executor)
{
    OP_LOGI("Start dump node info for operator %s", executor->opType);
    // 支持非Mc2算子或者通信模式为AICORE的Mc2算子落盘
    bool isOpSupportDump = (!executor->mc2.enabled) ||
                           (executor->mc2.enabled &&
                            (executor->mc2.serverType == NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_MTE));
    if (op::internal::opProfilingSwitch.recordOpArgFlag && isOpSupportDump) {
        int8_t binType = 1; // 1表示dynamic
        if (executor->args->binInfo->isStaticShape) {
            NNOPBASE_ASSERT_OK_RETVAL(NnopbaseTilingContextBuild(executor));
            binType = 0; // 0表示static
        }
        OP_LOGD("Dump node info, op type[%s] bin path %s, bin type %d.", executor->opType,
                executor->args->binInfo->binPath.c_str(), binType);
        executor->dfx.opKernelInfo = aclnnOpInfoRecord::OpKernelInfo(executor->args->binInfo->binPath, binType);
        executor->dfx.opKernelInfo.isMc2 = executor->mc2.enabled;
        aclnnOpInfoRecord::OpInfoSerialize(
            op::internal::PtrCastTo<const gert::TilingContext>(executor->tiling.contextExt.context),
            aclnnOpInfoRecord::OpCompilerOption(g_nnopbaseSysCfgParams.implMode, executor->deterministicLevel),
            &executor->dfx.opKernelInfo);
        OP_LOGI("Dump node info for operator %s finished.", executor->opType);
    }
    return OK;
}

static aclnnStatus NnopbaseExecutorLaunchKernel(NnopbaseExecutor* executor, aclrtStream stream,
                                                const uint32_t numBlocks, const bool is195x = false)
{
    aclrtFuncHandle funcHandle;
    uint8_t scheMode = static_cast<uint8_t>(executor->args->tilingInfo.scheMode);
    uint32_t dynUbufSize = executor->args->tilingInfo.dynUbufSize;
    auto attrs = CreateRtsLaunchCfgAttrs(scheMode, dynUbufSize, is195x);
    if (executor->args->binInfo->isStaticShape) {
        OP_LOGI("Launch static kernel %s task, numBlocks is %u, scheMode is %u.", executor->opType, numBlocks,
                scheMode);
        NNOPBASE_ASSERT_RTOK_RETVAL(GetFuncHandleByKernelName(
            executor->args->binInfo->binHandle, executor->args->binInfo->kernelName.c_str(), &funcHandle));
    } else {
        const uint64_t tilingKey = executor->args->tilingInfo.tilingKey;
        OP_LOGI("Launch dynamic kernel %s task, tilingKey is %lu, scheMode is %u, numBlocks is %u, stream is %p",
                executor->opType, tilingKey, scheMode, numBlocks, stream);
        NNOPBASE_ASSERT_OK_RETVAL(GetFuncHandleByEntry(executor->args->binInfo->binHandle, tilingKey, &funcHandle));
    }

    aclrtLaunchKernelCfg cfg;
    cfg.attrs = attrs.data();
    cfg.numAttrs = attrs.size();
    auto rtsHostPlaceHolder = NnopbaseGetRTSPlaceHolder(&executor->argsExt);
    NNOPBASE_ASSERT_OK_RETVAL(aclrtLaunchKernelWithHostArgs(funcHandle, numBlocks, stream, &cfg, executor->argsExt.args,
                                                            executor->argsExt.argsSize, rtsHostPlaceHolder.data(),
                                                            rtsHostPlaceHolder.size()));
    return OK;
}

static aclnnStatus NnopbaseExecutorVectorCoreLaunchKernel(NnopbaseExecutor* executor, aclrtStream stream,
                                                          const uint32_t numBlocks, const uint8_t schemMode,
                                                          const uint32_t numBlocksOffset)
{
    aclrtFuncHandle funcHandle;
    if (executor->args->binInfo->isStaticShape) {
        OP_LOGI("Get static func handle of op %s by kernel name, numBlocks is %u", executor->opType, numBlocks);
        NNOPBASE_ASSERT_RTOK_RETVAL(GetFuncHandleByKernelName(
            executor->args->binInfo->binHandle, executor->args->binInfo->kernelName.c_str(), &funcHandle));
    } else {
        OP_LOGI("Get dynamic func handle of op %s by tilingKey, tilingKey is %lu, numBlocks is %u", executor->opType,
                executor->args->tilingInfo.tilingKey, numBlocks);
        NNOPBASE_ASSERT_RTOK_RETVAL(GetFuncHandleByEntry(executor->args->binInfo->binHandle,
                                                         executor->args->tilingInfo.tilingKey, &funcHandle));
    }

    std::vector<aclrtLaunchKernelAttr> attrs;
    aclrtLaunchKernelAttr schemModeAttr = {
        .id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE,
        .value = {.schemMode = schemMode},
    };
    aclrtLaunchKernelAttr engineTypeAttr = {
        .id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE,
        .value = {.engineType = ACL_RT_ENGINE_TYPE_AIV},
    };
    aclrtLaunchKernelAttr numBlocksOffsetAttr = {
        .id = ACL_RT_LAUNCH_KERNEL_ATTR_BLOCKDIM_OFFSET,
        .value = {.blockDimOffset = numBlocksOffset},
    };
    attrs.push_back(schemModeAttr);
    attrs.push_back(engineTypeAttr);
    attrs.push_back(numBlocksOffsetAttr);

    aclrtLaunchKernelCfg cfg;
    cfg.attrs = attrs.data();
    cfg.numAttrs = attrs.size();
    auto rtsHostPlaceHolder = NnopbaseGetRTSPlaceHolder(&executor->argsExt);
    NNOPBASE_ASSERT_OK_RETVAL(aclrtLaunchKernelWithHostArgs(funcHandle, numBlocks, stream, &cfg, executor->argsExt.args,
                                                            executor->argsExt.argsSize, rtsHostPlaceHolder.data(),
                                                            rtsHostPlaceHolder.size()));
    return OK;
}

static aclnnStatus NnopbaseExecutorLaunchKernelForVectorCore(NnopbaseExecutor* executor, aclrtStream stream,
                                                             const NnopbaseBlockInfoForVectorCore& blockInfo)
{
    OP_LOGI("AicNumBlocks is %u, aivNumBlocks is %u, aivNumBlocksOffset is %u.", blockInfo.aicNumBlocks,
            blockInfo.aivNumBlocks, blockInfo.aivNumBlocksOffset);
    NnopbaseStreamForCombineExecution nnopbaseStream = {};
    std::shared_ptr<std::mutex> streamLckPtr;
    NNOPBASE_ASSERT_RTOK_RETVAL(NnopbaseExecutorGetStreamAndEvent(
        stream, &nnopbaseStream.stream, &nnopbaseStream.eventA, &nnopbaseStream.eventB, streamLckPtr));
    NNOPBASE_ASSERT_NOTNULL_RETVAL(streamLckPtr);
    std::lock_guard<std::mutex> lock(*streamLckPtr);
    NNOPBASE_ASSERT_RTOK_RETVAL(aclrtRecordEvent(nnopbaseStream.eventA, stream));
    NNOPBASE_ASSERT_RTOK_RETVAL(aclrtStreamWaitEvent(nnopbaseStream.stream, nnopbaseStream.eventA));
    const uint64_t launchBeginTime = NnopbaseMsprofSysTime();
    // aicore kernel launch
    NNOPBASE_ASSERT_RTOK_RETVAL(NnopbaseExecutorLaunchKernel(executor, stream, blockInfo.aicNumBlocks, true));
    NnopbaseExecutorReportProfiling(executor, blockInfo.aicNumBlocks, MSPROF_GE_TASK_TYPE_AI_CORE, launchBeginTime,
                                    stream);
    OP_LOGI("Main stream launched successfully.");

    const uint8_t scheMode = static_cast<uint8_t>(executor->args->tilingInfo.scheMode);
    const uint64_t aivLaunchBeginTime = NnopbaseMsprofSysTime();
    // vector core kernel launch
    NNOPBASE_ASSERT_RTOK_RETVAL(NnopbaseExecutorVectorCoreLaunchKernel(
        executor, nnopbaseStream.stream, blockInfo.aivNumBlocks, scheMode, blockInfo.aivNumBlocksOffset));
    NnopbaseExecutorReportProfiling(executor, blockInfo.aivNumBlocks, MSPROF_GE_TASK_TYPE_AIV, aivLaunchBeginTime,
                                    stream);
    NNOPBASE_ASSERT_RTOK_RETVAL(aclrtRecordEvent(nnopbaseStream.eventB, nnopbaseStream.stream));
    NNOPBASE_ASSERT_RTOK_RETVAL(aclrtStreamWaitEvent(stream, nnopbaseStream.eventB));
    OP_LOGI("Sub stream launch successfully.");
    return OK;
}

static aclnnStatus NnopbaseExecutorRunKernelForVectorCore(NnopbaseExecutor* executor, aclrtStream stream,
                                                          uint32_t numBlocks, const CoreType coreType)
{
    NnopbaseBlockInfoForVectorCore blockInfo = {};
    uint32_t deviceId = 0;
    NNOPBASE_ASSERT_RTOK_RETVAL(aclrtGetDevice(op::internal::PtrCastTo<int32_t>(&deviceId)));
    NNOPBASE_ASSERT_RTOK_RETVAL(aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM,
                                                   op::internal::PtrCastTo<int64_t>(&blockInfo.aivNumBlocks)));

    if (coreType == kMixAiCore) { // MIX_AICORE场景 taskRation 1:1 双kernel下发
        blockInfo.aicNumBlocks = numBlocks;
        NNOPBASE_ASSERT_RTOK_RETVAL(NnopbaseExecutorLaunchKernelForVectorCore(executor, stream, blockInfo));
        return OK;
    } else { // MIX_AIV场景 1:1 场景
        NNOPBASE_ASSERT_RTOK_RETVAL(aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_AICORE_CORE_NUM,
                                                       op::internal::PtrCastTo<int64_t>(&blockInfo.aicNumBlocks)));
        const uint32_t coreNum = blockInfo.aicNumBlocks + blockInfo.aivNumBlocks;
        OP_LOGI("AicNumBlocks is %u, aivNumBlocks is %u, aivNumBlocksOffset is %u, numBlocks is %u.",
                blockInfo.aicNumBlocks, blockInfo.aivNumBlocks, blockInfo.aivNumBlocksOffset, numBlocks);
        if (numBlocks <= blockInfo.aicNumBlocks) {
            const uint64_t launchBeginTime = NnopbaseMsprofSysTime();
            // 动静kernel下发
            NNOPBASE_ASSERT_RTOK_RETVAL(NnopbaseExecutorLaunchKernel(executor, stream, numBlocks, true));
            NnopbaseExecutorReportProfiling(executor, numBlocks, MSPROF_GE_TASK_TYPE_AI_CORE, launchBeginTime, stream);
            return OK;
        } else if (numBlocks > coreNum) {
            blockInfo.aicNumBlocks = (numBlocks * blockInfo.aicNumBlocks + coreNum - 1U) / coreNum;
            blockInfo.aivNumBlocks = numBlocks - blockInfo.aicNumBlocks;
        } else {
            blockInfo.aivNumBlocks = numBlocks - blockInfo.aicNumBlocks;
        }
        blockInfo.aivNumBlocksOffset = blockInfo.aicNumBlocks;
        NNOPBASE_ASSERT_RTOK_RETVAL(NnopbaseExecutorLaunchKernelForVectorCore(executor, stream, blockInfo));
    }
    return OK;
}

static inline void NnopbaseExecutorSetWorkspaces(NnopbaseExecutor* executor, void* workspace)
{
    const auto workspacesSizes = NnopbaseGetWorkspacesSizesFromArgs(executor->args);
    executor->workspaces.workspaces[0] = workspace;
    for (size_t i = 1U; i < executor->workspaces.num; i++) {
        auto lastAddr = static_cast<uint8_t*>(executor->workspaces.workspaces[i - 1]);
        executor->workspaces.workspaces[i] = lastAddr + workspacesSizes->GetData()[i - 1];
        // for workspace dump info
        if (((executor->args->binInfo->dfxInfo.isPrintEnable) || (executor->args->binInfo->dfxInfo.isAssertEnable) ||
             (executor->args->binInfo->dfxInfo.isTimeStampEnable)) &&
            (i == 1U)) {
            auto addrBase = static_cast<uint8_t*>(executor->workspaces.workspaces[i]);
            executor->workspaces.workspaces[i] = addrBase + executor->args->binInfo->debugBufSize;
        }
    }
}

aclnnStatus NnopbaseExecutorKernelLaunch(NnopbaseExecutor* executor, aclrtStream stream)
{
    OP_LOGI("Launch kernel.");
    if (NnopbaseSkipKernelLaunch(executor)) {
        return OK;
    }
    CoreType coreType = executor->args->binInfo->coreType;
    const uint64_t launchBeginTime = NnopbaseMsprofSysTime();
    const uint32_t numBlocks = executor->args->binInfo->isStaticShape ? executor->args->binInfo->numBlocks :
                                                                        executor->args->tilingInfo.numBlocks;

    NnopbaseTaskRation taskRation = kRationEnd;
    if (executor->args->binInfo->multiKernelType == 1) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseExecutorGetCoreTypeAndTaskRation(executor, coreType, taskRation));
    }

    // 目前只有310p有这两种coretype，其他形态不会走进来
    if (((coreType == kMixAiCore) || (coreType == kMixAiv)) && (taskRation == kRationEnd || taskRation == kRation11)) {
        NNOPBASE_ASSERT_RTOK_RETVAL(NnopbaseExecutorRunKernelForVectorCore(executor, stream, numBlocks, coreType));
    } else {
        NNOPBASE_ASSERT_RTOK_RETVAL(NnopbaseExecutorLaunchKernel(executor, stream, numBlocks));
        const uint32_t taskType = NnopbaseExecutorGetTaskType(coreType, executor->args->binInfo->taskRation);
        NnopbaseExecutorReportProfiling(executor, numBlocks, taskType, launchBeginTime, stream);
    }

    OP_LOGI("Kernel launch successfully.");
    return OK;
}

aclnnStatus NnopbaseExecutorRefreshOutputShape(NnopbaseExecutor* executor)
{
    uint8_t* outPutShape = op::internal::PtrCastTo<uint8_t>(executor->workspaces.workspaces[0]) +
                           (executor->workspaces.length - executor->args->outputs.outPutShapeSize);
    OP_LOGI("OutputShape addr is %p, outputShape size is %zu.", outPutShape, executor->args->outputs.outPutShapeSize);
    if (executor->args->outputs.outPutShapeSize > executor->outputShapeData.size()) {
        executor->outputShapeData.resize(executor->args->outputs.outPutShapeSize);
    }
    auto ret = memset_s(executor->outputShapeData.data(), executor->args->outputs.outPutShapeSize, 0,
                        executor->args->outputs.outPutShapeSize);
    CHECK_COND(
        ret == EOK, ACLNN_ERR_INNER,
        "Failed to execute memset_s for outputShapeData, ret %d, outputShapeData addr is %p, outPutShapeSize is %zu.",
        ret, executor->outputShapeData.data(), executor->args->outputs.outPutShapeSize);
    ret = aclrtMemcpy(executor->outputShapeData.data(), executor->outputShapeData.size(), outPutShape,
                      executor->args->outputs.outPutShapeSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_COND(ret == ACL_SUCCESS, ACLNN_ERR_RUNTIME_ERROR,
               "Failed to execute aclrtMemcpy for outputShapeData,"
               " ret is %d, src is %p, size is %zu bytes, dst is %p, size is %zu bytes.",
               ret, executor->outputShapeData.data(), executor->outputShapeData.size(), outPutShape,
               executor->args->outputs.outPutShapeSize);
    // outputShape: [output_shape1_dim, output_shape1, output_shape2_dim, output_shape2]
    // output_shape1_dim=2, output_shape1=[32,64,0,0,0,0,0,0]
    uint64_t* hostData = op::internal::PtrCastTo<uint64_t>(executor->outputShapeData.data());
    for (auto it = executor->args->outputs.outPutShapeMap.begin(); it != executor->args->outputs.outPutShapeMap.end();
         ++it) {
        gert::Shape newShape;
        uint64_t dimNum = hostData[0] & 0x7F; // 最高位1表示uint64_t类型，解析dimNum要去掉
        if (dimNum > 8U) {                    // 最大dimNum不能超过8
            std::string reason = "DimNum of " + std::to_string(it->first) + "th output is " + std::to_string(dimNum) +
                                 ", dim number cannot be greater than 8";
            OP_LOGE_FOR_EXECUTION_ERROR_WITHOUT_SOLUTION(reason.c_str());
            return ACLNN_ERR_PARAM_INVALID;
        }
        newShape.SetDimNum(dimNum);
        constexpr uint64_t dimsize = 9; // 9 表示1个count和8个dim
        for (uint64_t i = 0; i < dimNum; i++) {
            newShape.SetDim(i, hostData[i + 1]);
        }
        hostData += dimsize;
        it->second->SetStorageShape(newShape);
        it->second->SetOriginalShape(newShape);
        it->second->SetViewShape(newShape);
        OP_LOGI("Refresh output[%u] shape successfully, shape is %s", it->first, op::ToString(newShape).GetString());
    }
    return OK;
}

aclnnStatus NnopbaseExecutorRunWithWorkspace(NnopbaseExecutor* executor, aclrtStream stream, void* workspace,
                                             const uint64_t workspaceLen)
{
    std::unique_ptr<NnopbaseGuard> guard = nullptr;
    if (!executor->repeatFlag) {
        guard = std::make_unique<NnopbaseGuard>([&executor]() { NnopbaseExecutorClear(executor); });
    }
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor->args);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor->args->binInfo);
    NnopbaseExecutorPrintIo(executor);
    // set workspace & update args
    NnopbaseExecutorSetWorkspaces(executor, workspace);
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseExecutorUpdateAddr(executor, workspace, workspaceLen));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseExecutorPrepareParamsExt(executor, stream));
    if ((executor->hasMemset) && (!executor->isOutEmpty)) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseLaunchMemsetTask(executor, stream));
    }

    // dump input data
    NnopbaseDumpData(executor, Adx::TensorType::INPUT, stream, executor->opType);

    // launch & profiling
    RecordNnopbaseTime(executor, NnopbaseTimeIdx::kBeforeLaunch);
    // sizeInfo绑定下一次launch，此处设置需要紧接着launch函数，dump data内部会调用cpu的launch
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseArgsExceptionDumpAddr(executor));
    if (executor->mc2.enabled) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseMC2KernelLaunch(executor, stream));
    } else {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseExecutorKernelLaunch(executor, stream));
    }
    RecordNnopbaseTime(executor, NnopbaseTimeIdx::kAfterLaunch);

    // dump node info & overflow dump & exception info & output data & workspace
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseDumpNodeInfo(executor));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseOverflowDump(executor, stream));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbasePrepareExceptionDumpInfo(executor, stream));
    NnopbaseDumpData(executor, Adx::TensorType::OUTPUT, stream, executor->opType);
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseDumpWorkspaceData(executor, stream));
    if (executor->args->outputs.outPutShapeSize != 0U) {
        NNOPBASE_ASSERT_RTOK_RETVAL(aclrtSynchronizeStream(stream));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseExecutorRefreshOutputShape(executor));
    }
    op::internal::GetThreadLocalContext().poolIndex_ = executor->poolIndex;
    return OK;
}

void NnopbaseExecutorGetUnContiguousTensors(NnopbaseExecutor* executor, const aclTensorList** inTensors)
{
    auto& inUnContTensors = executor->isCachedArgs ? executor->args->inputs.unContiguousTensors :
                                                     executor->ownArgs.inputs.unContiguousTensors;
    auto& inVec = inUnContTensors.tensors;
    OP_LOGI("Op %s has %zu uncontiguous input, executor addr is %p.", executor->opType, inVec.size(), executor);
    if (!inVec.empty()) {
        inUnContTensors.tensorList.tensors = (&inVec[0U]);
        inUnContTensors.tensorList.size = inVec.size();
        *inTensors = op::internal::PtrCastTo<const aclTensorList>(&inUnContTensors.tensorList);
    } else {
        *inTensors = nullptr;
    }
}

#ifdef __cplusplus
}
#endif
