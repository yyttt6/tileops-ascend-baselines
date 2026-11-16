/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <mutex>
#include "indv_executor.h"
#include "indv_collector.h"
#include "utils/indv_base.h"
#include "utils/indv_debug_assert.h"
#include "utils/indv_soc.h"
#include "securec.h"
#include "mmpa/mmpa_api.h"
#include "profiling/aprof_pub.h"
#include "thread_local_context.h"
#include "op_dfx_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

static std::atomic<uint32_t> gNnopbaseDfxId = 0x60000U;
static std::mutex gNnopbaseDfxIdMtx;
constexpr uint32_t NNOPBASE_BIT_OFFSET = 16U;

inline void NnopbaseRegTypeInfo(NnopbaseDfxId& dfxId)
{
    const std::lock_guard<std::mutex> lock(gNnopbaseDfxIdMtx);
    if (!dfxId.hasReg) {
        dfxId.id = gNnopbaseDfxId++;
        OP_LOGD("Reg TypeInfo, id is %u, funcname %s", dfxId.id, dfxId.funcName);
        (void)MsprofRegTypeInfo(MSPROF_REPORT_ACL_LEVEL, dfxId.id, dfxId.funcName);
        dfxId.hasReg = true;
    }
    return;
}

inline void NnopbaseBuildNodeBasicInfo(const uint32_t numBlocks, const std::pair<uint64_t, uint64_t>& opNameAndTypeHash,
                                       const uint32_t taskType, MsprofCompactInfo& nodeBasicInfo,
                                       const uint64_t timeStamp)
{
    auto& profNodeBasicInfo = nodeBasicInfo.data.nodeBasicInfo;
    profNodeBasicInfo.opName = opNameAndTypeHash.first;
    profNodeBasicInfo.opType = opNameAndTypeHash.second;
    profNodeBasicInfo.taskType = taskType;
    profNodeBasicInfo.blockDim = numBlocks;
    nodeBasicInfo.level = static_cast<uint16_t>(MSPROF_REPORT_NODE_LEVEL);
    nodeBasicInfo.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
    nodeBasicInfo.timeStamp = timeStamp;
    nodeBasicInfo.threadId = static_cast<uint32_t>(mmGetTid());
}

inline void NnopbaseBuildTensor(const MsprofGeTensorType tensor_type, const NnopbaseTensors& tensors, const size_t idx,
                                MsrofTensorData& tensorData)
{
    tensorData.tensorType = tensor_type;
    if (!tensors.extTensors[idx].isNull) {
        const auto& tensor = tensors.extTensors[idx].rt2Tensor;
        tensorData.format = tensor.GetStorageFormat();
        tensorData.dataType = tensor.GetDataType();
        size_t index = 0U;
        const auto& shape = tensor.GetStorageShape();
        for (; (index < MSPROF_GE_TENSOR_DATA_SHAPE_LEN) && (index < shape.GetDimNum()); ++index) {
            tensorData.shape[index] = static_cast<uint32_t>(shape[index]);
        }
        if (index < MSPROF_GE_TENSOR_DATA_SHAPE_LEN) {
            tensorData.shape[index] = 0U;
        }
    } else {
        tensorData.format = ge::FORMAT_NULL;
        tensorData.dataType = ge::DT_UNDEFINED;
        tensorData.shape[0U] = 0U;
    }
}

inline void NnopbaseBuildTensorData(const NnopbaseExecutor* const executor, const size_t index,
                                    MsrofTensorData& tensorData)
{
    const size_t inputSize = executor->args->inputs.num;
    if (index < inputSize) {
        NnopbaseBuildTensor(MSPROF_GE_TENSOR_TYPE_INPUT, executor->args->inputs, index, tensorData);
    } else {
        NnopbaseBuildTensor(MSPROF_GE_TENSOR_TYPE_OUTPUT, executor->args->outputs, index - inputSize, tensorData);
    }
}

inline void NnopbaseBuildCacheTensorInfo(const NnopbaseExecutor* const executor, op::internal::CacheOpInfoBasic* opInfo)
{
    for (uint32_t i = 0U; i < executor->args->inputs.num; ++i) {
        NnopbaseBuildTensor(MSPROF_GE_TENSOR_TYPE_INPUT, executor->args->inputs, i, opInfo->tensorData[i]);
    }
    for (uint32_t i = 0U; i < executor->args->outputs.num; ++i) {
        NnopbaseBuildTensor(MSPROF_GE_TENSOR_TYPE_OUTPUT, executor->args->outputs, i,
                            opInfo->tensorData[i + executor->args->inputs.num]);
    }
}

inline void NnopbaseBuildCacheAttrInfo(NnopbaseExecutor* const executor, op::internal::CacheOpInfoBasic* opInfo)
{
    size_t offset = 0U;
    for (size_t i = 0U; i < executor->attrs.num; ++i) {
        executor->attrs.attrs[i].addr.addr = op::internal::PtrCastTo<void>(executor->args->attrsData.data() + offset);
        offset += executor->attrs.attrs[i].addr.size;
    }
    const std::string attrStr = NnopbaseGetAttrVal(executor->attrs);
    if (!attrStr.empty()) {
        OP_LOGI("Report op [%s] attr info cache: %s.", executor->opType, attrStr.c_str());
        opInfo->attrId = MsprofGetHashId(attrStr.c_str(), attrStr.size());
    }
}

aclnnStatus NnopbaseGetTilingKeyInfo(NnopbaseExecutor* const opExecutor, NnopbaseTaskRation& taskRation,
                                     CoreType& coreType, uint32_t& ration)
{
    static const std::map<NnopbaseTaskRation, uint32_t> RATION_MAP{
        {kRation01, 0U}, {kRation10, 0U}, {kRation11, 1U}, {kRation12, 2U}}; // ration is 2.

    const uint64_t tilingKey = opExecutor->args->tilingInfo.tilingKey;
    if (opExecutor->args->binInfo->tilingKeyInfo.find(tilingKey) != opExecutor->args->binInfo->tilingKeyInfo.end()) {
        coreType = opExecutor->args->binInfo->tilingKeyInfo[tilingKey].coreType;
        taskRation = opExecutor->args->binInfo->tilingKeyInfo[tilingKey].taskRation;
        const auto& iter = RATION_MAP.find(taskRation);
        CHECK_COND((iter != RATION_MAP.end()), ACLNN_ERR_PARAM_INVALID, "Op %s taskRation %d is not supported!",
                   opExecutor->opType, taskRation);
        ration = iter->second;
    } else {
        OP_LOGW("%s cannot find tilingKey %lu from tilingKeyInfo.", opExecutor->opType, tilingKey);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return OK;
}

// 对混合算子，将ration编码到numBlocks高16位
static void NnopbaseEncodeMixOpNumBlocks(NnopbaseExecutor* const executor, uint32_t& numBlocks)
{
    if (executor->args->binInfo->coreType == kMix) {
        uint32_t ration = 2U;
        if (executor->args->binInfo->multiKernelType == 1) {
            NnopbaseTaskRation taskRation;
            CoreType kernelType;
            (void)NnopbaseGetTilingKeyInfo(executor, taskRation, kernelType, ration);
        }
        numBlocks = ((numBlocks & 0xFFFFU) | (ration << NNOPBASE_BIT_OFFSET));
    }
}

void NnopbaseReportCacheOpInfo(NnopbaseExecutor* const executor, uint32_t numBlocks, uint32_t taskType,
                               aclrtStream stream)
{
    aclrtStreamAttrValue value = {};
    value.cacheOpInfoSwitch = 0;
    OP_CHECK(stream != nullptr, OP_LOGW("stream is nullptr, do not support aclGraph profiling capture."), return);
    aclError ret = aclrtGetStreamAttribute(stream, ACL_STREAM_ATTR_CACHE_OP_INFO, &value);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Failed to get stream attribute, ret is [%d]", ret);
        return;
    }
    if (!static_cast<bool>(value.cacheOpInfoSwitch)) {
        OP_LOGI("Profiling for AclGraph is disabled.");
        return;
    }

    NnopbaseEncodeMixOpNumBlocks(executor, numBlocks);

    uint32_t totalNum = executor->args->inputs.num + executor->args->outputs.num;
    uint64_t itemId = executor->dfx.itemId;
    if (taskType == MSPROF_GE_TASK_TYPE_AI_CPU) {
        totalNum = 0U;
        itemId = executor->dfx.aicpuItemId;
    }
    size_t totalSize = sizeof(op::internal::CacheOpInfoBasic) + sizeof(MsrofTensorData) * totalNum;
    void* buffer = malloc(totalSize);
    OP_CHECK(buffer != nullptr, OP_LOGE(ACLNN_ERR_INNER, "Failed to malloc buffer, strerror[%s]", strerror(errno)),
             return);
    (void)memset_s(buffer, totalSize, 0, totalSize);
    op::internal::CacheOpInfoBasic* opInfo = static_cast<op::internal::CacheOpInfoBasic*>(buffer);

    opInfo->taskType = taskType;
    opInfo->numBlocks = numBlocks;
    opInfo->nodeId = itemId;
    opInfo->opType = itemId;
    opInfo->attrId = 0;
    opInfo->opFlag = 0;
    opInfo->tensorNum = totalNum;

    if (taskType != MSPROF_GE_TASK_TYPE_AI_CPU) {
        NnopbaseBuildCacheTensorInfo(executor, opInfo);
        NnopbaseBuildCacheAttrInfo(executor, opInfo);
    }

    ret = aclrtCacheLastTaskOpInfo(buffer, totalSize);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Failed to report operator info cache, ret is [%d]", ret);
    }
    OP_LOGI("Report op [%s] info cache successfully, task type[%u], numBlocks[%u], attrId[%llu] size[%zu]",
            executor->opType, taskType, numBlocks, opInfo->attrId, totalSize);
    free(buffer);
}

static void NnopbaseBuildSingleProfTensorInfo(NnopbaseExecutor* const executor, const size_t index,
                                              const uint32_t tensorNum, MsprofAdditionalInfo& tensorInfo,
                                              const uint64_t timeStamp)
{
    tensorInfo.type = MSPROF_REPORT_NODE_TENSOR_INFO_TYPE;
    tensorInfo.level = static_cast<uint16_t>(MSPROF_REPORT_NODE_LEVEL);
    tensorInfo.timeStamp = timeStamp;
    tensorInfo.threadId = static_cast<uint32_t>(mmGetTid());

    tensorInfo.dataLen = static_cast<uint32_t>(sizeof(MsprofTensorInfo));
    auto profTensorData = op::internal::PtrCastTo<MsprofTensorInfo>(tensorInfo.data);

    profTensorData->opName = executor->dfx.itemId;
    profTensorData->tensorNum = tensorNum;
    for (size_t k = 0UL; k < static_cast<size_t>(tensorNum); ++k) {
        const size_t tensorIndex = (index * static_cast<size_t>(MSPROF_GE_TENSOR_DATA_NUM)) + k;
        NnopbaseBuildTensorData(executor, tensorIndex, profTensorData->tensorData[k]);
    }
}

void NnopbaseReportTensorInfo(NnopbaseExecutor* const executor, const uint64_t timeStamp)
{
    const size_t totalNum = static_cast<size_t>(executor->args->inputs.num + executor->args->outputs.num);
    OP_LOGD("[Cann Profiling]tensor size is %zu", totalNum);

    const size_t index = totalNum / static_cast<size_t>(MSPROF_GE_TENSOR_DATA_NUM);
    for (size_t j = 0UL; j < index; ++j) {
        MsprofAdditionalInfo tensorInfo{};
        NnopbaseBuildSingleProfTensorInfo(executor, j, static_cast<uint32_t>(MSPROF_GE_TENSOR_DATA_NUM), tensorInfo,
                                          timeStamp);
        (void)MsprofReportAdditionalInfo(static_cast<uint32_t>(true), &tensorInfo,
                                         static_cast<uint32_t>(sizeof(MsprofAdditionalInfo)));
    }

    const size_t remainIndex = totalNum % static_cast<size_t>(MSPROF_GE_TENSOR_DATA_NUM);
    if (remainIndex > 0UL) {
        MsprofAdditionalInfo tensorInfo{};
        NnopbaseBuildSingleProfTensorInfo(executor, index, static_cast<uint32_t>(remainIndex), tensorInfo, timeStamp);
        MsprofReportAdditionalInfo(static_cast<uint32_t>(true), &tensorInfo,
                                   static_cast<uint32_t>(sizeof(MsprofAdditionalInfo)));
    }
    return;
}

void NnopbaseReportContextIdInfo(const NnopbaseExecutor* const executor, const uint64_t timeStamp)
{
    MsprofAdditionalInfo info{};

    info.type = MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE;
    info.level = MSPROF_REPORT_NODE_LEVEL;
    info.timeStamp = timeStamp;
    info.threadId = static_cast<uint32_t>(mmGetTid());
    info.dataLen = static_cast<uint32_t>(sizeof(uint32_t));
    auto contextIdInfo = op::internal::PtrCastTo<MsprofContextIdInfo>(info.data);
    contextIdInfo->ctxIdNum = 1U;
    contextIdInfo->ctxIds[0] = 0U;
    contextIdInfo->opName = executor->dfx.itemId;
    (void)MsprofReportAdditionalInfo(static_cast<uint32_t>(true), &info,
                                     static_cast<uint32_t>(sizeof(MsprofAdditionalInfo)));
    OP_LOGD("OP [%s] has reported contextid info.", executor->opType);
    return;
}

void NnopbaseReportContextIdInfoByRation(NnopbaseExecutor* const opExecutor, const uint64_t timeStamp,
                                         uint32_t& numBlocks, uint32_t& taskType)
{
    uint32_t ration = 2U;
    if (opExecutor->args->binInfo->multiKernelType == 1) {
        NnopbaseTaskRation taskRation;
        CoreType kernelType;
        OP_CHECK(NnopbaseGetTilingKeyInfo(opExecutor, taskRation, kernelType, ration) == OK,
                 OP_LOGW("Failed to get tilingKey info."), return);
        if ((taskRation == kRation01 || taskRation == kRation10)) {
            const uint64_t tilingKey = opExecutor->args->tilingInfo.tilingKey;
            if (opExecutor->args->binInfo->tilingKeyInfo[tilingKey].crossCoreSync) {
                numBlocks = ((numBlocks & 0xFFFFU) | (ration << NNOPBASE_BIT_OFFSET));
                NnopbaseReportContextIdInfo(opExecutor, timeStamp);
            } else {
                // 硬同步为0，修改taskType
                taskType = (taskRation == kRation01) ? MSPROF_GE_TASK_TYPE_AIV : MSPROF_GE_TASK_TYPE_AI_CORE;
            }
        } else {
            numBlocks = ((numBlocks & 0xFFFFU) | (ration << NNOPBASE_BIT_OFFSET));
            NnopbaseReportContextIdInfo(opExecutor, timeStamp);
        }
        OP_LOGI("Get op[%s] multiKernelType is [%u], tilingKey is [%lu], kernelType is [%d], taskRation is [%d].",
                opExecutor->opType, opExecutor->args->binInfo->multiKernelType, opExecutor->args->tilingInfo.tilingKey,
                kernelType, taskRation);
    } else {
        // 针对mix算子，低16位为主加速器numBlocks，高16位为从加速器的ratio值
        numBlocks = ((numBlocks & 0xFFFFU) | (ration << NNOPBASE_BIT_OFFSET));
        NnopbaseReportContextIdInfo(opExecutor, timeStamp);
    }
    return;
}

void NnopbaseReportAdditionInfo(void* const executor, uint32_t numBlocks, uint32_t taskType, const uint64_t timeStamp)
{
    if (op::internal::opProfilingSwitch.reportFlag || op::internal::opProfilingSwitch.kernelLaunchFlag) {
        NnopbaseExecutor* const opExecutor = (NnopbaseExecutor*)executor;
        if (opExecutor->args->binInfo->coreType == kMix) { // 310p没有mix类型coretype，不会走进来
            OP_LOGI("Get soc version is %s, set mix op type.",
                    nnopbase::IndvSoc::GetInstance().GetCurSocVersion().c_str());
            NnopbaseReportContextIdInfoByRation(opExecutor, timeStamp, numBlocks, taskType);
        }
        OP_LOGI("[Cann Profiling] node type is %s, taskType is %u", opExecutor->opType, taskType);
        if (op::internal::opProfilingSwitch.additionInfoFlag) {
            const uint64_t typeHash = opExecutor->dfx.itemId;
            MsprofCompactInfo nodeBasicInfo{};
            NnopbaseBuildNodeBasicInfo(numBlocks, {typeHash, typeHash}, taskType, nodeBasicInfo, timeStamp);
            (void)MsprofReportCompactInfo(static_cast<uint32_t>(true), &nodeBasicInfo,
                                          static_cast<uint32_t>(sizeof(MsprofCompactInfo)));
            NnopbaseReportTensorInfo(opExecutor, timeStamp);
        }
    }
    return;
}

void NnopbaseReportMemsetAdditionInfo(const NnopbaseExecutor* const executor, uint32_t numBlocks, uint32_t taskType,
                                      const uint64_t timeStamp)
{
    if (op::internal::opProfilingSwitch.reportFlag || op::internal::opProfilingSwitch.kernelLaunchFlag) {
        OP_LOGI("[Cann Profiling] node type is MemSet, taskType is %u", taskType);
        if (op::internal::opProfilingSwitch.additionInfoFlag) {
            const uint64_t typeHash = executor->dfx.memsetItemId;
            MsprofCompactInfo nodeBasicInfo{};
            NnopbaseBuildNodeBasicInfo(numBlocks, {typeHash, typeHash}, taskType, nodeBasicInfo, timeStamp);
            (void)MsprofReportCompactInfo(static_cast<uint32_t>(true), &nodeBasicInfo,
                                          static_cast<uint32_t>(sizeof(MsprofCompactInfo)));
        }
    }
    return;
}

inline void NnopbaseMsprofReportApi(const uint64_t beginTime, MsprofApi& info)
{
    const uint64_t endTime = NnopbaseMsprofSysTime();
    info.threadId = static_cast<uint32_t>(mmGetTid());
    info.beginTime = beginTime;
    info.endTime = endTime;
    info.magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
    info.reserve = 0U;
    const int32_t res = MsprofReportApi(true, &info);
    OP_LOGD("Call MsprofReportApi res = %d", res);
}

void NnopbaseReportApiInfo(const uint64_t beginTime, NnopbaseDfxId& dfxId)
{
    if (op::internal::opProfilingSwitch.reportFlag) {
        OP_LOGD("Report ApiInfo, api name is %s", dfxId.funcName);
        NnopbaseRegTypeInfo(dfxId);
        MsprofApi info{};
        info.type = dfxId.id;
        info.itemId = 0UL;
        info.level = MSPROF_REPORT_ACL_LEVEL;
        NnopbaseMsprofReportApi(beginTime, info);
    }
}

void NnopbaseReportLaunchInfo(const uint64_t beginTime, const char* const opType)
{
    if (op::internal::opProfilingSwitch.reportFlag || op::internal::opProfilingSwitch.kernelLaunchFlag) {
        OP_LOGD("Report LaunchInfo, op type is %s", opType);
        MsprofApi info{};
        info.type = MSPROF_REPORT_NODE_LAUNCH_TYPE;
        const size_t typeLen = strlen(opType);
        info.itemId = MsprofGetHashId(opType, typeLen);
        info.level = MSPROF_REPORT_NODE_LEVEL;
        NnopbaseMsprofReportApi(beginTime, info);
    }
}

void NnopbaseInnerReportLaunchInfo(const uint64_t beginTime, const uint64_t itemId)
{
    if (op::internal::opProfilingSwitch.reportFlag || op::internal::opProfilingSwitch.kernelLaunchFlag) {
        OP_LOGD("Report LaunchInfo, itemId is %lu", itemId);
        MsprofApi info{};
        info.type = MSPROF_REPORT_NODE_LAUNCH_TYPE;
        info.itemId = itemId;
        info.level = MSPROF_REPORT_NODE_LEVEL;
        NnopbaseMsprofReportApi(beginTime, info);
    }
}

aclnnStatus NnopbaseReportAicpuAdditionInfo(const uint64_t timeStamp, const char* const opType)
{
    if (op::internal::opProfilingSwitch.reportFlag || op::internal::opProfilingSwitch.kernelLaunchFlag) {
        uint32_t numBlocks = 0;
        auto taskType = MSPROF_GE_TASK_TYPE_AI_CPU;

        if (op::internal::opProfilingSwitch.additionInfoFlag) {
            const size_t typeLen = strlen(opType);
            const uint64_t typeHash = MsprofGetHashId(opType, typeLen);
            MsprofCompactInfo nodeBasicInfo{};
            NnopbaseBuildNodeBasicInfo(numBlocks, {typeHash, typeHash}, static_cast<uint32_t>(taskType), nodeBasicInfo,
                                       timeStamp);
            (void)MsprofReportCompactInfo(static_cast<uint32_t>(true), &nodeBasicInfo,
                                          static_cast<uint32_t>(sizeof(MsprofCompactInfo)));
        }
    }
    return OK;
}

void NnopbaseReportTimeStampInfo(const std::vector<MsprofAicTimeStampInfo>& timeStampInfo)
{
    if ((op::internal::opProfilingSwitch.timeStampFlag) && !timeStampInfo.empty()) {
        const size_t batchSize = MSPROF_ADDTIONAL_INFO_DATA_LENGTH / sizeof(MsprofAicTimeStampInfo);
        size_t j = 0U;
        for (size_t i = 0U; i < timeStampInfo.size(); i += batchSize) {
            size_t batchEnd = std::min(i + batchSize, timeStampInfo.size());     // 防止越界
            size_t sizeToCopy = (batchEnd - i) * sizeof(MsprofAicTimeStampInfo); // 计算每次拷贝的字节数
            MsprofAdditionalInfo additionInfo{};
            const uint64_t timeStamp = NnopbaseMsprofSysTime();
            additionInfo.level = MSPROF_REPORT_AIC_LEVEL;
            additionInfo.type = MSPROF_REPORT_AIC_TIMESTAMP_TYPE;
            additionInfo.threadId = static_cast<uint32_t>(mmGetTid());
            additionInfo.timeStamp = timeStamp;
            additionInfo.dataLen = sizeToCopy;
            OP_CHECK(
                memcpy_s(additionInfo.data, MSPROF_ADDTIONAL_INFO_DATA_LENGTH, &timeStampInfo[i], sizeToCopy) == EOK,
                OP_LOGW("Failed to memcpy timestamp additional info, data is %p, timeStampInfo is %p, dataLen is %u",
                        additionInfo.data, &timeStampInfo[i], additionInfo.dataLen),
                return);

            OP_LOGI("Report dataLen is %u.", additionInfo.dataLen);

            (void)MsprofReportAdditionalInfo(static_cast<uint32_t>(true), &additionInfo,
                                             static_cast<uint32_t>(sizeof(MsprofAdditionalInfo)));
            j += (batchEnd - i);
        }
    }
}

std::string NnopbaseGetAttrVal(const NnopbaseAttrs& attrs)
{
    std::string attrStr;
    for (size_t i = 0U; i < attrs.num; i++) {
        attrStr += "attr_" + std::to_string(i) + ":";
        attrStr += nnopbase::ToStr(attrs.attrs[i]);
        if (i + 1 < attrs.num) {
            attrStr += "|";
        }
    }
    return attrStr;
}

static void NnopbaseBuildAttrInfo(MsprofCompactInfo& compactInfo, const uint64_t itemId, const uint64_t id,
                                  const uint64_t timeStamp)
{
    MsprofAttrInfo attrInfo;
    attrInfo.opName = itemId;
    attrInfo.attrType = OP_ATTR;
    attrInfo.hashId = id;

    compactInfo.magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
    compactInfo.level = MSPROF_REPORT_NODE_LEVEL;
    compactInfo.type = MSPROF_REPORT_NODE_ATTR_INFO_TYPE;
    compactInfo.threadId = static_cast<uint32_t>(mmGetTid());
    compactInfo.dataLen = sizeof(MsprofAttrInfo);
    compactInfo.timeStamp = timeStamp;
    OP_CHECK(memcpy_s(compactInfo.data.info, MSPROF_COMPACT_INFO_DATA_LENGTH, &attrInfo, sizeof(MsprofAttrInfo)) == EOK,
             OP_LOGW("Failed to memcpy attr info."), return);
    return;
}

static void NnopbaseReportInfo(const std::string& valStr, const uint64_t itemId, const uint64_t timeStamp)
{
    OP_LOGI("Report str is  %s", valStr.c_str());
    uint64_t id = MsprofGetHashId(valStr.c_str(), valStr.size());
    MsprofCompactInfo compactInfo;
    NnopbaseBuildAttrInfo(compactInfo, itemId, id, timeStamp);
    auto res = MsprofReportCompactInfo(true, &compactInfo, sizeof(compactInfo));
    OP_LOGI("NnopbaseReportInfo, res = %d, compactInfo.timeStamp = %lu", res, compactInfo.timeStamp);
}

static void NnopbaseReportAttrInfo(const NnopbaseExecutor* const executor, const uint64_t timeStamp)
{
    std::string attrStr = NnopbaseGetAttrVal(executor->attrs);
    if (!attrStr.empty()) {
        NnopbaseReportInfo(attrStr, executor->dfx.itemId, timeStamp);
    }
}

static void NnopbaseGetHostInputStr(const NnopbaseTensor& tensor, const size_t irIndex, std::string& inputStr)
{
    if ((!tensor.isNull) && (tensor.rt2Tensor.GetPlacement() == gert::kOnHost)) {
        inputStr += "input_" + std::to_string(irIndex) + ":";
        inputStr += nnopbase::ToStr(tensor.rt2Tensor);
    }
}

static void NnopbaseGetDynamicHostInputStr(const NnopbaseTensors& tensors, const size_t irIndex, std::string& inputStr)
{
    const size_t startIndex = tensors.paramDescs.instances[irIndex].startIndex;
    const size_t size = tensors.paramDescs.instances[irIndex].num;
    for (size_t k = 0U; k < size; k++) {
        if ((!tensors.extTensors[startIndex + k].isNull) &&
            (tensors.extTensors[startIndex + k].rt2Tensor.GetPlacement() == gert::kOnHost)) {
            inputStr += "input_" + std::to_string(irIndex) + "_" + std::to_string(k) + ":";
            inputStr += nnopbase::ToStr(tensors.extTensors[startIndex + k].rt2Tensor);
        }
        if (k + 1 < size) {
            inputStr += "|";
        }
    }
}

static std::string NnopbaseGetHostTensorStr(const NnopbaseTensors& tensors, const size_t irIndex, size_t& tensorIndex)
{
    std::string inputStr;
    if (!tensors.paramDescs.instances[irIndex].isDynamic) {
        NnopbaseGetHostInputStr(tensors.extTensors[tensorIndex], irIndex, inputStr);
        tensorIndex += 1U;
    } else {
        NnopbaseGetDynamicHostInputStr(tensors, irIndex, inputStr);
        tensorIndex += tensors.paramDescs.instances[irIndex].num;
    }
    if (!inputStr.empty()) {
        inputStr += "|";
    }
    return inputStr;
}

std::string NnopbaseGetHostInfoStr(const NnopbaseTensors& tensors)
{
    std::string inputStr;
    size_t j = 0U;
    for (size_t i = 0U; i < tensors.paramDescs.count; i++) {
        inputStr += NnopbaseGetHostTensorStr(tensors, i, j);
    }
    if (!inputStr.empty() && inputStr.back() == '|') {
        inputStr.pop_back();
    }
    return inputStr;
}

static void NnopbaseReportHostInfo(const NnopbaseExecutor* const executor, const uint64_t timeStamp)
{
    std::string inputStr = NnopbaseGetHostInfoStr(executor->args->inputs);
    if (!inputStr.empty()) {
        NnopbaseReportInfo(inputStr, executor->dfx.itemId, timeStamp);
    }
}

void NnopbaseReportAttrAndHostInfo(const NnopbaseExecutor* const executor, const uint64_t timeStamp)
{
    if (op::internal::opProfilingSwitch.level2ProfilingFlag) {
        NnopbaseReportAttrInfo(executor, timeStamp);
        NnopbaseReportHostInfo(executor, timeStamp);
    }
}

#ifdef __cplusplus
}
#endif
