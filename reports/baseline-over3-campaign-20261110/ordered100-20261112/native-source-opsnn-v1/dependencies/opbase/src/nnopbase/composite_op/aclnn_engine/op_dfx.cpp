/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>
#include <string>
#include <vector>
#include <mutex>

#include "acl/acl_rt.h"
#include "mmpa/mmpa_api.h"
#include "dump/adump_pub.h"
#include "profiling/aprof_pub.h"
#include "platform/soc_spec.h"
#include "register/op_binary_resource_manager.h"

#include "opdev/op_def.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "op_info_serialize.h"
#include "kernel_launcher.h"
#include "op_dfx_internal.h"
#include "non_finite_check_op.h"
#include "utils/string_utils.h"
#include "dlopen_api.h"

namespace op {

static std::atomic<uint32_t> opTypeCounter = 1; // 0 作为默认值使用，不要占用
static std::atomic<uint32_t> opProfilingCounter = 0x50000;
static std::atomic<uint64_t> opLogSequenceCounter = 0;
static std::map<std::string, uint32_t> profilingTable;
static thread_local int oPProfilingTid = 0;
static std::mutex profilingTableMutex;
static std::mutex opProfilingSwitchMutex;

namespace internal {
OpProfilingSwitch opProfilingSwitch;
}
ge::AscendString ToString(const std::string& str) { return ge::AscendString(str.c_str()); }

constexpr uint32_t OP_RESOURCE_FUNC_IDX = 0;
constexpr uint32_t OP_RESOURCE_BINARY_IDX = 1;
constexpr uint32_t OP_RESOURCE_RUNTIME_KB_IDX = 2;

uint32_t GenOpTypeId(const char* opName, const OP_RESOURCES& opResources)
{
    for (const auto& [op_type, op_res] : opResources) {
        // 注册算子用到的资源
        nnopbase::OpBinaryResourceManager::GetInstance().AddOpFuncHandle(op_type,
                                                                         std::get<OP_RESOURCE_FUNC_IDX>(op_res));
        nnopbase::OpBinaryResourceManager::GetInstance().AddBinary(op_type, std::get<OP_RESOURCE_BINARY_IDX>(op_res));
        nnopbase::OpBinaryResourceManager::GetInstance().AddRuntimeKB(op_type,
                                                                      std::get<OP_RESOURCE_RUNTIME_KB_IDX>(op_res));
        if (ge::AscendString(opName) != op_type) {
            GenOpTypeId(op_type.GetString());
        }
    }
    return GenOpTypeId(opName);
}

uint32_t GenOpTypeId(const char* opName, const OP_SOC_RESOURCES& opResources)
{
    ge::AscendString socVersion = ToString(GetCurrentPlatformInfo().GetSocVersion());
    std::string tmp = socVersion.GetString();
    std::transform(tmp.begin(), tmp.end(), tmp.begin(), ::tolower);
    for (const auto& [op_type, op_res] : opResources) {
        // 注册算子用到的资源
        auto it = std::get<OP_RESOURCE_BINARY_IDX>(op_res).find(tmp.c_str());
        if (it == std::get<OP_RESOURCE_BINARY_IDX>(op_res).end()) {
            OP_LOGW("failed to find op resources, soc version %s", socVersion.GetString());
            return 0;
        }
        nnopbase::OpBinaryResourceManager::GetInstance().AddOpFuncHandle(op_type,
                                                                         std::get<OP_RESOURCE_FUNC_IDX>(op_res));
        nnopbase::OpBinaryResourceManager::GetInstance().AddBinary(op_type, it->second);
        nnopbase::OpBinaryResourceManager::GetInstance().AddRuntimeKB(op_type,
                                                                      std::get<OP_RESOURCE_RUNTIME_KB_IDX>(op_res));
        if (ge::AscendString(opName) != op_type) {
            GenOpTypeId(op_type.GetString());
        }
    }
    return GenOpTypeId(opName);
}

uint32_t GenOpTypeId(const char* opName)
{
    if (opName == nullptr) {
        OP_LOGW("opName is nullptr when genOpTypeId.");
        return 0;
    }
    string opNameString = string(opName);
    TrimWith(opNameString, ' ');
    if (opNameString.empty()) {
        OP_LOGW("opName is empty when genOpTypeId.");
        return 0;
    }
    uint32_t id = 0;
    if (OpTypeDict::Add(id, opName) == ACLNN_SUCCESS) {
        return id;
    }
    OP_LOGW("The opName %s is added repeatedly.", opName);
    return 0;
}

namespace internal {

// OpProfilingSwitch 线程安全访问辅助函数实现 (仅针对 recordOpArgFlag)
bool GetOpProfilingRecordArgFlag()
{
    std::lock_guard<std::mutex> lock(opProfilingSwitchMutex);
    return opProfilingSwitch.recordOpArgFlag;
}

void SetOpProfilingRecordArgFlag(bool value)
{
    std::lock_guard<std::mutex> lock(opProfilingSwitchMutex);
    opProfilingSwitch.recordOpArgFlag = value;
}

std::atomic<bool> g_aclnnDebugEnabled{false};
static std::once_flag g_aclnnDebugOnceFlag;

void InitAclnnDebugSwitch()
{
    std::call_once(g_aclnnDebugOnceFlag, []() {
        if (CheckLogLevelInner(OP_ID, OP_LOG_INFO) == 1) {
            g_aclnnDebugEnabled.store(true, std::memory_order_relaxed);
            OP_LOGI("aclnn debug mode enabled by log level");
        }
    });
}

OpThreadLocalContext& GetThreadLocalContext()
{
    thread_local static OpThreadLocalContext oPThreadLocalContext;
    return oPThreadLocalContext;
}

SystemConfig systemConfig;

uint64_t OpGetLogSequence()
{
    uint64_t res = opLogSequenceCounter++;
    return res;
}

void OpCacheTid() { oPProfilingTid = mmGetTid(); }

int OpGetTid()
{
    if (oPProfilingTid == 0) {
        OpCacheTid();
    }
    return oPProfilingTid;
}

uint32_t CollectProfilingStr(const char* s)
{
    if (s == nullptr) {
        return 0;
    }
    const std::lock_guard<std::mutex> lock(profilingTableMutex);
    if (profilingTable.find(s) != profilingTable.end()) {
        return profilingTable[s];
    }
    uint32_t id = opProfilingCounter++;
    int32_t res = MsprofRegTypeInfo(MSPROF_REPORT_ACL_LEVEL, id, s);
    OP_LOGI("MsprofRegTypeInfo, id = %u, s = %s", id, s);
    if (res != 0) {
        OP_LOGW("Failed to MsprofRegTypeInfo, id = %u, s = %s", id, s);
    }
    profilingTable[s] = id;
    return id;
}

uint64_t GenKernelLauncherId(const char* l0Name)
{
    if (l0Name == nullptr) {
        return 0;
    }
    uint64_t id = MsprofGetHashId(l0Name, strlen(l0Name));
    OP_LOGI("GenKernelLauncherId, id = %lu, l0Name = %s", id, l0Name);
    return id;
}

uint64_t GenSummaryItemId(const char* l2Name, const char* l0Name)
{
    if (l2Name == nullptr || l0Name == nullptr) {
        OP_LOGW("Failed to call GenSummaryItemId, l2Name = %p, l0Name = %p", l2Name, l0Name);
        return 0;
    }
    std::string combined = std::string(l2Name) + "_" + l0Name;
    uint64_t id = MsprofGetHashId(combined.c_str(), combined.size());
    OP_LOGI("GenSummaryItemId, l2Name = %s, l0Name = %s, id = %lu", l2Name, l0Name, id);
    return id;
}

uint64_t GenSummaryItemId(const char* l2Name, const char* l0Name, const char* opType)
{
    if (l2Name == nullptr || l0Name == nullptr || opType == nullptr) {
        OP_LOGW("Failed to call GenSummaryItemId, l2Name = %p, l0Name = %p, opType = %p", l2Name, l0Name, opType);
        return 0;
    }
    uint64_t l2NameLen = strlen(l2Name);
    uint64_t l0NameLen = strlen(l0Name);
    uint64_t opTypeLen = strlen(opType);
    // first and second 1 for '_', last 1 for '\0'
    uint64_t totalSize = l2NameLen + l0NameLen + opTypeLen + 1 + 1 + 1;
    auto hashHolder = std::unique_ptr<char[]>(new (std::nothrow) char[totalSize]);
    OP_CHECK_NOTNULL(hashHolder);
    auto hashPtr = hashHolder.get();
    auto rc = strncpy_s(hashPtr, totalSize, l2Name, l2NameLen);
    CHECK_COND(rc == EOK, ACLNN_ERR_INNER, "construct hash info l2 name: %s failed.", l2Name);
    rc = strncpy_s(hashPtr + l2NameLen, totalSize - l2NameLen, "_", 1);
    CHECK_COND(rc == EOK, ACLNN_ERR_INNER, "construct hash info delimiter failed.");
    rc = strncpy_s(hashPtr + l2NameLen + 1, totalSize - l2NameLen - 1, l0Name, l0NameLen);
    CHECK_COND(rc == EOK, ACLNN_ERR_INNER, "construct hash info l0 name: %s failed.", l0Name);
    rc = strncpy_s(hashPtr + l2NameLen + l0NameLen + 1, totalSize - l2NameLen - l0NameLen - 1, "_", 1);
    CHECK_COND(rc == EOK, ACLNN_ERR_INNER, "construct hash info delimiter failed.");
    rc = strncpy_s(hashPtr + l2NameLen + l0NameLen + 1 + 1, totalSize - l2NameLen - l0NameLen - 1 - 1, opType,
                   opTypeLen);
    CHECK_COND(rc == EOK, ACLNN_ERR_INNER, "construct hash info opType: %s failed.", opType);
    uint64_t id = MsprofGetHashId(hashPtr, strlen(hashPtr));
    OP_LOGI("GenSummaryItemId, l2Name = %s, l0Name = %s, opType = %s, id = %lu", l2Name, l0Name, opType, id);
    return id;
}

int32_t RecordOpArgCallbacker::RecordOpArgCallback([[maybe_unused]] uint64_t dumpSwitch,
                                                   [[maybe_unused]] char* dumpConfig, [[maybe_unused]] int32_t size)
{
    SetOpProfilingRecordArgFlag((dumpSwitch & Adx::OP_INFO_RECORD_DUMP) != 0 ? true : false);
    OP_LOGI("RecordOpArgCallback, get record op arg flag = %d", GetOpProfilingRecordArgFlag());
    return 0;
}

int32_t RecordOpArgCallbacker::RecordOpArgDump([[maybe_unused]] uint64_t dumpSwitch, [[maybe_unused]] char* dumpConfig,
                                               [[maybe_unused]] int32_t size)
{
    SetOpProfilingRecordArgFlag((dumpSwitch & Adx::OP_INFO_RECORD_DUMP) != 0 ? true : false);
    if (!GetOpProfilingRecordArgFlag()) {
        OP_LOGI("Record stop, start RecordOpArgDump");
        int32_t res = aclnnOpInfoRecord::OpInfoDump();
        CHECK_COND(res == EOK, ACLNN_ERR_INNER, "OpInfoDump filed");
    }
    return 0;
}

OpProfilingSwitch::OpProfilingSwitch()
    : reportFlag(false),
      kernelLaunchFlag(false),
      additionInfoFlag(false),
      recordOpArgFlag(false),
      level2ProfilingFlag(false),
      timeStampFlag(false)
{
    Adx::AdumpRegisterCallback(OP_ID, reinterpret_cast<Adx::AdumpCallback>(RecordOpArgCallbacker::RecordOpArgCallback),
                               reinterpret_cast<Adx::AdumpCallback>(RecordOpArgCallbacker::RecordOpArgDump));
}

int32_t ProfilingCallBack(uint32_t type, VOID_PTR data, uint32_t len)
{
    if (data == nullptr) {
        OP_LOGW("data is nullptr.");
        return -1;
    }
    if (len != sizeof(MsprofCommandHandle)) {
        OP_LOGW("len(%u) != sizeof MsprofCommandHandle(%zu).", len, sizeof(MsprofCommandHandle));
        return -1;
    }

    if (type != PROF_CTRL_SWITCH) {
        OP_LOGI("ProfilingCallBack, type = %u, discard this type.", type);
        return 0;
    }

    MsprofCommandHandle* handle = (MsprofCommandHandle*)data;

    if (handle->type != PROF_COMMANDHANDLE_TYPE_START && handle->type != PROF_COMMANDHANDLE_TYPE_STOP) {
        OP_LOGI("ProfilingCallBack, handle type = %u, discard this type.", handle->type);
        return 0;
    }

    if (handle->type == PROF_COMMANDHANDLE_TYPE_STOP) {
        opProfilingSwitch.reportFlag = false;
        opProfilingSwitch.kernelLaunchFlag = false;
        opProfilingSwitch.additionInfoFlag = false;
        opProfilingSwitch.level2ProfilingFlag = false;
        opProfilingSwitch.timeStampFlag = false;
        OP_LOGI("After ProfilingCallBack Stop, reportFlag=%d, kernelLaunchFlag=%d, additionInfoFlag=%d, "
                "level2ProfilingFlag=%d, timeStampFlag=%d",
                opProfilingSwitch.reportFlag, opProfilingSwitch.kernelLaunchFlag, opProfilingSwitch.additionInfoFlag,
                opProfilingSwitch.level2ProfilingFlag, opProfilingSwitch.timeStampFlag);
        return 0;
    }
    OP_LOGI("ProfilingCallBack, profSwitch = %lu", handle->profSwitch);

    OP_LOGI("Before ProfilingCallBack Update, reportFlag=%d, kernelLaunchFlag=%d, additionInfoFlag=%d, "
            "level2ProfilingFlag=%d",
            opProfilingSwitch.reportFlag, opProfilingSwitch.kernelLaunchFlag, opProfilingSwitch.additionInfoFlag,
            opProfilingSwitch.level2ProfilingFlag);

    opProfilingSwitch.reportFlag = (handle->profSwitch & PROF_ACL_API_MASK) != 0;
    opProfilingSwitch.kernelLaunchFlag = (handle->profSwitch & PROF_TASK_TIME_MASK) != 0;
    opProfilingSwitch.additionInfoFlag = (handle->profSwitch & PROF_TASK_TIME_L1_MASK) != 0;
    opProfilingSwitch.level2ProfilingFlag = ((handle->profSwitch & PROF_TASK_TIME_L2_MASK) != 0 ||
                                             (handle->profSwitch & PROF_OP_ATTR_MASK) != 0);
    opProfilingSwitch.timeStampFlag = handle->profSwitch & PROF_OP_TIMESTAMP_MASK;

    OP_LOGI("After ProfilingCallBack Update, reportFlag=%d, kernelLaunchFlag=%d, additionInfoFlag=%d, "
            "level2ProfilingFlag=%d, timeStampFlag=%d",
            opProfilingSwitch.reportFlag, opProfilingSwitch.kernelLaunchFlag, opProfilingSwitch.additionInfoFlag,
            opProfilingSwitch.level2ProfilingFlag, opProfilingSwitch.timeStampFlag);

    if ((opProfilingSwitch.reportFlag || opProfilingSwitch.kernelLaunchFlag || opProfilingSwitch.timeStampFlag) &&
        opProfilingSwitch.additionInfoFlag) {
        int32_t res = MsprofRegTypeInfo(MSPROF_REPORT_NODE_LEVEL, MSPROF_REPORT_NODE_BASIC_INFO_TYPE,
                                        "node_basic_info");
        if (res != 0) {
            OP_LOGW("MsprofRegTypeInfo node basic info return failed, res = %d", res);
            return -1;
        }
    }
    if (opProfilingSwitch.level2ProfilingFlag) {
        int32_t res = MsprofRegTypeInfo(MSPROF_REPORT_NODE_LEVEL, MSPROF_REPORT_NODE_ATTR_INFO_TYPE, "node_attr_info");
        if (res != 0) {
            OP_LOGW("MsprofRegTypeInfo node attr info return failed, res = %d", res);
            return -1;
        }
    }
    return 0;
}

int32_t oPProfilingReg = MsprofRegisterCallback(61, ProfilingCallBack);

void PrepareBasicInfo(MsprofCompactInfo& compactInfo, MsprofGeTaskType taskType, uint64_t id, uint64_t summaryId)
{
    compactInfo.magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
    compactInfo.level = MSPROF_REPORT_NODE_LEVEL;
    compactInfo.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
    compactInfo.threadId = OpGetTid();
    compactInfo.dataLen = sizeof(MsprofNodeBasicInfo);
    compactInfo.timeStamp = op::internal::GetThreadLocalContext().kernelLauncherStartTime_;
    compactInfo.data.nodeBasicInfo.opName = summaryId;
    compactInfo.data.nodeBasicInfo.taskType = taskType;
    compactInfo.data.nodeBasicInfo.opType = id;
    uint32_t numBlocks = op::internal::GetThreadLocalContext().numBlocks_;
    OP_LOGI("numBlocks is %u before calculation", numBlocks);
    if (taskType == MSPROF_GE_TASK_TYPE_MIX_AIC) {
        constexpr uint32_t constTwo = 2U;
        numBlocks = ((numBlocks & 0xFFFFU) | (constTwo << 16U));
    }
    OP_LOGI("numBlocks is %u after calculation", numBlocks);
    (taskType == MSPROF_GE_TASK_TYPE_AI_CPU || taskType == MSPROF_GE_TASK_TYPE_DSA) ?
        compactInfo.data.nodeBasicInfo.blockDim = 0 :
        compactInfo.data.nodeBasicInfo.blockDim = numBlocks;
    compactInfo.data.nodeBasicInfo.opFlag = 0;
    auto res = MsprofReportCompactInfo(true, &compactInfo, sizeof(compactInfo));
    OP_LOGI("PrepareBasicInfo, res = %d, compactInfo.timeStamp = %lu, nodeBasicInfo.taskType = %u.", res,
            compactInfo.timeStamp, compactInfo.data.nodeBasicInfo.taskType);
}

void PrepareBasicInfo(MsprofCompactInfo& compactInfo, const TaskInfo& taskInfo, uint64_t id, uint64_t summaryId)
{
    compactInfo.magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
    compactInfo.level = MSPROF_REPORT_NODE_LEVEL;
    compactInfo.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
    compactInfo.threadId = OpGetTid();
    compactInfo.dataLen = sizeof(MsprofNodeBasicInfo);
    compactInfo.timeStamp = op::internal::GetThreadLocalContext().kernelLauncherStartTime_;
    compactInfo.data.nodeBasicInfo.opName = summaryId;
    compactInfo.data.nodeBasicInfo.taskType = taskInfo.type;
    compactInfo.data.nodeBasicInfo.opType = id;
    uint32_t numBlocks = op::internal::GetThreadLocalContext().numBlocks_;
    OP_LOGI("numBlocks is %u, taskInfo.ration is %u", numBlocks, taskInfo.ration);
    if (taskInfo.ration == OP_DFX_TASK_RATION_TWO) {
        constexpr uint32_t constTwo = 2U;
        numBlocks = ((numBlocks & 0xFFFFU) | (constTwo << 16U));
    } else if (taskInfo.ration == 1) {
        constexpr uint32_t constOne = 1U;
        numBlocks = ((numBlocks & 0xFFFFU) | (constOne << 16U));
    }
    OP_LOGI("numBlocks is %u after calculation", numBlocks);
    compactInfo.data.nodeBasicInfo.blockDim = numBlocks;
    compactInfo.data.nodeBasicInfo.opFlag = (static_cast<uint32_t>(taskInfo.execMode) &
                                             static_cast<uint32_t>(OpExecMode::OP_EXEC_MODE_HF32)) != 0 ?
                                                1 :
                                                0;
    auto res = MsprofReportCompactInfo(true, &compactInfo, sizeof(compactInfo));
    OP_LOGI("PrepareBasicInfo, res = %d, compactInfo.timeStamp = %lu, nodeBasicInfo.taskType = %u.", res,
            compactInfo.timeStamp, compactInfo.data.nodeBasicInfo.taskType);
}

void GetCacheOpInfoSwitch([[maybe_unused]] const aclrtStream& stream)
{
    aclrtStreamAttr stmAttrType = ACL_STREAM_ATTR_CACHE_OP_INFO;
    aclrtStreamAttrValue value = {};
    value.cacheOpInfoSwitch = 0;
    if (stream != nullptr) {
        OP_CHECK_NO_RETURN(aclrtGetStreamAttribute(stream, stmAttrType, &value) == ACL_SUCCESS,
                           OP_LOGW("aclrtGetStreamAttribute is not working as expected."));
    } else {
        OP_LOGI("stream is nullptr, cannot get cache op info switch from stream attribute.");
    }
#if defined(NNOPBASE_UT) || defined(NNOPBASE_ST)
    op::internal::GetThreadLocalContext().cacheOpInfoSwitch_ = true;
#else
    op::internal::GetThreadLocalContext().cacheOpInfoSwitch_ = static_cast<bool>(value.cacheOpInfoSwitch);
#endif
}

static void ReportCacheOpInfoTensor(uint8_t* dest, uint64_t& destOffset, const uint32_t& totalSize,
                                    const FVector<const aclTensor*>& tensors, MsprofGeTensorType type)
{
    for (size_t i = 0; i < tensors.size(); i++) {
        // MsrofTensorData: tensorType/format/dataType/shape
        MsrofTensorData msTensor;
        msTensor.tensorType = static_cast<uint32_t>(type);
        msTensor.format = static_cast<uint32_t>(tensors[i]->GetStorageFormat());
        msTensor.dataType = static_cast<uint32_t>(tensors[i]->GetDataType());
        auto& tensorShape = tensors[i]->GetStorageShape();
        size_t dim = tensorShape.GetDimNum();
        for (size_t j = 0; j < dim && j < MSPROF_GE_TENSOR_DATA_SHAPE_LEN; j++) {
            msTensor.shape[j] = tensorShape[j];
        }
        for (size_t j = dim; j < MSPROF_GE_TENSOR_DATA_SHAPE_LEN; j++) {
            msTensor.shape[j] = 0;
        }
        OP_CHECK(memcpy_s(dest + destOffset, totalSize - destOffset, &msTensor, sizeof(MsrofTensorData)) == EOK,
                 OP_LOGE(ACLNN_ERR_INNER, "call memcpy_s failed."),
                 throw std::runtime_error("aclGraph profiling memcpy runtime error."));
        destOffset += sizeof(MsrofTensorData);
        OP_LOGI("tensorIndex %zu, tensorType %u, format %u, dataType %u, shape(%u, %u, %u, %u, %u, %u, %u, %u)", i,
                msTensor.tensorType, msTensor.format, msTensor.dataType, msTensor.shape[0], msTensor.shape[1],
                msTensor.shape[2], msTensor.shape[3], msTensor.shape[4], msTensor.shape[5], msTensor.shape[6],
                msTensor.shape[7]);
    }
}

void ReportCacheOpInfo(const TaskInfo& taskInfo, OpArgContext* args, const uint64_t& opType, const uint64_t& attrId)
{
    CacheOpInfoBasic cacheOpInfoBasic;
    OpArgList& inputs = *args->GetOpArg(op::OP_INPUT_ARG);
    OpArgList& outputs = *args->GetOpArg(op::OP_OUTPUT_ARG);
    FVector<const aclTensor*> inTensors;
    FVector<const aclTensor*> outTensors;
    GetProfilingTensors(inputs, inTensors);
    GetProfilingTensors(outputs, outTensors);
    uint32_t tensorNum = inTensors.size() + outTensors.size();
    size_t totalSize = sizeof(CacheOpInfoBasic) + (sizeof(MsrofTensorData) * tensorNum);
    void* infoPtr = op::internal::Allocate(totalSize);
    OP_CHECK(infoPtr != nullptr, OP_LOGE(ACLNN_ERR_INNER, "infoPtr allocate failed."), throw std::bad_alloc());
    uint8_t* dest = static_cast<uint8_t*>(infoPtr);
    uint64_t destOffset = 0;
    cacheOpInfoBasic.attrId = attrId;
    cacheOpInfoBasic.taskType = static_cast<uint32_t>(taskInfo.type);
    cacheOpInfoBasic.nodeId = GenSummaryItemId(GetThreadLocalContext().logInfo_.l2ApiName,
                                               GetThreadLocalContext().logInfo_.l0Name,
                                               op::OpTypeDict::ToString(opType).GetString());
    cacheOpInfoBasic.opType = GenKernelLauncherId(op::OpTypeDict::ToString(opType).GetString());
    uint32_t numBlocks = op::internal::GetThreadLocalContext().numBlocks_;
    OP_LOGI("numBlocks is %u, taskInfo.ration is %u", numBlocks, taskInfo.ration);
    if (taskInfo.ration == OP_DFX_TASK_RATION_TWO) {
        numBlocks = ((numBlocks & 0xFFFFU) | (0x20000));
    } else if (taskInfo.ration == 1) {
        numBlocks = ((numBlocks & 0xFFFFU) | (0x10000));
    }
    cacheOpInfoBasic.numBlocks = numBlocks;
    OP_LOGI("numBlocks is %u after calculation", numBlocks);
    cacheOpInfoBasic.opFlag = (static_cast<uint32_t>(taskInfo.execMode) &
                               static_cast<uint32_t>(OpExecMode::OP_EXEC_MODE_HF32)) != 0 ?
                                  1 :
                                  0;
    cacheOpInfoBasic.tensorNum = tensorNum;
    OP_CHECK(memcpy_s(dest + destOffset, totalSize - destOffset, &cacheOpInfoBasic, sizeof(CacheOpInfoBasic)) == EOK,
             OP_LOGE(ACLNN_ERR_INNER, "call memcpy_s failed."),
             throw std::runtime_error("aclGraph profiling memcpy runtime error."));
    destOffset += sizeof(CacheOpInfoBasic);
    OP_LOGI("taskType %u, nodeId %zu, opType %zu, attrId %zu, numBlocks %u, opFlag %u, tensorNum %u",
            cacheOpInfoBasic.taskType, cacheOpInfoBasic.nodeId, cacheOpInfoBasic.opType, cacheOpInfoBasic.attrId,
            cacheOpInfoBasic.numBlocks, cacheOpInfoBasic.opFlag, cacheOpInfoBasic.tensorNum);
    ReportCacheOpInfoTensor(dest, destOffset, totalSize, inTensors, MSPROF_GE_TENSOR_TYPE_INPUT);
    ReportCacheOpInfoTensor(dest, destOffset, totalSize, outTensors, MSPROF_GE_TENSOR_TYPE_OUTPUT);
    OP_CHECK_NO_RETURN(aclrtCacheLastTaskOpInfo(infoPtr, totalSize) == ACL_SUCCESS,
                       OP_LOGW("aclrtCacheLastTaskOpInfo failed"));
    op::internal::DeAllocate(infoPtr);
}

void ReportCacheOpInfoDSA(OpArgList& inputs, OpArgList& outputs, MsprofGeTaskType taskType)
{
    OP_LOGI("Entering function ReportCacheOpInfoDSA.");
    CacheOpInfoBasic cacheOpInfoBasic;
    FVector<const aclTensor*> inTensors;
    FVector<const aclTensor*> outTensors;
    GetProfilingTensors(inputs, inTensors);
    GetProfilingTensors(outputs, outTensors);
    uint32_t tensorNum = inTensors.size() + outTensors.size();
    size_t totalSize = sizeof(CacheOpInfoBasic) + (sizeof(MsrofTensorData) * tensorNum);
    void* infoPtr = op::internal::Allocate(totalSize);
    OP_CHECK(infoPtr != nullptr, OP_LOGE(ACLNN_ERR_INNER, "infoPtr allocate failed."), throw std::bad_alloc());
    uint8_t* dest = static_cast<uint8_t*>(infoPtr);
    uint64_t destOffset = 0;
    // CacheOpInfoBasic: taskType/nodeId/opType/numBlocks/opFlag/tensorNum
    cacheOpInfoBasic.taskType = static_cast<uint32_t>(taskType);
    cacheOpInfoBasic.nodeId = GenSummaryItemId(GetThreadLocalContext().logInfo_.l2ApiName,
                                               GetThreadLocalContext().logInfo_.l0Name,
                                               GetThreadLocalContext().logInfo_.l0Name);
    cacheOpInfoBasic.opType = GenKernelLauncherId(GetThreadLocalContext().logInfo_.l0Name);
    cacheOpInfoBasic.numBlocks = 0;
    cacheOpInfoBasic.opFlag = 0;
    cacheOpInfoBasic.tensorNum = tensorNum;
    OP_CHECK(memcpy_s(dest + destOffset, totalSize - destOffset, &cacheOpInfoBasic, sizeof(CacheOpInfoBasic)) == EOK,
             OP_LOGE(ACLNN_ERR_INNER, "call memcpy_s failed."),
             throw std::runtime_error("aclGraph profiling memcpy runtime error."));
    destOffset += sizeof(CacheOpInfoBasic);
    OP_LOGI("taskType %u, nodeId %zu, opType %zu, numBlocks %u, opFlag %u, tensorNum %u", cacheOpInfoBasic.taskType,
            cacheOpInfoBasic.nodeId, cacheOpInfoBasic.opType, cacheOpInfoBasic.numBlocks, cacheOpInfoBasic.opFlag,
            cacheOpInfoBasic.tensorNum);
    ReportCacheOpInfoTensor(dest, destOffset, totalSize, inTensors, MSPROF_GE_TENSOR_TYPE_INPUT);
    ReportCacheOpInfoTensor(dest, destOffset, totalSize, outTensors, MSPROF_GE_TENSOR_TYPE_OUTPUT);
    OP_CHECK_NO_RETURN(aclrtCacheLastTaskOpInfo(infoPtr, totalSize) == ACL_SUCCESS,
                       OP_LOGW("aclrtCacheLastTaskOpInfo failed"));
    op::internal::DeAllocate(infoPtr);
}

void PrepareAdditionInfo(MsprofAdditionalInfo& additionInfo)
{
    additionInfo.magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
    additionInfo.level = MSPROF_REPORT_NODE_LEVEL;
    additionInfo.type = MSPROF_REPORT_NODE_TENSOR_INFO_TYPE;
    additionInfo.threadId = OpGetTid();
    additionInfo.dataLen = sizeof(MsprofTensorInfo);
    additionInfo.timeStamp = op::internal::GetThreadLocalContext().kernelLauncherStartTime_;
}

void ChangeAclTenorAddrWithOffset(const aclTensor* tensor, std::vector<AddrRestorer>& record)
{
    void* oriAddr = tensor->GetStorageAddr();
    tensor->GetTensor()->MutableTensorData().SetAddr(tensor->GetData(), nullptr);
    record.push_back({tensor, oriAddr});
}

void RecoverAclTensorAddr(std::vector<AddrRestorer>& record)
{
    for (AddrRestorer& t : record) {
        std::get<0>(t)->GetTensor()->MutableTensorData().SetAddr(std::get<1>(t), nullptr);
    }
}

void PrepareTensorData(const FVector<const aclTensor*>& tensors, MsprofTensorInfo& tensorInfo, MsprofGeTensorType type,
                       uint32_t tensorNum, uint32_t baseIndex)
{
    for (size_t i = 0; i < tensorNum; i++) {
        MsrofTensorData& tensorData = tensorInfo.tensorData[i];
        tensorData.tensorType = type;
        tensorData.format = tensors[i + baseIndex]->GetStorageFormat();
        tensorData.dataType = tensors[i + baseIndex]->GetDataType();
        auto& shape = tensors[i + baseIndex]->GetStorageShape();
        size_t dim = shape.GetDimNum();
        for (size_t j = 0; j < dim && j < MSPROF_GE_TENSOR_DATA_SHAPE_LEN; j++) {
            tensorData.shape[j] = shape[j];
        }
        for (size_t j = dim; j < MSPROF_GE_TENSOR_DATA_SHAPE_LEN; j++) {
            tensorData.shape[j] = 0;
        }
    }
}

void PrepareTensorInfo(const FVector<const aclTensor*>& tensors, MsprofTensorInfo& tensorInfo, uint64_t summaryId,
                       MsprofGeTensorType type, uint32_t tensorNum, uint32_t baseIndex)
{
    tensorInfo.opName = summaryId;
    tensorInfo.tensorNum = tensorNum;
    PrepareTensorData(tensors, tensorInfo, type, tensorNum, baseIndex);
}

void ReportAdditionInfo(const TaskInfo& taskInfo, uint64_t id, uint64_t summaryId)
{
    if (MSPROF_ADDTIONAL_INFO_DATA_LENGTH != sizeof(MsprofTensorInfo)) {
        OP_LOGI("Mismatch length, MSPROF_ADDTIONAL_INFO_DATA_LENGTH = %u, sizeof(MsprofTensorInfo) = %zu",
                MSPROF_ADDTIONAL_INFO_DATA_LENGTH, sizeof(MsprofTensorInfo));
        return;
    }

    MsprofCompactInfo compactInfo;
    MsprofAdditionalInfo additionInfo;

    PrepareBasicInfo(compactInfo, taskInfo, id, summaryId);
    PrepareAdditionInfo(additionInfo);

    MsprofTensorInfo emptyTensor;
    emptyTensor.opName = summaryId;
    emptyTensor.tensorNum = 0;
    OP_CHECK(
        memcpy_s(additionInfo.data, MSPROF_ADDTIONAL_INFO_DATA_LENGTH, &emptyTensor, sizeof(MsprofTensorInfo)) == EOK,
        OP_LOGW("Failed to memcpy."), return);
    MsprofReportAdditionalInfo(true, (void*)(&additionInfo), sizeof(MsprofAdditionalInfo));
}

void ReportNodeContextIdInfo(uint64_t summaryId)
{
    MsprofAdditionalInfo additionInfo;
    additionInfo.magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
    additionInfo.level = MSPROF_REPORT_NODE_LEVEL;
    additionInfo.type = MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE;
    additionInfo.threadId = OpGetTid();
    additionInfo.dataLen = sizeof(MsprofContextIdInfo);
    additionInfo.timeStamp = op::internal::GetThreadLocalContext().kernelLauncherStartTime_;

    MsprofContextIdInfo contextIdInfo;
    contextIdInfo.opName = summaryId;
    contextIdInfo.ctxIdNum = 1;
    contextIdInfo.ctxIds[0] = 0;
    OP_CHECK(memcpy_s(additionInfo.data, MSPROF_ADDTIONAL_INFO_DATA_LENGTH, &contextIdInfo,
                      sizeof(MsprofContextIdInfo)) == EOK,
             OP_LOGW("Failed to memcpy context info."), return);
    MsprofReportAdditionalInfo(true, (void*)(&additionInfo), sizeof(MsprofAdditionalInfo));
}

static void ReportTensorAdditionInfo(const FVector<const aclTensor*>& tensors, MsprofAdditionalInfo& additionInfo,
                                     uint64_t summaryId, MsprofGeTensorType type)
{
    MsprofTensorInfo tensorInfo;
    uint32_t loop = tensors.size() / MSPROF_GE_TENSOR_DATA_NUM;
    uint32_t tail = tensors.size() % MSPROF_GE_TENSOR_DATA_NUM;
    for (uint32_t i = 0; i < loop; i++) {
        PrepareTensorInfo(tensors, tensorInfo, summaryId, type, MSPROF_GE_TENSOR_DATA_NUM,
                          i * MSPROF_GE_TENSOR_DATA_NUM);
        OP_CHECK(memcpy_s(additionInfo.data, MSPROF_ADDTIONAL_INFO_DATA_LENGTH, &tensorInfo,
                          sizeof(MsprofTensorInfo)) == EOK,
                 OP_LOGW("Failed to memcpy tensor additional info."), return);
        MsprofReportAdditionalInfo(true, (void*)(&additionInfo), sizeof(MsprofAdditionalInfo));
    }
    if (tail != 0) {
        PrepareTensorInfo(tensors, tensorInfo, summaryId, type, tail, loop * MSPROF_GE_TENSOR_DATA_NUM);
        OP_CHECK(memcpy_s(additionInfo.data, MSPROF_ADDTIONAL_INFO_DATA_LENGTH, &tensorInfo,
                          sizeof(MsprofTensorInfo)) == EOK,
                 OP_LOGW("Failed to memcpy tensor tail additional info."), return);
        MsprofReportAdditionalInfo(true, (void*)(&additionInfo), sizeof(MsprofAdditionalInfo));
    }
}

static void ReportAdditionInfoImpl(FVector<const aclTensor*>& inTensors, FVector<const aclTensor*>& outTensors,
                                   uint64_t summaryId)
{
    if (MSPROF_ADDTIONAL_INFO_DATA_LENGTH != sizeof(MsprofTensorInfo)) {
        OP_LOGI("Mismatch length, MSPROF_ADDTIONAL_INFO_DATA_LENGTH = %u, sizeof(MsprofTensorInfo) = %zu",
                MSPROF_ADDTIONAL_INFO_DATA_LENGTH, sizeof(MsprofTensorInfo));
        return;
    }
    MsprofAdditionalInfo additionInfo;
    PrepareAdditionInfo(additionInfo);
    ReportTensorAdditionInfo(inTensors, additionInfo, summaryId, MSPROF_GE_TENSOR_TYPE_INPUT);
    ReportTensorAdditionInfo(outTensors, additionInfo, summaryId, MSPROF_GE_TENSOR_TYPE_OUTPUT);
}

void ReportAdditionInfo(FVector<const aclTensor*>& inTensors, FVector<const aclTensor*>& outTensors,
                        MsprofGeTaskType taskType, uint64_t summaryId)
{
    OP_LOGD("Entering function ReportAdditionInfo");
    MsprofCompactInfo compactInfo;
    PrepareBasicInfo(compactInfo, taskType, op::internal::GetThreadLocalContext().profilingInfoId_.kernelLauncherId_,
                     summaryId);
    ReportAdditionInfoImpl(inTensors, outTensors, summaryId);
}

void ReportAdditionInfo(FVector<const aclTensor*>& inTensors, FVector<const aclTensor*>& outTensors,
                        const TaskInfo& taskInfo, uint64_t summaryId)
{
    OP_LOGD("Entering function ReportAdditionInfo input TaskInfo");
    MsprofCompactInfo compactInfo;
    PrepareBasicInfo(compactInfo, taskInfo, op::internal::GetThreadLocalContext().profilingInfoId_.kernelLauncherId_,
                     summaryId);
    ReportAdditionInfoImpl(inTensors, outTensors, summaryId);
}

void ReportAttrInfo(std::string& attrStr, uint64_t summaryId)
{
    uint64_t id = MsprofGetHashId(attrStr.c_str(), attrStr.size());
    OP_LOGI("GenAttrInfoId, attr str = %s, id = %lu, opName = %lu", attrStr.c_str(), id, summaryId);
    MsprofAttrInfo attrInfo;
    attrInfo.opName = summaryId;
    attrInfo.attrType = OP_ATTR;
    attrInfo.hashId = id;

    MsprofCompactInfo compactInfo;
    compactInfo.magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
    compactInfo.level = MSPROF_REPORT_NODE_LEVEL;
    compactInfo.type = MSPROF_REPORT_NODE_ATTR_INFO_TYPE;
    compactInfo.threadId = OpGetTid();
    compactInfo.dataLen = sizeof(MsprofAttrInfo);
    compactInfo.timeStamp = op::internal::GetThreadLocalContext().kernelLauncherStartTime_;
    OP_CHECK(memcpy_s(compactInfo.data.info, MSPROF_COMPACT_INFO_DATA_LENGTH, &attrInfo, sizeof(MsprofAttrInfo)) == EOK,
             OP_LOGW("Failed to memcpy attr info."), return);
    auto res = MsprofReportCompactInfo(true, &compactInfo, sizeof(compactInfo));
    OP_LOGI("ReportAttrInfo, res = %d, compactInfo.timeStamp = %lu", res, compactInfo.timeStamp);
}

std::string GetLogApiInfo()
{
    std::string res;
    if (op::internal::GetThreadLocalContext().logInfo_.l2ApiName) {
        res = std::string(op::internal::GetThreadLocalContext().logInfo_.l2ApiName) + "_" +
              std::to_string(op::internal::GetThreadLocalContext().logInfo_.l2SequenceCounter);
    }
    if (op::internal::GetThreadLocalContext().logInfo_.l0Name) {
        if (op::internal::GetThreadLocalContext().logInfo_.l2ApiName) {
            res += "_";
        }
        res += op::internal::GetThreadLocalContext().logInfo_.l0Name;
    }
    return res;
}

static void FillDumpTensorInfo(Adx::TensorInfoV2& info, const aclTensor* tensor)
{
    info.tensorSize = static_cast<size_t>(
        op::CalcShapeBytes(tensor->GetStorageShape().GetShapeSize(), tensor->GetDataType()));
    info.format = tensor->GetStorageFormat();
    info.dataType = tensor->GetDataType();
    info.tensorAddr = static_cast<int64_t*>(tensor->GetData());
    info.addrType = Adx::AddressType::TRADITIONAL;
    info.placement = tensor->GetPlacement();
    auto& storageShape = tensor->GetStorageShape();
    for (size_t ii = 0; ii < storageShape.GetDimNum(); ++ii) {
        info.shape.push_back(storageShape.GetDim(ii));
    }
    auto& originShape = tensor->GetOriginalShape();
    for (size_t jj = 0; jj < originShape.GetDimNum(); ++jj) {
        info.originShape.push_back(originShape.GetDim(jj));
    }
}

void PrepareDumpTensor(std::vector<Adx::TensorInfoV2>& dumpTensors, const std::vector<const aclTensor*>& aclTensors,
                       OpIOType ioType, std::vector<AddrRestorer>& record)
{
    for (size_t i = 0; i < aclTensors.size(); i++) {
        Adx::TensorInfoV2 info;
        ChangeAclTenorAddrWithOffset(aclTensors[i], record);
        (ioType == OpInputType) ? info.type = Adx::TensorType::INPUT : info.type = Adx::TensorType::OUTPUT;
        FillDumpTensorInfo(info, aclTensors[i]);
        dumpTensors.push_back(info);
    }
}

void PrepareDumpTensor(std::vector<Adx::TensorInfoV2>& dumpTensors, const std::vector<const aclTensor*>& aclTensors,
                       const std::vector<uint32_t>& tensorsIdxList, OpIOType ioType, std::vector<AddrRestorer>& record)
{
    Adx::TensorType tensortype[] = {Adx::TensorType::INPUT, Adx::TensorType::OUTPUT, Adx::TensorType::WORKSPACE};

    for (size_t i = 0; i < aclTensors.size(); i++) {
        Adx::TensorInfoV2 info;
        ChangeAclTenorAddrWithOffset(aclTensors[i], record);
        info.type = tensortype[ioType];
        FillDumpTensorInfo(info, aclTensors[i]);
        info.argsOffSet = tensorsIdxList[i];
        dumpTensors.push_back(info);
    }
}

void PrepareDumpTensor(std::vector<Adx::TensorInfoV2>& dumpTensors, const std::vector<op::Tensor*>& opTensors,
                       OpIOType ioType)
{
    for (size_t i = 0; i < opTensors.size(); i++) {
        Adx::TensorInfoV2 info;
        (ioType == OpInputType) ? info.type = Adx::TensorType::INPUT : info.type = Adx::TensorType::OUTPUT;
        info.tensorSize = static_cast<size_t>(
            op::CalcShapeBytes(opTensors[i]->GetStorageShape().GetShapeSize(), opTensors[i]->GetDataType()));
        info.format = opTensors[i]->GetStorageFormat();
        info.dataType = opTensors[i]->GetDataType();
        info.tensorAddr = static_cast<int64_t*>(opTensors[i]->GetAddr());
        info.addrType = Adx::AddressType::TRADITIONAL;
        info.placement = opTensors[i]->GetPlacement();
        auto& storageShape = opTensors[i]->GetStorageShape();
        for (size_t ii = 0; ii < storageShape.GetDimNum(); ++ii) {
            info.shape.push_back(storageShape.GetDim(ii));
        }
        auto& originShape = opTensors[i]->GetOriginShape();
        for (size_t jj = 0; jj < originShape.GetDimNum(); ++jj) {
            info.originShape.push_back(originShape.GetDim(jj));
        }
        dumpTensors.push_back(info);
    }
}

void PrepareL2DumpTensor(std::vector<Adx::TensorInfoV2>& dumpTensors, const std::vector<const aclTensor*>& aclTensors,
                         OpIOType ioType)
{
    for (size_t i = 0; i < aclTensors.size(); i++) {
        Adx::TensorInfoV2 info;
        (ioType == OpInputType) ? info.type = Adx::TensorType::INPUT : info.type = Adx::TensorType::OUTPUT;
        info.tensorSize = static_cast<size_t>(
            op::CalcShapeBytes(aclTensors[i]->GetStorageShape().GetShapeSize(), aclTensors[i]->GetDataType()));
        info.format = aclTensors[i]->GetStorageFormat();
        info.dataType = aclTensors[i]->GetDataType();
        info.tensorAddr = static_cast<int64_t*>(aclTensors[i]->GetData());
        info.addrType = Adx::AddressType::TRADITIONAL;
        info.placement = aclTensors[i]->GetPlacement();
        auto& storageShape = aclTensors[i]->GetStorageShape();
        for (size_t ii = 0; ii < storageShape.GetDimNum(); ++ii) {
            info.shape.push_back(storageShape.GetDim(ii));
        }
        auto& originShape = aclTensors[i]->GetOriginalShape();
        for (size_t jj = 0; jj < originShape.GetDimNum(); ++jj) {
            info.originShape.push_back(originShape.GetDim(jj));
        }
        dumpTensors.push_back(info);
    }
}

void DumpL2(const std::vector<const aclTensor*>& tensors, const OpLogInfo& opLogInfo, OpIOType ioType,
            aclrtStream stream)
{
    if (op::internal::GetThreadLocalContext().opConfigInfo_.isOpDumpEnable_) {
        std::vector<Adx::TensorInfoV2> dumpTensors;
        PrepareL2DumpTensor(dumpTensors, tensors, ioType);
        std::string l2Name = (opLogInfo.l2ApiName != nullptr) ? opLogInfo.l2ApiName : "L2DfxAbscent";
        l2Name += std::string("_") + std::to_string(opLogInfo.l2SequenceCounter) + std::string("_L2");
        std::string l2Type = (opLogInfo.l2ApiName != nullptr) ? opLogInfo.l2ApiName : "L2DfxAbscent";
        auto res = Adx::AdumpDumpTensorV2(l2Name, l2Type, dumpTensors, stream);
        OP_LOGI("AdumpDumpTensor res = %d\n", res);
    }
}

void SummaryAttrArg([[maybe_unused]] size_t idx, OpArg& value, std::vector<AttrInfo>& attrInfos, std::string& attrStr)
{
    if (idx < attrInfos.size()) {
        attrStr += attrInfos[idx].attrName + ":";
    }
    if (value.type == OpArgType::OPARG_DATATYPE) {
        attrStr += ToString(static_cast<op::DataType>(value->value)).GetString();
    } else if (value.type == OpArgType::OPARG_BOOL || value.type == OpArgType::OPARG_INT ||
               value.type == OpArgType::OPARG_UINT) {
        std::stringstream ss;
        ss << value->value;
        attrStr += ss.str();
    } else if (value.type == OpArgType::OPARG_FLOAT) {
        std::stringstream ss;
        ss << value->fvalue;
        attrStr += ss.str();
    } else if (value.type == OpArgType::OPARG_DOUBLE) {
        std::stringstream ss;
        ss << value->dvalue;
        attrStr += ss.str();
    } else if (value.type == OpArgType::OPARG_STRING) {
        if (value->pointer != nullptr) {
            attrStr += std::string(reinterpret_cast<char*>(value->pointer));
        }
    } else if (value.type == OpArgType::OPARG_ACLSCALAR) {
        if (value->pointer != nullptr) {
            attrStr += (reinterpret_cast<aclScalar*>(value->pointer))->ToString().GetString();
        }
    } else if (value.type == OpArgType::OPARG_INT_LIST) {
        if (value->pointer != nullptr) {
            attrStr += (reinterpret_cast<aclIntArray*>(value->pointer))->ToString().GetString();
        }
    } else if (value.type == OpArgType::OPARG_FLOAT_LIST) {
        if (value->pointer != nullptr) {
            attrStr += (reinterpret_cast<aclFloatArray*>(value->pointer))->ToString().GetString();
        }
    }
    attrStr += "|";
}

template <typename T>
void ValueToStr(const T* t, std::string& attrStr, int64_t size)
{
    if constexpr (std::is_same<op::fp16_t, typename std::decay<T>::type>::value) {
        for (int64_t i = 0; i < size; i++) {
            attrStr += std::to_string((static_cast<T>(t[i])).toFloat()) + ",";
        }
    } else if constexpr (std::is_same<op::bfloat16, typename std::decay<T>::type>::value) {
        for (int64_t i = 0; i < size; i++) {
            attrStr += std::to_string(float(t[i])) + ",";
        }
    } else {
        for (int64_t i = 0; i < size; i++) {
            attrStr += std::to_string(t[i]) + ",";
        }
    }
}

static void AddToAttrArg([[maybe_unused]] size_t idx, bool isTensor, std::string& attrStr, const aclTensor* tensor,
                         size_t listIdx = 0)
{
    if (tensor == nullptr) {
        return;
    }
    if (tensor->GetPlacement() != gert::TensorPlacement::kOnHost) {
        return;
    }
    OP_LOGI("input_%zu is host tensor", idx);
    if (isTensor) {
        attrStr += "input_" + std::to_string(idx) + ":";
    } else {
        attrStr += "input_" + std::to_string(idx) + "_" + std::to_string(listIdx) + ":";
    }

    switch (tensor->GetDataType()) {
        case op::DataType::DT_FLOAT: {
            float* tensorValue = PtrCastTo<float>(tensor->GetData());
            ValueToStr(tensorValue, attrStr, tensor->Size());
            break;
        }
        case op::DataType::DT_FLOAT16: {
            fp16_t* tensorValue = PtrCastTo<fp16_t>(tensor->GetData());
            ValueToStr(tensorValue, attrStr, tensor->Size());
            break;
        }
        case op::DataType::DT_DOUBLE: {
            double* tensorValue = PtrCastTo<double>(tensor->GetData());
            ValueToStr(tensorValue, attrStr, tensor->Size());
            break;
        }
        case op::DataType::DT_BF16: {
            bfloat16* tensorValue = PtrCastTo<bfloat16>(tensor->GetData());
            ValueToStr(tensorValue, attrStr, tensor->Size());
            break;
        }
        case op::DataType::DT_INT32: {
            int32_t* tensorValue = PtrCastTo<int32_t>(tensor->GetData());
            ValueToStr(tensorValue, attrStr, tensor->Size());
            break;
        }
        case op::DataType::DT_INT64: {
            int64_t* tensorValue = PtrCastTo<int64_t>(tensor->GetData());
            ValueToStr(tensorValue, attrStr, tensor->Size());
            break;
        }
        case op::DataType::DT_INT8: {
            int8_t* tensorValue = PtrCastTo<int8_t>(tensor->GetData());
            ValueToStr(tensorValue, attrStr, tensor->Size());
            break;
        }
        case op::DataType::DT_INT16: {
            int16_t* tensorValue = PtrCastTo<int16_t>(tensor->GetData());
            ValueToStr(tensorValue, attrStr, tensor->Size());
            break;
        }
        case op::DataType::DT_UINT32: {
            uint32_t* tensorValue = PtrCastTo<uint32_t>(tensor->GetData());
            ValueToStr(tensorValue, attrStr, tensor->Size());
            break;
        }
        case op::DataType::DT_UINT64: {
            uint64_t* tensorValue = PtrCastTo<uint64_t>(tensor->GetData());
            ValueToStr(tensorValue, attrStr, tensor->Size());
            break;
        }
        case op::DataType::DT_UINT8: {
            uint8_t* tensorValue = PtrCastTo<uint8_t>(tensor->GetData());
            ValueToStr(tensorValue, attrStr, tensor->Size());
            break;
        }
        case op::DataType::DT_UINT16: {
            uint16_t* tensorValue = PtrCastTo<uint16_t>(tensor->GetData());
            ValueToStr(tensorValue, attrStr, tensor->Size());
            break;
        }
        case op::DataType::DT_BOOL: {
            bool* tensorValue = PtrCastTo<bool>(tensor->GetData());
            ValueToStr(tensorValue, attrStr, tensor->Size());
            break;
        }
        default: {
            OP_LOGI("Tensor datatype is: %s", ToString(tensor->GetDataType()).GetString());
            break;
        }
    }
    attrStr += "|";
}

void SummaryAttrArg([[maybe_unused]] size_t idx, OpArg& value, std::string& attrStr)
{
    if (value.type == OpArgType::OPARG_ACLTENSOR) {
        aclTensor* tensor = PtrCastTo<aclTensor>(value->pointer);
        AddToAttrArg(idx, true, attrStr, tensor);
    }
    if (value.type == OpArgType::OPARG_ACLTENSOR_LIST) {
        aclTensorList* tensorList = PtrCastTo<aclTensorList>(value->pointer);
        if (tensorList == nullptr) {
            return;
        }
        for (uint64_t i = 0; i < tensorList->Size(); i++) {
            aclTensor* tensorCurrent = PtrCastTo<aclTensor>((*tensorList)[i]);
            AddToAttrArg(idx, false, attrStr, tensorCurrent, i);
        }
    }
}

void PrepareExceptionDumpInfo(const std::vector<op::Tensor*>& in, const std::vector<op::Tensor*>& out,
                              const OpLogInfo& opLogInfo, const ExceptionDumpInfo& exceptionDumpInfo,
                              const aclrtStream stream)
{
    int32_t deviceId = 0;
    uint32_t taskId = 0;
    int32_t streamId = 0;
    std::vector<Adx::TensorInfoV2> dumpInTensors;
    std::vector<Adx::TensorInfoV2> dumpOutTensors;

    auto ret = aclrtGetThreadLastTaskId(&taskId);
    OP_CHECK(ret == ACL_SUCCESS, OP_LOGW("Failed to get taskId."), return);
    ret = aclrtStreamGetId(stream, &streamId);
    OP_CHECK(ret == ACL_SUCCESS, OP_LOGW("Failed to get streamId."), return);
    ret = aclrtGetDevice(&deviceId);
    OP_CHECK(ret == ACL_ERROR_NONE, OP_LOGW("Failed to get deviceId."), return);

    PrepareDumpTensor(dumpInTensors, in, OpInputType);
    PrepareDumpTensor(dumpOutTensors, out, OpOutputType);

    std::string l2Name = (opLogInfo.l2ApiName != nullptr) ? opLogInfo.l2ApiName : "L2DfxAbscent";
    l2Name += std::string("_L0");
    std::string l0Name = (opLogInfo.l0Name != nullptr) ? opLogInfo.l0Name : "L0DfxAbscent";
    Adx::OperatorInfoV2 opInfo = OperatorInfoBuilder(l2Name, l0Name)
                                     .Task(deviceId, taskId, static_cast<uint32_t>(streamId))
                                     .TensorInfo(dumpInTensors)
                                     .TensorInfo(dumpOutTensors)
                                     .AdditionInfo(BLOCK_DIM_STR, exceptionDumpInfo.numBlocks_)
                                     .AdditionInfo(DEV_FUNC, exceptionDumpInfo.devFunc_)
                                     .AdditionInfo(TVM_MAGIC, exceptionDumpInfo.magic_)
                                     .AdditionInfo(TILING_KEY_STR, exceptionDumpInfo.tilingKey_)
                                     .AdditionInfo(TILING_DATA_STR, exceptionDumpInfo.tilingData_)
                                     .DeviceInfo(ARGS_BEFORE_EXEC,
                                                 op::internal::GetThreadLocalContext().exceptionDumpInfo_.rtsArgs_,
                                                 op::internal::GetThreadLocalContext().exceptionDumpInfo_.rtsArgsSize_)
                                     .Build();
    auto res = AdumpAddExceptionOperatorInfoV2(opInfo);
    OP_LOGI("AdumpAddExceptionOperatorInfo block_dim: %s, dev_func: %s, magic: %s, tiling key: %s, tiling data: %s\n",
            exceptionDumpInfo.numBlocks_.c_str(), exceptionDumpInfo.devFunc_.c_str(), exceptionDumpInfo.magic_.c_str(),
            exceptionDumpInfo.tilingKey_.c_str(), exceptionDumpInfo.tilingData_.c_str());
    OP_LOGI("AdumpAddExceptionOperatorInfo L2 = %s, L0 = %s, device_id = %d, task_id = %u, stream_id = %u, res = %d\n",
            l2Name.c_str(), l0Name.c_str(), deviceId, taskId, streamId, res);
}

void PrepareExceptionDumpInfo(OpArgContext* args, const OpLogInfo& opLogInfo, bool genPlaceholder, bool hasDevPtrArg_,
                              bool interCoreSync, std::vector<int32_t>& tensorOffset, const aclrtStream stream)
{
    std::vector<Adx::TensorInfoV2> dumpInTensors;
    std::vector<Adx::TensorInfoV2> dumpOutTensors;
    std::vector<Adx::TensorInfoV2> dumpWorkSpaceTensors;

    std::vector<const aclTensor*> aclInTensors;
    std::vector<const aclTensor*> aclOutTensors;
    std::vector<const aclTensor*> aclWorkSpaceTensors;
    std::vector<uint32_t> inTensorsIdxList;
    std::vector<uint32_t> outTensorsIdxList;
    std::vector<uint32_t> workSpaceTensorsIdxList;

    int32_t currentIdx = (interCoreSync) ? 0 : -1;

    TraitsAclTensorAndIdx(aclInTensors, inTensorsIdxList, *args->GetOpArg(op::OP_INPUT_ARG), genPlaceholder,
                          hasDevPtrArg_, currentIdx, tensorOffset);

    TraitsAclTensorAndIdx(aclOutTensors, outTensorsIdxList, *args->GetOpArg(op::OP_OUTPUT_ARG), genPlaceholder,
                          currentIdx);
    if (inTensorsIdxList.size() != aclInTensors.size() || outTensorsIdxList.size() != aclOutTensors.size()) {
        OP_LOGW("Calculate input/output failed.");
        return;
    }

    std::vector<AddrRestorer> record;
    PrepareDumpTensor(dumpInTensors, aclInTensors, inTensorsIdxList, OpInputType, record);
    PrepareDumpTensor(dumpOutTensors, aclOutTensors, outTensorsIdxList, OpOutputType, record);

    if (args->ContainsOpArgType(op::OP_WORKSPACE_ARG)) {
        TraitsAclTensorAndIdx(aclWorkSpaceTensors, workSpaceTensorsIdxList, *args->GetOpArg(op::OP_WORKSPACE_ARG),
                              genPlaceholder, currentIdx);

        if (workSpaceTensorsIdxList.size() != aclWorkSpaceTensors.size()) {
            OP_LOGW("Calculate workspace failed.");
            return;
        }

        PrepareDumpTensor(dumpWorkSpaceTensors, aclWorkSpaceTensors, workSpaceTensorsIdxList, OpWorkspaceType, record);
    }

    AddExceptionDumpOperatorInfo(dumpInTensors, dumpOutTensors, dumpWorkSpaceTensors, opLogInfo, stream);

    RecoverAclTensorAddr(record);
}

void AddExceptionDumpOperatorInfo(const std::vector<Adx::TensorInfoV2>& dumpInTensors,
                                  const std::vector<Adx::TensorInfoV2>& dumpOutTensors,
                                  const std::vector<Adx::TensorInfoV2>& dumpWorkSpaceTensors,
                                  const OpLogInfo& opLogInfo, const aclrtStream stream)
{
    int32_t deviceId = 0;
    uint32_t taskId = 0;
    int32_t streamId = 0;
    auto ret = aclrtGetThreadLastTaskId(&taskId);
    OP_CHECK(ret == ACL_SUCCESS, OP_LOGW("Failed to get taskId."), return);
    ret = aclrtStreamGetId(stream, &streamId);
    OP_CHECK(ret == ACL_SUCCESS && streamId >= 0, OP_LOGW("Failed to get streamId."), return);
    ret = aclrtGetDevice(&deviceId);
    OP_CHECK(ret == ACL_SUCCESS, OP_LOGW("Failed to get deviceId."), return);

    std::string l2Name = (opLogInfo.l2ApiName != nullptr) ? opLogInfo.l2ApiName : "L2DfxAbscent";
    l2Name += std::string("_") + std::to_string(opLogInfo.l2SequenceCounter) + std::string("_L0");
    std::string l0Name = (opLogInfo.l0Name != nullptr) ? opLogInfo.l0Name : "L0DfxAbscent";
    Adx::OperatorInfoV2
        opInfo = OperatorInfoBuilder(l2Name, l0Name)
                     .Task(deviceId, taskId, static_cast<uint32_t>(streamId))
                     .TensorInfo(dumpInTensors)
                     .TensorInfo(dumpOutTensors)
                     .TensorInfo(dumpWorkSpaceTensors)
                     .AdditionInfo(BLOCK_DIM_STR, op::internal::GetThreadLocalContext().exceptionDumpInfo_.numBlocks_)
                     .AdditionInfo(DEV_FUNC, op::internal::GetThreadLocalContext().exceptionDumpInfo_.devFunc_)
                     .AdditionInfo(TVM_MAGIC, op::internal::GetThreadLocalContext().exceptionDumpInfo_.magic_)
                     .AdditionInfo(TILING_KEY_STR, op::internal::GetThreadLocalContext().exceptionDumpInfo_.tilingKey_)
                     .AdditionInfo(TILING_DATA_STR,
                                   op::internal::GetThreadLocalContext().exceptionDumpInfo_.tilingData_)
                     .DeviceInfo(ARGS_BEFORE_EXEC, op::internal::GetThreadLocalContext().exceptionDumpInfo_.rtsArgs_,
                                 op::internal::GetThreadLocalContext().exceptionDumpInfo_.rtsArgsSize_)
                     .Build();
    auto res = AdumpAddExceptionOperatorInfoV2(opInfo);
    OP_LOGI("AdumpAddExceptionOperatorInfo block_dim: %s, dev_func: %s, magic: %s, tiling key: %s, tiling data: %s\n",
            op::internal::GetThreadLocalContext().exceptionDumpInfo_.numBlocks_.c_str(),
            op::internal::GetThreadLocalContext().exceptionDumpInfo_.devFunc_.c_str(),
            op::internal::GetThreadLocalContext().exceptionDumpInfo_.magic_.c_str(),
            op::internal::GetThreadLocalContext().exceptionDumpInfo_.tilingKey_.c_str(),
            op::internal::GetThreadLocalContext().exceptionDumpInfo_.tilingData_.c_str());
    OP_LOGI("AdumpAddExceptionOperatorInfo L2 = %s, L0 = %s, device_id = %d, task_id = %u, stream_id = %u, res = %d\n",
            l2Name.c_str(), l0Name.c_str(), deviceId, taskId, streamId, res);
}

void DumpL0(OpArgList& tensors, const OpLogInfo& opLogInfo, OpIOType ioType, aclrtStream stream)
{
    std::vector<Adx::TensorInfoV2> dumpTensors;
    std::vector<AddrRestorer> record;
    PrepareL0DumpTensor(dumpTensors, tensors, ioType, record);
    std::string l2Name = (opLogInfo.l2ApiName != nullptr) ? opLogInfo.l2ApiName : "L2DfxAbscent";
    l2Name += std::string("_") + std::to_string(opLogInfo.l2SequenceCounter) + std::string("_L0");
    std::string l0Name = (opLogInfo.l0Name != nullptr) ? opLogInfo.l0Name : "L0DfxAbscent";
    auto res = Adx::AdumpDumpTensorV2(l2Name, l0Name, dumpTensors, stream);
    RecoverAclTensorAddr(record);
    OP_LOGD("AdumpDumpTensor res = %d\n", res);
}

void DumpL0(OpArgList& inputTensors, OpArgList& outputTensors, const OpLogInfo& opLogInfo, aclrtStream stream)
{
    std::vector<Adx::TensorInfoV2> dumpTensors;
    std::vector<AddrRestorer> record;
    PrepareL0DumpTensor(dumpTensors, inputTensors, OpInputType, record);
    PrepareL0DumpTensor(dumpTensors, outputTensors, OpOutputType, record);
    std::string l2Name = (opLogInfo.l2ApiName != nullptr) ? opLogInfo.l2ApiName : "L2DfxAbscent";
    l2Name += std::string("_") + std::to_string(opLogInfo.l2SequenceCounter) + std::string("_L0");
    std::string l0Name = (opLogInfo.l0Name != nullptr) ? opLogInfo.l0Name : "L0DfxAbscent";
    auto res = Adx::AdumpDumpTensorV2(l2Name, l0Name, dumpTensors, stream);
    RecoverAclTensorAddr(record);
    OP_LOGD("AdumpDumpTensor res = %d\n", res);
}

static bool IsSaturationOverflow(aclrtStream stream)
{
    void* descBuf;
    uint64_t descBufLen = sizeof(uint64_t) * 8;

    NpuArch npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    aclrtMallocAttrValue moduleIdValue;
    moduleIdValue.moduleId = kModelId;
    aclrtMallocAttribute attrs{.attr = ACL_RT_MEM_ATTR_MODULE_ID, .value = moduleIdValue};
    aclrtMallocConfig cfg{.attrs = &attrs, .numAttrs = 1};
    auto ret = aclrtMallocWithCfg(&descBuf, (npuArch == NpuArch::DAV_2201) ? (descBufLen * 2) : descBufLen,
                                  ACL_MEM_TYPE_HIGH_BAND_WIDTH, &cfg);
    CHECK_COND(ret == ACL_SUCCESS, false, "failed to call aclrtMallocWithCfg, ret %d", ret);

    DevPtrGuard guard(descBuf);

    uint64_t realAddr = PtrToValue(descBuf);
#if !defined(NNOPBASE_UT) && !defined(NNOPBASE_ST)
    if (npuArch == NpuArch::DAV_2201) {
        realAddr = PtrToValue(PtrShift(descBuf, descBufLen));
        ret = aclrtMemcpy(descBuf, sizeof(uint64_t), &realAddr, sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_COND(ret == ACL_SUCCESS, false, "aclrtMemcpy failed, ret %d", ret);
    }
#endif
    uint32_t checkMode = 0;
    ret = aclrtNpuGetFloatOverFlowStatus(descBuf, descBufLen, checkMode, stream);
    CHECK_COND(ret == ACL_SUCCESS, false, "aclrtNpuGetFloatOverFlowStatus failed, ret %d", ret);

    auto rc = aclrtSynchronizeStream(stream);
    OP_CHECK(rc == ACL_SUCCESS, OP_LOGW("aclrtSynchronizeStream failed, ret %d", ret), return ACLNN_ERR_INNER);

    uint8_t status = 0;
    ret = aclrtMemcpy(&status, sizeof(uint8_t), ValueToPtr(realAddr), sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_COND(ret == ACL_SUCCESS, false, "aclrtMemcpy failed, ret %d", ret);

    ret = aclrtNpuClearFloatOverFlowStatus(checkMode, stream);
    CHECK_COND(ret == ACL_SUCCESS, false, "aclrtNpuClearFloatOverFlowStatus failed, ret %d", ret);

    OP_LOGD("get saturation overflow status %u", status);
    return status;
}

static bool IsInfNanOverflow(OpArgContext* args, aclOpExecutor* executor, aclrtStream stream)
{
    bool dump = false;
    NonFiniteCheckOpContext nonFiniteCheckOpCtx(executor, stream, args);
    auto ret = NonFiniteCheckOp::RunNonfiniteCheckOp(nonFiniteCheckOpCtx, dump);
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGW("RunNonfiniteCheckOp failed, ret %d", ret), return false);
    OP_LOGI("inf nan overflow status %d", dump);
    return dump;
}

aclnnStatus OverflowDumpProcess(OpArgContext* args, aclOpExecutor* executor, aclrtStream stream,
                                op::internal::OpLogInfo& logInfo)
{
    aclrtFloatOverflowMode floatOverflowMode = ACL_RT_OVERFLOW_MODE_SATURATION;
    aclError ret = aclrtGetDeviceSatMode(&floatOverflowMode);
    OP_CHECK((ret == ACL_SUCCESS && (floatOverflowMode == ACL_RT_OVERFLOW_MODE_SATURATION ||
                                     floatOverflowMode == ACL_RT_OVERFLOW_MODE_INFNAN)),
             OP_LOGW("check aclrtGetDeviceSatMode failed, ret: %d, overflow mode %d", ret, floatOverflowMode),
             return ACLNN_ERR_INNER);

    ret = aclrtSynchronizeStream(stream);
    OP_CHECK_NO_RETURN(ret == ACL_SUCCESS, OP_LOGW("aclrtSynchronizeStream failed, ret %d", ret));

    bool dump = false;
    if (floatOverflowMode == ACL_RT_OVERFLOW_MODE_SATURATION) {
        dump = IsSaturationOverflow(stream);
    } else if (floatOverflowMode == ACL_RT_OVERFLOW_MODE_INFNAN) {
        dump = IsInfNanOverflow(args, executor, stream);
    }
    if (dump) {
        op::internal::DumpL0(*args->GetOpArg(op::OP_INPUT_ARG), *args->GetOpArg(op::OP_OUTPUT_ARG), logInfo, stream);
    }
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

bool CanUsePTACache(const char* api)
{
    if (!CheckCacheable()) {
        return false;
    }
    op::internal::GetThreadLocalContext().logInfo_.l2ApiName = api;
    op::internal::GetThreadLocalContext().cacheApi_ = api;
    op::internal::GetThreadLocalContext().logInfo_.l2SequenceCounter = OpGetLogSequence();
    return true;
}

#ifdef __cplusplus
}
#endif

} // namespace internal

class OpDfxProfiler {
public:
    // L2_DFX
    explicit OpDfxProfiler(const char* funcName);
    // L0_DFX
    explicit OpDfxProfiler(uint32_t id);
    // infershape, tiling, kernel_launch
    OpDfxProfiler(uint64_t id, DfxProfilingType type);

    ~OpDfxProfiler();

private:
    void PrepareProfiling(uint32_t id);
    void PrepareProfilingKernelLaunch(uint64_t id);

private:
    bool profilingFlag_{false};
    MsprofApi info_;
};

OpDfxProfiler* CreateDfxProfiler(const char* funcName) { return new OpDfxProfiler(funcName); }
OpDfxProfiler* CreateDfxProfiler(uint32_t id) { return new OpDfxProfiler(id); }

bool IsDumpEnabled() { return op::internal::GetThreadLocalContext().opConfigInfo_.isOpDumpEnable_; }

void InitThreadLocalContext() { op::internal::GetThreadLocalContext().l2IOTensors_.Init(); }

void AddInputTensorToThreadLocalCtx(const aclTensor* const t)
{
    op::internal::AddToList(op::internal::GetThreadLocalContext().l2IOTensors_.inputTensors_, t);
}

void AddInputTensorToThreadLocalCtx(const aclTensorList* const t)
{
    op::internal::AddToList(op::internal::GetThreadLocalContext().l2IOTensors_.inputTensors_, t);
}

void AddInputTensorToThreadLocalCtx(aclTensor* const t)
{
    op::internal::AddToList(op::internal::GetThreadLocalContext().l2IOTensors_.inputTensors_, t);
}

void AddInputTensorToThreadLocalCtx(aclTensorList* const t)
{
    op::internal::AddToList(op::internal::GetThreadLocalContext().l2IOTensors_.inputTensors_, t);
}

void AddOutputTensorToThreadLocalCtx(const aclTensor* const t)
{
    op::internal::AddToList(op::internal::GetThreadLocalContext().l2IOTensors_.outputTensors_, t);
}

void AddOutputTensorToThreadLocalCtx(aclTensor* const t)
{
    op::internal::AddToList(op::internal::GetThreadLocalContext().l2IOTensors_.outputTensors_, t);
}

void AddOutputTensorToThreadLocalCtx(const aclTensorList* const t)
{
    op::internal::AddToList(op::internal::GetThreadLocalContext().l2IOTensors_.outputTensors_, t);
}

void AddOutputTensorToThreadLocalCtx(aclTensorList* const t)
{
    op::internal::AddToList(op::internal::GetThreadLocalContext().l2IOTensors_.outputTensors_, t);
}

op::OpDfxProfiler::OpDfxProfiler(const char* funcName)
{
    // L2_DFX
    OP_LOGI("OpDfxProfiler OpDfxProfiler start, %s", funcName);
    profilingFlag_ = true;
    auto id = op::internal::CollectProfilingStr(funcName);
    PrepareProfiling(id);
}

op::OpDfxProfiler::OpDfxProfiler(uint64_t id, DfxProfilingType type)
{
    OP_LOGI("OpDfxProfiler OpDfxProfiler start, %ld, %d", id, type);
    // infershape, tiling, kernel_launch
    profilingFlag_ = true;
    if (type == DfxProfilingKernelLaunch) {
        PrepareProfilingKernelLaunch(id);
        // profiling AdditionInfo report time should be in the range of kernel launcher time.
        op::internal::GetThreadLocalContext().kernelLauncherStartTime_ = info_.beginTime + 1;
    } else {
        PrepareProfiling(static_cast<uint32_t>(id));
    }
}
op::OpDfxProfiler::OpDfxProfiler(uint32_t id)
{
    // L0_DFX
    profilingFlag_ = true;
    PrepareProfiling(id);
}

op::OpDfxProfiler::~OpDfxProfiler()
{
    if (profilingFlag_) {
        info_.endTime = MsprofSysCycleTime();
        OP_LOGI("DfxGuard destructor, get msprof.timeStamp = %lu, level = %hu.", info_.endTime, info_.level);
        auto res = MsprofReportApi(1, &info_);
        if (res != 0) {
            OP_LOGW("Failed to call MsprofReportApi, res = %d", res);
        } else {
            OP_LOGD("~OpDfxGuard MsprofReportApi success");
        }
    }
}
// paramNames: "aa, bb, cc"
void op::OpDfxGuard::StringToVec(const char* paramNames, std::vector<std::string>& v)
{
    char* buf = new (std::nothrow) char[strlen(paramNames) + 1];
    CHECK_RET(buf, );
    auto rc = strncpy_s(buf, strlen(paramNames) + 1, paramNames, strlen(paramNames));
    OP_CHECK(rc == EOK, delete[] buf, return);
    char* token = nullptr;
    char* saveptr = nullptr;
    token = strtok_r(buf, ", ", &saveptr);
    while (token != nullptr) {
        v.push_back(token);
        token = strtok_r(NULL, ", ", &saveptr);
    }
    delete[] buf;
}

// paramNames: "DFX_IN(aa, bb, cc)" or "DFX_OUT(aa, bb)"
void op::OpDfxGuard::StringToVecWithBrackets(const char* paramNames, std::vector<std::string>& v)
{
    size_t begin = 0;
    size_t end = 0;
    for (size_t i = 0; i < strlen(paramNames); i++) {
        if (paramNames[i] == '(') {
            begin = i;
        }
        if (paramNames[i] == ')') {
            end = i;
        }
    }
    if (begin == 0 || end == 0 || end < begin) {
        return;
    }
    char* buf = new (std::nothrow) char[strlen(paramNames) + 1];
    CHECK_RET(buf, );
    auto rc = strncpy_s(buf, strlen(paramNames) + 1, paramNames + begin + 1, end - begin - 1);
    OP_CHECK(rc == EOK, delete[] buf, return);
    char* token = nullptr;
    char* saveptr = nullptr;
    token = strtok_r(buf, ", ", &saveptr);
    while (token != nullptr) {
        v.push_back(token);
        token = strtok_r(NULL, ", ", &saveptr);
    }
    delete[] buf;
}

void op::OpDfxProfiler::PrepareProfiling(uint32_t id)
{
    info_.beginTime = MsprofSysCycleTime();
    OP_LOGI("PrepareProfiling, get msprof.timeStamp = %lu", info_.beginTime);
    info_.threadId = internal::OpGetTid();
    info_.type = id;
    info_.magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
    info_.reserve = 0;
    info_.itemId = 0;
    info_.level = MSPROF_REPORT_ACL_LEVEL;
}

void op::OpDfxProfiler::PrepareProfilingKernelLaunch(uint64_t id)
{
    info_.beginTime = MsprofSysCycleTime();
    OP_LOGI("PrepareProfilingKernelLaunch, get msprof.timeStamp = %lu", info_.beginTime);
    info_.threadId = internal::OpGetTid();
    info_.type = MSPROF_REPORT_NODE_LAUNCH_TYPE;
    info_.magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
    info_.reserve = 0;
    info_.itemId = id;
    info_.level = MSPROF_REPORT_NODE_LEVEL;
}

// L2_DFX PHASE_TWO
OpDfxGuard::OpDfxGuard(const char* file, int line, OpLevel level, const char* funcName)
    : funcName_(funcName), file_(file), printLog_(true), line_(line), level_(level), profilingType_(DfxProfilingDefault)
{
    if (op::internal::opProfilingSwitch.reportFlag) {
        opDfxProfiler_ = new OpDfxProfiler(funcName);
    }
    OP_DFX_LOGI(file, line, funcName_, "Entering function %s", funcName);
}

// infershape, tiling, kernel_launch
OpDfxGuard::OpDfxGuard(uint64_t id, DfxProfilingType type)
    : funcName_(__func__), file_(__FILE__), printLog_(false), line_(__LINE__), level_(LevelTwo), profilingType_(type)
{
    if ((op::internal::opProfilingSwitch.reportFlag ||
         (type == DfxProfilingKernelLaunch && op::internal::opProfilingSwitch.kernelLaunchFlag)) &&
        (id != kInvalidProfilingId)) {
        opDfxProfiler_ = new OpDfxProfiler(id, type);
    }
}

OpDfxGuard::~OpDfxGuard()
{
    if (printLog_) {
        OP_DFX_LOGI(file_, line_, funcName_, "Leaving function %s", funcName_);
    }
    if (opDfxProfiler_) {
        delete opDfxProfiler_;
    }
    if (profilingType_ == DfxProfilingType::DfxProfilingDefault) {
        if (level_ == LevelZero) {
            op::internal::GetThreadLocalContext().logInfo_.InitLevelZero();
        } else if (level_ == LevelTwo) {
            op::internal::GetThreadLocalContext().logInfo_.InitLevelTwo();
        }
    }
}

aclnnStatus CheckPhase1Params(aclOpExecutor** executor, uint64_t* workspaceSize)
{
    if (executor == nullptr || workspaceSize == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "executor or workspaceSize is nullptr");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    return ACLNN_SUCCESS;
}

} // namespace op
