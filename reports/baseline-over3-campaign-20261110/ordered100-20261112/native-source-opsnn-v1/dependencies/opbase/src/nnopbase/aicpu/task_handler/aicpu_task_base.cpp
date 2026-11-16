/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <array>
#include <exception>
#include <memory>

#include "opdev/aicpu/aicpu_task.h"
#include "aicpu_engine_struct.h"
#include "aclnn/aclnn_base.h"
#include "opdev/fast_vector.h"
#include "mmpa/mmpa_api.h"
#include "op_dfx_internal.h"
#include "opdev/op_dfx.h"

namespace op {
namespace internal {
namespace {
constexpr size_t kMaxTotalHostLen = 1024U;
constexpr size_t kDefaultAicpuCacheLimit = 10000U;
constexpr size_t kMaxAicpuCacheLimit = 10000000U;

size_t ReadAicpuCacheLimit()
{
    const char* cacheLimit = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ACLNN_CACHE_LIMIT, cacheLimit);
    size_t c = kDefaultAicpuCacheLimit;
    if (cacheLimit != nullptr) {
        try {
            c = std::stoull(cacheLimit);
        } catch (const std::exception& e) {
            OP_LOGW("Env variable ACLNN_CACHE_LIMIT[%s] is invalid! must be a number!", cacheLimit);
        }
    }
    c = std::min(c, kMaxAicpuCacheLimit);
    return c;
}

const size_t kAicpuCacheLimit = ReadAicpuCacheLimit();

thread_local std::unique_ptr<AicpuTask> gPendingOverflowTask = nullptr;

AicpuTask* SavePendingOverflowTask(std::unique_ptr<AicpuTask> task)
{
    if (gPendingOverflowTask != nullptr) {
        OP_LOGW("Discard stale pending overflow aicpu task.");
    }
    gPendingOverflowTask = std::move(task);
    return gPendingOverflowTask.get();
}
} // namespace
AicpuTimeStamp gAicpuTimeStamp = {};
std::set<AicpuTaskSpace*> gAicpuTaskSpaceSet;
std::mutex gAicpuTaskSpaceMutex_;

std::unique_ptr<AicpuTask> TakePendingOverflowTask(AicpuTask* task)
{
    if (task == nullptr) {
        return nullptr;
    }
    if (gPendingOverflowTask == nullptr) {
        return nullptr;
    }
    if (gPendingOverflowTask.get() != task) {
        OP_LOGW("Take pending overflow aicpu task skipped, pending task does not match.");
        return nullptr;
    }
    return std::move(gPendingOverflowTask);
}

void aclnnAicpuFinalize()
{
    OP_LOGD("Entering func: aclnnAicpuFinalize, size=%zu.", gAicpuTaskSpaceSet.size());
    const std::lock_guard<std::mutex> lk(gAicpuTaskSpaceMutex_);
    /*
     * space中clear()勿加锁，否则可能导致finalize及task
     * init接口同时调用的错误场景下，space成员锁与gAicpuTaskSpaceMutex_互锁.
     */
    for (auto space : gAicpuTaskSpaceSet) {
        if (space != nullptr) {
            space->Clear();
        }
    }
    gAicpuTaskSpaceSet.clear();
    OP_LOGD("Leaving func: aclnnAicpuFinalize");
}

void SaveAicpuTaskSpace(void* aicpuTaskSpace)
{
    const std::lock_guard<std::mutex> lk(gAicpuTaskSpaceMutex_);
    const size_t oldSize = gAicpuTaskSpaceSet.size();
    const auto insertRet = gAicpuTaskSpaceSet.insert((AicpuTaskSpace*)aicpuTaskSpace);
    if (insertRet.second) {
        OP_LOGI("Save aicpu task space, old_size=%zu, new_size=%zu.", oldSize, gAicpuTaskSpaceSet.size());
    }
}

bool EnableAicpuTimeStamp()
{
    constexpr uint32_t kProfilingStdLengh = 128U;
    std::array<char, kProfilingStdLengh> profilingToStdOut = {};
    auto ret = mmGetEnv("GE_PROFILING_TO_STD_OUT", &profilingToStdOut[0U], kProfilingStdLengh);
    return ((ret == EN_OK) && (profilingToStdOut[0U] == '1'));
}

void AicpuPrintTimeInfo(const char* info, unsigned short startIndex, unsigned short endIndex)
{
    unsigned long time = 0U;
    if (startIndex < kAicpuTimeStampNum && endIndex < kAicpuTimeStampNum) {
        if (gAicpuTimeStamp.tp[startIndex].tv_sec != gAicpuTimeStamp.tp[endIndex].tv_sec) {
            // 1000000000 for time stamp
            time = gAicpuTimeStamp.tp[endIndex].tv_nsec + 1000000000 - gAicpuTimeStamp.tp[startIndex].tv_nsec;
        } else {
            time = gAicpuTimeStamp.tp[endIndex].tv_nsec - gAicpuTimeStamp.tp[startIndex].tv_nsec;
        }
        printf("%s : %f \n", info, time / 1000.0); // 1000.0 for time us
    }
}

void PrintAicpuAllTimeStampInfo(const char* opType)
{
    if (gAicpuTimeStamp.isEnable) {
        static std::atomic<std::uint64_t> index = 0;
        const unsigned long long opIdx = index++;
        printf("test case opType: %s, index: %llu\n", opType, opIdx);
        AicpuPrintTimeInfo("FindTask          ", kFindTaskStart, kFindTaskEnd);
        AicpuPrintTimeInfo("UpdateShape       ", kUpdateShapeStart, kUpdateShapeEnd);
        AicpuPrintTimeInfo("ShapeH2D          ", kUpdateShapeEnd, kShapeH2DEnd);
        AicpuPrintTimeInfo("UpdateArgs        ", kUpdateArgsStart, kUpdateArgsEnd);
        AicpuPrintTimeInfo("Launch            ", kUpdateArgsEnd, kLaunchEnd);
        AicpuPrintTimeInfo("CopyShapeD2H      ", kLaunchEnd, kShapeD2hCopyEnd);
        AicpuPrintTimeInfo("UpdateOutputShape ", kShapeD2hCopyEnd, kUpdateOutputShapeEnd);
        printf("\n");
    }
}

static void GenKeyByAttrsImpl(OpArg& attr, uint8_t*& key, size_t& keyLen)
{
    switch (attr.type) {
        case OpArgType::OPARG_INT_LIST:
            AppendAttrForKey(reinterpret_cast<aclIntArray*>(attr->pointer), key, keyLen);
            break;
        case OpArgType::OPARG_ACLTENSOR_LIST:
            AppendAttrForKey(reinterpret_cast<aclTensorList*>(attr->pointer), key, keyLen);
            break;
        case OpArgType::OPARG_FLOAT_LIST:
            AppendAttrForKey(reinterpret_cast<aclFloatArray*>(attr->pointer), key, keyLen);
            break;
        case OpArgType::OPARG_BOOL:
            AppendAttrForKey(static_cast<bool>(attr->value), key, keyLen);
            break;
        case OpArgType::OPARG_INT:
            AppendAttrForKey(static_cast<int64_t>(attr->ivalue), key, keyLen);
            break;
        case OpArgType::OPARG_UINT:
            AppendAttrForKey(static_cast<uint64_t>(attr->value), key, keyLen);
            break;
        case OpArgType::OPARG_FLOAT:
            AppendAttrForKey(static_cast<float>(attr->fvalue), key, keyLen);
            break;
        case OpArgType::OPARG_DOUBLE:
            AppendAttrForKey(static_cast<double>(attr->dvalue), key, keyLen);
            break;
        case OpArgType::OPARG_DATATYPE:
            AppendAttrForKey(static_cast<op::DataType>(attr->value), key, keyLen);
            break;
        case OpArgType::OPARG_STRING: {
            std::string keyStr(reinterpret_cast<char_t*>(attr->pointer));
            AppendAttrForKey(keyStr, key, keyLen);
            break;
        }
        default:
            OP_LOGE(ACLNN_ERR_INNER, "invalid attr arg type %d.", static_cast<int>(attr.type));
            return;
    }
}

static void CreateAicpuAttrsImpl(OpArg& attr, size_t idx, const FVector<std::string>& attrNames, AicpuAttrs& attrs)
{
    switch (attr.type) {
        case OpArgType::OPARG_INT_LIST:
            AddAicpuAttr(reinterpret_cast<aclIntArray*>(attr->pointer), attrNames[idx], attrs);
            break;
        case OpArgType::OPARG_ACLTENSOR_LIST:
            AddAicpuAttr(reinterpret_cast<aclTensorList*>(attr->pointer), attrNames[idx], attrs);
            break;
        case OpArgType::OPARG_FLOAT_LIST:
            AddAicpuAttr(reinterpret_cast<aclFloatArray*>(attr->pointer), attrNames[idx], attrs);
            break;
        case OpArgType::OPARG_BOOL:
            AddAicpuAttr(static_cast<bool>(attr->value), attrNames[idx], attrs);
            break;
        case OpArgType::OPARG_INT:
            AddAicpuAttr(static_cast<int64_t>(attr->ivalue), attrNames[idx], attrs);
            break;
        case OpArgType::OPARG_UINT:
            AddAicpuAttr(static_cast<int64_t>(attr->value), attrNames[idx], attrs);
            break;
        case OpArgType::OPARG_FLOAT:
            AddAicpuAttr(static_cast<float>(attr->fvalue), attrNames[idx], attrs);
            break;
        case OpArgType::OPARG_DOUBLE:
            AddAicpuAttr(static_cast<double>(attr->dvalue), attrNames[idx], attrs);
            break;
        case OpArgType::OPARG_DATATYPE:
            AddAicpuAttr(static_cast<op::DataType>(attr->value), attrNames[idx], attrs);
            break;
        case OpArgType::OPARG_STRING: {
            std::string keyStr(reinterpret_cast<char_t*>(attr->pointer));
            AddAicpuAttr(keyStr, attrNames[idx], attrs);
            break;
        }
        default:
            OP_LOGE(ACLNN_ERR_INNER, "invalid attr arg type %d.", static_cast<int>(attr.type));
            return;
    }
}

static aclnnStatus CreateAicpuAttrs(OpArgList& attrsArg, const FVector<std::string>& attrNames, AicpuAttrs& attrs)
{
    AICPU_ASSERT_TRUE_RETVAL(attrsArg.count == attrNames.size());
    attrsArg.VisitByNoReturn(
        [&]([[maybe_unused]] size_t idx, OpArg& arg) { CreateAicpuAttrsImpl(arg, idx, attrNames, attrs); });
    return OK;
}

static void GenKeyByAttrs(OpArgList& attr, uint8_t*& key, size_t& keyLen)
{
    attr.VisitByNoReturn([&]([[maybe_unused]] size_t idx, OpArg& arg) { GenKeyByAttrsImpl(arg, key, keyLen); });
}

size_t AicpuTaskSpace::GenHashBinary(const uint8_t* addr, uint32_t len)
{
    const std::hash<uint64_t> hasher;
    uint32_t size = len / sizeof(uint64_t);
    uint8_t rem = len % sizeof(uint64_t);
    uint32_t i = 0;
    size_t seed = 0;
    uint64_t* ptr = (uint64_t*)addr;

    while (i < size) {
        seed ^= hasher(*ptr) + kHashSeed + (seed << 6U) + (seed >> 2U);
        ptr++;
        i++;
    }
    if (rem != 0) {
        uint8_t* p = (uint8_t*)ptr;
        uint64_t val = 0U;
        while (rem > 0) {
            val |= (*p);
            val = val << 8U;
            p++;
            rem--;
        }
        seed ^= hasher(val) + kHashSeed + (seed << 6U) + (seed >> 2U);
    }
    return seed;
}

void AicpuTaskSpace::SetRef(const size_t index, const bool isInput)
{
    if (hasInit_) {
        return;
    }
    const std::lock_guard<std::mutex> lk(mutex_);
    if (isInput) {
        (void)inputRefIndexes_.insert(index);
    } else {
        (void)outputRefIndexes_.insert(index);
    }
}

bool AicpuTaskSpace::IsRef(const size_t index, const bool isInput) const
{
    if (isInput) {
        return inputRefIndexes_.find(index) != inputRefIndexes_.end();
    } else {
        return outputRefIndexes_.find(index) != outputRefIndexes_.end();
    }
}

void AicpuTask::SetVisit(bool visit) { isVisit_ = visit; }

uint64_t RoundUp(const uint64_t originValue, const uint64_t multipleOf)
{
    if (multipleOf == 0) {
        return 0;
    }
    return (originValue + multipleOf - 1) / multipleOf * multipleOf;
}

uint64_t AicpuTaskSpace::CalcHostInputDataSize(const FVector<const aclTensor*>& inputs, size_t alignBytes) const
{
    uint64_t totalSize = 0;
    for (size_t i = 0; i < inputs.size(); i++) {
        if ((inputs[i] == nullptr) || (inputs[i]->GetPlacement() != op::TensorPlacement::kOnHost)) {
            continue;
        }
        int64_t dataSize = op::CalcShapeBytes(inputs[i]->GetStorageShape().GetShapeSize(), inputs[i]->GetDataType());
        const size_t alignSize = RoundUp(static_cast<uint64_t>(dataSize), alignBytes);
        totalSize += alignSize;
    }
    return totalSize;
}

uint64_t AicpuTaskSpace::CalcDeviceCacheSize(const FVector<const aclTensor*>& inputs,
                                             std::unique_ptr<AicpuTask>& aicpuTask) const
{
    uint64_t cacheSize = 0;
    if (unknownType_ == ge::DEPEND_SHAPE_RANGE) {
        cacheSize += aicpuTask->deviceExtMemSize_;
    }

    uint64_t inputDeviceCache = CalcHostInputDataSize(inputs, aicpuTask->argsHandle_->GetInputAddrAlignBytes());
    if (inputDeviceCache > kMaxTotalHostLen) {
        cacheSize += inputDeviceCache;
    }
    return cacheSize;
}

size_t AicpuTaskSpace::GenTaskKey(uint8_t inputKey[], size_t& keyLen, op::OpArgContext* args,
                                  const FVector<const aclTensor*>& inputs) const
{
    keyLen = 0U;
    uint8_t* key = &inputKey[0];
    AppendAttrForKey(GetTid(), key, keyLen);
    for (size_t i = 0U; i < inputs.size(); ++i) {
        if (inputs[i] == nullptr) {
            continue;
        }
        key = Append1Byte(key, static_cast<uint8_t>(inputs[i]->GetDataType()));
        ++keyLen;
        key = Append1Byte(key, static_cast<uint8_t>(inputs[i]->GetViewFormat()));
        ++keyLen;
    }
    key = Append1Byte(key, static_cast<uint8_t>('/'));
    ++keyLen;
    GenKeyByAttrs(*args->GetOpArg(op::OP_ATTR_ARG), key, keyLen);
    auto seed = GenHashBinary(&inputKey[0], keyLen);
    return seed;
}

AicpuTask* AicpuTaskSpace::FindTask(aclOpExecutor* executor, op::OpArgContext* args,
                                    const FVector<const aclTensor*>& inputs)
{
    const std::lock_guard<std::mutex> lk(mutex_);
    RecordAicpuTime(kFindTaskStart);
    uint8_t inputKey[kAicpuKeyBufLen];
    size_t keyLen = 0;
    auto seed = GenTaskKey(inputKey, keyLen, args, inputs);
    const auto& iter = hashMap_.find(seed);
    if (iter != hashMap_.end()) {
        for (auto& task : iter->second) {
            if (task->isVisit_) {
                continue;
            }
            if (task->keyLen_ != keyLen) {
                continue;
            }
            size_t i = 0U;
            for (; i < keyLen; ++i) {
                if (task->inputKey_[i] != inputKey[i]) {
                    break;
                }
            }
            if (i == keyLen) {
                uint64_t deviceCacheSize = CalcDeviceCacheSize(inputs, task);
                if (deviceCacheSize > executor->workspaceDeviceAicpuMem_) {
                    executor->workspaceDeviceAicpuMem_ = deviceCacheSize;
                }
                RecordAicpuTime(kFindTaskEnd);
                task->isVisit_ = true;
                OP_LOGI("Find %s task success, no need create, cache_size=%lu.", opType_.c_str(),
                        executor->workspaceDeviceAicpuMem_);
                return task.get();
            }
        }
    }
    OP_LOGI("Do not find %s task, need to create task.", opType_.c_str());
    RecordAicpuTime(kFindTaskEnd);
    return nullptr;
}

AicpuTask* AicpuTaskSpace::GetOrCreateTask(aclOpExecutor* executor, const FVector<std::string>& attrNames,
                                           op::OpArgContext* args)
{
    gAicpuTimeStamp.isEnable = EnableAicpuTimeStamp();
    FVector<const aclTensor*> inputs;
    FVector<aclTensor*> outputs;
    CreateTensorList(executor, *args->GetOpArg(op::OP_INPUT_ARG), inputs);
    auto cacheTaskNum = [this]() {
        size_t taskNum = 0U;
        for (const auto& item : hashMap_) {
            taskNum += item.second.size();
        }
        return taskNum;
    };
    auto task = FindTask(executor, args, inputs);
    if (task != nullptr) {
        return task;
    }
    // 将task设置到全局的space中，需要加锁
    const std::lock_guard<std::mutex> lk(mutex_);
    OP_LOGI("Start to create %s task!", opType_.c_str());
    std::unique_ptr<AicpuTask> uniqueTask;
    if (isTf_) {
        // todo ge::MakeUnique
        uniqueTask = std::make_unique<AicpuTfTask>(opType_, unknownType_);
    } else {
        uniqueTask = std::make_unique<AicpuCCTask>(opType_, unknownType_);
    }
    if (uniqueTask == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER, "Make aicpu task failed!.");
        return nullptr;
    }
    uniqueTask->SetSpace(this);
    AicpuAttrs attrs;
    CreateAicpuAttrs(*args->GetOpArg(op::OP_ATTR_ARG), attrNames, attrs);
    CreateTensorList(executor, *args->GetOpArg(op::OP_OUTPUT_ARG), outputs);
    if (uniqueTask->Init(inputs, outputs, attrs) != OK) {
        OP_LOGE(ACLNN_ERR_INNER, "Aicpu task init failed!.");
        return nullptr;
    }
    SaveAicpuTaskSpace(uniqueTask->space_);
    uint8_t inputKey[kAicpuKeyBufLen];
    size_t keyLen = 0;
    auto seed = GenTaskKey(inputKey, keyLen, args, inputs);
    uniqueTask->keyLen_ = keyLen;
    for (size_t i = 0U; i < keyLen; ++i) {
        uniqueTask->inputKey_[i] = inputKey[i];
    }
    uniqueTask->isVisit_ = true;
    uint64_t deviceCacheSize = CalcDeviceCacheSize(inputs, uniqueTask);
    if (deviceCacheSize > executor->workspaceDeviceAicpuMem_) {
        executor->workspaceDeviceAicpuMem_ = deviceCacheSize;
    }
    hasInit_ = true;
    const size_t cachedTaskNum = cacheTaskNum();
    if (cachedTaskNum < kAicpuCacheLimit) {
        hashMap_[seed].emplace_back(std::move(uniqueTask));
        task = hashMap_[seed].back().get();
        OP_LOGI("Create %s task success, total=%zu, cache_size=%lu.", opType_.c_str(), cachedTaskNum + 1U,
                executor->workspaceDeviceAicpuMem_);
        return task;
    }

    task = SavePendingOverflowTask(std::move(uniqueTask));
    if (task == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER, "Create %s overflow task failed.", opType_.c_str());
        return nullptr;
    }
    OP_LOGI("Create %s task success without cache, cache full: total=%zu, max=%zu, cache_size=%lu.", opType_.c_str(),
            cachedTaskNum, kAicpuCacheLimit, executor->workspaceDeviceAicpuMem_);
    return task;
}

aclnnStatus AicpuTask::SetIoTensors(aclOpExecutor* executor, op::OpArgContext* args)
{
    RecordAicpuTime(kUpdateShapeStart);
    inputs_.clear();
    outputs_.clear();
    CreateTensorList(executor, *args->GetOpArg(op::OP_INPUT_ARG), inputs_);
    CreateTensorList(executor, *args->GetOpArg(op::OP_OUTPUT_ARG), outputs_);
    return extInfoHandle_->UpdateInputAndOutputShape(inputs_, outputs_, executor->GetStream(), executor,
                                                     deviceExtMemSize_, deviceCacheOffset_);
}
} // namespace internal
} // namespace op
