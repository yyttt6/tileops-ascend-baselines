/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/op_executor.h"

#include <sstream>
#include <chrono>

#include "kernel_graph.h"
#include "memory_allocator.h"
#include "opdev/op_config.h"
#include "opdev/op_dfx.h"
#include "kernel_launcher.h"
#include "bridge_graph.h"
#include "bridge_pool.h"
#include "dsa_task.h"
#include "opdev/tensor_view_utils.h"
#include "kernel_workspace.h"
#include "shape_inference.h"
#include "op_cache_internal.h"
#include "op_feature_internal.h"
#include "kernel_utils.h"
#include "thread_local_context.h"
#include "op_dfx_internal.h"
#include "dlopen_api.h"

using namespace op::internal;

enum class RepeatMode { Default, Unrepeatable, Repeat };

namespace op {
constexpr size_t K_PAIR_STORAGE_RELATION = 2;
class OpExecutorImpl : public op::Object {
public:
    OpExecutorImpl();
    ~OpExecutorImpl();
    void SetWorkspaceOffsets(const op::FVector<uint64_t>& workspaceOffsets);

    const op::FVector<uint64_t>& GetWorkspaceOffsets() const;

    void* GetGraph();
    void SetGraph(void* graph);

    op::internal::OpLogInfo GetLogInfo() const;

    void SetLogInfo(const op::internal::OpLogInfo& logInfo);

    op::OpConfigInfo GetOpConfigInfo() const;

    void SetOpConfigInfo(const op::OpConfigInfo& opConfigInfo);

    void SetIOTensorList(const op::internal::L2IOTensors& ioTensors);

    std::vector<const aclTensor*>& GetInputTensors();

    std::vector<const aclTensor*>& GetOutputTensors();

    void AddTensorRelation(const aclTensor* tensorOut, const aclTensor* tensorMiddle);

    void AddCache();

    void DeleteCache();

    void AbandonCache(bool disableRepeat = false);

    op::internal::OpExecCache* GetOpExecCache();

    void OpExecCacheSetWorkspaceSize(uint64_t workspaceSize);
    void OpExecCacheSetCacheBuf(void* buf);
    aclrtStream GetStream() const;
    void SetStream(aclrtStream stream);
    void* GetWorkspaceAddr() const;
    void SetWorkspaceAddr(void* workspaceAddr);
    void SetWorkspaceSize(const uint64_t workspaceSize);
    uint64_t GetWorkspaceSize() const;
    int32_t GethugeMemPoolIndex() const;

    void UpdateStorageAddr();
    aclnnStatus SetRepeatable(const op::FVector<op::KernelLauncher*>& launchers);
    void FinalizeCache();
    aclnnStatus RepeatRunWithCache(void* workspaceAddr, const aclrtStream stream);
    bool IsRepeatable() const;
    std::string ReportAddrForRepeat();

private:
    std::vector<const aclTensor*> inputTensors_;
    std::vector<const aclTensor*> outputTensors_;
    std::vector<const aclTensor*> workspaceTensors_;
    void* graph_{nullptr};
    aclrtStream stream_{nullptr};
    void* workspace_{nullptr};
    mutable uint64_t workspaceSize_{0};
    int32_t hugeMemPoolIndex_{op::kInvalidHugeMemIndexId};
    op::internal::OpLogInfo logInfo_;
    op::OpConfigInfo opConfigInfo_;
    op::FVector<uint64_t> workspaceOffsets_;
    op::internal::OpExecCache* opExecCache_{nullptr};
    op::FVector<const aclTensor*> tensorRelation_;
    std::vector<const aclStorage*> cachedStorageList_;
    std::vector<void*> cacheAddrLists_;
    RepeatMode repeatMode_{RepeatMode::Default};
    void* opExecCacheManager_{nullptr};
};

OpExecutorImpl::OpExecutorImpl()
{
    // Put RecordHugeMem at the begining, if huge mem pool not be acitved, return kInvalidHugeMemIndexId(-1)
    hugeMemPoolIndex_ = GetPoolIndex();
    OP_LOGI("Hugemem trace: get huge memory pool index: %d", hugeMemPoolIndex_);
    graph_ = CreateGraphImpl();
    if (!GetThreadLocalContext().cacheHasFull_) {
        opExecCache_ = new OpExecCache();
    }
}

OpExecutorImpl::~OpExecutorImpl()
{
    if (repeatMode_ == RepeatMode::Repeat && opExecCache_ != nullptr) {
        OP_LOGI("delete opExecCache in repeatable executor destructor");
        delete opExecCache_;
        opExecCache_ = nullptr;
    }
    if (opExecCache_ != nullptr) {
        opExecCache_->Finalize();
        opExecCache_->SetUse();
        ReleaseOpExecCacheManager(opExecCacheManager_);
    }
}

void OpExecutorImpl::FinalizeCache()
{
    if (opExecCache_ != nullptr) {
        OP_CHECK(!(opExecCache_->CanUse()), OP_LOGI("OpExecCache has been can used, cant finalize cache."), return);
        opExecCache_->Finalize();
        opExecCache_->SetUse();
    }
}

void OpExecutorImpl::SetWorkspaceOffsets(const op::FVector<uint64_t>& workspaceOffsets)
{
    workspaceOffsets_ = workspaceOffsets;
}

void* OpExecutorImpl::GetGraph() { return graph_; }

void OpExecutorImpl::SetGraph(void* graph) { graph_ = graph; }

aclnnStatus OpExecutorImpl::SetRepeatable(const op::FVector<op::KernelLauncher*>& launchers)
{
    OP_CHECK((repeatMode_ != RepeatMode::Unrepeatable),
             OP_LOGW("unrepeatable executor, find keyword MarkOpCacheInvalid in log."), return ACLNN_ERR_INNER;);
    OP_CHECK((hugeMemPoolIndex_ == op::kInvalidHugeMemIndexId),
             OP_LOGW("cant set executor repeatable when use huge page mem."), return ACLNN_ERR_INNER;);

    OP_CHECK((tensorRelation_.size() % K_PAIR_STORAGE_RELATION == 0),
             OP_LOGW("size if tensor relation must be pair of tensor"), return ACLNN_ERR_INNER;);
    // check tensor can repeat
    auto& opTlsCtx = op::internal::GetThreadLocalContext();
    cachedStorageList_.assign(opTlsCtx.cachedStorageList_.begin(),
                              opTlsCtx.cachedStorageList_.begin() + opTlsCtx.cachedStorageListSize_);
    // middle -> output
    std::unordered_map<const aclStorage*, const aclStorage*> relation;
    for (size_t i = 0; i < tensorRelation_.size(); i += K_PAIR_STORAGE_RELATION) {
        relation[tensorRelation_[i + 1]->GetStorage()] = tensorRelation_[i]->GetStorage();
    }
    for (op::KernelLauncher* launcher : launchers) {
        if (!launcher->CheckRepeatable(relation, cachedStorageList_)) {
            return ACLNN_ERR_INNER;
        }
    }

    repeatMode_ = RepeatMode::Repeat;
    if (opExecCache_ != nullptr) {
        ReleaseOpExecCacheManager(opExecCacheManager_);
        RemoveExecCache(opExecCache_);
    }
    OP_LOGI("OpExecutorImpl::SetRepeatable success.");
    return OK;
}

bool OpExecutorImpl::IsRepeatable() const { return repeatMode_ == RepeatMode::Repeat; }

void OpExecutorImpl::UpdateStorageAddr()
{
    if (repeatMode_ != RepeatMode::Repeat) {
        return;
    }
    OP_CHECK((tensorRelation_.size() % 2 == 0), OP_LOGE(ACLNN_ERR_INNER, "the size of tensorRelation must be even"),
             return);
    for (size_t i = 0; i < tensorRelation_.size(); i += K_PAIR_STORAGE_RELATION) {
        const aclTensor* output = tensorRelation_[i];
        const aclTensor* middle = tensorRelation_[i + 1];
        const_cast<aclTensor*>(middle)->SetStorageAddr(output->GetStorageAddr());
    }
    OP_LOGI("Repeat run executor, device ptr after update: %s", ReportAddrForRepeat().c_str());
}

aclnnStatus OpExecutorImpl::RepeatRunWithCache(void* workspaceAddr, const aclrtStream stream)
{
    if (repeatMode_ != RepeatMode::Repeat) {
        return ACLNN_ERR_INNER;
    }
    OP_CHECK((opExecCache_ != nullptr && opExecCache_->CanUse()), OP_LOGW("don't have cache or cache can't use"),
             return ACLNN_ERR_INNER;);
    OP_CHECK(CheckCacheable(), OP_LOGW("cache can't be used because enable some dfx options."),
             return ACLNN_ERR_INNER;);
    OP_LOGI("Repeat run executor, device ptr after update: %s", ReportAddrForRepeat().c_str());
    cacheAddrLists_.clear();
    for (const aclStorage* s : cachedStorageList_) {
        cacheAddrLists_.push_back(s->GetAddr());
    }
    return opExecCache_->Run(workspaceAddr, stream, cacheAddrLists_);
}

std::string OpExecutorImpl::ReportAddrForRepeat()
{
    std::ostringstream oss;
    for (auto storage : cachedStorageList_) {
        oss << storage->GetAddr();
        oss << ", ";
    }
    oss << "relation addr : ";
    for (auto tensor : tensorRelation_) {
        oss << tensor->GetStorageAddr();
        oss << ", ";
    }
    return oss.str();
}

op::internal::OpLogInfo OpExecutorImpl::GetLogInfo() const { return logInfo_; }

void OpExecutorImpl::SetLogInfo(const op::internal::OpLogInfo& logInfo) { logInfo_ = logInfo; }

op::OpConfigInfo OpExecutorImpl::GetOpConfigInfo() const { return opConfigInfo_; }

void OpExecutorImpl::SetOpConfigInfo(const op::OpConfigInfo& opConfigInfo) { opConfigInfo_ = opConfigInfo; }

void OpExecutorImpl::SetIOTensorList(const op::internal::L2IOTensors& ioTensors)
{
    inputTensors_ = ioTensors.inputTensors_;
    outputTensors_ = ioTensors.outputTensors_;
    workspaceTensors_ = ioTensors.workspaceTensors_;
}

std::vector<const aclTensor*>& OpExecutorImpl::GetInputTensors() { return inputTensors_; }

std::vector<const aclTensor*>& OpExecutorImpl::GetOutputTensors() { return outputTensors_; }

void OpExecutorImpl::AddCache()
{
    if (opExecCache_ == nullptr) {
        return;
    }
    if (!AddOpExecCache(opExecCache_)) {
        opExecCache_ = nullptr;
    } else {
        opExecCacheManager_ = GetOpExecCacheManager();
    }
}

void OpExecutorImpl::DeleteCache()
{
    if (opExecCache_ != nullptr) {
        delete opExecCache_;
        opExecCache_ = nullptr;
    }
}

void OpExecutorImpl::AbandonCache(bool disableRepeat)
{
    if (disableRepeat) {
        repeatMode_ = RepeatMode::Unrepeatable;
    }
    if (opExecCache_ != nullptr) {
        opExecCache_->MarkOpCacheInvalid();
    }
}

op::internal::OpExecCache* OpExecutorImpl::GetOpExecCache() { return opExecCache_; }

void OpExecutorImpl::AddTensorRelation(const aclTensor* tensorOut, const aclTensor* tensorMiddle)
{
    if (hugeMemPoolIndex_ == op::kInvalidHugeMemIndexId && repeatMode_ != RepeatMode::Unrepeatable) {
        tensorRelation_.emplace_back(tensorOut);
        tensorRelation_.emplace_back(tensorMiddle);
    }
    if (opExecCache_ == nullptr) {
        return;
    }
    opExecCache_->AddTensorRelation(tensorOut, tensorMiddle);
}

const op::FVector<uint64_t>& OpExecutorImpl::GetWorkspaceOffsets() const { return workspaceOffsets_; }

aclrtStream OpExecutorImpl::GetStream() const { return stream_; }

void OpExecutorImpl::SetStream(aclrtStream stream) { stream_ = stream; }
void* OpExecutorImpl::GetWorkspaceAddr() const { return workspace_; }
void OpExecutorImpl::SetWorkspaceAddr(void* workspaceAddr) { workspace_ = workspaceAddr; }
uint64_t OpExecutorImpl::GetWorkspaceSize() const { return workspaceSize_; }
void OpExecutorImpl::SetWorkspaceSize(const uint64_t workspaceSize) { workspaceSize_ = workspaceSize; }
int32_t OpExecutorImpl::GethugeMemPoolIndex() const { return hugeMemPoolIndex_; }

void OpExecutorImpl::OpExecCacheSetWorkspaceSize(uint64_t workspaceSize)
{
    opExecCache_->SetWorkspaceSize(workspaceSize);
}

void OpExecutorImpl::OpExecCacheSetCacheBuf(void* buf)
{
    OP_CHECK(!(opExecCache_->CanUse()), OP_LOGI("OpExecCache has been can used, cant change cache buf."), return);
    opExecCache_->SetCacheBuf(buf);
}
} // namespace op

void aclOpExecutor::AddTensorRelation(const aclTensor* tensorOut, const aclTensor* tensorMiddle)
{
    impl_->AddTensorRelation(tensorOut, tensorMiddle);
}

const op::FVector<uint64_t>& aclOpExecutor::GetWorkspaceOffsets() const { return impl_->GetWorkspaceOffsets(); }

void aclOpExecutor::SetWorkspaceOffsets(const op::FVector<uint64_t>& workspaceOffsets)
{
    return impl_->SetWorkspaceOffsets(workspaceOffsets);
}

void aclOpExecutor::SetLogInfo(const op::internal::OpLogInfo& logInfo) { impl_->SetLogInfo(logInfo); }

void aclOpExecutor::SetIOTensorList()
{
    auto ioTensors = GetThreadLocalContext().l2IOTensors_;
    impl_->SetIOTensorList(ioTensors);
}

aclOpExecutor::aclOpExecutor() { impl_ = new op::OpExecutorImpl(); }

aclOpExecutor::~aclOpExecutor()
{
    FreeGraphImpl(impl_->GetGraph());
    OP_LOGI("~aclOpExecutor, addr: %p", this);
    impl_->SetGraph(nullptr);
    for (size_t i = 0; i < kernelLaunchObjList_.size(); i++) {
        delete kernelLaunchObjList_[i];
    }

    for (size_t i = 0; i < allocatedObjList_.size(); i++) {
        delete allocatedObjList_[i];
    }

    GetThreadLocalContext().Init();

    // Update pool index to thread local for the destruction in pta, put this line at the end!
    UpdateHugeMemIndex(impl_->GethugeMemPoolIndex());
    // To ensure the correctness of the cache function, the SetUse interface must be invoked at
    // the last step of the destructor.
    delete impl_;
}

void aclOpExecutor::UpdateTensorAddr(void* workspaceAddr, const size_t size)
{
    impl_->SetWorkspaceAddr(workspaceAddr);
    for (auto tensor : allocatedTensorList_) {
        if (tensor->IsFromWorkspace()) {
            tensor->SetStorageAddr(workspaceAddr);
            tensor->GetTensor()->MutableTensorData().SetAddr(
                static_cast<char*>(workspaceAddr) + tensor->GetWorkspaceOffset(), nullptr);
            tensor->GetTensor()->MutableTensorData().SetSize(
                op::CalcShapeBytes(tensor->GetStorageShape().GetShapeSize(), tensor->GetDataType()));
        }
    }
    impl_->SetWorkspaceSize(size);
    if (impl_->GetOpExecCache() != nullptr) {
        impl_->OpExecCacheSetCacheBuf(GetCacheBuf());
    }
}

aclTensor* aclOpExecutor::AllocTensor(const op::Shape& shape, op::DataType dataType, op::Format format)
{
    aclTensor* tensor = nullptr;
    ADD_TRY_CATCH(tensor = new aclTensor(shape, dataType, format, nullptr); allocatedObjList_.push_back(tensor);
                  allocatedTensorList_.push_back(tensor); return tensor;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocTensor failed."); delete tensor; return nullptr;);
}

aclTensor* aclOpExecutor::AllocTensor(const op::Shape& storageShape, const op::Shape& originShape,
                                      op::DataType dataType, op::Format storageFormat, op::Format originFormat)
{
    aclTensor* tensor = nullptr;
    ADD_TRY_CATCH(tensor = new aclTensor(storageShape, originShape, dataType, storageFormat, originFormat, nullptr);
                  allocatedObjList_.push_back(tensor); allocatedTensorList_.push_back(tensor); return tensor;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocTensor failed."); delete tensor; return nullptr;);
}
aclTensor* aclOpExecutor::AllocTensor(op::DataType dataType, op::Format storageFormat, op::Format originFormat)
{
    aclTensor* tensor = nullptr;
    ADD_TRY_CATCH(tensor = new aclTensor(dataType, storageFormat, originFormat); allocatedObjList_.push_back(tensor);
                  allocatedTensorList_.push_back(tensor); return tensor;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocTensor failed."); delete tensor; return nullptr;);
}

aclTensor* aclOpExecutor::AllocHostTensor(const op::Shape& shape, op::DataType datatype, op::Format format)
{
    aclTensor* tensor = nullptr;
    ADD_TRY_CATCH(tensor = new aclTensor(shape, datatype, format); allocatedObjList_.push_back(tensor);
                  allocatedTensorList_.push_back(tensor); return tensor;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocHostTensor failed."); delete tensor; return nullptr;);
}

aclTensor* aclOpExecutor::AllocHostTensor(const op::Shape& storageShape, const op::Shape& originShape,
                                          op::DataType dataType, op::Format storageFormat, op::Format originFormat)
{
    aclTensor* tensor = nullptr;
    ADD_TRY_CATCH(tensor = new aclTensor(storageShape, originShape, dataType, storageFormat, originFormat);
                  allocatedObjList_.push_back(tensor); allocatedTensorList_.push_back(tensor); return tensor;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocHostTensor failed."); delete tensor; return nullptr;);
}

aclTensor* aclOpExecutor::AllocHostTensor(const int64_t* value, uint64_t size, op::DataType dataType)
{
    aclTensor* tensor = nullptr;
    ADD_TRY_CATCH(tensor = new aclTensor(value, size, dataType); allocatedObjList_.push_back(tensor);
                  allocatedTensorList_.push_back(tensor); return tensor;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocHostTensor failed."); delete tensor; return nullptr;);
}

aclTensor* aclOpExecutor::AllocHostTensor(const bool* value, uint64_t size, op::DataType dataType)
{
    aclTensor* tensor = nullptr;
    ADD_TRY_CATCH(tensor = new aclTensor(value, size, dataType); allocatedObjList_.push_back(tensor);
                  allocatedTensorList_.push_back(tensor); return tensor;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocHostTensor failed."); delete tensor; return nullptr;);
}
aclTensor* aclOpExecutor::AllocHostTensor(const uint64_t* value, uint64_t size, op::DataType dataType)
{
    aclTensor* tensor = nullptr;
    ADD_TRY_CATCH(tensor = new aclTensor(value, size, dataType); allocatedObjList_.push_back(tensor);
                  allocatedTensorList_.push_back(tensor); return tensor;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocHostTensor failed."); delete tensor; return nullptr;);
}
aclTensor* aclOpExecutor::AllocHostTensor(const char* value, uint64_t size, op::DataType dataType)
{
    aclTensor* tensor = nullptr;
    ADD_TRY_CATCH(tensor = new aclTensor(value, size, dataType); allocatedObjList_.push_back(tensor);
                  allocatedTensorList_.push_back(tensor); return tensor;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocHostTensor failed."); delete tensor; return nullptr;);
}
aclTensor* aclOpExecutor::AllocHostTensor(const int32_t* value, uint64_t size, op::DataType dataType)
{
    aclTensor* tensor = nullptr;
    ADD_TRY_CATCH(tensor = new aclTensor(value, size, dataType); allocatedObjList_.push_back(tensor);
                  allocatedTensorList_.push_back(tensor); return tensor;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocHostTensor failed."); delete tensor; return nullptr;);
}
aclTensor* aclOpExecutor::AllocHostTensor(const uint32_t* value, uint64_t size, op::DataType dataType)
{
    aclTensor* tensor = nullptr;
    ADD_TRY_CATCH(tensor = new aclTensor(value, size, dataType); allocatedObjList_.push_back(tensor);
                  allocatedTensorList_.push_back(tensor); return tensor;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocHostTensor failed."); delete tensor; return nullptr;);
}
aclTensor* aclOpExecutor::AllocHostTensor(const int16_t* value, uint64_t size, op::DataType dataType)
{
    aclTensor* tensor = nullptr;
    ADD_TRY_CATCH(tensor = new aclTensor(value, size, dataType); allocatedObjList_.push_back(tensor);
                  allocatedTensorList_.push_back(tensor); return tensor;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocHostTensor failed."); delete tensor; return nullptr;);
}
aclTensor* aclOpExecutor::AllocHostTensor(const uint16_t* value, uint64_t size, op::DataType dataType)
{
    aclTensor* tensor = nullptr;
    ADD_TRY_CATCH(tensor = new aclTensor(value, size, dataType); allocatedObjList_.push_back(tensor);
                  allocatedTensorList_.push_back(tensor); return tensor;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocHostTensor failed."); delete tensor; return nullptr;);
}
aclTensor* aclOpExecutor::AllocHostTensor(const int8_t* value, uint64_t size, op::DataType dataType)
{
    aclTensor* tensor = nullptr;
    ADD_TRY_CATCH(tensor = new aclTensor(value, size, dataType); allocatedObjList_.push_back(tensor);
                  allocatedTensorList_.push_back(tensor); return tensor;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocHostTensor failed."); delete tensor; return nullptr;);
}
aclTensor* aclOpExecutor::AllocHostTensor(const uint8_t* value, uint64_t size, op::DataType dataType)
{
    aclTensor* tensor = nullptr;
    ADD_TRY_CATCH(tensor = new aclTensor(value, size, dataType); allocatedObjList_.push_back(tensor);
                  allocatedTensorList_.push_back(tensor); return tensor;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocHostTensor failed."); delete tensor; return nullptr;);
}
aclTensor* aclOpExecutor::AllocHostTensor(const double* value, uint64_t size, op::DataType dataType)
{
    aclTensor* tensor = nullptr;
    ADD_TRY_CATCH(tensor = new aclTensor(value, size, dataType); allocatedObjList_.push_back(tensor);
                  allocatedTensorList_.push_back(tensor); return tensor;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocHostTensor failed."); delete tensor; return nullptr;);
}

aclTensor* aclOpExecutor::AllocHostTensor(const float* value, uint64_t size, op::DataType dataType)
{
    aclTensor* tensor = nullptr;
    ADD_TRY_CATCH(tensor = new aclTensor(value, size, dataType); allocatedObjList_.push_back(tensor);
                  allocatedTensorList_.push_back(tensor); return tensor;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocHostTensor failed."); delete tensor; return nullptr;);
}

aclTensor* aclOpExecutor::AllocHostTensor(const op::fp16_t* value, uint64_t size, op::DataType dataType)
{
    aclTensor* tensor = nullptr;
    ADD_TRY_CATCH(tensor = new aclTensor(value, size, dataType); allocatedObjList_.push_back(tensor);
                  allocatedTensorList_.push_back(tensor); return tensor;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocHostTensor failed."); delete tensor; return nullptr;);
}

aclTensor* aclOpExecutor::AllocHostTensor(const op::bfloat16* value, uint64_t size, op::DataType dataType)
{
    aclTensor* tensor = nullptr;
    ADD_TRY_CATCH(tensor = new aclTensor(value, size, dataType); allocatedObjList_.push_back(tensor);
                  allocatedTensorList_.push_back(tensor); return tensor;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocHostTensor failed."); delete tensor; return nullptr;);
}

const aclTensor* aclOpExecutor::ConvertToTensor(const aclIntArray* value, op::DataType dataType)
{
    aclTensor* tensor = nullptr;
    ADD_TRY_CATCH(tensor = new aclTensor(value, dataType); allocatedObjList_.push_back(tensor);
                  allocatedTensorList_.push_back(tensor); return tensor;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::ConvertToTensor failed."); delete tensor; return nullptr;);
}

const aclTensor* aclOpExecutor::ConvertToTensor(const aclBoolArray* value, op::DataType dataType)
{
    aclTensor* tensor = nullptr;
    ADD_TRY_CATCH(tensor = new aclTensor(value, dataType); allocatedObjList_.push_back(tensor);
                  allocatedTensorList_.push_back(tensor); return tensor;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::ConvertToTensor failed."); delete tensor; return nullptr;);
}

const aclTensor* aclOpExecutor::ConvertToTensor(const aclFloatArray* value, op::DataType dataType)
{
    aclTensor* tensor = nullptr;
    ADD_TRY_CATCH(tensor = new aclTensor(value, dataType); allocatedObjList_.push_back(tensor);
                  allocatedTensorList_.push_back(tensor); return tensor;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::ConvertToTensor failed."); delete tensor; return nullptr;);
}

const aclTensor* aclOpExecutor::ConvertToTensor(const aclFp16Array* value, op::DataType dataType)
{
    aclTensor* tensor = nullptr;
    ADD_TRY_CATCH(tensor = new aclTensor(value, dataType); allocatedObjList_.push_back(tensor);
                  allocatedTensorList_.push_back(tensor); return tensor;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::ConvertToTensor failed."); delete tensor; return nullptr;);
}

const aclTensor* aclOpExecutor::ConvertToTensor(const aclBf16Array* value, op::DataType dataType)
{
    aclTensor* tensor = nullptr;
    ADD_TRY_CATCH(tensor = new aclTensor(value, dataType); allocatedObjList_.push_back(tensor);
                  allocatedTensorList_.push_back(tensor); return tensor;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::ConvertToTensor failed."); delete tensor; return nullptr;);
}

const aclTensor* aclOpExecutor::ConvertToTensor(const aclScalar* value, op::DataType dataType)
{
    aclTensor* tensor = nullptr;
    ADD_TRY_CATCH(tensor = new aclTensor(value, dataType); allocatedObjList_.push_back(tensor);
                  allocatedTensorList_.push_back(tensor); return tensor;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::ConvertToTensor failed."); delete tensor; return nullptr;);
}

aclnnStatus aclOpExecutor::Run()
{
    aclnnStatus status = ACLNN_SUCCESS;
    auto graph = (op::mem::KernelGraph*)(impl_->GetGraph());
    CHECK_RET(graph != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    auto& sortedNodes = graph->GetSortedNodes();
    auto nodeCount = kernelLaunchObjList_.size();
    for (size_t i = 0; i < nodeCount; i++) {
        auto node = sortedNodes[i];
        OP_CHECK_NOTNULL(node);
        OP_CHECK(node->GetOriginalId() < static_cast<int64_t>(nodeCount),
                 OP_LOGE(ACLNN_ERR_INNER,
                         "check node's original id failed, it should be less than %zu, but actually is %zu.", nodeCount,
                         node->GetOriginalId()),
                 return ACLNN_ERR_INNER);
        auto& launcher = kernelLaunchObjList_[node->GetOriginalId()];
        launcher->UpdateThreadLocal();
        OP_LOGI("%zu start to Launch %s, original id:%ld.", i,
                op::OpTypeDict::ToString(launcher->GetOpType()).GetString(), node->GetOriginalId());
        status = launcher->Launch();
        if (status != ACLNN_SUCCESS) {
            OP_LOGE(status, "launch failed for %s, errno:%d.",
                    op::OpTypeDict::ToString(launcher->GetOpType()).GetString(), status);
            break;
        }
    }

    return status;
}

void* aclOpExecutor::GetWorkspaceAddr() const { return impl_->GetWorkspaceAddr(); }

uint64_t aclOpExecutor::GetWorkspaceSize() const
{
    if (impl_->GetWorkspaceSize() != 0) {
        return impl_->GetWorkspaceSize();
    }
    auto graph = (op::mem::KernelGraph*)(impl_->GetGraph());
    auto& tensors = graph->GetSortedKernelTensors();

    OP_LOGD("workspace tensor count:%zu.", tensors.size());
    auto allocator = op::mem::MaxAllocator();
    workspaceDeviceAicpuTaskOffset_ = allocator.Allocate(tensors);
    uint64_t workspaceSize = workspaceDeviceAicpuTaskOffset_ + workspaceDeviceAicpuMem_;
    if (impl_->GetOpExecCache() != nullptr) {
        impl_->OpExecCacheSetWorkspaceSize(workspaceSize);
    }
    OP_LOGD("workspaceSize_:%lu, workspaceDeviceAicpuTaskOffset_:%lu", workspaceSize, workspaceDeviceAicpuTaskOffset_);
    return workspaceSize;
}

uint64_t aclOpExecutor::GetLinearWorkspaceSize() const
{
    if (impl_->GetWorkspaceSize() != 0) {
        return impl_->GetWorkspaceSize();
    }
    auto graph = (op::mem::KernelGraph*)(impl_->GetGraph());
    auto& tensors = graph->GetSortedKernelTensors();

    OP_LOGD("workspace count:%zu.", tensors.size());
    auto allocator = op::mem::LinearAllocator();
    workspaceDeviceAicpuTaskOffset_ = allocator.Allocate(tensors);
    uint64_t workspaceSize = workspaceDeviceAicpuTaskOffset_ + workspaceDeviceAicpuMem_;
    if (impl_->GetOpExecCache() != nullptr) {
        impl_->OpExecCacheSetWorkspaceSize(workspaceSize);
    }
    OP_LOGD("workspaceSize_:%ld, workspaceDeviceAicpuTaskOffset_:%ld", workspaceSize, workspaceDeviceAicpuTaskOffset_);
    return workspaceSize;
}

aclrtStream aclOpExecutor::GetStream() const { return impl_->GetStream(); }

void aclOpExecutor::SetStream(aclrtStream stream) { impl_->SetStream(stream); }

aclIntArray* aclOpExecutor::AllocIntArray(const int64_t* value, uint64_t size)
{
    aclIntArray* array = nullptr;
    ADD_TRY_CATCH(array = new aclIntArray(value, size); allocatedObjList_.push_back(array); return array;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocIntArray failed."); delete array; return nullptr;);
}

aclFloatArray* aclOpExecutor::AllocFloatArray(const float* value, uint64_t size)
{
    aclFloatArray* array = nullptr;
    ADD_TRY_CATCH(array = new aclFloatArray(value, size); allocatedObjList_.push_back(array); return array;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocFloatArray failed."); delete array; return nullptr;);
}

aclBoolArray* aclOpExecutor::AllocBoolArray(const bool* value, uint64_t size)
{
    aclBoolArray* array = nullptr;
    ADD_TRY_CATCH(array = new aclBoolArray(value, size); allocatedObjList_.push_back(array); return array;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocBoolArray failed."); delete array; return nullptr;);
}

aclTensorList* aclOpExecutor::AllocTensorList(const aclTensor* const* tensors, uint64_t size)
{
    aclTensorList* list = nullptr;
    ADD_TRY_CATCH(list = new aclTensorList(tensors, size); allocatedObjList_.push_back(list); return list;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocTensorList failed."); delete list; return nullptr;);
}

aclScalarList* aclOpExecutor::AllocScalarList(const aclScalar* const* scalars, uint64_t size)
{
    aclScalarList* list = nullptr;
    ADD_TRY_CATCH(list = new aclScalarList(scalars, size); allocatedObjList_.push_back(list); return list;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocScalarList failed."); delete list; return nullptr;);
}

aclTensor* aclOpExecutor::CreateView(const aclTensor* tensor, const op::Shape& shape, int64_t offset)
{
    aclTensor* viewTensor = nullptr;
    ADD_TRY_CATCH(viewTensor = new aclTensor(*tensor, shape, offset); allocatedObjList_.push_back(viewTensor);
                  allocatedTensorList_.push_back(viewTensor); return viewTensor;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::CreateView failed."); delete viewTensor; return nullptr;);
}

aclTensor* aclOpExecutor::CreateView(const aclTensor* tensor, const op::Shape& oriShape, const op::Shape& storageShape,
                                     const op::Strides& oriStride, int64_t offset)
{
    aclTensor* viewTensor = nullptr;

    op::Shape storageShapeCorrect{1};
    int64_t storageSize = 1;
    for (uint64_t i = 0; i < storageShape.GetDimNum(); i++) {
        storageSize *= storageShape.GetDim(i);
    }
    int64_t actualShapeLen = storageSize - offset;
    OP_CHECK((actualShapeLen >= 0 && actualShapeLen <= storageSize),
             OP_LOGE(ACLNN_ERR_INNER, "The tensor's offset is invalid with the specified storageShape"),
             return nullptr;);
    // 在非连续场景下，传入tiling的storageshape会被oom用来计算合法内存大小，aclnn在下发时会将gm地址加上offset偏移，此时算子合法内存大小应为原始内存大小减offset
    storageShapeCorrect.SetDim(0, actualShapeLen);
    OP_LOGI("storageSize: %lu, actualShapeLen: %lu", storageSize, actualShapeLen);

    ADD_TRY_CATCH(viewTensor = new aclTensor(*tensor, oriShape, storageShapeCorrect, oriStride, offset);
                  allocatedObjList_.push_back(viewTensor); allocatedTensorList_.push_back(viewTensor);
                  return viewTensor;, OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::CreateView failed."); delete viewTensor;
                  return nullptr;);
}

aclScalar* aclOpExecutor::AllocScalar(float value)
{
    aclScalar* scalar = nullptr;
    ADD_TRY_CATCH(scalar = new aclScalar(value); allocatedObjList_.push_back(scalar); return scalar;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocScalar failed."); delete scalar; return nullptr;);
}

aclScalar* aclOpExecutor::AllocScalar(double value)
{
    aclScalar* scalar = nullptr;
    ADD_TRY_CATCH(scalar = new aclScalar(value); allocatedObjList_.push_back(scalar); return scalar;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocScalar failed."); delete scalar; return nullptr;);
}

aclScalar* aclOpExecutor::AllocScalar(op::fp16_t value)
{
    aclScalar* scalar = nullptr;
    ADD_TRY_CATCH(scalar = new aclScalar(value); allocatedObjList_.push_back(scalar); return scalar;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocScalar failed."); delete scalar; return nullptr;);
}

aclScalar* aclOpExecutor::AllocScalar(op::bfloat16 value)
{
    aclScalar* scalar = nullptr;
    ADD_TRY_CATCH(scalar = new aclScalar(value); allocatedObjList_.push_back(scalar); return scalar;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocScalar failed."); delete scalar; return nullptr;);
}

aclScalar* aclOpExecutor::AllocScalar(int32_t value)
{
    aclScalar* scalar = nullptr;
    ADD_TRY_CATCH(scalar = new aclScalar(value); allocatedObjList_.push_back(scalar); return scalar;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocScalar failed."); delete scalar; return nullptr;);
}

aclScalar* aclOpExecutor::AllocScalar(int64_t value)
{
    aclScalar* scalar = nullptr;
    ADD_TRY_CATCH(scalar = new aclScalar(value); allocatedObjList_.push_back(scalar); return scalar;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocScalar failed."); delete scalar; return nullptr;);
}

aclScalar* aclOpExecutor::AllocScalar(int16_t value)
{
    aclScalar* scalar = nullptr;
    ADD_TRY_CATCH(scalar = new aclScalar(value); allocatedObjList_.push_back(scalar); return scalar;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocScalar failed."); delete scalar; return nullptr;);
}

aclScalar* aclOpExecutor::AllocScalar(int8_t value)
{
    aclScalar* scalar = nullptr;
    ADD_TRY_CATCH(scalar = new aclScalar(value); allocatedObjList_.push_back(scalar); return scalar;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocScalar failed."); delete scalar; return nullptr;);
}
aclScalar* aclOpExecutor::AllocScalar(uint32_t value)
{
    aclScalar* scalar = nullptr;
    ADD_TRY_CATCH(scalar = new aclScalar(value); allocatedObjList_.push_back(scalar); return scalar;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocScalar failed."); delete scalar; return nullptr;);
}

aclScalar* aclOpExecutor::AllocScalar(uint64_t value)
{
    aclScalar* scalar = nullptr;
    ADD_TRY_CATCH(scalar = new aclScalar(value); allocatedObjList_.push_back(scalar); return scalar;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocScalar failed."); delete scalar; return nullptr;);
}

aclScalar* aclOpExecutor::AllocScalar(uint16_t value)
{
    aclScalar* scalar = nullptr;
    ADD_TRY_CATCH(scalar = new aclScalar(value); allocatedObjList_.push_back(scalar); return scalar;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocScalar failed."); delete scalar; return nullptr;);
}

aclScalar* aclOpExecutor::AllocScalar(uint8_t value)
{
    aclScalar* scalar = nullptr;
    ADD_TRY_CATCH(scalar = new aclScalar(value); allocatedObjList_.push_back(scalar); return scalar;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocScalar failed."); delete scalar; return nullptr;);
}

aclScalar* aclOpExecutor::AllocScalar(bool value)
{
    aclScalar* scalar = nullptr;
    ADD_TRY_CATCH(scalar = new aclScalar(value); allocatedObjList_.push_back(scalar); return scalar;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocScalar failed."); delete scalar; return nullptr;);
}

aclScalar* aclOpExecutor::AllocScalar(const void* data, op::DataType dataType)
{
    aclScalar* scalar = nullptr;
    ADD_TRY_CATCH(scalar = new aclScalar(data, dataType); allocatedObjList_.push_back(scalar); return scalar;
                  , OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::AllocScalar failed."); delete scalar; return nullptr;);
}

op::internal::OpLogInfo aclOpExecutor::GetLogInfo() const { return impl_->GetLogInfo(); }

op::OpConfigInfo aclOpExecutor::GetOpConfigInfo() const { return impl_->GetOpConfigInfo(); }

void aclOpExecutor::SetOpConfigInfo(const op::OpConfigInfo& opConfigInfo) { impl_->SetOpConfigInfo(opConfigInfo); }

std::vector<const aclTensor*>& aclOpExecutor::GetInputTensors() { return impl_->GetInputTensors(); }

std::vector<const aclTensor*>& aclOpExecutor::GetOutputTensors() { return impl_->GetOutputTensors(); }

op::internal::OpExecCache* aclOpExecutor::GetOpExecCache() { return impl_->GetOpExecCache(); }

void aclOpExecutor::AbandonCache(bool disableRepeat) { impl_->AbandonCache(disableRepeat); }

void aclOpExecutor::UpdateStorageAddr() { impl_->UpdateStorageAddr(); }

aclnnStatus aclOpExecutor::SetRepeatable() { return impl_->SetRepeatable(kernelLaunchObjList_); }

void aclOpExecutor::FinalizeCache() { impl_->FinalizeCache(); }

bool aclOpExecutor::IsRepeatable() const { return impl_->IsRepeatable(); }

aclnnStatus aclOpExecutor::RepeatRunWithCache(void* workspaceAddr, const aclrtStream stream)
{
    return impl_->RepeatRunWithCache(workspaceAddr, stream);
};

void aclOpExecutor::AddCache() { impl_->AddCache(); }

void aclOpExecutor::DeleteCache() { impl_->DeleteCache(); }

void* aclOpExecutor::GetGraph() { return impl_->GetGraph(); }

void aclOpExecutor::AddToKernelLauncherList(op::KernelLauncher* obj) { kernelLaunchObjList_.emplace_back(obj); }

bool aclOpExecutor::CheckLauncherRepeatable()
{
    std::unordered_map<const aclStorage*, const aclStorage*> relation;
    std::vector<const aclStorage*> oriStorage;
    for (op::KernelLauncher* launcher : kernelLaunchObjList_) {
        if (!launcher->CheckRepeatable(relation, oriStorage)) {
            return false;
        }
    }
    return true;
}

aclnnStatus aclOpExecutor::AddToKernelLauncherListDvpp(uint32_t opType, op::KernelLauncher* obj, op::OpArgContext* args)
{
    // dvpp
    kernelLaunchObjList_.emplace_back(obj);
    auto ret = op::internal::BuildGraph(impl_->GetGraph(), opType, *args->GetOpArg(op::OP_INPUT_ARG),
                                        *args->GetOpArg(op::OP_OUTPUT_ARG), *args->GetOpArg(op::OP_WORKSPACE_ARG));
    return ret;
}

aclnnStatus aclOpExecutor::AddToKernelLauncherListCopyTask(uint32_t opType, op::KernelLauncher* obj,
                                                           op::OpArgList& inputs, op::OpArgList& outputs,
                                                           op::OpArgList& workspace)
{
    // z_framework_op
    impl_->AbandonCache();
    kernelLaunchObjList_.emplace_back(obj);
    auto ret = op::internal::BuildGraph(impl_->GetGraph(), opType, inputs, outputs, workspace);
    return ret;
}

aclnnStatus aclOpExecutor::AddToKernelLauncherListAiCpu(int32_t opType, op::KernelLauncher* obj, op::OpArgContext* args)
{
    impl_->AbandonCache(true);
    kernelLaunchObjList_.emplace_back(obj);
    auto ret = op::internal::BuildGraph(impl_->GetGraph(), opType, *args->GetOpArg(op::OP_INPUT_ARG),
                                        *args->GetOpArg(op::OP_OUTPUT_ARG), *args->GetOpArg(op::OP_WORKSPACE_ARG));
    return ret;
}
uint64_t aclOpExecutor::GetMagicNumber() { return magicNumber_; }

UniqueExecutor::UniqueExecutor(const char* funcName) : funcName_(funcName), uniqueExecutor_(new aclOpExecutor())
{
    OP_LOGI("Create executor: %p", uniqueExecutor_.get());
    OP_CHECK(uniqueExecutor_ != nullptr, OP_LOGE(ACLNN_ERR_INNER, "executor constructed failed."),
             throw std::bad_alloc());
    auto& threadLocalCtx = op::internal::GetThreadLocalContext();
    uniqueExecutor_->SetLogInfo(threadLocalCtx.logInfo_);
    uniqueExecutor_->SetOpConfigInfo(threadLocalCtx.opConfigInfo_);
    if (threadLocalCtx.opConfigInfo_.isOpDumpEnable_) {
        uniqueExecutor_->SetIOTensorList();
    }
}

#ifdef __cplusplus
extern "C" {
#endif

aclOpExecutor* PTAGetExecCache(uint64_t hash, uint64_t* workspaceSize)
{
    auto cache = GetOpExecCache(hash);
    if (cache == nullptr) {
        OP_LOGW("cache is nullptr.");
        return nullptr;
    }
    *workspaceSize = cache->GetWorkspaceSize();
    OpExecCacheWrap* cacheWrap = CreateCacheWrap(cache);
    return PtrCastTo<aclOpExecutor>(cacheWrap);
}

aclOpExecutor* PTAFindExecCache(uint8_t* buf, size_t len, uint64_t* workspaceSize)
{
    OpCacheKey key(buf, len);
    auto cache = GetOpExecCache(key);
    if (cache == nullptr) {
        OP_LOGW("cache is nullptr.");
        return nullptr;
    }
    *workspaceSize = cache->GetWorkspaceSize();
    OpExecCacheWrap* cacheWrap = CreateCacheWrap(cache);
    return PtrCastTo<aclOpExecutor>(cacheWrap);
}

#ifdef __cplusplus
}
#endif

op::internal::OpExecCacheWrap* GetOpExecCacheFromExecutor(aclOpExecutor* executor)
{
    op::internal::OpExecCacheWrap* cache = PtrCastTo<op::internal::OpExecCacheWrap>(executor);
    if (cache->magicNumber_ == K_CACHE_WRAP_MAGIC_NUMBER) {
        return cache;
    }
    return nullptr;
}

static void DeleteExecutorForError(aclOpExecutor* executor)
{
    OpExecCacheWrap* cacheWrap = GetOpExecCacheFromExecutor(executor);
    if (cacheWrap != nullptr) {
        delete cacheWrap;
        return;
    }
    OpExecCache* cache = executor->GetOpExecCache();
    OP_CHECK_NO_RETURN(!(cache != nullptr), cache->MarkOpCacheInvalid());
    delete executor;
}

aclnnStatus CommonOpExecutorRun(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    static thread_local OpCacheGuard cacheGuard;
    if (unlikely(executor == nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "executor is nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (unlikely(workspaceSize > 0 && workspace == nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "workspace is nullptr.");
        DeleteExecutorForError(executor);
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    OP_LOGI("Workspace addr: %p, size: %lu, executor: %p", workspace, workspaceSize, executor);
    GetCacheOpInfoSwitch(stream);
    OpExecCacheWrap* cache = GetOpExecCacheFromExecutor(executor);
    if (cache != nullptr) {
        auto res = cache->Run(workspace, stream);
        delete cache;
        return res;
    }
    if (executor->RepeatRunWithCache(workspace, stream) == ACLNN_SUCCESS) {
        return ACLNN_SUCCESS;
    }
    executor->UpdateStorageAddr();
    OpExecCache* cachePtr = executor->GetOpExecCache();
    GetOpCacheContext().SetOpCache(cachePtr);
    executor->SetStream(stream);
    executor->UpdateTensorAddr(workspace, workspaceSize);
    op::internal::GetThreadLocalContext().executor_ = executor;
    DumpL2(executor->GetInputTensors(), executor->GetLogInfo(), op::OpInputType, stream);
    auto res = executor->Run();
    DumpL2(executor->GetOutputTensors(), executor->GetLogInfo(), op::OpOutputType, stream);
    OP_CHECK_NO_RETURN(!(res != ACLNN_SUCCESS && cachePtr != nullptr), cachePtr->MarkOpCacheInvalid());
    if (executor->IsRepeatable()) {
        executor->FinalizeCache();
    } else {
        delete executor;
    }
    return res;
}

namespace {
// 从 ACL runtime 获取确定性计算 level
// ACL_OPT_DETERMINISTIC 取值含义：
//   0 = 关闭
//   1 = 开启确定性计算
//   2 = 开启强一致性（前向包含 1）
//   3 = 开启 batch 一致性（前向包含 2）
// 兼容老 runtime：当返回值为 1 时（老 runtime 仅返回 0/1），二次获取
// ACL_OPT_STRONG_CONSISTENCY 确认是否实际应为 level 2
int64_t GetDeterministicLevelFromRt()
{
    int64_t deterministicLevel = 0;
    aclError aclRet = aclrtGetSysParamOpt(ACL_OPT_DETERMINISTIC, &deterministicLevel);
    OP_CHECK_NO_RETURN(aclRet == ACL_SUCCESS, deterministicLevel = 0;
                       OP_LOGW("can not get system param deterministic, ret= %d.", aclRet));

    if (deterministicLevel == 1) {
        int64_t consistency = 0;
        aclError consistencyRet = aclrtGetSysParamOpt(ACL_OPT_STRONG_CONSISTENCY, &consistency);
        OP_CHECK_NO_RETURN(consistencyRet == ACL_SUCCESS, consistency = 0;
                           OP_LOGW("can not get system param strong consistency, ret= %d.", consistencyRet));
        if (consistency == 1) {
            deterministicLevel = 2;
            OP_LOGI("Upgrade deterministic level from 1 to 2, because strong consistency is on.");
        }
    }
    return deterministicLevel;
}
} // namespace

void InitL2Phase1Context(const char* l2Name, [[maybe_unused]] aclOpExecutor** executor)
{
    InitAclnnDebugSwitch();
    OP_CHECK_NO_RETURN(op::internal::InitPcieThroughInfo() == ACLNN_SUCCESS, OP_LOGW("InitPcieThroughInfo failed"));
    auto& opTlsCtx = op::internal::GetThreadLocalContext();
    opTlsCtx.logInfo_.l2ApiName = l2Name;
    opTlsCtx.logInfo_.l2SequenceCounter = op::internal::OpGetLogSequence();
    uint32_t controlCoreNum = 0;
    OP_CHECK_NO_RETURN(aclrtGetResInCurrentThread(ACL_RT_DEV_RES_CUBE_CORE, &controlCoreNum) == ACL_RT_SUCCESS,
                       controlCoreNum = op::GetCurrentPlatformInfo().GetCubeCoreNum());
    opTlsCtx.opConfigInfo_.aicNum_ = controlCoreNum;
    OP_CHECK_NO_RETURN(aclrtGetResInCurrentThread(ACL_RT_DEV_RES_VECTOR_CORE, &controlCoreNum) == ACL_RT_SUCCESS,
                       controlCoreNum = op::GetCurrentPlatformInfo().GetVectorCoreNum());
    opTlsCtx.opConfigInfo_.aivNum_ = controlCoreNum;
    opTlsCtx.opConfigInfo_.isOpDumpEnable_ = op::internal::IsDumpEnable();

    int64_t deterministicLevel = GetDeterministicLevelFromRt();
    opTlsCtx.opConfigInfo_.deterministicLevel_ = static_cast<uint8_t>(deterministicLevel);
    opTlsCtx.opConfigInfo_.isDeterministicOn_ = (deterministicLevel >= 1);
    OP_LOGI("aic num: %u, aiv num: %u, deterministic level: %d, is deterministic on: %d, is op dump enable: %d",
            opTlsCtx.opConfigInfo_.aicNum_, opTlsCtx.opConfigInfo_.aivNum_, opTlsCtx.opConfigInfo_.deterministicLevel_,
            opTlsCtx.opConfigInfo_.isDeterministicOn_, opTlsCtx.opConfigInfo_.isOpDumpEnable_);
}

void InitL2Phase2Context([[maybe_unused]] const char* l2Name, aclOpExecutor* executor)
{
    OP_CHECK(executor != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "executor is nullptr."), return);
    auto cacheWrap = GetOpExecCacheFromExecutor(executor);
    auto& opTlsCtx = op::internal::GetThreadLocalContext();
    if (cacheWrap != nullptr) {
        opTlsCtx.logInfo_ = cacheWrap->opLogInfo_;
    } else {
        if (executor->GetMagicNumber() == K_EXECUTOR_MAGIC_NUMBER) {
            opTlsCtx.logInfo_ = executor->GetLogInfo();
            opTlsCtx.opConfigInfo_ = executor->GetOpConfigInfo();
        }
    }
    opTlsCtx.logInfo_.l0Name = nullptr;
}

void InitL0Context(const char* profilingName, aclOpExecutor* executor)
{
    OP_CHECK_NO_RETURN(executor != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "executor cann't be nullptr."));
    if (executor->GetMagicNumber() == K_EXECUTOR_MAGIC_NUMBER) {
        op::internal::OpLogInfo tmpLogInfo = executor->GetLogInfo();
        tmpLogInfo.l0Name = profilingName;
        executor->SetLogInfo(tmpLogInfo);
        op::internal::GetThreadLocalContext().logInfo_ = executor->GetLogInfo();
    }
}

aclnnStatus InferShape(uint32_t optype, op::OpArgList& inputs, op::OpArgList& outputs, op::OpArgList& attrs)
{
    return op::internal::InferShape(optype, inputs, outputs, attrs);
}

aclnnStatus CreatAiCoreKernelLauncher([[maybe_unused]] const char* l0Name, uint32_t opType, aclOpExecutor* executor,
                                      op::OpArgContext* args)
{
    CHECK_RET(args != nullptr && executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    op::internal::ProfilingInfoId profilingInfoId;
    op::internal::SetOpImplModeCtx(*args->GetOpArg(op::OP_OPTION_ARG));

    if (args->ContainsOpArgType(op::OP_OUTSHAPE_ARG)) {
        executor->AbandonCache(true);
    }
    op::internal::GetLauncherCtx().SetOpKernelBin(nullptr);
    aclnnStatus addToLaunchRet = ACLNN_SUCCESS;
    if (args->ContainsOpArgType(op::OP_WORKSPACE_ARG)) {
        auto* launcher = new op::AiCoreKernelLauncher{opType, op::AI_CORE, profilingInfoId, executor, args};
        executor->AddToKernelLauncherList(launcher);
        op::internal::BuildGraph(executor->GetGraph(), opType, *args->GetOpArg(op::OP_INPUT_ARG),
                                 *args->GetOpArg(op::OP_OUTPUT_ARG), *args->GetOpArg(op::OP_WORKSPACE_ARG));
    } else {
        aclTensorList* workspace = nullptr;
        op::internal::GetLauncherCtx().ClearTilingCache();
        addToLaunchRet = op::internal::GetWorkspace(opType, &workspace, executor, *args->GetOpArg(op::OP_INPUT_ARG),
                                                    *args->GetOpArg(op::OP_OUTPUT_ARG),
                                                    *args->GetOpArg(op::OP_ATTR_ARG));
        args->AppendOpWorkspaceArg(workspace);
        auto* launcher = new op::AiCoreKernelLauncher{opType, op::AI_CORE, profilingInfoId, executor, args};
        launcher->SaveLaunchCtx(std::move(op::internal::GetLauncherCtx()));

        executor->AddToKernelLauncherList(launcher);
        op::internal::BuildGraph(executor->GetGraph(), opType, *args->GetOpArg(op::OP_INPUT_ARG),
                                 *args->GetOpArg(op::OP_OUTPUT_ARG), *args->GetOpArg(op::OP_WORKSPACE_ARG),
                                 *args->GetOpArg(op::OP_OUTSHAPE_ARG));
    }
    return addToLaunchRet;
}
void CreatDSAKernelLauncher([[maybe_unused]] const char* l0Name, uint32_t opType, DSA_TASK_TYPE dsaTask,
                            aclOpExecutor* executor, op::OpArgContext* args)
{
    if (args == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Failed to create dsa task");
        return;
    }
    // DSA
    op::internal::ProfilingInfoId profilingInfoId;
    aclTensorList* workspace = nullptr;
    op::internal::GetDSATaskWorkspace(executor, &workspace);
    args->AppendOpWorkspaceArg(workspace);

    op::KernelLauncher* launcher = nullptr;
    switch (dsaTask) {
        case DSARandomNormalTaskType: {
            op::internal::DSARandomNormalTask task1{};
            launcher = new op::internal::DSAKernelLauncher{opType, op::AI_CORE, profilingInfoId, executor, task1, args};
            break;
        }
        case DSARandomTruncatedNormalTaskType: {
            op::internal::DSARandomTruncatedNormalTask task2{};
            launcher = new op::internal::DSAKernelLauncher{opType, op::AI_CORE, profilingInfoId, executor, task2, args};
            break;
        }
        case DSARandomUniformTaskType: {
            op::internal::DSARandomUniformTask task3{};
            launcher = new op::internal::DSAKernelLauncher{opType, op::AI_CORE, profilingInfoId, executor, task3, args};
            break;
        }
        case DSAGenBitMaskTaskType: {
            op::internal::DSAGenBitMaskTask task4{};
            launcher = new op::internal::DSAKernelLauncher{opType, op::AI_CORE, profilingInfoId, executor, task4, args};
            break;
        }
        default: {
            OP_LOGW("Failed to create dsa task");
            break;
        }
    }
    executor->AddToKernelLauncherList(launcher);
    executor->AbandonCache(true);

    op::internal::BuildGraph(executor->GetGraph(), opType, *args->GetOpArg(op::OP_INPUT_ARG),
                             *args->GetOpArg(op::OP_OUTPUT_ARG), *args->GetOpArg(op::OP_WORKSPACE_ARG));
}
