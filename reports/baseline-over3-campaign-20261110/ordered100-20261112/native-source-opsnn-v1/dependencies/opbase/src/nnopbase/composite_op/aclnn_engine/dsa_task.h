/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_OP_API_COMMON_INC_OPDEV_INTERNAL_DSA_TASK_H
#define OP_API_OP_API_COMMON_INC_OPDEV_INTERNAL_DSA_TASK_H
#include <unordered_map>

#include "acl/acl_rt.h"
#include "kernel_launcher.h"
#include "opdev/op_arg_def.h"

namespace op {
namespace internal {
constexpr size_t IDX_PARAM0 = 16;
constexpr size_t IDX_PARAM1 = 17;

constexpr size_t DSA_TASK_INPUT_NUM = 5;
constexpr size_t DSA_GENBIT_TASK_INPUT_NUM = 4;
constexpr size_t DSA_TASK_OUTPUT_NUM = 1;

class DSATask {
public:
    DSATask()
    {
        memset_s(workspaceHolder_, sizeof(workspaceHolder_), 0, sizeof(workspaceHolder_));
        memset_s(&randomNumTaskInfo_, sizeof(randomNumTaskInfo_), 0, sizeof(randomNumTaskInfo_));
    }

protected:
    aclnnStatus VisitOutputArg(const aclTensor* const arg)
    {
        randomNumTaskInfo_.randomResultAddr = arg->GetData();
        randomNumTaskInfo_.dataType = ToAclDataType(arg->GetDataType());
        return ACLNN_SUCCESS;
    }

    aclnnStatus VisitOutputArgs(OpArgList& argList)
    {
        CHECK_COND(argList.count == DSA_TASK_OUTPUT_NUM, ACLNN_ERR_INNER, "output arg num must be 1.");
        return VisitOutputArg(reinterpret_cast<aclTensor*>(argList[0]->pointer));
    }

    void VisitWorkspaceArg(OpArgList& argList)
    {
        aclTensorList& tensorList = *reinterpret_cast<aclTensorList*>(argList[0]->pointer);
        workspaceBaseAddr_ = tensorList[0]->GetData();
        workspacePhiloxCountAddr_ = workspaceBaseAddr_;
        randomNumTaskInfo_.randomParaAddr = nullptr;
    }

    aclnnStatus ParamMemCpy(const aclrtStream stream) const;

    aclnnStatus Distribute(const aclrtStream stream, OpArgList& inputList, OpArgList& outputList)
    {
        {
            GetThreadLocalContext().profilingInfoId_.summaryItemId_ = GenSummaryItemId(
                GetThreadLocalContext().logInfo_.l2ApiName, GetThreadLocalContext().logInfo_.l0Name,
                GetThreadLocalContext().logInfo_.l0Name);
            OpDfxGuard kernelLaunchGuard(GetThreadLocalContext().profilingInfoId_.summaryItemId_,
                                         DfxProfilingKernelLaunch);
            OP_LOGI("funcType is %d", randomNumTaskInfo_.randomNumFuncParaInfo.funcType);
            auto ret = aclrtRandomNumAsync(&randomNumTaskInfo_, stream, nullptr);
            if (ret != ACL_SUCCESS) {
                OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "rtsLaunchRandomNumTask failed, runtime error code: %d", ret);
                return ACLNN_ERR_RUNTIME_ERROR;
            }
        }
        if (op::internal::opProfilingSwitch.kernelLaunchFlag && op::internal::opProfilingSwitch.additionInfoFlag) {
            op::internal::GetThreadLocalContext().profilingInfoId_.kernelLauncherId_ = GenKernelLauncherId(
                op::internal::GetThreadLocalContext().logInfo_.l0Name);
            GetThreadLocalContext().profilingInfoId_.summaryItemId_ = GenSummaryItemId(
                GetThreadLocalContext().logInfo_.l2ApiName, GetThreadLocalContext().logInfo_.l0Name,
                GetThreadLocalContext().logInfo_.l0Name);
            op::internal::ReportAdditionInfo(inputList, outputList, MSPROF_GE_TASK_TYPE_DSA,
                                             GetThreadLocalContext().profilingInfoId_.summaryItemId_);
        }
        GetCacheOpInfoSwitch(stream);
        if (GetThreadLocalContext().cacheOpInfoSwitch_) {
            ReportCacheOpInfoDSA(inputList, outputList, MSPROF_GE_TASK_TYPE_DSA);
        }
        return ACLNN_SUCCESS;
    }
    void* workspaceBaseAddr_{nullptr};
    void* workspacePhiloxCountAddr_{nullptr};
    uint64_t workspaceHolder_[32]{};
    aclrtRandomNumTaskInfo randomNumTaskInfo_;
};

class DSARandomNormalTask : public DSATask {
public:
    DSARandomNormalTask() : DSATask()
    {
        randomNumTaskInfo_.randomNumFuncParaInfo.funcType = ACL_RT_RANDOM_NUM_FUNC_TYPE_NORMAL_DIS;
    }

    aclnnStatus Run(aclrtStream stream, OpArgContext* opArgCtx)
    {
        VisitWorkspaceArg(*opArgCtx->GetOpArg(op::OP_WORKSPACE_ARG));
        CHECK_RET_CODE(VisitOutputArgs(*opArgCtx->GetOpArg(op::OP_OUTPUT_ARG)), "VisitOutputArgs failed");
        CHECK_RET_CODE(VisitInputArgs(*opArgCtx->GetOpArg(op::OP_INPUT_ARG)), "VisitInputArgs failed");
        CHECK_RET_CODE(ParamMemCpy(stream), "ParamMemCpy failed");
        CHECK_RET_CODE(
            Distribute(stream, *opArgCtx->GetOpArg(op::OP_INPUT_ARG), *opArgCtx->GetOpArg(op::OP_OUTPUT_ARG)),
            "Distribute failed");

        return ACLNN_SUCCESS;
    }

private:
    constexpr static size_t idxCount = 0;
    constexpr static size_t idxSeed = 1;
    constexpr static size_t idxCounter = 2;
    constexpr static size_t idxMean = 3;
    constexpr static size_t idxStd = 4;

    inline aclnnStatus VisitInputArg(size_t idx, const aclTensor* arg)
    {
        if (idx == idxCount) {
            void* dataAddr = arg->GetData();
            memcpy_s(randomNumTaskInfo_.randomNum.valueOrAddr, sizeof(void*), &dataAddr, sizeof(void*));
            randomNumTaskInfo_.randomNum.size = sizeof(void*);
            randomNumTaskInfo_.randomNum.isAddr = true;
        } else if (idx == idxSeed) {
            void* dataAddr = arg->GetData();
            memcpy_s(randomNumTaskInfo_.randomSeed.valueOrAddr, sizeof(void*), &dataAddr, sizeof(void*));
            randomNumTaskInfo_.randomSeed.size = sizeof(void*);
            randomNumTaskInfo_.randomSeed.isAddr = true;
        } else if (idx == idxCounter) {
            randomNumTaskInfo_.randomCounterAddr = arg->GetData();
        } else if (idx == idxMean) {
            workspaceHolder_[IDX_PARAM0] = PtrToValue(arg->GetData());
            void* dataAddr = arg->GetData();
            memcpy_s(randomNumTaskInfo_.randomNumFuncParaInfo.paramInfo.normalDisInfo.mean.valueOrAddr, sizeof(void*),
                     &dataAddr, sizeof(void*));
            randomNumTaskInfo_.randomNumFuncParaInfo.paramInfo.normalDisInfo.mean.size = sizeof(void*);
            randomNumTaskInfo_.randomNumFuncParaInfo.paramInfo.normalDisInfo.mean.isAddr = true;
        } else if (idx == idxStd) {
            workspaceHolder_[IDX_PARAM1] = PtrToValue(arg->GetData());
            void* dataAddr = arg->GetData();
            memcpy_s(randomNumTaskInfo_.randomNumFuncParaInfo.paramInfo.normalDisInfo.stddev.valueOrAddr, sizeof(void*),
                     &dataAddr, sizeof(void*));
            randomNumTaskInfo_.randomNumFuncParaInfo.paramInfo.normalDisInfo.stddev.size = sizeof(void*);
            randomNumTaskInfo_.randomNumFuncParaInfo.paramInfo.normalDisInfo.stddev.isAddr = true;
        }
        return ACLNN_SUCCESS;
    }

    inline aclnnStatus VisitInputArg(size_t idx, const uint64_t arg)
    {
        if (idx == idxCount) {
            memcpy_s(randomNumTaskInfo_.randomNum.valueOrAddr, sizeof(uint64_t), &arg, sizeof(uint64_t));
            randomNumTaskInfo_.randomNum.size = sizeof(uint64_t);
            randomNumTaskInfo_.randomNum.isAddr = false;
        } else if (idx == idxSeed) {
            memcpy_s(randomNumTaskInfo_.randomSeed.valueOrAddr, sizeof(uint64_t), &arg, sizeof(uint64_t));
            randomNumTaskInfo_.randomSeed.size = sizeof(uint64_t);
            randomNumTaskInfo_.randomSeed.isAddr = false;
        } else if (idx == idxCounter) {
            workspaceHolder_[0] = 0;
            workspaceHolder_[1] = arg;
            randomNumTaskInfo_.randomCounterAddr = workspacePhiloxCountAddr_;
        }
        return ACLNN_SUCCESS;
    }

    inline aclnnStatus VisitInputArg(size_t idx, const aclScalar* arg)
    {
        if (idx == idxMean) {
            const auto rc = memcpy_s(&workspaceHolder_[IDX_PARAM0], sizeof(uint64_t), arg->GetData(), arg->Size());
            if (rc != EOK) {
                OP_LOGE(ACLNN_ERR_INNER, "memcpy first input arg fail.");
                return ACLNN_ERR_INNER;
            }

            memcpy_s(randomNumTaskInfo_.randomNumFuncParaInfo.paramInfo.normalDisInfo.mean.valueOrAddr, arg->Size(),
                     arg->GetData(), arg->Size());
            randomNumTaskInfo_.randomNumFuncParaInfo.paramInfo.normalDisInfo.mean.size = arg->Size();
            randomNumTaskInfo_.randomNumFuncParaInfo.paramInfo.normalDisInfo.mean.isAddr = false;
        } else if (idx == idxStd) {
            const auto rc = memcpy_s(&workspaceHolder_[IDX_PARAM1], sizeof(uint64_t), arg->GetData(), arg->Size());
            if (rc != EOK) {
                OP_LOGE(ACLNN_ERR_INNER, "memcpy first input arg fail.");
                return ACLNN_ERR_INNER;
            }

            memcpy_s(randomNumTaskInfo_.randomNumFuncParaInfo.paramInfo.normalDisInfo.stddev.valueOrAddr, arg->Size(),
                     arg->GetData(), arg->Size());
            randomNumTaskInfo_.randomNumFuncParaInfo.paramInfo.normalDisInfo.stddev.size = arg->Size();
            randomNumTaskInfo_.randomNumFuncParaInfo.paramInfo.normalDisInfo.stddev.isAddr = false;
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus VisitInputArg(size_t idx, OpArg& arg)
    {
        switch (arg.type) {
            case OpArgType::OPARG_ACLTENSOR:
                return VisitInputArg(idx, reinterpret_cast<aclTensor*>(arg->pointer));
            case OpArgType::OPARG_INT:
            case OpArgType::OPARG_UINT:
                return VisitInputArg(idx, reinterpret_cast<uint64_t>(arg->value));
            case OpArgType::OPARG_ACLSCALAR:
                return VisitInputArg(idx, reinterpret_cast<aclScalar*>(arg->pointer));
            default:
                OP_LOGE(ACLNN_ERR_INNER, "invalid arg type %d.", static_cast<int>(arg.type));
                return ACLNN_ERR_INNER;
        }
    }

    aclnnStatus VisitInputArgs(OpArgList& argList)
    {
        CHECK_COND(argList.count == DSA_TASK_INPUT_NUM, ACLNN_ERR_INNER,
                   "DSARandomNormal or DSARandomTruncatedNormal input arg num must be 5.");
        return argList.VisitBy([this](size_t idx, OpArg& arg) -> aclnnStatus { return VisitInputArg(idx, arg); });
    }
};

class DSARandomTruncatedNormalTask : public DSARandomNormalTask {
public:
    DSARandomTruncatedNormalTask() : DSARandomNormalTask()
    {
        randomNumTaskInfo_.randomNumFuncParaInfo.funcType = ACL_RT_RANDOM_NUM_FUNC_TYPE_TRUNCATED_NORMAL_DIS;
    }
};

class DSARandomUniformTask : public DSATask {
public:
    DSARandomUniformTask() : DSATask()
    {
        randomNumTaskInfo_.randomNumFuncParaInfo.funcType = ACL_RT_RANDOM_NUM_FUNC_TYPE_UNIFORM_DIS;
    }

    aclnnStatus Run(aclrtStream stream, OpArgContext* opArgCtx)
    {
        VisitWorkspaceArg(*opArgCtx->GetOpArg(op::OP_WORKSPACE_ARG));
        CHECK_RET_CODE(VisitOutputArgs(*opArgCtx->GetOpArg(op::OP_OUTPUT_ARG)), "VisitOutputArgs failed");
        CHECK_RET_CODE(VisitInputArgs(*opArgCtx->GetOpArg(op::OP_INPUT_ARG)), "VisitInputArgs failed");
        CHECK_RET_CODE(ParamMemCpy(stream), "ParamMemCpy failed");
        CHECK_RET_CODE(
            Distribute(stream, *opArgCtx->GetOpArg(op::OP_INPUT_ARG), *opArgCtx->GetOpArg(op::OP_OUTPUT_ARG)),
            "Distribute failed");

        return ACLNN_SUCCESS;
    }

private:
    constexpr static size_t idxCount = 0;
    constexpr static size_t idxSeed = 1;
    constexpr static size_t idxCounter = 2;
    constexpr static size_t idxMin = 3;
    constexpr static size_t idxMax = 4;

    inline aclnnStatus VisitInputArg(size_t idx, const aclTensor* arg)
    {
        if (idx == idxCount) {
            void* dataAddr = arg->GetData();
            memcpy_s(randomNumTaskInfo_.randomNum.valueOrAddr, sizeof(void*), &dataAddr, sizeof(void*));
            randomNumTaskInfo_.randomNum.size = sizeof(void*);
            randomNumTaskInfo_.randomNum.isAddr = true;
        } else if (idx == idxSeed) {
            void* dataAddr = arg->GetData();
            memcpy_s(randomNumTaskInfo_.randomSeed.valueOrAddr, sizeof(void*), &dataAddr, sizeof(void*));
            randomNumTaskInfo_.randomSeed.size = sizeof(void*);
            randomNumTaskInfo_.randomSeed.isAddr = true;
        } else if (idx == idxCounter) {
            randomNumTaskInfo_.randomCounterAddr = arg->GetData();
        } else if (idx == idxMin) {
            workspaceHolder_[IDX_PARAM0] = PtrToValue(arg->GetData());
            void* dataAddr = arg->GetData();
            memcpy_s(randomNumTaskInfo_.randomNumFuncParaInfo.paramInfo.uniformDisInfo.min.valueOrAddr, sizeof(void*),
                     &dataAddr, sizeof(void*));
            randomNumTaskInfo_.randomNumFuncParaInfo.paramInfo.uniformDisInfo.min.size = sizeof(void*);
            randomNumTaskInfo_.randomNumFuncParaInfo.paramInfo.uniformDisInfo.min.isAddr = true;
        } else if (idx == idxMax) {
            workspaceHolder_[IDX_PARAM1] = PtrToValue(arg->GetData());
            void* dataAddr = arg->GetData();
            memcpy_s(randomNumTaskInfo_.randomNumFuncParaInfo.paramInfo.uniformDisInfo.max.valueOrAddr, sizeof(void*),
                     &dataAddr, sizeof(void*));
            randomNumTaskInfo_.randomNumFuncParaInfo.paramInfo.uniformDisInfo.max.size = sizeof(void*);
            randomNumTaskInfo_.randomNumFuncParaInfo.paramInfo.uniformDisInfo.max.isAddr = true;
        }

        return ACLNN_SUCCESS;
    }

    inline aclnnStatus VisitInputArg(size_t idx, const uint64_t arg)
    {
        if (idx == idxCount) {
            memcpy_s(randomNumTaskInfo_.randomNum.valueOrAddr, sizeof(uint64_t), &arg, sizeof(uint64_t));
            randomNumTaskInfo_.randomNum.size = sizeof(uint64_t);
            randomNumTaskInfo_.randomNum.isAddr = false;
        } else if (idx == idxSeed) {
            memcpy_s(randomNumTaskInfo_.randomSeed.valueOrAddr, sizeof(uint64_t), &arg, sizeof(uint64_t));
            randomNumTaskInfo_.randomSeed.size = sizeof(uint64_t);
            randomNumTaskInfo_.randomSeed.isAddr = false;
        } else if (idx == idxCounter) {
            workspaceHolder_[0] = 0;
            workspaceHolder_[1] = arg;
            randomNumTaskInfo_.randomCounterAddr = workspacePhiloxCountAddr_;
        }

        return ACLNN_SUCCESS;
    }

    inline aclnnStatus VisitInputArg(size_t idx, const aclScalar* arg)
    {
        if (idx == idxMin) {
            const auto rc = memcpy_s(&workspaceHolder_[IDX_PARAM0], sizeof(uint64_t), arg->GetData(), arg->Size());
            if (rc != EOK) {
                OP_LOGE(ACLNN_ERR_INNER, "memcpy first input arg fail.");
                return ACLNN_ERR_INNER;
            }
            memcpy_s(randomNumTaskInfo_.randomNumFuncParaInfo.paramInfo.uniformDisInfo.min.valueOrAddr, arg->Size(),
                     arg->GetData(), arg->Size());
            randomNumTaskInfo_.randomNumFuncParaInfo.paramInfo.uniformDisInfo.min.size = arg->Size();
            randomNumTaskInfo_.randomNumFuncParaInfo.paramInfo.uniformDisInfo.min.isAddr = false;
        } else if (idx == idxMax) {
            const auto rc = memcpy_s(&workspaceHolder_[IDX_PARAM1], sizeof(uint64_t), arg->GetData(), arg->Size());
            if (rc != EOK) {
                OP_LOGE(ACLNN_ERR_INNER, "memcpy first input arg fail.");
                return ACLNN_ERR_INNER;
            }
            memcpy_s(randomNumTaskInfo_.randomNumFuncParaInfo.paramInfo.uniformDisInfo.max.valueOrAddr, arg->Size(),
                     arg->GetData(), arg->Size());
            randomNumTaskInfo_.randomNumFuncParaInfo.paramInfo.uniformDisInfo.max.size = arg->Size();
            randomNumTaskInfo_.randomNumFuncParaInfo.paramInfo.uniformDisInfo.max.isAddr = false;
        }

        return ACLNN_SUCCESS;
    }

    aclnnStatus VisitInputArg(size_t idx, OpArg& arg)
    {
        switch (arg.type) {
            case OpArgType::OPARG_ACLTENSOR:
                return VisitInputArg(idx, reinterpret_cast<aclTensor*>(arg->pointer));
            case OpArgType::OPARG_INT:
            case OpArgType::OPARG_UINT:
                return VisitInputArg(idx, reinterpret_cast<uint64_t>(arg->value));
            case OpArgType::OPARG_ACLSCALAR:
                return VisitInputArg(idx, reinterpret_cast<aclScalar*>(arg->pointer));
            default:
                OP_LOGE(ACLNN_ERR_INNER, "invalid arg type %d.", static_cast<int>(arg.type));
                return ACLNN_ERR_INNER;
        }
    }

    aclnnStatus VisitInputArgs(OpArgList& argList)
    {
        CHECK_COND(argList.count == DSA_TASK_INPUT_NUM, ACLNN_ERR_INNER,
                   "DSARandomUniformTask input arg num must be 5.");
        return argList.VisitBy([this](size_t idx, OpArg& arg) -> aclnnStatus { return VisitInputArg(idx, arg); });
    }
};

class DSAGenBitMaskTask : public DSATask {
public:
    DSAGenBitMaskTask() : DSATask()
    {
        randomNumTaskInfo_.randomNumFuncParaInfo.funcType = ACL_RT_RANDOM_NUM_FUNC_TYPE_DROPOUT_BITMASK;
    }

    aclnnStatus Run(aclrtStream stream, OpArgContext* opArgCtx)
    {
        VisitWorkspaceArg(*opArgCtx->GetOpArg(op::OP_WORKSPACE_ARG));
        CHECK_RET_CODE(VisitOutputArgs(*opArgCtx->GetOpArg(op::OP_OUTPUT_ARG)), "VisitOutputArgs failed");
        CHECK_RET_CODE(VisitInputArgs(*opArgCtx->GetOpArg(op::OP_INPUT_ARG)), "VisitInputArgs failed");
        CHECK_RET_CODE(ParamMemCpy(stream), "ParamMemCpy failed");
        CHECK_RET_CODE(
            Distribute(stream, *opArgCtx->GetOpArg(op::OP_INPUT_ARG), *opArgCtx->GetOpArg(op::OP_OUTPUT_ARG)),
            "Distribute failed");

        return ACLNN_SUCCESS;
    }

private:
    constexpr static size_t idxCount = 0;
    constexpr static size_t idxSeed = 1;
    constexpr static size_t idxCounter = 2;
    constexpr static size_t idxDropoutRatio = 3;

    inline aclnnStatus VisitInputArg(size_t idx, const aclTensor* arg)
    {
        if (idx == idxCount) {
            void* dataAddr = arg->GetData();
            memcpy_s(randomNumTaskInfo_.randomNum.valueOrAddr, sizeof(void*), &dataAddr, sizeof(void*));
            randomNumTaskInfo_.randomNum.size = sizeof(void*);
            randomNumTaskInfo_.randomNum.isAddr = true;
        } else if (idx == idxSeed) {
            void* dataAddr = arg->GetData();
            memcpy_s(randomNumTaskInfo_.randomSeed.valueOrAddr, sizeof(void*), &dataAddr, sizeof(void*));
            randomNumTaskInfo_.randomSeed.size = sizeof(void*);
            randomNumTaskInfo_.randomSeed.isAddr = true;
        } else if (idx == idxCounter) {
            randomNumTaskInfo_.randomCounterAddr = arg->GetData();
        } else if (idx == idxDropoutRatio) {
            workspaceHolder_[IDX_PARAM0] = PtrToValue(arg->GetData());
            void* dataAddr = arg->GetData();
            memcpy_s(randomNumTaskInfo_.randomNumFuncParaInfo.paramInfo.dropoutBitmaskInfo.dropoutRation.valueOrAddr,
                     sizeof(void*), &dataAddr, sizeof(void*));
            randomNumTaskInfo_.randomNumFuncParaInfo.paramInfo.dropoutBitmaskInfo.dropoutRation.size = sizeof(void*);
            randomNumTaskInfo_.randomNumFuncParaInfo.paramInfo.dropoutBitmaskInfo.dropoutRation.isAddr = true;
        }

        return ACLNN_SUCCESS;
    }

    inline aclnnStatus VisitInputArg(size_t idx, const uint64_t arg)
    {
        if (idx == idxCount) {
            memcpy_s(randomNumTaskInfo_.randomNum.valueOrAddr, sizeof(uint64_t), &arg, sizeof(uint64_t));
            randomNumTaskInfo_.randomNum.size = sizeof(uint64_t);
            randomNumTaskInfo_.randomNum.isAddr = false;
        } else if (idx == idxSeed) {
            memcpy_s(randomNumTaskInfo_.randomSeed.valueOrAddr, sizeof(uint64_t), &arg, sizeof(uint64_t));
            randomNumTaskInfo_.randomSeed.size = sizeof(uint64_t);
            randomNumTaskInfo_.randomSeed.isAddr = false;
        } else if (idx == idxCounter) {
            workspaceHolder_[0] = 0;
            workspaceHolder_[1] = arg;
            randomNumTaskInfo_.randomCounterAddr = workspacePhiloxCountAddr_;
        }

        return ACLNN_SUCCESS;
    }

    inline aclnnStatus VisitInputArg(size_t idx, const aclScalar* arg)
    {
        if (idx == idxDropoutRatio) {
            const auto rc = memcpy_s(&workspaceHolder_[IDX_PARAM0], sizeof(uint64_t), arg->GetData(), arg->Size());
            if (rc != EOK) {
                OP_LOGE(ACLNN_ERR_INNER, "memcpy first input arg fail.");
                return ACLNN_ERR_INNER;
            }
            memcpy_s(randomNumTaskInfo_.randomNumFuncParaInfo.paramInfo.dropoutBitmaskInfo.dropoutRation.valueOrAddr,
                     arg->Size(), arg->GetData(), arg->Size());
            randomNumTaskInfo_.randomNumFuncParaInfo.paramInfo.dropoutBitmaskInfo.dropoutRation.size = arg->Size();
            randomNumTaskInfo_.randomNumFuncParaInfo.paramInfo.dropoutBitmaskInfo.dropoutRation.isAddr = false;

            randomNumTaskInfo_.dataType = ToAclDataType(arg->GetDataType());
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus VisitInputArg(size_t idx, OpArg& arg)
    {
        switch (arg.type) {
            case OpArgType::OPARG_ACLTENSOR:
                return VisitInputArg(idx, reinterpret_cast<aclTensor*>(arg->pointer));
            case OpArgType::OPARG_INT:
            case OpArgType::OPARG_UINT:
                return VisitInputArg(idx, reinterpret_cast<uint64_t>(arg->value));
            case OpArgType::OPARG_ACLSCALAR:
                return VisitInputArg(idx, reinterpret_cast<aclScalar*>(arg->pointer));
            default:
                OP_LOGE(ACLNN_ERR_INNER, "invalid arg type %d.", static_cast<int>(arg.type));
                return ACLNN_ERR_INNER;
        }
    }

    aclnnStatus VisitInputArgs(OpArgList& argList)
    {
        CHECK_COND(argList.count == DSA_GENBIT_TASK_INPUT_NUM, ACLNN_ERR_INNER,
                   "DSARandomUniformTask input arg num must be 4.");
        return argList.VisitBy([this](size_t idx, OpArg& arg) -> aclnnStatus { return VisitInputArg(idx, arg); });
    }

    aclnnStatus VisitOutputArg(const aclTensor* const arg)
    {
        randomNumTaskInfo_.randomResultAddr = arg->GetData();
        return ACLNN_SUCCESS;
    }

    aclnnStatus VisitOutputArgs(OpArgList& argList)
    {
        CHECK_COND(argList.count == DSA_TASK_OUTPUT_NUM, ACLNN_ERR_INNER, "output arg num must be 1.");
        return VisitOutputArg(reinterpret_cast<aclTensor*>(argList[0]->pointer));
    }
};

template <typename C>
class DSAKernelLauncher : public KernelLauncher {
public:
    DSAKernelLauncher(uint32_t opType, CoreType coreType, const ProfilingInfoId& profilingId,
                      const aclOpExecutor* executor, const C& task, OpArgContext* opArgCtx)
        : KernelLauncher(opType, coreType, executor, profilingId), args_(opArgCtx), task_(task)
    {}

    ~DSAKernelLauncher() override
    {
        if (args_) {
            op::DestroyOpArgContext(args_);
            args_ = nullptr;
        }
    }

    aclnnStatus Launch() override
    {
        op::internal::GetThreadLocalContext().logInfo_.l0Name = opLogInfo_.l0Name;
        op::internal::GetThreadLocalContext().profilingInfoId_ = profilingInfoId_;

        const auto ret = task_.Run(executor_->GetStream(), args_);
        if (ret != ACLNN_SUCCESS) {
            OP_LOGE(ACLNN_ERR_INNER, "DSA task distribute error.");
            return ACLNN_ERR_INNER;
        }
        return ACLNN_SUCCESS;
    }

    internal::OpKernelBin* GetBin() override { return nullptr; }

    bool CheckRepeatable([[maybe_unused]] const std::unordered_map<const aclStorage*, const aclStorage*>& relation,
                         [[maybe_unused]] const std::vector<const aclStorage*>& oriStorage) override
    {
        return false;
    }

private:
    OpArgContext* args_{nullptr};
    C task_;
};

void GetDSATaskWorkspace(aclOpExecutor* executor, aclTensorList** workspace);
} // namespace internal
} // namespace op
#endif
