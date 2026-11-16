/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>

#include "opdev/aicpu/aicpu_task.h"
#include "opdev/aicpu/aicpu_kernel_launcher.h"
#include "aicpu_engine_struct.h"
#include "aclnn/aclnn_base.h"
#include "opdev/fast_vector.h"
#include "mmpa/mmpa_api.h"
#include "op_dfx_internal.h"
#include "opdev/op_dfx.h"
#include "aicpu_json_load_manager.h"
#include "runtime/rt_external.h"

namespace op {
namespace internal {
namespace {
constexpr uint16_t kModelId = 36; // AICPU
const std::string kDefaultKernelSo = "libcpu_kernels.so";
const std::set<std::string> pt_ops = {"Index", "IndexPut", "LeftShift"};
const std::string kPtKernelSo = "libpt_kernels.so";
constexpr uint32_t KERNEL_TYPE_FWK = 1U;
constexpr uint32_t KERNEL_TYPE_AICPU = 2U;
constexpr uint32_t RT_KERNEL_DEFAULT_IN_AICPU = 0U;

/* To ensure compatibility between the opp_kernel and runtime packages,
   the following compatibility allowlist is established for operators
   that were previously invoked at the L0 layer but are not included in the operator information library.
   Subsequent efforts will focus on refactoring the operators in this allowlist before eventually removing it.
*/
const std::set<std::string> tfCompatibleOps = {
    "ResizeBilinear", "ResizeNearestNeighbor", "Roll", "Max", "Min", "Mean", "Prod", "Sum"};
const std::set<std::string> aicpuCompatibleOps = {"CacheVerification",
                                                  "AicpuPathwayVerification",
                                                  "FaultInjection",
                                                  "DataCompare",
                                                  "VoltageRegulator",
                                                  "Floor",
                                                  "Pad",
                                                  "ReduceSum",
                                                  "Sigmoid"};

void GetConstVec(const op::FVector<aclTensor*>& in, op::FVector<const aclTensor*>& out)
{
    for (auto t : in) {
        out.emplace_back(t);
    }
}
void GetValidInputVec(const op::FVector<const aclTensor*>& in, op::FVector<const aclTensor*>& out)
{
    for (auto t : in) {
        if (t == nullptr) {
            continue;
        }
        out.emplace_back(t);
    }
}

class AiCpuOneShotKernelLauncher : public KernelLauncher {
public:
    AiCpuOneShotKernelLauncher(uint32_t optype, op::CoreType coreType, const aclOpExecutor* opExe,
                               op::internal::ProfilingInfoId& profilingInfoId, std::unique_ptr<AicpuTask> task,
                               OpArgContext* opArgCtx)
        : KernelLauncher(optype, coreType, opExe, profilingInfoId), task_(std::move(task)), args_(opArgCtx)
    {}

    ~AiCpuOneShotKernelLauncher() override
    {
        OP_LOGI("Destroy one-shot aicpu launcher, opType=%u, task_alive=%d, args_alive=%d.", opType_, task_ != nullptr,
                args_ != nullptr);
        if (args_) {
            op::DestroyOpArgContext(args_);
            args_ = nullptr;
        }
    }

    aclnnStatus Launch() override
    {
        AICPU_ASSERT_NOTNULL_RETVAL(task_);
        AICPU_ASSERT_OK_RETVAL(task_->SetIoTensors(const_cast<aclOpExecutor*>(executor_), args_));
        return task_->Run(const_cast<aclOpExecutor*>(executor_), executor_->GetStream());
    }

    internal::OpKernelBin* GetBin() override { return nullptr; }

    bool CheckRepeatable([[maybe_unused]] const std::unordered_map<const aclStorage*, const aclStorage*>& relation,
                         [[maybe_unused]] const std::vector<const aclStorage*>& oriStorage) override
    {
        return false;
    }

private:
    std::unique_ptr<AicpuTask> task_;
    OpArgContext* args_{nullptr};
};
} // namespace

std::unique_ptr<AicpuTask> TakePendingOverflowTask(AicpuTask* task);

aclnnStatus AicpuTfTask::Init(const FVector<const aclTensor*>& inputs, const FVector<aclTensor*>& outputs,
                              const AicpuAttrs& attrs)
{
    launchId_ = MsprofGetHashId(opType_.c_str(), opType_.size());
    summaryItemId_ = GenSummaryItemId(GetThreadLocalContext().logInfo_.l2ApiName,
                                      GetThreadLocalContext().logInfo_.l0Name, opType_.c_str());
    // ge::MakeUnique
    extInfoHandle_ = std::make_unique<AicpuExtInfoHandler>(opType_, inputs.size(), outputs.size(), unknownType_);
    AICPU_ASSERT_NOTNULL_RETVAL(extInfoHandle_);
    extInfoHandle_->SetSpace(space_);
    std::string extInfo;
    AICPU_ASSERT_OK_RETVAL(extInfoHandle_->GenTfExtBuffer(inputs, outputs, extInfo));

    const bool needDeviceExt = (unknownType_ == ge::DEPEND_SHAPE_RANGE) ? true : false;
    if (needDeviceExt) {
        deviceExtMemSize_ = static_cast<uint64_t>(extInfo.size());
    }
    uint32_t ioNum = inputs.size() + outputs.size();
    // ge::MakeUnique
    auto tf_argsHandle = std::make_unique<AicpuTfArgsHandler>(opType_, opType_, ioNum, needDeviceExt);
    AICPU_ASSERT_NOTNULL_RETVAL(tf_argsHandle);
    tf_argsHandle->SetSpace(space_);

    // build tf args
    STR_FWK_OP_KERNEL fwkOpKernel;
    std::string taskInfo;
    AICPU_ASSERT_OK_RETVAL(tf_argsHandle->GenTfArgs(inputs, outputs, attrs, fwkOpKernel, taskInfo));
    AICPU_ASSERT_OK_RETVAL(tf_argsHandle->BuildTfArgs(fwkOpKernel, taskInfo, extInfo.size()));

    // build ext info handle
    uint8_t* hostExtAddr = tf_argsHandle->GetExtInfoAddr();
    AICPU_ASSERT_OK_RETVAL(extInfoHandle_->Parse(extInfo, hostExtAddr));
    argsHandle_ = std::move(tf_argsHandle);

    // load bin from json
    AICPU_ASSERT_OK_RETVAL(JsonLoadManger::LoadTfBinaryFromJson());
    tfBinHandle_ = JsonLoadManger::GetTfBinaryHandle();
    AICPU_ASSERT_NOTNULL_RETVAL(tfBinHandle_);
    AICPU_ASSERT_OK_RETVAL(JsonLoadManger::SetSupportedNewLaunchFlag());
    if ((tfCompatibleOps.find(opType_) != tfCompatibleOps.end()) || !JsonLoadManger::IsSupportedNewLaunch()) {
        OP_LOGI("Cannot find the '%s' opType function by name. Try compatible with the old process.", opType_.c_str());
        useNewLaunchInterface_ = false;
    } else {
        useNewLaunchInterface_ = true;
    }

    return OK;
}

aclnnStatus AicpuTfTask::LaunchKernelByNewInterface(aclrtStream stream)
{
    AICPU_ASSERT_OK_RETVAL(aclrtBinaryGetFunction(tfBinHandle_, opType_.c_str(), &funcHandle_));
    AICPU_ASSERT_NOTNULL_RETVAL(funcHandle_);
    auto lauchKernelAttr = std::make_unique<aclrtLaunchKernelAttr>();
    AICPU_ASSERT_NOTNULL_RETVAL(lauchKernelAttr);
    aclrtLaunchKernelCfg kernelLaunchCfg = {lauchKernelAttr.get(), 0U};
    const auto& aicpuArg = argsHandle_->GetArgsEx();
    AICPU_ASSERT_OK_RETVAL(aclrtLaunchKernelWithHostArgs(funcHandle_, 1U, stream, &kernelLaunchCfg, aicpuArg.args,
                                                         aicpuArg.argsSize, aicpuArg.hostInputInfoPtr,
                                                         aicpuArg.hostInputInfoNum));

    return OK;
}

aclnnStatus AicpuTfTask::Run(aclOpExecutor* executor, aclrtStream stream)
{
    RecordAicpuTime(kUpdateArgsStart);
    argsHandle_->UpdateDeviceExtInfoAddr(extInfoHandle_->deviceExtInfo_);
    AICPU_ASSERT_OK_RETVAL(
        argsHandle_->UpdateIoAddr(inputs_, outputs_, stream, executor, deviceExtMemSize_, deviceCacheOffset_));
    RecordAicpuTime(kUpdateArgsEnd);
    {
        OpDfxGuard kernelLaunchGuard(summaryItemId_, DfxProfilingKernelLaunch);
        if (useNewLaunchInterface_) {
            AICPU_ASSERT_RTOK_RETVAL(LaunchKernelByNewInterface(stream));
        } else {
            OP_LOGI("Compatible with the old process.");
            const auto& args = argsHandle_->GetArgsEx();
            rtAicpuArgsEx_t rtsLaunchArgs = {};
            rtsLaunchArgs.args = args.args;
            rtsLaunchArgs.argsSize = args.argsSize;
            std::vector<rtHostInputInfo_t> hostInputs;
            for (size_t i = 0; i < args.hostInputInfoNum; ++i) {
                rtHostInputInfo_t hostInput = {};
                hostInput.addrOffset = args.hostInputInfoPtr[i].addrOffset;
                hostInput.dataOffset = args.hostInputInfoPtr[i].dataOffset;
                hostInputs.emplace_back(hostInput);
            }
            rtsLaunchArgs.hostInputInfoPtr = hostInputs.data();
            rtsLaunchArgs.hostInputInfoNum = args.hostInputInfoNum;
            rtsLaunchArgs.soNameAddrOffset = args.soNameAddrOffset;
            rtsLaunchArgs.kernelNameAddrOffset = args.kernelNameAddrOffset;

            AICPU_ASSERT_RTOK_RETVAL(rtAicpuKernelLaunchExWithArgs(KERNEL_TYPE_FWK, opType_.c_str(), 1U, &rtsLaunchArgs,
                                                                   nullptr, stream, RT_KERNEL_DEFAULT_IN_AICPU));
        }
    }
    RecordAicpuTime(kLaunchEnd);

    RecordAicpuTime(kShapeD2hCopyEnd);
    if (unknownType_ == ge::DEPEND_SHAPE_RANGE) {
        AICPU_ASSERT_OK_RETVAL(extInfoHandle_->UpdateOutputShapeFromExtInfo(outputs_, stream));
    }
    RecordAicpuTime(kUpdateOutputShapeEnd);

    if (op::internal::opProfilingSwitch.kernelLaunchFlag && op::internal::opProfilingSwitch.additionInfoFlag) {
        GetThreadLocalContext().profilingInfoId_.kernelLauncherId_ = launchId_; // optype
        op::FVector<const aclTensor*> input;
        op::FVector<const aclTensor*> out;
        GetValidInputVec(inputs_, input);
        GetConstVec(outputs_, out);
        // summaryItemId for op name
        op::internal::ReportAdditionInfo(input, out, MSPROF_GE_TASK_TYPE_AI_CPU, summaryItemId_);
    }

    op::internal::PrintAicpuAllTimeStampInfo(opType_.c_str());
    return OK;
}

aclnnStatus AicpuCCTask::GetKernelNameAndSoName(std::string& kernelSoName)
{
    if (JsonLoadManger::FindAndGetInCustomRegistry(opType_, kernelSoName, functionName_)) {
        OP_LOGI("The Operator %s found in package registry: kernel so name is %s, function name is %s.",
                opType_.c_str(), kernelSoName.c_str(), functionName_.c_str());
        std::string kernelSoPath = "";
        AICPU_ASSERT_OK_RETVAL(JsonLoadManger::LoadAicpuCustBinaryFromJson(opType_, kernelSoName, kernelSoPath));
        aicpuBinHandle_ = JsonLoadManger::GetAicpuCustBinaryHandle(kernelSoPath);
        AICPU_ASSERT_NOTNULL_RETVAL(aicpuBinHandle_);
        // If not specifically configured, the RTS interface will be updated to the actual so name, causing
        // inconsistency with the so name on the device side.
        kernelSoName = "custom_sub_repository.so";
        isCustomTask_ = true;
        useNewLaunchInterface_ = true;
    } else {
        if (pt_ops.find(opType_) != pt_ops.end()) {
            kernelSoName = kPtKernelSo;
        }
        // load bin from json
        AICPU_ASSERT_OK_RETVAL(JsonLoadManger::LoadAicpuBinaryFromJson());
        aicpuBinHandle_ = JsonLoadManger::GetAicpuBinaryHandle();
        AICPU_ASSERT_NOTNULL_RETVAL(aicpuBinHandle_);
        if ((aicpuCompatibleOps.find(opType_) != aicpuCompatibleOps.end()) || !JsonLoadManger::IsSupportedNewLaunch()) {
            OP_LOGI("Cannot find the '%s' opType function by name. Try compatible with the old process.",
                    opType_.c_str());
            useNewLaunchInterface_ = false;
        } else {
            useNewLaunchInterface_ = true;
        }
        isCustomTask_ = false;
    }
    return OK;
}

aclnnStatus AicpuCCTask::Init(const FVector<const aclTensor*>& inputs, const FVector<aclTensor*>& outputs,
                              const AicpuAttrs& attrs)
{
    OP_LOGI("Start AicpuCCTask::Init, opType[%s]", opType_.c_str());
    // Only check if custom operators exist; If present, parse it. if not, continue with the original process
    // unaffected.
    (void)JsonLoadManger::CustJsonLoadAndParse();
    launchId_ = MsprofGetHashId(opType_.c_str(), opType_.size());
    summaryItemId_ = GenSummaryItemId(op::internal::GetThreadLocalContext().logInfo_.l2ApiName,
                                      op::internal::GetThreadLocalContext().logInfo_.l0Name, opType_.c_str());

    extInfoHandle_ = std::make_unique<AicpuExtInfoHandler>(opType_, inputs.size(), outputs.size(), unknownType_);
    AICPU_ASSERT_NOTNULL_RETVAL(extInfoHandle_);
    std::string extInfo;
    AICPU_ASSERT_OK_RETVAL(extInfoHandle_->GenCCExtBuffer(inputs, outputs, extInfo));

    const bool needDeviceExt = (unknownType_ == ge::DEPEND_SHAPE_RANGE) ? true : false;
    if (needDeviceExt) {
        deviceExtMemSize_ = static_cast<uint64_t>(extInfo.size());
    }
    const uint32_t ioNum = inputs.size() + outputs.size();
    auto cc_argsHandle = std::make_unique<AicpuCCArgsHandler>(opType_, opType_, ioNum, needDeviceExt);
    AICPU_ASSERT_NOTNULL_RETVAL(cc_argsHandle);

    std::string taskInfo;
    AICPU_ASSERT_OK_RETVAL(cc_argsHandle->GenCCArgs(inputs, outputs, attrs, taskInfo));
    AICPU_ASSERT_OK_RETVAL(JsonLoadManger::SetSupportedNewLaunchFlag());

    // 这个需要传递进来
    std::string kernelSoName = kDefaultKernelSo;
    AICPU_ASSERT_OK_RETVAL(GetKernelNameAndSoName(kernelSoName));
    AICPU_ASSERT_OK_RETVAL(cc_argsHandle->BuildCCArgs(taskInfo, functionName_, kernelSoName, extInfo.size()));
    uint8_t* hostExtAddr = cc_argsHandle->GetExtInfoAddr();
    // 解析extend info 并且填充AicpuExtInfoHandler的结构体应该没啥必要, 应该只需要input/output shape
    extInfoHandle_->Parse(extInfo, hostExtAddr);
    argsHandle_ = std::move(cc_argsHandle);
    OP_LOGI("Finish AicpuCCTask::Init, opType[%s]", opType_.c_str());
    return OK;
}

aclnnStatus AicpuCCTask::LaunchKernelByNewInterface(aclrtStream stream)
{
    if (isCustomTask_) {
        AICPU_ASSERT_OK_RETVAL(
            aclrtRegisterCpuFunc(aicpuBinHandle_, functionName_.c_str(), opType_.c_str(), &funcHandle_));
    } else {
        AICPU_ASSERT_OK_RETVAL(aclrtBinaryGetFunction(aicpuBinHandle_, opType_.c_str(), &funcHandle_));
    }
    AICPU_ASSERT_NOTNULL_RETVAL(funcHandle_);
    auto lauchKernelAttr = std::make_unique<aclrtLaunchKernelAttr>();
    AICPU_ASSERT_NOTNULL_RETVAL(lauchKernelAttr);
    aclrtLaunchKernelCfg kernelLaunchCfg = {lauchKernelAttr.get(), 0U};
    const auto& aicpuArg = argsHandle_->GetArgsEx();
    AICPU_ASSERT_RTOK_RETVAL(aclrtLaunchKernelWithHostArgs(funcHandle_, 1U, stream, &kernelLaunchCfg, aicpuArg.args,
                                                           aicpuArg.argsSize, aicpuArg.hostInputInfoPtr,
                                                           aicpuArg.hostInputInfoNum));

    return OK;
}

aclnnStatus AicpuCCTask::Run(aclOpExecutor* executor, aclrtStream stream)
{
    RecordAicpuTime(kUpdateArgsStart);
    OP_LOGI("Start AicpuCCTask::Run, opType[%s]", opType_.c_str());
    argsHandle_->UpdateDeviceExtInfoAddr(extInfoHandle_->deviceExtInfo_);
    if ((stream_ != nullptr) && (stream_ != stream)) {
        extInfoHandle_->UpdateKernelId();
    }
    stream_ = stream;

    AICPU_ASSERT_OK_RETVAL(
        argsHandle_->UpdateIoAddr(inputs_, outputs_, stream, executor, deviceExtMemSize_, deviceCacheOffset_));
    RecordAicpuTime(kUpdateArgsEnd);
    {
        OpDfxGuard kernelLaunchGuard(summaryItemId_, DfxProfilingKernelLaunch);
        if (useNewLaunchInterface_) {
            AICPU_ASSERT_OK_RETVAL(LaunchKernelByNewInterface(stream));
        } else {
            OP_LOGI("Compatible with the old process.");
            const auto& args = argsHandle_->GetArgsEx();
            rtAicpuArgsEx_t rtsLaunchArgs = {};
            rtsLaunchArgs.args = args.args;
            rtsLaunchArgs.argsSize = args.argsSize;
            std::vector<rtHostInputInfo_t> hostInputs;
            for (size_t i = 0; i < args.hostInputInfoNum; ++i) {
                rtHostInputInfo_t hostInput = {};
                hostInput.addrOffset = args.hostInputInfoPtr[i].addrOffset;
                hostInput.dataOffset = args.hostInputInfoPtr[i].dataOffset;
                hostInputs.emplace_back(hostInput);
            }
            rtsLaunchArgs.hostInputInfoPtr = hostInputs.data();
            rtsLaunchArgs.hostInputInfoNum = args.hostInputInfoNum;
            rtsLaunchArgs.soNameAddrOffset = args.soNameAddrOffset;
            rtsLaunchArgs.kernelNameAddrOffset = args.kernelNameAddrOffset;

            AICPU_ASSERT_RTOK_RETVAL(rtAicpuKernelLaunchExWithArgs(
                KERNEL_TYPE_AICPU, opType_.c_str(), 1U, &rtsLaunchArgs, nullptr, stream, RT_KERNEL_DEFAULT_IN_AICPU));
        }
    }
    RecordAicpuTime(kLaunchEnd);

    RecordAicpuTime(kShapeD2hCopyEnd);
    if (unknownType_ == ge::DEPEND_SHAPE_RANGE) {
        OP_LOGI("op [%s] is 3th op\n", opType_.c_str());
        AICPU_ASSERT_OK_RETVAL(extInfoHandle_->UpdateOutputShapeFromExtInfo(outputs_, stream));
    }
    OP_LOGI("Finish AicpuCCTask::Run, opType[%s]", opType_.c_str());
    RecordAicpuTime(kUpdateOutputShapeEnd);

    if (op::internal::opProfilingSwitch.kernelLaunchFlag && op::internal::opProfilingSwitch.additionInfoFlag) {
        GetThreadLocalContext().profilingInfoId_.kernelLauncherId_ = launchId_; // optype
        op::FVector<const aclTensor*> input;
        op::FVector<const aclTensor*> out;
        GetValidInputVec(inputs_, input);
        GetConstVec(outputs_, out);
        // summaryItemId for op name
        op::internal::ReportAdditionInfo(input, out, MSPROF_GE_TASK_TYPE_AI_CPU, summaryItemId_);
    }

    op::internal::PrintAicpuAllTimeStampInfo(opType_.c_str());
    return OK;
}

aclnnStatus CreatAicpuKernelLauncher(uint32_t opType, op::internal::AicpuTaskSpace& space, aclOpExecutor* executor,
                                     const FVector<std::string>& attrNames, op::OpArgContext* args)
{
    CHECK_RET(args != nullptr, ACLNN_ERR_PARAM_INVALID);
    auto task = space.GetOrCreateTask(executor, attrNames, args);
    CHECK_RET(task != nullptr, ACLNN_ERR_INNER);
    op::internal::ProfilingInfoId profilingInfoId;
    auto overflowTask = TakePendingOverflowTask(task);
    op::KernelLauncher* launcher = nullptr;
    if (overflowTask != nullptr) {
        executor->AbandonCache(true);
        launcher = new AiCpuOneShotKernelLauncher{
            opType, op::AI_CPU, executor, profilingInfoId, std::move(overflowTask), args};
    } else {
        executor->AbandonCache();
        launcher = new op::internal::AiCpuKernelLauncher{opType, op::AI_CPU, executor, profilingInfoId, task, args};
    }
    executor->AddToKernelLauncherList(launcher);
    op::internal::BuildGraph(executor->GetGraph(), opType, *args->GetOpArg(op::OP_INPUT_ARG),
                             *args->GetOpArg(op::OP_OUTPUT_ARG), *args->GetOpArg(op::OP_WORKSPACE_ARG));
    return ACLNN_SUCCESS;
}
} // namespace internal
} // namespace op
