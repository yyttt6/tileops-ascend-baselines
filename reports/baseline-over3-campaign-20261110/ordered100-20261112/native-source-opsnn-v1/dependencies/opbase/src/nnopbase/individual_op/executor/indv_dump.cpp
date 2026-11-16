/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <set>
#include <iostream>
#include <iomanip>
#include <unordered_map>
#include "indv_executor.h"
#include "indv_non_finite_check_op.h"
#include "utils/indv_soc.h"
#include "opdev/fp16_t.h"
#include "graph/types.h"
#include "opdev/small_vector.h"
#include "op_dfx_util.h"
#include "opdev/op_dfx.h"
#include "bridge_dfx.h"
#include "acl/acl_rt.h"
#include "runtime/runtime/rts/rts_kernel.h"
#include "dump/adump_api.h"

using namespace op::internal;
namespace {
constexpr uint32_t NNOPBASE_EXCEPTION_DUMP_HEAD = 2U;
constexpr uint16_t NNOPBASE_MODEL_ID = 61; // OP model

namespace {
void FillTensorInfo(Adx::TensorInfoV2& info, gert::TensorV2* rtTensor)
{
    info.tensorSize = static_cast<size_t>(
        op::CalcShapeBytes(rtTensor->GetStorageShape().GetShapeSize(), rtTensor->GetDataType()));
    info.format = rtTensor->GetStorageFormat();
    info.dataType = rtTensor->GetDataType();
    info.tensorAddr = static_cast<int64_t*>(rtTensor->GetAddr());
    info.placement = rtTensor->GetPlacement();
    const auto& storageShape = rtTensor->GetStorageShape();
    for (size_t i = 0U; i < storageShape.GetDimNum(); ++i) {
        info.shape.push_back(storageShape.GetDim(i));
    }
    const auto& originShape = rtTensor->GetOriginShape();
    for (size_t i = 0U; i < originShape.GetDimNum(); ++i) {
        info.originShape.push_back(originShape.GetDim(i));
    }
}
} // namespace
template <typename T>
std::string NnopbaseToHex(T num)
{
    std::stringstream stream;
    stream << std::hex << num;
    return stream.str();
}

void NnopbaseGetKernelInfoFromBin(NnopbaseBinInfo* binInfo, std::string& kernelInfo, std::string& devFunc)
{
    std::vector<std::string> parts;
    NnopbaseSplitStr(binInfo->binPath, "/", parts);
    if (parts.size() == 0U) {
        return;
    }
    kernelInfo = parts.back();
    const auto pos = kernelInfo.find(".o");
    NNOPBASE_ASSERT(pos != std::string::npos);
    devFunc = kernelInfo.substr(0, pos);
}

void NnopbaseGetTilingDataAndKey(NnopbaseExecutor* const executor, uint64_t& tilingKey, std::string& tilingData)
{
    if (executor->hasTiling != 0) {
        auto& tilingInfo = executor->args->tilingInfo;
        tilingKey = tilingInfo.tilingKey;
        auto tiling = op::internal::PtrCastTo<NnopbaseTilingData>(tilingInfo.tilingData);
        std::stringstream tilingDataStr;
        NnopbasePrintHex(op::internal::PtrCastTo<uint8_t>(tiling->GetData()), tiling->GetDataSize(), tilingDataStr);
        tilingData = tilingDataStr.str();
    }
}

void NnopbaseGetWorkspaceInfo(NnopbaseExecutor* const executor, std::string& workspaceBytes,
                              std::string& workspaceAddrs)
{
    const auto workspacesSizes = NnopbaseGetWorkspacesSizesFromArgs(executor->args);
    std::stringstream workspaceAddrStr;
    for (size_t i = 0U; i < workspacesSizes->GetSize(); i++) {
        workspaceBytes += std::to_string(workspacesSizes->GetData()[i]);
        workspaceBytes += " ";
        workspaceAddrStr << std::hex << executor->workspaces.workspaces[i] << " ";
    }
    workspaceAddrs = workspaceAddrStr.str();
}

void NnopbasePrepareDumpTensor(NnopbaseTensors& tensors, const Adx::TensorType ioType, const NnopbaseChar* const opType,
                               std::vector<Adx::TensorInfoV2>& dumpTensors)
{
    for (size_t i = 0U; i < tensors.num; i++) {
        OP_LOGI("Start to parse op %s index %zu to dump.", opType, i);
        auto& tensor = tensors.extTensors[i];
        if (tensor.isNull) {
            OP_LOGD("Op %s optional input %zu is null.", opType, i);
            continue;
        }
        Adx::TensorInfoV2 info;
        FillTensorInfo(info, &(tensor.rt2Tensor));
        info.type = ioType;
        info.addrType = Adx::AddressType::TRADITIONAL;
        info.argsOffSet = tensor.argsOffset / static_cast<uint32_t>(sizeof(void*));
        OP_LOGI("IO[%zu] size is %zu bytes, argsOffSet is %u", i, info.tensorSize, info.argsOffSet);
        dumpTensors.push_back(info);
    }
}

inline void NnopbasePrepareOutputShapeDumpTensor(NnopbaseExecutor* const executor, gert::TensorV2* outputShapeTensor,
                                                 std::vector<Adx::TensorInfoV2>& dumpTensors)
{
    Adx::TensorInfoV2 info;
    *outputShapeTensor = {
        {{9}, {9}},                         // shape
        {ge::FORMAT_ND, ge::FORMAT_ND, {}}, // format
        gert::kOnDeviceHbm,                 // placement
        ge::DT_INT64,                       // data type
        op::internal::PtrCastTo<void>(op::internal::PtrCastTo<uint8_t>(executor->workspaces.workspaces[0]) +
                                      (executor->workspaces.length - executor->args->outputs.outPutShapeSize))};
    FillTensorInfo(info, outputShapeTensor);
    info.type = Adx::TensorType::OUTPUT;
    info.addrType = Adx::AddressType::TRADITIONAL;
    info.argsOffSet = executor->args->outputs.outPutShapeArgsOffset;
    dumpTensors.push_back(info);
}

inline void NnopbaseDumpTensors(NnopbaseExecutor* executor, Adx::TensorType ioType, aclrtStream stream,
                                NnopbaseChar* opType)
{
    auto& tensors = ioType == Adx::TensorType::INPUT ? executor->args->inputs : executor->args->outputs;
    OP_LOGI("Start to dump, op type %s.", opType);
    std::vector<Adx::TensorInfoV2> dumpTensors;
    NnopbasePrepareDumpTensor(tensors, ioType, opType, dumpTensors);
    gert::TensorV2 outputShapeTensor;
    if ((ioType == Adx::TensorType::OUTPUT) && (tensors.outPutShapeArgsOffset != 0U)) {
        NnopbasePrepareOutputShapeDumpTensor(executor, &outputShapeTensor, dumpTensors);
    }

    std::string l2Name = std::string("L2DfxAbscent_") + std::to_string(op::internal::OpGetLogSequence());
    std::string l0Name = std::string("_L0") + std::string(opType);
    const auto res = Adx::AdumpDumpTensorV2(l2Name, l0Name, dumpTensors, stream);
    OP_LOGI("AdumpDumpTensorV2 res = %d.", res);
}

aclnnStatus NnopbaseIsSaturationOverflow(const bool is910b, aclrtStream stream, bool& isOverflow)
{
    void* dst = nullptr;
    constexpr uint64_t dstLen = 64; // 溢出检测大小固定为64字节
    const NnopbaseGuard guard([&dst]() {
        if (dst != nullptr) {
            aclrtFree(dst);
            dst = nullptr;
        }
    });
    aclrtMallocAttrValue moduleIdValue;
    moduleIdValue.moduleId = NNOPBASE_MODEL_ID;
    aclrtMallocAttribute attrs{.attr = ACL_RT_MEM_ATTR_MODULE_ID, .value = moduleIdValue};
    aclrtMallocConfig cfg{.attrs = &attrs, .numAttrs = 1};
    if (is910b) {
        // 1971读取的二级指针，需要多申请一块内存，后续把实际的地址刷过来
        NNOPBASE_ASSERT_RTOK_RETVAL(aclrtMallocWithCfg(&dst, dstLen * 2U, ACL_MEM_TYPE_HIGH_BAND_WIDTH, &cfg));
    } else {
        NNOPBASE_ASSERT_RTOK_RETVAL(aclrtMallocWithCfg(&dst, dstLen, ACL_MEM_TYPE_HIGH_BAND_WIDTH, &cfg));
    }

    uint64_t realAddr = PtrToValue(dst);
    if (is910b) {
        realAddr += dstLen;
        NNOPBASE_ASSERT_RTOK_RETVAL(
            aclrtMemcpy(dst, sizeof(uint64_t), &realAddr, sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE));
    }
    constexpr uint32_t checkMode = 0U;
    NNOPBASE_ASSERT_RTOK_RETVAL(rtsNpuGetFloatOverFlowDebugStatus(dst, dstLen, checkMode, stream));
    uint8_t status = 0U;
    NNOPBASE_ASSERT_RTOK_RETVAL(aclrtSynchronizeStream(stream));
    NNOPBASE_ASSERT_RTOK_RETVAL(
        aclrtMemcpy(&status, sizeof(uint8_t), ValueToPtr(realAddr), sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST));

    OP_LOGI("Get rtsNpuGetFloatOverFlowDebugStatus check result is %u.", status);
    NNOPBASE_ASSERT_RTOK_RETVAL(rtsNpuClearFloatOverFlowDebugStatus(checkMode, stream));
    isOverflow = (status != 0U);
    return OK;
}

// 维测能力，若中间发生错误不终止流程
void NnopbaseCheckOverflowAndDump(NnopbaseExecutor* const executor, aclrtStream stream)
{
    aclrtFloatOverflowMode overflowMode = ACL_RT_OVERFLOW_MODE_SATURATION;
    NNOPBASE_ASSERT_RTOK(aclrtGetDeviceSatMode(&overflowMode));
    NNOPBASE_ASSERT_TRUE((overflowMode == ACL_RT_OVERFLOW_MODE_SATURATION) ||
                         (overflowMode == ACL_RT_OVERFLOW_MODE_INFNAN));
    OP_LOGD("Start to test op %s overflow, mode is %d.", executor->opType, overflowMode);

    bool isOverflow = false;
    if (overflowMode == ACL_RT_OVERFLOW_MODE_SATURATION) {
        NNOPBASE_ASSERT_OK(NnopbaseIsSaturationOverflow(
            nnopbase::IndvSoc::GetInstance().NeedsExtraMemoryForOverflowDump(), stream, isOverflow));
    } else {
        NNOPBASE_ASSERT_OK(NnopbaseRunNonFiniteCheckOp(executor->args->outputs, stream, isOverflow));
        // 上述接口会修改日志打印里面的optype
        GetThreadLocalContext().logInfo_.l2ApiName = executor->opType;
    }

    OP_LOGD("Op %s overflow test result is %d.", executor->opType, isOverflow);
    if (isOverflow) {
        NnopbaseDumpTensors(executor, Adx::TensorType::INPUT, stream, executor->opType);
        NnopbaseDumpTensors(executor, Adx::TensorType::OUTPUT, stream, executor->opType);
    }
}
} // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus NnopbaseDumpWorkspaceData(NnopbaseExecutor* executor, aclrtStream stream)
{
    if (NnopbaseSkipKernelLaunch(executor)) {
        return OK;
    }
    if (executor->args->binInfo->dfxInfo.isPrintEnable || executor->args->binInfo->dfxInfo.isTimeStampEnable) {
        OP_LOGI("Start to dump workspace info, op type %s.", executor->opType);
        Adx::AdumpPrintConfig printConfig = {op::internal::opProfilingSwitch.timeStampFlag};
        Adx::AdumpPrintSetConfig(printConfig);
        const auto addrBase = executor->workspaces.workspaces[0]; // 开dump时一定有workspace
        if (op::internal::opProfilingSwitch.timeStampFlag && executor->args->binInfo->dfxInfo.isTimeStampEnable) {
            std::vector<MsprofAicTimeStampInfo> timeStampInfo;
            Adx::AdumpPrintAndGetTimeStampInfo(addrBase, executor->args->binInfo->debugBufSize, stream,
                                               executor->opType, timeStampInfo);
            NnopbaseReportTimeStampInfo(timeStampInfo);
        } else {
            Adx::AdumpPrintWorkSpace(addrBase, executor->args->binInfo->debugBufSize, stream, executor->opType);
        }
    }
    return OK;
}

void NnopbaseDumpData(NnopbaseExecutor* executor, Adx::TensorType ioType, aclrtStream stream, NnopbaseChar* opType)
{
    if (Adx::AdumpGetDumpSwitch(Adx::DumpType::OPERATOR) != 0) {
        NnopbaseDumpTensors(executor, ioType, stream, opType);
    }
}

aclnnStatus NnopbasePrepareWorkspaceDumpTensor(NnopbaseExecutor* executor, std::vector<Adx::TensorInfoV2>& dumpTensors)
{
    const auto workspacesSizes = NnopbaseGetWorkspacesSizesFromArgs(executor->args);
    executor->workspaces.workspaceTensor = std::vector<gert::TensorV2>(workspacesSizes->GetSize());
    for (size_t i = 0U; i < workspacesSizes->GetSize(); i++) {
        executor->workspaces.workspaceTensor[i].MutableOriginShape() = {
            static_cast<int64_t>(workspacesSizes->GetData()[i])};
        executor->workspaces.workspaceTensor[i].MutableStorageShape() = {
            static_cast<int64_t>(workspacesSizes->GetData()[i])};
        executor->workspaces.workspaceTensor[i].SetDataType(ge::DT_UINT8);
        executor->workspaces.workspaceTensor[i].SetOriginFormat(ge::FORMAT_ND);
        executor->workspaces.workspaceTensor[i].SetStorageFormat(ge::FORMAT_ND);
        executor->workspaces.workspaceTensor[i].SetPlacement(gert::kOnDeviceHbm);
        executor->workspaces.workspaceTensor[i].SetSize(static_cast<int64_t>(workspacesSizes->GetData()[i]));
        NNOPBASE_ASSERT_TRUE_RETVAL(executor->workspaces.workspaceTensor[i].MutableTensorData().SetAddr(
                                        executor->workspaces.workspaces[i], nullptr) == ge::GRAPH_SUCCESS);
        Adx::TensorInfoV2 info;
        FillTensorInfo(info, &executor->workspaces.workspaceTensor[i]);
        info.type = Adx::TensorType::WORKSPACE;
        info.addrType = Adx::AddressType::TRADITIONAL;
        info.argsOffSet = executor->workspaces.workspaceArgsOffset[i] / static_cast<uint32_t>(sizeof(void*));
        OP_LOGI("Workspace[%zu] size is %zu bytes, argsOffSet is %u.", i, info.tensorSize, info.argsOffSet);
        dumpTensors.push_back(info);
    }

    return OK;
}

aclnnStatus NnopbasePrepareExceptionDumpInfo(NnopbaseExecutor* const executor, aclrtStream stream)
{
    if (NnopbaseIsExceptionDumpEnable() && !NnopbaseSkipKernelLaunch(executor)) {
        int32_t deviceId = 0;
        uint32_t taskId = 0U;
        int32_t streamId = 0U;
        NNOPBASE_ASSERT_RTOK_RETVAL(aclrtGetThreadLastTaskId(&taskId));
        NNOPBASE_ASSERT_RTOK_RETVAL(aclrtStreamGetId(stream, &streamId));
        NNOPBASE_ASSERT_RTOK_RETVAL(aclrtGetDevice(&deviceId));

        std::vector<Adx::TensorInfoV2> dumpInTensors;
        std::vector<Adx::TensorInfoV2> dumpOutTensors;
        NnopbasePrepareDumpTensor(executor->args->inputs, Adx::TensorType::INPUT, executor->opType, dumpInTensors);
        NnopbasePrepareDumpTensor(executor->args->outputs, Adx::TensorType::OUTPUT, executor->opType, dumpOutTensors);
        gert::TensorV2 outputShapeTensor;
        if (executor->args->outputs.outPutShapeArgsOffset != 0U) {
            NnopbasePrepareOutputShapeDumpTensor(executor, &outputShapeTensor, dumpOutTensors);
        }
        std::vector<Adx::TensorInfoV2> dumpWorkSpaceTensors;
        NNOPBASE_ASSERT_OK_RETVAL(NnopbasePrepareWorkspaceDumpTensor(executor, dumpWorkSpaceTensors));

        const std::string l2Name(executor->opType);
        const std::string l0Name(executor->opType);
        const uint32_t numBlocks = NnopbaseExecutorGetNumBlocks(executor);
        const std::string& magic = nnopbase::GetMagicFormBin(executor->collector->useCoreTypeMagic,
                                                             executor->args->binInfo);
        std::string devFunc;
        std::string kernelInfo;
        NnopbaseGetKernelInfoFromBin(executor->args->binInfo, kernelInfo, devFunc);
        uint64_t tilingKey = 0U;
        std::string tilingData;
        NnopbaseGetTilingDataAndKey(executor, tilingKey, tilingData);
        std::string workspaceBytes;
        std::string workspaceAddrs;
        NnopbaseGetWorkspaceInfo(executor, workspaceBytes, workspaceAddrs);
        const std::string& allAttrs = nnopbase::ToStr(executor->attrs);

        Adx::OperatorInfoV2 opInfo = op::internal::OperatorInfoBuilder(l2Name, l0Name)
                                         .Task(deviceId, taskId, static_cast<uint32_t>(streamId))
                                         .TensorInfo(dumpInTensors)
                                         .TensorInfo(dumpOutTensors)
                                         .TensorInfo(dumpWorkSpaceTensors)
                                         .AdditionInfo("block_dim", std::to_string(numBlocks))
                                         .AdditionInfo("workspace_bytes", workspaceBytes)
                                         .AdditionInfo("workspace_addrs", workspaceAddrs)
                                         .AdditionInfo("all_attrs", allAttrs)
                                         .AdditionInfo("dev_func", devFunc)
                                         .AdditionInfo("tvm_magic", magic)
                                         .AdditionInfo("kernel_info", kernelInfo)
                                         .AdditionInfo("tiling_key", std::to_string(tilingKey))
                                         .AdditionInfo("tiling_data", tilingData)
                                         .DeviceInfo("args before execute", executor->argsExt.args,
                                                     executor->argsExt.argsSize)
                                         .Build();
        auto res = AdumpAddExceptionOperatorInfoV2(opInfo);
        OP_LOGI("AdumpAddExceptionOperatorInfo L2 = %s, L0 = %s, device_id = %d, task_id = %u, stream_id = %u, "
                "res = %d.",
                l2Name.c_str(), l0Name.c_str(), deviceId, taskId, streamId, res);
    }
    return OK;
}

aclnnStatus NnopbaseOverflowDump(NnopbaseExecutor* const executor, aclrtStream stream)
{
    // NonFiniteCheck算子进来无需检测溢出，防止无限递归调用
    if ((Adx::AdumpGetDumpSwitch(Adx::DumpType::OP_OVERFLOW) != 0) &&
        (std::string(executor->opType) != "NonFiniteCheck")) {
        aclmdlRICaptureStatus status;
        aclmdlRI captureMdl;
        if ((aclmdlRICaptureGetInfo(stream, &status, &captureMdl) == ACL_SUCCESS) &&
            (status == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE)) {
            OP_LOGI("Skip stream synchronization during capture phase.");
            return OK;
        }
        NNOPBASE_ASSERT_RTOK_RETVAL(aclrtSynchronizeStream(stream));
        NnopbaseCheckOverflowAndDump(executor, stream);
    }
    return OK;
}

static uint64_t* NnopbaseAddIoArgsExceptionDumpAddr(const NnopbaseTensors& tensors, uint64_t* sizeInfoAddr)
{
    size_t j = 0U;
    for (size_t i = 0U; i < tensors.paramDescs.count; i++) {
        if (!tensors.paramDescs.instances[i].isDynamic) {
            if (tensors.extTensors[j].isNull) {
                *sizeInfoAddr = 0U;
            } else {
                *sizeInfoAddr = tensors.extTensors[j].rt2Tensor.GetSize();
            }
            sizeInfoAddr++;
            j += 1U;
        } else {
            // 高8位为2，表示带shape的动态输入，不存在其他场景动态输入，低56位设置动态输入的个数
            constexpr uint64_t dynamicMode = (2ULL << 56ULL);
            *sizeInfoAddr = dynamicMode | static_cast<uint64_t>(tensors.paramDescs.instances[i].num);
            sizeInfoAddr++;
            for (uint32_t k = 0U; k < tensors.paramDescs.instances[i].num; k++) {
                *sizeInfoAddr = tensors.extTensors[j + k].rt2Tensor.GetSize();
                sizeInfoAddr++;
            }
            j += static_cast<size_t>(tensors.paramDescs.instances[i].num);
        }
    }
    return sizeInfoAddr;
}

static uint64_t* NnopbaseAddWorkspaceExceptionInfo(NnopbaseExecutor* const executor, uint64_t* sizeInfoAddr)
{
    const auto workspacesSizes = NnopbaseGetWorkspacesSizesFromArgs(executor->args);

    if (executor->workspaces.num > 0) {
        *sizeInfoAddr = workspacesSizes->GetData()[0] + executor->args->binInfo->debugBufSize;
        if (executor->args->binInfo->dfxInfo.isAssertEnable) {
            // 开启assert开关时，workspaceSize的最高8bit设置为4 低56位设置worksapceSize大小
            constexpr uint64_t workspaceOffset = (4ULL << 56ULL);
            *sizeInfoAddr |= workspaceOffset;
            OP_LOGD("WorkspaceSize is %lu bytes, set high 8 bit is %lu.", workspacesSizes->GetData()[0], *sizeInfoAddr);
        }
        sizeInfoAddr++;
    }

    for (size_t i = 1U; i < executor->workspaces.num; i++) {
        *sizeInfoAddr = workspacesSizes->GetData()[i];
        sizeInfoAddr++;
    }
    return sizeInfoAddr;
}

static bool NnopbaseNeedL0ExceptionDump(void)
{
    static const bool ret = (nnopbase::IndvSoc::GetInstance().SupportL0ExceptionDump() &&
                             op::internal::IsArgExceptionDumpEnable());
    return ret;
}

aclnnStatus NnopbaseArgsExceptionDumpAddr(NnopbaseExecutor* const executor)
{
    if (executor->args->binInfo->dfxInfo.isAssertEnable || NnopbaseNeedL0ExceptionDump()) {
        OP_LOGI("ArgExceptionDump is enabled.");
        uint32_t atomicIndex = 0U;
        size_t mc2ContextNum = 0U;
        if (executor->mc2.enabled) {
            mc2ContextNum = executor->mc2.contextAddrs.size();
        }
        const uint32_t addrNum = executor->args->inputs.num + executor->args->inputs.dynamicNum +
                                 executor->args->outputs.num + executor->args->outputs.dynamicNum +
                                 static_cast<uint32_t>(executor->workspaces.num) + mc2ContextNum;
        void* exceptionDumpAddr = Adx::AdumpGetSizeInfoAddr(addrNum + NNOPBASE_EXCEPTION_DUMP_HEAD, atomicIndex);
        OP_LOGD("AdumpGetSizeInfoAddr is %p.", exceptionDumpAddr);
        NNOPBASE_ASSERT_NOTNULL_RETVAL(exceptionDumpAddr);

        // atmoic index
        uint64_t* sizeInfoAddr = static_cast<uint64_t*>(exceptionDumpAddr);
        *sizeInfoAddr = static_cast<uint64_t>(atomicIndex);
        sizeInfoAddr++;
        bool hasCtrlAddr = false;
        // 只有910B/910_93需要设置L2Ctrl/Ffts标志位
        if (nnopbase::IndvSoc::GetInstance().UseCoreTypeMagic() && (executor->args->binInfo->coreType == kMix)) {
            hasCtrlAddr = true;
        }
        // 低32位表示inputs/outputs/workspaces nums
        *sizeInfoAddr = static_cast<uint64_t>(addrNum);
        if (hasCtrlAddr) {
            // 高32表示首个输入是否是CtrlAddr(mix is 1, other is 0)
            constexpr uint64_t inputOffset = (1ULL << 32ULL);
            *sizeInfoAddr |= inputOffset;
        }
        sizeInfoAddr++;
        if (executor->mc2.enabled) {
            for (size_t i = 0U; i < executor->mc2.contextAddrs.size(); i++) {
                *sizeInfoAddr = 32ULL; // 只打印context结构体的前32字节
                sizeInfoAddr++;
            }
        }
        // inputs/outputs/workspaces byte sizes
        sizeInfoAddr = NnopbaseAddIoArgsExceptionDumpAddr(executor->args->inputs, sizeInfoAddr);
        sizeInfoAddr = NnopbaseAddIoArgsExceptionDumpAddr(executor->args->outputs, sizeInfoAddr);
        if (executor->args->outputs.outPutShapeSize != 0U) {
            *sizeInfoAddr = executor->args->outputs.outPutShapeSize;
            sizeInfoAddr++;
        }

        sizeInfoAddr = NnopbaseAddWorkspaceExceptionInfo(executor, sizeInfoAddr);

        const rtArgsSizeInfo sizeInfo = {exceptionDumpAddr, atomicIndex};
        NNOPBASE_ASSERT_RTOK_RETVAL(rtSetExceptionExtInfo(&sizeInfo));
    }
    return OK;
}

#ifdef __cplusplus
}
#endif
