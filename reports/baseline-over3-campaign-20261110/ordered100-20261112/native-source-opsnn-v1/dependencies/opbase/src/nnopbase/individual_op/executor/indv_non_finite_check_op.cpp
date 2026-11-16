/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "indv_non_finite_check_op.h"
#include "indv_executor.h"
#include "indv_collector.h"
#include "utils/indv_soc.h"
#include "individual_op_api.h"

namespace {
constexpr const char* opType = "NonFiniteCheck";
constexpr uint16_t kModelId = 61; // OP model
constexpr float f05 = 0.5F;

constexpr ge::DataType supportDtypes[] = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
uint32_t socSupportListLen = 0;
uint32_t* socSupportList = nnopbase::IndvSoc::GetInstance().GetNonFiniteCheckSocSupportList(socSupportListLen);

TensorDesc inputDesc0_0[] = {{ge::DT_FLOAT16, ge::FORMAT_ND}};
TensorDesc inputDesc0_1[] = {{ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc0_2[] = {{ge::DT_BF16, ge::FORMAT_ND}};
TensorDesc outputDesc0_0[] = {{ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc0_1[] = {{ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc0_2[] = {{ge::DT_FLOAT, ge::FORMAT_ND}};

constexpr SupportInfo list0_0 = {inputDesc0_0, 1, outputDesc0_0, 1};
constexpr SupportInfo list0_1 = {inputDesc0_1, 1, outputDesc0_1, 1};
constexpr SupportInfo list0_2 = {inputDesc0_2, 1, outputDesc0_2, 1};
SupportInfo supportInfo0[] = {list0_0, list0_1, list0_2};
constexpr uint32_t supportInfoLen = sizeof(supportInfo0) / sizeof(SupportInfo);
constexpr OpSocSupportInfo socSupportInfo0 = {supportInfo0, supportInfoLen};

// 两种芯片的支持类型相同
OpSocSupportInfo opSocSupportList[] = {socSupportInfo0, socSupportInfo0};
OpSupportList supportList = {opSocSupportList, socSupportListLen};

aclnnStatus NnopbaseCopyTensor(const GertTensor& src, NnopbaseTensor& dst)
{
    dst.isNull = false;
    auto& rt2Tensor = dst.rt2Tensor;
    rt2Tensor.GetShape() = src.GetShape();
    rt2Tensor.SetDataType(src.GetDataType());
    rt2Tensor.MutableFormat() = src.GetFormat();
    rt2Tensor.MutableTensorData().SetPlacement(src.GetPlacement());
    rt2Tensor.SetSize(src.GetSize());
    NNOPBASE_ASSERT_OK_RETVAL(rt2Tensor.MutableTensorData().SetAddr(src.GetTensorData().GetAddr(), nullptr));
    rt2Tensor.MutableStride() = src.GetStride();
    rt2Tensor.SetOffset(src.GetOffset());
    return OK;
}

aclnnStatus NnopbaseNonFiniteCheckAddInputs(const std::vector<const GertTensor*>& inputs, NnopbaseExecutor* executor)
{
    // startIndex在初始化时已设置为0
    executor->ownArgs.inputs.paramDescs.instances[0U].isInput = true;
    executor->ownArgs.inputs.paramDescs.instances[0U].startIndex = 0U;
    executor->ownArgs.inputs.paramDescs.instances[0U].num = inputs.size();
    executor->ownArgs.inputs.num = inputs.size();
    executor->ownArgs.inputs.usedNum = inputs.size();
    executor->ownArgs.inputs.dynamicCnt = inputs.size();
    if (executor->ownArgs.inputs.num >= executor->ownArgs.inputs.arrayLen) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseExecutorExtendIoCaches(&executor->ownArgs.inputs));
    }

    for (uint64_t i = 0U; i < inputs.size(); i++) {
        NNOPBASE_ASSERT_NOTNULL_RETVAL(inputs[i]);
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseCopyTensor(*(inputs[i]), executor->ownArgs.inputs.extTensors[i]));
        executor->ownArgs.inputs.extTensors[i].isRequired = false;
    }
    executor->ownArgs.inputs.extTensors[0U].isRequired = true;
    return OK;
}

aclnnStatus NnopbaseNonFiniteCheckAddOutputs(NnopbaseExecutor* executor, void* const outputAddr)
{
    auto& outputs = executor->ownArgs.outputs;
    // num在初始化时设置过为1
    outputs.paramDescs.instances[0U].isInput = false;

    outputs.extTensors[0U].isNull = false;
    auto& rt2Tensor = outputs.extTensors[0U].rt2Tensor;
    rt2Tensor.MutableOriginShape() = {1};
    rt2Tensor.MutableStorageShape() = {1};
    rt2Tensor.SetDataType(ge::DataType::DT_FLOAT);
    rt2Tensor.SetOriginFormat(ge::FORMAT_ND);
    rt2Tensor.SetStorageFormat(ge::FORMAT_ND);
    rt2Tensor.MutableTensorData().SetPlacement(gert::kOnDeviceHbm);
    rt2Tensor.SetSize(sizeof(float));
    rt2Tensor.MutableStride().SetDimNum(0U);
    rt2Tensor.SetOffset(0);
    NNOPBASE_ASSERT_OK_RETVAL(rt2Tensor.MutableTensorData().SetAddr(outputAddr, nullptr));
    return OK;
}

// 出参和入参
aclnnStatus NnopbaseNonFiniteCheckRunForWorkspaceSize(const std::vector<const GertTensor*>& inputs,
                                                      void* const outputAddr, uint64_t* workspaceSize,
                                                      NnopbaseExecutor** executor)
{
    const uint64_t timeStamp = NnopbaseMsprofSysTime();
    static NnopbaseDfxId dfxId = {0x60000U, __func__, false};
    static NnopbaseDfxId tilingId = {0x60000U, "NonFiniteCheckTiling", false};

    // 创建space
    static void* space = nullptr;
    if (space == nullptr) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseCreateExecutorSpace(&space));
    }

    // 获取executor
    NnopbaseChar inputDesc[] = {2}; // 2表示动态输入
    NnopbaseChar outputDesc[] = {1};
    NnopbaseChar attrDesc[] = {};
    *executor = (NnopbaseExecutor*)NnopbaseGetExecutor(space, opType, inputDesc, 1U, outputDesc, 1U, attrDesc, 0U);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(*executor);

    // 添加输入输出
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseNonFiniteCheckAddInputs(inputs, *executor));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseNonFiniteCheckAddOutputs(*executor, outputAddr));

    // 添加维测信息
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddTilingId(*executor, &tilingId));
    NnopbaseExecutorAddSupportList((*executor)->opCompatibility, &supportList, socSupportList, socSupportListLen);

    // 执行一阶段流程
    const aclnnStatus ret = NnopbaseExecutorRunForWorkspace(*executor, workspaceSize);
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ret;
}

aclnnStatus NnopbaseNonFiniteCheckRunWithWorkspace(void* workspace, uint64_t workspaceSize, NnopbaseExecutor* executor,
                                                   aclrtStream stream)
{
    const uint64_t timeStamp = NnopbaseMsprofSysTime();
    static NnopbaseDfxId dfxId = {0x60000U, __func__, false};
    const aclnnStatus ret = NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceSize);
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ret;
}

aclnnStatus NnopbaseNonFiniteCheckOp(const std::vector<const GertTensor*>& inputs, aclrtStream stream, bool& isOverflow)
{
    void* workspace = nullptr;
    void* outputAddr = nullptr;
    aclrtMallocAttrValue moduleIdValue;
    moduleIdValue.moduleId = kModelId;
    aclrtMallocAttribute attrs{.attr = ACL_RT_MEM_ATTR_MODULE_ID, .value = moduleIdValue};
    aclrtMallocConfig cfg{.attrs = &attrs, .numAttrs = 1};
    NNOPBASE_ASSERT_RTOK_RETVAL(aclrtMallocWithCfg(&outputAddr, sizeof(float), ACL_MEM_TYPE_HIGH_BAND_WIDTH, &cfg));
    const NnopbaseGuard guard([&workspace, &outputAddr]() {
        if (workspace != nullptr) {
            (void)aclrtFree(workspace);
            workspace = nullptr;
        }
        if (outputAddr != nullptr) {
            (void)aclrtFree(outputAddr);
            outputAddr = nullptr;
        }
    });

    // 调用一、二段接口
    NnopbaseExecutor* executor = nullptr;
    uint64_t workspaceSize = 0U;
    NNOPBASE_ASSERT_RTOK_RETVAL(
        NnopbaseNonFiniteCheckRunForWorkspaceSize(inputs, outputAddr, &workspaceSize, &executor));
    if (workspaceSize > 0U) {
        NNOPBASE_ASSERT_RTOK_RETVAL(aclrtMallocWithCfg(&workspace, workspaceSize, ACL_MEM_TYPE_HIGH_BAND_WIDTH, &cfg));
    }
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseNonFiniteCheckRunWithWorkspace(workspace, workspaceSize, executor, stream));

    // 同步，并查看结果
    NNOPBASE_ASSERT_RTOK_RETVAL(aclrtSynchronizeStream(stream));
    float status = 0.0;
    NNOPBASE_ASSERT_RTOK_RETVAL(
        aclrtMemcpy(&status, sizeof(float), outputAddr, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));

    // 由于计算结果为float，为防止类型转换的精度问题，这里使用0.5作为阈值判别结果
    isOverflow = isOverflow || (status > f05);
    return OK;
}

std::vector<const GertTensor*> GetNonFiniteCheckInputTensors(const NnopbaseTensors& tensors, const ge::DataType dtype)
{
    std::vector<const GertTensor*> res;
    for (size_t i = 0U; i < tensors.num; i++) {
        const auto& tensor = tensors.extTensors[i];
        if (tensor.isNull || tensor.rt2Tensor.GetShapeSize() == 0) {
            OP_LOGD("OutputTensor[%zu] is nullptr or shape size is 0, skip checking.", i);
            continue;
        }
        if (tensor.rt2Tensor.GetDataType() == dtype) {
            (void)res.emplace_back(&tensor.rt2Tensor);
        }
    }
    return res;
}
} // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus NnopbaseRunNonFiniteCheckOp(const NnopbaseTensors& tensors, aclrtStream stream, bool& isOverflow)
{
    for (const auto type : supportDtypes) {
        // 由于NonFiniteCheck算子仅支持同样的类型输入，被检测算子输出可能有不同类型，将同样的类型分组分别送入NonFiniteCheck检测
        const auto& inputs = GetNonFiniteCheckInputTensors(tensors, type);
        if (!inputs.empty()) {
            NNOPBASE_ASSERT_OK_RETVAL(NnopbaseNonFiniteCheckOp(inputs, stream, isOverflow));
        }
    }
    return OK;
}

#ifdef __cplusplus
}
#endif
