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
#include "indv_tilingcontext_builder.h"
#include "op_dfx_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus NnopbaseSetRefTensorAddr(NnopbaseExecutor* executor, const size_t index, const void* const addr,
                                     const std::vector<NnopbaseParamInstance>& inInstances,
                                     const std::vector<NnopbaseParamInstance>& outInstances)
{
    if (inInstances.size() != 0U) {
        int64_t irIndex = inInstances.size() - 1;
        for (; irIndex >= 0; irIndex--) {
            if (inInstances[irIndex].startIndex <= index) {
                if (inInstances[irIndex].refIdx != -1) {
                    size_t offset = index - inInstances[irIndex].startIndex;
                    size_t outputIrIndex = inInstances[irIndex].refIdx;
                    size_t outputStartIndex = outInstances[outputIrIndex].startIndex;
                    CHECK_COND(NnopbaseSetOutputTensorAddr(executor, outputStartIndex + offset, addr) == OK,
                               ACLNN_ERR_PARAM_INVALID,
                               "Failed to set output address, outputStartIndex[%zu], offset[%zu].", outputStartIndex,
                               offset);
                }
                break;
            }
        }
    }
    return OK;
}

aclnnStatus NnopbaseUpdateDynamicTensors(NnopbaseTensors* dstTensors, NnopbaseTensors* tensors, uint32_t index)
{
    const uint32_t startIndex = dstTensors->paramDescs.instances[index].startIndex;
    const uint32_t dynamicNum = dstTensors->paramDescs.instances[index].num;
    const uint32_t tensorListSize = static_cast<uint32_t>(tensors->paramDescs.instances[index].tensorList->Size());
    CHECK_COND((dynamicNum == tensorListSize), ACLNN_ERR_PARAM_INVALID,
               "Failed to update dynamic tensor[%u], instance num is %u, tensorListSize is %u.", index, dynamicNum,
               tensorListSize);
    for (uint32_t j = 0U; j < dynamicNum; j++) {
        NNOPBASE_ASSERT_OK_RETVAL(dstTensors->extTensors[startIndex + j].rt2Tensor.MutableTensorData().SetAddr(
            (*tensors->paramDescs.instances[index].tensorList)[j]->GetData(), nullptr));
        OP_LOGI("Update dynamic tensor[%u] addr %p successfully.", startIndex + j,
                dstTensors->extTensors[startIndex + j].rt2Tensor.GetAddr());
    }
    return OK;
}

static aclnnStatus NnopbaseUpdateScalarAddr(NnopbaseExecutor* executor, const aclScalar* scalar, const uint32_t index)
{
    ge::DataType dataType = scalar->GetDataType();
    ge::DataType dtype = executor->args->inputs.paramDescs.instances[index].scalarDtype;
    const int32_t srcIndex = executor->args->inputs.paramDescs.instances[index].srcIndex;
    if (srcIndex != -1) {
        dataType = executor->args->inputs.extTensors[executor->args->inputs.paramDescs.instances[srcIndex].startIndex]
                       .rt2Tensor.GetDataType();
    } else if (dtype != ge::DT_UNDEFINED) {
        dataType = dtype;
    }

    auto rt2Tensor = &executor->args->inputs.extTensors[executor->args->inputs.paramDescs.instances[index].startIndex]
                          .rt2Tensor;
    NNOPBASE_ASSERT_NOTNULL_RETVAL(rt2Tensor);
    if (scalar->GetDataType() == ge::DataType::DT_DOUBLE) {
        dataType = ge::DataType::DT_FLOAT;
        executor->args->inputs.paramDescs.instances[index].scalarValue.resize(sizeof(float));
        *op::internal::PtrCastTo<float>(
            executor->args->inputs.paramDescs.instances[index].scalarValue.data()) = scalar->ToFloat();
        NNOPBASE_ASSERT_OK_RETVAL(rt2Tensor->MutableTensorData().SetAddr(
            executor->args->inputs.paramDescs.instances[index].scalarValue.data(), nullptr));
    } else if (scalar->GetDataType() != dataType) {
        executor->args->inputs.paramDescs.instances[index].scalarValue.resize(op::TypeSize(dataType));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseExecutorConvertScalarType(
            executor->args->inputs.paramDescs.instances[index].scalarValue, scalar, dataType, 0U));
        NNOPBASE_ASSERT_OK_RETVAL(rt2Tensor->MutableTensorData().SetAddr(
            executor->args->inputs.paramDescs.instances[index].scalarValue.data(), nullptr));
    } else {
        NNOPBASE_ASSERT_OK_RETVAL(rt2Tensor->MutableTensorData().SetAddr(scalar->GetData(), nullptr));
    }
    OP_LOGI("Update scalar[%u] addr successfully.", index);
    return OK;
}

static aclnnStatus NnopbaseUpdateScalarListAddr(NnopbaseExecutor* executor, const aclScalarList* scalarList,
                                                const uint32_t index)
{
    const int32_t srcIndex = executor->args->inputs.paramDescs.instances[index].srcIndex;
    ge::DataType dtype = executor->args->inputs.paramDescs.instances[index].scalarDtype;
    ge::DataType dataType = (*scalarList)[0]->GetDataType();
    if (srcIndex != -1) {
        dataType = executor->args->inputs.extTensors[executor->args->inputs.paramDescs.instances[srcIndex].startIndex]
                       .rt2Tensor.GetDataType();
    } else if (dtype != ge::DT_UNDEFINED) {
        dataType = dtype;
    }
    auto rt2Tensor = &executor->args->inputs.extTensors[executor->args->inputs.paramDescs.instances[index].startIndex]
                          .rt2Tensor;
    NNOPBASE_ASSERT_NOTNULL_RETVAL(rt2Tensor);

    const size_t elementSize = op::TypeSize(dataType);
    const size_t tensorSize = elementSize * static_cast<size_t>(scalarList->Size());
    size_t offset = 0U;
    for (uint64_t i = 0U; i < scalarList->Size(); i++) {
        NNOPBASE_ASSERT_NOTNULL_RETVAL((*scalarList)[i]);
        if ((*scalarList)[i]->GetDataType() == ge::DataType::DT_DOUBLE) {
            dataType = ge::DataType::DT_FLOAT;
            *op::internal::PtrCastTo<float>(executor->args->inputs.paramDescs.instances[index].scalarValue.data() +
                                            offset) = (*scalarList)[i]->ToFloat();
            offset += op::TypeSize(ge::DataType::DT_FLOAT);
        } else if ((*scalarList)[i]->GetDataType() != dataType) {
            NNOPBASE_ASSERT_OK_RETVAL(NnopbaseExecutorConvertScalarType(
                executor->args->inputs.paramDescs.instances[index].scalarValue, (*scalarList)[i], dataType, offset));
            offset += op::TypeSize(dataType);
        } else {
            NNOPBASE_ASSERT_TRUE_RETVAL(
                memcpy_s(executor->args->inputs.paramDescs.instances[index].scalarValue.data() + offset,
                         tensorSize - offset, (*scalarList)[i]->GetData(), elementSize) == EOK);
            offset += elementSize;
        }
    }
    NNOPBASE_ASSERT_OK_RETVAL(rt2Tensor->MutableTensorData().SetAddr(
        executor->args->inputs.paramDescs.instances[index].scalarValue.data(), nullptr));
    OP_LOGI("Update scalarList[%u] addr successfully.", index);
    return OK;
}

static aclnnStatus NnopbaseSetRt2Tensor(NnopbaseTensor* dstTensors, NnopbaseTensor* srcTensor)
{
    GertTensor* dstRt2Tensor = &dstTensors->rt2Tensor;
    GertTensor* srcRt2Tensor = &srcTensor->rt2Tensor;
    dstRt2Tensor->MutableOriginShape() = srcRt2Tensor->MutableOriginShape();
    dstRt2Tensor->MutableStorageShape() = srcRt2Tensor->MutableStorageShape();
    dstRt2Tensor->SetDataType(srcRt2Tensor->GetDataType());
    dstRt2Tensor->SetOriginFormat(srcRt2Tensor->GetOriginFormat());
    dstRt2Tensor->SetStorageFormat(srcRt2Tensor->GetStorageFormat());
    dstRt2Tensor->MutableTensorData().SetPlacement(srcRt2Tensor->MutableTensorData().GetPlacement());
    dstRt2Tensor->MutableTensorData().SetSize(srcRt2Tensor->MutableTensorData().GetSize());
    NNOPBASE_ASSERT_OK_RETVAL(dstRt2Tensor->MutableTensorData().SetAddr(srcRt2Tensor->GetAddr(), nullptr));
    dstRt2Tensor->MutableStride() = srcRt2Tensor->GetStride();
    dstRt2Tensor->SetOffset(srcRt2Tensor->GetOffset());
    return OK;
}

static void NnopbaseSaveParamDesc(NnopbaseTensors* dstTensors, NnopbaseTensors* tensors)
{
    dstTensors->paramDescs.emptyNum = tensors->paramDescs.emptyNum;
    dstTensors->paramDescs.activeInputCount = tensors->paramDescs.activeInputCount;
    dstTensors->paramDescs.count = tensors->paramDescs.count;
    dstTensors->paramDescs.instances = std::vector<NnopbaseParamInstance>(tensors->paramDescs.count);
    for (uint32_t i = 0U; i < tensors->paramDescs.count; i++) {
        dstTensors->paramDescs.instances[i].num = tensors->paramDescs.instances[i].num;
        dstTensors->paramDescs.instances[i].cfgNum = tensors->paramDescs.instances[i].cfgNum;
        dstTensors->paramDescs.instances[i].startIndex = tensors->paramDescs.instances[i].startIndex;
        dstTensors->paramDescs.instances[i].isInput = tensors->paramDescs.instances[i].isInput;
        dstTensors->paramDescs.instances[i].isDynamic = tensors->paramDescs.instances[i].isDynamic;
        dstTensors->paramDescs.instances[i].scalarValue = tensors->paramDescs.instances[i].scalarValue;
        dstTensors->paramDescs.instances[i].scalarIndex = tensors->paramDescs.instances[i].scalarIndex;
        dstTensors->paramDescs.instances[i].refIdx = tensors->paramDescs.instances[i].refIdx;
        dstTensors->paramDescs.instances[i].name = tensors->paramDescs.instances[i].name;
        dstTensors->paramDescs.instances[i].isDisable = tensors->paramDescs.instances[i].isDisable;
    }
}

aclnnStatus NnopbaseSaveCachedTensor(NnopbaseTensors* dstTensors, NnopbaseTensors* tensors, bool isInput)
{
    NnopbaseSaveParamDesc(dstTensors, tensors);
    dstTensors->usedNum = tensors->usedNum;
    dstTensors->arrayLen = tensors->arrayLen;
    dstTensors->requiredCnt = tensors->requiredCnt;
    dstTensors->nonDynamicCnt = tensors->nonDynamicCnt;
    dstTensors->num = tensors->num;
    dstTensors->hasDynamic = tensors->hasDynamic;
    dstTensors->hostInputNum = tensors->hostInputNum;
    dstTensors->expectIndex = tensors->expectIndex;
    dstTensors->hostInputSize = tensors->hostInputSize;
    dstTensors->dynamicNum = tensors->dynamicNum;
    dstTensors->dynamicCnt = tensors->dynamicCnt;
    dstTensors->extTensors.clear();
    dstTensors->extTensors = std::vector<NnopbaseTensor>(tensors->arrayLen);
    for (size_t i = 0U; i < tensors->num; i++) {
        if (!tensors->extTensors[i].isNull) {
            NNOPBASE_ASSERT_OK_RETVAL(NnopbaseSetRt2Tensor(&dstTensors->extTensors[i], &tensors->extTensors[i]));
        }
        dstTensors->extTensors[i].isNull = tensors->extTensors[i].isNull;
        dstTensors->extTensors[i].isRequired = tensors->extTensors[i].isRequired;
        dstTensors->extTensors[i].isOptional = tensors->extTensors[i].isOptional;
        dstTensors->extTensors[i].valueDepend = tensors->extTensors[i].valueDepend;
        dstTensors->extTensors[i].argsOffset = tensors->extTensors[i].argsOffset;
    }
    if (!isInput) {
        dstTensors->outPutShapeSize = tensors->outPutShapeSize;
        dstTensors->outPutShapeArgsOffset = tensors->outPutShapeArgsOffset;
        dstTensors->outPutShapeMap = tensors->outPutShapeMap;
    }
    return OK;
}

void NnopbaseSetCachedInfo(NnopbaseExecutor* executor)
{
    executor->workspaces.num = executor->args->workspaceNum;
    executor->workspaces.length = executor->args->workspaceLen;
    executor->isOutEmpty = executor->args->isOutEmpty;
    executor->hasMemset = executor->args->hasMemset;
}

aclnnStatus NnopbaseUpdateInputAddr(NnopbaseExecutor* executor)
{
    auto& dstTensors = executor->args->inputs;
    auto& tensors = executor->ownArgs.inputs;
    for (uint32_t i = 0U; i < tensors.paramDescs.count; i++) {
        if (tensors.paramDescs.instances[i].tensorList != nullptr) {
            NNOPBASE_ASSERT_OK_RETVAL(NnopbaseUpdateDynamicTensors(&dstTensors, &tensors, i));
        } else {
            const uint32_t startIndex = dstTensors.paramDescs.instances[i].startIndex;
            if (tensors.paramDescs.instances[i].tensor != nullptr) {
                NNOPBASE_ASSERT_OK_RETVAL(dstTensors.extTensors[startIndex].rt2Tensor.MutableTensorData().SetAddr(
                    tensors.paramDescs.instances[i].tensor->GetData(), nullptr));
            } else if (tensors.paramDescs.instances[i].intArray != nullptr) {
                NNOPBASE_ASSERT_OK_RETVAL(dstTensors.extTensors[startIndex].rt2Tensor.MutableTensorData().SetAddr(
                    tensors.paramDescs.instances[i].intArray->GetData(), nullptr));
            } else if (tensors.paramDescs.instances[i].boolArray != nullptr) {
                NNOPBASE_ASSERT_OK_RETVAL(dstTensors.extTensors[startIndex].rt2Tensor.MutableTensorData().SetAddr(
                    tensors.paramDescs.instances[i].boolArray->GetData(), nullptr));
            } else if (tensors.paramDescs.instances[i].floatArray != nullptr) {
                NNOPBASE_ASSERT_OK_RETVAL(dstTensors.extTensors[startIndex].rt2Tensor.MutableTensorData().SetAddr(
                    tensors.paramDescs.instances[i].floatArray->GetData(), nullptr));
            } else if (tensors.paramDescs.instances[i].scalar != nullptr) {
                NNOPBASE_ASSERT_OK_RETVAL(
                    NnopbaseUpdateScalarAddr(executor, tensors.paramDescs.instances[i].scalar, i));
            } else if (tensors.paramDescs.instances[i].scalarList != nullptr) {
                NNOPBASE_ASSERT_OK_RETVAL(
                    NnopbaseUpdateScalarListAddr(executor, tensors.paramDescs.instances[i].scalarList, i));
            }
            OP_LOGI("Update input tensor[%u] addr %p successfully.", startIndex,
                    dstTensors.extTensors[startIndex].rt2Tensor.GetAddr());
        }
    }
    return OK;
}

aclnnStatus NnopbaseUpdateOutputAddr(NnopbaseTensors* dstTensors, NnopbaseTensors* tensors)
{
    for (uint32_t i = 0U; i < tensors->paramDescs.count; i++) {
        if (tensors->paramDescs.instances[i].tensor != nullptr) {
            const uint32_t startIndex = dstTensors->paramDescs.instances[i].startIndex;
            NNOPBASE_ASSERT_OK_RETVAL(dstTensors->extTensors[startIndex].rt2Tensor.MutableTensorData().SetAddr(
                tensors->paramDescs.instances[i].tensor->GetData(), nullptr));
            OP_LOGI("Tensor[%u] update addr %p successfully", startIndex,
                    dstTensors->extTensors[startIndex].rt2Tensor.GetAddr());
        } else if (tensors->paramDescs.instances[i].tensorList != nullptr) {
            NNOPBASE_ASSERT_OK_RETVAL(NnopbaseUpdateDynamicTensors(dstTensors, tensors, i));
        }
    }
    return OK;
}

static aclnnStatus NnopbaseAddInputsTensors(NnopbaseExecutor* executor, NnopbaseTensors* tensors, uint32_t index)
{
    NnopbaseParamInstance& irInsts = tensors->paramDescs.instances[index];
    if (irInsts.tensorList != nullptr) {
        return NnopbaseExecutorAddDynamicTensors(executor, irInsts.tensorList, index, true, irInsts.ignoreCont);
    } else {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseExecutorUpdateTensorsIndex(tensors, index));
        if (irInsts.intArray != nullptr) {
            return nnopbase::NnopbaseExecutorAddArrayInput<aclIntArray>(tensors, irInsts.intArray, index);
        } else if (irInsts.boolArray != nullptr) {
            return nnopbase::NnopbaseExecutorAddArrayInput<aclBoolArray>(tensors, irInsts.boolArray, index);
        } else if (irInsts.floatArray != nullptr) {
            return nnopbase::NnopbaseExecutorAddArrayInput<aclFloatArray>(tensors, irInsts.floatArray, index);
        } else if (irInsts.scalar != nullptr) {
            return NnopbaseExecutorAddScalarInput(tensors, irInsts.scalar, index, irInsts.srcIndex,
                                                  irInsts.scalarDtype);
        } else if (irInsts.scalarList != nullptr) {
            return NnopbaseExecutorAddScalarListInput(tensors, irInsts.scalarList, index, irInsts.srcIndex,
                                                      irInsts.scalarDtype);
        } else if (irInsts.tensor != nullptr) {
            return NnopbaseExecutorAddTensor(executor, irInsts.tensor, index, true, irInsts.ignoreCont);
        } else {
            executor->ownArgs.inputs.paramDescs.instances[index].isInput = true;
            executor->ownArgs.inputs.paramDescs.instances[index].num = 0U;
        }
    }
    return OK;
}

static aclnnStatus NnopbaseAddOutputsTensors(NnopbaseExecutor* executor, NnopbaseTensors* tensors, uint32_t index)
{
    NnopbaseParamInstance& irInsts = tensors->paramDescs.instances[index];
    if (irInsts.tensorList != nullptr) {
        return NnopbaseExecutorAddDynamicTensors(executor, irInsts.tensorList, index, false);
    } else if (irInsts.tensor != nullptr) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseExecutorUpdateTensorsIndex(tensors, index));
        if (irInsts.tensor != nullptr) {
            return NnopbaseExecutorAddTensor(executor, irInsts.tensor, index, false, irInsts.ignoreCont);
        }
    } else {
        executor->ownArgs.outputs.paramDescs.instances[index].isInput = false;
        executor->ownArgs.outputs.paramDescs.instances[index].num = 0U;
    }
    return OK;
}

void NnopbaseClearParamInstance(NnopbaseTensors* tensors)
{
    for (uint32_t i = 0; i < tensors->paramDescs.count; i++) {
        tensors->paramDescs.instances[i].tensor = nullptr;
        tensors->paramDescs.instances[i].tensorList = nullptr;
        tensors->paramDescs.instances[i].intArray = nullptr;
        tensors->paramDescs.instances[i].boolArray = nullptr;
        tensors->paramDescs.instances[i].floatArray = nullptr;
        tensors->paramDescs.instances[i].scalar = nullptr;
        tensors->paramDescs.instances[i].scalarList = nullptr;
        tensors->paramDescs.instances[i].scalarDtype = ge::DT_UNDEFINED;
        tensors->paramDescs.instances[i].scalarIndex = -1;
        tensors->paramDescs.instances[i].ignoreCont = false;
    }
}

aclnnStatus NnopbaseAddIoTensors(NnopbaseExecutor* executor)
{
    for (uint32_t i = 0U; i < executor->ownArgs.inputs.paramDescs.count; i++) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInputsTensors(executor, &executor->ownArgs.inputs, i));
    }
    for (uint32_t i = 0U; i < executor->ownArgs.outputs.paramDescs.count; i++) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddOutputsTensors(executor, &executor->ownArgs.outputs, i));
    }
    return OK;
}

aclnnStatus UpdateArgsIoAddr(NnopbaseTensors* dstTensors, NnopbaseTensors* tensors)
{
    for (uint32_t i = 0; i < dstTensors->num; ++i) {
        NNOPBASE_ASSERT_OK_RETVAL(dstTensors->extTensors[i].rt2Tensor.MutableTensorData().SetAddr(
            tensors->extTensors[i].rt2Tensor.GetAddr(), nullptr));
        OP_LOGI("After update dst tensor[%u] addr is %p, src tensor addr is %p.", i,
                dstTensors->extTensors[i].rt2Tensor.GetAddr(), tensors->extTensors[i].rt2Tensor.GetAddr());
    }
    return OK;
}

void UpdateArgsUncontiguousTensor(NnopbaseUnContTensors& dstUncontTensors, NnopbaseUnContTensors& srcUncontTensors)
{
    // 刷新非连续tensor地址
    dstUncontTensors.idxs = srcUncontTensors.idxs;
    dstUncontTensors.workspaceOffsets = srcUncontTensors.workspaceOffsets;
    dstUncontTensors.tensors = srcUncontTensors.tensors;
    dstUncontTensors.tensorList = srcUncontTensors.tensorList;
    // ref算子非连续tensor地址刷新
    // refContTensorList和refUnContTensorList在NnopbaseRunForWorkspace后的NnopbaseGetRefUnContiguousTensors里刷新
    dstUncontTensors.refIdxs = srcUncontTensors.refIdxs;
    dstUncontTensors.refUnContTensors = srcUncontTensors.refUnContTensors;
    dstUncontTensors.refContTensors = srcUncontTensors.refContTensors;
}

bool NnopbaseIsV2CacheKeyEnabled(const NnopbaseExecutor* executor)
{
    return ((executor->matchArgsV2) && (g_nnopbaseSysCfgParams.enableArgsCache) &&
            !op::internal::GetOpProfilingRecordArgFlag());
}

void NnopbaseCheckHasContiguous(NnopbaseExecutor* executor)
{
    NnopbaseTensors* inputs = &executor->args->inputs;
    NnopbaseExecutorClearUnContiguousTensors(inputs);
    for (uint32_t i = 0U; i < inputs->paramDescs.count; i++) {
        auto& tensors = executor->ownArgs.inputs;
        if (tensors.paramDescs.instances[i].ignoreCont) {
            continue;
        }
        const size_t startIndex = inputs->paramDescs.instances[i].startIndex;
        if (inputs->paramDescs.instances[i].cfgNum == 2) { // 2表示dynamic
            const uint32_t dynamicNum = inputs->paramDescs.instances[i].num;
            for (uint32_t j = 0U; j < dynamicNum; j++) {
                NnopbaseCheckContiguous(inputs, (*tensors.paramDescs.instances[i].tensorList)[j], startIndex + j, i);
            }
        } else if (tensors.paramDescs.instances[i].tensor != nullptr) {
            NnopbaseCheckContiguous(inputs, tensors.paramDescs.instances[i].tensor, startIndex, i);
        }
    }
}

void NnopbaseSaveUnContiguousTensors(NnopbaseTensors* dstTensors, NnopbaseTensors* tensors)
{
    OP_LOGI("UnContiguous tensor size is %zu.", tensors->unContiguousTensors.tensors.size());
    if (tensors->unContiguousTensors.tensors.empty()) {
        return;
    }
    auto& inUnContTensors = dstTensors->unContiguousTensors;
    inUnContTensors.tensors = tensors->unContiguousTensors.tensors;
    inUnContTensors.idxs = tensors->unContiguousTensors.idxs;
    inUnContTensors.workspaceOffsets = tensors->unContiguousTensors.workspaceOffsets;
    OP_LOGI("UnContiguous tensors idxes size is %zu, workspaceOffsets size is %zu.", inUnContTensors.idxs.size(),
            inUnContTensors.workspaceOffsets.size());
    auto& inVec = inUnContTensors.tensors;
    inUnContTensors.tensorList.tensors = (&inVec[0U]);
    inUnContTensors.tensorList.size = inVec.size();
    if (!tensors->unContiguousTensors.refIdxs.empty()) {
        inUnContTensors.refIdxs = tensors->unContiguousTensors.refIdxs;
        inUnContTensors.refUnContTensors = tensors->unContiguousTensors.refUnContTensors;
        inUnContTensors.refContTensors = tensors->unContiguousTensors.refContTensors;
    }
}

#ifdef __cplusplus
}
#endif
