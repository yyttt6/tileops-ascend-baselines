/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "indv_cache_key_builder.h"
#include "utils/indv_hash.h"
#include "utils/thread_var_container.h"

namespace {
inline bool CompareTensor(const aclTensor* tensor, const aclTensor* prev)
{
    return (tensor->GetDataType() == prev->GetDataType()) && (tensor->GetViewShape() == prev->GetViewShape()) &&
           (tensor->GetStorageFormat() == prev->GetStorageFormat()) &&
           (tensor->GetViewStrides() == prev->GetViewStrides()) && (tensor->GetViewOffset() == prev->GetViewOffset());
}

uint8_t* AddShapeInfo(const GertTensor& rt2Tensor, uint8_t* key)
{
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(rt2Tensor.GetDataType()));
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(rt2Tensor.GetStorageFormat()));
    const GertShape& shape = rt2Tensor.GetStorageShape();
    const size_t dimNum = shape.GetDimNum();
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(dimNum));
    for (size_t j = 0U; j < dimNum; j++) {
        key = static_cast<uint8_t*>(NnopbaseAppend8Byte(key, static_cast<uint64_t>(shape.GetDim(j))));
    }
    const auto& stride = rt2Tensor.GetStride();
    const size_t strideNum = stride.GetDimNum();
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(strideNum));
    for (size_t j = 0U; j < strideNum; j++) {
        key = static_cast<uint8_t*>(NnopbaseAppend8Byte(key, static_cast<uint64_t>(stride.GetStride(j))));
    }
    key = static_cast<uint8_t*>(NnopbaseAppend8Byte(key, static_cast<uint64_t>(rt2Tensor.GetOffset())));
    return key;
}
} // namespace
namespace Indv {
size_t CacheKeyBuilder::CalculateCacheKeyLenV1(NnopbaseExecutor* executor)
{
    auto inputs = &executor->ownArgs.inputs;
    auto outputs = &executor->ownArgs.outputs;
    auto attrs = &executor->attrs;
    size_t keyLen = strlen(executor->opType); // 前期校验保证opType不为空
    for (uint32_t i = 0U; i < inputs->num; i++) {
        if (inputs->extTensors[i].isNull) {
            ++keyLen;
            continue;
        }
        keyLen += BASE_BYTES + SHAPE_BYTES * inputs->extTensors[i].rt2Tensor.GetStorageShape().GetDimNum() +
                  STRIDE_NUM_BYTES + SHAPE_BYTES * inputs->extTensors[i].rt2Tensor.GetStride().GetDimNum() +
                  OFFSET_BYTES;
        if (inputs->extTensors[i].valueDepend) {
            keyLen += inputs->extTensors[i].rt2Tensor.GetSize();
        }
    }
    ++keyLen;
    for (uint32_t i = 0U; i < outputs->num; i++) {
        if (outputs->extTensors[i].isNull) {
            ++keyLen;
            continue;
        }
        keyLen += BASE_BYTES + SHAPE_BYTES * outputs->extTensors[i].rt2Tensor.GetStorageShape().GetDimNum() +
                  STRIDE_NUM_BYTES + SHAPE_BYTES * outputs->extTensors[i].rt2Tensor.GetStride().GetDimNum() +
                  OFFSET_BYTES;
    }
    ++keyLen;
    if (attrs->num > 0U) {
        for (size_t i = 0U; i < attrs->num; i++) {
            keyLen += attrs->attrs[i].addr.size;
        }
    }
    ++keyLen; // append '/'
    keyLen += sizeof(NnopbaseCoreNum);
    ++keyLen; // append '/'
    keyLen += sizeof(uint32_t);
    ++keyLen; // append '/'
    keyLen += sizeof(uint8_t);
    return keyLen;
}

// 溢出检测、确定性计算等是全局配置选项，不需要在后续每次匹配时都考虑
// optype, input dtype/format/shape, output dtype/format/shape, attr
void CacheKeyBuilder::GenerateCacheArgsKeyV1(NnopbaseExecutor* executor)
{
    executor->ownArgs.keyLen = CalculateCacheKeyLenV1(executor);
    OP_LOGI("KeyLen is %zu.", executor->ownArgs.keyLen);
    if (executor->ownArgs.keyLen > executor->ownArgs.inputKey.size()) {
        executor->ownArgs.inputKey.resize(executor->ownArgs.keyLen);
    }
    const auto inputs = &executor->ownArgs.inputs;
    const auto outputs = &executor->ownArgs.outputs;
    const auto attrs = &executor->attrs;
    const auto coreNum = &executor->coreNum;
    uint8_t* key = executor->ownArgs.inputKey.data();
    key = static_cast<uint8_t*>(
        NnopbaseAppendBinary(key, executor->ownArgs.inputKey.size(), executor->opType, strlen(executor->opType)));
    for (uint32_t i = 0U; i < inputs->num; i++) {
        if (inputs->extTensors[i].isNull) {
            key = NnopbaseAppend1Byte(key, '/');
            continue;
        }
        key = AddShapeInfo(inputs->extTensors[i].rt2Tensor, key);
        if (inputs->extTensors[i].valueDepend) {
            const auto addr = inputs->extTensors[i].rt2Tensor.GetAddr();
            const auto length = inputs->extTensors[i].rt2Tensor.GetSize();
            key = static_cast<uint8_t*>(NnopbaseAppendBinary(key, length, addr, length));
        }
    }
    key = NnopbaseAppend1Byte(key, '/');
    for (uint32_t i = 0U; i < outputs->num; i++) {
        if (outputs->extTensors[i].isNull) {
            key = NnopbaseAppend1Byte(key, '/');
            continue;
        }
        key = AddShapeInfo(outputs->extTensors[i].rt2Tensor, key);
    }
    key = NnopbaseAppend1Byte(key, '/');
    if (attrs->num > 0U) {
        for (size_t i = 0U; i < attrs->num; i++) {
            key = op::internal::PtrCastTo<NnopbaseUChar>(NnopbaseAppendBinary(
                key, attrs->attrs[i].addr.size, attrs->attrs[i].addr.addr, attrs->attrs[i].addr.size));
        }
    }
    key = NnopbaseAppend1Byte(key, '/');
    key = op::internal::PtrCastTo<NnopbaseUChar>(
        NnopbaseAppendBinary(key, sizeof(NnopbaseCoreNum), coreNum, sizeof(NnopbaseCoreNum)));
    key = NnopbaseAppend1Byte(key, '/');
    uint32_t mc2RankId = nnopbase::utils::ThreadVarContainer::GetCurMc2RankIdInThread();
    key = op::internal::PtrCastTo<NnopbaseUChar>(
        NnopbaseAppendBinary(key, sizeof(uint32_t), &mc2RankId, sizeof(uint32_t)));
    key = NnopbaseAppend1Byte(key, '/');
    key = op::internal::PtrCastTo<NnopbaseUChar>(
        NnopbaseAppendBinary(key, sizeof(uint8_t), &(executor->deterministicLevel), sizeof(uint8_t)));
    executor->ownArgs.seed = NnopbaseHashBinary(executor->ownArgs.inputKey.data(), executor->ownArgs.keyLen);
}

void CacheKeyBuilder::EnsureCapacity(NnopbaseExecutorArgs* args, size_t len)
{
    if (args->remainKeyLen < len) {
        args->remainKeyLen += NNOPBASE_MAX_ARGS_KEY_LEN;
        args->inputKey.resize(args->inputKey.size() + NNOPBASE_MAX_ARGS_KEY_LEN);
    }
}

NnopbaseUChar* CacheKeyBuilder::AppendShapeInfo(NnopbaseExecutorArgs* args, const aclTensor* tensor)
{
    const op::Shape& shape = tensor->GetViewShape();
    const size_t dimNum = shape.GetDimNum();
    const auto& strides = tensor->GetViewStrides();
    const size_t strideNum = strides.size();
    size_t len = BASE_BYTES + dimNum * SHAPE_BYTES + STRIDE_NUM_BYTES + strideNum * SHAPE_BYTES + OFFSET_BYTES;
    EnsureCapacity(args, len);
    NnopbaseUChar* key = op::internal::PtrCastTo<NnopbaseUChar>(args->inputKey.data()) + args->keyLen;
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(tensor->GetDataType()));
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(tensor->GetStorageFormat()));
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(dimNum));
    for (size_t j = 0U; j < dimNum; j++) {
        key = NnopbaseAppend8Byte(key, static_cast<uint64_t>(shape.GetDim(j)));
    }
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(strideNum));
    for (size_t j = 0U; j < strideNum; j++) {
        key = NnopbaseAppend8Byte(key, static_cast<uint64_t>(strides[j]));
    }
    key = NnopbaseAppend8Byte(key, static_cast<uint64_t>(tensor->GetViewOffset()));
    args->remainKeyLen -= len;
    args->keyLen += len;
    return key;
}

NnopbaseUChar* CacheKeyBuilder::AppendPlaceHolder(NnopbaseExecutorArgs* args, NnopbaseUChar* key)
{
    EnsureCapacity(args, 1);
    key = op::internal::PtrCastTo<NnopbaseUChar>(args->inputKey.data() + args->keyLen);
    key = NnopbaseAppend1Byte(key, '/');
    args->keyLen += 1;
    args->remainKeyLen -= 1;
    return key;
}

NnopbaseUChar* CacheKeyBuilder::AppendCoreNum(NnopbaseExecutorArgs* args, const NnopbaseCoreNum* coreNum)
{
    NnopbaseUChar* key = op::internal::PtrCastTo<NnopbaseUChar>(args->inputKey.data() + args->keyLen);
    key = AppendPlaceHolder(args, key);
    EnsureCapacity(args, sizeof(NnopbaseCoreNum));
    key = op::internal::PtrCastTo<NnopbaseUChar>(args->inputKey.data() + args->keyLen);
    key = op::internal::PtrCastTo<NnopbaseUChar>(
        NnopbaseAppendBinary(key, args->remainKeyLen, coreNum, sizeof(NnopbaseCoreNum)));
    args->keyLen += sizeof(NnopbaseCoreNum);
    args->remainKeyLen -= sizeof(NnopbaseCoreNum);
    return key;
}

NnopbaseUChar* CacheKeyBuilder::AppendDeterministicLevel(NnopbaseExecutorArgs* args, const uint8_t* deterministicLevel)
{
    NnopbaseUChar* key = op::internal::PtrCastTo<NnopbaseUChar>(args->inputKey.data() + args->keyLen);
    key = AppendPlaceHolder(args, key);
    EnsureCapacity(args, sizeof(uint8_t));
    key = op::internal::PtrCastTo<NnopbaseUChar>(args->inputKey.data() + args->keyLen);
    key = op::internal::PtrCastTo<NnopbaseUChar>(
        NnopbaseAppendBinary(key, args->remainKeyLen, deterministicLevel, sizeof(uint8_t)));
    args->keyLen += sizeof(uint8_t);
    args->remainKeyLen -= sizeof(uint8_t);
    return key;
}

NnopbaseUChar* CacheKeyBuilder::AppendMc2RankId(NnopbaseExecutorArgs* args, const uint32_t* rankId)
{
    NnopbaseUChar* key = op::internal::PtrCastTo<NnopbaseUChar>(args->inputKey.data() + args->keyLen);
    key = AppendPlaceHolder(args, key);
    EnsureCapacity(args, sizeof(uint32_t));
    key = op::internal::PtrCastTo<NnopbaseUChar>(args->inputKey.data() + args->keyLen);
    key = op::internal::PtrCastTo<NnopbaseUChar>(
        NnopbaseAppendBinary(key, args->remainKeyLen, rankId, sizeof(uint32_t)));
    args->keyLen += sizeof(uint32_t);
    args->remainKeyLen -= sizeof(uint32_t);
    return key;
}

void CacheKeyBuilder::AppendTensor(NnopbaseExecutorArgs* args, const aclTensor* tensor)
{
    NnopbaseUChar* key = op::internal::PtrCastTo<NnopbaseUChar>(args->inputKey.data() + args->keyLen);
    if (tensor == nullptr) {
        (void)AppendPlaceHolder(args, key);
    } else {
        (void)AppendShapeInfo(args, tensor);
    }
}

void CacheKeyBuilder::AppendValueDependTensor(NnopbaseExecutorArgs* args, const void* addr, const uint64_t dim,
                                              const uint64_t dataLen, ge::DataType dType)
{
    const size_t shapeLen = BASE_BYTES + SHAPE_BYTES;
    if (args->remainKeyLen < shapeLen + dataLen) {
        size_t multiples = (shapeLen + dataLen) / NNOPBASE_MAX_ARGS_KEY_LEN + 1;
        args->remainKeyLen += multiples * NNOPBASE_MAX_ARGS_KEY_LEN;
        args->inputKey.resize(args->inputKey.size() + multiples * NNOPBASE_MAX_ARGS_KEY_LEN);
    }
    NnopbaseUChar* key = op::internal::PtrCastTo<NnopbaseUChar>(args->inputKey.data() + args->keyLen);
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(dType));
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(ge::FORMAT_ND));
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(1U));
    key = NnopbaseAppend8Byte(key, dim);
    args->remainKeyLen -= shapeLen;
    args->keyLen += shapeLen;
    key = NnopbaseAppendBinary(key, args->remainKeyLen, addr, dataLen);
    args->remainKeyLen -= dataLen;
    args->keyLen += dataLen;
}

void CacheKeyBuilder::AppendTensorList(NnopbaseExecutorArgs* args, const aclTensorList* tensorList)
{
    NnopbaseUChar* key = op::internal::PtrCastTo<NnopbaseUChar>(args->inputKey.data() + args->keyLen);
    if ((tensorList == nullptr) || (tensorList->Size() == 0U)) {
        AppendPlaceHolder(args, key);
        return;
    }
    uint32_t cnt = 1U;
    auto prev = (*tensorList)[0];
    for (uint64_t i = 0U; i < tensorList->Size(); i++) {
        if (i > 0U) {
            if (((*tensorList)[i] != nullptr) && prev != nullptr && CompareTensor((*tensorList)[i], prev)) {
                cnt++;
            } else {
                if (prev != nullptr) {
                    key = AppendShapeInfo(args, prev);
                    EnsureCapacity(args, 1);
                    key = op::internal::PtrCastTo<NnopbaseUChar>(args->inputKey.data() + args->keyLen);
                    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(cnt));
                    args->keyLen += 1;
                    args->remainKeyLen -= 1;
                }
                if ((*tensorList)[i] == nullptr) {
                    key = AppendPlaceHolder(args, key);
                    prev = (*tensorList)[i];
                    continue;
                }
                prev = (*tensorList)[i];
                cnt = 1;
            }
        }
    }
    if (prev != nullptr) {
        key = AppendShapeInfo(args, prev);
        EnsureCapacity(args, 1);
        key = op::internal::PtrCastTo<NnopbaseUChar>(args->inputKey.data() + args->keyLen);
        key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(cnt));
        args->keyLen += 1;
        args->remainKeyLen -= 1;
    }
}

NnopbaseUChar* CacheKeyBuilder::AppendScalarInfo(NnopbaseExecutorArgs* args, const ge::DataType dtype)
{
    // scalar dimNum为0，无需占用8字节填shape信息
    EnsureCapacity(args, BASE_BYTES);
    NnopbaseUChar* key = op::internal::PtrCastTo<NnopbaseUChar>(args->inputKey.data()) + args->keyLen;
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(dtype));
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(ge::FORMAT_ND));
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(0U));
    args->remainKeyLen -= BASE_BYTES;
    args->keyLen += BASE_BYTES;
    return key;
}

void CacheKeyBuilder::AppendScalar(NnopbaseExecutorArgs* args, const aclScalar* scalar, const uint32_t index,
                                   const int32_t srcIndex, const ge::DataType dtype)
{
    if (scalar == nullptr) {
        NnopbaseUChar* key = op::internal::PtrCastTo<NnopbaseUChar>(args->inputKey.data() + args->keyLen);
        key = AppendPlaceHolder(args, key);
    } else {
        ge::DataType dataType = scalar->GetDataType();
        if (dtype != ge::DT_UNDEFINED) {
            dataType = dtype;
        } else if ((srcIndex != -1) && (static_cast<uint32_t>(srcIndex) < index) &&
                   static_cast<size_t>(srcIndex) < args->inputs.paramDescs.instances.size() &&
                   args->inputs.extTensors.size() >
                       args->inputs.paramDescs.instances[static_cast<size_t>(srcIndex)].startIndex) {
            dataType = args->inputs
                           .extTensors[args->inputs.paramDescs.instances[static_cast<size_t>(srcIndex)].startIndex]
                           .rt2Tensor.GetDataType();
        }
        (void)AppendScalarInfo(args, dataType);
    }
}

NnopbaseUChar* CacheKeyBuilder::AppendScalarListInfo(NnopbaseExecutorArgs* args, const ge::DataType dtype,
                                                     const uint64_t size)
{
    size_t len = BASE_BYTES + SHAPE_BYTES;
    EnsureCapacity(args, len);
    NnopbaseUChar* key = op::internal::PtrCastTo<NnopbaseUChar>(args->inputKey.data()) + args->keyLen;
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(dtype));
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(ge::FORMAT_ND));
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(1U));
    key = NnopbaseAppend8Byte(key, static_cast<uint64_t>(size));
    args->remainKeyLen -= BASE_BYTES + SHAPE_BYTES;
    args->keyLen += BASE_BYTES + SHAPE_BYTES;
    return key;
}

void CacheKeyBuilder::AppendScalarList(NnopbaseExecutorArgs* args, const aclScalarList* scalarList,
                                       const uint32_t index, const int32_t srcIndex, const ge::DataType dtype)
{
    NnopbaseUChar* key = op::internal::PtrCastTo<NnopbaseUChar>(args->inputKey.data() + args->keyLen);
    if ((scalarList == nullptr) || (scalarList->Size() == 0U)) {
        key = AppendPlaceHolder(args, key);
    } else {
        ge::DataType dataType = (*scalarList)[0]->GetDataType();
        if (dtype != ge::DT_UNDEFINED) {
            dataType = dtype;
        } else if ((srcIndex != -1) && (static_cast<uint32_t>(srcIndex) < index) &&
                   static_cast<size_t>(srcIndex) < args->inputs.paramDescs.instances.size() &&
                   args->inputs.extTensors.size() >
                       args->inputs.paramDescs.instances[static_cast<size_t>(srcIndex)].startIndex) {
            dataType = args->inputs
                           .extTensors[args->inputs.paramDescs.instances[static_cast<size_t>(srcIndex)].startIndex]
                           .rt2Tensor.GetDataType();
        }
        (void)AppendScalarListInfo(args, dataType, scalarList->Size());
    }
}
} // namespace Indv
