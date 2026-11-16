/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/framework_op.h"
#include "kernel_launcher.h"
#include "opdev/op_dfx.h"
#include "acl/acl_rt.h"

namespace op {

OP_TYPE_REGISTER(CopyToNpu)

OP_TYPE_REGISTER(CopyNpuToNpu)

OP_TYPE_REGISTER(CopyToNpuSync)

constexpr uint64_t BYTE_BITS_MINUS_ONE = 7;
constexpr uint64_t LOG_2_BYTE_BITS = 3;
constexpr uint16_t kModelId = 36; // AICPU

static inline aclnnStatus CalcTensorNBytes(const aclTensor* tensor, uint64_t& nBytes)
{
    const size_t tensorTypeSize = TypeSize(tensor->GetDataType());
    if (tensorTypeSize > kDataTypeSizeBitOffset) {
        OP_CHECK(!ge::MulOverflow((tensorTypeSize - kDataTypeSizeBitOffset), tensor->GetStorageShape().GetShapeSize(),
                                  nBytes),
                 OP_LOGE(ACLNN_ERR_INNER, "Tensor size is too large, calc tensor nbytes overflow."),
                 return ACLNN_ERR_INNER);

        nBytes = (nBytes + BYTE_BITS_MINUS_ONE) >> LOG_2_BYTE_BITS;
        return ACLNN_SUCCESS;
    } else {
        OP_CHECK(!ge::MulOverflow((tensorTypeSize), tensor->GetStorageShape().GetShapeSize(), nBytes),
                 OP_LOGE(ACLNN_ERR_INNER, "Tensor size is too large, calc tensor nbytes overflow."),
                 return ACLNN_ERR_INNER);
        return ACLNN_SUCCESS;
    }
}

class CopyToNpuKernelLauncher : public KernelLauncher {
public:
    CopyToNpuKernelLauncher(uint32_t opType, op::CoreType coreType, op::internal::ProfilingInfoId& profilingId,
                            const aclOpExecutor* executor, const aclTensor* src, const aclTensor* dst)
        : KernelLauncher(opType, coreType, executor, profilingId), src_(src), dst_(dst) {};
    aclnnStatus Launch() override
    {
        uint64_t dstNByptes = 0;
        uint64_t srcNByptes = 0;

        auto calcRet = CalcTensorNBytes(src_, srcNByptes);
        CHECK_RET(calcRet == ACLNN_SUCCESS, calcRet);
        calcRet = CalcTensorNBytes(dst_, dstNByptes);
        CHECK_RET(calcRet == ACLNN_SUCCESS, calcRet);
        OP_CHECK(
            dstNByptes >= srcNByptes,
            OP_LOGE(ACLNN_ERR_INNER, "Dst tensor size:%lu is less than src tensor size: %lu.", dstNByptes, srcNByptes),
            return ACLNN_ERR_INNER);
        auto ret = aclrtMemcpyAsync(dst_->GetData(), dstNByptes, src_->GetData(), srcNByptes,
                                    ACL_MEMCPY_HOST_TO_BUF_TO_DEVICE, executor_->GetStream());
        OP_CHECK(ret == ACL_SUCCESS,
                 OP_LOGE(ACLNN_ERR_INNER, "Call aclrtMemcpyAsync failed when do CopyToNpuOp, ret code: %d", ret),
                 return ACLNN_ERR_RUNTIME_ERROR);
        return ACLNN_SUCCESS;
    }

    internal::OpKernelBin* GetBin() override { return nullptr; };

    bool CheckRepeatable([[maybe_unused]] const std::unordered_map<const aclStorage*, const aclStorage*>& relation,
                         [[maybe_unused]] const std::vector<const aclStorage*>& oriStorage) override
    {
        return false;
    }

private:
    const aclTensor* src_;
    const aclTensor* dst_;
};

class CopyNpuToNpuKernelLauncher : public KernelLauncher {
public:
    CopyNpuToNpuKernelLauncher(uint32_t opType, op::CoreType coreType, op::internal::ProfilingInfoId& profilingId,
                               const aclOpExecutor* executor, const aclTensor* src, const aclTensor* dst)
        : KernelLauncher(opType, coreType, executor, profilingId), src_(src), dst_(dst) {};
    aclnnStatus Launch() override
    {
        uint64_t dstNByptes = 0;
        uint64_t srcNByptes = 0;

        auto calcRet = CalcTensorNBytes(src_, srcNByptes);
        CHECK_RET(calcRet == ACLNN_SUCCESS, calcRet);
        calcRet = CalcTensorNBytes(dst_, dstNByptes);
        CHECK_RET(calcRet == ACLNN_SUCCESS, calcRet);
        OP_CHECK(
            dstNByptes >= srcNByptes,
            OP_LOGE(ACLNN_ERR_INNER, "Dst tensor size:%lu is less than src tensor size: %lu.", dstNByptes, srcNByptes),
            return ACLNN_ERR_INNER);
        auto ret = aclrtMemcpyAsync(dst_->GetData(), dstNByptes, src_->GetData(), srcNByptes,
                                    ACL_MEMCPY_DEVICE_TO_DEVICE, executor_->GetStream());
        OP_CHECK(ret == ACL_SUCCESS,
                 OP_LOGE(ACLNN_ERR_INNER, "Call aclrtMemcpyAsync failed when do CopyNpuToNpuOp, ret code: %d", ret),
                 return ACLNN_ERR_RUNTIME_ERROR);
        return ACLNN_SUCCESS;
    }

    internal::OpKernelBin* GetBin() override { return nullptr; };

    bool CheckRepeatable([[maybe_unused]] const std::unordered_map<const aclStorage*, const aclStorage*>& relation,
                         [[maybe_unused]] const std::vector<const aclStorage*>& oriStorage) override
    {
        return false;
    }

private:
    const aclTensor* src_;
    const aclTensor* dst_;
};

const aclTensor* CopyToNpu(const aclTensor* src, aclOpExecutor* executor)
{
    // because rts memcpy cannot be cached, so abandon cache when use memcpy
    L0_DFX(CopyToNpu, src)

    OP_CHECK(src->GetPlacement() == op::TensorPlacement::kOnHost,
             OP_LOGE(ACLNN_ERR_INNER, "Get input param's placement:%d when expect kOnHost.", src->GetPlacement()),
             return nullptr);
    auto dst = executor->AllocTensor(src->GetStorageShape(), src->GetOriginalShape(), src->GetDataType(),
                                     src->GetStorageFormat(), src->GetOriginalFormat());

    op::internal::ProfilingInfoId profilingInfoId;

    auto* launcher = new CopyToNpuKernelLauncher{CopyToNpuOpTypeId(), op::NO_CALC, profilingInfoId, executor, src, dst};
    OpArg srcArg;
    srcArg.type = OpArgType::OPARG_ACLTENSOR;
    srcArg->pointer = const_cast<aclTensor*>(src);
    OpArg dstArg;
    dstArg.type = OpArgType::OPARG_ACLTENSOR;
    dstArg->pointer = dst;
    OpArgList srcArgList(&srcArg, 1);
    OpArgList dstArgList(&dstArg, 1);
    OpArgList emptyArgList;
    auto ret = executor->AddToKernelLauncherListCopyTask(CopyToNpuOpTypeId(), launcher, srcArgList, dstArgList,
                                                         emptyArgList);
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER, "Build input output relation for workspace calc failed"),
             return nullptr);
    return dst;
}

const aclTensor* CopyToNpuSync(const aclTensor* src, aclOpExecutor* executor)
{
    executor->AbandonCache(true);
    L0_DFX(CopyToNpuSync, src)
    OP_CHECK(src->GetPlacement() == op::TensorPlacement::kOnHost,
             OP_LOGE(ACLNN_ERR_INNER, "Get input param's placement:%d when expect kOnHost.", src->GetPlacement()),
             return nullptr);
    auto dst = executor->AllocTensor(src->GetStorageShape(), src->GetOriginalShape(), src->GetDataType(),
                                     src->GetStorageFormat(), src->GetOriginalFormat());
    OP_CHECK(dst != nullptr, OP_LOGE(ACLNN_ERR_INNER, "failed to alloc tensor"), return nullptr);

    uint64_t srcNByptes = 0;
    auto calcRet = CalcTensorNBytes(src, srcNByptes);
    CHECK_RET(calcRet == ACLNN_SUCCESS, nullptr);

    void* deviceMem = nullptr;
    aclrtMallocAttrValue moduleIdValue;
    moduleIdValue.moduleId = kModelId;
    aclrtMallocAttribute attrs{.attr = ACL_RT_MEM_ATTR_MODULE_ID, .value = moduleIdValue};
    aclrtMallocConfig cfg{.attrs = &attrs, .numAttrs = 1};
    aclError rc = aclrtMallocWithCfg(&deviceMem, srcNByptes, ACL_MEM_TYPE_HIGH_BAND_WIDTH, &cfg);
    OP_CHECK(rc == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER, "aclrtMallocWithCfg failed, rc = %d", rc), return nullptr);

    rc = aclrtMemcpy(deviceMem, srcNByptes, src->GetData(), srcNByptes, ACL_MEMCPY_HOST_TO_DEVICE);

    OP_CHECK(rc == ACL_SUCCESS, aclrtFree(deviceMem), return nullptr);
    dst->SetFromWorkspace(false);
    dst->SetStorageAddr(deviceMem);
    return dst;
}

aclnnStatus CopyNpuToNpu(const aclTensor* src, const aclTensor* dst, aclOpExecutor* executor)
{
    // because rts memcpy cannot be cached, so abandon cache when use memcpy
    L0_DFX(CopyNpuToNpu, src)

    OP_CHECK(src->GetPlacement() == op::TensorPlacement::kOnDeviceHbm,
             OP_LOGE(ACLNN_ERR_INNER, "Get input param's placement:%d when expect kOnDeviceHbm.", src->GetPlacement()),
             return ACLNN_ERR_INNER);
    OP_CHECK(dst->GetPlacement() == op::TensorPlacement::kOnDeviceHbm,
             OP_LOGE(ACLNN_ERR_INNER, "Get out param's placement:%d when expect kOnDeviceHbm.", src->GetPlacement()),
             return ACLNN_ERR_INNER);

    op::internal::ProfilingInfoId profilingInfoId;

    auto* launcher = new CopyNpuToNpuKernelLauncher{
        CopyToNpuOpTypeId(), op::NO_CALC, profilingInfoId, executor, src, dst};
    OpArg srcArg;
    srcArg.type = OpArgType::OPARG_ACLTENSOR;
    srcArg->pointer = const_cast<aclTensor*>(src);
    OpArg dstArg;
    dstArg.type = OpArgType::OPARG_ACLTENSOR;
    dstArg->pointer = const_cast<aclTensor*>(dst);
    OpArgList srcArgList(&srcArg, 1);
    OpArgList dstArgList(&dstArg, 1);
    OpArgList emptyArgList;
    auto ret = executor->AddToKernelLauncherListCopyTask(CopyNpuToNpuOpTypeId(), launcher, srcArgList, dstArgList,
                                                         emptyArgList);
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER, "Build input output relation for workspace calc failed"),
             return ACLNN_ERR_INNER);
    return ACLNN_SUCCESS;
}

} // namespace op
