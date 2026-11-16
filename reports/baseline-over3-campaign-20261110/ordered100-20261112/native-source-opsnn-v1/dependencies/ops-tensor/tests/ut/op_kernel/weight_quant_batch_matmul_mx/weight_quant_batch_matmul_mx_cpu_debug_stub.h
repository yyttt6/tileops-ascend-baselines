/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file weight_quant_batch_matmul_mx_cpu_debug_stub.h
 * \brief tikicpulib compile compatibility for the Weight Quant MX MicroAPI tile functions.
 */

#pragma once

#if defined(ASCENDC_CPU_DEBUG)

#include "blaze/gemm/utils/common_utils.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"

namespace WeightQuantBatchMatmulMxUT::CpuSim {

template <typename T>
__ubuf__ T* ToPhysicalUbAddress(__ubuf__ T* logicalAddress)
{
    uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(logicalAddress));
    AscendC::LocalTensor<uint8_t> localTensor(AscendC::TPosition::VECCALC, offset, AscendC::TOTAL_UB_SIZE - offset);
    return reinterpret_cast<__ubuf__ T*>(localTensor.GetPhyAddr());
}

} // namespace WeightQuantBatchMatmulMxUT::CpuSim

namespace AscendC::Reg {

template <typename T = DefaultType, LoadDist dist = LoadDist::DIST_NORM, typename U>
void LoadAlignCpuDebug(U& destination, __ubuf__ T* source, AddrReg address)
{
    auto* physicalSource = WeightQuantBatchMatmulMxUT::CpuSim::ToPhysicalUbAddress(source);
    LoadAlign<T, dist>(destination, physicalSource, address);
}

template <typename T = DefaultType, StoreDist dist = StoreDist::DIST_NORM, typename U>
void StoreAlignCpuDebug(__ubuf__ T* destination, U& source, AddrReg address, MaskReg& mask)
{
    auto* physicalDestination = WeightQuantBatchMatmulMxUT::CpuSim::ToPhysicalUbAddress(destination);
    StoreAlign<T, dist>(physicalDestination, source, address, mask);
}

template <typename T = DefaultType, DataCopyMode dataMode, PostLiteral postMode, typename U>
void StoreAlignCpuDebug(__ubuf__ T*& destination, U& source, uint32_t blockStride, uint32_t repeatStride, MaskReg& mask)
{
    auto* physicalDestination = WeightQuantBatchMatmulMxUT::CpuSim::ToPhysicalUbAddress(destination);
    auto* physicalBase = physicalDestination;
    StoreAlign<T, dataMode, postMode>(physicalDestination, source, blockStride, repeatStride, mask);
    destination += physicalDestination - physicalBase;
}

} // namespace AscendC::Reg

#define LoadAlign LoadAlignCpuDebug
#define StoreAlign StoreAlignCpuDebug

// NPU builtin type aliases must be defined before any arch35 tile header is parsed.
#define __fp8e4m3 fp8_e4m3fn_t
#define __fp4e2m1x2 fp4x2_e2m1_t

// tikicpulib's half constructor is not constexpr; keep this compatibility change local to scale_mx_bias.
// shift_w4_to_w8 must stay outside the guarded window: it relies on real constexpr for
// static_assert and if constexpr layout dispatch. Its own include of scale_mx_bias.h is a no-op
// here (#pragma once), so the guarded definition above is reused.
#define constexpr
#include "blaze/gemm/tile/arch35/scale_mx_bias.h"
#undef constexpr
#include "blaze/gemm/tile/arch35/shift_w4_to_w8.h"

#undef __fp4e2m1x2
#undef __fp8e4m3
#undef StoreAlign
#undef LoadAlign

#endif
