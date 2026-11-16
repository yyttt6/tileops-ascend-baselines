/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rms_softmax.h
 * \brief Apply RMS normalization and stable softmax to FP32 UB tensors.
 */
#pragma once

#include "tensor_api/tensor.h"
#include "kernel_operator.h"

namespace Blaze::Epilogue::Tile {

#ifdef __NPU_ARCH__
constexpr AscendC::Reg::DivSpecificMode RMS_SOFTMAX_DIV_0ULP_FTZ_TRUE_MODE = {
    AscendC::Reg::MaskMergeMode::ZEROING,
    true,
    AscendC::DivAlgo::PRECISION_0ULP_FTZ_TRUE,
};
#endif // __NPU_ARCH__

class RmsSoftmax {
public:
    template <typename SumSquareTensor, typename DotTensor, typename MaxTensor, typename SumTensor>
    __aicore__ inline static void Run(const SumSquareTensor& sumSquareTensor, const DotTensor& dotTensor,
                                      const MaxTensor& maxTensor, const SumTensor& sumTensor, float reciprocalD,
                                      float epsilon)
    {
        using SumSquareElementType = asc::te::get_attribute_element_type<typename SumSquareTensor::element_type*>;
        using DotElementType = asc::te::get_attribute_element_type<typename DotTensor::element_type*>;
        using MaxElementType = asc::te::get_attribute_element_type<typename MaxTensor::element_type*>;
        using SumElementType = asc::te::get_attribute_element_type<typename SumTensor::element_type*>;
        using SumSquareLayoutPattern = asc::te::get_layout_pattern<typename SumSquareTensor::layout_type>;
        using DotLayoutPattern = asc::te::get_layout_pattern<typename DotTensor::layout_type>;
        using MaxLayoutPattern = asc::te::get_layout_pattern<typename MaxTensor::layout_type>;
        using SumLayoutPattern = asc::te::get_layout_pattern<typename SumTensor::layout_type>;
        static_assert(
            AscendC::Std::is_same_v<SumSquareElementType, float> && AscendC::Std::is_same_v<DotElementType, float> &&
                AscendC::Std::is_same_v<MaxElementType, float> && AscendC::Std::is_same_v<SumElementType, float>,
            "RmsSoftmax only supports FP32 tensors.");
        static_assert(AscendC::Std::is_same_v<asc::te::get_mem_location<SumSquareTensor>, asc::te::location::ub> &&
                          AscendC::Std::is_same_v<asc::te::get_mem_location<DotTensor>, asc::te::location::ub> &&
                          AscendC::Std::is_same_v<asc::te::get_mem_location<MaxTensor>, asc::te::location::ub> &&
                          AscendC::Std::is_same_v<asc::te::get_mem_location<SumTensor>, asc::te::location::ub>,
                      "RmsSoftmax only supports UB tensors.");
        static_assert(AscendC::Std::is_same_v<SumSquareLayoutPattern, asc::te::nd_ext_layout_ptn> &&
                          AscendC::Std::is_same_v<DotLayoutPattern, asc::te::nd_ext_layout_ptn> &&
                          AscendC::Std::is_same_v<MaxLayoutPattern, asc::te::nd_ext_layout_ptn> &&
                          AscendC::Std::is_same_v<SumLayoutPattern, asc::te::nd_ext_layout_ptn>,
                      "RmsSoftmax requires NDExt tensor layouts.");

        const uint32_t validN = static_cast<uint32_t>(asc::te::get_total_column_shape(dotTensor.layout()));
        const uint32_t nAlign = static_cast<uint32_t>(asc::te::get<1>(asc::te::get<0>(dotTensor.layout().stride())));
        auto sumSquareAddr = reinterpret_cast<__ubuf__ float*>(sumSquareTensor.data().get());
        auto dotAddr = reinterpret_cast<__ubuf__ float*>(dotTensor.data().get());
        auto maxAddr = reinterpret_cast<__ubuf__ float*>(maxTensor.data().get());
        auto sumAddr = reinterpret_cast<__ubuf__ float*>(sumTensor.data().get());
        asc_vf_call<RmsSoftmaxVf>(sumSquareAddr, dotAddr, maxAddr, sumAddr, validN, nAlign, reciprocalD, epsilon);
    }

private:
    static __simd_vf__ inline void RmsSoftmaxVf(__ubuf__ float* sumSquareAddr, __ubuf__ float* dotAddr,
                                                __ubuf__ float* maxAddr, __ubuf__ float* sumAddr, uint32_t validN,
                                                uint32_t nAlign, float reciprocalD, float epsilon)
    {
        AscendC::Reg::RegTensor<float> sumSquareReg;
        AscendC::Reg::RegTensor<float> dotReg;
        AscendC::Reg::RegTensor<float> rmsReg;
        AscendC::Reg::RegTensor<float> normalizedReg;
        AscendC::Reg::RegTensor<float> maxReg;
        AscendC::Reg::RegTensor<float> maxBroadcastReg;
        AscendC::Reg::RegTensor<float> expReg;
        AscendC::Reg::RegTensor<float> expSumReg;
        AscendC::Reg::RegTensor<float> zeroReg;
        AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg oneMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
        uint32_t validRemaining = validN;
        uint32_t alignRemaining = nAlign;
        AscendC::Reg::MaskReg validMask = AscendC::Reg::UpdateMask<float>(validRemaining);
        AscendC::Reg::MaskReg alignMask = AscendC::Reg::UpdateMask<float>(alignRemaining);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(sumSquareReg, sumSquareAddr);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(dotReg, dotAddr);
        AscendC::Reg::Muls<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(sumSquareReg, sumSquareReg, reciprocalD,
                                                                               validMask);
        AscendC::Reg::Adds<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(sumSquareReg, sumSquareReg, epsilon,
                                                                               validMask);
        AscendC::Reg::Sqrt<float, AscendC::Reg::MaskMergeMode::ZEROING>(rmsReg, sumSquareReg, validMask);
        AscendC::Reg::Div<float, &RMS_SOFTMAX_DIV_0ULP_FTZ_TRUE_MODE>(normalizedReg, dotReg, rmsReg, validMask);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::MAX, float, float, AscendC::Reg::MaskMergeMode::ZEROING>(
            maxReg, normalizedReg, validMask);
        AscendC::Reg::Duplicate<float, AscendC::Reg::HighLowPart::LOWEST, AscendC::Reg::MaskMergeMode::ZEROING>(
            maxBroadcastReg, maxReg, allMask);
        AscendC::Reg::Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(expReg, normalizedReg, maxBroadcastReg,
                                                                       validMask);
        AscendC::Reg::Exp<float, AscendC::Reg::MaskMergeMode::ZEROING>(expReg, expReg, validMask);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM, float, float, AscendC::Reg::MaskMergeMode::ZEROING>(
            expSumReg, expReg, validMask);
        AscendC::Reg::Duplicate(zeroReg, 0.0F, allMask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(dotAddr, zeroReg, alignMask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(dotAddr, expReg, validMask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(maxAddr, maxReg, oneMask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(sumAddr, expSumReg, oneMask);
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    }
};

} // namespace Blaze::Epilogue::Tile
