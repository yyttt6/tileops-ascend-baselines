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
 * \file dequant.h
 * \brief Vector dequantization tile: Dequant<Params>::Run(inputTensor, scaleTensor, outTensor, params).
 *        The Params type selects the dequant semantics; dtype/layout differences are dispatched on the
 *        InputTensor type. Epilogue::Tile exposes only the Dequant entry: the per-group W4 VF
 *        kernels are private static members of the Dequant specialization below and are not exposed
 *        to callers.
 */

#pragma once

#include "kernel_operator.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "tensor_api/tensor.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_struct.h"
#include "blaze/gemm/utils/layout_utils.h"

namespace Reg = AscendC::Reg;

namespace Blaze::Epilogue::Tile {
using AscendC::IsSameType;

struct DefaultDequantParams {};

template <class Params_ = DefaultDequantParams>
class Dequant {
public:
    using Params = Params_;

    template <class InputTensor, class ScaleTensor, class OutTensor>
    __aicore__ inline static void Run(const InputTensor& input, const ScaleTensor& scale, const OutTensor& output,
                                      const Params& params = {})
    {
        (void)input;
        (void)scale;
        (void)output;
        (void)params;
        static_assert(!AscendC::Std::is_same_v<Params, Params>, "Dequant has no implementation for this Params type");
    }
};

// Per-group W4 weight dequant: packed FP4 weight + per-group scale -> FP8 (group size 32
// only). validK floors validK / GROUP_SIZE (a trailing partial K group is not converted),
// a value no layout encodes, so it rides in Params; scaleMaskAddr carries the VF-internal
// scale-select mask blob owned by the caller's UB storage.
template <uint32_t GroupSize_>
struct A8W4TcgDequantParams {
    static constexpr uint32_t GROUP_SIZE = GroupSize_;

    uint32_t validK{0};
    __ubuf__ uint8_t* scaleMaskAddr{nullptr};
};

template <uint32_t GroupSize>
class Dequant<A8W4TcgDequantParams<GroupSize>> {
public:
    using Params = A8W4TcgDequantParams<GroupSize>;

    static constexpr uint32_t GROUP_SIZE_32 = 32;

    template <class InputTensor, class ScaleTensor, class OutTensor>
    __aicore__ inline static void Run(const InputTensor& input, const ScaleTensor& scale, const OutTensor& output,
                                      const Params& params)
    {
        using XType = asc::te::get_attribute_element_type<typename OutTensor::element_type*>;
        using WType = asc::te::get_attribute_element_type<typename InputTensor::element_type*>;
        using SType = asc::te::get_attribute_element_type<typename ScaleTensor::element_type*>;
        // Entry gate: only the supported per-group quantization shapes may proceed.
        static_assert(GroupSize == GROUP_SIZE_32, "Per-group W4 dequant currently only supports a group size of 32.");
        static_assert(AscendC::Std::is_same_v<asc::te::get_mem_location<InputTensor>, asc::te::location::ub> &&
                          AscendC::Std::is_same_v<asc::te::get_mem_location<ScaleTensor>, asc::te::location::ub> &&
                          AscendC::Std::is_same_v<asc::te::get_mem_location<OutTensor>, asc::te::location::ub>,
                      "Per-group W4 dequant only supports UB tensors.");
        static_assert(AscendC::IsSameType<XType, fp8_e4m3fn_t>::value,
                      "Per-group W4 dequant expects fp8_e4m3fn output.");
        static_assert(Blaze::Gemm::IsFp4<WType>(), "Per-group W4 dequant expects packed FP4 input.");
        static_assert(AscendC::IsSameType<SType, half>::value || AscendC::IsSameType<SType, bfloat16_t>::value,
                      "Per-group W4 dequant expects FP16/BF16 scale.");

        using InputPattern = asc::te::get_layout_pattern<typename InputTensor::layout_type>;
        constexpr bool IS_DN_TRANS_INPUT = AscendC::Std::is_same_v<InputPattern, asc::te::dn_ext_layout_ptn>;
        constexpr bool IS_NZ_INPUT = AscendC::Std::is_same_v<InputPattern, asc::te::nz_layout_ptn>;
        static_assert(IS_DN_TRANS_INPUT || IS_NZ_INPUT,
                      "Per-group W4 dequant supports DNExt (ND-trans) and NZ input layouts.");

        if constexpr (IS_DN_TRANS_INPUT) {
            RunNdTrans(input, scale, output, params);
        } else {
            RunNz(input, scale, output, params);
        }
    }

private:
    static constexpr int32_t PERGROUP_OFFSET_8 = 8;
    static constexpr int32_t PERGROUP_OFFSET_FOR_4_BITS = 128;
    static constexpr int32_t OFFSET_8 = 8;
    static constexpr int32_t OFFSET_64 = 64;
    static constexpr int32_t ALIGNED_32_SIZE = 32;
    static constexpr int32_t UB_ALIGN_SIZE_FOR_4_BITS = 64;
    static constexpr int64_t C0_SIZE_B8 = static_cast<int64_t>(Blaze::Gemm::C0_SIZE_B8);
    static constexpr int64_t ELEMENTS_PER_GROUP = static_cast<int64_t>(GroupSize);
    static constexpr int32_t INT4_PACK_SHIFT = 1;
    static constexpr int64_t INT4_DTYPE_FACTOR = 2;

    static constexpr Reg::CastTrait castF162F32Trait0 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                         Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    static constexpr Reg::CastTrait castF162F32Trait1 = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                         Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    static constexpr Reg::CastTrait castF322F8Trait0 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                        Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    static constexpr Reg::CastTrait castF322F8Trait2 = {Reg::RegLayout::TWO, Reg::SatMode::NO_SAT,
                                                        Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    static constexpr Reg::CastTrait castF42F16Trait0 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                        Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    static constexpr Reg::CastTrait castBF162FP16Trait0 = {Reg::RegLayout::UNKNOWN, Reg::SatMode::NO_SAT,
                                                           Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

    template <typename xType, typename wType, typename scaleType>
    struct DequantW4Pergroup32NKParams {
        uint32_t maskWeight;
        uint16_t outerExtend;
        uint16_t innerExtend;
        uint32_t outerStrideScale;
        uint32_t outerStrideWeight;
        uint32_t dataBlockStride;
        uint32_t repeatStride;
        int32_t outDimOffset;
        __ubuf__ scaleType* scaleBaseAddr;
        __ubuf__ int8_t* weightInBaseAddr0;
        __ubuf__ int8_t* weightInBaseAddr1;
        __ubuf__ xType* weightOutBaseAddr;
    };

    template <typename xType, typename wType, typename scaleType>
    struct DequantW4PergroupKNParams {
        uint32_t groupNumUb;
        uint32_t vLLoopNumInGroup;
        uint32_t n1LoopNum;
        uint32_t scaleN1Stride;
        uint32_t bubNLen;
        uint32_t weightInGroupIdStride;
        uint32_t weightInN1Stride;
        uint32_t weightOutN1Stride;
        uint32_t weightOutGroupIdStride;
        uint32_t weightOutVlStride;
        __ubuf__ scaleType* scaleBaseAddr0;
        __ubuf__ scaleType* scaleBaseAddr1;
        __ubuf__ uint8_t* scaleMaskBaseAddr;
        __ubuf__ int8_t* weightInBaseAddr0;
        __ubuf__ int8_t* weightInBaseAddr1;
        __ubuf__ xType* weightOutBaseAddr;
    };

    template <typename wType, typename scaleType>
    static __simd_callee__ inline void CastWeightF4ToF16(Reg::RegTensor<scaleType>& weightF16Reg0,
                                                         Reg::RegTensor<scaleType>& weightF16Reg1,
                                                         Reg::RegTensor<wType>& weightInReg0,
                                                         Reg::RegTensor<wType>& weightInReg1, Reg::MaskReg& maskRegALL)
    {
        if constexpr (AscendC::IsSameType<scaleType, half>::value) {
            Reg::RegTensor<bfloat16_t> weightBF16Reg0, weightBF16Reg1;
            Reg::Cast<bfloat16_t, wType, castF42F16Trait0>(weightBF16Reg0, weightInReg0, maskRegALL);
            Reg::Cast<bfloat16_t, wType, castF42F16Trait0>(weightBF16Reg1, weightInReg1, maskRegALL);
            Reg::Cast<scaleType, bfloat16_t, castBF162FP16Trait0>(weightF16Reg0, weightBF16Reg0, maskRegALL);
            Reg::Cast<scaleType, bfloat16_t, castBF162FP16Trait0>(weightF16Reg1, weightBF16Reg1, maskRegALL);
        } else {
            Reg::Cast<scaleType, wType, castF42F16Trait0>(weightF16Reg0, weightInReg0, maskRegALL);
            Reg::Cast<scaleType, wType, castF42F16Trait0>(weightF16Reg1, weightInReg1, maskRegALL);
        }
    }

    template <typename xType, typename wType, typename scaleType>
    static __simd_vf__ inline void DequantW4Pergroup32NK(DequantW4Pergroup32NKParams<xType, wType, scaleType> p)
    {
        static constexpr Reg::CastTrait castTrait0 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                      Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTrait1 = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                      Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTrait2 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                      Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        static constexpr Reg::CastTrait castTrait3 = {Reg::RegLayout::TWO, Reg::SatMode::NO_SAT,
                                                      Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

        Reg::RegTensor<scaleType> scaleLoad, scaleCompute0, scaleCompute1;
        Reg::RegTensor<wType> wLoad0, wLoad1;
        Reg::RegTensor<scaleType> wCvt0, wCvt1, wMul0, wMul1;
        Reg::RegTensor<xType> wCvtB8N0, wCvtB8N1, wCvtB8N2, wCvtB8N3, wSel0, wSel1, wSel2, wSel3;
        Reg::RegTensor<xType> wDIntlv0, wDIntlv1;
        Reg::RegTensor<float> wCvtF32N0, wCvtF32N1, wCvtF32N2, wCvtF32N3;

        Reg::MaskReg maskRegB4 = Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();
        Reg::MaskReg maskRegB16 = Reg::CreateMask<uint16_t, AscendC::Reg::MaskPattern::ALL>();
        Reg::MaskReg maskRegVsel = Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::M4>();
        Reg::MaskReg maskWeight;
        uint32_t maskWeightTmp;

        for (uint16_t outerIdx = 0; outerIdx < p.outerExtend; ++outerIdx) {
            maskWeightTmp = p.maskWeight;
            for (uint16_t innerIdx = 0; innerIdx < p.innerExtend; ++innerIdx) {
                Reg::AddrReg addrRegScale = Reg::CreateAddrReg<scaleType>(outerIdx, p.outerStrideScale, innerIdx,
                                                                          PERGROUP_OFFSET_8);
                Reg::AddrReg addrRegWeight = Reg::CreateAddrReg<uint8_t>(outerIdx, p.outerStrideWeight, innerIdx,
                                                                         PERGROUP_OFFSET_FOR_4_BITS);
                maskWeight = Reg::UpdateMask<xType>(maskWeightTmp);
                Reg::DataCopy<scaleType, Reg::LoadDist::DIST_E2B_B16>(scaleLoad, p.scaleBaseAddr, addrRegScale);
                Reg::Interleave(scaleCompute0, scaleCompute1, scaleLoad, scaleLoad);
                Reg::DataCopy<uint8_t, Reg::LoadDist::DIST_UNPACK4_B8>(
                    (Reg::RegTensor<uint8_t>&)wLoad0, (__ubuf__ uint8_t*)p.weightInBaseAddr0, addrRegWeight);
                Reg::DataCopy<uint8_t, Reg::LoadDist::DIST_UNPACK4_B8>(
                    (Reg::RegTensor<uint8_t>&)wLoad1, (__ubuf__ uint8_t*)p.weightInBaseAddr1, addrRegWeight);

                CastWeightF4ToF16<wType, scaleType>(wCvt0, wCvt1, wLoad0, wLoad1, maskRegB4);

                Reg::Mul(wMul0, wCvt0, scaleCompute0, maskRegB16);
                Reg::Mul(wMul1, wCvt1, scaleCompute1, maskRegB16);
                Reg::Cast<float, scaleType, castTrait0>(wCvtF32N0, wMul0, maskRegB16);
                Reg::Cast<float, scaleType, castTrait1>(wCvtF32N1, wMul0, maskRegB16);
                Reg::Cast<float, scaleType, castTrait0>(wCvtF32N2, wMul1, maskRegB16);
                Reg::Cast<float, scaleType, castTrait1>(wCvtF32N3, wMul1, maskRegB16);
                Reg::Cast<xType, float, castTrait2>(wCvtB8N0, wCvtF32N0, maskRegB16);
                Reg::Cast<xType, float, castTrait3>(wCvtB8N1, wCvtF32N1, maskRegB16);
                Reg::Cast<xType, float, castTrait2>(wCvtB8N2, wCvtF32N2, maskRegB16);
                Reg::Cast<xType, float, castTrait3>(wCvtB8N3, wCvtF32N3, maskRegB16);
                Reg::Select(wSel0, wCvtB8N0, wCvtB8N1, maskRegVsel);
                Reg::Select(wSel1, wCvtB8N2, wCvtB8N3, maskRegVsel);
                Reg::DeInterleave(wDIntlv0, wDIntlv1, wSel0, wSel1);

                Reg::DataCopy<xType, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                    p.weightOutBaseAddr, wDIntlv0, p.dataBlockStride, p.repeatStride, maskWeight);
            }
            p.weightOutBaseAddr += p.outDimOffset;
        }
    }

    template <typename xType, typename wType, typename scaleType>
    static __simd_vf__ inline void DequantW4PergroupKN(DequantW4PergroupKNParams<xType, wType, scaleType> param)
    {
        Reg::RegTensor<scaleType> scaleRegCompute, scaleRegAssist, weightF16Reg0, weightF16Reg1;
        Reg::RegTensor<wType> weightInReg0, weightInReg1;
        Reg::RegTensor<float> weightF32Reg0, weightF32Reg1, weightF32Reg2, weightF32Reg3;
        Reg::RegTensor<xType> weightF8Reg0, weightF8Reg1, weightF8Reg2, weightF8Reg3, weightF8SelReg0, weightF8SelReg1;
        Reg::MaskReg scaleMaskReg = Reg::CreateMask<uint8_t>();
        Reg::MaskReg maskRegALL = Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();
        Reg::MaskReg maskRegVsel = Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::M4>();

        Reg::DataCopy(scaleMaskReg, param.scaleMaskBaseAddr);
        for (uint16_t n1Idx = 0; n1Idx < param.n1LoopNum; n1Idx++) {
            for (uint16_t groupIdx = 0; groupIdx < param.groupNumUb; groupIdx++) {
                Reg::AddrReg scaleAddrReg = Reg::CreateAddrReg<scaleType>(n1Idx, param.scaleN1Stride, groupIdx,
                                                                          param.bubNLen);
                Reg::DataCopy<scaleType, Reg::LoadDist::DIST_BLK>(scaleRegCompute, param.scaleBaseAddr0, scaleAddrReg);
                Reg::DataCopy<scaleType, Reg::LoadDist::DIST_BLK>(scaleRegAssist, param.scaleBaseAddr1, scaleAddrReg);
                Reg::Select(scaleRegCompute, scaleRegCompute, scaleRegAssist, scaleMaskReg);
                for (uint16_t vLIdx = 0; vLIdx < param.vLLoopNumInGroup; vLIdx++) {
                    Reg::AddrReg weightInAddrReg = Reg::CreateAddrReg<uint8_t>(n1Idx, param.weightInN1Stride, groupIdx,
                                                                               param.weightInGroupIdStride, vLIdx,
                                                                               PERGROUP_OFFSET_FOR_4_BITS);
                    Reg::AddrReg weightOutAddrReg = Reg::CreateAddrReg<uint8_t>(n1Idx, param.weightOutN1Stride,
                                                                                groupIdx, param.weightOutGroupIdStride,
                                                                                vLIdx, param.weightOutVlStride);
                    Reg::DataCopy<uint8_t, Reg::LoadDist::DIST_UNPACK4_B8>((Reg::RegTensor<uint8_t>&)weightInReg0,
                                                                           (__ubuf__ uint8_t*)param.weightInBaseAddr0,
                                                                           weightInAddrReg);
                    Reg::DataCopy<uint8_t, Reg::LoadDist::DIST_UNPACK4_B8>((Reg::RegTensor<uint8_t>&)weightInReg1,
                                                                           (__ubuf__ uint8_t*)param.weightInBaseAddr1,
                                                                           weightInAddrReg);
                    CastWeightF4ToF16<wType, scaleType>(weightF16Reg0, weightF16Reg1, weightInReg0, weightInReg1,
                                                        maskRegALL);
                    Reg::Mul(weightF16Reg0, weightF16Reg0, scaleRegCompute, maskRegALL);
                    Reg::Mul(weightF16Reg1, weightF16Reg1, scaleRegCompute, maskRegALL);

                    Reg::Cast<float, scaleType, castF162F32Trait0>(weightF32Reg0, weightF16Reg0, maskRegALL);
                    Reg::Cast<float, scaleType, castF162F32Trait1>(weightF32Reg1, weightF16Reg0, maskRegALL);
                    Reg::Cast<float, scaleType, castF162F32Trait0>(weightF32Reg2, weightF16Reg1, maskRegALL);
                    Reg::Cast<float, scaleType, castF162F32Trait1>(weightF32Reg3, weightF16Reg1, maskRegALL);
                    Reg::Cast<xType, float, castF322F8Trait0>(weightF8Reg0, weightF32Reg0, maskRegALL);
                    Reg::Cast<xType, float, castF322F8Trait2>(weightF8Reg1, weightF32Reg1, maskRegALL);
                    Reg::Cast<xType, float, castF322F8Trait0>(weightF8Reg2, weightF32Reg2, maskRegALL);
                    Reg::Cast<xType, float, castF322F8Trait2>(weightF8Reg3, weightF32Reg3, maskRegALL);
                    Reg::Select(weightF8SelReg0, weightF8Reg0, weightF8Reg1, maskRegVsel);
                    Reg::Select(weightF8SelReg1, weightF8Reg2, weightF8Reg3, maskRegVsel);

                    Reg::DataCopy<xType, Reg::StoreDist::DIST_PACK_B16>(param.weightOutBaseAddr, weightF8SelReg0,
                                                                        weightOutAddrReg, maskRegALL);
                    Reg::DataCopy<xType, Reg::StoreDist::DIST_PACK_B16>(param.weightOutBaseAddr + 128, weightF8SelReg1,
                                                                        weightOutAddrReg, maskRegALL);
                }
            }
        }
    }

    // ND-trans (bTrans == true) path: DN fp4 input -> ZN L1 fractal output.
    // Input: DNExt (k, n) packed FP4 whose physical n-rows are padded to
    // UB_ALIGN_SIZE_FOR_4_BITS. Scale: DNExt (kGroup, n) whose n-rows are padded
    // to 32B. Output: Weight8BitDnToZnUbLayoutPtn FP8.
    template <class InputTensor, class ScaleTensor, class OutTensor>
    __aicore__ inline static void RunNdTrans(const InputTensor& input, const ScaleTensor& scale,
                                             const OutTensor& output, const Params& params)
    {
        using ScalePattern = asc::te::get_layout_pattern<typename ScaleTensor::layout_type>;
        using OutPattern = asc::te::get_layout_pattern<typename OutTensor::layout_type>;
        static_assert(AscendC::Std::is_same_v<ScalePattern, asc::te::dn_ext_layout_ptn>,
                      "The ND-trans dequant requires a DNExt scale tensor.");
        static_assert(AscendC::Std::is_same_v<OutPattern, Blaze::Gemm::Weight8BitDnToZnUbLayoutPtn>,
                      "The ND-trans dequant requires the DnToZn converted-weight UB layout.");
        (void)params;

        using XType = asc::te::get_attribute_element_type<typename OutTensor::element_type*>;
        using WType = asc::te::get_attribute_element_type<typename InputTensor::element_type*>;
        using SType = asc::te::get_attribute_element_type<typename ScaleTensor::element_type*>;

        const int64_t kLen = AscendC::Std::get<1>(AscendC::Std::get<0>(input.layout().shape()));
        const int64_t nLen = AscendC::Std::get<1>(AscendC::Std::get<1>(input.layout().shape()));
        // DNExt (k, n): physical n-row pitch in packed-FP4 elements.
        const int64_t weightRowPitch = AscendC::Std::get<1>(AscendC::Std::get<1>(input.layout().stride()));
        // DNExt (kGroup, n): physical n-row pitch in scale elements.
        const int64_t scaleRowPitch = AscendC::Std::get<1>(AscendC::Std::get<1>(scale.layout().stride()));
        // DnToZn UB layout: k1-slab stride == dataBlockStride * C0.
        const int64_t dataBlockStride = AscendC::Std::get<1>(AscendC::Std::get<0>(output.layout().stride())) /
                                        C0_SIZE_B8;

        DequantW4Pergroup32NKParams<XType, WType, SType> vfParams;
        vfParams.outerExtend = static_cast<uint16_t>(nLen);
        vfParams.innerExtend = static_cast<uint16_t>(
            Blaze::Gemm::CeilDiv<int64_t>(Blaze::Gemm::CeilAlign<int64_t>(kLen, UB_ALIGN_SIZE_FOR_4_BITS),
                                          static_cast<int64_t>(AscendC::VECTOR_REG_WIDTH)));
        vfParams.dataBlockStride = static_cast<uint32_t>(dataBlockStride);
        vfParams.repeatStride = vfParams.dataBlockStride * static_cast<uint32_t>(OFFSET_8);
        vfParams.outDimOffset = static_cast<int32_t>(static_cast<int64_t>(AscendC::ONE_BLK_SIZE) -
                                                     static_cast<int64_t>(vfParams.innerExtend) *
                                                         static_cast<int64_t>(vfParams.repeatStride) *
                                                         static_cast<int64_t>(AscendC::ONE_BLK_SIZE));
        vfParams.outerStrideScale = static_cast<uint32_t>(scaleRowPitch);
        vfParams.outerStrideWeight = static_cast<uint32_t>(weightRowPitch >> INT4_PACK_SHIFT);
        vfParams.maskWeight = static_cast<uint32_t>(kLen);
        vfParams.scaleBaseAddr = (__ubuf__ SType*)scale.data().get();
        vfParams.weightInBaseAddr0 = (__ubuf__ int8_t*)input.data().get();
        vfParams.weightInBaseAddr1 = vfParams.weightInBaseAddr0 + OFFSET_64;
        vfParams.weightOutBaseAddr = (__ubuf__ XType*)output.data().get();
        asc_vf_call<DequantW4Pergroup32NK<XType, WType, SType>>(vfParams);
    }

    // NZ (bTrans == false) path: NZ fp4 input -> NZ L1 fractal output. Input: NZ-pattern
    // (k, n) packed FP4 with a tight n1-fractal pitch; Scale: NDExt (kGroup, n) padded to
    // the 32-aligned N size; Output: NzRowPaddingLayoutPtn FP8 with vector chunks
    // interleaved across BL1 ping-pong buffers.
    template <class InputTensor, class ScaleTensor, class OutTensor>
    __aicore__ inline static void RunNz(const InputTensor& input, const ScaleTensor& scale, const OutTensor& output,
                                        const Params& params)
    {
        using ScalePattern = asc::te::get_layout_pattern<typename ScaleTensor::layout_type>;
        using OutPattern = asc::te::get_layout_pattern<typename OutTensor::layout_type>;
        static_assert(AscendC::Std::is_same_v<ScalePattern, asc::te::nd_ext_layout_ptn>,
                      "The NZ dequant requires an NDExt scale tensor.");
        static_assert(AscendC::Std::is_same_v<OutPattern, Blaze::Gemm::NzRowPaddingLayoutPtn>,
                      "The NZ dequant requires the NzToNz converted-weight UB layout.");

        using XType = asc::te::get_attribute_element_type<typename OutTensor::element_type*>;
        using WType = asc::te::get_attribute_element_type<typename InputTensor::element_type*>;
        using SType = asc::te::get_attribute_element_type<typename ScaleTensor::element_type*>;

        const int64_t n1LoopNum = AscendC::Std::get<1>(AscendC::Std::get<1>(input.layout().shape()));
        // NDExt (kGroup, n): kGroup-row pitch in scale elements (32-aligned N size).
        const int64_t scaleRowPitch = AscendC::Std::get<1>(AscendC::Std::get<0>(scale.layout().stride()));
        // NzToNz UB layout strides: (1, groupIdStride), (vlStride, n1Stride).
        const int64_t weightOutVlStride = AscendC::Std::get<0>(AscendC::Std::get<1>(output.layout().stride()));
        const int64_t weightOutGroupIdStride = AscendC::Std::get<1>(AscendC::Std::get<0>(output.layout().stride()));
        const int64_t weightOutN1Stride = AscendC::Std::get<1>(AscendC::Std::get<1>(output.layout().stride()));

        // groupNumUb intentionally floors validK / GROUP_SIZE, matching the VF group
        // loop: a trailing partial K group is not converted.
        const uint32_t groupNumUb = params.validK / GroupSize;
        DequantW4PergroupKNParams<XType, WType, SType> vfParams;
        vfParams.groupNumUb = groupNumUb;
        vfParams.vLLoopNumInGroup = static_cast<uint32_t>(ELEMENTS_PER_GROUP * C0_SIZE_B8 /
                                                          static_cast<int64_t>(AscendC::VECTOR_REG_WIDTH));
        vfParams.n1LoopNum = static_cast<uint32_t>(n1LoopNum);
        vfParams.bubNLen = static_cast<uint32_t>(n1LoopNum * C0_SIZE_B8);
        vfParams.scaleN1Stride = static_cast<uint32_t>(scaleRowPitch / n1LoopNum);
        vfParams.weightInGroupIdStride = static_cast<uint32_t>(ELEMENTS_PER_GROUP * C0_SIZE_B8 / INT4_DTYPE_FACTOR);
        // Matches the VF's floor-based group addressing: groupIdStride * groupNumUb.
        vfParams.weightInN1Stride = vfParams.weightInGroupIdStride * groupNumUb;
        vfParams.weightOutN1Stride = static_cast<uint32_t>(weightOutN1Stride);
        vfParams.weightOutGroupIdStride = static_cast<uint32_t>(weightOutGroupIdStride);
        vfParams.weightOutVlStride = static_cast<uint32_t>(weightOutVlStride);
        vfParams.scaleBaseAddr0 = (__ubuf__ SType*)scale.data().get();
        vfParams.scaleBaseAddr1 = vfParams.scaleBaseAddr0 + Blaze::Gemm::BLOCK_CUBE;
        vfParams.scaleMaskBaseAddr = params.scaleMaskAddr;
        vfParams.weightInBaseAddr0 = (__ubuf__ int8_t*)input.data().get();
        vfParams.weightInBaseAddr1 = vfParams.weightInBaseAddr0 + OFFSET_64;
        vfParams.weightOutBaseAddr = (__ubuf__ XType*)output.data().get();
        asc_vf_call<DequantW4PergroupKN<XType, WType, SType>>(vfParams);
    }
};

} // namespace Blaze::Epilogue::Tile
