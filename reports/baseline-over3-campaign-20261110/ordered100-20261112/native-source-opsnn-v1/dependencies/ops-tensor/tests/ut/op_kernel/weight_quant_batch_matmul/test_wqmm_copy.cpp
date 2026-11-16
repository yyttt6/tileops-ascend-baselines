/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <string>
#include "gtest/gtest.h"
#include "wqmm_test_utils.h"
#include "wqmm_copy_observer.h"
#include "blaze/gemm/tile/arch35/copy_ub_to_l1.h"

namespace {
using namespace WeightQuantBatchMatmulUT;
using Axes = std::array<int64_t, 4>;
struct CopyCase {
    std::string name;
    bool zn;
    bool bf16;
    Axes shape, srcStride, dstShape, dstStride;
};
int64_t Offset(const Axes& shape, const Axes& stride, int64_t row, int64_t col)
{
    return row % shape[0] * stride[0] + row / shape[0] * stride[1] + col % shape[2] * stride[2] +
           col / shape[2] * stride[3];
}
size_t Extent(const Axes& shape, const Axes& stride)
{
    int64_t size = 1;
    for (size_t i = 0; i < shape.size(); ++i)
        size += (shape[i] - 1) * stride[i];
    return size;
}
auto Nested(const Axes& a)
{
    return asc::te::make_shape(asc::te::make_shape(a[0], a[1]), asc::te::make_shape(a[2], a[3]));
}
struct CopyArgs {
    Axes shape, srcStride, dstShape, dstStride;
    uint64_t srcWords, dstWords;
};
// The common UT links k3_pvwrap.cpp with empty PV memory operations. Observe
// the actual SDK instruction instead of claiming to validate simulated DMA.
template <typename T, bool Zn>
__global__ __aicore__ void CopyKernel(GM_ADDR config, GM_ADDR captured)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIV {
        if (AscendC::GetSubBlockIdx() != 0)
            return;
        const auto& a = *reinterpret_cast<const CopyArgs*>(config);
        auto* record = reinterpret_cast<CopyInstruction*>(captured);
        observedCopy = record;
        using Pattern = AscendC::Std::conditional_t<Zn, Blaze::Gemm::zn_row_padding_layout_ptn,
                                                    Blaze::Gemm::nz_col_padding_layout_ptn>;
        using Destination = AscendC::Std::conditional_t<Zn, asc::te::zn_layout_ptn, asc::te::nz_layout_ptn>;
        auto sl = asc::te::make_pattern_layout<Pattern, asc::te::layout_trait_default<T>>(Nested(a.shape),
                                                                                          Nested(a.srcStride));
        auto dl = asc::te::make_pattern_layout<Destination, asc::te::layout_trait_default<T>>(Nested(a.dstShape),
                                                                                              Nested(a.dstStride));
        AscendC::LocalTensor<uint16_t> ub(AscendC::TPosition::VECCALC, 0, a.srcWords);
        AscendC::LocalTensor<uint16_t> l1(AscendC::TPosition::A1, 0, a.dstWords);
        auto* srcBase = reinterpret_cast<T*>(ub.GetPhyAddr());
        auto* dstBase = reinterpret_cast<T*>(l1.GetPhyAddr());
        auto src = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub>(srcBase + 32), sl);
        auto dst = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1>(dstBase + 48), dl);
        auto copyUB2L1 = asc::te::make_copy(Blaze::Gemm::Tile::CopyPaddedUBToL1{});
        asc::te::copy(copyUB2L1, dst, src);
        observedCopy = nullptr;
        record->src -= reinterpret_cast<uintptr_t>(srcBase);
        record->dst -= reinterpret_cast<uintptr_t>(dstBase);
    }
}
std::vector<CopyCase> Cases()
{
    std::vector<CopyCase> base;
    auto zn = [&](const char* name, int64_t k, int64_t n, int64_t sp, int64_t dn) {
        base.push_back({name,
                        true,
                        false,
                        {16, Align16(k) / 16, 1, n},
                        {1, sp * 16, 0, 16},
                        {16, Align16(k) / 16, 16, Align16(dn) / 16},
                        {1, Align16(dn) * 16, 16, 256}});
    };
    auto nz = [&](const char* name, int64_t k, int64_t n, int64_t sp, int64_t dk) {
        base.push_back({name,
                        false,
                        false,
                        {1, k, 16, Align16(n) / 16},
                        {0, 16, 1, sp * 16},
                        {16, Align16(dk) / 16, 16, Align16(n) / 16},
                        {16, 256, 1, Align16(dk) * 16}});
    };
    zn("zn_singleton", 1, 1, 2, 16);
    nz("nz_singleton", 1, 1, 2, 16);
    zn("zn_vf", 256, 64, 65, 256);
    zn("zn_vf_tail", 129, 37, 65, 256);
    nz("nz_vf", 64, 256, 65, 256);
    nz("nz_vf_tail", 37, 129, 65, 256);
    base.push_back({"zn_blocked", true, false, {16, 2, 16, 2}, {1, 544, 16, 256}, {16, 2, 16, 2}, {1, 576, 16, 256}});
    base.push_back({"nz_blocked", false, false, {16, 2, 16, 2}, {16, 256, 1, 544}, {16, 2, 16, 2}, {16, 256, 1, 576}});
    std::vector<CopyCase> all;
    for (auto c : base)
        for (bool bf16 : {false, true}) {
            c.bf16 = bf16;
            auto named = c;
            named.name += bf16 ? "_bf16" : "_fp16";
            all.push_back(named);
        }
    return all;
}
class WqmmCopyTest : public testing::TestWithParam<CopyCase> {};
TEST_P(WqmmCopyTest, InstructionMatchesCoordinatesAndSentinelModel)
{
    const auto& c = GetParam();
    CopyArgs a{c.shape,
               c.srcStride,
               c.dstShape,
               c.dstStride,
               32 + Extent(c.shape, c.srcStride) + 32,
               48 + Extent(c.dstShape, c.dstStride) + 32};
    std::vector<uint16_t> src(a.srcWords, 0xCDCD), initial(a.dstWords, 0xA5A5), expected(initial);
    std::vector<bool> written(a.dstWords, false);
    for (int64_t row = 0; row < c.shape[0] * c.shape[1]; ++row) {
        for (int64_t col = 0; col < c.shape[2] * c.shape[3]; ++col) {
            size_t si = 32 + Offset(c.shape, c.srcStride, row, col),
                   di = 48 + Offset(c.dstShape, c.dstStride, row, col);
            ASSERT_LT(si, src.size() - 32);
            ASSERT_LT(di, expected.size() - 32);
            ASSERT_FALSE(written[di]);
            written[di] = true;
            src[si] = expected[di] = static_cast<uint16_t>(row * 257 + col * 17 + 0x1357);
        }
    }
    GmBuffer config(sizeof(a)), capture(sizeof(CopyInstruction));
    config.Set(std::vector<CopyArgs>{a});
    using Kernel = decltype(&CopyKernel<half, true>);
    Kernel fn = c.bf16 ? (c.zn ? &CopyKernel<bfloat16_t, true> : &CopyKernel<bfloat16_t, false>) :
                         (c.zn ? &CopyKernel<half, true> : &CopyKernel<half, false>);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ASSERT_TRUE(KERNEL_RUN_KF(fn, 1U, config.Get(), capture.Get()));
    const auto& instruction = *reinterpret_cast<const CopyInstruction*>(capture.Get());
    ASSERT_EQ(instruction.calls, 1U)
        << "UB-to-L1 observer must intercept exactly one instruction; check SDK ABI and --wrap";
    ASSERT_EQ(instruction.src, 64U);
    ASSERT_EQ(instruction.dst, 96U);
    ASSERT_EQ(instruction.sid, 0);
    // Derive instruction geometry from independent logical-coordinate walks.
    const int64_t groups = c.zn ? c.shape[1] : c.shape[3];
    const int64_t rows = c.zn ? c.shape[0] : c.shape[0] * c.shape[1];
    const int64_t cols = c.zn ? c.shape[2] * c.shape[3] : c.shape[2];
    const auto sourcePitch = Offset(c.shape, c.srcStride, c.zn ? rows : 0, c.zn ? 0 : cols);
    const auto targetPitch = Offset(c.dstShape, c.dstStride, c.zn ? rows : 0, c.zn ? 0 : cols);
    ASSERT_EQ(instruction.count, groups);
    ASSERT_EQ(instruction.length * 16, rows * cols);
    ASSERT_EQ((instruction.length + instruction.srcGap) * 16, sourcePitch);
    ASSERT_EQ((instruction.length + instruction.dstGap) * 16, targetPitch);
    // This is an explicit instruction address model, not CPU-DMA readback.
    auto actual = initial;
    for (uint64_t group = 0; group < instruction.count; ++group) {
        size_t si = instruction.src / 2 + group * (instruction.length + instruction.srcGap) * 16;
        size_t di = instruction.dst / 2 + group * (instruction.length + instruction.dstGap) * 16;
        size_t words = instruction.length * 16;
        ASSERT_LE(si + words, src.size() - 32);
        ASSERT_LE(di + words, actual.size() - 32);
        std::copy_n(src.begin() + si, words, actual.begin() + di);
    }
    auto mismatch = std::mismatch(expected.begin(), expected.end(), actual.begin());
    if (mismatch.first != expected.end()) {
        FAIL() << "destination word " << (mismatch.first - expected.begin()) << ": expected " << *mismatch.first
               << ", actual " << *mismatch.second;
    }
}
INSTANTIATE_TEST_SUITE_P(PaddedLayouts, WqmmCopyTest, testing::ValuesIn(Cases()),
                         [](const testing::TestParamInfo<CopyCase>& p) { return p.param.name; });
} // namespace
