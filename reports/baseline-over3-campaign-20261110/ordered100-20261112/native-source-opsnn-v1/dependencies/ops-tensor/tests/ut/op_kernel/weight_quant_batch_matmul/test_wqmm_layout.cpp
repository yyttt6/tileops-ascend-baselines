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
#include "gtest/gtest.h"
#include "wqmm_test_utils.h"
#include "blaze/gemm/utils/layout_struct.h"

namespace {
using namespace WeightQuantBatchMatmulUT;
template <typename T, bool Zn>
void CheckPaddingLayout(int64_t k, int64_t n, int64_t pitch)
{
    auto layout = [&] {
        if constexpr (Zn)
            return Blaze::Gemm::ZnRowPaddingUBLayout<T>{}(k, n, pitch);
        else
            return Blaze::Gemm::NzColPaddingUBLayout<T>{}(k, n, pitch);
    }();
    using Pattern = asc::te::get_layout_pattern<decltype(layout)>;
    using ExpectedPattern = AscendC::Std::conditional_t<Zn, Blaze::Gemm::zn_row_padding_layout_ptn,
                                                        Blaze::Gemm::nz_col_padding_layout_ptn>;
    static_assert(AscendC::Std::is_same_v<Pattern, ExpectedPattern>);
    auto shape = asc::te::get_shape(layout);
    EXPECT_EQ(int64_t(asc::te::get<0, 0>(shape)), Zn ? 16 : 1);
    EXPECT_EQ(int64_t(asc::te::get<0, 1>(shape)), Zn ? Align16(k) / 16 : k);
    EXPECT_EQ(int64_t(asc::te::get<1, 0>(shape)), Zn ? 1 : 16);
    EXPECT_EQ(int64_t(asc::te::get<1, 1>(shape)), Zn ? n : Align16(n) / 16);
    auto rows = Zn ? Align16(k) : k, cols = Zn ? n : Align16(n);
    for (int64_t row = 0; row < rows; ++row)
        for (int64_t col = 0; col < cols; ++col) {
            int64_t reference = Zn ? (row / 16 * pitch + col * 16 + row % 16) :
                                     (col / 16 * pitch + row * 16 + col % 16);
            ASSERT_EQ(int64_t(layout(asc::te::make_coord(row, col))), reference);
        }
}
TEST(WqmmLayoutTest, ProductionPaddingHelpersKeepFixedPitchOnTails)
{
    // Singleton, C0 boundary, partial tile and full VF tile.
    for (const auto& shape : {std::array<int64_t, 2>{1, 1}, {16, 16}, {129, 37}, {256, 64}}) {
        const auto outer = shape[0], inner = shape[1];
        CheckPaddingLayout<half, true>(outer, inner, 65 * 16);
        CheckPaddingLayout<bfloat16_t, true>(outer, inner, 65 * 16);
        CheckPaddingLayout<half, false>(inner, outer, 65 * 16);
        CheckPaddingLayout<bfloat16_t, false>(inner, outer, 65 * 16);
    }
}
} // namespace
