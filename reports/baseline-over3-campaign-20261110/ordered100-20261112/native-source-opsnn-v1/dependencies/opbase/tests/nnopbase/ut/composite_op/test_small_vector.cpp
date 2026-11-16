/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include "opdev/small_vector.h"
#include "opdev/fast_vector.h"
#include "gtest/gtest.h"

namespace {
using SV = op::internal::SmallVector<int, 4>;

bool ContentEqual(const SV& v, const std::vector<int>& expect)
{
    if (v.size() != expect.size()) {
        return false;
    }
    for (size_t i = 0; i < expect.size(); ++i) {
        if (v[i] != expect[i]) {
            return false;
        }
    }
    return true;
}
} // namespace

class TestSmallVector : public testing::Test {};

// Regression for issue #301: SmallVector(InputIt, InputIt) iterator-range constructor.
// Previously the guard `if (count >= 0) { return; }` was inverted, so:
//   - legal ranges produced an always-empty vector (elements never copied);
//   - reversed ranges fell through to InitStorage with a negative value cast to
//     a huge size_type, triggering an unbounded allocation.
// The empty-range path also left size_/capacity_/allocated_storage_ uninitialized.

TEST_F(TestSmallVector, IteratorConstructor_EmptyRangeHasZeroSizeAndCleanMembers)
{
    std::vector<int> empty;
    SV v(empty.begin(), empty.end());
    EXPECT_EQ(v.size(), 0UL);
    EXPECT_EQ(v.capacity(), 4UL);
}

TEST_F(TestSmallVector, IteratorConstructor_WithinInlineCapacityCopiesElements)
{
    std::vector<int> src = {1, 2, 3};
    SV v(src.begin(), src.end());
    EXPECT_TRUE(ContentEqual(v, src));
}

TEST_F(TestSmallVector, IteratorConstructor_AtInlineCapacityCopiesElements)
{
    std::vector<int> src = {1, 2, 3, 4};
    SV v(src.begin(), src.end());
    EXPECT_TRUE(ContentEqual(v, src));
}

TEST_F(TestSmallVector, IteratorConstructor_ExceedsInlineCapacitySpillsToHeap)
{
    std::vector<int> src = {10, 20, 30, 40, 50, 60};
    SV v(src.begin(), src.end());
    EXPECT_GT(v.capacity(), 4UL);
    EXPECT_TRUE(ContentEqual(v, src));
}

TEST_F(TestSmallVector, IteratorConstructor_AcceptsRawPointerRange)
{
    int arr[] = {7, 8, 9};
    SV v(std::begin(arr), std::end(arr));
    EXPECT_EQ(v.size(), 3UL);
    EXPECT_EQ(v[0], 7);
    EXPECT_EQ(v[1], 8);
    EXPECT_EQ(v[2], 9);
}

TEST_F(TestSmallVector, IteratorConstructor_ReversedRangeThrowsRangeError)
{
    std::vector<int> src = {1, 2, 3};
    EXPECT_THROW(SV(src.end(), src.begin()), std::range_error);
}

// FVector is the public alias of SmallVector (uses PoolAllocator); cover the same path.

class TestFVector : public testing::Test {};

TEST_F(TestFVector, IteratorConstructor_EmptyRangeHasZeroSize)
{
    std::vector<int> empty;
    op::FVector<int, 4> v(empty.begin(), empty.end());
    EXPECT_EQ(v.size(), 0UL);
}

TEST_F(TestFVector, IteratorConstructor_ValidRangeCopiesElements)
{
    std::vector<int> src = {10, 20, 30, 40, 50};
    op::FVector<int, 4> v(src.begin(), src.end());
    ASSERT_EQ(v.size(), src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        EXPECT_EQ(v[i], src[i]);
    }
}

TEST_F(TestFVector, IteratorConstructor_ReversedRangeThrowsRangeError)
{
    std::vector<int> src = {1, 2, 3};
    auto buildReversed = [&]() { return op::FVector<int, 4>(src.end(), src.begin()); };
    EXPECT_THROW(buildReversed(), std::range_error);
}
