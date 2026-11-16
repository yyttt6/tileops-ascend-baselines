/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <array>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <stdlib.h>

#include "opdev/op_cache_container.h"
#include "opdev/op_cache.h"

using namespace op::internal;

class OpCacheContainerUt : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

TEST_F(OpCacheContainerUt, InsertAndFind)
{
    OpCacheContainer<OpCacheKey, OpCacheValue, OpCacheKeyHash, OpCacheKeyEqual> container;
    container.init(10);

    uint8_t* buf[10] = {nullptr};

    for (int i = 0; i < 5; i++) {
        buf[i] = new uint8_t[128];
        memset_s(buf[i], 128, 0, 128);
        *(uint64_t*)buf[i] = i;
        OpCacheKey key(buf[i], 128);
        OpCacheValue value(nullptr, key);
        auto res = container.insert(value);
        EXPECT_EQ(res.second, true);
    }

    for (int i = 5; i < 10; i++) {
        buf[i] = new uint8_t[128];
        memset_s(buf[i], 128, 0, 128);
        *(uint64_t*)buf[i] = i;
        OpCacheKey key(buf[i], 128);
        OpCacheValue value(nullptr, key);
        container[key] = value;
    }
    EXPECT_EQ(container.size(), 10);
    EXPECT_EQ(container.empty(), false);

    uint64_t v = 9;
    for (auto it = container.begin(); it != container.end(); it++) {
        OpCacheValue& value = *it;
        OpCacheKey& key = dynamic_cast<OpCacheKey&>(value);
        std::cout << *(uint64_t*)key.buf << std::endl;
        EXPECT_EQ(*(uint64_t*)key.buf, v--);
    }

    v = 0;
    for (auto it = container.rbegin(); it != container.rend(); it++) {
        OpCacheValue& value = *it;
        OpCacheKey& key = dynamic_cast<OpCacheKey&>(value);
        EXPECT_EQ(*(uint64_t*)key.buf, v++);
    }

    for (int i = 0; i < 10; i++) {
        if (buf[i]) {
            delete[] buf[i];
        }
    }
}

TEST_F(OpCacheContainerUt, InsertDuplicate)
{
    OpCacheContainer<OpCacheKey, OpCacheValue, OpCacheKeyHash, OpCacheKeyEqual> container;
    container.init(10);

    uint8_t* buf[10] = {nullptr};

    buf[0] = new uint8_t[128];
    memset_s(buf[0], 128, 0, 128);
    OpCacheKey key(buf[0], 128);
    OpCacheValue value(nullptr, key);
    auto res = container.insert(value);
    EXPECT_EQ(res.second, true);

    for (int i = 1; i < 5; i++) {
        buf[i] = new uint8_t[128];
        memset_s(buf[i], 128, 0, 128);
        OpCacheKey key(buf[i], 128);
        OpCacheValue value(nullptr, key);
        auto res = container.insert(value);
        EXPECT_EQ(res.second, true);
    }

    for (int i = 5; i < 10; i++) {
        buf[i] = new uint8_t[128];
        memset_s(buf[i], 128, 0, 128);
        OpCacheKey key(buf[i], 128);
        OpCacheValue value(nullptr, key);
        container[key] = value;
    }
    EXPECT_EQ(container.size(), 10);
    EXPECT_EQ(container.empty(), false);

    for (int i = 0; i < 10; i++) {
        if (buf[i]) {
            delete[] buf[i];
        }
    }
}

TEST_F(OpCacheContainerUt, FindAndErase)
{
    OpCacheContainer<OpCacheKey, OpCacheValue, OpCacheKeyHash, OpCacheKeyEqual> container;
    container.init(10);

    uint8_t* buf[10] = {nullptr};

    for (int i = 0; i < 10; i++) {
        buf[i] = new uint8_t[128];
        memset_s(buf[i], 128, 0, 128);
        *(uint64_t*)buf[i] = i;
        OpCacheKey key(buf[i], 128);
        OpCacheValue value(nullptr, key);
        container[key] = value;
    }
    EXPECT_EQ(container.size(), 10);
    EXPECT_EQ(container.empty(), false);

    for (int i = 9; i >= 0; i--) {
        OpCacheKey key(buf[i], 128);
        auto it = container.find(key);
        EXPECT_EQ(it != container.end(), false);
    }

    for (int i = 9; i >= 0; i--) {
        if (container.rbegin() != container.rend()) {
            OpCacheValue* v = container.rbegin().operator->();
            container.erase(*v);
            OpCacheKey* key = dynamic_cast<OpCacheKey*>(v);
            delete v;
        }
    }

    for (int i = 0; i < 10; i++) {
        OpCacheKey key(buf[i], 128);
        auto it = container.find(key);
        EXPECT_EQ(it == container.end(), true);
    }
    EXPECT_EQ(container.size(), 0);
    EXPECT_EQ(container.empty(), true);

    for (int i = 0; i < 10; i++) {
        if (buf[i]) {
            delete[] buf[i];
        }
    }
}