/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include "executor/indv_bininfo.h"
#include "executor/indv_collector.h"
#include "executor/indv_executor.h"
#include "executor/indv_compute_node_info.h"
#include "utils/file_faker.h"
#include "individual_op_api.h"
#include "depends/op/op_stub.h"
#include "depends/profiler/profiler_stub.h"

class NnopbaseExecutorIOTest : public testing::Test {
protected:
    void SetUp() { setenv("ASCEND_C", "1", 1); }
    void TearDown() { unsetenv("ASCEND_C"); }
};

/**********InitIoCachesUT**********/
void NnopbaseExecutorUnitTestIoCachesTensors(char* inputDesc, uint32_t num, NnopbaseTensors* tensors)
{
    uint32_t dynamicCnt = std::count(inputDesc, inputDesc + num, 2);
    uint32_t requiredCnt = std::count(inputDesc, inputDesc + num, 1);
    uint32_t optionCnt = std::count(inputDesc, inputDesc + num, 0);
    ASSERT_EQ(tensors->hostInputSize, 0);
    ASSERT_EQ(tensors->expectIndex, 0);
    ASSERT_EQ(tensors->hostInputNum, 0);
    ASSERT_EQ(tensors->hasDynamic, (dynamicCnt > 0 ? true : false));
    ASSERT_EQ(tensors->num, num - dynamicCnt); // 统计非动态输入，即0和1的值
    ASSERT_EQ(tensors->nonDynamicCnt, tensors->num);
    ASSERT_EQ(tensors->requiredCnt, requiredCnt); // 统计必选输入，即1的值
    ASSERT_EQ(tensors->arrayLen, (tensors->hasDynamic ? (NNOPBASE_NORM_DEF_IO_NUMS + num) : num));
    ASSERT_EQ(tensors->usedNum, (tensors->hasDynamic ? 0 : num));
    ASSERT_EQ(tensors->paramDescs.count, num);
    for (uint32_t i = 0; i < num; i++) {
        ASSERT_EQ(tensors->paramDescs.instances[i].num, (inputDesc[i] == 2 ? 0 : 1));
        ASSERT_EQ(tensors->paramDescs.instances[i].cfgNum, inputDesc[i]);
        ASSERT_EQ(tensors->paramDescs.instances[i].startIndex, (tensors->hasDynamic ? 0 : i));
    }
    for (uint32_t i = 0; i < tensors->arrayLen; i++) {
        ASSERT_EQ(tensors->extTensors[i].isNull, true);
        if (tensors->hasDynamic) {
            ASSERT_EQ(tensors->extTensors[i].isRequired, false); // 有动态输入，全为false
        } else {
            ASSERT_EQ(tensors->extTensors[i].isRequired, (inputDesc[i] == 1 ? true : false)); // 无动态输入，根据输入
        }
    }
    return;
}

TEST_F(NnopbaseExecutorIOTest, ExecutorInitIoCachesRequiredInput)
{
    char inputDesc[] = {1, 1, 1};
    uint32_t num = sizeof(inputDesc) / sizeof(char);
    NnopbaseTensors tensors;
    auto ret = NnopbaseExecutorInitIoCaches(&tensors, inputDesc, num);
    ASSERT_EQ(ret, OK);
    NnopbaseExecutorUnitTestIoCachesTensors(inputDesc, num, &tensors);
}

TEST_F(NnopbaseExecutorIOTest, ExecutorInitIoCachesOptionInputTest)
{
    char inputDesc[] = {0, 0, 0};
    uint32_t num = sizeof(inputDesc) / sizeof(char);
    NnopbaseTensors tensors;
    auto ret = NnopbaseExecutorInitIoCaches(&tensors, inputDesc, num);
    ASSERT_EQ(ret, OK);
    NnopbaseExecutorUnitTestIoCachesTensors(inputDesc, num, &tensors);
}

TEST_F(NnopbaseExecutorIOTest, ExecutorInitIoCachesDynamicInput)
{
    char inputDesc[] = {2, 2, 2};
    uint32_t num = sizeof(inputDesc) / sizeof(char);
    NnopbaseTensors tensors;
    auto ret = NnopbaseExecutorInitIoCaches(&tensors, inputDesc, num);
    ASSERT_EQ(ret, OK);
    NnopbaseExecutorUnitTestIoCachesTensors(inputDesc, num, &tensors);
}

TEST_F(NnopbaseExecutorIOTest, ExecutorInitIoCachesMixedInput01)
{
    char inputDesc[] = {0, 1, 2};
    uint32_t num = sizeof(inputDesc) / sizeof(char);
    NnopbaseTensors tensors;
    auto ret = NnopbaseExecutorInitIoCaches(&tensors, inputDesc, num);
    ASSERT_EQ(ret, OK);
    NnopbaseExecutorUnitTestIoCachesTensors(inputDesc, num, &tensors);
}

TEST_F(NnopbaseExecutorIOTest, ExecutorInitIoCachesMixedInput02)
{
    char inputDesc[] = {0, 2, 1};
    uint32_t num = sizeof(inputDesc) / sizeof(char);
    NnopbaseTensors tensors;
    auto ret = NnopbaseExecutorInitIoCaches(&tensors, inputDesc, num);
    ASSERT_EQ(ret, OK);
    NnopbaseExecutorUnitTestIoCachesTensors(inputDesc, num, &tensors);
}

TEST_F(NnopbaseExecutorIOTest, ExecutorInitIoCachesMixedInput03)
{
    char inputDesc[] = {1, 0, 2};
    uint32_t num = sizeof(inputDesc) / sizeof(char);
    NnopbaseTensors tensors;
    auto ret = NnopbaseExecutorInitIoCaches(&tensors, inputDesc, num);
    ASSERT_EQ(ret, OK);
    NnopbaseExecutorUnitTestIoCachesTensors(inputDesc, num, &tensors);
}

TEST_F(NnopbaseExecutorIOTest, ExecutorInitIoCachesMixedInput04)
{
    char inputDesc[] = {1, 2, 0};
    uint32_t num = sizeof(inputDesc) / sizeof(char);
    NnopbaseTensors tensors;
    auto ret = NnopbaseExecutorInitIoCaches(&tensors, inputDesc, num);
    ASSERT_EQ(ret, OK);
    NnopbaseExecutorUnitTestIoCachesTensors(inputDesc, num, &tensors);
}

TEST_F(NnopbaseExecutorIOTest, ExecutorInitIoCachesMixedInput05)
{
    char inputDesc[] = {2, 0, 1};
    uint32_t num = sizeof(inputDesc) / sizeof(char);
    NnopbaseTensors tensors;
    auto ret = NnopbaseExecutorInitIoCaches(&tensors, inputDesc, num);
    ASSERT_EQ(ret, OK);
    NnopbaseExecutorUnitTestIoCachesTensors(inputDesc, num, &tensors);
}

TEST_F(NnopbaseExecutorIOTest, ExecutorInitIoCachesMixedInput06)
{
    char inputDesc[] = {2, 1, 0};
    uint32_t num = sizeof(inputDesc) / sizeof(char);
    NnopbaseTensors tensors;
    auto ret = NnopbaseExecutorInitIoCaches(&tensors, inputDesc, num);
    ASSERT_EQ(ret, OK);
    NnopbaseExecutorUnitTestIoCachesTensors(inputDesc, num, &tensors);
}

TEST_F(NnopbaseExecutorIOTest, ExecutorExtendIoCaches01)
{
    char inputDesc[30] = {2, 1, 0};
    uint32_t num = sizeof(inputDesc) / sizeof(char);
    NnopbaseTensors tensors;
    auto ret = NnopbaseExecutorInitIoCaches(&tensors, inputDesc, num);
    ASSERT_EQ(ret, OK);
    NnopbaseExecutorUnitTestIoCachesTensors(inputDesc, num, &tensors);
    NnopbaseExecutorExtendIoCaches(&tensors);
    ASSERT_EQ(tensors.arrayLen, 58);
    for (uint32_t i = num + 20; i < tensors.arrayLen; i++) {
        ASSERT_EQ(tensors.extTensors[i].isNull, true);
        ASSERT_EQ(tensors.extTensors[i].isRequired, false);
        ASSERT_EQ(tensors.extTensors[i].isOptional, false);
        ASSERT_EQ(tensors.extTensors[i].valueDepend, false);
    }
}

TEST_F(NnopbaseExecutorIOTest, ExecutorIoCachesMixedInputSkipOptionalInput)
{
    NnopbaseExecutor* executor = new NnopbaseExecutor;
    ASSERT_NE(executor, nullptr);

    char inputDesc[] = {0, 1, 0, 2};
    char outputDesc[] = {};
    char attrDesc[] = {};
    auto ret = NnopbaseExecutorInit(
        executor, {inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc, sizeof(outputDesc) / sizeof(char), attrDesc,
                   sizeof(attrDesc) / sizeof(char)});
    ASSERT_EQ(ret, OK);
    uint32_t num = sizeof(inputDesc) / sizeof(char);
    ret = NnopbaseExecutorInitIoCaches(&(executor->ownArgs.inputs), inputDesc, num);
    ASSERT_EQ(ret, OK);
    op::Shape shape({1, 1, 1, 1, 1});
    aclTensor* tensor = new aclTensor(shape, shape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND,
                                      op::Format::FORMAT_ND);
    ASSERT_EQ(NnopbaseExecutorUpdateTensorsIndex(&(executor->ownArgs.inputs), 1), OK);
    ASSERT_EQ(NnopbaseExecutorAddTensor(executor, tensor, 1, true, false), OK);

    std::vector<const aclTensor*> tensor_list_a;
    tensor_list_a.push_back(tensor);
    aclTensorList* aclTensorTestList = aclCreateTensorList(tensor_list_a.data(), tensor_list_a.size());
    ASSERT_EQ(NnopbaseExecutorAddDynamicTensors(executor, aclTensorTestList, 3, true), OK);
    ASSERT_EQ(executor->ownArgs.inputs.usedNum, 4);
    NnopbaseExecutorDeInit(executor);
    delete executor;
    aclDestroyTensorList((const aclTensorList*)aclTensorTestList);
}

TEST_F(NnopbaseExecutorIOTest, ExecutorIoCachesMixedInputDynamicTensorsNull)
{
    NnopbaseExecutor* executor = new NnopbaseExecutor;
    ASSERT_NE(executor, nullptr);

    char inputDesc[] = {};
    char outputDesc[] = {2};
    char attrDesc[] = {};
    auto ret = NnopbaseExecutorInit(
        executor, {inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc, sizeof(outputDesc) / sizeof(char), attrDesc,
                   sizeof(attrDesc) / sizeof(char)});
    ASSERT_EQ(ret, OK);
    uint32_t num = sizeof(outputDesc) / sizeof(char);
    ret = NnopbaseExecutorInitIoCaches(&(executor->ownArgs.inputs), outputDesc, num);
    ASSERT_EQ(ret, OK);
    ASSERT_EQ(NnopbaseExecutorAddDynamicTensors(executor, nullptr, 0, true), OK);
    ASSERT_EQ(executor->ownArgs.inputs.usedNum, 0);
    NnopbaseExecutorDeInit(executor);
    delete executor;
}