/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file test_tqbmm_mx.cpp
 * \brief TQBMM MX Kernel UT测试用例
 */

#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "blaze_kernel_stub.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"
#include "kernel_operator.h"

#include "tqbmm_mx.cpp"

class TqbmmMxTest : public testing::Test {
protected:
    static void TearDownTestCase()
    {
        std::string cleanCmd = std::string("cd ") + UT_KERNEL_SRC_DIR + "/tqbmm_mx/tqbmm_data && rm -rf *.bin";
        system(cleanCmd.c_str());
    }

    void SetUp() override
    {
        aGM = nullptr;
        bGM = nullptr;
        biasGM = nullptr;
        cGM = nullptr;
        scaleAGM = nullptr;
        scaleBGM = nullptr;
        tilingGM = nullptr;
    }

    void TearDown() override
    {
        if (aGM)
            AscendC::GmFree((void*)aGM);
        if (bGM)
            AscendC::GmFree((void*)bGM);
        if (biasGM)
            AscendC::GmFree((void*)biasGM);
        if (cGM)
            AscendC::GmFree((void*)cGM);
        if (scaleAGM)
            AscendC::GmFree((void*)scaleAGM);
        if (scaleBGM)
            AscendC::GmFree((void*)scaleBGM);
        if (tilingGM)
            AscendC::GmFree((void*)tilingGM);
    }

    GM_ADDR aGM;
    GM_ADDR bGM;
    GM_ADDR biasGM;
    GM_ADDR cGM;
    GM_ADDR scaleAGM;
    GM_ADDR scaleBGM;
    GM_ADDR tilingGM;
};

static constexpr uint64_t MXFP_DIVISOR_SIZE = 64UL;
static constexpr uint64_t MXFP_MULTI_BASE_SIZE = 2UL;

static uint64_t GetScaleKLen(uint64_t k)
{
    return (k + MXFP_DIVISOR_SIZE - 1) / MXFP_DIVISOR_SIZE * MXFP_MULTI_BASE_SIZE;
}

static size_t GetFp4Size(uint64_t count) { return (count + 1) / 2; }

template <typename T>
static size_t GetInputSize(uint64_t count)
{
    if constexpr (Blaze::Gemm::IsFp4<T>())
        return GetFp4Size(count);
    return static_cast<size_t>(count) * sizeof(T);
}

template <typename T>
static void ReadBinFile(const std::string& path, void* dst, size_t size)
{
    std::ifstream f(path, std::ios::binary);
    ASSERT_TRUE(f.good()) << "Failed to open " << path;
    f.read(static_cast<char*>(dst), size);
    ASSERT_TRUE(f.good()) << "Failed to read " << path;
}

static std::string GetDataDir() { return std::string(UT_KERNEL_SRC_DIR) + "/tqbmm_mx/tqbmm_data"; }

static void GenData(const std::string& dtype, uint32_t m, uint32_t n, uint32_t k, uint32_t batch,
                    bool transBatchA = false)
{
    std::string dir = GetDataDir();
    std::string cmd = "cd " + dir + " && rm -rf *.bin && python3 gen_data.py --m " + std::to_string(m) + " --n " +
                      std::to_string(n) + " --k " + std::to_string(k) + " --batch " + std::to_string(batch) +
                      " --dtype " + dtype;
    if (transBatchA) {
        cmd += " --trans_batch_a";
    }
    ASSERT_EQ(system(cmd.c_str()), 0);
}

// === MX FP8, batch=1 ===
TEST_F(TqbmmMxTest, Test_MX_FP8_Batch1)
{
    const uint32_t M = 64, N = 128, K = 128, BATCH = 1, blockNum = 1;
    uint64_t scaleKLen = GetScaleKLen(K);

    size_t aSize = GetInputSize<fp8_e4m3fn_t>(BATCH * M * K);
    size_t bSize = GetInputSize<fp8_e4m3fn_t>(BATCH * K * N);
    size_t biasSize = N * sizeof(float);
    size_t cSize = M * BATCH * N * sizeof(half);
    size_t scaleASize = BATCH * M * scaleKLen * sizeof(fp8_e8m0_t);
    size_t scaleBSize = BATCH * scaleKLen * N * sizeof(fp8_e8m0_t);

    aGM = (GM_ADDR)AscendC::GmAlloc(aSize);
    bGM = (GM_ADDR)AscendC::GmAlloc(bSize);
    biasGM = (GM_ADDR)AscendC::GmAlloc(biasSize);
    cGM = (GM_ADDR)AscendC::GmAlloc(cSize);
    scaleAGM = (GM_ADDR)AscendC::GmAlloc(scaleASize);
    scaleBGM = (GM_ADDR)AscendC::GmAlloc(scaleBSize);
    tilingGM = (GM_ADDR)AscendC::GmAlloc(sizeof(TqbmmMxTilingData));
    ASSERT_NE(aGM, nullptr);
    ASSERT_NE(bGM, nullptr);
    ASSERT_NE(cGM, nullptr);

    GenData("fp8_e4m3", M, N, K, BATCH);
    std::string dir = GetDataDir();
    ReadBinFile<fp8_e4m3fn_t>(dir + "/input_a.bin", aGM, aSize);
    ReadBinFile<fp8_e4m3fn_t>(dir + "/input_b.bin", bGM, bSize);
    ReadBinFile<fp8_e8m0_t>(dir + "/scale_a.bin", scaleAGM, scaleASize);
    ReadBinFile<fp8_e8m0_t>(dir + "/scale_b.bin", scaleBGM, scaleBSize);

    auto* td = reinterpret_cast<TqbmmMxTilingData*>(tilingGM);
    TqbmmUT::FillTqbmmMxTilingDataDefault(*td, M, N, K, BATCH, blockNum);

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto kf = tqbmm_mx_kernel_entry<OP_TYPE_TQBMM_MX_BASIC, fp8_e4m3fn_t, fp8_e4m3fn_t, half, float>;
    ASSERT_TRUE(KERNEL_RUN_KF(kf, blockNum, aGM, bGM, biasGM, scaleAGM, scaleBGM, cGM, tilingGM))
        << "TQBMM MX FP8 batch=1 kernel execution failed";
}

// === MX FP8, batch=2, trans batch A ===
TEST_F(TqbmmMxTest, Test_MX_FP8_Batch2_TransBatchA)
{
    const uint32_t M = 64, N = 128, K = 128, BATCH = 2, blockNum = 1;
    uint64_t scaleKLen = GetScaleKLen(K);

    size_t aSize = M * BATCH * K * sizeof(fp8_e4m3fn_t);
    size_t bSize = BATCH * K * N * sizeof(fp8_e4m3fn_t);
    size_t biasSize = N * sizeof(float);
    size_t cSize = M * BATCH * N * sizeof(half);
    size_t scaleASize = M * BATCH * scaleKLen * sizeof(fp8_e8m0_t);
    size_t scaleBSize = BATCH * scaleKLen * N * sizeof(fp8_e8m0_t);

    aGM = (GM_ADDR)AscendC::GmAlloc(aSize);
    bGM = (GM_ADDR)AscendC::GmAlloc(bSize);
    biasGM = (GM_ADDR)AscendC::GmAlloc(biasSize);
    cGM = (GM_ADDR)AscendC::GmAlloc(cSize);
    scaleAGM = (GM_ADDR)AscendC::GmAlloc(scaleASize);
    scaleBGM = (GM_ADDR)AscendC::GmAlloc(scaleBSize);
    tilingGM = (GM_ADDR)AscendC::GmAlloc(sizeof(TqbmmMxTilingData));
    ASSERT_NE(aGM, nullptr);
    ASSERT_NE(bGM, nullptr);
    ASSERT_NE(cGM, nullptr);

    GenData("fp8_e4m3", M, N, K, BATCH, true);
    std::string dir = GetDataDir();
    ReadBinFile<fp8_e4m3fn_t>(dir + "/input_a.bin", aGM, aSize);
    ReadBinFile<fp8_e4m3fn_t>(dir + "/input_b.bin", bGM, bSize);
    ReadBinFile<fp8_e8m0_t>(dir + "/scale_a.bin", scaleAGM, scaleASize);
    ReadBinFile<fp8_e8m0_t>(dir + "/scale_b.bin", scaleBGM, scaleBSize);

    auto* td = reinterpret_cast<TqbmmMxTilingData*>(tilingGM);
    TqbmmUT::FillTqbmmMxTilingDataDefault(*td, M, N, K, BATCH, blockNum);

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto kf = tqbmm_mx_kernel_entry<OP_TYPE_TQBMM_MX_TRANS_BATCH_A, fp8_e4m3fn_t, fp8_e4m3fn_t, half, float,
                                    static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_PERM_X1)>;
    ASSERT_TRUE(KERNEL_RUN_KF(kf, blockNum, aGM, bGM, biasGM, scaleAGM, scaleBGM, cGM, tilingGM))
        << "TQBMM MX FP8 batch=2 trans-batch-A kernel execution failed";
}

// === MX FP4, batch=1 ===
TEST_F(TqbmmMxTest, Test_MX_FP4_Batch1)
{
    const uint32_t M = 128, N = 256, K = 128, BATCH = 1, blockNum = 1;
    uint64_t scaleKLen = GetScaleKLen(K);

    size_t aSize = GetInputSize<fp4x2_e2m1_t>(BATCH * M * K);
    size_t bSize = GetInputSize<fp4x2_e2m1_t>(BATCH * K * N);
    size_t biasSize = N * sizeof(float);
    size_t cSize = M * BATCH * N * sizeof(half);
    size_t scaleASize = BATCH * M * scaleKLen * sizeof(fp8_e8m0_t);
    size_t scaleBSize = BATCH * scaleKLen * N * sizeof(fp8_e8m0_t);

    aGM = (GM_ADDR)AscendC::GmAlloc(aSize);
    bGM = (GM_ADDR)AscendC::GmAlloc(bSize);
    biasGM = (GM_ADDR)AscendC::GmAlloc(biasSize);
    cGM = (GM_ADDR)AscendC::GmAlloc(cSize);
    scaleAGM = (GM_ADDR)AscendC::GmAlloc(scaleASize);
    scaleBGM = (GM_ADDR)AscendC::GmAlloc(scaleBSize);
    tilingGM = (GM_ADDR)AscendC::GmAlloc(sizeof(TqbmmMxTilingData));
    ASSERT_NE(aGM, nullptr);
    ASSERT_NE(bGM, nullptr);
    ASSERT_NE(cGM, nullptr);

    GenData("fp4_e2m1", M, N, K, BATCH);
    std::string dir = GetDataDir();
    ReadBinFile<fp4x2_e2m1_t>(dir + "/input_a.bin", aGM, aSize);
    ReadBinFile<fp4x2_e2m1_t>(dir + "/input_b.bin", bGM, bSize);
    ReadBinFile<fp8_e8m0_t>(dir + "/scale_a.bin", scaleAGM, scaleASize);
    ReadBinFile<fp8_e8m0_t>(dir + "/scale_b.bin", scaleBGM, scaleBSize);

    auto* td = reinterpret_cast<TqbmmMxTilingData*>(tilingGM);
    TqbmmUT::FillTqbmmMxTilingDataDefault(*td, M, N, K, BATCH, blockNum);

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto kf = tqbmm_mx_kernel_entry<OP_TYPE_TQBMM_MX_BASIC, fp4x2_e2m1_t, fp4x2_e2m1_t, half, float>;
    ASSERT_TRUE(KERNEL_RUN_KF(kf, blockNum, aGM, bGM, biasGM, scaleAGM, scaleBGM, cGM, tilingGM))
        << "TQBMM MX FP4 batch=1 kernel execution failed";
}

// === MX FP4, batch=2, trans batch A (fp4 + perm_x1 + B>=2) ===
// Covers the fp4 A row-stride fix (no double SIZE_SHIFT) and the [M,B,N] C
// layout fix (B stride = N, M stride = B*N). Without the fixes this case
// misaligns rows M>=1 / batch>=1 while the kernel still reports [SUCCESS].
TEST_F(TqbmmMxTest, Test_MX_FP4_Batch2_TransBatchA)
{
    const uint32_t M = 64, N = 64, K = 128, BATCH = 2, blockNum = 1;
    uint64_t scaleKLen = GetScaleKLen(K);

    // A is stored [M, B, K] for trans_batch_a (perm_x1=[1,0,2]).
    size_t aSize = GetInputSize<fp4x2_e2m1_t>(BATCH * M * K);
    size_t bSize = GetInputSize<fp4x2_e2m1_t>(BATCH * K * N);
    size_t biasSize = N * sizeof(float);
    size_t cSize = M * BATCH * N * sizeof(half);
    size_t scaleASize = M * BATCH * scaleKLen * sizeof(fp8_e8m0_t);
    size_t scaleBSize = BATCH * scaleKLen * N * sizeof(fp8_e8m0_t);

    aGM = (GM_ADDR)AscendC::GmAlloc(aSize);
    bGM = (GM_ADDR)AscendC::GmAlloc(bSize);
    biasGM = (GM_ADDR)AscendC::GmAlloc(biasSize);
    cGM = (GM_ADDR)AscendC::GmAlloc(cSize);
    scaleAGM = (GM_ADDR)AscendC::GmAlloc(scaleASize);
    scaleBGM = (GM_ADDR)AscendC::GmAlloc(scaleBSize);
    tilingGM = (GM_ADDR)AscendC::GmAlloc(sizeof(TqbmmMxTilingData));
    ASSERT_NE(aGM, nullptr);
    ASSERT_NE(bGM, nullptr);
    ASSERT_NE(cGM, nullptr);

    GenData("fp4_e2m1", M, N, K, BATCH, true);
    std::string dir = GetDataDir();
    ReadBinFile<fp4x2_e2m1_t>(dir + "/input_a.bin", aGM, aSize);
    ReadBinFile<fp4x2_e2m1_t>(dir + "/input_b.bin", bGM, bSize);
    ReadBinFile<fp8_e8m0_t>(dir + "/scale_a.bin", scaleAGM, scaleASize);
    ReadBinFile<fp8_e8m0_t>(dir + "/scale_b.bin", scaleBGM, scaleBSize);

    auto* td = reinterpret_cast<TqbmmMxTilingData*>(tilingGM);
    TqbmmUT::FillTqbmmMxTilingDataDefault(*td, M, N, K, BATCH, blockNum);

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto kf = tqbmm_mx_kernel_entry<OP_TYPE_TQBMM_MX_TRANS_BATCH_A, fp4x2_e2m1_t, fp4x2_e2m1_t, half, float,
                                    static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_PERM_X1)>;
    ASSERT_TRUE(KERNEL_RUN_KF(kf, blockNum, aGM, bGM, biasGM, scaleAGM, scaleBGM, cGM, tilingGM))
        << "TQBMM MX FP4 batch=2 trans-batch-A kernel execution failed";
}

// === MX FP8, batch=4, multi-core ===
TEST_F(TqbmmMxTest, Test_MX_FP8_Batch4_MultiCore)
{
    const uint32_t M = 32, N = 32, K = 64, BATCH = 4, blockNum = 4;
    uint64_t scaleKLen = GetScaleKLen(K);

    size_t aSize = GetInputSize<fp8_e4m3fn_t>(BATCH * M * K);
    size_t bSize = GetInputSize<fp8_e4m3fn_t>(BATCH * K * N);
    size_t biasSize = N * sizeof(float);
    size_t cSize = M * BATCH * N * sizeof(half);
    size_t scaleASize = BATCH * M * scaleKLen * sizeof(fp8_e8m0_t);
    size_t scaleBSize = BATCH * scaleKLen * N * sizeof(fp8_e8m0_t);

    aGM = (GM_ADDR)AscendC::GmAlloc(aSize);
    bGM = (GM_ADDR)AscendC::GmAlloc(bSize);
    biasGM = (GM_ADDR)AscendC::GmAlloc(biasSize);
    cGM = (GM_ADDR)AscendC::GmAlloc(cSize);
    scaleAGM = (GM_ADDR)AscendC::GmAlloc(scaleASize);
    scaleBGM = (GM_ADDR)AscendC::GmAlloc(scaleBSize);
    tilingGM = (GM_ADDR)AscendC::GmAlloc(sizeof(TqbmmMxTilingData));
    ASSERT_NE(aGM, nullptr);
    ASSERT_NE(bGM, nullptr);
    ASSERT_NE(cGM, nullptr);

    GenData("fp8_e4m3", M, N, K, BATCH);
    std::string dir = GetDataDir();
    ReadBinFile<fp8_e4m3fn_t>(dir + "/input_a.bin", aGM, aSize);
    ReadBinFile<fp8_e4m3fn_t>(dir + "/input_b.bin", bGM, bSize);
    ReadBinFile<fp8_e8m0_t>(dir + "/scale_a.bin", scaleAGM, scaleASize);
    ReadBinFile<fp8_e8m0_t>(dir + "/scale_b.bin", scaleBGM, scaleBSize);

    auto* td = reinterpret_cast<TqbmmMxTilingData*>(tilingGM);
    TqbmmUT::FillTqbmmMxTilingDataDefault(*td, M, N, K, BATCH, blockNum);

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto kf = tqbmm_mx_kernel_entry<OP_TYPE_TQBMM_MX_BASIC, fp8_e4m3fn_t, fp8_e4m3fn_t, half, float>;
    ASSERT_TRUE(KERNEL_RUN_KF(kf, blockNum, aGM, bGM, biasGM, scaleAGM, scaleBGM, cGM, tilingGM))
        << "TQBMM MX FP8 batch=4 multi-core kernel execution failed";
}
