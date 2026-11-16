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
#include "kernel_common.h"
#include "kernel_tpipe_impl.h"
#include "stub_reg.h"

int32_t main(int32_t argc, char** argv)
{
    AscendC::StubInit();
    AscendC::TPipe tpipe;
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

__aicore__ AscendC::TPipe* GetTPipePtr()
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201 || __NPU_ARCH__ == 3510)
#ifdef SPLIT_CORE_CUBE
    return g_cubeTPipePtr;
#elif defined(SPLIT_CORE_VEC)
    return g_vecTPipePtr;
#else
    return g_tPipePtr;
#endif
#else
    return g_tPipePtr;
#endif
    ASSERT(false && "Only supported ascend910B1, ascend910, ascend310p");
    return nullptr;
}