/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "k3_pvwrap.h"
#include "securec.h"
#include <thread>
uint64_t g_k3CtrlValue = 0;

void pv_init(int pv_mode, int hijack, int bLoadInstByPvwrapSelf, const char* out_dir, uint32_t core_id) {}
void pv_launch_sub_core(uint64_t pc, const char* bin_file, uint32_t sub_core_id, uint32_t core_id) {}
uint32_t pv_step(uint32_t pipe_id, uint32_t sub_core_id, uint32_t core_id)
{
    return 0;
}
void pv_reset(uint32_t sub_core_id, uint32_t core_id) {}

void set_read_record() {}

bool pv_reg_read(uint32_t reg_type, uint32_t reg_id, svOpenArrayHandle buf, uint32_t sub_core_id, uint32_t core_id)
{
    if (reg_id == 3 && reg_type == 1) { // only when reg_type is SPR then update the CTRL register
        *reinterpret_cast<uint64_t*>(buf) = g_k3CtrlValue;
    }
    return true;
}
bool pv_reg_write(uint32_t reg_type, uint32_t reg_id, svOpenArrayHandle buf, uint32_t sub_core_id, uint32_t core_id)
{
    if (reg_id == 3 && reg_type == 1) { // only when reg_type is SPR then update the CTRL register
        g_k3CtrlValue = *(reinterpret_cast<uint64_t*>(buf));
    }
    return true;
}
bool pv_mem_read(uint32_t mem_type, uint64_t addr, uint64_t size, svOpenArrayHandle buf, uint32_t sub_core_id,
    uint32_t core_id)
{
    memset_s(buf, size, 0x00, size);
    return true;
}
bool pv_mem_write(uint32_t mem_type, uint64_t addr, uint64_t size, svOpenArrayHandle buf, uint32_t sub_core_id,
    uint32_t core_id)
{
    return true;
}
