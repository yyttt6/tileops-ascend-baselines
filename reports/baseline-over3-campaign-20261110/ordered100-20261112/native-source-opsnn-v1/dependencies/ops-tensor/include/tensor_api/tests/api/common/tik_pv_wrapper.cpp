/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "tik_pv_wrapper.h"
#include "securec.h"
#include <thread>

PVMemAccessList pv_mem_access_list;
uint64_t g_tikCtrlValue = 0;
void set_config_path(const char* path) {}

PVEnv* pv_create(uint32_t chip_version, const char* path, uint32_t core_id)
{
    PVEnv* p_env = new (std::nothrow) PVEnv();
    return p_env;
}

void pv_destroy(PVEnv* p_env)
{
    if (p_env != nullptr) {
        delete p_env;
        p_env = nullptr;
    }
}

// for the instructions below: re-turn 0 means success, re-turn 1 means fail.
int32_t pv_step(PVEnv* p_env, uint32_t instr)
{
    return 0;
}

int32_t pv_read_gpr_register(PVEnv* p_env, uint64_t register_name, uint64_t* value)
{
    return 0;
}

int32_t pv_write_gpr_register(PVEnv* p_env, uint64_t register_name, uint64_t value)
{
    return 0;
}

int32_t pv_read_spr_register(PVEnv* p_env, const char* register_name, uint64_t* value)
{
    int32_t ret = strcmp(register_name, "CTRL");
    if (ret == 0) {
        *value = g_tikCtrlValue;
    }
    return 0;
}

int32_t pv_write_spr_register(PVEnv* p_env, const char* register_name, uint64_t value)
{
    int32_t ret = strcmp(register_name, "CTRL");
    if (ret == 0) {
        g_tikCtrlValue = value;
    }
    return 0;
}

// buffer is managed by caller
// len is length of buffer in bytes
int32_t pv_read_memory(PVEnv* p_env, int64_t addr, int32_t scope, void* buffer, int32_t len)
{
    memset_s(buffer, len, 0x00, len);
    return 0;
}

int32_t pv_write_memory(PVEnv* p_env, int64_t addr, int32_t scope, void* buffer, int32_t len)
{
    return 0;
}

// get capacity of UB/L0ABC ..
int32_t pv_get_memory_capacity(PVEnv* p_env, int32_t scope, int32_t* capacity)
{
    return 0;
}

// re-turn list of memory access in the last instruction
// the pointer is valid until next pv_step
struct PVMemAccessList* pv_get_mem_access()
{
    pv_mem_access_list.list = nullptr;
    pv_mem_access_list.len = 0;
    return &pv_mem_access_list;
}

int32_t pv_read_va_register(PVEnv* p_env, int32_t va_index, uint64_t* val0, uint64_t* val1)
{
    return 0;
}

int32_t pv_write_va_register(PVEnv* p_env, int32_t va_index, uint64_t* val0, uint64_t* val1)
{
    return 0;
}

int32_t pv_read_smask_tbl(const PVEnv* p_env, int64_t addr, void* buffer, int32_t len)
{
    return 0;
}
int32_t pv_write_smask_tbl(const PVEnv* p_env, int64_t addr, const void* buffer, int32_t len)
{
    return 0;
}