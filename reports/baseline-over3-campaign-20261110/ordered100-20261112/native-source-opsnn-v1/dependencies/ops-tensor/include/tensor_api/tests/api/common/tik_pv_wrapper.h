/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef __STUB_TIK_PV_WRAPPER_H__
#define __STUB_TIK_PV_WRAPPER_H__

#include <cstdint>
#include <memory>
#include <vector>

// #define TIK_PV_API __attribute__((__visibility__("default")))
#define TIK_PV_API

class TIK_PV_API PVEnv {
public:
    PVEnv() {};
    virtual ~PVEnv() {};
};

struct PvMemAccess {
    int32_t mem_scope = 0;
    bool is_read = true;
    uint64_t address = 0;
    uint64_t size = 0;

    PvMemAccess(int32_t _mem_scope, bool _is_read) : mem_scope(_mem_scope), is_read(_is_read) {}

    PvMemAccess() {}
};

struct PVMemAccessList {
    PvMemAccess* list = nullptr;
    int32_t len = 0;
};


extern "C" TIK_PV_API void set_config_path(const char* path);
extern "C" TIK_PV_API PVEnv* pv_create(uint32_t chip_version, const char* path, uint32_t core_id);

extern "C" TIK_PV_API void pv_destroy(PVEnv* p_env);

// for the instructions below: re-turn 0 means success, re-turn 1 means fail.
extern "C" TIK_PV_API int32_t pv_step(PVEnv* p_env, uint32_t instr);

extern "C" TIK_PV_API int32_t pv_read_gpr_register(PVEnv* p_env, uint64_t register_name, uint64_t* value);

extern "C" TIK_PV_API int32_t pv_write_gpr_register(PVEnv* p_env, uint64_t register_name, uint64_t value);
extern "C" TIK_PV_API int32_t pv_read_spr_register(PVEnv* p_env, const char* register_name, uint64_t* value);

extern "C" TIK_PV_API int32_t pv_write_spr_register(PVEnv* p_env, const char* register_name, uint64_t value);

// buffer is managed by caller
// len is length of buffer in bytes
extern "C" TIK_PV_API int32_t pv_read_memory(PVEnv* p_env, int64_t addr, int32_t scope, void* buffer, int32_t len);

extern "C" TIK_PV_API int32_t pv_write_memory(PVEnv* p_env, int64_t addr, int32_t scope, void* buffer, int32_t len);

// get capacity of UB/L0ABC ..
extern "C" TIK_PV_API int32_t pv_get_memory_capacity(PVEnv* p_env, int32_t scope, int32_t* capacity);

// re-turn list of memory access in the last instruction
// the pointer is valid until next pv_step
extern "C" TIK_PV_API struct PVMemAccessList* pv_get_mem_access();

extern "C" TIK_PV_API int32_t pv_read_va_register(PVEnv* p_env, int32_t va_index, uint64_t* val0, uint64_t* val1);
extern "C" TIK_PV_API int32_t pv_write_va_register(PVEnv* p_env, int32_t va_index, uint64_t* val0, uint64_t* val1);
extern "C" TIK_PV_API int32_t pv_read_smask_tbl(const PVEnv* p_env, int64_t addr, void* buffer, int32_t len);
extern "C" TIK_PV_API int32_t pv_write_smask_tbl(const PVEnv* p_env, int64_t addr, const void* buffer, int32_t len);


#endif