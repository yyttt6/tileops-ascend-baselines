/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include "mmpa/mmpa_api.h"
#include "mmpa_stub.h"

std::shared_ptr<Adx::MmpaStub> Adx::MmpaStub::instance_;
Adx::MmpaStub* Adx::MmpaStub::fake_instance_;

#ifdef __cplusplus
extern "C" {
#endif

INT32 mmGetEnv(const CHAR* name, CHAR* value, UINT32 len)
{
    printf("Get stub mmpa.\n");
    const char* env = getenv(name);
    if (env == nullptr) {
        return EN_ERROR;
    }

    strncpy(value, env, len);
    return EN_OK;
}

INT32 mmRealPath(const CHAR* path, CHAR* realPath, INT32 realPathLen)
{
    if (path == nullptr || realPath == nullptr || realPathLen < MMPA_MAX_PATH) {
        return EN_INVALID_PARAM;
    }

    std::string str_path = path;
    if (str_path.find("binary_info_config.json") != std::string::npos ||
        str_path.find("kernel_bninference_d") != std::string::npos) {
        strncpy(realPath, path, realPathLen);
        return EN_OK;
    }

    if (realpath(path, realPath) == nullptr) {
        return EN_ERROR;
    }
    return EN_OK;
}

VOID* mmDlopen(const CHAR* fileName, INT32 mode) { return Adx::MmpaStub::GetInstance()->mmDlopen(fileName, mode); }

INT32 mmDlclose(VOID* handle) { return EN_OK; }

VOID* mmDlsym(VOID* handle, const CHAR* funcName) { return Adx::MmpaStub::GetInstance()->mmDlsym(handle, funcName); }

CHAR* mmDlerror() { return ""; }

INT32 mmGetErrorCode() { return 0; }

CHAR* mmGetErrorFormatMessage(int errnum, CHAR* buf, size_t size)
{
    if (buf == NULL || size <= 0) {
        return NULL;
    }
    return strerror_r(errnum, buf, size);
}
#ifdef __cplusplus
}
#endif
