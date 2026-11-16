/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INDV_LINUX_H
#define INDV_LINUX_H

#include <dirent.h>
#include <securec.h>
#include "mmpa/mmpa_api.h"

#ifdef __cplusplus
extern "C" {
#endif
#define INDV_EN_OK 0
#define INDV_EN_ERROR -1
#define INDV_EN_INVALID_PARAM -2

typedef struct dirent IndvDirent;
typedef int32_t (*IndvFilter)(const IndvDirent* entry);
typedef int32_t (*IndvSort)(const IndvDirent** a, const IndvDirent** b);

int32_t IndvScandir(const char* path, IndvDirent*** entryList, IndvFilter filter, IndvSort sort);
void IndvScandirFree(IndvDirent** entryList, int32_t fileCount);
#ifdef __cplusplus
}
#endif
#endif