/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file ascendc_pack_kernel.h
 * \brief
 */
#ifndef __ASCENDC_PACK_KERNEL_H__
#define __ASCENDC_PACK_KERNEL_H__

#include "ascendc_elf_tool.h"

enum {
    ELF_TYPE_ELF = 0,
    ELF_TYPE_AIVEC,
    ELF_TYPE_AICUBE,
    ELF_TYPE_MAX
} elf_type_t;

#ifdef __cplusplus
extern "C"
{
#endif
size_t GetFileSize(const char* filePath);
size_t ReadFile(const char *file, void *buf, size_t len);
size_t WriteFile(const char *file, void *buf, size_t len);
#ifdef __cplusplus
}
#endif

#if !(defined(UT_TEST) || defined(ST_TEST))
int main(int argc, char *argv[]);
#else
#ifdef __cplusplus
extern "C"
{
#endif
int AscendcPackKernelMain(int argc, char *argv[]);
#ifdef __cplusplus
}
#endif
#endif

#endif