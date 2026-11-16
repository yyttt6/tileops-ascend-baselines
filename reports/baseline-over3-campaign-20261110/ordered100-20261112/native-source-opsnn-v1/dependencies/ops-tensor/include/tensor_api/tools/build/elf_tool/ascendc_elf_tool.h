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
 * \file ascendc_elf_tool.h
 * \brief
 */
#ifndef __TOOLS_ELF_TOOL_H__
#define __TOOLS_ELF_TOOL_H__
#include <stdlib.h>
#include <sys/types.h>
#include <stdint.h>
#include <string.h>
#include "securec.h"
#include <stdbool.h>

/*
 * ***************************
 * File Header        *
 * ***************************
 * Prog Header(optional)  *
 * ***************************
 * Section 1          *
 * ***************************
 * Section 2          *
 * ***************************
 * Section ...        *
 * ***************************
 * Section Header Table   *
 * ***************************
 */

typedef unsigned long Elf_Addr;
typedef unsigned short Elf_Half;
typedef unsigned long Elf_Off;
typedef uint32_t Elf_Word;
typedef unsigned long Elf_Xword;

#define EI_NIDENT 16
#define ELFCLASSNONE 0 /* Invalid class */
#define ELFCLASS32 1   /* 32-bit objects */
#define ELFCLASS64 2   /* 64-bit objects */
#define ELFCLASSNUM 3

#define EI_MAG0 0 /* e_ident[] indexes */
#define EI_MAG1 1
#define EI_MAG2 2
#define EI_MAG3 3
#define EI_CLASS 4

/* elf endian in elf header e_ident[] EI_DATA code */
#define EI_DATA 5
#define ELFDATA2LSB 1
#define ELFDATA2MSB 2

#define EI_VERSION 6
#define EI_OSABI 7
#define EI_PAD 8

#define EM_M32 1
#define EM_386 3
#define EM_X86_64 62 /* AMD x86-64 */
#define EM_ARM 40
#define EM_MIPS 8
#define EM_PPC 20
#define EM_PPC64 21

#define STN_UNDEF 0

/* These constants define the different elf file types */
#define ET_NONE 0
#define ET_REL 1
#define ET_EXEC 2
#define ET_DYN 3
#define ET_CORE 4
#define ET_LOPROC 0xff00
#define ET_HIPROC 0xffff

#define SHT_NOBITS 8

#define MAX_SYMNAME_LEN 512

#define ELF_ERR_NULL_POINTER 1
#define ELF_ERR_BUFFER_TOO_SMALL 2
#define ELF_ERR_NOT_64BIT  3
#define ELF_ERR_UNEXPECTED_PROG_HEADER 4

#define ELF_SUCCESS 0
#define ELF_NO_TABLE 1
#define ELF_NO_SYMBOL 2

/*
 * Elf header
 */
typedef struct {
    unsigned char e_ident[EI_NIDENT];
    Elf_Half e_type;
    Elf_Half e_machine;
    Elf_Word e_version;
    Elf_Addr e_entry;
    Elf_Off e_phoff;
    Elf_Off e_shoff;
    Elf_Word e_flags;
    Elf_Half e_ehsize;
    Elf_Half e_phentsize;
    Elf_Half e_phnum;
    Elf_Half e_shentsize;
    Elf_Half e_shnum;
    Elf_Half e_shstrndx;
} Elf_Ehdr;
/*
 * Section header
 */
typedef struct {
    Elf_Word sh_name;
    Elf_Word sh_type;   /* SHT_... */
    Elf_Xword sh_flags; /* SHF_... */
    Elf_Addr sh_addr;
    Elf_Off sh_offset;
    Elf_Xword sh_size;
    Elf_Word sh_link;
    Elf_Word sh_info;
    Elf_Xword sh_addralign;
    Elf_Xword sh_entsize;
} Elf_Shdr;

/*
 * Program header
 */
typedef struct {
    Elf_Word p_type; /* Type of segment */
    Elf_Word p_flags; /* Segment attributes */
    Elf_Off p_offset; /* Offset in file */
    Elf_Addr p_vaddr; /* Virtual address in memory */
    Elf_Addr p_paddr; /* Reserved */
    Elf_Xword p_filesz; /* Size of segment in file */
    Elf_Xword p_memsz; /* Size of segment in memory */
    Elf_Xword p_align; /* Alignment of segment */
} Elf_Phdr;

/*
 * Symbol table
 */
typedef struct {
    Elf_Word st_name; /* Symbol name */
    unsigned char st_info; /* Type and Binding attributes */
    unsigned char st_other; /* Reserved */
    Elf_Half st_shndx; /* Section table index */
    Elf_Addr st_value; /* Symbol value */
    Elf_Xword st_size; /* Size of object (e.g., common) */
} Elf64_Sym;

#define ASCEND_KERNEL_SECTION_LEN_POS 1
#define ASCEND_KERNEL_FILE_LEN_POS 2
#define ASCEND_KERNEL_HEADER_CNT 3

#define CHECK_COND_AND_DO(condition, dosomething) \
    if ((condition)) {                          \
        dosomething;                             \
    }

struct AscendKernelHeader {
    uint32_t version;
    uint32_t typeCnt;
};

#ifdef __cplusplus
extern "C" {
#endif
size_t ElfAddSection(uint8_t* elf, size_t elfSize, uint8_t* jit, size_t jitSize, uint8_t* sec, size_t secSize,
    uint32_t type);

int32_t ElfGetSymbolOffset(uint8_t* elf, size_t elfSize, const char* symbolName, size_t* offset, size_t* size);
#if defined(UT_TEST) || defined(ST_TEST)
int32_t ElfHeaderCheck(uint8_t* elf, size_t elfSize, bool checkProgHeader);
#endif
#ifdef __cplusplus
}
#endif

#endif // __TOOLS_ELF_TOOL_H__