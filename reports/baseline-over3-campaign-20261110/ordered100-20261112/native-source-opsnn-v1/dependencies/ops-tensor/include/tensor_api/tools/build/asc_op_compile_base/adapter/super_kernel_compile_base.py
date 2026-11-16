#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""
super kernel compile base
"""


def gen_super_dump_code(is_mix: bool, dump_size: int, offset: int):
    source = ""
    source += "    #if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON\n"
    source += "    constexpr uint32_t ASCENDC_DUMP_SIZE = 123;\n"
    if is_mix:
        source += f"    AscendC::InitDump(true, workspace + {offset}, ASCENDC_DUMP_SIZE);\n"
    else:
        source += f"    AscendC::InitDump(false, workspace + {offset}, ASCENDC_DUMP_SIZE);\n"
    source += "    #endif\n"
    return source