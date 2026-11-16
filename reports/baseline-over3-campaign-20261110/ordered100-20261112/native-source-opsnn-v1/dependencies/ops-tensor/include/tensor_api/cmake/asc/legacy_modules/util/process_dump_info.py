#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
import re


def get_dump_info_by_source(data: str):
    dump_info = {"dump_type" : "", "dump_size" : 1048576}

    match_printf = re.search(r"__enable_feature_for_compile_printf = 1", data)
    match_assert = re.search(r"__enable_feature_for_compile_assert = 1;", data)
    if match_printf and match_assert:
        dump_info["dump_type"] = "printf,assert"
    elif match_printf:
        dump_info["dump_type"] = "printf"
    elif match_assert:
        dump_info["dump_type"] = "assert"

    match = re.search(r"__enable_feature_for_compile_printfBufSize = \s*([0-9]{1,})", data)
    if match:
        dump_info["dump_size"] = int(match.group(1))
    else:
        match = re.search(r"__enable_feature_for_compile_assertBufSize = \s*([0-9]{1,})", data)
        if match:
            dump_info["dump_size"] = int(match.group(1))

    return dump_info