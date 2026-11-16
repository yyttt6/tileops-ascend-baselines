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

"""Struct parser."""

import re
from typing import Tuple


STRUCT_CONTENT = re.compile(r'\{(.+)\}', re.DOTALL)

class ParseError(Exception):
    """Parse error."""


def extract_type_name(line: str) -> str:
    """Extract struct member type name."""
    line = line.split(',')[0].strip()
    line = line.split('=')[0].strip()
    parts = line.split()
    # drop the last part to get type name.
    return ' '.join(parts[:-1])


def parse_struct_by_str(struct_str: str) -> Tuple[str, ...]:
    """Parse STRUCT by string."""
    search_obj = STRUCT_CONTENT.search(struct_str)
    if not search_obj:
        raise ParseError()

    struct_content = search_obj.group(1)
    struct_mems = struct_content.split(';')
    struct_mems = [mem.strip() for mem in struct_mems if mem.strip()]
    struct_types = tuple(extract_type_name(mem) for mem in struct_mems)
    return struct_types
