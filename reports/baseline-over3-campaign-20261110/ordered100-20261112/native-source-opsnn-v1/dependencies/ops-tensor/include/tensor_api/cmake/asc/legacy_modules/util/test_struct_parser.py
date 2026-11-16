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

"""Testcases for struct_parser.py"""

import pytest

from struct_parser import ParseError, parse_struct_by_str


TAIL_STRUCT = """struct TailStruct {
    uint32_t tail[16];
};"""

TAIL_STRUCT_2 = """struct TailStruct {
    unsigned int tail[16];
};"""

TILING_STRUCT = """struct TilingStruct {
    uint32_t a, b, c, d;
    uint64_t x,y,z;
    TailStruct tail;
};"""

MY_STRUCT = """struct MyStruct{
    uint8_t x = 0;
};"""

MY_STRUCT_2 = """struct MyStruct{
    unsigned long long x = 0, y;
};"""

MY_STRUCT_EX = """struct MyStruct{
    uint8_t x = 0;"""


def test_parse_struct_by_str_01():
    result = parse_struct_by_str(TAIL_STRUCT)
    assert result == ('uint32_t',)


def test_parse_struct_by_str_02():
    result = parse_struct_by_str(TAIL_STRUCT_2)
    assert result == ('unsigned int',)


def test_parse_struct_by_str_03():
    result = parse_struct_by_str(TILING_STRUCT)
    assert result == ('uint32_t', 'uint64_t', 'TailStruct')


def test_parse_struct_by_str_04():
    result = parse_struct_by_str(MY_STRUCT)
    assert result == ('uint8_t',)


def test_parse_struct_by_str_05():
    result = parse_struct_by_str(MY_STRUCT_2)
    assert result == ('unsigned long long',)


def test_parse_struct_by_str_ex_01():
    with pytest.raises(ParseError):
        parse_struct_by_str(MY_STRUCT_EX)
