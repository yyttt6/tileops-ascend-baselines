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
identify meta section info
"""

import subprocess
import re
import os
import sys
import struct
from dataclasses import dataclass
from typing import Optional
from asc_op_compile_base.common.utils.log_utils import LogUtil, AscendCLogLevel


F_TYPE_AIV_TYPE_FLAG = 12
AIV_TYPE_SIMT_VF_ONLY = 3
AIV_TYPE_SIMD_SIMT_MIX_VF = 4


@dataclass
class BaseTlv:
    type: int
    length: int

    @classmethod
    def from_bytes(cls, data: bytes) -> 'BaseTlv':
        type_val, length_val = struct.unpack('<HH', data[:4])
        return cls(type_val, length_val)


def extract_hex_data(hex_dump: str) -> str:
    hex_parts = []
    if hex_dump is None:
        return " "
    for line in hex_dump.splitlines():
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) < 2:
            continue

        for i in range(1, len(parts)):
            part = parts[i]
            if '.' in part or len(part) > 8:
                break
            hex_parts.append(part)

    return ''.join(hex_parts)


def parse_hexdump_to_tlv(hexdump: str) -> str:
    tlv_lines = []
    for line in hexdump.splitlines():
        if not line.strip():
            continue

        match = re.match(r'^(0x)?[0-9a-f]+\s+([0-9a-f\s]+)(?:\s+(.+))?$', line.lstrip())
        if not match:
            continue
        address = match.group(1)
        hex_data = match.group(2).strip()
        ascii_part = match.group(3) if match.group(3) else ''

        hex_chars = hex_data.replace(' ', '')
        hex_parts = []
        for i in range(0, len(hex_chars), 8):
            if i + 8 <= len(hex_chars):
                hex_parts.append(hex_chars[i: i + 8])
        
        formatted_hex = ' '.join(hex_parts)
        tlv_lines.append(f"{address} {formatted_hex.ljust(39)} {ascii_part}")

    return '\n'.join(tlv_lines)


# .o生成tlv
def generate_tlv_from_object_file(object_file_path: str, kernel_name: str) -> Optional[str]:
    if not os.path.exists(object_file_path):
        raise FileNotFoundError(f"object file {object_file_path} not found, please check it.")
    section_name = ".ascend.meta." + kernel_name
    generate_cmd = ['llvm-objdump', '-s', '-j', '{}'.format(section_name), '{}'.format(object_file_path)]
    LogUtil.print_compile_log(kernel_name, f"start generate tlv info ...", AscendCLogLevel.LOG_INFO)
    try:
        proc = subprocess.Popen(generate_cmd, stdout=subprocess.PIPE, stderr=None)
        (tlv_info, _) = proc.communicate()
    except Exception as e:
        LogUtil.print_compile_log(kernel_name, f"generate tlv info failed: {e}", AscendCLogLevel.LOG_WARNING)
        return None

    tlv_info = tlv_info.decode('utf-8')
    
    tlv_string = parse_hexdump_to_tlv(tlv_info)
    
    return tlv_string


def parse_tlv_get_simt_value(tlv_string: str, kernel_name: str):
    try:        
        LogUtil.print_compile_log(kernel_name, f"start parse tlv info ...", AscendCLogLevel.LOG_INFO)
        hex_data = extract_hex_data(tlv_string)
        if not hex_data:
            return None
        data = bytes.fromhex(hex_data)
        return _parse_simt_value_from_bytes(data)
    except Exception as e:
        LogUtil.print_compile_log(kernel_name, f"parse TLV data failed", \
                AscendCLogLevel.LOG_WARNING)
        return None


def _parse_simt_value_from_bytes(data: bytes) -> str:
    offset = 0
    while offset + 4 <= len(data):
        tlv_type, length = _parse_tlv_header_and_check(data, offset)
        
        if tlv_type is None:  
            break
        
        if tlv_type == F_TYPE_AIV_TYPE_FLAG:
            return _extract_tlv_value(data, offset, length)
        
        offset += 4 + length
    return None


def _parse_tlv_header_and_check(data: bytes, offset: int):

    tlv_type = data[offset] | (data[offset + 1] << 8)
    length = data[offset + 2] | (data[offset + 3] << 8)
    
    if offset + 4 + length > len(data):
        return None, None
        
    return tlv_type, length


def _extract_tlv_value(data: bytes, offset: int, length: int) -> str:
    value_start = offset + 4
    if length >= 4:
        value = data[value_start] | (data[value_start + 1] << 8) | \
                (data[value_start + 2] << 16) | (data[value_start + 3] << 24)
    else:
        value = 0
        for i in range(min(length, 4)):
            value |= data[value_start + i] << (i * 8)
    return f"0x{value:08x}"


def check_op_type_is_simt(object_file_path: str, kernel_name: str):
    tlv_string = generate_tlv_from_object_file(object_file_path, kernel_name)
    simt_value = parse_tlv_get_simt_value(tlv_string, kernel_name)
    if not simt_value:
        LogUtil.print_compile_log(kernel_name, f"simt type in section is None", \
                AscendCLogLevel.LOG_WARNING)
        return False
    else:
        decimal_val = int(simt_value.lstrip('0x'), 16)
        target_values = {AIV_TYPE_SIMD_SIMT_MIX_VF, AIV_TYPE_SIMT_VF_ONLY}
        if decimal_val in target_values:
            LogUtil.print_compile_log(kernel_name, f"{kernel_name} is simt op", AscendCLogLevel.LOG_INFO)
            return True
        LogUtil.print_compile_log(kernel_name, f"{kernel_name} is not simt op", AscendCLogLevel.LOG_INFO)
        return False