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
ascendc_parse_dumpinfo.py
"""
import argparse
import os
import struct
from enum import Enum

import numpy as np


class DataType(Enum):
    DT_FLOAT = 0
    DT_FLOAT16 = 1
    DT_INT8 = 2
    DT_INT32 = 3
    DT_UINT8 = 4
    DT_INT16 = 6
    DT_UINT16 = 7
    DT_UINT32 = 8
    DT_INT64 = 9
    DT_UINT64 = 10


class Position(Enum):
    DEFAULT = 0
    UB = 1
    L1 = 2


class CorePosition(Enum):
    MIX = 0
    AIC = 1
    AIV = 2


class DumpType(Enum):
    DEFAULT_TYPE = 0
    SCALAR_TYPE = 1
    TENSOR_TYPE = 2


def str2bool(v):
    if v.lower() in ('True', 'yes', 'true', 't', 'T', 'y', '1'):
        return True
    elif v.lower() in ('False', 'no', 'false', 'f', 'F', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Invalid boolean value: %s' % v)


def parse_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--log-path', dest='log_path', required=True,
                        help="log path where store dump tensor log information, e.g. xx/out.bin")
    parser.add_argument('-s', '--save-path', dest='save_path', required=False, default="all",
                        help="information path after parsing, e.g. xx/parsing_info.txt")
    parser.add_argument('-i', '--is-show', dest='is_show', required=False, const=False, type=str2bool, nargs='?',
                        help="whether show on the screen: true:show, false:unshow")
    parser.add_argument('-r', '--write-mode', dest='write_mode', required=False, default="w", choices=['a', 'w'],
                        help="mode for parsed file write, default option is overwrite: w:overwrite, a:append")
    dump_parse_args = parser.parse_args()
    return dump_parse_args


def get_dump_type_str(dump_type_input):
    if dump_type_input == DumpType.SCALAR_TYPE.value:
        return "DumpScalar: "
    elif dump_type_input == DumpType.TENSOR_TYPE.value:
        return "DumpTensor: "
    else:
        raise Exception('[ERROR]:Invalid dump Type')


def get_data_type_size(data_type):
    if data_type == DataType.DT_INT64.value or data_type == DataType.DT_UINT64.value:
        return 8
    elif data_type == DataType.DT_FLOAT.value or data_type == DataType.DT_INT32.value or \
            data_type == DataType.DT_UINT32.value:
        return 4
    elif data_type == DataType.DT_FLOAT16.value or data_type == DataType.DT_INT16.value or \
            data_type == DataType.DT_UINT16.value:
        return 2
    elif data_type == DataType.DT_INT8.value or data_type == DataType.DT_UINT8.value:
        return 1
    else:
        raise Exception('[ERROR]:Invalid Data Type')


def get_unpack_type(data_type):
    if data_type == DataType.DT_FLOAT.value:
        return ['f', 'float32']
    elif data_type == DataType.DT_INT32.value:
        return ['i', 'int32_t']
    elif data_type == DataType.DT_UINT32.value:
        return ['I', 'uin32_t']
    elif data_type == DataType.DT_FLOAT16.value:
        return ['e', 'half']
    elif data_type == DataType.DT_INT16.value:
        return ['h', 'int16_t']
    elif data_type == DataType.DT_UINT16.value:
        return ['H', 'uint16_t']
    elif data_type == DataType.DT_INT8.value:
        return ['b', 'int8_t']
    elif data_type == DataType.DT_UINT8.value:
        return ['B', 'uint8_t']
    elif data_type == DataType.DT_UINT64.value:
        return ['Q', 'uint64_t']
    elif data_type == DataType.DT_INT64.value:
        return ['q', 'int64_t']
    else:
        raise Exception('[ERROR]:Invalid Data Type')


def get_positon(position, dump_type):
    if dump_type == DumpType.SCALAR_TYPE.value:
        if position == CorePosition.MIX.value:
            return 'MIX'
        elif position == CorePosition.AIC.value:
            return 'AIC'
        elif position == CorePosition.AIV.value:
            return 'AIV'
        else:
            raise Exception('[ERROR]:Invalid Position')

    elif dump_type == DumpType.TENSOR_TYPE.value:
        if position == Position.UB.value:
            return 'UB'
        elif position == Position.L1.value:
            return 'L1'
        else:
            raise Exception('[ERROR]:Invalid Position')

    else:
        raise Exception('[ERROR]:Invalid Position')


def process_log(dump_parse_args):
    bin_file_path = dump_parse_args.log_path
    if dump_parse_args.save_path != "all":
        save_path = dump_parse_args.save_path
    else:
        save_path = os.path.dirname(bin_file_path) + '/' + 'dumptensor_log.txt'
    binfile = open(bin_file_path, 'rb')
    index = 0
    head_msg_len = 32
    check_msg_len = 32
    size_of_u32 = 4
    size_of_int32 = 4
    size_of_int64 = 8
    magic_golden = 1520811213 # 0x5aa5bccd
    binfile_size = os.path.getsize(bin_file_path)
    write_method = dump_parse_args.write_mode
    fp = open(save_path, write_method, encoding='utf-8')

    while index < binfile_size:
        struct_info_len = struct.unpack('I', binfile.read(size_of_int32))[0]
        struct_info_blockid = struct.unpack('i', binfile.read(size_of_int32))[0]
        struct_info_block_dim = struct.unpack('I', binfile.read(size_of_int32))[0]
        struct_info_dump_offset = struct.unpack('I', binfile.read(size_of_int32))[0]
        struct_info_magic = struct.unpack('I', binfile.read(size_of_int32))[0]
        struct_info_rsv = struct.unpack('I', binfile.read(size_of_int32))[0]
        struct_info_dump_addr = struct.unpack('Q', binfile.read(size_of_int64))[0]

        if struct_info_magic != magic_golden:
            raise Exception('[ERROR]:Check Dump Message head failed!')

        checkhead = {'block_id': struct_info_blockid, 'total_block_num': struct_info_block_dim,
                    'block_remain_len': struct_info_dump_offset, 'block_initial_space': struct_info_len,
                    'magic': hex(struct_info_magic)}
        fp.write('DumpHead: ')
        for k, v in checkhead.items():
            if k == 'block_id':
                fp.write(k+'='+str(v))
            else:
                fp.write(', ' + k+'='+str(v))
        index += check_msg_len
        fp.write('\n')
        block_dump_index = check_msg_len

        while block_dump_index < struct_info_len:
            dump_type = struct.unpack('I', binfile.read(size_of_u32))[0]
            if dump_type != DumpType.SCALAR_TYPE.value and dump_type != DumpType.TENSOR_TYPE.value:
                empty_offset = struct_info_len - block_dump_index - size_of_u32
                index = index + empty_offset + size_of_u32
                binfile.read(empty_offset)
                break
            dump_type_str = get_dump_type_str(dump_type)
            addr = struct.unpack('I', binfile.read(size_of_u32))[0]
            data_type = struct.unpack('I', binfile.read(size_of_u32))[0]
            desc = struct.unpack('I', binfile.read(size_of_u32))[0]
            buffer_id = struct.unpack('I', binfile.read(size_of_u32))[0]
            message_body_len = struct.unpack('I', binfile.read(size_of_u32))[0]
            position = struct.unpack('I', binfile.read(size_of_u32))[0]
            rev = struct.unpack('I', binfile.read(size_of_u32))[0]
            phy_position = get_positon(position, dump_type)
            index = index + head_msg_len
            block_dump_index = block_dump_index + head_msg_len
            data_type_info = get_unpack_type(data_type)
            datatype_size = get_data_type_size(data_type)
            messagehead = {'desc': desc, 'addr': hex(addr), 'message_body_len': message_body_len,
                            'data_type': data_type_info[1], 'position': phy_position}
            fp.write(dump_type_str)
            for k, v in messagehead.items():
                if k == 'desc':
                    fp.write(k+'='+str(v))
                else:
                    fp.write(', ' + k+'='+str(v))
            fp.write('\n')
            fp.write('[')
            for i in range(0, message_body_len):
                if (index + i * datatype_size) >= binfile_size:
                    err_index = index + i * datatype_size
                    raise Exception('idx is out of range, idx is {} binfile_size is {}'.format(err_index, binfile_size))
                num = struct.unpack(data_type_info[0], binfile.read(datatype_size))[0]
                if i != message_body_len - 1:
                    fp.write(str(num) + ', ')
                else:
                    fp.write(str(num))

                if i != 0 and i % 32 == 0:
                    fp.write('\n')
            fp.write(']')
            fp.write('\n')
            index = index + message_body_len * datatype_size
            block_dump_index = block_dump_index + message_body_len * datatype_size
    binfile.close()
    fp.close()
    return save_path


def print_log(saving_path_input: str):
    saving_txt_path = saving_path_input
    with open(saving_txt_path, 'rb') as f:
        for line in f:
            print(line.decode('utf-8'))


if __name__ == "__main__":
    args = parse_input_args()
    saving_path = process_log(args)
    if args.is_show:
        print_log(saving_path)