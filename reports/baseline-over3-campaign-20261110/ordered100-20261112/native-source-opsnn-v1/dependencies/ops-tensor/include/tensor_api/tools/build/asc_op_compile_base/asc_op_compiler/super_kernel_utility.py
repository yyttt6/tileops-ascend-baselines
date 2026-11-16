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
super kernel utility
"""

# import all func or types for sk, if need delete grep super kernel code in 4.1.2
import enum
from asc_op_compile_base.common.platform.platform_info import get_soc_spec
from asc_op_compile_base.common.buildcfg import get_current_build_config
from asc_op_compile_base.common.buildcfg.buildcfg_mapping import op_debug_config
from .ascendc_constants import KernelMetaType, STR_TO_KERNEL_TYPE_V220
from .ascendc_common_utility import CommonUtility, gen_func_align_attribute
from .ascendc_compile_base import search_in_line, extract_file_path
from asc_op_compile_base.common.utils.log_utils import AscendCLogLevel, CompileStage
from .super_kernel_constants import ERR_CODE


def get_op_debug_config():
    return get_current_build_config(op_debug_config)


def run_local_cmd(cmds, compile_log_path=None):
    output, ret_code = CommonUtility.run_cmd_ascendc(cmds, "compile")
    CommonUtility.dump_build_log(output.decode(), cmds, CompileStage.SPLIT_SUB_OBJS, ret_code)
    CommonUtility.dump_compile_log(cmds, CompileStage.SPLIT_SUB_OBJS, compile_log_path)


def get_wait_flag_for_chip(flag_id):
    if CommonUtility.is_c310():
        return f"wait_flag_dev(PIPE_S, {flag_id});"
    return f"wait_flag_dev({flag_id});"


def check_exist_forbidden_symbols(dst_i_file, forbidden_symbols, allow_path):
    need_check: bool = True
    block_file_path = ''
    err_str = ''
    result_symbol_list = []
    path_list = []
    line_result = []
    with open(dst_i_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("#"):
                need_check = True
                result, result_instrs = search_in_line(line, allow_path)
                if result:
                    need_check = False
                block_file_path = line
                continue
            if not need_check:
                continue
            result, result_instrs = search_in_line(line, forbidden_symbols)
            if result:
                result_symbol_list.append(result_instrs)
                path_list.append(extract_file_path(block_file_path))
                line_result.append(line)
    return result_symbol_list, path_list, line_result


def check_exist_instrinsic_when_super_kernel(dst_i_file):
    forbbiden_instrinsis = ['get_block_idx', 'get_block_num', 'get_task_ration', 'block_idx']
    allow_instrinsic_path_mark = ['bisheng_compiler', 'ccec_compiler', 'tikcpp/tikcfw', 'asc/impl', 'impl/basic_api']
    result_symbol_list, path_list, line_result =\
        check_exist_forbidden_symbols(dst_i_file, forbbiden_instrinsis, allow_instrinsic_path_mark)
    result_str = ''
    len_result_symbol_list = len(result_symbol_list)
    if len_result_symbol_list != 0:
        result_str += f'ERROR({len_result_symbol_list}): '
        for i, result in enumerate(result_symbol_list):
            if result == 'get_block_idx':
                result += ' or block_idx'
            result_str += f"instrinsic {result} is forbidden in sub operator "\
                          f"{path_list[i]}, code line is : {line_result[i]}"
            if i != len_result_symbol_list - 1:
                result_str += '\n'
        CommonUtility().ascendc_raise_python_err(ERR_CODE, result_str)
