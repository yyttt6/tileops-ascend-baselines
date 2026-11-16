#!/usr/bin/env python
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


import json
from asc_op_compile_base.common.utils import log as logger
from op_manager import get_inout_info_from_opstore
from constant import (CompileParam, OpcOptions, OptionalInOutMode, OpParamType)


def check_and_feed_optional_param(inout_infos, inouts):
    if len(inouts) != len(inout_infos):
        return False
    if len(inouts) == 0 or hasattr(inouts[0], CompileParam.PARAM_TYPE):
        return True
    for i, inout_info in enumerate(inout_infos):
        if inout_info.param_type == OpParamType.OPT and inouts[i] is not None:
            inouts[i][CompileParam.PARAM_TYPE] = "optional"
    return True


def check_op_optional_paramtype(json_dict, opc_compile_args):
    opt_input_mode = opc_compile_args.get(OpcOptions.OPTIONAL_INPUT_MODE)
    if opt_input_mode != OptionalInOutMode.GEN_PLACEHOLDER:
        logger.debug("No need to check paraType, cur mode is no_placeholder.")
        return True
    inputs_infos, output_infos = get_inout_info_from_opstore(json_dict.get("op_type"))
    if inputs_infos is None or output_infos is None:
        logger.warn("get inputs_infos or output_infos from op_store is None, will not check by ops info store.")
        return True
    op_list = json_dict.get("op_list")
    inputs = op_list[0].get("inputs")
    outputs = op_list[0].get("outputs")
    if not check_and_feed_optional_param(inputs_infos, inputs):
        logger.error("Op[type=%s] invalid input nums[%d], which should be equal to input nums[%d] in op store.",
                        str(json_dict.get("op_type")), len(inputs), len(inputs_infos))
        return False
    if not check_and_feed_optional_param(output_infos, outputs):
        logger.error("Op[type=%s] invalid output nums[%d], which should be equal to output nums[%d] in op store.",
                        str(json_dict.get("op_type")), len(outputs), len(output_infos))
        return False
    op_list[0]["inputs"] = inputs
    op_list[0]["outputs"] = outputs
    json_dict["op_list"] = op_list
    return True


def check_op_compilation_json(option, opc_compile_args):
    """
    check op compilation json
    """
    json_path = opc_compile_args.get(option)
    try:
        with open(json_path, "r") as file_handle:
            json_dict = json.load(file_handle)
    except Exception as e:
        logger.error("load file[%s] failed, reason: %s.", json_path, str(e))
        return False, None
    finally:
        pass

    if not isinstance(json_dict, dict):
        logger.error("File content of [%s] is not dict.", json_path)
        return False, None

    if "op_list" not in json_dict:
        logger.error("There is no op_list in json file[%s].", json_path)
        return False, None

    if not check_op_optional_paramtype(json_dict, opc_compile_args):
        logger.error("Invalid json file content[%s].", json_path)
        return False, None

    return True, json_dict


def check_op_compilation_dict(op_params, opc_compile_args):
    """
    check op compilation op_params
    """
    if not isinstance(op_params, dict):
        logger.error("File content of [%s] is not str.", op_params)
        return False, None

    if "op_list" not in op_params:
        logger.error("There is no op_list in json dict[%s].", op_params)
        return False, None

    if not check_op_optional_paramtype(op_params, opc_compile_args):
        logger.error("Invalid json dict content[%s].", op_params)
        return False, None

    return True, op_params