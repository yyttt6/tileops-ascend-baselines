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
error manager util
"""
import json
import os


def get_error_message(args):
    """
    :param args: dict
    keys in dict must be in accordance with xlsx
    :return: string
    formatted message
    """
    error_code = args.get("errCode")
    file_name = os.path.dirname(os.path.abspath(__file__))
    with open("{}/errormanager.json".format(file_name)) as file_content:
        data = json.load(file_content)
        error_dict = {}
        for error_message in data:
            error_dict[error_message['errCode']] = error_message
        error_json = error_dict
    error_stmt = error_json.get(error_code)
    if error_stmt is None:
        return "errCode = {} has not been defined".format(error_code)
    arg_list = error_stmt.get("argList").split(",")
    arg_value = []
    for arg_name in arg_list:
        if arg_name.strip() not in args:
            arg_value.append("")
        else:
            arg_value.append(args.get(arg_name.strip()))
    msg = error_json.get(error_code).get("errMessage") % tuple(arg_value)
    msg = msg.replace("[]", "")
    return msg


def raise_runtime_error(dict_args):
    """
    raise runtime error
    :param dict_args: error message dict
    """
    raise RuntimeError(dict_args,
                       get_error_message(dict_args))

