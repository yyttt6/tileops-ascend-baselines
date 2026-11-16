#!/usr/bin/env python3
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
split_opc_json_with_op_list.py
"""
import sys
import os
import json
import stat


def wr_json(json_obj, json_file):
    flags = os.O_WRONLY | os.O_CREAT
    modes = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(json_file, flags, modes), "w") as f:
        json.dump(json_obj, f, indent=2)


def split_json_files(ori_json, output_dir):
    """
    gen output json by binary_file and opc json file
    """
    if not os.path.exists(ori_json):
        print("[ERROR]the ori_json doesn't exist")
        return []
    if not os.path.exists(output_dir):
        print("[ERROR]the out_dir of split_json doesn't exist")
        return []
    with open(ori_json, "r") as file_wr:
        binary_json = json.load(file_wr)

    op_type = binary_json.get("op_type")
    op_list = binary_json.get("op_list", list())
    op_list_num = len(op_list)

    generated_files = []

    for idx, op in enumerate(op_list):
        new_binary_json = {"op_type": op_type}
        new_binary_json["op_list"] = [op]

        bin_filename = op.get("bin_filename")
        if not bin_filename:
            print(f"[ERROR]bin_filename field not found in op {idx}")
            return []
        base_name = os.path.splitext(bin_filename)[0]
        new_binary_file = os.path.join(output_dir, f"{base_name}.json")

        generated_files.append(new_binary_file)

        wr_json(new_binary_json, new_binary_file)

    return generated_files

if __name__ == '__main__':
    generated_files = split_json_files(sys.argv[1], sys.argv[2])
