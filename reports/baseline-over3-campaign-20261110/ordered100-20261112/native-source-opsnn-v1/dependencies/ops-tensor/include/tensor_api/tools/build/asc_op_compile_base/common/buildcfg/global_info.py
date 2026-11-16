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


# 'pylint: disable=too-few-public-methods
class GlobalInfoContainer():
    """Manage L1 buffer
    Save and call L1 buffer function
    """
    global_info = {
        "op_L1_space": -1,
        "op_L1_fusion_type": -1,
        "L2_mode": 0,
        "L1_fusion_enabled": False,
        "L2_fusion_enabled": False,
        "kernel_meta_parent_dir": ".",
        "enable_op_prebuild": False,
        "is_supernetwork_header": False,
    }


NameDict = {"op_L1_fusion_type": "tir.op_L1_fusion_type",
            "L1_fusion_enabled": "tir.enable_L1_fusion",
            "L2_fusion_enabled": "tir.enable_L2_fusion",
            "L2_mode": "tir.l2_mode"}
