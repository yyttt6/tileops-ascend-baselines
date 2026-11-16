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
fusion build config manager
"""

# fusion build config dict
_fusion_build_configs = {}

def set_fusion_buildcfg(op_type, build_config):
    """
    set fusion build config

    Parameters
    ----------
    op_type : string
        op type
    config: dict
        build configs

    Returns
    -------
    decorator : decorator
        decorator to set fusion build config
    """
    global _fusion_build_configs
    if op_type is None:
        raise RuntimeError("set fusion buildcfg failed, op_type is none")
    if op_type in _fusion_build_configs.keys():
        _fusion_build_configs[op_type].update(build_config)
    else:
        _fusion_build_configs[op_type] = build_config


def get_fusion_buildcfg(op_type=None):
    """
    :return:
    """
    if op_type is None:
        return _fusion_build_configs
    return _fusion_build_configs.get(op_type)
