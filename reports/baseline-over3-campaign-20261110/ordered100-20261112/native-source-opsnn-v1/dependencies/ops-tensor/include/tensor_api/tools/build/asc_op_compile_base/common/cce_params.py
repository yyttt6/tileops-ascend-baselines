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
CCE configuration constants
"""
from __future__ import absolute_import as _abs

import threading


# Example: pylint: disable=invalid-name, useless-object-inheritance, too-few-public-methods
# Switch in tik/api/tik_build.py
class OUTPUT_PATH_CLASS(object):
    """
    cce output path
    """
    output_path = "kernel_meta"


# Example: pylint: disable=invalid-name, useless-object-inheritance, too-few-public-methods
class GM_NAME_MAP_CLASS(object):
    """
    get gm_name map
    """
    gm_name_map = {}


MIX_AIC_SUFFIX = "_mix_aic"
MIX_AIV_SUFFIX = "_mix_aiv"
JSON_KEY_TASK_RATION = "taskRation"
FILE_EXISTS_ERR = 17


# Example: pylint: disable=invalid-name, useless-object-inheritance, too-few-public-methods
class conv_buffer_ex(object):
    """
    conv buffer
    """
    offsetPad = None


# represent 5 soc, currently contains in tik
ASCEND_310 = "Ascend310"
ASCEND_910 = "Ascend910"
ASCEND_910H = "Ascend910B"
ASCEND_910M = "Ascend910A"
ASCEND_910P = "Ascend910ProA"
HI3796CV300ES = "Hi3796CV300ES"
HI3796CV300CS = "Hi3796CV300CS"
SD3403 = "SD3403"
ASCEND_610 = "Ascend610"
ASCEND_310P = "Ascend310P"
BS9SX1A = "BS9SX1A"
ASCEND_610B = "Ascend610B"
ASCEND_910B = "Ascend910B"
ASCEND_910C = "Ascend910C"
ASCEND_310B = "Ascend310B"
ASCEND_SD = "SD3403"
_AIC_ENGINE = "AiCore"
_VEC_ENGINE = "VectorCore"

AIC_310P = ASCEND_310P + _AIC_ENGINE
VEC_310P = ASCEND_310P + _VEC_ENGINE
AIC_610 = ASCEND_610 + _AIC_ENGINE
VEC_610 = ASCEND_610 + _VEC_ENGINE
AIC_BS9SX1A = BS9SX1A + _AIC_ENGINE
VEC_BS9SX1A = BS9SX1A + _VEC_ENGINE
AIC_610B = ASCEND_610B + _AIC_ENGINE
VEC_610B = ASCEND_610B + _VEC_ENGINE
AIC_310B = ASCEND_310B + _AIC_ENGINE
VEC_310B = ASCEND_310B + _VEC_ENGINE
HI3796CV300ESAIC = HI3796CV300ES + _AIC_ENGINE
HI3796CV300CSAIC = HI3796CV300CS + _AIC_ENGINE
SD3403AIC = SD3403 + _AIC_ENGINE
ASCEND_310AIC = ASCEND_310 + _AIC_ENGINE
ASCEND_910AIC = ASCEND_910 + _AIC_ENGINE
ASCEND_910BAIC = ASCEND_910B + _AIC_ENGINE
ASCEND_910BVEC = ASCEND_910B + _VEC_ENGINE
ASCEND_910CAIC = ASCEND_910C + _AIC_ENGINE
ASCEND_910CVEC = ASCEND_910C + _VEC_ENGINE
ASCEND_SD_AIC = "SD3403" + _AIC_ENGINE

