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
tbe utils
"""
from . import const


from .para_check import KERNEL_NAME
from .para_check import REQUIRED_INPUT
from .para_check import OPTION_INPUT
from .para_check import DYNAMIC_INPUT
from .para_check import REQUIRED_OUTPUT
from .para_check import OPTION_OUTPUT
from .para_check import DYNAMIC_OUTPUT
from .para_check import REQUIRED_ATTR_INT
from .para_check import REQUIRED_ATTR_FLOAT
from .para_check import REQUIRED_ATTR_STR
from .para_check import REQUIRED_ATTR_BOOL
from .para_check import REQUIRED_ATTR_TYPE
from .para_check import REQUIRED_ATTR_LIST_INT
from .para_check import REQUIRED_ATTR_LIST_FLOAT
from .para_check import REQUIRED_ATTR_LIST_BOOL
from .para_check import REQUIRED_ATTR_LIST_LIST_INT
from .para_check import OPTION_ATTR_INT
from .para_check import OPTION_ATTR_FLOAT
from .para_check import OPTION_ATTR_STR
from .para_check import OPTION_ATTR_BOOL
from .para_check import OPTION_ATTR_TYPE
from .para_check import OPTION_ATTR_LIST_INT
from .para_check import OPTION_ATTR_LIST_FLOAT
from .para_check import OPTION_ATTR_LIST_BOOL
from .para_check import OPTION_ATTR_LIST_LIST_INT
from .para_check import OP_ERROR_CODE_000
from .para_check import OP_ERROR_CODE_001
from .para_check import OP_ERROR_CODE_002
from .para_check import OP_ERROR_CODE_003
from .para_check import OP_ERROR_CODE_004
from .para_check import OP_ERROR_CODE_005
from .para_check import OP_ERROR_CODE_006
from .para_check import OP_ERROR_CODE_007
from .para_check import OP_ERROR_CODE_008
from .para_check import OP_ERROR_CODE_009
from .para_check import OP_ERROR_CODE_010
from .para_check import OP_ERROR_CODE_011
from .para_check import OP_ERROR_CODE_012
from .para_check import OP_ERROR_CODE_013
from .para_check import OP_ERROR_CODE_014
from .para_check import OP_ERROR_CODE_015
from .para_check import OP_ERROR_CODE_016
from .para_check import OP_ERROR_CODE_017
from .para_check import OP_ERROR_CODE_018
from .para_check import OP_ERROR_CODE_019
from .para_check import OP_ERROR_CODE_020
from .para_check import OP_ERROR_CODE_021
from .para_check import OP_ERROR_CODE_022
from .para_check import OP_ERROR_CODE_023
from .para_check import OP_ERROR_CODE_024
from .para_check import OP_ERROR_CODE_025
from .para_check import OP_ERROR_CODE_026
from .para_check import OP_ERROR_CODE_027
from .para_check import OpParamInfoKey
from .para_check import TensorFormat
from .para_check import ALL_DTYPE_LIST
from .para_check import check_op_params
from .para_check import check_shape
