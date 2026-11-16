#!/usr/bin/env python3
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
from enum import Enum


class CodeMode(Enum):
    """Code mode."""
    MIX = 0
    AIV = 1
    AIC = 2
    NORMAL = 3
    MIX_VECTOR_CORE = 4
    KERNEL_TYPE_AIV_ONLY = 5
    KERNEL_TYPE_AIC_ONLY = 6
    KERNEL_TYPE_MIX_AIV_1_0 = 7
    KERNEL_TYPE_MIX_AIC_1_0 = 8
    KERNEL_TYPE_MIX_AIC_1_1 = 9
    KERNEL_TYPE_MIX_AIC_1_2 = 10


class FuncMetaType(Enum):
    """function meta type."""
    F_TYPE_KTYPE = 1
    F_TYPE_CROSS_CORE_SYNC = 2
    F_TYPE_MAX = 3


class ExtractError(Exception):
    """Extract host stub base exception."""


class ParseFuncSignatureError(ExtractError):
    """Parse function signature error."""


class FuncNameNotFound(ParseFuncSignatureError):
    """Function name not found."""


class MultiFuncNameFound(ParseFuncSignatureError):
    """Multiple function name not found."""


class FuncParamsNotFound(ParseFuncSignatureError):
    """Function parameters in function signature not found."""


class TooFewFuncParamParts(ParseFuncSignatureError):
    """Function parameter parts is too few."""


class TooMuchFuncParamParts(ParseFuncSignatureError):
    """Function parameter parts is too much."""


class GetFuncParamPartError(ExtractError):
    """Get function parameter part error."""
    def __init__(self, func_param: 'FuncParam'):
        super().__init__(func_param)
        self.func_param = func_param


class GetFuncParamTypeError(GetFuncParamPartError):
    """Get function parameter type error."""


class GetFuncParamNameError(GetFuncParamPartError):
    """Get function parameter type error."""


class ArgumentError(ExtractError):
    """Argument error."""


class GetOfileModeError(ExtractError):
    """Get ofile mode error."""


class CxxfiltCMDError(ExtractError):
    """C++filt command error."""


class SetKernelTypeError(ExtractError):
    """Set kernel type error."""