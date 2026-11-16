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
operation function manager
"""
import functools

from .class_manager import Operator, OpCompute

# op compute func dict
_op_computes = {}
# op realization func dict
_operators = {}
# op param generalization func dict
# 'pylint: disable=C0103
_generalization = {}
_op_no_trans_bool_to_s8 = {}


def register_operator(op_type, pattern=None, trans_bool_to_s8=True):
    """
    register op realization func

    Parameters
    ----------
    op_type : string
        op type
    pattern: string
        op fusion pattern
    trans_bool_to_s8: bool
        if need trans bool to int8
    Returns
    -------
    decorator : decorator
        decorator to register realization func
    """
    if op_type is None:
        raise RuntimeError("register operator failed, op_type is none")
    global _op_no_trans_bool_to_s8
    if not trans_bool_to_s8:
        _op_no_trans_bool_to_s8[op_type] = "True"
    global _operators
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        _operators[op_type] = Operator(pattern, wrapper)
        return wrapper
    return decorator


def register_op_compute(op_type, op_mode="dynamic", support_fusion=True):
    """
    register op compute func

    Parameters
    ----------
    op_type: string
        op_func_name(old process) or op type(new process)
    op_mode: string
        dynamic or static shape
    support_fusion: bool
        support dynamic shape UB fusion
    fusion_pattern: string
        undefined(default) or func(diff shape, diff pattern) or  Elewise/Conv/...

    Returns
    -------
    decorator : decorator
        decorator to register compute func
    support_bfp16 case:
        decorator to return output_tensor
    """
    if op_type is None:
        raise RuntimeError("register op compute failed, op_type is none")
    global _op_computes
    global _op_register_pattern
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        _op_register_pattern.pop(op_type)
        _op_computes[(op_type, op_mode)] = OpCompute(support_fusion, wrapper)
        return wrapper
    return decorator


def get_op_compute(op_type, op_mode="dynamic"):
    """
    :return:
    """
    if op_type is None:
        raise RuntimeError("get op compute failed, op_type is none")
    return _op_computes.get((op_type, op_mode))


def get_operator(op_type):
    """
    :return:
    """
    if op_type is None:
        raise RuntimeError("get operator failed, op_type is none")
    return _operators.get(op_type)


def register_param_generalization(op_type):
    """
    register op param generalization func

    Parameters
    ----------
    op_type : string
        op type

    Returns
    -------
    decorator : decorator
        decorator to register generalization func
    """
    if op_type is None:
        raise RuntimeError("register generalization func failed, op_type is none")
    global _generalization
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        _generalization[op_type] = wrapper
        return wrapper
    return decorator


def get_param_generalization(op_type):
    """
    :return:
    """
    if op_type is None:
        raise RuntimeError("get generalization func failed, op_type is none")
    return _generalization.get(op_type)


def is_no_need_trans_bool_to_s8(op_type):
    global _op_no_trans_bool_to_s8
    if op_type in _op_no_trans_bool_to_s8.keys():
        return _op_no_trans_bool_to_s8[op_type] == "True"
    else:
        return False