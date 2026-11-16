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
tbe log
"""
import inspect
import logging
import os
import sys

LOG_DEFAULT_LEVEL = logging.INFO

IS_USE_SLOG = True
try:
    from .AscendLog import AscendLog
    S_LOGGER = AscendLog()
except BaseException:
    IS_USE_SLOG = False

    GLOBAL_LOG_ID = "TBE_AUTOSCHEDULE"
    LOGGER = logging.getLogger(GLOBAL_LOG_ID)
    LOGGER.propagate = 0
    LOGGER.setLevel(LOG_DEFAULT_LEVEL)

    STREAM_HANDLER = logging.StreamHandler(stream=sys.stdout)
    STREAM_HANDLER.setLevel(LOG_DEFAULT_LEVEL)
    LOG_FORMAT = "[%(levelname)s][%(asctime)s]%(message)s"
    STREAM_HANDLER.setFormatter(logging.Formatter(LOG_FORMAT))
    LOGGER.addHandler(STREAM_HANDLER)


def info(log_msg, *log_paras):
    """
    info log
    :param log_msg:
    :param log_paras:
    """
    line_no = inspect.currentframe().f_back.f_lineno
    funcname = inspect.currentframe().f_back.f_code.co_name
    co_filename = inspect.currentframe().f_back.f_code.co_filename
    filename = os.path.basename(co_filename)
    log_str = '[%s:%d][%s] ' % (filename, line_no, funcname)
    if log_paras:
        log_msg = log_msg % log_paras
    log_all_msg = log_str + log_msg

    if IS_USE_SLOG:
        S_LOGGER.info(S_LOGGER.module.tbe, log_all_msg)
    else:
        LOGGER.info(log_all_msg)


def debug(log_msg, *log_paras):
    """
    debug log
    :param log_msg:
    :param log_paras:
    """
    line_no = inspect.currentframe().f_back.f_lineno
    funcname = inspect.currentframe().f_back.f_code.co_name
    co_filename = inspect.currentframe().f_back.f_code.co_filename
    filename = os.path.basename(co_filename)
    log_str = '[%s:%d][%s] ' % (filename, line_no, funcname)
    if log_paras:
        log_msg = log_msg % log_paras
    log_all_msg = log_str + log_msg

    if IS_USE_SLOG:
        S_LOGGER.debug(S_LOGGER.module.tbe, log_all_msg)
    else:
        LOGGER.debug(log_all_msg)


def warn(log_msg, *log_paras):
    """
    warning log
    :param log_msg:
    :param log_paras:
    """
    line_no = inspect.currentframe().f_back.f_lineno
    funcname = inspect.currentframe().f_back.f_code.co_name
    co_filename = inspect.currentframe().f_back.f_code.co_filename
    filename = os.path.basename(co_filename)
    log_str = '[%s:%d][%s] ' % (filename, line_no, funcname)
    if log_paras:
        log_msg = log_msg % log_paras
    log_all_msg = log_str + log_msg

    if IS_USE_SLOG:
        S_LOGGER.warn(S_LOGGER.module.tbe, log_all_msg)
    else:
        LOGGER.warning(log_all_msg)


def error(log_msg, *log_paras):
    """
    error log
    :param log_msg:
    :param log_paras:
    """
    line_no = inspect.currentframe().f_back.f_lineno
    funcname = inspect.currentframe().f_back.f_code.co_name
    co_filename = inspect.currentframe().f_back.f_code.co_filename
    filename = os.path.basename(co_filename)
    log_str = '[%s:%d][%s] ' % (filename, line_no, funcname)
    if log_paras:
        log_msg = log_msg % log_paras
    log_all_msg = log_str + log_msg

    if IS_USE_SLOG:
        S_LOGGER.error(S_LOGGER.module.tbe, log_all_msg)
    else:
        LOGGER.error(log_all_msg)


def event(log_msg, *log_paras):
    """
    event log
    :param log_msg:
    :param log_paras:
    """
    line_no = inspect.currentframe().f_back.f_lineno
    funcname = inspect.currentframe().f_back.f_code.co_name
    co_filename = inspect.currentframe().f_back.f_code.co_filename
    filename = os.path.basename(co_filename)
    log_str = '[%s:%d][%s] ' % (filename, line_no, funcname)
    if log_paras:
        log_msg = log_msg % log_paras
    log_all_msg = log_str + log_msg

    if IS_USE_SLOG:
        S_LOGGER.event(S_LOGGER.module.tbe, log_all_msg)
