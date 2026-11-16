#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import time
import inspect
import logging

"""
保留 logging.basicConfig 调用以兼容依赖 logging 模块的第三方库（如 protobuf、grpc 等），
它们会在 import 时检查 logging 配置。CommLog 自身不使用 logging，而是通过 print 直接输出，
因为 CMake 的 execute_process 通过 stdout 捕获输出，使用 logging 可能导致消息被重定向到 stderr。
"""
logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(pathname)s] [line:%(lineno)d] %(message)s',
                    level=logging.INFO)


class CommLog:
    @staticmethod
    def cilog_get_timestamp():
        return time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())

    @staticmethod
    def cilog_print_element(cilog_element):
        print("["+cilog_element+"]", end=' ')
        return

    @staticmethod
    def cilog_logmsg(log_level, filename, line_no, log_msg, *log_paras):
        """
        调用方式有两种：
        1. cilog_error("msg %s %s", arg1, arg2)  — 使用 % 格式化, log_paras 非空
        2. cilog_error(f"msg {var}")              — f-string 已完成格式化, log_paras 为空

        不要混用两种方式（如 cilog_error(f"msg %s", arg)），否则 f-string 中的 % 会被误解析。
        格式化失败时降级输出原始 log_msg，避免吞掉调用方 except 块中的原始异常。
        """
        log_timestamp = CommLog.cilog_get_timestamp()
        CommLog.cilog_print_element(log_timestamp)
        CommLog.cilog_print_element(log_level)
        CommLog.cilog_print_element(filename)
        CommLog.cilog_print_element(str(line_no))
        try:
            if log_paras:
                print(log_msg % log_paras[0])
            else:
                print(log_msg)
        except (TypeError, ValueError):
            print(log_msg)
        return

    @staticmethod
    def cilog_error(log_msg, *log_paras):
        frame = inspect.currentframe().f_back
        line_no = frame.f_lineno
        filename = frame.f_code.co_filename
        CommLog.cilog_logmsg("ERROR", filename, line_no, log_msg, log_paras)
        return

    @staticmethod
    def cilog_warning(log_msg, *log_paras):
        frame = inspect.currentframe().f_back
        line_no = frame.f_lineno
        filename = frame.f_code.co_filename
        CommLog.cilog_logmsg("WARNING", filename, line_no, log_msg, log_paras)
        return

    @staticmethod
    def cilog_info(log_msg, *log_paras):
        frame = inspect.currentframe().f_back
        line_no = frame.f_lineno
        filename = frame.f_code.co_filename
        CommLog.cilog_logmsg("INFO", filename, line_no, log_msg, log_paras)
        return
