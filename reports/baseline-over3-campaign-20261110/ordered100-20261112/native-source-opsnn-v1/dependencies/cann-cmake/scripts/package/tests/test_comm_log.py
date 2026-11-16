# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest

from ..utils.comm_log import CommLog


def test_comm_log_timestamp_format():
    """cilog_get_timestamp 返回 '%Y-%m-%d %H:%M:%S' 格式字符串"""
    ts = CommLog.cilog_get_timestamp()
    assert isinstance(ts, str)
    assert len(ts) == 19
    parts = ts.split(' ')
    assert len(parts) == 2
    date_parts = parts[0].split('-')
    assert len(date_parts) == 3
    time_parts = parts[1].split(':')
    assert len(time_parts) == 3


def test_cilog_print_element(capsys):
    """cilog_print_element 输出 '[elem]' 并以空格结尾"""
    CommLog.cilog_print_element("hello")
    captured = capsys.readouterr()
    assert captured.out == "[hello] "


def test_cilog_info_single_param(capsys):
    """cilog_info 单参数格式化正常"""
    CommLog.cilog_info("message %s", "arg1")
    captured = capsys.readouterr()
    assert "message arg1" in captured.out


def test_cilog_warning_single_param(capsys):
    """cilog_warning 单参数格式化正常"""
    CommLog.cilog_warning("warn %s", "arg1")
    captured = capsys.readouterr()
    assert "warn arg1" in captured.out
    assert "[WARNING]" in captured.out


def test_cilog_error_single_param(capsys):
    """cilog_error 单参数格式化正常"""
    CommLog.cilog_error("err %s", "arg1")
    captured = capsys.readouterr()
    assert "err arg1" in captured.out
    assert "[ERROR]" in captured.out


def test_cilog_error_multi_params(capsys):
    """cilog_error 多参数格式化: 'msg %s %s' + ('A', 'B') → 'msg A B'"""
    CommLog.cilog_error("err %s %s", "A", "B")
    captured = capsys.readouterr()
    assert "err A B" in captured.out


def test_cilog_info_multi_params(capsys):
    """cilog_info 多参数格式化正常"""
    CommLog.cilog_info("info %s %s %s", "A", "B", "C")
    captured = capsys.readouterr()
    assert "info A B C" in captured.out


def test_cilog_warning_multi_params(capsys):
    """cilog_warning 多参数格式化正常"""
    CommLog.cilog_warning("warn %s %s", "X", "Y")
    captured = capsys.readouterr()
    assert "warn X Y" in captured.out


def test_cilog_no_format_args(capsys):
    """无格式化占位符时, 消息原样输出"""
    CommLog.cilog_info("plain message")
    captured = capsys.readouterr()
    assert "plain message" in captured.out


def test_cilog_error_no_args_with_placeholder(capsys):
    """有 %s 但无参数时, 降级输出原始消息, 不抛异常"""
    CommLog.cilog_error("msg %s")
    captured = capsys.readouterr()
    assert "msg %s" in captured.out


def test_cilog_info_mismatched_args(capsys):
    """2 个 %s 只传 1 个 arg → 格式化失败, 降级输出原始消息"""
    CommLog.cilog_info("msg %s %s", "only_one")
    captured = capsys.readouterr()
    assert "msg %s %s" in captured.out


def test_cilog_error_too_many_args(capsys):
    """1 个 %s 传 2 个 arg → 格式化失败, 降级输出原始消息"""
    CommLog.cilog_error("msg %s", "A", "B")
    captured = capsys.readouterr()
    assert "msg %s" in captured.out


def test_cilog_logmsg_direct(capsys):
    """直接调用 cilog_logmsg, 验证日志格式含时间戳/级别/文件/行号"""
    CommLog.cilog_logmsg("ERROR", "test_file.py", 42, "direct %s", ("arg",))
    captured = capsys.readouterr()
    assert "[ERROR]" in captured.out
    assert "[test_file.py]" in captured.out
    assert "[42]" in captured.out
    assert "direct arg" in captured.out


def test_cilog_error_in_except_block(capsys):
    """except 块中 cilog_error 格式化失败时, 降级输出原始消息, 不吞掉原始异常"""
    try:
        raise RuntimeError("original business error")
    except Exception:
        CommLog.cilog_error("error: %s %s", "one_arg_only")
    captured = capsys.readouterr()
    assert "error: %s %s" in captured.out


def test_cilog_error_message_containing_percent(capsys):
    """消息含 % 字符时, 降级输出原始消息, 不抛异常"""
    CommLog.cilog_error("path is /usr/local/%lib/runtime")
    captured = capsys.readouterr()
    assert "path is /usr/local/%lib/runtime" in captured.out


def test_cilog_includes_filename_and_lineno(capsys):
    """cilog_error 输出中包含调用者文件名和行号"""
    CommLog.cilog_error("test message")
    captured = capsys.readouterr()
    assert "[ERROR]" in captured.out
    assert "test_comm_log.py" in captured.out

