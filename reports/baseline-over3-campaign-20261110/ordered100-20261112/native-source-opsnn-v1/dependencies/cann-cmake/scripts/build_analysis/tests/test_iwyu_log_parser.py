#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""iwyu_log_parser 单元测试。"""

import argparse
import io
import logging
import os
import stat
import sys
from pathlib import Path

import pytest
from pytest import LogCaptureFixture, MonkeyPatch

import iwyu_log_parser
from iwyu_log_parser import (
    IwyuLogParser,
    IwyuLogParserError,
    IWYU_FIX_MARKER,
    annotate_lines,
    atomic_write,
    build_remove_token_set,
    get_include_token,
    is_system_header,
    read_log,
    strip_commented_lines,
    validate_args,
)


# ===================== 测试辅助 =====================

SAMPLE_LOG = """\
/tmp/test/sample.cpp should add these lines:
#include "missing.h"

/tmp/test/sample.cpp should remove these lines:
- #include "redundant.h"  // lines 2-2

The full include-list for /tmp/test/sample.cpp:
#include <stdio.h>
#include "needed.h"
"""


def make_source_file(tmp_path: Path, name: str, content: str) -> str:
    """在 tmp_path 下创建源文件并返回路径。"""
    fpath = str(tmp_path / name)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content)
    return fpath


def make_iwyu_log(tmp_path: Path, fpath: str,
                  add: list[str] | None = None,
                  remove: list[str] | None = None,
                  full: list[str] | None = None) -> str:
    """生成指向 fpath 的 IWYU 日志文本。"""
    lines = []
    if add:
        lines.append(f"{fpath} should add these lines:")
        lines.extend(add)
        lines.append("")
    if remove:
        lines.append(f"{fpath} should remove these lines:")
        lines.extend(f"- {inc}  // lines 1-1" for inc in remove)
        lines.append("")
    if full:
        lines.append(f"The full include-list for {fpath}:")
        lines.extend(full)
        lines.append("")
    return "\n".join(lines)


# ===================== atomic_write 测试 =====================

class TestAtomicWrite:
    """atomic_write 函数测试。"""

    @staticmethod
    def test_write_new_file(tmp_path: Path) -> None:
        fpath = str(tmp_path / "new.txt")
        atomic_write(fpath, "hello")
        with open(fpath, encoding="utf-8") as f:
            assert f.read() == "hello"

    @staticmethod
    def test_overwrite_existing_file(tmp_path: Path) -> None:
        fpath = str(tmp_path / "exist.txt")
        with open(fpath, "w", encoding="utf-8") as f:
            f.write("old")
        atomic_write(fpath, "new")
        with open(fpath, encoding="utf-8") as f:
            assert f.read() == "new"

    @staticmethod
    def test_preserve_file_mode(tmp_path: Path) -> None:
        fpath = str(tmp_path / "mode.txt")
        with open(fpath, "w", encoding="utf-8") as f:
            f.write("old")
        os.chmod(fpath, 0o640)
        atomic_write(fpath, "new")
        mode = stat.S_IMODE(os.stat(fpath).st_mode)
        assert mode == 0o640

    @staticmethod
    def test_no_tmp_file_left(tmp_path: Path) -> None:
        fpath = str(tmp_path / "clean.txt")
        atomic_write(fpath, "data")
        tmp_files = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
        assert tmp_files == []


# ===================== get_include_token / is_system_header 测试 =====================

class TestStaticHelpers:
    """get_include_token 和 is_system_header 函数测试。"""

    @staticmethod
    @pytest.mark.parametrize("line,expected", [
        ("#include <stdio.h>", "<stdio.h>"),
        ('#include "myheader.h"', '"myheader.h"'),
        ("#include  <a.h>", "<a.h>"),
        ('#include   "b.h"', '"b.h"'),
        ("#include <vector>", "<vector>"),
        ("not an include", ""),
        ("", ""),
    ])
    def test_get_include_token(line: str, expected: str) -> None:
        assert get_include_token(line) == expected

    @staticmethod
    @pytest.mark.parametrize("token,expected", [
        ("<stdio.h>", True),
        ('"myheader.h"', False),
        ("<vector>", True),
        ('"local.h"', False),
        ("", False),
        ("<>", True),
    ])
    def test_is_system_header(*, token: str, expected: bool) -> None:
        assert is_system_header(token) == expected


# ===================== build_remove_token_set 测试 =====================

class TestBuildRemoveTokenSet:
    """build_remove_token_set 函数测试。"""

    @staticmethod
    def test_normal_lines() -> None:
        lines = ['#include "a.h"', "#include <b.h>"]
        result = build_remove_token_set(lines)
        assert result == {'"a.h"', "<b.h>"}

    @staticmethod
    def test_dedup() -> None:
        lines = ['#include "a.h"', '#include "a.h"']
        result = build_remove_token_set(lines)
        assert result == {'"a.h"'}

    @staticmethod
    def test_empty_list() -> None:
        result = build_remove_token_set([])
        assert result == set()

    @staticmethod
    def test_lines_without_include() -> None:
        lines = ["some random text", ""]
        result = build_remove_token_set(lines)
        assert result == set()


# ===================== annotate_lines 测试 =====================

class TestAnnotateLines:
    """annotate_lines 函数测试。"""

    @staticmethod
    def test_comment_redundant_user_header() -> None:
        src = ['#include "redundant.h"\n', '#include "needed.h"\n']
        rm_set = {'"redundant.h"'}
        new_lines, changed = annotate_lines(src, rm_set)
        assert changed is True
        assert IWYU_FIX_MARKER in new_lines[0]
        assert new_lines[1] == '#include "needed.h"\n'

    @staticmethod
    def test_skip_system_header() -> None:
        src = ["#include <redundant.h>\n"]
        rm_set = {"<redundant.h>"}
        new_lines, changed = annotate_lines(src, rm_set)
        assert changed is False
        assert new_lines == src

    @staticmethod
    def test_skip_already_commented() -> None:
        src = [f'// {IWYU_FIX_MARKER} #include "a.h"\n']
        rm_set = {'"a.h"'}
        new_lines, changed = annotate_lines(src, rm_set)
        assert changed is False
        assert new_lines == src

    @staticmethod
    def test_preserve_indentation() -> None:
        src = ['  #include "a.h"\n']
        rm_set = {'"a.h"'}
        new_lines, _ = annotate_lines(src, rm_set)
        assert new_lines[0].startswith("  // ")

    @staticmethod
    def test_no_change_when_no_match() -> None:
        src = ['#include "other.h"\n']
        rm_set = {'"a.h"'}
        new_lines, changed = annotate_lines(src, rm_set)
        assert changed is False
        assert new_lines == src

    @staticmethod
    def test_empty_src() -> None:
        new_lines, changed = annotate_lines([], {'"a.h"'})
        assert changed is False
        assert new_lines == []

    @staticmethod
    def test_empty_rm_set() -> None:
        src = ['#include "a.h"\n']
        new_lines, changed = annotate_lines(src, set())
        assert changed is False
        assert new_lines == src


# ===================== strip_commented_lines 测试 =====================

class TestStripCommentedLines:
    """strip_commented_lines 函数测试。"""

    @staticmethod
    def test_removes_commented_lines() -> None:
        src = [f'// {IWYU_FIX_MARKER} #include "a.h"\n', '#include "b.h"\n']
        new_lines, changed = strip_commented_lines(src)
        assert changed is True
        assert new_lines == ['#include "b.h"\n']

    @staticmethod
    def test_no_markers() -> None:
        src = ['#include "a.h"\n']
        new_lines, changed = strip_commented_lines(src)
        assert changed is False
        assert new_lines == src

    @staticmethod
    def test_empty_src() -> None:
        new_lines, changed = strip_commented_lines([])
        assert changed is False
        assert new_lines == []


# ===================== parse_log 测试 =====================

class TestParseLog:
    """parse_log 解析逻辑测试。"""

    @staticmethod
    def test_parse_full_log() -> None:
        parser = IwyuLogParser()
        parser.parse_log(SAMPLE_LOG)
        key = "/tmp/test/sample.cpp"
        assert key in parser.file_data
        assert '#include "missing.h"' in parser.file_data[key]["add"]
        remove_items = parser.file_data[key]["remove"]
        assert any('#include "redundant.h"' in item for item in remove_items)
        assert "#include <stdio.h>" in parser.file_data[key]["full"]
        assert '#include "needed.h"' in parser.file_data[key]["full"]

    @staticmethod
    def test_parse_empty_log() -> None:
        parser = IwyuLogParser()
        parser.parse_log("")
        assert len(parser.file_data) == 0

    @staticmethod
    def test_parse_no_include_lines() -> None:
        log = "/tmp/test/a.cpp should add these lines:\n\n"
        parser = IwyuLogParser()
        parser.parse_log(log)
        # 区块头匹配但无include行，defaultdict不会创建key
        assert len(parser.file_data) == 0

    @staticmethod
    def test_parse_multiple_files() -> None:
        log = (
            "/tmp/a.cpp should add these lines:\n"
            '#include "a.h"\n'
            "\n"
            "/tmp/b.cpp should remove these lines:\n"
            '- #include "b.h"  // lines 1-1\n'
            "\n"
        )
        parser = IwyuLogParser()
        parser.parse_log(log)
        assert len(parser.file_data) == 2
        assert '#include "a.h"' in parser.file_data["/tmp/a.cpp"]["add"]
        remove_items = parser.file_data["/tmp/b.cpp"]["remove"]
        assert any('#include "b.h"' in item for item in remove_items)

    @staticmethod
    def test_parse_dedup() -> None:
        log = (
            "/tmp/a.cpp should add these lines:\n"
            '#include "dup.h"\n'
            '#include "dup.h"\n'
            "\n"
        )
        parser = IwyuLogParser()
        parser.parse_log(log)
        assert len(parser.file_data["/tmp/a.cpp"]["add"]) == 1

    @staticmethod
    def test_parse_block_transition_on_blank_line() -> None:
        log = (
            "/tmp/a.cpp should add these lines:\n"
            '#include "a.h"\n'
            "\n"
            '#include "not_in_block.h"\n'
        )
        parser = IwyuLogParser()
        parser.parse_log(log)
        assert len(parser.file_data["/tmp/a.cpp"]["add"]) == 1

    @staticmethod
    @pytest.mark.parametrize("ext", [".c", ".cpp", ".cc", ".h", ".hpp", ".hh"])
    def test_parse_all_extensions(ext: str) -> None:
        log = (
            f"/tmp/test/file{ext} should add these lines:\n"
            '#include "a.h"\n'
            "\n"
        )
        parser = IwyuLogParser()
        parser.parse_log(log)
        assert f"/tmp/test/file{ext}" in parser.file_data

    @staticmethod
    def test_parse_non_include_line_in_block_ignored() -> None:
        log = (
            "/tmp/a.cpp should add these lines:\n"
            "this is not an include\n"
            '#include "a.h"\n'
            "\n"
        )
        parser = IwyuLogParser()
        parser.parse_log(log)
        assert len(parser.file_data["/tmp/a.cpp"]["add"]) == 1


# ===================== generate_report 测试 =====================

class TestGenerateReport:
    """generate_report 报告生成测试。"""

    @staticmethod
    def test_empty_data_report() -> None:
        parser = IwyuLogParser()
        report = parser.generate_report()
        assert "IWYU 头文件依赖分析报告" in report
        assert "未解析到任何文件数据" in report

    @staticmethod
    def test_report_contains_file() -> None:
        parser = IwyuLogParser()
        parser.parse_log(SAMPLE_LOG)
        report = parser.generate_report()
        assert "/tmp/test/sample.cpp" in report
        assert "should add" in report
        assert "should remove" in report
        assert "The full include-list" in report

    @staticmethod
    def test_report_counts_source_file() -> None:
        parser = IwyuLogParser()
        parser.parse_log(SAMPLE_LOG)
        report = parser.generate_report()
        assert "可修复源码(.c, .cpp, .cc)数量: 1" in report
        assert "建议新增头文件总行数(已去重): 1" in report
        assert "冗余待清理头文件总行数(已去重): 1" in report

    @staticmethod
    def test_report_header_file_not_counted_as_fixable() -> None:
        log = (
            "/tmp/test/header.h should remove these lines:\n"
            '- #include "x.h"  // lines 1-1\n'
            "\n"
        )
        parser = IwyuLogParser()
        parser.parse_log(log)
        report = parser.generate_report()
        assert "可修复源码(.c, .cpp, .cc)数量: 0" in report

    @staticmethod
    def test_report_skips_file_with_no_data() -> None:
        log = "/tmp/a.cpp should add these lines:\n\n"
        parser = IwyuLogParser()
        parser.parse_log(log)
        report = parser.generate_report()
        # 文件存在但无数据行，不出现在报告明细中
        assert "【文件】" not in report


# ===================== annotate_redundant_user_headers 测试 =====================

class TestAnnotateRedundantUserHeaders:
    """annotate_redundant_user_headers --fix 逻辑测试。"""

    @staticmethod
    def test_fix_modifies_source_file(tmp_path: Path) -> None:
        fpath = make_source_file(tmp_path, "a.cpp",
                                 '#include "redundant.h"\n#include "needed.h"\n')
        log = make_iwyu_log(tmp_path, fpath, remove=['#include "redundant.h"'])
        parser = IwyuLogParser()
        parser.parse_log(log)
        modified = parser.annotate_redundant_user_headers()
        assert fpath in modified
        with open(fpath, encoding="utf-8") as f:
            content = f.read()
        assert IWYU_FIX_MARKER in content
        assert '#include "needed.h"' in content

    @staticmethod
    def test_fix_dry_run_no_write(tmp_path: Path) -> None:
        fpath = make_source_file(tmp_path, "a.cpp", '#include "redundant.h"\n')
        log = make_iwyu_log(tmp_path, fpath, remove=['#include "redundant.h"'])
        parser = IwyuLogParser()
        parser.parse_log(log)
        modified = parser.annotate_redundant_user_headers(dry_run=True)
        assert fpath in modified
        with open(fpath, encoding="utf-8") as f:
            content = f.read()
        assert content == '#include "redundant.h"\n'

    @staticmethod
    def test_fix_skips_header_file(tmp_path: Path) -> None:
        fpath = make_source_file(tmp_path, "a.h", '#include "redundant.h"\n')
        log = make_iwyu_log(tmp_path, fpath, remove=['#include "redundant.h"'])
        parser = IwyuLogParser()
        parser.parse_log(log)
        modified = parser.annotate_redundant_user_headers()
        assert fpath not in modified

    @staticmethod
    def test_fix_skips_nonexistent_file(tmp_path: Path) -> None:
        fpath = str(tmp_path / "nonexist.cpp")
        log = make_iwyu_log(tmp_path, fpath, remove=['#include "redundant.h"'])
        parser = IwyuLogParser()
        parser.parse_log(log)
        modified = parser.annotate_redundant_user_headers()
        assert fpath not in modified

    @staticmethod
    def test_fix_skips_system_header(tmp_path: Path) -> None:
        fpath = make_source_file(tmp_path, "a.cpp", "#include <redundant.h>\n")
        log = make_iwyu_log(tmp_path, fpath, remove=["#include <redundant.h>"])
        parser = IwyuLogParser()
        parser.parse_log(log)
        modified = parser.annotate_redundant_user_headers()
        assert fpath not in modified
        with open(fpath, encoding="utf-8") as f:
            assert f.read() == "#include <redundant.h>\n"

    @staticmethod
    def test_fix_no_redundant_in_source(tmp_path: Path) -> None:
        fpath = make_source_file(tmp_path, "a.cpp", '#include "needed.h"\n')
        log = make_iwyu_log(tmp_path, fpath, remove=['#include "other.h"'])
        parser = IwyuLogParser()
        parser.parse_log(log)
        modified = parser.annotate_redundant_user_headers()
        assert fpath not in modified

    @staticmethod
    def test_fix_preserves_indentation(tmp_path: Path) -> None:
        fpath = make_source_file(tmp_path, "a.cpp", '  #include "redundant.h"\n')
        log = make_iwyu_log(tmp_path, fpath, remove=['#include "redundant.h"'])
        parser = IwyuLogParser()
        parser.parse_log(log)
        parser.annotate_redundant_user_headers()
        with open(fpath, encoding="utf-8") as f:
            content = f.read()
        assert content.startswith("  // ")


# ===================== clean_commented_lines 测试 =====================

class TestCleanCommentedLines:
    """clean_commented_lines --clean-commented 逻辑测试。"""

    @staticmethod
    def test_clean_removes_commented_lines(tmp_path: Path) -> None:
        fpath = make_source_file(
            tmp_path, "a.cpp",
            f'// {IWYU_FIX_MARKER} #include "a.h"\n#include "b.h"\n')
        cleaned = IwyuLogParser.clean_commented_lines([fpath])
        assert fpath in cleaned
        with open(fpath, encoding="utf-8") as f:
            content = f.read()
        assert content == '#include "b.h"\n'

    @staticmethod
    def test_clean_dry_run_no_write(tmp_path: Path) -> None:
        original = f'// {IWYU_FIX_MARKER} #include "a.h"\n'
        fpath = make_source_file(tmp_path, "a.cpp", original)
        cleaned = IwyuLogParser.clean_commented_lines([fpath], dry_run=True)
        assert fpath in cleaned
        with open(fpath, encoding="utf-8") as f:
            assert f.read() == original

    @staticmethod
    def test_clean_skips_header_file(tmp_path: Path) -> None:
        fpath = make_source_file(
            tmp_path, "a.h",
            f'// {IWYU_FIX_MARKER} #include "a.h"\n')
        cleaned = IwyuLogParser.clean_commented_lines([fpath])
        assert fpath not in cleaned

    @staticmethod
    def test_clean_skips_nonexistent_file(tmp_path: Path) -> None:
        fpath = str(tmp_path / "nonexist.cpp")
        cleaned = IwyuLogParser.clean_commented_lines([fpath])
        assert cleaned == []

    @staticmethod
    def test_clean_no_markers(tmp_path: Path) -> None:
        fpath = make_source_file(tmp_path, "a.cpp", '#include "a.h"\n')
        cleaned = IwyuLogParser.clean_commented_lines([fpath])
        assert fpath not in cleaned

    @staticmethod
    def test_clean_preserves_normal_comments(tmp_path: Path) -> None:
        fpath = make_source_file(
            tmp_path, "a.cpp",
            '// #include "user_comment.h"\n'
            f'// {IWYU_FIX_MARKER} #include "redundant.h"\n')
        cleaned = IwyuLogParser.clean_commented_lines([fpath])
        assert fpath in cleaned
        with open(fpath, encoding="utf-8") as f:
            content = f.read()
        assert '// #include "user_comment.h"' in content
        assert IWYU_FIX_MARKER not in content

    @staticmethod
    def test_clean_empty_file_list() -> None:
        cleaned = IwyuLogParser.clean_commented_lines([])
        assert cleaned == []


# ===================== main / CLI 测试 =====================

class TestMainCLI:
    """main 函数与命令行参数测试。"""

    @staticmethod
    def test_mutex_fix_and_clean_commented(caplog: LogCaptureFixture) -> None:
        args = argparse.Namespace(fix=True, clean_commented=True, dry_run=False,
                                  log_input="dummy.log", report_out=None)
        with pytest.raises(IwyuLogParserError):
            validate_args(args)
        assert "不能同时使用" in caplog.text

    @staticmethod
    def test_dry_run_alone_exits(caplog: LogCaptureFixture) -> None:
        args = argparse.Namespace(fix=False, clean_commented=False, dry_run=True,
                                  log_input="dummy.log", report_out=None)
        with pytest.raises(IwyuLogParserError):
            validate_args(args)
        assert "dry-run" in caplog.text

    @staticmethod
    def test_missing_log_file(caplog: LogCaptureFixture) -> None:
        with pytest.raises(IwyuLogParserError):
            read_log("/nonexistent/path.log")
        assert "日志文件不存在" in caplog.text

    @staticmethod
    def test_report_only(tmp_path: Path, caplog: LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO)
        log_file = tmp_path / "test.log"
        log_file.write_text(SAMPLE_LOG, encoding="utf-8")
        sys.argv = ["prog", str(log_file)]
        iwyu_log_parser.main()
        assert "IWYU 头文件依赖分析报告" in caplog.text
        assert "/tmp/test/sample.cpp" in caplog.text

    @staticmethod
    def test_report_out_saves_file(tmp_path: Path, caplog: LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO)
        log_file = tmp_path / "test.log"
        log_file.write_text(SAMPLE_LOG, encoding="utf-8")
        report_file = tmp_path / "report.txt"
        sys.argv = ["prog", str(log_file), "--report-out", str(report_file)]
        iwyu_log_parser.main()
        assert "报告已保存至" in caplog.text
        assert report_file.exists()
        content = report_file.read_text(encoding="utf-8")
        assert "IWYU 头文件依赖分析报告" in content

    @staticmethod
    def test_fix_flow(tmp_path: Path, caplog: LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO)
        fpath = make_source_file(tmp_path, "a.cpp",
                                 '#include "redundant.h"\n#include "needed.h"\n')
        log_file = tmp_path / "test.log"
        log_file.write_text(
            make_iwyu_log(tmp_path, fpath, remove=['#include "redundant.h"']),
            encoding="utf-8")
        sys.argv = ["prog", str(log_file), "--fix"]
        iwyu_log_parser.main()
        assert "MODIFIED:" in caplog.text
        assert fpath in caplog.text
        with open(fpath, encoding="utf-8") as f:
            assert IWYU_FIX_MARKER in f.read()

    @staticmethod
    def test_fix_dry_run_flow(tmp_path: Path, caplog: LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO)
        original = '#include "redundant.h"\n'
        fpath = make_source_file(tmp_path, "a.cpp", original)
        log_file = tmp_path / "test.log"
        log_file.write_text(
            make_iwyu_log(tmp_path, fpath, remove=['#include "redundant.h"']),
            encoding="utf-8")
        sys.argv = ["prog", str(log_file), "--fix", "--dry-run"]
        iwyu_log_parser.main()
        assert "MODIFIED:" in caplog.text
        assert "dry-run 预览" in caplog.text
        with open(fpath, encoding="utf-8") as f:
            assert f.read() == original

    @staticmethod
    def test_clean_commented_flow(tmp_path: Path, caplog: LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO)
        fpath = make_source_file(
            tmp_path, "a.cpp",
            f'// {IWYU_FIX_MARKER} #include "redundant.h"\n#include "needed.h"\n')
        log_file = tmp_path / "test.log"
        log_file.write_text(
            make_iwyu_log(tmp_path, fpath, remove=['#include "redundant.h"']),
            encoding="utf-8")
        sys.argv = ["prog", str(log_file), "--clean-commented"]
        iwyu_log_parser.main()
        assert "CLEANED:" in caplog.text
        with open(fpath, encoding="utf-8") as f:
            content = f.read()
        assert content == '#include "needed.h"\n'

    @staticmethod
    def test_stdin_input(caplog: LogCaptureFixture, monkeypatch: MonkeyPatch) -> None:
        caplog.set_level(logging.INFO)
        monkeypatch.setattr(sys, "stdin", io.StringIO(SAMPLE_LOG))
        sys.argv = ["prog", "-"]
        iwyu_log_parser.main()
        assert "IWYU 头文件依赖分析报告" in caplog.text

    @staticmethod
    def test_fix_no_modification_message(tmp_path: Path, caplog: LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO)
        fpath = make_source_file(tmp_path, "a.cpp", '#include "needed.h"\n')
        log_file = tmp_path / "test.log"
        log_file.write_text(
            make_iwyu_log(tmp_path, fpath, remove=['#include "nonexist.h"']),
            encoding="utf-8")
        sys.argv = ["prog", str(log_file), "--fix"]
        iwyu_log_parser.main()
        assert "无需要注释的自定义冗余头文件" in caplog.text

    @staticmethod
    def test_clean_no_modification_message(tmp_path: Path, caplog: LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO)
        fpath = make_source_file(tmp_path, "a.cpp", '#include "a.h"\n')
        log_file = tmp_path / "test.log"
        log_file.write_text(SAMPLE_LOG, encoding="utf-8")
        sys.argv = ["prog", str(log_file), "--clean-commented"]
        iwyu_log_parser.main()
        assert "无注释冗余行可清理" in caplog.text


# ===================== 端到端流程测试 =====================

class TestEndToEnd:
    """fix -> clean-commented 完整两阶段流程测试。"""

    @staticmethod
    def test_fix_then_clean_restores_clean_source(tmp_path: Path) -> None:
        original = '#include <stdio.h>\n#include "redundant.h"\n#include "needed.h"\n'
        fpath = make_source_file(tmp_path, "e2e.cpp", original)
        log_file = tmp_path / "e2e.log"
        log_file.write_text(
            make_iwyu_log(tmp_path, fpath,
                          remove=['#include "redundant.h"']),
            encoding="utf-8")

        # 阶段1: --fix
        sys.argv = ["prog", str(log_file), "--fix"]
        iwyu_log_parser.main()
        with open(fpath, encoding="utf-8") as f:
            after_fix = f.read()
        assert IWYU_FIX_MARKER in after_fix
        assert '#include "redundant.h"' in after_fix  # 被注释但仍存在

        # 阶段2: --clean-commented
        sys.argv = ["prog", str(log_file), "--clean-commented"]
        iwyu_log_parser.main()
        with open(fpath, encoding="utf-8") as f:
            after_clean = f.read()
        assert IWYU_FIX_MARKER not in after_clean
        assert '#include "redundant.h"' not in after_clean
        assert "#include <stdio.h>" in after_clean
        assert '#include "needed.h"' in after_clean
