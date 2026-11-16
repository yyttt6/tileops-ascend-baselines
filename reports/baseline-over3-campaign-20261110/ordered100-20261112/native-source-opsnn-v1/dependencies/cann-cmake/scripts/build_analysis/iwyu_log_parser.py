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
import argparse
import logging
import os
import re
import stat
import sys
import tempfile
from collections import defaultdict
from typing import Callable

# ===================== 文件扩展名常量 =====================
SOURCE_EXTENSIONS = (".c", ".cpp", ".cc")
HEADER_EXTENSIONS = (".h", ".hpp", ".hh")
ALL_EXTENSIONS = SOURCE_EXTENSIONS + HEADER_EXTENSIONS

# 扩展名正则片段，供IWYU日志区块头正则复用
_EXT_PATTERN = "|".join(ext[1:] for ext in ALL_EXTENSIONS)

# ===================== 正则（完全对齐真实IWYU日志格式） =====================
# 1. should add 区块开头: /path/to/file.c should add these lines:
PATTERN_ADD_BLOCK_HEAD = re.compile(
    rf"^(.+\.(?:{_EXT_PATTERN})) should add these lines:$")
# add区块内完整行（保留末尾//注释）
PATTERN_ADD_INCLUDE_LINE = re.compile(r"^\s*(#include\s*[<\"].*?[>\"].*)$")

# 2. should remove 区块开头: /path/to/file.c should remove these lines:
PATTERN_REMOVE_BLOCK_HEAD = re.compile(
    rf"^(.+\.(?:{_EXT_PATTERN})) should remove these lines:$")
# remove内待删除完整行: - #include "example.h"  // lines xx
PATTERN_REMOVE_INCLUDE_LINE = re.compile(r"^-\s*(#include\s*[<\"].*?[>\"].*)$")

# 3. full list头部：The full include-list for /path/to/file.c:
PATTERN_FULL_BLOCK_HEAD = re.compile(
    rf"^The full include-list for (.+\.(?:{_EXT_PATTERN})):$")
# full区块内完整#include行，保留尾部注释
PATTERN_FULL_INCLUDE_LINE = re.compile(r"^\s*(#include\s*[<\"].*?[>\"].*)$")

# 提取include内引号内容（仅用于修复阶段判断是否系统头文件，展示用完整行）
PATTERN_INCLUDE_TOKEN = re.compile(r"#include\s*([<\"].*?[>\"])")

# 源码原生无注释#include行（用于fix替换）
PATTERN_RAW_INCLUDE_TOKEN = re.compile(r"^\s*#include\s*([<\"].*?[>\"])")
# 脚本生成的注释冗余行 // [iwyu-fix] #include <...> 或 // [iwyu-fix] #include "..."
IWYU_FIX_MARKER = "[iwyu-fix]"
_IWYU_FIX_MARKER_ESCAPED = re.escape(IWYU_FIX_MARKER)
PATTERN_COMMENTED_INCLUDE = re.compile(
    rf"^\s*//\s*{_IWYU_FIX_MARKER_ESCAPED}\s*#include\s*([<\"].*?[>\"])")

FILE_ENCODING = "utf-8"
FILE_ERRORS = "replace"

logger = logging.getLogger(__name__)


class IwyuLogParserError(Exception):
    """IWYU日志解析工具异常，用于替代sys.exit。"""


def atomic_write(filepath: str, content: str) -> None:
    """原子写入文件内容。"""
    dir_name = os.path.dirname(filepath) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=FILE_ENCODING, errors=FILE_ERRORS) as f:
            f.write(content)
        if os.path.exists(filepath):
            orig_mode = os.stat(filepath).st_mode
            os.chmod(tmp_path, stat.S_IMODE(orig_mode))
        os.replace(tmp_path, filepath)
    except OSError:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def get_include_token(full_line: str) -> str:
    """从完整#include行提取<xxx.h>/"xxx.h"用于区分系统头文件。"""
    m = PATTERN_INCLUDE_TOKEN.search(full_line)
    if m:
        return m.group(1)
    return ""


def is_system_header(token: str) -> bool:
    """判断是否<>系统头文件，修复时不处理。"""
    return token.startswith("<") and token.endswith(">")


def build_remove_token_set(remove_lines: list[str]) -> set:
    """从remove行列表中提取include token集合，用于源码匹配。"""
    rm_token_set = set()
    for line in remove_lines:
        tok = get_include_token(line)
        if tok:
            rm_token_set.add(tok)
    return rm_token_set


def annotate_lines(src_lines: list[str], rm_token_set: set) -> tuple[list[str], bool]:
    """逐行注释冗余用户头文件，返回(新行列表, 是否变更)。"""
    changed = False
    new_lines = []
    for line in src_lines:
        s_line = line.strip()
        if PATTERN_COMMENTED_INCLUDE.match(s_line):
            new_lines.append(line)
            continue
        m_tok = PATTERN_RAW_INCLUDE_TOKEN.match(s_line)
        if m_tok:
            inc_token = m_tok.group(1)
            if inc_token in rm_token_set and not is_system_header(inc_token):
                indent = line[:len(line) - len(line.lstrip())]
                new_line = f"{indent}// {IWYU_FIX_MARKER} {s_line}\n"
                new_lines.append(new_line)
                changed = True
                continue
        new_lines.append(line)
    return new_lines, changed


def strip_commented_lines(src_lines: list[str]) -> tuple[list[str], bool]:
    """删除脚本注释生成的// [iwyu-fix] #include行，返回(新行列表, 是否变更)。"""
    changed = False
    output_lines = []
    for line in src_lines:
        if PATTERN_COMMENTED_INCLUDE.match(line.strip()):
            changed = True
            continue
        output_lines.append(line)
    return output_lines, changed


def _log_failed_files(failed: list[tuple[str, str]]) -> None:
    """记录处理失败文件日志。"""
    if failed:
        logger.warning("以下文件处理失败：")
        for fpath, err in failed:
            logger.warning("  - %s: %s", fpath, err)


def _process_file(fpath: str, *, dry_run: bool,
                  transform_fn: Callable[[list[str]], tuple[list[str], bool]]) -> bool:
    """通用文件处理：读取、变换、写回。返回是否发生变更。"""
    try:
        with open(fpath, encoding=FILE_ENCODING, errors=FILE_ERRORS) as f:
            src_lines = f.readlines()
        new_lines, changed = transform_fn(src_lines)
        if changed and not dry_run:
            atomic_write(fpath, "".join(new_lines))
        return changed
    except OSError as e:
        raise IwyuLogParserError(f"{fpath}: {e}") from e


def _make_annotate_fn(rm_token_set: set) -> Callable[[list[str]], tuple[list[str], bool]]:
    """创建绑定特定rm_token_set的annotate_lines闭包。"""
    def _annotate(src_lines: list[str]) -> tuple[list[str], bool]:
        return annotate_lines(src_lines, rm_token_set)
    return _annotate


class IwyuLogParser:
    """IWYU日志解析器，支持解析、报告生成、冗余头文件注释与清理。"""

    def __init__(self) -> None:
        # 文件路径 -> {add: set, remove: set, full: set} 先用集合自动去重
        self.file_data = defaultdict(lambda: {"add": set(), "remove": set(), "full": set()})
        self.cur_file = None
        self.in_add_block = False
        self.in_remove_block = False
        self.in_full_block = False

    @staticmethod
    def clean_commented_lines(all_files: list[str], *, dry_run: bool = False) -> list[str]:
        """--clean-commented：永久删除脚本注释生成的// [iwyu-fix] #include 行。"""
        cleaned = []
        failed = []
        for fpath in all_files:
            if not os.path.exists(fpath) or fpath.endswith(HEADER_EXTENSIONS):
                continue
            try:
                changed = _process_file(
                    fpath, dry_run=dry_run, transform_fn=strip_commented_lines)
                if changed:
                    cleaned.append(fpath)
            except IwyuLogParserError as e:
                failed.append((fpath, str(e)))
        _log_failed_files(failed)
        return cleaned

    @staticmethod
    def _append_file_report(report: list[str], fpath: str, data: dict,
                            stats: dict) -> None:
        """向report追加单个文件的报告段落并更新统计。"""
        add_lines = sorted(data["add"])
        rm_lines = sorted(data["remove"])
        full_lines = sorted(data["full"])
        if not add_lines and not rm_lines and not full_lines:
            return

        is_src = fpath.endswith(SOURCE_EXTENSIONS)
        report.append(f"【文件】{fpath}")

        IwyuLogParser._append_add_section(report, add_lines, is_src=is_src, stats=stats)
        IwyuLogParser._append_remove_section(report, rm_lines, is_src=is_src, stats=stats)

        if full_lines:
            report.append(f"  [3] The full include-list 全部头文件({len(full_lines)}):")
            report.extend(f"      {line}" for line in full_lines)

        report.append("-" * 70)

    @staticmethod
    def _append_add_section(report: list[str], add_lines: list[str],
                            *, is_src: bool, stats: dict) -> None:
        """向report追加should add段落并更新统计。"""
        if not add_lines:
            return
        report.append(f"  [1] should add 建议新增头文件({len(add_lines)}):")
        for line in add_lines:
            report.append(f"      + {line}")
            stats["add"] += 1
            if is_src:
                stats["add_src"] += 1

    @staticmethod
    def _append_remove_section(report: list[str], rm_lines: list[str],
                               *, is_src: bool, stats: dict) -> None:
        """向report追加should remove段落并更新统计。"""
        if not rm_lines:
            return
        report.append(f"  [2] should remove 冗余头文件({len(rm_lines)}):")
        for line in rm_lines:
            report.append(f"      - {line}")
            stats["remove"] += 1
            if is_src:
                stats["remove_src"] += 1
        if is_src:
            stats["modify"] += 1

    @staticmethod
    def _append_summary(report: list[str], stats: dict) -> None:
        """向report追加全局汇总统计。"""
        report.extend([
            "",
            "==== 全局汇总统计 ====",
            f"解析文件总数(已去重): {stats.get('file_count', 0)}",
            f"可修复源码({', '.join(SOURCE_EXTENSIONS)})数量: {stats['modify']}",
            f"建议新增头文件总行数(已去重): {stats['add']} (其中源文件: {stats['add_src']})",
            f"冗余待清理头文件总行数(已去重): {stats['remove']} (其中源文件: {stats['remove_src']})",
            f"修复规则：仅注释双引号自定义头文件，<>系统头文件不修改；"
            f"仅处理{'/'.join(SOURCE_EXTENSIONS)}，{'/'.join(HEADER_EXTENSIONS)}文件不改动",
            "=" * 90,
        ])

    def parse_log(self, log_text: str) -> None:
        """解析IWYU日志文本，填充file_data。"""
        lines = log_text.splitlines()
        for raw_line in lines:
            line = raw_line.rstrip()
            strip_line = line.strip()
            self._parse_line(line, strip_line)

    def generate_report(self) -> str:
        """生成分析报告字符串。"""
        report = ["=" * 90, "IWYU 头文件依赖分析报告", "=" * 90, ""]
        if not self.file_data:
            report.append("【警告】未解析到任何文件数据，请检查日志格式！")
            return "\n".join(report)

        stats = {"add": 0, "remove": 0, "add_src": 0, "remove_src": 0,
                 "modify": 0, "file_count": len(self.file_data)}
        for fpath, data in sorted(self.file_data.items()):
            self._append_file_report(report, fpath, data, stats)

        self._append_summary(report, stats)
        return "\n".join(report)

    def annotate_redundant_user_headers(self, *, dry_run: bool = False) -> list[str]:
        """--fix：仅注释源文件内双引号冗余头文件，<>系统头文件跳过、头文件跳过。"""
        modified = []
        failed = []
        for fpath, data in self.file_data.items():
            if fpath.endswith(HEADER_EXTENSIONS):
                continue
            remove_lines = list(data["remove"])
            if not remove_lines or not os.path.exists(fpath):
                continue
            rm_token_set = build_remove_token_set(remove_lines)
            if not rm_token_set:
                continue
            try:
                changed = _process_file(
                    fpath, dry_run=dry_run,
                    transform_fn=_make_annotate_fn(rm_token_set))
                if changed:
                    modified.append(fpath)
            except IwyuLogParserError as e:
                failed.append((fpath, str(e)))
        _log_failed_files(failed)
        return modified

    def _parse_line(self, line: str, strip_line: str) -> None:
        """解析单行日志，分发到add/remove/full区块。"""
        if not strip_line:
            self.in_add_block = False
            self.in_remove_block = False
            self.in_full_block = False
            return

        if self._try_parse_add_block(line, strip_line):
            return
        if self._try_parse_remove_block(line, strip_line):
            return
        self._try_parse_full_block(line, strip_line)

    def _try_parse_add_block(self, line: str, strip_line: str) -> bool:
        """处理should add区块，返回是否匹配。"""
        m_add_head = PATTERN_ADD_BLOCK_HEAD.match(line)
        if m_add_head:
            self.cur_file = m_add_head.group(1)
            self.in_add_block = True
            self.in_remove_block = False
            self.in_full_block = False
            return True
        if self.in_add_block:
            m_line = PATTERN_ADD_INCLUDE_LINE.match(strip_line)
            if m_line:
                self.file_data[self.cur_file]["add"].add(m_line.group(1))
            return True
        return False

    def _try_parse_remove_block(self, line: str, strip_line: str) -> bool:
        """处理should remove区块，返回是否匹配。"""
        m_rm_head = PATTERN_REMOVE_BLOCK_HEAD.match(line)
        if m_rm_head:
            self.cur_file = m_rm_head.group(1)
            self.in_remove_block = True
            self.in_add_block = False
            self.in_full_block = False
            return True
        if self.in_remove_block:
            m_line = PATTERN_REMOVE_INCLUDE_LINE.match(strip_line)
            if m_line:
                self.file_data[self.cur_file]["remove"].add(m_line.group(1))
            return True
        return False

    def _try_parse_full_block(self, line: str, strip_line: str) -> bool:
        """处理full include-list区块，返回是否匹配。"""
        m_full_head = PATTERN_FULL_BLOCK_HEAD.match(line)
        if m_full_head:
            self.cur_file = m_full_head.group(1)
            self.in_full_block = True
            self.in_add_block = False
            self.in_remove_block = False
            return True
        if self.in_full_block:
            m_line = PATTERN_FULL_INCLUDE_LINE.match(strip_line)
            if m_line:
                self.file_data[self.cur_file]["full"].add(m_line.group(1))
            return True
        return False


def _build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    arg_parser = argparse.ArgumentParser(
        description="IWYU日志解析修复工具（自动去重，报告顺序：add -> remove -> full list）")
    arg_parser.add_argument("log_input", help="IWYU日志文件路径；输入'-'从管道读取")
    arg_parser.add_argument("--report-out", type=str, help="导出报告到指定文件")
    arg_parser.add_argument(
        "--fix", action="store_true",
        help=f"阶段1：注释{'/'.join(SOURCE_EXTENSIONS)}冗余自定义头文件，<>系统头文件跳过")
    arg_parser.add_argument(
        "--clean-commented", action="store_true",
        help="阶段2：彻底删除注释的冗余include行（提交前使用）")
    arg_parser.add_argument(
        "--dry-run", action="store_true",
        help="预览模式：仅展示将要变更的文件，不实际修改")
    return arg_parser


def validate_args(args: argparse.Namespace) -> None:
    """校验参数互斥与组合约束，失败时抛出IwyuLogParserError。"""
    if args.fix and args.clean_commented:
        logger.error("--fix 和 --clean-commented 不能同时使用，请分两步执行：")
        logger.error("  1. 先运行 --fix 注释冗余头文件")
        logger.error("  2. 编译验证无误后，再运行 --clean-commented 清理注释行")
        raise IwyuLogParserError("参数互斥: --fix 与 --clean-commented")
    if args.dry_run and not args.fix and not args.clean_commented:
        logger.warning("--dry-run 需要与 --fix 或 --clean-commented 配合使用，单独使用无实际效果")
        raise IwyuLogParserError("参数无效: --dry-run 需配合 --fix 或 --clean-commented")


def read_log(log_input: str) -> str:
    """读取日志内容，支持'-'从stdin读取。"""
    if log_input == "-":
        try:
            return sys.stdin.read()
        except KeyboardInterrupt:
            logger.error("读取中断")
            raise IwyuLogParserError("读取中断") from None
    if not os.path.isfile(log_input):
        logger.error("日志文件不存在: %s", log_input)
        raise IwyuLogParserError(f"日志文件不存在: {log_input}")
    try:
        with open(log_input, encoding=FILE_ENCODING, errors=FILE_ERRORS) as f:
            return f.read()
    except OSError as e:
        logger.error("读取日志文件失败: %s: %s", log_input, e)
        raise IwyuLogParserError(f"读取日志文件失败: {log_input}: {e}") from e


def _save_report(report_text: str, report_out: str) -> None:
    """保存报告到文件。"""
    try:
        with open(report_out, "w", encoding=FILE_ENCODING, errors=FILE_ERRORS) as f:
            f.write(report_text)
        logger.info("报告已保存至: %s", report_out)
    except OSError as e:
        logger.error("保存报告失败: %s: %s", report_out, e)


def _run_fix(parser: IwyuLogParser, *, dry_run: bool) -> None:
    """执行--fix阶段：注释冗余头文件并输出变更清单。"""
    modified_files = parser.annotate_redundant_user_headers(dry_run=dry_run)
    if dry_run:
        logger.info("==== --fix --dry-run 预览：以下文件将被注释冗余头文件 ====")
    else:
        logger.info("==== --fix 注释冗余自定义头文件 变更清单 ====")
    if modified_files:
        for fp in modified_files:
            logger.info("MODIFIED: %s", fp)
    else:
        logger.info("无需要注释的自定义冗余头文件")
    logger.info("本次修改文件总数: %d", len(modified_files))


def _run_clean(all_files: list[str], *, dry_run: bool) -> None:
    """执行--clean-commented阶段：清理注释行并输出变更清单。"""
    clean_files = IwyuLogParser.clean_commented_lines(all_files, dry_run=dry_run)
    if dry_run:
        logger.info("==== --clean-commented --dry-run 预览：以下文件将被清理注释行 ====")
    else:
        logger.info("==== --clean-commented 永久清理注释行 变更清单 ====")
    if clean_files:
        for fp in clean_files:
            logger.info("CLEANED: %s", fp)
    else:
        logger.info("无注释冗余行可清理")
    logger.info("本次清理文件总数: %d", len(clean_files))


def main() -> None:
    """主入口函数。"""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s")
    args = _build_arg_parser().parse_args()

    try:
        validate_args(args)
        log_content = read_log(args.log_input)

        parser = IwyuLogParser()
        parser.parse_log(log_content)

        report_text = parser.generate_report()
        logger.info("\n%s", report_text)

        if args.report_out and not args.dry_run:
            _save_report(report_text, args.report_out)

        if args.fix:
            _run_fix(parser, dry_run=args.dry_run)

        if args.clean_commented:
            all_file_list = list(parser.file_data.keys())
            _run_clean(all_file_list, dry_run=args.dry_run)
    except IwyuLogParserError:
        sys.exit(1)


if __name__ == "__main__":
    main()
