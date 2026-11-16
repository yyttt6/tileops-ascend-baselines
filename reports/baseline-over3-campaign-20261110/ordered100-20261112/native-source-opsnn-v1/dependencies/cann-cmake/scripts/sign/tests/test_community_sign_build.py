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

"""community_sign_build.py 单元测试。

测试覆盖：
- 模块常量
- 查询 flag（--print-sign-ext / --print-certtype）：打印后退出/不执行签名/优先级/nargs
- get_sign_cmd：签名命令构建
- check_result：p7s 文件校验（存在 + 非空）
- prepare_crl：CRL 准备（本地优先/下载成功/重试/全部失败/超时）
- sign_single_file：单文件签名（成功/失败/超时）
- run_sign：签名主流程（成功/跳过/失败/CRL只准备一次）
- main：入口（成功/签名失败/校验失败/参数不足/返回退出码/查询模式）
"""

import os
from contextlib import contextmanager
from subprocess import CompletedProcess, TimeoutExpired
from unittest import mock

import pytest
import community_sign_build


@contextmanager
def mock_crl_download_ok(getsize=100):
    """Mock CRL 文件存在且非空（isfile=True, getsize 默认 100）。"""
    with (
        mock.patch("community_sign_build.os.path.isfile", return_value=True),
        mock.patch("community_sign_build.os.path.getsize", return_value=getsize),
    ):
        yield


@contextmanager
def mock_crl_at_path(expected_crl, getsize=100):
    """Mock CRL 文件仅对 expected_crl 路径返回存在且非空。"""

    def isfile_side(path):
        return path == expected_crl

    with (
        mock.patch("community_sign_build.os.path.isfile", side_effect=isfile_side),
        mock.patch("community_sign_build.os.path.getsize", return_value=getsize),
    ):
        yield


# ===================== 模块常量 =====================


class TestConstants:
    """模块级常量校验。"""

    @staticmethod
    def test_crl_filename():
        assert community_sign_build.CRL_FILENAME == "SWSCRL.crl"

    @staticmethod
    def test_cms_tag():
        assert community_sign_build.CMS_TAG == ".p7s"

    @staticmethod
    def test_cert_type():
        assert community_sign_build.CERT_TYPE == "0x1"

    @staticmethod
    def test_subprocess_timeout_positive():
        assert community_sign_build.SUBPROCESS_TIMEOUT > 0

    @staticmethod
    def test_sign_client_path_nonempty():
        assert community_sign_build.SIGN_CLIENT_PATH

    @staticmethod
    def test_sign_client_config_nonempty():
        assert community_sign_build.SIGN_CLIENT_CONFIG

    @staticmethod
    def test_crl_download_url_nonempty():
        assert community_sign_build.CRL_DOWNLOAD_URL

    @staticmethod
    def test_crl_download_retries_positive():
        assert community_sign_build.CRL_DOWNLOAD_RETRIES > 0


# ===================== 查询 flag（--print-sign-ext / --print-certtype）=====================


class TestPrintSignType:
    """查询 flag：--print-sign-ext / --print-certtype。

    编排器通过这两个 flag 查询签名脚本的产物扩展名与证书类型，
    脚本打印后立即退出，不执行签名、不下载 CRL、不调用 signatrust_client。
    """

    @staticmethod
    @mock.patch("community_sign_build.run_sign")
    def test_print_sign_ext_outputs_p7s(mock_run_sign):
        """--print-sign-ext 打印 .p7s 并返回 True，不执行签名。"""
        assert community_sign_build.main(["script.py", "--print-sign-ext"]) is True
        mock_run_sign.assert_not_called()

    @staticmethod
    @mock.patch("community_sign_build.run_sign")
    def test_print_certtype_outputs_0x1(mock_run_sign):
        """--print-certtype 打印 0x1 并返回 True，不执行签名。"""
        assert community_sign_build.main(["script.py", "--print-certtype"]) is True
        mock_run_sign.assert_not_called()

    @staticmethod
    @mock.patch("community_sign_build.check_result", return_value=True)
    @mock.patch("community_sign_build.run_sign")
    def test_print_sign_ext_priority_over_certtype(mock_run_sign, mock_check):
        """--print-sign-ext 与 --print-certtype 同时传入时 ext 优先打印并退出。"""
        assert (
            community_sign_build.main(
                ["script.py", "--print-sign-ext", "--print-certtype"]
            )
            is True
        )
        mock_run_sign.assert_not_called()

    @staticmethod
    @mock.patch("community_sign_build.run_sign")
    def test_print_sign_ext_not_call_signatrust(mock_run_sign, tmp_path):
        """查询模式不调用 signatrust_client（run_sign 未被调用）。"""
        assert (
            community_sign_build.main(
                ["script.py", "--print-sign-ext", "--crl-dir", str(tmp_path)]
            )
            is True
        )
        mock_run_sign.assert_not_called()

    @staticmethod
    def test_sign_mode_no_files_returns_false():
        """签名模式不带 files 参数返回 False（显式校验）。"""
        assert community_sign_build.main(["script.py", "--crl-dir", "/tmp"]) is False

    @staticmethod
    @mock.patch("community_sign_build.check_result", return_value=True)
    @mock.patch("community_sign_build.run_sign")
    def test_sign_mode_with_files_not_affected_by_query_flags(
        mock_run_sign, mock_check
    ):
        """签名模式（带 files、不带查询 flag）行为与改动前一致。"""
        mock_run_sign.return_value = (True, ["/tmp/a.bin"])
        assert community_sign_build.main(["script.py", "/tmp/a.bin"]) is True
        mock_run_sign.assert_called_once()

    @staticmethod
    def test_define_parser_has_print_sign_ext():
        """define_parser 包含 --print-sign-ext flag。"""
        parser = community_sign_build.define_parser()
        args = parser.parse_args(["--print-sign-ext"])
        assert args.print_sign_ext is True

    @staticmethod
    def test_define_parser_has_print_certtype():
        """define_parser 包含 --print-certtype flag。"""
        parser = community_sign_build.define_parser()
        args = parser.parse_args(["--print-certtype"])
        assert args.print_certtype is True

    @staticmethod
    def test_define_parser_files_nargs_star():
        """files 参数为 nargs='*'，允许查询模式不带文件。"""
        parser = community_sign_build.define_parser()
        args = parser.parse_args([])
        assert args.files == []

    @staticmethod
    def test_define_parser_query_flags_store_true():
        """查询 flag 为 store_true，默认 False。"""
        parser = community_sign_build.define_parser()
        set_args = parser.parse_args(["--print-sign-ext", "--print-certtype"])
        assert set_args.print_sign_ext is True
        assert set_args.print_certtype is True
        default_args = parser.parse_args([])
        assert default_args.print_sign_ext is False
        assert default_args.print_certtype is False


CRL_PATH = "/tmp/SWSCRL.crl"


# ===================== get_sign_cmd =====================


class TestGetSignCmd:
    """签名命令构建。"""

    @staticmethod
    def test_different_crl_paths_produce_different_cmds():
        cmd_a = community_sign_build.get_sign_cmd("/tmp/foo.bin", "/tmp/a.crl")
        cmd_b = community_sign_build.get_sign_cmd("/tmp/foo.bin", "/tmp/b.crl")
        assert cmd_a != cmd_b

    @staticmethod
    def test_returns_list():
        cmd = community_sign_build.get_sign_cmd("/tmp/foo.bin", CRL_PATH)
        assert isinstance(cmd, list)

    @staticmethod
    def test_first_element_is_client():
        cmd = community_sign_build.get_sign_cmd("/tmp/foo.bin", CRL_PATH)
        assert cmd[0] == community_sign_build.SIGN_CLIENT_PATH

    @staticmethod
    def test_contains_config():
        cmd = community_sign_build.get_sign_cmd("/tmp/foo.bin", CRL_PATH)
        assert "--config" in cmd
        idx = cmd.index("--config")
        assert cmd[idx + 1] == community_sign_build.SIGN_CLIENT_CONFIG

    @staticmethod
    def test_contains_input_file():
        cmd = community_sign_build.get_sign_cmd("/tmp/foo.bin", CRL_PATH)
        assert "/tmp/foo.bin" in cmd

    @staticmethod
    def test_contains_required_flags():
        cmd = community_sign_build.get_sign_cmd("/tmp/foo.bin", CRL_PATH)
        expected = (
            "add --file-type p7s --key-type x509 --key-name "
            "SignCert --detached --timestamp-key TimeCert --crl"
        ).split()
        for token in expected:
            assert token in cmd, "missing token: %s" % token

    @staticmethod
    def test_contains_crl_path():
        cmd = community_sign_build.get_sign_cmd("/tmp/foo.bin", CRL_PATH)
        idx = cmd.index("--crl")
        assert idx + 1 < len(cmd)
        assert cmd[idx + 1] == CRL_PATH

    @staticmethod
    def test_different_files_produce_different_cmds():
        cmd_a = community_sign_build.get_sign_cmd("/tmp/a.bin", CRL_PATH)
        cmd_b = community_sign_build.get_sign_cmd("/tmp/b.bin", CRL_PATH)
        assert cmd_a != cmd_b


# ===================== check_result =====================


class TestCheckResult:
    """p7s 签名文件校验（存在 + 非空）。"""

    @staticmethod
    def test_all_p7s_exist_and_nonempty(tmp_path):
        """所有 p7s 文件存在且非空 → True。"""
        f1 = tmp_path / "a.bin"
        f1.touch()
        (tmp_path / "a.bin.p7s").write_text("signature")
        f2 = tmp_path / "b.bin"
        f2.touch()
        (tmp_path / "b.bin.p7s").write_text("signature")
        assert community_sign_build.check_result([str(f1), str(f2)]) is True

    @staticmethod
    def test_missing_p7s(tmp_path):
        """缺少 p7s 文件 → False。"""
        f1 = tmp_path / "a.bin"
        f1.touch()
        (tmp_path / "a.bin.p7s").write_text("signature")
        f2 = tmp_path / "b.bin"
        f2.touch()
        assert community_sign_build.check_result([str(f1), str(f2)]) is False

    @staticmethod
    def test_empty_p7s_file(tmp_path):
        """p7s 文件存在但为空 → False。"""
        f = tmp_path / "x.bin"
        f.touch()
        (tmp_path / "x.bin.p7s").touch()
        assert community_sign_build.check_result([str(f)]) is False

    @staticmethod
    def test_empty_list():
        """空文件列表 → False（无文件被签名，不应视为成功）。"""
        assert community_sign_build.check_result([]) is False

    @staticmethod
    def test_single_file_p7s_exists(tmp_path):
        """单个文件 p7s 存在且非空 → True。"""
        f = tmp_path / "x.bin"
        f.touch()
        (tmp_path / "x.bin.p7s").write_text("signature")
        assert community_sign_build.check_result([str(f)]) is True

    @staticmethod
    def test_single_file_p7s_missing(tmp_path):
        """单个文件 p7s 缺失 → False。"""
        f = tmp_path / "x.bin"
        f.touch()
        assert community_sign_build.check_result([str(f)]) is False


# ===================== prepare_crl =====================


class TestPrepareCrl:
    """CRL 文件准备（本地优先 / 远程下载 / 重试）。"""

    @staticmethod
    def test_use_local_crl_when_configured_and_exists(tmp_path, monkeypatch):
        """用户配置 CRL_FILE_PATH 且文件存在 → 直接使用，不下载。"""
        crl = tmp_path / "SWSCRL.crl"
        crl.touch()
        monkeypatch.setenv("CRL_FILE_PATH", str(crl))
        with mock.patch("community_sign_build.subprocess.run") as mock_run:
            result = community_sign_build.prepare_crl()
            assert result == str(crl)
            mock_run.assert_not_called()

    @staticmethod
    @mock.patch("community_sign_build.subprocess.run")
    def test_fallback_to_download_when_local_not_exist(mock_run, tmp_path, monkeypatch):
        """用户配置了路径但文件不存在 → 回退下载。"""
        monkeypatch.setenv("CRL_FILE_PATH", "/nonexistent/path/SWSCRL.crl")
        mock_run.return_value = CompletedProcess(args=[], returncode=0)
        crl_path = os.path.join(
            community_sign_build.SCRIPT_DIR, community_sign_build.CRL_FILENAME
        )

        with mock_crl_at_path(crl_path):
            result = community_sign_build.prepare_crl()
        assert result is not None
        assert result.endswith(community_sign_build.CRL_FILENAME)
        mock_run.assert_called_once()

    @staticmethod
    @mock.patch("community_sign_build.subprocess.run")
    def test_download_success_when_no_local_config(mock_run):
        """未配置 CRL_FILE_PATH → 下载到脚本目录，返回路径。"""
        mock_run.return_value = CompletedProcess(args=[], returncode=0)
        with mock_crl_download_ok():
            result = community_sign_build.prepare_crl()
        assert result is not None
        assert result.endswith(community_sign_build.CRL_FILENAME)
        mock_run.assert_called_once()

    @staticmethod
    @mock.patch("community_sign_build.subprocess.run")
    def test_download_retry_then_success(mock_run):
        """首次下载失败，重试成功 → 返回路径。"""
        mock_run.side_effect = [
            CompletedProcess(args=[], returncode=1),
            CompletedProcess(args=[], returncode=0),
        ]
        with mock_crl_download_ok():
            result = community_sign_build.prepare_crl()
        assert result is not None
        assert mock_run.call_count == 2

    @staticmethod
    @mock.patch("community_sign_build.subprocess.run")
    def test_download_all_retries_fail(mock_run):
        """所有重试均失败 → None。"""
        mock_run.return_value = CompletedProcess(args=[], returncode=1)
        result = community_sign_build.prepare_crl()
        assert result is None
        assert mock_run.call_count == community_sign_build.CRL_DOWNLOAD_RETRIES

    @staticmethod
    @mock.patch("community_sign_build.subprocess.run")
    def test_download_timeout_then_success(mock_run):
        """首次超时，重试成功 → 返回路径。"""
        mock_run.side_effect = [
            TimeoutExpired("curl", 600),
            CompletedProcess(args=[], returncode=0),
        ]
        with mock_crl_download_ok():
            result = community_sign_build.prepare_crl()
        assert result is not None
        assert mock_run.call_count == 2

    @staticmethod
    @mock.patch("community_sign_build.subprocess.run")
    def test_download_all_timeout(mock_run):
        """所有重试均超时 → None。"""
        mock_run.side_effect = TimeoutExpired("curl", 600)
        result = community_sign_build.prepare_crl()
        assert result is None
        assert mock_run.call_count == community_sign_build.CRL_DOWNLOAD_RETRIES

    @staticmethod
    @mock.patch("community_sign_build.subprocess.run")
    def test_download_uses_curl(mock_run):
        """下载命令使用 curl。"""
        mock_run.return_value = CompletedProcess(args=[], returncode=0)
        with mock_crl_download_ok():
            community_sign_build.prepare_crl()
        args, _ = mock_run.call_args
        assert args[0][0] == "curl"

    @staticmethod
    @mock.patch("community_sign_build.subprocess.run")
    def test_download_url_in_cmd(mock_run):
        """下载命令包含 CRL_DOWNLOAD_URL。"""
        mock_run.return_value = CompletedProcess(args=[], returncode=0)
        with mock_crl_download_ok():
            community_sign_build.prepare_crl()
        args, _ = mock_run.call_args
        assert community_sign_build.CRL_DOWNLOAD_URL in args[0]

    @staticmethod
    @mock.patch("community_sign_build.subprocess.run")
    def test_download_shell_false(mock_run):
        """下载调用使用 shell=False。"""
        mock_run.return_value = CompletedProcess(args=[], returncode=0)
        with mock_crl_download_ok():
            community_sign_build.prepare_crl()
        _, kwargs = mock_run.call_args
        assert kwargs.get("shell") is False

    @staticmethod
    @mock.patch("community_sign_build.subprocess.run")
    def test_download_empty_crl_retries(mock_run):
        """P1: curl 返回码 0 但下载的 CRL 文件为空 → 重试。"""
        mock_run.return_value = CompletedProcess(args=[], returncode=0)
        with mock_crl_download_ok(getsize=0):
            result = community_sign_build.prepare_crl()
        assert result is None
        assert mock_run.call_count == community_sign_build.CRL_DOWNLOAD_RETRIES

    @pytest.fixture(autouse=True)
    def mock_sleep(self):
        """避免重试退避的 time.sleep 拖慢测试。"""
        with mock.patch("community_sign_build.time.sleep"):
            yield

    @pytest.fixture(autouse=True)
    def clear_crl_env(self, monkeypatch):
        """默认清除 CRL_FILE_PATH 环境变量，测试需设置时用 monkeypatch.setenv。"""
        monkeypatch.delenv("CRL_FILE_PATH", raising=False)


# ===================== sign_single_file =====================


class TestSignSingleFile:
    """单文件签名。"""

    @staticmethod
    @mock.patch("community_sign_build.subprocess.run")
    def test_success(mock_run, tmp_path):
        """签名成功 → True。"""
        f = tmp_path / "a.bin"
        f.touch()
        mock_run.return_value = CompletedProcess(args=[], returncode=0, stdout="")
        assert community_sign_build.sign_single_file(str(f), CRL_PATH) is True

    @staticmethod
    @mock.patch("community_sign_build.subprocess.run")
    def test_returncode_nonzero(mock_run, tmp_path):
        """signatrust 返回非零 → False。"""
        f = tmp_path / "a.bin"
        f.touch()
        mock_run.return_value = CompletedProcess(args=[], returncode=1, stdout="error")
        assert community_sign_build.sign_single_file(str(f), CRL_PATH) is False

    @staticmethod
    @mock.patch("community_sign_build.subprocess.run")
    def test_timeout(mock_run, tmp_path):
        """签名超时 → False，不抛异常。"""
        f = tmp_path / "a.bin"
        f.touch()
        mock_run.side_effect = TimeoutExpired("signatrust", 600)
        assert community_sign_build.sign_single_file(str(f), CRL_PATH) is False

    @staticmethod
    @mock.patch("community_sign_build.subprocess.run")
    def test_shell_false(mock_run, tmp_path):
        """签名调用使用 shell=False。"""
        f = tmp_path / "a.bin"
        f.touch()
        mock_run.return_value = CompletedProcess(args=[], returncode=0, stdout="")
        community_sign_build.sign_single_file(str(f), CRL_PATH)
        _, kwargs = mock_run.call_args
        assert kwargs.get("shell") is False

    @staticmethod
    @mock.patch("community_sign_build.subprocess.run")
    def test_timeout_passed(mock_run, tmp_path):
        """签名调用传入 timeout 参数。"""
        f = tmp_path / "a.bin"
        f.touch()
        mock_run.return_value = CompletedProcess(args=[], returncode=0, stdout="")
        community_sign_build.sign_single_file(str(f), CRL_PATH)
        _, kwargs = mock_run.call_args
        assert kwargs.get("timeout") == community_sign_build.SUBPROCESS_TIMEOUT

    @staticmethod
    @mock.patch("community_sign_build.subprocess.run")
    def test_crl_path_in_cmd(mock_run, tmp_path):
        """签名命令包含传入的 crl_path。"""
        f = tmp_path / "a.bin"
        f.touch()
        mock_run.return_value = CompletedProcess(args=[], returncode=0, stdout="")
        community_sign_build.sign_single_file(str(f), CRL_PATH)
        args, _ = mock_run.call_args
        assert CRL_PATH in args[0]


# ===================== run_sign =====================


class TestRunSign:
    """签名主流程。"""

    @staticmethod
    @mock.patch("community_sign_build.subprocess.run")
    @mock.patch("community_sign_build.prepare_crl", return_value=CRL_PATH)
    def test_all_files_signed(mock_prepare, mock_run, tmp_path):
        """所有文件签名成功 → (True, signed_files)。"""
        f1 = tmp_path / "a.bin"
        f1.touch()
        f2 = tmp_path / "b.bin"
        f2.touch()
        mock_run.return_value = CompletedProcess(args=[], returncode=0, stdout="")
        ret, signed = community_sign_build.run_sign([str(f1), str(f2)])
        assert ret is True
        assert sorted(signed) == sorted([str(f1), str(f2)])
        assert mock_run.call_count == 2

    @staticmethod
    @mock.patch("community_sign_build.subprocess.run")
    @mock.patch("community_sign_build.prepare_crl", return_value=CRL_PATH)
    def test_skip_missing_file(mock_prepare, mock_run, tmp_path):
        """不存在的文件被跳过，不出现在 signed_files 中。"""
        f1 = tmp_path / "a.bin"
        f1.touch()
        f2 = tmp_path / "missing.bin"
        mock_run.return_value = CompletedProcess(args=[], returncode=0, stdout="")
        ret, signed = community_sign_build.run_sign([str(f1), str(f2)])
        assert ret is True
        assert str(f1) in signed
        assert str(f2) not in signed
        assert mock_run.call_count == 1

    @staticmethod
    @mock.patch("community_sign_build.subprocess.run")
    @mock.patch("community_sign_build.prepare_crl", return_value=CRL_PATH)
    def test_signatrust_fails(mock_prepare, mock_run, tmp_path):
        """signatrust 返回非零 → (False, partial)。"""
        f1 = tmp_path / "a.bin"
        f1.touch()
        f2 = tmp_path / "b.bin"
        f2.touch()
        mock_run.return_value = CompletedProcess(args=[], returncode=1, stdout="error")
        ret, signed = community_sign_build.run_sign([str(f1), str(f2)])
        assert ret is False
        assert len(signed) == 0

    @staticmethod
    @mock.patch("community_sign_build.subprocess.run")
    @mock.patch("community_sign_build.prepare_crl", return_value=CRL_PATH)
    def test_sign_timeout(mock_prepare, mock_run, tmp_path):
        """签名超时 → (False, [])，不抛异常。"""
        f1 = tmp_path / "a.bin"
        f1.touch()
        mock_run.side_effect = TimeoutExpired("signatrust", 600)
        ret, signed = community_sign_build.run_sign([str(f1)])
        assert ret is False
        assert signed == []

    @staticmethod
    @mock.patch("community_sign_build.subprocess.run")
    @mock.patch("community_sign_build.prepare_crl", return_value=CRL_PATH)
    def test_crl_prepared_once(mock_prepare, mock_run, tmp_path):
        """CRL 只准备一次，不论文件数量。"""
        files = []
        for name in ("a.bin", "b.bin", "c.bin"):
            f = tmp_path / name
            f.touch()
            files.append(str(f))
        mock_run.return_value = CompletedProcess(args=[], returncode=0, stdout="")
        community_sign_build.run_sign(files)
        assert mock_prepare.call_count == 1

    @staticmethod
    @mock.patch("community_sign_build.subprocess.run")
    @mock.patch("community_sign_build.prepare_crl", return_value=None)
    def test_crl_prepare_fails_no_signing(mock_prepare, mock_run, tmp_path):
        """CRL 准备失败 → (False, [])，不执行签名。"""
        f1 = tmp_path / "a.bin"
        f1.touch()
        ret, signed = community_sign_build.run_sign([str(f1)])
        assert ret is False
        assert signed == []
        mock_run.assert_not_called()

    @staticmethod
    @mock.patch("community_sign_build.subprocess.run")
    @mock.patch("community_sign_build.prepare_crl", return_value=CRL_PATH)
    def test_all_files_missing(mock_prepare, mock_run, tmp_path):
        """所有文件都不存在 → (False, [])。"""
        f1 = tmp_path / "ghost1.bin"
        f2 = tmp_path / "ghost2.bin"
        ret, signed = community_sign_build.run_sign([str(f1), str(f2)])
        assert ret is False
        assert signed == []
        mock_run.assert_not_called()

    @staticmethod
    @mock.patch("community_sign_build.subprocess.run")
    @mock.patch("community_sign_build.prepare_crl", return_value=CRL_PATH)
    def test_shell_false(mock_prepare, mock_run, tmp_path):
        """签名调用使用 shell=False。"""
        f1 = tmp_path / "a.bin"
        f1.touch()
        mock_run.return_value = CompletedProcess(args=[], returncode=0, stdout="")
        community_sign_build.run_sign([str(f1)])
        _, kwargs = mock_run.call_args
        assert kwargs.get("shell") is False

    @staticmethod
    @mock.patch("community_sign_build.subprocess.run")
    @mock.patch("community_sign_build.prepare_crl", return_value=CRL_PATH)
    def test_timeout_passed_to_subprocess(mock_prepare, mock_run, tmp_path):
        """签名调用传入 timeout 参数。"""
        f1 = tmp_path / "a.bin"
        f1.touch()
        mock_run.return_value = CompletedProcess(args=[], returncode=0, stdout="")
        community_sign_build.run_sign([str(f1)])
        _, kwargs = mock_run.call_args
        assert kwargs.get("timeout") == community_sign_build.SUBPROCESS_TIMEOUT

    @staticmethod
    @mock.patch("community_sign_build.subprocess.run")
    @mock.patch("community_sign_build.prepare_crl", return_value=CRL_PATH)
    def test_crl_path_passed_to_sign_cmd(mock_prepare, mock_run, tmp_path):
        """prepare_crl 返回的 crl_path 被传入签名命令。"""
        f1 = tmp_path / "a.bin"
        f1.touch()
        mock_run.return_value = CompletedProcess(args=[], returncode=0, stdout="")
        community_sign_build.run_sign([str(f1)])
        args, _ = mock_run.call_args
        assert CRL_PATH in args[0]


# ===================== main =====================


class TestMain:
    """入口函数（返回布尔值，不调用 sys.exit）。"""

    @staticmethod
    @mock.patch("community_sign_build.check_result", return_value=True)
    @mock.patch("community_sign_build.run_sign")
    def test_success(mock_run_sign, mock_check):
        """签名成功且校验通过 → 返回 True。"""
        mock_run_sign.return_value = (True, ["/tmp/a.bin"])
        assert community_sign_build.main(["script.py", "/tmp/a.bin"]) is True

    @staticmethod
    @mock.patch("community_sign_build.check_result", return_value=False)
    @mock.patch("community_sign_build.run_sign")
    def test_check_fail(mock_run_sign, mock_check):
        """签名成功但校验失败 → 返回 False。"""
        mock_run_sign.return_value = (True, ["/tmp/a.bin"])
        assert community_sign_build.main(["script.py", "/tmp/a.bin"]) is False

    @staticmethod
    @mock.patch("community_sign_build.run_sign")
    def test_sign_fail(mock_run_sign):
        """签名失败 → 返回 False。"""
        mock_run_sign.return_value = (False, [])
        assert community_sign_build.main(["script.py", "/tmp/a.bin"]) is False

    @staticmethod
    def test_too_few_args():
        """无文件参数 → 返回 False。"""
        assert community_sign_build.main(["script.py"]) is False

    @staticmethod
    def test_help_returns_true():
        """--help 返回 True（SystemExit(0) 被捕获转为布尔值）。"""
        assert community_sign_build.main(["script.py", "--help"]) is True

    @staticmethod
    def test_help_not_first_position():
        """--help 不在首位时也返回 True。"""
        assert (
            community_sign_build.main(["script.py", "--crl-dir", "/tmp", "--help"])
            is True
        )

    @staticmethod
    def test_unknown_flag_returns_false():
        """未知参数返回 False（SystemExit(2) 被捕获转为布尔值）。"""
        assert (
            community_sign_build.main(["script.py", "--unknown", "/tmp/a.bin"]) is False
        )

    @staticmethod
    def test_crl_dir_missing_value_returns_false():
        """--crl-dir 缺值返回 False（SystemExit(2) 被捕获转为布尔值）。"""
        assert community_sign_build.main(["script.py", "--crl-dir"]) is False

    @staticmethod
    @mock.patch("community_sign_build.check_result", return_value=True)
    @mock.patch("community_sign_build.run_sign")
    def test_check_result_receives_signed_files_only(mock_run_sign, mock_check):
        """check_result 仅接收实际已签名的文件（不含被跳过的）。

        场景：传入 a.bin（存在）和 missing.bin（不存在），
        run_sign 跳过 missing.bin 只签名 a.bin，返回 (True, ["/tmp/a.bin"])，
        check_result 只接收 ["/tmp/a.bin"]。
        """
        mock_run_sign.return_value = (True, ["/tmp/a.bin"])
        assert (
            community_sign_build.main(["script.py", "/tmp/a.bin", "/tmp/missing.bin"])
            is True
        )
        mock_check.assert_called_once_with(["/tmp/a.bin"])

    @staticmethod
    @mock.patch("community_sign_build.check_result", return_value=True)
    @mock.patch("community_sign_build.run_sign")
    def test_multiple_files_checked(mock_run_sign, mock_check):
        """多文件场景：check_result 接收全部已签名文件。"""
        signed = ["/tmp/a.bin", "/tmp/b.bin"]
        mock_run_sign.return_value = (True, signed)
        assert (
            community_sign_build.main(["script.py", "/tmp/a.bin", "/tmp/b.bin"]) is True
        )
        mock_check.assert_called_once_with(signed)

    @staticmethod
    @mock.patch("community_sign_build.check_result")
    @mock.patch("community_sign_build.run_sign")
    def test_empty_signed_list_fail(mock_run_sign, mock_check):
        """所有文件被跳过 → run_sign 返回 (False, []) → 返回 False，不调用 check_result。"""
        mock_run_sign.return_value = (False, [])
        assert community_sign_build.main(["script.py", "/tmp/ghost.bin"]) is False
        mock_check.assert_not_called()


# ===================== --crl-dir 参数与并发隔离 =====================


class TestCrlDirArg:
    """--crl-dir 参数与 CRL 输出目录隔离（方案 C）。"""

    @staticmethod
    def test_parser_accepts_crl_dir():
        """define_parser 接受 --crl-dir 参数。"""
        parser = community_sign_build.define_parser()
        args = parser.parse_args(["--crl-dir", "/tmp/sign_dir", "/tmp/a.bin"])
        assert args.crl_dir == "/tmp/sign_dir"
        assert args.files == ["/tmp/a.bin"]

    @staticmethod
    def test_parser_crl_dir_default_empty():
        """--crl-dir 默认为空字符串（回退到脚本目录）。"""
        parser = community_sign_build.define_parser()
        args = parser.parse_args(["/tmp/a.bin"])
        assert args.crl_dir == ""

    @staticmethod
    @mock.patch("community_sign_build.check_result", return_value=True)
    @mock.patch("community_sign_build.run_sign")
    def test_main_passes_crl_dir_to_run_sign(mock_run_sign, mock_check):
        """main 将 --crl-dir 透传给 run_sign。"""
        mock_run_sign.return_value = (True, ["/tmp/a.bin"])
        community_sign_build.main(
            ["script.py", "--crl-dir", "/tmp/sign_dir", "/tmp/a.bin"]
        )
        mock_run_sign.assert_called_once()
        call_args = mock_run_sign.call_args
        assert call_args.args[1] == "/tmp/sign_dir"

    @staticmethod
    @mock.patch("community_sign_build.check_result", return_value=True)
    @mock.patch("community_sign_build.run_sign")
    def test_main_no_crl_dir_passes_empty(mock_run_sign, mock_check):
        """未指定 --crl-dir 时传空字符串给 run_sign（回退到脚本目录）。"""
        mock_run_sign.return_value = (True, ["/tmp/a.bin"])
        community_sign_build.main(["script.py", "/tmp/a.bin"])
        call_args = mock_run_sign.call_args
        assert call_args.args[1] == ""

    @staticmethod
    @mock.patch("community_sign_build.check_result", return_value=True)
    @mock.patch("community_sign_build.run_sign")
    def test_main_crl_dir_interleaved_with_files(mock_run_sign, mock_check):
        """T4: --crl-dir 在位置参数之后时也能正确解析（argparse 不支持交错，测前后两种顺序）。"""
        mock_run_sign.return_value = (True, ["/tmp/a.bin", "/tmp/b.bin"])
        # --crl-dir 在位置参数之后
        community_sign_build.main(
            ["script.py", "/tmp/a.bin", "/tmp/b.bin", "--crl-dir", "/tmp/sign_dir"]
        )
        call_args = mock_run_sign.call_args
        assert call_args.args[0] == ["/tmp/a.bin", "/tmp/b.bin"]
        assert call_args.args[1] == "/tmp/sign_dir"


class TestPrepareCrlOutputDir:
    """prepare_crl 的 crl_output_dir 参数隔离行为。"""

    @staticmethod
    @mock.patch("community_sign_build.subprocess.run")
    def test_crl_downloaded_to_crl_output_dir(mock_run, tmp_path):
        """指定 crl_output_dir 时 CRL 下载到该目录而非 SCRIPT_DIR。"""
        mock_run.return_value = CompletedProcess(args=[], returncode=0)
        out_dir = tmp_path / "sign_dir"
        out_dir.mkdir()
        expected_crl = os.path.join(str(out_dir), "SWSCRL.crl")
        # T3: isfile 仅对下载的 crl_path 返回 True，避免全局 patch 掩盖 CRL_FILE_PATH 分支问题

        with mock_crl_at_path(expected_crl):
            result = community_sign_build.prepare_crl(str(out_dir))
        assert result is not None
        assert result.startswith(str(out_dir))
        assert result.endswith("SWSCRL.crl")

    @staticmethod
    @mock.patch("community_sign_build.subprocess.run")
    def test_crl_download_cmd_uses_output_dir(mock_run, tmp_path):
        """下载命令的 -o 路径指向 crl_output_dir。"""
        mock_run.return_value = CompletedProcess(args=[], returncode=0)
        out_dir = tmp_path / "sign_dir"
        out_dir.mkdir()
        expected_crl = os.path.join(str(out_dir), "SWSCRL.crl")

        with mock_crl_at_path(expected_crl):
            community_sign_build.prepare_crl(str(out_dir))
        args, _ = mock_run.call_args
        cmd = args[0]
        out_idx = cmd.index("-o")
        assert cmd[out_idx + 1] == os.path.join(str(out_dir), "SWSCRL.crl")

    @staticmethod
    @mock.patch("community_sign_build.subprocess.run")
    def test_crl_fallback_to_script_dir_when_no_arg(mock_run, tmp_path):
        """未传 crl_output_dir 时回退到 SCRIPT_DIR（保持兼容）。"""
        mock_run.return_value = CompletedProcess(args=[], returncode=0)
        expected_crl = os.path.join(
            community_sign_build.SCRIPT_DIR, community_sign_build.CRL_FILENAME
        )

        with mock_crl_at_path(expected_crl):
            result = community_sign_build.prepare_crl()
        assert result is not None
        assert result == expected_crl

    @staticmethod
    @mock.patch("community_sign_build.subprocess.run")
    def test_crl_env_overrides_output_dir(mock_run, tmp_path, monkeypatch):
        """CRL_FILE_PATH 已设且文件存在时优先复用，忽略 crl_output_dir。"""
        real_crl = tmp_path / "real.crl"
        real_crl.touch()
        out_dir = tmp_path / "sign_dir"
        out_dir.mkdir()
        monkeypatch.setenv("CRL_FILE_PATH", str(real_crl))
        result = community_sign_build.prepare_crl(str(out_dir))
        assert result == str(real_crl)
        mock_run.assert_not_called()

    @pytest.fixture(autouse=True)
    def mock_sleep(self):
        with mock.patch("community_sign_build.time.sleep"):
            yield

    @pytest.fixture(autouse=True)
    def clear_crl_env(self, monkeypatch):
        """默认清除 CRL_FILE_PATH 环境变量，测试需设置时用 monkeypatch.setenv。"""
        monkeypatch.delenv("CRL_FILE_PATH", raising=False)


class TestRunSignCrlDir:
    """run_sign 透传 crl_output_dir 给 prepare_crl。"""

    @staticmethod
    @mock.patch("community_sign_build.prepare_crl")
    @mock.patch("community_sign_build.subprocess.run")
    def test_run_sign_passes_crl_dir_to_prepare_crl(mock_run, mock_prepare, tmp_path):
        """run_sign 将 crl_output_dir 传给 prepare_crl。"""
        mock_prepare.return_value = CRL_PATH
        f = tmp_path / "a.bin"
        f.touch()
        mock_run.return_value = CompletedProcess(args=[], returncode=0, stdout="")
        community_sign_build.run_sign([str(f)], "/tmp/sign_dir")
        mock_prepare.assert_called_once_with("/tmp/sign_dir")

    @staticmethod
    @mock.patch("community_sign_build.prepare_crl")
    @mock.patch("community_sign_build.subprocess.run")
    def test_run_sign_default_crl_dir_empty(mock_run, mock_prepare, tmp_path):
        """run_sign 未传 crl_output_dir 时传空字符串。"""
        mock_prepare.return_value = CRL_PATH
        f = tmp_path / "a.bin"
        f.touch()
        mock_run.return_value = CompletedProcess(args=[], returncode=0, stdout="")
        community_sign_build.run_sign([str(f)])
        mock_prepare.assert_called_once_with("")
