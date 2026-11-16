#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# **************************************************************
# 签名脚本：调用 signatrust_client 对指定文件制作 CMS (p7s) detached 签名
# 流程：准备 CRL（本地优先，否则下载）→ 逐文件签名 → 校验 p7s 生成
#
# 调用方式：
#   python3 community_sign_build.py <file1> [file2 ...]
#   argv[1:] = 待签名文件列表
# **************************************************************

import os
import sys
import time
import logging
import argparse
import subprocess
from typing import List, Optional, Tuple

# --- 常量（支持环境变量覆盖，便于不同环境部署）---
CRL_FILENAME = "SWSCRL.crl"
CMS_TAG = ".p7s"
# 证书类型：0x1 社区证书，与 image_pack.py 的 -certtype 语义一致
CERT_TYPE = "0x1"
SIGN_CLIENT_PATH = os.environ.get(
    "SIGN_CLIENT_PATH", "/home/jenkins/signatrust_client/signatrust_client"
)
SIGN_CLIENT_CONFIG = os.environ.get(
    "SIGN_CLIENT_CONFIG", "/home/jenkins/signatrust_client/client.toml"
)
CRL_DOWNLOAD_URL = os.environ.get(
    "CRL_DOWNLOAD_URL",
    "https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/cff6dec5-acc3-415b-9957-5d4effd59669.crl",
)
# 用户可通过环境变量指定本地已有 CRL 文件，避免每次下载
# 注意：CRL_FILE_PATH 在 prepare_crl 内运行时读取（与 add_header_sign.py 一致），不设模块级常量
SUBPROCESS_TIMEOUT = 60  # 秒
CRL_DOWNLOAD_RETRIES = 3  # CRL 下载最大重试次数
CRL_DOWNLOAD_BACKOFF = 2  # CRL 下载重试退避基数（秒），实际退避 = BACKOFF * attempt

# 脚本所在目录，CRL 文件与签名工作目录均基于此路径
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)


def setup_logging():
    """初始化日志配置，仅在脚本直接运行时调用。"""
    logging.basicConfig(
        level=os.environ.get("SIGN_LOG_LEVEL", "INFO").upper(),
        format="[CannSign] %(asctime)s line:%(lineno)d %(levelname)s:%(name)s:%(message)s",
        datefmt="%H:%M:%S",
    )


def _validate_downloaded_crl(crl_path: str) -> bool:
    """校验下载的 CRL 文件存在且非空，防止 CDN 异常返回空内容。"""
    try:
        if not os.path.isfile(crl_path) or os.path.getsize(crl_path) == 0:
            return False
        return True
    except OSError as e:
        logger.warning("crl file check failed: %s", e)
        return False


def _sleep_backoff(attempt: int) -> None:
    """重试退避，仅在非最后一次重试时 sleep。"""
    if attempt < CRL_DOWNLOAD_RETRIES:
        time.sleep(CRL_DOWNLOAD_BACKOFF * attempt)


def prepare_crl(crl_output_dir: str = "") -> Optional[str]:
    """准备 CRL 文件，返回实际使用的 CRL 路径，失败返回 None。

    优先使用用户通过 CRL_FILE_PATH 环境变量配置的本地文件（CI 环境预置，避免下载）；
    若未配置或文件不存在，则从远程下载到 crl_output_dir 指定目录（含重试），
    未指定目录时回退到脚本目录。每目标独立目录可避免并行签名时的下载竞争。
    """
    # 运行时读取环境变量，与 add_header_sign.py 的读取时机保持一致
    crl_file_path = os.environ.get("CRL_FILE_PATH", "")
    # 优先使用用户配置的本地 CRL 文件，避免远程下载
    if crl_file_path and os.path.isfile(crl_file_path):
        logger.info("use user-specified crl: %s", crl_file_path)
        return crl_file_path

    # 未配置或文件不存在，下载 CRL 到指定目录（每目标独立），未指定时回退到脚本目录
    output_dir = crl_output_dir if crl_output_dir else SCRIPT_DIR
    # 确保输出目录存在，避免 curl 因目录不存在而重试 3 次后才报错
    os.makedirs(output_dir, exist_ok=True)
    crl_path = os.path.join(output_dir, CRL_FILENAME)
    cmd = ["curl", "-sSL", CRL_DOWNLOAD_URL, "-o", crl_path]
    logger.info("execute:%s", " ".join(cmd))

    # check=False：curl 失败时通过 returncode 判断，统一在循环中重试
    for attempt in range(1, CRL_DOWNLOAD_RETRIES + 1):
        if _download_crl_once(cmd, crl_path, attempt):
            return crl_path
        _sleep_backoff(attempt)

    logger.error("download crl failed after %d attempts", CRL_DOWNLOAD_RETRIES)
    # 清理可能残留的损坏或空 CRL 文件
    if os.path.isfile(crl_path):
        try:
            os.remove(crl_path)
        except OSError:
            pass
    return None


def _download_crl_once(cmd: List[str], crl_path: str, attempt: int) -> bool:
    """执行一次 CRL 下载并校验，成功返回 True。"""
    try:
        result = subprocess.run(
            cmd,
            shell=False,
            check=False,
            timeout=SUBPROCESS_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            "download crl timeout, attempt %d/%d", attempt, CRL_DOWNLOAD_RETRIES
        )
        return False

    if result.returncode != 0:
        logger.warning(
            "download crl failed (returncode=%d), attempt %d/%d",
            result.returncode,
            attempt,
            CRL_DOWNLOAD_RETRIES,
        )
        return False

    if not _validate_downloaded_crl(crl_path):
        logger.warning(
            "downloaded crl is empty or missing, attempt %d/%d",
            attempt,
            CRL_DOWNLOAD_RETRIES,
        )
        return False

    return True


def get_sign_cmd(input_file: str, crl_path: str) -> List[str]:
    """构建 signatrust_client 签名命令（列表格式，配合 shell=False 消除注入风险）。

    命令语义：使用 x509 证书 SignCert 对 input_file 制作 detached p7s 签名，
    并附带 TimeCert 时间戳与本地 CRL 吊销列表。
    """
    return [
        SIGN_CLIENT_PATH,
        "--config",
        SIGN_CLIENT_CONFIG,
        "add",
        "--file-type",
        "p7s",
        "--key-type",
        "x509",
        "--key-name",
        "SignCert",
        "--detached",
        input_file,
        "--timestamp-key",
        "TimeCert",
        "--crl",
        crl_path,
    ]


def sign_single_file(input_file: str, crl_path: str) -> bool:
    """对单个文件执行签名，成功返回 True，失败或超时返回 False。"""
    cmd = get_sign_cmd(input_file, crl_path)
    logger.info("execute:%s", " ".join(cmd))

    # check=False：signatrust 失败时通过 returncode 判断，不抛异常
    # cwd=SCRIPT_DIR：signatrust_client 的 --config 参数若为相对路径，从此目录解析；
    #                  p7s 始终生成在 input_file 同目录，不受 cwd 影响
    try:
        result = subprocess.run(
            cmd,
            cwd=SCRIPT_DIR,
            shell=False,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=SUBPROCESS_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        logger.error("sign timeout: %s", input_file)
        return False

    # 签名失败，打印 signatrust 输出便于定位
    if result.returncode != 0:
        logger.error("sign output: %s", result.stdout)
        logger.error("file %s signed failed", input_file)
        return False

    return True


def check_result(files: List[str]) -> bool:
    """校验每个输入文件对应的 .p7s 签名文件是否已生成且非空。

    p7s 文件由 signatrust 生成在 input_file 同目录，命名为 input_file + .p7s。
    """
    # 空列表守卫：无文件被签名说明输入全部不存在或路径错误，不应视为成功
    if not files:
        logger.error("no files were signed")
        return False
    for input_file in files:
        p7s_path = input_file + CMS_TAG
        # isfile 与 getsize 之间存在 TOCTOU 间隙，用 try/except 防止文件被删除时抛异常
        try:
            if not os.path.isfile(p7s_path):
                logger.error("cms file: %s does not exist", p7s_path)
                return False
            # 校验 p7s 文件非空，防止残留空文件误判成功
            if os.path.getsize(p7s_path) == 0:
                logger.error("cms file: %s is empty", p7s_path)
                return False
        except OSError as e:
            logger.error("cms file check failed: %s, %s", p7s_path, e)
            return False
    return True


def run_sign(
    input_files: List[str], crl_output_dir: str = ""
) -> Tuple[bool, List[str]]:
    """对文件列表执行签名。

    crl_output_dir 传入每目标独立的 CRL 输出目录，避免并行签名时共享下载路径。

    Returns:
        (是否全部成功, 实际已签名的文件列表)
    """
    signed_files: List[str] = []

    # 准备 CRL 文件（整个签名流程只准备一次）
    crl_path = prepare_crl(crl_output_dir)
    if crl_path is None:
        return False, signed_files

    # 逐文件签名，任一文件失败则立即终止
    for input_file in input_files:
        # 跳过不存在的文件，不影响其他文件签名
        if not os.path.isfile(input_file):
            logger.warning("input file: %s does not exist, skip", input_file)
            continue

        # 转为绝对路径，确保 p7s 生成在 input_file 同目录而非 cwd
        input_file = os.path.abspath(input_file)

        if not sign_single_file(input_file, crl_path):
            return False, signed_files

        signed_files.append(input_file)

    # 无文件被签名（全部被跳过），视为失败
    if not signed_files:
        logger.error("no files were signed")
        return False, signed_files

    return True, signed_files


def define_parser():
    """定义命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="调用 signatrust_client 对指定文件制作 CMS (p7s) detached 签名"
    )
    # CRL 输出目录，由 add_header_sign.py 传入每目标独立的 sign_file_dir，实现并行签名隔离
    parser.add_argument(
        "--crl-dir", default="", help="CRL 输出目录，未指定时下载到脚本目录"
    )
    # 查询 flag：打印后立即退出，不执行签名；编排器据此获取产物扩展名与证书类型
    parser.add_argument(
        "--print-sign-ext", action="store_true", help="打印签名产物扩展名并退出"
    )
    parser.add_argument(
        "--print-certtype", action="store_true", help="打印证书类型并退出"
    )
    # files 改为 nargs='*'，允许查询模式不带文件参数；签名模式在 main() 显式校验非空
    parser.add_argument("files", nargs="*", help="待签名文件列表")
    return parser


def main(argv=None) -> bool:
    """主流程，返回布尔值（True 成功，False 失败）。

    argv[1:] 为待签名文件列表，可选 --crl-dir 指定 CRL 输出目录。
    调用方式：python3 community_sign_build.py [--crl-dir <dir>] <file1> [file2 ...]
    查询方式：python3 community_sign_build.py --print-sign-ext|--print-certtype
    """
    if argv is None:
        argv = sys.argv
    try:
        args = define_parser().parse_args(argv[1:])
    except SystemExit as e:
        # argparse 对 --help 返回 code=0（成功），对非法参数返回 code=2（失败）
        return e.code == 0 or e.code is None

    # 查询模式：打印后立即返回，不执行签名、不下载 CRL、不调用 signatrust_client
    if args.print_sign_ext:
        print(CMS_TAG)
        return True
    if args.print_certtype:
        print(CERT_TYPE)
        return True

    # 签名模式：显式校验 files 非空（替代原 nargs='+' 的强制约束）
    input_files = args.files
    if not input_files:
        logger.error("no files to sign")
        return False

    # 执行签名
    success, signed_files = run_sign(input_files, args.crl_dir)
    if not success:
        logger.error("signature build fail")
        return False

    # 校验 p7s 文件是否生成（仅校验实际已签名的文件）
    if not check_result(signed_files):
        logger.error("check signature result fail")
        return False

    logger.info("signature build success, signed files: %s", signed_files)
    return True


if __name__ == "__main__":
    setup_logging()
    sys.exit(0 if main(sys.argv) else 1)
