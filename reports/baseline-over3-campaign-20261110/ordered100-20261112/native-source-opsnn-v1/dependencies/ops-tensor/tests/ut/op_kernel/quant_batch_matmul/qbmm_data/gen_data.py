# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

#!/usr/bin/env python3

import argparse
import os
from dataclasses import dataclass
import numpy as np


@dataclass
class QbmmGenConfig:
    output_dir: str = "./"
    x1_mode: str = "default"  # 激活(x1)量化模式：default / pertoken / pertensor
    x2_mode: str = "pertensor"  # 权重(x2)量化模式：perchannel / pertensor
    bias: bool = False  # 是否生成 bias.bin 并计入 golden
    bias_dtype: str = "float16"  # bias 张量 dtype：float16 / float32 / bfloat16 / int32
    scale_dtype: str = "uint64"  # x2 scale(scale.bin) 编码：uint64 / float32
    out_dtype: str = "float16"  # golden 输出 dtype：float16 / float32


def _fp32_to_bf16_u16(arr_fp32):
    # numpy 无原生 bfloat16：取 float32 高 16 位（round-to-nearest-even）存为 uint16。
    u32 = arr_fp32.astype(np.float32).view(np.uint32).astype(np.uint64)
    rounding_bias = 0x7FFF + ((u32 >> 16) & 1)
    u16 = ((u32 + rounding_bias) >> 16).astype(np.uint16)
    return u16


def _write_bias(bias_fp32, bias_dtype, path):
    if bias_dtype == "float32":
        bias_fp32.astype(np.float32).tofile(path)
    elif bias_dtype == "float16":
        bias_fp32.astype(np.float16).tofile(path)
    elif bias_dtype == "bfloat16":
        _fp32_to_bf16_u16(bias_fp32).tofile(path)
    elif bias_dtype == "int32":
        bias_fp32.astype(np.int32).tofile(path)
    else:
        raise ValueError("unsupported bias_dtype: " + bias_dtype)


def gen_qbmm_data(m, n, k, cfg=None):
    if cfg is None:
        cfg = QbmmGenConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)

    # A (M, K) / B (K, N) int8 输入
    a = np.random.randint(-128, 127, size=(m, k), dtype=np.int8)
    b = np.random.randint(-128, 127, size=(k, n), dtype=np.int8)

    # x1 激活 scale：per-token=M 向量 / per-tensor=标量 / default=占位标量（epilogue 忽略）。取 1.0 便于对照。
    if cfg.x1_mode == "pertoken":
        x1_scale = np.ones(m, dtype=np.float32)
    else:
        x1_scale = np.array([1.0], dtype=np.float32)

    # x2 权重 scale：per-channel=N 向量 / per-tensor=标量。
    if cfg.x2_mode == "perchannel":
        x2_scale_fp32 = np.ones(n, dtype=np.float32)
    else:
        x2_scale_fp32 = np.array([1.0], dtype=np.float32)

    # fixpipe 路径：scale 以 uint64 编码 1.0f（0x3F80000000000000）；MIX 路径：float32。
    if cfg.scale_dtype == "uint64":
        scale_count = n if cfg.x2_mode == "perchannel" else 1
        scale_out = np.full(scale_count, 0x3F80000000000000, dtype=np.uint64)
    else:
        scale_out = x2_scale_fp32.astype(np.float32)

    # bias：N 向量（此处取 0，smoke 用；值不参与 KERNEL_RUN_KF 崩溃判定）。
    bias_fp32 = np.zeros(n, dtype=np.float32)

    # golden（当前 UT 与 PR #61 一致，仅 smoke 测试，不读回比对；保留以便后续扩展）。
    a_fp32 = a.astype(np.float32)
    b_fp32 = b.astype(np.float32)
    c_fp32 = np.matmul(a_fp32, b_fp32)
    c_fp32 = (
        c_fp32 * x1_scale.reshape(-1, 1)
        if cfg.x1_mode == "pertoken"
        else c_fp32 * x1_scale[0]
    )
    c_fp32 = (
        c_fp32 * x2_scale_fp32.reshape(1, -1)
        if cfg.x2_mode == "perchannel"
        else c_fp32 * x2_scale_fp32[0]
    )
    if cfg.bias:
        c_fp32 = c_fp32 + bias_fp32.reshape(1, -1)
    with np.errstate(over="ignore"):
        golden = c_fp32.astype(np.float16 if cfg.out_dtype == "float16" else np.float32)

    a.tofile(os.path.join(cfg.output_dir, "input_a.bin"))
    b.tofile(os.path.join(cfg.output_dir, "input_b.bin"))
    x1_scale.tofile(os.path.join(cfg.output_dir, "pertoken_scale.bin"))
    scale_out.tofile(os.path.join(cfg.output_dir, "scale.bin"))
    _write_bias(bias_fp32, cfg.bias_dtype, os.path.join(cfg.output_dir, "bias.bin"))
    golden.tofile(os.path.join(cfg.output_dir, "golden_c.bin"))

    return a, b, x1_scale, scale_out, golden


def main():
    parser = argparse.ArgumentParser(description="Generate QBMM A8W8 test data")
    parser.add_argument("--m", type=int, required=True, help="M dimension")
    parser.add_argument("--n", type=int, required=True, help="N dimension")
    parser.add_argument("--k", type=int, required=True, help="K dimension")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument(
        "--x1_mode",
        type=str,
        default="default",
        choices=["default", "pertoken", "pertensor"],
        help="activation(x1) quant mode",
    )
    parser.add_argument(
        "--x2_mode",
        type=str,
        default="pertensor",
        choices=["perchannel", "pertensor"],
        help="weight(x2) quant mode",
    )
    parser.add_argument(
        "--bias",
        action="store_true",
        help="generate bias.bin and include bias in golden",
    )
    parser.add_argument(
        "--bias_dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16", "int32"],
        help="bias tensor dtype",
    )
    parser.add_argument(
        "--scale_dtype",
        type=str,
        default="uint64",
        choices=["uint64", "float32"],
        help="x2 scale(scale.bin) encoding",
    )
    parser.add_argument(
        "--out_dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="golden output dtype",
    )

    args = parser.parse_args()

    cfg = QbmmGenConfig(
        output_dir=args.output_dir,
        x1_mode=args.x1_mode,
        x2_mode=args.x2_mode,
        bias=args.bias,
        bias_dtype=args.bias_dtype,
        scale_dtype=args.scale_dtype,
        out_dtype=args.out_dtype,
    )
    gen_qbmm_data(args.m, args.n, args.k, cfg)


if __name__ == "__main__":
    main()
