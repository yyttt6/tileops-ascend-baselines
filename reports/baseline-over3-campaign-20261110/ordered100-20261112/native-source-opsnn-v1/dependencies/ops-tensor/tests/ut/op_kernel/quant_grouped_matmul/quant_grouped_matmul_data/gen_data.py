#!/usr/bin/env python3

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

import argparse
import os
from dataclasses import dataclass

import numpy as np


MX_SCALE_ONE = np.uint8(0x7F)
TYPE_INFO = {
    "mxfp8_e4m3": {
        "values": np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32),
        "codes": np.array([0x00, 0x30, 0xB0, 0x38, 0xB8], dtype=np.uint8),
        "c0": 32,
    },
    "mxfp8_e5m2": {
        "values": np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32),
        "codes": np.array([0x00, 0x38, 0xB8, 0x3C, 0xBC], dtype=np.uint8),
        "c0": 32,
    },
    "mxfp4_e2m1": {
        "values": np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32),
        "codes": np.array([0x0, 0x1, 0x9, 0x2, 0xA], dtype=np.uint8),
        "c0": 64,
    },
    "mxfp4_e1m2": {
        "values": np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32),
        "codes": np.array([0x0, 0x2, 0xA, 0x4, 0xC], dtype=np.uint8),
        "c0": 64,
    },
}


@dataclass(frozen=True)
class QgmmDataConfig:
    group_m: list
    n: int
    k: int
    dtype_a: str = "mxfp8_e4m3"
    dtype_b: str = "mxfp8_e4m3"
    weight_format: str = "nd"
    multi_tensor: bool = False
    with_bias: bool = False
    group_list_type: str = "length"
    output_dir: str = "./"
    seed: int = 42


@dataclass(frozen=True)
class QgmmGeneratedData:
    input_a: np.ndarray
    scale_a: np.ndarray
    bias: np.ndarray
    group_list: np.ndarray
    output: np.ndarray
    golden: np.ndarray


def _align_up(value, alignment):
    return ((value + alignment - 1) // alignment) * alignment


def _pack_fp4(nibbles):
    flat = np.asarray(nibbles, dtype=np.uint8).reshape(-1)
    if flat.size % 2:
        flat = np.pad(flat, (0, 1))
    return flat[0::2] | (flat[1::2] << 4)


def _encode(codes, dtype):
    return (
        _pack_fp4(codes)
        if dtype.startswith("mxfp4")
        else np.asarray(codes, dtype=np.uint8).reshape(-1)
    )


def _to_nz(codes, k, n, c0):
    k_aligned = _align_up(k, 16)
    n_aligned = _align_up(n, c0)
    padded = np.zeros((k_aligned, n_aligned), dtype=np.uint8)
    padded[:k, :n] = codes
    # NZ physical order: [N/C0, K/16, 16, C0].
    return (
        padded.reshape(k_aligned // 16, 16, n_aligned // c0, c0)
        .transpose(2, 0, 1, 3)
        .reshape(-1)
    )


def _to_zn(codes, k, n, c0):
    k_aligned = _align_up(k, c0)
    n_aligned = _align_up(n, 16)
    padded = np.zeros((n_aligned, k_aligned), dtype=np.uint8)
    padded[:n, :k] = codes.T
    # ZN physical order: [K/C0, N/16, 16, C0].
    return (
        padded.reshape(n_aligned // 16, 16, k_aligned // c0, c0)
        .transpose(2, 0, 1, 3)
        .reshape(-1)
    )


def _encode_weights(codes, config, info):
    encoded = []
    for group_codes in codes:
        if config.weight_format == "dn":
            group_codes = group_codes.T.reshape(-1)
        elif config.weight_format == "nz":
            group_codes = _to_nz(group_codes, config.k, config.n, info["c0"])
        elif config.weight_format == "zn":
            group_codes = _to_zn(group_codes, config.k, config.n, info["c0"])
        encoded.append(_encode(group_codes, config.dtype_b))
    return encoded


def _make_group_list(config):
    if config.group_list_type == "offset":
        return np.cumsum(config.group_m, dtype=np.int64)
    if config.group_list_type == "sparse":
        return np.column_stack(
            (
                np.arange(len(config.group_m), dtype=np.int64),
                np.asarray(config.group_m, dtype=np.int64),
            )
        ).reshape(-1)
    return np.asarray(config.group_m, dtype=np.int64)


def _make_golden(a_values, b_values, bias, group_m):
    parts = []
    m_offset = 0
    for group_idx, current_m in enumerate(group_m):
        group_slice = slice(m_offset, m_offset + current_m)
        parts.append(a_values[group_slice] @ b_values[group_idx] + bias[group_idx])
        m_offset += current_m
    return np.concatenate(parts, axis=0).astype(np.float32)


def _write_weight_data(config, encoded_b, scale_b):
    if config.multi_tensor:
        for group_idx, weight in enumerate(encoded_b):
            weight.tofile(os.path.join(config.output_dir, f"input_b_{group_idx}.bin"))
            scale_b[group_idx].tofile(
                os.path.join(config.output_dir, f"scale_b_{group_idx}.bin")
            )
        return
    np.concatenate(encoded_b).tofile(os.path.join(config.output_dir, "input_b.bin"))
    np.concatenate(scale_b).tofile(os.path.join(config.output_dir, "scale_b.bin"))


def _write_common_data(config, data):
    data.input_a.tofile(os.path.join(config.output_dir, "input_a.bin"))
    data.scale_a.tofile(os.path.join(config.output_dir, "scale_a.bin"))
    data.bias.tofile(os.path.join(config.output_dir, "bias.bin"))
    data.group_list.tofile(os.path.join(config.output_dir, "group_list.bin"))
    data.output.tofile(os.path.join(config.output_dir, "output_c.bin"))
    # CPU-debug KERNEL_RUN_KF is smoke-only; retain golden for future on-device validation.
    data.golden.tofile(os.path.join(config.output_dir, "golden_c.bin"))


def gen_qgmm_data(config):
    if not config.group_m or any(value <= 0 for value in config.group_m):
        raise ValueError("group_m must contain positive values")
    if config.n <= 0 or config.k <= 0:
        raise ValueError("n and k must be positive")

    os.makedirs(config.output_dir, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    info_a = TYPE_INFO[config.dtype_a]
    info_b = TYPE_INFO[config.dtype_b]
    group_num = len(config.group_m)
    total_m = sum(config.group_m)
    scale_k = ((config.k + 63) // 64) * 2

    a_indices = rng.integers(0, len(info_a["values"]), size=(total_m, config.k))
    b_indices = rng.integers(
        0, len(info_b["values"]), size=(group_num, config.k, config.n)
    )
    input_a = _encode(info_a["codes"][a_indices], config.dtype_a)
    encoded_b = _encode_weights(info_b["codes"][b_indices], config, info_b)

    scale_a = np.full((total_m, scale_k), MX_SCALE_ONE, dtype=np.uint8)
    scale_b = [
        np.full((config.n, scale_k), MX_SCALE_ONE, dtype=np.uint8)
        for _ in range(group_num)
    ]
    bias = (
        np.full((group_num, config.n), 0.25, dtype=np.float32)
        if config.with_bias
        else np.zeros((group_num, config.n), dtype=np.float32)
    )
    group_list = _make_group_list(config)
    output = np.zeros((total_m, config.n), dtype=np.float32)
    golden = _make_golden(
        info_a["values"][a_indices], info_b["values"][b_indices], bias, config.group_m
    )
    _write_weight_data(config, encoded_b, scale_b)
    _write_common_data(
        config, QgmmGeneratedData(input_a, scale_a, bias, group_list, output, golden)
    )


def main():
    parser = argparse.ArgumentParser(description="Generate QGMM MX kernel-UT data")
    parser.add_argument(
        "--group_m", type=int, nargs="+", required=True, help="M length of each group"
    )
    parser.add_argument("--n", type=int, required=True, help="N dimension")
    parser.add_argument("--k", type=int, required=True, help="K dimension")
    dtype_choices = tuple(TYPE_INFO)
    parser.add_argument("--dtype_a", choices=dtype_choices, default="mxfp8_e4m3")
    parser.add_argument("--dtype_b", choices=dtype_choices, default="mxfp8_e4m3")
    parser.add_argument(
        "--weight_format", choices=["nd", "dn", "nz", "zn"], default="nd"
    )
    parser.add_argument(
        "--multi_tensor", action="store_true", help="write one B/ScaleB file per group"
    )
    parser.add_argument("--with_bias", action="store_true")
    parser.add_argument(
        "--group_list_type", choices=["offset", "length", "sparse"], default="length"
    )
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = QgmmDataConfig(
        group_m=args.group_m,
        n=args.n,
        k=args.k,
        dtype_a=args.dtype_a,
        dtype_b=args.dtype_b,
        weight_format=args.weight_format,
        multi_tensor=args.multi_tensor,
        with_bias=args.with_bias,
        group_list_type=args.group_list_type,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    gen_qgmm_data(config)


if __name__ == "__main__":
    main()
