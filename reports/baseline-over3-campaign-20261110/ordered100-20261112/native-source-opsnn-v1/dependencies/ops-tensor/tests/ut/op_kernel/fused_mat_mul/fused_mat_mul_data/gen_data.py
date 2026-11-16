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
import numpy as np


def _fp32_to_bf16_u16(arr_fp32):
    # numpy无原生bfloat16，使用round-to-nearest-even转换后以uint16保存。
    u32 = arr_fp32.astype(np.float32).view(np.uint32).astype(np.uint64)
    rounding_bias = 0x7FFF + ((u32 >> 16) & 1)
    return ((u32 + rounding_bias) >> 16).astype(np.uint16)


def _bf16_u16_to_fp32(arr_bf16):
    return (arr_bf16.astype(np.uint32) << 16).view(np.float32)


def _cast_input(arr_fp32, dtype):
    if dtype == "float16":
        storage = arr_fp32.astype(np.float16)
        return storage, storage.astype(np.float32)
    storage = _fp32_to_bf16_u16(arr_fp32)
    return storage, _bf16_u16_to_fp32(storage)


def gen_fused_mat_mul_data(
    m, n, k, batch, dtype, alpha, beta, output_dir="./", seed=42
):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    a, a_fp32 = _cast_input(
        rng.uniform(-1.0, 1.0, (batch, m, k)).astype(np.float32), dtype
    )
    b, b_fp32 = _cast_input(
        rng.uniform(-1.0, 1.0, (batch, k, n)).astype(np.float32), dtype
    )
    x3, x3_fp32 = _cast_input(
        rng.uniform(-1.0, 1.0, (batch, m, n)).astype(np.float32), dtype
    )

    golden_fp32 = alpha * np.matmul(a_fp32, b_fp32) + beta * x3_fp32
    golden = (
        golden_fp32.astype(np.float16)
        if dtype == "float16"
        else _fp32_to_bf16_u16(golden_fp32)
    )

    a.tofile(os.path.join(output_dir, "input_a.bin"))
    b.tofile(os.path.join(output_dir, "input_b.bin"))
    x3.tofile(os.path.join(output_dir, "input_x3.bin"))
    golden.tofile(os.path.join(output_dir, "golden_c.bin"))

    return a, b, x3, golden


def main():
    parser = argparse.ArgumentParser(
        description="Generate FusedMatMul scale-add test data"
    )
    parser.add_argument("--m", type=int, required=True, help="M dimension")
    parser.add_argument("--n", type=int, required=True, help="N dimension")
    parser.add_argument("--k", type=int, required=True, help="K dimension")
    parser.add_argument("--batch", type=int, default=1, help="Batch dimension")
    parser.add_argument(
        "--dtype", type=str, default="float16", choices=["float16", "bfloat16"]
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="Matmul result scale")
    parser.add_argument("--beta", type=float, default=1.0, help="x3 scale")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gen_fused_mat_mul_data(
        args.m,
        args.n,
        args.k,
        args.batch,
        args.dtype,
        args.alpha,
        args.beta,
        args.output_dir,
        args.seed,
    )


if __name__ == "__main__":
    main()
