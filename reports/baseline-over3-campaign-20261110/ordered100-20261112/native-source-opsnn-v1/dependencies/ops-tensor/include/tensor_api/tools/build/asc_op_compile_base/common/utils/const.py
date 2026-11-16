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
utils const
"""
from enum import Enum


WEIGHT_SPARSE_4_2 = "weight_sparse_4_2"


class SplitAxisMode(Enum):
    split_hw = 0
    split_w = 1


class ComputeFlow(Enum):
    tacit = 0 # means no special process for inputs, contrains Nz input and ND input except milan
    on_the_fly = 1 # trans nd2nz by gm2l1
    mix_l2 = 2 # trans nd2nz by mixL2
    weight_nz = 3 # fmap nd, weight nz
