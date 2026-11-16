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

"""Tests for funcbase utility functions."""

import pytest
from package.utils import funcbase


class TestDispatchExtended:
    """Extended tests for dispatch."""

    @staticmethod
    def test_dispatch_basic():
        """Test dispatch function."""
        func = funcbase.dispatch(
            lambda x: x * 2,
            lambda x: x + 1
        )
        # dispatch returns a generator
        result = list(func(5))
        # Should apply all functions
        assert 10 in result  # 5 * 2
        assert 6 in result   # 5 + 1


class TestPipeExtended:
    """Extended tests for pipe."""

    @staticmethod
    def test_pipe_basic():
        """Test pipe function."""
        func = funcbase.pipe(
            lambda x: x * 2,
            lambda x: x + 1
        )
        result = func(5)
        assert result == 11  # (5 * 2) + 1


class TestInvokeExtended:
    """Extended tests for invoke."""

    @staticmethod
    def test_invoke_basic():
        """Test invoke function."""
        result = funcbase.invoke(
            list,
            iter([1, 2, 3])
        )
        assert result == [1, 2, 3]
