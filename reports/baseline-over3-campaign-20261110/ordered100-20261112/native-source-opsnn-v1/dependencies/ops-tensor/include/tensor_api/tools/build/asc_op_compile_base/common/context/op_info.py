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
Op Info
"""
from typing import List
from typing import Optional


class OpInfo:
    """
    OpInfo
    """

    def __init__(self, op_name: str, op_type: str):
        """

        :param op_name:
        :param op_type:
        """
        self._op_name = op_name  # type: str
        self._op_type = op_type  # type: str
        self._pattern = None  # type: Optional[str]
        self._inputs = []  # type: List[dict]
        self._outputs = []  # type: List[dict]
        self._attrs = {}  # type: dict
        self._private_attrs = {} # type: dict
        self._kernel_name = None  # type: Optional[str]
        self._extra_params = {}  # type: dict
        self._precision_mode = None # type: Optional[str]

    @property
    def op_name(self):
        # type: () -> str
        """

        :return:
        """
        return self._op_name

    @property
    def op_type(self):
        # type: () -> str
        """

        :return:
        """
        return self._op_type

    @property
    def pattern(self):
        # type: () -> Optional[str]
        """

        :return:
        """
        return self._pattern

    @pattern.setter
    def pattern(self, pattern):
        # type: (str) -> None
        """

        :param pattern:
        :return:
        """
        self._pattern = pattern

    @property
    def inputs(self):
        # type: () -> List[dict]
        """

        :return:
        """
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        # type: (List[dict]) -> None
        """

        :param inputs:
        :return:
        """
        self._inputs = inputs

    @property
    def outputs(self):
        # type: () -> List[dict]
        """

        :return:
        """
        return self._outputs

    @outputs.setter
    def outputs(self, outputs):
        # type: (List[dict]) -> None
        """

        :param outputs:
        :return:
        """
        self._outputs = outputs

    @property
    def attrs(self):
        # type: () -> dict
        """

        :return:
        """
        return self._attrs

    @attrs.setter
    def attrs(self, attrs):
        # type: (dict) -> None
        """

        :param attrs:
        :return:
        """
        self._attrs = attrs

    @property
    def private_attrs(self):
        # type: () -> dict
        """

        :return:
        """
        return self._private_attrs

    @private_attrs.setter
    def private_attrs(self, private_attrs):
        # type: (dict) -> None
        """

        :param private_attrs:
        :return:
        """
        self._private_attrs = private_attrs

    @property
    def kernel_name(self):
        # type: () -> Optional[str]
        """

        :return:
        """
        return self._kernel_name

    @kernel_name.setter
    def kernel_name(self, kernel_name):
        # type: (str) -> None
        """

        :param kernel_name:
        :return:
        """
        self._kernel_name = kernel_name

    @property
    def extra_params(self):
        # type: () -> dict
        """

        :return:
        """
        return self._extra_params

    @extra_params.setter
    def extra_params(self, extra_params):
        # type: (dict) -> None
        """

        :param extra_params:
        :return:
        """
        self._extra_params = extra_params

    @property
    def precision_mode(self):
        # type: () -> Optional[str]
        """

        :return:
        """
        return self._precision_mode

    @precision_mode.setter
    def precision_mode(self, precision_mode):
        # type: (str) -> None
        """

        :param precision_mode:
        :return:
        """
        self._precision_mode = precision_mode
