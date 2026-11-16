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
tbe register
"""
from .register_api import register_operator
from .register_api import get_operator
from .register_api import get_op_compute
from .register_api import register_param_generalization
from .register_api import get_param_generalization
from .register_api import get_fusion_buildcfg
from .register_api import set_fusion_buildcfg


from .class_manager import InvokeStage
from .class_manager import Priority
from .class_manager import OpCompute
from .class_manager import Operator
