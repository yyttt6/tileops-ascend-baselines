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
api_check_support_func = {}
TIK_API_CHECK_SUPPORT_FUNC_TYPE = "TIK"
DSL_API_CHECK_SUPPORT_FUNC_TYPE = "DSL"

# product version
# This is used for DSL/AutoSchedule ONLY!
# For other components, use te.platform.get_soc_spec("SHORT_SOC_VERSION")!
VERSION_MINI = "1910"
