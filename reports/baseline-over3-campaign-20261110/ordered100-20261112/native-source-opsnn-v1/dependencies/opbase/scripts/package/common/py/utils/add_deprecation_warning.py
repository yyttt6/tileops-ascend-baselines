#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import sys


def _is_skip_line(stripped):
    if not stripped:
        return True
    comment_prefixes = ('//', '*', '/*!')
    return stripped.startswith(comment_prefixes)


def add_deprecation_warning(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    # Find the insert position: after the copyright comment block, before #ifndef
    insert_idx = 0
    in_comment = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not in_comment and stripped.startswith('/*'):
            in_comment = True
        if in_comment:
            if stripped.endswith('*/'):
                in_comment = False
            continue
        if _is_skip_line(stripped):
            continue
        insert_idx = i
        break
    lines.insert(
        insert_idx,
        '#warning "This file is scheduled to be deprecated in June 2027, '
        'and will be replaced by the file with the same name under include/aclnnop '
        'in the CANN package installation path. '
        'We apologize for any inconvenience caused '
        'and appreciate your timely migration to the new interface. "\n'
    )
    with open(filepath, 'w') as f:
        f.writelines(lines)


if __name__ == '__main__':
    for filepath in sys.argv[1:]:
        if filepath:
            add_deprecation_warning(filepath)
