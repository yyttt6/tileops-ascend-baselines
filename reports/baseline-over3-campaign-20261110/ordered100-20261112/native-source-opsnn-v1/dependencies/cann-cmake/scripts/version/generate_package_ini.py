#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""生成包的 .ini 文件。"""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone


def gen_package_ini_content(version: str) -> str:
    """生成 .ini 文件内容。"""
    lines = []
    lines.append(f'Version={version}')
    
    if os.environ.get('tagInfo'):
        tag_info = os.environ.get('tagInfo')
        timestamp = '_'.join(tag_info.split('_')[-3:-1])
    else:
        timestamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S%f')[:-3]
    lines.append(f'timestamp={timestamp}')
    lines.append('')
    
    return '\n'.join(lines)


def main():
    """主流程。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('version', help='Version number.')
    parser.add_argument('--output', required=True, help='Output file path.')
    args = parser.parse_args()

    content = gen_package_ini_content(args.version)
    with open(args.output, 'w', encoding='utf-8') as file:
        file.write(content)

    return True


if __name__ == '__main__':
    if not main():
        sys.exit(1)