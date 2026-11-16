/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/format_utils.h"
#include <graph/utils/type_utils.h>
#include <map>
#include <string>

using std::map;
using std::string;

namespace op {

bool IsPrivateFormat(Format format)
{
    if (format == Format::FORMAT_NC1HWC0 || format == Format::FORMAT_FRACTAL_Z || format == Format::FORMAT_NDC1HWC0 ||
        format == Format::FORMAT_FRACTAL_Z_3D || format == Format::FORMAT_FRACTAL_NZ ||
        format == Format::FORMAT_NC1HWC0_C04 || format == Format::FORMAT_FRACTAL_NZ_C0_16 ||
        format == Format::FORMAT_FRACTAL_NZ_C0_32 || format == Format::FORMAT_FRACTAL_NZ_C0_2 ||
        format == Format::FORMAT_FRACTAL_NZ_C0_4 || format == Format::FORMAT_FRACTAL_NZ_C0_8) {
        return true;
    }
    return false;
}

Format ToFormat(const std::string& formatStr)
{
    static const std::map<std::string, Format> kStringToFormatMap = {
        {"NCHW", Format::FORMAT_NCHW},
        {"NHWC", Format::FORMAT_NHWC},
        {"ND", Format::FORMAT_ND},
        {"NC1HWC0", Format::FORMAT_NC1HWC0},
        {"FRACTAL_Z", Format::FORMAT_FRACTAL_Z},
        {"NC1C0HWPAD", Format::FORMAT_NC1C0HWPAD},
        {"NHWC1C0", Format::FORMAT_NHWC1C0},
        {"FSR_NCHW", Format::FORMAT_FSR_NCHW},
        {"FRACTAL_DECONV", Format::FORMAT_FRACTAL_DECONV},
        {"C1HWNC0", Format::FORMAT_C1HWNC0},
        {"FRACTAL_DECONV_TRANSPOSE", Format::FORMAT_FRACTAL_DECONV_TRANSPOSE},
        {"FRACTAL_DECONV_SP_STRIDE_TRANS", Format::FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS},
        {"NC1HWC0_C04", Format::FORMAT_NC1HWC0_C04},
        {"FRACTAL_Z_C04", Format::FORMAT_FRACTAL_Z_C04},
        {"CHWN", Format::FORMAT_CHWN},
        {"DECONV_SP_STRIDE8_TRANS", Format::FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS},
        {"NC1KHKWHWC0", Format::FORMAT_NC1KHKWHWC0},
        {"BN_WEIGHT", Format::FORMAT_BN_WEIGHT},
        {"FILTER_HWCK", Format::FORMAT_FILTER_HWCK},
        {"HWCN", Format::FORMAT_HWCN},
        {"LOOKUP_LOOKUPS", Format::FORMAT_HASHTABLE_LOOKUP_LOOKUPS},
        {"LOOKUP_KEYS", Format::FORMAT_HASHTABLE_LOOKUP_KEYS},
        {"LOOKUP_VALUE", Format::FORMAT_HASHTABLE_LOOKUP_VALUE},
        {"LOOKUP_OUTPUT", Format::FORMAT_HASHTABLE_LOOKUP_OUTPUT},
        {"LOOKUP_HITS", Format::FORMAT_HASHTABLE_LOOKUP_HITS},
        {"MD", Format::FORMAT_MD},
        {"C1HWNCoC0", Format::FORMAT_C1HWNCoC0},
        {"FRACTAL_NZ", Format::FORMAT_FRACTAL_NZ},
        {"NDHWC", Format::FORMAT_NDHWC},
        {"NCDHW", Format::FORMAT_NCDHW},
        {"DHWCN", Format::FORMAT_DHWCN},
        {"DHWNC", Format::FORMAT_DHWNC},
        {"NDC1HWC0", Format::FORMAT_NDC1HWC0},
        {"FRACTAL_Z_3D", Format::FORMAT_FRACTAL_Z_3D},
        {"FRACTAL_Z_3D_TRANSPOSE", Format::FORMAT_FRACTAL_Z_3D_TRANSPOSE},
        {"CN", Format::FORMAT_CN},
        {"NC", Format::FORMAT_NC},
        {"FRACTAL_ZN_LSTM", Format::FORMAT_FRACTAL_ZN_LSTM},
        {"FRACTAL_Z_G", Format::FORMAT_FRACTAL_Z_G},
        {"FORMAT_RESERVED", Format::FORMAT_RESERVED},
        {"ALL", Format::FORMAT_ALL},
        {"NULL", Format::FORMAT_NULL},
        // add for json input
        {"ND_RNN_BIAS", Format::FORMAT_ND_RNN_BIAS},
        {"FRACTAL_ZN_RNN", Format::FORMAT_FRACTAL_ZN_RNN},
        {"NYUV", Format::FORMAT_NYUV},
        {"NYUV_A", Format::FORMAT_NYUV_A},
        {"NCL", Format::FORMAT_NCL},
        {"FRACTAL_NZ_C0_16", Format::FORMAT_FRACTAL_NZ_C0_16},
        {"FRACTAL_NZ_C0_32", Format::FORMAT_FRACTAL_NZ_C0_32},
        {"FRACTAL_NZ_C0_2", Format::FORMAT_FRACTAL_NZ_C0_2},
        {"FRACTAL_NZ_C0_4", Format::FORMAT_FRACTAL_NZ_C0_4},
        {"FRACTAL_NZ_C0_8", Format::FORMAT_FRACTAL_NZ_C0_8},
        {"RESERVED", Format::FORMAT_RESERVED},
        {"UNDEFINED", Format::FORMAT_RESERVED}};

    Format format = Format::FORMAT_RESERVED;
    auto found = kStringToFormatMap.find(formatStr);
    if (found != kStringToFormatMap.end()) {
        format = found->second;
    } else {
        OP_LOGW("unknown format:%s.", formatStr.c_str());
    }

    return format;
}

ge::AscendString ToString(Format format)
{
    static const std::map<Format, std::string> kFormatToStringMap = {
        {Format::FORMAT_NCHW, "NCHW"},
        {Format::FORMAT_NHWC, "NHWC"},
        {Format::FORMAT_ND, "ND"},
        {Format::FORMAT_NC1HWC0, "NC1HWC0"},
        {Format::FORMAT_FRACTAL_Z, "FRACTAL_Z"},
        {Format::FORMAT_NC1C0HWPAD, "NC1C0HWPAD"},
        {Format::FORMAT_NHWC1C0, "NHWC1C0"},
        {Format::FORMAT_FSR_NCHW, "FSR_NCHW"},
        {Format::FORMAT_FRACTAL_DECONV, "FRACTAL_DECONV"},
        {Format::FORMAT_C1HWNC0, "C1HWNC0"},
        {Format::FORMAT_FRACTAL_DECONV_TRANSPOSE, "FRACTAL_DECONV_TRANSPOSE"},
        {Format::FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS, "FRACTAL_DECONV_SP_STRIDE_TRANS"},
        {Format::FORMAT_NC1HWC0_C04, "NC1HWC0_C04"},
        {Format::FORMAT_FRACTAL_Z_C04, "FRACTAL_Z_C04"},
        {Format::FORMAT_CHWN, "CHWN"},
        {Format::FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS, "DECONV_SP_STRIDE8_TRANS"},
        {Format::FORMAT_NC1KHKWHWC0, "NC1KHKWHWC0"},
        {Format::FORMAT_BN_WEIGHT, "BN_WEIGHT"},
        {Format::FORMAT_FILTER_HWCK, "FILTER_HWCK"},
        {Format::FORMAT_HWCN, "HWCN"},
        {Format::FORMAT_HASHTABLE_LOOKUP_LOOKUPS, "LOOKUP_LOOKUPS"},
        {Format::FORMAT_HASHTABLE_LOOKUP_KEYS, "LOOKUP_KEYS"},
        {Format::FORMAT_HASHTABLE_LOOKUP_VALUE, "LOOKUP_VALUE"},
        {Format::FORMAT_HASHTABLE_LOOKUP_OUTPUT, "LOOKUP_OUTPUT"},
        {Format::FORMAT_HASHTABLE_LOOKUP_HITS, "LOOKUP_HITS"},
        {Format::FORMAT_MD, "MD"},
        {Format::FORMAT_NDHWC, "NDHWC"},
        {Format::FORMAT_NCDHW, "NCDHW"},
        {Format::FORMAT_DHWCN, "DHWCN"},
        {Format::FORMAT_DHWNC, "DHWNC"},
        {Format::FORMAT_NDC1HWC0, "NDC1HWC0"},
        {Format::FORMAT_FRACTAL_Z_3D, "FRACTAL_Z_3D"},
        {Format::FORMAT_FRACTAL_Z_3D_TRANSPOSE, "FRACTAL_Z_3D_TRANSPOSE"},
        {Format::FORMAT_C1HWNCoC0, "C1HWNCoC0"},
        {Format::FORMAT_FRACTAL_NZ, "FRACTAL_NZ"},
        {Format::FORMAT_CN, "CN"},
        {Format::FORMAT_NC, "NC"},
        {Format::FORMAT_FRACTAL_ZN_LSTM, "FRACTAL_ZN_LSTM"},
        {Format::FORMAT_FRACTAL_Z_G, "FRACTAL_Z_G"},
        {Format::FORMAT_ND_RNN_BIAS, "ND_RNN_BIAS"},
        {Format::FORMAT_FRACTAL_ZN_RNN, "FRACTAL_ZN_RNN"},
        {Format::FORMAT_NYUV, "NYUV"},
        {Format::FORMAT_NYUV_A, "NYUV_A"},
        {Format::FORMAT_NCL, "NCL"},
        {Format::FORMAT_FRACTAL_NZ_C0_16, "FRACTAL_NZ_C0_16"},
        {Format::FORMAT_FRACTAL_NZ_C0_32, "FRACTAL_NZ_C0_32"},
        {Format::FORMAT_FRACTAL_NZ_C0_2, "FRACTAL_NZ_C0_2"},
        {Format::FORMAT_FRACTAL_NZ_C0_4, "FRACTAL_NZ_C0_4"},
        {Format::FORMAT_FRACTAL_NZ_C0_8, "FRACTAL_NZ_C0_8"},
        {Format::FORMAT_RESERVED, "Format::FORMAT_RESERVED"},
        {Format::FORMAT_ALL, "ALL"},
        {Format::FORMAT_NULL, "NULL"},
        {Format::FORMAT_END, "END"},
        {Format::FORMAT_MAX, "UNDEFINED"}};
    static const std::string reserved("RESERVED");
    const auto it = kFormatToStringMap.find(static_cast<Format>(GetPrimaryFormat(static_cast<int32_t>(format))));
    if (it != kFormatToStringMap.end()) {
        if (op::HasSubFormat(format)) {
            return ge::AscendString((it->second + ":" + std::to_string(op::GetSubFormat(format))).c_str());
        }
        return ge::AscendString((it->second).c_str());
    } else {
        OP_LOGW("unknown format:%d.", format);
        return ge::AscendString(reserved.c_str());
    }
}

} // namespace op
