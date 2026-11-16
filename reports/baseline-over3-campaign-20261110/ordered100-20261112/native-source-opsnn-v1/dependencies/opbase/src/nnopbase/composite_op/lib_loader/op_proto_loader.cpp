/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_proto_loader.h"
#include <memory>

#include "graph/ascend_string.h"
#include "base/registry/opp_package_utils.h"
#include "base/registry/op_impl_space_registry_v2.h"

#include "file_utils.h"
#include "opdev/op_log.h"
#include "kernel_utils.h"

using namespace std;
namespace op {
namespace opploader {

constexpr char const* OP_PROTO_PATH_FROM_NEW_OPP_PKG = "built-in/op_proto/";
constexpr char const* OP_PROTO_PATH_FROM_OLD_OPP_PKG = "op_proto/built-in/";
constexpr char const* OP_PROTO_SO_SUFFIX = "rt2.0.so";

static void GetBuiltinOpProtoPath(const std::string& oppPath, std::string& opProtoPath)
{
    std::string newVersionPath = oppPath + OP_PROTO_PATH_FROM_NEW_OPP_PKG;
    if (IsDir(newVersionPath)) {
        opProtoPath = newVersionPath;
    } else {
        opProtoPath = oppPath + OP_PROTO_PATH_FROM_OLD_OPP_PKG;
    }
}

static void LoadProtoSo(const string& so)
{
    gert::OppImplVersionTag oppVersionTag = op::internal::GetOppImplVersion();
    auto spaceRegistryV2 = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry(oppVersionTag);
    if (spaceRegistryV2 == nullptr) {
        spaceRegistryV2 = std::make_shared<gert::OpImplSpaceRegistryV2>();
        OP_CHECK(spaceRegistryV2 != nullptr, OP_LOGW("New space registry v2 failed"), return);
        gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(spaceRegistryV2, oppVersionTag);
        OP_LOGI("Default space registry v2 is null, set new space registry v2");
    }
    const ge::AscendString soPath(so.c_str());
    const ge::AscendString desc("built-in");
    std::vector<ge::AscendString> soPathVec;
    soPathVec.emplace_back(soPath);
    gert::OppSoDesc soDesc(soPathVec, desc);
    OP_CHECK(spaceRegistryV2->AddSoToRegistry(soDesc) == ge::GRAPH_SUCCESS,
             OP_LOGW("AddSoToRegistry failed, so path: %s", so.c_str()), return);
    OP_LOGI("AddSoToRegistry successfully, so path: %s", so.c_str());
}

aclnnStatus LoadBuiltinOpProto(const string& oppPath)
{
    string opProtoPath;
    GetBuiltinOpProtoPath(oppPath, opProtoPath);
    OP_LOGI("Load builtin op proto, op proto path: %s", opProtoPath.c_str());
    const string osType = "linux";
#ifdef __x86_64__
    const string cpuType = "x86_64";
#elif defined(__aarch64__)
    const string cpuType = "aarch64";
#else
    const string cpuType = "unknown";
#endif
    string opProtoSoPath = opProtoPath + "lib/" + osType + "/" + cpuType + "/";
    vector<string> soList;
    GetFilesWithSuffix(opProtoSoPath, OP_PROTO_SO_SUFFIX, soList);
    for (const auto& soRealPath : soList) {
        LoadProtoSo(soRealPath);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus LoadOpProto(const string& oppPath) { return LoadBuiltinOpProto(oppPath); }

} // namespace opploader
} // namespace op
