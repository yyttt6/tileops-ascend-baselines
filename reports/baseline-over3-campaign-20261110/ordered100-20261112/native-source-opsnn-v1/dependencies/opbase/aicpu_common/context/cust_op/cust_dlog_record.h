/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AICPU_CONTEXT_CUST_OP_CUST_DLOG_RECORD_H_
#define AICPU_CONTEXT_CUST_OP_CUST_DLOG_RECORD_H_

#include <dlfcn.h>

#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>

#include "cust_cpu_utils.h"
#include "dlog_pub.h"

namespace aicpu {
class CustCpuKernelDlogUtils {
public:
    static CustCpuKernelDlogUtils& GetInstance();
    int32_t CustSetCpuKernelContext(uint64_t workspace_size, uint64_t workspace_addr);
    int64_t GetTid();
    std::shared_ptr<CpuKernelContext> GetCpuKernelContext();
    void DumpLog(int32_t module_id, int32_t level, const char* fmt, va_list args);
    const std::unordered_map<int, std::string> module_to_string_map_;

private:
    CustCpuKernelDlogUtils()
        : module_to_string_map_({{SLOG, "SLOG"},
                                 {IDEDD, "IDEDD"},
                                 {HCCL, "HCCL"},
                                 {FMK, "FMK"},
                                 {DVPP, "DVPP"},
                                 {RUNTIME, "RUNTIME"},
                                 {CCE, "CCE"},
                                 {HDC, "HDC"},
                                 {DRV, "DRV"},
                                 {DEVMM, "DEVMM"},
                                 {KERNEL, "KERNEL"},
                                 {LIBMEDIA, "LIBMEDIA"},
                                 {CCECPU, "CCECPU"},
                                 {ROS, "ROS"},
                                 {HCCP, "HCCP"},
                                 {ROCE, "ROCE"},
                                 {TEFUSION, "TEFUSION"},
                                 {PROFILING, "PROFILING"},
                                 {DP, "DP"},
                                 {APP, "APP"},
                                 {TS, "TS"},
                                 {TSDUMP, "TSDUMP"},
                                 {AICPU, "AICPU"},
                                 {LP, "LP"},
                                 {TDT, "TDT"},
                                 {FE, "FE"},
                                 {MD, "MD"},
                                 {MB, "MB"},
                                 {ME, "ME"},
                                 {IMU, "IMU"},
                                 {IMP, "IMP"},
                                 {GE, "GE"},
                                 {CAMERA, "CAMERA"},
                                 {ASCENDCL, "ASCENDCL"},
                                 {TEEOS, "TEEOS"},
                                 {ISP, "ISP"},
                                 {SIS, "SIS"},
                                 {HSM, "HSM"},
                                 {DSS, "DSS"},
                                 {PROCMGR, "PROCMGR"},
                                 {BBOX, "BBOX"},
                                 {AIVECTOR, "AIVECTOR"},
                                 {TBE, "TBE"},
                                 {FV, "FV"},
                                 {TUNE, "TUNE"},
                                 {HSS, "HSS"},
                                 {FFTS, "FFTS"},
                                 {OP, "OP"},
                                 {UDF, "UDF"},
                                 {HICAID, "HICAID"},
                                 {TSYNC, "TSYNC"},
                                 {AUDIO, "AUDIO"},
                                 {TPRT, "TPRT"},
                                 {ASCENDCKERNEL, "ASCENDCKERNEL"},
                                 {ASYS, "ASYS"},
                                 {ATRACE, "ATRACE"},
                                 {RTC, "RTC"},
                                 {SYSMONITOR, "SYSMONITOR"},
                                 {AML, "AML"},
                                 {ADETECT, "ADETECT"},
                                 {INVLID_MOUDLE_ID, "INVLID_MOUDLE_ID"}}) {};
    ~CustCpuKernelDlogUtils();
    CustCpuKernelDlogUtils(const CustCpuKernelDlogUtils&) = delete;
    CustCpuKernelDlogUtils(CustCpuKernelDlogUtils&&) = delete;
    CustCpuKernelDlogUtils& operator=(const CustCpuKernelDlogUtils&) = delete;
    CustCpuKernelDlogUtils& operator=(CustCpuKernelDlogUtils&&) = delete;
    std::unordered_map<int64_t, std::shared_ptr<CpuKernelContext>> ctx_map_;
    std::mutex ctx_mutex_;
};
} // namespace aicpu

extern "C" {
__attribute__((visibility("default"))) int32_t CustSetCpuKernelContext(uint64_t workspace_size,
                                                                       uint64_t workspace_addr);
}

#endif // AICPU_CONTEXT_CUST_OP_CUST_DLOG_RECORD_H_