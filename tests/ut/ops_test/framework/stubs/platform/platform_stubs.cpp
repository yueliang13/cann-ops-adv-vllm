/**
 * Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file platform_stubs.cpp
 * \brief
 */

#include "tests/utils/platform.h"

using Platform = ops::adv::tests::utils::Platform;

namespace fe {
enum class LocalMemType {
    L0_A = 0,
    L0_B = 1,
    L0_C = 2,
    L1 = 3,
    L2 = 4,
    UB = 5,
    HBM = 6,
    RESERVED
};

class PlatFormInfos {
public:
    uint32_t GetCoreNum(void) const;
    bool GetPlatformRes(const std::string &label, const std::string &key, std::string &val);
    bool GetPlatformResWithLock(const std::string &label, const std::string &key, std::string &val);
    void GetLocalMemSize(const LocalMemType &mem_type, uint64_t &size);
    void GetLocalMemBw(const LocalMemType &mem_type, uint64_t &bw_size);
};

uint32_t PlatFormInfos::GetCoreNum(void) const
{
    auto *platform = Platform::GetGlobalPlatform();
    if (platform != nullptr) {
        uint32_t coreNum = 0;
        return platform->socSpec.Get("SoCInfo", "ai_core_cnt", coreNum) ? coreNum : 0;
    } else {
        return 0;
    }
}

bool PlatFormInfos::GetPlatformRes(const std::string &label, const std::string &key, std::string &val)
{
    auto *platform = Platform::GetGlobalPlatform();
    if (platform != nullptr) {
        return platform->socSpec.Get(label.c_str(), key.c_str(), val);
    } else {
        return false;
    }
}

bool PlatFormInfos::GetPlatformResWithLock(const std::string &label, const std::string &key, std::string &val)
{
    return GetPlatformRes(label, key, val);
}

void PlatFormInfos::GetLocalMemSize(const LocalMemType &mem_type, uint64_t &size)
{
    auto *platform = Platform::GetGlobalPlatform();
    if (platform != nullptr) {
        std::string memType = std::to_string(static_cast<int32_t>(mem_type));
        (void)platform->socSpec.Get("LocalMemSize", memType.c_str(), size);
    } else {
        size = 0;
    }
}

void PlatFormInfos::GetLocalMemBw(const LocalMemType &mem_type, uint64_t &bw_size)
{
    auto *platform = Platform::GetGlobalPlatform();
    if (platform != nullptr) {
        std::string memType = std::to_string(static_cast<int32_t>(mem_type));
        (void)platform->socSpec.Get("LocalMemBw", memType.c_str(), bw_size);
    } else {
        bw_size = 0;
    }
}

class PlatformInfoManager {
    uint32_t GetPlatformInstanceByDevice(unsigned int const &, fe::PlatFormInfos &);
    uint32_t InitializePlatformInfo();
};

uint32_t PlatformInfoManager::GetPlatformInstanceByDevice(unsigned int const &, fe::PlatFormInfos &)
{
    return 0;
}

uint32_t PlatformInfoManager::InitializePlatformInfo()
{
    return 0;
}

} // namespace fe


namespace op {
// add stub func for GetCurrentPlatformInfo()&GetSocVersion() in aclnn
enum class SocVersion {
    ASCEND910B = 1,
    ASCEND310P = 6
};

class PlatformInfo {
public:
    SocVersion GetSocVersion() const;
};

SocVersion PlatformInfo::GetSocVersion() const
{
    auto *platform = Platform::GetGlobalPlatform();
    std::string socVersion("Ascend910B");
    if (platform != nullptr) {
        platform->socSpec.Get("version", "Short_SoC_version", socVersion);
    }
    if (socVersion == "Ascend310P") {
        return SocVersion::ASCEND310P;
    }
    return SocVersion::ASCEND910B;
}

static PlatformInfo platformInfo;

const PlatformInfo &GetCurrentPlatformInfo() {
    return platformInfo;
}
} // namespace op
