/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file io.h
 * \brief
 */

#pragma once

#include <string>

namespace ops::adv::tests::utils {

[[maybe_unused]] bool FileExist(const std::string &filePath);

[[maybe_unused]] bool ReadFile(const std::string &filePath, size_t &fileSize, void *buffer, size_t bufferSize);

[[maybe_unused]] bool WriteFile(const std::string &filePath, const void *buffer, size_t size);

} // namespace ops::adv::tests::utils
