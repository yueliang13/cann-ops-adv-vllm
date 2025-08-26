/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lock.h
 * \brief temporary rwlock implementation from willzhang4a58.github.io
 */

#ifndef CANN_OPS_BUILT_IN_LOCK_H_
#define CANN_OPS_BUILT_IN_LOCK_H_

#include <mutex>
#include <condition_variable>

namespace optiling {
class RWLock {
public:
    RWLock() : _status(0), _waiting_readers(0), _waiting_writers(0) {}
    ~RWLock() {};
    RWLock(const RWLock&) = delete;
    RWLock(RWLock&&) = delete;
    RWLock& operator = (const RWLock&) = delete;
    RWLock& operator = (RWLock&&) = delete;

    void rdlock();

    void wrlock();

    void unlock();

private:
    // negative: writer
    // positive: reader
    int32_t _status;
    int32_t _waiting_readers;
    int32_t _waiting_writers;
    std::mutex _mtx;
    std::condition_variable _read_cv;
    std::condition_variable _write_cv;
};
} // namespace optiling
#endif // CANN_OPS_BUILT_IN_LOCK_H_
