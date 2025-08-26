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
 * \file ffn_high_performence.h
 * \brief
 */

#ifndef ASCENDC_FFN_HIGH_PERFORMENCE_H
#define ASCENDC_FFN_HIGH_PERFORMENCE_H

#include "ffn_base.h"

namespace FFN {
template <typename T, class mm1Type, class mm2Type = mm1Type, typename c1T = T, typename c2T = c1T, typename BiasT = T>
class FFNHighPerformence : public FFNBase<T, mm1Type, mm2Type, c1T, c2T, BiasT> {
public:
    __aicore__ inline FFNHighPerformence(typename mm1Type::MT &mm1_, typename mm2Type::MT &mm2_)
        : FFNBase<T, mm1Type, mm2Type, c1T, c2T, BiasT>(mm1_, mm2_)
    {
    }

    /** @brief main entry function.
     */
    __aicore__ inline void Process()
    {
        if (unlikely(this->ProcessZeroN1())) {
            return;
        }

        this->ProcessNormal();
    }
};

} // namespace FFN

#endif // ASCENDC_FFN_HIGH_PERFORMENCE_H