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
 * \file flash_attention_score_grad_tiling_unpadded_attension.cc
 * \brief
 */

#include "flash_attention_score_grad_tiling_s1s2_bn2gs1s2.h"
#include "tiling/tiling_templates_registry.h"

namespace optiling {

class FlashAttentionScoreGradTilingUnpaddedAttension : public FlashAttentionScoreGradTilingS1s2Bn2gs1s2 {
public:
    explicit FlashAttentionScoreGradTilingUnpaddedAttension(gert::TilingContext *context)
        : FlashAttentionScoreGradTilingS1s2Bn2gs1s2(context)
    {
    }

    bool IsCapable() override
    {
        if (tnd2bsh) {
            OPS_LOG_I("FlashAttentionScoreGradTilingUnpaddedAttension is not support tnd to bsh.");
            return false;
        }

        auto actualSeqQLenTensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_LEN);
        if (actualSeqQLenTensor != nullptr && actualSeqQLenTensor->GetShapeSize() != 0) {
            OPS_LOG_D("FlashAttentionScoreGradTilingUnpaddedAttension hit");
            return true;
        }

        return false;
    };
};

REGISTER_TILING_TEMPLATE("FlashAttentionScoreGrad", FlashAttentionScoreGradTilingUnpaddedAttension, 2000);

} // namespace optiling
