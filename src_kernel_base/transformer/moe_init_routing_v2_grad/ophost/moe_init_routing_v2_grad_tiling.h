/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_init_routing_v2_grad_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_INIT_ROUTING_V2_GRAD_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_INIT_ROUTING_V2_GRAD_H
#include <cmath>
#include <register/op_impl_registry.h>

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/tiling_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MoeV2GradComputeTilingData)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, perCoreElements);
TILING_DATA_FIELD_DEF(int64_t, lastCoreElements);
TILING_DATA_FIELD_DEF(int64_t, elementCopyLoops);
TILING_DATA_FIELD_DEF(int64_t, elementPerCopyCols);
TILING_DATA_FIELD_DEF(int64_t, elementLastCopyCols);
TILING_DATA_FIELD_DEF(int64_t, binaryAddBufferNum);
TILING_DATA_FIELD_DEF(int64_t, tmpBufferNum);
TILING_DATA_FIELD_DEF(int64_t, exponentOfBinary);
TILING_DATA_FIELD_DEF(int64_t, copyBufferSize);
TILING_DATA_FIELD_DEF(int64_t, tokensFormer);
TILING_DATA_FIELD_DEF(int64_t, perCoreTokensLoop);
TILING_DATA_FIELD_DEF(int64_t, perCoreTailTokensFormer);
TILING_DATA_FIELD_DEF(int64_t, lastCoreTokensLoop);
TILING_DATA_FIELD_DEF(int64_t, lastCoreTailTokensFormer);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeV2GradComputeTilingDataOp, MoeV2GradComputeTilingData)

BEGIN_TILING_DATA_DEF(MoeInitRoutingV2GradRegbaseSplitHTilingData)
TILING_DATA_FIELD_DEF(int64_t, n);
TILING_DATA_FIELD_DEF(int64_t, kUbFactor);
TILING_DATA_FIELD_DEF(int64_t, k);
TILING_DATA_FIELD_DEF(int64_t, activeNum);
TILING_DATA_FIELD_DEF(int64_t, h);
TILING_DATA_FIELD_DEF(int64_t, hBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, hUbFactor);
TILING_DATA_FIELD_DEF(int64_t, blockDim);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeInitRoutingV2Grad_200001, MoeInitRoutingV2GradRegbaseSplitHTilingData)
REGISTER_TILING_DATA_CLASS(MoeInitRoutingV2Grad_200002, MoeInitRoutingV2GradRegbaseSplitHTilingData)
REGISTER_TILING_DATA_CLASS(MoeInitRoutingV2Grad_200003, MoeInitRoutingV2GradRegbaseSplitHTilingData)

BEGIN_TILING_DATA_DEF(MoeInitRoutingV2GradRegbaseFullLoadTilingData)
TILING_DATA_FIELD_DEF(int64_t, n);
TILING_DATA_FIELD_DEF(int64_t, nBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, nUbFactor);
TILING_DATA_FIELD_DEF(int64_t, k);
TILING_DATA_FIELD_DEF(int64_t, activeNum);
TILING_DATA_FIELD_DEF(int64_t, h);
TILING_DATA_FIELD_DEF(int64_t, hUbFactor);
TILING_DATA_FIELD_DEF(int64_t, blockDim);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeInitRoutingV2Grad_300001, MoeInitRoutingV2GradRegbaseFullLoadTilingData)
REGISTER_TILING_DATA_CLASS(MoeInitRoutingV2Grad_300002, MoeInitRoutingV2GradRegbaseFullLoadTilingData)
REGISTER_TILING_DATA_CLASS(MoeInitRoutingV2Grad_300003, MoeInitRoutingV2GradRegbaseFullLoadTilingData)

BEGIN_TILING_DATA_DEF(MoeInitRoutingV2GradRegbaseTilingData)
TILING_DATA_FIELD_DEF(int64_t, n);
TILING_DATA_FIELD_DEF(int64_t, nBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, nUbFactor);
TILING_DATA_FIELD_DEF(int64_t, k);
TILING_DATA_FIELD_DEF(int64_t, kUbFactor);
TILING_DATA_FIELD_DEF(int64_t, activeNum);
TILING_DATA_FIELD_DEF(int64_t, h);
TILING_DATA_FIELD_DEF(int64_t, hUbFactor);
TILING_DATA_FIELD_DEF(int64_t, blockDim);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeInitRoutingV2Grad_400001, MoeInitRoutingV2GradRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(MoeInitRoutingV2Grad_400002, MoeInitRoutingV2GradRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(MoeInitRoutingV2Grad_400003, MoeInitRoutingV2GradRegbaseTilingData)

BEGIN_TILING_DATA_DEF(MoeInitRoutingV2GradTilingData)
TILING_DATA_FIELD_DEF(int64_t, coreNum);
TILING_DATA_FIELD_DEF(int64_t, n);
TILING_DATA_FIELD_DEF(int64_t, e);
TILING_DATA_FIELD_DEF(int64_t, c);
TILING_DATA_FIELD_DEF(int64_t, cols);
TILING_DATA_FIELD_DEF(int64_t, k);
TILING_DATA_FIELD_DEF(int64_t, activeNum);
TILING_DATA_FIELD_DEF_STRUCT(MoeV2GradComputeTilingData, MoeV2GradComputeParamsOp);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeInitRoutingV2Grad, MoeInitRoutingV2GradTilingData)

class MoeInitRoutingV2GradTilingBaseClass : public TilingBaseClass {
 public:
  explicit MoeInitRoutingV2GradTilingBaseClass(gert::TilingContext* context) : TilingBaseClass(context) {
  }

  void Reset(gert::TilingContext* context) override {
    TilingBaseClass::Reset(context);
  }

 private:
  ge::graphStatus CheckParamsValidity(const gert::Shape& xShape, const gert::Shape& rowIdxShape,
                                      const gert::Shape& gradXShape) const;
  ge::graphStatus CheckDtypeValidity();
  ge::graphStatus CheckShapeAllPositive(const gert::Shape& shape, std::string name);
  ge::graphStatus CheckShapeValidity(const gert::Shape& xShape, const gert::Shape& rowIdxShape,
                                     const gert::Shape& gradXShape);

 protected:
  ge::graphStatus GetPlatformInfo() override;
  ge::graphStatus GetShapeAttrsInfo() override;
  ge::graphStatus GetWorkspaceSize() override;
  ge::graphStatus DoLibApiTiling() override;
  int64_t ClipMax(int64_t cur, int64_t max) {
    return (cur > max) ? max : cur;
  }

  int64_t aivNum = 0;
  int64_t dropPadMode = 0;
  int64_t topK = 1;
  int64_t activeNum = 0;
  int64_t hiddenSize = 0;
  int64_t N = 0;
  int64_t E = 0;
  int64_t C = 0;
  ge::DataType inDtype = ge::DT_FLOAT;
  const char* opName = "";
  platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
};

struct MoeInitRoutingV2GradCompileInfo {};

template <typename T>
inline uint32_t GetUbBlockSize(T* context)
{
    return 32U;
}

template <typename T>
inline uint32_t GetVRegSize(T* context)
{
    return 256U;
}
}  // namespace optiling

template <typename T1, typename T2> inline T1 CeilDiv(T1 a, T2 b)
{
    if (b == 0) { return 0; }
    return (a + b - 1) / b;
};

template <typename T> inline T AlignUp(T a, T base)
{
    if (base == 0) {
        return 0;
    }
    return (a + base - 1) / base * base;
}

template <typename T> inline T AlignDown(T a, T base)
{
    if (base == 0) {
        return a;
    }
    return a / base * base;
}
#endif