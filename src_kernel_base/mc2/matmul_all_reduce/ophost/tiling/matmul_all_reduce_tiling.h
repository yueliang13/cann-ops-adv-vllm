/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file matmul_all_reduce_tiling.h
 * \brief
 */
#ifndef MC2_MM_ALLREDUCE_TILING_H
#define MC2_MM_ALLREDUCE_TILING_H


#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "all_reduce_formulaic_tiling.h"
#include "ophost/weight_quant_batch_matmul_v2_weight_nz_tiling.h"
#include "ophost/quant_batch_matmul_v3_tiling.h"
#include "allreduce_tiling_struct.h"
#include "ophost/matmul_formulaic_tiling.h"
#include "ophost/mat_mul_v3_tiling.h"
#include "mc2_tiling_utils.h"
#include "context_transfer.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MatmulAllReduceTilingData)
    TILING_DATA_FIELD_DEF_STRUCT(Mc2Msg, msg);
    TILING_DATA_FIELD_DEF_STRUCT(RCSTiling, param);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, tailTiling);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling2);
    TILING_DATA_FIELD_DEF_STRUCT(L2cacheTilePara, tileL2cacheTiling);
    TILING_DATA_FIELD_DEF_STRUCT(L2cacheTilePara, tailL2cacheTiling);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatmulAllReduce, MatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduceTilingDataOp, MatmulAllReduceTilingData);


BEGIN_TILING_DATA_DEF(WeightQuantMatmulAllReduceNzTilingData)
TILING_DATA_FIELD_DEF_STRUCT(Mc2Msg, msg);
TILING_DATA_FIELD_DEF_STRUCT(RCSTiling, param);
TILING_DATA_FIELD_DEF_STRUCT(WeightQuantBatchMatmulV2NzTilingData, tilematmulTiling);
TILING_DATA_FIELD_DEF_STRUCT(WeightQuantBatchMatmulV2NzTilingData, tailmatmulTiling);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatmulAllReduce_80010, WeightQuantMatmulAllReduceNzTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_80011, WeightQuantMatmulAllReduceNzTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_80020, WeightQuantMatmulAllReduceNzTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_80021, WeightQuantMatmulAllReduceNzTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_80110, WeightQuantMatmulAllReduceNzTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_80111, WeightQuantMatmulAllReduceNzTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_80120, WeightQuantMatmulAllReduceNzTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_80121, WeightQuantMatmulAllReduceNzTilingData);

using AntiQuantType = QuantType;

constexpr uint8_t MC2_DEBUG_ONLY_AICPU = 4;
constexpr uint8_t DIM_NUM_THREE = 3;
constexpr uint8_t DIM_NUM_FOUR = 4;
constexpr uint8_t DIM_NUM_TWO = 2;
constexpr uint8_t DIM_NUM_ONE = 1;
constexpr uint64_t EMPTY_TENSOR_KEY = 10000000000000000009UL;
constexpr uint64_t WEIGHT_QUANT_EMPTY_TENSOR_KEY = 10000000000000000008UL;
constexpr uint64_t CUBE_ONLY_KEY = 10000000000000001100UL;
constexpr uint64_t MM_ALINGNED_TILING_KEY = 10000000000000000001UL;
constexpr uint64_t TILING_KEY_BASE_VALUE = 10000000000000000000UL;

struct L2TilePara {
    uint32_t mTile;
    uint32_t nTile;
    uint32_t mTileBlock;
    uint32_t nTileBlock;
    uint32_t mBlockCntTail;
    uint32_t nBlockCntTail;
};
enum class MatmulAllReduceTiling {
    ALL_REDUCE_GENERAL_910 = 1,
    ALL_REDUCE_GENERAL_310 = 2,
    ALL_REDUCE_A16W8 = 3,
    ALL_REDUCE_A16W4 = 4,
    ALL_REDUCE_A8W8 = 5,
};
const std::map<ge::DataType, uint64_t> D_TYPE_SIZE_MAP = {
    {ge::DT_BF16, 2},
    {ge::DT_FLOAT16, 2},
    {ge::DT_FLOAT, 4},
    {ge::DT_INT8, 1},
    {ge::DT_INT32, 4},
    {ge::DT_INT4, ge::GetSizeByDataType(ge::DT_INT4)},
};

const std::map<matmul_tiling::DataType, uint64_t> D_MTYPE_SIZE_MAP = {
    {matmul_tiling::DataType::DT_BFLOAT16, 2},
    {matmul_tiling::DataType::DT_FLOAT16, 2},
    {matmul_tiling::DataType::DT_FLOAT, 4},
    {matmul_tiling::DataType::DT_INT8, 1},
    {matmul_tiling::DataType::DT_INT32, 4},
    // DT_INT4的key值代码里面保证有效
    {matmul_tiling::DataType::DT_INT4, D_TYPE_SIZE_MAP.at(ge::DT_INT4)},
};

// currently 310p use
enum class ParamValue {
    INPUT = 0,
    WEIGHT = 1,
    BIAS = 2,
    X3 =  3,
    ANTIQUANT_SCALE = 4,
    ANTIQUANT_OFFSET = 5,
    DEQUANT = 6,
};

class MatmulAllReduceTilingBase : public TilingBaseClass {
public:
    explicit MatmulAllReduceTilingBase(gert::TilingContext *context)
        : TilingBaseClass(context), mmrCtxInfo_(mmrCtxInfoSelf_), tilingData_(tilingDataSelf_)
    {
        Reset();
        // 持有self代表作为独立个体工作，这个时候进行初始化设置tilingdata指向context内存
        tilingData_.SetDataPtr(context_->GetRawTilingData()->GetData());
        OP_TILING_CHECK(memset_s(context_->GetRawTilingData()->GetData(),
                                 context_->GetRawTilingData()->GetCapacity(), 0,
                                 context_->GetRawTilingData()->GetCapacity()) != EOK,
                        VECTOR_INNER_ERR_REPORT_TILIING(opName_, "fail to memset tiling data"),
                        return;);
    }
    MatmulAllReduceTilingBase(gert::TilingContext *context, MMRCtxInfo *mmrCtxInfo)
        : TilingBaseClass(context), mmrCtxInfo_(*mmrCtxInfo), tilingData_(tilingDataSelf_)
    { Reset(); }
    MatmulAllReduceTilingBase(gert::TilingContext *context, MMRCtxInfo *mmrCtxInfo,
                              MatmulAllReduceTilingData *tilingData)
        : TilingBaseClass(context), mmrCtxInfo_(*mmrCtxInfo), tilingData_(*tilingData)
    { Reset(); }
    ~MatmulAllReduceTilingBase() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    // every subclass need do IsCapable() DoOpTiling() and GetTilingKey()
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    virtual RCSTiling &MutableRCSTilingData()
    {
        return tilingData_.param;
    }
    virtual Mc2Msg &MutableMc2MsgData()
    {
        return tilingData_.msg;
    }
    virtual TCubeTiling &MutableTCubeTileTilingData()
    {
        return tilingData_.matmulTiling;
    }
    virtual TCubeTiling &MutableTCubeTailTilingData()
    {
        return tilingData_.tailTiling;
    }
    ge::graphStatus AnalyzeShapeAttr();
    void PrintTilingData();
    // tiling
    void DoAllReduceTiling(bool useHcclApi=false);
    void DoRCSTiling();
    void SetMCutSocVersion(SocVersion& inputSocVersion);
    void DoSplitMTiling();
    ge::graphStatus DoMatmulTiling(matmul_tiling::MultiCoreMatmulTiling& mm1, TCubeTiling& cubeTiling);
    void DoL2CacheTiling(L2cacheTilePara& l2cacheTiling);
    void setUseBufferType();

    void Reset();
    bool AnalyzeAttrs();
    void SetQuantData();
    void SetAntiQuantData();
    void SetCommQuantScale();
    void GetAtomicAddData();
    bool CheckDequantScaleShape(const uint64_t nValue) const;
    bool CheckPertokenScaleShape(const uint64_t mValue) const;
    bool CheckCommQuantScaleShape(const uint64_t nValue) const;
    bool CheckAntiQuantScaleShape(const uint64_t kValue, const uint64_t nValue);
    bool CheckAntiQuantOffsetValid() const;
    bool CheckA16W4Shape(const uint64_t kValue, const uint64_t nValue);
    bool CheckPlatformInfo() const;
    bool AnalyzeInputs();
    bool SetArgs(ge::DataType aType, ge::DataType bType, ge::DataType cType, ge::DataType biasType, bool isBias);
    uint64_t GetNValue() const;
    uint64_t GetKValue();
    uint64_t GetMValue();
    virtual ge::graphStatus CheckInput();
    ge::graphStatus CheckA16W16();
    ge::graphStatus CheckA8W8();
    ge::graphStatus CheckA16W8();
    mc2tiling::HcclDataType GetDataType(ge::DataType type);
    virtual void DoEmptyTensorTiling()
    {
    };
    virtual void GetL2CacheParm (uint64_t &l2CacheSize, uint64_t &singleMatrixSize, uint32_t &tileSize,
                                 uint32_t &tileLimit, bool useNewPara)
    {
        (void)l2CacheSize;
        (void)singleMatrixSize;
        (void)tileSize;
        (void)tileLimit;
        (void)useNewPara;
    };
    bool CalL2TilePara(L2TilePara& tileL2, uint64_t mValue, uint64_t kValue, uint64_t nValue, uint32_t cubeCoreNum);
    void PrintTilingData(optiling::TCubeTiling& tiling);
    void PrintTilingData(optiling::RCSTiling& tiling);
    void PrintTilingData(optiling::Mc2Msg& msg);
    AntiQuantType GetAntiQuantType();
    bool HasAntiQuantOffset() const;
    void CalcUbTiling() const;
    ge::graphStatus CheckRanksizePlatformSupported() const;
    uint64_t tileMValue_;
    uint64_t tailMValue_;
    bool isQuantKey_;
    bool isPerTensor_;
    AntiQuantType antiQuantType_{AntiQuantType::NONE};
    QuantType quantType_{QuantType::PER_TENSOR};
    uint64_t antiGroupSize_{0U};  // anti quant per group info
    bool isUbQuant_;
    bool enableL2Cache_;
    bool enableBiasConvert_;
    const char* opName_;
    const char* reduceOp_;
    uint32_t rankSize_;
    uint32_t libApiWorkSpaceSize_{0U};
    platform_ascendc::SocVersion socVersion_;
    bool supportL0c2Out_{false};
    mc2tiling::TilingArgs args_;
    bool is_weight_nz_{false};
    bool isKZero_{false};
    bool isA8W8_{false};
    bool isA16W8_{false};
    bool isA16W4_{false};
    MMRCtxInfo mmrCtxInfoSelf_{};
    MMRCtxInfo &mmrCtxInfo_;
    MatmulAllReduceTilingData tilingDataSelf_;
    MatmulAllReduceTilingData &tilingData_;
};
}  // namespace optiling
#endif  // MC2_MM_ALLREDUCE_TILING_H