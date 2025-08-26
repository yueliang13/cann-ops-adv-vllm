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
 * \file hccl_performance.h
 * \brief
 */
#ifndef __HCCL_PERFORMANCE_H__
#define __HCCL_PERFORMANCE_H__

#pragma once
#include "matmul_formulaic_tiling.h"
#include "formulaic_tiling_datatype.h"
constexpr uint64_t HCCL_MIN_TILE_LEN = 64 * ONE_KBYTE;
constexpr uint64_t HCCL_MIN_TILE_LEN_COARSE = 2 * ONE_MBYTE;
constexpr auto DEFAULT_KEY_FOR_FITTING_MAP = "0_0";
constexpr double FULL_MESH_TIME_FACTOR = 2.0;
constexpr uint64_t DOUBLE_RING_RANKTILE = 2;
constexpr uint64_t RING_TOTAL_STEP_PAR1 = 1;
constexpr uint64_t RING_TOTAL_STEP_PAR2 = 2;
constexpr uint64_t RANK_FOUR = 4;
constexpr uint64_t FITTING_RANK = 8;
constexpr double LOCAL_REDUCE_FACTOR = 0.4; // 确定性Local ReduceScatter额外耗时参数

class HCCLPerformanceModel {
public:
    HCCLFittingParameters commEstimatePar_;
    HCCLInfo commTypeInfo_;
    double commTimeFactor_ = 1.0;  // allreduce time is 2.0x allgather or reducescatter
    uint64_t lookUpTileNum_ = 1; // 数据拟合查表参数
    std::string keyToFittingMap_ = DEFAULT_KEY_FOR_FITTING_MAP;

    void SetCommParametersBaseSocType(SocVersion inputSocVersion)
    {
        if (inputSocVersion == SocVersion::SOC310_P) {
            SetCommMethodVersion310P();  // 310P只支持MatmulAllreduce算子
        } else {  // Default 910B
            InitParametersForFullMesh();
        }
    }
// Constructor
explicit HCCLPerformanceModel(uint32_t inputRankDim, KernelType inputKernelType,
                              SocVersion inputSocVersion = SocVersion::SOC910_B)
    {
        commTypeInfo_.kernelType = inputKernelType; // 区分哪个MC2算子
        commTypeInfo_.rankDim = std::max(static_cast<uint64_t>(inputRankDim), MIN_COMM_RANKDIM); // 并行维度最小为2
        SetCommParametersBaseSocType(inputSocVersion);
        SetMaxStepSize();
        keyToFittingMap_ = GetCommMethodString(inputSocVersion);
        GetCommEstimateParameters();
    };
    std::string GetCommMethodString(SocVersion socType) {
        return std::to_string(static_cast<int>(socType)) + "_" +
            std::to_string(static_cast<int>(commTypeInfo_.commMethod));
    }

    void SetCommMethodVersion310P();
    void SetMaxStepSize();
    void SetCommShapeLen(uint64_t len)
    {
        commTypeInfo_.commMatrixLen = len;
    }
    void SetCommDTypeSize(uint64_t dTypeSize)
    {
        commTypeInfo_.commDtypeSize = dTypeSize;
    }
    void SetCommTimeFactor(double factor)
    {
        commTimeFactor_ = factor;
    }
    void ChangeCommTimeFactorByDivision(double factor);
    void SetFullMeshCommTimeFactor();
    void SetRingCommTimeFactor();
    void InitParametersForFullMesh();
    void InitSOC91093();
    void SetLocalReduceFactor();

    uint64_t GetFullMeshRankTileNum();
    uint64_t GetRankTileNum();
    void GetCommEstimateParameters();
    uint64_t GetMaxStepSize();
    uint64_t GetLinearThresholdLen();
    uint64_t GetLinearThresholdLenCoarse();

    // 性能拟合函数
    double CommTime(uint64_t mSize) const;
    uint64_t InverseCommTime(double targetTime) const;
    void FindStepSize();
};

#endif // __HCCL_PERFORMANCE_H__