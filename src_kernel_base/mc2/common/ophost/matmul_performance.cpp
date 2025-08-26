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
 * \file matmul_performance.cpp
 * \brief
 */

#include <iostream>
#include <string>
#include <map>
#include "log/ops_log.h"
#include "matmul_performance.h"

const std::map<std::string, double> CUBE_CALC_PER_CYCLE_MAP = {
    {DEFAULT_KEY_FOR_PAR_MAP, COMPUTES_PER_CYCLE},
    {"1_1_1_1_2", 8192},
    {"0_1_1_1_2", 8192},
    {"3_1_1_1_2", 8192},
    {"4_1_1_1_2", 8192},
};

const std::map<std::string, L2CacheEstimateParameters> L2_PARAMETER_MAP = {
    {DEFAULT_KEY_FOR_PAR_MAP, L2CacheEstimateParameters{128, 192, 0.85, 0.75, 0.65}},
    {"3_1_1_1_2", L2CacheEstimateParameters{96, 96, 0.4, 0.4, 0.35}},
    {"4_0_2_2_2", L2CacheEstimateParameters{80, 128, 0.95, 0.85, 0.75}},
    // 需要确认量化场景，如何定义其mac利用率，1）量化时的利用率，2）allgather、reducescatter和allreduce的差异
    {"4_1_1_1_2", L2CacheEstimateParameters{80, 128, 0.95, 0.85, 0.75}},
};

void MatmulPerformanceModel::GetMachineParameters()
{
    mmShapeInfo_.computesPerCycle = COMPUTES_PER_CYCLE;
    auto key = GetCalcTypeString();
    if (CUBE_CALC_PER_CYCLE_MAP.find(key) != CUBE_CALC_PER_CYCLE_MAP.end()) {
        OPS_LOG_W("common", "cube calculation power mapping HIT\n");
        mmShapeInfo_.computesPerCycle = CUBE_CALC_PER_CYCLE_MAP.at(key);
        return;
    }
}

uint64_t MatmulPerformanceModel::GetLinearThresholdLen(uint64_t rankTileNum)
{
    uint64_t tmpN = std::max(mmShapeInfo_.baseM, mmShapeInfo_.nValue);
    uint64_t tmpK = std::max(mmShapeInfo_.baseM, mmShapeInfo_.kValue);
    bool bmmFlag = (mmShapeInfo_.batchSize > 1 ? true : false);
    if (bmmFlag) {
        uint64_t mmMinDataSize = BMM_DATASIZE_LARGE_K;
        if (mmShapeInfo_.kValue < BMM_DATASIZE_K_BAR) {
            mmMinDataSize = BMM_DATASIZE_SMALL_K;
        }
        return std::max(mmMinDataSize / (tmpN * tmpK * rankTileNum),
            mmShapeInfo_.baseM);
    }

    bool discreteFlag = rankTileNum > SMALL_RANKTILE;
    if (discreteFlag) {
        uint64_t mmMinDataSize = 10 * ONE_GBYTE; // M * N * K >= 10G
        return std::max(mmMinDataSize / (tmpN * tmpK * rankTileNum),
            mmShapeInfo_.baseM);
    }

    uint64_t thresholdSize = 0;
    MinMatmulShapeParameters minShapePar;
    bool adaptCoreNumFlag = (mmShapeInfo_.socType != SocVersion::SOC310_P);
    if (adaptCoreNumFlag) {
        minShapePar.mmMinDataSize1 = minShapePar.mmMinDataSize1 / MARK_CORE_NUM_SOC910B * mmShapeInfo_.coreNum;
        minShapePar.mmMinDataSize2 = minShapePar.mmMinDataSize2 / MARK_CORE_NUM_SOC910B * mmShapeInfo_.coreNum;
        minShapePar.minMSize = MIN_M_SIZE_FACTOR * mmShapeInfo_.baseM;
    }
    thresholdSize = std::max(minShapePar.mmMinDataSize1 / (tmpN * tmpK),
        minShapePar.mmMinDataSize2 / (tmpN * tmpK / ONE_KBYTE + tmpN));
    if (rankTileNum > 1) {
        thresholdSize /= rankTileNum;
    }
    thresholdSize = std::max(minShapePar.minMSize, thresholdSize);

    return thresholdSize;
}

void MatmulPerformanceModel::CheckKvalueAlignVersion310P()
{
    if (mmShapeInfo_.kValue % K_ALIGN_LEN != 0) {
        cubeUtil_ *= K_UNALIGN_UTIL_RATIO_SOC310P;
    }
}

double MatmulPerformanceModel::FindCubeUtilByL2Usage(uint64_t mSize, uint64_t rankTileNum, uint64_t* maxTileLen)
{
    L2CacheEstimateParameters l2EstimatePar = L2_PARAMETER_MAP.at(DEFAULT_KEY_FOR_PAR_MAP);
    auto key = GetCalcTypeString();
    if (L2_PARAMETER_MAP.find(key) != L2_PARAMETER_MAP.end()) {
        OPS_LOG_W("Common", "l2 cache parameter mapping HIT\n");
        l2EstimatePar = L2_PARAMETER_MAP.at(key);
    }
    uint64_t tmpBaseM = std::min(mmShapeInfo_.baseM, mSize);
    uint64_t tmpBaseN = mmShapeInfo_.baseN;
    uint64_t mBlockNum = (mSize + tmpBaseM - 1) / tmpBaseM * rankTileNum;
    uint64_t nBlockNum = (mmShapeInfo_.nValue + tmpBaseN - 1) / tmpBaseN;
    uint64_t memUsage = std::min(mBlockNum, mmShapeInfo_.coreNum) * tmpBaseM +
        std::min(nBlockNum, mmShapeInfo_.coreNum) * tmpBaseN;
    memUsage = memUsage * mmShapeInfo_.kValue * mmShapeInfo_.inMatrixADtypeSize / ONE_MBYTE;
    double tmpUtil;
    double diff = 0;
    if (memUsage <= l2EstimatePar.sizePartL2) {
        tmpUtil = l2EstimatePar.utilPartL2; // 0.85x cube utilization rate
        diff = l2EstimatePar.sizePartL2 - memUsage;
    } else if (memUsage <= l2EstimatePar.sizeFullL2) {
        tmpUtil = l2EstimatePar.utilFullL2; // 0.75x cube utilization rate
        diff = l2EstimatePar.sizeFullL2 - memUsage;
    } else {
        tmpUtil = l2EstimatePar.utilOutL2; // 0.65x cube utilization rate
    }
    tmpUtil *= static_cast<double>(std::min(mBlockNum * nBlockNum, mmShapeInfo_.coreNum) /
        static_cast<double>(mmShapeInfo_.coreNum));

    if (maxTileLen != nullptr) {
        *maxTileLen = mmShapeInfo_.mValue;
        uint64_t tmpSize = std::min(mBlockNum + rankTileNum, mmShapeInfo_.coreNum) -
            std::min(mBlockNum, mmShapeInfo_.coreNum);
        tmpSize = tmpSize * mmShapeInfo_.inMatrixADtypeSize * tmpBaseM * mmShapeInfo_.kValue / ONE_MBYTE;
        if (tmpSize > diff && memUsage <= l2EstimatePar.sizeFullL2 && nBlockNum > mmShapeInfo_.coreNum) {
            *maxTileLen = (mSize + mmShapeInfo_.baseM - 1) / mmShapeInfo_.baseM * mmShapeInfo_.baseM;
        }
    }
    return tmpUtil;
}

double MatmulPerformanceModel::FindCubeUtilQuantVersion310P() const
{
    return 1.0 / (UTIL_BY_K_DENOMINATOR_OFFSET_SOC310P +
        UTIL_BY_K_DENOMINATOR_GRADIENT_SOC310P * static_cast<double>(ONE_KBYTE) /
        static_cast<double>(mmShapeInfo_.kValue));
}

double MatmulPerformanceModel::FindCubeUtilVersion310P() const
{
    return 1.0 / (UTIL_BY_K_DENOMINATOR_OFFSET_SOC310P_NO_QUANT +
        UTIL_BY_K_DENOMINATOR_GRADIENT_SOC310P_NO_QUANT * static_cast<double>(ONE_KBYTE) /
        static_cast<double>(mmShapeInfo_.kValue));
}

void MatmulPerformanceModel::ChangeCubeUtilByKAlign()
{
    uint64_t kLength = mmShapeInfo_.kValue * mmShapeInfo_.inMatrixADtypeSize;
    if (mmShapeInfo_.socType == SocVersion::SOC910_B4) {
        if (kLength % HALF_CACHE_LINE_LEN != 0) {
            cubeUtil_ *= K_UNALIGN_UTIL_RATIO_910_B4;
        }
        return;
    }

    if (kLength % HALF_CACHE_LINE_LEN != 0) {
        cubeUtil_ *= HALF_K_UNALIGN_UTIL_RATIO;
    } else if (kLength % CACHE_LINE_LEN != 0) {
        cubeUtil_ *= K_UNALIGN_UTIL_RATIO;
    }
}

void MatmulPerformanceModel::FindCubeUtil(uint64_t mSize, uint64_t rankTileNum, bool flagAllReduce, uint64_t* maxTileLen)
{
    if (mmShapeInfo_.socType == SocVersion::SOC310_P) {
        if (calcType_ == MatmulCalcType::QUANT) {
            cubeUtil_ = FindCubeUtilQuantVersion310P();
        }
        else {
            cubeUtil_ = FindCubeUtilVersion310P();
        }
        CheckKvalueAlignVersion310P();
        return;
    }

    cubeUtil_ = FindCubeUtilByL2Usage(mSize, rankTileNum, maxTileLen);
    if (flagAllReduce) {
        if (calcType_ == MatmulCalcType::QUANT) { // 全量化
            return ChangeCubeUtilByKAlign();
        }
        if (mmShapeInfo_.kValue % K_ALIGN_LEN != 0) { // 非量化和伪量化
            cubeUtil_ *= K_UNALIGN_UTIL_RATIO;
        }
    }
}

void MatmulPerformanceModel::GetMatmulGradient()
{
    uint64_t tmpN = std::max(mmShapeInfo_.baseM, mmShapeInfo_.nValue);
    uint64_t tmpK = std::max(mmShapeInfo_.baseM, mmShapeInfo_.kValue);
    matmulGradient_ = (tmpN * tmpK) /
        (mmShapeInfo_.computesPerCycle * mmShapeInfo_.cyclePerMicroSec  * mmShapeInfo_.coreNum * cubeUtil_);
}

double MatmulPerformanceModel::MatmulTime(uint64_t tileSize, uint64_t rankTileNum)
{
    tileSize = std::max(mmShapeInfo_.baseM, tileSize);
    return static_cast<double>(tileSize) * static_cast<double>(rankTileNum) * matmulGradient_;
}

uint64_t MatmulPerformanceModel::InverseMatmulTime(double targetTime, uint64_t rankTileNum)
{
    uint64_t tileSize = static_cast<uint64_t>(targetTime / (static_cast<double>(rankTileNum) * matmulGradient_));
    tileSize = std::max(mmShapeInfo_.baseM, tileSize);
    return tileSize;
}