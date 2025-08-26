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
 * \file mc2_tiling_struct.h
 * \brief
 */
#ifndef MC2_TILING_STRUCT_H
#define MC2_TILING_STRUCT_H

struct RCSTiling {
    uint32_t rankDim;
    uint32_t rankID;
    uint32_t commtype;
    uint32_t subtype;
    uint32_t tileCnt;
    uint32_t tailM;
    uint32_t tailCnt;
    uint32_t biasLen;
    uint32_t isAdd;
    uint32_t rankM;
    uint32_t rankN;
    uint32_t rankK;
    uint32_t gatherIndex;
    uint32_t isTransposeA;
    uint32_t isTransposeB;
    uint32_t storageGather;
    uint64_t nd2NzWorkLen;
    uint64_t cToFloatLen;
    uint64_t gatherLen;
    uint32_t workspaceAddr4;
    uint32_t aicCoreNum;
    uint32_t needUbBuffer;
    uint32_t addX3UbCnt;
    uint32_t commWorkSpaceSize;
    uint32_t isInputCommQuantScale;
    uint32_t dataType;
};

struct TileL2Tiling {
    uint32_t mL2TileCnt;
    uint32_t nL2TileCnt;
    uint32_t mTileBlocks;
    uint32_t nTileBlocks;
    uint32_t mTailBlocks;
    uint32_t nTailBlocks;
    uint32_t rankTileNum;
    uint32_t calcOrder;
    uint32_t enableL2Tile;
};

#endif // MC2_TILING_STRUCT_H
