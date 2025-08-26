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
 * \file ts_grouped_matmul_tiling.cpp
 * \brief GroupedMatmul tiling用例.
 */

 #include "ts_grouped_matmul.h"

namespace {
TEST_P(Ts_GroupedMatmul_WithParam_Ascend910B3, Tc_Tiling_GroupedMatmul)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mOpInfo.mExp.mSuccess);
}

TEST_P(Ts_GroupedMatmul_WithParam_Ascend310P3, Tc_Tiling310_GroupedMatmul)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mOpInfo.mExp.mSuccess);
}

const auto Tc_GroupedMatmul_Tiling_Case = ::testing::Values(
    GroupedMatmulCase( 
        "GroupedMatmul_Case0", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{256, 512}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{512, 128}}, ge::DataType::DT_FLOAT16),              
               GenTensorList("bias", {{128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 128}}, ge::DataType::DT_FLOAT16)}, 
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {64, 64, 64, 64}, 0, -1, false, false, -1, 1, 0),
        0),
    GroupedMatmulCase( 
        "GroupedMatmul_Case1", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{256, 300}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{300, 128}}, ge::DataType::DT_FLOAT16),              
               GenTensorList("bias", {{128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 128}}, ge::DataType::DT_FLOAT16)}, 
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {64, 64, 64, 64}, 0, -1, false, false, 1, 1, 0),
        0),
    GroupedMatmulCase( 
        "GroupedMatmul_Case2", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{256, 400}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{400, 128}}, ge::DataType::DT_FLOAT16),              
               GenTensorList("bias", {{128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 128}, {256, 128}}, ge::DataType::DT_FLOAT16)}, 
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {128, 128}, 0, -1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( 
        "GroupedMatmul_Case3", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{256, 300}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{300, 128}, {300, 128}}, ge::DataType::DT_FLOAT16),              
               GenTensorList("bias", {{128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 128}}, ge::DataType::DT_FLOAT16)}, 
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {128, 128}, 3, -1, false, false, 2, 1, 0),
        0),
    GroupedMatmulCase( /* single tensor split m */
        "GroupedMatmul_Case4", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{256, 512}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{4, 512, 128}}, ge::DataType::DT_FLOAT16),              
               GenTensorList("bias", {{4, 128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 128}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {64, 64, 64, 64}, 3, -1, false, false, 0, 1, 0),
        0), 
    GroupedMatmulCase( /* single tensor + transpose w */
        "GroupedMatmul_Case5", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 2,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{256, 512}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{4, 128, 512}}, ge::DataType::DT_FLOAT16),              
               GenTensorList("bias", {{4, 128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 128}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {64, 64, 64, 64}, 3, -1, true, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* single tensor + transpose x */
        "GroupedMatmul_Case6", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 1,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{512, 256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{4, 512, 128}}, ge::DataType::DT_FLOAT16),              
               GenTensorList("bias", {{4, 128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 128}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {64, 64, 64, 64}, 3, -1, false, true, 0, 1, 0),
        0),
    GroupedMatmulCase( /* antiquant */
        "GroupedMatmul_Case7", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{16, 5}, {16, 5}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{5, 10}, {5, 10}}, ge::DataType::DT_INT8),              
               GenTensorList("bias", {{10}, {10}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{10}, {10}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{10}, {10}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{16, 10}, {16, 10}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {}, ge::DataType::DT_INT64),
              {}, 0, -1, false, false, -1, 0, 0),
        0),
    GroupedMatmulCase( /* per token quant weight NZ */
        "GroupedMatmul_Case8", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{360, 1024}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{16, 1024, 8191}}, ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ),   
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{16, 8191}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),        
               GenTensorList("y", {{360, 8191}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {360}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {16}, ge::DataType::DT_INT64),
              {40, 40, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20}, 2, 1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* per token quant weight NZ */
        "GroupedMatmul_Case9", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{360, 1024}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{16, 256, 64, 16, 32}}, ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ),   
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{16, 8192}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),        
               GenTensorList("y", {{360, 8192}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {360}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {16}, ge::DataType::DT_INT64),
              {20, 20, 40, 40, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20}, 2, 1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* per token quant weight NZ */
        "GroupedMatmul_Case10", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{360, 8192}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{32, 32, 512, 16, 32}}, ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ),   
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{32, 1024}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),         
               GenTensorList("y", {{360, 1024}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {360}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {32}, ge::DataType::DT_INT64),
              {11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 11, 12, 11, 12, 11, 12, 
               11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 11, 12, 11, 12, 11, 12}, 2, 1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* per token quant weight NZ */
        "GroupedMatmul_Case11", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{120, 8192}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{4, 32, 512, 16, 32}}, ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ),   
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{4, 1024}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),         
               GenTensorList("y", {{120, 1024}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {120}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {30, 30, 30, 30}, 2, 1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* quant int32 weight NZ */
        "GroupedMatmul_Case12", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 0, 8)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{120, 8192}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{1, 32, 512, 16, 32}}, ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ),   
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{0}}, ge::DataType::DT_UINT64),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),         
               GenTensorList("y", {{120, 1024}}, ge::DataType::DT_INT32)},
              GenTensor("per_token_scale", {120}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {120}, 2, 2, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* per token quant weight NZ */
        "GroupedMatmul_Case13", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 0, 8)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{120, 8192}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{1, 32, 512, 16, 32}}, ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ),   
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{1, 1024}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),         
               GenTensorList("y", {{120, 1024}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {120}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
              {120}, 2, 1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( 
        "GroupedMatmul_Case_FullLoad", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{25600, 384}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{4, 384, 2560}}, ge::DataType::DT_FLOAT16),              
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{25600, 2560}}, ge::DataType::DT_FLOAT16)}, 
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {6400, 6400, 6400, 6400}, 3, -1, false, false, 0, 1, 0),
        0));
    
INSTANTIATE_TEST_SUITE_P(GroupedMatmul, Ts_GroupedMatmul_WithParam_Ascend910B3, Tc_GroupedMatmul_Tiling_Case);

const auto Tc_GroupedMatmul_Tiling310_Case = ::testing::Values(
       GroupedMatmulCase( 
           "GroupedMatmul_Case0", true, "", /* CaseName, Enable, DebugInfo */
           OpInfo(ControlInfo(true, false),
                  ExpectInfo(true, 0, 8)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
           Param({GenTensorList("x", {{256, 512}}, ge::DataType::DT_FLOAT16),
                  GenTensorList("weight", {{512, 128}}, ge::DataType::DT_FLOAT16),              
                  GenTensorList("bias", {{128}}, ge::DataType::DT_FLOAT16),
                  GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
                  GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
                  GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
                  GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
                  GenTensorList("y", {{256, 128}}, ge::DataType::DT_FLOAT16)}, 
                 GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT), 
                 GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
                 {64, 64, 64, 64}, 0, -1, false, false, -1, 1, 0),
           0));

INSTANTIATE_TEST_SUITE_P(GroupedMatmul, Ts_GroupedMatmul_WithParam_Ascend310P3, Tc_GroupedMatmul_Tiling310_Case);

} // namespace