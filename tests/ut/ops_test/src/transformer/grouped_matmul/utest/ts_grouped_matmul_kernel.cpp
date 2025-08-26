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
 * \file ts_grouped_matmul_kernel.cpp
 * \brief GroupedMatmul kernel用例.
 */

#include "ts_grouped_matmul.h"

namespace {
TEST_P(Ts_GroupedMatmul_WithParam_Ascend910B3, Tc_Kernel_GroupedMatmul)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mOpInfo.mExp.mSuccess);
}

const auto Tc_GroupedMatmul_Kernel_Case = ::testing::Values(
    GroupedMatmulCase( /* no split, m-m-m */
        "GroupedMatmul_Case0", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 256}, {1024, 256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{256, 256}, {256, 1024}}, ge::DataType::DT_FLOAT16),              
               GenTensorList("bias", {{256}, {1024}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 256}, {1024, 1024}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {}, ge::DataType::DT_INT64),
              {}, 0, -1, false, false, -1, 1, 0),
        0),
        GroupedMatmulCase( /* quant int8*/
            "GroupedMatmul_Case39", true, "", /* CaseName, Enable, DebugInfo */
            OpInfo(ControlInfo(true, true),
                   ExpectInfo(true, 8,
                              ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
            Param({GenTensorList("x", {{64,256}}, ge::DataType::DT_INT8),
                   GenTensorList("weight", {{1,1,16,16,64}}, ge::DataType::DT_INT4, ge::FORMAT_FRACTAL_NZ),   
                   GenTensorList("bias", {{1,64}}, ge::DataType::DT_FLOAT),
                   GenTensorList("scale", {{4,1,64}}, ge::DataType::DT_UINT64),
                   GenTensorList("offset", {{2}}, ge::DataType::DT_FLOAT),
                   GenTensorList("antiquant_scale", {{2}}, ge::DataType::DT_FLOAT16),
                   GenTensorList("antiquant_offset", {{2}}, ge::DataType::DT_FLOAT16),         
                   GenTensorList("y", {{64,64}}, ge::DataType::DT_BF16)},
                  GenTensor("per_token_scale", {64,1}, ge::DataType::DT_FLOAT), 
                  GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
                  {64}, 3, -1, false, false, 0, 1, 0),
            0),
    GroupedMatmulCase( /* split M, m-m-m */
        "GroupedMatmul_Case1", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{128, 256}, {1024, 256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{256, 256}, {256, 1024}}, ge::DataType::DT_FLOAT16),              
               GenTensorList("bias", {{256}, {1024}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{128, 256}, {1024, 1024}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {}, ge::DataType::DT_INT64),
              {}, 0, -1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* split M, s-m-s */
        "GroupedMatmul_Case2", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{256, 256}, {256, 256}}, ge::DataType::DT_FLOAT16),              
               GenTensorList("bias", {{256}, {256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 256}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {128, 128}, 3, -1, false, false, 0, 1, 0),
        0), 
    GroupedMatmulCase( /* split M, s-m-m */
        "GroupedMatmul_Case3", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{256, 256}, {256, 256}}, ge::DataType::DT_FLOAT16),              
               GenTensorList("bias", {{256}, {256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{128, 256}, {128, 256}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {128, 128}, 0, -1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase(  /* split K s-s-s + transpose x */
        "GroupedMatmul_Case4", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 1,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 768}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{768, 256}}, ge::DataType::DT_FLOAT16),              
               GenTensorList("y", {{256, 256}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {256, 512}, 3, -1, false, true, 2, 1, 0),
        0),   
    GroupedMatmulCase(  /* split K m-s-m + transpose x */
        "GroupedMatmul_Case5", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 1,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{256, 256}, {256, 512}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{768, 256}}, ge::DataType::DT_FLOAT16),              
               GenTensorList("y", {{256, 256}, {256, 256}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {}, ge::DataType::DT_INT64),
              {}, 0, -1, false, true, 2, 1, 0),
        0),
    GroupedMatmulCase( /* single tensor + transpose weight */
        "GroupedMatmul_Case6", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 2,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{256, 256}, {1024, 256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{2, 256, 256}}, ge::DataType::DT_FLOAT16),              
               GenTensorList("y", {{256, 256}, {1024, 256}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {}, ge::DataType::DT_INT64),
              {}, 0, -1, true, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* single tensor + w nz */
        "GroupedMatmul_Case7", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{64, 64}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{2, 4, 4, 16, 16}}, ge::DataType::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ),   
               GenTensorList("bias", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),        
               GenTensorList("y", {{32, 64}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {32, 32}, 3, -1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* per token quant */
        "GroupedMatmul_Case8", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{16, 5}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 5, 10}}, ge::DataType::DT_INT8),   
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 10}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),          
               GenTensorList("y", {{16, 10}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {16}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {8, 8}, 3, 1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* per token quant */
        "GroupedMatmul_Case9", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{360, 1024}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{16, 1024, 8192}}, ge::DataType::DT_INT8),   
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{16, 8192}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),          
               GenTensorList("y", {{360, 8192}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {360}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {16}, ge::DataType::DT_INT64),
              {40, 40, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20}, 2, 1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* per token quant */
        "GroupedMatmul_Case10", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{360, 8192}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{16, 8192, 1024}}, ge::DataType::DT_INT8),   
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{16, 1024}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),         
               GenTensorList("y", {{360, 1024}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {360}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {16}, ge::DataType::DT_INT64),
              {40, 40, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20}, 2, 1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* per token quant act */
        "GroupedMatmul_Case11", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{16, 10}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 10, 10}}, ge::DataType::DT_INT8),   
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 10}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),         
               GenTensorList("y", {{16, 10}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {16}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {8, 8}, 3, 1, false, false, 0, 1, 1),
        0),
    GroupedMatmulCase( /* quant act */
        "GroupedMatmul_Case12", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{16, 20}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 20, 10}}, ge::DataType::DT_INT8),   
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 10}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),       
               GenTensorList("y", {{16, 10}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {8, 8}, 3, 1, false, false, 0, 1, 1),
        0),
    GroupedMatmulCase( /* pertoken quant fp16*/
        "GroupedMatmul_Case13", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{16, 5}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 5, 10}}, ge::DataType::DT_INT8),   
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 10}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),         
               GenTensorList("y", {{16, 10}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {16}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {8, 8}, 3, 0, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* per token quant weight NZ */
        "GroupedMatmul_Case14", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{16, 128}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{4, 8, 8, 16, 32}}, ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ),   
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{4, 256}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),        
               GenTensorList("y", {{16, 256}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {16}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {4, 4, 4, 4}, 3, 1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* pertoken quant fp16*/
        "GroupedMatmul_Case15", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{32, 5}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 5, 10}}, ge::DataType::DT_INT8),   
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 10}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),         
               GenTensorList("y", {{32, 10}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {32}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {16, 16}, 3, 0, false, false, 0, 1, 1),
        0),
    GroupedMatmulCase( /* pertoken quant fp16*/
        "GroupedMatmul_Case16", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 4,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{32, 5}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 5, 20}}, ge::DataType::DT_INT8),   
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 20}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),         
               GenTensorList("y", {{32, 20}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {32}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {16, 16}, 3, 0, false, false, 0, 1, 2),
        0),
    GroupedMatmulCase( /* pertoken quant fp16*/
        "GroupedMatmul_Case17", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{32, 5}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 5, 30}}, ge::DataType::DT_INT8),   
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 30}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),         
               GenTensorList("y", {{32, 30}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {32}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {16, 16}, 3, 0, false, false, 0, 1, 4),
        0),
    GroupedMatmulCase( /* pertoken quant fp16*/
        "GroupedMatmul_Case18", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{32, 5}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 5, 40}}, ge::DataType::DT_INT8),   
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 40}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),         
               GenTensorList("y", {{32, 40}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {32}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {16, 16}, 3, 0, false, false, 0, 1, 5),
        0),
    GroupedMatmulCase( /* pertoken quant fp16*/
        "GroupedMatmul_Case19", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 5,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{16, 5}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 10, 5}}, ge::DataType::DT_INT8),   
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 10}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),         
               GenTensorList("y", {{16, 10}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {16}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {8, 8}, 3, 0, true, false, 0, 1, 2),
        0),
    GroupedMatmulCase( /* quant + transpose weight*/
        "GroupedMatmul_Case20", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 2,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{16, 64}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{4, 64, 32}}, ge::DataType::DT_INT8),              
               GenTensorList("bias", {{4, 32}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{4, 32}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{16, 32}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {4, 4, 4, 4}, 3, 1, true, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* quant 有问题*/
        "GroupedMatmul_Case21", false, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{256, 256}, {1024, 256}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{256, 256}, {256, 1024}}, ge::DataType::DT_INT8),              
               GenTensorList("bias", {{256}, {1024}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{256}, {1024}}, ge::DataType::DT_UINT64),
               GenTensorList("offset", {{256}, {1024}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 256}, {1024, 1024}}, ge::DataType::DT_INT8)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {}, ge::DataType::DT_INT64),
              {}, 0, -1, false, false, -1, 1, 0),
        0),
    GroupedMatmulCase( /* a16w8 fp16 */
        "GroupedMatmul_Case22", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{256, 256}, {128, 256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{256, 256}, {256, 128}}, ge::DataType::DT_INT8),              
               GenTensorList("bias", {{256}, {128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{256}, {128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{256}, {128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 256}, {128, 128}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {}, ge::DataType::DT_INT64),
              {}, 0, -1, false, false, -1, 1, 0),
        0),
    GroupedMatmulCase( /* a16w8 bf16 */
        "GroupedMatmul_Case23", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{256, 256}, {64, 256}}, ge::DataType::DT_BF16),
               GenTensorList("weight", {{256, 256}, {256, 64}}, ge::DataType::DT_INT8),              
               GenTensorList("bias", {{256}, {64}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{256}, {64}}, ge::DataType::DT_BF16),
               GenTensorList("antiquant_offset", {{256}, {64}}, ge::DataType::DT_BF16),
               GenTensorList("y", {{256, 256}, {64, 64}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {}, ge::DataType::DT_INT64),
              {}, 0, -1, false, false, -1, 1, 0),
        0),
    GroupedMatmulCase( /* a16w8 bf16 */
        "GroupedMatmul_Case24", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{128, 256}}, ge::DataType::DT_BF16),
               GenTensorList("weight", {{256, 256}, {256, 256}}, ge::DataType::DT_INT8),              
               GenTensorList("bias", {{2, 256}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{2, 256}}, ge::DataType::DT_BF16),
               GenTensorList("antiquant_offset", {{2, 256}}, ge::DataType::DT_BF16),
               GenTensorList("y", {{128, 256}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {8, 8}, 3, -1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* a16w4 fp16 */
        "GroupedMatmul_Case25", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{256, 256},}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{4, 256, 256}}, ge::DataType::DT_INT4),              
               GenTensorList("bias", {{4, 256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{4, 256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{4, 256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 256}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {64, 64, 64, 64}, 3, -1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* a16w4 fp16 transpose weight*/
        "GroupedMatmul_Case26", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 02,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{256, 256},}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{8, 256, 256}}, ge::DataType::DT_INT4),              
               GenTensorList("bias", {{8, 256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{8, 256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{8, 256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 256}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {32, 32, 32, 32, 32, 32, 32, 32}, 3, -1, true, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* a16w4 bf16 */
        "GroupedMatmul_Case27", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{256, 256},}, ge::DataType::DT_BF16),
               GenTensorList("weight", {{4, 256, 256}}, ge::DataType::DT_INT4),              
               GenTensorList("bias", {{4, 256}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{4, 256}}, ge::DataType::DT_BF16),
               GenTensorList("antiquant_offset", {{4, 256}}, ge::DataType::DT_BF16),
               GenTensorList("y", {{256, 256}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {64, 64, 64, 64}, 3, -1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* a16w8 fp16 + transpose weight */
        "GroupedMatmul_Case28", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 2,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{256, 256}, {128, 256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{256, 256}, {128, 256}}, ge::DataType::DT_INT8),              
               GenTensorList("bias", {{256}, {128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{256}, {128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{256}, {128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 256}, {128, 128}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {}, ge::DataType::DT_INT64),
              {}, 0, -1, true, false, -1, 1, 0),
        0),
    GroupedMatmulCase( /* antiquant performance */
        "GroupedMatmul_Case29", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 3,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{1024, 2048}, {1024, 2048}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{2048, 1000}, {2048, 1000}}, ge::DataType::DT_INT8),              
               GenTensorList("bias", {{1000}, {1000}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{1000}, {1000}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{1000}, {1000}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{1024, 1000}, {1024, 1000}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {256, 256, 256, 256}, 0, -1, false, false, -1, 1, 0),
        0),
    GroupedMatmulCase( /* antiquant msd */
        "GroupedMatmul_Case30", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 6,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{16, 128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{4, 128, 1024}}, ge::DataType::DT_INT8),              
               GenTensorList("bias", {{4, 1024}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{4, 1024}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{4, 1024}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{16, 1024}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {4, 4, 4, 4}, 3, -1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* antiquant msd + transpsoe weight */
        "GroupedMatmul_Case31", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 7,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{16, 128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{2, 1024, 128}}, ge::DataType::DT_INT8),              
               GenTensorList("bias", {{2, 1024}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{2, 1024}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{2, 1024}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{16, 1024}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {8, 8}, 3, -1, true, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* antiquant msd bf16 */
        "GroupedMatmul_Case32", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 6,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{16, 128}}, ge::DataType::DT_BF16),
               GenTensorList("weight", {{4, 128, 1024}}, ge::DataType::DT_INT8),              
               GenTensorList("bias", {{4, 1024}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{4, 1024}}, ge::DataType::DT_BF16),
               GenTensorList("antiquant_offset", {{4, 1024}}, ge::DataType::DT_BF16),
               GenTensorList("y", {{16, 1024}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {4, 4, 4, 4}, 3, -1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* antiquant msd bf16 + transpsoe weight */
        "GroupedMatmul_Case33", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 7,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{16, 128}}, ge::DataType::DT_BF16),
               GenTensorList("weight", {{2, 1024, 128}}, ge::DataType::DT_INT8),              
               GenTensorList("bias", {{2, 1024}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{2, 1024}}, ge::DataType::DT_BF16),
               GenTensorList("antiquant_offset", {{2, 1024}}, ge::DataType::DT_BF16),
               GenTensorList("y", {{16, 1024}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {8, 8}, 3, -1, true, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* antiquant pergroup */
        "GroupedMatmul_Case34", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{16, 16}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{2, 16, 10}}, ge::DataType::DT_INT8),              
               GenTensorList("bias", {{2, 10}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{2, 2, 10}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{2, 2, 10}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{16, 10}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {8, 8}, 0, -1, false, false, -1, 1, 0),
        0),
    GroupedMatmulCase(  /* split K s-s-s + transpose x */
        "GroupedMatmul_Case35", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 1,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{256, 768}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{768, 256}}, ge::DataType::DT_FLOAT16),              
               GenTensorList("y", {{256, 256}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {256, 512, 512, 768}, 3, -1, false, true, 2, 0, 0),
        0),
    GroupedMatmulCase( /* quant int8*/
        "GroupedMatmul_Case36", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{16, 5}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{5, 10}, {5, 10}}, ge::DataType::DT_INT8),   
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 10}}, ge::DataType::DT_UINT64),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),         
               GenTensorList("y", {{16, 10}}, ge::DataType::DT_INT8)},
              GenTensor("per_token_scale", {16}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {8, 8}, 3, -1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* quant int8*/
        "GroupedMatmul_Case37", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */        
        Param({GenTensorList("x", {{16, 5}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 5, 10}}, ge::DataType::DT_INT8),   
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 10}}, ge::DataType::DT_UINT64),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),         
               GenTensorList("y", {{16, 10}}, ge::DataType::DT_INT8)},
              GenTensor("per_token_scale", {16}, ge::DataType::DT_FLOAT), 
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {8, 8}, 3, -1, false, false, 0, 1, 0),
        0));

INSTANTIATE_TEST_SUITE_P(GroupedMatmul, Ts_GroupedMatmul_WithParam_Ascend910B3, Tc_GroupedMatmul_Kernel_Case);
} // namespace