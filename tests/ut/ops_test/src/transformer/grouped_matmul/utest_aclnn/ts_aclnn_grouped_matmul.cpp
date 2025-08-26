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
 * \file ts_aclnn_grouped_matmul.cpp
 * \brief GroupedMatmul ACLNN 测试用例.
 */

#include "ts_aclnn_grouped_matmul.h"

namespace {
TEST_P(Ts_Aclnn_GroupedMatmul_WithParam_Ascend910B3, Tc_Aclnn_GroupedMatmul)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mOpInfo.mExp.mSuccess);
}

const auto Tc_GroupedMatmul_Aclnn_Case = ::testing::Values(
    AclnnGroupedMatmulCase("Test_GMMV4_A8W4", true, "",                         /* CaseName,Enable,DebugInfo */
                            OpInfo(ControlInfo(true, true),                     /* RunTiling,RunKernel */
                                   ExpectInfo(true,                             /* ExpectSuccess */
                                          ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                          ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                            AclnnGroupedMatmulParam({GenTensorList("x", {{64,256}}, ge::DataType::DT_INT8),
                                                 GenTensorList("weight", {{1,256,64}}, ge::DataType::DT_INT4, ge::FORMAT_FRACTAL_NZ),   
                                                 GenTensorList("bias", {{1,64}}, ge::DataType::DT_FLOAT),
                                                 GenTensorList("scale", {{4,1,64}}, ge::DataType::DT_UINT64),
                                                 GenTensorList("offset", {{2}}, ge::DataType::DT_FLOAT),
                                                 GenTensorList("antiquant_scale", {{2}}, ge::DataType::DT_FLOAT16),
                                                 GenTensorList("antiquant_offset", {{2}}, ge::DataType::DT_FLOAT16),  
                                                 GenTensorList("pertoken_scale", {{64,1}}, ge::DataType::DT_FLOAT), 
                                                 GenTensorList("y", {{64,64}}, ge::DataType::DT_BF16)},
                                                 GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
                                          {64}, 2, -1, false, false, 0, 1, 0, FunctionType::QUANT_PERTOKEN, 
                                          AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase("Test_GMMV4_A16W4_INT32_PACKED_WEIGHT", true, "",       /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{84, 512}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{2, 512, 320}}, ge::DataType::DT_INT32),
                                          GenTensorList("bias", {{2, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("antiquant_scale", {{2, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("antiquant_offset", {{2, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{84, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {50, 84}, 3, -1, false, false, 0, 1, 0, FunctionType::ANTIQUANT, 
                                         AclnnGroupedMatmulVersion::V4)),     
    AclnnGroupedMatmulCase("Test_GMMV1_SPLIT_0", true, "",                         /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{50, 5120}, {65, 5120}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{5120, 2560}, {5120, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("bias", {{2560}, {2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{50, 2560}, {65, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {0}, ge::DataType::DT_INT64),
                                         {}, 0, -1, false, false, -1, 1, 0, FunctionType::NO_QUANT, 
                                         AclnnGroupedMatmulVersion::V1)),
    AclnnGroupedMatmulCase("Test_GMMV1_SPLIT_3", true, "",                         /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{50, 5120}, {15, 5120}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{5120, 2560}, {5120, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("bias", {{2560}, {2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{65, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {50, 65}, 2, -1, false, false, 0, 1, 0, FunctionType::NO_QUANT, 
                                         AclnnGroupedMatmulVersion::V2)),
    AclnnGroupedMatmulCase("Test_GMMV2_01", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{84, 5120}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{4, 5120, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("bias", {{4, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{84, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
                                         {50, 65, 69, 84}, 3, -1, false, false, 0, 1, 0, FunctionType::NO_QUANT, 
                                         AclnnGroupedMatmulVersion::V3)),
    AclnnGroupedMatmulCase("Test_GMMV4_01", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{84, 5120}}, ge::DataType::DT_INT8),
                                          GenTensorList("weight", {{4, 5120, 2560}}, ge::DataType::DT_INT8),
                                          GenTensorList("bias", {{4, 2560}}, ge::DataType::DT_INT32),
                                          GenTensorList("scale", {{4, 2560}}, ge::DataType::DT_INT64),
                                          GenTensorList("y", {{84, 2560}}, ge::DataType::DT_INT8)},
                                         GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
                                         {50, 65, 69, 84}, 3, -1, false, false, 0, 1, 0, FunctionType::QUANT, 
                                         AclnnGroupedMatmulVersion::V4)), 
    AclnnGroupedMatmulCase("Test_GMMV4_02", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{84, 5120}}, ge::DataType::DT_INT8),
                                          GenTensorList("weight", {{4, 5120, 2560}}, ge::DataType::DT_INT8),
                                          GenTensorList("bias", {{4, 2560}}, ge::DataType::DT_INT32),
                                          GenTensorList("scale", {{4, 2560}}, ge::DataType::DT_FLOAT),
                                          GenTensorList("y", {{84, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
                                         {50, 65, 69, 84}, 3, -1, true, false, 0, 1, 0, FunctionType::QUANT, 
                                         AclnnGroupedMatmulVersion::V4)), 
    AclnnGroupedMatmulCase("Test_GMMV4_03", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{84, 5120}}, ge::DataType::DT_INT8),
                                          GenTensorList("weight", {{4, 5120, 2560}}, ge::DataType::DT_INT8),
                                          GenTensorList("bias", {{4, 2560}}, ge::DataType::DT_INT32),
                                          GenTensorList("scale", {{4, 2560}}, ge::DataType::DT_BF16),
                                          GenTensorList("y", {{84, 2560}}, ge::DataType::DT_BF16)},
                                         GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
                                         {50, 65, 69, 84}, 3, -1, false, false, 0, 1, 0, FunctionType::QUANT, 
                                         AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase("Test_GMMV4_04", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{84, 5120}}, ge::DataType::DT_INT8),
                                          GenTensorList("weight", {{4, 5120, 2560}}, ge::DataType::DT_INT8),
                                          GenTensorList("bias", {{4, 2560}}, ge::DataType::DT_INT32),
                                          GenTensorList("scale", {{4, 2560}}, ge::DataType::DT_BF16),
                                          GenTensorList("pertoken_scale", {{84}}, ge::DataType::DT_FLOAT),
                                          GenTensorList("y", {{84, 2560}}, ge::DataType::DT_BF16)},
                                         GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
                                         {50, 65, 69, 84}, 3, -1, false, false, 0, 1, 0, FunctionType::QUANT_PERTOKEN, 
                                         AclnnGroupedMatmulVersion::V4)),      
    AclnnGroupedMatmulCase("Test_GMMV4_05", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{84, 5120}}, ge::DataType::DT_INT8),
                                          GenTensorList("weight", {{4, 5120, 2560}}, ge::DataType::DT_INT8),
                                          GenTensorList("bias", {{4, 2560}}, ge::DataType::DT_INT32),
                                          GenTensorList("scale", {{4, 2560}}, ge::DataType::DT_FLOAT),
                                          GenTensorList("pertoken_scale", {{84}}, ge::DataType::DT_FLOAT),
                                          GenTensorList("y", {{84, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
                                         {50, 65, 69, 84}, 3, -1, false, false, 0, 1, 0, FunctionType::QUANT_PERTOKEN, 
                                         AclnnGroupedMatmulVersion::V4)), 
    AclnnGroupedMatmulCase("Test_GMMV4_06", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{84, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{4, 2560, 2560}}, ge::DataType::DT_INT8),
                                          GenTensorList("bias", {{4, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("antiquant_scale", {{4, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("antiquant_offset", {{4, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{84, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
                                         {50, 65, 69, 84}, 3, -1, false, false, 0, 1, 0, FunctionType::ANTIQUANT, 
                                         AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase("Test_GMMV2_07", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{50, 5120}, {15, 5120}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{5120, 2560}, {5120, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("bias", {{2560}, {2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{50, 2560}, {15, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {0}, ge::DataType::DT_INT64),
                                         {}, 0, -1, false, false, -1, 1, 0, FunctionType::NO_QUANT, 
                                         AclnnGroupedMatmulVersion::V2)),
    AclnnGroupedMatmulCase("Test_GMMV1_08", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{50, 5120}, {15, 5120}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{5120, 2560}, {5120, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("bias", {{2560}, {2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{65, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {50, 65}, 3, -1, false, false, 0, 0, 0, FunctionType::NO_QUANT, 
                                         AclnnGroupedMatmulVersion::V1)),
    AclnnGroupedMatmulCase("Test_GMMV4_09", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{65, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{2560, 2560}, {2560, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{65, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {50, 65}, 3, -1, false, false, 0, 0, 0, FunctionType::NO_QUANT, 
                                         AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase("Test_GMMV4_10", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{65, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{2560, 2560}, {2560, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{50, 2560}, {15, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {50, 65}, 0, -1, false, false, 0, 0, 0, FunctionType::NO_QUANT, 
                                         AclnnGroupedMatmulVersion::V1)),
    AclnnGroupedMatmulCase("Test_GMMV4_11", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{1280, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{2560, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{2, 1280, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {1280, 1280}, 3, -1, false, true, 2, 1, 0, FunctionType::NO_QUANT, 
                                         AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase("Test_GMMV4_12", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{1280, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{2560, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{2, 1280, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {1280, 2560}, 3, -1, false, true, 2, 1, 0, FunctionType::NO_QUANT, 
                                         AclnnGroupedMatmulVersion::V2)), 
    AclnnGroupedMatmulCase("Test_GMMV4_13", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{16, 16}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{2, 16, 10}}, ge::DataType::DT_INT4),
                                          GenTensorList("bias", {{2, 10}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("antiquant_scale", {{2, 2, 10}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("antiquant_offset", {{2, 2, 10}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{16, 10}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {8, 8}, 3, -1, false, false, 0, 1, 0, FunctionType::ANTIQUANT, 
                                         AclnnGroupedMatmulVersion::V4)),    
    AclnnGroupedMatmulCase("Test_GMMV1_SPLIT_0", true, "",                         /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(false,                                /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{50, 5120}, {65, 5120}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{5120, 2560}, {5120, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("bias", {{2560}, {2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{50, 2560}, {65, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {0}, ge::DataType::DT_INT64),
                                         {}, 0, -1, false, false, -1, 1, 6, FunctionType::NO_QUANT, 
                                         AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase("Test_GMMV1__Error0", true, "",                         /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(false,                                /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{65, 5120}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{2, 5120, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("bias", {{2, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{65, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {50, 65}, 1, -1, false, false, -1, 1, 0, FunctionType::NO_QUANT, 
                                         AclnnGroupedMatmulVersion::V1)),
    AclnnGroupedMatmulCase("Test_GMMV1_Error_1", true, "",                         /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(false,                                /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{50, 5120}, {65, 5120}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{5120, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("bias", {{2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{50, 2560}, {65, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {0}, ge::DataType::DT_INT64),
                                         {}, 0, -1, false, false, -1, 1, 0, FunctionType::NO_QUANT, 
                                         AclnnGroupedMatmulVersion::V1)),
    AclnnGroupedMatmulCase("Test_GMMV2_Error_0", true, "",                         /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(false,                                /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{50, 512}, {15, 512}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{512, 256}, {512, 256}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("bias", {{256}, {256}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{65, 256}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {50, 65}, 2, -1, false, false, 1, 1, 0, FunctionType::NO_QUANT, 
                                         AclnnGroupedMatmulVersion::V2)),
    AclnnGroupedMatmulCase("Test_GMMV2_Error_1", true, "",                         /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(false,                                /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{2, 128, 256}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{256, 256}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{2, 128, 256}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {128, 256}, 3, -1, false, true, 2, 1, 0, FunctionType::NO_QUANT, 
                                         AclnnGroupedMatmulVersion::V2)),
    AclnnGroupedMatmulCase("Test_GMMV2_Error_2", true, "",                         /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(false,                                /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{1280, 1280}, {1280, 1280}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{2560, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{2, 1280, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {1280, 2560}, 3, -1, false, true, 2, 1, 0, FunctionType::NO_QUANT, 
                                         AclnnGroupedMatmulVersion::V2)),
    AclnnGroupedMatmulCase("Test_GMMV2_Error_3", true, "",                         /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(false,                                /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{65, 5120}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{2, 5120, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("bias", {{2, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{65, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {40, 50, 65}, 2, -1, false, false, 0, 1, 0, FunctionType::NO_QUANT, 
                                         AclnnGroupedMatmulVersion::V2)),
    AclnnGroupedMatmulCase("Test_GMMV2_Error_4", true, "",                         /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(false,                                /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{50, 5120}, {15, 5120}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{5120, 2560}, {5120, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("bias", {{2560}, {2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{40, 2560}, {25, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {}, ge::DataType::DT_INT64),
                                         {}, 0, -1, false, false, -1, 1, 0, FunctionType::NO_QUANT, 
                                         AclnnGroupedMatmulVersion::V2)),
    AclnnGroupedMatmulCase("Test_GMMV2_Error_5", true, "",                         /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(false,                                /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{50, 5120}, {15, 5120}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{5120, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("bias", {{2560}, {2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{50, 2560}, {15, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {50, 65}, 2, -1, false, false, 0, 1, 0, FunctionType::NO_QUANT, 
                                         AclnnGroupedMatmulVersion::V2)),
    AclnnGroupedMatmulCase("Test_GMMV2_Error_6", true, "",                         /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(false,                                /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{20, 512}, {20, 512}, {25, 512}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{512, 2560}, {512, 2560}, {512, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("bias", {{2560}, {2560}, {2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{65, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {20, 50, 65}, 2, -1, false, false, 0, 1, 0, FunctionType::NO_QUANT, 
                                         AclnnGroupedMatmulVersion::V2)),
    AclnnGroupedMatmulCase("Test_GMMV4_Error_0", true, "",                         /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(false,                                /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{2560, 1280}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{2, 2560, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{1280, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {1280, 1280}, 3, -1, false, false, 2, 1, 0, FunctionType::NO_QUANT, 
                                         AclnnGroupedMatmulVersion::V4)),      
    AclnnGroupedMatmulCase("Test_GMMV2_Error_1", true, "",                         /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(false,                                /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{84, 512}}, ge::DataType::DT_INT8),
                                          GenTensorList("weight", {{512, 2560}, {512, 2560}}, ge::DataType::DT_INT8),
                                          GenTensorList("bias", {{2, 2560}}, ge::DataType::DT_INT32),
                                          GenTensorList("scale", {{2, 2560}}, ge::DataType::DT_FLOAT),
                                          GenTensorList("pertoken_scale", {{84}}, ge::DataType::DT_FLOAT),
                                          GenTensorList("y", {{84, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {50, 84}, 3, -1, false, false, 0, 1, 0, FunctionType::QUANT_PERTOKEN, 
                                         AclnnGroupedMatmulVersion::V4)), 
    AclnnGroupedMatmulCase("Test_GMMV2_Error_2", true, "",                         /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(false,                                /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{84, 5120}}, ge::DataType::DT_INT8),
                                          GenTensorList("weight", {{4, 5120, 2560}}, ge::DataType::DT_INT8),
                                          GenTensorList("bias", {{4, 2560}}, ge::DataType::DT_INT32),
                                          GenTensorList("scale", {{4, 2560}}, ge::DataType::DT_BF16),
                                          GenTensorList("y", {{84, 2560}}, ge::DataType::DT_FLOAT)},
                                         GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
                                         {50, 65, 69, 84}, 3, -1, false, false, 0, 1, 0, FunctionType::QUANT, 
                                         AclnnGroupedMatmulVersion::V4))    
);
    
INSTANTIATE_TEST_SUITE_P(GroupedMatmul, Ts_Aclnn_GroupedMatmul_WithParam_Ascend910B3, Tc_GroupedMatmul_Aclnn_Case);
} // namespace