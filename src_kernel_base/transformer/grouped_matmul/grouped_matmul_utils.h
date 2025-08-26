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
 * \file grouped_matmul_utils.h
 * \brief
 */
#ifndef ASCENDC_GROUPED_MATMUL_UTILS_H
#define ASCENDC_GROUPED_MATMUL_UTILS_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#if defined(ORIG_DTYPE_X) && defined(ORIG_DTYPE_WEIGHT) && defined(ORIG_DTYPE_Y) && defined(DT_INT8) && \
    defined(DT_BF16) && defined(DT_INT4)
  #if ORIG_DTYPE_X == ORIG_DTYPE_WEIGHT
    #if ORIG_DTYPE_X == DT_INT8
      #if ORIG_DTYPE_Y == DT_BF16
        #define GMM_QUANT_BF16
        #define MM_DTYPE_Y int32_t
      #elif ORIG_DTYPE_Y == DT_FLOAT16
        #define GMM_QUANT_FLOAT16
        #define MM_DTYPE_Y int32_t
      #elif ORIG_DTYPE_Y == DT_INT32
        #define GMM_QUANT_INT32
      #else
        #define GMM_QUANT_INT8
      #endif
    #else
      #define GMM_FLOAT
    #endif
  #else
    #define GMM_ANTI_QUANT
    #if ORIG_DTYPE_X == DT_INT8 && ORIG_DTYPE_WEIGHT == DT_INT4
      #define GMM_ANTI_QUANT_A8W4_MSD
      #if ORIG_DTYPE_Y == DT_BF16
        #define GMM_ANTI_QUANT_A8W4_MSD_OUT_BF16
      #else
        #define GMM_ANTI_QUANT_A8W4_MSD_OUT_FP16
      #endif
      #define MM_DTYPE_Y int32_t
    #else
      #define GMM_ANTI_QUANT
    #endif
  #endif
#endif

#if defined(DTYPE_Y) && !defined(MM_DTYPE_Y)
    #define MM_DTYPE_Y DTYPE_Y
#endif

#if defined(CONST_TILING)
  #define TILING_TYPE const int32_t
#else
  #define TILING_TYPE __gm__ int32_t
#endif

#if defined(CONST_TILING)
  #define GET_TILING_DATA_MEMBER_ADDR(tilingType, member, var, tiling)            \
    GET_TILING_DATA_MEMBER(GMMTilingData, member, obj, tiling);                   \
    const int32_t* (var) = (const int32_t*)((const uint8_t*)&obj);
#else
  #define GET_TILING_DATA_MEMBER_ADDR(tilingType, member, var, tiling)            \
    size_t offset##var = (size_t)(&((tilingType*)0)->member);                     \
    __gm__ int32_t* (var) = (__gm__ int32_t*)((tiling) + (offset##var));
#endif

namespace GROUPED_MATMUL {
using namespace AscendC;

constexpr uint32_t INT8_BITS = 8;  // a int8 number has 8 bits
constexpr int32_t MKN_LIST_LEN = 128;  // 128: predefined array legnth
constexpr uint32_t UB_BLOCK_UNIT_SIZE = 32;  // 32: a block has 32 bytes data
constexpr uint32_t UB_BLOCK_DOUBLE_UNIT_SIZE = 64;  // 64: a block has 64 bytes data
constexpr uint32_t HALF_UB_BLOCK_UNIT_SIZE = UB_BLOCK_UNIT_SIZE / 2;  // 2: a float16 data has two bytes
constexpr MatmulConfig NZ_CFG_MDL = GetMDLConfig(false, false, 0, true, false, false, false);
constexpr MatmulConfig matmulCFGUnitFlag{false, false, true, 0, 0, 0, false, false, false, false, false, 0, 0, 0,
                                         0, 0, 0, 0, true};

constexpr uint64_t SYNC_AIV_AIC_FLAG = 3;
constexpr uint64_t SYNC_AIC_AIV_FLAG = 5;
constexpr uint64_t SYNC_MODE2 = 2;

template<class AT_, class BT_, class CT_, class BiasT_, const auto& MM_CFG = CFG_MDL>
struct MMType {
  using AT = AT_;
  using BT = BT_;
  using CT = CT_;
  using BiasT = BiasT_;
  using MT = matmul::Matmul<AT, BT, CT, BiasT, MM_CFG>;
};

template<class AT_, class BT_, class CT_, class BiasT_, const auto& MM_CFG = CFG_MDL>
struct MMImplType {
  using AT = AT_;
  using BT = BT_;
  using CT = CT_;
  using BiasT = BiasT_;
  using MT = matmul::MatmulImpl<AT, BT, CT, BiasT, MM_CFG>;
};

enum class ActiveType {
  INVALID_TYPE = 0,
  RELU,
  GELU_TANH,
  GELU_ERR_FUNC,
  FASTGELU,
  SILU
};

template <typename T>
__aicore__ inline T GreatestCommonDivisor(T a, T b) {
    T c = a;
    if (a < b) {
      a = b;
      b = c;
    }
    while (b != 0) {
      c = a;
      a = b;
      b = c % b;
    }
    return a;
}

template <typename T>
__aicore__ inline T LeastCommonMultiple(T a, T b) {
    return a * b / GreatestCommonDivisor(a, b);
}

template <typename T>
__aicore__ inline T Max(T a, T b) {
    return a > b ? a : b;
}

template <typename T>
__aicore__ inline T Min(T a, T b) {
    return a > b ? b : a;
}

template <uint32_t base, typename T = uint32_t>
__aicore__ inline T AlignUp(T a) {
  return (a + base - 1) / base * base;
}

template <typename T>
__aicore__ inline T AlignUp(T a, T base) {
  return (a + base - 1) / base * base;
}

template <typename T>
__aicore__ inline T AlignDown(T a, T base) {
  if (unlikely(base == 0)) {
    return a;
  }
  return a / base * base;
}

template <>
__aicore__ inline uint32_t AlignUp<4, uint32_t>(uint32_t a) {
  // to be Multiple of 4, result should be in a format of b(xxxx,x100).
  // This means last two bits should be zero, requiring that
  // result = num & b(1111,1100) = num & (~3).
  // &(~3) operator may reduces num into the range [num, num - 3].
  // As the result should be no less than a (result >= a), it means num - 3 >= a in the worst case.
  // In this case, num >= a+3. On the other hand, num should also be less then a+4, otherwise,
  // the result will not be least multiple of 4 for 3. In other cases like [num, num - 2],
  // num = a + 3 also satisfies the goal condition.
  return (a + 3) & ~3;  // & ~3: set last two bits of (a+3) to be zero
}

template <>
__aicore__ inline uint32_t AlignUp<8, uint32_t>(uint32_t a) {
  // In general, if we want to get the least multiple of b (b is the power of 2) for a,
  // it comes to a conclusion from the above comment: result = (a + (b - 1)) & (~b)
  return (a + 7) & ~7;  // & ~7: set last four bits of (a+7) to be zero
}

template <>
__aicore__ inline uint32_t AlignUp<16, uint32_t>(uint32_t a) {
  // In general, if we want to get the least multiple of b (b is the power of 2) for a,
  // it comes to a conclusion from the above comment: result = (a + (b - 1)) & (~b)
  return (a + 15) & ~15;  // & ~15: set last four bits of (a+15) to be zero
}

template <>
__aicore__ inline uint32_t AlignUp<32, uint32_t>(uint32_t a) {
  // refer to the above comments.
  return (a + 31) & ~31;  // & ~31: set last five bits of (a+31) to be zero}
}

template <typename T>
__aicore__ inline __gm__ T* GetTensorAddr(uint16_t index, GM_ADDR tensorPtr) {
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;  // The offset of the data address from the first address.
    // Moving 3 bits to the right means dividing by sizeof(uint64 t).
    __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(retPtr + index));
}

__aicore__ inline int32_t GetSplitValueFromGroupList(uint32_t groupIdx, int32_t &preOffset,
                                                     const GMMBaseParams* __restrict &gmmBaseParams,
                                                     const GlobalTensor<int64_t> &groupListGm) {
    int32_t splitValue = 0;
    if (likely(gmmBaseParams->groupType != -1)) {  // -1: no  need to split
        if (gmmBaseParams->groupListType == 0) {
            int32_t offset = static_cast<int32_t>(groupListGm.GetValue(groupIdx));
            splitValue = offset - preOffset;
            preOffset = offset;
        } else {
            splitValue = static_cast<int32_t>(groupListGm.GetValue(groupIdx));
        }
    }
    return splitValue;
}

template <typename T>
__aicore__ inline constexpr uint32_t GetTypeBits() {
    if constexpr (IsSameType<T, int4b_t>::value) {
        return 4;  // 4: int4 bits number
    }
    return sizeof(T) * INT8_BITS;
}

}  // namespace GROUPED_MATMUL

#endif  // ASCENDC_GROUPED_MATMUL_UTILS_H
