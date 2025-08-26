/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file param_context_base.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_RUNTIME_UTIL_CONTEXT_BASE_H_
#define OPS_BUILT_IN_OP_PROTO_RUNTIME_UTIL_CONTEXT_BASE_H_

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ops {
using GetAttrAdditional = const uint8_t *(*)(void *, size_t);

template <typename T_IN, typename T_OUT = T_IN>
class ParamsContextBase {
public:
    ParamsContextBase() = delete;
    template <typename U = T_IN, typename std::enable_if<std::is_same<T_OUT, U>::value, int>::type = 0>
    explicit ParamsContextBase(const T_IN &inOutData, GetAttrAdditional func = nullptr)
        : inData_(&inOutData),
          outData_(const_cast<T_IN *>(&inOutData)),
          getAttr_(func)
    {
    }
    explicit ParamsContextBase(const T_IN &inData, T_OUT &outData, GetAttrAdditional func = nullptr)
        : inData_(&inData),
          outData_(&outData),
          getAttr_(func)
    {
    }

    inline int32_t GetInputFormat(size_t inputIdx) const
    {
        return inData_->GetInputFormat(inputIdx);
    }
    inline int32_t GetInputDType(size_t inputIdx) const
    {
        return inData_->GetInputDType(inputIdx);
    }
    inline size_t GetInputDimNum(size_t inputIdx) const
    {
        return inData_->GetInputDimNum(inputIdx);
    }
    inline size_t GetInputDim(size_t inputIdx, size_t dimIdx) const
    {
        return inData_->GetInputDim(inputIdx, dimIdx);
    }
    inline size_t GetInputShapeSize(size_t inputIdx) const
    {
        return inData_->GetInputShapeSize(inputIdx);
    }
    inline size_t GetOutputDimNum(size_t outputIdx) const
    {
        return outData_->GetOutputDimNum(outputIdx);
    }
    inline const uint8_t *GetInputTensorData(size_t inputIdx) const
    {
        return inData_->GetInputTensorData(inputIdx);
    }
    template <typename T>
    inline const T *GetAttr(size_t index) const
    {
        return reinterpret_cast<const T *>(inData_->GetAttr(index, getAttr_));
    }
    template <typename T>
    inline const T *GetAttrList(size_t index) const
    {
        return reinterpret_cast<const T *>(inData_->GetAttrList(index, getAttr_));
    }
    inline void SetOutputFormat(size_t outputIdx, uint32_t format) const
    {
        outData_->SetOutputFormat(outputIdx, format);
    }
    inline void SetOutputDType(size_t outputIdx, uint32_t dtype) const
    {
        outData_->SetOutputDType(outputIdx, dtype);
    }
    inline void SetOutputDimNum(size_t outputIdx, size_t dimNum) const
    {
        outData_->SetOutputDimNum(outputIdx, dimNum);
    }
    inline void SetOutputDim(size_t outputIdx, size_t dimIdx, int64_t value) const
    {
        outData_->SetOutputDim(outputIdx, dimIdx, value);
    }

    bool InputDimsEqual(size_t inputIdx1, size_t inputIdx2) const
    {
        if (GetInputDimNum(inputIdx1) != GetInputDimNum(inputIdx2)) {
            return false;
        }
        for (size_t dimIdx = 0; dimIdx < GetInputDimNum(inputIdx1); dimIdx++) {
            if (GetInputDim(inputIdx1, dimIdx) != GetInputDim(inputIdx2, dimIdx)) {
                return false;
            }
        }
        return true;
    }

    void CopyInputDimsToOutput(size_t inputIdx, size_t outputIdx) const
    {
        size_t dimNum = GetInputDimNum(inputIdx);
        SetOutputDimNum(outputIdx, dimNum);
        for (size_t dimIdx = 0; dimIdx < dimNum; dimIdx++) {
            SetOutputDim(outputIdx, dimIdx, GetInputDim(inputIdx, dimIdx));
        }
    }

private:
    const T_IN *inData_ = nullptr;
    T_OUT *outData_ = nullptr;
    GetAttrAdditional getAttr_ = nullptr;  // only for MKI
};

enum DataType : int32_t {
    DTYPE_FLOAT = 0,            // float type
    DTYPE_FLOAT16 = 1,          // fp16 type
    DTYPE_INT8 = 2,             // int8 type
    DTYPE_INT32 = 3,            // int32 type
    DTYPE_UINT8 = 4,            // uint8 type
                                // reserved
    DTYPE_INT16 = 6,            // int16 type
    DTYPE_UINT16 = 7,           // uint16 type
    DTYPE_UINT32 = 8,           // unsigned int32
    DTYPE_INT64 = 9,            // int64 type
    DTYPE_UINT64 = 10,          // unsigned int64
    DTYPE_DOUBLE = 11,          // double type
    DTYPE_BOOL = 12,            // bool type
    DTYPE_STRING = 13,          // string type
    DTYPE_DUAL_SUB_INT8 = 14,   // dual output int8 type
    DTYPE_DUAL_SUB_UINT8 = 15,  // dual output uint8 type
    DTYPE_COMPLEX64 = 16,       // complex64 type
    DTYPE_COMPLEX128 = 17,      // complex128 type
    DTYPE_QINT8 = 18,           // qint8 type
    DTYPE_QINT16 = 19,          // qint16 type
    DTYPE_QINT32 = 20,          // qint32 type
    DTYPE_QUINT8 = 21,          // quint8 type
    DTYPE_QUINT16 = 22,         // quint16 type
    DTYPE_RESOURCE = 23,        // resource type
    DTYPE_STRING_REF = 24,      // string ref type
    DTYPE_DUAL = 25,            // dual output type
    DTYPE_VARIANT = 26,         // dt_variant type
    DTYPE_BF16 = 27,            // bf16 type
    DTYPE_UNDEFINED = 28,       // Used to indicate a DataType field has not been set.
    DTYPE_INT4 = 29,            // int4 type
    DTYPE_UINT1 = 30,           // uint1 type
    DTYPE_INT2 = 31,            // int2 type
    DTYPE_UINT2 = 32,           // uint2 type
    DTYPE_COMPLEX32 = 33,       // complex32 type
    DTYPE_HIFLOAT8 = 34,        // hifloat8 type
    DTYPE_FLOAT8_E5M2 = 35,     // float8_e5m2 type
    DTYPE_FLOAT8_E4M3FN = 36,   // float8_e4m3fn type
    DTYPE_FLOAT8_E8M0 = 37,     // float8_e8m0 type
    DTYPE_FLOAT6_E3M2 = 38,     // float6_e3m2 type
    DTYPE_FLOAT6_E2M3 = 39,     // float6_e2m3 type
    DTYPE_FLOAT4_E2M1 = 40,     // float4_e2m1 type
    DTYPE_FLOAT4_E1M2 = 41,     // float4_e1m2 type
    DTYPE_MAX                   // Mark the boundaries of data types
};

const uint32_t INFER_FAILED = 0xFFFFFFFF;
const uint32_t INFER_SUCCESS = 0;
}
#endif  // OPS_BUILT_IN_OP_PROTO_RUNTIME_UTIL_CONTEXT_BASE_H_
