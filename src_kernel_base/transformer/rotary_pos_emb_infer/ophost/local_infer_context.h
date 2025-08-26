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
 * \file local_infer_context.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_UTIL_LOCAL_INFER_CONTEXT_H_
#define OPS_BUILT_IN_OP_PROTO_UTIL_LOCAL_INFER_CONTEXT_H_

#include <exception>
#include "param_context_base.h"
#include "register/op_def_registry.h"

#define OP_LOGD(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")
#define OP_LOGI(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")
#define OP_LOGW(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")
#define OP_LOGE(op_name, ...) std::printf(op_name, ##__VA_ARGS__)

#define INFER_LOG_DEBUG(op, format, ...) OP_LOGD(op, format, ##__VA_ARGS__)
#define INFER_LOG_INFO(op, format, ...) OP_LOGI(op, format, ##__VA_ARGS__)
#define INFER_LOG_WARN(op, format, ...) OP_LOGW(op, format, ##__VA_ARGS__)
#define INFER_LOG_ERROR(op, format, ...) OP_LOGE(op, format, ##__VA_ARGS__)

#define REG_OP_INFERSHAPE(opType, infershapeFunc, inferDataTypeFunc)                                           \
    static ge::graphStatus Reg##opType##InferShape(gert::InferShapeContext *context)                           \
    {                                                                                                          \
        ops::LocalInferShapeParams params(reinterpret_cast<const ops::GeInferShapeContext &>(*context));       \
        return (infershapeFunc(params) == ops::INFER_SUCCESS) ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;          \
    }                                                                                                          \
                                                                                                               \
    static ge::graphStatus Reg##opType##InferDataType(gert::InferDataTypeContext *context)                     \
    {                                                                                                          \
        ops::LocalInferDataTypeParams params(reinterpret_cast<const ops::GeInferDataTypeContext &>(*context)); \
        return (inferDataTypeFunc(params) == ops::INFER_SUCCESS) ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;       \
    }                                                                                                          \
                                                                                                               \
    IMPL_OP_INFERSHAPE(opType).InferShape(Reg##opType##InferShape).InferDataType(Reg##opType##InferDataType)

class NullPtrException : public std::exception {
public:
    explicit NullPtrException(const char *message = "Null pointer exception occurred") : msg_(message) {}
    const char *what() const noexcept override
    {
        return msg_;
    }

private:
    const char *msg_;
};

namespace ops {
struct GeInferShapeContext {
    gert::InferShapeContext infos;

    inline uint32_t GetInputFormat(size_t inputIdx) const
    {
        const auto *tensor = GetInputTensor(inputIdx);
        if (tensor == nullptr) {
            throw NullPtrException("tensor is nullptr");
        }
        return tensor->GetFormat().GetStorageFormat();
    }

    inline uint32_t GetInputDType(size_t inputIdx) const
    {
        const auto *tensor = GetInputTensor(inputIdx);
        if (tensor == nullptr) {
            throw NullPtrException("tensor is nullptr");
        }
        return tensor->GetDataType();
    }

    inline size_t GetInputDimNum(size_t inputIdx) const
    {
        const auto *shape = GetInputShape(inputIdx);
        if (shape == nullptr) {
            throw NullPtrException("shape is nullptr");
        }
        return shape->GetDimNum();
    }

    inline size_t GetInputDim(size_t inputIdx, size_t dimIdx) const
    {
        const auto *shape = GetInputShape(inputIdx);
        if (shape == nullptr) {
            throw NullPtrException("shape is nullptr");
        }
        return shape->GetDim(dimIdx);
    }

    inline size_t GetInputShapeSize(size_t inputIdx) const
    {
        const auto *tensor = GetInputTensor(inputIdx);
        if (tensor == nullptr) {
            throw NullPtrException("tensor is nullptr");
        }
        return tensor->GetShapeSize();
    }

    inline const uint8_t *GetInputTensorData(size_t inputIdx) const
    {
        const auto *tensor = GetInputTensor(inputIdx);
        if (tensor == nullptr) {
            throw NullPtrException("tensor is nullptr");
        }
        return reinterpret_cast<const uint8_t *>(tensor->GetData<uint8_t>());
    }

    inline const uint8_t *GetAttr(size_t index, GetAttrAdditional getAttrOffset) const
    {
        (void)getAttrOffset;
        const auto *attrs = infos.GetAttrs();
        if (attrs == nullptr) {
            throw NullPtrException("attrs is nullptr");
        }
        return attrs->GetAttrPointer<uint8_t>(index);
    }

    inline const uint8_t *GetAttrList(size_t index, GetAttrAdditional getAttrOffset) const
    {
        (void)getAttrOffset;
        const auto *attrs = infos.GetAttrs();
        if (attrs == nullptr) {
            throw NullPtrException("attrs is nullptr");
        }
        const auto *attrList = attrs->GetAttrPointer<gert::TypedContinuousVector<uint8_t>>(index);
        if (attrList == nullptr) {
            throw NullPtrException("attr list is nullptr");
        }
        return attrList->GetData();
    }

    inline void SetOutputDimNum(size_t outputIdx, size_t dimNum)
    {
        auto *shape = GetOutputShape(outputIdx);
        if (shape == nullptr) {
            throw NullPtrException("shape is nullptr");
        }
        shape->SetDimNum(dimNum);
    }

    inline size_t GetOutputDimNum(size_t outputIdx)
    {
        auto *shape = GetOutputShape(outputIdx);
        if (shape == nullptr) {
            throw NullPtrException("shape is nullptr");
        }
        return shape->GetDimNum();
    }

    inline void SetOutputFormat(size_t outputIdx, uint32_t format)
    {
        (void)outputIdx;
        (void)format;
    }

    inline void SetOutputDType(size_t outputIdx, uint32_t dtype)
    {
        (void)outputIdx;
        (void)dtype;
    }

    inline void SetOutputDim(size_t outputIdx, size_t dimIdx, int64_t value)
    {
        auto *shape = GetOutputShape(outputIdx);
        if (shape == nullptr) {
            throw NullPtrException("shape is nullptr");
        }
        shape->SetDim(dimIdx, value);
    }

private:
    inline const gert::Tensor *GetInputTensor(size_t inputIdx) const
    {
        return infos.GetInputTensor(inputIdx);
    }

    inline const gert::Shape *GetInputShape(size_t inputIdx) const
    {
        return infos.GetInputShape(inputIdx);
    }

    inline gert::Shape *GetOutputShape(size_t outputIdx)
    {
        return infos.GetOutputShape(outputIdx);
    }
};

struct GeInferDataTypeContext {
    gert::InferDataTypeContext infos;

    inline uint32_t GetInputFormat(size_t inputIdx) const
    {
        (void)inputIdx;
        return 0;
    }

    inline uint32_t GetInputDType(size_t inputIdx) const
    {
        return infos.GetInputDataType(inputIdx);
    }

    inline size_t GetInputDimNum(size_t inputIdx) const
    {
        (void)inputIdx;
        return 0;
    }

    inline size_t GetInputDim(size_t inputIdx, size_t dimIdx) const
    {
        (void)inputIdx;
        (void)dimIdx;
        return 0;
    }

    inline size_t GetInputShapeSize(size_t inputIdx) const
    {
        (void)inputIdx;
        return 0;
    }

    inline const uint8_t *GetAttrList(size_t index, GetAttrAdditional getAttrOffset) const
    {
        (void)getAttrOffset;
        const auto *attrs = infos.GetAttrs();
        if (attrs == nullptr) {
            throw NullPtrException("attrs is nullptr");
        }
        const auto *attrList = attrs->GetAttrPointer<gert::TypedContinuousVector<uint8_t>>(index);
        if (attrList == nullptr) {
            throw NullPtrException("attr list is nullptr");
        }
        return attrList->GetData();
    }

    inline const uint8_t *GetAttr(size_t index, GetAttrAdditional getAttrOffset) const
    {
        (void)getAttrOffset;
        const auto *attrs = infos.GetAttrs();
        if (attrs == nullptr) {
            throw NullPtrException("attrs is nullptr");
        }
        return attrs->GetAttrPointer<uint8_t>(index);
    }

    inline void SetOutputDimNum(size_t outputIdx, size_t dimNum)
    {
        (void)outputIdx;
        (void)dimNum;
    }

    inline size_t GetOutputDimNum(size_t outputIdx) const
    {
        (void)outputIdx;
        return 0;
    }

    inline void SetOutputFormat(size_t outputIdx, uint32_t format)
    {
        (void)outputIdx;
        (void)format;
    }

    inline void SetOutputDType(size_t outputIdx, uint32_t dtype)
    {
        if (infos.SetOutputDataType(outputIdx, static_cast<ge::DataType>(dtype)) != ge::GRAPH_SUCCESS) {
            throw std::out_of_range("set output datatype out of range");
        }
    }

    inline void SetOutputDim(size_t outputIdx, size_t dimIdx, int64_t value)
    {
        (void)outputIdx;
        (void)dimIdx;
        (void)value;
    }
};

using LocalInferShapeParams = ParamsContextBase<GeInferShapeContext>;
using LocalInferDataTypeParams = ParamsContextBase<GeInferDataTypeContext>;
}

#endif  // OPS_BUILT_IN_OP_PROTO_UTIL_LOCAL_INFER_CONTEXT_H_
