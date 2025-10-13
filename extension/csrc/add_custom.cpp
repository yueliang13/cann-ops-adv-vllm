/**
 * @file add_custom.cpp
 *
 * Copyright (C) 2024-2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <iostream> // 1. 包含 iostream 头文件
#include "acl/acl.h"
#include "aclnnop/aclnn_incre_flash_attention_v4.h"
#include "aclnn_sparse_paged_attention.h"
#include "aclnn_sparse_paged_fusion_attention.h"
#include "aclnn_compute_cent.h"
#include "aclnn_select_position.h"
#include "aclnn_cent_select.h"
#include <torch/torch.h>
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using variable_list = std::vector<at::Tensor>;

// #define DEBUG 

// 辅助函数：将 at::Tensor 转换为 aclTensor
aclTensor *ConvertTensorToAcl(const at::Tensor &tensor)
{
    if (!tensor.defined() || tensor.numel() == 0)
    {
        return nullptr;
    }

    // 获取张量的形状和步长
    auto sizes = tensor.sizes().vec();
    auto strides = tensor.strides().vec();

    // 确定 ACL 数据类型
    aclDataType aclType;
    if (tensor.dtype() == torch::kFloat16)
    {
        aclType = ACL_FLOAT16;
    }
    else if (tensor.dtype() == torch::kBFloat16)
    {
        aclType = ACL_BF16;
    }
    else if (tensor.dtype() == torch::kFloat32)
    {
        aclType = ACL_FLOAT;
    }
    else if (tensor.dtype() == torch::kInt8)
    {
        aclType = ACL_INT8;
    }
    else if (tensor.dtype() == torch::kInt32)
    {
        aclType = ACL_INT32;
    }
    else if (tensor.dtype() == torch::kInt64)
    {
        aclType = ACL_INT64;
    }
    else
    {
        throw std::runtime_error("Unsupported tensor dtype");
    }

    // 创建 aclTensor
    return aclCreateTensor(sizes.data(), sizes.size(), aclType, strides.data(),
                           0, aclFormat::ACL_FORMAT_ND, sizes.data(), sizes.size(),
                           tensor.data_ptr());
}

// 辅助函数：将 TensorList 转换为 aclTensorList
aclTensorList *ConvertTensorListToAcl(const at::TensorList &tensor_list)
{
    if (tensor_list.empty())
    {
        return nullptr;
    }

    std::vector<aclTensor *> acl_tensors;
    for (const auto &tensor : tensor_list)
    {
        aclTensor *acl_tensor = ConvertTensorToAcl(tensor);
        if (acl_tensor != nullptr)
        {
            acl_tensors.push_back(acl_tensor);
        }
    }

    if (acl_tensors.empty())
    {
        return nullptr;
    }

    return aclCreateTensorList(acl_tensors.data(), acl_tensors.size());
}

// 辅助函数：将 at::Tensor 转换为 aclIntArray（用于 actual_seq_lengths）
aclIntArray *ConvertToIntArray(const at::Tensor &tensor)
{
    if (!tensor.defined() || tensor.numel() == 0)
    {
        return nullptr;
    }

    // 将 tensor 转换为 CPU 并获取数据
    auto cpu_tensor = tensor.cpu();
    auto data_ptr = cpu_tensor.data_ptr<int64_t>();
    auto size = cpu_tensor.numel();

    return aclCreateIntArray(data_ptr, size);
}

// 为NPU设备注册前向实现 - 按照官方接口完整参数
// 主要实现函数
at::Tensor incre_flash_attention_v4_impl_npu(const at::Tensor &query,
                                             const std::vector<at::Tensor> &key_list_vec,
                                             const std::vector<at::Tensor> &value_list_vec,
                                             const at::Tensor &pse_shift,
                                             const at::Tensor &attention_mask,
                                             const at::Tensor &actual_seq_lengths,
                                             const at::Tensor &dequant_scale1,
                                             const at::Tensor &quant_scale1,
                                             const at::Tensor &dequant_scale2,
                                             const at::Tensor &quant_scale2,
                                             const at::Tensor &quant_offset2,
                                             const at::Tensor &antiquant_scale,
                                             const at::Tensor &antiquant_offset,
                                             const at::Tensor &blocktable,
                                             const at::Tensor &kv_padding_size,
                                             int64_t num_heads,
                                             double scale_value,
                                             const std::string &input_layout,
                                             int64_t num_key_value_heads,
                                             int64_t block_size,
                                             int64_t inner_precise)
{
    #ifdef DEBUG
    std::cout << "[LOG] Entering direct CANN API implementation" << std::endl;
    std::cout << "[LOG] Query shape: " << query.sizes() << std::endl;
    std::cout << "[LOG] Key list size: " << key_list_vec.size() << std::endl;
    std::cout << "[LOG] Value list size: " << value_list_vec.size() << std::endl;
    std::cout.flush();
    #endif
    // 1. 创建输出张量
    auto out_shape = query.sizes().vec();
    at::Tensor result = at::empty(out_shape, query.options());

    try
    {
        // 2. 转换所有张量为 ACL 格式
        #ifdef DEBUG
        std::cout << "[LOG] Converting tensors to ACL format..." << std::endl;
        #endif

        aclTensor *queryTensor = ConvertTensorToAcl(query);
        if (!queryTensor)
        {
            throw std::runtime_error("Failed to convert query tensor");
        }

        // 转换 key 和 value 列表
        at::TensorList key_tensor_list(key_list_vec);
        at::TensorList value_tensor_list(value_list_vec);

        aclTensorList *tensorKeyList = ConvertTensorListToAcl(key_tensor_list);
        aclTensorList *tensorValueList = ConvertTensorListToAcl(value_tensor_list);

        if (!tensorKeyList || !tensorValueList)
        {
            throw std::runtime_error("Failed to convert key/value tensor lists");
        }

        // 转换所有输入参数（包括可选的）
        aclTensor *pseShiftTensor = ConvertTensorToAcl(pse_shift);
        aclTensor *attenTensor = ConvertTensorToAcl(attention_mask);
        aclIntArray *actualSeqLengths = ConvertToIntArray(actual_seq_lengths);
        aclTensor *dequantScale1Tensor = ConvertTensorToAcl(dequant_scale1);
        aclTensor *quantScale1Tensor = ConvertTensorToAcl(quant_scale1);
        aclTensor *dequantScale2Tensor = ConvertTensorToAcl(dequant_scale2);
        aclTensor *quantScale2Tensor = ConvertTensorToAcl(quant_scale2);
        aclTensor *quantOffset2Tensor = ConvertTensorToAcl(quant_offset2);
        aclTensor *antiquantScaleTensor = ConvertTensorToAcl(antiquant_scale);
        aclTensor *antiquantOffsetTensor = ConvertTensorToAcl(antiquant_offset);
        aclTensor *blocktableTensor = ConvertTensorToAcl(blocktable);
        aclTensor *kvPaddingSizeTensor = ConvertTensorToAcl(kv_padding_size);
        aclTensor *outTensor = ConvertTensorToAcl(result);

        // if (!attenTensor || !actualSeqLengths || !outTensor)
        // {
        //     throw std::runtime_error("Failed to convert attention_mask, actual_seq_lengths, or output tensor");
        // }

        // 3. 准备字符串参数
        char layerOut[input_layout.length() + 1];
        strcpy(layerOut, input_layout.c_str());

        #ifdef DEBUG
        std::cout << "[LOG] All tensors converted successfully" << std::endl;
        std::cout << "[LOG] Checking converted tensors:" << std::endl;
        std::cout << "[LOG]   pseShiftTensor: " << (pseShiftTensor ? "valid" : "null") << std::endl;
        std::cout << "[LOG]   dequantScale1Tensor: " << (dequantScale1Tensor ? "valid" : "null") << std::endl;
        std::cout << "[LOG]   quantScale1Tensor: " << (quantScale1Tensor ? "valid" : "null") << std::endl;
        std::cout << "[LOG]   dequantScale2Tensor: " << (dequantScale2Tensor ? "valid" : "null") << std::endl;
        std::cout << "[LOG]   quantScale2Tensor: " << (quantScale2Tensor ? "valid" : "null") << std::endl;
        std::cout << "[LOG]   quantOffset2Tensor: " << (quantOffset2Tensor ? "valid" : "null") << std::endl;
        std::cout << "[LOG]   antiquantScaleTensor: " << (antiquantScaleTensor ? "valid" : "null") << std::endl;
        std::cout << "[LOG]   antiquantOffsetTensor: " << (antiquantOffsetTensor ? "valid" : "null") << std::endl;
        std::cout << "[LOG]   blocktableTensor: " << (blocktableTensor ? "valid" : "null") << std::endl;
        std::cout << "[LOG]   kvPaddingSizeTensor: " << (kvPaddingSizeTensor ? "valid" : "null") << std::endl;
        std::cout.flush();
        #endif

        // 4. 第一段：获取工作空间大小
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;

        #ifdef DEBUG
        std::cout << "[LOG] Calling aclnnIncreFlashAttentionV4GetWorkspaceSize..." << std::endl;
        std::cout.flush();
        #endif
        int ret = aclnnIncreFlashAttentionV4GetWorkspaceSize(
            queryTensor,           // 1. queryTensor
            tensorKeyList,         // 2. tensorKeyList
            tensorValueList,       // 3. tensorValueList
            pseShiftTensor,        // 4. pseShift (传递真实的 tensor 或 nullptr)
            attenTensor,           // 5. attenTensor
            actualSeqLengths,      // 6. actualSeqLengths
            dequantScale1Tensor,   // 7. dequantScale1 (传递真实的 tensor 或 nullptr)
            quantScale1Tensor,     // 8. quantScale1 (传递真实的 tensor 或 nullptr)
            dequantScale2Tensor,   // 9. dequantScale2 (传递真实的 tensor 或 nullptr)
            quantScale2Tensor,     // 10. quantScale2 (传递真实的 tensor 或 nullptr)
            quantOffset2Tensor,    // 11. quantOffset2 (传递真实的 tensor 或 nullptr)
            antiquantScaleTensor,  // 12. antiquantScale (传递真实的 tensor 或 nullptr)
            antiquantOffsetTensor, // 13. antiquantOffset (传递真实的 tensor 或 nullptr)
            blocktableTensor,      // 14. blocktable (传递真实的 tensor 或 nullptr)
            kvPaddingSizeTensor,   // 15. kvPaddingSize (传递真实的 tensor 或 nullptr)
            num_heads,             // 16. numHeads
            scale_value,           // 17. scaleValue
            layerOut,              // 18. layerOut
            num_key_value_heads,   // 19. numKeyValueHeads
            block_size,            // 20. blockSize
            inner_precise,         // 21. innerPrecise
            outTensor,             // 22. outTensor
            &workspaceSize,        // 23. workspaceSize (输出)
            &executor              // 24. executor (输出)
        );
        #ifdef DEBUG
        std::cout << "[LOG] GetWorkspaceSize returned: " << ret << std::endl;
        std::cout << "[LOG] Workspace size: " << workspaceSize << " bytes ("
                  << (workspaceSize / 1024.0 / 1024.0) << " MB)" << std::endl;
        std::cout.flush();
        #endif
        if (ret != 0)
        {
            #ifdef DEBUG
            std::cout << "[LOG] aclGetRecentErrMsg: " << aclGetRecentErrMsg() << std::endl;
            #endif
            throw std::runtime_error("aclnnIncreFlashAttentionV4GetWorkspaceSize failed with code: " + std::to_string(ret));
        }

        // 5. 分配工作空间
        void *workspaceAddr = nullptr;
        at::Tensor workspace_tensor;

        if (workspaceSize > 0)
        {
            #ifdef DEBUG
            std::cout << "[LOG] Allocating workspace..." << std::endl;
            #endif
            // 使用 PyTorch 的内存分配器分配 NPU 内存
            workspace_tensor = at::empty({static_cast<int64_t>(workspaceSize)},
                                         query.options().dtype(at::kByte));
            workspaceAddr = workspace_tensor.data_ptr();

            #ifdef DEBUG
            std::cout << "[LOG] Workspace allocated at: " << workspaceAddr << std::endl;
            #endif
        }
        else
        {
            #ifdef DEBUG
            std::cout << "[LOG] No workspace needed" << std::endl;
            #endif
        }

        // 6. 第二段：执行算子
        #ifdef DEBUG
        std::cout << "[LOG] Calling aclnnIncreFlashAttentionV4..." << std::endl;
        std::cout.flush();
        #endif

        // 获取当前 NPU 流
        auto stream = c10_npu::getCurrentNPUStream().stream(false);

        ret = aclnnIncreFlashAttentionV4(workspaceAddr, workspaceSize, executor, stream);

        #ifdef DEBUG
        std::cout << "[LOG] aclnnIncreFlashAttentionV4 returned: " << ret << std::endl;
        #endif

        if (ret != 0)
        {
            #ifdef DEBUG
            std::cout << "[LOG] aclGetRecentErrMsg: " << aclGetRecentErrMsg() << std::endl;
            #endif
            throw std::runtime_error("aclnnIncreFlashAttentionV4 failed with code: " + std::to_string(ret));
        }

        // 7. 清理资源
        if (queryTensor)
            aclDestroyTensor(queryTensor);
        if (tensorKeyList)
            aclDestroyTensorList(tensorKeyList);
        if (tensorValueList)
            aclDestroyTensorList(tensorValueList);
        if (pseShiftTensor)
            aclDestroyTensor(pseShiftTensor);
        if (attenTensor)
            aclDestroyTensor(attenTensor);
        if (actualSeqLengths)
            aclDestroyIntArray(actualSeqLengths);
        if (dequantScale1Tensor)
            aclDestroyTensor(dequantScale1Tensor);
        if (quantScale1Tensor)
            aclDestroyTensor(quantScale1Tensor);
        if (dequantScale2Tensor)
            aclDestroyTensor(dequantScale2Tensor);
        if (quantScale2Tensor)
            aclDestroyTensor(quantScale2Tensor);
        if (quantOffset2Tensor)
            aclDestroyTensor(quantOffset2Tensor);
        if (antiquantScaleTensor)
            aclDestroyTensor(antiquantScaleTensor);
        if (antiquantOffsetTensor)
            aclDestroyTensor(antiquantOffsetTensor);
        if (blocktableTensor)
            aclDestroyTensor(blocktableTensor);
        if (kvPaddingSizeTensor)
            aclDestroyTensor(kvPaddingSizeTensor);
        if (outTensor)
            aclDestroyTensor(outTensor);

        #ifdef DEBUG
        std::cout << "[LOG] Direct CANN API call completed successfully!" << std::endl;
        #endif

        return result;
    }
    catch (const std::exception &e)
    {
        #ifdef DEBUG
        std::cout << "[LOG] Direct CANN API call failed: " << e.what() << std::endl;
        #endif
        throw;
    }
    catch (...)
    {
        #ifdef DEBUG
        std::cout << "[LOG] Direct CANN API call failed with unknown exception" << std::endl;
        #endif
        throw;
    }
}

//理论上用这个函数会规范一些 但是关于CMD宏的调用存在BUG 暂时使用算子直接调用的方式 不用宏实现
at::Tensor incre_flash_attention_v4_impl_npu_ues_CMD(const at::Tensor &query,
                                             const std::vector<at::Tensor> &key_list_vec,
                                             const std::vector<at::Tensor> &value_list_vec,
                                             const at::Tensor &pse_shift,
                                             const at::Tensor &attention_mask,
                                             const at::Tensor &actual_seq_lengths,
                                             const at::Tensor &dequant_scale1,
                                             const at::Tensor &quant_scale1,
                                             const at::Tensor &dequant_scale2,
                                             const at::Tensor &quant_scale2,
                                             const at::Tensor &quant_offset2,
                                             const at::Tensor &antiquant_scale,
                                             const at::Tensor &antiquant_offset,
                                             const at::Tensor &blocktable,
                                             const at::Tensor &kv_padding_size,
                                             int64_t num_heads,
                                             double scale_value,
                                             const std::string &input_layout,
                                             int64_t num_key_value_heads,
                                             int64_t block_size,
                                             int64_t inner_precise)
{
    // 日志打印
    #ifdef DEBUG
    std::cout << "[LOG] Entering C++ function: incre_flash_attention_v4_impl_npu" << std::endl;
    std::cout << "[LOG]   - query.size: " << query.sizes() << std::endl;
    std::cout << "[LOG]   - key_list_vec.size: " << key_list_vec.size() << std::endl;
    std::cout << "[LOG]   - value_list_vec.size: " << value_list_vec.size() << std::endl;
    std::cout << "[LOG]   - scale_value: " << scale_value << std::endl;
    std::cout << "[LOG]   - input_layout: " << input_layout << std::endl;
    std::cout.flush();
    #endif
    // 创建输出内存
    auto out_shape = query.sizes().vec();
    at::Tensor result = at::empty(out_shape, query.options());

    // 转换字符串
    const int MAX_LAYER_NAME = 256;
    char input_layout_str[MAX_LAYER_NAME];
    strncpy(input_layout_str, input_layout.c_str(), MAX_LAYER_NAME - 1);
    input_layout_str[MAX_LAYER_NAME - 1] = '\0';

    // 显式转换为 at::TensorList
    #ifdef DEBUG
    std::cout << "[LOG] Converting std::vector to at::TensorList..." << std::endl;
    #endif

    std::vector<at::Tensor> key_vec;
    for (const auto &tensor : key_list_vec)
    {
        key_vec.push_back(tensor);
        #ifdef DEBUG
        std::cout << "[LOG] Key tensor shape: " << tensor.sizes() << std::endl;
        #endif
    }
    at::TensorList key_tensor_list(key_vec);

    std::vector<at::Tensor> value_vec;
    for (const auto &tensor : value_list_vec)
    {
        value_vec.push_back(tensor);
        #ifdef DEBUG
        std::cout << "[LOG] Value tensor shape: " << tensor.sizes() << std::endl;
        #endif
    }
    at::TensorList value_tensor_list(value_vec);

    #ifdef DEBUG
    std::cout << "[LOG] TensorList conversion completed!" << std::endl;
    std::cout.flush();
    #endif

    // --- 修复：预先处理所有可选参数为左值 ---
    at::Tensor final_pse_shift;
    at::Tensor final_attention_mask;
    at::Tensor final_actual_seq_lengths;

    if (pse_shift.numel() > 0)
    {
        final_pse_shift = pse_shift;
    }
    else
    {
        final_pse_shift = at::empty({0}, query.options());
    }

    if (attention_mask.numel() > 0)
    {
        final_attention_mask = attention_mask;
    }
    else
    {
        final_attention_mask = at::empty({0}, query.options());
    }

    if (actual_seq_lengths.numel() > 0)
    {
        final_actual_seq_lengths = actual_seq_lengths;
    }
    else
    {
        final_actual_seq_lengths = at::empty({0}, query.options().dtype(at::kLong));
    }
    // 创建所有需要的变量，避免使用右值
    std::nullptr_t null_ptr = nullptr;

    // 为所有空的tensor创建实际的空tensor对象
    at::Tensor empty_tensor = at::empty({0}, query.options());
    #ifdef DEBUG
    std::cout << "[LOG] About to call EXEC_NPU_CMD..." << std::endl;
    std::cout << "[LOG] final_pse_shift.numel(): " << final_pse_shift.numel() << std::endl;
    std::cout << "[LOG] final_attention_mask.numel(): " << final_attention_mask.numel() << std::endl;
    std::cout << "[LOG] final_actual_seq_lengths.numel(): " << final_actual_seq_lengths.numel() << std::endl;
    std::cout.flush();
    #endif
    try
    {
        // 按照官方示例的精确顺序
        EXEC_NPU_CMD(aclnnIncreFlashAttentionV4,
                     query,               // 1. queryTensor
                     key_tensor_list,     // 2. tensorKeyList
                     value_tensor_list,   // 3. tensorValueList
                     null_ptr,            // 4. pseShift (nullptr)
                     attention_mask,      // 5. attenTensor
                     actual_seq_lengths,  // 6. actualSeqLengths
                     null_ptr,            // 7. dequantScale1 (nullptr)
                     null_ptr,            // 8. quantScale1 (nullptr)
                     null_ptr,            // 9. dequantScale2 (nullptr)
                     null_ptr,            // 10. quantScale2 (nullptr)
                     null_ptr,            // 11. quantOffset2 (nullptr)
                     null_ptr,            // 12. antiquantScale (nullptr)
                     null_ptr,            // 13. antiquantOffset (nullptr)
                     null_ptr,            // 14. blocktable (nullptr)
                     null_ptr,            // 15. kvPaddingSize (nullptr)
                     num_heads,           // 16. numHeads
                     scale_value,         // 17. scaleValue
                     input_layout_str,    // 18. layerOut
                     num_key_value_heads, // 19. numKeyValueHeads
                     block_size,          // 20. blockSize
                     inner_precise,       // 21. innerPrecise
                     result               // 22. outTensor
                                          // 23-24. workspace_size_addr, executor_addr 由宏自动添加
        );
        #ifdef DEBUG
        std::cout << "[LOG] EXEC_NPU_CMD completed successfully!" << std::endl;
        #endif
    }
    catch (const std::exception &e)
    {
        #ifdef DEBUG
        std::cout << "[LOG] EXEC_NPU_CMD failed with exception: " << e.what() << std::endl;
        #endif
        throw;
    }
    catch (...)
    {
        #ifdef DEBUG
        std::cout << "[LOG] EXEC_NPU_CMD failed with unknown exception" << std::endl;
        #endif
        throw;
    }

    return result;
}

// 为NPU设备注册前向实现 - 按照官方接口完整参数
// 主要实现函数
at::Tensor sparse_paged_attention_impl_npu(const at::Tensor &query, const std::vector<at::Tensor> &key_list_vec,
                                             const std::vector<at::Tensor> &value_list_vec, const at::Tensor &pse_shift,
                                             const at::Tensor &attention_mask, const at::Tensor &actual_seq_lengths,
                                             const at::Tensor &dequant_scale1, const at::Tensor &quant_scale1,
                                             const at::Tensor &dequant_scale2, const at::Tensor &quant_scale2,
                                             const at::Tensor &quant_offset2, const at::Tensor &antiquant_scale,
                                             const at::Tensor &antiquant_offset, const at::Tensor &blocktable,
                                             const at::Tensor &kv_padding_size, const at::Tensor &blockposition,
                                             int64_t num_heads, double scale_value, const std::string &input_layout,
                                             int64_t num_key_value_heads, int64_t block_size, int64_t inner_precise)
{
    #ifdef DEBUG
    std::cout << "[LOG] Entering direct CANN API implementation" << std::endl;
    std::cout << "[LOG] Query shape: " << query.sizes() << std::endl;
    std::cout << "[LOG] Key list size: " << key_list_vec.size() << std::endl;
    std::cout << "[LOG] Value list size: " << value_list_vec.size() << std::endl;
    std::cout.flush();
    #endif  
    // 1. 创建输出张量
    auto out_shape = query.sizes().vec();
    at::Tensor result = at::empty(out_shape, query.options());

    try {
        // 2. 转换所有张量为 ACL 格式
        #ifdef DEBUG
        std::cout << "[LOG] Converting tensors to ACL format..." << std::endl;
        #endif

        aclTensor *queryTensor = ConvertTensorToAcl(query);
        if (!queryTensor) {
            throw std::runtime_error("Failed to convert query tensor");
        }

        // 转换 key 和 value 列表
        at::TensorList key_tensor_list(key_list_vec);
        at::TensorList value_tensor_list(value_list_vec);

        aclTensorList *tensorKeyList = ConvertTensorListToAcl(key_tensor_list);
        aclTensorList *tensorValueList = ConvertTensorListToAcl(value_tensor_list);

        if (!tensorKeyList || !tensorValueList) {
            throw std::runtime_error("Failed to convert key/value tensor lists");
        }
        
        // 转换所有输入参数（包括可选的）
        aclTensor *pseShiftTensor = ConvertTensorToAcl(pse_shift);
        aclTensor *attenTensor = ConvertTensorToAcl(attention_mask);
        aclIntArray *actualSeqLengths = ConvertToIntArray(actual_seq_lengths);
        aclTensor *dequantScale1Tensor = ConvertTensorToAcl(dequant_scale1);
        aclTensor *quantScale1Tensor = ConvertTensorToAcl(quant_scale1);
        aclTensor *dequantScale2Tensor = ConvertTensorToAcl(dequant_scale2);
        aclTensor *quantScale2Tensor = ConvertTensorToAcl(quant_scale2);
        aclTensor *quantOffset2Tensor = ConvertTensorToAcl(quant_offset2);
        aclTensor *antiquantScaleTensor = ConvertTensorToAcl(antiquant_scale);
        aclTensor *antiquantOffsetTensor = ConvertTensorToAcl(antiquant_offset);
        aclTensor *blocktableTensor = ConvertTensorToAcl(blocktable);
        aclTensor *kvPaddingSizeTensor = ConvertTensorToAcl(kv_padding_size);
        aclTensor *blockpositionTensor = ConvertTensorToAcl(blockposition);
        aclTensor *outTensor = ConvertTensorToAcl(result);

        // if (!attenTensor || !actualSeqLengths || !outTensor) {
        //     throw std::runtime_error("Failed to convert attention_mask, actual_seq_lengths, or output tensor");
        // }

        // 3. 准备字符串参数
        char layerOut[input_layout.length() + 1];
        strcpy(layerOut, input_layout.c_str());

        #ifdef DEBUG
        std::cout << "[LOG] All tensors converted successfully" << std::endl;
        std::cout << "[LOG] Checking converted tensors:" << std::endl;
        std::cout << "[LOG]   pseShiftTensor: " << (pseShiftTensor ? "valid" : "null") << std::endl;
        std::cout << "[LOG]   dequantScale1Tensor: " << (dequantScale1Tensor ? "valid" : "null") << std::endl;
        std::cout << "[LOG]   quantScale1Tensor: " << (quantScale1Tensor ? "valid" : "null") << std::endl;
        std::cout << "[LOG]   dequantScale2Tensor: " << (dequantScale2Tensor ? "valid" : "null") << std::endl;
        std::cout << "[LOG]   quantScale2Tensor: " << (quantScale2Tensor ? "valid" : "null") << std::endl;
        std::cout << "[LOG]   quantOffset2Tensor: " << (quantOffset2Tensor ? "valid" : "null") << std::endl;
        std::cout << "[LOG]   antiquantScaleTensor: " << (antiquantScaleTensor ? "valid" : "null") << std::endl;
        std::cout << "[LOG]   antiquantOffsetTensor: " << (antiquantOffsetTensor ? "valid" : "null") << std::endl;
        std::cout << "[LOG]   blocktableTensor: " << (blocktableTensor ? "valid" : "null") << std::endl;
        std::cout << "[LOG]   kvPaddingSizeTensor: " << (kvPaddingSizeTensor ? "valid" : "null") << std::endl;
        std::cout << "[LOG]   blockpositionTensor: " << (blockpositionTensor ? "valid" : "null") << std::endl;
        std::cout.flush();
        #endif
        // 4. 第一段：获取工作空间大小
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;

        #ifdef DEBUG
        std::cout << "[LOG] Calling aclnnSparsePagedAttentionGetWorkspaceSize..." << std::endl;
        std::cout.flush();
        #endif

        int ret = aclnnSparsePagedAttentionGetWorkspaceSize(
            queryTensor,           // 1. queryTensor
            tensorKeyList,         // 2. tensorKeyList
            tensorValueList,       // 3. tensorValueList
            pseShiftTensor,        // 4. pseShift (传递真实的 tensor 或 nullptr)
            attenTensor,           // 5. attenTensor
            actualSeqLengths,      // 6. actualSeqLengths
            dequantScale1Tensor,   // 7. dequantScale1 (传递真实的 tensor 或 nullptr)
            quantScale1Tensor,     // 8. quantScale1 (传递真实的 tensor 或 nullptr)
            dequantScale2Tensor,   // 9. dequantScale2 (传递真实的 tensor 或 nullptr)
            quantScale2Tensor,     // 10. quantScale2 (传递真实的 tensor 或 nullptr)
            quantOffset2Tensor,    // 11. quantOffset2 (传递真实的 tensor 或 nullptr)
            antiquantScaleTensor,  // 12. antiquantScale (传递真实的 tensor 或 nullptr)
            antiquantOffsetTensor, // 13. antiquantOffset (传递真实的 tensor 或 nullptr)
            blocktableTensor,      // 14. blocktable (传递真实的 tensor 或 nullptr)
            kvPaddingSizeTensor,   // 15. kvPaddingSize (传递真实的 tensor 或 nullptr)
            blockpositionTensor,   // blockPosition - 新增的三维映射表
            num_heads,             // 16. numHeads
            scale_value,           // 17. scaleValue
            layerOut,              // 18. layerOut
            num_key_value_heads,   // 19. numKeyValueHeads
            block_size,            // 20. blockSize
            inner_precise,         // 21. innerPrecise
            outTensor,             // 22. outTensor
            &workspaceSize,        // 23. workspaceSize (输出)
            &executor              // 24. executor (输出)
        );

        #ifdef DEBUG
        std::cout << "[LOG] GetWorkspaceSize returned: " << ret << std::endl;
        std::cout << "[LOG] Workspace size: " << workspaceSize << " bytes (" << (workspaceSize / 1024.0 / 1024.0)
                  << " MB)" << std::endl;
        std::cout.flush();
        #endif
        if (ret != 0) {
            std::cout << "[LOG] aclGetRecentErrMsg: " << aclGetRecentErrMsg() << std::endl;
            throw std::runtime_error("aclnnIncreFlashAttentionV4GetWorkspaceSize failed with code: " +
                                     std::to_string(ret));
        }

        // 5. 分配工作空间
        void *workspaceAddr = nullptr;
        at::Tensor workspace_tensor;

        if (workspaceSize > 0) {
            #ifdef DEBUG
            std::cout << "[LOG] Allocating workspace..." << std::endl;
            #endif

            // 使用 PyTorch 的内存分配器分配 NPU 内存
            workspace_tensor = at::empty({static_cast<int64_t>(workspaceSize)}, query.options().dtype(at::kByte));
            workspaceAddr = workspace_tensor.data_ptr();

            #ifdef DEBUG
            std::cout << "[LOG] Workspace allocated at: " << workspaceAddr << std::endl;
            #endif
        } else {
            #ifdef DEBUG
            std::cout << "[LOG] No workspace needed" << std::endl;
            #endif
        }

        // 6. 第二段：执行算子
        #ifdef DEBUG
        std::cout << "[LOG] Calling aclnnIncreFlashAttentionV5..." << std::endl;
        std::cout.flush();
        #endif

        // 获取当前 NPU 流
        auto stream = c10_npu::getCurrentNPUStream().stream(false);

        ret = aclnnSparsePagedAttention(workspaceAddr, workspaceSize, executor, stream);

        #ifdef DEBUG
        std::cout << "[LOG] aclnnSparsePagedAttention returned: " << ret << std::endl;
        #endif

        if (ret != 0) {
            std::cout << "[LOG] aclGetRecentErrMsg: " << aclGetRecentErrMsg() << std::endl;
            throw std::runtime_error("aclnnSparsePagedAttention failed with code: " + std::to_string(ret));
        }

        // 7. 清理资源
        if (queryTensor)
            aclDestroyTensor(queryTensor);
        if (tensorKeyList)
            aclDestroyTensorList(tensorKeyList);
        if (tensorValueList)
            aclDestroyTensorList(tensorValueList);
        if (pseShiftTensor)
            aclDestroyTensor(pseShiftTensor);
        if (attenTensor)
            aclDestroyTensor(attenTensor);
        if (actualSeqLengths)
            aclDestroyIntArray(actualSeqLengths);
        if (dequantScale1Tensor)
            aclDestroyTensor(dequantScale1Tensor);
        if (quantScale1Tensor)
            aclDestroyTensor(quantScale1Tensor);
        if (dequantScale2Tensor)
            aclDestroyTensor(dequantScale2Tensor);
        if (quantScale2Tensor)
            aclDestroyTensor(quantScale2Tensor);
        if (quantOffset2Tensor)
            aclDestroyTensor(quantOffset2Tensor);
        if (antiquantScaleTensor)
            aclDestroyTensor(antiquantScaleTensor);
        if (antiquantOffsetTensor)
            aclDestroyTensor(antiquantOffsetTensor);
        if (blocktableTensor)
            aclDestroyTensor(blocktableTensor);
        if (blockpositionTensor)
            aclDestroyTensor(blockpositionTensor);
        if (kvPaddingSizeTensor)
            aclDestroyTensor(kvPaddingSizeTensor);
        if (outTensor)
            aclDestroyTensor(outTensor);

        #ifdef DEBUG
        std::cout << "[LOG] Direct CANN API call completed successfully!" << std::endl;
        #endif

        return result;
    } catch (const std::exception &e) {
        #ifdef DEBUG
        std::cout << "[LOG] Direct CANN API call failed: " << e.what() << std::endl;
        #endif
        throw;
    } catch (...) {
        #ifdef DEBUG
        std::cout << "[LOG] Direct CANN API call failed with unknown exception" << std::endl;
        #endif
        throw;
    }
}

// 融合算子：CentSelect + SparsePagedAttention
std::tuple<at::Tensor, at::Tensor, at::Tensor> sparse_paged_fusion_attention_impl_npu(
    const at::Tensor &query,
    const std::vector<at::Tensor> &key_list_vec,
    const std::vector<at::Tensor> &value_list_vec,
    const at::Tensor &pse_shift,
    const at::Tensor &attention_mask,
    const at::Tensor &actual_seq_lengths,
    const at::Tensor &dequant_scale1,
    const at::Tensor &quant_scale1,
    const at::Tensor &dequant_scale2,
    const at::Tensor &quant_scale2,
    const at::Tensor &quant_offset2,
    const at::Tensor &antiquant_scale,
    const at::Tensor &antiquant_offset,
    const at::Tensor &blocktable,
    const at::Tensor &kv_padding_size,
    const at::Tensor &l1_cent,
    const at::Tensor &block_ids,
    const at::Tensor &total_seq_len,
    const at::Tensor &block_position,
    const at::Tensor &page_position_length,
    const at::Tensor &max_page_position_length,
    int64_t num_heads,
    double scale_value,
    const std::string &input_layout,
    int64_t num_key_value_heads,
    int64_t block_size,
    int64_t inner_precise)
{
    // 输出：attentionOut（与 query 同形）
    auto out_shape = query.sizes().vec();
    at::Tensor attention_out = at::empty(out_shape, query.options());

    try {
        // 转 ACL
        aclTensor *queryTensor = ConvertTensorToAcl(query);
        at::TensorList key_tensor_list(key_list_vec);
        at::TensorList value_tensor_list(value_list_vec);
        aclTensorList *tensorKeyList = ConvertTensorListToAcl(key_tensor_list);
        aclTensorList *tensorValueList = ConvertTensorListToAcl(value_tensor_list);
        aclTensor *pseShiftTensor = ConvertTensorToAcl(pse_shift);
        aclTensor *attenTensor = ConvertTensorToAcl(attention_mask);
        aclIntArray *actualSeqLengths = ConvertToIntArray(actual_seq_lengths);
        aclTensor *dequantScale1Tensor = ConvertTensorToAcl(dequant_scale1);
        aclTensor *quantScale1Tensor = ConvertTensorToAcl(quant_scale1);
        aclTensor *dequantScale2Tensor = ConvertTensorToAcl(dequant_scale2);
        aclTensor *quantScale2Tensor = ConvertTensorToAcl(quant_scale2);
        aclTensor *quantOffset2Tensor = ConvertTensorToAcl(quant_offset2);
        aclTensor *antiquantScaleTensor = ConvertTensorToAcl(antiquant_scale);
        aclTensor *antiquantOffsetTensor = ConvertTensorToAcl(antiquant_offset);
        aclTensor *blocktableTensor = ConvertTensorToAcl(blocktable);
        aclTensor *kvPaddingSizeTensor = ConvertTensorToAcl(kv_padding_size);
        aclTensor *l1CentTensor = ConvertTensorToAcl(l1_cent);
        aclTensor *blockIdsTensor = ConvertTensorToAcl(block_ids);
        aclTensor *totalSeqLenTensor = ConvertTensorToAcl(total_seq_len);
        aclTensor *blockPositionTensor = ConvertTensorToAcl(block_position);
        aclTensor *pagePositionLengthTensor = ConvertTensorToAcl(page_position_length);
        aclTensor *maxPagePositionLengthTensor = ConvertTensorToAcl(max_page_position_length);
        aclTensor *attentionOutTensor = ConvertTensorToAcl(attention_out);

        char layerOut[input_layout.length() + 1];
        strcpy(layerOut, input_layout.c_str());

        // GetWorkspaceSize
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        int ret = aclnnSparsePagedFusionAttentionGetWorkspaceSize(
            queryTensor, tensorKeyList, tensorValueList, pseShiftTensor,
            attenTensor, actualSeqLengths, dequantScale1Tensor, quantScale1Tensor, dequantScale2Tensor, quantScale2Tensor,
            quantOffset2Tensor, antiquantScaleTensor, antiquantOffsetTensor,
            blocktableTensor, kvPaddingSizeTensor,
            l1CentTensor, blockIdsTensor, totalSeqLenTensor,
            num_heads, scale_value, layerOut, num_key_value_heads, block_size, inner_precise,
            blockPositionTensor, pagePositionLengthTensor, maxPagePositionLengthTensor,
            attentionOutTensor, &workspaceSize, &executor);
        
        // 设置executor为可复用
        aclSetAclOpExecutorRepeatable(executor);  
        // void *addr;
        // aclSetDynamicInputTensorAddr(executor, 0, 0, tensorKeyList, addr);   // 刷新输入tensorlist中第1个aclTensor的device地址
        // aclSetDynamicInputTensorAddr(executor, 0, 1, tensorValueList, addr);  // 刷新输入tensorlist中第2个aclTensor的device地址
        
        #ifdef DEBUG
        std::cout << "[LOG] GetWorkspaceSize returned: " << ret << std::endl;
        std::cout << "[LOG] Workspace size: " << workspaceSize << " bytes ("
                    << (workspaceSize / 1024.0 / 1024.0) << " MB)" << std::endl;
        std::cout.flush();
        #endif
        if (ret != 0) {
            #ifdef DEBUG
            std::cout << "[LOG] aclGetRecentErrMsg: " << aclGetRecentErrMsg() << std::endl;
            #endif
            throw std::runtime_error("aclnnSparsePagedFusionAttentionGetWorkspaceSize failed: " + std::to_string(ret));
        }

        // workspace & run
        void *workspaceAddr = nullptr;
        at::Tensor workspace_tensor;
        if (workspaceSize > 0) {
            #ifdef DEBUG
            std::cout << "[LOG] Allocating workspace..." << std::endl;
            #endif
            workspace_tensor = at::empty({static_cast<int64_t>(workspaceSize)}, query.options().dtype(at::kByte));
            workspaceAddr = workspace_tensor.data_ptr();
            #ifdef DEBUG
            std::cout << "[LOG] Workspace allocated at: " << workspaceAddr << std::endl;
            #endif
        }
        else
        {
            #ifdef DEBUG
            std::cout << "[LOG] No workspace needed" << std::endl;
            #endif
        }
        auto stream = c10_npu::getCurrentNPUStream().stream(false);

        for (int i = 0; i < 100; i++) {
            ret = aclnnSparsePagedFusionAttention(workspaceAddr, workspaceSize, executor, stream);
            // if (ret != 0) {
            //     #ifdef DEBUG
            //     std::cout << "[LOG] aclGetRecentErrMsg: " << aclGetRecentErrMsg() << std::endl;
            //     #endif
            //     throw std::runtime_error("aclnnSparsePagedFusionAttention failed: " + std::to_string(ret));
            // }
        }
        ret = aclnnSparsePagedFusionAttention(workspaceAddr, workspaceSize, executor, stream);


        if (ret != 0) { 
            #ifdef DEBUG
            std::cout << "[LOG] aclGetRecentErrMsg: " << aclGetRecentErrMsg() << std::endl;
            #endif
            throw std::runtime_error("aclnnSparsePagedFusionAttention failed: " + std::to_string(ret));
        }

        // 清理（必要对象）
        if (queryTensor) aclDestroyTensor(queryTensor);
        if (tensorKeyList) aclDestroyTensorList(tensorKeyList);
        if (tensorValueList) aclDestroyTensorList(tensorValueList);
        if (pseShiftTensor) aclDestroyTensor(pseShiftTensor);
        if (attenTensor) aclDestroyTensor(attenTensor);
        if (actualSeqLengths) aclDestroyIntArray(actualSeqLengths);
        if (dequantScale1Tensor) aclDestroyTensor(dequantScale1Tensor);
        if (quantScale1Tensor) aclDestroyTensor(quantScale1Tensor);
        if (dequantScale2Tensor) aclDestroyTensor(dequantScale2Tensor);
        if (quantScale2Tensor) aclDestroyTensor(quantScale2Tensor);
        if (quantOffset2Tensor) aclDestroyTensor(quantOffset2Tensor);
        if (antiquantScaleTensor) aclDestroyTensor(antiquantScaleTensor);
        if (antiquantOffsetTensor) aclDestroyTensor(antiquantOffsetTensor);
        if (blocktableTensor) aclDestroyTensor(blocktableTensor);
        if (kvPaddingSizeTensor) aclDestroyTensor(kvPaddingSizeTensor);
        if (l1CentTensor) aclDestroyTensor(l1CentTensor);
        if (blockIdsTensor) aclDestroyTensor(blockIdsTensor);
        if (totalSeqLenTensor) aclDestroyTensor(totalSeqLenTensor);
        if (blockPositionTensor) aclDestroyTensor(blockPositionTensor);
        if (pagePositionLengthTensor) aclDestroyTensor(pagePositionLengthTensor);
        if (maxPagePositionLengthTensor) aclDestroyTensor(maxPagePositionLengthTensor);
        if (attentionOutTensor) aclDestroyTensor(attentionOutTensor);

        aclDestroyAclOpExecutor(executor);  

        return std::make_tuple(attention_out, block_position, max_page_position_length);
    } catch (...) {
        throw;
    }
}

at::Tensor compute_cent_impl_npu(const at::Tensor &query, const at::Tensor &l1_cent)
{
    // 1. 创建输出张量
    auto k = 64;
    auto out_shape = query.sizes().vec();
    at::Tensor result = at::empty({out_shape[0], out_shape[1], k}, query.options().dtype(torch::kInt32));

    // call aclnn interface to perform the computation
    try {
        EXEC_NPU_CMD(aclnnComputeCent, query, l1_cent, result);
    } catch (const std::exception &e) {
        throw;
    } catch (...) {
        throw;
    }
    return result;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> select_position_impl_npu(const at::Tensor &block_ids, const at::Tensor &block_table, const at::Tensor &seq_len, const at::Tensor &indices)
{
    // 1. 创建输出张量
    auto max_page_num = 256;
    auto batch_size = indices.sizes()[0];
    auto q_head_num = indices.sizes()[1];
    auto max_page_len = block_table.sizes()[1];
    at::Tensor page_position = at::empty({batch_size, q_head_num, max_page_num}, block_ids.options());
    at::Tensor page_position_length = at::empty({batch_size, q_head_num, 8}, block_ids.options());
    at::Tensor block_table_gather = at::empty({batch_size, q_head_num, max_page_len}, block_ids.options());

    // call aclnn interface to perform the computation
    try {
        EXEC_NPU_CMD(aclnnSelectPosition, block_ids, block_table, seq_len, indices, page_position, page_position_length, block_table_gather);
    } catch (const std::exception &e) {        
        std::cout << "[LOG] EXEC_NPU_CMD failed with exception: " << e.what() << std::endl;
        throw;
    } catch (...) {
        std::cout << "[LOG] EXEC_NPU_CMD failed with unknown exception" << std::endl;
        throw;
    }
    return std::make_tuple(page_position, page_position_length, block_table_gather);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>  cent_select_impl_npu(const at::Tensor &query, const at::Tensor &l1_cent, const at::Tensor &block_ids, const at::Tensor &block_table, const at::Tensor &seq_len){

    // 1. 创建输出张量
    auto max_page_num = 256;
    auto batch_size = query.sizes()[0];
    auto q_head_num = query.sizes()[1];
    auto max_page_len = block_table.sizes()[1];
    at::Tensor page_position = at::empty({batch_size, q_head_num, max_page_num}, block_ids.options());
    at::Tensor page_position_length = at::empty({batch_size, q_head_num, 8}, block_ids.options());
    at::Tensor max_page_position_length = at::empty({batch_size, 8}, block_ids.options().dtype(torch::kInt64));
    at::Tensor importance = at::empty({batch_size, q_head_num}, query.options().dtype(torch::kFloat32));
    // call aclnn interface to perform the computation
    try {
        EXEC_NPU_CMD(aclnnCentSelect, query, l1_cent, block_ids, block_table, seq_len, 
        importance, page_position, page_position_length, max_page_position_length);
    } catch (const std::exception &e) {
        std::cout << "[LOG] EXEC_NPU_CMD failed with exception: " << e.what() << std::endl;
        throw;
    } catch (...) {
        std::cout << "[LOG] EXEC_NPU_CMD failed with unknown exception" << std::endl;
        throw;
    }
    return std::make_tuple(page_position, page_position_length, max_page_position_length);
}

// 为NPU设备注册前反向实现
// NPU设备在pytorch 2.1及以上版本使用的设备名称是PrivateUse1，在2.1以下版本用的是XLA，如果是2.1以下版本PrivateUse1需要改成XLA
// 为NPU设备注册前反向实现
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m)
{
    m.impl("incre_flash_attention_v4", &incre_flash_attention_v4_impl_npu);
    m.impl("sparse_paged_attention", &sparse_paged_attention_impl_npu);
    m.impl("compute_cent", &compute_cent_impl_npu);
    m.impl("select_position", &select_position_impl_npu);
    m.impl("cent_select", &cent_select_impl_npu);
    m.impl("sparse_paged_fusion_attention", &sparse_paged_fusion_attention_impl_npu);
}

// // 给op绑定NPU的自动求导实现
// TORCH_LIBRARY_IMPL(myops, AutogradPrivateUse1, m)
// {
//     m.impl("incre_flash_attention_v4", &incre_flash_attention_v4_autograd);
// }
