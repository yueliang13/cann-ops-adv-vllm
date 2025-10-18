// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/framework/FormatHelper.h"
#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/SparseOpsInterface.h"
#include "op_plugin/OpInterface.h"

namespace op_plugin {
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> npu_hans_encode_out(const at::Tensor & input, bool statistic, bool reshuff, at::Tensor & pdf, at::Tensor & mantissa, at::Tensor & fixed, at::Tensor & var){
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool pdf_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(pdf);
    bool mantissa_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mantissa);
    bool fixed_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(fixed);
    bool var_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(var);

    if (!input_base_format || !pdf_base_format || !mantissa_base_format || !fixed_base_format || !var_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_hans_encode_out do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_hans_encode_out(input, statistic, reshuff, pdf, mantissa, fixed, var);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> _linalg_svd_out(const at::Tensor & A, bool full_matrices, bool compute_uv, c10::optional<c10::string_view> driver, at::Tensor & U, at::Tensor & S, at::Tensor & Vh){
    return acl_op::_linalg_svd_out(A, full_matrices, compute_uv, driver, U, S, Vh);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> native_batch_norm_out(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double momentum, double eps, at::Tensor & out, at::Tensor & save_mean, at::Tensor & save_invstd){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias);
    bool running_mean_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(running_mean);
    bool running_var_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(running_var);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);
    bool save_mean_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(save_mean);
    bool save_invstd_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(save_invstd);

    ASCEND_LOGI("native_batch_norm_out exec with jit compile: %d, input is internal format: %d, weight is internal format: %d, bias is internal format: %d, running_mean is internal format: %d, running_var is internal format: %d, out is internal format: %d, save_mean is internal format: %d, save_invstd is internal format: %d",
                !is_jit_disable, !input_base_format, !weight_base_format, !bias_base_format, !running_mean_base_format, !running_var_base_format, !out_base_format, !save_mean_base_format, !save_invstd_base_format);
    if (is_jit_disable && input_base_format && weight_base_format && bias_base_format && running_mean_base_format && running_var_base_format && out_base_format && save_mean_base_format && save_invstd_base_format) {
        return op_api::native_batch_norm_out(input, weight, bias, running_mean, running_var, training, momentum, eps, out, save_mean, save_invstd);
    } else {
        return acl_op::native_batch_norm_out(input, weight, bias, running_mean, running_var, training, momentum, eps, out, save_mean, save_invstd);
    }
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> npu_apply_adam_out(const at::Scalar & beta1_power, const at::Scalar & beta2_power, const at::Scalar & lr, const at::Scalar & beta1, const at::Scalar & beta2, const at::Scalar & epsilon, const at::Tensor & grad, c10::optional<bool> use_locking, c10::optional<bool> use_nesterov, at::Tensor & var, at::Tensor & m, at::Tensor & v){
    return acl_op::npu_apply_adam_out(beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, use_locking, use_nesterov, var, m, v);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> npu_apply_adam_w_out(const at::Scalar & beta1_power, const at::Scalar & beta2_power, const at::Scalar & lr, const at::Scalar & weight_decay, const at::Scalar & beta1, const at::Scalar & beta2, const at::Scalar & epsilon, const at::Tensor & grad, const c10::optional<at::Tensor> & max_grad_norm, c10::optional<bool> amsgrad, c10::optional<bool> maximize, at::Tensor & var, at::Tensor & m, at::Tensor & v){
    return acl_op::npu_apply_adam_w_out(beta1_power, beta2_power, lr, weight_decay, beta1, beta2, epsilon, grad, max_grad_norm, amsgrad, maximize, var, m, v);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> npu_bert_apply_adam_out(const at::Scalar & lr, const at::Scalar & beta1, const at::Scalar & beta2, const at::Scalar & epsilon, const at::Tensor & grad, const at::Scalar & max_grad_norm, const at::Scalar & global_grad_norm, const at::Scalar & weight_decay, const c10::optional<at::Scalar> & step_size, int64_t adam_mode, at::Tensor & var, at::Tensor & m, at::Tensor & v){
    return acl_op::npu_bert_apply_adam_out(lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay, step_size, adam_mode, var, m, v);
}
::std::tuple<at::Tensor &,at::Tensor &> adaptive_max_pool2d_out(const at::Tensor & self, at::IntArrayRef output_size, at::Tensor & out, at::Tensor & indices){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);

    ASCEND_LOGI("adaptive_max_pool2d_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d, indices is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format, !indices_base_format);
    if (is_jit_disable && self_base_format && out_base_format && indices_base_format) {
        return op_api::adaptive_max_pool2d_out(self, output_size, out, indices);
    } else {
        return acl_op::adaptive_max_pool2d_out(self, output_size, out, indices);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> adaptive_max_pool3d_out(const at::Tensor & self, at::IntArrayRef output_size, at::Tensor & out, at::Tensor & indices){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);

    if (!self_base_format || !out_base_format || !indices_base_format) {
        TORCH_CHECK(false,
            "Current operator adaptive_max_pool3d_out do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::adaptive_max_pool3d_out(self, output_size, out, indices);
}
::std::tuple<at::Tensor &,at::Tensor &> aminmax_out(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim, at::Tensor & min, at::Tensor & max){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool min_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(min);
    bool max_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(max);

    ASCEND_LOGI("aminmax_out exec with jit compile: %d, self is internal format: %d, min is internal format: %d, max is internal format: %d",
                !is_jit_disable, !self_base_format, !min_base_format, !max_base_format);
    if (is_jit_disable && self_base_format && min_base_format && max_base_format) {
        return op_api::aminmax_out(self, dim, keepdim, min, max);
    } else {
        return acl_op::aminmax_out(self, dim, keepdim, min, max);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> kthvalue_out(const at::Tensor & self, int64_t k, at::Dimname dim, bool keepdim, at::Tensor & values, at::Tensor & indices){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool values_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(values);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);

    ASCEND_LOGI("kthvalue_out exec with jit compile: %d, self is internal format: %d, values is internal format: %d, indices is internal format: %d",
                !is_jit_disable, !self_base_format, !values_base_format, !indices_base_format);
    if (is_jit_disable && self_base_format && values_base_format && indices_base_format) {
        return op_api::kthvalue_out(self, k, dim, keepdim, values, indices);
    } else {
        return acl_op::kthvalue_out(self, k, dim, keepdim, values, indices);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> kthvalue_out(const at::Tensor & self, int64_t k, int64_t dim, bool keepdim, at::Tensor & values, at::Tensor & indices){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool values_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(values);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);

    ASCEND_LOGI("kthvalue_out exec with jit compile: %d, self is internal format: %d, values is internal format: %d, indices is internal format: %d",
                !is_jit_disable, !self_base_format, !values_base_format, !indices_base_format);
    if (is_jit_disable && self_base_format && values_base_format && indices_base_format) {
        return op_api::kthvalue_out(self, k, dim, keepdim, values, indices);
    } else {
        return acl_op::kthvalue_out(self, k, dim, keepdim, values, indices);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> linalg_qr_out(const at::Tensor & self, c10::string_view mode, at::Tensor & Q, at::Tensor & R){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool Q_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(Q);
    bool R_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(R);

    ASCEND_LOGI("linalg_qr_out exec with jit compile: %d, self is internal format: %d, Q is internal format: %d, R is internal format: %d",
                !is_jit_disable, !self_base_format, !Q_base_format, !R_base_format);
    if (is_jit_disable && self_base_format && Q_base_format && R_base_format) {
        return op_api::linalg_qr_out(self, mode, Q, R);
    } else {
        return acl_op::linalg_qr_out(self, mode, Q, R);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> log_sigmoid_forward_out(const at::Tensor & self, at::Tensor & output, at::Tensor & buffer){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(output);
    bool buffer_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(buffer);

    ASCEND_LOGI("log_sigmoid_forward_out exec with jit compile: %d, self is internal format: %d, output is internal format: %d, buffer is internal format: %d",
                !is_jit_disable, !self_base_format, !output_base_format, !buffer_base_format);
    if (is_jit_disable && self_base_format && output_base_format && buffer_base_format) {
        return op_api::log_sigmoid_forward_out(self, output, buffer);
    } else {
        return acl_op::log_sigmoid_forward_out(self, output, buffer);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> max_out(const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & max, at::Tensor & max_values){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool max_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(max);
    bool max_values_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(max_values);

    ASCEND_LOGI("max_out exec with jit compile: %d, self is internal format: %d, max is internal format: %d, max_values is internal format: %d",
                !is_jit_disable, !self_base_format, !max_base_format, !max_values_base_format);
    if (is_jit_disable && self_base_format && max_base_format && max_values_base_format) {
        return op_api::max_out(self, dim, keepdim, max, max_values);
    } else {
        return acl_op::max_out(self, dim, keepdim, max, max_values);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> max_out(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & max, at::Tensor & max_values){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool max_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(max);
    bool max_values_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(max_values);

    ASCEND_LOGI("max_out exec with jit compile: %d, self is internal format: %d, max is internal format: %d, max_values is internal format: %d",
                !is_jit_disable, !self_base_format, !max_base_format, !max_values_base_format);
    if (is_jit_disable && self_base_format && max_base_format && max_values_base_format) {
        return op_api::max_out(self, dim, keepdim, max, max_values);
    } else {
        return acl_op::max_out(self, dim, keepdim, max, max_values);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> max_pool2d_with_indices_out(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out, at::Tensor & indices){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);

    ASCEND_LOGI("max_pool2d_with_indices_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d, indices is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format, !indices_base_format);
    if (is_jit_disable && self_base_format && out_base_format && indices_base_format) {
        return op_api::max_pool2d_with_indices_out(self, kernel_size, stride, padding, dilation, ceil_mode, out, indices);
    } else {
        return acl_op::max_pool2d_with_indices_out(self, kernel_size, stride, padding, dilation, ceil_mode, out, indices);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> max_pool3d_with_indices_out(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out, at::Tensor & indices){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);

    ASCEND_LOGI("max_pool3d_with_indices_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d, indices is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format, !indices_base_format);
    if (is_jit_disable && self_base_format && out_base_format && indices_base_format) {
        return op_api::max_pool3d_with_indices_out(self, kernel_size, stride, padding, dilation, ceil_mode, out, indices);
    } else {
        return acl_op::max_pool3d_with_indices_out(self, kernel_size, stride, padding, dilation, ceil_mode, out, indices);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> median_out(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & values, at::Tensor & indices){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool values_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(values);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);

    ASCEND_LOGI("median_out exec with jit compile: %d, self is internal format: %d, values is internal format: %d, indices is internal format: %d",
                !is_jit_disable, !self_base_format, !values_base_format, !indices_base_format);
    if (is_jit_disable && self_base_format && values_base_format && indices_base_format) {
        return op_api::median_out(self, dim, keepdim, values, indices);
    } else {
        return acl_op::median_out(self, dim, keepdim, values, indices);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> min_out(const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & min, at::Tensor & min_indices){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool min_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(min);
    bool min_indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(min_indices);

    ASCEND_LOGI("min_out exec with jit compile: %d, self is internal format: %d, min is internal format: %d, min_indices is internal format: %d",
                !is_jit_disable, !self_base_format, !min_base_format, !min_indices_base_format);
    if (is_jit_disable && self_base_format && min_base_format && min_indices_base_format) {
        return op_api::min_out(self, dim, keepdim, min, min_indices);
    } else {
        return acl_op::min_out(self, dim, keepdim, min, min_indices);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> min_out(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & min, at::Tensor & min_indices){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool min_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(min);
    bool min_indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(min_indices);

    ASCEND_LOGI("min_out exec with jit compile: %d, self is internal format: %d, min is internal format: %d, min_indices is internal format: %d",
                !is_jit_disable, !self_base_format, !min_base_format, !min_indices_base_format);
    if (is_jit_disable && self_base_format && min_base_format && min_indices_base_format) {
        return op_api::min_out(self, dim, keepdim, min, min_indices);
    } else {
        return acl_op::min_out(self, dim, keepdim, min, min_indices);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> multilabel_margin_loss_forward_out(const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & output, at::Tensor & is_target){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target);
    bool output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(output);
    bool is_target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(is_target);

    ASCEND_LOGI("multilabel_margin_loss_forward_out exec with jit compile: %d, self is internal format: %d, target is internal format: %d, output is internal format: %d, is_target is internal format: %d",
                !is_jit_disable, !self_base_format, !target_base_format, !output_base_format, !is_target_base_format);
    if (is_jit_disable && self_base_format && target_base_format && output_base_format && is_target_base_format) {
        return op_api::multilabel_margin_loss_forward_out(self, target, reduction, output, is_target);
    } else {
        return acl_op::multilabel_margin_loss_forward_out(self, target, reduction, output, is_target);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> nll_loss2d_forward_out(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, at::Tensor & output, at::Tensor & total_weight){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(output);
    bool total_weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(total_weight);

    ASCEND_LOGI("nll_loss2d_forward_out exec with jit compile: %d, self is internal format: %d, target is internal format: %d, weight is internal format: %d, output is internal format: %d, total_weight is internal format: %d",
                !is_jit_disable, !self_base_format, !target_base_format, !weight_base_format, !output_base_format, !total_weight_base_format);
    if (is_jit_disable && self_base_format && target_base_format && weight_base_format && output_base_format && total_weight_base_format) {
        return op_api::nll_loss2d_forward_out(self, target, weight, reduction, ignore_index, output, total_weight);
    } else {
        return acl_op::nll_loss2d_forward_out(self, target, weight, reduction, ignore_index, output, total_weight);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> nll_loss_forward_out(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, at::Tensor & output, at::Tensor & total_weight){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(output);
    bool total_weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(total_weight);

    ASCEND_LOGI("nll_loss_forward_out exec with jit compile: %d, self is internal format: %d, target is internal format: %d, weight is internal format: %d, output is internal format: %d, total_weight is internal format: %d",
                !is_jit_disable, !self_base_format, !target_base_format, !weight_base_format, !output_base_format, !total_weight_base_format);
    if (is_jit_disable && self_base_format && target_base_format && weight_base_format && output_base_format && total_weight_base_format) {
        return op_api::nll_loss_forward_out(self, target, weight, reduction, ignore_index, output, total_weight);
    } else {
        return acl_op::nll_loss_forward_out(self, target, weight, reduction, ignore_index, output, total_weight);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> npu_fused_infer_attention_score_out_symint(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const c10::optional<at::Tensor> & pse_shift, const c10::optional<at::Tensor> & atten_mask, at::OptionalSymIntArrayRef actual_seq_lengths, at::OptionalSymIntArrayRef actual_seq_lengths_kv, const c10::optional<at::Tensor> & dequant_scale1, const c10::optional<at::Tensor> & quant_scale1, const c10::optional<at::Tensor> & dequant_scale2, const c10::optional<at::Tensor> & quant_scale2, const c10::optional<at::Tensor> & quant_offset2, const c10::optional<at::Tensor> & antiquant_scale, const c10::optional<at::Tensor> & antiquant_offset, const c10::optional<at::Tensor> & key_antiquant_scale, const c10::optional<at::Tensor> & key_antiquant_offset, const c10::optional<at::Tensor> & value_antiquant_scale, const c10::optional<at::Tensor> & value_antiquant_offset, const c10::optional<at::Tensor> & block_table, const c10::optional<at::Tensor> & query_padding_size, const c10::optional<at::Tensor> & kv_padding_size, const c10::optional<at::Tensor> & key_shared_prefix, const c10::optional<at::Tensor> & value_shared_prefix, at::OptionalSymIntArrayRef actual_shared_prefix_len, const c10::optional<at::Tensor> & query_rope, const c10::optional<at::Tensor> & key_rope, const c10::optional<at::Tensor> & key_rope_antiquant_scale, int64_t num_heads, double scale, int64_t pre_tokens, int64_t next_tokens, c10::string_view input_layout, int64_t num_key_value_heads, int64_t sparse_mode, int64_t inner_precise, int64_t block_size, int64_t antiquant_mode, int64_t key_antiquant_mode, int64_t value_antiquant_mode, bool softmax_lse_flag, const c10::optional<at::Tensor> & workspace, at::Tensor & attention_out, at::Tensor & softmax_lse){
    return op_api::npu_fused_infer_attention_score_out_symint(query, key, value, pse_shift, atten_mask, actual_seq_lengths, actual_seq_lengths_kv, dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2, antiquant_scale, antiquant_offset, key_antiquant_scale, key_antiquant_offset, value_antiquant_scale, value_antiquant_offset, block_table, query_padding_size, kv_padding_size, key_shared_prefix, value_shared_prefix, actual_shared_prefix_len, query_rope, key_rope, key_rope_antiquant_scale, num_heads, scale, pre_tokens, next_tokens, input_layout, num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, key_antiquant_mode, value_antiquant_mode, softmax_lse_flag, workspace, attention_out, softmax_lse);
}
::std::tuple<at::Tensor &,at::Tensor &> sort_out(const at::Tensor & self, at::Dimname dim, bool descending, at::Tensor & values, at::Tensor & indices){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool values_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(values);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);

    ASCEND_LOGI("sort_out exec with jit compile: %d, self is internal format: %d, values is internal format: %d, indices is internal format: %d",
                !is_jit_disable, !self_base_format, !values_base_format, !indices_base_format);
    if (is_jit_disable && self_base_format && values_base_format && indices_base_format) {
        return op_api::sort_out(self, dim, descending, values, indices);
    } else {
        return acl_op::sort_out(self, dim, descending, values, indices);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> sort_out(const at::Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending, at::Tensor & values, at::Tensor & indices){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool values_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(values);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);

    if (!self_base_format || !values_base_format || !indices_base_format) {
        TORCH_CHECK(false,
            "Current operator sort_out do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::sort_out(self, stable, dim, descending, values, indices);
}
::std::tuple<at::Tensor &,at::Tensor &> sort_out(const at::Tensor & self, int64_t dim, bool descending, at::Tensor & values, at::Tensor & indices){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool values_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(values);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);

    ASCEND_LOGI("sort_out exec with jit compile: %d, self is internal format: %d, values is internal format: %d, indices is internal format: %d",
                !is_jit_disable, !self_base_format, !values_base_format, !indices_base_format);
    if (is_jit_disable && self_base_format && values_base_format && indices_base_format) {
        return op_api::sort_out(self, dim, descending, values, indices);
    } else {
        return acl_op::sort_out(self, dim, descending, values, indices);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> topk_out(const at::Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted, at::Tensor & values, at::Tensor & indices){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool values_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(values);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);

    ASCEND_LOGI("topk_out exec with jit compile: %d, self is internal format: %d, values is internal format: %d, indices is internal format: %d",
                !is_jit_disable, !self_base_format, !values_base_format, !indices_base_format);
    if (is_jit_disable && self_base_format && values_base_format && indices_base_format) {
        return op_api::topk_out(self, k, dim, largest, sorted, values, indices);
    } else {
        return acl_op::topk_out(self, k, dim, largest, sorted, values, indices);
    }
}
::std::tuple<at::Tensor &,at::Tensor &> triangular_solve_out(const at::Tensor & self, const at::Tensor & A, bool upper, bool transpose, bool unitriangular, at::Tensor & X, at::Tensor & M){
    return acl_op::triangular_solve_out(self, A, upper, transpose, unitriangular, X, M);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_multi_head_attention_backward(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & query_weight, const at::Tensor & key_weight, const at::Tensor & value_weight, const at::Tensor & out_proj_weight, const c10::optional<at::Tensor> & query_bias, const c10::optional<at::Tensor> & key_bias, const c10::optional<at::Tensor> & value_bias, const c10::optional<at::Tensor> & out_proj_bias, const at::Tensor & query_res, const at::Tensor & key_res, const at::Tensor & value_res, const at::Tensor & attn_scores, const at::Tensor & attn_res, const at::Tensor & context, const at::Tensor & y_grad, const at::Tensor & dropout_mask, int64_t attn_head_num, int64_t attn_dim_per_head, int64_t src_len, int64_t tgt_len, double dropout_prob, bool softmax_use_float){
    return acl_op::npu_multi_head_attention_backward(query, key, value, query_weight, key_weight, value_weight, out_proj_weight, query_bias, key_bias, value_bias, out_proj_bias, query_res, key_res, value_res, attn_scores, attn_res, context, y_grad, dropout_mask, attn_head_num, attn_dim_per_head, src_len, tgt_len, dropout_prob, softmax_use_float);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_lstm(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, const at::Tensor & seq_mask, const at::Tensor & h, const at::Tensor & c, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first, bool flag_seq, bool direction){
    return acl_op::npu_lstm(input, weight, bias, seq_mask, h, c, has_biases, num_layers, dropout, train, bidirectional, batch_first, flag_seq, direction);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_lstm_cell(const at::Tensor & input, const at::Tensor & w_ih, const at::Tensor & w_hh, const at::Tensor & h, const at::Tensor & c, const c10::optional<at::Tensor> & b_ih, const c10::optional<at::Tensor> & b_hh){
    return acl_op::npu_lstm_cell(input, w_ih, w_hh, h, c, b_ih, b_hh);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_lstm_data(const at::Tensor & input, const at::Tensor & batch_sizes, const at::Tensor & weight, const at::Tensor & bias, const at::Tensor & seq_mask, const at::Tensor & h, const at::Tensor & c, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first, bool flag_seq, bool direction){
    return acl_op::npu_lstm_data(input, batch_sizes, weight, bias, seq_mask, h, c, has_biases, num_layers, dropout, train, bidirectional, batch_first, flag_seq, direction);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_multi_head_attention(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & query_weight, const at::Tensor & key_weight, const at::Tensor & value_weight, const at::Tensor & attn_mask, const at::Tensor & out_proj_weight, const c10::optional<at::Tensor> & query_bias, const c10::optional<at::Tensor> & key_bias, const c10::optional<at::Tensor> & value_bias, const c10::optional<at::Tensor> & out_proj_bias, const c10::optional<at::Tensor> & dropout_mask, int64_t attn_head_num, int64_t attn_dim_per_head, int64_t src_len, int64_t tgt_len, double dropout_prob, bool softmax_use_float){
    return acl_op::npu_multi_head_attention(query, key, value, query_weight, key_weight, value_weight, attn_mask, out_proj_weight, query_bias, key_bias, value_bias, out_proj_bias, dropout_mask, attn_head_num, attn_dim_per_head, src_len, tgt_len, dropout_prob, softmax_use_float);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_lstm_cell_backward(const c10::optional<at::Tensor> & grady, const c10::optional<at::Tensor> & gradh, const c10::optional<at::Tensor> & gradc, const at::Tensor & input, const at::Tensor & w_ih, const at::Tensor & w_hh, const at::Tensor & h, const at::Tensor & c, const at::Tensor & y_output, const at::Tensor & h_output, const at::Tensor & c_output, const at::Tensor & i, const at::Tensor & j, const at::Tensor & f, const at::Tensor & o, const at::Tensor & tanhc){
    return acl_op::npu_lstm_cell_backward(grady, gradh, gradc, input, w_ih, w_hh, h, c, y_output, h_output, c_output, i, j, f, o, tanhc);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_moe_distribute_dispatch(const at::Tensor & x, const at::Tensor & expert_ids, c10::string_view group_ep, int64_t ep_world_size, int64_t ep_rank_id, int64_t moe_expert_num, const c10::optional<at::Tensor> & scales, const c10::optional<at::Tensor> & x_active_mask, const c10::optional<at::Tensor> & expert_scales, c10::string_view group_tp, int64_t tp_world_size, int64_t tp_rank_id, int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num, int64_t quant_mode, int64_t global_bs, int64_t expert_token_nums_type){
    bool x_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x);
    bool expert_ids_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(expert_ids);
    bool scales_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(scales);
    bool x_active_mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x_active_mask);
    bool expert_scales_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(expert_scales);

    if (!x_base_format || !expert_ids_base_format || !scales_base_format || !x_active_mask_base_format || !expert_scales_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_moe_distribute_dispatch do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_moe_distribute_dispatch(x, expert_ids, group_ep, ep_world_size, ep_rank_id, moe_expert_num, scales, x_active_mask, expert_scales, group_tp, tp_world_size, tp_rank_id, expert_shard_type, shared_expert_num, shared_expert_rank_num, quant_mode, global_bs, expert_token_nums_type);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_moe_distribute_dispatch_v2(const at::Tensor & x, const at::Tensor & expert_ids, c10::string_view group_ep, int64_t ep_world_size, int64_t ep_rank_id, int64_t moe_expert_num, const c10::optional<at::Tensor> & scales, const c10::optional<at::Tensor> & x_active_mask, const c10::optional<at::Tensor> & expert_scales, c10::string_view group_tp, int64_t tp_world_size, int64_t tp_rank_id, int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num, int64_t quant_mode, int64_t global_bs, int64_t expert_token_nums_type){
    bool x_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x);
    bool expert_ids_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(expert_ids);
    bool scales_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(scales);
    bool x_active_mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x_active_mask);
    bool expert_scales_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(expert_scales);

    if (!x_base_format || !expert_ids_base_format || !scales_base_format || !x_active_mask_base_format || !expert_scales_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_moe_distribute_dispatch_v2 do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_moe_distribute_dispatch_v2(x, expert_ids, group_ep, ep_world_size, ep_rank_id, moe_expert_num, scales, x_active_mask, expert_scales, group_tp, tp_world_size, tp_rank_id, expert_shard_type, shared_expert_num, shared_expert_rank_num, quant_mode, global_bs, expert_token_nums_type);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_fusion_attention_grad_v2(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & dy, int64_t head_num, c10::string_view input_layout, const c10::optional<at::Tensor> & pse, const c10::optional<at::Tensor> & padding_mask, const c10::optional<at::Tensor> & atten_mask, const c10::optional<at::Tensor> & softmax_max, const c10::optional<at::Tensor> & softmax_sum, const c10::optional<at::Tensor> & softmax_in, const c10::optional<at::Tensor> & attention_in, const c10::optional<at::Tensor> & query_rope, const c10::optional<at::Tensor> & key_rope, double scale_value, double keep_prob, int64_t pre_tokens, int64_t next_tokens, int64_t inner_precise, int64_t seed, int64_t offset, int64_t numels, at::OptionalIntArrayRef prefix, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_seq_kvlen, int64_t sparse_mode, bool gen_mask_parallel, bool sync, int64_t pse_type, at::OptionalIntArrayRef q_start_idx, at::OptionalIntArrayRef kv_start_idx){
    bool query_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(query);
    bool key_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(key);
    bool value_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(value);
    bool dy_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(dy);
    bool pse_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(pse);
    bool padding_mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(padding_mask);
    bool atten_mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(atten_mask);
    bool softmax_max_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(softmax_max);
    bool softmax_sum_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(softmax_sum);
    bool softmax_in_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(softmax_in);
    bool attention_in_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(attention_in);
    bool query_rope_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(query_rope);
    bool key_rope_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(key_rope);

    if (!query_base_format || !key_base_format || !value_base_format || !dy_base_format || !pse_base_format || !padding_mask_base_format || !atten_mask_base_format || !softmax_max_base_format || !softmax_sum_base_format || !softmax_in_base_format || !attention_in_base_format || !query_rope_base_format || !key_rope_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_fusion_attention_grad_v2 do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_fusion_attention_grad_v2(query, key, value, dy, head_num, input_layout, pse, padding_mask, atten_mask, softmax_max, softmax_sum, softmax_in, attention_in, query_rope, key_rope, scale_value, keep_prob, pre_tokens, next_tokens, inner_precise, seed, offset, numels, prefix, actual_seq_qlen, actual_seq_kvlen, sparse_mode, gen_mask_parallel, sync, pse_type, q_start_idx, kv_start_idx);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_gru(const at::Tensor & input, const at::Tensor & hx, const at::Tensor & weight_input, const at::Tensor & weight_hidden, const at::Tensor & bias_input, const at::Tensor & bias_hidden, const at::Tensor & seq_length, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first){
    return acl_op::npu_gru(input, hx, weight_input, weight_hidden, bias_input, bias_hidden, seq_length, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_gru_backward(const c10::optional<at::Tensor> & grady, const c10::optional<at::Tensor> & gradh, const at::Tensor & input, const at::Tensor & weight_input, const at::Tensor & weight_hidden, const at::Tensor & bias_input, const at::Tensor & bias_hidden, const at::Tensor & seq_length, const at::Tensor & hx, const at::Tensor & y_output, const at::Tensor & h_output, const at::Tensor & output_updata, const at::Tensor & output_reset, const at::Tensor & output_new, const at::Tensor & hidden_new){
    return acl_op::npu_gru_backward(grady, gradh, input, weight_input, weight_hidden, bias_input, bias_hidden, seq_length, hx, y_output, h_output, output_updata, output_reset, output_new, hidden_new);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_dequant_rope_quant_kvcache(const at::Tensor & x, const at::Tensor & cos, const at::Tensor & sin, const at::Tensor & k_cache, const at::Tensor & v_cache, const at::Tensor & indices, const at::Tensor & scale_k, const at::Tensor & scale_v, at::IntArrayRef size_splits, const c10::optional<at::Tensor> & offset_k, const c10::optional<at::Tensor> & offset_v, const c10::optional<at::Tensor> & weight_scale, const c10::optional<at::Tensor> & activation_scale, const c10::optional<at::Tensor> & bias, int64_t quant_mode, c10::string_view input_layout, bool kv_output, c10::string_view cache_mode){
    bool x_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x);
    bool cos_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(cos);
    bool sin_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(sin);
    bool k_cache_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(k_cache);
    bool v_cache_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(v_cache);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);
    bool scale_k_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(scale_k);
    bool scale_v_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(scale_v);
    bool offset_k_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(offset_k);
    bool offset_v_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(offset_v);
    bool weight_scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight_scale);
    bool activation_scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(activation_scale);
    bool bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias);

    if (!x_base_format || !cos_base_format || !sin_base_format || !k_cache_base_format || !v_cache_base_format || !indices_base_format || !scale_k_base_format || !scale_v_base_format || !offset_k_base_format || !offset_v_base_format || !weight_scale_base_format || !activation_scale_base_format || !bias_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_dequant_rope_quant_kvcache do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_dequant_rope_quant_kvcache(x, cos, sin, k_cache, v_cache, indices, scale_k, scale_v, size_splits, offset_k, offset_v, weight_scale, activation_scale, bias, quant_mode, input_layout, kv_output, cache_mode);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_lstm_backward(const c10::optional<at::Tensor> & grady, const c10::optional<at::Tensor> & gradh, const c10::optional<at::Tensor> & gradc, const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, const at::Tensor & hx, const at::Tensor & cx, const at::Tensor & y_output, const at::Tensor & h_output, const at::Tensor & c_output, const at::Tensor & i, const at::Tensor & j, const at::Tensor & f, const at::Tensor & o, const at::Tensor & tanhc){
    return acl_op::npu_lstm_backward(grady, gradh, gradc, input, weight, bias, hx, cx, y_output, h_output, c_output, i, j, f, o, tanhc);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_lstm_data_backward(const c10::optional<at::Tensor> & grady_opt, const c10::optional<at::Tensor> & gradh_opt, const c10::optional<at::Tensor> & gradc_opt, const at::Tensor & input, const at::Tensor & batch_sizes, const at::Tensor & weight, const at::Tensor & bias, const at::Tensor & init_h, const at::Tensor & init_c, const at::Tensor & y, const at::Tensor & h, const at::Tensor & c, const at::Tensor & i, const at::Tensor & j, const at::Tensor & f, const at::Tensor & o, const at::Tensor & tanhc, bool flag_direction){
    return acl_op::npu_lstm_data_backward(grady_opt, gradh_opt, gradc_opt, input, batch_sizes, weight, bias, init_h, init_c, y, h, c, i, j, f, o, tanhc, flag_direction);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_mla_prolog_v2(const at::Tensor & token_x, const at::Tensor & weight_dq, const at::Tensor & weight_uq_qr, const at::Tensor & weight_uk, const at::Tensor & weight_dkv_kr, const at::Tensor & rmsnorm_gamma_cq, const at::Tensor & rmsnorm_gamma_ckv, const at::Tensor & rope_sin, const at::Tensor & rope_cos, const at::Tensor & cache_index, const at::Tensor & kv_cache, const at::Tensor & kr_cache, const c10::optional<at::Tensor> & dequant_scale_x, const c10::optional<at::Tensor> & dequant_scale_w_dq, const c10::optional<at::Tensor> & dequant_scale_w_uq_qr, const c10::optional<at::Tensor> & dequant_scale_w_dkv_kr, const c10::optional<at::Tensor> & quant_scale_ckv, const c10::optional<at::Tensor> & quant_scale_ckr, const c10::optional<at::Tensor> & smooth_scales_cq, double rmsnorm_epsilon_cq, double rmsnorm_epsilon_ckv, c10::string_view cache_mode){
    return op_api::npu_mla_prolog_v2(token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uq_qr, dequant_scale_w_dkv_kr, quant_scale_ckv, quant_scale_ckr, smooth_scales_cq, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv, cache_mode);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_rope_quant_kvcache(const at::Tensor & x, const at::Tensor & cos, const at::Tensor & sin, const at::Tensor & k_cache, const at::Tensor & v_cache, const at::Tensor & indices, const at::Tensor & scale_k, const at::Tensor & scale_v, at::IntArrayRef size_splits, const c10::optional<at::Tensor> & offset_k, const c10::optional<at::Tensor> & offset_v, int64_t quant_mode, c10::string_view input_layout, bool kv_output, c10::string_view cache_mode){
    bool x_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x);
    bool cos_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(cos);
    bool sin_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(sin);
    bool k_cache_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(k_cache);
    bool v_cache_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(v_cache);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);
    bool scale_k_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(scale_k);
    bool scale_v_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(scale_v);
    bool offset_k_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(offset_k);
    bool offset_v_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(offset_v);

    if (!x_base_format || !cos_base_format || !sin_base_format || !k_cache_base_format || !v_cache_base_format || !indices_base_format || !scale_k_base_format || !scale_v_base_format || !offset_k_base_format || !offset_v_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_rope_quant_kvcache do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_rope_quant_kvcache(x, cos, sin, k_cache, v_cache, indices, scale_k, scale_v, size_splits, offset_k, offset_v, quant_mode, input_layout, kv_output, cache_mode);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t,int64_t,int64_t> npu_fusion_attention(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, int64_t head_num, c10::string_view input_layout, const c10::optional<at::Tensor> & pse, const c10::optional<at::Tensor> & padding_mask, const c10::optional<at::Tensor> & atten_mask, double scale, double keep_prob, int64_t pre_tockens, int64_t next_tockens, int64_t inner_precise, at::OptionalIntArrayRef prefix, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_seq_kvlen, int64_t sparse_mode, bool gen_mask_parallel, bool sync){
    bool query_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(query);
    bool key_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(key);
    bool value_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(value);
    bool pse_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(pse);
    bool padding_mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(padding_mask);
    bool atten_mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(atten_mask);

    if (!query_base_format || !key_base_format || !value_base_format || !pse_base_format || !padding_mask_base_format || !atten_mask_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_fusion_attention do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_fusion_attention(query, key, value, head_num, input_layout, pse, padding_mask, atten_mask, scale, keep_prob, pre_tockens, next_tockens, inner_precise, prefix, actual_seq_qlen, actual_seq_kvlen, sparse_mode, gen_mask_parallel, sync);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t,int64_t,int64_t> npu_fusion_attention_v2(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, int64_t head_num, c10::string_view input_layout, const c10::optional<at::Tensor> & pse, const c10::optional<at::Tensor> & padding_mask, const c10::optional<at::Tensor> & atten_mask, const c10::optional<at::Tensor> & query_rope, const c10::optional<at::Tensor> & key_rope, double scale, double keep_prob, int64_t pre_tokens, int64_t next_tokens, int64_t inner_precise, at::OptionalIntArrayRef prefix, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_seq_kvlen, int64_t sparse_mode, bool gen_mask_parallel, bool sync, int64_t pse_type, at::OptionalIntArrayRef q_start_idx, at::OptionalIntArrayRef kv_start_idx){
    bool query_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(query);
    bool key_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(key);
    bool value_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(value);
    bool pse_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(pse);
    bool padding_mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(padding_mask);
    bool atten_mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(atten_mask);
    bool query_rope_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(query_rope);
    bool key_rope_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(key_rope);

    if (!query_base_format || !key_base_format || !value_base_format || !pse_base_format || !padding_mask_base_format || !atten_mask_base_format || !query_rope_base_format || !key_rope_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_fusion_attention_v2 do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_fusion_attention_v2(query, key, value, head_num, input_layout, pse, padding_mask, atten_mask, query_rope, key_rope, scale, keep_prob, pre_tokens, next_tokens, inner_precise, prefix, actual_seq_qlen, actual_seq_kvlen, sparse_mode, gen_mask_parallel, sync, pse_type, q_start_idx, kv_start_idx);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t> _batch_norm_impl_index(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double momentum, double eps, bool cudnn_enabled){
    return acl_op::_batch_norm_impl_index(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> _embedding_bag(const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<at::Tensor> & per_sample_weights, bool include_last_offset, int64_t padding_idx){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);
    bool offsets_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(offsets);
    bool per_sample_weights_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(per_sample_weights);

    ASCEND_LOGI("_embedding_bag exec with jit compile: %d, weight is internal format: %d, indices is internal format: %d, offsets is internal format: %d, per_sample_weights is internal format: %d",
                !is_jit_disable, !weight_base_format, !indices_base_format, !offsets_base_format, !per_sample_weights_base_format);
    if (is_jit_disable && weight_base_format && indices_base_format && offsets_base_format && per_sample_weights_base_format) {
        return op_api::_embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx);
    } else {
        return acl_op::_embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> _embedding_bag_forward_only(const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<at::Tensor> & per_sample_weights, bool include_last_offset, int64_t padding_idx){
    return acl_op::_embedding_bag_forward_only(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> batch_norm_backward_reduce(const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const c10::optional<at::Tensor> & weight, bool input_g, bool weight_g, bool bias_g){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_out);
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool mean_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mean);
    bool invstd_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(invstd);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);

    ASCEND_LOGI("batch_norm_backward_reduce exec with jit compile: %d, grad_out is internal format: %d, input is internal format: %d, mean is internal format: %d, invstd is internal format: %d, weight is internal format: %d",
                !is_jit_disable, !grad_out_base_format, !input_base_format, !mean_base_format, !invstd_base_format, !weight_base_format);
    if (is_jit_disable && grad_out_base_format && input_base_format && mean_base_format && invstd_base_format && weight_base_format) {
        return op_api::batch_norm_backward_reduce(grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g);
    } else {
        return acl_op::batch_norm_backward_reduce(grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_add_layer_norm(const at::Tensor & x1, const at::Tensor & x2, const at::Tensor & gamma, const at::Tensor & beta, double epsilon, bool additional_output){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool x1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x1);
    bool x2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x2);
    bool gamma_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(gamma);
    bool beta_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(beta);

    ASCEND_LOGI("npu_add_layer_norm exec with jit compile: %d, x1 is internal format: %d, x2 is internal format: %d, gamma is internal format: %d, beta is internal format: %d",
                !is_jit_disable, !x1_base_format, !x2_base_format, !gamma_base_format, !beta_base_format);
    if (is_jit_disable && x1_base_format && x2_base_format && gamma_base_format && beta_base_format) {
        return op_api::npu_add_layer_norm(x1, x2, gamma, beta, epsilon, additional_output);
    } else {
        return acl_op::npu_add_layer_norm(x1, x2, gamma, beta, epsilon, additional_output);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_add_layer_norm_backward(const c10::optional<at::Tensor> & dy_opt, const at::Tensor & x1, const at::Tensor & x2, const at::Tensor & rstd, const at::Tensor & mean, const at::Tensor & gamma, const c10::optional<at::Tensor> & dsum_opt){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool dy_opt_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(dy_opt);
    bool x1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x1);
    bool x2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x2);
    bool rstd_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(rstd);
    bool mean_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mean);
    bool gamma_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(gamma);
    bool dsum_opt_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(dsum_opt);

    ASCEND_LOGI("npu_add_layer_norm_backward exec with jit compile: %d, dy_opt is internal format: %d, x1 is internal format: %d, x2 is internal format: %d, rstd is internal format: %d, mean is internal format: %d, gamma is internal format: %d, dsum_opt is internal format: %d",
                !is_jit_disable, !dy_opt_base_format, !x1_base_format, !x2_base_format, !rstd_base_format, !mean_base_format, !gamma_base_format, !dsum_opt_base_format);
    if (is_jit_disable && dy_opt_base_format && x1_base_format && x2_base_format && rstd_base_format && mean_base_format && gamma_base_format && dsum_opt_base_format) {
        return op_api::npu_add_layer_norm_backward(dy_opt, x1, x2, rstd, mean, gamma, dsum_opt);
    } else {
        return acl_op::npu_add_layer_norm_backward(dy_opt, x1, x2, rstd, mean, gamma, dsum_opt);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_add_rms_norm_cast(const at::Tensor & x1, const at::Tensor & x2, const at::Tensor & gamma, double epsilon){
    bool x1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x1);
    bool x2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x2);
    bool gamma_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(gamma);

    if (!x1_base_format || !x2_base_format || !gamma_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_add_rms_norm_cast do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_add_rms_norm_cast(x1, x2, gamma, epsilon);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_batch_nms(const at::Tensor & self, const at::Tensor & scores, double score_threshold, double iou_threshold, int64_t max_size_per_class, int64_t max_total_size, bool change_coordinate_frame, bool transpose_box){
    return acl_op::npu_batch_nms(self, scores, score_threshold, iou_threshold, max_size_per_class, max_total_size, change_coordinate_frame, transpose_box);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_cross_entropy_loss(const at::Tensor & input, const at::Tensor & target, const c10::optional<at::Tensor> & weight, c10::string_view reduction, int64_t ignore_index, double label_smoothing, double lse_square_scale_for_zloss, bool return_zloss){
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);

    if (!input_base_format || !target_base_format || !weight_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_cross_entropy_loss do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_cross_entropy_loss(input, target, weight, reduction, ignore_index, label_smoothing, lse_square_scale_for_zloss, return_zloss);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_deep_norm_backward(const at::Tensor & dy, const at::Tensor & x, const at::Tensor & gx, const at::Tensor & gamma, const at::Tensor & mean, const at::Tensor & rstd, double alpha){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool dy_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(dy);
    bool x_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x);
    bool gx_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(gx);
    bool gamma_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(gamma);
    bool mean_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mean);
    bool rstd_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(rstd);

    ASCEND_LOGI("npu_deep_norm_backward exec with jit compile: %d, dy is internal format: %d, x is internal format: %d, gx is internal format: %d, gamma is internal format: %d, mean is internal format: %d, rstd is internal format: %d",
                !is_jit_disable, !dy_base_format, !x_base_format, !gx_base_format, !gamma_base_format, !mean_base_format, !rstd_base_format);
    if (is_jit_disable && dy_base_format && x_base_format && gx_base_format && gamma_base_format && mean_base_format && rstd_base_format) {
        return op_api::npu_deep_norm_backward(dy, x, gx, gamma, mean, rstd, alpha);
    } else {
        return acl_op::npu_deep_norm_backward(dy, x, gx, gamma, mean, rstd, alpha);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_deformable_conv2dbk(const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & offset_out, const at::Tensor & weight, const at::Tensor & offset, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, int64_t deformable_groups, bool modulated){
    return acl_op::npu_deformable_conv2dbk(input, grad_output, offset_out, weight, offset, kernel_size, stride, padding, dilation, groups, deformable_groups, modulated);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_fusion_attention_grad(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & dy, int64_t head_num, c10::string_view input_layout, const c10::optional<at::Tensor> & pse, const c10::optional<at::Tensor> & padding_mask, const c10::optional<at::Tensor> & atten_mask, const c10::optional<at::Tensor> & softmax_max, const c10::optional<at::Tensor> & softmax_sum, const c10::optional<at::Tensor> & softmax_in, const c10::optional<at::Tensor> & attention_in, double scale_value, double keep_prob, int64_t pre_tockens, int64_t next_tockens, int64_t inner_precise, int64_t seed, int64_t offset, int64_t numels, at::OptionalIntArrayRef prefix, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_seq_kvlen, int64_t sparse_mode, bool gen_mask_parallel, bool sync){
    bool query_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(query);
    bool key_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(key);
    bool value_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(value);
    bool dy_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(dy);
    bool pse_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(pse);
    bool padding_mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(padding_mask);
    bool atten_mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(atten_mask);
    bool softmax_max_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(softmax_max);
    bool softmax_sum_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(softmax_sum);
    bool softmax_in_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(softmax_in);
    bool attention_in_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(attention_in);

    if (!query_base_format || !key_base_format || !value_base_format || !dy_base_format || !pse_base_format || !padding_mask_base_format || !atten_mask_base_format || !softmax_max_base_format || !softmax_sum_base_format || !softmax_in_base_format || !attention_in_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_fusion_attention_grad do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_fusion_attention_grad(query, key, value, dy, head_num, input_layout, pse, padding_mask, atten_mask, softmax_max, softmax_sum, softmax_in, attention_in, scale_value, keep_prob, pre_tockens, next_tockens, inner_precise, seed, offset, numels, prefix, actual_seq_qlen, actual_seq_kvlen, sparse_mode, gen_mask_parallel, sync);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_kv_rmsnorm_rope_cache(const at::Tensor & kv, const at::Tensor & gamma, const at::Tensor & cos, const at::Tensor & sin, const at::Tensor & index, const at::Tensor & k_cache, const at::Tensor & ckv_cache, const c10::optional<at::Tensor> & k_rope_scale, const c10::optional<at::Tensor> & c_kv_scale, const c10::optional<at::Tensor> & k_rope_offset, const c10::optional<at::Tensor> & c_kv_offset, double epsilon, c10::string_view cache_mode, bool is_output_kv){
    bool kv_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(kv);
    bool gamma_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(gamma);
    bool cos_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(cos);
    bool sin_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(sin);
    bool index_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(index);
    bool k_cache_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(k_cache);
    bool ckv_cache_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(ckv_cache);
    bool k_rope_scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(k_rope_scale);
    bool c_kv_scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(c_kv_scale);
    bool k_rope_offset_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(k_rope_offset);
    bool c_kv_offset_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(c_kv_offset);

    if (!kv_base_format || !gamma_base_format || !cos_base_format || !sin_base_format || !index_base_format || !k_cache_base_format || !ckv_cache_base_format || !k_rope_scale_base_format || !c_kv_scale_base_format || !k_rope_offset_base_format || !c_kv_offset_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_kv_rmsnorm_rope_cache do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_kv_rmsnorm_rope_cache(kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope_scale, c_kv_scale, k_rope_offset, c_kv_offset, epsilon, cache_mode, is_output_kv);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_mla_prolog(const at::Tensor & token_x, const at::Tensor & weight_dq, const at::Tensor & weight_uq_qr, const at::Tensor & weight_uk, const at::Tensor & weight_dkv_kr, const at::Tensor & rmsnorm_gamma_cq, const at::Tensor & rmsnorm_gamma_ckv, const at::Tensor & rope_sin, const at::Tensor & rope_cos, const at::Tensor & cache_index, const at::Tensor & kv_cache, const at::Tensor & kr_cache, const c10::optional<at::Tensor> & dequant_scale_x, const c10::optional<at::Tensor> & dequant_scale_w_dq, const c10::optional<at::Tensor> & dequant_scale_w_uq_qr, const c10::optional<at::Tensor> & dequant_scale_w_dkv_kr, const c10::optional<at::Tensor> & quant_scale_ckv, const c10::optional<at::Tensor> & quant_scale_ckr, const c10::optional<at::Tensor> & smooth_scales_cq, double rmsnorm_epsilon_cq, double rmsnorm_epsilon_ckv, c10::string_view cache_mode){
    return acl_op::npu_mla_prolog(token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uq_qr, dequant_scale_w_dkv_kr, quant_scale_ckv, quant_scale_ckr, smooth_scales_cq, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv, cache_mode);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_moe_init_routing_v2(const at::Tensor & x, const at::Tensor & expert_idx, const c10::optional<at::Tensor> & scale, const c10::optional<at::Tensor> & offset, int64_t active_num, int64_t expert_capacity, int64_t expert_num, int64_t drop_pad_mode, int64_t expert_tokens_num_type, bool expert_tokens_num_flag, int64_t quant_mode, at::IntArrayRef active_expert_range, int64_t row_idx_type){
    bool x_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x);
    bool expert_idx_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(expert_idx);
    bool scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(scale);
    bool offset_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(offset);

    if (!x_base_format || !expert_idx_base_format || !scale_base_format || !offset_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_moe_init_routing_v2 do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_moe_init_routing_v2(x, expert_idx, scale, offset, active_num, expert_capacity, expert_num, drop_pad_mode, expert_tokens_num_type, expert_tokens_num_flag, quant_mode, active_expert_range, row_idx_type);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_moe_re_routing(const at::Tensor & tokens, const at::Tensor & expert_token_num_per_rank, const c10::optional<at::Tensor> & per_token_scales, int64_t expert_token_num_type, int64_t idx_type){
    bool tokens_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tokens);
    bool expert_token_num_per_rank_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(expert_token_num_per_rank);
    bool per_token_scales_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(per_token_scales);

    if (!tokens_base_format || !expert_token_num_per_rank_base_format || !per_token_scales_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_moe_re_routing do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_moe_re_routing(tokens, expert_token_num_per_rank, per_token_scales, expert_token_num_type, idx_type);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_nsa_compress_attention(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, double scale_value, int64_t head_num, int64_t compress_block_size, int64_t compress_stride, int64_t select_block_size, int64_t select_block_count, const c10::optional<at::Tensor> & topk_mask, const c10::optional<at::Tensor> & atten_mask, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_cmp_seq_kvlen, at::OptionalIntArrayRef actual_sel_seq_kvlen){
    bool query_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(query);
    bool key_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(key);
    bool value_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(value);
    bool topk_mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(topk_mask);
    bool atten_mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(atten_mask);

    if (!query_base_format || !key_base_format || !value_base_format || !topk_mask_base_format || !atten_mask_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_nsa_compress_attention do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_nsa_compress_attention(query, key, value, scale_value, head_num, compress_block_size, compress_stride, select_block_size, select_block_count, topk_mask, atten_mask, actual_seq_qlen, actual_cmp_seq_kvlen, actual_sel_seq_kvlen);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _batch_norm_impl_index_backward(int64_t impl_index, const at::Tensor & input, const at::Tensor & grad_output, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, const c10::optional<at::Tensor> & save_mean, const c10::optional<at::Tensor> & save_var_transform, bool train, double eps, ::std::array<bool,3> output_mask, const at::Tensor & reservedSpace){
    return acl_op::_batch_norm_impl_index_backward(impl_index, input, grad_output, weight, running_mean, running_var, save_mean, save_var_transform, train, eps, output_mask, reservedSpace);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _linalg_svd(const at::Tensor & A, bool full_matrices, bool compute_uv, c10::optional<c10::string_view> driver){
    return acl_op::_linalg_svd(A, full_matrices, compute_uv, driver);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _native_batch_norm_legit(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, at::Tensor & running_mean, at::Tensor & running_var, bool training, double momentum, double eps){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias);
    bool running_mean_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(running_mean);
    bool running_var_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(running_var);

    ASCEND_LOGI("_native_batch_norm_legit exec with jit compile: %d, input is internal format: %d, weight is internal format: %d, bias is internal format: %d, running_mean is internal format: %d, running_var is internal format: %d",
                !is_jit_disable, !input_base_format, !weight_base_format, !bias_base_format, !running_mean_base_format, !running_var_base_format);
    if (is_jit_disable && input_base_format && weight_base_format && bias_base_format && running_mean_base_format && running_var_base_format) {
        return op_api::_native_batch_norm_legit(input, weight, bias, running_mean, running_var, training, momentum, eps);
    } else {
        return acl_op::_native_batch_norm_legit(input, weight, bias, running_mean, running_var, training, momentum, eps);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _slow_conv2d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, ::std::array<bool,3> output_mask){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);

    ASCEND_LOGI("_slow_conv2d_backward exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, weight is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !weight_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && weight_base_format) {
        return op_api::_slow_conv2d_backward(grad_output, self, weight, kernel_size, stride, padding, output_mask);
    } else {
        return acl_op::_slow_conv2d_backward(grad_output, self, weight, kernel_size, stride, padding, output_mask);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _transform_bias_rescale_qkv(const at::Tensor & qkv, const at::Tensor & qkv_bias, int64_t num_heads){
    bool qkv_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(qkv);
    bool qkv_bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(qkv_bias);

    if (!qkv_base_format || !qkv_bias_base_format) {
        TORCH_CHECK(false,
            "Current operator _transform_bias_rescale_qkv do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_transform_bias_rescale_qkv(qkv, qkv_bias, num_heads);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _unique2(const at::Tensor & self, bool sorted, bool return_inverse, bool return_counts){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("_unique2 exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::_unique2(self, sorted, return_inverse, return_counts);
    } else {
        return acl_op::_unique2(self, sorted, return_inverse, return_counts);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> conv_tbc_backward(const at::Tensor & self, const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, int64_t pad){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias);

    ASCEND_LOGI("conv_tbc_backward exec with jit compile: %d, self is internal format: %d, input is internal format: %d, weight is internal format: %d, bias is internal format: %d",
                !is_jit_disable, !self_base_format, !input_base_format, !weight_base_format, !bias_base_format);
    if (is_jit_disable && self_base_format && input_base_format && weight_base_format && bias_base_format) {
        return op_api::conv_tbc_backward(self, input, weight, bias, pad);
    } else {
        return acl_op::conv_tbc_backward(self, input, weight, bias, pad);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> convolution_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, at::OptionalIntArrayRef bias_sizes, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, ::std::array<bool,3> output_mask){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);

    ASCEND_LOGI("convolution_backward exec with jit compile: %d, grad_output is internal format: %d, input is internal format: %d, weight is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !input_base_format, !weight_base_format);
    if (is_jit_disable && grad_output_base_format && input_base_format && weight_base_format) {
        return op_api::convolution_backward(grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups, output_mask);
    } else {
        return acl_op::convolution_backward(grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups, output_mask);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> convolution_backward_overrideable(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, ::std::array<bool,3> output_mask){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);

    ASCEND_LOGI("convolution_backward_overrideable exec with jit compile: %d, grad_output is internal format: %d, input is internal format: %d, weight is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !input_base_format, !weight_base_format);
    if (is_jit_disable && grad_output_base_format && input_base_format && weight_base_format) {
        return op_api::convolution_backward_overrideable(grad_output, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask);
    } else {
        return acl_op::convolution_backward_overrideable(grad_output, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> lstm(const at::Tensor & data, const at::Tensor & batch_sizes, at::TensorList hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional){
    return acl_op::lstm(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> lstm(const at::Tensor & input, at::TensorList hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first){
    return acl_op::lstm(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> matmul_double_backward(const c10::optional<at::Tensor> & grad_self, const c10::optional<at::Tensor> & grad_other, const at::Tensor & grad_out, const at::Tensor & self, const at::Tensor & other, ::std::array<bool,3> mask){
    bool grad_self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_self);
    bool grad_other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_other);
    bool grad_out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_out);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    if (!grad_self_base_format || !grad_other_base_format || !grad_out_base_format || !self_base_format || !other_base_format) {
        TORCH_CHECK(false,
            "Current operator matmul_double_backward do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::matmul_double_backward(grad_self, grad_other, grad_out, self, other, mask);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_batch_norm(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double momentum, double eps){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias);
    bool running_mean_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(running_mean);
    bool running_var_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(running_var);

    ASCEND_LOGI("native_batch_norm exec with jit compile: %d, input is internal format: %d, weight is internal format: %d, bias is internal format: %d, running_mean is internal format: %d, running_var is internal format: %d",
                !is_jit_disable, !input_base_format, !weight_base_format, !bias_base_format, !running_mean_base_format, !running_var_base_format);
    if (is_jit_disable && input_base_format && weight_base_format && bias_base_format && running_mean_base_format && running_var_base_format) {
        return op_api::native_batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps);
    } else {
        return acl_op::native_batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_batch_norm_backward(const at::Tensor & grad_out, const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, const c10::optional<at::Tensor> & save_mean, const c10::optional<at::Tensor> & save_invstd, bool train, double eps, ::std::array<bool,3> output_mask){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_out);
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool running_mean_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(running_mean);
    bool running_var_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(running_var);
    bool save_mean_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(save_mean);
    bool save_invstd_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(save_invstd);

    ASCEND_LOGI("native_batch_norm_backward exec with jit compile: %d, grad_out is internal format: %d, input is internal format: %d, weight is internal format: %d, running_mean is internal format: %d, running_var is internal format: %d, save_mean is internal format: %d, save_invstd is internal format: %d",
                !is_jit_disable, !grad_out_base_format, !input_base_format, !weight_base_format, !running_mean_base_format, !running_var_base_format, !save_mean_base_format, !save_invstd_base_format);
    if (is_jit_disable && grad_out_base_format && input_base_format && weight_base_format && running_mean_base_format && running_var_base_format && save_mean_base_format && save_invstd_base_format) {
        return op_api::native_batch_norm_backward(grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask);
    } else {
        return acl_op::native_batch_norm_backward(grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_group_norm(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, int64_t N, int64_t C, int64_t HxW, int64_t group, double eps){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias);

    ASCEND_LOGI("native_group_norm exec with jit compile: %d, input is internal format: %d, weight is internal format: %d, bias is internal format: %d",
                !is_jit_disable, !input_base_format, !weight_base_format, !bias_base_format);
    if (is_jit_disable && input_base_format && weight_base_format && bias_base_format) {
        return op_api::native_group_norm(input, weight, bias, N, C, HxW, group, eps);
    } else {
        return acl_op::native_group_norm(input, weight, bias, N, C, HxW, group, eps);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_group_norm_backward(const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & rstd, const c10::optional<at::Tensor> & weight, int64_t N, int64_t C, int64_t HxW, int64_t group, ::std::array<bool,3> output_mask){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_out);
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool mean_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mean);
    bool rstd_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(rstd);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);

    ASCEND_LOGI("native_group_norm_backward exec with jit compile: %d, grad_out is internal format: %d, input is internal format: %d, mean is internal format: %d, rstd is internal format: %d, weight is internal format: %d",
                !is_jit_disable, !grad_out_base_format, !input_base_format, !mean_base_format, !rstd_base_format, !weight_base_format);
    if (is_jit_disable && grad_out_base_format && input_base_format && mean_base_format && rstd_base_format && weight_base_format) {
        return op_api::native_group_norm_backward(grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask);
    } else {
        return acl_op::native_group_norm_backward(grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_layer_norm(const at::Tensor & input, at::IntArrayRef normalized_shape, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, double eps){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias);

    ASCEND_LOGI("native_layer_norm exec with jit compile: %d, input is internal format: %d, weight is internal format: %d, bias is internal format: %d",
                !is_jit_disable, !input_base_format, !weight_base_format, !bias_base_format);
    if (is_jit_disable && input_base_format && weight_base_format && bias_base_format) {
        return op_api::native_layer_norm(input, normalized_shape, weight, bias, eps);
    } else {
        return acl_op::native_layer_norm(input, normalized_shape, weight, bias, eps);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_layer_norm_backward(const at::Tensor & grad_out, const at::Tensor & input, at::IntArrayRef normalized_shape, const at::Tensor & mean, const at::Tensor & rstd, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, ::std::array<bool,3> output_mask){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_out);
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool mean_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mean);
    bool rstd_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(rstd);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias);

    ASCEND_LOGI("native_layer_norm_backward exec with jit compile: %d, grad_out is internal format: %d, input is internal format: %d, mean is internal format: %d, rstd is internal format: %d, weight is internal format: %d, bias is internal format: %d",
                !is_jit_disable, !grad_out_base_format, !input_base_format, !mean_base_format, !rstd_base_format, !weight_base_format, !bias_base_format);
    if (is_jit_disable && grad_out_base_format && input_base_format && mean_base_format && rstd_base_format && weight_base_format && bias_base_format) {
        return op_api::native_layer_norm_backward(grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask);
    } else {
        return acl_op::native_layer_norm_backward(grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_add_rms_norm(const at::Tensor & x1, const at::Tensor & x2, const at::Tensor & gamma, double epsilon){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool x1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x1);
    bool x2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x2);
    bool gamma_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(gamma);

    ASCEND_LOGI("npu_add_rms_norm exec with jit compile: %d, x1 is internal format: %d, x2 is internal format: %d, gamma is internal format: %d",
                !is_jit_disable, !x1_base_format, !x2_base_format, !gamma_base_format);
    if (is_jit_disable && x1_base_format && x2_base_format && gamma_base_format) {
        return op_api::npu_add_rms_norm(x1, x2, gamma, epsilon);
    } else {
        return acl_op::npu_add_rms_norm(x1, x2, gamma, epsilon);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_add_rms_norm_quant(const at::Tensor & x1, const at::Tensor & x2, const at::Tensor & gamma, const at::Tensor & scales1, const c10::optional<at::Tensor> & zero_points1, const c10::optional<at::Tensor> & scales2, const c10::optional<at::Tensor> & zero_points2, int64_t axis, double epsilon, bool div_mode){
    bool x1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x1);
    bool x2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x2);
    bool gamma_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(gamma);
    bool scales1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(scales1);
    bool zero_points1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(zero_points1);
    bool scales2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(scales2);
    bool zero_points2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(zero_points2);

    if (!x1_base_format || !x2_base_format || !gamma_base_format || !scales1_base_format || !zero_points1_base_format || !scales2_base_format || !zero_points2_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_add_rms_norm_quant do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_add_rms_norm_quant(x1, x2, gamma, scales1, zero_points1, scales2, zero_points2, axis, epsilon, div_mode);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_alltoallv_gmm(const at::Tensor & gmm_x, const at::Tensor & gmm_weight, c10::string_view hcom, int64_t ep_world_size, at::IntArrayRef send_counts, at::IntArrayRef recv_counts, const c10::optional<at::Tensor> & send_counts_tensor, const c10::optional<at::Tensor> & recv_counts_tensor, const c10::optional<at::Tensor> & mm_x, const c10::optional<at::Tensor> & mm_weight, bool trans_gmm_weight, bool trans_mm_weight, bool permute_out_flag){
    bool gmm_x_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(gmm_x);
    bool gmm_weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(gmm_weight);
    bool send_counts_tensor_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(send_counts_tensor);
    bool recv_counts_tensor_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(recv_counts_tensor);
    bool mm_x_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mm_x);
    bool mm_weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mm_weight);

    if (!gmm_x_base_format || !gmm_weight_base_format || !send_counts_tensor_base_format || !recv_counts_tensor_base_format || !mm_x_base_format || !mm_weight_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_alltoallv_gmm do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_alltoallv_gmm(gmm_x, gmm_weight, hcom, ep_world_size, send_counts, recv_counts, send_counts_tensor, recv_counts_tensor, mm_x, mm_weight, trans_gmm_weight, trans_mm_weight, permute_out_flag);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_apply_adam(const at::Scalar & beta1_power, const at::Scalar & beta2_power, const at::Scalar & lr, const at::Scalar & beta1, const at::Scalar & beta2, const at::Scalar & epsilon, const at::Tensor & grad, c10::optional<bool> use_locking, c10::optional<bool> use_nesterov){
    return acl_op::npu_apply_adam(beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, use_locking, use_nesterov);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_apply_adam_w(const at::Scalar & beta1_power, const at::Scalar & beta2_power, const at::Scalar & lr, const at::Scalar & weight_decay, const at::Scalar & beta1, const at::Scalar & beta2, const at::Scalar & epsilon, const at::Tensor & grad, const c10::optional<at::Tensor> & max_grad_norm, c10::optional<bool> amsgrad, c10::optional<bool> maximize){
    return acl_op::npu_apply_adam_w(beta1_power, beta2_power, lr, weight_decay, beta1, beta2, epsilon, grad, max_grad_norm, amsgrad, maximize);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_bert_apply_adam(const at::Scalar & lr, const at::Scalar & beta1, const at::Scalar & beta2, const at::Scalar & epsilon, const at::Tensor & grad, const at::Scalar & max_grad_norm, const at::Scalar & global_grad_norm, const at::Scalar & weight_decay, const c10::optional<at::Scalar> & step_size, int64_t adam_mode){
    return acl_op::npu_bert_apply_adam(lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay, step_size, adam_mode);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_conv2d_backward(const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, ::std::array<bool,3> output_mask){
    return acl_op::npu_conv2d_backward(input, grad_output, weight, stride, padding, dilation, groups, output_mask);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_conv3d_backward(const at::Tensor & input, const at::Tensor & grad, const at::Tensor & weight, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, ::std::array<bool,3> output_mask){
    return acl_op::npu_conv3d_backward(input, grad, weight, stride, padding, dilation, groups, output_mask);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_conv_transpose2d_backward(const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, ::std::array<bool,3> output_mask){
    return acl_op::npu_conv_transpose2d_backward(input, grad_output, weight, padding, output_padding, stride, dilation, groups, output_mask);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_conv_transpose3d_backward(const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, ::std::array<bool,3> output_mask){
    return acl_op::npu_conv_transpose3d_backward(input, grad_output, weight, padding, output_padding, stride, dilation, groups, output_mask);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_convolution_backward(const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, ::std::array<bool,3> output_mask){
    return acl_op::npu_convolution_backward(input, grad_output, weight, stride, padding, dilation, groups, output_mask);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_convolution_transpose_backward(const at::Tensor & input, const at::Tensor & grad, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, ::std::array<bool,3> grad_input_mask){
    return acl_op::npu_convolution_transpose_backward(input, grad, weight, padding, output_padding, stride, dilation, groups, grad_input_mask);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_deep_norm(const at::Tensor & x, const at::Tensor & gx, const at::Tensor & beta, const at::Tensor & gamma, double alpha, double epsilon){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool x_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x);
    bool gx_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(gx);
    bool beta_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(beta);
    bool gamma_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(gamma);

    ASCEND_LOGI("npu_deep_norm exec with jit compile: %d, x is internal format: %d, gx is internal format: %d, beta is internal format: %d, gamma is internal format: %d",
                !is_jit_disable, !x_base_format, !gx_base_format, !beta_base_format, !gamma_base_format);
    if (is_jit_disable && x_base_format && gx_base_format && beta_base_format && gamma_base_format) {
        return op_api::npu_deep_norm(x, gx, beta, gamma, alpha, epsilon);
    } else {
        return acl_op::npu_deep_norm(x, gx, beta, gamma, alpha, epsilon);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_dropout_with_add_softmax(const at::Tensor & self, const at::Tensor & x1, const at::Scalar & alpha, double prob, int64_t dim){
    return acl_op::npu_dropout_with_add_softmax(self, x1, alpha, prob, dim);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_dynamic_quant_asymmetric(const at::Tensor & input, const c10::optional<at::Tensor> & smooth_scales, const c10::optional<at::Tensor> & group_index, c10::optional<at::ScalarType> dst_type){
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool smooth_scales_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(smooth_scales);
    bool group_index_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(group_index);

    if (!input_base_format || !smooth_scales_base_format || !group_index_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_dynamic_quant_asymmetric do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_dynamic_quant_asymmetric(input, smooth_scales, group_index, dst_type);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_fused_attention_score_backward(const at::Tensor & grad_output, const at::Tensor & softmax_output, const at::Tensor & query_layer, const at::Tensor & key_layer, const at::Tensor & value_layer, const at::Tensor & mask, const at::Scalar & scale, double keep_prob, bool query_transpose, bool key_transpose, bool value_transpose, bool dx_transpose){
    return acl_op::npu_fused_attention_score_backward(grad_output, softmax_output, query_layer, key_layer, value_layer, mask, scale, keep_prob, query_transpose, key_transpose, value_transpose, dx_transpose);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_fused_attention_score_fwd(const at::Tensor & query_layer, const at::Tensor & key_layer, const at::Tensor & value_layer, const at::Tensor & attention_mask, const at::Scalar & scale, double keep_prob, bool query_transpose, bool key_transpose, bool bmm_score_transpose_a, bool bmm_score_transpose_b, bool value_transpose, bool dx_transpose){
    return acl_op::npu_fused_attention_score_fwd(query_layer, key_layer, value_layer, attention_mask, scale, keep_prob, query_transpose, key_transpose, bmm_score_transpose_a, bmm_score_transpose_b, value_transpose, dx_transpose);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_fused_attention_score_grad(const at::Tensor & grad_output, const at::Tensor & softmax_output, const at::Tensor & query_layer, const at::Tensor & key_layer, const at::Tensor & value_layer, const at::Tensor & mask, const at::Scalar & scale, double keep_prob, bool query_transpose, bool key_transpose, bool value_transpose, bool dx_transpose){
    return acl_op::npu_fused_attention_score_grad(grad_output, softmax_output, query_layer, key_layer, value_layer, mask, scale, keep_prob, query_transpose, key_transpose, value_transpose, dx_transpose);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_group_norm_silu(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, int64_t group, double eps){
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias);

    if (!input_base_format || !weight_base_format || !bias_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_group_norm_silu do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_group_norm_silu(input, weight, bias, group, eps);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_group_norm_swish(const at::Tensor & input, int64_t num_groups, const at::Tensor & weight, const at::Tensor & bias, c10::optional<double> eps, c10::optional<double> swish_scale){
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias);

    if (!input_base_format || !weight_base_format || !bias_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_group_norm_swish do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_group_norm_swish(input, num_groups, weight, bias, eps, swish_scale);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_group_norm_swish_grad(const at::Tensor & grad, const at::Tensor & input, int64_t num_groups, const at::Tensor & weight, const at::Tensor & bias, const at::Tensor & mean, const at::Tensor & rstd, ::std::array<bool,3> grad_input_mask, c10::optional<double> swish_scale){
    bool grad_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad);
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias);
    bool mean_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mean);
    bool rstd_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(rstd);

    if (!grad_base_format || !input_base_format || !weight_base_format || !bias_base_format || !mean_base_format || !rstd_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_group_norm_swish_grad do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_group_norm_swish_grad(grad, input, num_groups, weight, bias, mean, rstd, grad_input_mask, swish_scale);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_grouped_matmul_swiglu_quant(const at::Tensor & x, const at::Tensor & weight, const at::Tensor & group_list, const at::Tensor & weight_scale, const at::Tensor & x_scale, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & offset){
    return op_api::npu_grouped_matmul_swiglu_quant(x, weight, group_list, weight_scale, x_scale, bias, offset);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_layernorm_grad(const at::Tensor & grad_out, const at::Tensor & input, at::IntArrayRef normalized_shape, const at::Tensor & mean, const at::Tensor & rstd, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias){
    return acl_op::npu_layernorm_grad(grad_out, input, normalized_shape, mean, rstd, weight, bias);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_moe_distribute_combine_add_rms_norm(const at::Tensor & expand_x, const at::Tensor & expert_ids, const at::Tensor & expand_idx, const at::Tensor & ep_send_counts, const at::Tensor & expert_scales, const at::Tensor & residual_x, const at::Tensor & gamma, c10::string_view group_ep, int64_t ep_world_size, int64_t ep_rank_id, int64_t moe_expert_num, const c10::optional<at::Tensor> & tp_send_counts, const c10::optional<at::Tensor> & x_active_mask, const c10::optional<at::Tensor> & activation_scale, const c10::optional<at::Tensor> & weight_scale, const c10::optional<at::Tensor> & group_list, const c10::optional<at::Tensor> & expand_scales, const c10::optional<at::Tensor> & shared_expert_x, c10::string_view group_tp, int64_t tp_world_size, int64_t tp_rank_id, int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num, int64_t global_bs, int64_t out_dtype, int64_t comm_quant_mode, int64_t group_list_type, c10::string_view comm_alg, double norm_eps){
    bool expand_x_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(expand_x);
    bool expert_ids_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(expert_ids);
    bool expand_idx_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(expand_idx);
    bool ep_send_counts_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(ep_send_counts);
    bool expert_scales_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(expert_scales);
    bool residual_x_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(residual_x);
    bool gamma_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(gamma);
    bool tp_send_counts_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tp_send_counts);
    bool x_active_mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x_active_mask);
    bool activation_scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(activation_scale);
    bool weight_scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight_scale);
    bool group_list_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(group_list);
    bool expand_scales_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(expand_scales);
    bool shared_expert_x_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(shared_expert_x);

    if (!expand_x_base_format || !expert_ids_base_format || !expand_idx_base_format || !ep_send_counts_base_format || !expert_scales_base_format || !residual_x_base_format || !gamma_base_format || !tp_send_counts_base_format || !x_active_mask_base_format || !activation_scale_base_format || !weight_scale_base_format || !group_list_base_format || !expand_scales_base_format || !shared_expert_x_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_moe_distribute_combine_add_rms_norm do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_moe_distribute_combine_add_rms_norm(expand_x, expert_ids, expand_idx, ep_send_counts, expert_scales, residual_x, gamma, group_ep, ep_world_size, ep_rank_id, moe_expert_num, tp_send_counts, x_active_mask, activation_scale, weight_scale, group_list, expand_scales, shared_expert_x, group_tp, tp_world_size, tp_rank_id, expert_shard_type, shared_expert_num, shared_expert_rank_num, global_bs, out_dtype, comm_quant_mode, group_list_type, comm_alg, norm_eps);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_moe_gating_top_k(const at::Tensor & x, int64_t k, const c10::optional<at::Tensor> & bias, int64_t k_group, int64_t group_count, int64_t group_select_mode, int64_t renorm, int64_t norm_type, bool out_flag, double routed_scaling_factor, double eps){
    bool x_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x);
    bool bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias);

    if (!x_base_format || !bias_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_moe_gating_top_k do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_moe_gating_top_k(x, k, bias, k_group, group_count, group_select_mode, renorm, norm_type, out_flag, routed_scaling_factor, eps);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_moe_gating_top_k_softmax(const at::Tensor & x, const c10::optional<at::Tensor> & finished, int64_t k){
    bool x_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x);
    bool finished_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(finished);

    if (!x_base_format || !finished_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_moe_gating_top_k_softmax do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_moe_gating_top_k_softmax(x, finished, k);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_moe_init_routing(const at::Tensor & x, const at::Tensor & row_idx, const at::Tensor & expert_idx, int64_t active_num){
    bool x_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x);
    bool row_idx_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(row_idx);
    bool expert_idx_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(expert_idx);

    if (!x_base_format || !row_idx_base_format || !expert_idx_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_moe_init_routing do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_moe_init_routing(x, row_idx, expert_idx, active_num);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_multi_head_attention_v2_grad(const at::Tensor & attention_score_grad, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & softmax_log_max_sum, const at::Tensor & attention_score, const c10::optional<at::Tensor> & atten_mask, const c10::optional<at::Tensor> & alibi_mask, double scale, int64_t head_num, c10::string_view input_layout, double keep_prob, int64_t pre_tokens, int64_t next_tokens, int64_t seed, int64_t offset, int64_t numels, bool gen_mask_parallel, bool sync){
    bool attention_score_grad_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(attention_score_grad);
    bool query_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(query);
    bool key_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(key);
    bool value_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(value);
    bool softmax_log_max_sum_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(softmax_log_max_sum);
    bool attention_score_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(attention_score);
    bool atten_mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(atten_mask);
    bool alibi_mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(alibi_mask);

    if (!attention_score_grad_base_format || !query_base_format || !key_base_format || !value_base_format || !softmax_log_max_sum_base_format || !attention_score_base_format || !atten_mask_base_format || !alibi_mask_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_multi_head_attention_v2_grad do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_multi_head_attention_v2_grad(attention_score_grad, query, key, value, softmax_log_max_sum, attention_score, atten_mask, alibi_mask, scale, head_num, input_layout, keep_prob, pre_tokens, next_tokens, seed, offset, numels, gen_mask_parallel, sync);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_nms_with_mask(const at::Tensor & input, const at::Scalar & iou_threshold){
    return acl_op::npu_nms_with_mask(input, iou_threshold);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_nsa_select_attention(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & topk_indices, double scale_value, int64_t head_num, int64_t select_block_size, int64_t select_block_count, const c10::optional<at::Tensor> & atten_mask, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_seq_kvlen){
    bool query_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(query);
    bool key_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(key);
    bool value_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(value);
    bool topk_indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(topk_indices);
    bool atten_mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(atten_mask);

    if (!query_base_format || !key_base_format || !value_base_format || !topk_indices_base_format || !atten_mask_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_nsa_select_attention do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_nsa_select_attention(query, key, value, topk_indices, scale_value, head_num, select_block_size, select_block_count, atten_mask, actual_seq_qlen, actual_seq_kvlen);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_nsa_select_attention_grad(const at::Tensor & grad, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & attention_out, const at::Tensor & softmax_max, const at::Tensor & softmax_sum, const at::Tensor & topk_indices, double scale_value, int64_t head_num, int64_t select_block_size, int64_t select_block_count, const c10::optional<at::Tensor> & atten_mask, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_seq_kvlen){
    bool grad_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad);
    bool query_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(query);
    bool key_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(key);
    bool value_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(value);
    bool attention_out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(attention_out);
    bool softmax_max_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(softmax_max);
    bool softmax_sum_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(softmax_sum);
    bool topk_indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(topk_indices);
    bool atten_mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(atten_mask);

    if (!grad_base_format || !query_base_format || !key_base_format || !value_base_format || !attention_out_base_format || !softmax_max_base_format || !softmax_sum_base_format || !topk_indices_base_format || !atten_mask_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_nsa_select_attention_grad do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_nsa_select_attention_grad(grad, query, key, value, attention_out, softmax_max, softmax_sum, topk_indices, scale_value, head_num, select_block_size, select_block_count, atten_mask, actual_seq_qlen, actual_seq_kvlen);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_rotary_mul_backward(const at::Tensor & grad, const at::Tensor & self, const at::Tensor & r1, const at::Tensor & r2, c10::string_view rotary_mode){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool r1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(r1);
    bool r2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(r2);

    ASCEND_LOGI("npu_rotary_mul_backward exec with jit compile: %d, grad is internal format: %d, self is internal format: %d, r1 is internal format: %d, r2 is internal format: %d",
                !is_jit_disable, !grad_base_format, !self_base_format, !r1_base_format, !r2_base_format);
    if (is_jit_disable && grad_base_format && self_base_format && r1_base_format && r2_base_format) {
        return op_api::npu_rotary_mul_backward(grad, self, r1, r2, rotary_mode);
    } else {
        return acl_op::npu_rotary_mul_backward(grad, self, r1, r2, rotary_mode);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_sparse_paged_fusion_attention_symint(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & blocktable, const at::Tensor & l1_cent, const at::Tensor & block_ids, const at::Tensor & total_seq_len, const c10::optional<at::Tensor> & pse_shift, const c10::optional<at::Tensor> & attention_mask, at::OptionalSymIntArrayRef actual_seq_lengths, const c10::optional<at::Tensor> & dequant_scale1, const c10::optional<at::Tensor> & quant_scale1, const c10::optional<at::Tensor> & dequant_scale2, const c10::optional<at::Tensor> & quant_scale2, const c10::optional<at::Tensor> & quant_offset2, const c10::optional<at::Tensor> & antiquant_scale, const c10::optional<at::Tensor> & antiquant_offset, const c10::optional<at::Tensor> & kv_padding_size, int64_t num_heads, double scale_value, c10::string_view input_layout, int64_t num_key_value_heads, int64_t block_size, int64_t inner_precise){
    bool query_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(query);
    bool key_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(key);
    bool value_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(value);
    bool blocktable_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(blocktable);
    bool l1_cent_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(l1_cent);
    bool block_ids_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(block_ids);
    bool total_seq_len_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(total_seq_len);
    bool pse_shift_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(pse_shift);
    bool attention_mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(attention_mask);
    bool dequant_scale1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(dequant_scale1);
    bool quant_scale1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(quant_scale1);
    bool dequant_scale2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(dequant_scale2);
    bool quant_scale2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(quant_scale2);
    bool quant_offset2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(quant_offset2);
    bool antiquant_scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(antiquant_scale);
    bool antiquant_offset_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(antiquant_offset);
    bool kv_padding_size_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(kv_padding_size);

    if (!query_base_format || !key_base_format || !value_base_format || !blocktable_base_format || !l1_cent_base_format || !block_ids_base_format || !total_seq_len_base_format || !pse_shift_base_format || !attention_mask_base_format || !dequant_scale1_base_format || !quant_scale1_base_format || !dequant_scale2_base_format || !quant_scale2_base_format || !quant_offset2_base_format || !antiquant_scale_base_format || !antiquant_offset_base_format || !kv_padding_size_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_sparse_paged_fusion_attention_symint do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_sparse_paged_fusion_attention_symint(query, key, value, blocktable, l1_cent, block_ids, total_seq_len, pse_shift, attention_mask, actual_seq_lengths, dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2, antiquant_scale, antiquant_offset, kv_padding_size, num_heads, scale_value, input_layout, num_key_value_heads, block_size, inner_precise);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> slow_conv_dilated2d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, ::std::array<bool,3> output_mask){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);

    ASCEND_LOGI("slow_conv_dilated2d_backward exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, weight is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !weight_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && weight_base_format) {
        return op_api::slow_conv_dilated2d_backward(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
    } else {
        return acl_op::slow_conv_dilated2d_backward(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> slow_conv_transpose2d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, ::std::array<bool,3> output_mask){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);

    ASCEND_LOGI("slow_conv_transpose2d_backward exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, weight is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !weight_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && weight_base_format) {
        return op_api::slow_conv_transpose2d_backward(grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, output_mask);
    } else {
        return acl_op::slow_conv_transpose2d_backward(grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, output_mask);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> unique_consecutive(const at::Tensor & self, bool return_inverse, bool return_counts, c10::optional<int64_t> dim){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("unique_consecutive exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::unique_consecutive(self, return_inverse, return_counts, dim);
    } else {
        return acl_op::unique_consecutive(self, return_inverse, return_counts, dim);
    }
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> unique_dim(const at::Tensor & self, int64_t dim, bool sorted, bool return_inverse, bool return_counts){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("unique_dim exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::unique_dim(self, dim, sorted, return_inverse, return_counts);
    } else {
        return acl_op::unique_dim(self, dim, sorted, return_inverse, return_counts);
    }
}
::std::tuple<at::Tensor,at::Tensor,int64_t,int64_t,int64_t> npu_multi_head_attention_v2(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const c10::optional<at::Tensor> & atten_mask, const c10::optional<at::Tensor> & alibi_mask, double scale, int64_t head_num, c10::string_view input_layout, double keep_prob, int64_t pre_tokens, int64_t next_tokens, bool gen_mask_parallel, bool sync){
    bool query_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(query);
    bool key_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(key);
    bool value_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(value);
    bool atten_mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(atten_mask);
    bool alibi_mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(alibi_mask);

    if (!query_base_format || !key_base_format || !value_base_format || !atten_mask_base_format || !alibi_mask_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_multi_head_attention_v2 do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_multi_head_attention_v2(query, key, value, atten_mask, alibi_mask, scale, head_num, input_layout, keep_prob, pre_tokens, next_tokens, gen_mask_parallel, sync);
}
::std::tuple<at::Tensor,at::Tensor> _aminmax(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("_aminmax exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::_aminmax(self);
    } else {
        return acl_op::_aminmax(self);
    }
}
::std::tuple<at::Tensor,at::Tensor> _aminmax(const at::Tensor & self, int64_t dim, bool keepdim){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("_aminmax exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::_aminmax(self, dim, keepdim);
    } else {
        return acl_op::_aminmax(self, dim, keepdim);
    }
}
::std::tuple<at::Tensor,at::Tensor> _conv_depthwise2d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, ::std::array<bool,2> output_mask){
    return acl_op::_conv_depthwise2d_backward(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
}
::std::tuple<at::Tensor,at::Tensor> _ctc_loss(const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, int64_t blank, bool zero_infinity){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool log_probs_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(log_probs);
    bool targets_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(targets);

    ASCEND_LOGI("_ctc_loss exec with jit compile: %d, log_probs is internal format: %d, targets is internal format: %d",
                !is_jit_disable, !log_probs_base_format, !targets_base_format);
    if (is_jit_disable && log_probs_base_format && targets_base_format) {
        return op_api::_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, zero_infinity);
    } else {
        return acl_op::_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, zero_infinity);
    }
}
::std::tuple<at::Tensor,at::Tensor> _dropout_with_byte_mask(const at::Tensor & self, double p){
    return acl_op::_dropout_with_byte_mask(self, p);
}
::std::tuple<at::Tensor,at::Tensor> _native_multi_head_attention(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, int64_t embed_dim, int64_t num_head, const at::Tensor & qkv_weight, const at::Tensor & qkv_bias, const at::Tensor & proj_weight, const at::Tensor & proj_bias, const c10::optional<at::Tensor> & mask, bool need_weights, bool average_attn_weights, c10::optional<int64_t> mask_type){
    bool query_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(query);
    bool key_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(key);
    bool value_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(value);
    bool qkv_weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(qkv_weight);
    bool qkv_bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(qkv_bias);
    bool proj_weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(proj_weight);
    bool proj_bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(proj_bias);
    bool mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mask);

    if (!query_base_format || !key_base_format || !value_base_format || !qkv_weight_base_format || !qkv_bias_base_format || !proj_weight_base_format || !proj_bias_base_format || !mask_base_format) {
        TORCH_CHECK(false,
            "Current operator _native_multi_head_attention do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_native_multi_head_attention(query, key, value, embed_dim, num_head, qkv_weight, qkv_bias, proj_weight, proj_bias, mask, need_weights, average_attn_weights, mask_type);
}
::std::tuple<at::Tensor,at::Tensor> _npu_ciou(const at::Tensor & self, const at::Tensor & gtboxes, bool trans, bool is_cross, int64_t mode, bool atan_sub_flag){
    return acl_op::_npu_ciou(self, gtboxes, trans, is_cross, mode, atan_sub_flag);
}
::std::tuple<at::Tensor,at::Tensor> _npu_dropout(const at::Tensor & self, double p){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("_npu_dropout exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::_npu_dropout(self, p);
    } else {
        return acl_op::_npu_dropout(self, p);
    }
}
::std::tuple<at::Tensor,at::Tensor> _pack_padded_sequence(const at::Tensor & input, const at::Tensor & lengths, bool batch_first){
    return acl_op::_pack_padded_sequence(input, lengths, batch_first);
}
::std::tuple<at::Tensor,at::Tensor> _pad_packed_sequence(const at::Tensor & data, const at::Tensor & batch_sizes, bool batch_first, const at::Scalar & padding_value, int64_t total_length){
    return acl_op::_pad_packed_sequence(data, batch_sizes, batch_first, padding_value, total_length);
}
::std::tuple<at::Tensor,at::Tensor> _prelu_kernel_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);

    ASCEND_LOGI("_prelu_kernel_backward exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, weight is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !weight_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && weight_base_format) {
        return op_api::_prelu_kernel_backward(grad_output, self, weight);
    } else {
        return acl_op::_prelu_kernel_backward(grad_output, self, weight);
    }
}
::std::tuple<at::Tensor,at::Tensor> _thnn_fused_gru_cell(const at::Tensor & input_gates, const at::Tensor & hidden_gates, const at::Tensor & hx, const c10::optional<at::Tensor> & input_bias, const c10::optional<at::Tensor> & hidden_bias){
    bool input_gates_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input_gates);
    bool hidden_gates_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(hidden_gates);
    bool hx_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(hx);
    bool input_bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input_bias);
    bool hidden_bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(hidden_bias);

    if (!input_gates_base_format || !hidden_gates_base_format || !hx_base_format || !input_bias_base_format || !hidden_bias_base_format) {
        TORCH_CHECK(false,
            "Current operator _thnn_fused_gru_cell do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_thnn_fused_gru_cell(input_gates, hidden_gates, hx, input_bias, hidden_bias);
}
::std::tuple<at::Tensor,at::Tensor> _unique(const at::Tensor & self, bool sorted, bool return_inverse){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("_unique exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::_unique(self, sorted, return_inverse);
    } else {
        return acl_op::_unique(self, sorted, return_inverse);
    }
}
::std::tuple<at::Tensor,at::Tensor> adaptive_max_pool2d(const at::Tensor & self, at::IntArrayRef output_size){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("adaptive_max_pool2d exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::adaptive_max_pool2d(self, output_size);
    } else {
        return acl_op::adaptive_max_pool2d(self, output_size);
    }
}
::std::tuple<at::Tensor,at::Tensor> adaptive_max_pool3d(const at::Tensor & self, at::IntArrayRef output_size){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator adaptive_max_pool3d do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::adaptive_max_pool3d(self, output_size);
}
::std::tuple<at::Tensor,at::Tensor> aminmax(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator aminmax do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::aminmax(self, dim, keepdim);
}
::std::tuple<at::Tensor,at::Tensor> batch_norm_gather_stats_update(const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, double momentum, double eps, const at::Tensor & counts){
    return acl_op::batch_norm_gather_stats_update(input, mean, invstd, running_mean, running_var, momentum, eps, counts);
}
::std::tuple<at::Tensor,at::Tensor> batch_norm_gather_stats_with_counts(const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, double momentum, double eps, const at::Tensor & counts){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool mean_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mean);
    bool invstd_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(invstd);
    bool running_mean_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(running_mean);
    bool running_var_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(running_var);
    bool counts_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(counts);

    ASCEND_LOGI("batch_norm_gather_stats_with_counts exec with jit compile: %d, input is internal format: %d, mean is internal format: %d, invstd is internal format: %d, running_mean is internal format: %d, running_var is internal format: %d, counts is internal format: %d",
                !is_jit_disable, !input_base_format, !mean_base_format, !invstd_base_format, !running_mean_base_format, !running_var_base_format, !counts_base_format);
    if (is_jit_disable && input_base_format && mean_base_format && invstd_base_format && running_mean_base_format && running_var_base_format && counts_base_format) {
        return op_api::batch_norm_gather_stats_with_counts(input, mean, invstd, running_mean, running_var, momentum, eps, counts);
    } else {
        return acl_op::batch_norm_gather_stats_with_counts(input, mean, invstd, running_mean, running_var, momentum, eps, counts);
    }
}
::std::tuple<at::Tensor,at::Tensor> batch_norm_reduce(const at::Tensor & input, double eps){
    return acl_op::batch_norm_reduce(input, eps);
}
::std::tuple<at::Tensor,at::Tensor> batch_norm_stats(const at::Tensor & input, double eps){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);

    ASCEND_LOGI("batch_norm_stats exec with jit compile: %d, input is internal format: %d",
                !is_jit_disable, !input_base_format);
    if (is_jit_disable && input_base_format) {
        return op_api::batch_norm_stats(input, eps);
    } else {
        return acl_op::batch_norm_stats(input, eps);
    }
}
::std::tuple<at::Tensor,at::Tensor> grid_sampler_2d_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners, ::std::array<bool,2> output_mask){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool grid_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grid);

    ASCEND_LOGI("grid_sampler_2d_backward exec with jit compile: %d, grad_output is internal format: %d, input is internal format: %d, grid is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !input_base_format, !grid_base_format);
    if (is_jit_disable && grad_output_base_format && input_base_format && grid_base_format) {
        return op_api::grid_sampler_2d_backward(grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask);
    } else {
        return acl_op::grid_sampler_2d_backward(grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask);
    }
}
::std::tuple<at::Tensor,at::Tensor> grid_sampler_3d_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners, ::std::array<bool,2> output_mask){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool grid_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grid);

    ASCEND_LOGI("grid_sampler_3d_backward exec with jit compile: %d, grad_output is internal format: %d, input is internal format: %d, grid is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !input_base_format, !grid_base_format);
    if (is_jit_disable && grad_output_base_format && input_base_format && grid_base_format) {
        return op_api::grid_sampler_3d_backward(grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask);
    } else {
        return acl_op::grid_sampler_3d_backward(grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask);
    }
}
::std::tuple<at::Tensor,at::Tensor> gru(const at::Tensor & input, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first){
    return acl_op::gru(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}
::std::tuple<at::Tensor,at::Tensor> kthvalue(const at::Tensor & self, int64_t k, at::Dimname dim, bool keepdim){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("kthvalue exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::kthvalue(self, k, dim, keepdim);
    } else {
        return acl_op::kthvalue(self, k, dim, keepdim);
    }
}
::std::tuple<at::Tensor,at::Tensor> kthvalue(const at::Tensor & self, int64_t k, int64_t dim, bool keepdim){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("kthvalue exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::kthvalue(self, k, dim, keepdim);
    } else {
        return acl_op::kthvalue(self, k, dim, keepdim);
    }
}
::std::tuple<at::Tensor,at::Tensor> linalg_qr(const at::Tensor & self, c10::string_view mode){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("linalg_qr exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::linalg_qr(self, mode);
    } else {
        return acl_op::linalg_qr(self, mode);
    }
}
::std::tuple<at::Tensor,at::Tensor> log_sigmoid_forward(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("log_sigmoid_forward exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::log_sigmoid_forward(self);
    } else {
        return acl_op::log_sigmoid_forward(self);
    }
}
::std::tuple<at::Tensor,at::Tensor> lstm_cell(const at::Tensor & input, at::TensorList hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const c10::optional<at::Tensor> & b_ih, const c10::optional<at::Tensor> & b_hh){
    return acl_op::lstm_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
}
::std::tuple<at::Tensor,at::Tensor> matmul_backward(const at::Tensor & grad, const at::Tensor & self, const at::Tensor & other, ::std::array<bool,2> mask){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("matmul_backward exec with jit compile: %d, grad is internal format: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !grad_base_format, !self_base_format, !other_base_format);
    if (is_jit_disable && grad_base_format && self_base_format && other_base_format) {
        return op_api::matmul_backward(grad, self, other, mask);
    } else {
        return acl_op::matmul_backward(grad, self, other, mask);
    }
}
::std::tuple<at::Tensor,at::Tensor> max(const at::Tensor & self, at::Dimname dim, bool keepdim){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("max exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::max(self, dim, keepdim);
    } else {
        return acl_op::max(self, dim, keepdim);
    }
}
::std::tuple<at::Tensor,at::Tensor> max(const at::Tensor & self, int64_t dim, bool keepdim){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("max exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::max(self, dim, keepdim);
    } else {
        return acl_op::max(self, dim, keepdim);
    }
}
::std::tuple<at::Tensor,at::Tensor> max_pool2d_with_indices(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("max_pool2d_with_indices exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::max_pool2d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
    } else {
        return acl_op::max_pool2d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
    }
}
::std::tuple<at::Tensor,at::Tensor> max_pool3d_with_indices(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("max_pool3d_with_indices exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::max_pool3d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
    } else {
        return acl_op::max_pool3d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
    }
}
::std::tuple<at::Tensor,at::Tensor> median(const at::Tensor & self, int64_t dim, bool keepdim){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("median exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::median(self, dim, keepdim);
    } else {
        return acl_op::median(self, dim, keepdim);
    }
}
::std::tuple<at::Tensor,at::Tensor> min(const at::Tensor & self, at::Dimname dim, bool keepdim){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("min exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::min(self, dim, keepdim);
    } else {
        return acl_op::min(self, dim, keepdim);
    }
}
::std::tuple<at::Tensor,at::Tensor> min(const at::Tensor & self, int64_t dim, bool keepdim){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("min exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::min(self, dim, keepdim);
    } else {
        return acl_op::min(self, dim, keepdim);
    }
}
::std::tuple<at::Tensor,at::Tensor> multilabel_margin_loss_forward(const at::Tensor & self, const at::Tensor & target, int64_t reduction){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target);

    ASCEND_LOGI("multilabel_margin_loss_forward exec with jit compile: %d, self is internal format: %d, target is internal format: %d",
                !is_jit_disable, !self_base_format, !target_base_format);
    if (is_jit_disable && self_base_format && target_base_format) {
        return op_api::multilabel_margin_loss_forward(self, target, reduction);
    } else {
        return acl_op::multilabel_margin_loss_forward(self, target, reduction);
    }
}
::std::tuple<at::Tensor,at::Tensor> nanmedian(const at::Tensor & self, int64_t dim, bool keepdim){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("nanmedian exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::nanmedian(self, dim, keepdim);
    } else {
        return acl_op::nanmedian(self, dim, keepdim);
    }
}
::std::tuple<at::Tensor,at::Tensor> native_dropout(const at::Tensor & input, double p, c10::optional<bool> train){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);

    ASCEND_LOGI("native_dropout exec with jit compile: %d, input is internal format: %d",
                !is_jit_disable, !input_base_format);
    if (is_jit_disable && input_base_format) {
        return op_api::native_dropout(input, p, train);
    } else {
        return acl_op::native_dropout(input, p, train);
    }
}
::std::tuple<at::Tensor,at::Tensor> nll_loss2d_forward(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);

    ASCEND_LOGI("nll_loss2d_forward exec with jit compile: %d, self is internal format: %d, target is internal format: %d, weight is internal format: %d",
                !is_jit_disable, !self_base_format, !target_base_format, !weight_base_format);
    if (is_jit_disable && self_base_format && target_base_format && weight_base_format) {
        return op_api::nll_loss2d_forward(self, target, weight, reduction, ignore_index);
    } else {
        return acl_op::nll_loss2d_forward(self, target, weight, reduction, ignore_index);
    }
}
::std::tuple<at::Tensor,at::Tensor> nll_loss_forward(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);

    ASCEND_LOGI("nll_loss_forward exec with jit compile: %d, self is internal format: %d, target is internal format: %d, weight is internal format: %d",
                !is_jit_disable, !self_base_format, !target_base_format, !weight_base_format);
    if (is_jit_disable && self_base_format && target_base_format && weight_base_format) {
        return op_api::nll_loss_forward(self, target, weight, reduction, ignore_index);
    } else {
        return acl_op::nll_loss_forward(self, target, weight, reduction, ignore_index);
    }
}
::std::tuple<at::Tensor,at::Tensor> npu_all_gather_base_mm(const at::Tensor & self, const at::Tensor & x2, c10::string_view hcom, int64_t world_size, const c10::optional<at::Tensor> & bias, int64_t gather_index, bool gather_output, int64_t comm_turn){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool x2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x2);
    bool bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias);

    if (!self_base_format || !x2_base_format || !bias_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_all_gather_base_mm do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_all_gather_base_mm(self, x2, hcom, world_size, bias, gather_index, gather_output, comm_turn);
}
::std::tuple<at::Tensor,at::Tensor> npu_apply_rotary_pos_emb(const at::Tensor & query, const at::Tensor & key, const at::Tensor & cos, const at::Tensor & sin, c10::string_view layout){
    bool query_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(query);
    bool key_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(key);
    bool cos_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(cos);
    bool sin_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(sin);

    if (!query_base_format || !key_base_format || !cos_base_format || !sin_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_apply_rotary_pos_emb do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_apply_rotary_pos_emb(query, key, cos, sin, layout);
}
::std::tuple<at::Tensor,at::Tensor> npu_cent_select(const at::Tensor & query, const at::Tensor & l1_cent, const at::Tensor & block_ids, const at::Tensor & block_table, const at::Tensor & seq_len){
    bool query_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(query);
    bool l1_cent_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(l1_cent);
    bool block_ids_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(block_ids);
    bool block_table_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(block_table);
    bool seq_len_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(seq_len);

    if (!query_base_format || !l1_cent_base_format || !block_ids_base_format || !block_table_base_format || !seq_len_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_cent_select do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_cent_select(query, l1_cent, block_ids, block_table, seq_len);
}
::std::tuple<at::Tensor,at::Tensor> npu_ciou_backward(const at::Tensor & grad, const at::Tensor & bboxes, const at::Tensor & gtboxes, const c10::optional<at::Tensor> & atan_sub, bool trans, bool is_cross, int64_t mode){
    return acl_op::npu_ciou_backward(grad, bboxes, gtboxes, atan_sub, trans, is_cross, mode);
}
::std::tuple<at::Tensor,at::Tensor> npu_deformable_conv2d(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & offset, const c10::optional<at::Tensor> & bias, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, int64_t deformable_groups, bool modulated){
    return acl_op::npu_deformable_conv2d(input, weight, offset, bias, kernel_size, stride, padding, dilation, groups, deformable_groups, modulated);
}
::std::tuple<at::Tensor,at::Tensor> npu_dequant_swiglu_quant(const at::Tensor & x, const c10::optional<at::Tensor> & weight_scale, const c10::optional<at::Tensor> & activation_scale, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & quant_scale, const c10::optional<at::Tensor> & quant_offset, const c10::optional<at::Tensor> & group_index, bool activate_left, int64_t quant_mode){
    bool x_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x);
    bool weight_scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight_scale);
    bool activation_scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(activation_scale);
    bool bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias);
    bool quant_scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(quant_scale);
    bool quant_offset_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(quant_offset);
    bool group_index_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(group_index);

    if (!x_base_format || !weight_scale_base_format || !activation_scale_base_format || !bias_base_format || !quant_scale_base_format || !quant_offset_base_format || !group_index_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_dequant_swiglu_quant do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_dequant_swiglu_quant(x, weight_scale, activation_scale, bias, quant_scale, quant_offset, group_index, activate_left, quant_mode);
}
::std::tuple<at::Tensor,at::Tensor> npu_diou_backward(const at::Tensor & grad, const at::Tensor & bboxes, const at::Tensor & gtboxes, bool trans, bool is_cross, int64_t mode){
    return acl_op::npu_diou_backward(grad, bboxes, gtboxes, trans, is_cross, mode);
}
::std::tuple<at::Tensor,at::Tensor> npu_dropout_do_mask(const at::Tensor & self, const at::Tensor & mask, double p){
    return acl_op::npu_dropout_do_mask(self, mask, p);
}
::std::tuple<at::Tensor,at::Tensor> npu_dropout_with_add_softmax_backward(const at::Tensor & grad, const at::Tensor & mask, const at::Tensor & softmax_out, const at::Scalar & alpha, double prob, int64_t dim){
    return acl_op::npu_dropout_with_add_softmax_backward(grad, mask, softmax_out, alpha, prob, dim);
}
::std::tuple<at::Tensor,at::Tensor> npu_dynamic_quant(const at::Tensor & input, const c10::optional<at::Tensor> & smooth_scales, const c10::optional<at::Tensor> & group_index, c10::optional<at::ScalarType> dst_type){
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool smooth_scales_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(smooth_scales);
    bool group_index_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(group_index);

    if (!input_base_format || !smooth_scales_base_format || !group_index_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_dynamic_quant do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_dynamic_quant(input, smooth_scales, group_index, dst_type);
}
::std::tuple<at::Tensor,at::Tensor> npu_fused_infer_attention_score_symint(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const c10::optional<at::Tensor> & pse_shift, const c10::optional<at::Tensor> & atten_mask, at::OptionalSymIntArrayRef actual_seq_lengths, at::OptionalSymIntArrayRef actual_seq_lengths_kv, const c10::optional<at::Tensor> & dequant_scale1, const c10::optional<at::Tensor> & quant_scale1, const c10::optional<at::Tensor> & dequant_scale2, const c10::optional<at::Tensor> & quant_scale2, const c10::optional<at::Tensor> & quant_offset2, const c10::optional<at::Tensor> & antiquant_scale, const c10::optional<at::Tensor> & antiquant_offset, const c10::optional<at::Tensor> & key_antiquant_scale, const c10::optional<at::Tensor> & key_antiquant_offset, const c10::optional<at::Tensor> & value_antiquant_scale, const c10::optional<at::Tensor> & value_antiquant_offset, const c10::optional<at::Tensor> & block_table, const c10::optional<at::Tensor> & query_padding_size, const c10::optional<at::Tensor> & kv_padding_size, const c10::optional<at::Tensor> & key_shared_prefix, const c10::optional<at::Tensor> & value_shared_prefix, at::OptionalSymIntArrayRef actual_shared_prefix_len, const c10::optional<at::Tensor> & query_rope, const c10::optional<at::Tensor> & key_rope, const c10::optional<at::Tensor> & key_rope_antiquant_scale, int64_t num_heads, double scale, int64_t pre_tokens, int64_t next_tokens, c10::string_view input_layout, int64_t num_key_value_heads, int64_t sparse_mode, int64_t inner_precise, int64_t block_size, int64_t antiquant_mode, int64_t key_antiquant_mode, int64_t value_antiquant_mode, bool softmax_lse_flag){
    return op_api::npu_fused_infer_attention_score_symint(query, key, value, pse_shift, atten_mask, actual_seq_lengths, actual_seq_lengths_kv, dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2, antiquant_scale, antiquant_offset, key_antiquant_scale, key_antiquant_offset, value_antiquant_scale, value_antiquant_offset, block_table, query_padding_size, kv_padding_size, key_shared_prefix, value_shared_prefix, actual_shared_prefix_len, query_rope, key_rope, key_rope_antiquant_scale, num_heads, scale, pre_tokens, next_tokens, input_layout, num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, key_antiquant_mode, value_antiquant_mode, softmax_lse_flag);
}
::std::tuple<at::Tensor,at::Tensor> npu_geglu(const at::Tensor & self, int64_t dim, int64_t approximate, bool activate_left){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_geglu do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_geglu(self, dim, approximate, activate_left);
}
::std::tuple<at::Tensor,at::Tensor> npu_gemma_rms_norm(const at::Tensor & self, const at::Tensor & gamma, double epsilon){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool gamma_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(gamma);

    if (!self_base_format || !gamma_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_gemma_rms_norm do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_gemma_rms_norm(self, gamma, epsilon);
}
::std::tuple<at::Tensor,at::Tensor> npu_giou_backward(const at::Tensor & grad, const at::Tensor & bboxes, const at::Tensor & gtboxes, bool trans, bool is_cross, int64_t mode){
    return acl_op::npu_giou_backward(grad, bboxes, gtboxes, trans, is_cross, mode);
}
::std::tuple<at::Tensor,at::Tensor> npu_gmm_alltoallv(const at::Tensor & gmm_x, const at::Tensor & gmm_weight, c10::string_view hcom, int64_t ep_world_size, at::IntArrayRef send_counts, at::IntArrayRef recv_counts, const c10::optional<at::Tensor> & send_counts_tensor, const c10::optional<at::Tensor> & recv_counts_tensor, const c10::optional<at::Tensor> & mm_x, const c10::optional<at::Tensor> & mm_weight, bool trans_gmm_weight, bool trans_mm_weight){
    bool gmm_x_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(gmm_x);
    bool gmm_weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(gmm_weight);
    bool send_counts_tensor_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(send_counts_tensor);
    bool recv_counts_tensor_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(recv_counts_tensor);
    bool mm_x_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mm_x);
    bool mm_weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mm_weight);

    if (!gmm_x_base_format || !gmm_weight_base_format || !send_counts_tensor_base_format || !recv_counts_tensor_base_format || !mm_x_base_format || !mm_weight_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_gmm_alltoallv do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_gmm_alltoallv(gmm_x, gmm_weight, hcom, ep_world_size, send_counts, recv_counts, send_counts_tensor, recv_counts_tensor, mm_x, mm_weight, trans_gmm_weight, trans_mm_weight);
}
::std::tuple<at::Tensor,at::Tensor> npu_ifmr(const at::Tensor & data, const at::Tensor & data_min, const at::Tensor & data_max, const at::Tensor & cumsum, double min_percentile, double max_percentile, double search_start, double search_end, double search_step, bool with_offset){
    return acl_op::npu_ifmr(data, data_min, data_max, cumsum, min_percentile, max_percentile, search_start, search_end, search_step, with_offset);
}
::std::tuple<at::Tensor,at::Tensor> npu_kronecker_quant(const at::Tensor & x, const at::Tensor & kronecker_p1, const at::Tensor & kronecker_p2, c10::optional<double> clip_ratio, c10::optional<at::ScalarType> dst_dtype){
    bool x_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x);
    bool kronecker_p1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(kronecker_p1);
    bool kronecker_p2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(kronecker_p2);

    if (!x_base_format || !kronecker_p1_base_format || !kronecker_p2_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_kronecker_quant do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_kronecker_quant(x, kronecker_p1, kronecker_p2, clip_ratio, dst_dtype);
}
::std::tuple<at::Tensor,at::Tensor> npu_linear_backward(const at::Tensor & grad, const at::Tensor & input, const at::Tensor & weight){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad);
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);

    ASCEND_LOGI("npu_linear_backward exec with jit compile: %d, grad is internal format: %d, input is internal format: %d, weight is internal format: %d",
                !is_jit_disable, !grad_base_format, !input_base_format, !weight_base_format);
    if (is_jit_disable && grad_base_format && input_base_format && weight_base_format) {
        return op_api::npu_linear_backward(grad, input, weight);
    } else {
        return acl_op::npu_linear_backward(grad, input, weight);
    }
}
::std::tuple<at::Tensor,at::Tensor> npu_max(const at::Tensor & self, at::Dimname dim, bool keepdim){
    return acl_op::npu_max(self, dim, keepdim);
}
::std::tuple<at::Tensor,at::Tensor> npu_max(const at::Tensor & self, int64_t dim, bool keepdim){
    return acl_op::npu_max(self, dim, keepdim);
}
::std::tuple<at::Tensor,at::Tensor> npu_min(const at::Tensor & self, at::Dimname dim, bool keepdim){
    return acl_op::npu_min(self, dim, keepdim);
}
::std::tuple<at::Tensor,at::Tensor> npu_min(const at::Tensor & self, int64_t dim, bool keepdim){
    return acl_op::npu_min(self, dim, keepdim);
}
::std::tuple<at::Tensor,at::Tensor> npu_moe_token_permute(const at::Tensor & tokens, const at::Tensor & indices, c10::optional<int64_t> num_out_tokens, bool padded_mode){
    bool tokens_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tokens);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);

    if (!tokens_base_format || !indices_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_moe_token_permute do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_moe_token_permute(tokens, indices, num_out_tokens, padded_mode);
}
::std::tuple<at::Tensor,at::Tensor> npu_mrope(const at::Tensor & positions, const at::Tensor & query, const at::Tensor & key, const at::Tensor & cos_sin_cache, int64_t head_size, at::OptionalIntArrayRef mrope_section, c10::optional<c10::string_view> rotary_mode){
    bool positions_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(positions);
    bool query_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(query);
    bool key_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(key);
    bool cos_sin_cache_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(cos_sin_cache);

    if (!positions_base_format || !query_base_format || !key_base_format || !cos_sin_cache_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_mrope do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_mrope(positions, query, key, cos_sin_cache, head_size, mrope_section, rotary_mode);
}
::std::tuple<at::Tensor,at::Tensor> npu_nms_rotated(const at::Tensor & self, const at::Tensor & scores, double iou_threshold, double scores_threshold, int64_t max_output_size, int64_t mode){
    return acl_op::npu_nms_rotated(self, scores, iou_threshold, scores_threshold, max_output_size, mode);
}
::std::tuple<at::Tensor,at::Tensor> npu_nms_v4(const at::Tensor & self, const at::Tensor & scores, const at::Scalar & max_output_size, const at::Tensor & iou_threshold, const at::Tensor & scores_threshold, bool pad_to_max_output_size){
    return acl_op::npu_nms_v4(self, scores, max_output_size, iou_threshold, scores_threshold, pad_to_max_output_size);
}
::std::tuple<at::Tensor,at::Tensor> npu_nsa_compress_attention_infer(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, double scale_value, int64_t head_num, int64_t key_value_head_num, int64_t select_block_size, int64_t select_block_count, int64_t page_block_size, int64_t compress_block_size, int64_t compress_stride, const c10::optional<at::Tensor> & atten_mask, const c10::optional<at::Tensor> & block_table, const c10::optional<at::Tensor> & topk_mask, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_cmp_seq_kvlen, at::OptionalIntArrayRef actual_sel_seq_kvlen){
    bool query_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(query);
    bool key_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(key);
    bool value_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(value);
    bool atten_mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(atten_mask);
    bool block_table_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(block_table);
    bool topk_mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(topk_mask);

    if (!query_base_format || !key_base_format || !value_base_format || !atten_mask_base_format || !block_table_base_format || !topk_mask_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_nsa_compress_attention_infer do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_nsa_compress_attention_infer(query, key, value, scale_value, head_num, key_value_head_num, select_block_size, select_block_count, page_block_size, compress_block_size, compress_stride, atten_mask, block_table, topk_mask, actual_seq_qlen, actual_cmp_seq_kvlen, actual_sel_seq_kvlen);
}
::std::tuple<at::Tensor,at::Tensor> npu_nsa_compress_grad(const at::Tensor & grad, const at::Tensor & input, const at::Tensor & weight, int64_t compress_block_size, int64_t compress_stride, at::OptionalIntArrayRef actual_seq_len){
    bool grad_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad);
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);

    if (!grad_base_format || !input_base_format || !weight_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_nsa_compress_grad do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_nsa_compress_grad(grad, input, weight, compress_block_size, compress_stride, actual_seq_len);
}
::std::tuple<at::Tensor,at::Tensor> npu_random_choice_with_mask(const at::Tensor & x, int64_t count, int64_t seed, int64_t seed2){
    return acl_op::npu_random_choice_with_mask(x, count, seed, seed2);
}
::std::tuple<at::Tensor,at::Tensor> npu_rms_norm(const at::Tensor & self, const at::Tensor & gamma, double epsilon){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool gamma_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(gamma);

    ASCEND_LOGI("npu_rms_norm exec with jit compile: %d, self is internal format: %d, gamma is internal format: %d",
                !is_jit_disable, !self_base_format, !gamma_base_format);
    if (is_jit_disable && self_base_format && gamma_base_format) {
        return op_api::npu_rms_norm(self, gamma, epsilon);
    } else {
        return acl_op::npu_rms_norm(self, gamma, epsilon);
    }
}
::std::tuple<at::Tensor,at::Tensor> npu_rms_norm_backward(const at::Tensor & dy, const at::Tensor & self, const at::Tensor & gamma, const at::Tensor & rstd){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool dy_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(dy);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool gamma_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(gamma);
    bool rstd_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(rstd);

    ASCEND_LOGI("npu_rms_norm_backward exec with jit compile: %d, dy is internal format: %d, self is internal format: %d, gamma is internal format: %d, rstd is internal format: %d",
                !is_jit_disable, !dy_base_format, !self_base_format, !gamma_base_format, !rstd_base_format);
    if (is_jit_disable && dy_base_format && self_base_format && gamma_base_format && rstd_base_format) {
        return op_api::npu_rms_norm_backward(dy, self, gamma, rstd);
    } else {
        return acl_op::npu_rms_norm_backward(dy, self, gamma, rstd);
    }
}
::std::tuple<at::Tensor,at::Tensor> slogdet(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("slogdet exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::slogdet(self);
    } else {
        return acl_op::slogdet(self);
    }
}
::std::tuple<at::Tensor,at::Tensor> sort(const at::Tensor & self, at::Dimname dim, bool descending){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("sort exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::sort(self, dim, descending);
    } else {
        return acl_op::sort(self, dim, descending);
    }
}
::std::tuple<at::Tensor,at::Tensor> sort(const at::Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator sort do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::sort(self, stable, dim, descending);
}
::std::tuple<at::Tensor,at::Tensor> sort(const at::Tensor & self, int64_t dim, bool descending){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("sort exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::sort(self, dim, descending);
    } else {
        return acl_op::sort(self, dim, descending);
    }
}
::std::tuple<at::Tensor,at::Tensor> std_mean(const at::Tensor & self, at::OptionalIntArrayRef dim, const c10::optional<at::Scalar> & correction, bool keepdim){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("std_mean exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::std_mean(self, dim, correction, keepdim);
    } else {
        return acl_op::std_mean(self, dim, correction, keepdim);
    }
}
::std::tuple<at::Tensor,at::Tensor> topk(const at::Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("topk exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::topk(self, k, dim, largest, sorted);
    } else {
        return acl_op::topk(self, k, dim, largest, sorted);
    }
}
::std::tuple<at::Tensor,at::Tensor> var_mean(const at::Tensor & self, at::OptionalIntArrayRef dim, const c10::optional<at::Scalar> & correction, bool keepdim){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("var_mean exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::var_mean(self, dim, correction, keepdim);
    } else {
        return acl_op::var_mean(self, dim, correction, keepdim);
    }
}
::std::vector<at::Tensor> _foreach_abs(at::TensorList tensors){
    bool tensors_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors);

    if (!tensors_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_abs do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_abs(tensors);
}
::std::vector<at::Tensor> _foreach_acos(at::TensorList tensors){
    bool tensors_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors);

    if (!tensors_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_acos do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_acos(tensors);
}
::std::vector<at::Tensor> _foreach_add(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars){
    bool tensors_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors);

    if (!tensors_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_add do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_add(tensors, scalars);
}
::std::vector<at::Tensor> _foreach_add(at::TensorList tensors, const at::Scalar & scalar){
    bool tensors_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors);

    if (!tensors_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_add do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_add(tensors, scalar);
}
::std::vector<at::Tensor> _foreach_add(at::TensorList tensors1, at::TensorList tensors2, const at::Scalar & alpha){
    bool tensors1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors1);
    bool tensors2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors2);

    if (!tensors1_base_format || !tensors2_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_add do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_add(tensors1, tensors2, alpha);
}
::std::vector<at::Tensor> _foreach_addcdiv(at::TensorList input, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars){
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool tensor1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor1);
    bool tensor2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor2);

    if (!input_base_format || !tensor1_base_format || !tensor2_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_addcdiv do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_addcdiv(input, tensor1, tensor2, scalars);
}
::std::vector<at::Tensor> _foreach_addcdiv(at::TensorList input, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value){
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool tensor1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor1);
    bool tensor2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor2);

    if (!input_base_format || !tensor1_base_format || !tensor2_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_addcdiv do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_addcdiv(input, tensor1, tensor2, value);
}
::std::vector<at::Tensor> _foreach_addcdiv(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Tensor & scalars){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool tensor1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor1);
    bool tensor2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor2);
    bool scalars_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(scalars);

    if (!self_base_format || !tensor1_base_format || !tensor2_base_format || !scalars_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_addcdiv do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_addcdiv(self, tensor1, tensor2, scalars);
}
::std::vector<at::Tensor> _foreach_addcmul(at::TensorList input, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars){
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool tensor1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor1);
    bool tensor2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor2);

    if (!input_base_format || !tensor1_base_format || !tensor2_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_addcmul do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_addcmul(input, tensor1, tensor2, scalars);
}
::std::vector<at::Tensor> _foreach_addcmul(at::TensorList input, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value){
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool tensor1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor1);
    bool tensor2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor2);

    if (!input_base_format || !tensor1_base_format || !tensor2_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_addcmul do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_addcmul(input, tensor1, tensor2, value);
}
::std::vector<at::Tensor> _foreach_addcmul(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Tensor & scalars){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool tensor1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor1);
    bool tensor2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor2);
    bool scalars_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(scalars);

    if (!self_base_format || !tensor1_base_format || !tensor2_base_format || !scalars_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_addcmul do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_addcmul(self, tensor1, tensor2, scalars);
}
::std::vector<at::Tensor> _foreach_asin(at::TensorList tensors){
    bool tensors_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors);

    if (!tensors_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_asin do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_asin(tensors);
}
::std::vector<at::Tensor> _foreach_atan(at::TensorList tensors){
    bool tensors_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors);

    if (!tensors_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_atan do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_atan(tensors);
}
::std::vector<at::Tensor> _foreach_ceil(at::TensorList self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_ceil do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_ceil(self);
}
::std::vector<at::Tensor> _foreach_clamp_max(at::TensorList self, at::ArrayRef<at::Scalar> scalars){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_clamp_max do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_clamp_max(self, scalars);
}
::std::vector<at::Tensor> _foreach_clamp_max(at::TensorList self, at::TensorList other){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    if (!self_base_format || !other_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_clamp_max do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_clamp_max(self, other);
}
::std::vector<at::Tensor> _foreach_clamp_max(at::TensorList self, const at::Scalar & scalar){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_clamp_max do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_clamp_max(self, scalar);
}
::std::vector<at::Tensor> _foreach_clamp_min(at::TensorList self, at::ArrayRef<at::Scalar> scalars){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_clamp_min do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_clamp_min(self, scalars);
}
::std::vector<at::Tensor> _foreach_clamp_min(at::TensorList self, at::TensorList other){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    if (!self_base_format || !other_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_clamp_min do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_clamp_min(self, other);
}
::std::vector<at::Tensor> _foreach_clamp_min(at::TensorList self, const at::Scalar & scalar){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_clamp_min do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_clamp_min(self, scalar);
}
::std::vector<at::Tensor> _foreach_cos(at::TensorList tensors){
    bool tensors_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors);

    if (!tensors_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_cos do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_cos(tensors);
}
::std::vector<at::Tensor> _foreach_cosh(at::TensorList tensors){
    bool tensors_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors);

    if (!tensors_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_cosh do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_cosh(tensors);
}
::std::vector<at::Tensor> _foreach_div(at::TensorList self, const at::Scalar & scalar){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_div do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_div(self, scalar);
}
::std::vector<at::Tensor> _foreach_div(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars){
    bool tensors_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors);

    if (!tensors_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_div do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_div(tensors, scalars);
}
::std::vector<at::Tensor> _foreach_div(at::TensorList tensors1, at::TensorList tensors2){
    bool tensors1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors1);
    bool tensors2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors2);

    if (!tensors1_base_format || !tensors2_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_div do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_div(tensors1, tensors2);
}
::std::vector<at::Tensor> _foreach_erf(at::TensorList tensors){
    bool tensors_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors);

    if (!tensors_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_erf do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_erf(tensors);
}
::std::vector<at::Tensor> _foreach_erfc(at::TensorList tensors){
    bool tensors_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors);

    if (!tensors_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_erfc do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_erfc(tensors);
}
::std::vector<at::Tensor> _foreach_exp(at::TensorList tensors){
    bool tensors_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors);

    if (!tensors_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_exp do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_exp(tensors);
}
::std::vector<at::Tensor> _foreach_expm1(at::TensorList tensors){
    bool tensors_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors);

    if (!tensors_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_expm1 do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_expm1(tensors);
}
::std::vector<at::Tensor> _foreach_floor(at::TensorList self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_floor do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_floor(self);
}
::std::vector<at::Tensor> _foreach_frac(at::TensorList self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_frac do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_frac(self);
}
::std::vector<at::Tensor> _foreach_lerp(at::TensorList self, at::TensorList tensors1, at::TensorList weights){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool tensors1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors1);
    bool weights_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weights);

    if (!self_base_format || !tensors1_base_format || !weights_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_lerp do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_lerp(self, tensors1, weights);
}
::std::vector<at::Tensor> _foreach_lerp(at::TensorList self, at::TensorList tensors1, const at::Scalar & weight){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool tensors1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors1);

    if (!self_base_format || !tensors1_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_lerp do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_lerp(self, tensors1, weight);
}
::std::vector<at::Tensor> _foreach_log(at::TensorList tensors){
    bool tensors_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors);

    if (!tensors_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_log do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_log(tensors);
}
::std::vector<at::Tensor> _foreach_log10(at::TensorList tensors){
    bool tensors_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors);

    if (!tensors_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_log10 do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_log10(tensors);
}
::std::vector<at::Tensor> _foreach_log1p(at::TensorList tensors){
    bool tensors_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors);

    if (!tensors_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_log1p do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_log1p(tensors);
}
::std::vector<at::Tensor> _foreach_log2(at::TensorList tensors){
    bool tensors_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors);

    if (!tensors_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_log2 do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_log2(tensors);
}
::std::vector<at::Tensor> _foreach_maximum(at::TensorList self, at::ArrayRef<at::Scalar> scalars){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_maximum do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_maximum(self, scalars);
}
::std::vector<at::Tensor> _foreach_maximum(at::TensorList self, at::TensorList other){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    if (!self_base_format || !other_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_maximum do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_maximum(self, other);
}
::std::vector<at::Tensor> _foreach_maximum(at::TensorList self, const at::Scalar & scalar){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_maximum do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_maximum(self, scalar);
}
::std::vector<at::Tensor> _foreach_minimum(at::TensorList self, at::ArrayRef<at::Scalar> scalars){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_minimum do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_minimum(self, scalars);
}
::std::vector<at::Tensor> _foreach_minimum(at::TensorList self, at::TensorList other){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    if (!self_base_format || !other_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_minimum do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_minimum(self, other);
}
::std::vector<at::Tensor> _foreach_minimum(at::TensorList self, const at::Scalar & scalar){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_minimum do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_minimum(self, scalar);
}
::std::vector<at::Tensor> _foreach_mul(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars){
    bool tensors_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors);

    if (!tensors_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_mul do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_mul(tensors, scalars);
}
::std::vector<at::Tensor> _foreach_mul(at::TensorList tensors, const at::Scalar & scalar){
    bool tensors_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors);

    if (!tensors_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_mul do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_mul(tensors, scalar);
}
::std::vector<at::Tensor> _foreach_mul(at::TensorList tensors1, at::TensorList tensors2){
    bool tensors1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors1);
    bool tensors2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors2);

    if (!tensors1_base_format || !tensors2_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_mul do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_mul(tensors1, tensors2);
}
::std::vector<at::Tensor> _foreach_neg(at::TensorList tensors){
    bool tensors_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors);

    if (!tensors_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_neg do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_neg(tensors);
}
::std::vector<at::Tensor> _foreach_norm(at::TensorList tensors, const at::Scalar & ord, c10::optional<at::ScalarType> dtype){
    bool tensors_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors);

    if (!tensors_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_norm do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_norm(tensors, ord, dtype);
}
::std::vector<at::Tensor> _foreach_pow(at::TensorList self, at::ArrayRef<at::Scalar> exponent){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_pow do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_pow(self, exponent);
}
::std::vector<at::Tensor> _foreach_pow(at::TensorList self, at::TensorList exponent){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool exponent_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(exponent);

    if (!self_base_format || !exponent_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_pow do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_pow(self, exponent);
}
::std::vector<at::Tensor> _foreach_pow(at::TensorList tensors, const at::Scalar & scalar){
    bool tensors_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors);

    if (!tensors_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_pow do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_pow(tensors, scalar);
}
::std::vector<at::Tensor> _foreach_pow(const at::Scalar & self, at::TensorList exponent){
    bool exponent_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(exponent);

    if (!exponent_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_pow do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_pow(self, exponent);
}
::std::vector<at::Tensor> _foreach_reciprocal(at::TensorList tensors){
    bool tensors_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors);

    if (!tensors_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_reciprocal do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_reciprocal(tensors);
}
::std::vector<at::Tensor> _foreach_round(at::TensorList self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_round do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_round(self);
}
::std::vector<at::Tensor> _foreach_sigmoid(at::TensorList tensors){
    bool tensors_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors);

    if (!tensors_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_sigmoid do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_sigmoid(tensors);
}
::std::vector<at::Tensor> _foreach_sign(at::TensorList self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_sign do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_sign(self);
}
::std::vector<at::Tensor> _foreach_sin(at::TensorList tensors){
    bool tensors_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors);

    if (!tensors_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_sin do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_sin(tensors);
}
::std::vector<at::Tensor> _foreach_sinh(at::TensorList tensors){
    bool tensors_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors);

    if (!tensors_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_sinh do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_sinh(tensors);
}
::std::vector<at::Tensor> _foreach_sqrt(at::TensorList tensors){
    bool tensors_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors);

    if (!tensors_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_sqrt do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_sqrt(tensors);
}
::std::vector<at::Tensor> _foreach_sub(at::TensorList self, const at::Scalar & scalar){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_sub do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_sub(self, scalar);
}
::std::vector<at::Tensor> _foreach_sub(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars){
    bool tensors_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors);

    if (!tensors_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_sub do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_sub(tensors, scalars);
}
::std::vector<at::Tensor> _foreach_sub(at::TensorList tensors1, at::TensorList tensors2, const at::Scalar & alpha){
    bool tensors1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors1);
    bool tensors2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors2);

    if (!tensors1_base_format || !tensors2_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_sub do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_sub(tensors1, tensors2, alpha);
}
::std::vector<at::Tensor> _foreach_tan(at::TensorList tensors){
    bool tensors_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors);

    if (!tensors_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_tan do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_tan(tensors);
}
::std::vector<at::Tensor> _foreach_tanh(at::TensorList tensors){
    bool tensors_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors);

    if (!tensors_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_tanh do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_tanh(tensors);
}
::std::vector<at::Tensor> _foreach_trunc(at::TensorList self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_trunc do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_trunc(self);
}
::std::vector<at::Tensor> npu_fused_attention_layernorm_qkv_fwd(const at::Tensor & x, const at::Tensor & kernel_query, const at::Tensor & kernel_key, const at::Tensor & kernel_value, const at::Tensor & gamma, const at::Tensor & beta, const c10::optional<at::Tensor> & bias_query, const c10::optional<at::Tensor> & bias_key, const c10::optional<at::Tensor> & bias_value, int64_t seq_len, int64_t num_heads, double eps){
    return acl_op::npu_fused_attention_layernorm_qkv_fwd(x, kernel_query, kernel_key, kernel_value, gamma, beta, bias_query, bias_key, bias_value, seq_len, num_heads, eps);
}
::std::vector<at::Tensor> npu_fused_attention_qkv_grad(const at::Tensor & grad_output_query, const at::Tensor & grad_output_key, const at::Tensor & grad_output_value, const at::Tensor & query_kernel, const at::Tensor & key_kernel, const at::Tensor & value_kernel, const at::Tensor & hidden_states, const at::Tensor & grad_output_ln){
    return acl_op::npu_fused_attention_qkv_grad(grad_output_query, grad_output_key, grad_output_value, query_kernel, key_kernel, value_kernel, hidden_states, grad_output_ln);
}
::std::vector<at::Tensor> npu_grouped_matmul(at::TensorList x, at::TensorList weight, c10::optional<at::TensorList> bias, c10::optional<at::TensorList> scale, c10::optional<at::TensorList> offset, c10::optional<at::TensorList> antiquant_scale, c10::optional<at::TensorList> antiquant_offset, c10::optional<at::TensorList> per_token_scale, at::OptionalIntArrayRef group_list, c10::optional<at::TensorList> activation_input, c10::optional<at::TensorList> activation_quant_scale, c10::optional<at::TensorList> activation_quant_offset, c10::optional<int64_t> split_item, c10::optional<int64_t> group_type, c10::optional<int64_t> group_list_type, c10::optional<int64_t> act_type, c10::optional<at::ScalarType> output_dtype){
    bool x_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias);
    bool scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(scale);
    bool offset_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(offset);
    bool antiquant_scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(antiquant_scale);
    bool antiquant_offset_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(antiquant_offset);
    bool per_token_scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(per_token_scale);
    bool activation_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(activation_input);
    bool activation_quant_scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(activation_quant_scale);
    bool activation_quant_offset_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(activation_quant_offset);

    if (!x_base_format || !weight_base_format || !bias_base_format || !scale_base_format || !offset_base_format || !antiquant_scale_base_format || !antiquant_offset_base_format || !per_token_scale_base_format || !activation_input_base_format || !activation_quant_scale_base_format || !activation_quant_offset_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_grouped_matmul do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_grouped_matmul(x, weight, bias, scale, offset, antiquant_scale, antiquant_offset, per_token_scale, group_list, activation_input, activation_quant_scale, activation_quant_offset, split_item, group_type, group_list_type, act_type, output_dtype);
}
::std::vector<at::Tensor> npu_grouped_matmul(at::TensorList x, at::TensorList weight, c10::optional<at::TensorList> bias, c10::optional<at::TensorList> scale, c10::optional<at::TensorList> offset, c10::optional<at::TensorList> antiquant_scale, c10::optional<at::TensorList> antiquant_offset, c10::optional<at::TensorList> per_token_scale, const c10::optional<at::Tensor> & group_list, c10::optional<at::TensorList> activation_input, c10::optional<at::TensorList> activation_quant_scale, c10::optional<at::TensorList> activation_quant_offset, c10::optional<int64_t> split_item, c10::optional<int64_t> group_type, c10::optional<int64_t> group_list_type, c10::optional<int64_t> act_type, at::OptionalIntArrayRef tuning_config, c10::optional<at::ScalarType> output_dtype){
    return op_api::npu_grouped_matmul(x, weight, bias, scale, offset, antiquant_scale, antiquant_offset, per_token_scale, group_list, activation_input, activation_quant_scale, activation_quant_offset, split_item, group_type, group_list_type, act_type, tuning_config, output_dtype);
}
::std::vector<at::Tensor> npu_scatter_list(at::TensorList self, const at::Tensor & indices, const at::Tensor & updates, const c10::optional<at::Tensor> & mask, c10::string_view reduce, int64_t axis){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);
    bool updates_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(updates);
    bool mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mask);

    if (!self_base_format || !indices_base_format || !updates_base_format || !mask_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_scatter_list do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_scatter_list(self, indices, updates, mask, reduce, axis);
}
::std::vector<at::Tensor> where(const at::Tensor & condition){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool condition_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(condition);

    ASCEND_LOGI("where exec with jit compile: %d, condition is internal format: %d",
                !is_jit_disable, !condition_base_format);
    if (is_jit_disable && condition_base_format) {
        return op_api::where(condition);
    } else {
        return acl_op::where(condition);
    }
}
at::Tensor & __ilshift__(at::Tensor & self, const at::Scalar & other){
    return acl_op::__ilshift__(self, other);
}
at::Tensor & __ilshift__(at::Tensor & self, const at::Tensor & other){
    return acl_op::__ilshift__(self, other);
}
at::Tensor & __ior__(at::Tensor & self, const at::Scalar & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("__ior__ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::__ior__(self, other);
    } else {
        return acl_op::__ior__(self, other);
    }
}
at::Tensor & __ior__(at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("__ior__ exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::__ior__(self, other);
    } else {
        return acl_op::__ior__(self, other);
    }
}
at::Tensor & __irshift__(at::Tensor & self, const at::Scalar & other){
    return acl_op::__irshift__(self, other);
}
at::Tensor & __irshift__(at::Tensor & self, const at::Tensor & other){
    return acl_op::__irshift__(self, other);
}
at::Tensor & _add_relu_(at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("_add_relu_ exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::_add_relu_(self, other, alpha);
    } else {
        return acl_op::_add_relu_(self, other, alpha);
    }
}
at::Tensor & _add_relu_out(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("_add_relu_out exec with jit compile: %d, self is internal format: %d, other is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && other_base_format && out_base_format) {
        return op_api::_add_relu_out(self, other, alpha, out);
    } else {
        return acl_op::_add_relu_out(self, other, alpha, out);
    }
}
at::Tensor & _amp_update_scale_(at::Tensor & self, at::Tensor & growth_tracker, const at::Tensor & found_inf, double growth_factor, double backoff_factor, int64_t growth_interval){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool growth_tracker_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(growth_tracker);
    bool found_inf_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(found_inf);

    if (!self_base_format || !growth_tracker_base_format || !found_inf_base_format) {
        TORCH_CHECK(false,
            "Current operator _amp_update_scale_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_amp_update_scale_(self, growth_tracker, found_inf, growth_factor, backoff_factor, growth_interval);
}
at::Tensor & _fft_c2c_out(const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, bool forward, at::Tensor & out){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    if (!self_base_format || !out_base_format) {
        TORCH_CHECK(false,
            "Current operator _fft_c2c_out do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_fft_c2c_out(self, dim, normalization, forward, out);
}
at::Tensor & _fft_c2r_out(const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, int64_t last_dim_size, at::Tensor & out){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    if (!self_base_format || !out_base_format) {
        TORCH_CHECK(false,
            "Current operator _fft_c2r_out do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_fft_c2r_out(self, dim, normalization, last_dim_size, out);
}
at::Tensor & _fft_r2c_out(const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, bool onesided, at::Tensor & out){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    if (!self_base_format || !out_base_format) {
        TORCH_CHECK(false,
            "Current operator _fft_r2c_out do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_fft_r2c_out(self, dim, normalization, onesided, out);
}
at::Tensor & _index_put_impl_(at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate, bool unsafe){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);
    bool values_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(values);

    ASCEND_LOGI("_index_put_impl_ exec with jit compile: %d, self is internal format: %d, indices is internal format: %d, values is internal format: %d",
                !is_jit_disable, !self_base_format, !indices_base_format, !values_base_format);
    if (is_jit_disable && self_base_format && indices_base_format && values_base_format) {
        return op_api::_index_put_impl_(self, indices, values, accumulate, unsafe);
    } else {
        return acl_op::_index_put_impl_(self, indices, values, accumulate, unsafe);
    }
}
at::Tensor & _log_softmax_backward_data_out(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, at::ScalarType input_dtype, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(output);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("_log_softmax_backward_data_out exec with jit compile: %d, grad_output is internal format: %d, output is internal format: %d, out is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !output_base_format, !out_base_format);
    if (is_jit_disable && grad_output_base_format && output_base_format && out_base_format) {
        return op_api::_log_softmax_backward_data_out(grad_output, output, dim, input_dtype, out);
    } else {
        return acl_op::_log_softmax_backward_data_out(grad_output, output, dim, input_dtype, out);
    }
}
at::Tensor & _log_softmax_out(const at::Tensor & self, int64_t dim, bool half_to_float, at::Tensor & out){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    if (!self_base_format || !out_base_format) {
        TORCH_CHECK(false,
            "Current operator _log_softmax_out do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_log_softmax_out(self, dim, half_to_float, out);
}
at::Tensor & _slow_conv2d_forward_out(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & output){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias);
    bool output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(output);

    ASCEND_LOGI("_slow_conv2d_forward_out exec with jit compile: %d, self is internal format: %d, weight is internal format: %d, bias is internal format: %d, output is internal format: %d",
                !is_jit_disable, !self_base_format, !weight_base_format, !bias_base_format, !output_base_format);
    if (is_jit_disable && self_base_format && weight_base_format && bias_base_format && output_base_format) {
        return op_api::_slow_conv2d_forward_out(self, weight, kernel_size, bias, stride, padding, output);
    } else {
        return acl_op::_slow_conv2d_forward_out(self, weight, kernel_size, bias, stride, padding, output);
    }
}
at::Tensor & _softmax_backward_data_out(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, at::ScalarType input_dtype, at::Tensor & grad_input){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(output);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    ASCEND_LOGI("_softmax_backward_data_out exec with jit compile: %d, grad_output is internal format: %d, output is internal format: %d, grad_input is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !output_base_format, !grad_input_base_format);
    if (is_jit_disable && grad_output_base_format && output_base_format && grad_input_base_format) {
        return op_api::_softmax_backward_data_out(grad_output, output, dim, input_dtype, grad_input);
    } else {
        return acl_op::_softmax_backward_data_out(grad_output, output, dim, input_dtype, grad_input);
    }
}
at::Tensor & _softmax_out(const at::Tensor & self, int64_t dim, bool half_to_float, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("_softmax_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::_softmax_out(self, dim, half_to_float, out);
    } else {
        return acl_op::_softmax_out(self, dim, half_to_float, out);
    }
}
at::Tensor & _upsample_bicubic2d_aa_backward_out(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input){
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    if (!grad_output_base_format || !grad_input_base_format) {
        TORCH_CHECK(false,
            "Current operator _upsample_bicubic2d_aa_backward_out do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_upsample_bicubic2d_aa_backward_out(grad_output, output_size, input_size, align_corners, scales_h, scales_w, grad_input);
}
at::Tensor & _upsample_bicubic2d_aa_out(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    if (!self_base_format || !out_base_format) {
        TORCH_CHECK(false,
            "Current operator _upsample_bicubic2d_aa_out do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_upsample_bicubic2d_aa_out(self, output_size, align_corners, scales_h, scales_w, out);
}
at::Tensor & _upsample_bilinear2d_aa_backward_out(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input){
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    if (!grad_output_base_format || !grad_input_base_format) {
        TORCH_CHECK(false,
            "Current operator _upsample_bilinear2d_aa_backward_out do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_upsample_bilinear2d_aa_backward_out(grad_output, output_size, input_size, align_corners, scales_h, scales_w, grad_input);
}
at::Tensor & _upsample_bilinear2d_aa_out(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    if (!self_base_format || !out_base_format) {
        TORCH_CHECK(false,
            "Current operator _upsample_bilinear2d_aa_out do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_upsample_bilinear2d_aa_out(self, output_size, align_corners, scales_h, scales_w, out);
}
at::Tensor & _upsample_nearest_exact1d_backward_out(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales, at::Tensor & grad_input){
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    if (!grad_output_base_format || !grad_input_base_format) {
        TORCH_CHECK(false,
            "Current operator _upsample_nearest_exact1d_backward_out do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_upsample_nearest_exact1d_backward_out(grad_output, output_size, input_size, scales, grad_input);
}
at::Tensor & _upsample_nearest_exact1d_out(const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales, at::Tensor & out){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    if (!self_base_format || !out_base_format) {
        TORCH_CHECK(false,
            "Current operator _upsample_nearest_exact1d_out do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_upsample_nearest_exact1d_out(self, output_size, scales, out);
}
at::Tensor & _upsample_nearest_exact2d_backward_out(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input){
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    if (!grad_output_base_format || !grad_input_base_format) {
        TORCH_CHECK(false,
            "Current operator _upsample_nearest_exact2d_backward_out do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_upsample_nearest_exact2d_backward_out(grad_output, output_size, input_size, scales_h, scales_w, grad_input);
}
at::Tensor & _upsample_nearest_exact2d_out(const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    if (!self_base_format || !out_base_format) {
        TORCH_CHECK(false,
            "Current operator _upsample_nearest_exact2d_out do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_upsample_nearest_exact2d_out(self, output_size, scales_h, scales_w, out);
}
at::Tensor & _upsample_nearest_exact3d_backward_out(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input){
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    if (!grad_output_base_format || !grad_input_base_format) {
        TORCH_CHECK(false,
            "Current operator _upsample_nearest_exact3d_backward_out do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_upsample_nearest_exact3d_backward_out(grad_output, output_size, input_size, scales_d, scales_h, scales_w, grad_input);
}
at::Tensor & _upsample_nearest_exact3d_out(const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    if (!self_base_format || !out_base_format) {
        TORCH_CHECK(false,
            "Current operator _upsample_nearest_exact3d_out do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_upsample_nearest_exact3d_out(self, output_size, scales_d, scales_h, scales_w, out);
}
at::Tensor & abs_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("abs_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::abs_(self);
    } else {
        return acl_op::abs_(self);
    }
}
at::Tensor & abs_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("abs_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::abs_out(self, out);
    } else {
        return acl_op::abs_out(self, out);
    }
}
at::Tensor & acos_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("acos_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::acos_(self);
    } else {
        return acl_op::acos_(self);
    }
}
at::Tensor & acos_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("acos_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::acos_out(self, out);
    } else {
        return acl_op::acos_out(self, out);
    }
}
at::Tensor & acosh_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("acosh_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::acosh_(self);
    } else {
        return acl_op::acosh_(self);
    }
}
at::Tensor & acosh_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("acosh_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::acosh_out(self, out);
    } else {
        return acl_op::acosh_out(self, out);
    }
}
at::Tensor & adaptive_avg_pool2d_out(const at::Tensor & self, at::IntArrayRef output_size, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("adaptive_avg_pool2d_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::adaptive_avg_pool2d_out(self, output_size, out);
    } else {
        return acl_op::adaptive_avg_pool2d_out(self, output_size, out);
    }
}
at::Tensor & adaptive_avg_pool3d_backward_out(const at::Tensor & grad_output, const at::Tensor & self, at::Tensor & grad_input){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    ASCEND_LOGI("adaptive_avg_pool3d_backward_out exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, grad_input is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !grad_input_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && grad_input_base_format) {
        return op_api::adaptive_avg_pool3d_backward_out(grad_output, self, grad_input);
    } else {
        return acl_op::adaptive_avg_pool3d_backward_out(grad_output, self, grad_input);
    }
}
at::Tensor & adaptive_avg_pool3d_out(const at::Tensor & self, at::IntArrayRef output_size, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("adaptive_avg_pool3d_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::adaptive_avg_pool3d_out(self, output_size, out);
    } else {
        return acl_op::adaptive_avg_pool3d_out(self, output_size, out);
    }
}
at::Tensor & adaptive_max_pool2d_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::Tensor & grad_input){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    ASCEND_LOGI("adaptive_max_pool2d_backward_out exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, indices is internal format: %d, grad_input is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !indices_base_format, !grad_input_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && indices_base_format && grad_input_base_format) {
        return op_api::adaptive_max_pool2d_backward_out(grad_output, self, indices, grad_input);
    } else {
        return acl_op::adaptive_max_pool2d_backward_out(grad_output, self, indices, grad_input);
    }
}
at::Tensor & adaptive_max_pool3d_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::Tensor & grad_input){
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    if (!grad_output_base_format || !self_base_format || !indices_base_format || !grad_input_base_format) {
        TORCH_CHECK(false,
            "Current operator adaptive_max_pool3d_backward_out do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::adaptive_max_pool3d_backward_out(grad_output, self, indices, grad_input);
}
at::Tensor & add_(at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("add_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::add_(self, other, alpha);
    } else {
        return acl_op::add_(self, other, alpha);
    }
}
at::Tensor & add_(at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("add_ exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::add_(self, other, alpha);
    } else {
        return acl_op::add_(self, other, alpha);
    }
}
at::Tensor & add_out(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("add_out exec with jit compile: %d, self is internal format: %d, other is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && other_base_format && out_base_format) {
        return op_api::add_out(self, other, alpha, out);
    } else {
        return acl_op::add_out(self, other, alpha, out);
    }
}
at::Tensor & add_out_sparse(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out){
    return sparse::add_out_sparse(self, other, alpha, out);
}
at::Tensor & addbmm_(at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool batch1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(batch1);
    bool batch2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(batch2);

    ASCEND_LOGI("addbmm_ exec with jit compile: %d, self is internal format: %d, batch1 is internal format: %d, batch2 is internal format: %d",
                !is_jit_disable, !self_base_format, !batch1_base_format, !batch2_base_format);
    if (is_jit_disable && self_base_format && batch1_base_format && batch2_base_format) {
        return op_api::addbmm_(self, batch1, batch2, beta, alpha);
    } else {
        return acl_op::addbmm_(self, batch1, batch2, beta, alpha);
    }
}
at::Tensor & addbmm_out(const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool batch1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(batch1);
    bool batch2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(batch2);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("addbmm_out exec with jit compile: %d, self is internal format: %d, batch1 is internal format: %d, batch2 is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !batch1_base_format, !batch2_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && batch1_base_format && batch2_base_format && out_base_format) {
        return op_api::addbmm_out(self, batch1, batch2, beta, alpha, out);
    } else {
        return acl_op::addbmm_out(self, batch1, batch2, beta, alpha, out);
    }
}
at::Tensor & addcdiv_(at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool tensor1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor1);
    bool tensor2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor2);

    ASCEND_LOGI("addcdiv_ exec with jit compile: %d, self is internal format: %d, tensor1 is internal format: %d, tensor2 is internal format: %d",
                !is_jit_disable, !self_base_format, !tensor1_base_format, !tensor2_base_format);
    if (is_jit_disable && self_base_format && tensor1_base_format && tensor2_base_format) {
        return op_api::addcdiv_(self, tensor1, tensor2, value);
    } else {
        return acl_op::addcdiv_(self, tensor1, tensor2, value);
    }
}
at::Tensor & addcdiv_out(const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool tensor1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor1);
    bool tensor2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor2);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("addcdiv_out exec with jit compile: %d, self is internal format: %d, tensor1 is internal format: %d, tensor2 is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !tensor1_base_format, !tensor2_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && tensor1_base_format && tensor2_base_format && out_base_format) {
        return op_api::addcdiv_out(self, tensor1, tensor2, value, out);
    } else {
        return acl_op::addcdiv_out(self, tensor1, tensor2, value, out);
    }
}
at::Tensor & addcmul_(at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool tensor1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor1);
    bool tensor2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor2);

    ASCEND_LOGI("addcmul_ exec with jit compile: %d, self is internal format: %d, tensor1 is internal format: %d, tensor2 is internal format: %d",
                !is_jit_disable, !self_base_format, !tensor1_base_format, !tensor2_base_format);
    if (is_jit_disable && self_base_format && tensor1_base_format && tensor2_base_format) {
        return op_api::addcmul_(self, tensor1, tensor2, value);
    } else {
        return acl_op::addcmul_(self, tensor1, tensor2, value);
    }
}
at::Tensor & addcmul_out(const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool tensor1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor1);
    bool tensor2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor2);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("addcmul_out exec with jit compile: %d, self is internal format: %d, tensor1 is internal format: %d, tensor2 is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !tensor1_base_format, !tensor2_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && tensor1_base_format && tensor2_base_format && out_base_format) {
        return op_api::addcmul_out(self, tensor1, tensor2, value, out);
    } else {
        return acl_op::addcmul_out(self, tensor1, tensor2, value, out);
    }
}
at::Tensor & addmm_(at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool mat1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mat1);
    bool mat2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mat2);

    ASCEND_LOGI("addmm_ exec with jit compile: %d, self is internal format: %d, mat1 is internal format: %d, mat2 is internal format: %d",
                !is_jit_disable, !self_base_format, !mat1_base_format, !mat2_base_format);
    if (is_jit_disable && self_base_format && mat1_base_format && mat2_base_format) {
        return op_api::addmm_(self, mat1, mat2, beta, alpha);
    } else {
        return acl_op::addmm_(self, mat1, mat2, beta, alpha);
    }
}
at::Tensor & addmm_out(const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool mat1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mat1);
    bool mat2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mat2);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("addmm_out exec with jit compile: %d, self is internal format: %d, mat1 is internal format: %d, mat2 is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !mat1_base_format, !mat2_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && mat1_base_format && mat2_base_format && out_base_format) {
        return op_api::addmm_out(self, mat1, mat2, beta, alpha, out);
    } else {
        return acl_op::addmm_out(self, mat1, mat2, beta, alpha, out);
    }
}
at::Tensor & addmv_(at::Tensor & self, const at::Tensor & mat, const at::Tensor & vec, const at::Scalar & beta, const at::Scalar & alpha){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool mat_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mat);
    bool vec_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(vec);

    ASCEND_LOGI("addmv_ exec with jit compile: %d, self is internal format: %d, mat is internal format: %d, vec is internal format: %d",
                !is_jit_disable, !self_base_format, !mat_base_format, !vec_base_format);
    if (is_jit_disable && self_base_format && mat_base_format && vec_base_format) {
        return op_api::addmv_(self, mat, vec, beta, alpha);
    } else {
        return acl_op::addmv_(self, mat, vec, beta, alpha);
    }
}
at::Tensor & addmv_out(const at::Tensor & self, const at::Tensor & mat, const at::Tensor & vec, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool mat_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mat);
    bool vec_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(vec);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("addmv_out exec with jit compile: %d, self is internal format: %d, mat is internal format: %d, vec is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !mat_base_format, !vec_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && mat_base_format && vec_base_format && out_base_format) {
        return op_api::addmv_out(self, mat, vec, beta, alpha, out);
    } else {
        return acl_op::addmv_out(self, mat, vec, beta, alpha, out);
    }
}
at::Tensor & addr_(at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2, const at::Scalar & beta, const at::Scalar & alpha){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool vec1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(vec1);
    bool vec2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(vec2);

    ASCEND_LOGI("addr_ exec with jit compile: %d, self is internal format: %d, vec1 is internal format: %d, vec2 is internal format: %d",
                !is_jit_disable, !self_base_format, !vec1_base_format, !vec2_base_format);
    if (is_jit_disable && self_base_format && vec1_base_format && vec2_base_format) {
        return op_api::addr_(self, vec1, vec2, beta, alpha);
    } else {
        return acl_op::addr_(self, vec1, vec2, beta, alpha);
    }
}
at::Tensor & addr_out(const at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool vec1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(vec1);
    bool vec2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(vec2);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("addr_out exec with jit compile: %d, self is internal format: %d, vec1 is internal format: %d, vec2 is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !vec1_base_format, !vec2_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && vec1_base_format && vec2_base_format && out_base_format) {
        return op_api::addr_out(self, vec1, vec2, beta, alpha, out);
    } else {
        return acl_op::addr_out(self, vec1, vec2, beta, alpha, out);
    }
}
at::Tensor & all_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("all_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::all_out(self, out);
    } else {
        return acl_op::all_out(self, out);
    }
}
at::Tensor & all_out(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("all_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::all_out(self, dim, keepdim, out);
    } else {
        return acl_op::all_out(self, dim, keepdim, out);
    }
}
at::Tensor & amax_out(const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("amax_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::amax_out(self, dim, keepdim, out);
    } else {
        return acl_op::amax_out(self, dim, keepdim, out);
    }
}
at::Tensor & amin_out(const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("amin_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::amin_out(self, dim, keepdim, out);
    } else {
        return acl_op::amin_out(self, dim, keepdim, out);
    }
}
at::Tensor & angle_out(const at::Tensor & self, at::Tensor & out){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    if (!self_base_format || !out_base_format) {
        TORCH_CHECK(false,
            "Current operator angle_out do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::angle_out(self, out);
}
at::Tensor & any_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("any_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::any_out(self, out);
    } else {
        return acl_op::any_out(self, out);
    }
}
at::Tensor & any_out(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("any_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::any_out(self, dim, keepdim, out);
    } else {
        return acl_op::any_out(self, dim, keepdim, out);
    }
}
at::Tensor & arange_out(const at::Scalar & end, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("arange_out exec with jit compile: %d, out is internal format: %d",
                !is_jit_disable, !out_base_format);
    if (is_jit_disable && out_base_format) {
        return op_api::arange_out(end, out);
    } else {
        return acl_op::arange_out(end, out);
    }
}
at::Tensor & arange_out(const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("arange_out exec with jit compile: %d, out is internal format: %d",
                !is_jit_disable, !out_base_format);
    if (is_jit_disable && out_base_format) {
        return op_api::arange_out(start, end, step, out);
    } else {
        return acl_op::arange_out(start, end, step, out);
    }
}
at::Tensor & argmax_out(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("argmax_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::argmax_out(self, dim, keepdim, out);
    } else {
        return acl_op::argmax_out(self, dim, keepdim, out);
    }
}
at::Tensor & argmin_out(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("argmin_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::argmin_out(self, dim, keepdim, out);
    } else {
        return acl_op::argmin_out(self, dim, keepdim, out);
    }
}
at::Tensor & asin_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("asin_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::asin_(self);
    } else {
        return acl_op::asin_(self);
    }
}
at::Tensor & asin_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("asin_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::asin_out(self, out);
    } else {
        return acl_op::asin_out(self, out);
    }
}
at::Tensor & asinh_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("asinh_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::asinh_(self);
    } else {
        return acl_op::asinh_(self);
    }
}
at::Tensor & asinh_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("asinh_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::asinh_out(self, out);
    } else {
        return acl_op::asinh_out(self, out);
    }
}
at::Tensor & atan2_(at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("atan2_ exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::atan2_(self, other);
    } else {
        return acl_op::atan2_(self, other);
    }
}
at::Tensor & atan2_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("atan2_out exec with jit compile: %d, self is internal format: %d, other is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && other_base_format && out_base_format) {
        return op_api::atan2_out(self, other, out);
    } else {
        return acl_op::atan2_out(self, other, out);
    }
}
at::Tensor & atan_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("atan_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::atan_(self);
    } else {
        return acl_op::atan_(self);
    }
}
at::Tensor & atan_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("atan_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::atan_out(self, out);
    } else {
        return acl_op::atan_out(self, out);
    }
}
at::Tensor & atanh_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("atanh_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::atanh_(self);
    } else {
        return acl_op::atanh_(self);
    }
}
at::Tensor & atanh_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("atanh_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::atanh_out(self, out);
    } else {
        return acl_op::atanh_out(self, out);
    }
}
at::Tensor & avg_pool2d_backward_out(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, at::Tensor & grad_input){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    ASCEND_LOGI("avg_pool2d_backward_out exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, grad_input is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !grad_input_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && grad_input_base_format) {
        return op_api::avg_pool2d_backward_out(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, grad_input);
    } else {
        return acl_op::avg_pool2d_backward_out(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, grad_input);
    }
}
at::Tensor & avg_pool2d_out(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("avg_pool2d_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::avg_pool2d_out(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, out);
    } else {
        return acl_op::avg_pool2d_out(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, out);
    }
}
at::Tensor & avg_pool3d_backward_out(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, at::Tensor & grad_input){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    ASCEND_LOGI("avg_pool3d_backward_out exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, grad_input is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !grad_input_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && grad_input_base_format) {
        return op_api::avg_pool3d_backward_out(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, grad_input);
    } else {
        return acl_op::avg_pool3d_backward_out(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, grad_input);
    }
}
at::Tensor & avg_pool3d_out(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("avg_pool3d_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::avg_pool3d_out(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, out);
    } else {
        return acl_op::avg_pool3d_out(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, out);
    }
}
at::Tensor & baddbmm_(at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool batch1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(batch1);
    bool batch2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(batch2);

    ASCEND_LOGI("baddbmm_ exec with jit compile: %d, self is internal format: %d, batch1 is internal format: %d, batch2 is internal format: %d",
                !is_jit_disable, !self_base_format, !batch1_base_format, !batch2_base_format);
    if (is_jit_disable && self_base_format && batch1_base_format && batch2_base_format) {
        return op_api::baddbmm_(self, batch1, batch2, beta, alpha);
    } else {
        return acl_op::baddbmm_(self, batch1, batch2, beta, alpha);
    }
}
at::Tensor & baddbmm_out(const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool batch1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(batch1);
    bool batch2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(batch2);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("baddbmm_out exec with jit compile: %d, self is internal format: %d, batch1 is internal format: %d, batch2 is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !batch1_base_format, !batch2_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && batch1_base_format && batch2_base_format && out_base_format) {
        return op_api::baddbmm_out(self, batch1, batch2, beta, alpha, out);
    } else {
        return acl_op::baddbmm_out(self, batch1, batch2, beta, alpha, out);
    }
}
at::Tensor & batch_norm_elemt_out(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const at::Tensor & mean, const at::Tensor & invstd, double eps, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias);
    bool mean_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mean);
    bool invstd_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(invstd);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("batch_norm_elemt_out exec with jit compile: %d, input is internal format: %d, weight is internal format: %d, bias is internal format: %d, mean is internal format: %d, invstd is internal format: %d, out is internal format: %d",
                !is_jit_disable, !input_base_format, !weight_base_format, !bias_base_format, !mean_base_format, !invstd_base_format, !out_base_format);
    if (is_jit_disable && input_base_format && weight_base_format && bias_base_format && mean_base_format && invstd_base_format && out_base_format) {
        return op_api::batch_norm_elemt_out(input, weight, bias, mean, invstd, eps, out);
    } else {
        return acl_op::batch_norm_elemt_out(input, weight, bias, mean, invstd, eps, out);
    }
}
at::Tensor & bernoulli_(at::Tensor & self, const at::Tensor & p, c10::optional<at::Generator> generator){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool p_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(p);

    ASCEND_LOGI("bernoulli_ exec with jit compile: %d, self is internal format: %d, p is internal format: %d",
                !is_jit_disable, !self_base_format, !p_base_format);
    if (is_jit_disable && self_base_format && p_base_format) {
        return op_api::bernoulli_(self, p, generator);
    } else {
        return acl_op::bernoulli_(self, p, generator);
    }
}
at::Tensor & bernoulli_(at::Tensor & self, double p, c10::optional<at::Generator> generator){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("bernoulli_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::bernoulli_(self, p, generator);
    } else {
        return acl_op::bernoulli_(self, p, generator);
    }
}
at::Tensor & bernoulli_out(const at::Tensor & self, c10::optional<at::Generator> generator, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("bernoulli_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::bernoulli_out(self, generator, out);
    } else {
        return acl_op::bernoulli_out(self, generator, out);
    }
}
at::Tensor & binary_cross_entropy_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & grad_input){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    ASCEND_LOGI("binary_cross_entropy_backward_out exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, target is internal format: %d, weight is internal format: %d, grad_input is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !target_base_format, !weight_base_format, !grad_input_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && target_base_format && weight_base_format && grad_input_base_format) {
        return op_api::binary_cross_entropy_backward_out(grad_output, self, target, weight, reduction, grad_input);
    } else {
        return acl_op::binary_cross_entropy_backward_out(grad_output, self, target, weight, reduction, grad_input);
    }
}
at::Tensor & binary_cross_entropy_out(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("binary_cross_entropy_out exec with jit compile: %d, self is internal format: %d, target is internal format: %d, weight is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !target_base_format, !weight_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && target_base_format && weight_base_format && out_base_format) {
        return op_api::binary_cross_entropy_out(self, target, weight, reduction, out);
    } else {
        return acl_op::binary_cross_entropy_out(self, target, weight, reduction, out);
    }
}
at::Tensor & bitwise_and_(at::Tensor & self, const at::Scalar & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("bitwise_and_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::bitwise_and_(self, other);
    } else {
        return acl_op::bitwise_and_(self, other);
    }
}
at::Tensor & bitwise_and_(at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("bitwise_and_ exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::bitwise_and_(self, other);
    } else {
        return acl_op::bitwise_and_(self, other);
    }
}
at::Tensor & bitwise_and_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("bitwise_and_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::bitwise_and_out(self, other, out);
    } else {
        return acl_op::bitwise_and_out(self, other, out);
    }
}
at::Tensor & bitwise_and_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("bitwise_and_out exec with jit compile: %d, self is internal format: %d, other is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && other_base_format && out_base_format) {
        return op_api::bitwise_and_out(self, other, out);
    } else {
        return acl_op::bitwise_and_out(self, other, out);
    }
}
at::Tensor & bitwise_not_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("bitwise_not_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::bitwise_not_(self);
    } else {
        return acl_op::bitwise_not_(self);
    }
}
at::Tensor & bitwise_not_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("bitwise_not_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::bitwise_not_out(self, out);
    } else {
        return acl_op::bitwise_not_out(self, out);
    }
}
at::Tensor & bitwise_or_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("bitwise_or_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::bitwise_or_out(self, other, out);
    } else {
        return acl_op::bitwise_or_out(self, other, out);
    }
}
at::Tensor & bitwise_or_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("bitwise_or_out exec with jit compile: %d, self is internal format: %d, other is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && other_base_format && out_base_format) {
        return op_api::bitwise_or_out(self, other, out);
    } else {
        return acl_op::bitwise_or_out(self, other, out);
    }
}
at::Tensor & bitwise_xor_(at::Tensor & self, const at::Scalar & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("bitwise_xor_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::bitwise_xor_(self, other);
    } else {
        return acl_op::bitwise_xor_(self, other);
    }
}
at::Tensor & bitwise_xor_(at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("bitwise_xor_ exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::bitwise_xor_(self, other);
    } else {
        return acl_op::bitwise_xor_(self, other);
    }
}
at::Tensor & bitwise_xor_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("bitwise_xor_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::bitwise_xor_out(self, other, out);
    } else {
        return acl_op::bitwise_xor_out(self, other, out);
    }
}
at::Tensor & bitwise_xor_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("bitwise_xor_out exec with jit compile: %d, self is internal format: %d, other is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && other_base_format && out_base_format) {
        return op_api::bitwise_xor_out(self, other, out);
    } else {
        return acl_op::bitwise_xor_out(self, other, out);
    }
}
at::Tensor & bmm_out(const at::Tensor & self, const at::Tensor & mat2, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool mat2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mat2);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("bmm_out exec with jit compile: %d, self is internal format: %d, mat2 is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !mat2_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && mat2_base_format && out_base_format) {
        return op_api::bmm_out(self, mat2, out);
    } else {
        return acl_op::bmm_out(self, mat2, out);
    }
}
at::Tensor & bucketize_out(const at::Tensor & self, const at::Tensor & boundaries, bool out_int32, bool right, at::Tensor & out){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool boundaries_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(boundaries);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    if (!self_base_format || !boundaries_base_format || !out_base_format) {
        TORCH_CHECK(false,
            "Current operator bucketize_out do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::bucketize_out(self, boundaries, out_int32, right, out);
}
at::Tensor & cat_out(at::TensorList tensors, at::Dimname dim, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool tensors_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("cat_out exec with jit compile: %d, tensors is internal format: %d, out is internal format: %d",
                !is_jit_disable, !tensors_base_format, !out_base_format);
    if (is_jit_disable && tensors_base_format && out_base_format) {
        return op_api::cat_out(tensors, dim, out);
    } else {
        return acl_op::cat_out(tensors, dim, out);
    }
}
at::Tensor & cat_out(const at::ITensorListRef & tensors, int64_t dim, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool tensors_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("cat_out exec with jit compile: %d, tensors is internal format: %d, out is internal format: %d",
                !is_jit_disable, !tensors_base_format, !out_base_format);
    if (is_jit_disable && tensors_base_format && out_base_format) {
        return op_api::cat_out(tensors, dim, out);
    } else {
        return acl_op::cat_out(tensors, dim, out);
    }
}
at::Tensor & ceil_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("ceil_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::ceil_(self);
    } else {
        return acl_op::ceil_(self);
    }
}
at::Tensor & ceil_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("ceil_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::ceil_out(self, out);
    } else {
        return acl_op::ceil_out(self, out);
    }
}
at::Tensor & celu_(at::Tensor & self, const at::Scalar & alpha){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("celu_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::celu_(self, alpha);
    } else {
        return acl_op::celu_(self, alpha);
    }
}
at::Tensor & clamp_(at::Tensor & self, const c10::optional<at::Scalar> & min, const c10::optional<at::Scalar> & max){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("clamp_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::clamp_(self, min, max);
    } else {
        return acl_op::clamp_(self, min, max);
    }
}
at::Tensor & clamp_(at::Tensor & self, const c10::optional<at::Tensor> & min, const c10::optional<at::Tensor> & max){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool min_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(min);
    bool max_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(max);

    ASCEND_LOGI("clamp_ exec with jit compile: %d, self is internal format: %d, min is internal format: %d, max is internal format: %d",
                !is_jit_disable, !self_base_format, !min_base_format, !max_base_format);
    if (is_jit_disable && self_base_format && min_base_format && max_base_format) {
        return op_api::clamp_(self, min, max);
    } else {
        return acl_op::clamp_(self, min, max);
    }
}
at::Tensor & clamp_max_(at::Tensor & self, const at::Scalar & max){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("clamp_max_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::clamp_max_(self, max);
    } else {
        return acl_op::clamp_max_(self, max);
    }
}
at::Tensor & clamp_max_(at::Tensor & self, const at::Tensor & max){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool max_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(max);

    ASCEND_LOGI("clamp_max_ exec with jit compile: %d, self is internal format: %d, max is internal format: %d",
                !is_jit_disable, !self_base_format, !max_base_format);
    if (is_jit_disable && self_base_format && max_base_format) {
        return op_api::clamp_max_(self, max);
    } else {
        return acl_op::clamp_max_(self, max);
    }
}
at::Tensor & clamp_max_out(const at::Tensor & self, const at::Scalar & max, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("clamp_max_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::clamp_max_out(self, max, out);
    } else {
        return acl_op::clamp_max_out(self, max, out);
    }
}
at::Tensor & clamp_max_out(const at::Tensor & self, const at::Tensor & max, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool max_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(max);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("clamp_max_out exec with jit compile: %d, self is internal format: %d, max is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !max_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && max_base_format && out_base_format) {
        return op_api::clamp_max_out(self, max, out);
    } else {
        return acl_op::clamp_max_out(self, max, out);
    }
}
at::Tensor & clamp_min_(at::Tensor & self, const at::Scalar & min){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("clamp_min_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::clamp_min_(self, min);
    } else {
        return acl_op::clamp_min_(self, min);
    }
}
at::Tensor & clamp_min_(at::Tensor & self, const at::Tensor & min){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool min_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(min);

    ASCEND_LOGI("clamp_min_ exec with jit compile: %d, self is internal format: %d, min is internal format: %d",
                !is_jit_disable, !self_base_format, !min_base_format);
    if (is_jit_disable && self_base_format && min_base_format) {
        return op_api::clamp_min_(self, min);
    } else {
        return acl_op::clamp_min_(self, min);
    }
}
at::Tensor & clamp_min_out(const at::Tensor & self, const at::Scalar & min, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("clamp_min_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::clamp_min_out(self, min, out);
    } else {
        return acl_op::clamp_min_out(self, min, out);
    }
}
at::Tensor & clamp_min_out(const at::Tensor & self, const at::Tensor & min, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool min_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(min);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("clamp_min_out exec with jit compile: %d, self is internal format: %d, min is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !min_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && min_base_format && out_base_format) {
        return op_api::clamp_min_out(self, min, out);
    } else {
        return acl_op::clamp_min_out(self, min, out);
    }
}
at::Tensor & clamp_out(const at::Tensor & self, const c10::optional<at::Scalar> & min, const c10::optional<at::Scalar> & max, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("clamp_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::clamp_out(self, min, max, out);
    } else {
        return acl_op::clamp_out(self, min, max, out);
    }
}
at::Tensor & clamp_out(const at::Tensor & self, const c10::optional<at::Tensor> & min, const c10::optional<at::Tensor> & max, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool min_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(min);
    bool max_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(max);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("clamp_out exec with jit compile: %d, self is internal format: %d, min is internal format: %d, max is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !min_base_format, !max_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && min_base_format && max_base_format && out_base_format) {
        return op_api::clamp_out(self, min, max, out);
    } else {
        return acl_op::clamp_out(self, min, max, out);
    }
}
at::Tensor & col2im_out(const at::Tensor & self, at::IntArrayRef output_size, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("col2im_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::col2im_out(self, output_size, kernel_size, dilation, padding, stride, out);
    } else {
        return acl_op::col2im_out(self, output_size, kernel_size, dilation, padding, stride, out);
    }
}
at::Tensor & complex_out(const at::Tensor & real, const at::Tensor & imag, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool real_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(real);
    bool imag_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(imag);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("complex_out exec with jit compile: %d, real is internal format: %d, imag is internal format: %d, out is internal format: %d",
                !is_jit_disable, !real_base_format, !imag_base_format, !out_base_format);
    if (is_jit_disable && real_base_format && imag_base_format && out_base_format) {
        return op_api::complex_out(real, imag, out);
    } else {
        return acl_op::complex_out(real, imag, out);
    }
}
at::Tensor & cos_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("cos_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::cos_(self);
    } else {
        return acl_op::cos_(self);
    }
}
at::Tensor & cos_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("cos_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::cos_out(self, out);
    } else {
        return acl_op::cos_out(self, out);
    }
}
at::Tensor & cosh_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("cosh_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::cosh_(self);
    } else {
        return acl_op::cosh_(self);
    }
}
at::Tensor & cosh_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("cosh_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::cosh_out(self, out);
    } else {
        return acl_op::cosh_out(self, out);
    }
}
at::Tensor & cumprod_(at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype){
    return acl_op::cumprod_(self, dim, dtype);
}
at::Tensor & cumprod_out(const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype, at::Tensor & out){
    return acl_op::cumprod_out(self, dim, dtype, out);
}
at::Tensor & cumprod_out(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("cumprod_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::cumprod_out(self, dim, dtype, out);
    } else {
        return acl_op::cumprod_out(self, dim, dtype, out);
    }
}
at::Tensor & cumsum_out(const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("cumsum_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::cumsum_out(self, dim, dtype, out);
    } else {
        return acl_op::cumsum_out(self, dim, dtype, out);
    }
}
at::Tensor & cumsum_out(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("cumsum_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::cumsum_out(self, dim, dtype, out);
    } else {
        return acl_op::cumsum_out(self, dim, dtype, out);
    }
}
at::Tensor & div_(at::Tensor & self, const at::Scalar & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("div_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::div_(self, other);
    } else {
        return acl_op::div_(self, other);
    }
}
at::Tensor & div_(at::Tensor & self, const at::Scalar & other, c10::optional<c10::string_view> rounding_mode){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("div_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::div_(self, other, rounding_mode);
    } else {
        return acl_op::div_(self, other, rounding_mode);
    }
}
at::Tensor & div_(at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("div_ exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::div_(self, other);
    } else {
        return acl_op::div_(self, other);
    }
}
at::Tensor & div_(at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("div_ exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::div_(self, other, rounding_mode);
    } else {
        return acl_op::div_(self, other, rounding_mode);
    }
}
at::Tensor & div_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("div_out exec with jit compile: %d, self is internal format: %d, other is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && other_base_format && out_base_format) {
        return op_api::div_out(self, other, out);
    } else {
        return acl_op::div_out(self, other, out);
    }
}
at::Tensor & div_out(const at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("div_out exec with jit compile: %d, self is internal format: %d, other is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && other_base_format && out_base_format) {
        return op_api::div_out(self, other, rounding_mode, out);
    } else {
        return acl_op::div_out(self, other, rounding_mode, out);
    }
}
at::Tensor & dot_out(const at::Tensor & self, const at::Tensor & tensor, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool tensor_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("dot_out exec with jit compile: %d, self is internal format: %d, tensor is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !tensor_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && tensor_base_format && out_base_format) {
        return op_api::dot_out(self, tensor, out);
    } else {
        return acl_op::dot_out(self, tensor, out);
    }
}
at::Tensor & dropout_(at::Tensor & self, double p, bool train){
    return op_api::dropout_(self, p, train);
}
at::Tensor & elu_(at::Tensor & self, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("elu_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::elu_(self, alpha, scale, input_scale);
    } else {
        return acl_op::elu_(self, alpha, scale, input_scale);
    }
}
at::Tensor & elu_backward_out(const at::Tensor & grad_output, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale, bool is_result, const at::Tensor & self_or_result, at::Tensor & grad_input){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_or_result_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self_or_result);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    ASCEND_LOGI("elu_backward_out exec with jit compile: %d, grad_output is internal format: %d, self_or_result is internal format: %d, grad_input is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_or_result_base_format, !grad_input_base_format);
    if (is_jit_disable && grad_output_base_format && self_or_result_base_format && grad_input_base_format) {
        return op_api::elu_backward_out(grad_output, alpha, scale, input_scale, is_result, self_or_result, grad_input);
    } else {
        return acl_op::elu_backward_out(grad_output, alpha, scale, input_scale, is_result, self_or_result, grad_input);
    }
}
at::Tensor & elu_out(const at::Tensor & self, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("elu_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::elu_out(self, alpha, scale, input_scale, out);
    } else {
        return acl_op::elu_out(self, alpha, scale, input_scale, out);
    }
}
at::Tensor & embedding_renorm_(at::Tensor & self, const at::Tensor & indices, double max_norm, double norm_type){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);

    ASCEND_LOGI("embedding_renorm_ exec with jit compile: %d, self is internal format: %d, indices is internal format: %d",
                !is_jit_disable, !self_base_format, !indices_base_format);
    if (is_jit_disable && self_base_format && indices_base_format) {
        return op_api::embedding_renorm_(self, indices, max_norm, norm_type);
    } else {
        return acl_op::embedding_renorm_(self, indices, max_norm, norm_type);
    }
}
at::Tensor & eq_(at::Tensor & self, const at::Scalar & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("eq_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::eq_(self, other);
    } else {
        return acl_op::eq_(self, other);
    }
}
at::Tensor & eq_(at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("eq_ exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::eq_(self, other);
    } else {
        return acl_op::eq_(self, other);
    }
}
at::Tensor & eq_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("eq_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::eq_out(self, other, out);
    } else {
        return acl_op::eq_out(self, other, out);
    }
}
at::Tensor & eq_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("eq_out exec with jit compile: %d, self is internal format: %d, other is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && other_base_format && out_base_format) {
        return op_api::eq_out(self, other, out);
    } else {
        return acl_op::eq_out(self, other, out);
    }
}
at::Tensor & erf_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("erf_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::erf_(self);
    } else {
        return acl_op::erf_(self);
    }
}
at::Tensor & erf_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("erf_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::erf_out(self, out);
    } else {
        return acl_op::erf_out(self, out);
    }
}
at::Tensor & erfc_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("erfc_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::erfc_(self);
    } else {
        return acl_op::erfc_(self);
    }
}
at::Tensor & erfc_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("erfc_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::erfc_out(self, out);
    } else {
        return acl_op::erfc_out(self, out);
    }
}
at::Tensor & erfinv_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("erfinv_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::erfinv_(self);
    } else {
        return acl_op::erfinv_(self);
    }
}
at::Tensor & erfinv_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("erfinv_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::erfinv_out(self, out);
    } else {
        return acl_op::erfinv_out(self, out);
    }
}
at::Tensor & exp2_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("exp2_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::exp2_(self);
    } else {
        return acl_op::exp2_(self);
    }
}
at::Tensor & exp2_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("exp2_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::exp2_out(self, out);
    } else {
        return acl_op::exp2_out(self, out);
    }
}
at::Tensor & exp_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("exp_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::exp_(self);
    } else {
        return acl_op::exp_(self);
    }
}
at::Tensor & exp_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("exp_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::exp_out(self, out);
    } else {
        return acl_op::exp_out(self, out);
    }
}
at::Tensor & expm1_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("expm1_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::expm1_(self);
    } else {
        return acl_op::expm1_(self);
    }
}
at::Tensor & expm1_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("expm1_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::expm1_out(self, out);
    } else {
        return acl_op::expm1_out(self, out);
    }
}
at::Tensor & exponential_(at::Tensor & self, double lambd, c10::optional<at::Generator> generator){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator exponential_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::exponential_(self, lambd, generator);
}
at::Tensor & eye_out(int64_t n, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("eye_out exec with jit compile: %d, out is internal format: %d",
                !is_jit_disable, !out_base_format);
    if (is_jit_disable && out_base_format) {
        return op_api::eye_out(n, out);
    } else {
        return acl_op::eye_out(n, out);
    }
}
at::Tensor & eye_out(int64_t n, int64_t m, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("eye_out exec with jit compile: %d, out is internal format: %d",
                !is_jit_disable, !out_base_format);
    if (is_jit_disable && out_base_format) {
        return op_api::eye_out(n, m, out);
    } else {
        return acl_op::eye_out(n, m, out);
    }
}
at::Tensor & fill_(at::Tensor & self, const at::Scalar & value){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("fill_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::fill_(self, value);
    } else {
        return acl_op::fill_(self, value);
    }
}
at::Tensor & fill_(at::Tensor & self, const at::Tensor & value){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool value_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(value);

    ASCEND_LOGI("fill_ exec with jit compile: %d, self is internal format: %d, value is internal format: %d",
                !is_jit_disable, !self_base_format, !value_base_format);
    if (is_jit_disable && self_base_format && value_base_format) {
        return op_api::fill_(self, value);
    } else {
        return acl_op::fill_(self, value);
    }
}
at::Tensor & fill_diagonal_(at::Tensor & self, const at::Scalar & fill_value, bool wrap){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("fill_diagonal_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::fill_diagonal_(self, fill_value, wrap);
    } else {
        return acl_op::fill_diagonal_(self, fill_value, wrap);
    }
}
at::Tensor & floor_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("floor_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::floor_(self);
    } else {
        return acl_op::floor_(self);
    }
}
at::Tensor & floor_divide_(at::Tensor & self, const at::Scalar & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("floor_divide_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::floor_divide_(self, other);
    } else {
        return acl_op::floor_divide_(self, other);
    }
}
at::Tensor & floor_divide_(at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("floor_divide_ exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::floor_divide_(self, other);
    } else {
        return acl_op::floor_divide_(self, other);
    }
}
at::Tensor & floor_divide_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("floor_divide_out exec with jit compile: %d, self is internal format: %d, other is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && other_base_format && out_base_format) {
        return op_api::floor_divide_out(self, other, out);
    } else {
        return acl_op::floor_divide_out(self, other, out);
    }
}
at::Tensor & floor_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("floor_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::floor_out(self, out);
    } else {
        return acl_op::floor_out(self, out);
    }
}
at::Tensor & fmod_(at::Tensor & self, const at::Scalar & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("fmod_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::fmod_(self, other);
    } else {
        return acl_op::fmod_(self, other);
    }
}
at::Tensor & fmod_(at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("fmod_ exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::fmod_(self, other);
    } else {
        return acl_op::fmod_(self, other);
    }
}
at::Tensor & fmod_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("fmod_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::fmod_out(self, other, out);
    } else {
        return acl_op::fmod_out(self, other, out);
    }
}
at::Tensor & fmod_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("fmod_out exec with jit compile: %d, self is internal format: %d, other is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && other_base_format && out_base_format) {
        return op_api::fmod_out(self, other, out);
    } else {
        return acl_op::fmod_out(self, other, out);
    }
}
at::Tensor & frac_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("frac_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::frac_(self);
    } else {
        return acl_op::frac_(self);
    }
}
at::Tensor & frac_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("frac_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::frac_out(self, out);
    } else {
        return acl_op::frac_out(self, out);
    }
}
at::Tensor & gather_out(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, bool sparse_grad, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool index_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(index);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("gather_out exec with jit compile: %d, self is internal format: %d, index is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !index_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && index_base_format && out_base_format) {
        return op_api::gather_out(self, dim, index, sparse_grad, out);
    } else {
        return acl_op::gather_out(self, dim, index, sparse_grad, out);
    }
}
at::Tensor & gather_out(const at::Tensor & self, int64_t dim, const at::Tensor & index, bool sparse_grad, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool index_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(index);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("gather_out exec with jit compile: %d, self is internal format: %d, index is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !index_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && index_base_format && out_base_format) {
        return op_api::gather_out(self, dim, index, sparse_grad, out);
    } else {
        return acl_op::gather_out(self, dim, index, sparse_grad, out);
    }
}
at::Tensor & gcd_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("gcd_out exec with jit compile: %d, self is internal format: %d, other is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && other_base_format && out_base_format) {
        return op_api::gcd_out(self, other, out);
    } else {
        return acl_op::gcd_out(self, other, out);
    }
}
at::Tensor & ge_(at::Tensor & self, const at::Scalar & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("ge_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::ge_(self, other);
    } else {
        return acl_op::ge_(self, other);
    }
}
at::Tensor & ge_(at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("ge_ exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::ge_(self, other);
    } else {
        return acl_op::ge_(self, other);
    }
}
at::Tensor & ge_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("ge_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::ge_out(self, other, out);
    } else {
        return acl_op::ge_out(self, other, out);
    }
}
at::Tensor & ge_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("ge_out exec with jit compile: %d, self is internal format: %d, other is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && other_base_format && out_base_format) {
        return op_api::ge_out(self, other, out);
    } else {
        return acl_op::ge_out(self, other, out);
    }
}
at::Tensor & glu_backward_out(const at::Tensor & grad_output, const at::Tensor & self, int64_t dim, at::Tensor & grad_input){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    ASCEND_LOGI("glu_backward_out exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, grad_input is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !grad_input_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && grad_input_base_format) {
        return op_api::glu_backward_out(grad_output, self, dim, grad_input);
    } else {
        return acl_op::glu_backward_out(grad_output, self, dim, grad_input);
    }
}
at::Tensor & glu_out(const at::Tensor & self, int64_t dim, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("glu_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::glu_out(self, dim, out);
    } else {
        return acl_op::glu_out(self, dim, out);
    }
}
at::Tensor & gt_(at::Tensor & self, const at::Scalar & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("gt_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::gt_(self, other);
    } else {
        return acl_op::gt_(self, other);
    }
}
at::Tensor & gt_(at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("gt_ exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::gt_(self, other);
    } else {
        return acl_op::gt_(self, other);
    }
}
at::Tensor & gt_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("gt_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::gt_out(self, other, out);
    } else {
        return acl_op::gt_out(self, other, out);
    }
}
at::Tensor & gt_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("gt_out exec with jit compile: %d, self is internal format: %d, other is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && other_base_format && out_base_format) {
        return op_api::gt_out(self, other, out);
    } else {
        return acl_op::gt_out(self, other, out);
    }
}
at::Tensor & hardshrink_backward_out(const at::Tensor & grad_out, const at::Tensor & self, const at::Scalar & lambd, at::Tensor & grad_input){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_out);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    ASCEND_LOGI("hardshrink_backward_out exec with jit compile: %d, grad_out is internal format: %d, self is internal format: %d, grad_input is internal format: %d",
                !is_jit_disable, !grad_out_base_format, !self_base_format, !grad_input_base_format);
    if (is_jit_disable && grad_out_base_format && self_base_format && grad_input_base_format) {
        return op_api::hardshrink_backward_out(grad_out, self, lambd, grad_input);
    } else {
        return acl_op::hardshrink_backward_out(grad_out, self, lambd, grad_input);
    }
}
at::Tensor & hardshrink_out(const at::Tensor & self, const at::Scalar & lambd, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("hardshrink_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::hardshrink_out(self, lambd, out);
    } else {
        return acl_op::hardshrink_out(self, lambd, out);
    }
}
at::Tensor & hardsigmoid_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("hardsigmoid_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::hardsigmoid_(self);
    } else {
        return acl_op::hardsigmoid_(self);
    }
}
at::Tensor & hardsigmoid_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("hardsigmoid_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::hardsigmoid_out(self, out);
    } else {
        return acl_op::hardsigmoid_out(self, out);
    }
}
at::Tensor & hardswish_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("hardswish_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::hardswish_(self);
    } else {
        return acl_op::hardswish_(self);
    }
}
at::Tensor & hardswish_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("hardswish_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::hardswish_out(self, out);
    } else {
        return acl_op::hardswish_out(self, out);
    }
}
at::Tensor & hardtanh_(at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("hardtanh_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::hardtanh_(self, min_val, max_val);
    } else {
        return acl_op::hardtanh_(self, min_val, max_val);
    }
}
at::Tensor & hardtanh_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val, at::Tensor & grad_input){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    ASCEND_LOGI("hardtanh_backward_out exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, grad_input is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !grad_input_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && grad_input_base_format) {
        return op_api::hardtanh_backward_out(grad_output, self, min_val, max_val, grad_input);
    } else {
        return acl_op::hardtanh_backward_out(grad_output, self, min_val, max_val, grad_input);
    }
}
at::Tensor & hardtanh_out(const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("hardtanh_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::hardtanh_out(self, min_val, max_val, out);
    } else {
        return acl_op::hardtanh_out(self, min_val, max_val, out);
    }
}
at::Tensor & histc_out(const at::Tensor & self, int64_t bins, const at::Scalar & min, const at::Scalar & max, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("histc_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::histc_out(self, bins, min, max, out);
    } else {
        return acl_op::histc_out(self, bins, min, max, out);
    }
}
at::Tensor & im2col_out(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("im2col_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::im2col_out(self, kernel_size, dilation, padding, stride, out);
    } else {
        return acl_op::im2col_out(self, kernel_size, dilation, padding, stride, out);
    }
}
at::Tensor & index_add_out(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool index_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(index);
    bool source_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(source);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("index_add_out exec with jit compile: %d, self is internal format: %d, index is internal format: %d, source is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !index_base_format, !source_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && index_base_format && source_base_format && out_base_format) {
        return op_api::index_add_out(self, dim, index, source, alpha, out);
    } else {
        return acl_op::index_add_out(self, dim, index, source, alpha, out);
    }
}
at::Tensor & index_copy_(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool index_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(index);
    bool source_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(source);

    ASCEND_LOGI("index_copy_ exec with jit compile: %d, self is internal format: %d, index is internal format: %d, source is internal format: %d",
                !is_jit_disable, !self_base_format, !index_base_format, !source_base_format);
    if (is_jit_disable && self_base_format && index_base_format && source_base_format) {
        return op_api::index_copy_(self, dim, index, source);
    } else {
        return acl_op::index_copy_(self, dim, index, source);
    }
}
at::Tensor & index_copy_out(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, at::Tensor & out){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool index_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(index);
    bool source_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(source);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    if (!self_base_format || !index_base_format || !source_base_format || !out_base_format) {
        TORCH_CHECK(false,
            "Current operator index_copy_out do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::index_copy_out(self, dim, index, source, out);
}
at::Tensor & index_fill_(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool index_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(index);

    ASCEND_LOGI("index_fill_ exec with jit compile: %d, self is internal format: %d, index is internal format: %d",
                !is_jit_disable, !self_base_format, !index_base_format);
    if (is_jit_disable && self_base_format && index_base_format) {
        return op_api::index_fill_(self, dim, index, value);
    } else {
        return acl_op::index_fill_(self, dim, index, value);
    }
}
at::Tensor & index_fill_(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & value){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool index_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(index);
    bool value_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(value);

    ASCEND_LOGI("index_fill_ exec with jit compile: %d, self is internal format: %d, index is internal format: %d, value is internal format: %d",
                !is_jit_disable, !self_base_format, !index_base_format, !value_base_format);
    if (is_jit_disable && self_base_format && index_base_format && value_base_format) {
        return op_api::index_fill_(self, dim, index, value);
    } else {
        return acl_op::index_fill_(self, dim, index, value);
    }
}
at::Tensor & index_put_(at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);
    bool values_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(values);

    ASCEND_LOGI("index_put_ exec with jit compile: %d, self is internal format: %d, indices is internal format: %d, values is internal format: %d",
                !is_jit_disable, !self_base_format, !indices_base_format, !values_base_format);
    if (is_jit_disable && self_base_format && indices_base_format && values_base_format) {
        return op_api::index_put_(self, indices, values, accumulate);
    } else {
        return acl_op::index_put_(self, indices, values, accumulate);
    }
}
at::Tensor & index_select_out(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool index_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(index);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("index_select_out exec with jit compile: %d, self is internal format: %d, index is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !index_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && index_base_format && out_base_format) {
        return op_api::index_select_out(self, dim, index, out);
    } else {
        return acl_op::index_select_out(self, dim, index, out);
    }
}
at::Tensor & index_select_out(const at::Tensor & self, int64_t dim, const at::Tensor & index, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool index_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(index);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("index_select_out exec with jit compile: %d, self is internal format: %d, index is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !index_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && index_base_format && out_base_format) {
        return op_api::index_select_out(self, dim, index, out);
    } else {
        return acl_op::index_select_out(self, dim, index, out);
    }
}
at::Tensor & inverse_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("inverse_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::inverse_out(self, out);
    } else {
        return acl_op::inverse_out(self, out);
    }
}
at::Tensor & isin_out(const at::Scalar & element, const at::Tensor & test_elements, bool assume_unique, bool invert, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool test_elements_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(test_elements);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("isin_out exec with jit compile: %d, test_elements is internal format: %d, out is internal format: %d",
                !is_jit_disable, !test_elements_base_format, !out_base_format);
    if (is_jit_disable && test_elements_base_format && out_base_format) {
        return op_api::isin_out(element, test_elements, assume_unique, invert, out);
    } else {
        return acl_op::isin_out(element, test_elements, assume_unique, invert, out);
    }
}
at::Tensor & isin_out(const at::Tensor & element, const at::Scalar & test_elements, bool assume_unique, bool invert, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool element_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(element);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("isin_out exec with jit compile: %d, element is internal format: %d, out is internal format: %d",
                !is_jit_disable, !element_base_format, !out_base_format);
    if (is_jit_disable && element_base_format && out_base_format) {
        return op_api::isin_out(element, test_elements, assume_unique, invert, out);
    } else {
        return acl_op::isin_out(element, test_elements, assume_unique, invert, out);
    }
}
at::Tensor & isin_out(const at::Tensor & elements, const at::Tensor & test_elements, bool assume_unique, bool invert, at::Tensor & out){
    bool elements_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(elements);
    bool test_elements_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(test_elements);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    if (!elements_base_format || !test_elements_base_format || !out_base_format) {
        TORCH_CHECK(false,
            "Current operator isin_out do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::isin_out(elements, test_elements, assume_unique, invert, out);
}
at::Tensor & isneginf_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("isneginf_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::isneginf_out(self, out);
    } else {
        return acl_op::isneginf_out(self, out);
    }
}
at::Tensor & isposinf_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("isposinf_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::isposinf_out(self, out);
    } else {
        return acl_op::isposinf_out(self, out);
    }
}
at::Tensor & le_(at::Tensor & self, const at::Scalar & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("le_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::le_(self, other);
    } else {
        return acl_op::le_(self, other);
    }
}
at::Tensor & le_(at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("le_ exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::le_(self, other);
    } else {
        return acl_op::le_(self, other);
    }
}
at::Tensor & le_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("le_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::le_out(self, other, out);
    } else {
        return acl_op::le_out(self, other, out);
    }
}
at::Tensor & le_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("le_out exec with jit compile: %d, self is internal format: %d, other is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && other_base_format && out_base_format) {
        return op_api::le_out(self, other, out);
    } else {
        return acl_op::le_out(self, other, out);
    }
}
at::Tensor & leaky_relu_(at::Tensor & self, const at::Scalar & negative_slope){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("leaky_relu_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::leaky_relu_(self, negative_slope);
    } else {
        return acl_op::leaky_relu_(self, negative_slope);
    }
}
at::Tensor & leaky_relu_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & negative_slope, bool self_is_result, at::Tensor & grad_input){
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    if (!grad_output_base_format || !self_base_format || !grad_input_base_format) {
        TORCH_CHECK(false,
            "Current operator leaky_relu_backward_out do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::leaky_relu_backward_out(grad_output, self, negative_slope, self_is_result, grad_input);
}
at::Tensor & leaky_relu_out(const at::Tensor & self, const at::Scalar & negative_slope, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("leaky_relu_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::leaky_relu_out(self, negative_slope, out);
    } else {
        return acl_op::leaky_relu_out(self, negative_slope, out);
    }
}
at::Tensor & lerp_(at::Tensor & self, const at::Tensor & end, const at::Scalar & weight){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool end_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(end);

    ASCEND_LOGI("lerp_ exec with jit compile: %d, self is internal format: %d, end is internal format: %d",
                !is_jit_disable, !self_base_format, !end_base_format);
    if (is_jit_disable && self_base_format && end_base_format) {
        return op_api::lerp_(self, end, weight);
    } else {
        return acl_op::lerp_(self, end, weight);
    }
}
at::Tensor & lerp_(at::Tensor & self, const at::Tensor & end, const at::Tensor & weight){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool end_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(end);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);

    ASCEND_LOGI("lerp_ exec with jit compile: %d, self is internal format: %d, end is internal format: %d, weight is internal format: %d",
                !is_jit_disable, !self_base_format, !end_base_format, !weight_base_format);
    if (is_jit_disable && self_base_format && end_base_format && weight_base_format) {
        return op_api::lerp_(self, end, weight);
    } else {
        return acl_op::lerp_(self, end, weight);
    }
}
at::Tensor & lerp_out(const at::Tensor & self, const at::Tensor & end, const at::Scalar & weight, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool end_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(end);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("lerp_out exec with jit compile: %d, self is internal format: %d, end is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !end_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && end_base_format && out_base_format) {
        return op_api::lerp_out(self, end, weight, out);
    } else {
        return acl_op::lerp_out(self, end, weight, out);
    }
}
at::Tensor & lerp_out(const at::Tensor & self, const at::Tensor & end, const at::Tensor & weight, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool end_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(end);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("lerp_out exec with jit compile: %d, self is internal format: %d, end is internal format: %d, weight is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !end_base_format, !weight_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && end_base_format && weight_base_format && out_base_format) {
        return op_api::lerp_out(self, end, weight, out);
    } else {
        return acl_op::lerp_out(self, end, weight, out);
    }
}
at::Tensor & linalg_cross_out(const at::Tensor & self, const at::Tensor & other, int64_t dim, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("linalg_cross_out exec with jit compile: %d, self is internal format: %d, other is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && other_base_format && out_base_format) {
        return op_api::linalg_cross_out(self, other, dim, out);
    } else {
        return acl_op::linalg_cross_out(self, other, dim, out);
    }
}
at::Tensor & linalg_solve_triangular_out(const at::Tensor & self, const at::Tensor & B, bool upper, bool left, bool unitriangular, at::Tensor & out){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool B_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(B);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    if (!self_base_format || !B_base_format || !out_base_format) {
        TORCH_CHECK(false,
            "Current operator linalg_solve_triangular_out do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::linalg_solve_triangular_out(self, B, upper, left, unitriangular, out);
}
at::Tensor & linalg_vector_norm_out(const at::Tensor & self, const at::Scalar & ord, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("linalg_vector_norm_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::linalg_vector_norm_out(self, ord, dim, keepdim, dtype, out);
    } else {
        return acl_op::linalg_vector_norm_out(self, ord, dim, keepdim, dtype, out);
    }
}
at::Tensor & linspace_out(const at::Scalar & start, const at::Scalar & end, int64_t steps, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("linspace_out exec with jit compile: %d, out is internal format: %d",
                !is_jit_disable, !out_base_format);
    if (is_jit_disable && out_base_format) {
        return op_api::linspace_out(start, end, steps, out);
    } else {
        return acl_op::linspace_out(start, end, steps, out);
    }
}
at::Tensor & log10_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("log10_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::log10_(self);
    } else {
        return acl_op::log10_(self);
    }
}
at::Tensor & log10_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("log10_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::log10_out(self, out);
    } else {
        return acl_op::log10_out(self, out);
    }
}
at::Tensor & log1p_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("log1p_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::log1p_(self);
    } else {
        return acl_op::log1p_(self);
    }
}
at::Tensor & log1p_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("log1p_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::log1p_out(self, out);
    } else {
        return acl_op::log1p_out(self, out);
    }
}
at::Tensor & log2_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("log2_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::log2_(self);
    } else {
        return acl_op::log2_(self);
    }
}
at::Tensor & log2_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("log2_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::log2_out(self, out);
    } else {
        return acl_op::log2_out(self, out);
    }
}
at::Tensor & log_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("log_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::log_(self);
    } else {
        return acl_op::log_(self);
    }
}
at::Tensor & log_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("log_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::log_out(self, out);
    } else {
        return acl_op::log_out(self, out);
    }
}
at::Tensor & log_sigmoid_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & buffer, at::Tensor & grad_input){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool buffer_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(buffer);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    ASCEND_LOGI("log_sigmoid_backward_out exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, buffer is internal format: %d, grad_input is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !buffer_base_format, !grad_input_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && buffer_base_format && grad_input_base_format) {
        return op_api::log_sigmoid_backward_out(grad_output, self, buffer, grad_input);
    } else {
        return acl_op::log_sigmoid_backward_out(grad_output, self, buffer, grad_input);
    }
}
at::Tensor & log_sigmoid_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("log_sigmoid_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::log_sigmoid_out(self, out);
    } else {
        return acl_op::log_sigmoid_out(self, out);
    }
}
at::Tensor & logaddexp2_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("logaddexp2_out exec with jit compile: %d, self is internal format: %d, other is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && other_base_format && out_base_format) {
        return op_api::logaddexp2_out(self, other, out);
    } else {
        return acl_op::logaddexp2_out(self, other, out);
    }
}
at::Tensor & logaddexp_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("logaddexp_out exec with jit compile: %d, self is internal format: %d, other is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && other_base_format && out_base_format) {
        return op_api::logaddexp_out(self, other, out);
    } else {
        return acl_op::logaddexp_out(self, other, out);
    }
}
at::Tensor & logical_and_(at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("logical_and_ exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::logical_and_(self, other);
    } else {
        return acl_op::logical_and_(self, other);
    }
}
at::Tensor & logical_and_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("logical_and_out exec with jit compile: %d, self is internal format: %d, other is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && other_base_format && out_base_format) {
        return op_api::logical_and_out(self, other, out);
    } else {
        return acl_op::logical_and_out(self, other, out);
    }
}
at::Tensor & logical_not_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("logical_not_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::logical_not_(self);
    } else {
        return acl_op::logical_not_(self);
    }
}
at::Tensor & logical_not_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("logical_not_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::logical_not_out(self, out);
    } else {
        return acl_op::logical_not_out(self, out);
    }
}
at::Tensor & logical_or_(at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("logical_or_ exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::logical_or_(self, other);
    } else {
        return acl_op::logical_or_(self, other);
    }
}
at::Tensor & logical_or_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("logical_or_out exec with jit compile: %d, self is internal format: %d, other is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && other_base_format && out_base_format) {
        return op_api::logical_or_out(self, other, out);
    } else {
        return acl_op::logical_or_out(self, other, out);
    }
}
at::Tensor & logical_xor_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("logical_xor_out exec with jit compile: %d, self is internal format: %d, other is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && other_base_format && out_base_format) {
        return op_api::logical_xor_out(self, other, out);
    } else {
        return acl_op::logical_xor_out(self, other, out);
    }
}
at::Tensor & logit_(at::Tensor & self, c10::optional<double> eps){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator logit_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::logit_(self, eps);
}
at::Tensor & logit_backward_out(const at::Tensor & grad_output, const at::Tensor & self, c10::optional<double> eps, at::Tensor & grad_input){
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    if (!grad_output_base_format || !self_base_format || !grad_input_base_format) {
        TORCH_CHECK(false,
            "Current operator logit_backward_out do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::logit_backward_out(grad_output, self, eps, grad_input);
}
at::Tensor & logit_out(const at::Tensor & self, c10::optional<double> eps, at::Tensor & out){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    if (!self_base_format || !out_base_format) {
        TORCH_CHECK(false,
            "Current operator logit_out do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::logit_out(self, eps, out);
}
at::Tensor & logspace_out(const at::Scalar & start, const at::Scalar & end, int64_t steps, double base, at::Tensor & out){
    return acl_op::logspace_out(start, end, steps, base, out);
}
at::Tensor & logsumexp_out(const at::Tensor & self, at::DimnameList dim, bool keepdim, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("logsumexp_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::logsumexp_out(self, dim, keepdim, out);
    } else {
        return acl_op::logsumexp_out(self, dim, keepdim, out);
    }
}
at::Tensor & logsumexp_out(const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("logsumexp_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::logsumexp_out(self, dim, keepdim, out);
    } else {
        return acl_op::logsumexp_out(self, dim, keepdim, out);
    }
}
at::Tensor & lt_(at::Tensor & self, const at::Scalar & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("lt_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::lt_(self, other);
    } else {
        return acl_op::lt_(self, other);
    }
}
at::Tensor & lt_(at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("lt_ exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::lt_(self, other);
    } else {
        return acl_op::lt_(self, other);
    }
}
at::Tensor & lt_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("lt_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::lt_out(self, other, out);
    } else {
        return acl_op::lt_out(self, other, out);
    }
}
at::Tensor & lt_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("lt_out exec with jit compile: %d, self is internal format: %d, other is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && other_base_format && out_base_format) {
        return op_api::lt_out(self, other, out);
    } else {
        return acl_op::lt_out(self, other, out);
    }
}
at::Tensor & masked_fill_(at::Tensor & self, const at::Tensor & mask, const at::Scalar & value){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mask);

    ASCEND_LOGI("masked_fill_ exec with jit compile: %d, self is internal format: %d, mask is internal format: %d",
                !is_jit_disable, !self_base_format, !mask_base_format);
    if (is_jit_disable && self_base_format && mask_base_format) {
        return op_api::masked_fill_(self, mask, value);
    } else {
        return acl_op::masked_fill_(self, mask, value);
    }
}
at::Tensor & masked_fill_(at::Tensor & self, const at::Tensor & mask, const at::Tensor & value){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mask);
    bool value_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(value);

    ASCEND_LOGI("masked_fill_ exec with jit compile: %d, self is internal format: %d, mask is internal format: %d, value is internal format: %d",
                !is_jit_disable, !self_base_format, !mask_base_format, !value_base_format);
    if (is_jit_disable && self_base_format && mask_base_format && value_base_format) {
        return op_api::masked_fill_(self, mask, value);
    } else {
        return acl_op::masked_fill_(self, mask, value);
    }
}
at::Tensor & masked_scatter_(at::Tensor & self, const at::Tensor & mask, const at::Tensor & source){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mask);
    bool source_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(source);

    ASCEND_LOGI("masked_scatter_ exec with jit compile: %d, self is internal format: %d, mask is internal format: %d, source is internal format: %d",
                !is_jit_disable, !self_base_format, !mask_base_format, !source_base_format);
    if (is_jit_disable && self_base_format && mask_base_format && source_base_format) {
        return op_api::masked_scatter_(self, mask, source);
    } else {
        return acl_op::masked_scatter_(self, mask, source);
    }
}
at::Tensor & masked_select_out(const at::Tensor & self, const at::Tensor & mask, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mask);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("masked_select_out exec with jit compile: %d, self is internal format: %d, mask is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !mask_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && mask_base_format && out_base_format) {
        return op_api::masked_select_out(self, mask, out);
    } else {
        return acl_op::masked_select_out(self, mask, out);
    }
}
at::Tensor & matmul_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("matmul_out exec with jit compile: %d, self is internal format: %d, other is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && other_base_format && out_base_format) {
        return op_api::matmul_out(self, other, out);
    } else {
        return acl_op::matmul_out(self, other, out);
    }
}
at::Tensor & max_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("max_out exec with jit compile: %d, self is internal format: %d, other is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && other_base_format && out_base_format) {
        return op_api::max_out(self, other, out);
    } else {
        return acl_op::max_out(self, other, out);
    }
}
at::Tensor & max_out_sparse(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    return sparse::max_out_sparse(self, other, out);
}
at::Tensor & max_pool2d_with_indices_backward_out(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices, at::Tensor & grad_input){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    ASCEND_LOGI("max_pool2d_with_indices_backward_out exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, indices is internal format: %d, grad_input is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !indices_base_format, !grad_input_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && indices_base_format && grad_input_base_format) {
        return op_api::max_pool2d_with_indices_backward_out(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices, grad_input);
    } else {
        return acl_op::max_pool2d_with_indices_backward_out(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices, grad_input);
    }
}
at::Tensor & max_pool3d_with_indices_backward_out(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices, at::Tensor & grad_input){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    ASCEND_LOGI("max_pool3d_with_indices_backward_out exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, indices is internal format: %d, grad_input is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !indices_base_format, !grad_input_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && indices_base_format && grad_input_base_format) {
        return op_api::max_pool3d_with_indices_backward_out(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices, grad_input);
    } else {
        return acl_op::max_pool3d_with_indices_backward_out(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices, grad_input);
    }
}
at::Tensor & max_unpool2d_out(const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("max_unpool2d_out exec with jit compile: %d, self is internal format: %d, indices is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !indices_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && indices_base_format && out_base_format) {
        return op_api::max_unpool2d_out(self, indices, output_size, out);
    } else {
        return acl_op::max_unpool2d_out(self, indices, output_size, out);
    }
}
at::Tensor & max_unpool3d_out(const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("max_unpool3d_out exec with jit compile: %d, self is internal format: %d, indices is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !indices_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && indices_base_format && out_base_format) {
        return op_api::max_unpool3d_out(self, indices, output_size, stride, padding, out);
    } else {
        return acl_op::max_unpool3d_out(self, indices, output_size, stride, padding, out);
    }
}
at::Tensor & maximum_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("maximum_out exec with jit compile: %d, self is internal format: %d, other is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && other_base_format && out_base_format) {
        return op_api::maximum_out(self, other, out);
    } else {
        return acl_op::maximum_out(self, other, out);
    }
}
at::Tensor & mean_out(const at::Tensor & self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("mean_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::mean_out(self, dim, keepdim, dtype, out);
    } else {
        return acl_op::mean_out(self, dim, keepdim, dtype, out);
    }
}
at::Tensor & mean_out(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("mean_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::mean_out(self, dim, keepdim, dtype, out);
    } else {
        return acl_op::mean_out(self, dim, keepdim, dtype, out);
    }
}
at::Tensor & min_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("min_out exec with jit compile: %d, self is internal format: %d, other is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && other_base_format && out_base_format) {
        return op_api::min_out(self, other, out);
    } else {
        return acl_op::min_out(self, other, out);
    }
}
at::Tensor & minimum_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("minimum_out exec with jit compile: %d, self is internal format: %d, other is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && other_base_format && out_base_format) {
        return op_api::minimum_out(self, other, out);
    } else {
        return acl_op::minimum_out(self, other, out);
    }
}
at::Tensor & mish_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("mish_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::mish_(self);
    } else {
        return acl_op::mish_(self);
    }
}
at::Tensor & mish_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("mish_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::mish_out(self, out);
    } else {
        return acl_op::mish_out(self, out);
    }
}
at::Tensor & mm_out(const at::Tensor & self, const at::Tensor & mat2, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool mat2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mat2);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("mm_out exec with jit compile: %d, self is internal format: %d, mat2 is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !mat2_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && mat2_base_format && out_base_format) {
        return op_api::mm_out(self, mat2, out);
    } else {
        return acl_op::mm_out(self, mat2, out);
    }
}
at::Tensor & mse_loss_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & grad_input){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    ASCEND_LOGI("mse_loss_backward_out exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, target is internal format: %d, grad_input is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !target_base_format, !grad_input_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && target_base_format && grad_input_base_format) {
        return op_api::mse_loss_backward_out(grad_output, self, target, reduction, grad_input);
    } else {
        return acl_op::mse_loss_backward_out(grad_output, self, target, reduction, grad_input);
    }
}
at::Tensor & mse_loss_out(const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("mse_loss_out exec with jit compile: %d, self is internal format: %d, target is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !target_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && target_base_format && out_base_format) {
        return op_api::mse_loss_out(self, target, reduction, out);
    } else {
        return acl_op::mse_loss_out(self, target, reduction, out);
    }
}
at::Tensor & mul_(at::Tensor & self, const at::Scalar & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("mul_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::mul_(self, other);
    } else {
        return acl_op::mul_(self, other);
    }
}
at::Tensor & mul_(at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("mul_ exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::mul_(self, other);
    } else {
        return acl_op::mul_(self, other);
    }
}
at::Tensor & mul_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("mul_out exec with jit compile: %d, self is internal format: %d, other is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && other_base_format && out_base_format) {
        return op_api::mul_out(self, other, out);
    } else {
        return acl_op::mul_out(self, other, out);
    }
}
at::Tensor & multilabel_margin_loss_out(const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("multilabel_margin_loss_out exec with jit compile: %d, self is internal format: %d, target is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !target_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && target_base_format && out_base_format) {
        return op_api::multilabel_margin_loss_out(self, target, reduction, out);
    } else {
        return acl_op::multilabel_margin_loss_out(self, target, reduction, out);
    }
}
at::Tensor & multinomial_out(const at::Tensor & self, int64_t num_samples, bool replacement, c10::optional<at::Generator> generator, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("multinomial_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::multinomial_out(self, num_samples, replacement, generator, out);
    } else {
        return acl_op::multinomial_out(self, num_samples, replacement, generator, out);
    }
}
at::Tensor & mv_out(const at::Tensor & self, const at::Tensor & vec, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool vec_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(vec);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("mv_out exec with jit compile: %d, self is internal format: %d, vec is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !vec_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && vec_base_format && out_base_format) {
        return op_api::mv_out(self, vec, out);
    } else {
        return acl_op::mv_out(self, vec, out);
    }
}
at::Tensor & nan_to_num_(at::Tensor & self, c10::optional<double> nan, c10::optional<double> posinf, c10::optional<double> neginf){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("nan_to_num_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::nan_to_num_(self, nan, posinf, neginf);
    } else {
        return acl_op::nan_to_num_(self, nan, posinf, neginf);
    }
}
at::Tensor & nan_to_num_out(const at::Tensor & self, c10::optional<double> nan, c10::optional<double> posinf, c10::optional<double> neginf, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("nan_to_num_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::nan_to_num_out(self, nan, posinf, neginf, out);
    } else {
        return acl_op::nan_to_num_out(self, nan, posinf, neginf, out);
    }
}
at::Tensor & nansum_out(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    if (!self_base_format || !out_base_format) {
        TORCH_CHECK(false,
            "Current operator nansum_out do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::nansum_out(self, dim, keepdim, dtype, out);
}
at::Tensor & ne_(at::Tensor & self, const at::Scalar & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("ne_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::ne_(self, other);
    } else {
        return acl_op::ne_(self, other);
    }
}
at::Tensor & ne_(at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("ne_ exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::ne_(self, other);
    } else {
        return acl_op::ne_(self, other);
    }
}
at::Tensor & ne_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("ne_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::ne_out(self, other, out);
    } else {
        return acl_op::ne_out(self, other, out);
    }
}
at::Tensor & ne_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("ne_out exec with jit compile: %d, self is internal format: %d, other is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && other_base_format && out_base_format) {
        return op_api::ne_out(self, other, out);
    } else {
        return acl_op::ne_out(self, other, out);
    }
}
at::Tensor & neg_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("neg_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::neg_(self);
    } else {
        return acl_op::neg_(self);
    }
}
at::Tensor & neg_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("neg_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::neg_out(self, out);
    } else {
        return acl_op::neg_out(self, out);
    }
}
at::Tensor & nll_loss2d_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight, at::Tensor & grad_input){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool total_weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(total_weight);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    ASCEND_LOGI("nll_loss2d_backward_out exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, target is internal format: %d, weight is internal format: %d, total_weight is internal format: %d, grad_input is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !target_base_format, !weight_base_format, !total_weight_base_format, !grad_input_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && target_base_format && weight_base_format && total_weight_base_format && grad_input_base_format) {
        return op_api::nll_loss2d_backward_out(grad_output, self, target, weight, reduction, ignore_index, total_weight, grad_input);
    } else {
        return acl_op::nll_loss2d_backward_out(grad_output, self, target, weight, reduction, ignore_index, total_weight, grad_input);
    }
}
at::Tensor & nll_loss2d_out(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, at::Tensor & out){
    return acl_op::nll_loss2d_out(self, target, weight, reduction, ignore_index, out);
}
at::Tensor & nll_loss_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight, at::Tensor & grad_input){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool total_weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(total_weight);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    ASCEND_LOGI("nll_loss_backward_out exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, target is internal format: %d, weight is internal format: %d, total_weight is internal format: %d, grad_input is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !target_base_format, !weight_base_format, !total_weight_base_format, !grad_input_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && target_base_format && weight_base_format && total_weight_base_format && grad_input_base_format) {
        return op_api::nll_loss_backward_out(grad_output, self, target, weight, reduction, ignore_index, total_weight, grad_input);
    } else {
        return acl_op::nll_loss_backward_out(grad_output, self, target, weight, reduction, ignore_index, total_weight, grad_input);
    }
}
at::Tensor & nll_loss_out(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, at::Tensor & out){
    return acl_op::nll_loss_out(self, target, weight, reduction, ignore_index, out);
}
at::Tensor & nonzero_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("nonzero_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::nonzero_out(self, out);
    } else {
        return acl_op::nonzero_out(self, out);
    }
}
at::Tensor & norm_out(const at::Tensor & self, const c10::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::ScalarType dtype, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("norm_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::norm_out(self, p, dim, keepdim, dtype, out);
    } else {
        return acl_op::norm_out(self, p, dim, keepdim, dtype, out);
    }
}
at::Tensor & norm_out(const at::Tensor & self, const c10::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("norm_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::norm_out(self, p, dim, keepdim, out);
    } else {
        return acl_op::norm_out(self, p, dim, keepdim, out);
    }
}
at::Tensor & normal_(at::Tensor & self, double mean, double std, c10::optional<at::Generator> generator){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("normal_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::normal_(self, mean, std, generator);
    } else {
        return acl_op::normal_(self, mean, std, generator);
    }
}
at::Tensor & normal_out(const at::Tensor & mean, const at::Tensor & std, c10::optional<at::Generator> generator, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool mean_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mean);
    bool std_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(std);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("normal_out exec with jit compile: %d, mean is internal format: %d, std is internal format: %d, out is internal format: %d",
                !is_jit_disable, !mean_base_format, !std_base_format, !out_base_format);
    if (is_jit_disable && mean_base_format && std_base_format && out_base_format) {
        return op_api::normal_out(mean, std, generator, out);
    } else {
        return acl_op::normal_out(mean, std, generator, out);
    }
}
at::Tensor & normal_out(const at::Tensor & mean, double std, c10::optional<at::Generator> generator, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool mean_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mean);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("normal_out exec with jit compile: %d, mean is internal format: %d, out is internal format: %d",
                !is_jit_disable, !mean_base_format, !out_base_format);
    if (is_jit_disable && mean_base_format && out_base_format) {
        return op_api::normal_out(mean, std, generator, out);
    } else {
        return acl_op::normal_out(mean, std, generator, out);
    }
}
at::Tensor & normal_out(double mean, const at::Tensor & std, c10::optional<at::Generator> generator, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool std_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(std);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("normal_out exec with jit compile: %d, std is internal format: %d, out is internal format: %d",
                !is_jit_disable, !std_base_format, !out_base_format);
    if (is_jit_disable && std_base_format && out_base_format) {
        return op_api::normal_out(mean, std, generator, out);
    } else {
        return acl_op::normal_out(mean, std, generator, out);
    }
}
at::Tensor & normal_out(double mean, double std, at::IntArrayRef size, c10::optional<at::Generator> generator, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("normal_out exec with jit compile: %d, out is internal format: %d",
                !is_jit_disable, !out_base_format);
    if (is_jit_disable && out_base_format) {
        return op_api::normal_out(mean, std, size, generator, out);
    } else {
        return acl_op::normal_out(mean, std, size, generator, out);
    }
}
at::Tensor & npu_attn_softmax_(at::Tensor & self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_attn_softmax_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_attn_softmax_(self);
}
at::Tensor & npu_attn_softmax_backward_(at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & values){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool values_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(values);

    if (!self_base_format || !grad_output_base_format || !values_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_attn_softmax_backward_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_attn_softmax_backward_(self, grad_output, values);
}
at::Tensor & npu_batch_gather_matmul_(at::Tensor & self, const at::Tensor & x, const at::Tensor & weight_b, const at::Tensor & indices, const c10::optional<at::Tensor> & weight_a, int64_t layer_idx, double scale, int64_t y_offset, int64_t y_slice_size){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool x_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x);
    bool weight_b_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight_b);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);
    bool weight_a_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight_a);

    if (!self_base_format || !x_base_format || !weight_b_base_format || !indices_base_format || !weight_a_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_batch_gather_matmul_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_batch_gather_matmul_(self, x, weight_b, indices, weight_a, layer_idx, scale, y_offset, y_slice_size);
}
at::Tensor & npu_broadcast_out(const at::Tensor & self, at::IntArrayRef size, at::Tensor & out){
    return acl_op::npu_broadcast_out(self, size, out);
}
at::Tensor & npu_conv2d_out(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, at::Tensor & out){
    return acl_op::npu_conv2d_out(input, weight, bias, stride, padding, dilation, groups, out);
}
at::Tensor & npu_conv3d_out(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, at::Tensor & out){
    return acl_op::npu_conv3d_out(input, weight, bias, stride, padding, dilation, groups, out);
}
at::Tensor & npu_dtype_cast_(at::Tensor & self, const at::Tensor & src){
    return acl_op::npu_dtype_cast_(self, src);
}
at::Tensor & npu_grouped_matmul_add_(at::Tensor & self, const at::Tensor & x, const at::Tensor & weight, const at::Tensor & group_list, bool transpose_x, bool transpose_weight, int64_t group_type){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool x_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool group_list_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(group_list);

    if (!self_base_format || !x_base_format || !weight_base_format || !group_list_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_grouped_matmul_add_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_grouped_matmul_add_(self, x, weight, group_list, transpose_x, transpose_weight, group_type);
}
at::Tensor & npu_hans_decode_out(const at::Tensor & mantissa, const at::Tensor & fixed, const at::Tensor & var, const at::Tensor & pdf, bool reshuff, at::Tensor & out){
    bool mantissa_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mantissa);
    bool fixed_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(fixed);
    bool var_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(var);
    bool pdf_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(pdf);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    if (!mantissa_base_format || !fixed_base_format || !var_base_format || !pdf_base_format || !out_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_hans_decode_out do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_hans_decode_out(mantissa, fixed, var, pdf, reshuff, out);
}
at::Tensor & npu_indexing_out(const at::Tensor & self, at::IntArrayRef begin, at::IntArrayRef end, at::IntArrayRef strides, int64_t begin_mask, int64_t end_mask, int64_t ellipsis_mask, int64_t new_axis_mask, int64_t shrink_axis_mask, at::Tensor & out){
    return acl_op::npu_indexing_out(self, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask, out);
}
at::Tensor & npu_nsa_compress_infer_out(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & slot_mapping, int64_t compress_block_size, int64_t compress_stride, int64_t page_block_size, const c10::optional<at::Tensor> & block_table, at::OptionalIntArrayRef actual_seq_len, at::Tensor & cache){
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool slot_mapping_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(slot_mapping);
    bool block_table_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(block_table);
    bool cache_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(cache);

    if (!input_base_format || !weight_base_format || !slot_mapping_base_format || !block_table_base_format || !cache_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_nsa_compress_infer_out do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_nsa_compress_infer_out(input, weight, slot_mapping, compress_block_size, compress_stride, page_block_size, block_table, actual_seq_len, cache);
}
at::Tensor & npu_quant_scatter_(at::Tensor & self, const at::Tensor & indices, const at::Tensor & updates, const at::Tensor & quant_scales, const c10::optional<at::Tensor> & quant_zero_points, int64_t axis, int64_t quant_axis, c10::string_view reduce){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);
    bool updates_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(updates);
    bool quant_scales_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(quant_scales);
    bool quant_zero_points_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(quant_zero_points);

    if (!self_base_format || !indices_base_format || !updates_base_format || !quant_scales_base_format || !quant_zero_points_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_quant_scatter_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_quant_scatter_(self, indices, updates, quant_scales, quant_zero_points, axis, quant_axis, reduce);
}
at::Tensor & npu_reshape_out(const at::Tensor & self, at::IntArrayRef shape, bool can_refresh, at::Tensor & out){
    return acl_op::npu_reshape_out(self, shape, can_refresh, out);
}
at::Tensor & npu_scatter_nd_update_(at::Tensor & self, const at::Tensor & indices, const at::Tensor & updates){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);
    bool updates_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(updates);

    if (!self_base_format || !indices_base_format || !updates_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_scatter_nd_update_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_scatter_nd_update_(self, indices, updates);
}
at::Tensor & npu_silu_(at::Tensor & self){
    return acl_op::npu_silu_(self);
}
at::Tensor & npu_slice_out(const at::Tensor & self, at::IntArrayRef offsets, at::IntArrayRef size, at::Tensor & out){
    return acl_op::npu_slice_out(self, offsets, size, out);
}
at::Tensor & npu_sort_v2_out(const at::Tensor & self, int64_t dim, bool descending, at::Tensor & out){
    return acl_op::npu_sort_v2_out(self, dim, descending, out);
}
at::Tensor & npu_stride_copy_out(const at::Tensor & self, at::IntArrayRef shape, at::IntArrayRef stride, const at::Scalar & storage_offset, at::Tensor & out){
    return acl_op::npu_stride_copy_out(self, shape, stride, storage_offset, out);
}
at::Tensor & npu_transpose_out(const at::Tensor & self, at::IntArrayRef perm, bool require_contiguous, at::Tensor & out){
    return acl_op::npu_transpose_out(self, perm, require_contiguous, out);
}
at::Tensor & npu_view_copy(at::Tensor & self, const at::Tensor & other, bool non_blocking){
    return acl_op::npu_view_copy(self, other, non_blocking);
}
at::Tensor & one_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("one_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::one_(self);
    } else {
        return acl_op::one_(self);
    }
}
at::Tensor & ones_out(at::IntArrayRef size, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("ones_out exec with jit compile: %d, out is internal format: %d",
                !is_jit_disable, !out_base_format);
    if (is_jit_disable && out_base_format) {
        return op_api::ones_out(size, out);
    } else {
        return acl_op::ones_out(size, out);
    }
}
at::Tensor & polar_out(const at::Tensor & abs, const at::Tensor & angle, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool abs_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(abs);
    bool angle_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(angle);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("polar_out exec with jit compile: %d, abs is internal format: %d, angle is internal format: %d, out is internal format: %d",
                !is_jit_disable, !abs_base_format, !angle_base_format, !out_base_format);
    if (is_jit_disable && abs_base_format && angle_base_format && out_base_format) {
        return op_api::polar_out(abs, angle, out);
    } else {
        return acl_op::polar_out(abs, angle, out);
    }
}
at::Tensor & pow_(at::Tensor & self, const at::Scalar & exponent){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("pow_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::pow_(self, exponent);
    } else {
        return acl_op::pow_(self, exponent);
    }
}
at::Tensor & pow_(at::Tensor & self, const at::Tensor & exponent){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool exponent_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(exponent);

    ASCEND_LOGI("pow_ exec with jit compile: %d, self is internal format: %d, exponent is internal format: %d",
                !is_jit_disable, !self_base_format, !exponent_base_format);
    if (is_jit_disable && self_base_format && exponent_base_format) {
        return op_api::pow_(self, exponent);
    } else {
        return acl_op::pow_(self, exponent);
    }
}
at::Tensor & pow_out(const at::Scalar & self, const at::Tensor & exponent, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool exponent_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(exponent);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("pow_out exec with jit compile: %d, exponent is internal format: %d, out is internal format: %d",
                !is_jit_disable, !exponent_base_format, !out_base_format);
    if (is_jit_disable && exponent_base_format && out_base_format) {
        return op_api::pow_out(self, exponent, out);
    } else {
        return acl_op::pow_out(self, exponent, out);
    }
}
at::Tensor & pow_out(const at::Tensor & self, const at::Scalar & exponent, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("pow_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::pow_out(self, exponent, out);
    } else {
        return acl_op::pow_out(self, exponent, out);
    }
}
at::Tensor & pow_out(const at::Tensor & self, const at::Tensor & exponent, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool exponent_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(exponent);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("pow_out exec with jit compile: %d, self is internal format: %d, exponent is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !exponent_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && exponent_base_format && out_base_format) {
        return op_api::pow_out(self, exponent, out);
    } else {
        return acl_op::pow_out(self, exponent, out);
    }
}
at::Tensor & prod_out(const at::Tensor & self, int64_t dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("prod_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::prod_out(self, dim, keepdim, dtype, out);
    } else {
        return acl_op::prod_out(self, dim, keepdim, dtype, out);
    }
}
at::Tensor & put_(at::Tensor & self, const at::Tensor & index, const at::Tensor & source, bool accumulate){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool index_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(index);
    bool source_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(source);

    ASCEND_LOGI("put_ exec with jit compile: %d, self is internal format: %d, index is internal format: %d, source is internal format: %d",
                !is_jit_disable, !self_base_format, !index_base_format, !source_base_format);
    if (is_jit_disable && self_base_format && index_base_format && source_base_format) {
        return op_api::put_(self, index, source, accumulate);
    } else {
        return acl_op::put_(self, index, source, accumulate);
    }
}
at::Tensor & random_(at::Tensor & self, c10::optional<at::Generator> generator){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("random_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::random_(self, generator);
    } else {
        return acl_op::random_(self, generator);
    }
}
at::Tensor & random_(at::Tensor & self, int64_t from, c10::optional<int64_t> to, c10::optional<at::Generator> generator){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("random_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::random_(self, from, to, generator);
    } else {
        return acl_op::random_(self, from, to, generator);
    }
}
at::Tensor & random_(at::Tensor & self, int64_t to, c10::optional<at::Generator> generator){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("random_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::random_(self, to, generator);
    } else {
        return acl_op::random_(self, to, generator);
    }
}
at::Tensor & randperm_out(int64_t n, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("randperm_out exec with jit compile: %d, out is internal format: %d",
                !is_jit_disable, !out_base_format);
    if (is_jit_disable && out_base_format) {
        return op_api::randperm_out(n, out);
    } else {
        return acl_op::randperm_out(n, out);
    }
}
at::Tensor & randperm_out(int64_t n, c10::optional<at::Generator> generator, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("randperm_out exec with jit compile: %d, out is internal format: %d",
                !is_jit_disable, !out_base_format);
    if (is_jit_disable && out_base_format) {
        return op_api::randperm_out(n, generator, out);
    } else {
        return acl_op::randperm_out(n, generator, out);
    }
}
at::Tensor & range_out(const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("range_out exec with jit compile: %d, out is internal format: %d",
                !is_jit_disable, !out_base_format);
    if (is_jit_disable && out_base_format) {
        return op_api::range_out(start, end, step, out);
    } else {
        return acl_op::range_out(start, end, step, out);
    }
}
at::Tensor & reciprocal_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("reciprocal_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::reciprocal_(self);
    } else {
        return acl_op::reciprocal_(self);
    }
}
at::Tensor & reciprocal_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("reciprocal_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::reciprocal_out(self, out);
    } else {
        return acl_op::reciprocal_out(self, out);
    }
}
at::Tensor & reflection_pad1d_backward_out(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    ASCEND_LOGI("reflection_pad1d_backward_out exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, grad_input is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !grad_input_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && grad_input_base_format) {
        return op_api::reflection_pad1d_backward_out(grad_output, self, padding, grad_input);
    } else {
        return acl_op::reflection_pad1d_backward_out(grad_output, self, padding, grad_input);
    }
}
at::Tensor & reflection_pad1d_out(const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("reflection_pad1d_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::reflection_pad1d_out(self, padding, out);
    } else {
        return acl_op::reflection_pad1d_out(self, padding, out);
    }
}
at::Tensor & reflection_pad2d_backward_out(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    ASCEND_LOGI("reflection_pad2d_backward_out exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, grad_input is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !grad_input_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && grad_input_base_format) {
        return op_api::reflection_pad2d_backward_out(grad_output, self, padding, grad_input);
    } else {
        return acl_op::reflection_pad2d_backward_out(grad_output, self, padding, grad_input);
    }
}
at::Tensor & reflection_pad2d_out(const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("reflection_pad2d_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::reflection_pad2d_out(self, padding, out);
    } else {
        return acl_op::reflection_pad2d_out(self, padding, out);
    }
}
at::Tensor & reflection_pad3d_backward_out(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input){
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    if (!grad_output_base_format || !self_base_format || !grad_input_base_format) {
        TORCH_CHECK(false,
            "Current operator reflection_pad3d_backward_out do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::reflection_pad3d_backward_out(grad_output, self, padding, grad_input);
}
at::Tensor & reflection_pad3d_out(const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    if (!self_base_format || !out_base_format) {
        TORCH_CHECK(false,
            "Current operator reflection_pad3d_out do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::reflection_pad3d_out(self, padding, out);
}
at::Tensor & relu_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("relu_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::relu_(self);
    } else {
        return acl_op::relu_(self);
    }
}
at::Tensor & remainder_(at::Tensor & self, const at::Scalar & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("remainder_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::remainder_(self, other);
    } else {
        return acl_op::remainder_(self, other);
    }
}
at::Tensor & remainder_(at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("remainder_ exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::remainder_(self, other);
    } else {
        return acl_op::remainder_(self, other);
    }
}
at::Tensor & remainder_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("remainder_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::remainder_out(self, other, out);
    } else {
        return acl_op::remainder_out(self, other, out);
    }
}
at::Tensor & remainder_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("remainder_out exec with jit compile: %d, self is internal format: %d, other is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && other_base_format && out_base_format) {
        return op_api::remainder_out(self, other, out);
    } else {
        return acl_op::remainder_out(self, other, out);
    }
}
at::Tensor & renorm_(at::Tensor & self, const at::Scalar & p, int64_t dim, const at::Scalar & maxnorm){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("renorm_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::renorm_(self, p, dim, maxnorm);
    } else {
        return acl_op::renorm_(self, p, dim, maxnorm);
    }
}
at::Tensor & renorm_out(const at::Tensor & self, const at::Scalar & p, int64_t dim, const at::Scalar & maxnorm, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("renorm_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::renorm_out(self, p, dim, maxnorm, out);
    } else {
        return acl_op::renorm_out(self, p, dim, maxnorm, out);
    }
}
at::Tensor & replication_pad1d_backward_out(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    ASCEND_LOGI("replication_pad1d_backward_out exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, grad_input is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !grad_input_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && grad_input_base_format) {
        return op_api::replication_pad1d_backward_out(grad_output, self, padding, grad_input);
    } else {
        return acl_op::replication_pad1d_backward_out(grad_output, self, padding, grad_input);
    }
}
at::Tensor & replication_pad1d_out(const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("replication_pad1d_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::replication_pad1d_out(self, padding, out);
    } else {
        return acl_op::replication_pad1d_out(self, padding, out);
    }
}
at::Tensor & replication_pad2d_backward_out(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    ASCEND_LOGI("replication_pad2d_backward_out exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, grad_input is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !grad_input_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && grad_input_base_format) {
        return op_api::replication_pad2d_backward_out(grad_output, self, padding, grad_input);
    } else {
        return acl_op::replication_pad2d_backward_out(grad_output, self, padding, grad_input);
    }
}
at::Tensor & replication_pad2d_out(const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("replication_pad2d_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::replication_pad2d_out(self, padding, out);
    } else {
        return acl_op::replication_pad2d_out(self, padding, out);
    }
}
at::Tensor & replication_pad3d_backward_out(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input){
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    if (!grad_output_base_format || !self_base_format || !grad_input_base_format) {
        TORCH_CHECK(false,
            "Current operator replication_pad3d_backward_out do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::replication_pad3d_backward_out(grad_output, self, padding, grad_input);
}
at::Tensor & replication_pad3d_out(const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    if (!self_base_format || !out_base_format) {
        TORCH_CHECK(false,
            "Current operator replication_pad3d_out do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::replication_pad3d_out(self, padding, out);
}
at::Tensor & round_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("round_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::round_(self);
    } else {
        return acl_op::round_(self);
    }
}
at::Tensor & round_(at::Tensor & self, int64_t decimals){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("round_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::round_(self, decimals);
    } else {
        return acl_op::round_(self, decimals);
    }
}
at::Tensor & round_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("round_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::round_out(self, out);
    } else {
        return acl_op::round_out(self, out);
    }
}
at::Tensor & round_out(const at::Tensor & self, int64_t decimals, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("round_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::round_out(self, decimals, out);
    } else {
        return acl_op::round_out(self, decimals, out);
    }
}
at::Tensor & rrelu_with_noise_(at::Tensor & self, const at::Tensor & noise, const at::Scalar & lower, const at::Scalar & upper, bool training, c10::optional<at::Generator> generator){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool noise_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(noise);

    ASCEND_LOGI("rrelu_with_noise_ exec with jit compile: %d, self is internal format: %d, noise is internal format: %d",
                !is_jit_disable, !self_base_format, !noise_base_format);
    if (is_jit_disable && self_base_format && noise_base_format) {
        return op_api::rrelu_with_noise_(self, noise, lower, upper, training, generator);
    } else {
        return acl_op::rrelu_with_noise_(self, noise, lower, upper, training, generator);
    }
}
at::Tensor & rrelu_with_noise_out(const at::Tensor & self, const at::Tensor & noise, const at::Scalar & lower, const at::Scalar & upper, bool training, c10::optional<at::Generator> generator, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool noise_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(noise);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("rrelu_with_noise_out exec with jit compile: %d, self is internal format: %d, noise is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !noise_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && noise_base_format && out_base_format) {
        return op_api::rrelu_with_noise_out(self, noise, lower, upper, training, generator, out);
    } else {
        return acl_op::rrelu_with_noise_out(self, noise, lower, upper, training, generator, out);
    }
}
at::Tensor & rsqrt_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("rsqrt_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::rsqrt_(self);
    } else {
        return acl_op::rsqrt_(self);
    }
}
at::Tensor & rsqrt_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("rsqrt_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::rsqrt_out(self, out);
    } else {
        return acl_op::rsqrt_out(self, out);
    }
}
at::Tensor & scatter_(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool index_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(index);

    ASCEND_LOGI("scatter_ exec with jit compile: %d, self is internal format: %d, index is internal format: %d",
                !is_jit_disable, !self_base_format, !index_base_format);
    if (is_jit_disable && self_base_format && index_base_format) {
        return op_api::scatter_(self, dim, index, value);
    } else {
        return acl_op::scatter_(self, dim, index, value);
    }
}
at::Tensor & scatter_(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool index_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(index);
    bool src_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(src);

    ASCEND_LOGI("scatter_ exec with jit compile: %d, self is internal format: %d, index is internal format: %d, src is internal format: %d",
                !is_jit_disable, !self_base_format, !index_base_format, !src_base_format);
    if (is_jit_disable && self_base_format && index_base_format && src_base_format) {
        return op_api::scatter_(self, dim, index, src);
    } else {
        return acl_op::scatter_(self, dim, index, src);
    }
}
at::Tensor & scatter_add_(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool index_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(index);
    bool src_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(src);

    ASCEND_LOGI("scatter_add_ exec with jit compile: %d, self is internal format: %d, index is internal format: %d, src is internal format: %d",
                !is_jit_disable, !self_base_format, !index_base_format, !src_base_format);
    if (is_jit_disable && self_base_format && index_base_format && src_base_format) {
        return op_api::scatter_add_(self, dim, index, src);
    } else {
        return acl_op::scatter_add_(self, dim, index, src);
    }
}
at::Tensor & scatter_out(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool index_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(index);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("scatter_out exec with jit compile: %d, self is internal format: %d, index is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !index_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && index_base_format && out_base_format) {
        return op_api::scatter_out(self, dim, index, value, out);
    } else {
        return acl_op::scatter_out(self, dim, index, value, out);
    }
}
at::Tensor & scatter_out(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool index_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(index);
    bool src_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(src);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("scatter_out exec with jit compile: %d, self is internal format: %d, index is internal format: %d, src is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !index_base_format, !src_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && index_base_format && src_base_format && out_base_format) {
        return op_api::scatter_out(self, dim, index, src, out);
    } else {
        return acl_op::scatter_out(self, dim, index, src, out);
    }
}
at::Tensor & scatter_update_(at::Tensor & self, const at::Tensor & indices, const at::Tensor & updates, int64_t axis){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);
    bool updates_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(updates);

    ASCEND_LOGI("scatter_update_ exec with jit compile: %d, self is internal format: %d, indices is internal format: %d, updates is internal format: %d",
                !is_jit_disable, !self_base_format, !indices_base_format, !updates_base_format);
    if (is_jit_disable && self_base_format && indices_base_format && updates_base_format) {
        return op_api::scatter_update_(self, indices, updates, axis);
    } else {
        return acl_op::scatter_update_(self, indices, updates, axis);
    }
}
at::Tensor & searchsorted_out(const at::Tensor & sorted_sequence, const at::Tensor & self, bool out_int32, bool right, c10::optional<c10::string_view> side, const c10::optional<at::Tensor> & sorter, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool sorted_sequence_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(sorted_sequence);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool sorter_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(sorter);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("searchsorted_out exec with jit compile: %d, sorted_sequence is internal format: %d, self is internal format: %d, sorter is internal format: %d, out is internal format: %d",
                !is_jit_disable, !sorted_sequence_base_format, !self_base_format, !sorter_base_format, !out_base_format);
    if (is_jit_disable && sorted_sequence_base_format && self_base_format && sorter_base_format && out_base_format) {
        return op_api::searchsorted_out(sorted_sequence, self, out_int32, right, side, sorter, out);
    } else {
        return acl_op::searchsorted_out(sorted_sequence, self, out_int32, right, side, sorter, out);
    }
}
at::Tensor & sgn_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("sgn_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::sgn_out(self, out);
    } else {
        return acl_op::sgn_out(self, out);
    }
}
at::Tensor & sigmoid_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("sigmoid_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::sigmoid_(self);
    } else {
        return acl_op::sigmoid_(self);
    }
}
at::Tensor & sigmoid_backward_out(const at::Tensor & grad_output, const at::Tensor & output, at::Tensor & grad_input){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(output);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    ASCEND_LOGI("sigmoid_backward_out exec with jit compile: %d, grad_output is internal format: %d, output is internal format: %d, grad_input is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !output_base_format, !grad_input_base_format);
    if (is_jit_disable && grad_output_base_format && output_base_format && grad_input_base_format) {
        return op_api::sigmoid_backward_out(grad_output, output, grad_input);
    } else {
        return acl_op::sigmoid_backward_out(grad_output, output, grad_input);
    }
}
at::Tensor & sigmoid_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("sigmoid_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::sigmoid_out(self, out);
    } else {
        return acl_op::sigmoid_out(self, out);
    }
}
at::Tensor & sign_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("sign_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::sign_(self);
    } else {
        return acl_op::sign_(self);
    }
}
at::Tensor & sign_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("sign_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::sign_out(self, out);
    } else {
        return acl_op::sign_out(self, out);
    }
}
at::Tensor & signbit_out(const at::Tensor & self, at::Tensor & out){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    if (!self_base_format || !out_base_format) {
        TORCH_CHECK(false,
            "Current operator signbit_out do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::signbit_out(self, out);
}
at::Tensor & silu_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("silu_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::silu_(self);
    } else {
        return acl_op::silu_(self);
    }
}
at::Tensor & silu_backward_out(const at::Tensor & grad_output, const at::Tensor & self, at::Tensor & grad_input){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    ASCEND_LOGI("silu_backward_out exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, grad_input is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !grad_input_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && grad_input_base_format) {
        return op_api::silu_backward_out(grad_output, self, grad_input);
    } else {
        return acl_op::silu_backward_out(grad_output, self, grad_input);
    }
}
at::Tensor & silu_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("silu_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::silu_out(self, out);
    } else {
        return acl_op::silu_out(self, out);
    }
}
at::Tensor & sin_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("sin_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::sin_(self);
    } else {
        return acl_op::sin_(self);
    }
}
at::Tensor & sin_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("sin_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::sin_out(self, out);
    } else {
        return acl_op::sin_out(self, out);
    }
}
at::Tensor & sinc_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("sinc_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::sinc_(self);
    } else {
        return acl_op::sinc_(self);
    }
}
at::Tensor & sinc_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("sinc_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::sinc_out(self, out);
    } else {
        return acl_op::sinc_out(self, out);
    }
}
at::Tensor & sinh_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("sinh_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::sinh_(self);
    } else {
        return acl_op::sinh_(self);
    }
}
at::Tensor & sinh_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("sinh_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::sinh_out(self, out);
    } else {
        return acl_op::sinh_out(self, out);
    }
}
at::Tensor & slow_conv3d_forward_out(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & output){
    return acl_op::slow_conv3d_forward_out(self, weight, kernel_size, bias, stride, padding, output);
}
at::Tensor & slow_conv3d_out(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & out){
    return acl_op::slow_conv3d_out(self, weight, kernel_size, bias, stride, padding, out);
}
at::Tensor & slow_conv_transpose2d_out(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("slow_conv_transpose2d_out exec with jit compile: %d, self is internal format: %d, weight is internal format: %d, bias is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !weight_base_format, !bias_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && weight_base_format && bias_base_format && out_base_format) {
        return op_api::slow_conv_transpose2d_out(self, weight, kernel_size, bias, stride, padding, output_padding, dilation, out);
    } else {
        return acl_op::slow_conv_transpose2d_out(self, weight, kernel_size, bias, stride, padding, output_padding, dilation, out);
    }
}
at::Tensor & smooth_l1_loss_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta, at::Tensor & grad_input){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    ASCEND_LOGI("smooth_l1_loss_backward_out exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, target is internal format: %d, grad_input is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !target_base_format, !grad_input_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && target_base_format && grad_input_base_format) {
        return op_api::smooth_l1_loss_backward_out(grad_output, self, target, reduction, beta, grad_input);
    } else {
        return acl_op::smooth_l1_loss_backward_out(grad_output, self, target, reduction, beta, grad_input);
    }
}
at::Tensor & smooth_l1_loss_out(const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("smooth_l1_loss_out exec with jit compile: %d, self is internal format: %d, target is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !target_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && target_base_format && out_base_format) {
        return op_api::smooth_l1_loss_out(self, target, reduction, beta, out);
    } else {
        return acl_op::smooth_l1_loss_out(self, target, reduction, beta, out);
    }
}
at::Tensor & soft_margin_loss_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & grad_input){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    ASCEND_LOGI("soft_margin_loss_backward_out exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, target is internal format: %d, grad_input is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !target_base_format, !grad_input_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && target_base_format && grad_input_base_format) {
        return op_api::soft_margin_loss_backward_out(grad_output, self, target, reduction, grad_input);
    } else {
        return acl_op::soft_margin_loss_backward_out(grad_output, self, target, reduction, grad_input);
    }
}
at::Tensor & soft_margin_loss_out(const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("soft_margin_loss_out exec with jit compile: %d, self is internal format: %d, target is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !target_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && target_base_format && out_base_format) {
        return op_api::soft_margin_loss_out(self, target, reduction, out);
    } else {
        return acl_op::soft_margin_loss_out(self, target, reduction, out);
    }
}
at::Tensor & softplus_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold, at::Tensor & grad_input){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    ASCEND_LOGI("softplus_backward_out exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, grad_input is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !grad_input_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && grad_input_base_format) {
        return op_api::softplus_backward_out(grad_output, self, beta, threshold, grad_input);
    } else {
        return acl_op::softplus_backward_out(grad_output, self, beta, threshold, grad_input);
    }
}
at::Tensor & softplus_out(const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("softplus_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::softplus_out(self, beta, threshold, out);
    } else {
        return acl_op::softplus_out(self, beta, threshold, out);
    }
}
at::Tensor & softshrink_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & lambd, at::Tensor & grad_input){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    ASCEND_LOGI("softshrink_backward_out exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, grad_input is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !grad_input_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && grad_input_base_format) {
        return op_api::softshrink_backward_out(grad_output, self, lambd, grad_input);
    } else {
        return acl_op::softshrink_backward_out(grad_output, self, lambd, grad_input);
    }
}
at::Tensor & softshrink_out(const at::Tensor & self, const at::Scalar & lambd, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("softshrink_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::softshrink_out(self, lambd, out);
    } else {
        return acl_op::softshrink_out(self, lambd, out);
    }
}
at::Tensor & sqrt_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("sqrt_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::sqrt_(self);
    } else {
        return acl_op::sqrt_(self);
    }
}
at::Tensor & sqrt_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("sqrt_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::sqrt_out(self, out);
    } else {
        return acl_op::sqrt_out(self, out);
    }
}
at::Tensor & stack_out(at::TensorList tensors, int64_t dim, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool tensors_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("stack_out exec with jit compile: %d, tensors is internal format: %d, out is internal format: %d",
                !is_jit_disable, !tensors_base_format, !out_base_format);
    if (is_jit_disable && tensors_base_format && out_base_format) {
        return op_api::stack_out(tensors, dim, out);
    } else {
        return acl_op::stack_out(tensors, dim, out);
    }
}
at::Tensor & std_out(const at::Tensor & self, at::OptionalIntArrayRef dim, const c10::optional<at::Scalar> & correction, bool keepdim, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("std_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::std_out(self, dim, correction, keepdim, out);
    } else {
        return acl_op::std_out(self, dim, correction, keepdim, out);
    }
}
at::Tensor & sub_(at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("sub_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::sub_(self, other, alpha);
    } else {
        return acl_op::sub_(self, other, alpha);
    }
}
at::Tensor & sub_(at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("sub_ exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::sub_(self, other, alpha);
    } else {
        return acl_op::sub_(self, other, alpha);
    }
}
at::Tensor & sub_out(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("sub_out exec with jit compile: %d, self is internal format: %d, other is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && other_base_format && out_base_format) {
        return op_api::sub_out(self, other, alpha, out);
    } else {
        return acl_op::sub_out(self, other, alpha, out);
    }
}
at::Tensor & sum_out(const at::Tensor & self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("sum_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::sum_out(self, dim, keepdim, dtype, out);
    } else {
        return acl_op::sum_out(self, dim, keepdim, dtype, out);
    }
}
at::Tensor & sum_out(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("sum_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::sum_out(self, dim, keepdim, dtype, out);
    } else {
        return acl_op::sum_out(self, dim, keepdim, dtype, out);
    }
}
at::Tensor & take_out(const at::Tensor & self, const at::Tensor & index, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool index_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(index);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("take_out exec with jit compile: %d, self is internal format: %d, index is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !index_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && index_base_format && out_base_format) {
        return op_api::take_out(self, index, out);
    } else {
        return acl_op::take_out(self, index, out);
    }
}
at::Tensor & tan_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("tan_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::tan_(self);
    } else {
        return acl_op::tan_(self);
    }
}
at::Tensor & tan_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("tan_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::tan_out(self, out);
    } else {
        return acl_op::tan_out(self, out);
    }
}
at::Tensor & tanh_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("tanh_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::tanh_(self);
    } else {
        return acl_op::tanh_(self);
    }
}
at::Tensor & tanh_backward_out(const at::Tensor & grad_output, const at::Tensor & output, at::Tensor & grad_input){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(output);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    ASCEND_LOGI("tanh_backward_out exec with jit compile: %d, grad_output is internal format: %d, output is internal format: %d, grad_input is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !output_base_format, !grad_input_base_format);
    if (is_jit_disable && grad_output_base_format && output_base_format && grad_input_base_format) {
        return op_api::tanh_backward_out(grad_output, output, grad_input);
    } else {
        return acl_op::tanh_backward_out(grad_output, output, grad_input);
    }
}
at::Tensor & tanh_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("tanh_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::tanh_out(self, out);
    } else {
        return acl_op::tanh_out(self, out);
    }
}
at::Tensor & threshold_(at::Tensor & self, const at::Scalar & threshold, const at::Scalar & value){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("threshold_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::threshold_(self, threshold, value);
    } else {
        return acl_op::threshold_(self, threshold, value);
    }
}
at::Tensor & threshold_out(const at::Tensor & self, const at::Scalar & threshold, const at::Scalar & value, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("threshold_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::threshold_out(self, threshold, value, out);
    } else {
        return acl_op::threshold_out(self, threshold, value, out);
    }
}
at::Tensor & tril_(at::Tensor & self, int64_t diagonal){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("tril_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::tril_(self, diagonal);
    } else {
        return acl_op::tril_(self, diagonal);
    }
}
at::Tensor & tril_out(const at::Tensor & self, int64_t diagonal, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("tril_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::tril_out(self, diagonal, out);
    } else {
        return acl_op::tril_out(self, diagonal, out);
    }
}
at::Tensor & triu_(at::Tensor & self, int64_t diagonal){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("triu_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::triu_(self, diagonal);
    } else {
        return acl_op::triu_(self, diagonal);
    }
}
at::Tensor & triu_out(const at::Tensor & self, int64_t diagonal, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("triu_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::triu_out(self, diagonal, out);
    } else {
        return acl_op::triu_out(self, diagonal, out);
    }
}
at::Tensor & trunc_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("trunc_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::trunc_(self);
    } else {
        return acl_op::trunc_(self);
    }
}
at::Tensor & trunc_out(const at::Tensor & self, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("trunc_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::trunc_out(self, out);
    } else {
        return acl_op::trunc_out(self, out);
    }
}
at::Tensor & uniform_(at::Tensor & self, double from, double to, c10::optional<at::Generator> generator){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("uniform_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::uniform_(self, from, to, generator);
    } else {
        return acl_op::uniform_(self, from, to, generator);
    }
}
at::Tensor & upsample_bicubic2d_backward_out(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    ASCEND_LOGI("upsample_bicubic2d_backward_out exec with jit compile: %d, grad_output is internal format: %d, grad_input is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !grad_input_base_format);
    if (is_jit_disable && grad_output_base_format && grad_input_base_format) {
        return op_api::upsample_bicubic2d_backward_out(grad_output, output_size, input_size, align_corners, scales_h, scales_w, grad_input);
    } else {
        return acl_op::upsample_bicubic2d_backward_out(grad_output, output_size, input_size, align_corners, scales_h, scales_w, grad_input);
    }
}
at::Tensor & upsample_bicubic2d_out(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("upsample_bicubic2d_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::upsample_bicubic2d_out(self, output_size, align_corners, scales_h, scales_w, out);
    } else {
        return acl_op::upsample_bicubic2d_out(self, output_size, align_corners, scales_h, scales_w, out);
    }
}
at::Tensor & upsample_bilinear2d_backward_out(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    ASCEND_LOGI("upsample_bilinear2d_backward_out exec with jit compile: %d, grad_output is internal format: %d, grad_input is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !grad_input_base_format);
    if (is_jit_disable && grad_output_base_format && grad_input_base_format) {
        return op_api::upsample_bilinear2d_backward_out(grad_output, output_size, input_size, align_corners, scales_h, scales_w, grad_input);
    } else {
        return acl_op::upsample_bilinear2d_backward_out(grad_output, output_size, input_size, align_corners, scales_h, scales_w, grad_input);
    }
}
at::Tensor & upsample_bilinear2d_out(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("upsample_bilinear2d_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::upsample_bilinear2d_out(self, output_size, align_corners, scales_h, scales_w, out);
    } else {
        return acl_op::upsample_bilinear2d_out(self, output_size, align_corners, scales_h, scales_w, out);
    }
}
at::Tensor & upsample_linear1d_out(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("upsample_linear1d_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::upsample_linear1d_out(self, output_size, align_corners, scales, out);
    } else {
        return acl_op::upsample_linear1d_out(self, output_size, align_corners, scales, out);
    }
}
at::Tensor & upsample_nearest1d_backward_out(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales, at::Tensor & grad_input){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    ASCEND_LOGI("upsample_nearest1d_backward_out exec with jit compile: %d, grad_output is internal format: %d, grad_input is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !grad_input_base_format);
    if (is_jit_disable && grad_output_base_format && grad_input_base_format) {
        return op_api::upsample_nearest1d_backward_out(grad_output, output_size, input_size, scales, grad_input);
    } else {
        return acl_op::upsample_nearest1d_backward_out(grad_output, output_size, input_size, scales, grad_input);
    }
}
at::Tensor & upsample_nearest1d_out(const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("upsample_nearest1d_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::upsample_nearest1d_out(self, output_size, scales, out);
    } else {
        return acl_op::upsample_nearest1d_out(self, output_size, scales, out);
    }
}
at::Tensor & upsample_nearest2d_backward_out(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    ASCEND_LOGI("upsample_nearest2d_backward_out exec with jit compile: %d, grad_output is internal format: %d, grad_input is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !grad_input_base_format);
    if (is_jit_disable && grad_output_base_format && grad_input_base_format) {
        return op_api::upsample_nearest2d_backward_out(grad_output, output_size, input_size, scales_h, scales_w, grad_input);
    } else {
        return acl_op::upsample_nearest2d_backward_out(grad_output, output_size, input_size, scales_h, scales_w, grad_input);
    }
}
at::Tensor & upsample_nearest2d_out(const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("upsample_nearest2d_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::upsample_nearest2d_out(self, output_size, scales_h, scales_w, out);
    } else {
        return acl_op::upsample_nearest2d_out(self, output_size, scales_h, scales_w, out);
    }
}
at::Tensor & upsample_nearest3d_backward_out(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    ASCEND_LOGI("upsample_nearest3d_backward_out exec with jit compile: %d, grad_output is internal format: %d, grad_input is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !grad_input_base_format);
    if (is_jit_disable && grad_output_base_format && grad_input_base_format) {
        return op_api::upsample_nearest3d_backward_out(grad_output, output_size, input_size, scales_d, scales_h, scales_w, grad_input);
    } else {
        return acl_op::upsample_nearest3d_backward_out(grad_output, output_size, input_size, scales_d, scales_h, scales_w, grad_input);
    }
}
at::Tensor & upsample_nearest3d_out(const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("upsample_nearest3d_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::upsample_nearest3d_out(self, output_size, scales_d, scales_h, scales_w, out);
    } else {
        return acl_op::upsample_nearest3d_out(self, output_size, scales_d, scales_h, scales_w, out);
    }
}
at::Tensor & upsample_trilinear3d_backward_out(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool grad_input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_input);

    ASCEND_LOGI("upsample_trilinear3d_backward_out exec with jit compile: %d, grad_output is internal format: %d, grad_input is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !grad_input_base_format);
    if (is_jit_disable && grad_output_base_format && grad_input_base_format) {
        return op_api::upsample_trilinear3d_backward_out(grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w, grad_input);
    } else {
        return acl_op::upsample_trilinear3d_backward_out(grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w, grad_input);
    }
}
at::Tensor & upsample_trilinear3d_out(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("upsample_trilinear3d_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::upsample_trilinear3d_out(self, output_size, align_corners, scales_d, scales_h, scales_w, out);
    } else {
        return acl_op::upsample_trilinear3d_out(self, output_size, align_corners, scales_d, scales_h, scales_w, out);
    }
}
at::Tensor & var_out(const at::Tensor & self, at::OptionalIntArrayRef dim, const c10::optional<at::Scalar> & correction, bool keepdim, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("var_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::var_out(self, dim, correction, keepdim, out);
    } else {
        return acl_op::var_out(self, dim, correction, keepdim, out);
    }
}
at::Tensor & vdot_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("vdot_out exec with jit compile: %d, self is internal format: %d, other is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && other_base_format && out_base_format) {
        return op_api::vdot_out(self, other, out);
    } else {
        return acl_op::vdot_out(self, other, out);
    }
}
at::Tensor & where_out(const at::Tensor & condition, const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool condition_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(condition);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("where_out exec with jit compile: %d, condition is internal format: %d, self is internal format: %d, other is internal format: %d, out is internal format: %d",
                !is_jit_disable, !condition_base_format, !self_base_format, !other_base_format, !out_base_format);
    if (is_jit_disable && condition_base_format && self_base_format && other_base_format && out_base_format) {
        return op_api::where_out(condition, self, other, out);
    } else {
        return acl_op::where_out(condition, self, other, out);
    }
}
at::Tensor & xlogy_(at::Tensor & self, const at::Scalar & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("xlogy_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::xlogy_(self, other);
    } else {
        return acl_op::xlogy_(self, other);
    }
}
at::Tensor & xlogy_(at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("xlogy_ exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::xlogy_(self, other);
    } else {
        return acl_op::xlogy_(self, other);
    }
}
at::Tensor & xlogy_out(const at::Scalar & self, const at::Tensor & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("xlogy_out exec with jit compile: %d, other is internal format: %d, out is internal format: %d",
                !is_jit_disable, !other_base_format, !out_base_format);
    if (is_jit_disable && other_base_format && out_base_format) {
        return op_api::xlogy_out(self, other, out);
    } else {
        return acl_op::xlogy_out(self, other, out);
    }
}
at::Tensor & xlogy_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("xlogy_out exec with jit compile: %d, self is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && out_base_format) {
        return op_api::xlogy_out(self, other, out);
    } else {
        return acl_op::xlogy_out(self, other, out);
    }
}
at::Tensor & xlogy_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("xlogy_out exec with jit compile: %d, self is internal format: %d, other is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && other_base_format && out_base_format) {
        return op_api::xlogy_out(self, other, out);
    } else {
        return acl_op::xlogy_out(self, other, out);
    }
}
at::Tensor & zero_(at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("zero_ exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::zero_(self);
    } else {
        return acl_op::zero_(self);
    }
}
at::Tensor & zeros_out(at::IntArrayRef size, at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("zeros_out exec with jit compile: %d, out is internal format: %d",
                !is_jit_disable, !out_base_format);
    if (is_jit_disable && out_base_format) {
        return op_api::zeros_out(size, out);
    } else {
        return acl_op::zeros_out(size, out);
    }
}
at::Tensor __lshift__(const at::Tensor & self, const at::Scalar & other){
    return acl_op::__lshift__(self, other);
}
at::Tensor __lshift__(const at::Tensor & self, const at::Tensor & other){
    return acl_op::__lshift__(self, other);
}
at::Tensor __rshift__(const at::Tensor & self, const at::Scalar & other){
    return acl_op::__rshift__(self, other);
}
at::Tensor __rshift__(const at::Tensor & self, const at::Tensor & other){
    return acl_op::__rshift__(self, other);
}
at::Tensor __xor__(const at::Tensor & self, const at::Scalar & other){
    return acl_op::__xor__(self, other);
}
at::Tensor __xor__(const at::Tensor & self, const at::Tensor & other){
    return acl_op::__xor__(self, other);
}
at::Tensor _adaptive_avg_pool2d(const at::Tensor & self, at::IntArrayRef output_size){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("_adaptive_avg_pool2d exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::_adaptive_avg_pool2d(self, output_size);
    } else {
        return acl_op::_adaptive_avg_pool2d(self, output_size);
    }
}
at::Tensor _adaptive_avg_pool2d_backward(const at::Tensor & grad_output, const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("_adaptive_avg_pool2d_backward exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format) {
        return op_api::_adaptive_avg_pool2d_backward(grad_output, self);
    } else {
        return acl_op::_adaptive_avg_pool2d_backward(grad_output, self);
    }
}
at::Tensor _adaptive_avg_pool3d(const at::Tensor & self, at::IntArrayRef output_size){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("_adaptive_avg_pool3d exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::_adaptive_avg_pool3d(self, output_size);
    } else {
        return acl_op::_adaptive_avg_pool3d(self, output_size);
    }
}
at::Tensor _adaptive_avg_pool3d_backward(const at::Tensor & grad_output, const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("_adaptive_avg_pool3d_backward exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format) {
        return op_api::_adaptive_avg_pool3d_backward(grad_output, self);
    } else {
        return acl_op::_adaptive_avg_pool3d_backward(grad_output, self);
    }
}
at::Tensor _add_relu(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("_add_relu exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::_add_relu(self, other, alpha);
    } else {
        return acl_op::_add_relu(self, other, alpha);
    }
}
at::Tensor _cdist_backward(const at::Tensor & grad, const at::Tensor & x1, const at::Tensor & x2, double p, const at::Tensor & cdist){
    return acl_op::_cdist_backward(grad, x1, x2, p, cdist);
}
at::Tensor _cdist_forward(const at::Tensor & x1, const at::Tensor & x2, double p, c10::optional<int64_t> compute_mode){
    return acl_op::_cdist_forward(x1, x2, p, compute_mode);
}
at::Tensor _coalesce_sparse(const at::Tensor & self){
    return sparse::_coalesce_sparse(self);
}
at::Tensor _conv_depthwise2d(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias);

    ASCEND_LOGI("_conv_depthwise2d exec with jit compile: %d, self is internal format: %d, weight is internal format: %d, bias is internal format: %d",
                !is_jit_disable, !self_base_format, !weight_base_format, !bias_base_format);
    if (is_jit_disable && self_base_format && weight_base_format && bias_base_format) {
        return op_api::_conv_depthwise2d(self, weight, kernel_size, bias, stride, padding, dilation);
    } else {
        return acl_op::_conv_depthwise2d(self, weight, kernel_size, bias, stride, padding, dilation);
    }
}
at::Tensor _convolution(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias);

    ASCEND_LOGI("_convolution exec with jit compile: %d, input is internal format: %d, weight is internal format: %d, bias is internal format: %d",
                !is_jit_disable, !input_base_format, !weight_base_format, !bias_base_format);
    if (is_jit_disable && input_base_format && weight_base_format && bias_base_format) {
        return op_api::_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32);
    } else {
        return acl_op::_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32);
    }
}
at::Tensor _ctc_loss_backward(const at::Tensor & grad, const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, const at::Tensor & neg_log_likelihood, const at::Tensor & log_alpha, int64_t blank, bool zero_infinity){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad);
    bool log_probs_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(log_probs);
    bool targets_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(targets);
    bool neg_log_likelihood_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(neg_log_likelihood);
    bool log_alpha_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(log_alpha);

    ASCEND_LOGI("_ctc_loss_backward exec with jit compile: %d, grad is internal format: %d, log_probs is internal format: %d, targets is internal format: %d, neg_log_likelihood is internal format: %d, log_alpha is internal format: %d",
                !is_jit_disable, !grad_base_format, !log_probs_base_format, !targets_base_format, !neg_log_likelihood_base_format, !log_alpha_base_format);
    if (is_jit_disable && grad_base_format && log_probs_base_format && targets_base_format && neg_log_likelihood_base_format && log_alpha_base_format) {
        return op_api::_ctc_loss_backward(grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank, zero_infinity);
    } else {
        return acl_op::_ctc_loss_backward(grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank, zero_infinity);
    }
}
at::Tensor _dim_arange(const at::Tensor & like, int64_t dim){
    return acl_op::_dim_arange(like, dim);
}
at::Tensor _dropout_with_byte_mask_backward(const at::Tensor & grad_output, const at::Tensor & mask, double p){
    return acl_op::_dropout_with_byte_mask_backward(grad_output, mask, p);
}
at::Tensor _embedding_bag_per_sample_weights_backward(const at::Tensor & grad, const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, const at::Tensor & offset2bag, int64_t mode, int64_t padding_idx){
    return acl_op::_embedding_bag_per_sample_weights_backward(grad, weight, indices, offsets, offset2bag, mode, padding_idx);
}
at::Tensor _empty_affine_quantized(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, double scale, int64_t zero_point, c10::optional<at::MemoryFormat> memory_format){
    return acl_op::_empty_affine_quantized(size, dtype, layout, device, pin_memory, scale, zero_point, memory_format);
}
at::Tensor _fft_c2c(const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, bool forward){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _fft_c2c do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_fft_c2c(self, dim, normalization, forward);
}
at::Tensor _fft_c2r(const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, int64_t last_dim_size){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _fft_c2r do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_fft_c2r(self, dim, normalization, last_dim_size);
}
at::Tensor _fft_r2c(const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, bool onesided){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _fft_r2c do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_fft_r2c(self, dim, normalization, onesided);
}
at::Tensor _log_softmax(const at::Tensor & self, int64_t dim, bool half_to_float){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("_log_softmax exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::_log_softmax(self, dim, half_to_float);
    } else {
        return acl_op::_log_softmax(self, dim, half_to_float);
    }
}
at::Tensor _log_softmax_backward_data(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, at::ScalarType input_dtype){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(output);

    ASCEND_LOGI("_log_softmax_backward_data exec with jit compile: %d, grad_output is internal format: %d, output is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !output_base_format);
    if (is_jit_disable && grad_output_base_format && output_base_format) {
        return op_api::_log_softmax_backward_data(grad_output, output, dim, input_dtype);
    } else {
        return acl_op::_log_softmax_backward_data(grad_output, output, dim, input_dtype);
    }
}
at::Tensor _masked_softmax(const at::Tensor & self, const at::Tensor & mask, c10::optional<int64_t> dim, c10::optional<int64_t> mask_type){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mask);

    if (!self_base_format || !mask_base_format) {
        TORCH_CHECK(false,
            "Current operator _masked_softmax do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_masked_softmax(self, mask, dim, mask_type);
}
at::Tensor _nnpack_spatial_convolution(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef stride){
    return acl_op::_nnpack_spatial_convolution(input, weight, bias, padding, stride);
}
at::Tensor _npu_distribute_barrier(const at::Tensor & x_ref, c10::string_view group, int64_t world_size){
    bool x_ref_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x_ref);

    if (!x_ref_base_format) {
        TORCH_CHECK(false,
            "Current operator _npu_distribute_barrier do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_npu_distribute_barrier(x_ref, group, world_size);
}
at::Tensor _npu_dropout_gen_mask(const at::Tensor & self, at::IntArrayRef size, double p, int64_t seed, int64_t offset, c10::optional<bool> parallel, c10::optional<bool> sync){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("_npu_dropout_gen_mask exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::_npu_dropout_gen_mask(self, size, p, seed, offset, parallel, sync);
    } else {
        return acl_op::_npu_dropout_gen_mask(self, size, p, seed, offset, parallel, sync);
    }
}
at::Tensor _npu_fused_infer_attention_score_get_max_workspace_symint(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const c10::optional<at::Tensor> & pse_shift, const c10::optional<at::Tensor> & atten_mask, at::OptionalSymIntArrayRef actual_seq_lengths, at::OptionalSymIntArrayRef actual_seq_lengths_kv, const c10::optional<at::Tensor> & dequant_scale1, const c10::optional<at::Tensor> & quant_scale1, const c10::optional<at::Tensor> & dequant_scale2, const c10::optional<at::Tensor> & quant_scale2, const c10::optional<at::Tensor> & quant_offset2, const c10::optional<at::Tensor> & antiquant_scale, const c10::optional<at::Tensor> & antiquant_offset, const c10::optional<at::Tensor> & key_antiquant_scale, const c10::optional<at::Tensor> & key_antiquant_offset, const c10::optional<at::Tensor> & value_antiquant_scale, const c10::optional<at::Tensor> & value_antiquant_offset, const c10::optional<at::Tensor> & block_table, const c10::optional<at::Tensor> & query_padding_size, const c10::optional<at::Tensor> & kv_padding_size, const c10::optional<at::Tensor> & key_shared_prefix, const c10::optional<at::Tensor> & value_shared_prefix, at::OptionalSymIntArrayRef actual_shared_prefix_len, const c10::optional<at::Tensor> & query_rope, const c10::optional<at::Tensor> & key_rope, const c10::optional<at::Tensor> & key_rope_antiquant_scale, int64_t num_heads, double scale, int64_t pre_tokens, int64_t next_tokens, c10::string_view input_layout, int64_t num_key_value_heads, int64_t sparse_mode, int64_t inner_precise, int64_t block_size, int64_t antiquant_mode, int64_t key_antiquant_mode, int64_t value_antiquant_mode, bool softmax_lse_flag){
    bool query_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(query);
    bool key_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(key);
    bool value_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(value);
    bool pse_shift_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(pse_shift);
    bool atten_mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(atten_mask);
    bool dequant_scale1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(dequant_scale1);
    bool quant_scale1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(quant_scale1);
    bool dequant_scale2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(dequant_scale2);
    bool quant_scale2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(quant_scale2);
    bool quant_offset2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(quant_offset2);
    bool antiquant_scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(antiquant_scale);
    bool antiquant_offset_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(antiquant_offset);
    bool key_antiquant_scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(key_antiquant_scale);
    bool key_antiquant_offset_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(key_antiquant_offset);
    bool value_antiquant_scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(value_antiquant_scale);
    bool value_antiquant_offset_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(value_antiquant_offset);
    bool block_table_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(block_table);
    bool query_padding_size_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(query_padding_size);
    bool kv_padding_size_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(kv_padding_size);
    bool key_shared_prefix_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(key_shared_prefix);
    bool value_shared_prefix_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(value_shared_prefix);
    bool query_rope_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(query_rope);
    bool key_rope_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(key_rope);
    bool key_rope_antiquant_scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(key_rope_antiquant_scale);

    if (!query_base_format || !key_base_format || !value_base_format || !pse_shift_base_format || !atten_mask_base_format || !dequant_scale1_base_format || !quant_scale1_base_format || !dequant_scale2_base_format || !quant_scale2_base_format || !quant_offset2_base_format || !antiquant_scale_base_format || !antiquant_offset_base_format || !key_antiquant_scale_base_format || !key_antiquant_offset_base_format || !value_antiquant_scale_base_format || !value_antiquant_offset_base_format || !block_table_base_format || !query_padding_size_base_format || !kv_padding_size_base_format || !key_shared_prefix_base_format || !value_shared_prefix_base_format || !query_rope_base_format || !key_rope_base_format || !key_rope_antiquant_scale_base_format) {
        TORCH_CHECK(false,
            "Current operator _npu_fused_infer_attention_score_get_max_workspace_symint do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_npu_fused_infer_attention_score_get_max_workspace_symint(query, key, value, pse_shift, atten_mask, actual_seq_lengths, actual_seq_lengths_kv, dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2, antiquant_scale, antiquant_offset, key_antiquant_scale, key_antiquant_offset, value_antiquant_scale, value_antiquant_offset, block_table, query_padding_size, kv_padding_size, key_shared_prefix, value_shared_prefix, actual_shared_prefix_len, query_rope, key_rope, key_rope_antiquant_scale, num_heads, scale, pre_tokens, next_tokens, input_layout, num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, key_antiquant_mode, value_antiquant_mode, softmax_lse_flag);
}
at::Tensor _npu_silent_check(at::Tensor & input_grad, const at::Tensor & val, at::Tensor & pre_val, at::Tensor & min_val, at::Tensor & max_val, const at::Tensor & val_counter, int64_t c_min_steps, double c_thresh_l1, double c_coeff_l1, double c_thresh_l2, double c_coeff_l2){
    return acl_op::_npu_silent_check(input_grad, val, pre_val, min_val, max_val, val_counter, c_min_steps, c_thresh_l1, c_coeff_l1, c_thresh_l2, c_coeff_l2);
}
at::Tensor _npu_silent_check_v2(const at::Tensor & val, at::Tensor & input_grad, at::Tensor & sfda, at::Tensor & step, int64_t c_min_steps, double c_thresh_l1, double c_coeff_l1, double c_thresh_l2, double c_coeff_l2, int64_t npu_asd_detect){
    bool val_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(val);
    bool input_grad_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input_grad);
    bool sfda_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(sfda);
    bool step_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(step);

    if (!val_base_format || !input_grad_base_format || !sfda_base_format || !step_base_format) {
        TORCH_CHECK(false,
            "Current operator _npu_silent_check_v2 do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_npu_silent_check_v2(val, input_grad, sfda, step, c_min_steps, c_thresh_l1, c_coeff_l1, c_thresh_l2, c_coeff_l2, npu_asd_detect);
}
at::Tensor _npu_silent_check_v3(const at::Tensor & val, at::Tensor & input_grad, at::Tensor & step, at::Tensor & max, at::Tensor & avg, double c_thresh_l1, double c_thresh_l2, double betal, int64_t npu_asd_detect){
    bool val_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(val);
    bool input_grad_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input_grad);
    bool step_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(step);
    bool max_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(max);
    bool avg_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(avg);

    if (!val_base_format || !input_grad_base_format || !step_base_format || !max_base_format || !avg_base_format) {
        TORCH_CHECK(false,
            "Current operator _npu_silent_check_v3 do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_npu_silent_check_v3(val, input_grad, step, max, avg, c_thresh_l1, c_thresh_l2, betal, npu_asd_detect);
}
at::Tensor _pdist_forward(const at::Tensor & self, double p){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("_pdist_forward exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::_pdist_forward(self, p);
    } else {
        return acl_op::_pdist_forward(self, p);
    }
}
at::Tensor _prelu_kernel(const at::Tensor & self, const at::Tensor & weight){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);

    ASCEND_LOGI("_prelu_kernel exec with jit compile: %d, self is internal format: %d, weight is internal format: %d",
                !is_jit_disable, !self_base_format, !weight_base_format);
    if (is_jit_disable && self_base_format && weight_base_format) {
        return op_api::_prelu_kernel(self, weight);
    } else {
        return acl_op::_prelu_kernel(self, weight);
    }
}
at::Tensor _slow_conv2d_forward(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias);

    ASCEND_LOGI("_slow_conv2d_forward exec with jit compile: %d, self is internal format: %d, weight is internal format: %d, bias is internal format: %d",
                !is_jit_disable, !self_base_format, !weight_base_format, !bias_base_format);
    if (is_jit_disable && self_base_format && weight_base_format && bias_base_format) {
        return op_api::_slow_conv2d_forward(self, weight, kernel_size, bias, stride, padding);
    } else {
        return acl_op::_slow_conv2d_forward(self, weight, kernel_size, bias, stride, padding);
    }
}
at::Tensor _softmax(const at::Tensor & self, int64_t dim, bool half_to_float){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("_softmax exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::_softmax(self, dim, half_to_float);
    } else {
        return acl_op::_softmax(self, dim, half_to_float);
    }
}
at::Tensor _softmax_backward_data(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, at::ScalarType input_dtype){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(output);

    ASCEND_LOGI("_softmax_backward_data exec with jit compile: %d, grad_output is internal format: %d, output is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !output_base_format);
    if (is_jit_disable && grad_output_base_format && output_base_format) {
        return op_api::_softmax_backward_data(grad_output, output, dim, input_dtype);
    } else {
        return acl_op::_softmax_backward_data(grad_output, output, dim, input_dtype);
    }
}
at::Tensor _upsample_bicubic2d_aa(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _upsample_bicubic2d_aa do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_upsample_bicubic2d_aa(self, output_size, align_corners, scales_h, scales_w);
}
at::Tensor _upsample_bicubic2d_aa_backward(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w){
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);

    if (!grad_output_base_format) {
        TORCH_CHECK(false,
            "Current operator _upsample_bicubic2d_aa_backward do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_upsample_bicubic2d_aa_backward(grad_output, output_size, input_size, align_corners, scales_h, scales_w);
}
at::Tensor _upsample_bilinear2d_aa(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _upsample_bilinear2d_aa do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_upsample_bilinear2d_aa(self, output_size, align_corners, scales_h, scales_w);
}
at::Tensor _upsample_bilinear2d_aa_backward(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w){
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);

    if (!grad_output_base_format) {
        TORCH_CHECK(false,
            "Current operator _upsample_bilinear2d_aa_backward do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_upsample_bilinear2d_aa_backward(grad_output, output_size, input_size, align_corners, scales_h, scales_w);
}
at::Tensor _upsample_nearest_exact1d(const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _upsample_nearest_exact1d do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_upsample_nearest_exact1d(self, output_size, scales);
}
at::Tensor _upsample_nearest_exact1d_backward(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales){
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);

    if (!grad_output_base_format) {
        TORCH_CHECK(false,
            "Current operator _upsample_nearest_exact1d_backward do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_upsample_nearest_exact1d_backward(grad_output, output_size, input_size, scales);
}
at::Tensor _upsample_nearest_exact2d(const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _upsample_nearest_exact2d do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_upsample_nearest_exact2d(self, output_size, scales_h, scales_w);
}
at::Tensor _upsample_nearest_exact2d_backward(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales_h, c10::optional<double> scales_w){
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);

    if (!grad_output_base_format) {
        TORCH_CHECK(false,
            "Current operator _upsample_nearest_exact2d_backward do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_upsample_nearest_exact2d_backward(grad_output, output_size, input_size, scales_h, scales_w);
}
at::Tensor _upsample_nearest_exact3d(const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _upsample_nearest_exact3d do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_upsample_nearest_exact3d(self, output_size, scales_d, scales_h, scales_w);
}
at::Tensor _upsample_nearest_exact3d_backward(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w){
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);

    if (!grad_output_base_format) {
        TORCH_CHECK(false,
            "Current operator _upsample_nearest_exact3d_backward do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_upsample_nearest_exact3d_backward(grad_output, output_size, input_size, scales_d, scales_h, scales_w);
}
at::Tensor _weight_norm(const at::Tensor & v, const at::Tensor & g, int64_t dim){
    return acl_op::_weight_norm(v, g, dim);
}
at::Tensor abs(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("abs exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::abs(self);
    } else {
        return acl_op::abs(self);
    }
}
at::Tensor acos(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("acos exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::acos(self);
    } else {
        return acl_op::acos(self);
    }
}
at::Tensor acosh(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("acosh exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::acosh(self);
    } else {
        return acl_op::acosh(self);
    }
}
at::Tensor adaptive_avg_pool1d(const at::Tensor & self, at::IntArrayRef output_size){
    return acl_op::adaptive_avg_pool1d(self, output_size);
}
at::Tensor adaptive_avg_pool2d(const at::Tensor & self, at::IntArrayRef output_size){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("adaptive_avg_pool2d exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::adaptive_avg_pool2d(self, output_size);
    } else {
        return acl_op::adaptive_avg_pool2d(self, output_size);
    }
}
at::Tensor adaptive_max_pool2d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);

    ASCEND_LOGI("adaptive_max_pool2d_backward exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, indices is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !indices_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && indices_base_format) {
        return op_api::adaptive_max_pool2d_backward(grad_output, self, indices);
    } else {
        return acl_op::adaptive_max_pool2d_backward(grad_output, self, indices);
    }
}
at::Tensor adaptive_max_pool3d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices){
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);

    if (!grad_output_base_format || !self_base_format || !indices_base_format) {
        TORCH_CHECK(false,
            "Current operator adaptive_max_pool3d_backward do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::adaptive_max_pool3d_backward(grad_output, self, indices);
}
at::Tensor add(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("add exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::add(self, other, alpha);
    } else {
        return acl_op::add(self, other, alpha);
    }
}
at::Tensor add(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("add exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::add(self, other, alpha);
    } else {
        return acl_op::add(self, other, alpha);
    }
}
at::Tensor addbmm(const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool batch1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(batch1);
    bool batch2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(batch2);

    ASCEND_LOGI("addbmm exec with jit compile: %d, self is internal format: %d, batch1 is internal format: %d, batch2 is internal format: %d",
                !is_jit_disable, !self_base_format, !batch1_base_format, !batch2_base_format);
    if (is_jit_disable && self_base_format && batch1_base_format && batch2_base_format) {
        return op_api::addbmm(self, batch1, batch2, beta, alpha);
    } else {
        return acl_op::addbmm(self, batch1, batch2, beta, alpha);
    }
}
at::Tensor addcdiv(const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool tensor1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor1);
    bool tensor2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor2);

    ASCEND_LOGI("addcdiv exec with jit compile: %d, self is internal format: %d, tensor1 is internal format: %d, tensor2 is internal format: %d",
                !is_jit_disable, !self_base_format, !tensor1_base_format, !tensor2_base_format);
    if (is_jit_disable && self_base_format && tensor1_base_format && tensor2_base_format) {
        return op_api::addcdiv(self, tensor1, tensor2, value);
    } else {
        return acl_op::addcdiv(self, tensor1, tensor2, value);
    }
}
at::Tensor addcmul(const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool tensor1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor1);
    bool tensor2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor2);

    ASCEND_LOGI("addcmul exec with jit compile: %d, self is internal format: %d, tensor1 is internal format: %d, tensor2 is internal format: %d",
                !is_jit_disable, !self_base_format, !tensor1_base_format, !tensor2_base_format);
    if (is_jit_disable && self_base_format && tensor1_base_format && tensor2_base_format) {
        return op_api::addcmul(self, tensor1, tensor2, value);
    } else {
        return acl_op::addcmul(self, tensor1, tensor2, value);
    }
}
at::Tensor addmm(const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool mat1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mat1);
    bool mat2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mat2);

    ASCEND_LOGI("addmm exec with jit compile: %d, self is internal format: %d, mat1 is internal format: %d, mat2 is internal format: %d",
                !is_jit_disable, !self_base_format, !mat1_base_format, !mat2_base_format);
    if (is_jit_disable && self_base_format && mat1_base_format && mat2_base_format) {
        return op_api::addmm(self, mat1, mat2, beta, alpha);
    } else {
        return acl_op::addmm(self, mat1, mat2, beta, alpha);
    }
}
at::Tensor addmv(const at::Tensor & self, const at::Tensor & mat, const at::Tensor & vec, const at::Scalar & beta, const at::Scalar & alpha){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool mat_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mat);
    bool vec_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(vec);

    ASCEND_LOGI("addmv exec with jit compile: %d, self is internal format: %d, mat is internal format: %d, vec is internal format: %d",
                !is_jit_disable, !self_base_format, !mat_base_format, !vec_base_format);
    if (is_jit_disable && self_base_format && mat_base_format && vec_base_format) {
        return op_api::addmv(self, mat, vec, beta, alpha);
    } else {
        return acl_op::addmv(self, mat, vec, beta, alpha);
    }
}
at::Tensor addr(const at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2, const at::Scalar & beta, const at::Scalar & alpha){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool vec1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(vec1);
    bool vec2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(vec2);

    ASCEND_LOGI("addr exec with jit compile: %d, self is internal format: %d, vec1 is internal format: %d, vec2 is internal format: %d",
                !is_jit_disable, !self_base_format, !vec1_base_format, !vec2_base_format);
    if (is_jit_disable && self_base_format && vec1_base_format && vec2_base_format) {
        return op_api::addr(self, vec1, vec2, beta, alpha);
    } else {
        return acl_op::addr(self, vec1, vec2, beta, alpha);
    }
}
at::Tensor affine_grid_generator(const at::Tensor & theta, at::IntArrayRef size, bool align_corners){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool theta_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(theta);

    ASCEND_LOGI("affine_grid_generator exec with jit compile: %d, theta is internal format: %d",
                !is_jit_disable, !theta_base_format);
    if (is_jit_disable && theta_base_format) {
        return op_api::affine_grid_generator(theta, size, align_corners);
    } else {
        return acl_op::affine_grid_generator(theta, size, align_corners);
    }
}
at::Tensor affine_grid_generator_backward(const at::Tensor & grad, at::IntArrayRef size, bool align_corners){
    return acl_op::affine_grid_generator_backward(grad, size, align_corners);
}
at::Tensor all(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("all exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::all(self);
    } else {
        return acl_op::all(self);
    }
}
at::Tensor all(const at::Tensor & self, int64_t dim, bool keepdim){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("all exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::all(self, dim, keepdim);
    } else {
        return acl_op::all(self, dim, keepdim);
    }
}
at::Tensor amax(const at::Tensor & self, at::IntArrayRef dim, bool keepdim){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("amax exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::amax(self, dim, keepdim);
    } else {
        return acl_op::amax(self, dim, keepdim);
    }
}
at::Tensor amin(const at::Tensor & self, at::IntArrayRef dim, bool keepdim){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("amin exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::amin(self, dim, keepdim);
    } else {
        return acl_op::amin(self, dim, keepdim);
    }
}
at::Tensor angle(const at::Tensor & self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator angle do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::angle(self);
}
at::Tensor any(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("any exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::any(self);
    } else {
        return acl_op::any(self);
    }
}
at::Tensor any(const at::Tensor & self, int64_t dim, bool keepdim){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("any exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::any(self, dim, keepdim);
    } else {
        return acl_op::any(self, dim, keepdim);
    }
}
at::Tensor arange(const at::Scalar & end, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();

    ASCEND_LOGI("arange exec with jit compile: %d",
                !is_jit_disable);
    if (is_jit_disable) {
        return op_api::arange(end, dtype, layout, device, pin_memory);
    } else {
        return acl_op::arange(end, dtype, layout, device, pin_memory);
    }
}
at::Tensor arange(const at::Scalar & start, const at::Scalar & end, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();

    ASCEND_LOGI("arange exec with jit compile: %d",
                !is_jit_disable);
    if (is_jit_disable) {
        return op_api::arange(start, end, dtype, layout, device, pin_memory);
    } else {
        return acl_op::arange(start, end, dtype, layout, device, pin_memory);
    }
}
at::Tensor arange(const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();

    ASCEND_LOGI("arange exec with jit compile: %d",
                !is_jit_disable);
    if (is_jit_disable) {
        return op_api::arange(start, end, step, dtype, layout, device, pin_memory);
    } else {
        return acl_op::arange(start, end, step, dtype, layout, device, pin_memory);
    }
}
at::Tensor argmin(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("argmin exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::argmin(self, dim, keepdim);
    } else {
        return acl_op::argmin(self, dim, keepdim);
    }
}
at::Tensor argsort(const at::Tensor & self, at::Dimname dim, bool descending){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("argsort exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::argsort(self, dim, descending);
    } else {
        return acl_op::argsort(self, dim, descending);
    }
}
at::Tensor argsort(const at::Tensor & self, bool stable, int64_t dim, bool descending){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator argsort do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::argsort(self, stable, dim, descending);
}
at::Tensor argsort(const at::Tensor & self, int64_t dim, bool descending){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("argsort exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::argsort(self, dim, descending);
    } else {
        return acl_op::argsort(self, dim, descending);
    }
}
at::Tensor asin(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("asin exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::asin(self);
    } else {
        return acl_op::asin(self);
    }
}
at::Tensor asinh(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("asinh exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::asinh(self);
    } else {
        return acl_op::asinh(self);
    }
}
at::Tensor atan(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("atan exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::atan(self);
    } else {
        return acl_op::atan(self);
    }
}
at::Tensor atan2(const at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("atan2 exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::atan2(self, other);
    } else {
        return acl_op::atan2(self, other);
    }
}
at::Tensor atanh(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("atanh exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::atanh(self);
    } else {
        return acl_op::atanh(self);
    }
}
at::Tensor avg_pool2d(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("avg_pool2d exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::avg_pool2d(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
    } else {
        return acl_op::avg_pool2d(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
    }
}
at::Tensor avg_pool2d_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("avg_pool2d_backward exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format) {
        return op_api::avg_pool2d_backward(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
    } else {
        return acl_op::avg_pool2d_backward(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
    }
}
at::Tensor avg_pool3d(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("avg_pool3d exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::avg_pool3d(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
    } else {
        return acl_op::avg_pool3d(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
    }
}
at::Tensor avg_pool3d_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("avg_pool3d_backward exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format) {
        return op_api::avg_pool3d_backward(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
    } else {
        return acl_op::avg_pool3d_backward(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
    }
}
at::Tensor baddbmm(const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool batch1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(batch1);
    bool batch2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(batch2);

    ASCEND_LOGI("baddbmm exec with jit compile: %d, self is internal format: %d, batch1 is internal format: %d, batch2 is internal format: %d",
                !is_jit_disable, !self_base_format, !batch1_base_format, !batch2_base_format);
    if (is_jit_disable && self_base_format && batch1_base_format && batch2_base_format) {
        return op_api::baddbmm(self, batch1, batch2, beta, alpha);
    } else {
        return acl_op::baddbmm(self, batch1, batch2, beta, alpha);
    }
}
at::Tensor batch_norm(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double momentum, double eps, bool cudnn_enabled){
    return acl_op::batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
}
at::Tensor batch_norm_backward_elemt(const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const c10::optional<at::Tensor> & weight, const at::Tensor & sum_dy, const at::Tensor & sum_dy_xmu, const at::Tensor & count){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_out);
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool mean_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mean);
    bool invstd_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(invstd);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool sum_dy_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(sum_dy);
    bool sum_dy_xmu_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(sum_dy_xmu);
    bool count_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(count);

    ASCEND_LOGI("batch_norm_backward_elemt exec with jit compile: %d, grad_out is internal format: %d, input is internal format: %d, mean is internal format: %d, invstd is internal format: %d, weight is internal format: %d, sum_dy is internal format: %d, sum_dy_xmu is internal format: %d, count is internal format: %d",
                !is_jit_disable, !grad_out_base_format, !input_base_format, !mean_base_format, !invstd_base_format, !weight_base_format, !sum_dy_base_format, !sum_dy_xmu_base_format, !count_base_format);
    if (is_jit_disable && grad_out_base_format && input_base_format && mean_base_format && invstd_base_format && weight_base_format && sum_dy_base_format && sum_dy_xmu_base_format && count_base_format) {
        return op_api::batch_norm_backward_elemt(grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count);
    } else {
        return acl_op::batch_norm_backward_elemt(grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count);
    }
}
at::Tensor batch_norm_elemt(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const at::Tensor & mean, const at::Tensor & invstd, double eps){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias);
    bool mean_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mean);
    bool invstd_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(invstd);

    ASCEND_LOGI("batch_norm_elemt exec with jit compile: %d, input is internal format: %d, weight is internal format: %d, bias is internal format: %d, mean is internal format: %d, invstd is internal format: %d",
                !is_jit_disable, !input_base_format, !weight_base_format, !bias_base_format, !mean_base_format, !invstd_base_format);
    if (is_jit_disable && input_base_format && weight_base_format && bias_base_format && mean_base_format && invstd_base_format) {
        return op_api::batch_norm_elemt(input, weight, bias, mean, invstd, eps);
    } else {
        return acl_op::batch_norm_elemt(input, weight, bias, mean, invstd, eps);
    }
}
at::Tensor bernoulli(const at::Tensor & self, c10::optional<at::Generator> generator){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("bernoulli exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::bernoulli(self, generator);
    } else {
        return acl_op::bernoulli(self, generator);
    }
}
at::Tensor bernoulli(const at::Tensor & self, double p, c10::optional<at::Generator> generator){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("bernoulli exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::bernoulli(self, p, generator);
    } else {
        return acl_op::bernoulli(self, p, generator);
    }
}
at::Tensor binary_cross_entropy(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);

    ASCEND_LOGI("binary_cross_entropy exec with jit compile: %d, self is internal format: %d, target is internal format: %d, weight is internal format: %d",
                !is_jit_disable, !self_base_format, !target_base_format, !weight_base_format);
    if (is_jit_disable && self_base_format && target_base_format && weight_base_format) {
        return op_api::binary_cross_entropy(self, target, weight, reduction);
    } else {
        return acl_op::binary_cross_entropy(self, target, weight, reduction);
    }
}
at::Tensor binary_cross_entropy_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);

    ASCEND_LOGI("binary_cross_entropy_backward exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, target is internal format: %d, weight is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !target_base_format, !weight_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && target_base_format && weight_base_format) {
        return op_api::binary_cross_entropy_backward(grad_output, self, target, weight, reduction);
    } else {
        return acl_op::binary_cross_entropy_backward(grad_output, self, target, weight, reduction);
    }
}
at::Tensor binary_cross_entropy_with_logits(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & pos_weight, int64_t reduction){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool pos_weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(pos_weight);

    ASCEND_LOGI("binary_cross_entropy_with_logits exec with jit compile: %d, self is internal format: %d, target is internal format: %d, weight is internal format: %d, pos_weight is internal format: %d",
                !is_jit_disable, !self_base_format, !target_base_format, !weight_base_format, !pos_weight_base_format);
    if (is_jit_disable && self_base_format && target_base_format && weight_base_format && pos_weight_base_format) {
        return op_api::binary_cross_entropy_with_logits(self, target, weight, pos_weight, reduction);
    } else {
        return acl_op::binary_cross_entropy_with_logits(self, target, weight, pos_weight, reduction);
    }
}
at::Tensor bincount(const at::Tensor & self, const c10::optional<at::Tensor> & weights, int64_t minlength){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool weights_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weights);

    ASCEND_LOGI("bincount exec with jit compile: %d, self is internal format: %d, weights is internal format: %d",
                !is_jit_disable, !self_base_format, !weights_base_format);
    if (is_jit_disable && self_base_format && weights_base_format) {
        return op_api::bincount(self, weights, minlength);
    } else {
        return acl_op::bincount(self, weights, minlength);
    }
}
at::Tensor bitwise_and(const at::Tensor & self, const at::Scalar & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("bitwise_and exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::bitwise_and(self, other);
    } else {
        return acl_op::bitwise_and(self, other);
    }
}
at::Tensor bitwise_and(const at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("bitwise_and exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::bitwise_and(self, other);
    } else {
        return acl_op::bitwise_and(self, other);
    }
}
at::Tensor bitwise_not(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("bitwise_not exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::bitwise_not(self);
    } else {
        return acl_op::bitwise_not(self);
    }
}
at::Tensor bitwise_or(const at::Tensor & self, const at::Scalar & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("bitwise_or exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::bitwise_or(self, other);
    } else {
        return acl_op::bitwise_or(self, other);
    }
}
at::Tensor bitwise_or(const at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("bitwise_or exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::bitwise_or(self, other);
    } else {
        return acl_op::bitwise_or(self, other);
    }
}
at::Tensor bitwise_xor(const at::Tensor & self, const at::Scalar & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("bitwise_xor exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::bitwise_xor(self, other);
    } else {
        return acl_op::bitwise_xor(self, other);
    }
}
at::Tensor bitwise_xor(const at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("bitwise_xor exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::bitwise_xor(self, other);
    } else {
        return acl_op::bitwise_xor(self, other);
    }
}
at::Tensor bmm(const at::Tensor & self, const at::Tensor & mat2){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool mat2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mat2);

    ASCEND_LOGI("bmm exec with jit compile: %d, self is internal format: %d, mat2 is internal format: %d",
                !is_jit_disable, !self_base_format, !mat2_base_format);
    if (is_jit_disable && self_base_format && mat2_base_format) {
        return op_api::bmm(self, mat2);
    } else {
        return acl_op::bmm(self, mat2);
    }
}
at::Tensor bucketize(const at::Scalar & self, const at::Tensor & boundaries, bool out_int32, bool right){
    bool boundaries_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(boundaries);

    if (!boundaries_base_format) {
        TORCH_CHECK(false,
            "Current operator bucketize do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::bucketize(self, boundaries, out_int32, right);
}
at::Tensor bucketize(const at::Tensor & self, const at::Tensor & boundaries, bool out_int32, bool right){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool boundaries_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(boundaries);

    if (!self_base_format || !boundaries_base_format) {
        TORCH_CHECK(false,
            "Current operator bucketize do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::bucketize(self, boundaries, out_int32, right);
}
at::Tensor cat(at::TensorList tensors, at::Dimname dim){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool tensors_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors);

    ASCEND_LOGI("cat exec with jit compile: %d, tensors is internal format: %d",
                !is_jit_disable, !tensors_base_format);
    if (is_jit_disable && tensors_base_format) {
        return op_api::cat(tensors, dim);
    } else {
        return acl_op::cat(tensors, dim);
    }
}
at::Tensor cat(const at::ITensorListRef & tensors, int64_t dim){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool tensors_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors);

    ASCEND_LOGI("cat exec with jit compile: %d, tensors is internal format: %d",
                !is_jit_disable, !tensors_base_format);
    if (is_jit_disable && tensors_base_format) {
        return op_api::cat(tensors, dim);
    } else {
        return acl_op::cat(tensors, dim);
    }
}
at::Tensor cdist(const at::Tensor & x1, const at::Tensor & x2, double p, c10::optional<int64_t> compute_mode){
    return acl_op::cdist(x1, x2, p, compute_mode);
}
at::Tensor ceil(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("ceil exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::ceil(self);
    } else {
        return acl_op::ceil(self);
    }
}
at::Tensor celu(const at::Tensor & self, const at::Scalar & alpha){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("celu exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::celu(self, alpha);
    } else {
        return acl_op::celu(self, alpha);
    }
}
at::Tensor channel_shuffle(const at::Tensor & self, int64_t groups){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("channel_shuffle exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::channel_shuffle(self, groups);
    } else {
        return acl_op::channel_shuffle(self, groups);
    }
}
at::Tensor clamp(const at::Tensor & self, const c10::optional<at::Scalar> & min, const c10::optional<at::Scalar> & max){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("clamp exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::clamp(self, min, max);
    } else {
        return acl_op::clamp(self, min, max);
    }
}
at::Tensor clamp(const at::Tensor & self, const c10::optional<at::Tensor> & min, const c10::optional<at::Tensor> & max){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool min_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(min);
    bool max_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(max);

    ASCEND_LOGI("clamp exec with jit compile: %d, self is internal format: %d, min is internal format: %d, max is internal format: %d",
                !is_jit_disable, !self_base_format, !min_base_format, !max_base_format);
    if (is_jit_disable && self_base_format && min_base_format && max_base_format) {
        return op_api::clamp(self, min, max);
    } else {
        return acl_op::clamp(self, min, max);
    }
}
at::Tensor clamp_max(const at::Tensor & self, const at::Scalar & max){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("clamp_max exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::clamp_max(self, max);
    } else {
        return acl_op::clamp_max(self, max);
    }
}
at::Tensor clamp_max(const at::Tensor & self, const at::Tensor & max){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool max_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(max);

    ASCEND_LOGI("clamp_max exec with jit compile: %d, self is internal format: %d, max is internal format: %d",
                !is_jit_disable, !self_base_format, !max_base_format);
    if (is_jit_disable && self_base_format && max_base_format) {
        return op_api::clamp_max(self, max);
    } else {
        return acl_op::clamp_max(self, max);
    }
}
at::Tensor clamp_min(const at::Tensor & self, const at::Scalar & min){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("clamp_min exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::clamp_min(self, min);
    } else {
        return acl_op::clamp_min(self, min);
    }
}
at::Tensor clamp_min(const at::Tensor & self, const at::Tensor & min){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool min_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(min);

    ASCEND_LOGI("clamp_min exec with jit compile: %d, self is internal format: %d, min is internal format: %d",
                !is_jit_disable, !self_base_format, !min_base_format);
    if (is_jit_disable && self_base_format && min_base_format) {
        return op_api::clamp_min(self, min);
    } else {
        return acl_op::clamp_min(self, min);
    }
}
at::Tensor col2im(const at::Tensor & self, at::IntArrayRef output_size, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("col2im exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::col2im(self, output_size, kernel_size, dilation, padding, stride);
    } else {
        return acl_op::col2im(self, output_size, kernel_size, dilation, padding, stride);
    }
}
at::Tensor complex(const at::Tensor & real, const at::Tensor & imag){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool real_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(real);
    bool imag_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(imag);

    ASCEND_LOGI("complex exec with jit compile: %d, real is internal format: %d, imag is internal format: %d",
                !is_jit_disable, !real_base_format, !imag_base_format);
    if (is_jit_disable && real_base_format && imag_base_format) {
        return op_api::complex(real, imag);
    } else {
        return acl_op::complex(real, imag);
    }
}
at::Tensor constant_pad_nd(const at::Tensor & self, at::IntArrayRef pad, const at::Scalar & value){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("constant_pad_nd exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::constant_pad_nd(self, pad, value);
    } else {
        return acl_op::constant_pad_nd(self, pad, value);
    }
}
at::Tensor conv_tbc(const at::Tensor & self, const at::Tensor & weight, const at::Tensor & bias, int64_t pad){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias);

    ASCEND_LOGI("conv_tbc exec with jit compile: %d, self is internal format: %d, weight is internal format: %d, bias is internal format: %d",
                !is_jit_disable, !self_base_format, !weight_base_format, !bias_base_format);
    if (is_jit_disable && self_base_format && weight_base_format && bias_base_format) {
        return op_api::conv_tbc(self, weight, bias, pad);
    } else {
        return acl_op::conv_tbc(self, weight, bias, pad);
    }
}
at::Tensor conv_transpose2d(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, int64_t groups, at::IntArrayRef dilation){
    return acl_op::conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, dilation);
}
at::Tensor conv_transpose3d(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, int64_t groups, at::IntArrayRef dilation){
    return acl_op::conv_transpose3d(input, weight, bias, stride, padding, output_padding, groups, dilation);
}
at::Tensor convolution(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias);

    ASCEND_LOGI("convolution exec with jit compile: %d, input is internal format: %d, weight is internal format: %d, bias is internal format: %d",
                !is_jit_disable, !input_base_format, !weight_base_format, !bias_base_format);
    if (is_jit_disable && input_base_format && weight_base_format && bias_base_format) {
        return op_api::convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
    } else {
        return acl_op::convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
    }
}
at::Tensor convolution_overrideable(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias);

    ASCEND_LOGI("convolution_overrideable exec with jit compile: %d, input is internal format: %d, weight is internal format: %d, bias is internal format: %d",
                !is_jit_disable, !input_base_format, !weight_base_format, !bias_base_format);
    if (is_jit_disable && input_base_format && weight_base_format && bias_base_format) {
        return op_api::convolution_overrideable(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
    } else {
        return acl_op::convolution_overrideable(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
    }
}
at::Tensor cos(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("cos exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::cos(self);
    } else {
        return acl_op::cos(self);
    }
}
at::Tensor cosh(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("cosh exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::cosh(self);
    } else {
        return acl_op::cosh(self);
    }
}
at::Tensor count_nonzero(const at::Tensor & self, at::IntArrayRef dim){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("count_nonzero exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::count_nonzero(self, dim);
    } else {
        return acl_op::count_nonzero(self, dim);
    }
}
at::Tensor count_nonzero(const at::Tensor & self, c10::optional<int64_t> dim){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("count_nonzero exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::count_nonzero(self, dim);
    } else {
        return acl_op::count_nonzero(self, dim);
    }
}
at::Tensor ctc_loss(const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, int64_t blank, int64_t reduction, bool zero_infinity){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool log_probs_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(log_probs);
    bool targets_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(targets);

    ASCEND_LOGI("ctc_loss exec with jit compile: %d, log_probs is internal format: %d, targets is internal format: %d",
                !is_jit_disable, !log_probs_base_format, !targets_base_format);
    if (is_jit_disable && log_probs_base_format && targets_base_format) {
        return op_api::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
    } else {
        return acl_op::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
    }
}
at::Tensor ctc_loss(const at::Tensor & log_probs, const at::Tensor & targets, const at::Tensor & input_lengths, const at::Tensor & target_lengths, int64_t blank, int64_t reduction, bool zero_infinity){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool log_probs_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(log_probs);
    bool targets_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(targets);
    bool input_lengths_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input_lengths);
    bool target_lengths_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target_lengths);

    ASCEND_LOGI("ctc_loss exec with jit compile: %d, log_probs is internal format: %d, targets is internal format: %d, input_lengths is internal format: %d, target_lengths is internal format: %d",
                !is_jit_disable, !log_probs_base_format, !targets_base_format, !input_lengths_base_format, !target_lengths_base_format);
    if (is_jit_disable && log_probs_base_format && targets_base_format && input_lengths_base_format && target_lengths_base_format) {
        return op_api::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
    } else {
        return acl_op::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
    }
}
at::Tensor cumsum(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("cumsum exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::cumsum(self, dim, dtype);
    } else {
        return acl_op::cumsum(self, dim, dtype);
    }
}
at::Tensor dequantize(const at::Tensor & self){
    return acl_op::dequantize(self);
}
at::Tensor div(const at::Tensor & self, const at::Scalar & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("div exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::div(self, other);
    } else {
        return acl_op::div(self, other);
    }
}
at::Tensor div(const at::Tensor & self, const at::Scalar & other, c10::optional<c10::string_view> rounding_mode){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("div exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::div(self, other, rounding_mode);
    } else {
        return acl_op::div(self, other, rounding_mode);
    }
}
at::Tensor div(const at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("div exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::div(self, other);
    } else {
        return acl_op::div(self, other);
    }
}
at::Tensor div(const at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("div exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::div(self, other, rounding_mode);
    } else {
        return acl_op::div(self, other, rounding_mode);
    }
}
at::Tensor dot(const at::Tensor & self, const at::Tensor & tensor){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool tensor_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor);

    ASCEND_LOGI("dot exec with jit compile: %d, self is internal format: %d, tensor is internal format: %d",
                !is_jit_disable, !self_base_format, !tensor_base_format);
    if (is_jit_disable && self_base_format && tensor_base_format) {
        return op_api::dot(self, tensor);
    } else {
        return acl_op::dot(self, tensor);
    }
}
at::Tensor dropout(const at::Tensor & input, double p, bool train){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    ASCEND_LOGI("dropout exec with jit compile: %d", !is_jit_disable);
    if (is_jit_disable) {
        return op_api::dropout(input, p, train);
    } else {
        return acl_op::dropout(input, p, train);
    }
}
at::Tensor dropout_with_byte_mask(const at::Tensor & self, double p, bool train){
    return acl_op::dropout_with_byte_mask(self, p, train);
}
at::Tensor elu(const at::Tensor & self, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("elu exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::elu(self, alpha, scale, input_scale);
    } else {
        return acl_op::elu(self, alpha, scale, input_scale);
    }
}
at::Tensor elu_backward(const at::Tensor & grad_output, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale, bool is_result, const at::Tensor & self_or_result){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_or_result_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self_or_result);

    ASCEND_LOGI("elu_backward exec with jit compile: %d, grad_output is internal format: %d, self_or_result is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_or_result_base_format);
    if (is_jit_disable && grad_output_base_format && self_or_result_base_format) {
        return op_api::elu_backward(grad_output, alpha, scale, input_scale, is_result, self_or_result);
    } else {
        return acl_op::elu_backward(grad_output, alpha, scale, input_scale, is_result, self_or_result);
    }
}
at::Tensor embedding_dense_backward(const at::Tensor & grad_output, const at::Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);

    ASCEND_LOGI("embedding_dense_backward exec with jit compile: %d, grad_output is internal format: %d, indices is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !indices_base_format);
    if (is_jit_disable && grad_output_base_format && indices_base_format) {
        return op_api::embedding_dense_backward(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq);
    } else {
        return acl_op::embedding_dense_backward(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq);
    }
}
at::Tensor embedding_symint(const at::Tensor & weight, const at::Tensor & indices, c10::SymInt padding_idx, bool scale_grad_by_freq, bool sparse){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);

    ASCEND_LOGI("embedding_symint exec with jit compile: %d, weight is internal format: %d, indices is internal format: %d",
                !is_jit_disable, !weight_base_format, !indices_base_format);
    if (is_jit_disable && weight_base_format && indices_base_format) {
        return op_api::embedding_symint(weight, indices, padding_idx, scale_grad_by_freq, sparse);
    } else {
        return acl_op::embedding_symint(weight, indices, padding_idx, scale_grad_by_freq, sparse);
    }
}
at::Tensor eq(const at::Tensor & self, const at::Scalar & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("eq exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::eq(self, other);
    } else {
        return acl_op::eq(self, other);
    }
}
at::Tensor eq(const at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("eq exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::eq(self, other);
    } else {
        return acl_op::eq(self, other);
    }
}
at::Tensor erf(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("erf exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::erf(self);
    } else {
        return acl_op::erf(self);
    }
}
at::Tensor erfc(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("erfc exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::erfc(self);
    } else {
        return acl_op::erfc(self);
    }
}
at::Tensor erfinv(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("erfinv exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::erfinv(self);
    } else {
        return acl_op::erfinv(self);
    }
}
at::Tensor exp(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("exp exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::exp(self);
    } else {
        return acl_op::exp(self);
    }
}
at::Tensor exp2(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("exp2 exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::exp2(self);
    } else {
        return acl_op::exp2(self);
    }
}
at::Tensor expm1(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("expm1 exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::expm1(self);
    } else {
        return acl_op::expm1(self);
    }
}
at::Tensor eye(int64_t n, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();

    ASCEND_LOGI("eye exec with jit compile: %d",
                !is_jit_disable);
    if (is_jit_disable) {
        return op_api::eye(n, dtype, layout, device, pin_memory);
    } else {
        return acl_op::eye(n, dtype, layout, device, pin_memory);
    }
}
at::Tensor eye(int64_t n, int64_t m, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();

    ASCEND_LOGI("eye exec with jit compile: %d",
                !is_jit_disable);
    if (is_jit_disable) {
        return op_api::eye(n, m, dtype, layout, device, pin_memory);
    } else {
        return acl_op::eye(n, m, dtype, layout, device, pin_memory);
    }
}
at::Tensor fast_gelu(const at::Tensor & self){
    return acl_op::fast_gelu(self);
}
at::Tensor fft_c2r_backward(const at::Tensor & grad, at::IntArrayRef dim, int64_t normalization){
    bool grad_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad);

    if (!grad_base_format) {
        TORCH_CHECK(false,
            "Current operator fft_c2r_backward do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::fft_c2r_backward(grad, dim, normalization);
}
at::Tensor fft_fftshift(const at::Tensor & self, at::OptionalIntArrayRef dim){
    return op_api::fft_fftshift(self, dim);
}
at::Tensor fft_ifftshift(const at::Tensor & self, at::OptionalIntArrayRef dim){
    return op_api::fft_ifftshift(self, dim);
}
at::Tensor fft_r2c_backward(const at::Tensor & grad, at::IntArrayRef dim, int64_t normalization, bool onesided, int64_t last_dim_size){
    bool grad_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad);

    if (!grad_base_format) {
        TORCH_CHECK(false,
            "Current operator fft_r2c_backward do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::fft_r2c_backward(grad, dim, normalization, onesided, last_dim_size);
}
at::Tensor flip(const at::Tensor & self, at::IntArrayRef dims){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("flip exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::flip(self, dims);
    } else {
        return acl_op::flip(self, dims);
    }
}
at::Tensor floor(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("floor exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::floor(self);
    } else {
        return acl_op::floor(self);
    }
}
at::Tensor floor_divide(const at::Tensor & self, const at::Scalar & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("floor_divide exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::floor_divide(self, other);
    } else {
        return acl_op::floor_divide(self, other);
    }
}
at::Tensor floor_divide(const at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("floor_divide exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::floor_divide(self, other);
    } else {
        return acl_op::floor_divide(self, other);
    }
}
at::Tensor fmod(const at::Tensor & self, const at::Scalar & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("fmod exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::fmod(self, other);
    } else {
        return acl_op::fmod(self, other);
    }
}
at::Tensor fmod(const at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("fmod exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::fmod(self, other);
    } else {
        return acl_op::fmod(self, other);
    }
}
at::Tensor frac(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("frac exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::frac(self);
    } else {
        return acl_op::frac(self);
    }
}
at::Tensor gather(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, bool sparse_grad){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool index_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(index);

    ASCEND_LOGI("gather exec with jit compile: %d, self is internal format: %d, index is internal format: %d",
                !is_jit_disable, !self_base_format, !index_base_format);
    if (is_jit_disable && self_base_format && index_base_format) {
        return op_api::gather(self, dim, index, sparse_grad);
    } else {
        return acl_op::gather(self, dim, index, sparse_grad);
    }
}
at::Tensor gather(const at::Tensor & self, int64_t dim, const at::Tensor & index, bool sparse_grad){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool index_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(index);

    ASCEND_LOGI("gather exec with jit compile: %d, self is internal format: %d, index is internal format: %d",
                !is_jit_disable, !self_base_format, !index_base_format);
    if (is_jit_disable && self_base_format && index_base_format) {
        return op_api::gather(self, dim, index, sparse_grad);
    } else {
        return acl_op::gather(self, dim, index, sparse_grad);
    }
}
at::Tensor ge(const at::Tensor & self, const at::Scalar & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("ge exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::ge(self, other);
    } else {
        return acl_op::ge(self, other);
    }
}
at::Tensor ge(const at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("ge exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::ge(self, other);
    } else {
        return acl_op::ge(self, other);
    }
}
at::Tensor gelu(const at::Tensor & self, c10::string_view approximate){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("gelu exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::gelu(self, approximate);
    } else {
        return acl_op::gelu(self, approximate);
    }
}
at::Tensor gelu_backward(const at::Tensor & grad_output, const at::Tensor & self, c10::string_view approximate){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("gelu_backward exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format) {
        return op_api::gelu_backward(grad_output, self, approximate);
    } else {
        return acl_op::gelu_backward(grad_output, self, approximate);
    }
}
at::Tensor glu(const at::Tensor & self, int64_t dim){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("glu exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::glu(self, dim);
    } else {
        return acl_op::glu(self, dim);
    }
}
at::Tensor glu_backward(const at::Tensor & grad_output, const at::Tensor & self, int64_t dim){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("glu_backward exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format) {
        return op_api::glu_backward(grad_output, self, dim);
    } else {
        return acl_op::glu_backward(grad_output, self, dim);
    }
}
at::Tensor grid_sampler_2d(const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool grid_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grid);

    ASCEND_LOGI("grid_sampler_2d exec with jit compile: %d, input is internal format: %d, grid is internal format: %d",
                !is_jit_disable, !input_base_format, !grid_base_format);
    if (is_jit_disable && input_base_format && grid_base_format) {
        return op_api::grid_sampler_2d(input, grid, interpolation_mode, padding_mode, align_corners);
    } else {
        return acl_op::grid_sampler_2d(input, grid, interpolation_mode, padding_mode, align_corners);
    }
}
at::Tensor grid_sampler_3d(const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool grid_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grid);

    ASCEND_LOGI("grid_sampler_3d exec with jit compile: %d, input is internal format: %d, grid is internal format: %d",
                !is_jit_disable, !input_base_format, !grid_base_format);
    if (is_jit_disable && input_base_format && grid_base_format) {
        return op_api::grid_sampler_3d(input, grid, interpolation_mode, padding_mode, align_corners);
    } else {
        return acl_op::grid_sampler_3d(input, grid, interpolation_mode, padding_mode, align_corners);
    }
}
at::Tensor gt(const at::Tensor & self, const at::Scalar & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("gt exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::gt(self, other);
    } else {
        return acl_op::gt(self, other);
    }
}
at::Tensor gt(const at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("gt exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::gt(self, other);
    } else {
        return acl_op::gt(self, other);
    }
}
at::Tensor hardshrink(const at::Tensor & self, const at::Scalar & lambd){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("hardshrink exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::hardshrink(self, lambd);
    } else {
        return acl_op::hardshrink(self, lambd);
    }
}
at::Tensor hardshrink_backward(const at::Tensor & grad_out, const at::Tensor & self, const at::Scalar & lambd){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_out);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("hardshrink_backward exec with jit compile: %d, grad_out is internal format: %d, self is internal format: %d",
                !is_jit_disable, !grad_out_base_format, !self_base_format);
    if (is_jit_disable && grad_out_base_format && self_base_format) {
        return op_api::hardshrink_backward(grad_out, self, lambd);
    } else {
        return acl_op::hardshrink_backward(grad_out, self, lambd);
    }
}
at::Tensor hardsigmoid(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("hardsigmoid exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::hardsigmoid(self);
    } else {
        return acl_op::hardsigmoid(self);
    }
}
at::Tensor hardsigmoid_backward(const at::Tensor & grad_output, const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("hardsigmoid_backward exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format) {
        return op_api::hardsigmoid_backward(grad_output, self);
    } else {
        return acl_op::hardsigmoid_backward(grad_output, self);
    }
}
at::Tensor hardswish(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("hardswish exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::hardswish(self);
    } else {
        return acl_op::hardswish(self);
    }
}
at::Tensor hardswish_backward(const at::Tensor & grad_output, const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("hardswish_backward exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format) {
        return op_api::hardswish_backward(grad_output, self);
    } else {
        return acl_op::hardswish_backward(grad_output, self);
    }
}
at::Tensor hardtanh(const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("hardtanh exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::hardtanh(self, min_val, max_val);
    } else {
        return acl_op::hardtanh(self, min_val, max_val);
    }
}
at::Tensor hardtanh_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("hardtanh_backward exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format) {
        return op_api::hardtanh_backward(grad_output, self, min_val, max_val);
    } else {
        return acl_op::hardtanh_backward(grad_output, self, min_val, max_val);
    }
}
at::Tensor histc(const at::Tensor & self, int64_t bins, const at::Scalar & min, const at::Scalar & max){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("histc exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::histc(self, bins, min, max);
    } else {
        return acl_op::histc(self, bins, min, max);
    }
}
at::Tensor im2col(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("im2col exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::im2col(self, kernel_size, dilation, padding, stride);
    } else {
        return acl_op::im2col(self, kernel_size, dilation, padding, stride);
    }
}
at::Tensor index(const at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);

    ASCEND_LOGI("index exec with jit compile: %d, self is internal format: %d, indices is internal format: %d",
                !is_jit_disable, !self_base_format, !indices_base_format);
    if (is_jit_disable && self_base_format && indices_base_format) {
        return op_api::index(self, indices);
    } else {
        return acl_op::index(self, indices);
    }
}
at::Tensor index_add(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha){
    return acl_op::index_add(self, dim, index, source, alpha);
}
at::Tensor index_add(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool index_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(index);
    bool source_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(source);

    ASCEND_LOGI("index_add exec with jit compile: %d, self is internal format: %d, index is internal format: %d, source is internal format: %d",
                !is_jit_disable, !self_base_format, !index_base_format, !source_base_format);
    if (is_jit_disable && self_base_format && index_base_format && source_base_format) {
        return op_api::index_add(self, dim, index, source, alpha);
    } else {
        return acl_op::index_add(self, dim, index, source, alpha);
    }
}
at::Tensor index_copy(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool index_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(index);
    bool source_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(source);

    ASCEND_LOGI("index_copy exec with jit compile: %d, self is internal format: %d, index is internal format: %d, source is internal format: %d",
                !is_jit_disable, !self_base_format, !index_base_format, !source_base_format);
    if (is_jit_disable && self_base_format && index_base_format && source_base_format) {
        return op_api::index_copy(self, dim, index, source);
    } else {
        return acl_op::index_copy(self, dim, index, source);
    }
}
at::Tensor index_fill(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool index_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(index);

    ASCEND_LOGI("index_fill exec with jit compile: %d, self is internal format: %d, index is internal format: %d",
                !is_jit_disable, !self_base_format, !index_base_format);
    if (is_jit_disable && self_base_format && index_base_format) {
        return op_api::index_fill(self, dim, index, value);
    } else {
        return acl_op::index_fill(self, dim, index, value);
    }
}
at::Tensor index_fill(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & value){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool index_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(index);
    bool value_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(value);

    ASCEND_LOGI("index_fill exec with jit compile: %d, self is internal format: %d, index is internal format: %d, value is internal format: %d",
                !is_jit_disable, !self_base_format, !index_base_format, !value_base_format);
    if (is_jit_disable && self_base_format && index_base_format && value_base_format) {
        return op_api::index_fill(self, dim, index, value);
    } else {
        return acl_op::index_fill(self, dim, index, value);
    }
}
at::Tensor index_put(const at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);
    bool values_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(values);

    ASCEND_LOGI("index_put exec with jit compile: %d, self is internal format: %d, indices is internal format: %d, values is internal format: %d",
                !is_jit_disable, !self_base_format, !indices_base_format, !values_base_format);
    if (is_jit_disable && self_base_format && indices_base_format && values_base_format) {
        return op_api::index_put(self, indices, values, accumulate);
    } else {
        return acl_op::index_put(self, indices, values, accumulate);
    }
}
at::Tensor index_select(const at::Tensor & self, at::Dimname dim, const at::Tensor & index){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool index_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(index);

    ASCEND_LOGI("index_select exec with jit compile: %d, self is internal format: %d, index is internal format: %d",
                !is_jit_disable, !self_base_format, !index_base_format);
    if (is_jit_disable && self_base_format && index_base_format) {
        return op_api::index_select(self, dim, index);
    } else {
        return acl_op::index_select(self, dim, index);
    }
}
at::Tensor index_select(const at::Tensor & self, int64_t dim, const at::Tensor & index){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool index_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(index);

    ASCEND_LOGI("index_select exec with jit compile: %d, self is internal format: %d, index is internal format: %d",
                !is_jit_disable, !self_base_format, !index_base_format);
    if (is_jit_disable && self_base_format && index_base_format) {
        return op_api::index_select(self, dim, index);
    } else {
        return acl_op::index_select(self, dim, index);
    }
}
at::Tensor int_repr(const at::Tensor & self){
    return acl_op::int_repr(self);
}
at::Tensor inverse(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("inverse exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::inverse(self);
    } else {
        return acl_op::inverse(self);
    }
}
at::Tensor isclose(const at::Tensor & self, const at::Tensor & other, double rtol, double atol, bool equal_nan){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("isclose exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::isclose(self, other, rtol, atol, equal_nan);
    } else {
        return acl_op::isclose(self, other, rtol, atol, equal_nan);
    }
}
at::Tensor isfinite(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("isfinite exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::isfinite(self);
    } else {
        return acl_op::isfinite(self);
    }
}
at::Tensor isin(const at::Tensor & element, const at::Scalar & test_elements, bool assume_unique, bool invert){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool element_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(element);

    ASCEND_LOGI("isin exec with jit compile: %d, element is internal format: %d",
                !is_jit_disable, !element_base_format);
    if (is_jit_disable && element_base_format) {
        return op_api::isin(element, test_elements, assume_unique, invert);
    } else {
        return acl_op::isin(element, test_elements, assume_unique, invert);
    }
}
at::Tensor isin(const at::Tensor & elements, const at::Tensor & test_elements, bool assume_unique, bool invert){
    bool elements_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(elements);
    bool test_elements_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(test_elements);

    if (!elements_base_format || !test_elements_base_format) {
        TORCH_CHECK(false,
            "Current operator isin do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::isin(elements, test_elements, assume_unique, invert);
}
at::Tensor kl_div(const at::Tensor & self, const at::Tensor & target, int64_t reduction, bool log_target){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target);

    ASCEND_LOGI("kl_div exec with jit compile: %d, self is internal format: %d, target is internal format: %d",
                !is_jit_disable, !self_base_format, !target_base_format);
    if (is_jit_disable && self_base_format && target_base_format) {
        return op_api::kl_div(self, target, reduction, log_target);
    } else {
        return acl_op::kl_div(self, target, reduction, log_target);
    }
}
at::Tensor kl_div_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, bool log_target){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target);

    ASCEND_LOGI("kl_div_backward exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, target is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !target_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && target_base_format) {
        return op_api::kl_div_backward(grad_output, self, target, reduction, log_target);
    } else {
        return acl_op::kl_div_backward(grad_output, self, target, reduction, log_target);
    }
}
at::Tensor l1_loss(const at::Tensor & self, const at::Tensor & target, int64_t reduction){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target);

    ASCEND_LOGI("l1_loss exec with jit compile: %d, self is internal format: %d, target is internal format: %d",
                !is_jit_disable, !self_base_format, !target_base_format);
    if (is_jit_disable && self_base_format && target_base_format) {
        return op_api::l1_loss(self, target, reduction);
    } else {
        return acl_op::l1_loss(self, target, reduction);
    }
}
at::Tensor l1_loss_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target);

    ASCEND_LOGI("l1_loss_backward exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, target is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !target_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && target_base_format) {
        return op_api::l1_loss_backward(grad_output, self, target, reduction);
    } else {
        return acl_op::l1_loss_backward(grad_output, self, target, reduction);
    }
}
at::Tensor le(const at::Tensor & self, const at::Scalar & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("le exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::le(self, other);
    } else {
        return acl_op::le(self, other);
    }
}
at::Tensor le(const at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("le exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::le(self, other);
    } else {
        return acl_op::le(self, other);
    }
}
at::Tensor leaky_relu(const at::Tensor & self, const at::Scalar & negative_slope){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("leaky_relu exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::leaky_relu(self, negative_slope);
    } else {
        return acl_op::leaky_relu(self, negative_slope);
    }
}
at::Tensor leaky_relu_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & negative_slope, bool self_is_result){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("leaky_relu_backward exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format) {
        return op_api::leaky_relu_backward(grad_output, self, negative_slope, self_is_result);
    } else {
        return acl_op::leaky_relu_backward(grad_output, self, negative_slope, self_is_result);
    }
}
at::Tensor lerp(const at::Tensor & self, const at::Tensor & end, const at::Scalar & weight){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool end_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(end);

    ASCEND_LOGI("lerp exec with jit compile: %d, self is internal format: %d, end is internal format: %d",
                !is_jit_disable, !self_base_format, !end_base_format);
    if (is_jit_disable && self_base_format && end_base_format) {
        return op_api::lerp(self, end, weight);
    } else {
        return acl_op::lerp(self, end, weight);
    }
}
at::Tensor lerp(const at::Tensor & self, const at::Tensor & end, const at::Tensor & weight){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool end_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(end);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);

    ASCEND_LOGI("lerp exec with jit compile: %d, self is internal format: %d, end is internal format: %d, weight is internal format: %d",
                !is_jit_disable, !self_base_format, !end_base_format, !weight_base_format);
    if (is_jit_disable && self_base_format && end_base_format && weight_base_format) {
        return op_api::lerp(self, end, weight);
    } else {
        return acl_op::lerp(self, end, weight);
    }
}
at::Tensor linalg_cross(const at::Tensor & self, const at::Tensor & other, int64_t dim){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("linalg_cross exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::linalg_cross(self, other, dim);
    } else {
        return acl_op::linalg_cross(self, other, dim);
    }
}
at::Tensor linalg_solve_triangular(const at::Tensor & self, const at::Tensor & B, bool upper, bool left, bool unitriangular){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool B_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(B);

    if (!self_base_format || !B_base_format) {
        TORCH_CHECK(false,
            "Current operator linalg_solve_triangular do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::linalg_solve_triangular(self, B, upper, left, unitriangular);
}
at::Tensor linalg_vector_norm(const at::Tensor & self, const at::Scalar & ord, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("linalg_vector_norm exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::linalg_vector_norm(self, ord, dim, keepdim, dtype);
    } else {
        return acl_op::linalg_vector_norm(self, ord, dim, keepdim, dtype);
    }
}
at::Tensor linspace(const at::Scalar & start, const at::Scalar & end, int64_t steps, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();

    ASCEND_LOGI("linspace exec with jit compile: %d",
                !is_jit_disable);
    if (is_jit_disable) {
        return op_api::linspace(start, end, steps, dtype, layout, device, pin_memory);
    } else {
        return acl_op::linspace(start, end, steps, dtype, layout, device, pin_memory);
    }
}
at::Tensor log(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("log exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::log(self);
    } else {
        return acl_op::log(self);
    }
}
at::Tensor log10(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("log10 exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::log10(self);
    } else {
        return acl_op::log10(self);
    }
}
at::Tensor log1p(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("log1p exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::log1p(self);
    } else {
        return acl_op::log1p(self);
    }
}
at::Tensor log2(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("log2 exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::log2(self);
    } else {
        return acl_op::log2(self);
    }
}
at::Tensor log_sigmoid(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("log_sigmoid exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::log_sigmoid(self);
    } else {
        return acl_op::log_sigmoid(self);
    }
}
at::Tensor log_sigmoid_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & buffer){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool buffer_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(buffer);

    ASCEND_LOGI("log_sigmoid_backward exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, buffer is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !buffer_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && buffer_base_format) {
        return op_api::log_sigmoid_backward(grad_output, self, buffer);
    } else {
        return acl_op::log_sigmoid_backward(grad_output, self, buffer);
    }
}
at::Tensor log_softmax(const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype){
    return acl_op::log_softmax(self, dim, dtype);
}
at::Tensor log_softmax(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype){
    return acl_op::log_softmax(self, dim, dtype);
}
at::Tensor logaddexp(const at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("logaddexp exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::logaddexp(self, other);
    } else {
        return acl_op::logaddexp(self, other);
    }
}
at::Tensor logaddexp2(const at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("logaddexp2 exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::logaddexp2(self, other);
    } else {
        return acl_op::logaddexp2(self, other);
    }
}
at::Tensor logical_and(const at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("logical_and exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::logical_and(self, other);
    } else {
        return acl_op::logical_and(self, other);
    }
}
at::Tensor logical_not(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("logical_not exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::logical_not(self);
    } else {
        return acl_op::logical_not(self);
    }
}
at::Tensor logical_or(const at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("logical_or exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::logical_or(self, other);
    } else {
        return acl_op::logical_or(self, other);
    }
}
at::Tensor logical_xor(const at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("logical_xor exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::logical_xor(self, other);
    } else {
        return acl_op::logical_xor(self, other);
    }
}
at::Tensor logit(const at::Tensor & self, c10::optional<double> eps){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator logit do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::logit(self, eps);
}
at::Tensor logit_backward(const at::Tensor & grad_output, const at::Tensor & self, c10::optional<double> eps){
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!grad_output_base_format || !self_base_format) {
        TORCH_CHECK(false,
            "Current operator logit_backward do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::logit_backward(grad_output, self, eps);
}
at::Tensor logspace(const at::Scalar & start, const at::Scalar & end, int64_t steps, double base, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory){
    return acl_op::logspace(start, end, steps, base, dtype, layout, device, pin_memory);
}
at::Tensor logsumexp(const at::Tensor & self, at::DimnameList dim, bool keepdim){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("logsumexp exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::logsumexp(self, dim, keepdim);
    } else {
        return acl_op::logsumexp(self, dim, keepdim);
    }
}
at::Tensor logsumexp(const at::Tensor & self, at::IntArrayRef dim, bool keepdim){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("logsumexp exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::logsumexp(self, dim, keepdim);
    } else {
        return acl_op::logsumexp(self, dim, keepdim);
    }
}
at::Tensor lt(const at::Tensor & self, const at::Scalar & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("lt exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::lt(self, other);
    } else {
        return acl_op::lt(self, other);
    }
}
at::Tensor lt(const at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("lt exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::lt(self, other);
    } else {
        return acl_op::lt(self, other);
    }
}
at::Tensor masked_select(const at::Tensor & self, const at::Tensor & mask){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mask);

    ASCEND_LOGI("masked_select exec with jit compile: %d, self is internal format: %d, mask is internal format: %d",
                !is_jit_disable, !self_base_format, !mask_base_format);
    if (is_jit_disable && self_base_format && mask_base_format) {
        return op_api::masked_select(self, mask);
    } else {
        return acl_op::masked_select(self, mask);
    }
}
at::Tensor matmul(const at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("matmul exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::matmul(self, other);
    } else {
        return acl_op::matmul(self, other);
    }
}
at::Tensor max(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("max exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::max(self);
    } else {
        return acl_op::max(self);
    }
}
at::Tensor max_pool2d_with_indices_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);

    ASCEND_LOGI("max_pool2d_with_indices_backward exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, indices is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !indices_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && indices_base_format) {
        return op_api::max_pool2d_with_indices_backward(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
    } else {
        return acl_op::max_pool2d_with_indices_backward(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
    }
}
at::Tensor max_pool3d_with_indices_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);

    ASCEND_LOGI("max_pool3d_with_indices_backward exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, indices is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !indices_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && indices_base_format) {
        return op_api::max_pool3d_with_indices_backward(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
    } else {
        return acl_op::max_pool3d_with_indices_backward(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
    }
}
at::Tensor max_sparse(const at::Tensor & self){
    return sparse::max_sparse(self);
}
at::Tensor max_unpool2d(const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);

    ASCEND_LOGI("max_unpool2d exec with jit compile: %d, self is internal format: %d, indices is internal format: %d",
                !is_jit_disable, !self_base_format, !indices_base_format);
    if (is_jit_disable && self_base_format && indices_base_format) {
        return op_api::max_unpool2d(self, indices, output_size);
    } else {
        return acl_op::max_unpool2d(self, indices, output_size);
    }
}
at::Tensor max_unpool3d(const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::IntArrayRef stride, at::IntArrayRef padding){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);

    ASCEND_LOGI("max_unpool3d exec with jit compile: %d, self is internal format: %d, indices is internal format: %d",
                !is_jit_disable, !self_base_format, !indices_base_format);
    if (is_jit_disable && self_base_format && indices_base_format) {
        return op_api::max_unpool3d(self, indices, output_size, stride, padding);
    } else {
        return acl_op::max_unpool3d(self, indices, output_size, stride, padding);
    }
}
at::Tensor maximum(const at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("maximum exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::maximum(self, other);
    } else {
        return acl_op::maximum(self, other);
    }
}
at::Tensor mean(const at::Tensor & self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> dtype){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("mean exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::mean(self, dim, keepdim, dtype);
    } else {
        return acl_op::mean(self, dim, keepdim, dtype);
    }
}
at::Tensor mean(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("mean exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::mean(self, dim, keepdim, dtype);
    } else {
        return acl_op::mean(self, dim, keepdim, dtype);
    }
}
at::Tensor mean(const at::Tensor & self, c10::optional<at::ScalarType> dtype){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("mean exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::mean(self, dtype);
    } else {
        return acl_op::mean(self, dtype);
    }
}
at::Tensor median(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("median exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::median(self);
    } else {
        return acl_op::median(self);
    }
}
at::Tensor min(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("min exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::min(self);
    } else {
        return acl_op::min(self);
    }
}
at::Tensor minimum(const at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("minimum exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::minimum(self, other);
    } else {
        return acl_op::minimum(self, other);
    }
}
at::Tensor mish(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("mish exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::mish(self);
    } else {
        return acl_op::mish(self);
    }
}
at::Tensor mish_backward(const at::Tensor & grad_output, const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("mish_backward exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format) {
        return op_api::mish_backward(grad_output, self);
    } else {
        return acl_op::mish_backward(grad_output, self);
    }
}
at::Tensor mm(const at::Tensor & self, const at::Tensor & mat2){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool mat2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mat2);

    ASCEND_LOGI("mm exec with jit compile: %d, self is internal format: %d, mat2 is internal format: %d",
                !is_jit_disable, !self_base_format, !mat2_base_format);
    if (is_jit_disable && self_base_format && mat2_base_format) {
        return op_api::mm(self, mat2);
    } else {
        return acl_op::mm(self, mat2);
    }
}
at::Tensor mse_loss(const at::Tensor & self, const at::Tensor & target, int64_t reduction){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target);

    ASCEND_LOGI("mse_loss exec with jit compile: %d, self is internal format: %d, target is internal format: %d",
                !is_jit_disable, !self_base_format, !target_base_format);
    if (is_jit_disable && self_base_format && target_base_format) {
        return op_api::mse_loss(self, target, reduction);
    } else {
        return acl_op::mse_loss(self, target, reduction);
    }
}
at::Tensor mse_loss_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target);

    ASCEND_LOGI("mse_loss_backward exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, target is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !target_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && target_base_format) {
        return op_api::mse_loss_backward(grad_output, self, target, reduction);
    } else {
        return acl_op::mse_loss_backward(grad_output, self, target, reduction);
    }
}
at::Tensor mul(const at::Tensor & self, const at::Scalar & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("mul exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::mul(self, other);
    } else {
        return acl_op::mul(self, other);
    }
}
at::Tensor mul(const at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("mul exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::mul(self, other);
    } else {
        return acl_op::mul(self, other);
    }
}
at::Tensor multilabel_margin_loss(const at::Tensor & self, const at::Tensor & target, int64_t reduction){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target);

    ASCEND_LOGI("multilabel_margin_loss exec with jit compile: %d, self is internal format: %d, target is internal format: %d",
                !is_jit_disable, !self_base_format, !target_base_format);
    if (is_jit_disable && self_base_format && target_base_format) {
        return op_api::multilabel_margin_loss(self, target, reduction);
    } else {
        return acl_op::multilabel_margin_loss(self, target, reduction);
    }
}
at::Tensor multinomial(const at::Tensor & self, int64_t num_samples, bool replacement, c10::optional<at::Generator> generator){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("multinomial exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::multinomial(self, num_samples, replacement, generator);
    } else {
        return acl_op::multinomial(self, num_samples, replacement, generator);
    }
}
at::Tensor mv(const at::Tensor & self, const at::Tensor & vec){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool vec_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(vec);

    ASCEND_LOGI("mv exec with jit compile: %d, self is internal format: %d, vec is internal format: %d",
                !is_jit_disable, !self_base_format, !vec_base_format);
    if (is_jit_disable && self_base_format && vec_base_format) {
        return op_api::mv(self, vec);
    } else {
        return acl_op::mv(self, vec);
    }
}
at::Tensor nan_to_num(const at::Tensor & self, c10::optional<double> nan, c10::optional<double> posinf, c10::optional<double> neginf){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("nan_to_num exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::nan_to_num(self, nan, posinf, neginf);
    } else {
        return acl_op::nan_to_num(self, nan, posinf, neginf);
    }
}
at::Tensor nanmedian(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("nanmedian exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::nanmedian(self);
    } else {
        return acl_op::nanmedian(self);
    }
}
at::Tensor nansum(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator nansum do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::nansum(self, dim, keepdim, dtype);
}
at::Tensor native_dropout_backward(const at::Tensor & grad_output, const at::Tensor & mask, double scale){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mask);

    ASCEND_LOGI("native_dropout_backward exec with jit compile: %d, grad_output is internal format: %d, mask is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !mask_base_format);
    if (is_jit_disable && grad_output_base_format && mask_base_format) {
        return op_api::native_dropout_backward(grad_output, mask, scale);
    } else {
        return acl_op::native_dropout_backward(grad_output, mask, scale);
    }
}
at::Tensor ne(const at::Tensor & self, const at::Scalar & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("ne exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::ne(self, other);
    } else {
        return acl_op::ne(self, other);
    }
}
at::Tensor ne(const at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("ne exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::ne(self, other);
    } else {
        return acl_op::ne(self, other);
    }
}
at::Tensor neg(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("neg exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::neg(self);
    } else {
        return acl_op::neg(self);
    }
}
at::Tensor nll_loss(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index){
    return acl_op::nll_loss(self, target, weight, reduction, ignore_index);
}
at::Tensor nll_loss2d(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index){
    return acl_op::nll_loss2d(self, target, weight, reduction, ignore_index);
}
at::Tensor nll_loss2d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool total_weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(total_weight);

    ASCEND_LOGI("nll_loss2d_backward exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, target is internal format: %d, weight is internal format: %d, total_weight is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !target_base_format, !weight_base_format, !total_weight_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && target_base_format && weight_base_format && total_weight_base_format) {
        return op_api::nll_loss2d_backward(grad_output, self, target, weight, reduction, ignore_index, total_weight);
    } else {
        return acl_op::nll_loss2d_backward(grad_output, self, target, weight, reduction, ignore_index, total_weight);
    }
}
at::Tensor nll_loss_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool total_weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(total_weight);

    ASCEND_LOGI("nll_loss_backward exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, target is internal format: %d, weight is internal format: %d, total_weight is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !target_base_format, !weight_base_format, !total_weight_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && target_base_format && weight_base_format && total_weight_base_format) {
        return op_api::nll_loss_backward(grad_output, self, target, weight, reduction, ignore_index, total_weight);
    } else {
        return acl_op::nll_loss_backward(grad_output, self, target, weight, reduction, ignore_index, total_weight);
    }
}
at::Tensor nonzero(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("nonzero exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::nonzero(self);
    } else {
        return acl_op::nonzero(self);
    }
}
at::Tensor norm(const at::Tensor & self, const at::Scalar & p){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("norm exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::norm(self, p);
    } else {
        return acl_op::norm(self, p);
    }
}
at::Tensor norm(const at::Tensor & self, const c10::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("norm exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::norm(self, p, dim, keepdim);
    } else {
        return acl_op::norm(self, p, dim, keepdim);
    }
}
at::Tensor norm(const at::Tensor & self, const c10::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::ScalarType dtype){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("norm exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::norm(self, p, dim, keepdim, dtype);
    } else {
        return acl_op::norm(self, p, dim, keepdim, dtype);
    }
}
at::Tensor norm(const at::Tensor & self, const c10::optional<at::Scalar> & p, at::ScalarType dtype){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("norm exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::norm(self, p, dtype);
    } else {
        return acl_op::norm(self, p, dtype);
    }
}
at::Tensor normal(const at::Tensor & mean, const at::Tensor & std, c10::optional<at::Generator> generator){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool mean_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mean);
    bool std_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(std);

    ASCEND_LOGI("normal exec with jit compile: %d, mean is internal format: %d, std is internal format: %d",
                !is_jit_disable, !mean_base_format, !std_base_format);
    if (is_jit_disable && mean_base_format && std_base_format) {
        return op_api::normal(mean, std, generator);
    } else {
        return acl_op::normal(mean, std, generator);
    }
}
at::Tensor normal(const at::Tensor & mean, double std, c10::optional<at::Generator> generator){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool mean_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mean);

    ASCEND_LOGI("normal exec with jit compile: %d, mean is internal format: %d",
                !is_jit_disable, !mean_base_format);
    if (is_jit_disable && mean_base_format) {
        return op_api::normal(mean, std, generator);
    } else {
        return acl_op::normal(mean, std, generator);
    }
}
at::Tensor normal(double mean, const at::Tensor & std, c10::optional<at::Generator> generator){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool std_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(std);

    ASCEND_LOGI("normal exec with jit compile: %d, std is internal format: %d",
                !is_jit_disable, !std_base_format);
    if (is_jit_disable && std_base_format) {
        return op_api::normal(mean, std, generator);
    } else {
        return acl_op::normal(mean, std, generator);
    }
}
at::Tensor normal(double mean, double std, at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();

    ASCEND_LOGI("normal exec with jit compile: %d",
                !is_jit_disable);
    if (is_jit_disable) {
        return op_api::normal(mean, std, size, generator, dtype, layout, device, pin_memory);
    } else {
        return acl_op::normal(mean, std, size, generator, dtype, layout, device, pin_memory);
    }
}
at::Tensor npu_alloc_float_status(const at::Tensor & self){
    return acl_op::npu_alloc_float_status(self);
}
at::Tensor npu_anchor_response_flags(const at::Tensor & self, at::IntArrayRef featmap_size, at::IntArrayRef stride, int64_t num_base_anchors){
    return acl_op::npu_anchor_response_flags(self, featmap_size, stride, num_base_anchors);
}
at::Tensor npu_anti_quant(const at::Tensor & x, const at::Tensor & scale, const c10::optional<at::Tensor> & offset, c10::optional<at::ScalarType> dst_dtype, c10::optional<at::ScalarType> src_dtype){
    bool x_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x);
    bool scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(scale);
    bool offset_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(offset);

    if (!x_base_format || !scale_base_format || !offset_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_anti_quant do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_anti_quant(x, scale, offset, dst_dtype, src_dtype);
}
at::Tensor npu_batch_gather_matmul(const at::Tensor & self, const at::Tensor & x, const at::Tensor & weight_b, const at::Tensor & indices, const c10::optional<at::Tensor> & weight_a, int64_t layer_idx, double scale, int64_t y_offset, int64_t y_slice_size){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool x_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x);
    bool weight_b_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight_b);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);
    bool weight_a_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight_a);

    if (!self_base_format || !x_base_format || !weight_b_base_format || !indices_base_format || !weight_a_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_batch_gather_matmul do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_batch_gather_matmul(self, x, weight_b, indices, weight_a, layer_idx, scale, y_offset, y_slice_size);
}
at::Tensor npu_binary_cross_entropy_with_logits_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight_opt, const c10::optional<at::Tensor> & pos_weight_opt, int64_t reduction){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target);
    bool weight_opt_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight_opt);
    bool pos_weight_opt_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(pos_weight_opt);

    ASCEND_LOGI("npu_binary_cross_entropy_with_logits_backward exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, target is internal format: %d, weight_opt is internal format: %d, pos_weight_opt is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !target_base_format, !weight_opt_base_format, !pos_weight_opt_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && target_base_format && weight_opt_base_format && pos_weight_opt_base_format) {
        return op_api::npu_binary_cross_entropy_with_logits_backward(grad_output, self, target, weight_opt, pos_weight_opt, reduction);
    } else {
        return acl_op::npu_binary_cross_entropy_with_logits_backward(grad_output, self, target, weight_opt, pos_weight_opt, reduction);
    }
}
at::Tensor npu_bmmV2(const at::Tensor & self, const at::Tensor & mat2, at::IntArrayRef output_sizes){
    return acl_op::npu_bmmV2(self, mat2, output_sizes);
}
at::Tensor npu_bmm_v2_mat1_backward_symint(const at::Tensor & grad, const at::Tensor & mat1, const at::Tensor & mat2, c10::SymIntArrayRef size){
    return acl_op::npu_bmm_v2_mat1_backward_symint(grad, mat1, mat2, size);
}
at::Tensor npu_bmm_v2_mat2_backward_symint(const at::Tensor & grad, const at::Tensor & mat1, const at::Tensor & mat2, c10::SymIntArrayRef size){
    return acl_op::npu_bmm_v2_mat2_backward_symint(grad, mat1, mat2, size);
}
at::Tensor npu_bounding_box_decode(const at::Tensor & rois, const at::Tensor & deltas, double means0, double means1, double means2, double means3, double stds0, double stds1, double stds2, double stds3, at::IntArrayRef max_shape, double wh_ratio_clip){
    return acl_op::npu_bounding_box_decode(rois, deltas, means0, means1, means2, means3, stds0, stds1, stds2, stds3, max_shape, wh_ratio_clip);
}
at::Tensor npu_bounding_box_encode(const at::Tensor & anchor_box, const at::Tensor & ground_truth_box, double means0, double means1, double means2, double means3, double stds0, double stds1, double stds2, double stds3){
    return acl_op::npu_bounding_box_encode(anchor_box, ground_truth_box, means0, means1, means2, means3, stds0, stds1, stds2, stds3);
}
at::Tensor npu_broadcast(const at::Tensor & self, at::IntArrayRef size){
    return acl_op::npu_broadcast(self, size);
}
at::Tensor npu_ciou(const at::Tensor & self, const at::Tensor & gtboxes, bool trans, bool is_cross, int64_t mode, bool atan_sub_flag){
    return acl_op::npu_ciou(self, gtboxes, trans, is_cross, mode, atan_sub_flag);
}
at::Tensor npu_clear_float_status(const at::Tensor & self, int64_t mode){
    return acl_op::npu_clear_float_status(self, mode);
}
at::Tensor npu_confusion_transpose(const at::Tensor & self, at::IntArrayRef perm, at::IntArrayRef shape, bool transpose_first){
    return acl_op::npu_confusion_transpose(self, perm, shape, transpose_first);
}
at::Tensor npu_confusion_transpose_backward_symint(const at::Tensor & grad, at::IntArrayRef perm, c10::SymIntArrayRef shape, bool transpose_first){
    return acl_op::npu_confusion_transpose_backward_symint(grad, perm, shape, transpose_first);
}
at::Tensor npu_conv2d(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups){
    return acl_op::npu_conv2d(input, weight, bias, stride, padding, dilation, groups);
}
at::Tensor npu_conv3d(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups){
    return acl_op::npu_conv3d(input, weight, bias, stride, padding, dilation, groups);
}
at::Tensor npu_conv_transpose2d(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups){
    return acl_op::npu_conv_transpose2d(input, weight, bias, padding, output_padding, stride, dilation, groups);
}
at::Tensor npu_convert_weight_to_int4pack(const at::Tensor & weight, int64_t inner_k_tiles){
    return op_api::npu_convert_weight_to_int4pack(weight, inner_k_tiles);
}
at::Tensor npu_convolution(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups){
    return acl_op::npu_convolution(input, weight, bias, stride, padding, dilation, groups);
}
at::Tensor npu_convolution_transpose(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups){
    return acl_op::npu_convolution_transpose(input, weight, bias, padding, output_padding, stride, dilation, groups);
}
at::Tensor npu_cross_entropy_loss_backward(const at::Tensor & grad_loss, const at::Tensor & log_prob, const at::Tensor & target, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & grad_zloss, const c10::optional<at::Tensor> & lse_for_zloss, c10::string_view reduction, int64_t ignore_index, double label_smoothing, double lse_square_scale_for_zloss){
    bool grad_loss_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_loss);
    bool log_prob_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(log_prob);
    bool target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool grad_zloss_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_zloss);
    bool lse_for_zloss_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(lse_for_zloss);

    if (!grad_loss_base_format || !log_prob_base_format || !target_base_format || !weight_base_format || !grad_zloss_base_format || !lse_for_zloss_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_cross_entropy_loss_backward do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_cross_entropy_loss_backward(grad_loss, log_prob, target, weight, grad_zloss, lse_for_zloss, reduction, ignore_index, label_smoothing, lse_square_scale_for_zloss);
}
at::Tensor npu_dequant_bias(const at::Tensor & x, const at::Tensor & weight_scale, const c10::optional<at::Tensor> & activation_scale, const c10::optional<at::Tensor> & bias, c10::optional<at::ScalarType> output_dtype){
    bool x_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x);
    bool weight_scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight_scale);
    bool activation_scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(activation_scale);
    bool bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias);

    if (!x_base_format || !weight_scale_base_format || !activation_scale_base_format || !bias_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_dequant_bias do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_dequant_bias(x, weight_scale, activation_scale, bias, output_dtype);
}
at::Tensor npu_diou(const at::Tensor & self, const at::Tensor & gtboxes, bool trans, bool is_cross, int64_t mode){
    return acl_op::npu_diou(self, gtboxes, trans, is_cross, mode);
}
at::Tensor npu_dropout_backward(const at::Tensor & grad_output, const at::Tensor & mask, double p){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mask);

    ASCEND_LOGI("npu_dropout_backward exec with jit compile: %d, grad_output is internal format: %d, mask is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !mask_base_format);
    if (is_jit_disable && grad_output_base_format && mask_base_format) {
        return op_api::npu_dropout_backward(grad_output, mask, p);
    } else {
        return acl_op::npu_dropout_backward(grad_output, mask, p);
    }
}
at::Tensor npu_dropout_gen_mask(at::IntArrayRef size, double p, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory){
    return acl_op::npu_dropout_gen_mask(size, p, dtype, layout, device, pin_memory);
}
at::Tensor npu_dtype_cast(const at::Tensor & self, at::ScalarType dtype){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("npu_dtype_cast exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::npu_dtype_cast(self, dtype);
    } else {
        return acl_op::npu_dtype_cast(self, dtype);
    }
}
at::Tensor npu_dtype_cast_backward(const at::Tensor & grad, at::ScalarType dtype){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad);

    ASCEND_LOGI("npu_dtype_cast_backward exec with jit compile: %d, grad is internal format: %d",
                !is_jit_disable, !grad_base_format);
    if (is_jit_disable && grad_base_format) {
        return op_api::npu_dtype_cast_backward(grad, dtype);
    } else {
        return acl_op::npu_dtype_cast_backward(grad, dtype);
    }
}
at::Tensor npu_fast_gelu(const at::Tensor & self){
    return acl_op::npu_fast_gelu(self);
}
at::Tensor npu_fast_gelu_backward(const at::Tensor & grad, const at::Tensor & self){
    return acl_op::npu_fast_gelu_backward(grad, self);
}
at::Tensor npu_ffn(const at::Tensor & x, const at::Tensor & weight1, const at::Tensor & weight2, c10::string_view activation, at::OptionalIntArrayRef expert_tokens, at::OptionalIntArrayRef expert_tokens_index, const c10::optional<at::Tensor> & bias1, const c10::optional<at::Tensor> & bias2, const c10::optional<at::Tensor> & scale, const c10::optional<at::Tensor> & offset, const c10::optional<at::Tensor> & deq_scale1, const c10::optional<at::Tensor> & deq_scale2, const c10::optional<at::Tensor> & antiquant_scale1, const c10::optional<at::Tensor> & antiquant_scale2, const c10::optional<at::Tensor> & antiquant_offset1, const c10::optional<at::Tensor> & antiquant_offset2, c10::optional<int64_t> inner_precise, c10::optional<at::ScalarType> output_dtype){
    bool x_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x);
    bool weight1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight1);
    bool weight2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight2);
    bool bias1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias1);
    bool bias2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias2);
    bool scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(scale);
    bool offset_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(offset);
    bool deq_scale1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(deq_scale1);
    bool deq_scale2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(deq_scale2);
    bool antiquant_scale1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(antiquant_scale1);
    bool antiquant_scale2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(antiquant_scale2);
    bool antiquant_offset1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(antiquant_offset1);
    bool antiquant_offset2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(antiquant_offset2);

    if (!x_base_format || !weight1_base_format || !weight2_base_format || !bias1_base_format || !bias2_base_format || !scale_base_format || !offset_base_format || !deq_scale1_base_format || !deq_scale2_base_format || !antiquant_scale1_base_format || !antiquant_scale2_base_format || !antiquant_offset1_base_format || !antiquant_offset2_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_ffn do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_ffn(x, weight1, weight2, activation, expert_tokens, expert_tokens_index, bias1, bias2, scale, offset, deq_scale1, deq_scale2, antiquant_scale1, antiquant_scale2, antiquant_offset1, antiquant_offset2, inner_precise, output_dtype);
}
at::Tensor npu_fused_attention_score(const at::Tensor & query_layer, const at::Tensor & key_layer, const at::Tensor & value_layer, const at::Tensor & attention_mask, const at::Scalar & scale, double keep_prob, bool query_transpose, bool key_transpose, bool bmm_score_transpose_a, bool bmm_score_transpose_b, bool value_transpose, bool dx_transpose){
    return acl_op::npu_fused_attention_score(query_layer, key_layer, value_layer, attention_mask, scale, keep_prob, query_transpose, key_transpose, bmm_score_transpose_a, bmm_score_transpose_b, value_transpose, dx_transpose);
}
at::Tensor npu_gather_backward_symint(const at::Tensor & grad, c10::SymIntArrayRef self_size, int64_t dim, const at::Tensor & index, bool sparse_grad){
    bool grad_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad);
    bool index_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(index);

    if (!grad_base_format || !index_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_gather_backward_symint do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_gather_backward_symint(grad, self_size, dim, index, sparse_grad);
}
at::Tensor npu_gather_sparse_index(const at::Tensor & input, const at::Tensor & index){
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool index_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(index);

    if (!input_base_format || !index_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_gather_sparse_index do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_gather_sparse_index(input, index);
}
at::Tensor npu_geglu_grad(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & gelu, int64_t dim, int64_t approximate, bool activate_left){
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool gelu_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(gelu);

    if (!grad_output_base_format || !self_base_format || !gelu_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_geglu_grad do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_geglu_grad(grad_output, self, gelu, dim, approximate, activate_left);
}
at::Tensor npu_gelu(const at::Tensor & self, c10::string_view approximate){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_gelu do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_gelu(self, approximate);
}
at::Tensor npu_gelu_backward(const at::Tensor & grad_output, const at::Tensor & self, c10::string_view approximate){
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!grad_output_base_format || !self_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_gelu_backward do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_gelu_backward(grad_output, self, approximate);
}
at::Tensor npu_get_float_status(const at::Tensor & self, int64_t mode){
    return acl_op::npu_get_float_status(self, mode);
}
at::Tensor npu_giou(const at::Tensor & self, const at::Tensor & gtboxes, bool trans, bool is_cross, int64_t mode){
    return acl_op::npu_giou(self, gtboxes, trans, is_cross, mode);
}
at::Tensor npu_grid_assign_positive(const at::Tensor & self, const at::Tensor & overlaps, const at::Tensor & box_responsible_flags, const at::Tensor & max_overlaps, const at::Tensor & argmax_overlaps, const at::Tensor & gt_max_overlaps, const at::Tensor & gt_argmax_overlaps, int64_t num_gts, double pos_iou_thr, double min_pos_iou, bool gt_max_assign_all){
    return acl_op::npu_grid_assign_positive(self, overlaps, box_responsible_flags, max_overlaps, argmax_overlaps, gt_max_overlaps, gt_argmax_overlaps, num_gts, pos_iou_thr, min_pos_iou, gt_max_assign_all);
}
at::Tensor npu_group_quant(const at::Tensor & x, const at::Tensor & scale, const at::Tensor & group_index, const c10::optional<at::Tensor> & offset, c10::optional<at::ScalarType> dst_dtype){
    bool x_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x);
    bool scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(scale);
    bool group_index_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(group_index);
    bool offset_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(offset);

    if (!x_base_format || !scale_base_format || !group_index_base_format || !offset_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_group_quant do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_group_quant(x, scale, group_index, offset, dst_dtype);
}
at::Tensor npu_grouped_matmul_finalize_routing(const at::Tensor & x, const at::Tensor & w, const at::Tensor & group_list, const c10::optional<at::Tensor> & scale, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & offset, const c10::optional<at::Tensor> & pertoken_scale, const c10::optional<at::Tensor> & shared_input, const c10::optional<at::Tensor> & logit, const c10::optional<at::Tensor> & row_index, c10::optional<at::ScalarType> dtype, c10::optional<double> shared_input_weight, c10::optional<int64_t> shared_input_offset, c10::optional<int64_t> output_bs, c10::optional<int64_t> group_list_type){
    return op_api::npu_grouped_matmul_finalize_routing(x, w, group_list, scale, bias, offset, pertoken_scale, shared_input, logit, row_index, dtype, shared_input_weight, shared_input_offset, output_bs, group_list_type);
}
at::Tensor npu_indexing(const at::Tensor & self, at::IntArrayRef begin, at::IntArrayRef end, at::IntArrayRef strides, int64_t begin_mask, int64_t end_mask, int64_t ellipsis_mask, int64_t new_axis_mask, int64_t shrink_axis_mask){
    return acl_op::npu_indexing(self, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask);
}
at::Tensor npu_interleave_rope(const at::Tensor & x, const at::Tensor & cos, const at::Tensor & sin){
    bool x_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x);
    bool cos_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(cos);
    bool sin_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(sin);

    if (!x_base_format || !cos_base_format || !sin_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_interleave_rope do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_interleave_rope(x, cos, sin);
}
at::Tensor npu_iou(const at::Tensor & bboxes, const at::Tensor & gtboxes, int64_t mode){
    return acl_op::npu_iou(bboxes, gtboxes, mode);
}
at::Tensor npu_layer_norm_eval(const at::Tensor & input, at::IntArrayRef normalized_shape, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, double eps){
    return acl_op::npu_layer_norm_eval(input, normalized_shape, weight, bias, eps);
}
at::Tensor npu_linear(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias);

    ASCEND_LOGI("npu_linear exec with jit compile: %d, input is internal format: %d, weight is internal format: %d, bias is internal format: %d",
                !is_jit_disable, !input_base_format, !weight_base_format, !bias_base_format);
    if (is_jit_disable && input_base_format && weight_base_format && bias_base_format) {
        return op_api::npu_linear(input, weight, bias);
    } else {
        return acl_op::npu_linear(input, weight, bias);
    }
}
at::Tensor npu_masked_fill_range(const at::Tensor & self, const at::Tensor & start, const at::Tensor & end, const at::Tensor & value, int64_t axis){
    return acl_op::npu_masked_fill_range(self, start, end, value, axis);
}
at::Tensor npu_masked_softmax_with_rel_pos_bias(const at::Tensor & x, const c10::optional<at::Tensor> & atten_mask, const at::Tensor & relative_pos_bias, double scale_value, int64_t inner_precision_mode){
    bool x_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x);
    bool atten_mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(atten_mask);
    bool relative_pos_bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(relative_pos_bias);

    if (!x_base_format || !atten_mask_base_format || !relative_pos_bias_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_masked_softmax_with_rel_pos_bias do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_masked_softmax_with_rel_pos_bias(x, atten_mask, relative_pos_bias, scale_value, inner_precision_mode);
}
at::Tensor npu_max_backward_symint(const at::Tensor & grad, int64_t dim, const at::Tensor & indices, c10::SymIntArrayRef sizes, bool keepdim){
    return acl_op::npu_max_backward_symint(grad, dim, indices, sizes, keepdim);
}
at::Tensor npu_min_backward_symint(const at::Tensor & grad, int64_t dim, const at::Tensor & indices, c10::SymIntArrayRef sizes, bool keepdim){
    return acl_op::npu_min_backward_symint(grad, dim, indices, sizes, keepdim);
}
at::Tensor npu_mish(const at::Tensor & self){
    return acl_op::npu_mish(self);
}
at::Tensor npu_mish_backward(const at::Tensor & grad, const at::Tensor & input){
    return acl_op::npu_mish_backward(grad, input);
}
at::Tensor npu_mm_all_reduce_base(const at::Tensor & x1, const at::Tensor & x2, c10::string_view hcom, c10::string_view reduce_op, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & antiquant_scale, const c10::optional<at::Tensor> & antiquant_offset, const c10::optional<at::Tensor> & x3, const c10::optional<at::Tensor> & dequant_scale, const c10::optional<at::Tensor> & pertoken_scale, const c10::optional<at::Tensor> & comm_quant_scale_1, const c10::optional<at::Tensor> & comm_quant_scale_2, int64_t antiquant_group_size, int64_t comm_turn){
    return op_api::npu_mm_all_reduce_base(x1, x2, hcom, reduce_op, bias, antiquant_scale, antiquant_offset, x3, dequant_scale, pertoken_scale, comm_quant_scale_1, comm_quant_scale_2, antiquant_group_size, comm_turn);
}
at::Tensor npu_mm_reduce_scatter_base(const at::Tensor & self, const at::Tensor & x2, c10::string_view hcom, int64_t world_size, c10::string_view reduce_op, const c10::optional<at::Tensor> & bias, int64_t comm_turn){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool x2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x2);
    bool bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias);

    if (!self_base_format || !x2_base_format || !bias_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_mm_reduce_scatter_base do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_mm_reduce_scatter_base(self, x2, hcom, world_size, reduce_op, bias, comm_turn);
}
at::Tensor npu_moe_compute_expert_tokens(const at::Tensor & sorted_expert_for_source_row, int64_t num_expert){
    bool sorted_expert_for_source_row_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(sorted_expert_for_source_row);

    if (!sorted_expert_for_source_row_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_moe_compute_expert_tokens do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_moe_compute_expert_tokens(sorted_expert_for_source_row, num_expert);
}
at::Tensor npu_moe_distribute_combine(const at::Tensor & expand_x, const at::Tensor & expert_ids, const at::Tensor & expand_idx, const at::Tensor & ep_send_counts, const at::Tensor & expert_scales, c10::string_view group_ep, int64_t ep_world_size, int64_t ep_rank_id, int64_t moe_expert_num, const c10::optional<at::Tensor> & tp_send_counts, const c10::optional<at::Tensor> & x_active_mask, const c10::optional<at::Tensor> & activation_scale, const c10::optional<at::Tensor> & weight_scale, const c10::optional<at::Tensor> & group_list, const c10::optional<at::Tensor> & expand_scales, const c10::optional<at::Tensor> & shared_expert_x, c10::string_view group_tp, int64_t tp_world_size, int64_t tp_rank_id, int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num, int64_t global_bs, int64_t out_dtype, int64_t comm_quant_mode, int64_t group_list_type){
    bool expand_x_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(expand_x);
    bool expert_ids_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(expert_ids);
    bool expand_idx_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(expand_idx);
    bool ep_send_counts_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(ep_send_counts);
    bool expert_scales_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(expert_scales);
    bool tp_send_counts_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tp_send_counts);
    bool x_active_mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x_active_mask);
    bool activation_scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(activation_scale);
    bool weight_scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight_scale);
    bool group_list_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(group_list);
    bool expand_scales_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(expand_scales);
    bool shared_expert_x_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(shared_expert_x);

    if (!expand_x_base_format || !expert_ids_base_format || !expand_idx_base_format || !ep_send_counts_base_format || !expert_scales_base_format || !tp_send_counts_base_format || !x_active_mask_base_format || !activation_scale_base_format || !weight_scale_base_format || !group_list_base_format || !expand_scales_base_format || !shared_expert_x_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_moe_distribute_combine do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_moe_distribute_combine(expand_x, expert_ids, expand_idx, ep_send_counts, expert_scales, group_ep, ep_world_size, ep_rank_id, moe_expert_num, tp_send_counts, x_active_mask, activation_scale, weight_scale, group_list, expand_scales, shared_expert_x, group_tp, tp_world_size, tp_rank_id, expert_shard_type, shared_expert_num, shared_expert_rank_num, global_bs, out_dtype, comm_quant_mode, group_list_type);
}
at::Tensor npu_moe_distribute_combine_v2(const at::Tensor & expand_x, const at::Tensor & expert_ids, const at::Tensor & assist_info_for_combine, const at::Tensor & ep_send_counts, const at::Tensor & expert_scales, c10::string_view group_ep, int64_t ep_world_size, int64_t ep_rank_id, int64_t moe_expert_num, const c10::optional<at::Tensor> & tp_send_counts, const c10::optional<at::Tensor> & x_active_mask, const c10::optional<at::Tensor> & expand_scales, const c10::optional<at::Tensor> & shared_expert_x, c10::string_view group_tp, int64_t tp_world_size, int64_t tp_rank_id, int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num, int64_t global_bs, int64_t comm_quant_mode){
    bool expand_x_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(expand_x);
    bool expert_ids_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(expert_ids);
    bool assist_info_for_combine_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(assist_info_for_combine);
    bool ep_send_counts_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(ep_send_counts);
    bool expert_scales_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(expert_scales);
    bool tp_send_counts_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tp_send_counts);
    bool x_active_mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x_active_mask);
    bool expand_scales_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(expand_scales);
    bool shared_expert_x_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(shared_expert_x);

    if (!expand_x_base_format || !expert_ids_base_format || !assist_info_for_combine_base_format || !ep_send_counts_base_format || !expert_scales_base_format || !tp_send_counts_base_format || !x_active_mask_base_format || !expand_scales_base_format || !shared_expert_x_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_moe_distribute_combine_v2 do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_moe_distribute_combine_v2(expand_x, expert_ids, assist_info_for_combine, ep_send_counts, expert_scales, group_ep, ep_world_size, ep_rank_id, moe_expert_num, tp_send_counts, x_active_mask, expand_scales, shared_expert_x, group_tp, tp_world_size, tp_rank_id, expert_shard_type, shared_expert_num, shared_expert_rank_num, global_bs, comm_quant_mode);
}
at::Tensor npu_moe_eplb_update_expert(const at::Tensor & expert_ids, const at::Tensor & eplb_table, int64_t local_rank_id, int64_t world_size, int64_t balance_mode){
    bool expert_ids_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(expert_ids);
    bool eplb_table_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(eplb_table);

    if (!expert_ids_base_format || !eplb_table_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_moe_eplb_update_expert do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_moe_eplb_update_expert(expert_ids, eplb_table, local_rank_id, world_size, balance_mode);
}
at::Tensor npu_moe_finalize_routing(const at::Tensor & expanded_permuted_rows, const c10::optional<at::Tensor> & skip1, const c10::optional<at::Tensor> & skip2, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & scales, const at::Tensor & expanded_src_to_dst_row, const c10::optional<at::Tensor> & export_for_source_row, c10::optional<int64_t> drop_pad_mode){
    bool expanded_permuted_rows_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(expanded_permuted_rows);
    bool skip1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(skip1);
    bool skip2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(skip2);
    bool bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias);
    bool scales_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(scales);
    bool expanded_src_to_dst_row_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(expanded_src_to_dst_row);
    bool export_for_source_row_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(export_for_source_row);

    if (!expanded_permuted_rows_base_format || !skip1_base_format || !skip2_base_format || !bias_base_format || !scales_base_format || !expanded_src_to_dst_row_base_format || !export_for_source_row_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_moe_finalize_routing do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_moe_finalize_routing(expanded_permuted_rows, skip1, skip2, bias, scales, expanded_src_to_dst_row, export_for_source_row, drop_pad_mode);
}
at::Tensor npu_moe_token_unpermute(const at::Tensor & permuted_tokens, const at::Tensor & sorted_indices, const c10::optional<at::Tensor> & probs, bool padded_mode, at::OptionalIntArrayRef restore_shape){
    bool permuted_tokens_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(permuted_tokens);
    bool sorted_indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(sorted_indices);
    bool probs_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(probs);

    if (!permuted_tokens_base_format || !sorted_indices_base_format || !probs_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_moe_token_unpermute do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_moe_token_unpermute(permuted_tokens, sorted_indices, probs, padded_mode, restore_shape);
}
at::Tensor npu_normalize_batch(const at::Tensor & self, const at::Tensor & seq_len, int64_t normalize_type){
    return acl_op::npu_normalize_batch(self, seq_len, normalize_type);
}
at::Tensor npu_nsa_compress(const at::Tensor & input, const at::Tensor & weight, int64_t compress_block_size, int64_t compress_stride, at::OptionalIntArrayRef actual_seq_len){
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);

    if (!input_base_format || !weight_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_nsa_compress do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_nsa_compress(input, weight, compress_block_size, compress_stride, actual_seq_len);
}
at::Tensor npu_nsa_select_attention_infer(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & topk_indices, double scale_value, int64_t head_num, int64_t key_value_head_num, int64_t select_block_size, int64_t select_block_count, int64_t page_block_size, c10::string_view layout, const c10::optional<at::Tensor> & atten_mask, const c10::optional<at::Tensor> & block_table, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_seq_kvlen){
    bool query_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(query);
    bool key_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(key);
    bool value_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(value);
    bool topk_indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(topk_indices);
    bool atten_mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(atten_mask);
    bool block_table_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(block_table);

    if (!query_base_format || !key_base_format || !value_base_format || !topk_indices_base_format || !atten_mask_base_format || !block_table_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_nsa_select_attention_infer do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_nsa_select_attention_infer(query, key, value, topk_indices, scale_value, head_num, key_value_head_num, select_block_size, select_block_count, page_block_size, layout, atten_mask, block_table, actual_seq_qlen, actual_seq_kvlen);
}
at::Tensor npu_one_hot(const at::Tensor & self, int64_t num_classes, int64_t depth, const at::Scalar & on_value, const at::Scalar & off_value){
    return acl_op::npu_one_hot(self, num_classes, depth, on_value, off_value);
}
at::Tensor npu_our_incre_flash_attention_symint(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const c10::optional<at::Tensor> & padding_mask, const c10::optional<at::Tensor> & atten_mask, const c10::optional<at::Tensor> & pse_shift, at::OptionalSymIntArrayRef actual_seq_lengths, const c10::optional<at::Tensor> & antiquant_scale, const c10::optional<at::Tensor> & antiquant_offset, const c10::optional<at::Tensor> & block_table, const c10::optional<at::Tensor> & dequant_scale1, const c10::optional<at::Tensor> & quant_scale1, const c10::optional<at::Tensor> & dequant_scale2, const c10::optional<at::Tensor> & quant_scale2, const c10::optional<at::Tensor> & quant_offset2, const c10::optional<at::Tensor> & kv_padding_size, int64_t num_heads, double scale_value, c10::string_view input_layout, int64_t num_key_value_heads, int64_t block_size, int64_t inner_precise){
    bool query_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(query);
    bool key_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(key);
    bool value_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(value);
    bool padding_mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(padding_mask);
    bool atten_mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(atten_mask);
    bool pse_shift_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(pse_shift);
    bool antiquant_scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(antiquant_scale);
    bool antiquant_offset_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(antiquant_offset);
    bool block_table_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(block_table);
    bool dequant_scale1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(dequant_scale1);
    bool quant_scale1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(quant_scale1);
    bool dequant_scale2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(dequant_scale2);
    bool quant_scale2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(quant_scale2);
    bool quant_offset2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(quant_offset2);
    bool kv_padding_size_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(kv_padding_size);

    if (!query_base_format || !key_base_format || !value_base_format || !padding_mask_base_format || !atten_mask_base_format || !pse_shift_base_format || !antiquant_scale_base_format || !antiquant_offset_base_format || !block_table_base_format || !dequant_scale1_base_format || !quant_scale1_base_format || !dequant_scale2_base_format || !quant_scale2_base_format || !quant_offset2_base_format || !kv_padding_size_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_our_incre_flash_attention_symint do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_our_incre_flash_attention_symint(query, key, value, padding_mask, atten_mask, pse_shift, actual_seq_lengths, antiquant_scale, antiquant_offset, block_table, dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2, kv_padding_size, num_heads, scale_value, input_layout, num_key_value_heads, block_size, inner_precise);
}
at::Tensor npu_pad(const at::Tensor & input, at::IntArrayRef paddings){
    return acl_op::npu_pad(input, paddings);
}
at::Tensor npu_prompt_flash_attention(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const c10::optional<at::Tensor> & padding_mask, const c10::optional<at::Tensor> & atten_mask, const c10::optional<at::Tensor> & pse_shift, at::OptionalIntArrayRef actual_seq_lengths, const c10::optional<at::Tensor> & deq_scale1, const c10::optional<at::Tensor> & quant_scale1, const c10::optional<at::Tensor> & deq_scale2, const c10::optional<at::Tensor> & quant_scale2, const c10::optional<at::Tensor> & quant_offset2, int64_t num_heads, double scale_value, int64_t pre_tokens, int64_t next_tokens, c10::string_view input_layout, int64_t num_key_value_heads, at::OptionalIntArrayRef actual_seq_lengths_kv, int64_t sparse_mode){
    bool query_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(query);
    bool key_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(key);
    bool value_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(value);
    bool padding_mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(padding_mask);
    bool atten_mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(atten_mask);
    bool pse_shift_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(pse_shift);
    bool deq_scale1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(deq_scale1);
    bool quant_scale1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(quant_scale1);
    bool deq_scale2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(deq_scale2);
    bool quant_scale2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(quant_scale2);
    bool quant_offset2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(quant_offset2);

    if (!query_base_format || !key_base_format || !value_base_format || !padding_mask_base_format || !atten_mask_base_format || !pse_shift_base_format || !deq_scale1_base_format || !quant_scale1_base_format || !deq_scale2_base_format || !quant_scale2_base_format || !quant_offset2_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_prompt_flash_attention do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_prompt_flash_attention(query, key, value, padding_mask, atten_mask, pse_shift, actual_seq_lengths, deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2, num_heads, scale_value, pre_tokens, next_tokens, input_layout, num_key_value_heads, actual_seq_lengths_kv, sparse_mode);
}
at::Tensor npu_ps_roi_pooling(const at::Tensor & self, const at::Tensor & rois, double spatial_scale, int64_t group_size, int64_t output_dim){
    return acl_op::npu_ps_roi_pooling(self, rois, spatial_scale, group_size, output_dim);
}
at::Tensor npu_ps_roi_pooling_backward_symint(const at::Tensor & output_grad, const at::Tensor & rois, double spatial_scale, int64_t group_size, int64_t output_dim, c10::SymIntArrayRef input_size){
    return acl_op::npu_ps_roi_pooling_backward_symint(output_grad, rois, spatial_scale, group_size, output_dim, input_size);
}
at::Tensor npu_ptiou(const at::Tensor & bboxes, const at::Tensor & gtboxes, int64_t mode){
    return acl_op::npu_ptiou(bboxes, gtboxes, mode);
}
at::Tensor npu_quant_conv2d(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & scale, at::IntArrayRef strides, at::IntArrayRef pads, at::IntArrayRef dilations, int64_t groups, int64_t offset_x, c10::string_view round_mode, c10::optional<at::ScalarType> output_dtype, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & offset){
    return acl_op::npu_quant_conv2d(input, weight, scale, strides, pads, dilations, groups, offset_x, round_mode, output_dtype, bias, offset);
}
at::Tensor npu_quant_grouped_matmul_dequant(const at::Tensor & x, const at::Tensor & quantized_weight, const at::Tensor & weight_scale, const at::Tensor & group_list, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & x_scale, const c10::optional<at::Tensor> & x_offset, const c10::optional<at::Tensor> & smooth_scale, c10::optional<c10::string_view> quant_mode){
    bool x_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x);
    bool quantized_weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(quantized_weight);
    bool weight_scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight_scale);
    bool group_list_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(group_list);
    bool bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias);
    bool x_scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x_scale);
    bool x_offset_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x_offset);
    bool smooth_scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(smooth_scale);

    if (!x_base_format || !quantized_weight_base_format || !weight_scale_base_format || !group_list_base_format || !bias_base_format || !x_scale_base_format || !x_offset_base_format || !smooth_scale_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_quant_grouped_matmul_dequant do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_quant_grouped_matmul_dequant(x, quantized_weight, weight_scale, group_list, bias, x_scale, x_offset, smooth_scale, quant_mode);
}
at::Tensor npu_quant_matmul_dequant(const at::Tensor & x, const at::Tensor & quantized_weight, const at::Tensor & weight_scale, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & x_scale, const c10::optional<at::Tensor> & x_offset, const c10::optional<at::Tensor> & smooth_scale, c10::optional<c10::string_view> quant_mode){
    bool x_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x);
    bool quantized_weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(quantized_weight);
    bool weight_scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight_scale);
    bool bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias);
    bool x_scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x_scale);
    bool x_offset_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(x_offset);
    bool smooth_scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(smooth_scale);

    if (!x_base_format || !quantized_weight_base_format || !weight_scale_base_format || !bias_base_format || !x_scale_base_format || !x_offset_base_format || !smooth_scale_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_quant_matmul_dequant do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_quant_matmul_dequant(x, quantized_weight, weight_scale, bias, x_scale, x_offset, smooth_scale, quant_mode);
}
at::Tensor npu_quant_matmul_symint(const at::Tensor & x1, const at::Tensor & x2, const at::Tensor & scale, const c10::optional<at::Tensor> & offset, const c10::optional<at::Tensor> & pertoken_scale, const c10::optional<at::Tensor> & bias, c10::optional<at::ScalarType> output_dtype, at::OptionalSymIntArrayRef group_sizes){
    return op_api::npu_quant_matmul_symint(x1, x2, scale, offset, pertoken_scale, bias, output_dtype, group_sizes);
}
at::Tensor npu_quant_scatter(const at::Tensor & self, const at::Tensor & indices, const at::Tensor & updates, const at::Tensor & quant_scales, const c10::optional<at::Tensor> & quant_zero_points, int64_t axis, int64_t quant_axis, c10::string_view reduce){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);
    bool updates_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(updates);
    bool quant_scales_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(quant_scales);
    bool quant_zero_points_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(quant_zero_points);

    if (!self_base_format || !indices_base_format || !updates_base_format || !quant_scales_base_format || !quant_zero_points_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_quant_scatter do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_quant_scatter(self, indices, updates, quant_scales, quant_zero_points, axis, quant_axis, reduce);
}
at::Tensor npu_quantize(const at::Tensor & self, const at::Tensor & scales, const c10::optional<at::Tensor> & zero_points, at::ScalarType dtype, int64_t axis, bool div_mode){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool scales_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(scales);
    bool zero_points_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(zero_points);

    ASCEND_LOGI("npu_quantize exec with jit compile: %d, self is internal format: %d, scales is internal format: %d, zero_points is internal format: %d",
                !is_jit_disable, !self_base_format, !scales_base_format, !zero_points_base_format);
    if (is_jit_disable && self_base_format && scales_base_format && zero_points_base_format) {
        return op_api::npu_quantize(self, scales, zero_points, dtype, axis, div_mode);
    } else {
        return acl_op::npu_quantize(self, scales, zero_points, dtype, axis, div_mode);
    }
}
at::Tensor npu_reshape(const at::Tensor & self, at::IntArrayRef shape, bool can_refresh){
    return acl_op::npu_reshape(self, shape, can_refresh);
}
at::Tensor npu_roi_align(const at::Tensor & self, const at::Tensor & rois, double spatial_scale, int64_t pooled_height, int64_t pooled_width, int64_t sample_num, int64_t roi_end_mode){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool rois_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(rois);

    ASCEND_LOGI("npu_roi_align exec with jit compile: %d, self is internal format: %d, rois is internal format: %d",
                !is_jit_disable, !self_base_format, !rois_base_format);
    if (is_jit_disable && self_base_format && rois_base_format) {
        return op_api::npu_roi_align(self, rois, spatial_scale, pooled_height, pooled_width, sample_num, roi_end_mode);
    } else {
        return acl_op::npu_roi_align(self, rois, spatial_scale, pooled_height, pooled_width, sample_num, roi_end_mode);
    }
}
at::Tensor npu_roi_alignbk(const at::Tensor & self, const at::Tensor & rois, at::IntArrayRef xdiff_shape, int64_t pooled_width, int64_t pooled_height, double spatial_scale, int64_t sample_num, c10::optional<int64_t> roi_end_mode){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool rois_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(rois);

    ASCEND_LOGI("npu_roi_alignbk exec with jit compile: %d, self is internal format: %d, rois is internal format: %d",
                !is_jit_disable, !self_base_format, !rois_base_format);
    if (is_jit_disable && self_base_format && rois_base_format) {
        return op_api::npu_roi_alignbk(self, rois, xdiff_shape, pooled_width, pooled_height, spatial_scale, sample_num, roi_end_mode);
    } else {
        return acl_op::npu_roi_alignbk(self, rois, xdiff_shape, pooled_width, pooled_height, spatial_scale, sample_num, roi_end_mode);
    }
}
at::Tensor npu_rotary_mul(const at::Tensor & self, const at::Tensor & r1, const at::Tensor & r2, c10::string_view rotary_mode){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool r1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(r1);
    bool r2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(r2);

    ASCEND_LOGI("npu_rotary_mul exec with jit compile: %d, self is internal format: %d, r1 is internal format: %d, r2 is internal format: %d",
                !is_jit_disable, !self_base_format, !r1_base_format, !r2_base_format);
    if (is_jit_disable && self_base_format && r1_base_format && r2_base_format) {
        return op_api::npu_rotary_mul(self, r1, r2, rotary_mode);
    } else {
        return acl_op::npu_rotary_mul(self, r1, r2, rotary_mode);
    }
}
at::Tensor npu_rotated_box_decode(const at::Tensor & self, const at::Tensor & deltas, const at::Tensor & weight){
    return acl_op::npu_rotated_box_decode(self, deltas, weight);
}
at::Tensor npu_rotated_box_encode(const at::Tensor & self, const at::Tensor & gt_bboxes, const at::Tensor & weight){
    return acl_op::npu_rotated_box_encode(self, gt_bboxes, weight);
}
at::Tensor npu_rotated_iou(const at::Tensor & self, const at::Tensor & query_boxes, bool trans, int64_t mode, bool is_cross, double v_threshold, double e_threshold){
    return acl_op::npu_rotated_iou(self, query_boxes, trans, mode, is_cross, v_threshold, e_threshold);
}
at::Tensor npu_rotated_overlaps(const at::Tensor & self, const at::Tensor & query_boxes, bool trans){
    return acl_op::npu_rotated_overlaps(self, query_boxes, trans);
}
at::Tensor npu_scaled_masked_softmax(const at::Tensor & x, const at::Tensor & mask, const at::Scalar & scale, bool fixed_triu_mask){
    return acl_op::npu_scaled_masked_softmax(x, mask, scale, fixed_triu_mask);
}
at::Tensor npu_scaled_masked_softmax_backward(const at::Tensor & y_grad, const at::Tensor & y, const at::Tensor & mask, const at::Scalar & scale, bool fixed_triu_mask){
    return acl_op::npu_scaled_masked_softmax_backward(y_grad, y, mask, scale, fixed_triu_mask);
}
at::Tensor npu_scatter(const at::Tensor & self, const at::Tensor & indices, const at::Tensor & updates, int64_t dim){
    return acl_op::npu_scatter(self, indices, updates, dim);
}
at::Tensor npu_scatter_nd_update(const at::Tensor & self, const at::Tensor & indices, const at::Tensor & updates){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);
    bool updates_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(updates);

    if (!self_base_format || !indices_base_format || !updates_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_scatter_nd_update do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_scatter_nd_update(self, indices, updates);
}
at::Tensor npu_sign_bits_pack(const at::Tensor & self, int64_t size){
    return acl_op::npu_sign_bits_pack(self, size);
}
at::Tensor npu_sign_bits_unpack(const at::Tensor & input, int64_t size, at::ScalarType dtype){
    return acl_op::npu_sign_bits_unpack(input, size, dtype);
}
at::Tensor npu_silu(const at::Tensor & self){
    return acl_op::npu_silu(self);
}
at::Tensor npu_silu_backward(const at::Tensor & grad_output, const at::Tensor & x0, const at::Tensor & x1){
    return acl_op::npu_silu_backward(grad_output, x0, x1);
}
at::Tensor npu_slice(const at::Tensor & self, at::IntArrayRef offsets, at::IntArrayRef size){
    return acl_op::npu_slice(self, offsets, size);
}
at::Tensor npu_softmax_cross_entropy_with_logits(const at::Tensor & self, const at::Tensor & labels){
    return acl_op::npu_softmax_cross_entropy_with_logits(self, labels);
}
at::Tensor npu_softmax_cross_entropy_with_logits_backward(const at::Tensor & grad, const at::Tensor & self, const at::Tensor & labels){
    return acl_op::npu_softmax_cross_entropy_with_logits_backward(grad, self, labels);
}
at::Tensor npu_sort_v2(const at::Tensor & self, int64_t dim, bool descending){
    return acl_op::npu_sort_v2(self, dim, descending);
}
at::Tensor npu_sparse_paged_attention_symint(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const c10::optional<at::Tensor> & padding_mask, const c10::optional<at::Tensor> & atten_mask, const c10::optional<at::Tensor> & pse_shift, at::OptionalSymIntArrayRef actual_seq_lengths, const c10::optional<at::Tensor> & antiquant_scale, const c10::optional<at::Tensor> & antiquant_offset, const c10::optional<at::Tensor> & block_table, const c10::optional<at::Tensor> & block_position, const c10::optional<at::Tensor> & dequant_scale1, const c10::optional<at::Tensor> & quant_scale1, const c10::optional<at::Tensor> & dequant_scale2, const c10::optional<at::Tensor> & quant_scale2, const c10::optional<at::Tensor> & quant_offset2, const c10::optional<at::Tensor> & kv_padding_size, int64_t num_heads, double scale_value, c10::string_view input_layout, int64_t num_key_value_heads, int64_t block_size, int64_t inner_precise){
    bool query_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(query);
    bool key_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(key);
    bool value_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(value);
    bool padding_mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(padding_mask);
    bool atten_mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(atten_mask);
    bool pse_shift_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(pse_shift);
    bool antiquant_scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(antiquant_scale);
    bool antiquant_offset_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(antiquant_offset);
    bool block_table_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(block_table);
    bool block_position_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(block_position);
    bool dequant_scale1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(dequant_scale1);
    bool quant_scale1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(quant_scale1);
    bool dequant_scale2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(dequant_scale2);
    bool quant_scale2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(quant_scale2);
    bool quant_offset2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(quant_offset2);
    bool kv_padding_size_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(kv_padding_size);

    if (!query_base_format || !key_base_format || !value_base_format || !padding_mask_base_format || !atten_mask_base_format || !pse_shift_base_format || !antiquant_scale_base_format || !antiquant_offset_base_format || !block_table_base_format || !block_position_base_format || !dequant_scale1_base_format || !quant_scale1_base_format || !dequant_scale2_base_format || !quant_scale2_base_format || !quant_offset2_base_format || !kv_padding_size_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_sparse_paged_attention_symint do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_sparse_paged_attention_symint(query, key, value, padding_mask, atten_mask, pse_shift, actual_seq_lengths, antiquant_scale, antiquant_offset, block_table, block_position, dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2, kv_padding_size, num_heads, scale_value, input_layout, num_key_value_heads, block_size, inner_precise);
}
at::Tensor npu_stride_add(const at::Tensor & self, const at::Tensor & other, const at::Scalar & offset1, const at::Scalar & offset2, const at::Scalar & c1_len){
    return acl_op::npu_stride_add(self, other, offset1, offset2, c1_len);
}
at::Tensor npu_stride_copy(const at::Tensor & self, at::IntArrayRef shape, at::IntArrayRef stride, const at::Scalar & storage_offset){
    return acl_op::npu_stride_copy(self, shape, stride, storage_offset);
}
at::Tensor npu_sub_sample(const at::Tensor & self, int64_t per_images, double positive_fraction){
    return acl_op::npu_sub_sample(self, per_images, positive_fraction);
}
at::Tensor npu_swiglu(const at::Tensor & self, int64_t dim){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_swiglu do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_swiglu(self, dim);
}
at::Tensor npu_swiglu_backward(const at::Tensor & grad_output, const at::Tensor & self, int64_t dim){
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!grad_output_base_format || !self_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_swiglu_backward do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_swiglu_backward(grad_output, self, dim);
}
at::Tensor npu_top_k_top_p(const at::Tensor & logits, const at::Tensor & p, const at::Tensor & k){
    bool logits_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(logits);
    bool p_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(p);
    bool k_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(k);

    if (!logits_base_format || !p_base_format || !k_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_top_k_top_p do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_top_k_top_p(logits, p, k);
}
at::Tensor npu_trans_quant_param(const at::Tensor & scale, const c10::optional<at::Tensor> & offset, c10::optional<int64_t> round_mode){
    bool scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(scale);
    bool offset_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(offset);

    if (!scale_base_format || !offset_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_trans_quant_param do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_trans_quant_param(scale, offset, round_mode);
}
at::Tensor npu_transpose(const at::Tensor & self, at::IntArrayRef perm, bool require_contiguous){
    return acl_op::npu_transpose(self, perm, require_contiguous);
}
at::Tensor npu_transpose_batchmatmul(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & scale, at::OptionalIntArrayRef perm_x1, at::OptionalIntArrayRef perm_x2, at::OptionalIntArrayRef perm_y, c10::optional<int64_t> batch_split_factor){
    bool input_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias);
    bool scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(scale);

    if (!input_base_format || !weight_base_format || !bias_base_format || !scale_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_transpose_batchmatmul do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_transpose_batchmatmul(input, weight, bias, scale, perm_x1, perm_x2, perm_y, batch_split_factor);
}
at::Tensor npu_weight_quant_batchmatmul(const at::Tensor & x, const at::Tensor & weight, const at::Tensor & antiquant_scale, const c10::optional<at::Tensor> & antiquant_offset, const c10::optional<at::Tensor> & quant_scale, const c10::optional<at::Tensor> & quant_offset, const c10::optional<at::Tensor> & bias, int64_t antiquant_group_size, int64_t inner_precise){
    return op_api::npu_weight_quant_batchmatmul(x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias, antiquant_group_size, inner_precise);
}
at::Tensor npu_yolo_boxes_encode(const at::Tensor & self, const at::Tensor & gt_bboxes, const at::Tensor & stride, bool performance_mode){
    return acl_op::npu_yolo_boxes_encode(self, gt_bboxes, stride, performance_mode);
}
at::Tensor one_hot(const at::Tensor & self, int64_t num_classes){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("one_hot exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::one_hot(self, num_classes);
    } else {
        return acl_op::one_hot(self, num_classes);
    }
}
at::Tensor ones(at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();

    ASCEND_LOGI("ones exec with jit compile: %d",
                !is_jit_disable);
    if (is_jit_disable) {
        return op_api::ones(size, names, dtype, layout, device, pin_memory);
    } else {
        return acl_op::ones(size, names, dtype, layout, device, pin_memory);
    }
}
at::Tensor ones(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();

    ASCEND_LOGI("ones exec with jit compile: %d",
                !is_jit_disable);
    if (is_jit_disable) {
        return op_api::ones(size, dtype, layout, device, pin_memory);
    } else {
        return acl_op::ones(size, dtype, layout, device, pin_memory);
    }
}
at::Tensor ones_like(const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("ones_like exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::ones_like(self, dtype, layout, device, pin_memory, memory_format);
    } else {
        return acl_op::ones_like(self, dtype, layout, device, pin_memory, memory_format);
    }
}
at::Tensor pdist(const at::Tensor & self, double p){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("pdist exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::pdist(self, p);
    } else {
        return acl_op::pdist(self, p);
    }
}
at::Tensor polar(const at::Tensor & abs, const at::Tensor & angle){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool abs_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(abs);
    bool angle_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(angle);

    ASCEND_LOGI("polar exec with jit compile: %d, abs is internal format: %d, angle is internal format: %d",
                !is_jit_disable, !abs_base_format, !angle_base_format);
    if (is_jit_disable && abs_base_format && angle_base_format) {
        return op_api::polar(abs, angle);
    } else {
        return acl_op::polar(abs, angle);
    }
}
at::Tensor pow(const at::Scalar & self, const at::Tensor & exponent){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool exponent_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(exponent);

    ASCEND_LOGI("pow exec with jit compile: %d, exponent is internal format: %d",
                !is_jit_disable, !exponent_base_format);
    if (is_jit_disable && exponent_base_format) {
        return op_api::pow(self, exponent);
    } else {
        return acl_op::pow(self, exponent);
    }
}
at::Tensor pow(const at::Tensor & self, const at::Scalar & exponent){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("pow exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::pow(self, exponent);
    } else {
        return acl_op::pow(self, exponent);
    }
}
at::Tensor pow(const at::Tensor & self, const at::Tensor & exponent){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool exponent_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(exponent);

    ASCEND_LOGI("pow exec with jit compile: %d, self is internal format: %d, exponent is internal format: %d",
                !is_jit_disable, !self_base_format, !exponent_base_format);
    if (is_jit_disable && self_base_format && exponent_base_format) {
        return op_api::pow(self, exponent);
    } else {
        return acl_op::pow(self, exponent);
    }
}
at::Tensor prod(const at::Tensor & self, c10::optional<at::ScalarType> dtype){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("prod exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::prod(self, dtype);
    } else {
        return acl_op::prod(self, dtype);
    }
}
at::Tensor prod(const at::Tensor & self, int64_t dim, bool keepdim, c10::optional<at::ScalarType> dtype){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("prod exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::prod(self, dim, keepdim, dtype);
    } else {
        return acl_op::prod(self, dim, keepdim, dtype);
    }
}
at::Tensor quantize_per_channel(const at::Tensor & self, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, at::ScalarType dtype){
    return acl_op::quantize_per_channel(self, scales, zero_points, axis, dtype);
}
at::Tensor quantize_per_tensor(const at::Tensor & self, double scale, int64_t zero_point, at::ScalarType dtype){
    return acl_op::quantize_per_tensor(self, scale, zero_point, dtype);
}
at::Tensor randperm(int64_t n, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();

    ASCEND_LOGI("randperm exec with jit compile: %d",
                !is_jit_disable);
    if (is_jit_disable) {
        return op_api::randperm(n, generator, dtype, layout, device, pin_memory);
    } else {
        return acl_op::randperm(n, generator, dtype, layout, device, pin_memory);
    }
}
at::Tensor randperm(int64_t n, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();

    ASCEND_LOGI("randperm exec with jit compile: %d",
                !is_jit_disable);
    if (is_jit_disable) {
        return op_api::randperm(n, dtype, layout, device, pin_memory);
    } else {
        return acl_op::randperm(n, dtype, layout, device, pin_memory);
    }
}
at::Tensor range(const at::Scalar & start, const at::Scalar & end, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();

    ASCEND_LOGI("range exec with jit compile: %d",
                !is_jit_disable);
    if (is_jit_disable) {
        return op_api::range(start, end, dtype, layout, device, pin_memory);
    } else {
        return acl_op::range(start, end, dtype, layout, device, pin_memory);
    }
}
at::Tensor range(const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();

    ASCEND_LOGI("range exec with jit compile: %d",
                !is_jit_disable);
    if (is_jit_disable) {
        return op_api::range(start, end, step, dtype, layout, device, pin_memory);
    } else {
        return acl_op::range(start, end, step, dtype, layout, device, pin_memory);
    }
}
at::Tensor reciprocal(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("reciprocal exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::reciprocal(self);
    } else {
        return acl_op::reciprocal(self);
    }
}
at::Tensor reflection_pad1d(const at::Tensor & self, at::IntArrayRef padding){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("reflection_pad1d exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::reflection_pad1d(self, padding);
    } else {
        return acl_op::reflection_pad1d(self, padding);
    }
}
at::Tensor reflection_pad1d_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("reflection_pad1d_backward exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format) {
        return op_api::reflection_pad1d_backward(grad_output, self, padding);
    } else {
        return acl_op::reflection_pad1d_backward(grad_output, self, padding);
    }
}
at::Tensor reflection_pad2d(const at::Tensor & self, at::IntArrayRef padding){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("reflection_pad2d exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::reflection_pad2d(self, padding);
    } else {
        return acl_op::reflection_pad2d(self, padding);
    }
}
at::Tensor reflection_pad2d_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("reflection_pad2d_backward exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format) {
        return op_api::reflection_pad2d_backward(grad_output, self, padding);
    } else {
        return acl_op::reflection_pad2d_backward(grad_output, self, padding);
    }
}
at::Tensor reflection_pad3d(const at::Tensor & self, at::IntArrayRef padding){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator reflection_pad3d do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::reflection_pad3d(self, padding);
}
at::Tensor reflection_pad3d_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding){
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!grad_output_base_format || !self_base_format) {
        TORCH_CHECK(false,
            "Current operator reflection_pad3d_backward do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::reflection_pad3d_backward(grad_output, self, padding);
}
at::Tensor relu(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("relu exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::relu(self);
    } else {
        return acl_op::relu(self);
    }
}
at::Tensor remainder(const at::Scalar & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("remainder exec with jit compile: %d, other is internal format: %d",
                !is_jit_disable, !other_base_format);
    if (is_jit_disable && other_base_format) {
        return op_api::remainder(self, other);
    } else {
        return acl_op::remainder(self, other);
    }
}
at::Tensor remainder(const at::Tensor & self, const at::Scalar & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("remainder exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::remainder(self, other);
    } else {
        return acl_op::remainder(self, other);
    }
}
at::Tensor remainder(const at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("remainder exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::remainder(self, other);
    } else {
        return acl_op::remainder(self, other);
    }
}
at::Tensor renorm(const at::Tensor & self, const at::Scalar & p, int64_t dim, const at::Scalar & maxnorm){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("renorm exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::renorm(self, p, dim, maxnorm);
    } else {
        return acl_op::renorm(self, p, dim, maxnorm);
    }
}
at::Tensor repeat(const at::Tensor & self, at::IntArrayRef repeats){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("repeat exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::repeat(self, repeats);
    } else {
        return acl_op::repeat(self, repeats);
    }
}
at::Tensor repeat_interleave_backward_int_symint(const at::Tensor & grad, const at::Tensor & self, c10::SymInt repeats, c10::optional<int64_t> dim){
    bool grad_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!grad_base_format || !self_base_format) {
        TORCH_CHECK(false,
            "Current operator repeat_interleave_backward_int_symint do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::repeat_interleave_backward_int_symint(grad, self, repeats, dim);
}
at::Tensor repeat_interleave_backward_tensor(const at::Tensor & grad, const at::Tensor & self, const at::Tensor & repeats, c10::optional<int64_t> dim){
    bool grad_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool repeats_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(repeats);

    if (!grad_base_format || !self_base_format || !repeats_base_format) {
        TORCH_CHECK(false,
            "Current operator repeat_interleave_backward_tensor do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::repeat_interleave_backward_tensor(grad, self, repeats, dim);
}
at::Tensor repeat_interleave_symint(const at::Tensor & self, c10::SymInt repeats, c10::optional<int64_t> dim, c10::optional<c10::SymInt> output_size){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("repeat_interleave_symint exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::repeat_interleave_symint(self, repeats, dim, output_size);
    } else {
        return acl_op::repeat_interleave_symint(self, repeats, dim, output_size);
    }
}
at::Tensor repeat_interleave_symint(const at::Tensor & self, const at::Tensor & repeats, c10::optional<int64_t> dim, c10::optional<c10::SymInt> output_size){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool repeats_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(repeats);

    ASCEND_LOGI("repeat_interleave_symint exec with jit compile: %d, self is internal format: %d, repeats is internal format: %d",
                !is_jit_disable, !self_base_format, !repeats_base_format);
    if (is_jit_disable && self_base_format && repeats_base_format) {
        return op_api::repeat_interleave_symint(self, repeats, dim, output_size);
    } else {
        return acl_op::repeat_interleave_symint(self, repeats, dim, output_size);
    }
}
at::Tensor replication_pad1d(const at::Tensor & self, at::IntArrayRef padding){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("replication_pad1d exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::replication_pad1d(self, padding);
    } else {
        return acl_op::replication_pad1d(self, padding);
    }
}
at::Tensor replication_pad1d_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("replication_pad1d_backward exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format) {
        return op_api::replication_pad1d_backward(grad_output, self, padding);
    } else {
        return acl_op::replication_pad1d_backward(grad_output, self, padding);
    }
}
at::Tensor replication_pad2d(const at::Tensor & self, at::IntArrayRef padding){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("replication_pad2d exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::replication_pad2d(self, padding);
    } else {
        return acl_op::replication_pad2d(self, padding);
    }
}
at::Tensor replication_pad2d_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("replication_pad2d_backward exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format) {
        return op_api::replication_pad2d_backward(grad_output, self, padding);
    } else {
        return acl_op::replication_pad2d_backward(grad_output, self, padding);
    }
}
at::Tensor replication_pad3d(const at::Tensor & self, at::IntArrayRef padding){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator replication_pad3d do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::replication_pad3d(self, padding);
}
at::Tensor replication_pad3d_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding){
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!grad_output_base_format || !self_base_format) {
        TORCH_CHECK(false,
            "Current operator replication_pad3d_backward do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::replication_pad3d_backward(grad_output, self, padding);
}
at::Tensor roll(const at::Tensor & self, at::IntArrayRef shifts, at::IntArrayRef dims){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("roll exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::roll(self, shifts, dims);
    } else {
        return acl_op::roll(self, shifts, dims);
    }
}
at::Tensor round(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("round exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::round(self);
    } else {
        return acl_op::round(self);
    }
}
at::Tensor round(const at::Tensor & self, int64_t decimals){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("round exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::round(self, decimals);
    } else {
        return acl_op::round(self, decimals);
    }
}
at::Tensor rrelu_with_noise(const at::Tensor & self, const at::Tensor & noise, const at::Scalar & lower, const at::Scalar & upper, bool training, c10::optional<at::Generator> generator){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool noise_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(noise);

    ASCEND_LOGI("rrelu_with_noise exec with jit compile: %d, self is internal format: %d, noise is internal format: %d",
                !is_jit_disable, !self_base_format, !noise_base_format);
    if (is_jit_disable && self_base_format && noise_base_format) {
        return op_api::rrelu_with_noise(self, noise, lower, upper, training, generator);
    } else {
        return acl_op::rrelu_with_noise(self, noise, lower, upper, training, generator);
    }
}
at::Tensor rsqrt(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("rsqrt exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::rsqrt(self);
    } else {
        return acl_op::rsqrt(self);
    }
}
at::Tensor rsub(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("rsub exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::rsub(self, other, alpha);
    } else {
        return acl_op::rsub(self, other, alpha);
    }
}
at::Tensor rsub(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("rsub exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::rsub(self, other, alpha);
    } else {
        return acl_op::rsub(self, other, alpha);
    }
}
at::Tensor scaled_dot_product_attention(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const c10::optional<at::Tensor> & attn_mask, double dropout_p, bool is_causal, c10::optional<double> scale, bool enable_gqa){
    bool query_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(query);
    bool key_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(key);
    bool value_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(value);
    bool attn_mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(attn_mask);

    if (!query_base_format || !key_base_format || !value_base_format || !attn_mask_base_format) {
        TORCH_CHECK(false,
            "Current operator scaled_dot_product_attention do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::scaled_dot_product_attention(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);
}
at::Tensor scatter_add(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & src){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool index_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(index);
    bool src_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(src);

    ASCEND_LOGI("scatter_add exec with jit compile: %d, self is internal format: %d, index is internal format: %d, src is internal format: %d",
                !is_jit_disable, !self_base_format, !index_base_format, !src_base_format);
    if (is_jit_disable && self_base_format && index_base_format && src_base_format) {
        return op_api::scatter_add(self, dim, index, src);
    } else {
        return acl_op::scatter_add(self, dim, index, src);
    }
}
at::Tensor scatter_add(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool index_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(index);
    bool src_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(src);

    ASCEND_LOGI("scatter_add exec with jit compile: %d, self is internal format: %d, index is internal format: %d, src is internal format: %d",
                !is_jit_disable, !self_base_format, !index_base_format, !src_base_format);
    if (is_jit_disable && self_base_format && index_base_format && src_base_format) {
        return op_api::scatter_add(self, dim, index, src);
    } else {
        return acl_op::scatter_add(self, dim, index, src);
    }
}
at::Tensor scatter_update(const at::Tensor & self, const at::Tensor & indices, const at::Tensor & updates, int64_t axis){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);
    bool updates_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(updates);

    ASCEND_LOGI("scatter_update exec with jit compile: %d, self is internal format: %d, indices is internal format: %d, updates is internal format: %d",
                !is_jit_disable, !self_base_format, !indices_base_format, !updates_base_format);
    if (is_jit_disable && self_base_format && indices_base_format && updates_base_format) {
        return op_api::scatter_update(self, indices, updates, axis);
    } else {
        return acl_op::scatter_update(self, indices, updates, axis);
    }
}
at::Tensor searchsorted(const at::Tensor & sorted_sequence, const at::Scalar & self, bool out_int32, bool right, c10::optional<c10::string_view> side, const c10::optional<at::Tensor> & sorter){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool sorted_sequence_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(sorted_sequence);
    bool sorter_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(sorter);

    ASCEND_LOGI("searchsorted exec with jit compile: %d, sorted_sequence is internal format: %d, sorter is internal format: %d",
                !is_jit_disable, !sorted_sequence_base_format, !sorter_base_format);
    if (is_jit_disable && sorted_sequence_base_format && sorter_base_format) {
        return op_api::searchsorted(sorted_sequence, self, out_int32, right, side, sorter);
    } else {
        return acl_op::searchsorted(sorted_sequence, self, out_int32, right, side, sorter);
    }
}
at::Tensor searchsorted(const at::Tensor & sorted_sequence, const at::Tensor & self, bool out_int32, bool right, c10::optional<c10::string_view> side, const c10::optional<at::Tensor> & sorter){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool sorted_sequence_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(sorted_sequence);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool sorter_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(sorter);

    ASCEND_LOGI("searchsorted exec with jit compile: %d, sorted_sequence is internal format: %d, self is internal format: %d, sorter is internal format: %d",
                !is_jit_disable, !sorted_sequence_base_format, !self_base_format, !sorter_base_format);
    if (is_jit_disable && sorted_sequence_base_format && self_base_format && sorter_base_format) {
        return op_api::searchsorted(sorted_sequence, self, out_int32, right, side, sorter);
    } else {
        return acl_op::searchsorted(sorted_sequence, self, out_int32, right, side, sorter);
    }
}
at::Tensor sgn(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("sgn exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::sgn(self);
    } else {
        return acl_op::sgn(self);
    }
}
at::Tensor sigmoid(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("sigmoid exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::sigmoid(self);
    } else {
        return acl_op::sigmoid(self);
    }
}
at::Tensor sigmoid_backward(const at::Tensor & grad_output, const at::Tensor & output){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(output);

    ASCEND_LOGI("sigmoid_backward exec with jit compile: %d, grad_output is internal format: %d, output is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !output_base_format);
    if (is_jit_disable && grad_output_base_format && output_base_format) {
        return op_api::sigmoid_backward(grad_output, output);
    } else {
        return acl_op::sigmoid_backward(grad_output, output);
    }
}
at::Tensor sign(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("sign exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::sign(self);
    } else {
        return acl_op::sign(self);
    }
}
at::Tensor silu(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("silu exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::silu(self);
    } else {
        return acl_op::silu(self);
    }
}
at::Tensor silu_backward(const at::Tensor & grad_output, const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("silu_backward exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format) {
        return op_api::silu_backward(grad_output, self);
    } else {
        return acl_op::silu_backward(grad_output, self);
    }
}
at::Tensor sin(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("sin exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::sin(self);
    } else {
        return acl_op::sin(self);
    }
}
at::Tensor sinc(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("sinc exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::sinc(self);
    } else {
        return acl_op::sinc(self);
    }
}
at::Tensor sinh(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("sinh exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::sinh(self);
    } else {
        return acl_op::sinh(self);
    }
}
at::Tensor slow_conv3d(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding){
    return acl_op::slow_conv3d(self, weight, kernel_size, bias, stride, padding);
}
at::Tensor slow_conv3d_forward(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding){
    return acl_op::slow_conv3d_forward(self, weight, kernel_size, bias, stride, padding);
}
at::Tensor slow_conv_dilated2d(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias);

    ASCEND_LOGI("slow_conv_dilated2d exec with jit compile: %d, self is internal format: %d, weight is internal format: %d, bias is internal format: %d",
                !is_jit_disable, !self_base_format, !weight_base_format, !bias_base_format);
    if (is_jit_disable && self_base_format && weight_base_format && bias_base_format) {
        return op_api::slow_conv_dilated2d(self, weight, kernel_size, bias, stride, padding, dilation);
    } else {
        return acl_op::slow_conv_dilated2d(self, weight, kernel_size, bias, stride, padding, dilation);
    }
}
at::Tensor slow_conv_transpose2d(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias);

    ASCEND_LOGI("slow_conv_transpose2d exec with jit compile: %d, self is internal format: %d, weight is internal format: %d, bias is internal format: %d",
                !is_jit_disable, !self_base_format, !weight_base_format, !bias_base_format);
    if (is_jit_disable && self_base_format && weight_base_format && bias_base_format) {
        return op_api::slow_conv_transpose2d(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
    } else {
        return acl_op::slow_conv_transpose2d(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
    }
}
at::Tensor smooth_l1_loss(const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target);

    ASCEND_LOGI("smooth_l1_loss exec with jit compile: %d, self is internal format: %d, target is internal format: %d",
                !is_jit_disable, !self_base_format, !target_base_format);
    if (is_jit_disable && self_base_format && target_base_format) {
        return op_api::smooth_l1_loss(self, target, reduction, beta);
    } else {
        return acl_op::smooth_l1_loss(self, target, reduction, beta);
    }
}
at::Tensor smooth_l1_loss_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target);

    ASCEND_LOGI("smooth_l1_loss_backward exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, target is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !target_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && target_base_format) {
        return op_api::smooth_l1_loss_backward(grad_output, self, target, reduction, beta);
    } else {
        return acl_op::smooth_l1_loss_backward(grad_output, self, target, reduction, beta);
    }
}
at::Tensor soft_margin_loss(const at::Tensor & self, const at::Tensor & target, int64_t reduction){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target);

    ASCEND_LOGI("soft_margin_loss exec with jit compile: %d, self is internal format: %d, target is internal format: %d",
                !is_jit_disable, !self_base_format, !target_base_format);
    if (is_jit_disable && self_base_format && target_base_format) {
        return op_api::soft_margin_loss(self, target, reduction);
    } else {
        return acl_op::soft_margin_loss(self, target, reduction);
    }
}
at::Tensor soft_margin_loss_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool target_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(target);

    ASCEND_LOGI("soft_margin_loss_backward exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d, target is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format, !target_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format && target_base_format) {
        return op_api::soft_margin_loss_backward(grad_output, self, target, reduction);
    } else {
        return acl_op::soft_margin_loss_backward(grad_output, self, target, reduction);
    }
}
at::Tensor softmax(const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("softmax exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::softmax(self, dim, dtype);
    } else {
        return acl_op::softmax(self, dim, dtype);
    }
}
at::Tensor softmax(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("softmax exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::softmax(self, dim, dtype);
    } else {
        return acl_op::softmax(self, dim, dtype);
    }
}
at::Tensor softplus(const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("softplus exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::softplus(self, beta, threshold);
    } else {
        return acl_op::softplus(self, beta, threshold);
    }
}
at::Tensor softshrink(const at::Tensor & self, const at::Scalar & lambd){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("softshrink exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::softshrink(self, lambd);
    } else {
        return acl_op::softshrink(self, lambd);
    }
}
at::Tensor softshrink_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & lambd){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("softshrink_backward exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format) {
        return op_api::softshrink_backward(grad_output, self, lambd);
    } else {
        return acl_op::softshrink_backward(grad_output, self, lambd);
    }
}
at::Tensor sqrt(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("sqrt exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::sqrt(self);
    } else {
        return acl_op::sqrt(self);
    }
}
at::Tensor stack(at::TensorList tensors, int64_t dim){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool tensors_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors);

    ASCEND_LOGI("stack exec with jit compile: %d, tensors is internal format: %d",
                !is_jit_disable, !tensors_base_format);
    if (is_jit_disable && tensors_base_format) {
        return op_api::stack(tensors, dim);
    } else {
        return acl_op::stack(tensors, dim);
    }
}
at::Tensor std(const at::Tensor & self, at::OptionalIntArrayRef dim, const c10::optional<at::Scalar> & correction, bool keepdim){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("std exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::std(self, dim, correction, keepdim);
    } else {
        return acl_op::std(self, dim, correction, keepdim);
    }
}
at::Tensor stft(const at::Tensor & self, int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const c10::optional<at::Tensor> & window, bool normalized, c10::optional<bool> onesided, c10::optional<bool> return_complex){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool window_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(window);

    if (!self_base_format || !window_base_format) {
        TORCH_CHECK(false,
            "Current operator stft do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::stft(self, n_fft, hop_length, win_length, window, normalized, onesided, return_complex);
}
at::Tensor stft_backward(const at::Tensor & grad_output, const at::Tensor & self, int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const c10::optional<at::Tensor> & window, bool normalized, c10::optional<bool> onesided, c10::optional<bool> return_complex){
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool window_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(window);

    if (!grad_output_base_format || !self_base_format || !window_base_format) {
        TORCH_CHECK(false,
            "Current operator stft_backward do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::stft_backward(grad_output, self, n_fft, hop_length, win_length, window, normalized, onesided, return_complex);
}
at::Tensor sub(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("sub exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::sub(self, other, alpha);
    } else {
        return acl_op::sub(self, other, alpha);
    }
}
at::Tensor sub(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("sub exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::sub(self, other, alpha);
    } else {
        return acl_op::sub(self, other, alpha);
    }
}
at::Tensor sum(const at::Tensor & self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> dtype){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("sum exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::sum(self, dim, keepdim, dtype);
    } else {
        return acl_op::sum(self, dim, keepdim, dtype);
    }
}
at::Tensor sum(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("sum exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::sum(self, dim, keepdim, dtype);
    } else {
        return acl_op::sum(self, dim, keepdim, dtype);
    }
}
at::Tensor sum(const at::Tensor & self, c10::optional<at::ScalarType> dtype){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("sum exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::sum(self, dtype);
    } else {
        return acl_op::sum(self, dtype);
    }
}
at::Tensor take(const at::Tensor & self, const at::Tensor & index){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool index_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(index);

    ASCEND_LOGI("take exec with jit compile: %d, self is internal format: %d, index is internal format: %d",
                !is_jit_disable, !self_base_format, !index_base_format);
    if (is_jit_disable && self_base_format && index_base_format) {
        return op_api::take(self, index);
    } else {
        return acl_op::take(self, index);
    }
}
at::Tensor tan(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("tan exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::tan(self);
    } else {
        return acl_op::tan(self);
    }
}
at::Tensor tanh(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("tanh exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::tanh(self);
    } else {
        return acl_op::tanh(self);
    }
}
at::Tensor tanh_backward(const at::Tensor & grad_output, const at::Tensor & output){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(output);

    ASCEND_LOGI("tanh_backward exec with jit compile: %d, grad_output is internal format: %d, output is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !output_base_format);
    if (is_jit_disable && grad_output_base_format && output_base_format) {
        return op_api::tanh_backward(grad_output, output);
    } else {
        return acl_op::tanh_backward(grad_output, output);
    }
}
at::Tensor threshold(const at::Tensor & self, const at::Scalar & threshold, const at::Scalar & value){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("threshold exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::threshold(self, threshold, value);
    } else {
        return acl_op::threshold(self, threshold, value);
    }
}
at::Tensor threshold_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & threshold){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("threshold_backward exec with jit compile: %d, grad_output is internal format: %d, self is internal format: %d",
                !is_jit_disable, !grad_output_base_format, !self_base_format);
    if (is_jit_disable && grad_output_base_format && self_base_format) {
        return op_api::threshold_backward(grad_output, self, threshold);
    } else {
        return acl_op::threshold_backward(grad_output, self, threshold);
    }
}
at::Tensor trace(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("trace exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::trace(self);
    } else {
        return acl_op::trace(self);
    }
}
at::Tensor tril(const at::Tensor & self, int64_t diagonal){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("tril exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::tril(self, diagonal);
    } else {
        return acl_op::tril(self, diagonal);
    }
}
at::Tensor triu(const at::Tensor & self, int64_t diagonal){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("triu exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::triu(self, diagonal);
    } else {
        return acl_op::triu(self, diagonal);
    }
}
at::Tensor trunc(const at::Tensor & self){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("trunc exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::trunc(self);
    } else {
        return acl_op::trunc(self);
    }
}
at::Tensor upsample_bicubic2d(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("upsample_bicubic2d exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::upsample_bicubic2d(self, output_size, align_corners, scales_h, scales_w);
    } else {
        return acl_op::upsample_bicubic2d(self, output_size, align_corners, scales_h, scales_w);
    }
}
at::Tensor upsample_bicubic2d_backward(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);

    ASCEND_LOGI("upsample_bicubic2d_backward exec with jit compile: %d, grad_output is internal format: %d",
                !is_jit_disable, !grad_output_base_format);
    if (is_jit_disable && grad_output_base_format) {
        return op_api::upsample_bicubic2d_backward(grad_output, output_size, input_size, align_corners, scales_h, scales_w);
    } else {
        return acl_op::upsample_bicubic2d_backward(grad_output, output_size, input_size, align_corners, scales_h, scales_w);
    }
}
at::Tensor upsample_bilinear2d(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("upsample_bilinear2d exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::upsample_bilinear2d(self, output_size, align_corners, scales_h, scales_w);
    } else {
        return acl_op::upsample_bilinear2d(self, output_size, align_corners, scales_h, scales_w);
    }
}
at::Tensor upsample_bilinear2d_backward(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);

    ASCEND_LOGI("upsample_bilinear2d_backward exec with jit compile: %d, grad_output is internal format: %d",
                !is_jit_disable, !grad_output_base_format);
    if (is_jit_disable && grad_output_base_format) {
        return op_api::upsample_bilinear2d_backward(grad_output, output_size, input_size, align_corners, scales_h, scales_w);
    } else {
        return acl_op::upsample_bilinear2d_backward(grad_output, output_size, input_size, align_corners, scales_h, scales_w);
    }
}
at::Tensor upsample_linear1d(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("upsample_linear1d exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::upsample_linear1d(self, output_size, align_corners, scales);
    } else {
        return acl_op::upsample_linear1d(self, output_size, align_corners, scales);
    }
}
at::Tensor upsample_linear1d_backward(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);

    ASCEND_LOGI("upsample_linear1d_backward exec with jit compile: %d, grad_output is internal format: %d",
                !is_jit_disable, !grad_output_base_format);
    if (is_jit_disable && grad_output_base_format) {
        return op_api::upsample_linear1d_backward(grad_output, output_size, input_size, align_corners, scales);
    } else {
        return acl_op::upsample_linear1d_backward(grad_output, output_size, input_size, align_corners, scales);
    }
}
at::Tensor upsample_nearest1d(const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("upsample_nearest1d exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::upsample_nearest1d(self, output_size, scales);
    } else {
        return acl_op::upsample_nearest1d(self, output_size, scales);
    }
}
at::Tensor upsample_nearest1d_backward(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);

    ASCEND_LOGI("upsample_nearest1d_backward exec with jit compile: %d, grad_output is internal format: %d",
                !is_jit_disable, !grad_output_base_format);
    if (is_jit_disable && grad_output_base_format) {
        return op_api::upsample_nearest1d_backward(grad_output, output_size, input_size, scales);
    } else {
        return acl_op::upsample_nearest1d_backward(grad_output, output_size, input_size, scales);
    }
}
at::Tensor upsample_nearest2d(const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("upsample_nearest2d exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::upsample_nearest2d(self, output_size, scales_h, scales_w);
    } else {
        return acl_op::upsample_nearest2d(self, output_size, scales_h, scales_w);
    }
}
at::Tensor upsample_nearest2d_backward(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales_h, c10::optional<double> scales_w){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);

    ASCEND_LOGI("upsample_nearest2d_backward exec with jit compile: %d, grad_output is internal format: %d",
                !is_jit_disable, !grad_output_base_format);
    if (is_jit_disable && grad_output_base_format) {
        return op_api::upsample_nearest2d_backward(grad_output, output_size, input_size, scales_h, scales_w);
    } else {
        return acl_op::upsample_nearest2d_backward(grad_output, output_size, input_size, scales_h, scales_w);
    }
}
at::Tensor upsample_nearest3d(const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("upsample_nearest3d exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::upsample_nearest3d(self, output_size, scales_d, scales_h, scales_w);
    } else {
        return acl_op::upsample_nearest3d(self, output_size, scales_d, scales_h, scales_w);
    }
}
at::Tensor upsample_nearest3d_backward(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);

    ASCEND_LOGI("upsample_nearest3d_backward exec with jit compile: %d, grad_output is internal format: %d",
                !is_jit_disable, !grad_output_base_format);
    if (is_jit_disable && grad_output_base_format) {
        return op_api::upsample_nearest3d_backward(grad_output, output_size, input_size, scales_d, scales_h, scales_w);
    } else {
        return acl_op::upsample_nearest3d_backward(grad_output, output_size, input_size, scales_d, scales_h, scales_w);
    }
}
at::Tensor upsample_trilinear3d(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("upsample_trilinear3d exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::upsample_trilinear3d(self, output_size, align_corners, scales_d, scales_h, scales_w);
    } else {
        return acl_op::upsample_trilinear3d(self, output_size, align_corners, scales_d, scales_h, scales_w);
    }
}
at::Tensor upsample_trilinear3d_backward(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool grad_output_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_output);

    ASCEND_LOGI("upsample_trilinear3d_backward exec with jit compile: %d, grad_output is internal format: %d",
                !is_jit_disable, !grad_output_base_format);
    if (is_jit_disable && grad_output_base_format) {
        return op_api::upsample_trilinear3d_backward(grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w);
    } else {
        return acl_op::upsample_trilinear3d_backward(grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w);
    }
}
at::Tensor var(const at::Tensor & self, at::OptionalIntArrayRef dim, const c10::optional<at::Scalar> & correction, bool keepdim){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("var exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::var(self, dim, correction, keepdim);
    } else {
        return acl_op::var(self, dim, correction, keepdim);
    }
}
at::Tensor vdot(const at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("vdot exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::vdot(self, other);
    } else {
        return acl_op::vdot(self, other);
    }
}
at::Tensor view_as_complex(const at::Tensor & self){
    return acl_op::view_as_complex(self);
}
at::Tensor view_as_real(const at::Tensor & self){
    return acl_op::view_as_real(self);
}
at::Tensor where(const at::Tensor & condition, const at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool condition_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(condition);
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("where exec with jit compile: %d, condition is internal format: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !condition_base_format, !self_base_format, !other_base_format);
    if (is_jit_disable && condition_base_format && self_base_format && other_base_format) {
        return op_api::where(condition, self, other);
    } else {
        return acl_op::where(condition, self, other);
    }
}
at::Tensor xlogy(const at::Scalar & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("xlogy exec with jit compile: %d, other is internal format: %d",
                !is_jit_disable, !other_base_format);
    if (is_jit_disable && other_base_format) {
        return op_api::xlogy(self, other);
    } else {
        return acl_op::xlogy(self, other);
    }
}
at::Tensor xlogy(const at::Tensor & self, const at::Scalar & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("xlogy exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::xlogy(self, other);
    } else {
        return acl_op::xlogy(self, other);
    }
}
at::Tensor xlogy(const at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("xlogy exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::xlogy(self, other);
    } else {
        return acl_op::xlogy(self, other);
    }
}
at::Tensor zeros(at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();

    ASCEND_LOGI("zeros exec with jit compile: %d",
                !is_jit_disable);
    if (is_jit_disable) {
        return op_api::zeros(size, names, dtype, layout, device, pin_memory);
    } else {
        return acl_op::zeros(size, names, dtype, layout, device, pin_memory);
    }
}
at::Tensor zeros_like(const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    ASCEND_LOGI("zeros_like exec with jit compile: %d, self is internal format: %d",
                !is_jit_disable, !self_base_format);
    if (is_jit_disable && self_base_format) {
        return op_api::zeros_like(self, dtype, layout, device, pin_memory, memory_format);
    } else {
        return acl_op::zeros_like(self, dtype, layout, device, pin_memory, memory_format);
    }
}
at::Tensor zeros_symint(c10::SymIntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();

    ASCEND_LOGI("zeros_symint exec with jit compile: %d",
                !is_jit_disable);
    if (is_jit_disable) {
        return op_api::zeros_symint(size, dtype, layout, device, pin_memory);
    } else {
        return acl_op::zeros_symint(size, dtype, layout, device, pin_memory);
    }
}
bool _amp_foreach_non_finite_check(at::TensorList scaled_grads){
    return acl_op::_amp_foreach_non_finite_check(scaled_grads);
}
bool equal(const at::Tensor & self, const at::Tensor & other){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    ASCEND_LOGI("equal exec with jit compile: %d, self is internal format: %d, other is internal format: %d",
                !is_jit_disable, !self_base_format, !other_base_format);
    if (is_jit_disable && self_base_format && other_base_format) {
        return op_api::equal(self, other);
    } else {
        return acl_op::equal(self, other);
    }
}
const at::Tensor & _conv_depthwise2d_out(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, const at::Tensor & out){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool weight_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
    bool bias_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(bias);
    bool out_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(out);

    ASCEND_LOGI("_conv_depthwise2d_out exec with jit compile: %d, self is internal format: %d, weight is internal format: %d, bias is internal format: %d, out is internal format: %d",
                !is_jit_disable, !self_base_format, !weight_base_format, !bias_base_format, !out_base_format);
    if (is_jit_disable && self_base_format && weight_base_format && bias_base_format && out_base_format) {
        return op_api::_conv_depthwise2d_out(self, weight, kernel_size, bias, stride, padding, dilation, out);
    } else {
        return acl_op::_conv_depthwise2d_out(self, weight, kernel_size, bias, stride, padding, dilation, out);
    }
}
void _amp_foreach_non_finite_check_and_unscale_(at::TensorList self, at::Tensor & found_inf, const at::Tensor & inv_scale){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool found_inf_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(found_inf);
    bool inv_scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(inv_scale);

    ASCEND_LOGI("_amp_foreach_non_finite_check_and_unscale_ exec with jit compile: %d, self is internal format: %d, found_inf is internal format: %d, inv_scale is internal format: %d",
                !is_jit_disable, !self_base_format, !found_inf_base_format, !inv_scale_base_format);
    if (is_jit_disable && self_base_format && found_inf_base_format && inv_scale_base_format) {
        return op_api::_amp_foreach_non_finite_check_and_unscale_(self, found_inf, inv_scale);
    } else {
        return acl_op::_amp_foreach_non_finite_check_and_unscale_(self, found_inf, inv_scale);
    }
}
void _cummax_helper(const at::Tensor & self, at::Tensor & values, at::Tensor & indices, int64_t dim){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool values_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(values);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);

    ASCEND_LOGI("_cummax_helper exec with jit compile: %d, self is internal format: %d, values is internal format: %d, indices is internal format: %d",
                !is_jit_disable, !self_base_format, !values_base_format, !indices_base_format);
    if (is_jit_disable && self_base_format && values_base_format && indices_base_format) {
        return op_api::_cummax_helper(self, values, indices, dim);
    } else {
        return acl_op::_cummax_helper(self, values, indices, dim);
    }
}
void _cummin_helper(const at::Tensor & self, at::Tensor & values, at::Tensor & indices, int64_t dim){
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool values_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(values);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);

    ASCEND_LOGI("_cummin_helper exec with jit compile: %d, self is internal format: %d, values is internal format: %d, indices is internal format: %d",
                !is_jit_disable, !self_base_format, !values_base_format, !indices_base_format);
    if (is_jit_disable && self_base_format && values_base_format && indices_base_format) {
        return op_api::_cummin_helper(self, values, indices, dim);
    } else {
        return acl_op::_cummin_helper(self, values, indices, dim);
    }
}
void _foreach_abs_(at::TensorList self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_abs_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_abs_(self);
}
void _foreach_acos_(at::TensorList self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_acos_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_acos_(self);
}
void _foreach_add_(at::TensorList self, at::ArrayRef<at::Scalar> scalars){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_add_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_add_(self, scalars);
}
void _foreach_add_(at::TensorList self, at::TensorList other, const at::Scalar & alpha){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    if (!self_base_format || !other_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_add_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_add_(self, other, alpha);
}
void _foreach_add_(at::TensorList self, const at::Scalar & scalar){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_add_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_add_(self, scalar);
}
void _foreach_addcdiv_(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool tensor1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor1);
    bool tensor2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor2);

    if (!self_base_format || !tensor1_base_format || !tensor2_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_addcdiv_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_addcdiv_(self, tensor1, tensor2, scalars);
}
void _foreach_addcdiv_(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool tensor1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor1);
    bool tensor2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor2);

    if (!self_base_format || !tensor1_base_format || !tensor2_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_addcdiv_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_addcdiv_(self, tensor1, tensor2, value);
}
void _foreach_addcdiv_(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Tensor & scalars){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool tensor1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor1);
    bool tensor2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor2);
    bool scalars_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(scalars);

    if (!self_base_format || !tensor1_base_format || !tensor2_base_format || !scalars_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_addcdiv_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_addcdiv_(self, tensor1, tensor2, scalars);
}
void _foreach_addcmul_(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool tensor1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor1);
    bool tensor2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor2);

    if (!self_base_format || !tensor1_base_format || !tensor2_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_addcmul_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_addcmul_(self, tensor1, tensor2, scalars);
}
void _foreach_addcmul_(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool tensor1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor1);
    bool tensor2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor2);

    if (!self_base_format || !tensor1_base_format || !tensor2_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_addcmul_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_addcmul_(self, tensor1, tensor2, value);
}
void _foreach_addcmul_(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Tensor & scalars){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool tensor1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor1);
    bool tensor2_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor2);
    bool scalars_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(scalars);

    if (!self_base_format || !tensor1_base_format || !tensor2_base_format || !scalars_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_addcmul_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_addcmul_(self, tensor1, tensor2, scalars);
}
void _foreach_asin_(at::TensorList self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_asin_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_asin_(self);
}
void _foreach_atan_(at::TensorList self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_atan_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_atan_(self);
}
void _foreach_ceil_(at::TensorList self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_ceil_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_ceil_(self);
}
void _foreach_clamp_max_(at::TensorList self, at::ArrayRef<at::Scalar> scalars){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_clamp_max_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_clamp_max_(self, scalars);
}
void _foreach_clamp_max_(at::TensorList self, at::TensorList other){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    if (!self_base_format || !other_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_clamp_max_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_clamp_max_(self, other);
}
void _foreach_clamp_max_(at::TensorList self, const at::Scalar & scalar){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_clamp_max_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_clamp_max_(self, scalar);
}
void _foreach_clamp_min_(at::TensorList self, at::ArrayRef<at::Scalar> scalars){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_clamp_min_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_clamp_min_(self, scalars);
}
void _foreach_clamp_min_(at::TensorList self, at::TensorList other){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    if (!self_base_format || !other_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_clamp_min_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_clamp_min_(self, other);
}
void _foreach_clamp_min_(at::TensorList self, const at::Scalar & scalar){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_clamp_min_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_clamp_min_(self, scalar);
}
void _foreach_copy_(at::TensorList self, at::TensorList src, bool non_blocking){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool src_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(src);

    if (!self_base_format || !src_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_copy_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_copy_(self, src, non_blocking);
}
void _foreach_cos_(at::TensorList self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_cos_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_cos_(self);
}
void _foreach_cosh_(at::TensorList self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_cosh_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_cosh_(self);
}
void _foreach_div_(at::TensorList self, at::ArrayRef<at::Scalar> scalars){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_div_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_div_(self, scalars);
}
void _foreach_div_(at::TensorList self, at::TensorList other){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    if (!self_base_format || !other_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_div_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_div_(self, other);
}
void _foreach_div_(at::TensorList self, const at::Scalar & scalar){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_div_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_div_(self, scalar);
}
void _foreach_erf_(at::TensorList self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_erf_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_erf_(self);
}
void _foreach_erfc_(at::TensorList self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_erfc_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_erfc_(self);
}
void _foreach_exp_(at::TensorList self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_exp_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_exp_(self);
}
void _foreach_expm1_(at::TensorList self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_expm1_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_expm1_(self);
}
void _foreach_floor_(at::TensorList self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_floor_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_floor_(self);
}
void _foreach_frac_(at::TensorList self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_frac_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_frac_(self);
}
void _foreach_lerp_(at::TensorList self, at::TensorList tensors1, at::TensorList weights){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool tensors1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors1);
    bool weights_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(weights);

    if (!self_base_format || !tensors1_base_format || !weights_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_lerp_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_lerp_(self, tensors1, weights);
}
void _foreach_lerp_(at::TensorList self, at::TensorList tensors1, const at::Scalar & weight){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool tensors1_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(tensors1);

    if (!self_base_format || !tensors1_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_lerp_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_lerp_(self, tensors1, weight);
}
void _foreach_log10_(at::TensorList self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_log10_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_log10_(self);
}
void _foreach_log1p_(at::TensorList self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_log1p_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_log1p_(self);
}
void _foreach_log2_(at::TensorList self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_log2_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_log2_(self);
}
void _foreach_log_(at::TensorList self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_log_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_log_(self);
}
void _foreach_maximum_(at::TensorList self, at::ArrayRef<at::Scalar> scalars){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_maximum_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_maximum_(self, scalars);
}
void _foreach_maximum_(at::TensorList self, at::TensorList other){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    if (!self_base_format || !other_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_maximum_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_maximum_(self, other);
}
void _foreach_maximum_(at::TensorList self, const at::Scalar & scalar){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_maximum_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_maximum_(self, scalar);
}
void _foreach_minimum_(at::TensorList self, at::ArrayRef<at::Scalar> scalars){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_minimum_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_minimum_(self, scalars);
}
void _foreach_minimum_(at::TensorList self, at::TensorList other){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    if (!self_base_format || !other_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_minimum_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_minimum_(self, other);
}
void _foreach_minimum_(at::TensorList self, const at::Scalar & scalar){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_minimum_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_minimum_(self, scalar);
}
void _foreach_mul_(at::TensorList self, at::ArrayRef<at::Scalar> scalars){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_mul_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_mul_(self, scalars);
}
void _foreach_mul_(at::TensorList self, at::TensorList other){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    if (!self_base_format || !other_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_mul_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_mul_(self, other);
}
void _foreach_mul_(at::TensorList self, const at::Scalar & scalar){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_mul_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_mul_(self, scalar);
}
void _foreach_neg_(at::TensorList self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_neg_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_neg_(self);
}
void _foreach_pow_(at::TensorList self, at::ArrayRef<at::Scalar> exponent){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_pow_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_pow_(self, exponent);
}
void _foreach_pow_(at::TensorList self, at::TensorList exponent){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool exponent_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(exponent);

    if (!self_base_format || !exponent_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_pow_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_pow_(self, exponent);
}
void _foreach_pow_(at::TensorList self, const at::Scalar & scalar){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_pow_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_pow_(self, scalar);
}
void _foreach_reciprocal_(at::TensorList self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_reciprocal_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_reciprocal_(self);
}
void _foreach_round_(at::TensorList self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_round_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_round_(self);
}
void _foreach_sigmoid_(at::TensorList self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_sigmoid_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_sigmoid_(self);
}
void _foreach_sign_(at::TensorList self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_sign_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_sign_(self);
}
void _foreach_sin_(at::TensorList self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_sin_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_sin_(self);
}
void _foreach_sinh_(at::TensorList self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_sinh_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_sinh_(self);
}
void _foreach_sqrt_(at::TensorList self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_sqrt_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_sqrt_(self);
}
void _foreach_sub_(at::TensorList self, at::ArrayRef<at::Scalar> scalars){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_sub_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_sub_(self, scalars);
}
void _foreach_sub_(at::TensorList self, at::TensorList other, const at::Scalar & alpha){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool other_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(other);

    if (!self_base_format || !other_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_sub_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_sub_(self, other, alpha);
}
void _foreach_sub_(at::TensorList self, const at::Scalar & scalar){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_sub_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_sub_(self, scalar);
}
void _foreach_tan_(at::TensorList self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_tan_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_tan_(self);
}
void _foreach_tanh_(at::TensorList self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_tanh_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_tanh_(self);
}
void _foreach_trunc_(at::TensorList self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_trunc_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_trunc_(self);
}
void _foreach_zero_(at::TensorList self){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);

    if (!self_base_format) {
        TORCH_CHECK(false,
            "Current operator _foreach_zero_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_foreach_zero_(self);
}
void _fused_adamw_(at::TensorList self, at::TensorList grads, at::TensorList exp_avgs, at::TensorList exp_avg_sqs, at::TensorList max_exp_avg_sqs, at::TensorList state_steps, double lr, double beta1, double beta2, double weight_decay, double eps, bool amsgrad, bool maximize, const c10::optional<at::Tensor> & grad_scale, const c10::optional<at::Tensor> & found_inf){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool grads_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grads);
    bool exp_avgs_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(exp_avgs);
    bool exp_avg_sqs_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(exp_avg_sqs);
    bool max_exp_avg_sqs_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(max_exp_avg_sqs);
    bool state_steps_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(state_steps);
    bool grad_scale_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(grad_scale);
    bool found_inf_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(found_inf);

    if (!self_base_format || !grads_base_format || !exp_avgs_base_format || !exp_avg_sqs_base_format || !max_exp_avg_sqs_base_format || !state_steps_base_format || !grad_scale_base_format || !found_inf_base_format) {
        TORCH_CHECK(false,
            "Current operator _fused_adamw_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::_fused_adamw_(self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize, grad_scale, found_inf);
}
void npu_advance_step_flashattn(at::Tensor & input_tokens, const at::Tensor & sampled_token_ids, at::Tensor & input_positions, at::Tensor & seq_lens, at::Tensor & slot_mapping, const at::Tensor & block_tables, int64_t num_seqs, int64_t num_queries, int64_t block_size){
    bool input_tokens_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input_tokens);
    bool sampled_token_ids_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(sampled_token_ids);
    bool input_positions_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(input_positions);
    bool seq_lens_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(seq_lens);
    bool slot_mapping_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(slot_mapping);
    bool block_tables_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(block_tables);

    if (!input_tokens_base_format || !sampled_token_ids_base_format || !input_positions_base_format || !seq_lens_base_format || !slot_mapping_base_format || !block_tables_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_advance_step_flashattn do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_advance_step_flashattn(input_tokens, sampled_token_ids, input_positions, seq_lens, slot_mapping, block_tables, num_seqs, num_queries, block_size);
}
void npu_prefetch(const at::Tensor & self, const c10::optional<at::Tensor> & dependency, int64_t max_size, int64_t offset){
    return op_api::npu_prefetch(self, dependency, max_size, offset);
}
void npu_scatter_list_(at::TensorList self, const at::Tensor & indices, const at::Tensor & updates, const c10::optional<at::Tensor> & mask, c10::string_view reduce, int64_t axis){
    bool self_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(self);
    bool indices_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(indices);
    bool updates_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(updates);
    bool mask_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat(mask);

    if (!self_base_format || !indices_base_format || !updates_base_format || !mask_base_format) {
        TORCH_CHECK(false,
            "Current operator npu_scatter_list_ do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return op_api::npu_scatter_list_(self, indices, updates, mask, reduce, axis);
}
}  // namespace acl_op
