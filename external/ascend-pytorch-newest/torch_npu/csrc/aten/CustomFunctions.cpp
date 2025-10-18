#include <ATen/core/dispatch/Dispatcher.h>

#include "torch_npu/csrc/aten/CustomFunctions.h"


namespace at_npu {
namespace native {
namespace custom_ops {

int64_t npu_change_data_ptr(const at::Tensor & dst, const at::Tensor & src, int64_t index) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_change_data_ptr", "").typed<int64_t (const at::Tensor &, const at::Tensor &, int64_t)>();
    return op.call(dst, src, index);
}
int64_t get_npu_format(const at::Tensor & self) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::get_npu_format", "").typed<int64_t (const at::Tensor &)>();
    return op.call(self);
}
at::Tensor npu_format_cast(const at::Tensor & self, const at::Tensor & dst) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_format_cast", "Tensor").typed<at::Tensor (const at::Tensor &, const at::Tensor &)>();
    return op.call(self, dst);
}
at::Tensor & npu_format_cast_(at::Tensor & self, int64_t acl_format) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_format_cast_", "acl_format").typed<at::Tensor & (at::Tensor &, int64_t)>();
    return op.call(self, acl_format);
}
at::Tensor & npu_format_cast_(at::Tensor & self, const at::Tensor & src) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_format_cast_", "").typed<at::Tensor & (at::Tensor &, const at::Tensor &)>();
    return op.call(self, src);
}
at::Tensor empty_with_format(at::IntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, int64_t acl_format, ::std::optional<int64_t> base_addr_aligned_kb) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::empty_with_format", "").typed<at::Tensor (at::IntArrayRef, ::std::optional<at::ScalarType>, ::std::optional<at::Layout>, ::std::optional<at::Device>, ::std::optional<bool>, int64_t, ::std::optional<int64_t>)>();
    return op.call(size, dtype, layout, device, pin_memory, acl_format, base_addr_aligned_kb);
}
at::Tensor unsafe_empty_with_format(at::IntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, int64_t acl_format, bool keep_format) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::unsafe_empty_with_format", "").typed<at::Tensor (at::IntArrayRef, ::std::optional<at::ScalarType>, ::std::optional<at::Layout>, ::std::optional<at::Device>, ::std::optional<bool>, int64_t, bool)>();
    return op.call(size, dtype, layout, device, pin_memory, acl_format, keep_format);
}
at::Tensor empty_with_format(at::IntArrayRef size, ::std::optional<at::DimnameList> names, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, int64_t acl_format) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::empty_with_format", "names").typed<at::Tensor (at::IntArrayRef, ::std::optional<at::DimnameList>, ::std::optional<at::ScalarType>, ::std::optional<at::Layout>, ::std::optional<at::Device>, ::std::optional<bool>, int64_t)>();
    return op.call(size, names, dtype, layout, device, pin_memory, acl_format);
}
at::Tensor & copy_memory_(at::Tensor & self, const at::Tensor & src, bool non_blocking) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::copy_memory_", "").typed<at::Tensor & (at::Tensor &, const at::Tensor &, bool)>();
    return op.call(self, src, non_blocking);
}
int64_t get_storage_size(const at::Tensor & self) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::get_storage_size", "").typed<int64_t (const at::Tensor &)>();
    return op.call(self);
}
at::Tensor npu_format_cast(const at::Tensor & self, int64_t acl_format) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_format_cast", "").typed<at::Tensor (const at::Tensor &, int64_t)>();
    return op.call(self, acl_format);
}
at::Tensor _npu_format_cast(const at::Tensor & self, int64_t acl_format) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::_npu_format_cast", "").typed<at::Tensor (const at::Tensor &, int64_t)>();
    return op.call(self, acl_format);
}
at::Tensor empty_with_swapped_memory(at::IntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Device> device) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::empty_with_swapped_memory", "").typed<at::Tensor (at::IntArrayRef, ::std::optional<at::ScalarType>, ::std::optional<at::Device>)>();
    return op.call(size, dtype, device);
}
at::Tensor npu_gather_backward(const at::Tensor & grad, c10::SymIntArrayRef self_size, int64_t dim, const at::Tensor & index, bool sparse_grad) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_gather_backward", "").typed<at::Tensor (const at::Tensor &, c10::SymIntArrayRef, int64_t, const at::Tensor &, bool)>();
    return op.call(grad, self_size, dim, index, sparse_grad);
}
bool _amp_foreach_non_finite_check(at::TensorList scaled_grads) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::_amp_foreach_non_finite_check", "").typed<bool (at::TensorList)>();
    return op.call(scaled_grads);
}
at::Tensor npu_gelu(const at::Tensor & self, c10::string_view approximate) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_gelu", "").typed<at::Tensor (const at::Tensor &, c10::string_view)>();
    return op.call(self, approximate);
}
at::Tensor npu_gelu_backward(const at::Tensor & grad_output, const at::Tensor & self, c10::string_view approximate) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_gelu_backward", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, c10::string_view)>();
    return op.call(grad_output, self, approximate);
}
::std::tuple<at::Tensor,at::Tensor> _conv_depthwise2d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, ::std::array<bool,2> output_mask) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::_conv_depthwise2d_backward", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, ::std::array<bool,2>)>();
    return op.call(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
}
::std::tuple<at::Tensor,at::Tensor> _dropout_with_byte_mask(const at::Tensor & self, double p) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::_dropout_with_byte_mask", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, double)>();
    return op.call(self, p);
}
at::Tensor _dropout_with_byte_mask_backward(const at::Tensor & grad_output, const at::Tensor & mask, double p) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::_dropout_with_byte_mask_backward", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, double)>();
    return op.call(grad_output, mask, p);
}
::std::tuple<at::Tensor,at::Tensor> _npu_ciou(const at::Tensor & self, const at::Tensor & gtboxes, bool trans, bool is_cross, int64_t mode, bool atan_sub_flag) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::_npu_ciou", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, bool, bool, int64_t, bool)>();
    return op.call(self, gtboxes, trans, is_cross, mode, atan_sub_flag);
}
::std::tuple<at::Tensor,at::Tensor> _npu_dropout(const at::Tensor & self, double p) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::_npu_dropout", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, double)>();
    return op.call(self, p);
}
at::Tensor _npu_dropout_gen_mask(const at::Tensor & self, at::IntArrayRef size, double p, int64_t seed, int64_t offset, ::std::optional<bool> parallel, ::std::optional<bool> sync) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::_npu_dropout_gen_mask", "Tensor").typed<at::Tensor (const at::Tensor &, at::IntArrayRef, double, int64_t, int64_t, ::std::optional<bool>, ::std::optional<bool>)>();
    return op.call(self, size, p, seed, offset, parallel, sync);
}
at::Tensor _npu_silent_check(at::Tensor & input_grad, const at::Tensor & val, at::Tensor & pre_val, at::Tensor & min_val, at::Tensor & max_val, const at::Tensor & val_counter, int64_t c_min_steps, double c_thresh_l1, double c_coeff_l1, double c_thresh_l2, double c_coeff_l2) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::_npu_silent_check", "").typed<at::Tensor (at::Tensor &, const at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, const at::Tensor &, int64_t, double, double, double, double)>();
    return op.call(input_grad, val, pre_val, min_val, max_val, val_counter, c_min_steps, c_thresh_l1, c_coeff_l1, c_thresh_l2, c_coeff_l2);
}
at::Tensor _npu_silent_check_v2(const at::Tensor & val, at::Tensor & input_grad, at::Tensor & sfda, at::Tensor & step, int64_t c_min_steps, double c_thresh_l1, double c_coeff_l1, double c_thresh_l2, double c_coeff_l2, int64_t npu_asd_detect) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::_npu_silent_check_v2", "").typed<at::Tensor (const at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, int64_t, double, double, double, double, int64_t)>();
    return op.call(val, input_grad, sfda, step, c_min_steps, c_thresh_l1, c_coeff_l1, c_thresh_l2, c_coeff_l2, npu_asd_detect);
}
at::Tensor _npu_silent_check_v3(const at::Tensor & val, at::Tensor & input_grad, at::Tensor & step, at::Tensor & max, at::Tensor & avg, double c_thresh_l1, double c_thresh_l2, double betal, int64_t npu_asd_detect) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::_npu_silent_check_v3", "").typed<at::Tensor (const at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, double, double, double, int64_t)>();
    return op.call(val, input_grad, step, max, avg, c_thresh_l1, c_thresh_l2, betal, npu_asd_detect);
}
::std::tuple<at::Tensor,at::Tensor> batch_norm_gather_stats_update(const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, double momentum, double eps, const at::Tensor & counts) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::batch_norm_gather_stats_update", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, double, double, const at::Tensor &)>();
    return op.call(input, mean, invstd, running_mean, running_var, momentum, eps, counts);
}
::std::tuple<at::Tensor,at::Tensor> batch_norm_reduce(const at::Tensor & input, double eps) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::batch_norm_reduce", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, double)>();
    return op.call(input, eps);
}
at::Tensor dropout_with_byte_mask(const at::Tensor & self, double p, bool train) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::dropout_with_byte_mask", "").typed<at::Tensor (const at::Tensor &, double, bool)>();
    return op.call(self, p, train);
}
at::Tensor fast_gelu(const at::Tensor & self) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::fast_gelu", "").typed<at::Tensor (const at::Tensor &)>();
    return op.call(self);
}
at::Tensor kl_div_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, bool log_target) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::kl_div_backward", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, bool)>();
    return op.call(grad_output, self, target, reduction, log_target);
}
at::Tensor l1_loss_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::l1_loss_backward", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t)>();
    return op.call(grad_output, self, target, reduction);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> slow_conv_dilated2d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, ::std::array<bool,3> output_mask) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::slow_conv_dilated2d_backward", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, ::std::array<bool,3>)>();
    return op.call(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> matmul_double_backward(const ::std::optional<at::Tensor> & grad_self, const ::std::optional<at::Tensor> & grad_other, const at::Tensor & grad_out, const at::Tensor & self, const at::Tensor & other, ::std::array<bool,3> mask) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::matmul_double_backward", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const at::Tensor &, const at::Tensor &, const at::Tensor &, ::std::array<bool,3>)>();
    return op.call(grad_self, grad_other, grad_out, self, other, mask);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_add_layer_norm(const at::Tensor & x1, const at::Tensor & x2, const at::Tensor & gamma, const at::Tensor & beta, double epsilon, bool additional_output) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_add_layer_norm", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, double, bool)>();
    return op.call(x1, x2, gamma, beta, epsilon, additional_output);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_add_layer_norm_backward(const ::std::optional<at::Tensor> & dy_opt, const at::Tensor & x1, const at::Tensor & x2, const at::Tensor & rstd, const at::Tensor & mean, const at::Tensor & gamma, const ::std::optional<at::Tensor> & dsum_opt) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_add_layer_norm_backward", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const ::std::optional<at::Tensor> &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &)>();
    return op.call(dy_opt, x1, x2, rstd, mean, gamma, dsum_opt);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_add_rms_norm(const at::Tensor & x1, const at::Tensor & x2, const at::Tensor & gamma, double epsilon) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_add_rms_norm", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, double)>();
    return op.call(x1, x2, gamma, epsilon);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_add_rms_norm_cast(const at::Tensor & x1, const at::Tensor & x2, const at::Tensor & gamma, double epsilon) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_add_rms_norm_cast", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, double)>();
    return op.call(x1, x2, gamma, epsilon);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_add_rms_norm_quant(const at::Tensor & x1, const at::Tensor & x2, const at::Tensor & gamma, const at::Tensor & scales1, const ::std::optional<at::Tensor> & zero_points1, const ::std::optional<at::Tensor> & scales2, const ::std::optional<at::Tensor> & zero_points2, int64_t axis, double epsilon, bool div_mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_add_rms_norm_quant", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, int64_t, double, bool)>();
    return op.call(x1, x2, gamma, scales1, zero_points1, scales2, zero_points2, axis, epsilon, div_mode);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_alltoallv_gmm(const at::Tensor & gmm_x, const at::Tensor & gmm_weight, c10::string_view hcom, int64_t ep_world_size, at::IntArrayRef send_counts, at::IntArrayRef recv_counts, const ::std::optional<at::Tensor> & send_counts_tensor, const ::std::optional<at::Tensor> & recv_counts_tensor, const ::std::optional<at::Tensor> & mm_x, const ::std::optional<at::Tensor> & mm_weight, bool trans_gmm_weight, bool trans_mm_weight, bool permute_out_flag) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_alltoallv_gmm", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, c10::string_view, int64_t, at::IntArrayRef, at::IntArrayRef, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, bool, bool, bool)>();
    return op.call(gmm_x, gmm_weight, hcom, ep_world_size, send_counts, recv_counts, send_counts_tensor, recv_counts_tensor, mm_x, mm_weight, trans_gmm_weight, trans_mm_weight, permute_out_flag);
}
::std::tuple<at::Tensor,at::Tensor> npu_gmm_alltoallv(const at::Tensor & gmm_x, const at::Tensor & gmm_weight, c10::string_view hcom, int64_t ep_world_size, at::IntArrayRef send_counts, at::IntArrayRef recv_counts, const ::std::optional<at::Tensor> & send_counts_tensor, const ::std::optional<at::Tensor> & recv_counts_tensor, const ::std::optional<at::Tensor> & mm_x, const ::std::optional<at::Tensor> & mm_weight, bool trans_gmm_weight, bool trans_mm_weight) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_gmm_alltoallv", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, c10::string_view, int64_t, at::IntArrayRef, at::IntArrayRef, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, bool, bool)>();
    return op.call(gmm_x, gmm_weight, hcom, ep_world_size, send_counts, recv_counts, send_counts_tensor, recv_counts_tensor, mm_x, mm_weight, trans_gmm_weight, trans_mm_weight);
}
::std::tuple<at::Tensor,at::Tensor> npu_all_gather_base_mm(const at::Tensor & self, const at::Tensor & x2, c10::string_view hcom, int64_t world_size, const ::std::optional<at::Tensor> & bias, int64_t gather_index, bool gather_output, int64_t comm_turn) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_all_gather_base_mm", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, c10::string_view, int64_t, const ::std::optional<at::Tensor> &, int64_t, bool, int64_t)>();
    return op.call(self, x2, hcom, world_size, bias, gather_index, gather_output, comm_turn);
}
at::Tensor npu_alloc_float_status(const at::Tensor & self) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_alloc_float_status", "").typed<at::Tensor (const at::Tensor &)>();
    return op.call(self);
}
at::Tensor npu_anchor_response_flags(const at::Tensor & self, at::IntArrayRef featmap_size, at::IntArrayRef stride, int64_t num_base_anchors) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_anchor_response_flags", "").typed<at::Tensor (const at::Tensor &, at::IntArrayRef, at::IntArrayRef, int64_t)>();
    return op.call(self, featmap_size, stride, num_base_anchors);
}
at::Tensor npu_anti_quant(const at::Tensor & x, const at::Tensor & scale, const ::std::optional<at::Tensor> & offset, ::std::optional<at::ScalarType> dst_dtype, ::std::optional<at::ScalarType> src_dtype) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_anti_quant", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, ::std::optional<at::ScalarType>, ::std::optional<at::ScalarType>)>();
    return op.call(x, scale, offset, dst_dtype, src_dtype);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_apply_adam(const at::Scalar & beta1_power, const at::Scalar & beta2_power, const at::Scalar & lr, const at::Scalar & beta1, const at::Scalar & beta2, const at::Scalar & epsilon, const at::Tensor & grad, ::std::optional<bool> use_locking, ::std::optional<bool> use_nesterov) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_apply_adam", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Tensor &, ::std::optional<bool>, ::std::optional<bool>)>();
    return op.call(beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, use_locking, use_nesterov);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> npu_apply_adam_out(const at::Scalar & beta1_power, const at::Scalar & beta2_power, const at::Scalar & lr, const at::Scalar & beta1, const at::Scalar & beta2, const at::Scalar & epsilon, const at::Tensor & grad, ::std::optional<bool> use_locking, ::std::optional<bool> use_nesterov, at::Tensor & var, at::Tensor & m, at::Tensor & v) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_apply_adam", "out").typed<::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> (const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Tensor &, ::std::optional<bool>, ::std::optional<bool>, at::Tensor &, at::Tensor &, at::Tensor &)>();
    return op.call(beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, use_locking, use_nesterov, var, m, v);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_apply_adam_w(const at::Scalar & beta1_power, const at::Scalar & beta2_power, const at::Scalar & lr, const at::Scalar & weight_decay, const at::Scalar & beta1, const at::Scalar & beta2, const at::Scalar & epsilon, const at::Tensor & grad, const ::std::optional<at::Tensor> & max_grad_norm, ::std::optional<bool> amsgrad, ::std::optional<bool> maximize) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_apply_adam_w", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Tensor &, const ::std::optional<at::Tensor> &, ::std::optional<bool>, ::std::optional<bool>)>();
    return op.call(beta1_power, beta2_power, lr, weight_decay, beta1, beta2, epsilon, grad, max_grad_norm, amsgrad, maximize);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> npu_apply_adam_w_out(const at::Scalar & beta1_power, const at::Scalar & beta2_power, const at::Scalar & lr, const at::Scalar & weight_decay, const at::Scalar & beta1, const at::Scalar & beta2, const at::Scalar & epsilon, const at::Tensor & grad, const ::std::optional<at::Tensor> & max_grad_norm, ::std::optional<bool> amsgrad, ::std::optional<bool> maximize, at::Tensor & var, at::Tensor & m, at::Tensor & v) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_apply_adam_w", "out").typed<::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> (const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Tensor &, const ::std::optional<at::Tensor> &, ::std::optional<bool>, ::std::optional<bool>, at::Tensor &, at::Tensor &, at::Tensor &)>();
    return op.call(beta1_power, beta2_power, lr, weight_decay, beta1, beta2, epsilon, grad, max_grad_norm, amsgrad, maximize, var, m, v);
}
::std::tuple<at::Tensor,at::Tensor> npu_apply_rotary_pos_emb(const at::Tensor & query, const at::Tensor & key, const at::Tensor & cos, const at::Tensor & sin, c10::string_view layout) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_apply_rotary_pos_emb", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, c10::string_view)>();
    return op.call(query, key, cos, sin, layout);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_kv_rmsnorm_rope_cache(const at::Tensor & kv, const at::Tensor & gamma, const at::Tensor & cos, const at::Tensor & sin, const at::Tensor & index, const at::Tensor & k_cache, const at::Tensor & ckv_cache, const ::std::optional<at::Tensor> & k_rope_scale, const ::std::optional<at::Tensor> & c_kv_scale, const ::std::optional<at::Tensor> & k_rope_offset, const ::std::optional<at::Tensor> & c_kv_offset, double epsilon, c10::string_view cache_mode, bool is_output_kv) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_kv_rmsnorm_rope_cache", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, double, c10::string_view, bool)>();
    return op.call(kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope_scale, c_kv_scale, k_rope_offset, c_kv_offset, epsilon, cache_mode, is_output_kv);
}
at::Tensor npu_batch_gather_matmul(const at::Tensor & self, const at::Tensor & x, const at::Tensor & weight_b, const at::Tensor & indices, const ::std::optional<at::Tensor> & weight_a, int64_t layer_idx, double scale, int64_t y_offset, int64_t y_slice_size) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_batch_gather_matmul", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, int64_t, double, int64_t, int64_t)>();
    return op.call(self, x, weight_b, indices, weight_a, layer_idx, scale, y_offset, y_slice_size);
}
at::Tensor & npu_batch_gather_matmul_(at::Tensor & self, const at::Tensor & x, const at::Tensor & weight_b, const at::Tensor & indices, const ::std::optional<at::Tensor> & weight_a, int64_t layer_idx, double scale, int64_t y_offset, int64_t y_slice_size) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_batch_gather_matmul_", "").typed<at::Tensor & (at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, int64_t, double, int64_t, int64_t)>();
    return op.call(self, x, weight_b, indices, weight_a, layer_idx, scale, y_offset, y_slice_size);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_batch_nms(const at::Tensor & self, const at::Tensor & scores, double score_threshold, double iou_threshold, int64_t max_size_per_class, int64_t max_total_size, bool change_coordinate_frame, bool transpose_box) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_batch_nms", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, double, double, int64_t, int64_t, bool, bool)>();
    return op.call(self, scores, score_threshold, iou_threshold, max_size_per_class, max_total_size, change_coordinate_frame, transpose_box);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_bert_apply_adam(const at::Scalar & lr, const at::Scalar & beta1, const at::Scalar & beta2, const at::Scalar & epsilon, const at::Tensor & grad, const at::Scalar & max_grad_norm, const at::Scalar & global_grad_norm, const at::Scalar & weight_decay, const ::std::optional<at::Scalar> & step_size, int64_t adam_mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_bert_apply_adam", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Tensor &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const ::std::optional<at::Scalar> &, int64_t)>();
    return op.call(lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay, step_size, adam_mode);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> npu_bert_apply_adam_out(const at::Scalar & lr, const at::Scalar & beta1, const at::Scalar & beta2, const at::Scalar & epsilon, const at::Tensor & grad, const at::Scalar & max_grad_norm, const at::Scalar & global_grad_norm, const at::Scalar & weight_decay, const ::std::optional<at::Scalar> & step_size, int64_t adam_mode, at::Tensor & var, at::Tensor & m, at::Tensor & v) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_bert_apply_adam", "out").typed<::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> (const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Tensor &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const ::std::optional<at::Scalar> &, int64_t, at::Tensor &, at::Tensor &, at::Tensor &)>();
    return op.call(lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay, step_size, adam_mode, var, m, v);
}
at::Tensor npu_binary_cross_entropy_with_logits_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight_opt, const ::std::optional<at::Tensor> & pos_weight_opt, int64_t reduction) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_binary_cross_entropy_with_logits_backward", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, int64_t)>();
    return op.call(grad_output, self, target, weight_opt, pos_weight_opt, reduction);
}
at::Tensor npu_bmmV2(const at::Tensor & self, const at::Tensor & mat2, at::IntArrayRef output_sizes) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_bmmV2", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, at::IntArrayRef)>();
    return op.call(self, mat2, output_sizes);
}
at::Tensor npu_bmm_v2_mat1_backward(const at::Tensor & grad, const at::Tensor & mat1, const at::Tensor & mat2, c10::SymIntArrayRef size) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_bmm_v2_mat1_backward", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, c10::SymIntArrayRef)>();
    return op.call(grad, mat1, mat2, size);
}
at::Tensor npu_bmm_v2_mat2_backward(const at::Tensor & grad, const at::Tensor & mat1, const at::Tensor & mat2, c10::SymIntArrayRef size) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_bmm_v2_mat2_backward", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, c10::SymIntArrayRef)>();
    return op.call(grad, mat1, mat2, size);
}
at::Tensor npu_bounding_box_decode(const at::Tensor & rois, const at::Tensor & deltas, double means0, double means1, double means2, double means3, double stds0, double stds1, double stds2, double stds3, at::IntArrayRef max_shape, double wh_ratio_clip) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_bounding_box_decode", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, double, double, double, double, double, double, double, double, at::IntArrayRef, double)>();
    return op.call(rois, deltas, means0, means1, means2, means3, stds0, stds1, stds2, stds3, max_shape, wh_ratio_clip);
}
at::Tensor npu_bounding_box_encode(const at::Tensor & anchor_box, const at::Tensor & ground_truth_box, double means0, double means1, double means2, double means3, double stds0, double stds1, double stds2, double stds3) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_bounding_box_encode", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, double, double, double, double, double, double, double, double)>();
    return op.call(anchor_box, ground_truth_box, means0, means1, means2, means3, stds0, stds1, stds2, stds3);
}
at::Tensor npu_broadcast(const at::Tensor & self, at::IntArrayRef size) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_broadcast", "").typed<at::Tensor (const at::Tensor &, at::IntArrayRef)>();
    return op.call(self, size);
}
at::Tensor & npu_broadcast_out(const at::Tensor & self, at::IntArrayRef size, at::Tensor & out) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_broadcast", "out").typed<at::Tensor & (const at::Tensor &, at::IntArrayRef, at::Tensor &)>();
    return op.call(self, size, out);
}
at::Tensor npu_ciou(const at::Tensor & self, const at::Tensor & gtboxes, bool trans, bool is_cross, int64_t mode, bool atan_sub_flag) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_ciou", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, bool, bool, int64_t, bool)>();
    return op.call(self, gtboxes, trans, is_cross, mode, atan_sub_flag);
}
::std::tuple<at::Tensor,at::Tensor> npu_ciou_backward(const at::Tensor & grad, const at::Tensor & bboxes, const at::Tensor & gtboxes, const ::std::optional<at::Tensor> & atan_sub, bool trans, bool is_cross, int64_t mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_ciou_backward", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, bool, bool, int64_t)>();
    return op.call(grad, bboxes, gtboxes, atan_sub, trans, is_cross, mode);
}
at::Tensor npu_clear_float_status(const at::Tensor & self, int64_t mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_clear_float_status", "").typed<at::Tensor (const at::Tensor &, int64_t)>();
    return op.call(self, mode);
}
at::Tensor npu_confusion_transpose(const at::Tensor & self, at::IntArrayRef perm, at::IntArrayRef shape, bool transpose_first) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_confusion_transpose", "").typed<at::Tensor (const at::Tensor &, at::IntArrayRef, at::IntArrayRef, bool)>();
    return op.call(self, perm, shape, transpose_first);
}
at::Tensor npu_confusion_transpose_backward(const at::Tensor & grad, at::IntArrayRef perm, c10::SymIntArrayRef shape, bool transpose_first) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_confusion_transpose_backward", "").typed<at::Tensor (const at::Tensor &, at::IntArrayRef, c10::SymIntArrayRef, bool)>();
    return op.call(grad, perm, shape, transpose_first);
}
at::Tensor npu_conv2d(const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_conv2d", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t)>();
    return op.call(input, weight, bias, stride, padding, dilation, groups);
}
at::Tensor & npu_conv2d_out(const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, at::Tensor & out) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_conv2d", "out").typed<at::Tensor & (const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, at::Tensor &)>();
    return op.call(input, weight, bias, stride, padding, dilation, groups, out);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_conv2d_backward(const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, ::std::array<bool,3> output_mask) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_conv2d_backward", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, ::std::array<bool,3>)>();
    return op.call(input, grad_output, weight, stride, padding, dilation, groups, output_mask);
}
at::Tensor npu_conv3d(const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_conv3d", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t)>();
    return op.call(input, weight, bias, stride, padding, dilation, groups);
}
at::Tensor & npu_conv3d_out(const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, at::Tensor & out) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_conv3d", "out").typed<at::Tensor & (const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, at::Tensor &)>();
    return op.call(input, weight, bias, stride, padding, dilation, groups, out);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_conv3d_backward(const at::Tensor & input, const at::Tensor & grad, const at::Tensor & weight, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, ::std::array<bool,3> output_mask) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_conv3d_backward", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, ::std::array<bool,3>)>();
    return op.call(input, grad, weight, stride, padding, dilation, groups, output_mask);
}
at::Tensor npu_conv_transpose2d(const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_conv_transpose2d", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t)>();
    return op.call(input, weight, bias, padding, output_padding, stride, dilation, groups);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_conv_transpose2d_backward(const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, ::std::array<bool,3> output_mask) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_conv_transpose2d_backward", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, ::std::array<bool,3>)>();
    return op.call(input, grad_output, weight, padding, output_padding, stride, dilation, groups, output_mask);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_conv_transpose3d_backward(const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, ::std::array<bool,3> output_mask) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_conv_transpose3d_backward", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, ::std::array<bool,3>)>();
    return op.call(input, grad_output, weight, padding, output_padding, stride, dilation, groups, output_mask);
}
at::Tensor npu_convert_weight_to_int4pack(const at::Tensor & weight, int64_t inner_k_tiles) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_convert_weight_to_int4pack", "").typed<at::Tensor (const at::Tensor &, int64_t)>();
    return op.call(weight, inner_k_tiles);
}
at::Tensor npu_convolution(const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_convolution", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t)>();
    return op.call(input, weight, bias, stride, padding, dilation, groups);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_convolution_backward(const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, ::std::array<bool,3> output_mask) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_convolution_backward", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, ::std::array<bool,3>)>();
    return op.call(input, grad_output, weight, stride, padding, dilation, groups, output_mask);
}
at::Tensor npu_convolution_transpose(const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_convolution_transpose", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t)>();
    return op.call(input, weight, bias, padding, output_padding, stride, dilation, groups);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_convolution_transpose_backward(const at::Tensor & input, const at::Tensor & grad, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, ::std::array<bool,3> grad_input_mask) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_convolution_transpose_backward", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, ::std::array<bool,3>)>();
    return op.call(input, grad, weight, padding, output_padding, stride, dilation, groups, grad_input_mask);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_deep_norm(const at::Tensor & x, const at::Tensor & gx, const at::Tensor & beta, const at::Tensor & gamma, double alpha, double epsilon) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_deep_norm", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, double, double)>();
    return op.call(x, gx, beta, gamma, alpha, epsilon);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_deep_norm_backward(const at::Tensor & dy, const at::Tensor & x, const at::Tensor & gx, const at::Tensor & gamma, const at::Tensor & mean, const at::Tensor & rstd, double alpha) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_deep_norm_backward", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, double)>();
    return op.call(dy, x, gx, gamma, mean, rstd, alpha);
}
::std::tuple<at::Tensor,at::Tensor> npu_deformable_conv2d(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & offset, const ::std::optional<at::Tensor> & bias, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, int64_t deformable_groups, bool modulated) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_deformable_conv2d", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, int64_t, bool)>();
    return op.call(input, weight, offset, bias, kernel_size, stride, padding, dilation, groups, deformable_groups, modulated);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_deformable_conv2dbk(const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & offset_out, const at::Tensor & weight, const at::Tensor & offset, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, int64_t deformable_groups, bool modulated) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_deformable_conv2dbk", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, int64_t, bool)>();
    return op.call(input, grad_output, offset_out, weight, offset, kernel_size, stride, padding, dilation, groups, deformable_groups, modulated);
}
at::Tensor npu_diou(const at::Tensor & self, const at::Tensor & gtboxes, bool trans, bool is_cross, int64_t mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_diou", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, bool, bool, int64_t)>();
    return op.call(self, gtboxes, trans, is_cross, mode);
}
::std::tuple<at::Tensor,at::Tensor> npu_diou_backward(const at::Tensor & grad, const at::Tensor & bboxes, const at::Tensor & gtboxes, bool trans, bool is_cross, int64_t mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_diou_backward", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, bool, bool, int64_t)>();
    return op.call(grad, bboxes, gtboxes, trans, is_cross, mode);
}
at::Tensor npu_dropout_backward(const at::Tensor & grad_output, const at::Tensor & mask, double p) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_dropout_backward", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, double)>();
    return op.call(grad_output, mask, p);
}
::std::tuple<at::Tensor,at::Tensor> npu_dropout_do_mask(const at::Tensor & self, const at::Tensor & mask, double p) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_dropout_do_mask", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, double)>();
    return op.call(self, mask, p);
}
at::Tensor npu_dropout_gen_mask(at::IntArrayRef size, double p, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_dropout_gen_mask", "").typed<at::Tensor (at::IntArrayRef, double, ::std::optional<at::ScalarType>, ::std::optional<at::Layout>, ::std::optional<at::Device>, ::std::optional<bool>)>();
    return op.call(size, p, dtype, layout, device, pin_memory);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_dropout_with_add_softmax(const at::Tensor & self, const at::Tensor & x1, const at::Scalar & alpha, double prob, int64_t dim) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_dropout_with_add_softmax", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Scalar &, double, int64_t)>();
    return op.call(self, x1, alpha, prob, dim);
}
::std::tuple<at::Tensor,at::Tensor> npu_dropout_with_add_softmax_backward(const at::Tensor & grad, const at::Tensor & mask, const at::Tensor & softmax_out, const at::Scalar & alpha, double prob, int64_t dim) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_dropout_with_add_softmax_backward", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar &, double, int64_t)>();
    return op.call(grad, mask, softmax_out, alpha, prob, dim);
}
at::Tensor npu_dtype_cast(const at::Tensor & self, at::ScalarType dtype) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_dtype_cast", "").typed<at::Tensor (const at::Tensor &, at::ScalarType)>();
    return op.call(self, dtype);
}
at::Tensor & npu_dtype_cast_(at::Tensor & self, const at::Tensor & src) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_dtype_cast_", "").typed<at::Tensor & (at::Tensor &, const at::Tensor &)>();
    return op.call(self, src);
}
at::Tensor npu_dtype_cast_backward(const at::Tensor & grad, at::ScalarType dtype) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_dtype_cast_backward", "").typed<at::Tensor (const at::Tensor &, at::ScalarType)>();
    return op.call(grad, dtype);
}
::std::tuple<at::Tensor,at::Tensor> npu_dynamic_quant(const at::Tensor & input, const ::std::optional<at::Tensor> & smooth_scales, const ::std::optional<at::Tensor> & group_index, ::std::optional<at::ScalarType> dst_type) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_dynamic_quant", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, ::std::optional<at::ScalarType>)>();
    return op.call(input, smooth_scales, group_index, dst_type);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_dynamic_quant_asymmetric(const at::Tensor & input, const ::std::optional<at::Tensor> & smooth_scales, const ::std::optional<at::Tensor> & group_index, ::std::optional<at::ScalarType> dst_type) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_dynamic_quant_asymmetric", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, ::std::optional<at::ScalarType>)>();
    return op.call(input, smooth_scales, group_index, dst_type);
}
at::Tensor npu_fast_gelu(const at::Tensor & self) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_fast_gelu", "").typed<at::Tensor (const at::Tensor &)>();
    return op.call(self);
}
at::Tensor npu_fast_gelu_backward(const at::Tensor & grad, const at::Tensor & self) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_fast_gelu_backward", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &)>();
    return op.call(grad, self);
}
at::Tensor npu_ffn(const at::Tensor & x, const at::Tensor & weight1, const at::Tensor & weight2, c10::string_view activation, at::OptionalIntArrayRef expert_tokens, at::OptionalIntArrayRef expert_tokens_index, const ::std::optional<at::Tensor> & bias1, const ::std::optional<at::Tensor> & bias2, const ::std::optional<at::Tensor> & scale, const ::std::optional<at::Tensor> & offset, const ::std::optional<at::Tensor> & deq_scale1, const ::std::optional<at::Tensor> & deq_scale2, const ::std::optional<at::Tensor> & antiquant_scale1, const ::std::optional<at::Tensor> & antiquant_scale2, const ::std::optional<at::Tensor> & antiquant_offset1, const ::std::optional<at::Tensor> & antiquant_offset2, ::std::optional<int64_t> inner_precise, ::std::optional<at::ScalarType> output_dtype) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_ffn", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, c10::string_view, at::OptionalIntArrayRef, at::OptionalIntArrayRef, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, ::std::optional<int64_t>, ::std::optional<at::ScalarType>)>();
    return op.call(x, weight1, weight2, activation, expert_tokens, expert_tokens_index, bias1, bias2, scale, offset, deq_scale1, deq_scale2, antiquant_scale1, antiquant_scale2, antiquant_offset1, antiquant_offset2, inner_precise, output_dtype);
}
at::Tensor npu_fused_attention_score(const at::Tensor & query_layer, const at::Tensor & key_layer, const at::Tensor & value_layer, const at::Tensor & attention_mask, const at::Scalar & scale, double keep_prob, bool query_transpose, bool key_transpose, bool bmm_score_transpose_a, bool bmm_score_transpose_b, bool value_transpose, bool dx_transpose) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_fused_attention_score", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar &, double, bool, bool, bool, bool, bool, bool)>();
    return op.call(query_layer, key_layer, value_layer, attention_mask, scale, keep_prob, query_transpose, key_transpose, bmm_score_transpose_a, bmm_score_transpose_b, value_transpose, dx_transpose);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_fused_attention_score_backward(const at::Tensor & grad_output, const at::Tensor & softmax_output, const at::Tensor & query_layer, const at::Tensor & key_layer, const at::Tensor & value_layer, const at::Tensor & mask, const at::Scalar & scale, double keep_prob, bool query_transpose, bool key_transpose, bool value_transpose, bool dx_transpose) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_fused_attention_score_backward", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar &, double, bool, bool, bool, bool)>();
    return op.call(grad_output, softmax_output, query_layer, key_layer, value_layer, mask, scale, keep_prob, query_transpose, key_transpose, value_transpose, dx_transpose);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_fused_attention_score_fwd(const at::Tensor & query_layer, const at::Tensor & key_layer, const at::Tensor & value_layer, const at::Tensor & attention_mask, const at::Scalar & scale, double keep_prob, bool query_transpose, bool key_transpose, bool bmm_score_transpose_a, bool bmm_score_transpose_b, bool value_transpose, bool dx_transpose) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_fused_attention_score_fwd", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar &, double, bool, bool, bool, bool, bool, bool)>();
    return op.call(query_layer, key_layer, value_layer, attention_mask, scale, keep_prob, query_transpose, key_transpose, bmm_score_transpose_a, bmm_score_transpose_b, value_transpose, dx_transpose);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_fused_attention_score_grad(const at::Tensor & grad_output, const at::Tensor & softmax_output, const at::Tensor & query_layer, const at::Tensor & key_layer, const at::Tensor & value_layer, const at::Tensor & mask, const at::Scalar & scale, double keep_prob, bool query_transpose, bool key_transpose, bool value_transpose, bool dx_transpose) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_fused_attention_score_grad", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar &, double, bool, bool, bool, bool)>();
    return op.call(grad_output, softmax_output, query_layer, key_layer, value_layer, mask, scale, keep_prob, query_transpose, key_transpose, value_transpose, dx_transpose);
}
::std::tuple<at::Tensor,at::Tensor> npu_fused_infer_attention_score(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & pse_shift, const ::std::optional<at::Tensor> & atten_mask, at::OptionalSymIntArrayRef actual_seq_lengths, at::OptionalSymIntArrayRef actual_seq_lengths_kv, const ::std::optional<at::Tensor> & dequant_scale1, const ::std::optional<at::Tensor> & quant_scale1, const ::std::optional<at::Tensor> & dequant_scale2, const ::std::optional<at::Tensor> & quant_scale2, const ::std::optional<at::Tensor> & quant_offset2, const ::std::optional<at::Tensor> & antiquant_scale, const ::std::optional<at::Tensor> & antiquant_offset, const ::std::optional<at::Tensor> & key_antiquant_scale, const ::std::optional<at::Tensor> & key_antiquant_offset, const ::std::optional<at::Tensor> & value_antiquant_scale, const ::std::optional<at::Tensor> & value_antiquant_offset, const ::std::optional<at::Tensor> & block_table, const ::std::optional<at::Tensor> & query_padding_size, const ::std::optional<at::Tensor> & kv_padding_size, const ::std::optional<at::Tensor> & key_shared_prefix, const ::std::optional<at::Tensor> & value_shared_prefix, at::OptionalSymIntArrayRef actual_shared_prefix_len, const ::std::optional<at::Tensor> & query_rope, const ::std::optional<at::Tensor> & key_rope, const ::std::optional<at::Tensor> & key_rope_antiquant_scale, int64_t num_heads, double scale, int64_t pre_tokens, int64_t next_tokens, c10::string_view input_layout, int64_t num_key_value_heads, int64_t sparse_mode, int64_t inner_precise, int64_t block_size, int64_t antiquant_mode, int64_t key_antiquant_mode, int64_t value_antiquant_mode, bool softmax_lse_flag) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_fused_infer_attention_score", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, at::OptionalSymIntArrayRef, at::OptionalSymIntArrayRef, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, at::OptionalSymIntArrayRef, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, int64_t, double, int64_t, int64_t, c10::string_view, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool)>();
    return op.call(query, key, value, pse_shift, atten_mask, actual_seq_lengths, actual_seq_lengths_kv, dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2, antiquant_scale, antiquant_offset, key_antiquant_scale, key_antiquant_offset, value_antiquant_scale, value_antiquant_offset, block_table, query_padding_size, kv_padding_size, key_shared_prefix, value_shared_prefix, actual_shared_prefix_len, query_rope, key_rope, key_rope_antiquant_scale, num_heads, scale, pre_tokens, next_tokens, input_layout, num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, key_antiquant_mode, value_antiquant_mode, softmax_lse_flag);
}
::std::tuple<at::Tensor &,at::Tensor &> npu_fused_infer_attention_score_out(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & pse_shift, const ::std::optional<at::Tensor> & atten_mask, at::OptionalSymIntArrayRef actual_seq_lengths, at::OptionalSymIntArrayRef actual_seq_lengths_kv, const ::std::optional<at::Tensor> & dequant_scale1, const ::std::optional<at::Tensor> & quant_scale1, const ::std::optional<at::Tensor> & dequant_scale2, const ::std::optional<at::Tensor> & quant_scale2, const ::std::optional<at::Tensor> & quant_offset2, const ::std::optional<at::Tensor> & antiquant_scale, const ::std::optional<at::Tensor> & antiquant_offset, const ::std::optional<at::Tensor> & key_antiquant_scale, const ::std::optional<at::Tensor> & key_antiquant_offset, const ::std::optional<at::Tensor> & value_antiquant_scale, const ::std::optional<at::Tensor> & value_antiquant_offset, const ::std::optional<at::Tensor> & block_table, const ::std::optional<at::Tensor> & query_padding_size, const ::std::optional<at::Tensor> & kv_padding_size, const ::std::optional<at::Tensor> & key_shared_prefix, const ::std::optional<at::Tensor> & value_shared_prefix, at::OptionalSymIntArrayRef actual_shared_prefix_len, const ::std::optional<at::Tensor> & query_rope, const ::std::optional<at::Tensor> & key_rope, const ::std::optional<at::Tensor> & key_rope_antiquant_scale, int64_t num_heads, double scale, int64_t pre_tokens, int64_t next_tokens, c10::string_view input_layout, int64_t num_key_value_heads, int64_t sparse_mode, int64_t inner_precise, int64_t block_size, int64_t antiquant_mode, int64_t key_antiquant_mode, int64_t value_antiquant_mode, bool softmax_lse_flag, const ::std::optional<at::Tensor> & workspace, at::Tensor & attention_out, at::Tensor & softmax_lse) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_fused_infer_attention_score", "out").typed<::std::tuple<at::Tensor &,at::Tensor &> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, at::OptionalSymIntArrayRef, at::OptionalSymIntArrayRef, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, at::OptionalSymIntArrayRef, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, int64_t, double, int64_t, int64_t, c10::string_view, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool, const ::std::optional<at::Tensor> &, at::Tensor &, at::Tensor &)>();
    return op.call(query, key, value, pse_shift, atten_mask, actual_seq_lengths, actual_seq_lengths_kv, dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2, antiquant_scale, antiquant_offset, key_antiquant_scale, key_antiquant_offset, value_antiquant_scale, value_antiquant_offset, block_table, query_padding_size, kv_padding_size, key_shared_prefix, value_shared_prefix, actual_shared_prefix_len, query_rope, key_rope, key_rope_antiquant_scale, num_heads, scale, pre_tokens, next_tokens, input_layout, num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, key_antiquant_mode, value_antiquant_mode, softmax_lse_flag, workspace, attention_out, softmax_lse);
}
at::Tensor _npu_fused_infer_attention_score_get_max_workspace(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & pse_shift, const ::std::optional<at::Tensor> & atten_mask, at::OptionalSymIntArrayRef actual_seq_lengths, at::OptionalSymIntArrayRef actual_seq_lengths_kv, const ::std::optional<at::Tensor> & dequant_scale1, const ::std::optional<at::Tensor> & quant_scale1, const ::std::optional<at::Tensor> & dequant_scale2, const ::std::optional<at::Tensor> & quant_scale2, const ::std::optional<at::Tensor> & quant_offset2, const ::std::optional<at::Tensor> & antiquant_scale, const ::std::optional<at::Tensor> & antiquant_offset, const ::std::optional<at::Tensor> & key_antiquant_scale, const ::std::optional<at::Tensor> & key_antiquant_offset, const ::std::optional<at::Tensor> & value_antiquant_scale, const ::std::optional<at::Tensor> & value_antiquant_offset, const ::std::optional<at::Tensor> & block_table, const ::std::optional<at::Tensor> & query_padding_size, const ::std::optional<at::Tensor> & kv_padding_size, const ::std::optional<at::Tensor> & key_shared_prefix, const ::std::optional<at::Tensor> & value_shared_prefix, at::OptionalSymIntArrayRef actual_shared_prefix_len, const ::std::optional<at::Tensor> & query_rope, const ::std::optional<at::Tensor> & key_rope, const ::std::optional<at::Tensor> & key_rope_antiquant_scale, int64_t num_heads, double scale, int64_t pre_tokens, int64_t next_tokens, c10::string_view input_layout, int64_t num_key_value_heads, int64_t sparse_mode, int64_t inner_precise, int64_t block_size, int64_t antiquant_mode, int64_t key_antiquant_mode, int64_t value_antiquant_mode, bool softmax_lse_flag) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::_npu_fused_infer_attention_score_get_max_workspace", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, at::OptionalSymIntArrayRef, at::OptionalSymIntArrayRef, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, at::OptionalSymIntArrayRef, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, int64_t, double, int64_t, int64_t, c10::string_view, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool)>();
    return op.call(query, key, value, pse_shift, atten_mask, actual_seq_lengths, actual_seq_lengths_kv, dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2, antiquant_scale, antiquant_offset, key_antiquant_scale, key_antiquant_offset, value_antiquant_scale, value_antiquant_offset, block_table, query_padding_size, kv_padding_size, key_shared_prefix, value_shared_prefix, actual_shared_prefix_len, query_rope, key_rope, key_rope_antiquant_scale, num_heads, scale, pre_tokens, next_tokens, input_layout, num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, key_antiquant_mode, value_antiquant_mode, softmax_lse_flag);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t,int64_t,int64_t> npu_fusion_attention(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, int64_t head_num, c10::string_view input_layout, const ::std::optional<at::Tensor> & pse, const ::std::optional<at::Tensor> & padding_mask, const ::std::optional<at::Tensor> & atten_mask, double scale, double keep_prob, int64_t pre_tockens, int64_t next_tockens, int64_t inner_precise, at::OptionalIntArrayRef prefix, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_seq_kvlen, int64_t sparse_mode, bool gen_mask_parallel, bool sync) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_fusion_attention", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t,int64_t,int64_t> (const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, c10::string_view, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, double, double, int64_t, int64_t, int64_t, at::OptionalIntArrayRef, at::OptionalIntArrayRef, at::OptionalIntArrayRef, int64_t, bool, bool)>();
    return op.call(query, key, value, head_num, input_layout, pse, padding_mask, atten_mask, scale, keep_prob, pre_tockens, next_tockens, inner_precise, prefix, actual_seq_qlen, actual_seq_kvlen, sparse_mode, gen_mask_parallel, sync);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_fusion_attention_grad(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & dy, int64_t head_num, c10::string_view input_layout, const ::std::optional<at::Tensor> & pse, const ::std::optional<at::Tensor> & padding_mask, const ::std::optional<at::Tensor> & atten_mask, const ::std::optional<at::Tensor> & softmax_max, const ::std::optional<at::Tensor> & softmax_sum, const ::std::optional<at::Tensor> & softmax_in, const ::std::optional<at::Tensor> & attention_in, double scale_value, double keep_prob, int64_t pre_tockens, int64_t next_tockens, int64_t inner_precise, int64_t seed, int64_t offset, int64_t numels, at::OptionalIntArrayRef prefix, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_seq_kvlen, int64_t sparse_mode, bool gen_mask_parallel, bool sync) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_fusion_attention_grad", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, c10::string_view, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, double, double, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, at::OptionalIntArrayRef, at::OptionalIntArrayRef, at::OptionalIntArrayRef, int64_t, bool, bool)>();
    return op.call(query, key, value, dy, head_num, input_layout, pse, padding_mask, atten_mask, softmax_max, softmax_sum, softmax_in, attention_in, scale_value, keep_prob, pre_tockens, next_tockens, inner_precise, seed, offset, numels, prefix, actual_seq_qlen, actual_seq_kvlen, sparse_mode, gen_mask_parallel, sync);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t,int64_t,int64_t> npu_fusion_attention_v2(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, int64_t head_num, c10::string_view input_layout, const ::std::optional<at::Tensor> & pse, const ::std::optional<at::Tensor> & padding_mask, const ::std::optional<at::Tensor> & atten_mask, const ::std::optional<at::Tensor> & query_rope, const ::std::optional<at::Tensor> & key_rope, double scale, double keep_prob, int64_t pre_tokens, int64_t next_tokens, int64_t inner_precise, at::OptionalIntArrayRef prefix, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_seq_kvlen, int64_t sparse_mode, bool gen_mask_parallel, bool sync, int64_t pse_type, at::OptionalIntArrayRef q_start_idx, at::OptionalIntArrayRef kv_start_idx) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_fusion_attention_v2", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t,int64_t,int64_t> (const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, c10::string_view, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, double, double, int64_t, int64_t, int64_t, at::OptionalIntArrayRef, at::OptionalIntArrayRef, at::OptionalIntArrayRef, int64_t, bool, bool, int64_t, at::OptionalIntArrayRef, at::OptionalIntArrayRef)>();
    return op.call(query, key, value, head_num, input_layout, pse, padding_mask, atten_mask, query_rope, key_rope, scale, keep_prob, pre_tokens, next_tokens, inner_precise, prefix, actual_seq_qlen, actual_seq_kvlen, sparse_mode, gen_mask_parallel, sync, pse_type, q_start_idx, kv_start_idx);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_fusion_attention_grad_v2(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & dy, int64_t head_num, c10::string_view input_layout, const ::std::optional<at::Tensor> & pse, const ::std::optional<at::Tensor> & padding_mask, const ::std::optional<at::Tensor> & atten_mask, const ::std::optional<at::Tensor> & softmax_max, const ::std::optional<at::Tensor> & softmax_sum, const ::std::optional<at::Tensor> & softmax_in, const ::std::optional<at::Tensor> & attention_in, const ::std::optional<at::Tensor> & query_rope, const ::std::optional<at::Tensor> & key_rope, double scale_value, double keep_prob, int64_t pre_tokens, int64_t next_tokens, int64_t inner_precise, int64_t seed, int64_t offset, int64_t numels, at::OptionalIntArrayRef prefix, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_seq_kvlen, int64_t sparse_mode, bool gen_mask_parallel, bool sync, int64_t pse_type, at::OptionalIntArrayRef q_start_idx, at::OptionalIntArrayRef kv_start_idx) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_fusion_attention_grad_v2", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, c10::string_view, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, double, double, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, at::OptionalIntArrayRef, at::OptionalIntArrayRef, at::OptionalIntArrayRef, int64_t, bool, bool, int64_t, at::OptionalIntArrayRef, at::OptionalIntArrayRef)>();
    return op.call(query, key, value, dy, head_num, input_layout, pse, padding_mask, atten_mask, softmax_max, softmax_sum, softmax_in, attention_in, query_rope, key_rope, scale_value, keep_prob, pre_tokens, next_tokens, inner_precise, seed, offset, numels, prefix, actual_seq_qlen, actual_seq_kvlen, sparse_mode, gen_mask_parallel, sync, pse_type, q_start_idx, kv_start_idx);
}
::std::vector<at::Tensor> npu_fused_attention_layernorm_qkv_fwd(const at::Tensor & x, const at::Tensor & kernel_query, const at::Tensor & kernel_key, const at::Tensor & kernel_value, const at::Tensor & gamma, const at::Tensor & beta, const ::std::optional<at::Tensor> & bias_query, const ::std::optional<at::Tensor> & bias_key, const ::std::optional<at::Tensor> & bias_value, int64_t seq_len, int64_t num_heads, double eps) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_fused_attention_layernorm_qkv_fwd", "").typed<::std::vector<at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, int64_t, int64_t, double)>();
    return op.call(x, kernel_query, kernel_key, kernel_value, gamma, beta, bias_query, bias_key, bias_value, seq_len, num_heads, eps);
}
::std::vector<at::Tensor> npu_fused_attention_qkv_grad(const at::Tensor & grad_output_query, const at::Tensor & grad_output_key, const at::Tensor & grad_output_value, const at::Tensor & query_kernel, const at::Tensor & key_kernel, const at::Tensor & value_kernel, const at::Tensor & hidden_states, const at::Tensor & grad_output_ln) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_fused_attention_qkv_grad", "").typed<::std::vector<at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &)>();
    return op.call(grad_output_query, grad_output_key, grad_output_value, query_kernel, key_kernel, value_kernel, hidden_states, grad_output_ln);
}
::std::tuple<at::Tensor,at::Tensor> npu_geglu(const at::Tensor & self, int64_t dim, int64_t approximate, bool activate_left) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_geglu", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, int64_t, int64_t, bool)>();
    return op.call(self, dim, approximate, activate_left);
}
at::Tensor npu_geglu_grad(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & gelu, int64_t dim, int64_t approximate, bool activate_left) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_geglu_grad", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, int64_t, bool)>();
    return op.call(grad_output, self, gelu, dim, approximate, activate_left);
}
at::Tensor npu_get_float_status(const at::Tensor & self, int64_t mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_get_float_status", "").typed<at::Tensor (const at::Tensor &, int64_t)>();
    return op.call(self, mode);
}
at::Tensor npu_giou(const at::Tensor & self, const at::Tensor & gtboxes, bool trans, bool is_cross, int64_t mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_giou", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, bool, bool, int64_t)>();
    return op.call(self, gtboxes, trans, is_cross, mode);
}
::std::tuple<at::Tensor,at::Tensor> npu_giou_backward(const at::Tensor & grad, const at::Tensor & bboxes, const at::Tensor & gtboxes, bool trans, bool is_cross, int64_t mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_giou_backward", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, bool, bool, int64_t)>();
    return op.call(grad, bboxes, gtboxes, trans, is_cross, mode);
}
at::Tensor npu_grid_assign_positive(const at::Tensor & self, const at::Tensor & overlaps, const at::Tensor & box_responsible_flags, const at::Tensor & max_overlaps, const at::Tensor & argmax_overlaps, const at::Tensor & gt_max_overlaps, const at::Tensor & gt_argmax_overlaps, int64_t num_gts, double pos_iou_thr, double min_pos_iou, bool gt_max_assign_all) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_grid_assign_positive", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, double, double, bool)>();
    return op.call(self, overlaps, box_responsible_flags, max_overlaps, argmax_overlaps, gt_max_overlaps, gt_argmax_overlaps, num_gts, pos_iou_thr, min_pos_iou, gt_max_assign_all);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_group_norm_silu(const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, int64_t group, double eps) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_group_norm_silu", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, int64_t, double)>();
    return op.call(input, weight, bias, group, eps);
}
::std::vector<at::Tensor> npu_grouped_matmul(at::TensorList x, at::TensorList weight, ::std::optional<at::TensorList> bias, ::std::optional<at::TensorList> scale, ::std::optional<at::TensorList> offset, ::std::optional<at::TensorList> antiquant_scale, ::std::optional<at::TensorList> antiquant_offset, ::std::optional<at::TensorList> per_token_scale, const ::std::optional<at::Tensor> & group_list, ::std::optional<at::TensorList> activation_input, ::std::optional<at::TensorList> activation_quant_scale, ::std::optional<at::TensorList> activation_quant_offset, ::std::optional<int64_t> split_item, ::std::optional<int64_t> group_type, ::std::optional<int64_t> group_list_type, ::std::optional<int64_t> act_type, at::OptionalIntArrayRef tuning_config, ::std::optional<at::ScalarType> output_dtype) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_grouped_matmul", "").typed<::std::vector<at::Tensor> (at::TensorList, at::TensorList, ::std::optional<at::TensorList>, ::std::optional<at::TensorList>, ::std::optional<at::TensorList>, ::std::optional<at::TensorList>, ::std::optional<at::TensorList>, ::std::optional<at::TensorList>, const ::std::optional<at::Tensor> &, ::std::optional<at::TensorList>, ::std::optional<at::TensorList>, ::std::optional<at::TensorList>, ::std::optional<int64_t>, ::std::optional<int64_t>, ::std::optional<int64_t>, ::std::optional<int64_t>, at::OptionalIntArrayRef, ::std::optional<at::ScalarType>)>();
    return op.call(x, weight, bias, scale, offset, antiquant_scale, antiquant_offset, per_token_scale, group_list, activation_input, activation_quant_scale, activation_quant_offset, split_item, group_type, group_list_type, act_type, tuning_config, output_dtype);
}
::std::vector<at::Tensor> npu_grouped_matmul(at::TensorList x, at::TensorList weight, ::std::optional<at::TensorList> bias, ::std::optional<at::TensorList> scale, ::std::optional<at::TensorList> offset, ::std::optional<at::TensorList> antiquant_scale, ::std::optional<at::TensorList> antiquant_offset, ::std::optional<at::TensorList> per_token_scale, at::OptionalIntArrayRef group_list, ::std::optional<at::TensorList> activation_input, ::std::optional<at::TensorList> activation_quant_scale, ::std::optional<at::TensorList> activation_quant_offset, ::std::optional<int64_t> split_item, ::std::optional<int64_t> group_type, ::std::optional<int64_t> group_list_type, ::std::optional<int64_t> act_type, ::std::optional<at::ScalarType> output_dtype) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_grouped_matmul", "List").typed<::std::vector<at::Tensor> (at::TensorList, at::TensorList, ::std::optional<at::TensorList>, ::std::optional<at::TensorList>, ::std::optional<at::TensorList>, ::std::optional<at::TensorList>, ::std::optional<at::TensorList>, ::std::optional<at::TensorList>, at::OptionalIntArrayRef, ::std::optional<at::TensorList>, ::std::optional<at::TensorList>, ::std::optional<at::TensorList>, ::std::optional<int64_t>, ::std::optional<int64_t>, ::std::optional<int64_t>, ::std::optional<int64_t>, ::std::optional<at::ScalarType>)>();
    return op.call(x, weight, bias, scale, offset, antiquant_scale, antiquant_offset, per_token_scale, group_list, activation_input, activation_quant_scale, activation_quant_offset, split_item, group_type, group_list_type, act_type, output_dtype);
}
at::Tensor npu_grouped_matmul_finalize_routing(const at::Tensor & x, const at::Tensor & w, const at::Tensor & group_list, const ::std::optional<at::Tensor> & scale, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & offset, const ::std::optional<at::Tensor> & pertoken_scale, const ::std::optional<at::Tensor> & shared_input, const ::std::optional<at::Tensor> & logit, const ::std::optional<at::Tensor> & row_index, ::std::optional<at::ScalarType> dtype, ::std::optional<double> shared_input_weight, ::std::optional<int64_t> shared_input_offset, ::std::optional<int64_t> output_bs, ::std::optional<int64_t> group_list_type) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_grouped_matmul_finalize_routing", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, ::std::optional<at::ScalarType>, ::std::optional<double>, ::std::optional<int64_t>, ::std::optional<int64_t>, ::std::optional<int64_t>)>();
    return op.call(x, w, group_list, scale, bias, offset, pertoken_scale, shared_input, logit, row_index, dtype, shared_input_weight, shared_input_offset, output_bs, group_list_type);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_gru(const at::Tensor & input, const at::Tensor & hx, const at::Tensor & weight_input, const at::Tensor & weight_hidden, const at::Tensor & bias_input, const at::Tensor & bias_hidden, const at::Tensor & seq_length, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_gru", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, bool, int64_t, double, bool, bool, bool)>();
    return op.call(input, hx, weight_input, weight_hidden, bias_input, bias_hidden, seq_length, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_gru_backward(const ::std::optional<at::Tensor> & grady, const ::std::optional<at::Tensor> & gradh, const at::Tensor & input, const at::Tensor & weight_input, const at::Tensor & weight_hidden, const at::Tensor & bias_input, const at::Tensor & bias_hidden, const at::Tensor & seq_length, const at::Tensor & hx, const at::Tensor & y_output, const at::Tensor & h_output, const at::Tensor & output_updata, const at::Tensor & output_reset, const at::Tensor & output_new, const at::Tensor & hidden_new) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_gru_backward", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &)>();
    return op.call(grady, gradh, input, weight_input, weight_hidden, bias_input, bias_hidden, seq_length, hx, y_output, h_output, output_updata, output_reset, output_new, hidden_new);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> npu_hans_encode_out(const at::Tensor & input, bool statistic, bool reshuff, at::Tensor & pdf, at::Tensor & mantissa, at::Tensor & fixed, at::Tensor & var) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_hans_encode", "out").typed<::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> (const at::Tensor &, bool, bool, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &)>();
    return op.call(input, statistic, reshuff, pdf, mantissa, fixed, var);
}
at::Tensor & npu_hans_decode_out(const at::Tensor & mantissa, const at::Tensor & fixed, const at::Tensor & var, const at::Tensor & pdf, bool reshuff, at::Tensor & out) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_hans_decode", "out").typed<at::Tensor & (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, bool, at::Tensor &)>();
    return op.call(mantissa, fixed, var, pdf, reshuff, out);
}
::std::tuple<at::Tensor,at::Tensor> npu_ifmr(const at::Tensor & data, const at::Tensor & data_min, const at::Tensor & data_max, const at::Tensor & cumsum, double min_percentile, double max_percentile, double search_start, double search_end, double search_step, bool with_offset) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_ifmr", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, double, double, double, double, double, bool)>();
    return op.call(data, data_min, data_max, cumsum, min_percentile, max_percentile, search_start, search_end, search_step, with_offset);
}
at::Tensor npu_our_incre_flash_attention(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & padding_mask, const ::std::optional<at::Tensor> & atten_mask, const ::std::optional<at::Tensor> & pse_shift, at::OptionalSymIntArrayRef actual_seq_lengths, const ::std::optional<at::Tensor> & antiquant_scale, const ::std::optional<at::Tensor> & antiquant_offset, const ::std::optional<at::Tensor> & block_table, const ::std::optional<at::Tensor> & dequant_scale1, const ::std::optional<at::Tensor> & quant_scale1, const ::std::optional<at::Tensor> & dequant_scale2, const ::std::optional<at::Tensor> & quant_scale2, const ::std::optional<at::Tensor> & quant_offset2, const ::std::optional<at::Tensor> & kv_padding_size, int64_t num_heads, double scale_value, c10::string_view input_layout, int64_t num_key_value_heads, int64_t block_size, int64_t inner_precise) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_our_incre_flash_attention", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, at::OptionalSymIntArrayRef, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, int64_t, double, c10::string_view, int64_t, int64_t, int64_t)>();
    return op.call(query, key, value, padding_mask, atten_mask, pse_shift, actual_seq_lengths, antiquant_scale, antiquant_offset, block_table, dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2, kv_padding_size, num_heads, scale_value, input_layout, num_key_value_heads, block_size, inner_precise);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_sparse_paged_fusion_attention(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & blocktable, const at::Tensor & l1_cent, const at::Tensor & block_ids, const at::Tensor & total_seq_len, const ::std::optional<at::Tensor> & pse_shift, const ::std::optional<at::Tensor> & attention_mask, at::OptionalSymIntArrayRef actual_seq_lengths, const ::std::optional<at::Tensor> & dequant_scale1, const ::std::optional<at::Tensor> & quant_scale1, const ::std::optional<at::Tensor> & dequant_scale2, const ::std::optional<at::Tensor> & quant_scale2, const ::std::optional<at::Tensor> & quant_offset2, const ::std::optional<at::Tensor> & antiquant_scale, const ::std::optional<at::Tensor> & antiquant_offset, const ::std::optional<at::Tensor> & kv_padding_size, int64_t num_heads, double scale_value, c10::string_view input_layout, int64_t num_key_value_heads, int64_t block_size, int64_t inner_precise) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_sparse_paged_fusion_attention", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, at::OptionalSymIntArrayRef, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, int64_t, double, c10::string_view, int64_t, int64_t, int64_t)>();
    return op.call(query, key, value, blocktable, l1_cent, block_ids, total_seq_len, pse_shift, attention_mask, actual_seq_lengths, dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2, antiquant_scale, antiquant_offset, kv_padding_size, num_heads, scale_value, input_layout, num_key_value_heads, block_size, inner_precise);
}
at::Tensor npu_sparse_paged_attention(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & padding_mask, const ::std::optional<at::Tensor> & atten_mask, const ::std::optional<at::Tensor> & pse_shift, at::OptionalSymIntArrayRef actual_seq_lengths, const ::std::optional<at::Tensor> & antiquant_scale, const ::std::optional<at::Tensor> & antiquant_offset, const ::std::optional<at::Tensor> & block_table, const ::std::optional<at::Tensor> & block_position, const ::std::optional<at::Tensor> & dequant_scale1, const ::std::optional<at::Tensor> & quant_scale1, const ::std::optional<at::Tensor> & dequant_scale2, const ::std::optional<at::Tensor> & quant_scale2, const ::std::optional<at::Tensor> & quant_offset2, const ::std::optional<at::Tensor> & kv_padding_size, int64_t num_heads, double scale_value, c10::string_view input_layout, int64_t num_key_value_heads, int64_t block_size, int64_t inner_precise) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_sparse_paged_attention", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, at::OptionalSymIntArrayRef, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, int64_t, double, c10::string_view, int64_t, int64_t, int64_t)>();
    return op.call(query, key, value, padding_mask, atten_mask, pse_shift, actual_seq_lengths, antiquant_scale, antiquant_offset, block_table, block_position, dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2, kv_padding_size, num_heads, scale_value, input_layout, num_key_value_heads, block_size, inner_precise);
}
::std::tuple<at::Tensor,at::Tensor> npu_cent_select(const at::Tensor & query, const at::Tensor & l1_cent, const at::Tensor & block_ids, const at::Tensor & block_table, const at::Tensor & seq_len) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_cent_select", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &)>();
    return op.call(query, l1_cent, block_ids, block_table, seq_len);
}
at::Tensor npu_interleave_rope(const at::Tensor & x, const at::Tensor & cos, const at::Tensor & sin) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_interleave_rope", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &)>();
    return op.call(x, cos, sin);
}
at::Tensor npu_indexing(const at::Tensor & self, at::IntArrayRef begin, at::IntArrayRef end, at::IntArrayRef strides, int64_t begin_mask, int64_t end_mask, int64_t ellipsis_mask, int64_t new_axis_mask, int64_t shrink_axis_mask) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_indexing", "").typed<at::Tensor (const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, int64_t, int64_t, int64_t, int64_t)>();
    return op.call(self, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask);
}
at::Tensor & npu_indexing_out(const at::Tensor & self, at::IntArrayRef begin, at::IntArrayRef end, at::IntArrayRef strides, int64_t begin_mask, int64_t end_mask, int64_t ellipsis_mask, int64_t new_axis_mask, int64_t shrink_axis_mask, at::Tensor & out) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_indexing", "out").typed<at::Tensor & (const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, int64_t, int64_t, int64_t, int64_t, at::Tensor &)>();
    return op.call(self, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask, out);
}
at::Tensor npu_iou(const at::Tensor & bboxes, const at::Tensor & gtboxes, int64_t mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_iou", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, int64_t)>();
    return op.call(bboxes, gtboxes, mode);
}
at::Tensor npu_layer_norm_eval(const at::Tensor & input, at::IntArrayRef normalized_shape, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, double eps) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_layer_norm_eval", "").typed<at::Tensor (const at::Tensor &, at::IntArrayRef, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, double)>();
    return op.call(input, normalized_shape, weight, bias, eps);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_layernorm_grad(const at::Tensor & grad_out, const at::Tensor & input, at::IntArrayRef normalized_shape, const at::Tensor & mean, const at::Tensor & rstd, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_layernorm_grad", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, at::IntArrayRef, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &)>();
    return op.call(grad_out, input, normalized_shape, mean, rstd, weight, bias);
}
at::Tensor npu_linear(const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_linear", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &)>();
    return op.call(input, weight, bias);
}
::std::tuple<at::Tensor,at::Tensor> npu_linear_backward(const at::Tensor & grad, const at::Tensor & input, const at::Tensor & weight) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_linear_backward", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &)>();
    return op.call(grad, input, weight);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_lstm(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, const at::Tensor & seq_mask, const at::Tensor & h, const at::Tensor & c, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first, bool flag_seq, bool direction) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_lstm", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, bool, int64_t, double, bool, bool, bool, bool, bool)>();
    return op.call(input, weight, bias, seq_mask, h, c, has_biases, num_layers, dropout, train, bidirectional, batch_first, flag_seq, direction);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_lstm_backward(const ::std::optional<at::Tensor> & grady, const ::std::optional<at::Tensor> & gradh, const ::std::optional<at::Tensor> & gradc, const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, const at::Tensor & hx, const at::Tensor & cx, const at::Tensor & y_output, const at::Tensor & h_output, const at::Tensor & c_output, const at::Tensor & i, const at::Tensor & j, const at::Tensor & f, const at::Tensor & o, const at::Tensor & tanhc) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_lstm_backward", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &)>();
    return op.call(grady, gradh, gradc, input, weight, bias, hx, cx, y_output, h_output, c_output, i, j, f, o, tanhc);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_lstm_cell(const at::Tensor & input, const at::Tensor & w_ih, const at::Tensor & w_hh, const at::Tensor & h, const at::Tensor & c, const ::std::optional<at::Tensor> & b_ih, const ::std::optional<at::Tensor> & b_hh) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_lstm_cell", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &)>();
    return op.call(input, w_ih, w_hh, h, c, b_ih, b_hh);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_lstm_cell_backward(const ::std::optional<at::Tensor> & grady, const ::std::optional<at::Tensor> & gradh, const ::std::optional<at::Tensor> & gradc, const at::Tensor & input, const at::Tensor & w_ih, const at::Tensor & w_hh, const at::Tensor & h, const at::Tensor & c, const at::Tensor & y_output, const at::Tensor & h_output, const at::Tensor & c_output, const at::Tensor & i, const at::Tensor & j, const at::Tensor & f, const at::Tensor & o, const at::Tensor & tanhc) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_lstm_cell_backward", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &)>();
    return op.call(grady, gradh, gradc, input, w_ih, w_hh, h, c, y_output, h_output, c_output, i, j, f, o, tanhc);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_lstm_data(const at::Tensor & input, const at::Tensor & batch_sizes, const at::Tensor & weight, const at::Tensor & bias, const at::Tensor & seq_mask, const at::Tensor & h, const at::Tensor & c, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first, bool flag_seq, bool direction) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_lstm_data", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, bool, int64_t, double, bool, bool, bool, bool, bool)>();
    return op.call(input, batch_sizes, weight, bias, seq_mask, h, c, has_biases, num_layers, dropout, train, bidirectional, batch_first, flag_seq, direction);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_lstm_data_backward(const ::std::optional<at::Tensor> & grady_opt, const ::std::optional<at::Tensor> & gradh_opt, const ::std::optional<at::Tensor> & gradc_opt, const at::Tensor & input, const at::Tensor & batch_sizes, const at::Tensor & weight, const at::Tensor & bias, const at::Tensor & init_h, const at::Tensor & init_c, const at::Tensor & y, const at::Tensor & h, const at::Tensor & c, const at::Tensor & i, const at::Tensor & j, const at::Tensor & f, const at::Tensor & o, const at::Tensor & tanhc, bool flag_direction) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_lstm_data_backward", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, bool)>();
    return op.call(grady_opt, gradh_opt, gradc_opt, input, batch_sizes, weight, bias, init_h, init_c, y, h, c, i, j, f, o, tanhc, flag_direction);
}
at::Tensor npu_masked_fill_range(const at::Tensor & self, const at::Tensor & start, const at::Tensor & end, const at::Tensor & value, int64_t axis) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_masked_fill_range", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t)>();
    return op.call(self, start, end, value, axis);
}
at::Tensor npu_masked_softmax_with_rel_pos_bias(const at::Tensor & x, const ::std::optional<at::Tensor> & atten_mask, const at::Tensor & relative_pos_bias, double scale_value, int64_t inner_precision_mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_masked_softmax_with_rel_pos_bias", "").typed<at::Tensor (const at::Tensor &, const ::std::optional<at::Tensor> &, const at::Tensor &, double, int64_t)>();
    return op.call(x, atten_mask, relative_pos_bias, scale_value, inner_precision_mode);
}
::std::tuple<at::Tensor,at::Tensor> npu_max(const at::Tensor & self, int64_t dim, bool keepdim) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_max", "dim").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, int64_t, bool)>();
    return op.call(self, dim, keepdim);
}
::std::tuple<at::Tensor,at::Tensor> npu_max(const at::Tensor & self, at::Dimname dim, bool keepdim) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_max", "names_dim").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, at::Dimname, bool)>();
    return op.call(self, dim, keepdim);
}
at::Tensor npu_max_backward(const at::Tensor & grad, int64_t dim, const at::Tensor & indices, c10::SymIntArrayRef sizes, bool keepdim) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_max_backward", "").typed<at::Tensor (const at::Tensor &, int64_t, const at::Tensor &, c10::SymIntArrayRef, bool)>();
    return op.call(grad, dim, indices, sizes, keepdim);
}
::std::tuple<at::Tensor,at::Tensor> npu_min(const at::Tensor & self, int64_t dim, bool keepdim) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_min", "dim").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, int64_t, bool)>();
    return op.call(self, dim, keepdim);
}
::std::tuple<at::Tensor,at::Tensor> npu_min(const at::Tensor & self, at::Dimname dim, bool keepdim) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_min", "names_dim").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, at::Dimname, bool)>();
    return op.call(self, dim, keepdim);
}
at::Tensor npu_min_backward(const at::Tensor & grad, int64_t dim, const at::Tensor & indices, c10::SymIntArrayRef sizes, bool keepdim) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_min_backward", "").typed<at::Tensor (const at::Tensor &, int64_t, const at::Tensor &, c10::SymIntArrayRef, bool)>();
    return op.call(grad, dim, indices, sizes, keepdim);
}
at::Tensor npu_mish(const at::Tensor & self) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_mish", "").typed<at::Tensor (const at::Tensor &)>();
    return op.call(self);
}
at::Tensor npu_mish_backward(const at::Tensor & grad, const at::Tensor & input) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_mish_backward", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &)>();
    return op.call(grad, input);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_mla_prolog(const at::Tensor & token_x, const at::Tensor & weight_dq, const at::Tensor & weight_uq_qr, const at::Tensor & weight_uk, const at::Tensor & weight_dkv_kr, const at::Tensor & rmsnorm_gamma_cq, const at::Tensor & rmsnorm_gamma_ckv, const at::Tensor & rope_sin, const at::Tensor & rope_cos, const at::Tensor & cache_index, const at::Tensor & kv_cache, const at::Tensor & kr_cache, const ::std::optional<at::Tensor> & dequant_scale_x, const ::std::optional<at::Tensor> & dequant_scale_w_dq, const ::std::optional<at::Tensor> & dequant_scale_w_uq_qr, const ::std::optional<at::Tensor> & dequant_scale_w_dkv_kr, const ::std::optional<at::Tensor> & quant_scale_ckv, const ::std::optional<at::Tensor> & quant_scale_ckr, const ::std::optional<at::Tensor> & smooth_scales_cq, double rmsnorm_epsilon_cq, double rmsnorm_epsilon_ckv, c10::string_view cache_mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_mla_prolog", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, double, double, c10::string_view)>();
    return op.call(token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uq_qr, dequant_scale_w_dkv_kr, quant_scale_ckv, quant_scale_ckr, smooth_scales_cq, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv, cache_mode);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_mla_prolog_v2(const at::Tensor & token_x, const at::Tensor & weight_dq, const at::Tensor & weight_uq_qr, const at::Tensor & weight_uk, const at::Tensor & weight_dkv_kr, const at::Tensor & rmsnorm_gamma_cq, const at::Tensor & rmsnorm_gamma_ckv, const at::Tensor & rope_sin, const at::Tensor & rope_cos, const at::Tensor & cache_index, const at::Tensor & kv_cache, const at::Tensor & kr_cache, const ::std::optional<at::Tensor> & dequant_scale_x, const ::std::optional<at::Tensor> & dequant_scale_w_dq, const ::std::optional<at::Tensor> & dequant_scale_w_uq_qr, const ::std::optional<at::Tensor> & dequant_scale_w_dkv_kr, const ::std::optional<at::Tensor> & quant_scale_ckv, const ::std::optional<at::Tensor> & quant_scale_ckr, const ::std::optional<at::Tensor> & smooth_scales_cq, double rmsnorm_epsilon_cq, double rmsnorm_epsilon_ckv, c10::string_view cache_mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_mla_prolog_v2", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, double, double, c10::string_view)>();
    return op.call(token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uq_qr, dequant_scale_w_dkv_kr, quant_scale_ckv, quant_scale_ckr, smooth_scales_cq, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv, cache_mode);
}
at::Tensor npu_mm_all_reduce_base(const at::Tensor & x1, const at::Tensor & x2, c10::string_view hcom, c10::string_view reduce_op, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & antiquant_scale, const ::std::optional<at::Tensor> & antiquant_offset, const ::std::optional<at::Tensor> & x3, const ::std::optional<at::Tensor> & dequant_scale, const ::std::optional<at::Tensor> & pertoken_scale, const ::std::optional<at::Tensor> & comm_quant_scale_1, const ::std::optional<at::Tensor> & comm_quant_scale_2, int64_t antiquant_group_size, int64_t comm_turn) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_mm_all_reduce_base", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, c10::string_view, c10::string_view, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, int64_t, int64_t)>();
    return op.call(x1, x2, hcom, reduce_op, bias, antiquant_scale, antiquant_offset, x3, dequant_scale, pertoken_scale, comm_quant_scale_1, comm_quant_scale_2, antiquant_group_size, comm_turn);
}
at::Tensor npu_mm_reduce_scatter_base(const at::Tensor & self, const at::Tensor & x2, c10::string_view hcom, int64_t world_size, c10::string_view reduce_op, const ::std::optional<at::Tensor> & bias, int64_t comm_turn) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_mm_reduce_scatter_base", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, c10::string_view, int64_t, c10::string_view, const ::std::optional<at::Tensor> &, int64_t)>();
    return op.call(self, x2, hcom, world_size, reduce_op, bias, comm_turn);
}
at::Tensor npu_moe_compute_expert_tokens(const at::Tensor & sorted_expert_for_source_row, int64_t num_expert) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_moe_compute_expert_tokens", "").typed<at::Tensor (const at::Tensor &, int64_t)>();
    return op.call(sorted_expert_for_source_row, num_expert);
}
at::Tensor npu_moe_finalize_routing(const at::Tensor & expanded_permuted_rows, const ::std::optional<at::Tensor> & skip1, const ::std::optional<at::Tensor> & skip2, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & scales, const at::Tensor & expanded_src_to_dst_row, const ::std::optional<at::Tensor> & export_for_source_row, ::std::optional<int64_t> drop_pad_mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_moe_finalize_routing", "").typed<at::Tensor (const at::Tensor &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const at::Tensor &, const ::std::optional<at::Tensor> &, ::std::optional<int64_t>)>();
    return op.call(expanded_permuted_rows, skip1, skip2, bias, scales, expanded_src_to_dst_row, export_for_source_row, drop_pad_mode);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_moe_gating_top_k_softmax(const at::Tensor & x, const ::std::optional<at::Tensor> & finished, int64_t k) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_moe_gating_top_k_softmax", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const ::std::optional<at::Tensor> &, int64_t)>();
    return op.call(x, finished, k);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_moe_gating_top_k(const at::Tensor & x, int64_t k, const ::std::optional<at::Tensor> & bias, int64_t k_group, int64_t group_count, int64_t group_select_mode, int64_t renorm, int64_t norm_type, bool out_flag, double routed_scaling_factor, double eps) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_moe_gating_top_k", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, int64_t, const ::std::optional<at::Tensor> &, int64_t, int64_t, int64_t, int64_t, int64_t, bool, double, double)>();
    return op.call(x, k, bias, k_group, group_count, group_select_mode, renorm, norm_type, out_flag, routed_scaling_factor, eps);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_moe_init_routing(const at::Tensor & x, const at::Tensor & row_idx, const at::Tensor & expert_idx, int64_t active_num) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_moe_init_routing", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t)>();
    return op.call(x, row_idx, expert_idx, active_num);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_moe_init_routing_v2(const at::Tensor & x, const at::Tensor & expert_idx, const ::std::optional<at::Tensor> & scale, const ::std::optional<at::Tensor> & offset, int64_t active_num, int64_t expert_capacity, int64_t expert_num, int64_t drop_pad_mode, int64_t expert_tokens_num_type, bool expert_tokens_num_flag, int64_t quant_mode, at::IntArrayRef active_expert_range, int64_t row_idx_type) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_moe_init_routing_v2", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, int64_t, int64_t, int64_t, int64_t, int64_t, bool, int64_t, at::IntArrayRef, int64_t)>();
    return op.call(x, expert_idx, scale, offset, active_num, expert_capacity, expert_num, drop_pad_mode, expert_tokens_num_type, expert_tokens_num_flag, quant_mode, active_expert_range, row_idx_type);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_grouped_matmul_swiglu_quant(const at::Tensor & x, const at::Tensor & weight, const at::Tensor & group_list, const at::Tensor & weight_scale, const at::Tensor & x_scale, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & offset) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_grouped_matmul_swiglu_quant", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &)>();
    return op.call(x, weight, group_list, weight_scale, x_scale, bias, offset);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_moe_distribute_dispatch(const at::Tensor & x, const at::Tensor & expert_ids, c10::string_view group_ep, int64_t ep_world_size, int64_t ep_rank_id, int64_t moe_expert_num, const ::std::optional<at::Tensor> & scales, const ::std::optional<at::Tensor> & x_active_mask, const ::std::optional<at::Tensor> & expert_scales, c10::string_view group_tp, int64_t tp_world_size, int64_t tp_rank_id, int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num, int64_t quant_mode, int64_t global_bs, int64_t expert_token_nums_type) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_moe_distribute_dispatch", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, c10::string_view, int64_t, int64_t, int64_t, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, c10::string_view, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t)>();
    return op.call(x, expert_ids, group_ep, ep_world_size, ep_rank_id, moe_expert_num, scales, x_active_mask, expert_scales, group_tp, tp_world_size, tp_rank_id, expert_shard_type, shared_expert_num, shared_expert_rank_num, quant_mode, global_bs, expert_token_nums_type);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_moe_distribute_dispatch_v2(const at::Tensor & x, const at::Tensor & expert_ids, c10::string_view group_ep, int64_t ep_world_size, int64_t ep_rank_id, int64_t moe_expert_num, const ::std::optional<at::Tensor> & scales, const ::std::optional<at::Tensor> & x_active_mask, const ::std::optional<at::Tensor> & expert_scales, c10::string_view group_tp, int64_t tp_world_size, int64_t tp_rank_id, int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num, int64_t quant_mode, int64_t global_bs, int64_t expert_token_nums_type) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_moe_distribute_dispatch_v2", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, c10::string_view, int64_t, int64_t, int64_t, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, c10::string_view, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t)>();
    return op.call(x, expert_ids, group_ep, ep_world_size, ep_rank_id, moe_expert_num, scales, x_active_mask, expert_scales, group_tp, tp_world_size, tp_rank_id, expert_shard_type, shared_expert_num, shared_expert_rank_num, quant_mode, global_bs, expert_token_nums_type);
}
at::Tensor npu_moe_distribute_combine(const at::Tensor & expand_x, const at::Tensor & expert_ids, const at::Tensor & expand_idx, const at::Tensor & ep_send_counts, const at::Tensor & expert_scales, c10::string_view group_ep, int64_t ep_world_size, int64_t ep_rank_id, int64_t moe_expert_num, const ::std::optional<at::Tensor> & tp_send_counts, const ::std::optional<at::Tensor> & x_active_mask, const ::std::optional<at::Tensor> & activation_scale, const ::std::optional<at::Tensor> & weight_scale, const ::std::optional<at::Tensor> & group_list, const ::std::optional<at::Tensor> & expand_scales, const ::std::optional<at::Tensor> & shared_expert_x, c10::string_view group_tp, int64_t tp_world_size, int64_t tp_rank_id, int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num, int64_t global_bs, int64_t out_dtype, int64_t comm_quant_mode, int64_t group_list_type) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_moe_distribute_combine", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, c10::string_view, int64_t, int64_t, int64_t, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, c10::string_view, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t)>();
    return op.call(expand_x, expert_ids, expand_idx, ep_send_counts, expert_scales, group_ep, ep_world_size, ep_rank_id, moe_expert_num, tp_send_counts, x_active_mask, activation_scale, weight_scale, group_list, expand_scales, shared_expert_x, group_tp, tp_world_size, tp_rank_id, expert_shard_type, shared_expert_num, shared_expert_rank_num, global_bs, out_dtype, comm_quant_mode, group_list_type);
}
at::Tensor npu_moe_distribute_combine_v2(const at::Tensor & expand_x, const at::Tensor & expert_ids, const at::Tensor & assist_info_for_combine, const at::Tensor & ep_send_counts, const at::Tensor & expert_scales, c10::string_view group_ep, int64_t ep_world_size, int64_t ep_rank_id, int64_t moe_expert_num, const ::std::optional<at::Tensor> & tp_send_counts, const ::std::optional<at::Tensor> & x_active_mask, const ::std::optional<at::Tensor> & expand_scales, const ::std::optional<at::Tensor> & shared_expert_x, c10::string_view group_tp, int64_t tp_world_size, int64_t tp_rank_id, int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num, int64_t global_bs, int64_t comm_quant_mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_moe_distribute_combine_v2", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, c10::string_view, int64_t, int64_t, int64_t, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, c10::string_view, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t)>();
    return op.call(expand_x, expert_ids, assist_info_for_combine, ep_send_counts, expert_scales, group_ep, ep_world_size, ep_rank_id, moe_expert_num, tp_send_counts, x_active_mask, expand_scales, shared_expert_x, group_tp, tp_world_size, tp_rank_id, expert_shard_type, shared_expert_num, shared_expert_rank_num, global_bs, comm_quant_mode);
}
at::Tensor _npu_distribute_barrier(const at::Tensor & x_ref, c10::string_view group, int64_t world_size) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::_npu_distribute_barrier", "").typed<at::Tensor (const at::Tensor &, c10::string_view, int64_t)>();
    return op.call(x_ref, group, world_size);
}
at::Tensor npu_moe_eplb_update_expert(const at::Tensor & expert_ids, const at::Tensor & eplb_table, int64_t local_rank_id, int64_t world_size, int64_t balance_mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_moe_eplb_update_expert", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, int64_t, int64_t, int64_t)>();
    return op.call(expert_ids, eplb_table, local_rank_id, world_size, balance_mode);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_moe_re_routing(const at::Tensor & tokens, const at::Tensor & expert_token_num_per_rank, const ::std::optional<at::Tensor> & per_token_scales, int64_t expert_token_num_type, int64_t idx_type) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_moe_re_routing", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, int64_t, int64_t)>();
    return op.call(tokens, expert_token_num_per_rank, per_token_scales, expert_token_num_type, idx_type);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_moe_distribute_combine_add_rms_norm(const at::Tensor & expand_x, const at::Tensor & expert_ids, const at::Tensor & expand_idx, const at::Tensor & ep_send_counts, const at::Tensor & expert_scales, const at::Tensor & residual_x, const at::Tensor & gamma, c10::string_view group_ep, int64_t ep_world_size, int64_t ep_rank_id, int64_t moe_expert_num, const ::std::optional<at::Tensor> & tp_send_counts, const ::std::optional<at::Tensor> & x_active_mask, const ::std::optional<at::Tensor> & activation_scale, const ::std::optional<at::Tensor> & weight_scale, const ::std::optional<at::Tensor> & group_list, const ::std::optional<at::Tensor> & expand_scales, const ::std::optional<at::Tensor> & shared_expert_x, c10::string_view group_tp, int64_t tp_world_size, int64_t tp_rank_id, int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num, int64_t global_bs, int64_t out_dtype, int64_t comm_quant_mode, int64_t group_list_type, c10::string_view comm_alg, double norm_eps) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_moe_distribute_combine_add_rms_norm", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, c10::string_view, int64_t, int64_t, int64_t, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, c10::string_view, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, c10::string_view, double)>();
    return op.call(expand_x, expert_ids, expand_idx, ep_send_counts, expert_scales, residual_x, gamma, group_ep, ep_world_size, ep_rank_id, moe_expert_num, tp_send_counts, x_active_mask, activation_scale, weight_scale, group_list, expand_scales, shared_expert_x, group_tp, tp_world_size, tp_rank_id, expert_shard_type, shared_expert_num, shared_expert_rank_num, global_bs, out_dtype, comm_quant_mode, group_list_type, comm_alg, norm_eps);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_multi_head_attention(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & query_weight, const at::Tensor & key_weight, const at::Tensor & value_weight, const at::Tensor & attn_mask, const at::Tensor & out_proj_weight, const ::std::optional<at::Tensor> & query_bias, const ::std::optional<at::Tensor> & key_bias, const ::std::optional<at::Tensor> & value_bias, const ::std::optional<at::Tensor> & out_proj_bias, const ::std::optional<at::Tensor> & dropout_mask, int64_t attn_head_num, int64_t attn_dim_per_head, int64_t src_len, int64_t tgt_len, double dropout_prob, bool softmax_use_float) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_multi_head_attention", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, int64_t, int64_t, int64_t, int64_t, double, bool)>();
    return op.call(query, key, value, query_weight, key_weight, value_weight, attn_mask, out_proj_weight, query_bias, key_bias, value_bias, out_proj_bias, dropout_mask, attn_head_num, attn_dim_per_head, src_len, tgt_len, dropout_prob, softmax_use_float);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_multi_head_attention_backward(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & query_weight, const at::Tensor & key_weight, const at::Tensor & value_weight, const at::Tensor & out_proj_weight, const ::std::optional<at::Tensor> & query_bias, const ::std::optional<at::Tensor> & key_bias, const ::std::optional<at::Tensor> & value_bias, const ::std::optional<at::Tensor> & out_proj_bias, const at::Tensor & query_res, const at::Tensor & key_res, const at::Tensor & value_res, const at::Tensor & attn_scores, const at::Tensor & attn_res, const at::Tensor & context, const at::Tensor & y_grad, const at::Tensor & dropout_mask, int64_t attn_head_num, int64_t attn_dim_per_head, int64_t src_len, int64_t tgt_len, double dropout_prob, bool softmax_use_float) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_multi_head_attention_backward", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, int64_t, int64_t, int64_t, double, bool)>();
    return op.call(query, key, value, query_weight, key_weight, value_weight, out_proj_weight, query_bias, key_bias, value_bias, out_proj_bias, query_res, key_res, value_res, attn_scores, attn_res, context, y_grad, dropout_mask, attn_head_num, attn_dim_per_head, src_len, tgt_len, dropout_prob, softmax_use_float);
}
::std::tuple<at::Tensor,at::Tensor,int64_t,int64_t,int64_t> npu_multi_head_attention_v2(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & atten_mask, const ::std::optional<at::Tensor> & alibi_mask, double scale, int64_t head_num, c10::string_view input_layout, double keep_prob, int64_t pre_tokens, int64_t next_tokens, bool gen_mask_parallel, bool sync) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_multi_head_attention_v2", "").typed<::std::tuple<at::Tensor,at::Tensor,int64_t,int64_t,int64_t> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, double, int64_t, c10::string_view, double, int64_t, int64_t, bool, bool)>();
    return op.call(query, key, value, atten_mask, alibi_mask, scale, head_num, input_layout, keep_prob, pre_tokens, next_tokens, gen_mask_parallel, sync);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_multi_head_attention_v2_grad(const at::Tensor & attention_score_grad, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & softmax_log_max_sum, const at::Tensor & attention_score, const ::std::optional<at::Tensor> & atten_mask, const ::std::optional<at::Tensor> & alibi_mask, double scale, int64_t head_num, c10::string_view input_layout, double keep_prob, int64_t pre_tokens, int64_t next_tokens, int64_t seed, int64_t offset, int64_t numels, bool gen_mask_parallel, bool sync) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_multi_head_attention_v2_grad", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, double, int64_t, c10::string_view, double, int64_t, int64_t, int64_t, int64_t, int64_t, bool, bool)>();
    return op.call(attention_score_grad, query, key, value, softmax_log_max_sum, attention_score, atten_mask, alibi_mask, scale, head_num, input_layout, keep_prob, pre_tokens, next_tokens, seed, offset, numels, gen_mask_parallel, sync);
}
::std::tuple<at::Tensor,at::Tensor> npu_nms_rotated(const at::Tensor & self, const at::Tensor & scores, double iou_threshold, double scores_threshold, int64_t max_output_size, int64_t mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_nms_rotated", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, double, double, int64_t, int64_t)>();
    return op.call(self, scores, iou_threshold, scores_threshold, max_output_size, mode);
}
::std::tuple<at::Tensor,at::Tensor> npu_nms_v4(const at::Tensor & self, const at::Tensor & scores, const at::Scalar & max_output_size, const at::Tensor & iou_threshold, const at::Tensor & scores_threshold, bool pad_to_max_output_size) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_nms_v4", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Scalar &, const at::Tensor &, const at::Tensor &, bool)>();
    return op.call(self, scores, max_output_size, iou_threshold, scores_threshold, pad_to_max_output_size);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_nms_with_mask(const at::Tensor & input, const at::Scalar & iou_threshold) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_nms_with_mask", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Scalar &)>();
    return op.call(input, iou_threshold);
}
at::Tensor npu_normalize_batch(const at::Tensor & self, const at::Tensor & seq_len, int64_t normalize_type) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_normalize_batch", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, int64_t)>();
    return op.call(self, seq_len, normalize_type);
}
at::Tensor npu_one_hot(const at::Tensor & self, int64_t num_classes, int64_t depth, const at::Scalar & on_value, const at::Scalar & off_value) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_one_hot", "").typed<at::Tensor (const at::Tensor &, int64_t, int64_t, const at::Scalar &, const at::Scalar &)>();
    return op.call(self, num_classes, depth, on_value, off_value);
}
at::Tensor npu_pad(const at::Tensor & input, at::IntArrayRef paddings) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_pad", "").typed<at::Tensor (const at::Tensor &, at::IntArrayRef)>();
    return op.call(input, paddings);
}
at::Tensor npu_prompt_flash_attention(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & padding_mask, const ::std::optional<at::Tensor> & atten_mask, const ::std::optional<at::Tensor> & pse_shift, at::OptionalIntArrayRef actual_seq_lengths, const ::std::optional<at::Tensor> & deq_scale1, const ::std::optional<at::Tensor> & quant_scale1, const ::std::optional<at::Tensor> & deq_scale2, const ::std::optional<at::Tensor> & quant_scale2, const ::std::optional<at::Tensor> & quant_offset2, int64_t num_heads, double scale_value, int64_t pre_tokens, int64_t next_tokens, c10::string_view input_layout, int64_t num_key_value_heads, at::OptionalIntArrayRef actual_seq_lengths_kv, int64_t sparse_mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_prompt_flash_attention", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, at::OptionalIntArrayRef, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, int64_t, double, int64_t, int64_t, c10::string_view, int64_t, at::OptionalIntArrayRef, int64_t)>();
    return op.call(query, key, value, padding_mask, atten_mask, pse_shift, actual_seq_lengths, deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2, num_heads, scale_value, pre_tokens, next_tokens, input_layout, num_key_value_heads, actual_seq_lengths_kv, sparse_mode);
}
at::Tensor npu_ps_roi_pooling(const at::Tensor & self, const at::Tensor & rois, double spatial_scale, int64_t group_size, int64_t output_dim) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_ps_roi_pooling", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, double, int64_t, int64_t)>();
    return op.call(self, rois, spatial_scale, group_size, output_dim);
}
at::Tensor npu_ps_roi_pooling_backward(const at::Tensor & output_grad, const at::Tensor & rois, double spatial_scale, int64_t group_size, int64_t output_dim, c10::SymIntArrayRef input_size) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_ps_roi_pooling_backward", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, double, int64_t, int64_t, c10::SymIntArrayRef)>();
    return op.call(output_grad, rois, spatial_scale, group_size, output_dim, input_size);
}
at::Tensor npu_ptiou(const at::Tensor & bboxes, const at::Tensor & gtboxes, int64_t mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_ptiou", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, int64_t)>();
    return op.call(bboxes, gtboxes, mode);
}
void npu_prefetch(const at::Tensor & self, const ::std::optional<at::Tensor> & dependency, int64_t max_size, int64_t offset) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_prefetch", "").typed<void (const at::Tensor &, const ::std::optional<at::Tensor> &, int64_t, int64_t)>();
    return op.call(self, dependency, max_size, offset);
}
at::Tensor npu_quant_conv2d(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & scale, at::IntArrayRef strides, at::IntArrayRef pads, at::IntArrayRef dilations, int64_t groups, int64_t offset_x, c10::string_view round_mode, ::std::optional<at::ScalarType> output_dtype, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & offset) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_quant_conv2d", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, int64_t, c10::string_view, ::std::optional<at::ScalarType>, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &)>();
    return op.call(input, weight, scale, strides, pads, dilations, groups, offset_x, round_mode, output_dtype, bias, offset);
}
at::Tensor npu_quant_matmul(const at::Tensor & x1, const at::Tensor & x2, const at::Tensor & scale, const ::std::optional<at::Tensor> & offset, const ::std::optional<at::Tensor> & pertoken_scale, const ::std::optional<at::Tensor> & bias, ::std::optional<at::ScalarType> output_dtype, at::OptionalSymIntArrayRef group_sizes) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_quant_matmul", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, ::std::optional<at::ScalarType>, at::OptionalSymIntArrayRef)>();
    return op.call(x1, x2, scale, offset, pertoken_scale, bias, output_dtype, group_sizes);
}
at::Tensor npu_quant_matmul_dequant(const at::Tensor & x, const at::Tensor & quantized_weight, const at::Tensor & weight_scale, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & x_scale, const ::std::optional<at::Tensor> & x_offset, const ::std::optional<at::Tensor> & smooth_scale, ::std::optional<c10::string_view> quant_mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_quant_matmul_dequant", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, ::std::optional<c10::string_view>)>();
    return op.call(x, quantized_weight, weight_scale, bias, x_scale, x_offset, smooth_scale, quant_mode);
}
at::Tensor npu_quant_grouped_matmul_dequant(const at::Tensor & x, const at::Tensor & quantized_weight, const at::Tensor & weight_scale, const at::Tensor & group_list, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & x_scale, const ::std::optional<at::Tensor> & x_offset, const ::std::optional<at::Tensor> & smooth_scale, ::std::optional<c10::string_view> quant_mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_quant_grouped_matmul_dequant", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, ::std::optional<c10::string_view>)>();
    return op.call(x, quantized_weight, weight_scale, group_list, bias, x_scale, x_offset, smooth_scale, quant_mode);
}
at::Tensor npu_quant_scatter(const at::Tensor & self, const at::Tensor & indices, const at::Tensor & updates, const at::Tensor & quant_scales, const ::std::optional<at::Tensor> & quant_zero_points, int64_t axis, int64_t quant_axis, c10::string_view reduce) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_quant_scatter", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, int64_t, int64_t, c10::string_view)>();
    return op.call(self, indices, updates, quant_scales, quant_zero_points, axis, quant_axis, reduce);
}
at::Tensor & npu_quant_scatter_(at::Tensor & self, const at::Tensor & indices, const at::Tensor & updates, const at::Tensor & quant_scales, const ::std::optional<at::Tensor> & quant_zero_points, int64_t axis, int64_t quant_axis, c10::string_view reduce) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_quant_scatter_", "").typed<at::Tensor & (at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, int64_t, int64_t, c10::string_view)>();
    return op.call(self, indices, updates, quant_scales, quant_zero_points, axis, quant_axis, reduce);
}
at::Tensor npu_quantize(const at::Tensor & self, const at::Tensor & scales, const ::std::optional<at::Tensor> & zero_points, at::ScalarType dtype, int64_t axis, bool div_mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_quantize", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, at::ScalarType, int64_t, bool)>();
    return op.call(self, scales, zero_points, dtype, axis, div_mode);
}
::std::tuple<at::Tensor,at::Tensor> npu_kronecker_quant(const at::Tensor & x, const at::Tensor & kronecker_p1, const at::Tensor & kronecker_p2, ::std::optional<double> clip_ratio, ::std::optional<at::ScalarType> dst_dtype) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_kronecker_quant", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, ::std::optional<double>, ::std::optional<at::ScalarType>)>();
    return op.call(x, kronecker_p1, kronecker_p2, clip_ratio, dst_dtype);
}
at::Tensor npu_group_quant(const at::Tensor & x, const at::Tensor & scale, const at::Tensor & group_index, const ::std::optional<at::Tensor> & offset, ::std::optional<at::ScalarType> dst_dtype) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_group_quant", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, ::std::optional<at::ScalarType>)>();
    return op.call(x, scale, group_index, offset, dst_dtype);
}
::std::tuple<at::Tensor,at::Tensor> npu_random_choice_with_mask(const at::Tensor & x, int64_t count, int64_t seed, int64_t seed2) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_random_choice_with_mask", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, int64_t, int64_t, int64_t)>();
    return op.call(x, count, seed, seed2);
}
at::Tensor npu_reshape(const at::Tensor & self, at::IntArrayRef shape, bool can_refresh) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_reshape", "").typed<at::Tensor (const at::Tensor &, at::IntArrayRef, bool)>();
    return op.call(self, shape, can_refresh);
}
at::Tensor & npu_reshape_out(const at::Tensor & self, at::IntArrayRef shape, bool can_refresh, at::Tensor & out) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_reshape", "out").typed<at::Tensor & (const at::Tensor &, at::IntArrayRef, bool, at::Tensor &)>();
    return op.call(self, shape, can_refresh, out);
}
::std::tuple<at::Tensor,at::Tensor> npu_rms_norm(const at::Tensor & self, const at::Tensor & gamma, double epsilon) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_rms_norm", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, double)>();
    return op.call(self, gamma, epsilon);
}
::std::tuple<at::Tensor,at::Tensor> npu_gemma_rms_norm(const at::Tensor & self, const at::Tensor & gamma, double epsilon) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_gemma_rms_norm", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, double)>();
    return op.call(self, gamma, epsilon);
}
::std::tuple<at::Tensor,at::Tensor> npu_rms_norm_backward(const at::Tensor & dy, const at::Tensor & self, const at::Tensor & gamma, const at::Tensor & rstd) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_rms_norm_backward", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &)>();
    return op.call(dy, self, gamma, rstd);
}
at::Tensor npu_roi_align(const at::Tensor & self, const at::Tensor & rois, double spatial_scale, int64_t pooled_height, int64_t pooled_width, int64_t sample_num, int64_t roi_end_mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_roi_align", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, double, int64_t, int64_t, int64_t, int64_t)>();
    return op.call(self, rois, spatial_scale, pooled_height, pooled_width, sample_num, roi_end_mode);
}
at::Tensor npu_roi_alignbk(const at::Tensor & self, const at::Tensor & rois, at::IntArrayRef xdiff_shape, int64_t pooled_width, int64_t pooled_height, double spatial_scale, int64_t sample_num, ::std::optional<int64_t> roi_end_mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_roi_alignbk", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, at::IntArrayRef, int64_t, int64_t, double, int64_t, ::std::optional<int64_t>)>();
    return op.call(self, rois, xdiff_shape, pooled_width, pooled_height, spatial_scale, sample_num, roi_end_mode);
}
at::Tensor npu_rotary_mul(const at::Tensor & self, const at::Tensor & r1, const at::Tensor & r2, c10::string_view rotary_mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_rotary_mul", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, c10::string_view)>();
    return op.call(self, r1, r2, rotary_mode);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_rotary_mul_backward(const at::Tensor & grad, const at::Tensor & self, const at::Tensor & r1, const at::Tensor & r2, c10::string_view rotary_mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_rotary_mul_backward", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, c10::string_view)>();
    return op.call(grad, self, r1, r2, rotary_mode);
}
::std::tuple<at::Tensor,at::Tensor> npu_mrope(const at::Tensor & positions, const at::Tensor & query, const at::Tensor & key, const at::Tensor & cos_sin_cache, int64_t head_size, at::OptionalIntArrayRef mrope_section, ::std::optional<c10::string_view> rotary_mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_mrope", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, at::OptionalIntArrayRef, ::std::optional<c10::string_view>)>();
    return op.call(positions, query, key, cos_sin_cache, head_size, mrope_section, rotary_mode);
}
at::Tensor npu_rotated_box_decode(const at::Tensor & self, const at::Tensor & deltas, const at::Tensor & weight) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_rotated_box_decode", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &)>();
    return op.call(self, deltas, weight);
}
at::Tensor npu_rotated_box_encode(const at::Tensor & self, const at::Tensor & gt_bboxes, const at::Tensor & weight) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_rotated_box_encode", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &)>();
    return op.call(self, gt_bboxes, weight);
}
at::Tensor npu_rotated_iou(const at::Tensor & self, const at::Tensor & query_boxes, bool trans, int64_t mode, bool is_cross, double v_threshold, double e_threshold) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_rotated_iou", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, bool, int64_t, bool, double, double)>();
    return op.call(self, query_boxes, trans, mode, is_cross, v_threshold, e_threshold);
}
at::Tensor npu_rotated_overlaps(const at::Tensor & self, const at::Tensor & query_boxes, bool trans) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_rotated_overlaps", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, bool)>();
    return op.call(self, query_boxes, trans);
}
at::Tensor npu_scaled_masked_softmax(const at::Tensor & x, const at::Tensor & mask, const at::Scalar & scale, bool fixed_triu_mask) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_scaled_masked_softmax", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Scalar &, bool)>();
    return op.call(x, mask, scale, fixed_triu_mask);
}
at::Tensor npu_scaled_masked_softmax_backward(const at::Tensor & y_grad, const at::Tensor & y, const at::Tensor & mask, const at::Scalar & scale, bool fixed_triu_mask) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_scaled_masked_softmax_backward", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar &, bool)>();
    return op.call(y_grad, y, mask, scale, fixed_triu_mask);
}
at::Tensor npu_scatter(const at::Tensor & self, const at::Tensor & indices, const at::Tensor & updates, int64_t dim) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_scatter", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t)>();
    return op.call(self, indices, updates, dim);
}
::std::vector<at::Tensor> npu_scatter_list(at::TensorList self, const at::Tensor & indices, const at::Tensor & updates, const ::std::optional<at::Tensor> & mask, c10::string_view reduce, int64_t axis) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_scatter_list", "").typed<::std::vector<at::Tensor> (at::TensorList, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, c10::string_view, int64_t)>();
    return op.call(self, indices, updates, mask, reduce, axis);
}
void npu_scatter_list_(at::TensorList self, const at::Tensor & indices, const at::Tensor & updates, const ::std::optional<at::Tensor> & mask, c10::string_view reduce, int64_t axis) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_scatter_list_", "").typed<void (at::TensorList, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, c10::string_view, int64_t)>();
    return op.call(self, indices, updates, mask, reduce, axis);
}
at::Tensor npu_scatter_nd_update(const at::Tensor & self, const at::Tensor & indices, const at::Tensor & updates) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_scatter_nd_update", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &)>();
    return op.call(self, indices, updates);
}
at::Tensor & npu_scatter_nd_update_(at::Tensor & self, const at::Tensor & indices, const at::Tensor & updates) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_scatter_nd_update_", "").typed<at::Tensor & (at::Tensor &, const at::Tensor &, const at::Tensor &)>();
    return op.call(self, indices, updates);
}
at::Tensor npu_sign_bits_pack(const at::Tensor & self, int64_t size) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_sign_bits_pack", "").typed<at::Tensor (const at::Tensor &, int64_t)>();
    return op.call(self, size);
}
at::Tensor npu_sign_bits_unpack(const at::Tensor & input, int64_t size, at::ScalarType dtype) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_sign_bits_unpack", "").typed<at::Tensor (const at::Tensor &, int64_t, at::ScalarType)>();
    return op.call(input, size, dtype);
}
at::Tensor npu_silu(const at::Tensor & self) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_silu", "").typed<at::Tensor (const at::Tensor &)>();
    return op.call(self);
}
at::Tensor & npu_silu_(at::Tensor & self) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_silu_", "").typed<at::Tensor & (at::Tensor &)>();
    return op.call(self);
}
at::Tensor npu_silu_backward(const at::Tensor & grad_output, const at::Tensor & x0, const at::Tensor & x1) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_silu_backward", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &)>();
    return op.call(grad_output, x0, x1);
}
at::Tensor npu_slice(const at::Tensor & self, at::IntArrayRef offsets, at::IntArrayRef size) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_slice", "").typed<at::Tensor (const at::Tensor &, at::IntArrayRef, at::IntArrayRef)>();
    return op.call(self, offsets, size);
}
at::Tensor & npu_slice_out(const at::Tensor & self, at::IntArrayRef offsets, at::IntArrayRef size, at::Tensor & out) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_slice", "out").typed<at::Tensor & (const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::Tensor &)>();
    return op.call(self, offsets, size, out);
}
at::Tensor npu_softmax_cross_entropy_with_logits(const at::Tensor & self, const at::Tensor & labels) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_softmax_cross_entropy_with_logits", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &)>();
    return op.call(self, labels);
}
at::Tensor npu_softmax_cross_entropy_with_logits_backward(const at::Tensor & grad, const at::Tensor & self, const at::Tensor & labels) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_softmax_cross_entropy_with_logits_backward", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &)>();
    return op.call(grad, self, labels);
}
at::Tensor npu_sort_v2(const at::Tensor & self, int64_t dim, bool descending) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_sort_v2", "").typed<at::Tensor (const at::Tensor &, int64_t, bool)>();
    return op.call(self, dim, descending);
}
at::Tensor & npu_sort_v2_out(const at::Tensor & self, int64_t dim, bool descending, at::Tensor & out) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_sort_v2", "out").typed<at::Tensor & (const at::Tensor &, int64_t, bool, at::Tensor &)>();
    return op.call(self, dim, descending, out);
}
at::Tensor npu_stride_add(const at::Tensor & self, const at::Tensor & other, const at::Scalar & offset1, const at::Scalar & offset2, const at::Scalar & c1_len) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_stride_add", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Scalar &, const at::Scalar &, const at::Scalar &)>();
    return op.call(self, other, offset1, offset2, c1_len);
}
at::Tensor npu_stride_copy(const at::Tensor & self, at::IntArrayRef shape, at::IntArrayRef stride, const at::Scalar & storage_offset) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_stride_copy", "").typed<at::Tensor (const at::Tensor &, at::IntArrayRef, at::IntArrayRef, const at::Scalar &)>();
    return op.call(self, shape, stride, storage_offset);
}
at::Tensor & npu_stride_copy_out(const at::Tensor & self, at::IntArrayRef shape, at::IntArrayRef stride, const at::Scalar & storage_offset, at::Tensor & out) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_stride_copy", "out").typed<at::Tensor & (const at::Tensor &, at::IntArrayRef, at::IntArrayRef, const at::Scalar &, at::Tensor &)>();
    return op.call(self, shape, stride, storage_offset, out);
}
at::Tensor npu_sub_sample(const at::Tensor & self, int64_t per_images, double positive_fraction) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_sub_sample", "").typed<at::Tensor (const at::Tensor &, int64_t, double)>();
    return op.call(self, per_images, positive_fraction);
}
at::Tensor npu_swiglu(const at::Tensor & self, int64_t dim) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_swiglu", "").typed<at::Tensor (const at::Tensor &, int64_t)>();
    return op.call(self, dim);
}
at::Tensor npu_swiglu_backward(const at::Tensor & grad_output, const at::Tensor & self, int64_t dim) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_swiglu_backward", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, int64_t)>();
    return op.call(grad_output, self, dim);
}
::std::tuple<at::Tensor,at::Tensor> npu_dequant_swiglu_quant(const at::Tensor & x, const ::std::optional<at::Tensor> & weight_scale, const ::std::optional<at::Tensor> & activation_scale, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & quant_scale, const ::std::optional<at::Tensor> & quant_offset, const ::std::optional<at::Tensor> & group_index, bool activate_left, int64_t quant_mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_dequant_swiglu_quant", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, bool, int64_t)>();
    return op.call(x, weight_scale, activation_scale, bias, quant_scale, quant_offset, group_index, activate_left, quant_mode);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_dequant_rope_quant_kvcache(const at::Tensor & x, const at::Tensor & cos, const at::Tensor & sin, const at::Tensor & k_cache, const at::Tensor & v_cache, const at::Tensor & indices, const at::Tensor & scale_k, const at::Tensor & scale_v, at::IntArrayRef size_splits, const ::std::optional<at::Tensor> & offset_k, const ::std::optional<at::Tensor> & offset_v, const ::std::optional<at::Tensor> & weight_scale, const ::std::optional<at::Tensor> & activation_scale, const ::std::optional<at::Tensor> & bias, int64_t quant_mode, c10::string_view input_layout, bool kv_output, c10::string_view cache_mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_dequant_rope_quant_kvcache", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, at::IntArrayRef, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, int64_t, c10::string_view, bool, c10::string_view)>();
    return op.call(x, cos, sin, k_cache, v_cache, indices, scale_k, scale_v, size_splits, offset_k, offset_v, weight_scale, activation_scale, bias, quant_mode, input_layout, kv_output, cache_mode);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_rope_quant_kvcache(const at::Tensor & x, const at::Tensor & cos, const at::Tensor & sin, const at::Tensor & k_cache, const at::Tensor & v_cache, const at::Tensor & indices, const at::Tensor & scale_k, const at::Tensor & scale_v, at::IntArrayRef size_splits, const ::std::optional<at::Tensor> & offset_k, const ::std::optional<at::Tensor> & offset_v, int64_t quant_mode, c10::string_view input_layout, bool kv_output, c10::string_view cache_mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_rope_quant_kvcache", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, at::IntArrayRef, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, int64_t, c10::string_view, bool, c10::string_view)>();
    return op.call(x, cos, sin, k_cache, v_cache, indices, scale_k, scale_v, size_splits, offset_k, offset_v, quant_mode, input_layout, kv_output, cache_mode);
}
at::Tensor npu_dequant_bias(const at::Tensor & x, const at::Tensor & weight_scale, const ::std::optional<at::Tensor> & activation_scale, const ::std::optional<at::Tensor> & bias, ::std::optional<at::ScalarType> output_dtype) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_dequant_bias", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, ::std::optional<at::ScalarType>)>();
    return op.call(x, weight_scale, activation_scale, bias, output_dtype);
}
at::Tensor npu_trans_quant_param(const at::Tensor & scale, const ::std::optional<at::Tensor> & offset, ::std::optional<int64_t> round_mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_trans_quant_param", "").typed<at::Tensor (const at::Tensor &, const ::std::optional<at::Tensor> &, ::std::optional<int64_t>)>();
    return op.call(scale, offset, round_mode);
}
at::Tensor npu_transpose(const at::Tensor & self, at::IntArrayRef perm, bool require_contiguous) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_transpose", "").typed<at::Tensor (const at::Tensor &, at::IntArrayRef, bool)>();
    return op.call(self, perm, require_contiguous);
}
at::Tensor & npu_transpose_out(const at::Tensor & self, at::IntArrayRef perm, bool require_contiguous, at::Tensor & out) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_transpose", "out").typed<at::Tensor & (const at::Tensor &, at::IntArrayRef, bool, at::Tensor &)>();
    return op.call(self, perm, require_contiguous, out);
}
at::Tensor & npu_view_copy(at::Tensor & self, const at::Tensor & other, bool non_blocking) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_view_copy", "").typed<at::Tensor & (at::Tensor &, const at::Tensor &, bool)>();
    return op.call(self, other, non_blocking);
}
at::Tensor npu_weight_quant_batchmatmul(const at::Tensor & x, const at::Tensor & weight, const at::Tensor & antiquant_scale, const ::std::optional<at::Tensor> & antiquant_offset, const ::std::optional<at::Tensor> & quant_scale, const ::std::optional<at::Tensor> & quant_offset, const ::std::optional<at::Tensor> & bias, int64_t antiquant_group_size, int64_t inner_precise) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_weight_quant_batchmatmul", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, int64_t, int64_t)>();
    return op.call(x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias, antiquant_group_size, inner_precise);
}
at::Tensor npu_transpose_batchmatmul(const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & scale, at::OptionalIntArrayRef perm_x1, at::OptionalIntArrayRef perm_x2, at::OptionalIntArrayRef perm_y, ::std::optional<int64_t> batch_split_factor) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_transpose_batchmatmul", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, at::OptionalIntArrayRef, at::OptionalIntArrayRef, at::OptionalIntArrayRef, ::std::optional<int64_t>)>();
    return op.call(input, weight, bias, scale, perm_x1, perm_x2, perm_y, batch_split_factor);
}
at::Tensor npu_yolo_boxes_encode(const at::Tensor & self, const at::Tensor & gt_bboxes, const at::Tensor & stride, bool performance_mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_yolo_boxes_encode", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, bool)>();
    return op.call(self, gt_bboxes, stride, performance_mode);
}
at::Tensor & one_(at::Tensor & self) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::one_", "").typed<at::Tensor & (at::Tensor &)>();
    return op.call(self);
}
at::Tensor repeat_interleave_backward_int(const at::Tensor & grad, const at::Tensor & self, c10::SymInt repeats, ::std::optional<int64_t> dim) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::repeat_interleave_backward_int", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, c10::SymInt, ::std::optional<int64_t>)>();
    return op.call(grad, self, repeats, dim);
}
at::Tensor repeat_interleave_backward_tensor(const at::Tensor & grad, const at::Tensor & self, const at::Tensor & repeats, ::std::optional<int64_t> dim) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::repeat_interleave_backward_tensor", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, ::std::optional<int64_t>)>();
    return op.call(grad, self, repeats, dim);
}
at::Tensor scatter_update(const at::Tensor & self, const at::Tensor & indices, const at::Tensor & updates, int64_t axis) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::scatter_update", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t)>();
    return op.call(self, indices, updates, axis);
}
at::Tensor & scatter_update_(at::Tensor & self, const at::Tensor & indices, const at::Tensor & updates, int64_t axis) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::scatter_update_", "").typed<at::Tensor & (at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t)>();
    return op.call(self, indices, updates, axis);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> slow_conv_transpose2d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, ::std::array<bool,3> output_mask) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::slow_conv_transpose2d_backward", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, ::std::array<bool,3>)>();
    return op.call(grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, output_mask);
}
at::Tensor stft_backward(const at::Tensor & grad_output, const at::Tensor & self, int64_t n_fft, ::std::optional<int64_t> hop_length, ::std::optional<int64_t> win_length, const ::std::optional<at::Tensor> & window, bool normalized, ::std::optional<bool> onesided, ::std::optional<bool> return_complex) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::stft_backward", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, int64_t, ::std::optional<int64_t>, ::std::optional<int64_t>, const ::std::optional<at::Tensor> &, bool, ::std::optional<bool>, ::std::optional<bool>)>();
    return op.call(grad_output, self, n_fft, hop_length, win_length, window, normalized, onesided, return_complex);
}
at::Tensor fft_r2c_backward(const at::Tensor & grad, at::IntArrayRef dim, int64_t normalization, bool onesided, int64_t last_dim_size) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::fft_r2c_backward", "").typed<at::Tensor (const at::Tensor &, at::IntArrayRef, int64_t, bool, int64_t)>();
    return op.call(grad, dim, normalization, onesided, last_dim_size);
}
at::Tensor fft_c2r_backward(const at::Tensor & grad, at::IntArrayRef dim, int64_t normalization) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::fft_c2r_backward", "").typed<at::Tensor (const at::Tensor &, at::IntArrayRef, int64_t)>();
    return op.call(grad, dim, normalization);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_cross_entropy_loss(const at::Tensor & input, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, c10::string_view reduction, int64_t ignore_index, double label_smoothing, double lse_square_scale_for_zloss, bool return_zloss) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_cross_entropy_loss", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, c10::string_view, int64_t, double, double, bool)>();
    return op.call(input, target, weight, reduction, ignore_index, label_smoothing, lse_square_scale_for_zloss, return_zloss);
}
at::Tensor npu_cross_entropy_loss_backward(const at::Tensor & grad_loss, const at::Tensor & log_prob, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & grad_zloss, const ::std::optional<at::Tensor> & lse_for_zloss, c10::string_view reduction, int64_t ignore_index, double label_smoothing, double lse_square_scale_for_zloss) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_cross_entropy_loss_backward", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, c10::string_view, int64_t, double, double)>();
    return op.call(grad_loss, log_prob, target, weight, grad_zloss, lse_for_zloss, reduction, ignore_index, label_smoothing, lse_square_scale_for_zloss);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_group_norm_swish(const at::Tensor & input, int64_t num_groups, const at::Tensor & weight, const at::Tensor & bias, ::std::optional<double> eps, ::std::optional<double> swish_scale) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_group_norm_swish", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, int64_t, const at::Tensor &, const at::Tensor &, ::std::optional<double>, ::std::optional<double>)>();
    return op.call(input, num_groups, weight, bias, eps, swish_scale);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_group_norm_swish_grad(const at::Tensor & grad, const at::Tensor & input, int64_t num_groups, const at::Tensor & weight, const at::Tensor & bias, const at::Tensor & mean, const at::Tensor & rstd, ::std::array<bool,3> grad_input_mask, ::std::optional<double> swish_scale) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_group_norm_swish_grad", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, int64_t, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, ::std::array<bool,3>, ::std::optional<double>)>();
    return op.call(grad, input, num_groups, weight, bias, mean, rstd, grad_input_mask, swish_scale);
}
void npu_advance_step_flashattn(at::Tensor & input_tokens, const at::Tensor & sampled_token_ids, at::Tensor & input_positions, at::Tensor & seq_lens, at::Tensor & slot_mapping, const at::Tensor & block_tables, int64_t num_seqs, int64_t num_queries, int64_t block_size) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_advance_step_flashattn", "").typed<void (at::Tensor &, const at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, const at::Tensor &, int64_t, int64_t, int64_t)>();
    return op.call(input_tokens, sampled_token_ids, input_positions, seq_lens, slot_mapping, block_tables, num_seqs, num_queries, block_size);
}
at::Tensor & npu_grouped_matmul_add_(at::Tensor & self, const at::Tensor & x, const at::Tensor & weight, const at::Tensor & group_list, bool transpose_x, bool transpose_weight, int64_t group_type) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_grouped_matmul_add_", "").typed<at::Tensor & (at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, bool, bool, int64_t)>();
    return op.call(self, x, weight, group_list, transpose_x, transpose_weight, group_type);
}
at::Tensor & npu_attn_softmax_(at::Tensor & self) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_attn_softmax_", "").typed<at::Tensor & (at::Tensor &)>();
    return op.call(self);
}
at::Tensor & npu_attn_softmax_backward_(at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & values) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_attn_softmax_backward_", "").typed<at::Tensor & (at::Tensor &, const at::Tensor &, const at::Tensor &)>();
    return op.call(self, grad_output, values);
}
at::Tensor npu_gather_sparse_index(const at::Tensor & input, const at::Tensor & index) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_gather_sparse_index", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &)>();
    return op.call(input, index);
}
at::Tensor npu_nsa_compress(const at::Tensor & input, const at::Tensor & weight, int64_t compress_block_size, int64_t compress_stride, at::OptionalIntArrayRef actual_seq_len) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_nsa_compress", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, int64_t, int64_t, at::OptionalIntArrayRef)>();
    return op.call(input, weight, compress_block_size, compress_stride, actual_seq_len);
}
::std::tuple<at::Tensor,at::Tensor> npu_nsa_compress_grad(const at::Tensor & grad, const at::Tensor & input, const at::Tensor & weight, int64_t compress_block_size, int64_t compress_stride, at::OptionalIntArrayRef actual_seq_len) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_nsa_compress_grad", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, int64_t, at::OptionalIntArrayRef)>();
    return op.call(grad, input, weight, compress_block_size, compress_stride, actual_seq_len);
}
at::Tensor & npu_nsa_compress_infer_out(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & slot_mapping, int64_t compress_block_size, int64_t compress_stride, int64_t page_block_size, const ::std::optional<at::Tensor> & block_table, at::OptionalIntArrayRef actual_seq_len, at::Tensor & cache) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_nsa_compress_infer", "cache").typed<at::Tensor & (const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, int64_t, int64_t, const ::std::optional<at::Tensor> &, at::OptionalIntArrayRef, at::Tensor &)>();
    return op.call(input, weight, slot_mapping, compress_block_size, compress_stride, page_block_size, block_table, actual_seq_len, cache);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_nsa_compress_attention(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, double scale_value, int64_t head_num, int64_t compress_block_size, int64_t compress_stride, int64_t select_block_size, int64_t select_block_count, const ::std::optional<at::Tensor> & topk_mask, const ::std::optional<at::Tensor> & atten_mask, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_cmp_seq_kvlen, at::OptionalIntArrayRef actual_sel_seq_kvlen) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_nsa_compress_attention", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, double, int64_t, int64_t, int64_t, int64_t, int64_t, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, at::OptionalIntArrayRef, at::OptionalIntArrayRef, at::OptionalIntArrayRef)>();
    return op.call(query, key, value, scale_value, head_num, compress_block_size, compress_stride, select_block_size, select_block_count, topk_mask, atten_mask, actual_seq_qlen, actual_cmp_seq_kvlen, actual_sel_seq_kvlen);
}
::std::tuple<at::Tensor,at::Tensor> npu_nsa_compress_attention_infer(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, double scale_value, int64_t head_num, int64_t key_value_head_num, int64_t select_block_size, int64_t select_block_count, int64_t page_block_size, int64_t compress_block_size, int64_t compress_stride, const ::std::optional<at::Tensor> & atten_mask, const ::std::optional<at::Tensor> & block_table, const ::std::optional<at::Tensor> & topk_mask, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_cmp_seq_kvlen, at::OptionalIntArrayRef actual_sel_seq_kvlen) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_nsa_compress_attention_infer", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, double, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, at::OptionalIntArrayRef, at::OptionalIntArrayRef, at::OptionalIntArrayRef)>();
    return op.call(query, key, value, scale_value, head_num, key_value_head_num, select_block_size, select_block_count, page_block_size, compress_block_size, compress_stride, atten_mask, block_table, topk_mask, actual_seq_qlen, actual_cmp_seq_kvlen, actual_sel_seq_kvlen);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_nsa_select_attention(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & topk_indices, double scale_value, int64_t head_num, int64_t select_block_size, int64_t select_block_count, const ::std::optional<at::Tensor> & atten_mask, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_seq_kvlen) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_nsa_select_attention", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, double, int64_t, int64_t, int64_t, const ::std::optional<at::Tensor> &, at::OptionalIntArrayRef, at::OptionalIntArrayRef)>();
    return op.call(query, key, value, topk_indices, scale_value, head_num, select_block_size, select_block_count, atten_mask, actual_seq_qlen, actual_seq_kvlen);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_nsa_select_attention_grad(const at::Tensor & grad, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & attention_out, const at::Tensor & softmax_max, const at::Tensor & softmax_sum, const at::Tensor & topk_indices, double scale_value, int64_t head_num, int64_t select_block_size, int64_t select_block_count, const ::std::optional<at::Tensor> & atten_mask, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_seq_kvlen) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_nsa_select_attention_grad", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, double, int64_t, int64_t, int64_t, const ::std::optional<at::Tensor> &, at::OptionalIntArrayRef, at::OptionalIntArrayRef)>();
    return op.call(grad, query, key, value, attention_out, softmax_max, softmax_sum, topk_indices, scale_value, head_num, select_block_size, select_block_count, atten_mask, actual_seq_qlen, actual_seq_kvlen);
}
at::Tensor npu_nsa_select_attention_infer(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & topk_indices, double scale_value, int64_t head_num, int64_t key_value_head_num, int64_t select_block_size, int64_t select_block_count, int64_t page_block_size, c10::string_view layout, const ::std::optional<at::Tensor> & atten_mask, const ::std::optional<at::Tensor> & block_table, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_seq_kvlen) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_nsa_select_attention_infer", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, double, int64_t, int64_t, int64_t, int64_t, int64_t, c10::string_view, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, at::OptionalIntArrayRef, at::OptionalIntArrayRef)>();
    return op.call(query, key, value, topk_indices, scale_value, head_num, key_value_head_num, select_block_size, select_block_count, page_block_size, layout, atten_mask, block_table, actual_seq_qlen, actual_seq_kvlen);
}
at::Tensor npu_top_k_top_p(const at::Tensor & logits, const at::Tensor & p, const at::Tensor & k) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_top_k_top_p", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &)>();
    return op.call(logits, p, k);
}
::std::tuple<at::Tensor,at::Tensor> npu_moe_token_permute(const at::Tensor & tokens, const at::Tensor & indices, ::std::optional<int64_t> num_out_tokens, bool padded_mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_moe_token_permute", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, ::std::optional<int64_t>, bool)>();
    return op.call(tokens, indices, num_out_tokens, padded_mode);
}
at::Tensor npu_moe_token_unpermute(const at::Tensor & permuted_tokens, const at::Tensor & sorted_indices, const ::std::optional<at::Tensor> & probs, bool padded_mode, at::OptionalIntArrayRef restore_shape) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_moe_token_unpermute", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, bool, at::OptionalIntArrayRef)>();
    return op.call(permuted_tokens, sorted_indices, probs, padded_mode, restore_shape);
}

}  // namespace custom_ops
}  // namespace native
}  // namespace at_npu
