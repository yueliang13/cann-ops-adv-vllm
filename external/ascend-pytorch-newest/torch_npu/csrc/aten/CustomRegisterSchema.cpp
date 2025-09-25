#include <c10/core/TensorImpl.h>
#include <c10/core/Allocator.h>
#include <ATen/DeviceGuard.h>
#include <ATen/NativeFunctions.h>
#include <ATen/MetaFunctions.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/Dispatch.h>
#include <c10/util/Half.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/util/Optional.h>
#include <ATen/Tensor.h>
#include <ATen/Functions.h>
#include <ATen/native/Resize.h>
#include <ATen/core/op_registration/adaption.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

#include <ATen/Config.h>
#include <torch/library.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/jit/frontend/tracer.h>

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/VariableType.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPURecovery.h"
#include "op_plugin/OpInterface.h"

namespace at_npu {

namespace native {

int64_t wrapper__npu_change_data_ptr(const at::Tensor & dst, const at::Tensor & src, int64_t index) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(dst);
      c10_npu::check_npu_tensor_is_safe(src);
  }
    // No device check
  const c10::OptionalDeviceGuard device_guard(device_of(dst));
  return at_npu::native::NPUNativeFunctions::npu_change_data_ptr(dst, src, index);
}

int64_t wrapper__get_npu_format(const at::Tensor & self) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__get_npu_format", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return at_npu::native::NPUNativeFunctions::get_npu_format(self);
}

at::Tensor wrapper_Tensor_npu_format_cast(const at::Tensor & self, const at::Tensor & dst) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(dst);
  }
    // No device check
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return at_npu::native::NPUNativeFunctions::npu_format_cast(self, dst);
}

at::Tensor & wrapper_acl_format_npu_format_cast_(at::Tensor & self, int64_t acl_format) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper_acl_format_npu_format_cast_", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return at_npu::native::NPUNativeFunctions::npu_format_cast_(self, acl_format);
}

at::Tensor & wrapper__npu_format_cast_(at::Tensor & self, const at::Tensor & src) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(src);
  }
    // No device check
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return at_npu::native::NPUNativeFunctions::npu_format_cast_(self, src);
}

at::Tensor wrapper__empty_with_format(at::IntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, int64_t acl_format, ::std::optional<int64_t> base_addr_aligned_kb) {

   // No unsafe tensor check
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  
  const c10::DeviceGuard device_guard(device_or_default(device));
  return at_npu::native::NPUNativeFunctions::empty_with_format(size, dtype, layout, device, pin_memory, acl_format, base_addr_aligned_kb);
}

at::Tensor wrapper__unsafe_empty_with_format(at::IntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, int64_t acl_format, bool keep_format) {

   // No unsafe tensor check
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  
  const c10::DeviceGuard device_guard(device_or_default(device));
  return at_npu::native::NPUNativeFunctions::unsafe_empty_with_format(size, dtype, layout, device, pin_memory, acl_format, keep_format);
}

at::Tensor wrapper_names_empty_with_format(at::IntArrayRef size, ::std::optional<at::DimnameList> names, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, int64_t acl_format) {

   // No unsafe tensor check
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  
  const c10::DeviceGuard device_guard(device_or_default(device));
  return at_npu::native::NPUNativeFunctions::empty_with_format(size, names, dtype, layout, device, pin_memory, acl_format);
}

at::Tensor & wrapper__copy_memory_(at::Tensor & self, const at::Tensor & src, bool non_blocking) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(src);
  }
    // No device check
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return at_npu::native::NPUNativeFunctions::copy_memory_(self, src, non_blocking);
}

int64_t wrapper__get_storage_size(const at::Tensor & self) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__get_storage_size", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return at_npu::native::NPUNativeFunctions::get_storage_size(self);
}

at::Tensor wrapper__npu_format_cast(const at::Tensor & self, int64_t acl_format) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_format_cast", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return at_npu::native::NPUNativeFunctions::npu_format_cast(self, acl_format);
}

at::Tensor wrapper___npu_format_cast(const at::Tensor & self, int64_t acl_format) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper___npu_format_cast", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return at_npu::native::NPUNativeFunctions::_npu_format_cast(self, acl_format);
}

at::Tensor wrapper__empty_with_swapped_memory(at::IntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Device> device) {

   // No unsafe tensor check
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning

  return at_npu::native::NPUNativeFunctions::empty_with_swapped_memory(size, dtype, device);
}

at::Tensor wrapper__npu_gather_backward(const at::Tensor & grad, c10::SymIntArrayRef self_size, int64_t dim, const at::Tensor & index, bool sparse_grad) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(grad);
      c10_npu::check_npu_tensor_is_safe(index);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad, "wrapper__npu_gather_backward", "grad");
  c10::impl::check_and_update_common_device(common_device, index, "wrapper__npu_gather_backward", "index");
  const c10::OptionalDeviceGuard device_guard(device_of(grad));
  return op_plugin::npu_gather_backward_symint(grad, self_size, dim, index, sparse_grad);
}

bool wrapper___amp_foreach_non_finite_check(at::TensorList scaled_grads) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(scaled_grads);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, scaled_grads, "wrapper___amp_foreach_non_finite_check", "scaled_grads");
  const c10::OptionalDeviceGuard device_guard(device_of(scaled_grads));
  return op_plugin::_amp_foreach_non_finite_check(scaled_grads);
}

at::Tensor wrapper__npu_gelu(const at::Tensor & self, c10::string_view approximate) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_gelu", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_gelu(self, approximate);
}

at::Tensor wrapper__npu_gelu_backward(const at::Tensor & grad_output, const at::Tensor & self, c10::string_view approximate) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(grad_output);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad_output, "wrapper__npu_gelu_backward", "grad_output");
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_gelu_backward", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_gelu_backward(grad_output, self, approximate);
}

::std::tuple<at::Tensor,at::Tensor> wrapper___conv_depthwise2d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, ::std::array<bool,2> output_mask) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(grad_output);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(weight);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad_output, "wrapper___conv_depthwise2d_backward", "grad_output");
  c10::impl::check_and_update_common_device(common_device, self, "wrapper___conv_depthwise2d_backward", "self");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper___conv_depthwise2d_backward", "weight");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::_conv_depthwise2d_backward(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
}

::std::tuple<at::Tensor,at::Tensor> wrapper___dropout_with_byte_mask(const at::Tensor & self, double p) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper___dropout_with_byte_mask", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::_dropout_with_byte_mask(self, p);
}

at::Tensor wrapper___dropout_with_byte_mask_backward(const at::Tensor & grad_output, const at::Tensor & mask, double p) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(grad_output);
      c10_npu::check_npu_tensor_is_safe(mask);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad_output, "wrapper___dropout_with_byte_mask_backward", "grad_output");
  c10::impl::check_and_update_common_device(common_device, mask, "wrapper___dropout_with_byte_mask_backward", "mask");
  const c10::OptionalDeviceGuard device_guard(device_of(grad_output));
  return op_plugin::_dropout_with_byte_mask_backward(grad_output, mask, p);
}

::std::tuple<at::Tensor,at::Tensor> wrapper___npu_ciou(const at::Tensor & self, const at::Tensor & gtboxes, bool trans, bool is_cross, int64_t mode, bool atan_sub_flag) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(gtboxes);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper___npu_ciou", "self");
  c10::impl::check_and_update_common_device(common_device, gtboxes, "wrapper___npu_ciou", "gtboxes");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::_npu_ciou(self, gtboxes, trans, is_cross, mode, atan_sub_flag);
}

::std::tuple<at::Tensor,at::Tensor> wrapper___npu_dropout(const at::Tensor & self, double p) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper___npu_dropout", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::_npu_dropout(self, p);
}

at::Tensor wrapper_Tensor__npu_dropout_gen_mask(const at::Tensor & self, at::IntArrayRef size, double p, int64_t seed, int64_t offset, ::std::optional<bool> parallel, ::std::optional<bool> sync) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper_Tensor__npu_dropout_gen_mask", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::_npu_dropout_gen_mask(self, size, p, seed, offset, parallel, sync);
}

at::Tensor wrapper___npu_silent_check(at::Tensor & input_grad, const at::Tensor & val, at::Tensor & pre_val, at::Tensor & min_val, at::Tensor & max_val, const at::Tensor & val_counter, int64_t c_min_steps, double c_thresh_l1, double c_coeff_l1, double c_thresh_l2, double c_coeff_l2) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(input_grad);
      c10_npu::check_npu_tensor_is_safe(val);
      c10_npu::check_npu_tensor_is_safe(pre_val);
      c10_npu::check_npu_tensor_is_safe(min_val);
      c10_npu::check_npu_tensor_is_safe(max_val);
      c10_npu::check_npu_tensor_is_safe(val_counter);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, input_grad, "wrapper___npu_silent_check", "input_grad");
  c10::impl::check_and_update_common_device(common_device, val, "wrapper___npu_silent_check", "val");
  c10::impl::check_and_update_common_device(common_device, pre_val, "wrapper___npu_silent_check", "pre_val");
  c10::impl::check_and_update_common_device(common_device, min_val, "wrapper___npu_silent_check", "min_val");
  c10::impl::check_and_update_common_device(common_device, max_val, "wrapper___npu_silent_check", "max_val");
  c10::impl::check_and_update_common_device(common_device, val_counter, "wrapper___npu_silent_check", "val_counter");
  const c10::OptionalDeviceGuard device_guard(device_of(input_grad));
  return op_plugin::_npu_silent_check(input_grad, val, pre_val, min_val, max_val, val_counter, c_min_steps, c_thresh_l1, c_coeff_l1, c_thresh_l2, c_coeff_l2);
}

at::Tensor wrapper___npu_silent_check_v2(const at::Tensor & val, at::Tensor & input_grad, at::Tensor & sfda, at::Tensor & step, int64_t c_min_steps, double c_thresh_l1, double c_coeff_l1, double c_thresh_l2, double c_coeff_l2, int64_t npu_asd_detect) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(val);
      c10_npu::check_npu_tensor_is_safe(input_grad);
      c10_npu::check_npu_tensor_is_safe(sfda);
      c10_npu::check_npu_tensor_is_safe(step);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, val, "wrapper___npu_silent_check_v2", "val");
  c10::impl::check_and_update_common_device(common_device, input_grad, "wrapper___npu_silent_check_v2", "input_grad");
  c10::impl::check_and_update_common_device(common_device, sfda, "wrapper___npu_silent_check_v2", "sfda");
  c10::impl::check_and_update_common_device(common_device, step, "wrapper___npu_silent_check_v2", "step");
  const c10::OptionalDeviceGuard device_guard(device_of(val));
  return op_plugin::_npu_silent_check_v2(val, input_grad, sfda, step, c_min_steps, c_thresh_l1, c_coeff_l1, c_thresh_l2, c_coeff_l2, npu_asd_detect);
}

at::Tensor wrapper___npu_silent_check_v3(const at::Tensor & val, at::Tensor & input_grad, at::Tensor & step, at::Tensor & max, at::Tensor & avg, double c_thresh_l1, double c_thresh_l2, double betal, int64_t npu_asd_detect) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(val);
      c10_npu::check_npu_tensor_is_safe(input_grad);
      c10_npu::check_npu_tensor_is_safe(step);
      c10_npu::check_npu_tensor_is_safe(max);
      c10_npu::check_npu_tensor_is_safe(avg);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, val, "wrapper___npu_silent_check_v3", "val");
  c10::impl::check_and_update_common_device(common_device, input_grad, "wrapper___npu_silent_check_v3", "input_grad");
  c10::impl::check_and_update_common_device(common_device, step, "wrapper___npu_silent_check_v3", "step");
  c10::impl::check_and_update_common_device(common_device, max, "wrapper___npu_silent_check_v3", "max");
  c10::impl::check_and_update_common_device(common_device, avg, "wrapper___npu_silent_check_v3", "avg");
  const c10::OptionalDeviceGuard device_guard(device_of(val));
  return op_plugin::_npu_silent_check_v3(val, input_grad, step, max, avg, c_thresh_l1, c_thresh_l2, betal, npu_asd_detect);
}

::std::tuple<at::Tensor,at::Tensor> wrapper__batch_norm_gather_stats_update(const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, double momentum, double eps, const at::Tensor & counts) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(mean);
      c10_npu::check_npu_tensor_is_safe(invstd);
      c10_npu::check_npu_tensor_is_safe(running_mean);
      c10_npu::check_npu_tensor_is_safe(running_var);
      c10_npu::check_npu_tensor_is_safe(counts);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__batch_norm_gather_stats_update", "input");
  c10::impl::check_and_update_common_device(common_device, mean, "wrapper__batch_norm_gather_stats_update", "mean");
  c10::impl::check_and_update_common_device(common_device, invstd, "wrapper__batch_norm_gather_stats_update", "invstd");
  c10::impl::check_and_update_common_device(common_device, running_mean, "wrapper__batch_norm_gather_stats_update", "running_mean");
  c10::impl::check_and_update_common_device(common_device, running_var, "wrapper__batch_norm_gather_stats_update", "running_var");
  c10::impl::check_and_update_common_device(common_device, counts, "wrapper__batch_norm_gather_stats_update", "counts");
  const c10::OptionalDeviceGuard device_guard(device_of(input));
  return op_plugin::batch_norm_gather_stats_update(input, mean, invstd, running_mean, running_var, momentum, eps, counts);
}

::std::tuple<at::Tensor,at::Tensor> wrapper__batch_norm_reduce(const at::Tensor & input, double eps) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(input);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__batch_norm_reduce", "input");
  const c10::OptionalDeviceGuard device_guard(device_of(input));
  return op_plugin::batch_norm_reduce(input, eps);
}

at::Tensor wrapper__dropout_with_byte_mask(const at::Tensor & self, double p, bool train) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__dropout_with_byte_mask", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::dropout_with_byte_mask(self, p, train);
}

at::Tensor wrapper__fast_gelu(const at::Tensor & self) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__fast_gelu", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::fast_gelu(self);
}

at::Tensor wrapper__kl_div_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, bool log_target) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(grad_output);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(target);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad_output, "wrapper__kl_div_backward", "grad_output");
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__kl_div_backward", "self");
  c10::impl::check_and_update_common_device(common_device, target, "wrapper__kl_div_backward", "target");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::kl_div_backward(grad_output, self, target, reduction, log_target);
}

at::Tensor wrapper__l1_loss_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(grad_output);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(target);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad_output, "wrapper__l1_loss_backward", "grad_output");
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__l1_loss_backward", "self");
  c10::impl::check_and_update_common_device(common_device, target, "wrapper__l1_loss_backward", "target");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::l1_loss_backward(grad_output, self, target, reduction);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapper__slow_conv_dilated2d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, ::std::array<bool,3> output_mask) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(grad_output);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(weight);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad_output, "wrapper__slow_conv_dilated2d_backward", "grad_output");
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__slow_conv_dilated2d_backward", "self");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__slow_conv_dilated2d_backward", "weight");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::slow_conv_dilated2d_backward(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapper__matmul_double_backward(const ::std::optional<at::Tensor> & grad_self, const ::std::optional<at::Tensor> & grad_other, const at::Tensor & grad_out, const at::Tensor & self, const at::Tensor & other, ::std::array<bool,3> mask) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(grad_self);
      c10_npu::check_npu_tensor_is_safe(grad_other);
      c10_npu::check_npu_tensor_is_safe(grad_out);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(other);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad_self, "wrapper__matmul_double_backward", "grad_self");
  c10::impl::check_and_update_common_device(common_device, grad_other, "wrapper__matmul_double_backward", "grad_other");
  c10::impl::check_and_update_common_device(common_device, grad_out, "wrapper__matmul_double_backward", "grad_out");
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__matmul_double_backward", "self");
  c10::impl::check_and_update_common_device(common_device, other, "wrapper__matmul_double_backward", "other");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::matmul_double_backward(grad_self, grad_other, grad_out, self, other, mask);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrapper__npu_add_layer_norm(const at::Tensor & x1, const at::Tensor & x2, const at::Tensor & gamma, const at::Tensor & beta, double epsilon, bool additional_output) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(x1);
      c10_npu::check_npu_tensor_is_safe(x2);
      c10_npu::check_npu_tensor_is_safe(gamma);
      c10_npu::check_npu_tensor_is_safe(beta);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, x1, "wrapper__npu_add_layer_norm", "x1");
  c10::impl::check_and_update_common_device(common_device, x2, "wrapper__npu_add_layer_norm", "x2");
  c10::impl::check_and_update_common_device(common_device, gamma, "wrapper__npu_add_layer_norm", "gamma");
  c10::impl::check_and_update_common_device(common_device, beta, "wrapper__npu_add_layer_norm", "beta");
  const c10::OptionalDeviceGuard device_guard(device_of(x1));
  return op_plugin::npu_add_layer_norm(x1, x2, gamma, beta, epsilon, additional_output);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrapper__npu_add_layer_norm_backward(const ::std::optional<at::Tensor> & dy_opt, const at::Tensor & x1, const at::Tensor & x2, const at::Tensor & rstd, const at::Tensor & mean, const at::Tensor & gamma, const ::std::optional<at::Tensor> & dsum_opt) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(dy_opt);
      c10_npu::check_npu_tensor_is_safe(x1);
      c10_npu::check_npu_tensor_is_safe(x2);
      c10_npu::check_npu_tensor_is_safe(rstd);
      c10_npu::check_npu_tensor_is_safe(mean);
      c10_npu::check_npu_tensor_is_safe(gamma);
      c10_npu::check_npu_tensor_is_safe(dsum_opt);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, dy_opt, "wrapper__npu_add_layer_norm_backward", "dy_opt");
  c10::impl::check_and_update_common_device(common_device, x1, "wrapper__npu_add_layer_norm_backward", "x1");
  c10::impl::check_and_update_common_device(common_device, x2, "wrapper__npu_add_layer_norm_backward", "x2");
  c10::impl::check_and_update_common_device(common_device, rstd, "wrapper__npu_add_layer_norm_backward", "rstd");
  c10::impl::check_and_update_common_device(common_device, mean, "wrapper__npu_add_layer_norm_backward", "mean");
  c10::impl::check_and_update_common_device(common_device, gamma, "wrapper__npu_add_layer_norm_backward", "gamma");
  c10::impl::check_and_update_common_device(common_device, dsum_opt, "wrapper__npu_add_layer_norm_backward", "dsum_opt");
  const c10::OptionalDeviceGuard device_guard(device_of(dy_opt));
  return op_plugin::npu_add_layer_norm_backward(dy_opt, x1, x2, rstd, mean, gamma, dsum_opt);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapper__npu_add_rms_norm(const at::Tensor & x1, const at::Tensor & x2, const at::Tensor & gamma, double epsilon) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(x1);
      c10_npu::check_npu_tensor_is_safe(x2);
      c10_npu::check_npu_tensor_is_safe(gamma);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, x1, "wrapper__npu_add_rms_norm", "x1");
  c10::impl::check_and_update_common_device(common_device, x2, "wrapper__npu_add_rms_norm", "x2");
  c10::impl::check_and_update_common_device(common_device, gamma, "wrapper__npu_add_rms_norm", "gamma");
  const c10::OptionalDeviceGuard device_guard(device_of(x1));
  return op_plugin::npu_add_rms_norm(x1, x2, gamma, epsilon);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrapper__npu_add_rms_norm_cast(const at::Tensor & x1, const at::Tensor & x2, const at::Tensor & gamma, double epsilon) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(x1);
      c10_npu::check_npu_tensor_is_safe(x2);
      c10_npu::check_npu_tensor_is_safe(gamma);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, x1, "wrapper__npu_add_rms_norm_cast", "x1");
  c10::impl::check_and_update_common_device(common_device, x2, "wrapper__npu_add_rms_norm_cast", "x2");
  c10::impl::check_and_update_common_device(common_device, gamma, "wrapper__npu_add_rms_norm_cast", "gamma");
  const c10::OptionalDeviceGuard device_guard(device_of(x1));
  return op_plugin::npu_add_rms_norm_cast(x1, x2, gamma, epsilon);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapper__npu_add_rms_norm_quant(const at::Tensor & x1, const at::Tensor & x2, const at::Tensor & gamma, const at::Tensor & scales1, const ::std::optional<at::Tensor> & zero_points1, const ::std::optional<at::Tensor> & scales2, const ::std::optional<at::Tensor> & zero_points2, int64_t axis, double epsilon, bool div_mode) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(x1);
      c10_npu::check_npu_tensor_is_safe(x2);
      c10_npu::check_npu_tensor_is_safe(gamma);
      c10_npu::check_npu_tensor_is_safe(scales1);
      c10_npu::check_npu_tensor_is_safe(zero_points1);
      c10_npu::check_npu_tensor_is_safe(scales2);
      c10_npu::check_npu_tensor_is_safe(zero_points2);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, x1, "wrapper__npu_add_rms_norm_quant", "x1");
  c10::impl::check_and_update_common_device(common_device, x2, "wrapper__npu_add_rms_norm_quant", "x2");
  c10::impl::check_and_update_common_device(common_device, gamma, "wrapper__npu_add_rms_norm_quant", "gamma");
  c10::impl::check_and_update_common_device(common_device, scales1, "wrapper__npu_add_rms_norm_quant", "scales1");
  c10::impl::check_and_update_common_device(common_device, zero_points1, "wrapper__npu_add_rms_norm_quant", "zero_points1");
  c10::impl::check_and_update_common_device(common_device, scales2, "wrapper__npu_add_rms_norm_quant", "scales2");
  c10::impl::check_and_update_common_device(common_device, zero_points2, "wrapper__npu_add_rms_norm_quant", "zero_points2");
  const c10::OptionalDeviceGuard device_guard(device_of(x1));
  return op_plugin::npu_add_rms_norm_quant(x1, x2, gamma, scales1, zero_points1, scales2, zero_points2, axis, epsilon, div_mode);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapper__npu_alltoallv_gmm(const at::Tensor & gmm_x, const at::Tensor & gmm_weight, c10::string_view hcom, int64_t ep_world_size, at::IntArrayRef send_counts, at::IntArrayRef recv_counts, const ::std::optional<at::Tensor> & send_counts_tensor, const ::std::optional<at::Tensor> & recv_counts_tensor, const ::std::optional<at::Tensor> & mm_x, const ::std::optional<at::Tensor> & mm_weight, bool trans_gmm_weight, bool trans_mm_weight, bool permute_out_flag) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(gmm_x);
      c10_npu::check_npu_tensor_is_safe(gmm_weight);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, gmm_x, "wrapper__npu_alltoallv_gmm", "gmm_x");
  c10::impl::check_and_update_common_device(common_device, gmm_weight, "wrapper__npu_alltoallv_gmm", "gmm_weight");
  c10::impl::check_and_update_common_device(common_device, send_counts_tensor, "wrapper__npu_alltoallv_gmm", "send_counts_tensor");
  c10::impl::check_and_update_common_device(common_device, recv_counts_tensor, "wrapper__npu_alltoallv_gmm", "recv_counts_tensor");
  c10::impl::check_and_update_common_device(common_device, mm_x, "wrapper__npu_alltoallv_gmm", "mm_x");
  c10::impl::check_and_update_common_device(common_device, mm_weight, "wrapper__npu_alltoallv_gmm", "mm_weight");
  const c10::OptionalDeviceGuard device_guard(device_of(gmm_x));
  return op_plugin::npu_alltoallv_gmm(gmm_x, gmm_weight, hcom, ep_world_size, send_counts, recv_counts, send_counts_tensor, recv_counts_tensor, mm_x, mm_weight, trans_gmm_weight, trans_mm_weight, permute_out_flag);
}

::std::tuple<at::Tensor,at::Tensor> wrapper__npu_gmm_alltoallv(const at::Tensor & gmm_x, const at::Tensor & gmm_weight, c10::string_view hcom, int64_t ep_world_size, at::IntArrayRef send_counts, at::IntArrayRef recv_counts, const ::std::optional<at::Tensor> & send_counts_tensor, const ::std::optional<at::Tensor> & recv_counts_tensor, const ::std::optional<at::Tensor> & mm_x, const ::std::optional<at::Tensor> & mm_weight, bool trans_gmm_weight, bool trans_mm_weight) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(gmm_x);
      c10_npu::check_npu_tensor_is_safe(gmm_weight);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, gmm_x, "wrapper__npu_gmm_alltoallv", "gmm_x");
  c10::impl::check_and_update_common_device(common_device, gmm_weight, "wrapper__npu_gmm_alltoallv", "gmm_weight");
  c10::impl::check_and_update_common_device(common_device, send_counts_tensor, "wrapper__npu_gmm_alltoallv", "send_counts_tensor");
  c10::impl::check_and_update_common_device(common_device, recv_counts_tensor, "wrapper__npu_gmm_alltoallv", "recv_counts_tensor");
  c10::impl::check_and_update_common_device(common_device, mm_x, "wrapper__npu_gmm_alltoallv", "mm_x");
  c10::impl::check_and_update_common_device(common_device, mm_weight, "wrapper__npu_gmm_alltoallv", "mm_weight");
  const c10::OptionalDeviceGuard device_guard(device_of(gmm_x));
  return op_plugin::npu_gmm_alltoallv(gmm_x, gmm_weight, hcom, ep_world_size, send_counts, recv_counts, send_counts_tensor, recv_counts_tensor, mm_x, mm_weight, trans_gmm_weight, trans_mm_weight);
}

::std::tuple<at::Tensor,at::Tensor> wrapper__npu_all_gather_base_mm(const at::Tensor & self, const at::Tensor & x2, c10::string_view hcom, int64_t world_size, const ::std::optional<at::Tensor> & bias, int64_t gather_index, bool gather_output, int64_t comm_turn) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(x2);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_all_gather_base_mm", "self");
  c10::impl::check_and_update_common_device(common_device, x2, "wrapper__npu_all_gather_base_mm", "x2");
  c10::impl::check_and_update_common_device(common_device, bias, "wrapper__npu_all_gather_base_mm", "bias");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_all_gather_base_mm(self, x2, hcom, world_size, bias, gather_index, gather_output, comm_turn);
}

at::Tensor wrapper__npu_alloc_float_status(const at::Tensor & self) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_alloc_float_status", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_alloc_float_status(self);
}

at::Tensor wrapper__npu_anchor_response_flags(const at::Tensor & self, at::IntArrayRef featmap_size, at::IntArrayRef stride, int64_t num_base_anchors) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_anchor_response_flags", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_anchor_response_flags(self, featmap_size, stride, num_base_anchors);
}

at::Tensor wrapper__npu_anti_quant(const at::Tensor & x, const at::Tensor & scale, const ::std::optional<at::Tensor> & offset, ::std::optional<at::ScalarType> dst_dtype, ::std::optional<at::ScalarType> src_dtype) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(x);
      c10_npu::check_npu_tensor_is_safe(scale);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, x, "wrapper__npu_anti_quant", "x");
  c10::impl::check_and_update_common_device(common_device, scale, "wrapper__npu_anti_quant", "scale");
  c10::impl::check_and_update_common_device(common_device, offset, "wrapper__npu_anti_quant", "offset");
  const c10::OptionalDeviceGuard device_guard(device_of(x));
  return op_plugin::npu_anti_quant(x, scale, offset, dst_dtype, src_dtype);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapper__npu_apply_adam(const at::Scalar & beta1_power, const at::Scalar & beta2_power, const at::Scalar & lr, const at::Scalar & beta1, const at::Scalar & beta2, const at::Scalar & epsilon, const at::Tensor & grad, ::std::optional<bool> use_locking, ::std::optional<bool> use_nesterov) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(grad);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad, "wrapper__npu_apply_adam", "grad");
  const c10::OptionalDeviceGuard device_guard(device_of(grad));
  return op_plugin::npu_apply_adam(beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, use_locking, use_nesterov);
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor> wrapper_out_npu_apply_adam_out(const at::Scalar & beta1_power,const at::Scalar & beta2_power,const at::Scalar & lr,const at::Scalar & beta1,const at::Scalar & beta2,const at::Scalar & epsilon,const at::Tensor & grad,::std::optional<bool> use_locking,::std::optional<bool> use_nesterov, at::TensorList out) {
  TORCH_CHECK(out.size() == 3, "expected tuple of 3 elements but got ", out.size(), OPS_ERROR(ErrCode::PARAM));
  at::Tensor var = out[0];
  at::Tensor m = out[1];
  at::Tensor v = out[2];
  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(var);
      c10_npu::check_npu_tensor_is_safe(m);
      c10_npu::check_npu_tensor_is_safe(v);
      c10_npu::check_npu_tensor_is_safe(grad);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, var, "wrapper_out_npu_apply_adam_out", "var");
  c10::impl::check_and_update_common_device(common_device, m, "wrapper_out_npu_apply_adam_out", "m");
  c10::impl::check_and_update_common_device(common_device, v, "wrapper_out_npu_apply_adam_out", "v");
  c10::impl::check_and_update_common_device(common_device, grad, "wrapper_out_npu_apply_adam_out", "grad");
  const c10::OptionalDeviceGuard device_guard(device_of(var));
  return op_plugin::npu_apply_adam_out(beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, use_locking, use_nesterov, var, m, v);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapper__npu_apply_adam_w(const at::Scalar & beta1_power, const at::Scalar & beta2_power, const at::Scalar & lr, const at::Scalar & weight_decay, const at::Scalar & beta1, const at::Scalar & beta2, const at::Scalar & epsilon, const at::Tensor & grad, const ::std::optional<at::Tensor> & max_grad_norm, ::std::optional<bool> amsgrad, ::std::optional<bool> maximize) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(grad);
      c10_npu::check_npu_tensor_is_safe(max_grad_norm);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad, "wrapper__npu_apply_adam_w", "grad");
  c10::impl::check_and_update_common_device(common_device, max_grad_norm, "wrapper__npu_apply_adam_w", "max_grad_norm");
  const c10::OptionalDeviceGuard device_guard(device_of(grad));
  return op_plugin::npu_apply_adam_w(beta1_power, beta2_power, lr, weight_decay, beta1, beta2, epsilon, grad, max_grad_norm, amsgrad, maximize);
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor> wrapper_out_npu_apply_adam_w_out(const at::Scalar & beta1_power,const at::Scalar & beta2_power,const at::Scalar & lr,const at::Scalar & weight_decay,const at::Scalar & beta1,const at::Scalar & beta2,const at::Scalar & epsilon,const at::Tensor & grad,const ::std::optional<at::Tensor> & max_grad_norm,::std::optional<bool> amsgrad,::std::optional<bool> maximize, at::TensorList out) {
  TORCH_CHECK(out.size() == 3, "expected tuple of 3 elements but got ", out.size(), OPS_ERROR(ErrCode::PARAM));
  at::Tensor var = out[0];
  at::Tensor m = out[1];
  at::Tensor v = out[2];
  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(var);
      c10_npu::check_npu_tensor_is_safe(m);
      c10_npu::check_npu_tensor_is_safe(v);
      c10_npu::check_npu_tensor_is_safe(grad);
      c10_npu::check_npu_tensor_is_safe(max_grad_norm);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, var, "wrapper_out_npu_apply_adam_w_out", "var");
  c10::impl::check_and_update_common_device(common_device, m, "wrapper_out_npu_apply_adam_w_out", "m");
  c10::impl::check_and_update_common_device(common_device, v, "wrapper_out_npu_apply_adam_w_out", "v");
  c10::impl::check_and_update_common_device(common_device, grad, "wrapper_out_npu_apply_adam_w_out", "grad");
  c10::impl::check_and_update_common_device(common_device, max_grad_norm, "wrapper_out_npu_apply_adam_w_out", "max_grad_norm");
  const c10::OptionalDeviceGuard device_guard(device_of(var));
  return op_plugin::npu_apply_adam_w_out(beta1_power, beta2_power, lr, weight_decay, beta1, beta2, epsilon, grad, max_grad_norm, amsgrad, maximize, var, m, v);
}

::std::tuple<at::Tensor,at::Tensor> wrapper__npu_apply_rotary_pos_emb(const at::Tensor & query, const at::Tensor & key, const at::Tensor & cos, const at::Tensor & sin, c10::string_view layout) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(query);
      c10_npu::check_npu_tensor_is_safe(key);
      c10_npu::check_npu_tensor_is_safe(cos);
      c10_npu::check_npu_tensor_is_safe(sin);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, query, "wrapper__npu_apply_rotary_pos_emb", "query");
  c10::impl::check_and_update_common_device(common_device, key, "wrapper__npu_apply_rotary_pos_emb", "key");
  c10::impl::check_and_update_common_device(common_device, cos, "wrapper__npu_apply_rotary_pos_emb", "cos");
  c10::impl::check_and_update_common_device(common_device, sin, "wrapper__npu_apply_rotary_pos_emb", "sin");
  const c10::OptionalDeviceGuard device_guard(device_of(query));
  return op_plugin::npu_apply_rotary_pos_emb(query, key, cos, sin, layout);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrapper__npu_kv_rmsnorm_rope_cache(const at::Tensor & kv, const at::Tensor & gamma, const at::Tensor & cos, const at::Tensor & sin, const at::Tensor & index, const at::Tensor & k_cache, const at::Tensor & ckv_cache, const ::std::optional<at::Tensor> & k_rope_scale, const ::std::optional<at::Tensor> & c_kv_scale, const ::std::optional<at::Tensor> & k_rope_offset, const ::std::optional<at::Tensor> & c_kv_offset, double epsilon, c10::string_view cache_mode, bool is_output_kv) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(kv);
      c10_npu::check_npu_tensor_is_safe(gamma);
      c10_npu::check_npu_tensor_is_safe(cos);
      c10_npu::check_npu_tensor_is_safe(sin);
      c10_npu::check_npu_tensor_is_safe(index);
      c10_npu::check_npu_tensor_is_safe(k_cache);
      c10_npu::check_npu_tensor_is_safe(ckv_cache);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, kv, "wrapper__npu_kv_rmsnorm_rope_cache", "kv");
  c10::impl::check_and_update_common_device(common_device, gamma, "wrapper__npu_kv_rmsnorm_rope_cache", "gamma");
  c10::impl::check_and_update_common_device(common_device, cos, "wrapper__npu_kv_rmsnorm_rope_cache", "cos");
  c10::impl::check_and_update_common_device(common_device, sin, "wrapper__npu_kv_rmsnorm_rope_cache", "sin");
  c10::impl::check_and_update_common_device(common_device, index, "wrapper__npu_kv_rmsnorm_rope_cache", "index");
  c10::impl::check_and_update_common_device(common_device, k_cache, "wrapper__npu_kv_rmsnorm_rope_cache", "k_cache");
  c10::impl::check_and_update_common_device(common_device, ckv_cache, "wrapper__npu_kv_rmsnorm_rope_cache", "ckv_cache");
  c10::impl::check_and_update_common_device(common_device, k_rope_scale, "wrapper__npu_kv_rmsnorm_rope_cache", "k_rope_scale");
  c10::impl::check_and_update_common_device(common_device, c_kv_scale, "wrapper__npu_kv_rmsnorm_rope_cache", "c_kv_scale");
  c10::impl::check_and_update_common_device(common_device, k_rope_offset, "wrapper__npu_kv_rmsnorm_rope_cache", "k_rope_offset");
  c10::impl::check_and_update_common_device(common_device, c_kv_offset, "wrapper__npu_kv_rmsnorm_rope_cache", "c_kv_offset");
  const c10::OptionalDeviceGuard device_guard(device_of(kv));
  return op_plugin::npu_kv_rmsnorm_rope_cache(kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope_scale, c_kv_scale, k_rope_offset, c_kv_offset, epsilon, cache_mode, is_output_kv);
}

at::Tensor wrapper__npu_batch_gather_matmul(const at::Tensor & self, const at::Tensor & x, const at::Tensor & weight_b, const at::Tensor & indices, const ::std::optional<at::Tensor> & weight_a, int64_t layer_idx, double scale, int64_t y_offset, int64_t y_slice_size) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(x);
      c10_npu::check_npu_tensor_is_safe(weight_b);
      c10_npu::check_npu_tensor_is_safe(indices);
      c10_npu::check_npu_tensor_is_safe(weight_a);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_batch_gather_matmul", "self");
  c10::impl::check_and_update_common_device(common_device, x, "wrapper__npu_batch_gather_matmul", "x");
  c10::impl::check_and_update_common_device(common_device, weight_b, "wrapper__npu_batch_gather_matmul", "weight_b");
  c10::impl::check_and_update_common_device(common_device, indices, "wrapper__npu_batch_gather_matmul", "indices");
  c10::impl::check_and_update_common_device(common_device, weight_a, "wrapper__npu_batch_gather_matmul", "weight_a");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_batch_gather_matmul(self, x, weight_b, indices, weight_a, layer_idx, scale, y_offset, y_slice_size);
}

at::Tensor & wrapper__npu_batch_gather_matmul_(at::Tensor & self, const at::Tensor & x, const at::Tensor & weight_b, const at::Tensor & indices, const ::std::optional<at::Tensor> & weight_a, int64_t layer_idx, double scale, int64_t y_offset, int64_t y_slice_size) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(x);
      c10_npu::check_npu_tensor_is_safe(weight_b);
      c10_npu::check_npu_tensor_is_safe(indices);
      c10_npu::check_npu_tensor_is_safe(weight_a);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_batch_gather_matmul_", "self");
  c10::impl::check_and_update_common_device(common_device, x, "wrapper__npu_batch_gather_matmul_", "x");
  c10::impl::check_and_update_common_device(common_device, weight_b, "wrapper__npu_batch_gather_matmul_", "weight_b");
  c10::impl::check_and_update_common_device(common_device, indices, "wrapper__npu_batch_gather_matmul_", "indices");
  c10::impl::check_and_update_common_device(common_device, weight_a, "wrapper__npu_batch_gather_matmul_", "weight_a");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_batch_gather_matmul_(self, x, weight_b, indices, weight_a, layer_idx, scale, y_offset, y_slice_size);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrapper__npu_batch_nms(const at::Tensor & self, const at::Tensor & scores, double score_threshold, double iou_threshold, int64_t max_size_per_class, int64_t max_total_size, bool change_coordinate_frame, bool transpose_box) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(scores);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_batch_nms", "self");
  c10::impl::check_and_update_common_device(common_device, scores, "wrapper__npu_batch_nms", "scores");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_batch_nms(self, scores, score_threshold, iou_threshold, max_size_per_class, max_total_size, change_coordinate_frame, transpose_box);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapper__npu_bert_apply_adam(const at::Scalar & lr, const at::Scalar & beta1, const at::Scalar & beta2, const at::Scalar & epsilon, const at::Tensor & grad, const at::Scalar & max_grad_norm, const at::Scalar & global_grad_norm, const at::Scalar & weight_decay, const ::std::optional<at::Scalar> & step_size, int64_t adam_mode) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(grad);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad, "wrapper__npu_bert_apply_adam", "grad");
  const c10::OptionalDeviceGuard device_guard(device_of(grad));
  return op_plugin::npu_bert_apply_adam(lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay, step_size, adam_mode);
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor> wrapper_out_npu_bert_apply_adam_out(const at::Scalar & lr,const at::Scalar & beta1,const at::Scalar & beta2,const at::Scalar & epsilon,const at::Tensor & grad,const at::Scalar & max_grad_norm,const at::Scalar & global_grad_norm,const at::Scalar & weight_decay,const ::std::optional<at::Scalar> & step_size,int64_t adam_mode, at::TensorList out) {
  TORCH_CHECK(out.size() == 3, "expected tuple of 3 elements but got ", out.size(), OPS_ERROR(ErrCode::PARAM));
  at::Tensor var = out[0];
  at::Tensor m = out[1];
  at::Tensor v = out[2];
  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(var);
      c10_npu::check_npu_tensor_is_safe(m);
      c10_npu::check_npu_tensor_is_safe(v);
      c10_npu::check_npu_tensor_is_safe(grad);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, var, "wrapper_out_npu_bert_apply_adam_out", "var");
  c10::impl::check_and_update_common_device(common_device, m, "wrapper_out_npu_bert_apply_adam_out", "m");
  c10::impl::check_and_update_common_device(common_device, v, "wrapper_out_npu_bert_apply_adam_out", "v");
  c10::impl::check_and_update_common_device(common_device, grad, "wrapper_out_npu_bert_apply_adam_out", "grad");
  const c10::OptionalDeviceGuard device_guard(device_of(var));
  return op_plugin::npu_bert_apply_adam_out(lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay, step_size, adam_mode, var, m, v);
}

at::Tensor wrapper__npu_binary_cross_entropy_with_logits_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight_opt, const ::std::optional<at::Tensor> & pos_weight_opt, int64_t reduction) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(grad_output);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(target);
      c10_npu::check_npu_tensor_is_safe(weight_opt);
      c10_npu::check_npu_tensor_is_safe(pos_weight_opt);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad_output, "wrapper__npu_binary_cross_entropy_with_logits_backward", "grad_output");
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_binary_cross_entropy_with_logits_backward", "self");
  c10::impl::check_and_update_common_device(common_device, target, "wrapper__npu_binary_cross_entropy_with_logits_backward", "target");
  c10::impl::check_and_update_common_device(common_device, weight_opt, "wrapper__npu_binary_cross_entropy_with_logits_backward", "weight_opt");
  c10::impl::check_and_update_common_device(common_device, pos_weight_opt, "wrapper__npu_binary_cross_entropy_with_logits_backward", "pos_weight_opt");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_binary_cross_entropy_with_logits_backward(grad_output, self, target, weight_opt, pos_weight_opt, reduction);
}

at::Tensor wrapper__npu_bmmV2(const at::Tensor & self, const at::Tensor & mat2, at::IntArrayRef output_sizes) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(mat2);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_bmmV2", "self");
  c10::impl::check_and_update_common_device(common_device, mat2, "wrapper__npu_bmmV2", "mat2");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_bmmV2(self, mat2, output_sizes);
}

at::Tensor wrapper__npu_bmm_v2_mat1_backward(const at::Tensor & grad, const at::Tensor & mat1, const at::Tensor & mat2, c10::SymIntArrayRef size) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(grad);
      c10_npu::check_npu_tensor_is_safe(mat1);
      c10_npu::check_npu_tensor_is_safe(mat2);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad, "wrapper__npu_bmm_v2_mat1_backward", "grad");
  c10::impl::check_and_update_common_device(common_device, mat1, "wrapper__npu_bmm_v2_mat1_backward", "mat1");
  c10::impl::check_and_update_common_device(common_device, mat2, "wrapper__npu_bmm_v2_mat1_backward", "mat2");
  const c10::OptionalDeviceGuard device_guard(device_of(grad));
  return op_plugin::npu_bmm_v2_mat1_backward_symint(grad, mat1, mat2, size);
}

at::Tensor wrapper__npu_bmm_v2_mat2_backward(const at::Tensor & grad, const at::Tensor & mat1, const at::Tensor & mat2, c10::SymIntArrayRef size) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(grad);
      c10_npu::check_npu_tensor_is_safe(mat1);
      c10_npu::check_npu_tensor_is_safe(mat2);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad, "wrapper__npu_bmm_v2_mat2_backward", "grad");
  c10::impl::check_and_update_common_device(common_device, mat1, "wrapper__npu_bmm_v2_mat2_backward", "mat1");
  c10::impl::check_and_update_common_device(common_device, mat2, "wrapper__npu_bmm_v2_mat2_backward", "mat2");
  const c10::OptionalDeviceGuard device_guard(device_of(grad));
  return op_plugin::npu_bmm_v2_mat2_backward_symint(grad, mat1, mat2, size);
}

at::Tensor wrapper__npu_bounding_box_decode(const at::Tensor & rois, const at::Tensor & deltas, double means0, double means1, double means2, double means3, double stds0, double stds1, double stds2, double stds3, at::IntArrayRef max_shape, double wh_ratio_clip) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(rois);
      c10_npu::check_npu_tensor_is_safe(deltas);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, rois, "wrapper__npu_bounding_box_decode", "rois");
  c10::impl::check_and_update_common_device(common_device, deltas, "wrapper__npu_bounding_box_decode", "deltas");
  const c10::OptionalDeviceGuard device_guard(device_of(rois));
  return op_plugin::npu_bounding_box_decode(rois, deltas, means0, means1, means2, means3, stds0, stds1, stds2, stds3, max_shape, wh_ratio_clip);
}

at::Tensor wrapper__npu_bounding_box_encode(const at::Tensor & anchor_box, const at::Tensor & ground_truth_box, double means0, double means1, double means2, double means3, double stds0, double stds1, double stds2, double stds3) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(anchor_box);
      c10_npu::check_npu_tensor_is_safe(ground_truth_box);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, anchor_box, "wrapper__npu_bounding_box_encode", "anchor_box");
  c10::impl::check_and_update_common_device(common_device, ground_truth_box, "wrapper__npu_bounding_box_encode", "ground_truth_box");
  const c10::OptionalDeviceGuard device_guard(device_of(anchor_box));
  return op_plugin::npu_bounding_box_encode(anchor_box, ground_truth_box, means0, means1, means2, means3, stds0, stds1, stds2, stds3);
}

at::Tensor wrapper__npu_broadcast(const at::Tensor & self, at::IntArrayRef size) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_broadcast", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_broadcast(self, size);
}

at::Tensor & wrapper_out_npu_broadcast_out(const at::Tensor & self, at::IntArrayRef size, at::Tensor & out) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(out);
      c10_npu::check_npu_tensor_is_safe(self);
  }
    // No device check
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_broadcast_out(self, size, out);
}

at::Tensor wrapper__npu_ciou(const at::Tensor & self, const at::Tensor & gtboxes, bool trans, bool is_cross, int64_t mode, bool atan_sub_flag) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(gtboxes);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_ciou", "self");
  c10::impl::check_and_update_common_device(common_device, gtboxes, "wrapper__npu_ciou", "gtboxes");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_ciou(self, gtboxes, trans, is_cross, mode, atan_sub_flag);
}

::std::tuple<at::Tensor,at::Tensor> wrapper__npu_ciou_backward(const at::Tensor & grad, const at::Tensor & bboxes, const at::Tensor & gtboxes, const ::std::optional<at::Tensor> & atan_sub, bool trans, bool is_cross, int64_t mode) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(grad);
      c10_npu::check_npu_tensor_is_safe(bboxes);
      c10_npu::check_npu_tensor_is_safe(gtboxes);
      c10_npu::check_npu_tensor_is_safe(atan_sub);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad, "wrapper__npu_ciou_backward", "grad");
  c10::impl::check_and_update_common_device(common_device, bboxes, "wrapper__npu_ciou_backward", "bboxes");
  c10::impl::check_and_update_common_device(common_device, gtboxes, "wrapper__npu_ciou_backward", "gtboxes");
  c10::impl::check_and_update_common_device(common_device, atan_sub, "wrapper__npu_ciou_backward", "atan_sub");
  const c10::OptionalDeviceGuard device_guard(device_of(grad));
  return op_plugin::npu_ciou_backward(grad, bboxes, gtboxes, atan_sub, trans, is_cross, mode);
}

at::Tensor wrapper__npu_clear_float_status(const at::Tensor & self, int64_t mode) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_clear_float_status", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_clear_float_status(self, mode);
}

at::Tensor wrapper__npu_confusion_transpose(const at::Tensor & self, at::IntArrayRef perm, at::IntArrayRef shape, bool transpose_first) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_confusion_transpose", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_confusion_transpose(self, perm, shape, transpose_first);
}

at::Tensor wrapper__npu_confusion_transpose_backward(const at::Tensor & grad, at::IntArrayRef perm, c10::SymIntArrayRef shape, bool transpose_first) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(grad);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad, "wrapper__npu_confusion_transpose_backward", "grad");
  const c10::OptionalDeviceGuard device_guard(device_of(grad));
  return op_plugin::npu_confusion_transpose_backward_symint(grad, perm, shape, transpose_first);
}

at::Tensor wrapper__npu_conv2d(const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(weight);
      c10_npu::check_npu_tensor_is_safe(bias);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_conv2d", "input");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__npu_conv2d", "weight");
  c10::impl::check_and_update_common_device(common_device, bias, "wrapper__npu_conv2d", "bias");
  const c10::OptionalDeviceGuard device_guard(device_of(input));
  return op_plugin::npu_conv2d(input, weight, bias, stride, padding, dilation, groups);
}

at::Tensor & wrapper_out_npu_conv2d_out(const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, at::Tensor & out) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(out);
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(weight);
      c10_npu::check_npu_tensor_is_safe(bias);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, out, "wrapper_out_npu_conv2d_out", "out");
  c10::impl::check_and_update_common_device(common_device, input, "wrapper_out_npu_conv2d_out", "input");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper_out_npu_conv2d_out", "weight");
  c10::impl::check_and_update_common_device(common_device, bias, "wrapper_out_npu_conv2d_out", "bias");
  const c10::OptionalDeviceGuard device_guard(device_of(out));
  return op_plugin::npu_conv2d_out(input, weight, bias, stride, padding, dilation, groups, out);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapper__npu_conv2d_backward(const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, ::std::array<bool,3> output_mask) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(grad_output);
      c10_npu::check_npu_tensor_is_safe(weight);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_conv2d_backward", "input");
  c10::impl::check_and_update_common_device(common_device, grad_output, "wrapper__npu_conv2d_backward", "grad_output");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__npu_conv2d_backward", "weight");
  const c10::OptionalDeviceGuard device_guard(device_of(input));
  return op_plugin::npu_conv2d_backward(input, grad_output, weight, stride, padding, dilation, groups, output_mask);
}

at::Tensor wrapper__npu_conv3d(const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(weight);
      c10_npu::check_npu_tensor_is_safe(bias);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_conv3d", "input");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__npu_conv3d", "weight");
  c10::impl::check_and_update_common_device(common_device, bias, "wrapper__npu_conv3d", "bias");
  const c10::OptionalDeviceGuard device_guard(device_of(input));
  return op_plugin::npu_conv3d(input, weight, bias, stride, padding, dilation, groups);
}

at::Tensor & wrapper_out_npu_conv3d_out(const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, at::Tensor & out) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(out);
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(weight);
      c10_npu::check_npu_tensor_is_safe(bias);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, out, "wrapper_out_npu_conv3d_out", "out");
  c10::impl::check_and_update_common_device(common_device, input, "wrapper_out_npu_conv3d_out", "input");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper_out_npu_conv3d_out", "weight");
  c10::impl::check_and_update_common_device(common_device, bias, "wrapper_out_npu_conv3d_out", "bias");
  const c10::OptionalDeviceGuard device_guard(device_of(out));
  return op_plugin::npu_conv3d_out(input, weight, bias, stride, padding, dilation, groups, out);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapper__npu_conv3d_backward(const at::Tensor & input, const at::Tensor & grad, const at::Tensor & weight, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, ::std::array<bool,3> output_mask) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(grad);
      c10_npu::check_npu_tensor_is_safe(weight);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_conv3d_backward", "input");
  c10::impl::check_and_update_common_device(common_device, grad, "wrapper__npu_conv3d_backward", "grad");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__npu_conv3d_backward", "weight");
  const c10::OptionalDeviceGuard device_guard(device_of(input));
  return op_plugin::npu_conv3d_backward(input, grad, weight, stride, padding, dilation, groups, output_mask);
}

at::Tensor wrapper__npu_conv_transpose2d(const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(weight);
      c10_npu::check_npu_tensor_is_safe(bias);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_conv_transpose2d", "input");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__npu_conv_transpose2d", "weight");
  c10::impl::check_and_update_common_device(common_device, bias, "wrapper__npu_conv_transpose2d", "bias");
  const c10::OptionalDeviceGuard device_guard(device_of(input));
  return op_plugin::npu_conv_transpose2d(input, weight, bias, padding, output_padding, stride, dilation, groups);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapper__npu_conv_transpose2d_backward(const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, ::std::array<bool,3> output_mask) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(grad_output);
      c10_npu::check_npu_tensor_is_safe(weight);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_conv_transpose2d_backward", "input");
  c10::impl::check_and_update_common_device(common_device, grad_output, "wrapper__npu_conv_transpose2d_backward", "grad_output");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__npu_conv_transpose2d_backward", "weight");
  const c10::OptionalDeviceGuard device_guard(device_of(input));
  return op_plugin::npu_conv_transpose2d_backward(input, grad_output, weight, padding, output_padding, stride, dilation, groups, output_mask);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapper__npu_conv_transpose3d_backward(const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, ::std::array<bool,3> output_mask) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(grad_output);
      c10_npu::check_npu_tensor_is_safe(weight);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_conv_transpose3d_backward", "input");
  c10::impl::check_and_update_common_device(common_device, grad_output, "wrapper__npu_conv_transpose3d_backward", "grad_output");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__npu_conv_transpose3d_backward", "weight");
  const c10::OptionalDeviceGuard device_guard(device_of(input));
  return op_plugin::npu_conv_transpose3d_backward(input, grad_output, weight, padding, output_padding, stride, dilation, groups, output_mask);
}

at::Tensor wrapper__npu_convert_weight_to_int4pack(const at::Tensor & weight, int64_t inner_k_tiles) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(weight);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__npu_convert_weight_to_int4pack", "weight");
  const c10::OptionalDeviceGuard device_guard(device_of(weight));
  return op_plugin::npu_convert_weight_to_int4pack(weight, inner_k_tiles);
}

at::Tensor wrapper__npu_convolution(const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(weight);
      c10_npu::check_npu_tensor_is_safe(bias);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_convolution", "input");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__npu_convolution", "weight");
  c10::impl::check_and_update_common_device(common_device, bias, "wrapper__npu_convolution", "bias");
  const c10::OptionalDeviceGuard device_guard(device_of(input));
  return op_plugin::npu_convolution(input, weight, bias, stride, padding, dilation, groups);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapper__npu_convolution_backward(const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, ::std::array<bool,3> output_mask) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(grad_output);
      c10_npu::check_npu_tensor_is_safe(weight);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_convolution_backward", "input");
  c10::impl::check_and_update_common_device(common_device, grad_output, "wrapper__npu_convolution_backward", "grad_output");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__npu_convolution_backward", "weight");
  const c10::OptionalDeviceGuard device_guard(device_of(input));
  return op_plugin::npu_convolution_backward(input, grad_output, weight, stride, padding, dilation, groups, output_mask);
}

at::Tensor wrapper__npu_convolution_transpose(const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(weight);
      c10_npu::check_npu_tensor_is_safe(bias);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_convolution_transpose", "input");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__npu_convolution_transpose", "weight");
  c10::impl::check_and_update_common_device(common_device, bias, "wrapper__npu_convolution_transpose", "bias");
  const c10::OptionalDeviceGuard device_guard(device_of(input));
  return op_plugin::npu_convolution_transpose(input, weight, bias, padding, output_padding, stride, dilation, groups);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapper__npu_convolution_transpose_backward(const at::Tensor & input, const at::Tensor & grad, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, ::std::array<bool,3> grad_input_mask) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(grad);
      c10_npu::check_npu_tensor_is_safe(weight);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_convolution_transpose_backward", "input");
  c10::impl::check_and_update_common_device(common_device, grad, "wrapper__npu_convolution_transpose_backward", "grad");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__npu_convolution_transpose_backward", "weight");
  const c10::OptionalDeviceGuard device_guard(device_of(input));
  return op_plugin::npu_convolution_transpose_backward(input, grad, weight, padding, output_padding, stride, dilation, groups, grad_input_mask);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapper__npu_deep_norm(const at::Tensor & x, const at::Tensor & gx, const at::Tensor & beta, const at::Tensor & gamma, double alpha, double epsilon) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(x);
      c10_npu::check_npu_tensor_is_safe(gx);
      c10_npu::check_npu_tensor_is_safe(beta);
      c10_npu::check_npu_tensor_is_safe(gamma);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, x, "wrapper__npu_deep_norm", "x");
  c10::impl::check_and_update_common_device(common_device, gx, "wrapper__npu_deep_norm", "gx");
  c10::impl::check_and_update_common_device(common_device, beta, "wrapper__npu_deep_norm", "beta");
  c10::impl::check_and_update_common_device(common_device, gamma, "wrapper__npu_deep_norm", "gamma");
  const c10::OptionalDeviceGuard device_guard(device_of(x));
  return op_plugin::npu_deep_norm(x, gx, beta, gamma, alpha, epsilon);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrapper__npu_deep_norm_backward(const at::Tensor & dy, const at::Tensor & x, const at::Tensor & gx, const at::Tensor & gamma, const at::Tensor & mean, const at::Tensor & rstd, double alpha) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(dy);
      c10_npu::check_npu_tensor_is_safe(x);
      c10_npu::check_npu_tensor_is_safe(gx);
      c10_npu::check_npu_tensor_is_safe(gamma);
      c10_npu::check_npu_tensor_is_safe(mean);
      c10_npu::check_npu_tensor_is_safe(rstd);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, dy, "wrapper__npu_deep_norm_backward", "dy");
  c10::impl::check_and_update_common_device(common_device, x, "wrapper__npu_deep_norm_backward", "x");
  c10::impl::check_and_update_common_device(common_device, gx, "wrapper__npu_deep_norm_backward", "gx");
  c10::impl::check_and_update_common_device(common_device, gamma, "wrapper__npu_deep_norm_backward", "gamma");
  c10::impl::check_and_update_common_device(common_device, mean, "wrapper__npu_deep_norm_backward", "mean");
  c10::impl::check_and_update_common_device(common_device, rstd, "wrapper__npu_deep_norm_backward", "rstd");
  const c10::OptionalDeviceGuard device_guard(device_of(dy));
  return op_plugin::npu_deep_norm_backward(dy, x, gx, gamma, mean, rstd, alpha);
}

::std::tuple<at::Tensor,at::Tensor> wrapper__npu_deformable_conv2d(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & offset, const ::std::optional<at::Tensor> & bias, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, int64_t deformable_groups, bool modulated) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(weight);
      c10_npu::check_npu_tensor_is_safe(offset);
      c10_npu::check_npu_tensor_is_safe(bias);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_deformable_conv2d", "input");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__npu_deformable_conv2d", "weight");
  c10::impl::check_and_update_common_device(common_device, offset, "wrapper__npu_deformable_conv2d", "offset");
  c10::impl::check_and_update_common_device(common_device, bias, "wrapper__npu_deformable_conv2d", "bias");
  const c10::OptionalDeviceGuard device_guard(device_of(input));
  return op_plugin::npu_deformable_conv2d(input, weight, offset, bias, kernel_size, stride, padding, dilation, groups, deformable_groups, modulated);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrapper__npu_deformable_conv2dbk(const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & offset_out, const at::Tensor & weight, const at::Tensor & offset, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, int64_t deformable_groups, bool modulated) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(grad_output);
      c10_npu::check_npu_tensor_is_safe(offset_out);
      c10_npu::check_npu_tensor_is_safe(weight);
      c10_npu::check_npu_tensor_is_safe(offset);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_deformable_conv2dbk", "input");
  c10::impl::check_and_update_common_device(common_device, grad_output, "wrapper__npu_deformable_conv2dbk", "grad_output");
  c10::impl::check_and_update_common_device(common_device, offset_out, "wrapper__npu_deformable_conv2dbk", "offset_out");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__npu_deformable_conv2dbk", "weight");
  c10::impl::check_and_update_common_device(common_device, offset, "wrapper__npu_deformable_conv2dbk", "offset");
  const c10::OptionalDeviceGuard device_guard(device_of(input));
  return op_plugin::npu_deformable_conv2dbk(input, grad_output, offset_out, weight, offset, kernel_size, stride, padding, dilation, groups, deformable_groups, modulated);
}

at::Tensor wrapper__npu_diou(const at::Tensor & self, const at::Tensor & gtboxes, bool trans, bool is_cross, int64_t mode) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(gtboxes);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_diou", "self");
  c10::impl::check_and_update_common_device(common_device, gtboxes, "wrapper__npu_diou", "gtboxes");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_diou(self, gtboxes, trans, is_cross, mode);
}

::std::tuple<at::Tensor,at::Tensor> wrapper__npu_diou_backward(const at::Tensor & grad, const at::Tensor & bboxes, const at::Tensor & gtboxes, bool trans, bool is_cross, int64_t mode) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(grad);
      c10_npu::check_npu_tensor_is_safe(bboxes);
      c10_npu::check_npu_tensor_is_safe(gtboxes);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad, "wrapper__npu_diou_backward", "grad");
  c10::impl::check_and_update_common_device(common_device, bboxes, "wrapper__npu_diou_backward", "bboxes");
  c10::impl::check_and_update_common_device(common_device, gtboxes, "wrapper__npu_diou_backward", "gtboxes");
  const c10::OptionalDeviceGuard device_guard(device_of(grad));
  return op_plugin::npu_diou_backward(grad, bboxes, gtboxes, trans, is_cross, mode);
}

at::Tensor wrapper__npu_dropout_backward(const at::Tensor & grad_output, const at::Tensor & mask, double p) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(grad_output);
      c10_npu::check_npu_tensor_is_safe(mask);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad_output, "wrapper__npu_dropout_backward", "grad_output");
  c10::impl::check_and_update_common_device(common_device, mask, "wrapper__npu_dropout_backward", "mask");
  const c10::OptionalDeviceGuard device_guard(device_of(grad_output));
  return op_plugin::npu_dropout_backward(grad_output, mask, p);
}

::std::tuple<at::Tensor,at::Tensor> wrapper__npu_dropout_do_mask(const at::Tensor & self, const at::Tensor & mask, double p) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(mask);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_dropout_do_mask", "self");
  c10::impl::check_and_update_common_device(common_device, mask, "wrapper__npu_dropout_do_mask", "mask");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_dropout_do_mask(self, mask, p);
}

at::Tensor wrapper__npu_dropout_gen_mask(at::IntArrayRef size, double p, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {

   // No unsafe tensor check
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  
  const c10::DeviceGuard device_guard(device_or_default(device));
  return op_plugin::npu_dropout_gen_mask(size, p, dtype, layout, device, pin_memory);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapper__npu_dropout_with_add_softmax(const at::Tensor & self, const at::Tensor & x1, const at::Scalar & alpha, double prob, int64_t dim) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(x1);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_dropout_with_add_softmax", "self");
  c10::impl::check_and_update_common_device(common_device, x1, "wrapper__npu_dropout_with_add_softmax", "x1");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_dropout_with_add_softmax(self, x1, alpha, prob, dim);
}

::std::tuple<at::Tensor,at::Tensor> wrapper__npu_dropout_with_add_softmax_backward(const at::Tensor & grad, const at::Tensor & mask, const at::Tensor & softmax_out, const at::Scalar & alpha, double prob, int64_t dim) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(grad);
      c10_npu::check_npu_tensor_is_safe(mask);
      c10_npu::check_npu_tensor_is_safe(softmax_out);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad, "wrapper__npu_dropout_with_add_softmax_backward", "grad");
  c10::impl::check_and_update_common_device(common_device, mask, "wrapper__npu_dropout_with_add_softmax_backward", "mask");
  c10::impl::check_and_update_common_device(common_device, softmax_out, "wrapper__npu_dropout_with_add_softmax_backward", "softmax_out");
  const c10::OptionalDeviceGuard device_guard(device_of(grad));
  return op_plugin::npu_dropout_with_add_softmax_backward(grad, mask, softmax_out, alpha, prob, dim);
}

at::Tensor wrapper__npu_dtype_cast(const at::Tensor & self, at::ScalarType dtype) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_dtype_cast", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_dtype_cast(self, dtype);
}

at::Tensor & wrapper__npu_dtype_cast_(at::Tensor & self, const at::Tensor & src) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(src);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_dtype_cast_", "self");
  c10::impl::check_and_update_common_device(common_device, src, "wrapper__npu_dtype_cast_", "src");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_dtype_cast_(self, src);
}

at::Tensor wrapper__npu_dtype_cast_backward(const at::Tensor & grad, at::ScalarType dtype) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(grad);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad, "wrapper__npu_dtype_cast_backward", "grad");
  const c10::OptionalDeviceGuard device_guard(device_of(grad));
  return op_plugin::npu_dtype_cast_backward(grad, dtype);
}

::std::tuple<at::Tensor,at::Tensor> wrapper__npu_dynamic_quant(const at::Tensor & input, const ::std::optional<at::Tensor> & smooth_scales, const ::std::optional<at::Tensor> & group_index, ::std::optional<at::ScalarType> dst_type) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(input);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_dynamic_quant", "input");
  c10::impl::check_and_update_common_device(common_device, smooth_scales, "wrapper__npu_dynamic_quant", "smooth_scales");
  c10::impl::check_and_update_common_device(common_device, group_index, "wrapper__npu_dynamic_quant", "group_index");
  const c10::OptionalDeviceGuard device_guard(device_of(input));
  return op_plugin::npu_dynamic_quant(input, smooth_scales, group_index, dst_type);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapper__npu_dynamic_quant_asymmetric(const at::Tensor & input, const ::std::optional<at::Tensor> & smooth_scales, const ::std::optional<at::Tensor> & group_index, ::std::optional<at::ScalarType> dst_type) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(input);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_dynamic_quant_asymmetric", "input");
  c10::impl::check_and_update_common_device(common_device, smooth_scales, "wrapper__npu_dynamic_quant_asymmetric", "smooth_scales");
  c10::impl::check_and_update_common_device(common_device, group_index, "wrapper__npu_dynamic_quant_asymmetric", "group_index");
  const c10::OptionalDeviceGuard device_guard(device_of(input));
  return op_plugin::npu_dynamic_quant_asymmetric(input, smooth_scales, group_index, dst_type);
}

at::Tensor wrapper__npu_fast_gelu(const at::Tensor & self) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_fast_gelu", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_fast_gelu(self);
}

at::Tensor wrapper__npu_fast_gelu_backward(const at::Tensor & grad, const at::Tensor & self) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(grad);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad, "wrapper__npu_fast_gelu_backward", "grad");
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_fast_gelu_backward", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_fast_gelu_backward(grad, self);
}

at::Tensor wrapper__npu_ffn(const at::Tensor & x, const at::Tensor & weight1, const at::Tensor & weight2, c10::string_view activation, at::OptionalIntArrayRef expert_tokens, at::OptionalIntArrayRef expert_tokens_index, const ::std::optional<at::Tensor> & bias1, const ::std::optional<at::Tensor> & bias2, const ::std::optional<at::Tensor> & scale, const ::std::optional<at::Tensor> & offset, const ::std::optional<at::Tensor> & deq_scale1, const ::std::optional<at::Tensor> & deq_scale2, const ::std::optional<at::Tensor> & antiquant_scale1, const ::std::optional<at::Tensor> & antiquant_scale2, const ::std::optional<at::Tensor> & antiquant_offset1, const ::std::optional<at::Tensor> & antiquant_offset2, ::std::optional<int64_t> inner_precise, ::std::optional<at::ScalarType> output_dtype) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(x);
      c10_npu::check_npu_tensor_is_safe(weight1);
      c10_npu::check_npu_tensor_is_safe(weight2);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, x, "wrapper__npu_ffn", "x");
  c10::impl::check_and_update_common_device(common_device, weight1, "wrapper__npu_ffn", "weight1");
  c10::impl::check_and_update_common_device(common_device, weight2, "wrapper__npu_ffn", "weight2");
  c10::impl::check_and_update_common_device(common_device, bias1, "wrapper__npu_ffn", "bias1");
  c10::impl::check_and_update_common_device(common_device, bias2, "wrapper__npu_ffn", "bias2");
  c10::impl::check_and_update_common_device(common_device, scale, "wrapper__npu_ffn", "scale");
  c10::impl::check_and_update_common_device(common_device, offset, "wrapper__npu_ffn", "offset");
  c10::impl::check_and_update_common_device(common_device, deq_scale1, "wrapper__npu_ffn", "deq_scale1");
  c10::impl::check_and_update_common_device(common_device, deq_scale2, "wrapper__npu_ffn", "deq_scale2");
  c10::impl::check_and_update_common_device(common_device, antiquant_scale1, "wrapper__npu_ffn", "antiquant_scale1");
  c10::impl::check_and_update_common_device(common_device, antiquant_scale2, "wrapper__npu_ffn", "antiquant_scale2");
  c10::impl::check_and_update_common_device(common_device, antiquant_offset1, "wrapper__npu_ffn", "antiquant_offset1");
  c10::impl::check_and_update_common_device(common_device, antiquant_offset2, "wrapper__npu_ffn", "antiquant_offset2");
  const c10::OptionalDeviceGuard device_guard(device_of(x));
  return op_plugin::npu_ffn(x, weight1, weight2, activation, expert_tokens, expert_tokens_index, bias1, bias2, scale, offset, deq_scale1, deq_scale2, antiquant_scale1, antiquant_scale2, antiquant_offset1, antiquant_offset2, inner_precise, output_dtype);
}

at::Tensor wrapper__npu_fused_attention_score(const at::Tensor & query_layer, const at::Tensor & key_layer, const at::Tensor & value_layer, const at::Tensor & attention_mask, const at::Scalar & scale, double keep_prob, bool query_transpose, bool key_transpose, bool bmm_score_transpose_a, bool bmm_score_transpose_b, bool value_transpose, bool dx_transpose) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(query_layer);
      c10_npu::check_npu_tensor_is_safe(key_layer);
      c10_npu::check_npu_tensor_is_safe(value_layer);
      c10_npu::check_npu_tensor_is_safe(attention_mask);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, query_layer, "wrapper__npu_fused_attention_score", "query_layer");
  c10::impl::check_and_update_common_device(common_device, key_layer, "wrapper__npu_fused_attention_score", "key_layer");
  c10::impl::check_and_update_common_device(common_device, value_layer, "wrapper__npu_fused_attention_score", "value_layer");
  c10::impl::check_and_update_common_device(common_device, attention_mask, "wrapper__npu_fused_attention_score", "attention_mask");
  const c10::OptionalDeviceGuard device_guard(device_of(query_layer));
  return op_plugin::npu_fused_attention_score(query_layer, key_layer, value_layer, attention_mask, scale, keep_prob, query_transpose, key_transpose, bmm_score_transpose_a, bmm_score_transpose_b, value_transpose, dx_transpose);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapper__npu_fused_attention_score_backward(const at::Tensor & grad_output, const at::Tensor & softmax_output, const at::Tensor & query_layer, const at::Tensor & key_layer, const at::Tensor & value_layer, const at::Tensor & mask, const at::Scalar & scale, double keep_prob, bool query_transpose, bool key_transpose, bool value_transpose, bool dx_transpose) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(grad_output);
      c10_npu::check_npu_tensor_is_safe(softmax_output);
      c10_npu::check_npu_tensor_is_safe(query_layer);
      c10_npu::check_npu_tensor_is_safe(key_layer);
      c10_npu::check_npu_tensor_is_safe(value_layer);
      c10_npu::check_npu_tensor_is_safe(mask);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad_output, "wrapper__npu_fused_attention_score_backward", "grad_output");
  c10::impl::check_and_update_common_device(common_device, softmax_output, "wrapper__npu_fused_attention_score_backward", "softmax_output");
  c10::impl::check_and_update_common_device(common_device, query_layer, "wrapper__npu_fused_attention_score_backward", "query_layer");
  c10::impl::check_and_update_common_device(common_device, key_layer, "wrapper__npu_fused_attention_score_backward", "key_layer");
  c10::impl::check_and_update_common_device(common_device, value_layer, "wrapper__npu_fused_attention_score_backward", "value_layer");
  c10::impl::check_and_update_common_device(common_device, mask, "wrapper__npu_fused_attention_score_backward", "mask");
  const c10::OptionalDeviceGuard device_guard(device_of(grad_output));
  return op_plugin::npu_fused_attention_score_backward(grad_output, softmax_output, query_layer, key_layer, value_layer, mask, scale, keep_prob, query_transpose, key_transpose, value_transpose, dx_transpose);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapper__npu_fused_attention_score_fwd(const at::Tensor & query_layer, const at::Tensor & key_layer, const at::Tensor & value_layer, const at::Tensor & attention_mask, const at::Scalar & scale, double keep_prob, bool query_transpose, bool key_transpose, bool bmm_score_transpose_a, bool bmm_score_transpose_b, bool value_transpose, bool dx_transpose) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(query_layer);
      c10_npu::check_npu_tensor_is_safe(key_layer);
      c10_npu::check_npu_tensor_is_safe(value_layer);
      c10_npu::check_npu_tensor_is_safe(attention_mask);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, query_layer, "wrapper__npu_fused_attention_score_fwd", "query_layer");
  c10::impl::check_and_update_common_device(common_device, key_layer, "wrapper__npu_fused_attention_score_fwd", "key_layer");
  c10::impl::check_and_update_common_device(common_device, value_layer, "wrapper__npu_fused_attention_score_fwd", "value_layer");
  c10::impl::check_and_update_common_device(common_device, attention_mask, "wrapper__npu_fused_attention_score_fwd", "attention_mask");
  const c10::OptionalDeviceGuard device_guard(device_of(query_layer));
  return op_plugin::npu_fused_attention_score_fwd(query_layer, key_layer, value_layer, attention_mask, scale, keep_prob, query_transpose, key_transpose, bmm_score_transpose_a, bmm_score_transpose_b, value_transpose, dx_transpose);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapper__npu_fused_attention_score_grad(const at::Tensor & grad_output, const at::Tensor & softmax_output, const at::Tensor & query_layer, const at::Tensor & key_layer, const at::Tensor & value_layer, const at::Tensor & mask, const at::Scalar & scale, double keep_prob, bool query_transpose, bool key_transpose, bool value_transpose, bool dx_transpose) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(grad_output);
      c10_npu::check_npu_tensor_is_safe(softmax_output);
      c10_npu::check_npu_tensor_is_safe(query_layer);
      c10_npu::check_npu_tensor_is_safe(key_layer);
      c10_npu::check_npu_tensor_is_safe(value_layer);
      c10_npu::check_npu_tensor_is_safe(mask);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad_output, "wrapper__npu_fused_attention_score_grad", "grad_output");
  c10::impl::check_and_update_common_device(common_device, softmax_output, "wrapper__npu_fused_attention_score_grad", "softmax_output");
  c10::impl::check_and_update_common_device(common_device, query_layer, "wrapper__npu_fused_attention_score_grad", "query_layer");
  c10::impl::check_and_update_common_device(common_device, key_layer, "wrapper__npu_fused_attention_score_grad", "key_layer");
  c10::impl::check_and_update_common_device(common_device, value_layer, "wrapper__npu_fused_attention_score_grad", "value_layer");
  c10::impl::check_and_update_common_device(common_device, mask, "wrapper__npu_fused_attention_score_grad", "mask");
  const c10::OptionalDeviceGuard device_guard(device_of(grad_output));
  return op_plugin::npu_fused_attention_score_grad(grad_output, softmax_output, query_layer, key_layer, value_layer, mask, scale, keep_prob, query_transpose, key_transpose, value_transpose, dx_transpose);
}

::std::tuple<at::Tensor,at::Tensor> wrapper__npu_fused_infer_attention_score(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & pse_shift, const ::std::optional<at::Tensor> & atten_mask, at::OptionalSymIntArrayRef actual_seq_lengths, at::OptionalSymIntArrayRef actual_seq_lengths_kv, const ::std::optional<at::Tensor> & dequant_scale1, const ::std::optional<at::Tensor> & quant_scale1, const ::std::optional<at::Tensor> & dequant_scale2, const ::std::optional<at::Tensor> & quant_scale2, const ::std::optional<at::Tensor> & quant_offset2, const ::std::optional<at::Tensor> & antiquant_scale, const ::std::optional<at::Tensor> & antiquant_offset, const ::std::optional<at::Tensor> & key_antiquant_scale, const ::std::optional<at::Tensor> & key_antiquant_offset, const ::std::optional<at::Tensor> & value_antiquant_scale, const ::std::optional<at::Tensor> & value_antiquant_offset, const ::std::optional<at::Tensor> & block_table, const ::std::optional<at::Tensor> & query_padding_size, const ::std::optional<at::Tensor> & kv_padding_size, const ::std::optional<at::Tensor> & key_shared_prefix, const ::std::optional<at::Tensor> & value_shared_prefix, at::OptionalSymIntArrayRef actual_shared_prefix_len, const ::std::optional<at::Tensor> & query_rope, const ::std::optional<at::Tensor> & key_rope, const ::std::optional<at::Tensor> & key_rope_antiquant_scale, int64_t num_heads, double scale, int64_t pre_tokens, int64_t next_tokens, c10::string_view input_layout, int64_t num_key_value_heads, int64_t sparse_mode, int64_t inner_precise, int64_t block_size, int64_t antiquant_mode, int64_t key_antiquant_mode, int64_t value_antiquant_mode, bool softmax_lse_flag) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(query);
      c10_npu::check_npu_tensor_is_safe(key);
      c10_npu::check_npu_tensor_is_safe(value);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, query, "wrapper__npu_fused_infer_attention_score", "query");
  c10::impl::check_and_update_common_device(common_device, key, "wrapper__npu_fused_infer_attention_score", "key");
  c10::impl::check_and_update_common_device(common_device, value, "wrapper__npu_fused_infer_attention_score", "value");
  c10::impl::check_and_update_common_device(common_device, pse_shift, "wrapper__npu_fused_infer_attention_score", "pse_shift");
  c10::impl::check_and_update_common_device(common_device, atten_mask, "wrapper__npu_fused_infer_attention_score", "atten_mask");
  c10::impl::check_and_update_common_device(common_device, dequant_scale1, "wrapper__npu_fused_infer_attention_score", "dequant_scale1");
  c10::impl::check_and_update_common_device(common_device, quant_scale1, "wrapper__npu_fused_infer_attention_score", "quant_scale1");
  c10::impl::check_and_update_common_device(common_device, dequant_scale2, "wrapper__npu_fused_infer_attention_score", "dequant_scale2");
  c10::impl::check_and_update_common_device(common_device, quant_scale2, "wrapper__npu_fused_infer_attention_score", "quant_scale2");
  c10::impl::check_and_update_common_device(common_device, quant_offset2, "wrapper__npu_fused_infer_attention_score", "quant_offset2");
  c10::impl::check_and_update_common_device(common_device, antiquant_scale, "wrapper__npu_fused_infer_attention_score", "antiquant_scale");
  c10::impl::check_and_update_common_device(common_device, antiquant_offset, "wrapper__npu_fused_infer_attention_score", "antiquant_offset");
  c10::impl::check_and_update_common_device(common_device, key_antiquant_scale, "wrapper__npu_fused_infer_attention_score", "key_antiquant_scale");
  c10::impl::check_and_update_common_device(common_device, key_antiquant_offset, "wrapper__npu_fused_infer_attention_score", "key_antiquant_offset");
  c10::impl::check_and_update_common_device(common_device, value_antiquant_scale, "wrapper__npu_fused_infer_attention_score", "value_antiquant_scale");
  c10::impl::check_and_update_common_device(common_device, value_antiquant_offset, "wrapper__npu_fused_infer_attention_score", "value_antiquant_offset");
  c10::impl::check_and_update_common_device(common_device, block_table, "wrapper__npu_fused_infer_attention_score", "block_table");
  c10::impl::check_and_update_common_device(common_device, query_padding_size, "wrapper__npu_fused_infer_attention_score", "query_padding_size");
  c10::impl::check_and_update_common_device(common_device, kv_padding_size, "wrapper__npu_fused_infer_attention_score", "kv_padding_size");
  c10::impl::check_and_update_common_device(common_device, key_shared_prefix, "wrapper__npu_fused_infer_attention_score", "key_shared_prefix");
  c10::impl::check_and_update_common_device(common_device, value_shared_prefix, "wrapper__npu_fused_infer_attention_score", "value_shared_prefix");
  c10::impl::check_and_update_common_device(common_device, query_rope, "wrapper__npu_fused_infer_attention_score", "query_rope");
  c10::impl::check_and_update_common_device(common_device, key_rope, "wrapper__npu_fused_infer_attention_score", "key_rope");
  c10::impl::check_and_update_common_device(common_device, key_rope_antiquant_scale, "wrapper__npu_fused_infer_attention_score", "key_rope_antiquant_scale");
  const c10::OptionalDeviceGuard device_guard(device_of(query));
  return op_plugin::npu_fused_infer_attention_score_symint(query, key, value, pse_shift, atten_mask, actual_seq_lengths, actual_seq_lengths_kv, dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2, antiquant_scale, antiquant_offset, key_antiquant_scale, key_antiquant_offset, value_antiquant_scale, value_antiquant_offset, block_table, query_padding_size, kv_padding_size, key_shared_prefix, value_shared_prefix, actual_shared_prefix_len, query_rope, key_rope, key_rope_antiquant_scale, num_heads, scale, pre_tokens, next_tokens, input_layout, num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, key_antiquant_mode, value_antiquant_mode, softmax_lse_flag);
}

::std::tuple<at::Tensor, at::Tensor> wrapper_out_npu_fused_infer_attention_score_out(const at::Tensor & query,const at::Tensor & key,const at::Tensor & value,const ::std::optional<at::Tensor> & pse_shift,const ::std::optional<at::Tensor> & atten_mask,at::OptionalSymIntArrayRef actual_seq_lengths,at::OptionalSymIntArrayRef actual_seq_lengths_kv,const ::std::optional<at::Tensor> & dequant_scale1,const ::std::optional<at::Tensor> & quant_scale1,const ::std::optional<at::Tensor> & dequant_scale2,const ::std::optional<at::Tensor> & quant_scale2,const ::std::optional<at::Tensor> & quant_offset2,const ::std::optional<at::Tensor> & antiquant_scale,const ::std::optional<at::Tensor> & antiquant_offset,const ::std::optional<at::Tensor> & key_antiquant_scale,const ::std::optional<at::Tensor> & key_antiquant_offset,const ::std::optional<at::Tensor> & value_antiquant_scale,const ::std::optional<at::Tensor> & value_antiquant_offset,const ::std::optional<at::Tensor> & block_table,const ::std::optional<at::Tensor> & query_padding_size,const ::std::optional<at::Tensor> & kv_padding_size,const ::std::optional<at::Tensor> & key_shared_prefix,const ::std::optional<at::Tensor> & value_shared_prefix,at::OptionalSymIntArrayRef actual_shared_prefix_len,const ::std::optional<at::Tensor> & query_rope,const ::std::optional<at::Tensor> & key_rope,const ::std::optional<at::Tensor> & key_rope_antiquant_scale,int64_t num_heads,double scale,int64_t pre_tokens,int64_t next_tokens,c10::string_view input_layout,int64_t num_key_value_heads,int64_t sparse_mode,int64_t inner_precise,int64_t block_size,int64_t antiquant_mode,int64_t key_antiquant_mode,int64_t value_antiquant_mode,bool softmax_lse_flag,const ::std::optional<at::Tensor> & workspace, at::TensorList out) {
  TORCH_CHECK(out.size() == 2, "expected tuple of 2 elements but got ", out.size(), OPS_ERROR(ErrCode::PARAM));
  at::Tensor attention_out = out[0];
  at::Tensor softmax_lse = out[1];
  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(attention_out);
      c10_npu::check_npu_tensor_is_safe(softmax_lse);
      c10_npu::check_npu_tensor_is_safe(query);
      c10_npu::check_npu_tensor_is_safe(key);
      c10_npu::check_npu_tensor_is_safe(value);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, attention_out, "wrapper_out_npu_fused_infer_attention_score_out", "attention_out");
  c10::impl::check_and_update_common_device(common_device, softmax_lse, "wrapper_out_npu_fused_infer_attention_score_out", "softmax_lse");
  c10::impl::check_and_update_common_device(common_device, query, "wrapper_out_npu_fused_infer_attention_score_out", "query");
  c10::impl::check_and_update_common_device(common_device, key, "wrapper_out_npu_fused_infer_attention_score_out", "key");
  c10::impl::check_and_update_common_device(common_device, value, "wrapper_out_npu_fused_infer_attention_score_out", "value");
  c10::impl::check_and_update_common_device(common_device, pse_shift, "wrapper_out_npu_fused_infer_attention_score_out", "pse_shift");
  c10::impl::check_and_update_common_device(common_device, atten_mask, "wrapper_out_npu_fused_infer_attention_score_out", "atten_mask");
  c10::impl::check_and_update_common_device(common_device, dequant_scale1, "wrapper_out_npu_fused_infer_attention_score_out", "dequant_scale1");
  c10::impl::check_and_update_common_device(common_device, quant_scale1, "wrapper_out_npu_fused_infer_attention_score_out", "quant_scale1");
  c10::impl::check_and_update_common_device(common_device, dequant_scale2, "wrapper_out_npu_fused_infer_attention_score_out", "dequant_scale2");
  c10::impl::check_and_update_common_device(common_device, quant_scale2, "wrapper_out_npu_fused_infer_attention_score_out", "quant_scale2");
  c10::impl::check_and_update_common_device(common_device, quant_offset2, "wrapper_out_npu_fused_infer_attention_score_out", "quant_offset2");
  c10::impl::check_and_update_common_device(common_device, antiquant_scale, "wrapper_out_npu_fused_infer_attention_score_out", "antiquant_scale");
  c10::impl::check_and_update_common_device(common_device, antiquant_offset, "wrapper_out_npu_fused_infer_attention_score_out", "antiquant_offset");
  c10::impl::check_and_update_common_device(common_device, key_antiquant_scale, "wrapper_out_npu_fused_infer_attention_score_out", "key_antiquant_scale");
  c10::impl::check_and_update_common_device(common_device, key_antiquant_offset, "wrapper_out_npu_fused_infer_attention_score_out", "key_antiquant_offset");
  c10::impl::check_and_update_common_device(common_device, value_antiquant_scale, "wrapper_out_npu_fused_infer_attention_score_out", "value_antiquant_scale");
  c10::impl::check_and_update_common_device(common_device, value_antiquant_offset, "wrapper_out_npu_fused_infer_attention_score_out", "value_antiquant_offset");
  c10::impl::check_and_update_common_device(common_device, block_table, "wrapper_out_npu_fused_infer_attention_score_out", "block_table");
  c10::impl::check_and_update_common_device(common_device, query_padding_size, "wrapper_out_npu_fused_infer_attention_score_out", "query_padding_size");
  c10::impl::check_and_update_common_device(common_device, kv_padding_size, "wrapper_out_npu_fused_infer_attention_score_out", "kv_padding_size");
  c10::impl::check_and_update_common_device(common_device, key_shared_prefix, "wrapper_out_npu_fused_infer_attention_score_out", "key_shared_prefix");
  c10::impl::check_and_update_common_device(common_device, value_shared_prefix, "wrapper_out_npu_fused_infer_attention_score_out", "value_shared_prefix");
  c10::impl::check_and_update_common_device(common_device, query_rope, "wrapper_out_npu_fused_infer_attention_score_out", "query_rope");
  c10::impl::check_and_update_common_device(common_device, key_rope, "wrapper_out_npu_fused_infer_attention_score_out", "key_rope");
  c10::impl::check_and_update_common_device(common_device, key_rope_antiquant_scale, "wrapper_out_npu_fused_infer_attention_score_out", "key_rope_antiquant_scale");
  c10::impl::check_and_update_common_device(common_device, workspace, "wrapper_out_npu_fused_infer_attention_score_out", "workspace");
  const c10::OptionalDeviceGuard device_guard(device_of(attention_out));
  return op_plugin::npu_fused_infer_attention_score_out_symint(query, key, value, pse_shift, atten_mask, actual_seq_lengths, actual_seq_lengths_kv, dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2, antiquant_scale, antiquant_offset, key_antiquant_scale, key_antiquant_offset, value_antiquant_scale, value_antiquant_offset, block_table, query_padding_size, kv_padding_size, key_shared_prefix, value_shared_prefix, actual_shared_prefix_len, query_rope, key_rope, key_rope_antiquant_scale, num_heads, scale, pre_tokens, next_tokens, input_layout, num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, key_antiquant_mode, value_antiquant_mode, softmax_lse_flag, workspace, attention_out, softmax_lse);
}

at::Tensor wrapper___npu_fused_infer_attention_score_get_max_workspace(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & pse_shift, const ::std::optional<at::Tensor> & atten_mask, at::OptionalSymIntArrayRef actual_seq_lengths, at::OptionalSymIntArrayRef actual_seq_lengths_kv, const ::std::optional<at::Tensor> & dequant_scale1, const ::std::optional<at::Tensor> & quant_scale1, const ::std::optional<at::Tensor> & dequant_scale2, const ::std::optional<at::Tensor> & quant_scale2, const ::std::optional<at::Tensor> & quant_offset2, const ::std::optional<at::Tensor> & antiquant_scale, const ::std::optional<at::Tensor> & antiquant_offset, const ::std::optional<at::Tensor> & key_antiquant_scale, const ::std::optional<at::Tensor> & key_antiquant_offset, const ::std::optional<at::Tensor> & value_antiquant_scale, const ::std::optional<at::Tensor> & value_antiquant_offset, const ::std::optional<at::Tensor> & block_table, const ::std::optional<at::Tensor> & query_padding_size, const ::std::optional<at::Tensor> & kv_padding_size, const ::std::optional<at::Tensor> & key_shared_prefix, const ::std::optional<at::Tensor> & value_shared_prefix, at::OptionalSymIntArrayRef actual_shared_prefix_len, const ::std::optional<at::Tensor> & query_rope, const ::std::optional<at::Tensor> & key_rope, const ::std::optional<at::Tensor> & key_rope_antiquant_scale, int64_t num_heads, double scale, int64_t pre_tokens, int64_t next_tokens, c10::string_view input_layout, int64_t num_key_value_heads, int64_t sparse_mode, int64_t inner_precise, int64_t block_size, int64_t antiquant_mode, int64_t key_antiquant_mode, int64_t value_antiquant_mode, bool softmax_lse_flag) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(query);
      c10_npu::check_npu_tensor_is_safe(key);
      c10_npu::check_npu_tensor_is_safe(value);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, query, "wrapper___npu_fused_infer_attention_score_get_max_workspace", "query");
  c10::impl::check_and_update_common_device(common_device, key, "wrapper___npu_fused_infer_attention_score_get_max_workspace", "key");
  c10::impl::check_and_update_common_device(common_device, value, "wrapper___npu_fused_infer_attention_score_get_max_workspace", "value");
  c10::impl::check_and_update_common_device(common_device, pse_shift, "wrapper___npu_fused_infer_attention_score_get_max_workspace", "pse_shift");
  c10::impl::check_and_update_common_device(common_device, atten_mask, "wrapper___npu_fused_infer_attention_score_get_max_workspace", "atten_mask");
  c10::impl::check_and_update_common_device(common_device, dequant_scale1, "wrapper___npu_fused_infer_attention_score_get_max_workspace", "dequant_scale1");
  c10::impl::check_and_update_common_device(common_device, quant_scale1, "wrapper___npu_fused_infer_attention_score_get_max_workspace", "quant_scale1");
  c10::impl::check_and_update_common_device(common_device, dequant_scale2, "wrapper___npu_fused_infer_attention_score_get_max_workspace", "dequant_scale2");
  c10::impl::check_and_update_common_device(common_device, quant_scale2, "wrapper___npu_fused_infer_attention_score_get_max_workspace", "quant_scale2");
  c10::impl::check_and_update_common_device(common_device, quant_offset2, "wrapper___npu_fused_infer_attention_score_get_max_workspace", "quant_offset2");
  c10::impl::check_and_update_common_device(common_device, antiquant_scale, "wrapper___npu_fused_infer_attention_score_get_max_workspace", "antiquant_scale");
  c10::impl::check_and_update_common_device(common_device, antiquant_offset, "wrapper___npu_fused_infer_attention_score_get_max_workspace", "antiquant_offset");
  c10::impl::check_and_update_common_device(common_device, key_antiquant_scale, "wrapper___npu_fused_infer_attention_score_get_max_workspace", "key_antiquant_scale");
  c10::impl::check_and_update_common_device(common_device, key_antiquant_offset, "wrapper___npu_fused_infer_attention_score_get_max_workspace", "key_antiquant_offset");
  c10::impl::check_and_update_common_device(common_device, value_antiquant_scale, "wrapper___npu_fused_infer_attention_score_get_max_workspace", "value_antiquant_scale");
  c10::impl::check_and_update_common_device(common_device, value_antiquant_offset, "wrapper___npu_fused_infer_attention_score_get_max_workspace", "value_antiquant_offset");
  c10::impl::check_and_update_common_device(common_device, block_table, "wrapper___npu_fused_infer_attention_score_get_max_workspace", "block_table");
  c10::impl::check_and_update_common_device(common_device, query_padding_size, "wrapper___npu_fused_infer_attention_score_get_max_workspace", "query_padding_size");
  c10::impl::check_and_update_common_device(common_device, kv_padding_size, "wrapper___npu_fused_infer_attention_score_get_max_workspace", "kv_padding_size");
  c10::impl::check_and_update_common_device(common_device, key_shared_prefix, "wrapper___npu_fused_infer_attention_score_get_max_workspace", "key_shared_prefix");
  c10::impl::check_and_update_common_device(common_device, value_shared_prefix, "wrapper___npu_fused_infer_attention_score_get_max_workspace", "value_shared_prefix");
  c10::impl::check_and_update_common_device(common_device, query_rope, "wrapper___npu_fused_infer_attention_score_get_max_workspace", "query_rope");
  c10::impl::check_and_update_common_device(common_device, key_rope, "wrapper___npu_fused_infer_attention_score_get_max_workspace", "key_rope");
  c10::impl::check_and_update_common_device(common_device, key_rope_antiquant_scale, "wrapper___npu_fused_infer_attention_score_get_max_workspace", "key_rope_antiquant_scale");
  const c10::OptionalDeviceGuard device_guard(device_of(query));
  return op_plugin::_npu_fused_infer_attention_score_get_max_workspace_symint(query, key, value, pse_shift, atten_mask, actual_seq_lengths, actual_seq_lengths_kv, dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2, antiquant_scale, antiquant_offset, key_antiquant_scale, key_antiquant_offset, value_antiquant_scale, value_antiquant_offset, block_table, query_padding_size, kv_padding_size, key_shared_prefix, value_shared_prefix, actual_shared_prefix_len, query_rope, key_rope, key_rope_antiquant_scale, num_heads, scale, pre_tokens, next_tokens, input_layout, num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, key_antiquant_mode, value_antiquant_mode, softmax_lse_flag);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t,int64_t,int64_t> wrapper__npu_fusion_attention(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, int64_t head_num, c10::string_view input_layout, const ::std::optional<at::Tensor> & pse, const ::std::optional<at::Tensor> & padding_mask, const ::std::optional<at::Tensor> & atten_mask, double scale, double keep_prob, int64_t pre_tockens, int64_t next_tockens, int64_t inner_precise, at::OptionalIntArrayRef prefix, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_seq_kvlen, int64_t sparse_mode, bool gen_mask_parallel, bool sync) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(query);
      c10_npu::check_npu_tensor_is_safe(key);
      c10_npu::check_npu_tensor_is_safe(value);
      c10_npu::check_npu_tensor_is_safe(pse);
      c10_npu::check_npu_tensor_is_safe(padding_mask);
      c10_npu::check_npu_tensor_is_safe(atten_mask);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, query, "wrapper__npu_fusion_attention", "query");
  c10::impl::check_and_update_common_device(common_device, key, "wrapper__npu_fusion_attention", "key");
  c10::impl::check_and_update_common_device(common_device, value, "wrapper__npu_fusion_attention", "value");
  c10::impl::check_and_update_common_device(common_device, pse, "wrapper__npu_fusion_attention", "pse");
  c10::impl::check_and_update_common_device(common_device, padding_mask, "wrapper__npu_fusion_attention", "padding_mask");
  c10::impl::check_and_update_common_device(common_device, atten_mask, "wrapper__npu_fusion_attention", "atten_mask");
  const c10::OptionalDeviceGuard device_guard(device_of(query));
  return op_plugin::npu_fusion_attention(query, key, value, head_num, input_layout, pse, padding_mask, atten_mask, scale, keep_prob, pre_tockens, next_tockens, inner_precise, prefix, actual_seq_qlen, actual_seq_kvlen, sparse_mode, gen_mask_parallel, sync);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrapper__npu_fusion_attention_grad(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & dy, int64_t head_num, c10::string_view input_layout, const ::std::optional<at::Tensor> & pse, const ::std::optional<at::Tensor> & padding_mask, const ::std::optional<at::Tensor> & atten_mask, const ::std::optional<at::Tensor> & softmax_max, const ::std::optional<at::Tensor> & softmax_sum, const ::std::optional<at::Tensor> & softmax_in, const ::std::optional<at::Tensor> & attention_in, double scale_value, double keep_prob, int64_t pre_tockens, int64_t next_tockens, int64_t inner_precise, int64_t seed, int64_t offset, int64_t numels, at::OptionalIntArrayRef prefix, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_seq_kvlen, int64_t sparse_mode, bool gen_mask_parallel, bool sync) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(query);
      c10_npu::check_npu_tensor_is_safe(key);
      c10_npu::check_npu_tensor_is_safe(value);
      c10_npu::check_npu_tensor_is_safe(dy);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, query, "wrapper__npu_fusion_attention_grad", "query");
  c10::impl::check_and_update_common_device(common_device, key, "wrapper__npu_fusion_attention_grad", "key");
  c10::impl::check_and_update_common_device(common_device, value, "wrapper__npu_fusion_attention_grad", "value");
  c10::impl::check_and_update_common_device(common_device, dy, "wrapper__npu_fusion_attention_grad", "dy");
  c10::impl::check_and_update_common_device(common_device, pse, "wrapper__npu_fusion_attention_grad", "pse");
  c10::impl::check_and_update_common_device(common_device, padding_mask, "wrapper__npu_fusion_attention_grad", "padding_mask");
  c10::impl::check_and_update_common_device(common_device, atten_mask, "wrapper__npu_fusion_attention_grad", "atten_mask");
  c10::impl::check_and_update_common_device(common_device, softmax_max, "wrapper__npu_fusion_attention_grad", "softmax_max");
  c10::impl::check_and_update_common_device(common_device, softmax_sum, "wrapper__npu_fusion_attention_grad", "softmax_sum");
  c10::impl::check_and_update_common_device(common_device, softmax_in, "wrapper__npu_fusion_attention_grad", "softmax_in");
  c10::impl::check_and_update_common_device(common_device, attention_in, "wrapper__npu_fusion_attention_grad", "attention_in");
  const c10::OptionalDeviceGuard device_guard(device_of(query));
  return op_plugin::npu_fusion_attention_grad(query, key, value, dy, head_num, input_layout, pse, padding_mask, atten_mask, softmax_max, softmax_sum, softmax_in, attention_in, scale_value, keep_prob, pre_tockens, next_tockens, inner_precise, seed, offset, numels, prefix, actual_seq_qlen, actual_seq_kvlen, sparse_mode, gen_mask_parallel, sync);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t,int64_t,int64_t> wrapper__npu_fusion_attention_v2(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, int64_t head_num, c10::string_view input_layout, const ::std::optional<at::Tensor> & pse, const ::std::optional<at::Tensor> & padding_mask, const ::std::optional<at::Tensor> & atten_mask, const ::std::optional<at::Tensor> & query_rope, const ::std::optional<at::Tensor> & key_rope, double scale, double keep_prob, int64_t pre_tokens, int64_t next_tokens, int64_t inner_precise, at::OptionalIntArrayRef prefix, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_seq_kvlen, int64_t sparse_mode, bool gen_mask_parallel, bool sync, int64_t pse_type, at::OptionalIntArrayRef q_start_idx, at::OptionalIntArrayRef kv_start_idx) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(query);
      c10_npu::check_npu_tensor_is_safe(key);
      c10_npu::check_npu_tensor_is_safe(value);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, query, "wrapper__npu_fusion_attention_v2", "query");
  c10::impl::check_and_update_common_device(common_device, key, "wrapper__npu_fusion_attention_v2", "key");
  c10::impl::check_and_update_common_device(common_device, value, "wrapper__npu_fusion_attention_v2", "value");
  c10::impl::check_and_update_common_device(common_device, pse, "wrapper__npu_fusion_attention_v2", "pse");
  c10::impl::check_and_update_common_device(common_device, padding_mask, "wrapper__npu_fusion_attention_v2", "padding_mask");
  c10::impl::check_and_update_common_device(common_device, atten_mask, "wrapper__npu_fusion_attention_v2", "atten_mask");
  c10::impl::check_and_update_common_device(common_device, query_rope, "wrapper__npu_fusion_attention_v2", "query_rope");
  c10::impl::check_and_update_common_device(common_device, key_rope, "wrapper__npu_fusion_attention_v2", "key_rope");
  const c10::OptionalDeviceGuard device_guard(device_of(query));
  return op_plugin::npu_fusion_attention_v2(query, key, value, head_num, input_layout, pse, padding_mask, atten_mask, query_rope, key_rope, scale, keep_prob, pre_tokens, next_tokens, inner_precise, prefix, actual_seq_qlen, actual_seq_kvlen, sparse_mode, gen_mask_parallel, sync, pse_type, q_start_idx, kv_start_idx);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrapper__npu_fusion_attention_grad_v2(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & dy, int64_t head_num, c10::string_view input_layout, const ::std::optional<at::Tensor> & pse, const ::std::optional<at::Tensor> & padding_mask, const ::std::optional<at::Tensor> & atten_mask, const ::std::optional<at::Tensor> & softmax_max, const ::std::optional<at::Tensor> & softmax_sum, const ::std::optional<at::Tensor> & softmax_in, const ::std::optional<at::Tensor> & attention_in, const ::std::optional<at::Tensor> & query_rope, const ::std::optional<at::Tensor> & key_rope, double scale_value, double keep_prob, int64_t pre_tokens, int64_t next_tokens, int64_t inner_precise, int64_t seed, int64_t offset, int64_t numels, at::OptionalIntArrayRef prefix, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_seq_kvlen, int64_t sparse_mode, bool gen_mask_parallel, bool sync, int64_t pse_type, at::OptionalIntArrayRef q_start_idx, at::OptionalIntArrayRef kv_start_idx) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(query);
      c10_npu::check_npu_tensor_is_safe(key);
      c10_npu::check_npu_tensor_is_safe(value);
      c10_npu::check_npu_tensor_is_safe(dy);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, query, "wrapper__npu_fusion_attention_grad_v2", "query");
  c10::impl::check_and_update_common_device(common_device, key, "wrapper__npu_fusion_attention_grad_v2", "key");
  c10::impl::check_and_update_common_device(common_device, value, "wrapper__npu_fusion_attention_grad_v2", "value");
  c10::impl::check_and_update_common_device(common_device, dy, "wrapper__npu_fusion_attention_grad_v2", "dy");
  c10::impl::check_and_update_common_device(common_device, pse, "wrapper__npu_fusion_attention_grad_v2", "pse");
  c10::impl::check_and_update_common_device(common_device, padding_mask, "wrapper__npu_fusion_attention_grad_v2", "padding_mask");
  c10::impl::check_and_update_common_device(common_device, atten_mask, "wrapper__npu_fusion_attention_grad_v2", "atten_mask");
  c10::impl::check_and_update_common_device(common_device, softmax_max, "wrapper__npu_fusion_attention_grad_v2", "softmax_max");
  c10::impl::check_and_update_common_device(common_device, softmax_sum, "wrapper__npu_fusion_attention_grad_v2", "softmax_sum");
  c10::impl::check_and_update_common_device(common_device, softmax_in, "wrapper__npu_fusion_attention_grad_v2", "softmax_in");
  c10::impl::check_and_update_common_device(common_device, attention_in, "wrapper__npu_fusion_attention_grad_v2", "attention_in");
  c10::impl::check_and_update_common_device(common_device, query_rope, "wrapper__npu_fusion_attention_grad_v2", "query_rope");
  c10::impl::check_and_update_common_device(common_device, key_rope, "wrapper__npu_fusion_attention_grad_v2", "key_rope");
  const c10::OptionalDeviceGuard device_guard(device_of(query));
  return op_plugin::npu_fusion_attention_grad_v2(query, key, value, dy, head_num, input_layout, pse, padding_mask, atten_mask, softmax_max, softmax_sum, softmax_in, attention_in, query_rope, key_rope, scale_value, keep_prob, pre_tokens, next_tokens, inner_precise, seed, offset, numels, prefix, actual_seq_qlen, actual_seq_kvlen, sparse_mode, gen_mask_parallel, sync, pse_type, q_start_idx, kv_start_idx);
}

::std::vector<at::Tensor> wrapper__npu_fused_attention_layernorm_qkv_fwd(const at::Tensor & x, const at::Tensor & kernel_query, const at::Tensor & kernel_key, const at::Tensor & kernel_value, const at::Tensor & gamma, const at::Tensor & beta, const ::std::optional<at::Tensor> & bias_query, const ::std::optional<at::Tensor> & bias_key, const ::std::optional<at::Tensor> & bias_value, int64_t seq_len, int64_t num_heads, double eps) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(x);
      c10_npu::check_npu_tensor_is_safe(kernel_query);
      c10_npu::check_npu_tensor_is_safe(kernel_key);
      c10_npu::check_npu_tensor_is_safe(kernel_value);
      c10_npu::check_npu_tensor_is_safe(gamma);
      c10_npu::check_npu_tensor_is_safe(beta);
      c10_npu::check_npu_tensor_is_safe(bias_query);
      c10_npu::check_npu_tensor_is_safe(bias_key);
      c10_npu::check_npu_tensor_is_safe(bias_value);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, x, "wrapper__npu_fused_attention_layernorm_qkv_fwd", "x");
  c10::impl::check_and_update_common_device(common_device, kernel_query, "wrapper__npu_fused_attention_layernorm_qkv_fwd", "kernel_query");
  c10::impl::check_and_update_common_device(common_device, kernel_key, "wrapper__npu_fused_attention_layernorm_qkv_fwd", "kernel_key");
  c10::impl::check_and_update_common_device(common_device, kernel_value, "wrapper__npu_fused_attention_layernorm_qkv_fwd", "kernel_value");
  c10::impl::check_and_update_common_device(common_device, gamma, "wrapper__npu_fused_attention_layernorm_qkv_fwd", "gamma");
  c10::impl::check_and_update_common_device(common_device, beta, "wrapper__npu_fused_attention_layernorm_qkv_fwd", "beta");
  c10::impl::check_and_update_common_device(common_device, bias_query, "wrapper__npu_fused_attention_layernorm_qkv_fwd", "bias_query");
  c10::impl::check_and_update_common_device(common_device, bias_key, "wrapper__npu_fused_attention_layernorm_qkv_fwd", "bias_key");
  c10::impl::check_and_update_common_device(common_device, bias_value, "wrapper__npu_fused_attention_layernorm_qkv_fwd", "bias_value");
  const c10::OptionalDeviceGuard device_guard(device_of(x));
  return op_plugin::npu_fused_attention_layernorm_qkv_fwd(x, kernel_query, kernel_key, kernel_value, gamma, beta, bias_query, bias_key, bias_value, seq_len, num_heads, eps);
}

::std::vector<at::Tensor> wrapper__npu_fused_attention_qkv_grad(const at::Tensor & grad_output_query, const at::Tensor & grad_output_key, const at::Tensor & grad_output_value, const at::Tensor & query_kernel, const at::Tensor & key_kernel, const at::Tensor & value_kernel, const at::Tensor & hidden_states, const at::Tensor & grad_output_ln) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(grad_output_query);
      c10_npu::check_npu_tensor_is_safe(grad_output_key);
      c10_npu::check_npu_tensor_is_safe(grad_output_value);
      c10_npu::check_npu_tensor_is_safe(query_kernel);
      c10_npu::check_npu_tensor_is_safe(key_kernel);
      c10_npu::check_npu_tensor_is_safe(value_kernel);
      c10_npu::check_npu_tensor_is_safe(hidden_states);
      c10_npu::check_npu_tensor_is_safe(grad_output_ln);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad_output_query, "wrapper__npu_fused_attention_qkv_grad", "grad_output_query");
  c10::impl::check_and_update_common_device(common_device, grad_output_key, "wrapper__npu_fused_attention_qkv_grad", "grad_output_key");
  c10::impl::check_and_update_common_device(common_device, grad_output_value, "wrapper__npu_fused_attention_qkv_grad", "grad_output_value");
  c10::impl::check_and_update_common_device(common_device, query_kernel, "wrapper__npu_fused_attention_qkv_grad", "query_kernel");
  c10::impl::check_and_update_common_device(common_device, key_kernel, "wrapper__npu_fused_attention_qkv_grad", "key_kernel");
  c10::impl::check_and_update_common_device(common_device, value_kernel, "wrapper__npu_fused_attention_qkv_grad", "value_kernel");
  c10::impl::check_and_update_common_device(common_device, hidden_states, "wrapper__npu_fused_attention_qkv_grad", "hidden_states");
  c10::impl::check_and_update_common_device(common_device, grad_output_ln, "wrapper__npu_fused_attention_qkv_grad", "grad_output_ln");
  const c10::OptionalDeviceGuard device_guard(device_of(grad_output_query));
  return op_plugin::npu_fused_attention_qkv_grad(grad_output_query, grad_output_key, grad_output_value, query_kernel, key_kernel, value_kernel, hidden_states, grad_output_ln);
}

::std::tuple<at::Tensor,at::Tensor> wrapper__npu_geglu(const at::Tensor & self, int64_t dim, int64_t approximate, bool activate_left) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_geglu", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_geglu(self, dim, approximate, activate_left);
}

at::Tensor wrapper__npu_geglu_grad(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & gelu, int64_t dim, int64_t approximate, bool activate_left) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(grad_output);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(gelu);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad_output, "wrapper__npu_geglu_grad", "grad_output");
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_geglu_grad", "self");
  c10::impl::check_and_update_common_device(common_device, gelu, "wrapper__npu_geglu_grad", "gelu");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_geglu_grad(grad_output, self, gelu, dim, approximate, activate_left);
}

at::Tensor wrapper__npu_get_float_status(const at::Tensor & self, int64_t mode) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_get_float_status", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_get_float_status(self, mode);
}

at::Tensor wrapper__npu_giou(const at::Tensor & self, const at::Tensor & gtboxes, bool trans, bool is_cross, int64_t mode) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(gtboxes);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_giou", "self");
  c10::impl::check_and_update_common_device(common_device, gtboxes, "wrapper__npu_giou", "gtboxes");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_giou(self, gtboxes, trans, is_cross, mode);
}

::std::tuple<at::Tensor,at::Tensor> wrapper__npu_giou_backward(const at::Tensor & grad, const at::Tensor & bboxes, const at::Tensor & gtboxes, bool trans, bool is_cross, int64_t mode) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(grad);
      c10_npu::check_npu_tensor_is_safe(bboxes);
      c10_npu::check_npu_tensor_is_safe(gtboxes);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad, "wrapper__npu_giou_backward", "grad");
  c10::impl::check_and_update_common_device(common_device, bboxes, "wrapper__npu_giou_backward", "bboxes");
  c10::impl::check_and_update_common_device(common_device, gtboxes, "wrapper__npu_giou_backward", "gtboxes");
  const c10::OptionalDeviceGuard device_guard(device_of(grad));
  return op_plugin::npu_giou_backward(grad, bboxes, gtboxes, trans, is_cross, mode);
}

at::Tensor wrapper__npu_grid_assign_positive(const at::Tensor & self, const at::Tensor & overlaps, const at::Tensor & box_responsible_flags, const at::Tensor & max_overlaps, const at::Tensor & argmax_overlaps, const at::Tensor & gt_max_overlaps, const at::Tensor & gt_argmax_overlaps, int64_t num_gts, double pos_iou_thr, double min_pos_iou, bool gt_max_assign_all) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(overlaps);
      c10_npu::check_npu_tensor_is_safe(box_responsible_flags);
      c10_npu::check_npu_tensor_is_safe(max_overlaps);
      c10_npu::check_npu_tensor_is_safe(argmax_overlaps);
      c10_npu::check_npu_tensor_is_safe(gt_max_overlaps);
      c10_npu::check_npu_tensor_is_safe(gt_argmax_overlaps);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_grid_assign_positive", "self");
  c10::impl::check_and_update_common_device(common_device, overlaps, "wrapper__npu_grid_assign_positive", "overlaps");
  c10::impl::check_and_update_common_device(common_device, box_responsible_flags, "wrapper__npu_grid_assign_positive", "box_responsible_flags");
  c10::impl::check_and_update_common_device(common_device, max_overlaps, "wrapper__npu_grid_assign_positive", "max_overlaps");
  c10::impl::check_and_update_common_device(common_device, argmax_overlaps, "wrapper__npu_grid_assign_positive", "argmax_overlaps");
  c10::impl::check_and_update_common_device(common_device, gt_max_overlaps, "wrapper__npu_grid_assign_positive", "gt_max_overlaps");
  c10::impl::check_and_update_common_device(common_device, gt_argmax_overlaps, "wrapper__npu_grid_assign_positive", "gt_argmax_overlaps");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_grid_assign_positive(self, overlaps, box_responsible_flags, max_overlaps, argmax_overlaps, gt_max_overlaps, gt_argmax_overlaps, num_gts, pos_iou_thr, min_pos_iou, gt_max_assign_all);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapper__npu_group_norm_silu(const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, int64_t group, double eps) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(weight);
      c10_npu::check_npu_tensor_is_safe(bias);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_group_norm_silu", "input");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__npu_group_norm_silu", "weight");
  c10::impl::check_and_update_common_device(common_device, bias, "wrapper__npu_group_norm_silu", "bias");
  const c10::OptionalDeviceGuard device_guard(device_of(input));
  return op_plugin::npu_group_norm_silu(input, weight, bias, group, eps);
}

::std::vector<at::Tensor> wrapper__npu_grouped_matmul(at::TensorList x, at::TensorList weight, ::std::optional<at::TensorList> bias, ::std::optional<at::TensorList> scale, ::std::optional<at::TensorList> offset, ::std::optional<at::TensorList> antiquant_scale, ::std::optional<at::TensorList> antiquant_offset, ::std::optional<at::TensorList> per_token_scale, const ::std::optional<at::Tensor> & group_list, ::std::optional<at::TensorList> activation_input, ::std::optional<at::TensorList> activation_quant_scale, ::std::optional<at::TensorList> activation_quant_offset, ::std::optional<int64_t> split_item, ::std::optional<int64_t> group_type, ::std::optional<int64_t> group_list_type, ::std::optional<int64_t> act_type, at::OptionalIntArrayRef tuning_config, ::std::optional<at::ScalarType> output_dtype) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(x);
      c10_npu::check_npu_tensor_is_safe(weight);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, x, "wrapper__npu_grouped_matmul", "x");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__npu_grouped_matmul", "weight");
  c10::impl::check_and_update_common_device(common_device, group_list, "wrapper__npu_grouped_matmul", "group_list");
  const c10::OptionalDeviceGuard device_guard(device_of(x));
  return op_plugin::npu_grouped_matmul(x, weight, bias, scale, offset, antiquant_scale, antiquant_offset, per_token_scale, group_list, activation_input, activation_quant_scale, activation_quant_offset, split_item, group_type, group_list_type, act_type, tuning_config, output_dtype);
}

::std::vector<at::Tensor> wrapper_List_npu_grouped_matmul(at::TensorList x, at::TensorList weight, ::std::optional<at::TensorList> bias, ::std::optional<at::TensorList> scale, ::std::optional<at::TensorList> offset, ::std::optional<at::TensorList> antiquant_scale, ::std::optional<at::TensorList> antiquant_offset, ::std::optional<at::TensorList> per_token_scale, at::OptionalIntArrayRef group_list, ::std::optional<at::TensorList> activation_input, ::std::optional<at::TensorList> activation_quant_scale, ::std::optional<at::TensorList> activation_quant_offset, ::std::optional<int64_t> split_item, ::std::optional<int64_t> group_type, ::std::optional<int64_t> group_list_type, ::std::optional<int64_t> act_type, ::std::optional<at::ScalarType> output_dtype) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(x);
      c10_npu::check_npu_tensor_is_safe(weight);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, x, "wrapper_List_npu_grouped_matmul", "x");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper_List_npu_grouped_matmul", "weight");
  const c10::OptionalDeviceGuard device_guard(device_of(x));
  return op_plugin::npu_grouped_matmul(x, weight, bias, scale, offset, antiquant_scale, antiquant_offset, per_token_scale, group_list, activation_input, activation_quant_scale, activation_quant_offset, split_item, group_type, group_list_type, act_type, output_dtype);
}

at::Tensor wrapper__npu_grouped_matmul_finalize_routing(const at::Tensor & x, const at::Tensor & w, const at::Tensor & group_list, const ::std::optional<at::Tensor> & scale, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & offset, const ::std::optional<at::Tensor> & pertoken_scale, const ::std::optional<at::Tensor> & shared_input, const ::std::optional<at::Tensor> & logit, const ::std::optional<at::Tensor> & row_index, ::std::optional<at::ScalarType> dtype, ::std::optional<double> shared_input_weight, ::std::optional<int64_t> shared_input_offset, ::std::optional<int64_t> output_bs, ::std::optional<int64_t> group_list_type) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(x);
      c10_npu::check_npu_tensor_is_safe(w);
      c10_npu::check_npu_tensor_is_safe(group_list);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, x, "wrapper__npu_grouped_matmul_finalize_routing", "x");
  c10::impl::check_and_update_common_device(common_device, w, "wrapper__npu_grouped_matmul_finalize_routing", "w");
  c10::impl::check_and_update_common_device(common_device, group_list, "wrapper__npu_grouped_matmul_finalize_routing", "group_list");
  c10::impl::check_and_update_common_device(common_device, scale, "wrapper__npu_grouped_matmul_finalize_routing", "scale");
  c10::impl::check_and_update_common_device(common_device, bias, "wrapper__npu_grouped_matmul_finalize_routing", "bias");
  c10::impl::check_and_update_common_device(common_device, offset, "wrapper__npu_grouped_matmul_finalize_routing", "offset");
  c10::impl::check_and_update_common_device(common_device, pertoken_scale, "wrapper__npu_grouped_matmul_finalize_routing", "pertoken_scale");
  c10::impl::check_and_update_common_device(common_device, shared_input, "wrapper__npu_grouped_matmul_finalize_routing", "shared_input");
  c10::impl::check_and_update_common_device(common_device, logit, "wrapper__npu_grouped_matmul_finalize_routing", "logit");
  c10::impl::check_and_update_common_device(common_device, row_index, "wrapper__npu_grouped_matmul_finalize_routing", "row_index");
  const c10::OptionalDeviceGuard device_guard(device_of(x));
  return op_plugin::npu_grouped_matmul_finalize_routing(x, w, group_list, scale, bias, offset, pertoken_scale, shared_input, logit, row_index, dtype, shared_input_weight, shared_input_offset, output_bs, group_list_type);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrapper__npu_gru(const at::Tensor & input, const at::Tensor & hx, const at::Tensor & weight_input, const at::Tensor & weight_hidden, const at::Tensor & bias_input, const at::Tensor & bias_hidden, const at::Tensor & seq_length, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(hx);
      c10_npu::check_npu_tensor_is_safe(weight_input);
      c10_npu::check_npu_tensor_is_safe(weight_hidden);
      c10_npu::check_npu_tensor_is_safe(bias_input);
      c10_npu::check_npu_tensor_is_safe(bias_hidden);
      c10_npu::check_npu_tensor_is_safe(seq_length);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_gru", "input");
  c10::impl::check_and_update_common_device(common_device, hx, "wrapper__npu_gru", "hx");
  c10::impl::check_and_update_common_device(common_device, weight_input, "wrapper__npu_gru", "weight_input");
  c10::impl::check_and_update_common_device(common_device, weight_hidden, "wrapper__npu_gru", "weight_hidden");
  c10::impl::check_and_update_common_device(common_device, bias_input, "wrapper__npu_gru", "bias_input");
  c10::impl::check_and_update_common_device(common_device, bias_hidden, "wrapper__npu_gru", "bias_hidden");
  c10::impl::check_and_update_common_device(common_device, seq_length, "wrapper__npu_gru", "seq_length");
  const c10::OptionalDeviceGuard device_guard(device_of(input));
  return op_plugin::npu_gru(input, hx, weight_input, weight_hidden, bias_input, bias_hidden, seq_length, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrapper__npu_gru_backward(const ::std::optional<at::Tensor> & grady, const ::std::optional<at::Tensor> & gradh, const at::Tensor & input, const at::Tensor & weight_input, const at::Tensor & weight_hidden, const at::Tensor & bias_input, const at::Tensor & bias_hidden, const at::Tensor & seq_length, const at::Tensor & hx, const at::Tensor & y_output, const at::Tensor & h_output, const at::Tensor & output_updata, const at::Tensor & output_reset, const at::Tensor & output_new, const at::Tensor & hidden_new) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(grady);
      c10_npu::check_npu_tensor_is_safe(gradh);
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(weight_input);
      c10_npu::check_npu_tensor_is_safe(weight_hidden);
      c10_npu::check_npu_tensor_is_safe(bias_input);
      c10_npu::check_npu_tensor_is_safe(bias_hidden);
      c10_npu::check_npu_tensor_is_safe(seq_length);
      c10_npu::check_npu_tensor_is_safe(hx);
      c10_npu::check_npu_tensor_is_safe(y_output);
      c10_npu::check_npu_tensor_is_safe(h_output);
      c10_npu::check_npu_tensor_is_safe(output_updata);
      c10_npu::check_npu_tensor_is_safe(output_reset);
      c10_npu::check_npu_tensor_is_safe(output_new);
      c10_npu::check_npu_tensor_is_safe(hidden_new);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grady, "wrapper__npu_gru_backward", "grady");
  c10::impl::check_and_update_common_device(common_device, gradh, "wrapper__npu_gru_backward", "gradh");
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_gru_backward", "input");
  c10::impl::check_and_update_common_device(common_device, weight_input, "wrapper__npu_gru_backward", "weight_input");
  c10::impl::check_and_update_common_device(common_device, weight_hidden, "wrapper__npu_gru_backward", "weight_hidden");
  c10::impl::check_and_update_common_device(common_device, bias_input, "wrapper__npu_gru_backward", "bias_input");
  c10::impl::check_and_update_common_device(common_device, bias_hidden, "wrapper__npu_gru_backward", "bias_hidden");
  c10::impl::check_and_update_common_device(common_device, seq_length, "wrapper__npu_gru_backward", "seq_length");
  c10::impl::check_and_update_common_device(common_device, hx, "wrapper__npu_gru_backward", "hx");
  c10::impl::check_and_update_common_device(common_device, y_output, "wrapper__npu_gru_backward", "y_output");
  c10::impl::check_and_update_common_device(common_device, h_output, "wrapper__npu_gru_backward", "h_output");
  c10::impl::check_and_update_common_device(common_device, output_updata, "wrapper__npu_gru_backward", "output_updata");
  c10::impl::check_and_update_common_device(common_device, output_reset, "wrapper__npu_gru_backward", "output_reset");
  c10::impl::check_and_update_common_device(common_device, output_new, "wrapper__npu_gru_backward", "output_new");
  c10::impl::check_and_update_common_device(common_device, hidden_new, "wrapper__npu_gru_backward", "hidden_new");
  const c10::OptionalDeviceGuard device_guard(device_of(grady));
  return op_plugin::npu_gru_backward(grady, gradh, input, weight_input, weight_hidden, bias_input, bias_hidden, seq_length, hx, y_output, h_output, output_updata, output_reset, output_new, hidden_new);
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> wrapper_out_npu_hans_encode_out(const at::Tensor & input,bool statistic,bool reshuff, at::TensorList out) {
  TORCH_CHECK(out.size() == 4, "expected tuple of 4 elements but got ", out.size(), OPS_ERROR(ErrCode::PARAM));
  at::Tensor pdf = out[0];
  at::Tensor mantissa = out[1];
  at::Tensor fixed = out[2];
  at::Tensor var = out[3];
  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(pdf);
      c10_npu::check_npu_tensor_is_safe(mantissa);
      c10_npu::check_npu_tensor_is_safe(fixed);
      c10_npu::check_npu_tensor_is_safe(var);
      c10_npu::check_npu_tensor_is_safe(input);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, pdf, "wrapper_out_npu_hans_encode_out", "pdf");
  c10::impl::check_and_update_common_device(common_device, mantissa, "wrapper_out_npu_hans_encode_out", "mantissa");
  c10::impl::check_and_update_common_device(common_device, fixed, "wrapper_out_npu_hans_encode_out", "fixed");
  c10::impl::check_and_update_common_device(common_device, var, "wrapper_out_npu_hans_encode_out", "var");
  c10::impl::check_and_update_common_device(common_device, input, "wrapper_out_npu_hans_encode_out", "input");
  const c10::OptionalDeviceGuard device_guard(device_of(pdf));
  return op_plugin::npu_hans_encode_out(input, statistic, reshuff, pdf, mantissa, fixed, var);
}

at::Tensor & wrapper_out_npu_hans_decode_out(const at::Tensor & mantissa, const at::Tensor & fixed, const at::Tensor & var, const at::Tensor & pdf, bool reshuff, at::Tensor & out) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(out);
      c10_npu::check_npu_tensor_is_safe(mantissa);
      c10_npu::check_npu_tensor_is_safe(fixed);
      c10_npu::check_npu_tensor_is_safe(var);
      c10_npu::check_npu_tensor_is_safe(pdf);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, out, "wrapper_out_npu_hans_decode_out", "out");
  c10::impl::check_and_update_common_device(common_device, mantissa, "wrapper_out_npu_hans_decode_out", "mantissa");
  c10::impl::check_and_update_common_device(common_device, fixed, "wrapper_out_npu_hans_decode_out", "fixed");
  c10::impl::check_and_update_common_device(common_device, var, "wrapper_out_npu_hans_decode_out", "var");
  c10::impl::check_and_update_common_device(common_device, pdf, "wrapper_out_npu_hans_decode_out", "pdf");
  const c10::OptionalDeviceGuard device_guard(device_of(out));
  return op_plugin::npu_hans_decode_out(mantissa, fixed, var, pdf, reshuff, out);
}

::std::tuple<at::Tensor,at::Tensor> wrapper__npu_ifmr(const at::Tensor & data, const at::Tensor & data_min, const at::Tensor & data_max, const at::Tensor & cumsum, double min_percentile, double max_percentile, double search_start, double search_end, double search_step, bool with_offset) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(data);
      c10_npu::check_npu_tensor_is_safe(data_min);
      c10_npu::check_npu_tensor_is_safe(data_max);
      c10_npu::check_npu_tensor_is_safe(cumsum);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, data, "wrapper__npu_ifmr", "data");
  c10::impl::check_and_update_common_device(common_device, data_min, "wrapper__npu_ifmr", "data_min");
  c10::impl::check_and_update_common_device(common_device, data_max, "wrapper__npu_ifmr", "data_max");
  c10::impl::check_and_update_common_device(common_device, cumsum, "wrapper__npu_ifmr", "cumsum");
  const c10::OptionalDeviceGuard device_guard(device_of(data));
  return op_plugin::npu_ifmr(data, data_min, data_max, cumsum, min_percentile, max_percentile, search_start, search_end, search_step, with_offset);
}

at::Tensor wrapper__npu_our_incre_flash_attention(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & padding_mask, const ::std::optional<at::Tensor> & atten_mask, const ::std::optional<at::Tensor> & pse_shift, at::OptionalSymIntArrayRef actual_seq_lengths, const ::std::optional<at::Tensor> & antiquant_scale, const ::std::optional<at::Tensor> & antiquant_offset, const ::std::optional<at::Tensor> & block_table, const ::std::optional<at::Tensor> & dequant_scale1, const ::std::optional<at::Tensor> & quant_scale1, const ::std::optional<at::Tensor> & dequant_scale2, const ::std::optional<at::Tensor> & quant_scale2, const ::std::optional<at::Tensor> & quant_offset2, const ::std::optional<at::Tensor> & kv_padding_size, int64_t num_heads, double scale_value, c10::string_view input_layout, int64_t num_key_value_heads, int64_t block_size, int64_t inner_precise) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(query);
      c10_npu::check_npu_tensor_is_safe(key);
      c10_npu::check_npu_tensor_is_safe(value);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, query, "wrapper__npu_our_incre_flash_attention", "query");
  c10::impl::check_and_update_common_device(common_device, key, "wrapper__npu_our_incre_flash_attention", "key");
  c10::impl::check_and_update_common_device(common_device, value, "wrapper__npu_our_incre_flash_attention", "value");
  c10::impl::check_and_update_common_device(common_device, padding_mask, "wrapper__npu_our_incre_flash_attention", "padding_mask");
  c10::impl::check_and_update_common_device(common_device, atten_mask, "wrapper__npu_our_incre_flash_attention", "atten_mask");
  c10::impl::check_and_update_common_device(common_device, pse_shift, "wrapper__npu_our_incre_flash_attention", "pse_shift");
  c10::impl::check_and_update_common_device(common_device, antiquant_scale, "wrapper__npu_our_incre_flash_attention", "antiquant_scale");
  c10::impl::check_and_update_common_device(common_device, antiquant_offset, "wrapper__npu_our_incre_flash_attention", "antiquant_offset");
  c10::impl::check_and_update_common_device(common_device, block_table, "wrapper__npu_our_incre_flash_attention", "block_table");
  c10::impl::check_and_update_common_device(common_device, dequant_scale1, "wrapper__npu_our_incre_flash_attention", "dequant_scale1");
  c10::impl::check_and_update_common_device(common_device, quant_scale1, "wrapper__npu_our_incre_flash_attention", "quant_scale1");
  c10::impl::check_and_update_common_device(common_device, dequant_scale2, "wrapper__npu_our_incre_flash_attention", "dequant_scale2");
  c10::impl::check_and_update_common_device(common_device, quant_scale2, "wrapper__npu_our_incre_flash_attention", "quant_scale2");
  c10::impl::check_and_update_common_device(common_device, quant_offset2, "wrapper__npu_our_incre_flash_attention", "quant_offset2");
  c10::impl::check_and_update_common_device(common_device, kv_padding_size, "wrapper__npu_our_incre_flash_attention", "kv_padding_size");
  const c10::OptionalDeviceGuard device_guard(device_of(query));
  return op_plugin::npu_our_incre_flash_attention_symint(query, key, value, padding_mask, atten_mask, pse_shift, actual_seq_lengths, antiquant_scale, antiquant_offset, block_table, dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2, kv_padding_size, num_heads, scale_value, input_layout, num_key_value_heads, block_size, inner_precise);
}

at::Tensor wrapper__npu_sparse_paged_attention(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & padding_mask, const ::std::optional<at::Tensor> & atten_mask, const ::std::optional<at::Tensor> & pse_shift, at::OptionalSymIntArrayRef actual_seq_lengths, const ::std::optional<at::Tensor> & antiquant_scale, const ::std::optional<at::Tensor> & antiquant_offset, const ::std::optional<at::Tensor> & block_table, const ::std::optional<at::Tensor> & block_position, const ::std::optional<at::Tensor> & dequant_scale1, const ::std::optional<at::Tensor> & quant_scale1, const ::std::optional<at::Tensor> & dequant_scale2, const ::std::optional<at::Tensor> & quant_scale2, const ::std::optional<at::Tensor> & quant_offset2, const ::std::optional<at::Tensor> & kv_padding_size, int64_t num_heads, double scale_value, c10::string_view input_layout, int64_t num_key_value_heads, int64_t block_size, int64_t inner_precise) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(query);
      c10_npu::check_npu_tensor_is_safe(key);
      c10_npu::check_npu_tensor_is_safe(value);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, query, "wrapper__npu_sparse_paged_attention", "query");
  c10::impl::check_and_update_common_device(common_device, key, "wrapper__npu_sparse_paged_attention", "key");
  c10::impl::check_and_update_common_device(common_device, value, "wrapper__npu_sparse_paged_attention", "value");
  c10::impl::check_and_update_common_device(common_device, padding_mask, "wrapper__npu_sparse_paged_attention", "padding_mask");
  c10::impl::check_and_update_common_device(common_device, atten_mask, "wrapper__npu_sparse_paged_attention", "atten_mask");
  c10::impl::check_and_update_common_device(common_device, pse_shift, "wrapper__npu_sparse_paged_attention", "pse_shift");
  c10::impl::check_and_update_common_device(common_device, antiquant_scale, "wrapper__npu_sparse_paged_attention", "antiquant_scale");
  c10::impl::check_and_update_common_device(common_device, antiquant_offset, "wrapper__npu_sparse_paged_attention", "antiquant_offset");
  c10::impl::check_and_update_common_device(common_device, block_table, "wrapper__npu_sparse_paged_attention", "block_table");
  c10::impl::check_and_update_common_device(common_device, block_position, "wrapper__npu_sparse_paged_attention", "block_position");
  c10::impl::check_and_update_common_device(common_device, dequant_scale1, "wrapper__npu_sparse_paged_attention", "dequant_scale1");
  c10::impl::check_and_update_common_device(common_device, quant_scale1, "wrapper__npu_sparse_paged_attention", "quant_scale1");
  c10::impl::check_and_update_common_device(common_device, dequant_scale2, "wrapper__npu_sparse_paged_attention", "dequant_scale2");
  c10::impl::check_and_update_common_device(common_device, quant_scale2, "wrapper__npu_sparse_paged_attention", "quant_scale2");
  c10::impl::check_and_update_common_device(common_device, quant_offset2, "wrapper__npu_sparse_paged_attention", "quant_offset2");
  c10::impl::check_and_update_common_device(common_device, kv_padding_size, "wrapper__npu_sparse_paged_attention", "kv_padding_size");
  const c10::OptionalDeviceGuard device_guard(device_of(query));
  return op_plugin::npu_sparse_paged_attention_symint(query, key, value, padding_mask, atten_mask, pse_shift, actual_seq_lengths, antiquant_scale, antiquant_offset, block_table, block_position, dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2, kv_padding_size, num_heads, scale_value, input_layout, num_key_value_heads, block_size, inner_precise);
}

::std::tuple<at::Tensor,at::Tensor> wrapper__npu_cent_select(const at::Tensor & query, const at::Tensor & l1_cent, const at::Tensor & block_ids, const at::Tensor & block_table, const at::Tensor & seq_len) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(query);
      c10_npu::check_npu_tensor_is_safe(l1_cent);
      c10_npu::check_npu_tensor_is_safe(block_ids);
      c10_npu::check_npu_tensor_is_safe(block_table);
      c10_npu::check_npu_tensor_is_safe(seq_len);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, query, "wrapper__npu_cent_select", "query");
  c10::impl::check_and_update_common_device(common_device, l1_cent, "wrapper__npu_cent_select", "l1_cent");
  c10::impl::check_and_update_common_device(common_device, block_ids, "wrapper__npu_cent_select", "block_ids");
  c10::impl::check_and_update_common_device(common_device, block_table, "wrapper__npu_cent_select", "block_table");
  c10::impl::check_and_update_common_device(common_device, seq_len, "wrapper__npu_cent_select", "seq_len");
  const c10::OptionalDeviceGuard device_guard(device_of(query));
  return op_plugin::npu_cent_select(query, l1_cent, block_ids, block_table, seq_len);
}

at::Tensor wrapper__npu_interleave_rope(const at::Tensor & x, const at::Tensor & cos, const at::Tensor & sin) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(x);
      c10_npu::check_npu_tensor_is_safe(cos);
      c10_npu::check_npu_tensor_is_safe(sin);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, x, "wrapper__npu_interleave_rope", "x");
  c10::impl::check_and_update_common_device(common_device, cos, "wrapper__npu_interleave_rope", "cos");
  c10::impl::check_and_update_common_device(common_device, sin, "wrapper__npu_interleave_rope", "sin");
  const c10::OptionalDeviceGuard device_guard(device_of(x));
  return op_plugin::npu_interleave_rope(x, cos, sin);
}

at::Tensor wrapper__npu_indexing(const at::Tensor & self, at::IntArrayRef begin, at::IntArrayRef end, at::IntArrayRef strides, int64_t begin_mask, int64_t end_mask, int64_t ellipsis_mask, int64_t new_axis_mask, int64_t shrink_axis_mask) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_indexing", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_indexing(self, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask);
}

at::Tensor & wrapper_out_npu_indexing_out(const at::Tensor & self, at::IntArrayRef begin, at::IntArrayRef end, at::IntArrayRef strides, int64_t begin_mask, int64_t end_mask, int64_t ellipsis_mask, int64_t new_axis_mask, int64_t shrink_axis_mask, at::Tensor & out) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(out);
      c10_npu::check_npu_tensor_is_safe(self);
  }
    // No device check
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_indexing_out(self, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask, out);
}

at::Tensor wrapper__npu_iou(const at::Tensor & bboxes, const at::Tensor & gtboxes, int64_t mode) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(bboxes);
      c10_npu::check_npu_tensor_is_safe(gtboxes);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, bboxes, "wrapper__npu_iou", "bboxes");
  c10::impl::check_and_update_common_device(common_device, gtboxes, "wrapper__npu_iou", "gtboxes");
  const c10::OptionalDeviceGuard device_guard(device_of(bboxes));
  return op_plugin::npu_iou(bboxes, gtboxes, mode);
}

at::Tensor wrapper__npu_layer_norm_eval(const at::Tensor & input, at::IntArrayRef normalized_shape, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, double eps) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(weight);
      c10_npu::check_npu_tensor_is_safe(bias);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_layer_norm_eval", "input");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__npu_layer_norm_eval", "weight");
  c10::impl::check_and_update_common_device(common_device, bias, "wrapper__npu_layer_norm_eval", "bias");
  const c10::OptionalDeviceGuard device_guard(device_of(input));
  return op_plugin::npu_layer_norm_eval(input, normalized_shape, weight, bias, eps);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapper__npu_layernorm_grad(const at::Tensor & grad_out, const at::Tensor & input, at::IntArrayRef normalized_shape, const at::Tensor & mean, const at::Tensor & rstd, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(grad_out);
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(mean);
      c10_npu::check_npu_tensor_is_safe(rstd);
      c10_npu::check_npu_tensor_is_safe(weight);
      c10_npu::check_npu_tensor_is_safe(bias);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad_out, "wrapper__npu_layernorm_grad", "grad_out");
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_layernorm_grad", "input");
  c10::impl::check_and_update_common_device(common_device, mean, "wrapper__npu_layernorm_grad", "mean");
  c10::impl::check_and_update_common_device(common_device, rstd, "wrapper__npu_layernorm_grad", "rstd");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__npu_layernorm_grad", "weight");
  c10::impl::check_and_update_common_device(common_device, bias, "wrapper__npu_layernorm_grad", "bias");
  const c10::OptionalDeviceGuard device_guard(device_of(grad_out));
  return op_plugin::npu_layernorm_grad(grad_out, input, normalized_shape, mean, rstd, weight, bias);
}

at::Tensor wrapper__npu_linear(const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(weight);
      c10_npu::check_npu_tensor_is_safe(bias);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_linear", "input");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__npu_linear", "weight");
  c10::impl::check_and_update_common_device(common_device, bias, "wrapper__npu_linear", "bias");
  const c10::OptionalDeviceGuard device_guard(device_of(input));
  return op_plugin::npu_linear(input, weight, bias);
}

::std::tuple<at::Tensor,at::Tensor> wrapper__npu_linear_backward(const at::Tensor & grad, const at::Tensor & input, const at::Tensor & weight) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(grad);
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(weight);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad, "wrapper__npu_linear_backward", "grad");
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_linear_backward", "input");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__npu_linear_backward", "weight");
  const c10::OptionalDeviceGuard device_guard(device_of(grad));
  return op_plugin::npu_linear_backward(grad, input, weight);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrapper__npu_lstm(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, const at::Tensor & seq_mask, const at::Tensor & h, const at::Tensor & c, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first, bool flag_seq, bool direction) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(weight);
      c10_npu::check_npu_tensor_is_safe(bias);
      c10_npu::check_npu_tensor_is_safe(seq_mask);
      c10_npu::check_npu_tensor_is_safe(h);
      c10_npu::check_npu_tensor_is_safe(c);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_lstm", "input");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__npu_lstm", "weight");
  c10::impl::check_and_update_common_device(common_device, bias, "wrapper__npu_lstm", "bias");
  c10::impl::check_and_update_common_device(common_device, seq_mask, "wrapper__npu_lstm", "seq_mask");
  c10::impl::check_and_update_common_device(common_device, h, "wrapper__npu_lstm", "h");
  c10::impl::check_and_update_common_device(common_device, c, "wrapper__npu_lstm", "c");
  const c10::OptionalDeviceGuard device_guard(device_of(input));
  return op_plugin::npu_lstm(input, weight, bias, seq_mask, h, c, has_biases, num_layers, dropout, train, bidirectional, batch_first, flag_seq, direction);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrapper__npu_lstm_backward(const ::std::optional<at::Tensor> & grady, const ::std::optional<at::Tensor> & gradh, const ::std::optional<at::Tensor> & gradc, const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, const at::Tensor & hx, const at::Tensor & cx, const at::Tensor & y_output, const at::Tensor & h_output, const at::Tensor & c_output, const at::Tensor & i, const at::Tensor & j, const at::Tensor & f, const at::Tensor & o, const at::Tensor & tanhc) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(grady);
      c10_npu::check_npu_tensor_is_safe(gradh);
      c10_npu::check_npu_tensor_is_safe(gradc);
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(weight);
      c10_npu::check_npu_tensor_is_safe(bias);
      c10_npu::check_npu_tensor_is_safe(hx);
      c10_npu::check_npu_tensor_is_safe(cx);
      c10_npu::check_npu_tensor_is_safe(y_output);
      c10_npu::check_npu_tensor_is_safe(h_output);
      c10_npu::check_npu_tensor_is_safe(c_output);
      c10_npu::check_npu_tensor_is_safe(i);
      c10_npu::check_npu_tensor_is_safe(j);
      c10_npu::check_npu_tensor_is_safe(f);
      c10_npu::check_npu_tensor_is_safe(o);
      c10_npu::check_npu_tensor_is_safe(tanhc);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grady, "wrapper__npu_lstm_backward", "grady");
  c10::impl::check_and_update_common_device(common_device, gradh, "wrapper__npu_lstm_backward", "gradh");
  c10::impl::check_and_update_common_device(common_device, gradc, "wrapper__npu_lstm_backward", "gradc");
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_lstm_backward", "input");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__npu_lstm_backward", "weight");
  c10::impl::check_and_update_common_device(common_device, bias, "wrapper__npu_lstm_backward", "bias");
  c10::impl::check_and_update_common_device(common_device, hx, "wrapper__npu_lstm_backward", "hx");
  c10::impl::check_and_update_common_device(common_device, cx, "wrapper__npu_lstm_backward", "cx");
  c10::impl::check_and_update_common_device(common_device, y_output, "wrapper__npu_lstm_backward", "y_output");
  c10::impl::check_and_update_common_device(common_device, h_output, "wrapper__npu_lstm_backward", "h_output");
  c10::impl::check_and_update_common_device(common_device, c_output, "wrapper__npu_lstm_backward", "c_output");
  c10::impl::check_and_update_common_device(common_device, i, "wrapper__npu_lstm_backward", "i");
  c10::impl::check_and_update_common_device(common_device, j, "wrapper__npu_lstm_backward", "j");
  c10::impl::check_and_update_common_device(common_device, f, "wrapper__npu_lstm_backward", "f");
  c10::impl::check_and_update_common_device(common_device, o, "wrapper__npu_lstm_backward", "o");
  c10::impl::check_and_update_common_device(common_device, tanhc, "wrapper__npu_lstm_backward", "tanhc");
  const c10::OptionalDeviceGuard device_guard(device_of(grady));
  return op_plugin::npu_lstm_backward(grady, gradh, gradc, input, weight, bias, hx, cx, y_output, h_output, c_output, i, j, f, o, tanhc);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrapper__npu_lstm_cell(const at::Tensor & input, const at::Tensor & w_ih, const at::Tensor & w_hh, const at::Tensor & h, const at::Tensor & c, const ::std::optional<at::Tensor> & b_ih, const ::std::optional<at::Tensor> & b_hh) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(w_ih);
      c10_npu::check_npu_tensor_is_safe(w_hh);
      c10_npu::check_npu_tensor_is_safe(h);
      c10_npu::check_npu_tensor_is_safe(c);
      c10_npu::check_npu_tensor_is_safe(b_ih);
      c10_npu::check_npu_tensor_is_safe(b_hh);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_lstm_cell", "input");
  c10::impl::check_and_update_common_device(common_device, w_ih, "wrapper__npu_lstm_cell", "w_ih");
  c10::impl::check_and_update_common_device(common_device, w_hh, "wrapper__npu_lstm_cell", "w_hh");
  c10::impl::check_and_update_common_device(common_device, h, "wrapper__npu_lstm_cell", "h");
  c10::impl::check_and_update_common_device(common_device, c, "wrapper__npu_lstm_cell", "c");
  c10::impl::check_and_update_common_device(common_device, b_ih, "wrapper__npu_lstm_cell", "b_ih");
  c10::impl::check_and_update_common_device(common_device, b_hh, "wrapper__npu_lstm_cell", "b_hh");
  const c10::OptionalDeviceGuard device_guard(device_of(input));
  return op_plugin::npu_lstm_cell(input, w_ih, w_hh, h, c, b_ih, b_hh);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrapper__npu_lstm_cell_backward(const ::std::optional<at::Tensor> & grady, const ::std::optional<at::Tensor> & gradh, const ::std::optional<at::Tensor> & gradc, const at::Tensor & input, const at::Tensor & w_ih, const at::Tensor & w_hh, const at::Tensor & h, const at::Tensor & c, const at::Tensor & y_output, const at::Tensor & h_output, const at::Tensor & c_output, const at::Tensor & i, const at::Tensor & j, const at::Tensor & f, const at::Tensor & o, const at::Tensor & tanhc) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(grady);
      c10_npu::check_npu_tensor_is_safe(gradh);
      c10_npu::check_npu_tensor_is_safe(gradc);
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(w_ih);
      c10_npu::check_npu_tensor_is_safe(w_hh);
      c10_npu::check_npu_tensor_is_safe(h);
      c10_npu::check_npu_tensor_is_safe(c);
      c10_npu::check_npu_tensor_is_safe(y_output);
      c10_npu::check_npu_tensor_is_safe(h_output);
      c10_npu::check_npu_tensor_is_safe(c_output);
      c10_npu::check_npu_tensor_is_safe(i);
      c10_npu::check_npu_tensor_is_safe(j);
      c10_npu::check_npu_tensor_is_safe(f);
      c10_npu::check_npu_tensor_is_safe(o);
      c10_npu::check_npu_tensor_is_safe(tanhc);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grady, "wrapper__npu_lstm_cell_backward", "grady");
  c10::impl::check_and_update_common_device(common_device, gradh, "wrapper__npu_lstm_cell_backward", "gradh");
  c10::impl::check_and_update_common_device(common_device, gradc, "wrapper__npu_lstm_cell_backward", "gradc");
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_lstm_cell_backward", "input");
  c10::impl::check_and_update_common_device(common_device, w_ih, "wrapper__npu_lstm_cell_backward", "w_ih");
  c10::impl::check_and_update_common_device(common_device, w_hh, "wrapper__npu_lstm_cell_backward", "w_hh");
  c10::impl::check_and_update_common_device(common_device, h, "wrapper__npu_lstm_cell_backward", "h");
  c10::impl::check_and_update_common_device(common_device, c, "wrapper__npu_lstm_cell_backward", "c");
  c10::impl::check_and_update_common_device(common_device, y_output, "wrapper__npu_lstm_cell_backward", "y_output");
  c10::impl::check_and_update_common_device(common_device, h_output, "wrapper__npu_lstm_cell_backward", "h_output");
  c10::impl::check_and_update_common_device(common_device, c_output, "wrapper__npu_lstm_cell_backward", "c_output");
  c10::impl::check_and_update_common_device(common_device, i, "wrapper__npu_lstm_cell_backward", "i");
  c10::impl::check_and_update_common_device(common_device, j, "wrapper__npu_lstm_cell_backward", "j");
  c10::impl::check_and_update_common_device(common_device, f, "wrapper__npu_lstm_cell_backward", "f");
  c10::impl::check_and_update_common_device(common_device, o, "wrapper__npu_lstm_cell_backward", "o");
  c10::impl::check_and_update_common_device(common_device, tanhc, "wrapper__npu_lstm_cell_backward", "tanhc");
  const c10::OptionalDeviceGuard device_guard(device_of(grady));
  return op_plugin::npu_lstm_cell_backward(grady, gradh, gradc, input, w_ih, w_hh, h, c, y_output, h_output, c_output, i, j, f, o, tanhc);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrapper__npu_lstm_data(const at::Tensor & input, const at::Tensor & batch_sizes, const at::Tensor & weight, const at::Tensor & bias, const at::Tensor & seq_mask, const at::Tensor & h, const at::Tensor & c, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first, bool flag_seq, bool direction) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(batch_sizes);
      c10_npu::check_npu_tensor_is_safe(weight);
      c10_npu::check_npu_tensor_is_safe(bias);
      c10_npu::check_npu_tensor_is_safe(seq_mask);
      c10_npu::check_npu_tensor_is_safe(h);
      c10_npu::check_npu_tensor_is_safe(c);
  }
    // No device check
  const c10::OptionalDeviceGuard device_guard(device_of(input));
  return op_plugin::npu_lstm_data(input, batch_sizes, weight, bias, seq_mask, h, c, has_biases, num_layers, dropout, train, bidirectional, batch_first, flag_seq, direction);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrapper__npu_lstm_data_backward(const ::std::optional<at::Tensor> & grady_opt, const ::std::optional<at::Tensor> & gradh_opt, const ::std::optional<at::Tensor> & gradc_opt, const at::Tensor & input, const at::Tensor & batch_sizes, const at::Tensor & weight, const at::Tensor & bias, const at::Tensor & init_h, const at::Tensor & init_c, const at::Tensor & y, const at::Tensor & h, const at::Tensor & c, const at::Tensor & i, const at::Tensor & j, const at::Tensor & f, const at::Tensor & o, const at::Tensor & tanhc, bool flag_direction) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(grady_opt);
      c10_npu::check_npu_tensor_is_safe(gradh_opt);
      c10_npu::check_npu_tensor_is_safe(gradc_opt);
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(batch_sizes);
      c10_npu::check_npu_tensor_is_safe(weight);
      c10_npu::check_npu_tensor_is_safe(bias);
      c10_npu::check_npu_tensor_is_safe(init_h);
      c10_npu::check_npu_tensor_is_safe(init_c);
      c10_npu::check_npu_tensor_is_safe(y);
      c10_npu::check_npu_tensor_is_safe(h);
      c10_npu::check_npu_tensor_is_safe(c);
      c10_npu::check_npu_tensor_is_safe(i);
      c10_npu::check_npu_tensor_is_safe(j);
      c10_npu::check_npu_tensor_is_safe(f);
      c10_npu::check_npu_tensor_is_safe(o);
      c10_npu::check_npu_tensor_is_safe(tanhc);
  }
    // No device check
  const c10::OptionalDeviceGuard device_guard(device_of(grady_opt));
  return op_plugin::npu_lstm_data_backward(grady_opt, gradh_opt, gradc_opt, input, batch_sizes, weight, bias, init_h, init_c, y, h, c, i, j, f, o, tanhc, flag_direction);
}

at::Tensor wrapper__npu_masked_fill_range(const at::Tensor & self, const at::Tensor & start, const at::Tensor & end, const at::Tensor & value, int64_t axis) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(start);
      c10_npu::check_npu_tensor_is_safe(end);
      c10_npu::check_npu_tensor_is_safe(value);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_masked_fill_range", "self");
  c10::impl::check_and_update_common_device(common_device, start, "wrapper__npu_masked_fill_range", "start");
  c10::impl::check_and_update_common_device(common_device, end, "wrapper__npu_masked_fill_range", "end");
  c10::impl::check_and_update_common_device(common_device, value, "wrapper__npu_masked_fill_range", "value");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_masked_fill_range(self, start, end, value, axis);
}

at::Tensor wrapper__npu_masked_softmax_with_rel_pos_bias(const at::Tensor & x, const ::std::optional<at::Tensor> & atten_mask, const at::Tensor & relative_pos_bias, double scale_value, int64_t inner_precision_mode) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(x);
      c10_npu::check_npu_tensor_is_safe(atten_mask);
      c10_npu::check_npu_tensor_is_safe(relative_pos_bias);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, x, "wrapper__npu_masked_softmax_with_rel_pos_bias", "x");
  c10::impl::check_and_update_common_device(common_device, atten_mask, "wrapper__npu_masked_softmax_with_rel_pos_bias", "atten_mask");
  c10::impl::check_and_update_common_device(common_device, relative_pos_bias, "wrapper__npu_masked_softmax_with_rel_pos_bias", "relative_pos_bias");
  const c10::OptionalDeviceGuard device_guard(device_of(x));
  return op_plugin::npu_masked_softmax_with_rel_pos_bias(x, atten_mask, relative_pos_bias, scale_value, inner_precision_mode);
}

::std::tuple<at::Tensor,at::Tensor> wrapper_dim_npu_max(const at::Tensor & self, int64_t dim, bool keepdim) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper_dim_npu_max", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_max(self, dim, keepdim);
}

::std::tuple<at::Tensor,at::Tensor> wrapper_names_dim_npu_max(const at::Tensor & self, at::Dimname dim, bool keepdim) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper_names_dim_npu_max", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_max(self, dim, keepdim);
}

at::Tensor wrapper__npu_max_backward(const at::Tensor & grad, int64_t dim, const at::Tensor & indices, c10::SymIntArrayRef sizes, bool keepdim) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(grad);
      c10_npu::check_npu_tensor_is_safe(indices);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad, "wrapper__npu_max_backward", "grad");
  c10::impl::check_and_update_common_device(common_device, indices, "wrapper__npu_max_backward", "indices");
  const c10::OptionalDeviceGuard device_guard(device_of(grad));
  return op_plugin::npu_max_backward_symint(grad, dim, indices, sizes, keepdim);
}

::std::tuple<at::Tensor,at::Tensor> wrapper_dim_npu_min(const at::Tensor & self, int64_t dim, bool keepdim) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper_dim_npu_min", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_min(self, dim, keepdim);
}

::std::tuple<at::Tensor,at::Tensor> wrapper_names_dim_npu_min(const at::Tensor & self, at::Dimname dim, bool keepdim) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper_names_dim_npu_min", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_min(self, dim, keepdim);
}

at::Tensor wrapper__npu_min_backward(const at::Tensor & grad, int64_t dim, const at::Tensor & indices, c10::SymIntArrayRef sizes, bool keepdim) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(grad);
      c10_npu::check_npu_tensor_is_safe(indices);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad, "wrapper__npu_min_backward", "grad");
  c10::impl::check_and_update_common_device(common_device, indices, "wrapper__npu_min_backward", "indices");
  const c10::OptionalDeviceGuard device_guard(device_of(grad));
  return op_plugin::npu_min_backward_symint(grad, dim, indices, sizes, keepdim);
}

at::Tensor wrapper__npu_mish(const at::Tensor & self) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_mish", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_mish(self);
}

at::Tensor wrapper__npu_mish_backward(const at::Tensor & grad, const at::Tensor & input) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(grad);
      c10_npu::check_npu_tensor_is_safe(input);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad, "wrapper__npu_mish_backward", "grad");
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_mish_backward", "input");
  const c10::OptionalDeviceGuard device_guard(device_of(grad));
  return op_plugin::npu_mish_backward(grad, input);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrapper__npu_mla_prolog(const at::Tensor & token_x, const at::Tensor & weight_dq, const at::Tensor & weight_uq_qr, const at::Tensor & weight_uk, const at::Tensor & weight_dkv_kr, const at::Tensor & rmsnorm_gamma_cq, const at::Tensor & rmsnorm_gamma_ckv, const at::Tensor & rope_sin, const at::Tensor & rope_cos, const at::Tensor & cache_index, const at::Tensor & kv_cache, const at::Tensor & kr_cache, const ::std::optional<at::Tensor> & dequant_scale_x, const ::std::optional<at::Tensor> & dequant_scale_w_dq, const ::std::optional<at::Tensor> & dequant_scale_w_uq_qr, const ::std::optional<at::Tensor> & dequant_scale_w_dkv_kr, const ::std::optional<at::Tensor> & quant_scale_ckv, const ::std::optional<at::Tensor> & quant_scale_ckr, const ::std::optional<at::Tensor> & smooth_scales_cq, double rmsnorm_epsilon_cq, double rmsnorm_epsilon_ckv, c10::string_view cache_mode) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(token_x);
      c10_npu::check_npu_tensor_is_safe(weight_dq);
      c10_npu::check_npu_tensor_is_safe(weight_uq_qr);
      c10_npu::check_npu_tensor_is_safe(weight_uk);
      c10_npu::check_npu_tensor_is_safe(weight_dkv_kr);
      c10_npu::check_npu_tensor_is_safe(rmsnorm_gamma_cq);
      c10_npu::check_npu_tensor_is_safe(rmsnorm_gamma_ckv);
      c10_npu::check_npu_tensor_is_safe(rope_sin);
      c10_npu::check_npu_tensor_is_safe(rope_cos);
      c10_npu::check_npu_tensor_is_safe(cache_index);
      c10_npu::check_npu_tensor_is_safe(kv_cache);
      c10_npu::check_npu_tensor_is_safe(kr_cache);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, token_x, "wrapper__npu_mla_prolog", "token_x");
  c10::impl::check_and_update_common_device(common_device, weight_dq, "wrapper__npu_mla_prolog", "weight_dq");
  c10::impl::check_and_update_common_device(common_device, weight_uq_qr, "wrapper__npu_mla_prolog", "weight_uq_qr");
  c10::impl::check_and_update_common_device(common_device, weight_uk, "wrapper__npu_mla_prolog", "weight_uk");
  c10::impl::check_and_update_common_device(common_device, weight_dkv_kr, "wrapper__npu_mla_prolog", "weight_dkv_kr");
  c10::impl::check_and_update_common_device(common_device, rmsnorm_gamma_cq, "wrapper__npu_mla_prolog", "rmsnorm_gamma_cq");
  c10::impl::check_and_update_common_device(common_device, rmsnorm_gamma_ckv, "wrapper__npu_mla_prolog", "rmsnorm_gamma_ckv");
  c10::impl::check_and_update_common_device(common_device, rope_sin, "wrapper__npu_mla_prolog", "rope_sin");
  c10::impl::check_and_update_common_device(common_device, rope_cos, "wrapper__npu_mla_prolog", "rope_cos");
  c10::impl::check_and_update_common_device(common_device, cache_index, "wrapper__npu_mla_prolog", "cache_index");
  c10::impl::check_and_update_common_device(common_device, kv_cache, "wrapper__npu_mla_prolog", "kv_cache");
  c10::impl::check_and_update_common_device(common_device, kr_cache, "wrapper__npu_mla_prolog", "kr_cache");
  c10::impl::check_and_update_common_device(common_device, dequant_scale_x, "wrapper__npu_mla_prolog", "dequant_scale_x");
  c10::impl::check_and_update_common_device(common_device, dequant_scale_w_dq, "wrapper__npu_mla_prolog", "dequant_scale_w_dq");
  c10::impl::check_and_update_common_device(common_device, dequant_scale_w_uq_qr, "wrapper__npu_mla_prolog", "dequant_scale_w_uq_qr");
  c10::impl::check_and_update_common_device(common_device, dequant_scale_w_dkv_kr, "wrapper__npu_mla_prolog", "dequant_scale_w_dkv_kr");
  c10::impl::check_and_update_common_device(common_device, quant_scale_ckv, "wrapper__npu_mla_prolog", "quant_scale_ckv");
  c10::impl::check_and_update_common_device(common_device, quant_scale_ckr, "wrapper__npu_mla_prolog", "quant_scale_ckr");
  c10::impl::check_and_update_common_device(common_device, smooth_scales_cq, "wrapper__npu_mla_prolog", "smooth_scales_cq");
  const c10::OptionalDeviceGuard device_guard(device_of(token_x));
  return op_plugin::npu_mla_prolog(token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uq_qr, dequant_scale_w_dkv_kr, quant_scale_ckv, quant_scale_ckr, smooth_scales_cq, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv, cache_mode);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrapper__npu_mla_prolog_v2(const at::Tensor & token_x, const at::Tensor & weight_dq, const at::Tensor & weight_uq_qr, const at::Tensor & weight_uk, const at::Tensor & weight_dkv_kr, const at::Tensor & rmsnorm_gamma_cq, const at::Tensor & rmsnorm_gamma_ckv, const at::Tensor & rope_sin, const at::Tensor & rope_cos, const at::Tensor & cache_index, const at::Tensor & kv_cache, const at::Tensor & kr_cache, const ::std::optional<at::Tensor> & dequant_scale_x, const ::std::optional<at::Tensor> & dequant_scale_w_dq, const ::std::optional<at::Tensor> & dequant_scale_w_uq_qr, const ::std::optional<at::Tensor> & dequant_scale_w_dkv_kr, const ::std::optional<at::Tensor> & quant_scale_ckv, const ::std::optional<at::Tensor> & quant_scale_ckr, const ::std::optional<at::Tensor> & smooth_scales_cq, double rmsnorm_epsilon_cq, double rmsnorm_epsilon_ckv, c10::string_view cache_mode) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(token_x);
      c10_npu::check_npu_tensor_is_safe(weight_dq);
      c10_npu::check_npu_tensor_is_safe(weight_uq_qr);
      c10_npu::check_npu_tensor_is_safe(weight_uk);
      c10_npu::check_npu_tensor_is_safe(weight_dkv_kr);
      c10_npu::check_npu_tensor_is_safe(rmsnorm_gamma_cq);
      c10_npu::check_npu_tensor_is_safe(rmsnorm_gamma_ckv);
      c10_npu::check_npu_tensor_is_safe(rope_sin);
      c10_npu::check_npu_tensor_is_safe(rope_cos);
      c10_npu::check_npu_tensor_is_safe(cache_index);
      c10_npu::check_npu_tensor_is_safe(kv_cache);
      c10_npu::check_npu_tensor_is_safe(kr_cache);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, token_x, "wrapper__npu_mla_prolog_v2", "token_x");
  c10::impl::check_and_update_common_device(common_device, weight_dq, "wrapper__npu_mla_prolog_v2", "weight_dq");
  c10::impl::check_and_update_common_device(common_device, weight_uq_qr, "wrapper__npu_mla_prolog_v2", "weight_uq_qr");
  c10::impl::check_and_update_common_device(common_device, weight_uk, "wrapper__npu_mla_prolog_v2", "weight_uk");
  c10::impl::check_and_update_common_device(common_device, weight_dkv_kr, "wrapper__npu_mla_prolog_v2", "weight_dkv_kr");
  c10::impl::check_and_update_common_device(common_device, rmsnorm_gamma_cq, "wrapper__npu_mla_prolog_v2", "rmsnorm_gamma_cq");
  c10::impl::check_and_update_common_device(common_device, rmsnorm_gamma_ckv, "wrapper__npu_mla_prolog_v2", "rmsnorm_gamma_ckv");
  c10::impl::check_and_update_common_device(common_device, rope_sin, "wrapper__npu_mla_prolog_v2", "rope_sin");
  c10::impl::check_and_update_common_device(common_device, rope_cos, "wrapper__npu_mla_prolog_v2", "rope_cos");
  c10::impl::check_and_update_common_device(common_device, cache_index, "wrapper__npu_mla_prolog_v2", "cache_index");
  c10::impl::check_and_update_common_device(common_device, kv_cache, "wrapper__npu_mla_prolog_v2", "kv_cache");
  c10::impl::check_and_update_common_device(common_device, kr_cache, "wrapper__npu_mla_prolog_v2", "kr_cache");
  c10::impl::check_and_update_common_device(common_device, dequant_scale_x, "wrapper__npu_mla_prolog_v2", "dequant_scale_x");
  c10::impl::check_and_update_common_device(common_device, dequant_scale_w_dq, "wrapper__npu_mla_prolog_v2", "dequant_scale_w_dq");
  c10::impl::check_and_update_common_device(common_device, dequant_scale_w_uq_qr, "wrapper__npu_mla_prolog_v2", "dequant_scale_w_uq_qr");
  c10::impl::check_and_update_common_device(common_device, dequant_scale_w_dkv_kr, "wrapper__npu_mla_prolog_v2", "dequant_scale_w_dkv_kr");
  c10::impl::check_and_update_common_device(common_device, quant_scale_ckv, "wrapper__npu_mla_prolog_v2", "quant_scale_ckv");
  c10::impl::check_and_update_common_device(common_device, quant_scale_ckr, "wrapper__npu_mla_prolog_v2", "quant_scale_ckr");
  c10::impl::check_and_update_common_device(common_device, smooth_scales_cq, "wrapper__npu_mla_prolog_v2", "smooth_scales_cq");
  const c10::OptionalDeviceGuard device_guard(device_of(token_x));
  return op_plugin::npu_mla_prolog_v2(token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uq_qr, dequant_scale_w_dkv_kr, quant_scale_ckv, quant_scale_ckr, smooth_scales_cq, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv, cache_mode);
}

at::Tensor wrapper__npu_mm_all_reduce_base(const at::Tensor & x1, const at::Tensor & x2, c10::string_view hcom, c10::string_view reduce_op, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & antiquant_scale, const ::std::optional<at::Tensor> & antiquant_offset, const ::std::optional<at::Tensor> & x3, const ::std::optional<at::Tensor> & dequant_scale, const ::std::optional<at::Tensor> & pertoken_scale, const ::std::optional<at::Tensor> & comm_quant_scale_1, const ::std::optional<at::Tensor> & comm_quant_scale_2, int64_t antiquant_group_size, int64_t comm_turn) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(x1);
      c10_npu::check_npu_tensor_is_safe(x2);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, x1, "wrapper__npu_mm_all_reduce_base", "x1");
  c10::impl::check_and_update_common_device(common_device, x2, "wrapper__npu_mm_all_reduce_base", "x2");
  c10::impl::check_and_update_common_device(common_device, bias, "wrapper__npu_mm_all_reduce_base", "bias");
  c10::impl::check_and_update_common_device(common_device, antiquant_scale, "wrapper__npu_mm_all_reduce_base", "antiquant_scale");
  c10::impl::check_and_update_common_device(common_device, antiquant_offset, "wrapper__npu_mm_all_reduce_base", "antiquant_offset");
  c10::impl::check_and_update_common_device(common_device, x3, "wrapper__npu_mm_all_reduce_base", "x3");
  c10::impl::check_and_update_common_device(common_device, dequant_scale, "wrapper__npu_mm_all_reduce_base", "dequant_scale");
  c10::impl::check_and_update_common_device(common_device, pertoken_scale, "wrapper__npu_mm_all_reduce_base", "pertoken_scale");
  c10::impl::check_and_update_common_device(common_device, comm_quant_scale_1, "wrapper__npu_mm_all_reduce_base", "comm_quant_scale_1");
  c10::impl::check_and_update_common_device(common_device, comm_quant_scale_2, "wrapper__npu_mm_all_reduce_base", "comm_quant_scale_2");
  const c10::OptionalDeviceGuard device_guard(device_of(x1));
  return op_plugin::npu_mm_all_reduce_base(x1, x2, hcom, reduce_op, bias, antiquant_scale, antiquant_offset, x3, dequant_scale, pertoken_scale, comm_quant_scale_1, comm_quant_scale_2, antiquant_group_size, comm_turn);
}

at::Tensor wrapper__npu_mm_reduce_scatter_base(const at::Tensor & self, const at::Tensor & x2, c10::string_view hcom, int64_t world_size, c10::string_view reduce_op, const ::std::optional<at::Tensor> & bias, int64_t comm_turn) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(x2);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_mm_reduce_scatter_base", "self");
  c10::impl::check_and_update_common_device(common_device, x2, "wrapper__npu_mm_reduce_scatter_base", "x2");
  c10::impl::check_and_update_common_device(common_device, bias, "wrapper__npu_mm_reduce_scatter_base", "bias");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_mm_reduce_scatter_base(self, x2, hcom, world_size, reduce_op, bias, comm_turn);
}

at::Tensor wrapper__npu_moe_compute_expert_tokens(const at::Tensor & sorted_expert_for_source_row, int64_t num_expert) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(sorted_expert_for_source_row);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, sorted_expert_for_source_row, "wrapper__npu_moe_compute_expert_tokens", "sorted_expert_for_source_row");
  const c10::OptionalDeviceGuard device_guard(device_of(sorted_expert_for_source_row));
  return op_plugin::npu_moe_compute_expert_tokens(sorted_expert_for_source_row, num_expert);
}

at::Tensor wrapper__npu_moe_finalize_routing(const at::Tensor & expanded_permuted_rows, const ::std::optional<at::Tensor> & skip1, const ::std::optional<at::Tensor> & skip2, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & scales, const at::Tensor & expanded_src_to_dst_row, const ::std::optional<at::Tensor> & export_for_source_row, ::std::optional<int64_t> drop_pad_mode) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(expanded_permuted_rows);
      c10_npu::check_npu_tensor_is_safe(skip1);
      c10_npu::check_npu_tensor_is_safe(skip2);
      c10_npu::check_npu_tensor_is_safe(bias);
      c10_npu::check_npu_tensor_is_safe(scales);
      c10_npu::check_npu_tensor_is_safe(expanded_src_to_dst_row);
      c10_npu::check_npu_tensor_is_safe(export_for_source_row);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, expanded_permuted_rows, "wrapper__npu_moe_finalize_routing", "expanded_permuted_rows");
  c10::impl::check_and_update_common_device(common_device, skip1, "wrapper__npu_moe_finalize_routing", "skip1");
  c10::impl::check_and_update_common_device(common_device, skip2, "wrapper__npu_moe_finalize_routing", "skip2");
  c10::impl::check_and_update_common_device(common_device, bias, "wrapper__npu_moe_finalize_routing", "bias");
  c10::impl::check_and_update_common_device(common_device, scales, "wrapper__npu_moe_finalize_routing", "scales");
  c10::impl::check_and_update_common_device(common_device, expanded_src_to_dst_row, "wrapper__npu_moe_finalize_routing", "expanded_src_to_dst_row");
  c10::impl::check_and_update_common_device(common_device, export_for_source_row, "wrapper__npu_moe_finalize_routing", "export_for_source_row");
  const c10::OptionalDeviceGuard device_guard(device_of(expanded_permuted_rows));
  return op_plugin::npu_moe_finalize_routing(expanded_permuted_rows, skip1, skip2, bias, scales, expanded_src_to_dst_row, export_for_source_row, drop_pad_mode);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapper__npu_moe_gating_top_k_softmax(const at::Tensor & x, const ::std::optional<at::Tensor> & finished, int64_t k) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(x);
      c10_npu::check_npu_tensor_is_safe(finished);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, x, "wrapper__npu_moe_gating_top_k_softmax", "x");
  c10::impl::check_and_update_common_device(common_device, finished, "wrapper__npu_moe_gating_top_k_softmax", "finished");
  const c10::OptionalDeviceGuard device_guard(device_of(x));
  return op_plugin::npu_moe_gating_top_k_softmax(x, finished, k);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapper__npu_moe_gating_top_k(const at::Tensor & x, int64_t k, const ::std::optional<at::Tensor> & bias, int64_t k_group, int64_t group_count, int64_t group_select_mode, int64_t renorm, int64_t norm_type, bool out_flag, double routed_scaling_factor, double eps) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(x);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, x, "wrapper__npu_moe_gating_top_k", "x");
  c10::impl::check_and_update_common_device(common_device, bias, "wrapper__npu_moe_gating_top_k", "bias");
  const c10::OptionalDeviceGuard device_guard(device_of(x));
  return op_plugin::npu_moe_gating_top_k(x, k, bias, k_group, group_count, group_select_mode, renorm, norm_type, out_flag, routed_scaling_factor, eps);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapper__npu_moe_init_routing(const at::Tensor & x, const at::Tensor & row_idx, const at::Tensor & expert_idx, int64_t active_num) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(x);
      c10_npu::check_npu_tensor_is_safe(row_idx);
      c10_npu::check_npu_tensor_is_safe(expert_idx);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, x, "wrapper__npu_moe_init_routing", "x");
  c10::impl::check_and_update_common_device(common_device, row_idx, "wrapper__npu_moe_init_routing", "row_idx");
  c10::impl::check_and_update_common_device(common_device, expert_idx, "wrapper__npu_moe_init_routing", "expert_idx");
  const c10::OptionalDeviceGuard device_guard(device_of(x));
  return op_plugin::npu_moe_init_routing(x, row_idx, expert_idx, active_num);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrapper__npu_moe_init_routing_v2(const at::Tensor & x, const at::Tensor & expert_idx, const ::std::optional<at::Tensor> & scale, const ::std::optional<at::Tensor> & offset, int64_t active_num, int64_t expert_capacity, int64_t expert_num, int64_t drop_pad_mode, int64_t expert_tokens_num_type, bool expert_tokens_num_flag, int64_t quant_mode, at::IntArrayRef active_expert_range, int64_t row_idx_type) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(x);
      c10_npu::check_npu_tensor_is_safe(expert_idx);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, x, "wrapper__npu_moe_init_routing_v2", "x");
  c10::impl::check_and_update_common_device(common_device, expert_idx, "wrapper__npu_moe_init_routing_v2", "expert_idx");
  c10::impl::check_and_update_common_device(common_device, scale, "wrapper__npu_moe_init_routing_v2", "scale");
  c10::impl::check_and_update_common_device(common_device, offset, "wrapper__npu_moe_init_routing_v2", "offset");
  const c10::OptionalDeviceGuard device_guard(device_of(x));
  return op_plugin::npu_moe_init_routing_v2(x, expert_idx, scale, offset, active_num, expert_capacity, expert_num, drop_pad_mode, expert_tokens_num_type, expert_tokens_num_flag, quant_mode, active_expert_range, row_idx_type);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapper__npu_grouped_matmul_swiglu_quant(const at::Tensor & x, const at::Tensor & weight, const at::Tensor & group_list, const at::Tensor & weight_scale, const at::Tensor & x_scale, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & offset) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(x);
      c10_npu::check_npu_tensor_is_safe(weight);
      c10_npu::check_npu_tensor_is_safe(group_list);
      c10_npu::check_npu_tensor_is_safe(weight_scale);
      c10_npu::check_npu_tensor_is_safe(x_scale);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, x, "wrapper__npu_grouped_matmul_swiglu_quant", "x");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__npu_grouped_matmul_swiglu_quant", "weight");
  c10::impl::check_and_update_common_device(common_device, group_list, "wrapper__npu_grouped_matmul_swiglu_quant", "group_list");
  c10::impl::check_and_update_common_device(common_device, weight_scale, "wrapper__npu_grouped_matmul_swiglu_quant", "weight_scale");
  c10::impl::check_and_update_common_device(common_device, x_scale, "wrapper__npu_grouped_matmul_swiglu_quant", "x_scale");
  c10::impl::check_and_update_common_device(common_device, bias, "wrapper__npu_grouped_matmul_swiglu_quant", "bias");
  c10::impl::check_and_update_common_device(common_device, offset, "wrapper__npu_grouped_matmul_swiglu_quant", "offset");
  const c10::OptionalDeviceGuard device_guard(device_of(x));
  return op_plugin::npu_grouped_matmul_swiglu_quant(x, weight, group_list, weight_scale, x_scale, bias, offset);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrapper__npu_moe_distribute_dispatch(const at::Tensor & x, const at::Tensor & expert_ids, c10::string_view group_ep, int64_t ep_world_size, int64_t ep_rank_id, int64_t moe_expert_num, const ::std::optional<at::Tensor> & scales, const ::std::optional<at::Tensor> & x_active_mask, const ::std::optional<at::Tensor> & expert_scales, c10::string_view group_tp, int64_t tp_world_size, int64_t tp_rank_id, int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num, int64_t quant_mode, int64_t global_bs, int64_t expert_token_nums_type) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(x);
      c10_npu::check_npu_tensor_is_safe(expert_ids);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, x, "wrapper__npu_moe_distribute_dispatch", "x");
  c10::impl::check_and_update_common_device(common_device, expert_ids, "wrapper__npu_moe_distribute_dispatch", "expert_ids");
  c10::impl::check_and_update_common_device(common_device, scales, "wrapper__npu_moe_distribute_dispatch", "scales");
  c10::impl::check_and_update_common_device(common_device, x_active_mask, "wrapper__npu_moe_distribute_dispatch", "x_active_mask");
  c10::impl::check_and_update_common_device(common_device, expert_scales, "wrapper__npu_moe_distribute_dispatch", "expert_scales");
  const c10::OptionalDeviceGuard device_guard(device_of(x));
  return op_plugin::npu_moe_distribute_dispatch(x, expert_ids, group_ep, ep_world_size, ep_rank_id, moe_expert_num, scales, x_active_mask, expert_scales, group_tp, tp_world_size, tp_rank_id, expert_shard_type, shared_expert_num, shared_expert_rank_num, quant_mode, global_bs, expert_token_nums_type);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrapper__npu_moe_distribute_dispatch_v2(const at::Tensor & x, const at::Tensor & expert_ids, c10::string_view group_ep, int64_t ep_world_size, int64_t ep_rank_id, int64_t moe_expert_num, const ::std::optional<at::Tensor> & scales, const ::std::optional<at::Tensor> & x_active_mask, const ::std::optional<at::Tensor> & expert_scales, c10::string_view group_tp, int64_t tp_world_size, int64_t tp_rank_id, int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num, int64_t quant_mode, int64_t global_bs, int64_t expert_token_nums_type) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(x);
      c10_npu::check_npu_tensor_is_safe(expert_ids);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, x, "wrapper__npu_moe_distribute_dispatch_v2", "x");
  c10::impl::check_and_update_common_device(common_device, expert_ids, "wrapper__npu_moe_distribute_dispatch_v2", "expert_ids");
  c10::impl::check_and_update_common_device(common_device, scales, "wrapper__npu_moe_distribute_dispatch_v2", "scales");
  c10::impl::check_and_update_common_device(common_device, x_active_mask, "wrapper__npu_moe_distribute_dispatch_v2", "x_active_mask");
  c10::impl::check_and_update_common_device(common_device, expert_scales, "wrapper__npu_moe_distribute_dispatch_v2", "expert_scales");
  const c10::OptionalDeviceGuard device_guard(device_of(x));
  return op_plugin::npu_moe_distribute_dispatch_v2(x, expert_ids, group_ep, ep_world_size, ep_rank_id, moe_expert_num, scales, x_active_mask, expert_scales, group_tp, tp_world_size, tp_rank_id, expert_shard_type, shared_expert_num, shared_expert_rank_num, quant_mode, global_bs, expert_token_nums_type);
}

at::Tensor wrapper__npu_moe_distribute_combine(const at::Tensor & expand_x, const at::Tensor & expert_ids, const at::Tensor & expand_idx, const at::Tensor & ep_send_counts, const at::Tensor & expert_scales, c10::string_view group_ep, int64_t ep_world_size, int64_t ep_rank_id, int64_t moe_expert_num, const ::std::optional<at::Tensor> & tp_send_counts, const ::std::optional<at::Tensor> & x_active_mask, const ::std::optional<at::Tensor> & activation_scale, const ::std::optional<at::Tensor> & weight_scale, const ::std::optional<at::Tensor> & group_list, const ::std::optional<at::Tensor> & expand_scales, const ::std::optional<at::Tensor> & shared_expert_x, c10::string_view group_tp, int64_t tp_world_size, int64_t tp_rank_id, int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num, int64_t global_bs, int64_t out_dtype, int64_t comm_quant_mode, int64_t group_list_type) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(expand_x);
      c10_npu::check_npu_tensor_is_safe(expert_ids);
      c10_npu::check_npu_tensor_is_safe(expand_idx);
      c10_npu::check_npu_tensor_is_safe(ep_send_counts);
      c10_npu::check_npu_tensor_is_safe(expert_scales);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, expand_x, "wrapper__npu_moe_distribute_combine", "expand_x");
  c10::impl::check_and_update_common_device(common_device, expert_ids, "wrapper__npu_moe_distribute_combine", "expert_ids");
  c10::impl::check_and_update_common_device(common_device, expand_idx, "wrapper__npu_moe_distribute_combine", "expand_idx");
  c10::impl::check_and_update_common_device(common_device, ep_send_counts, "wrapper__npu_moe_distribute_combine", "ep_send_counts");
  c10::impl::check_and_update_common_device(common_device, expert_scales, "wrapper__npu_moe_distribute_combine", "expert_scales");
  c10::impl::check_and_update_common_device(common_device, tp_send_counts, "wrapper__npu_moe_distribute_combine", "tp_send_counts");
  c10::impl::check_and_update_common_device(common_device, x_active_mask, "wrapper__npu_moe_distribute_combine", "x_active_mask");
  c10::impl::check_and_update_common_device(common_device, activation_scale, "wrapper__npu_moe_distribute_combine", "activation_scale");
  c10::impl::check_and_update_common_device(common_device, weight_scale, "wrapper__npu_moe_distribute_combine", "weight_scale");
  c10::impl::check_and_update_common_device(common_device, group_list, "wrapper__npu_moe_distribute_combine", "group_list");
  c10::impl::check_and_update_common_device(common_device, expand_scales, "wrapper__npu_moe_distribute_combine", "expand_scales");
  c10::impl::check_and_update_common_device(common_device, shared_expert_x, "wrapper__npu_moe_distribute_combine", "shared_expert_x");
  const c10::OptionalDeviceGuard device_guard(device_of(expand_x));
  return op_plugin::npu_moe_distribute_combine(expand_x, expert_ids, expand_idx, ep_send_counts, expert_scales, group_ep, ep_world_size, ep_rank_id, moe_expert_num, tp_send_counts, x_active_mask, activation_scale, weight_scale, group_list, expand_scales, shared_expert_x, group_tp, tp_world_size, tp_rank_id, expert_shard_type, shared_expert_num, shared_expert_rank_num, global_bs, out_dtype, comm_quant_mode, group_list_type);
}

at::Tensor wrapper__npu_moe_distribute_combine_v2(const at::Tensor & expand_x, const at::Tensor & expert_ids, const at::Tensor & assist_info_for_combine, const at::Tensor & ep_send_counts, const at::Tensor & expert_scales, c10::string_view group_ep, int64_t ep_world_size, int64_t ep_rank_id, int64_t moe_expert_num, const ::std::optional<at::Tensor> & tp_send_counts, const ::std::optional<at::Tensor> & x_active_mask, const ::std::optional<at::Tensor> & expand_scales, const ::std::optional<at::Tensor> & shared_expert_x, c10::string_view group_tp, int64_t tp_world_size, int64_t tp_rank_id, int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num, int64_t global_bs, int64_t comm_quant_mode) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(expand_x);
      c10_npu::check_npu_tensor_is_safe(expert_ids);
      c10_npu::check_npu_tensor_is_safe(assist_info_for_combine);
      c10_npu::check_npu_tensor_is_safe(ep_send_counts);
      c10_npu::check_npu_tensor_is_safe(expert_scales);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, expand_x, "wrapper__npu_moe_distribute_combine_v2", "expand_x");
  c10::impl::check_and_update_common_device(common_device, expert_ids, "wrapper__npu_moe_distribute_combine_v2", "expert_ids");
  c10::impl::check_and_update_common_device(common_device, assist_info_for_combine, "wrapper__npu_moe_distribute_combine_v2", "assist_info_for_combine");
  c10::impl::check_and_update_common_device(common_device, ep_send_counts, "wrapper__npu_moe_distribute_combine_v2", "ep_send_counts");
  c10::impl::check_and_update_common_device(common_device, expert_scales, "wrapper__npu_moe_distribute_combine_v2", "expert_scales");
  c10::impl::check_and_update_common_device(common_device, tp_send_counts, "wrapper__npu_moe_distribute_combine_v2", "tp_send_counts");
  c10::impl::check_and_update_common_device(common_device, x_active_mask, "wrapper__npu_moe_distribute_combine_v2", "x_active_mask");
  c10::impl::check_and_update_common_device(common_device, expand_scales, "wrapper__npu_moe_distribute_combine_v2", "expand_scales");
  c10::impl::check_and_update_common_device(common_device, shared_expert_x, "wrapper__npu_moe_distribute_combine_v2", "shared_expert_x");
  const c10::OptionalDeviceGuard device_guard(device_of(expand_x));
  return op_plugin::npu_moe_distribute_combine_v2(expand_x, expert_ids, assist_info_for_combine, ep_send_counts, expert_scales, group_ep, ep_world_size, ep_rank_id, moe_expert_num, tp_send_counts, x_active_mask, expand_scales, shared_expert_x, group_tp, tp_world_size, tp_rank_id, expert_shard_type, shared_expert_num, shared_expert_rank_num, global_bs, comm_quant_mode);
}

at::Tensor wrapper___npu_distribute_barrier(const at::Tensor & x_ref, c10::string_view group, int64_t world_size) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(x_ref);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, x_ref, "wrapper___npu_distribute_barrier", "x_ref");
  const c10::OptionalDeviceGuard device_guard(device_of(x_ref));
  return op_plugin::_npu_distribute_barrier(x_ref, group, world_size);
}

at::Tensor wrapper__npu_moe_eplb_update_expert(const at::Tensor & expert_ids, const at::Tensor & eplb_table, int64_t local_rank_id, int64_t world_size, int64_t balance_mode) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(expert_ids);
      c10_npu::check_npu_tensor_is_safe(eplb_table);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, expert_ids, "wrapper__npu_moe_eplb_update_expert", "expert_ids");
  c10::impl::check_and_update_common_device(common_device, eplb_table, "wrapper__npu_moe_eplb_update_expert", "eplb_table");
  const c10::OptionalDeviceGuard device_guard(device_of(expert_ids));
  return op_plugin::npu_moe_eplb_update_expert(expert_ids, eplb_table, local_rank_id, world_size, balance_mode);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrapper__npu_moe_re_routing(const at::Tensor & tokens, const at::Tensor & expert_token_num_per_rank, const ::std::optional<at::Tensor> & per_token_scales, int64_t expert_token_num_type, int64_t idx_type) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(tokens);
      c10_npu::check_npu_tensor_is_safe(expert_token_num_per_rank);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, tokens, "wrapper__npu_moe_re_routing", "tokens");
  c10::impl::check_and_update_common_device(common_device, expert_token_num_per_rank, "wrapper__npu_moe_re_routing", "expert_token_num_per_rank");
  c10::impl::check_and_update_common_device(common_device, per_token_scales, "wrapper__npu_moe_re_routing", "per_token_scales");
  const c10::OptionalDeviceGuard device_guard(device_of(tokens));
  return op_plugin::npu_moe_re_routing(tokens, expert_token_num_per_rank, per_token_scales, expert_token_num_type, idx_type);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapper__npu_moe_distribute_combine_add_rms_norm(const at::Tensor & expand_x, const at::Tensor & expert_ids, const at::Tensor & expand_idx, const at::Tensor & ep_send_counts, const at::Tensor & expert_scales, const at::Tensor & residual_x, const at::Tensor & gamma, c10::string_view group_ep, int64_t ep_world_size, int64_t ep_rank_id, int64_t moe_expert_num, const ::std::optional<at::Tensor> & tp_send_counts, const ::std::optional<at::Tensor> & x_active_mask, const ::std::optional<at::Tensor> & activation_scale, const ::std::optional<at::Tensor> & weight_scale, const ::std::optional<at::Tensor> & group_list, const ::std::optional<at::Tensor> & expand_scales, const ::std::optional<at::Tensor> & shared_expert_x, c10::string_view group_tp, int64_t tp_world_size, int64_t tp_rank_id, int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num, int64_t global_bs, int64_t out_dtype, int64_t comm_quant_mode, int64_t group_list_type, c10::string_view comm_alg, double norm_eps) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(expand_x);
      c10_npu::check_npu_tensor_is_safe(expert_ids);
      c10_npu::check_npu_tensor_is_safe(expand_idx);
      c10_npu::check_npu_tensor_is_safe(ep_send_counts);
      c10_npu::check_npu_tensor_is_safe(expert_scales);
      c10_npu::check_npu_tensor_is_safe(residual_x);
      c10_npu::check_npu_tensor_is_safe(gamma);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, expand_x, "wrapper__npu_moe_distribute_combine_add_rms_norm", "expand_x");
  c10::impl::check_and_update_common_device(common_device, expert_ids, "wrapper__npu_moe_distribute_combine_add_rms_norm", "expert_ids");
  c10::impl::check_and_update_common_device(common_device, expand_idx, "wrapper__npu_moe_distribute_combine_add_rms_norm", "expand_idx");
  c10::impl::check_and_update_common_device(common_device, ep_send_counts, "wrapper__npu_moe_distribute_combine_add_rms_norm", "ep_send_counts");
  c10::impl::check_and_update_common_device(common_device, expert_scales, "wrapper__npu_moe_distribute_combine_add_rms_norm", "expert_scales");
  c10::impl::check_and_update_common_device(common_device, residual_x, "wrapper__npu_moe_distribute_combine_add_rms_norm", "residual_x");
  c10::impl::check_and_update_common_device(common_device, gamma, "wrapper__npu_moe_distribute_combine_add_rms_norm", "gamma");
  c10::impl::check_and_update_common_device(common_device, tp_send_counts, "wrapper__npu_moe_distribute_combine_add_rms_norm", "tp_send_counts");
  c10::impl::check_and_update_common_device(common_device, x_active_mask, "wrapper__npu_moe_distribute_combine_add_rms_norm", "x_active_mask");
  c10::impl::check_and_update_common_device(common_device, activation_scale, "wrapper__npu_moe_distribute_combine_add_rms_norm", "activation_scale");
  c10::impl::check_and_update_common_device(common_device, weight_scale, "wrapper__npu_moe_distribute_combine_add_rms_norm", "weight_scale");
  c10::impl::check_and_update_common_device(common_device, group_list, "wrapper__npu_moe_distribute_combine_add_rms_norm", "group_list");
  c10::impl::check_and_update_common_device(common_device, expand_scales, "wrapper__npu_moe_distribute_combine_add_rms_norm", "expand_scales");
  c10::impl::check_and_update_common_device(common_device, shared_expert_x, "wrapper__npu_moe_distribute_combine_add_rms_norm", "shared_expert_x");
  const c10::OptionalDeviceGuard device_guard(device_of(expand_x));
  return op_plugin::npu_moe_distribute_combine_add_rms_norm(expand_x, expert_ids, expand_idx, ep_send_counts, expert_scales, residual_x, gamma, group_ep, ep_world_size, ep_rank_id, moe_expert_num, tp_send_counts, x_active_mask, activation_scale, weight_scale, group_list, expand_scales, shared_expert_x, group_tp, tp_world_size, tp_rank_id, expert_shard_type, shared_expert_num, shared_expert_rank_num, global_bs, out_dtype, comm_quant_mode, group_list_type, comm_alg, norm_eps);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrapper__npu_multi_head_attention(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & query_weight, const at::Tensor & key_weight, const at::Tensor & value_weight, const at::Tensor & attn_mask, const at::Tensor & out_proj_weight, const ::std::optional<at::Tensor> & query_bias, const ::std::optional<at::Tensor> & key_bias, const ::std::optional<at::Tensor> & value_bias, const ::std::optional<at::Tensor> & out_proj_bias, const ::std::optional<at::Tensor> & dropout_mask, int64_t attn_head_num, int64_t attn_dim_per_head, int64_t src_len, int64_t tgt_len, double dropout_prob, bool softmax_use_float) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(query);
      c10_npu::check_npu_tensor_is_safe(key);
      c10_npu::check_npu_tensor_is_safe(value);
      c10_npu::check_npu_tensor_is_safe(query_weight);
      c10_npu::check_npu_tensor_is_safe(key_weight);
      c10_npu::check_npu_tensor_is_safe(value_weight);
      c10_npu::check_npu_tensor_is_safe(attn_mask);
      c10_npu::check_npu_tensor_is_safe(out_proj_weight);
      c10_npu::check_npu_tensor_is_safe(query_bias);
      c10_npu::check_npu_tensor_is_safe(key_bias);
      c10_npu::check_npu_tensor_is_safe(value_bias);
      c10_npu::check_npu_tensor_is_safe(out_proj_bias);
      c10_npu::check_npu_tensor_is_safe(dropout_mask);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, query, "wrapper__npu_multi_head_attention", "query");
  c10::impl::check_and_update_common_device(common_device, key, "wrapper__npu_multi_head_attention", "key");
  c10::impl::check_and_update_common_device(common_device, value, "wrapper__npu_multi_head_attention", "value");
  c10::impl::check_and_update_common_device(common_device, query_weight, "wrapper__npu_multi_head_attention", "query_weight");
  c10::impl::check_and_update_common_device(common_device, key_weight, "wrapper__npu_multi_head_attention", "key_weight");
  c10::impl::check_and_update_common_device(common_device, value_weight, "wrapper__npu_multi_head_attention", "value_weight");
  c10::impl::check_and_update_common_device(common_device, attn_mask, "wrapper__npu_multi_head_attention", "attn_mask");
  c10::impl::check_and_update_common_device(common_device, out_proj_weight, "wrapper__npu_multi_head_attention", "out_proj_weight");
  c10::impl::check_and_update_common_device(common_device, query_bias, "wrapper__npu_multi_head_attention", "query_bias");
  c10::impl::check_and_update_common_device(common_device, key_bias, "wrapper__npu_multi_head_attention", "key_bias");
  c10::impl::check_and_update_common_device(common_device, value_bias, "wrapper__npu_multi_head_attention", "value_bias");
  c10::impl::check_and_update_common_device(common_device, out_proj_bias, "wrapper__npu_multi_head_attention", "out_proj_bias");
  c10::impl::check_and_update_common_device(common_device, dropout_mask, "wrapper__npu_multi_head_attention", "dropout_mask");
  const c10::OptionalDeviceGuard device_guard(device_of(query));
  return op_plugin::npu_multi_head_attention(query, key, value, query_weight, key_weight, value_weight, attn_mask, out_proj_weight, query_bias, key_bias, value_bias, out_proj_bias, dropout_mask, attn_head_num, attn_dim_per_head, src_len, tgt_len, dropout_prob, softmax_use_float);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrapper__npu_multi_head_attention_backward(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & query_weight, const at::Tensor & key_weight, const at::Tensor & value_weight, const at::Tensor & out_proj_weight, const ::std::optional<at::Tensor> & query_bias, const ::std::optional<at::Tensor> & key_bias, const ::std::optional<at::Tensor> & value_bias, const ::std::optional<at::Tensor> & out_proj_bias, const at::Tensor & query_res, const at::Tensor & key_res, const at::Tensor & value_res, const at::Tensor & attn_scores, const at::Tensor & attn_res, const at::Tensor & context, const at::Tensor & y_grad, const at::Tensor & dropout_mask, int64_t attn_head_num, int64_t attn_dim_per_head, int64_t src_len, int64_t tgt_len, double dropout_prob, bool softmax_use_float) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(query);
      c10_npu::check_npu_tensor_is_safe(key);
      c10_npu::check_npu_tensor_is_safe(value);
      c10_npu::check_npu_tensor_is_safe(query_weight);
      c10_npu::check_npu_tensor_is_safe(key_weight);
      c10_npu::check_npu_tensor_is_safe(value_weight);
      c10_npu::check_npu_tensor_is_safe(out_proj_weight);
      c10_npu::check_npu_tensor_is_safe(query_bias);
      c10_npu::check_npu_tensor_is_safe(key_bias);
      c10_npu::check_npu_tensor_is_safe(value_bias);
      c10_npu::check_npu_tensor_is_safe(out_proj_bias);
      c10_npu::check_npu_tensor_is_safe(query_res);
      c10_npu::check_npu_tensor_is_safe(key_res);
      c10_npu::check_npu_tensor_is_safe(value_res);
      c10_npu::check_npu_tensor_is_safe(attn_scores);
      c10_npu::check_npu_tensor_is_safe(attn_res);
      c10_npu::check_npu_tensor_is_safe(context);
      c10_npu::check_npu_tensor_is_safe(y_grad);
      c10_npu::check_npu_tensor_is_safe(dropout_mask);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, query, "wrapper__npu_multi_head_attention_backward", "query");
  c10::impl::check_and_update_common_device(common_device, key, "wrapper__npu_multi_head_attention_backward", "key");
  c10::impl::check_and_update_common_device(common_device, value, "wrapper__npu_multi_head_attention_backward", "value");
  c10::impl::check_and_update_common_device(common_device, query_weight, "wrapper__npu_multi_head_attention_backward", "query_weight");
  c10::impl::check_and_update_common_device(common_device, key_weight, "wrapper__npu_multi_head_attention_backward", "key_weight");
  c10::impl::check_and_update_common_device(common_device, value_weight, "wrapper__npu_multi_head_attention_backward", "value_weight");
  c10::impl::check_and_update_common_device(common_device, out_proj_weight, "wrapper__npu_multi_head_attention_backward", "out_proj_weight");
  c10::impl::check_and_update_common_device(common_device, query_bias, "wrapper__npu_multi_head_attention_backward", "query_bias");
  c10::impl::check_and_update_common_device(common_device, key_bias, "wrapper__npu_multi_head_attention_backward", "key_bias");
  c10::impl::check_and_update_common_device(common_device, value_bias, "wrapper__npu_multi_head_attention_backward", "value_bias");
  c10::impl::check_and_update_common_device(common_device, out_proj_bias, "wrapper__npu_multi_head_attention_backward", "out_proj_bias");
  c10::impl::check_and_update_common_device(common_device, query_res, "wrapper__npu_multi_head_attention_backward", "query_res");
  c10::impl::check_and_update_common_device(common_device, key_res, "wrapper__npu_multi_head_attention_backward", "key_res");
  c10::impl::check_and_update_common_device(common_device, value_res, "wrapper__npu_multi_head_attention_backward", "value_res");
  c10::impl::check_and_update_common_device(common_device, attn_scores, "wrapper__npu_multi_head_attention_backward", "attn_scores");
  c10::impl::check_and_update_common_device(common_device, attn_res, "wrapper__npu_multi_head_attention_backward", "attn_res");
  c10::impl::check_and_update_common_device(common_device, context, "wrapper__npu_multi_head_attention_backward", "context");
  c10::impl::check_and_update_common_device(common_device, y_grad, "wrapper__npu_multi_head_attention_backward", "y_grad");
  c10::impl::check_and_update_common_device(common_device, dropout_mask, "wrapper__npu_multi_head_attention_backward", "dropout_mask");
  const c10::OptionalDeviceGuard device_guard(device_of(query));
  return op_plugin::npu_multi_head_attention_backward(query, key, value, query_weight, key_weight, value_weight, out_proj_weight, query_bias, key_bias, value_bias, out_proj_bias, query_res, key_res, value_res, attn_scores, attn_res, context, y_grad, dropout_mask, attn_head_num, attn_dim_per_head, src_len, tgt_len, dropout_prob, softmax_use_float);
}

::std::tuple<at::Tensor,at::Tensor,int64_t,int64_t,int64_t> wrapper__npu_multi_head_attention_v2(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & atten_mask, const ::std::optional<at::Tensor> & alibi_mask, double scale, int64_t head_num, c10::string_view input_layout, double keep_prob, int64_t pre_tokens, int64_t next_tokens, bool gen_mask_parallel, bool sync) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(query);
      c10_npu::check_npu_tensor_is_safe(key);
      c10_npu::check_npu_tensor_is_safe(value);
      c10_npu::check_npu_tensor_is_safe(atten_mask);
      c10_npu::check_npu_tensor_is_safe(alibi_mask);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, query, "wrapper__npu_multi_head_attention_v2", "query");
  c10::impl::check_and_update_common_device(common_device, key, "wrapper__npu_multi_head_attention_v2", "key");
  c10::impl::check_and_update_common_device(common_device, value, "wrapper__npu_multi_head_attention_v2", "value");
  c10::impl::check_and_update_common_device(common_device, atten_mask, "wrapper__npu_multi_head_attention_v2", "atten_mask");
  c10::impl::check_and_update_common_device(common_device, alibi_mask, "wrapper__npu_multi_head_attention_v2", "alibi_mask");
  const c10::OptionalDeviceGuard device_guard(device_of(query));
  return op_plugin::npu_multi_head_attention_v2(query, key, value, atten_mask, alibi_mask, scale, head_num, input_layout, keep_prob, pre_tokens, next_tokens, gen_mask_parallel, sync);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapper__npu_multi_head_attention_v2_grad(const at::Tensor & attention_score_grad, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & softmax_log_max_sum, const at::Tensor & attention_score, const ::std::optional<at::Tensor> & atten_mask, const ::std::optional<at::Tensor> & alibi_mask, double scale, int64_t head_num, c10::string_view input_layout, double keep_prob, int64_t pre_tokens, int64_t next_tokens, int64_t seed, int64_t offset, int64_t numels, bool gen_mask_parallel, bool sync) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(attention_score_grad);
      c10_npu::check_npu_tensor_is_safe(query);
      c10_npu::check_npu_tensor_is_safe(key);
      c10_npu::check_npu_tensor_is_safe(value);
      c10_npu::check_npu_tensor_is_safe(softmax_log_max_sum);
      c10_npu::check_npu_tensor_is_safe(attention_score);
      c10_npu::check_npu_tensor_is_safe(atten_mask);
      c10_npu::check_npu_tensor_is_safe(alibi_mask);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, attention_score_grad, "wrapper__npu_multi_head_attention_v2_grad", "attention_score_grad");
  c10::impl::check_and_update_common_device(common_device, query, "wrapper__npu_multi_head_attention_v2_grad", "query");
  c10::impl::check_and_update_common_device(common_device, key, "wrapper__npu_multi_head_attention_v2_grad", "key");
  c10::impl::check_and_update_common_device(common_device, value, "wrapper__npu_multi_head_attention_v2_grad", "value");
  c10::impl::check_and_update_common_device(common_device, softmax_log_max_sum, "wrapper__npu_multi_head_attention_v2_grad", "softmax_log_max_sum");
  c10::impl::check_and_update_common_device(common_device, attention_score, "wrapper__npu_multi_head_attention_v2_grad", "attention_score");
  c10::impl::check_and_update_common_device(common_device, atten_mask, "wrapper__npu_multi_head_attention_v2_grad", "atten_mask");
  c10::impl::check_and_update_common_device(common_device, alibi_mask, "wrapper__npu_multi_head_attention_v2_grad", "alibi_mask");
  const c10::OptionalDeviceGuard device_guard(device_of(attention_score_grad));
  return op_plugin::npu_multi_head_attention_v2_grad(attention_score_grad, query, key, value, softmax_log_max_sum, attention_score, atten_mask, alibi_mask, scale, head_num, input_layout, keep_prob, pre_tokens, next_tokens, seed, offset, numels, gen_mask_parallel, sync);
}

::std::tuple<at::Tensor,at::Tensor> wrapper__npu_nms_rotated(const at::Tensor & self, const at::Tensor & scores, double iou_threshold, double scores_threshold, int64_t max_output_size, int64_t mode) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(scores);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_nms_rotated", "self");
  c10::impl::check_and_update_common_device(common_device, scores, "wrapper__npu_nms_rotated", "scores");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_nms_rotated(self, scores, iou_threshold, scores_threshold, max_output_size, mode);
}

::std::tuple<at::Tensor,at::Tensor> wrapper__npu_nms_v4(const at::Tensor & self, const at::Tensor & scores, const at::Scalar & max_output_size, const at::Tensor & iou_threshold, const at::Tensor & scores_threshold, bool pad_to_max_output_size) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(scores);
      c10_npu::check_npu_tensor_is_safe(iou_threshold);
      c10_npu::check_npu_tensor_is_safe(scores_threshold);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_nms_v4", "self");
  c10::impl::check_and_update_common_device(common_device, scores, "wrapper__npu_nms_v4", "scores");
  c10::impl::check_and_update_common_device(common_device, iou_threshold, "wrapper__npu_nms_v4", "iou_threshold");
  c10::impl::check_and_update_common_device(common_device, scores_threshold, "wrapper__npu_nms_v4", "scores_threshold");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_nms_v4(self, scores, max_output_size, iou_threshold, scores_threshold, pad_to_max_output_size);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapper__npu_nms_with_mask(const at::Tensor & input, const at::Scalar & iou_threshold) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(input);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_nms_with_mask", "input");
  const c10::OptionalDeviceGuard device_guard(device_of(input));
  return op_plugin::npu_nms_with_mask(input, iou_threshold);
}

at::Tensor wrapper__npu_normalize_batch(const at::Tensor & self, const at::Tensor & seq_len, int64_t normalize_type) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(seq_len);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_normalize_batch", "self");
  c10::impl::check_and_update_common_device(common_device, seq_len, "wrapper__npu_normalize_batch", "seq_len");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_normalize_batch(self, seq_len, normalize_type);
}

at::Tensor wrapper__npu_one_hot(const at::Tensor & self, int64_t num_classes, int64_t depth, const at::Scalar & on_value, const at::Scalar & off_value) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_one_hot", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_one_hot(self, num_classes, depth, on_value, off_value);
}

at::Tensor wrapper__npu_pad(const at::Tensor & input, at::IntArrayRef paddings) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(input);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_pad", "input");
  const c10::OptionalDeviceGuard device_guard(device_of(input));
  return op_plugin::npu_pad(input, paddings);
}

at::Tensor wrapper__npu_prompt_flash_attention(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & padding_mask, const ::std::optional<at::Tensor> & atten_mask, const ::std::optional<at::Tensor> & pse_shift, at::OptionalIntArrayRef actual_seq_lengths, const ::std::optional<at::Tensor> & deq_scale1, const ::std::optional<at::Tensor> & quant_scale1, const ::std::optional<at::Tensor> & deq_scale2, const ::std::optional<at::Tensor> & quant_scale2, const ::std::optional<at::Tensor> & quant_offset2, int64_t num_heads, double scale_value, int64_t pre_tokens, int64_t next_tokens, c10::string_view input_layout, int64_t num_key_value_heads, at::OptionalIntArrayRef actual_seq_lengths_kv, int64_t sparse_mode) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(query);
      c10_npu::check_npu_tensor_is_safe(key);
      c10_npu::check_npu_tensor_is_safe(value);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, query, "wrapper__npu_prompt_flash_attention", "query");
  c10::impl::check_and_update_common_device(common_device, key, "wrapper__npu_prompt_flash_attention", "key");
  c10::impl::check_and_update_common_device(common_device, value, "wrapper__npu_prompt_flash_attention", "value");
  c10::impl::check_and_update_common_device(common_device, padding_mask, "wrapper__npu_prompt_flash_attention", "padding_mask");
  c10::impl::check_and_update_common_device(common_device, atten_mask, "wrapper__npu_prompt_flash_attention", "atten_mask");
  c10::impl::check_and_update_common_device(common_device, pse_shift, "wrapper__npu_prompt_flash_attention", "pse_shift");
  c10::impl::check_and_update_common_device(common_device, deq_scale1, "wrapper__npu_prompt_flash_attention", "deq_scale1");
  c10::impl::check_and_update_common_device(common_device, quant_scale1, "wrapper__npu_prompt_flash_attention", "quant_scale1");
  c10::impl::check_and_update_common_device(common_device, deq_scale2, "wrapper__npu_prompt_flash_attention", "deq_scale2");
  c10::impl::check_and_update_common_device(common_device, quant_scale2, "wrapper__npu_prompt_flash_attention", "quant_scale2");
  c10::impl::check_and_update_common_device(common_device, quant_offset2, "wrapper__npu_prompt_flash_attention", "quant_offset2");
  const c10::OptionalDeviceGuard device_guard(device_of(query));
  return op_plugin::npu_prompt_flash_attention(query, key, value, padding_mask, atten_mask, pse_shift, actual_seq_lengths, deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2, num_heads, scale_value, pre_tokens, next_tokens, input_layout, num_key_value_heads, actual_seq_lengths_kv, sparse_mode);
}

at::Tensor wrapper__npu_ps_roi_pooling(const at::Tensor & self, const at::Tensor & rois, double spatial_scale, int64_t group_size, int64_t output_dim) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(rois);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_ps_roi_pooling", "self");
  c10::impl::check_and_update_common_device(common_device, rois, "wrapper__npu_ps_roi_pooling", "rois");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_ps_roi_pooling(self, rois, spatial_scale, group_size, output_dim);
}

at::Tensor wrapper__npu_ps_roi_pooling_backward(const at::Tensor & output_grad, const at::Tensor & rois, double spatial_scale, int64_t group_size, int64_t output_dim, c10::SymIntArrayRef input_size) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(output_grad);
      c10_npu::check_npu_tensor_is_safe(rois);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, output_grad, "wrapper__npu_ps_roi_pooling_backward", "output_grad");
  c10::impl::check_and_update_common_device(common_device, rois, "wrapper__npu_ps_roi_pooling_backward", "rois");
  const c10::OptionalDeviceGuard device_guard(device_of(output_grad));
  return op_plugin::npu_ps_roi_pooling_backward_symint(output_grad, rois, spatial_scale, group_size, output_dim, input_size);
}

at::Tensor wrapper__npu_ptiou(const at::Tensor & bboxes, const at::Tensor & gtboxes, int64_t mode) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(bboxes);
      c10_npu::check_npu_tensor_is_safe(gtboxes);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, bboxes, "wrapper__npu_ptiou", "bboxes");
  c10::impl::check_and_update_common_device(common_device, gtboxes, "wrapper__npu_ptiou", "gtboxes");
  const c10::OptionalDeviceGuard device_guard(device_of(bboxes));
  return op_plugin::npu_ptiou(bboxes, gtboxes, mode);
}

void wrapper__npu_prefetch(const at::Tensor & self, const ::std::optional<at::Tensor> & dependency, int64_t max_size, int64_t offset) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(dependency);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_prefetch", "self");
  c10::impl::check_and_update_common_device(common_device, dependency, "wrapper__npu_prefetch", "dependency");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_prefetch(self, dependency, max_size, offset);
}

at::Tensor wrapper__npu_quant_conv2d(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & scale, at::IntArrayRef strides, at::IntArrayRef pads, at::IntArrayRef dilations, int64_t groups, int64_t offset_x, c10::string_view round_mode, ::std::optional<at::ScalarType> output_dtype, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & offset) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(weight);
      c10_npu::check_npu_tensor_is_safe(scale);
      c10_npu::check_npu_tensor_is_safe(bias);
      c10_npu::check_npu_tensor_is_safe(offset);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_quant_conv2d", "input");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__npu_quant_conv2d", "weight");
  c10::impl::check_and_update_common_device(common_device, scale, "wrapper__npu_quant_conv2d", "scale");
  c10::impl::check_and_update_common_device(common_device, bias, "wrapper__npu_quant_conv2d", "bias");
  c10::impl::check_and_update_common_device(common_device, offset, "wrapper__npu_quant_conv2d", "offset");
  const c10::OptionalDeviceGuard device_guard(device_of(input));
  return op_plugin::npu_quant_conv2d(input, weight, scale, strides, pads, dilations, groups, offset_x, round_mode, output_dtype, bias, offset);
}

at::Tensor wrapper__npu_quant_matmul(const at::Tensor & x1, const at::Tensor & x2, const at::Tensor & scale, const ::std::optional<at::Tensor> & offset, const ::std::optional<at::Tensor> & pertoken_scale, const ::std::optional<at::Tensor> & bias, ::std::optional<at::ScalarType> output_dtype, at::OptionalSymIntArrayRef group_sizes) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(x1);
      c10_npu::check_npu_tensor_is_safe(x2);
      c10_npu::check_npu_tensor_is_safe(scale);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, x1, "wrapper__npu_quant_matmul", "x1");
  c10::impl::check_and_update_common_device(common_device, x2, "wrapper__npu_quant_matmul", "x2");
  c10::impl::check_and_update_common_device(common_device, scale, "wrapper__npu_quant_matmul", "scale");
  c10::impl::check_and_update_common_device(common_device, offset, "wrapper__npu_quant_matmul", "offset");
  c10::impl::check_and_update_common_device(common_device, pertoken_scale, "wrapper__npu_quant_matmul", "pertoken_scale");
  c10::impl::check_and_update_common_device(common_device, bias, "wrapper__npu_quant_matmul", "bias");
  const c10::OptionalDeviceGuard device_guard(device_of(x1));
  return op_plugin::npu_quant_matmul_symint(x1, x2, scale, offset, pertoken_scale, bias, output_dtype, group_sizes);
}

at::Tensor wrapper__npu_quant_matmul_dequant(const at::Tensor & x, const at::Tensor & quantized_weight, const at::Tensor & weight_scale, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & x_scale, const ::std::optional<at::Tensor> & x_offset, const ::std::optional<at::Tensor> & smooth_scale, ::std::optional<c10::string_view> quant_mode) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(x);
      c10_npu::check_npu_tensor_is_safe(quantized_weight);
      c10_npu::check_npu_tensor_is_safe(weight_scale);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, x, "wrapper__npu_quant_matmul_dequant", "x");
  c10::impl::check_and_update_common_device(common_device, quantized_weight, "wrapper__npu_quant_matmul_dequant", "quantized_weight");
  c10::impl::check_and_update_common_device(common_device, weight_scale, "wrapper__npu_quant_matmul_dequant", "weight_scale");
  c10::impl::check_and_update_common_device(common_device, bias, "wrapper__npu_quant_matmul_dequant", "bias");
  c10::impl::check_and_update_common_device(common_device, x_scale, "wrapper__npu_quant_matmul_dequant", "x_scale");
  c10::impl::check_and_update_common_device(common_device, x_offset, "wrapper__npu_quant_matmul_dequant", "x_offset");
  c10::impl::check_and_update_common_device(common_device, smooth_scale, "wrapper__npu_quant_matmul_dequant", "smooth_scale");
  const c10::OptionalDeviceGuard device_guard(device_of(x));
  return op_plugin::npu_quant_matmul_dequant(x, quantized_weight, weight_scale, bias, x_scale, x_offset, smooth_scale, quant_mode);
}

at::Tensor wrapper__npu_quant_grouped_matmul_dequant(const at::Tensor & x, const at::Tensor & quantized_weight, const at::Tensor & weight_scale, const at::Tensor & group_list, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & x_scale, const ::std::optional<at::Tensor> & x_offset, const ::std::optional<at::Tensor> & smooth_scale, ::std::optional<c10::string_view> quant_mode) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(x);
      c10_npu::check_npu_tensor_is_safe(quantized_weight);
      c10_npu::check_npu_tensor_is_safe(weight_scale);
      c10_npu::check_npu_tensor_is_safe(group_list);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, x, "wrapper__npu_quant_grouped_matmul_dequant", "x");
  c10::impl::check_and_update_common_device(common_device, quantized_weight, "wrapper__npu_quant_grouped_matmul_dequant", "quantized_weight");
  c10::impl::check_and_update_common_device(common_device, weight_scale, "wrapper__npu_quant_grouped_matmul_dequant", "weight_scale");
  c10::impl::check_and_update_common_device(common_device, group_list, "wrapper__npu_quant_grouped_matmul_dequant", "group_list");
  c10::impl::check_and_update_common_device(common_device, bias, "wrapper__npu_quant_grouped_matmul_dequant", "bias");
  c10::impl::check_and_update_common_device(common_device, x_scale, "wrapper__npu_quant_grouped_matmul_dequant", "x_scale");
  c10::impl::check_and_update_common_device(common_device, x_offset, "wrapper__npu_quant_grouped_matmul_dequant", "x_offset");
  c10::impl::check_and_update_common_device(common_device, smooth_scale, "wrapper__npu_quant_grouped_matmul_dequant", "smooth_scale");
  const c10::OptionalDeviceGuard device_guard(device_of(x));
  return op_plugin::npu_quant_grouped_matmul_dequant(x, quantized_weight, weight_scale, group_list, bias, x_scale, x_offset, smooth_scale, quant_mode);
}

at::Tensor wrapper__npu_quant_scatter(const at::Tensor & self, const at::Tensor & indices, const at::Tensor & updates, const at::Tensor & quant_scales, const ::std::optional<at::Tensor> & quant_zero_points, int64_t axis, int64_t quant_axis, c10::string_view reduce) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(indices);
      c10_npu::check_npu_tensor_is_safe(updates);
      c10_npu::check_npu_tensor_is_safe(quant_scales);
      c10_npu::check_npu_tensor_is_safe(quant_zero_points);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_quant_scatter", "self");
  c10::impl::check_and_update_common_device(common_device, indices, "wrapper__npu_quant_scatter", "indices");
  c10::impl::check_and_update_common_device(common_device, updates, "wrapper__npu_quant_scatter", "updates");
  c10::impl::check_and_update_common_device(common_device, quant_scales, "wrapper__npu_quant_scatter", "quant_scales");
  c10::impl::check_and_update_common_device(common_device, quant_zero_points, "wrapper__npu_quant_scatter", "quant_zero_points");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_quant_scatter(self, indices, updates, quant_scales, quant_zero_points, axis, quant_axis, reduce);
}

at::Tensor & wrapper__npu_quant_scatter_(at::Tensor & self, const at::Tensor & indices, const at::Tensor & updates, const at::Tensor & quant_scales, const ::std::optional<at::Tensor> & quant_zero_points, int64_t axis, int64_t quant_axis, c10::string_view reduce) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(indices);
      c10_npu::check_npu_tensor_is_safe(updates);
      c10_npu::check_npu_tensor_is_safe(quant_scales);
      c10_npu::check_npu_tensor_is_safe(quant_zero_points);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_quant_scatter_", "self");
  c10::impl::check_and_update_common_device(common_device, indices, "wrapper__npu_quant_scatter_", "indices");
  c10::impl::check_and_update_common_device(common_device, updates, "wrapper__npu_quant_scatter_", "updates");
  c10::impl::check_and_update_common_device(common_device, quant_scales, "wrapper__npu_quant_scatter_", "quant_scales");
  c10::impl::check_and_update_common_device(common_device, quant_zero_points, "wrapper__npu_quant_scatter_", "quant_zero_points");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_quant_scatter_(self, indices, updates, quant_scales, quant_zero_points, axis, quant_axis, reduce);
}

at::Tensor wrapper__npu_quantize(const at::Tensor & self, const at::Tensor & scales, const ::std::optional<at::Tensor> & zero_points, at::ScalarType dtype, int64_t axis, bool div_mode) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(scales);
      c10_npu::check_npu_tensor_is_safe(zero_points);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_quantize", "self");
  c10::impl::check_and_update_common_device(common_device, scales, "wrapper__npu_quantize", "scales");
  c10::impl::check_and_update_common_device(common_device, zero_points, "wrapper__npu_quantize", "zero_points");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_quantize(self, scales, zero_points, dtype, axis, div_mode);
}

::std::tuple<at::Tensor,at::Tensor> wrapper__npu_kronecker_quant(const at::Tensor & x, const at::Tensor & kronecker_p1, const at::Tensor & kronecker_p2, ::std::optional<double> clip_ratio, ::std::optional<at::ScalarType> dst_dtype) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(x);
      c10_npu::check_npu_tensor_is_safe(kronecker_p1);
      c10_npu::check_npu_tensor_is_safe(kronecker_p2);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, x, "wrapper__npu_kronecker_quant", "x");
  c10::impl::check_and_update_common_device(common_device, kronecker_p1, "wrapper__npu_kronecker_quant", "kronecker_p1");
  c10::impl::check_and_update_common_device(common_device, kronecker_p2, "wrapper__npu_kronecker_quant", "kronecker_p2");
  const c10::OptionalDeviceGuard device_guard(device_of(x));
  return op_plugin::npu_kronecker_quant(x, kronecker_p1, kronecker_p2, clip_ratio, dst_dtype);
}

at::Tensor wrapper__npu_group_quant(const at::Tensor & x, const at::Tensor & scale, const at::Tensor & group_index, const ::std::optional<at::Tensor> & offset, ::std::optional<at::ScalarType> dst_dtype) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(x);
      c10_npu::check_npu_tensor_is_safe(scale);
      c10_npu::check_npu_tensor_is_safe(group_index);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, x, "wrapper__npu_group_quant", "x");
  c10::impl::check_and_update_common_device(common_device, scale, "wrapper__npu_group_quant", "scale");
  c10::impl::check_and_update_common_device(common_device, group_index, "wrapper__npu_group_quant", "group_index");
  c10::impl::check_and_update_common_device(common_device, offset, "wrapper__npu_group_quant", "offset");
  const c10::OptionalDeviceGuard device_guard(device_of(x));
  return op_plugin::npu_group_quant(x, scale, group_index, offset, dst_dtype);
}

::std::tuple<at::Tensor,at::Tensor> wrapper__npu_random_choice_with_mask(const at::Tensor & x, int64_t count, int64_t seed, int64_t seed2) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(x);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, x, "wrapper__npu_random_choice_with_mask", "x");
  const c10::OptionalDeviceGuard device_guard(device_of(x));
  return op_plugin::npu_random_choice_with_mask(x, count, seed, seed2);
}

at::Tensor wrapper__npu_reshape(const at::Tensor & self, at::IntArrayRef shape, bool can_refresh) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
    // No device check
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_reshape(self, shape, can_refresh);
}

at::Tensor & wrapper_out_npu_reshape_out(const at::Tensor & self, at::IntArrayRef shape, bool can_refresh, at::Tensor & out) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(out);
      c10_npu::check_npu_tensor_is_safe(self);
  }
    // No device check
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_reshape_out(self, shape, can_refresh, out);
}

::std::tuple<at::Tensor,at::Tensor> wrapper__npu_rms_norm(const at::Tensor & self, const at::Tensor & gamma, double epsilon) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(gamma);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_rms_norm", "self");
  c10::impl::check_and_update_common_device(common_device, gamma, "wrapper__npu_rms_norm", "gamma");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_rms_norm(self, gamma, epsilon);
}

::std::tuple<at::Tensor,at::Tensor> wrapper__npu_gemma_rms_norm(const at::Tensor & self, const at::Tensor & gamma, double epsilon) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(gamma);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_gemma_rms_norm", "self");
  c10::impl::check_and_update_common_device(common_device, gamma, "wrapper__npu_gemma_rms_norm", "gamma");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_gemma_rms_norm(self, gamma, epsilon);
}

::std::tuple<at::Tensor,at::Tensor> wrapper__npu_rms_norm_backward(const at::Tensor & dy, const at::Tensor & self, const at::Tensor & gamma, const at::Tensor & rstd) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(dy);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(gamma);
      c10_npu::check_npu_tensor_is_safe(rstd);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, dy, "wrapper__npu_rms_norm_backward", "dy");
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_rms_norm_backward", "self");
  c10::impl::check_and_update_common_device(common_device, gamma, "wrapper__npu_rms_norm_backward", "gamma");
  c10::impl::check_and_update_common_device(common_device, rstd, "wrapper__npu_rms_norm_backward", "rstd");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_rms_norm_backward(dy, self, gamma, rstd);
}

at::Tensor wrapper__npu_roi_align(const at::Tensor & self, const at::Tensor & rois, double spatial_scale, int64_t pooled_height, int64_t pooled_width, int64_t sample_num, int64_t roi_end_mode) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(rois);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_roi_align", "self");
  c10::impl::check_and_update_common_device(common_device, rois, "wrapper__npu_roi_align", "rois");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_roi_align(self, rois, spatial_scale, pooled_height, pooled_width, sample_num, roi_end_mode);
}

at::Tensor wrapper__npu_roi_alignbk(const at::Tensor & self, const at::Tensor & rois, at::IntArrayRef xdiff_shape, int64_t pooled_width, int64_t pooled_height, double spatial_scale, int64_t sample_num, ::std::optional<int64_t> roi_end_mode) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(rois);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_roi_alignbk", "self");
  c10::impl::check_and_update_common_device(common_device, rois, "wrapper__npu_roi_alignbk", "rois");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_roi_alignbk(self, rois, xdiff_shape, pooled_width, pooled_height, spatial_scale, sample_num, roi_end_mode);
}

at::Tensor wrapper__npu_rotary_mul(const at::Tensor & self, const at::Tensor & r1, const at::Tensor & r2, c10::string_view rotary_mode) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(r1);
      c10_npu::check_npu_tensor_is_safe(r2);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_rotary_mul", "self");
  c10::impl::check_and_update_common_device(common_device, r1, "wrapper__npu_rotary_mul", "r1");
  c10::impl::check_and_update_common_device(common_device, r2, "wrapper__npu_rotary_mul", "r2");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_rotary_mul(self, r1, r2, rotary_mode);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapper__npu_rotary_mul_backward(const at::Tensor & grad, const at::Tensor & self, const at::Tensor & r1, const at::Tensor & r2, c10::string_view rotary_mode) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(grad);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(r1);
      c10_npu::check_npu_tensor_is_safe(r2);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad, "wrapper__npu_rotary_mul_backward", "grad");
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_rotary_mul_backward", "self");
  c10::impl::check_and_update_common_device(common_device, r1, "wrapper__npu_rotary_mul_backward", "r1");
  c10::impl::check_and_update_common_device(common_device, r2, "wrapper__npu_rotary_mul_backward", "r2");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_rotary_mul_backward(grad, self, r1, r2, rotary_mode);
}

::std::tuple<at::Tensor,at::Tensor> wrapper__npu_mrope(const at::Tensor & positions, const at::Tensor & query, const at::Tensor & key, const at::Tensor & cos_sin_cache, int64_t head_size, at::OptionalIntArrayRef mrope_section, ::std::optional<c10::string_view> rotary_mode) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(positions);
      c10_npu::check_npu_tensor_is_safe(query);
      c10_npu::check_npu_tensor_is_safe(key);
      c10_npu::check_npu_tensor_is_safe(cos_sin_cache);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, positions, "wrapper__npu_mrope", "positions");
  c10::impl::check_and_update_common_device(common_device, query, "wrapper__npu_mrope", "query");
  c10::impl::check_and_update_common_device(common_device, key, "wrapper__npu_mrope", "key");
  c10::impl::check_and_update_common_device(common_device, cos_sin_cache, "wrapper__npu_mrope", "cos_sin_cache");
  const c10::OptionalDeviceGuard device_guard(device_of(positions));
  return op_plugin::npu_mrope(positions, query, key, cos_sin_cache, head_size, mrope_section, rotary_mode);
}

at::Tensor wrapper__npu_rotated_box_decode(const at::Tensor & self, const at::Tensor & deltas, const at::Tensor & weight) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(deltas);
      c10_npu::check_npu_tensor_is_safe(weight);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_rotated_box_decode", "self");
  c10::impl::check_and_update_common_device(common_device, deltas, "wrapper__npu_rotated_box_decode", "deltas");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__npu_rotated_box_decode", "weight");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_rotated_box_decode(self, deltas, weight);
}

at::Tensor wrapper__npu_rotated_box_encode(const at::Tensor & self, const at::Tensor & gt_bboxes, const at::Tensor & weight) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(gt_bboxes);
      c10_npu::check_npu_tensor_is_safe(weight);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_rotated_box_encode", "self");
  c10::impl::check_and_update_common_device(common_device, gt_bboxes, "wrapper__npu_rotated_box_encode", "gt_bboxes");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__npu_rotated_box_encode", "weight");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_rotated_box_encode(self, gt_bboxes, weight);
}

at::Tensor wrapper__npu_rotated_iou(const at::Tensor & self, const at::Tensor & query_boxes, bool trans, int64_t mode, bool is_cross, double v_threshold, double e_threshold) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(query_boxes);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_rotated_iou", "self");
  c10::impl::check_and_update_common_device(common_device, query_boxes, "wrapper__npu_rotated_iou", "query_boxes");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_rotated_iou(self, query_boxes, trans, mode, is_cross, v_threshold, e_threshold);
}

at::Tensor wrapper__npu_rotated_overlaps(const at::Tensor & self, const at::Tensor & query_boxes, bool trans) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(query_boxes);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_rotated_overlaps", "self");
  c10::impl::check_and_update_common_device(common_device, query_boxes, "wrapper__npu_rotated_overlaps", "query_boxes");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_rotated_overlaps(self, query_boxes, trans);
}

at::Tensor wrapper__npu_scaled_masked_softmax(const at::Tensor & x, const at::Tensor & mask, const at::Scalar & scale, bool fixed_triu_mask) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(x);
      c10_npu::check_npu_tensor_is_safe(mask);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, x, "wrapper__npu_scaled_masked_softmax", "x");
  c10::impl::check_and_update_common_device(common_device, mask, "wrapper__npu_scaled_masked_softmax", "mask");
  const c10::OptionalDeviceGuard device_guard(device_of(x));
  return op_plugin::npu_scaled_masked_softmax(x, mask, scale, fixed_triu_mask);
}

at::Tensor wrapper__npu_scaled_masked_softmax_backward(const at::Tensor & y_grad, const at::Tensor & y, const at::Tensor & mask, const at::Scalar & scale, bool fixed_triu_mask) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(y_grad);
      c10_npu::check_npu_tensor_is_safe(y);
      c10_npu::check_npu_tensor_is_safe(mask);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, y_grad, "wrapper__npu_scaled_masked_softmax_backward", "y_grad");
  c10::impl::check_and_update_common_device(common_device, y, "wrapper__npu_scaled_masked_softmax_backward", "y");
  c10::impl::check_and_update_common_device(common_device, mask, "wrapper__npu_scaled_masked_softmax_backward", "mask");
  const c10::OptionalDeviceGuard device_guard(device_of(y_grad));
  return op_plugin::npu_scaled_masked_softmax_backward(y_grad, y, mask, scale, fixed_triu_mask);
}

at::Tensor wrapper__npu_scatter(const at::Tensor & self, const at::Tensor & indices, const at::Tensor & updates, int64_t dim) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(indices);
      c10_npu::check_npu_tensor_is_safe(updates);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_scatter", "self");
  c10::impl::check_and_update_common_device(common_device, indices, "wrapper__npu_scatter", "indices");
  c10::impl::check_and_update_common_device(common_device, updates, "wrapper__npu_scatter", "updates");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_scatter(self, indices, updates, dim);
}

::std::vector<at::Tensor> wrapper__npu_scatter_list(at::TensorList self, const at::Tensor & indices, const at::Tensor & updates, const ::std::optional<at::Tensor> & mask, c10::string_view reduce, int64_t axis) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(indices);
      c10_npu::check_npu_tensor_is_safe(updates);
      c10_npu::check_npu_tensor_is_safe(mask);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_scatter_list", "self");
  c10::impl::check_and_update_common_device(common_device, indices, "wrapper__npu_scatter_list", "indices");
  c10::impl::check_and_update_common_device(common_device, updates, "wrapper__npu_scatter_list", "updates");
  c10::impl::check_and_update_common_device(common_device, mask, "wrapper__npu_scatter_list", "mask");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_scatter_list(self, indices, updates, mask, reduce, axis);
}

void wrapper__npu_scatter_list_(at::TensorList self, const at::Tensor & indices, const at::Tensor & updates, const ::std::optional<at::Tensor> & mask, c10::string_view reduce, int64_t axis) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(indices);
      c10_npu::check_npu_tensor_is_safe(updates);
      c10_npu::check_npu_tensor_is_safe(mask);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_scatter_list_", "self");
  c10::impl::check_and_update_common_device(common_device, indices, "wrapper__npu_scatter_list_", "indices");
  c10::impl::check_and_update_common_device(common_device, updates, "wrapper__npu_scatter_list_", "updates");
  c10::impl::check_and_update_common_device(common_device, mask, "wrapper__npu_scatter_list_", "mask");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_scatter_list_(self, indices, updates, mask, reduce, axis);
}

at::Tensor wrapper__npu_scatter_nd_update(const at::Tensor & self, const at::Tensor & indices, const at::Tensor & updates) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(indices);
      c10_npu::check_npu_tensor_is_safe(updates);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_scatter_nd_update", "self");
  c10::impl::check_and_update_common_device(common_device, indices, "wrapper__npu_scatter_nd_update", "indices");
  c10::impl::check_and_update_common_device(common_device, updates, "wrapper__npu_scatter_nd_update", "updates");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_scatter_nd_update(self, indices, updates);
}

at::Tensor & wrapper__npu_scatter_nd_update_(at::Tensor & self, const at::Tensor & indices, const at::Tensor & updates) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(indices);
      c10_npu::check_npu_tensor_is_safe(updates);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_scatter_nd_update_", "self");
  c10::impl::check_and_update_common_device(common_device, indices, "wrapper__npu_scatter_nd_update_", "indices");
  c10::impl::check_and_update_common_device(common_device, updates, "wrapper__npu_scatter_nd_update_", "updates");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_scatter_nd_update_(self, indices, updates);
}

at::Tensor wrapper__npu_sign_bits_pack(const at::Tensor & self, int64_t size) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_sign_bits_pack", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_sign_bits_pack(self, size);
}

at::Tensor wrapper__npu_sign_bits_unpack(const at::Tensor & input, int64_t size, at::ScalarType dtype) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(input);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_sign_bits_unpack", "input");
  const c10::OptionalDeviceGuard device_guard(device_of(input));
  return op_plugin::npu_sign_bits_unpack(input, size, dtype);
}

at::Tensor wrapper__npu_silu(const at::Tensor & self) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_silu", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_silu(self);
}

at::Tensor & wrapper__npu_silu_(at::Tensor & self) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_silu_", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_silu_(self);
}

at::Tensor wrapper__npu_silu_backward(const at::Tensor & grad_output, const at::Tensor & x0, const at::Tensor & x1) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(grad_output);
      c10_npu::check_npu_tensor_is_safe(x0);
      c10_npu::check_npu_tensor_is_safe(x1);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad_output, "wrapper__npu_silu_backward", "grad_output");
  c10::impl::check_and_update_common_device(common_device, x0, "wrapper__npu_silu_backward", "x0");
  c10::impl::check_and_update_common_device(common_device, x1, "wrapper__npu_silu_backward", "x1");
  const c10::OptionalDeviceGuard device_guard(device_of(grad_output));
  return op_plugin::npu_silu_backward(grad_output, x0, x1);
}

at::Tensor wrapper__npu_slice(const at::Tensor & self, at::IntArrayRef offsets, at::IntArrayRef size) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_slice", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_slice(self, offsets, size);
}

at::Tensor & wrapper_out_npu_slice_out(const at::Tensor & self, at::IntArrayRef offsets, at::IntArrayRef size, at::Tensor & out) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(out);
      c10_npu::check_npu_tensor_is_safe(self);
  }
    // No device check
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_slice_out(self, offsets, size, out);
}

at::Tensor wrapper__npu_softmax_cross_entropy_with_logits(const at::Tensor & self, const at::Tensor & labels) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(labels);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_softmax_cross_entropy_with_logits", "self");
  c10::impl::check_and_update_common_device(common_device, labels, "wrapper__npu_softmax_cross_entropy_with_logits", "labels");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_softmax_cross_entropy_with_logits(self, labels);
}

at::Tensor wrapper__npu_softmax_cross_entropy_with_logits_backward(const at::Tensor & grad, const at::Tensor & self, const at::Tensor & labels) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(grad);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(labels);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad, "wrapper__npu_softmax_cross_entropy_with_logits_backward", "grad");
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_softmax_cross_entropy_with_logits_backward", "self");
  c10::impl::check_and_update_common_device(common_device, labels, "wrapper__npu_softmax_cross_entropy_with_logits_backward", "labels");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_softmax_cross_entropy_with_logits_backward(grad, self, labels);
}

at::Tensor wrapper__npu_sort_v2(const at::Tensor & self, int64_t dim, bool descending) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_sort_v2", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_sort_v2(self, dim, descending);
}

at::Tensor & wrapper_out_npu_sort_v2_out(const at::Tensor & self, int64_t dim, bool descending, at::Tensor & out) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(out);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, out, "wrapper_out_npu_sort_v2_out", "out");
  c10::impl::check_and_update_common_device(common_device, self, "wrapper_out_npu_sort_v2_out", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_sort_v2_out(self, dim, descending, out);
}

at::Tensor wrapper__npu_stride_add(const at::Tensor & self, const at::Tensor & other, const at::Scalar & offset1, const at::Scalar & offset2, const at::Scalar & c1_len) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(other);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_stride_add", "self");
  c10::impl::check_and_update_common_device(common_device, other, "wrapper__npu_stride_add", "other");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_stride_add(self, other, offset1, offset2, c1_len);
}

at::Tensor wrapper__npu_stride_copy(const at::Tensor & self, at::IntArrayRef shape, at::IntArrayRef stride, const at::Scalar & storage_offset) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_stride_copy", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_stride_copy(self, shape, stride, storage_offset);
}

at::Tensor & wrapper_out_npu_stride_copy_out(const at::Tensor & self, at::IntArrayRef shape, at::IntArrayRef stride, const at::Scalar & storage_offset, at::Tensor & out) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(out);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, out, "wrapper_out_npu_stride_copy_out", "out");
  c10::impl::check_and_update_common_device(common_device, self, "wrapper_out_npu_stride_copy_out", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_stride_copy_out(self, shape, stride, storage_offset, out);
}

at::Tensor wrapper__npu_sub_sample(const at::Tensor & self, int64_t per_images, double positive_fraction) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_sub_sample", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_sub_sample(self, per_images, positive_fraction);
}

at::Tensor wrapper__npu_swiglu(const at::Tensor & self, int64_t dim) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_swiglu", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_swiglu(self, dim);
}

at::Tensor wrapper__npu_swiglu_backward(const at::Tensor & grad_output, const at::Tensor & self, int64_t dim) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(grad_output);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad_output, "wrapper__npu_swiglu_backward", "grad_output");
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_swiglu_backward", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_swiglu_backward(grad_output, self, dim);
}

::std::tuple<at::Tensor,at::Tensor> wrapper__npu_dequant_swiglu_quant(const at::Tensor & x, const ::std::optional<at::Tensor> & weight_scale, const ::std::optional<at::Tensor> & activation_scale, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & quant_scale, const ::std::optional<at::Tensor> & quant_offset, const ::std::optional<at::Tensor> & group_index, bool activate_left, int64_t quant_mode) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(x);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, x, "wrapper__npu_dequant_swiglu_quant", "x");
  c10::impl::check_and_update_common_device(common_device, weight_scale, "wrapper__npu_dequant_swiglu_quant", "weight_scale");
  c10::impl::check_and_update_common_device(common_device, activation_scale, "wrapper__npu_dequant_swiglu_quant", "activation_scale");
  c10::impl::check_and_update_common_device(common_device, bias, "wrapper__npu_dequant_swiglu_quant", "bias");
  c10::impl::check_and_update_common_device(common_device, quant_scale, "wrapper__npu_dequant_swiglu_quant", "quant_scale");
  c10::impl::check_and_update_common_device(common_device, quant_offset, "wrapper__npu_dequant_swiglu_quant", "quant_offset");
  c10::impl::check_and_update_common_device(common_device, group_index, "wrapper__npu_dequant_swiglu_quant", "group_index");
  const c10::OptionalDeviceGuard device_guard(device_of(x));
  return op_plugin::npu_dequant_swiglu_quant(x, weight_scale, activation_scale, bias, quant_scale, quant_offset, group_index, activate_left, quant_mode);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrapper__npu_dequant_rope_quant_kvcache(const at::Tensor & x, const at::Tensor & cos, const at::Tensor & sin, const at::Tensor & k_cache, const at::Tensor & v_cache, const at::Tensor & indices, const at::Tensor & scale_k, const at::Tensor & scale_v, at::IntArrayRef size_splits, const ::std::optional<at::Tensor> & offset_k, const ::std::optional<at::Tensor> & offset_v, const ::std::optional<at::Tensor> & weight_scale, const ::std::optional<at::Tensor> & activation_scale, const ::std::optional<at::Tensor> & bias, int64_t quant_mode, c10::string_view input_layout, bool kv_output, c10::string_view cache_mode) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(x);
      c10_npu::check_npu_tensor_is_safe(cos);
      c10_npu::check_npu_tensor_is_safe(sin);
      c10_npu::check_npu_tensor_is_safe(k_cache);
      c10_npu::check_npu_tensor_is_safe(v_cache);
      c10_npu::check_npu_tensor_is_safe(indices);
      c10_npu::check_npu_tensor_is_safe(scale_k);
      c10_npu::check_npu_tensor_is_safe(scale_v);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, x, "wrapper__npu_dequant_rope_quant_kvcache", "x");
  c10::impl::check_and_update_common_device(common_device, cos, "wrapper__npu_dequant_rope_quant_kvcache", "cos");
  c10::impl::check_and_update_common_device(common_device, sin, "wrapper__npu_dequant_rope_quant_kvcache", "sin");
  c10::impl::check_and_update_common_device(common_device, k_cache, "wrapper__npu_dequant_rope_quant_kvcache", "k_cache");
  c10::impl::check_and_update_common_device(common_device, v_cache, "wrapper__npu_dequant_rope_quant_kvcache", "v_cache");
  c10::impl::check_and_update_common_device(common_device, indices, "wrapper__npu_dequant_rope_quant_kvcache", "indices");
  c10::impl::check_and_update_common_device(common_device, scale_k, "wrapper__npu_dequant_rope_quant_kvcache", "scale_k");
  c10::impl::check_and_update_common_device(common_device, scale_v, "wrapper__npu_dequant_rope_quant_kvcache", "scale_v");
  c10::impl::check_and_update_common_device(common_device, offset_k, "wrapper__npu_dequant_rope_quant_kvcache", "offset_k");
  c10::impl::check_and_update_common_device(common_device, offset_v, "wrapper__npu_dequant_rope_quant_kvcache", "offset_v");
  c10::impl::check_and_update_common_device(common_device, weight_scale, "wrapper__npu_dequant_rope_quant_kvcache", "weight_scale");
  c10::impl::check_and_update_common_device(common_device, activation_scale, "wrapper__npu_dequant_rope_quant_kvcache", "activation_scale");
  c10::impl::check_and_update_common_device(common_device, bias, "wrapper__npu_dequant_rope_quant_kvcache", "bias");
  const c10::OptionalDeviceGuard device_guard(device_of(x));
  return op_plugin::npu_dequant_rope_quant_kvcache(x, cos, sin, k_cache, v_cache, indices, scale_k, scale_v, size_splits, offset_k, offset_v, weight_scale, activation_scale, bias, quant_mode, input_layout, kv_output, cache_mode);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrapper__npu_rope_quant_kvcache(const at::Tensor & x, const at::Tensor & cos, const at::Tensor & sin, const at::Tensor & k_cache, const at::Tensor & v_cache, const at::Tensor & indices, const at::Tensor & scale_k, const at::Tensor & scale_v, at::IntArrayRef size_splits, const ::std::optional<at::Tensor> & offset_k, const ::std::optional<at::Tensor> & offset_v, int64_t quant_mode, c10::string_view input_layout, bool kv_output, c10::string_view cache_mode) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(x);
      c10_npu::check_npu_tensor_is_safe(cos);
      c10_npu::check_npu_tensor_is_safe(sin);
      c10_npu::check_npu_tensor_is_safe(k_cache);
      c10_npu::check_npu_tensor_is_safe(v_cache);
      c10_npu::check_npu_tensor_is_safe(indices);
      c10_npu::check_npu_tensor_is_safe(scale_k);
      c10_npu::check_npu_tensor_is_safe(scale_v);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, x, "wrapper__npu_rope_quant_kvcache", "x");
  c10::impl::check_and_update_common_device(common_device, cos, "wrapper__npu_rope_quant_kvcache", "cos");
  c10::impl::check_and_update_common_device(common_device, sin, "wrapper__npu_rope_quant_kvcache", "sin");
  c10::impl::check_and_update_common_device(common_device, k_cache, "wrapper__npu_rope_quant_kvcache", "k_cache");
  c10::impl::check_and_update_common_device(common_device, v_cache, "wrapper__npu_rope_quant_kvcache", "v_cache");
  c10::impl::check_and_update_common_device(common_device, indices, "wrapper__npu_rope_quant_kvcache", "indices");
  c10::impl::check_and_update_common_device(common_device, scale_k, "wrapper__npu_rope_quant_kvcache", "scale_k");
  c10::impl::check_and_update_common_device(common_device, scale_v, "wrapper__npu_rope_quant_kvcache", "scale_v");
  c10::impl::check_and_update_common_device(common_device, offset_k, "wrapper__npu_rope_quant_kvcache", "offset_k");
  c10::impl::check_and_update_common_device(common_device, offset_v, "wrapper__npu_rope_quant_kvcache", "offset_v");
  const c10::OptionalDeviceGuard device_guard(device_of(x));
  return op_plugin::npu_rope_quant_kvcache(x, cos, sin, k_cache, v_cache, indices, scale_k, scale_v, size_splits, offset_k, offset_v, quant_mode, input_layout, kv_output, cache_mode);
}

at::Tensor wrapper__npu_dequant_bias(const at::Tensor & x, const at::Tensor & weight_scale, const ::std::optional<at::Tensor> & activation_scale, const ::std::optional<at::Tensor> & bias, ::std::optional<at::ScalarType> output_dtype) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(x);
      c10_npu::check_npu_tensor_is_safe(weight_scale);
      c10_npu::check_npu_tensor_is_safe(activation_scale);
      c10_npu::check_npu_tensor_is_safe(bias);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, x, "wrapper__npu_dequant_bias", "x");
  c10::impl::check_and_update_common_device(common_device, weight_scale, "wrapper__npu_dequant_bias", "weight_scale");
  c10::impl::check_and_update_common_device(common_device, activation_scale, "wrapper__npu_dequant_bias", "activation_scale");
  c10::impl::check_and_update_common_device(common_device, bias, "wrapper__npu_dequant_bias", "bias");
  const c10::OptionalDeviceGuard device_guard(device_of(x));
  return op_plugin::npu_dequant_bias(x, weight_scale, activation_scale, bias, output_dtype);
}

at::Tensor wrapper__npu_trans_quant_param(const at::Tensor & scale, const ::std::optional<at::Tensor> & offset, ::std::optional<int64_t> round_mode) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(scale);
      c10_npu::check_npu_tensor_is_safe(offset);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, scale, "wrapper__npu_trans_quant_param", "scale");
  c10::impl::check_and_update_common_device(common_device, offset, "wrapper__npu_trans_quant_param", "offset");
  const c10::OptionalDeviceGuard device_guard(device_of(scale));
  return op_plugin::npu_trans_quant_param(scale, offset, round_mode);
}

at::Tensor wrapper__npu_transpose(const at::Tensor & self, at::IntArrayRef perm, bool require_contiguous) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_transpose", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_transpose(self, perm, require_contiguous);
}

at::Tensor & wrapper_out_npu_transpose_out(const at::Tensor & self, at::IntArrayRef perm, bool require_contiguous, at::Tensor & out) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(out);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, out, "wrapper_out_npu_transpose_out", "out");
  c10::impl::check_and_update_common_device(common_device, self, "wrapper_out_npu_transpose_out", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_transpose_out(self, perm, require_contiguous, out);
}

at::Tensor & wrapper__npu_view_copy(at::Tensor & self, const at::Tensor & other, bool non_blocking) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(other);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_view_copy", "self");
  c10::impl::check_and_update_common_device(common_device, other, "wrapper__npu_view_copy", "other");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_view_copy(self, other, non_blocking);
}

at::Tensor wrapper__npu_weight_quant_batchmatmul(const at::Tensor & x, const at::Tensor & weight, const at::Tensor & antiquant_scale, const ::std::optional<at::Tensor> & antiquant_offset, const ::std::optional<at::Tensor> & quant_scale, const ::std::optional<at::Tensor> & quant_offset, const ::std::optional<at::Tensor> & bias, int64_t antiquant_group_size, int64_t inner_precise) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(x);
      c10_npu::check_npu_tensor_is_safe(weight);
      c10_npu::check_npu_tensor_is_safe(antiquant_scale);
      c10_npu::check_npu_tensor_is_safe(antiquant_offset);
      c10_npu::check_npu_tensor_is_safe(quant_scale);
      c10_npu::check_npu_tensor_is_safe(quant_offset);
      c10_npu::check_npu_tensor_is_safe(bias);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, x, "wrapper__npu_weight_quant_batchmatmul", "x");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__npu_weight_quant_batchmatmul", "weight");
  c10::impl::check_and_update_common_device(common_device, antiquant_scale, "wrapper__npu_weight_quant_batchmatmul", "antiquant_scale");
  c10::impl::check_and_update_common_device(common_device, antiquant_offset, "wrapper__npu_weight_quant_batchmatmul", "antiquant_offset");
  c10::impl::check_and_update_common_device(common_device, quant_scale, "wrapper__npu_weight_quant_batchmatmul", "quant_scale");
  c10::impl::check_and_update_common_device(common_device, quant_offset, "wrapper__npu_weight_quant_batchmatmul", "quant_offset");
  c10::impl::check_and_update_common_device(common_device, bias, "wrapper__npu_weight_quant_batchmatmul", "bias");
  const c10::OptionalDeviceGuard device_guard(device_of(x));
  return op_plugin::npu_weight_quant_batchmatmul(x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias, antiquant_group_size, inner_precise);
}

at::Tensor wrapper__npu_transpose_batchmatmul(const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & scale, at::OptionalIntArrayRef perm_x1, at::OptionalIntArrayRef perm_x2, at::OptionalIntArrayRef perm_y, ::std::optional<int64_t> batch_split_factor) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(weight);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_transpose_batchmatmul", "input");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__npu_transpose_batchmatmul", "weight");
  c10::impl::check_and_update_common_device(common_device, bias, "wrapper__npu_transpose_batchmatmul", "bias");
  c10::impl::check_and_update_common_device(common_device, scale, "wrapper__npu_transpose_batchmatmul", "scale");
  const c10::OptionalDeviceGuard device_guard(device_of(input));
  return op_plugin::npu_transpose_batchmatmul(input, weight, bias, scale, perm_x1, perm_x2, perm_y, batch_split_factor);
}

at::Tensor wrapper__npu_yolo_boxes_encode(const at::Tensor & self, const at::Tensor & gt_bboxes, const at::Tensor & stride, bool performance_mode) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(gt_bboxes);
      c10_npu::check_npu_tensor_is_safe(stride);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_yolo_boxes_encode", "self");
  c10::impl::check_and_update_common_device(common_device, gt_bboxes, "wrapper__npu_yolo_boxes_encode", "gt_bboxes");
  c10::impl::check_and_update_common_device(common_device, stride, "wrapper__npu_yolo_boxes_encode", "stride");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_yolo_boxes_encode(self, gt_bboxes, stride, performance_mode);
}

at::Tensor & wrapper__one_(at::Tensor & self) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__one_", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::one_(self);
}

at::Tensor wrapper__repeat_interleave_backward_int(const at::Tensor & grad, const at::Tensor & self, c10::SymInt repeats, ::std::optional<int64_t> dim) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(grad);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad, "wrapper__repeat_interleave_backward_int", "grad");
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__repeat_interleave_backward_int", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::repeat_interleave_backward_int_symint(grad, self, repeats, dim);
}

at::Tensor wrapper__repeat_interleave_backward_tensor(const at::Tensor & grad, const at::Tensor & self, const at::Tensor & repeats, ::std::optional<int64_t> dim) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(grad);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(repeats);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad, "wrapper__repeat_interleave_backward_tensor", "grad");
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__repeat_interleave_backward_tensor", "self");
  c10::impl::check_and_update_common_device(common_device, repeats, "wrapper__repeat_interleave_backward_tensor", "repeats");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::repeat_interleave_backward_tensor(grad, self, repeats, dim);
}

at::Tensor wrapper__scatter_update(const at::Tensor & self, const at::Tensor & indices, const at::Tensor & updates, int64_t axis) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(indices);
      c10_npu::check_npu_tensor_is_safe(updates);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__scatter_update", "self");
  c10::impl::check_and_update_common_device(common_device, indices, "wrapper__scatter_update", "indices");
  c10::impl::check_and_update_common_device(common_device, updates, "wrapper__scatter_update", "updates");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::scatter_update(self, indices, updates, axis);
}

at::Tensor & wrapper__scatter_update_(at::Tensor & self, const at::Tensor & indices, const at::Tensor & updates, int64_t axis) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(indices);
      c10_npu::check_npu_tensor_is_safe(updates);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__scatter_update_", "self");
  c10::impl::check_and_update_common_device(common_device, indices, "wrapper__scatter_update_", "indices");
  c10::impl::check_and_update_common_device(common_device, updates, "wrapper__scatter_update_", "updates");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::scatter_update_(self, indices, updates, axis);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapper__slow_conv_transpose2d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, ::std::array<bool,3> output_mask) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(grad_output);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(weight);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad_output, "wrapper__slow_conv_transpose2d_backward", "grad_output");
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__slow_conv_transpose2d_backward", "self");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__slow_conv_transpose2d_backward", "weight");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::slow_conv_transpose2d_backward(grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, output_mask);
}

at::Tensor wrapper__stft_backward(const at::Tensor & grad_output, const at::Tensor & self, int64_t n_fft, ::std::optional<int64_t> hop_length, ::std::optional<int64_t> win_length, const ::std::optional<at::Tensor> & window, bool normalized, ::std::optional<bool> onesided, ::std::optional<bool> return_complex) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(grad_output);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(window);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad_output, "wrapper__stft_backward", "grad_output");
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__stft_backward", "self");
  c10::impl::check_and_update_common_device(common_device, window, "wrapper__stft_backward", "window");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::stft_backward(grad_output, self, n_fft, hop_length, win_length, window, normalized, onesided, return_complex);
}

at::Tensor wrapper__fft_r2c_backward(const at::Tensor & grad, at::IntArrayRef dim, int64_t normalization, bool onesided, int64_t last_dim_size) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(grad);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad, "wrapper__fft_r2c_backward", "grad");
  const c10::OptionalDeviceGuard device_guard(device_of(grad));
  return op_plugin::fft_r2c_backward(grad, dim, normalization, onesided, last_dim_size);
}

at::Tensor wrapper__fft_c2r_backward(const at::Tensor & grad, at::IntArrayRef dim, int64_t normalization) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(grad);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad, "wrapper__fft_c2r_backward", "grad");
  const c10::OptionalDeviceGuard device_guard(device_of(grad));
  return op_plugin::fft_c2r_backward(grad, dim, normalization);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrapper__npu_cross_entropy_loss(const at::Tensor & input, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, c10::string_view reduction, int64_t ignore_index, double label_smoothing, double lse_square_scale_for_zloss, bool return_zloss) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(target);
      c10_npu::check_npu_tensor_is_safe(weight);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_cross_entropy_loss", "input");
  c10::impl::check_and_update_common_device(common_device, target, "wrapper__npu_cross_entropy_loss", "target");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__npu_cross_entropy_loss", "weight");
  const c10::OptionalDeviceGuard device_guard(device_of(input));
  return op_plugin::npu_cross_entropy_loss(input, target, weight, reduction, ignore_index, label_smoothing, lse_square_scale_for_zloss, return_zloss);
}

at::Tensor wrapper__npu_cross_entropy_loss_backward(const at::Tensor & grad_loss, const at::Tensor & log_prob, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & grad_zloss, const ::std::optional<at::Tensor> & lse_for_zloss, c10::string_view reduction, int64_t ignore_index, double label_smoothing, double lse_square_scale_for_zloss) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(grad_loss);
      c10_npu::check_npu_tensor_is_safe(log_prob);
      c10_npu::check_npu_tensor_is_safe(target);
      c10_npu::check_npu_tensor_is_safe(weight);
      c10_npu::check_npu_tensor_is_safe(grad_zloss);
      c10_npu::check_npu_tensor_is_safe(lse_for_zloss);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad_loss, "wrapper__npu_cross_entropy_loss_backward", "grad_loss");
  c10::impl::check_and_update_common_device(common_device, log_prob, "wrapper__npu_cross_entropy_loss_backward", "log_prob");
  c10::impl::check_and_update_common_device(common_device, target, "wrapper__npu_cross_entropy_loss_backward", "target");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__npu_cross_entropy_loss_backward", "weight");
  c10::impl::check_and_update_common_device(common_device, grad_zloss, "wrapper__npu_cross_entropy_loss_backward", "grad_zloss");
  c10::impl::check_and_update_common_device(common_device, lse_for_zloss, "wrapper__npu_cross_entropy_loss_backward", "lse_for_zloss");
  const c10::OptionalDeviceGuard device_guard(device_of(grad_loss));
  return op_plugin::npu_cross_entropy_loss_backward(grad_loss, log_prob, target, weight, grad_zloss, lse_for_zloss, reduction, ignore_index, label_smoothing, lse_square_scale_for_zloss);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapper__npu_group_norm_swish(const at::Tensor & input, int64_t num_groups, const at::Tensor & weight, const at::Tensor & bias, ::std::optional<double> eps, ::std::optional<double> swish_scale) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(weight);
      c10_npu::check_npu_tensor_is_safe(bias);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_group_norm_swish", "input");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__npu_group_norm_swish", "weight");
  c10::impl::check_and_update_common_device(common_device, bias, "wrapper__npu_group_norm_swish", "bias");
  const c10::OptionalDeviceGuard device_guard(device_of(input));
  return op_plugin::npu_group_norm_swish(input, num_groups, weight, bias, eps, swish_scale);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapper__npu_group_norm_swish_grad(const at::Tensor & grad, const at::Tensor & input, int64_t num_groups, const at::Tensor & weight, const at::Tensor & bias, const at::Tensor & mean, const at::Tensor & rstd, ::std::array<bool,3> grad_input_mask, ::std::optional<double> swish_scale) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(grad);
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(weight);
      c10_npu::check_npu_tensor_is_safe(bias);
      c10_npu::check_npu_tensor_is_safe(mean);
      c10_npu::check_npu_tensor_is_safe(rstd);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad, "wrapper__npu_group_norm_swish_grad", "grad");
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_group_norm_swish_grad", "input");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__npu_group_norm_swish_grad", "weight");
  c10::impl::check_and_update_common_device(common_device, bias, "wrapper__npu_group_norm_swish_grad", "bias");
  c10::impl::check_and_update_common_device(common_device, mean, "wrapper__npu_group_norm_swish_grad", "mean");
  c10::impl::check_and_update_common_device(common_device, rstd, "wrapper__npu_group_norm_swish_grad", "rstd");
  const c10::OptionalDeviceGuard device_guard(device_of(grad));
  return op_plugin::npu_group_norm_swish_grad(grad, input, num_groups, weight, bias, mean, rstd, grad_input_mask, swish_scale);
}

void wrapper__npu_advance_step_flashattn(at::Tensor & input_tokens, const at::Tensor & sampled_token_ids, at::Tensor & input_positions, at::Tensor & seq_lens, at::Tensor & slot_mapping, const at::Tensor & block_tables, int64_t num_seqs, int64_t num_queries, int64_t block_size) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(input_tokens);
      c10_npu::check_npu_tensor_is_safe(sampled_token_ids);
      c10_npu::check_npu_tensor_is_safe(input_positions);
      c10_npu::check_npu_tensor_is_safe(seq_lens);
      c10_npu::check_npu_tensor_is_safe(slot_mapping);
      c10_npu::check_npu_tensor_is_safe(block_tables);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, input_tokens, "wrapper__npu_advance_step_flashattn", "input_tokens");
  c10::impl::check_and_update_common_device(common_device, sampled_token_ids, "wrapper__npu_advance_step_flashattn", "sampled_token_ids");
  c10::impl::check_and_update_common_device(common_device, input_positions, "wrapper__npu_advance_step_flashattn", "input_positions");
  c10::impl::check_and_update_common_device(common_device, seq_lens, "wrapper__npu_advance_step_flashattn", "seq_lens");
  c10::impl::check_and_update_common_device(common_device, slot_mapping, "wrapper__npu_advance_step_flashattn", "slot_mapping");
  c10::impl::check_and_update_common_device(common_device, block_tables, "wrapper__npu_advance_step_flashattn", "block_tables");
  const c10::OptionalDeviceGuard device_guard(device_of(input_tokens));
  return op_plugin::npu_advance_step_flashattn(input_tokens, sampled_token_ids, input_positions, seq_lens, slot_mapping, block_tables, num_seqs, num_queries, block_size);
}

at::Tensor & wrapper__npu_grouped_matmul_add_(at::Tensor & self, const at::Tensor & x, const at::Tensor & weight, const at::Tensor & group_list, bool transpose_x, bool transpose_weight, int64_t group_type) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(x);
      c10_npu::check_npu_tensor_is_safe(weight);
      c10_npu::check_npu_tensor_is_safe(group_list);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_grouped_matmul_add_", "self");
  c10::impl::check_and_update_common_device(common_device, x, "wrapper__npu_grouped_matmul_add_", "x");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__npu_grouped_matmul_add_", "weight");
  c10::impl::check_and_update_common_device(common_device, group_list, "wrapper__npu_grouped_matmul_add_", "group_list");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_grouped_matmul_add_(self, x, weight, group_list, transpose_x, transpose_weight, group_type);
}

at::Tensor & wrapper__npu_attn_softmax_(at::Tensor & self) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_attn_softmax_", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_attn_softmax_(self);
}

at::Tensor & wrapper__npu_attn_softmax_backward_(at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & values) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(self);
      c10_npu::check_npu_tensor_is_safe(grad_output);
      c10_npu::check_npu_tensor_is_safe(values);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper__npu_attn_softmax_backward_", "self");
  c10::impl::check_and_update_common_device(common_device, grad_output, "wrapper__npu_attn_softmax_backward_", "grad_output");
  c10::impl::check_and_update_common_device(common_device, values, "wrapper__npu_attn_softmax_backward_", "values");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return op_plugin::npu_attn_softmax_backward_(self, grad_output, values);
}

at::Tensor wrapper__npu_gather_sparse_index(const at::Tensor & input, const at::Tensor & index) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(index);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_gather_sparse_index", "input");
  c10::impl::check_and_update_common_device(common_device, index, "wrapper__npu_gather_sparse_index", "index");
  const c10::OptionalDeviceGuard device_guard(device_of(input));
  return op_plugin::npu_gather_sparse_index(input, index);
}

at::Tensor wrapper__npu_nsa_compress(const at::Tensor & input, const at::Tensor & weight, int64_t compress_block_size, int64_t compress_stride, at::OptionalIntArrayRef actual_seq_len) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(weight);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_nsa_compress", "input");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__npu_nsa_compress", "weight");
  const c10::OptionalDeviceGuard device_guard(device_of(input));
  return op_plugin::npu_nsa_compress(input, weight, compress_block_size, compress_stride, actual_seq_len);
}

::std::tuple<at::Tensor,at::Tensor> wrapper__npu_nsa_compress_grad(const at::Tensor & grad, const at::Tensor & input, const at::Tensor & weight, int64_t compress_block_size, int64_t compress_stride, at::OptionalIntArrayRef actual_seq_len) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(grad);
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(weight);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad, "wrapper__npu_nsa_compress_grad", "grad");
  c10::impl::check_and_update_common_device(common_device, input, "wrapper__npu_nsa_compress_grad", "input");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper__npu_nsa_compress_grad", "weight");
  const c10::OptionalDeviceGuard device_guard(device_of(grad));
  return op_plugin::npu_nsa_compress_grad(grad, input, weight, compress_block_size, compress_stride, actual_seq_len);
}

at::Tensor & wrapper_cache_npu_nsa_compress_infer_out(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & slot_mapping, int64_t compress_block_size, int64_t compress_stride, int64_t page_block_size, const ::std::optional<at::Tensor> & block_table, at::OptionalIntArrayRef actual_seq_len, at::Tensor & cache) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(cache);
      c10_npu::check_npu_tensor_is_safe(input);
      c10_npu::check_npu_tensor_is_safe(weight);
      c10_npu::check_npu_tensor_is_safe(slot_mapping);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, cache, "wrapper_cache_npu_nsa_compress_infer_out", "cache");
  c10::impl::check_and_update_common_device(common_device, input, "wrapper_cache_npu_nsa_compress_infer_out", "input");
  c10::impl::check_and_update_common_device(common_device, weight, "wrapper_cache_npu_nsa_compress_infer_out", "weight");
  c10::impl::check_and_update_common_device(common_device, slot_mapping, "wrapper_cache_npu_nsa_compress_infer_out", "slot_mapping");
  c10::impl::check_and_update_common_device(common_device, block_table, "wrapper_cache_npu_nsa_compress_infer_out", "block_table");
  const c10::OptionalDeviceGuard device_guard(device_of(cache));
  return op_plugin::npu_nsa_compress_infer_out(input, weight, slot_mapping, compress_block_size, compress_stride, page_block_size, block_table, actual_seq_len, cache);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> wrapper__npu_nsa_compress_attention(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, double scale_value, int64_t head_num, int64_t compress_block_size, int64_t compress_stride, int64_t select_block_size, int64_t select_block_count, const ::std::optional<at::Tensor> & topk_mask, const ::std::optional<at::Tensor> & atten_mask, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_cmp_seq_kvlen, at::OptionalIntArrayRef actual_sel_seq_kvlen) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(query);
      c10_npu::check_npu_tensor_is_safe(key);
      c10_npu::check_npu_tensor_is_safe(value);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, query, "wrapper__npu_nsa_compress_attention", "query");
  c10::impl::check_and_update_common_device(common_device, key, "wrapper__npu_nsa_compress_attention", "key");
  c10::impl::check_and_update_common_device(common_device, value, "wrapper__npu_nsa_compress_attention", "value");
  c10::impl::check_and_update_common_device(common_device, topk_mask, "wrapper__npu_nsa_compress_attention", "topk_mask");
  c10::impl::check_and_update_common_device(common_device, atten_mask, "wrapper__npu_nsa_compress_attention", "atten_mask");
  const c10::OptionalDeviceGuard device_guard(device_of(query));
  return op_plugin::npu_nsa_compress_attention(query, key, value, scale_value, head_num, compress_block_size, compress_stride, select_block_size, select_block_count, topk_mask, atten_mask, actual_seq_qlen, actual_cmp_seq_kvlen, actual_sel_seq_kvlen);
}

::std::tuple<at::Tensor,at::Tensor> wrapper__npu_nsa_compress_attention_infer(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, double scale_value, int64_t head_num, int64_t key_value_head_num, int64_t select_block_size, int64_t select_block_count, int64_t page_block_size, int64_t compress_block_size, int64_t compress_stride, const ::std::optional<at::Tensor> & atten_mask, const ::std::optional<at::Tensor> & block_table, const ::std::optional<at::Tensor> & topk_mask, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_cmp_seq_kvlen, at::OptionalIntArrayRef actual_sel_seq_kvlen) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(query);
      c10_npu::check_npu_tensor_is_safe(key);
      c10_npu::check_npu_tensor_is_safe(value);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, query, "wrapper__npu_nsa_compress_attention_infer", "query");
  c10::impl::check_and_update_common_device(common_device, key, "wrapper__npu_nsa_compress_attention_infer", "key");
  c10::impl::check_and_update_common_device(common_device, value, "wrapper__npu_nsa_compress_attention_infer", "value");
  c10::impl::check_and_update_common_device(common_device, atten_mask, "wrapper__npu_nsa_compress_attention_infer", "atten_mask");
  c10::impl::check_and_update_common_device(common_device, block_table, "wrapper__npu_nsa_compress_attention_infer", "block_table");
  c10::impl::check_and_update_common_device(common_device, topk_mask, "wrapper__npu_nsa_compress_attention_infer", "topk_mask");
  const c10::OptionalDeviceGuard device_guard(device_of(query));
  return op_plugin::npu_nsa_compress_attention_infer(query, key, value, scale_value, head_num, key_value_head_num, select_block_size, select_block_count, page_block_size, compress_block_size, compress_stride, atten_mask, block_table, topk_mask, actual_seq_qlen, actual_cmp_seq_kvlen, actual_sel_seq_kvlen);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapper__npu_nsa_select_attention(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & topk_indices, double scale_value, int64_t head_num, int64_t select_block_size, int64_t select_block_count, const ::std::optional<at::Tensor> & atten_mask, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_seq_kvlen) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(query);
      c10_npu::check_npu_tensor_is_safe(key);
      c10_npu::check_npu_tensor_is_safe(value);
      c10_npu::check_npu_tensor_is_safe(topk_indices);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, query, "wrapper__npu_nsa_select_attention", "query");
  c10::impl::check_and_update_common_device(common_device, key, "wrapper__npu_nsa_select_attention", "key");
  c10::impl::check_and_update_common_device(common_device, value, "wrapper__npu_nsa_select_attention", "value");
  c10::impl::check_and_update_common_device(common_device, topk_indices, "wrapper__npu_nsa_select_attention", "topk_indices");
  c10::impl::check_and_update_common_device(common_device, atten_mask, "wrapper__npu_nsa_select_attention", "atten_mask");
  const c10::OptionalDeviceGuard device_guard(device_of(query));
  return op_plugin::npu_nsa_select_attention(query, key, value, topk_indices, scale_value, head_num, select_block_size, select_block_count, atten_mask, actual_seq_qlen, actual_seq_kvlen);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> wrapper__npu_nsa_select_attention_grad(const at::Tensor & grad, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & attention_out, const at::Tensor & softmax_max, const at::Tensor & softmax_sum, const at::Tensor & topk_indices, double scale_value, int64_t head_num, int64_t select_block_size, int64_t select_block_count, const ::std::optional<at::Tensor> & atten_mask, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_seq_kvlen) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(grad);
      c10_npu::check_npu_tensor_is_safe(query);
      c10_npu::check_npu_tensor_is_safe(key);
      c10_npu::check_npu_tensor_is_safe(value);
      c10_npu::check_npu_tensor_is_safe(attention_out);
      c10_npu::check_npu_tensor_is_safe(softmax_max);
      c10_npu::check_npu_tensor_is_safe(softmax_sum);
      c10_npu::check_npu_tensor_is_safe(topk_indices);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, grad, "wrapper__npu_nsa_select_attention_grad", "grad");
  c10::impl::check_and_update_common_device(common_device, query, "wrapper__npu_nsa_select_attention_grad", "query");
  c10::impl::check_and_update_common_device(common_device, key, "wrapper__npu_nsa_select_attention_grad", "key");
  c10::impl::check_and_update_common_device(common_device, value, "wrapper__npu_nsa_select_attention_grad", "value");
  c10::impl::check_and_update_common_device(common_device, attention_out, "wrapper__npu_nsa_select_attention_grad", "attention_out");
  c10::impl::check_and_update_common_device(common_device, softmax_max, "wrapper__npu_nsa_select_attention_grad", "softmax_max");
  c10::impl::check_and_update_common_device(common_device, softmax_sum, "wrapper__npu_nsa_select_attention_grad", "softmax_sum");
  c10::impl::check_and_update_common_device(common_device, topk_indices, "wrapper__npu_nsa_select_attention_grad", "topk_indices");
  c10::impl::check_and_update_common_device(common_device, atten_mask, "wrapper__npu_nsa_select_attention_grad", "atten_mask");
  const c10::OptionalDeviceGuard device_guard(device_of(grad));
  return op_plugin::npu_nsa_select_attention_grad(grad, query, key, value, attention_out, softmax_max, softmax_sum, topk_indices, scale_value, head_num, select_block_size, select_block_count, atten_mask, actual_seq_qlen, actual_seq_kvlen);
}

at::Tensor wrapper__npu_nsa_select_attention_infer(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & topk_indices, double scale_value, int64_t head_num, int64_t key_value_head_num, int64_t select_block_size, int64_t select_block_count, int64_t page_block_size, c10::string_view layout, const ::std::optional<at::Tensor> & atten_mask, const ::std::optional<at::Tensor> & block_table, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_seq_kvlen) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(query);
      c10_npu::check_npu_tensor_is_safe(key);
      c10_npu::check_npu_tensor_is_safe(value);
      c10_npu::check_npu_tensor_is_safe(topk_indices);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, query, "wrapper__npu_nsa_select_attention_infer", "query");
  c10::impl::check_and_update_common_device(common_device, key, "wrapper__npu_nsa_select_attention_infer", "key");
  c10::impl::check_and_update_common_device(common_device, value, "wrapper__npu_nsa_select_attention_infer", "value");
  c10::impl::check_and_update_common_device(common_device, topk_indices, "wrapper__npu_nsa_select_attention_infer", "topk_indices");
  c10::impl::check_and_update_common_device(common_device, atten_mask, "wrapper__npu_nsa_select_attention_infer", "atten_mask");
  c10::impl::check_and_update_common_device(common_device, block_table, "wrapper__npu_nsa_select_attention_infer", "block_table");
  const c10::OptionalDeviceGuard device_guard(device_of(query));
  return op_plugin::npu_nsa_select_attention_infer(query, key, value, topk_indices, scale_value, head_num, key_value_head_num, select_block_size, select_block_count, page_block_size, layout, atten_mask, block_table, actual_seq_qlen, actual_seq_kvlen);
}

at::Tensor wrapper__npu_top_k_top_p(const at::Tensor & logits, const at::Tensor & p, const at::Tensor & k) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(logits);
      c10_npu::check_npu_tensor_is_safe(p);
      c10_npu::check_npu_tensor_is_safe(k);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, logits, "wrapper__npu_top_k_top_p", "logits");
  c10::impl::check_and_update_common_device(common_device, p, "wrapper__npu_top_k_top_p", "p");
  c10::impl::check_and_update_common_device(common_device, k, "wrapper__npu_top_k_top_p", "k");
  const c10::OptionalDeviceGuard device_guard(device_of(logits));
  return op_plugin::npu_top_k_top_p(logits, p, k);
}

::std::tuple<at::Tensor,at::Tensor> wrapper__npu_moe_token_permute(const at::Tensor & tokens, const at::Tensor & indices, ::std::optional<int64_t> num_out_tokens, bool padded_mode) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(tokens);
      c10_npu::check_npu_tensor_is_safe(indices);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, tokens, "wrapper__npu_moe_token_permute", "tokens");
  c10::impl::check_and_update_common_device(common_device, indices, "wrapper__npu_moe_token_permute", "indices");
  const c10::OptionalDeviceGuard device_guard(device_of(tokens));
  return op_plugin::npu_moe_token_permute(tokens, indices, num_out_tokens, padded_mode);
}

at::Tensor wrapper__npu_moe_token_unpermute(const at::Tensor & permuted_tokens, const at::Tensor & sorted_indices, const ::std::optional<at::Tensor> & probs, bool padded_mode, at::OptionalIntArrayRef restore_shape) {

  if (c10_npu::get_npu_data_unsafe_flag()) {
      c10_npu::check_npu_tensor_is_safe(permuted_tokens);
      c10_npu::check_npu_tensor_is_safe(sorted_indices);
      c10_npu::check_npu_tensor_is_safe(probs);
  }
  c10::optional<at::Device> common_device = at::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, permuted_tokens, "wrapper__npu_moe_token_unpermute", "permuted_tokens");
  c10::impl::check_and_update_common_device(common_device, sorted_indices, "wrapper__npu_moe_token_unpermute", "sorted_indices");
  c10::impl::check_and_update_common_device(common_device, probs, "wrapper__npu_moe_token_unpermute", "probs");
  const c10::OptionalDeviceGuard device_guard(device_of(permuted_tokens));
  return op_plugin::npu_moe_token_unpermute(permuted_tokens, sorted_indices, probs, padded_mode, restore_shape);
}

namespace {

TORCH_LIBRARY(npu, m) {

  const std::vector<at::Tag> tags_0 = {at::Tag::pt2_compliant_tag};
  m.def("npu_change_data_ptr(Tensor dst, Tensor src, int index) -> int", tags_0);
  m.def("get_npu_format(Tensor input) -> int", tags_0);
  m.def("npu_format_cast.Tensor(Tensor input, Tensor dst) -> Tensor", tags_0);
  m.def("npu_format_cast_.acl_format(Tensor(a!) input, int acl_format) -> Tensor(a!)", tags_0);
  m.def("npu_format_cast_(Tensor(a!) input, Tensor src) -> Tensor(a!)", tags_0);
  m.def("empty_with_format(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, int acl_format=2, int? base_addr_aligned_kb=None) -> Tensor", TORCH_FN(at_npu::native::wrapper__empty_with_format), tags_0);
  m.def("unsafe_empty_with_format(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, int acl_format=2, bool keep_format=False) -> Tensor", TORCH_FN(at_npu::native::wrapper__unsafe_empty_with_format), tags_0);
  m.def("empty_with_format.names(int[] size, Dimname[]? names, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, int acl_format=2) -> Tensor", TORCH_FN(at_npu::native::wrapper_names_empty_with_format), tags_0);
  m.def("copy_memory_(Tensor(a!) input, Tensor src, bool non_blocking=False) -> Tensor(a!)", tags_0);
  m.def("get_storage_size(Tensor input) -> int", tags_0);
  m.def("npu_format_cast(Tensor input, int acl_format) -> Tensor", tags_0);
  m.def("_npu_format_cast(Tensor input, int acl_format) -> Tensor", tags_0);
  m.def("empty_with_swapped_memory(int[] size, *, ScalarType? dtype=None, Device? device=None) -> Tensor", TORCH_FN(at_npu::native::wrapper__empty_with_swapped_memory), tags_0);
  m.def("npu_gather_backward(Tensor grad, SymInt[] self_size, int dim, Tensor index, bool sparse_grad) -> Tensor", tags_0);
  m.def("_amp_foreach_non_finite_check(Tensor[] scaled_grads) -> bool", tags_0);
  m.def("npu_gelu(Tensor input, *, str approximate='none') -> Tensor", tags_0);
  m.def("npu_gelu_backward(Tensor grad_output, Tensor input, *, str approximate='none') -> Tensor", tags_0);
  m.def("_conv_depthwise2d_backward(Tensor grad_output, Tensor input, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool[2] output_mask) -> (Tensor grad_input, Tensor grad_weight)", tags_0);
  const std::vector<at::Tag> tags_1 = {at::Tag::nondeterministic_seeded, at::Tag::pt2_compliant_tag};
  m.def("_dropout_with_byte_mask(Tensor input, float p) -> (Tensor, Tensor)", tags_1);
  m.def("_dropout_with_byte_mask_backward(Tensor grad_output, Tensor mask, float p) -> Tensor", tags_0);
  m.def("_npu_ciou(Tensor input, Tensor gtboxes, bool trans=False, bool is_cross=True, int mode=0, bool atan_sub_flag=False) -> (Tensor, Tensor)", tags_0);
  m.def("_npu_dropout(Tensor input, float p) -> (Tensor, Tensor)", tags_1);
  m.def("_npu_dropout_gen_mask.Tensor(Tensor input, int[] size, float p, int seed, int offset, *, bool? parallel=True, bool? sync=None) -> Tensor", tags_1);
  m.def("_npu_silent_check(Tensor(a!) input_grad, Tensor val, Tensor(b!) pre_val, Tensor(c!) min_val, Tensor(d!) max_val, Tensor val_counter, int c_min_steps, float c_thresh_l1, float c_coeff_l1, float c_thresh_l2, float c_coeff_l2) -> Tensor", tags_0);
  m.def("_npu_silent_check_v2(Tensor val, Tensor(a!) input_grad, Tensor(b!) sfda, Tensor(c!) step, int c_min_steps, float c_thresh_l1, float c_coeff_l1, float c_thresh_l2, float c_coeff_l2, int npu_asd_detect) -> Tensor", tags_0);
  m.def("_npu_silent_check_v3(Tensor val, Tensor(a!) input_grad, Tensor(b!) step, Tensor(c!) max, Tensor(d!) avg, float c_thresh_l1, float c_thresh_l2, float betal, int npu_asd_detect) -> Tensor", tags_0);
  m.def("batch_norm_gather_stats_update(Tensor input, Tensor mean, Tensor invstd, Tensor? running_mean, Tensor? running_var, float momentum, float eps, Tensor counts) -> (Tensor, Tensor)", tags_0);
  m.def("batch_norm_reduce(Tensor input, float eps) -> (Tensor, Tensor)", tags_0);
  m.def("dropout_with_byte_mask(Tensor input, float p, bool train) -> Tensor", tags_1);
  m.def("fast_gelu(Tensor input) -> Tensor", tags_0);
  m.def("kl_div_backward(Tensor grad_output, Tensor input, Tensor target, int reduction=Mean, *, bool log_target=False) -> Tensor", tags_0);
  m.def("l1_loss_backward(Tensor grad_output, Tensor input, Tensor target, int reduction) -> Tensor", tags_0);
  m.def("slow_conv_dilated2d_backward(Tensor grad_output, Tensor input, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)", tags_0);
  m.def("matmul_double_backward(Tensor? grad_self, Tensor? grad_other, Tensor grad_out, Tensor input, Tensor other, bool[3] mask) -> (Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_add_layer_norm(Tensor x1, Tensor x2, Tensor gamma, Tensor beta, float epsilon=1e-05, bool additional_output=False) -> (Tensor, Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_add_layer_norm_backward(Tensor? dy_opt, Tensor x1, Tensor x2, Tensor rstd, Tensor mean, Tensor gamma, Tensor? dsum_opt) -> (Tensor, Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_add_rms_norm(Tensor x1, Tensor x2, Tensor gamma, float epsilon=1e-06) -> (Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_add_rms_norm_cast(Tensor x1, Tensor x2, Tensor gamma, float epsilon=1e-06) -> (Tensor, Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_add_rms_norm_quant(Tensor x1, Tensor x2, Tensor gamma, Tensor scales1, Tensor? zero_points1, Tensor? scales2=None, Tensor? zero_points2=None, *, int axis=-1, float epsilon=1e-06, bool div_mode=True) -> (Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_alltoallv_gmm(Tensor gmm_x, Tensor gmm_weight, str hcom, int ep_world_size, int[] send_counts, int[] recv_counts, *, Tensor? send_counts_tensor=None, Tensor? recv_counts_tensor=None, Tensor? mm_x=None, Tensor? mm_weight=None, bool trans_gmm_weight=False, bool trans_mm_weight=False, bool permute_out_flag=False) -> (Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_gmm_alltoallv(Tensor gmm_x, Tensor gmm_weight, str hcom, int ep_world_size, int[] send_counts, int[] recv_counts, *, Tensor? send_counts_tensor=None, Tensor? recv_counts_tensor=None, Tensor? mm_x=None, Tensor? mm_weight=None, bool trans_gmm_weight=False, bool trans_mm_weight=False) -> (Tensor, Tensor)", tags_0);
  m.def("npu_all_gather_base_mm(Tensor input, Tensor x2, str hcom, int world_size, *, Tensor? bias=None, int gather_index=0, bool gather_output=True, int comm_turn=0) -> (Tensor, Tensor)", tags_0);
  m.def("npu_alloc_float_status(Tensor input) -> Tensor", tags_0);
  m.def("npu_anchor_response_flags(Tensor input, int[2] featmap_size, int[2] stride, int num_base_anchors) -> Tensor", tags_0);
  m.def("npu_anti_quant(Tensor x, Tensor scale, *, Tensor? offset=None, ScalarType? dst_dtype=None, ScalarType? src_dtype=None) -> Tensor", tags_0);
  m.def("npu_apply_adam(Scalar beta1_power, Scalar beta2_power, Scalar lr, Scalar beta1, Scalar beta2, Scalar epsilon, Tensor grad, bool? use_locking, bool? use_nesterov) -> (Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_apply_adam.out(Scalar beta1_power, Scalar beta2_power, Scalar lr, Scalar beta1, Scalar beta2, Scalar epsilon, Tensor grad, bool? use_locking, bool? use_nesterov, *, Tensor[] out) -> (Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_apply_adam_w(Scalar beta1_power, Scalar beta2_power, Scalar lr, Scalar weight_decay, Scalar beta1, Scalar beta2, Scalar epsilon, Tensor grad, Tensor? max_grad_norm, bool? amsgrad, bool? maximize) -> (Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_apply_adam_w.out(Scalar beta1_power, Scalar beta2_power, Scalar lr, Scalar weight_decay, Scalar beta1, Scalar beta2, Scalar epsilon, Tensor grad, Tensor? max_grad_norm, bool? amsgrad, bool? maximize, *, Tensor[] out) -> (Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_apply_rotary_pos_emb(Tensor query, Tensor key, Tensor cos, Tensor sin, str layout='BSH') -> (Tensor, Tensor)", tags_0);
  m.def("npu_kv_rmsnorm_rope_cache(Tensor kv, Tensor gamma, Tensor cos, Tensor sin, Tensor index, Tensor k_cache, Tensor ckv_cache, *, Tensor? k_rope_scale=None, Tensor? c_kv_scale=None, Tensor? k_rope_offset=None, Tensor? c_kv_offset=None, float epsilon=1e-5, str cache_mode='Norm', bool is_output_kv=False) -> (Tensor, Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_batch_gather_matmul(Tensor input, Tensor x, Tensor weight_b, Tensor indices, Tensor? weight_a=None, int layer_idx=0, float scale=1e-3, int y_offset=0, int y_slice_size=-1) -> Tensor", tags_0);
  m.def("npu_batch_gather_matmul_(Tensor(a!) input, Tensor x, Tensor weight_b, Tensor indices, Tensor? weight_a=None, int layer_idx=0, float scale=1e-3, int y_offset=0, int y_slice_size=-1) -> Tensor(a!)", tags_0);
  m.def("npu_batch_nms(Tensor input, Tensor scores, float score_threshold, float iou_threshold, int max_size_per_class, int max_total_size, bool change_coordinate_frame=False, bool transpose_box=False) -> (Tensor, Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_bert_apply_adam(Scalar lr, Scalar beta1, Scalar beta2, Scalar epsilon, Tensor grad, Scalar max_grad_norm, Scalar global_grad_norm, Scalar weight_decay, Scalar? step_size=None, int adam_mode=0) -> (Tensor var, Tensor m, Tensor v)", tags_0);
  m.def("npu_bert_apply_adam.out(Scalar lr, Scalar beta1, Scalar beta2, Scalar epsilon, Tensor grad, Scalar max_grad_norm, Scalar global_grad_norm, Scalar weight_decay, Scalar? step_size=None, int adam_mode=0, *, Tensor[] out) -> (Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_binary_cross_entropy_with_logits_backward(Tensor grad_output, Tensor input, Tensor target, Tensor? weight_opt, Tensor? pos_weight_opt, int reduction) -> Tensor", tags_0);
  m.def("npu_bmmV2(Tensor input, Tensor mat2, int[] output_sizes) -> Tensor", tags_0);
  m.def("npu_bmm_v2_mat1_backward(Tensor grad, Tensor mat1, Tensor mat2, SymInt[] size) -> Tensor", tags_0);
  m.def("npu_bmm_v2_mat2_backward(Tensor grad, Tensor mat1, Tensor mat2, SymInt[] size) -> Tensor", tags_0);
  m.def("npu_bounding_box_decode(Tensor rois, Tensor deltas, float means0, float means1, float means2, float means3, float stds0, float stds1, float stds2, float stds3, int[1] max_shape, float wh_ratio_clip) -> Tensor", tags_0);
  m.def("npu_bounding_box_encode(Tensor anchor_box, Tensor ground_truth_box, float means0, float means1, float means2, float means3, float stds0, float stds1, float stds2, float stds3) -> Tensor", tags_0);
  m.def("npu_broadcast(Tensor input, int[] size) -> Tensor", tags_0);
  m.def("npu_broadcast.out(Tensor input, int[] size, *, Tensor(a!) out) -> Tensor(a!)", tags_0);
  m.def("npu_ciou(Tensor input, Tensor gtboxes, bool trans=False, bool is_cross=True, int mode=0, bool atan_sub_flag=False) -> Tensor", tags_0);
  m.def("npu_ciou_backward(Tensor grad, Tensor bboxes, Tensor gtboxes, Tensor? atan_sub, bool trans=False, bool is_cross=True, int mode=0) -> (Tensor, Tensor)", tags_0);
  m.def("npu_clear_float_status(Tensor input, int mode=0) -> Tensor", tags_0);
  m.def("npu_confusion_transpose(Tensor input, int[] perm, int[] shape, bool transpose_first) -> Tensor", tags_0);
  m.def("npu_confusion_transpose_backward(Tensor grad, int[] perm, SymInt[] shape, bool transpose_first) -> Tensor", tags_0);
  m.def("npu_conv2d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor", tags_0);
  m.def("npu_conv2d.out(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups, *, Tensor(a!) out) -> Tensor(a!)", tags_0);
  m.def("npu_conv2d_backward(Tensor input, Tensor grad_output, Tensor weight, int[] stride, int[] padding, int[] dilation, int groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_conv3d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor", tags_0);
  m.def("npu_conv3d.out(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups, *, Tensor(a!) out) -> Tensor(a!)", tags_0);
  m.def("npu_conv3d_backward(Tensor input, Tensor grad, Tensor weight, int[] stride, int[] padding, int[] dilation, int groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_conv_transpose2d(Tensor input, Tensor weight, Tensor? bias, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups) -> Tensor", tags_0);
  m.def("npu_conv_transpose2d_backward(Tensor input, Tensor grad_output, Tensor weight, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_conv_transpose3d_backward(Tensor input, Tensor grad_output, Tensor weight, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_convert_weight_to_int4pack(Tensor weight, int inner_k_tiles=0) -> Tensor", tags_0);
  m.def("npu_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor", tags_0);
  m.def("npu_convolution_backward(Tensor input, Tensor grad_output, Tensor weight, int[] stride, int[] padding, int[] dilation, int groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_convolution_transpose(Tensor input, Tensor weight, Tensor? bias, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups) -> Tensor", tags_0);
  m.def("npu_convolution_transpose_backward(Tensor input, Tensor grad, Tensor weight, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool[3] grad_input_mask) -> (Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_deep_norm(Tensor x, Tensor gx, Tensor beta, Tensor gamma, float alpha=0.3, float epsilon=1e-06) -> (Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_deep_norm_backward(Tensor dy, Tensor x, Tensor gx, Tensor gamma, Tensor mean, Tensor rstd, float alpha=0.3) -> (Tensor, Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_deformable_conv2d(Tensor input, Tensor weight, Tensor offset, Tensor? bias, int[2] kernel_size, int[] stride, int[] padding, int[] dilation=[1,1,1,1], int groups=1, int deformable_groups=1, bool modulated=True) -> (Tensor, Tensor)", tags_0);
  m.def("npu_deformable_conv2dbk(Tensor input, Tensor grad_output, Tensor offset_out, Tensor weight, Tensor offset, int[2] kernel_size, int[] stride, int[] padding, int[] dilation=[1,1,1,1], int groups=1, int deformable_groups=1, bool modulated=True) -> (Tensor, Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_diou(Tensor input, Tensor gtboxes, bool trans=False, bool is_cross=False, int mode=0) -> Tensor", tags_0);
  m.def("npu_diou_backward(Tensor grad, Tensor bboxes, Tensor gtboxes, bool trans=False, bool is_cross=False, int mode=0) -> (Tensor, Tensor)", tags_0);
  m.def("npu_dropout_backward(Tensor grad_output, Tensor mask, float p) -> Tensor", tags_0);
  m.def("npu_dropout_do_mask(Tensor input, Tensor mask, float p) -> (Tensor, Tensor)", tags_1);
  m.def("npu_dropout_gen_mask(int[] size, float p, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor", TORCH_FN(at_npu::native::wrapper__npu_dropout_gen_mask), tags_1);
  m.def("npu_dropout_with_add_softmax(Tensor input, Tensor x1, Scalar alpha, float prob, int dim) -> (Tensor, Tensor, Tensor)", tags_1);
  m.def("npu_dropout_with_add_softmax_backward(Tensor grad, Tensor mask, Tensor softmax_out, Scalar alpha, float prob, int dim) -> (Tensor, Tensor)", tags_0);
  m.def("npu_dtype_cast(Tensor input, ScalarType dtype) -> Tensor", tags_0);
  m.def("npu_dtype_cast_(Tensor(a!) input, Tensor src) -> Tensor(a!)", tags_0);
  m.def("npu_dtype_cast_backward(Tensor grad, ScalarType dtype) -> Tensor", tags_0);
  m.def("npu_dynamic_quant(Tensor input, *, Tensor? smooth_scales=None, Tensor? group_index=None, ScalarType? dst_type=None) -> (Tensor, Tensor)", tags_0);
  m.def("npu_dynamic_quant_asymmetric(Tensor input, *, Tensor? smooth_scales=None, Tensor? group_index=None, ScalarType? dst_type=None) -> (Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_fast_gelu(Tensor input) -> Tensor", tags_0);
  m.def("npu_fast_gelu_backward(Tensor grad, Tensor input) -> Tensor", tags_0);
  m.def("npu_ffn(Tensor x, Tensor weight1, Tensor weight2, str activation, *, int[]? expert_tokens=None, int[]? expert_tokens_index=None, Tensor? bias1=None, Tensor? bias2=None, Tensor? scale=None, Tensor? offset=None, Tensor? deq_scale1=None, Tensor? deq_scale2=None, Tensor? antiquant_scale1=None, Tensor? antiquant_scale2=None, Tensor? antiquant_offset1=None, Tensor? antiquant_offset2=None, int? inner_precise=None, ScalarType? output_dtype=None) -> Tensor", tags_0);
  m.def("npu_fused_attention_score(Tensor query_layer, Tensor key_layer, Tensor value_layer, Tensor attention_mask, Scalar scale, float keep_prob, bool query_transpose=False, bool key_transpose=False, bool bmm_score_transpose_a=False, bool bmm_score_transpose_b=False, bool value_transpose=False, bool dx_transpose=False) -> Tensor", tags_0);
  m.def("npu_fused_attention_score_backward(Tensor grad_output, Tensor softmax_output, Tensor query_layer, Tensor key_layer, Tensor value_layer, Tensor mask, Scalar scale, float keep_prob, bool query_transpose=False, bool key_transpose=False, bool value_transpose=False, bool dx_transpose=False) -> (Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_fused_attention_score_fwd(Tensor query_layer, Tensor key_layer, Tensor value_layer, Tensor attention_mask, Scalar scale, float keep_prob, bool query_transpose=False, bool key_transpose=False, bool bmm_score_transpose_a=False, bool bmm_score_transpose_b=False, bool value_transpose=False, bool dx_transpose=False) -> (Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_fused_attention_score_grad(Tensor grad_output, Tensor softmax_output, Tensor query_layer, Tensor key_layer, Tensor value_layer, Tensor mask, Scalar scale, float keep_prob, bool query_transpose=False, bool key_transpose=False, bool value_transpose=False, bool dx_transpose=False) -> (Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_fused_infer_attention_score(Tensor query, Tensor key, Tensor value, *, Tensor? pse_shift=None, Tensor? atten_mask=None, SymInt[]? actual_seq_lengths=None, SymInt[]? actual_seq_lengths_kv=None, Tensor? dequant_scale1=None, Tensor? quant_scale1=None, Tensor? dequant_scale2=None, Tensor? quant_scale2=None, Tensor? quant_offset2=None, Tensor? antiquant_scale=None, Tensor? antiquant_offset=None, Tensor? key_antiquant_scale=None, Tensor? key_antiquant_offset=None, Tensor? value_antiquant_scale=None, Tensor? value_antiquant_offset=None, Tensor? block_table=None, Tensor? query_padding_size=None, Tensor? kv_padding_size=None, Tensor? key_shared_prefix=None, Tensor? value_shared_prefix=None, SymInt[]? actual_shared_prefix_len=None, Tensor? query_rope=None, Tensor? key_rope=None, Tensor? key_rope_antiquant_scale=None, int num_heads=1, float scale=1.0, int pre_tokens=2147483647, int next_tokens=2147483647, str input_layout=\"BSH\", int num_key_value_heads=0, int sparse_mode=0, int inner_precise=0, int block_size=0, int antiquant_mode=0, int key_antiquant_mode=0, int value_antiquant_mode=0, bool softmax_lse_flag=False) -> (Tensor, Tensor)", tags_0);
  m.def("npu_fused_infer_attention_score.out(Tensor query, Tensor key, Tensor value, *, Tensor? pse_shift=None, Tensor? atten_mask=None, SymInt[]? actual_seq_lengths=None, SymInt[]? actual_seq_lengths_kv=None, Tensor? dequant_scale1=None, Tensor? quant_scale1=None, Tensor? dequant_scale2=None, Tensor? quant_scale2=None, Tensor? quant_offset2=None, Tensor? antiquant_scale=None, Tensor? antiquant_offset=None, Tensor? key_antiquant_scale=None, Tensor? key_antiquant_offset=None, Tensor? value_antiquant_scale=None, Tensor? value_antiquant_offset=None, Tensor? block_table=None, Tensor? query_padding_size=None, Tensor? kv_padding_size=None, Tensor? key_shared_prefix=None, Tensor? value_shared_prefix=None, SymInt[]? actual_shared_prefix_len=None, Tensor? query_rope=None, Tensor? key_rope=None, Tensor? key_rope_antiquant_scale=None, int num_heads=1, float scale=1.0, int pre_tokens=2147483647, int next_tokens=2147483647, str input_layout=\"BSH\", int num_key_value_heads=0, int sparse_mode=0, int inner_precise=0, int block_size=0, int antiquant_mode=0, int key_antiquant_mode=0, int value_antiquant_mode=0, bool softmax_lse_flag=False, Tensor? workspace=None, Tensor[] out) -> (Tensor, Tensor)", tags_0);
  m.def("_npu_fused_infer_attention_score_get_max_workspace(Tensor query, Tensor key, Tensor value, *, Tensor? pse_shift=None, Tensor? atten_mask=None, SymInt[]? actual_seq_lengths=None, SymInt[]? actual_seq_lengths_kv=None, Tensor? dequant_scale1=None, Tensor? quant_scale1=None, Tensor? dequant_scale2=None, Tensor? quant_scale2=None, Tensor? quant_offset2=None, Tensor? antiquant_scale=None, Tensor? antiquant_offset=None, Tensor? key_antiquant_scale=None, Tensor? key_antiquant_offset=None, Tensor? value_antiquant_scale=None, Tensor? value_antiquant_offset=None, Tensor? block_table=None, Tensor? query_padding_size=None, Tensor? kv_padding_size=None, Tensor? key_shared_prefix=None, Tensor? value_shared_prefix=None, SymInt[]? actual_shared_prefix_len=None, Tensor? query_rope=None, Tensor? key_rope=None, Tensor? key_rope_antiquant_scale=None, int num_heads=1, float scale=1.0, int pre_tokens=2147483647, int next_tokens=2147483647, str input_layout=\"BSH\", int num_key_value_heads=0, int sparse_mode=0, int inner_precise=0, int block_size=0, int antiquant_mode=0, int key_antiquant_mode=0, int value_antiquant_mode=0, bool softmax_lse_flag=False) -> Tensor", tags_0);
  m.def("npu_fusion_attention(Tensor query, Tensor key, Tensor value, int head_num, str input_layout, Tensor? pse=None, Tensor? padding_mask=None, Tensor? atten_mask=None, float scale=1., float keep_prob=1., int pre_tockens=2147483647, int next_tockens=2147483647, int inner_precise=0, int[]? prefix=None, int[]? actual_seq_qlen=None, int[]? actual_seq_kvlen=None, int sparse_mode=0, bool gen_mask_parallel=True, bool sync=False) -> (Tensor, Tensor, Tensor, Tensor, int, int, int)", tags_0);
  m.def("npu_fusion_attention_grad(Tensor query, Tensor key, Tensor value, Tensor dy, int head_num, str input_layout, *, Tensor? pse=None, Tensor? padding_mask=None, Tensor? atten_mask=None, Tensor? softmax_max=None, Tensor? softmax_sum=None, Tensor? softmax_in=None, Tensor? attention_in=None, float scale_value=1., float keep_prob=1., int pre_tockens=2147483647, int next_tockens=2147483647, int inner_precise=0, int seed=0, int offset=0, int numels=0, int[]? prefix=None, int[]? actual_seq_qlen=None, int[]? actual_seq_kvlen=None, int sparse_mode=0, bool gen_mask_parallel=True, bool sync=False) -> (Tensor, Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_fusion_attention_v2(Tensor query, Tensor key, Tensor value, int head_num, str input_layout, *, Tensor? pse=None, Tensor? padding_mask=None, Tensor? atten_mask=None, Tensor? query_rope=None, Tensor? key_rope=None, float scale=1., float keep_prob=1., int pre_tokens=2147483647, int next_tokens=2147483647, int inner_precise=0, int[]? prefix=None, int[]? actual_seq_qlen=None, int[]? actual_seq_kvlen=None, int sparse_mode=0, bool gen_mask_parallel=True, bool sync=False, int pse_type=1, int[]? q_start_idx=None, int[]? kv_start_idx=None) -> (Tensor, Tensor, Tensor, Tensor, int, int, int)", tags_0);
  m.def("npu_fusion_attention_grad_v2(Tensor query, Tensor key, Tensor value, Tensor dy, int head_num, str input_layout, *, Tensor? pse=None, Tensor? padding_mask=None, Tensor? atten_mask=None, Tensor? softmax_max=None, Tensor? softmax_sum=None, Tensor? softmax_in=None, Tensor? attention_in=None, Tensor? query_rope=None, Tensor? key_rope=None, float scale_value=1., float keep_prob=1., int pre_tokens=2147483647, int next_tokens=2147483647, int inner_precise=0, int seed=0, int offset=0, int numels=0, int[]? prefix=None, int[]? actual_seq_qlen=None, int[]? actual_seq_kvlen=None, int sparse_mode=0, bool gen_mask_parallel=True, bool sync=False, int pse_type=1, int[]? q_start_idx=None, int[]? kv_start_idx=None) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_fused_attention_layernorm_qkv_fwd(Tensor x, Tensor kernel_query, Tensor kernel_key, Tensor kernel_value, Tensor gamma, Tensor beta, Tensor? bias_query=None, Tensor? bias_key=None, Tensor? bias_value=None, int seq_len=128, int num_heads=12, float eps=1e-05) -> Tensor[]", tags_0);
  m.def("npu_fused_attention_qkv_grad(Tensor grad_output_query, Tensor grad_output_key, Tensor grad_output_value, Tensor query_kernel, Tensor key_kernel, Tensor value_kernel, Tensor hidden_states, Tensor grad_output_ln) -> Tensor[]", tags_0);
  m.def("npu_geglu(Tensor input, int dim=-1, int approximate=1, bool activate_left=False) -> (Tensor, Tensor)", tags_0);
  m.def("npu_geglu_grad(Tensor grad_output, Tensor input, Tensor gelu, int dim=-1, int approximate=1, bool activate_left=False) -> Tensor", tags_0);
  m.def("npu_get_float_status(Tensor input, int mode=0) -> Tensor", tags_0);
  m.def("npu_giou(Tensor input, Tensor gtboxes, bool trans=False, bool is_cross=False, int mode=0) -> Tensor", tags_0);
  m.def("npu_giou_backward(Tensor grad, Tensor bboxes, Tensor gtboxes, bool trans=False, bool is_cross=False, int mode=0) -> (Tensor, Tensor)", tags_0);
  m.def("npu_grid_assign_positive(Tensor input, Tensor overlaps, Tensor box_responsible_flags, Tensor max_overlaps, Tensor argmax_overlaps, Tensor gt_max_overlaps, Tensor gt_argmax_overlaps, int num_gts, float pos_iou_thr, float min_pos_iou, bool gt_max_assign_all) -> Tensor", tags_0);
  m.def("npu_group_norm_silu(Tensor input, Tensor? weight, Tensor? bias, int group, float eps=0.00001) -> (Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_grouped_matmul(Tensor[] x, Tensor[] weight, *, Tensor[]? bias=None, Tensor[]? scale=None, Tensor[]? offset=None, Tensor[]? antiquant_scale=None, Tensor[]? antiquant_offset=None, Tensor[]? per_token_scale=None, Tensor? group_list=None, Tensor[]? activation_input=None, Tensor[]? activation_quant_scale=None, Tensor[]? activation_quant_offset=None, int? split_item=0, int? group_type=None, int? group_list_type=0, int? act_type=0, int[]? tuning_config=None, ScalarType? output_dtype=None) -> Tensor[]", tags_0);
  m.def("npu_grouped_matmul.List(Tensor[] x, Tensor[] weight, *, Tensor[]? bias=None, Tensor[]? scale=None, Tensor[]? offset=None, Tensor[]? antiquant_scale=None, Tensor[]? antiquant_offset=None, Tensor[]? per_token_scale=None, int[]? group_list=None, Tensor[]? activation_input=None, Tensor[]? activation_quant_scale=None, Tensor[]? activation_quant_offset=None, int? split_item=0, int? group_type=None, int? group_list_type=0, int? act_type=0, ScalarType? output_dtype=None) -> Tensor[]", tags_0);
  m.def("npu_grouped_matmul_finalize_routing(Tensor x, Tensor w, Tensor group_list, *, Tensor? scale=None, Tensor? bias=None, Tensor? offset=None, Tensor? pertoken_scale=None, Tensor? shared_input=None, Tensor? logit=None, Tensor? row_index=None, ScalarType? dtype=None, float? shared_input_weight=1.0, int? shared_input_offset=0, int? output_bs=0, int? group_list_type=1) -> Tensor", tags_0);
  m.def("npu_gru(Tensor input, Tensor hx, Tensor weight_input, Tensor weight_hidden, Tensor bias_input, Tensor bias_hidden, Tensor seq_length, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)", tags_1);
  m.def("npu_gru_backward(Tensor? grady, Tensor? gradh, Tensor input, Tensor weight_input, Tensor weight_hidden, Tensor bias_input, Tensor bias_hidden, Tensor seq_length, Tensor hx, Tensor y_output, Tensor h_output, Tensor output_updata, Tensor output_reset, Tensor output_new, Tensor hidden_new) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_hans_encode.out(Tensor input, bool statistic=False, bool reshuff=False, *, Tensor[] out) -> (Tensor, Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_hans_decode.out(Tensor mantissa, Tensor fixed, Tensor var, Tensor pdf, bool reshuff=False, *, Tensor(a!) out) -> Tensor(a!)", tags_0);
  m.def("npu_ifmr(Tensor data, Tensor data_min, Tensor data_max, Tensor cumsum, float min_percentile, float max_percentile, float search_start, float search_end, float search_step, bool with_offset) -> (Tensor, Tensor)", tags_0);
  m.def("npu_our_incre_flash_attention(Tensor query, Tensor key, Tensor value, *, Tensor? padding_mask=None, Tensor? atten_mask=None, Tensor? pse_shift=None, SymInt[]? actual_seq_lengths=None, Tensor? antiquant_scale=None, Tensor? antiquant_offset=None, Tensor? block_table=None, Tensor? dequant_scale1=None, Tensor? quant_scale1=None, Tensor? dequant_scale2=None, Tensor? quant_scale2=None, Tensor? quant_offset2=None, Tensor? kv_padding_size=None, int num_heads=1, float scale_value=1.0, str input_layout=\"BSH\", int num_key_value_heads=0, int block_size=0, int inner_precise=1) -> Tensor", tags_0);
  m.def("npu_sparse_paged_attention(Tensor query, Tensor key, Tensor value, *, Tensor? padding_mask=None, Tensor? atten_mask=None, Tensor? pse_shift=None, SymInt[]? actual_seq_lengths=None, Tensor? antiquant_scale=None, Tensor? antiquant_offset=None, Tensor? block_table=None, Tensor? block_position=None, Tensor? dequant_scale1=None, Tensor? quant_scale1=None, Tensor? dequant_scale2=None, Tensor? quant_scale2=None, Tensor? quant_offset2=None, Tensor? kv_padding_size=None, int num_heads=1, float scale_value=1.0, str input_layout=\"BSH\", int num_key_value_heads=0, int block_size=0, int inner_precise=1) -> Tensor", tags_0);
  m.def("npu_cent_select(Tensor query, Tensor l1_cent, Tensor block_ids, Tensor block_table, Tensor seq_len) -> (Tensor, Tensor)", tags_0);
  m.def("npu_interleave_rope(Tensor x, Tensor cos, Tensor sin) -> Tensor", tags_0);
  m.def("npu_indexing(Tensor input, int[] begin, int[] end, int[] strides, int begin_mask=0, int end_mask=0, int ellipsis_mask=0, int new_axis_mask=0, int shrink_axis_mask=0) -> Tensor", tags_0);
  m.def("npu_indexing.out(Tensor input, int[] begin, int[] end, int[] strides, int begin_mask=0, int end_mask=0, int ellipsis_mask=0, int new_axis_mask=0, int shrink_axis_mask=0, *, Tensor(a!) out) -> Tensor(a!)", tags_0);
  m.def("npu_iou(Tensor bboxes, Tensor gtboxes, int mode=0) -> Tensor", tags_0);
  m.def("npu_layer_norm_eval(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05) -> Tensor", tags_0);
  m.def("npu_layernorm_grad(Tensor grad_out, Tensor input, int[] normalized_shape, Tensor mean, Tensor rstd, Tensor? weight, Tensor? bias) -> (Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor", tags_0);
  m.def("npu_linear_backward(Tensor grad, Tensor input, Tensor weight) -> (Tensor, Tensor)", tags_0);
  m.def("npu_lstm(Tensor input, Tensor weight, Tensor bias, Tensor seq_mask, Tensor h, Tensor c, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first, bool flag_seq, bool direction) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)", tags_1);
  m.def("npu_lstm_backward(Tensor? grady, Tensor? gradh, Tensor? gradc, Tensor input, Tensor weight, Tensor bias, Tensor hx, Tensor cx, Tensor y_output, Tensor h_output, Tensor c_output, Tensor i, Tensor j, Tensor f, Tensor o, Tensor tanhc) -> (Tensor, Tensor, Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_lstm_cell(Tensor input, Tensor w_ih, Tensor w_hh, Tensor h, Tensor c, Tensor? b_ih=None, Tensor? b_hh=None) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_lstm_cell_backward(Tensor? grady, Tensor? gradh, Tensor? gradc, Tensor input, Tensor w_ih, Tensor w_hh, Tensor h, Tensor c, Tensor y_output, Tensor h_output, Tensor c_output, Tensor i, Tensor j, Tensor f, Tensor o, Tensor tanhc) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_lstm_data(Tensor input, Tensor batch_sizes, Tensor weight, Tensor bias, Tensor seq_mask, Tensor h, Tensor c, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first, bool flag_seq, bool direction) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)", tags_1);
  m.def("npu_lstm_data_backward(Tensor? grady_opt, Tensor? gradh_opt, Tensor? gradc_opt, Tensor input, Tensor batch_sizes, Tensor weight, Tensor bias, Tensor init_h, Tensor init_c, Tensor y, Tensor h, Tensor c, Tensor i, Tensor j, Tensor f, Tensor o, Tensor tanhc, bool flag_direction) -> (Tensor, Tensor, Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_masked_fill_range(Tensor input, Tensor start, Tensor end, Tensor value, int axis=-1) -> Tensor", tags_0);
  m.def("npu_masked_softmax_with_rel_pos_bias(Tensor x, Tensor? atten_mask, Tensor relative_pos_bias, float scale_value=1.0, int inner_precision_mode=0) -> Tensor", tags_0);
  m.def("npu_max.dim(Tensor input, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)", tags_0);
  m.def("npu_max.names_dim(Tensor input, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)", tags_0);
  m.def("npu_max_backward(Tensor grad, int dim, Tensor indices, SymInt[] sizes, bool keepdim) -> Tensor", tags_0);
  m.def("npu_min.dim(Tensor input, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)", tags_0);
  m.def("npu_min.names_dim(Tensor input, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)", tags_0);
  m.def("npu_min_backward(Tensor grad, int dim, Tensor indices, SymInt[] sizes, bool keepdim) -> Tensor", tags_0);
  m.def("npu_mish(Tensor input) -> Tensor", tags_0);
  m.def("npu_mish_backward(Tensor grad, Tensor input) -> Tensor", tags_0);
  m.def("npu_mla_prolog(Tensor token_x, Tensor weight_dq, Tensor weight_uq_qr, Tensor weight_uk, Tensor weight_dkv_kr, Tensor rmsnorm_gamma_cq, Tensor rmsnorm_gamma_ckv, Tensor rope_sin, Tensor rope_cos, Tensor cache_index, Tensor kv_cache, Tensor kr_cache, *, Tensor? dequant_scale_x=None, Tensor? dequant_scale_w_dq=None, Tensor? dequant_scale_w_uq_qr=None, Tensor? dequant_scale_w_dkv_kr=None, Tensor? quant_scale_ckv=None, Tensor? quant_scale_ckr=None, Tensor? smooth_scales_cq=None, float rmsnorm_epsilon_cq=1e-05, float rmsnorm_epsilon_ckv=1e-05, str cache_mode=\"PA_BSND\") -> (Tensor, Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_mla_prolog_v2(Tensor token_x, Tensor weight_dq, Tensor weight_uq_qr, Tensor weight_uk, Tensor weight_dkv_kr, Tensor rmsnorm_gamma_cq, Tensor rmsnorm_gamma_ckv, Tensor rope_sin, Tensor rope_cos, Tensor cache_index, Tensor kv_cache, Tensor kr_cache, *, Tensor? dequant_scale_x=None, Tensor? dequant_scale_w_dq=None, Tensor? dequant_scale_w_uq_qr=None, Tensor? dequant_scale_w_dkv_kr=None, Tensor? quant_scale_ckv=None, Tensor? quant_scale_ckr=None, Tensor? smooth_scales_cq=None, float rmsnorm_epsilon_cq=1e-05, float rmsnorm_epsilon_ckv=1e-05, str cache_mode=\"PA_BSND\") -> (Tensor, Tensor, Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_mm_all_reduce_base(Tensor x1, Tensor x2, str hcom, *, str reduce_op='sum', Tensor? bias=None, Tensor? antiquant_scale=None, Tensor? antiquant_offset=None, Tensor? x3=None, Tensor? dequant_scale=None, Tensor? pertoken_scale=None, Tensor? comm_quant_scale_1=None, Tensor? comm_quant_scale_2=None, int antiquant_group_size=0, int comm_turn=0) -> Tensor", tags_0);
  m.def("npu_mm_reduce_scatter_base(Tensor input, Tensor x2, str hcom, int world_size, *, str reduce_op='sum', Tensor? bias=None, int comm_turn=0) -> Tensor", tags_0);
  m.def("npu_moe_compute_expert_tokens(Tensor sorted_expert_for_source_row, int num_expert) -> Tensor", tags_0);
  m.def("npu_moe_finalize_routing(Tensor expanded_permuted_rows, Tensor? skip1, Tensor? skip2, Tensor? bias, Tensor? scales, Tensor expanded_src_to_dst_row, Tensor? export_for_source_row, int? drop_pad_mode=0) -> Tensor", tags_0);
  m.def("npu_moe_gating_top_k_softmax(Tensor x, Tensor? finished=None, int k=1) -> (Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_moe_gating_top_k(Tensor x, int k, *, Tensor? bias=None, int k_group=1, int group_count=1, int group_select_mode=0, int renorm=0, int norm_type=0, bool out_flag=False, float routed_scaling_factor=1.0, float eps=1e-20) -> (Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_moe_init_routing(Tensor x, Tensor row_idx, Tensor expert_idx, int active_num) -> (Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_moe_init_routing_v2(Tensor x, Tensor expert_idx, *, Tensor? scale=None, Tensor? offset=None, int active_num=-1, int expert_capacity=-1, int expert_num=-1, int drop_pad_mode=0, int expert_tokens_num_type=0, bool expert_tokens_num_flag=False, int quant_mode=0, int[2] active_expert_range=[], int row_idx_type=0) -> (Tensor, Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_grouped_matmul_swiglu_quant(Tensor x, Tensor weight, Tensor group_list, Tensor weight_scale, Tensor x_scale, *, Tensor? bias=None, Tensor? offset=None) -> (Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_moe_distribute_dispatch(Tensor x, Tensor expert_ids, str group_ep, int ep_world_size, int ep_rank_id, int moe_expert_num, *, Tensor? scales=None, Tensor? x_active_mask=None, Tensor? expert_scales=None, str group_tp=\"\", int tp_world_size=0, int tp_rank_id=0, int expert_shard_type=0, int shared_expert_num=1, int shared_expert_rank_num=0, int quant_mode=0, int global_bs=0, int expert_token_nums_type=1) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_moe_distribute_dispatch_v2(Tensor x, Tensor expert_ids, str group_ep, int ep_world_size, int ep_rank_id, int moe_expert_num, *, Tensor? scales=None, Tensor? x_active_mask=None, Tensor? expert_scales=None, str group_tp=\"\", int tp_world_size=0, int tp_rank_id=0, int expert_shard_type=0, int shared_expert_num=1, int shared_expert_rank_num=0, int quant_mode=0, int global_bs=0, int expert_token_nums_type=1) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_moe_distribute_combine(Tensor expand_x, Tensor expert_ids, Tensor expand_idx, Tensor ep_send_counts, Tensor expert_scales, str group_ep, int ep_world_size, int ep_rank_id, int moe_expert_num, *, Tensor? tp_send_counts=None, Tensor? x_active_mask=None, Tensor? activation_scale=None, Tensor? weight_scale=None, Tensor? group_list=None, Tensor? expand_scales=None, Tensor? shared_expert_x=None, str group_tp=\"\", int tp_world_size=0, int tp_rank_id=0, int expert_shard_type=0, int shared_expert_num=1, int shared_expert_rank_num=0, int global_bs=0, int out_dtype=0, int comm_quant_mode=0, int group_list_type=0) -> Tensor", tags_0);
  m.def("npu_moe_distribute_combine_v2(Tensor expand_x, Tensor expert_ids, Tensor assist_info_for_combine, Tensor ep_send_counts, Tensor expert_scales, str group_ep, int ep_world_size, int ep_rank_id, int moe_expert_num, *, Tensor? tp_send_counts=None, Tensor? x_active_mask=None, Tensor? expand_scales=None, Tensor? shared_expert_x=None, str group_tp=\"\", int tp_world_size=0, int tp_rank_id=0, int expert_shard_type=0, int shared_expert_num=1, int shared_expert_rank_num=0, int global_bs=0, int comm_quant_mode=0) -> Tensor", tags_0);
  m.def("_npu_distribute_barrier(Tensor x_ref, str group, int world_size) -> Tensor", tags_0);
  m.def("npu_moe_eplb_update_expert(Tensor expert_ids, Tensor eplb_table, int local_rank_id, int world_size, *, int balance_mode=0) -> Tensor", tags_0);
  m.def("npu_moe_re_routing(Tensor tokens, Tensor expert_token_num_per_rank, *, Tensor? per_token_scales=None, int expert_token_num_type=1, int idx_type=0) -> (Tensor, Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_moe_distribute_combine_add_rms_norm(Tensor expand_x, Tensor expert_ids, Tensor expand_idx, Tensor ep_send_counts, Tensor expert_scales, Tensor residual_x, Tensor gamma, str group_ep, int ep_world_size, int ep_rank_id, int moe_expert_num, *, Tensor? tp_send_counts=None, Tensor? x_active_mask=None, Tensor? activation_scale=None, Tensor? weight_scale=None, Tensor? group_list=None, Tensor? expand_scales=None, Tensor? shared_expert_x=None, str group_tp=\"\", int tp_world_size=0, int tp_rank_id=0, int expert_shard_type=0, int shared_expert_num=1, int shared_expert_rank_num=0, int global_bs=0, int out_dtype=0, int comm_quant_mode=0, int group_list_type=0, str comm_alg=\"\", float norm_eps=1e-06) -> (Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_multi_head_attention(Tensor query, Tensor key, Tensor value, Tensor query_weight, Tensor key_weight, Tensor value_weight, Tensor attn_mask, Tensor out_proj_weight, Tensor? query_bias, Tensor? key_bias, Tensor? value_bias, Tensor? out_proj_bias, Tensor? dropout_mask, int attn_head_num, int attn_dim_per_head, int src_len, int tgt_len, float dropout_prob, bool softmax_use_float) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)", tags_1);
  m.def("npu_multi_head_attention_backward(Tensor query, Tensor key, Tensor value, Tensor query_weight, Tensor key_weight, Tensor value_weight, Tensor out_proj_weight, Tensor? query_bias, Tensor? key_bias, Tensor? value_bias, Tensor? out_proj_bias, Tensor query_res, Tensor key_res, Tensor value_res, Tensor attn_scores, Tensor attn_res, Tensor context, Tensor y_grad, Tensor dropout_mask, int attn_head_num, int attn_dim_per_head, int src_len, int tgt_len, float dropout_prob, bool softmax_use_float) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_multi_head_attention_v2(Tensor query, Tensor key, Tensor value, Tensor? atten_mask=None, Tensor? alibi_mask=None, float scale=1.0, int head_num=1, str input_layout=\"BNSD\", float keep_prob=1., int pre_tokens=2147483647, int next_tokens=1, bool gen_mask_parallel=True, bool sync=False) -> (Tensor, Tensor, int, int, int)", tags_0);
  m.def("npu_multi_head_attention_v2_grad(Tensor attention_score_grad, Tensor query, Tensor key, Tensor value, Tensor softmax_log_max_sum, Tensor attention_score, Tensor? atten_mask=None, Tensor? alibi_mask=None, float scale=1.0, int head_num=1, str input_layout=\"BNSD\", float keep_prob=1., int pre_tokens=2147483647, int next_tokens=1, int seed=0, int offset=0, int numels=0, bool gen_mask_parallel=True, bool sync=False) -> (Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_nms_rotated(Tensor input, Tensor scores, float iou_threshold, float scores_threshold=0, int max_output_size=-1, int mode=0) -> (Tensor, Tensor)", tags_0);
  m.def("npu_nms_v4(Tensor input, Tensor scores, Scalar max_output_size, Tensor iou_threshold, Tensor scores_threshold, bool pad_to_max_output_size=False) -> (Tensor, Tensor)", tags_0);
  m.def("npu_nms_with_mask(Tensor input, Scalar iou_threshold) -> (Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_normalize_batch(Tensor input, Tensor seq_len, int normalize_type=0) -> Tensor", tags_0);
  m.def("npu_one_hot(Tensor input, int num_classes=-1, int depth=1, Scalar on_value=1, Scalar off_value=0) -> Tensor", tags_0);
  m.def("npu_pad(Tensor input, int[] paddings) -> Tensor", tags_0);
  m.def("npu_prompt_flash_attention(Tensor query, Tensor key, Tensor value, *, Tensor? padding_mask=None, Tensor? atten_mask=None, Tensor? pse_shift=None, int[]? actual_seq_lengths=None, Tensor? deq_scale1=None, Tensor? quant_scale1=None, Tensor? deq_scale2=None, Tensor? quant_scale2=None, Tensor? quant_offset2=None, int num_heads=1, float scale_value=1.0, int pre_tokens=2147473647, int next_tokens=0, str input_layout=\"BSH\", int num_key_value_heads=0, int[]? actual_seq_lengths_kv=None, int sparse_mode=0) -> Tensor", tags_0);
  m.def("npu_ps_roi_pooling(Tensor input, Tensor rois, float spatial_scale, int group_size, int output_dim) -> Tensor", tags_0);
  m.def("npu_ps_roi_pooling_backward(Tensor output_grad, Tensor rois, float spatial_scale, int group_size, int output_dim, SymInt[] input_size) -> Tensor", tags_0);
  m.def("npu_ptiou(Tensor bboxes, Tensor gtboxes, int mode=0) -> Tensor", tags_0);
  m.def("npu_prefetch(Tensor input, Tensor? dependency, int max_size, int offset=0) -> ()", tags_0);
  m.def("npu_quant_conv2d(Tensor input, Tensor weight, Tensor scale, int[2] strides=1, int[2] pads=0, int[2] dilations=1, int groups=1, int offset_x=0, str round_mode='rint', ScalarType? output_dtype=None, Tensor? bias=None, Tensor? offset=None) -> Tensor", tags_0);
  m.def("npu_quant_matmul(Tensor x1, Tensor x2, Tensor scale, *, Tensor? offset=None, Tensor? pertoken_scale=None, Tensor? bias=None, ScalarType? output_dtype=None, SymInt[]? group_sizes=None) -> Tensor", tags_0);
  m.def("npu_quant_matmul_dequant(Tensor x, Tensor quantized_weight, Tensor weight_scale, *, Tensor? bias=None, Tensor? x_scale=None, Tensor? x_offset=None, Tensor? smooth_scale=None, str? quant_mode='pertoken') -> Tensor", tags_0);
  m.def("npu_quant_grouped_matmul_dequant(Tensor x, Tensor quantized_weight, Tensor weight_scale, Tensor group_list, *, Tensor? bias=None, Tensor? x_scale=None, Tensor? x_offset=None, Tensor? smooth_scale=None, str? quant_mode='pertoken') -> Tensor", tags_0);
  m.def("npu_quant_scatter(Tensor input, Tensor indices, Tensor updates, Tensor quant_scales, Tensor? quant_zero_points=None, int axis=0, int quant_axis=1, str reduce='update') -> Tensor", tags_0);
  m.def("npu_quant_scatter_(Tensor(a!) input, Tensor indices, Tensor updates, Tensor quant_scales, Tensor? quant_zero_points=None, int axis=0, int quant_axis=1, str reduce='update') -> Tensor(a!)", tags_0);
  m.def("npu_quantize(Tensor input, Tensor scales, Tensor? zero_points, ScalarType dtype, int axis=1, bool div_mode=True) -> Tensor", tags_0);
  m.def("npu_kronecker_quant(Tensor x, Tensor kronecker_p1, Tensor kronecker_p2, float? clip_ratio=None, ScalarType? dst_dtype=None) -> (Tensor out, Tensor quant_scale)", tags_0);
  m.def("npu_group_quant(Tensor x, Tensor scale, Tensor group_index, *, Tensor? offset=None, ScalarType? dst_dtype=None) -> Tensor", tags_0);
  m.def("npu_random_choice_with_mask(Tensor x, int count=256, int seed=0, int seed2=0) -> (Tensor, Tensor)", tags_1);
  m.def("npu_reshape(Tensor input, int[] shape, bool can_refresh=False) -> Tensor", tags_0);
  m.def("npu_reshape.out(Tensor input, int[] shape, bool can_refresh=False, *, Tensor(a!) out) -> Tensor(a!)", tags_0);
  m.def("npu_rms_norm(Tensor input, Tensor gamma, float epsilon=1e-06) -> (Tensor, Tensor)", tags_0);
  m.def("npu_gemma_rms_norm(Tensor input, Tensor gamma, float epsilon=1e-06) -> (Tensor, Tensor)", tags_0);
  m.def("npu_rms_norm_backward(Tensor dy, Tensor input, Tensor gamma, Tensor rstd) -> (Tensor, Tensor)", tags_0);
  m.def("npu_roi_align(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width, int sample_num, int roi_end_mode) -> Tensor", tags_0);
  m.def("npu_roi_alignbk(Tensor input, Tensor rois, int[] xdiff_shape, int pooled_width, int pooled_height, float spatial_scale, int sample_num, int? roi_end_mode=None) -> Tensor", tags_0);
  m.def("npu_rotary_mul(Tensor input, Tensor r1, Tensor r2, str rotary_mode='half') -> Tensor", tags_0);
  m.def("npu_rotary_mul_backward(Tensor grad, Tensor input, Tensor r1, Tensor r2, str rotary_mode='half') -> (Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_mrope(Tensor positions, Tensor query, Tensor key, Tensor cos_sin_cache, int head_size, *, int[]? mrope_section=None, str? rotary_mode='half') -> (Tensor, Tensor)", tags_0);
  m.def("npu_rotated_box_decode(Tensor input, Tensor deltas, Tensor weight) -> Tensor", tags_0);
  m.def("npu_rotated_box_encode(Tensor input, Tensor gt_bboxes, Tensor weight) -> Tensor", tags_0);
  m.def("npu_rotated_iou(Tensor input, Tensor query_boxes, bool trans=False, int mode=0, bool is_cross=True, float v_threshold=0.0, float e_threshold=0.0) -> Tensor", tags_0);
  m.def("npu_rotated_overlaps(Tensor input, Tensor query_boxes, bool trans=False) -> Tensor", tags_0);
  m.def("npu_scaled_masked_softmax(Tensor x, Tensor mask, Scalar scale=1, bool fixed_triu_mask=False) -> Tensor", tags_0);
  m.def("npu_scaled_masked_softmax_backward(Tensor y_grad, Tensor y, Tensor mask, Scalar scale, bool fixed_triu_mask) -> Tensor", tags_0);
  m.def("npu_scatter(Tensor input, Tensor indices, Tensor updates, int dim) -> Tensor", tags_0);
  m.def("npu_scatter_list(Tensor[] input, Tensor indices, Tensor updates, Tensor? mask=None, str reduce='update', int axis=-2) -> Tensor[]", tags_0);
  m.def("npu_scatter_list_(Tensor(a!)[] input, Tensor indices, Tensor updates, Tensor? mask=None, str reduce='update', int axis=-2) -> ()", tags_0);
  m.def("npu_scatter_nd_update(Tensor input, Tensor indices, Tensor updates) -> Tensor", tags_0);
  m.def("npu_scatter_nd_update_(Tensor(a!) input, Tensor indices, Tensor updates) -> Tensor(a!)", tags_0);
  m.def("npu_sign_bits_pack(Tensor input, int size) -> Tensor", tags_0);
  m.def("npu_sign_bits_unpack(Tensor input, int size, ScalarType dtype) -> Tensor", tags_0);
  m.def("npu_silu(Tensor input) -> Tensor", tags_0);
  m.def("npu_silu_(Tensor(a!) input) -> Tensor(a!)", tags_0);
  m.def("npu_silu_backward(Tensor grad_output, Tensor x0, Tensor x1) -> Tensor", tags_0);
  m.def("npu_slice(Tensor input, int[] offsets, int[] size) -> Tensor", tags_0);
  m.def("npu_slice.out(Tensor input, int[] offsets, int[] size, *, Tensor(a!) out) -> Tensor(a!)", tags_0);
  m.def("npu_softmax_cross_entropy_with_logits(Tensor input, Tensor labels) -> Tensor", tags_0);
  m.def("npu_softmax_cross_entropy_with_logits_backward(Tensor grad, Tensor input, Tensor labels) -> Tensor", tags_0);
  m.def("npu_sort_v2(Tensor input, int dim=-1, bool descending=False) -> Tensor", tags_0);
  m.def("npu_sort_v2.out(Tensor input, int dim=-1, bool descending=False, *, Tensor(a!) out) -> Tensor(a!)", tags_0);
  m.def("npu_stride_add(Tensor input, Tensor other, Scalar offset1, Scalar offset2, Scalar c1_len) -> Tensor", tags_0);
  m.def("npu_stride_copy(Tensor input, int[] shape, int[] stride, Scalar storage_offset) -> Tensor", tags_0);
  m.def("npu_stride_copy.out(Tensor input, int[] shape, int[] stride, Scalar storage_offset, *, Tensor(a!) out) -> Tensor(a!)", tags_0);
  m.def("npu_sub_sample(Tensor input, int per_images, float positive_fraction) -> Tensor", tags_0);
  m.def("npu_swiglu(Tensor input, int dim=-1) -> Tensor", tags_0);
  m.def("npu_swiglu_backward(Tensor grad_output, Tensor input, int dim=-1) -> Tensor", tags_0);
  m.def("npu_dequant_swiglu_quant(Tensor x, *, Tensor? weight_scale=None, Tensor? activation_scale=None, Tensor? bias=None, Tensor? quant_scale=None, Tensor? quant_offset=None, Tensor? group_index=None, bool activate_left=False, int quant_mode=0) -> (Tensor, Tensor)", tags_0);
  m.def("npu_dequant_rope_quant_kvcache(Tensor x, Tensor cos, Tensor sin, Tensor k_cache, Tensor v_cache, Tensor indices, Tensor scale_k, Tensor scale_v, int[3] size_splits, *, Tensor? offset_k=None, Tensor? offset_v=None, Tensor? weight_scale=None, Tensor? activation_scale=None, Tensor? bias=None, int quant_mode=0, str input_layout=\"BSND\", bool kv_output=False, str cache_mode=\"contiguous\") -> (Tensor, Tensor, Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_rope_quant_kvcache(Tensor x, Tensor cos, Tensor sin, Tensor k_cache, Tensor v_cache, Tensor indices, Tensor scale_k, Tensor scale_v, int[3] size_splits, *, Tensor? offset_k=None, Tensor? offset_v=None, int quant_mode=0, str input_layout=\"BSND\", bool kv_output=False, str cache_mode=\"contiguous\") -> (Tensor, Tensor, Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_dequant_bias(Tensor x, Tensor weight_scale, Tensor? activation_scale, Tensor? bias, *, ScalarType? output_dtype=None) -> Tensor", tags_0);
  m.def("npu_trans_quant_param(Tensor scale, Tensor? offset=None, int? round_mode=0) -> Tensor", tags_0);
  m.def("npu_transpose(Tensor input, int[] perm, bool require_contiguous=True) -> Tensor", tags_0);
  m.def("npu_transpose.out(Tensor input, int[] perm, bool require_contiguous=True, *, Tensor(a!) out) -> Tensor(a!)", tags_0);
  m.def("npu_view_copy(Tensor(a!) input, Tensor other, bool non_blocking) -> Tensor(a!)", tags_0);
  m.def("npu_weight_quant_batchmatmul(Tensor x, Tensor weight, Tensor antiquant_scale, Tensor? antiquant_offset=None, Tensor? quant_scale=None, Tensor? quant_offset=None, Tensor? bias=None, int antiquant_group_size=0, int inner_precise=0) -> Tensor", tags_0);
  m.def("npu_transpose_batchmatmul(Tensor input, Tensor weight, *, Tensor? bias=None, Tensor? scale=None, int[]? perm_x1=None, int[]? perm_x2=None, int[]? perm_y=None, int? batch_split_factor=1) -> Tensor", tags_0);
  m.def("npu_yolo_boxes_encode(Tensor input, Tensor gt_bboxes, Tensor stride, bool performance_mode=False) -> Tensor", tags_0);
  m.def("one_(Tensor(a!) input) -> Tensor(a!)", tags_0);
  m.def("repeat_interleave_backward_int(Tensor grad, Tensor input, SymInt repeats, int? dim=None) -> Tensor", tags_0);
  m.def("repeat_interleave_backward_tensor(Tensor grad, Tensor input, Tensor repeats, int? dim=None) -> Tensor", tags_0);
  m.def("scatter_update(Tensor input, Tensor indices, Tensor updates, int axis) -> Tensor", tags_0);
  m.def("scatter_update_(Tensor(a!) input, Tensor indices, Tensor updates, int axis) -> Tensor(a!)", tags_0);
  m.def("slow_conv_transpose2d_backward(Tensor grad_output, Tensor input, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] output_padding, int[2] dilation, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)", tags_0);
  m.def("stft_backward(Tensor grad_output, Tensor input, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool normalized=False, bool? onesided=None, bool? return_complex=None) -> Tensor", tags_0);
  m.def("fft_r2c_backward(Tensor grad, int[] dim, int normalization, bool onesided, int last_dim_size) -> Tensor", tags_0);
  m.def("fft_c2r_backward(Tensor grad, int[] dim, int normalization) -> Tensor", tags_0);
  m.def("npu_cross_entropy_loss(Tensor input, Tensor target, Tensor? weight=None, str reduction='mean', int ignore_index=-100, float label_smoothing=0.0, float lse_square_scale_for_zloss=0.0, bool return_zloss=False) -> (Tensor, Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_cross_entropy_loss_backward(Tensor grad_loss, Tensor log_prob, Tensor target, Tensor? weight=None, Tensor? grad_zloss=None, Tensor? lse_for_zloss=None, str reduction='mean', int ignore_index=-100, float label_smoothing=0.0, float lse_square_scale_for_zloss=0.0) -> Tensor", tags_0);
  m.def("npu_group_norm_swish(Tensor input, int num_groups, Tensor weight, Tensor bias, float? eps=1e-5, float? swish_scale=1.0) -> (Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_group_norm_swish_grad(Tensor grad, Tensor input, int num_groups, Tensor weight, Tensor bias, Tensor mean, Tensor rstd, bool[3] grad_input_mask, float? swish_scale=1.0) -> (Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_advance_step_flashattn(Tensor(a!) input_tokens, Tensor sampled_token_ids, Tensor(b!) input_positions, Tensor(c!) seq_lens, Tensor(d!) slot_mapping, Tensor block_tables, int num_seqs, int num_queries, int block_size) -> ()", tags_0);
  m.def("npu_grouped_matmul_add_(Tensor(a!) input, Tensor x, Tensor weight, Tensor group_list, *, bool transpose_x=True, bool transpose_weight=False, int group_type=2) -> Tensor(a!)", tags_0);
  m.def("npu_attn_softmax_(Tensor(a!) input) -> Tensor(a!)", tags_0);
  m.def("npu_attn_softmax_backward_(Tensor(a!) input, Tensor grad_output, Tensor values) -> Tensor(a!)", tags_0);
  m.def("npu_gather_sparse_index(Tensor input, Tensor index) -> Tensor", tags_0);
  m.def("npu_nsa_compress(Tensor input, Tensor weight, int compress_block_size, int compress_stride, *, int[]? actual_seq_len=None) -> Tensor", tags_0);
  m.def("npu_nsa_compress_grad(Tensor grad, Tensor input, Tensor weight, int compress_block_size, int compress_stride, *, int[]? actual_seq_len=None) -> (Tensor, Tensor)", tags_0);
  m.def("npu_nsa_compress_infer.cache(Tensor input, Tensor weight, Tensor slot_mapping, int compress_block_size, int compress_stride, int page_block_size, *, Tensor? block_table=None, int[]? actual_seq_len=None, Tensor(a!) cache) -> Tensor(a!)", tags_0);
  m.def("npu_nsa_compress_attention(Tensor query, Tensor key, Tensor value, float scale_value, int head_num, int compress_block_size, int compress_stride, int select_block_size, int select_block_count, *, Tensor? topk_mask=None, Tensor? atten_mask=None, int[]? actual_seq_qlen=None, int[]? actual_cmp_seq_kvlen=None, int[]? actual_sel_seq_kvlen=None) -> (Tensor, Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_nsa_compress_attention_infer(Tensor query, Tensor key, Tensor value, float scale_value, int head_num, int key_value_head_num, int select_block_size, int select_block_count, int page_block_size, int compress_block_size, int compress_stride, *, Tensor? atten_mask=None, Tensor? block_table=None, Tensor? topk_mask=None, int[]? actual_seq_qlen=None, int[]? actual_cmp_seq_kvlen=None, int[]? actual_sel_seq_kvlen=None) -> (Tensor, Tensor)", tags_0);
  m.def("npu_nsa_select_attention(Tensor query, Tensor key, Tensor value, Tensor topk_indices, float scale_value, int head_num, int select_block_size, int select_block_count, *, Tensor? atten_mask=None, int[]? actual_seq_qlen=None, int[]? actual_seq_kvlen=None) -> (Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_nsa_select_attention_grad(Tensor grad, Tensor query, Tensor key, Tensor value, Tensor attention_out, Tensor softmax_max, Tensor softmax_sum, Tensor topk_indices, float scale_value, int head_num, int select_block_size, int select_block_count, *, Tensor? atten_mask=None, int[]? actual_seq_qlen=None, int[]? actual_seq_kvlen=None) -> (Tensor, Tensor, Tensor)", tags_0);
  m.def("npu_nsa_select_attention_infer(Tensor query, Tensor key, Tensor value, Tensor topk_indices, float scale_value, int head_num, int key_value_head_num, int select_block_size, int select_block_count, int page_block_size, *, str layout='BSND', Tensor? atten_mask=None, Tensor? block_table=None, int[]? actual_seq_qlen=None, int[]? actual_seq_kvlen=None) -> Tensor", tags_0);
  m.def("npu_top_k_top_p(Tensor logits, Tensor p, Tensor k) -> Tensor", tags_0);
  m.def("npu_moe_token_permute(Tensor tokens, Tensor indices, int? num_out_tokens=None, bool padded_mode=False) -> (Tensor, Tensor)", tags_0);
  m.def("npu_moe_token_unpermute(Tensor permuted_tokens, Tensor sorted_indices, Tensor? probs=None, bool padded_mode=False, int[]? restore_shape=None) -> Tensor", tags_0);
}

} // anonymous namespace

namespace {

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {

  m.impl("npu_change_data_ptr", TORCH_FN(at_npu::native::wrapper__npu_change_data_ptr));
  m.impl("get_npu_format", TORCH_FN(at_npu::native::wrapper__get_npu_format));
  m.impl("npu_format_cast.Tensor", TORCH_FN(at_npu::native::wrapper_Tensor_npu_format_cast));
  m.impl("npu_format_cast_.acl_format", TORCH_FN(at_npu::native::wrapper_acl_format_npu_format_cast_));
  m.impl("npu_format_cast_", TORCH_FN(at_npu::native::wrapper__npu_format_cast_));
  m.impl("copy_memory_", TORCH_FN(at_npu::native::wrapper__copy_memory_));
  m.impl("get_storage_size", TORCH_FN(at_npu::native::wrapper__get_storage_size));
  m.impl("npu_format_cast", TORCH_FN(at_npu::native::wrapper__npu_format_cast));
  m.impl("_npu_format_cast", TORCH_FN(at_npu::native::wrapper___npu_format_cast));
  m.impl("npu_gather_backward", TORCH_FN(at_npu::native::wrapper__npu_gather_backward));
  m.impl("_amp_foreach_non_finite_check", TORCH_FN(at_npu::native::wrapper___amp_foreach_non_finite_check));
  m.impl("npu_gelu", TORCH_FN(at_npu::native::wrapper__npu_gelu));
  m.impl("npu_gelu_backward", TORCH_FN(at_npu::native::wrapper__npu_gelu_backward));
  m.impl("_conv_depthwise2d_backward", TORCH_FN(at_npu::native::wrapper___conv_depthwise2d_backward));
  m.impl("_dropout_with_byte_mask", TORCH_FN(at_npu::native::wrapper___dropout_with_byte_mask));
  m.impl("_dropout_with_byte_mask_backward", TORCH_FN(at_npu::native::wrapper___dropout_with_byte_mask_backward));
  m.impl("_npu_ciou", TORCH_FN(at_npu::native::wrapper___npu_ciou));
  m.impl("_npu_dropout", TORCH_FN(at_npu::native::wrapper___npu_dropout));
  m.impl("_npu_dropout_gen_mask.Tensor", TORCH_FN(at_npu::native::wrapper_Tensor__npu_dropout_gen_mask));
  m.impl("_npu_silent_check", TORCH_FN(at_npu::native::wrapper___npu_silent_check));
  m.impl("_npu_silent_check_v2", TORCH_FN(at_npu::native::wrapper___npu_silent_check_v2));
  m.impl("_npu_silent_check_v3", TORCH_FN(at_npu::native::wrapper___npu_silent_check_v3));
  m.impl("batch_norm_gather_stats_update", TORCH_FN(at_npu::native::wrapper__batch_norm_gather_stats_update));
  m.impl("batch_norm_reduce", TORCH_FN(at_npu::native::wrapper__batch_norm_reduce));
  m.impl("dropout_with_byte_mask", TORCH_FN(at_npu::native::wrapper__dropout_with_byte_mask));
  m.impl("fast_gelu", TORCH_FN(at_npu::native::wrapper__fast_gelu));
  m.impl("kl_div_backward", TORCH_FN(at_npu::native::wrapper__kl_div_backward));
  m.impl("l1_loss_backward", TORCH_FN(at_npu::native::wrapper__l1_loss_backward));
  m.impl("slow_conv_dilated2d_backward", TORCH_FN(at_npu::native::wrapper__slow_conv_dilated2d_backward));
  m.impl("matmul_double_backward", TORCH_FN(at_npu::native::wrapper__matmul_double_backward));
  m.impl("npu_add_layer_norm", TORCH_FN(at_npu::native::wrapper__npu_add_layer_norm));
  m.impl("npu_add_layer_norm_backward", TORCH_FN(at_npu::native::wrapper__npu_add_layer_norm_backward));
  m.impl("npu_add_rms_norm", TORCH_FN(at_npu::native::wrapper__npu_add_rms_norm));
  m.impl("npu_add_rms_norm_cast", TORCH_FN(at_npu::native::wrapper__npu_add_rms_norm_cast));
  m.impl("npu_add_rms_norm_quant", TORCH_FN(at_npu::native::wrapper__npu_add_rms_norm_quant));
  m.impl("npu_alltoallv_gmm", TORCH_FN(at_npu::native::wrapper__npu_alltoallv_gmm));
  m.impl("npu_gmm_alltoallv", TORCH_FN(at_npu::native::wrapper__npu_gmm_alltoallv));
  m.impl("npu_all_gather_base_mm", TORCH_FN(at_npu::native::wrapper__npu_all_gather_base_mm));
  m.impl("npu_alloc_float_status", TORCH_FN(at_npu::native::wrapper__npu_alloc_float_status));
  m.impl("npu_anchor_response_flags", TORCH_FN(at_npu::native::wrapper__npu_anchor_response_flags));
  m.impl("npu_anti_quant", TORCH_FN(at_npu::native::wrapper__npu_anti_quant));
  m.impl("npu_apply_adam", TORCH_FN(at_npu::native::wrapper__npu_apply_adam));
  m.impl("npu_apply_adam.out", TORCH_FN(at_npu::native::wrapper_out_npu_apply_adam_out));
  m.impl("npu_apply_adam_w", TORCH_FN(at_npu::native::wrapper__npu_apply_adam_w));
  m.impl("npu_apply_adam_w.out", TORCH_FN(at_npu::native::wrapper_out_npu_apply_adam_w_out));
  m.impl("npu_apply_rotary_pos_emb", TORCH_FN(at_npu::native::wrapper__npu_apply_rotary_pos_emb));
  m.impl("npu_kv_rmsnorm_rope_cache", TORCH_FN(at_npu::native::wrapper__npu_kv_rmsnorm_rope_cache));
  m.impl("npu_batch_gather_matmul", TORCH_FN(at_npu::native::wrapper__npu_batch_gather_matmul));
  m.impl("npu_batch_gather_matmul_", TORCH_FN(at_npu::native::wrapper__npu_batch_gather_matmul_));
  m.impl("npu_batch_nms", TORCH_FN(at_npu::native::wrapper__npu_batch_nms));
  m.impl("npu_bert_apply_adam", TORCH_FN(at_npu::native::wrapper__npu_bert_apply_adam));
  m.impl("npu_bert_apply_adam.out", TORCH_FN(at_npu::native::wrapper_out_npu_bert_apply_adam_out));
  m.impl("npu_binary_cross_entropy_with_logits_backward", TORCH_FN(at_npu::native::wrapper__npu_binary_cross_entropy_with_logits_backward));
  m.impl("npu_bmmV2", TORCH_FN(at_npu::native::wrapper__npu_bmmV2));
  m.impl("npu_bmm_v2_mat1_backward", TORCH_FN(at_npu::native::wrapper__npu_bmm_v2_mat1_backward));
  m.impl("npu_bmm_v2_mat2_backward", TORCH_FN(at_npu::native::wrapper__npu_bmm_v2_mat2_backward));
  m.impl("npu_bounding_box_decode", TORCH_FN(at_npu::native::wrapper__npu_bounding_box_decode));
  m.impl("npu_bounding_box_encode", TORCH_FN(at_npu::native::wrapper__npu_bounding_box_encode));
  m.impl("npu_broadcast", TORCH_FN(at_npu::native::wrapper__npu_broadcast));
  m.impl("npu_broadcast.out", TORCH_FN(at_npu::native::wrapper_out_npu_broadcast_out));
  m.impl("npu_ciou", TORCH_FN(at_npu::native::wrapper__npu_ciou));
  m.impl("npu_ciou_backward", TORCH_FN(at_npu::native::wrapper__npu_ciou_backward));
  m.impl("npu_clear_float_status", TORCH_FN(at_npu::native::wrapper__npu_clear_float_status));
  m.impl("npu_confusion_transpose", TORCH_FN(at_npu::native::wrapper__npu_confusion_transpose));
  m.impl("npu_confusion_transpose_backward", TORCH_FN(at_npu::native::wrapper__npu_confusion_transpose_backward));
  m.impl("npu_conv2d", TORCH_FN(at_npu::native::wrapper__npu_conv2d));
  m.impl("npu_conv2d.out", TORCH_FN(at_npu::native::wrapper_out_npu_conv2d_out));
  m.impl("npu_conv2d_backward", TORCH_FN(at_npu::native::wrapper__npu_conv2d_backward));
  m.impl("npu_conv3d", TORCH_FN(at_npu::native::wrapper__npu_conv3d));
  m.impl("npu_conv3d.out", TORCH_FN(at_npu::native::wrapper_out_npu_conv3d_out));
  m.impl("npu_conv3d_backward", TORCH_FN(at_npu::native::wrapper__npu_conv3d_backward));
  m.impl("npu_conv_transpose2d", TORCH_FN(at_npu::native::wrapper__npu_conv_transpose2d));
  m.impl("npu_conv_transpose2d_backward", TORCH_FN(at_npu::native::wrapper__npu_conv_transpose2d_backward));
  m.impl("npu_conv_transpose3d_backward", TORCH_FN(at_npu::native::wrapper__npu_conv_transpose3d_backward));
  m.impl("npu_convert_weight_to_int4pack", TORCH_FN(at_npu::native::wrapper__npu_convert_weight_to_int4pack));
  m.impl("npu_convolution", TORCH_FN(at_npu::native::wrapper__npu_convolution));
  m.impl("npu_convolution_backward", TORCH_FN(at_npu::native::wrapper__npu_convolution_backward));
  m.impl("npu_convolution_transpose", TORCH_FN(at_npu::native::wrapper__npu_convolution_transpose));
  m.impl("npu_convolution_transpose_backward", TORCH_FN(at_npu::native::wrapper__npu_convolution_transpose_backward));
  m.impl("npu_deep_norm", TORCH_FN(at_npu::native::wrapper__npu_deep_norm));
  m.impl("npu_deep_norm_backward", TORCH_FN(at_npu::native::wrapper__npu_deep_norm_backward));
  m.impl("npu_deformable_conv2d", TORCH_FN(at_npu::native::wrapper__npu_deformable_conv2d));
  m.impl("npu_deformable_conv2dbk", TORCH_FN(at_npu::native::wrapper__npu_deformable_conv2dbk));
  m.impl("npu_diou", TORCH_FN(at_npu::native::wrapper__npu_diou));
  m.impl("npu_diou_backward", TORCH_FN(at_npu::native::wrapper__npu_diou_backward));
  m.impl("npu_dropout_backward", TORCH_FN(at_npu::native::wrapper__npu_dropout_backward));
  m.impl("npu_dropout_do_mask", TORCH_FN(at_npu::native::wrapper__npu_dropout_do_mask));
  m.impl("npu_dropout_with_add_softmax", TORCH_FN(at_npu::native::wrapper__npu_dropout_with_add_softmax));
  m.impl("npu_dropout_with_add_softmax_backward", TORCH_FN(at_npu::native::wrapper__npu_dropout_with_add_softmax_backward));
  m.impl("npu_dtype_cast", TORCH_FN(at_npu::native::wrapper__npu_dtype_cast));
  m.impl("npu_dtype_cast_", TORCH_FN(at_npu::native::wrapper__npu_dtype_cast_));
  m.impl("npu_dtype_cast_backward", TORCH_FN(at_npu::native::wrapper__npu_dtype_cast_backward));
  m.impl("npu_dynamic_quant", TORCH_FN(at_npu::native::wrapper__npu_dynamic_quant));
  m.impl("npu_dynamic_quant_asymmetric", TORCH_FN(at_npu::native::wrapper__npu_dynamic_quant_asymmetric));
  m.impl("npu_fast_gelu", TORCH_FN(at_npu::native::wrapper__npu_fast_gelu));
  m.impl("npu_fast_gelu_backward", TORCH_FN(at_npu::native::wrapper__npu_fast_gelu_backward));
  m.impl("npu_ffn", TORCH_FN(at_npu::native::wrapper__npu_ffn));
  m.impl("npu_fused_attention_score", TORCH_FN(at_npu::native::wrapper__npu_fused_attention_score));
  m.impl("npu_fused_attention_score_backward", TORCH_FN(at_npu::native::wrapper__npu_fused_attention_score_backward));
  m.impl("npu_fused_attention_score_fwd", TORCH_FN(at_npu::native::wrapper__npu_fused_attention_score_fwd));
  m.impl("npu_fused_attention_score_grad", TORCH_FN(at_npu::native::wrapper__npu_fused_attention_score_grad));
  m.impl("npu_fused_infer_attention_score", TORCH_FN(at_npu::native::wrapper__npu_fused_infer_attention_score));
  m.impl("npu_fused_infer_attention_score.out", TORCH_FN(at_npu::native::wrapper_out_npu_fused_infer_attention_score_out));
  m.impl("_npu_fused_infer_attention_score_get_max_workspace", TORCH_FN(at_npu::native::wrapper___npu_fused_infer_attention_score_get_max_workspace));
  m.impl("npu_fusion_attention", TORCH_FN(at_npu::native::wrapper__npu_fusion_attention));
  m.impl("npu_fusion_attention_grad", TORCH_FN(at_npu::native::wrapper__npu_fusion_attention_grad));
  m.impl("npu_fusion_attention_v2", TORCH_FN(at_npu::native::wrapper__npu_fusion_attention_v2));
  m.impl("npu_fusion_attention_grad_v2", TORCH_FN(at_npu::native::wrapper__npu_fusion_attention_grad_v2));
  m.impl("npu_fused_attention_layernorm_qkv_fwd", TORCH_FN(at_npu::native::wrapper__npu_fused_attention_layernorm_qkv_fwd));
  m.impl("npu_fused_attention_qkv_grad", TORCH_FN(at_npu::native::wrapper__npu_fused_attention_qkv_grad));
  m.impl("npu_geglu", TORCH_FN(at_npu::native::wrapper__npu_geglu));
  m.impl("npu_geglu_grad", TORCH_FN(at_npu::native::wrapper__npu_geglu_grad));
  m.impl("npu_get_float_status", TORCH_FN(at_npu::native::wrapper__npu_get_float_status));
  m.impl("npu_giou", TORCH_FN(at_npu::native::wrapper__npu_giou));
  m.impl("npu_giou_backward", TORCH_FN(at_npu::native::wrapper__npu_giou_backward));
  m.impl("npu_grid_assign_positive", TORCH_FN(at_npu::native::wrapper__npu_grid_assign_positive));
  m.impl("npu_group_norm_silu", TORCH_FN(at_npu::native::wrapper__npu_group_norm_silu));
  m.impl("npu_grouped_matmul", TORCH_FN(at_npu::native::wrapper__npu_grouped_matmul));
  m.impl("npu_grouped_matmul.List", TORCH_FN(at_npu::native::wrapper_List_npu_grouped_matmul));
  m.impl("npu_grouped_matmul_finalize_routing", TORCH_FN(at_npu::native::wrapper__npu_grouped_matmul_finalize_routing));
  m.impl("npu_gru", TORCH_FN(at_npu::native::wrapper__npu_gru));
  m.impl("npu_gru_backward", TORCH_FN(at_npu::native::wrapper__npu_gru_backward));
  m.impl("npu_hans_encode.out", TORCH_FN(at_npu::native::wrapper_out_npu_hans_encode_out));
  m.impl("npu_hans_decode.out", TORCH_FN(at_npu::native::wrapper_out_npu_hans_decode_out));
  m.impl("npu_ifmr", TORCH_FN(at_npu::native::wrapper__npu_ifmr));
  m.impl("npu_our_incre_flash_attention", TORCH_FN(at_npu::native::wrapper__npu_our_incre_flash_attention));
  m.impl("npu_sparse_paged_attention", TORCH_FN(at_npu::native::wrapper__npu_sparse_paged_attention));
  m.impl("npu_cent_select", TORCH_FN(at_npu::native::wrapper__npu_cent_select));
  m.impl("npu_interleave_rope", TORCH_FN(at_npu::native::wrapper__npu_interleave_rope));
  m.impl("npu_indexing", TORCH_FN(at_npu::native::wrapper__npu_indexing));
  m.impl("npu_indexing.out", TORCH_FN(at_npu::native::wrapper_out_npu_indexing_out));
  m.impl("npu_iou", TORCH_FN(at_npu::native::wrapper__npu_iou));
  m.impl("npu_layer_norm_eval", TORCH_FN(at_npu::native::wrapper__npu_layer_norm_eval));
  m.impl("npu_layernorm_grad", TORCH_FN(at_npu::native::wrapper__npu_layernorm_grad));
  m.impl("npu_linear", TORCH_FN(at_npu::native::wrapper__npu_linear));
  m.impl("npu_linear_backward", TORCH_FN(at_npu::native::wrapper__npu_linear_backward));
  m.impl("npu_lstm", TORCH_FN(at_npu::native::wrapper__npu_lstm));
  m.impl("npu_lstm_backward", TORCH_FN(at_npu::native::wrapper__npu_lstm_backward));
  m.impl("npu_lstm_cell", TORCH_FN(at_npu::native::wrapper__npu_lstm_cell));
  m.impl("npu_lstm_cell_backward", TORCH_FN(at_npu::native::wrapper__npu_lstm_cell_backward));
  m.impl("npu_lstm_data", TORCH_FN(at_npu::native::wrapper__npu_lstm_data));
  m.impl("npu_lstm_data_backward", TORCH_FN(at_npu::native::wrapper__npu_lstm_data_backward));
  m.impl("npu_masked_fill_range", TORCH_FN(at_npu::native::wrapper__npu_masked_fill_range));
  m.impl("npu_masked_softmax_with_rel_pos_bias", TORCH_FN(at_npu::native::wrapper__npu_masked_softmax_with_rel_pos_bias));
  m.impl("npu_max.dim", TORCH_FN(at_npu::native::wrapper_dim_npu_max));
  m.impl("npu_max.names_dim", TORCH_FN(at_npu::native::wrapper_names_dim_npu_max));
  m.impl("npu_max_backward", TORCH_FN(at_npu::native::wrapper__npu_max_backward));
  m.impl("npu_min.dim", TORCH_FN(at_npu::native::wrapper_dim_npu_min));
  m.impl("npu_min.names_dim", TORCH_FN(at_npu::native::wrapper_names_dim_npu_min));
  m.impl("npu_min_backward", TORCH_FN(at_npu::native::wrapper__npu_min_backward));
  m.impl("npu_mish", TORCH_FN(at_npu::native::wrapper__npu_mish));
  m.impl("npu_mish_backward", TORCH_FN(at_npu::native::wrapper__npu_mish_backward));
  m.impl("npu_mla_prolog", TORCH_FN(at_npu::native::wrapper__npu_mla_prolog));
  m.impl("npu_mla_prolog_v2", TORCH_FN(at_npu::native::wrapper__npu_mla_prolog_v2));
  m.impl("npu_mm_all_reduce_base", TORCH_FN(at_npu::native::wrapper__npu_mm_all_reduce_base));
  m.impl("npu_mm_reduce_scatter_base", TORCH_FN(at_npu::native::wrapper__npu_mm_reduce_scatter_base));
  m.impl("npu_moe_compute_expert_tokens", TORCH_FN(at_npu::native::wrapper__npu_moe_compute_expert_tokens));
  m.impl("npu_moe_finalize_routing", TORCH_FN(at_npu::native::wrapper__npu_moe_finalize_routing));
  m.impl("npu_moe_gating_top_k_softmax", TORCH_FN(at_npu::native::wrapper__npu_moe_gating_top_k_softmax));
  m.impl("npu_moe_gating_top_k", TORCH_FN(at_npu::native::wrapper__npu_moe_gating_top_k));
  m.impl("npu_moe_init_routing", TORCH_FN(at_npu::native::wrapper__npu_moe_init_routing));
  m.impl("npu_moe_init_routing_v2", TORCH_FN(at_npu::native::wrapper__npu_moe_init_routing_v2));
  m.impl("npu_grouped_matmul_swiglu_quant", TORCH_FN(at_npu::native::wrapper__npu_grouped_matmul_swiglu_quant));
  m.impl("npu_moe_distribute_dispatch", TORCH_FN(at_npu::native::wrapper__npu_moe_distribute_dispatch));
  m.impl("npu_moe_distribute_dispatch_v2", TORCH_FN(at_npu::native::wrapper__npu_moe_distribute_dispatch_v2));
  m.impl("npu_moe_distribute_combine", TORCH_FN(at_npu::native::wrapper__npu_moe_distribute_combine));
  m.impl("npu_moe_distribute_combine_v2", TORCH_FN(at_npu::native::wrapper__npu_moe_distribute_combine_v2));
  m.impl("_npu_distribute_barrier", TORCH_FN(at_npu::native::wrapper___npu_distribute_barrier));
  m.impl("npu_moe_eplb_update_expert", TORCH_FN(at_npu::native::wrapper__npu_moe_eplb_update_expert));
  m.impl("npu_moe_re_routing", TORCH_FN(at_npu::native::wrapper__npu_moe_re_routing));
  m.impl("npu_moe_distribute_combine_add_rms_norm", TORCH_FN(at_npu::native::wrapper__npu_moe_distribute_combine_add_rms_norm));
  m.impl("npu_multi_head_attention", TORCH_FN(at_npu::native::wrapper__npu_multi_head_attention));
  m.impl("npu_multi_head_attention_backward", TORCH_FN(at_npu::native::wrapper__npu_multi_head_attention_backward));
  m.impl("npu_multi_head_attention_v2", TORCH_FN(at_npu::native::wrapper__npu_multi_head_attention_v2));
  m.impl("npu_multi_head_attention_v2_grad", TORCH_FN(at_npu::native::wrapper__npu_multi_head_attention_v2_grad));
  m.impl("npu_nms_rotated", TORCH_FN(at_npu::native::wrapper__npu_nms_rotated));
  m.impl("npu_nms_v4", TORCH_FN(at_npu::native::wrapper__npu_nms_v4));
  m.impl("npu_nms_with_mask", TORCH_FN(at_npu::native::wrapper__npu_nms_with_mask));
  m.impl("npu_normalize_batch", TORCH_FN(at_npu::native::wrapper__npu_normalize_batch));
  m.impl("npu_one_hot", TORCH_FN(at_npu::native::wrapper__npu_one_hot));
  m.impl("npu_pad", TORCH_FN(at_npu::native::wrapper__npu_pad));
  m.impl("npu_prompt_flash_attention", TORCH_FN(at_npu::native::wrapper__npu_prompt_flash_attention));
  m.impl("npu_ps_roi_pooling", TORCH_FN(at_npu::native::wrapper__npu_ps_roi_pooling));
  m.impl("npu_ps_roi_pooling_backward", TORCH_FN(at_npu::native::wrapper__npu_ps_roi_pooling_backward));
  m.impl("npu_ptiou", TORCH_FN(at_npu::native::wrapper__npu_ptiou));
  m.impl("npu_prefetch", TORCH_FN(at_npu::native::wrapper__npu_prefetch));
  m.impl("npu_quant_conv2d", TORCH_FN(at_npu::native::wrapper__npu_quant_conv2d));
  m.impl("npu_quant_matmul", TORCH_FN(at_npu::native::wrapper__npu_quant_matmul));
  m.impl("npu_quant_matmul_dequant", TORCH_FN(at_npu::native::wrapper__npu_quant_matmul_dequant));
  m.impl("npu_quant_grouped_matmul_dequant", TORCH_FN(at_npu::native::wrapper__npu_quant_grouped_matmul_dequant));
  m.impl("npu_quant_scatter", TORCH_FN(at_npu::native::wrapper__npu_quant_scatter));
  m.impl("npu_quant_scatter_", TORCH_FN(at_npu::native::wrapper__npu_quant_scatter_));
  m.impl("npu_quantize", TORCH_FN(at_npu::native::wrapper__npu_quantize));
  m.impl("npu_kronecker_quant", TORCH_FN(at_npu::native::wrapper__npu_kronecker_quant));
  m.impl("npu_group_quant", TORCH_FN(at_npu::native::wrapper__npu_group_quant));
  m.impl("npu_random_choice_with_mask", TORCH_FN(at_npu::native::wrapper__npu_random_choice_with_mask));
  m.impl("npu_reshape", TORCH_FN(at_npu::native::wrapper__npu_reshape));
  m.impl("npu_reshape.out", TORCH_FN(at_npu::native::wrapper_out_npu_reshape_out));
  m.impl("npu_rms_norm", TORCH_FN(at_npu::native::wrapper__npu_rms_norm));
  m.impl("npu_gemma_rms_norm", TORCH_FN(at_npu::native::wrapper__npu_gemma_rms_norm));
  m.impl("npu_rms_norm_backward", TORCH_FN(at_npu::native::wrapper__npu_rms_norm_backward));
  m.impl("npu_roi_align", TORCH_FN(at_npu::native::wrapper__npu_roi_align));
  m.impl("npu_roi_alignbk", TORCH_FN(at_npu::native::wrapper__npu_roi_alignbk));
  m.impl("npu_rotary_mul", TORCH_FN(at_npu::native::wrapper__npu_rotary_mul));
  m.impl("npu_rotary_mul_backward", TORCH_FN(at_npu::native::wrapper__npu_rotary_mul_backward));
  m.impl("npu_mrope", TORCH_FN(at_npu::native::wrapper__npu_mrope));
  m.impl("npu_rotated_box_decode", TORCH_FN(at_npu::native::wrapper__npu_rotated_box_decode));
  m.impl("npu_rotated_box_encode", TORCH_FN(at_npu::native::wrapper__npu_rotated_box_encode));
  m.impl("npu_rotated_iou", TORCH_FN(at_npu::native::wrapper__npu_rotated_iou));
  m.impl("npu_rotated_overlaps", TORCH_FN(at_npu::native::wrapper__npu_rotated_overlaps));
  m.impl("npu_scaled_masked_softmax", TORCH_FN(at_npu::native::wrapper__npu_scaled_masked_softmax));
  m.impl("npu_scaled_masked_softmax_backward", TORCH_FN(at_npu::native::wrapper__npu_scaled_masked_softmax_backward));
  m.impl("npu_scatter", TORCH_FN(at_npu::native::wrapper__npu_scatter));
  m.impl("npu_scatter_list", TORCH_FN(at_npu::native::wrapper__npu_scatter_list));
  m.impl("npu_scatter_list_", TORCH_FN(at_npu::native::wrapper__npu_scatter_list_));
  m.impl("npu_scatter_nd_update", TORCH_FN(at_npu::native::wrapper__npu_scatter_nd_update));
  m.impl("npu_scatter_nd_update_", TORCH_FN(at_npu::native::wrapper__npu_scatter_nd_update_));
  m.impl("npu_sign_bits_pack", TORCH_FN(at_npu::native::wrapper__npu_sign_bits_pack));
  m.impl("npu_sign_bits_unpack", TORCH_FN(at_npu::native::wrapper__npu_sign_bits_unpack));
  m.impl("npu_silu", TORCH_FN(at_npu::native::wrapper__npu_silu));
  m.impl("npu_silu_", TORCH_FN(at_npu::native::wrapper__npu_silu_));
  m.impl("npu_silu_backward", TORCH_FN(at_npu::native::wrapper__npu_silu_backward));
  m.impl("npu_slice", TORCH_FN(at_npu::native::wrapper__npu_slice));
  m.impl("npu_slice.out", TORCH_FN(at_npu::native::wrapper_out_npu_slice_out));
  m.impl("npu_softmax_cross_entropy_with_logits", TORCH_FN(at_npu::native::wrapper__npu_softmax_cross_entropy_with_logits));
  m.impl("npu_softmax_cross_entropy_with_logits_backward", TORCH_FN(at_npu::native::wrapper__npu_softmax_cross_entropy_with_logits_backward));
  m.impl("npu_sort_v2", TORCH_FN(at_npu::native::wrapper__npu_sort_v2));
  m.impl("npu_sort_v2.out", TORCH_FN(at_npu::native::wrapper_out_npu_sort_v2_out));
  m.impl("npu_stride_add", TORCH_FN(at_npu::native::wrapper__npu_stride_add));
  m.impl("npu_stride_copy", TORCH_FN(at_npu::native::wrapper__npu_stride_copy));
  m.impl("npu_stride_copy.out", TORCH_FN(at_npu::native::wrapper_out_npu_stride_copy_out));
  m.impl("npu_sub_sample", TORCH_FN(at_npu::native::wrapper__npu_sub_sample));
  m.impl("npu_swiglu", TORCH_FN(at_npu::native::wrapper__npu_swiglu));
  m.impl("npu_swiglu_backward", TORCH_FN(at_npu::native::wrapper__npu_swiglu_backward));
  m.impl("npu_dequant_swiglu_quant", TORCH_FN(at_npu::native::wrapper__npu_dequant_swiglu_quant));
  m.impl("npu_dequant_rope_quant_kvcache", TORCH_FN(at_npu::native::wrapper__npu_dequant_rope_quant_kvcache));
  m.impl("npu_rope_quant_kvcache", TORCH_FN(at_npu::native::wrapper__npu_rope_quant_kvcache));
  m.impl("npu_dequant_bias", TORCH_FN(at_npu::native::wrapper__npu_dequant_bias));
  m.impl("npu_trans_quant_param", TORCH_FN(at_npu::native::wrapper__npu_trans_quant_param));
  m.impl("npu_transpose", TORCH_FN(at_npu::native::wrapper__npu_transpose));
  m.impl("npu_transpose.out", TORCH_FN(at_npu::native::wrapper_out_npu_transpose_out));
  m.impl("npu_view_copy", TORCH_FN(at_npu::native::wrapper__npu_view_copy));
  m.impl("npu_weight_quant_batchmatmul", TORCH_FN(at_npu::native::wrapper__npu_weight_quant_batchmatmul));
  m.impl("npu_transpose_batchmatmul", TORCH_FN(at_npu::native::wrapper__npu_transpose_batchmatmul));
  m.impl("npu_yolo_boxes_encode", TORCH_FN(at_npu::native::wrapper__npu_yolo_boxes_encode));
  m.impl("one_", TORCH_FN(at_npu::native::wrapper__one_));
  m.impl("repeat_interleave_backward_int", TORCH_FN(at_npu::native::wrapper__repeat_interleave_backward_int));
  m.impl("repeat_interleave_backward_tensor", TORCH_FN(at_npu::native::wrapper__repeat_interleave_backward_tensor));
  m.impl("scatter_update", TORCH_FN(at_npu::native::wrapper__scatter_update));
  m.impl("scatter_update_", TORCH_FN(at_npu::native::wrapper__scatter_update_));
  m.impl("slow_conv_transpose2d_backward", TORCH_FN(at_npu::native::wrapper__slow_conv_transpose2d_backward));
  m.impl("stft_backward", TORCH_FN(at_npu::native::wrapper__stft_backward));
  m.impl("fft_r2c_backward", TORCH_FN(at_npu::native::wrapper__fft_r2c_backward));
  m.impl("fft_c2r_backward", TORCH_FN(at_npu::native::wrapper__fft_c2r_backward));
  m.impl("npu_cross_entropy_loss", TORCH_FN(at_npu::native::wrapper__npu_cross_entropy_loss));
  m.impl("npu_cross_entropy_loss_backward", TORCH_FN(at_npu::native::wrapper__npu_cross_entropy_loss_backward));
  m.impl("npu_group_norm_swish", TORCH_FN(at_npu::native::wrapper__npu_group_norm_swish));
  m.impl("npu_group_norm_swish_grad", TORCH_FN(at_npu::native::wrapper__npu_group_norm_swish_grad));
  m.impl("npu_advance_step_flashattn", TORCH_FN(at_npu::native::wrapper__npu_advance_step_flashattn));
  m.impl("npu_grouped_matmul_add_", TORCH_FN(at_npu::native::wrapper__npu_grouped_matmul_add_));
  m.impl("npu_attn_softmax_", TORCH_FN(at_npu::native::wrapper__npu_attn_softmax_));
  m.impl("npu_attn_softmax_backward_", TORCH_FN(at_npu::native::wrapper__npu_attn_softmax_backward_));
  m.impl("npu_gather_sparse_index", TORCH_FN(at_npu::native::wrapper__npu_gather_sparse_index));
  m.impl("npu_nsa_compress", TORCH_FN(at_npu::native::wrapper__npu_nsa_compress));
  m.impl("npu_nsa_compress_grad", TORCH_FN(at_npu::native::wrapper__npu_nsa_compress_grad));
  m.impl("npu_nsa_compress_infer.cache", TORCH_FN(at_npu::native::wrapper_cache_npu_nsa_compress_infer_out));
  m.impl("npu_nsa_compress_attention", TORCH_FN(at_npu::native::wrapper__npu_nsa_compress_attention));
  m.impl("npu_nsa_compress_attention_infer", TORCH_FN(at_npu::native::wrapper__npu_nsa_compress_attention_infer));
  m.impl("npu_nsa_select_attention", TORCH_FN(at_npu::native::wrapper__npu_nsa_select_attention));
  m.impl("npu_nsa_select_attention_grad", TORCH_FN(at_npu::native::wrapper__npu_nsa_select_attention_grad));
  m.impl("npu_nsa_select_attention_infer", TORCH_FN(at_npu::native::wrapper__npu_nsa_select_attention_infer));
  m.impl("npu_top_k_top_p", TORCH_FN(at_npu::native::wrapper__npu_top_k_top_p));
  m.impl("npu_moe_token_permute", TORCH_FN(at_npu::native::wrapper__npu_moe_token_permute));
  m.impl("npu_moe_token_unpermute", TORCH_FN(at_npu::native::wrapper__npu_moe_token_unpermute));
}

} // anonymous namespace

} // namespace native

} // namespace at_npu
