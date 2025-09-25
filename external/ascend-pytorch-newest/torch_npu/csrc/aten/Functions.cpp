#include <torch/csrc/dynamo/compiled_autograd.h>

#include "torch_npu/csrc/framework/autograd/FunctionsManual.h"

#include "torch_npu/csrc/aten/CustomFunctions.h"

// @generated from ../../../../../../../workspace/profile/cann-ops-adv-vllm/external/ascend-pytorch-newest/codegen/autograd/templates/Functions.cpp

// The manual function definitions that used to be here are now in torch/csrc/autograd/FunctionsManual.cpp
// This speeds up re-compilation and allow to share these implementations so that they can be
// used for forward mode AD formulas as well.

using namespace at_npu::autograd::generated::details;
using namespace at_npu::native::custom_ops;
using at::Tensor;
using at::Scalar;
using at::IntArrayRef;
using at::TensorList;

namespace at_npu { namespace autograd { namespace generated {

variable_list GatherBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto index = index_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (npu_gather_backward(grad, self_sym_sizes, dim, index, sparse_grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void GatherBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(index_, false);
    args.collect(self_sym_sizes);
    args.collect(sparse_grad);
}
variable_list GatherBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(index_);
    saved.before(self_sym_sizes);
    saved.before(sparse_grad);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(index_);
    saved.after(self_sym_sizes);
    saved.after(sparse_grad);
    return result;
}
variable_list DropoutWithByteMaskBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto result1 = result1_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (_dropout_with_byte_mask_backward(grad, result1, p)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void DropoutWithByteMaskBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(p);
    args.collect(result1_, true);
}
variable_list DropoutWithByteMaskBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(p);
    saved.before(result1_);
    variable_list result = apply(variable_list(grads));
    saved.after(p);
    saved.after(result1_);
    return result;
}
variable_list NpuCiouBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto gtboxes_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto gtboxes = gtboxes_.unpack();
  auto self = self_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  if (task_should_compute_output({ self_ix, gtboxes_ix })) {
  
    auto grad_result = npu_ciou_backward(grad, self, gtboxes, result1, trans, is_cross, mode);
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ gtboxes_ix })) {
        copy_range(grad_inputs, gtboxes_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
void NpuCiouBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(gtboxes_, false);
    args.collect(is_cross);
    args.collect(mode);
    args.collect(self_, false);
    args.collect(trans);
    args.collect(result1_, true);
}
variable_list NpuCiouBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(gtboxes_);
    saved.before(is_cross);
    saved.before(mode);
    saved.before(self_);
    saved.before(trans);
    saved.before(result1_);
    variable_list result = apply(variable_list(grads));
    saved.after(gtboxes_);
    saved.after(is_cross);
    saved.after(mode);
    saved.after(self_);
    saved.after(trans);
    saved.after(result1_);
    return result;
}
variable_list NpuDropoutBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto result1 = result1_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (npu_dropout_backward(grad, result1, p)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void NpuDropoutBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(p);
    args.collect(result1_, true);
}
variable_list NpuDropoutBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(p);
    saved.before(result1_);
    variable_list result = apply(variable_list(grads));
    saved.after(p);
    saved.after(result1_);
    return result;
}
variable_list NpuFormatCastBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void NpuFormatCastBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list NpuFormatCastBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list BinaryCrossEntropyWithLogitsBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto pos_weight = pos_weight_.unpack();
  auto self = self_.unpack();
  auto target = target_.unpack();
  auto weight = weight_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (npu_binary_cross_entropy_with_logits_backward(grad, self, target, weight, pos_weight, reduction)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void BinaryCrossEntropyWithLogitsBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(pos_weight_, false);
    args.collect(reduction);
    args.collect(self_, false);
    args.collect(target_, false);
    args.collect(weight_, false);
}
variable_list BinaryCrossEntropyWithLogitsBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(pos_weight_);
    saved.before(reduction);
    saved.before(self_);
    saved.before(target_);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(pos_weight_);
    saved.after(reduction);
    saved.after(self_);
    saved.after(target_);
    saved.after(weight_);
    return result;
}
variable_list FastGeluBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (npu_fast_gelu_backward(grad, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void FastGeluBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_, false);
}
variable_list FastGeluBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list KlDivBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto target = target_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (kl_div_backward(grad, self, target, reduction, log_target)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void KlDivBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(log_target);
    args.collect(reduction);
    args.collect(self_, false);
    args.collect(target_, false);
}
variable_list KlDivBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(log_target);
    saved.before(reduction);
    saved.before(self_);
    saved.before(target_);
    variable_list result = apply(variable_list(grads));
    saved.after(log_target);
    saved.after(reduction);
    saved.after(self_);
    saved.after(target_);
    return result;
}
variable_list L1LossBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto target_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto target = target_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (l1_loss_backward(grad, self, target, reduction)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ target_ix })) {
    auto grad_result = any_grad_defined ? (l1_loss_backward(grad, self, target, reduction) * -1) : Tensor();
    copy_range(grad_inputs, target_ix, grad_result);
  }
  return grad_inputs;
}
void L1LossBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(reduction);
    args.collect(self_, false);
    args.collect(target_, false);
}
variable_list L1LossBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(reduction);
    saved.before(self_);
    saved.before(target_);
    variable_list result = apply(variable_list(grads));
    saved.after(reduction);
    saved.after(self_);
    saved.after(target_);
    return result;
}
variable_list MatmulBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_out_ix = gen.range(1);
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto grad_out = grad_out_.unpack();
  auto other = other_.unpack();
  auto self = self_.unpack();
  if (task_should_compute_output({ grad_out_ix, self_ix, other_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ grad_out_ix }),
        task_should_compute_output({ self_ix }),
        task_should_compute_output({ other_ix }),
      };
    auto grad_result = matmul_double_backward(grads[0], grads[1], grad_out, self, other, grad_input_mask);
      if (task_should_compute_output({ grad_out_ix })) {
        copy_range(grad_inputs, grad_out_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ other_ix })) {
        copy_range(grad_inputs, other_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void MatmulBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(grad_out_, false);
    args.collect(other_, false);
    args.collect(self_, false);
}
variable_list MatmulBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(grad_out_);
    saved.before(other_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(grad_out_);
    saved.after(other_);
    saved.after(self_);
    return result;
}
variable_list NpuAddLayerNormBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto x1_ix = gen.range(1);
  auto x2_ix = gen.range(1);
  auto gamma_ix = gen.range(1);
  auto beta_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto gamma = gamma_.unpack();
  auto x1 = x1_.unpack();
  auto x2 = x2_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  if (task_should_compute_output({ x1_ix, x2_ix, gamma_ix, beta_ix })) {
  
    auto grad_result = npu_add_layer_norm_backward(grads[0], x1, x2, result2, result1, gamma, grads[1]);
      if (task_should_compute_output({ x1_ix })) {
        copy_range(grad_inputs, x1_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ x2_ix })) {
        copy_range(grad_inputs, x2_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ gamma_ix })) {
        copy_range(grad_inputs, gamma_ix, std::get<2>(grad_result));
      }
      if (task_should_compute_output({ beta_ix })) {
        copy_range(grad_inputs, beta_ix, std::get<3>(grad_result));
      }
  }
  return grad_inputs;
}
void NpuAddLayerNormBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(gamma_, false);
    args.collect(x1_, false);
    args.collect(x2_, false);
    args.collect(result1_, true);
    args.collect(result2_, true);
}
variable_list NpuAddLayerNormBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(gamma_);
    saved.before(x1_);
    saved.before(x2_);
    saved.before(result1_);
    saved.before(result2_);
    variable_list result = apply(variable_list(grads));
    saved.after(gamma_);
    saved.after(x1_);
    saved.after(x2_);
    saved.after(result1_);
    saved.after(result2_);
    return result;
}
variable_list NpuGeluBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (npu_gelu_backward(grad, self, approximate)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void NpuGeluBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(approximate);
    args.collect(self_, false);
}
variable_list NpuGeluBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(approximate);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(approximate);
    saved.after(self_);
    return result;
}
variable_list NpuBmmv2Backward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto mat2_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto mat2 = mat2_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ mat2_ix })) {
    auto grad_result = any_grad_defined ? (npu_bmm_v2_mat2_backward(grad, self, mat2, mat2_sym_sizes)) : Tensor();
    copy_range(grad_inputs, mat2_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (npu_bmm_v2_mat1_backward(grad, self, mat2, self_sym_sizes)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void NpuBmmv2Backward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(mat2_, false);
    args.collect(mat2_sym_sizes);
    args.collect(self_, false);
    args.collect(self_sym_sizes);
}
variable_list NpuBmmv2Backward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(mat2_);
    saved.before(mat2_sym_sizes);
    saved.before(self_);
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(mat2_);
    saved.after(mat2_sym_sizes);
    saved.after(self_);
    saved.after(self_sym_sizes);
    return result;
}
variable_list NpuConfusionTransposeBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (npu_confusion_transpose_backward(grad, perm, self_sym_sizes, !transpose_first)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void NpuConfusionTransposeBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(perm);
    args.collect(self_sym_sizes);
    args.collect(transpose_first);
}
variable_list NpuConfusionTransposeBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(perm);
    saved.before(self_sym_sizes);
    saved.before(transpose_first);
    variable_list result = apply(variable_list(grads));
    saved.after(perm);
    saved.after(self_sym_sizes);
    saved.after(transpose_first);
    return result;
}
variable_list NpuConvolutionBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto input = input_.unpack();
  auto weight = weight_.unpack();
  if (task_should_compute_output({ input_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ input_ix }),
        task_should_compute_output({ weight_ix }),
        task_should_compute_output({ bias_ix }),
      };
    auto grad_result = npu_convolution_backward(input, grad, weight, stride, padding, dilation, groups, grad_input_mask);
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void NpuConvolutionBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dilation);
    args.collect(groups);
    args.collect(input_, false);
    args.collect(padding);
    args.collect(stride);
    args.collect(weight_, false);
}
variable_list NpuConvolutionBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dilation);
    saved.before(groups);
    saved.before(input_);
    saved.before(padding);
    saved.before(stride);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(dilation);
    saved.after(groups);
    saved.after(input_);
    saved.after(padding);
    saved.after(stride);
    saved.after(weight_);
    return result;
}
variable_list NpuConvolutionTransposeBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto input = input_.unpack();
  auto weight = weight_.unpack();
  if (task_should_compute_output({ input_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ input_ix }),
        task_should_compute_output({ weight_ix }),
        task_should_compute_output({ bias_ix }),
      };
    auto grad_result = npu_convolution_transpose_backward(input, grad, weight, padding, output_padding, stride, dilation, groups, grad_input_mask);
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void NpuConvolutionTransposeBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dilation);
    args.collect(groups);
    args.collect(input_, false);
    args.collect(output_padding);
    args.collect(padding);
    args.collect(stride);
    args.collect(weight_, false);
}
variable_list NpuConvolutionTransposeBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dilation);
    saved.before(groups);
    saved.before(input_);
    saved.before(output_padding);
    saved.before(padding);
    saved.before(stride);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(dilation);
    saved.after(groups);
    saved.after(input_);
    saved.after(output_padding);
    saved.after(padding);
    saved.after(stride);
    saved.after(weight_);
    return result;
}
variable_list NpuDeepNormBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto x_ix = gen.range(1);
  auto gx_ix = gen.range(1);
  auto beta_ix = gen.range(1);
  auto gamma_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto gamma = gamma_.unpack();
  auto gx = gx_.unpack();
  auto x = x_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  if (task_should_compute_output({ x_ix, gx_ix, beta_ix, gamma_ix })) {
  
    auto grad_result = npu_deep_norm_backward(grad, x, gx, gamma, result0, result1, alpha);
      if (task_should_compute_output({ x_ix })) {
        copy_range(grad_inputs, x_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ gx_ix })) {
        copy_range(grad_inputs, gx_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ beta_ix })) {
        copy_range(grad_inputs, beta_ix, std::get<2>(grad_result));
      }
      if (task_should_compute_output({ gamma_ix })) {
        copy_range(grad_inputs, gamma_ix, std::get<3>(grad_result));
      }
  }
  return grad_inputs;
}
void NpuDeepNormBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(alpha);
    args.collect(gamma_, false);
    args.collect(gx_, false);
    args.collect(x_, false);
    args.collect(result0_, true);
    args.collect(result1_, true);
}
variable_list NpuDeepNormBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(alpha);
    saved.before(gamma_);
    saved.before(gx_);
    saved.before(x_);
    saved.before(result0_);
    saved.before(result1_);
    variable_list result = apply(variable_list(grads));
    saved.after(alpha);
    saved.after(gamma_);
    saved.after(gx_);
    saved.after(x_);
    saved.after(result0_);
    saved.after(result1_);
    return result;
}
variable_list NpuDeformableConv2DBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto offset_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto input = input_.unpack();
  auto offset = offset_.unpack();
  auto weight = weight_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  if (task_should_compute_output({ input_ix, weight_ix, offset_ix, bias_ix })) {
  
    auto grad_result = npu_deformable_conv2dbk(input, grad, result1, weight, offset, kernel_size, stride, padding, dilation, groups, deformable_groups, modulated);
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ offset_ix })) {
        copy_range(grad_inputs, offset_ix, std::get<2>(grad_result));
      }
      if (task_should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<3>(grad_result));
      }
  }
  return grad_inputs;
}
void NpuDeformableConv2DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(deformable_groups);
    args.collect(dilation);
    args.collect(groups);
    args.collect(input_, false);
    args.collect(kernel_size);
    args.collect(modulated);
    args.collect(offset_, false);
    args.collect(padding);
    args.collect(stride);
    args.collect(weight_, false);
    args.collect(result1_, true);
}
variable_list NpuDeformableConv2DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(deformable_groups);
    saved.before(dilation);
    saved.before(groups);
    saved.before(input_);
    saved.before(kernel_size);
    saved.before(modulated);
    saved.before(offset_);
    saved.before(padding);
    saved.before(stride);
    saved.before(weight_);
    saved.before(result1_);
    variable_list result = apply(variable_list(grads));
    saved.after(deformable_groups);
    saved.after(dilation);
    saved.after(groups);
    saved.after(input_);
    saved.after(kernel_size);
    saved.after(modulated);
    saved.after(offset_);
    saved.after(padding);
    saved.after(stride);
    saved.after(weight_);
    saved.after(result1_);
    return result;
}
variable_list NpuDiouBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto gtboxes_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto gtboxes = gtboxes_.unpack();
  auto self = self_.unpack();
  if (task_should_compute_output({ self_ix, gtboxes_ix })) {
  
    auto grad_result = npu_diou_backward(grad, self, gtboxes, trans, is_cross, mode);
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ gtboxes_ix })) {
        copy_range(grad_inputs, gtboxes_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
void NpuDiouBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(gtboxes_, false);
    args.collect(is_cross);
    args.collect(mode);
    args.collect(self_, false);
    args.collect(trans);
}
variable_list NpuDiouBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(gtboxes_);
    saved.before(is_cross);
    saved.before(mode);
    saved.before(self_);
    saved.before(trans);
    variable_list result = apply(variable_list(grads));
    saved.after(gtboxes_);
    saved.after(is_cross);
    saved.after(mode);
    saved.after(self_);
    saved.after(trans);
    return result;
}
variable_list NpuDropoutDoMaskBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto result1 = result1_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (npu_dropout_backward(grad, result1, p)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void NpuDropoutDoMaskBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(p);
    args.collect(result1_, true);
}
variable_list NpuDropoutDoMaskBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(p);
    saved.before(result1_);
    variable_list result = apply(variable_list(grads));
    saved.after(p);
    saved.after(result1_);
    return result;
}
variable_list NpuDropoutWithAddSoftmaxBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto x1_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  if (task_should_compute_output({ self_ix, x1_ix })) {
  
    auto grad_result = npu_dropout_with_add_softmax_backward(grad, result0, result1, alpha, prob, dim);
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ x1_ix })) {
        copy_range(grad_inputs, x1_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
void NpuDropoutWithAddSoftmaxBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(alpha);
    args.collect(dim);
    args.collect(prob);
    args.collect(result0_, true);
    args.collect(result1_, true);
}
variable_list NpuDropoutWithAddSoftmaxBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(alpha);
    saved.before(dim);
    saved.before(prob);
    saved.before(result0_);
    saved.before(result1_);
    variable_list result = apply(variable_list(grads));
    saved.after(alpha);
    saved.after(dim);
    saved.after(prob);
    saved.after(result0_);
    saved.after(result1_);
    return result;
}
variable_list NpuDtypeCastBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (npu_dtype_cast_backward(grad, self_scalar_type)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void NpuDtypeCastBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_scalar_type);
}
variable_list NpuDtypeCastBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_scalar_type);
    variable_list result = apply(variable_list(grads));
    saved.after(self_scalar_type);
    return result;
}
variable_list NpuFusedAttentionScoreFwdBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto query_layer_ix = gen.range(1);
  auto key_layer_ix = gen.range(1);
  auto value_layer_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto key_layer = key_layer_.unpack();
  auto query_layer = query_layer_.unpack();
  auto value_layer = value_layer_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  if (task_should_compute_output({ query_layer_ix, key_layer_ix, value_layer_ix })) {
  
    auto grad_result = npu_fused_attention_score_backward(grad, result1, query_layer, key_layer, value_layer, result2, scale, keep_prob, query_transpose, key_transpose, value_transpose, dx_transpose);
      if (task_should_compute_output({ query_layer_ix })) {
        copy_range(grad_inputs, query_layer_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ key_layer_ix })) {
        copy_range(grad_inputs, key_layer_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ value_layer_ix })) {
        copy_range(grad_inputs, value_layer_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void NpuFusedAttentionScoreFwdBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dx_transpose);
    args.collect(keep_prob);
    args.collect(key_layer_, false);
    args.collect(key_transpose);
    args.collect(query_layer_, false);
    args.collect(query_transpose);
    args.collect(scale);
    args.collect(value_layer_, false);
    args.collect(value_transpose);
    args.collect(result1_, true);
    args.collect(result2_, true);
}
variable_list NpuFusedAttentionScoreFwdBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dx_transpose);
    saved.before(keep_prob);
    saved.before(key_layer_);
    saved.before(key_transpose);
    saved.before(query_layer_);
    saved.before(query_transpose);
    saved.before(scale);
    saved.before(value_layer_);
    saved.before(value_transpose);
    saved.before(result1_);
    saved.before(result2_);
    variable_list result = apply(variable_list(grads));
    saved.after(dx_transpose);
    saved.after(keep_prob);
    saved.after(key_layer_);
    saved.after(key_transpose);
    saved.after(query_layer_);
    saved.after(query_transpose);
    saved.after(scale);
    saved.after(value_layer_);
    saved.after(value_transpose);
    saved.after(result1_);
    saved.after(result2_);
    return result;
}
variable_list NpuFusionAttentionBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto query_ix = gen.range(1);
  auto key_ix = gen.range(1);
  auto value_ix = gen.range(1);
  auto pse_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto atten_mask = atten_mask_.unpack();
  auto key = key_.unpack();
  auto padding_mask = padding_mask_.unpack();
  auto pse = pse_.unpack();
  auto query = query_.unpack();
  auto value = value_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  if (task_should_compute_output({ query_ix, key_ix, value_ix, pse_ix })) {
  
    auto grad_result = npu_fusion_attention_grad(query, key, value, grad, head_num, input_layout, pse, padding_mask, atten_mask, result1, result2, result3, result0, scale, keep_prob, pre_tockens, next_tockens, inner_precise, result4, result5, result6, prefix, actual_seq_qlen, actual_seq_kvlen, sparse_mode, gen_mask_parallel, sync);
      if (task_should_compute_output({ query_ix })) {
        copy_range(grad_inputs, query_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ key_ix })) {
        copy_range(grad_inputs, key_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ value_ix })) {
        copy_range(grad_inputs, value_ix, std::get<2>(grad_result));
      }
      if (task_should_compute_output({ pse_ix })) {
        copy_range(grad_inputs, pse_ix, std::get<3>(grad_result));
      }
  }
  return grad_inputs;
}
void NpuFusionAttentionBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(actual_seq_kvlen);
    args.collect(actual_seq_qlen);
    args.collect(atten_mask_, false);
    args.collect(gen_mask_parallel);
    args.collect(head_num);
    args.collect(inner_precise);
    args.collect(input_layout);
    args.collect(keep_prob);
    args.collect(key_, false);
    args.collect(next_tockens);
    args.collect(padding_mask_, false);
    args.collect(pre_tockens);
    args.collect(prefix);
    args.collect(pse_, false);
    args.collect(query_, false);
    args.collect(scale);
    args.collect(sparse_mode);
    args.collect(sync);
    args.collect(value_, false);
    args.collect(result0_, true);
    args.collect(result1_, true);
    args.collect(result2_, true);
    args.collect(result3_, true);
    args.collect(result4);
    args.collect(result5);
    args.collect(result6);
}
variable_list NpuFusionAttentionBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(actual_seq_kvlen);
    saved.before(actual_seq_qlen);
    saved.before(atten_mask_);
    saved.before(gen_mask_parallel);
    saved.before(head_num);
    saved.before(inner_precise);
    saved.before(input_layout);
    saved.before(keep_prob);
    saved.before(key_);
    saved.before(next_tockens);
    saved.before(padding_mask_);
    saved.before(pre_tockens);
    saved.before(prefix);
    saved.before(pse_);
    saved.before(query_);
    saved.before(scale);
    saved.before(sparse_mode);
    saved.before(sync);
    saved.before(value_);
    saved.before(result0_);
    saved.before(result1_);
    saved.before(result2_);
    saved.before(result3_);
    saved.before(result4);
    saved.before(result5);
    saved.before(result6);
    variable_list result = apply(variable_list(grads));
    saved.after(actual_seq_kvlen);
    saved.after(actual_seq_qlen);
    saved.after(atten_mask_);
    saved.after(gen_mask_parallel);
    saved.after(head_num);
    saved.after(inner_precise);
    saved.after(input_layout);
    saved.after(keep_prob);
    saved.after(key_);
    saved.after(next_tockens);
    saved.after(padding_mask_);
    saved.after(pre_tockens);
    saved.after(prefix);
    saved.after(pse_);
    saved.after(query_);
    saved.after(scale);
    saved.after(sparse_mode);
    saved.after(sync);
    saved.after(value_);
    saved.after(result0_);
    saved.after(result1_);
    saved.after(result2_);
    saved.after(result3_);
    saved.after(result4);
    saved.after(result5);
    saved.after(result6);
    return result;
}
variable_list NpuFusionAttentionV2Backward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto query_ix = gen.range(1);
  auto key_ix = gen.range(1);
  auto value_ix = gen.range(1);
  auto pse_ix = gen.range(1);
  auto query_rope_ix = gen.range(1);
  auto key_rope_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto atten_mask = atten_mask_.unpack();
  auto key = key_.unpack();
  auto key_rope = key_rope_.unpack();
  auto padding_mask = padding_mask_.unpack();
  auto pse = pse_.unpack();
  auto query = query_.unpack();
  auto query_rope = query_rope_.unpack();
  auto value = value_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  if (task_should_compute_output({ query_ix, key_ix, value_ix, pse_ix, query_rope_ix, key_rope_ix })) {
  
    auto grad_result = npu_fusion_attention_grad_v2(query, key, value, grad, head_num, input_layout, pse, padding_mask, atten_mask, result1, result2, result3, result0, query_rope, key_rope, scale, keep_prob, pre_tokens, next_tokens, inner_precise, result4, result5, result6, prefix, actual_seq_qlen, actual_seq_kvlen, sparse_mode, gen_mask_parallel, sync, pse_type, q_start_idx, kv_start_idx);
      if (task_should_compute_output({ query_ix })) {
        copy_range(grad_inputs, query_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ key_ix })) {
        copy_range(grad_inputs, key_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ value_ix })) {
        copy_range(grad_inputs, value_ix, std::get<2>(grad_result));
      }
      if (task_should_compute_output({ pse_ix })) {
        copy_range(grad_inputs, pse_ix, std::get<3>(grad_result));
      }
      if (task_should_compute_output({ query_rope_ix })) {
        copy_range(grad_inputs, query_rope_ix, std::get<4>(grad_result));
      }
      if (task_should_compute_output({ key_rope_ix })) {
        copy_range(grad_inputs, key_rope_ix, std::get<5>(grad_result));
      }
  }
  return grad_inputs;
}
void NpuFusionAttentionV2Backward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(actual_seq_kvlen);
    args.collect(actual_seq_qlen);
    args.collect(atten_mask_, false);
    args.collect(gen_mask_parallel);
    args.collect(head_num);
    args.collect(inner_precise);
    args.collect(input_layout);
    args.collect(keep_prob);
    args.collect(key_, false);
    args.collect(key_rope_, false);
    args.collect(kv_start_idx);
    args.collect(next_tokens);
    args.collect(padding_mask_, false);
    args.collect(pre_tokens);
    args.collect(prefix);
    args.collect(pse_, false);
    args.collect(pse_type);
    args.collect(q_start_idx);
    args.collect(query_, false);
    args.collect(query_rope_, false);
    args.collect(scale);
    args.collect(sparse_mode);
    args.collect(sync);
    args.collect(value_, false);
    args.collect(result0_, true);
    args.collect(result1_, true);
    args.collect(result2_, true);
    args.collect(result3_, true);
    args.collect(result4);
    args.collect(result5);
    args.collect(result6);
}
variable_list NpuFusionAttentionV2Backward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(actual_seq_kvlen);
    saved.before(actual_seq_qlen);
    saved.before(atten_mask_);
    saved.before(gen_mask_parallel);
    saved.before(head_num);
    saved.before(inner_precise);
    saved.before(input_layout);
    saved.before(keep_prob);
    saved.before(key_);
    saved.before(key_rope_);
    saved.before(kv_start_idx);
    saved.before(next_tokens);
    saved.before(padding_mask_);
    saved.before(pre_tokens);
    saved.before(prefix);
    saved.before(pse_);
    saved.before(pse_type);
    saved.before(q_start_idx);
    saved.before(query_);
    saved.before(query_rope_);
    saved.before(scale);
    saved.before(sparse_mode);
    saved.before(sync);
    saved.before(value_);
    saved.before(result0_);
    saved.before(result1_);
    saved.before(result2_);
    saved.before(result3_);
    saved.before(result4);
    saved.before(result5);
    saved.before(result6);
    variable_list result = apply(variable_list(grads));
    saved.after(actual_seq_kvlen);
    saved.after(actual_seq_qlen);
    saved.after(atten_mask_);
    saved.after(gen_mask_parallel);
    saved.after(head_num);
    saved.after(inner_precise);
    saved.after(input_layout);
    saved.after(keep_prob);
    saved.after(key_);
    saved.after(key_rope_);
    saved.after(kv_start_idx);
    saved.after(next_tokens);
    saved.after(padding_mask_);
    saved.after(pre_tokens);
    saved.after(prefix);
    saved.after(pse_);
    saved.after(pse_type);
    saved.after(q_start_idx);
    saved.after(query_);
    saved.after(query_rope_);
    saved.after(scale);
    saved.after(sparse_mode);
    saved.after(sync);
    saved.after(value_);
    saved.after(result0_);
    saved.after(result1_);
    saved.after(result2_);
    saved.after(result3_);
    saved.after(result4);
    saved.after(result5);
    saved.after(result6);
    return result;
}
variable_list NpuGegluBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (npu_geglu_grad(grad, self, result1, dim, approximate, activate_left)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void NpuGegluBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(activate_left);
    args.collect(approximate);
    args.collect(dim);
    args.collect(self_, false);
    args.collect(result1_, true);
}
variable_list NpuGegluBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(activate_left);
    saved.before(approximate);
    saved.before(dim);
    saved.before(self_);
    saved.before(result1_);
    variable_list result = apply(variable_list(grads));
    saved.after(activate_left);
    saved.after(approximate);
    saved.after(dim);
    saved.after(self_);
    saved.after(result1_);
    return result;
}
variable_list NpuGiouBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto gtboxes_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto gtboxes = gtboxes_.unpack();
  auto self = self_.unpack();
  if (task_should_compute_output({ self_ix, gtboxes_ix })) {
  
    auto grad_result = npu_giou_backward(grad, self, gtboxes, trans, is_cross, mode);
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ gtboxes_ix })) {
        copy_range(grad_inputs, gtboxes_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
void NpuGiouBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(gtboxes_, false);
    args.collect(is_cross);
    args.collect(mode);
    args.collect(self_, false);
    args.collect(trans);
}
variable_list NpuGiouBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(gtboxes_);
    saved.before(is_cross);
    saved.before(mode);
    saved.before(self_);
    saved.before(trans);
    variable_list result = apply(variable_list(grads));
    saved.after(gtboxes_);
    saved.after(is_cross);
    saved.after(mode);
    saved.after(self_);
    saved.after(trans);
    return result;
}
variable_list NpuGruBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto hx_ix = gen.range(1);
  auto weight_input_ix = gen.range(1);
  auto weight_hidden_ix = gen.range(1);
  auto bias_input_ix = gen.range(1);
  auto bias_hidden_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto bias_hidden = bias_hidden_.unpack();
  auto bias_input = bias_input_.unpack();
  auto hx = hx_.unpack();
  auto input = input_.unpack();
  auto seq_length = seq_length_.unpack();
  auto weight_hidden = weight_hidden_.unpack();
  auto weight_input = weight_input_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  auto result4 = result4_.unpack(shared_from_this());
  auto result5 = result5_.unpack(shared_from_this());
  if (task_should_compute_output({ weight_input_ix, weight_hidden_ix, input_ix, bias_input_ix, bias_hidden_ix, hx_ix })) {
  
    auto grad_result = npu_gru_backward(grads[0], grads[1], input, weight_input, weight_hidden, bias_input, bias_hidden, seq_length, hx, result0, result1, result2, result3, result4, result5);
      if (task_should_compute_output({ weight_input_ix })) {
        copy_range(grad_inputs, weight_input_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_hidden_ix })) {
        copy_range(grad_inputs, weight_hidden_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<2>(grad_result));
      }
      if (task_should_compute_output({ bias_input_ix })) {
        copy_range(grad_inputs, bias_input_ix, std::get<3>(grad_result));
      }
      if (task_should_compute_output({ bias_hidden_ix })) {
        copy_range(grad_inputs, bias_hidden_ix, std::get<4>(grad_result));
      }
      if (task_should_compute_output({ hx_ix })) {
        copy_range(grad_inputs, hx_ix, std::get<5>(grad_result));
      }
  }
  return grad_inputs;
}
void NpuGruBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(bias_hidden_, false);
    args.collect(bias_input_, false);
    args.collect(hx_, false);
    args.collect(input_, false);
    args.collect(seq_length_, false);
    args.collect(weight_hidden_, false);
    args.collect(weight_input_, false);
    args.collect(result0_, true);
    args.collect(result1_, true);
    args.collect(result2_, true);
    args.collect(result3_, true);
    args.collect(result4_, true);
    args.collect(result5_, true);
}
variable_list NpuGruBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(bias_hidden_);
    saved.before(bias_input_);
    saved.before(hx_);
    saved.before(input_);
    saved.before(seq_length_);
    saved.before(weight_hidden_);
    saved.before(weight_input_);
    saved.before(result0_);
    saved.before(result1_);
    saved.before(result2_);
    saved.before(result3_);
    saved.before(result4_);
    saved.before(result5_);
    variable_list result = apply(variable_list(grads));
    saved.after(bias_hidden_);
    saved.after(bias_input_);
    saved.after(hx_);
    saved.after(input_);
    saved.after(seq_length_);
    saved.after(weight_hidden_);
    saved.after(weight_input_);
    saved.after(result0_);
    saved.after(result1_);
    saved.after(result2_);
    saved.after(result3_);
    saved.after(result4_);
    saved.after(result5_);
    return result;
}
variable_list NpuLinearBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto input = input_.unpack();
  auto weight = weight_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ bias_ix })) {
    auto grad_result = any_grad_defined ? (maybe_multiply(grad, 1)) : Tensor();
    copy_range(grad_inputs, bias_ix, grad_result);
  }
  if (task_should_compute_output({ input_ix, weight_ix })) {
  
    auto grad_result = npu_linear_backward(grad, input, weight);
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
void NpuLinearBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(input_, false);
    args.collect(weight_, false);
}
variable_list NpuLinearBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(input_);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(input_);
    saved.after(weight_);
    return result;
}
variable_list NpuLstmBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  auto h_ix = gen.range(1);
  auto c_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto bias = bias_.unpack();
  auto c = c_.unpack();
  auto h = h_.unpack();
  auto input = input_.unpack();
  auto weight = weight_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  auto result4 = result4_.unpack(shared_from_this());
  auto result5 = result5_.unpack(shared_from_this());
  auto result6 = result6_.unpack(shared_from_this());
  auto result7 = result7_.unpack(shared_from_this());
  if (task_should_compute_output({ input_ix, weight_ix, bias_ix, h_ix, c_ix })) {
  
    auto grad_result = npu_lstm_backward(grads[0], grads[1], grads[2], input, weight, bias, h, c, result0, result1, result2, result3, result4, result5, result6, result7);
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
      if (task_should_compute_output({ h_ix })) {
        copy_range(grad_inputs, h_ix, std::get<3>(grad_result));
      }
      if (task_should_compute_output({ c_ix })) {
        copy_range(grad_inputs, c_ix, std::get<4>(grad_result));
      }
  }
  return grad_inputs;
}
void NpuLstmBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(bias_, false);
    args.collect(c_, false);
    args.collect(h_, false);
    args.collect(input_, false);
    args.collect(weight_, false);
    args.collect(result0_, true);
    args.collect(result1_, true);
    args.collect(result2_, true);
    args.collect(result3_, true);
    args.collect(result4_, true);
    args.collect(result5_, true);
    args.collect(result6_, true);
    args.collect(result7_, true);
}
variable_list NpuLstmBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(bias_);
    saved.before(c_);
    saved.before(h_);
    saved.before(input_);
    saved.before(weight_);
    saved.before(result0_);
    saved.before(result1_);
    saved.before(result2_);
    saved.before(result3_);
    saved.before(result4_);
    saved.before(result5_);
    saved.before(result6_);
    saved.before(result7_);
    variable_list result = apply(variable_list(grads));
    saved.after(bias_);
    saved.after(c_);
    saved.after(h_);
    saved.after(input_);
    saved.after(weight_);
    saved.after(result0_);
    saved.after(result1_);
    saved.after(result2_);
    saved.after(result3_);
    saved.after(result4_);
    saved.after(result5_);
    saved.after(result6_);
    saved.after(result7_);
    return result;
}
variable_list NpuLstmCellBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto w_ih_ix = gen.range(1);
  auto w_hh_ix = gen.range(1);
  auto h_ix = gen.range(1);
  auto c_ix = gen.range(1);
  auto b_ih_ix = gen.range(1);
  auto b_hh_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto c = c_.unpack();
  auto h = h_.unpack();
  auto input = input_.unpack();
  auto w_hh = w_hh_.unpack();
  auto w_ih = w_ih_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  auto result4 = result4_.unpack(shared_from_this());
  auto result5 = result5_.unpack(shared_from_this());
  auto result6 = result6_.unpack(shared_from_this());
  auto result7 = result7_.unpack(shared_from_this());
  if (task_should_compute_output({ input_ix, w_ih_ix, w_hh_ix, b_ih_ix, b_hh_ix, h_ix, c_ix })) {
  
    auto grad_result = npu_lstm_cell_backward(grads[0], grads[1], grads[2], input, w_ih, w_hh, h, c, result0, result1, result2, result3, result4, result5, result6, result7);
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ w_ih_ix })) {
        copy_range(grad_inputs, w_ih_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ w_hh_ix })) {
        copy_range(grad_inputs, w_hh_ix, std::get<2>(grad_result));
      }
      if (task_should_compute_output({ b_ih_ix })) {
        copy_range(grad_inputs, b_ih_ix, std::get<3>(grad_result));
      }
      if (task_should_compute_output({ b_hh_ix })) {
        copy_range(grad_inputs, b_hh_ix, std::get<4>(grad_result));
      }
      if (task_should_compute_output({ h_ix })) {
        copy_range(grad_inputs, h_ix, std::get<5>(grad_result));
      }
      if (task_should_compute_output({ c_ix })) {
        copy_range(grad_inputs, c_ix, std::get<6>(grad_result));
      }
  }
  return grad_inputs;
}
void NpuLstmCellBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(c_, false);
    args.collect(h_, false);
    args.collect(input_, false);
    args.collect(w_hh_, false);
    args.collect(w_ih_, false);
    args.collect(result0_, true);
    args.collect(result1_, true);
    args.collect(result2_, true);
    args.collect(result3_, true);
    args.collect(result4_, true);
    args.collect(result5_, true);
    args.collect(result6_, true);
    args.collect(result7_, true);
}
variable_list NpuLstmCellBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(c_);
    saved.before(h_);
    saved.before(input_);
    saved.before(w_hh_);
    saved.before(w_ih_);
    saved.before(result0_);
    saved.before(result1_);
    saved.before(result2_);
    saved.before(result3_);
    saved.before(result4_);
    saved.before(result5_);
    saved.before(result6_);
    saved.before(result7_);
    variable_list result = apply(variable_list(grads));
    saved.after(c_);
    saved.after(h_);
    saved.after(input_);
    saved.after(w_hh_);
    saved.after(w_ih_);
    saved.after(result0_);
    saved.after(result1_);
    saved.after(result2_);
    saved.after(result3_);
    saved.after(result4_);
    saved.after(result5_);
    saved.after(result6_);
    saved.after(result7_);
    return result;
}
variable_list NpuLstmDataBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  auto h_ix = gen.range(1);
  auto c_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto batch_sizes = batch_sizes_.unpack();
  auto bias = bias_.unpack();
  auto c = c_.unpack();
  auto h = h_.unpack();
  auto input = input_.unpack();
  auto weight = weight_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  auto result4 = result4_.unpack(shared_from_this());
  auto result5 = result5_.unpack(shared_from_this());
  auto result6 = result6_.unpack(shared_from_this());
  auto result7 = result7_.unpack(shared_from_this());
  if (task_should_compute_output({ input_ix, weight_ix, bias_ix, h_ix, c_ix })) {
  
    auto grad_result = npu_lstm_data_backward(grads[0], grads[1], grads[2], input, batch_sizes, weight, bias, h, c, result0, result1, result2, result3, result4, result5, result6, result7, direction);
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
      if (task_should_compute_output({ h_ix })) {
        copy_range(grad_inputs, h_ix, std::get<3>(grad_result));
      }
      if (task_should_compute_output({ c_ix })) {
        copy_range(grad_inputs, c_ix, std::get<4>(grad_result));
      }
  }
  return grad_inputs;
}
void NpuLstmDataBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(batch_sizes_, false);
    args.collect(bias_, false);
    args.collect(c_, false);
    args.collect(direction);
    args.collect(h_, false);
    args.collect(input_, false);
    args.collect(weight_, false);
    args.collect(result0_, true);
    args.collect(result1_, true);
    args.collect(result2_, true);
    args.collect(result3_, true);
    args.collect(result4_, true);
    args.collect(result5_, true);
    args.collect(result6_, true);
    args.collect(result7_, true);
}
variable_list NpuLstmDataBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(batch_sizes_);
    saved.before(bias_);
    saved.before(c_);
    saved.before(direction);
    saved.before(h_);
    saved.before(input_);
    saved.before(weight_);
    saved.before(result0_);
    saved.before(result1_);
    saved.before(result2_);
    saved.before(result3_);
    saved.before(result4_);
    saved.before(result5_);
    saved.before(result6_);
    saved.before(result7_);
    variable_list result = apply(variable_list(grads));
    saved.after(batch_sizes_);
    saved.after(bias_);
    saved.after(c_);
    saved.after(direction);
    saved.after(h_);
    saved.after(input_);
    saved.after(weight_);
    saved.after(result0_);
    saved.after(result1_);
    saved.after(result2_);
    saved.after(result3_);
    saved.after(result4_);
    saved.after(result5_);
    saved.after(result6_);
    saved.after(result7_);
    return result;
}
variable_list NpuMaxBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto indices = indices_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (npu_max_backward(grad, dim, indices, self_sym_sizes, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void NpuMaxBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(keepdim);
    args.collect(self_sym_sizes);
    args.collect(indices_, true);
}
variable_list NpuMaxBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(keepdim);
    saved.before(self_sym_sizes);
    saved.before(indices_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(keepdim);
    saved.after(self_sym_sizes);
    saved.after(indices_);
    return result;
}
variable_list NpuMinBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto indices = indices_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (npu_min_backward(grad, dim, indices, self_sym_sizes, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void NpuMinBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(keepdim);
    args.collect(self_sym_sizes);
    args.collect(indices_, true);
}
variable_list NpuMinBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(keepdim);
    saved.before(self_sym_sizes);
    saved.before(indices_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(keepdim);
    saved.after(self_sym_sizes);
    saved.after(indices_);
    return result;
}
variable_list NpuMishBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (npu_mish_backward(grad, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void NpuMishBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_, false);
}
variable_list NpuMishBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list NpuMultiHeadAttentionBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto query_ix = gen.range(1);
  auto key_ix = gen.range(1);
  auto value_ix = gen.range(1);
  auto query_weight_ix = gen.range(1);
  auto key_weight_ix = gen.range(1);
  auto value_weight_ix = gen.range(1);
  auto out_proj_weight_ix = gen.range(1);
  auto query_bias_ix = gen.range(1);
  auto key_bias_ix = gen.range(1);
  auto value_bias_ix = gen.range(1);
  auto out_proj_bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto key = key_.unpack();
  auto key_bias = key_bias_.unpack();
  auto key_weight = key_weight_.unpack();
  auto out_proj_bias = out_proj_bias_.unpack();
  auto out_proj_weight = out_proj_weight_.unpack();
  auto query = query_.unpack();
  auto query_bias = query_bias_.unpack();
  auto query_weight = query_weight_.unpack();
  auto value = value_.unpack();
  auto value_bias = value_bias_.unpack();
  auto value_weight = value_weight_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  auto result4 = result4_.unpack(shared_from_this());
  auto result5 = result5_.unpack(shared_from_this());
  auto result6 = result6_.unpack(shared_from_this());
  auto result7 = result7_.unpack(shared_from_this());
  if (task_should_compute_output({ query_weight_ix, key_weight_ix, value_weight_ix, out_proj_weight_ix, query_ix, key_ix, value_ix, query_bias_ix, key_bias_ix, value_bias_ix, out_proj_bias_ix })) {
  
    auto grad_result = npu_multi_head_attention_backward(query, key, value, query_weight, key_weight, value_weight, out_proj_weight, query_bias, key_bias, value_bias, out_proj_bias, result2, result3, result4, result5, result6, result7, grad, result1, attn_head_num, attn_dim_per_head, src_len, tgt_len, dropout_prob, softmax_use_float);
      if (task_should_compute_output({ query_weight_ix })) {
        copy_range(grad_inputs, query_weight_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ key_weight_ix })) {
        copy_range(grad_inputs, key_weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ value_weight_ix })) {
        copy_range(grad_inputs, value_weight_ix, std::get<2>(grad_result));
      }
      if (task_should_compute_output({ out_proj_weight_ix })) {
        copy_range(grad_inputs, out_proj_weight_ix, std::get<3>(grad_result));
      }
      if (task_should_compute_output({ query_ix })) {
        copy_range(grad_inputs, query_ix, std::get<4>(grad_result));
      }
      if (task_should_compute_output({ key_ix })) {
        copy_range(grad_inputs, key_ix, std::get<5>(grad_result));
      }
      if (task_should_compute_output({ value_ix })) {
        copy_range(grad_inputs, value_ix, std::get<6>(grad_result));
      }
      if (task_should_compute_output({ query_bias_ix })) {
        copy_range(grad_inputs, query_bias_ix, std::get<7>(grad_result));
      }
      if (task_should_compute_output({ key_bias_ix })) {
        copy_range(grad_inputs, key_bias_ix, std::get<8>(grad_result));
      }
      if (task_should_compute_output({ value_bias_ix })) {
        copy_range(grad_inputs, value_bias_ix, std::get<9>(grad_result));
      }
      if (task_should_compute_output({ out_proj_bias_ix })) {
        copy_range(grad_inputs, out_proj_bias_ix, std::get<10>(grad_result));
      }
  }
  return grad_inputs;
}
void NpuMultiHeadAttentionBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(attn_dim_per_head);
    args.collect(attn_head_num);
    args.collect(dropout_prob);
    args.collect(key_, false);
    args.collect(key_bias_, false);
    args.collect(key_weight_, false);
    args.collect(out_proj_bias_, false);
    args.collect(out_proj_weight_, false);
    args.collect(query_, false);
    args.collect(query_bias_, false);
    args.collect(query_weight_, false);
    args.collect(softmax_use_float);
    args.collect(src_len);
    args.collect(tgt_len);
    args.collect(value_, false);
    args.collect(value_bias_, false);
    args.collect(value_weight_, false);
    args.collect(result1_, true);
    args.collect(result2_, true);
    args.collect(result3_, true);
    args.collect(result4_, true);
    args.collect(result5_, true);
    args.collect(result6_, true);
    args.collect(result7_, true);
}
variable_list NpuMultiHeadAttentionBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(attn_dim_per_head);
    saved.before(attn_head_num);
    saved.before(dropout_prob);
    saved.before(key_);
    saved.before(key_bias_);
    saved.before(key_weight_);
    saved.before(out_proj_bias_);
    saved.before(out_proj_weight_);
    saved.before(query_);
    saved.before(query_bias_);
    saved.before(query_weight_);
    saved.before(softmax_use_float);
    saved.before(src_len);
    saved.before(tgt_len);
    saved.before(value_);
    saved.before(value_bias_);
    saved.before(value_weight_);
    saved.before(result1_);
    saved.before(result2_);
    saved.before(result3_);
    saved.before(result4_);
    saved.before(result5_);
    saved.before(result6_);
    saved.before(result7_);
    variable_list result = apply(variable_list(grads));
    saved.after(attn_dim_per_head);
    saved.after(attn_head_num);
    saved.after(dropout_prob);
    saved.after(key_);
    saved.after(key_bias_);
    saved.after(key_weight_);
    saved.after(out_proj_bias_);
    saved.after(out_proj_weight_);
    saved.after(query_);
    saved.after(query_bias_);
    saved.after(query_weight_);
    saved.after(softmax_use_float);
    saved.after(src_len);
    saved.after(tgt_len);
    saved.after(value_);
    saved.after(value_bias_);
    saved.after(value_weight_);
    saved.after(result1_);
    saved.after(result2_);
    saved.after(result3_);
    saved.after(result4_);
    saved.after(result5_);
    saved.after(result6_);
    saved.after(result7_);
    return result;
}
variable_list NpuMultiHeadAttentionV2Backward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto query_ix = gen.range(1);
  auto key_ix = gen.range(1);
  auto value_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto alibi_mask = alibi_mask_.unpack();
  auto atten_mask = atten_mask_.unpack();
  auto key = key_.unpack();
  auto query = query_.unpack();
  auto value = value_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  if (task_should_compute_output({ query_ix, key_ix, value_ix })) {
  
    auto grad_result = npu_multi_head_attention_v2_grad(grad, query, key, value, result1, result0, atten_mask, alibi_mask, scale, head_num, input_layout, keep_prob, pre_tokens, next_tokens, result2, result3, result4, gen_mask_parallel, sync);
      if (task_should_compute_output({ query_ix })) {
        copy_range(grad_inputs, query_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ key_ix })) {
        copy_range(grad_inputs, key_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ value_ix })) {
        copy_range(grad_inputs, value_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void NpuMultiHeadAttentionV2Backward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(alibi_mask_, false);
    args.collect(atten_mask_, false);
    args.collect(gen_mask_parallel);
    args.collect(head_num);
    args.collect(input_layout);
    args.collect(keep_prob);
    args.collect(key_, false);
    args.collect(next_tokens);
    args.collect(pre_tokens);
    args.collect(query_, false);
    args.collect(scale);
    args.collect(sync);
    args.collect(value_, false);
    args.collect(result0_, true);
    args.collect(result1_, true);
    args.collect(result2);
    args.collect(result3);
    args.collect(result4);
}
variable_list NpuMultiHeadAttentionV2Backward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(alibi_mask_);
    saved.before(atten_mask_);
    saved.before(gen_mask_parallel);
    saved.before(head_num);
    saved.before(input_layout);
    saved.before(keep_prob);
    saved.before(key_);
    saved.before(next_tokens);
    saved.before(pre_tokens);
    saved.before(query_);
    saved.before(scale);
    saved.before(sync);
    saved.before(value_);
    saved.before(result0_);
    saved.before(result1_);
    saved.before(result2);
    saved.before(result3);
    saved.before(result4);
    variable_list result = apply(variable_list(grads));
    saved.after(alibi_mask_);
    saved.after(atten_mask_);
    saved.after(gen_mask_parallel);
    saved.after(head_num);
    saved.after(input_layout);
    saved.after(keep_prob);
    saved.after(key_);
    saved.after(next_tokens);
    saved.after(pre_tokens);
    saved.after(query_);
    saved.after(scale);
    saved.after(sync);
    saved.after(value_);
    saved.after(result0_);
    saved.after(result1_);
    saved.after(result2);
    saved.after(result3);
    saved.after(result4);
    return result;
}
variable_list NpuPsRoiPoolingBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto rois = rois_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (npu_ps_roi_pooling_backward(grad, rois, spatial_scale, group_size, output_dim, {self_sym_argsize_2, self_sym_argsize_3})) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void NpuPsRoiPoolingBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(group_size);
    args.collect(output_dim);
    args.collect(rois_, false);
    args.collect(self_sym_argsize_2);
    args.collect(self_sym_argsize_3);
    args.collect(spatial_scale);
}
variable_list NpuPsRoiPoolingBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(group_size);
    saved.before(output_dim);
    saved.before(rois_);
    saved.before(self_sym_argsize_2);
    saved.before(self_sym_argsize_3);
    saved.before(spatial_scale);
    variable_list result = apply(variable_list(grads));
    saved.after(group_size);
    saved.after(output_dim);
    saved.after(rois_);
    saved.after(self_sym_argsize_2);
    saved.after(self_sym_argsize_3);
    saved.after(spatial_scale);
    return result;
}
variable_list NpuRmsNormBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto gamma_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto gamma = gamma_.unpack();
  auto self = self_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  if (task_should_compute_output({ self_ix, gamma_ix })) {
  
    auto grad_result = npu_rms_norm_backward(grad, self, gamma, result1);
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ gamma_ix })) {
        copy_range(grad_inputs, gamma_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
void NpuRmsNormBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(gamma_, false);
    args.collect(self_, false);
    args.collect(result1_, true);
}
variable_list NpuRmsNormBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(gamma_);
    saved.before(self_);
    saved.before(result1_);
    variable_list result = apply(variable_list(grads));
    saved.after(gamma_);
    saved.after(self_);
    saved.after(result1_);
    return result;
}
variable_list NpuRotaryMulBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto r1_ix = gen.range(1);
  auto r2_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto r1 = r1_.unpack();
  auto r2 = r2_.unpack();
  auto self = self_.unpack();
  if (task_should_compute_output({ self_ix, r1_ix, r2_ix })) {
  
    auto grad_result = npu_rotary_mul_backward(grad, self, r1, r2, rotary_mode);
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ r1_ix })) {
        copy_range(grad_inputs, r1_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ r2_ix })) {
        copy_range(grad_inputs, r2_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void NpuRotaryMulBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(r1_, false);
    args.collect(r2_, false);
    args.collect(rotary_mode);
    args.collect(self_, false);
}
variable_list NpuRotaryMulBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(r1_);
    saved.before(r2_);
    saved.before(rotary_mode);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(r1_);
    saved.after(r2_);
    saved.after(rotary_mode);
    saved.after(self_);
    return result;
}
variable_list NpuScaledMaskedSoftmaxBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto x_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto mask = mask_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ x_ix })) {
    auto grad_result = any_grad_defined ? (npu_scaled_masked_softmax_backward(grad, result, mask, scale, fixed_triu_mask)) : Tensor();
    copy_range(grad_inputs, x_ix, grad_result);
  }
  return grad_inputs;
}
void NpuScaledMaskedSoftmaxBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(fixed_triu_mask);
    args.collect(mask_, false);
    args.collect(scale);
    args.collect(result_, true);
}
variable_list NpuScaledMaskedSoftmaxBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(fixed_triu_mask);
    saved.before(mask_);
    saved.before(scale);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(fixed_triu_mask);
    saved.after(mask_);
    saved.after(scale);
    saved.after(result_);
    return result;
}
variable_list NpuSiluBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (npu_silu_backward(grad, self, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void NpuSiluBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_, false);
    args.collect(result_, true);
}
variable_list NpuSiluBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    saved.after(result_);
    return result;
}
variable_list NpuSoftmaxCrossEntropyWithLogitsBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto labels = labels_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (npu_softmax_cross_entropy_with_logits_backward(grad, self, labels)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void NpuSoftmaxCrossEntropyWithLogitsBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(labels_, false);
    args.collect(self_, false);
}
variable_list NpuSoftmaxCrossEntropyWithLogitsBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(labels_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(labels_);
    saved.after(self_);
    return result;
}
variable_list NpuSwigluBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (npu_swiglu_backward(grad, self, dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void NpuSwigluBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(self_, false);
}
variable_list NpuSwigluBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(self_);
    return result;
}
variable_list RepeatInterleaveBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto repeats = repeats_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (repeat_interleave_backward_tensor(grad, self, repeats, dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void RepeatInterleaveBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(repeats_, false);
    args.collect(self_, false);
}
variable_list RepeatInterleaveBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(repeats_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(repeats_);
    saved.after(self_);
    return result;
}
variable_list RepeatInterleaveBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (repeat_interleave_backward_int(grad, self, repeats, dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void RepeatInterleaveBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(repeats);
    args.collect(self_, false);
}
variable_list RepeatInterleaveBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(repeats);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(repeats);
    saved.after(self_);
    return result;
}
variable_list StftBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto window = window_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (stft_backward(grad, self, n_fft, hop_length, win_length, window, normalized, onesided, return_complex)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void StftBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(hop_length);
    args.collect(n_fft);
    args.collect(normalized);
    args.collect(onesided);
    args.collect(return_complex);
    args.collect(self_, false);
    args.collect(win_length);
    args.collect(window_, false);
}
variable_list StftBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(hop_length);
    saved.before(n_fft);
    saved.before(normalized);
    saved.before(onesided);
    saved.before(return_complex);
    saved.before(self_);
    saved.before(win_length);
    saved.before(window_);
    variable_list result = apply(variable_list(grads));
    saved.after(hop_length);
    saved.after(n_fft);
    saved.after(normalized);
    saved.after(onesided);
    saved.after(return_complex);
    saved.after(self_);
    saved.after(win_length);
    saved.after(window_);
    return result;
}
variable_list FftR2CBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (fft_r2c_backward(grad, dim, normalization, onesided, self.size(dim.back()))) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void FftR2CBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(normalization);
    args.collect(onesided);
    args.collect(self_, false);
}
variable_list FftR2CBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(normalization);
    saved.before(onesided);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(normalization);
    saved.after(onesided);
    saved.after(self_);
    return result;
}
variable_list FftC2RBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (fft_c2r_backward(grad, dim, normalization)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void FftC2RBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(normalization);
}
variable_list FftC2RBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(normalization);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(normalization);
    return result;
}
variable_list NpuGroupNormSwishBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto bias = bias_.unpack();
  auto input = input_.unpack();
  auto weight = weight_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  if (task_should_compute_output({ input_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ input_ix }),
        task_should_compute_output({ weight_ix }),
        task_should_compute_output({ bias_ix }),
      };
    auto grad_result = npu_group_norm_swish_grad(grad, input, num_groups, weight, bias, result1, result2, grad_input_mask, swish_scale);
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void NpuGroupNormSwishBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(bias_, false);
    args.collect(input_, false);
    args.collect(num_groups);
    args.collect(swish_scale);
    args.collect(weight_, false);
    args.collect(result1_, true);
    args.collect(result2_, true);
}
variable_list NpuGroupNormSwishBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(bias_);
    saved.before(input_);
    saved.before(num_groups);
    saved.before(swish_scale);
    saved.before(weight_);
    saved.before(result1_);
    saved.before(result2_);
    variable_list result = apply(variable_list(grads));
    saved.after(bias_);
    saved.after(input_);
    saved.after(num_groups);
    saved.after(swish_scale);
    saved.after(weight_);
    saved.after(result1_);
    saved.after(result2_);
    return result;
}
variable_list NpuCrossEntropyLossBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto target = target_.unpack();
  auto weight = weight_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ input_ix })) {
    auto grad_result = any_grad_defined ? (npu_cross_entropy_loss_backward(grads[0], result1, target, weight, grads[1], result3, reduction, ignore_index, label_smoothing, lse_square_scale_for_zloss)) : Tensor();
    copy_range(grad_inputs, input_ix, grad_result);
  }
  return grad_inputs;
}
void NpuCrossEntropyLossBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(ignore_index);
    args.collect(label_smoothing);
    args.collect(lse_square_scale_for_zloss);
    args.collect(reduction);
    args.collect(target_, false);
    args.collect(weight_, false);
    args.collect(result1_, true);
    args.collect(result3_, true);
}
variable_list NpuCrossEntropyLossBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(ignore_index);
    saved.before(label_smoothing);
    saved.before(lse_square_scale_for_zloss);
    saved.before(reduction);
    saved.before(target_);
    saved.before(weight_);
    saved.before(result1_);
    saved.before(result3_);
    variable_list result = apply(variable_list(grads));
    saved.after(ignore_index);
    saved.after(label_smoothing);
    saved.after(lse_square_scale_for_zloss);
    saved.after(reduction);
    saved.after(target_);
    saved.after(weight_);
    saved.after(result1_);
    saved.after(result3_);
    return result;
}
variable_list NpuNsaCompressBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto input = input_.unpack();
  auto weight = weight_.unpack();
  if (task_should_compute_output({ input_ix, weight_ix })) {
  
    auto grad_result = npu_nsa_compress_grad(grad, input, weight, compress_block_size, compress_stride, actual_seq_len);
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
void NpuNsaCompressBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(actual_seq_len);
    args.collect(compress_block_size);
    args.collect(compress_stride);
    args.collect(input_, false);
    args.collect(weight_, false);
}
variable_list NpuNsaCompressBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(actual_seq_len);
    saved.before(compress_block_size);
    saved.before(compress_stride);
    saved.before(input_);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(actual_seq_len);
    saved.after(compress_block_size);
    saved.after(compress_stride);
    saved.after(input_);
    saved.after(weight_);
    return result;
}
variable_list NpuNsaSelectAttentionBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto query_ix = gen.range(1);
  auto key_ix = gen.range(1);
  auto value_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto atten_mask = atten_mask_.unpack();
  auto key = key_.unpack();
  auto query = query_.unpack();
  auto topk_indices = topk_indices_.unpack();
  auto value = value_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  if (task_should_compute_output({ query_ix, key_ix, value_ix })) {
  
    auto grad_result = npu_nsa_select_attention_grad(grad, query, key, value, result0, result1, result2, topk_indices, scale_value, head_num, select_block_size, select_block_count, atten_mask, actual_seq_qlen, actual_seq_kvlen);
      if (task_should_compute_output({ query_ix })) {
        copy_range(grad_inputs, query_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ key_ix })) {
        copy_range(grad_inputs, key_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ value_ix })) {
        copy_range(grad_inputs, value_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void NpuNsaSelectAttentionBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(actual_seq_kvlen);
    args.collect(actual_seq_qlen);
    args.collect(atten_mask_, false);
    args.collect(head_num);
    args.collect(key_, false);
    args.collect(query_, false);
    args.collect(scale_value);
    args.collect(select_block_count);
    args.collect(select_block_size);
    args.collect(topk_indices_, false);
    args.collect(value_, false);
    args.collect(result0_, true);
    args.collect(result1_, true);
    args.collect(result2_, true);
}
variable_list NpuNsaSelectAttentionBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(actual_seq_kvlen);
    saved.before(actual_seq_qlen);
    saved.before(atten_mask_);
    saved.before(head_num);
    saved.before(key_);
    saved.before(query_);
    saved.before(scale_value);
    saved.before(select_block_count);
    saved.before(select_block_size);
    saved.before(topk_indices_);
    saved.before(value_);
    saved.before(result0_);
    saved.before(result1_);
    saved.before(result2_);
    variable_list result = apply(variable_list(grads));
    saved.after(actual_seq_kvlen);
    saved.after(actual_seq_qlen);
    saved.after(atten_mask_);
    saved.after(head_num);
    saved.after(key_);
    saved.after(query_);
    saved.after(scale_value);
    saved.after(select_block_count);
    saved.after(select_block_size);
    saved.after(topk_indices_);
    saved.after(value_);
    saved.after(result0_);
    saved.after(result1_);
    saved.after(result2_);
    return result;
}
variable_list NpuNsaCompressAttentionBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto query_ix = gen.range(1);
  auto key_ix = gen.range(1);
  auto value_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto atten_mask = atten_mask_.unpack();
  auto key = key_.unpack();
  auto query = query_.unpack();
  auto value = value_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  if (task_should_compute_output({ query_ix, key_ix, value_ix })) {
  
    auto grad_result = npu_fusion_attention_grad(query, key, value, grad, head_num, "TND", at::Tensor(), at::Tensor(), atten_mask, result2, result3, at::Tensor(), result0, scale_value, 1., 2147483647, 2147483647, 0, 0, 0, 0, at::IntArrayRef{}, actual_seq_qlen, actual_cmp_seq_kvlen, 1, true, false);
      if (task_should_compute_output({ query_ix })) {
        copy_range(grad_inputs, query_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ key_ix })) {
        copy_range(grad_inputs, key_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ value_ix })) {
        copy_range(grad_inputs, value_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void NpuNsaCompressAttentionBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(actual_cmp_seq_kvlen);
    args.collect(actual_seq_qlen);
    args.collect(atten_mask_, false);
    args.collect(head_num);
    args.collect(key_, false);
    args.collect(query_, false);
    args.collect(scale_value);
    args.collect(value_, false);
    args.collect(result0_, true);
    args.collect(result2_, true);
    args.collect(result3_, true);
}
variable_list NpuNsaCompressAttentionBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(actual_cmp_seq_kvlen);
    saved.before(actual_seq_qlen);
    saved.before(atten_mask_);
    saved.before(head_num);
    saved.before(key_);
    saved.before(query_);
    saved.before(scale_value);
    saved.before(value_);
    saved.before(result0_);
    saved.before(result2_);
    saved.before(result3_);
    variable_list result = apply(variable_list(grads));
    saved.after(actual_cmp_seq_kvlen);
    saved.after(actual_seq_qlen);
    saved.after(atten_mask_);
    saved.after(head_num);
    saved.after(key_);
    saved.after(query_);
    saved.after(scale_value);
    saved.after(value_);
    saved.after(result0_);
    saved.after(result2_);
    saved.after(result3_);
    return result;
}

}}} // namespace at_npu::autograd::generated
