#include <torch/csrc/autograd/VariableTypeUtils.h>

#include <c10/core/impl/TorchDispatchModeTLS.h>
#include <ATen/core/TorchDispatchUtils.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/RedispatchFunctions.h>
#include <torch/library.h>

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/VariableType.h"
#include "torch_npu/csrc/framework/autograd/FunctionsManual.h"
#include "torch_npu/csrc/aten/CustomRedispatch.h"


// @generated from /root/workspace/vllm-ascend/examples/develop/xzh_test/cann-ops-adv-vllm/external/ascend-pytorch-newest/codegen/autograd/templates/VariableType.cpp

// NOTE [Sharded File]: on this file's split-into-shards state
//
// Back in the good old days, VariableType.cpp was generated as one
// file with every function in it, and everything was great and
// simple.
//
// However, this file was also very large (over 36,000 lines), and
// compiling it was very slow, and in fact was a significant
// bottleneck for incremental rebuilds. To address this, we now
// generate the file split across multiple shards, named
// VariableType_0.cpp and so on, which can be compiled in parallel.
//
// For ease of inspection and debugging, so that it's not necessary to
// go rooting around in multiple files, we also generate all the
// functions together in VariableTypeEverything.cpp. This generated
// file is only for convenience; it's not actually used in the
// build. If the file you're looking at now is one of the shards, you
// may want to switch over to the Everything variant to make you
// grepping smoother.

using namespace at;
using namespace at_npu::autograd::generated;
using namespace at_npu::autograd::generated::details;

namespace at_npu { namespace autograd {

namespace VariableType {
namespace {
  C10_UNUSED void reset_grad_accumulator(Variable & self) {
    AutogradMeta* meta = torch::autograd::impl::get_autograd_meta(self);
    if (meta != nullptr) {
      meta->grad_accumulator_.reset();
    }
  }
}

at::Tensor binary_cross_entropy_with_logits(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & pos_weight, int64_t reduction) {
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self );
  
  check_no_requires_grad(weight, "weight", "binary_cross_entropy_with_logits");
  check_no_requires_grad(pos_weight, "pos_weight", "binary_cross_entropy_with_logits");
  std::shared_ptr<BinaryCrossEntropyWithLogitsBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<BinaryCrossEntropyWithLogitsBackward0>(new BinaryCrossEntropyWithLogitsBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->pos_weight_ = SavedVariable(pos_weight, false);
    grad_fn->reduction = reduction;
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->weight_ = SavedVariable(weight, false);
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  auto target__storage_saved =
    target_.has_storage() ? ::std::optional<Storage>(target_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at::redispatch::binary_cross_entropy_with_logits(ks & c10::after_autograd_keyset, self_, target_, weight, pos_weight, reduction);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(target_) &&
      !at::impl::tensor_has_dispatch(target_))
    TORCH_INTERNAL_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(target_))
    TORCH_INTERNAL_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result))
    TORCH_INTERNAL_ASSERT(result.use_count() <= 1, "function: binary_cross_entropy_with_logits");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "binary_cross_entropy_with_logits");
  return result;
}
at::Tensor _fft_r2c(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, bool onesided) {
  auto& self_ = unpack(self, "self", 0);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self );
  
  std::shared_ptr<FftR2CBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<FftR2CBackward0>(new FftR2CBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dim = dim.vec();
    grad_fn->normalization = normalization;
    grad_fn->onesided = onesided;
    grad_fn->self_ = SavedVariable(self, false);
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at::redispatch::_fft_r2c(ks & c10::after_autograd_keyset, self_, dim, normalization, onesided);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result))
    TORCH_INTERNAL_ASSERT(result.use_count() <= 1, "function: _fft_r2c");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
at::Tensor & _fft_r2c_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, bool onesided, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 4);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self );
  
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_fft_r2c");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("_fft_r2c");
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  auto out__storage_saved =
    out_.has_storage() ? ::std::optional<Storage>(out_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::_fft_r2c_outf(ks & c10::after_autograd_keyset, self_, dim, normalization, onesided, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(out_) &&
      !at::impl::tensor_has_dispatch(out_))
    TORCH_INTERNAL_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(out_))
    TORCH_INTERNAL_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!((isFwGradDefined(self) || isFwGradDefined(out))), "Trying to use forward AD with _fft_r2c_out that does not support it because it is an out= function");
  return out;
}
at::Tensor _fft_c2r(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, c10::SymInt last_dim_size) {
  auto& self_ = unpack(self, "self", 0);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self );
  
  std::shared_ptr<FftC2RBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<FftC2RBackward0>(new FftC2RBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dim = dim.vec();
    grad_fn->normalization = normalization;
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at::redispatch::_fft_c2r_symint(ks & c10::after_autograd_keyset, self_, dim, normalization, last_dim_size);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result))
    TORCH_INTERNAL_ASSERT(result.use_count() <= 1, "function: _fft_c2r");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_fft_c2r");
  return result;
}
at::Tensor & _fft_c2r_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, c10::SymInt last_dim_size, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 4);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self );
  
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_fft_c2r");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("_fft_c2r");
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  auto out__storage_saved =
    out_.has_storage() ? ::std::optional<Storage>(out_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::_fft_c2r_symint_outf(ks & c10::after_autograd_keyset, self_, dim, normalization, last_dim_size, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(out_) &&
      !at::impl::tensor_has_dispatch(out_))
    TORCH_INTERNAL_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(out_))
    TORCH_INTERNAL_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!((isFwGradDefined(self) || isFwGradDefined(out))), "Trying to use forward AD with _fft_c2r_out that does not support it because it is an out= function");
  return out;
}
at::Tensor kl_div(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, int64_t reduction, bool log_target) {
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self );
  
  std::shared_ptr<KlDivBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<KlDivBackward0>(new KlDivBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->log_target = log_target;
    grad_fn->reduction = reduction;
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  auto target__storage_saved =
    target_.has_storage() ? ::std::optional<Storage>(target_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at::redispatch::kl_div(ks & c10::after_autograd_keyset, self_, target_, reduction, log_target);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(target_) &&
      !at::impl::tensor_has_dispatch(target_))
    TORCH_INTERNAL_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(target_))
    TORCH_INTERNAL_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result))
    TORCH_INTERNAL_ASSERT(result.use_count() <= 1, "function: kl_div");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "kl_div");
  return result;
}
::std::tuple<at::Tensor,at::Tensor> matmul_backward(c10::DispatchKeySet ks, const at::Tensor & grad_out, const at::Tensor & self, const at::Tensor & other, ::std::array<bool,2> mask) {
  auto& grad_out_ = unpack(grad_out, "grad_out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( grad_out, self, other );
  
  std::shared_ptr<MatmulBackwardBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MatmulBackwardBackward0>(new MatmulBackwardBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_out, self, other ));
    grad_fn->grad_out_ = SavedVariable(grad_out, false);
    grad_fn->other_ = SavedVariable(other, false);
    grad_fn->self_ = SavedVariable(self, false);
  }
  #ifndef NDEBUG
  auto grad_out__storage_saved =
    grad_out_.has_storage() ? ::std::optional<Storage>(grad_out_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_out__impl_saved;
  if (grad_out_.defined()) grad_out__impl_saved = grad_out_.getIntrusivePtr();
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  auto other__storage_saved =
    other_.has_storage() ? ::std::optional<Storage>(other_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at::redispatch::matmul_backward(ks & c10::after_autograd_keyset, grad_out_, self_, other_, mask);
  })();
  auto [result0, result1] = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_out__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(grad_out_) &&
      !at::impl::tensor_has_dispatch(grad_out_))
    TORCH_INTERNAL_ASSERT(grad_out__storage_saved.value().is_alias_of(grad_out_.storage()));
  if (grad_out__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(grad_out_))
    TORCH_INTERNAL_ASSERT(grad_out__impl_saved == grad_out_.getIntrusivePtr());
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(other_) &&
      !at::impl::tensor_has_dispatch(other_))
    TORCH_INTERNAL_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(other_))
    TORCH_INTERNAL_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result0))
    TORCH_INTERNAL_ASSERT(result0.use_count() <= 1, "function: matmul_backward");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result1))
    TORCH_INTERNAL_ASSERT(result1.use_count() <= 1, "function: matmul_backward");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "matmul_backward");
  throw_error_for_complex_autograd(result1, "matmul_backward");
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor repeat_interleave_self_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & repeats, ::std::optional<int64_t> dim, ::std::optional<c10::SymInt> output_size) {
  auto& self_ = unpack(self, "self", 0);
  auto& repeats_ = unpack(repeats, "repeats", 1);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self );
  
  check_no_requires_grad(repeats, "repeats", "repeat_interleave");
  std::shared_ptr<RepeatInterleaveBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<RepeatInterleaveBackward0>(new RepeatInterleaveBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dim = dim;
    grad_fn->repeats_ = SavedVariable(repeats, false);
    grad_fn->self_ = SavedVariable(self, false);
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  auto repeats__storage_saved =
    repeats_.has_storage() ? ::std::optional<Storage>(repeats_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> repeats__impl_saved;
  if (repeats_.defined()) repeats__impl_saved = repeats_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at::redispatch::repeat_interleave_symint(ks & c10::after_autograd_keyset, self_, repeats_, dim, output_size);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (repeats__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(repeats_) &&
      !at::impl::tensor_has_dispatch(repeats_))
    TORCH_INTERNAL_ASSERT(repeats__storage_saved.value().is_alias_of(repeats_.storage()));
  if (repeats__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(repeats_))
    TORCH_INTERNAL_ASSERT(repeats__impl_saved == repeats_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result))
    TORCH_INTERNAL_ASSERT(result.use_count() <= 1, "function: repeat_interleave_self_Tensor");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "repeat_interleave");
  return result;
}
at::Tensor repeat_interleave_self_int(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymInt repeats, ::std::optional<int64_t> dim, ::std::optional<c10::SymInt> output_size) {
  auto& self_ = unpack(self, "self", 0);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self );
  
  std::shared_ptr<RepeatInterleaveBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<RepeatInterleaveBackward1>(new RepeatInterleaveBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dim = dim;
    grad_fn->repeats = repeats;
    grad_fn->self_ = SavedVariable(self, false);
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at::redispatch::repeat_interleave_symint(ks & c10::after_autograd_keyset, self_, repeats, dim, output_size);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result))
    TORCH_INTERNAL_ASSERT(result.use_count() <= 1, "function: repeat_interleave_self_int");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "repeat_interleave");
  return result;
}
at::Tensor stft(c10::DispatchKeySet ks, const at::Tensor & self, int64_t n_fft, ::std::optional<int64_t> hop_length, ::std::optional<int64_t> win_length, const ::std::optional<at::Tensor> & window, bool normalized, ::std::optional<bool> onesided, ::std::optional<bool> return_complex) {
  auto& self_ = unpack(self, "self", 0);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self );
  
  check_no_requires_grad(window, "window", "stft");
  std::shared_ptr<StftBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<StftBackward0>(new StftBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->hop_length = hop_length;
    grad_fn->n_fft = n_fft;
    grad_fn->normalized = normalized;
    grad_fn->onesided = onesided;
    grad_fn->return_complex = return_complex;
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->win_length = win_length;
    grad_fn->window_ = SavedVariable(window, false);
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at::redispatch::stft(ks & c10::after_autograd_keyset, self_, n_fft, hop_length, win_length, window, normalized, onesided, return_complex);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result))
    TORCH_INTERNAL_ASSERT(result.use_count() <= 1, "function: stft");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
at::Tensor & gather_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, bool sparse_grad, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto& out_ = unpack(out, "out", 4);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self );
  
  [[maybe_unused]] auto _any_has_forward_grad_result = (isFwGradDefined(self));
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("gather");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("gather");
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  auto index__storage_saved =
    index_.has_storage() ? ::std::optional<Storage>(index_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> index__impl_saved;
  if (index_.defined()) index__impl_saved = index_.getIntrusivePtr();
  auto out__storage_saved =
    out_.has_storage() ? ::std::optional<Storage>(out_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::gather_outf(ks & c10::after_autograd_keyset, self_, dim, index_, sparse_grad, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (index__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(index_) &&
      !at::impl::tensor_has_dispatch(index_))
    TORCH_INTERNAL_ASSERT(index__storage_saved.value().is_alias_of(index_.storage()));
  if (index__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(index_))
    TORCH_INTERNAL_ASSERT(index__impl_saved == index_.getIntrusivePtr());
  if (out__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(out_) &&
      !at::impl::tensor_has_dispatch(out_))
    TORCH_INTERNAL_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(out_))
    TORCH_INTERNAL_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!((isFwGradDefined(self) || isFwGradDefined(out))), "Trying to use forward AD with gather_out that does not support it because it is an out= function");
  return out;
}
at::Tensor gather(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, bool sparse_grad) {
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self );
  
  [[maybe_unused]] auto _any_has_forward_grad_result = (isFwGradDefined(self));
  std::shared_ptr<GatherBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<GatherBackward0>(new GatherBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dim = dim;
    grad_fn->index_ = SavedVariable(index, false);
    grad_fn->self_sym_sizes = self.sym_sizes().vec();
    grad_fn->sparse_grad = sparse_grad;
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  auto index__storage_saved =
    index_.has_storage() ? ::std::optional<Storage>(index_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> index__impl_saved;
  if (index_.defined()) index__impl_saved = index_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::gather(ks & c10::after_autograd_keyset, self_, dim, index_, sparse_grad);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (index__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(index_) &&
      !at::impl::tensor_has_dispatch(index_))
    TORCH_INTERNAL_ASSERT(index__storage_saved.value().is_alias_of(index_.storage()));
  if (index__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(index_))
    TORCH_INTERNAL_ASSERT(index__impl_saved == index_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result))
    TORCH_INTERNAL_ASSERT(result.use_count() <= 1, "function: gather");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  ::std::optional<at::Tensor> result_new_fw_grad_opt = ::std::nullopt;
  if (_any_has_forward_grad_result && (result.defined())) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_tensor = toNonOptTensor(self);
      auto self_t = (self_t_raw.defined() || !self_tensor.defined())
        ? self_t_raw : at::_efficientzerotensor_symint(self_tensor.sym_sizes(), self_tensor.options());
      result_new_fw_grad_opt = at::gather(self_t, dim, index, sparse_grad);
  }
  if (result_new_fw_grad_opt.has_value() && result_new_fw_grad_opt.value().defined() && result.defined()) {
    // The hardcoded 0 here will need to be updated once we support multiple levels.
    result._set_fw_grad(result_new_fw_grad_opt.value(), /* level */ 0, /* is_inplace_op */ false);
  }
  return result;
}
at::Tensor l1_loss(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self, target );
  
  std::shared_ptr<L1LossBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<L1LossBackward0>(new L1LossBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, target ));
    grad_fn->reduction = reduction;
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  auto target__storage_saved =
    target_.has_storage() ? ::std::optional<Storage>(target_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at::redispatch::l1_loss(ks & c10::after_autograd_keyset, self_, target_, reduction);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(target_) &&
      !at::impl::tensor_has_dispatch(target_))
    TORCH_INTERNAL_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(target_))
    TORCH_INTERNAL_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result))
    TORCH_INTERNAL_ASSERT(result.use_count() <= 1, "function: l1_loss");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "l1_loss");
  return result;
}
at::Tensor _npu_format_cast(c10::DispatchKeySet ks, const at::Tensor & self, int64_t acl_format) {
  auto& self_ = unpack(self, "self", 0);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self );
  
  std::shared_ptr<NpuFormatCastBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuFormatCastBackward0>(new NpuFormatCastBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::_npu_format_cast(ks & c10::after_autograd_keyset, self_, acl_format);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result))
    TORCH_INTERNAL_ASSERT(result.use_count() <= 1, "function: _npu_format_cast");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_npu_format_cast");
  return result;
}
at::Tensor npu_gelu(c10::DispatchKeySet ks, const at::Tensor & self, c10::string_view approximate) {
  auto& self_ = unpack(self, "self", 0);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self );
  
  std::shared_ptr<NpuGeluBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuGeluBackward0>(new NpuGeluBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->approximate = std::string(approximate);
    grad_fn->self_ = SavedVariable(self, false);
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_gelu(ks & c10::after_autograd_keyset, self_, approximate);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result))
    TORCH_INTERNAL_ASSERT(result.use_count() <= 1, "function: npu_gelu");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "npu_gelu");
  return result;
}
::std::tuple<at::Tensor,at::Tensor> _dropout_with_byte_mask(c10::DispatchKeySet ks, const at::Tensor & self, double p) {
  auto& self_ = unpack(self, "self", 0);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self );
  
  std::shared_ptr<DropoutWithByteMaskBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<DropoutWithByteMaskBackward0>(new DropoutWithByteMaskBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->p = p;
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::_dropout_with_byte_mask(ks & c10::after_autograd_keyset, self_, p);
  })();
  auto [result0, result1] = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result0))
    TORCH_INTERNAL_ASSERT(result0.use_count() <= 1, "function: _dropout_with_byte_mask");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result1))
    TORCH_INTERNAL_ASSERT(result1.use_count() <= 1, "function: _dropout_with_byte_mask");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "_dropout_with_byte_mask");
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
::std::tuple<at::Tensor,at::Tensor> _npu_ciou(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & gtboxes, bool trans, bool is_cross, int64_t mode, bool atan_sub_flag) {
  auto& self_ = unpack(self, "self", 0);
  auto& gtboxes_ = unpack(gtboxes, "gtboxes", 1);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self, gtboxes );
  
  std::shared_ptr<NpuCiouBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuCiouBackward0>(new NpuCiouBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, gtboxes ));
    grad_fn->gtboxes_ = SavedVariable(gtboxes, false);
    grad_fn->is_cross = is_cross;
    grad_fn->mode = mode;
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->trans = trans;
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  auto gtboxes__storage_saved =
    gtboxes_.has_storage() ? ::std::optional<Storage>(gtboxes_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> gtboxes__impl_saved;
  if (gtboxes_.defined()) gtboxes__impl_saved = gtboxes_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::_npu_ciou(ks & c10::after_autograd_keyset, self_, gtboxes_, trans, is_cross, mode, atan_sub_flag);
  })();
  auto [result0, result1] = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (gtboxes__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(gtboxes_) &&
      !at::impl::tensor_has_dispatch(gtboxes_))
    TORCH_INTERNAL_ASSERT(gtboxes__storage_saved.value().is_alias_of(gtboxes_.storage()));
  if (gtboxes__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(gtboxes_))
    TORCH_INTERNAL_ASSERT(gtboxes__impl_saved == gtboxes_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result0))
    TORCH_INTERNAL_ASSERT(result0.use_count() <= 1, "function: _npu_ciou");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result1))
    TORCH_INTERNAL_ASSERT(result1.use_count() <= 1, "function: _npu_ciou");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "_npu_ciou");
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
::std::tuple<at::Tensor,at::Tensor> _npu_dropout(c10::DispatchKeySet ks, const at::Tensor & self, double p) {
  auto& self_ = unpack(self, "self", 0);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self );
  
  std::shared_ptr<NpuDropoutBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuDropoutBackward0>(new NpuDropoutBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->p = p;
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::_npu_dropout(ks & c10::after_autograd_keyset, self_, p);
  })();
  auto [result0, result1] = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result0))
    TORCH_INTERNAL_ASSERT(result0.use_count() <= 1, "function: _npu_dropout");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result1))
    TORCH_INTERNAL_ASSERT(result1.use_count() <= 1, "function: _npu_dropout");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "_npu_dropout");
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor fast_gelu(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self );
  
  std::shared_ptr<FastGeluBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<FastGeluBackward0>(new FastGeluBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::fast_gelu(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result))
    TORCH_INTERNAL_ASSERT(result.use_count() <= 1, "function: fast_gelu");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "fast_gelu");
  return result;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_add_layer_norm(c10::DispatchKeySet ks, const at::Tensor & x1, const at::Tensor & x2, const at::Tensor & gamma, const at::Tensor & beta, double epsilon, bool additional_output) {
  auto& x1_ = unpack(x1, "x1", 0);
  auto& x2_ = unpack(x2, "x2", 1);
  auto& gamma_ = unpack(gamma, "gamma", 2);
  auto& beta_ = unpack(beta, "beta", 3);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( x1, x2, gamma, beta );
  
  std::shared_ptr<NpuAddLayerNormBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuAddLayerNormBackward0>(new NpuAddLayerNormBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( x1, x2, gamma, beta ));
    grad_fn->gamma_ = SavedVariable(gamma, false);
    grad_fn->x1_ = SavedVariable(x1, false);
    grad_fn->x2_ = SavedVariable(x2, false);
  }
  #ifndef NDEBUG
  auto x1__storage_saved =
    x1_.has_storage() ? ::std::optional<Storage>(x1_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> x1__impl_saved;
  if (x1_.defined()) x1__impl_saved = x1_.getIntrusivePtr();
  auto x2__storage_saved =
    x2_.has_storage() ? ::std::optional<Storage>(x2_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> x2__impl_saved;
  if (x2_.defined()) x2__impl_saved = x2_.getIntrusivePtr();
  auto gamma__storage_saved =
    gamma_.has_storage() ? ::std::optional<Storage>(gamma_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> gamma__impl_saved;
  if (gamma_.defined()) gamma__impl_saved = gamma_.getIntrusivePtr();
  auto beta__storage_saved =
    beta_.has_storage() ? ::std::optional<Storage>(beta_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> beta__impl_saved;
  if (beta_.defined()) beta__impl_saved = beta_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_add_layer_norm(ks & c10::after_autograd_keyset, x1_, x2_, gamma_, beta_, epsilon, additional_output);
  })();
  auto [result0, result1, result2, result3] = std::move(_tmp);
  #ifndef NDEBUG
  if (x1__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(x1_) &&
      !at::impl::tensor_has_dispatch(x1_))
    TORCH_INTERNAL_ASSERT(x1__storage_saved.value().is_alias_of(x1_.storage()));
  if (x1__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(x1_))
    TORCH_INTERNAL_ASSERT(x1__impl_saved == x1_.getIntrusivePtr());
  if (x2__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(x2_) &&
      !at::impl::tensor_has_dispatch(x2_))
    TORCH_INTERNAL_ASSERT(x2__storage_saved.value().is_alias_of(x2_.storage()));
  if (x2__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(x2_))
    TORCH_INTERNAL_ASSERT(x2__impl_saved == x2_.getIntrusivePtr());
  if (gamma__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(gamma_) &&
      !at::impl::tensor_has_dispatch(gamma_))
    TORCH_INTERNAL_ASSERT(gamma__storage_saved.value().is_alias_of(gamma_.storage()));
  if (gamma__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(gamma_))
    TORCH_INTERNAL_ASSERT(gamma__impl_saved == gamma_.getIntrusivePtr());
  if (beta__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(beta_) &&
      !at::impl::tensor_has_dispatch(beta_))
    TORCH_INTERNAL_ASSERT(beta__storage_saved.value().is_alias_of(beta_.storage()));
  if (beta__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(beta_))
    TORCH_INTERNAL_ASSERT(beta__impl_saved == beta_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result0))
    TORCH_INTERNAL_ASSERT(result0.use_count() <= 1, "function: npu_add_layer_norm");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result1))
    TORCH_INTERNAL_ASSERT(result1.use_count() <= 1, "function: npu_add_layer_norm");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result2))
    TORCH_INTERNAL_ASSERT(result2.use_count() <= 1, "function: npu_add_layer_norm");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result3))
    TORCH_INTERNAL_ASSERT(result3.use_count() <= 1, "function: npu_add_layer_norm");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result3 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "npu_add_layer_norm");
  throw_error_for_complex_autograd(result3, "npu_add_layer_norm");
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
    grad_fn->result2_ = SavedVariable(result2, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
at::Tensor npu_bmmV2(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat2, at::IntArrayRef output_sizes) {
  auto& self_ = unpack(self, "self", 0);
  auto& mat2_ = unpack(mat2, "mat2", 1);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self, mat2 );
  
  std::shared_ptr<NpuBmmv2Backward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuBmmv2Backward0>(new NpuBmmv2Backward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, mat2 ));
    grad_fn->mat2_ = SavedVariable(mat2, false);
    grad_fn->mat2_sym_sizes = mat2.sym_sizes().vec();
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->self_sym_sizes = self.sym_sizes().vec();
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  auto mat2__storage_saved =
    mat2_.has_storage() ? ::std::optional<Storage>(mat2_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> mat2__impl_saved;
  if (mat2_.defined()) mat2__impl_saved = mat2_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_bmmV2(ks & c10::after_autograd_keyset, self_, mat2_, output_sizes);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mat2__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(mat2_) &&
      !at::impl::tensor_has_dispatch(mat2_))
    TORCH_INTERNAL_ASSERT(mat2__storage_saved.value().is_alias_of(mat2_.storage()));
  if (mat2__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(mat2_))
    TORCH_INTERNAL_ASSERT(mat2__impl_saved == mat2_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result))
    TORCH_INTERNAL_ASSERT(result.use_count() <= 1, "function: npu_bmmV2");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "npu_bmmV2");
  return result;
}
at::Tensor npu_confusion_transpose(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef perm, at::IntArrayRef shape, bool transpose_first) {
  auto& self_ = unpack(self, "self", 0);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self );
  
  std::shared_ptr<NpuConfusionTransposeBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuConfusionTransposeBackward0>(new NpuConfusionTransposeBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->perm = perm.vec();
    grad_fn->self_sym_sizes = self.sym_sizes().vec();
    grad_fn->transpose_first = transpose_first;
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_confusion_transpose(ks & c10::after_autograd_keyset, self_, perm, shape, transpose_first);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result))
    TORCH_INTERNAL_ASSERT(result.use_count() <= 1, "function: npu_confusion_transpose");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "npu_confusion_transpose");
  return result;
}
at::Tensor npu_convolution(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
  auto& input_ = unpack(input, "input", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( input, weight, bias );
  
  std::shared_ptr<NpuConvolutionBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuConvolutionBackward0>(new NpuConvolutionBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, weight, bias ));
    grad_fn->dilation = dilation.vec();
    grad_fn->groups = groups;
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->padding = padding.vec();
    grad_fn->stride = stride.vec();
    grad_fn->weight_ = SavedVariable(weight, false);
  }
  #ifndef NDEBUG
  auto input__storage_saved =
    input_.has_storage() ? ::std::optional<Storage>(input_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  auto weight__storage_saved =
    weight_.has_storage() ? ::std::optional<Storage>(weight_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_convolution(ks & c10::after_autograd_keyset, input_, weight_, bias, stride, padding, dilation, groups);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(input_) &&
      !at::impl::tensor_has_dispatch(input_))
    TORCH_INTERNAL_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(input_))
    TORCH_INTERNAL_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (weight__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(weight_) &&
      !at::impl::tensor_has_dispatch(weight_))
    TORCH_INTERNAL_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(weight_))
    TORCH_INTERNAL_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result))
    TORCH_INTERNAL_ASSERT(result.use_count() <= 1, "function: npu_convolution");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "npu_convolution");
  return result;
}
at::Tensor npu_convolution_transpose(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups) {
  auto& input_ = unpack(input, "input", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( input, weight, bias );
  
  std::shared_ptr<NpuConvolutionTransposeBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuConvolutionTransposeBackward0>(new NpuConvolutionTransposeBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, weight, bias ));
    grad_fn->dilation = dilation.vec();
    grad_fn->groups = groups;
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->output_padding = output_padding.vec();
    grad_fn->padding = padding.vec();
    grad_fn->stride = stride.vec();
    grad_fn->weight_ = SavedVariable(weight, false);
  }
  #ifndef NDEBUG
  auto input__storage_saved =
    input_.has_storage() ? ::std::optional<Storage>(input_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  auto weight__storage_saved =
    weight_.has_storage() ? ::std::optional<Storage>(weight_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_convolution_transpose(ks & c10::after_autograd_keyset, input_, weight_, bias, padding, output_padding, stride, dilation, groups);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(input_) &&
      !at::impl::tensor_has_dispatch(input_))
    TORCH_INTERNAL_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(input_))
    TORCH_INTERNAL_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (weight__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(weight_) &&
      !at::impl::tensor_has_dispatch(weight_))
    TORCH_INTERNAL_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(weight_))
    TORCH_INTERNAL_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result))
    TORCH_INTERNAL_ASSERT(result.use_count() <= 1, "function: npu_convolution_transpose");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "npu_convolution_transpose");
  return result;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_deep_norm(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & gx, const at::Tensor & beta, const at::Tensor & gamma, double alpha, double epsilon) {
  auto& x_ = unpack(x, "x", 0);
  auto& gx_ = unpack(gx, "gx", 1);
  auto& beta_ = unpack(beta, "beta", 2);
  auto& gamma_ = unpack(gamma, "gamma", 3);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( x, gx, beta, gamma );
  
  std::shared_ptr<NpuDeepNormBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuDeepNormBackward0>(new NpuDeepNormBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( x, gx, beta, gamma ));
    grad_fn->alpha = alpha;
    grad_fn->gamma_ = SavedVariable(gamma, false);
    grad_fn->gx_ = SavedVariable(gx, false);
    grad_fn->x_ = SavedVariable(x, false);
  }
  #ifndef NDEBUG
  auto x__storage_saved =
    x_.has_storage() ? ::std::optional<Storage>(x_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> x__impl_saved;
  if (x_.defined()) x__impl_saved = x_.getIntrusivePtr();
  auto gx__storage_saved =
    gx_.has_storage() ? ::std::optional<Storage>(gx_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> gx__impl_saved;
  if (gx_.defined()) gx__impl_saved = gx_.getIntrusivePtr();
  auto beta__storage_saved =
    beta_.has_storage() ? ::std::optional<Storage>(beta_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> beta__impl_saved;
  if (beta_.defined()) beta__impl_saved = beta_.getIntrusivePtr();
  auto gamma__storage_saved =
    gamma_.has_storage() ? ::std::optional<Storage>(gamma_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> gamma__impl_saved;
  if (gamma_.defined()) gamma__impl_saved = gamma_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_deep_norm(ks & c10::after_autograd_keyset, x_, gx_, beta_, gamma_, alpha, epsilon);
  })();
  auto [result0, result1, result2] = std::move(_tmp);
  #ifndef NDEBUG
  if (x__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(x_) &&
      !at::impl::tensor_has_dispatch(x_))
    TORCH_INTERNAL_ASSERT(x__storage_saved.value().is_alias_of(x_.storage()));
  if (x__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(x_))
    TORCH_INTERNAL_ASSERT(x__impl_saved == x_.getIntrusivePtr());
  if (gx__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(gx_) &&
      !at::impl::tensor_has_dispatch(gx_))
    TORCH_INTERNAL_ASSERT(gx__storage_saved.value().is_alias_of(gx_.storage()));
  if (gx__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(gx_))
    TORCH_INTERNAL_ASSERT(gx__impl_saved == gx_.getIntrusivePtr());
  if (beta__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(beta_) &&
      !at::impl::tensor_has_dispatch(beta_))
    TORCH_INTERNAL_ASSERT(beta__storage_saved.value().is_alias_of(beta_.storage()));
  if (beta__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(beta_))
    TORCH_INTERNAL_ASSERT(beta__impl_saved == beta_.getIntrusivePtr());
  if (gamma__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(gamma_) &&
      !at::impl::tensor_has_dispatch(gamma_))
    TORCH_INTERNAL_ASSERT(gamma__storage_saved.value().is_alias_of(gamma_.storage()));
  if (gamma__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(gamma_))
    TORCH_INTERNAL_ASSERT(gamma__impl_saved == gamma_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result0))
    TORCH_INTERNAL_ASSERT(result0.use_count() <= 1, "function: npu_deep_norm");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result1))
    TORCH_INTERNAL_ASSERT(result1.use_count() <= 1, "function: npu_deep_norm");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result2))
    TORCH_INTERNAL_ASSERT(result2.use_count() <= 1, "function: npu_deep_norm");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result2 ), grad_fn);
  }
  throw_error_for_complex_autograd(result2, "npu_deep_norm");
  if (grad_fn) {
    grad_fn->result0_ = SavedVariable(result0, true);
    grad_fn->result1_ = SavedVariable(result1, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
::std::tuple<at::Tensor,at::Tensor> npu_deformable_conv2d(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight, const at::Tensor & offset, const ::std::optional<at::Tensor> & bias, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, int64_t deformable_groups, bool modulated) {
  auto& input_ = unpack(input, "input", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto& offset_ = unpack(offset, "offset", 2);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( input, weight, offset, bias );
  
  std::shared_ptr<NpuDeformableConv2DBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuDeformableConv2DBackward0>(new NpuDeformableConv2DBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, weight, offset, bias ));
    grad_fn->deformable_groups = deformable_groups;
    grad_fn->dilation = dilation.vec();
    grad_fn->groups = groups;
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->modulated = modulated;
    grad_fn->offset_ = SavedVariable(offset, false);
    grad_fn->padding = padding.vec();
    grad_fn->stride = stride.vec();
    grad_fn->weight_ = SavedVariable(weight, false);
  }
  #ifndef NDEBUG
  auto input__storage_saved =
    input_.has_storage() ? ::std::optional<Storage>(input_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  auto weight__storage_saved =
    weight_.has_storage() ? ::std::optional<Storage>(weight_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  auto offset__storage_saved =
    offset_.has_storage() ? ::std::optional<Storage>(offset_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> offset__impl_saved;
  if (offset_.defined()) offset__impl_saved = offset_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_deformable_conv2d(ks & c10::after_autograd_keyset, input_, weight_, offset_, bias, kernel_size, stride, padding, dilation, groups, deformable_groups, modulated);
  })();
  auto [result0, result1] = std::move(_tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(input_) &&
      !at::impl::tensor_has_dispatch(input_))
    TORCH_INTERNAL_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(input_))
    TORCH_INTERNAL_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (weight__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(weight_) &&
      !at::impl::tensor_has_dispatch(weight_))
    TORCH_INTERNAL_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(weight_))
    TORCH_INTERNAL_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (offset__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(offset_) &&
      !at::impl::tensor_has_dispatch(offset_))
    TORCH_INTERNAL_ASSERT(offset__storage_saved.value().is_alias_of(offset_.storage()));
  if (offset__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(offset_))
    TORCH_INTERNAL_ASSERT(offset__impl_saved == offset_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result0))
    TORCH_INTERNAL_ASSERT(result0.use_count() <= 1, "function: npu_deformable_conv2d");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result1))
    TORCH_INTERNAL_ASSERT(result1.use_count() <= 1, "function: npu_deformable_conv2d");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "npu_deformable_conv2d");
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor npu_diou(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & gtboxes, bool trans, bool is_cross, int64_t mode) {
  auto& self_ = unpack(self, "self", 0);
  auto& gtboxes_ = unpack(gtboxes, "gtboxes", 1);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self, gtboxes );
  
  std::shared_ptr<NpuDiouBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuDiouBackward0>(new NpuDiouBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, gtboxes ));
    grad_fn->gtboxes_ = SavedVariable(gtboxes, false);
    grad_fn->is_cross = is_cross;
    grad_fn->mode = mode;
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->trans = trans;
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  auto gtboxes__storage_saved =
    gtboxes_.has_storage() ? ::std::optional<Storage>(gtboxes_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> gtboxes__impl_saved;
  if (gtboxes_.defined()) gtboxes__impl_saved = gtboxes_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_diou(ks & c10::after_autograd_keyset, self_, gtboxes_, trans, is_cross, mode);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (gtboxes__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(gtboxes_) &&
      !at::impl::tensor_has_dispatch(gtboxes_))
    TORCH_INTERNAL_ASSERT(gtboxes__storage_saved.value().is_alias_of(gtboxes_.storage()));
  if (gtboxes__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(gtboxes_))
    TORCH_INTERNAL_ASSERT(gtboxes__impl_saved == gtboxes_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result))
    TORCH_INTERNAL_ASSERT(result.use_count() <= 1, "function: npu_diou");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "npu_diou");
  return result;
}
::std::tuple<at::Tensor,at::Tensor> npu_dropout_do_mask(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mask, double p) {
  auto& self_ = unpack(self, "self", 0);
  auto& mask_ = unpack(mask, "mask", 1);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self );
  
  check_no_requires_grad(mask, "mask", "npu_dropout_do_mask");
  std::shared_ptr<NpuDropoutDoMaskBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuDropoutDoMaskBackward0>(new NpuDropoutDoMaskBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->p = p;
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  auto mask__storage_saved =
    mask_.has_storage() ? ::std::optional<Storage>(mask_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> mask__impl_saved;
  if (mask_.defined()) mask__impl_saved = mask_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_dropout_do_mask(ks & c10::after_autograd_keyset, self_, mask_, p);
  })();
  auto [result0, result1] = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mask__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(mask_) &&
      !at::impl::tensor_has_dispatch(mask_))
    TORCH_INTERNAL_ASSERT(mask__storage_saved.value().is_alias_of(mask_.storage()));
  if (mask__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(mask_))
    TORCH_INTERNAL_ASSERT(mask__impl_saved == mask_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result0))
    TORCH_INTERNAL_ASSERT(result0.use_count() <= 1, "function: npu_dropout_do_mask");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result1))
    TORCH_INTERNAL_ASSERT(result1.use_count() <= 1, "function: npu_dropout_do_mask");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "npu_dropout_do_mask");
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_dropout_with_add_softmax(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & x1, const at::Scalar & alpha, double prob, int64_t dim) {
  auto& self_ = unpack(self, "self", 0);
  auto& x1_ = unpack(x1, "x1", 1);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self, x1 );
  
  std::shared_ptr<NpuDropoutWithAddSoftmaxBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuDropoutWithAddSoftmaxBackward0>(new NpuDropoutWithAddSoftmaxBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, x1 ));
    grad_fn->alpha = alpha;
    grad_fn->dim = dim;
    grad_fn->prob = prob;
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  auto x1__storage_saved =
    x1_.has_storage() ? ::std::optional<Storage>(x1_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> x1__impl_saved;
  if (x1_.defined()) x1__impl_saved = x1_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_dropout_with_add_softmax(ks & c10::after_autograd_keyset, self_, x1_, alpha, prob, dim);
  })();
  auto [result0, result1, result2] = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (x1__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(x1_) &&
      !at::impl::tensor_has_dispatch(x1_))
    TORCH_INTERNAL_ASSERT(x1__storage_saved.value().is_alias_of(x1_.storage()));
  if (x1__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(x1_))
    TORCH_INTERNAL_ASSERT(x1__impl_saved == x1_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result0))
    TORCH_INTERNAL_ASSERT(result0.use_count() <= 1, "function: npu_dropout_with_add_softmax");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result1))
    TORCH_INTERNAL_ASSERT(result1.use_count() <= 1, "function: npu_dropout_with_add_softmax");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result2))
    TORCH_INTERNAL_ASSERT(result2.use_count() <= 1, "function: npu_dropout_with_add_softmax");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result2 ), grad_fn);
  }
  throw_error_for_complex_autograd(result2, "npu_dropout_with_add_softmax");
  if (grad_fn) {
    grad_fn->result0_ = SavedVariable(result0, true);
    grad_fn->result1_ = SavedVariable(result1, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
at::Tensor npu_dtype_cast(c10::DispatchKeySet ks, const at::Tensor & self, at::ScalarType dtype) {
  auto& self_ = unpack(self, "self", 0);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self );
  _any_requires_grad &= (isDifferentiableType(dtype));
  std::shared_ptr<NpuDtypeCastBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuDtypeCastBackward0>(new NpuDtypeCastBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_scalar_type = self.scalar_type();
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_dtype_cast(ks & c10::after_autograd_keyset, self_, dtype);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result))
    TORCH_INTERNAL_ASSERT(result.use_count() <= 1, "function: npu_dtype_cast");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_fused_attention_score_fwd(c10::DispatchKeySet ks, const at::Tensor & query_layer, const at::Tensor & key_layer, const at::Tensor & value_layer, const at::Tensor & attention_mask, const at::Scalar & scale, double keep_prob, bool query_transpose, bool key_transpose, bool bmm_score_transpose_a, bool bmm_score_transpose_b, bool value_transpose, bool dx_transpose) {
  auto& query_layer_ = unpack(query_layer, "query_layer", 0);
  auto& key_layer_ = unpack(key_layer, "key_layer", 1);
  auto& value_layer_ = unpack(value_layer, "value_layer", 2);
  auto& attention_mask_ = unpack(attention_mask, "attention_mask", 3);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( query_layer, key_layer, value_layer );
  
  check_no_requires_grad(attention_mask, "attention_mask", "npu_fused_attention_score_fwd");
  std::shared_ptr<NpuFusedAttentionScoreFwdBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuFusedAttentionScoreFwdBackward0>(new NpuFusedAttentionScoreFwdBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( query_layer, key_layer, value_layer ));
    grad_fn->dx_transpose = dx_transpose;
    grad_fn->keep_prob = keep_prob;
    grad_fn->key_layer_ = SavedVariable(key_layer, false);
    grad_fn->key_transpose = key_transpose;
    grad_fn->query_layer_ = SavedVariable(query_layer, false);
    grad_fn->query_transpose = query_transpose;
    grad_fn->scale = scale;
    grad_fn->value_layer_ = SavedVariable(value_layer, false);
    grad_fn->value_transpose = value_transpose;
  }
  #ifndef NDEBUG
  auto query_layer__storage_saved =
    query_layer_.has_storage() ? ::std::optional<Storage>(query_layer_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> query_layer__impl_saved;
  if (query_layer_.defined()) query_layer__impl_saved = query_layer_.getIntrusivePtr();
  auto key_layer__storage_saved =
    key_layer_.has_storage() ? ::std::optional<Storage>(key_layer_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> key_layer__impl_saved;
  if (key_layer_.defined()) key_layer__impl_saved = key_layer_.getIntrusivePtr();
  auto value_layer__storage_saved =
    value_layer_.has_storage() ? ::std::optional<Storage>(value_layer_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> value_layer__impl_saved;
  if (value_layer_.defined()) value_layer__impl_saved = value_layer_.getIntrusivePtr();
  auto attention_mask__storage_saved =
    attention_mask_.has_storage() ? ::std::optional<Storage>(attention_mask_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> attention_mask__impl_saved;
  if (attention_mask_.defined()) attention_mask__impl_saved = attention_mask_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_fused_attention_score_fwd(ks & c10::after_autograd_keyset, query_layer_, key_layer_, value_layer_, attention_mask_, scale, keep_prob, query_transpose, key_transpose, bmm_score_transpose_a, bmm_score_transpose_b, value_transpose, dx_transpose);
  })();
  auto [result0, result1, result2] = std::move(_tmp);
  #ifndef NDEBUG
  if (query_layer__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(query_layer_) &&
      !at::impl::tensor_has_dispatch(query_layer_))
    TORCH_INTERNAL_ASSERT(query_layer__storage_saved.value().is_alias_of(query_layer_.storage()));
  if (query_layer__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(query_layer_))
    TORCH_INTERNAL_ASSERT(query_layer__impl_saved == query_layer_.getIntrusivePtr());
  if (key_layer__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(key_layer_) &&
      !at::impl::tensor_has_dispatch(key_layer_))
    TORCH_INTERNAL_ASSERT(key_layer__storage_saved.value().is_alias_of(key_layer_.storage()));
  if (key_layer__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(key_layer_))
    TORCH_INTERNAL_ASSERT(key_layer__impl_saved == key_layer_.getIntrusivePtr());
  if (value_layer__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(value_layer_) &&
      !at::impl::tensor_has_dispatch(value_layer_))
    TORCH_INTERNAL_ASSERT(value_layer__storage_saved.value().is_alias_of(value_layer_.storage()));
  if (value_layer__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(value_layer_))
    TORCH_INTERNAL_ASSERT(value_layer__impl_saved == value_layer_.getIntrusivePtr());
  if (attention_mask__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(attention_mask_) &&
      !at::impl::tensor_has_dispatch(attention_mask_))
    TORCH_INTERNAL_ASSERT(attention_mask__storage_saved.value().is_alias_of(attention_mask_.storage()));
  if (attention_mask__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(attention_mask_))
    TORCH_INTERNAL_ASSERT(attention_mask__impl_saved == attention_mask_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result0))
    TORCH_INTERNAL_ASSERT(result0.use_count() <= 1, "function: npu_fused_attention_score_fwd");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result1))
    TORCH_INTERNAL_ASSERT(result1.use_count() <= 1, "function: npu_fused_attention_score_fwd");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result2))
    TORCH_INTERNAL_ASSERT(result2.use_count() <= 1, "function: npu_fused_attention_score_fwd");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "npu_fused_attention_score_fwd");
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
    grad_fn->result2_ = SavedVariable(result2, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t,int64_t,int64_t> npu_fusion_attention(c10::DispatchKeySet ks, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, int64_t head_num, c10::string_view input_layout, const ::std::optional<at::Tensor> & pse, const ::std::optional<at::Tensor> & padding_mask, const ::std::optional<at::Tensor> & atten_mask, double scale, double keep_prob, int64_t pre_tockens, int64_t next_tockens, int64_t inner_precise, at::OptionalIntArrayRef prefix, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_seq_kvlen, int64_t sparse_mode, bool gen_mask_parallel, bool sync) {
  auto& query_ = unpack(query, "query", 0);
  auto& key_ = unpack(key, "key", 1);
  auto& value_ = unpack(value, "value", 2);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( query, key, value, pse );
  
  check_no_requires_grad(padding_mask, "padding_mask", "npu_fusion_attention");
  check_no_requires_grad(atten_mask, "atten_mask", "npu_fusion_attention");
  std::shared_ptr<NpuFusionAttentionBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuFusionAttentionBackward0>(new NpuFusionAttentionBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( query, key, value, pse ));
    grad_fn->actual_seq_kvlen = actual_seq_kvlen;
    grad_fn->actual_seq_qlen = actual_seq_qlen;
    grad_fn->atten_mask_ = SavedVariable(atten_mask, false);
    grad_fn->gen_mask_parallel = gen_mask_parallel;
    grad_fn->head_num = head_num;
    grad_fn->inner_precise = inner_precise;
    grad_fn->input_layout = std::string(input_layout);
    grad_fn->keep_prob = keep_prob;
    grad_fn->key_ = SavedVariable(key, false);
    grad_fn->next_tockens = next_tockens;
    grad_fn->padding_mask_ = SavedVariable(padding_mask, false);
    grad_fn->pre_tockens = pre_tockens;
    grad_fn->prefix = prefix;
    grad_fn->pse_ = SavedVariable(pse, false);
    grad_fn->query_ = SavedVariable(query, false);
    grad_fn->scale = scale;
    grad_fn->sparse_mode = sparse_mode;
    grad_fn->sync = sync;
    grad_fn->value_ = SavedVariable(value, false);
  }
  #ifndef NDEBUG
  auto query__storage_saved =
    query_.has_storage() ? ::std::optional<Storage>(query_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> query__impl_saved;
  if (query_.defined()) query__impl_saved = query_.getIntrusivePtr();
  auto key__storage_saved =
    key_.has_storage() ? ::std::optional<Storage>(key_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> key__impl_saved;
  if (key_.defined()) key__impl_saved = key_.getIntrusivePtr();
  auto value__storage_saved =
    value_.has_storage() ? ::std::optional<Storage>(value_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> value__impl_saved;
  if (value_.defined()) value__impl_saved = value_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_fusion_attention(ks & c10::after_autograd_keyset, query_, key_, value_, head_num, input_layout, pse, padding_mask, atten_mask, scale, keep_prob, pre_tockens, next_tockens, inner_precise, prefix, actual_seq_qlen, actual_seq_kvlen, sparse_mode, gen_mask_parallel, sync);
  })();
  auto [result0, result1, result2, result3, result4, result5, result6] = std::move(_tmp);
  #ifndef NDEBUG
  if (query__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(query_) &&
      !at::impl::tensor_has_dispatch(query_))
    TORCH_INTERNAL_ASSERT(query__storage_saved.value().is_alias_of(query_.storage()));
  if (query__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(query_))
    TORCH_INTERNAL_ASSERT(query__impl_saved == query_.getIntrusivePtr());
  if (key__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(key_) &&
      !at::impl::tensor_has_dispatch(key_))
    TORCH_INTERNAL_ASSERT(key__storage_saved.value().is_alias_of(key_.storage()));
  if (key__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(key_))
    TORCH_INTERNAL_ASSERT(key__impl_saved == key_.getIntrusivePtr());
  if (value__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(value_) &&
      !at::impl::tensor_has_dispatch(value_))
    TORCH_INTERNAL_ASSERT(value__storage_saved.value().is_alias_of(value_.storage()));
  if (value__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(value_))
    TORCH_INTERNAL_ASSERT(value__impl_saved == value_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result0))
    TORCH_INTERNAL_ASSERT(result0.use_count() <= 1, "function: npu_fusion_attention");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result1))
    TORCH_INTERNAL_ASSERT(result1.use_count() <= 1, "function: npu_fusion_attention");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result2))
    TORCH_INTERNAL_ASSERT(result2.use_count() <= 1, "function: npu_fusion_attention");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result3))
    TORCH_INTERNAL_ASSERT(result3.use_count() <= 1, "function: npu_fusion_attention");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1, result2, result3 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "npu_fusion_attention");
  throw_error_for_complex_autograd(result1, "npu_fusion_attention");
  throw_error_for_complex_autograd(result2, "npu_fusion_attention");
  throw_error_for_complex_autograd(result3, "npu_fusion_attention");
  if (grad_fn) {
    grad_fn->result0_ = SavedVariable(result0, true);
    grad_fn->result1_ = SavedVariable(result1, true);
    grad_fn->result2_ = SavedVariable(result2, true);
    grad_fn->result3_ = SavedVariable(result3, true);
    grad_fn->result4 = result4;
    grad_fn->result5 = result5;
    grad_fn->result6 = result6;
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3), std::move(result4), std::move(result5), std::move(result6));
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t,int64_t,int64_t> npu_fusion_attention_v2(c10::DispatchKeySet ks, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, int64_t head_num, c10::string_view input_layout, const ::std::optional<at::Tensor> & pse, const ::std::optional<at::Tensor> & padding_mask, const ::std::optional<at::Tensor> & atten_mask, const ::std::optional<at::Tensor> & query_rope, const ::std::optional<at::Tensor> & key_rope, double scale, double keep_prob, int64_t pre_tokens, int64_t next_tokens, int64_t inner_precise, at::OptionalIntArrayRef prefix, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_seq_kvlen, int64_t sparse_mode, bool gen_mask_parallel, bool sync, int64_t pse_type, at::OptionalIntArrayRef q_start_idx, at::OptionalIntArrayRef kv_start_idx) {
  auto& query_ = unpack(query, "query", 0);
  auto& key_ = unpack(key, "key", 1);
  auto& value_ = unpack(value, "value", 2);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( query, key, value, pse, query_rope, key_rope );
  
  check_no_requires_grad(padding_mask, "padding_mask", "npu_fusion_attention_v2");
  check_no_requires_grad(atten_mask, "atten_mask", "npu_fusion_attention_v2");
  std::shared_ptr<NpuFusionAttentionV2Backward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuFusionAttentionV2Backward0>(new NpuFusionAttentionV2Backward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( query, key, value, pse, query_rope, key_rope ));
    grad_fn->actual_seq_kvlen = actual_seq_kvlen;
    grad_fn->actual_seq_qlen = actual_seq_qlen;
    grad_fn->atten_mask_ = SavedVariable(atten_mask, false);
    grad_fn->gen_mask_parallel = gen_mask_parallel;
    grad_fn->head_num = head_num;
    grad_fn->inner_precise = inner_precise;
    grad_fn->input_layout = std::string(input_layout);
    grad_fn->keep_prob = keep_prob;
    grad_fn->key_ = SavedVariable(key, false);
    grad_fn->key_rope_ = SavedVariable(key_rope, false);
    grad_fn->kv_start_idx = kv_start_idx;
    grad_fn->next_tokens = next_tokens;
    grad_fn->padding_mask_ = SavedVariable(padding_mask, false);
    grad_fn->pre_tokens = pre_tokens;
    grad_fn->prefix = prefix;
    grad_fn->pse_ = SavedVariable(pse, false);
    grad_fn->pse_type = pse_type;
    grad_fn->q_start_idx = q_start_idx;
    grad_fn->query_ = SavedVariable(query, false);
    grad_fn->query_rope_ = SavedVariable(query_rope, false);
    grad_fn->scale = scale;
    grad_fn->sparse_mode = sparse_mode;
    grad_fn->sync = sync;
    grad_fn->value_ = SavedVariable(value, false);
  }
  #ifndef NDEBUG
  auto query__storage_saved =
    query_.has_storage() ? ::std::optional<Storage>(query_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> query__impl_saved;
  if (query_.defined()) query__impl_saved = query_.getIntrusivePtr();
  auto key__storage_saved =
    key_.has_storage() ? ::std::optional<Storage>(key_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> key__impl_saved;
  if (key_.defined()) key__impl_saved = key_.getIntrusivePtr();
  auto value__storage_saved =
    value_.has_storage() ? ::std::optional<Storage>(value_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> value__impl_saved;
  if (value_.defined()) value__impl_saved = value_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_fusion_attention_v2(ks & c10::after_autograd_keyset, query_, key_, value_, head_num, input_layout, pse, padding_mask, atten_mask, query_rope, key_rope, scale, keep_prob, pre_tokens, next_tokens, inner_precise, prefix, actual_seq_qlen, actual_seq_kvlen, sparse_mode, gen_mask_parallel, sync, pse_type, q_start_idx, kv_start_idx);
  })();
  auto [result0, result1, result2, result3, result4, result5, result6] = std::move(_tmp);
  #ifndef NDEBUG
  if (query__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(query_) &&
      !at::impl::tensor_has_dispatch(query_))
    TORCH_INTERNAL_ASSERT(query__storage_saved.value().is_alias_of(query_.storage()));
  if (query__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(query_))
    TORCH_INTERNAL_ASSERT(query__impl_saved == query_.getIntrusivePtr());
  if (key__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(key_) &&
      !at::impl::tensor_has_dispatch(key_))
    TORCH_INTERNAL_ASSERT(key__storage_saved.value().is_alias_of(key_.storage()));
  if (key__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(key_))
    TORCH_INTERNAL_ASSERT(key__impl_saved == key_.getIntrusivePtr());
  if (value__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(value_) &&
      !at::impl::tensor_has_dispatch(value_))
    TORCH_INTERNAL_ASSERT(value__storage_saved.value().is_alias_of(value_.storage()));
  if (value__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(value_))
    TORCH_INTERNAL_ASSERT(value__impl_saved == value_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result0))
    TORCH_INTERNAL_ASSERT(result0.use_count() <= 1, "function: npu_fusion_attention_v2");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result1))
    TORCH_INTERNAL_ASSERT(result1.use_count() <= 1, "function: npu_fusion_attention_v2");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result2))
    TORCH_INTERNAL_ASSERT(result2.use_count() <= 1, "function: npu_fusion_attention_v2");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result3))
    TORCH_INTERNAL_ASSERT(result3.use_count() <= 1, "function: npu_fusion_attention_v2");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1, result2, result3 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "npu_fusion_attention_v2");
  throw_error_for_complex_autograd(result1, "npu_fusion_attention_v2");
  throw_error_for_complex_autograd(result2, "npu_fusion_attention_v2");
  throw_error_for_complex_autograd(result3, "npu_fusion_attention_v2");
  if (grad_fn) {
    grad_fn->result0_ = SavedVariable(result0, true);
    grad_fn->result1_ = SavedVariable(result1, true);
    grad_fn->result2_ = SavedVariable(result2, true);
    grad_fn->result3_ = SavedVariable(result3, true);
    grad_fn->result4 = result4;
    grad_fn->result5 = result5;
    grad_fn->result6 = result6;
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3), std::move(result4), std::move(result5), std::move(result6));
}
::std::tuple<at::Tensor,at::Tensor> npu_geglu(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, int64_t approximate, bool activate_left) {
  auto& self_ = unpack(self, "self", 0);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self );
  
  std::shared_ptr<NpuGegluBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuGegluBackward0>(new NpuGegluBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->activate_left = activate_left;
    grad_fn->approximate = approximate;
    grad_fn->dim = dim;
    grad_fn->self_ = SavedVariable(self, false);
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_geglu(ks & c10::after_autograd_keyset, self_, dim, approximate, activate_left);
  })();
  auto [result0, result1] = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result0))
    TORCH_INTERNAL_ASSERT(result0.use_count() <= 1, "function: npu_geglu");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result1))
    TORCH_INTERNAL_ASSERT(result1.use_count() <= 1, "function: npu_geglu");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "npu_geglu");
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor npu_giou(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & gtboxes, bool trans, bool is_cross, int64_t mode) {
  auto& self_ = unpack(self, "self", 0);
  auto& gtboxes_ = unpack(gtboxes, "gtboxes", 1);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self, gtboxes );
  
  std::shared_ptr<NpuGiouBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuGiouBackward0>(new NpuGiouBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, gtboxes ));
    grad_fn->gtboxes_ = SavedVariable(gtboxes, false);
    grad_fn->is_cross = is_cross;
    grad_fn->mode = mode;
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->trans = trans;
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  auto gtboxes__storage_saved =
    gtboxes_.has_storage() ? ::std::optional<Storage>(gtboxes_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> gtboxes__impl_saved;
  if (gtboxes_.defined()) gtboxes__impl_saved = gtboxes_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_giou(ks & c10::after_autograd_keyset, self_, gtboxes_, trans, is_cross, mode);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (gtboxes__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(gtboxes_) &&
      !at::impl::tensor_has_dispatch(gtboxes_))
    TORCH_INTERNAL_ASSERT(gtboxes__storage_saved.value().is_alias_of(gtboxes_.storage()));
  if (gtboxes__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(gtboxes_))
    TORCH_INTERNAL_ASSERT(gtboxes__impl_saved == gtboxes_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result))
    TORCH_INTERNAL_ASSERT(result.use_count() <= 1, "function: npu_giou");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "npu_giou");
  return result;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_gru(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & hx, const at::Tensor & weight_input, const at::Tensor & weight_hidden, const at::Tensor & bias_input, const at::Tensor & bias_hidden, const at::Tensor & seq_length, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
  auto& input_ = unpack(input, "input", 0);
  auto& hx_ = unpack(hx, "hx", 1);
  auto& weight_input_ = unpack(weight_input, "weight_input", 2);
  auto& weight_hidden_ = unpack(weight_hidden, "weight_hidden", 3);
  auto& bias_input_ = unpack(bias_input, "bias_input", 4);
  auto& bias_hidden_ = unpack(bias_hidden, "bias_hidden", 5);
  auto& seq_length_ = unpack(seq_length, "seq_length", 6);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( input, hx, weight_input, weight_hidden, bias_input, bias_hidden );
  
  check_no_requires_grad(seq_length, "seq_length", "npu_gru");
  std::shared_ptr<NpuGruBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuGruBackward0>(new NpuGruBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, hx, weight_input, weight_hidden, bias_input, bias_hidden ));
    grad_fn->bias_hidden_ = SavedVariable(bias_hidden, false);
    grad_fn->bias_input_ = SavedVariable(bias_input, false);
    grad_fn->hx_ = SavedVariable(hx, false);
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->seq_length_ = SavedVariable(seq_length, false);
    grad_fn->weight_hidden_ = SavedVariable(weight_hidden, false);
    grad_fn->weight_input_ = SavedVariable(weight_input, false);
  }
  #ifndef NDEBUG
  auto input__storage_saved =
    input_.has_storage() ? ::std::optional<Storage>(input_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  auto hx__storage_saved =
    hx_.has_storage() ? ::std::optional<Storage>(hx_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> hx__impl_saved;
  if (hx_.defined()) hx__impl_saved = hx_.getIntrusivePtr();
  auto weight_input__storage_saved =
    weight_input_.has_storage() ? ::std::optional<Storage>(weight_input_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> weight_input__impl_saved;
  if (weight_input_.defined()) weight_input__impl_saved = weight_input_.getIntrusivePtr();
  auto weight_hidden__storage_saved =
    weight_hidden_.has_storage() ? ::std::optional<Storage>(weight_hidden_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> weight_hidden__impl_saved;
  if (weight_hidden_.defined()) weight_hidden__impl_saved = weight_hidden_.getIntrusivePtr();
  auto bias_input__storage_saved =
    bias_input_.has_storage() ? ::std::optional<Storage>(bias_input_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> bias_input__impl_saved;
  if (bias_input_.defined()) bias_input__impl_saved = bias_input_.getIntrusivePtr();
  auto bias_hidden__storage_saved =
    bias_hidden_.has_storage() ? ::std::optional<Storage>(bias_hidden_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> bias_hidden__impl_saved;
  if (bias_hidden_.defined()) bias_hidden__impl_saved = bias_hidden_.getIntrusivePtr();
  auto seq_length__storage_saved =
    seq_length_.has_storage() ? ::std::optional<Storage>(seq_length_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> seq_length__impl_saved;
  if (seq_length_.defined()) seq_length__impl_saved = seq_length_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_gru(ks & c10::after_autograd_keyset, input_, hx_, weight_input_, weight_hidden_, bias_input_, bias_hidden_, seq_length_, has_biases, num_layers, dropout, train, bidirectional, batch_first);
  })();
  auto [result0, result1, result2, result3, result4, result5] = std::move(_tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(input_) &&
      !at::impl::tensor_has_dispatch(input_))
    TORCH_INTERNAL_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(input_))
    TORCH_INTERNAL_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (hx__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(hx_) &&
      !at::impl::tensor_has_dispatch(hx_))
    TORCH_INTERNAL_ASSERT(hx__storage_saved.value().is_alias_of(hx_.storage()));
  if (hx__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(hx_))
    TORCH_INTERNAL_ASSERT(hx__impl_saved == hx_.getIntrusivePtr());
  if (weight_input__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(weight_input_) &&
      !at::impl::tensor_has_dispatch(weight_input_))
    TORCH_INTERNAL_ASSERT(weight_input__storage_saved.value().is_alias_of(weight_input_.storage()));
  if (weight_input__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(weight_input_))
    TORCH_INTERNAL_ASSERT(weight_input__impl_saved == weight_input_.getIntrusivePtr());
  if (weight_hidden__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(weight_hidden_) &&
      !at::impl::tensor_has_dispatch(weight_hidden_))
    TORCH_INTERNAL_ASSERT(weight_hidden__storage_saved.value().is_alias_of(weight_hidden_.storage()));
  if (weight_hidden__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(weight_hidden_))
    TORCH_INTERNAL_ASSERT(weight_hidden__impl_saved == weight_hidden_.getIntrusivePtr());
  if (bias_input__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(bias_input_) &&
      !at::impl::tensor_has_dispatch(bias_input_))
    TORCH_INTERNAL_ASSERT(bias_input__storage_saved.value().is_alias_of(bias_input_.storage()));
  if (bias_input__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(bias_input_))
    TORCH_INTERNAL_ASSERT(bias_input__impl_saved == bias_input_.getIntrusivePtr());
  if (bias_hidden__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(bias_hidden_) &&
      !at::impl::tensor_has_dispatch(bias_hidden_))
    TORCH_INTERNAL_ASSERT(bias_hidden__storage_saved.value().is_alias_of(bias_hidden_.storage()));
  if (bias_hidden__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(bias_hidden_))
    TORCH_INTERNAL_ASSERT(bias_hidden__impl_saved == bias_hidden_.getIntrusivePtr());
  if (seq_length__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(seq_length_) &&
      !at::impl::tensor_has_dispatch(seq_length_))
    TORCH_INTERNAL_ASSERT(seq_length__storage_saved.value().is_alias_of(seq_length_.storage()));
  if (seq_length__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(seq_length_))
    TORCH_INTERNAL_ASSERT(seq_length__impl_saved == seq_length_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result0))
    TORCH_INTERNAL_ASSERT(result0.use_count() <= 1, "function: npu_gru");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result1))
    TORCH_INTERNAL_ASSERT(result1.use_count() <= 1, "function: npu_gru");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result2))
    TORCH_INTERNAL_ASSERT(result2.use_count() <= 1, "function: npu_gru");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result3))
    TORCH_INTERNAL_ASSERT(result3.use_count() <= 1, "function: npu_gru");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result4))
    TORCH_INTERNAL_ASSERT(result4.use_count() <= 1, "function: npu_gru");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result5))
    TORCH_INTERNAL_ASSERT(result5.use_count() <= 1, "function: npu_gru");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "npu_gru");
  throw_error_for_complex_autograd(result1, "npu_gru");
  if (grad_fn) {
    grad_fn->result0_ = SavedVariable(result0, true);
    grad_fn->result1_ = SavedVariable(result1, true);
    grad_fn->result2_ = SavedVariable(result2, true);
    grad_fn->result3_ = SavedVariable(result3, true);
    grad_fn->result4_ = SavedVariable(result4, true);
    grad_fn->result5_ = SavedVariable(result5, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3), std::move(result4), std::move(result5));
}
at::Tensor npu_linear(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias) {
  auto& input_ = unpack(input, "input", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( input, weight, bias );
  
  std::shared_ptr<NpuLinearBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuLinearBackward0>(new NpuLinearBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, weight, bias ));
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->weight_ = SavedVariable(weight, false);
  }
  #ifndef NDEBUG
  auto input__storage_saved =
    input_.has_storage() ? ::std::optional<Storage>(input_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  auto weight__storage_saved =
    weight_.has_storage() ? ::std::optional<Storage>(weight_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_linear(ks & c10::after_autograd_keyset, input_, weight_, bias);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(input_) &&
      !at::impl::tensor_has_dispatch(input_))
    TORCH_INTERNAL_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(input_))
    TORCH_INTERNAL_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (weight__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(weight_) &&
      !at::impl::tensor_has_dispatch(weight_))
    TORCH_INTERNAL_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(weight_))
    TORCH_INTERNAL_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result))
    TORCH_INTERNAL_ASSERT(result.use_count() <= 1, "function: npu_linear");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "npu_linear");
  return result;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_lstm(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, const at::Tensor & seq_mask, const at::Tensor & h, const at::Tensor & c, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first, bool flag_seq, bool direction) {
  auto& input_ = unpack(input, "input", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto& bias_ = unpack(bias, "bias", 2);
  auto& seq_mask_ = unpack(seq_mask, "seq_mask", 3);
  auto& h_ = unpack(h, "h", 4);
  auto& c_ = unpack(c, "c", 5);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( input, weight, bias, h, c );
  
  check_no_requires_grad(seq_mask, "seq_mask", "npu_lstm");
  std::shared_ptr<NpuLstmBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuLstmBackward0>(new NpuLstmBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, weight, bias, h, c ));
    grad_fn->bias_ = SavedVariable(bias, false);
    grad_fn->c_ = SavedVariable(c, false);
    grad_fn->h_ = SavedVariable(h, false);
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->weight_ = SavedVariable(weight, false);
  }
  #ifndef NDEBUG
  auto input__storage_saved =
    input_.has_storage() ? ::std::optional<Storage>(input_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  auto weight__storage_saved =
    weight_.has_storage() ? ::std::optional<Storage>(weight_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  auto bias__storage_saved =
    bias_.has_storage() ? ::std::optional<Storage>(bias_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> bias__impl_saved;
  if (bias_.defined()) bias__impl_saved = bias_.getIntrusivePtr();
  auto seq_mask__storage_saved =
    seq_mask_.has_storage() ? ::std::optional<Storage>(seq_mask_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> seq_mask__impl_saved;
  if (seq_mask_.defined()) seq_mask__impl_saved = seq_mask_.getIntrusivePtr();
  auto h__storage_saved =
    h_.has_storage() ? ::std::optional<Storage>(h_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> h__impl_saved;
  if (h_.defined()) h__impl_saved = h_.getIntrusivePtr();
  auto c__storage_saved =
    c_.has_storage() ? ::std::optional<Storage>(c_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> c__impl_saved;
  if (c_.defined()) c__impl_saved = c_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_lstm(ks & c10::after_autograd_keyset, input_, weight_, bias_, seq_mask_, h_, c_, has_biases, num_layers, dropout, train, bidirectional, batch_first, flag_seq, direction);
  })();
  auto [result0, result1, result2, result3, result4, result5, result6, result7] = std::move(_tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(input_) &&
      !at::impl::tensor_has_dispatch(input_))
    TORCH_INTERNAL_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(input_))
    TORCH_INTERNAL_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (weight__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(weight_) &&
      !at::impl::tensor_has_dispatch(weight_))
    TORCH_INTERNAL_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(weight_))
    TORCH_INTERNAL_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (bias__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(bias_) &&
      !at::impl::tensor_has_dispatch(bias_))
    TORCH_INTERNAL_ASSERT(bias__storage_saved.value().is_alias_of(bias_.storage()));
  if (bias__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(bias_))
    TORCH_INTERNAL_ASSERT(bias__impl_saved == bias_.getIntrusivePtr());
  if (seq_mask__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(seq_mask_) &&
      !at::impl::tensor_has_dispatch(seq_mask_))
    TORCH_INTERNAL_ASSERT(seq_mask__storage_saved.value().is_alias_of(seq_mask_.storage()));
  if (seq_mask__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(seq_mask_))
    TORCH_INTERNAL_ASSERT(seq_mask__impl_saved == seq_mask_.getIntrusivePtr());
  if (h__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(h_) &&
      !at::impl::tensor_has_dispatch(h_))
    TORCH_INTERNAL_ASSERT(h__storage_saved.value().is_alias_of(h_.storage()));
  if (h__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(h_))
    TORCH_INTERNAL_ASSERT(h__impl_saved == h_.getIntrusivePtr());
  if (c__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(c_) &&
      !at::impl::tensor_has_dispatch(c_))
    TORCH_INTERNAL_ASSERT(c__storage_saved.value().is_alias_of(c_.storage()));
  if (c__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(c_))
    TORCH_INTERNAL_ASSERT(c__impl_saved == c_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result0))
    TORCH_INTERNAL_ASSERT(result0.use_count() <= 1, "function: npu_lstm");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result1))
    TORCH_INTERNAL_ASSERT(result1.use_count() <= 1, "function: npu_lstm");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result2))
    TORCH_INTERNAL_ASSERT(result2.use_count() <= 1, "function: npu_lstm");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result3))
    TORCH_INTERNAL_ASSERT(result3.use_count() <= 1, "function: npu_lstm");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result4))
    TORCH_INTERNAL_ASSERT(result4.use_count() <= 1, "function: npu_lstm");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result5))
    TORCH_INTERNAL_ASSERT(result5.use_count() <= 1, "function: npu_lstm");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result6))
    TORCH_INTERNAL_ASSERT(result6.use_count() <= 1, "function: npu_lstm");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result7))
    TORCH_INTERNAL_ASSERT(result7.use_count() <= 1, "function: npu_lstm");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1, result2 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "npu_lstm");
  throw_error_for_complex_autograd(result1, "npu_lstm");
  throw_error_for_complex_autograd(result2, "npu_lstm");
  if (grad_fn) {
    grad_fn->result0_ = SavedVariable(result0, true);
    grad_fn->result1_ = SavedVariable(result1, true);
    grad_fn->result2_ = SavedVariable(result2, true);
    grad_fn->result3_ = SavedVariable(result3, true);
    grad_fn->result4_ = SavedVariable(result4, true);
    grad_fn->result5_ = SavedVariable(result5, true);
    grad_fn->result6_ = SavedVariable(result6, true);
    grad_fn->result7_ = SavedVariable(result7, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3), std::move(result4), std::move(result5), std::move(result6), std::move(result7));
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_lstm_cell(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & w_ih, const at::Tensor & w_hh, const at::Tensor & h, const at::Tensor & c, const ::std::optional<at::Tensor> & b_ih, const ::std::optional<at::Tensor> & b_hh) {
  auto& input_ = unpack(input, "input", 0);
  auto& w_ih_ = unpack(w_ih, "w_ih", 1);
  auto& w_hh_ = unpack(w_hh, "w_hh", 2);
  auto& h_ = unpack(h, "h", 3);
  auto& c_ = unpack(c, "c", 4);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( input, w_ih, w_hh, h, c, b_ih, b_hh );
  
  std::shared_ptr<NpuLstmCellBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuLstmCellBackward0>(new NpuLstmCellBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, w_ih, w_hh, h, c, b_ih, b_hh ));
    grad_fn->c_ = SavedVariable(c, false);
    grad_fn->h_ = SavedVariable(h, false);
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->w_hh_ = SavedVariable(w_hh, false);
    grad_fn->w_ih_ = SavedVariable(w_ih, false);
  }
  #ifndef NDEBUG
  auto input__storage_saved =
    input_.has_storage() ? ::std::optional<Storage>(input_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  auto w_ih__storage_saved =
    w_ih_.has_storage() ? ::std::optional<Storage>(w_ih_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> w_ih__impl_saved;
  if (w_ih_.defined()) w_ih__impl_saved = w_ih_.getIntrusivePtr();
  auto w_hh__storage_saved =
    w_hh_.has_storage() ? ::std::optional<Storage>(w_hh_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> w_hh__impl_saved;
  if (w_hh_.defined()) w_hh__impl_saved = w_hh_.getIntrusivePtr();
  auto h__storage_saved =
    h_.has_storage() ? ::std::optional<Storage>(h_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> h__impl_saved;
  if (h_.defined()) h__impl_saved = h_.getIntrusivePtr();
  auto c__storage_saved =
    c_.has_storage() ? ::std::optional<Storage>(c_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> c__impl_saved;
  if (c_.defined()) c__impl_saved = c_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_lstm_cell(ks & c10::after_autograd_keyset, input_, w_ih_, w_hh_, h_, c_, b_ih, b_hh);
  })();
  auto [result0, result1, result2, result3, result4, result5, result6, result7] = std::move(_tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(input_) &&
      !at::impl::tensor_has_dispatch(input_))
    TORCH_INTERNAL_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(input_))
    TORCH_INTERNAL_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (w_ih__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(w_ih_) &&
      !at::impl::tensor_has_dispatch(w_ih_))
    TORCH_INTERNAL_ASSERT(w_ih__storage_saved.value().is_alias_of(w_ih_.storage()));
  if (w_ih__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(w_ih_))
    TORCH_INTERNAL_ASSERT(w_ih__impl_saved == w_ih_.getIntrusivePtr());
  if (w_hh__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(w_hh_) &&
      !at::impl::tensor_has_dispatch(w_hh_))
    TORCH_INTERNAL_ASSERT(w_hh__storage_saved.value().is_alias_of(w_hh_.storage()));
  if (w_hh__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(w_hh_))
    TORCH_INTERNAL_ASSERT(w_hh__impl_saved == w_hh_.getIntrusivePtr());
  if (h__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(h_) &&
      !at::impl::tensor_has_dispatch(h_))
    TORCH_INTERNAL_ASSERT(h__storage_saved.value().is_alias_of(h_.storage()));
  if (h__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(h_))
    TORCH_INTERNAL_ASSERT(h__impl_saved == h_.getIntrusivePtr());
  if (c__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(c_) &&
      !at::impl::tensor_has_dispatch(c_))
    TORCH_INTERNAL_ASSERT(c__storage_saved.value().is_alias_of(c_.storage()));
  if (c__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(c_))
    TORCH_INTERNAL_ASSERT(c__impl_saved == c_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result0))
    TORCH_INTERNAL_ASSERT(result0.use_count() <= 1, "function: npu_lstm_cell");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result1))
    TORCH_INTERNAL_ASSERT(result1.use_count() <= 1, "function: npu_lstm_cell");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result2))
    TORCH_INTERNAL_ASSERT(result2.use_count() <= 1, "function: npu_lstm_cell");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result3))
    TORCH_INTERNAL_ASSERT(result3.use_count() <= 1, "function: npu_lstm_cell");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result4))
    TORCH_INTERNAL_ASSERT(result4.use_count() <= 1, "function: npu_lstm_cell");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result5))
    TORCH_INTERNAL_ASSERT(result5.use_count() <= 1, "function: npu_lstm_cell");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result6))
    TORCH_INTERNAL_ASSERT(result6.use_count() <= 1, "function: npu_lstm_cell");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result7))
    TORCH_INTERNAL_ASSERT(result7.use_count() <= 1, "function: npu_lstm_cell");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1, result2 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "npu_lstm_cell");
  throw_error_for_complex_autograd(result1, "npu_lstm_cell");
  throw_error_for_complex_autograd(result2, "npu_lstm_cell");
  if (grad_fn) {
    grad_fn->result0_ = SavedVariable(result0, true);
    grad_fn->result1_ = SavedVariable(result1, true);
    grad_fn->result2_ = SavedVariable(result2, true);
    grad_fn->result3_ = SavedVariable(result3, true);
    grad_fn->result4_ = SavedVariable(result4, true);
    grad_fn->result5_ = SavedVariable(result5, true);
    grad_fn->result6_ = SavedVariable(result6, true);
    grad_fn->result7_ = SavedVariable(result7, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3), std::move(result4), std::move(result5), std::move(result6), std::move(result7));
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_lstm_data(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & batch_sizes, const at::Tensor & weight, const at::Tensor & bias, const at::Tensor & seq_mask, const at::Tensor & h, const at::Tensor & c, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first, bool flag_seq, bool direction) {
  auto& input_ = unpack(input, "input", 0);
  auto& batch_sizes_ = unpack(batch_sizes, "batch_sizes", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto& bias_ = unpack(bias, "bias", 3);
  auto& seq_mask_ = unpack(seq_mask, "seq_mask", 4);
  auto& h_ = unpack(h, "h", 5);
  auto& c_ = unpack(c, "c", 6);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( input, weight, bias, h, c );
  
  check_no_requires_grad(batch_sizes, "batch_sizes", "npu_lstm_data");
  check_no_requires_grad(seq_mask, "seq_mask", "npu_lstm_data");
  std::shared_ptr<NpuLstmDataBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuLstmDataBackward0>(new NpuLstmDataBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, weight, bias, h, c ));
    grad_fn->batch_sizes_ = SavedVariable(batch_sizes, false);
    grad_fn->bias_ = SavedVariable(bias, false);
    grad_fn->c_ = SavedVariable(c, false);
    grad_fn->direction = direction;
    grad_fn->h_ = SavedVariable(h, false);
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->weight_ = SavedVariable(weight, false);
  }
  #ifndef NDEBUG
  auto input__storage_saved =
    input_.has_storage() ? ::std::optional<Storage>(input_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  auto batch_sizes__storage_saved =
    batch_sizes_.has_storage() ? ::std::optional<Storage>(batch_sizes_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> batch_sizes__impl_saved;
  if (batch_sizes_.defined()) batch_sizes__impl_saved = batch_sizes_.getIntrusivePtr();
  auto weight__storage_saved =
    weight_.has_storage() ? ::std::optional<Storage>(weight_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  auto bias__storage_saved =
    bias_.has_storage() ? ::std::optional<Storage>(bias_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> bias__impl_saved;
  if (bias_.defined()) bias__impl_saved = bias_.getIntrusivePtr();
  auto seq_mask__storage_saved =
    seq_mask_.has_storage() ? ::std::optional<Storage>(seq_mask_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> seq_mask__impl_saved;
  if (seq_mask_.defined()) seq_mask__impl_saved = seq_mask_.getIntrusivePtr();
  auto h__storage_saved =
    h_.has_storage() ? ::std::optional<Storage>(h_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> h__impl_saved;
  if (h_.defined()) h__impl_saved = h_.getIntrusivePtr();
  auto c__storage_saved =
    c_.has_storage() ? ::std::optional<Storage>(c_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> c__impl_saved;
  if (c_.defined()) c__impl_saved = c_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_lstm_data(ks & c10::after_autograd_keyset, input_, batch_sizes_, weight_, bias_, seq_mask_, h_, c_, has_biases, num_layers, dropout, train, bidirectional, batch_first, flag_seq, direction);
  })();
  auto [result0, result1, result2, result3, result4, result5, result6, result7] = std::move(_tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(input_) &&
      !at::impl::tensor_has_dispatch(input_))
    TORCH_INTERNAL_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(input_))
    TORCH_INTERNAL_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (batch_sizes__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(batch_sizes_) &&
      !at::impl::tensor_has_dispatch(batch_sizes_))
    TORCH_INTERNAL_ASSERT(batch_sizes__storage_saved.value().is_alias_of(batch_sizes_.storage()));
  if (batch_sizes__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(batch_sizes_))
    TORCH_INTERNAL_ASSERT(batch_sizes__impl_saved == batch_sizes_.getIntrusivePtr());
  if (weight__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(weight_) &&
      !at::impl::tensor_has_dispatch(weight_))
    TORCH_INTERNAL_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(weight_))
    TORCH_INTERNAL_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (bias__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(bias_) &&
      !at::impl::tensor_has_dispatch(bias_))
    TORCH_INTERNAL_ASSERT(bias__storage_saved.value().is_alias_of(bias_.storage()));
  if (bias__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(bias_))
    TORCH_INTERNAL_ASSERT(bias__impl_saved == bias_.getIntrusivePtr());
  if (seq_mask__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(seq_mask_) &&
      !at::impl::tensor_has_dispatch(seq_mask_))
    TORCH_INTERNAL_ASSERT(seq_mask__storage_saved.value().is_alias_of(seq_mask_.storage()));
  if (seq_mask__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(seq_mask_))
    TORCH_INTERNAL_ASSERT(seq_mask__impl_saved == seq_mask_.getIntrusivePtr());
  if (h__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(h_) &&
      !at::impl::tensor_has_dispatch(h_))
    TORCH_INTERNAL_ASSERT(h__storage_saved.value().is_alias_of(h_.storage()));
  if (h__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(h_))
    TORCH_INTERNAL_ASSERT(h__impl_saved == h_.getIntrusivePtr());
  if (c__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(c_) &&
      !at::impl::tensor_has_dispatch(c_))
    TORCH_INTERNAL_ASSERT(c__storage_saved.value().is_alias_of(c_.storage()));
  if (c__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(c_))
    TORCH_INTERNAL_ASSERT(c__impl_saved == c_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result0))
    TORCH_INTERNAL_ASSERT(result0.use_count() <= 1, "function: npu_lstm_data");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result1))
    TORCH_INTERNAL_ASSERT(result1.use_count() <= 1, "function: npu_lstm_data");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result2))
    TORCH_INTERNAL_ASSERT(result2.use_count() <= 1, "function: npu_lstm_data");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result3))
    TORCH_INTERNAL_ASSERT(result3.use_count() <= 1, "function: npu_lstm_data");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result4))
    TORCH_INTERNAL_ASSERT(result4.use_count() <= 1, "function: npu_lstm_data");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result5))
    TORCH_INTERNAL_ASSERT(result5.use_count() <= 1, "function: npu_lstm_data");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result6))
    TORCH_INTERNAL_ASSERT(result6.use_count() <= 1, "function: npu_lstm_data");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result7))
    TORCH_INTERNAL_ASSERT(result7.use_count() <= 1, "function: npu_lstm_data");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1, result2 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "npu_lstm_data");
  throw_error_for_complex_autograd(result1, "npu_lstm_data");
  throw_error_for_complex_autograd(result2, "npu_lstm_data");
  if (grad_fn) {
    grad_fn->result0_ = SavedVariable(result0, true);
    grad_fn->result1_ = SavedVariable(result1, true);
    grad_fn->result2_ = SavedVariable(result2, true);
    grad_fn->result3_ = SavedVariable(result3, true);
    grad_fn->result4_ = SavedVariable(result4, true);
    grad_fn->result5_ = SavedVariable(result5, true);
    grad_fn->result6_ = SavedVariable(result6, true);
    grad_fn->result7_ = SavedVariable(result7, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3), std::move(result4), std::move(result5), std::move(result6), std::move(result7));
}
::std::tuple<at::Tensor,at::Tensor> npu_max_dim(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim) {
  auto& self_ = unpack(self, "self", 0);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self );
  
  std::shared_ptr<NpuMaxBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuMaxBackward0>(new NpuMaxBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dim = dim;
    grad_fn->keepdim = keepdim;
    grad_fn->self_sym_sizes = self.sym_sizes().vec();
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_max(ks & c10::after_autograd_keyset, self_, dim, keepdim);
  })();
  auto [values, indices] = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(values))
    TORCH_INTERNAL_ASSERT(values.use_count() <= 1, "function: npu_max_dim");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(indices))
    TORCH_INTERNAL_ASSERT(indices.use_count() <= 1, "function: npu_max_dim");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( values ), grad_fn);
  }
  throw_error_for_complex_autograd(values, "npu_max");
  if (grad_fn) {
    grad_fn->indices_ = SavedVariable(indices, true);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
::std::tuple<at::Tensor,at::Tensor> npu_min_dim(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim) {
  auto& self_ = unpack(self, "self", 0);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self );
  
  std::shared_ptr<NpuMinBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuMinBackward0>(new NpuMinBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dim = dim;
    grad_fn->keepdim = keepdim;
    grad_fn->self_sym_sizes = self.sym_sizes().vec();
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_min(ks & c10::after_autograd_keyset, self_, dim, keepdim);
  })();
  auto [values, indices] = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(values))
    TORCH_INTERNAL_ASSERT(values.use_count() <= 1, "function: npu_min_dim");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(indices))
    TORCH_INTERNAL_ASSERT(indices.use_count() <= 1, "function: npu_min_dim");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( values ), grad_fn);
  }
  throw_error_for_complex_autograd(values, "npu_min");
  if (grad_fn) {
    grad_fn->indices_ = SavedVariable(indices, true);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
at::Tensor npu_mish(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self );
  
  std::shared_ptr<NpuMishBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuMishBackward0>(new NpuMishBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_mish(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result))
    TORCH_INTERNAL_ASSERT(result.use_count() <= 1, "function: npu_mish");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "npu_mish");
  return result;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_multi_head_attention(c10::DispatchKeySet ks, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & query_weight, const at::Tensor & key_weight, const at::Tensor & value_weight, const at::Tensor & attn_mask, const at::Tensor & out_proj_weight, const ::std::optional<at::Tensor> & query_bias, const ::std::optional<at::Tensor> & key_bias, const ::std::optional<at::Tensor> & value_bias, const ::std::optional<at::Tensor> & out_proj_bias, const ::std::optional<at::Tensor> & dropout_mask, int64_t attn_head_num, int64_t attn_dim_per_head, int64_t src_len, int64_t tgt_len, double dropout_prob, bool softmax_use_float) {
  auto& query_ = unpack(query, "query", 0);
  auto& key_ = unpack(key, "key", 1);
  auto& value_ = unpack(value, "value", 2);
  auto& query_weight_ = unpack(query_weight, "query_weight", 3);
  auto& key_weight_ = unpack(key_weight, "key_weight", 4);
  auto& value_weight_ = unpack(value_weight, "value_weight", 5);
  auto& attn_mask_ = unpack(attn_mask, "attn_mask", 6);
  auto& out_proj_weight_ = unpack(out_proj_weight, "out_proj_weight", 7);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( query, key, value, query_weight, key_weight, value_weight, out_proj_weight, query_bias, key_bias, value_bias, out_proj_bias );
  
  check_no_requires_grad(attn_mask, "attn_mask", "npu_multi_head_attention");
  check_no_requires_grad(dropout_mask, "dropout_mask", "npu_multi_head_attention");
  std::shared_ptr<NpuMultiHeadAttentionBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuMultiHeadAttentionBackward0>(new NpuMultiHeadAttentionBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( query, key, value, query_weight, key_weight, value_weight, out_proj_weight, query_bias, key_bias, value_bias, out_proj_bias ));
    grad_fn->attn_dim_per_head = attn_dim_per_head;
    grad_fn->attn_head_num = attn_head_num;
    grad_fn->dropout_prob = dropout_prob;
    grad_fn->key_ = SavedVariable(key, false);
    grad_fn->key_bias_ = SavedVariable(key_bias, false);
    grad_fn->key_weight_ = SavedVariable(key_weight, false);
    grad_fn->out_proj_bias_ = SavedVariable(out_proj_bias, false);
    grad_fn->out_proj_weight_ = SavedVariable(out_proj_weight, false);
    grad_fn->query_ = SavedVariable(query, false);
    grad_fn->query_bias_ = SavedVariable(query_bias, false);
    grad_fn->query_weight_ = SavedVariable(query_weight, false);
    grad_fn->softmax_use_float = softmax_use_float;
    grad_fn->src_len = src_len;
    grad_fn->tgt_len = tgt_len;
    grad_fn->value_ = SavedVariable(value, false);
    grad_fn->value_bias_ = SavedVariable(value_bias, false);
    grad_fn->value_weight_ = SavedVariable(value_weight, false);
  }
  #ifndef NDEBUG
  auto query__storage_saved =
    query_.has_storage() ? ::std::optional<Storage>(query_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> query__impl_saved;
  if (query_.defined()) query__impl_saved = query_.getIntrusivePtr();
  auto key__storage_saved =
    key_.has_storage() ? ::std::optional<Storage>(key_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> key__impl_saved;
  if (key_.defined()) key__impl_saved = key_.getIntrusivePtr();
  auto value__storage_saved =
    value_.has_storage() ? ::std::optional<Storage>(value_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> value__impl_saved;
  if (value_.defined()) value__impl_saved = value_.getIntrusivePtr();
  auto query_weight__storage_saved =
    query_weight_.has_storage() ? ::std::optional<Storage>(query_weight_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> query_weight__impl_saved;
  if (query_weight_.defined()) query_weight__impl_saved = query_weight_.getIntrusivePtr();
  auto key_weight__storage_saved =
    key_weight_.has_storage() ? ::std::optional<Storage>(key_weight_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> key_weight__impl_saved;
  if (key_weight_.defined()) key_weight__impl_saved = key_weight_.getIntrusivePtr();
  auto value_weight__storage_saved =
    value_weight_.has_storage() ? ::std::optional<Storage>(value_weight_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> value_weight__impl_saved;
  if (value_weight_.defined()) value_weight__impl_saved = value_weight_.getIntrusivePtr();
  auto attn_mask__storage_saved =
    attn_mask_.has_storage() ? ::std::optional<Storage>(attn_mask_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> attn_mask__impl_saved;
  if (attn_mask_.defined()) attn_mask__impl_saved = attn_mask_.getIntrusivePtr();
  auto out_proj_weight__storage_saved =
    out_proj_weight_.has_storage() ? ::std::optional<Storage>(out_proj_weight_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> out_proj_weight__impl_saved;
  if (out_proj_weight_.defined()) out_proj_weight__impl_saved = out_proj_weight_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_multi_head_attention(ks & c10::after_autograd_keyset, query_, key_, value_, query_weight_, key_weight_, value_weight_, attn_mask_, out_proj_weight_, query_bias, key_bias, value_bias, out_proj_bias, dropout_mask, attn_head_num, attn_dim_per_head, src_len, tgt_len, dropout_prob, softmax_use_float);
  })();
  auto [result0, result1, result2, result3, result4, result5, result6, result7] = std::move(_tmp);
  #ifndef NDEBUG
  if (query__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(query_) &&
      !at::impl::tensor_has_dispatch(query_))
    TORCH_INTERNAL_ASSERT(query__storage_saved.value().is_alias_of(query_.storage()));
  if (query__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(query_))
    TORCH_INTERNAL_ASSERT(query__impl_saved == query_.getIntrusivePtr());
  if (key__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(key_) &&
      !at::impl::tensor_has_dispatch(key_))
    TORCH_INTERNAL_ASSERT(key__storage_saved.value().is_alias_of(key_.storage()));
  if (key__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(key_))
    TORCH_INTERNAL_ASSERT(key__impl_saved == key_.getIntrusivePtr());
  if (value__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(value_) &&
      !at::impl::tensor_has_dispatch(value_))
    TORCH_INTERNAL_ASSERT(value__storage_saved.value().is_alias_of(value_.storage()));
  if (value__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(value_))
    TORCH_INTERNAL_ASSERT(value__impl_saved == value_.getIntrusivePtr());
  if (query_weight__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(query_weight_) &&
      !at::impl::tensor_has_dispatch(query_weight_))
    TORCH_INTERNAL_ASSERT(query_weight__storage_saved.value().is_alias_of(query_weight_.storage()));
  if (query_weight__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(query_weight_))
    TORCH_INTERNAL_ASSERT(query_weight__impl_saved == query_weight_.getIntrusivePtr());
  if (key_weight__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(key_weight_) &&
      !at::impl::tensor_has_dispatch(key_weight_))
    TORCH_INTERNAL_ASSERT(key_weight__storage_saved.value().is_alias_of(key_weight_.storage()));
  if (key_weight__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(key_weight_))
    TORCH_INTERNAL_ASSERT(key_weight__impl_saved == key_weight_.getIntrusivePtr());
  if (value_weight__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(value_weight_) &&
      !at::impl::tensor_has_dispatch(value_weight_))
    TORCH_INTERNAL_ASSERT(value_weight__storage_saved.value().is_alias_of(value_weight_.storage()));
  if (value_weight__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(value_weight_))
    TORCH_INTERNAL_ASSERT(value_weight__impl_saved == value_weight_.getIntrusivePtr());
  if (attn_mask__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(attn_mask_) &&
      !at::impl::tensor_has_dispatch(attn_mask_))
    TORCH_INTERNAL_ASSERT(attn_mask__storage_saved.value().is_alias_of(attn_mask_.storage()));
  if (attn_mask__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(attn_mask_))
    TORCH_INTERNAL_ASSERT(attn_mask__impl_saved == attn_mask_.getIntrusivePtr());
  if (out_proj_weight__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(out_proj_weight_) &&
      !at::impl::tensor_has_dispatch(out_proj_weight_))
    TORCH_INTERNAL_ASSERT(out_proj_weight__storage_saved.value().is_alias_of(out_proj_weight_.storage()));
  if (out_proj_weight__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(out_proj_weight_))
    TORCH_INTERNAL_ASSERT(out_proj_weight__impl_saved == out_proj_weight_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result0))
    TORCH_INTERNAL_ASSERT(result0.use_count() <= 1, "function: npu_multi_head_attention");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result1))
    TORCH_INTERNAL_ASSERT(result1.use_count() <= 1, "function: npu_multi_head_attention");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result2))
    TORCH_INTERNAL_ASSERT(result2.use_count() <= 1, "function: npu_multi_head_attention");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result3))
    TORCH_INTERNAL_ASSERT(result3.use_count() <= 1, "function: npu_multi_head_attention");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result4))
    TORCH_INTERNAL_ASSERT(result4.use_count() <= 1, "function: npu_multi_head_attention");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result5))
    TORCH_INTERNAL_ASSERT(result5.use_count() <= 1, "function: npu_multi_head_attention");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result6))
    TORCH_INTERNAL_ASSERT(result6.use_count() <= 1, "function: npu_multi_head_attention");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result7))
    TORCH_INTERNAL_ASSERT(result7.use_count() <= 1, "function: npu_multi_head_attention");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "npu_multi_head_attention");
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
    grad_fn->result2_ = SavedVariable(result2, true);
    grad_fn->result3_ = SavedVariable(result3, true);
    grad_fn->result4_ = SavedVariable(result4, true);
    grad_fn->result5_ = SavedVariable(result5, true);
    grad_fn->result6_ = SavedVariable(result6, true);
    grad_fn->result7_ = SavedVariable(result7, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3), std::move(result4), std::move(result5), std::move(result6), std::move(result7));
}
::std::tuple<at::Tensor,at::Tensor,int64_t,int64_t,int64_t> npu_multi_head_attention_v2(c10::DispatchKeySet ks, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & atten_mask, const ::std::optional<at::Tensor> & alibi_mask, double scale, int64_t head_num, c10::string_view input_layout, double keep_prob, int64_t pre_tokens, int64_t next_tokens, bool gen_mask_parallel, bool sync) {
  auto& query_ = unpack(query, "query", 0);
  auto& key_ = unpack(key, "key", 1);
  auto& value_ = unpack(value, "value", 2);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( query, key, value );
  
  check_no_requires_grad(atten_mask, "atten_mask", "npu_multi_head_attention_v2");
  check_no_requires_grad(alibi_mask, "alibi_mask", "npu_multi_head_attention_v2");
  std::shared_ptr<NpuMultiHeadAttentionV2Backward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuMultiHeadAttentionV2Backward0>(new NpuMultiHeadAttentionV2Backward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( query, key, value ));
    grad_fn->alibi_mask_ = SavedVariable(alibi_mask, false);
    grad_fn->atten_mask_ = SavedVariable(atten_mask, false);
    grad_fn->gen_mask_parallel = gen_mask_parallel;
    grad_fn->head_num = head_num;
    grad_fn->input_layout = std::string(input_layout);
    grad_fn->keep_prob = keep_prob;
    grad_fn->key_ = SavedVariable(key, false);
    grad_fn->next_tokens = next_tokens;
    grad_fn->pre_tokens = pre_tokens;
    grad_fn->query_ = SavedVariable(query, false);
    grad_fn->scale = scale;
    grad_fn->sync = sync;
    grad_fn->value_ = SavedVariable(value, false);
  }
  #ifndef NDEBUG
  auto query__storage_saved =
    query_.has_storage() ? ::std::optional<Storage>(query_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> query__impl_saved;
  if (query_.defined()) query__impl_saved = query_.getIntrusivePtr();
  auto key__storage_saved =
    key_.has_storage() ? ::std::optional<Storage>(key_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> key__impl_saved;
  if (key_.defined()) key__impl_saved = key_.getIntrusivePtr();
  auto value__storage_saved =
    value_.has_storage() ? ::std::optional<Storage>(value_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> value__impl_saved;
  if (value_.defined()) value__impl_saved = value_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_multi_head_attention_v2(ks & c10::after_autograd_keyset, query_, key_, value_, atten_mask, alibi_mask, scale, head_num, input_layout, keep_prob, pre_tokens, next_tokens, gen_mask_parallel, sync);
  })();
  auto [result0, result1, result2, result3, result4] = std::move(_tmp);
  #ifndef NDEBUG
  if (query__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(query_) &&
      !at::impl::tensor_has_dispatch(query_))
    TORCH_INTERNAL_ASSERT(query__storage_saved.value().is_alias_of(query_.storage()));
  if (query__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(query_))
    TORCH_INTERNAL_ASSERT(query__impl_saved == query_.getIntrusivePtr());
  if (key__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(key_) &&
      !at::impl::tensor_has_dispatch(key_))
    TORCH_INTERNAL_ASSERT(key__storage_saved.value().is_alias_of(key_.storage()));
  if (key__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(key_))
    TORCH_INTERNAL_ASSERT(key__impl_saved == key_.getIntrusivePtr());
  if (value__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(value_) &&
      !at::impl::tensor_has_dispatch(value_))
    TORCH_INTERNAL_ASSERT(value__storage_saved.value().is_alias_of(value_.storage()));
  if (value__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(value_))
    TORCH_INTERNAL_ASSERT(value__impl_saved == value_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result0))
    TORCH_INTERNAL_ASSERT(result0.use_count() <= 1, "function: npu_multi_head_attention_v2");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result1))
    TORCH_INTERNAL_ASSERT(result1.use_count() <= 1, "function: npu_multi_head_attention_v2");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "npu_multi_head_attention_v2");
  if (grad_fn) {
    grad_fn->result0_ = SavedVariable(result0, true);
    grad_fn->result1_ = SavedVariable(result1, true);
    grad_fn->result2 = result2;
    grad_fn->result3 = result3;
    grad_fn->result4 = result4;
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3), std::move(result4));
}
at::Tensor npu_ps_roi_pooling(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & rois, double spatial_scale, int64_t group_size, int64_t output_dim) {
  auto& self_ = unpack(self, "self", 0);
  auto& rois_ = unpack(rois, "rois", 1);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self );
  
  check_no_requires_grad(rois, "rois", "npu_ps_roi_pooling");
  std::shared_ptr<NpuPsRoiPoolingBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuPsRoiPoolingBackward0>(new NpuPsRoiPoolingBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->group_size = group_size;
    grad_fn->output_dim = output_dim;
    grad_fn->rois_ = SavedVariable(rois, false);
    grad_fn->self_sym_argsize_2 = self.sym_size(2);
    grad_fn->self_sym_argsize_3 = self.sym_size(3);
    grad_fn->spatial_scale = spatial_scale;
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  auto rois__storage_saved =
    rois_.has_storage() ? ::std::optional<Storage>(rois_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> rois__impl_saved;
  if (rois_.defined()) rois__impl_saved = rois_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_ps_roi_pooling(ks & c10::after_autograd_keyset, self_, rois_, spatial_scale, group_size, output_dim);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (rois__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(rois_) &&
      !at::impl::tensor_has_dispatch(rois_))
    TORCH_INTERNAL_ASSERT(rois__storage_saved.value().is_alias_of(rois_.storage()));
  if (rois__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(rois_))
    TORCH_INTERNAL_ASSERT(rois__impl_saved == rois_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result))
    TORCH_INTERNAL_ASSERT(result.use_count() <= 1, "function: npu_ps_roi_pooling");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "npu_ps_roi_pooling");
  return result;
}
::std::tuple<at::Tensor,at::Tensor> npu_rms_norm(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & gamma, double epsilon) {
  auto& self_ = unpack(self, "self", 0);
  auto& gamma_ = unpack(gamma, "gamma", 1);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self, gamma );
  
  std::shared_ptr<NpuRmsNormBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuRmsNormBackward0>(new NpuRmsNormBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, gamma ));
    grad_fn->gamma_ = SavedVariable(gamma, false);
    grad_fn->self_ = SavedVariable(self, false);
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  auto gamma__storage_saved =
    gamma_.has_storage() ? ::std::optional<Storage>(gamma_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> gamma__impl_saved;
  if (gamma_.defined()) gamma__impl_saved = gamma_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_rms_norm(ks & c10::after_autograd_keyset, self_, gamma_, epsilon);
  })();
  auto [result0, result1] = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (gamma__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(gamma_) &&
      !at::impl::tensor_has_dispatch(gamma_))
    TORCH_INTERNAL_ASSERT(gamma__storage_saved.value().is_alias_of(gamma_.storage()));
  if (gamma__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(gamma_))
    TORCH_INTERNAL_ASSERT(gamma__impl_saved == gamma_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result0))
    TORCH_INTERNAL_ASSERT(result0.use_count() <= 1, "function: npu_rms_norm");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result1))
    TORCH_INTERNAL_ASSERT(result1.use_count() <= 1, "function: npu_rms_norm");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "npu_rms_norm");
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor npu_rotary_mul(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & r1, const at::Tensor & r2, c10::string_view rotary_mode) {
  auto& self_ = unpack(self, "self", 0);
  auto& r1_ = unpack(r1, "r1", 1);
  auto& r2_ = unpack(r2, "r2", 2);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self, r1, r2 );
  
  std::shared_ptr<NpuRotaryMulBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuRotaryMulBackward0>(new NpuRotaryMulBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, r1, r2 ));
    grad_fn->r1_ = SavedVariable(r1, false);
    grad_fn->r2_ = SavedVariable(r2, false);
    grad_fn->rotary_mode = std::string(rotary_mode);
    grad_fn->self_ = SavedVariable(self, false);
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  auto r1__storage_saved =
    r1_.has_storage() ? ::std::optional<Storage>(r1_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> r1__impl_saved;
  if (r1_.defined()) r1__impl_saved = r1_.getIntrusivePtr();
  auto r2__storage_saved =
    r2_.has_storage() ? ::std::optional<Storage>(r2_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> r2__impl_saved;
  if (r2_.defined()) r2__impl_saved = r2_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_rotary_mul(ks & c10::after_autograd_keyset, self_, r1_, r2_, rotary_mode);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (r1__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(r1_) &&
      !at::impl::tensor_has_dispatch(r1_))
    TORCH_INTERNAL_ASSERT(r1__storage_saved.value().is_alias_of(r1_.storage()));
  if (r1__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(r1_))
    TORCH_INTERNAL_ASSERT(r1__impl_saved == r1_.getIntrusivePtr());
  if (r2__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(r2_) &&
      !at::impl::tensor_has_dispatch(r2_))
    TORCH_INTERNAL_ASSERT(r2__storage_saved.value().is_alias_of(r2_.storage()));
  if (r2__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(r2_))
    TORCH_INTERNAL_ASSERT(r2__impl_saved == r2_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result))
    TORCH_INTERNAL_ASSERT(result.use_count() <= 1, "function: npu_rotary_mul");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "npu_rotary_mul");
  return result;
}
at::Tensor npu_scaled_masked_softmax(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & mask, const at::Scalar & scale, bool fixed_triu_mask) {
  auto& x_ = unpack(x, "x", 0);
  auto& mask_ = unpack(mask, "mask", 1);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( x );
  
  check_no_requires_grad(mask, "mask", "npu_scaled_masked_softmax");
  std::shared_ptr<NpuScaledMaskedSoftmaxBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuScaledMaskedSoftmaxBackward0>(new NpuScaledMaskedSoftmaxBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( x ));
    grad_fn->fixed_triu_mask = fixed_triu_mask;
    grad_fn->mask_ = SavedVariable(mask, false);
    grad_fn->scale = scale;
  }
  #ifndef NDEBUG
  auto x__storage_saved =
    x_.has_storage() ? ::std::optional<Storage>(x_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> x__impl_saved;
  if (x_.defined()) x__impl_saved = x_.getIntrusivePtr();
  auto mask__storage_saved =
    mask_.has_storage() ? ::std::optional<Storage>(mask_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> mask__impl_saved;
  if (mask_.defined()) mask__impl_saved = mask_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_scaled_masked_softmax(ks & c10::after_autograd_keyset, x_, mask_, scale, fixed_triu_mask);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (x__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(x_) &&
      !at::impl::tensor_has_dispatch(x_))
    TORCH_INTERNAL_ASSERT(x__storage_saved.value().is_alias_of(x_.storage()));
  if (x__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(x_))
    TORCH_INTERNAL_ASSERT(x__impl_saved == x_.getIntrusivePtr());
  if (mask__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(mask_) &&
      !at::impl::tensor_has_dispatch(mask_))
    TORCH_INTERNAL_ASSERT(mask__storage_saved.value().is_alias_of(mask_.storage()));
  if (mask__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(mask_))
    TORCH_INTERNAL_ASSERT(mask__impl_saved == mask_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result))
    TORCH_INTERNAL_ASSERT(result.use_count() <= 1, "function: npu_scaled_masked_softmax");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "npu_scaled_masked_softmax");
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
at::Tensor npu_silu(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self );
  
  std::shared_ptr<NpuSiluBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuSiluBackward0>(new NpuSiluBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_silu(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result))
    TORCH_INTERNAL_ASSERT(result.use_count() <= 1, "function: npu_silu");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "npu_silu");
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
at::Tensor & npu_silu_(c10::DispatchKeySet ks, at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self );
  
  check_inplace(self, _any_requires_grad);
  ::std::optional<at::Tensor> original_self;
  std::shared_ptr<NpuSiluBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuSiluBackward0>(new NpuSiluBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    if (!original_self.has_value()) original_self = self.clone();
    grad_fn->self_ = SavedVariable(original_self.value(), false);
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at_npu::redispatch::npu_silu_(ks & c10::after_autograd_keyset, self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!((isFwGradDefined(self))), "Trying to use forward AD with npu_silu_ that does not support it because it has not been implemented yet.\nPlease file an issue to PyTorch at https://github.com/pytorch/pytorch/issues/new?template=feature-request.yml so that we can prioritize its implementation.");
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(self, true, self.is_view());
  }
  return self;
}
at::Tensor npu_softmax_cross_entropy_with_logits(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & labels) {
  auto& self_ = unpack(self, "self", 0);
  auto& labels_ = unpack(labels, "labels", 1);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self );
  
  check_no_requires_grad(labels, "labels", "npu_softmax_cross_entropy_with_logits");
  std::shared_ptr<NpuSoftmaxCrossEntropyWithLogitsBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuSoftmaxCrossEntropyWithLogitsBackward0>(new NpuSoftmaxCrossEntropyWithLogitsBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->labels_ = SavedVariable(labels, false);
    grad_fn->self_ = SavedVariable(self, false);
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  auto labels__storage_saved =
    labels_.has_storage() ? ::std::optional<Storage>(labels_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> labels__impl_saved;
  if (labels_.defined()) labels__impl_saved = labels_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_softmax_cross_entropy_with_logits(ks & c10::after_autograd_keyset, self_, labels_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (labels__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(labels_) &&
      !at::impl::tensor_has_dispatch(labels_))
    TORCH_INTERNAL_ASSERT(labels__storage_saved.value().is_alias_of(labels_.storage()));
  if (labels__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(labels_))
    TORCH_INTERNAL_ASSERT(labels__impl_saved == labels_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result))
    TORCH_INTERNAL_ASSERT(result.use_count() <= 1, "function: npu_softmax_cross_entropy_with_logits");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "npu_softmax_cross_entropy_with_logits");
  return result;
}
at::Tensor npu_swiglu(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim) {
  auto& self_ = unpack(self, "self", 0);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self );
  
  std::shared_ptr<NpuSwigluBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuSwigluBackward0>(new NpuSwigluBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dim = dim;
    grad_fn->self_ = SavedVariable(self, false);
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_swiglu(ks & c10::after_autograd_keyset, self_, dim);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result))
    TORCH_INTERNAL_ASSERT(result.use_count() <= 1, "function: npu_swiglu");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "npu_swiglu");
  return result;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_cross_entropy_loss(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, c10::string_view reduction, int64_t ignore_index, double label_smoothing, double lse_square_scale_for_zloss, bool return_zloss) {
  auto& input_ = unpack(input, "input", 0);
  auto& target_ = unpack(target, "target", 1);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( input );
  
  check_no_requires_grad(target, "target", "npu_cross_entropy_loss");
  check_no_requires_grad(weight, "weight", "npu_cross_entropy_loss");
  std::shared_ptr<NpuCrossEntropyLossBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuCrossEntropyLossBackward0>(new NpuCrossEntropyLossBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input ));
    grad_fn->ignore_index = ignore_index;
    grad_fn->label_smoothing = label_smoothing;
    grad_fn->lse_square_scale_for_zloss = lse_square_scale_for_zloss;
    grad_fn->reduction = std::string(reduction);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->weight_ = SavedVariable(weight, false);
  }
  #ifndef NDEBUG
  auto input__storage_saved =
    input_.has_storage() ? ::std::optional<Storage>(input_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  auto target__storage_saved =
    target_.has_storage() ? ::std::optional<Storage>(target_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_cross_entropy_loss(ks & c10::after_autograd_keyset, input_, target_, weight, reduction, ignore_index, label_smoothing, lse_square_scale_for_zloss, return_zloss);
  })();
  auto [result0, result1, result2, result3] = std::move(_tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(input_) &&
      !at::impl::tensor_has_dispatch(input_))
    TORCH_INTERNAL_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(input_))
    TORCH_INTERNAL_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (target__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(target_) &&
      !at::impl::tensor_has_dispatch(target_))
    TORCH_INTERNAL_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(target_))
    TORCH_INTERNAL_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result0))
    TORCH_INTERNAL_ASSERT(result0.use_count() <= 1, "function: npu_cross_entropy_loss");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result1))
    TORCH_INTERNAL_ASSERT(result1.use_count() <= 1, "function: npu_cross_entropy_loss");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result2))
    TORCH_INTERNAL_ASSERT(result2.use_count() <= 1, "function: npu_cross_entropy_loss");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result3))
    TORCH_INTERNAL_ASSERT(result3.use_count() <= 1, "function: npu_cross_entropy_loss");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result2 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "npu_cross_entropy_loss");
  throw_error_for_complex_autograd(result2, "npu_cross_entropy_loss");
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
    grad_fn->result3_ = SavedVariable(result3, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_group_norm_swish(c10::DispatchKeySet ks, const at::Tensor & input, int64_t num_groups, const at::Tensor & weight, const at::Tensor & bias, ::std::optional<double> eps, ::std::optional<double> swish_scale) {
  auto& input_ = unpack(input, "input", 0);
  auto& weight_ = unpack(weight, "weight", 2);
  auto& bias_ = unpack(bias, "bias", 3);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( input, weight, bias );
  
  std::shared_ptr<NpuGroupNormSwishBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuGroupNormSwishBackward0>(new NpuGroupNormSwishBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, weight, bias ));
    grad_fn->bias_ = SavedVariable(bias, false);
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->num_groups = num_groups;
    grad_fn->swish_scale = swish_scale;
    grad_fn->weight_ = SavedVariable(weight, false);
  }
  #ifndef NDEBUG
  auto input__storage_saved =
    input_.has_storage() ? ::std::optional<Storage>(input_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  auto weight__storage_saved =
    weight_.has_storage() ? ::std::optional<Storage>(weight_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  auto bias__storage_saved =
    bias_.has_storage() ? ::std::optional<Storage>(bias_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> bias__impl_saved;
  if (bias_.defined()) bias__impl_saved = bias_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_group_norm_swish(ks & c10::after_autograd_keyset, input_, num_groups, weight_, bias_, eps, swish_scale);
  })();
  auto [result0, result1, result2] = std::move(_tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(input_) &&
      !at::impl::tensor_has_dispatch(input_))
    TORCH_INTERNAL_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(input_))
    TORCH_INTERNAL_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (weight__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(weight_) &&
      !at::impl::tensor_has_dispatch(weight_))
    TORCH_INTERNAL_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(weight_))
    TORCH_INTERNAL_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (bias__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(bias_) &&
      !at::impl::tensor_has_dispatch(bias_))
    TORCH_INTERNAL_ASSERT(bias__storage_saved.value().is_alias_of(bias_.storage()));
  if (bias__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(bias_))
    TORCH_INTERNAL_ASSERT(bias__impl_saved == bias_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result0))
    TORCH_INTERNAL_ASSERT(result0.use_count() <= 1, "function: npu_group_norm_swish");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result1))
    TORCH_INTERNAL_ASSERT(result1.use_count() <= 1, "function: npu_group_norm_swish");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result2))
    TORCH_INTERNAL_ASSERT(result2.use_count() <= 1, "function: npu_group_norm_swish");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "npu_group_norm_swish");
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
    grad_fn->result2_ = SavedVariable(result2, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
at::Tensor npu_nsa_compress(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight, int64_t compress_block_size, int64_t compress_stride, at::OptionalIntArrayRef actual_seq_len) {
  auto& input_ = unpack(input, "input", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( input, weight );
  
  std::shared_ptr<NpuNsaCompressBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuNsaCompressBackward0>(new NpuNsaCompressBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, weight ));
    grad_fn->actual_seq_len = actual_seq_len;
    grad_fn->compress_block_size = compress_block_size;
    grad_fn->compress_stride = compress_stride;
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->weight_ = SavedVariable(weight, false);
  }
  #ifndef NDEBUG
  auto input__storage_saved =
    input_.has_storage() ? ::std::optional<Storage>(input_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  auto weight__storage_saved =
    weight_.has_storage() ? ::std::optional<Storage>(weight_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_nsa_compress(ks & c10::after_autograd_keyset, input_, weight_, compress_block_size, compress_stride, actual_seq_len);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(input_) &&
      !at::impl::tensor_has_dispatch(input_))
    TORCH_INTERNAL_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(input_))
    TORCH_INTERNAL_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (weight__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(weight_) &&
      !at::impl::tensor_has_dispatch(weight_))
    TORCH_INTERNAL_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(weight_))
    TORCH_INTERNAL_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result))
    TORCH_INTERNAL_ASSERT(result.use_count() <= 1, "function: npu_nsa_compress");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "npu_nsa_compress");
  return result;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_nsa_compress_attention(c10::DispatchKeySet ks, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, double scale_value, int64_t head_num, int64_t compress_block_size, int64_t compress_stride, int64_t select_block_size, int64_t select_block_count, const ::std::optional<at::Tensor> & topk_mask, const ::std::optional<at::Tensor> & atten_mask, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_cmp_seq_kvlen, at::OptionalIntArrayRef actual_sel_seq_kvlen) {
  auto& query_ = unpack(query, "query", 0);
  auto& key_ = unpack(key, "key", 1);
  auto& value_ = unpack(value, "value", 2);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( query, key, value );
  
  check_no_requires_grad(topk_mask, "topk_mask", "npu_nsa_compress_attention");
  check_no_requires_grad(atten_mask, "atten_mask", "npu_nsa_compress_attention");
  std::shared_ptr<NpuNsaCompressAttentionBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuNsaCompressAttentionBackward0>(new NpuNsaCompressAttentionBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( query, key, value ));
    grad_fn->actual_cmp_seq_kvlen = actual_cmp_seq_kvlen;
    grad_fn->actual_seq_qlen = actual_seq_qlen;
    grad_fn->atten_mask_ = SavedVariable(atten_mask, false);
    grad_fn->head_num = head_num;
    grad_fn->key_ = SavedVariable(key, false);
    grad_fn->query_ = SavedVariable(query, false);
    grad_fn->scale_value = scale_value;
    grad_fn->value_ = SavedVariable(value, false);
  }
  #ifndef NDEBUG
  auto query__storage_saved =
    query_.has_storage() ? ::std::optional<Storage>(query_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> query__impl_saved;
  if (query_.defined()) query__impl_saved = query_.getIntrusivePtr();
  auto key__storage_saved =
    key_.has_storage() ? ::std::optional<Storage>(key_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> key__impl_saved;
  if (key_.defined()) key__impl_saved = key_.getIntrusivePtr();
  auto value__storage_saved =
    value_.has_storage() ? ::std::optional<Storage>(value_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> value__impl_saved;
  if (value_.defined()) value__impl_saved = value_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_nsa_compress_attention(ks & c10::after_autograd_keyset, query_, key_, value_, scale_value, head_num, compress_block_size, compress_stride, select_block_size, select_block_count, topk_mask, atten_mask, actual_seq_qlen, actual_cmp_seq_kvlen, actual_sel_seq_kvlen);
  })();
  auto [result0, result1, result2, result3] = std::move(_tmp);
  #ifndef NDEBUG
  if (query__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(query_) &&
      !at::impl::tensor_has_dispatch(query_))
    TORCH_INTERNAL_ASSERT(query__storage_saved.value().is_alias_of(query_.storage()));
  if (query__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(query_))
    TORCH_INTERNAL_ASSERT(query__impl_saved == query_.getIntrusivePtr());
  if (key__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(key_) &&
      !at::impl::tensor_has_dispatch(key_))
    TORCH_INTERNAL_ASSERT(key__storage_saved.value().is_alias_of(key_.storage()));
  if (key__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(key_))
    TORCH_INTERNAL_ASSERT(key__impl_saved == key_.getIntrusivePtr());
  if (value__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(value_) &&
      !at::impl::tensor_has_dispatch(value_))
    TORCH_INTERNAL_ASSERT(value__storage_saved.value().is_alias_of(value_.storage()));
  if (value__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(value_))
    TORCH_INTERNAL_ASSERT(value__impl_saved == value_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result0))
    TORCH_INTERNAL_ASSERT(result0.use_count() <= 1, "function: npu_nsa_compress_attention");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result1))
    TORCH_INTERNAL_ASSERT(result1.use_count() <= 1, "function: npu_nsa_compress_attention");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result2))
    TORCH_INTERNAL_ASSERT(result2.use_count() <= 1, "function: npu_nsa_compress_attention");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result3))
    TORCH_INTERNAL_ASSERT(result3.use_count() <= 1, "function: npu_nsa_compress_attention");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "npu_nsa_compress_attention");
  if (grad_fn) {
    grad_fn->result0_ = SavedVariable(result0, true);
    grad_fn->result2_ = SavedVariable(result2, true);
    grad_fn->result3_ = SavedVariable(result3, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_nsa_select_attention(c10::DispatchKeySet ks, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & topk_indices, double scale_value, int64_t head_num, int64_t select_block_size, int64_t select_block_count, const ::std::optional<at::Tensor> & atten_mask, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_seq_kvlen) {
  auto& query_ = unpack(query, "query", 0);
  auto& key_ = unpack(key, "key", 1);
  auto& value_ = unpack(value, "value", 2);
  auto& topk_indices_ = unpack(topk_indices, "topk_indices", 3);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( query, key, value );
  
  check_no_requires_grad(topk_indices, "topk_indices", "npu_nsa_select_attention");
  check_no_requires_grad(atten_mask, "atten_mask", "npu_nsa_select_attention");
  std::shared_ptr<NpuNsaSelectAttentionBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NpuNsaSelectAttentionBackward0>(new NpuNsaSelectAttentionBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( query, key, value ));
    grad_fn->actual_seq_kvlen = actual_seq_kvlen;
    grad_fn->actual_seq_qlen = actual_seq_qlen;
    grad_fn->atten_mask_ = SavedVariable(atten_mask, false);
    grad_fn->head_num = head_num;
    grad_fn->key_ = SavedVariable(key, false);
    grad_fn->query_ = SavedVariable(query, false);
    grad_fn->scale_value = scale_value;
    grad_fn->select_block_count = select_block_count;
    grad_fn->select_block_size = select_block_size;
    grad_fn->topk_indices_ = SavedVariable(topk_indices, false);
    grad_fn->value_ = SavedVariable(value, false);
  }
  #ifndef NDEBUG
  auto query__storage_saved =
    query_.has_storage() ? ::std::optional<Storage>(query_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> query__impl_saved;
  if (query_.defined()) query__impl_saved = query_.getIntrusivePtr();
  auto key__storage_saved =
    key_.has_storage() ? ::std::optional<Storage>(key_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> key__impl_saved;
  if (key_.defined()) key__impl_saved = key_.getIntrusivePtr();
  auto value__storage_saved =
    value_.has_storage() ? ::std::optional<Storage>(value_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> value__impl_saved;
  if (value_.defined()) value__impl_saved = value_.getIntrusivePtr();
  auto topk_indices__storage_saved =
    topk_indices_.has_storage() ? ::std::optional<Storage>(topk_indices_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> topk_indices__impl_saved;
  if (topk_indices_.defined()) topk_indices__impl_saved = topk_indices_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
      return at_npu::redispatch::npu_nsa_select_attention(ks & c10::after_autograd_keyset, query_, key_, value_, topk_indices_, scale_value, head_num, select_block_size, select_block_count, atten_mask, actual_seq_qlen, actual_seq_kvlen);
  })();
  auto [result0, result1, result2] = std::move(_tmp);
  #ifndef NDEBUG
  if (query__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(query_) &&
      !at::impl::tensor_has_dispatch(query_))
    TORCH_INTERNAL_ASSERT(query__storage_saved.value().is_alias_of(query_.storage()));
  if (query__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(query_))
    TORCH_INTERNAL_ASSERT(query__impl_saved == query_.getIntrusivePtr());
  if (key__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(key_) &&
      !at::impl::tensor_has_dispatch(key_))
    TORCH_INTERNAL_ASSERT(key__storage_saved.value().is_alias_of(key_.storage()));
  if (key__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(key_))
    TORCH_INTERNAL_ASSERT(key__impl_saved == key_.getIntrusivePtr());
  if (value__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(value_) &&
      !at::impl::tensor_has_dispatch(value_))
    TORCH_INTERNAL_ASSERT(value__storage_saved.value().is_alias_of(value_.storage()));
  if (value__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(value_))
    TORCH_INTERNAL_ASSERT(value__impl_saved == value_.getIntrusivePtr());
  if (topk_indices__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(topk_indices_) &&
      !at::impl::tensor_has_dispatch(topk_indices_))
    TORCH_INTERNAL_ASSERT(topk_indices__storage_saved.value().is_alias_of(topk_indices_.storage()));
  if (topk_indices__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(topk_indices_))
    TORCH_INTERNAL_ASSERT(topk_indices__impl_saved == topk_indices_.getIntrusivePtr());
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result0))
    TORCH_INTERNAL_ASSERT(result0.use_count() <= 1, "function: npu_nsa_select_attention");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result1))
    TORCH_INTERNAL_ASSERT(result1.use_count() <= 1, "function: npu_nsa_select_attention");
  
  if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(result2))
    TORCH_INTERNAL_ASSERT(result2.use_count() <= 1, "function: npu_nsa_select_attention");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "npu_nsa_select_attention");
  if (grad_fn) {
    grad_fn->result0_ = SavedVariable(result0, true);
    grad_fn->result1_ = SavedVariable(result1, true);
    grad_fn->result2_ = SavedVariable(result2, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
at::Tensor & binary_cross_entropy_with_logits_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & pos_weight, int64_t reduction, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto& out_ = unpack(out, "out", 5);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self, target, weight, pos_weight );
  
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, target, weight, pos_weight )) {
    throw_error_out_requires_grad("binary_cross_entropy_with_logits");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("binary_cross_entropy_with_logits");
  }
  #ifndef NDEBUG
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  auto target__storage_saved =
    target_.has_storage() ? ::std::optional<Storage>(target_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  auto out__storage_saved =
    out_.has_storage() ? ::std::optional<Storage>(out_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::binary_cross_entropy_with_logits_outf(ks & c10::after_autograd_keyset, self_, target_, weight, pos_weight, reduction, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(target_) &&
      !at::impl::tensor_has_dispatch(target_))
    TORCH_INTERNAL_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(target_))
    TORCH_INTERNAL_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  if (out__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(out_) &&
      !at::impl::tensor_has_dispatch(out_))
    TORCH_INTERNAL_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(out_))
    TORCH_INTERNAL_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!((isFwGradDefined(self) || isFwGradDefined(target) || isFwGradDefined(weight) || isFwGradDefined(pos_weight) || isFwGradDefined(out))), "Trying to use forward AD with binary_cross_entropy_with_logits_out that does not support it because it is an out= function");
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> matmul_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_out, const at::Tensor & self, const at::Tensor & other, ::std::array<bool,2> mask, at::Tensor & out0, at::Tensor & out1) {
  auto& grad_out_ = unpack(grad_out, "grad_out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  auto& out0_ = unpack(out0, "out0", 4);
  auto& out1_ = unpack(out1, "out1", 5);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( grad_out, self, other );
  
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_out, self, other )) {
    throw_error_out_requires_grad("matmul_backward");
  }
  if (compute_requires_grad( out0, out1 )) {
    throw_error_out_requires_grad("matmul_backward");
  }
  #ifndef NDEBUG
  auto grad_out__storage_saved =
    grad_out_.has_storage() ? ::std::optional<Storage>(grad_out_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_out__impl_saved;
  if (grad_out_.defined()) grad_out__impl_saved = grad_out_.getIntrusivePtr();
  auto self__storage_saved =
    self_.has_storage() ? ::std::optional<Storage>(self_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  auto other__storage_saved =
    other_.has_storage() ? ::std::optional<Storage>(other_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  auto out0__storage_saved =
    out0_.has_storage() ? ::std::optional<Storage>(out0_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> out0__impl_saved;
  if (out0_.defined()) out0__impl_saved = out0_.getIntrusivePtr();
  auto out1__storage_saved =
    out1_.has_storage() ? ::std::optional<Storage>(out1_.storage()) : ::std::nullopt;
  c10::intrusive_ptr<TensorImpl> out1__impl_saved;
  if (out1_.defined()) out1__impl_saved = out1_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::matmul_backward_outf(ks & c10::after_autograd_keyset, grad_out_, self_, other_, mask, out0_, out1_);
  }
  #ifndef NDEBUG
  if (grad_out__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(grad_out_) &&
      !at::impl::tensor_has_dispatch(grad_out_))
    TORCH_INTERNAL_ASSERT(grad_out__storage_saved.value().is_alias_of(grad_out_.storage()));
  if (grad_out__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(grad_out_))
    TORCH_INTERNAL_ASSERT(grad_out__impl_saved == grad_out_.getIntrusivePtr());
  if (self__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(self_) &&
      !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(self_))
    TORCH_INTERNAL_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(other_) &&
      !at::impl::tensor_has_dispatch(other_))
    TORCH_INTERNAL_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(other_))
    TORCH_INTERNAL_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  if (out0__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(out0_) &&
      !at::impl::tensor_has_dispatch(out0_))
    TORCH_INTERNAL_ASSERT(out0__storage_saved.value().is_alias_of(out0_.storage()));
  if (out0__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(out0_))
    TORCH_INTERNAL_ASSERT(out0__impl_saved == out0_.getIntrusivePtr());
  if (out1__storage_saved.has_value() &&
      !at::impl::dispatch_mode_enabled() &&
      !at::impl::tensor_has_dispatch(out1_) &&
      !at::impl::tensor_has_dispatch(out1_))
    TORCH_INTERNAL_ASSERT(out1__storage_saved.value().is_alias_of(out1_.storage()));
  if (out1__impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(out1_))
    TORCH_INTERNAL_ASSERT(out1__impl_saved == out1_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out0, out1 ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!((isFwGradDefined(grad_out) || isFwGradDefined(self) || isFwGradDefined(other) || isFwGradDefined(out0) || isFwGradDefined(out1))), "Trying to use forward AD with matmul_backward_out that does not support it because it is an out= function");
  return std::forward_as_tuple(out0, out1);
}

} // namespace VariableType

namespace {

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
    m.impl("binary_cross_entropy_with_logits",
           TORCH_FN(VariableType::binary_cross_entropy_with_logits)
    );
    m.impl("_fft_r2c",
           TORCH_FN(VariableType::_fft_r2c)
    );
    m.impl("_fft_r2c.out",
           TORCH_FN(VariableType::_fft_r2c_out_out)
    );
    m.impl("_fft_c2r",
           TORCH_FN(VariableType::_fft_c2r)
    );
    m.impl("_fft_c2r.out",
           TORCH_FN(VariableType::_fft_c2r_out_out)
    );
    m.impl("kl_div",
           TORCH_FN(VariableType::kl_div)
    );
    m.impl("matmul_backward",
           TORCH_FN(VariableType::matmul_backward)
    );
    m.impl("repeat_interleave.self_Tensor",
           TORCH_FN(VariableType::repeat_interleave_self_Tensor)
    );
    m.impl("repeat_interleave.self_int",
           TORCH_FN(VariableType::repeat_interleave_self_int)
    );
    m.impl("stft",
           TORCH_FN(VariableType::stft)
    );
    m.impl("gather.out",
           TORCH_FN(VariableType::gather_out_out)
    );
    m.impl("gather",
           TORCH_FN(VariableType::gather)
    );
    m.impl("l1_loss",
           TORCH_FN(VariableType::l1_loss)
    );
    m.impl("binary_cross_entropy_with_logits.out",
           TORCH_FN(VariableType::binary_cross_entropy_with_logits_out_out)
    );
    m.impl("matmul_backward.out",
           TORCH_FN(VariableType::matmul_backward_out_out)
    );
}

TORCH_LIBRARY_IMPL(npu, AutogradPrivateUse1, m) {
    m.impl("_npu_format_cast",
           TORCH_FN(VariableType::_npu_format_cast)
    );
    m.impl("npu_gelu",
           TORCH_FN(VariableType::npu_gelu)
    );
    m.impl("_dropout_with_byte_mask",
           TORCH_FN(VariableType::_dropout_with_byte_mask)
    );
    m.impl("_npu_ciou",
           TORCH_FN(VariableType::_npu_ciou)
    );
    m.impl("_npu_dropout",
           TORCH_FN(VariableType::_npu_dropout)
    );
    m.impl("fast_gelu",
           TORCH_FN(VariableType::fast_gelu)
    );
    m.impl("npu_add_layer_norm",
           TORCH_FN(VariableType::npu_add_layer_norm)
    );
    m.impl("npu_bmmV2",
           TORCH_FN(VariableType::npu_bmmV2)
    );
    m.impl("npu_confusion_transpose",
           TORCH_FN(VariableType::npu_confusion_transpose)
    );
    m.impl("npu_convolution",
           TORCH_FN(VariableType::npu_convolution)
    );
    m.impl("npu_convolution_transpose",
           TORCH_FN(VariableType::npu_convolution_transpose)
    );
    m.impl("npu_deep_norm",
           TORCH_FN(VariableType::npu_deep_norm)
    );
    m.impl("npu_deformable_conv2d",
           TORCH_FN(VariableType::npu_deformable_conv2d)
    );
    m.impl("npu_diou",
           TORCH_FN(VariableType::npu_diou)
    );
    m.impl("npu_dropout_do_mask",
           TORCH_FN(VariableType::npu_dropout_do_mask)
    );
    m.impl("npu_dropout_with_add_softmax",
           TORCH_FN(VariableType::npu_dropout_with_add_softmax)
    );
    m.impl("npu_dtype_cast",
           TORCH_FN(VariableType::npu_dtype_cast)
    );
    m.impl("npu_fused_attention_score_fwd",
           TORCH_FN(VariableType::npu_fused_attention_score_fwd)
    );
    m.impl("npu_fusion_attention",
           TORCH_FN(VariableType::npu_fusion_attention)
    );
    m.impl("npu_fusion_attention_v2",
           TORCH_FN(VariableType::npu_fusion_attention_v2)
    );
    m.impl("npu_geglu",
           TORCH_FN(VariableType::npu_geglu)
    );
    m.impl("npu_giou",
           TORCH_FN(VariableType::npu_giou)
    );
    m.impl("npu_gru",
           TORCH_FN(VariableType::npu_gru)
    );
    m.impl("npu_linear",
           TORCH_FN(VariableType::npu_linear)
    );
    m.impl("npu_lstm",
           TORCH_FN(VariableType::npu_lstm)
    );
    m.impl("npu_lstm_cell",
           TORCH_FN(VariableType::npu_lstm_cell)
    );
    m.impl("npu_lstm_data",
           TORCH_FN(VariableType::npu_lstm_data)
    );
    m.impl("npu_max.dim",
           TORCH_FN(VariableType::npu_max_dim)
    );
    m.impl("npu_min.dim",
           TORCH_FN(VariableType::npu_min_dim)
    );
    m.impl("npu_mish",
           TORCH_FN(VariableType::npu_mish)
    );
    m.impl("npu_multi_head_attention",
           TORCH_FN(VariableType::npu_multi_head_attention)
    );
    m.impl("npu_multi_head_attention_v2",
           TORCH_FN(VariableType::npu_multi_head_attention_v2)
    );
    m.impl("npu_ps_roi_pooling",
           TORCH_FN(VariableType::npu_ps_roi_pooling)
    );
    m.impl("npu_rms_norm",
           TORCH_FN(VariableType::npu_rms_norm)
    );
    m.impl("npu_rotary_mul",
           TORCH_FN(VariableType::npu_rotary_mul)
    );
    m.impl("npu_scaled_masked_softmax",
           TORCH_FN(VariableType::npu_scaled_masked_softmax)
    );
    m.impl("npu_silu",
           TORCH_FN(VariableType::npu_silu)
    );
    m.impl("npu_silu_",
           TORCH_FN(VariableType::npu_silu_)
    );
    m.impl("npu_softmax_cross_entropy_with_logits",
           TORCH_FN(VariableType::npu_softmax_cross_entropy_with_logits)
    );
    m.impl("npu_swiglu",
           TORCH_FN(VariableType::npu_swiglu)
    );
    m.impl("npu_cross_entropy_loss",
           TORCH_FN(VariableType::npu_cross_entropy_loss)
    );
    m.impl("npu_group_norm_swish",
           TORCH_FN(VariableType::npu_group_norm_swish)
    );
    m.impl("npu_nsa_compress",
           TORCH_FN(VariableType::npu_nsa_compress)
    );
    m.impl("npu_nsa_compress_attention",
           TORCH_FN(VariableType::npu_nsa_compress_attention)
    );
    m.impl("npu_nsa_select_attention",
           TORCH_FN(VariableType::npu_nsa_select_attention)
    );
}

}

}} // namespace at_npu::autograd
