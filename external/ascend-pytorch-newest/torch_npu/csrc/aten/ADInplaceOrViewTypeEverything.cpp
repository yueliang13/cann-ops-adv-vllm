#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include "torch/csrc/autograd/VariableTypeUtils.h"
#include <torch/library.h>

#include "torch_npu/csrc/aten/CustomRedispatch.h"

// @generated from ../../../../../../../xzh_workspace/cann-ops-adv-vllm/external/ascend-pytorch-newest/codegen/autograd/templates/ADInplaceOrViewType.cpp


using namespace at;
using torch::autograd::CreationMeta;
using torch::autograd::as_view;
using torch::autograd::increment_version;

namespace at_npu {

namespace ADInplaceOrView {

namespace {
at::Tensor & npu_silu_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at_npu::redispatch::npu_silu_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
}  // namespace
}  // namespace ADInplaceOrView

namespace {

TORCH_LIBRARY_IMPL(npu, ADInplaceOrView, m) {
  m.impl("npu_silu_",
         TORCH_FN(ADInplaceOrView::npu_silu_)
  );;
}

}  // namespace
} // namespace at_npu
