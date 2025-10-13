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

}  // namespace
}  // namespace ADInplaceOrView

namespace {

TORCH_LIBRARY_IMPL(npu, ADInplaceOrView, m) {
;
}

}  // namespace
} // namespace at_npu
