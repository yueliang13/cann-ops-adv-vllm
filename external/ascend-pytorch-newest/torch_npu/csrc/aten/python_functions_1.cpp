#ifndef BUILD_LIBTORCH
#include <torch/csrc/autograd/generated/python_functions.h>

// @ generated from ../../../../../../../workspace/profile/cann-ops-adv-vllm/external/ascend-pytorch-newest/codegen/autograd/templates/python_functions.cpp

#include <Python.h>
#include <ATen/ATen.h>

#include <c10/core/SymNodeImpl.h>
#include <torch/csrc/autograd/generated/Functions.h>
#include <torch/csrc/autograd/python_cpp_function.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>
#include "torch_npu/csrc/aten/Functions.h"

// NOTE: See [Sharded File] comment in VariableType

namespace at_npu {
namespace autograd {
namespace generated {

template <typename C>
static void addClass(PyObject *module, PyTypeObject &type, const char *name,
                     PyGetSetDef *function_properties = NULL,
                     PyMethodDef *function_methods = NULL)
{
    _initFunctionPyTypeObject(type, name, function_properties, function_methods);
    Py_INCREF(&type);
    PyModule_AddObject(module, name, (PyObject *)&type);
    registerCppFunction(typeid(C), &type);
}

PyObject* THPGatherBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<GatherBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPGatherBackward0_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GatherBackward0*>(self->cdata.get())->index_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPGatherBackward0_index_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GatherBackward0*>(self->cdata.get())->index_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPGatherBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<GatherBackward0*>(self->cdata.get())->self_sym_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPGatherBackward0_sparse_grad_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<GatherBackward0*>(self->cdata.get())->sparse_grad;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef GatherBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPGatherBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_index", (getter)THPGatherBackward0_index_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_index", (getter)THPGatherBackward0_index_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPGatherBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_sparse_grad", (getter)THPGatherBackward0_sparse_grad_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNpuCiouBackward0_gtboxes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuCiouBackward0*>(self->cdata.get())->gtboxes_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuCiouBackward0_gtboxes_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuCiouBackward0*>(self->cdata.get())->gtboxes_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuCiouBackward0_is_cross_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuCiouBackward0*>(self->cdata.get())->is_cross;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuCiouBackward0_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuCiouBackward0*>(self->cdata.get())->mode;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuCiouBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuCiouBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuCiouBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuCiouBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuCiouBackward0_trans_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuCiouBackward0*>(self->cdata.get())->trans;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuCiouBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuCiouBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuCiouBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuCiouBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuCiouBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_gtboxes", (getter)THPNpuCiouBackward0_gtboxes_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_gtboxes", (getter)THPNpuCiouBackward0_gtboxes_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_is_cross", (getter)THPNpuCiouBackward0_is_cross_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mode", (getter)THPNpuCiouBackward0_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPNpuCiouBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPNpuCiouBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_trans", (getter)THPNpuCiouBackward0_trans_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPNpuCiouBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPNpuCiouBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNpuDropoutBackward0_p_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuDropoutBackward0*>(self->cdata.get())->p;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDropoutBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuDropoutBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDropoutBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuDropoutBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuDropoutBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_p", (getter)THPNpuDropoutBackward0_p_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPNpuDropoutBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPNpuDropoutBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMatmulBackwardBackward0_grad_out_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MatmulBackwardBackward0*>(self->cdata.get())->grad_out_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMatmulBackwardBackward0_grad_out_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MatmulBackwardBackward0*>(self->cdata.get())->grad_out_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMatmulBackwardBackward0_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MatmulBackwardBackward0*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMatmulBackwardBackward0_other_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MatmulBackwardBackward0*>(self->cdata.get())->other_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMatmulBackwardBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MatmulBackwardBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMatmulBackwardBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MatmulBackwardBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MatmulBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_grad_out", (getter)THPMatmulBackwardBackward0_grad_out_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_grad_out", (getter)THPMatmulBackwardBackward0_grad_out_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_other", (getter)THPMatmulBackwardBackward0_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_other", (getter)THPMatmulBackwardBackward0_other_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPMatmulBackwardBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPMatmulBackwardBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNpuBmmv2Backward0_mat2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuBmmv2Backward0*>(self->cdata.get())->mat2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuBmmv2Backward0_mat2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuBmmv2Backward0*>(self->cdata.get())->mat2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuBmmv2Backward0_mat2_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuBmmv2Backward0*>(self->cdata.get())->mat2_sym_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuBmmv2Backward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuBmmv2Backward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuBmmv2Backward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuBmmv2Backward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuBmmv2Backward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuBmmv2Backward0*>(self->cdata.get())->self_sym_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuBmmv2Backward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_mat2", (getter)THPNpuBmmv2Backward0_mat2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_mat2", (getter)THPNpuBmmv2Backward0_mat2_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mat2_sym_sizes", (getter)THPNpuBmmv2Backward0_mat2_sym_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPNpuBmmv2Backward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPNpuBmmv2Backward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPNpuBmmv2Backward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNpuConvolutionBackward0_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuConvolutionBackward0*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuConvolutionBackward0_groups_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuConvolutionBackward0*>(self->cdata.get())->groups;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuConvolutionBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuConvolutionBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuConvolutionBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuConvolutionBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuConvolutionBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuConvolutionBackward0*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuConvolutionBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuConvolutionBackward0*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuConvolutionBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuConvolutionBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuConvolutionBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuConvolutionBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuConvolutionBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dilation", (getter)THPNpuConvolutionBackward0_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_groups", (getter)THPNpuConvolutionBackward0_groups_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPNpuConvolutionBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPNpuConvolutionBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPNpuConvolutionBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPNpuConvolutionBackward0_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPNpuConvolutionBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPNpuConvolutionBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNpuConvolutionTransposeBackward0_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuConvolutionTransposeBackward0*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuConvolutionTransposeBackward0_groups_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuConvolutionTransposeBackward0*>(self->cdata.get())->groups;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuConvolutionTransposeBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuConvolutionTransposeBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuConvolutionTransposeBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuConvolutionTransposeBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuConvolutionTransposeBackward0_output_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuConvolutionTransposeBackward0*>(self->cdata.get())->output_padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuConvolutionTransposeBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuConvolutionTransposeBackward0*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuConvolutionTransposeBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuConvolutionTransposeBackward0*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuConvolutionTransposeBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuConvolutionTransposeBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuConvolutionTransposeBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuConvolutionTransposeBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuConvolutionTransposeBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dilation", (getter)THPNpuConvolutionTransposeBackward0_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_groups", (getter)THPNpuConvolutionTransposeBackward0_groups_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPNpuConvolutionTransposeBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPNpuConvolutionTransposeBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_padding", (getter)THPNpuConvolutionTransposeBackward0_output_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPNpuConvolutionTransposeBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPNpuConvolutionTransposeBackward0_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPNpuConvolutionTransposeBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPNpuConvolutionTransposeBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNpuDropoutDoMaskBackward0_p_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuDropoutDoMaskBackward0*>(self->cdata.get())->p;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDropoutDoMaskBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuDropoutDoMaskBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDropoutDoMaskBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuDropoutDoMaskBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuDropoutDoMaskBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_p", (getter)THPNpuDropoutDoMaskBackward0_p_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPNpuDropoutDoMaskBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPNpuDropoutDoMaskBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNpuDropoutWithAddSoftmaxBackward0_alpha_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuDropoutWithAddSoftmaxBackward0*>(self->cdata.get())->alpha;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDropoutWithAddSoftmaxBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuDropoutWithAddSoftmaxBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDropoutWithAddSoftmaxBackward0_prob_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuDropoutWithAddSoftmaxBackward0*>(self->cdata.get())->prob;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDropoutWithAddSoftmaxBackward0_result0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuDropoutWithAddSoftmaxBackward0*>(self->cdata.get())->result0_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDropoutWithAddSoftmaxBackward0_result0_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuDropoutWithAddSoftmaxBackward0*>(self->cdata.get())->result0_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDropoutWithAddSoftmaxBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuDropoutWithAddSoftmaxBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDropoutWithAddSoftmaxBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuDropoutWithAddSoftmaxBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuDropoutWithAddSoftmaxBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_alpha", (getter)THPNpuDropoutWithAddSoftmaxBackward0_alpha_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPNpuDropoutWithAddSoftmaxBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_prob", (getter)THPNpuDropoutWithAddSoftmaxBackward0_prob_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result0", (getter)THPNpuDropoutWithAddSoftmaxBackward0_result0_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result0", (getter)THPNpuDropoutWithAddSoftmaxBackward0_result0_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPNpuDropoutWithAddSoftmaxBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPNpuDropoutWithAddSoftmaxBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef NpuDtypeCastBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPNpuFusedAttentionScoreFwdBackward0_dx_transpose_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuFusedAttentionScoreFwdBackward0*>(self->cdata.get())->dx_transpose;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusedAttentionScoreFwdBackward0_keep_prob_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuFusedAttentionScoreFwdBackward0*>(self->cdata.get())->keep_prob;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusedAttentionScoreFwdBackward0_key_layer_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusedAttentionScoreFwdBackward0*>(self->cdata.get())->key_layer_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusedAttentionScoreFwdBackward0_key_layer_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusedAttentionScoreFwdBackward0*>(self->cdata.get())->key_layer_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusedAttentionScoreFwdBackward0_key_transpose_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuFusedAttentionScoreFwdBackward0*>(self->cdata.get())->key_transpose;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusedAttentionScoreFwdBackward0_query_layer_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusedAttentionScoreFwdBackward0*>(self->cdata.get())->query_layer_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusedAttentionScoreFwdBackward0_query_layer_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusedAttentionScoreFwdBackward0*>(self->cdata.get())->query_layer_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusedAttentionScoreFwdBackward0_query_transpose_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuFusedAttentionScoreFwdBackward0*>(self->cdata.get())->query_transpose;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusedAttentionScoreFwdBackward0_scale_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuFusedAttentionScoreFwdBackward0*>(self->cdata.get())->scale;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusedAttentionScoreFwdBackward0_value_layer_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusedAttentionScoreFwdBackward0*>(self->cdata.get())->value_layer_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusedAttentionScoreFwdBackward0_value_layer_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusedAttentionScoreFwdBackward0*>(self->cdata.get())->value_layer_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusedAttentionScoreFwdBackward0_value_transpose_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuFusedAttentionScoreFwdBackward0*>(self->cdata.get())->value_transpose;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusedAttentionScoreFwdBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusedAttentionScoreFwdBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusedAttentionScoreFwdBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusedAttentionScoreFwdBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusedAttentionScoreFwdBackward0_result2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusedAttentionScoreFwdBackward0*>(self->cdata.get())->result2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusedAttentionScoreFwdBackward0_result2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusedAttentionScoreFwdBackward0*>(self->cdata.get())->result2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuFusedAttentionScoreFwdBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dx_transpose", (getter)THPNpuFusedAttentionScoreFwdBackward0_dx_transpose_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keep_prob", (getter)THPNpuFusedAttentionScoreFwdBackward0_keep_prob_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_key_layer", (getter)THPNpuFusedAttentionScoreFwdBackward0_key_layer_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_key_layer", (getter)THPNpuFusedAttentionScoreFwdBackward0_key_layer_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_key_transpose", (getter)THPNpuFusedAttentionScoreFwdBackward0_key_transpose_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_query_layer", (getter)THPNpuFusedAttentionScoreFwdBackward0_query_layer_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_query_layer", (getter)THPNpuFusedAttentionScoreFwdBackward0_query_layer_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_query_transpose", (getter)THPNpuFusedAttentionScoreFwdBackward0_query_transpose_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale", (getter)THPNpuFusedAttentionScoreFwdBackward0_scale_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_value_layer", (getter)THPNpuFusedAttentionScoreFwdBackward0_value_layer_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_value_layer", (getter)THPNpuFusedAttentionScoreFwdBackward0_value_layer_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_value_transpose", (getter)THPNpuFusedAttentionScoreFwdBackward0_value_transpose_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPNpuFusedAttentionScoreFwdBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPNpuFusedAttentionScoreFwdBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result2", (getter)THPNpuFusedAttentionScoreFwdBackward0_result2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result2", (getter)THPNpuFusedAttentionScoreFwdBackward0_result2_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNpuGiouBackward0_gtboxes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGiouBackward0*>(self->cdata.get())->gtboxes_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGiouBackward0_gtboxes_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGiouBackward0*>(self->cdata.get())->gtboxes_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGiouBackward0_is_cross_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuGiouBackward0*>(self->cdata.get())->is_cross;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGiouBackward0_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuGiouBackward0*>(self->cdata.get())->mode;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGiouBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGiouBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGiouBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGiouBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGiouBackward0_trans_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuGiouBackward0*>(self->cdata.get())->trans;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuGiouBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_gtboxes", (getter)THPNpuGiouBackward0_gtboxes_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_gtboxes", (getter)THPNpuGiouBackward0_gtboxes_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_is_cross", (getter)THPNpuGiouBackward0_is_cross_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mode", (getter)THPNpuGiouBackward0_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPNpuGiouBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPNpuGiouBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_trans", (getter)THPNpuGiouBackward0_trans_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNpuGruBackward0_bias_hidden_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGruBackward0*>(self->cdata.get())->bias_hidden_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGruBackward0_bias_hidden_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGruBackward0*>(self->cdata.get())->bias_hidden_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGruBackward0_bias_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGruBackward0*>(self->cdata.get())->bias_input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGruBackward0_bias_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGruBackward0*>(self->cdata.get())->bias_input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGruBackward0_hx_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGruBackward0*>(self->cdata.get())->hx_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGruBackward0_hx_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGruBackward0*>(self->cdata.get())->hx_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGruBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGruBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGruBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGruBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGruBackward0_seq_length_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGruBackward0*>(self->cdata.get())->seq_length_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGruBackward0_seq_length_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGruBackward0*>(self->cdata.get())->seq_length_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGruBackward0_weight_hidden_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGruBackward0*>(self->cdata.get())->weight_hidden_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGruBackward0_weight_hidden_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGruBackward0*>(self->cdata.get())->weight_hidden_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGruBackward0_weight_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGruBackward0*>(self->cdata.get())->weight_input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGruBackward0_weight_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGruBackward0*>(self->cdata.get())->weight_input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGruBackward0_result0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGruBackward0*>(self->cdata.get())->result0_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGruBackward0_result0_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGruBackward0*>(self->cdata.get())->result0_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGruBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGruBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGruBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGruBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGruBackward0_result2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGruBackward0*>(self->cdata.get())->result2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGruBackward0_result2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGruBackward0*>(self->cdata.get())->result2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGruBackward0_result3_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGruBackward0*>(self->cdata.get())->result3_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGruBackward0_result3_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGruBackward0*>(self->cdata.get())->result3_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGruBackward0_result4_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGruBackward0*>(self->cdata.get())->result4_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGruBackward0_result4_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGruBackward0*>(self->cdata.get())->result4_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGruBackward0_result5_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGruBackward0*>(self->cdata.get())->result5_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGruBackward0_result5_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGruBackward0*>(self->cdata.get())->result5_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuGruBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_bias_hidden", (getter)THPNpuGruBackward0_bias_hidden_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_bias_hidden", (getter)THPNpuGruBackward0_bias_hidden_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_bias_input", (getter)THPNpuGruBackward0_bias_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_bias_input", (getter)THPNpuGruBackward0_bias_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_hx", (getter)THPNpuGruBackward0_hx_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_hx", (getter)THPNpuGruBackward0_hx_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPNpuGruBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPNpuGruBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_seq_length", (getter)THPNpuGruBackward0_seq_length_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_seq_length", (getter)THPNpuGruBackward0_seq_length_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight_hidden", (getter)THPNpuGruBackward0_weight_hidden_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight_hidden", (getter)THPNpuGruBackward0_weight_hidden_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight_input", (getter)THPNpuGruBackward0_weight_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight_input", (getter)THPNpuGruBackward0_weight_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result0", (getter)THPNpuGruBackward0_result0_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result0", (getter)THPNpuGruBackward0_result0_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPNpuGruBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPNpuGruBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result2", (getter)THPNpuGruBackward0_result2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result2", (getter)THPNpuGruBackward0_result2_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result3", (getter)THPNpuGruBackward0_result3_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result3", (getter)THPNpuGruBackward0_result3_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result4", (getter)THPNpuGruBackward0_result4_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result4", (getter)THPNpuGruBackward0_result4_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result5", (getter)THPNpuGruBackward0_result5_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result5", (getter)THPNpuGruBackward0_result5_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNpuLinearBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLinearBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLinearBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLinearBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLinearBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLinearBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLinearBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLinearBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuLinearBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_input", (getter)THPNpuLinearBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPNpuLinearBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPNpuLinearBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPNpuLinearBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNpuLstmCellBackward0_c_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmCellBackward0*>(self->cdata.get())->c_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmCellBackward0_c_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmCellBackward0*>(self->cdata.get())->c_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmCellBackward0_h_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmCellBackward0*>(self->cdata.get())->h_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmCellBackward0_h_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmCellBackward0*>(self->cdata.get())->h_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmCellBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmCellBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmCellBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmCellBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmCellBackward0_w_hh_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmCellBackward0*>(self->cdata.get())->w_hh_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmCellBackward0_w_hh_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmCellBackward0*>(self->cdata.get())->w_hh_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmCellBackward0_w_ih_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmCellBackward0*>(self->cdata.get())->w_ih_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmCellBackward0_w_ih_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmCellBackward0*>(self->cdata.get())->w_ih_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmCellBackward0_result0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmCellBackward0*>(self->cdata.get())->result0_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmCellBackward0_result0_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmCellBackward0*>(self->cdata.get())->result0_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmCellBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmCellBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmCellBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmCellBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmCellBackward0_result2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmCellBackward0*>(self->cdata.get())->result2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmCellBackward0_result2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmCellBackward0*>(self->cdata.get())->result2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmCellBackward0_result3_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmCellBackward0*>(self->cdata.get())->result3_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmCellBackward0_result3_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmCellBackward0*>(self->cdata.get())->result3_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmCellBackward0_result4_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmCellBackward0*>(self->cdata.get())->result4_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmCellBackward0_result4_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmCellBackward0*>(self->cdata.get())->result4_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmCellBackward0_result5_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmCellBackward0*>(self->cdata.get())->result5_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmCellBackward0_result5_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmCellBackward0*>(self->cdata.get())->result5_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmCellBackward0_result6_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmCellBackward0*>(self->cdata.get())->result6_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmCellBackward0_result6_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmCellBackward0*>(self->cdata.get())->result6_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmCellBackward0_result7_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmCellBackward0*>(self->cdata.get())->result7_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmCellBackward0_result7_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmCellBackward0*>(self->cdata.get())->result7_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuLstmCellBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_c", (getter)THPNpuLstmCellBackward0_c_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_c", (getter)THPNpuLstmCellBackward0_c_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_h", (getter)THPNpuLstmCellBackward0_h_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_h", (getter)THPNpuLstmCellBackward0_h_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPNpuLstmCellBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPNpuLstmCellBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_w_hh", (getter)THPNpuLstmCellBackward0_w_hh_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_w_hh", (getter)THPNpuLstmCellBackward0_w_hh_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_w_ih", (getter)THPNpuLstmCellBackward0_w_ih_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_w_ih", (getter)THPNpuLstmCellBackward0_w_ih_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result0", (getter)THPNpuLstmCellBackward0_result0_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result0", (getter)THPNpuLstmCellBackward0_result0_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPNpuLstmCellBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPNpuLstmCellBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result2", (getter)THPNpuLstmCellBackward0_result2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result2", (getter)THPNpuLstmCellBackward0_result2_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result3", (getter)THPNpuLstmCellBackward0_result3_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result3", (getter)THPNpuLstmCellBackward0_result3_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result4", (getter)THPNpuLstmCellBackward0_result4_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result4", (getter)THPNpuLstmCellBackward0_result4_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result5", (getter)THPNpuLstmCellBackward0_result5_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result5", (getter)THPNpuLstmCellBackward0_result5_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result6", (getter)THPNpuLstmCellBackward0_result6_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result6", (getter)THPNpuLstmCellBackward0_result6_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result7", (getter)THPNpuLstmCellBackward0_result7_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result7", (getter)THPNpuLstmCellBackward0_result7_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNpuMaxBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuMaxBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMaxBackward0_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuMaxBackward0*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMaxBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuMaxBackward0*>(self->cdata.get())->self_sym_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMaxBackward0_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMaxBackward0*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMaxBackward0_indices_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMaxBackward0*>(self->cdata.get())->indices_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuMaxBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPNpuMaxBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPNpuMaxBackward0_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPNpuMaxBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_indices", (getter)THPNpuMaxBackward0_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_indices", (getter)THPNpuMaxBackward0_indices_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNpuMishBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMishBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMishBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMishBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuMishBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPNpuMishBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPNpuMishBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNpuMultiHeadAttentionBackward0_attn_dim_per_head_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->attn_dim_per_head;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_attn_head_num_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->attn_head_num;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_dropout_prob_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->dropout_prob;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_key_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->key_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_key_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->key_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_key_bias_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->key_bias_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_key_bias_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->key_bias_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_key_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->key_weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_key_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->key_weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_out_proj_bias_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->out_proj_bias_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_out_proj_bias_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->out_proj_bias_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_out_proj_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->out_proj_weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_out_proj_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->out_proj_weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_query_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->query_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_query_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->query_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_query_bias_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->query_bias_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_query_bias_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->query_bias_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_query_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->query_weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_query_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->query_weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_softmax_use_float_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->softmax_use_float;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_src_len_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->src_len;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_tgt_len_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->tgt_len;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_value_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->value_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_value_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->value_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_value_bias_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->value_bias_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_value_bias_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->value_bias_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_value_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->value_weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_value_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->value_weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_result2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->result2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_result2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->result2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_result3_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->result3_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_result3_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->result3_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_result4_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->result4_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_result4_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->result4_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_result5_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->result5_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_result5_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->result5_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_result6_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->result6_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_result6_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->result6_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_result7_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->result7_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionBackward0_result7_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionBackward0*>(self->cdata.get())->result7_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuMultiHeadAttentionBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_attn_dim_per_head", (getter)THPNpuMultiHeadAttentionBackward0_attn_dim_per_head_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_attn_head_num", (getter)THPNpuMultiHeadAttentionBackward0_attn_head_num_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dropout_prob", (getter)THPNpuMultiHeadAttentionBackward0_dropout_prob_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_key", (getter)THPNpuMultiHeadAttentionBackward0_key_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_key", (getter)THPNpuMultiHeadAttentionBackward0_key_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_key_bias", (getter)THPNpuMultiHeadAttentionBackward0_key_bias_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_key_bias", (getter)THPNpuMultiHeadAttentionBackward0_key_bias_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_key_weight", (getter)THPNpuMultiHeadAttentionBackward0_key_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_key_weight", (getter)THPNpuMultiHeadAttentionBackward0_key_weight_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_out_proj_bias", (getter)THPNpuMultiHeadAttentionBackward0_out_proj_bias_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_out_proj_bias", (getter)THPNpuMultiHeadAttentionBackward0_out_proj_bias_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_out_proj_weight", (getter)THPNpuMultiHeadAttentionBackward0_out_proj_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_out_proj_weight", (getter)THPNpuMultiHeadAttentionBackward0_out_proj_weight_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_query", (getter)THPNpuMultiHeadAttentionBackward0_query_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_query", (getter)THPNpuMultiHeadAttentionBackward0_query_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_query_bias", (getter)THPNpuMultiHeadAttentionBackward0_query_bias_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_query_bias", (getter)THPNpuMultiHeadAttentionBackward0_query_bias_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_query_weight", (getter)THPNpuMultiHeadAttentionBackward0_query_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_query_weight", (getter)THPNpuMultiHeadAttentionBackward0_query_weight_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_softmax_use_float", (getter)THPNpuMultiHeadAttentionBackward0_softmax_use_float_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_src_len", (getter)THPNpuMultiHeadAttentionBackward0_src_len_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_tgt_len", (getter)THPNpuMultiHeadAttentionBackward0_tgt_len_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_value", (getter)THPNpuMultiHeadAttentionBackward0_value_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_value", (getter)THPNpuMultiHeadAttentionBackward0_value_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_value_bias", (getter)THPNpuMultiHeadAttentionBackward0_value_bias_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_value_bias", (getter)THPNpuMultiHeadAttentionBackward0_value_bias_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_value_weight", (getter)THPNpuMultiHeadAttentionBackward0_value_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_value_weight", (getter)THPNpuMultiHeadAttentionBackward0_value_weight_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPNpuMultiHeadAttentionBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPNpuMultiHeadAttentionBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result2", (getter)THPNpuMultiHeadAttentionBackward0_result2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result2", (getter)THPNpuMultiHeadAttentionBackward0_result2_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result3", (getter)THPNpuMultiHeadAttentionBackward0_result3_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result3", (getter)THPNpuMultiHeadAttentionBackward0_result3_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result4", (getter)THPNpuMultiHeadAttentionBackward0_result4_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result4", (getter)THPNpuMultiHeadAttentionBackward0_result4_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result5", (getter)THPNpuMultiHeadAttentionBackward0_result5_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result5", (getter)THPNpuMultiHeadAttentionBackward0_result5_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result6", (getter)THPNpuMultiHeadAttentionBackward0_result6_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result6", (getter)THPNpuMultiHeadAttentionBackward0_result6_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result7", (getter)THPNpuMultiHeadAttentionBackward0_result7_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result7", (getter)THPNpuMultiHeadAttentionBackward0_result7_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNpuRmsNormBackward0_gamma_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuRmsNormBackward0*>(self->cdata.get())->gamma_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuRmsNormBackward0_gamma_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuRmsNormBackward0*>(self->cdata.get())->gamma_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuRmsNormBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuRmsNormBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuRmsNormBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuRmsNormBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuRmsNormBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuRmsNormBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuRmsNormBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuRmsNormBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuRmsNormBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_gamma", (getter)THPNpuRmsNormBackward0_gamma_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_gamma", (getter)THPNpuRmsNormBackward0_gamma_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPNpuRmsNormBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPNpuRmsNormBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPNpuRmsNormBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPNpuRmsNormBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNpuSiluBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuSiluBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuSiluBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuSiluBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuSiluBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuSiluBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuSiluBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuSiluBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuSiluBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPNpuSiluBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPNpuSiluBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPNpuSiluBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPNpuSiluBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPRepeatInterleaveBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<RepeatInterleaveBackward0*>(self->cdata.get())->dim;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPRepeatInterleaveBackward0_repeats_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RepeatInterleaveBackward0*>(self->cdata.get())->repeats_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPRepeatInterleaveBackward0_repeats_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RepeatInterleaveBackward0*>(self->cdata.get())->repeats_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPRepeatInterleaveBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RepeatInterleaveBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPRepeatInterleaveBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RepeatInterleaveBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef RepeatInterleaveBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPRepeatInterleaveBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_repeats", (getter)THPRepeatInterleaveBackward0_repeats_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_repeats", (getter)THPRepeatInterleaveBackward0_repeats_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPRepeatInterleaveBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPRepeatInterleaveBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPRepeatInterleaveBackward1_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<RepeatInterleaveBackward1*>(self->cdata.get())->dim;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPRepeatInterleaveBackward1_repeats_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RepeatInterleaveBackward1*>(self->cdata.get())->repeats;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPRepeatInterleaveBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RepeatInterleaveBackward1*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPRepeatInterleaveBackward1_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RepeatInterleaveBackward1*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef RepeatInterleaveBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPRepeatInterleaveBackward1_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_repeats", (getter)THPRepeatInterleaveBackward1_repeats_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPRepeatInterleaveBackward1_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPRepeatInterleaveBackward1_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPFftR2CBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FftR2CBackward0*>(self->cdata.get())->dim;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPFftR2CBackward0_normalization_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FftR2CBackward0*>(self->cdata.get())->normalization;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPFftR2CBackward0_onesided_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FftR2CBackward0*>(self->cdata.get())->onesided;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPFftR2CBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FftR2CBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFftR2CBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FftR2CBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FftR2CBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPFftR2CBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_normalization", (getter)THPFftR2CBackward0_normalization_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_onesided", (getter)THPFftR2CBackward0_onesided_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPFftR2CBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPFftR2CBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNpuNsaCompressBackward0_actual_seq_len_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<NpuNsaCompressBackward0*>(self->cdata.get())->actual_seq_len;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaCompressBackward0_compress_block_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuNsaCompressBackward0*>(self->cdata.get())->compress_block_size;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaCompressBackward0_compress_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuNsaCompressBackward0*>(self->cdata.get())->compress_stride;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaCompressBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuNsaCompressBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaCompressBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuNsaCompressBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaCompressBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuNsaCompressBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaCompressBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuNsaCompressBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuNsaCompressBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_actual_seq_len", (getter)THPNpuNsaCompressBackward0_actual_seq_len_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_compress_block_size", (getter)THPNpuNsaCompressBackward0_compress_block_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_compress_stride", (getter)THPNpuNsaCompressBackward0_compress_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPNpuNsaCompressBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPNpuNsaCompressBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPNpuNsaCompressBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPNpuNsaCompressBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

void initialize_autogenerated_functions_1(PyObject *module)
{
    static PyTypeObject GatherBackward0Class;
    addClass<GatherBackward0>(module, GatherBackward0Class, "GatherBackward0", GatherBackward0_properties);
    static PyTypeObject NpuCiouBackward0Class;
    addClass<NpuCiouBackward0>(module, NpuCiouBackward0Class, "NpuCiouBackward0", NpuCiouBackward0_properties);
    static PyTypeObject NpuDropoutBackward0Class;
    addClass<NpuDropoutBackward0>(module, NpuDropoutBackward0Class, "NpuDropoutBackward0", NpuDropoutBackward0_properties);
    static PyTypeObject MatmulBackwardBackward0Class;
    addClass<MatmulBackwardBackward0>(module, MatmulBackwardBackward0Class, "MatmulBackwardBackward0", MatmulBackwardBackward0_properties);
    static PyTypeObject NpuBmmv2Backward0Class;
    addClass<NpuBmmv2Backward0>(module, NpuBmmv2Backward0Class, "NpuBmmv2Backward0", NpuBmmv2Backward0_properties);
    static PyTypeObject NpuConvolutionBackward0Class;
    addClass<NpuConvolutionBackward0>(module, NpuConvolutionBackward0Class, "NpuConvolutionBackward0", NpuConvolutionBackward0_properties);
    static PyTypeObject NpuConvolutionTransposeBackward0Class;
    addClass<NpuConvolutionTransposeBackward0>(module, NpuConvolutionTransposeBackward0Class, "NpuConvolutionTransposeBackward0", NpuConvolutionTransposeBackward0_properties);
    static PyTypeObject NpuDropoutDoMaskBackward0Class;
    addClass<NpuDropoutDoMaskBackward0>(module, NpuDropoutDoMaskBackward0Class, "NpuDropoutDoMaskBackward0", NpuDropoutDoMaskBackward0_properties);
    static PyTypeObject NpuDropoutWithAddSoftmaxBackward0Class;
    addClass<NpuDropoutWithAddSoftmaxBackward0>(module, NpuDropoutWithAddSoftmaxBackward0Class, "NpuDropoutWithAddSoftmaxBackward0", NpuDropoutWithAddSoftmaxBackward0_properties);
    static PyTypeObject NpuDtypeCastBackward0Class;
    addClass<NpuDtypeCastBackward0>(module, NpuDtypeCastBackward0Class, "NpuDtypeCastBackward0", NpuDtypeCastBackward0_properties);
    static PyTypeObject NpuFusedAttentionScoreFwdBackward0Class;
    addClass<NpuFusedAttentionScoreFwdBackward0>(module, NpuFusedAttentionScoreFwdBackward0Class, "NpuFusedAttentionScoreFwdBackward0", NpuFusedAttentionScoreFwdBackward0_properties);
    static PyTypeObject NpuGiouBackward0Class;
    addClass<NpuGiouBackward0>(module, NpuGiouBackward0Class, "NpuGiouBackward0", NpuGiouBackward0_properties);
    static PyTypeObject NpuGruBackward0Class;
    addClass<NpuGruBackward0>(module, NpuGruBackward0Class, "NpuGruBackward0", NpuGruBackward0_properties);
    static PyTypeObject NpuLinearBackward0Class;
    addClass<NpuLinearBackward0>(module, NpuLinearBackward0Class, "NpuLinearBackward0", NpuLinearBackward0_properties);
    static PyTypeObject NpuLstmCellBackward0Class;
    addClass<NpuLstmCellBackward0>(module, NpuLstmCellBackward0Class, "NpuLstmCellBackward0", NpuLstmCellBackward0_properties);
    static PyTypeObject NpuMaxBackward0Class;
    addClass<NpuMaxBackward0>(module, NpuMaxBackward0Class, "NpuMaxBackward0", NpuMaxBackward0_properties);
    static PyTypeObject NpuMishBackward0Class;
    addClass<NpuMishBackward0>(module, NpuMishBackward0Class, "NpuMishBackward0", NpuMishBackward0_properties);
    static PyTypeObject NpuMultiHeadAttentionBackward0Class;
    addClass<NpuMultiHeadAttentionBackward0>(module, NpuMultiHeadAttentionBackward0Class, "NpuMultiHeadAttentionBackward0", NpuMultiHeadAttentionBackward0_properties);
    static PyTypeObject NpuRmsNormBackward0Class;
    addClass<NpuRmsNormBackward0>(module, NpuRmsNormBackward0Class, "NpuRmsNormBackward0", NpuRmsNormBackward0_properties);
    static PyTypeObject NpuSiluBackward0Class;
    addClass<NpuSiluBackward0>(module, NpuSiluBackward0Class, "NpuSiluBackward0", NpuSiluBackward0_properties);
    static PyTypeObject RepeatInterleaveBackward0Class;
    addClass<RepeatInterleaveBackward0>(module, RepeatInterleaveBackward0Class, "RepeatInterleaveBackward0", RepeatInterleaveBackward0_properties);
    static PyTypeObject RepeatInterleaveBackward1Class;
    addClass<RepeatInterleaveBackward1>(module, RepeatInterleaveBackward1Class, "RepeatInterleaveBackward1", RepeatInterleaveBackward1_properties);
    static PyTypeObject FftR2CBackward0Class;
    addClass<FftR2CBackward0>(module, FftR2CBackward0Class, "FftR2CBackward0", FftR2CBackward0_properties);
    static PyTypeObject NpuNsaCompressBackward0Class;
    addClass<NpuNsaCompressBackward0>(module, NpuNsaCompressBackward0Class, "NpuNsaCompressBackward0", NpuNsaCompressBackward0_properties);
}
}
}
} // namespace at_npu::autograd::generated

#endif
