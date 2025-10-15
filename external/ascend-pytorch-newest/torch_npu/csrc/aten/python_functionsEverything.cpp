#ifndef BUILD_LIBTORCH
#include <torch/csrc/autograd/generated/python_functions.h>

// @ generated from ../../../../../../../root/workspace/vllm-ascend/examples/develop/xzh_test/cann-ops-adv-vllm/external/ascend-pytorch-newest/codegen/autograd/templates/python_functions.cpp

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

PyObject* THPDropoutWithByteMaskBackward0_p_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<DropoutWithByteMaskBackward0*>(self->cdata.get())->p;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPDropoutWithByteMaskBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<DropoutWithByteMaskBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPDropoutWithByteMaskBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<DropoutWithByteMaskBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef DropoutWithByteMaskBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_p", (getter)THPDropoutWithByteMaskBackward0_p_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPDropoutWithByteMaskBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPDropoutWithByteMaskBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
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



static struct PyGetSetDef NpuFormatCastBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPBinaryCrossEntropyWithLogitsBackward0_pos_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyWithLogitsBackward0*>(self->cdata.get())->pos_weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyWithLogitsBackward0_pos_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyWithLogitsBackward0*>(self->cdata.get())->pos_weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyWithLogitsBackward0_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<BinaryCrossEntropyWithLogitsBackward0*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyWithLogitsBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyWithLogitsBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyWithLogitsBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyWithLogitsBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyWithLogitsBackward0_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyWithLogitsBackward0*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyWithLogitsBackward0_target_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyWithLogitsBackward0*>(self->cdata.get())->target_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyWithLogitsBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyWithLogitsBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyWithLogitsBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyWithLogitsBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef BinaryCrossEntropyWithLogitsBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_pos_weight", (getter)THPBinaryCrossEntropyWithLogitsBackward0_pos_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_pos_weight", (getter)THPBinaryCrossEntropyWithLogitsBackward0_pos_weight_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPBinaryCrossEntropyWithLogitsBackward0_reduction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPBinaryCrossEntropyWithLogitsBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPBinaryCrossEntropyWithLogitsBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPBinaryCrossEntropyWithLogitsBackward0_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_target", (getter)THPBinaryCrossEntropyWithLogitsBackward0_target_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPBinaryCrossEntropyWithLogitsBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPBinaryCrossEntropyWithLogitsBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPFastGeluBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FastGeluBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFastGeluBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FastGeluBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FastGeluBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPFastGeluBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPFastGeluBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPKlDivBackward0_log_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<KlDivBackward0*>(self->cdata.get())->log_target;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPKlDivBackward0_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<KlDivBackward0*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPKlDivBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<KlDivBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPKlDivBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<KlDivBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPKlDivBackward0_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<KlDivBackward0*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPKlDivBackward0_target_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<KlDivBackward0*>(self->cdata.get())->target_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef KlDivBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_log_target", (getter)THPKlDivBackward0_log_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPKlDivBackward0_reduction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPKlDivBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPKlDivBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPKlDivBackward0_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_target", (getter)THPKlDivBackward0_target_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPL1LossBackward0_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<L1LossBackward0*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPL1LossBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<L1LossBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPL1LossBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<L1LossBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPL1LossBackward0_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<L1LossBackward0*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPL1LossBackward0_target_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<L1LossBackward0*>(self->cdata.get())->target_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef L1LossBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_reduction", (getter)THPL1LossBackward0_reduction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPL1LossBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPL1LossBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPL1LossBackward0_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_target", (getter)THPL1LossBackward0_target_raw_getter, nullptr, nullptr, nullptr},
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

PyObject* THPNpuAddLayerNormBackward0_gamma_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuAddLayerNormBackward0*>(self->cdata.get())->gamma_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuAddLayerNormBackward0_gamma_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuAddLayerNormBackward0*>(self->cdata.get())->gamma_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuAddLayerNormBackward0_x1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuAddLayerNormBackward0*>(self->cdata.get())->x1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuAddLayerNormBackward0_x1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuAddLayerNormBackward0*>(self->cdata.get())->x1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuAddLayerNormBackward0_x2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuAddLayerNormBackward0*>(self->cdata.get())->x2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuAddLayerNormBackward0_x2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuAddLayerNormBackward0*>(self->cdata.get())->x2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuAddLayerNormBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuAddLayerNormBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuAddLayerNormBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuAddLayerNormBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuAddLayerNormBackward0_result2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuAddLayerNormBackward0*>(self->cdata.get())->result2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuAddLayerNormBackward0_result2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuAddLayerNormBackward0*>(self->cdata.get())->result2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuAddLayerNormBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_gamma", (getter)THPNpuAddLayerNormBackward0_gamma_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_gamma", (getter)THPNpuAddLayerNormBackward0_gamma_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_x1", (getter)THPNpuAddLayerNormBackward0_x1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_x1", (getter)THPNpuAddLayerNormBackward0_x1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_x2", (getter)THPNpuAddLayerNormBackward0_x2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_x2", (getter)THPNpuAddLayerNormBackward0_x2_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPNpuAddLayerNormBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPNpuAddLayerNormBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result2", (getter)THPNpuAddLayerNormBackward0_result2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result2", (getter)THPNpuAddLayerNormBackward0_result2_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNpuGeluBackward0_approximate_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuGeluBackward0*>(self->cdata.get())->approximate;
  return PyUnicode_FromStringAndSize(prop.data(), prop.size());
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGeluBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGeluBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGeluBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGeluBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuGeluBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_approximate", (getter)THPNpuGeluBackward0_approximate_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPNpuGeluBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPNpuGeluBackward0_self_raw_getter, nullptr, nullptr, nullptr},
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

PyObject* THPNpuConfusionTransposeBackward0_perm_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuConfusionTransposeBackward0*>(self->cdata.get())->perm;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuConfusionTransposeBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuConfusionTransposeBackward0*>(self->cdata.get())->self_sym_sizes;
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

PyObject* THPNpuConfusionTransposeBackward0_transpose_first_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuConfusionTransposeBackward0*>(self->cdata.get())->transpose_first;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuConfusionTransposeBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_perm", (getter)THPNpuConfusionTransposeBackward0_perm_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPNpuConfusionTransposeBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_transpose_first", (getter)THPNpuConfusionTransposeBackward0_transpose_first_getter, nullptr, nullptr, nullptr},
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

PyObject* THPNpuDeepNormBackward0_alpha_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuDeepNormBackward0*>(self->cdata.get())->alpha;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDeepNormBackward0_gamma_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuDeepNormBackward0*>(self->cdata.get())->gamma_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDeepNormBackward0_gamma_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuDeepNormBackward0*>(self->cdata.get())->gamma_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDeepNormBackward0_gx_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuDeepNormBackward0*>(self->cdata.get())->gx_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDeepNormBackward0_gx_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuDeepNormBackward0*>(self->cdata.get())->gx_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDeepNormBackward0_x_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuDeepNormBackward0*>(self->cdata.get())->x_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDeepNormBackward0_x_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuDeepNormBackward0*>(self->cdata.get())->x_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDeepNormBackward0_result0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuDeepNormBackward0*>(self->cdata.get())->result0_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDeepNormBackward0_result0_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuDeepNormBackward0*>(self->cdata.get())->result0_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDeepNormBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuDeepNormBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDeepNormBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuDeepNormBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuDeepNormBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_alpha", (getter)THPNpuDeepNormBackward0_alpha_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_gamma", (getter)THPNpuDeepNormBackward0_gamma_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_gamma", (getter)THPNpuDeepNormBackward0_gamma_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_gx", (getter)THPNpuDeepNormBackward0_gx_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_gx", (getter)THPNpuDeepNormBackward0_gx_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_x", (getter)THPNpuDeepNormBackward0_x_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_x", (getter)THPNpuDeepNormBackward0_x_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result0", (getter)THPNpuDeepNormBackward0_result0_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result0", (getter)THPNpuDeepNormBackward0_result0_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPNpuDeepNormBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPNpuDeepNormBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNpuDeformableConv2DBackward0_deformable_groups_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuDeformableConv2DBackward0*>(self->cdata.get())->deformable_groups;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDeformableConv2DBackward0_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuDeformableConv2DBackward0*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDeformableConv2DBackward0_groups_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuDeformableConv2DBackward0*>(self->cdata.get())->groups;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDeformableConv2DBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuDeformableConv2DBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDeformableConv2DBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuDeformableConv2DBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDeformableConv2DBackward0_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuDeformableConv2DBackward0*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDeformableConv2DBackward0_modulated_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuDeformableConv2DBackward0*>(self->cdata.get())->modulated;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDeformableConv2DBackward0_offset_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuDeformableConv2DBackward0*>(self->cdata.get())->offset_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDeformableConv2DBackward0_offset_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuDeformableConv2DBackward0*>(self->cdata.get())->offset_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDeformableConv2DBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuDeformableConv2DBackward0*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDeformableConv2DBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuDeformableConv2DBackward0*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDeformableConv2DBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuDeformableConv2DBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDeformableConv2DBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuDeformableConv2DBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDeformableConv2DBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuDeformableConv2DBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDeformableConv2DBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuDeformableConv2DBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuDeformableConv2DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_deformable_groups", (getter)THPNpuDeformableConv2DBackward0_deformable_groups_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPNpuDeformableConv2DBackward0_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_groups", (getter)THPNpuDeformableConv2DBackward0_groups_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPNpuDeformableConv2DBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPNpuDeformableConv2DBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_kernel_size", (getter)THPNpuDeformableConv2DBackward0_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_modulated", (getter)THPNpuDeformableConv2DBackward0_modulated_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_offset", (getter)THPNpuDeformableConv2DBackward0_offset_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_offset", (getter)THPNpuDeformableConv2DBackward0_offset_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPNpuDeformableConv2DBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPNpuDeformableConv2DBackward0_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPNpuDeformableConv2DBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPNpuDeformableConv2DBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPNpuDeformableConv2DBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPNpuDeformableConv2DBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNpuDiouBackward0_gtboxes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuDiouBackward0*>(self->cdata.get())->gtboxes_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDiouBackward0_gtboxes_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuDiouBackward0*>(self->cdata.get())->gtboxes_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDiouBackward0_is_cross_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuDiouBackward0*>(self->cdata.get())->is_cross;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDiouBackward0_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuDiouBackward0*>(self->cdata.get())->mode;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDiouBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuDiouBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDiouBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuDiouBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuDiouBackward0_trans_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuDiouBackward0*>(self->cdata.get())->trans;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuDiouBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_gtboxes", (getter)THPNpuDiouBackward0_gtboxes_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_gtboxes", (getter)THPNpuDiouBackward0_gtboxes_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_is_cross", (getter)THPNpuDiouBackward0_is_cross_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mode", (getter)THPNpuDiouBackward0_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPNpuDiouBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPNpuDiouBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_trans", (getter)THPNpuDiouBackward0_trans_getter, nullptr, nullptr, nullptr},
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

PyObject* THPNpuFusionAttentionBackward0_actual_seq_kvlen_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<NpuFusionAttentionBackward0*>(self->cdata.get())->actual_seq_kvlen;
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

PyObject* THPNpuFusionAttentionBackward0_actual_seq_qlen_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<NpuFusionAttentionBackward0*>(self->cdata.get())->actual_seq_qlen;
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

PyObject* THPNpuFusionAttentionBackward0_atten_mask_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionBackward0*>(self->cdata.get())->atten_mask_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionBackward0_atten_mask_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionBackward0*>(self->cdata.get())->atten_mask_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionBackward0_gen_mask_parallel_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuFusionAttentionBackward0*>(self->cdata.get())->gen_mask_parallel;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionBackward0_head_num_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuFusionAttentionBackward0*>(self->cdata.get())->head_num;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionBackward0_inner_precise_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuFusionAttentionBackward0*>(self->cdata.get())->inner_precise;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionBackward0_input_layout_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuFusionAttentionBackward0*>(self->cdata.get())->input_layout;
  return PyUnicode_FromStringAndSize(prop.data(), prop.size());
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionBackward0_keep_prob_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuFusionAttentionBackward0*>(self->cdata.get())->keep_prob;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionBackward0_key_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionBackward0*>(self->cdata.get())->key_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionBackward0_key_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionBackward0*>(self->cdata.get())->key_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionBackward0_next_tockens_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuFusionAttentionBackward0*>(self->cdata.get())->next_tockens;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionBackward0_padding_mask_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionBackward0*>(self->cdata.get())->padding_mask_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionBackward0_padding_mask_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionBackward0*>(self->cdata.get())->padding_mask_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionBackward0_pre_tockens_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuFusionAttentionBackward0*>(self->cdata.get())->pre_tockens;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionBackward0_prefix_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<NpuFusionAttentionBackward0*>(self->cdata.get())->prefix;
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

PyObject* THPNpuFusionAttentionBackward0_pse_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionBackward0*>(self->cdata.get())->pse_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionBackward0_pse_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionBackward0*>(self->cdata.get())->pse_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionBackward0_query_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionBackward0*>(self->cdata.get())->query_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionBackward0_query_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionBackward0*>(self->cdata.get())->query_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionBackward0_scale_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuFusionAttentionBackward0*>(self->cdata.get())->scale;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionBackward0_sparse_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuFusionAttentionBackward0*>(self->cdata.get())->sparse_mode;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionBackward0_sync_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuFusionAttentionBackward0*>(self->cdata.get())->sync;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionBackward0_value_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionBackward0*>(self->cdata.get())->value_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionBackward0_value_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionBackward0*>(self->cdata.get())->value_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionBackward0_result0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionBackward0*>(self->cdata.get())->result0_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionBackward0_result0_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionBackward0*>(self->cdata.get())->result0_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionBackward0_result2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionBackward0*>(self->cdata.get())->result2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionBackward0_result2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionBackward0*>(self->cdata.get())->result2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionBackward0_result3_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionBackward0*>(self->cdata.get())->result3_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionBackward0_result3_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionBackward0*>(self->cdata.get())->result3_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionBackward0_result4_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuFusionAttentionBackward0*>(self->cdata.get())->result4;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionBackward0_result5_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuFusionAttentionBackward0*>(self->cdata.get())->result5;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionBackward0_result6_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuFusionAttentionBackward0*>(self->cdata.get())->result6;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuFusionAttentionBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_actual_seq_kvlen", (getter)THPNpuFusionAttentionBackward0_actual_seq_kvlen_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_actual_seq_qlen", (getter)THPNpuFusionAttentionBackward0_actual_seq_qlen_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_atten_mask", (getter)THPNpuFusionAttentionBackward0_atten_mask_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_atten_mask", (getter)THPNpuFusionAttentionBackward0_atten_mask_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_gen_mask_parallel", (getter)THPNpuFusionAttentionBackward0_gen_mask_parallel_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_head_num", (getter)THPNpuFusionAttentionBackward0_head_num_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_inner_precise", (getter)THPNpuFusionAttentionBackward0_inner_precise_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input_layout", (getter)THPNpuFusionAttentionBackward0_input_layout_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keep_prob", (getter)THPNpuFusionAttentionBackward0_keep_prob_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_key", (getter)THPNpuFusionAttentionBackward0_key_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_key", (getter)THPNpuFusionAttentionBackward0_key_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_next_tockens", (getter)THPNpuFusionAttentionBackward0_next_tockens_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding_mask", (getter)THPNpuFusionAttentionBackward0_padding_mask_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_padding_mask", (getter)THPNpuFusionAttentionBackward0_padding_mask_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_pre_tockens", (getter)THPNpuFusionAttentionBackward0_pre_tockens_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_prefix", (getter)THPNpuFusionAttentionBackward0_prefix_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_pse", (getter)THPNpuFusionAttentionBackward0_pse_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_pse", (getter)THPNpuFusionAttentionBackward0_pse_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_query", (getter)THPNpuFusionAttentionBackward0_query_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_query", (getter)THPNpuFusionAttentionBackward0_query_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale", (getter)THPNpuFusionAttentionBackward0_scale_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_sparse_mode", (getter)THPNpuFusionAttentionBackward0_sparse_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_sync", (getter)THPNpuFusionAttentionBackward0_sync_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_value", (getter)THPNpuFusionAttentionBackward0_value_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_value", (getter)THPNpuFusionAttentionBackward0_value_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result0", (getter)THPNpuFusionAttentionBackward0_result0_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result0", (getter)THPNpuFusionAttentionBackward0_result0_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPNpuFusionAttentionBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPNpuFusionAttentionBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result2", (getter)THPNpuFusionAttentionBackward0_result2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result2", (getter)THPNpuFusionAttentionBackward0_result2_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result3", (getter)THPNpuFusionAttentionBackward0_result3_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result3", (getter)THPNpuFusionAttentionBackward0_result3_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result4", (getter)THPNpuFusionAttentionBackward0_result4_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result5", (getter)THPNpuFusionAttentionBackward0_result5_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result6", (getter)THPNpuFusionAttentionBackward0_result6_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNpuFusionAttentionV2Backward0_actual_seq_kvlen_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->actual_seq_kvlen;
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

PyObject* THPNpuFusionAttentionV2Backward0_actual_seq_qlen_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->actual_seq_qlen;
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

PyObject* THPNpuFusionAttentionV2Backward0_atten_mask_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->atten_mask_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_atten_mask_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->atten_mask_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_gen_mask_parallel_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->gen_mask_parallel;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_head_num_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->head_num;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_inner_precise_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->inner_precise;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_input_layout_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->input_layout;
  return PyUnicode_FromStringAndSize(prop.data(), prop.size());
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_keep_prob_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->keep_prob;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_key_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->key_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_key_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->key_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_key_rope_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->key_rope_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_key_rope_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->key_rope_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_kv_start_idx_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->kv_start_idx;
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

PyObject* THPNpuFusionAttentionV2Backward0_next_tokens_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->next_tokens;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_padding_mask_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->padding_mask_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_padding_mask_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->padding_mask_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_pre_tokens_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->pre_tokens;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_prefix_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->prefix;
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

PyObject* THPNpuFusionAttentionV2Backward0_pse_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->pse_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_pse_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->pse_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_pse_type_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->pse_type;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_q_start_idx_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->q_start_idx;
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

PyObject* THPNpuFusionAttentionV2Backward0_query_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->query_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_query_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->query_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_query_rope_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->query_rope_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_query_rope_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->query_rope_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_scale_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->scale;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_sparse_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->sparse_mode;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_sync_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->sync;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_value_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->value_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_value_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->value_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_result0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->result0_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_result0_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->result0_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_result2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->result2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_result2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->result2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_result3_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->result3_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_result3_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->result3_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_result4_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->result4;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_result5_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->result5;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuFusionAttentionV2Backward0_result6_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuFusionAttentionV2Backward0*>(self->cdata.get())->result6;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuFusionAttentionV2Backward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_actual_seq_kvlen", (getter)THPNpuFusionAttentionV2Backward0_actual_seq_kvlen_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_actual_seq_qlen", (getter)THPNpuFusionAttentionV2Backward0_actual_seq_qlen_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_atten_mask", (getter)THPNpuFusionAttentionV2Backward0_atten_mask_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_atten_mask", (getter)THPNpuFusionAttentionV2Backward0_atten_mask_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_gen_mask_parallel", (getter)THPNpuFusionAttentionV2Backward0_gen_mask_parallel_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_head_num", (getter)THPNpuFusionAttentionV2Backward0_head_num_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_inner_precise", (getter)THPNpuFusionAttentionV2Backward0_inner_precise_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input_layout", (getter)THPNpuFusionAttentionV2Backward0_input_layout_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keep_prob", (getter)THPNpuFusionAttentionV2Backward0_keep_prob_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_key", (getter)THPNpuFusionAttentionV2Backward0_key_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_key", (getter)THPNpuFusionAttentionV2Backward0_key_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_key_rope", (getter)THPNpuFusionAttentionV2Backward0_key_rope_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_key_rope", (getter)THPNpuFusionAttentionV2Backward0_key_rope_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_kv_start_idx", (getter)THPNpuFusionAttentionV2Backward0_kv_start_idx_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_next_tokens", (getter)THPNpuFusionAttentionV2Backward0_next_tokens_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding_mask", (getter)THPNpuFusionAttentionV2Backward0_padding_mask_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_padding_mask", (getter)THPNpuFusionAttentionV2Backward0_padding_mask_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_pre_tokens", (getter)THPNpuFusionAttentionV2Backward0_pre_tokens_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_prefix", (getter)THPNpuFusionAttentionV2Backward0_prefix_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_pse", (getter)THPNpuFusionAttentionV2Backward0_pse_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_pse", (getter)THPNpuFusionAttentionV2Backward0_pse_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_pse_type", (getter)THPNpuFusionAttentionV2Backward0_pse_type_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_q_start_idx", (getter)THPNpuFusionAttentionV2Backward0_q_start_idx_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_query", (getter)THPNpuFusionAttentionV2Backward0_query_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_query", (getter)THPNpuFusionAttentionV2Backward0_query_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_query_rope", (getter)THPNpuFusionAttentionV2Backward0_query_rope_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_query_rope", (getter)THPNpuFusionAttentionV2Backward0_query_rope_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale", (getter)THPNpuFusionAttentionV2Backward0_scale_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_sparse_mode", (getter)THPNpuFusionAttentionV2Backward0_sparse_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_sync", (getter)THPNpuFusionAttentionV2Backward0_sync_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_value", (getter)THPNpuFusionAttentionV2Backward0_value_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_value", (getter)THPNpuFusionAttentionV2Backward0_value_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result0", (getter)THPNpuFusionAttentionV2Backward0_result0_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result0", (getter)THPNpuFusionAttentionV2Backward0_result0_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPNpuFusionAttentionV2Backward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPNpuFusionAttentionV2Backward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result2", (getter)THPNpuFusionAttentionV2Backward0_result2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result2", (getter)THPNpuFusionAttentionV2Backward0_result2_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result3", (getter)THPNpuFusionAttentionV2Backward0_result3_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result3", (getter)THPNpuFusionAttentionV2Backward0_result3_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result4", (getter)THPNpuFusionAttentionV2Backward0_result4_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result5", (getter)THPNpuFusionAttentionV2Backward0_result5_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result6", (getter)THPNpuFusionAttentionV2Backward0_result6_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNpuGegluBackward0_activate_left_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuGegluBackward0*>(self->cdata.get())->activate_left;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGegluBackward0_approximate_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuGegluBackward0*>(self->cdata.get())->approximate;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGegluBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuGegluBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGegluBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGegluBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGegluBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGegluBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGegluBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGegluBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGegluBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGegluBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuGegluBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_activate_left", (getter)THPNpuGegluBackward0_activate_left_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_approximate", (getter)THPNpuGegluBackward0_approximate_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPNpuGegluBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPNpuGegluBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPNpuGegluBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPNpuGegluBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPNpuGegluBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
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

PyObject* THPNpuLstmBackward0_bias_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmBackward0*>(self->cdata.get())->bias_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmBackward0_bias_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmBackward0*>(self->cdata.get())->bias_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmBackward0_c_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmBackward0*>(self->cdata.get())->c_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmBackward0_c_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmBackward0*>(self->cdata.get())->c_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmBackward0_h_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmBackward0*>(self->cdata.get())->h_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmBackward0_h_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmBackward0*>(self->cdata.get())->h_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmBackward0_result0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmBackward0*>(self->cdata.get())->result0_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmBackward0_result0_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmBackward0*>(self->cdata.get())->result0_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmBackward0_result2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmBackward0*>(self->cdata.get())->result2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmBackward0_result2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmBackward0*>(self->cdata.get())->result2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmBackward0_result3_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmBackward0*>(self->cdata.get())->result3_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmBackward0_result3_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmBackward0*>(self->cdata.get())->result3_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmBackward0_result4_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmBackward0*>(self->cdata.get())->result4_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmBackward0_result4_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmBackward0*>(self->cdata.get())->result4_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmBackward0_result5_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmBackward0*>(self->cdata.get())->result5_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmBackward0_result5_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmBackward0*>(self->cdata.get())->result5_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmBackward0_result6_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmBackward0*>(self->cdata.get())->result6_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmBackward0_result6_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmBackward0*>(self->cdata.get())->result6_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmBackward0_result7_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmBackward0*>(self->cdata.get())->result7_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmBackward0_result7_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmBackward0*>(self->cdata.get())->result7_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuLstmBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_bias", (getter)THPNpuLstmBackward0_bias_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_bias", (getter)THPNpuLstmBackward0_bias_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_c", (getter)THPNpuLstmBackward0_c_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_c", (getter)THPNpuLstmBackward0_c_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_h", (getter)THPNpuLstmBackward0_h_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_h", (getter)THPNpuLstmBackward0_h_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPNpuLstmBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPNpuLstmBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPNpuLstmBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPNpuLstmBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result0", (getter)THPNpuLstmBackward0_result0_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result0", (getter)THPNpuLstmBackward0_result0_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPNpuLstmBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPNpuLstmBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result2", (getter)THPNpuLstmBackward0_result2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result2", (getter)THPNpuLstmBackward0_result2_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result3", (getter)THPNpuLstmBackward0_result3_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result3", (getter)THPNpuLstmBackward0_result3_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result4", (getter)THPNpuLstmBackward0_result4_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result4", (getter)THPNpuLstmBackward0_result4_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result5", (getter)THPNpuLstmBackward0_result5_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result5", (getter)THPNpuLstmBackward0_result5_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result6", (getter)THPNpuLstmBackward0_result6_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result6", (getter)THPNpuLstmBackward0_result6_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result7", (getter)THPNpuLstmBackward0_result7_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result7", (getter)THPNpuLstmBackward0_result7_raw_getter, nullptr, nullptr, nullptr},
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

PyObject* THPNpuLstmDataBackward0_batch_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmDataBackward0*>(self->cdata.get())->batch_sizes_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmDataBackward0_batch_sizes_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmDataBackward0*>(self->cdata.get())->batch_sizes_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmDataBackward0_bias_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmDataBackward0*>(self->cdata.get())->bias_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmDataBackward0_bias_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmDataBackward0*>(self->cdata.get())->bias_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmDataBackward0_c_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmDataBackward0*>(self->cdata.get())->c_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmDataBackward0_c_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmDataBackward0*>(self->cdata.get())->c_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmDataBackward0_direction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuLstmDataBackward0*>(self->cdata.get())->direction;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmDataBackward0_h_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmDataBackward0*>(self->cdata.get())->h_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmDataBackward0_h_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmDataBackward0*>(self->cdata.get())->h_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmDataBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmDataBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmDataBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmDataBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmDataBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmDataBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmDataBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmDataBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmDataBackward0_result0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmDataBackward0*>(self->cdata.get())->result0_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmDataBackward0_result0_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmDataBackward0*>(self->cdata.get())->result0_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmDataBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmDataBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmDataBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmDataBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmDataBackward0_result2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmDataBackward0*>(self->cdata.get())->result2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmDataBackward0_result2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmDataBackward0*>(self->cdata.get())->result2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmDataBackward0_result3_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmDataBackward0*>(self->cdata.get())->result3_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmDataBackward0_result3_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmDataBackward0*>(self->cdata.get())->result3_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmDataBackward0_result4_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmDataBackward0*>(self->cdata.get())->result4_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmDataBackward0_result4_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmDataBackward0*>(self->cdata.get())->result4_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmDataBackward0_result5_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmDataBackward0*>(self->cdata.get())->result5_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmDataBackward0_result5_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmDataBackward0*>(self->cdata.get())->result5_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmDataBackward0_result6_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmDataBackward0*>(self->cdata.get())->result6_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmDataBackward0_result6_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmDataBackward0*>(self->cdata.get())->result6_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmDataBackward0_result7_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmDataBackward0*>(self->cdata.get())->result7_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuLstmDataBackward0_result7_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuLstmDataBackward0*>(self->cdata.get())->result7_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuLstmDataBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_batch_sizes", (getter)THPNpuLstmDataBackward0_batch_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_batch_sizes", (getter)THPNpuLstmDataBackward0_batch_sizes_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_bias", (getter)THPNpuLstmDataBackward0_bias_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_bias", (getter)THPNpuLstmDataBackward0_bias_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_c", (getter)THPNpuLstmDataBackward0_c_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_c", (getter)THPNpuLstmDataBackward0_c_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_direction", (getter)THPNpuLstmDataBackward0_direction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_h", (getter)THPNpuLstmDataBackward0_h_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_h", (getter)THPNpuLstmDataBackward0_h_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPNpuLstmDataBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPNpuLstmDataBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPNpuLstmDataBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPNpuLstmDataBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result0", (getter)THPNpuLstmDataBackward0_result0_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result0", (getter)THPNpuLstmDataBackward0_result0_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPNpuLstmDataBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPNpuLstmDataBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result2", (getter)THPNpuLstmDataBackward0_result2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result2", (getter)THPNpuLstmDataBackward0_result2_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result3", (getter)THPNpuLstmDataBackward0_result3_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result3", (getter)THPNpuLstmDataBackward0_result3_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result4", (getter)THPNpuLstmDataBackward0_result4_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result4", (getter)THPNpuLstmDataBackward0_result4_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result5", (getter)THPNpuLstmDataBackward0_result5_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result5", (getter)THPNpuLstmDataBackward0_result5_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result6", (getter)THPNpuLstmDataBackward0_result6_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result6", (getter)THPNpuLstmDataBackward0_result6_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result7", (getter)THPNpuLstmDataBackward0_result7_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result7", (getter)THPNpuLstmDataBackward0_result7_raw_getter, nullptr, nullptr, nullptr},
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

PyObject* THPNpuMinBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuMinBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMinBackward0_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuMinBackward0*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMinBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuMinBackward0*>(self->cdata.get())->self_sym_sizes;
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

PyObject* THPNpuMinBackward0_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMinBackward0*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMinBackward0_indices_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMinBackward0*>(self->cdata.get())->indices_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuMinBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPNpuMinBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPNpuMinBackward0_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPNpuMinBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_indices", (getter)THPNpuMinBackward0_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_indices", (getter)THPNpuMinBackward0_indices_raw_getter, nullptr, nullptr, nullptr},
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

PyObject* THPNpuMultiHeadAttentionV2Backward0_alibi_mask_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionV2Backward0*>(self->cdata.get())->alibi_mask_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionV2Backward0_alibi_mask_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionV2Backward0*>(self->cdata.get())->alibi_mask_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionV2Backward0_atten_mask_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionV2Backward0*>(self->cdata.get())->atten_mask_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionV2Backward0_atten_mask_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionV2Backward0*>(self->cdata.get())->atten_mask_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionV2Backward0_gen_mask_parallel_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuMultiHeadAttentionV2Backward0*>(self->cdata.get())->gen_mask_parallel;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionV2Backward0_head_num_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuMultiHeadAttentionV2Backward0*>(self->cdata.get())->head_num;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionV2Backward0_input_layout_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuMultiHeadAttentionV2Backward0*>(self->cdata.get())->input_layout;
  return PyUnicode_FromStringAndSize(prop.data(), prop.size());
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionV2Backward0_keep_prob_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuMultiHeadAttentionV2Backward0*>(self->cdata.get())->keep_prob;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionV2Backward0_key_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionV2Backward0*>(self->cdata.get())->key_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionV2Backward0_key_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionV2Backward0*>(self->cdata.get())->key_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionV2Backward0_next_tokens_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuMultiHeadAttentionV2Backward0*>(self->cdata.get())->next_tokens;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionV2Backward0_pre_tokens_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuMultiHeadAttentionV2Backward0*>(self->cdata.get())->pre_tokens;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionV2Backward0_query_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionV2Backward0*>(self->cdata.get())->query_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionV2Backward0_query_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionV2Backward0*>(self->cdata.get())->query_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionV2Backward0_scale_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuMultiHeadAttentionV2Backward0*>(self->cdata.get())->scale;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionV2Backward0_sync_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuMultiHeadAttentionV2Backward0*>(self->cdata.get())->sync;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionV2Backward0_value_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionV2Backward0*>(self->cdata.get())->value_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionV2Backward0_value_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionV2Backward0*>(self->cdata.get())->value_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionV2Backward0_result0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionV2Backward0*>(self->cdata.get())->result0_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionV2Backward0_result0_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionV2Backward0*>(self->cdata.get())->result0_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionV2Backward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionV2Backward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionV2Backward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuMultiHeadAttentionV2Backward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionV2Backward0_result2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuMultiHeadAttentionV2Backward0*>(self->cdata.get())->result2;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionV2Backward0_result3_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuMultiHeadAttentionV2Backward0*>(self->cdata.get())->result3;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuMultiHeadAttentionV2Backward0_result4_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuMultiHeadAttentionV2Backward0*>(self->cdata.get())->result4;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuMultiHeadAttentionV2Backward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_alibi_mask", (getter)THPNpuMultiHeadAttentionV2Backward0_alibi_mask_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_alibi_mask", (getter)THPNpuMultiHeadAttentionV2Backward0_alibi_mask_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_atten_mask", (getter)THPNpuMultiHeadAttentionV2Backward0_atten_mask_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_atten_mask", (getter)THPNpuMultiHeadAttentionV2Backward0_atten_mask_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_gen_mask_parallel", (getter)THPNpuMultiHeadAttentionV2Backward0_gen_mask_parallel_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_head_num", (getter)THPNpuMultiHeadAttentionV2Backward0_head_num_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input_layout", (getter)THPNpuMultiHeadAttentionV2Backward0_input_layout_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keep_prob", (getter)THPNpuMultiHeadAttentionV2Backward0_keep_prob_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_key", (getter)THPNpuMultiHeadAttentionV2Backward0_key_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_key", (getter)THPNpuMultiHeadAttentionV2Backward0_key_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_next_tokens", (getter)THPNpuMultiHeadAttentionV2Backward0_next_tokens_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_pre_tokens", (getter)THPNpuMultiHeadAttentionV2Backward0_pre_tokens_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_query", (getter)THPNpuMultiHeadAttentionV2Backward0_query_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_query", (getter)THPNpuMultiHeadAttentionV2Backward0_query_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale", (getter)THPNpuMultiHeadAttentionV2Backward0_scale_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_sync", (getter)THPNpuMultiHeadAttentionV2Backward0_sync_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_value", (getter)THPNpuMultiHeadAttentionV2Backward0_value_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_value", (getter)THPNpuMultiHeadAttentionV2Backward0_value_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result0", (getter)THPNpuMultiHeadAttentionV2Backward0_result0_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result0", (getter)THPNpuMultiHeadAttentionV2Backward0_result0_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPNpuMultiHeadAttentionV2Backward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPNpuMultiHeadAttentionV2Backward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result2", (getter)THPNpuMultiHeadAttentionV2Backward0_result2_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result3", (getter)THPNpuMultiHeadAttentionV2Backward0_result3_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result4", (getter)THPNpuMultiHeadAttentionV2Backward0_result4_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNpuPsRoiPoolingBackward0_group_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuPsRoiPoolingBackward0*>(self->cdata.get())->group_size;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuPsRoiPoolingBackward0_output_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuPsRoiPoolingBackward0*>(self->cdata.get())->output_dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuPsRoiPoolingBackward0_rois_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuPsRoiPoolingBackward0*>(self->cdata.get())->rois_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuPsRoiPoolingBackward0_rois_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuPsRoiPoolingBackward0*>(self->cdata.get())->rois_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuPsRoiPoolingBackward0_self_sym_argsize_2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuPsRoiPoolingBackward0*>(self->cdata.get())->self_sym_argsize_2;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuPsRoiPoolingBackward0_self_sym_argsize_3_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuPsRoiPoolingBackward0*>(self->cdata.get())->self_sym_argsize_3;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuPsRoiPoolingBackward0_spatial_scale_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuPsRoiPoolingBackward0*>(self->cdata.get())->spatial_scale;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuPsRoiPoolingBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_group_size", (getter)THPNpuPsRoiPoolingBackward0_group_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_dim", (getter)THPNpuPsRoiPoolingBackward0_output_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_rois", (getter)THPNpuPsRoiPoolingBackward0_rois_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_rois", (getter)THPNpuPsRoiPoolingBackward0_rois_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_argsize_2", (getter)THPNpuPsRoiPoolingBackward0_self_sym_argsize_2_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_argsize_3", (getter)THPNpuPsRoiPoolingBackward0_self_sym_argsize_3_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_spatial_scale", (getter)THPNpuPsRoiPoolingBackward0_spatial_scale_getter, nullptr, nullptr, nullptr},
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

PyObject* THPNpuRotaryMulBackward0_r1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuRotaryMulBackward0*>(self->cdata.get())->r1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuRotaryMulBackward0_r1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuRotaryMulBackward0*>(self->cdata.get())->r1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuRotaryMulBackward0_r2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuRotaryMulBackward0*>(self->cdata.get())->r2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuRotaryMulBackward0_r2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuRotaryMulBackward0*>(self->cdata.get())->r2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuRotaryMulBackward0_rotary_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuRotaryMulBackward0*>(self->cdata.get())->rotary_mode;
  return PyUnicode_FromStringAndSize(prop.data(), prop.size());
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuRotaryMulBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuRotaryMulBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuRotaryMulBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuRotaryMulBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuRotaryMulBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_r1", (getter)THPNpuRotaryMulBackward0_r1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_r1", (getter)THPNpuRotaryMulBackward0_r1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_r2", (getter)THPNpuRotaryMulBackward0_r2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_r2", (getter)THPNpuRotaryMulBackward0_r2_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_rotary_mode", (getter)THPNpuRotaryMulBackward0_rotary_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPNpuRotaryMulBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPNpuRotaryMulBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNpuScaledMaskedSoftmaxBackward0_fixed_triu_mask_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuScaledMaskedSoftmaxBackward0*>(self->cdata.get())->fixed_triu_mask;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuScaledMaskedSoftmaxBackward0_mask_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuScaledMaskedSoftmaxBackward0*>(self->cdata.get())->mask_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuScaledMaskedSoftmaxBackward0_mask_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuScaledMaskedSoftmaxBackward0*>(self->cdata.get())->mask_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuScaledMaskedSoftmaxBackward0_scale_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuScaledMaskedSoftmaxBackward0*>(self->cdata.get())->scale;
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

PyObject* THPNpuScaledMaskedSoftmaxBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuScaledMaskedSoftmaxBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuScaledMaskedSoftmaxBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuScaledMaskedSoftmaxBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuScaledMaskedSoftmaxBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_fixed_triu_mask", (getter)THPNpuScaledMaskedSoftmaxBackward0_fixed_triu_mask_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mask", (getter)THPNpuScaledMaskedSoftmaxBackward0_mask_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_mask", (getter)THPNpuScaledMaskedSoftmaxBackward0_mask_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale", (getter)THPNpuScaledMaskedSoftmaxBackward0_scale_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPNpuScaledMaskedSoftmaxBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPNpuScaledMaskedSoftmaxBackward0_result_raw_getter, nullptr, nullptr, nullptr},
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

PyObject* THPNpuSoftmaxCrossEntropyWithLogitsBackward0_labels_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuSoftmaxCrossEntropyWithLogitsBackward0*>(self->cdata.get())->labels_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuSoftmaxCrossEntropyWithLogitsBackward0_labels_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuSoftmaxCrossEntropyWithLogitsBackward0*>(self->cdata.get())->labels_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuSoftmaxCrossEntropyWithLogitsBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuSoftmaxCrossEntropyWithLogitsBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuSoftmaxCrossEntropyWithLogitsBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuSoftmaxCrossEntropyWithLogitsBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuSoftmaxCrossEntropyWithLogitsBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_labels", (getter)THPNpuSoftmaxCrossEntropyWithLogitsBackward0_labels_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_labels", (getter)THPNpuSoftmaxCrossEntropyWithLogitsBackward0_labels_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPNpuSoftmaxCrossEntropyWithLogitsBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPNpuSoftmaxCrossEntropyWithLogitsBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNpuSwigluBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuSwigluBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuSwigluBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuSwigluBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuSwigluBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuSwigluBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuSwigluBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPNpuSwigluBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPNpuSwigluBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPNpuSwigluBackward0_self_raw_getter, nullptr, nullptr, nullptr},
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

PyObject* THPStftBackward0_hop_length_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<StftBackward0*>(self->cdata.get())->hop_length;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPStftBackward0_n_fft_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<StftBackward0*>(self->cdata.get())->n_fft;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPStftBackward0_normalized_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<StftBackward0*>(self->cdata.get())->normalized;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPStftBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<StftBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPStftBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<StftBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPStftBackward0_win_length_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<StftBackward0*>(self->cdata.get())->win_length;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPStftBackward0_window_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<StftBackward0*>(self->cdata.get())->window_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPStftBackward0_window_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<StftBackward0*>(self->cdata.get())->window_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef StftBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_hop_length", (getter)THPStftBackward0_hop_length_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_n_fft", (getter)THPStftBackward0_n_fft_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_normalized", (getter)THPStftBackward0_normalized_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPStftBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPStftBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_win_length", (getter)THPStftBackward0_win_length_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_window", (getter)THPStftBackward0_window_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_window", (getter)THPStftBackward0_window_raw_getter, nullptr, nullptr, nullptr},
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

PyObject* THPFftC2RBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FftC2RBackward0*>(self->cdata.get())->dim;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPFftC2RBackward0_normalization_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FftC2RBackward0*>(self->cdata.get())->normalization;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FftC2RBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPFftC2RBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_normalization", (getter)THPFftC2RBackward0_normalization_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNpuGroupNormSwishBackward0_bias_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGroupNormSwishBackward0*>(self->cdata.get())->bias_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGroupNormSwishBackward0_bias_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGroupNormSwishBackward0*>(self->cdata.get())->bias_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGroupNormSwishBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGroupNormSwishBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGroupNormSwishBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGroupNormSwishBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGroupNormSwishBackward0_num_groups_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuGroupNormSwishBackward0*>(self->cdata.get())->num_groups;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGroupNormSwishBackward0_swish_scale_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<NpuGroupNormSwishBackward0*>(self->cdata.get())->swish_scale;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGroupNormSwishBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGroupNormSwishBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGroupNormSwishBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGroupNormSwishBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGroupNormSwishBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGroupNormSwishBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGroupNormSwishBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGroupNormSwishBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGroupNormSwishBackward0_result2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGroupNormSwishBackward0*>(self->cdata.get())->result2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuGroupNormSwishBackward0_result2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuGroupNormSwishBackward0*>(self->cdata.get())->result2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuGroupNormSwishBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_bias", (getter)THPNpuGroupNormSwishBackward0_bias_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_bias", (getter)THPNpuGroupNormSwishBackward0_bias_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPNpuGroupNormSwishBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPNpuGroupNormSwishBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_num_groups", (getter)THPNpuGroupNormSwishBackward0_num_groups_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_swish_scale", (getter)THPNpuGroupNormSwishBackward0_swish_scale_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPNpuGroupNormSwishBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPNpuGroupNormSwishBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPNpuGroupNormSwishBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPNpuGroupNormSwishBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result2", (getter)THPNpuGroupNormSwishBackward0_result2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result2", (getter)THPNpuGroupNormSwishBackward0_result2_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNpuCrossEntropyLossBackward0_ignore_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuCrossEntropyLossBackward0*>(self->cdata.get())->ignore_index;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuCrossEntropyLossBackward0_label_smoothing_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuCrossEntropyLossBackward0*>(self->cdata.get())->label_smoothing;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuCrossEntropyLossBackward0_lse_square_scale_for_zloss_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuCrossEntropyLossBackward0*>(self->cdata.get())->lse_square_scale_for_zloss;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuCrossEntropyLossBackward0_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuCrossEntropyLossBackward0*>(self->cdata.get())->reduction;
  return PyUnicode_FromStringAndSize(prop.data(), prop.size());
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuCrossEntropyLossBackward0_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuCrossEntropyLossBackward0*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuCrossEntropyLossBackward0_target_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuCrossEntropyLossBackward0*>(self->cdata.get())->target_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuCrossEntropyLossBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuCrossEntropyLossBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuCrossEntropyLossBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuCrossEntropyLossBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuCrossEntropyLossBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuCrossEntropyLossBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuCrossEntropyLossBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuCrossEntropyLossBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuCrossEntropyLossBackward0_result3_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuCrossEntropyLossBackward0*>(self->cdata.get())->result3_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuCrossEntropyLossBackward0_result3_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuCrossEntropyLossBackward0*>(self->cdata.get())->result3_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuCrossEntropyLossBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_ignore_index", (getter)THPNpuCrossEntropyLossBackward0_ignore_index_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_label_smoothing", (getter)THPNpuCrossEntropyLossBackward0_label_smoothing_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_lse_square_scale_for_zloss", (getter)THPNpuCrossEntropyLossBackward0_lse_square_scale_for_zloss_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPNpuCrossEntropyLossBackward0_reduction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPNpuCrossEntropyLossBackward0_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_target", (getter)THPNpuCrossEntropyLossBackward0_target_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPNpuCrossEntropyLossBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPNpuCrossEntropyLossBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPNpuCrossEntropyLossBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPNpuCrossEntropyLossBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result3", (getter)THPNpuCrossEntropyLossBackward0_result3_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result3", (getter)THPNpuCrossEntropyLossBackward0_result3_raw_getter, nullptr, nullptr, nullptr},
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

PyObject* THPNpuNsaSelectAttentionBackward0_actual_seq_kvlen_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<NpuNsaSelectAttentionBackward0*>(self->cdata.get())->actual_seq_kvlen;
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

PyObject* THPNpuNsaSelectAttentionBackward0_actual_seq_qlen_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<NpuNsaSelectAttentionBackward0*>(self->cdata.get())->actual_seq_qlen;
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

PyObject* THPNpuNsaSelectAttentionBackward0_atten_mask_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuNsaSelectAttentionBackward0*>(self->cdata.get())->atten_mask_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaSelectAttentionBackward0_atten_mask_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuNsaSelectAttentionBackward0*>(self->cdata.get())->atten_mask_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaSelectAttentionBackward0_head_num_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuNsaSelectAttentionBackward0*>(self->cdata.get())->head_num;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaSelectAttentionBackward0_key_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuNsaSelectAttentionBackward0*>(self->cdata.get())->key_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaSelectAttentionBackward0_key_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuNsaSelectAttentionBackward0*>(self->cdata.get())->key_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaSelectAttentionBackward0_query_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuNsaSelectAttentionBackward0*>(self->cdata.get())->query_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaSelectAttentionBackward0_query_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuNsaSelectAttentionBackward0*>(self->cdata.get())->query_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaSelectAttentionBackward0_scale_value_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuNsaSelectAttentionBackward0*>(self->cdata.get())->scale_value;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaSelectAttentionBackward0_select_block_count_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuNsaSelectAttentionBackward0*>(self->cdata.get())->select_block_count;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaSelectAttentionBackward0_select_block_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuNsaSelectAttentionBackward0*>(self->cdata.get())->select_block_size;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaSelectAttentionBackward0_topk_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuNsaSelectAttentionBackward0*>(self->cdata.get())->topk_indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaSelectAttentionBackward0_topk_indices_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuNsaSelectAttentionBackward0*>(self->cdata.get())->topk_indices_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaSelectAttentionBackward0_value_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuNsaSelectAttentionBackward0*>(self->cdata.get())->value_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaSelectAttentionBackward0_value_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuNsaSelectAttentionBackward0*>(self->cdata.get())->value_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaSelectAttentionBackward0_result0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuNsaSelectAttentionBackward0*>(self->cdata.get())->result0_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaSelectAttentionBackward0_result0_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuNsaSelectAttentionBackward0*>(self->cdata.get())->result0_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaSelectAttentionBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuNsaSelectAttentionBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaSelectAttentionBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuNsaSelectAttentionBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaSelectAttentionBackward0_result2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuNsaSelectAttentionBackward0*>(self->cdata.get())->result2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaSelectAttentionBackward0_result2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuNsaSelectAttentionBackward0*>(self->cdata.get())->result2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuNsaSelectAttentionBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_actual_seq_kvlen", (getter)THPNpuNsaSelectAttentionBackward0_actual_seq_kvlen_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_actual_seq_qlen", (getter)THPNpuNsaSelectAttentionBackward0_actual_seq_qlen_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_atten_mask", (getter)THPNpuNsaSelectAttentionBackward0_atten_mask_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_atten_mask", (getter)THPNpuNsaSelectAttentionBackward0_atten_mask_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_head_num", (getter)THPNpuNsaSelectAttentionBackward0_head_num_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_key", (getter)THPNpuNsaSelectAttentionBackward0_key_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_key", (getter)THPNpuNsaSelectAttentionBackward0_key_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_query", (getter)THPNpuNsaSelectAttentionBackward0_query_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_query", (getter)THPNpuNsaSelectAttentionBackward0_query_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale_value", (getter)THPNpuNsaSelectAttentionBackward0_scale_value_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_select_block_count", (getter)THPNpuNsaSelectAttentionBackward0_select_block_count_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_select_block_size", (getter)THPNpuNsaSelectAttentionBackward0_select_block_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_topk_indices", (getter)THPNpuNsaSelectAttentionBackward0_topk_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_topk_indices", (getter)THPNpuNsaSelectAttentionBackward0_topk_indices_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_value", (getter)THPNpuNsaSelectAttentionBackward0_value_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_value", (getter)THPNpuNsaSelectAttentionBackward0_value_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result0", (getter)THPNpuNsaSelectAttentionBackward0_result0_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result0", (getter)THPNpuNsaSelectAttentionBackward0_result0_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPNpuNsaSelectAttentionBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPNpuNsaSelectAttentionBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result2", (getter)THPNpuNsaSelectAttentionBackward0_result2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result2", (getter)THPNpuNsaSelectAttentionBackward0_result2_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNpuNsaCompressAttentionBackward0_actual_cmp_seq_kvlen_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<NpuNsaCompressAttentionBackward0*>(self->cdata.get())->actual_cmp_seq_kvlen;
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

PyObject* THPNpuNsaCompressAttentionBackward0_actual_seq_qlen_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<NpuNsaCompressAttentionBackward0*>(self->cdata.get())->actual_seq_qlen;
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

PyObject* THPNpuNsaCompressAttentionBackward0_atten_mask_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuNsaCompressAttentionBackward0*>(self->cdata.get())->atten_mask_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaCompressAttentionBackward0_atten_mask_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuNsaCompressAttentionBackward0*>(self->cdata.get())->atten_mask_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaCompressAttentionBackward0_head_num_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuNsaCompressAttentionBackward0*>(self->cdata.get())->head_num;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaCompressAttentionBackward0_key_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuNsaCompressAttentionBackward0*>(self->cdata.get())->key_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaCompressAttentionBackward0_key_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuNsaCompressAttentionBackward0*>(self->cdata.get())->key_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaCompressAttentionBackward0_query_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuNsaCompressAttentionBackward0*>(self->cdata.get())->query_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaCompressAttentionBackward0_query_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuNsaCompressAttentionBackward0*>(self->cdata.get())->query_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaCompressAttentionBackward0_scale_value_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NpuNsaCompressAttentionBackward0*>(self->cdata.get())->scale_value;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaCompressAttentionBackward0_value_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuNsaCompressAttentionBackward0*>(self->cdata.get())->value_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaCompressAttentionBackward0_value_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuNsaCompressAttentionBackward0*>(self->cdata.get())->value_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaCompressAttentionBackward0_result0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuNsaCompressAttentionBackward0*>(self->cdata.get())->result0_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaCompressAttentionBackward0_result0_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuNsaCompressAttentionBackward0*>(self->cdata.get())->result0_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaCompressAttentionBackward0_result2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuNsaCompressAttentionBackward0*>(self->cdata.get())->result2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaCompressAttentionBackward0_result2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuNsaCompressAttentionBackward0*>(self->cdata.get())->result2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaCompressAttentionBackward0_result3_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuNsaCompressAttentionBackward0*>(self->cdata.get())->result3_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNpuNsaCompressAttentionBackward0_result3_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NpuNsaCompressAttentionBackward0*>(self->cdata.get())->result3_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NpuNsaCompressAttentionBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_actual_cmp_seq_kvlen", (getter)THPNpuNsaCompressAttentionBackward0_actual_cmp_seq_kvlen_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_actual_seq_qlen", (getter)THPNpuNsaCompressAttentionBackward0_actual_seq_qlen_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_atten_mask", (getter)THPNpuNsaCompressAttentionBackward0_atten_mask_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_atten_mask", (getter)THPNpuNsaCompressAttentionBackward0_atten_mask_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_head_num", (getter)THPNpuNsaCompressAttentionBackward0_head_num_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_key", (getter)THPNpuNsaCompressAttentionBackward0_key_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_key", (getter)THPNpuNsaCompressAttentionBackward0_key_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_query", (getter)THPNpuNsaCompressAttentionBackward0_query_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_query", (getter)THPNpuNsaCompressAttentionBackward0_query_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale_value", (getter)THPNpuNsaCompressAttentionBackward0_scale_value_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_value", (getter)THPNpuNsaCompressAttentionBackward0_value_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_value", (getter)THPNpuNsaCompressAttentionBackward0_value_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result0", (getter)THPNpuNsaCompressAttentionBackward0_result0_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result0", (getter)THPNpuNsaCompressAttentionBackward0_result0_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result2", (getter)THPNpuNsaCompressAttentionBackward0_result2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result2", (getter)THPNpuNsaCompressAttentionBackward0_result2_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result3", (getter)THPNpuNsaCompressAttentionBackward0_result3_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result3", (getter)THPNpuNsaCompressAttentionBackward0_result3_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

void initialize_autogenerated_functionsEverything(PyObject *module)
{
    static PyTypeObject GatherBackward0Class;
    addClass<GatherBackward0>(module, GatherBackward0Class, "GatherBackward0", GatherBackward0_properties);
    static PyTypeObject DropoutWithByteMaskBackward0Class;
    addClass<DropoutWithByteMaskBackward0>(module, DropoutWithByteMaskBackward0Class, "DropoutWithByteMaskBackward0", DropoutWithByteMaskBackward0_properties);
    static PyTypeObject NpuCiouBackward0Class;
    addClass<NpuCiouBackward0>(module, NpuCiouBackward0Class, "NpuCiouBackward0", NpuCiouBackward0_properties);
    static PyTypeObject NpuDropoutBackward0Class;
    addClass<NpuDropoutBackward0>(module, NpuDropoutBackward0Class, "NpuDropoutBackward0", NpuDropoutBackward0_properties);
    static PyTypeObject NpuFormatCastBackward0Class;
    addClass<NpuFormatCastBackward0>(module, NpuFormatCastBackward0Class, "NpuFormatCastBackward0", NpuFormatCastBackward0_properties);
    static PyTypeObject BinaryCrossEntropyWithLogitsBackward0Class;
    addClass<BinaryCrossEntropyWithLogitsBackward0>(module, BinaryCrossEntropyWithLogitsBackward0Class, "BinaryCrossEntropyWithLogitsBackward0", BinaryCrossEntropyWithLogitsBackward0_properties);
    static PyTypeObject FastGeluBackward0Class;
    addClass<FastGeluBackward0>(module, FastGeluBackward0Class, "FastGeluBackward0", FastGeluBackward0_properties);
    static PyTypeObject KlDivBackward0Class;
    addClass<KlDivBackward0>(module, KlDivBackward0Class, "KlDivBackward0", KlDivBackward0_properties);
    static PyTypeObject L1LossBackward0Class;
    addClass<L1LossBackward0>(module, L1LossBackward0Class, "L1LossBackward0", L1LossBackward0_properties);
    static PyTypeObject MatmulBackwardBackward0Class;
    addClass<MatmulBackwardBackward0>(module, MatmulBackwardBackward0Class, "MatmulBackwardBackward0", MatmulBackwardBackward0_properties);
    static PyTypeObject NpuAddLayerNormBackward0Class;
    addClass<NpuAddLayerNormBackward0>(module, NpuAddLayerNormBackward0Class, "NpuAddLayerNormBackward0", NpuAddLayerNormBackward0_properties);
    static PyTypeObject NpuGeluBackward0Class;
    addClass<NpuGeluBackward0>(module, NpuGeluBackward0Class, "NpuGeluBackward0", NpuGeluBackward0_properties);
    static PyTypeObject NpuBmmv2Backward0Class;
    addClass<NpuBmmv2Backward0>(module, NpuBmmv2Backward0Class, "NpuBmmv2Backward0", NpuBmmv2Backward0_properties);
    static PyTypeObject NpuConfusionTransposeBackward0Class;
    addClass<NpuConfusionTransposeBackward0>(module, NpuConfusionTransposeBackward0Class, "NpuConfusionTransposeBackward0", NpuConfusionTransposeBackward0_properties);
    static PyTypeObject NpuConvolutionBackward0Class;
    addClass<NpuConvolutionBackward0>(module, NpuConvolutionBackward0Class, "NpuConvolutionBackward0", NpuConvolutionBackward0_properties);
    static PyTypeObject NpuConvolutionTransposeBackward0Class;
    addClass<NpuConvolutionTransposeBackward0>(module, NpuConvolutionTransposeBackward0Class, "NpuConvolutionTransposeBackward0", NpuConvolutionTransposeBackward0_properties);
    static PyTypeObject NpuDeepNormBackward0Class;
    addClass<NpuDeepNormBackward0>(module, NpuDeepNormBackward0Class, "NpuDeepNormBackward0", NpuDeepNormBackward0_properties);
    static PyTypeObject NpuDeformableConv2DBackward0Class;
    addClass<NpuDeformableConv2DBackward0>(module, NpuDeformableConv2DBackward0Class, "NpuDeformableConv2DBackward0", NpuDeformableConv2DBackward0_properties);
    static PyTypeObject NpuDiouBackward0Class;
    addClass<NpuDiouBackward0>(module, NpuDiouBackward0Class, "NpuDiouBackward0", NpuDiouBackward0_properties);
    static PyTypeObject NpuDropoutDoMaskBackward0Class;
    addClass<NpuDropoutDoMaskBackward0>(module, NpuDropoutDoMaskBackward0Class, "NpuDropoutDoMaskBackward0", NpuDropoutDoMaskBackward0_properties);
    static PyTypeObject NpuDropoutWithAddSoftmaxBackward0Class;
    addClass<NpuDropoutWithAddSoftmaxBackward0>(module, NpuDropoutWithAddSoftmaxBackward0Class, "NpuDropoutWithAddSoftmaxBackward0", NpuDropoutWithAddSoftmaxBackward0_properties);
    static PyTypeObject NpuDtypeCastBackward0Class;
    addClass<NpuDtypeCastBackward0>(module, NpuDtypeCastBackward0Class, "NpuDtypeCastBackward0", NpuDtypeCastBackward0_properties);
    static PyTypeObject NpuFusedAttentionScoreFwdBackward0Class;
    addClass<NpuFusedAttentionScoreFwdBackward0>(module, NpuFusedAttentionScoreFwdBackward0Class, "NpuFusedAttentionScoreFwdBackward0", NpuFusedAttentionScoreFwdBackward0_properties);
    static PyTypeObject NpuFusionAttentionBackward0Class;
    addClass<NpuFusionAttentionBackward0>(module, NpuFusionAttentionBackward0Class, "NpuFusionAttentionBackward0", NpuFusionAttentionBackward0_properties);
    static PyTypeObject NpuFusionAttentionV2Backward0Class;
    addClass<NpuFusionAttentionV2Backward0>(module, NpuFusionAttentionV2Backward0Class, "NpuFusionAttentionV2Backward0", NpuFusionAttentionV2Backward0_properties);
    static PyTypeObject NpuGegluBackward0Class;
    addClass<NpuGegluBackward0>(module, NpuGegluBackward0Class, "NpuGegluBackward0", NpuGegluBackward0_properties);
    static PyTypeObject NpuGiouBackward0Class;
    addClass<NpuGiouBackward0>(module, NpuGiouBackward0Class, "NpuGiouBackward0", NpuGiouBackward0_properties);
    static PyTypeObject NpuGruBackward0Class;
    addClass<NpuGruBackward0>(module, NpuGruBackward0Class, "NpuGruBackward0", NpuGruBackward0_properties);
    static PyTypeObject NpuLinearBackward0Class;
    addClass<NpuLinearBackward0>(module, NpuLinearBackward0Class, "NpuLinearBackward0", NpuLinearBackward0_properties);
    static PyTypeObject NpuLstmBackward0Class;
    addClass<NpuLstmBackward0>(module, NpuLstmBackward0Class, "NpuLstmBackward0", NpuLstmBackward0_properties);
    static PyTypeObject NpuLstmCellBackward0Class;
    addClass<NpuLstmCellBackward0>(module, NpuLstmCellBackward0Class, "NpuLstmCellBackward0", NpuLstmCellBackward0_properties);
    static PyTypeObject NpuLstmDataBackward0Class;
    addClass<NpuLstmDataBackward0>(module, NpuLstmDataBackward0Class, "NpuLstmDataBackward0", NpuLstmDataBackward0_properties);
    static PyTypeObject NpuMaxBackward0Class;
    addClass<NpuMaxBackward0>(module, NpuMaxBackward0Class, "NpuMaxBackward0", NpuMaxBackward0_properties);
    static PyTypeObject NpuMinBackward0Class;
    addClass<NpuMinBackward0>(module, NpuMinBackward0Class, "NpuMinBackward0", NpuMinBackward0_properties);
    static PyTypeObject NpuMishBackward0Class;
    addClass<NpuMishBackward0>(module, NpuMishBackward0Class, "NpuMishBackward0", NpuMishBackward0_properties);
    static PyTypeObject NpuMultiHeadAttentionBackward0Class;
    addClass<NpuMultiHeadAttentionBackward0>(module, NpuMultiHeadAttentionBackward0Class, "NpuMultiHeadAttentionBackward0", NpuMultiHeadAttentionBackward0_properties);
    static PyTypeObject NpuMultiHeadAttentionV2Backward0Class;
    addClass<NpuMultiHeadAttentionV2Backward0>(module, NpuMultiHeadAttentionV2Backward0Class, "NpuMultiHeadAttentionV2Backward0", NpuMultiHeadAttentionV2Backward0_properties);
    static PyTypeObject NpuPsRoiPoolingBackward0Class;
    addClass<NpuPsRoiPoolingBackward0>(module, NpuPsRoiPoolingBackward0Class, "NpuPsRoiPoolingBackward0", NpuPsRoiPoolingBackward0_properties);
    static PyTypeObject NpuRmsNormBackward0Class;
    addClass<NpuRmsNormBackward0>(module, NpuRmsNormBackward0Class, "NpuRmsNormBackward0", NpuRmsNormBackward0_properties);
    static PyTypeObject NpuRotaryMulBackward0Class;
    addClass<NpuRotaryMulBackward0>(module, NpuRotaryMulBackward0Class, "NpuRotaryMulBackward0", NpuRotaryMulBackward0_properties);
    static PyTypeObject NpuScaledMaskedSoftmaxBackward0Class;
    addClass<NpuScaledMaskedSoftmaxBackward0>(module, NpuScaledMaskedSoftmaxBackward0Class, "NpuScaledMaskedSoftmaxBackward0", NpuScaledMaskedSoftmaxBackward0_properties);
    static PyTypeObject NpuSiluBackward0Class;
    addClass<NpuSiluBackward0>(module, NpuSiluBackward0Class, "NpuSiluBackward0", NpuSiluBackward0_properties);
    static PyTypeObject NpuSoftmaxCrossEntropyWithLogitsBackward0Class;
    addClass<NpuSoftmaxCrossEntropyWithLogitsBackward0>(module, NpuSoftmaxCrossEntropyWithLogitsBackward0Class, "NpuSoftmaxCrossEntropyWithLogitsBackward0", NpuSoftmaxCrossEntropyWithLogitsBackward0_properties);
    static PyTypeObject NpuSwigluBackward0Class;
    addClass<NpuSwigluBackward0>(module, NpuSwigluBackward0Class, "NpuSwigluBackward0", NpuSwigluBackward0_properties);
    static PyTypeObject RepeatInterleaveBackward0Class;
    addClass<RepeatInterleaveBackward0>(module, RepeatInterleaveBackward0Class, "RepeatInterleaveBackward0", RepeatInterleaveBackward0_properties);
    static PyTypeObject RepeatInterleaveBackward1Class;
    addClass<RepeatInterleaveBackward1>(module, RepeatInterleaveBackward1Class, "RepeatInterleaveBackward1", RepeatInterleaveBackward1_properties);
    static PyTypeObject StftBackward0Class;
    addClass<StftBackward0>(module, StftBackward0Class, "StftBackward0", StftBackward0_properties);
    static PyTypeObject FftR2CBackward0Class;
    addClass<FftR2CBackward0>(module, FftR2CBackward0Class, "FftR2CBackward0", FftR2CBackward0_properties);
    static PyTypeObject FftC2RBackward0Class;
    addClass<FftC2RBackward0>(module, FftC2RBackward0Class, "FftC2RBackward0", FftC2RBackward0_properties);
    static PyTypeObject NpuGroupNormSwishBackward0Class;
    addClass<NpuGroupNormSwishBackward0>(module, NpuGroupNormSwishBackward0Class, "NpuGroupNormSwishBackward0", NpuGroupNormSwishBackward0_properties);
    static PyTypeObject NpuCrossEntropyLossBackward0Class;
    addClass<NpuCrossEntropyLossBackward0>(module, NpuCrossEntropyLossBackward0Class, "NpuCrossEntropyLossBackward0", NpuCrossEntropyLossBackward0_properties);
    static PyTypeObject NpuNsaCompressBackward0Class;
    addClass<NpuNsaCompressBackward0>(module, NpuNsaCompressBackward0Class, "NpuNsaCompressBackward0", NpuNsaCompressBackward0_properties);
    static PyTypeObject NpuNsaSelectAttentionBackward0Class;
    addClass<NpuNsaSelectAttentionBackward0>(module, NpuNsaSelectAttentionBackward0Class, "NpuNsaSelectAttentionBackward0", NpuNsaSelectAttentionBackward0_properties);
    static PyTypeObject NpuNsaCompressAttentionBackward0Class;
    addClass<NpuNsaCompressAttentionBackward0>(module, NpuNsaCompressAttentionBackward0Class, "NpuNsaCompressAttentionBackward0", NpuNsaCompressAttentionBackward0_properties);
}
}
}
} // namespace at_npu::autograd::generated

#endif
