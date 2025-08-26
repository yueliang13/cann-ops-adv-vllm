# MoeFinalizeRouting

## 支持的产品型号

- Atlas A2 训练系列产品/Atlas 800I A2 推理产品
- Atlas A3 训练系列产品/Atlas 800I A3 推理产品

## 接口原型

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnMoeFinalizeRoutingGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeFinalizeRouting”接口执行计算。

* `aclnnStatus aclnnMoeFinalizeRoutingGetWorkspaceSize(const aclTensor* expandedX, const aclTensor* x1, const aclTensor* x2Optional, const aclTensor* bias, const aclTensor* scales, const aclTensor* expandedRowIdx, const aclTensor* expandedExpertIdx, const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
* `aclnnStatus aclnnMoeFinalizeRouting(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)`

## 功能说明

-   **算子功能**：MoE计算中，最后处理合并MoE FFN的输出结果。
-   **计算公式**：

    $$
    expertid=expandedExpertIdx[i,k]
    $$
    
    $$
    out(i,j)=x1_{i,j}+x2Optional_{i,j}+\sum_{k=0}^{K}(scales_{i,k}*(expandedX_{expandedRowIdx_{i+k*num_rows},j}+bias_{expertid,j}))
    $$

## aclnnMoeFinalizeRoutingGetWorkspaceSize

-   **参数说明：**
    -   expandedX （aclTensor\*，计算输入）：Device侧的aclTensor，公式中的expandedX ，MoE的FFN输出，要求是一个2D的Tensor，数据类型支持FLOAT16、BFLOAT16、FLOAT32，[数据格式](common/数据格式.md)要求为ND。限制：其shape支持（NUM\_ROWS \* K, H），NUM\_ROWS为行数，K为从总的专家E中选出K个专家，H为列数。
    -   x1（aclTensor\*，计算输入）：Device侧的aclTensor，公式中的x1，要求是一个2D的Tensor，数据类型要求与expandedX一致 ，shape要求与out的shape一致。
    -   x2Optional（aclTensor\*，计算输入）：Device侧的aclTensor，公式中的x2Optional，要求是一个2D的Tensor，数据类型要求与expandedX一致 ，shape要求与out的shape一致。
    -   bias（aclTensor\*，计算输入）：Device侧的aclTensor，公式中的bias，要求是一个2D的Tensor，数据类型要求与expandedX一致。限制：其shape支持（E，H），E为总的专家个数，H为列数。
    -   scales（aclTensor\*，计算输入）：Device侧的aclTensor，公式中的scales，要求是一个2D的Tensor，数据类型要求与expandedX一致 ，限制：其shape支持（NUM\_ROWS，K）。
    -   expandedRowIdx（aclTensor\*，计算输入）：Device侧的aclTensor，公式中的expandedRowIdx，要求是一个1D的Tensor，数据类型支持INT32。限制：其shape支持（NUM\_ROWS \* K），Tensor中的值取值范围是[0,NUM\_ROWS \* K-1]。
    -   expandedExpertIdx（aclTensor\*，计算输入）：Device侧的aclTensor，公式中的expandedExpertIdx，要求是一个2D的Tensor，数据类型支持INT32。限制：其shape支持（NUM\_ROWS，K），Tensor中的值取值范围是[0, E-1]，E为总的专家个数。
    -   out（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出，要求是一个2D的Tensor，数据类型与expandedX 需要保持一致。限制：其shape支持（NUM\_ROWS, H）。
    -   workspaceSize（uint64\_t\*，出参）：返回需要在Device侧申请的workspace大小。
    -   executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

    ```
    第一段接口完成入参校验，出现以下场景时报错:
    161001 (ACLNN_ERR_PARAM_NULLPTR): 1. 传入的expandedX、x1、x2Optional、bias、scales、expandedRowIdx和expandedExpertIdx是空指针时。
    161002 (ACLNN_ERR_PARAM_INVALID): 1. expandedX、x1、x2Optional、bias、scales、expandedRowIdx和expandedExpertIdx的数据类型不在支持的范围之内。
                                      2. expandedX、x1、x2Optional、bias、scales、expandedRowIdx和expandedExpertIdx的shape不在支持的范围之内。
    ```

## aclnnMoeFinalizeRouting

-   **参数说明：**
    -   workspace（void\*，入参）：在Device侧申请的workspace内存地址。
    -   workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnMoeFinalizeRoutingGetWorkspaceSize获取。
    -   executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
    -   stream（aclrtStream,入参）：指定执行任务的AscendCL stream流。

-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。


## 约束说明
无


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。
```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_finalize_routing.h"
#include <iostream>
#include <vector>

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}
int Init(int32_t deviceId, aclrtStream* stream) {
    // 固定写法，AscendCL初始化
    auto  ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
  // 1. （固定写法）device/stream初始化, 参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> expandedXShape = {3 * 2, 4};
  std::vector<int64_t> x1Shape = {3, 4};
  std::vector<int64_t> x2OptionalShape = {3, 4};
  std::vector<int64_t> biasShape = {2, 4};
  std::vector<int64_t> scalesShape = {3, 2};
  std::vector<int64_t> expandedExpertIdxShape = {3, 2};
  std::vector<int64_t> expandedRowIdxShape = {3 * 2};
  std::vector<int64_t> outShape = {3, 4};
  void* expandedXAddr = nullptr;
  void* x1Addr = nullptr;
  void* x2OptionalAddr = nullptr;
  void* biasAddr = nullptr;
  void* scalesDeviceAddr = nullptr;
  void* expandedExpertIdxAddr = nullptr;
  void* expandedRowIdxAddr = nullptr;
  void* outDeviceAddr = nullptr;
  
  aclTensor* expandedX = nullptr;
  aclTensor* x1 = nullptr;
  aclTensor* x2Optional = nullptr;
  aclTensor* bias = nullptr;
  aclTensor* scales = nullptr;
  aclTensor* expandedExpertIdx = nullptr;
  aclTensor* expandedRowIdx = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> expandedXHostData = {0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1,
                                                     0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1};
  std::vector<float> x1HostData = {0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2, 11.2};
  std::vector<float> x2OptionalHostData = {0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2, 11.2};
  std::vector<float> biasHostData = {0.2, 0.4, 0.2, 0.4, 0.2, 0.4, 0.2, 0.4};
  std::vector<float> scalesHostData = {1.3, 1.6, 1.2, 1.8, 1.2, 2.3};
  std::vector<int32_t> expandedExpertIdxHostData = {0, 1, 0, 1, 0, 1};
  std::vector<int32_t> expandedRowIdxHostData = {2, 1, 4, 3, 0, 5};
  std::vector<float> outHostData(12, 0.0f);
  // 创建expandedX aclTensor
  ret = CreateAclTensor(expandedXHostData, expandedXShape, &expandedXAddr,
                        aclDataType::ACL_FLOAT, &expandedX);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建x1 aclTensor
  ret = CreateAclTensor(x1HostData, x1Shape, &x1Addr, aclDataType::ACL_FLOAT, &x1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建x2Optional aclScalar
  ret = CreateAclTensor(x2OptionalHostData, x2OptionalShape, &x2OptionalAddr, aclDataType::ACL_FLOAT, &x2Optional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建bias aclTensor
  ret = CreateAclTensor(biasHostData, biasShape, &biasAddr, aclDataType::ACL_FLOAT, &bias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建totalWeightOut aclTensor
  ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  // 创建expandedExpertIdx aclTensor
  ret = CreateAclTensor(expandedExpertIdxHostData, expandedExpertIdxShape, &expandedExpertIdxAddr,
                        aclDataType::ACL_INT32, &expandedExpertIdx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建expandedRowIdx aclTensor
  ret = CreateAclTensor(expandedRowIdxHostData, expandedRowIdxShape, &expandedRowIdxAddr,
                        aclDataType::ACL_INT32, &expandedRowIdx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建Out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3.调用CANN算子库API，需要修改为具体的算子接口
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnMoeFinalizeRouting第一段接口
  ret = aclnnMoeFinalizeRoutingGetWorkspaceSize(expandedX, x1, x2Optional, bias, scales,
                                                expandedRowIdx, expandedExpertIdx, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeFinalizeRoutingGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnMoeFinalizeRouting第二段接口
  ret = aclnnMoeFinalizeRouting(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeFinalizeRouting failed. ERROR: %d\n", ret); return ret);

  // 4.（ 固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0.0f);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    outDeviceAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(expandedX);
  aclDestroyTensor(x1);
  aclDestroyTensor(x2Optional);
  aclDestroyTensor(bias);
  aclDestroyTensor(scales);
  aclDestroyTensor(expandedExpertIdx);
  aclDestroyTensor(expandedRowIdx);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(expandedXAddr);
  aclrtFree(x1Addr);
  aclrtFree(x2OptionalAddr);
  aclrtFree(biasAddr);
  aclrtFree(scalesDeviceAddr);
  aclrtFree(expandedExpertIdxAddr);
  aclrtFree(expandedRowIdxAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```

