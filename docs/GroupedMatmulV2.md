声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# GroupedMatmulV2

## 支持的产品型号
- Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)

## 功能说明

- 算子功能：实现分组矩阵乘计算，每组矩阵乘的维度大小可以不同。基本功能为矩阵乘，如$y_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i], i=1...g$，其中g为分组个数，$m_i/k_i/n_i$为对应shape。相较于[GroupedMatmul](GroupedMatmul.md)接口，**此接口新增**：
    -   支持不同分组轴，由groupType表示。
    -   非量化场景，支持x，weight转置（转置指若shape为[M,K]时，则stride为[1,M],数据排布为[K,M]的场景）。
    -   非量化场景支持x，weight输入都为float32类型。
    -   量化、伪量化场景，支持weight转置，支持weight为单tensor。

- 计算公式：
  - **非量化场景：**

  $$
   y_i=x_i\times weight_i + bias_i
  $$

  - **量化场景：**

  $$
   y_i=(x_i\times weight_i + bias_i) * scale_i + offset_i
  $$

  - **反量化场景：**

  $$
   y_i=(x_i\times weight_i + bias_i) * scale_i
  $$

  - **伪量化场景：**

  $$
   y_i=x_i\times (weight_i + antiquant\_offset_i) * antiquant\_scale_i + bias_i
  $$

## 实现原理

详细实现原理参考[GroupedMatmul设计](./common/GroupedMatmul算子设计介绍.md)。

## 算子执行接口

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnGroupedMatmulV2GetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnGroupedMatmulV2”接口执行计算。

* `aclnnStatus aclnnGroupedMatmulV2GetWorkspaceSize(const aclTensorList* x, const aclTensorList* weight, const aclTensorList* biasOptional, const aclTensorList* scaleOptional, const aclTensorList* offsetOptional, const aclTensorList* antiquantScaleOptional, const aclTensorList* antiquantOffsetOptional, const aclIntArray* groupListOptional, int64_t splitItem, int64_t groupType, const aclTensorList* y, uint64_t* workspaceSize, aclOpExecutor** executor)`
* `aclnnStatus aclnnGroupedMatmulV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnGroupedMatmulV2GetWorkspaceSize

- **参数说明：**
  -   x（aclTensorList\*，计算输入）：必选参数，Device侧的aclTensorList，公式中的输入x，数据类型支持FLOAT16、BFLOAT16、INT8、FLOAT32，[数据格式](common/数据格式.md)支持ND，支持的最大长度为128个。
  -   weight（aclTensorList\*，计算输入）：必选参数，Device侧的aclTensorList，公式中的weight，数据类型支持FLOAT16、BFLOAT16、INT8、FLOAT32，[数据格式](common/数据格式.md)支持ND，支持的最大长度为128个。
  -   biasOptional（aclTensorList\*，计算输入）可选参数，Device侧的aclTensorList，公式中的bias，数据类型支持FLOAT16、FLOAT32、INT32，[数据格式](common/数据格式.md)支持ND，长度与weight相同。
  -   scaleOptional（aclTensorList\*，计算输入）可选参数，Device侧的aclTensorList，代表量化参数中的缩放因子，数据类型支持UINT64，[数据格式](common/数据格式.md)支持ND，长度与weight相同。
  -   offsetOptional（aclTensorList\*，计算输入）可选参数，Device侧的aclTensorList，代表量化参数中的偏移量，数据类型支持FLOAT32，[数据格式](common/数据格式.md)支持ND，长度与weight相同。
  -   antiquantScaleOptional（aclTensorList\*，计算输入）可选参数，Device侧的aclTensorList，代表伪量化参数中的缩放因子，数据类型支持FLOAT16、BFLOAT16，[数据格式](common/数据格式.md)支持ND，长度与weight相同。
  -   antiquantOffsetOptional（aclTensorList\*，计算输入）可选参数，Device侧的aclTensorList，代表伪量化参数中的偏移量，数据类型支持FLOAT16、BFLOAT16，[数据格式](common/数据格式.md)支持ND，长度与weight相同。
  -   groupListOptional（aclIntArray\*，计算输入）：可选参数，Host侧的aclIntArray类型，分组轴方向的matmul索引情况，分组轴由参数groupType表示，数据类型支持INT64，[数据格式](common/数据格式.md)支持ND，长度与weight相同。
  -   splitItem（int64\_t，计算输入）：整数型参数，代表输出是否要做tensor切分，0/1代表输出为多tensor；2/3代表输出为单tensor，默认值为0。
  -   groupType（int64\_t，计算输入）：整数型参数，代表需要分组的轴，如矩阵乘为C[m,n]=A[m,k]xB[k,n]，则groupType取值-1：不分组，0：m轴分组，1：n轴分组，2：k轴分组，默认值为-1，当前不支持n轴分组。
  -   y（aclTensorList\*，计算输出）：Device侧的aclTensorList，公式中的输出y，数据类型支持FLOAT16、BFLOAT16、INT8、FLOAT32，[数据格式](common/数据格式.md)支持ND，支持的最大长度为128个。
  -   workspaceSize（uint64\_t\*，出参）：返回需要在Device侧申请的workspace大小。
  -   executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：
  1.如果传入参数是必选输入、输出或者必选属性，且是空指针。
  2.传入参数weight的元素存在空指针。
  3.传入参数x的元素为空指针，且传出参数y的元素不为空指针。
  4.传入参数x的元素不为空指针，且传出参数y的元素为空指针。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：
  1.x、weight、biasOptional、scaleOptional、offsetOptional、antiquantScaleOptional、antiquantOffsetOptional、groupListOptional、splitItem、groupType、y的数据类型和数据格式不在支持的范围内。
  2.weight的长度大于128；若bias不为空，bias的长度不等于weight的长度。
  3.splitItem为2、3的场景，y长度不等于1；
  4.splitItem为0、1的场景，y长度不等于weight的长度，groupListOptional长度不等于weight的长度。
  ```

### aclnnGroupedMatmulV2

-   **参数说明：**
    -   workspace（void\*，入参）：在Device侧申请的workspace内存地址。
    -   workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnGroupedMatmulV2GetWorkspaceSize获取。
    -   executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
    -   stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明
  - 非量化场景支持的输入类型为：
    - x为FLOAT16、weight为FLOAT16、biasOptional为FLOAT16、scaleOptional为空、offsetOptional为空、antiquantScaleOptional为空、antiquantOffsetOptional为空、y为FLOAT16；
    - x为BFLOAT16、weight为BFLOAT16、biasOptional为FLOAT32、scaleOptional为空、offsetOptional为空、antiquantScaleOptional为空、antiquantOffsetOptional为空、y为BFLOAT16；
    - x为FLOAT32、weight为FLOAT32、biasOptional为FLOAT32、scaleOptional为空、offsetOptional为空、antiquantScaleOptional为空、antiquantOffsetOptional为空、y为FLOAT32；
  - 量化场景支持的输入类型为：

    - x为INT8、weight为INT8、biasOptional为INT32、scaleOptional为UINT64、offsetOptional为空、antiquantScaleOptional为空、antiquantOffsetOptional为空、y为INT8；
  - 伪量化场景支持的输入类型为：
    - x为FLOAT16、weight为INT8、biasOptional为FLOAT16、scaleOptional为空，offsetOptional为空，antiquantScaleOptional为FLOAT16、antiquantOffsetOptional为FLOAT16、y为FLOAT16；
    - x为BFLOAT16、weight为INT8、biasOptional为FLOAT32、scaleOptional为空，offsetOptional为空，antiquantScaleOptional为BFLOAT16、antiquantOffsetOptional为BFLOAT16、y为BFLOAT16；
  - 如果传入groupListOptional，groupListOptional必须为非负递增数列，groupListOptional长度不能为1。
  - 不同groupType支持场景:
    - 量化、伪量化仅支持groupType为-1和0场景。
    - 支持场景中单表示单tensor，多表示多tensor，表示顺序为x、weight、y。例如单多单表示支持x为单tensor、weight多tensor、y单tensor的场景。

    | groupType | 支持场景 | 场景限制 |
    |:---------:|:-------:| :-------|
    | -1 | 多多多 |1）仅支持splitItem为0/1<br>2）x中tensor支持2-6维，weight中tensor需为2维，y中tensor维度和x保持一致<br>3）groupListOptional必须传空 |
    | 0 | 单单单 |1）仅支持splitItem为2/3<br>2）weight中tensor需为3维，x，y中tensor需为2维<br>3）必须传groupListOptional，且最后一个值与x中tensor的第一维相等 |
    | 0 | 单多单 |1）仅支持splitItem为2/3<br>2）必须传groupListOptional，且最后一个值与x中tensor的第一维相等<br>3）x,weight,y中tensor需为2维<br>4）weight中每个tensor的N轴必须相等 |
    | 0 | 单多多 |1）仅支持splitItem为0/1<br>2）必须传groupListOptional，groupListOptional的差值需与y中tensor的第一维一一对应<br>3）x,weight,y中tensor需为2维 |
    | 0 | 多多单 |1）仅支持splitItem为2/3<br>2）x,weight,y中tensor需为2维<br>3）weight中每个tensor的N轴必须相等<br>4）若传入groupListOptional，groupListOptional的差值需与x中tensor的第一维一一对应 |
    | 2 | 单单单 |1）仅支持splitItem为2/3<br>2）x，weight中tensor需为2维，y中tensor需为3维<br>3）必须传groupListOptional，且最后一个值与x中tensor的第二维相等|
  - x和weight中每一组tensor的最后一维大小都应小于65536。$x_i$的最后一维指当属性transpose_x为false时$x_i$的K轴或当transpose_x为true时$x_i$的M轴。$weight_i$的最后一维指当属性transpose_weight为false时$weight_i$的N轴或当transpose_weight为true时$weight_i$的K轴。
  - x和weight中每一组tensor的每一维大小在32字节对齐后都应小于int32的最大值2147483647。
  
## 算子原型

```c++
REG_OP(GroupedMatmul)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT}))
    .DYNAMIC_INPUT(weight, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT, DT_INT4}))
    .DYNAMIC_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .DYNAMIC_INPUT(scale, TensorType({DT_UINT64, DT_BF16, DT_FLOAT32}))
    .DYNAMIC_INPUT(offset, TensorType({DT_FLOAT32}))
    .DYNAMIC_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(group_list, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(per_token_scale, TensorType({DT_FLOAT}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT}))
    .ATTR(split_item, Int, 0)
    .ATTR(dtype, Int, 0)
    .ATTR(transpose_weight, Bool, false)
    .ATTR(transpose_x, Bool, false)
    .ATTR(group_type, Int, -1)
    .ATTR(group_list_type, Int, 0)
    .ATTR(act_type, Int, 0)
    .OP_END_FACTORY_REG(GroupedMatmul)
```

参数解释请参见**算子执行接口**。

## 调用示例
- PyTorch框架调用

  如果通过PyTorch单算子方式调用该融合算子，则需要参考PyTorch融合算子[torch_npu.npu_grouped_matmul](https://hiascend.com/document/redirect/PyTorchAPI)；如果用户定制了该融合算子，则需要参考《Ascend C算子开发》手册[适配PyTorch框架](https://hiascend.com/document/redirect/CannCommunityAscendCInvorkOnNetwork)。

- aclnn单算子调用方式

  通过aclnn单算子调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_v2.h"

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
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // 固定写法，AscendCL初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请Device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
  std::vector<T> hostData(size, 0);
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


int CreateAclTensorList(const std::vector<std::vector<int64_t>>& shapes, void** deviceAddr,
                        aclDataType dataType, aclTensorList** tensor) {
  int size = shapes.size();
  aclTensor* tensors[size];
  for (int i = 0; i < size; i++) {
    int ret = CreateAclTensor<uint16_t>(shapes[i], deviceAddr + i, dataType, tensors + i);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
  }
  *tensor = aclCreateTensorList(tensors, size);
  return ACL_SUCCESS;
}


int main() {
  // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<std::vector<int64_t>> xShape = {{1, 16}, {4, 32}};
  std::vector<std::vector<int64_t>> weightShape= {{16, 24}, {32, 16}};
  std::vector<std::vector<int64_t>> biasShape = {{24}, {16}};
  std::vector<std::vector<int64_t>> yShape = {{1, 24}, {4, 16}};
  void* xDeviceAddr[2];
  void* weightDeviceAddr[2];
  void* biasDeviceAddr[2];
  void* yDeviceAddr[2];
  aclTensorList* x = nullptr;
  aclTensorList* weight = nullptr;
  aclTensorList* bias = nullptr;
  aclIntArray* groupedList = nullptr;
  aclTensorList* scale = nullptr;
  aclTensorList* offset = nullptr;
  aclTensorList* antiquantScale = nullptr;
  aclTensorList* antiquantOffset = nullptr;
  aclTensorList* y = nullptr;
  int64_t splitItem = 0;
  int64_t groupType = -1;

  // 创建x aclTensorList
  ret = CreateAclTensorList(xShape, xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建weight aclTensorList
  ret = CreateAclTensorList(weightShape, weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建bias aclTensorList
  ret = CreateAclTensorList(biasShape, biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建y aclTensorList
  ret = CreateAclTensorList(yShape, yDeviceAddr, aclDataType::ACL_FLOAT16, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 3. 调用CANN算子库API
  // 调用aclnnGroupedMatmulV2第一段接口
  ret = aclnnGroupedMatmulV2GetWorkspaceSize(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupedList, splitItem, groupType, y, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnGroupedMatmulV2第二段接口
  ret = aclnnGroupedMatmulV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmul failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
  for (int i = 0; i < 2; i++) {
    auto size = GetShapeSize(yShape[i]);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), size * sizeof(resultData[0]), yDeviceAddr[i],
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t j = 0; j < size; j++) {
        LOG_PRINT("result[%ld] is: %d\n", j, resultData[j]);
    }
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensorList(x);
  aclDestroyTensorList(weight);
  aclDestroyTensorList(bias);
  aclDestroyTensorList(y);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  for (int i = 0; i < 2; i++) {
    aclrtFree(xDeviceAddr[i]);
    aclrtFree(weightDeviceAddr[i]);
    aclrtFree(biasDeviceAddr[i]);
    aclrtFree(yDeviceAddr[i]);
  }
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```

