声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# aclnnSwinTransformerLnQkvQuant

## 支持的产品型号

- Atlas 推理系列加速卡产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能说明
- 算子功能：Swin Transformer 网络模型 完成 Q、K、V 的计算。  
- 计算公式：  

  q/k/v = (Quant(Layernorm(x).transpose)  * weight).dequant.transpose.split
  其中，weight 是 Q、K、V 三个矩阵权重的拼接。

## 算子执行接口

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnSwinTransformerLnQkvQuantGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSwinTransformerLnQkvQuant”接口执行计算。
* `aclnnStatus aclnnSwinTransformerLnQkvQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, const aclTensor *beta, const aclTensor *weight, const aclTensor *bias, const aclTensor *quantScale, const aclTensor *quantOffset, const aclTensor *dequantScale, int64_t headNum, int64_t seqLength, double epsilon, int64_t oriHeight, int64_t oriWeight, int64_t hWinSize, int64_t wWinSize, bool weightTranspose, const aclTensor *queryOutputOut, const aclTensor *keyOutputOut, const aclTensor *valueOutputOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnSwinTransformerLnQkvQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnSwinTransformerLnQkvQuantGetWorkspaceSize
- **参数说明**：
  - x(aclTensor*,计算输入): 表示待进行归一化计算的目标张量，公式中的x， Device侧的aclTensor，数据类型支持FLOAT16。只支持维度为[B,S,H]，其中B为batch size且只支持[1,32],S为原始图像长宽的乘积，H为序列长度和通道数的乘积且小于等于1024，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - gamma(aclTensor*,计算输入): 表示layernrom计算中尺度缩放的大小，维度只支持1维且为[H]，Device侧的aclTensor，数据类型支持FLOAT16，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - beta(aclTensor*,计算输入): 表示layernrom计算中尺度偏移的大小，维度只支持1维且维度为[H]，Device侧的aclTensor，数据类型支持FLOAT16，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - weight(aclTensor*,计算输入): 表示目标张量转换使用的权重矩阵，维度只支持2维且维度为[H, 3 * H],Device侧的aclTensor，数据类型支持INT8，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - bias(aclTensor*,计算输入): 表示目标张量转换使用的偏移矩阵，维度只支持1维且维度为[3 * H]，Device侧的aclTensor，数据类型支持INT32，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - quantScale(aclTensor*,计算输入):  表示目标张量量化使用的缩放参数，维度只支持1维且维度为[H]，Device侧的aclTensor，数据类型支持FLOAT16，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - quantOffset(aclTensor*,计算输入): 表示目标张量量化使用的偏移参数，维度只支持1维且维度为[H]，Device侧的aclTensor，数据类型支持FLOAT16，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - dequantScale(aclTensor*,计算输入): 表示目标张量乘以权重矩阵之后反量化使用的缩放参数，维度只支持1维且维度为[3 * H]，Device侧的aclTensor，数据类型支持UINT64，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - headNum(int，计算输入): 表示转换使用的通道数；支持范围[1,32]。
  - seqLength(int，计算输入): 表示转换使用的通道深度。只支持32/64两种。
  - epsilon(float,计算输入): layernrom 计算除0保护值；为了保证精度，建议小于等于1e-4。
  - oriHeight(int,计算输入): layernrom 中S轴transpose的维度；oriHeight*oriWeight需等于输入x的第二维S的大小，且为hWinSize的整数倍。
  - oriWeight(int,计算输入): layernrom 中S轴transpose的维度；oriHeight*oriWeight需等于输入x的第二维S的大小，且为wWinSize的整数倍。
  - hWinSize(int,计算输入): 使用的特征窗高度大小；支持范围[7,32]。
  - wWinSize(int,计算输入): 使用的特征窗宽度大小；支持范围[7,32]。
  - weightTranspose(bool,计算输入): weight矩阵需要转置，当前不支持不转置场景。
  - queryOutputOut(aclTensor*, 计算输出)：表示转换之后的张量，公式中的Q，Device侧的aclTensor，数据类型支持FLOAT16，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - keyOutputOut(aclTensor*, 计算输出)：表示转换之后的张量，公式中的K，Device侧的aclTensor，数据类型支持FLOAT16，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - valueOutputOut(aclTensor*, 计算输出)：表示转换之后的张量，公式中的V，Device侧的aclTensor，数据类型支持FLOAT16，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - workspaceSize(uint64_t*，出参)：返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor**，出参)：返回op执行器，包含了算子计算流程。
- **返回值**：
  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。
  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  161001(ACLNN_ERR_PARAM_NULLPTR)：1. 传入的输入tensor是空指针。
  161002(ACLNN_ERR_PARAM_INVALID)：1. 输入或输出参数的数据类型/数据格式不在支持的范围内。
  ```

## aclnnSwinTransformerLnQkvQuant
- **参数说明**：
  - workspace(void \*, 入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnSwinTransformerLnQkvQuantGetWorkspaceSize获取。
  - executor(aclOpExecutor \*, 入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream, 入参)：指定执行任务的AscendCL Stream流。

- **返回值**：
  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明
- seqLength只支持32/64。
- oriHeight*oriWeight=输入x Tensor的第二维度，且oriHeight为hWinSize的整数倍，oriWeight为wWinSize的整数倍。
- hWinSize和wWinSize范围只支持7~32。
- 输入x Tensor的第一维度B只支持1~32。
- weight需要转置。

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_swin_transformer_ln_qkv_quant.h"

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
  // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {1, 49, 32};
  std::vector<int64_t> gammaShape = {32};
  std::vector<int64_t> weightShape = {32*3, 32};
  std::vector<int64_t> biasShape = {3 * 32};
  std::vector<int64_t> outShape = {1,1,49, 32};
  void* xDeviceAddr = nullptr;
  void* gammaDeviceAddr = nullptr;
  void* betaDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* biasDeviceAddr = nullptr;
  void* scaleDeviceAddr = nullptr;
  void* offsetDeviceAddr = nullptr;
  void* dequantDeviceAddr = nullptr;

  void* outqDeviceAddr = nullptr;
  void* outkDeviceAddr = nullptr;
  void* outvDeviceAddr = nullptr;
  aclTensor* x = nullptr;
  aclTensor* gamma = nullptr;
  aclTensor* beta = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* bias = nullptr;
  aclTensor* quantScale = nullptr;
  aclTensor* quantOffset = nullptr;
  aclTensor* dequantScale = nullptr;
  aclTensor* queryOutput = nullptr;
  aclTensor* keyOutput = nullptr;
  aclTensor* valueOutput = nullptr;

  std::vector<uint16_t> selfHostData(49*32, 0x1);
  std::vector<int32_t> biasHostData(3*32, 0x1);
  std::vector<uint16_t> gammaHostData(32, 0x1);
  std::vector<uint16_t> betaHostData(32, 0x1);
  std::vector<int8_t> weightHostData(3*32*32, 0x1);
  std::vector<uint16_t> scaleHostData(32, 0x1);
  std::vector<uint16_t> offsetHostData(32, 0x1);
  std::vector<uint64_t> dequantHostData(3*32, 0x1);

  std::vector<uint16_t> outqHostData(49*32, 0x0);
  std::vector<uint16_t> outkHostData(49*32, 0x0);
  std::vector<uint16_t> outvHostData(49*32, 0x0);

  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(betaHostData, gammaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT16, &beta);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_INT8, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_INT32, &bias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(scaleHostData, gammaShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT16, &quantScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(offsetHostData, gammaShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT16, &quantOffset);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(dequantHostData, biasShape, &dequantDeviceAddr, aclDataType::ACL_UINT64, &dequantScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outqHostData, outShape, &outqDeviceAddr, aclDataType::ACL_FLOAT16, &queryOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(outkHostData, outShape, &outkDeviceAddr, aclDataType::ACL_FLOAT16, &keyOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outvHostData, outShape, &outvDeviceAddr, aclDataType::ACL_FLOAT16, &valueOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  float epsilon = 0.0001;
  int64_t oriHeight = 7;
  int64_t oriWeight = 7;
  int64_t hWinSize = 7;
  int64_t wWinSize = 7;
  int64_t headNum = 1;
  int64_t seqLength = 32;
  bool weightTranspose = true;

  // 调用aclnnSwinTransformerLnQkvQuant第一段接口
  ret = aclnnSwinTransformerLnQkvQuantGetWorkspaceSize(x,gamma,beta,weight, bias, quantScale, quantOffset, dequantScale, headNum, seqLength, epsilon, oriHeight, oriWeight, hWinSize, wWinSize, weightTranspose, queryOutput, keyOutput, valueOutput, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwinTransformerLnQkvQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnSwinTransformerLnQkvQuant第二段接口
  ret = aclnnSwinTransformerLnQkvQuant(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwinTransformerLnQkvQuant failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<uint16_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outqDeviceAddr,size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(x);
  aclDestroyTensor(queryOutput);
  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(xDeviceAddr);
  aclrtFree(outqDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```