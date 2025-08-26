# aclnnDequantRopeQuantKvcache

## 支持的产品型号

- 昇腾910B AI处理器。

## 接口原型

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnDequantRopeQuantKvcacheGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnDequantRopeQuantKvcache”接口执行计算。

* `aclnnStatus aclnnDequantRopeQuantKvcacheGetWorkspaceSize(const aclTensor *x, const aclTensor *cos, const aclTensor *sin, aclTensor *kCacheRef, aclTensor *vCacheRef, const aclTensor *indices, const aclTensor *scaleK, const aclTensor *scaleV, const aclTensor *offsetKOptional, const aclTensor *offsetVOptional, const aclTensor *weightScaleOptional, const aclTensor *activationScaleOptional, const aclTensor *biasOptional, const aclIntArray *sizeSplits, char *quantModeOptional, char *layoutOptional, bool kvOutput, char *cacheModeOptional, const aclTensor *qOut, const aclTensor *kOut, const aclTensor *vOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnDequantRopeQuantKvcache(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## 功能说明

- 算子功能：对输入x进行dequant(可选)后，按`sizeSplits`对尾轴进行切分划分为q，k，v，对q，k，进行旋转位置编码，生成qOut和kOut ，之后对kOut 和v进行量化并按照indices更新到kCacheRef和vCacheRef上。（`sizeSplits` 为切分的长度）
- 计算公式：
  
  $$
  dequantX = Dequant(x,weightScaleOptional,activationScaleOptional,biasOptional)
  $$
  
  $$
  q,k,vOut = SplitTensor(dequantX,dim=-1,`sizeSplits`)
  $$
  
  $$
  qOut,kOut = ApplyRotaryPosEmb(q,k,cos,sin)
  $$
  
  $$
  quantK = Quant(kOut,scaleK,offsetKOptional)
  $$
  
  $$
  quantV = Quant(vOut,scaleV,offsetVOptional)
  $$
  
  如果cacheModeOptional为contiguous则：
  
  $$
  kCacheRef[i][indice[i]]=quantK[i]
  $$
  
  $$
  vCacheRef[i][indice[i]]=quantV[i]
  $$
  
  如果cacheModeOptional为page则：
  
  $$
  kCacheRefView=kCacheRef.view(-1,kCacheRef[-2],kCacheRef[-1])
  $$
  
  $$
  vCacheRefView=vCacheRef.view(-1,vCacheRef[-2],vCacheRef[-1])
  $$
  
  $$
  kCacheRefView[indices[i]]=quantK[i]
  $$
  
  $$
  vCacheRefView[indices[i]]=quantV[i]
  $$

## aclnnDequantRopeQuantKvcacheGetWorkspaceSize

- **参数说明：**
  
  * x(aclTensor\*，计算输入)： 公式中的用于切分的输入`x`，Device侧的aclTensor，shape为[B，S，H]或[B，H]，H=(Nq+Nkv+Nkv)*D，数据类型支持FLOAT16、INT32、BFLOAT16。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND，shape维度只支持2维或3维。
  * cos(aclTensor\*，计算输入)： 公式中的用于位置编码的输入`cos`，Device侧的aclTensor，`x`为3维时shape为[B，S，1，D]，`x`为二维时shape为[B，D]，数据类型支持FLOAT16、BFLOAT16，数据类型和`sin`保持一致。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND，shape维度只支持2维或4维。
  * sin(aclTensor\*，计算输入)： 公式中的用于位置编码的输入`sin`，Device侧的aclTensor，`x`为3维时shape为[B，S，1，D]，`x`为二维时shape为[B，D]，数据类型支持FLOAT16、BFLOAT16，数据类型和`cos`保持一致。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND，shape维度只支持2维或4维。
  * kCacheRef(aclTensor\*，计算输入)： 公式中用于缓存k的的输入`kCacheRef`，Device侧的aclTensor，shape为[C_1，C_2，Nkv，D]，数据类型支持int8。不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND，shape维度只支持4维。
  * vCacheRef(aclTensor\*，计算输入)： 公式中用于缓存v的的输入`vCacheRef`，Device侧的aclTensor，shape为[C_1，C_2，Nkv，D]，数据类型支持int8。不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND，shape维度只支持4维。
  * indices(aclTensor\*，计算输入)： 公式中表示Kvcache的token位置信息的输入`indices`，Device侧的aclTensor，当cache_mode为`page`且x为3维时shape为[B*S]，否则shape为[B]，数据类型支持int32。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND，shape维度只支持1维或2维。
  * scaleK(aclTensor\*，计算输入)： 公式中的输入`scaleK`用于量化`k`的sacle因子，Device侧的aclTensor，shape为[Nkv，D]，数据类型支持FLOAT。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND，shape维度只支持2维。
  * scaleV(aclTensor\*，计算输入)： 公式中的输入`scaleV`用于量化`v`的sacle因子，Device侧的aclTensor，shape为[Nkv，D]，数据类型支持FLOAT。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND，shape维度只支持2维。
  * offsetKOptional(aclTensor\*，计算输入)： 公式中的输入`offsetKoptional`用于量化k的offset因子，Device侧的aclTensor，shape为[Nkv，D]，数据类型支持FLOAT。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND，shape维度只支持2维。
  * offsetVOptional(aclTensor\*，计算输入)： 公式中的输入`offsetVoptional`用于量化的offset因子，Device侧的aclTensor，shape为[Nkv，D]，数据类型支持FLOAT。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND，shape维度只支持2维。
  * weightScaleOptional(aclTensor\*，计算输入)： 公式中的输入`weightScaleoptional`用于反量化的权重scale因子，Device侧的aclTensor，shape为[H]，数据类型支持FLOAT。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND，shape维度只支持1维。
  * activationScaleOptional(aclTensor\*，计算输入)： 公式中的输入`activationScaleOptional`用于反量化的激活scale因子，Device侧的aclTensor，`x`为3维时shape为[B*S]，`x`为二维时shape为[B]，数据类型支持FLOAT。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND，shape维度只支持1维。
  * biasOptional(aclTensor\*，计算输入)： 公式中的输入用于反量化的偏置`biasOptional`，Device侧的aclTensor，shape为[H]，数据类型支持FLOAT、FLOAT16(HALF)、INT32、BFLOAT16。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND，shape维度只支持1维。
  * sizeSplits(aclIntArray \*，计算输入)：host侧的aclIntArray，数据类型为int数组，size大小为3，值为[Nq * D，Nkv * D，Nkv * D]。表示输入的qkv进行切分的长度。
  * quantModeOptional(char\*，计算输入)：Host侧 表达式字符串。表示支持的量化类型，目前仅支持`static`。
  * layoutOptional(char\*，计算输入)：Host侧 表达式字符串。表示支持的数据格式，目前仅支持`BSND`。
  * kvOutput(bool，计算输入)：Host侧 表达式布尔值。表示是否输出`k`和`v`，默认false。
  * cacheModeOptional(char\*，计算输入)：Host侧 表达式字符串。表示`kCacheRef`的更新方式，目前仅支持`page`和`contiguous`，默认为`contiguous`。
  * qOut(aclTensor\*，计算输出)： 公式中的输出`qOut`，表示经过处理的q，Device侧的aclTensor，`x`为3维时shape为[B，S，Nq，D]，`x`为二维时shape为[B，Nq，D]，数据类型支持FLOAT16、BFLOAT16，数据类型和`sin`保持一致。不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  * kOut(aclTensor\*，计算输出)： 公式中的输出`kOut`，表示经过处理的k，Device侧的aclTenso，当`kv_outputOptional`为false时`kOut`为空，否则，`x`为3维时shape为[B，S，Nkv，D]，`x`为二维时shape为[B，Nkv，D]，数据类型支持FLOAT16、BFLOAT16，数据类型和`sin`保持一致。不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  * vOut(aclTensor\*，计算输出)： 公式中的输出`vOut`，表示经过处理的v，Device侧的aclTensor，当`kv_outputOptional`为false时`vOut`为空，否则，`x`为3维时shape为[B，S，Nkv，D]，`x`为二维时shape为[B，Nkv，D]，数据类型支持FLOAT16、BFLOAT16，数据类型和`sin`保持一致。不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  * workspaceSize(uint64_t\*，出参)： 返回需要在Device侧申请的workspace大小。
  * executor(aclOpExecutor\*\*，出参)： 返回op执行器，包含了算子计算流程。
- **返回值：**
  
  aclnnStatus： 返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。
  
  ```
  第一段接口完成入参校验，出现以下场景时报错：
    161001(ACLNN_ERR_PARAM_NULLPTR): 1. 输入和输出的Tensor是空指针。
    161002(ACLNN_ERR_PARAM_INVALID): 1. 输入和输出的数据类型不在支持的范围内。
  ```

## aclnnDequantRopeQuantKvcache

- **参数说明：**
  
  * workspace(void\*， 入参)： 在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t， 入参)： 在Device侧申请的workspace大小，由第一段接口aclnnDequantRopeQuantKvcacheGetWorkspaceSize获取。
  * executor(aclOpExecutor\*， 入参)： op执行器，包含了算子计算流程。
  * stream(aclrtStream， 入参)： 指定执行任务的 AscendCL Stream流。
- **返回值：**
  
  aclnnStatus： 返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明

1. cacheModeOptional为contiguous时：kCacheRef的第0维大于x的第0维，indices数据值大于等于0且小于等于vCacheRef的第1维([b，s，n，d]格式中的s)减x的第1维；cacheModeOptional为page时：inidces 数据值大于等于0，小于kCacheRef的第0维*第1维且不重复。
2. x的尾轴小于等于4096，且按64对齐
3. 输入x不为int32时，x、cos、sin与输出qOut、kOut、vOut的数据类型保持一致，此时activationScaleOptional，weightScaleOptional、biasOptional不生效；x为int32时，cos、sin与输出qOut、kOut、vOut的数据类型保持一致，此时weightScaleOptional必选，activationScaleOptional、biasOptional可选（biasOptional不需要与其他输入类型一致）。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_dequant_rope_quant_kvcache.h"

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

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<int8_t> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %d\n", i, resultData[i]);
  }
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
  // 1. 固定写法，device/stream初始化, 参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口定义构造
  std::vector<int64_t> inputShape = {320, 1, 1280};
  std::vector<int64_t> cosShape = {320, 1, 1, 128};
  std::vector<int64_t> sinShape = {320, 1, 1, 128};
  std::vector<int64_t> kcacheShape = {320, 1280, 1, 128};
  std::vector<int64_t> vcacheShape = {320, 1280, 1, 128};
  std::vector<int64_t> indicesShape = {320};
  std::vector<int64_t> kscaleShape = {128};
  std::vector<int64_t> vscaleShape = {128};
  std::vector<int64_t> koffsetShape = {128};
  std::vector<int64_t> voffsetShape = {128};

  std::vector<int64_t> weightShape = {1280};
  std::vector<int64_t> activationShape = {1280};
  std::vector<int64_t> biasShape = {8192};

  std::vector<int16_t> inputHostData(320*1280, 1);
  std::vector<int16_t> cosHostData(320*128, 1);
  std::vector<int16_t> sinHostData(320*128, 1);
  std::vector<int8_t> kcacheHostData(320*1280*128, 6);
  std::vector<int8_t> vcacheHostData(320*1280*128, 6);
  std::vector<int32_t> indicesHostData(320, 0);
  std::vector<int32_t> kscaleHostData(128, 2);
  std::vector<int32_t> vscaleHostData(128, 2);
  std::vector<int32_t> koffsetHostData(128, 2);
  std::vector<int32_t> voffsetHostData(128, 2);

  std::vector<int32_t> weightHostData(1280, 2);
  std::vector<int32_t> activationHostData(1280, 2);
  std::vector<int32_t> biasHostData(8192, 2);

  void* inputDeviceAddr = nullptr;
  void* cosDeviceAddr = nullptr;
  void* sinDeviceAddr = nullptr;
  void* kcacheDeviceAddr = nullptr;
  void* vcacheDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* kscaleDeviceAddr = nullptr;
  void* vscaleDeviceAddr = nullptr;
  void* koffsetDeviceAddr = nullptr;
  void* voffsetDeviceAddr = nullptr;

  void* weightDeviceAddr = nullptr;
  void* activationDeviceAddr = nullptr;
  void* biasDeviceAddr = nullptr;

  aclTensor* input = nullptr;
  aclTensor* cos = nullptr;
  aclTensor* sin = nullptr;
  aclTensor* kcache = nullptr;
  aclTensor* vcache = nullptr;
  aclTensor* indices = nullptr;
  aclTensor* kscale = nullptr;
  aclTensor* vscale = nullptr;
  aclTensor* koffset = nullptr;
  aclTensor* voffset = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* activation = nullptr;
  aclTensor* bias = nullptr;

  ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_INT32, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cosHostData, cosShape, &cosDeviceAddr, aclDataType::ACL_FLOAT16, &cos);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sinHostData, sinShape, &sinDeviceAddr, aclDataType::ACL_FLOAT16, &sin);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kcacheHostData, kcacheShape, &kcacheDeviceAddr, aclDataType::ACL_INT8, &kcache);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vcacheHostData, vcacheShape, &vcacheDeviceAddr, aclDataType::ACL_INT8, &vcache);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT32, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kscaleHostData, kscaleShape, &kscaleDeviceAddr, aclDataType::ACL_FLOAT, &kscale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vscaleHostData, vscaleShape, &vscaleDeviceAddr, aclDataType::ACL_FLOAT, &vscale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(koffsetHostData, koffsetShape, &koffsetDeviceAddr, aclDataType::ACL_FLOAT, &koffset);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(voffsetHostData, voffsetShape, &voffsetDeviceAddr, aclDataType::ACL_FLOAT, &voffset);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(activationHostData, activationShape, &activationDeviceAddr, aclDataType::ACL_FLOAT, &activation);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);


  std::vector<int64_t> qShape = {320,1,8,128};
  std::vector<int16_t> qHostData(320*8*128, 9);
  aclTensor* q = nullptr;
  void* qDeviceAddr = nullptr;
  std::vector<int64_t> kShape = {320,1,1,128};
  std::vector<int16_t> kHostData(320*128, 10);
  aclTensor* k = nullptr;
  void* kDeviceAddr = nullptr;
  std::vector<int64_t> vShape = {320,1,1, 128};
  std::vector<int16_t> vHostData(320*128, 10);
  aclTensor* v = nullptr;
  void* vDeviceAddr = nullptr;

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> splitData = {1024, 128, 128};
  aclIntArray *sizeSplits = aclCreateIntArray(splitData.data(), splitData.size());

  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnDequantRopeQuantKvcache第一段接口
  ret = aclnnDequantRopeQuantKvcacheGetWorkspaceSize(input, cos, sin, kcache, vcache, indices, kscale, vscale, koffset,
                                                     voffset, weight, activation, bias,sizeSplits, "static", "BSND", true,
                                                     "contiguous", q, k, v, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDequantRopeQuantKvcacheGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnDequantRopeQuantKvcache第二段接口
  ret = aclnnDequantRopeQuantKvcache(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDequantRopeQuantKvcache failed. ERROR: %d\n", ret); return ret);

  // 4. 固定写法，同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(kcacheShape, &kcacheDeviceAddr);
  PrintOutResult(vcacheShape, &vcacheDeviceAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(input);
  aclDestroyTensor(q);

  // 7. 释放device 资源
  aclrtFree(inputDeviceAddr);
  aclrtFree(qDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
