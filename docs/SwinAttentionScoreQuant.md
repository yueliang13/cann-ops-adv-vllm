声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# aclnnSwinAttentionScoreQuant

## 支持的产品型号

- Atlas 推理系列加速卡产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能说明

+ 算子功能：完成swin-transformer场景的Attetion计算，相较于SwinAttentionScore算子，支持int8量化功能
+ 计算公式如下：

$$
out= Softmax(QK^T + bias1 + bias2)V
$$

## 算子执行接口

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnSwinAttentionScoreQuantGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSwinAttentionScoreQuant”接口执行计算。

* `aclnnStatus aclnnSwinAttentionScoreQuantGetWorkspaceSize(const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *scaleQuant, const aclTensor *scaleDequant1, const aclTensor *scaleDequant2, const aclTensor *biasQuantOptional, const aclTensor *biasDequant1Optional, const aclTensor *biasDequant2Optional, const aclTensor *paddingMask1Optional, const aclTensor *paddingMask2Optional, bool queryTranspose, bool keyTranspose, bool valueTranspose, int64_t softmaxAxes, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnSwinAttentionScoreQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnSwinAttentionScoreQuantGetWorkspaceSize

* **参数说明**：
  - query(aclTensor*,计算输入)：表示输入样本的查询张量，Device侧的aclTensor，公式中的Q，维度支持四维，输入维度[N,C,S,H]需要和key、value保持一致，其中N代表batch size，C为通道深度，S为序列长度，H为headNum，S<=1024，H=32/64，NC维度支持任意值，数据类型支持INT8，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - key(aclTensor*,计算输入): 表示输入样本的每个位置的特征张量，Device侧的aclTensor，公式中的K，维度支持四维，输入维度[N,C,S,H]需要和query、value保持一致，S<=1024，H=32/64，NC维度支持任意值，数据类型支持INT8，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - value(aclTensor*,计算输入): 表示计算注意力后的每个位置值张量，Device侧的aclTensor，公式中的V，维度支持四维，输入维度[N,C,S,H]需要和query、key保持一致，S<=1024，H=32/64，NC维度支持任意值，数据类型支持INT8，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - scaleQuant(aclTensor*,计算输入): 表示对注意力softmax归一化后进行量化的尺度缩放张量，Device侧的aclTensor，维度支持二维，输入维度[1,S]，S<=1024，数据类型支持FLOAT16，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - scaleDequant1(aclTensor*,计算输入)：表示计算注意力时进行反量化的尺度缩放张量，Device侧的aclTensor，维度支持二维，输入维度[1,S]，S<=1024，数据类型支持UINT64，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - scaleDequant2(aclTensor*,计算输入): 表示计算注意力后的输出进行反量化的尺度缩放张量，Device侧的aclTensor，维度支持二维，输入维度[1,H]，H=32/64，数据类型支持UINT64，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - biasQuantOptional(aclTensor*,计算输入): 表示对注意力softmax归一化后进行量化的偏移张量，Device侧的aclTensor，维度支持二维，输入维度[1,S]，S<=1024，数据类型支持FLOAT16，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - biasDequant1Optional(aclTensor*,计算输入): 表示计算注意力时进行反量化的偏移张量，Device侧的aclTensor，维度支持二维，输入维度[1,S]，S<=1024，数据类型支持INT32，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - biasDequant2Optional(aclTensor*,计算输入)：表示计算注意力后的输出进行反量化的偏移张量 Device侧的aclTensor，维度支持二维，输入维度[1,H]，H=32/64，数据类型支持INT32，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - paddingMask1Optional(aclTensor*,计算输入): Device侧的aclTensor，公式中的bias1，支持输入nullptr，或者四维的tensor，维度[1,C,S,S]，S<=1024，C维度支持任意值，数据类型支持FLOAT16，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - paddingMask2Optional(aclTensor*,计算输入): 预留参数，公式中的bias2，当前仅支持输入nullptr。
  - queryTranspose(bool，计算输入): Host侧的bool，表示query是否转置，当前仅支持不转置false。
  - keyTranspose(bool，计算输入): Host侧的bool，表示key是否转置，当前仅支持不转置false。
  - valueTranspose(bool，计算输入): Host侧的bool，表示value是否转置，当前仅支持不转置false。
  - softmaxAxes(int，计算输入): Host侧的int，用于指定softmax计算的维度，当前仅支持取-1（即tensor的最后一维）。
  - out(aclTensor*, 计算输出)：Device侧的aclTensor，维度支持四维，输出维度[N,C,S,H]，S<=1024，H=32/64，数据类型支持FLOAT16，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - workspaceSize(uint64_t*, 出参)：返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor**, 出参)：返回op执行器，包含了算子计算流程。
* **返回值**：
  aclnnStatus：返回状态码，具体参见[aclnn返回码](./common/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  返回161001(ACLNN_ERR_PARAM_NULLPTR): 1. 传入的tensor是空指针。
  返回161002(ACLNN_ERR_PARAM_INVALID): 1. 输入或输出参数的数据类型/数据格式不在支持的范围。
  ```

## aclnnSwinAttentionScoreQuant

* **参数说明**

  - workspace(void *，入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t，入参)：在Device侧申请的workspace大小，由第一段接口aclnnSwinAttentionScoreQuantGetWorkspaceSize获取。
  - executor(aclOpExecutor *，入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream，入参)：指定执行任务的AscendCL Stream流。

* **返回值**

  返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明
- QKV输入维度是[N,C,S,H]的情况下，S<=1024，H=32/64，NC维度支持任意值
- 不支持维度是[N,C,S,H]的QKV转置后输入
- 只支持非对称量化
- 不支持加bias2的功能
- 只支持对QK^T + bias1 + bias2的最后一维进行softmax操作

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_swin_attention_score_quant.h"

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
    int64_t B = 1288;
    int64_t N = 3;
    int64_t S = 49;
    int64_t H = 32;
    std::vector<int64_t> qkvShape = {B, N, S, H};
    std::vector<int64_t> sShape = {1, S};
    std::vector<int64_t> hShape = {1, H};
    std::vector<int64_t> mask1Shape = {1, N, S, S};
    std::vector<int64_t> attentionScoreShape = {B, N, S, H};
    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;
    void* valueDeviceAddr = nullptr;
    void* scaleQuantDeviceAddr = nullptr;
    void* scaleDequant1DeviceAddr = nullptr;
    void* scaleDequant2DeviceAddr = nullptr;
    void* biasQuantDeviceAddr = nullptr;
    void* biasDequant1DeviceAddr = nullptr;
    void* biasDequant2DeviceAddr = nullptr;
    void* paddingMask1DeviceAddr = nullptr;
    void* attentionScoreDeviceAddr = nullptr;
    aclTensor* query = nullptr;
    aclTensor* key = nullptr;
    aclTensor* value = nullptr;
    aclTensor* scaleQuant = nullptr;
    aclTensor* scaleDequant1 = nullptr;
    aclTensor* scaleDequant2 = nullptr;
    aclTensor* biasQuantOptional = nullptr;
    aclTensor* biasDequant1Optional = nullptr;
    aclTensor* biasDequant2Optional = nullptr;
    aclTensor* paddingMask1Optional = nullptr;
    aclTensor* paddingMask2Optional = nullptr;
    aclTensor* attentionScore = nullptr;
    std::vector<int8_t> queryHostData(B*N*S*H, 1);
    std::vector<int8_t> keyHostData(B*N*S*H, 1);
    std::vector<int8_t> valueHostData(B*N*S*H, 1);
    std::vector<uint16_t> scaleQuantHostData(S, 1);
    std::vector<uint64_t> scaleDequant1HostData(S, 1);
    std::vector<uint64_t> scaleDequant2HostData(H, 1);
    std::vector<uint16_t> biasQuantHostData(S, 1);
    std::vector<int32_t> biasDequant1HostData(S, 1);
    std::vector<int32_t> biasDequant2HostData(H, 1);
    std::vector<uint16_t> paddingMask1HostData(1*N*S*H, 1);
    std::vector<uint16_t> attentionScoreHostData(B*N*S*H, 0);
    // 创建input aclTensor
    ret = CreateAclTensor(queryHostData, qkvShape, &queryDeviceAddr, aclDataType::ACL_INT8, &query);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(keyHostData, qkvShape, &keyDeviceAddr, aclDataType::ACL_INT8, &key);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(valueHostData, qkvShape, &valueDeviceAddr, aclDataType::ACL_INT8, &value);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scaleQuantHostData, sShape, &scaleQuantDeviceAddr, aclDataType::ACL_FLOAT16, &scaleQuant);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scaleDequant1HostData, sShape, &scaleDequant1DeviceAddr, aclDataType::ACL_UINT64, &scaleDequant1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scaleDequant2HostData, hShape, &scaleDequant2DeviceAddr, aclDataType::ACL_UINT64, &scaleDequant2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(biasQuantHostData, sShape, &biasQuantDeviceAddr, aclDataType::ACL_FLOAT16, &biasQuantOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(biasDequant1HostData, sShape, &biasDequant1DeviceAddr, aclDataType::ACL_INT32, &biasDequant1Optional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(biasDequant2HostData, hShape, &biasDequant2DeviceAddr, aclDataType::ACL_INT32, &biasDequant2Optional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(paddingMask1HostData, mask1Shape, &paddingMask1DeviceAddr, aclDataType::ACL_FLOAT16, &paddingMask1Optional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(attentionScoreHostData, attentionScoreShape, &attentionScoreDeviceAddr, aclDataType::ACL_FLOAT16, &attentionScore);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // aclnn接口调用示例
    // 3. 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnn第一段接口
    ret = aclnnSwinAttentionScoreQuantGetWorkspaceSize(query, key, value, scaleQuant, scaleDequant1, scaleDequant2,
        biasQuantOptional, biasDequant1Optional, biasDequant2Optional, paddingMask1Optional, paddingMask2Optional,
        false, false, false, -1, attentionScore, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSubsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnn第二段接口
    ret = aclnnSwinAttentionScoreQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSubs failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧
    auto size = GetShapeSize(attentionScoreShape);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), attentionScoreDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(query);
    aclDestroyTensor(key);
    aclDestroyTensor(value);
    aclDestroyTensor(scaleQuant);
    aclDestroyTensor(scaleDequant1);
    aclDestroyTensor(scaleDequant2);
    aclDestroyTensor(biasQuantOptional);
    aclDestroyTensor(biasDequant1Optional);
    aclDestroyTensor(biasDequant2Optional);
    aclDestroyTensor(paddingMask1Optional);
    aclDestroyTensor(attentionScore);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(queryDeviceAddr);
    aclrtFree(keyDeviceAddr);
    aclrtFree(valueDeviceAddr);
    aclrtFree(scaleQuantDeviceAddr);
    aclrtFree(scaleDequant1DeviceAddr);
    aclrtFree(scaleDequant2DeviceAddr);
    aclrtFree(biasQuantDeviceAddr);
    aclrtFree(biasDequant1DeviceAddr);
    aclrtFree(biasDequant2DeviceAddr);
    aclrtFree(paddingMask1DeviceAddr);
    aclrtFree(attentionScoreDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```