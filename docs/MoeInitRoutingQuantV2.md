声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# MoeInitRoutingQuantV2

## 支持的产品型号

Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能说明

- 算子功能：该算子对应MoE（Mixture of Experts，混合专家模型）中的**Routing计算**，以MoeGatingTopKSoftmax算子的输出x和expert_idx作为输入，并对输出量化后的Routing矩阵expanded_x等结果供后续计算使用。本接口相对接口[MoeInitRoutingV2](./MoeInitRoutingV2.md)增加了对输出expandedXOut的量化操作

    **说明：**
    Routing计算是MoE模型中的一个环节。MoE模型主要由一组专家模型和一个门控模型组成，在计算时，输入的数据会先根据门控网络（Gating Network，包含MoeGatingTopKSoftmax算子）计算出每个数据元素对应权重最高的k个专家，然后该结果会输入MoeInitRouting算子，生成Routing矩阵。在后续，模型中的每个专家会根据Routing矩阵处理其应处理的数据，产生相应的输出。各专家的输出最后与权重加权求和，形成最终的预测结果。

- 计算公式：  

    1.对输入expertIdx做排序，得出排序后的结果sortedExpertIdx和对应的序号sortedRowIdx：
      $$
      sortedExpertIdx, sortedRowIdx=keyValueSort(expertIdx)
      $$
    2.以sortedRowIdx做位置映射得出expandedRowIdxOut：
      $$
      expandedRowIdxOut[sortedRowIdx[i]]=i
      $$

    3.在dropless模式下，对sortedExpertIdx的每个专家统计直方图结果，再进行Cumsum，得出expertTokensCountOrCumsumOut：
      $$
      expertTokensCountOrCumsumOut[i]=Cumsum(Histogram(sortedExpertIdx))
      $$

    4.在drop模式下，对sortedExpertIdx的每个专家统计直方图结果，得出expertTokensBeforeCapacityOut：
      $$
      expertTokensBeforeCapacityOut[i]=Histogram(sortedExpertIdx)
      $$

    5.计算quant结果：
      - 静态quant：
          $$
          quantResult = round((x * scaleOptional) + offsetOptional)
          $$
      - 动态quant：
          - 若不输入scale：
              $$
              dynamicQuantScaleOut = row\_max(abs(x)) / 127
              $$
              $$
              quantResult = round(x / dynamicQuantScaleOut)
              $$
          - 若输入scale:
              $$
              dynamicQuantScaleOut = row\_max(abs(x * scaleOptional)) / 127
              $$
              $$
              quantResult = round(x / dynamicQuantScaleOut)
              $$
    6.对quantResult取前NUM\_ROWS个sortedRowIdx的对应位置的值，得出expandedXOut：
      $$
      expandedXOut[i]=quantResult[sortedRowIdx[i]\%NUM\_ROWS]
      $$


## 算子执行接口

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用 “aclnnMoeInitRoutingQuantV2GetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeInitRoutingQuantV2”接口执行计算。

* `aclnnStatus aclnnMoeInitRoutingQuantV2GetWorkspaceSize(const aclTensor *x, const aclTensor *expertIdx, const aclTensor *scaleOptional, const aclTensor *offsetOptional, int64_t activeNum, int64_t expertCapacity, int64_t expertNum, int64_t dropPadMode, int64_t expertTokensCountOrCumsumFlag, bool expertTokensBeforeCapacityFlag, int64_t quantMode, aclTensor *expandedXOut, aclTensor *expandedRowIdxOut, aclTensor *expertTokensCountOrCumsumOut, aclTensor *expertTokensBeforeCapacityOut, aclTensor *dynamicQuantScaleOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnMoeInitRoutingQuantV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。


### aclnnMoeInitRoutingQuantV2GetWorkspaceSize

-   **参数说明**：
    -   x（aclTensor\*，计算输入）：MOE的输入即token特征输入，要求为一个2D的Tensor，shape为[NUM\_ROWS, H]，H代表每个Token的长度，数据类型支持FLOAT16、BFLOAT16、FLOAT32，[数据格式](common/数据格式.md)要求为ND，支持[非连续的Tensor](common/非连续的Tensor.md)。
    -   expertIdx （aclTensor\*，计算输入）：[aclnnMoeGatingTopKSoftmaxV2](aclnnMoeGatingTopKSoftmaxV2.md)的输出每一行特征对应的K个处理专家，要求是一个2D的shape [NUM\_ROWS, K]。数据类型支持INT32，[数据格式](common/数据格式.md)要求为ND，支持[非连续的Tensor](common/非连续的Tensor.md)。在Drop/Pad场景下或者非Drop/Pad场景下且需要输出expertTokensCountOrCumsumOut时，要求值域范围是[0, expertNum - 1]，其他场景要求大于等于0。
    -   scaleOptional（aclTensor\*，计算输入）：表示用于计算quant结果的参数，要求静态quant场景下必须输入，为一个1D的shape [1，]。动态quant场景下如果不输入，表示计算过程中不用scale；如果输入则要求为一个2D的Tensor，shape为 [expertNum，H]或者[1，H]。数据类型支持float32，[数据格式](common/数据格式.md)要求为ND，支持[非连续的Tensor](common/非连续的Tensor.md)。
    -   offsetOptional（aclTensor\*，计算输入）：表示用于计算quant结果的偏移值，要求在静态quant场景下必须输入，为一个1D的shape [1，]。数据类型支持FLOAT32，[数据格式](common/数据格式.md)要求为ND，支持[非连续的Tensor](common/非连续的Tensor.md)。
    -   activeNum（int64\_t，计算输入）：表示是否为Active场景，该属性在dropPadMode为0时生效，值范围大于等于0；0表示Dropless场景，大于0时表示Active场景，约束所有专家共同处理tokens总量
    -   expertCapacity（int64\_t， 计算输入）：表示每个专家能够处理的tokens数，值范围大于等于0；Drop/Pad场景下值域范围\(0, NUM\_ROWS\]，此时各专家将超过capacity的tokens drop掉，不够capacity阈值时则pad全0 tokens；其他场景不关心该属性值。
    -   expertNum（int64\_t， 计算输入）：表示专家数，值范围大于等于0；Drop/Pad场景下或者expertTokensCountOrCumsumFlag大于0需要输出expertTokensCountOrCumsumOut时，expertNum需大于0。
    -   dropPadMode（int64\_t， 计算输入）：表示是否为Drop/Pad场景，取值为0和1。
        - 0：表示非Drop/Pad场景，该场景下不校验expertCapacity。
        - 1：表示Drop/Pad场景，需要校验expertNum和expertCapacity，对于每个专家处理的超过和不足expertCapacity的值会做相应的处理。
    -   expertTokensCountOrCumsumFlag（int64\_t， 计算输入）：取值为0、1和2。
        - 0：表示不输出expertTokensCountOrCumsumOut。
        - 1：表示输出的值为各个专家处理的token数量的累计值。
        - 2：表示输出的值为各个专家处理的token数量。
    -   expertTokensBeforeCapacityFlag（bool，计算输入）：取值为false和true。
        - false：表示不输出expertTokensBeforeCapacityOut。
        - true：表示输出的值为在drop之前各个专家处理的token数量。
    -   quantMode（int64\_t， 计算输入）：取值为0和1。
        - 0：表示静态quant场景。
        - 1：表示动态quant场景。
    -   expandedXOut（aclTensor\*，计算输出）：根据expertIdx进行扩展过的特征，在Dropless/Active场景下要求是一个2D的Tensor，Dropless场景shape为[NUM\_ROWS \* K, H]，Active场景shape为[min\(activeNum, NUM\_ROWS \* K\), H]，在Drop/Pad场景下要求是一个3D的Tensor，shape为[expertNum, expertCapacity, H]。数据类型支持INT8，[数据格式](common/数据格式.md)要求为ND，不支持[非连续的Tensor](common/非连续的Tensor.md)。
    -   expandedRowIdxOut（aclTensor\*，计算输出）：expandedXOut和x的索引映射关系， 要求是一个1D的Tensor，Shape为[NUM\_ROWS\*K, ]，数据类型支持INT32，[数据格式](common/数据格式.md)要求为ND，不支持[非连续的Tensor](common/非连续的Tensor.md)。
    -   expertTokensCountOrCumsumOut（aclTensor\*，计算输出）：输出每个专家处理的token数量的统计结果及累加值，通过expertTokensCountOrCumsumFlag参数控制是否输出，该值仅在非Drop/Pad场景下输出，要求是一个1D的Tensor，Shape为[expertNum, ]，数据类型支持INT32，[数据格式](common/数据格式.md)要求为ND，不支持[非连续的Tensor](common/非连续的Tensor.md)。
    -   expertTokensBeforeCapacityOut（aclTensor\*，计算输出）：输出drop之前每个专家处理的token数量的统计结果，通过expertTokensBeforeCapacityFlag参数控制是否输出，该值仅在Drop/Pad场景下输出，要求是一个1D的Tensor，Shape为[expertNum, ]，数据类型支持INT32，[数据格式](common/数据格式.md)要求为ND，不支持[非连续的Tensor](common/非连续的Tensor.md)。
    -   dynamicQuantScaleOut（aclTensor\*，计算输出）：输出动态quant计算过程中的中间值，该值仅在动态quant场景下输出，要求是一个1D的Tensor，Shape为expandedXOut的shape去掉最后一维之后所有维度的乘积，数据类型支持float32，[数据格式](common/数据格式.md)要求为ND，不支持[非连续的Tensor](common/非连续的Tensor.md)。
    -   workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
    -   executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

-   **返回值**

    返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。
    ```
    第一段接口完成入参校验，出现以下场景时报错：
    161001(ACLNN_ERR_PARAM_NULLPTR)：1. 计算输入和必选计算输出是空指针
    161002(ACLNN_ERR_PARAM_INVALID)：1. 计算输入和输出的数据类型和格式不在支持的范围内
    561002(ACLNN_ERR_INNER_TILING_ERROR): 1. x和expertIdx的shape维度不等于2,且第一维不相等
                                          2. activeNum、expertNum、expertCapacity的值小于0
                                          3. dropPadMode、expertTokensCountOrCumsumFlag、expertTokensBeforeCapacityFlag、quantMode不在取值范围内
                                          4. dropPadMode等于1时，expertCapacity和expertNum等于0
                                          5. expertTokensCountOrCumsumOut需要输出时，expertNum等于0
    ```

## aclnnMoeInitRoutingQuantV2

-   **参数说明：**
    -   workspace（void\*，入参）：在Device侧申请的workspace内存地址。
    -   workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnMoeInitRoutingQuantV2GetWorkspaceSize获取。
    -   executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
    -   stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明

无。

## 算子原型

```c++
REG_OP(MoeInitRoutingQuantV2)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(expert_idx, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
    .OUTPUT(expanded_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(expanded_row_idx, TensorType({DT_INT32}))
    .OUTPUT(expert_tokens_count_or_cumsum, TensorType({DT_INT32}))
    .OUTPUT(expert_tokens_before_capacity, TensorType({DT_INT32}))
    .OUTPUT(expert_tokens_before_capacity, TensorType({DT_INT32}))
    .OUTPUT(dynamicQuantScale, TensorType({DT_FLOAT}))
    .ATTR(active_num, Int, 0)
    .ATTR(expert_capacity, Int, 0)
    .ATTR(expert_num, Int, 0)
    .ATTR(drop_pad_mode, Int, 0)
    .ATTR(expert_tokens_count_or_cumsum_flag, Int, 0)
    .ATTR(expert_tokens_before_capacity_flag, Bool, false)
    .ATTR(quantMode, Int, 0)
    .OP_END_FACTORY_REG(MoeInitRoutingQuantV2)
```

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_init_routing_quant_v2.h"
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
    // 1. 固定写法，device/stream初始化, 参考acl对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口定义构造
    std::vector<int64_t> xShape = {3, 4};
    std::vector<int64_t> idxShape = {3, 2};
    std::vector<int64_t> scaleShape = {1};
    std::vector<int64_t> expandedXOutShape = {3, 2, 4};
    std::vector<int64_t> idxOutShape = {6};
    std::vector<int64_t> expertTokenOutShape = {3};
    std::vector<int64_t> dynamicQuantScaleOutShape = {6};
    void* xDeviceAddr = nullptr;
    void* expertIdxDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;
    void* offsetDeviceAddr = nullptr;
    void* expandedXOutDeviceAddr = nullptr;
    void* expandedRowIdxOutDeviceAddr = nullptr;
    void* expertTokenBeforeCapacityOutDeviceAddr = nullptr;
    void* dynamicQuantScaleOutDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* expertIdx = nullptr;
    aclTensor* scale = nullptr;
    aclTensor* offset = nullptr;
    int64_t activeNum = 0;
    int64_t expertCapacity = 2;
    int64_t expertNum = 3;
    int64_t dropPadMode = 1;
    int64_t expertTokensCountOrCumsumFlag = 0;
    bool expertTokensBeforeCapacityFlag = true;
    int64_t quantMode = 0;
    aclTensor* expandedXOut = nullptr;
    aclTensor* expandedRowIdxOut = nullptr;
    aclTensor* expertTokensBeforeCapacityOut = nullptr;
    aclTensor* dynamicQuantScaleOut = nullptr;
    std::vector<float> xHostData = {0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3};
    std::vector<int> expertIdxHostData = {1, 2, 0, 1, 0, 2};
    std::vector<float> scaleHostData = {0.3452};
    std::vector<float> offsetHostData = {1.8369};
    std::vector<int8_t> expandedXOutHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int> expandedRowIdxOutHostData = {0, 0, 0, 0, 0, 0};
    std::vector<int> expertTokensBeforeCapacityOutHostData = {0, 0, 0};
    std::vector<float> dynamicQuantScaleOutHostData = {0, 0, 0, 0, 0, 0};
    // 创建self aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertIdxHostData, idxShape, &expertIdxDeviceAddr, aclDataType::ACL_INT32, &expertIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(offsetHostData, scaleShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(expandedXOutHostData, expandedXOutShape, &expandedXOutDeviceAddr, aclDataType::ACL_INT8, &expandedXOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedRowIdxOutHostData, idxOutShape, &expandedRowIdxOutDeviceAddr, aclDataType::ACL_INT32, &expandedRowIdxOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertTokensBeforeCapacityOutHostData, expertTokenOutShape, &expertTokenBeforeCapacityOutDeviceAddr, aclDataType::ACL_INT32, &expertTokensBeforeCapacityOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dynamicQuantScaleOutHostData, dynamicQuantScaleOutShape, &dynamicQuantScaleOutDeviceAddr, aclDataType::ACL_FLOAT, &dynamicQuantScaleOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnMoeInitRoutingQuantV2第一段接口
    ret = aclnnMoeInitRoutingQuantV2GetWorkspaceSize(x, expertIdx, scale, offset, activeNum, expertCapacity, expertNum, dropPadMode, expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag, quantMode, expandedXOut, expandedRowIdxOut, nullptr, expertTokensBeforeCapacityOut, dynamicQuantScaleOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingQuantV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnMoeInitRoutingQuantV2第二段接口
    ret = aclnnMoeInitRoutingQuantV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingQuantV2 failed. ERROR: %d\n", ret); return ret);
    // 4. 固定写法，同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto expandedXSize = GetShapeSize(expandedXOutShape);
    std::vector<int8_t> expandedXData(expandedXSize, 0);
    ret = aclrtMemcpy(expandedXData.data(), expandedXData.size() * sizeof(expandedXData[0]), expandedXOutDeviceAddr, expandedXSize * sizeof(int8_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedXSize; i++) {
        LOG_PRINT("expandedXData[%ld] is: %d\n", i, expandedXData[i]);
    }
    auto expandedRowIdxSize = GetShapeSize(idxOutShape);
    std::vector<int> expandedRowIdxData(expandedRowIdxSize, 0);
    ret = aclrtMemcpy(expandedRowIdxData.data(), expandedRowIdxData.size() * sizeof(expandedRowIdxData[0]), expandedRowIdxOutDeviceAddr, expandedRowIdxSize * sizeof(int32_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedRowIdxSize; i++) {
        LOG_PRINT("expandedRowIdxData[%ld] is: %d\n", i, expandedRowIdxData[i]);
    }
    auto expertTokensBeforeCapacitySize = GetShapeSize(expertTokenOutShape);
    std::vector<int> expertTokenIdxData(expertTokensBeforeCapacitySize, 0);
    ret = aclrtMemcpy(expertTokenIdxData.data(), expertTokenIdxData.size() * sizeof(expertTokenIdxData[0]), expertTokenBeforeCapacityOutDeviceAddr, expertTokensBeforeCapacitySize * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expertTokensBeforeCapacitySize; i++) {
        LOG_PRINT("expertTokenIdxData[%ld] is: %d\n", i, expertTokenIdxData[i]);
    }

    auto dynamicQuantScaleSize = GetShapeSize(dynamicQuantScaleOutShape);
    std::vector<float> dynamicQuantScaleData(dynamicQuantScaleSize, 0);
    ret = aclrtMemcpy(dynamicQuantScaleData.data(), dynamicQuantScaleData.size() * sizeof(dynamicQuantScaleData[0]), dynamicQuantScaleOutDeviceAddr, dynamicQuantScaleSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < dynamicQuantScaleSize; i++) {
        LOG_PRINT("dynamicQuantScaleData[%ld] is: %f\n", i, dynamicQuantScaleData[i]);
    }
    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(expertIdx);
    aclDestroyTensor(scale);
    aclDestroyTensor(offset);
    aclDestroyTensor(expandedXOut);
    aclDestroyTensor(expandedRowIdxOut);
    aclDestroyTensor(expertTokensBeforeCapacityOut);
    aclDestroyTensor(dynamicQuantScaleOut);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(xDeviceAddr);
    aclrtFree(expertIdxDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    aclrtFree(offsetDeviceAddr);
    aclrtFree(expandedXOutDeviceAddr);
    aclrtFree(expandedRowIdxOutDeviceAddr);
    aclrtFree(expertTokenBeforeCapacityOutDeviceAddr);
    aclrtFree(dynamicQuantScaleOutDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```

