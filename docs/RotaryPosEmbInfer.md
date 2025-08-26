声明：本文使用[Creative Commons License version 4.0](https://gitee.com/link?target=https%3A%2F%2Fcreativecommons.org%2Flicenses%2Fby%2F4.0%2Flegalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# RotaryPosEmbInfer

## 支持的产品型号

- Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件
- Atlas 推理系列加速卡产品

产品形态详细说明请参见[昇腾产品形态说明](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fdocument%2Fredirect%2FCannCommunityProductForm)。

## 功能说明

- **算子功能**：推理网络为了提升性能，将query和key两路算子融合成一路。执行旋转位置编码计算，计算结果执行原地更新。

- **计算公式**：其中 x∈{q,k}x∈{q,k}

- 当rotaryCoeff == headim (cos的最后一维)时：

  x_even=x[...,::2]x_odd=x[...,1::2]x_rotate=torch.stack((−x_odd,x_even),dim=−1)x_rotate=rearrange(x_rotate,′...dtwo−>...(dtwo)′,two=2)x_embed=(x∗cos)+x_rotate∗sinx_even=x[...,::2]x_odd=x[...,1::2]x_rotate=torch.stack((−x_odd,x_even),dim=−1)x_rotate=rearrange(x_rotate,′...dtwo−>...(dtwo)′,two=2)x_embed=(x∗cos)+x_rotate∗sin

  

- 当rotaryCoeff == 2时：

  x_1=x[...,:x.shape[−1]//2]x_2=x[...,:x.shape[−1]//2:]x_rotate=torch.cat((−x_2,x_1),dim=−1)x_embed=(x∗cos)+x_rotate∗sinx_1=x[...,:x.shape[−1]//2]x_2=x[...,:x.shape[−1]//2:]x_rotate=torch.cat((−x_2,x_1),dim=−1)x_embed=(x∗cos)+x_rotate∗sin

  

## 算子执行接口

每个算子分为[两段式接口](https://gitee.com/ascend/cann-ops-adv/pulls/common/两段式接口.md)，必须先调用“aclnnRotaryPosEmbInferGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnRotaryPosEmbInfer”接口执行计算。

- `aclnnStatus aclnnRotaryPosEmbInferGetWorkspaceSize (aclTensor *q, aclTensor *k, const aclTensor *cos, const aclTensor *sin, const aclTensor *seqlen, int64_t rotaryCoeff, int64_t cosForamt, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnRotaryPosEmbInfer(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

**说明：**

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://gitee.com/link?target=https%3A%2F%2Fhiascend.com%2Fdocument%2Fredirect%2FCannCommunityCppOpcall)章节。

## aclnnRotaryPosEmbInferGetWorkspaceSize

- **参数说明：**

  - q（aclTensor*，计算输入）：表示要执行旋转位置编码的第一个张量，公式中的`q`，Device侧的aclTensor。支持[非连续的Tensor](https://gitee.com/ascend/cann-ops-adv/pulls/common/非连续的Tensor.md)，[数据格式](https://gitee.com/ascend/cann-ops-adv/pulls/common/数据格式.md)支持ND，维度为2维。
  - k（aclTensor*，计算输入）：表示要执行旋转位置编码的第二个张量，公式中的`k`，Device侧的aclTensor。支持[非连续的Tensor](https://gitee.com/ascend/cann-ops-adv/pulls/common/非连续的Tensor.md)，[数据格式](https://gitee.com/ascend/cann-ops-adv/pulls/common/数据格式.md)支持ND，维度为2维。
  - cos（aclTensor*，计算输入）：表示参与计算的位置编码张量，公式中的`cos`，Device侧的aclTensor。支持[非连续的Tensor](https://gitee.com/ascend/cann-ops-adv/pulls/common/非连续的Tensor.md)，[数据格式](https://gitee.com/ascend/cann-ops-adv/pulls/common/数据格式.md)支持ND，维度为2维。
  - sin（aclTensor*，计算输入）：表示参与计算的位置编码张量，公式中的`sin`，Device侧的aclTensor，支持[非连续的Tensor](https://gitee.com/ascend/cann-ops-adv/pulls/common/非连续的Tensor.md)，[数据格式](https://gitee.com/ascend/cann-ops-adv/pulls/common/数据格式.md)支持ND，维度为2维。
  - seqlen（aclTensor*，计算输入）：表示每个位置有多少tokens，Device侧的aclTensor，支持[非连续的Tensor](https://gitee.com/ascend/cann-ops-adv/pulls/common/非连续的Tensor.md)，[数据格式](https://gitee.com/ascend/cann-ops-adv/pulls/common/数据格式.md)支持ND，维度为1维。dtype类型为int32
  - rotaryCoeff（int64_t，计算输入）：决定旋转拆分模式，数值为 2,4 或 等于headim (cos的最后一维)
  - cosFormat（int64_t，计算输入）：保留参数，不参与计算。数值为 0,1。
  - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus： 返回状态码，具体参见[aclnn返回码](https://gitee.com/ascend/cann-ops-adv/pulls/common/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：1.传入的q、k、cos、sin或seqlen是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1.传入的q、k、cos、sin或seqlen的数据类型、数据格式不在支持的范围内或shape不匹配。
  									 2.传入的rotaryCoeff参数不在支持范围内。
  									 3.传入的cosFormat参数不在支持范围内。
  ```

## aclnnRotaryPosEmbInfer

- **参数说明：**

  - workspace（void*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnRotaryPosEmbInferGetWorkspaceSize获取。
  - executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](https://gitee.com/ascend/cann-ops-adv/pulls/common/aclnn返回码.md)。

## 约束说明

- 输入张量q、k、cos、sin只支持2维的shape，seqlen只支持1维度
- 输入张量q、k、cos、sin 的dtype必须相同，且4个输入shape的前1维必须相等，cos和sin的shape第2维必须相等
- 如果rotaryCoeff==head_dim 要使用workspace去存错位差的数据。workspace分配的大小为：
  ntokens * hidden_size_q * sizeof(half)
- 不支持空tensor场景

## 算子原型

```
REG_OP(RotaryPosEmbInfer)
    .INPUT(q, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(k, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(cos, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(sin, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(seqlen, TensorType({DT_INT32, DT_UINT32}))
    .ATTR(rotaryCoeff, Int, 1)
    .ATTR(cosFormat, Int, 1)
    .OUTPUT(query,TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OUTPUT(key,TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OP_END_FACTORY_REG(RotaryPosEmbInfer)
```

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitee.com/ascend/cann-ops-adv/pulls/common/编译与运行样例.md)。

```
/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file main.cpp
 */
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <fcntl.h>

#include "acl/acl.h"
#include "aclnn_rotary_pos_emb_infer.h"

const int SUCCESS = 0;
const int FAILED = 1;

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

bool ReadFile(const std::string &filePath, size_t fileSize, void *buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("failed to get file %s", filePath.c_str());
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file", filePath.c_str());
        return false;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        ERROR_LOG("file size is 0");
        file.close();
        return false;
    }
    if (size > bufferSize) {
        ERROR_LOG("file size is larger than buffer size");
        file.close();
        return false;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    fileSize = size;
    file.close();
    return true;
}

bool WriteFile(const std::string &filePath, const void *buffer, size_t size)
{
    if (buffer == nullptr) {
        ERROR_LOG("Write file failed. buffer is nullptr");
        return false;
    }

    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
    if (fd < 0) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    auto writeSize = write(fd, buffer, size);
    (void) close(fd);
    if (writeSize != size) {
        ERROR_LOG("Write file Failed.");
        return false;
    }

    return true;
}

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    // 固定写法，acl初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return FAILED);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return FAILED);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return FAILED);

    return SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return FAILED);

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return FAILED);

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    return SUCCESS;
}

int main(int argc, char **argv)
{
    // 1. （固定写法）device/stream初始化, 参考acl对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return FAILED);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    // 构造输入
    int batch = 1;
    int headDim = 256;
    int hiddenSizeQ = 256;
    int hiddenSizeK = 256;

    std::vector<int64_t> qShape = {batch, hiddenSizeQ};
    std::vector<int64_t> kShape = {batch, hiddenSizeK};
    std::vector<int64_t> cosShape = {batch, headDim};
    std::vector<int64_t> sinShape = {batch, headDim};
    std::vector<int64_t> seqlenShape = {batch, 1};
    std::vector<int64_t> ropeQShape = {batch, hiddenSizeQ};
    std::vector<int64_t> ropeKShape = {batch, hiddenSizeK};

    void *qDeviceAddr = nullptr;
    void *kDeviceAddr = nullptr;
    void *cosDeviceAddr = nullptr;
    void *sinDeviceAddr = nullptr;
    void *seqlenDeviceAddr = nullptr;
    void *ropeQDeviceAddr = nullptr;
    void *ropeKDeviceAddr = nullptr;
        
    aclTensor *q = nullptr;
    aclTensor *k = nullptr;
    aclTensor *cos = nullptr;
    aclTensor *sin = nullptr;
    aclTensor *seqlen = nullptr;
    aclTensor *ropeQ = nullptr;
    aclTensor *ropeK = nullptr;

    size_t qShapeSize = qShape[0] * qShape[1];
    size_t kShapeSize = kShape[0] * kShape[1];
    size_t cosShapeSize = cosShape[0] * cosShape[1];
    size_t sinShapeSize = sinShape[0] * sinShape[1];
    size_t seqlenShapeSize = seqlenShape[0] * seqlenShape[1];
    size_t ropeQShapeSize = ropeQShape[0] * ropeQShape[1];
    size_t ropeKShapeSize = ropeKShape[0] * ropeKShape[1];

    std::vector<float> qHostData = {-0.2263 ,  0.282  ,  0.588  ,  0.138  , -0.06665, -0.5527 ,  0.3132 ,
        0.591  ,  0.2676 , -0.899  , -0.662  , -0.8867 ,  0.649  , -0.3171 ,
        0.4373 ,  0.9233 , -0.7246 ,  0.4438 , -0.722  , -0.7563 , -0.413  ,
        0.5815 ,  0.7744 , -0.5435 ,  0.2422 , -0.6357 , -0.957  ,  0.04874,
        -0.884  , -0.1632 ,  0.1587 , -0.2058 ,  0.9346 , -0.1755 , -0.8535 ,
        -0.1599 ,  0.9883 , -0.561  ,  0.879  ,  0.539  , -0.555  ,  0.2351 ,
        0.4207 ,  0.0179 ,  0.9253 ,  0.705  ,  0.743  , -0.5557 ,  0.7866 ,
        -0.1127 ,  0.618  , -0.543  ,  0.4927 ,  0.9365 , -0.4475 , -0.467  ,
        -0.3054 ,  0.807  ,  0.623  , -0.5835 ,  0.6074 , -0.1504 , -0.644  ,
        0.1307 , -0.05502,  0.87   , -0.1569 , -0.04315,  0.939  ,  0.3857 ,
        -0.01403,  0.6616 ,  0.2423 , -0.801  ,  0.3118 ,  0.9077 , -0.6187 ,
        -0.923  ,  0.272  ,  0.2361 , -0.10925,  0.4636 ,  0.694  , -0.9336 ,
        -0.8247 ,  0.6343 , -0.05515,  0.7456 , -0.854  , -0.9746 ,  0.3489 ,
        -0.864  , -0.07526, -0.764  ,  0.05063,  0.6123 , -0.9463 ,  0.5806 ,
        0.1224 ,  0.4963 ,  0.772  ,  0.1567 , -0.9937 ,  0.7485 , -0.9473 ,
        0.4397 ,  0.963  ,  0.6772 , -0.28   ,  0.2379 ,  0.5225 , -0.787  ,
        -0.847  , -0.1835 ,  0.6836 ,  0.04117,  0.774  ,  0.5933 , -0.4326 ,
        -0.804  , -0.702  , -0.8213 ,  0.815  , -0.1741 , -0.894  , -0.283  ,
        0.544  , -0.4019 ,  0.1757 , -0.757  ,  0.977  , -0.495  ,  0.705  ,
        -0.1721 ,  0.4402 , -0.0755 ,  0.1538 ,  0.4185 ,  0.3718 , -0.1796 ,
        -0.943  , -0.5054 ,  0.71   ,  0.6533 , -0.3682 ,  0.6284 ,  0.07495,
        -0.498  , -0.308  ,  0.401  ,  0.6973 ,  0.8184 , -0.4143 , -0.383  ,
        -0.865  ,  0.8237 ,  0.5825 , -0.4988 ,  0.651  , -0.413  , -0.01037,
        0.828  , -0.4753 , -0.10095,  0.7812 ,  0.5483 , -0.9194 , -0.6787 ,
        -0.269  ,  0.03516,  0.1418 ,  0.1849 ,  0.9565 , -0.5825 , -0.56   ,
        -0.528  , -0.509  ,  0.9194 , -0.04816,  0.4272 ,  0.693  ,  0.6084 ,
        -0.9966 ,  0.2688 , -0.966  ,  0.08026, -0.6504 ,  0.968  , -0.705  ,
        0.3713 , -0.8486 ,  0.114  ,  0.1611 ,  0.0687 ,  0.5747 ,  0.772  ,
        0.0981 ,  0.9434 ,  0.976  , -0.9106 ,  0.763  ,  0.1252 , -0.4824 ,
        -0.2219 ,  0.8647 ,  0.2944 ,  0.9746 , -0.838  ,  0.2166 , -0.6245 ,
        0.766  , -0.713  , -0.9    ,  0.7944 , -0.934  , -0.0377 ,  0.597  ,
        -0.1017 ,  0.75   , -0.2808 , -0.925  , -0.508  ,  0.875  ,  0.4697 ,
        0.546  ,  0.6963 ,  0.733  , -0.6465 , -0.04187, -0.553  , -0.1903 ,
        -0.9717 ,  0.348  ,  0.0746 ,  0.755  , -0.2505 ,  0.2173 ,  0.4185 ,
        -0.3733 ,  0.02762, -0.4246 , -0.0761 , -0.6855 ,  0.628  , -0.3167 ,
        0.653  , -0.3613 ,  0.2507 ,  0.6885 , -0.509  , -0.4338 ,  0.8916 ,
        0.9067 ,  0.958  , -0.924  ,  0.0642};
    std::vector<float> kHostData = {-0.7334 , -0.3223 ,  0.8696 ,  0.866  ,  0.0721 , -0.556  , -0.2502 ,
        -0.694  , -0.719  , -0.7827 ,  0.4453 , -0.6016 ,  0.3772 , -0.4626 ,
        0.119  ,  0.5454 ,  0.523  ,  0.4597 ,  0.507  , -0.10736, -0.0706 ,
        0.2961 , -0.323  ,  0.783  ,  0.06113,  0.5283 ,  0.602  , -0.7627 ,
        0.71   , -0.417  , -0.1873 ,  0.2015 ,  0.2145 , -0.276  ,  0.542  ,
        -0.986  , -0.5474 , -0.692  ,  0.952  ,  0.562  ,  0.7036 ,  0.379  ,
        -0.1478 ,  0.671  , -0.285  ,  0.73   , -0.5884 ,  0.452  ,  0.6333 ,
        -0.1236 , -0.9756 ,  0.8765 , -0.476  , -0.1313 , -0.5083 ,  0.408  ,
        -0.2408 ,  0.4333 ,  0.983  , -0.0586 ,  0.2708 , -0.7964 , -0.6987 ,
        -0.8696 ,  0.8447 , -0.8945 ,  0.7285 ,  0.322  ,  0.6733 ,  0.649  ,
        0.09985,  0.2812 , -0.4521 , -0.6323 ,  0.8335 , -0.831  ,  0.1194 ,
        0.592  ,  0.4495 ,  0.1206 , -0.4734 , -0.3381 ,  0.3765 ,  0.3025 ,
        -0.4097 , -0.03885,  0.232  ,  0.3909 , -0.5273 , -0.816  , -0.3865 ,
        -0.5605 ,  0.298  , -0.974  , -0.9434 ,  0.707  ,  0.825  ,  0.2489 ,
        -0.869  ,  0.9556 , -0.692  ,  0.6543 ,  0.827  ,  0.585  ,  0.4905 ,
        -0.9297 , -0.4407 ,  0.782  , -0.4927 ,  0.7505 ,  0.7485 ,  0.599  ,
        0.6553 ,  0.11487,  0.4958 ,  0.8965 , -0.7563 ,  0.5864 , -0.6343 ,
        -0.3267 ,  0.815  , -0.249  , -0.67   ,  0.522  ,  0.656  ,  0.1301 ,
        0.455  ,  0.441  , -0.169  , -0.381  ,  0.9634 , -0.486  ,  0.1511 ,
        -0.8926 , -0.10565,  0.7993 ,  0.06036, -0.8374 ,  0.813  , -0.703  ,
        -0.1802 , -0.8345 , -0.6924 , -0.3323 , -0.7197 , -0.3118 ,  0.2086 ,
        0.901  , -0.0591 ,  0.783  ,  0.0596 , -0.7837 , -0.3625 , -0.644  ,
        -0.6187 , -0.2054 , -0.03345, -0.318  , -0.4714 , -0.4065 , -0.362  ,
        0.3896 ,  0.3972 ,  0.949  ,  0.758  ,  0.5474 , -0.3438 , -0.2742 ,
        0.4485 ,  0.63   ,  0.2041 ,  0.8867 , -0.657  ,  0.8296 ,  0.441  ,
        0.2856 , -0.4253 , -0.3696 , -0.02412, -0.982  , -0.5356 ,  0.2047 ,
        0.7905 ,  0.627  , -0.4983 ,  0.6445 ,  0.9316 , -0.1954 ,  0.02481,
        0.08203,  0.913  , -0.866  ,  0.9937 ,  0.2408 ,  0.4077 , -0.0671 ,
        0.1877 , -0.88   ,  0.04657,  0.688  ,  0.2825 ,  0.3215 ,  0.3115 ,
        -0.9204 , -0.306  ,  0.392  , -0.11774, -0.684  ,  0.765  , -0.949  ,
        0.8594 , -0.1406 , -0.2295 , -0.3752 , -0.1464 ,  0.7124 ,  0.625  ,
        -0.8716 , -0.8086 , -0.2617 , -0.08875, -0.3105 , -0.1542 , -0.6797 ,
        0.2715 ,  0.2654 ,  0.771  ,  0.6284 , -0.683  , -0.6284 , -0.6143 ,
        0.1184 , -0.16   ,  0.399  ,  0.2368 , -0.775  , -0.8647 ,  0.7676 ,
        -0.846  , -0.1422 , -0.45   ,  0.712  , -0.3657 , -0.1667 , -0.3962 ,
        -0.233  ,  0.5557 , -0.665  , -0.482  , -0.737  ,  0.4265 ,  0.0352 ,
        0.8555 ,  0.4236 , -0.619  , -0.8433};
    std::vector<float> cosHostData = {0.2094  ,  0.2094  ,  0.8477  ,  0.8477  , -0.8467  , -0.8467  ,
        -0.2244  , -0.2244  , -0.2328  , -0.2328  , -0.8906  , -0.8906  ,
        0.0762  ,  0.0762  ,  0.645   ,  0.645   ,  0.3103  ,  0.3103  ,
        -0.7856  , -0.7856  ,  0.08093 ,  0.08093 , -0.6406  , -0.6406  ,
        -0.6924  , -0.6924  , -0.1904  , -0.1904  , -0.005634, -0.005634,
        -0.911   , -0.911   , -0.643   , -0.643   , -0.8794  , -0.8794  ,
        0.9937  ,  0.9937  , -0.7827  , -0.7827  , -0.543   , -0.543   ,
        -0.2261  , -0.2261  ,  0.851   ,  0.851   ,  0.2495  ,  0.2495  ,
        -0.956   , -0.956   , -0.01392 , -0.01392 , -0.2257  , -0.2257  ,
        0.8647  ,  0.8647  ,  0.5933  ,  0.5933  , -0.8413  , -0.8413  ,
        -0.9307  , -0.9307  ,  0.854   ,  0.854   , -0.6787  , -0.6787  ,
        0.0501  ,  0.0501  , -0.6826  , -0.6826  ,  0.857   ,  0.857   ,
        -0.859   , -0.859   , -0.7983  , -0.7983  ,  0.8564  ,  0.8564  ,
        0.1493  ,  0.1493  ,  0.05966 ,  0.05966 ,  0.6855  ,  0.6855  ,
        -0.667   , -0.667   , -0.1881  , -0.1881  , -0.28    , -0.28    ,
        -0.345   , -0.345   , -0.4136  , -0.4136  ,  0.2465  ,  0.2465  ,
        0.354   ,  0.354   ,  0.5913  ,  0.5913  ,  0.535   ,  0.535   ,
        0.21    ,  0.21    ,  0.236   ,  0.236   , -0.697   , -0.697   ,
        -0.553   , -0.553   ,  0.619   ,  0.619   ,  0.688   ,  0.688   ,
        0.168   ,  0.168   ,  0.003584,  0.003584,  0.857   ,  0.857   ,
        -0.3696  , -0.3696  , -0.1498  , -0.1498  , -0.6567  , -0.6567  ,
        0.4026  ,  0.4026  ,  0.7773  ,  0.7773  ,  0.605   ,  0.605   ,
        0.5737  ,  0.5737  ,  0.462   ,  0.462   , -0.45    , -0.45    ,
        -0.286   , -0.286   ,  0.2935  ,  0.2935  ,  0.6704  ,  0.6704  ,
        -0.77    , -0.77    , -0.4368  , -0.4368  , -0.5303  , -0.5303  ,
        -0.026   , -0.026   ,  0.634   ,  0.634   ,  0.4995  ,  0.4995  ,
        -0.5425  , -0.5425  ,  0.4338  ,  0.4338  ,  0.1971  ,  0.1971  ,
        -0.846   , -0.846   ,  0.1365  ,  0.1365  , -0.0839  , -0.0839  ,
        0.726   ,  0.726   ,  0.563   ,  0.563   , -0.9297  , -0.9297  ,
        -0.1465  , -0.1465  ,  0.839   ,  0.839   , -0.2317  , -0.2317  ,
        0.10345 ,  0.10345 , -0.16    , -0.16    , -0.973   , -0.973   ,
        0.8105  ,  0.8105  , -0.5933  , -0.5933  ,  0.2422  ,  0.2422  ,
        0.7676  ,  0.7676  ,  0.696   ,  0.696   ,  0.467   ,  0.467   ,
        -0.955   , -0.955   ,  0.864   ,  0.864   ,  0.9736  ,  0.9736  ,
        0.2979  ,  0.2979  ,  0.823   ,  0.823   ,  0.628   ,  0.628   ,
        -0.8247  , -0.8247  ,  0.1696  ,  0.1696  , -0.3638  , -0.3638  ,
        0.11084 ,  0.11084 , -0.2438  , -0.2438  ,  0.3772  ,  0.3772  ,
        0.9883  ,  0.9883  , -0.3958  , -0.3958  , -0.0846  , -0.0846  ,
        0.972   ,  0.972   ,  0.4468  ,  0.4468  , -0.343   , -0.343   ,
        0.589   ,  0.589   , -0.655   , -0.655   , -0.4558  , -0.4558  ,
        -0.466   , -0.466   , -0.0411  , -0.0411  ,  0.649   ,  0.649   ,
        -0.709   , -0.709   , -0.04984 , -0.04984 ,  0.6377  ,  0.6377  ,
        0.0187  ,  0.0187  ,  0.944   ,  0.944 };
    std::vector<float> sinHostData = {0.402  ,  0.402  ,  0.5693 ,  0.5693 , -0.8555 , -0.8555 , -0.19   ,
        -0.19   , -0.736  , -0.736  , -0.2573 , -0.2573 ,  0.812  ,  0.812  ,
        -0.4346 , -0.4346 ,  0.7773 ,  0.7773 , -0.4585 , -0.4585 , -0.783  ,
        -0.783  ,  0.3442 ,  0.3442 , -0.566  , -0.566  ,  0.2148 ,  0.2148 ,
        -0.1804 , -0.1804 , -0.8677 , -0.8677 ,  0.10565,  0.10565, -0.781  ,
        -0.781  , -0.8677 , -0.8677 , -0.8916 , -0.8916 ,  0.499  ,  0.499  ,
        0.00551,  0.00551,  0.1846 ,  0.1846 ,  0.2375 ,  0.2375 , -0.726  ,
        -0.726  , -0.09174, -0.09174,  0.3552 ,  0.3552 ,  0.1721 ,  0.1721 ,
        -0.6406 , -0.6406 , -0.01968, -0.01968,  0.2527 ,  0.2527 ,  0.7544 ,
        0.7544 ,  0.621  ,  0.621  , -0.9683 , -0.9683 , -0.3726 , -0.3726 ,
        -0.3347 , -0.3347 , -0.9087 , -0.9087 ,  0.889  ,  0.889  ,  0.511  ,
        0.511  , -0.3413 , -0.3413 ,  0.4236 ,  0.4236 , -0.9136 , -0.9136 ,
        -0.995  , -0.995  ,  0.6216 ,  0.6216 , -0.709  , -0.709  ,  0.511  ,
        0.511  ,  0.415  ,  0.415  ,  0.2445 ,  0.2445 ,  0.4639 ,  0.4639 ,
        0.1278 ,  0.1278 , -0.9126 , -0.9126 ,  0.8013 ,  0.8013 ,  0.4417 ,
        0.4417 , -0.1209 , -0.1209 ,  0.0367 ,  0.0367 ,  0.731  ,  0.731  ,
        -0.1754 , -0.1754 ,  0.782  ,  0.782  , -0.63   , -0.63   , -0.5127 ,
        -0.5127 , -0.4316 , -0.4316 , -0.649  , -0.649  , -0.93   , -0.93   ,
        -0.1869 , -0.1869 , -0.97   , -0.97   , -0.01024, -0.01024,  0.8003 ,
        0.8003 ,  0.9277 ,  0.9277 ,  0.1605 ,  0.1605 ,  0.8735 ,  0.8735 ,
        -0.3806 , -0.3806 , -0.171  , -0.171  ,  0.2825 ,  0.2825 ,  0.9795 ,
        0.9795 , -0.5    , -0.5    ,  0.3936 ,  0.3936 , -0.1199 , -0.1199 ,
        -0.8906 , -0.8906 ,  0.5933 ,  0.5933 , -0.361  , -0.361  , -0.1172 ,
        -0.1172 , -0.2795 , -0.2795 ,  0.2827 ,  0.2827 ,  0.487  ,  0.487  ,
        -0.01736, -0.01736, -0.958  , -0.958  ,  0.91   ,  0.91   , -0.1652 ,
        -0.1652 , -0.917  , -0.917  ,  0.671  ,  0.671  ,  0.7393 ,  0.7393 ,
        -0.137  , -0.137  , -0.3672 , -0.3672 , -0.0813 , -0.0813 ,  0.822  ,
        0.822  , -0.3157 , -0.3157 ,  0.0429 ,  0.0429 ,  0.05914,  0.05914,
        0.8413 ,  0.8413 , -0.562  , -0.562  ,  0.583  ,  0.583  ,  0.226  ,
        0.226  , -0.8994 , -0.8994 , -0.51   , -0.51   , -0.952  , -0.952  ,
        0.3708 ,  0.3708 ,  0.644  ,  0.644  ,  0.8926 ,  0.8926 ,  0.904  ,
        0.904  ,  0.1405 ,  0.1405 ,  0.227  ,  0.227  , -0.4944 , -0.4944 ,
        -0.723  , -0.723  , -0.0702 , -0.0702 , -0.85   , -0.85   ,  0.423  ,
        0.423  ,  0.1385 ,  0.1385 ,  0.606  ,  0.606  ,  0.4053 ,  0.4053 ,
        0.587  ,  0.587  , -0.864  , -0.864  ,  0.985  ,  0.985  , -0.8047 ,
        -0.8047 ,  0.6826 ,  0.6826 ,  0.3652 ,  0.3652 ,  0.0569 ,  0.0569 ,
        -0.4944 , -0.4944 , -0.766  , -0.766};
    std::vector<int32_t> seqlenHostData = {1};
    std::vector<float> ropeQHostData(ropeQShapeSize);
    std::vector<float> ropeKHostData(ropeKShapeSize);
 
    // 创建输入输出Tensor
    // 创建 q aclTensor
    ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);
    // 创建 k aclTensor
    ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);
    // 创建 cos aclTensor
    ret = CreateAclTensor(cosHostData, cosShape, &cosDeviceAddr, aclDataType::ACL_FLOAT16, &cos);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);
    // 创建 sin aclTensor
    ret = CreateAclTensor(sinHostData, sinShape, &sinDeviceAddr, aclDataType::ACL_FLOAT16, &sin);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);
    // 创建 seqlen aclTensor
    ret = CreateAclTensor(seqlenHostData, seqlenShape, &seqlenDeviceAddr, aclDataType::ACL_INT32, &seqlen);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);
    // 创建 ropeQ aclTensor
    ret = CreateAclTensor(ropeQHostData, ropeQShape, &ropeQDeviceAddr, aclDataType::ACL_FLOAT16, &ropeQ);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);
    // 创建 ropeK aclTensor
    ret = CreateAclTensor(ropeKHostData, ropeKShape, &ropeKDeviceAddr, aclDataType::ACL_FLOAT16, &ropeK);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);

    // 其他属性
    int64_t rotaryCoeff = 2;
    int64_t cosForamt = 0;
    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnRotaryPosEmbInfer第一段接口
    ret = aclnnRotaryPosEmbInferGetWorkspaceSize(q, k, cos, sin, seqlen, rotaryCoeff, cosForamt,
        ropeQ, ropeK, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRotaryPosEmbInferGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnStridedSliceAssignV2第二段接口
    ret = aclnnRotaryPosEmbInfer(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRotaryPosEmbInfer failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(ropeQShape);
    std::vector<float> ropeQData(size, 0);
    ret = aclrtMemcpy(ropeQData.data(), ropeQData.size() * sizeof(ropeQData[0]), ropeQDeviceAddr,
                      size * sizeof(ropeQData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy out result from device to host failed. ERROR: %d\n", ret);
        return ret);
  
    size = GetShapeSize(ropeKShape);
    std::vector<float> ropeKData(size, 0);
    ret = aclrtMemcpy(ropeKData.data(), ropeKData.size() * sizeof(ropeKData[0]), ropeKDeviceAddr,
                      size * sizeof(ropeKData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy out result from device to host failed. ERROR: %d\n", ret);
        return ret);
    //打印数据
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, ropeQData[i]);
    }
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, ropeKData[i]);
    }

     // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(q);
    aclDestroyTensor(k);
    aclDestroyTensor(cos);
    aclDestroyTensor(sin);
    aclDestroyTensor(seqlen);
    aclDestroyTensor(ropeQ);
    aclDestroyTensor(ropeK);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(qDeviceAddr);
    aclrtFree(kDeviceAddr);
    aclrtFree(cosDeviceAddr);
    aclrtFree(sinDeviceAddr);
    aclrtFree(seqlenDeviceAddr);
    aclrtFree(ropeQDeviceAddr);
    aclrtFree(ropeKDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return SUCCESS;
}
```