声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# IncreFlashAttentionV4

## 支持的产品型号

- Atlas A2 训练系列产品/Atlas 800I A2 推理产品
- Atlas 推理系列加速卡产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

-   算子功能：兼容[IncreFlashAttentionV3](IncreFlashAttentionV3.md)接口功能，在其基础上**新增kv左Padding特性**。

    对于自回归（Auto-regressive）的语言模型，随着新词的生成，推理输入长度不断增大。在原来全量推理的基础上**实现增量推理**，query的S轴固定为1，key和value是经过KV Cache后，将之前推理过的state信息，叠加在一起，每个Batch对应S轴的实际长度可能不一样，输入的数据是经过padding后的固定长度数据。

    相比全量场景的FlashAttention算子（[PromptFlashAttention](PromptFlashAttention.md)），增量推理的流程与正常全量推理并不完全等价，不过增量推理的精度并无明显劣化。

    **说明：** 
KV Cache是大模型推理性能优化的一个常用技术。采样时，Transformer模型会以给定的prompt/context作为初始输入进行推理（可以并行处理），随后逐一生成额外的token来继续完善生成的序列（体现了模型的自回归性质）。在采样过程中，Transformer会执行自注意力操作，为此需要给当前序列中的每个项目（无论是prompt/context还是生成的token）提取键值（KV）向量。这些向量存储在一个矩阵中，通常被称为kv缓存（KV Cache）。
    
-   计算公式：

    self-attention（自注意力）利用输入样本自身的关系构建了一种注意力模型。其原理是假设有一个长度为$n$的输入样本序列$x$，$x$的每个元素都是一个$d$维向量，可以将每个$d$维向量看作一个token embedding，将这样一条序列经过3个权重矩阵变换得到3个维度为$n*d$的矩阵。

    self-attention的计算公式一般定义如下，其中$Q、K、V$为输入样本的重要属性元素，是输入样本经过空间变换得到，且可以统一到一个特征空间中。

    $$
    Attention(Q,K,V)=Score(Q,K)V
    $$

    本算子中Score函数采用Softmax函数，self-attention计算公式为:

    $$
    Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
    $$

    其中$Q$和$K^T$的乘积代表输入$x$的注意力，为避免该值变得过大，通常除以$d$的开根号进行缩放，并对每行进行softmax归一化，与$V$相乘后得到一个$n*d$的矩阵。


## 实现原理

详细实现原理参考[IFA设计](./common/IFA算子设计介绍.md)。

## 算子执行接口

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnIncreFlashAttentionV4GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnIncreFlashAttentionV4”接口执行计算。

* `aclnnStatus aclnnIncreFlashAttentionV4GetWorkspaceSize(const aclTensor *query, const aclTensorList *key, const aclTensorList *value, const aclTensor *pseShift, const aclTensor *attenMask, const aclIntArray *actualSeqLengths,  const aclTensor *dequantScale1, const aclTensor *quantScale1, const aclTensor *dequantScale2, const aclTensor *quantScale2, const aclTensor *quantOffset2, const aclTensor *antiquantScale, const aclTensor *antiquantOffset, const aclTensor *blocktable, const aclTensor *kvPaddingSize, int64_t numHeads, double scaleValue, char *inputLayout, int64_t numKeyValueHeads, int64_t blockSize, int64_t innerPrecise, const aclTensor *attentionOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnstatus aclnnIncreFlashAttentionV4(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnIncreFlashAttentionV4GetWorkspaceSize

- **参数说明：**
  - query（aclTensor\*，计算输入）：Device侧的aclTensor，公式中的输入Q，[数据格式](common/数据格式.md)支持ND。
    - Atlas A2 训练系列产品/Atlas 800I A2 推理产品：数据类型支持FLOAT16、BFLOAT16
    - Atlas 推理系列加速卡产品：数据类型仅支持FLOAT16

  - key（aclTensorList\*，计算输入）：Device侧的aclTensorList，公式中的输入K，[数据格式](common/数据格式.md)支持ND。
    - Atlas A2 训练系列产品/Atlas 800I A2 推理产品：数据类型支持FLOAT16、INT8、BFLOAT16
    - Atlas 推理系列加速卡产品：数据类型仅支持FLOAT16

  - value（aclTensorList\*，计算输入）：Device侧的aclTensorList，公式中的输入V，[数据格式](common/数据格式.md)支持ND。
    - Atlas A2 训练系列产品/Atlas 800I A2 推理产品：数据类型支持FLOAT16、INT8、BFLOAT16
    - Atlas 推理系列加速卡产品：数据类型仅支持FLOAT16

  - pseShift（aclTensor\*，计算输入）：Device侧的aclTensor，可选参数，位置编码参数，数据类型支持FLOAT16、BFLOAT16，[数据格式](common/数据格式.md)支持ND，输入shape为(1,N,1,S)或(B,N,1,S)。如不使用该功能时可传入nullptr。
  - attenMask（aclTensor\*，计算输入）：Device侧的aclTensor，可选参数，表示attention掩码矩阵，数据类型支持BOOL、INT8、UINT8，[数据格式](common/数据格式.md)支持ND。
  - actualSeqLengths（aclIntArray\*，计算输入）：Host侧的aclIntArray，可选参数，表示key和value的S轴实际长度，数据类型支持INT64。
  - dequantScale1（aclTensor\*，计算输入）：Device侧的aclTensor，[数据格式](common/数据格式.md)支持ND，表示BMM1后面反量化的量化因子，支持per-tensor（scalar）。 如不使用该参数可传入nullptr。
    - Atlas A2 训练系列产品/Atlas 800I A2 推理产品：数据类型支持UINT64、FLOAT32
    - Atlas 推理系列加速卡产品：仅支持nullptr

  - quantScale1（aclTensor\*，计算输入）：Device侧的aclTensor，[数据格式](common/数据格式.md)支持ND，表示BMM2前面量化的量化因子，支持per-tensor（scalar）。 如不使用该参数可传入nullptr。
    - Atlas A2 训练系列产品/Atlas 800I A2 推理产品：数据类型支持FLOAT32
    - Atlas 推理系列加速卡产品：仅支持nullptr

  - dequantScale2（aclTensor\*，计算输入）：Device侧的aclTensor，[数据格式](common/数据格式.md)支持ND，表示BMM2后面量化的量化因子，支持per-tensor（scalar）。 如不使用该参数可传入nullptr。
    - Atlas A2 训练系列产品/Atlas 800I A2 推理产品：数据类型支持UINT64、FLOAT32
    - Atlas 推理系列加速卡产品：仅支持nullptr

  - quantScale2（aclTensor\*，计算输入）：Device侧的aclTensor，[数据格式](common/数据格式.md)支持ND，表示输出量化的量化因子，支持per-tensor，per-channel。 如不使用该参数可传入nullptr。
    - Atlas A2 训练系列产品/Atlas 800I A2 推理产品：数据类型支持FLOAT32、BFLOAT16
    - Atlas 推理系列加速卡产品：仅支持nullptr

  - quantOffset2（aclTensor\*，计算输入）：Device侧的aclTensor，[数据格式](common/数据格式.md)支持ND，表示输出量化的量化偏移，支持per-tensor，per-channel。 如不使用该参数可传入nullptr。
    - Atlas A2 训练系列产品/Atlas 800I A2 推理产品：数据类型支持FLOAT32、BFLOAT16
    - Atlas 推理系列加速卡产品：仅支持nullptr
  - antiquantScale（aclTensor\*，计算输入）：Device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16、FLOAT32，[数据格式](common/数据格式.md)支持ND，表示量化因子，支持per-tensor，per-channel，per-token。综合约束请见**约束与限制**。如不使用该功能时可传入nullptr。
  - antiquantOffset（aclTensor\*，计算输入）：Device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16、FLOAT32 。[数据格式](common/数据格式.md)支持ND，表示量化偏移，支持per-tensor，per-channel，per-token。综合约束请见**约束与限制**。如不使用该功能时可传入nullptr。
  - blocktable（aclTensor\*，计算输入）：Device侧的aclTensor，数据类型支持INT32，[数据格式](common/数据格式.md)支持ND，表示page attention中KV存储使用的block映射表。 如不使用该功能时可传入nullptr。
  - kvPaddingSize（aclTensor\*，计算输入）：Device侧的aclTensor，数据类型支持INT64，[数据格式](common/数据格式.md)支持ND，表示kv左padding场景。 如不使用该功能时可传入nullptr。
  - numHeads（int64\_t，计算输入 ）：Host侧的int64\_t，代表head个数，数据类型支持INT64。
  - scaleValue（double，计算输入）：Host侧的double，公式中d开根号的倒数，代表缩放系数，作为计算流中Muls的scalar值，数据类型支持DOUBLE。
  - inputLayout（char\*，计算输入）：Host侧的字符指针，用于标识输入query、key、value的数据排布格式，当前支持BSH、BNSD、BSND。用户不特意指定时可传入默认值"BSH"。

    **说明：** 
    query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Head-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N。

  - numKeyValueHeads（int64\_t，计算输入 ）：Host侧的int64\_t，代表key、value中head个数，用于支持GQA（Grouped-Query Attention，分组查询注意力）场景，默认为0，表示和query的head个数相等；numHeads与numKeyValueHeads的比值不能大于64。
    - Atlas A2 训练系列产品/Atlas 800I A2 推理产品：数据类型支持INT64
    - Atlas 推理系列加速卡产品：仅支持默认值0
  - blockSize （int64\_t，计算输入）：Host侧的int64\_t，page attention中KV存储每个block中最大的token个数，默认为0，数据类型支持INT64。
  - innerPrecise （int64\_t，计算输入）：Host侧的int64\_t，代表高精度/高性能选择，0代表高精度，1代表高性能，默认值为1， 数据类型支持INT64，当前仅支持高精度和高性能两种模式。
  - attentionOut（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出，[数据格式](common/数据格式.md)支持ND。
    - Atlas A2 训练系列产品/Atlas 800I A2 推理产品：数据类型支持FLOAT16、INT8、BFLOAT16
    - Atlas 推理系列加速卡产品：数据类型支持FLOAT16
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：query、key、value、pseShift、attenMask、attentionOut的数据类型和数据格式不在支持的范围内。
  - 返回361001（ACLNN_ERR_RUNTIME_ERROR）：API内存调用npu runtime的接口异常。
  ```

### aclnnIncreFlashAttentionV4

-   **参数说明：**
    -   workspace（void\*，入参）：在Device侧申请的workspace内存地址。
    -   workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnIncreFlashAttentionV4GetWorkspaceSize获取。
    -   executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
    -   stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束与限制

-   参数key、value 中对应tensor的shape需要完全一致。
-   参数query和attentionOut的shape需要完全一致。
-   参数query中的N和numHeads值相等，key、value的N和numKeyValueHeads值相等，并且numHeads是numKeyValueHeads的倍数关系。
-   非连续场景下，参数key、value的tensorlist中tensor的个数等于query的B(由于tensorlist限制, 非连续场景下B需要小于等于256)。shape除S外需要完全一致，且batch只能为1。
-   query，key，value输入，功能使用限制如下：
    -   Atlas A2 训练系列产品/Atlas 800I A2 推理产品：
        - 支持B轴小于等于65536；
        - 支持N轴小于等于256；
        - 支持D轴小于等于512；
    -   Atlas 推理系列加速卡产品：
        - 支持B轴小于等于256；
        - 支持N轴小于等于256；
        - 支持D轴小于等于512；
        - 支持key、value的S轴小于等于65536；
    -   query、key、value输入均为INT8的场景暂不支持。
    -   仅支持query的S轴等于1。
-   INT8量化相关入参数量与输入、输出数据格式的综合限制：
    - query、key、value输入为FLOAT16，输出为INT8的场景：入参quantScale2必填，quantOffset2可选，不能传入dequantScale1、quantScale1、dequantScale2（即为nullptr）参数。
-   pseShift功能使用限制如下：
    - pseShift数据类型需与query数据类型保持一致。
    - 仅支持D轴对齐，即D轴可以被16整除。
-   page attention场景:
    - page attention的使能必要条件是blockTable存在且有效，同时key、value是按照blockTable中的索引在一片连续内存中排布，支持key、value dtype为FLOAT16/BFLOAT16/INT8，在该场景下key、value的inputLayout参数无效。
    - blockSize是用户自定义的参数，该参数的取值会影响page attention的性能，在使能page attention场景下，blockSize需要传入非0值, 且blocksize最大不超过512。key、value输入类型为FLOAT16/BFLOAT16时需要16对齐，key、value 输入类型为INT8时需要32对齐，推荐使用128。通常情况下，page attention可以提高吞吐量，但会带来性能上的下降。
    -   page attention场景下，当query的inputLayout为BNSD时，kv cache排布支持（blocknum, blocksize, H）和（blocknum, KV_N, blocksize, D）两种格式，当query的inputLayout为BSH、BSND时，kv cache排布只支持（blocknum, blocksize, H）一种格式。blocknum不能小于根据actualSeqLengthsKv和blockSize计算的每个batch的block数量之和。且key和value的shape需保证一致。
    -   page attention场景下，kv cache排布为（blocknum, KV_N, blocksize, D）时性能通常优于kv cache排布为（blocknum, blocksize, H）时的性能，建议优先选择（blocknum, KV_N, blocksize, D）格式。
    -   page attention使能场景下，当输入kv cache排布格式为（blocknum, blocksize, H），且 numKvHeads * headDim 超过64k时，受硬件指令约束，会被拦截报错。可通过使能GQA（减小 numKvHeads）或调整kv cache排布格式为（blocknum, numKvHeads, blocksize, D）解决。
    -   page attention场景下，必须传入输入actualSeqLengths。
    -   page attention场景下，blockTable必须为二维，第一维长度需等于B，第二维长度不能小于maxBlockNumPerSeq（maxBlockNumPerSeq为每个batch中最大actualSeqLengthsKv对应的block数量）。
    -   page attention使能场景下，以下场景输入S需要大于等于maxBlockNumPerSeq * blockSize
      - 使能 Attention mask，例如 mask shape为 \(B, 1, 1, S\)
      - 使能 pseShift，例如 pseShift shape为\(B, N, 1, S\)
      - 使能伪量化 per-token模式：输入参数 antiquantScale和antiquantOffset 的shape均为\(2, B, S\)
- kv左padding场景:
    -   kvCache的搬运起点计算公式为：Smax - kvPaddingSize - actualSeqLengths；kvCache的搬运终点计算公式为：Smax - kvPaddingSize。其中kvCache的搬运起点或终点小于0时，返回数据结果为全0。
    -   kvPaddingSize小于0时将被置为0。
    -   需要与actualSeqLengths参数一起使能，否则默认为kv右padding场景。
    -   与attenMask参数一起使能时，需要保证attenMask含义正确，即能够正确的对无效数据进行隐藏。否则将引入精度问题。
-   antiquantScale和antiquantOffset参数约束：
    - 支持per-channel、per-tensor两种模式：
      - per-channel模式：两个参数的shape可支持\(2, N, 1, D\)，\(2, N, D\)，\(2, H\)，N为numKeyValueHeads。参数数据类型和query数据类型相同。
      - per-tensor模式：两个参数的shape均为(2)，数据类型和query数据类型相同。

## 算子原型

```c++
REG_OP(IncreFlashAttention)
    .INPUT(query, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .DYNAMIC_INPUT(key, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .DYNAMIC_INPUT(value, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OPTIONAL_INPUT(pse_shift, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_BOOL, DT_INT8, DT_UINT8}))
    .OPTIONAL_INPUT(actual_seq_lengths, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(dequant_scale1, TensorType({DT_UINT64, DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale1, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(dequant_scale2, TensorType({DT_UINT64, DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale2, TensorType({DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(quant_offset2, TensorType({DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(block_table, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(kv_padding_size, TensorType({DT_INT64}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .REQUIRED_ATTR(num_heads, Int)
    .ATTR(scale_value, Float, 1.0)
    .ATTR(input_layout, String, "BSH")
    .ATTR(num_key_value_heads, Int, 0)
    .ATTR(block_size, Int, 0)
    .ATTR(inner_precise, Int, 1)
    .OP_END_FACTORY_REG(IncreFlashAttention)
```
参数解释请参见**算子执行接口**。

## 调用示例

- PyTorch框架调用

  如果通过PyTorch单算子方式调用该融合算子，则需要参考PyTorch融合算子[torch_npu.npu_incre_flash_attention](https://hiascend.com/document/redirect/PyTorchAPI)；如果用户定制了该融合算子，则需要参考《Ascend C算子开发》手册[适配PyTorch框架](https://hiascend.com/document/redirect/CannCommunityAscendCInvorkOnNetwork)。

- aclnn单算子调用方式

  通过aclnn单算子调用示例代码如下（以Atlas A2 训练系列产品/Atlas 800I A2 推理产品为例），仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include <math.h>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_incre_flash_attention_v4.h"
 
using namespace std;
 
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
  int32_t batchSize = 1;
  int32_t numHeads = 2;
  int32_t headDims = 16;
  int32_t keyNumHeads = 2;
  int32_t sequenceLengthKV = 16;
  std::vector<int64_t> queryShape = {batchSize, numHeads, 1, headDims}; // BNSD
  std::vector<int64_t> keyShape = {batchSize, keyNumHeads, sequenceLengthKV, headDims}; // BNSD
  std::vector<int64_t> valueShape = {batchSize, keyNumHeads, sequenceLengthKV, headDims}; // BNSD
  std::vector<int64_t> attenShape = {batchSize, 1, 1, sequenceLengthKV}; // B11S
  std::vector<int64_t> outShape = {batchSize, numHeads, 1, headDims}; // BNSD
  void *queryDeviceAddr = nullptr;
  void *keyDeviceAddr = nullptr;
  void *valueDeviceAddr = nullptr;
  void *attenDeviceAddr = nullptr;
  void *outDeviceAddr = nullptr;
  aclTensor *queryTensor = nullptr;
  aclTensor *keyTensor = nullptr;
  aclTensor *valueTensor = nullptr;
  aclTensor *attenTensor = nullptr;
  aclTensor *outTensor = nullptr;
  std::vector<float> queryHostData(batchSize * numHeads * headDims, 1.0f);
  std::vector<float> keyHostData(batchSize * keyNumHeads * sequenceLengthKV * headDims, 1.0f);
  std::vector<float> valueHostData(batchSize * keyNumHeads * sequenceLengthKV * headDims, 1.0f);
  std::vector<int8_t> attenHostData(batchSize * sequenceLengthKV, 0);
  std::vector<float> outHostData(batchSize * numHeads * headDims, 1.0f);
 
  // 创建query aclTensor
  ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建key aclTensor
  ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  int kvTensorNum = 1;
  aclTensor *tensorsOfKey[kvTensorNum];
  tensorsOfKey[0] = keyTensor;
  auto tensorKeyList = aclCreateTensorList(tensorsOfKey, kvTensorNum);
  // 创建value aclTensor
  ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor *tensorsOfValue[kvTensorNum];
  tensorsOfValue[0] = valueTensor;
  auto tensorValueList = aclCreateTensorList(tensorsOfValue, kvTensorNum);
  // 创建atten aclTensor
  ret = CreateAclTensor(attenHostData, attenShape, &attenDeviceAddr, aclDataType::ACL_INT8, &attenTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &outTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  std::vector<int64_t> actualSeqlenVector = {sequenceLengthKV};
  auto actualSeqLengths = aclCreateIntArray(actualSeqlenVector.data(), actualSeqlenVector.size());
  
  int64_t numKeyValueHeads = numHeads;
  int64_t blockSize = 1;
  int64_t innerPrecise = 1;
  double scaleValue = 1 / sqrt(headDims); // 1/sqrt(d)
  string sLayerOut = "BNSD";
  char layerOut[sLayerOut.length()];
  strcpy(layerOut, sLayerOut.c_str());
  // 3. 调用CANN算子库API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用第一段接口
  ret = aclnnIncreFlashAttentionV4GetWorkspaceSize(queryTensor, tensorKeyList, tensorValueList, nullptr, attenTensor, actualSeqLengths, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                  nullptr, nullptr, nullptr, nullptr, numHeads, scaleValue, layerOut, numKeyValueHeads, blockSize, innerPrecise, outTensor, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIncreFlashAttentionV4GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用第二段接口
  ret = aclnnIncreFlashAttentionV4(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIncreFlashAttentionV4 failed. ERROR: %d\n", ret); return ret);
 
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
 
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<double> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
 
  // 6. 释放资源
  aclDestroyTensor(queryTensor);
  aclDestroyTensor(keyTensor);
  aclDestroyTensor(valueTensor);
  aclDestroyTensor(attenTensor);
  aclDestroyTensor(outTensor);
  aclDestroyIntArray(actualSeqLengths);
  aclrtFree(queryDeviceAddr);
  aclrtFree(keyDeviceAddr);
  aclrtFree(valueDeviceAddr);
  aclrtFree(attenDeviceAddr);
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