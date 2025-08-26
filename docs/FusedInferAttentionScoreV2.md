声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# FusedInferAttentionScoreV2

## 支持的产品型号

Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能说明

-   算子功能：适配增量&全量推理场景的FlashAttention算子，既可以支持全量计算场景（PromptFlashAttention），也可支持增量计算场景（IncreFlashAttention）。当Query矩阵的S为1，进入IncreFlashAttention分支，其余场景进入PromptFlashAttention分支。
-   计算公式：详细内容可参考[PromptFlashAttentionV3](PromptFlashAttentionV3.md)及[IncreFlashAttentionV4](IncreFlashAttentionV4.md)。

## 实现原理
该算子是全量计算场景（PromptFlashAttention）和增量计算场景（IncreFlashAttention）的融合算子，详细实现原理可参考[PromptFlashAttentionV3](PromptFlashAttentionV3.md)及[IncreFlashAttentionV4](IncreFlashAttentionV4.md)。

## 算子执行接口

算子执行接口为[两段式接口](common/两段式接口.md)，必须先调用“aclnnFusedInferAttentionScoreV2GetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnFusedInferAttentionScoreV2”接口执行计算。

* `aclnnStatus aclnnFusedInferAttentionScoreV2GetWorkspaceSize(const aclTensor *query, const aclTensorList *key, const aclTensorList *value, const aclTensor *pseShiftOptional, const aclTensor *attenMaskOptional, const aclIntArray *actualSeqLengthsOptional, const aclIntArray *actualSeqLengthsKvOptional, const aclTensor *deqScale1Optional, const aclTensor *quantScale1Optional, const aclTensor *deqScale2Optional, const aclTensor *quantScale2Optional, const aclTensor *quantOffset2Optional, const aclTensor *antiquantScaleOptional, const aclTensor *antiquantOffsetOptional, const aclTensor *blockTableOptional, const aclTensor *queryPaddingSizeOptional, const aclTensor *kvPaddingSizeOptional,  const aclTensor *keyAntiquantScaleOptional, const aclTensor *keyAntiquantOffsetOptional, const aclTensor *valueAntiquantScaleOptional, const aclTensor *valueAntiquantOffsetOptional, const aclTensor *keySharedPrefixOptional, const aclTensor *valueSharedPrefixOptional, const aclIntArray *actualSharedPrefixLenOptional, int64_t numHeads, double scaleValue, int64_t preTokens, int64_t nextTokens, char *inputLayout, int64_t numKeyValueHeads, int64_t sparseMode, int64_t innerPrecise, int64_t blockSize, int64_t antiquantMode, bool softmaxLseFlag, int64_t keyAntiquantMode, int64_t valueAntiquantMode, const aclTensor *attentionOut, const aclTensor *softmaxLse, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnFusedInferAttentionScoreV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`

**说明：**

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnFusedInferAttentionScoreV2GetWorkspaceSize

-   **参数说明：**

    - query（aclTensor\*，计算输入）：Device侧的aclTensor，attention结构的Query输入，数据类型支持FLOAT16、BFLOAT16、INT8，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
    
    - key（aclTensorList\*，计算输入）：Device侧的aclTensorList，attention结构的Key输入，数据类型支持FLOAT16、BFLOAT16、INT8，INT4（INT32），不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
    
    - value（aclTensorList\*，计算输入）：Device侧的aclTensorList，attention结构的Value输入，数据类型支持FLOAT16、BFLOAT16、INT8，INT4（INT32），不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
    
    -   pseShiftOptional（aclTensor\*，计算输入）：Device侧的aclTensor，在attention结构内部的位置编码参数，数据类型支持FLOAT16、BFLOAT16，数据类型与query的数据类型需满足数据类型推导规则。不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。如不使用该功能时可传入nullptr。
        - Q_S不为1，要求在pseShiftOptional为FLOAT16类型时，此时的query为FLOAT16或INT8类型，而在pseShiftOptional为BFLOAT16类型时，要求此时的query为BFLOAT16类型。输入shape类型需为 (B,N,Q_S,KV_S) 或 (1,N,Q_S,KV_S)，其中Q_S为query的shape中的S，KV_S为key和value的shape中的S。对于pseShiftOptional的KV_S为非32对齐的场景，建议padding到32字节来提高性能，多余部分的填充值不做要求。
        - Q_S为1，要求在pseShiftOptional为FLOAT16类型时，此时的query为FLOAT16类型，而在pseShiftOptional为BFLOAT16类型时，要求此时的query为BFLOAT16类型。输入shape类型需为 (B,N,1,KV_S) 或 (1,N,1,KV_S)，其中KV_S为key和value的shape中的S。对于pseShiftOptional的KV_S为非32对齐的场景，建议padding到32字节来提高性能，多余部分的填充值不做要求。
    
    -   attenMaskOptional（aclTensor\*，计算输入）：Device侧的aclTensor，对QK的结果进行mask，用于指示是否计算Token间的相关性，不支持[非连续的Tensor](common/非连续的Tensor.md)，数据类型支持BOOL、INT8和UINT8。[数据格式](common/数据格式.md)支持ND。如果不使用该功能可传入nullptr。
        -  Q_S不为1时建议shape输入 (Q_S,KV_S); (B,Q_S,KV_S); (1,Q_S,KV_S); (B,1,Q_S,KV_S); (1,1,Q_S,KV_S)。
        -  Q_S为1时建议shape输入(B,KV_S); (B,1,KV_S); (B,1,1,KV_S)。
    
        其中Q_S为query的shape中的S，KV_S为key和value的shape中的S，但如果Q_S、KV_S非16或32对齐，可以向上取到对齐的S。综合约束请见[约束说明](#约束说明)。
        
    - actualSeqLengthsOptional（aclIntArray\*，计算输入）：Host侧的aclIntArray，代表不同Batch中query的有效Sequence Length，数据类型支持INT64。如果不指定seqlen可以传入nullptr，表示和query的shape的S长度相同。限制：该入参中每个batch的有效Sequence Length应该不大于query中对应batch的Sequence Length，Q_S为1时该参数无效。seqlen的传入长度为1时，每个Batch使用相同seqlen；传入长度大于等于Batch时取seqlen的前Batch个数。其他长度不支持。
    
    - actualSeqLengthsKvOptional（aclIntArray\*，计算输入）：Host侧的aclIntArray，可传入nullptr，代表不同Batch中key/value的有效Sequence Length。数据类型支持INT64。如果不指定seqlen可以传入nullptr，表示和key/value的shape的S长度相同。限制：该入参中每个batch的有效Sequence Length应该不大于key/value中对应batch的Sequence Length。seqlenKv的传入长度为1时，每个Batch使用相同seqlenKv；传入长度大于等于Batch时取seqlenKv的前Batch个数。其他长度不支持。
    
    - deqScale1Optional（aclTensor\*，计算输入）：Device侧的aclTensor，数据类型支持UINT64, FLOAT32。[数据格式](common/数据格式.md)支持ND，表示BMM1后面的反量化因子，支持per-tensor。 如不使用该功能时可传入nullptr，综合约束请见[约束说明](#约束说明)。
    
    - quantScale1Optional（aclTensor\*，计算输入）：Device侧的aclTensor，数据类型支持FLOAT32。[数据格式](common/数据格式.md)支持ND，表示BMM2前面的量化因子，支持per-tensor。 如不使用该功能时可传入nullptr，综合约束请见[约束说明](#约束说明)。
    
    - deqScale2Optional（aclTensor\*，计算输入）：Device侧的aclTensor，数据类型支持UINT64, FLOAT32。[数据格式](common/数据格式.md)支持ND，表示BMM2后面的反量化因子，支持per-tensor。 如不使用该功能时可传入nullptr，综合约束请见[约束说明](#约束说明)。
    
    - quantScale2Optional（aclTensor\*，计算输入）：Device侧的aclTensor，数据类型支持FLOAT32、BFLOAT16。[数据格式](common/数据格式.md)支持ND，表示输出的量化因子，支持per-tensor，per-channel。 如不使用该功能时可传入nullptr，综合约束请见[约束说明](#约束说明)。
    
    - quantOffset2Optional（aclTensor\*，计算输入）：Device侧的aclTensor，数据类型支持FLOAT32、BFLOAT16。[数据格式](common/数据格式.md)支持ND，表示输出的量化偏移，支持per-tensor，per-channel。 如不使用该功能时可传入nullptr，综合约束请见[约束说明](#约束说明)。
    
    - antiquantScaleOptional（aclTensor\*，计算输入）：Device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16、FLOAT32。[数据格式](common/数据格式.md)支持ND，表示伪量化因子，支持per-tensor，per-channel，per-token。Q_S大于等于2时只支持FLOAT16，如不使用该功能时可传入nullptr。综合约束请见[约束说明](#约束说明)。
    
    - antiquantOffsetOptional（aclTensor\*，计算输入）：Device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16、FLOAT32。[数据格式](common/数据格式.md)支持ND，表示伪量化偏移，支持per-tensor，per-channel，per-token。Q_S大于等于2时只支持FLOAT16，如不使用该功能时可传入nullptr。综合约束请见[约束说明](#约束说明)。
    
    - blockTableOptional（aclTensor\*，计算输入）：Device侧的aclTensor，数据类型支持INT32。[数据格式](common/数据格式.md)支持ND。表示PageAttention中KV存储使用的block映射表，如不使用该功能可传入nullptr。
    
    - queryPaddingSizeOptional（aclTensor\*，计算输入）：Device侧的aclTensor，数据类型支持INT64。[数据格式](common/数据格式.md)支持ND。表示Query中每个batch的数据是否右对齐，且右对齐的个数是多少。仅支持Q_S大于1，其余场景该参数无效。用户不特意指定时可传入默认值nullptr。

    - kvPaddingSizeOptional（aclTensor\*，计算输入）：Device侧的aclTensor，数据类型支持INT64。[数据格式](common/数据格式.md)支持ND。表示key/value中每个batch的数据是否右对齐，且右对齐的个数是多少。用户不特意指定时可传入默认值nullptr。
    
    - keyAntiquantScaleOptional（aclTensor\*，计算输入）：Device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16、FLOAT32。[数据格式](common/数据格式.md)支持ND，kv伪量化参数分离时表示key的反量化因子，支持per-tensor，per-channel，per-token。Q_S大于等于2时仅支持per-token和per-channel模式，如不使用该功能时可传入nullptr。综合约束请见[约束说明](#约束说明)。
    
    - keyAntiquantOffsetOptional（aclTensor\*，计算输入）：Device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16、FLOAT32。[数据格式](common/数据格式.md)支持ND，kv伪量化参数分离时表示key的反量化偏移，支持per-tensor，per-channel，per-token。Q_S大于等于2时仅支持per-token和per-channel模式，如不使用该功能时可传入nullptr。综合约束请见[约束说明](#约束说明)。
    
    - valueAntiquantScaleOptional（aclTensor\*，计算输入）：Device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16、FLOAT32。[数据格式](common/数据格式.md)支持ND，kv伪量化参数分离时表示value的反量化因子，支持per-tensor，per-channel，per-token。Q_S大于等于2时仅支持per-token和per-channel模式，如不使用该功能时可传入nullptr。综合约束请见[约束说明](#约束说明)。
    
    - valueAntiquantOffsetOptional（aclTensor\*，计算输入）：Device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16、FLOAT32。[数据格式](common/数据格式.md)支持ND，kv伪量化参数分离时表示value的反量化偏移，支持per-tensor，per-channel，per-token。Q_S大于等于2时仅支持per-token和per-channel模式，如不使用该功能时可传入nullptr。综合约束请见[约束说明](#约束说明)。
    
    - keySharedPrefixOptional（aclTensor\*，计算输入）：Device侧的aclTensor，attention结构中Key的系统前缀部分的参数，数据类型支持FLOAT16、BFLOAT16、INT8，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。如不使用该功能时可传入nullptr。综合约束请见[约束说明](#约束说明)。
    
    - valueSharedPrefixOptional（aclTensor\*，计算输入）：Device侧的aclTensor，attention结构中Value的系统前缀部分的输入，数据类型支持FLOAT16、BFLOAT16、INT8，不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。如不使用该功能时可传入nullptr。综合约束请见[约束说明](#约束说明)。
    
    - actualSharedPrefixLenOptional（aclIntArray\*，计算输入）：Host侧的aclIntArray，可传入nullptr，代表keySharedPrefix/valueSharedPrefix的有效Sequence Length。数据类型支持INT64。如果不指定seqlen可以传入nullptr，表示和keySharedPrefix/valueSharedPrefix的s长度相同。限制：该入参中的有效Sequence Length应该不大于keySharedPrefix/valueSharedPrefix中的Sequence Length。
    
    - numHeads（int64\_t，计算输入）：Host侧的int，代表query的head个数，数据类型支持INT64，在BNSD场景下，需要与shape中的query的N轴shape值相同，否则执行异常。
    
    - scaleValue（double，计算输入）：Host侧的double，公式中d开根号的倒数，代表缩放系数，作为计算流中Muls的scalar值，数据类型支持DOUBLE。数据类型与query的数据类型需满足数据类型推导规则。用户不特意指定时可传入默认值1.0。
    
    - preTokens（int64\_t，计算输入）：Host侧的int，用于稀疏计算，表示attention需要和前几个Token计算关联，数据类型支持INT64。用户不特意指定时可传入默认值2147483647，Q_S为1时该参数无效。
    
    - nextTokens（int64\_t，计算输入）：Host侧的int，用于稀疏计算，表示attention需要和后几个Token计算关联。数据类型支持INT64。用户不特意指定时可传入默认值2147483647，Q_S为1时该参数无效。
    
    - inputLayout（char\*，计算输入）：Host侧的字符指针CHAR\*，用于标识输入query、key、value的数据排布格式，当前支持BSH、BSND、BNSD、BNSD_BSND(输入为BNSD时，输出格式为BSND，仅支持Q_S大于1)。用户不特意指定时可传入默认值"BSH"。
    
       **说明**：
       query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Head-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N
    
    - numKeyValueHeads（int64\_t，计算输入）：Host侧的int，代表key、value中head个数，用于支持GQA（Grouped-Query Attention，分组查询注意力）场景，数据类型支持INT64。用户不特意指定时可传入默认值0，表示key/value和query的head个数相等，需要满足numHeads整除numKeyValueHeads，numHeads与numKeyValueHeads的比值不能大于64。在BSND、BNSD、BNSD_BSND场景下，还需要与shape中的key/value的N轴shape值相同，否则执行异常。
    
    - sparseMode（int64\_t，计算输入）：Host侧的int，表示sparse的模式。数据类型支持INT64。Q_S为1时该参数无效。
      -   sparseMode为0时，代表defaultMask模式，如果attenmask未传入则不做mask操作，忽略preTokens和nextTokens（内部赋值为INT\_MAX）；如果传入，则需要传入完整的attenmask矩阵（S1 \* S2），表示preTokens和nextTokens之间的部分需要计算。
      -   sparseMode为1时，代表allMask，必须传入完整的attenmask矩阵（S1 \* S2）。
      -   sparseMode为2时，代表leftUpCausal模式的mask，需要传入优化后的attenmask矩阵（2048\*2048）。
      -   sparseMode为3时，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景，需要传入优化后的attenmask矩阵（2048\*2048）。
      -   sparseMode为4时，代表band模式的mask，需要传入优化后的attenmask矩阵（2048\*2048）。
      -   sparseMode为5、6、7、8时，分别代表prefix、global、dilated、block\_local，**均暂不支持**。用户不特意指定时可传入默认值0。综合约束请见[约束说明](#约束说明)。
    
    - innerPrecise（int64\_t，计算输入）：Host侧的int，一共4种模式：0、1、2、3。一共两位bit位，第0位（bit0）表示高精度或者高性能选择，第1位（bit1）表示是否做行无效修正。数据类型支持INT64。Q_S>1时，sparse_mode为0或1，并传入用户自定义mask的情况下，建议开启行无效；Q_S为1时该参数仅支持innerPrecise为0和1。综合约束请见[约束说明](#约束说明)。
      - innerPrecise为0时，代表开启高精度模式，且不做行无效修正。
    
      - innerPrecise为1时，代表高性能模式，且不做行无效修正。
    
      - innerPrecise为2时，代表开启高精度模式，且做行无效修正。
    
      - innerPrecise为3时，代表高性能模式，且做行无效修正。
    
      **说明：**
      BFLOAT16和INT8不区分高精度和高性能，行无效修正对FLOAT16、BFLOAT16和INT8均生效。
      当前0、1为保留配置值，当计算过程中“参与计算的mask部分”存在某整行全为1的情况时，精度可能会有损失。此时可以尝试将该参数配置为2或3来使能行无效功能以提升精度，但是该配置会导致性能下降。
      如果算子可判断出存在无效行场景，会自动使能无效行计算，例如sparse_mode为3，Sq > Skv场景。
    
    - blockSize（int64\_t，计算输入）：Host侧的int64\_t，PageAttention中KV存储每个block中最大的token个数，默认为0，数据类型支持INT64。
    
    - antiquantMode（int64，计算输入）：伪量化的方式，传入0时表示为per-channel（per-channel包含per-tensor），传入1时表示per-token。Q_S大于等于2时该参数无效，用户不特意指定时可传入默认值0，传入0和1之外的其他值会执行异常。
    
    - softmaxLseFlag（bool，计算输入）：是否输出softmax_lse，支持S轴外切（增加输出）。用户不特意指定时可传入默认值false。
    
    - keyAntiquantMode（int64，计算输入）：key 的伪量化的方式。Q_S大于等于2时仅支持传入值为0、1，用户不特意指定时可传入默认值0，传入0、1、2、3、4和5之外的其他值会执行异常。除了keyAntiquantMode为0并且valueAntiquantMode为1的场景外，需要与valueAntiquantMode一致。综合约束请见[约束说明](#约束说明)。
        - keyAntiquantMode为0时，代表per-channel模式（per-channel包含per-tensor）。
        - keyAntiquantMode为1时，代表per-token模式。
        - keyAntiquantMode为2时，代表per-tensor叠加per-head模式。
        - keyAntiquantMode为3时，代表per-token叠加per-head模式。
        - keyAntiquantMode为4时，代表per-token叠加使用page attention模式管理scale/offset模式。
        - keyAntiquantMode为5时，代表per-token叠加per head并使用page attention模式管理scale/offset模式。
    - valueAntiquantMode（int64，计算输入）：value 的伪量化的方式，模式编号与keyAntiquantMode一致。Q_S大于等于2时仅支持传入值为0、1，用户不特意指定时可传入默认值0，传入0、1、2、3、4和5之外的其他值会执行异常。除了keyAntiquantMode为0并且valueAntiquantMode为1的场景外，需要与 keyAntiquantMode 一致。综合约束请见[约束说明](#约束说明)。
    
    - attentionOut（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出，数据类型支持FLOAT16、BFLOAT16、INT8。[数据格式](common/数据格式.md)支持ND。限制：当inputLayout为BNSD_BSND时，输入query的shape是BNSD，输出shape为BSND；其余情况该入参的shape需要与入参query的shape保持一致。
    
    - softmaxLse（aclTensor\*，计算输出）：ring attention算法对query乘key的结果，先取max得到softmax_max。query乘key的结果减去softmax_max, 再取exp，接着求sum，得到softmax_sum。最后对softmax_sum取log，再加上softmax_max得到的结果。用户不特意指定时可传入默认值nullptr。数据类型支持FLOAT32，softmaxLseFlag为True时,shape必须为[B,N,Q_S,1],数据为inf的代表无效数据；softmaxLseFlag为False时，如果softmaxLse传入的Tensor非空，则直接返回该Tensor数据，如果softmaxLse传入的是nullptr，则返回shape为{1}全0的Tensor。
    
    - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
    
    - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。


-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

    ```
    第一段接口完成入参校验，若出现以下错误码，则对应原因为：
    -  返回161001（ACLNN_ERR_PARAM_NULLPTR）：传入的query、key、value、attentionOut是空指针。
    -  返回161002（ACLNN_ERR_PARAM_INVALID）：query、key、value、pseShift、attenMask、attentionOut的数据类型和数据格式不在支持的范围内。
    -  返回361001（ACLNN_ERR_RUNTIME_ERROR）：API内存调用npu runtime的接口异常。
    ```

### aclnnFusedInferAttentionScoreV2

-   **参数说明：**
    -   workspace（void\*，入参）：在Device侧申请的workspace内存地址。
    -   workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnFusedInferAttentionScoreV2GetWorkspaceSize获取。
    -   executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
    -   stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明

-   该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
-   入参为空的处理：算子内部需要判断参数query是否为空，如果是空则直接返回。参数query不为空Tensor，参数key、value为空tensor（即S2为0），则attentionOut填充为全零。attentionOut为空Tensor时，AscendCLNN框架会处理。其余在上述参数说明中标注了"可传入nullptr"的入参为空指针时，不进行处理。
-   参数key、value中对应tensor的shape需要完全一致；非连续场景下 key、value的tensorlist中的batch只能为1，个数等于query的B，N和D需要相等。由于tensorlist限制, 非连续场景下B不能大于256。
-   int8量化相关入参数量与输入、输出[数据格式](common/数据格式.md)的综合限制：
    - 输入为INT8，输出为INT8的场景：入参deqScale1、quantScale1、deqScale2、quantScale2需要同时存在，quantOffset2可选，不传时默认为0。
    - 输入为INT8，输出为FLOAT16的场景：入参deqScale1、quantScale1、deqScale2需要同时存在，若存在入参quantOffset2 或 quantScale2（即不为nullptr），则报错并返回。
    - 输入全为FLOAT16或BFLOAT16，输出为INT8的场景：入参quantScale2需存在，quantOffset2可选，不传时默认为0，若存在入参deqScale1 或 quantScale1 或 deqScale2（即不为nullptr），则报错并返回。
    - 入参 quantScale2 和 quantOffset2 支持 per-tensor/per-channel 两种格式和 FLOAT32/BFLOAT16 两种数据类型。若传入 quantOffset2 ，需保证其类型和shape信息与 quantScale2 一致。当输入为BFLOAT16时，同时支持FLOAT32和BFLOAT16，否则仅支持FLOAT32 。per-channel 格式，当输出layout为BSH时，要求 quantScale2 所有维度的乘积等于H；其他layout要求乘积等于N*D。（建议输出layout为BSH时，quantScale2 shape传入[1,1,H]或[H]；输出为BNSD时，建议传入[1,N,1,D]或[N,D]；输出为BSND时，建议传入[1,1,N,D]或[N,D]）
-   伪量化参数 antiquantScale和antiquantOffset约束：
    - 支持per-channel、per-tensor和per-token三种模式：
      - per-channel模式：两个参数的shape可支持\(2, N, 1, D\)，\(2, N, D\)，\(2, H\)，N为numKeyValueHeads。参数数据类型和query数据类型相同，antiquantMode置0。
      - per-tensor模式:两个参数的shape均为(2)，数据类型和query数据类型相同, antiquantMode置0。
      - per-token模式:两个参数的shape均为\(2, B, S\), 数据类型固定为FLOAT32, antiquantMode置1。
    - 支持对称量化和非对称量化：
      - 非对称量化模式下， antiquantScale和antiquantOffset参数需同时存在。
      - 对称量化模式下，antiquantOffset可以为空（即nullptr）；当antiquantOffset参数为空时，执行对称量化，否则执行非对称量化。

- **当Q_S大于1时**：
   -   query，key，value输入，功能使用限制如下：
        -   支持B轴小于等于65536。如果输入类型为INT8且D轴不是32字节对齐，则B轴的最大支持值为128。若输入类型为FLOAT16或BFLOAT16且D轴不是16字节对齐，B轴同样仅支持到128。
        -   支持N轴小于等于256，支持D轴小于等于512。inputLayout为BSH或者BSND时，要求N*D小于65535。
        -   S支持小于等于20971520（20M）。部分长序列场景下，如果计算量过大可能会导致pfa算子执行超时（aicore error类型报错，errorStr为:timeout or trap error），此场景下建议做S切分处理，注：这里计算量会受B、S、N、D等的影响，值越大计算量越大。典型的会超时的长序列(即B、S、N、D的乘积较大)场景包括但不限于： 
              - （1）B=1, Q_N=20, Q_S=2097152, D = 256, KV_N=1, KV_S=2097152;
              - （2）B=1, Q_N=2, Q_S=20971520, D = 256, KV_N=2, KV_S=20971520;
              - （3）B=20, Q_N=1, Q_S=2097152, D = 256, KV_N=1, KV_S=2097152;
              - （4）B=1, Q_N=10, Q_S=2097152, D = 512, KV_N=1, KV_S=2097152。
        -   query、key、value或attentionOut类型包含INT8时，D轴需要32对齐；query、key、value或attentionOut类型包含INT4时，D轴需要64对齐；类型全为FLOAT16、BFLOAT16时，D轴需16对齐。
   -   参数sparseMode当前仅支持值为0、1、2、3、4的场景，取其它值时会报错。
        -   sparseMode = 0时，attenMask如果为空指针,或者在左padding场景传入attenMask，则忽略入参preTokens、nextTokens。
        -   sparseMode = 2、3、4时，attenMask的shape需要为S,S或1,S,S或1,1,S,S,其中S的值需要固定为2048，且需要用户保证传入的attenMask为下三角，不传入attenMask或者传入的shape不正确报错。
        -   sparseMode = 1、2、3的场景忽略入参preTokens、nextTokens并按照相关规则赋值。
   -   kvCache反量化的合成参数场景仅支持query为FLOAT16时，将INT8类型的key和value反量化到FLOAT16。入参key/value的datarange与入参antiquantScale的datarange乘积范围在（-1，1）范围内，高性能模式可以保证精度，否则需要开启高精度模式来保证精度。
   -   page attention场景:
        -   page attention的使能必要条件是blockTable存在且有效，同时key、value是按照blockTable中的索引在一片连续内存中排布，支持key、value dtype为FLOAT16/BFLOAT16/INT8，在该场景下key、value的inputLayout参数无效。blockTable中填充的是blockid，当前不会对blockid的合法性进行校验，需用户自行保证。
        -   blockSize是用户自定义的参数，该参数的取值会影响page attention的性能，在使能page attention场景下，blockSize最小为128, 最大为512，且要求是128的倍数。通常情况下，page attention可以提高吞吐量，但会带来性能上的下降。
        -   page attention场景下，当输入kv cache排布格式为（blocknum, blocksize, H），且 KV_N * D 超过65535时，受硬件指令约束，会被拦截报错。可通过使能GQA（减小 KV_N）或调整kv cache排布格式为（blocknum, KV_N, blocksize, D）解决。当query的inputLayout为BNSD时，kv cache排布支持（blocknum, blocksize, H）和（blocknum, KV_N, blocksize, D）两种格式，当query的inputLayout为BSH、BSND时，kv cache排布只支持（blocknum, blocksize, H）一种格式。blocknum不能小于根据actualSeqLengthsKv和blockSize计算的每个batch的block数量之和。且key和value的shape需保证一致。
        -   page attention不支持伪量化场景，不支持tensorlist场景，不支持左padding场景。
        -   page attention场景下，必须传入actualSeqLengthsKv。
        -   page attention场景下，不支持Q为BF16/FP16、KV为INT4的场景。
        -   page attention场景下，blockTable必须为二维，第一维长度需等于B，第二维长度不能小于maxBlockNumPerSeq（maxBlockNumPerSeq为不同batch中最大actualSeqLengthsKv对应的block数量）。
        -   page attention场景下，不支持query为int8。
        -   page attention的使能场景下，以下场景输入KV_S需要大于等于maxBlockNumPerSeq * blockSize
            - 传入attenMask时，例如 mask shape为(B, 1, Q_S, KV_S)
            - 传入pseShift时，例如 pseShift shape为(B, N, Q_S, KV_S)
   -   query左padding场景:
        -   query左padding场景query的搬运起点计算公式为：Q_S - queryPaddingSize - actualSeqLengths。query的搬运终点计算公式为：Q_S - queryPaddingSize。其中query的搬运起点不能小于0，终点不能大于Q_S，否则结果将不符合预期。
        -   query左padding场景kvPaddingSize小于0时将被置为0。
        -   query左padding场景需要与actualSeqLengths参数一起使能，否则默认为query右padding场景。
        -   query左padding场景不支持PageAttention，不能与blocktable参数一起使能。
        -   query左padding场景不支持Q为BF16/FP16、KV为INT4的场景。
   -   kv左padding场景:
        -   kv左padding场景key和value的搬运起点计算公式为：KV_S - kvPaddingSize - actualSeqLengthsKv。key和value的搬运终点计算公式为：KV_S - kvPaddingSize。其中key和value的搬运起点不能小于0，终点不能大于KV_S，否则结果将不符合预期。
        -   kv左padding场景kvPaddingSize小于0时将被置为0。
        -   kv左padding场景需要与actualSeqLengthsKv参数一起使能，否则默认为kv右padding场景。
        -   kv左padding场景不支持PageAttention，不能与blocktable参数一起使能。
        -   kv左padding场景不支持Q为BF16/FP16、KV为INT4的场景。
   -   输出为int8时，quantScale2 和 quantOffset2 为 per-channel 时，暂不支持左padding、Ring Attention或者D非32Byte对齐的场景。
   -   输出为int8时，暂不支持sparse为band且preTokens/nextTokens为负数。
   -   pseShift功能使用限制如下：
        - 支持query数据类型为FLOAT16或BFLOAT16或INT8场景下使用该功能。
        - query数据类型为FLOAT16且pseShift存在时，强制走高精度模式，对应的限制继承自高精度模式的限制。
        - Q_S需大于等于query的S长度，KV_S需大于等于key的S长度。prefix场景KV_S需大于等于actualSharedPrefixLen与key的S长度之和。
   -   输出为INT8时，入参quantOffset2传入非空指针和非空tensor值，并且sparseMode、preTokens和nextTokens满足以下条件，矩阵会存在某几行不参与计算的情况，导致计算结果误差，该场景会拦截(解决方案：如果希望该场景不被拦截，需要在FIA接口外部做后量化操作，不在FIA接口内部使能)：
        -   sparseMode = 0，attenMask如果非空指针，每个batch actualSeqLengths — actualSeqLengthsKV - actualSharedPrefixLen - preTokens > 0 或 nextTokens < 0 时，满足拦截条件
        -   sparseMode = 1 或 2，不会出现满足拦截条件的情况
        -   sparseMode = 3，每个batch actualSeqLengthsKV + actualSharedPrefixLen - actualSeqLengths < 0，满足拦截条件
        -   sparseMode = 4，preTokens < 0 或 每个batch nextTokens + actualSeqLengthsKV + actualSharedPrefixLen - actualSeqLengths < 0 时，满足拦截条件
   -   prefix相关参数约束：
        -   keySharedPrefix和valueSharedPrefix要么都为空，要么都不为空
        -   keySharedPrefix和valueSharedPrefix都不为空时，keySharedPrefix、valueSharedPrefix、key、value的维度相同、dtype保持一致。
        -   keySharedPrefix和valueSharedPrefix都不为空时，keySharedPrefix的shape第一维batch必须为1，layout为BNSD和BSND情况下N、D轴要与key一致、BSH情况下H要与key一致，valueSharedPrefix同理。keySharedPrefix和valueSharedPrefix的S应相等
        -   当actualSharedPrefixLen存在时，actualSharedPrefixLen的shape需要为[1]，值不能大于keySharedPrefix和valueSharedPrefix的S
        -   公共前缀的S加上key或value的S的结果，要满足原先key或value的S的限制
        -   prefix不支持PageAttention场景、不支持左padding场景、不支持tensorlist场景
        -   prefix场景，sparse为0或1时，如果传入attenmask，则S2需大于等于actualSharedPrefixLen与key的S长度之和
        -   prefix场景，不支持输入qkv全部为int8的情况
   -   kv伪量化参数分离
        - keyAntiquantMode 和 valueAntiquantMode需要保持一致
        - keyAntiquantScale 和 valueAntiquantScale要么都为空，要么都不为空；keyAntiquantOffset 和 valueAntiquantOffset要么都为空，要么都不为空
        - KeyAntiquantScale 和valueAntiquantScale都不为空时，其shape需要保持一致；keyAntiquantOffset 和 valueAntiquantOffset都不为空时，其shape需要保持一致
        - 仅支持per-token和per-channel模式，per-token模式下要求两个参数的shape均为\(B, S\)，数据类型固定为FLOAT32；per-channel模式下要求两个参数的shape为(N, D\)，(N, 1, D\)或(H\)，数据类型固定为BF16。
        - 当伪量化参数 和 KV分离量化参数同时传入时，以KV分离量化参数为准。
        - keyAntiquantScale与valueAntiquantScale非空场景，要求query的s小于等于16。
        - keyAntiquantScale与valueAntiquantScale非空场景，要求query的dtype为BFLOAT16,key、value的dtype为INT8，输出的dtype为BFLOAT16。
        - keyAntiquantScale与valueAntiquantScale非空场景，不支持tensorlist、左padding、page attention特性。

- **当Q_S等于1时**：
  -   query，key，value输入，功能使用限制如下：
      -   支持B轴小于等于65536，支持N轴小于等于256，支持D轴小于等于512。
      -   query、key、value输入类型均为INT8的场景暂不支持。
      -   在INT4（INT32）伪量化场景下，aclnn单算子调用支持KV INT4输入或者INT4拼接成INT32输入（建议通过dynamicQuant生成INT4格式的数据，因为dynamicQuant就是一个INT32包括8个INT4）。
      -   在INT4（INT32）伪量化场景下，若KV INT4拼接成INT32输入，那么KV的N、D或者H是实际值的八分之一（prefix同理）。并且，INT4伪量化仅支持D 64对齐（INT32支持D 8对齐）。
  -   page attention场景:
      -   page attention的使能必要条件是blocktable存在且有效，同时key、value是按照blocktable中的索引在一片连续内存中排布，支持key、value dtype为FLOAT16/BFLOAT16/INT8，在该场景下key、value的inputLayout参数无效。
      -   blockSize是用户自定义的参数，该参数的取值会影响page attention的性能，在使能page attention场景下，blockSize需要传入非0值, 且blocksize最大不超过512。key、value输入类型为FLOAT16/BFLOAT16时需要16对齐，key、value 输入类型为INT8时需要32对齐，推荐使用128。通常情况下，page attention可以提高吞吐量，但会带来性能上的下降。
      -   page attention场景下，当query的inputLayout为BNSD时，kv cache排布支持（blocknum, blocksize, H）和（blocknum, KV_N, blocksize, D）两种格式，当query的inputLayout为BSH、BSND时，kv cache排布只支持（blocknum, blocksize, H）一种格式。blocknum不能小于根据actualSeqLengthsKv和blockSize计算的每个batch的block数量之和。且key和value的shape需保证一致。
      -   page attention场景下，kv cache排布为（blocknum, KV_N, blocksize, D）时性能通常优于kv cache排布为（blocknum, blocksize, H）时的性能，建议优先选择（blocknum, KV_N, blocksize, D）格式。
      -   page attention使能场景下，当输入kv cache排布格式为（blocknum, blocksize, H），且 numKvHeads * headDim 超过64k时，受硬件指令约束，会被拦截报错。可通过使能GQA（减小 numKvHeads）或调整kv cache排布格式为（blocknum, numKvHeads, blocksize, D）解决。
      -   page attention不支持tensorlist场景，不支持左padding场景，不支持Q为BF16/FP16、KV为INT4（INT32）的场景。
      -   page attention场景下，必须传入actualSeqLengthsKv。
      -   page attention场景下，blockTable必须为二维，第一维长度需等于B，第二维长度不能小于maxBlockNumPerSeq（maxBlockNumPerSeq为每个batch中最大actualSeqLengthsKv对应的block数量）。
      -   page attention的使能场景下，以下场景输入S需要大于等于maxBlockNumPerSeq * blockSize。
          - 使能Attention mask，如mask shape为 \(B, 1, 1, S\)。
          - 使能pseShift，如pseShift shape为\(B, N, 1, S\)。
          - 使能伪量化per-token模式：输入参数antiquantScale和antiquantOffset的shape均为\(2, B, S\)。
  -   kv左padding场景:
          kv左padding场景不支持Q为BF16/FP16、KV为INT4（INT32）的场景。
      -   kv左padding场景中kvCache的搬运起点计算公式为：KV_S - kvPaddingSize - actualSeqLengths。kvCache的搬运终点计算公式为：KV_S - kvPaddingSize。其中kvCache的搬运起点或终点小于0时，返回数据结果为全0。
      -   kv左padding场景中kvPaddingSize小于0时将被置为0。
      -   kv左padding场景需要与actualSeqLengths参数一起使能，否则默认为kv右padding场景。
      -   kv左padding场景与attenMask参数一起使能时，需要保证attenMask含义正确，即能够正确的对无效数据进行隐藏。否则将引入精度问题。
  -   pseShift功能使用限制如下：
      - pseShift数据类型需与query数据类型保持一致。
      - 仅支持D轴对齐，即D轴可以被16整除。
  -   kv伪量化参数分离
      - 除了keyAntiquantMode为0并且valueAntiquantMode为1的场景外，keyAntiquantMode 和 valueAntiquantMode需要保持一致
      - keyAntiquantScale 和 valueAntiquantScale要么都为空，要么都不为空；keyAntiquantOffset 和 valueAntiquantOffset要么都为空，要么都不为空
      - KeyAntiquantScale 和valueAntiquantScale都不为空时，除了keyAntiquantMode为0并且valueAntiquantMode为1的场景外，其shape需要保持一致；keyAntiquantOffset 和 valueAntiquantOffset都不为空时，除了keyAntiquantMode为0并且valueAntiquantMode为1的场景外，其shape需要保持一致
      - 支持per-channel、per-tensor、per-token、per-tensor叠加per-head、per-token叠加per-head、per-token叠加使用page attention模式管理scale/offset、per-token叠加per head并使用page attention模式管理scale/offset，key支持per-channel叠加value支持per-token八种模式，以下N均为numKeyValueHeads：
        - per-channel模式：两个参数的shape可支持\(1, N, 1, D\)，\(1, N, D\)，\(1, H\)。参数数据类型和query数据类型相同。
        - per-tensor模式：两个参数的shape均为(1)，数据类型和query数据类型相同。
        - per-token模式：两个参数的shape均为\(1, B, S\)，数据类型固定为FLOAT32。
        - per-tensor叠加per-head模式：两个参数的shape均为(N)，数据类型和query数据类型相同。
        - per-token叠加per-head模式：两个参数的shape均为\(B, N, S\)，数据类型固定为FLOAT32。
        - per-token叠加使用page attention模式管理scale/offset模式：两个参数的shape均为\(blocknum, blocksize\)，数据类型固定为FLOAT32。
        - per-token叠加per head并使用page attention模式管理scale/offset模式：两个参数的shape均为\(blocknum, N, blocksize\)，数据类型固定为FLOAT32。
        - key支持per-channel叠加value支持per-token模式：对于key支持per-channel，两个参数的shape可支持\(1, N, 1, D\)，\(1, N, D\)，\(1, H\)并且参数数据类型和query数据类型相同；对于value支持per-token，两个参数的shape均为\(1, B, S\)并且数据类型固定为FLOAT32；当key和value的输入类型为INT8时，仅支持query和输出的dtype为FLOAT16。
      - 当伪量化参数 和 KV分离量化参数同时传入时，以KV分离量化参数为准。
      - INT4（INT32）伪量化场景仅支持KV伪量化参数分离，具体包括：
        - per-channel模式；
        - per-token模式；
        - per-token叠加per-head模式；
        - key支持per-channel叠加value支持per-token模式。
      - INT4（INT32）伪量化场景不支持后量化。
  -   prefix相关参数约束：
      - keySharedPrefix和valueSharedPrefix要么都为空，要么都不为空
      - keySharedPrefix和valueSharedPrefix都不为空时，keySharedPrefix、valueSharedPrefix、key、value的维度相同、dtype保持一致。
      - keySharedPrefix和valueSharedPrefix都不为空时，keySharedPrefix的shape第一维batch必须为1，layout为BNSD和BSND情况下N、D轴要与key一致、BSH情况下H要与key一致，valueSharedPrefix同理。keySharedPrefix和valueSharedPrefix的S应相等
      - 当actualSharedPrefixLen存在时，actualSharedPrefixLen的shape需要为[1]，值不能大于keySharedPrefix和valueSharedPrefix的S
      - 公共前缀的S加上key或value的S的结果，要满足原先key或value的S的限制
## 算子原型
```
REG_OP(FusedInferAttentionScore)
    .INPUT(query, TensorType({DT_INT8, DT_FLOAT16,DT_BF16}))
    .DYNAMIC_INPUT(key, TensorType({DT_INT8, DT_FLOAT16,DT_BF16}))
    .DYNAMIC_INPUT(value, TensorType({DT_INT8, DT_FLOAT16,DT_BF16}))
    .OPTIONAL_INPUT(pse_shift, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_FLOAT16, DT_BOOL, DT_UINT8, DT_INT8}))
    .OPTIONAL_INPUT(actual_seq_lengths, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_lengths_kv, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(dequant_scale1, TensorType({DT_UINT64, DT_FLOAT32}))
    .OPTIONAL_INPUT(quant_scale1, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(dequant_scale2, TensorType({DT_UINT64, DT_FLOAT32}))
    .OPTIONAL_INPUT(quant_scale2, TensorType({DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(quant_offset2, TensorType({DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(block_table, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(query_padding_size, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(kv_padding_size, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(key_antiquant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(key_antiquant_offset, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(value_antiquant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(value_antiquant_offset, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(key_shared_prefix, TensorType({DT_INT8, DT_FLOAT16,DT_BF16}))
    .OPTIONAL_INPUT(value_shared_prefix, TensorType({DT_INT8, DT_FLOAT16,DT_BF16}))
    .OPTIONAL_INPUT(actual_shared_prefix_len, TensorType({DT_INT64}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_INT8, DT_BF16}))
    .OUTPUT(softmax_lse, TensorType({DT_FLOAT32}))
    .REQUIRED_ATTR(num_heads, Int)
    .ATTR(scale, Float, 1.0)
    .ATTR(pre_tokens, Int, 2147483647)
    .ATTR(next_tokens, Int, 2147483647)
    .ATTR(input_layout, String, "BSH")
    .ATTR(num_key_value_heads, Int, 0)
    .ATTR(sparse_mode, Int, 0)
    .ATTR(inner_precise, Int, 1)
    .ATTR(block_size, Int, 0)
    .ATTR(antiquant_mode, Int, 0)
    .ATTR(softmax_lse_flag, Bool, false)
    .ATTR(key_antiquant_mode, Int, 0)
    .ATTR(value_antiquant_mode, Int, 0)
    .OP_END_FACTORY_REG(FusedInferAttentionScore)
```
参数解释请参见**算子执行接口**。

## 调用示例

该融合算子有两种调用方式：

- PyTorch框架调用

  如果通过PyTorch单算子方式调用该融合算子，则需要参考PyTorch融合算子[torch_npu.npu_fused_infer_attention_score](https://hiascend.com/document/redirect/PyTorchAPI)；如果用户定制了该融合算子，则需要参考《Ascend C算子开发》手册[适配PyTorch框架](https://hiascend.com/document/redirect/CannCommunityAscendCInvorkOnNetwork)。

- aclnn单算子调用方式

  通过aclnn单算子调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。
```c++

#include <iostream>
#include <vector>
#include <math.h>
#include <cstring>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_fused_infer_attention_score_v2.h"

using namespace std;

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
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

int main()
{
    // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    int32_t batchSize = 1;
    int32_t numHeads = 2;
    int32_t numKeyValueHeads = 2;
    int32_t sequenceLengthQ = 1;
    int32_t sequenceLengthKV = 512;
    int32_t headDims = 128;
    std::vector<int64_t> queryShape = {batchSize, numHeads, sequenceLengthQ, headDims};     // BNSD
    std::vector<int64_t> keyShape = {batchSize, numKeyValueHeads, sequenceLengthKV, headDims};   // BNSD
    std::vector<int64_t> valueShape = {batchSize, numKeyValueHeads, sequenceLengthKV, headDims}; // BNSD
    std::vector<int64_t> attenMaskShape = {batchSize, 1, 1, sequenceLengthKV};                  // B11S
    std::vector<int64_t> outShape = {batchSize, numHeads, sequenceLengthQ, headDims};       // BNSD
    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;
    void* valueDeviceAddr = nullptr;
    void* attenMaskDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* queryTensor = nullptr;
    aclTensor* keyTensor = nullptr;
    aclTensor* valueTensor = nullptr;
    aclTensor* attenMaskTensor = nullptr;
    aclTensor* outTensor = nullptr;
    std::vector<op::fp16_t> queryHostData(batchSize * numHeads * sequenceLengthQ * headDims, 1.0);
    std::vector<op::fp16_t> keyHostData(batchSize * numKeyValueHeads * sequenceLengthKV * headDims, 1.0);
    std::vector<op::fp16_t> valueHostData(batchSize * numKeyValueHeads * sequenceLengthKV * headDims, 1.0);
    std::vector<int8_t> attenMaskHostData(batchSize * sequenceLengthKV, 0);
    std::vector<op::fp16_t> outHostData(batchSize * numHeads * sequenceLengthQ * headDims, 1.0);

    // 创建query aclTensor
    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建key aclTensor
    ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    int kvTensorNum = 1;
    aclTensor* tensorsOfKey[kvTensorNum];
    tensorsOfKey[0] = keyTensor;
    auto tensorKeyList = aclCreateTensorList(tensorsOfKey, kvTensorNum);
    // 创建value aclTensor
    ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    aclTensor* tensorsOfValue[kvTensorNum];
    tensorsOfValue[0] = valueTensor;
    auto tensorValueList = aclCreateTensorList(tensorsOfValue, kvTensorNum);
    // 创建attenMask aclTensor
    ret = CreateAclTensor(attenMaskHostData, attenMaskShape, &attenMaskDeviceAddr, aclDataType::ACL_INT8, &attenMaskTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &outTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> actualSeqlenVector = {sequenceLengthKV};
    auto actualSeqLengths = aclCreateIntArray(actualSeqlenVector.data(), actualSeqlenVector.size());

    double scaleValue = 1 / sqrt(headDims); // 1 / sqrt(d)
    int64_t preTokens = 65535;
    int64_t nextTokens = 65535;
    string sLayerOut = "BNSD";
    char layerOut[sLayerOut.length()];
    strcpy(layerOut, sLayerOut.c_str());
    int64_t sparseMode = 0;
    int64_t innerPrecise = 0;
    int blockSize = 0;
    int antiquantMode = 0;
    bool softmaxLseFlag = false;
    int keyAntiquantMode = 0;
    int valueAntiquantMode = 0;

    // 3. 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用第一段接口
    ret = aclnnFusedInferAttentionScoreV2GetWorkspaceSize(
        queryTensor, tensorKeyList, tensorValueList, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, numHeads, scaleValue, preTokens, nextTokens, layerOut, numKeyValueHeads, sparseMode,
        innerPrecise, blockSize, antiquantMode, softmaxLseFlag, keyAntiquantMode, valueAntiquantMode, outTensor,
        nullptr, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedInferAttentionScoreV2GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用第二段接口
    ret = aclnnFusedInferAttentionScoreV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedInferAttentionScoreV2 failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    std::vector<op::fp16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        std::cout << "index: " << i << ": " << static_cast<float>(resultData[i]) << std::endl;
    }

    // 6. 释放资源
    aclDestroyTensor(queryTensor);
    aclDestroyTensor(keyTensor);
    aclDestroyTensor(valueTensor);
    aclDestroyTensor(attenMaskTensor);
    aclDestroyTensor(outTensor);
    aclDestroyIntArray(actualSeqLengths);
    aclrtFree(queryDeviceAddr);
    aclrtFree(keyDeviceAddr);
    aclrtFree(valueDeviceAddr);
    aclrtFree(attenMaskDeviceAddr);
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
