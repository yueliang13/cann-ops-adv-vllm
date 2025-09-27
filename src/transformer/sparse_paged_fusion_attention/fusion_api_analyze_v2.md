### Q1: 相比传统IFA Cent_select多了哪些参数 分别是什么意义？
差不多，只要把中间结果从输入输出去掉，在内部同时把inQueue和outQueue改成临时buf即可。然后将两个算子的其余输入输出放到一起。

```C++
__attribute__((visibility("default"))) aclnnStatus aclnnSparsePagedFusionAttentionGetWorkspaceSize(
    const aclTensor *query, const aclTensorList *key, const aclTensorList *value, const aclTensor *pseShift,
    const aclTensor *attenMask, const aclIntArray *actualSeqLengths, const aclTensor *dequantScale1,
    const aclTensor *quantScale1, const aclTensor *dequantScale2, const aclTensor *quantScale2,
    const aclTensor *quantOffset2, const aclTensor *antiquantScale, const aclTensor *antiquantOffset,
    const aclTensor *blocktable, const aclTensor *kvPaddingSize, const aclTensor *blockposition, int64_t numHeads,
    double scaleValue, char *inputLayout, int64_t numKeyValueHeads, int64_t blockSize, int64_t innerPrecise,
    const aclTensor *attentionOut, uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnCentSelectGetWorkspaceSize(
    const aclTensor *query, const aclTensor *l1_cent, const aclTensor *block_ids, const aclTensor *block_table, const aclTensor *seq_len, const aclTensor *page_position, const aclTensor *page_position_length,
    const aclTensor *workspace, const aclTensor *tiling,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * 如果要融合到IFA 中 需要做哪些操作？
 * 
 * < 需要新增的入参 >
 * l1_cent: required 新增输入
 * block_ids: 新增输入
 * seq_len 真实长度 和之前的应该不是一个意思 可以理解为total_seqlen [C? INT8?]
 * => 我们需要用这个值 算出IFA需要的 actualSeqLengths, page_position
 * 
 * op.Process(page_position_length); => page_position, page_position_length, max_page_position_length 
 * 
 * < 可复用入参 >
 * query : required 通用 [B,N,D]
 * block_table 通用
 * 
 * < 计算产生的中间变量 >
 * page_position 中间计算结果 就不用传了 
 * page_position_length 中间计算结果 对应actual_seq -> 需要再求个[:,0]max?
 * workspace -> 给IFA的workspace扩一下
 * tiling -> tillingkey的意思吗？ 
 * 
 */

```

### Q2: 新的workspace怎么加？
workspace不是自动算得吗？
```C++
/**
 * Cent_select 做了两件事 
 * CentSelectTilingData 的数据填充
 * WorkSpace的计算 
 * 
 * Q:在同一个Kernel内，各种内存空间的申请和分核是否应该统一？
 * A:这样的话，workspace就不能直接搬过去复用，应该按照计算逻辑重写 <有点繁琐>
 */
```

### Q3: 新的计算添加
不需要，直接把cent_select的代码都搬过来就可，即将它的process中的代码都放到ifa的process中，调用方式不变。

```C++
/**
 * 
 * 
 * 
 * 
 */

// Cent_select 调用
extern "C" __global__ __aicore__ void cent_select(GM_ADDR query, GM_ADDR l1_cent, GM_ADDR block_ids, GM_ADDR block_table, GM_ADDR seq_len, GM_ADDR page_position, GM_ADDR page_position_length, GM_ADDR max_page_position_length, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(CentSelectTilingData, tilingDataIn, tiling);
    const CentSelectTilingData *__restrict tilingData = &tilingDataIn;

    CentSelect op; 
    op.Init(query, l1_cent, block_ids, block_table, seq_len, page_position,page_position_length, max_page_position_length, workspace, tilingData);
    op.Process(page_position_length);
}


// IFA的调用 
#define INVOKE_IFA_ALL_VEC_OP_IMPL(templateClass, ...)                                                                 \
    do {                                                                                                               \
        templateClass<IFAType<__VA_ARGS__>> op;                                                                        \
        COPY_TILING_DATA_NO_CUBE(tiling);                                                                              \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, blocktable, kvPaddingSize, blockPosition, attentionOut,     \
                softmaxLse, user, tiling_data, &tPipe);                                                                \
        op.InitQuant(deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset,    \
                     user);                                                                                            \
        // TODO: Need to add
        // op.Init_cent_select(***)
        // op.Cent_select_process(***)
        op.Process();                                                                                                  \
    } while (0)


// 当前构思是在外面的时候 类似op.InitQuant的调用，来调用op.cent_select()
// 为了实现两个算子的真正融合 ( 最好基于IFA申请的一些Buffer来做数据搬运,但是这样的话，就不能做类似import cent_select()的方案 )




```