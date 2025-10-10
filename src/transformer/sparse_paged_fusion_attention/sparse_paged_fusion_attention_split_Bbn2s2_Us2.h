/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_paged_attention_split_Bbn2s2_Us2.h
 * \brief
 */
#ifndef SPARCE_PAGED_FUSION_ATTENTION_SPLIT_BBN2S2_US2
#define SPARCE_PAGED_FUSION_ATTENTION_SPLIT_BBN2S2_US2

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "ifa_public_define.h"

using namespace matmul;
using AscendC::CacheMode;

#define V5_SPARSE_DEBUG_ENABLE_CUBE 0 // 设置为1启用调试，设置为0关闭所有调试输出

#if V5_SPARSE_DEBUG_ENABLE_CUBE
#define V5_DEBUG_PRINTF(...) AscendC::printf(__VA_ARGS__)
#else
#define V5_DEBUG_PRINTF(...)                                                                                           \
    do {                                                                                                               \
    } while (0)
#endif

template <typename IFAT> class SparsePagedFusionAttentionAttenSplitBbn2s2Us2 {
public:
    __aicore__ inline SparsePagedFusionAttentionAttenSplitBbn2s2Us2(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *pseShift, __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths,
                                __gm__ uint8_t *blockTable, __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *blockPosition, __gm__ uint8_t *attentionOut,
                                __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
                                const SparsePagedFusionAttentionTilingData *__restrict tiling, __gm__ uint8_t *gmTiling,
                                TPipe *tPipe, bool isPrefix = false);
    __aicore__ inline void InitCentSelect(__gm__ uint8_t *query, __gm__ uint8_t *l1_cent, __gm__ uint8_t *block_ids,
                                          __gm__ uint8_t *block_table, __gm__ uint8_t *total_seq_len,
                                          __gm__ uint8_t *blockPosition, __gm__ uint8_t *pagePositionLength,
                                          __gm__ uint8_t *maxPagePositionLength, __gm__ uint8_t *workspace,
                                          const SparsePagedFusionAttentionTilingData *__restrict tiling, __gm__ uint8_t *gmTiling,
                                          TPipe *tPipe);
    __aicore__ inline void InitQuant(__gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2,
                                     __gm__ uint8_t *quantScale2, __gm__ uint8_t *quantOffset2,
                                     __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
                                     __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset,
                                     __gm__ uint8_t *valueAntiquantScale, __gm__ uint8_t *valueAntiquantOffset,
                                     __gm__ uint8_t *workspace);
    __aicore__ inline void InitAntiquant(__gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
                                     __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset,
                                     __gm__ uint8_t *valueAntiquantScale, __gm__ uint8_t *valueAntiquantOffse);
    __aicore__ inline void InitPostQuant(__gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2,
                                     __gm__ uint8_t *quantScale2, __gm__ uint8_t *quantOffset2);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessCentSelect(TPipe *tPipe);
    __aicore__ inline void CentCopyIn();
    __aicore__ inline void ReleaseCentSelectBuffers(TPipe *tPipe);
    __aicore__ inline void CentComputeTopK(LocalTensor<int32_t> &dstIndexLocal);
    __aicore__ inline void VectorCompute(LocalTensor<float> &mmResUb, LocalTensor<half> &aUb, LocalTensor<half> &bUb, uint32_t dealRowCount);
    __aicore__ inline void BlockingL1CentCopyIn(uint32_t idx);
    __aicore__ inline void CentSelectPosition(LocalTensor<int32_t> indicesLocal);
    __aicore__ inline void CentCopyOut();
    __aicore__ inline void CentMaxReducePagePositionLength(uint32_t localBlockIdx);

    __aicore__ inline void InitPrefix(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                      __gm__ uint8_t *pseShift, __gm__ uint8_t *attenMask,
                                      __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *blockTable,
                                      __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *blockPosition, __gm__ uint8_t *attentionOut,
                                      __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
                                      const SparsePagedFusionAttentionTilingDataPrefix *__restrict tiling,
                                      __gm__ uint8_t *gmTiling, TPipe *tPipe);
    __aicore__ inline void ProcessSysPrefixCombine();

    // 中间计算数据类型为float，高精度模式
    using T = float;

    using Q_T = typename IFAT::queryType;
    using KV_T = typename IFAT::kvType;
    using OUT_T = typename IFAT::outputType;
    using ORIGIN_T = typename IFAT::orginalType;
    static constexpr bool PAGE_ATTENTION = IFAT::pageAttention;
    static constexpr bool FLASH_DECODE = IFAT::flashDecode;
    static constexpr LAYOUT LAYOUT_T = IFAT::layout;
    static constexpr uint8_t PER_CHANNEL_MODE = 0; // 伪量化: K V per-channel
    static constexpr uint8_t PER_TOKEN_MODE = 1; // 伪量化: K V per-token
    static constexpr uint8_t PER_CHANNEL_TOKEN_MODE = 2; // 伪量化: K per-channel and V per-token
    static constexpr uint8_t ANTIQUANT_MODE = IFAT::antiquantMode;
    static constexpr bool SHARED_PREFIX = IFAT::sharedPrefix;

    static constexpr bool ANTIQUANT = !IsSameType<Q_T, KV_T>::value;
    static constexpr bool KVINT4 = IsSameType<KV_T, int4b_t>::value;
    static constexpr bool QUANT = (IsSameType<Q_T, KV_T>::value && IsSameType<KV_T, int8_t>::value);
    static constexpr bool ANTIQUANT_PER_CHANNEL_TOKEN = (ANTIQUANT && (ANTIQUANT_MODE == PER_CHANNEL_TOKEN_MODE));
    static constexpr bool ANTIQUANT_PER_TOKEN = (ANTIQUANT && (ANTIQUANT_MODE == PER_TOKEN_MODE));
    static constexpr bool ANTIQUANT_PER_CHANNEL = (ANTIQUANT && (ANTIQUANT_MODE == PER_CHANNEL_MODE));
    using ANTIQ_PARAMS_T_KEY = typename AscendC::Conditional<ANTIQUANT_PER_TOKEN, T, Q_T>::type;
    using ANTIQ_PARAMS_T_VALUE = typename AscendC::Conditional<ANTIQUANT_PER_CHANNEL, Q_T, T>::type;
    // 后接量化的条件需要重新审视
    static constexpr bool POST_QUANT = IsSameType<OUT_T, int8_t>::value;
    using MM_OUT_T = typename AscendC::Conditional<(ANTIQUANT || QUANT), int32_t, T>::type;

    using singleRowAType = MatmulType<TPosition::GM, CubeFormat::VECTOR, KV_T, false>;
    using multiRowAType = MatmulType<TPosition::GM, CubeFormat::ND, KV_T, false>;
    // using AType = typename AscendC::Conditional<ANTIQUANT, multiRowAType, singleRowAType>::type;
    using AType = multiRowAType;

    // define pse datetype
    using pseShiftType = typename AscendC::Conditional<AscendC::IsSameType<Q_T, int8_t>::value, half, Q_T>::type;

    template <typename SRC_T> static __aicore__ inline constexpr int32_t GetC0SizeBySrcType()
    {
        if (sizeof(SRC_T) == sizeof(float)) {
            return 8;
        } else if (sizeof(SRC_T) == sizeof(int8_t)) {
            return 32;
        }
        return 16;
    }

    // 参考mamtul_impl.h中实现
    template <typename SRC_T>
    static __aicore__ void
    CopyND2NZ(const LocalTensor<SRC_T> &dst, const GlobalTensor<SRC_T> &src, const int row, const int col,
              const int height, const int width, const int gCol, const int ndNum = 1, const int srcNdMatrixStride = 0,
              const int dstNzMatrixStride = 0, const int dstNzC0Stride = 0)
    {
        int64_t srcOffset = ((int64_t)row * (int64_t)gCol + (int64_t)col);

        Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = ndNum;
        nd2nzParams.nValue = height;
        nd2nzParams.dValue = width;
        nd2nzParams.srcNdMatrixStride = srcNdMatrixStride;
        nd2nzParams.srcDValue = gCol;

        nd2nzParams.dstNzC0Stride = dstNzC0Stride;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = dstNzMatrixStride;

        DataCopy(dst, src[srcOffset], nd2nzParams);
    }

    // 在CUBE核中实现CopyZero函数，参考Vector核中的实现
    // 可能存在问题 或者 优化空间
    template <typename SRC_T>
    static __aicore__ void CopyZero(const LocalTensor<SRC_T> &dst, const uint32_t height, const uint32_t width)
    {
        // 临时禁用零填充操作，用于性能测试
        // 注释掉所有实际操作，只保留函数框架
        
        // 创建一个int16_t类型的缓冲区（支持的类型）
        LocalTensor<int16_t> zeroBuffer;
        zeroBuffer.SetSize(width);

        // 使用Duplicate填充零值
        Duplicate(zeroBuffer, static_cast<int16_t>(0), width);

        // 创建一个临时缓冲区
        LocalTensor<SRC_T> dstBuffer;
        dstBuffer.SetSize(width);

        // 手动将int16_t类型的0转换为SRC_T类型的0
        // 使用DataCopy而不是直接赋值
        DataCopyParams paramsCast;
        paramsCast.blockCount = 1;
        paramsCast.blockLen = width;
        paramsCast.dstStride = 0;
        paramsCast.srcStride = 0;

        // 这里我们使用DataCopy来"转换"类型，而不是使用Cast
        // 由于源是全0，目标也会是全0，无论类型如何
        DataCopy(dstBuffer, zeroBuffer, paramsCast);

        // 使用标准的DataCopyParams进行复制到最终目标
        DataCopyParams copyParams;
        copyParams.blockCount = height;
        copyParams.blockLen = width;
        copyParams.dstStride = 0;
        copyParams.srcStride = 0;

        // 执行拷贝
        DataCopy(dst, dstBuffer, copyParams);
    }


    // bmm1 回调，row方向对应k、d；col方向对应n、s2
    static __aicore__ void bmm1CopyB1(const LocalTensor<int8_t> &bMatrix, const __gm__ void *gm, int row, int col,
                                      int useK, int useN, const uint64_t tilingPtr, const uint64_t dataPtr)
    {
        // 回调函数，当前有2种方式获取 TilingData：
        // (1) 路径3，在线编译，此时 tilingDataPtr
        // 为空，但SparsePagedFusionAttentionTilingDataV2结构体中各成员默认值即为tiling结果 (2) 其它场景，tilingDataPtr
        // 非空，从其指向的GM内存中获取 tiling data，但tilingDataPtr 需要在vector侧配置给cube
        SparsePagedFusionAttentionTilingDataV2 allTilingDataV2;
        SparsePagedFusionAttentionTilingData allTilingData = allTilingDataV2.tilingBase;
        SparsePagedFusionAttentionTilingData *tilingDataPtr = reinterpret_cast<SparsePagedFusionAttentionTilingData *>(tilingPtr);
        if (tilingDataPtr != nullptr) {
            allTilingData = *tilingDataPtr;
        }
        uint32_t maxBlockNumPerBatch = allTilingData.baseParams.maxBlockNumPerBatch;
        uint32_t maxPositionNumPerBatch = allTilingData.baseParams.maxPositionNumPerBatch;
        uint64_t singleProcessSInnerSize = allTilingData.sparsePagedFusionAttentionSingleCoreParams.singleProcessSInnerSize;
        uint64_t kvCacheBlockSize = allTilingData.baseParams.blockSize;
        uint32_t totalBlockNum = allTilingData.baseParams.totalBlockNum;
        uint32_t headSize = allTilingData.baseParams.headSize;
        uint32_t bmm1BaseN = allTilingData.bmm1TilingData.baseN;
        uint32_t kvHeadNum = allTilingData.baseParams.kvHeadNum;
        uint32_t bmm1Kb = allTilingData.bmm1TilingData.Kb;
        uint32_t bmm1StepKb = allTilingData.bmm1TilingData.stepKb;
        uint32_t bmm1BaseK = allTilingData.bmm1TilingData.baseK;

        GlobalTensor<uint32_t> bmm1LocalInfo;
        bmm1LocalInfo.SetGlobalBuffer((__gm__ uint32_t *)dataPtr, 16);
        uint32_t bmm1BIdx = bmm1LocalInfo.GetValue(0);
        uint32_t bmm1N2Idx = bmm1LocalInfo.GetValue(1);
        uint32_t bmm1SInnerLoopIdx = bmm1LocalInfo.GetValue(2);
        // DataCopy 不支持64位拷贝，2个gm地址需在V侧设置时拆分，在回调里拼接
        uint32_t bmm1TensorBAddrHigh = bmm1LocalInfo.GetValue(3);
        uint32_t bmm1TensorBAddrLow = bmm1LocalInfo.GetValue(4);

        uint32_t bmm1BlockTableAddrHigh = bmm1LocalInfo.GetValue(5);
        uint32_t bmm1BlockTableAddrLow = bmm1LocalInfo.GetValue(6);
        uint64_t bmm1TensorBAddr =
            (static_cast<uint64_t>(bmm1TensorBAddrHigh) << 32) | static_cast<uint64_t>(bmm1TensorBAddrLow);
        uint64_t bmm1BlockTableAddr =
            (static_cast<uint64_t>(bmm1BlockTableAddrHigh) << 32) | static_cast<uint64_t>(bmm1BlockTableAddrLow);

        // PA 新增，BlockPosition 支持
        uint32_t bmm1BlockPositionAddrHigh = bmm1LocalInfo.GetValue(7);
        uint32_t bmm1BlockPositionAddrLow = bmm1LocalInfo.GetValue(8);
        uint64_t bmm1BlockPositionAddr =
            (static_cast<uint64_t>(bmm1BlockPositionAddrHigh) << 32) | static_cast<uint64_t>(bmm1BlockPositionAddrLow);

        uint32_t curActualSeqLenHigh = bmm1LocalInfo.GetValue(9);
        uint32_t curActualSeqLenLow = bmm1LocalInfo.GetValue(10);
        uint64_t curActualSeqLen =
            (static_cast<uint64_t>(curActualSeqLenHigh) << 32) | static_cast<uint64_t>(curActualSeqLenLow);




        // 添加调试日志：打印blockPosition地址信息
        // V5_DEBUG_PRINTF("bmm1CopyB1: BlockPositionAddr=%llu (High=%u, Low=%u)\n", 
        //                 bmm1BlockPositionAddr, bmm1BlockPositionAddrHigh, bmm1BlockPositionAddrLow);

        uint64_t s2BatchOffset = bmm1SInnerLoopIdx * singleProcessSInnerSize; // single块在当前batch的s2方向起始位置
        uint32_t startRow = col * bmm1BaseN;                                  // 在single块内偏移
        uint64_t curSeqIdx = s2BatchOffset + startRow;
        uint32_t copyFinishRowCnt = 0;
        uint64_t bmm1N2Offset = 0;
        if constexpr (LAYOUT_T == LAYOUT::BSH) {
            bmm1N2Offset = bmm1N2Idx * headSize;
        } else {
            bmm1N2Offset = bmm1N2Idx * headSize * kvCacheBlockSize; // BNSD 方向上偏移
        }

        GlobalTensor<KV_T> src;
        uint64_t tensorBTotalSize = (uint64_t)totalBlockNum * kvCacheBlockSize * kvHeadNum * headSize;
        src.SetGlobalBuffer((__gm__ KV_T *)bmm1TensorBAddr, tensorBTotalSize);
        LocalTensor<KV_T> dst = bMatrix.template ReinterpretCast<KV_T>();

        uint64_t blockIdBaseOffset = bmm1BIdx * maxBlockNumPerBatch;
        uint64_t blockPositionBaseOffset = bmm1BIdx * kvHeadNum * maxPositionNumPerBatch;

        // 添加调试日志：打印基础偏移信息
        //  V5_DEBUG_PRINTF("bmm1CopyB1: blockIdBaseOffset=%llu, blockPositionBaseOffset=%llu\n", 
        //                 blockIdBaseOffset, blockPositionBaseOffset);

        uint32_t blockElementCnt = BYTE_BLOCK / sizeof(KV_T);
        while (copyFinishRowCnt < useN) {
            uint64_t blockIdOffset = curSeqIdx / kvCacheBlockSize; // 获取block table上的索引
            uint64_t offsetInBlock = curSeqIdx % kvCacheBlockSize; // 获取在单个块上超出的行数

            uint32_t currentCopyRowCnt = kvCacheBlockSize - offsetInBlock;
            if (copyFinishRowCnt + currentCopyRowCnt > useN) { // S2方向上尾块处理
                currentCopyRowCnt = useN - copyFinishRowCnt;
            }

            // 添加blockPosition处理逻辑 - 最小修改
            if (bmm1BlockPositionAddr != 0) {
                // 计算在blockPosition中的偏移
                uint64_t positionOffset = blockPositionBaseOffset + 
                                         (uint64_t)(bmm1N2Idx * maxPositionNumPerBatch) + 
                                         blockIdOffset;

                // 添加调试日志：打印偏移计算信息
                //  V5_DEBUG_PRINTF("bmm1CopyB1: blockIdOffset=%llu, N2Idx=%u, positionOffset=%llu\n", 
                //                 blockIdOffset, bmm1N2Idx, positionOffset);
                
                // 修改：使用GlobalTensor的GetValue方法，与Vector核保持一致
                uint32_t newBlockIdOffset =
                    *(reinterpret_cast<__gm__ int32_t *>(bmm1BlockPositionAddr) + positionOffset);

                // 添加调试日志：打印读取的值
                //  V5_DEBUG_PRINTF("bmm1CopyB1: newBlockIdOffset=%d (0x%x)\n", newBlockIdOffset, newBlockIdOffset);

                // 如果newBlockIdOffset是无效值(0x7FFFFFFF)，则填充零值并继续
                if (newBlockIdOffset == 0x7FFFFFFF) {
                    uint64_t fix_length = 30; // 旧逻辑中固定的拷贝长度
                    // 1. 计算实际序列末尾 (curActualSeqLen) 所在的逻辑块和块内偏移
                    uint64_t final_logical_block_idx = curActualSeqLen / kvCacheBlockSize;
                    uint64_t final_row_offset_in_block = curActualSeqLen % kvCacheBlockSize;

                    // // 2. 两次间接寻址，找到末尾Token所在的物理块ID
                    // uint64_t final_positionOffset = blockPositionBaseOffset + 
                    //                                 (uint64_t)(bmm1N2Idx * maxPositionNumPerBatch) + 
                    //                                 final_logical_block_idx;
                    // uint32_t final_block_id_for_table = 
                    //     *(reinterpret_cast<__gm__ int32_t *>(bmm1BlockPositionAddr) + final_positionOffset);
                    
                     // 假设 final_block_id_for_table 是有效的
                    uint32_t final_physical_block_id = 
                        *(reinterpret_cast<__gm__ int32_t *>(bmm1BlockTableAddr) + blockIdBaseOffset + final_logical_block_idx);
                    
                    V5_DEBUG_PRINTF("bmm1CopyB1: Final Table: offset=%llu, blockId=%u\n", 
                            blockIdBaseOffset + final_logical_block_idx, final_physical_block_id);

                    // 3. 计算源地址的基地址
                    uint64_t final_src_base_offset =
                        (uint64_t)final_physical_block_id * kvCacheBlockSize * kvHeadNum * headSize +
                        bmm1N2Offset;
                    
                    // 4. 执行拷贝，同时保持新代码的K维度分块和CopyND2NZ格式
                    uint32_t alignedUseN = ((useN - 1 + ALIGN_BLOCK_SIZE) / ALIGN_BLOCK_SIZE) * ALIGN_BLOCK_SIZE;


                    // 根据不同分支进行零值填充
                    if (bmm1BaseN == kvCacheBlockSize) {
                        // 分支1的零值填充
                       CopyND2NZ(dst[copyFinishRowCnt * blockElementCnt], 
                          src[final_src_base_offset + row * bmm1BaseK],
                          final_row_offset_in_block, // 源矩阵的行偏移
                          0,                         // 源矩阵的列偏移
                          fix_length,                // 需要复制的实际行数
                          useK,                      // 要复制的列数
                          bmm1Kb,                    // 源矩阵的列步长
                          1, 0, alignedUseN); 
                    } else {
                         for (int i = 0; i < bmm1StepKb; i++) {
                            uint32_t remainColCnt = headSize - row * bmm1BaseK - i * bmm1BaseK;
                            uint32_t currentCopyColCnt = remainColCnt < bmm1BaseK ? remainColCnt : bmm1BaseK;
                            uint32_t dstOffset = copyFinishRowCnt * blockElementCnt;
                            dstOffset += i * bmm1BaseK * alignedUseN;

                            CopyND2NZ(dst[dstOffset], 
                                    src[final_src_base_offset + row * bmm1BaseK + i * bmm1BaseK], 
                                    final_row_offset_in_block, // 源矩阵的行偏移
                                    0,
                                    fix_length,                // 需要复制的实际行数
                                    currentCopyColCnt,
                                    bmm1Kb, 
                                    1, 0, 0, alignedUseN);
                        }
                    }

                    // 更新循环变量
                    copyFinishRowCnt += fix_length;
                    curSeqIdx += fix_length;
                    break;
                } 
                else {
                    // 修改：正确转换类型
                    blockIdOffset = static_cast<uint64_t>(newBlockIdOffset);
                    
                    // 添加调试日志：打印转换后的blockIdOffset
                    V5_DEBUG_PRINTF("bmm1CopyB1: 转换后blockIdOffset=%llu\n", blockIdOffset);
                }
            }

            uint32_t blockId = *(reinterpret_cast<__gm__ int32_t *>(bmm1BlockTableAddr) + blockIdBaseOffset +
                                 blockIdOffset); // 从block table上的获取编号
                                 
            // // 添加调试日志：打印从blockTable读取的blockId
            V5_DEBUG_PRINTF("bmm1CopyB1: blockTable读取: offset=%llu, blockId=%u\n", 
                            blockIdBaseOffset + blockIdOffset, blockId);

            uint64_t srcOffset =
                (uint64_t)blockId * kvCacheBlockSize * kvHeadNum * headSize + // 整个 blocksize在kv cache偏移
                bmm1N2Offset;                                                 // 多n，n方向上偏移

            uint32_t baseRowOffsetInSingle = col * bmm1BaseN; // 当前base起始点在single中偏移
            // 处理大包搬运时baseN < blocksize的情况，且需要考虑多组col共用某个block
            uint32_t baseRowOffsetInBlock = (baseRowOffsetInSingle + copyFinishRowCnt) % kvCacheBlockSize;
            // 大包模式，非尾块情况下useN = stepN * baseN，尾块情况下补齐 dstNzC0Stride 时需要16元素对齐

            uint32_t alignedUseN = ((useN - 1 + ALIGN_BLOCK_SIZE) / ALIGN_BLOCK_SIZE) * ALIGN_BLOCK_SIZE;

            if (bmm1BaseN == kvCacheBlockSize) { // bmm1BaseN = kvCacheBlockSize时不需要考虑k方向step，一次拷贝效率更高
                CopyND2NZ(dst[copyFinishRowCnt * blockElementCnt], src[srcOffset + row * bmm1BaseK],
                          0,                 // 源矩阵的行偏移(srcRowOffset)
                          0,                 // 源矩阵的列偏移(srcColOffset)
                          currentCopyRowCnt, // 需要复制的实际行数，考虑了边界情况
                          useK,              // 要复制的列数，即K维度大小
                          bmm1Kb,            // 源矩阵的列步长(srcColStride)
                          1,                 // 目标矩阵的行步长(dstRowStride)
                          0,                 // 目标矩阵的列步长(dstColStride)
                          alignedUseN);      // 目标矩阵的列数，考虑了对齐
                                             // 矩阵计算是按有宽高的小块搬的，所以需要考虑srcRowOffset和srcColOffset
            } else {
                for (int i = 0; i < bmm1StepKb; i++) { // K方向多Step
                    uint32_t alignedCurrentCopyRowCnt =
                        (currentCopyRowCnt + blockElementCnt - 1) / blockElementCnt * blockElementCnt;
                    // K方向上尾块，需要特殊处理拷贝列数
                    uint32_t remainColCnt = headSize - row * bmm1BaseK - i * bmm1BaseK;
                    uint32_t currentCopyColCnt = remainColCnt < bmm1BaseK ? remainColCnt : bmm1BaseK;
                    uint32_t dstOffset = copyFinishRowCnt * blockElementCnt;
                    // Kb方向多step时，算dst L1上偏移时N方向需要考虑完整的useN，且需要对齐处理
                    dstOffset += i * bmm1BaseK * alignedUseN;
                    CopyND2NZ(dst[dstOffset], src[srcOffset + row * bmm1BaseK + i * bmm1BaseK], baseRowOffsetInBlock, 0,
                              currentCopyRowCnt, currentCopyColCnt, bmm1Kb, 1, 0, 0, alignedUseN);
                }
            }
            
            V5_DEBUG_PRINTF("bmm1CopyB1: Compute Finish: offset=%llu, blockId=%u \n",
                blockIdBaseOffset + blockIdOffset, blockId);
            // 更新循环变量
            copyFinishRowCnt += currentCopyRowCnt;
            curSeqIdx += currentCopyRowCnt;
        }
    }


    // bmm2 回调，row方向对应k、s2；col方向对应n、d
    static __aicore__ void bmm2CopyB1(const LocalTensor<int8_t> &bMatrix, const __gm__ void *gm, int row, int col,
                                      int useK, int useN, const uint64_t tilingPtr, const uint64_t dataPtr)
    {
        // 回调函数，当前有2种方式获取 TilingData：
        // (1) 路径3，在线编译，此时 tilingDataPtr 为空，但SparsePagedFusionAttentionTilingDataV2结构体中各成员默认值即为tiling结果
        // (2) 其它场景，tilingDataPtr 非空，从其指向的GM内存中获取 tiling data，但tilingDataPtr 需要在vector侧配置给cube
        SparsePagedFusionAttentionTilingDataV2 allTilingDataV2;
        SparsePagedFusionAttentionTilingData allTilingData = allTilingDataV2.tilingBase;
        SparsePagedFusionAttentionTilingData *tilingDataPtr = reinterpret_cast<SparsePagedFusionAttentionTilingData *>(tilingPtr);
        if (tilingDataPtr != nullptr) {
            allTilingData = *tilingDataPtr;
        }
        uint32_t maxBlockNumPerBatch = allTilingData.baseParams.maxBlockNumPerBatch;
        uint32_t maxPositionNumPerBatch = allTilingData.baseParams.maxPositionNumPerBatch;
        uint64_t singleProcessSInnerSize = allTilingData.sparsePagedFusionAttentionSingleCoreParams.singleProcessSInnerSize;
        uint64_t kvCacheBlockSize = allTilingData.baseParams.blockSize;
        uint32_t totalBlockNum = allTilingData.baseParams.totalBlockNum;
        uint32_t headSize = allTilingData.baseParams.headSize;
        uint32_t kvHeadNum = allTilingData.baseParams.kvHeadNum;
        uint32_t bmm2BaseK = allTilingData.bmm2TilingData.baseK;
        uint32_t bmm2N = allTilingData.bmm2TilingData.N;
        uint32_t bmm2StepK = allTilingData.bmm2TilingData.stepKb;
        uint32_t bmm2BaseN = allTilingData.bmm2TilingData.baseN;
        uint32_t bmm2StepN = allTilingData.bmm2TilingData.stepN;

        GlobalTensor<uint32_t> bmm2LocalInfo;
        bmm2LocalInfo.SetGlobalBuffer((__gm__ uint32_t *)dataPtr, 16); // 修改为9，支持读取blockPosition地址

        uint32_t bmm2BIdx = bmm2LocalInfo.GetValue(0);
        uint32_t bmm2N2Idx = bmm2LocalInfo.GetValue(1);
        uint32_t bmm2SInnerLoopIdx = bmm2LocalInfo.GetValue(2);
        // DataCopy 不支持64位拷贝，2个gm地址需在V侧设置时拆分，在回调里拼接
        uint32_t bmm2TensorBAddrHigh = bmm2LocalInfo.GetValue(3);
        uint32_t bmm2TensorBAddrLow = bmm2LocalInfo.GetValue(4);
        uint32_t bmm2BlockTableAddrHigh = bmm2LocalInfo.GetValue(5);
        uint32_t bmm2BlockTableAddrLow = bmm2LocalInfo.GetValue(6);
        uint32_t bmm2BlockPositionAddrHigh = bmm2LocalInfo.GetValue(7); // 新增：获取blockPosition地址高32位
        uint32_t bmm2BlockPositionAddrLow = bmm2LocalInfo.GetValue(8);  // 新增：获取blockPosition地址低32位
        
        uint64_t bmm2TensorBAddr =
            (static_cast<uint64_t>(bmm2TensorBAddrHigh) << 32) | static_cast<uint64_t>(bmm2TensorBAddrLow);
        uint64_t bmm2BlockTableAddr =
            (static_cast<uint64_t>(bmm2BlockTableAddrHigh) << 32) | static_cast<uint64_t>(bmm2BlockTableAddrLow);
        uint64_t bmm2BlockPositionAddr = // 新增：合并blockPosition地址
            (static_cast<uint64_t>(bmm2BlockPositionAddrHigh) << 32) | static_cast<uint64_t>(bmm2BlockPositionAddrLow);

        uint32_t curActualSeqLenHigh = bmm2LocalInfo.GetValue(9);
        uint32_t curActualSeqLenLow = bmm2LocalInfo.GetValue(10);
        uint64_t curActualSeqLen =
            (static_cast<uint64_t>(curActualSeqLenHigh) << 32) | static_cast<uint64_t>(curActualSeqLenLow);


        // // 添加调试日志：打印blockPosition地址信息
        //  V5_DEBUG_PRINTF("bmm2CopyB1: BlockTableAddr=%llu (High=%u, Low=%u)\n", bmm2BlockTableAddr,
        //                 bmm2BlockTableAddrHigh, bmm2BlockTableAddrLow);


        // // 添加调试日志：打印blockPosition地址信息
        //  V5_DEBUG_PRINTF("bmm2CopyB1: BlockPositionAddr=%llu (High=%u, Low=%u)\n", 
        //                 bmm2BlockPositionAddr, bmm2BlockPositionAddrHigh, bmm2BlockPositionAddrLow);

        uint64_t s2BatchOffset = bmm2SInnerLoopIdx * singleProcessSInnerSize;
        uint32_t startRow = row * bmm2BaseK;
        uint64_t curSeqIdx = s2BatchOffset + startRow;
        uint32_t copyFinishRowCnt = 0;
        uint64_t bmm2N2Offset = 0;
        if constexpr (LAYOUT_T == LAYOUT::BSH) {
            bmm2N2Offset = bmm2N2Idx * headSize;
        } else {
            bmm2N2Offset = bmm2N2Idx * headSize * kvCacheBlockSize;
        }

        GlobalTensor<KV_T> src;
        uint64_t tensorBTotalSize = (uint64_t)totalBlockNum * kvCacheBlockSize * kvHeadNum * headSize;
        src.SetGlobalBuffer((__gm__ KV_T *)bmm2TensorBAddr, tensorBTotalSize);
        LocalTensor<KV_T> dst = bMatrix.template ReinterpretCast<KV_T>();
        uint64_t blockIdBaseOffset = bmm2BIdx * maxBlockNumPerBatch;
        uint64_t blockPositionBaseOffset = bmm2BIdx * kvHeadNum * maxPositionNumPerBatch;
        
        // // 添加调试日志：打印基础偏移信息
        //  V5_DEBUG_PRINTF("bmm2CopyB1: blockIdBaseOffset=%llu, blockPositionBaseOffset=%llu\n", 
        //                 blockIdBaseOffset, blockPositionBaseOffset);

        // 分块拷贝,块数为ndNum
        uint32_t blockElementCnt = 32 / sizeof(KV_T);
        // 考虑S2方向上不对齐场景，dst 多个分形之间的间隔需要考虑对齐
        uint32_t alignedUseK = ((useK - 1 + blockElementCnt) / blockElementCnt) * blockElementCnt;
        while (copyFinishRowCnt < useK) {
            uint64_t blockIdOffset = curSeqIdx / kvCacheBlockSize; // 获取block table上的索引
            uint64_t offsetInBlock = curSeqIdx % kvCacheBlockSize; // 获取在单个block块上超出的行数

            uint32_t currentCopyRowCnt = kvCacheBlockSize - offsetInBlock;
            if (copyFinishRowCnt + currentCopyRowCnt > useK) { // S2方向上尾块处理
                currentCopyRowCnt = useK - copyFinishRowCnt;
            }

            // 添加blockPosition处理逻辑 - 最小修改
            if (bmm2BlockPositionAddr != 0) {
                // 计算在blockPosition中的偏移
                uint64_t positionOffset = blockPositionBaseOffset + 
                                         (uint64_t)(bmm2N2Idx * maxPositionNumPerBatch) + 
                                         blockIdOffset;
                
                // // 添加调试日志：打印偏移计算信息
                //  V5_DEBUG_PRINTF("bmm2CopyB1: blockIdOffset=%llu, N2Idx=%u, positionOffset=%llu\n", 
                //                 blockIdOffset, bmm2N2Idx, positionOffset);
                
                // 修改：使用GlobalTensor的GetValue方法，与Vector核保持一致
                uint32_t newBlockIdOffset =
                    *(reinterpret_cast<__gm__ int32_t *>(bmm2BlockPositionAddr) + positionOffset);

                // // 添加调试日志：打印读取的值
                //  V5_DEBUG_PRINTF("bmm2CopyB1: newBlockIdOffset=%d (0x%x)\n", newBlockIdOffset, newBlockIdOffset);

                // 如果blockIdOffset是无效值(0x7FFFFFFF)，则填充零值并继续
                if (newBlockIdOffset == (uint64_t)(0x7FFFFFFF)) {
                    // MDL下，每次回调拷贝一组step*base，每次都是从头开始拷贝，只需要关注copyFinishRowCnt
                    uint64_t fix_length = 30; // 旧逻辑中固定的拷贝长度

                    // 1. 计算实际序列末尾 (curActualSeqLen) 所在的逻辑块和块内偏移
                    uint64_t final_logical_block_idx = curActualSeqLen / kvCacheBlockSize;
                    uint64_t final_row_offset_in_block = curActualSeqLen % kvCacheBlockSize;
                    
                    // // 2. 两次间接寻址，找到末尾Token所在的物理块ID (使用bmm2的变量)
                    // uint64_t final_positionOffset = blockPositionBaseOffset + 
                    //                                 (uint64_t)(bmm2N2Idx * maxPositionNumPerBatch) + 
                    //                                 final_logical_block_idx;
                    // uint32_t final_block_id_for_table = 
                    //     *(reinterpret_cast<__gm__ int32_t *>(bmm2BlockPositionAddr) + final_positionOffset);
                    
                    // 假设 final_block_id_for_table 是有效的
                    uint32_t final_physical_block_id = 
                        *(reinterpret_cast<__gm__ int32_t *>(bmm2BlockTableAddr) + blockIdBaseOffset + final_logical_block_idx);

                    // 3. 计算源地址的基地址 (使用bmm2的变量)
                    uint64_t final_src_base_offset =
                        (uint64_t)final_physical_block_id * kvCacheBlockSize * kvHeadNum * headSize +
                        bmm2N2Offset;
                    // 同样考虑D方向(col)的偏移
                    if (useN < headSize) {
                        final_src_base_offset += col * bmm2BaseN * bmm2StepN;
                    }

                    // 4. 执行拷贝，使用本函数（bmm2）的CopyND2NZ格式
                    uint32_t dstOffset = copyFinishRowCnt * blockElementCnt;
                    CopyND2NZ(dst[dstOffset], 
                            src[final_src_base_offset], 
                            final_row_offset_in_block, // 源矩阵的行偏移
                            0, 
                            fix_length,                // 需要复制的实际行数
                            useN,                      // 要复制的列数
                            bmm2N, 
                            1, 0, 0,
                            alignedUseK);
                    
                    // 5. 更新循环变量
                    copyFinishRowCnt += fix_length;
                    curSeqIdx += fix_length;

                    // 6. 按照旧版逻辑，直接终止整个while循环
                    break; 
                }else{
                    blockIdOffset = static_cast<uint64_t>(newBlockIdOffset);
                    
                    // 添加调试日志：打印转换后的blockIdOffset
                    //  V5_DEBUG_PRINTF("bmm2CopyB1: 转换后blockIdOffset=%llu\n", blockIdOffset);
                }
            }

            uint32_t blockId = *(reinterpret_cast<__gm__ int32_t *>(bmm2BlockTableAddr) + blockIdBaseOffset +
                                 blockIdOffset); // 从block table上的获取编号

            // 添加调试日志：打印从blockTable读取的blockId
            //  V5_DEBUG_PRINTF("bmm2CopyB1: blockTable读取: offset=%llu, blockId=%u\n", 
                            // blockIdBaseOffset + blockIdOffset, blockId);

            uint64_t srcOffset = (uint64_t)blockId * kvCacheBlockSize * kvHeadNum * headSize + // 整个 blocksize 偏移
                                 bmm2N2Offset; // 多n，n方向上偏移
            // MDL模板， bmm2 D方向可能有多个step，因此尾块的srcOffset需要考虑D方向偏移，D方向尾块标志：useN < headSize
            if (useN < headSize) {
                srcOffset += col * bmm2BaseN * bmm2StepN;
            }

            uint32_t baseRowOffsetInSingle = row * bmm2BaseK;
            // 考虑一组stepK * baseK 跨多个block情况
            // 1. stepK * baseK的起点在block起始，但拷贝跨block  2. stepK * baseK的起点在block中间位置，但拷贝跨block
            uint32_t baseRowOffsetInBlock = (baseRowOffsetInSingle + copyFinishRowCnt) % kvCacheBlockSize;

            // MDL下，每次回调拷贝一组step*base，每次都是从头开始拷贝，只需要关注copyFinishRowCnt
            uint32_t dstOffset = copyFinishRowCnt * blockElementCnt;

            CopyND2NZ(dst[dstOffset], src[srcOffset], baseRowOffsetInBlock, 0, currentCopyRowCnt, useN, bmm2N, 1, 0, 0,
                      alignedUseK);

            // 更新循环变量
            copyFinishRowCnt += currentCopyRowCnt;
            curSeqIdx += currentCopyRowCnt;
        }
    }

    // define matmul1
    typedef MatmulType<TPosition::GM, CubeFormat::ND, KV_T, true> b1Type;
    typedef MatmulType<TPosition::GM, CubeFormat::ND, float> bias1Type;
    typedef MatmulType<TPosition::GM, CubeFormat::ND_ALIGN, MM_OUT_T> c1Type;
    using mm1Type = typename AscendC::Conditional<PAGE_ATTENTION,
                                                  Matmul<AType, b1Type, c1Type, bias1Type, CFG_MDL_EXCEED_INIT_CALLBACK,
                                                         matmul::MatmulCallBackFunc<nullptr, nullptr, bmm1CopyB1>>,
                                                  Matmul<AType, b1Type, c1Type, bias1Type, CFG_MDL_EXCEED_INIT>>::type;
    mm1Type mm;

    // define matmul2
    typedef MatmulType<TPosition::GM, CubeFormat::VECTOR, KV_T, false> a2Type;
    typedef MatmulType<TPosition::GM, CubeFormat::ND, KV_T, false> b2Type;
    typedef MatmulType<TPosition::GM, CubeFormat::ND, float> bias2Type;
    typedef MatmulType<TPosition::GM, CubeFormat::ND, MM_OUT_T> c2Type;

    using mm2Type = typename AscendC::Conditional<PAGE_ATTENTION,
                                                  Matmul<AType, b2Type, c2Type, bias2Type, CFG_MDL_EXCEED_INIT_CALLBACK,
                                                         matmul::MatmulCallBackFunc<nullptr, nullptr, bmm2CopyB1>>,
                                                  Matmul<AType, b2Type, c2Type, bias2Type, CFG_NORM_EXCEED_INIT>>::type;
    mm2Type bmm2;

    mm1Type mm1Sp;
    mm2Type mm2Sp;

protected:
    const SparsePagedFusionAttentionTilingData *__restrict tilingData = nullptr;
    TPipe *pipe = nullptr;

    GlobalTensor<Q_T> queryGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> valueGm;
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<float> softmaxLseGm;

    // atten mask
    GlobalTensor<bool> attenMaskBoolGm;

    // PSE
    GlobalTensor<pseShiftType> pseShiftGm;

    // antiquant
    GlobalTensor<ANTIQ_PARAMS_T_KEY> keyAntiqOffsetGm;
    GlobalTensor<ANTIQ_PARAMS_T_KEY> keyAntiqScaleGm;
    GlobalTensor<ANTIQ_PARAMS_T_VALUE> valueAntiqOffsetGm;
    GlobalTensor<ANTIQ_PARAMS_T_VALUE> valueAntiqScaleGm;
    GlobalTensor<uint64_t> actualSeqLengthsGm;
    // out quant
    GlobalTensor<float> quantScale2Gm;
    GlobalTensor<float> quantOffset2Gm;
    GlobalTensor<bfloat16_t> quantScale2Bf16Gm;
    GlobalTensor<bfloat16_t> quantOffset2Bf16Gm;
    // workspace
    GlobalTensor<KV_T> queryPreProcessResGm;
    GlobalTensor<Q_T> prefixQueryPreProcessResGm;
    GlobalTensor<MM_OUT_T> mm1ResGm;
    GlobalTensor<KV_T> vec1ResGm;
    GlobalTensor<MM_OUT_T> mm2ResGm;
    GlobalTensor<T> vec2ResGm;
    GlobalTensor<T> accumOutGm;
    GlobalTensor<T> lseSumFdGm;
    GlobalTensor<T> lseMaxFdGm;

    GlobalTensor<uint32_t> bmm1CallBackDataGm;
    GlobalTensor<uint32_t> bmm2CallBackDataGm;

    // kv_left_padding
    GlobalTensor<int64_t> kvPaddingSizeGm;

    // cent_select 相关全局张量
    GlobalTensor<half> queryGlobal;
    GlobalTensor<half> l1CentGlobal;
    GlobalTensor<int32_t> blockIdsGlobal;
    GlobalTensor<int32_t> blockTableGlobal;
    GlobalTensor<int32_t> seqLenGlobal;
    GlobalTensor<int32_t> blockPositionGlobal;
    GlobalTensor<int32_t> pagePositionLengthGlobal;
    GlobalTensor<int64_t> maxPagePositionLengthGlobal;

    // cent_select 相关队列
    TQue<QuePosition::VECIN, 1> inQuery;
    TQue<QuePosition::VECIN, 1> inL1Cent;
    TQue<QuePosition::VECIN, 1> inBlockIds;
    TQue<QuePosition::VECIN, 1> inBlockTable;
    TQue<QuePosition::VECOUT, 1> outPagePosition;
    TQue<QuePosition::VECOUT, 1> outPagePositionLength;
    TQue<QuePosition::VECOUT, 1> outMaxPagePositionLength;
    
    // cent_select 相关 TBuf
    TBuf<> tmpBmm1ResBuff;
    TBuf<> bmm1ResBuff;

    // TopK 相关 TBuf
    TBuf<QuePosition::VECCALC> topKDstValue;
    TBuf<QuePosition::VECCALC> topKDstIndex;
    TBuf<QuePosition::VECCALC> topKSrcIndexLocal;
    TBuf<QuePosition::VECCALC> topKWrokLocal;
    TBuf<QuePosition::VECCALC> topKFinishLocal;

    // 位置选择相关 TBuf
    TBuf<QuePosition::VECCALC> tmpBuffPageBatch;
    TBuf<QuePosition::VECCALC> tmpBuffSelectReduce;
    TBuf<QuePosition::VECCALC> tmpBuffSelectTmp;
    TBuf<QuePosition::VECCALC> selectBlockIdsIndexLocal;

    // 最大页面位置长度相关 TBuf
    TBuf<QuePosition::VECCALC> totalPagePositionLength;
    TBuf<QuePosition::VECCALC> totalPagePositionLengthFloat;
    TBuf<QuePosition::VECCALC> outMaxPagePositionLengthInt32;
    TBuf<QuePosition::VECCALC> maxWorkLocal;
    TBuf<QuePosition::VECCALC> dstMaxLocal;

    // cent_select 相关参数
    // base
    // int32_t blockIdx = 0;
    // compute cent
    int32_t clusterNum = 0;
    int32_t dimNum = 0;
    int32_t clusterBlockNum = 0;
    int32_t clusterBlockSize = 0;
    // select position
    int32_t kvPageLen = 0;
    int32_t maxBatch = 0;
    int32_t maxPage = 0;
    int32_t maxPageNum = 0;
    // topk
    int32_t k = 0;
    int32_t tmpsize = 0;
    int32_t maxWorkSize = 0;

    // cent_select 计算相关参数
    int32_t pagePositionLength = 0;
    int32_t seqLen = 0;
    float32_t importance_ = 0;
    int32_t pageLen = 0;
    int32_t gatherMaskLen = 0;
    int32_t workLoadThreshold = 0;
    int32_t gatherMaskU32Len = 0;
    int32_t kvBlockSize = 0;

    // TopK 相关
    TopkTiling topkTilingData;
    TopKInfo topKInfo;

    // queue
    TQue<QuePosition::VECIN, 1> inputQue1;   // 32K, inque
    TQue<QuePosition::VECIN, 1> inputQue2;   // 16K, inque
    TQue<QuePosition::VECOUT, 1> outputQue1; // 32K, outque
    TQue<QuePosition::VECOUT, 1> outputQue2; // 8K, outque

    // 临时tbuf
    TBuf<> tmpBuff1; // 32K
    TBuf<> tmpBuff2; // 32K
    TBuf<> tmpBuff3; // 2K

    // 常驻tbuf
    TBuf<> antiqScaleBuff;            // 4K
    TBuf<> antiqOffsetBuff;           // 4K
    TBuf<> qAmaxBuff;                 // 2K + 256B
    TBuf<> softmaxResAmaxBuff;        // 2K + 256B
    TBuf<> qRowSumBuff;               // 2K + 256B
    TBuf<> softmaxResRowSumBuff;      // 2K + 256B
    TBuf<> softmaxMaxBuff;            // 2K
    TBuf<> softmaxExpBuff;            // 2K
    TBuf<> softmaxSumBuff;            // 2K
    TBuf<> bmm1PageAttentionDataBuff; // 64B
    TBuf<> bmm2PageAttentionDataBuff; // 64B

    LocalTensor<T> softmaxMaxUb;
    LocalTensor<T> softmaxSumUb;
    LocalTensor<T> softmaxExpUb;

    LocalTensor<uint32_t> bmm1PageAttentionDataUb;
    LocalTensor<uint32_t> bmm2PageAttentionDataUb;

    // antiquant msd
    LocalTensor<T> aMaxBmm1Ub;
    LocalTensor<T> aMaxBmm2Ub;
    LocalTensor<T> softmaxResRowSumUb;
    LocalTensor<T> softmaxScaleResRowSumUb;
    LocalTensor<T> antiqScaleUb;
    LocalTensor<T> antiqOffsetUb;
    LocalTensor<T> qRowSumUb;

    // sys prefix tmpBuffer
    GlobalTensor<T> sysPrefixAttenOutGm;
    GlobalTensor<T> usrPromptAttenOutGm;
    GlobalTensor<T> lseSumGm;
    GlobalTensor<T> lseMaxGm;
    GlobalTensor<T> msdRowMax1Gm;
    GlobalTensor<T> msdRowMax2Gm;
    GlobalTensor<T> msdRowSum1Gm;
    GlobalTensor<T> msdRowSum2Gm;
    GlobalTensor<T> softmaxRowMaxGm;
    GlobalTensor<T> softmaxRowSumGm;
    GlobalTensor<T> softmaxRowExpGm;

    uint64_t msdRowMaxSize = 0;
    uint64_t msdRowSumSize = 0;
    uint64_t softmaxMaxSumExpSize = 0;

    uint64_t sysPrefixLen = 0;
    uint32_t formerCoreNumSp = 0;
    uint32_t blockSplitBn2RangeSp = 0;
    uint32_t tailBlockSplitBn2RangeSp = 0;
    uint32_t usedCoreNumSp = 0;
    bool calcSysPrefixFlag = false;
    uint32_t batchSizeQ = 0;

    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T);
    static constexpr uint32_t REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(T);
    static constexpr uint32_t BASE_BLOCK_MAX_ELEMENT_NUM = BUFFER_SIZE_BYTE_32K / sizeof(T);
    static constexpr uint32_t ADDRESS_ALIGN_NUM = 512 / sizeof(KV_T);
    static constexpr uint32_t ADDRESS_ALIGN_NUM_THRESHLOD = 128 / sizeof(KV_T);
    static constexpr T antiquantExpandCoeff = KVINT4 ? 14.98 : 254;
    static constexpr T antiqCoeff1 = KVINT4 ? 7.49 : 127;
    static constexpr T antiqCoeff2 = 1 / antiqCoeff1;
    static constexpr T SOFTMAX_MIN_NUM = -2e38;
    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    static constexpr T FP16_ATTEN_MASK_SCALAR_VALUE = -10000;           // 用于mask为fp16类型
    bool antiqOffsetExistFlag = false;
    uint32_t msdIterNum = 0U;
    bool antiquantPerHeadFlag = false;
    bool antiquantParamsInPagedAttentionFlag = false;
    uint32_t antiquantPerTensorFlag = 0U;
    uint64_t sUnitSize = 0;

    // cent_select 相关常量
    static constexpr uint32_t BUFFER_NUM = 1;
    static constexpr int32_t tplPadding = 8; // padding to 32B for int32
    static constexpr int32_t tplPaddingHalf = 4; // padding to 32B for int64
    static constexpr int32_t PAGESIZE = 128;


    // kv_left_padding
    uint32_t kvPaddingFlag = 0;
    uint64_t kvPaddingBeginOffset = 0;

    // for workspace pingpong
    const uint32_t dbWorkspaceRatio = 1;

    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;

    __gm__ uint8_t *key_ = nullptr;
    __gm__ uint8_t *value_ = nullptr;

    uint32_t tmpBlockIdx = 0U;
    __gm__ uint8_t *blocktablePtr = nullptr;

    __gm__ uint8_t *blockpositionPtr = nullptr;
    bool useBlockPosition = false;

    __gm__ uint32_t *bmm1CallBackDataPtr = nullptr;
    __gm__ uint32_t *bmm2CallBackDataPtr = nullptr;

    // tilingdata
    uint64_t singleProcessSInnerSize = 0U;
    uint32_t sInnerLoopTimes = 0U;
    uint64_t singleProcessSInnerSizeTail = 0U;
    uint32_t formerCoreNum = 0U;
    uint32_t usedCoreNum = 0U;
    uint32_t bIdx = 0U;
    uint32_t n1Idx = 0U;
    uint32_t n2Idx = 0U;

    uint32_t mmResUbSize = 0U;
    uint32_t bmm2ResUbSize = 0U;
    uint32_t batchContinuous = 0U;

    uint64_t batchSize = 0ULL;
    uint64_t qHeadNum = 0ULL;
    uint64_t kvHeadNum = 0ULL;
    uint64_t gSize = 0ULL;
    uint64_t kvSeqSize = 0ULL;
    uint64_t headDim = 0ULL;
    uint64_t headDimAlign = 0ULL;

    // 是否返回lse
    bool softmaxLseFlag;

    // attention mask
    bool attenMaskFlag = false;
    uint32_t selectWithByteMaskTmpMinSize = 0U;
    uint32_t attenMaskSizeAlign = 0U;
    // pse mask
    bool pseShiftFlag = false;
    uint32_t pseShiftB = 0U;
    uint32_t pseShiftS = 0U;
    uint64_t pseShiftOffset = 0U;
    uint64_t pseShiftCoreOffset = 0ULL;
    uint32_t pseMaskSizeAlign = 0U;
    // offset
    uint64_t tensorACoreOffset = 0ULL;
    uint64_t tensorBCoreOffset = 0ULL;
    uint64_t tensorBOffset = 0ULL;
    uint64_t valueOffset = 0ULL;
    uint64_t attenOutOffset = 0ULL;
    uint64_t antiqParamOffset = 0ULL;
    uint64_t attenMaskOffset = 0ULL;
    uint64_t attenMaskCoreOffset = 0ULL;
    uint64_t antiqKeyParamCoreOffsetPerToken = 0ULL;
    uint64_t antiqParamOffsetPerToken = 0ULL;
    uint64_t attenMaskSize = 0ULL;
    uint64_t antiqSeqSize = 0ULL;

    // splitKV
    uint32_t splitKVNum = 0U;
    uint32_t s2Idx = 0U;
    uint64_t sInnerLoopSize = 0ULL;
    uint32_t actualCombineLoopSize = 0U;
    uint64_t combineLseOffset = 0ULL;
    uint64_t combineAccumOutOffset = 0ULL;
    bool flashDecodeFlag = false;

    uint64_t curActualSeqLen = 0ULL;
    uint64_t curSingleProcessSInnerSizeAlign = 0ULL;
    uint64_t actualSingleProcessSInnerSize = 0ULL;
    uint64_t actualSingleProcessSInnerSizeAlign = 0ULL;
    uint32_t beforeBlockSplitBn2Nums = 0U;
    uint32_t bn2LoopTimes = 0U;

    uint32_t actualLenDims = 0U;
    // out quant
    bool isPerChnU8Out = false;
    bool isOutQuantTypeBf16 = false;
    float quantScale2Value = 0;
    float quantOffset2Value = 0;
    bool isQuantOffset2Exist = false;
    uint64_t perChannelQuantOffset = 0ULL;

    bool curActSeqLenIsZero = false;
    // PA
    const uint32_t mmPACallBackDataSize = 64U;

    template <typename T> __aicore__ inline T Align(T num, T rnd)
    {
        return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
    }
    __aicore__ inline void InitTilingData();
    __aicore__ inline void InitCalcParams();
    __aicore__ inline void InitCalcParamsEach();
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengths);
    __aicore__ inline void GetActualSeqLen();
    __aicore__ inline void UpdateInnerLoopCond();
    __aicore__ inline void CalculateSUnitSize();
    __aicore__ inline bool ComputeKVPaddingBeginOffset();

    __aicore__ inline void GetBN2id(const uint32_t bn2Idx);
    __aicore__ inline void CalcBN2Offset();
    __aicore__ inline void CalcBN2Params();

    __aicore__ inline void CalcSInnerOffsetAndParams(const uint32_t sInnerLoopIdx);
    __aicore__ inline void UpdateOffsetsVec(uint32_t sInnerLoopIdx);

    __aicore__ inline void AttenMaskCopyIn(uint64_t offset, uint32_t dealRowCount, uint32_t actualColumnCount);

    __aicore__ inline void CopyAntiquantScale(LocalTensor<T> &castUb, GlobalTensor<Q_T> srcGm, uint64_t offset);

    __aicore__ inline void CopyAntiquantParamsPerToken(GlobalTensor<ANTIQ_PARAMS_T_VALUE> srcGm, uint64_t offset,
                                                       uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void CopyAntiquantParamsPerTokenHead(GlobalTensor<ANTIQ_PARAMS_T_VALUE> srcGm, uint64_t offset,
                                                           uint32_t columnCount);
    __aicore__ inline void CopyAntiquantParamsParamsPagedAttention(GlobalTensor<ANTIQ_PARAMS_T_VALUE> srcGm,
                                                                           uint64_t offset, uint32_t actualColumnCount);
    __aicore__ inline void CopyAntiquantParamsParamsPagedAttentionImpl(GlobalTensor<ANTIQ_PARAMS_T_VALUE> srcGm,
                                                                           uint64_t offset, uint32_t actualColumnCount,
                                                                           uint32_t useKvHeadNum, uint32_t useN2Idx);
    __aicore__ inline void CopyAntiqQuery(LocalTensor<T> &queryCastUb, uint64_t qOffset, uint32_t dealRowCount,
                                          uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void AbsRowMax(LocalTensor<T> &tmpAMaxRes, LocalTensor<T> &srcUb, LocalTensor<T> tmpAUb,
                                     uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void AntiquantAIterExpand(GlobalTensor<KV_T> dstGm, LocalTensor<T> &tmpA1, LocalTensor<T> &tmpA2,
                                                uint32_t calcSize, bool isFirst, uint64_t outOffset);
    __aicore__ inline void AntiquantMatmulPreProcess(GlobalTensor<KV_T> dstGm, LocalTensor<T> aMaxResUb,
                                                     LocalTensor<T> srcUb, LocalTensor<T> tmpAFloorUb,
                                                     uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                     uint32_t actualColumnCount);
    __aicore__ inline void AntiquantSoftmaxResPreProcess(GlobalTensor<KV_T> dstGm, LocalTensor<T> srcUb,
                                                         LocalTensor<T> tmpAFloorUb, uint32_t startRow,
                                                         uint32_t dealRowCount, uint32_t columnCount,
                                                         uint32_t actualColumnCount);
    __aicore__ inline void DealQueryPreProcessBaseBlock(uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                        uint32_t actualColumnCount);
    __aicore__ inline void DealQueryPreProcessBaseBlockPerToken(uint32_t startRow, uint32_t dealRowCount,
                                                                uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void QueryPreProcess();
    __aicore__ inline void QueryPreProcessPerToken();
    __aicore__ inline void QueryPreProcessInner();
    __aicore__ inline void QueryPreProcessPerTokenInner();
    __aicore__ inline void SysPrefixQueryPreProcess();
    __aicore__ inline void SysPrefixQueryPreProcessInner();

    __aicore__ inline void FlashDecodeCompute();
    __aicore__ inline void SetMMOrgShape();
    __aicore__ inline void SetMMOrgShapeCommon();
    __aicore__ inline void SysPrefixSetMMOrgShape();
    __aicore__ inline void Bmm1Compute(const uint32_t bn2Idx, const uint32_t sInnerLoopIdx);
    __aicore__ inline void Bmm2Compute(const uint32_t bn2Idx, const uint32_t sInnerLoopIdx);
    __aicore__ inline void Bmm1ComputeCommon(const uint32_t bn2Idx, const uint32_t sInnerLoopIdx);
    __aicore__ inline void Bmm2ComputeCommon(const uint32_t bn2Idx, const uint32_t sInnerLoopIdx);
    __aicore__ inline void SysPrefixBmm1Compute(const uint32_t bn2Idx, const uint32_t sInnerLoopIdx);
    __aicore__ inline void SysPrefixBmm2Compute(const uint32_t bn2Idx, const uint32_t sInnerLoopIdx);

    __aicore__ inline void DealBmm1ResBaseBlock(const uint32_t sInnerLoopIdx, uint32_t startRow, uint32_t dealRowCount,
                                                uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void DealAntiqBmm1ResBaseBlock(const uint32_t sInnerLoopIdx, uint32_t startRow,
                                                     uint32_t dealRowCount, uint32_t columnCount,
                                                     uint32_t actualColumnCount);
    __aicore__ inline void DealAntiqBmm1ResBaseBlockPerToken(const uint32_t sInnerLoopIdx, uint32_t startRow,
                                                             uint32_t dealRowCount, uint32_t columnCount,
                                                             uint32_t actualColumnCount);
    __aicore__ inline void DealAntiqBmm1ResBaseBlockChannelToken(const uint32_t sInnerLoopIdx, uint32_t startRow,
                                                                 uint32_t dealRowCount, uint32_t columnCount,
                                                                 uint32_t actualColumnCount);
    __aicore__ inline void AntiquantMatmulResCombine(LocalTensor<T> bmmResUb, GlobalTensor<MM_OUT_T> srcGm,
                                                     uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                     uint32_t actualColumnCount);
    __aicore__ inline void ProcessVec1(const uint32_t sInnerLoopIdx);
    __aicore__ inline void ProcessVec1Inner(const uint32_t sInnerLoopIdx);
    __aicore__ inline void PreProcessVec1(uint32_t sInnerLoopIdx);
    __aicore__ inline void PostProcessVec1();

    __aicore__ inline void DealBmm2ResBaseBlock(const uint32_t sInnerLoopIdx, uint32_t startRow, uint32_t dealRowCount,
                                                uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void DealAntiqBmm2ResBaseBlock(const uint32_t sInnerLoopIdx, uint32_t startRow,
                                                     uint32_t dealRowCount, uint32_t columnCount,
                                                     uint32_t actualColumnCount);
    __aicore__ inline void DealAntiqBmm2ResBaseBlockPerToken(const uint32_t sInnerLoopIdx, uint32_t startRow,
                                                             uint32_t dealRowCount, uint32_t columnCount,
                                                             uint32_t actualColumnCount);
    __aicore__ inline void ProcessVec2(const uint32_t sInnerLoopIdx);
    __aicore__ inline void ProcessVec2Inner(const uint32_t sInnerLoopIdx);
    __aicore__ inline void PreProcessVec2(uint32_t sInnerLoopIdx);
    __aicore__ inline void SInnerLoopFunc(const uint32_t bn2Idx, const uint32_t sInnerLoopIdx);

    __aicore__ inline void SoftmaxFlashV2Compute(LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
                                                 uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                 uint32_t actualColumnCount);
    __aicore__ inline void PseShiftCopyIn(uint32_t startRow, uint32_t rowCount, uint32_t actualColumnCount);
    __aicore__ inline void ElewiseCompute(LocalTensor<T> &mmResUb, TBuf<> &tmpBuf, uint32_t startRow,
                                          uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);

    __aicore__ inline void Bmm2DataCopyOut(LocalTensor<OUT_T> &attenOutUb, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Bmm2CastAndCopyOut(LocalTensor<T> &bmm2ResUb, uint32_t startRow, uint32_t dealRowCount,
                                              uint32_t columnCount, uint32_t actualColumnCount);

    __aicore__ inline void CombineSplitKVRes();
    __aicore__ inline void CopyAccumOutIn(uint32_t splitKVIndex, uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void CopyLseIn(uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void ComputeLogSumExpAndCopyToGm(LocalTensor<T> &softmaxMaxUb, LocalTensor<T> &softmaxSumUb);
    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<T> &softmaxMaxUb, LocalTensor<T> &softmaxSumUb);
    __aicore__ inline void Bmm2FDDataCopyOut(LocalTensor<T> &bmm2ResUb, uint32_t startRow, uint32_t dealRowCount,
                                             uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void ComputeScaleValue(LocalTensor<T> &lseSum, LocalTensor<T> &lseMax, uint32_t startRow,
                                             uint32_t dealRowCount);
    __aicore__ inline void ReduceFinalRes(LocalTensor<T> &dst, LocalTensor<T> &lseLocal, uint32_t startRow,
                                          uint32_t dealRowCount);
    __aicore__ inline void CopyFinalResOut(LocalTensor<T> &accumOutLocal, uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void PostQuant(LocalTensor<T> &bmm2ResUb, LocalTensor<int8_t> &bmm2ResUbInt8, uint32_t startRow,
                                     uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void InitAllZeroOutput(uint32_t bIdx);
    __aicore__ inline void SysPrefixInitAllZeroOutput();
    __aicore__ inline void InitAllZeroInt8Output();
    __aicore__ inline uint64_t SeqLenFromTensorList(uint32_t bIdx);

    __aicore__ inline void SysPrefixAttenResCombine();
    __aicore__ inline void SysPrefixLseToScales(LocalTensor<T> &lseSum, LocalTensor<T> &lseMax);
    __aicore__ inline void SysPrefixAttenReduce(LocalTensor<T> &dst, GlobalTensor<T> &atten1, GlobalTensor<T> &atten2,
                                                LocalTensor<T> scales, uint32_t startRow, uint32_t rows);
    __aicore__ inline void SysPrefixAttenOutput(GlobalTensor<OUT_T> &dst, LocalTensor<T> &attenOut, uint32_t startRow,
                                                uint32_t rows);
    __aicore__ inline void SysPrefixSaveLse(uint32_t bIndex, uint32_t n2Index, LocalTensor<T> &softmaxSumUb,
                                            LocalTensor<T> &softmaxMaxUb, uint32_t start, uint32_t count,
                                            bool isPrefix);
    __aicore__ inline void SysPrefixSaveLseFd(LocalTensor<T> &lseSum, LocalTensor<T> &lseMax, uint32_t start,
                                              uint32_t count);
    __aicore__ inline void SysPrefixSaveLseFA();
    __aicore__ inline void SysPrefixSaveAttenRes(uint32_t bIndex, uint32_t n2Index, LocalTensor<T> &bmm2ResUb,
                                                 uint32_t startRow, uint32_t rows, bool isPrefix);

    __aicore__ inline void SysPrefixSaveZeroLse(uint32_t bIndex, uint32_t n2Index, bool isPrefix);
    __aicore__ inline void SysPrefixSaveZeroAttenRes(uint32_t bIndex, uint32_t n2Index, bool isPrefix);

    __aicore__ inline void SysPrefixSaveMsdMax1(uint32_t bIndex);
    __aicore__ inline void SysPrefixLoadMsdMax1(uint32_t bIndex);

    __aicore__ inline void SysPrefixSaveMsdMax2(uint32_t bIndex);
    __aicore__ inline void SysPrefixLoadMsdMax2(uint32_t bIndex);

    __aicore__ inline void SysPrefixSaveMsdSum1(uint32_t bIndex);
    __aicore__ inline void SysPrefixLoadMsdSum1(uint32_t bIndex);

    __aicore__ inline void SysPrefixSaveMsdSum2(uint32_t bIndex);
    __aicore__ inline void SysPrefixLoadMsdSum2(uint32_t bIndex);

    __aicore__ inline void SysPrefixSaveSoftmaxMax(uint32_t bIndex);
    __aicore__ inline void SysPrefixLoadSoftmaxMax(uint32_t bIndex);

    __aicore__ inline void SysPrefixSaveSoftmaxSum(uint32_t bIndex);
    __aicore__ inline void SysPrefixLoadSoftmaxSum(uint32_t bIndex);

    __aicore__ inline void SysPrefixSaveSoftmaxExp(uint32_t bIndex);
    __aicore__ inline void SysPrefixLoadSoftmaxExp(uint32_t bIndex);

    __aicore__ inline void CopyDataInByQueue1(LocalTensor<T> &dst, const GlobalTensor<T> &src, size_t size);
    __aicore__ inline void CopyDataInByQueue2(LocalTensor<T> &dst, const GlobalTensor<T> &src, size_t size);

    __aicore__ inline void CopyGmToFixedUb(LocalTensor<T> &dst, const GlobalTensor<T> &src, size_t size);
    __aicore__ inline void CopyFixedUbToGm(const GlobalTensor<T> &dst, const LocalTensor<T> &src, size_t size);
    __aicore__ inline void SoftmaxLseOutput(LocalTensor<T> &lse);

    __aicore__ inline void DealKvInt4ColumnOdd(uint32_t actualColumnCount);
};

template <typename IFAT> __aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::InitTilingData()
{
    singleProcessSInnerSize = tilingData->sparsePagedFusionAttentionSingleCoreParams.singleProcessSInnerSize;
    sInnerLoopTimes = tilingData->sparsePagedFusionAttentionSingleCoreParams.sInnerLoopTimes;
    singleProcessSInnerSizeTail = tilingData->sparsePagedFusionAttentionSingleCoreParams.singleProcessSInnerSizeTail;
    usedCoreNum = tilingData->sparsePagedFusionAttentionSingleCoreParams.usedCoreNum;
    formerCoreNum = tilingData->sparsePagedFusionAttentionSingleCoreParams.formerCoreNum;
    splitKVNum = tilingData->splitKVParams.s2;
    sInnerLoopSize = tilingData->splitKVParams.sInnerLoopSize;
    flashDecodeFlag = splitKVNum > 0;

    mmResUbSize = tilingData->sparsePagedFusionAttentionSingleCoreTensorSize.mmResUbSize;
    bmm2ResUbSize = tilingData->sparsePagedFusionAttentionSingleCoreTensorSize.bmm2ResUbSize;

    batchSize = tilingData->baseParams.batchSize;
    kvHeadNum = tilingData->baseParams.kvHeadNum;
    qHeadNum = tilingData->baseParams.qHeadNum;
    gSize = tilingData->baseParams.nNumOfQInOneGroup;
    kvSeqSize = tilingData->baseParams.seqSize;
    headDim = tilingData->baseParams.headSize;
    batchContinuous = tilingData->baseParams.batchContinuousFlag;
    msdIterNum = tilingData->baseParams.msdIterNum;
    antiquantPerTensorFlag = tilingData->baseParams.antiquantPerTensorFlag;
    antiquantPerHeadFlag = (tilingData->baseParams.antiquantPerHeadFlag == 1);
    antiquantParamsInPagedAttentionFlag = (tilingData->baseParams.antiquantParamsInPagedAttentionFlag == 1);

    headDimAlign = Align(headDim, BYTE_BLOCK);

    attenMaskFlag = (tilingData->baseParams.attenMaskFlag != 0) ? true : false;
    attenMaskSize = tilingData->baseParams.attenMaskSize;
    selectWithByteMaskTmpMinSize = tilingData->baseParams.selectWithByteMaskTmpMinSize;

    antiqSeqSize = tilingData->baseParams.antiqSeqSize;

    pseShiftFlag = (tilingData->baseParams.pseShiftFlag == 1) ? true : false;
    if (pseShiftFlag) {
        pseShiftB = tilingData->baseParams.pseShiftB;
        pseShiftS = tilingData->baseParams.pseShiftS;
    }

    kvPaddingFlag = tilingData->baseParams.kvPaddingFlag;

    // out quant
    isPerChnU8Out = tilingData->outputParams.isPerChnOut == 0 ? false : true;
    isOutQuantTypeBf16 = tilingData->outputParams.isOutQuantTypeBf16 == 0 ? false : true;

    // 是否输出lse
    softmaxLseFlag = tilingData->baseParams.softmaxLseFlag;
}

template <typename IFAT> __aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::InitBuffers()
{
    // queue
    pipe->InitBuffer(inputQue1, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(inputQue2, 1, BUFFER_SIZE_BYTE_16K);
    pipe->InitBuffer(outputQue1, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(outputQue2, 1, BUFFER_SIZE_BYTE_8K);

    // tmpBuff
    pipe->InitBuffer(tmpBuff1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(tmpBuff2, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(tmpBuff3, BUFFER_SIZE_BYTE_2K);

    // 常驻buffer
    pipe->InitBuffer(antiqScaleBuff, BUFFER_SIZE_BYTE_4K);
    pipe->InitBuffer(antiqOffsetBuff, BUFFER_SIZE_BYTE_4K);
    // 预留空间2K = 64 * 32，支持 gSize = 64
    // brcb 操作每次操作8*32字节输出，startRow接近64时，
    // 输出最多可能超出2k空间7*32字节， 这里预留256B防止越界
    pipe->InitBuffer(qAmaxBuff, BUFFER_SIZE_BYTE_2K + BUFFER_SIZE_BYTE_256B);
    pipe->InitBuffer(softmaxResAmaxBuff, BUFFER_SIZE_BYTE_2K + BUFFER_SIZE_BYTE_256B);
    pipe->InitBuffer(qRowSumBuff, BUFFER_SIZE_BYTE_2K + BUFFER_SIZE_BYTE_256B);
    pipe->InitBuffer(softmaxResRowSumBuff, BUFFER_SIZE_BYTE_2K + BUFFER_SIZE_BYTE_256B);
    pipe->InitBuffer(softmaxMaxBuff, BUFFER_SIZE_BYTE_2K);
    pipe->InitBuffer(softmaxExpBuff, BUFFER_SIZE_BYTE_2K);
    pipe->InitBuffer(softmaxSumBuff, BUFFER_SIZE_BYTE_2K);
    pipe->InitBuffer(bmm1PageAttentionDataBuff, BUFFER_SIZE_BYTE_64B);
    pipe->InitBuffer(bmm2PageAttentionDataBuff, BUFFER_SIZE_BYTE_64B);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::InitActualSeqLen(__gm__ uint8_t *actualSeqLengths)
{
    actualLenDims = tilingData->baseParams.actualLenDims;
    if (actualLenDims != 0) {
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengths, actualLenDims * 8);
    }
}

template <typename IFAT> __aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::InitAllZeroInt8Output()
{
    uint32_t gSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / headDimAlign;
    if (gSplitSize > gSize) {
        gSplitSize = gSize;
    }
    uint32_t loopCount = (gSize + gSplitSize - 1) / gSplitSize;
    uint32_t tailSplitSize = gSize - (loopCount - 1) * gSplitSize;

    for (uint32_t i = 0, dealSize = gSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        uint32_t startRow = gSplitSize * i;
        uint32_t dealRowCount = dealSize;
        uint32_t columnCount = headDimAlign;
        uint32_t actualColumnCount = headDim;
        LocalTensor<T> bmm2ResUb = tmpBuff1.Get<T>(); // bmm2 result is zero
        Duplicate(bmm2ResUb, static_cast<float>(0), dealRowCount * columnCount);
        LocalTensor<OUT_T> bmm2ResUbInt8 = outputQue1.AllocTensor<OUT_T>();

        PostQuant(bmm2ResUb, bmm2ResUbInt8, startRow, dealRowCount, columnCount, actualColumnCount);
        outputQue1.EnQue(bmm2ResUbInt8);
        outputQue1.DeQue<OUT_T>();

        attenOutOffset = tensorACoreOffset; // GM offset
        Bmm2DataCopyOut(bmm2ResUbInt8, startRow, dealRowCount, columnCount, actualColumnCount);
        outputQue1.FreeTensor(bmm2ResUbInt8);
    }
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::InitAllZeroOutput(uint32_t bIdx)
{
    uint32_t copySize = gSize * headDim;
    if constexpr (POST_QUANT) { // out int8
        InitAllZeroInt8Output();
    } else {
        matmul::InitOutput<OUT_T>(attentionOutGm[(bIdx * kvHeadNum + n2Idx) * copySize], copySize, 0);
    }

    if (softmaxLseFlag) {
        LocalTensor<T> softmaxlseOut = outputQue1.template AllocTensor<T>();
        float minf = -3.40E+38;
        Duplicate(softmaxlseOut, minf, gSize);
        outputQue1.EnQue(softmaxlseOut);
        outputQue1.DeQue();
        DataCopyExtParams intriParams1;
        intriParams1.blockLen = sizeof(float) * gSize;
        intriParams1.blockCount = 1;
        intriParams1.srcStride = 0;
        intriParams1.dstStride = 0;
        DataCopyPad(softmaxLseGm[(bIdx * kvHeadNum + n2Idx) * gSize], softmaxlseOut, intriParams1);
        outputQue1.FreeTensor(softmaxlseOut);
    }
}

template <typename IFAT> __aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::GetActualSeqLen()
{
    if (actualLenDims == 0) {
        curActualSeqLen = kvSeqSize;
        if (!batchContinuous) {
            curActualSeqLen = SeqLenFromTensorList(bIdx);
        }
    } else if (actualLenDims == 1) {
        curActualSeqLen = actualSeqLengthsGm.GetValue(0);
    } else {
        V5_DEBUG_PRINTF("[LOG] GetActualSeqLen bIdx: %d\n", bIdx);
        curActualSeqLen = actualSeqLengthsGm.GetValue(bIdx*8);
        V5_DEBUG_PRINTF("[LOG] GetActualSeqLen curActualSeqLen: %d\n", curActualSeqLen);
    }
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::GetBN2id(const uint32_t bn2Idx)
{
    if (flashDecodeFlag) {
        bIdx = tmpBlockIdx / (kvHeadNum * splitKVNum);
        n2Idx = (tmpBlockIdx / splitKVNum) % kvHeadNum;
        s2Idx = tmpBlockIdx % splitKVNum;
    } else {
        bIdx = (beforeBlockSplitBn2Nums + bn2Idx) / kvHeadNum;
        n2Idx = (beforeBlockSplitBn2Nums + bn2Idx) % kvHeadNum;
    }
}

template <typename IFAT> __aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::UpdateInnerLoopCond()
{
    if (curActualSeqLen == 0) {
        if constexpr (SHARED_PREFIX) {
            SysPrefixInitAllZeroOutput();
        } else {
            InitAllZeroOutput(bIdx);
        }

        curActSeqLenIsZero = true;
        return;
    }
    curActSeqLenIsZero = false;

    int32_t remainSinnerSize = (int32_t)curActualSeqLen;
    int32_t computeSinnerSize = (int32_t)curActualSeqLen;
    if (flashDecodeFlag) {
        remainSinnerSize = (int32_t)curActualSeqLen - sInnerLoopSize * s2Idx;
        computeSinnerSize = remainSinnerSize >= sInnerLoopSize ? sInnerLoopSize : remainSinnerSize;
        if (tmpBlockIdx >= batchSize * kvHeadNum * splitKVNum) {
            remainSinnerSize = 0;
        }
    }
    if (remainSinnerSize > 0) {
        if (computeSinnerSize <= singleProcessSInnerSize) {
            singleProcessSInnerSizeTail = computeSinnerSize;
            sInnerLoopTimes = 1;
        } else {
            sInnerLoopTimes = (computeSinnerSize + singleProcessSInnerSize - 1) / singleProcessSInnerSize;
            singleProcessSInnerSizeTail = computeSinnerSize - (sInnerLoopTimes - 1) * singleProcessSInnerSize;
        }
    } else {
        sInnerLoopTimes = 0;
    }
}

template <typename IFAT>
__aicore__ inline uint64_t SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SeqLenFromTensorList(uint32_t bIndex)
{
    uint64_t dimInfo[4]; // this mem is used to set shapeinfo, BSH(3) or BNSD(4)
    AscendC::TensorDesc<__gm__ uint8_t> keyTensorDesc;
    ListTensorDesc keyListTensorDesc((__gm__ void *)keyPtr);
    keyTensorDesc.SetShapeAddr(&dimInfo[0]);
    keyListTensorDesc.GetDesc(keyTensorDesc, bIndex);
    if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
        return keyTensorDesc.GetShape(1); // BSH, idx of s is 1
    } else {
        return keyTensorDesc.GetShape(2); // BNSD, idx of s is 2
    }
}

template <typename IFAT> __aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::CalculateSUnitSize()
{
    if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
        sUnitSize = kvHeadNum * headDim;
    } else {
        sUnitSize = headDim;
    }
    return;
}

template <typename IFAT>
__aicore__ inline bool SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::ComputeKVPaddingBeginOffset()
{
    if (kvPaddingFlag != 1) {
        return true;
    }
    int64_t paddingSize = kvPaddingSizeGm.GetValue(0);
    if (paddingSize < 0) {
        paddingSize = 0;
    }

    int64_t startPosition = kvSeqSize - paddingSize - curActualSeqLen;

    if (startPosition < 0) {
        InitAllZeroOutput(bIdx);
        return false;
    }

    kvPaddingBeginOffset = static_cast<uint64_t>(startPosition);
    return true;
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::InitPrefix(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *blockPosition, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
    const SparsePagedFusionAttentionTilingDataPrefix *__restrict tiling, __gm__ uint8_t *gmTiling, TPipe *tPipe)
{
    sysPrefixLen = tiling->prefixLen;
    formerCoreNumSp = tiling->formerCoreNum;
    blockSplitBn2RangeSp = tiling->blockSplitBn2Range;
    tailBlockSplitBn2RangeSp = tiling->tailSplitedBatchRange;
    usedCoreNumSp = tiling->usedCoreNum;
    batchSizeQ = tiling->batchSizeQ;

    sysPrefixAttenOutGm.SetGlobalBuffer((__gm__ T *)(workspace + tiling->prefixAttenOutOffset));
    usrPromptAttenOutGm.SetGlobalBuffer((__gm__ T *)(workspace + tiling->userPromptAttenOutOffset));
    lseSumGm.SetGlobalBuffer((__gm__ T *)(workspace + tiling->tmpLseOffset));
    lseMaxGm.SetGlobalBuffer((__gm__ T *)(workspace + tiling->tmpLseOffset +
                                          2 * batchSizeQ * tiling->base.baseParams.qHeadNum * BYTE_BLOCK));

    Init(query, key, value, pseShift, attenMask, actualSeqLengths, blockTable, kvPaddingSize, blockPosition, attentionOut, softmaxLse,
         workspace, &tiling->base, gmTiling, tPipe, true);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *blockPosition, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
    const SparsePagedFusionAttentionTilingData *__restrict tiling, __gm__ uint8_t *gmTiling, TPipe *tPipe, bool isPrefix)
{
    tmpBlockIdx = GetBlockIdx();
    // Only one vector core use one cube core when B*N number is less than half of core number
    if (tmpBlockIdx & 0x1) {
        // KERNEL_TYPE_MIX_AIC_1_1启用,使用opc编译时GetTaskRation()为1,ccec编译时GetTaskRation()为2
        tmpBlockIdx = (tmpBlockIdx + GetBlockNum() * GetTaskRation()) / 2;
    } else {
        tmpBlockIdx = tmpBlockIdx / 2;
    }

    // init tiling data
    tilingData = tiling;
    InitTilingData();
    // 初始化计算参数
    if (flashDecodeFlag) {
        InitCalcParams();
    } else {
        InitCalcParamsEach();
    }

    pipe = tPipe;
    keyPtr = key;
    valuePtr = value;
    blocktablePtr = blockTable;
    blockpositionPtr = blockPosition;
    useBlockPosition = (blockPosition != nullptr);
    
    // 添加调试日志：打印blockPosition地址信息
    //  V5_DEBUG_PRINTF("Init: blockPosition=%p, useBlockPosition=%d\n", 
                    // blockpositionPtr, useBlockPosition);

    // PA 新增，一次性tiling信息配置
    if constexpr (PAGE_ATTENTION) {
        mm.SetUserDefInfo(reinterpret_cast<uint64_t>(gmTiling));
        bmm2.SetUserDefInfo(reinterpret_cast<uint64_t>(gmTiling));
    }

    if (!isPrefix) {
        ListTensorDesc keyListTensorDesc((__gm__ void *)keyPtr);
        ListTensorDesc valueListTensorDesc((__gm__ void *)valuePtr);
        key_ = (__gm__ uint8_t *)keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(0);
        value_ = (__gm__ uint8_t *)valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(0);

        keyGm.SetGlobalBuffer((__gm__ KV_T *)key_);
        valueGm.SetGlobalBuffer((__gm__ KV_T *)value_);
    } else {
        keyGm.SetGlobalBuffer((__gm__ KV_T *)key);
        valueGm.SetGlobalBuffer((__gm__ KV_T *)value);
    }
    calcSysPrefixFlag = isPrefix;
    curSingleProcessSInnerSizeAlign = 0ULL; // prefix场景计算user prompt前必须重新初始化
    actualSingleProcessSInnerSize = 0ULL;
    actualSingleProcessSInnerSizeAlign = 0ULL;

    // init global buffer
    queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
    attentionOutGm.SetGlobalBuffer((__gm__ OUT_T *)attentionOut);

    if (tilingData->baseParams.l2CacheOffFlag) {
        // 关闭K、V的L2 Cache
#ifndef ASCENDC_OOM
        keyGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        valueGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
#endif
    }

    if (pipe != nullptr) {
        InitBuffers();
    }

    if (attenMaskFlag) {
        attenMaskBoolGm.SetGlobalBuffer((__gm__ bool *)attenMask);
    }

    InitActualSeqLen(actualSeqLengths);
    if (kvPaddingFlag == 1) {
        kvPaddingSizeGm.SetGlobalBuffer((__gm__ int64_t *)kvPaddingSize);
    }

    softmaxMaxUb = softmaxMaxBuff.Get<T>();
    softmaxSumUb = softmaxSumBuff.Get<T>();
    softmaxExpUb = softmaxExpBuff.Get<T>();

    uint64_t offset = 0;
    mm1ResGm.SetGlobalBuffer(
        (__gm__ MM_OUT_T *)(workspace + offset + tmpBlockIdx * mmResUbSize * dbWorkspaceRatio * sizeof(MM_OUT_T)));
    offset += GetBlockNum() * GetTaskRation() * mmResUbSize * dbWorkspaceRatio * sizeof(MM_OUT_T);
    if constexpr (KVINT4) {
        vec1ResGm.SetGlobalBuffer(
            (__gm__ KV_T *)(workspace + offset + (tmpBlockIdx * mmResUbSize * dbWorkspaceRatio * sizeof(KV_T) >> 1)));
        offset += (GetBlockNum() * GetTaskRation() * mmResUbSize * dbWorkspaceRatio * sizeof(KV_T) >> 1);
    } else {
        vec1ResGm.SetGlobalBuffer(
            (__gm__ KV_T *)(workspace + offset + tmpBlockIdx * mmResUbSize * dbWorkspaceRatio * sizeof(KV_T)));
        offset += GetBlockNum() * GetTaskRation() * mmResUbSize * dbWorkspaceRatio * sizeof(KV_T);
    }

    mm2ResGm.SetGlobalBuffer(
        (__gm__ MM_OUT_T *)(workspace + offset + tmpBlockIdx * bmm2ResUbSize * dbWorkspaceRatio * sizeof(MM_OUT_T)));
    offset += GetBlockNum() * GetTaskRation() * bmm2ResUbSize * dbWorkspaceRatio * sizeof(MM_OUT_T);
    vec2ResGm.SetGlobalBuffer(
        (__gm__ T *)(workspace + offset + tmpBlockIdx * bmm2ResUbSize * dbWorkspaceRatio * sizeof(T)));
    offset += GetBlockNum() * GetTaskRation() * bmm2ResUbSize * dbWorkspaceRatio * sizeof(T);
    if constexpr (ANTIQUANT) {
        if constexpr (KVINT4) {
            queryPreProcessResGm.SetGlobalBuffer(
                (__gm__ KV_T *)(workspace + offset +
                                (tmpBlockIdx * bmm2ResUbSize * dbWorkspaceRatio * sizeof(KV_T) >> 1)));
            offset += (GetBlockNum() * GetTaskRation() * bmm2ResUbSize * dbWorkspaceRatio * sizeof(KV_T) >> 1);
        } else {
            queryPreProcessResGm.SetGlobalBuffer(
                (__gm__ KV_T *)(workspace + offset + tmpBlockIdx * bmm2ResUbSize * dbWorkspaceRatio * sizeof(KV_T)));
            offset += GetBlockNum() * GetTaskRation() * bmm2ResUbSize * dbWorkspaceRatio * sizeof(KV_T);
        }
    }

    // GM for pse
    if (pseShiftFlag) {
        pseShiftGm.SetGlobalBuffer((__gm__ pseShiftType *)pseShift);
    }

    if (flashDecodeFlag) {
        accumOutGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
        offset = offset + tilingData->splitKVParams.accumOutSize * sizeof(float);
        lseSumFdGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
        lseMaxFdGm.SetGlobalBuffer((__gm__ float *)(workspace + offset) + tilingData->splitKVParams.logSumExpSize / 2);
        offset = offset + tilingData->splitKVParams.logSumExpSize * sizeof(float);
    }

    if (softmaxLseFlag) {
        softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
    }

    if constexpr (PAGE_ATTENTION) {
        // dcci cacheline 64B 对齐
        bmm1CallBackDataGm.SetGlobalBuffer(
            (__gm__ uint32_t *)(workspace + offset + tmpBlockIdx * mmPACallBackDataSize));
        bmm1CallBackDataPtr = (__gm__ uint32_t *)(workspace + offset + tmpBlockIdx * mmPACallBackDataSize);
        offset = offset + GetBlockNum() * GetTaskRation() * mmPACallBackDataSize;

        bmm2CallBackDataGm.SetGlobalBuffer(
            (__gm__ uint32_t *)(workspace + offset + tmpBlockIdx * mmPACallBackDataSize));
        bmm2CallBackDataPtr = (__gm__ uint32_t *)(workspace + offset + tmpBlockIdx * mmPACallBackDataSize);
        offset = offset + GetBlockNum() * GetTaskRation() * mmPACallBackDataSize;

        // TODO: CUBE 传参待完善 需要传入blockPosition
    }

    if constexpr (SHARED_PREFIX) {
        if (isPrefix) {
            if constexpr (ANTIQUANT) {
                size_t blockSize = gSize * BYTE_BLOCK * batchSizeQ;
                msdRowMax1Gm.SetGlobalBuffer((__gm__ T *)(workspace + offset + tmpBlockIdx * blockSize * 4));
                msdRowMax2Gm = msdRowMax1Gm[blockSize / sizeof(T)];
                msdRowSum1Gm = msdRowMax1Gm[2 * blockSize / sizeof(T)];
                msdRowSum2Gm = msdRowMax1Gm[3 * blockSize / sizeof(T)];
                offset = offset + GetBlockNum() * GetTaskRation() * blockSize * 4; // 包含4块相同大小的数据
                msdRowMaxSize = gSize * BYTE_BLOCK / sizeof(T);
                msdRowSumSize = msdRowMaxSize;
            }

            size_t blockSize = gSize * BYTE_BLOCK * batchSizeQ;
            softmaxRowMaxGm.SetGlobalBuffer((__gm__ T *)(workspace + offset + tmpBlockIdx * blockSize * 3));
            softmaxRowSumGm = softmaxRowMaxGm[blockSize / sizeof(T)];
            softmaxRowExpGm = softmaxRowMaxGm[2 * blockSize / sizeof(T)];
            offset = offset + GetBlockNum() * GetTaskRation() * blockSize * 3; // 包含3块相同大小的数据
            softmaxMaxSumExpSize = gSize * BYTE_BLOCK / sizeof(T);

            if constexpr (!ANTIQUANT) {
                size_t blockSize = batchSizeQ * gSize * headDimAlign * sizeof(Q_T);
                prefixQueryPreProcessResGm.SetGlobalBuffer(
                    (__gm__ Q_T *)(workspace + offset + tmpBlockIdx * blockSize));
                offset = offset + GetBlockNum() * GetTaskRation() * blockSize;
            }
        }
    }
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::InitCentSelect(__gm__ uint8_t *query, __gm__ uint8_t *l1_cent, __gm__ uint8_t *block_ids,
    __gm__ uint8_t *block_table, __gm__ uint8_t *total_seq_len, __gm__ uint8_t *blockPosition, __gm__ uint8_t *pagePositionLength,
    __gm__ uint8_t *maxPagePositionLength, __gm__ uint8_t *workspace,
    const SparsePagedFusionAttentionTilingData *__restrict tiling, __gm__ uint8_t *gmTiling, TPipe *tPipe) {
    // V5_DEBUG_PRINTF("[LOG] InitCentSelect Tilling Data Start\n");
    // Init tiling data
    tilingData = tiling;
    batchSize = tilingData->centSelectParams.bSize;
    qHeadNum = tilingData->centSelectParams.n1Size;
    kvHeadNum = tilingData->centSelectParams.n2Size;
    gSize = tilingData->centSelectParams.gSize;
    kvBlockSize = tilingData->centSelectParams.blockSize;
    usedCoreNum = tilingData->centSelectParams.usedCoreNum;
    // compute cent
    clusterNum = tilingData->centSelectParams.cSize;
    dimNum = tilingData->centSelectParams.dSize;
    clusterBlockNum = tilingData->centSelectParams.clusterBlockNum;
    clusterBlockSize = tilingData->centSelectParams.clusterBlockSize;
    // select position
    kvPageLen = tilingData->centSelectParams.kvPageLen;
    maxBatch = tilingData->centSelectParams.maxBatch;
    maxPage = tilingData->centSelectParams.maxPage;
    maxPageNum = tilingData->centSelectParams.maxPageNum;
    // topk
    k = tilingData->centSelectParams.k;
    tmpsize = tilingData->centSelectParams.tmpsize;
    topkTilingData = tilingData->centSelectParams.topkTilingData;
    // max
    maxWorkSize = qHeadNum * tplPadding * 2 * 32;
    pipe = tPipe;

    // blockIdx = AscendC::GetBlockIdx();
    
    // Set global buffers
    queryGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(query), batchSize * qHeadNum * dimNum);
    l1CentGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(l1_cent), kvHeadNum * clusterNum * dimNum);
    blockIdsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(block_ids), kvHeadNum * kvPageLen);
    blockTableGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(block_table), maxBatch * maxPage);
    seqLenGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(total_seq_len), batchSize);
    
    blockPositionGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(blockPosition), batchSize * qHeadNum * maxPageNum);
    pagePositionLengthGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(pagePositionLength), batchSize * qHeadNum * tplPadding);
    maxPagePositionLengthGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(maxPagePositionLength), batchSize * tplPadding);
    
    // Init input buffers
    pipe->InitBuffer(inQuery, BUFFER_NUM, dimNum * sizeof(half));
    pipe->InitBuffer(inL1Cent, BUFFER_NUM, clusterBlockSize * dimNum * sizeof(half));
    pipe->InitBuffer(inBlockIds, BUFFER_NUM, kvPageLen * sizeof(int32_t));
    pipe->InitBuffer(inBlockTable, BUFFER_NUM, maxPage * sizeof(int32_t));
    // Init output buffers
    pipe->InitBuffer(outPagePosition, BUFFER_NUM, maxPageNum * sizeof(int32_t));
    pipe->InitBuffer(outPagePositionLength, BUFFER_NUM, tplPadding * sizeof(int32_t));
    // Init compute buffers
    pipe->InitBuffer(tmpBuff1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(tmpBmm1ResBuff, clusterBlockSize * sizeof(float));
    pipe->InitBuffer(bmm1ResBuff, clusterNum * sizeof(float));
    // Init topk buffers
    pipe->InitBuffer(topKDstValue, k * sizeof(float));
    pipe->InitBuffer(topKDstIndex, k * sizeof(int32_t));
    pipe->InitBuffer(topKSrcIndexLocal, clusterNum * sizeof(int32_t));
    pipe->InitBuffer(topKWrokLocal, tmpsize * sizeof(uint8_t));
    pipe->InitBuffer(topKFinishLocal, tmpsize * sizeof(bool));
    // Init select position buffers
    pipe->InitBuffer(tmpBuffPageBatch, maxPage * sizeof(int32_t));
    pipe->InitBuffer(tmpBuffSelectReduce, maxPage / 8 * sizeof(uint8_t));
    pipe->InitBuffer(tmpBuffSelectTmp, maxPage / 8 * sizeof(uint8_t));
    pipe->InitBuffer(selectBlockIdsIndexLocal, maxPage * sizeof(int32_t));
    // Init max page position length buffers
    pipe->InitBuffer(totalPagePositionLength, qHeadNum * tplPadding * sizeof(int32_t));
    pipe->InitBuffer(totalPagePositionLengthFloat, qHeadNum * tplPadding * sizeof(float));
    pipe->InitBuffer(outMaxPagePositionLengthInt32, qHeadNum * tplPadding * sizeof(int32_t));
    // Init max buffers
    pipe->InitBuffer(maxWorkLocal, maxWorkSize * sizeof(uint8_t));
    pipe->InitBuffer(dstMaxLocal, 1 * sizeof(float));
    pipe->InitBuffer(outMaxPagePositionLength, 1, tplPadding * sizeof(int64_t));

    // V5_DEBUG_PRINTF("[LOG] InitCentSelect InitBuffer End\n");
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::ProcessCentSelect(TPipe *tPipe) {
    auto localBlockIdx = AscendC::GetBlockIdx();
    if (g_coreType == AIV && localBlockIdx >= usedCoreNum) {
        // skip
    } else {
        int64_t multiCoreInnerOffset = static_cast<int64_t>(localBlockIdx) * static_cast<int64_t>(kvBlockSize);
        int64_t multiCoreInnerLimit = multiCoreInnerOffset + static_cast<int64_t>(kvBlockSize);
        for (int64_t bn1Idx = multiCoreInnerOffset; bn1Idx < multiCoreInnerLimit; ++bn1Idx) {
            bIdx = static_cast<uint64_t>(bn1Idx) / qHeadNum;
            n1Idx = static_cast<uint64_t>(bn1Idx) % qHeadNum;
            n2Idx = n1Idx / gSize;
            if (bIdx < batchSize && n1Idx < qHeadNum) {
                CentCopyIn();
                auto dstIndexLocal = topKDstIndex.Get<int32_t>();
                CentComputeTopK(dstIndexLocal);
                CentSelectPosition(dstIndexLocal);
                CentCopyOut();
            }
        }
    }
    SyncAll();
    if (g_coreType == AIV && static_cast<uint32_t>(AscendC::GetBlockIdx()) < batchSize) {
        CentMaxReducePagePositionLength(static_cast<uint32_t>(AscendC::GetBlockIdx()));
    }
    // 第一个kernel完成后释放不再需要的内存
    ReleaseCentSelectBuffers(tPipe);
}

// 新增函数：释放cent_select不再需要的内存
template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::ReleaseCentSelectBuffers(TPipe *tPipe) {
    // TODO: 最好Tensor也释放掉
    pipe = tPipe;
    pipe->Reset();

    // // Set global buffers
    // queryGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(query), batchSize * qHeadNum * dimNum);
    // l1CentGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(l1_cent), kvHeadNum * clusterNum * dimNum);
    // blockIdsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(block_ids), kvHeadNum * kvPageLen);
    // blockTableGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(block_table), maxBatch * maxPage);
    // seqLenGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(total_seq_len), batchSize);
    
    // blockPositionGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(blockPosition), batchSize * qHeadNum * maxPageNum);
    // pagePositionLengthGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(pagePositionLength), batchSize * qHeadNum * tplPadding);
    // maxPagePositionLengthGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(maxPagePositionLength), batchSize * tplPadding);
        


    // // 释放输入缓冲区
    // pipe->FreeBuffer(inQuery);
    // pipe->FreeBuffer(inL1Cent);
    // pipe->FreeBuffer(inBlockIds);
    // pipe->FreeBuffer(inBlockTable);
    
    // // 释放计算缓冲区
    // pipe->FreeBuffer(tmpBmm1ResBuff);
    // pipe->FreeBuffer(bmm1ResBuff);
    
    // // 释放TopK缓冲区
    // pipe->FreeBuffer(topKDstValue);
    // pipe->FreeBuffer(topKDstIndex);
    // pipe->FreeBuffer(topKSrcIndexLocal);
    // pipe->FreeBuffer(topKWrokLocal);
    // pipe->FreeBuffer(topKFinishLocal);
    
    // // 释放选择位置缓冲区
    // pipe->FreeBuffer(tmpBuffPageBatch);
    // pipe->FreeBuffer(tmpBuffSelectReduce);
    // pipe->FreeBuffer(tmpBuffSelectTmp);
    // pipe->FreeBuffer(selectBlockIdsIndexLocal);
    
    // // 释放最大工作缓冲区
    // pipe->FreeBuffer(maxWorkLocal);
    // pipe->FreeBuffer(dstMaxLocal);
    
    // // 释放冲突的临时缓冲区，为第二个kernel重新分配做准备
    // pipe->FreeBuffer(tmpBuff1);
}



template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::CentCopyIn()
{
    int64_t queryOffset = static_cast<int64_t>(bIdx) * static_cast<int64_t>(qHeadNum * dimNum) + static_cast<int64_t>(n1Idx * dimNum);
    auto inputUa = inQuery.AllocTensor<half>();
    DataCopyParams intriParamsQ;
    uint32_t typeElementSizeQ = BYTE_BLOCK / sizeof(half);
    intriParamsQ.blockCount = 1;
    intriParamsQ.dstStride = 0;
    intriParamsQ.blockLen = dimNum / typeElementSizeQ;
    intriParamsQ.srcStride = 0;
    DataCopy(inputUa, queryGlobal[queryOffset], intriParamsQ);
    inQuery.EnQue(inputUa);

    int64_t blockIdsOffset = static_cast<int64_t>(n2Idx) * static_cast<int64_t>(kvPageLen);
    auto blockIdsLocal = inBlockIds.AllocTensor<int32_t>();
    AscendC::DataCopy(blockIdsLocal, blockIdsGlobal[blockIdsOffset], kvPageLen);
    blockIdsLocal.SetSize(kvPageLen);
    inBlockIds.EnQue(blockIdsLocal);

    int64_t blockTableOffset = static_cast<int64_t>(bIdx) * static_cast<int64_t>(maxPage);
    auto blockTableLocal = inBlockTable.AllocTensor<int32_t>();
    AscendC::DataCopy(blockTableLocal, blockTableGlobal[blockTableOffset], maxPage);
    blockTableLocal.SetSize(maxPage);
    inBlockTable.EnQue(blockTableLocal);

    seqLen = seqLenGlobal.GetValue(static_cast<uint32_t>(bIdx));
    pageLen = (seqLen + PAGESIZE - 1) / PAGESIZE;
    workLoadThreshold = pageLen / 8;
    gatherMaskLen = pageLen / 8;
    gatherMaskU32Len = pageLen / 32;
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::CentComputeTopK(LocalTensor<int32_t> &dstIndexLocal)
{
    auto inputUa = inQuery.DeQue<half>();
    auto mmResUb = bmm1ResBuff.Get<float>();
    for (uint32_t i = 0; i < static_cast<uint32_t>(clusterBlockNum); i++) {
        uint32_t clusterBlockOffset = i * static_cast<uint32_t>(clusterBlockSize);
        BlockingL1CentCopyIn(clusterBlockOffset);
        auto inputUb = inL1Cent.DeQue<half>();
        auto tmpBmm1ResUb = tmpBmm1ResBuff.Get<float>();
        VectorCompute(tmpBmm1ResUb, inputUa, inputUb, static_cast<uint32_t>(clusterBlockSize));
        DataCopy(mmResUb[clusterBlockOffset], tmpBmm1ResUb, static_cast<uint32_t>(clusterBlockSize));
        inL1Cent.FreeTensor(inputUb);
    }
    inQuery.FreeTensor(inputUa);
    pipe_barrier(PIPE_ALL);

    auto dstValueLocal = topKDstValue.Get<float>();
    auto srcLocalIndex = topKSrcIndexLocal.Get<int32_t>();
    auto finishLocal = topKFinishLocal.Get<bool>();
    auto tmpLocal = topKWrokLocal.Get<uint8_t>();

    topKInfo.outter = 1;
    topKInfo.inner = static_cast<uint32_t>(clusterNum);
    topKInfo.n = static_cast<uint32_t>(clusterNum);
    AscendC::TopK<float, false, false, false, AscendC::TopKMode::TOPK_NORMAL>(
        dstValueLocal,
        dstIndexLocal,
        mmResUb,
        srcLocalIndex,
        finishLocal,
        tmpLocal,
        static_cast<uint32_t>(k),
        topkTilingData,
        topKInfo,
        true
    );
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::BlockingL1CentCopyIn(uint32_t idx)
{
    int64_t l1CentOffset = static_cast<int64_t>(n2Idx) * static_cast<int64_t>(clusterNum * dimNum) + static_cast<int64_t>(idx * dimNum);
    auto inputUb = inL1Cent.AllocTensor<half>();
    DataCopyParams intriParams;
    uint32_t typeElementSize = BYTE_BLOCK / sizeof(half);
    intriParams.blockCount = static_cast<uint32_t>(clusterBlockSize);
    intriParams.dstStride = 0;
    intriParams.blockLen = static_cast<uint32_t>(dimNum) / typeElementSize;
    intriParams.srcStride = 0;
    DataCopy(inputUb, l1CentGlobal[l1CentOffset], intriParams);
    inL1Cent.EnQue(inputUb);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::VectorCompute(LocalTensor<float> &mmResUb, LocalTensor<half> &aUb, LocalTensor<half> &bUb, uint32_t dealRowCount)
{
    auto vmlaResUb = tmpBuff1.Get<float>();
    uint32_t elementSize = vmlaResUb.GetSize();
    uint32_t maxDealRowCount = elementSize / (BYTE_BLOCK / sizeof(half));
    uint32_t singleDealRowCnt = (maxDealRowCount > dealRowCount) ? dealRowCount : maxDealRowCount;
    uint32_t rowLoopCnt = (dealRowCount + singleDealRowCnt - 1) / singleDealRowCnt;
    uint32_t columnLoopCnt = static_cast<uint32_t>(dimNum) / FP16_ONE_BLOCK_SIZE;
    uint32_t rowElementCnt = static_cast<uint32_t>(dimNum);
    for (uint32_t i = 0, curDealRowCnt = singleDealRowCnt; i < rowLoopCnt; i++) {
        uint32_t rowStart = i * singleDealRowCnt;
        BinaryRepeatParams repeatParams;
        uint32_t repeat_times = curDealRowCnt / VMLA_ONE_REPEATE_ROW_COUNT;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 0;
        repeatParams.src1BlkStride = rowElementCnt / FP16_ONE_BLOCK_SIZE;
        repeatParams.dstRepStride = 2 * VMLA_ONE_REPEATE_ROW_COUNT;
        repeatParams.src0RepStride = 0;
        repeatParams.src1RepStride = VMLA_ONE_REPEATE_ROW_COUNT * rowElementCnt / FP16_ONE_BLOCK_SIZE;
        Duplicate(vmlaResUb, FLOAT_ZERO, repeat_times * VMLA_ONE_REPEATE_ROW_COUNT * 16);
        pipe_barrier(PIPE_V);
        for (uint32_t j = 0; j < columnLoopCnt; j++) {
            MulAddDst(vmlaResUb, aUb[j * FP16_ONE_BLOCK_SIZE], bUb[rowStart * rowElementCnt + j * FP16_ONE_BLOCK_SIZE],
                      64, repeat_times, repeatParams);
            pipe_barrier(PIPE_V);
        }
        repeat_times = IFA_MAX_REPEAT_TIMES - 1;
        for (uint32_t j = 0; j < curDealRowCnt; j += repeat_times) {
            if (j + repeat_times > curDealRowCnt) {
                repeat_times = curDealRowCnt - j;
            }
            BinaryRepeatParams addRepeatParamsForBmm1;
            addRepeatParamsForBmm1.src0BlkStride = 1;
            addRepeatParamsForBmm1.src1BlkStride = 1;
            addRepeatParamsForBmm1.dstBlkStride = 1;
            addRepeatParamsForBmm1.src0RepStride = 2;
            addRepeatParamsForBmm1.src1RepStride = 2;
            addRepeatParamsForBmm1.dstRepStride = 2;
            Add(vmlaResUb[j * FP16_ONE_BLOCK_SIZE], vmlaResUb[j * FP16_ONE_BLOCK_SIZE],
                vmlaResUb[j * FP16_ONE_BLOCK_SIZE + 8], 8, repeat_times, addRepeatParamsForBmm1);
        }
        pipe_barrier(PIPE_V);
        BlockReduceSum(mmResUb[rowStart], vmlaResUb[0], Align<uint32_t>(curDealRowCnt, static_cast<uint32_t>(8)) / 8, 64, 1, 2, 16);
        pipe_barrier(PIPE_V);
    }
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::CentSelectPosition(LocalTensor<int32_t> indicesLocal)
{
    auto blockTableLocal = inBlockTable.DeQue<int32_t>();
    AscendC::Muls(blockTableLocal, blockTableLocal, int32_t(4), static_cast<uint32_t>(pageLen));
    auto blockIdsLocal = inBlockIds.DeQue<int32_t>();
    auto pageBatchLocal = tmpBuffPageBatch.Get<int32_t>();
    pipe_barrier(PIPE_ALL);
    AscendC::Gather(pageBatchLocal, blockIdsLocal, blockTableLocal.ReinterpretCast<uint32_t>(), uint32_t(0), static_cast<uint32_t>(pageLen));

    auto dstResultMaskLocal = tmpBuffSelectReduce.Get<uint8_t>();
    auto dstMaskLocalTmp = tmpBuffSelectTmp.Get<uint8_t>();
    AscendC::CompareScalar(dstResultMaskLocal, pageBatchLocal, indicesLocal.GetValue(0), CMPMODE::EQ, static_cast<uint32_t>(pageLen));
    for (uint32_t i = 1; i < static_cast<uint32_t>(k); i++) {
        AscendC::CompareScalar(dstMaskLocalTmp, pageBatchLocal, indicesLocal.GetValue(i), CMPMODE::EQ, static_cast<uint32_t>(pageLen));
        pipe_barrier(PIPE_ALL);
        AscendC::Or(dstResultMaskLocal, dstResultMaskLocal, dstMaskLocalTmp, static_cast<uint32_t>(pageLen));
    }

    uint64_t rsvdCnt = 0;
    auto blockIdsIndex = selectBlockIdsIndexLocal.Get<int32_t>();
    AscendC::CreateVecIndex(blockIdsIndex, (int32_t)0, static_cast<uint32_t>(pageLen));

    auto pagePositionLengthLocal = outPagePositionLength.AllocTensor<int32_t>();
    pagePositionLengthLocal.SetSize(tplPadding);
    auto pagePositionLocal = outPagePosition.AllocTensor<int32_t>();
    pagePositionLocal.SetSize(maxPageNum);
    AscendC::Duplicate(pagePositionLocal, 0x7fffffff, static_cast<uint32_t>(maxPageNum));

    AscendC::GatherMask(pagePositionLocal, blockIdsIndex, dstResultMaskLocal.ReinterpretCast<uint32_t>(), true, static_cast<uint32_t>(pageLen), {1, 1, 8, 8}, rsvdCnt);
    if (rsvdCnt > static_cast<uint64_t>(workLoadThreshold)) {
        pagePositionLength = workLoadThreshold;
    } else {
        pagePositionLength = static_cast<int32_t>(rsvdCnt);
    }
    AscendC::Duplicate(pagePositionLengthLocal, pagePositionLength, tplPadding);
    outPagePositionLength.EnQue(pagePositionLengthLocal);
    outPagePosition.EnQue(pagePositionLocal);

    inBlockIds.FreeTensor(blockIdsLocal);
    inBlockTable.FreeTensor(blockTableLocal);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::CentCopyOut()
{
    int64_t pagePositionOffset = static_cast<int64_t>(bIdx) * static_cast<int64_t>(qHeadNum * maxPageNum) + static_cast<int64_t>(n1Idx * maxPageNum);
    auto pagePositionLocal = outPagePosition.DeQue<int32_t>();
    AscendC::DataCopy(blockPositionGlobal[pagePositionOffset], pagePositionLocal, static_cast<uint32_t>(maxPageNum));
    outPagePosition.FreeTensor(pagePositionLocal);

    int64_t pagePositionLengthOffset = static_cast<int64_t>(bIdx) * static_cast<int64_t>(qHeadNum * tplPadding) + static_cast<int64_t>(n1Idx * tplPadding);
    auto pagePositionLengthLocal = outPagePositionLength.DeQue<int32_t>();
    DataCopy(pagePositionLengthGlobal[pagePositionLengthOffset], pagePositionLengthLocal, tplPadding);
    outPagePositionLength.FreeTensor(pagePositionLengthLocal);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::CentMaxReducePagePositionLength(uint32_t localBlockIdx)
{
    auto totalPagePositionLengthLocal = totalPagePositionLength.Get<int32_t>();
    AscendC::DataCopy(totalPagePositionLengthLocal, pagePositionLengthGlobal[localBlockIdx * qHeadNum * tplPadding], static_cast<uint32_t>(qHeadNum * tplPadding));
    auto totalPagePositionLengthFloatLocal = totalPagePositionLengthFloat.Get<float>();
    pipe_barrier(PIPE_ALL);
    AscendC::Cast(totalPagePositionLengthFloatLocal, totalPagePositionLengthLocal, AscendC::RoundMode::CAST_CEIL, static_cast<uint32_t>(qHeadNum * tplPadding));
    auto worklocal = maxWorkLocal.Get<float>();
    auto dstLocal = dstMaxLocal.Get<float>();
    AscendC::ReduceMax(dstLocal, totalPagePositionLengthFloatLocal, worklocal, static_cast<uint32_t>(qHeadNum * tplPadding));
    int32_t maxSeqLen = static_cast<int32_t>(dstLocal.GetValue(0)) * PAGESIZE;
    auto outMaxPagePositionLengthInt32Local = outMaxPagePositionLengthInt32.Get<int32_t>();
    AscendC::Duplicate(outMaxPagePositionLengthInt32Local, maxSeqLen, tplPadding);
    auto outMaxPagePositionLengthLocal = outMaxPagePositionLength.AllocTensor<int64_t>();
    AscendC::Cast(outMaxPagePositionLengthLocal, outMaxPagePositionLengthInt32Local, AscendC::RoundMode::CAST_NONE, tplPadding);
    outMaxPagePositionLength.EnQue(outMaxPagePositionLengthLocal);
    outMaxPagePositionLength.DeQue<int64_t>();
    AscendC::DataCopy(maxPagePositionLengthGlobal[localBlockIdx * tplPadding], outMaxPagePositionLengthLocal, tplPadding);
    outMaxPagePositionLength.FreeTensor(outMaxPagePositionLengthLocal);
}



template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::InitQuant(
    __gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2,
    __gm__ uint8_t *quantOffset2, __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
    __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset, __gm__ uint8_t *valueAntiquantScale,
    __gm__ uint8_t *valueAntiquantOffset, __gm__ uint8_t *workspace)
{
    InitAntiquant(antiquantScale, antiquantOffset, keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset);
    InitPostQuant(deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2);
}
template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::InitAntiquant(
    __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset, __gm__ uint8_t *keyAntiquantScale,
    __gm__ uint8_t *keyAntiquantOffset, __gm__ uint8_t *valueAntiquantScale, __gm__ uint8_t *valueAntiquantOffset)
{
    if constexpr (ANTIQUANT) {
        if (keyAntiquantScale == nullptr) {
            int64_t antiValueOffsetInitPos = kvHeadNum * headDim;
            if (antiquantPerTensorFlag == 1) {
                antiValueOffsetInitPos = 1;
            }
            if constexpr (ANTIQUANT_PER_TOKEN) {
                antiValueOffsetInitPos = batchSize * antiqSeqSize;
            }
            keyAntiqScaleGm.SetGlobalBuffer((__gm__ ANTIQ_PARAMS_T_KEY *)antiquantScale);
            valueAntiqScaleGm.SetGlobalBuffer(((__gm__ ANTIQ_PARAMS_T_VALUE *)antiquantScale) + antiValueOffsetInitPos);
            antiqOffsetExistFlag = (antiquantOffset != nullptr);
            if (antiqOffsetExistFlag) {
                keyAntiqOffsetGm.SetGlobalBuffer((__gm__ ANTIQ_PARAMS_T_KEY *)antiquantOffset);
                valueAntiqOffsetGm.SetGlobalBuffer(((__gm__ ANTIQ_PARAMS_T_VALUE *)antiquantOffset) + antiValueOffsetInitPos);
            }
        } else {
            keyAntiqScaleGm.SetGlobalBuffer((__gm__ ANTIQ_PARAMS_T_KEY *)keyAntiquantScale);
            valueAntiqScaleGm.SetGlobalBuffer((__gm__ ANTIQ_PARAMS_T_VALUE *)valueAntiquantScale);
            antiqOffsetExistFlag = (keyAntiquantOffset != nullptr);
            if (antiqOffsetExistFlag) {
                keyAntiqOffsetGm.SetGlobalBuffer((__gm__ ANTIQ_PARAMS_T_KEY *)keyAntiquantOffset);
                valueAntiqOffsetGm.SetGlobalBuffer((__gm__ ANTIQ_PARAMS_T_VALUE *)valueAntiquantOffset);
            }
        }

        aMaxBmm1Ub = qAmaxBuff.Get<T>();
        aMaxBmm2Ub = softmaxResAmaxBuff.Get<T>();
        if constexpr (ANTIQUANT_PER_TOKEN) {
            qRowSumUb = qRowSumBuff.Get<T>();
            softmaxScaleResRowSumUb = softmaxResRowSumBuff.Get<T>();
        } else if constexpr (ANTIQUANT_PER_CHANNEL) {
            qRowSumUb = qRowSumBuff.Get<T>();
            softmaxResRowSumUb = softmaxResRowSumBuff.Get<T>();
            antiqScaleUb = antiqScaleBuff.Get<T>();
            antiqOffsetUb = antiqOffsetBuff.Get<T>();
        } else if constexpr (ANTIQUANT_PER_CHANNEL_TOKEN) {
            qRowSumUb = qRowSumBuff.Get<T>();
            softmaxScaleResRowSumUb = softmaxResRowSumBuff.Get<T>();
            antiqScaleUb = antiqScaleBuff.Get<T>();
            antiqOffsetUb = antiqOffsetBuff.Get<T>();
        }
    }
}
template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::InitPostQuant(__gm__ uint8_t *deqScale1,
    __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2, __gm__ uint8_t *quantOffset2)
{
    if constexpr (POST_QUANT) {
        if (!isPerChnU8Out && !isOutQuantTypeBf16) {
            if (quantScale2 != nullptr) {
                quantScale2Gm.SetGlobalBuffer((__gm__ float *)quantScale2);
                quantScale2Value = quantScale2Gm.GetValue(0);
            }
            if (quantOffset2 != nullptr) {
                quantOffset2Gm.SetGlobalBuffer((__gm__ float *)quantOffset2);
                quantOffset2Value = quantOffset2Gm.GetValue(0);
            } else {
                quantOffset2Value = 0;
            }
        }
        if (quantScale2 != nullptr && !isPerChnU8Out && isOutQuantTypeBf16) {
            quantScale2Bf16Gm.SetGlobalBuffer((__gm__ bfloat16_t *)quantScale2);
            quantScale2Value = ToFloat(quantScale2Bf16Gm.GetValue(0));
        }
        if (!isPerChnU8Out && isOutQuantTypeBf16) {
            if (quantOffset2 != nullptr) {
                quantOffset2Bf16Gm.SetGlobalBuffer((__gm__ bfloat16_t *)quantOffset2);
                quantOffset2Value = ToFloat(quantOffset2Bf16Gm.GetValue(0));
            } else {
                quantOffset2Value = 0;
            }
        }

        if (isPerChnU8Out && !isOutQuantTypeBf16) {
            if (quantScale2 != nullptr) {
                quantScale2Gm.SetGlobalBuffer((__gm__ float *)quantScale2);
            }
            if (quantOffset2 != nullptr) {
                isQuantOffset2Exist = true;
                quantOffset2Gm.SetGlobalBuffer((__gm__ float *)quantOffset2);
            }
        }

        if (isPerChnU8Out && isOutQuantTypeBf16) {
            if (quantScale2 != nullptr) {
                quantScale2Bf16Gm.SetGlobalBuffer((__gm__ bfloat16_t *)quantScale2);
            }
            if (quantOffset2 != nullptr) {
                isQuantOffset2Exist = true;
                quantOffset2Bf16Gm.SetGlobalBuffer((__gm__ bfloat16_t *)quantOffset2);
            }
        }
    }
}

template <typename IFAT> __aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::InitCalcParams()
{
    bn2LoopTimes = tilingData->sparsePagedFusionAttentionSingleCoreParams.blockSplitBn2Range;
    beforeBlockSplitBn2Nums = tmpBlockIdx * tilingData->sparsePagedFusionAttentionSingleCoreParams.blockSplitBn2Range;
    // tail core
    if (tmpBlockIdx >= formerCoreNum) {
        bn2LoopTimes = tilingData->sparsePagedFusionAttentionSingleCoreParams.tailSplitedBatchRange;
        beforeBlockSplitBn2Nums =
            formerCoreNum * tilingData->sparsePagedFusionAttentionSingleCoreParams.blockSplitBn2Range +
            (tmpBlockIdx - formerCoreNum) * tilingData->sparsePagedFusionAttentionSingleCoreParams.tailSplitedBatchRange;
    }
}

template <typename IFAT> __aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::InitCalcParamsEach()
{
    // 这里是编译器优化写法，定义一个局部数组变量coreSidxEnd(存在栈上)，使用copy_data_align64接口
    // 可以只从ub中拷贝tiling中coreSidxEnd的内容到栈上，而非将整个sparsePagedFusionAttentionCoreParams
    // 内容拷贝到栈，减少拷贝时间。
#ifdef ASCENDC_CPU_DEBUG
    const uint32_t *coreSidxEnd = tilingData->sparsePagedFusionAttentionCoreParams.coreSidxEnd;
#else
    uint32_t coreSidxEnd[50];
    copy_data_align64((uint8_t *)coreSidxEnd, (uint8_t *)(tilingData->sparsePagedFusionAttentionCoreParams.coreSidxEnd),
                      sizeof(coreSidxEnd));
#endif
    bn2LoopTimes = coreSidxEnd[tmpBlockIdx + 1] - coreSidxEnd[tmpBlockIdx];
    beforeBlockSplitBn2Nums = coreSidxEnd[tmpBlockIdx];
}

template <typename IFAT> __aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::CalcBN2Offset()
{
    if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
        // B,1,N2,G,D
        tensorACoreOffset = bIdx * qHeadNum * headDim + n2Idx * gSize * headDim;
        // B,S2,N2,D
        tensorBCoreOffset =
            bIdx * kvSeqSize * kvHeadNum * headDim + n2Idx * headDim + kvPaddingBeginOffset * kvHeadNum * headDim;

        if (!batchContinuous) {
            tensorBCoreOffset = n2Idx * headDim;
        }

        if (flashDecodeFlag) {
            tensorBCoreOffset += s2Idx * sInnerLoopSize * kvHeadNum * headDim;
        }
    } else {
        tensorACoreOffset = bIdx * qHeadNum * headDim + n2Idx * gSize * headDim;
        // B,N2,S2,D
        tensorBCoreOffset =
            bIdx * kvHeadNum * kvSeqSize * headDim + n2Idx * kvSeqSize * headDim + kvPaddingBeginOffset * headDim;

        if (!batchContinuous) {
            uint64_t seqSize = SeqLenFromTensorList(bIdx);
            tensorBCoreOffset = n2Idx * seqSize * headDim;
        }
        if (flashDecodeFlag) {
            tensorBCoreOffset += s2Idx * sInnerLoopSize * headDim;
        }
    }
}

template <typename IFAT> __aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::CalcBN2Params()
{
    attenMaskCoreOffset = bIdx * attenMaskSize + kvPaddingBeginOffset;
    if (flashDecodeFlag) {
        attenMaskCoreOffset += s2Idx * sInnerLoopSize;
    }
    // antiquant的offset和scale参数数据排列是先key后value
    if (antiquantPerTensorFlag == 1) {
        antiqParamOffset = 0;
    } else {
        antiqParamOffset = n2Idx * headDim;
    }
    antiqKeyParamCoreOffsetPerToken = bIdx * antiqSeqSize + kvPaddingBeginOffset;
    if (antiquantPerHeadFlag) {
        antiqKeyParamCoreOffsetPerToken = bIdx * antiqSeqSize * kvHeadNum + kvPaddingBeginOffset +
                                          n2Idx * antiqSeqSize;
    }
    if (flashDecodeFlag) {
        antiqKeyParamCoreOffsetPerToken += s2Idx * sInnerLoopSize;
    }
    if (antiquantPerHeadFlag) {
        antiqParamOffset = n2Idx;
    }
    // out quant
    perChannelQuantOffset = n2Idx * headDim * gSize;
    if (!batchContinuous) {
        ListTensorDesc keyListTensorDesc((__gm__ void *)keyPtr);
        ListTensorDesc valueListTensorDesc((__gm__ void *)valuePtr);
        __gm__ uint8_t *key = (__gm__ uint8_t *)keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);
        __gm__ uint8_t *value = (__gm__ uint8_t *)valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);

        keyGm.SetGlobalBuffer((__gm__ KV_T *)key);
        valueGm.SetGlobalBuffer((__gm__ KV_T *)value);
        if (tilingData->baseParams.l2CacheOffFlag) {
            // 关闭K、V的L2 Cache
#ifndef ASCENDC_OOM
            keyGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            valueGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
#endif
        }
    }
    // 更新actualSingleProcessSInnerSize，防止尾块值，影响第二次loop
    actualSingleProcessSInnerSize = singleProcessSInnerSize;
    if constexpr (KVINT4) {
        actualSingleProcessSInnerSizeAlign = Align(singleProcessSInnerSize, 64UL);
    } else {
        actualSingleProcessSInnerSizeAlign = Align(singleProcessSInnerSize, BYTE_BLOCK);
    }

    if (pseShiftFlag) {
        pseShiftCoreOffset = (pseShiftB == 1) ? (n2Idx * gSize * pseShiftS) : (bIdx * qHeadNum * pseShiftS + n2Idx * gSize * pseShiftS);
        if (flashDecodeFlag) {
            pseShiftCoreOffset += s2Idx * sInnerLoopSize;
        }
        pseShiftCoreOffset += kvPaddingBeginOffset; // kv_padding_size
    }
}

template <typename IFAT>
__aicore__ inline void
SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::CalcSInnerOffsetAndParams(const uint32_t sInnerLoopIdx)
{
    uint64_t sInnerOffsetDataSize = sInnerLoopIdx * singleProcessSInnerSize;
    if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
      // B,Si,N2,D
      tensorBOffset = tensorBCoreOffset + sInnerOffsetDataSize * kvHeadNum * headDim;
    } else {
      tensorBOffset = tensorBCoreOffset + sInnerOffsetDataSize * headDim;
    }
    attenOutOffset = tensorACoreOffset;
    valueOffset = tensorBOffset;
    attenMaskOffset = attenMaskCoreOffset + sInnerOffsetDataSize;
    antiqParamOffsetPerToken = antiqKeyParamCoreOffsetPerToken + sInnerOffsetDataSize;
    if constexpr (SHARED_PREFIX) {
        if (!calcSysPrefixFlag) {
            attenMaskOffset += sysPrefixLen;
            antiqParamOffsetPerToken += sysPrefixLen;
        }
    }

    // Calc Params
    if (sInnerLoopIdx == sInnerLoopTimes - 1) {
        actualSingleProcessSInnerSize = singleProcessSInnerSizeTail;
        if constexpr (KVINT4) {
            actualSingleProcessSInnerSizeAlign = Align(singleProcessSInnerSizeTail, 64UL);
        } else {
            actualSingleProcessSInnerSizeAlign = Align(singleProcessSInnerSizeTail, BYTE_BLOCK);
        }
    }

    // pse offset
    if (pseShiftFlag) {
        pseShiftOffset = pseShiftCoreOffset + sInnerOffsetDataSize;
        if constexpr (SHARED_PREFIX) {
            if (!calcSysPrefixFlag)
                pseShiftOffset += sysPrefixLen;
        }
    }
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::UpdateOffsetsVec(uint32_t sInnerLoopIdx)
{
    // antiquant的offset和scale参数数据排列是先key后value
    if (antiquantPerTensorFlag == 1) {
        antiqParamOffset = 0;
    } else {
        antiqParamOffset = n2Idx * headDim;
    }
    antiqKeyParamCoreOffsetPerToken = bIdx * antiqSeqSize + kvPaddingBeginOffset;
    if (antiquantPerHeadFlag) {
        antiqParamOffset = n2Idx;
        antiqKeyParamCoreOffsetPerToken = bIdx * kvHeadNum * antiqSeqSize + n2Idx * antiqSeqSize +
                                          kvPaddingBeginOffset;
    }
    if (flashDecodeFlag) {
        antiqKeyParamCoreOffsetPerToken += s2Idx * sInnerLoopSize;
    }
    // out quant
    perChannelQuantOffset = n2Idx * headDim * gSize;

    if (pseShiftFlag) {
        if (pseShiftB == 1) {
            pseShiftCoreOffset = n2Idx * gSize * pseShiftS;
        } else {
            pseShiftCoreOffset = bIdx * qHeadNum * pseShiftS + n2Idx * gSize * pseShiftS;
        }
        if (flashDecodeFlag) {
            pseShiftCoreOffset += s2Idx * sInnerLoopSize;
        }
    }

    uint64_t sInnerOffsetDataSize = sInnerLoopIdx * singleProcessSInnerSize;
    attenOutOffset = bIdx * qHeadNum * headDim + n2Idx * gSize * headDim;

    attenMaskCoreOffset = bIdx * attenMaskSize; // 前缀不用考虑左kvpadding
    if (flashDecodeFlag) {
        attenMaskCoreOffset += s2Idx * sInnerLoopSize;
    }
    attenMaskOffset = attenMaskCoreOffset + sInnerOffsetDataSize;
    antiqParamOffsetPerToken = antiqKeyParamCoreOffsetPerToken + sInnerOffsetDataSize;

    // pse offset
    if (pseShiftFlag) {
        pseShiftOffset = pseShiftCoreOffset + sInnerOffsetDataSize;
    }
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::AttenMaskCopyIn(uint64_t offset,
                                                                                     uint32_t dealRowCount,
                                                                                     uint32_t actualColumnCount)
{
    LocalTensor<bool> maskUb = inputQue2.AllocTensor<bool>();
    attenMaskSizeAlign = Align(actualColumnCount, 32U);
    maskUb.SetSize(dealRowCount * attenMaskSizeAlign);
#if (__CCE_AICORE__ > 200)
    if (actualColumnCount % 32 == 0) {
        DataCopy(maskUb, attenMaskBoolGm[offset], attenMaskSizeAlign);
    } else {
        uint32_t typeElementSize = BYTE_BLOCK / sizeof(bool);
        DataCopyExtParams intriParams;
        intriParams.blockLen = actualColumnCount * sizeof(bool);
        intriParams.blockCount = 1;
        intriParams.dstStride = (attenMaskSizeAlign - actualColumnCount) / typeElementSize;
        intriParams.srcStride = 0;
        DataCopyPadExtParams<bool> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = (attenMaskSizeAlign - actualColumnCount) % typeElementSize;
        padParams.paddingValue = 0;
        DataCopyPad(maskUb, attenMaskBoolGm[offset], intriParams, padParams);
    }
#else
    DataCopy(maskUb, attenMaskBoolGm[offset], attenMaskSizeAlign);
#endif
    inputQue2.template EnQue(maskUb);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::CopyAntiquantScale(LocalTensor<T> &castUb,
                                                                                        GlobalTensor<Q_T> srcGm,
                                                                                        uint64_t offset)
{
    if (antiquantPerHeadFlag || antiquantPerTensorFlag == 1) {
        if constexpr (AscendC::IsSameType<Q_T, half>::value) {
            Duplicate(castUb, static_cast<T>(srcGm.GetValue(offset)), headDimAlign);
        } else if constexpr (AscendC::IsSameType<Q_T, bfloat16_t>::value) {
            Duplicate(castUb, ToFloat(srcGm.GetValue(offset)), headDimAlign);
        }
    } else {
        uint32_t qTypeElementSize = BYTE_BLOCK / sizeof(Q_T);
        DataCopyExtParams copyInParams;
        DataCopyPadExtParams<Q_T> copyInPadParams;
        // antiq scale copy in
        copyInParams.blockCount = 1;
        copyInParams.blockLen = headDim * sizeof(Q_T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = (headDimAlign - headDim) / qTypeElementSize;

        copyInPadParams.isPad = true;
        copyInPadParams.leftPadding = 0;
        copyInPadParams.rightPadding = (headDimAlign - headDim) % qTypeElementSize;
        copyInPadParams.paddingValue = 0;

        LocalTensor<Q_T> inputUb = inputQue2.AllocTensor<Q_T>();
        DataCopyPad(inputUb, srcGm[offset], copyInParams, copyInPadParams);
        inputQue2.template EnQue(inputUb);

        inputUb = inputQue2.DeQue<Q_T>();
        Cast(castUb, inputUb, RoundMode::CAST_NONE, headDim);
        inputQue2.FreeTensor(inputUb);
    }
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::CopyAntiquantParamsPerTokenHead(
    GlobalTensor<ANTIQ_PARAMS_T_VALUE> srcGm, uint64_t offset, uint32_t columnCount) {
    LocalTensor<ANTIQ_PARAMS_T_VALUE> dstUb = inputQue1.AllocTensor<ANTIQ_PARAMS_T_VALUE>();
    DataCopy(dstUb, srcGm[offset], columnCount);
    inputQue1.template EnQue(dstUb);
}


template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::CopyAntiquantParamsParamsPagedAttentionImpl(
    GlobalTensor<ANTIQ_PARAMS_T_VALUE> srcGm, uint64_t offset, uint32_t actualColumnCount,
    uint32_t useKvHeadNum, uint32_t useN2Idx)
{
    // TODO: 本函数的 BlockPosition 待完善 -- 暂时不支持
    uint64_t kvCacheBlockSize = tilingData->baseParams.blockSize;
    uint32_t maxBlockNumPerBatch = tilingData->baseParams.maxBlockNumPerBatch;
    uint32_t paramsTypeElementSize = BYTE_BLOCK / sizeof(ANTIQ_PARAMS_T_VALUE);
    __gm__ int32_t *blockTableGmAddr = reinterpret_cast<__gm__ int32_t *>(blocktablePtr);

    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<ANTIQ_PARAMS_T_VALUE> copyInPadParams;
    copyInParams.blockCount = 1;
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;

    copyInPadParams.isPad = true;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.paddingValue = 0;

    uint64_t allBlockBaseIndex = bIdx * maxBlockNumPerBatch;
    uint64_t dstOffset = 0;
    uint32_t copyFinishElmeCnt = 0;
    uint64_t curSeqIdx = offset - bIdx * useKvHeadNum * kvSeqSize - useN2Idx * kvSeqSize;
    LocalTensor<ANTIQ_PARAMS_T_VALUE> paramsUb = inputQue1.AllocTensor<ANTIQ_PARAMS_T_VALUE>();

    while (copyFinishElmeCnt < actualColumnCount) {
        uint64_t blockIdOffset = curSeqIdx / kvCacheBlockSize;
        uint64_t reaminElemCnt = curSeqIdx % kvCacheBlockSize;
        uint32_t blockId = *(blockTableGmAddr + allBlockBaseIndex + blockIdOffset);
        uint32_t copyElemCnt = kvCacheBlockSize - reaminElemCnt;
        if (copyFinishElmeCnt + copyElemCnt > actualColumnCount) {
            copyElemCnt = actualColumnCount - copyFinishElmeCnt;
        }
        uint32_t copyElemCntAilgin = Align(copyElemCnt, paramsTypeElementSize);

        copyInPadParams.rightPadding = copyElemCntAilgin - copyElemCntAilgin;
        copyInParams.blockLen = copyElemCnt * sizeof(ANTIQ_PARAMS_T_VALUE);
        uint64_t srcOffst = blockId * useKvHeadNum * kvCacheBlockSize + useN2Idx * kvCacheBlockSize + reaminElemCnt;
        // 此处需注意paramsUb偏移量的合法性（UB地址需要32Byte对齐）
        // 在 (blockSize % sInnerSize) * sizeof(ANTIQ_PARAMS_T_VALUE)不是32对齐时，可能出现非法偏移
        // 当前blockSize为16对齐，sInnerSize为1024对齐，ANTIQ_PARAMS_T_VALUE为FP32，不会出现非法偏移
        // 后续上述三个元素限制有改动时，需要小心此处
        DataCopyPad(paramsUb[dstOffset], srcGm[srcOffst], copyInParams, copyInPadParams);

        dstOffset += copyElemCnt;
        copyFinishElmeCnt += copyElemCnt;
        curSeqIdx += copyElemCnt;
    }
    inputQue1.template EnQue(paramsUb);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::CopyAntiquantParamsParamsPagedAttention(
    GlobalTensor<ANTIQ_PARAMS_T_VALUE> srcGm, uint64_t offset, uint32_t actualColumnCount)
{
    uint32_t useKvHeadNum = 1;
    uint32_t useN2Idx = 0;
    if (antiquantPerHeadFlag) {
        useKvHeadNum = kvHeadNum;
        useN2Idx = n2Idx;
    }
    CopyAntiquantParamsParamsPagedAttentionImpl(srcGm, offset, actualColumnCount, useKvHeadNum, useN2Idx);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::CopyAntiquantParamsPerToken(
    GlobalTensor<ANTIQ_PARAMS_T_VALUE> srcGm, uint64_t offset, uint32_t columnCount, uint32_t actualColumnCount)
{
    if (antiquantParamsInPagedAttentionFlag) {
        CopyAntiquantParamsParamsPagedAttention(srcGm, offset, actualColumnCount);
        return;
    }
    if (antiquantPerHeadFlag) {
        CopyAntiquantParamsPerTokenHead(srcGm, offset, columnCount);
        return;
    }
    uint32_t paramsTypeElementSize = BYTE_BLOCK / sizeof(ANTIQ_PARAMS_T_VALUE);
    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<ANTIQ_PARAMS_T_VALUE> copyInPadParams;
    // antiq scale copy in
    copyInParams.blockCount = 1;
    copyInParams.blockLen = actualColumnCount * sizeof(ANTIQ_PARAMS_T_VALUE);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;

    copyInPadParams.isPad = true;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = (columnCount - actualColumnCount) % paramsTypeElementSize;
    copyInPadParams.paddingValue = 0;

    LocalTensor<ANTIQ_PARAMS_T_VALUE> paramsUb = inputQue1.AllocTensor<ANTIQ_PARAMS_T_VALUE>();
    DataCopyPad(paramsUb, srcGm[offset], copyInParams, copyInPadParams);
    inputQue1.template EnQue(paramsUb);
}

template <typename IFAT>
__aicore__ inline void
SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::CopyAntiqQuery(LocalTensor<T> &queryCastUb, uint64_t qOffset,
                                                             uint32_t dealRowCount, uint32_t columnCount,
                                                             uint32_t actualColumnCount)
{
    uint32_t qTypeElementSize = BYTE_BLOCK / sizeof(Q_T);
    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<Q_T> copyInPadParams;
    // antiq scale copy in
    copyInParams.blockCount = dealRowCount;
    copyInParams.blockLen = actualColumnCount * sizeof(Q_T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = (columnCount - actualColumnCount) / qTypeElementSize;

    copyInPadParams.isPad = true;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = (columnCount - actualColumnCount) % qTypeElementSize;
    copyInPadParams.paddingValue = 0;

    LocalTensor<Q_T> inputUb = inputQue1.AllocTensor<Q_T>();
    DataCopyPad(inputUb, queryGm[qOffset], copyInParams, copyInPadParams);
    inputQue1.template EnQue(inputUb);

    inputUb = inputQue1.DeQue<Q_T>();
    Cast(queryCastUb, inputUb, RoundMode::CAST_NONE, dealRowCount * columnCount);
    inputQue1.FreeTensor(inputUb);
}

template <typename IFAT>
__aicore__ inline void
SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::AbsRowMax(LocalTensor<T> &tmpAMaxRes, LocalTensor<T> &srcUb,
                                                        LocalTensor<T> tmpAUb, uint32_t dealRowCount,
                                                        uint32_t columnCount, uint32_t actualColumnCount)
{
    Abs(tmpAUb, srcUb, dealRowCount * columnCount);
    pipe_barrier(PIPE_V);
    LocalTensor<T> tmpRowMaxUb = tmpBuff3.Get<T>();
    RowMax(tmpRowMaxUb, tmpAUb, dealRowCount, columnCount, actualColumnCount);
    pipe_barrier(PIPE_V);
    Brcb(tmpAMaxRes, tmpRowMaxUb, (dealRowCount + 7) / 8, {1, 8});
}

template <typename IFAT>
__aicore__ inline void
SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::AntiquantAIterExpand(GlobalTensor<KV_T> dstGm, LocalTensor<T> &tmpA1,
                                                                   LocalTensor<T> &tmpA2, uint32_t calcSize,
                                                                   bool isFirst, uint64_t outOffset)
{
    if (!isFirst) {
        Sub(tmpA1, tmpA1, tmpA2, calcSize);
        pipe_barrier(PIPE_V);
        Muls(tmpA1, tmpA1, antiquantExpandCoeff, calcSize);
        pipe_barrier(PIPE_V);
    }

    Cast(tmpA2, tmpA1, RoundMode::CAST_ROUND, calcSize);
    pipe_barrier(PIPE_V);

    // cast-fp16
    LocalTensor<half> aResOutUb = outputQue1.template AllocTensor<half>();
    Cast(aResOutUb, tmpA2, RoundMode::CAST_ROUND, calcSize);
    pipe_barrier(PIPE_V);

    // cast-int8
    LocalTensor<KV_T> aResOutUbI8 = aResOutUb.template ReinterpretCast<KV_T>();
    aResOutUbI8.SetSize(aResOutUb.GetSize());
    Cast(aResOutUbI8, aResOutUb, RoundMode::CAST_ROUND, calcSize);

    // copyOut Ak
    outputQue1.template EnQue(aResOutUbI8);
    outputQue1.template DeQue<KV_T>();
    if constexpr (KVINT4) {
        DataCopy(dstGm[outOffset], aResOutUbI8, calcSize >> 1);
    } else {
        DataCopy(dstGm[outOffset], aResOutUbI8, calcSize);
    }
    outputQue1.FreeTensor(aResOutUbI8);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::AntiquantMatmulPreProcess(
    GlobalTensor<KV_T> dstGm, LocalTensor<T> aMaxResUb, LocalTensor<T> srcUb, LocalTensor<T> tmpAFloorUb,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t step = gSize * columnCount;
    uint32_t baseOffset = startRow * columnCount;
    uint32_t calcSize = dealRowCount * columnCount;

    LocalTensor<T> tmpAMaxRes = aMaxResUb[startRow * BLOCK_ELEMENT_NUM];
    AbsRowMax(tmpAMaxRes, srcUb, tmpAFloorUb, dealRowCount, columnCount, actualColumnCount);
    pipe_barrier(PIPE_V);

    // 128/(1.001*Amax)*A
    Duplicate(tmpAFloorUb, antiqCoeff1, dealRowCount * BLOCK_ELEMENT_NUM);
    pipe_barrier(PIPE_V);
    Div(tmpAFloorUb, tmpAFloorUb, tmpAMaxRes, dealRowCount * BLOCK_ELEMENT_NUM);
    pipe_barrier(PIPE_V);
    RowMuls(srcUb, srcUb, tmpAFloorUb, dealRowCount, columnCount, actualColumnCount);
    pipe_barrier(PIPE_V);

    for (uint32_t i = 0; i < msdIterNum; i++) {
        AntiquantAIterExpand(dstGm, srcUb, tmpAFloorUb, calcSize, (i == 0 ? true : false), step * i + baseOffset);
    }
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::AntiquantSoftmaxResPreProcess(
    GlobalTensor<KV_T> dstGm, LocalTensor<T> srcUb, LocalTensor<T> tmpAFloorUb, uint32_t startRow,
    uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t step = gSize * columnCount;
    uint32_t baseOffset = startRow * columnCount;
    uint32_t calcSize = dealRowCount * columnCount;

    Muls(srcUb, srcUb, antiqCoeff1, calcSize);
    pipe_barrier(PIPE_V);

    for (uint32_t i = 0; i < msdIterNum; i++) {
        AntiquantAIterExpand(dstGm, srcUb, tmpAFloorUb, calcSize, (i == 0 ? true : false), step * i + baseOffset);
    }
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::DealQueryPreProcessBaseBlock(
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint64_t qOffset = bIdx * qHeadNum * headDim + n2Idx * gSize * headDim;
    qOffset += startRow * actualColumnCount;

    LocalTensor<T> queryUb = tmpBuff1.Get<T>();
    LocalTensor<T> aFloorUb = tmpBuff2.Get<T>();
    CopyAntiqQuery(queryUb, qOffset, dealRowCount, columnCount, actualColumnCount);

    pipe_barrier(PIPE_V);
    // mul scale
    VecMulMat(queryUb, antiqScaleUb, queryUb, dealRowCount, columnCount, actualColumnCount);

    pipe_barrier(PIPE_V);

    if (softmaxLseFlag && antiqOffsetExistFlag) {
        LocalTensor<T> tmpRowSumUb = tmpBuff2.Get<T>();
        VecMulMat(tmpRowSumUb, antiqOffsetUb, queryUb, dealRowCount, columnCount, actualColumnCount);
        pipe_barrier(PIPE_V);
        RowSum(tmpRowSumUb, tmpRowSumUb, dealRowCount, columnCount, actualColumnCount);
        pipe_barrier(PIPE_V);
        Brcb(qRowSumUb[startRow * BLOCK_ELEMENT_NUM], tmpRowSumUb, (dealRowCount + 7) / 8,
             {1, 8}); // fill eight data blocks in the result tensor with eight data blocks in the input tensor
        pipe_barrier(PIPE_V);
    }

    size_t dstOffset = 0;
    if constexpr (SHARED_PREFIX) {
        if (calcSysPrefixFlag) {
            dstOffset = bIdx * gSize * msdIterNum * columnCount;
        }
    }

    // A pre process
    AntiquantMatmulPreProcess(queryPreProcessResGm[dstOffset], aMaxBmm1Ub, queryUb, aFloorUb, startRow, dealRowCount,
                              columnCount, actualColumnCount);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::DealQueryPreProcessBaseBlockPerToken(
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t baseOffset = startRow * BLOCK_ELEMENT_NUM;

    uint64_t qOffset = bIdx * qHeadNum * headDim + n2Idx * gSize * headDim;
    qOffset += startRow * actualColumnCount;

    LocalTensor<T> queryUb = tmpBuff1.Get<T>();
    LocalTensor<T> aFloorUb = tmpBuff2.Get<T>();
    CopyAntiqQuery(queryUb, qOffset, dealRowCount, columnCount, actualColumnCount);

    pipe_barrier(PIPE_V);
    if (antiqOffsetExistFlag) {
        LocalTensor<T> tmpRowSumUb = tmpBuff3.Get<T>();
        Adds(aFloorUb, queryUb, (T)0, dealRowCount * columnCount); // queryUb数据需要保留
        pipe_barrier(PIPE_V);
        RowSum(tmpRowSumUb, aFloorUb, dealRowCount, columnCount, actualColumnCount);
        pipe_barrier(PIPE_V);
        Brcb(qRowSumUb[baseOffset], tmpRowSumUb, (dealRowCount + 7) / 8, {1, 8});
        pipe_barrier(PIPE_V);
    }

    size_t dstOffset = 0;
    if constexpr (SHARED_PREFIX) {
        if (calcSysPrefixFlag) {
            dstOffset = bIdx * gSize * msdIterNum * columnCount;
        }
    }
    // A pre process
    AntiquantMatmulPreProcess(queryPreProcessResGm[dstOffset], aMaxBmm1Ub, queryUb, aFloorUb, startRow, dealRowCount,
                              columnCount, actualColumnCount);
}

template <typename IFAT> __aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::QueryPreProcessInner()
{
    CopyAntiquantScale(antiqScaleUb, keyAntiqScaleGm, antiqParamOffset);
    if (softmaxLseFlag && antiqOffsetExistFlag) {
        CopyAntiquantScale(antiqOffsetUb, keyAntiqOffsetGm, antiqParamOffset);
    }

    uint32_t gSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / headDimAlign;
    if (gSplitSize > gSize) {
        gSplitSize = gSize;
    }
    uint32_t loopCount = (gSize + gSplitSize - 1) / gSplitSize;
    uint32_t tailSplitSize = gSize - (loopCount - 1) * gSplitSize;

    for (uint32_t i = 0, dealSize = gSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealQueryPreProcessBaseBlock(i * gSplitSize, dealSize, headDimAlign, headDim);
    }
}

template <typename IFAT> __aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::QueryPreProcess()
{
    if constexpr (SHARED_PREFIX) {
        if (calcSysPrefixFlag) {
            uint32_t bIdxOld = bIdx;
            for (bIdx = 0; bIdx < batchSizeQ; bIdx++) {
                UpdateOffsetsVec(0);
                QueryPreProcessInner();
                SysPrefixSaveMsdMax1(bIdx);
                if (softmaxLseFlag && antiqOffsetExistFlag) {
                    SysPrefixSaveMsdSum1(bIdx);
                }
            }
            bIdx = bIdxOld;
            return;
        }
    }
    QueryPreProcessInner();
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SysPrefixQueryPreProcessInner()
{
    uint32_t gSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / headDimAlign;
    if (gSplitSize > gSize) {
        gSplitSize = gSize;
    }
    uint32_t loopCount = (gSize + gSplitSize - 1) / gSplitSize;
    uint32_t tailSplitSize = gSize - (loopCount - 1) * gSplitSize;

    for (uint32_t i = 0, dealSize = gSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        // 这里不对齐的d
        uint64_t qOffset = bIdx * qHeadNum * headDim + n2Idx * gSize * headDim;
        qOffset += gSplitSize * i * headDim;
        uint64_t qOutOffset = bIdx * gSize * headDimAlign + gSplitSize * i * headDimAlign;
        uint32_t calcSize = dealSize * headDimAlign;

        LocalTensor<Q_T> in = inputQue1.AllocTensor<Q_T>();
        if (headDim == headDimAlign) {
            DataCopy(in, queryGm[qOffset], calcSize);
        } else {
            uint32_t qTypeElementSize = BYTE_BLOCK / sizeof(Q_T);
            DataCopyExtParams copyInParams;
            DataCopyPadExtParams<Q_T> copyInPadParams;
            copyInParams.blockCount = dealSize;
            copyInParams.blockLen = headDim * sizeof(Q_T);
            copyInParams.srcStride = 0;
            copyInParams.dstStride = (headDimAlign - headDim) / qTypeElementSize;

            copyInPadParams.isPad = true;
            copyInPadParams.leftPadding = 0;
            copyInPadParams.rightPadding = (headDimAlign - headDim) % qTypeElementSize;
            copyInPadParams.paddingValue = 0;

            DataCopyPad(in, queryGm[qOffset], copyInParams, copyInPadParams);
        }
        inputQue1.template EnQue(in);
        inputQue1.template DeQue<Q_T>();

        LocalTensor<Q_T> out = outputQue1.AllocTensor<Q_T>();
        DataCopy(out, in, calcSize);
        inputQue1.FreeTensor(in);
        outputQue1.template EnQue(out);
        outputQue1.template DeQue<Q_T>();
        DataCopy(prefixQueryPreProcessResGm[qOutOffset], out, calcSize);
        outputQue1.FreeTensor(out);
    }
}

template <typename IFAT> __aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SysPrefixQueryPreProcess()
{
    if (calcSysPrefixFlag) {
        uint32_t bIdxOld = bIdx;
        for (bIdx = 0; bIdx < batchSizeQ; bIdx++) {
            UpdateOffsetsVec(0);
            SysPrefixQueryPreProcessInner(); // prefix 场景需要重排batch个q到连续内存中
        }
        bIdx = bIdxOld;
    }
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::QueryPreProcessPerTokenInner()
{
    uint32_t gSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / headDimAlign;
    if (gSplitSize > gSize) {
        gSplitSize = gSize;
    }
    uint32_t loopCount = (gSize + gSplitSize - 1) / gSplitSize;
    uint32_t tailSplitSize = gSize - (loopCount - 1) * gSplitSize;

    for (uint32_t i = 0, dealSize = gSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealQueryPreProcessBaseBlockPerToken(i * gSplitSize, dealSize, headDimAlign, headDim);
    }
}

template <typename IFAT> __aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::QueryPreProcessPerToken()
{
    if constexpr (SHARED_PREFIX) {
        if (calcSysPrefixFlag) {
            uint32_t bIdxOld = bIdx;
            for (bIdx = 0; bIdx < batchSizeQ; bIdx++) {
                UpdateOffsetsVec(0);
                QueryPreProcessPerTokenInner();
                SysPrefixSaveMsdMax1(bIdx);
                if (antiqOffsetExistFlag) {
                    SysPrefixSaveMsdSum1(bIdx);
                }
            }
            bIdx = bIdxOld;
            return;
        }
    }
    QueryPreProcessPerTokenInner();
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::CopyLseIn(uint32_t startRow, uint32_t dealRowCount)
{
    LocalTensor<T> lseSum = inputQue2.AllocTensor<T>();

    combineLseOffset = (bIdx * kvHeadNum * splitKVNum + n2Idx * splitKVNum) * gSize * FP32_ONE_BLOCK_SIZE +
                       startRow * FP32_ONE_BLOCK_SIZE;
    LocalTensor<T> lseMax = inputQue1.AllocTensor<T>();
    for (uint32_t i = 0; i < actualCombineLoopSize; i++) {
        DataCopy(lseSum[i * dealRowCount * FP32_ONE_BLOCK_SIZE],
                 lseSumFdGm[combineLseOffset + i * gSize * FP32_ONE_BLOCK_SIZE], dealRowCount * FP32_ONE_BLOCK_SIZE);
        DataCopy(lseMax[i * dealRowCount * FP32_ONE_BLOCK_SIZE],
                 lseMaxFdGm[combineLseOffset + i * gSize * FP32_ONE_BLOCK_SIZE], dealRowCount * FP32_ONE_BLOCK_SIZE);
    }
    inputQue2.EnQue(lseSum);
    inputQue1.EnQue(lseMax);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::CopyAccumOutIn(uint32_t splitKVIndex,
                                                                                    uint32_t startRow,
                                                                                    uint32_t dealRowCount)
{
    LocalTensor<T> accumOutLocal = inputQue1.AllocTensor<T>();

    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<T> copyInPadParams;
    copyInParams.blockCount = dealRowCount;
    copyInParams.blockLen = headDim * sizeof(T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = (headDimAlign - headDim) / BLOCK_ELEMENT_NUM;

    copyInPadParams.isPad = true;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = (headDimAlign - headDim) % BLOCK_ELEMENT_NUM;
    copyInPadParams.paddingValue = 0;

    combineAccumOutOffset =
        (bIdx * kvHeadNum * splitKVNum + n2Idx * splitKVNum + splitKVIndex) * gSize * headDim + startRow * headDim;
    DataCopyPad(accumOutLocal, accumOutGm[combineAccumOutOffset], copyInParams, copyInPadParams);
    inputQue1.EnQue(accumOutLocal);
}

template <typename IFAT>
__aicore__ inline void
SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::ComputeScaleValue(LocalTensor<T> &lseSum, LocalTensor<T> &lseMax,
                                                                uint32_t startRow, uint32_t dealRowCount)
{
    LocalTensor<T> lseMaxUb = softmaxMaxUb;
    LocalTensor<T> lseSumUb = softmaxSumUb;
    LocalTensor<T> lseExpUb = tmpBuff1.Get<T>();

    // lseLocal的shape为[actualCombineLoopSize,dealRowCount * FP32_ONE_BLOCK_SIZE]
    Duplicate(lseMaxUb, -FLOAT_MAX, dealRowCount * FP32_ONE_BLOCK_SIZE);
    Duplicate(lseSumUb, FLOAT_ZERO, dealRowCount * FP32_ONE_BLOCK_SIZE);
    pipe_barrier(PIPE_V);
    for (uint32_t i = 0; i < actualCombineLoopSize; ++i) {
        Max(lseMaxUb, lseMaxUb, lseMax[i * dealRowCount * FP32_ONE_BLOCK_SIZE], dealRowCount * FP32_ONE_BLOCK_SIZE);
        pipe_barrier(PIPE_V);
    }
    for (uint32_t i = 0; i < actualCombineLoopSize; ++i) {
        Sub(lseExpUb[i * dealRowCount * FP32_ONE_BLOCK_SIZE], lseMax[i * dealRowCount * FP32_ONE_BLOCK_SIZE], lseMaxUb,
            dealRowCount * FP32_ONE_BLOCK_SIZE);
    }
    pipe_barrier(PIPE_V);
    Exp(lseExpUb, lseExpUb, actualCombineLoopSize * dealRowCount * FP32_ONE_BLOCK_SIZE);
    pipe_barrier(PIPE_V);

    Mul(lseSum, lseSum, lseExpUb, actualCombineLoopSize * dealRowCount * FP32_ONE_BLOCK_SIZE);
    pipe_barrier(PIPE_V);

    for (uint32_t i = 0; i < actualCombineLoopSize; ++i) {
        Add(lseSumUb, lseSumUb, lseSum[i * dealRowCount * FP32_ONE_BLOCK_SIZE], dealRowCount * FP32_ONE_BLOCK_SIZE);
        pipe_barrier(PIPE_V);
    }

    for (uint32_t i = 0; i < actualCombineLoopSize; ++i) {
        Div(lseSum[i * dealRowCount * FP32_ONE_BLOCK_SIZE], lseSum[i * dealRowCount * FP32_ONE_BLOCK_SIZE], lseSumUb,
            dealRowCount * FP32_ONE_BLOCK_SIZE);
    }
    pipe_barrier(PIPE_V);

    if constexpr (SHARED_PREFIX) {
        SysPrefixSaveLseFd(lseSumUb, lseMaxUb, startRow, dealRowCount);
    } else if (softmaxLseFlag) {
        LocalTensor<T> softmaxlseUb = outputQue2.template AllocTensor<T>();
        Log(softmaxlseUb, lseSumUb, dealRowCount * FP32_ONE_BLOCK_SIZE);
        pipe_barrier(PIPE_V);
        Add(softmaxlseUb, softmaxlseUb, lseMaxUb, dealRowCount * FP32_ONE_BLOCK_SIZE);
        pipe_barrier(PIPE_V);
        outputQue2.EnQue(softmaxlseUb);
        outputQue2.DeQue<T>();

        DataCopyExtParams intriParams1;
        intriParams1.blockLen = sizeof(T);
        intriParams1.blockCount = dealRowCount;
        intriParams1.srcStride = 0;
        intriParams1.dstStride = 0;
        DataCopyPad(softmaxLseGm[bIdx * kvHeadNum * gSize + n2Idx * gSize + startRow], softmaxlseUb, intriParams1);
        outputQue2.FreeTensor(softmaxlseUb);
    }
}

template <typename IFAT>
__aicore__ inline void
SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::ReduceFinalRes(LocalTensor<T> &dst, LocalTensor<T> &lseLocal,
                                                             uint32_t startRow, uint32_t dealRowCount)
{
    BinaryRepeatParams repeatParams;
    repeatParams.src0RepStride = 1;
    repeatParams.src0BlkStride = 0;
    repeatParams.src1RepStride = headDimAlign / FP32_ONE_BLOCK_SIZE;
    repeatParams.dstRepStride = headDimAlign / FP32_ONE_BLOCK_SIZE;
    int32_t dtypeMask = 256 / sizeof(float);
    int32_t mulLoop = headDimAlign / dtypeMask;
    int32_t mulRemain = headDimAlign % dtypeMask;

    // 第一次，mul结果直接放到dst里
    CopyAccumOutIn(0, startRow, dealRowCount);
    LocalTensor<T> accumOutLocal = inputQue1.DeQue<T>();
    for (int i = 0; i < mulLoop; i++) {
        Mul(dst[i * dtypeMask], lseLocal, accumOutLocal[i * dtypeMask], dtypeMask, dealRowCount, repeatParams);
    }
    if (mulRemain > 0) {
        Mul(dst[mulLoop * dtypeMask], lseLocal, accumOutLocal[mulLoop * dtypeMask], mulRemain, dealRowCount,
            repeatParams);
    }
    pipe_barrier(PIPE_V);
    inputQue1.FreeTensor(accumOutLocal);

    for (uint32_t j = 1; j < actualCombineLoopSize; ++j) {
        CopyAccumOutIn(j, startRow, dealRowCount);
        LocalTensor<T> accumOutLocal = inputQue1.DeQue<T>();
        for (int i = 0; i < mulLoop; i++) {
            Mul(accumOutLocal[i * dtypeMask], lseLocal[j * dealRowCount * FP32_ONE_BLOCK_SIZE],
                accumOutLocal[i * dtypeMask], dtypeMask, dealRowCount, repeatParams);
        }
        if (mulRemain > 0) {
            Mul(accumOutLocal[mulLoop * dtypeMask], lseLocal[j * dealRowCount * FP32_ONE_BLOCK_SIZE],
                accumOutLocal[mulLoop * dtypeMask], mulRemain, dealRowCount, repeatParams);
        }
        pipe_barrier(PIPE_V);
        Add(dst, dst, accumOutLocal, dealRowCount * headDimAlign);
        pipe_barrier(PIPE_V);
        // pipe_barrier(PIPI_V)与inputQue1.FreeTensor之间没有关系，这里的PIPE_V是为了让Add和接下来的VEC指令隔开
        inputQue1.FreeTensor(accumOutLocal);
    }
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::CopyFinalResOut(LocalTensor<T> &accumOutLocal,
                                                                                     uint32_t startRow,
                                                                                     uint32_t dealRowCount)
{
    if constexpr (!POST_QUANT) {
        LocalTensor<OUT_T> tmpBmm2ResCastTensor = outputQue1.AllocTensor<OUT_T>();
        uint32_t shapeArray[] = {dealRowCount, (uint32_t)headDim};
        tmpBmm2ResCastTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
        if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
            Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_RINT, dealRowCount * headDimAlign);
        } else {
            Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_ROUND, dealRowCount * headDimAlign);
        }

        outputQue1.EnQue(tmpBmm2ResCastTensor);
        outputQue1.DeQue<OUT_T>();
        Bmm2DataCopyOut(tmpBmm2ResCastTensor, startRow, dealRowCount, headDimAlign, headDim);
        outputQue1.FreeTensor(tmpBmm2ResCastTensor);
    } else {
        LocalTensor<OUT_T> bmm2ResUbInt8 = outputQue1.AllocTensor<OUT_T>();
        uint32_t shapeArray[] = {dealRowCount, (uint32_t)headDim};
        bmm2ResUbInt8.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
        PostQuant(accumOutLocal, bmm2ResUbInt8, startRow, dealRowCount, headDimAlign, headDim);
        outputQue1.EnQue(bmm2ResUbInt8);
        outputQue1.DeQue<OUT_T>();
        Bmm2DataCopyOut(bmm2ResUbInt8, startRow, dealRowCount, headDimAlign, headDim);
        outputQue1.FreeTensor(bmm2ResUbInt8);
    }
}

template <typename IFAT> __aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::CombineSplitKVRes()
{
    if (curActualSeqLen != 0) {
        uint32_t gSplitSizeLse = BUFFER_SIZE_BYTE_16K / (BYTE_BLOCK * splitKVNum);
        uint32_t gSplitSizeAccumOut = BASE_BLOCK_MAX_ELEMENT_NUM / headDimAlign;
        // 取两者较小的，用来切g，保证ub够用
        uint32_t gSplitSize = (gSplitSizeLse < gSplitSizeAccumOut) ? gSplitSizeLse : gSplitSizeAccumOut;
        gSplitSize = (gSplitSize > gSize) ? gSize : gSplitSize; // 最大为gSize
        uint32_t loopCount = (gSize + gSplitSize - 1) / gSplitSize;
        uint32_t tailSplitSize = gSize - (loopCount - 1) * gSplitSize;

        // 尾块与非尾块都使用这些ub，减少处理次数
        for (uint32_t i = 0, actualGSplitSize = gSplitSize; i < loopCount; i++) {
            uint32_t startRow = i * gSplitSize;
            if ((i + 1) == loopCount) {
                actualGSplitSize = tailSplitSize;
            }
            CopyLseIn(startRow, actualGSplitSize);
            LocalTensor<T> lseSum = inputQue2.DeQue<T>();
            LocalTensor<T> lseMax = inputQue1.DeQue<T>();
            ComputeScaleValue(lseSum, lseMax, startRow, actualGSplitSize);
            inputQue1.FreeTensor(lseMax);

            uint32_t gSplitBmm2UbSize = headDimAlign * actualGSplitSize;
            LocalTensor<T> tmp1 = tmpBuff1.Get<T>(gSplitBmm2UbSize);
            ReduceFinalRes(tmp1, lseSum, startRow, actualGSplitSize);
            inputQue2.FreeTensor(lseSum);

            if constexpr (SHARED_PREFIX) {
                SysPrefixSaveAttenRes(bIdx, n2Idx, tmp1, startRow, actualGSplitSize, calcSysPrefixFlag);
            } else {
                CopyFinalResOut(tmp1, startRow, actualGSplitSize);
            }
        }
    }
}

template <typename IFAT> __aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::FlashDecodeCompute()
{
    bIdx = tmpBlockIdx / kvHeadNum;
    n2Idx = tmpBlockIdx % kvHeadNum;
    attenOutOffset = bIdx * kvHeadNum * gSize * headDim + n2Idx * gSize * headDim;
    perChannelQuantOffset = n2Idx * headDim * gSize;
    if (tmpBlockIdx >= batchSize * kvHeadNum) {
        return;
    }

    if (actualLenDims == 0) {
        curActualSeqLen = kvSeqSize;
        if (!batchContinuous) {
            curActualSeqLen = SeqLenFromTensorList(bIdx);
        }
    } else if (actualLenDims == 1) {
        curActualSeqLen = actualSeqLengthsGm.GetValue(0);
    } else {
        curActualSeqLen = actualSeqLengthsGm.GetValue(bIdx*8);
    }

    actualCombineLoopSize = (curActualSeqLen + sInnerLoopSize - 1) / sInnerLoopSize;

    if (SHARED_PREFIX) {
        if (calcSysPrefixFlag) {
            for (bIdx = 0; bIdx < batchSizeQ; bIdx++) {
                CombineSplitKVRes();
            }
            return;
        }
    }

    CombineSplitKVRes();
}

template <typename IFAT>
__aicore__ inline void
SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::ComputeLogSumExpAndCopyToGm(LocalTensor<T> &softmaxSumUb,
                                                                          LocalTensor<T> &softmaxMaxUb)
{
    size_t size = gSize * FP32_ONE_BLOCK_SIZE;
    size_t offset = bIdx * kvHeadNum * splitKVNum * gSize * FP32_ONE_BLOCK_SIZE +
                    n2Idx * splitKVNum * gSize * FP32_ONE_BLOCK_SIZE + s2Idx * gSize * FP32_ONE_BLOCK_SIZE;

    CopyFixedUbToGm(lseSumFdGm[offset], softmaxSumUb, size);

    if constexpr (ANTIQUANT && (ANTIQUANT_PER_CHANNEL || ANTIQUANT_PER_CHANNEL_TOKEN)) {
        if (softmaxLseFlag && antiqOffsetExistFlag) {
            // per chnnel msd mm1 计算优化舍弃了offset，输出lse需要补回，以保持和公式一致
            Muls(qRowSumUb, qRowSumUb, static_cast<T>(tilingData->baseParams.scaleValue), gSize * FP32_ONE_BLOCK_SIZE);
            pipe_barrier(PIPE_V);
            Add(softmaxMaxUb, softmaxMaxUb, qRowSumUb, gSize * FP32_ONE_BLOCK_SIZE);
            pipe_barrier(PIPE_V);
        }
    }

    CopyFixedUbToGm(lseMaxFdGm[offset], softmaxMaxUb, size);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SoftmaxLseCopyOut(LocalTensor<T> &softmaxSumUb,
                                                                                       LocalTensor<T> &softmaxMaxUb)
{
    LocalTensor<T> lseUb = tmpBuff3.Get<T>(gSize * FP32_ONE_BLOCK_SIZE);
    Log(lseUb, softmaxSumUb, gSize * FP32_ONE_BLOCK_SIZE);
    pipe_barrier(PIPE_V);
    Add(lseUb, lseUb, softmaxMaxUb, gSize * FP32_ONE_BLOCK_SIZE);
    pipe_barrier(PIPE_V);

    if constexpr (ANTIQUANT && (ANTIQUANT_PER_CHANNEL || ANTIQUANT_PER_CHANNEL_TOKEN)) {
        if (softmaxLseFlag && antiqOffsetExistFlag) {
            // per chnnel msd mm1 计算优化舍弃了offset，输出lse需要补回，以保持和公式一致
            Muls(qRowSumUb, qRowSumUb, static_cast<T>(tilingData->baseParams.scaleValue), gSize * FP32_ONE_BLOCK_SIZE);
            pipe_barrier(PIPE_V);
            Add(lseUb, lseUb, qRowSumUb, gSize * FP32_ONE_BLOCK_SIZE);
            pipe_barrier(PIPE_V);
        }
    }

    LocalTensor<T> softmaxlseUb = outputQue2.template AllocTensor<T>();
    DataCopy(softmaxlseUb, lseUb, gSize * FP32_ONE_BLOCK_SIZE);
    outputQue2.EnQue(softmaxlseUb);
    outputQue2.DeQue<T>();

    DataCopyExtParams intriParams1;
    intriParams1.blockLen = sizeof(T);
    intriParams1.blockCount = gSize;
    intriParams1.srcStride = 0;
    intriParams1.dstStride = 0;
    DataCopyPad(softmaxLseGm[bIdx * kvHeadNum * gSize + n2Idx * gSize], softmaxlseUb, intriParams1);
    outputQue2.FreeTensor(softmaxlseUb);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::Bmm1Compute(const uint32_t bn2Idx,
                                                                                 const uint32_t sInnerLoopIdx)
{
    if (SHARED_PREFIX) {
        if (calcSysPrefixFlag) {
            SysPrefixBmm1Compute(bn2Idx, sInnerLoopIdx);
            return;
        }
    }
    Bmm1ComputeCommon(bn2Idx, sInnerLoopIdx);
}
template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::Bmm1ComputeCommon(const uint32_t bn2Idx,
                                                                                       const uint32_t sInnerLoopIdx)
{
    if constexpr (PAGE_ATTENTION) {
        bmm1PageAttentionDataUb = bmm1PageAttentionDataBuff.Get<uint32_t>();
        bmm1PageAttentionDataUb.SetValue(0, bIdx);
        bmm1PageAttentionDataUb.SetValue(1, n2Idx);
        bmm1PageAttentionDataUb.SetValue(2, sInnerLoopIdx);
        
        // DataCopy 不支持64位拷贝，2个gm地址需在V侧设置时拆分，在回调里拼接
        bmm1PageAttentionDataUb.SetValue(3, (uint32_t)((reinterpret_cast<uint64_t>(key_) >> 32) & 0x00000000ffffffff));
        bmm1PageAttentionDataUb.SetValue(4, (uint32_t)(reinterpret_cast<uint64_t>(key_)));
        
        bmm1PageAttentionDataUb.SetValue(
            5, (uint32_t)((reinterpret_cast<uint64_t>(blocktablePtr) >> 32) & 0x00000000ffffffff));
        bmm1PageAttentionDataUb.SetValue(6, (uint32_t)(reinterpret_cast<uint64_t>(blocktablePtr)));

        // 添加调试日志：打印blockPosition地址信息
        //  V5_DEBUG_PRINTF("Bmm1ComputeCommon: blocktablePtr=%p, high=%u, low=%u\n", blocktablePtr,
                        // (uint32_t)((reinterpret_cast<uint64_t>(blocktablePtr) >> 32) & 0x00000000ffffffff),
                        // (uint32_t)(reinterpret_cast<uint64_t>(blocktablePtr)));

        // PA 新增，BlockPosition 支持
        bmm1PageAttentionDataUb.SetValue(
            7, (uint32_t)((reinterpret_cast<uint64_t>(blockpositionPtr) >> 32) & 0x00000000ffffffff));
        bmm1PageAttentionDataUb.SetValue(8, (uint32_t)(reinterpret_cast<uint64_t>(blockpositionPtr)));
        
        // 添加调试日志：打印blockPosition地址信息
        //  V5_DEBUG_PRINTF("Bmm1ComputeCommon: blockpositionPtr=%p, high=%u, low=%u\n", 
                        // blockpositionPtr, 
                        // (uint32_t)((reinterpret_cast<uint64_t>(blockpositionPtr) >> 32) & 0x00000000ffffffff),
                        // (uint32_t)(reinterpret_cast<uint64_t>(blockpositionPtr)));
        
        // 传入 uint64_t curActualSeqLen 
        bmm1PageAttentionDataUb.SetValue(
            9, (uint32_t)((reinterpret_cast<uint64_t>(curActualSeqLen) >> 32) & 0x00000000ffffffff));
        bmm1PageAttentionDataUb.SetValue(10, (uint32_t)(reinterpret_cast<uint64_t>(curActualSeqLen)));


        event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);

        DataCopy(bmm1CallBackDataGm, bmm1PageAttentionDataUb, 16); // 对齐

        event_t eventIDMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
        WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);

        mm.SetSelfDefineData(reinterpret_cast<uint64_t>(bmm1CallBackDataPtr));
    }
    
    if constexpr (ANTIQUANT) {
        mm.SetTensorA(queryPreProcessResGm);
    } else {
        mm.SetTensorA(queryGm[tensorACoreOffset]);
    }
    mm.SetTensorB(keyGm[tensorBOffset], true);

    mm.SetTail(msdIterNum * gSize, actualSingleProcessSInnerSize, headDim);
    mm.template IterateAll<false>(mm1ResGm, false, false, true);
    mm.WaitIterateAll();
    mm.End();
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SysPrefixBmm1Compute(const uint32_t bn2Idx,
                                                                                          const uint32_t sInnerLoopIdx)
{
    if constexpr (ANTIQUANT) {
        mm1Sp.SetTensorA(queryPreProcessResGm);
    } else {
        mm1Sp.SetTensorA(prefixQueryPreProcessResGm);
    }

    mm1Sp.SetTensorB(keyGm[tensorBOffset], true);

    uint32_t M = msdIterNum * gSize * batchSizeQ;

    mm1Sp.SetTail(M, actualSingleProcessSInnerSize, headDim);
    mm1Sp.template IterateAll<false>(mm1ResGm, false, false, true);
    mm1Sp.WaitIterateAll();
    mm1Sp.End();
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::Bmm2Compute(const uint32_t bn2Idx,
                                                                                 const uint32_t sInnerLoopIdx)
{
    if (SHARED_PREFIX) {
        if (calcSysPrefixFlag) {
            SysPrefixBmm2Compute(bn2Idx, sInnerLoopIdx);
            return;
        }
    }
    Bmm2ComputeCommon(bn2Idx, sInnerLoopIdx);
}
template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::Bmm2ComputeCommon(const uint32_t bn2Idx,
                                                                                       const uint32_t sInnerLoopIdx)
{
    if constexpr (PAGE_ATTENTION) {
        bmm2PageAttentionDataUb = bmm2PageAttentionDataBuff.Get<uint32_t>();
        bmm2PageAttentionDataUb.SetValue(0, bIdx);
        bmm2PageAttentionDataUb.SetValue(1, n2Idx);
        bmm2PageAttentionDataUb.SetValue(2, sInnerLoopIdx);
        // DataCopy 不支持64位拷贝，2个gm地址需在V侧设置时拆分，在回调里拼接
        bmm2PageAttentionDataUb.SetValue(3,
                                         (uint32_t)((reinterpret_cast<uint64_t>(value_) >> 32) & 0x00000000ffffffff));
        bmm2PageAttentionDataUb.SetValue(4, (uint32_t)(reinterpret_cast<uint64_t>(value_)));
        bmm2PageAttentionDataUb.SetValue(
            5, (uint32_t)((reinterpret_cast<uint64_t>(blocktablePtr) >> 32) & 0x00000000ffffffff));
        bmm2PageAttentionDataUb.SetValue(6, (uint32_t)(reinterpret_cast<uint64_t>(blocktablePtr)));

        bmm2PageAttentionDataUb.SetValue(
            7, (uint32_t)((reinterpret_cast<uint64_t>(blockpositionPtr) >> 32) & 0x00000000ffffffff));
        bmm2PageAttentionDataUb.SetValue(8, (uint32_t)(reinterpret_cast<uint64_t>(blockpositionPtr)));
            
        // 添加调试日志：打印blockPosition地址信息
        //  V5_DEBUG_PRINTF("Bmm2ComputeCommon: blockpositionPtr=%p, high=%u, low=%u\n", 
                        // blockpositionPtr, 
                        // (uint32_t)((reinterpret_cast<uint64_t>(blockpositionPtr) >> 32) & 0x00000000ffffffff),
                        // (uint32_t)(reinterpret_cast<uint64_t>(blockpositionPtr)));

        // 传入 uint64_t curActualSeqLen 
        bmm2PageAttentionDataUb.SetValue(
            9, (uint32_t)((reinterpret_cast<uint64_t>(curActualSeqLen) >> 32) & 0x00000000ffffffff));
        bmm2PageAttentionDataUb.SetValue(10, (uint32_t)(reinterpret_cast<uint64_t>(curActualSeqLen)));


        event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);

        DataCopy(bmm2CallBackDataGm, bmm2PageAttentionDataUb, 16); // 对齐

        event_t eventIDMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
        WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);

        bmm2.SetSelfDefineData(reinterpret_cast<uint64_t>(bmm2CallBackDataPtr));
    }
    
    bmm2.SetTensorA(vec1ResGm);
    bmm2.SetTensorB(valueGm[valueOffset]);
    if constexpr (KVINT4) {
        if (actualSingleProcessSInnerSize % 2 == 1) {
            bmm2.SetTail(msdIterNum * gSize, headDim, actualSingleProcessSInnerSize + 1);
        } else {
            bmm2.SetTail(msdIterNum * gSize, headDim, actualSingleProcessSInnerSize);
        }
    } else {
        bmm2.SetTail(msdIterNum * gSize, headDim, actualSingleProcessSInnerSize);
    }
    bmm2.template IterateAll<false>(mm2ResGm, false, false, true);
    bmm2.WaitIterateAll();
    bmm2.End();
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SysPrefixBmm2Compute(const uint32_t bn2Idx,
                                                                                          const uint32_t sInnerLoopIdx)
{
    mm2Sp.SetTensorA(vec1ResGm);
    mm2Sp.SetTensorB(valueGm[valueOffset]);

    uint32_t M = msdIterNum * gSize * batchSizeQ;
    if constexpr (KVINT4) {
        if (actualSingleProcessSInnerSize % 2 == 1) {
            mm2Sp.SetTail(M, headDim, actualSingleProcessSInnerSize + 1);
        } else {
            mm2Sp.SetTail(M, headDim, actualSingleProcessSInnerSize);
        }
    } else {
        mm2Sp.SetTail(M, headDim, actualSingleProcessSInnerSize);
    }
    mm2Sp.template IterateAll<false>(mm2ResGm, false, false, true);
    mm2Sp.WaitIterateAll();
    mm2Sp.End();
}

template <typename IFAT>
__aicore__ inline void
SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::ElewiseCompute(LocalTensor<T> &mmResUb, TBuf<> &tmpBuf, uint32_t startRow,
                                                             uint32_t dealRowCount, uint32_t columnCount,
                                                             uint32_t actualColumnCount)
{
    Muls(mmResUb, mmResUb, static_cast<T>(tilingData->baseParams.scaleValue), dealRowCount * columnCount);
    pipe_barrier(PIPE_V);

    // pse shift mask
    if (pseShiftFlag) {
        PseShiftCopyIn(startRow, dealRowCount, actualColumnCount);
        LocalTensor<pseShiftType> pseShiftUb = inputQue1.DeQue<pseShiftType>();
        LocalTensor<float> pseShiftUbFloat = tmpBuf.Get<float>();
        for (uint32_t i = 0; i < dealRowCount; ++i) {
            Cast(pseShiftUbFloat[i * columnCount], pseShiftUb[i * pseMaskSizeAlign], AscendC::RoundMode::CAST_NONE,
                 pseMaskSizeAlign);
        }

        inputQue1.FreeTensor(pseShiftUb);
        pipe_barrier(PIPE_V);
        Add(mmResUb, mmResUb, pseShiftUbFloat, dealRowCount * columnCount);
        pipe_barrier(PIPE_V);
    }

    // attenMask
    if (attenMaskFlag) {
        AttenMaskCopyIn(attenMaskOffset, dealRowCount, actualColumnCount);
        LocalTensor<bool> attenMaskUb = inputQue2.DeQue<bool>();
        for (int i = 1; i < dealRowCount; i++) {
            DataCopy(attenMaskUb[i * attenMaskSizeAlign], attenMaskUb, attenMaskSizeAlign);
        }
        pipe_barrier(PIPE_V);

        LocalTensor<uint8_t> ubWorkSpace = tmpBuf.Get<uint8_t>(selectWithByteMaskTmpMinSize);
        SelectWithBytesMaskShapeInfo selectWithBytesMaskShapeInfo;
        selectWithBytesMaskShapeInfo.firstAxis = dealRowCount;
        selectWithBytesMaskShapeInfo.srcLastAxis = columnCount;
        selectWithBytesMaskShapeInfo.maskLastAxis = attenMaskSizeAlign;
        attenMaskUb.SetSize(dealRowCount * attenMaskSizeAlign); // Select接口要求mask size与参数匹配
        mmResUb.SetSize(dealRowCount * columnCount);            // Select接口要求src size与参数匹配
        SelectWithBytesMask(mmResUb, mmResUb, BOOL_ATTEN_MASK_SCALAR_VALUE, attenMaskUb, ubWorkSpace,
                            selectWithBytesMaskShapeInfo);
        mmResUb.SetSize(BUFFER_SIZE_BYTE_32K / sizeof(T)); // mmResUb Size复原,mask不用复原,与原来一致
        inputQue2.FreeTensor(attenMaskUb);

        pipe_barrier(PIPE_V);
    }
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SoftmaxFlashV2Compute(
    LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb, uint32_t startRow, uint32_t dealRowCount,
    uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t baseOffset = startRow * BLOCK_ELEMENT_NUM;
    SoftMaxShapeInfo srcShape{dealRowCount, columnCount, dealRowCount, actualColumnCount};
    SoftMaxTiling newTiling =
        SoftMaxFlashV2TilingFunc(srcShape, sizeof(T), sizeof(T), softmaxTmpUb.GetSize(), true, false);
    SoftmaxFlashV2<T, true, true, false, false, IFA_SOFTMAX_FLASHV2_CFG>(
        mmResUb, softmaxSumUb[baseOffset], softmaxMaxUb[baseOffset], mmResUb, softmaxExpUb[baseOffset],
        softmaxSumUb[baseOffset], softmaxMaxUb[baseOffset], softmaxTmpUb, newTiling, srcShape);
}

template <typename IFAT>
__aicore__ inline void
SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::Bmm2FDDataCopyOut(LocalTensor<T> &attenOutUb, uint32_t startRow,
                                                                uint32_t dealRowCount, uint32_t columnCount,
                                                                uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(T));
    dataCopyParams.dstStride = 0;

    LocalTensor<T> tmp = outputQue1.AllocTensor<T>();
    DataCopy(tmp, attenOutUb, columnCount * dealRowCount);
    outputQue1.EnQue(tmp);
    outputQue1.DeQue<T>();

    size_t base = (bIdx * qHeadNum * headDim + n2Idx * gSize * headDim) * splitKVNum;
    DataCopyPad(accumOutGm[base + s2Idx * gSize * actualColumnCount + startRow * actualColumnCount], tmp,
                dataCopyParams);
    outputQue1.FreeTensor(tmp);
}

template <typename IFAT>
__aicore__ inline void
SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::Bmm2DataCopyOut(LocalTensor<OUT_T> &attenOutUb, uint32_t startRow,
                                                              uint32_t dealRowCount, uint32_t columnCount,
                                                              uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUT_T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(OUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(attentionOutGm[attenOutOffset + startRow * actualColumnCount], attenOutUb, dataCopyParams);
}

template <typename IFAT>
__aicore__ inline void
SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::Bmm2CastAndCopyOut(LocalTensor<T> &bmm2ResUb, uint32_t startRow,
                                                                 uint32_t dealRowCount, uint32_t columnCount,
                                                                 uint32_t actualColumnCount)
{
    if constexpr (FLASH_DECODE) {
        if (flashDecodeFlag) {
            Bmm2FDDataCopyOut(bmm2ResUb, startRow, dealRowCount, columnCount, actualColumnCount);
            return;
        }
    }

    if constexpr (SHARED_PREFIX) {
        SysPrefixSaveAttenRes(bIdx, n2Idx, bmm2ResUb, startRow, dealRowCount, calcSysPrefixFlag);
    } else {
        if constexpr (!POST_QUANT) {
            LocalTensor<OUT_T> tmpBmm2ResCastTensor = outputQue1.AllocTensor<OUT_T>();
            if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
                Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_RINT, dealRowCount * columnCount);
            } else {
                Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_ROUND, dealRowCount * columnCount);
            }
            outputQue1.EnQue(tmpBmm2ResCastTensor);
            outputQue1.DeQue<OUT_T>();
            Bmm2DataCopyOut(tmpBmm2ResCastTensor, startRow, dealRowCount, columnCount, actualColumnCount);
            outputQue1.FreeTensor(tmpBmm2ResCastTensor);
        } else {
            LocalTensor<OUT_T> bmm2ResUbInt8 = outputQue1.AllocTensor<OUT_T>();
            PostQuant(bmm2ResUb, bmm2ResUbInt8, startRow, dealRowCount, columnCount, actualColumnCount);
            outputQue1.EnQue(bmm2ResUbInt8);
            outputQue1.DeQue<OUT_T>();
            Bmm2DataCopyOut(bmm2ResUbInt8, startRow, dealRowCount, columnCount, actualColumnCount);
            outputQue1.FreeTensor(bmm2ResUbInt8);
        }
    }
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::PseShiftCopyIn(uint32_t startRow,
                                                                                    uint32_t rowCount,
                                                                                    uint32_t actualColumnCount)
{
    LocalTensor<pseShiftType> pseShiftUb = inputQue1.AllocTensor<pseShiftType>();
    pseMaskSizeAlign = Align(actualColumnCount, 16U); // 16: align to 32bytes
    uint32_t computeSize = rowCount * pseMaskSizeAlign;
    pseShiftUb.SetSize(computeSize);
#if (__CCE_AICORE__ > 200)
    if (actualColumnCount % 16 == 0) {
        for (uint32_t i = 0; i < rowCount; ++i) {
            DataCopy(pseShiftUb[i * pseMaskSizeAlign],
                     pseShiftGm[pseShiftOffset + startRow * pseShiftS + i * pseShiftS], pseMaskSizeAlign);
        }
    } else {
        uint32_t typeElementSize = BYTE_BLOCK / sizeof(pseShiftType);
        DataCopyExtParams intriParams;
        intriParams.blockLen = actualColumnCount * sizeof(pseShiftType);
        intriParams.blockCount = 1;
        intriParams.dstStride = (pseMaskSizeAlign - actualColumnCount) / typeElementSize;
        intriParams.srcStride = 0;
        DataCopyPadExtParams<pseShiftType> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = (pseMaskSizeAlign - actualColumnCount) % typeElementSize;
        padParams.paddingValue = 0;
        for (uint32_t i = 0; i < rowCount; ++i) {
            DataCopyPad(pseShiftUb[i * pseMaskSizeAlign],
                        pseShiftGm[pseShiftOffset + startRow * pseShiftS + i * pseShiftS], intriParams, padParams);
        }
    }
#else
    for (uint32_t i = 0; i < rowCount; ++i) {
        DataCopy(pseShiftUb[i * pseMaskSizeAlign], pseShiftGm[pseShiftOffset + startRow * pseShiftS + i * pseShiftS],
                 pseMaskSizeAlign);
    }
#endif
    inputQue1.EnQue(pseShiftUb);
}

template <typename IFAT>
__aicore__ inline void
SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::DealBmm1ResBaseBlock(const uint32_t sInnerLoopIdx, uint32_t startRow,
                                                                   uint32_t dealRowCount, uint32_t columnCount,
                                                                   uint32_t actualColumnCount)
{
    uint32_t computeSize = dealRowCount * columnCount;
    LocalTensor<T> mmResUb = tmpBuff1.Get<T>();
    uint64_t batchBase = 0;

    if constexpr (SHARED_PREFIX) {
        if (calcSysPrefixFlag) {
            batchBase = bIdx * gSize * columnCount;
        }
    }
    LocalTensor<MM_OUT_T> tmpMmResUb = inputQue1.AllocTensor<MM_OUT_T>();
    DataCopy(tmpMmResUb, mm1ResGm[batchBase + startRow * columnCount], computeSize);
    inputQue1.EnQue(tmpMmResUb);
    inputQue1.DeQue<MM_OUT_T>();
    DataCopy(mmResUb, tmpMmResUb, computeSize);
    inputQue1.FreeTensor(tmpMmResUb);
    pipe_barrier(PIPE_V);

    ElewiseCompute(mmResUb, tmpBuff2, startRow, dealRowCount, columnCount, actualColumnCount);

    LocalTensor<T> tmpAFloorUb = tmpBuff2.Get<T>();
    LocalTensor<uint8_t> softmaxTmpUb = tmpAFloorUb.template ReinterpretCast<uint8_t>();
    SoftmaxFlashV2Compute(mmResUb, softmaxTmpUb, startRow, dealRowCount, columnCount, actualColumnCount);
    pipe_barrier(PIPE_V);

    LocalTensor<KV_T> tmpMMResCastTensor = outputQue1.AllocTensor<KV_T>();
    Cast(tmpMMResCastTensor, mmResUb, AscendC::RoundMode::CAST_ROUND, computeSize);

    outputQue1.EnQue(tmpMMResCastTensor);
    outputQue1.DeQue<KV_T>();
    DataCopy(vec1ResGm[batchBase + startRow * columnCount], tmpMMResCastTensor, computeSize);
    outputQue1.FreeTensor(tmpMMResCastTensor);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::AntiquantMatmulResCombine(
    LocalTensor<T> bmmResUb, GlobalTensor<MM_OUT_T> srcGm, uint32_t startRow, uint32_t dealRowCount,
    uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t step = gSize * columnCount;
    uint32_t baseOffset = startRow * columnCount;
    uint32_t copySize = dealRowCount * columnCount;

    if constexpr (SHARED_PREFIX) {
        if (calcSysPrefixFlag) {
            baseOffset += bIdx * gSize * msdIterNum * columnCount;
        }
    }

    T scale = 1;
    uint32_t offset = baseOffset;
    for (uint32_t i = 0; i < msdIterNum; i++) {
        LocalTensor<MM_OUT_T> tmpCInt = inputQue1.AllocTensor<MM_OUT_T>();
        DataCopy(tmpCInt, srcGm[offset], copySize); // offset = i * step + baseOffset
        inputQue1.template EnQue(tmpCInt);

        tmpCInt = inputQue1.DeQue<MM_OUT_T>();
        if (i == 0) {
            Cast(bmmResUb, tmpCInt, AscendC::RoundMode::CAST_NONE, copySize);
        } else {
            LocalTensor<T> tmpCFp;
            tmpCFp = tmpCInt.template ReinterpretCast<T>();
            tmpCFp.SetSize(tmpCInt.GetSize());
            Cast(tmpCFp, tmpCInt, AscendC::RoundMode::CAST_NONE, copySize);
            pipe_barrier(PIPE_V);
            Muls(tmpCFp, tmpCFp, scale, copySize);
            pipe_barrier(PIPE_V);
            Add(bmmResUb, bmmResUb, tmpCFp, copySize);
        }
        inputQue1.FreeTensor(tmpCInt);

        offset += step;
        scale = scale / antiquantExpandCoeff;
    }
    pipe_barrier(PIPE_V);

    // muls 1/antiqCoeff1
    Muls(bmmResUb, bmmResUb, antiqCoeff2, copySize);
}

template <typename IFAT>
__aicore__ inline void
SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::DealAntiqBmm1ResBaseBlock(const uint32_t sInnerLoopIdx, uint32_t startRow,
                                                                        uint32_t dealRowCount, uint32_t columnCount,
                                                                        uint32_t actualColumnCount)
{
    LocalTensor<T> mmResUb = tmpBuff1.Get<T>();
    LocalTensor<T> aMax = aMaxBmm1Ub[startRow * BLOCK_ELEMENT_NUM];
    AntiquantMatmulResCombine(mmResUb, mm1ResGm, startRow, dealRowCount, columnCount, actualColumnCount);
    pipe_barrier(PIPE_V);
    RowMuls(mmResUb, mmResUb, aMax, dealRowCount, columnCount, actualColumnCount);
    pipe_barrier(PIPE_V);

    // mul scalar and mask
    ElewiseCompute(mmResUb, tmpBuff2, startRow, dealRowCount, columnCount, actualColumnCount);

    LocalTensor<T> tmpAFloorUb = tmpBuff2.Get<T>();
    LocalTensor<uint8_t> softmaxTmpUb = tmpAFloorUb.template ReinterpretCast<uint8_t>();
    SoftmaxFlashV2Compute(mmResUb, softmaxTmpUb, startRow, dealRowCount, columnCount, actualColumnCount);
    pipe_barrier(PIPE_V);
    DealKvInt4ColumnOdd(actualColumnCount);

    size_t dstOffset = 0;
    if constexpr (SHARED_PREFIX) {
        if (calcSysPrefixFlag) {
            dstOffset = bIdx * gSize * msdIterNum * columnCount;
        }
    }

    AntiquantSoftmaxResPreProcess(vec1ResGm[dstOffset], mmResUb, tmpAFloorUb, startRow, dealRowCount, columnCount,
                                  actualColumnCount);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::DealAntiqBmm1ResBaseBlockChannelToken(
    const uint32_t sInnerLoopIdx, uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
    uint32_t actualColumnCount) {
  LocalTensor<T> mmResUb = tmpBuff1.Get<T>();
  LocalTensor<T> aMax = aMaxBmm1Ub[startRow * BLOCK_ELEMENT_NUM];
  uint32_t baseOffset = startRow * BLOCK_ELEMENT_NUM;
  AntiquantMatmulResCombine(mmResUb, mm1ResGm, startRow, dealRowCount, columnCount, actualColumnCount); // 仅仅合并结果
  pipe_barrier(PIPE_V);
  RowMuls(mmResUb, mmResUb, aMax, dealRowCount, columnCount, actualColumnCount); // 乘以行最大值
  pipe_barrier(PIPE_V);

  // mul scalar and mask
  ElewiseCompute(mmResUb, tmpBuff2, startRow, dealRowCount, columnCount, actualColumnCount);

  LocalTensor<T> tmpAFloorUb = tmpBuff2.Get<T>();
  LocalTensor<uint8_t> softmaxTmpUb = tmpAFloorUb.template ReinterpretCast<uint8_t>();
  SoftmaxFlashV2Compute(mmResUb, softmaxTmpUb, startRow, dealRowCount, columnCount, actualColumnCount); // 计算softmax但未除以sum
  pipe_barrier(PIPE_V);
  DealKvInt4ColumnOdd(actualColumnCount); // plus

  size_t dstOffset = 0;
  if constexpr (SHARED_PREFIX) {
    if (calcSysPrefixFlag) {
      dstOffset = bIdx * gSize * msdIterNum * columnCount;
    }
  }

  // mmResUb mul scale
  CopyAntiquantParamsPerToken(valueAntiqScaleGm, antiqParamOffsetPerToken, columnCount, actualColumnCount); // mm2时，乘上valScale，再分块。
  LocalTensor<T> antiqScalePerTokenUb = inputQue1.DeQue<T>();
  VecMulMat(mmResUb, antiqScalePerTokenUb, mmResUb, dealRowCount, columnCount, actualColumnCount);
  pipe_barrier(PIPE_V);
  inputQue1.FreeTensor(antiqScalePerTokenUb);
  Adds(tmpAFloorUb, mmResUb, (T)0, dealRowCount * columnCount);  // mmResUb need to be stored
  pipe_barrier(PIPE_V);
  if (antiqOffsetExistFlag) {
    LocalTensor<T> tmpAMax = tmpBuff3.Get<T>();
    // (mmResUb * scale) · offset = rowsum(mmResUb * scale * offset)
    CopyAntiquantParamsPerToken(valueAntiqOffsetGm, antiqParamOffsetPerToken, columnCount, actualColumnCount); // antiqParamOffsetPerToken need checking
    antiqScalePerTokenUb = inputQue1.DeQue<T>();
    VecMulMat(tmpAFloorUb, antiqScalePerTokenUb, tmpAFloorUb, dealRowCount, columnCount, actualColumnCount);
    inputQue1.FreeTensor(antiqScalePerTokenUb);
    pipe_barrier(PIPE_V);
    RowSum(tmpAMax, tmpAFloorUb, dealRowCount, columnCount, actualColumnCount);
    pipe_barrier(PIPE_V);
    Brcb(softmaxScaleResRowSumUb[baseOffset], tmpAMax, (dealRowCount + 7) / 8, {1, 8});
    pipe_barrier(PIPE_V);
    Adds(tmpAFloorUb, mmResUb, (T)0, dealRowCount * columnCount);  // mmResUb need to be stored
    pipe_barrier(PIPE_V);
  }
  AntiquantMatmulPreProcess(vec1ResGm[dstOffset], aMaxBmm2Ub, mmResUb, tmpAFloorUb, startRow, dealRowCount, columnCount,
                            actualColumnCount); // 常规的分块、拼接函数
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::DealAntiqBmm1ResBaseBlockPerToken(
    const uint32_t sInnerLoopIdx, uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
    uint32_t actualColumnCount)
{
    LocalTensor<T> mmResUb = tmpBuff1.Get<T>();
    LocalTensor<T> aMax = aMaxBmm1Ub[startRow * BLOCK_ELEMENT_NUM];
    uint32_t baseOffset = startRow * BLOCK_ELEMENT_NUM;
    AntiquantMatmulResCombine(mmResUb, mm1ResGm, startRow, dealRowCount, columnCount, actualColumnCount);
    uint32_t dtypeMask = REPEAT_ELEMENT_NUM;
    int32_t mulLoop = actualColumnCount / dtypeMask;
    int32_t mulRemain = actualColumnCount % dtypeMask;
    BinaryRepeatParams repeatParams;

    if (antiqOffsetExistFlag) {
        CopyAntiquantParamsPerToken(keyAntiqOffsetGm, antiqParamOffsetPerToken, columnCount, actualColumnCount);
        LocalTensor<T> antiqOffsetPerTokenUb = inputQue1.DeQue<T>();
        LocalTensor<T> tmpOffset = tmpBuff2.Get<T>();
        LocalTensor<T> aRowSum = qRowSumUb[baseOffset];

        // rowsum(A) * offset
        repeatParams.src0RepStride = 1;
        repeatParams.src0BlkStride = 0;
        repeatParams.src1RepStride = 0;
        repeatParams.src1BlkStride = 1;
        repeatParams.dstRepStride = columnCount / BLOCK_ELEMENT_NUM;
        repeatParams.dstBlkStride = 1;
        pipe_barrier(PIPE_V);
        for (int i = 0; i < mulLoop; i++) {
            Mul(tmpOffset[i * dtypeMask], aRowSum, antiqOffsetPerTokenUb[i * dtypeMask], dtypeMask, dealRowCount,
                repeatParams);
        }
        if (mulRemain > 0) {
            Mul(tmpOffset[mulLoop * dtypeMask], aRowSum, antiqOffsetPerTokenUb[mulLoop * dtypeMask], mulRemain,
                dealRowCount, repeatParams);
        }
        inputQue1.FreeTensor(antiqOffsetPerTokenUb);

        // Amax * C + rowsum(A) * offset
        repeatParams.src0RepStride = 1;
        repeatParams.src0BlkStride = 0;
        repeatParams.src1RepStride = columnCount / BLOCK_ELEMENT_NUM;
        repeatParams.src1BlkStride = 1;
        repeatParams.dstRepStride = columnCount / BLOCK_ELEMENT_NUM;
        repeatParams.dstBlkStride = 1;
        pipe_barrier(PIPE_V);
        for (int j = 0; j < mulLoop; j++) {
            FusedMulAdd(mmResUb[j * dtypeMask], aMax, tmpOffset[j * dtypeMask], dtypeMask, dealRowCount,
                        repeatParams);
        }
        if (mulRemain > 0) {
            FusedMulAdd(mmResUb[mulLoop * dtypeMask], aMax, tmpOffset[mulLoop * dtypeMask], mulRemain,
                        dealRowCount, repeatParams);
        }
        pipe_barrier(PIPE_V);
    } else {
        // Amax * C
        repeatParams.src0RepStride = 1;
        repeatParams.src0BlkStride = 0;
        repeatParams.src1RepStride = columnCount / BLOCK_ELEMENT_NUM;
        repeatParams.src1BlkStride = 1;
        repeatParams.dstRepStride = columnCount / BLOCK_ELEMENT_NUM;
        repeatParams.dstBlkStride = 1;
        pipe_barrier(PIPE_V);
        for (int i = 0; i < mulLoop; i++) {
            Mul(mmResUb[i * dtypeMask], aMax, mmResUb[i * dtypeMask], dtypeMask, dealRowCount, repeatParams);
        }
        if (mulRemain > 0) {
            Mul(mmResUb[mulLoop * dtypeMask], aMax, mmResUb[mulLoop * dtypeMask], mulRemain, dealRowCount,
                repeatParams);
        }
    }
    CopyAntiquantParamsPerToken(keyAntiqScaleGm, antiqParamOffsetPerToken, columnCount, actualColumnCount);
    LocalTensor<T> antiqScalePerTokenUb = inputQue1.DeQue<T>();
    // (Amax * C + rowsum(A) * offset) * scale
    VecMulMat(mmResUb, antiqScalePerTokenUb, mmResUb, dealRowCount, columnCount, actualColumnCount);
    pipe_barrier(PIPE_V);
    inputQue1.FreeTensor(antiqScalePerTokenUb);

    // mul scalar and mask
    ElewiseCompute(mmResUb, tmpBuff2, startRow, dealRowCount, columnCount, actualColumnCount);

    LocalTensor<T> tmpAFloorUb = tmpBuff2.Get<T>();
    LocalTensor<uint8_t> softmaxTmpUb = tmpAFloorUb.template ReinterpretCast<uint8_t>();
    SoftmaxFlashV2Compute(mmResUb, softmaxTmpUb, startRow, dealRowCount, columnCount, actualColumnCount);
    pipe_barrier(PIPE_V);
    DealKvInt4ColumnOdd(actualColumnCount);

    // mmResUb mul scale
    CopyAntiquantParamsPerToken(valueAntiqScaleGm, antiqParamOffsetPerToken, columnCount, actualColumnCount);
    antiqScalePerTokenUb = inputQue1.DeQue<T>();
    VecMulMat(mmResUb, antiqScalePerTokenUb, mmResUb, dealRowCount, columnCount, actualColumnCount);
    pipe_barrier(PIPE_V);
    inputQue1.FreeTensor(antiqScalePerTokenUb);

    Adds(tmpAFloorUb, mmResUb, (T)0, dealRowCount * columnCount); // mmResUb need to be stored
    pipe_barrier(PIPE_V);
    if (antiqOffsetExistFlag) {
        LocalTensor<T> tmpAMax = tmpBuff3.Get<T>();

        // (mmResUb * scale) · offset = rowsum(mmResUb * scale * offset)
        CopyAntiquantParamsPerToken(valueAntiqOffsetGm, antiqParamOffsetPerToken, columnCount, actualColumnCount);
        antiqScalePerTokenUb = inputQue1.DeQue<T>();
        VecMulMat(tmpAFloorUb, antiqScalePerTokenUb, tmpAFloorUb, dealRowCount, columnCount, actualColumnCount);
        inputQue1.FreeTensor(antiqScalePerTokenUb);
        pipe_barrier(PIPE_V);
        RowSum(tmpAMax, tmpAFloorUb, dealRowCount, columnCount, actualColumnCount);
        pipe_barrier(PIPE_V);
        Brcb(softmaxScaleResRowSumUb[baseOffset], tmpAMax, (dealRowCount + 7) / 8, {1, 8});
        pipe_barrier(PIPE_V);
        Adds(tmpAFloorUb, mmResUb, (T)0, dealRowCount * columnCount); // mmResUb need to be stored
        pipe_barrier(PIPE_V);
    }

    size_t dstOffset = 0;
    if constexpr (SHARED_PREFIX) {
        if (calcSysPrefixFlag) {
            dstOffset = bIdx * gSize * msdIterNum * columnCount;
        }
    }

    AntiquantMatmulPreProcess(vec1ResGm[dstOffset], aMaxBmm2Ub, mmResUb, tmpAFloorUb, startRow, dealRowCount,
                              columnCount, actualColumnCount);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::PreProcessVec1(uint32_t sInnerLoopIdx)
{
    if constexpr (ANTIQUANT) {
        SysPrefixLoadMsdMax1(bIdx);
        if constexpr (ANTIQUANT_PER_TOKEN) {
            if (antiqOffsetExistFlag) {
                SysPrefixLoadMsdSum1(bIdx);
            }
        } else if (antiqOffsetExistFlag && softmaxLseFlag) {
            SysPrefixLoadMsdSum1(bIdx);
        }
    }

    if (sInnerLoopIdx != 0) {
        SysPrefixLoadSoftmaxMax(bIdx);
        SysPrefixLoadSoftmaxSum(bIdx);
        SysPrefixLoadSoftmaxExp(bIdx);
    } else {
        Duplicate(softmaxMaxUb, SOFTMAX_MIN_NUM, gSize * BYTE_BLOCK);
        Duplicate(softmaxSumUb, FLOAT_ZERO, gSize * BYTE_BLOCK);
    }
}

template <typename IFAT> __aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::PostProcessVec1()
{
    if constexpr (ANTIQUANT && ANTIQUANT_PER_TOKEN) {
        SysPrefixSaveMsdMax2(bIdx);
        if (antiqOffsetExistFlag) {
            SysPrefixSaveMsdSum2(bIdx);
        }
    }
    SysPrefixSaveSoftmaxMax(bIdx);
    SysPrefixSaveSoftmaxSum(bIdx);
    SysPrefixSaveSoftmaxExp(bIdx);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::ProcessVec1Inner(const uint32_t sInnerLoopIdx)
{
    uint32_t gSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / actualSingleProcessSInnerSizeAlign;
    if (gSplitSize > gSize) {
        gSplitSize = gSize;
    }
    uint32_t loopCount = (gSize + gSplitSize - 1) / gSplitSize;
    uint32_t tailSplitSize = gSize - (loopCount - 1) * gSplitSize;

    for (uint32_t i = 0, dealSize = gSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        if constexpr (ANTIQUANT) {
            if constexpr (ANTIQUANT_PER_CHANNEL) {
                DealAntiqBmm1ResBaseBlock(sInnerLoopIdx, i * gSplitSize, dealSize, actualSingleProcessSInnerSizeAlign,
                                          actualSingleProcessSInnerSize);
            } else if constexpr (ANTIQUANT_PER_TOKEN) {
                DealAntiqBmm1ResBaseBlockPerToken(sInnerLoopIdx, i * gSplitSize, dealSize,
                                                  actualSingleProcessSInnerSizeAlign, actualSingleProcessSInnerSize);
            } else if (ANTIQUANT_PER_CHANNEL_TOKEN) { // channel plus token
                DealAntiqBmm1ResBaseBlockChannelToken(sInnerLoopIdx, i * gSplitSize, dealSize, actualSingleProcessSInnerSizeAlign,
                                                    actualSingleProcessSInnerSize);
            }
        } else {
            DealBmm1ResBaseBlock(sInnerLoopIdx, i * gSplitSize, dealSize, actualSingleProcessSInnerSizeAlign,
                                 actualSingleProcessSInnerSize);
        }
    }

    if (sInnerLoopIdx == sInnerLoopTimes - 1) {
        if constexpr (SHARED_PREFIX) {
            if (!flashDecodeFlag) {
                SysPrefixSaveLseFA();
            } else if constexpr (FLASH_DECODE) {
                ComputeLogSumExpAndCopyToGm(softmaxSumUb, softmaxMaxUb);
            }
            return;
        }

        if constexpr (FLASH_DECODE) {
            ComputeLogSumExpAndCopyToGm(softmaxSumUb, softmaxMaxUb);
            return;
        }

        if (softmaxLseFlag) {
            // 将lse拷贝至GM
            SoftmaxLseCopyOut(softmaxSumUb, softmaxMaxUb);
        }
    }
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::ProcessVec1(const uint32_t sInnerLoopIdx)
{
    if constexpr (SHARED_PREFIX) {
        if (calcSysPrefixFlag) {
            uint32_t bIdxOld = bIdx;
            for (bIdx = 0; bIdx < batchSizeQ; bIdx++) {
                UpdateOffsetsVec(sInnerLoopIdx);
                PreProcessVec1(sInnerLoopIdx);
                ProcessVec1Inner(sInnerLoopIdx);
                PostProcessVec1();
            }
            bIdx = bIdxOld;
            return;
        }
    }
    ProcessVec1Inner(sInnerLoopIdx);
}

template <typename IFAT>
__aicore__ inline void
SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::DealBmm2ResBaseBlock(const uint32_t sInnerLoopIdx, uint32_t startRow,
                                                                   uint32_t dealRowCount, uint32_t columnCount,
                                                                   uint32_t actualColumnCount)
{
    uint32_t vec2ComputeSize = dealRowCount * columnCount;
    uint32_t baseOffset = startRow * BLOCK_ELEMENT_NUM;
    LocalTensor<T> bmm2ResUb = tmpBuff1.Get<T>();
    bmm2ResUb.SetSize(vec2ComputeSize);

    uint64_t batchBase = 0;
    if constexpr (SHARED_PREFIX) {
        if (calcSysPrefixFlag) {
            batchBase = bIdx * gSize * columnCount;
        }
    }

    {
        LocalTensor<MM_OUT_T> tmpBmm2ResUb = inputQue1.AllocTensor<MM_OUT_T>();
        DataCopy(tmpBmm2ResUb, mm2ResGm[batchBase + startRow * columnCount], vec2ComputeSize);
        inputQue1.EnQue(tmpBmm2ResUb);
        inputQue1.DeQue<MM_OUT_T>();
        DataCopy(bmm2ResUb, tmpBmm2ResUb, vec2ComputeSize);
        inputQue1.FreeTensor(tmpBmm2ResUb);
    }

    // 除第一个循环外，均需要更新中间计算结果
    if (sInnerLoopIdx > 0) {
        uint32_t singleDealRowCount = BASE_BLOCK_MAX_ELEMENT_NUM / 2 / columnCount; // 16K(inputQue2 Size)/D
        uint32_t loopCnt = (dealRowCount + singleDealRowCount - 1) / singleDealRowCount;
        uint32_t tailDealRowCount = dealRowCount - (loopCnt - 1) * singleDealRowCount;
        for (int i = 0, curDealRowCount = singleDealRowCount; i < loopCnt; i++) {
            if (i + 1 == loopCnt) {
                curDealRowCount = tailDealRowCount;
            }
            LocalTensor<T> bmm2ResPreUb = inputQue2.AllocTensor<T>();
            DataCopy(bmm2ResPreUb, vec2ResGm[batchBase + startRow * columnCount + i * singleDealRowCount * columnCount],
                     curDealRowCount * columnCount);
            inputQue2.EnQue(bmm2ResPreUb);

            inputQue2.DeQue<T>();
            pipe_barrier(PIPE_V);
            RowMuls(bmm2ResPreUb, bmm2ResPreUb, softmaxExpUb[baseOffset + i * singleDealRowCount * BLOCK_ELEMENT_NUM],
                    curDealRowCount, columnCount, actualColumnCount);
            pipe_barrier(PIPE_V);
            Add(bmm2ResUb[i * singleDealRowCount * columnCount], bmm2ResUb[i * singleDealRowCount * columnCount],
                bmm2ResPreUb, curDealRowCount * columnCount);
            inputQue2.FreeTensor(bmm2ResPreUb);
        }
    }

    // 最后一次输出计算结果，否则将中间结果暂存至workspace
    if (sInnerLoopIdx + 1 == sInnerLoopTimes) {
        pipe_barrier(PIPE_V);
        RowDivs(bmm2ResUb, bmm2ResUb, softmaxSumUb[baseOffset], dealRowCount, columnCount, actualColumnCount);

        pipe_barrier(PIPE_V);
        Bmm2CastAndCopyOut(bmm2ResUb, startRow, dealRowCount, columnCount, actualColumnCount);
    } else {
        pipe_barrier(PIPE_V);
        LocalTensor<T> tmpBmm2Res = outputQue1.AllocTensor<T>();
        DataCopy(tmpBmm2Res, bmm2ResUb, dealRowCount * columnCount);
        outputQue1.EnQue(tmpBmm2Res);
        outputQue1.DeQue<T>();

        DataCopy(vec2ResGm[batchBase + startRow * columnCount], tmpBmm2Res, vec2ComputeSize);

        outputQue1.FreeTensor(tmpBmm2Res);
    }
}

template <typename IFAT>
__aicore__ inline void
SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::PostQuant(LocalTensor<T> &bmm2ResUb, LocalTensor<int8_t> &bmm2ResUbInt8,
                                                        uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                        uint32_t actualColumnCount)
{
    uint32_t copySize = dealRowCount * columnCount;
    if (!isPerChnU8Out) {
        LocalTensor<uint8_t> sharedTempBuffer = tmpBuff2.Get<uint8_t>(); // AscendQuant接口要求sharedTempBuffer不能太小
        AscendQuant(bmm2ResUbInt8, bmm2ResUb, sharedTempBuffer, quantScale2Value, quantOffset2Value, copySize);
    } else {
        if (!isOutQuantTypeBf16) { // fp32
            DataCopyExtParams copyInParams;
            DataCopyPadExtParams<float> copyInPadParams;
            copyInParams.blockCount = dealRowCount;
            copyInParams.blockLen = actualColumnCount * sizeof(float);
            copyInParams.srcStride = 0;
            copyInParams.dstStride = (columnCount - actualColumnCount) / BLOCK_ELEMENT_NUM;

            copyInPadParams.isPad = true;
            copyInPadParams.leftPadding = 0;
            copyInPadParams.rightPadding = (columnCount - actualColumnCount) % BLOCK_ELEMENT_NUM;
            copyInPadParams.paddingValue = 0;
            {
                LocalTensor<float> quantScale2Ub = inputQue1.AllocTensor<float>();
                DataCopyPad(quantScale2Ub, quantScale2Gm[perChannelQuantOffset + startRow * actualColumnCount],
                            copyInParams, copyInPadParams);
                inputQue1.EnQue(quantScale2Ub);
                inputQue1.DeQue<float>();

                Mul(bmm2ResUb, quantScale2Ub, bmm2ResUb, copySize);
                inputQue1.FreeTensor(quantScale2Ub);
                pipe_barrier(PIPE_V);
            }
            if (isQuantOffset2Exist) {
                LocalTensor<float> quantOffset2Ub = inputQue1.AllocTensor<float>();
                DataCopyPad(quantOffset2Ub, quantOffset2Gm[perChannelQuantOffset + startRow * actualColumnCount],
                            copyInParams, copyInPadParams);
                inputQue1.EnQue(quantOffset2Ub);
                inputQue1.DeQue<float>();

                Add(bmm2ResUb, quantOffset2Ub, bmm2ResUb, copySize);
                inputQue1.FreeTensor(quantOffset2Ub);
                pipe_barrier(PIPE_V);
            }
        } else {
            uint32_t typeElementSize = BYTE_BLOCK / sizeof(bfloat16_t);
            DataCopyExtParams copyInParams;
            DataCopyPadExtParams<bfloat16_t> copyInPadParams;
            copyInParams.blockCount = dealRowCount;
            copyInParams.blockLen = actualColumnCount * sizeof(bfloat16_t);
            copyInParams.srcStride = 0;
            copyInParams.dstStride = (columnCount - actualColumnCount) / typeElementSize;

            copyInPadParams.isPad = true;
            copyInPadParams.leftPadding = 0;
            copyInPadParams.rightPadding = (columnCount - actualColumnCount) % typeElementSize;
            copyInPadParams.paddingValue = 0;
            LocalTensor<float> tempCastUb = tmpBuff2.Get<float>(copySize);
            {
                LocalTensor<bfloat16_t> quantScale2Ub = inputQue1.AllocTensor<bfloat16_t>();
                DataCopyPad(quantScale2Ub, quantScale2Bf16Gm[perChannelQuantOffset + startRow * actualColumnCount],
                            copyInParams, copyInPadParams);
                inputQue1.EnQue(quantScale2Ub);
                inputQue1.DeQue<bfloat16_t>();

                Cast(tempCastUb, quantScale2Ub, RoundMode::CAST_NONE, copySize);
                inputQue1.FreeTensor(quantScale2Ub);
                pipe_barrier(PIPE_V);
            }

            Mul(bmm2ResUb, tempCastUb, bmm2ResUb, copySize);
            pipe_barrier(PIPE_V);
            if (isQuantOffset2Exist) {
                LocalTensor<bfloat16_t> quantOffset2Ub = inputQue2.AllocTensor<bfloat16_t>();
                DataCopyPad(quantOffset2Ub, quantOffset2Bf16Gm[perChannelQuantOffset + startRow * actualColumnCount],
                            copyInParams, copyInPadParams);
                inputQue2.EnQue(quantOffset2Ub);
                inputQue2.DeQue<bfloat16_t>();

                Cast(tempCastUb, quantOffset2Ub, RoundMode::CAST_NONE, copySize);
                inputQue2.FreeTensor(quantOffset2Ub);
                pipe_barrier(PIPE_V);

                Add(bmm2ResUb, tempCastUb, bmm2ResUb, copySize);
                pipe_barrier(PIPE_V);
            }
        }
        LocalTensor<half> quantResultHalf = tmpBuff1.Get<half>(copySize);
        Cast(quantResultHalf, bmm2ResUb, RoundMode::CAST_ROUND, copySize);
        pipe_barrier(PIPE_V);

        Cast(bmm2ResUbInt8, quantResultHalf, RoundMode::CAST_ROUND, copySize);
        pipe_barrier(PIPE_V);
    }
}

template <typename IFAT>
__aicore__ inline void
SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::DealAntiqBmm2ResBaseBlock(const uint32_t sInnerLoopIdx, uint32_t startRow,
                                                                        uint32_t dealRowCount, uint32_t columnCount,
                                                                        uint32_t actualColumnCount)
{
    uint32_t vec2ComputeSize = dealRowCount * columnCount;
    LocalTensor<T> bmm2ResUb = tmpBuff1.Get<T>();
    AntiquantMatmulResCombine(bmm2ResUb, mm2ResGm, startRow, dealRowCount, columnCount, actualColumnCount);

    uint32_t baseOffset = startRow * BLOCK_ELEMENT_NUM;

    uint64_t batchBase = 0;
    if constexpr (SHARED_PREFIX) {
        if (calcSysPrefixFlag) {
            batchBase = bIdx * gSize * columnCount;
        }
    }

    // 除第一个循环外，均需要更新中间计算结果
    if (sInnerLoopIdx > 0) {
        uint32_t singleDealRowCount = BASE_BLOCK_MAX_ELEMENT_NUM / 2 / columnCount; // 16K(inputQue2 Size)/D
        uint32_t loopCnt = (dealRowCount + singleDealRowCount - 1) / singleDealRowCount;
        uint32_t tailDealRowCount = dealRowCount - (loopCnt - 1) * singleDealRowCount;
        for (int i = 0, curDealRowCount = singleDealRowCount; i < loopCnt; i++) {
            if (i + 1 == loopCnt) {
                curDealRowCount = tailDealRowCount;
            }
            LocalTensor<T> bmm2ResPreUb = inputQue2.AllocTensor<T>();
            DataCopy(bmm2ResPreUb, vec2ResGm[batchBase + startRow * columnCount + i * singleDealRowCount * columnCount],
                     curDealRowCount * columnCount);
            inputQue2.EnQue(bmm2ResPreUb);

            inputQue2.DeQue<T>();
            pipe_barrier(PIPE_V);
            RowMuls(bmm2ResPreUb, bmm2ResPreUb, softmaxExpUb[baseOffset + i * singleDealRowCount * BLOCK_ELEMENT_NUM],
                    curDealRowCount, columnCount, actualColumnCount);
            pipe_barrier(PIPE_V);
            Add(bmm2ResUb[i * singleDealRowCount * columnCount], bmm2ResUb[i * singleDealRowCount * columnCount],
                bmm2ResPreUb, curDealRowCount * columnCount);
            inputQue2.FreeTensor(bmm2ResPreUb);
        }
    }

    // 最后一次输出计算结果，否则将中间结果暂存至workspace
    if (sInnerLoopIdx + 1 == sInnerLoopTimes) {
        pipe_barrier(PIPE_V);
        RowDivs(bmm2ResUb, bmm2ResUb, softmaxSumUb[baseOffset], dealRowCount, columnCount, actualColumnCount);
        pipe_barrier(PIPE_V);

        if (antiqOffsetExistFlag) {
            // bmm2Res + offsetV
            CopyAntiquantScale(antiqOffsetUb, valueAntiqOffsetGm, antiqParamOffset);
            pipe_barrier(PIPE_V);
            VecAddMat(bmm2ResUb, antiqOffsetUb, bmm2ResUb, dealRowCount, columnCount, actualColumnCount);
            pipe_barrier(PIPE_V);
        }

        CopyAntiquantScale(antiqScaleUb, valueAntiqScaleGm, antiqParamOffset);
        pipe_barrier(PIPE_V);
        // ScaleV * bmm2Res
        VecMulMat(bmm2ResUb, antiqScaleUb, bmm2ResUb, dealRowCount, columnCount, actualColumnCount);
        pipe_barrier(PIPE_V);

        Bmm2CastAndCopyOut(bmm2ResUb, startRow, dealRowCount, columnCount, actualColumnCount);
    } else {
        pipe_barrier(PIPE_V);
        LocalTensor<T> tmpBmm2Res = outputQue1.AllocTensor<T>();
        DataCopy(tmpBmm2Res, bmm2ResUb, dealRowCount * columnCount);
        outputQue1.EnQue(tmpBmm2Res);
        outputQue1.DeQue<T>();
        DataCopy(vec2ResGm[startRow * columnCount + batchBase], tmpBmm2Res, vec2ComputeSize);
        outputQue1.FreeTensor(tmpBmm2Res);
    }
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::DealAntiqBmm2ResBaseBlockPerToken(
    const uint32_t sInnerLoopIdx, uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
    uint32_t actualColumnCount)
{
    uint32_t vec2ComputeSize = dealRowCount * columnCount;
    LocalTensor<T> bmm2ResUb = tmpBuff1.Get<T>();
    AntiquantMatmulResCombine(bmm2ResUb, mm2ResGm, startRow, dealRowCount, columnCount, actualColumnCount);

    uint32_t baseOffset = startRow * BLOCK_ELEMENT_NUM;
    LocalTensor<T> aRowMax = aMaxBmm2Ub[baseOffset];
    uint32_t dtypeMask = REPEAT_ELEMENT_NUM;
    int32_t mulLoop = actualColumnCount / dtypeMask;
    int32_t mulRemain = actualColumnCount % dtypeMask;
    BinaryRepeatParams repeatParams;
    if (antiqOffsetExistFlag) {
        LocalTensor<T> aRowSum = softmaxScaleResRowSumUb[baseOffset];

        repeatParams.src0RepStride = 1;
        repeatParams.src0BlkStride = 0;
        repeatParams.src1RepStride = 1;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstRepStride = columnCount / BLOCK_ELEMENT_NUM;
        repeatParams.dstBlkStride = 1;
        pipe_barrier(PIPE_V);
        for (int j = 0; j < mulLoop; j++) {
            FusedMulAdd(bmm2ResUb[j * dtypeMask], aRowMax, aRowSum, dtypeMask, dealRowCount, repeatParams);
        }
        if (mulRemain > 0) {
            FusedMulAdd(bmm2ResUb[mulLoop * dtypeMask], aRowMax, aRowSum, mulRemain, dealRowCount, repeatParams);
        }
    } else {
        repeatParams.src0RepStride = 1;
        repeatParams.src0BlkStride = 0;
        repeatParams.src1RepStride = columnCount / BLOCK_ELEMENT_NUM;
        repeatParams.src1BlkStride = 1;
        repeatParams.dstRepStride = columnCount / BLOCK_ELEMENT_NUM;
        repeatParams.dstBlkStride = 1;
        pipe_barrier(PIPE_V);
        for (int i = 0; i < mulLoop; i++) {
            Mul(bmm2ResUb[i * dtypeMask], aRowMax, bmm2ResUb[i * dtypeMask], dtypeMask, dealRowCount, repeatParams);
        }
        if (mulRemain > 0) {
            Mul(bmm2ResUb[mulLoop * dtypeMask], aRowMax, bmm2ResUb[mulLoop * dtypeMask], mulRemain, dealRowCount,
                repeatParams);
        }
    }

    uint64_t batchBase = 0;
    if constexpr (SHARED_PREFIX) {
        if (calcSysPrefixFlag) {
            batchBase = bIdx * gSize * columnCount;
        }
    }

    // 除第一个循环外，均需要更新中间计算结果
    if (sInnerLoopIdx > 0) {
        uint32_t singleDealRowCount = BASE_BLOCK_MAX_ELEMENT_NUM / 2 / columnCount; // 16K(inputQue2 Size)/D
        uint32_t loopCnt = (dealRowCount + singleDealRowCount - 1) / singleDealRowCount;
        uint32_t tailDealRowCount = dealRowCount - (loopCnt - 1) * singleDealRowCount;
        for (int i = 0, curDealRowCount = singleDealRowCount; i < loopCnt; i++) {
            if (i + 1 == loopCnt) {
                curDealRowCount = tailDealRowCount;
            }
            LocalTensor<T> bmm2ResPreUb = inputQue2.AllocTensor<T>();
            DataCopy(bmm2ResPreUb, vec2ResGm[batchBase + startRow * columnCount + i * singleDealRowCount * columnCount],
                     curDealRowCount * columnCount);
            inputQue2.EnQue(bmm2ResPreUb);

            inputQue2.DeQue<T>();
            pipe_barrier(PIPE_V);
            RowMuls(bmm2ResPreUb, bmm2ResPreUb, softmaxExpUb[baseOffset + i * singleDealRowCount * BLOCK_ELEMENT_NUM],
                    curDealRowCount, columnCount, actualColumnCount);
            pipe_barrier(PIPE_V);
            Add(bmm2ResUb[i * singleDealRowCount * columnCount], bmm2ResUb[i * singleDealRowCount * columnCount],
                bmm2ResPreUb, curDealRowCount * columnCount);
            inputQue2.FreeTensor(bmm2ResPreUb);
        }
    }

    // 最后一次输出计算结果，否则将中间结果暂存至workspace
    if (sInnerLoopIdx + 1 == sInnerLoopTimes) {
        pipe_barrier(PIPE_V);
        RowDivs(bmm2ResUb, bmm2ResUb, softmaxSumUb[baseOffset], dealRowCount, columnCount, actualColumnCount);

        pipe_barrier(PIPE_V);
        Bmm2CastAndCopyOut(bmm2ResUb, startRow, dealRowCount, columnCount, actualColumnCount);
    } else {
        pipe_barrier(PIPE_V);
        LocalTensor<T> tmpBmm2Res = outputQue1.AllocTensor<T>();
        DataCopy(tmpBmm2Res, bmm2ResUb, dealRowCount * columnCount);
        outputQue1.EnQue(tmpBmm2Res);
        outputQue1.DeQue<T>();
        DataCopy(vec2ResGm[startRow * columnCount + batchBase], tmpBmm2Res, vec2ComputeSize);
        outputQue1.FreeTensor(tmpBmm2Res);
    }
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::ProcessVec2Inner(const uint32_t sInnerLoopIdx)
{
    uint32_t gSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / headDimAlign;
    if (gSplitSize > gSize) {
        gSplitSize = gSize;
    }
    uint32_t loopCount = (gSize + gSplitSize - 1) / gSplitSize;
    uint32_t tailSplitSize = gSize - (loopCount - 1) * gSplitSize;

    for (uint32_t i = 0, dealSize = gSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        if constexpr (ANTIQUANT) {
            if constexpr (ANTIQUANT_PER_CHANNEL) {
                DealAntiqBmm2ResBaseBlock(sInnerLoopIdx, i * gSplitSize, dealSize, headDimAlign, headDim);
            } else if constexpr (ANTIQUANT_PER_TOKEN) {
                DealAntiqBmm2ResBaseBlockPerToken(sInnerLoopIdx, i * gSplitSize, dealSize, headDimAlign, headDim);
            } else if constexpr (ANTIQUANT_PER_CHANNEL_TOKEN) {
                DealAntiqBmm2ResBaseBlockPerToken(sInnerLoopIdx, i * gSplitSize, dealSize, headDimAlign, headDim);
            }
        } else {
            DealBmm2ResBaseBlock(sInnerLoopIdx, i * gSplitSize, dealSize, headDimAlign, headDim);
        }
    }
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::PreProcessVec2(uint32_t sInnerLoopIdx)
{
    if constexpr (ANTIQUANT && ANTIQUANT_PER_TOKEN) {
        SysPrefixLoadMsdMax2(bIdx);
        if (antiqOffsetExistFlag) {
            SysPrefixLoadMsdSum2(bIdx);
        }
    }
    SysPrefixLoadSoftmaxExp(bIdx);
    SysPrefixLoadSoftmaxSum(bIdx);
    SysPrefixLoadSoftmaxMax(bIdx);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::ProcessVec2(const uint32_t sInnerLoopIdx)
{
    if constexpr (SHARED_PREFIX) {
        if (calcSysPrefixFlag) {
            uint32_t bIdxOld = bIdx;
            for (bIdx = 0; bIdx < batchSizeQ; bIdx++) {
                PreProcessVec2(sInnerLoopIdx);
                UpdateOffsetsVec(sInnerLoopIdx);
                ProcessVec2Inner(sInnerLoopIdx);
            }
            bIdx = bIdxOld;
            return;
        }
    }
    ProcessVec2Inner(sInnerLoopIdx);
}

template <typename IFAT> __aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SetMMOrgShape()
{
    if (SHARED_PREFIX) {
        if (calcSysPrefixFlag) {
            SysPrefixSetMMOrgShape();
            return;
        }
    }
    SetMMOrgShapeCommon();
}
template <typename IFAT> __aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SetMMOrgShapeCommon()
{
    /**
     * 为了减少rpc通信开销，尽量减少SetOrgShape的调用次数。
     * 由于，setOrgShape接口中只有actualSingleProcessSInnerSizeAlign是可变的，
     * 因此，bn loop共用同一个SetOrgShape，只有当actualSingleProcessSInnerSizeAlign发生变化时，再重新设置。
     */
    if (curSingleProcessSInnerSizeAlign != actualSingleProcessSInnerSizeAlign) {
        // mm1 setOrgShape
        uint32_t orgKa;
        if constexpr (ANTIQUANT) {
            orgKa = headDimAlign;
        } else {
            orgKa = headDim;
        }
        if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND || PAGE_ATTENTION) {
            mm.SetOrgShape(msdIterNum * gSize, tilingData->baseParams.seqSize, orgKa, kvHeadNum * headDim,
                           actualSingleProcessSInnerSizeAlign);
            bmm2.SetOrgShape(msdIterNum * gSize, kvHeadNum * headDim, actualSingleProcessSInnerSizeAlign,
                             tilingData->baseParams.seqSize, headDimAlign);
        } else {
            mm.SetOrgShape(msdIterNum * gSize, tilingData->baseParams.seqSize, orgKa, headDim,
                           actualSingleProcessSInnerSizeAlign);
            bmm2.SetOrgShape(msdIterNum * gSize, headDim, actualSingleProcessSInnerSizeAlign,
                             tilingData->baseParams.seqSize, headDimAlign);
        }
        // 更新curSingleProcessSInnerSizeAlign，为了下一次判断是否进行setOrgShape使用
        curSingleProcessSInnerSizeAlign = actualSingleProcessSInnerSizeAlign;
    }
}

template <typename IFAT> __aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SysPrefixSetMMOrgShape()
{
    /**
     * 为了减少rpc通信开销，尽量减少SetOrgShape的调用次数。
     * 由于，setOrgShape接口中只有actualSingleProcessSInnerSizeAlign是可变的，
     * 因此，bn loop共用同一个SetOrgShape，只有当actualSingleProcessSInnerSizeAlign发生变化时，再重新设置。
     */
    if (curSingleProcessSInnerSizeAlign != actualSingleProcessSInnerSizeAlign) {
        // mm1 setOrgShape
        uint32_t orgKa = headDimAlign;
        uint32_t M = msdIterNum * gSize * batchSizeQ;
        if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
            mm1Sp.SetOrgShape(M, tilingData->baseParams.seqSize, orgKa, kvHeadNum * headDim,
                              actualSingleProcessSInnerSizeAlign);
            mm2Sp.SetOrgShape(M, kvHeadNum * headDim, actualSingleProcessSInnerSizeAlign,
                              tilingData->baseParams.seqSize, headDimAlign);
        } else {
            mm1Sp.SetOrgShape(M, tilingData->baseParams.seqSize, orgKa, headDim, actualSingleProcessSInnerSizeAlign);
            mm2Sp.SetOrgShape(M, headDim, actualSingleProcessSInnerSizeAlign, tilingData->baseParams.seqSize,
                              headDimAlign);
        }
        // 更新curSingleProcessSInnerSizeAlign，为了下一次判断是否进行setOrgShape使用
        curSingleProcessSInnerSizeAlign = actualSingleProcessSInnerSizeAlign;
    }
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SInnerLoopFunc(const uint32_t bn2Idx,
                                                                                    const uint32_t sInnerLoopIdx)
{
    V5_DEBUG_PRINTF("[LOG] SInnerLoopFunc bn2Idx: %d sInnerLoopIdx: %d\n", bn2Idx, sInnerLoopIdx);
    // setOrgShape
    SetMMOrgShape();

    V5_DEBUG_PRINTF("[LOG] SetMMOrgShape bn2Idx: %d sInnerLoopIdx: %d\n", bn2Idx, sInnerLoopIdx);
    // mm1
    Bmm1Compute(bn2Idx, sInnerLoopIdx);

    V5_DEBUG_PRINTF("[LOG] Bmm1Compute bn2Idx: %d sInnerLoopIdx: %d\n", bn2Idx, sInnerLoopIdx);
    // v1
    ProcessVec1(sInnerLoopIdx);

    V5_DEBUG_PRINTF("[LOG] ProcessVec1 bn2Idx: %d sInnerLoopIdx: %d\n", bn2Idx, sInnerLoopIdx);
    // mm2
    Bmm2Compute(bn2Idx, sInnerLoopIdx);

    V5_DEBUG_PRINTF("[LOG] Bmm2Compute bn2Idx: %d sInnerLoopIdx: %d\n", bn2Idx, sInnerLoopIdx);
    // v2
    ProcessVec2(sInnerLoopIdx);
}

template <typename IFAT> __aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::Process()
{
    if (g_coreType == AIV && tmpBlockIdx >= usedCoreNum) {
        // skip cores
    } else {
        for (uint32_t bn2Idx = 0; bn2Idx < bn2LoopTimes; bn2Idx++) {
            // V5_DEBUG_PRINTF("[LOG] Process bn2Idx: %d\n", bn2Idx);
            GetBN2id(bn2Idx);
            // V5_DEBUG_PRINTF("[LOG] GetBN2id bn2Idx: %d\n", bn2Idx);
            GetActualSeqLen();
            // V5_DEBUG_PRINTF("[LOG] GetActualSeqLen bn2Idx: %d\n", bn2Idx);
            CalculateSUnitSize();
            // V5_DEBUG_PRINTF("[LOG] CalculateSUnitSize bn2Idx: %d\n", bn2Idx);
            // ComputeKVPaddingBeginOffset return false means this loop skip calculation
            if (!ComputeKVPaddingBeginOffset()) {
                continue;
            }
            // V5_DEBUG_PRINTF("[LOG] ComputeKVPaddingBeginOffset bn2Idx: %d\n", bn2Idx);

            // 计算BN2方向的offset
            CalcBN2Offset();
            // V5_DEBUG_PRINTF("[LOG] CalcBN2Offset bn2Idx: %d\n", bn2Idx);
            CalcBN2Params();
            // V5_DEBUG_PRINTF("[LOG] CalcBN2Params bn2Idx: %d\n", bn2Idx);
            // 根据当前块实际长度, 重配flashattention循环条件
            UpdateInnerLoopCond();
            // V5_DEBUG_PRINTF("[LOG] UpdateInnerLoopCond bn2Idx: %d\n", bn2Idx);
            pipe_barrier(PIPE_V);
            if (curActSeqLenIsZero) {
                continue;
            }
            // V5_DEBUG_PRINTF("[LOG] ComputeKVPaddingBeginOffset bn2Idx: %d\n", bn2Idx);
            // softmax不区分首次
            Duplicate(softmaxMaxUb, SOFTMAX_MIN_NUM, BUFFER_SIZE_BYTE_2K / sizeof(T));
            Duplicate(softmaxSumUb, FLOAT_ZERO, BUFFER_SIZE_BYTE_2K / sizeof(T));
            // 如果S2开多核，可能出现多核重复预处理Q的情况，可以将Q的预处理做成一个前置小kernel，拼接FA，可能影响不大
            if constexpr (ANTIQUANT) {
                // V5_DEBUG_PRINTF("[LOG] Antiq bn2Idx: %d\n", bn2Idx);
                if constexpr (ANTIQUANT_PER_CHANNEL) {
                    QueryPreProcess();
                } else if constexpr (ANTIQUANT_PER_TOKEN) {
                    QueryPreProcessPerToken();
                } else if (ANTIQUANT_PER_CHANNEL_TOKEN){
                    QueryPreProcess(); // K per-channel 计算完成分块、拼接后的结果
                }
            } else if constexpr (SHARED_PREFIX) {
                SysPrefixQueryPreProcess();
            }
            // GQA场景需要处理G，1、mm1 A矩阵 singleM=G 2、mm1结果vector1内部切分mm1的M轴
            // 3、涉及souter的地方，需要注意GQA
            for (uint32_t sInnerLoopIdx = 0; sInnerLoopIdx < sInnerLoopTimes; sInnerLoopIdx++) {
                // 计算s2方向的offset
                CalcSInnerOffsetAndParams(sInnerLoopIdx);
                SInnerLoopFunc(bn2Idx, sInnerLoopIdx);
            }
        }
    }
    if constexpr (FLASH_DECODE) {
        if (flashDecodeFlag) {
            // 多核同步
            SyncAll();
            FlashDecodeCompute();
        }
    }
}

template <typename IFAT> __aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::ProcessSysPrefixCombine()
{
    // 多核同步
    SyncAll();

    if (tmpBlockIdx >= usedCoreNumSp) {
        return;
    }

    bn2LoopTimes = blockSplitBn2RangeSp;
    beforeBlockSplitBn2Nums = tmpBlockIdx * blockSplitBn2RangeSp;
    // tail cores
    if (tmpBlockIdx >= formerCoreNumSp) {
        bn2LoopTimes = tailBlockSplitBn2RangeSp;
        beforeBlockSplitBn2Nums =
            formerCoreNumSp * blockSplitBn2RangeSp + (tmpBlockIdx - formerCoreNumSp) * tailBlockSplitBn2RangeSp;
    }

    for (uint32_t bn2Idx = 0; bn2Idx < bn2LoopTimes; bn2Idx++) {
        bIdx = (beforeBlockSplitBn2Nums + bn2Idx) / kvHeadNum;
        n2Idx = (beforeBlockSplitBn2Nums + bn2Idx) % kvHeadNum;
        SysPrefixAttenResCombine();
    }
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::CopyDataInByQueue1(LocalTensor<T> &dst,
                                                                                        const GlobalTensor<T> &src,
                                                                                        size_t size)
{
    dst = inputQue1.AllocTensor<T>();
    DataCopy(dst, src, size);
    inputQue1.EnQue(dst);
    inputQue1.DeQue<T>();
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::CopyDataInByQueue2(LocalTensor<T> &dst,
                                                                                        const GlobalTensor<T> &src,
                                                                                        size_t size)
{
    dst = inputQue2.AllocTensor<T>();
    DataCopy(dst, src, size);
    inputQue2.EnQue(dst);
    inputQue2.DeQue<T>();
}

template <typename IFAT> __aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SysPrefixAttenResCombine()
{
    size_t lseSize = 2 * gSize * FP32_ONE_BLOCK_SIZE;
    size_t bn2 = bIdx * kvHeadNum + n2Idx;

    // InQueue2 is used by bf16 PostQuant, antiqScale ub is not used now, reuse it
    LocalTensor<T> lseSum = antiqScaleBuff.Get<T>(BUFFER_SIZE_BYTE_4K);
    CopyGmToFixedUb(lseSum, lseSumGm[bn2 * lseSize], lseSize);
    LocalTensor<T> lseMax;
    CopyDataInByQueue1(lseMax, lseMaxGm[bn2 * lseSize], lseSize);

    SysPrefixLseToScales(lseSum, lseMax);
    inputQue1.FreeTensor(lseMax);

    uint64_t attenOffset = bn2 * gSize * headDimAlign;
    GlobalTensor<T> atten1 = sysPrefixAttenOutGm[attenOffset];
    GlobalTensor<T> atten2 = usrPromptAttenOutGm[attenOffset];
    LocalTensor<T> attenRes = tmpBuff1.Get<T>(BUFFER_SIZE_BYTE_32K);
    GlobalTensor<OUT_T> attenOutGm = attentionOutGm[bn2 * gSize * headDim];

    uint32_t gSplitSize = BUFFER_SIZE_BYTE_32K / (headDimAlign * sizeof(T));
    uint32_t loops = (gSize + gSplitSize - 1) / gSplitSize;
    uint32_t gTailSize = gSize - (loops - 1) * gSplitSize;

    for (uint32_t i = 0; i < loops; i++) {
        uint32_t rows = (i == loops - 1) ? gTailSize : gSplitSize;
        auto lseScale = lseSum[i * gSplitSize * FP32_ONE_BLOCK_SIZE];
        SysPrefixAttenReduce(attenRes, atten1, atten2, lseScale, i * gSplitSize, rows);
        SysPrefixAttenOutput(attenOutGm, attenRes, i * gSplitSize, rows);
    }
}

template <typename IFAT>
__aicore__ inline void
SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SysPrefixAttenReduce(LocalTensor<T> &dst, GlobalTensor<T> &atten1Gm,
                                                                   GlobalTensor<T> &atten2Gm, LocalTensor<T> scales,
                                                                   uint32_t startRow, uint32_t rows)
{
    uint64_t attenOffset = startRow * headDimAlign;
    size_t attenSize = rows * headDimAlign;
    LocalTensor<T> atten1;
    CopyDataInByQueue1(atten1, atten1Gm[attenOffset], attenSize);

    BinaryRepeatParams repeatParams;
    repeatParams.src0RepStride = 1;
    repeatParams.src0BlkStride = 0;
    repeatParams.src1RepStride = (headDimAlign * sizeof(T)) / BYTE_BLOCK;
    repeatParams.dstRepStride = (headDimAlign * sizeof(T)) / BYTE_BLOCK;
    uint64_t mask = 256 / sizeof(T);
    uint32_t loops = (headDimAlign + mask - 1) / mask;
    uint32_t tail = headDimAlign - (loops - 1) * mask;

    // 第一次，mul结果直接放到dst里
    for (uint32_t i = 0; i < loops; i++) {
        Mul(dst[i * mask], scales, atten1[i * mask], (i != loops - 1) ? mask : tail, rows, repeatParams);
    }
    pipe_barrier(PIPE_V);
    inputQue1.FreeTensor(atten1);

    LocalTensor<T> atten2;
    CopyDataInByQueue1(atten2, atten2Gm[attenOffset], attenSize);
    LocalTensor<T> scales2 = scales[gSize * FP32_ONE_BLOCK_SIZE];
    for (uint32_t i = 0; i < loops; i++) {
        Mul(atten2[i * mask], scales2, atten2[i * mask], (i != loops - 1) ? mask : tail, rows, repeatParams);
    }

    pipe_barrier(PIPE_V);
    Add(dst, dst, atten2, rows * headDimAlign);
    pipe_barrier(PIPE_V);
    inputQue1.FreeTensor(atten2);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SysPrefixLseToScales(LocalTensor<T> &lseSum,
                                                                                          LocalTensor<T> &lseMax)
{
    size_t lseBlockSize = gSize * FP32_ONE_BLOCK_SIZE;
    LocalTensor<T> tmpMax = tmpBuff1.Get<T>(2 * lseBlockSize + 2 * lseBlockSize);
    LocalTensor<T> tmpSum = tmpMax[lseBlockSize];
    LocalTensor<T> tmpExp = tmpSum[lseBlockSize];

    LocalTensor<T> lseMax1 = lseMax[0];
    LocalTensor<T> lseMax2 = lseMax[lseBlockSize];

    Max(tmpMax, lseMax1, lseMax2, lseBlockSize);
    pipe_barrier(PIPE_V);

    Sub(tmpExp, lseMax1, tmpMax, lseBlockSize);
    Sub(tmpExp[lseBlockSize], lseMax2, tmpMax, lseBlockSize);
    pipe_barrier(PIPE_V);

    Exp(tmpExp, tmpExp, 2 * lseBlockSize);
    pipe_barrier(PIPE_V);

    Mul(lseSum, lseSum, tmpExp, 2 * lseBlockSize);
    pipe_barrier(PIPE_V);

    Add(tmpSum, lseSum[0], lseSum[lseBlockSize], lseBlockSize);
    pipe_barrier(PIPE_V);

    Div(lseSum[0], lseSum[0], tmpSum, lseBlockSize);
    Div(lseSum[lseBlockSize], lseSum[lseBlockSize], tmpSum, lseBlockSize);
    pipe_barrier(PIPE_V);

    if (softmaxLseFlag) {
        Log(tmpSum, tmpSum, lseBlockSize);
        pipe_barrier(PIPE_V);
        Add(tmpSum, tmpSum, tmpMax, lseBlockSize);
        pipe_barrier(PIPE_V);

        SoftmaxLseOutput(tmpSum);
    }
}

template <typename IFAT>
__aicore__ inline void
SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SysPrefixAttenOutput(GlobalTensor<OUT_T> &dst, LocalTensor<T> &attenRes,
                                                                   uint32_t startRow, uint32_t rows)
{
    LocalTensor<OUT_T> attenOut = outputQue1.AllocTensor<OUT_T>();
    if constexpr (!POST_QUANT) {
        if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
            Cast(attenOut, attenRes, AscendC::RoundMode::CAST_RINT, rows * headDimAlign);
        } else {
            Cast(attenOut, attenRes, AscendC::RoundMode::CAST_ROUND, rows * headDimAlign);
        }
    } else {
        perChannelQuantOffset = n2Idx * headDim * gSize;
        PostQuant(attenRes, attenOut, startRow, rows, headDimAlign, headDim);
    }

    outputQue1.EnQue(attenOut);
    outputQue1.DeQue<OUT_T>();
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = rows;
    dataCopyParams.blockLen = headDim * sizeof(OUT_T);
    dataCopyParams.srcStride = ((headDimAlign - headDim) * sizeof(OUT_T)) / BYTE_BLOCK;
    dataCopyParams.dstStride = 0;
    DataCopyPad(dst[startRow * headDim], attenOut, dataCopyParams);
    outputQue1.FreeTensor(attenOut);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SysPrefixSaveLse(uint32_t bIndex, uint32_t n2Index,
                                                                                      LocalTensor<T> &softmaxSumUb,
                                                                                      LocalTensor<T> &softmaxMaxUb,
                                                                                      uint32_t start, uint32_t count,
                                                                                      bool isPrefix)
{
    size_t lseSize = gSize * FP32_ONE_BLOCK_SIZE;
    uint64_t offset = (bIndex * kvHeadNum + n2Index) * lseSize * 2;
    if (!isPrefix) {
        offset += lseSize;
    }

    offset += (start * FP32_ONE_BLOCK_SIZE);
    CopyFixedUbToGm(lseSumGm[offset], softmaxSumUb, count * FP32_ONE_BLOCK_SIZE);
    CopyFixedUbToGm(lseMaxGm[offset], softmaxMaxUb, count * FP32_ONE_BLOCK_SIZE);
}

template <typename IFAT> __aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SysPrefixSaveLseFA()
{
    if constexpr (ANTIQUANT && (ANTIQUANT_PER_CHANNEL || ANTIQUANT_PER_CHANNEL_TOKEN)) {
        if (softmaxLseFlag && antiqOffsetExistFlag) {
            // per chnnel msd mm1 计算优化舍弃了offset，输出lse需要补回，以保持和公式一致
            Muls(qRowSumUb, qRowSumUb, static_cast<T>(tilingData->baseParams.scaleValue), gSize * FP32_ONE_BLOCK_SIZE);
            pipe_barrier(PIPE_V);
            Add(softmaxMaxUb, softmaxMaxUb, qRowSumUb, gSize * FP32_ONE_BLOCK_SIZE);
            pipe_barrier(PIPE_V);
        }
    }
    SysPrefixSaveLse(bIdx, n2Idx, softmaxSumUb, softmaxMaxUb, 0, gSize, calcSysPrefixFlag);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SysPrefixSaveLseFd(LocalTensor<T> &lseSum,
                                                                                        LocalTensor<T> &lseMax,
                                                                                        uint32_t start, uint32_t count)
{
    SysPrefixSaveLse(bIdx, n2Idx, lseSum, lseMax, start, count, calcSysPrefixFlag);
}

template <typename IFAT>
__aicore__ inline void
SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SysPrefixSaveZeroLse(uint32_t bIndex, uint32_t n2Index, bool isPrefix)
{
    size_t lseSize = gSize * FP32_ONE_BLOCK_SIZE;
    float minf = -3.40E+38;
    Duplicate(softmaxMaxUb, minf, lseSize);
    Duplicate(softmaxSumUb, FLOAT_ZERO, lseSize);
    pipe_barrier(PIPE_V);
    SysPrefixSaveLse(bIndex, n2Index, softmaxSumUb, softmaxMaxUb, 0, gSize, calcSysPrefixFlag);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SysPrefixSaveZeroAttenRes(uint32_t bIndex,
                                                                                               uint32_t n2Index,
                                                                                               bool isPrefix)
{
    uint64_t attenOffset = (bIndex * kvHeadNum + n2Index) * gSize * headDimAlign;
    GlobalTensor<T> dst = isPrefix ? sysPrefixAttenOutGm[attenOffset] : usrPromptAttenOutGm[attenOffset];
    matmul::InitOutput<T>(dst, gSize * headDimAlign, 0);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SysPrefixInitAllZeroOutput()
{
    if (calcSysPrefixFlag) {
        for (uint32_t i = 0; i < batchSizeQ; i++) {
            SysPrefixSaveZeroAttenRes(i, n2Idx, true);
            SysPrefixSaveZeroLse(i, n2Idx, true);
        }
    } else {
        SysPrefixSaveZeroAttenRes(bIdx, n2Idx, false);
        SysPrefixSaveZeroLse(bIdx, n2Idx, false);
    }
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SysPrefixSaveAttenRes(
    uint32_t bIndex, uint32_t n2Index, LocalTensor<T> &bmm2ResUb, uint32_t startRow, uint32_t rows, bool isPrefix)
{
    LocalTensor<T> outputUb = outputQue1.template AllocTensor<T>();
    DataCopy(outputUb, bmm2ResUb, rows * headDimAlign);

    uint64_t attenOffset = (bIndex * kvHeadNum + n2Index) * gSize * headDimAlign + startRow * headDimAlign;
    GlobalTensor<T> dst = isPrefix ? sysPrefixAttenOutGm[attenOffset] : usrPromptAttenOutGm[attenOffset];

    outputQue1.EnQue(outputUb);
    outputQue1.DeQue();
    DataCopy(dst, outputUb, rows * headDimAlign);
    outputQue1.FreeTensor(outputUb);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SoftmaxLseOutput(LocalTensor<T> &lse)
{
    LocalTensor<T> softmaxlseOut = outputQue2.template AllocTensor<T>();
    DataCopy(softmaxlseOut, lse, gSize * FP32_ONE_BLOCK_SIZE);
    outputQue2.EnQue(softmaxlseOut);
    outputQue2.DeQue<T>();

    DataCopyExtParams param;
    param.blockLen = sizeof(T);
    param.blockCount = gSize;
    param.srcStride = 0;
    param.dstStride = 0;
    DataCopyPad(softmaxLseGm[(bIdx * kvHeadNum + n2Idx) * gSize], softmaxlseOut, param);
    outputQue2.FreeTensor(softmaxlseOut);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::CopyFixedUbToGm(const GlobalTensor<T> &dst,
                                                                                     const LocalTensor<T> &src,
                                                                                     size_t size)
{
    LocalTensor<T> tmp = outputQue2.template AllocTensor<T>();
    DataCopy(tmp, src, size);

    outputQue2.EnQue(tmp);
    outputQue2.DeQue();
    DataCopy(dst, tmp, size);
    outputQue2.FreeTensor(tmp);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::CopyGmToFixedUb(LocalTensor<T> &dst,
                                                                                     const GlobalTensor<T> &src,
                                                                                     size_t size)
{
    LocalTensor<T> tmp = inputQue2.AllocTensor<T>();
    DataCopy(tmp, src, size);
    inputQue2.EnQue(tmp);
    inputQue2.DeQue<T>();
    DataCopy(dst, tmp, size);
    inputQue2.FreeTensor(tmp);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SysPrefixSaveMsdMax1(uint32_t bIndex)
{
    auto dst = msdRowMax1Gm[bIndex * msdRowMaxSize];
    CopyFixedUbToGm(dst, aMaxBmm1Ub, msdRowMaxSize);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SysPrefixLoadMsdMax1(uint32_t bIndex)
{
    CopyGmToFixedUb(aMaxBmm1Ub, msdRowMax1Gm[bIndex * msdRowMaxSize], msdRowMaxSize);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SysPrefixSaveMsdMax2(uint32_t bIndex)
{
    auto dst = msdRowMax2Gm[bIndex * msdRowMaxSize];
    CopyFixedUbToGm(dst, aMaxBmm2Ub, msdRowMaxSize);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SysPrefixLoadMsdMax2(uint32_t bIndex)
{
    CopyGmToFixedUb(aMaxBmm2Ub, msdRowMax2Gm[bIndex * msdRowMaxSize], msdRowMaxSize);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SysPrefixSaveMsdSum1(uint32_t bIndex)
{
    auto dst = msdRowSum1Gm[bIndex * msdRowSumSize];
    auto src = qRowSumBuff.Get<T>();
    CopyFixedUbToGm(dst, src, msdRowSumSize);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SysPrefixLoadMsdSum1(uint32_t bIndex)
{
    auto dst = qRowSumBuff.Get<T>();
    CopyGmToFixedUb(dst, msdRowSum1Gm[bIndex * msdRowMaxSize], msdRowMaxSize);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SysPrefixSaveMsdSum2(uint32_t bIndex)
{
    auto dst = msdRowSum2Gm[bIndex * msdRowSumSize];
    auto src = softmaxResRowSumBuff.Get<T>();
    CopyFixedUbToGm(dst, src, msdRowSumSize);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SysPrefixLoadMsdSum2(uint32_t bIndex)
{
    auto dst = softmaxResRowSumBuff.Get<T>();
    CopyGmToFixedUb(softmaxScaleResRowSumUb, msdRowSum2Gm[bIndex * msdRowMaxSize], msdRowMaxSize);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SysPrefixSaveSoftmaxMax(uint32_t bIndex)
{
    auto dst = softmaxRowMaxGm[bIndex * softmaxMaxSumExpSize];
    auto src = softmaxMaxBuff.Get<T>();
    CopyFixedUbToGm(dst, src, softmaxMaxSumExpSize);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SysPrefixLoadSoftmaxMax(uint32_t bIndex)
{
    auto dst = softmaxMaxBuff.Get<T>();
    CopyGmToFixedUb(dst, softmaxRowMaxGm[bIndex * softmaxMaxSumExpSize], softmaxMaxSumExpSize);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SysPrefixSaveSoftmaxSum(uint32_t bIndex)
{
    auto dst = softmaxRowSumGm[bIndex * softmaxMaxSumExpSize];
    auto src = softmaxSumBuff.Get<T>();
    CopyFixedUbToGm(dst, src, softmaxMaxSumExpSize);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SysPrefixLoadSoftmaxSum(uint32_t bIndex)
{
    auto dst = softmaxSumBuff.Get<T>();
    CopyGmToFixedUb(dst, softmaxRowSumGm[bIndex * softmaxMaxSumExpSize], softmaxMaxSumExpSize);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SysPrefixSaveSoftmaxExp(uint32_t bIndex)
{
    auto dst = softmaxRowExpGm[bIndex * softmaxMaxSumExpSize];
    auto src = softmaxExpBuff.Get<T>();
    CopyFixedUbToGm(dst, src, softmaxMaxSumExpSize);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::SysPrefixLoadSoftmaxExp(uint32_t bIndex)
{
    auto dst = softmaxExpBuff.Get<T>();
    CopyGmToFixedUb(dst, softmaxRowExpGm[bIndex * softmaxMaxSumExpSize], softmaxMaxSumExpSize);
}

template <typename IFAT>
__aicore__ inline void SparsePagedFusionAttentionAttenSplitBbn2s2Us2<IFAT>::DealKvInt4ColumnOdd(uint32_t actualColumnCount)
{
    LocalTensor<T> mmResUb = tmpBuff1.Get<T>();
    if constexpr (KVINT4) {
        if (actualSingleProcessSInnerSize % 2 == 1) {
            int blockIdx = actualColumnCount / FP32_ONE_BLOCK_SIZE;
            int offset = blockIdx * FP32_ONE_BLOCK_SIZE;
            int maskIdx = actualColumnCount % FP32_ONE_BLOCK_SIZE;
            uint64_t mask[2] = {1U << maskIdx, 0};

            Duplicate(mmResUb[offset], FLOAT_ZERO, mask, 1, 1, 8);
            pipe_barrier(PIPE_V);
        }
    }
}
#endif // SPARCE_PAGED_FUSION_ATTENTION_SPLIT_BBN2S2_US2
