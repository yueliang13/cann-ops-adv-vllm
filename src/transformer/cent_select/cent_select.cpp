#include "lib/matmul_intf.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;
using AscendC::MulAddDst;

constexpr uint32_t BUFFER_NUM = 1;

#define FP16_ONE_BLOCK_SIZE 16
#define VMLA_ONE_REPEATE_ROW_COUNT 4
#define VMLA_ONE_REPEATE_COLUMN_COUNT 16
#define FP16_ONE_BLOCK_SIZE 16
#define FP32_ONE_BLOCK_SIZE 8
#define ALIGN_BLOCK_SIZE 16

#define PAGESIZE 128
class CentSelect {
public:
    __aicore__ inline CentSelect() {}
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR l1_cent, GM_ADDR block_ids, GM_ADDR block_table, GM_ADDR seq_len, GM_ADDR page_position, GM_ADDR page_position_length, GM_ADDR max_page_position_length, GM_ADDR workspace, const CentSelectTilingData *__restrict tilingData)
    {
        // base
        batchSize = tilingData->bSize;
        qHeadNum = tilingData->n1Size;
        kvHeadNum = tilingData->n2Size;
        numOfGroups = tilingData->gSize;
        blockSize = tilingData->blockSize;
        usedCoreNum = tilingData->usedCoreNum;
        // compute cent
        clusterNum = tilingData->cSize;
        dimNum = tilingData->dSize;
        clusterBlockNum = tilingData->clusterBlockNum;
        clusterBlockSize = tilingData->clusterBlockSize;
        // select position
        kvPageLen = tilingData->kvPageLen;
        maxBatch = tilingData->maxBatch;
        maxPage = tilingData->maxPage;
        maxPageNum = tilingData->maxPageNum;
        // topk
        k = tilingData->k;
        tmpsize = tilingData->tmpsize;
        topkTilingData = tilingData->topkTilingData;

        // max
        maxWorkSize = qHeadNum * tplPadding * 2 * 32;

        blockIdx = AscendC::GetBlockIdx();

        queryGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(query), batchSize * qHeadNum * dimNum);
        l1CentGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(l1_cent), kvHeadNum * clusterNum * dimNum);
        blockIdsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(block_ids),  kvHeadNum * kvPageLen);
        blockTableGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(block_table), maxBatch * maxPage);
        seqLenGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(seq_len), batchSize);
        pagePositionGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(page_position), batchSize * qHeadNum * maxPageNum);
        pagePositionLengthGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(page_position_length), batchSize * qHeadNum * tplPadding);
        maxPagePositionLengthGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(max_page_position_length), batchSize * tplPadding);

        // 8 is padding to 32B

        // init buffer
        m_pipe.InitBuffer(inQuery, BUFFER_NUM, dimNum * sizeof(half));
        m_pipe.InitBuffer(inL1Cent, BUFFER_NUM, clusterBlockSize * dimNum * sizeof(half));
        m_pipe.InitBuffer(inBlockIds, BUFFER_NUM, kvPageLen * sizeof(int32_t));
        m_pipe.InitBuffer(inBlockTable, BUFFER_NUM, maxPage * sizeof(int32_t));
        // output
        m_pipe.InitBuffer(outPagePosition, BUFFER_NUM, maxPageNum * sizeof(int32_t));
        m_pipe.InitBuffer(outPagePositionLength, BUFFER_NUM, tplPadding * sizeof(int32_t));
        // compute cent
        m_pipe.InitBuffer(tmpBuff1, BUFFER_SIZE_BYTE_32K);
        m_pipe.InitBuffer(tmpBmm1ResBuff, clusterBlockSize * sizeof(float));
        m_pipe.InitBuffer(bmm1ResBuff, clusterNum * sizeof(float));

        // topk
        m_pipe.InitBuffer(topKDstValue, k * sizeof(float));
        m_pipe.InitBuffer(topKDstIndex, k * sizeof(int32_t));
        m_pipe.InitBuffer(topKSrcIndexLocal, clusterNum * sizeof(int32_t));
        m_pipe.InitBuffer(topKWrokLocal, tmpsize * sizeof(uint8_t));
        m_pipe.InitBuffer(topKFinishLocal, tmpsize * sizeof(bool));

        // select position
        m_pipe.InitBuffer(tmpBuffPageBatch, maxPage * sizeof(int32_t));
        m_pipe.InitBuffer(tmpBuffSelectReduce, maxPage / 8 * sizeof(uint8_t));
        m_pipe.InitBuffer(tmpBuffSelectTmp, maxPage / 8 * sizeof(uint8_t));
        m_pipe.InitBuffer(selectBlockIdsIndexLocal, maxPage * sizeof(int32_t));

        //max page position length
        // m_pipe.InitBuffer(gatherPagePositionLength, qHeadNum * tplPadding * sizeof(int32_t));
        m_pipe.InitBuffer(totalPagePositionLength, qHeadNum * tplPadding * sizeof(int32_t));
        m_pipe.InitBuffer(totalPagePositionLengthFloat, qHeadNum * tplPadding * sizeof(float));
        m_pipe.InitBuffer(outMaxPagePositionLengthInt32, qHeadNum * tplPadding * sizeof(int32_t));
        // m_pipe.InitBuffer(gatherPageMask, tplPadding * sizeof(uint32_t));

        // max
        m_pipe.InitBuffer(maxWorkLocal, maxWorkSize * sizeof(uint8_t));
        m_pipe.InitBuffer(dstMaxLocal, 1 * sizeof(float));
        m_pipe.InitBuffer(outMaxPagePositionLength,1 , tplPadding * sizeof(int64_t));

    }
    __aicore__ inline void Process(GM_ADDR page_position_length)
    {
        if (g_coreType == AIV && blockIdx >= usedCoreNum) {
            // skip
        } else {
            int64_t multiCoreInnerOffset = this->blockIdx * this->blockSize;
            int64_t multiCoreInnerLimit = multiCoreInnerOffset + this->blockSize;
            
            for (uint32_t bn1Idx = multiCoreInnerOffset; bn1Idx < multiCoreInnerLimit; bn1Idx++) {
                bIdx = bn1Idx / qHeadNum;
                n1Idx = bn1Idx % qHeadNum;
                n2Idx = n1Idx / numOfGroups;

                if (bIdx < batchSize && n1Idx < qHeadNum) {
                    CopyIn();
                    AscendC::LocalTensor<int32_t> dstIndexLocal = topKDstIndex.Get<int32_t>();
                    ComputeCent(dstIndexLocal);
                    SelectPosition(dstIndexLocal);
                    CopyOut();
                }
            }
        }
        SyncAll();
        if (g_coreType == AIV && blockIdx < batchSize) {
            AscendC::LocalTensor<int32_t> totalPagePositionLengthLocal = totalPagePositionLength.Get<int32_t>();
            // AscendC::LocalTensor<int32_t> gatherPagePositionLengthLocal = gatherPagePositionLength.Get<int32_t>();
            AscendC::DataCopy(totalPagePositionLengthLocal, pagePositionLengthGlobal[blockIdx * qHeadNum * tplPadding], qHeadNum*tplPadding);
            // DumpTensor(totalPagePositionLengthLocal, 0, qHeadNum*tplPadding);

            AscendC::LocalTensor<float> totalPagePositionLengthFloatLocal = totalPagePositionLengthFloat.Get<float>();
            pipe_barrier(PIPE_ALL);
            AscendC::Cast(totalPagePositionLengthFloatLocal, totalPagePositionLengthLocal,  AscendC::RoundMode::CAST_CEIL, qHeadNum*tplPadding);
            // DumpTensor(totalPagePositionLengthFloatLocal, 1, qHeadNum*tplPadding);

            AscendC::LocalTensor<float> worklocal = maxWorkLocal.Get<float>();
            AscendC::LocalTensor<float> dstLocal = dstMaxLocal.Get<float>();

            // pipe_barrier(PIPE_ALL);
            AscendC::ReduceMax(dstLocal, totalPagePositionLengthFloatLocal, worklocal, qHeadNum*tplPadding);
            
            // printf("dstLocal: %f,\n", dstLocal.GetValue(0));
            float tmp = dstLocal.GetValue(0);
            int32_t maxSeqLen = dstLocal.GetValue(0) * 128;
            AscendC::LocalTensor<int32_t> outMaxPagePositionLengthInt32Local = outMaxPagePositionLengthInt32.Get<int32_t>();
            AscendC::Duplicate(outMaxPagePositionLengthInt32Local, maxSeqLen, tplPadding);

            AscendC::LocalTensor<int64_t> outMaxPagePositionLengthLocal = outMaxPagePositionLength.AllocTensor<int64_t>();
            AscendC::Cast(outMaxPagePositionLengthLocal, outMaxPagePositionLengthInt32Local,  AscendC::RoundMode::CAST_NONE, tplPadding);
            outMaxPagePositionLength.EnQue(outMaxPagePositionLengthLocal);
            outMaxPagePositionLength.DeQue<int64_t>();
            AscendC::DataCopy(maxPagePositionLengthGlobal[blockIdx*tplPadding], outMaxPagePositionLengthLocal, tplPadding);
            outMaxPagePositionLength.FreeTensor(outMaxPagePositionLengthLocal);

            // gather
            // AscendC::LocalTensor<uint32_t> gatherPageMaskLocal = gatherPageMask.Get<uint32_t>();
            // AscendC::Duplicate(gatherPageMaskLocal, uint32_t(2155905152), tplPadding);
            // uint64_t rsvdCnt = 0;
            // AscendC::GatherMask(gatherPagePositionLengthLocal, totalPagePositionLengthLocal, gatherPageMaskLocal,true, qHeadNum*tplPadding, {1, 1, 8, 8}, rsvdCnt);
            // printf("rsvdCnt: %d\n", rsvdCnt);
        }
    }
private:
    template <typename T> __aicore__ inline T Align(T num, T rnd)
    {
        return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
    }

    template <typename T>
    __aicore__ inline void DataCopyIn(LocalTensor<T> &dstLocal, GlobalTensor<T> srcGm, uint64_t offset,uint32_t blkCnt, uint32_t CntAlign, uint32_t actualCnt)
    {
        /*
         * datablock(block) = 32B  （文档中的数据块）
         * blockCount 指的是有传输多少个blockLen
         * blockLen 指的是每次传输有多少个datablock  （文档中的连续数据块）   （因此总数据量是：blockCount * blockLen * 32B）
         * srcStride 原地址的连续数据块的间隔
         * dstStride 目的地址的连续数据块的间隔
        */
        uint32_t typeElementSize = ONE_BLK_SIZE / sizeof(T);
        DataCopyParams intriParams;
        intriParams.blockCount = blkCnt;
        intriParams.dstStride = (CntAlign - actualCnt) / typeElementSize;
        intriParams.blockLen = actualCnt / typeElementSize;
        intriParams.srcStride = 0;
        DataCopy(dstLocal, srcGm[offset], intriParams);
    }

    __aicore__ inline void BlockingL1CentCopyIn(uint32_t idx)
    {
        int64_t l1CentOffset = n2Idx * clusterNum * dimNum + idx * dimNum;
        LocalTensor<half> inputUb = inL1Cent.AllocTensor<half>();
        DataCopyIn<half>(inputUb, l1CentGlobal, l1CentOffset, clusterBlockSize, dimNum, dimNum);
        inL1Cent.EnQue(inputUb);
    }
   
    __aicore__ inline void CopyIn()
    {
        // copy in 
        int64_t queryOffset = bIdx * qHeadNum * dimNum + n1Idx * dimNum;
        LocalTensor<half> inputUa = inQuery.AllocTensor<half>();
        DataCopyIn<half>(inputUa, queryGlobal, queryOffset, 1, dimNum, dimNum);
        inQuery.EnQue(inputUa);

        int64_t blockIdsOffset = n2Idx * kvPageLen;
        AscendC::LocalTensor<int32_t> blockIdsLocal = inBlockIds.AllocTensor<int32_t>();
        AscendC::DataCopy(blockIdsLocal, blockIdsGlobal[blockIdsOffset], kvPageLen);
        blockIdsLocal.SetSize(kvPageLen);
        inBlockIds.EnQue(blockIdsLocal);

        int64_t blockTableOffset = bIdx * maxPage;
        AscendC::LocalTensor<int32_t> blockTableLocal = inBlockTable.AllocTensor<int32_t>();
        AscendC::DataCopy(blockTableLocal, blockTableGlobal[blockTableOffset], maxPage);
        blockTableLocal.SetSize(maxPage);
        inBlockTable.EnQue(blockTableLocal);

        seqLen = seqLenGlobal.GetValue(bIdx);
        pageLen = (seqLen + PAGESIZE - 1) / PAGESIZE; 
        gatherMaskLen = pageLen / 8; // page num of one batch
        gatherMaskU32Len = pageLen / 32;
    }   

    __aicore__ inline void ComputeCent(AscendC::LocalTensor<int32_t> &dstIndexLocal)
    {
        LocalTensor<half> inputUa = inQuery.DeQue<half>();
        // compute
        LocalTensor<float> mmResUb = bmm1ResBuff.Get<float>();
        for (uint32_t i = 0; i < clusterBlockNum; i++) {
            uint32_t clusterBlockOffset = i * clusterBlockSize;
            BlockingL1CentCopyIn(clusterBlockOffset);
            LocalTensor<half> inputUb = inL1Cent.DeQue<half>();
            LocalTensor<float> tmpBmm1ResUb = tmpBmm1ResBuff.Get<float>();
            VectorCompute(tmpBmm1ResUb, inputUa, inputUb, clusterBlockSize);
            DataCopy(mmResUb[clusterBlockOffset], tmpBmm1ResUb, clusterBlockSize);
            inL1Cent.FreeTensor(inputUb);
        }
        inQuery.FreeTensor(inputUa);

        pipe_barrier(PIPE_ALL);

        // topk
        AscendC::LocalTensor<float> dstValueLocal = topKDstValue.Get<float>();
        AscendC::LocalTensor<int32_t> srcLocalIndex = topKSrcIndexLocal.Get<int32_t>();
        LocalTensor<bool> finishLocal = topKFinishLocal.Get<bool>();
        AscendC::LocalTensor<uint8_t> tmpLocal = topKWrokLocal.Get<uint8_t>();

        topKInfo.outter = 1;  // 表示输入待排序数据的外轴长度
        topKInfo.inner = clusterNum;          // 表示输入待排序数据的内轴长度，inner必须是32的整数倍
        topKInfo.n = clusterNum;              // 表示输入待排序数据的内轴的实际长度
        AscendC::TopK<float, false, false, false, AscendC::TopKMode::TOPK_NORMAL>(
            dstValueLocal,  
            dstIndexLocal, 
            mmResUb, 
            srcLocalIndex, 
            finishLocal, 
            tmpLocal,
            k, 
            topkTilingData, 
            topKInfo, 
            true
        );
    }

    /**
     * @brief  Main process of matmul calculation
     * @param  mmResUb: 乘法结果.
     * @param  aUb: 输入向量A.
     * @param  bUb: 输入矩阵B.
     * @param  dealRowCount: 需要处理的总行数.
     * @retval None
     */
    __aicore__ inline void VectorCompute(LocalTensor<float> &mmResUb, LocalTensor<half> &aUb, LocalTensor<half> &bUb, uint32_t dealRowCount)
    {
        LocalTensor<float> vmlaResUb = tmpBuff1.Get<float>();
        uint32_t elementSize = vmlaResUb.GetSize();
        // printf("elementSize: %d\n", elementSize);
        uint32_t maxDealRowCount = elementSize / (ONE_BLK_SIZE / sizeof(half)); // 8192/16
        uint32_t singleDealRowCnt = maxDealRowCount; // 512
        if (maxDealRowCount > dealRowCount) {
            singleDealRowCnt = dealRowCount;
        }
        // 这里计算了总行数除以单次处理的行数（片上存储约束），向上取整，得到需要处理的迭代次数
        uint32_t rowLoopCnt = (dealRowCount + singleDealRowCnt - 1) / singleDealRowCnt;
        uint32_t columnLoopCnt = dimNum / FP16_ONE_BLOCK_SIZE;   // 128 / 16 = 8 ,相当于每次VMLA都处理FP16_ONE_BLOCK_SIZE行，总共处理8次
        uint32_t rowElementCnt = dimNum;
        
        // 这里的VMLA每次处理64个数，每个Blk是16个数（FP16_ONE_BLOCK_SIZE），所以每次处理4个Blk（VMLA_ONE_REPEATE_ROW_COUNT）。这里设置为64是因为乘积的结果是用float表示的（32B）
        for (uint32_t i = 0, curDealRowCnt = singleDealRowCnt; i < rowLoopCnt; i++) {
            uint32_t rowStart = i * singleDealRowCnt;
            BinaryRepeatParams repeatParams;
            // 存储大小足够处理curDealRowCnt行，而VMLA指令每次处理VMLA_ONE_REPEATE_ROW_COUNT，因此需要重复执行repeat_times次VMLA迭代
            // uint32_t repeat_times = Align<uint32_t>(curDealRowCnt, (uint32_t)VMLA_ONE_REPEATE_ROW_COUNT) / VMLA_ONE_REPEATE_ROW_COUNT;
            uint32_t repeat_times = curDealRowCnt / VMLA_ONE_REPEATE_ROW_COUNT;
            // 每个BlockData(Blk)是32B，每次取8个Blk 
            repeatParams.dstBlkStride = 1;   // 单次迭代内Blk间的步幅，1表示连续
            repeatParams.src0BlkStride = 0;
            repeatParams.src1BlkStride = rowElementCnt / FP16_ONE_BLOCK_SIZE;
            repeatParams.dstRepStride = 2 * VMLA_ONE_REPEATE_ROW_COUNT;      // 这里要乘2的原因是，dst是4B的FP32，而src是2B的FP16
            repeatParams.src0RepStride = 0;
            repeatParams.src1RepStride = VMLA_ONE_REPEATE_ROW_COUNT * rowElementCnt / FP16_ONE_BLOCK_SIZE;

            // vmlaResUb size = 128 * 4 * 16 = 8192       
            // printf("repeat_times * VMLA_ONE_REPEATE_ROW_COUNT * 16: %d\n", repeat_times * VMLA_ONE_REPEATE_ROW_COUNT * 16);
            Duplicate(vmlaResUb, FLOAT_ZERO, repeat_times * VMLA_ONE_REPEATE_ROW_COUNT * 16);
            pipe_barrier(PIPE_V);

            for (uint32_t j = 0; j < columnLoopCnt; j++) {
                // aUb size = 128, bUb size = 128
                MulAddDst(vmlaResUb, aUb[j * FP16_ONE_BLOCK_SIZE], bUb[rowStart * rowElementCnt + j * FP16_ONE_BLOCK_SIZE],
                        64, repeat_times, repeatParams);
                pipe_barrier(PIPE_V);
            }

            // 结论： 将向量分一小块，每次算一块向量和多列的矩阵块的乘法，然后放到vmlaResUb中（需要偏移的命令控制）。然后循环则是循环向量的总块数次

            // 第一次处理IFA_MAX_REPEAT_TIMES-1次，第二次则
            repeat_times = IFA_MAX_REPEAT_TIMES - 1;
            for (uint32_t j = 0; j < curDealRowCnt; j += repeat_times) {
                // printf("j: %d, repeat_times: %d, curDealRowCnt: %d\n", j, repeat_times, curDealRowCnt);
                if (j + repeat_times > curDealRowCnt) {   // curDealRowCnt是512或者更小（尾行）
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
            // 要保证rowStart偏移是32Byte对齐，即singleDealRowCnt是8个元素对齐
            BlockReduceSum(mmResUb[rowStart], vmlaResUb[0],  Align<uint32_t>(curDealRowCnt, (uint32_t)8) / 8, 64, 1, 2, 16);
            pipe_barrier(PIPE_V);
        }
    }

    __aicore__ inline void SelectPosition(AscendC::LocalTensor<int32_t> indicesLocal)
    {

        // offset
        AscendC::LocalTensor<int32_t> blockTableLocal = inBlockTable.DeQue<int32_t>();
        AscendC::Muls(blockTableLocal, blockTableLocal, int32_t(4), pageLen);
        // src
        AscendC::LocalTensor<int32_t> blockIdsLocal = inBlockIds.DeQue<int32_t>();
        //dst
        AscendC::LocalTensor<int32_t> pageBatchLocal = tmpBuffPageBatch.Get<int32_t>();
        pipe_barrier(PIPE_ALL);
        // pageBatchLocal.SetSize(pageLen);

        // const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const LocalTensor<uint32_t>& srcOffsetLocal, const uint32_t srcBaseAddr, const uint32_t count)
        AscendC::Gather(pageBatchLocal, blockIdsLocal, blockTableLocal.ReinterpretCast<uint32_t>(), uint32_t(0), pageLen);

        // 拿到mask掩码和blockIds的索引
        LocalTensor<uint8_t> dstResultMaskLocal = tmpBuffSelectReduce.Get<uint8_t>();
        LocalTensor<uint8_t> dstMaskLocalTmp = tmpBuffSelectTmp.Get<uint8_t>();

        AscendC::CompareScalar(dstResultMaskLocal, pageBatchLocal, indicesLocal.GetValue(0), CMPMODE::EQ, pageLen);
        for (uint32_t i = 1; i < k; i++) {
            AscendC::CompareScalar(dstMaskLocalTmp, pageBatchLocal, indicesLocal.GetValue(i), CMPMODE::EQ, pageLen);
            pipe_barrier(PIPE_ALL);
            AscendC::Or(dstResultMaskLocal, dstResultMaskLocal, dstMaskLocalTmp, pageLen);
        }

        uint64_t rsvdCnt = 0;  //该参数表示收集之后的
        // 表示页的所有顺序索引
        LocalTensor<int32_t> blockIdsIndex = selectBlockIdsIndexLocal.Get<int32_t>();
        AscendC::CreateVecIndex(blockIdsIndex, (int32_t)0, pageLen);

        // output
        AscendC::LocalTensor<int32_t> pagePositionLengthLocal = outPagePositionLength.AllocTensor<int32_t>();
        pagePositionLengthLocal.SetSize(tplPadding);
        AscendC::LocalTensor<int32_t> pagePositionLocal = outPagePosition.AllocTensor<int32_t>();
        pagePositionLocal.SetSize(maxPageNum);
        AscendC::Duplicate(pagePositionLocal, 0x7fffffff, maxPageNum);

        AscendC::GatherMask(pagePositionLocal, blockIdsIndex, dstResultMaskLocal.ReinterpretCast<uint32_t>(), true, pageLen, {1, 1, 8, 8}, rsvdCnt);
        pagePositionLength = rsvdCnt;

        //output
        AscendC::Duplicate(pagePositionLengthLocal, pagePositionLength, tplPadding);
        outPagePositionLength.EnQue(pagePositionLengthLocal);
        outPagePosition.EnQue(pagePositionLocal);

        inBlockIds.FreeTensor(blockIdsLocal);
        inBlockTable.FreeTensor(blockTableLocal);

    }
    __aicore__ inline void CopyOut()
    {
        int64_t pagePositionOffset = bIdx * qHeadNum * maxPageNum + n1Idx * maxPageNum;
        AscendC::LocalTensor<int32_t> pagePositionLocal = outPagePosition.DeQue<int32_t>();
        AscendC::DataCopy(pagePositionGlobal[pagePositionOffset], pagePositionLocal, maxPageNum);
        outPagePosition.FreeTensor(pagePositionLocal);

        int64_t pagePositionLengthOffset = bIdx * qHeadNum * tplPadding + n1Idx * tplPadding;
        AscendC::LocalTensor<int32_t> pagePositionLengthLocal = outPagePositionLength.DeQue<int32_t>();
        DataCopy(pagePositionLengthGlobal[pagePositionLengthOffset], pagePositionLengthLocal, tplPadding);
        outPagePositionLength.FreeTensor(pagePositionLengthLocal);
    }
    
private:
    AscendC::TPipe m_pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQuery;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inL1Cent;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inBlockIds;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inBlockTable;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outPagePosition;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outPagePositionLength;
    AscendC::GlobalTensor<int32_t> blockIdsGlobal;
    AscendC::GlobalTensor<int32_t> blockTableGlobal;
    AscendC::GlobalTensor<int32_t> seqLenGlobal;
    AscendC::GlobalTensor<half> queryGlobal;
    AscendC::GlobalTensor<half> l1CentGlobal;
    AscendC::GlobalTensor<int32_t> pagePositionGlobal;
    AscendC::GlobalTensor<int32_t> pagePositionLengthGlobal;
    AscendC::GlobalTensor<int64_t> maxPagePositionLengthGlobal;



    // position
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuffPageBatch;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuffSelectReduce;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuffSelectTmp;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> selectBlockIdsIndexLocal;
    // AscendC::TBuf<AscendC::QuePosition::VECCALC> gatherPagePositionLength;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> totalPagePositionLength;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> totalPagePositionLengthFloat;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> outMaxPagePositionLengthInt32;
    // AscendC::TBuf<AscendC::QuePosition::VECCALC> gatherPageMask;


    // compute cent
    AscendC::TBuf<> tmpBuff1;    // 32K
    AscendC::TBuf<> queryBuff;   // 1K
    AscendC::TBuf<> bmm1ResBuff;     
    AscendC::TBuf<> tmpBmm1ResBuff;    

    // topk
    AscendC::TBuf<AscendC::QuePosition::VECCALC> topKDstValue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> topKDstIndex;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> topKSrcIndexLocal;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> topKWrokLocal;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> topKFinishLocal;

    // max 
    AscendC::TBuf<AscendC::QuePosition::VECCALC> maxWorkLocal;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> dstMaxLocal;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outMaxPagePositionLength;


    TopkTiling topkTilingData;
    TopKInfo topKInfo;

    const int64_t ONE_BLK_SIZE{32};
    uint32_t IFA_MAX_REPEAT_TIMES = 256;
    uint32_t BUFFER_SIZE_BYTE_32K = 32768;
    uint64_t BYTE_BLOCK = 32UL;
    float FLOAT_ZERO = 0;
private:
    // base
    int32_t batchSize = 0;
    int32_t qHeadNum = 0;
    int32_t kvHeadNum = 0;
    int32_t blockIdx = 0;
    int32_t blockSize = 0;
    int32_t usedCoreNum = 0;
    int32_t numOfGroups = 0;
    // compute cent
    int32_t clusterNum = 0;
    int32_t dimNum = 0;
    int32_t clusterBlockNum = 0;
    int32_t clusterBlockSize = 0;
    // select position
    int32_t maxPageNum = 0;
    int32_t maxBatch = 0;
    int32_t maxPage = 0;
    int32_t kvPageLen = 0;

    // topk
    int32_t k = 0;
    int32_t tmpsize = 0;

    uint64_t bIdx = 0;
    uint64_t n1Idx = 0;
    uint64_t n2Idx = 0;
    int32_t pagePositionLength = 0;
    int32_t tplPadding = 8; // padding to 32B for int32
    int32_t tplPaddingHalf = 4; // padding to 32B for int64
    int32_t seqLen = 0;
    int32_t pageLen = 0;
    int32_t gatherMaskLen = 0;
    int32_t gatherMaskU32Len = 0;

    // max
    int32_t maxWorkSize = 0;

}; // class CentSelect

extern "C" __global__ __aicore__ void cent_select(GM_ADDR query, GM_ADDR l1_cent, GM_ADDR block_ids, GM_ADDR block_table, GM_ADDR seq_len, GM_ADDR page_position, GM_ADDR page_position_length, GM_ADDR max_page_position_length, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(CentSelectTilingData, tilingDataIn, tiling);
    const CentSelectTilingData *__restrict tilingData = &tilingDataIn;

    CentSelect op; 
    op.Init(query, l1_cent, block_ids, block_table, seq_len, page_position,page_position_length, max_page_position_length, workspace, tilingData);
    op.Process(page_position_length);
}
