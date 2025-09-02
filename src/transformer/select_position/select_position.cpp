#include "lib/matmul_intf.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

#define PAGESIZE 128
class SelectPosition {
public:
    __aicore__ inline SelectPosition() {}
    __aicore__ inline void Init(GM_ADDR block_ids, GM_ADDR block_table, GM_ADDR seq_len, GM_ADDR indices, GM_ADDR page_position, GM_ADDR page_position_length, GM_ADDR workspace, const SelectPositionTilingData *__restrict tilingData)
    {
        batchSize = tilingData->bSize;
        qHeadNum = tilingData->n1Size;
        kvHeadNum = tilingData->n2Size;
        kvPageLen = tilingData->kvPageLen;
        maxBatch = tilingData->maxBatch;
        maxPage = tilingData->maxPage;
        k = tilingData->k;
        maxPageNum = tilingData->maxPageNum;
        blockSize = tilingData->blockSize;
        usedCoreNum = tilingData->usedCoreNum;
        blockIdx = AscendC::GetBlockIdx();
        

        blockIdsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(block_ids),  kvHeadNum * kvPageLen);
        blockTableGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(block_table), maxBatch * maxPage);
        seqLenGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(seq_len), batchSize);
        indicesGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(indices), batchSize * qHeadNum * k);
        pagePositionGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(page_position), batchSize * qHeadNum * maxPageNum);
        pagePositionLengthGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(page_position_length), batchSize * qHeadNum*tplPadding);
        // 8 is padding to 32B

        m_pipe.InitBuffer(inBlockIds, 1, kvPageLen * sizeof(int32_t));
        m_pipe.InitBuffer(inBlockTable, 1, maxPage * sizeof(int32_t));
        m_pipe.InitBuffer(inIndices, 1, k * sizeof(int32_t));
        m_pipe.InitBuffer(outPagePosition, 1, maxPageNum * sizeof(int32_t));
        m_pipe.InitBuffer(outPagePositionLength, 1, tplPadding * sizeof(int32_t));
        m_pipe.InitBuffer(tmpBuffPageBatch, maxPage * sizeof(int32_t));
        m_pipe.InitBuffer(tmpBuffSelectReduce, maxPage / 8 * sizeof(uint8_t));
        m_pipe.InitBuffer(tmpBuffSelectTmp, maxPage / 8 * sizeof(uint8_t));
        m_pipe.InitBuffer(selectBlockIdsIndexLocal, maxPage * sizeof(int32_t));
    }
    __aicore__ inline void Process(GM_ADDR page_position_length)
    {
        if (g_coreType == AIV && blockIdx >= usedCoreNum) {
            return;
            // skip cores
        } else {
            int64_t multiCoreInnerOffset = this->blockIdx * this->blockSize;
            int64_t multiCoreInnerLimit = multiCoreInnerOffset + this->blockSize;
            
            for (uint32_t bn1Idx = multiCoreInnerOffset; bn1Idx < multiCoreInnerLimit; bn1Idx++) {
                bIdx = bn1Idx / qHeadNum;
                n1Idx = bn1Idx % qHeadNum;
                n2Idx = bn1Idx % kvHeadNum;
                if (bIdx >= batchSize || n1Idx >= qHeadNum) {
                    return;
                }

                int64_t indicesOffset = bIdx * qHeadNum * k + n1Idx * k;
                AscendC::LocalTensor<int32_t> indicesLocal = inIndices.AllocTensor<int32_t>();
                AscendC::DataCopy(indicesLocal, indicesGlobal[indicesOffset], k);
                inIndices.EnQue(indicesLocal);
                inIndices.DeQue<int32_t>();
                indicesLocal.SetSize(k);

                AscendC::LocalTensor<int32_t> pagePositionLengthLocal = outPagePositionLength.AllocTensor<int32_t>();
                pagePositionLengthLocal.SetSize(tplPadding);
                AscendC::LocalTensor<int32_t> pagePositionLocal = outPagePosition.AllocTensor<int32_t>();
                pagePositionLocal.SetSize(maxPageNum);
                AscendC::Duplicate(pagePositionLocal, 0x7fffffff, maxPageNum);

                CopyIn();
                Compute(pagePositionLocal, indicesLocal);
                AscendC::Duplicate(pagePositionLengthLocal, pagePositionLength, tplPadding);
                outPagePositionLength.EnQue(pagePositionLengthLocal);
                outPagePosition.EnQue(pagePositionLocal);
                CopyOut();
                inIndices.FreeTensor(indicesLocal);
            }
        
        }
    }
private:
    __aicore__ inline void CopyIn()
    {
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

    __aicore__ inline int32_t Align(uint64_t num, int32_t rnd)
    {
        return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
    }

    __aicore__ inline void Compute(AscendC::LocalTensor<int32_t> pagePositionLocal, AscendC::LocalTensor<int32_t> indicesLocal)
    {
        // offset
        AscendC::LocalTensor<int32_t> blockTableLocal = inBlockTable.DeQue<int32_t>();
        AscendC::Muls(blockTableLocal, blockTableLocal, int32_t(4), pageLen);
        // src
        AscendC::LocalTensor<int32_t> blockIdsLocal = inBlockIds.DeQue<int32_t>();
        //dst
        AscendC::LocalTensor<int32_t> pageBatchLocal = tmpBuffPageBatch.Get<int32_t>();

        // const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const LocalTensor<uint32_t>& srcOffsetLocal, const uint32_t srcBaseAddr, const uint32_t count)
        AscendC::Gather(pageBatchLocal, blockIdsLocal, blockTableLocal.ReinterpretCast<uint32_t>(), uint32_t(0), pageLen);

        // 拿到mask掩码和blockIds的索引
        LocalTensor<int32_t> dstResultMaskLocalI32 = tmpBuffSelectReduce.Get<int32_t>();
        LocalTensor<int32_t> dstMaskLocalTmpI32 = tmpBuffSelectTmp.Get<int32_t>();
        AscendC::Duplicate(dstResultMaskLocalI32, 0x0, gatherMaskU32Len);
        AscendC::Duplicate(dstMaskLocalTmpI32, 0x0, gatherMaskU32Len);
        LocalTensor<uint8_t> dstResultMaskLocal = dstResultMaskLocalI32.ReinterpretCast<uint8_t>();
        LocalTensor<uint8_t> dstMaskLocalTmp = dstMaskLocalTmpI32.ReinterpretCast<uint8_t>();

        // LocalTensor<uint8_t> dstResultMaskLocal = tmpBuffSelectReduce.Get<uint8_t>();
        // LocalTensor<uint8_t> dstMaskLocalTmp = tmpBuffSelectTmp.Get<uint8_t>();

        AscendC::CompareScalar(dstResultMaskLocal, pageBatchLocal, indicesLocal.GetValue(0), CMPMODE::EQ, pageLen);
        // DumpTensor(dstResultMaskLocal, 0, gatherMaskLen);
        for (uint32_t i = 1; i < k; i++) {
            AscendC::CompareScalar(dstMaskLocalTmp, pageBatchLocal, indicesLocal.GetValue(i), CMPMODE::EQ, pageLen);
            AscendC::Or(dstResultMaskLocal, dstResultMaskLocal, dstMaskLocalTmp, pageLen);
        }
        // DumpTensor(dstResultMaskLocal, 1, gatherMaskLen);
        // uint32_t mask = seqLen;   //该参数表示处理的src0Local(blockIdsIndex)的元素的数量
        uint64_t rsvdCnt = 0;  //该参数表示收集之后的dstLocal的元素数量
        // reduceMode = true; counter模式
        // src0BlockStride = 1; 单次迭代内数据间隔1个datablock，即数据连续读取和写入
        // repeatTimes = 1; 该参数表示只迭代一次
        // src0RepeatStride = 8;源操作数迭代间数据间隔8个datablock
        // src1RepeatStride = 8;源操作数迭代间数据间隔8个datablock
        AscendC::LocalTensor<uint32_t> resultMaskLocal = dstResultMaskLocal.ReinterpretCast<uint32_t>();
        // 表示页的所有顺序索引
        LocalTensor<int32_t> blockIdsIndex = selectBlockIdsIndexLocal.Get<int32_t>();
        AscendC::CreateVecIndex(blockIdsIndex, (int32_t)0, pageLen);

        AscendC::GatherMask(pagePositionLocal, blockIdsIndex, resultMaskLocal, true, pageLen, {1, 1, 8, 8}, rsvdCnt);
        // DumpTensor(pagePositionLocal, 2, rsvdCnt);

        pagePositionLength = rsvdCnt;

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
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inBlockIds;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inBlockTable;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inIndices;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outPagePosition;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outPagePositionLength;
    AscendC::GlobalTensor<int32_t> blockIdsGlobal;
    AscendC::GlobalTensor<int32_t> blockTableGlobal;
    AscendC::GlobalTensor<int32_t> seqLenGlobal;
    AscendC::GlobalTensor<int32_t> indicesGlobal;
    AscendC::GlobalTensor<int32_t> pagePositionGlobal;
    AscendC::GlobalTensor<int32_t> pagePositionLengthGlobal;
    // position
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuffPageBatch;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuffSelectReduce;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuffSelectTmp;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> selectBlockIdsIndexLocal;
private:
    int32_t batchSize = 0;
    int32_t qHeadNum = 0;
    int32_t kvHeadNum = 0;
    int32_t kvPageLen = 0;
    int32_t k = 0;
    int32_t maxPageNum = 0;
    int32_t maxBatch = 0;
    int32_t maxPage = 0;
    int32_t blockIdx = 0;
    int32_t blockSize = 0;
    int32_t usedCoreNum = 0;
    uint64_t bIdx = 0;
    uint64_t n1Idx = 0;
    uint64_t n2Idx = 0;
    int32_t pagePositionLength = 0;
    int32_t tplPadding = 8; // padding to 32B
    int32_t seqLen = 0;
    int32_t pageLen = 0;
    int32_t gatherMaskLen = 0;
    int32_t gatherMaskU32Len = 0;

}; // class SelectPosition

extern "C" __global__ __aicore__ void select_position(GM_ADDR block_ids, GM_ADDR block_table, GM_ADDR seq_len, GM_ADDR indices, GM_ADDR page_position, GM_ADDR page_position_length, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(SelectPositionTilingData, tilingDataIn, tiling);
    const SelectPositionTilingData *__restrict tilingData = &tilingDataIn;

    SelectPosition op; 
    op.Init(block_ids, block_table, seq_len, indices,page_position,page_position_length, workspace, tilingData);
    op.Process(page_position_length);
}
