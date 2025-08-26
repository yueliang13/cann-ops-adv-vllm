#include "lib/matmul_intf.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

class SelectPosition {
public:
    __aicore__ inline SelectPosition() {}
    __aicore__ inline void Init(GM_ADDR key_ids, GM_ADDR indices, GM_ADDR token_position, GM_ADDR token_position_length, GM_ADDR workspace, const SelectPositionTilingData *__restrict tilingData)
    {
        batchSize = tilingData->bSize;
        qHeadNum = tilingData->n1Size;
        seqLen = tilingData->seqLen;
        k = tilingData->k;
        maxTokenNum = tilingData->maxTokenNum;
        blockSize = tilingData->blockSize;
        usedCoreNum = tilingData->usedCoreNum;
        splitSeqNum = tilingData->splitSeqNum;
        splitSeqLen = tilingData->splitSeqLen;
        splitSeqRemainLen = tilingData->splitSeqRemainLen;
        blockIdx = AscendC::GetBlockIdx();

        keyIdsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(key_ids), batchSize * qHeadNum * seqLen);
        indicesGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(indices), batchSize * qHeadNum * k);
        tokenPositionGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(token_position), batchSize * qHeadNum * maxTokenNum);
        tokenPositionLengthGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(token_position_length), batchSize * qHeadNum*tplPadding);
        // 8 is padding to 32B

        m_pipe.InitBuffer(inKeyIds, 1, splitSeqLen * sizeof(int32_t));
        m_pipe.InitBuffer(inIndices, 1, k * sizeof(int32_t));
        m_pipe.InitBuffer(outTokenPosition, 1, maxTokenNum * sizeof(int32_t));
        m_pipe.InitBuffer(outTokenPositionLength, 1, tplPadding * sizeof(int32_t));
        m_pipe.InitBuffer(tmpBuffSelectReduce, splitSeqLen / 8 * sizeof(uint8_t));
        m_pipe.InitBuffer(tmpBuffSelectTmp, splitSeqLen / 8 * sizeof(uint8_t));
        m_pipe.InitBuffer(selectKeyIdsIndexLocal, splitSeqLen * sizeof(int32_t));
    }
    __aicore__ inline void Process(GM_ADDR token_position_length)
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
                if (bIdx >= batchSize || n1Idx >= qHeadNum) {
                    return;
                }
                int64_t indicesOffset = bIdx * qHeadNum * k + n1Idx * k;
                AscendC::LocalTensor<int32_t> indicesLocal = inIndices.AllocTensor<int32_t>();
                AscendC::DataCopy(indicesLocal, indicesGlobal[indicesOffset], k);
                inIndices.EnQue(indicesLocal);
                inIndices.DeQue<int32_t>();
                indicesLocal.SetSize(k);

                AscendC::LocalTensor<int32_t> tokenPositionLengthLocal = outTokenPositionLength.AllocTensor<int32_t>();
                tokenPositionLengthLocal.SetSize(tplPadding);
                AscendC::LocalTensor<int32_t> tokenPositionLocal = outTokenPosition.AllocTensor<int32_t>();
                tokenPositionLocal.SetSize(maxTokenNum);
                AscendC::Duplicate(tokenPositionLocal, 0x7fffffff, maxTokenNum);
                tokenPositionLength = 0;

                for (uint32_t splitSeqIdx = 0; splitSeqIdx < splitSeqNum; splitSeqIdx++) {
                    uint32_t handleLen = splitSeqIdx==splitSeqNum-1&&splitSeqRemainLen!=0 ? splitSeqRemainLen : splitSeqLen;
                    CopyIn(splitSeqIdx, handleLen);
                    Compute(splitSeqIdx, tokenPositionLocal, indicesLocal, handleLen);
                }
                AscendC::Duplicate(tokenPositionLengthLocal, tokenPositionLength, tplPadding);
                outTokenPositionLength.EnQue(tokenPositionLengthLocal);
                
            // tokenPositionLengthLocal.SetValue((bn1Idx-multiCoreInnerOffset)*8, tokenPositionLength);
                outTokenPosition.EnQue(tokenPositionLocal);
                CopyOut();
                inIndices.FreeTensor(indicesLocal);
                // tokenPositionLengthGlobal.SetValue(bIdx * qHeadNum + n1Idx, tokenPositionLength);
                // DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(tokenPositionLengthGlobal);
            }
        
        }
    }
private:
    __aicore__ inline void CopyIn(uint32_t splitSeqIdx, uint32_t handleLen)
    {
        int64_t keyIdsOffset = bIdx * qHeadNum * seqLen + n1Idx * seqLen + splitSeqIdx * splitSeqLen;
        AscendC::LocalTensor<int32_t> keyIdsLocal = inKeyIds.AllocTensor<int32_t>();
        AscendC::DataCopy(keyIdsLocal, keyIdsGlobal[keyIdsOffset], handleLen);
        inKeyIds.EnQue(keyIdsLocal);
    }   

    __aicore__ inline int32_t Align(uint64_t num, int32_t rnd)
    {
        return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
    }

    __aicore__ inline void Compute(uint32_t splitSeqIdx, AscendC::LocalTensor<int32_t> tokenPositionLocal, AscendC::LocalTensor<int32_t> indicesLocal, uint32_t handleLen)
    {

        AscendC::LocalTensor<int32_t> keyIdsLocal = inKeyIds.DeQue<int32_t>();
        keyIdsLocal.SetSize(handleLen);

        // 拿到mask掩码和keyids的索引
        LocalTensor<uint8_t> dstResultMaskLocal = tmpBuffSelectReduce.Get<uint8_t>();
        LocalTensor<uint8_t> dstMaskLocalTmp = tmpBuffSelectTmp.Get<uint8_t>();

        AscendC::CompareScalar(dstResultMaskLocal, keyIdsLocal, indicesLocal.GetValue(0), CMPMODE::EQ, handleLen);
        for (uint32_t i = 1; i < k; i++) {
            AscendC::CompareScalar(dstMaskLocalTmp, keyIdsLocal, indicesLocal.GetValue(i), CMPMODE::EQ, handleLen);
            AscendC::Or(dstResultMaskLocal, dstResultMaskLocal, dstMaskLocalTmp, handleLen);
        }
        LocalTensor<int32_t> keyIdsIndex = selectKeyIdsIndexLocal.Get<int32_t>();
        AscendC::CreateVecIndex(keyIdsIndex, (int32_t)splitSeqIdx * splitSeqLen, handleLen);
        // uint32_t mask = seqLen;   //该参数表示处理的src0Local(keyIdsIndex)的元素的数量
        uint64_t rsvdCnt = 0;  //该参数表示收集之后的dstLocal的元素数量
        // reduceMode = true; counter模式
        // src0BlockStride = 1; 单次迭代内数据间隔1个datablock，即数据连续读取和写入
        // repeatTimes = 1; 该参数表示只迭代一次
        // src0RepeatStride = 8;源操作数迭代间数据间隔8个datablock
        // src1RepeatStride = 8;源操作数迭代间数据间隔8个datablock
        AscendC::LocalTensor<uint32_t> resultMaskLocal = dstResultMaskLocal.ReinterpretCast<uint32_t>();
        AscendC::GatherMask(tokenPositionLocal[tokenPositionLength], keyIdsIndex, resultMaskLocal, true, handleLen, {1, 1, 8, 8}, rsvdCnt);
        tokenPositionLength += Align(rsvdCnt, 32);

        inKeyIds.FreeTensor(keyIdsLocal);
    }
    __aicore__ inline void CopyOut()
    {
        int64_t tokenPositionOffset = bIdx * qHeadNum * maxTokenNum + n1Idx * maxTokenNum;
        AscendC::LocalTensor<int32_t> tokenPositionLocal = outTokenPosition.DeQue<int32_t>();
        AscendC::DataCopy(tokenPositionGlobal[tokenPositionOffset], tokenPositionLocal, maxTokenNum);
        outTokenPosition.FreeTensor(tokenPositionLocal);

        int64_t tokenPositionLengthOffset = bIdx * qHeadNum * tplPadding + n1Idx * tplPadding;
        AscendC::LocalTensor<int32_t> tokenPositionLengthLocal = outTokenPositionLength.DeQue<int32_t>();
        DataCopy(tokenPositionLengthGlobal[tokenPositionLengthOffset], tokenPositionLengthLocal, tplPadding);
        outTokenPositionLength.FreeTensor(tokenPositionLengthLocal);
    }
    
private:
    AscendC::TPipe m_pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inKeyIds;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inIndices;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outTokenPosition;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outTokenPositionLength;
    AscendC::GlobalTensor<int32_t> keyIdsGlobal;
    AscendC::GlobalTensor<int32_t> indicesGlobal;
    AscendC::GlobalTensor<int32_t> tokenPositionGlobal;
    AscendC::GlobalTensor<int32_t> tokenPositionLengthGlobal;
    // position
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuffSelectReduce;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuffSelectTmp;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> selectKeyIdsIndexLocal;
private:
    int32_t batchSize = 0;
    int32_t qHeadNum = 0;
    int32_t seqLen = 0;
    int32_t k = 0;
    int32_t maxTokenNum = 0;
    int32_t blockIdx = 0;
    int32_t blockSize = 0;
    int32_t usedCoreNum = 0;
    int32_t splitSeqLen = 0;
    int32_t splitSeqNum = 0;
    int32_t splitSeqRemainLen = 0;
    int32_t bIdx = 0;
    int32_t n1Idx = 0;
    int32_t tokenPositionLength = 0;
    int32_t tplPadding = 8; // padding to 32B
}; // class SelectPosition

extern "C" __global__ __aicore__ void select_position(GM_ADDR key_ids, GM_ADDR indices, GM_ADDR token_position, GM_ADDR token_position_length, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(SelectPositionTilingData, tilingDataIn, tiling);
    const SelectPositionTilingData *__restrict tilingData = &tilingDataIn;

    SelectPosition op; 
    op.Init(key_ids, indices, token_position, token_position_length, workspace, tilingData);
    op.Process(token_position_length);
}
