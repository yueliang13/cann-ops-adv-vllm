#include "kernel_operator.h"
#include "lib/matmul_intf.h"
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

template <typename aType, typename cType> class SelectPositionKernel {
public:
    __aicore__ inline SelectPositionKernel(){};
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR l1_cent, GM_ADDR d_l1_cent, GM_ADDR mask_empty, GM_ADDR select_nprobe, GM_ADDR indices, GM_ADDR workspace, const SelectPositionTilingData *__restrict tiling, AscendC::TPipe *pipe);
    template <bool isInitIndex = false, bool isHasfinish = false, bool isReuseSrc = false, enum TopKMode topkMode = AscendC::TopKMode::TOPK_NORMAL> __aicore__ inline void Process();

    template <typename T> __aicore__ inline void CopyIn(LocalTensor<T> &dstLocal, GlobalTensor<T> srcGm, uint64_t offset,uint32_t blkCnt, uint32_t CntAlign, uint32_t actualCnt);

    __aicore__ inline void VectorCompute(LocalTensor<cType> &mmResUb, LocalTensor<aType> &aUb, LocalTensor<aType> &bUb, uint32_t dealRowCount);

    __aicore__ inline void CopyOut();
        
    AscendC::GlobalTensor<aType> queryGm;
    AscendC::GlobalTensor<aType> l1CentGm;
    AscendC::GlobalTensor<cType> dl1CentGm;
    AscendC::GlobalTensor<cType> maskEmptyGm;
    AscendC::GlobalTensor<cType> selectNprobeGm;
    AscendC::GlobalTensor<int32_t> indicesGm;

private:
    int32_t bIdx = 0; // batch index for the current core.
    int32_t n2Idx = 0; // kvhead index for the current core.
    int32_t n1Idx = 0; // qhead index for the current core.
    int32_t blockIdx = 0; // Current core block index.
    int32_t batchSize = 0; 
    int32_t usedCoreNum = 0;
    int32_t qHeadNum = 0; // Current core kvhead index.
    int32_t dimNum = 0; // Dimension number.
    uint32_t blockSize = 0; // Single block size.
    uint32_t kvHeadNum = 0; // Number of kvheads.
    uint32_t clusterNum = 0; // Number of clusters.
    uint32_t nNumOfQInOneGroup = 0; // Number of queries in one group.
    uint32_t k = 0;
    uint32_t tmpsize = 0;

    // queue
    TQue<QuePosition::VECIN, 1> inQuery;   // 1K, inque
    TQue<QuePosition::VECIN, 1> inL1cent;   // 32K, inque
    TQue<QuePosition::VECIN, 1> inMaskEmpty;  

    TBuf<> tmpBuff1;    // 32K
    TBuf<> queryBuff;   // 1K
    TBuf<> bmm1ResBuff;     // 32K

    // topk
    // AscendC::TQue<AscendC::QuePosition::VECCALC, 1> inQueueDl1Cent;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueValue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueIndices;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> topKSrcIndexLocal;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> topKWrokLocal;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> topKFinishLocal;

    TopkTiling topkTilingData;
    TopKInfo topKInfo;

    uint64_t tensorACoreOffset = 0ULL;
    uint64_t tensorBCoreOffset = 0ULL;

    const int64_t ONE_BLK_SIZE{32};
    uint32_t IFA_MAX_REPEAT_TIMES = 256;
    uint32_t BUFFER_SIZE_BYTE_32K = 32768;
    uint64_t BYTE_BLOCK = 32UL;
    cType FLOAT_ZERO = 0;

    template <typename T> __aicore__ inline T Align(T num, T rnd)
    {
        return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
    }
};

/**
  * @brief  Set matmulLeaky input and output gm addr of current core.
  * @param  query: A matrix gm addr.
  * @param  l1_cent: B matrix gm addr.
  * @param  d_l1_cent: C matrix gm addr.
  * @param  workspace: Temporary gm space addr required by matmul calc.
  * @param  tiling: matmul tiling data.
  * @param  pipe: Global memory and sync management TPipe object.
  * @retval None
  */
template <typename aType, typename cType>
__aicore__ inline void SelectPositionKernel<aType, cType>::Init(GM_ADDR query, GM_ADDR l1_cent, GM_ADDR d_l1_cent, GM_ADDR mask_empty, GM_ADDR select_nprobe, GM_ADDR indices, GM_ADDR workspace, const SelectPositionTilingData *__restrict tiling, AscendC::TPipe *pipe)
{
    // ComputeConstexpr
    blockSize = tiling->blockSize; // Calculate the loop times of bn1Idx.
    blockIdx = AscendC::GetBlockIdx();
    batchSize = tiling->bSize;
    usedCoreNum = tiling->usedCoreNum;
    qHeadNum = tiling->n1Size;
    kvHeadNum = tiling->n2Size;
    dimNum = tiling->dSize;
    clusterNum = tiling->cSize;
    nNumOfQInOneGroup = tiling->gSize;
    k = tiling->k;
    tmpsize = tiling->tmpsize;
    topkTilingData = tiling->topkTilingData;

    //InitInput
    queryGm.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(query), batchSize * qHeadNum * dimNum);
    l1CentGm.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(l1_cent), kvHeadNum * clusterNum * dimNum);
    dl1CentGm.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(d_l1_cent), batchSize * qHeadNum * clusterNum);
    maskEmptyGm.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(mask_empty), batchSize * qHeadNum * clusterNum);
    selectNprobeGm.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(select_nprobe), batchSize * qHeadNum * k);
    indicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(indices), batchSize * qHeadNum * k);

    // InitBuffer
    pipe->InitBuffer(inQuery, 1, dimNum * sizeof(aType));
    pipe->InitBuffer(inL1cent, 1, clusterNum*dimNum * sizeof(aType));
    pipe->InitBuffer(inMaskEmpty, 1, clusterNum * sizeof(cType));

    // 128-160K
    // tmpBuff
    pipe->InitBuffer(tmpBuff1, BUFFER_SIZE_BYTE_32K);

    pipe->InitBuffer(bmm1ResBuff, clusterNum * sizeof(cType));

    // Init
    // pipe->InitBuffer(inQueueDl1Cent, BUFFER_NUM, clusterNum * sizeof(cType));
    pipe->InitBuffer(outQueueValue, BUFFER_NUM, k * sizeof(cType));
    pipe->InitBuffer(outQueueIndices, BUFFER_NUM, k * sizeof(int32_t));
    pipe->InitBuffer(topKSrcIndexLocal, clusterNum * sizeof(int32_t));
    pipe->InitBuffer(topKWrokLocal, tmpsize * sizeof(uint8_t));
    pipe->InitBuffer(topKFinishLocal, tmpsize * sizeof(bool));
}

/**
  * @brief  Main process of matmul calculation
  * @param  pipe: Global memory and sync management TPipe object.
  * @retval None
  */
template <typename aType, typename cType>
template <bool isInitIndex, bool isHasfinish, bool isReuseSrc, enum TopKMode topkMode>
__aicore__ inline void SelectPositionKernel<aType, cType>::Process()
{
    if (g_coreType == AIV && blockIdx >= usedCoreNum) {
        return;
        // skip cores
    } else {

        int64_t multiCoreInnerOffset = this->blockIdx * this->blockSize;
        int64_t multiCoreInnerLimit = multiCoreInnerOffset + this->blockSize;
        for (uint32_t bn1Idx = multiCoreInnerOffset; bn1Idx < multiCoreInnerLimit; bn1Idx++) {
            // Set the current batch and kvhead index.

            bIdx = bn1Idx / qHeadNum;
            n1Idx = bn1Idx % qHeadNum;
            n2Idx = n1Idx / nNumOfQInOneGroup;

            // copy in 
            tensorBCoreOffset = n2Idx * clusterNum * dimNum;
            tensorACoreOffset = bIdx * qHeadNum * dimNum + n1Idx * dimNum;
            LocalTensor<aType> inputUa = inQuery.template AllocTensor<aType>();
            CopyIn<aType>(inputUa, queryGm, tensorACoreOffset, 1, dimNum, dimNum);
            // DataCopy(inputUa, queryGm[tensorACoreOffset], dimNum);
            inQuery.template EnQue(inputUa);
            inQuery.DeQue<aType>();
            
            LocalTensor<aType> inputUb = inL1cent.template AllocTensor<aType>();
            CopyIn<aType>(inputUb, l1CentGm, tensorBCoreOffset, clusterNum, dimNum, dimNum);
            inL1cent.template EnQue(inputUb);
            inL1cent.DeQue<aType>();
    
            // compute
            LocalTensor<cType> mmResUb = bmm1ResBuff.Get<cType>();
            VectorCompute(mmResUb, inputUa, inputUb, clusterNum);
            // inQuery.FreeTensor(inputUa);
            // inL1cent.FreeTensor(inputUb);

            pipe_barrier(PIPE_ALL);
            // DumpTensor(mmResUb, 4, 128);

            int outOffset = bIdx * qHeadNum * clusterNum + n1Idx  * clusterNum;
            DataCopy(dl1CentGm[outOffset], mmResUb, clusterNum);

            // // mask
            // LocalTensor<cType> maskLocal = inMaskEmpty.template AllocTensor<cType>();
            // CopyIn<cType>(maskLocal, maskEmptyGm, outOffset, 1, clusterNum, clusterNum);
            // inMaskEmpty.template EnQue(maskLocal);
            // inMaskEmpty.DeQue<cType>();
            // uint32_t repeat_times = clusterNum / 64;
            // BinaryRepeatParams repeatParams;
            // repeatParams.dstBlkStride = 1;   // 单次迭代内Blk间的步幅，1表示连续
            // repeatParams.src0BlkStride = 1;
            // repeatParams.src1BlkStride = 1;
            // repeatParams.dstRepStride = 8;      // 这里要乘2的原因是，dst是4B的FP32，而src是2B的FP16
            // repeatParams.src0RepStride = 8;
            // repeatParams.src1RepStride = 8;
            // Mul(mmResUb, mmResUb, maskLocal, 64, repeat_times, repeatParams);
            // pipe_barrier(PIPE_ALL);
            // DataCopy(dl1CentGm[outOffset], mmResUb, clusterNum);

            // topk
            AscendC::LocalTensor<cType> dstValueLocal = outQueueValue.AllocTensor<cType>();
            AscendC::LocalTensor<int32_t> dstIndexLocal = outQueueIndices.AllocTensor<int32_t>();
            AscendC::LocalTensor<int32_t> srcLocalIndex = topKSrcIndexLocal.Get<int32_t>();
            // uint32_t repeatTimes = clusterNum / 64;
            // uint32_t dstBlkStride = 1;
            // uint32_t dstRepStride = 8;
            // CreateVecIndex(srcLocalIndex, (int32_t)0, 64, repeatTimes, dstBlkStride, dstRepStride);
            // LocalTensor<bool> finishLocal;
            LocalTensor<bool> finishLocal = topKFinishLocal.Get<bool>();
            AscendC::LocalTensor<uint8_t> tmpLocal = topKWrokLocal.Get<uint8_t>();

            topKInfo.outter = 1;  // 表示输入待排序数据的外轴长度
            topKInfo.inner = clusterNum;          // 表示输入待排序数据的内轴长度，inner必须是32的整数倍
            topKInfo.n = clusterNum;              // 表示输入待排序数据的内轴的实际长度
            AscendC::TopK<cType, isInitIndex, isHasfinish, isReuseSrc, topkMode>(
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
            // DumpTensor(dstValueLocal, 4, 64);
            // DumpTensor(dstIndexLocal, 5, 64);
            
            outQueueValue.EnQue(dstValueLocal);
            outQueueIndices.EnQue(dstIndexLocal);
            CopyOut();

            // pipe_barrier(PIPE_ALL);
            // DumpTensor(mmResUb, 4, 128);
        }
    }
}

template <typename aType, typename cType>
__aicore__ inline void SelectPositionKernel<aType, cType>::CopyOut() {
    int outOffset = bIdx * qHeadNum * k + n1Idx  * k;
    AscendC::LocalTensor<cType> dstValueLocal = outQueueValue.DeQue<cType>();
    AscendC::DataCopy(selectNprobeGm[outOffset], dstValueLocal, k);
    outQueueValue.FreeTensor(dstValueLocal);

    AscendC::LocalTensor<int32_t> indicesLocalTensor = outQueueIndices.DeQue<int32_t>();
    AscendC::DataCopy(indicesGm[outOffset], indicesLocalTensor, k);
    outQueueIndices.FreeTensor(indicesLocalTensor);
  }

template <typename aType, typename cType>
template <typename T>
__aicore__ inline void SelectPositionKernel<aType, cType>::CopyIn(LocalTensor<T> &dstLocal, GlobalTensor<T> srcGm, uint64_t offset,uint32_t blkCnt, uint32_t CntAlign, uint32_t actualCnt)
{
    uint32_t typeElementSize = ONE_BLK_SIZE / sizeof(T);
    DataCopyParams intriParams;
    intriParams.blockCount = blkCnt;
    intriParams.dstStride = (CntAlign - actualCnt) / typeElementSize;
    intriParams.blockLen = actualCnt / typeElementSize;
    intriParams.srcStride = 0;
    DataCopy(dstLocal, srcGm[offset], intriParams);
}

/**
  * @brief  Main process of matmul calculation
  * @param  mmResUb: 乘法结果.
  * @param  aUb: 输入向量A.
  * @param  bUb: 输入矩阵B.
  * @param  dealRowCount: 需要处理的总行数.
  * @retval None
  */
template <typename aType, typename cType>
__aicore__ inline void SelectPositionKernel<aType, cType>::VectorCompute(LocalTensor<cType> &mmResUb, LocalTensor<aType> &aUb, LocalTensor<aType> &bUb, uint32_t dealRowCount)
{
    LocalTensor<cType> vmlaResUb = tmpBuff1.Get<cType>();
    uint32_t elementSize = vmlaResUb.GetSize();
    // printf("elementSize: %d\n", elementSize);
    uint32_t maxDealRowCount = elementSize / (ONE_BLK_SIZE / sizeof(aType)); // 8192/16
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

extern "C" __global__ __aicore__ void select_position(GM_ADDR query, GM_ADDR l1_cent, GM_ADDR d_l1_cent, GM_ADDR mask_empty, GM_ADDR select_nprobe, GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling) {
    // GET_TILING_DATA(tiling_data, tiling);
    GET_TILING_DATA_WITH_STRUCT(SelectPositionTilingData, tilingDataIn, tiling);
    const SelectPositionTilingData *__restrict tilingData = &tilingDataIn;
    AscendC::TPipe pipe;
    SelectPositionKernel<half, float> op;
    op.Init(query, l1_cent, d_l1_cent, mask_empty, select_nprobe, indices, workspace, tilingData, &pipe);
    // op.Process<false, false, false, AscendC::TopKMode::TOPK_NORMAL>();
    op.Process();
    // TODO: user kernel impl
}