
#include "register/op_def_registry.h"
#include "select_position_tiling.h"

using namespace ge;
using namespace AscendC;

#define SPLIT_SEQ_LEN 18432

namespace optiling {
// static ge::graphStatus TilingFunc(gert::TilingContext* context)
static ge::graphStatus TilingSelectPosition(gert::TilingContext* context)
{
  size_t ubSize_ = 0;
  size_t l1Size_ = 0;
  size_t l0cSize_ = 0;
  size_t l0bSize_ = 0;
  uint32_t coreNum_ = 2;
  uint32_t usedCoreNum_ = 0;
  uint32_t aicNum_ = 0;
  uint32_t aivNum_ = 0;

  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize_);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bSize_);
  aicNum_ = ascendcPlatform.GetCoreNumAic();
  aivNum_ = ascendcPlatform.GetCoreNumAiv();
  coreNum_ = aivNum_; // default aiv num

  // key_ids BNS 
  uint32_t batchSize_ = context->GetInputTensor(0)->GetStorageShape().GetDim(0);
  uint32_t qHeadNum_ = context->GetInputTensor(0)->GetStorageShape().GetDim(1);
  uint32_t seqLen_ = context->GetInputTensor(0)->GetStorageShape().GetDim(2);

  uint32_t splitSeqLen_ = SPLIT_SEQ_LEN > seqLen_ ? seqLen_ : SPLIT_SEQ_LEN;
  uint32_t splitSeqRemainLen_ = seqLen_ % splitSeqLen_;

  uint32_t splitSeqNum_ = (seqLen_ + splitSeqLen_ - 1) / splitSeqLen_;
  printf("seqLen_: %d, splitSeqLen_: %d, splitSeqNum_: %d, splitSeqRemainLen_: %d\n", seqLen_, splitSeqLen_, splitSeqNum_, splitSeqRemainLen_);

  // indices BNK
  uint32_t k_ = context->GetInputTensor(1)->GetStorageShape().GetDim(2); // Default topK value
  printf("indices shape: %d, %d, %d\n", context->GetInputTensor(1)->GetStorageShape().GetDim(0), context->GetInputTensor(1)->GetStorageShape().GetDim(1), context->GetInputTensor(1)->GetStorageShape().GetDim(2));

  // token_position BNmax
  uint32_t maxTokenNum_ = context->GetInputTensor(2)->GetStorageShape().GetDim(2);

  uint32_t bns = batchSize_ * qHeadNum_;
  usedCoreNum_ = bns > coreNum_ ? coreNum_ : bns;
//   uint32_t blockSize_ = bns / (usedCoreNum_);
  uint32_t blockSize_ = (bns + usedCoreNum_ - 1) / usedCoreNum_;

  printf("batchSize_: %d, qHeadNum_: %d, seqLen_: %d, k_: %d, maxTokenNum_: %d, usedCoreNum_: %u, blockSize_: %u\n", batchSize_, qHeadNum_, seqLen_, k_, maxTokenNum_, usedCoreNum_, blockSize_);

  SelectPositionTilingData tiling;
  tiling.set_bSize(batchSize_);
  tiling.set_n1Size(qHeadNum_);
  tiling.set_seqLen(seqLen_);
  tiling.set_splitSeqLen(splitSeqLen_);
  tiling.set_splitSeqNum(splitSeqNum_);
  tiling.set_maxTokenNum(maxTokenNum_);
  tiling.set_k(k_);
  tiling.set_blockSize(blockSize_);
  tiling.set_usedCoreNum(usedCoreNum_);
  tiling.set_splitSeqRemainLen(splitSeqRemainLen_);

  context->SetBlockDim(aivNum_);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  size_t userWorkspaceSize = 0;
  size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = userWorkspaceSize + systemWorkspaceSize;
 
  return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    return GRAPH_SUCCESS;
}
}