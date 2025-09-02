
#include "register/op_def_registry.h"
#include "select_position_tiling.h"

using namespace ge;
using namespace AscendC;
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

  // block_ids PN 
  uint32_t kvHeadNum_ = context->GetInputTensor(0)->GetStorageShape().GetDim(0);
  uint32_t kvPageLen_ = context->GetInputTensor(0)->GetStorageShape().GetDim(1);

  // block_table maxBP
  uint32_t maxBatch_ = context->GetInputTensor(1)->GetStorageShape().GetDim(0);
  uint32_t maxPage_ = context->GetInputTensor(1)->GetStorageShape().GetDim(1);

  // indices BNK
  uint32_t batchSize_ = context->GetInputTensor(3)->GetStorageShape().GetDim(0);
  uint32_t qHeadNum_ = context->GetInputTensor(3)->GetStorageShape().GetDim(1);
  uint32_t k_ = context->GetInputTensor(3)->GetStorageShape().GetDim(2); // Default topK value

  // page_position BNmax
  uint32_t maxPageNum_ = context->GetInputTensor(4)->GetStorageShape().GetDim(2);

  uint32_t bns = batchSize_ * qHeadNum_;
  usedCoreNum_ = bns > coreNum_ ? coreNum_ : bns;
//   uint32_t blockSize_ = bns / (usedCoreNum_);
  uint32_t blockSize_ = (bns + usedCoreNum_ - 1) / usedCoreNum_;

  printf("batchSize_: %d, qHeadNum_: %d, kvHeadNum_: %d, kvPageLen_: %d, k_: %d, maxBatch_: %d, maxPage_: %d, maxPageNum_: %d, usedCoreNum_: %u, blockSize_: %u\n", batchSize_, qHeadNum_, kvHeadNum_, kvPageLen_, k_, maxBatch_, maxPage_, maxPageNum_, usedCoreNum_, blockSize_);

  SelectPositionTilingData tiling;
  tiling.set_bSize(batchSize_);
  tiling.set_n1Size(qHeadNum_);
  tiling.set_n2Size(kvHeadNum_);
  tiling.set_kvPageLen(kvPageLen_);
  tiling.set_maxBatch(maxBatch_);
  tiling.set_maxPage(maxPage_);
  tiling.set_maxPageNum(maxPageNum_);
  tiling.set_k(k_);
  tiling.set_blockSize(blockSize_);
  tiling.set_usedCoreNum(usedCoreNum_);

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