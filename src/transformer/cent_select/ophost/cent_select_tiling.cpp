
#include "register/op_def_registry.h"
#include "cent_select_tiling.h"

#define CLUSTERBLOCKSIZE 256

using namespace ge;
using namespace AscendC;
namespace optiling {
// static ge::graphStatus TilingFunc(gert::TilingContext* context)
static ge::graphStatus TilingCentSelect(gert::TilingContext* context)
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

  // query 0 BN1D
  uint32_t batchSize_ = context->GetInputTensor(0)->GetStorageShape().GetDim(0);
  uint32_t qHeadNum_ = context->GetInputTensor(0)->GetStorageShape().GetDim(1);
  uint32_t dimNum_ = context->GetInputTensor(0)->GetStorageShape().GetDim(2);

  // l1_cent 1 N2CD
  uint32_t kvHeadNum_ = context->GetInputTensor(1)->GetStorageShape().GetDim(0);
  uint32_t clusterNum_ = context->GetInputTensor(1)->GetStorageShape().GetDim(1);
  uint32_t clusterBlockNum_ = (clusterNum_ + CLUSTERBLOCKSIZE - 1) / CLUSTERBLOCKSIZE;
  uint32_t clusterBlockSize_ = CLUSTERBLOCKSIZE;

  uint32_t numOfGroups_ = qHeadNum_ / kvHeadNum_;

  // block_ids 2 PN2 
  uint32_t kvPageLen_ = context->GetInputTensor(2)->GetStorageShape().GetDim(1);

  // block_table 3 maxBmaxP
  uint32_t maxBatch_ = context->GetInputTensor(3)->GetStorageShape().GetDim(0);
  uint32_t maxPage_ = context->GetInputTensor(3)->GetStorageShape().GetDim(1);

  // // indices BNK
  // uint32_t k_ = context->GetInputTensor(3)->GetStorageShape().GetDim(2); // Default topK value
  // uint32_t k_ = context->GetAttr("k")->GetInt(); // Default topK value
  uint32_t k_ = 64;


  // page_position 6 BNmax
  uint32_t maxPageNum_ = context->GetInputTensor(6)->GetStorageShape().GetDim(2);

  uint32_t bns = batchSize_ * qHeadNum_;
  usedCoreNum_ = bns > coreNum_ ? coreNum_ : bns;
//   uint32_t blockSize_ = bns / (usedCoreNum_);
  uint32_t blockSize_ = (bns + usedCoreNum_ - 1) / usedCoreNum_;

  printf("batchSize_: %d, qHeadNum_: %d, kvHeadNum_: %d, kvPageLen_: %d, k_: %d, maxBatch_: %d, maxPage_: %d, maxPageNum_: %d, usedCoreNum_: %u, blockSize_: %u, numOfGroups_: %u\n", batchSize_, qHeadNum_, kvHeadNum_, kvPageLen_, k_, maxBatch_, maxPage_, maxPageNum_, usedCoreNum_, blockSize_, numOfGroups_);

  CentSelectTilingData tiling;
  //base
  printf("batchSize_: %d, qHeadNum_: %d, kvHeadNum_: %d, numOfGroups_: %u, blockSize_: %u, usedCoreNum_: %u\n", batchSize_, qHeadNum_, kvHeadNum_, numOfGroups_, blockSize_, usedCoreNum_);
  tiling.set_bSize(batchSize_);
  tiling.set_n1Size(qHeadNum_);
  tiling.set_n2Size(kvHeadNum_);
  tiling.set_blockSize(blockSize_);
  tiling.set_usedCoreNum(usedCoreNum_);
  tiling.set_gSize(numOfGroups_);
  //compute cent
  printf("dimNum_: %u, clusterNum_: %u\n", dimNum_, clusterNum_);
  tiling.set_dSize(dimNum_);
  tiling.set_cSize(clusterNum_);
  tiling.set_clusterBlockNum(clusterBlockNum_);
  tiling.set_clusterBlockSize(clusterBlockSize_);
  //select position
  printf("kvPageLen_: %d, maxBatch_: %d, maxPage_: %d, maxPageNum_: %d\n", kvPageLen_, maxBatch_, maxPage_, maxPageNum_);
  tiling.set_kvPageLen(kvPageLen_);
  tiling.set_maxBatch(maxBatch_);
  tiling.set_maxPage(maxPage_);
  tiling.set_maxPageNum(maxPageNum_);
  // TopK Tiling
  printf("k_: %d\n", k_);
  uint32_t maxsize = 0;
  uint32_t minsize = 0;
  uint32_t dtypesize = 4;  // float32类型
  int32_t outter = 1;
  int32_t inner = clusterNum_;
  tiling.set_k(k_);
  AscendC::TopKTilingFunc(ascendcPlatform, inner, outter, k_, dtypesize, false, AscendC::TopKMode::TOPK_NORMAL, true, tiling.topkTilingData);
  AscendC::GetTopKMaxMinTmpSize(ascendcPlatform, inner, outter, false, false, AscendC::TopKMode::TOPK_NORMAL, true, dtypesize, maxsize, minsize);
  printf("TopK maxsize: %u, minsize: %u\n", maxsize, minsize);
  tiling.set_tmpsize(maxsize);

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