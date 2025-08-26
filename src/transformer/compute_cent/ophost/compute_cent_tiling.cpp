
#include "register/op_def_registry.h"
#include "compute_cent_tiling.h"

using namespace ge;
using namespace AscendC;

using namespace matmul_tiling;

namespace optiling {
// static ge::graphStatus TilingFunc(gert::TilingContext* context)
static ge::graphStatus TilingComputeCent(gert::TilingContext* context)
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
  printf("ubSize_: %d\n", ubSize_);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize_);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bSize_);
  aicNum_ = ascendcPlatform.GetCoreNumAic();
  aivNum_ = ascendcPlatform.GetCoreNumAiv();

  coreNum_ = aivNum_; // default aiv num

  // query, l1_cent, key_ids, d_l1_cent, mask_empty, select_nprobe, indices
  //BSNGD
  uint32_t batchSize_ = context->GetInputTensor(0)->GetStorageShape().GetDim(0);
  uint32_t qHeadNum_ = context->GetInputTensor(0)->GetStorageShape().GetDim(1);
  uint32_t seqSize_ = context->GetInputTensor(0)->GetStorageShape().GetDim(2);   //default seq size is 1
  uint32_t dimNum_ = context->GetInputTensor(0)->GetStorageShape().GetDim(3);
  //NCD
  uint32_t kvHeadNum_ = context->GetInputTensor(1)->GetStorageShape().GetDim(0);
  uint32_t clusterNum_ = context->GetInputTensor(1)->GetStorageShape().GetDim(1);
  uint32_t nNumOfQInOneGroup_ = qHeadNum_ / kvHeadNum_;

  //BNS key_ids
  uint32_t seqLen_ = context->GetInputTensor(2)->GetStorageShape().GetDim(2);

  // BSNK select_nprobe, indices
  int32_t k_ = context->GetInputTensor(5)->GetStorageShape().GetDim(3); // Default topK value

  // uint32_t bn = batchSize_ * kvHeadNum_;
  uint32_t bn = batchSize_ * qHeadNum_;
  usedCoreNum_ = bn > coreNum_ ? coreNum_ : bn;

  // uint32_t blockSize_ = batchSize_ * kvHeadNum_ / (usedCoreNum_);
  uint32_t blockSize_ = batchSize_ * qHeadNum_ / (usedCoreNum_);
  printf("batchSize_: %d, qHeadNum_: %d, seqSize_: %d, dimNum_: %d\n", batchSize_, qHeadNum_, seqSize_, dimNum_);
  printf("kvHeadNum_: %d, clusterNum_: %d, nNumOfQInOneGroup_: %d\n", kvHeadNum_, clusterNum_, nNumOfQInOneGroup_);
  printf("blockSize_: %d, seqSize_: %d, batchSize_: %d, kvHeadNum_: %d, qHeadNum_: %d, nNumOfQInOneGroup_: %d, dimNum_: %d, clusterNum_: %d, usedCoreNum_: %d\n", blockSize_, seqSize_, batchSize_, kvHeadNum_, qHeadNum_, nNumOfQInOneGroup_, dimNum_, clusterNum_, usedCoreNum_);
  
  ComputeCentTilingData tiling;
  tiling.set_blockSize(blockSize_);
  tiling.set_s1Size(seqSize_);
  tiling.set_bSize(batchSize_);
  tiling.set_n2Size(kvHeadNum_);
  tiling.set_n1Size(qHeadNum_);
  tiling.set_gSize(nNumOfQInOneGroup_);
  tiling.set_dSize(dimNum_);
  tiling.set_cSize(clusterNum_);
  tiling.set_usedCoreNum(usedCoreNum_);
  tiling.set_seqLen(seqLen_);
  // TopK Tiling
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