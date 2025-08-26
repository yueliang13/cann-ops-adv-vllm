/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_init_routing_v2_grad_regbase_split_h.cpp
 * \brief
 */

 #include "moe_init_routing_v2_grad_tiling.h"
 #include "register/op_def_registry.h"
 #include "tiling/tiling_templates_registry.h"
 
 namespace optiling {
 constexpr uint64_t TILINGKEY_SPLIT_H_DROPLESS = 200001;
 constexpr uint64_t TILINGKEY_SPLIT_H_DROP_PAD = 200002;
 constexpr uint64_t TILINGKEY_SPLIT_H_ACTIVE = 200003;
 const static int64_t MIN_SPLIT_H_TOPK_THRESHOLD = 64;
 const static int64_t MAX_CACHE_LINE_SIZE = 512;
 const static int64_t DOUBLE_BUFFER = 2;
 const static int64_t FP16_BF16_SIZE = 2;
 class MoeInitRoutingV2GradRegbaseSplitH : public MoeInitRoutingV2GradTilingBaseClass {
  public:
   explicit MoeInitRoutingV2GradRegbaseSplitH(gert::TilingContext* context)
       : MoeInitRoutingV2GradTilingBaseClass(context) { Reset(); }
   ~MoeInitRoutingV2GradRegbaseSplitH() override = default;
   void Reset(gert::TilingContext* context) override { MoeInitRoutingV2GradTilingBaseClass::Reset(context); Reset(); }
  protected:
   bool IsCapable() override {
     this->typeSize = (this->inDtype != ge::DT_FLOAT) ? FP16_BF16_SIZE : sizeof(float);
     // topk小于64，N小于coreNum, 并且H / 512大于N的场景，走到这个性能模版
     return (topK <= MIN_SPLIT_H_TOPK_THRESHOLD) && (N < aivNum) &&
            (N < (hiddenSize / MAX_CACHE_LINE_SIZE / this->typeSize));
   }
   // 3、计算数据切分TilingData
   ge::graphStatus DoOpTiling() override;
   // 5、计算TilingKey
   uint64_t GetTilingKey() const override;
   // 7、保存Tiling数据
   ge::graphStatus PostTiling() override;
   void Reset();
  private:
   void SetTilingShapeInfo();
   void SetTilingSplitCore();
   void SetTilingFactor();
   int64_t blockDim = 0;
   int64_t hBlockFactor = 0;
   int64_t typeSize = 0;
   int64_t blockSize = GetUbBlockSize(context_);
   MoeInitRoutingV2GradRegbaseSplitHTilingData moeInitRoutingV2GradTilingData;
 };
 void MoeInitRoutingV2GradRegbaseSplitH::Reset() { opName = nullptr; }
 ge::graphStatus MoeInitRoutingV2GradRegbaseSplitH::DoOpTiling() {
   SetTilingShapeInfo();
   SetTilingSplitCore();
   SetTilingFactor();
   return ge::GRAPH_SUCCESS;
 }
 void MoeInitRoutingV2GradRegbaseSplitH::SetTilingShapeInfo() {
   moeInitRoutingV2GradTilingData.set_h(hiddenSize);
   moeInitRoutingV2GradTilingData.set_n(N);
   moeInitRoutingV2GradTilingData.set_k(topK);
   moeInitRoutingV2GradTilingData.set_activeNum(activeNum);
 }
 void MoeInitRoutingV2GradRegbaseSplitH::SetTilingSplitCore() {
   this->hBlockFactor = CeilDiv(hiddenSize, aivNum);
   blockDim = CeilDiv(hiddenSize, this->hBlockFactor);  // 实际使用核数
   moeInitRoutingV2GradTilingData.set_blockDim(blockDim);
   moeInitRoutingV2GradTilingData.set_hBlockFactor(this->hBlockFactor);  // 单核处理的Token序列长度
 }
 void MoeInitRoutingV2GradRegbaseSplitH::SetTilingFactor() {
   int64_t ubSizeCanUse = aicoreParams_.ubSize;
   // 性能模版，topK小于64，优先切入topK，再切入subH
   int64_t rowsSizeForOneToken = (topK * DOUBLE_BUFFER + 1) * this->typeSize;
   auto hBlockAlign = AlignUp(this->hBlockFactor * this->typeSize, blockSize) / this->typeSize;
   bool canSubHSizeFullLoad = (rowsSizeForOneToken * hBlockAlign) <= ubSizeCanUse;
   if (canSubHSizeFullLoad) {
     // subH可以全载，ub内一共可以放nUbFactor个token
     int64_t tokensCanInUb = ubSizeCanUse / (rowsSizeForOneToken * hBlockAlign);
     moeInitRoutingV2GradTilingData.set_kUbFactor(tokensCanInUb);       //  可以放入topK 的个数
     moeInitRoutingV2GradTilingData.set_hUbFactor(this->hBlockFactor);  // 一次搬入subH 的大小
   } else {
     // subH无法全部放入，一次只能放入hUbFactor大小
     int64_t tokenSubHSizeCanInUb = ubSizeCanUse / rowsSizeForOneToken;
     tokenSubHSizeCanInUb = AlignDown(tokenSubHSizeCanInUb * this->typeSize, blockSize) / this->typeSize;
     moeInitRoutingV2GradTilingData.set_kUbFactor(1);
     moeInitRoutingV2GradTilingData.set_hUbFactor(tokenSubHSizeCanInUb);
   }
 }
 uint64_t MoeInitRoutingV2GradRegbaseSplitH::GetTilingKey() const {
   if (dropPadMode == 1) { return TILINGKEY_SPLIT_H_DROP_PAD; }
   else if (activeNum > 0) { return TILINGKEY_SPLIT_H_ACTIVE; }
   return TILINGKEY_SPLIT_H_DROPLESS;
 }
 ge::graphStatus MoeInitRoutingV2GradRegbaseSplitH::PostTiling() {
   context_->SetBlockDim(blockDim);
   size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
   OPS_LOG_E_IF_NULL(context_, currentWorkspace, return ge::GRAPH_FAILED);
   currentWorkspace[0] = workspaceSize_;
   auto tilingData = context_->GetRawTilingData();
   OPS_LOG_E_IF_NULL(context_, tilingData, return ge::GRAPH_FAILED);
   moeInitRoutingV2GradTilingData.SaveToBuffer(tilingData->GetData(), tilingData->GetCapacity());
   tilingData->SetDataSize(moeInitRoutingV2GradTilingData.GetDataSize());
   return ge::GRAPH_SUCCESS;
 }
 REGISTER_TILING_TEMPLATE("MoeInitRoutingV2Grad", MoeInitRoutingV2GradRegbaseSplitH, 20000);
 }  // namespace optiling