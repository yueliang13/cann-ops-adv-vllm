#include <tiling/tiling_api.h>
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SelectPositionTilingData)
    TILING_DATA_FIELD_DEF(int64_t, bSize);  // Batch size
    TILING_DATA_FIELD_DEF(int64_t, n1Size); // Number of qheads
    TILING_DATA_FIELD_DEF(int64_t, seqLen);  // Size of the sequence dimension of key_ids
    TILING_DATA_FIELD_DEF(int64_t, splitSeqLen);  // Size of the split sequence dimension of key_ids
    TILING_DATA_FIELD_DEF(int64_t, splitSeqNum);  // Number of the split sequence dimension of key_ids
    TILING_DATA_FIELD_DEF(int64_t, splitSeqRemainLen);  // Number of the split sequence dimension of key_ids
    TILING_DATA_FIELD_DEF(int64_t, maxTokenNum);  // Size of the sequence dimension of token_position
    TILING_DATA_FIELD_DEF(int32_t, k);
    TILING_DATA_FIELD_DEF(int32_t, blockSize);
    TILING_DATA_FIELD_DEF(int32_t, usedCoreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SelectPosition, SelectPositionTilingData)
}
