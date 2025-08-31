#include <tiling/tiling_api.h>
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SelectPositionTilingData)
    TILING_DATA_FIELD_DEF(int64_t, bSize);  // Batch size
    TILING_DATA_FIELD_DEF(int64_t, n1Size); // Number of qheads
    TILING_DATA_FIELD_DEF(int64_t, kvPageLen);  // Size of the sequence dimension of block_ids
    TILING_DATA_FIELD_DEF(int64_t, maxBatch);  // Size of the sequence dimension of block_table
    TILING_DATA_FIELD_DEF(int64_t, maxPage);  // Size of the sequence dimension of block_table
    TILING_DATA_FIELD_DEF(int64_t, maxPageNum);  // Size of the sequence dimension of page_position
    TILING_DATA_FIELD_DEF(int32_t, k);
    TILING_DATA_FIELD_DEF(int32_t, blockSize);
    TILING_DATA_FIELD_DEF(int32_t, usedCoreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SelectPosition, SelectPositionTilingData)
}
