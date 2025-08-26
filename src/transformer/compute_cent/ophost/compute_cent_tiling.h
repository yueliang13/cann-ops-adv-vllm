#include <tiling/tiling_api.h>
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ComputeCentTilingData)
    TILING_DATA_FIELD_DEF(int64_t, bSize);  // Batch size
    TILING_DATA_FIELD_DEF(int64_t, n1Size); // Number of qheads
    TILING_DATA_FIELD_DEF(int64_t, n2Size); // Number of kvheads
    TILING_DATA_FIELD_DEF(int64_t, gSize);  // Number of groups
    TILING_DATA_FIELD_DEF(int64_t, s1Size);  // Size of the  sequence dimension of query
    TILING_DATA_FIELD_DEF(int64_t, dSize);  // Dimension size of query and KV
    TILING_DATA_FIELD_DEF(int64_t, cSize);  // Size of the cluster dimension
    TILING_DATA_FIELD_DEF(int64_t, seqLen);  // Size of the sequence dimension of key_ids
    TILING_DATA_FIELD_DEF(int64_t, blockSize);  
    TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);  
    //TopK:
    TILING_DATA_FIELD_DEF(int32_t, k);
    TILING_DATA_FIELD_DEF(uint32_t, tmpsize);
    TILING_DATA_FIELD_DEF(int32_t, outter);
    TILING_DATA_FIELD_DEF(int32_t, inner);
    TILING_DATA_FIELD_DEF(int32_t, n);
    TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, topkTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ComputeCent, ComputeCentTilingData)
}
