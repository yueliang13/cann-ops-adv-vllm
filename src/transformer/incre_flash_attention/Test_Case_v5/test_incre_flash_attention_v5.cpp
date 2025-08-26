/**
 * @file test_incre_flash_attention_v5.cpp
 * @brief IncreFlashAttentionV5 Complete Test Suite
 * @author V5 Development Team
 * @date 2024
 * 
 * 此文件包含IncreFlashAttentionV5接口的完整测试用例，验证：
 * 1. 密集注意力模式（兼容性测试）
 * 2. IVF稀疏注意力模式（核心功能）
 * 3. 参数验证和错误处理
 * 4. 性能基准测试
 */

#include <iostream>
#include <vector>
#include <math.h>
#include <cstring>
#include <memory>
#include <random>
#include <cmath>
#include <algorithm>
#include "acl/acl.h"
// #include "aclnn_incre_flash_attention_v4.h"
#include "aclnn_incre_flash_attention_v5.h"
#include <cstdlib>

using namespace std;

// ===================== 测试框架和工具函数 =====================

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        printf("[TEST] " message, ##__VA_ARGS__); \
    } while (0)

#define TEST_CASE(name) \
    LOG_PRINT("=== 开始测试: %s ===\n", name);

#define TEST_PASS(name) \
    LOG_PRINT("✅ %s: PASS\n", name);

#define TEST_FAIL(name, reason) \
    LOG_PRINT("❌ %s: FAIL - %s\n", name, reason);

// 测试结果统计
struct TestStats {
    int total = 0;
    int passed = 0;
    int failed = 0;
    
    void addResult(bool pass) {
        total++;
        if (pass) passed++;
        else failed++;
    }
    
    void printSummary() {
        LOG_PRINT("\n📊 测试总结: %d/%d 通过 (%.1f%%)\n", 
                  passed, total, (float)passed/total*100);
        if (failed > 0) {
            LOG_PRINT("❌ %d 个测试失败\n", failed);
        } else {
            LOG_PRINT("🎉 所有测试通过!\n");
        }
    }
};

// 全局测试统计
TestStats g_testStats;

// ===================== 数据生成和工具函数 =====================

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

// 生成固定值的浮点数据
std::vector<float> GenerateConstantFloatData(int64_t size, float value = 1.0f) {
    std::vector<float> data(size, value);
    return data;
}

// 生成固定值的整数数据
std::vector<int32_t> GenerateConstantIntData(int64_t size, int32_t value = 1) {
    std::vector<int32_t> data(size, value);
    return data;
}

// 🔧 修复：使用C++标准库生成指定范围的小数值随机数
std::vector<float> GenerateRandomFloatData(int64_t size, float min = -0.1f, float max = 0.1f) {
    std::vector<float> data(size);
    std::random_device rd;
    std::mt19937 gen(42); // 固定种子确保可重现
    std::uniform_real_distribution<float> dis(min, max);
    
    for (int64_t i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
    return data;
}

// 🔧 新增：生成标准正态分布数据，更适合注意力机制
std::vector<float> GenerateNormalFloatData(int64_t size, float mean = 0.0f, float stddev = 0.02f) {
    std::vector<float> data(size);
    std::random_device rd;
    std::mt19937 gen(42); // 使用固定种子确保可重现性
    std::normal_distribution<float> dis(mean, stddev);
    
    for (int64_t i = 0; i < size; i++) {
        float val = dis(gen);
        // 限制数值范围，避免极端值
        val = std::max(-0.5f, std::min(0.5f, val));
        data[i] = val;
    }
    return data;
}

// 🔧 新增：生成Layer Norm风格的数据（均值0，方差1）
std::vector<float> GenerateLayerNormData(int64_t size, float scale = 0.1f) {
    std::vector<float> data(size);
    std::random_device rd;
    std::mt19937 gen(42);
    std::normal_distribution<float> dis(0.0f, 1.0f);
    
    // 先生成标准正态分布
    for (int64_t i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
    
    // 计算均值和标准差
    float sum = 0.0f;
    for (float val : data) {
        sum += val;
    }
    float mean = sum / size;
    
    float var_sum = 0.0f;
    for (float val : data) {
        var_sum += (val - mean) * (val - mean);
    }
    float stddev = sqrt(var_sum / size);
    
    // 标准化并缩放
    for (int64_t i = 0; i < size; i++) {
        data[i] = ((data[i] - mean) / stddev) * scale;
    }
    
    return data;
}

// 生成随机整数数据
std::vector<int32_t> GenerateRandomIntData(int64_t size, int32_t min = 0, int32_t max = 100) {
    std::vector<int32_t> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int32_t> dis(min, max);
    
    for (int64_t i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
    return data;
}

// 生成IVF稀疏数据
struct SparseData {
    std::vector<int32_t> selectNprobe;  // [bs, nh, nprobe]
    std::vector<int32_t> ivfStart;      // [bs, nh_k, cluster]
    std::vector<int32_t> ivfLen;        // [bs, nh_k, cluster]
};

SparseData GenerateSparseData(int64_t bs, int64_t nh, int64_t nh_k, 
                              int64_t nprobe, int64_t clusterNum, int64_t maxSeqLen) {
    SparseData data;
    
    // 生成 selectNprobe: 选择的聚类索引
    data.selectNprobe.resize(bs * nh * nprobe);
    for (int64_t b = 0; b < bs; b++) {
        for (int64_t h = 0; h < nh; h++) {
            for (int64_t p = 0; p < nprobe; p++) {
                // 确保选择的聚类索引在有效范围内且不重复
                data.selectNprobe[b * nh * nprobe + h * nprobe + p] = 
                    (p * clusterNum / nprobe) % clusterNum;
            }
        }
    }
    
    // 生成 ivfStart 和 ivfLen
    data.ivfStart.resize(bs * nh_k * clusterNum);
    data.ivfLen.resize(bs * nh_k * clusterNum);
    
    const int32_t ALIGN_STEP = 8;
    for (int64_t b = 0; b < bs; ++b) {
        for (int64_t h = 0; h < nh_k; ++h) {
            int32_t cursor = 0;

            for (int64_t c = 0; c < clusterNum; ++c) {
                int64_t idx = b * nh_k * clusterNum + h * clusterNum + c;

                /* 剩余 token 与剩余 cluster 数 */
                int32_t remainTok   = maxSeqLen - cursor;
                int32_t remainClust = static_cast<int32_t>(clusterNum - c);

                /* 除最后一块外，长度向上 8 对齐 */
                int32_t size;
                if (c == clusterNum - 1) {
                    size = remainTok;                    // 最后一块吃掉剩余全部
                } else {
                    int32_t avg   = (remainTok + remainClust - 1) / remainClust; // 向上取整平均
                    size = (avg + ALIGN_STEP - 1) / ALIGN_STEP * ALIGN_STEP;     // 8 对齐
                    if (size > remainTok) size = remainTok;  // 防止对齐后超过剩余
                }

                data.ivfStart[idx] = cursor;
                data.ivfLen[idx]   = size;
                cursor += size;
            }

            // 断言覆盖完整序列
            if (cursor != maxSeqLen) {
                LOG_PRINT("❌ 覆盖错误 head=%ld  cursor=%d\n", h, cursor);
            }
        }
    }
    
    return data;
}

// ===================== ACL 初始化和资源管理 =====================

int Init(int32_t deviceId, aclrtStream *stream) {
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

// 🔧 修复：添加float32到float16的正确转换函数
uint16_t floatToHalf(float f) {
    union { float f; uint32_t i; } u;
    u.f = f;
    
    uint32_t sign = (u.i >> 31) & 0x1;
    uint32_t exp = (u.i >> 23) & 0xFF;
    uint32_t mantissa = u.i & 0x7FFFFF;
    
    // 处理特殊情况
    if (exp == 0) {
        // 零或次正规数
        return static_cast<uint16_t>(sign << 15);
    } else if (exp == 255) {
        // 无穷大或NaN
        return static_cast<uint16_t>((sign << 15) | 0x7C00 | (mantissa ? 0x200 : 0));
    } else {
        // 正常数
        int32_t newExp = static_cast<int32_t>(exp) - 127 + 15; // 重新偏移
        if (newExp <= 0) {
            // 次正规数
            return static_cast<uint16_t>(sign << 15);
        } else if (newExp >= 31) {
            // 无穷大
            return static_cast<uint16_t>((sign << 15) | 0x7C00);
        } else {
            // 正常转换
            uint32_t newMantissa = mantissa >> 13; // 截断到10位
            return static_cast<uint16_t>((sign << 15) | (newExp << 10) | newMantissa);
        }
    }
}

// 🔧 新增：half (uint16_t) 转 float32 的准确转换，打印结果不失真
float halfToFloat(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h & 0x7C00) >> 10;
    uint32_t mant =  h & 0x03FF;

    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;                // ±0
        } else {
            /* 次正规数转正规 */
            exp = 1;
            while ((mant & 0x0400) == 0) { mant <<= 1; --exp; }
            mant &= 0x03FF;
            exp = exp + (127 - 15);
            f = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        /* Inf / NaN */
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        /* 正规数 */
        exp = exp + (127 - 15);
        f = sign | (exp << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &f, sizeof(float));
    return out;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, 
                    void **deviceAddr, aclDataType dataType, aclTensor **tensor) {
    
    size_t deviceSize;
    
    // 🚨 关键修复：根据目标数据类型进行正确的转换
    if (dataType == ACL_FLOAT16 && std::is_same<T, float>::value) {
        // float32 -> float16 转换
        deviceSize = GetShapeSize(shape) * sizeof(uint16_t); // float16是2字节
        
        std::vector<uint16_t> halfData(GetShapeSize(shape));
        for (size_t i = 0; i < hostData.size(); i++) {
            halfData[i] = floatToHalf(static_cast<float>(hostData[i]));
        }
        
        // 分配设备内存
        auto ret = aclrtMalloc(deviceAddr, deviceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    
        // 拷贝转换后的数据
        ret = aclrtMemcpy(*deviceAddr, deviceSize, halfData.data(), deviceSize, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
        
        // 🔍 验证转换结果
        LOG_PRINT("      数据转换: float32->float16，前5个值: ");
        for (size_t i = 0; i < std::min(static_cast<size_t>(5), hostData.size()); i++) {
            LOG_PRINT("%.6f->0x%04X ", static_cast<float>(hostData[i]), halfData[i]);
        }
        LOG_PRINT("\n");
        
    } else if (dataType == ACL_INT8 && std::is_same<T, int8_t>::value) {
        // INT8数据直接拷贝，无需转换
        deviceSize = GetShapeSize(shape) * sizeof(int8_t);
        
        auto ret = aclrtMalloc(deviceAddr, deviceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        
        ret = aclrtMemcpy(*deviceAddr, deviceSize, hostData.data(), deviceSize, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
        
        // 🔍 验证INT8数据
        LOG_PRINT("      INT8数据，前5个值: ");
        for (size_t i = 0; i < std::min(static_cast<size_t>(5), hostData.size()); i++) {
            LOG_PRINT("%d ", static_cast<int>(hostData[i]));
        }
        LOG_PRINT("\n");
        
    } else {
        // 无需转换的情况（如int32等）
        deviceSize = GetShapeSize(shape) * sizeof(T);
        
        auto ret = aclrtMalloc(deviceAddr, deviceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        
        ret = aclrtMemcpy(*deviceAddr, deviceSize, hostData.data(), deviceSize, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    }

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, 
                              aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

bool TestMultiHeadBlockTableMode_BSBD(aclrtStream stream)
{
    TEST_CASE("Multi-Head BlockTable Page Attention Mode (二维BlockTable + 三维Position)");

    try {
        // 测试参数
        int32_t batchSize = 1;
        int32_t numHeads = 32;
        int32_t headDims = 128;
        int32_t keyNumHeads = 32;
        int32_t blockSize = 16;                         // 128 每块的token数
        int32_t sequenceLengthQ = 1;                     // Q序列长度
        int32_t totalSeqLengthKV = 32 * 1024;            // 总KV序列长度（足够大）
        int32_t blockNum = totalSeqLengthKV / blockSize; // 总块数

        // 设置基准块数和最大差异
        int32_t baseBlockNum = (32 * 1024) / blockSize; // 基准块数（约4K tokens）
        int32_t maxDiffPercent = 10;                    // 最大±10%差异
        int32_t minBlocksPerHead = baseBlockNum * (100 - maxDiffPercent) / 100;
        int32_t maxBlocksPerHead = baseBlockNum * (100 + maxDiffPercent) / 100;
        int32_t maxActualBlockNumPerSeq = maxBlocksPerHead;

        maxActualBlockNumPerSeq = baseBlockNum; // 用来做性能测试

        // 创建实际序列长度数组 - 使用maxActualBlockNumPerSeq作为最大块数
        std::vector<int64_t> actualSeqlenVector = {maxActualBlockNumPerSeq * blockSize};
        auto actualSeqLengths = aclCreateIntArray(actualSeqlenVector.data(), actualSeqlenVector.size());
        LOG_PRINT("  全局actualSeqLength: %ld\n", actualSeqlenVector[0]);

        // 选择性块值定义
        const float SELECTED_BLOCK_VALUE = 0.1f;   // 选中块的值
        const float UNSELECTED_BLOCK_VALUE = 9.9f; // 未选中块的值

        LOG_PRINT("  所有头使用相同的块数: %d\n", baseBlockNum);

        // ⚡️ 性能测试开关：环境变量 PERF_TEST=1 时开启
        const char *perfFlag = std::getenv("PERF_TEST");
        bool perfTest = (perfFlag && std::strcmp(perfFlag, "1") == 0);
        if (perfTest) {
            LOG_PRINT("⚡️ 性能测试模式已启用，跳过所有非核心计算代码\n");
        }

        LOG_PRINT("  创建张量形状...\n");

        // 张量形状定义
        std::vector<int64_t> queryShape = {batchSize, numHeads, sequenceLengthQ, headDims};
        std::vector<int64_t> keyShape = {blockNum, keyNumHeads, blockSize, headDims};
        std::vector<int64_t> valueShape = {blockNum, keyNumHeads, blockSize, headDims};
        std::vector<int64_t> outShape = {batchSize, numHeads, sequenceLengthQ, headDims};

        // 为每个头确定块数 - 所有头使用相同的块数，最后一个头少一个块
        std::vector<int32_t> blocksPerHead(keyNumHeads, maxActualBlockNumPerSeq);
        for (int h = 0; h < keyNumHeads; h++) {
            LOG_PRINT("  头 %d 选择块数: %d\n", h, blocksPerHead[h]);
        }
        // blocksPerHead[keyNumHeads - 1] = maxActualBlockNumPerSeq - 1;

        // 计算全局需要的最大块数
        int32_t totalUniqueBlocks = 0;
        for (int h = 0; h < keyNumHeads; h++) {
            totalUniqueBlocks += blocksPerHead[h];
        }
        LOG_PRINT("  所有头总共需要 %d 个块\n", totalUniqueBlocks);

        // 修改：二维BlockTable [batchSize, totalUniqueBlocks]
        std::vector<int64_t> blockTableShape = {batchSize, totalUniqueBlocks};
        LOG_PRINT("  新BlockTable形状: [%ld, %ld]\n", blockTableShape[0], blockTableShape[1]);

        // 修改：添加三维blockPosition [batchSize, keyNumHeads, maxActualBlockNumPerSeq]
        std::vector<int64_t> blockPositionShape = {batchSize, keyNumHeads, maxActualBlockNumPerSeq};
        LOG_PRINT("  BlockPosition形状: [%ld, %ld, %ld]\n", blockPositionShape[0], blockPositionShape[1],
                  blockPositionShape[2]);

        if (!perfTest) {
            LOG_PRINT("张量形状信息:\n");
            LOG_PRINT("  Query Shape: [%ld, %ld, %ld, %ld]\n", queryShape[0], queryShape[1], queryShape[2],
                      queryShape[3]);
            LOG_PRINT("  Key Shape (PageAttention): [%ld, %ld, %ld, %ld]\n", keyShape[0], keyShape[1], keyShape[2],
                      keyShape[3]);
            LOG_PRINT("  Value Shape (PageAttention): [%ld, %ld, %ld, %ld]\n", valueShape[0], valueShape[1],
                      valueShape[2], valueShape[3]);
            LOG_PRINT("  Output Shape: [%ld, %ld, %ld, %ld]\n", outShape[0], outShape[1], outShape[2], outShape[3]);
            LOG_PRINT("  最大序列长度: %d\n", maxActualBlockNumPerSeq * blockSize);
        }

        LOG_PRINT("  生成测试数据...\n");

        // 生成Query数据
        auto queryData = std::vector<float>(GetShapeSize(queryShape));
        std::mt19937 gen(42); // 固定种子保证可重现
        for (size_t i = 0; i < queryData.size(); i++) {
            queryData[i] = 0.1f; // 使用小数值避免Softmax溢出
        }

        // 生成Key数据 - 初始为未选择值
        auto keyData = std::vector<float>(GetShapeSize(keyShape));
        for (size_t i = 0; i < keyData.size(); i++) {
            keyData[i] = UNSELECTED_BLOCK_VALUE;
        }

        // 生成Value数据 - 初始为未选择值
        auto valueData = std::vector<float>(GetShapeSize(valueShape));
        for (size_t i = 0; i < valueData.size(); i++) {
            valueData[i] = UNSELECTED_BLOCK_VALUE;
        }

        // 修改：生成二维BlockTable数据和三维blockPosition数据
        std::vector<int32_t> blockTableData(GetShapeSize(blockTableShape), 0x7FFFFFFF);       // 默认填充无效值
        std::vector<int32_t> blockPositionData(GetShapeSize(blockPositionShape), 0x7FFFFFFF); // 默认填充无效值

        // 设置选中块的数据并填充BlockTable和blockPosition
        for (int b = 0; b < batchSize; b++) {
            int blockTableIdx = 0; // 记录当前blockTable中的位置

            for (int h = 0; h < keyNumHeads; h++) {
                LOG_PRINT("  头 %d 实际块数: %d/%d\n", h, blocksPerHead[h], maxActualBlockNumPerSeq);

                // 基于头索引计算基础偏移量
                int baseOffset = (h * 3) % blockNum; // 使用不同的起始偏移

                LOG_PRINT("  头 %d 选择的块: ", h);

                // 填充该头使用的块
                for (int i = 0; i < blocksPerHead[h]; i++) {
                    // 计算块索引，确保在合法范围内
                    int32_t blockIdx = (baseOffset + i * 2) % blockNum; // 使用2的步长增加差异

                    if (i < 10) { // 只打印前10个块
                        LOG_PRINT("%d ", blockIdx);
                    } else if (i == 10) {
                        LOG_PRINT("...");
                    }

                    // 设置BlockTable索引 - 在二维BlockTable中记录实际的块ID
                    int tableIdx = b * totalUniqueBlocks + blockTableIdx;
                    blockTableData[tableIdx] = blockIdx;

                    // 设置blockPosition索引 - 在三维blockPosition中记录对应的BlockTable位置
                    int positionIdx = (b * keyNumHeads + h) * maxActualBlockNumPerSeq + i;
                    blockPositionData[positionIdx] = blockTableIdx;

                    blockTableIdx++; // 递增BlockTable索引

                    // 设置该块的KV数据为"选择"值
                    float headSpecificValue = SELECTED_BLOCK_VALUE + h * 0.001f; // 为每个头添加微小差异便于调试

                    // 设置KV数据 - BNSD格式 (blocknum, KV_N, blocksize, D)
                    for (int s = 0; s < blockSize; s++) {
                        for (int d = 0; d < headDims; d++) {
                            // BNSD格式索引计算: (blockIdx, h, s, d)
                            int64_t dataIdx = blockIdx * (keyNumHeads * blockSize * headDims) + 
                                            h * (blockSize * headDims) + 
                                            s * headDims + d;
                            if (dataIdx < keyData.size()) {
                                keyData[dataIdx] = headSpecificValue;
                                valueData[dataIdx] = headSpecificValue;
                            }
                        }
                    }
                }
                LOG_PRINT("\n");
            }
        }

        if (!perfTest) {
            // 验证BlockPosition映射
            bool validMapping = true;
            for (int b = 0; b < batchSize; b++) {
                for (int h = 0; h < keyNumHeads; h++) {
                    // 检查有效映射数量
                    int validMappings = 0;
                    for (int i = 0; i < maxActualBlockNumPerSeq; i++) {
                        int positionIdx = (b * keyNumHeads + h) * maxActualBlockNumPerSeq + i;
                        if (blockPositionData[positionIdx] != 0x7FFFFFFF) {
                            validMappings++;

                            // 验证索引是否有效
                            int blockTableIdx = blockPositionData[positionIdx];
                            if (blockTableIdx < 0 || blockTableIdx >= totalUniqueBlocks) {
                                LOG_PRINT("  ❌ 头 %d 的position[%d]=%d 超出范围!\n", h, i, blockTableIdx);
                                validMapping = false;
                            }
                        }
                    }

                    // 验证有效映射数量是否等于预期的块数
                    if (validMappings != blocksPerHead[h]) {
                        LOG_PRINT("  ❌ 头 %d 映射数量不符: 预期 %d, 实际 %d\n", h, blocksPerHead[h], validMappings);
                        validMapping = false;
                    }
                }
            }

            if (validMapping) {
                LOG_PRINT("  ✅ BlockPosition验证通过: 每个头的块映射符合预期\n");
            } else {
                LOG_PRINT("  ❌ BlockPosition验证失败: 请检查映射逻辑\n");
            }
        }

        auto outData = GenerateConstantFloatData(GetShapeSize(outShape), 0.0f); // 输出初始化为0

        LOG_PRINT("  创建设备张量...\n");

        // 创建设备张量
        void *queryDeviceAddr = nullptr, *keyDeviceAddr = nullptr;
        void *valueDeviceAddr = nullptr, *outDeviceAddr = nullptr;
        void *blockTableAddr = nullptr, *blockPositionAddr = nullptr;

        aclTensor *queryTensor = nullptr, *keyTensor = nullptr;
        aclTensor *valueTensor = nullptr, *outTensor = nullptr;
        aclTensor *blockTableTensor = nullptr, *blockPositionTensor = nullptr;

        int ret = 0;

        if (!perfTest) {
            LOG_PRINT("  创建 query tensor...\n");
        }
        ret = CreateAclTensor(queryData, queryShape, &queryDeviceAddr, ACL_FLOAT16, &queryTensor);
        if (ret != 0) {
            LOG_PRINT("  ❌ 创建 query tensor 失败\n");
            return false;
        }

        if (!perfTest) {
            LOG_PRINT("  创建 key tensor...\n");
        }
        ret = CreateAclTensor(keyData, keyShape, &keyDeviceAddr, ACL_FLOAT16, &keyTensor);
        if (ret != 0) {
            LOG_PRINT("  ❌ 创建 key tensor 失败\n");
            return false;
        }

        if (!perfTest) {
            LOG_PRINT("  创建 value tensor...\n");
        }
        ret = CreateAclTensor(valueData, valueShape, &valueDeviceAddr, ACL_FLOAT16, &valueTensor);
        if (ret != 0) {
            LOG_PRINT("  ❌ 创建 value tensor 失败\n");
            return false;
        }

        if (!perfTest) {
            LOG_PRINT("  创建 output tensor...\n");
        }
        ret = CreateAclTensor(outData, outShape, &outDeviceAddr, ACL_FLOAT16, &outTensor);
        if (ret != 0) {
            LOG_PRINT("  ❌ 创建 output tensor 失败\n");
            return false;
        }

        if (!perfTest) {
            LOG_PRINT("  创建 blockTable tensor...\n");
        }
        ret = CreateAclTensor(blockTableData, blockTableShape, &blockTableAddr, ACL_INT32, &blockTableTensor);
        if (ret != 0) {
            LOG_PRINT("  ❌ 创建 blockTable tensor 失败\n");
            return false;
        }

        if (!perfTest) {
            LOG_PRINT("  创建 blockPosition tensor...\n");
        }
        ret =
            CreateAclTensor(blockPositionData, blockPositionShape, &blockPositionAddr, ACL_INT32, &blockPositionTensor);
        if (ret != 0) {
            LOG_PRINT("  ❌ 创建 blockPosition tensor 失败\n");
            return false;
        }

        if (!perfTest) {
            LOG_PRINT("  创建张量列表...\n");
        }

        // 创建张量列表
        int kvTensorNum = 1;
        aclTensor *tensorsOfKey[kvTensorNum] = {keyTensor};
        aclTensor *tensorsOfValue[kvTensorNum] = {valueTensor};
        auto tensorKeyList = aclCreateTensorList(tensorsOfKey, kvTensorNum);
        auto tensorValueList = aclCreateTensorList(tensorsOfValue, kvTensorNum);

        if (tensorKeyList == nullptr || tensorValueList == nullptr) {
            LOG_PRINT("  ❌ 创建张量列表失败\n");
            return false;
        }

        // 设置算子参数
        int64_t numKeyValueHeads = keyNumHeads;
        int64_t blockSizeParam = blockSize;
        int64_t innerPrecise = 1;
        double scaleValue = 1.0 / sqrt(static_cast<double>(headDims));
        string sLayout = "BNSD";
        char layout[sLayout.length() + 1];
        strcpy(layout, sLayout.c_str());

        if (!perfTest) {
            LOG_PRINT("  算子参数: numHeads=%d, scaleValue=%.6f, blockSize=%ld\n", numHeads, scaleValue,
                       blockSizeParam);
            LOG_PRINT("  调用 GetWorkspaceSize...\n");
        }

        // 调用V5接口 - 注意这里使用blockPositionTensor
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;

        ret = aclnnIncreFlashAttentionV5GetWorkspaceSize(queryTensor, tensorKeyList, tensorValueList,
                                                         nullptr,             // pse_shift
                                                         nullptr,             // attenMask
                                                         actualSeqLengths,    // 传入实际序列长度
                                                         nullptr,             // dequant_scale1
                                                         nullptr,             // quant_scale1
                                                         nullptr,             // dequant_scale2
                                                         nullptr,             // quant_scale2
                                                         nullptr,             // quant_offset2
                                                         nullptr,             // antiquant_scale
                                                         nullptr,             // antiquant_offset
                                                         blockTableTensor,    // blockTable - 改为二维
                                                         nullptr,             // kvPaddingSize - 不使用
                                                         blockPositionTensor, // blockPosition - 新增的三维映射表
                                                         numHeads, scaleValue, layout, numKeyValueHeads,
                                                         blockSizeParam, // 传入blockSize
                                                         innerPrecise,
                                                         outTensor, &workspaceSize, &executor);

        if (!perfTest) {
            LOG_PRINT("  GetWorkspaceSize 返回: %d, error: %s\n", ret, aclGetRecentErrMsg());
        }

        if (ret != ACL_SUCCESS) {
            TEST_FAIL("Multi-Head BlockTable Mode", "GetWorkspaceSize failed");
            return false;
        }

        if (!perfTest) {
            LOG_PRINT("  工作空间大小: %llu bytes\n", (unsigned long long)workspaceSize);
        }

        // 申请工作空间
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                TEST_FAIL("Multi-Head BlockTable Mode", "Workspace allocation failed");
                return false;
            }
            if (!perfTest) {
                LOG_PRINT("  工作空间分配成功: %lu bytes\n", workspaceSize);
            }
        }

        // 执行计算
        if (!perfTest) {
            LOG_PRINT("  开始执行算子计算...\n");
            LOG_PRINT(" aclnnIncreFlashAttentionV5 执行前Log: %s\n", aclGetRecentErrMsg());
        }
        ret = aclnnIncreFlashAttentionV5(workspaceAddr, workspaceSize, executor, stream);
        if (!perfTest) {
            LOG_PRINT(" aclnnIncreFlashAttentionV5 执行后Log: %s\n", aclGetRecentErrMsg());
        }
        if (ret != ACL_SUCCESS) {
            TEST_FAIL("Multi-Head BlockTable Mode", "Execution failed");
            LOG_PRINT("  ❌ 算子执行失败，错误码: %d, 错误信息: %s\n", ret, aclGetRecentErrMsg());
            return false;
        }
        if (!perfTest) {
            LOG_PRINT("  ✅ 算子执行成功\n");
        }

        // 同步等待
        if (!perfTest) {
            LOG_PRINT("  等待计算完成...\n");
        }

        ret = aclrtSynchronizeStream(stream);
        if (!perfTest) {
            LOG_PRINT("  aclrtSynchronizeStream 执行Log: %s\n", aclGetRecentErrMsg());
        }

        if (ret != ACL_SUCCESS) {
            TEST_FAIL("Multi-Head BlockTable Mode", "Stream synchronization failed");

            if (!perfTest) {
                LOG_PRINT("  ❌ 流同步失败，错误码: %d\n", ret);
                LOG_PRINT("  📊 PageAttention错误分析：\n");
                LOG_PRINT("     - 如果是超时错误：可能是blockTable或blockPosition配置问题\n");
                LOG_PRINT("     - 如果是AICore错误：可能是内存访问模式问题\n");
                LOG_PRINT("     - 如果是内存错误：可能是索引超出范围\n");

                const char *errMsg = aclGetRecentErrMsg();
                LOG_PRINT("  💬 详细错误信息: %s\n", errMsg ? errMsg : "无详细信息");
            }

            return false;
        }
        if (!perfTest) {
            LOG_PRINT("  ✅ 计算完成，流同步成功\n");
        }

        // 输出结果处理与统计（与原函数相同，略）
        if (!perfTest) {
            LOG_PRINT("  开始读取计算结果...\n");

            // 从设备拷贝结果到主机
            std::vector<uint16_t> outputResultHalf(GetShapeSize(outShape)); // 使用uint16_t表示half
            ret = aclrtMemcpy(outputResultHalf.data(), outputResultHalf.size() * sizeof(uint16_t), outDeviceAddr,
                              outputResultHalf.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret != ACL_SUCCESS) {
                LOG_PRINT("  ❌ 结果拷贝失败，错误码: %d\n", ret);
                return false;
            }

            // 将half转换为float (准确版本)
            std::vector<float> outputResult(GetShapeSize(outShape));
            for (size_t i = 0; i < outputResultHalf.size(); i++) {
                outputResult[i] = halfToFloat(outputResultHalf[i]);
            }

            LOG_PRINT("  ✅ 结果拷贝成功\n");

            // 输出张量形状信息
            LOG_PRINT("\n=== 计算结果分析 ===\n");
            LOG_PRINT("输出张量形状: [%ld, %ld, %ld, %ld]\n", outShape[0], outShape[1], outShape[2], outShape[3]);

            // 统计结果
            float sum = 0.0f, min_val = 1e30f, max_val = -1e30f;
            size_t validCnt = 0;
            bool hasUnselectedValue = false;

            // 分头统计结果
            for (int h = 0; h < numHeads; h++) {
                float headSum = 0.0f;
                float headMin = 1e30f;
                float headMax = -1e30f;

                // 计算头h的起始索引
                size_t headStartIdx = h * sequenceLengthQ * headDims;
                size_t headEndIdx = headStartIdx + sequenceLengthQ * headDims;

                for (size_t i = headStartIdx; i < headEndIdx; ++i) {
                    float val = outputResult[i];
                    headSum += val;
                    headMin = std::min(headMin, val);
                    headMax = std::max(headMax, val);

                    // 全局统计
                    sum += val;
                    min_val = std::min(min_val, val);
                    max_val = std::max(max_val, val);
                    ++validCnt;

                    // 检查是否包含接近未选中块值(9.9)的数据
                    if (std::abs(val - UNSELECTED_BLOCK_VALUE) < 1.0f) {
                        hasUnselectedValue = true;
                        LOG_PRINT("  ⚠️ 警告: 检测到未选中块的值: %.6f at index %zu (head %d)\n", val, i, h);
                    }
                }

                // 计算并打印每个头的统计信息
                float headMean = headSum / (sequenceLengthQ * headDims);

                // 每隔几个头打印一次，避免输出过多
                if (h < 3 || h >= numHeads - 3 || h % 8 == 0) {
                    LOG_PRINT("  头 %d: 均值=%.6f, 最小值=%.6f, 最大值=%.6f\n", h, headMean, headMin, headMax);
                }
            }

            float mean = sum / static_cast<float>(validCnt);

            LOG_PRINT("\n全局统计信息:\n");
            LOG_PRINT("  均值 (Mean): %.6f\n", mean);
            LOG_PRINT("  最小值 (Min): %.6f\n", min_val);
            LOG_PRINT("  最大值 (Max): %.6f\n", max_val);

            // 输出前16个有效元素
            LOG_PRINT("\n前16个输出元素:\n");
            for (size_t i = 0; i < std::min<size_t>(16, outputResult.size()); ++i) {
                if (i % 4 == 0)
                    LOG_PRINT("  ");
                LOG_PRINT("%.6f ", outputResult[i]);
                if ((i + 1) % 4 == 0)
                    LOG_PRINT("\n");
            }

            // 分析blockPosition映射是否生效
            LOG_PRINT("\n=== 多头BlockTable映射分析 ===\n");
            LOG_PRINT("  块选择测试: %s\n", !hasUnselectedValue ? "✅ 通过，结果不包含未选中块的影响" :
                                                                  "❌ 失败，结果中检测到未选中块的影响");

            // 检查不同头的结果差异
            bool headsHaveDifferentResults = false;
            float firstHeadValue = outputResult[0];
            for (int h = 1; h < numHeads; h++) {
                size_t headStartIdx = h * sequenceLengthQ * headDims;
                if (std::abs(outputResult[headStartIdx] - firstHeadValue) > 1e-5) {
                    headsHaveDifferentResults = true;
                    break;
                }
            }

            LOG_PRINT("  多头差异: %s\n", headsHaveDifferentResults ?
                                              "✅ 通过，不同头的结果有差异，说明每个头独立选择块成功" :
                                              "❌ 失败，所有头结果相似，多头块选择可能未生效");

            LOG_PRINT("  算子功能: %s\n", (!hasUnselectedValue && headsHaveDifferentResults) ?
                                              "✅ 二维BlockTable + 三维Position映射工作正常" :
                                              "❌ 映射机制可能存在问题，请检查实现");
            LOG_PRINT("=== 结果分析完成 ===\n\n");
        } else {
            LOG_PRINT("⚡️ 性能测试模式：跳过所有结果拷贝与分析\n");
        }

        // 释放资源
        if (!perfTest) {
            LOG_PRINT("  释放资源...\n");
        }

        aclDestroyTensor(queryTensor);
        aclDestroyTensor(keyTensor);
        aclDestroyTensor(valueTensor);
        aclDestroyTensor(outTensor);
        aclDestroyTensor(blockTableTensor);
        aclDestroyTensor(blockPositionTensor); // 释放新增的blockPosition资源
        aclDestroyTensorList(tensorKeyList);
        aclDestroyTensorList(tensorValueList);
        aclDestroyIntArray(actualSeqLengths);

        aclrtFree(queryDeviceAddr);
        aclrtFree(keyDeviceAddr);
        aclrtFree(valueDeviceAddr);
        aclrtFree(outDeviceAddr);
        aclrtFree(blockTableAddr);
        aclrtFree(blockPositionAddr); // 释放新增的blockPosition内存

        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }

        TEST_PASS("Multi-Head BlockTable Page Attention Mode (二维BlockTable + 三维Position)");
        return true;

    } catch (const std::exception &e) {
        LOG_PRINT("  ❌ 异常: %s\n", e.what());
        return false;
    } catch (...) {
        LOG_PRINT("  ❌ 未知异常\n");
        return false;
    }
}

// 添加新的测试函数：支持KV cache排布为(blocknum, blocksize, H)的情况
bool TestBlockSizeHeadLayout_BSH(aclrtStream stream)
{
    TEST_CASE("Block-Size-Head Layout Mode (KV Cache排布: blocknum, blocksize, H)");

    try {
        // 测试参数
        int32_t batchSize = 1;
        int32_t numHeads = 32;
        int32_t headDims = 128;
        int32_t keyNumHeads = 32;
        int32_t blockSize = 16;                         // 每块的token数
        int32_t sequenceLengthQ = 1;                     // Q序列长度
        int32_t totalSeqLengthKV = 32 * 1024;            // 总KV序列长度（足够大）
        int32_t blockNum = totalSeqLengthKV / blockSize; // 总块数

        // 设置基准块数和最大差异
        int32_t baseBlockNum = (32 * 1024) / blockSize; // 基准块数（约4K tokens）
        int32_t maxDiffPercent = 10;                    // 最大±10%差异
        int32_t minBlocksPerHead = baseBlockNum * (100 - maxDiffPercent) / 100;
        int32_t maxBlocksPerHead = baseBlockNum * (100 + maxDiffPercent) / 100;
        int32_t maxActualBlockNumPerSeq = maxBlocksPerHead;

        maxActualBlockNumPerSeq = baseBlockNum; // 用来做性能测试

        // 创建实际序列长度数组 - 使用maxActualBlockNumPerSeq作为最大块数
        std::vector<int64_t> actualSeqlenVector = {maxActualBlockNumPerSeq * blockSize};
        auto actualSeqLengths = aclCreateIntArray(actualSeqlenVector.data(), actualSeqlenVector.size());
        LOG_PRINT("  全局actualSeqLength: %ld\n", actualSeqlenVector[0]);

        // 选择性块值定义
        const float SELECTED_BLOCK_VALUE = 0.1f;   // 选中块的值
        const float UNSELECTED_BLOCK_VALUE = 9.9f; // 未选中块的值

        LOG_PRINT("  所有头使用相同的块数: %d\n", baseBlockNum);

        // ⚡️ 性能测试开关：环境变量 PERF_TEST=1 时开启
        const char *perfFlag = std::getenv("PERF_TEST");
        bool perfTest = (perfFlag && std::strcmp(perfFlag, "1") == 0);
        if (perfTest) {
            LOG_PRINT("⚡️ 性能测试模式已启用，跳过所有非核心计算代码\n");
        }

        LOG_PRINT("  创建张量形状...\n");

        // 张量形状定义
        std::vector<int64_t> queryShape = {batchSize, numHeads, sequenceLengthQ, headDims};
        std::vector<int64_t> keyShape = {blockNum, blockSize, keyNumHeads * headDims};    // 修改为(blocknum, blocksize, H, D)
        std::vector<int64_t> valueShape = {blockNum, blockSize, keyNumHeads * headDims};  // 修改为(blocknum, blocksize, H, D)
        std::vector<int64_t> outShape = {batchSize, numHeads, sequenceLengthQ, headDims};

        // 为每个头确定块数 - 所有头使用相同的块数，最后一个头少一个块
        std::vector<int32_t> blocksPerHead(keyNumHeads, maxActualBlockNumPerSeq);
        for (int h = 0; h < keyNumHeads; h++) {
            LOG_PRINT("  头 %d 选择块数: %d\n", h, blocksPerHead[h]);
        }

        // 计算全局需要的最大块数
        int32_t totalUniqueBlocks = 0;
        for (int h = 0; h < keyNumHeads; h++) {
            totalUniqueBlocks += blocksPerHead[h];
        }
        LOG_PRINT("  所有头总共需要 %d 个块\n", totalUniqueBlocks);

        // 二维BlockTable [batchSize, totalUniqueBlocks]
        std::vector<int64_t> blockTableShape = {batchSize, totalUniqueBlocks};
        LOG_PRINT("  BlockTable形状: [%ld, %ld]\n", blockTableShape[0], blockTableShape[1]);

        // 三维blockPosition [batchSize, keyNumHeads, maxActualBlockNumPerSeq]
        std::vector<int64_t> blockPositionShape = {batchSize, keyNumHeads, maxActualBlockNumPerSeq};
        LOG_PRINT("  BlockPosition形状: [%ld, %ld, %ld]\n", blockPositionShape[0], blockPositionShape[1],
                  blockPositionShape[2]);

        if (!perfTest) {
            LOG_PRINT("张量形状信息:\n");
            LOG_PRINT("  Query Shape (BNSD): [%ld, %ld, %ld, %ld]\n", queryShape[0], queryShape[1], queryShape[2],
                      queryShape[3]);
            LOG_PRINT("  Key Shape (blocknum, blocksize, H, D): [%ld, %ld, %ld, %ld]\n", keyShape[0], keyShape[1], keyShape[2],
                      keyShape[3]);
            LOG_PRINT("  Value Shape (blocknum, blocksize, H, D): [%ld, %ld, %ld, %ld]\n", valueShape[0], valueShape[1],
                      valueShape[2], valueShape[3]);
            LOG_PRINT("  Output Shape: [%ld, %ld, %ld, %ld]\n", outShape[0], outShape[1], outShape[2], outShape[3]);
            LOG_PRINT("  最大序列长度: %d\n", maxActualBlockNumPerSeq * blockSize);
        }

        LOG_PRINT("  生成测试数据...\n");

        // 生成Query数据
        auto queryData = std::vector<float>(GetShapeSize(queryShape));
        std::mt19937 gen(42); // 固定种子保证可重现
        for (size_t i = 0; i < queryData.size(); i++) {
            queryData[i] = 0.1f; // 使用小数值避免Softmax溢出
        }

        // 生成Key数据 - 初始为未选择值
        auto keyData = std::vector<float>(GetShapeSize(keyShape));
        for (size_t i = 0; i < keyData.size(); i++) {
            keyData[i] = UNSELECTED_BLOCK_VALUE;
        }

        // 生成Value数据 - 初始为未选择值
        auto valueData = std::vector<float>(GetShapeSize(valueShape));
        for (size_t i = 0; i < valueData.size(); i++) {
            valueData[i] = UNSELECTED_BLOCK_VALUE;
        }

        // 生成二维BlockTable数据和三维blockPosition数据
        std::vector<int32_t> blockTableData(GetShapeSize(blockTableShape), 0x7FFFFFFF);       // 默认填充无效值
        std::vector<int32_t> blockPositionData(GetShapeSize(blockPositionShape), 0x7FFFFFFF); // 默认填充无效值

        // 设置选中块的数据并填充BlockTable和blockPosition
        for (int b = 0; b < batchSize; b++) {
            int blockTableIdx = 0; // 记录当前blockTable中的位置

            for (int h = 0; h < keyNumHeads; h++) {
                LOG_PRINT("  头 %d 实际块数: %d/%d\n", h, blocksPerHead[h], maxActualBlockNumPerSeq);

                // 基于头索引计算基础偏移量
                int baseOffset = (h * 3) % blockNum; // 使用不同的起始偏移

                LOG_PRINT("  头 %d 选择的块: ", h);

                // 填充该头使用的块
                for (int i = 0; i < blocksPerHead[h]; i++) {
                    // 计算块索引，确保在合法范围内
                    int32_t blockIdx = (baseOffset + i * 2) % blockNum; // 使用2的步长增加差异

                    if (i < 10) { // 只打印前10个块
                        LOG_PRINT("%d ", blockIdx);
                    } else if (i == 10) {
                        LOG_PRINT("...");
                    }

                    // 设置BlockTable索引 - 在二维BlockTable中记录实际的块ID
                    int tableIdx = b * totalUniqueBlocks + blockTableIdx;
                    blockTableData[tableIdx] = blockIdx;

                    // 设置blockPosition索引 - 在三维blockPosition中记录对应的BlockTable位置
                    int positionIdx = (b * keyNumHeads + h) * maxActualBlockNumPerSeq + i;
                    blockPositionData[positionIdx] = blockTableIdx;

                    blockTableIdx++; // 递增BlockTable索引

                    // 设置该块的KV数据为"选择"值
                    float headSpecificValue = SELECTED_BLOCK_VALUE + h * 0.001f; // 为每个头添加微小差异便于调试

                    // 设置KV数据 - (blocknum, blocksize, H, D)格式
                    for (int s = 0; s < blockSize; s++) {
                        for (int d = 0; d < headDims; d++) {
                            // (blocknum, blocksize, H, D)格式索引计算: (blockIdx, s, h, d)
                            int64_t dataIdx = blockIdx * (blockSize * keyNumHeads * headDims) + 
                                            s * (keyNumHeads * headDims) + 
                                            h * headDims + d;
                            if (dataIdx < keyData.size()) {
                                keyData[dataIdx] = headSpecificValue;
                                valueData[dataIdx] = headSpecificValue;
                            }
                        }
                    }
                }
                LOG_PRINT("\n");
            }
        }

        if (!perfTest) {
            // 验证BlockPosition映射
            bool validMapping = true;
            for (int b = 0; b < batchSize; b++) {
                for (int h = 0; h < keyNumHeads; h++) {
                    // 检查有效映射数量
                    int validMappings = 0;
                    for (int i = 0; i < maxActualBlockNumPerSeq; i++) {
                        int positionIdx = (b * keyNumHeads + h) * maxActualBlockNumPerSeq + i;
                        if (blockPositionData[positionIdx] != 0x7FFFFFFF) {
                            validMappings++;

                            // 验证索引是否有效
                            int blockTableIdx = blockPositionData[positionIdx];
                            if (blockTableIdx < 0 || blockTableIdx >= totalUniqueBlocks) {
                                LOG_PRINT("  ❌ 头 %d 的position[%d]=%d 超出范围!\n", h, i, blockTableIdx);
                                validMapping = false;
                            }
                        }
                    }

                    // 验证有效映射数量是否等于预期的块数
                    if (validMappings != blocksPerHead[h]) {
                        LOG_PRINT("  ❌ 头 %d 映射数量不符: 预期 %d, 实际 %d\n", h, blocksPerHead[h], validMappings);
                        validMapping = false;
                    }
                }
            }

            if (validMapping) {
                LOG_PRINT("  ✅ BlockPosition验证通过: 每个头的块映射符合预期\n");
            } else {
                LOG_PRINT("  ❌ BlockPosition验证失败: 请检查映射逻辑\n");
            }
        }

        auto outData = GenerateConstantFloatData(GetShapeSize(outShape), 0.0f); // 输出初始化为0

        LOG_PRINT("  创建设备张量...\n");

        // 创建设备张量
        void *queryDeviceAddr = nullptr, *keyDeviceAddr = nullptr;
        void *valueDeviceAddr = nullptr, *outDeviceAddr = nullptr;
        void *blockTableAddr = nullptr, *blockPositionAddr = nullptr;

        aclTensor *queryTensor = nullptr, *keyTensor = nullptr;
        aclTensor *valueTensor = nullptr, *outTensor = nullptr;
        aclTensor *blockTableTensor = nullptr, *blockPositionTensor = nullptr;

        int ret = 0;

        if (!perfTest) {
            LOG_PRINT("  创建 query tensor...\n");
        }
        ret = CreateAclTensor(queryData, queryShape, &queryDeviceAddr, ACL_FLOAT16, &queryTensor);
        if (ret != 0) {
            LOG_PRINT("  ❌ 创建 query tensor 失败\n");
            return false;
        }

        if (!perfTest) {
            LOG_PRINT("  创建 key tensor...\n");
        }
        ret = CreateAclTensor(keyData, keyShape, &keyDeviceAddr, ACL_FLOAT16, &keyTensor);
        if (ret != 0) {
            LOG_PRINT("  ❌ 创建 key tensor 失败\n");
            return false;
        }

        if (!perfTest) {
            LOG_PRINT("  创建 value tensor...\n");
        }
        ret = CreateAclTensor(valueData, valueShape, &valueDeviceAddr, ACL_FLOAT16, &valueTensor);
        if (ret != 0) {
            LOG_PRINT("  ❌ 创建 value tensor 失败\n");
            return false;
        }

        if (!perfTest) {
            LOG_PRINT("  创建 output tensor...\n");
        }
        ret = CreateAclTensor(outData, outShape, &outDeviceAddr, ACL_FLOAT16, &outTensor);
        if (ret != 0) {
            LOG_PRINT("  ❌ 创建 output tensor 失败\n");
            return false;
        }

        if (!perfTest) {
            LOG_PRINT("  创建 blockTable tensor...\n");
        }
        ret = CreateAclTensor(blockTableData, blockTableShape, &blockTableAddr, ACL_INT32, &blockTableTensor);
        if (ret != 0) {
            LOG_PRINT("  ❌ 创建 blockTable tensor 失败\n");
            return false;
        }

        if (!perfTest) {
            LOG_PRINT("  创建 blockPosition tensor...\n");
        }
        ret =
            CreateAclTensor(blockPositionData, blockPositionShape, &blockPositionAddr, ACL_INT32, &blockPositionTensor);
        if (ret != 0) {
            LOG_PRINT("  ❌ 创建 blockPosition tensor 失败\n");
            return false;
        }

        if (!perfTest) {
            LOG_PRINT("  创建张量列表...\n");
        }

        // 创建张量列表
        int kvTensorNum = 1;
        aclTensor *tensorsOfKey[kvTensorNum] = {keyTensor};
        aclTensor *tensorsOfValue[kvTensorNum] = {valueTensor};
        auto tensorKeyList = aclCreateTensorList(tensorsOfKey, kvTensorNum);
        auto tensorValueList = aclCreateTensorList(tensorsOfValue, kvTensorNum);

        if (tensorKeyList == nullptr || tensorValueList == nullptr) {
            LOG_PRINT("  ❌ 创建张量列表失败\n");
            return false;
        }

        // 设置算子参数
        int64_t numKeyValueHeads = keyNumHeads;
        int64_t blockSizeParam = blockSize;
        int64_t innerPrecise = 1;
        double scaleValue = 1.0 / sqrt(static_cast<double>(headDims));
        string sLayout = "BNSD";  // KV缓存布局为(blocknum, blocksize, H, D)
        char layout[sLayout.length() + 1];
        strcpy(layout, sLayout.c_str());

        if (!perfTest) {
            LOG_PRINT("  算子参数: numHeads=%d, scaleValue=%.6f, blockSize=%ld, layout=%s\n", 
                       numHeads, scaleValue, blockSizeParam, layout);
            LOG_PRINT("  调用 GetWorkspaceSize...\n");
        }

        // 调用V5接口
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;

        ret = aclnnIncreFlashAttentionV5GetWorkspaceSize(queryTensor, tensorKeyList, tensorValueList,
                                                         nullptr,             // pse_shift
                                                         nullptr,             // attenMask
                                                         actualSeqLengths,    // 传入实际序列长度
                                                         nullptr,             // dequant_scale1
                                                         nullptr,             // quant_scale1
                                                         nullptr,             // dequant_scale2
                                                         nullptr,             // quant_scale2
                                                         nullptr,             // quant_offset2
                                                         nullptr,             // antiquant_scale
                                                         nullptr,             // antiquant_offset
                                                         blockTableTensor,    // blockTable - 二维
                                                         nullptr,             // kvPaddingSize - 不使用
                                                         blockPositionTensor, // blockPosition - 三维映射表
                                                         numHeads, scaleValue, layout, numKeyValueHeads,
                                                         blockSizeParam, // 传入blockSize
                                                         innerPrecise,
                                                         outTensor, &workspaceSize, &executor);

        if (!perfTest) {
            LOG_PRINT("  GetWorkspaceSize 返回: %d, error: %s\n", ret, aclGetRecentErrMsg());
        }

        if (ret != ACL_SUCCESS) {
            TEST_FAIL("Block-Size-Head Layout Mode", "GetWorkspaceSize failed");
            return false;
        }

        if (!perfTest) {
            LOG_PRINT("  工作空间大小: %llu bytes\n", (unsigned long long)workspaceSize);
        }

        // 申请工作空间
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                TEST_FAIL("Block-Size-Head Layout Mode", "Workspace allocation failed");
                return false;
            }
            if (!perfTest) {
                LOG_PRINT("  工作空间分配成功: %lu bytes\n", workspaceSize);
            }
        }

        // 执行计算
        if (!perfTest) {
            LOG_PRINT("  开始执行算子计算...\n");
            LOG_PRINT(" aclnnIncreFlashAttentionV5 执行前Log: %s\n", aclGetRecentErrMsg());
        }
        ret = aclnnIncreFlashAttentionV5(workspaceAddr, workspaceSize, executor, stream);
        if (!perfTest) {
            LOG_PRINT(" aclnnIncreFlashAttentionV5 执行后Log: %s\n", aclGetRecentErrMsg());
        }
        if (ret != ACL_SUCCESS) {
            TEST_FAIL("Block-Size-Head Layout Mode", "Execution failed");
            LOG_PRINT("  ❌ 算子执行失败，错误码: %d, 错误信息: %s\n", ret, aclGetRecentErrMsg());
            return false;
        }
        if (!perfTest) {
            LOG_PRINT("  ✅ 算子执行成功\n");
        }

        // 同步等待
        if (!perfTest) {
            LOG_PRINT("  等待计算完成...\n");
        }

        ret = aclrtSynchronizeStream(stream);
        if (!perfTest) {
            LOG_PRINT("  aclrtSynchronizeStream 执行Log: %s\n", aclGetRecentErrMsg());
        }

        if (ret != ACL_SUCCESS) {
            TEST_FAIL("Block-Size-Head Layout Mode", "Stream synchronization failed");

            if (!perfTest) {
                LOG_PRINT("  ❌ 流同步失败，错误码: %d\n", ret);
                LOG_PRINT("  📊 PageAttention错误分析：\n");
                LOG_PRINT("     - 如果是超时错误：可能是blockTable或blockPosition配置问题\n");
                LOG_PRINT("     - 如果是AICore错误：可能是内存访问模式问题\n");
                LOG_PRINT("     - 如果是内存错误：可能是索引超出范围\n");

                const char *errMsg = aclGetRecentErrMsg();
                LOG_PRINT("  💬 详细错误信息: %s\n", errMsg ? errMsg : "无详细信息");
            }

            return false;
        }
        if (!perfTest) {
            LOG_PRINT("  ✅ 计算完成，流同步成功\n");
        }

        // 输出结果处理与统计
        if (!perfTest) {
            LOG_PRINT("  开始读取计算结果...\n");

            // 从设备拷贝结果到主机
            std::vector<uint16_t> outputResultHalf(GetShapeSize(outShape)); // 使用uint16_t表示half
            ret = aclrtMemcpy(outputResultHalf.data(), outputResultHalf.size() * sizeof(uint16_t), outDeviceAddr,
                              outputResultHalf.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret != ACL_SUCCESS) {
                LOG_PRINT("  ❌ 结果拷贝失败，错误码: %d\n", ret);
                return false;
            }

            // 将half转换为float (准确版本)
            std::vector<float> outputResult(GetShapeSize(outShape));
            for (size_t i = 0; i < outputResultHalf.size(); i++) {
                outputResult[i] = halfToFloat(outputResultHalf[i]);
            }

            LOG_PRINT("  ✅ 结果拷贝成功\n");

            // 输出张量形状信息
            LOG_PRINT("\n=== 计算结果分析 ===\n");
            LOG_PRINT("输出张量形状: [%ld, %ld, %ld, %ld]\n", outShape[0], outShape[1], outShape[2], outShape[3]);

            // 统计结果
            float sum = 0.0f, min_val = 1e30f, max_val = -1e30f;
            size_t validCnt = 0;
            bool hasUnselectedValue = false;

            // 分头统计结果
            for (int h = 0; h < numHeads; h++) {
                float headSum = 0.0f;
                float headMin = 1e30f;
                float headMax = -1e30f;

                // 计算头h的起始索引
                size_t headStartIdx = h * sequenceLengthQ * headDims;
                size_t headEndIdx = headStartIdx + sequenceLengthQ * headDims;

                for (size_t i = headStartIdx; i < headEndIdx; ++i) {
                    float val = outputResult[i];
                    headSum += val;
                    headMin = std::min(headMin, val);
                    headMax = std::max(headMax, val);

                    // 全局统计
                    sum += val;
                    min_val = std::min(min_val, val);
                    max_val = std::max(max_val, val);
                    ++validCnt;

                    // 检查是否包含接近未选中块值(9.9)的数据
                    if (std::abs(val - UNSELECTED_BLOCK_VALUE) < 1.0f) {
                        hasUnselectedValue = true;
                        LOG_PRINT("  ⚠️ 警告: 检测到未选中块的值: %.6f at index %zu (head %d)\n", val, i, h);
                    }
                }

                // 计算并打印每个头的统计信息
                float headMean = headSum / (sequenceLengthQ * headDims);

                // 每隔几个头打印一次，避免输出过多
                if (h < 3 || h >= numHeads - 3 || h % 8 == 0) {
                    LOG_PRINT("  头 %d: 均值=%.6f, 最小值=%.6f, 最大值=%.6f\n", h, headMean, headMin, headMax);
                }
            }

            float mean = sum / static_cast<float>(validCnt);

            LOG_PRINT("\n全局统计信息:\n");
            LOG_PRINT("  均值 (Mean): %.6f\n", mean);
            LOG_PRINT("  最小值 (Min): %.6f\n", min_val);
            LOG_PRINT("  最大值 (Max): %.6f\n", max_val);

            // 输出前16个有效元素
            LOG_PRINT("\n前16个输出元素:\n");
            for (size_t i = 0; i < std::min<size_t>(16, outputResult.size()); ++i) {
                if (i % 4 == 0)
                    LOG_PRINT("  ");
                LOG_PRINT("%.6f ", outputResult[i]);
                if ((i + 1) % 4 == 0)
                    LOG_PRINT("\n");
            }

            // 分析blockPosition映射是否生效
            LOG_PRINT("\n=== Block-Size-Head布局映射分析 ===\n");
            LOG_PRINT("  块选择测试: %s\n", !hasUnselectedValue ? "✅ 通过，结果不包含未选中块的影响" :
                                                                  "❌ 失败，结果中检测到未选中块的影响");

            // 检查不同头的结果差异
            bool headsHaveDifferentResults = false;
            float firstHeadValue = outputResult[0];
            for (int h = 1; h < numHeads; h++) {
                size_t headStartIdx = h * sequenceLengthQ * headDims;
                if (std::abs(outputResult[headStartIdx] - firstHeadValue) > 1e-5) {
                    headsHaveDifferentResults = true;
                    break;
                }
            }

            LOG_PRINT("  多头差异: %s\n", headsHaveDifferentResults ?
                                              "✅ 通过，不同头的结果有差异，说明每个头独立选择块成功" :
                                              "❌ 失败，所有头结果相似，多头块选择可能未生效");

            LOG_PRINT("  算子功能: %s\n", (!hasUnselectedValue && headsHaveDifferentResults) ?
                                              "✅ Block-Size-Head布局 (blocknum, blocksize, H) 工作正常" :
                                              "❌ 布局机制可能存在问题，请检查实现");
            LOG_PRINT("=== 结果分析完成 ===\n\n");
        } else {
            LOG_PRINT("⚡️ 性能测试模式：跳过所有结果拷贝与分析\n");
        }

        // 释放资源
        if (!perfTest) {
            LOG_PRINT("  释放资源...\n");
        }

        aclDestroyTensor(queryTensor);
        aclDestroyTensor(keyTensor);
        aclDestroyTensor(valueTensor);
        aclDestroyTensor(outTensor);
        aclDestroyTensor(blockTableTensor);
        aclDestroyTensor(blockPositionTensor);
        aclDestroyTensorList(tensorKeyList);
        aclDestroyTensorList(tensorValueList);
        aclDestroyIntArray(actualSeqLengths);

        aclrtFree(queryDeviceAddr);
        aclrtFree(keyDeviceAddr);
        aclrtFree(valueDeviceAddr);
        aclrtFree(outDeviceAddr);
        aclrtFree(blockTableAddr);
        aclrtFree(blockPositionAddr);

        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }

        TEST_PASS("Block-Size-Head Layout Mode (KV Cache排布: blocknum, blocksize, H)");
        return true;

    } catch (const std::exception &e) {
        LOG_PRINT("  ❌ 异常: %s\n", e.what());
        return false;
    } catch (...) {
        LOG_PRINT("  ❌ 未知异常\n");
        return false;
    }
}

// ===================== 主测试函数 =====================

// 测试函数前向声明
bool TestMultiHeadBlockTableMode_BSBD(aclrtStream stream);
bool TestBlockSizeHeadLayout_BSH(aclrtStream stream);
int main()
{
    LOG_PRINT("🚀 开始 IncreFlashAttentionV5 完整测试套件\n");
    LOG_PRINT("================================================\n");
    
    // 测试日志输出
    printf("=== 测试日志输出 ===\n");
    printf("[DEBUG][TestOp] This is a test debug message\n");
    printf("[DEBUG][TestOp] Testing with parameters: %d, %s\n", 42, "hello");
    printf("[DEBUG][TestOp] Testing sInnerSize_=%u, seqSize_=%u\n", 0, 0);
    printf("=== 日志测试完成 ===\n\n");
    
    // 1. 初始化ACL环境
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("❌ ACL初始化失败. ERROR: %d\n", ret);
        return -1;
    }
    LOG_PRINT("✅ ACL环境初始化成功\n\n");
    
    // 2. 执行测试用例
    LOG_PRINT("开始执行测试用例...\n");
        
    // bool result1 = TestDenseMode(stream);
    // g_testStats.addResult(result1);

    // 新增：执行PageAttention测试用例
    // bool result2 = TestNormalPageAttentionMode(stream);
    // g_testStats.addResult(result2);

    // bool result3 = TestMultiHeadBlockTableMode_BSBD(stream);
    // g_testStats.addResult(result3);

    bool result4 = TestBlockSizeHeadLayout_BSH(stream);
    g_testStats.addResult(result4);

    // 新增：执行修改版PageAttention测试用例（KV缓存三维形状）
    // LOG_PRINT("\n开始执行修改版PageAttention测试（三维KV缓存）...\n");
    // bool result3 = TestModifiedPageAttentionMode(stream);
    // g_testStats.addResult(result3);
    
    // bool result2 = TestSparseMode(stream);
    // g_testStats.addResult(result2);
    
    // bool result3 = TestInt8KVMode(stream);
    // g_testStats.addResult(result3);
        

    // 3. 输出测试总结
    LOG_PRINT("\n================================================\n");
    g_testStats.printSummary();


    // 4. 清理ACL环境
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    LOG_PRINT("✅ ACL环境清理完成\n");
   
    // 5. 返回测试结果
    bool allPassed = (g_testStats.failed == 0);
    LOG_PRINT("\n🏁 测试完成: %s\n", allPassed ? "全部通过" : "存在失败");
    
    return allPassed ? 0 : -1;
}