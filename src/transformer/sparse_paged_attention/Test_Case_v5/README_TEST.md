# IncreFlashAttentionV5 完整测试套件

## 概述

本测试套件为IncreFlashAttentionV5接口提供全面的功能验证，确保V5稀疏注意力功能的正确性和可靠性。

## 测试架构

### 测试文件结构
```
IncreFlashAttentionV5/
├── test_incre_flash_attention_v5.cpp  # 主测试文件
├── CMakeLists.txt                      # CMake构建配置
├── build.sh                           # 自动化构建脚本
├── README_TEST.md                      # 本文档
└── build/                             # 构建输出目录
    └── bin/
        └── incre_flash_attention_v5_test  # 可执行测试程序
```

## 测试用例详解

### 测试1: 密集注意力模式兼容性测试
- **目标**: 验证V5接口在密集模式下与V4的向后兼容性
- **测试参数**:
  - Batch Size: 2
  - Number of Heads: 4
  - Head Dimensions: 64
  - Sequence Length: 128
- **验证点**:
  - 接口调用成功
  - 工作空间计算正确
  - 算子执行无错误
  - 与V4结果一致性

### 测试2: IVF稀疏注意力模式测试
- **目标**: 验证V5接口的核心稀疏注意力功能
- **测试参数**:
  - Batch Size: 1
  - Number of Heads: 2
  - Head Dimensions: 32
  - Sequence Length: 64
  - Number of Probes: 4
  - Number of Clusters: 8
- **稀疏数据**:
  - `selectNprobe`: [1, 2, 4, 8] 形状的聚类索引
  - `ivfStart`: 每个聚类的起始位置
  - `ivfLen`: 每个聚类的长度
- **验证点**:
  - 稀疏参数正确传递
  - IVF聚类计算正确
  - 内存访问优化生效
  - 性能符合预期

### 测试3: 参数验证测试
- **目标**: 验证接口对非法参数的鲁棒性处理
- **测试场景**:
  - 3.1: 无效的稀疏模式值 (>3)
  - 3.2: 稀疏模式下缺少必要参数
  - 3.3: nprobe > clusterNum 的非法关系
- **验证点**:
  - 正确返回 ACL_ERROR_INVALID_PARAM
  - 错误信息清晰准确
  - 不会导致程序崩溃

### 测试4: 性能基准测试
- **目标**: 评估稀疏注意力的性能优势
- **测试配置**:
  - 大规模参数 (BS=8, NH=16, HD=128, SeqLen=2048)
  - 密集模式 vs 稀疏模式性能对比
- **性能指标**:
  - 执行时间
  - 内存使用量
  - 计算精度

## 稀疏注意力数据格式

### 核心数据结构
```cpp
struct SparseData {
    std::vector<int32_t> selectNprobe;  // [bs, nh, nprobe] 
    std::vector<int32_t> ivfStart;      // [bs, nh_k, cluster]
    std::vector<int32_t> ivfLen;        // [bs, nh_k, cluster]
};
```

### 数据生成逻辑
1. **selectNprobe**: 均匀分布选择聚类索引，确保不重复
2. **ivfStart**: 连续分配起始位置，模拟真实聚类布局
3. **ivfLen**: 动态长度分配，考虑负载均衡

## 编译和运行

### 前提条件
- CANN开发套件已安装
- 昇腾AI处理器设备可用
- CMake 3.14+ 和 G++编译器

### 快速启动
```bash
cd IncreFlashAttentionV5
chmod +x build.sh
./build.sh
```

### 手动编译
```bash
# 设置环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 编译
mkdir -p build && cd build
cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
make

# 运行
./bin/incre_flash_attention_v5_test
```

## 测试结果解释

### 成功输出示例
```
[TEST] 🚀 开始 IncreFlashAttentionV5 完整测试套件
[TEST] ================================================
[TEST] ✅ ACL环境初始化成功

[TEST] === 开始测试: Dense Mode Compatibility ===
[TEST] ✅ Dense Mode Compatibility: PASS

[TEST] === 开始测试: IVF Sparse Mode ===
[TEST] ✅ IVF Sparse Mode: PASS

[TEST] === 开始测试: Parameter Validation ===
[TEST] ✅ Parameter Validation: PASS

[TEST] === 开始测试: Performance Benchmark ===
[TEST] ✅ Performance Benchmark: PASS

[TEST] 📊 测试总结: 4/4 通过 (100.0%)
[TEST] 🎉 所有测试通过!
[TEST] 🏁 测试完成: 全部通过
```

### 失败处理
- 详细的错误日志输出
- 资源自动清理
- 明确的失败原因说明

## 扩展和定制

### 添加新测试用例
1. 在 `test_incre_flash_attention_v5.cpp` 中添加新的测试函数
2. 在 `main()` 函数中调用新测试
3. 更新 `g_testStats` 统计

### 修改测试参数
- 调整各测试函数中的配置参数
- 修改数据生成逻辑
- 自定义验证条件

### 性能分析
- 添加计时器测量执行时间
- 使用内存分析工具监控内存使用
- 集成性能分析框架

## 已知限制

1. **硬件依赖**: 需要真实的昇腾设备进行完整验证
2. **数据规模**: 当前测试使用小规模数据，生产环境需要大规模验证
3. **精度验证**: 需要添加数值精度对比测试

## 故障排除

### 常见问题
1. **编译错误**: 检查CANN环境设置和头文件路径
2. **运行时错误**: 验证设备可用性和驱动版本
3. **内存不足**: 调整测试数据规模或增加设备内存

### 调试建议
- 使用 `gdb` 进行断点调试
- 启用详细日志输出
- 检查ACL错误码含义

## 贡献指南

欢迎贡献新的测试用例和改进建议：
1. Fork 项目仓库
2. 创建功能分支
3. 添加测试用例
4. 提交 Pull Request

## 联系信息

如有问题或建议，请联系V5开发团队。 