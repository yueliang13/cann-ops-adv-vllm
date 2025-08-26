# ComputeCentVec算子测试套件

## 概述

本测试套件为 `ComputeCentVec` 算子提供完整的测试覆盖，包括功能验证、性能测试、边界条件测试等。测试套件基于参考的 `Test_Case_v5` 框架设计，确保测试的一致性和完整性。

## 测试内容

### 1. 基本功能测试 (`TestBasicFunctionality`)
- **目的**: 验证算子的基本功能是否正常
- **测试内容**: 
  - 小规模向量计算（batch_size=2, qhead_num=4, seq_len=8, dim=64）
  - ACL Tensor创建和销毁
  - Workspace大小计算
  - 算子执行
- **预期结果**: 所有操作成功完成，无错误返回

### 2. 数据类型兼容性测试 (`TestDataTypeCompatibility`)
- **目的**: 验证算子对不同数据类型的支持
- **测试内容**:
  - 输入数据类型: FP16, FP32
  - 输出数据类型: FP16, FP32
  - 不同数据类型组合的兼容性
- **预期结果**: 所有数据类型组合都能正常工作

### 3. 边界条件测试 (`TestBoundaryConditions`)
- **目的**: 验证算子在极端情况下的稳定性
- **测试内容**:
  - 最小维度测试（1x1x1x16）
  - 较大维度测试（4x8x64x512）
  - 零值数据测试
- **预期结果**: 在边界条件下仍能正常工作

### 4. 错误处理测试 (`TestErrorHandling`)
- **目的**: 验证算子的错误处理机制
- **测试内容**:
  - 空指针参数测试
  - 无效参数测试
- **预期结果**: 正确识别并处理错误情况

### 5. 性能基准测试 (`TestPerformanceBenchmark`)
- **目的**: 评估算子的性能表现
- **测试内容**:
  - 小规模配置（1x2x4x16x32x8x4）
  - 中等规模配置（2x4x8x32x128x16x8）
  - 大规模配置（4x8x16x64x256x32x16）
- **预期结果**: 提供workspace大小和计算时间信息

## 文件结构

```
test/
├── test_compute_cent.cpp    # 主测试代码
├── CMakeLists.txt           # CMake构建配置
├── build.sh                 # 构建脚本
└── README_TEST.md           # 测试说明文档
```

## 构建和运行

### 环境要求

- **操作系统**: Linux (推荐 Ubuntu 18.04+)
- **编译器**: GCC 7.0+ 或 Clang 5.0+
- **构建工具**: CMake 3.14+
- **CANN环境**: 已安装并配置CANN工具包
- **依赖库**: 
  - libascendcl.so
  - libnnopbase.so
  - libascendalog.so
  - libcust_opapi.so

### 环境变量设置

```bash
# 设置CANN环境变量（选择其中一个）
export ASCEND_CUSTOM_PATH=/path/to/custom/cann
# 或
export ASCEND_HOME_PATH=/path/to/ascend/toolkit

# 设置PATH
export PATH=$ASCEND_CUSTOM_PATH/bin:$PATH
# 或
export PATH=$ASCEND_HOME_PATH/bin:$PATH

# 设置库路径
export LD_LIBRARY_PATH=$ASCEND_CUSTOM_PATH/lib64:$LD_LIBRARY_PATH
# 或
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$LD_LIBRARY_PATH
```

### 构建步骤

1. **进入测试目录**
   ```bash
   cd /path/to/compute_cent/test
   ```

2. **设置执行权限**
   ```bash
   chmod +x build.sh
   ```

3. **构建项目**
   ```bash
   # 仅构建
   ./build.sh
   
   # 构建并运行测试
   ./build.sh --run-test
   
   # 清理构建目录
   ./build.sh --clean
   
   # 显示帮助信息
   ./build.sh --help
   ```

### 手动构建（可选）

如果不想使用构建脚本，也可以手动执行：

```bash
# 创建构建目录
mkdir -p build bin
cd build

# 配置CMake
cmake .. -DCMAKE_BUILD_TYPE=Debug

# 编译
make -j$(nproc)

# 返回上级目录
cd ..
```

## 运行测试

### 直接运行

```bash
# 进入bin目录
cd bin

# 运行测试
./compute_cent_test
```

### 测试输出示例

```
🚀 开始运行ComputeCentVec算子测试套件

[TEST] === 开始测试: 基本功能测试 ===
[TEST] [EXEC_NPU_CMD] Starting aclnnComputeCentGetWorkspaceSize
[TEST] [EXEC_NPU_CMD] Function addresses obtained
[TEST] [EXEC_NPU_CMD] Stream and variables initialized
[TEST] [EXEC_NPU_CMD] About to convert parameters...
[TEST] [EXEC_NPU_CMD] Parameters converted successfully
[TEST] [EXEC_NPU_CMD] About to call getWorkspaceSizeFunc...
[TEST] [EXEC_NPU_CMD] getWorkspaceSizeFunc returned status: 0
[TEST] [EXEC_NPU_CMD] workspace_size: 1024 bytes (0.001 MB)
[TEST] ✅ 基本功能测试: PASS

[TEST] === 开始测试: 数据类型兼容性测试 ===
[TEST] 测试数据类型组合: query=1, output=0
[TEST] 数据类型 1 兼容性测试通过
[TEST] 测试数据类型组合: query=1, output=1
[TEST] 数据类型 1 兼容性测试通过
[TEST] 测试数据类型组合: query=0, output=0
[TEST] 数据类型 0 兼容性测试通过
[TEST] 测试数据类型组合: query=0, output=1
[TEST] 数据类型 0 兼容性测试通过
[TEST] ✅ 数据类型兼容性测试: PASS

📊 测试总结: 5/5 通过 (100.0%)
🎉 所有测试通过!
🏁 ComputeCentVec算子测试套件运行完成
```

## 测试配置

### 测试参数

测试套件中的主要参数说明：

- **batch_size**: 批次大小
- **qhead_num**: 查询头数量
- **kvhead_num**: 键值头数量
- **seq_len**: 序列长度
- **dim**: 向量维度
- **cluster_num**: 聚类数量
- **k**: TopK选择的数量

### 自定义测试

如需添加新的测试用例，可以在 `test_compute_cent.cpp` 中添加新的测试函数，并在 `RunAllTests()` 函数中调用：

```cpp
// 添加新测试函数
bool TestNewFeature() {
    TEST_CASE("新功能测试");
    // ... 测试实现 ...
    TEST_PASS("新功能测试");
    return true;
}

// 在RunAllTests中调用
void RunAllTests() {
    // ... 现有测试 ...
    g_testStats.addResult(TestNewFeature());
    // ... 其他代码 ...
}
```

## 故障排除

### 常见问题

1. **编译错误: 找不到头文件**
   - 检查 `ASCEND_CUSTOM_PATH` 或 `ASCEND_HOME_PATH` 环境变量
   - 确认CANN工具包已正确安装

2. **链接错误: 找不到库文件**
   - 检查库文件路径是否正确
   - 确认 `LD_LIBRARY_PATH` 环境变量设置

3. **运行时错误: ACL初始化失败**
   - 检查CANN环境配置
   - 确认NPU设备可用

4. **测试失败: 算子执行失败**
   - 检查输入数据格式
   - 确认算子实现正确
   - 查看详细错误日志

### 调试技巧

1. **启用详细日志**
   ```bash
   export ASCEND_LOG_LEVEL=3
   ```

2. **检查NPU状态**
   ```bash
   npu-smi info
   ```

3. **查看系统日志**
   ```bash
   dmesg | grep -i npu
   ```

## 性能优化

### 编译优化

- 使用Release模式编译以获得最佳性能
- 启用编译器优化选项
- 使用适当的CPU架构标志

### 运行时优化

- 合理设置workspace大小
- 使用适当的batch size
- 优化内存访问模式

## 维护和更新

### 版本控制

- 测试代码应与算子实现保持同步
- 记录测试用例的变更历史
- 定期更新测试数据

### 持续集成

- 集成到CI/CD流程中
- 自动化测试执行
- 测试结果报告生成

## 联系信息

如有问题或建议，请联系：
- 测试团队: compute_cent_test@huawei.com
- 技术支持: cann_support@huawei.com

## 许可证

本测试套件遵循与主项目相同的许可证条款。 