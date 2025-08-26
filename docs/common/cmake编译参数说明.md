声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# cmake编译参数说明

本文旨在对cann-ops-adv仓的构建工程中常用的cmake参数进行说明，并按使用场景组织各参数。

## 公共

- **ENABLE_CCACHE**：是否使能CCACHE，默认使能，参数设置示例如下：

  ```shell
  # 使能CCACHE
  -DENABLE_CCACHE=ON

  # 关闭CCACHE
  -DENABLE_CCACHE=OFF
  ```

- **CUSTOM_CCACHE**：指定使用的ccache工具，如果环境上安装ccache，默认使用，未安装则不使用，示例如下：

  ```shell
  # 使用/usr/local/bin/ccache作为ccache工具
  -DCUSTOM_CCACHE="/usr/local/bin/ccache"
  ```

## 自定义算子包编译

- **ASCEND_COMPUTE_UNIT**：编译指定产品类型，支持指定一个或者多个，多个产品类型以分号分隔，并用引号包含；不设置时默认编译Atlas A2 训练系列产品，示例如下：

  ```shell
  # 编译Atlas A2 训练系列产品类型
  -DASCEND_COMPUTE_UNIT="<soc_version>"

  # 编译Atlas A2 训练系列产品和Atlas 推理系列产品类型
  -DASCEND_COMPUTE_UNIT="<soc_version1>;<soc_version2>"
  ```

  **说明：** *<soc_version>* 请配置为运行环境的AI处理器型号；如果无法确定具体的 *<soc_version>*，则在安装昇腾AI处理器的服务器执行**npu-smi info**命令进行查询，在查询到的*Name*前增加Ascend信息，例如*Name*对应取值为*xxxyz*，实际配置的 *<soc_version>* 值为Ascendxxxy。


- **ASCEND_OP_NAME**：编译指定算子，支持指定一个或者多个，多个算子以分号分隔，并用引号包含；不设置时默认编译所有算子，示例如下：

  ```shell
  # 编译flash_attention_score算子
  -DASCEND_OP_NAME="flash_attention_score"

  # 编译flash_attention_score和flash_attention_score_grad算子
  -DASCEND_OP_NAME="flash_attention_score;flash_attention_score_grad"
  ```

- **TILING_KEY**：指定tilingkey编译，支持指定一个或者多个，多个tilingkey以分号分隔，并用引号包含；不设置时默认编译所有tilingkey，建议配合**ASCEND_OP_NAME**参数一起使用，示例如下：

  ```shell
  -DASCEND_OP_NAME="flash_attention_score" -DTILING_KEY="10000000000220132943;10000000100220130943"
  ```

  **说明：**当前版本只有FlashAttentionScore、FlashAttentionScoreGrad、FlashAttentionScoreV2、FlashAttentionScoreGradV2融合算子支持该参数。

- **CMAKE_INSTALL_PREFIX**：编译完成后，自定义算子包生成目录，未设置时默认生成在根目录的output目录。示例如下：

   ```shell
   -DCMAKE_INSTALL_PREFIX="/home/code/custom_out"
   ```


## 单元测试编译执行（UT）

- **TESTS_UT_OPS_TEST**：执行单元UT测试，支持指定算子；未配置时不执行UT测试，示例如下：

   ```shell
   # 执行所有UT测试
   -DTESTS_UT_OPS_TEST=ALL

   # 执行flash_attention_score算子的UT测试
   -DTESTS_UT_OPS_TEST="flash_attention_score"

   # 执行flash_attention_score和flash_attention_score_grad算子的UT测试
   -DTESTS_UT_OPS_TEST="flash_attention_score;flash_attention_score_grad"
   ```


## 示例工程编译执行（Examples）

- **TESTS_EXAMPLE_OPS_TEST**：执行example用例测试，支持指定具体example用例；未配置时不执行example用例测试，示例如下：

   ```shell
   # 执行所有example用例
   -DTESTS_EXAMPLE_OPS_TEST=ALL

   # 执行单个example用例test_flash_attention_score
   -DTESTS_EXAMPLE_OPS_TEST="test_flash_attention_score"
   ```
