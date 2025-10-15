#!/bin/bash

# ==============================================================================
# 脚本说明 (V3 - 包含 torchair 的内部依赖):
# 这个脚本用于手动克隆项目所需的依赖库及其所有的嵌套子模块。
#
# 工作流程:
# 1. 克隆主依赖仓库。
# 2. 进入该仓库，递归地初始化并下载它所有的子模块。
# 3. (可选) 递归地删除所有 .git 目录和 .gitmodules 文件，使整个代码树
#    成为主项目的一部分，而不是 Git 子模块。
#
# 使用方法:
# 1. 将此脚本放置在 ascend-pytorch-newest 项目的根目录下。
# 2. 运行 chmod +x download_dependencies.sh
# 3. 运行 ./download_dependencies.sh
# ==============================================================================

# 设置颜色代码
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 定义要下载的依赖库列表
# 格式: "本地路径 Git仓库URL [分支或标签]"
dependencies=(
    # ascend-pytorch-newest 的直接依赖
    "third_party/googletest https://gitee.com/mirrors/googletest.git"
    "third_party/Tensorpipe https://gitee.com/ascend/Tensorpipe.git"
    "third_party/fmt https://gitee.com/mirrors/fmt.git"
    "third_party/nlohmann https://gitee.com/mirrors/nlohmann-json.git"
    # "third_party/op-plugin https://gitee.com/ascend/op-plugin.git 7.1.0"

    # torchair 的内部依赖 (路径已拼接)
    "third_party/torchair/torchair/third_party/secure_c https://gitee.com/openeuler/libboundscheck.git"
    "third_party/torchair/torchair/third_party/googletest https://gitee.com/mirrors/googletest.git"
)

echo -e "${GREEN}开始下载项目依赖 (包含所有嵌套子模块)...${NC}"

# 循环处理每个依赖
for item in "${dependencies[@]}"; do
    # 解析路径、URL和分支
    path=$(echo "$item" | awk '{print $1}')
    url=$(echo "$item" | awk '{print $2}')
    branch=$(echo "$item" | awk '{print $3}')

    echo -e "\n${YELLOW}--- 正在处理: $path ---${NC}"

    # 1. 清理可能存在的旧目录
    if [ -d "$path" ]; then
        echo "发现旧目录，正在清理: $path"
        rm -rf "$path"
    fi

    # 2. 克隆主仓库
    echo "正在从 $url 克隆主仓库..."
    git clone "$url" "$path"
    if [ $? -ne 0 ]; then
        echo -e "${RED}错误: 克隆 $url 失败。请检查网络和URL。${NC}"
        exit 1
    fi

    # 使用子shell ( ... ) 来执行目录切换，避免影响主脚本的当前路径
    (
        echo "进入 $path"
        cd "$path" || exit 1

        # 2a. 如果指定了分支或标签，进行切换
        if [ -n "$branch" ]; then
            echo "正在切换到分支/标签: $branch"
            git checkout "$branch"
            if [ $? -ne 0 ]; then
                echo -e "${RED}错误: 切换到 $branch 失败。${NC}"
                exit 1
            fi
        fi

        # 2b. 递归初始化并更新所有子模块
        echo "正在递归更新所有嵌套子模块..."
        git submodule update --init --recursive
        if [ $? -ne 0 ]; then
            echo -e "${RED}错误: 更新 $path 的子模块失败。${NC}"
            exit 1
        fi
    )

    # 检查子shell中的操作是否成功
    if [ $? -ne 0 ]; then
        exit 1
    fi

    echo -e "${GREEN}成功处理: $path (所有代码已下载)${NC}"
done

echo -e "\n${GREEN}所有依赖已成功下载！${NC}"
echo -e "${YELLOW}下一步: 您可以选择性地运行下面的命令来解绑Git关系，使依赖成为普通文件夹。${NC}"