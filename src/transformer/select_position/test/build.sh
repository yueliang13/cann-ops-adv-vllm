#!/bin/bash

# ComputeCentVec算子测试构建脚本
# 作者: ComputeCent测试团队
# 日期: 2024

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查环境变量
check_environment() {
    print_info "检查环境变量..."
    
    # 检查ASCEND相关环境变量
    if [ -z "$ASCEND_CUSTOM_PATH" ] && [ -z "$ASCEND_HOME_PATH" ]; then
        print_warning "未设置ASCEND_CUSTOM_PATH或ASCEND_HOME_PATH环境变量"
        print_info "将使用默认路径: /usr/local/Ascend/ascend-toolkit/latest"
    else
        if [ -n "$ASCEND_CUSTOM_PATH" ]; then
            print_info "使用ASCEND_CUSTOM_PATH: $ASCEND_CUSTOM_PATH"
        fi
        if [ -n "$ASCEND_HOME_PATH" ]; then
            print_info "使用ASCEND_HOME_PATH: $ASCEND_HOME_PATH"
        fi
    fi
    
    # 检查CANN环境
    if ! command -v acl &> /dev/null; then
        print_warning "未找到acl命令，请确保CANN环境已正确配置"
    fi
    
    # 检查cmake
    if ! command -v cmake &> /dev/null; then
        print_error "未找到cmake命令，请先安装cmake"
        exit 1
    fi
    
    # 检查g++
    if ! command -v g++ &> /dev/null; then
        print_error "未找到g++命令，请先安装g++"
        exit 1
    fi
}

# 清理构建目录
clean_build() {
    print_info "清理构建目录..."
    
    if [ -d "build" ]; then
        rm -rf build
        print_success "已清理build目录"
    fi
    
    if [ -d "bin" ]; then
        rm -rf bin
        print_success "已清理bin目录"
    fi
}

# 创建构建目录
create_build_dir() {
    print_info "创建构建目录..."
    
    mkdir -p build
    mkdir -p bin
    print_success "构建目录创建完成"
}

# 配置CMake
configure_cmake() {
    print_info "配置CMake..."
    
    cd build
    
    # 根据环境变量选择ASCEND路径
    local ascend_path=""
    if [ -n "$ASCEND_CUSTOM_PATH" ]; then
        ascend_path="$ASCEND_CUSTOM_PATH"
    elif [ -n "$ASCEND_HOME_PATH" ]; then
        ascend_path="$ASCEND_HOME_PATH"
    else
        ascend_path="/usr/local/Ascend/ascend-toolkit/latest"
    fi
    
    # 检查ASCEND路径是否存在
    if [ ! -d "$ascend_path" ]; then
        print_error "ASCEND路径不存在: $ascend_path"
        print_error "请检查环境变量设置或安装CANN工具包"
        exit 1
    fi
    
    print_info "使用ASCEND路径: $ascend_path"
    
    # 配置CMake
    cmake .. \
        -DCMAKE_BUILD_TYPE=Debug \
        -DASCEND_PATH="$ascend_path" \
        -DCMAKE_VERBOSE_MAKEFILE=ON
    
    if [ $? -eq 0 ]; then
        print_success "CMake配置成功"
    else
        print_error "CMake配置失败"
        exit 1
    fi
    
    cd ..
}

# 编译项目
build_project() {
    print_info "开始编译项目..."
    
    cd build
    
    # 使用多线程编译
    local cpu_count=$(nproc)
    print_info "使用 $cpu_count 个线程进行编译"
    
    make -j$cpu_count
    
    if [ $? -eq 0 ]; then
        print_success "编译成功"
    else
        print_error "编译失败"
        exit 1
    fi
    
    cd ..
}

# 检查可执行文件
check_executable() {
    print_info "检查可执行文件..."
    
    if [ -f "bin/compute_cent_test" ]; then
        print_success "可执行文件生成成功: bin/compute_cent_test"
        
        # 显示文件信息
        ls -lh bin/compute_cent_test
        
        # 检查文件类型
        file bin/compute_cent_test
    else
        print_error "可执行文件生成失败"
        exit 1
    fi
}

# 运行测试（可选）
run_test() {
    if [ "$1" = "--run-test" ]; then
        print_info "运行测试..."
        
        if [ -f "bin/compute_cent_test" ]; then
            cd bin
            ./compute_cent_test
            cd ..
        else
            print_error "可执行文件不存在，无法运行测试"
        fi
    fi
}

# 主函数
main() {
    print_info "开始构建ComputeCentVec算子测试项目..."
    print_info "当前目录: $(pwd)"
    
    # 检查环境
    check_environment
    
    # 清理构建目录
    clean_build
    
    # 创建构建目录
    create_build_dir
    
    # 配置CMake
    configure_cmake
    
    # 编译项目
    build_project
    
    # 检查可执行文件
    check_executable
    
    # 运行测试（如果指定）
    run_test "$1"
    
    print_success "构建完成！"
    print_info "可执行文件位置: bin/compute_cent_test"
    print_info "运行测试命令: ./bin/compute_cent_test"
}

# 显示帮助信息
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --run-test    构建完成后自动运行测试"
    echo "  --clean       清理构建目录后退出"
    echo "  --help        显示此帮助信息"
    echo ""
    echo "Examples:"
    echo "  $0              # 仅构建项目"
    echo "  $0 --run-test   # 构建并运行测试"
    echo "  $0 --clean      # 清理构建目录"
}

# 处理命令行参数
case "$1" in
    --help)
        show_help
        exit 0
        ;;
    --clean)
        clean_build
        exit 0
        ;;
    --run-test|"")
        main "$1"
        ;;
    *)
        print_error "未知参数: $1"
        show_help
        exit 1
        ;;
esac 