export DEBUG=0
# 设置华为日志相关的环境变量
export ASCEND_LOG_LEVEL=DEBUG
export ASCEND_GLOBAL_LOG_LEVEL=DEBUG
export ASCEND_OPP_LOG_LEVEL=DEBUG
export ASCEND_TILING_LOG_LEVEL=DEBUG

# 设置设备相关环境变量
export ASCEND_DEVICE_ID=2
export ASCEND_RT_VISIBLE_DEVICES=2

# 设置华为日志输出到控制台
export ASCEND_LOG_TO_CONSOLE=1
export ASCEND_LOG_TO_FILE=0

# 设置GLOG相关环境变量
export GLOG_v=4
export GLOG_logtostderr=1

# 设置库路径 - 确保我们的库在前面

source /usr/local/Ascend/ascend-toolkit/set_env.sh

# export LD_LIBRARY_PATH=/root/workspace/operators/AscendIvfAttention/cann-ops-adv/build:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}

# 显示当前设置
echo "=== 日志环境变量设置 ==="
echo "ASCEND_LOG_LEVEL: $ASCEND_LOG_LEVEL"
echo "ASCEND_GLOBAL_LOG_LEVEL: $ASCEND_GLOBAL_LOG_LEVEL"
echo "ASCEND_OPP_LOG_LEVEL: $ASCEND_OPP_LOG_LEVEL"
echo "ASCEND_TILING_LOG_LEVEL: $ASCEND_TILING_LOG_LEVEL"
echo "ASCEND_DEVICE_ID: $ASCEND_DEVICE_ID"
echo "ASCEND_VISIBLE_DEVICES: $ASCEND_VISIBLE_DEVICES"
echo "ASCEND_LOG_TO_CONSOLE: $ASCEND_LOG_TO_CONSOLE"
echo "ASCEND_LOG_TO_FILE: $ASCEND_LOG_TO_FILE"
echo "GLOG_v: $GLOG_v"
echo "GLOG_logtostderr: $GLOG_logtostderr"
echo "=========================="

rm -rf build/

mkdir -p build && cd build

rm -f ./bin/opapi_test

cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE

make

# echo "=== 运行测试程序 ==="
# unset PERF_TEST  # 或 export PERF_TEST=0
# stdbuf -oL ./bin/opapi_test 2>&1 | tee build.log

# # 性能测试模式
# export PERF_TEST=1
# # msprof op simulator --soc-version=Ascend910B2 --output=../output_data ./bin/opapi_test # 模拟器测试
# msprof op  ./bin/opapi_test  2>&1 | tee build.log # 算子速度测试


# 上传代码确认
# unset PERF_TEST  # 或 export PERF_TEST=0
# stdbuf -oL ./bin/opapi_test 2>&1 | tee build.log

# 测算子性能 默认到0卡上了 暂时没找到改的地方
export PERF_TEST=1
msprof op  ./bin/opapi_test  2>&1 | tee build.log # 算子速度测试
# msprof op simulator --soc-version=Ascend910B2 --output=../output_data ./bin/opapi_test # 模拟器测试
