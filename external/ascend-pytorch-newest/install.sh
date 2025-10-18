# 代码clone下来之后需要先执行这个脚本
# bash get_gitmodules.sh
# ref: https://www.hiascend.com/document/detail/zh/Pytorch/710/configandinstg/instg/insg_0005.html
# gcc 10.2.0 & cmake 3.31.0 以上

# 编译脚本
rm -rf build/
rm -rf dist/
# 如果提示Cmake版本太高 需要使用 ：cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
bash ci/build.sh --python=3.10
pip uninstall torch_npu -y #每次装包都要做一次卸载 不然会有问题！！！！
pip3 install --upgrade dist/*.whl #torch_npu-2.1.0.post13-cp38-cp38-linux_aarch64.whl