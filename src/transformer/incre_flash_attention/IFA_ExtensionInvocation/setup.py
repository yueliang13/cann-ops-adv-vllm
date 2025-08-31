import os
import glob
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension

import torch_npu
from torch_npu.utils.cpp_extension import NpuExtension

PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))
ASCEND_PATH = os.getenv('ASCEND_HOME_PATH', '/usr/local/Ascend/ascend-toolkit/latest')
USE_NINJA = os.getenv('USE_NINJA') == '1'
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

source_files = glob.glob(os.path.join(BASE_DIR, "csrc", "*.cpp"), recursive=True)

# 参考 CMakeLists.txt 的库路径配置
library_dirs = [
    os.path.join(ASCEND_PATH, "lib64"),  # 主要库文件位置
    os.path.join(PYTORCH_NPU_INSTALL_PATH, "lib"),
    os.path.join(ASCEND_PATH, "opp/vendors/customize/op_api/lib"),  # 添加自定义算子库路径
]

# --- 编译参数 (最关键的部分) ---
extra_compile_args = {
    'cxx': [
        '-std=c++17',
        '-fPIC',
        '-Wno-unused-variable' # 忽略未使用的变量警告
    ]
}

exts = []
ext = NpuExtension(
    name="custom_ops_lib",
    sources=source_files,
    include_dirs=[
        *torch.utils.cpp_extension.include_paths(),
        os.path.join(PYTORCH_NPU_INSTALL_PATH, "include/third_party/acl/inc"),
        os.path.join(ASCEND_PATH, "include"),  # 参考 CMakeLists.txt
        os.path.join(ASCEND_PATH, "include/aclnn"),  # 参考 CMakeLists.txt
        os.path.join(ASCEND_PATH, "acllib/include"),
        os.path.join(ASCEND_PATH, "opp/vendors/customize/op_api/include"),
    ],
    library_dirs=library_dirs,
    # 参考 CMakeLists.txt 的库配置
    libraries=['ascendcl', 'nnopbase','ascendalog', 'cust_opapi'],
    extra_compile_args=extra_compile_args,
    extra_link_args=[
        f'-Wl,-rpath,{os.path.join(ASCEND_PATH, "lib64")}',
        f'-Wl,-rpath,{os.path.join(ASCEND_PATH, "opp/vendors/customize/op_api/lib")}',  # 添加运行时路径

    ],
)
exts.append(ext)

setup(
    name="custom_ops",
    version='1.0',
    keywords='custom_ops',
    ext_modules=exts,
    packages=find_packages(),
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=USE_NINJA)},
)