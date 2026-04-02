import sys
import os

# 计算项目根目录（即 src 和 test 的父目录）
# __file__ 指当前文件（_test_setup.py）的路径
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 将根目录添加到 Python 搜索路径，确保能找到 src 目录
if root_dir not in sys.path:
    sys.path.append(root_dir)