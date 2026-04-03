"""
==============================================================================
文件名称: math_tools.py
所属部门: Utils (工具模块)
主要功能: 通用数学与数组操作工具
代码解读: 
    提供寻址、分布宽度计算(FWHM/1e2)、数组对半交换、FFT卷积以及泰勒展开等通用算法。
==============================================================================
"""

import math
import numpy as np
from typing import Tuple, Union, Iterable

def find_nearest(search_value: float, container: Union[np.ndarray, list]) -> Tuple[int, float]:
    """在数组中查找最接近目标值的元素及其索引"""
    container = np.asarray(container)
    index = int(np.argmin(np.abs(container - search_value)))
    return index, container[index]

def get_width(axis: np.ndarray, data: np.ndarray, meas_point: float = 0.5) -> float:
    """
    计算分布数据的宽度
    :param meas_point: 测量点比例，默认 0.5 即计算 FWHM (半高全宽)
    """
    points = len(axis)
    half_points = int(points / 2)
    maximum = np.amax(data)
    idx_maximum = int(np.argmax(data))
    
    # 确保测量时数据中心对齐
    data_rolled = np.roll(data, -idx_maximum + half_points)

    idx_1, _ = find_nearest(meas_point * maximum, data_rolled[0:half_points])
    idx_2, _ = find_nearest(meas_point * maximum, data_rolled[half_points:])
    idx_2 += half_points
    
    return float(np.abs(axis[idx_2] - axis[idx_1]))

def get_FWe2M(axis: np.ndarray, data: np.ndarray) -> float:
    """计算 1/e^2 处的全宽 (通常用于高斯光束宽度)"""
    return get_width(axis, data, meas_point=np.exp(-2))

def swap_halves(arr: np.ndarray, axis: int = -1) -> np.ndarray:
    """沿指定轴交换数组的两半 (类似于手动实现 fftshift)"""
    pts = arr.shape[axis]
    half_1 = arr.take(indices=range(0, int(pts / 2)), axis=axis)
    half_2 = arr.take(indices=range(int(pts / 2), pts), axis=axis)
    return np.append(half_2, half_1, axis=axis)

def fft_convolve(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """利用卷积定理进行快速傅里叶卷积 (比直接卷积快得多)"""
    if arr1.shape != arr2.shape:
        raise ValueError("arr1 和 arr2 必须具有相同的形状。")
    return np.fft.ifft(np.fft.fft(arr1) * np.fft.fft(arr2))

def taylor_expansion(coeffs: Iterable[float], axis: np.ndarray, axis_centre: float = 0.0) -> np.ndarray:
    """计算泰勒展开序列"""
    TE = np.zeros_like(axis, dtype=np.complex128)
    for i, tc in enumerate(coeffs):
        TE += tc * (axis - axis_centre)**i / math.factorial(i)
    return TE