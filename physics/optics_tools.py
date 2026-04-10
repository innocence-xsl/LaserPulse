"""
==============================================================================
文件名称: optics_tools.py
所属部门: Physics (物理核心计算模块)
主要功能: 激光脉冲相关物理计算
代码解读: 
    处理能谱/功率谱密度转换、Sellmeier方程计算折射率参数计算。
==============================================================================
"""

import numpy as np
import scipy.constants as const
from typing import Tuple

def get_ESD_and_PSD(lambda_window: np.ndarray, spectrum: np.ndarray, repetition_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    从实数光谱计算能量谱密度 (ESD) 和 功率谱密度 (PSD)
    
    返回:
        ESD: 能量谱密度 [J/m]
        PSD: 功率谱密度 [W/nm]
    """
    # 转换为 J/m
    energy_spectral_density = spectrum * 2 * np.pi * const.c / (lambda_window**2)
    # 转换为 W/nm (乘重频并单位换算)
    power_spectral_density = energy_spectral_density * repetition_rate * 1e-6
    
    return energy_spectral_density, power_spectral_density

def sellmeier_index(lambda_window: np.ndarray, coeffs_file: str) -> np.ndarray:
    """
    基于 Sellmeier 方程计算随波长变化的材料折射率
    """
    lw = 1e6 * lambda_window # 转换为微米
    coeffs = np.loadtxt(coeffs_file, skiprows=2, delimiter=',', encoding='utf-8')
    n_sq = np.ones_like(lw)
    
    for B, C in coeffs:
        n_sq += B * (lw**2) / (lw**2 - C**2)
        
    return np.sqrt(n_sq)

def get_taylor_coeffs_from_beta2(beta_2: np.ndarray, grid_omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """从群速度色散(beta2)曲线中提取高阶色散泰勒系数"""
    # 拟合最高到9阶
    tc = np.polyfit(grid_omega, beta_2, 9)[::-1]
    taylors = np.zeros(len(tc) + 2)
    taylors[2:] = tc # 前两阶(beta0, beta1)不影响包络形貌，设为0
    
    # 注意：这里如果需要计算完整 beta，需要调用上面的 taylor_expansion
    # beta = taylor_expansion(taylors, grid_omega)
    return taylors, tc

def get_omega_array_from_time(t_array: np.ndarray) -> np.ndarray:
    """
    根据给定的时间网格，生成对应的角频率网格，并进行 fftshift 排序。
    这样输出的 omega_array 是单调递增的，零频在中心。
    """
    N = len(t_array)
    dt = t_array[1] - t_array[0] 
    
    # 1. np.fft.fftfreq 生成标准的物理频率 (Hz) 网格
    freq_array = np.fft.fftfreq(N, d=dt)
    
    # 2. 乘以 2*pi 转换为角频率 omega
    omega_array = 2 * np.pi * freq_array
    
    # 3. 将零频移动到数组中心，符合物理直觉
    omega_array_shifted = np.fft.fftshift(omega_array)
    
    return omega_array_shifted

def get_spectrum_from_time_domain(t_array, E_t, lambda_range=(900e-9, 1100e-9)):
    """
    从时域获取频谱的函数。
    输入时域电场，输出可以直接用于画图的波长轴和光谱强度。
    lambda_range: 可选参数，指定要绘制的波长范围，默认 (900nm, 1100nm)，记得实际画图时进行调节！
    """
    dt = t_array[1] - t_array[0]
    E_omega = np.fft.fftshift(np.fft.fft(E_t))
    omega_array = np.fft.fftshift(np.fft.fftfreq(len(t_array), d=dt)) * 2 * np.pi
    
    # 避免除以 0，只取正频率部分
    valid_idx = omega_array > 0
    lambda_array = 2 * np.pi * const.c / omega_array[valid_idx]
    I_omega = np.abs(E_omega[valid_idx])**2
    
    # 截取画图感兴趣的波长范围
    plot_idx = (lambda_array > lambda_range[0]) & (lambda_array < lambda_range[1])
    
    return lambda_array[plot_idx], I_omega[plot_idx]

def get_spatial_profile(w, I_peak, r_max_ratio=3, points=500):
    """
    获取空间剖面的函数。
    输入束腰和峰值光强，返回径向坐标轴和对应的光强分布。
    """
    r_array = np.linspace(-r_max_ratio * w, r_max_ratio * w, points)
    I_r = I_peak * np.exp(-2 * (r_array / w)**2)
    return r_array, I_r