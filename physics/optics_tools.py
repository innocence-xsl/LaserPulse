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
    coeffs = np.loadtxt(coeffs_file, skiprows=1)
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