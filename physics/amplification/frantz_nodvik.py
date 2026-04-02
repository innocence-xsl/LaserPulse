"""
==============================================================================
文件名称: frantz_nodvik.py
所属部门: Physics (物理部)
主要功能: 基于Frantz-Nodvik模型的均匀加宽介质单程放大计算
代码解读: 
    单程放大模型 (针对均匀加宽介质)
    逻辑：
    1. 计算总输入能量通量 (Total Fluence)。
    2. 计算加权平均的发射/吸收截面 (Effective Sigma)。
    3. 使用标量版 Frantz-Nodvik 计算总提取能量。
    4. 根据总提取能量，反推平均剩余反转粒子数。
    5. 用修正后的粒子数重新计算光谱增益形状。

==============================================================================
"""
import numpy as np
from core.dataclasses_def import CrystalParameters, PhysicalConstants

def amplify_single_pass(
    current_spectrum_J: np.ndarray,
    N_upper: float,
    cry_params: CrystalParameters,
    seed_params,
    consts: PhysicalConstants,
    sigma_em: np.ndarray,
    sigma_abs: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    参数:
        current_spectrum_J: 输入光谱能量通量分布 [J/m²]
        N_upper: 初始上能级粒子数密度 [m^-3]
        cry_params: 晶体参数对象 (CrystalParameters)
        seed_params: 种子光参数对象
        consts: 物理常数对象 (PhysicalConstants)
        sigma_em: 发射截面网格 [m²] (对应偏振态)
        sigma_abs: 吸收截面网格 [m²] (对应偏振态)
    
    返回:
        spectrum_out_J: 放大后的光谱能量通量分布 [J/m²]
        N_upper_new: 放大后剩余上能级粒子数密度 [m^-3]
    """
    L = cry_params.thickness * cry_params.num_disks  # 晶体总长
    area = seed_params.seed_area
    
    # 1. 计算总通量 (Scalar)
    E_in_total = np.sum(current_spectrum_J)
    J_in_total = E_in_total / area if area > 1e-20 else 0.0
    
    if J_in_total < 1e-20:  # 避免除零
        return current_spectrum_J, N_upper

    # 2. 计算光谱加权后的有效截面 (Effective Cross-sections)
    if E_in_total > 1e-20:
        spectral_weights = current_spectrum_J / E_in_total
        sigma_em_eff = np.average(sigma_em, weights=spectral_weights)
        sigma_abs_eff = np.average(sigma_abs, weights=spectral_weights)
    else:
        # 能量太低时，默认用中心波长的截面
        idx = len(cry_params.wavelength_grid) // 2
        sigma_em_eff = sigma_em[idx]
        sigma_abs_eff = sigma_abs[idx]
    
    # 3. 计算有效饱和通量 (Effective Saturation Fluence)
    h = consts.h
    c = consts.c
    lambda_center = seed_params.lambda_s
    J_sat_eff = (h * c) / (lambda_center * (sigma_em_eff + sigma_abs_eff))
    
    # 4. 计算小信号增益 (Small Signal Gain) - 对应当前 N_upper
    N_lower = cry_params.N_doping - N_upper
    g0_coeff = (N_upper * sigma_em_eff) - (N_lower * sigma_abs_eff)
    G0_total = np.exp(g0_coeff * L)
    
    # 5. 使用标量 Frantz-Nodvik 计算总输出通量
    exp_arg = J_in_total / J_sat_eff
    exp_arg = 100 if exp_arg > 100 else exp_arg  # 防止指数溢出
    
    term_exp = np.exp(exp_arg)
    term_log = 1.0 + G0_total * (term_exp - 1.0)
    J_out_total = J_sat_eff * np.log(term_log)
    
    # 6. 计算放大后的总能量 & 粒子数消耗
    E_out_total = J_out_total * area
    E_extracted = E_out_total - E_in_total
    
    # 反推消耗了多少粒子数 density
    photon_E = (h * c) / lambda_center
    V_crystal = seed_params.seed_area * L  # 晶体体积
    consumed_density = E_extracted / (photon_E * V_crystal) if (photon_E * V_crystal) > 1e-20 else 0.0
    
    N_upper_new = max(N_upper - consumed_density, 0.0)
    
    # 7. 重构光谱形状 (Spectral Reshaping)
    N_avg = (N_upper + N_upper_new) / 2.0 
    N_lower_avg = cry_params.N_doping - N_avg
    
    # 计算针对每个波长的增益系数
    gain_profile = (N_avg * sigma_em) - (N_lower_avg * sigma_abs)
    G_lambda = np.exp(gain_profile * L)
    
    # 输出光谱 = 输入光谱 * 增益谱
    spectrum_out_J = current_spectrum_J * G_lambda
    
    # 归一化能量 (强制能量守恒，修正由于 N_avg 近似带来的微小误差)
    calc_E_out = np.sum(spectrum_out_J)
    if calc_E_out > 1e-20:
        correction_factor = E_out_total / calc_E_out
        spectrum_out_J *= correction_factor
        
    return spectrum_out_J, N_upper_new