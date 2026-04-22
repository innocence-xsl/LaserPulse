"""
==============================================================================
文件名称: pump_dynamics.py
所属部门: physics/amplification (物理部/放大模型)
主要功能: 泵浦动态模拟
代码解读: 
    模拟泵浦阶段的粒子数积累
    考虑基态耗尽(Ground State Bleaching)的动态泵浦过程
==============================================================================
"""

import numpy as np
from scipy.integrate import odeint
from core.dataclasses_def import CrystalParameters, PumpParameters, PhysicalConstants

# def pump_process(
#     time_duration: float,
#     N_upper_start: float,
#     cry_params: CrystalParameters,
#     pump_params: PumpParameters,
#     consts: PhysicalConstants,
#     pump_polarization: str,
#     pump_area: float,
#     L_total: float
# ) -> float:
#     """
#     参数:
#         time_duration: 泵浦持续时间 [s]
#         N_upper_start: 初始上能级粒子密度 [m^-3]
#         cry_params: 晶体参数对象 (CrystalParameters)
#         pump_params: 泵浦参数对象 (PumpParameters)
#         consts: 物理常数对象 (PhysicalConstants)
#         pump_polarization: 泵浦光偏振方向 ('pi'/'sigma')
#         pump_area: 泵浦光斑面积 [m²]
#         L_total: 晶体总长度 [m]
    
#     返回:
#         current_N_density: 泵浦后上能级粒子数密度 [m^-3]
#     """
#     # 1. 基础物理量准备
#     V_crystal = pump_area * L_total  # 晶体有效体积
    
#     # 泵浦光子输入速率 (Photons per second) = Power / PhotonEnergy
#     photon_E = consts.photon_energy(pump_params.lambda_p)
#     photons_in_per_sec = pump_params.P_pump_avg / photon_E if photon_E > 1e-20 else 0.0
    
#     # 吸收/发射截面 (对应泵浦波长+偏振)
#     sigma_abs = cry_params.get_sigma_at(pump_params.lambda_p, type='abs', polarization=pump_polarization)
#     sigma_em = cry_params.get_sigma_at(pump_params.lambda_p, type='em', polarization=pump_polarization)
#     N_doping = cry_params.N_doping
#     M_pump = pump_params.M_p
    
#     # 2. 时间迭代 (Time Stepping)
#     steps = 50 
#     dt = time_duration / steps
#     current_N_density = N_upper_start
    
#     for _ in range(steps):
#         # A. 计算当前的基态粒子密度 (Ground State Population)
#         N_ground = max(N_doping - current_N_density, 0.0)
        
#         # B. 计算瞬时吸收率 (Beer-Lambert Law)
#         effective_length = L_total * M_pump
#         alpha_net = (sigma_abs * N_ground) - (sigma_em * current_N_density)
#         alpha_net = max(alpha_net, 0.0)  # 泵浦光被漂白后不再吸收
#         optical_depth = alpha_net * effective_length
#         optical_depth = 100 if optical_depth > 100 else optical_depth  # 防止指数溢出
        
#         absorbance = 1.0 - np.exp(-optical_depth)
        
#         # C. 计算这段 dt 时间内被晶体“截获”的总光子数
#         absorbed_photons = photons_in_per_sec * dt * absorbance * cry_params.mode_overlap_efficiency
        
#         # D. 计算自发辐射损耗掉的粒子数
#         total_N_particles = current_N_density * V_crystal
#         decayed_particles = total_N_particles * (dt / cry_params.tau_f) if cry_params.tau_f > 1e-20 else 0.0
        
#         # E. 更新粒子数密度
#         delta_N_density = (absorbed_photons - decayed_particles) / V_crystal if V_crystal > 1e-20 else 0.0
#         current_N_density += delta_N_density
        
#         # F. 物理边界限制
#         current_N_density = np.clip(current_N_density, 0.0, N_doping)
    
#     return current_N_density

