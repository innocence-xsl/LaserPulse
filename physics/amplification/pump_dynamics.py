"""
==============================================================================
文件名称: pump_dynamics.py
所属部门: physics/amplification (物理部/放大模型)
主要功能: 泵浦动态模拟
代码解读:
    模拟泵浦阶段的上能级粒子数积累。
    当前模型考虑：泵浦吸收、基态漂白、泵浦受激发射、自发辐射损耗。
==============================================================================
"""

import numpy as np

from core.dataclasses_def import CrystalParameters, PumpParameters, PhysicalConstants


def pump_process(
    time_duration: float,
    N_upper_start: float,
    cry_params: CrystalParameters,
    pump_params: PumpParameters,
    consts: PhysicalConstants,
    pump_polarization: str,
    pump_area: float,
    L_total: float,
) -> float:
    """
    计算泵浦持续 time_duration 后的上能级粒子数密度。

    参数:
        time_duration: 泵浦持续时间 [s]
        N_upper_start: 初始上能级粒子密度 [m^-3]
        cry_params: 晶体参数对象
        pump_params: 泵浦参数对象
        consts: 物理常数对象
        pump_polarization: 泵浦光偏振方向 ('pi' / 'sigma')
        pump_area: 泵浦光斑面积 [m^2]
        L_total: 晶体总作用长度 [m]

    返回:
        N_upper_new: 泵浦后的上能级粒子数密度 [m^-3]
    """
    if time_duration <= 0:
        return float(np.clip(N_upper_start, 0.0, cry_params.N_doping))

    V_crystal = max(pump_area * L_total, 1e-30)
    photon_E = consts.photon_energy(pump_params.lambda_p)
    photons_in_per_sec = pump_params.P_pump_avg / photon_E if photon_E > 0 else 0.0

    sigma_abs = cry_params.get_sigma_at(pump_params.lambda_p, type="abs", polarization=pump_polarization)
    sigma_em = cry_params.get_sigma_at(pump_params.lambda_p, type="em", polarization=pump_polarization)
    N_doping = cry_params.N_doping
    M_pump = getattr(pump_params, "M_p", 1)

    steps = max(10, int(np.ceil(time_duration / 1e-6)))
    steps = min(steps, 2000)  # 防止过细时间步拖慢正式入口
    dt = time_duration / steps

    current_N_density = float(np.clip(N_upper_start, 0.0, N_doping))

    for _ in range(steps):
        N_ground = max(N_doping - current_N_density, 0.0)

        # 泵浦净吸收系数；若漂白后净吸收为负，则视为不再吸收泵浦。
        effective_length = L_total * M_pump
        alpha_net = (sigma_abs * N_ground) - (sigma_em * current_N_density)
        alpha_net = max(alpha_net, 0.0)

        optical_depth = np.clip(alpha_net * effective_length, 0.0, 100.0)
        absorbance = 1.0 - np.exp(-optical_depth)

        absorbed_photons = (
            photons_in_per_sec
            * dt
            * absorbance
            * getattr(cry_params, "mode_overlap_efficiency", 1.0)
            * getattr(pump_params, "duty_cycle", 1.0)
        )

        total_upper_particles = current_N_density * V_crystal
        decayed_particles = total_upper_particles * (dt / cry_params.tau_f) if cry_params.tau_f > 0 else 0.0

        delta_N_density = (absorbed_photons - decayed_particles) / V_crystal
        current_N_density = np.clip(current_N_density + delta_N_density, 0.0, N_doping)

    return float(current_N_density)
