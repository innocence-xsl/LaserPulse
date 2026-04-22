"""
==============================================================================
文件名称: component_interface.py
所属部门: physics/amplification (物理部/放大模型)
主要功能: 组件接口
代码解读: 
    定义组件接口，用于与外部组件交互,如放大器和泵浦过程,
    主要包含：
    1. propagate_pulse_broadband: 流水线Pulse传播接口，将Pulse转换为光谱通量→放大→还原Pulse
    2. trigger_pump: 流水线统一泵浦接口，触发泵浦过程并更新粒子数

==============================================================================
"""
import numpy as np
from core.pulse import Pulse
# from .frantz_nodvik import amplify_single_pass
from physics.amplification.gain import amplify_spectral_broadband

def propagate_pulse_broadband(
    pulse: Pulse,
    N_upper_start: float,
    N_total: float,
    sigma_em_grid: np.ndarray,
    sigma_abs_grid: np.ndarray,
    thickness: float,
    mode_area: float,
    num_slices: float = 1.0) -> tuple[Pulse, float, float]:
    """
    流水线 Pulse 传播接口 (宽带适配版)
    参数：
    pulse: 输入的 Pulse 对象，包含时域和频域状态
    N_upper_start: 初始上能级粒子数密度 [m^-3]
    N_total: 总粒子数密度 [m^-3]
    sigma_em_grid: 发射截面网格 (对应偏振)
    sigma_abs_grid: 吸收截面网格 (对应偏振)
    thickness: 晶体厚度 [m]
    mode_area: 模场面积 [m^2]
    num_slices: 切片数量
    """
    
    # 1. 拆包 (Unpacking)：从 Pulse 身上卸下所需数据
    # 提取光谱强度 (电场复振幅的模平方)
    spectrum_in = np.abs(pulse.A_f)**2
    # 提取相位 (为了算完增益后还能复原)
    phase_in = np.angle(pulse.A_f)
    # 提取物理波长网格
    lambda_grid = pulse.grid.lambda_window
    
    # 2. 加工 (Execution)：将纯数组送入底层物理机床
    spectrum_out, N_upper_end, current_gain = amplify_spectral_broadband(
        spectrum_in_intensity=spectrum_in,
        lambda_grid=lambda_grid,
        N_upper_start=N_upper_start,
        N_total=N_total,
        sigma_em_grid=sigma_em_grid,
        sigma_abs_grid=sigma_abs_grid,
        thickness=thickness,
        mode_area=mode_area,
        num_slices=num_slices
    )
    
    # 3. 重新打包 (Repacking)：更新 Pulse 状态
    pulse.A_f = np.sqrt(spectrum_out) * np.exp(1j * phase_in)
    
    # 同步更新时域状态 (调用 Pulse 自身的方法)
    pulse.to_time_domain()
    
    # 最终返回打包好的 Pulse 产品，以及需要传递的物理状态
    return pulse, N_upper_end, current_gain

# # def trigger_pump(
#     dt: float,
#     N_upper_start: float,
#     cry_params,
#     pump_params,
#     consts,
#     pump_polarization: str,
#     pump_area: float,
#     L_total: float) -> float:
#     """
#     参数:
#         dt: 泵浦持续时间 [s]
#         N_upper_start: 初始上能级粒子数密度 [m^-3]
#         cry_params: 晶体参数对象
#         pump_params: 泵浦参数对象
#         consts: 物理常数对象
#         pump_polarization: 泵浦偏振方向
#         pump_area: 泵浦光斑面积 [m²]
#         L_total: 晶体总长度 [m]
    
#     返回:
#         N_upper_new: 泵浦后上能级粒子数密度 [m^-3]
#     """
#     return pump_process(
#         time_duration=dt,
#         N_upper_start=N_upper_start,
#         cry_params=cry_params,
#         pump_params=pump_params,
#         consts=consts,
#         pump_polarization=pump_polarization,
#         pump_area=pump_area,
#         L_total=L_total
#     )

