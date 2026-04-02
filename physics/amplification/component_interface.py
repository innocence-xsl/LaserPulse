"""
==============================================================================
文件名称: component_interface.py
所属部门: Physics (物理部)
主要功能: 组件接口
代码解读: 
    定义组件接口，用于与外部组件交互,如放大器和泵浦过程,
    主要包含两个函数：
    1. propagate_pulse: 流水线Pulse传播接口，将Pulse转换为光谱通量→放大→还原Pulse
    2. trigger_pump: 流水线统一泵浦接口，触发泵浦过程并更新粒子数

==============================================================================
"""
import numpy as np
from core.pulse import Pulse
from .frantz_nodvik import amplify_single_pass
from .pump_dynamics import pump_process



def propagate_pulse(
    pulse: Pulse,
    N_upper: float,
    cry_params,
    seed_params,
    consts,
    sigma_em: np.ndarray,
    sigma_abs: np.ndarray
) -> tuple[Pulse, float]:
    """
    参数:
        pulse: 输入Pulse对象
        N_upper: 初始上能级粒子数密度 [m^-3]
        cry_params: 晶体参数对象
        seed_params: 种子光参数对象
        consts: 物理常数对象
        sigma_em: 发射截面网格 (对应偏振)
        sigma_abs: 吸收截面网格 (对应偏振)
    
    返回:
        pulse: 放大后的Pulse对象
        N_upper_new: 放大后剩余上能级粒子数密度 [m^-3]
    """
    # 第一步：解包Pulse（时域→频域，提取能量通量）
    d_omega = pulse.grid.df * 2 * np.pi
    current_spectrum_J = (np.abs(pulse.A_f)**2) * d_omega 
    phase = np.angle(pulse.A_f) 
    
    # 第二步：执行单程放大
    spectrum_out_J, N_upper_new = amplify_single_pass(
        current_spectrum_J=current_spectrum_J,
        N_upper=N_upper,
        cry_params=cry_params,
        seed_params=seed_params,
        consts=consts,
        sigma_em=sigma_em,
        sigma_abs=sigma_abs
    )
    
    # 第三步：重新打包Pulse（还原频域振幅+相位→转回时域）
    pulse.A_f = np.sqrt(spectrum_out_J / d_omega) * np.exp(1j * phase)
    pulse.to_time_domain()
    
    return pulse, N_upper_new

def trigger_pump(
    dt: float,
    N_upper_start: float,
    cry_params,
    pump_params,
    consts,
    pump_polarization: str,
    pump_area: float,
    L_total: float
) -> float:
    """
    参数:
        dt: 泵浦持续时间 [s]
        N_upper_start: 初始上能级粒子数密度 [m^-3]
        cry_params: 晶体参数对象
        pump_params: 泵浦参数对象
        consts: 物理常数对象
        pump_polarization: 泵浦偏振方向
        pump_area: 泵浦光斑面积 [m²]
        L_total: 晶体总长度 [m]
    
    返回:
        N_upper_new: 泵浦后上能级粒子数密度 [m^-3]
    """
    return pump_process(
        time_duration=dt,
        N_upper_start=N_upper_start,
        cry_params=cry_params,
        pump_params=pump_params,
        consts=consts,
        pump_polarization=pump_polarization,
        pump_area=pump_area,
        L_total=L_total
    )