"""
==============================================================================
文件名称: component_interface.py
所属部门: physics/amplification (物理部/放大模型)
主要功能: 组件接口
代码解读:
    定义 physics 层与 engineering 组件层之间的接口。
    这里负责把 Pulse / 参数对象拆包，送入底层物理算子，
    再把计算结果重新打包回 Pulse 或晶体状态。
==============================================================================
"""

import numpy as np

from core.pulse import Pulse
from physics.amplification.gain import amplify_spectral_broadband
from physics.amplification.pump_dynamics import pump_process


def propagate_pulse_broadband(
    pulse: Pulse,
    N_upper_start: float,
    N_total: float,
    sigma_em_grid: np.ndarray,
    sigma_abs_grid: np.ndarray,
    thickness: float,
    mode_area: float,
    num_slices: float = 1.0,
) -> tuple[Pulse, float, float]:
    """
    流水线 Pulse 传播接口（宽带适配版）。

    输入 Pulse 对象和晶体状态，调用底层宽带增益算子，
    然后更新 Pulse 的频域/时域状态，并返回新的 N_upper。
    """
    spectrum_in = np.abs(pulse.A_f) ** 2
    phase_in = np.angle(pulse.A_f)
    lambda_grid = pulse.grid.lambda_window

    spectrum_out, N_upper_end, current_gain = amplify_spectral_broadband(
        spectrum_in_intensity=spectrum_in,
        lambda_grid=lambda_grid,
        N_upper_start=N_upper_start,
        N_total=N_total,
        sigma_em_grid=sigma_em_grid,
        sigma_abs_grid=sigma_abs_grid,
        thickness=thickness,
        mode_area=mode_area,
        num_slices=num_slices,
    )

    pulse.A_f = np.sqrt(np.clip(spectrum_out, 0.0, None)) * np.exp(1j * phase_in)
    pulse.to_time_domain()

    return pulse, N_upper_end, current_gain


def propagate_pulse(
    pulse: Pulse,
    N_upper: float,
    cry_params,
    seed_params,
    consts=None,
    sigma_em=None,
    sigma_abs=None,
) -> tuple[Pulse, float]:
    """
    Engineering 组件使用的简洁传播接口。

    BulkCrystal 不需要知道底层采用的是 broadband 模型、F-N 模型还是其他模型。
    现阶段默认调用宽带频域增益模型。
    """
    if sigma_em is None or sigma_abs is None:
        raise ValueError("propagate_pulse 需要传入 sigma_em 和 sigma_abs 截面数组")

    thickness = cry_params.thickness * getattr(cry_params, "num_disks", 1)
    mode_area = getattr(seed_params, "seed_area", None)
    if mode_area is None or mode_area <= 0:
        mode_area = np.pi * getattr(seed_params, "w_s", 1.0) ** 2

    num_slices = int(getattr(cry_params, "num_slices", 1))

    pulse, N_upper_end, _gain = propagate_pulse_broadband(
        pulse=pulse,
        N_upper_start=N_upper,
        N_total=cry_params.N_doping,
        sigma_em_grid=sigma_em,
        sigma_abs_grid=sigma_abs,
        thickness=thickness,
        mode_area=mode_area,
        num_slices=max(num_slices, 1),
    )

    return pulse, N_upper_end


def trigger_pump(
    dt: float,
    N_upper_start: float,
    cry_params,
    pump_params,
    consts,
    pump_polarization: str,
    pump_area: float,
    L_total: float,
) -> float:
    """
    Engineering 组件使用的泵浦接口。

    输入当前 N_upper 和泵浦参数，返回泵浦 dt 后的新 N_upper。
    """
    return pump_process(
        time_duration=dt,
        N_upper_start=N_upper_start,
        cry_params=cry_params,
        pump_params=pump_params,
        consts=consts,
        pump_polarization=pump_polarization,
        pump_area=pump_area,
        L_total=L_total,
    )
