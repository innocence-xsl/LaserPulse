"""
==============================================================================
文件名称: test2.py
所属部门: applications（应用部）
主要功能: 测试工程化目标
代码解读: 
    测试工程化目标的实现，包括放大器和泵浦过程
==============================================================================
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.grid import Grid
from core.pulse import Pulse
from physics.amplification.component_interface import propagate_pulse_broadband

def load_and_interpolate(csv_path, y_multiplier=1e24):
    """
    读取CSV数据并返回插值函数 (从 basics_6 移植)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到文件: {csv_path}。请检查路径！")
        
    df = pd.read_csv(csv_path, header=None)
    wavelength_m = df.iloc[:, 0].to_numpy(dtype=float) * 1e-9
    values = df.iloc[:, 1].to_numpy(dtype=float) * y_multiplier
    interp_func = interp1d(wavelength_m, values, kind='linear', bounds_error=False, fill_value=0.0)
    return wavelength_m, values, interp_func

def main():
    # ==========================================
    # 1. 备料阶段：创建标准的流水线载体 (Grid 和 Pulse)
    # ==========================================
    # 实例化网格
    grid = Grid(points=1024, central_wl=1030e-9, max_wl=1080e-9)
    # 实例化脉冲
    pulse = Pulse(grid=grid)
    
    # 配置文件路径 (拼接绝对路径确保不会报错)
    path_abs = os.path.join(project_root, 'engineering', 'config', 'datafile', 'Yb_CALGO', 'sigma_abs.csv')
    path_emi = os.path.join(project_root, 'engineering', 'config', 'datafile', 'Yb_CALGO', 'sigma_emi.csv')
    path_seed = os.path.join(project_root, 'engineering', 'config', 'datafile', 'seed_spectrum.csv')

    _, _, func_sigma_abs = load_and_interpolate(path_abs, y_multiplier=1e-24)
    _, _, func_sigma_emi = load_and_interpolate(path_emi, y_multiplier=1e-24)
    _, _, func_seed = load_and_interpolate(path_seed, y_multiplier=1.0)

    # 截面直接映射到 grid 的波长系统上
    sigma_abs_grid = func_sigma_abs(grid.lambda_window)
    sigma_em_grid = func_sigma_emi(grid.lambda_window)

    # 光谱注入并归一化能量
    seed_energy = 5.2e-9
    spectrum_in = func_seed(grid.lambda_window)
    sum_S_current = np.sum(spectrum_in)
    if sum_S_current > 0:
        spectrum_in = spectrum_in * (seed_energy / sum_S_current)

    # 重点：把光谱初始化到 Pulse 对象的频域复振幅中！
    pulse.A_f = np.sqrt(spectrum_in) + 0j
    pulse.to_time_domain() # 同步一下时域

    # 晶体与泵浦状态
    N_total = 1.25e28 * 0.015       # 掺杂离子密度 [m^-3]
    current_N_upper = N_total * 0.5   # 假设泵浦已经将 50% 粒子抽运到上能级
    thickness = 6e-3                # 晶体厚度 [m]
    mode_area = np.pi * (100e-6)**2 # 模场面积 [m^2]
    T_rt = 0.95                     # 单圈透过率 (5% 损耗)

    # ==========================================
    # 2. 加工阶段 (调用流水线)
    # ==========================================
    max_round_trips = 50
    energy_history = []
    spectrum_history = []
    
    for trip in range(max_round_trips):
        pulse, current_N_upper, current_gain = propagate_pulse_broadband(
            pulse=pulse,
            N_upper_start=current_N_upper,
            N_total=N_total,
            sigma_em_grid=sigma_em_grid,
            sigma_abs_grid=sigma_abs_grid,
            thickness=thickness,
            mode_area=mode_area,
            num_slices=10
        )
        
        # 施加腔内反射镜损耗 (注意：损耗施加在电场上是开根号，施加在能量上是直接乘)
        pulse.A_f = pulse.A_f * np.sqrt(T_rt)
        
        # 从 pulse 身上提取能量记录下来
        current_spectrum = np.abs(pulse.A_f)**2
        current_energy = np.sum(current_spectrum)
        energy_history.append(current_energy)
        
        if trip % 10 == 0 or trip == max_round_trips - 1:
            spectrum_history.append(np.copy(current_spectrum))
            
        print(f"   -> 第 {trip+1:02d} 圈: 能量 = {current_energy*1e3:.4f} mJ | 单程增益 = {current_gain:.2f} | 剩余 N_2 比例 = {current_N_upper/N_total*100:.1f}%")
    # ==========================================
    # 3. 质检与可视化
    # ==========================================
    print("📊 生成对比图表...")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- 能量增长曲线 ---
    trips = np.arange(1, max_round_trips + 1)
    ax1.plot(trips, np.array(energy_history) * 1e3, 'r-o', linewidth=2)
    ax1.set_title('真实数据测试：能量演化 (Saturation)', fontsize=14)
    ax1.set_xlabel('Round Trips')
    ax1.set_ylabel('Energy (mJ)')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # --- 增益窄化光谱演化 ---
    # 从 grid 中获取波长画图
    lambda_nm = grid.lambda_window * 1e9
    S_init = spectrum_in / np.max(spectrum_in)
    S_mid = spectrum_history[len(spectrum_history)//2] / np.max(spectrum_history[len(spectrum_history)//2])
    S_final = np.abs(pulse.A_f)**2 / np.max(np.abs(pulse.A_f)**2)

    ax2.plot(lambda_nm, S_init, 'k-', linewidth=2, alpha=0.5, label='注入真实种子')
    ax2.plot(lambda_nm, S_mid, 'b-', linewidth=2, label='放大中期')
    ax2.plot(lambda_nm, S_final, 'r-', linewidth=2, label='最终饱和输出')
    ax2.set_title('真实数据测试：增益窄化 (Gain Narrowing)', fontsize=14)
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Normalized Intensity')
    ax2.set_xlim(1010, 1060)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()