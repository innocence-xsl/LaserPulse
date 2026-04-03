"""
==============================================================================
文件名称: energy_dynamics.py
所属部门: Applications (应用部)
主要功能: 速率方程能量与增益动态演化模拟
代码解读: 
    整合解析解模型与ODE数值模型，模拟再生放大器中，
    随着往返次数(Round Trips)的增加，脉冲能量和系统增益的演化趋势。
==============================================================================
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode

# 自动获取项目根目录并加入搜索路径
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(project_root)

# 导入底层物理算子
from physics.amplification.rate_equations import analytical_rate_step, rate_ode_system
from engineering.config.parameter_loader import ParameterLoader
from core.dataclasses_def import PhysicalConstants

def main():
    print("🚀 启动再生放大器能量与增益动态演化模拟...")

    # ==========================================
    # 1. 物理参数初始化 (整合自 test1, test2, test3)
    # ==========================================
    h = 6.6e-34         # 普朗克常数 (J·s)
    c = 3e8             # 光速 (m/s)
    Lambda = 1030e-9    # 信号光波长 (m)
    g0 = 0.5            # 初始增益
    L = 1.7             # 腔长 (m)
    Tr = 2 * L / c      # 腔内单次往返时间 (s)
    tL = 0.3e-3         # 增益介质荧光寿命 (s)
    loss = 0.04         # 腔内往返总损耗
    sigma = 3e-20       # 受激发射截面 (cm^2)
    w0 = 0.04           # 介质中束腰半径 (cm)
    
    # 物理量推导
    Esat = (h * c / Lambda) * (np.pi * w0**2) / sigma  # 饱和能量 (J)
    Eseed = 1e-9        # 种子光脉冲能量 (J)
    iterNum = 120       # 模拟往返圈数 (Round Trips)

    print(f"   -> 腔内往返时间 (Tr): {Tr:.2e} s")
    print(f"   -> 饱和能量 (Esat): {Esat:.2e} J")

    # ==========================================
    # 2. 方法一：解析解模型 (源自 test1, test3)
    # ==========================================
    ga_analytical = []
    energy_analytical = []
    
    for index in range(iterNum):
        tao = index * Tr
        g_val, e_val = analytical_rate_step(g0, Esat, Eseed, tao, Tr)
        ga_analytical.append(g_val)
        energy_analytical.append(e_val)

    energy_analytical = np.array(energy_analytical)

    # ==========================================
    # 3. 方法二：ODE 常微分方程模型 (源自 test2)
    # ==========================================
    t0 = 0
    tEnd = Tr * iterNum
    y0 = [g0, Eseed]  # 初值：[初始增益, 初始能量]
    
    T_ode = []
    Y_ode = []

    # 配置 vode 求解器
    r = ode(rate_ode_system).set_integrator('vode', method='bdf')
    r.set_f_params(tL, Esat, Tr, loss, g0)  # 传入附加参数
    r.set_initial_value(y0, t0)

    # 循环积分求解
    while r.successful() and r.t + Tr <= tEnd + 1e-15:
        r.integrate(r.t + Tr)
        Y_ode.append(r.y)
        T_ode.append(r.t / Tr) # 记录圈数

    Y_ode = np.array(Y_ode)
    ga_ode = Y_ode[:, 0]
    energy_ode = Y_ode[:, 1]

    # ==========================================
    # 4. 可视化大屏 (数据对比展示)
    # ==========================================
    print("📊 正在生成 Round Trip 演化趋势图...")
    nums = np.arange(iterNum)
    
    f, axarr = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # --- 屏幕 1: 增益演化 (Gain Dynamics) ---
    axarr[0].plot(nums, ga_analytical, color='green', linestyle='--', label='Analytical Model (No Loss)', linewidth=2)
    if len(T_ode) > 0:
        axarr[0].plot(T_ode, ga_ode, color='darkgreen', linestyle='-', label=f'ODE Numerical Model (Loss={loss*100}%)', linewidth=2)
    axarr[0].set_ylabel("Gain (a.u.)", fontsize=12, fontweight='bold')
    axarr[0].set_title("Regenerative Amplifier Dynamics (Gain & Pulse Energy)", fontsize=14, fontweight='bold')
    axarr[0].grid(True, linestyle='--', alpha=0.7)
    axarr[0].legend()

    # --- 屏幕 2: 能量演化 (Energy Dynamics) ---
    axarr[1].plot(nums, energy_analytical * 1e3, color='blue', linestyle='--', label='Analytical Model', linewidth=2)
    if len(T_ode) > 0:
        axarr[1].plot(T_ode, energy_ode * 1e3, color='darkblue', linestyle='-', label='ODE Numerical Model', linewidth=2)
    axarr[1].set_ylabel("Pulse Energy (mJ)", fontsize=12, fontweight='bold')
    axarr[1].set_xlabel("Round Trip Number", fontsize=12, fontweight='bold')
    axarr[1].grid(True, linestyle='--', alpha=0.7)
    axarr[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()