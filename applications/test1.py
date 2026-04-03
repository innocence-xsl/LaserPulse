"""
==============================================================================
文件名称: test1.py
所属部门: Test (测试部)
主要功能: 自己手搓的代码
代码解读: 
    用来测试整个再生放大器的流程，从参数读取、腔外预整形、晶体放大，到结果可视化。
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
    # ==========================================
    # 1. 参数加载
    # ==========================================

    loader = ParameterLoader()
    # 获取所有的参数对象
    cry_params = loader.get_crystal_params()
    pump_params = loader.get_pump_params()
    cav_params = loader.get_cavity_params()
    seed_pulse, seed_params = loader.get_seed_params()  # 注意这里会同时返回 Pulse 和 SeedParams
    consts = PhysicalConstants() # 物理常数直接实例化

    # ==========================================
    # 2. 从对象中提取所需变量
    # ==========================================
    h = consts.h         # 普朗克常数 (J·s)
    c = consts.c         # 光速 (m/s)

    Lambda = seed_params.lambda_s
    Eseed = seed_params.E_seed
    w0 = seed_params.w_s  
    iterNum = int(seed_params.round_trips) # 从配置文件读取要跑的圈数

    L = cav_params.length
    Tr = cav_params.round_trip_time  # dataclass 里已经自动算好了 2L/c！直接用！
    loss = cav_params.round_trip_loss # dataclass 里的 property，自动求和损耗

    tL = cry_params.tau_f # 晶体的光子寿命 (s)
    sigma = cry_params.get_sigma_at(wavelength_m=Lambda, type='em', polarization='sigma') # 受激发射截面 (cm^2)，通过方法自动计算

    g0 = 0.5   # 初始增益，暂时写死，后续可以改成从配置文件读取
    Esat = (h * c / Lambda) * (np.pi * w0**2) / sigma  # 饱和能量 (J)

    # ==========================================
    # 3. 解析解模型
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
    # 4. ODE 常微分方程模型
    # ==========================================
    t0 = 0
    tEnd = Tr * iterNum
    y0 = [g0, Eseed]  
    
    T_ode = []
    Y_ode = []

    r = ode(rate_ode_system).set_integrator('vode', method='bdf')
    r.set_f_params(tL, Esat, Tr, loss, g0)  
    r.set_initial_value(y0, t0)

    while r.successful() and r.t + Tr <= tEnd + 1e-15:
        r.integrate(r.t + Tr)
        Y_ode.append(r.y)
        T_ode.append(r.t / Tr) 

    Y_ode = np.array(Y_ode)
    ga_ode = Y_ode[:, 0]
    energy_ode = Y_ode[:, 1]

    # ==========================================
    # 5. 数据对比展示
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
    axarr[1].set_xlabel("Round Trip", fontsize=12, fontweight='bold')
    axarr[1].grid(True, linestyle='--', alpha=0.7)
    axarr[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()