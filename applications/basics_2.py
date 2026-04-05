"""
==============================================================================
文件名称: basics_2.py
所属部门: Applications (应用部)
主要功能: 上能级粒子数、光子数密度与能量、增益的关系
代码解读: 
    老子开始手搓代码了！！！
    从微观到宏观，建立N2（上能级粒子数）、φ（光子数密度）和能量、增益的关系。
    1. 光子数密度 (φ) 到 宏观能量 (Energy)：桥梁是单个光子的能量
    单个光子的能量 E = h * ν（激光频率）
    光强 I = φ * c * h * ν（激光频率）
    2. 上能级粒子数密度 (N2) 到 增益 (Gain)：桥梁是受激辐射截面
    单位长度的增益系数 g(t) = σ * N2(t)
    宏观单程增益 G = exp(g(t) * L) = exp(σ * N2(t) * L)
    3. 通过数值积分，观察N2、φ、Energy和Gain随时间的变化。
    4. 基于基本物理关系，定义增益与能量的计算公式。
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 物理常数
c = 3e8  # 光速 (m/s)
h = 6.626e-34  # 普朗克常数 (J·s)
# 晶体参数
sigma = 0.75e-24    # 受激辐射截面 (m²)
L = 0.01  # 晶体长度 (m)
A = np.pi * (0.5e-3)**2  # 光束横截面积 (m²), 直径 0.5 mm 的光斑对应的面积
n_refr = 1.93  # 晶体的折射率, eg:Yb:CALGO
tau_f = 420e-6 # 上能级寿命 (s)
tau_c = 5e-9  # 腔内光子寿命 (s),由光学谐振腔的结构决定的（与腔长和单圈损耗相关）

# 激光脉冲参数
lambda_L = 1030e-9  # 激光波长 (m)
E_seed = 1e-6  # 种子光脉冲能量 (J)
pulse_seed = 200e-15 # 种子脉冲宽度 (s)
E_stored = 1e-3  # 初始储能 (J), 泵浦光吸收后转化来的能量

def basic_2(y, t):
    N2 = y[0]  # 当前时刻的 上能级粒子数密度
    phi = y[1] # 当前时刻的 光子数密度

    v = c / n_refr # 计算光在晶体介质中的真实传播速度

    # 速率方程
    dN2_dt = - sigma * v * phi * N2

    # dphi_dt = sigma * v * N2 * phi - phi / tau_c  # 包含增益和损耗项
    dphi_dt = sigma * v * N2 * phi  # 单程放大，不考虑腔内损耗的影响

    return [dN2_dt, dphi_dt]

E_photon = h * c / lambda_L  # 单个光子的能量 (J)
N2_init = E_stored / (E_photon * A * L)  # 初始上能级粒子数密度 (m^-3)
V_pulse = A * (c / n_refr) * pulse_seed # 脉冲占据的体积 (m³)
phi_init = E_seed / (E_photon * V_pulse) # 初始光子数密度 (m^-3)
y0 = [N2_init, phi_init]

t = np.linspace(0, 80e-9, 1000) # 模拟时间范围 (s)
# 解常微分方程组,solution是一个二维数组，第一列是N2随时间的变化，第二列是phi随时间的变化
solution = odeint(basic_2, y0, t)

N2_result = solution[:, 0]   # 提取第一列，即所有时间点的 N2
phi_result = solution[:, 1]  # 提取第二列，即所有时间点的 phi

# 把光子密度转回宏观能量
Energy_result = phi_result * E_photon * V_pulse

# 把上能级粒子密度转为单程增益
Gain_result = np.exp(sigma * N2_result * L)

# 绘制结果
fig, ax1 = plt.subplots(figsize=(8, 5)) # 创建一个图和一个坐标轴
color = 'tab:red' # 设置第一条曲线的颜色

# 绘制phi随时间的变化曲线
ax1.set_xlabel('Time (ns)') # 设置x轴标签
ax1.set_ylabel('Energy (J)', color=color) # 设置y轴标签和颜色
ax1.plot(t*1e9, Energy_result, color=color, linewidth=2, label='Energy') # 绘制Energy随时间的变化曲线
ax1.tick_params(axis='y', labelcolor=color) # 设置y轴刻度颜色

# 绘制N2随时间的变化曲线
ax2 = ax1.twinx() # 创建第二个坐标轴，共享x轴
color = 'tab:blue' # 设置第二条曲线的颜色
ax2.set_ylabel('Gain', color=color) # 设置y轴标签和颜色
ax2.plot(t*1e9, Gain_result, color=color, linestyle='--', linewidth=2, label='Gain') # 绘制Gain随时间的变化曲线
ax2.tick_params(axis='y', labelcolor=color) # 设置y轴刻度颜色

plt.legend(handles=[*ax1.get_lines(), *ax2.get_lines()], loc='upper right') # 显示图例
plt.title('Energy & Gain Saturation') # 设置标题
fig.tight_layout() # 调整布局以避免标签重叠
plt.savefig('utils/record/basic_2.png', dpi=300, bbox_inches='tight') # 保存图形
plt.show() # 显示图形
