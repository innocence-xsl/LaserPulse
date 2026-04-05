"""
==============================================================================
文件名称: basics_1.py
所属部门: Applications (应用部)
主要功能: 受激辐射光放大的基础物理过程
代码解读: 
    老子开始手搓代码了！！！
    其实就是两个变量——N2（上能级粒子数）、φ（光子数密度）用最基础的变化方程，建立非线性常微分方程组，耦合两个变量。
    1.忽略自发辐射效应，只考虑受激辐射光放大过程。
    2.对时间演化图的认识：Low-Intensity Regime (低光强区间)和High-Intensity Regime (高光强区间)
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

c = 3e8  # 光速 (m/s)
sigma = 0.75e-24    # 受激辐射截面 (m^2)
tau_c = 5e-9  # 腔内光子寿命 (s),由光学谐振腔的结构决定的（与腔长和单圈损耗相关）
N_init = 3e24  # 初始上能级粒子数密度 (m^-3)
phi_init = 1e18  # 初始光子数密度 (m^-3)

def basic_1(y, t):
    # 列表的第一个元素是N2，第二个元素是phi
    N2 = y[0]  # 上能级粒子数密度
    phi = y[1] # 光子数密度

    # 根据物理公式，计算变化率（导数）
    dN2_dt = - sigma * c * phi * N2     # 上能级粒子数的变化率公式
    dphi_dt = sigma * c * N2 * phi - phi / tau_c  # 光子数密度的变化率公式

    return [dN2_dt, dphi_dt]

# 时间数组，从0到100纳秒，分成1000个步长
t = np.linspace(0, 100e-9, 1000)

# 初始条件，N2和phi的初始值
y0 = [N_init, phi_init]  

# 解常微分方程组,solution是一个二维数组，第一列是N2随时间的变化，第二列是phi随时间的变化
solution = odeint(basic_1, y0, t)

N2_result = solution[:, 0]   # 提取第一列，即所有时间点的 N2
phi_result = solution[:, 1]  # 提取第二列，即所有时间点的 phi

# 绘制结果
fig, ax1 = plt.subplots(figsize=(8, 5)) # 创建一个图和一个坐标轴
color = 'tab:red' # 设置第一条曲线的颜色

# 绘制phi随时间的变化曲线
ax1.set_xlabel('Time (ns)') # 设置x轴标签
ax1.set_ylabel('phi (m^-3)', color=color) # 设置y轴标签和颜色
ax1.plot(t*1e9, phi_result, color=color, linewidth=2, label='phi: Photon Density') # 绘制phi随时间的变化曲线
ax1.tick_params(axis='y', labelcolor=color) # 设置y轴刻度颜色

# 绘制N2随时间的变化曲线
ax2 = ax1.twinx() # 创建第二个坐标轴，共享x轴
color = 'tab:blue' # 设置第二条曲线的颜色
ax2.set_ylabel('N2 (m^-3)', color=color) # 设置y轴标签和颜色
ax2.plot(t*1e9, N2_result, color=color, linestyle='--', linewidth=2, label='N2: Population') # 绘制N2随时间的变化曲线
ax2.tick_params(axis='y', labelcolor=color) # 设置y轴刻度颜色

plt.legend(handles=[*ax2.get_lines(), *ax1.get_lines()], loc='upper right') # 显示图例
plt.title('N2 & phi Saturation') # 设置标题
fig.tight_layout() # 调整布局以避免标签重叠
plt.savefig('utils/record/basic_1.png', dpi=300, bbox_inches='tight') # 保存图形
plt.show() # 显示图形