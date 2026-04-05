"""
==============================================================================
文件名称: basics_3.py
所属部门: Applications (应用部)
主要功能: 探究不同参量（如受激辐射截面、腔内光子寿命等）对增益饱和的影响
代码解读: 
    老子开始手搓代码了！！！
    通过改变受激辐射截面（σ）和腔内光子寿命（τ_c）等关键参数，观察它们对增益饱和现象的影响。
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

# 定义待测试的自变量值列表（可根据需求增减）
sigma_list = [0.5e-24, 0.75e-24, 1.0e-24, 1.25e-24]  # 不同受激辐射截面
tau_c_list = [3e-9, 5e-9, 7e-9, 10e-9]  # 不同腔内光子寿命

# 为每个自变量分配颜色/线型（便于区分）
color_list = ['red', 'green', 'blue', 'purple']

def basic_3(y, t, sigma, tau_c): # （最重要的修改：新增sigma和tau_c参数，适配传参！！！）
    N2 = y[0]  # 上能级粒子数密度
    phi = y[1] # 光子数密度
    dN2_dt = - sigma * c * phi * N2     # 上能级粒子数的变化率公式
    dphi_dt = sigma * c * N2 * phi - phi / tau_c  # 光子数密度的变化率公式
    return [dN2_dt, dphi_dt]

# 时间数组，从0到100纳秒，分成1000个步长
t = np.linspace(0, 100e-9, 1000)

# 初始条件，N2和phi的初始值
y0 = [N_init, phi_init]  

# 初始化绘图画布
fig, (ax_phi, ax_n2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)  # 2行1列，共享X轴

# 遍历自变量_list，获取索引和当前自变量值
# eg：探究tau_c时，改成for idx, tau_c in enumerate(tau_c_list): 并 固定 sigma
#     探究sigma时，改成for idx, sigma in enumerate(sigma_list): 并 固定 tau_c

fixed_sigma = 0.75e-24  # 固定σ，探究τ_c的影响
# fixed_tau_c = 5e-9  # 固定τ_c，探究σ的影响

for idx, tau_c in enumerate(tau_c_list):  # 遍历tau_c_list，获取索引和当前tau_c值
    # 解微分方程，得到每个时间点的N2和phi（传入当前探究的自变量值）
    # args传参给basic_3, 一定要按照def basic_3(y, t, sigma, tau_c)的顺序传参，而且如果只有一个元素的元组，必须加逗号
    solution = odeint(basic_3, y0, t, args=(fixed_sigma, tau_c))  
    N2_result = solution[:, 0]
    phi_result = solution[:, 1]

    # 获取当前σ对应的颜色
    color = color_list[idx]
    # sigma_label = f'σ = {sigma*1e24:.2f}e-24 m²'  # 标签格式化
    tau_c_label = f'τ_c = {tau_c*1e9:.2f}e-9 s'  # 标签格式化
    
    # 绘制phi曲线（ax_phi）
    # ax_phi.plot(t*1e9, phi_result, color=color, linewidth=2, label=sigma_label)
    ax_phi.plot(t*1e9, phi_result, color=color, linewidth=2, label=tau_c_label)

    # 绘制N2曲线（ax_n2）
    # ax_n2.plot(t*1e9, N2_result, color=color, linewidth=2, label=sigma_label)
    ax_n2.plot(t*1e9, N2_result, color=color, linewidth=2, label=tau_c_label)

# 绘图格式设置（不用学，让AI写就行了）
ax_phi.set_ylabel('phi (m^-3)', fontsize=12)
ax_phi.set_title('Effect of τ_c (s) on Photon Density (phi)', fontsize=14)  # 只改标题中的变量名，其他格式保持不变
ax_phi.legend(loc='upper right', fontsize=10)
ax_phi.grid(alpha=0.3)

ax_n2.set_xlabel('Time (ns)', fontsize=12)
ax_n2.set_ylabel('N2 (m^-3)', fontsize=12)
ax_n2.set_title('Effect of τ_c (s) on Upper Level Population (N2)', fontsize=14)
ax_n2.legend(loc='upper right', fontsize=10)
ax_n2.grid(alpha=0.3)

fig.tight_layout()
plt.savefig('utils/record/basic_3.png', dpi=300, bbox_inches='tight') # 保存图形
plt.show()