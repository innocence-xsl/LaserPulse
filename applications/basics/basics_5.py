"""
==============================================================================
文件名称: basics_5.py
所属部门: Applications (应用部)
主要功能: 光线经过增益介质传播的模拟（能量角度）
代码解读: 
    种子光单次经过增益介质（类似行波放大过程）
    分为两个阶段：
    1. 泵浦充能阶段：建立反转粒子数
    2. 能量提取阶段：光子与增益介质作用，释放能量
    记录增益和能量的变化趋势。
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

## ==========================================
## 参数加载
## ==========================================

c = 3e8 # 光速
h = 6.62607015e-34 # 普朗克常数

#  种子光参数
E_pulse = 10e-9 # 单脉冲能量
tau_seed = 500e-12 # 脉宽
w_seed = 100e-6 # 光斑束腰半径
lambda_seed = 1030e-9 # 中心波长

# 泵浦光参数
P_pump = 100 # 泵浦功率（W）
w_pump = 110e-6 # 泵浦光斑束腰半径（m）
lambda_pump = 980e-9 # 泵浦波长（m）

# 增益介质参数
thickness = 6e-3 # 增益介质长度  [m]
N_total_base = 1.25e28 # 基质离子密度基数  [ions/m^3]
doping_at_percent = 1.5 # 掺杂浓度  [a.t.%]
sigma_em_1030 = 0.72e-24 # 1030nm处的受激发射截面 (m^2)
sigma_abs_1030 = 0.22e-24 # 1030nm处的吸射截面 (m^2)
sigma_em_980 = 0.01e-24 # 980nm处的发射截面 (m^2)
sigma_abs_980 = 2.7e-24 # 980nm处的吸射截面 (m^2)
tau_f = 420e-6 # 上能级寿命 (s)
n_CGA = 1.93   # calgo的折射率 (N0 = 1.9331、Ne = 1.9564)
N_total = N_total_base * (doping_at_percent / 100)  # 总掺杂离子密度

## ==========================================
## 泵浦充能 (建立反转粒子数)
## ==========================================
E_photon_pump = h * c / lambda_pump # 泵浦光子能量 (J)
Area_pump = np.pi * w_pump**2 # 泵浦光斑束面积 (m^2)
phi_pump = P_pump / (Area_pump * E_photon_pump) # 泵浦光子通量

def pump_ode(N2, t):
    N2_abs = phi_pump * sigma_abs_980 * (N_total - N2) # 吸收项
    N2_stimulated = phi_pump * sigma_em_980 * N2 # 受激发射项
    N2_spontaneous = N2 / tau_f # 自发辐射项

    dN2_dt = N2_abs- N2_stimulated - N2_spontaneous # 速率方程

    return dN2_dt

dt = 1e-3 # 泵浦时间
t_pump = np.linspace(0, dt, 1000)
N2_init = 0.0  # 上能级一开始是空的

# 解常微分方程组
solution = integrate.odeint(pump_ode, N2_init, t_pump)

N2_final = solution[-1, 0]   # 提取最后一行，即 N2 最终值

# 打印最终的物理状态
inversion_ratio = (N2_final / N_total) * 100
print("-" * 50)
print(f"泵浦时长: {dt*1e3} ms")
print(f"最终上能级粒子密度 (N2): {N2_final:.4e} ions/m^3")
print(f"达到的粒子数反转比例: {inversion_ratio:.2f} %")
print("-" * 50)

# ==========================================
# 脉冲单次放大阶段
# ==========================================
num_steps = 100 # 切片数量
dz = thickness / num_steps  # 空间步长
N2_z = np.full(num_steps, N2_final)  # N2_z 数组记录晶体不同深度的上能级粒子数密度

# 1. 构建时间轴 
N_time = 2000 # 时间步数量
t_array = np.linspace(-1000e-12, 1000e-12, N_time)
dt_pulse = t_array[1] - t_array[0] # 时间步长

# 2. 计算初始种子光的峰值光强 (I_peak)
Area_seed = np.pi * w_seed**2
P_peak_in = E_pulse / tau_seed 
I_peak_in = P_peak_in / Area_seed

# 3. 生成高斯光强时域包络 I_t (W/m^2)
I_t_in = I_peak_in * np.exp(-4 * np.log(2) * (t_array / tau_seed)**2)

# 准备一个数组用于在循环中不断更新光强
I_t_out = np.copy(I_t_in)

# 种子光单光子能量
E_photon_seed = h * c / lambda_seed

# 4. 循环遍历每个空间步长
for i in range(num_steps):
    # 获取当前这片 dz 的上能级粒子数
    current_N2 = N2_z[i]
    for j in range(N_time):
        # 当前时刻的光强
        I_current = I_t_out[j]
        # 光子通量 = 光强 / 单光子能量
        phi_seed = I_current / E_photon_seed 
        # dN2 = - (受激发射截面 * 当前N2 * 光子通量) * dt
        dN2 = - (sigma_em_1030 * current_N2 * phi_seed) * dt_pulse
        # 更新这片 dz 内的 N2
        current_N2 = current_N2 + dN2 
        # dI = 增益系数 * 当前光强 * dz = (受激发射截面 * 当前N2 - 吸射截面 * 总N2 - 当前N2) * 当前光强 * dz
        dI = (sigma_em_1030 * current_N2 - sigma_abs_1030 * (N_total - current_N2)) * I_current * dz
        # 更新出射光强
        I_t_out[j] = I_current + dI
    # 当整个脉冲彻底穿过这个 dz 后，把最终剩余的 N2 存回数组
    N2_z[i] = current_N2
# 循环结束，此时的 I_t_out 就是穿出 6mm 晶体后的最终光强包络！

# 积分求出放大后的总能量 ( E = ∫ I(t) dt * Area )
E_out = np.sum(I_t_out) * dt_pulse * Area_seed

# 计算单次增益 (Gain)
Gain = E_out / E_pulse

print(f"输入能量: {E_pulse * 1e9:.4f} nJ")
print(f"输出能量: {E_out * 1e9:.4f} nJ")
print(f"单次通过增益 (Gain): {Gain:.2f} 倍")

# # ==========================================
# # 绘制曲线
# # ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

# --------------------- 第一个子图：泵浦充能曲线 ---------------------
# 从 odeint 的二维返回结果中提取一维的 N2 数组
N2_array = solution[:, 0]

# 画充能曲线 (X轴转为毫秒 ms 方便观察)
ax1.plot(t_pump * 1e3, N2_array, 'b-', linewidth=2.5, label='上能级粒子数密度 $N_2$')
ax1.set_xlabel('泵浦时间 (ms)', fontsize=12)
ax1.set_ylabel('上能级粒子数密度 (ions/m$^3$)', color='b', fontsize=12)
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True, linestyle='--', alpha=0.7)

# 画一条红色的虚线作为总离子密度的参考线
ax1.axhline(y=N_total, color='r', linestyle='--', alpha=0.5, label=f'总掺杂密度: {N_total:.2e}')
ax1.set_title('Yb:CALGO 晶体泵浦充能动态过程', fontsize=14)
ax1.legend(loc='lower right')

# --------------------- 第二个子图：脉冲放大时域波形 ---------------------
# 画图对比输入输出时域波形
ax2.plot(t_array * 1e12, I_t_in, 'k--', label='输入脉冲 (Seed)')
ax2.plot(t_array * 1e12, I_t_out, 'r-', linewidth=2, label='输出脉冲 (Amplified)')
ax2.set_xlabel('时间 (ps)', fontsize=12)
ax2.set_ylabel('光强 (W/m$^2$)', fontsize=12)
ax2.set_title(f'脉冲时域放大对比 (单次经过 {thickness*1e3:.2f} mm 晶体)', fontsize=14)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

fig.suptitle('单次增益介质泵浦-放大过程(能量变化)', fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig('utils/record/basics_5.png', dpi=300)
plt.show()