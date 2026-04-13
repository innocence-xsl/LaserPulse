"""
==============================================================================
文件名称: basics_5plus.py
所属部门: Applications (应用部)
主要功能: 光线经过增益介质传播的模拟（能量角度）
代码解读: 
    种子光多次经过增益介质（再生放大过程、多程往返），记录增益和能量的变化趋势。
    新增部分：
    1. 多程往返次数，
    2. 使用指数衰减计算 N2 消耗,防止出现负数，
    3. 增益饱和现象：增益到达峰值后开始下降，
    4. 脉冲时域畸变：深度增益饱和引起的前倾。
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
tau_seed = 100e-12 # 脉宽
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

# ==========================================
# 脉冲多程放大阶段
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

round_trips = 40 # 多程往返次数

# 准备一个列表，记录每一圈出来的能量
energy_history = []
Gain_history = []

# 4. 循环遍历每个空间步长
for round_trip in range(round_trips):
    for i in range(num_steps):
        # 获取当前这片 dz 的上能级粒子数
        current_N2 = N2_z[i]
        for j in range(N_time):
            # 当前时刻的光强
            I_current = I_t_out[j]
            # 光子通量 = 光强 / 单光子能量
            phi_seed = I_current / E_photon_seed 
            # 使用指数衰减计算 N2 消耗
            current_N2 = current_N2 * np.exp(- sigma_em_1030 * phi_seed * dt_pulse)
            # 计算净增益系数 (g_net)
            g_net = sigma_em_1030 * current_N2 - sigma_abs_1030 * (N_total - current_N2)
            # 使用局部指数放大光强
            I_t_out[j] = I_current * np.exp(g_net * dz)
        # 当整个脉冲彻底穿过这个 dz 后，把最终剩余的 N2 存回数组    
        N2_z[i] = current_N2
    # === 一圈结束，记录数据 ===
    E_out_current = np.sum(I_t_out) * dt_pulse * Area_seed
    energy_history.append(E_out_current)

# ==========================================
# 绘制输出结果 (能量提取与时域畸变)
# ==========================================
# 提取输入光强
I_t_in = I_peak_in * np.exp(-4 * np.log(2) * (t_array / tau_seed)**2)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- 图1：多程放大能量增长曲线 ---
passes = np.arange(1, round_trips + 1)
energy_mJ = np.array(energy_history) * 1e3

ax1.plot(passes, energy_mJ, 'bo-', linewidth=2, markersize=4) 
ax1.set_xlabel('往返圈数 (Round Trips)', fontsize=12)
ax1.set_ylabel('单脉冲能量 (mJ)', color='b', fontsize=12)
ax1.set_title(f'再生腔内脉冲能量演化 (共 {round_trips} 圈)', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.7)

# --- 图2：时域波形畸变对比 (归一化) ---
I_in_norm = I_t_in / np.max(I_t_in)
I_out_norm = I_t_out / np.max(I_t_out)

ax2.plot(t_array * 1e12, I_in_norm, 'k--', linewidth=2, label='输入种子脉冲 (10 nJ)')
ax2.plot(t_array * 1e12, I_out_norm, 'r-', linewidth=2, label='放大后输出脉冲 (mJ级)')
ax2.set_xlabel('时间 (ps)', fontsize=12)
ax2.set_ylabel('归一化光强 (a.u.)', fontsize=12)
ax2.set_title('脉冲时域波形畸变对比 (归一化)', fontsize=14)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

# 打印最终物理状态
print("-" * 50)
print(f"总往返圈数: {round_trips} 圈")
print(f"输入种子能量: {E_pulse * 1e9:.4f} nJ")
print(f"最终输出能量: {energy_history[-1] * 1e3:.4f} mJ")
print(f"总能量放大倍数: {energy_history[-1] / E_pulse:.2e} 倍")
print("-" * 50)

plt.tight_layout()
plt.savefig('utils/record/basics_5plus.png', dpi=300)
plt.show()