"""
==============================================================================
文件名称: basics_6.py
所属部门: Applications (应用部)
主要功能: 光线经过增益介质传播的模拟（频域角度）
代码解读: 
    种子光多次经过增益介质（再生放大过程、多程往返），记录频域、光谱的变化趋势。
    根据饱和点的脉冲输出，观察放大前后的光谱信息和增益窄化现象。
==============================================================================
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.fft import fft, ifft, fftfreq, fftshift


c = 3e8 # 光速
h = 6.62607015e-34 # 普朗克常数

#  种子光参数
E_pulse = 5.2e-9 # 单脉冲能量
tau_seed = 5e-12 # 脉宽
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
tau_f = 420e-6 # 上能级寿命 (s)
n_CGA = 1.93   # calgo的折射率 (N0 = 1.9331、Ne = 1.9564)
N_total = N_total_base * (doping_at_percent / 100)  # 总掺杂离子密度

def load_and_interpolate(csv_path,y_multiplier=1e24):
    """
    读取CSV数据并返回插值函数
    :param csv_path: 文件路径
    :param y_multiplier: Y轴数据的单位缩放系数,数据文件中的截面单位是 1e-20 cm^2
    :return: 原始波长(m), 原始数值, 连续插值函数
    """
    # CSV没有表头：第一列是波长(nm)，第二列是数值
    df = pd.read_csv(csv_path, header=None)

    # 1. 提取并转换单位 (X轴 nm -> m)
    wavelength_m = df.iloc[:, 0].to_numpy(dtype=float) * 1e-9

    # 2. 提取并转换单位 (Y轴 数值乘以缩放系数)
    values = df.iloc[:, 1].to_numpy(dtype=float) * y_multiplier

    # 3. 创建插值函数
    interp_func = interp1d(wavelength_m, values, kind='linear', bounds_error=False, fill_value=0.0)
    return wavelength_m, values, interp_func

# ==========================================
# 时域网格与脉冲构建
# ==========================================
# 1. 物理常数与计算
omega_0 = 2 * np.pi * c / lambda_seed # 中心角频率
Area_seed = np.pi * w_seed**2          # 光斑面积
P_peak_in = E_pulse / tau_seed         # 峰值功率
I_peak_in = P_peak_in / Area_seed      # 峰值光强 (W/m^2)
E_amp = np.sqrt(I_peak_in)             # 峰值电场幅值 (V/m)

# 2. 时间网格设置
T_window = 10 * tau_seed # 时间窗口，需覆盖脉冲
dt = 0.5e-15 # 时间步长 0.5 fs，保证能分辨 1030nm 的载波
N_time = int(T_window / dt)
N_time = 2**(N_time - 1).bit_length()  # 强制 N_time 为 2 的幂次方，加速 FFT 运算效率！
t_array = np.arange(-N_time/2, N_time/2) * dt # 时间轴数组

# 3. 构建时域复电场 E(t)
CEP = 0 # 载波包络相位(弧度)
chirp_a = 0 # 线性啾啾系数，先设为0（无啾啾变换极限脉冲）
A_t = E_amp * np.exp(-2 * np.log(2) * (t_array / tau_seed)**2) # 高斯振幅包络
Phi_t = omega_0 * t_array + CEP + chirp_a * (t_array**2) # 总相位
E_t = A_t * np.exp(1j * Phi_t)  # 带有载波相位的复电场

# ==========================================
# 执行 FFT 与频域物理坐标轴映射
# ==========================================
# 1. 执行傅里叶变换
E_f_raw = fft(E_t)

# 2. 生成频率坐标轴并对齐中心（把零频移到中心）
f_array_raw = fftfreq(N_time, dt)
E_f = fftshift(E_f_raw)
f_array = fftshift(f_array_raw)

# 3. 提取物理上的正频率部分，并映射为波长
pos_mask = f_array > 0       # 掩码：剔除没有物理意义的负频率
f_pos = f_array[pos_mask]    # 正频率数组 (Hz)
E_f_pos = E_f[pos_mask]      # 对应的频域复电场

# 4. 波长坐标轴
lambda_array = c / f_pos  # 物理波长数组、正频率部分的波长
# print(f"-> 频域网格构建完成。波长范围: {lambda_array.min()*1e9:.1f} nm 到 {lambda_array.max()*1e9:.1f} nm")

# ==========================================
# 再生放大循环
# ==========================================
max_round_trips = 50 # 最大循环圈数
T_rt = 0.95 # 腔内单圈透过率
V_mode = Area_seed * thickness # 有效作用体积
"""
在种子光注入再生腔之前，泵浦二极管会先对晶体照射一段特定的时间（通常为几百微秒，接近上能级寿命 tau_f）,
在这个过程中，基质离子吸收泵浦光子跃迁至上能级，同时伴随着自发辐射的损耗,
真正的 N2_current 是泵浦速率与自发辐射衰减速率达到动态平衡（或在泵浦脉冲结束瞬间）时的物理极值,
需要先跑一遍只包含泵浦项和自发辐射项的微分方程,将其最终的稳态解作为这段再生放大循环的初始值。
"""
N2_current = N_total * 0.5 # 初始上能级粒子数 (假设泵浦达到了50%反转)
N1_current = N_total - N2_current
"""
真实的机制是：光谱上的每一个波长切片dλ，都携带着具有该波长特定能量E(λ) = h * c / λ 的光子。
当电场在频域被放大后，整个光谱强度 S(λ发生了形变。
通过计算放大前后光谱强度的差值△S(λ) = S(λ_out) - S(λ_in)，得到真实提取出来的“增量光谱”。
真正的光子消耗量，需要将这个增量光谱在频域上按对应的单光子能量进行积分（或离散求和）：
"""
# E_photon_1030 = h * c / 1030e-9 # 近似用中心波长的光子能量计算消耗
energy_history = []  # 记录每一圈出来的能量
spectrum_history = [] # 记录每一圈出来的光谱

wl_sigma_abs, val_sigma_abs, func_sigma_abs = load_and_interpolate('engineering/config/datafile/Yb_CALGO/sigma_abs.csv', y_multiplier=1e-24)
wl_sigma_emi, val_sigma_emi, func_sigma_emi = load_and_interpolate('engineering/config/datafile/Yb_CALGO/sigma_emi.csv', y_multiplier=1e-24)
wl_seed, val_seed, func_seed = load_and_interpolate('engineering/config/datafile/seed_spectrum.csv', y_multiplier=1.0)

# 1. 真实种子光谱注入
S_lambda_real = func_seed(lambda_array)
# 2. 替换频域电场的振幅
phase_f = np.angle(E_f_pos)
E_f_pos = np.sqrt(S_lambda_real) * np.exp(1j * phase_f)
# 3. 能量归一化：缩放电场，让真实光谱的总能量等于设定的 E_pulse (5.2 nJ)
sum_S_current = np.sum(np.abs(E_f_pos)**2)
scaling_factor = np.sqrt(E_pulse / sum_S_current)
E_f_pos = E_f_pos * scaling_factor

# 设定循环前的初始能量
E_current = E_pulse

for trip in range(max_round_trips):
    # 1. 记录放大前的频谱总强度
    S_lambda_in = np.abs(E_f_pos)**2
    sum_S_in = np.sum(S_lambda_in)

    # 2. 获取当前波长数组对应的截面 (调用插值函数)
    sigma_e = func_sigma_emi(lambda_array)
    sigma_a = func_sigma_abs(lambda_array)

    # 3. 计算当前 N2 状态下的宽带增益系数 G(lambda)
    G_lambda = (sigma_e * N2_current - sigma_a * (N_total - N2_current)) * thickness

    # 4. 频域电场放大：核心！(增益大的波长指数放大快，产生增益窄化)
    E_f_pos = E_f_pos * np.exp(G_lambda / 2)

    # 5. 计算真实的物理能量增加与 N2 消耗 (宏观守恒)
    S_lambda_out = np.abs(E_f_pos)**2
    sum_S_out = np.sum(S_lambda_out)

    # 算出当前状态下：[实际物理能量 (焦耳)] 与 [数学数组面积 (a.u.)] 的转换系数
    k_energy = E_current / sum_S_in

    # a. 得到一个数组，代表每一个极小的波长切片上，增加了多少物理能量 (J)
    delta_E_array = k_energy * (S_lambda_out - S_lambda_in)
    # b. 计算所有波长切片上的光子数 (个)
    delta_photons = np.sum(delta_E_array * lambda_array / (h * c))
    # c. 从晶体中扣除这些消耗掉的反转粒子数 (N2 最少不能低于0)
    N2_current = max(0, N2_current - delta_photons / V_mode)
    N1_current = N_total - N2_current

    # 利用频谱面积的等比例放大，算出晶体给出的放大后能量
    E_amplified = E_current * (sum_S_out / sum_S_in)

    # 6. 施加腔内单圈反射损耗
    E_f_pos = E_f_pos * np.sqrt(T_rt) # 每次撞击反射镜，电场衰减
    E_current = E_amplified * T_rt    # 能量衰减，作为下一圈的起点

    # 7. 记录数据
    energy_history.append(E_current)
    # 每 10 圈记录一次光谱，防止内存占用过大
    if trip % 10 == 0 or trip == max_round_trips - 1:
        spectrum_history.append(np.abs(E_f_pos)**2)

    print(f"第 {trip+1:02d} 圈: 能量 = {E_current*1e3:.4f} mJ, N2剩余占比 = {N2_current/N_total*100:.1f}%")

print("循环结束！")

# ==========================================
# 结果可视化 (能量饱和与增益窄化)
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- 图1：能量演化 S 型曲线 ---
trips = np.arange(1, max_round_trips + 1)
energy_mJ = np.array(energy_history) * 1e3

ax1.plot(trips, energy_mJ, 'r-o', linewidth=2, markersize=4, label='腔内脉冲能量')
ax1.axvline(x=35, color='k', linestyle='--', alpha=0.5, label=f'最佳倒出点 (Trip {35})')
ax1.set_xlabel('循环圈数 (Round Trips)', fontsize=12)
ax1.set_ylabel('能量 (mJ)', color='r', fontsize=12)
ax1.tick_params(axis='y', labelcolor='r')
ax1.set_title('再生腔内脉冲能量演化', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='upper left')

# --- 图2：频域光谱与增益窄化 ---
# 提取第一圈、峰值圈(近似)和最后一圈的光谱数据进行对比
# spectrum_history 记录了 trip 0, 10, 20, 30, 40, 49 的数据
idx_init = 0
idx_mid = len(spectrum_history) // 2
idx_final = -1

S_init = spectrum_history[idx_init]
S_mid = spectrum_history[idx_mid]
S_final = spectrum_history[idx_final]

# 归一化处理，方便对比谱宽
S_init_norm = S_init / np.max(S_init)
S_mid_norm = S_mid / np.max(S_mid)
S_final_norm = S_final / np.max(S_final)

# 转换波长单位为 nm
lambda_nm = lambda_array * 1e9

ax2.plot(lambda_nm, S_init_norm, 'k-', linewidth=2, alpha=0.5, label='注入种子光谱')
ax2.plot(lambda_nm, S_mid_norm, 'b-', linewidth=2, label='放大中期光谱')
ax2.plot(lambda_nm, S_final_norm, 'r-', linewidth=2, label='最终饱和光谱')

ax2.set_xlim(990, 1080) 
ax2.set_xlabel('波长 (nm)', fontsize=12)
ax2.set_ylabel('归一化光谱强度', fontsize=12)
ax2.set_title('频域增益窄化演化 (Gain Narrowing)', fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

plt.tight_layout()
# plt.savefig('utils/record/basics_6.png', dpi=300)
plt.show()


# # ==========================================
# # 实际调用与参数定义
# # ==========================================
# # 1. 读取 π 偏振下的吸收与发射截面 (转换为 m^2)
# wl_pi_abs, val_pi_abs, func_pi_abs = load_and_interpolate('engineering/config/datafile/Yb_CALGO/pai_abs.csv', y_multiplier=1e-24)
# wl_pi_emi, val_pi_emi, func_pi_emi = load_and_interpolate('engineering/config/datafile/Yb_CALGO/pai_emi.csv', y_multiplier=1e-24)

# # 2. 读取 σ 偏振下的吸收与发射截面 (转换为 m^2)
# wl_sigma_abs, val_sigma_abs, func_sigma_abs = load_and_interpolate('engineering/config/datafile/Yb_CALGO/sigma_abs.csv', y_multiplier=1e-24)
# wl_sigma_emi, val_sigma_emi, func_sigma_emi = load_and_interpolate('engineering/config/datafile/Yb_CALGO/sigma_emi.csv', y_multiplier=1e-24)

# # 3. 读取种子光谱 (光谱通常只看相对强度，缩放系数设为 1.0)
# wl_seed, val_seed, func_seed = load_and_interpolate('engineering/config/datafile/seed_spectrum.csv', y_multiplier=1.0)

# # ==========================================
# # 可视化测试：检查插值函数是否工作正常
# # ==========================================
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
# plt.rcParams['axes.unicode_minus'] = False

# # 生成一个比原始数据更密集的测试波长网格 (900nm 到 1100nm)
# test_wavelengths_nm = np.linspace(900, 1100, 500)
# test_wavelengths_m = test_wavelengths_nm * 1e-9

# fig, (ax1, ax2) = plt.subplots(figsize=(14, 6), ncols=2)

# # 调用生成的插值函数，获取平滑的曲线数值 (为了画图好看，除以 1e-24 换回常用单位)
# ax1.plot(test_wavelengths_nm, func_sigma_abs(test_wavelengths_m) / 1e-24, 'b-', label='$\\sigma$-吸收 (插值)')
# ax1.plot(test_wavelengths_nm, func_sigma_emi(test_wavelengths_m) / 1e-24, 'r-', label='$\\sigma$-发射 (插值)')

# # 画上原始的散点作为对比
# ax1.plot(wl_sigma_abs * 1e9, val_sigma_abs / 1e-24, 'k.', alpha=0.3, label='原始数据点')

# ax1.set_xlabel('波长 (nm)', fontsize=12)
# ax1.set_ylabel('截面 ($10^{-20}$ cm$^2$ / $10^{-24}$ m$^2$)', fontsize=12)
# ax1.set_title('晶体宽带截面读取与连续化处理', fontsize=14)
# ax1.grid(True, linestyle='--', alpha=0.6)
# ax1.legend()

# ax2.plot(test_wavelengths_nm, func_seed(test_wavelengths_m) / 1e-24, 'g-', label='种子光谱 (插值)')
# ax2.set_xlabel('波长 (nm)', fontsize=12)
# ax2.set_ylabel('intensity(au)', fontsize=12)
# ax2.set_title('种子光谱', fontsize=14)
# ax2.grid(True, linestyle='--', alpha=0.6)
# ax2.legend()
# plt.tight_layout()
# plt.show()