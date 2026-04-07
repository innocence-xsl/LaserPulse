"""
==============================================================================
文件名称: basics_3.py
所属部门: Applications (应用部)
主要功能: 构建高斯脉冲并绘制时域光强图和频域光谱图
代码解读: 
    老子开始手搓代码了！！！
    根据平面波传输理论、高斯光束传播、载波包络相位、啁啾分析等基础理论，构建一个脉冲：
    初始化中心波长 1030 nm、脉宽 200 fs 的高斯脉冲，
    并画出它的时域光强图和频域光谱图，并搞清楚光谱图到底能给出什么信息。
==============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift

# ==========================================
# 一. 定义常数与网格初始化
# ==========================================
# 网格初始化
c = 3e8 
lambda_0 = 1030e-9 # 中心波长
omega_0 = 2 * np.pi * c / lambda_0 # 中心角频率
tau_fwhm = 200e-15 # 脉宽
# 定义时间轴
T_window = 100 * tau_fwhm # 时间窗口，保证包络衰减到足够小
dt = 0.1e-15 # 时间步长(s),必须远小于一个光周期(~3.43fs)
t_array = np.arange(-T_window/2, T_window/2, dt) # 时间轴数组

# ==========================================
# 二. 构建时域高斯脉冲
# ==========================================
# --- 构建高斯脉冲 ---
CEP = 0 # 载波包络相位(弧度)
chirp_a = 0 # 线性啁啾系数，先设为0（无啁啾变换极限脉冲）

# 1. 计算高斯强度包络 (振幅)
# 公式: A(t) = exp( -2 * ln(2) * (t / tau_fwhm)^2 )
A_t = np.exp(-2 * np.log(2) * (t_array / tau_fwhm)**2)

# 2. 计算总相位 (载波 + CEP + 啁啾)
# 公式: Phi(t) = omega_0 * t + CEP + chirp_a * t^2
Phi_t = omega_0 * t_array + CEP + chirp_a * (t_array**2)

# 3. 计算复电场
# 公式: E(t) = A(t) * exp(1j * Phi(t))
E_complex = A_t * np.exp(1j * Phi_t) # 复电场
E_real = np.real(E_complex)  # 真实电场（实部）
I_t = np.abs(E_complex)**2 # 时域光强 (正比于复电场模的平方) 

# ==========================================
# 三. 频域转换 (FFT)
# ==========================================
# 1. 获取数组长度
N = len(t_array) # 时间点数量

# 2. 执行快速傅里叶变换
E_f_raw = fft(E_complex) # 频域复电场

# 3. 生成对应的频率物理坐标轴
f_array_raw = fftfreq(N, dt) # 频率数组

# 4. 移位对齐：将零频移到数组中心，使得频率从负到正递增
E_f_shifted = fftshift(E_f_raw) # 将频谱中心移到零频率
f_array_shifted = fftshift(f_array_raw) # 将频率数组中心移到零频率

# 5. 光谱画图通常只关心正频率部分（物理上负频率没有实际意义，只是数学对称）
pos_mask = f_array_shifted > 0  # 严格大于0
f_physical = f_array_shifted[pos_mask]  # 正频率数组、物理频率
E_f_physical = E_f_shifted[pos_mask]  # 物理频域电场、正频率部分的复电场模

# 6. 转换为波长坐标并计算光谱强度
lambda_array = c / f_physical  # 波长数组、物理波长，单位：米，避免除0错误
S_lambda = np.abs(E_f_physical)**2 # 光谱强度数组、物理波长数组的光谱强度

# 7. 归一化光强和光谱，方便画图对比
I_t_norm = I_t / np.max(I_t)
E_real_norm = E_real / np.max(A_t)
S_lambda_norm = S_lambda / np.max(S_lambda)

# ==========================================
# 4. 绘图 (时域光强与频域光谱)
# ==========================================
# 设置字体防止乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# --- 图1：时域 ---
# 画包络和实际电场，时间轴换算为 fs
ax1.plot(t_array * 1e15, I_t_norm, 'k--', linewidth=2, label='光强包络 $I(t)$')
ax1.plot(t_array * 1e15, E_real_norm, 'r-', linewidth=0.5, alpha=0.8, label='真实电场 $E(t)$')
ax1.plot(t_array * 1e15, A_t / np.max(A_t), 'b-', linewidth=2, label='振幅包络 $A(t)$')
ax1.set_xlim(-400, 400) # 放大看脉冲中心区域
ax1.set_xlabel('时间 (fs)', fontsize=12)
ax1.set_ylabel('归一化强度 / 振幅', fontsize=12)
ax1.set_title(f'时域脉冲 (无啁啾 {tau_fwhm * 1e15} fs)', fontsize=14)
ax1.legend()
ax1.grid(True, linestyle=':')

# --- 图2：频域光谱 ---
# 波长换算为 nm
ax2.plot(lambda_array * 1e9, S_lambda_norm, 'b-', linewidth=2, label='光谱强度 $S(\\lambda)$')
ax2.set_xlim(1015, 1045) # 1030nm 附近
ax2.set_xlabel('波长 (nm)', fontsize=12)
ax2.set_ylabel('归一化光谱强度', fontsize=12)
ax2.set_title(f'频域光谱 (中心波长 {lambda_0 * 1e9:.2f} nm)', fontsize=14)
ax2.legend()
ax2.grid(True, linestyle=':')

plt.tight_layout()
plt.savefig('utils/record/basics_3.png', dpi=300)
plt.show()