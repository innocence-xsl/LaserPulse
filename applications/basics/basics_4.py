"""
==============================================================================
文件名称: basics_4.py
所属部门: Applications (应用部)
主要功能: 光线经过介质传播的基本模拟
代码解读: 
    让给定参数值的种子光，经过自由空间、一块透明介质，看光束变化（包括前后对比的能量、增益、光谱信息，三联图）
==============================================================================
"""

import sys
from pathlib import Path

# 获取项目根目录：
# 当前文件在 applications/basics/basics_4.py
# parents[0] = applications/basics
# parents[1] = applications
# parents[2] = LaserPulse_v2 项目根目录
project_root = Path(__file__).resolve().parents[2]

# 把项目根目录加入 Python 搜索路径
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import physics.optics_tools as optics_tools

# ==============================================================================
# 参数设置
# ==============================================================================
c = 3e8 # 光速
lambda_0 = 1030e-9 # 中心波长
w_0 = 100e-6 # 光斑束腰半径
E_pulse = 1e-9 # 单脉冲能量
tau = 100e-15 # 脉宽

t_array = np.linspace(-5e-12, 5e-12, 16384) # 时间轴
L_medium = 10 # 介质长度
material_name = "Si3N4" # 介质名称

# ==============================================================================
# 自由空间传输 (Propagation in Free Space)
# 计算短脉冲在自由空间中传输 L 距离后的时域展宽和相位变化
# ==============================================================================
def propagate_free_space(z, w_0, lambda_0, E_pulse, tau):
    # 1. 计算瑞利长度 (Rayleigh length)
    z_R = (np.pi * w_0**2) / lambda_0

    # 2. 计算衍射导致的光斑半径扩张
    w_z = w_0 * np.sqrt(1 + (z / z_R)**2)

    # 3. 计算此时的光束有效面积 (1/e^2 面积)
    Area_z = (np.pi * w_z**2) / 2

    # 4. 计算脉冲的峰值功率
    P_peak = E_pulse / tau

    # 5. 计算 z 处的峰值光强
    I_peak_z = P_peak / Area_z

    return w_z, I_peak_z

# ==============================================================================
# 介质传输 (Propagation in Dispersive Medium)
# 计算短脉冲在色散介质中传输 L 距离后的时域展宽和相位变化
# 核心物理：线性色散导致的时间展宽（频域加相位）
# ==============================================================================
def propagate_in_medium(t_array, E_t_in, L_medium, material_csv_path):
    # 1. 时域转频域 (使用 fftshift 保证频率轴连续有序)
    dt = t_array[1] - t_array[0]
    E_omega = np.fft.fftshift(np.fft.fft(E_t_in)) # 时域转频域
    omega_array = np.fft.fftshift(np.fft.fftfreq(len(t_array), d=dt)) * 2 * np.pi # 转换为角频率

    # 2. Sellmeier方程只对正波长有效
    n_omega = np.ones_like(omega_array) # 默认折射率为1（真空）
    valid_idx = omega_array > 0         # 提取正频率
    lambda_array = 2 * np.pi * c / omega_array[valid_idx]

    # 3. 调用库函数计算真实材料的折射率 (基于 UVFS.csv)
    n_omega[valid_idx] = optics_tools.sellmeier_index(lambda_array, material_csv_path)

    # 4. 在频域施加相位调制：phi(omega) = n(omega) * omega * L / c
    phi_omega = (n_omega * omega_array / c) * L_medium
    # 找到中心角频率 omega_0 对应的索引
    omega_0 = 2 * np.pi * c / 1030e-9
    idx_0 = np.argmin(np.abs(omega_array - omega_0))
    # 用中心频率附近的点，通过差分计算群延迟斜率 (d_phi / d_omega)
    d_omega = omega_array[idx_0 + 1] - omega_array[idx_0 - 1]
    d_phi = phi_omega[idx_0 + 1] - phi_omega[idx_0 - 1]
    t_delay = d_phi / d_omega  # 脉冲在玻璃里跑的总时间
    phi_0 = phi_omega[idx_0]   # 绝对相位

    # 减去绝对相位和导致整体平移的线性相位
    # 剩下的 phi_dispersion 就只有纯纯的色散（展宽和啁啾）了！
    phi_dispersion = phi_omega - phi_0 - t_delay * (omega_array - omega_0)
    E_omega_out = E_omega * np.exp(-1j * phi_dispersion)

    # 5. 逆傅里叶变换回到时域 (注意要先 ifftshift 移回去)
    E_t_out = np.fft.ifft(np.fft.ifftshift(E_omega_out))

    # 6. 计算光强
    I_t_out = np.abs(E_t_out)**2
    
    return E_t_out, I_t_out


# ==============================================================================
# 自由空间传播主流程
# ==============================================================================
# if __name__ == "__main1__":
#     # --- 初始化光源 ---
#     omega_0 = 2 * np.pi * c / lambda_0
#     P_peak = E_pulse / tau  # 估算峰值功率
#     E_t_in = np.sqrt(P_peak) * np.exp(-2 * np.log(2) * (t_array / tau)**2) * np.exp(1j * omega_0 * t_array)
#     I_t_in = np.abs(E_t_in)**2

#     # --- 阶段 1：自由空间传输 ---
#     print(f"光束在自由空间传输 {L_medium} 米...")
#     # 传输一段距离后的光斑变化
#     w_out, I_peak_out = propagate_free_space(L_medium, w_0, lambda_0, E_pulse, tau)
#     print(f"   -> 初始光强: {P_peak / (np.pi*w_0**2/2):.2e} W/m^2")
#     print(f"   -> 传输后光强: {I_peak_out:.2e} W/m^2")

#     # 空间数据
#     r_in, I_r_in = optics_tools.get_spatial_profile(w_0, P_peak / (np.pi * w_0**2 / 2))
#     r_out, I_r_out = optics_tools.get_spatial_profile(w_out, I_peak_out)
#     # 光谱数据
#     lambda_plot, I_lambda_in = optics_tools.get_spectrum_from_time_domain(t_array, E_t_in, lambda_range=(980e-9, 1080e-9))

#     # --- 画三联图 ---
#     plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
#     plt.rcParams['axes.unicode_minus'] = False

#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

#     # --- 图1：横截面光强分布 (空间) ---
#     ax1.plot(r_in * 1e3, I_r_in, 'b--', label=f'初始 (w0={w_0*1e6:.0f}μm)')
#     ax1.plot(r_out * 1e3, I_r_out, 'r-', label=f'传输后 (w={w_out*1e6:.0f}μm)')
#     ax1.set_xlabel('径向位置 r (mm)')
#     ax1.set_ylabel('光强 (W/m^2)')
#     ax1.set_title('1. 自由空间: 横截面光强发散')
#     ax1.legend()
#     ax1.grid(True, linestyle='--', alpha=0.7)
    
#     # --- 图2：时域分布 (时间) ---
#     ax2.plot(t_array * 1e15, I_t_in, 'k-')
#     ax2.set_xlabel('时间 t (fs)')
#     ax2.set_ylabel('时域光强 (a.u.)')
#     ax2.set_title(f'2. 时域波形 (脉宽~{tau*1e15:.0f}fs)')
#     ax2.grid(True, linestyle='--', alpha=0.7)
#     ax2.set_xlim(-1000, 1000)

#     # --- 图3：频域光谱 (频率) ---
#     ax3.plot(lambda_plot * 1e9, I_lambda_in, 'g-')
#     ax3.set_xlabel('波长 λ (nm)')
#     ax3.set_ylabel('光谱强度 (a.u.)')
#     ax3.set_title(f'3. 初始光谱 (中心~{lambda_0*1e9:.0f}nm)')
#     ax3.grid(True, linestyle='--', alpha=0.7)

#     plt.tight_layout()
#     plt.savefig('utils/record/basics_4_free_space.png', dpi=300)
#     plt.show()



# ==============================================================================
# 透明介质传播主流程
# ==============================================================================
if __name__ == "__main__":
    # --- 初始化光源 ---
    omega_0 = 2 * np.pi * c / lambda_0
    P_peak = E_pulse / tau  
    E_t_in = np.sqrt(P_peak) * np.exp(-2 * np.log(2) * (t_array / tau)**2) * np.exp(1j * omega_0 * t_array)
    I_t_in = np.abs(E_t_in)**2

    # --- 阶段 2：色散介质传输 ---
    material_file = "./engineering/config/datafile/UVFS.csv" 
    print(f"光束进入长度为 {L_medium} 米的 UVFS 玻璃...")
    E_t_out, I_t_out = propagate_in_medium(t_array, E_t_in, L_medium, material_file)

    # 获取出入介质前后的光谱数据
    lambda_plot, I_lambda_in = optics_tools.get_spectrum_from_time_domain(t_array, E_t_in, lambda_range=(980e-9, 1080e-9))
    _, I_lambda_out = optics_tools.get_spectrum_from_time_domain(t_array, E_t_out, lambda_range=(980e-9, 1080e-9))

    # --- 画三联图 ---
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))

    # 图 1：时域分布
    ax1.plot(t_array * 1e15, I_t_in, 'k-', label=f'初始脉冲 ({tau*1e15:.0f} fs)')
    ax1.plot(t_array * 1e15, I_t_out, 'r-', linewidth=2, label='出玻璃后 (啁啾脉冲)')
    ax1.set_xlabel('时间 t (fs)')
    ax1.set_ylabel('时域光强 (W/m^2)')
    ax1.set_title('时域分布（色散导致展宽）')
    ax1.set_xlim(-500, 500) 
    ax1.legend()
    ax1.grid(True, alpha=0.5)

    # 图 2：频域光谱
    ax2.plot(lambda_plot * 1e9, I_lambda_in, 'k-', linewidth=4, label='初始光谱', alpha=0.5)
    ax2.plot(lambda_plot * 1e9, I_lambda_out, 'r--', linewidth=2, label='出玻璃后光谱')
    ax2.set_xlabel('波长 λ (nm)')
    ax2.set_ylabel('光谱强度 (a.u.)')
    ax2.set_title('光谱守恒 (线性色散)')
    ax2.legend()
    ax2.grid(True, alpha=0.5)

    # --- 图3：频域光谱 (频率) ---
    ax3.plot(lambda_plot * 1e9, I_lambda_in, 'g-')
    ax3.set_xlabel('波长 λ (nm)')
    ax3.set_ylabel('光谱强度 (a.u.)')
    ax3.set_title(f'3. 初始光谱 (中心~{lambda_0*1e9:.0f}nm)')
    ax3.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('utils/record/basics_4_in_medium.png', dpi=300)
    plt.show()
