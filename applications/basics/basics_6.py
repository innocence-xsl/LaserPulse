"""
==============================================================================
文件名称: basics_6.py
所属部门: Applications (应用部)
主要功能: 光线经过增益介质传播的模拟（频域角度）
代码解读:
    种子光多次经过增益介质（再生放大过程、多程往返），
    记录频域、光谱的变化趋势，并观察增益窄化现象。
==============================================================================
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.fft import fft, fftfreq, fftshift

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"

c = 3e8
h = 6.62607015e-34

# 种子光参数
E_pulse = 5.2e-9
_tau_seed = 5e-12
w_seed = 100e-6
lambda_seed = 1030e-9

# 增益介质参数
thickness = 6e-3
N_total_base = 1.25e28
doping_at_percent = 1.5
tau_f = 420e-6
n_CALGO = 1.93
N_total = N_total_base * (doping_at_percent / 100)


def load_and_interpolate(csv_path, y_multiplier=1.0):
    """读取 CSV 数据并返回插值函数。"""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, header=None)
    wavelength_m = df.iloc[:, 0].to_numpy(dtype=float) * 1e-9
    values = df.iloc[:, 1].to_numpy(dtype=float) * y_multiplier
    interp_func = interp1d(wavelength_m, values, kind="linear", bounds_error=False, fill_value=0.0)
    return wavelength_m, values, interp_func


# ==========================================
# 时域网格与脉冲构建
# ==========================================
omega_0 = 2 * np.pi * c / lambda_seed
Area_seed = np.pi * w_seed**2
P_peak_in = E_pulse / _tau_seed
I_peak_in = P_peak_in / Area_seed
E_amp = np.sqrt(I_peak_in)

T_window = 10 * _tau_seed
dt = 0.5e-15
N_time = int(T_window / dt)
N_time = 2 ** (N_time - 1).bit_length()
t_array = np.arange(-N_time / 2, N_time / 2) * dt

CEP = 0
chirp_a = 0
A_t = E_amp * np.exp(-2 * np.log(2) * (t_array / _tau_seed) ** 2)
Phi_t = omega_0 * t_array + CEP + chirp_a * (t_array**2)
E_t = A_t * np.exp(1j * Phi_t)

# ==========================================
# FFT 与频域物理坐标轴映射
# ==========================================
E_f_raw = fft(E_t)
f_array_raw = fftfreq(N_time, dt)
E_f = fftshift(E_f_raw)
f_array = fftshift(f_array_raw)

pos_mask = f_array > 0
f_pos = f_array[pos_mask]
E_f_pos = E_f[pos_mask]
lambda_array = c / f_pos

# ==========================================
# 再生放大循环
# ==========================================
max_round_trips = 50
T_rt = 0.95
V_mode = Area_seed * thickness

N2_current = N_total * 0.5
N1_current = N_total - N2_current

energy_history = []
spectrum_history = []

wl_sigma_abs, val_sigma_abs, func_sigma_abs = load_and_interpolate(DATA_DIR / "Yb_CALGO" / "sigma_abs.csv", y_multiplier=1e-24)
wl_sigma_emi, val_sigma_emi, func_sigma_emi = load_and_interpolate(DATA_DIR / "Yb_CALGO" / "sigma_emi.csv", y_multiplier=1e-24)
wl_seed, val_seed, func_seed = load_and_interpolate(DATA_DIR / "seed_spectrum.csv", y_multiplier=1.0)

# 真实种子光谱注入
S_lambda_real = func_seed(lambda_array)
phase_f = np.angle(E_f_pos)
E_f_pos = np.sqrt(np.clip(S_lambda_real, 0.0, None)) * np.exp(1j * phase_f)

sum_S_current = np.sum(np.abs(E_f_pos) ** 2)
if sum_S_current > 0:
    scaling_factor = np.sqrt(E_pulse / sum_S_current)
    E_f_pos = E_f_pos * scaling_factor

E_current = E_pulse
spectrum_initial = np.abs(E_f_pos) ** 2

for trip in range(max_round_trips):
    S_lambda_in = np.abs(E_f_pos) ** 2
    sum_S_in = max(np.sum(S_lambda_in), 1e-30)

    sigma_e = func_sigma_emi(lambda_array)
    sigma_a = func_sigma_abs(lambda_array)

    G_lambda = (sigma_e * N2_current - sigma_a * (N_total - N2_current)) * thickness
    E_f_pos = E_f_pos * np.exp(G_lambda / 2)

    S_lambda_out = np.abs(E_f_pos) ** 2
    sum_S_out = np.sum(S_lambda_out)

    k_energy = E_current / sum_S_in
    delta_E_array = k_energy * (S_lambda_out - S_lambda_in)
    delta_photons = np.sum(delta_E_array * lambda_array / (h * c))

    N2_current = max(0.0, N2_current - delta_photons / V_mode)
    N1_current = N_total - N2_current

    E_amplified = E_current * (sum_S_out / sum_S_in)

    E_f_pos = E_f_pos * np.sqrt(T_rt)
    E_current = E_amplified * T_rt

    energy_history.append(E_current)
    if trip % 10 == 0 or trip == max_round_trips - 1:
        spectrum_history.append(np.abs(E_f_pos) ** 2)

    print(f"第 {trip+1:02d} 圈: 能量 = {E_current*1e3:.4f} mJ, N2剩余占比 = {N2_current/N_total*100:.1f}%")

print("循环结束！")

# ==========================================
# 结果可视化
# ==========================================
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

trips = np.arange(1, max_round_trips + 1)
energy_mJ = np.array(energy_history) * 1e3
best_trip = int(np.argmax(energy_history) + 1)
best_energy_mJ = float(np.max(energy_history) * 1e3)

ax1.plot(trips, energy_mJ, "r-o", linewidth=2, markersize=4, label="腔内脉冲能量")
ax1.axvline(x=best_trip, color="k", linestyle="--", alpha=0.5, label=f"最佳倒出点 (Trip {best_trip})")
ax1.set_xlabel("循环圈数 (Round Trips)", fontsize=12)
ax1.set_ylabel("能量 (mJ)", color="r", fontsize=12)
ax1.tick_params(axis="y", labelcolor="r")
ax1.set_title("再生腔内脉冲能量演化", fontsize=14)
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(loc="upper left")

idx_mid = len(spectrum_history) // 2
S_init = spectrum_initial
S_mid = spectrum_history[idx_mid]
S_final = spectrum_history[-1]

S_init_norm = S_init / max(np.max(S_init), 1e-30)
S_mid_norm = S_mid / max(np.max(S_mid), 1e-30)
S_final_norm = S_final / max(np.max(S_final), 1e-30)

lambda_nm = lambda_array * 1e9

ax2.plot(lambda_nm, S_init_norm, "k-", linewidth=2, alpha=0.5, label="注入种子光谱")
ax2.plot(lambda_nm, S_mid_norm, "b-", linewidth=2, label="放大中期光谱")
ax2.plot(lambda_nm, S_final_norm, "r-", linewidth=2, label="最终饱和光谱")

ax2.set_xlim(990, 1080)
ax2.set_xlabel("波长 (nm)", fontsize=12)
ax2.set_ylabel("归一化光谱强度", fontsize=12)
ax2.set_title("频域增益窄化演化 (Gain Narrowing)", fontsize=14)
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend()

print(f"最佳倒出点: 第 {best_trip} 圈，峰值能量约 {best_energy_mJ:.4f} mJ")

plt.tight_layout()
plt.savefig("utils/record/basics_6.png", dpi=300)
plt.show()
