import os
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt


# 导入我们最新一代的 V8 引擎零件！
from system.parameter_loader import ParameterLoader
from components.active_components import AdvancedBulkCrystal
from components.passive_components import LossyMirror, SpectralFilter
from core.dataclasses_def import PhysicalConstants

def calculate_fwhm(wavelengths_nm, spectrum):
    """极其精准的半高全宽计算工具"""
    half_max = np.max(spectrum) / 2.0
    indices = np.where(spectrum >= half_max)[0]
    if len(indices) < 2:
        return 0.0
    return np.abs(wavelengths_nm[indices[-1]] - wavelengths_nm[indices[0]])

def run_dynamics(with_shaping: bool, max_rt=90):
    """
    战神级手动挡跑圈函数！
    不再是一键跑完，而是逐圈记录状态。
    """
    loader = ParameterLoader()
    cry_params = loader.get_crystal_params()
    pump_params = loader.get_pump_params()
    cav_params = loader.get_cavity_params()
    
    # 获取种子光，并对齐到 1040 nm
    seed_pulse, seed_params = loader.get_seed_params(align_to_peak=True, target_center_nm=1040.0)

    # ==========================================
    # 1. 光路注入设定 (对标 Fig 2a 的真实物理损耗)
    # ==========================================
    if with_shaping:
        pre_shaper = SpectralFilter(
            name="Pre_Shaper", center_wl=1040e-9, width=30e-9,
            depth_per_bounce=0.20, num_bounces=7
        )
        seed_pulse = pre_shaper.propagate(seed_pulse)
        # 整形器 + 腔注入的综合效率约 40%
        coupling_loss = LossyMirror(name="Injection", reflectivity=0.40)
        seed_pulse = coupling_loss.propagate(seed_pulse)
    else:
        # 无整形时的纯腔注入效率约 60%
        coupling_loss = LossyMirror(name="Injection", reflectivity=0.60)
        seed_pulse = coupling_loss.propagate(seed_pulse)

    # ==========================================
    # 2. 组装最强 V8 引擎晶体
    # ==========================================
    consts = PhysicalConstants()
    crystal = AdvancedBulkCrystal(
        name="Yb:CALGO", 
        crystal_params=cry_params, 
        seed_params=seed_params,
        pump_params=pump_params, 
        consts=consts, 
        pump_polarization='sigma',   # 必须是 sigma，大口干饭！
        signal_polarization='pi'  # 必须是 sigma，平坦放宽！
    )
    mirror = LossyMirror(name="Cavity_Mirror", reflectivity=1.0 - cav_params.loss_passive)

    # 🌟 绝杀：1.2ms 长充能，模拟 43kHz 连续泵浦下的稳态水池！
    # 如果输出功率不到 15.5W，可以在 1.2e-3 ~ 1.5e-3 之间微调这个值
    crystal.pump_crystal(dt=1.25e-3) 

    # ==========================================
    # 3. 手动挡跑圈，实时采样
    # ==========================================
    wavelengths_nm = seed_pulse.grid.lambda_window * 1e9
    rep_rate = 43000  # 文献给定的重复频率 43 kHz
    
    rt_list = []
    power_list = []
    bw_list = []

    for rt in range(1, max_rt + 1):
        # 穿过晶体 (放大)
        seed_pulse = crystal.propagate(seed_pulse)
        # 穿过腔镜 (损耗)
        seed_pulse = mirror.propagate(seed_pulse)
        
        # 采样当前能量并转换为平均功率 (W = J * Hz)
        current_energy = seed_pulse.get_energy()
        current_power = current_energy * rep_rate
        
        # 采样当前光谱并计算 FWHM
        spectrum = np.abs(seed_pulse.A_f)**2
        fwhm = calculate_fwhm(wavelengths_nm, spectrum)
        
        # 记录到小本本上
        rt_list.append(rt)
        power_list.append(current_power)
        bw_list.append(fwhm)
        
    return rt_list, power_list, bw_list

def main():
    print("🌟 正在复刻文献 Wang et al. (2021) Figure 2(c) 动态演化图 🌟\n")
    
    print("--- 🚀 启动 [无整形] 对照组 跑圈记录仪 ---")
    rt_unshaped, pwr_unshaped, bw_unshaped = run_dynamics(with_shaping=False, max_rt=90)
    
    print("--- 🚀 启动 [有整形] 实验组 跑圈记录仪 ---")
    rt_shaped, pwr_shaped, bw_shaped = run_dynamics(with_shaping=True, max_rt=90)
    
    print(f"\n✅ 跑圈结束！")
    print(f"最终功率对比 -> 无整形: {pwr_unshaped[-1]:.2f} W | 有整形: {pwr_shaped[-1]:.2f} W (目标: 约 15.5 W)")
    print(f"最终带宽对比 -> 无整形: {bw_unshaped[-1]:.2f} nm | 有整形: {bw_shaped[-1]:.2f} nm (目标: 9nm vs 19nm)")

    # ==========================================
    # 4. 严苛对标文献 Fig 2(c) 的顶刊画风
    # ==========================================
    print("\n📊 正在生成 Figure 2(c) 汇报图表...")
    
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    # 左 Y 轴：输出功率 (Output Power)
    ax1.plot(rt_unshaped, pwr_unshaped, 'k-o', linewidth=2, markersize=5, label='Power (without shaper)')
    ax1.plot(rt_shaped, pwr_shaped, 'r-^', linewidth=2, markersize=5, label='Power (with shaper)')
    
    ax1.set_xlabel("Amplification round trips", fontsize=14)
    ax1.set_ylabel("Output power (W)", fontsize=14)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 20)  # 留一点顶空给 15.5W
    ax1.tick_params(axis='both', labelsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 右 Y 轴：放大后带宽 (Amplified Bandwidth)
    ax2 = ax1.twinx()
    ax2.plot(rt_unshaped, bw_unshaped, 'b--o', linewidth=2, markersize=5, label='Bandwidth (without shaper)')
    ax2.plot(rt_shaped, bw_shaped, 'g--^', linewidth=2, markersize=5, label='Bandwidth (with shaper)')
    
    ax2.set_ylabel("Amplified bandwidth (nm)", fontsize=14)
    ax2.set_ylim(0, 25)  # 留顶空给 19nm
    ax2.tick_params(axis='y', labelsize=12)

    # 合并图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right', fontsize=11, frameon=True)

    plt.title("Evolution of Output Power and Spectral Bandwidth", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()