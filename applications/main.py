"""
==============================================================================
文件名称: main.py
所属部门: Applications（应用部）
主要功能: 主程序入口，负责协调所有组件的运行。
代码解读: 
    测试再生放大器的完整流程，包含腔外预整形、晶体放大、总控台跑圈，以及结果可视化。
==============================================================================
"""
# ==========================================
# 路径配置
# ==========================================
import sys
import os
# 获取当前文件（main.py）的路径
current_file_path = os.path.abspath(__file__)
# 获取项目根目录（LASERPULSE_V2/）：向上退一级（因为 main.py 在 applications/ 里）
project_root = os.path.dirname(os.path.dirname(current_file_path))
# 把项目根目录加入Python搜索路径
sys.path.append(project_root)

# ==========================================
# 导入核心组件
# ==========================================
import numpy as np
import matplotlib.pyplot as plt
from engineering.config.parameter_loader import ParameterLoader
from core.dataclasses_def import PhysicalConstants
from engineering.components.active_components import BulkCrystal
from engineering.components.passive_components import LossyMirror, SpectralFilter
from engineering.assembly.optical_assembly import RegenerativeAmplifierAssembly


def main():
    # ==========================================
    # 第一步：读取所有的物理参数和初始种子光
    # ==========================================
    print("📦 读取参数...")
    loader = ParameterLoader()
    
    # 依次拿到晶体、泵浦、腔和种子光的参数图纸
    cry_params = loader.get_crystal_params()
    pump_params = loader.get_pump_params()
    cav_params = loader.get_cavity_params()
    
    # 获取种子光：不仅给参数，还直接给了初始脉冲(Pulse)
    seed_pulse, seed_params = loader.get_seed_params()
    
    print(f"✅ 初始种子光能量: {seed_pulse.get_energy():.2e} J")

    # 🌟🌟🌟 完美复现 Wang et al. 2021 的腔外预整形 🌟🌟🌟
    if seed_params.shaping_depth > 0:
        print(f"\n🔪 正在启动腔外种子光预整形 (Multi-bounce Pre-shaping)...")
        
        # --- 物理参数反推 ---
        total_depth = seed_params.shaping_depth  # 比如文献中的 0.8 (80%)
        num_bounces = 7                          # 文献中的 7 次反射
        
        # 如果中心被挖掉了 80%，说明只剩下 20% (0.2) 透过去了
        target_center_transmission = 1.0 - total_depth
        single_bounce_transmission = target_center_transmission ** (1.0 / num_bounces)
        depth_per_bounce = 1.0 - single_bounce_transmission
        
        print(f"   -> 目标总挖坑深度: {total_depth*100:.1f}%, 反射次数: {num_bounces} 次")
        print(f"   -> 自动推算单次反射深度: {depth_per_bounce*100:.1f}%")
        
        # 召唤升级版的整形机
        pre_shaper = SpectralFilter(
            name="Extra_Cavity_Pre_Shaper",
            center_wl=seed_params.shaping_center,
            width=seed_params.shaping_width,
            depth_per_bounce=depth_per_bounce,
            num_bounces=num_bounces
        )
        
        # 让种子光穿过这台机器
        seed_pulse = pre_shaper.propagate(seed_pulse)
        print(f"✅ 预整形完成！整形后注入能量: {seed_pulse.get_energy():.2e} J")
    
    # ==========================================
    # 第二步：采购和安装机器
    # ==========================================
    consts = PhysicalConstants()
    
    # 1. 安装核心增益介质晶体 (v2 叫 BulkCrystal)
    crystal = BulkCrystal(
        name="Yb:CALGO_Crystal", 
        crystal_params=cry_params, 
        seed_params=seed_params, 
        pump_params=pump_params, 
        consts=consts,
        pump_polarization='pi',
        signal_polarization='sigma'  # 指定信号光偏振
    )
    
    # 2. 安装损耗镜（模拟腔内的固有损耗）
    mirror = LossyMirror(
        name="Cavity_Loss_Mirror", 
        reflectivity=1.0 - cav_params.loss_passive
    )

    # ==========================================
    # 第三步：给晶体充能（踩一脚油门！）
    # ==========================================
    print("\n⚡ 正在开启泵浦源，给晶体充能...")
    # 泵浦光预先照射一段时间（比如 1 毫秒）
    crystal.pump_crystal(dt=1e-3)
    print(f"✅ 充能完毕！当前晶体上能级粒子数密度: {crystal.N_upper:.2e} m^-3")

    # ==========================================
    # 第四步：组装流水线，开始跑圈！
    # ==========================================
    cavity_components = [crystal, mirror]
    
    # 成立总控台
    regen_amp = RegenerativeAmplifierAssembly(
        name="My_First_Regen_Amp", 
        cavity_components=cavity_components
    )
    
    # 厂长下达命令：让脉冲跑指定的圈数！(这里先跑50圈测试)
    final_pulse = regen_amp.simulate(seed_pulse=seed_pulse, num_round_trips=50)

    # ==========================================
    # 第五步：验收成果
    # ==========================================
    print("\n🎉 试车圆满结束！")
    print(f"🏆 最终输出脉冲能量: {final_pulse.get_energy():.2e} J")

    # ==========================================
    # 第六步：大屏幕数据可视化 (三联屏终极版)
    # ==========================================
    print("\n📊 正在生成总控室大屏幕汇报图表...")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    # ------------------------------------------
    # 📺 屏幕 1：能量暴涨曲线 (Energy vs. Round Trips)
    # ------------------------------------------
    round_trips = regen_amp.history['round_trip']
    energies = regen_amp.history['pulse_energy']

    ax1.plot(round_trips, energies, marker='o', color='red', linestyle='-', markersize=4)
    ax1.set_title("Pulse Energy Growth", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Round Trip Number", fontsize=12)
    ax1.set_ylabel("Energy (Joules)", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # ------------------------------------------
    # 📺 屏幕 2：最终输出光谱 (Final Spectrum)
    # ------------------------------------------
    wavelengths_nm = final_pulse.grid.lambda_window * 1e9
    final_spectrum = np.abs(final_pulse.A_f)**2

    ax2.plot(wavelengths_nm, final_spectrum, color='blue', linewidth=2)
    ax2.set_title("Final Output Spectrum", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Wavelength (nm)", fontsize=12)
    ax2.set_ylabel("Intensity (a.u.)", fontsize=12)
    ax2.set_xlim(1010, 1060)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, linestyle='--', alpha=0.7)

    # ------------------------------------------
    # 📺 屏幕 3：极限压缩后的时域脉冲 (FTL Time Domain)
    # ------------------------------------------
    # v2 专属升级：直接调用 Pulse 的内置方法获取 FTL 脉冲
    ftl_pulse = final_pulse.get_ftl_pulse()
    
    # 从底层 Grid 调取时间网格并转换为飞秒
    time_fs = ftl_pulse.grid.time_window * 1e15
    
    # 拿到纯粹的时域强度并归一化
    intensity_t = np.abs(ftl_pulse.A_t)**2
    intensity_t_norm = intensity_t / np.max(intensity_t)

    # 自动算出脉冲的半高全宽 (FWHM)
    half_max = 0.5
    indices = np.where(intensity_t_norm >= half_max)[0]
    fwhm_fs = time_fs[indices[-1]] - time_fs[indices[0]]
    
    print(f"✨ 极限脉冲宽度 (FTL FWHM): {fwhm_fs:.1f} fs ✨")

    ax3.plot(time_fs, intensity_t_norm, color='purple', linewidth=2)
    
    # 绘图高亮展示 FWHM
    ax3.fill_between(time_fs, intensity_t_norm, alpha=0.3, color='purple')
    ax3.axvspan(time_fs[indices[0]], time_fs[indices[-1]], color='yellow', alpha=0.3, label=f'FWHM: {fwhm_fs:.1f} fs')
    
    ax3.set_title("Transform-Limited Pulse", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Time (fs)", fontsize=12)
    ax3.set_ylabel("Normalized Intensity", fontsize=12)
    ax3.set_xlim(-500, 500) # 只看最核心的中心区域
    ax3.legend(loc="upper right", fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()