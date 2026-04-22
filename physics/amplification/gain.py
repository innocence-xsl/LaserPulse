"""
==============================================================================
文件名称: gain.py
所属部门: physics/amplification (物理部/放大模型)
主要功能: 增益计算
代码解读: 
    计算放大器的增益
    不依赖任何外部对象，只负责执行最底层的数学与物理映射。
    输入初始状态 -> 执行物理方程 -> 输出最终状态。
==============================================================================
"""
import numpy as np

# 物理常数
h = 6.62607015e-34  # 普朗克常数 [J·s]
c = 3e8             # 光速 [m/s]

def amplify_spectral_broadband(
    spectrum_in_intensity: np.ndarray,
    lambda_grid: np.ndarray,
    N_upper_start: float,
    N_total: float,
    sigma_em_grid: np.ndarray,
    sigma_abs_grid: np.ndarray,
    thickness: float,
    mode_area: float,
    num_slices: float = 1.0) -> tuple[np.ndarray, float, float]:
    """
    宽带光谱行波放大算子 (整合增益窄化与宏观能量守恒)
    
    参数:
        spectrum_in_intensity: 输入光谱强度分布 [a.u. 或 W/m^2/Hz均可，物理算子内部会做相对计算]
        lambda_grid: 波长网格 [m]
        N_upper_start: 初始上能级粒子数密度 [m^-3]
        N_total: 总掺杂离子密度 [m^-3]
        sigma_em_grid: 发射截面数组 [m^2]
        sigma_abs_grid: 吸收截面数组 [m^2]
        thickness: 增益介质厚度 [m]
        mode_area: 激光模式有效面积 [m^2]
        num_slices: 空间切片数量 (用于多步微分求解，默认为1即单程整体近似)
        
    返回:
        spectrum_out_intensity: 放大后的光谱强度分布
        N_upper_end: 消耗后剩余的上能级粒子数密度
        macroscopic_gain: 宏观单程能量增益
    """
    # 严格的类型转换：确保循环次数必须是整数，防止类型错误
    steps = int(num_slices)
    dz = thickness / steps

    # 建立内部工作变量
    N_upper_current = N_upper_start
    spectrum_current = np.copy(spectrum_in_intensity)
    V_slice = mode_area * dz  # 当前空间切片的有效相互作用体积

    # 提前计算初始输入状态的总量，避免后续重复计算
    sum_S_in_total = np.sum(spectrum_in_intensity)

    # 除零保护机制，设定一个安全下限以防归一化时报错
    sum_S_in_total = max(sum_S_in_total, 1e-20)

    # --- 空间切片步进循环 ---
    for _ in range(steps):
        N_lower_current = max(N_total - N_upper_current, 0.0)
        # 1. 计算局部小信号增益系数
        # 注意命名隔离，g_coef 代表系数，G_exp 代表宏观的总指数增益
        g_coef_lambda = (sigma_em_grid * N_upper_current) - (sigma_abs_grid * N_lower_current)

        # 2. 计算当前切片的指数单程增益
        G_exp_lambda = np.exp(g_coef_lambda * dz)

        # 3. 频域电场(强度)演化：光谱整形与增益窄化
        spectrum_out = spectrum_current * G_exp_lambda

        # 4. 计算宏观能量守恒与反转粒子消耗
        sum_S_current = max(np.sum(spectrum_current), 1e-20)
        sum_S_out = np.sum(spectrum_out)

        # 增量光谱形状 (代表真正被激发出光子的光谱分布)
        delta_spectrum = spectrum_out - spectrum_current

        # 将数学数组面积的增量，反向映射为物理光子数的消耗
        # 公式: delta_photons = sum( delta_S * lambda / (hc) ) * (某个常数比例因子)
        # 这里我们利用各波长切片的权重占比，结合总体能量变化来推算消耗的粒子密度

        # 为了物理算子的通用性，这里假设外部传递进来的 spectrum 就是具有能量物理意义的通量分布。
        photon_energy_grid = (h * c) / lambda_grid

        # 精确计算消耗的粒子数密度: dN = Energy_extracted / (Photon_Energy * Volume)
        consumed_density = np.sum(delta_spectrum / photon_energy_grid) / V_slice

        # 5. 更新系统状态 (指数衰减的底层物理下限保护，防止 N_upper 被抽空至负数)
        N_upper_current = max(0.0, N_upper_current - consumed_density)

        # 准备进入下一个空间切片
        spectrum_current = spectrum_out

        # 计算物理宏观增益 ===
        sum_S_out_total = np.sum(spectrum_current)
        # 宏观单程能量增益 = 输出能量 / 输入能量
        macroscopic_gain = sum_S_out_total / sum_S_in_total

    return spectrum_current, N_upper_current, macroscopic_gain
