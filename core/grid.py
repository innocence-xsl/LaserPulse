"""
==============================================================================
文件名称: grid.py
所属部门: Core (核心基建部)
主要功能: 物理坐标网格系统 (时空/频域标尺)
代码解读: 
    这里是网格坐标系统，计算的 “坐标系”。
    定义模拟的网格（时间网格 / 频率网格），激光脉冲模拟的核心是在网格上做数值计算，
    这个类确保了时域和频域网格满足 FFT 的要求（等间距、中心对称），并提供了必要的物理参数（如中心波长、频率范围）来指导模拟。

    时域刻度：用来描述脉冲在时间轴上的包络形状和相位（如啁啾）
    频域刻度：用来描述脉冲的光谱分布和相位（如色散）
==============================================================================
"""

import numpy as np
import scipy.constants as const

class Grid:
    """
    时空/频域物理坐标网格类
    为超短脉冲提供严格满足 FFT 要求的等间距频率和时间网格。
    """
    def __init__(self, points: int, central_wl: float, max_wl: float):
        """
        参数:
        points : int
            网格点数，必须是 2 的指数（例如 2048, 4096, 8192），以保证 FFT 效率。
        central_wl : float
            中心波长 [m] (例如 1030e-9)。这是整个频率网格的中心点。
        max_wl : float
            最大波长 [m]。用于界定网格的边界宽度。
        """
        # 1. 基础网格参数
        self.points = points
        self.midpoint = int(points / 2)
        # 构造以 0 为中心的索引数组: [-N/2, ..., N/2 - 1]
        axis = np.linspace(-self.midpoint, self.midpoint - 1, self.points)

        # 2. 核心频率计算
        self.lambda_c = central_wl
        self.lambda_max = max_wl
        self.f_c = const.c / self.lambda_c  # 中心频率 [Hz]
        self.omega_c = 2 * np.pi * self.f_c # 中心角频率 [rad/s]
        
        # 频率窗口宽度: f_range = 2 * (f_c - f_min)
        # 这保证了 f_c 严格位于网格的正中心
        self.f_range = 2 * (const.c / self.lambda_c - const.c / self.lambda_max)
        self.df = self.f_range / self.points       # 频率步长 [Hz]
        
        # 3. 频域网格
        self.f_window = self.f_c + self.df * axis  # 真实频率网格 [Hz]
        self.omega_window = 2 * np.pi * self.f_window # 角频率网格 [rad/s]
        
        # 4. 波长网格 (注意：这是非等间距的！)
        self.lambda_window = const.c / self.f_window  # 对应的波长网格 [m]
        
        # 5. 时域网格
        self.t_range = 1 / self.df                 # 时间窗口总宽度 [s]
        self.dt = self.t_range / self.points       # 时间步长 [s]
        self.time_window = self.dt * axis          # 真实时间网格 [s]

        # 6. FFT 缩放因子 (用于保持物理能量在时频域变换时的量纲一致性)
        self.FFT_scale = np.sqrt(2 * np.pi) / self.dt

        # 7. 相对角频率网格 (用于计算色散)
        self.delta_omega = self.omega_window - self.omega_c