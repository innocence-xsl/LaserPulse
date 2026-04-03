"""
==============================================================================
文件名称: grid.py
所属部门: Core (核心基建部)
主要功能: 物理坐标网格系统 (时空/频域标尺)
代码解读: 
    这里是网格坐标系统，计算的 “坐标系”。
    定义模拟的网格（时间网格 / 频率网格），激光脉冲模拟的核心是在网格上做数值计算，
    补全了非等距波长积分微元、光子能量网格、预设的 FFT-Shift 序列以及持久化功能,
    这个类确保了时域和频域网格满足 FFT 的要求（等间距、中心对称），并提供了必要的物理参数（如中心波长、频率范围）来指导模拟。

    时域刻度：用来描述脉冲在时间轴上的包络形状和相位（如啁啾）
    频域刻度：用来描述脉冲的光谱分布和相位（如色散）
==============================================================================
"""

import os
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
            网格点数，必须是 2 的指数（例如 2048, 4096），以保证 FFT 效率。
        central_wl : float
            中心波长 [m] (例如 1030e-9)。这是整个频率网格的中心点。
        max_wl : float
            最大波长 [m]。用于界定网格的边界宽度。
        """
        # 1. 基础网格参数
        self.points = points
        self.midpoint = int(points / 2)
        
        # 使用 arange 生成严格间距为 1 的索引序列，比 jsfeehan 的 linspace 逻辑更清晰且避免浮点误差
        axis = np.arange(-self.midpoint, self.midpoint)

        # 2. 核心频率计算
        self.lambda_c = central_wl
        self.lambda_max = max_wl
        self.f_c = const.c / self.lambda_c  # 中心频率 [Hz]
        self.omega_c = 2 * np.pi * self.f_c # 中心角频率 [rad/s]
        
        # 频率窗口宽度: f_range = 2 * (f_c - f_min)
        self.f_range = 2 * (const.c / self.lambda_c - const.c / self.lambda_max)
        self.df = self.f_range / self.points       # 频率步长 [Hz]
        self.dOmega = 2 * np.pi * self.df          # 角频率步长 [rad/s]
        
        # 3. 频域网格 (相对与绝对)
        self.omega = self.dOmega * axis               # 以 0 为中心的相对角频率网格 [rad/s]
        self.omega_window = self.omega_c + self.omega # 绝对角频率网格 [rad/s]
        
        # 【核心修正】预计算 FFT 移位后的频率网格，极大地优化后续分裂步长傅里叶法 (SSFM) 的效率
        self.omega_window_shift = np.fft.fftshift(self.omega_window)
        
        self.f_window = self.omega_window / (2 * np.pi) # 绝对频率网格 [Hz]
        
        # 4. 波长网格 
        self.lambda_window = const.c / self.f_window  # 对应的波长网格 [m]
        self.lambda_min = self.lambda_window.min()
        
        # 【核心修正】计算波长积分微元 d_wl。因为波长网格非等间距，积分或功率密度转换时必须使用此梯度！
        self.d_wl = np.gradient(-1 * self.lambda_window)
        
        # 5. 【核心修正】能量网格 (计算有源光纤/晶体增益时必须的基础标尺)
        self.energy_window = const.h * self.f_window
        self.energy_window_shift = np.fft.fftshift(self.energy_window)

        # 6. 时域网格
        self.t_range = 1 / self.df                 # 时间窗口总宽度 [s]
        self.dt = self.t_range / self.points       # 时间步长 [s]
        self.time_window = self.dt * axis          # 真实时间网格 [s]

        # 7. FFT 缩放因子
        self.FFT_scale = np.sqrt(2 * np.pi) / self.dt

        # 保持与原 innocence-xsl 兼容的别名
        self.delta_omega = self.omega

    def save(self, directory: str):
        """
        【功能补充】保存网格信息以便后续复现模拟
        只需保存三个构建参数即可，避免保存庞大的网格数组。
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        filepath = os.path.join(directory, 'grid.npz')
        np.savez(filepath,
                 points=self.points,
                 lambda_c=self.lambda_c,
                 lambda_max=self.lambda_max)

    @classmethod
    def from_file(cls, data_directory: str):
        """
        【架构优化】使用类方法替代 jsfeehan 原始版本中多余的继承子类
        允许通过 Grid.from_file('path/to/dir') 直接从文件恢复网格实例。
        """
        filepath = os.path.join(data_directory, 'grid.npz')
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"找不到网格数据文件: {filepath}")
            
        grid_data = np.load(filepath)
        return cls(
            points=int(grid_data['points']),
            central_wl=float(grid_data['lambda_c']),
            max_wl=float(grid_data['lambda_max'])
        )