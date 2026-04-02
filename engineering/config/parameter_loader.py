"""
==============================================================================
文件名称: parameter_loader.py
所属部门: Engineering (工程部)
主要功能: 参数加载器
代码解读: 
    这里是参数加载器，
    负责从文本文件加载参数，并将实验数据插值到我们定义的物理网格上，
    支持从 `parameters` 文件夹加载参数，支持 `txt` 格式，
    同时支持从 `datafile` 文件夹加载光谱数据，支持 `csv` 格式。
==============================================================================
"""

import os
import numpy as np
import pandas as pd

from core.grid import Grid
from core.pulse import Pulse

from scipy.interpolate import interp1d
from dataclasses import fields
from core.dataclasses_def import *

class ParameterLoader:
    def __init__(self, grid_points=4096, central_wl=1030e-9, max_wl=1150e-9):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.grid = Grid(points=grid_points, central_wl=central_wl, max_wl=max_wl)
        
        self.master_wavelengths = self.grid.lambda_window

    def _get_path(self, folder, filename):
        return os.path.join(self.base_dir, folder, filename)

    def _load_txt_params(self, filename: str, dataclass_type):
        """核心解析器：精简、高效"""
        path = self._get_path('parameters', filename)
        if not os.path.exists(path):
             raise FileNotFoundError(f"❌ 关键文件缺失: {path}")

        allowed_fields = {f.name for f in fields(dataclass_type)}
        params_dict = {}

        with open(path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                # 1. 预处理：去注释、去空行
                content = line.split('#')[0].split('//')[0].strip()
                if not content or '=' not in content:
                    continue
                
                # 2. 解析：分割键值对
                key_str, val_str = content.split('=', 1)
                key = key_str.strip()
                
                # 3. 校验与转换：仅处理有效字段
                if key in allowed_fields:
                    try:
                        params_dict[key] = float(val_str.strip())
                    except ValueError:
                        continue 

        return dataclass_type(**params_dict)

    def _load_spectrum_csv(self, filename: str):
        path = self._get_path('datafile', filename)
        if not os.path.exists(path): return None, None
        
        try:
            df = pd.read_csv(path, header=None, names=['wl', 'val'], 
                            sep=r'[,\s]+', encoding='utf-8-sig', engine='python')
            df['wl'] = pd.to_numeric(df['wl'], errors='coerce')
            df['val'] = pd.to_numeric(df['val'], errors='coerce')
            
            df = df.dropna().sort_values('wl').drop_duplicates('wl')
            # 直接返回numpy数组
            return df['wl'].to_numpy(dtype=np.float64), df['val'].to_numpy(dtype=np.float64)
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            return None, None

    def get_crystal_params(self) -> CrystalParameters:
        cry = self._load_txt_params('crystal.txt', CrystalParameters)
        
        # 加载光谱数据
        wl_pi_abs, raw_pi_abs = self._load_spectrum_csv('absorption_p.csv')    # π 偏振吸收截面
        wl_pi_em, raw_pi_em = self._load_spectrum_csv('emission_p.csv')       # π 偏振发射截面
        wl_sig_abs, raw_sig_abs = self._load_spectrum_csv('absorption_s.csv')   # σ 偏振吸收截面
        wl_sig_em, raw_sig_em = self._load_spectrum_csv('emission_s.csv')      # σ 偏振发射截面

        # 构建插值函数
        def create_interp(wl, val):
            if wl is None or val is None:
                return lambda x: np.zeros_like(x)
            return interp1d(wl * 1e-9, val * 1e-24, kind='cubic', bounds_error=False, fill_value=0)

        f_pi_abs = create_interp(wl_pi_abs, raw_pi_abs)
        f_pi_em  = create_interp(wl_pi_em,  raw_pi_em)
        f_sig_abs = create_interp(wl_sig_abs, raw_sig_abs)
        f_sig_em  = create_interp(wl_sig_em,  raw_sig_em)

        cry.wavelength_grid = self.master_wavelengths

        cry.sigma_abs_pi_grid  = f_pi_abs(self.master_wavelengths)
        cry.sigma_em_pi_grid   = f_pi_em(self.master_wavelengths)
        cry.sigma_abs_sig_grid = f_sig_abs(self.master_wavelengths)
        cry.sigma_em_sig_grid  = f_sig_em(self.master_wavelengths)

        return cry
    
    def get_seed_params(self, align_to_peak: bool = False, target_center_nm: float = 1040.0) -> tuple[Pulse, SeedParameters]: 
        seed = self._load_txt_params('seed.txt', SeedParameters)
        wl_raw, int_raw = self._load_spectrum_csv('seed_spectrum.csv')
        
        if wl_raw is not None:
            # 单位换算: nm -> m
            wl_raw_np = np.asarray(wl_raw, dtype=np.float64)  # 转为numpy浮点数组
            wl_m = wl_raw_np * 1e-9  # nm -> m
            
            # --- 自动对齐逻辑 ---
            if align_to_peak:
                current_center = np.average(wl_m, weights=int_raw)
                shift = (target_center_nm * 1e-9) - current_center
                wl_m += shift # 坐标轴平移
                print(f"🔧 [Design] Seed aligned: Shift {shift*1e9:+.2f} nm -> Target {target_center_nm} nm")
            else:
                center_nm = np.average(wl_m, weights=int_raw) * 1e9
                print(f"📊 [Exp] Raw Seed Center: {center_nm:.2f} nm")
            # -------------------

            f_interp = interp1d(wl_m, int_raw, kind='cubic', bounds_error=False, fill_value=0)
            seed.spectrum_grid = np.maximum(f_interp(self.master_wavelengths), 0.0)
        else:
            print("⚠️ 警告: 未找到种子光谱文件，使用默认高斯分布。")

        # 将实验光谱插值到我们严谨的物理网格 (Grid) 上
        # 注意：加上 bounds_error=False, fill_value=0.0，防止网格比实验数据宽导致越界报错
        f_interp = interp1d(wl_m, int_raw, kind='cubic', bounds_error=False, fill_value=0.0)
        intensity_on_grid = f_interp(self.grid.lambda_window)
        
        # 防止插值出现微小的负数引起复数计算错误
        intensity_on_grid = np.clip(intensity_on_grid, 0, None)

        # 4. 实例化 Pulse 对象
        p = Pulse(self.grid)
        
        # 光谱强度 (Intensity) 转换为 复振幅 (Amplitude) -> A = sqrt(Intensity)
        # 初始种子光我们假设其光谱相位为 0 (即它是一个完美的 FTL 脉冲)
        p.A_f = np.sqrt(intensity_on_grid) + 0j
        
        # 5. 能量归一化：强行缩放振幅，使得该 Pulse 的总能量严格等于 seed.txt 中设定的 E_seed
        current_energy = np.sum(np.abs(p.A_f)**2) * (self.grid.df * 2 * np.pi)
        if current_energy > 0:
            scale_factor = np.sqrt(seed.E_seed / current_energy)
            p.A_f = p.A_f * scale_factor
            
        # 6. 同步更新时域信息
        p.to_time_domain()
            
        return p, seed
    
    def get_pump_params(self): return self._load_txt_params('pump.txt', PumpParameters)
    def get_cavity_params(self): return self._load_txt_params('cavity.txt', CavityParameters)
