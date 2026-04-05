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
from scipy.interpolate import interp1d
from dataclasses import fields

from core.grid import Grid
from core.pulse import Pulse
from core.dataclasses_def import *

class ParameterLoader:
    def __init__(self, grid_points=4096, central_wl=1030e-9, max_wl=1150e-9):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.grid = Grid(points=grid_points, central_wl=central_wl, max_wl=max_wl)
        self.master_wl = self.grid.lambda_window  # 统一的主波长网格

    def _get_path(self, folder: str, filename: str) -> str:
        return os.path.join(self.base_dir, folder, filename)

    def _load_txt_params(self, filename: str, dataclass_type):
        """核心解析器：解析 txt 配置文件并映射到 Dataclass"""
        path = self._get_path('parameters', filename)
        if not os.path.exists(path):
             raise FileNotFoundError(f"❌ 关键文件缺失: {path}")

        allowed_fields = {f.name for f in fields(dataclass_type)}
        params_dict = {}

        with open(path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                content = line.split('#')[0].split('//')[0].strip()
                if not content or '=' not in content: continue
                
                key, val = [part.strip() for part in content.split('=', 1)]
                if key in allowed_fields:
                    try:
                        params_dict[key] = float(val)
                    except ValueError:
                        continue 

        return dataclass_type(**params_dict)

    def _load_spectrum_csv(self, filename: str):
        """基础读取：仅负责读取 CSV 并清洗，返回 numpy 数组"""
        path = self._get_path('datafile', filename)
        if not os.path.exists(path): 
            return None, None
        
        try:
            df = pd.read_csv(path, header=None, names=['wl', 'val'], 
                             sep=r'[,\s]+', encoding='utf-8-sig', engine='python')
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            df = df.sort_values('wl').drop_duplicates('wl')
            return df['wl'].to_numpy(), df['val'].to_numpy()
        except Exception as e:
            print(f"❌ 读取 {filename} 失败: {e}")
            return None, None
        
    def _load_and_interp(self, filename: str, x_scale: float = 1.0, y_scale: float = 1.0) -> np.ndarray:
        """
        加载、单位缩放、插值到主网格、剔除负值。
        """
        wl_raw, val_raw = self._load_spectrum_csv(filename)
        if wl_raw is None or val_raw is None or len(wl_raw) == 0:
            return np.zeros_like(self.master_wl)
        
        if len(wl_raw) < 2:
            return np.zeros_like(self.master_wl)
            
        f_interp = interp1d(wl_raw * x_scale, val_raw * y_scale, 
                            kind='cubic', bounds_error=False, fill_value=0.0)
        # 限制下限为0，防止由于三次样条插值产生的微小负值
        return np.clip(f_interp(self.master_wl), 0.0, None)

    # -----------------------------------------------------------------------
    # 业务接口层：干净、清爽、没有冗余的数学运算
    # -----------------------------------------------------------------------

    def get_crystal_params(self) -> CrystalParameters:
        cry = self._load_txt_params('crystal.txt', CrystalParameters)
        cry.wavelength_grid = self.master_wl
        
        # x_scale=1e-9 将波长转 m，y_scale=1e-24 将截面转 m^2
        cry.sigma_abs_pi_grid  = self._load_and_interp('absorption_p.csv', 1e-9, 1e-24)       # π 偏振吸收截面
        cry.sigma_em_pi_grid   = self._load_and_interp('emission_p.csv',   1e-9, 1e-24)       # π 偏振发射截面
        cry.sigma_abs_sig_grid = self._load_and_interp('absorption_s.csv', 1e-9, 1e-24)       # σ 偏振吸收截面
        cry.sigma_em_sig_grid  = self._load_and_interp('emission_s.csv',   1e-9, 1e-24)       # σ 偏振发射截面

        return cry
    
    def get_seed_params(self, align_to_peak: bool = False, target_center_nm: float = 1040.0) -> tuple[Pulse, SeedParameters]: 
        seed = self._load_txt_params('seed.txt', SeedParameters)
        wl_raw, int_raw = self._load_spectrum_csv('seed_spectrum.csv')
        
        # 1. 光谱处理与插值
        intensity_on_grid = np.zeros_like(self.master_wl)
        
        if wl_raw is not None:
            wl_m = wl_raw * 1e-9
            
            #--- 对齐逻辑 待定？？？：是否需要默认对齐到目标波长？我认为应该集成到具体实验操作中，读取数据应该只负责基础数据处理 ---
            if align_to_peak:
                current_center = np.average(wl_m, weights=int_raw)
                shift = (target_center_nm * 1e-9) - current_center
                wl_m += shift
                print(f"🔧 [Design] Seed aligned: Shift {shift*1e9:+.2f} nm -> Target {target_center_nm} nm")
            else:
                center_nm = np.average(wl_m, weights=int_raw) * 1e9
                print(f"📊 [Exp] Raw Seed Center: {center_nm:.2f} nm")

            # 仅执行一次插值
            f_interp = interp1d(wl_m, int_raw, kind='cubic', bounds_error=False, fill_value=0.0)
            intensity_on_grid = np.clip(f_interp(self.master_wl), 0.0, None)
        else:
            print("⚠️ 警告: 未找到种子光谱文件，使用初始化的空/默认分布。")

        seed.spectrum_grid = intensity_on_grid

        # 2. 实例化与能量初始化
        p = Pulse(self.grid)
        p.A_f = np.sqrt(intensity_on_grid) + 0j # 默认完美无啁啾
        
        # 3. 能量严格归一化 (匹配 txt 设定的能量)
        current_energy = np.sum(np.abs(p.A_f)**2) * (self.grid.df * 2 * np.pi)
        if current_energy > 0:
            p.A_f *= np.sqrt(seed.E_seed / current_energy)
            
        p.to_time_domain()
            
        return p, seed
    
    def get_pump_params(self) -> PumpParameters: 
        return self._load_txt_params('pump.txt', PumpParameters)
        
    def get_cavity_params(self) -> CavityParameters: 
        return self._load_txt_params('cavity.txt', CavityParameters)
