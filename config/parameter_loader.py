"""
==============================================================================
文件名称: parameter_loader.py
所属部门: Config / IO
主要功能: 参数加载器
代码解读:
    负责从 config/parameters 读取 txt 参数文件，
    从项目根目录 data/ 读取光谱、截面和材料 CSV 数据，
    并将实验数据插值到统一的 Grid 波长网格上。
==============================================================================
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from dataclasses import fields

# config/parameter_loader.py -> 项目根目录 LaserPulse_v2
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.grid import Grid
from core.pulse import Pulse
from core.dataclasses_def import *


class ParameterLoader:
    def __init__(self, grid_points=4096, central_wl=1030e-9, max_wl=1150e-9):
        self.project_root = PROJECT_ROOT
        self.config_dir = self.project_root / "config"
        self.param_dir = self.config_dir / "parameters"
        self.data_dir = self.project_root / "data"

        self.grid = Grid(points=grid_points, central_wl=central_wl, max_wl=max_wl)
        self.master_wl = self.grid.lambda_window  # 统一的主波长网格

    # -----------------------------------------------------------------------
    # 路径工具
    # -----------------------------------------------------------------------
    def _parameter_path(self, filename: str) -> Path:
        """返回参数文件路径，优先使用 config/parameters，兼容 data/parameters。"""
        candidates = [
            self.param_dir / filename,
            self.data_dir / "parameters" / filename,
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(f"❌ 关键参数文件缺失: {candidates[0]}")

    def _data_path(self, filename: str) -> Path:
        """返回 data/ 下的数据文件路径。filename 可以包含子目录。"""
        return self.data_dir / filename

    # -----------------------------------------------------------------------
    # 基础读取器
    # -----------------------------------------------------------------------
    def _load_txt_params(self, filename: str, dataclass_type):
        """解析 txt 配置文件并映射到 Dataclass。"""
        path = self._parameter_path(filename)
        allowed_fields = {f.name for f in fields(dataclass_type)}
        params_dict = {}

        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                content = line.split("#")[0].split("//")[0].strip()
                if not content or "=" not in content:
                    continue

                key, val = [part.strip() for part in content.split("=", 1)]

                # 兼容 xxx_nm = ... 这种写法：如果 dataclass 里有 xxx，就自动从 nm 转 m
                scale = 1.0
                target_key = key
                if key not in allowed_fields and key.endswith("_nm"):
                    maybe_key = key[:-3]
                    if maybe_key in allowed_fields:
                        target_key = maybe_key
                        scale = 1e-9

                if target_key in allowed_fields:
                    try:
                        params_dict[target_key] = float(val) * scale
                    except ValueError:
                        continue

        return dataclass_type(**params_dict)

    def _load_spectrum_csv(self, filename: str):
        """读取 data/ 下的 CSV 文件并清洗，返回 numpy 数组。"""
        path = self._data_path(filename)
        if not path.exists():
            print(f"⚠️ 未找到数据文件: {path}")
            return None, None

        try:
            df = pd.read_csv(
                path,
                header=None,
                names=["wl", "val"],
                sep=r"[,\s]+",
                encoding="utf-8-sig",
                engine="python",
            )
            df = df.apply(pd.to_numeric, errors="coerce").dropna()
            df = df.sort_values("wl").drop_duplicates("wl")
            return df["wl"].to_numpy(), df["val"].to_numpy()
        except Exception as e:
            print(f"❌ 读取 {filename} 失败: {e}")
            return None, None

    def _load_and_interp(self, filename: str, x_scale: float = 1.0, y_scale: float = 1.0) -> np.ndarray:
        """加载、单位缩放、插值到主网格，并剔除负值。"""
        wl_raw, val_raw = self._load_spectrum_csv(filename)
        if wl_raw is None or val_raw is None or len(wl_raw) < 2:
            return np.zeros_like(self.master_wl)

        f_interp = interp1d(
            wl_raw * x_scale,
            val_raw * y_scale,
            kind="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        return np.clip(f_interp(self.master_wl), 0.0, None)

    @staticmethod
    def _crystal_data_dir(crystal_name: str) -> str:
        """把用户友好的晶体名映射到 data/ 下的实际文件夹名。"""
        mapping = {
            "calgo": "Yb_CALGO",
            "yb_calgo": "Yb_CALGO",
            "yb:calgo": "Yb_CALGO",
            "Yb_CALGO": "Yb_CALGO",
        }
        return mapping.get(crystal_name, crystal_name)

    # -----------------------------------------------------------------------
    # 业务接口层
    # -----------------------------------------------------------------------
    def get_crystal_params(self, crystal_name: str = "calgo") -> CrystalParameters:
        """加载指定晶体的静态参数和截面数据。"""
        # 目前静态晶体参数统一放在 crystal.txt；后续多晶体时再扩展成 crystal_calgo.txt。
        cry = self._load_txt_params("crystal.txt", CrystalParameters)
        cry.wavelength_grid = self.master_wl

        data_subdir = self._crystal_data_dir(crystal_name)
        cry.sigma_abs_pi_grid = self._load_and_interp(f"{data_subdir}/pai_abs.csv", 1e-9, 1e-24)
        cry.sigma_em_pi_grid = self._load_and_interp(f"{data_subdir}/pai_emi.csv", 1e-9, 1e-24)
        cry.sigma_abs_sig_grid = self._load_and_interp(f"{data_subdir}/sigma_abs.csv", 1e-9, 1e-24)
        cry.sigma_em_sig_grid = self._load_and_interp(f"{data_subdir}/sigma_emi.csv", 1e-9, 1e-24)

        print(f"💎 [ParameterLoader] Successfully loaded crystal parameters for: {crystal_name.upper()}")
        return cry

    def get_seed_params(self, align_to_peak: bool = False, target_center_nm: float = 1040.0) -> tuple[Pulse, SeedParameters]:
        seed = self._load_txt_params("seed.txt", SeedParameters)
        wl_raw, int_raw = self._load_spectrum_csv("seed_spectrum.csv")

        intensity_on_grid = np.zeros_like(self.master_wl)

        if wl_raw is not None and int_raw is not None and len(wl_raw) >= 2:
            wl_m = wl_raw * 1e-9

            if align_to_peak:
                current_center = np.average(wl_m, weights=int_raw)
                shift = (target_center_nm * 1e-9) - current_center
                wl_m += shift
                print(f"🔧 [Design] Seed aligned: Shift {shift*1e9:+.2f} nm -> Target {target_center_nm} nm")
            else:
                center_nm = np.average(wl_m, weights=int_raw) * 1e9
                print(f"📊 [Exp] Raw Seed Center: {center_nm:.2f} nm")

            f_interp = interp1d(wl_m, int_raw, kind="linear", bounds_error=False, fill_value=0.0)
            intensity_on_grid = np.clip(f_interp(self.master_wl), 0.0, None)
        else:
            print("⚠️ 警告: 未找到种子光谱文件，使用初始化的空/默认分布。")

        seed.spectrum_grid = intensity_on_grid

        p = Pulse(self.grid)
        p.A_f = np.sqrt(intensity_on_grid) + 0j

        # 能量归一化：保持当前 Pulse/Grid 的既有能量约定。
        current_energy = np.sum(np.abs(p.A_f) ** 2) * (self.grid.df * 2 * np.pi)
        if current_energy > 0:
            p.A_f *= np.sqrt(seed.E_seed / current_energy)

        p.to_time_domain()
        return p, seed

    def get_pump_params(self) -> PumpParameters:
        return self._load_txt_params("pump.txt", PumpParameters)

    def get_cavity_params(self) -> CavityParameters:
        # 目前配置文件名为 cavity_calgo.txt；兼容未来改名为 cavity.txt。
        try:
            return self._load_txt_params("cavity_calgo.txt", CavityParameters)
        except FileNotFoundError:
            return self._load_txt_params("cavity.txt", CavityParameters)
