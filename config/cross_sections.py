"""
==============================================================================
文件名称: cross_sections.py
所属部门: Config / Data Utility
主要功能: 吸收和发射截面查看脚本
代码解读:
    读取 data/Yb_CALGO 中的吸收/发射截面 CSV，
    并画出 π / σ 偏振下的截面曲线。

注意:
    这个文件主要用于检查数据，不作为正式模型的必需入口。
==============================================================================
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"


def load_spectrum_csv(filename):
    """读取 data/ 下的 CSV 文件并清洗，返回波长和截面数组。"""
    file_path = DATA_DIR / filename
    try:
        df = pd.read_csv(
            file_path,
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


def smooth_by_interp(x, y, point_num=1000):
    """线性插值生成高密度曲线，保留原始数据趋势。"""
    interp_func = interp1d(x, y, kind="linear", bounds_error=False, fill_value=0.0)
    x_new = np.linspace(x.min(), x.max(), point_num)
    y_new = interp_func(x_new)
    return x_new, np.clip(y_new, 0.0, None)


def main():
    files = {
        "pi_abs": "Yb_CALGO/pai_abs.csv",
        "pi_emi": "Yb_CALGO/pai_emi.csv",
        "sigma_abs": "Yb_CALGO/sigma_abs.csv",
        "sigma_emi": "Yb_CALGO/sigma_emi.csv",
    }

    titles = [
        r"$\pi$-Polarization Absorption",
        r"$\pi$-Polarization Emission",
        r"$\sigma$-Polarization Absorption",
        r"$\sigma$-Polarization Emission",
    ]
    keys = ["pi_abs", "pi_emi", "sigma_abs", "sigma_emi"]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        key = keys[i]
        filename = files[key]
        wl_raw, val_raw = load_spectrum_csv(filename)

        if wl_raw is not None and len(wl_raw) >= 2:
            wl_smooth, val_smooth = smooth_by_interp(wl_raw, val_raw, point_num=1000)
            ax.plot(wl_smooth, val_smooth, color=colors[i], linewidth=2, label="Interpolated")
        else:
            ax.text(0.5, 0.5, "Data Load Failed", ha="center", va="center", transform=ax.transAxes)

        ax.set_title(titles[i], fontsize=14, pad=10)
        ax.set_xlabel("Wavelength (nm)", fontsize=12)
        ax.set_ylabel(r"Cross-section ($10^{-20}$ cm$^2$)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
