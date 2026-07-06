"""
==============================================================================
文件名称: cross_sections.py
所属部门: Engineering (工程部)
主要功能: 吸收和发射截面加载器
代码解读: 
    画出晶体的吸收截面和发射截面数据图
==============================================================================
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# 自动获取当前脚本所在目录，解决路径报错问题
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_spectrum_csv(filename):
    """读取 CSV 文件并清洗，返回波长和截面数组"""
    file_path = os.path.join(BASE_DIR, filename)
    try:
        df = pd.read_csv(file_path, header=None, names=['wl', 'val'], 
                         sep=r'[,\s]+', encoding='utf-8-sig', engine='python')
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        df = df.sort_values('wl').drop_duplicates('wl')
        return df['wl'].to_numpy(), df['val'].to_numpy()
    except Exception as e:
        print(f"❌ 读取 {filename} 失败: {e}")
        return None, None

def smooth_by_interp(x, y, point_num=1000):
    """线性插值生成高密度平滑曲线，保留原始数据趋势"""
    interp_func = interp1d(x, y, kind="linear", bounds_error=False, fill_value=0.0)
    
    # 生成的 x_new 严格限制在原始数据的 min 和 max 之间，因此实际上不会触发越界填充
    x_new = np.linspace(x.min(), x.max(), point_num)
    y_new = interp_func(x_new)
    
    # 截面积不可能为负数，加一层 clip 保护
    return x_new, np.clip(y_new, 0.0, None)

# 对应四种光谱的文件名、直接修改晶体名称crystal_name
files = {
    'pi_abs': 'datafile/Yb_CALGO/pai_abs.csv',
    'pi_emi': 'datafile/Yb_CALGO/pai_emi.csv',
    'sigma_abs': 'datafile/Yb_CALGO/sigma_abs.csv',
    'sigma_emi': 'datafile/Yb_CALGO/sigma_emi.csv'
}

# ========================
# 开始加载数据并绘制 2x2 子图
# ========================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

# 定义图表标题
titles = [
    r'$\pi$-Polarization Absorption', 
    r'$\pi$-Polarization Emission',
    r'$\sigma$-Polarization Absorption', 
    r'$\sigma$-Polarization Emission'
]
keys = ['pi_abs', 'pi_emi', 'sigma_abs', 'sigma_emi']
colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']

for i, ax in enumerate(axes):
    key = keys[i]
    filename = files[key]
    
    # 1. 读取原始数据
    wl_raw, val_raw = load_spectrum_csv(filename)
    
    if wl_raw is not None and len(wl_raw) >= 2:
        # 2. 使用引入的线性插值进行平滑
        wl_smooth, val_smooth = smooth_by_interp(wl_raw, val_raw, point_num=1000)
        
        # 3. 绘制平滑后的曲线
        ax.plot(wl_smooth, val_smooth, color=colors[i], linewidth=2, label='Interpolated')
        
        # （可选）把原始数据点用半透明的散点画出来，方便你对比插值是否准确
        # ax.scatter(wl_raw, val_raw, color='black', s=10, alpha=0.3, label='Raw Data')
        # ax.legend()
    else:
        ax.text(0.5, 0.5, 'Data Load Failed', ha='center', va='center', transform=ax.transAxes)

    # 4. 图表排版设置
    ax.set_title(titles[i], fontsize=14, pad=10)
    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Cross-section ($10^{-20}$ cm$^2$)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(bottom=0) # Y轴强制从0开始，截面无负值

plt.tight_layout()
plt.show()