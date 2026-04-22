"""
==============================================================================
文件名称: beam_propagation.py
所属部门: Propagation (光学传播模块)
主要功能: 光束传播模拟工具
代码解读: 
    用于模拟光束在光学系统中的传播，包括透镜、晶体等光学元件。
==============================================================================
"""
# import os
# import sys
# # 自动获取项目根目录并加入搜索路径
# current_file_path = os.path.abspath(__file__)
# project_root = os.path.dirname(os.path.dirname(current_file_path))
# sys.path.append(project_root)

import numpy as np
import matplotlib.pyplot as plt
import ABCD_matrix as ABCD
import Resonator

def build_cavity():
    """
    构建驻波腔物理模型。
    物理结构: M1(全反平镜) -> Air1(90mm) -> Crystal(8mm, n=1.93) * 2  -> Air2(90mm) -> M2(输出凹面镜, R=200mm)
    """
    # 1. 实例化端镜
    M1 = ABCD.SphericalMirror(R=float('inf'), name="Flat_Mirror_1")
    M2 = ABCD.SphericalMirror(R=190.0, name="Concave_Mirror_2")

    # 2. 实例化腔内元件
    air_gap_1 = ABCD.FreeSpace(length=90.0, n=1.0, name="Air_1")
    crystal_1 = ABCD.Crystal(length=10.0, n=1.93, name="Yb_CGA_1") 
    crystal_2 = ABCD.Crystal(length=10.0, n=1.93, name="Yb_CGA_2")
    air_gap_2 = ABCD.FreeSpace(length=90.0, n=1.0, name="Air_2")

    # 3. 组装谐振腔
    cavity = Resonator.LinearResonator(M1, M2, name="Test_Cavity")
    cavity.add_element(air_gap_1)
    cavity.add_element(crystal_1)
    cavity.add_element(crystal_2)
    cavity.add_element(air_gap_2)

    return cavity

def extract_w_from_q(q: complex, wavelength: float) -> float:
    """
    物理工具函数：从高斯光束复参数 q 中提取物理光斑半径 w。
    物理公式: 1/q = 1/R - j * lambda / (pi * w^2)
    """
    # 1. 计算 1/q 的数值，赋值给合法变量名 inv_q
    inv_q = 1.0 / q

    #2. 提取虚部, 对应物理公式中的 (-lambda / (pi * w^2))
    q_imag = np.imag(inv_q)

    # 3. 计算 w
    w = np.sqrt(-wavelength / (np.pi * q_imag))

    return w

def trace_cavity_mode(cavity, lambda_nm: float, dz=1.0):
    """
    追踪并记录光束在谐振腔内的空间演化数据。
    
    参数:
        cavity: LinearResonator 实例 (已定义的谐振腔)
        lambda_nm: 激光波长 (nm)
        dz: 空间切片步长 (mm)，控制曲线平滑度
    """
    wavelength = lambda_nm * 1e-6
    # 1. 确立物理起点
    inv_q_start = cavity.get_eigen_q()
    q_current = 1.0 / inv_q_start

    # 初始化数据记录器
    z_positions = [0.0]
    w_sizes = [extract_w_from_q(q_current, wavelength)]
    absolute_z = 0.0

    # 展开谐振腔, 按照 光路正向元件 -> M2镜反射 -> 光路反向元件 的顺序拼接
    unfolded_elements = cavity.elements + [cavity.M2] + cavity.elements[::-1]
    
    components_info = [] # 用于记录器件在展开图上的坐标
    
    # 2. 核心循环：遍历光路上的所有物理元件
    for element in unfolded_elements:
        # 情景 A: 连续演化 (有物理长度的介质，如 FreeSpace, Crystal)
        if hasattr(element, 'length') and element.length > 0:
            L = element.length
            n = getattr(element, 'n', 1.0)  # 提取折射率

            # 记录连续介质的位置信息
            components_info.append({
                'type': 'medium',
                'name': element.name,
                'start_z': absolute_z,
                'end_z': absolute_z + L,
                'n': n})

            # 执行空间切片
            steps = int(max(1, L / dz))
            q_entry = q_current # 记录进入介质瞬间的 q 值
            for i in range(1, steps + 1):
                local_z = i * (L / steps)
                # 均匀介质中 q 随路径线性增长
                q_local = q_entry + local_z / n
                
                # 记录数据
                z_positions.append(absolute_z + local_z)
                w_sizes.append(extract_w_from_q(q_local, wavelength))

            # 更新全局状态
            absolute_z += L  # 更新绝对 z 坐标
            q_current = q_entry + L / n  # 更新 q_current 到介质末端

        # 情景 B: 离散突变 (无厚度的界面或薄透镜，如 ThinLens, SphericalMirror)
        else:
            # 记录突变界面的位置信息
            components_info.append({
                'type': 'interface',
                'name': element.name,
                'z': absolute_z})

            # 获取该元件的 ABCD 矩阵
            M = element.get_matrix()
            A, B, C, D = M[0,0], M[0,1], M[1,0], M[1,1]

            # 应用 ABCD 定律进行复参数突变更新
            q_current = (A * q_current + B) / (C * q_current + D)

    return np.array(z_positions), np.array(w_sizes), components_info


# ======== 绘制高斯光束包络图 ========
def plot_cavity_mode(z, w, components_info, title="Resonator Mode Profile (Full Round-Trip)"):
    """绘制高斯光束包络图，并叠加物理器件分布"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 1. 绘制器件色块与标记
    for comp in components_info:
        if comp['type'] == 'medium':
            # 区分晶体和空气。晶体用橙色高亮，空气用浅灰色
            color = 'orange' if comp['n'] > 1.1 else 'gray'
            alpha = 0.2 if comp['n'] > 1.1 else 0.05
            
            ax.axvspan(comp['start_z'], comp['end_z'], color=color, alpha=alpha, lw=0)
            
            # 标注器件名称
            mid_z = (comp['start_z'] + comp['end_z']) / 2
            ax.text(mid_z, 0, comp['name'], rotation=90, va='center', ha='center', 
                    fontsize=10, color='black', alpha=0.6)
            
        elif comp['type'] == 'interface':
            # 标注没有厚度的反射镜/透镜
            ax.axvline(comp['z'], color='red', linestyle='--', lw=2)
            ax.text(comp['z'], max(w)*0.85, f" {comp['name']}", va='center', ha='left', 
                    color='red', fontsize=10, fontweight='bold')

    # 手动标注起点和终点 (平面镜 M1)
    ax.axvline(0, color='blue', linestyle='--', lw=2)
    ax.text(0, max(w)*0.85, " Flat_Mirror_1 (Start)", va='center', ha='left', color='blue', fontweight='bold')
    ax.axvline(max(z), color='blue', linestyle='--', lw=2)
    ax.text(max(z), max(w)*0.85, " Flat_Mirror_1 (End)", va='center', ha='right', color='blue', fontweight='bold')

    # 2. 绘制光束包络
    ax.plot(z, w, 'b-', linewidth=2, label="Beam Radius (w)")
    ax.plot(z, -w, 'b-', linewidth=2)
    ax.fill_between(z, w, -w, color='skyblue', alpha=0.3)
    
    # 3. 图像润色
    ax.axhline(0, color='black', linestyle='-.', linewidth=0.8)
    ax.set_xlabel("Unfolded Position z (mm)", fontsize=12)
    ax.set_ylabel("Radius w (mm)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(-max(w)*1.2, max(w)*1.2) # 留出空间显示标签
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    cavity = build_cavity()
    z_data, w_data, comp_info = trace_cavity_mode(cavity, lambda_nm=1030, dz=0.5)
    plot_cavity_mode(z_data, w_data, comp_info)

