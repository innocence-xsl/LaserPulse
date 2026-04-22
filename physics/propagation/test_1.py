"""
==============================================================================
文件名称: test_1.py
所属部门: propagation (传播模块)
主要功能: 测试谐体本征模和稳定性测试
代码解读: 
    1. 谐振腔本征模测试
    2. 谐振腔稳定性测试

==============================================================================
"""

import numpy as np
from ABCD_matrix import Crystal, FreeSpace, ThinLens, SphericalMirror, SphericalDielectric
from Resonator import LinearResonator


def build_test_cavity():
    """
    构建一个典型的驻波腔物理模型。
    物理结构: M1(全反平镜) -> Air1(90mm) -> Crystal(8mm, n=1.93) -> Air2(90mm) -> M2(输出凹面镜, R=200mm)
    """
    # 1. 实例化端镜
    M1 = SphericalMirror(R=float('inf'), name="Flat_Mirror_1")
    M2 = SphericalMirror(R=300.0, name="Concave_Mirror_2")

    # 2. 实例化腔内元件
    air_gap_1 = FreeSpace(length=90.0, n=1.0, name="Air_1")
    crystal = Crystal(length=8.0, n=1.93, name="Yb_CGA") 
    air_gap_2 = FreeSpace(length=90.0, n=1.0, name="Air_2")

    # 3. 组装谐振腔
    cavity = LinearResonator(M1, M2, name="Test_Cavity")
    cavity.add_element(air_gap_1)
    cavity.add_element(crystal)
    cavity.add_element(air_gap_2)

    return cavity

def test_cavity_matrices():
    """
    测试谐振腔的单程与往返矩阵计算。
    """
    cavity = build_test_cavity()
    print(f"--- 测试腔体: {cavity.name} ---")

    # 1. 测试单程矩阵 (Forward)
    # 物理意义: 计算光束从 M1 飞到 M2 (不含端镜反射) 所经历的衍射和相位演化。
    M_forward = cavity.get_one_way_matrix(reverse_direction=False)
    # M_backward = cavity.get_one_way_matrix(reverse_direction=True)
    print("正向单程矩阵 M_forward:\n", M_forward)

    # 2. 测试往返矩阵 (Round-trip)
    # 物理意义: 计算从 M1 出发，完成一个完整往返的总 ABCD 矩阵。
    M_rt = cavity.get_round_trip_matrix()
    print("往返矩阵 M_rt:\n", M_rt)

    # 3. 稳定性检验 (物理检验)
    # 物理规律: 稳定腔的充要条件是 -1 <= (A + D) / 2 <= 1
    trace_val = np.trace(M_rt)
    if -1 <= trace_val / 2 <= 1:
        print("腔体稳定")
    else:
        print(f"腔体不稳定，(A + D) / 2 = {trace_val / 2:.4f}")

def test_eigenmode():
    # 1. 构建稳定腔
    cavity = build_test_cavity()
    # 定义波长 (Yb:CGA 常用的 1030 nm = 1.03e-3 mm)
    lambda_nm = 1030
    wavelength = lambda_nm * 1e-6

    # 2. 计算 M1 处的本征 1/q
    try:
        inv_q = cavity.get_eigen_q()
        # 3. 提取物理参数
        # 1/q = 1/R - j * lambda / (pi * w^2)
        # 所以 w = sqrt(-lambda / (pi * imag(1/q)))
        w_m1 = np.sqrt(-wavelength / (np.pi * np.imag(inv_q)))

        print(f"--- 腔模解算 (波长: {lambda_nm} nm) ---")
        print(f"M1 处的本征 1/q: {inv_q}")
        print(f"M1 处的光斑半径 (Radius): {w_m1:.4f} mm")
        print(f"M1 处的光斑直径 (Diameter): {2*w_m1:.4f} mm")

    except ValueError as e:
        print(e)

# 腔体本征模测试
if __name__ == "__main__":
    test_eigenmode()

# 腔体稳定性测试
# if __name__ == "__main__":
#     test_cavity_matrices()
