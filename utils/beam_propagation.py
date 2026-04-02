"""
==============================================================================
文件名称: beam_propagation.py
所属部门: Utils (工具模块)
主要功能: 光束传播模拟工具
代码解读: 
    用于模拟光束在光学系统中的传播，包括透镜、晶体等光学元件。
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------- 1. 定义初始高斯光束参数 ----------------------
lambda0 = 1040e-9    # 波长，单位m（你的1040nm附近）
w0 = 800e-6          # 初始束腰半径，单位m（对应图里z=0处的800μm）
z0 = 0               # 初始束腰位置，单位m

# 高斯光束的瑞利长度和初始q参数
zR = np.pi * w0**2 / lambda0
q0 = z0 + 1j * zR

# ---------------------- 2. 定义你的光路元件（按顺序） ----------------------
# 格式：(元件到下一个元件的自由传播距离L, 元件的ABCD矩阵)
optical_elements = [
    # 第一段：传播到第一个透镜
    (0.3, np.array([[1, 0], [0, 1]])),  # 自由传播300mm
    # 第一个透镜（焦距f=-300mm，凹透镜）
    (0, np.array([[1, 0], [-1/0.3, 1]])),
    # 传播到第一个晶体位置
    (0.25, np.array([[1, 0], [0, 1]])),
    # 晶体+热透镜（等效为焦距300mm的透镜，和图里一致）
    (0, np.array([[1, 0], [-1/0.3, 1]])),
    # 继续按你的光路加元件...
]

# ---------------------- 3. 计算每个位置的光束半径 ----------------------
z_list = [0]       # 传播距离
w_list = [w0*1e6]  # 光束半径（单位μm）
q = q0

for L, M in optical_elements:
    # 1. 自由传播L距离的矩阵
    M_free = np.array([[1, L], [0, 1]])
    q = (M_free[0,0]*q + M_free[0,1]) / (M_free[1,0]*q + M_free[1,1])
    z_list.append(z_list[-1] + L*1000)  # 转成mm
    w_list.append(np.sqrt(-lambda0/(np.pi * np.imag(1/q))) * 1e6)  # 转成μm
    
    # 2. 经过光学元件的矩阵
    q = (M[0,0]*q + M[0,1]) / (M[1,0]*q + M[1,1])
    w_list.append(np.sqrt(-lambda0/(np.pi * np.imag(1/q))) * 1e6)
    z_list.append(z_list[-1])  # 元件位置z不变

# ---------------------- 4. 绘图 ----------------------
plt.figure(figsize=(10,6))
plt.plot(z_list, w_list, color='blue', linewidth=2)

# 标注元件位置（和你的图一样，用虚线标Cry/DM）
plt.axvline(x=600, color='green', linestyle='--', label='Cry')
plt.axvline(x=700, color='red', linestyle='--', label='DM')
plt.axvline(x=800, color='green', linestyle='--')

plt.xlabel('Z (mm)', fontsize=16)
plt.ylabel('beam radius (μm)', fontsize=16)
plt.ylim(100, 900)
plt.legend()
plt.grid(alpha=0.3)
plt.show()