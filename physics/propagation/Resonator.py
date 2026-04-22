"""
==============================================================================
文件名称: Resonator.py
所属部门: propagation (传播模块)
主要功能: 谐振腔
代码解读: 
    1. 定义线性谐振腔系统，包含端镜和腔内元件。
    
==============================================================================
"""
import numpy as np

import ABCD_matrix as ABCD
    
class LinearResonator:
    """
    线性驻波谐振腔系统
    """
    def __init__(self, M1, M2, name: str = "LinearCavity"):
        self.name = name
        # 明确指定腔的两个端镜
        self.M1 = M1  
        self.M2 = M2
        # elements 列表中只存放 M1 和 M2 之间的元件，严格按照从 M1 到 M2 的物理空间顺序排列
        self.elements = []

    def add_element(self, element):
        """按照从 M1 向 M2 的方向，依次放置元件"""
        self.elements.append(element)
    
    def get_one_way_matrix(self, reverse_direction: bool = False) -> np.ndarray:
        """计算单程传输矩阵"""
        M_total = np.identity(2)
        # 1. 确定物理上的遍历顺序
        if not reverse_direction:
            # 正向：从 M1 飞向 M2。物理顺序就是列表的原顺序
            physical_order = self.elements
        else:
            # 反向：从 M2 飞向 M1。物理上最先遇到的是列表末尾的元件
            # [::-1] 是 Python 中倒序切片的方法，生成一个倒序的新列表
            physical_order = self.elements[::-1]
        
        # 2. 执行数学上的矩阵左乘计算
        for element in physical_order:
            # 不论物理正向还是反向，新碰到的元件矩阵永远乘在 M_total 的左边！
            M_total = element.get_matrix() @ M_total
        return M_total
    
    def get_round_trip_matrix(self) -> np.ndarray:
        """
        计算从 M1 出发，完成一个完整往返的总 ABCD 矩阵。
        """
        # 1. 获取光线单程飞行的矩阵
        M_forward = self.get_one_way_matrix(reverse_direction=False)
        M_backward = self.get_one_way_matrix(reverse_direction=True)

        # 2. 获取端镜的反射矩阵
        M_mirror2 = self.M2.get_matrix()
        M_mirror1 = self.M1.get_matrix()

        # 3. 计算完整的往返矩阵, 按照物理发生顺序的逆序，进行连乘 (左乘法则)
        M_rt = M_mirror1 @ M_backward @ M_mirror2 @ M_forward
        return M_rt

    def get_eigen_q(self) -> complex:
        """
        根据自洽条件，计算端镜 M1 处的本征高斯光束复参数 q 的倒数 (1/q)。
        物理原理: q = (A*q + B) / (C*q + D)
        """
        # 1. 获取往返矩阵
        M = self.get_round_trip_matrix()
        A, B, C, D = M[0,0], M[0,1], M[1,0], M[1,1]

        # 2. 稳定性检查 (物理判定)
        trace_half = (A + D) / 2.0
        if abs(trace_half) >= 1.0:
            raise ValueError(f"腔体不稳定 (g1g2={trace_half:.4f})，无法形成本征模！")
        
        # 3. 求解本征 1/q
        imag_part = -np.sqrt(4.0 - (A + D)**2) / (2.0 * B)
        real_part = (D - A) / (2.0 * B)
        eigen_q = real_part + 1j * imag_part
        
        return eigen_q