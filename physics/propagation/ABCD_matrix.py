"""
==============================================================================
文件名称: ABCD_matrix.py
所属部门: propagation (传播模块)
主要功能: 计算ABCD矩阵的乘积
代码解读: 
    计算ABCD矩阵的乘积，返回新的ABCD矩阵
==============================================================================
"""
import numpy as np
from dataclasses import dataclass

class OpticalElement:
    """
    光学元件基类。
    使用 dataclass 自动管理参数初始化。
    """
    def __init__(self, name: str = "BaseElement"):
        self.name = name
    
    def get_matrix(self) -> np.ndarray:
        """
        返回该元件的 2x2 ABCD 矩阵。
        必须在子类中重写此方法。
        """
        raise NotImplementedError("子类必须实现 get_matrix 方法")

@dataclass
class FreeSpace(OpticalElement):
    """
    自由空间传播
    物理参数: 
        length: 传播距离 L
        n: 折射率，默认为 1.0 (空气/真空)
    物理公式: A=1, B=L/n, C=0, D=1
    """
    length: float = 0.0
    n: float = 1.0
    name: str = "FreeSpace"

    def get_matrix(self) -> np.ndarray:
        return np.array([
            [1.0, self.length / self.n],
            [0.0, 1.0]
        ])
    
@dataclass
class ThinLens(OpticalElement):
    """
    薄透镜
    物理参数:
        f: 焦距 (会聚透镜为正)
    物理公式: A=1, B=0, C=-1/f, D=1
    """
    f: float = float('inf') # 默认为无穷大（即没有光焦度）
    name: str = "ThinLens"

    def get_matrix(self) -> np.ndarray:
        # 使用 numpy 构建并返回薄透镜的 2x2 矩阵
        return np.array([
            [1.0, 0.0],
            [-1.0 / self.f, 1.0]
        ])
    
@dataclass
class SphericalMirror(OpticalElement):
    """
    球面镜
    物理参数:
        R: 曲率半径 (会聚镜为正)
    物理公式: A=1, B=0, C=-2/R, D=1
    """
    R: float = float('inf') # 默认为无穷大（即平面镜）
    name: str = "SphericalMirror"

    def get_matrix(self) -> np.ndarray:
        return np.array([
            [1.0, 0.0],
            [-2.0 / self.R, 1.0]
        ])
    
@dataclass
class SphericalDielectric(OpticalElement):
    """
    球面介质界面
    物理参数:
        R: 曲率半径 (界面向右凸为正)
        n1: 入射侧折射率
        n2: 出射侧折射率
    物理公式: A=1, B=0, C=(n2-n1)/(R*n2), D=n1/n2
    """
    R: float = float('inf') # 默认为无穷大（即平面界面）
    n1: float = 1.0
    n2: float = 1.0
    name: str = "Sphericaldielectric"

    def get_matrix(self) -> np.ndarray:
        return np.array([
            [1.0, 0.0],
            [(self.n2 - self.n1) / (self.R * self.n2), self.n1 / self.n2]
        ])

@dataclass
class Crystal(OpticalElement):
    """
    激光晶体元件 (当前为无热透镜的理想平行平板模型)
    物理参数:
        length: 晶体物理长度 L
        n: 晶体折射率
    物理公式: 等效于长度为 L/n 的自由空间传播
    """
    length: float = 0.0
    n: float = 1.0
    name: str = "LaserCrystal"

    def get_matrix(self) -> np.ndarray:
        return np.array([
            [1.0, self.length / self.n],
            [0.0, 1.0]
        ])

    
    
if __name__ == "__main__":
    # --- 测试脚本 ---
    
    # 1. 实例化自由空间 (假设传播距离 L = 100 mm)
    space = FreeSpace(length=100.0, name="Air_Gap_1")
    
    # 2. 实例化薄透镜 (假设焦距 f = 50 mm)
    lens = ThinLens(f=50.0, name="Focusing_Lens")

    # 3. 实例化球面镜 (假设曲率半径 R = 100 mm)
    mirror = SphericalMirror(R=100.0, name="Spherical_Mirror")

    # 4. 实例化球面介质界面 (假设曲率半径 R = 50 mm, n1=1.0, n2=1.5)
    dielectric = SphericalDielectric(R=50.0, n1=1.0, n2=1.5, name="Spherical_Dielectric")
    
    # 5. 打印元件信息和它们的 ABCD 矩阵
    print(f"--- {space.name} ---")
    print("元件参数:", space) # dataclass 会自动生成好看的字符串表示
    print("ABCD 矩阵:\n", space.get_matrix())
    print()
    
    print(f"--- {lens.name} ---")
    print("元件参数:", lens)
    print("ABCD 矩阵:\n", lens.get_matrix())

    print(f"--- {mirror.name} ---")
    print("元件参数:", mirror)
    print("ABCD 矩阵:\n", mirror.get_matrix())

    print(f"--- {dielectric.name} ---")
    print("元件参数:", dielectric)
    print("ABCD 矩阵:\n", dielectric.get_matrix())

