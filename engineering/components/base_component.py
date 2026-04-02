"""
==============================================================================
文件名称: base_component.py
所属部门: Engineering (工程部)
主要功能: 光学器件的抽象基类
代码解读: 
    这是所有光学器件的抽象基类（蓝图）,
    每个具体的器件（如增益介质、反射镜、透镜等）都将继承这个基类，
    并实现 propagate 方法，定义脉冲穿过该器件时发生的物理变化。

==============================================================================
"""

from core.pulse import Pulse

class BaseComponent:
    """
    所有光学器件的抽象基类 (蓝图)
    """
    def __init__(self, name="Unnamed Component"):
        self.name = name

    def propagate(self, pulse: Pulse) -> Pulse:
        """
        核心物理加工方法。
        脉冲传入该器件后，发生物理变化（损耗、色散、放大等）。
        必须在子类中被重写！
        """
        raise NotImplementedError(f"[{self.name}] 必须在子类中实现 propagate 方法！")