"""
==============================================================================
文件名称: base_physical_op.py
所属部门: Physics (物理部)
主要功能: 定义物理算子基类（黑盒接口）
代码解读: 
    定义所有物理过程必须实现的接口，强制标准化输入输出，屏蔽底层差异，
    确保所有物理算子（如增益、色散、非线性效应）都遵循统一的调用规范。

==============================================================================
"""

from abc import ABC, abstractmethod
from core.dataclasses_def import PulseParams, PhysicalConstants, OpResult

class BasePhysicalOp(ABC):
    """物理算子基类（黑盒接口）"""
    def __init__(self, constants: PhysicalConstants):
        # 仅接收物理常量（如普朗克常数、增益系数），不接收任何工程参数
        self.constants = constants

    @abstractmethod
    def validate_input(self, pulse_params: PulseParams) -> bool:
        """校验输入参数合法性（黑盒前置检查）"""
        pass

    @abstractmethod
    def compute(self, pulse_params: PulseParams) -> OpResult:
        """核心计算：输入脉冲参数→输出变换后的脉冲参数（纯物理公式）"""
        pass

    def __call__(self, pulse_params: PulseParams) -> OpResult:
        """快捷调用：校验+计算，对外暴露的唯一执行入口"""
        if not self.validate_input(pulse_params):
            raise ValueError(f"Invalid input for {self.__class__.__name__}")
        return self.compute(pulse_params)