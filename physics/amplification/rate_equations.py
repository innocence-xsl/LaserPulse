"""
==============================================================================
文件名称: rate_equations.py
所属部门: physics/amplification (物理部/放大模型)
主要功能: 激光速率方程基础理论
代码解读: 
    定义连续/脉冲泵浦下的再生放大器速率方程理论模型。
    提取自测试脚本，包含：
    1. 基于解析解的增益与能量演化模型。
    2. 基于 ODE (常微分方程) 数值积分的演化模型（考虑动态损耗和荧光寿命）。
==============================================================================
"""
import numpy as np

def analytical_rate_step(g0, Esat, epsilons, tao, Tr):
    """
    基于解析解的速率方程迭代单步计算
    参数:
        g0: 初始小信号增益
        Esat: 饱和能量 (J)
        epsilons: 种子光初始能量 (J)
        tao: 当前经过的时间 (s)
        Tr: 腔内往返时间 (s)
    返回:
        gain: 当前时刻的增益
        energy: 当前时刻的脉冲能量
    """
    # 计算当前增益
    gain = (g0 + epsilons / Esat) / (1 + epsilons / Esat / g0 * np.exp(g0 * tao / Tr))
    
    # 计算当前能量
    energy = Esat * (g0 + epsilons / Esat) / (1 + g0 / (epsilons / Esat) * np.exp(-g0 * tao / Tr))
    
    return gain, energy

def rate_ode_system(t, y, tL, Esat, Tr, loss, g0_pump):
    """
    基于常微分方程(ODE)的速率方程系统
    参数:
        t: 时间
        y: 状态数组，y[0]为增益，y[1]为脉冲能量
        tL: 荧光寿命 (s)
        Esat: 饱和能量 (J)
        Tr: 腔内往返时间 (s)
        loss: 腔内往返总损耗
        g0_pump: 连续泵浦对应的稳态初始增益
    返回:
        dy: 状态的导数 [d(Gain)/dt, d(Energy)/dt]
    """
    dy = np.zeros(2)
    # dy[0]: 增益的时间导数 = 泵浦补充 - 荧光自发辐射衰减 - 受激辐射提取
    dy[0] = (g0_pump - y[0]) / tL - y[0] * y[1] / (Esat * Tr)
    
    # dy[1]: 能量的时间导数 = 增益放大 - 腔内损耗
    dy[1] = y[1] / Tr * (y[0] - loss)
    
    return dy