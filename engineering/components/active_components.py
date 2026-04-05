"""
==============================================================================
文件名称: active_components.py
所属部门: Engineering (工程部)
主要功能: 有源增益介质 ( Yb:CALGO 晶体放大器)
代码解读: 
    定义有源增益介质组件，核心是 BulkCrystal 类。
    该类实现了脉冲在增益介质中的传播接口，以及泵浦充能接口。
    通过调用 physics.amplification.component_interface 中的 propagate_pulse 和 trigger_pump 函数，
    实现了增益介质内的物理过程模拟（包括放大、基态耗尽等）。
    
==============================================================================
"""

import numpy as np
from engineering.components.base_component import BaseComponent
from core.pulse import Pulse
from core.dataclasses_def import CrystalParameters, PumpParameters, PhysicalConstants
from physics.amplification.component_interface import propagate_pulse, trigger_pump

class BulkCrystal(BaseComponent):
    
    def __init__(self, name, crystal_params, seed_params, pump_params, consts, pump_polarization='pi', signal_polarization='sigma'):
        super().__init__(name)
        self.cry = crystal_params
        self.seed = seed_params
        self.pump = pump_params      
        self.consts = consts
        self.pump_pol = pump_polarization
        self.signal_polarization = signal_polarization.lower()

        # 记录当前的上能级粒子数密度
        self.N_upper = 0.0
        self.wavelengths = self.cry.wavelength_grid
        
        # 信号光的偏振接口
        if self.signal_polarization == 'pi':
            self.sigma_abs = self.cry.sigma_abs_pi_grid
            self.sigma_em  = self.cry.sigma_em_pi_grid
        elif self.signal_polarization == 'sigma':
            self.sigma_abs = self.cry.sigma_abs_sig_grid
            self.sigma_em  = self.cry.sigma_em_sig_grid
        else:
            raise ValueError(f"未知的偏振态: {self.signal_polarization}，只能选 'pi' 或 'sigma'")
        
        self.L_total = self.cry.thickness * self.cry.num_disks
        self.V_crystal = self.pump.pump_area * self.L_total

    def propagate(self, pulse: Pulse) -> Pulse:
        """流水线Pulse传播接口"""
        pulse, self.N_upper = propagate_pulse(
            pulse=pulse,
            N_upper=self.N_upper,
            cry_params=self.cry,
            seed_params=self.seed,
            consts=self.consts,
            sigma_em=self.sigma_em,
            sigma_abs=self.sigma_abs
        )
        return pulse
    
    def pump_crystal(self, dt: float):
        """流水线统一标准的充能踏板"""
        self.N_upper = trigger_pump(
            dt=dt,
            N_upper_start=self.N_upper,
            cry_params=self.cry,
            pump_params=self.pump,
            consts=self.consts,
            pump_polarization=self.pump_pol,
            pump_area=self.pump.pump_area,
            L_total=self.L_total
        )
