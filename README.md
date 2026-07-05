# LaserPulse_v2

LaserPulse_v2 是一个用于学习和搭建 **超短脉冲再生放大器数值仿真模型** 的 Python 项目。

这个分支不是旧版 `main` 分支的简单整理版，而是当前主要开发分支。它的目标是从最基础的受激辐射、速率方程、脉冲时频域表示开始，逐步发展到 Yb:CALGO 再生放大器中的泵浦储能、增益提取、增益窄化、光谱整形、腔内往返循环和传播矩阵计算。

当前项目仍处于早期研究和学习原型阶段。它更像是一个“从零搭建再生放大器仿真框架”的工作台，而不是已经完成的论文级软件。

---

## 1. 当前分支定位

`LaserPulse_v2` 目前有两条并行主线：

1. **学习主线：`applications/basics`**  
   用一组从 `basics_1.py` 到 `basics_6.py` 的脚本，逐步理解放大器仿真的基础物理过程。

2. **工程主线：`core` + `physics` + `engineering`**  
   把前面学到的物理过程逐步模块化，形成可以复用、可以测试、可以扩展的仿真框架。

简单说：

```text
applications/basics  = 学习区 / 草稿区 / 物理过程拆解
core                 = 核心数据结构
physics              = 物理公式与模型
engineering          = 工程化组装和参数读取
tests                = 自动测试与安全网
```

---

## 2. 项目结构

当前 `LaserPulse_v2` 分支的大致结构如下：

```text
LaserPulse_v2/
├── applications/
│   └── basics/
│       ├── basics_1.py        # 受激辐射放大的最小速率方程：N2 与 phi 耦合
│       ├── basics_2.py        # 从 N2 / phi 过渡到宏观能量与增益
│       ├── basics_2plus.py    # 改变截面、光子寿命等参数，观察增益饱和趋势
│       ├── basics_3.py        # 构建高斯脉冲，观察时域和频域表示
│       ├── basics_4.py        # 自由空间与透明介质传播，观察色散展宽
│       ├── basics_5.py        # 单次经过增益介质：泵浦储能与能量提取
│       ├── basics_5plus.py    # 多次经过增益介质：再生放大能量增长雏形
│       ├── basics_6.py        # 频域角度模拟增益窄化和光谱演化
│       └── basics.ipynb       # Notebook 草稿或学习记录
│
├── core/
│   ├── grid.py                # 时间 / 频率 / 波长网格
│   ├── pulse.py               # Pulse 对象，保存时域 A_t 与频域 A_f
│   └── dataclasses_def.py     # 物理常数、晶体、泵浦、种子、腔参数定义
│
├── physics/
│   ├── optics_tools.py        # 光学计算辅助函数
│   ├── base_physical_op.py    # 基础物理操作接口或草稿
│   ├── amplification/
│   │   ├── component_interface.py
│   │   ├── frantz_nodvik.py
│   │   ├── gain.py
│   │   ├── pump_dynamics.py
│   │   └── rate_equations.py
│   └── propagation/
│       ├── ABCD_matrix.py     # ABCD 矩阵与基本光学元件
│       ├── Resonator.py       # 线性谐振腔与本征模计算
│       ├── beam_propagation.py
│       └── test_1.py          # 谐振腔稳定性和本征模测试脚本
│
├── engineering/
│   ├── assembly/
│   │   └── optical_assembly.py    # 再生放大器装配体和循环控制
│   ├── components/
│   │   ├── base_component.py      # 光学组件基类
│   │   ├── active_components.py   # 有源增益介质组件
│   │   └── passive_components.py  # 损耗镜、滤波器、色散器、光栅压缩器等
│   └── config/
│       ├── parameter_loader.py    # 参数与光谱数据读取
│       ├── parameters/            # crystal / pump / seed / cavity 参数文件
│       └── datafile/
│           ├── Yb_CALGO/          # Yb:CALGO π / σ 偏振截面数据
│           ├── seed_spectrum.csv  # 种子光谱
│           └── UVFS.csv           # 透明材料色散数据
│
├── tests/
│   └── test_energy_consistency.py # Grid / Pulse 时频转换基础测试
│
├── .gitignore
├── .gitattributes
├── requirements.txt
└── README.md
```

---

## 3. 快速上手

### 3.1 克隆项目并切换分支

```bash
git clone https://github.com/innocence-xsl/LaserPulse.git
cd LaserPulse
git checkout LaserPulse_v2
```

如果本地还没有该分支：

```bash
git fetch origin
git checkout -b LaserPulse_v2 origin/LaserPulse_v2
```

### 3.2 创建独立 Python 环境

当前分支已在 Python 3.13 环境下通过基础测试。推荐使用独立 conda 环境，避免污染 `base`。

```bash
conda create -n laserpulse313 python=3.13
conda activate laserpulse313
pip install -r requirements.txt
```

### 3.3 运行测试

```bash
pytest
```

当前基础测试目标是确认 `Grid` / `Pulse` 的时频转换、能量保持和 FTL 脉冲生成没有被后续重构破坏。

---

## 4. 从哪里开始看？

如果你是第一次看这个项目，建议不要直接读 `engineering` 或 `physics/amplification` 里的复杂模块，而是按下面顺序从 `applications/basics` 入手。

### Step 1：受激辐射和速率方程

```bash
python applications/basics/basics_1.py
```

`basics_1.py` 用两个变量建立最基础的受激辐射放大模型：

- `N2`：上能级粒子数密度；
- `phi`：光子数密度。

它暂时忽略自发辐射，只观察受激辐射中 `N2` 消耗和光子数增长的耦合过程。

### Step 2：从微观量走向宏观能量和增益

```bash
python applications/basics/basics_2.py
```

`basics_2.py` 开始把微观光子数密度和上能级粒子数密度转化为：

- 单脉冲能量；
- 单程增益；
- 储能量与输出能量之间的关系。

### Step 3：参数扫描

```bash
python applications/basics/basics_2plus.py
```

`basics_2plus.py` 用来改变受激辐射截面、腔内光子寿命等参数，观察它们对增益饱和趋势的影响。

### Step 4：高斯脉冲的时域和频域

```bash
python applications/basics/basics_3.py
```

`basics_3.py` 构建 1030 nm、200 fs 的高斯脉冲，并通过 FFT 观察：

- 时域光强；
- 真实电场；
- 频域光谱；
- 脉宽和光谱带宽之间的关系。

### Step 5：自由空间和介质传播

```bash
python applications/basics/basics_4.py
```

`basics_4.py` 用自由空间传播和透明介质传播作为例子，观察光束半径、光强分布、频谱和时域波形的变化，重点理解线性色散导致的展宽。

### Step 6：单次经过增益介质

```bash
python applications/basics/basics_5.py
```

`basics_5.py` 把过程分成两段：

1. 泵浦充能，建立反转粒子数；
2. 种子脉冲单次经过增益介质，提取能量。

这是从“纯速率方程”走向“放大器模型”的第一步。

### Step 7：多程 / 再生放大雏形

```bash
python applications/basics/basics_5plus.py
```

`basics_5plus.py` 引入多次往返，让种子光反复经过增益介质，并记录能量随圈数增长的过程。

### Step 8：频域增益窄化

```bash
python applications/basics/basics_6.py
```

`basics_6.py` 从频域出发，用 Yb:CALGO 的 σ 偏振吸收 / 发射截面和种子光谱，观察多圈放大后的光谱演化和增益窄化趋势。

---

## 5. 核心模块说明

### 5.1 `core`

`core` 是项目的地基。

- `grid.py`：定义统一的时间、频率、角频率、波长网格；
- `pulse.py`：定义 `Pulse` 对象，同时保存 `A_t` 和 `A_f`；
- `dataclasses_def.py`：用 dataclass 管理物理常数、晶体、泵浦、种子和腔参数。

后续所有复杂模型都应该尽量基于 `Grid` 和 `Pulse` 构建，而不是在每个脚本里重复造轮子。

### 5.2 `physics`

`physics` 存放相对独立的物理公式和模型。

目前包括两大块：

1. `physics/amplification`：放大相关模型，例如速率方程、泵浦动力学、Frantz-Nodvik、增益计算等；
2. `physics/propagation`：传播相关模型，例如 ABCD 矩阵、谐振腔稳定性、本征模和光束传播。

### 5.3 `engineering`

`engineering` 是把物理模块组装成“器件”和“系统”的地方。

- `engineering/components`：把增益晶体、损耗镜、滤波器、色散器等封装成具有 `propagate(pulse)` 接口的组件；
- `engineering/assembly`：定义再生放大器装配体，控制脉冲在腔内多圈循环；
- `engineering/config`：读取参数文件和光谱数据，并插值到统一网格。

当前这一层还在发展中，后续会逐步成为正式仿真主线。

---

## 6. 数据文件说明

当前新版 Yb:CALGO 数据位于：

```text
engineering/config/datafile/Yb_CALGO/
├── pai_abs.csv
├── pai_emi.csv
├── sigma_abs.csv
└── sigma_emi.csv
```

其中：

- `pai_abs.csv`：π 偏振吸收截面；
- `pai_emi.csv`：π 偏振发射截面；
- `sigma_abs.csv`：σ 偏振吸收截面；
- `sigma_emi.csv`：σ 偏振发射截面。

另外：

- `seed_spectrum.csv`：实验或自定义种子光谱；
- `UVFS.csv`：透明材料色散数据。

注意：当前部分脚本直接读取 `Yb_CALGO` 文件夹，而 `ParameterLoader` 中仍有一些命名和路径逻辑需要后续统一。这是 v2 后续整理的重要任务之一。

---

## 7. 当前测试

目前测试文件包括：

```text
tests/test_energy_consistency.py
physics/propagation/test_1.py
```

测试目标包括：

- `Pulse` 在 `A_t -> A_f -> A_t` 转换后能量保持；
- FTL 脉冲生成不破坏原始脉冲；
- ABCD 矩阵和谐振腔本征模计算的基础逻辑可以运行。

运行方式：

```bash
pytest
```

---

## 8. 当前已知问题与后续整理方向

这个分支目前仍是学习型和工程化过渡型代码，存在一些需要逐步整理的地方：

1. `applications/basics` 中部分脚本仍有大量硬编码参数；
2. 部分脚本会把结果保存到 `utils/record`，后续建议统一改到 `outputs/figures`；
3. `Yb_CALGO` 数据目录和 `ParameterLoader` 的晶体名称参数需要统一；
4. `basics` 脚本还没有统一的函数封装和命令行入口；
5. `physics/amplification` 与 `engineering/components` 的接口需要继续梳理；
6. 目前模型主要是空间平均和简化增益模型，还没有完整加入 ASE、热透镜、B 积分、Pockels cell 动态开关等高级效应。

建议后续按以下顺序推进：

```text
第一阶段：整理 applications/basics，让每个学习脚本用途清楚、能独立运行
第二阶段：统一数据路径和参数文件命名
第三阶段：把 basics 中成熟的物理过程迁移到 physics 模块
第四阶段：把 physics 模块封装成 engineering/components
第五阶段：形成正式的再生放大器仿真入口脚本
第六阶段：增加 ASE、热效应、B 积分和 CPA 压缩等高级模型
```

---

## 9. Git 分支说明

当前建议：

```text
main            = 旧版本归档 / 稳定入口 / 最终整合分支
LaserPulse_v2   = 当前重点开发分支
```

现在请优先在 `LaserPulse_v2` 中开发、测试和整理。等 v2 的结构稳定后，再考虑合并或重建 `main`。

---

## 10. 给自己的开发原则

为了避免项目再次变乱，建议遵循：

```text
一次只改一个小目标
先跑 pytest
再跑相关 basics 脚本
确认 git status 干净
再 commit 和 push
```

不要一次同时改参数、物理模型、画图和目录结构。科研代码最重要的是：每一步都能解释，每一次结果变化都能追踪。
