# applications/basics 学习说明

`applications/basics` 是 `LaserPulse_v2` 的学习入口。这个文件夹不是正式工程模型，而是用一组循序渐进的小脚本，帮助理解超短脉冲再生放大器仿真中最基础的物理过程。

目前建议把这里看成：

```text
basics = 学习区 / 验证区 / 物理过程拆解区
```

它的目标不是一步到位写出完整再生放大器，而是先把每一个关键物理环节拆开：

```text
受激辐射 → 能量和增益 → 时频域脉冲 → 色散传播 → 单次放大 → 多程放大 → 增益窄化
```

---

## 1. 推荐学习顺序

建议按下面顺序阅读和运行：

```bash
python applications/basics/basics_1.py
python applications/basics/basics_2.py
python applications/basics/basics_2plus.py
python applications/basics/basics_3.py
python applications/basics/basics_4.py
python applications/basics/basics_5.py
python applications/basics/basics_5plus.py
python applications/basics/basics_6.py
```

如果运行前没有输出文件夹，可以手动创建：

```bash
mkdir utils/record
```

当前图片默认保存到：

```text
utils/record/
```

该目录已经被 `.gitignore` 忽略，所以运行生成的图片不会进入 GitHub。

---

## 2. 脚本功能总览

| 脚本 | 当前状态 | 核心问题 | 学习目标 |
|---|---|---|---|
| `basics_1.py` | 已跑通 | 上能级粒子数 `N2` 和光子密度 `phi` 如何耦合？ | 理解受激辐射导致的 `N2` 消耗和光子增长 |
| `basics_2.py` | 已跑通 | 微观光子数如何转成宏观能量？`N2` 如何转成增益？ | 建立 `N2 / phi / Energy / Gain` 的联系 |
| `basics_2plus.py` | 已跑通 | 参数变化如何影响增益饱和？ | 学会做简单参数扫描 |
| `basics_3.py` | 已跑通 | 高斯脉冲在时域和频域分别长什么样？ | 理解脉宽、中心波长、光谱带宽和 FFT |
| `basics_4.py` | 已跑通 | 透明介质色散如何影响脉冲？ | 理解只加相位时，时域展宽但光谱强度基本守恒 |
| `basics_5.py` | 已跑通 | 单次通过增益介质时能量如何增加？ | 理解泵浦储能和单次能量提取 |
| `basics_5plus.py` | 已跑通 | 多次通过晶体后能量是否一直增加？ | 理解能量饱和和最佳提取圈数 |
| `basics_6.py` | 已跑通 | 再生放大过程中光谱为什么会变窄？ | 理解频域增益窄化和光谱演化 |

---

## 3. 每个脚本的物理含义

### 3.1 `basics_1.py`：受激辐射最小模型

这个脚本只保留两个变量：

```text
N2  = 上能级粒子数密度
phi = 光子数密度
```

核心方程是：

```text
dN2/dt  = - sigma * c * phi * N2
dphi/dt =   sigma * c * N2 * phi - phi / tau_c
```

物理图像：

```text
前期：phi 很小，N2 基本不变
中期：phi 快速增长，N2 被快速消耗
后期：N2 被耗尽，phi 因腔内损耗下降
```

这个脚本回答的问题是：

> 为什么光子数会因为受激辐射快速增长？为什么放大过程会消耗上能级粒子数？

---

### 3.2 `basics_2.py`：从微观量到能量和增益

这个脚本把 `basics_1.py` 中的微观变量转换成更接近实验观测的宏观量：

```text
phi → Energy
N2  → Gain
```

关键关系包括：

```text
单光子能量：E_photon = h * c / lambda
单程增益：G = exp(sigma * N2 * L)
```

运行结果应该表现为：

```text
Energy 上升并趋于饱和
Gain 随 N2 消耗而下降
```

这个脚本回答的问题是：

> 光子密度怎样对应到单脉冲能量？上能级粒子数怎样对应到增益？

---

### 3.3 `basics_2plus.py`：参数扫描

这个脚本在 `basics_2.py` 的基础上改变参数，例如：

```text
sigma  = 受激辐射截面
tau_c  = 腔内光子寿命
```

当前主要演示的是改变 `tau_c` 的影响。

典型趋势：

```text
tau_c 越大：腔内光子寿命越长，损耗越小，phi 峰值更高，N2 消耗更快
tau_c 越小：腔内损耗越强，phi 峰值更低，N2 剩余更多
```

这个脚本回答的问题是：

> 改变一个物理参数，放大过程会如何响应？

这是以后做参数扫描和优化的雏形。

---

### 3.4 `basics_3.py`：高斯脉冲的时域和频域

这个脚本构建一个中心波长约 1030 nm、脉宽约 200 fs 的高斯脉冲，并画出：

```text
时域振幅包络 A(t)
真实电场 E(t)
时域光强 I(t)
频域光谱 S(lambda)
```

它是从“能量模型”走向“超短脉冲模型”的分界点。

学习重点：

```text
脉冲不仅有能量，还有时间宽度、中心波长、光谱宽度和相位
时域越短，频域通常越宽
光谱宽度决定了理论上能压缩到多短
```

这个脚本回答的问题是：

> 一个飞秒脉冲在时域和频域到底分别是什么样？

注意：当前从频率谱转到波长谱时主要用于画图展示，后续做严格能量积分时需要加入频率间隔或波长间隔权重。

---

### 3.5 `basics_4.py`：透明介质传播与线性色散

这个脚本让脉冲经过透明介质，例如 UVFS，观察色散带来的时域变化。

核心思想：

```text
频域电场 E(omega)
乘上材料传播相位 exp[-i phi_dispersion(omega)]
再 IFFT 回到时域
```

典型结果：

```text
时域脉冲展宽，峰值下降
光谱强度基本保持不变
```

这是因为当前只加入了线性色散相位，没有加入吸收、非线性或增益。

这个脚本回答的问题是：

> 为什么只改变频域相位，也会导致时域脉冲展宽？

---

### 3.6 `basics_5.py`：单次经过增益介质

这个脚本开始进入真正的放大器过程。它分成两段：

```text
1. 泵浦阶段：建立上能级反转粒子数 N2
2. 放大阶段：种子脉冲单次经过晶体，提取储能
```

它的输出包括：

```text
泵浦后的 N2
反转比例
输入能量
输出能量
单次通过增益
```

这个脚本回答的问题是：

> 泵浦如何变成储能？种子光如何从增益介质中提取能量？

当前模型仍然是空间平均和切片近似，没有加入 ASE、热效应、真实横向空间分布等高级效应。

---

### 3.7 `basics_5plus.py`：多程放大与最佳提取圈数

这个脚本是 `basics_5.py` 的多程版本。它让种子脉冲多次经过增益介质，记录每一圈后的能量。

典型现象：

```text
前期：能量很低，小信号增益积累
中期：能量快速上升
峰值附近：达到最佳提取点
后期：N2 被过度消耗，净增益下降，能量反而下降
```

这个脚本回答的问题是：

> 再生放大器为什么不是圈数越多越好？为什么需要选择最佳输出时刻？

当前模型还没有完整加入：

```text
Pockels cell
TFP
注入和提取开关
腔内真实损耗
ASE
热效应
```

所以它更适合作为“再生放大能量演化 toy model”。

---

### 3.8 `basics_6.py`：频域增益窄化

这个脚本开始从频域角度模拟多程再生放大。

主要做了三件事：

```text
1. 读取真实种子光谱
2. 读取 Yb:CALGO 的 sigma 偏振吸收 / 发射截面
3. 让不同波长根据各自净增益被放大
```

核心增益关系是：

```text
G(lambda) = [sigma_e(lambda) * N2 - sigma_a(lambda) * N1] * L
```

然后在频域中对电场施加：

```text
E_f(lambda) → E_f(lambda) * exp[G(lambda)/2]
```

典型结果：

```text
能量先增长，达到峰值后下降
宽种子光谱逐渐集中到高增益波段
最终出现明显增益窄化
```

这个脚本回答的问题是：

> 为什么再生放大器中能量变高的同时，光谱会变窄？

这是后续研究“百飞秒以下脉宽能否保持”的关键入口。

---

## 4. 当前体检记录

截至当前阶段，`applications/basics` 中所有基础脚本已经可以独立运行。

```text
basics_1.py       成功：N2 与 phi 的受激辐射耦合
basics_2.py       成功：N2 / phi 到 Energy / Gain
basics_2plus.py   成功：tau_c 参数扫描
basics_3.py       成功：高斯脉冲时域 / 频域表示
basics_4.py       成功：透明介质传播与线性色散
basics_5.py       成功：泵浦充能 + 单次经过增益介质
basics_5plus.py   成功：多程放大与最佳提取圈数
basics_6.py       成功：频域多程放大与增益窄化
```

---

## 5. 当前已知问题

这个文件夹目前仍然是学习脚本区，因此存在一些待整理问题：

1. 多数脚本仍是从上到下直接运行，尚未统一封装成 `main()`；
2. 参数大多写死在脚本顶部，还没有统一配置文件；
3. 图片输出仍然使用 `utils/record/`，后续建议统一为 `outputs/figures/basics/`；
4. 部分变量命名仍有历史痕迹，例如 `n_CGA` 与 Yb:CALGO 表述不完全统一；
5. `basics_6.py` 中最佳提取圈数目前还不是完全自动化标注；
6. 频域能量归一化还偏学习模型，后续应引入更严格的 `df` 或 `dλ` 权重；
7. 这些脚本还没有自动化测试，只是人工运行体检通过。

---

## 6. 后续整理路线

建议后续按下面顺序整理，不要一次性大改：

```text
第一步：给每个 basics 脚本加入 main() 入口
第二步：统一输出路径到 outputs/figures/basics/
第三步：把重复的画图逻辑提取到 utils 或 plotting 模块
第四步：把成熟的物理公式迁移到 physics 模块
第五步：用 tests 检查关键物理量是否合理
第六步：把 basics 保留为学习示例，正式模型转入 engineering 层
```

推荐 commit 粒度：

```text
一次只改一个脚本
每次改完运行该脚本
再运行 pytest
确认 git status
最后 commit + push
```

---

## 7. 和正式模型的关系

`applications/basics` 不应该永远承担正式仿真任务。它更像是“物理概念实验室”。

后续成熟逻辑应该逐步迁移：

```text
basics_1 / basics_2      → physics/amplification/rate_equations.py
basics_3                 → core/grid.py + core/pulse.py
basics_4                 → physics/propagation 或 passive dispersion component
basics_5 / basics_5plus  → physics/amplification + engineering/components
basics_6                 → spectral gain model + regenerative amplifier system
```

最终目标是：

```text
basics 负责教学和验证
physics 负责物理公式
engineering 负责系统组装
main/run scripts 负责正式仿真入口
```
