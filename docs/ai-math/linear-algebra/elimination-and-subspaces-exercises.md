# 习题与思考：方程与子空间

A 档巩固基础，B 档培养 ML 直觉。每题附完整标准答案。

---

## 二、方程与子空间（§6–§11）

### A 档

**A1.** 解方程组 $A\mathbf{x}=\mathbf{b}$，其中 $A=\begin{bmatrix}1&2\\2&4\end{bmatrix}$，$\mathbf{b}=(3,6)^{\mathsf T}$。

(1) 求通解；
(2) 求零空间 $N(A)$ 的一组基础解系。

> **标准答案：**
> 观察：\(A\) 的第二行是第一行的 2 倍 → \(\text{rank}(A)=1\)。方程组为：
>
> $$\begin{cases} x_1 + 2x_2 = 3 \\ 2x_1 + 4x_2 = 6 \end{cases}$$
>
> 第二方程是第一方程的 2 倍，不提供新信息。只有一个独立方程：\(x_1 + 2x_2 = 3\)。
>
> **(1)** 特解：令 \(x_2 = 0\)，则 \(x_1 = 3\) → \(\mathbf{x}_p = (3, 0)^{\mathsf T}\)。也可取 \(x_1 = 0\)，\(x_2 = 1.5\) → 另一个特解 \((0, 1.5)^{\mathsf T}\)。
>
> 零空间（齐次方程 \(x_1 + 2x_2 = 0\)）：基础解系 \(\mathbf{v} = (-2, 1)^{\mathsf T}\)（令 \(x_2 = 1\)，\(x_1 = -2\)）。
>
> 通解：\(\mathbf{x} = \mathbf{x}_p + c \cdot \mathbf{v} = \begin{bmatrix}3\\0\end{bmatrix} + c\begin{bmatrix}-2\\1\end{bmatrix}\)，\(c \in \mathbb{R}\)。
>
> **(2)** 零空间 \(N(A)\) 的基础解系：\(\{(-2, 1)^{\mathsf T}\}\)，维数 \(= n - r = 2 - 1 = 1\)。几何上，解集是 \(\mathbb{R}^2\) 中平行于 \((-2,1)\) 方向的一条直线。

---

**A2.** 用 QR 分解求 $A\mathbf{x}=\mathbf{b}$ 的最小二乘解，其中

$$A=\begin{bmatrix}1&1\\1&2\\1&3\end{bmatrix},\quad \mathbf{b}=\begin{bmatrix}1\\2\\2\end{bmatrix}.$$

> **标准答案：**
>
> **步骤 1：Gram-Schmidt 正交化。**
> \(\mathbf{a}_1 = (1, 1, 1)^{\mathsf T}\)，\(\|\mathbf{a}_1\| = \sqrt{3}\)。
>
> $$\mathbf{q}_1 = \frac{1}{\sqrt{3}}(1, 1, 1)^{\mathsf T}, \quad r_{11} = \sqrt{3}$$
>
> \(\mathbf{a}_2 = (1, 2, 3)^{\mathsf T}\)。减去在 \(\mathbf{q}_1\) 上的投影：
>
> $$\mathbf{a}_2 \cdot \mathbf{q}_1 = \frac{1+2+3}{\sqrt{3}} = \frac{6}{\sqrt{3}} = 2\sqrt{3}$$
>
> $$\mathbf{v}_2 = \mathbf{a}_2 - (\mathbf{a}_2 \cdot \mathbf{q}_1)\mathbf{q}_1 = \begin{bmatrix}1\\2\\3\end{bmatrix} - 2\sqrt{3} \cdot \frac{1}{\sqrt{3}}\begin{bmatrix}1\\1\\1\end{bmatrix} = \begin{bmatrix}1\\2\\3\end{bmatrix} - \begin{bmatrix}2\\2\\2\end{bmatrix} = \begin{bmatrix}-1\\0\\1\end{bmatrix}$$
>
> 注意 \(\mathbf{v}_2\) 与 \(\mathbf{q}_1\) 正交（\((-1,0,1) \cdot (1,1,1) = 0\) ✓）。
>
> $$\|\mathbf{v}_2\| = \sqrt{1+0+1} = \sqrt{2}, \quad \mathbf{q}_2 = \frac{1}{\sqrt{2}}(-1, 0, 1)^{\mathsf T}, \quad r_{22} = \sqrt{2}$$
>
> \(r_{12} = \mathbf{a}_2 \cdot \mathbf{q}_1 = 2\sqrt{3}\)。
>
> **结果**：
>
> $$Q = \begin{bmatrix} 1/\sqrt{3} & -1/\sqrt{2} \\ 1/\sqrt{3} & 0 \\ 1/\sqrt{3} & 1/\sqrt{2} \end{bmatrix}, \quad R = \begin{bmatrix} \sqrt{3} & 2\sqrt{3} \\ 0 & \sqrt{2} \end{bmatrix}$$
>
> 验证 \(QR = A\)：
> 第一列：\(\sqrt{3} \cdot \mathbf{q}_1 = (1,1,1)^{\mathsf T}\) ✓。第二列：\(2\sqrt{3}\mathbf{q}_1 + \sqrt{2}\mathbf{q}_2 = (2,2,2) + (-1,0,1) = (1,2,3)^{\mathsf T}\) ✓。
>
> **步骤 2：解 \(R\mathbf{x} = Q^{\mathsf T}\mathbf{b}\)。**
>
> $$Q^{\mathsf T}\mathbf{b} = \begin{bmatrix} 1/\sqrt{3} & 1/\sqrt{3} & 1/\sqrt{3} \\ -1/\sqrt{2} & 0 & 1/\sqrt{2} \end{bmatrix} \begin{bmatrix}1\\2\\2\end{bmatrix}$$
>
> $$= \begin{bmatrix}(1+2+2)/\sqrt{3} \\ (-1+0+2)/\sqrt{2}\end{bmatrix} = \begin{bmatrix}5/\sqrt{3} \\ 1/\sqrt{2}\end{bmatrix}$$
>
> 解上三角方程：
>
> $$\begin{bmatrix}\sqrt{3} & 2\sqrt{3} \\ 0 & \sqrt{2}\end{bmatrix} \begin{bmatrix}x_1\\x_2\end{bmatrix} = \begin{bmatrix}5/\sqrt{3} \\ 1/\sqrt{2}\end{bmatrix}$$
>
> \(\sqrt{2} x_2 = 1/\sqrt{2} \Rightarrow x_2 = 1/2\)。
> \(\sqrt{3}x_1 + 2\sqrt{3} \cdot \frac{1}{2} = 5/\sqrt{3} \Rightarrow \sqrt{3}x_1 = 5/\sqrt{3} - \sqrt{3} = \frac{5-3}{\sqrt{3}} = 2/\sqrt{3} \Rightarrow x_1 = 2/3\)。
> 最小二乘解：\(\mathbf{x} = (2/3,\; 1/2)^{\mathsf T}\)。
>
> **注**：QR 解法避免了正规方程中的条件数平方问题。\(A^{\mathsf T}A\) 的直接求解需要 \(\kappa(A)^2\) 的精度，QR 只需要 \(\kappa(A)\)。

---

**A3.** 给定协方差矩阵 $\Sigma = \begin{bmatrix}4&2\\2&3\end{bmatrix}$。

(1) 验证 $\Sigma$ 是正定的；
(2) 求其 Cholesky 分解 $\Sigma = LL^{\mathsf T}$；
(3) 用 $L$ 生成 $3$ 个来自 $\mathcal{N}(\mathbf{0},\Sigma)$ 的随机样本。

> **标准答案：**
>
> **(1)** 顺序主子式法：
>
> $$\Delta_1 = 4 > 0$$
>
> $$\Delta_2 = \det(\Sigma) = 4 \times 3 - 2 \times 2 = 12 - 4 = 8 > 0$$
>
> 所有顺序主子式 \(>0\) → \(\Sigma\) 正定 ✓。也可验证特征值：\(\det(\Sigma - \lambda I) = (4-\lambda)(3-\lambda) - 4 = \lambda^2 - 7\lambda + 8 = 0\)，\(\lambda = \frac{7 \pm \sqrt{49-32}}{2} = \frac{7 \pm \sqrt{17}}{2} \approx 5.56, 1.44\)，全部 \(>0\)。
>
> **(2)** Cholesky 分解递推公式：
>
> $$L_{11} = \sqrt{\Sigma_{11}} = \sqrt{4} = 2$$
>
> $$L_{21} = \frac{\Sigma_{21}}{L_{11}} = \frac{2}{2} = 1$$
>
> $$L_{22} = \sqrt{\Sigma_{22} - L_{21}^2} = \sqrt{3 - 1} = \sqrt{2}$$
>
> $$L = \begin{bmatrix}2 & 0 \\ 1 & \sqrt{2}\end{bmatrix}$$
>
> 验证：\(LL^{\mathsf T} = \begin{bmatrix}2&0\\1&\sqrt{2}\end{bmatrix}\begin{bmatrix}2&1\\0&\sqrt{2}\end{bmatrix} = \begin{bmatrix}4&2\\2&3\end{bmatrix} = \Sigma\) ✓。
>
> **(3)** 采样公式：\(\mathbf{x} = L\mathbf{z}\)，\(\mathbf{z} \sim \mathcal{N}(\mathbf{0}, I_2)\)。
> 验证：\(\mathbb{E}[\mathbf{x}\mathbf{x}^{\mathsf T}] = \mathbb{E}[L\mathbf{z}\mathbf{z}^{\mathsf T}L^{\mathsf T}] = L \cdot I \cdot L^{\mathsf T} = \Sigma\)。
> Python 代码：
> ```python
> import numpy as np
> L = np.array([[2, 0], [1, np.sqrt(2)]])
> z = np.random.randn(2, 3)  # 2维, 3个样本
> x = L @ z
> ```
> 这三个样本的样本协方差矩阵应近似 \(\Sigma\)（样本量小时有随机波动）。

---

**A4.** 对矩阵 $A = \begin{bmatrix}2&1&0\\4&3&1\\-2&2&3\end{bmatrix}$，求其 LU 分解 $A = LU$。然后用 LU 解 $A\mathbf{x} = (1,3,7)^{\mathsf T}$。

> **标准答案：**
>
> **消元步骤：**
> ① 消去 \((2,1)\)：乘数 \(\ell_{21} = 4/2 = 2\)。\(R_2 \leftarrow R_2 - 2R_1\)：
>
> $$U_1 = \begin{bmatrix}2&1&0\\0&1&1\\-2&2&3\end{bmatrix}$$
>
> 消去 \((3,1)\)：乘数 \(\ell_{31} = -2/2 = -1\)。\(R_3 \leftarrow R_3 - (-1)R_1\)（即 \(R_3 + R_1\)）：
>
> $$U_2 = \begin{bmatrix}2&1&0\\0&1&1\\0&3&3\end{bmatrix}$$
>
> ② 消去 \((3,2)\)：乘数 \(\ell_{32} = 3/1 = 3\)。\(R_3 \leftarrow R_3 - 3R_2\)：
>
> $$U = \begin{bmatrix}2&1&0\\0&1&1\\0&0&0\end{bmatrix}$$
>
> \(L\) 由乘数构成（对角线上全为 \(1\)）：
>
> $$L = \begin{bmatrix}1&0&0\\\ell_{21}&1&0\\\ell_{31}&\ell_{32}&1\end{bmatrix} = \begin{bmatrix}1&0&0\\2&1&0\\-1&3&1\end{bmatrix}$$
>
> 验证：\(LU = \begin{bmatrix}1&0&0\\2&1&0\\-1&3&1\end{bmatrix}\begin{bmatrix}2&1&0\\0&1&1\\0&0&0\end{bmatrix} = \begin{bmatrix}2&1&0\\4&3&1\\-2&2&3\end{bmatrix} = A\) ✓。
> 注意 \(U\) 有零行 → \(A\) 秩亏（秩 \(=2\)）。
>
> **用 LU 解方程**：
> ① 前向代入 \(L\mathbf{c} = \mathbf{b}\)：
>
> $$\begin{bmatrix}1&0&0\\2&1&0\\-1&3&1\end{bmatrix}\begin{bmatrix}c_1\\c_2\\c_3\end{bmatrix} = \begin{bmatrix}1\\3\\7\end{bmatrix}$$
>
> $$c_1 = 1$$
>
> $$2c_1 + c_2 = 3 \Rightarrow c_2 = 3 - 2 = 1$$
>
> $$-c_1 + 3c_2 + c_3 = 7 \Rightarrow -1 + 3 + c_3 = 7 \Rightarrow c_3 = 5$$
>
> ② 回代 \(U\mathbf{x} = \mathbf{c}\)：
>
> $$\begin{bmatrix}2&1&0\\0&1&1\\0&0&0\end{bmatrix}\begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix} = \begin{bmatrix}1\\1\\5\end{bmatrix}$$
>
> 第三行：\(0 \cdot x_3 = 5\) → **矛盾！** 方程组无解。\(\mathbf{b} = (1,3,7)^{\mathsf T}\) 不在 \(A\) 的列空间里。
> 验证列空间：\(A\) 的列空间由前两列张成，维数 \(=2\)。\(\mathbf{b}\) 是否在其中？检查 \(\mathbf{b}\) 减去前两列的线性组合后第三分量：\((1,3,7) - c_1(2,4,-2) - c_2(1,3,2)\)。第三分量 \(= 7 + 2c_1 - 2c_2\)。当 \(c_1 = 0, c_2 = -0.5\) 时，第三分量 \(= 7 + 0 + 1 = 8 \neq 0\)。\(\mathbf{b}\) 不在 \(xy\) 平面内 → 不在列空间中。

---

**A5.** 设 $A = \begin{bmatrix}1&2&0&0\\0&0&1&0\\0&0&0&0\end{bmatrix}$，$\mathbf{b} = (1,2,0)^{\mathsf T}$。

(1) 写出 $A$ 的四个基本子空间，每子空间各给一组基和维数。
(2) $\mathbf{b}$ 在列空间里吗？有解吗？若可解，求通解。

> **标准答案：**
> \(A\) 已是行阶梯形，秩 \(r=2\)（主元在第 1 列和第 3 列）。
>
> **(1)** 四个基本子空间：
> - **列空间** \(C(A) \subset \mathbb{R}^3\)：由 \(A\) 的主元列（第 1 和第 3 列）张成。基：
>
> $$\left\{\begin{bmatrix}1\\0\\0\end{bmatrix},\; \begin{bmatrix}0\\1\\0\end{bmatrix}\right\}$$
>
> 维数 \(=2\)。这是 \(\mathbb{R}^3\) 中的 \(xy\) 平面。
> - **零空间** \(N(A) \subset \mathbb{R}^4\)：解 \(A\mathbf{x} = \mathbf{0}\)：
>
> $$\begin{cases} x_1 + 2x_2 = 0 \\ x_3 = 0 \end{cases}$$
>
> 自由变量：\(x_2, x_4\)。令 \(x_2=1, x_4=0\)：\(\mathbf{n}_1 = (-2, 1, 0, 0)^{\mathsf T}\)。令 \(x_2=0, x_4=1\)：\(\mathbf{n}_2 = (0, 0, 0, 1)^{\mathsf T}\)。基础解系：\(\{\mathbf{n}_1, \mathbf{n}_2\}\)。维数 \(=4-2=2\)。
> - **行空间** \(C(A^{\mathsf T}) \subset \mathbb{R}^4\)：\(A\) 的非零行构成基：
>
> $$\{(1, 2, 0, 0),\; (0, 0, 1, 0)\}$$
>
> 维数 \(=2\)。注意行空间与零空间正交（内积验证：\((1,2,0,0) \cdot (-2,1,0,0) = 0\) ✓）。
> - **左零空间** \(N(A^{\mathsf T}) \subset \mathbb{R}^3\)：解 \(A^{\mathsf T}\mathbf{y} = \mathbf{0}\)：
>
> $$A^{\mathsf T} = \begin{bmatrix}1&0&0\\2&0&0\\0&1&0\\0&0&0\end{bmatrix},\quad A^{\mathsf T}\mathbf{y} = \begin{bmatrix}y_1\\2y_1\\y_2\\0\end{bmatrix} = \mathbf{0}$$
>
> \(y_1=0\)，\(y_2=0\)，\(y_3\) 自由。基：\(\{(0, 0, 1)^{\mathsf T}\}\)。维数 \(=3-2=1\)。几何上，左零空间是 \(z\) 轴——垂直于 \(xy\) 平面（列空间）。
>
> **(2)** \(\mathbf{b} = (1, 2, 0)^{\mathsf T}\) 的第三分量是 \(0\) → 在 \(xy\) 平面（列空间）内 → 有解。
> 特解：主元列变量 \(x_1, x_3\)，自由变量 \(x_2=x_4=0\)。由方程组：
>
> $$x_1 = 1,\quad x_3 = 2$$
>
> \(\mathbf{x}_p = (1, 0, 2, 0)^{\mathsf T}\)。
> 通解：\(\mathbf{x} = \mathbf{x}_p + c_1\mathbf{n}_1 + c_2\mathbf{n}_2 = (1, 0, 2, 0)^{\mathsf T} + c_1(-2, 1, 0, 0)^{\mathsf T} + c_2(0, 0, 0, 1)^{\mathsf T}\)。

---

**A6.** 设 $\mathbf{a} = (1,2,2)^{\mathsf T}$。求投影到 $\mathbf{a}$ 所在直线上的投影矩阵 $P$，并验证 $P^2 = P$ 和 $P^{\mathsf T} = P$。

> **标准答案：**
> 一维投影公式：\(P = \frac{\mathbf{a}\mathbf{a}^{\mathsf T}}{\mathbf{a}^{\mathsf T}\mathbf{a}}\)。
>
> $$\mathbf{a}^{\mathsf T}\mathbf{a} = 1^2 + 2^2 + 2^2 = 1 + 4 + 4 = 9$$
>
> $$\mathbf{a}\mathbf{a}^{\mathsf T} = \begin{bmatrix}1\\2\\2\end{bmatrix}\begin{bmatrix}1&2&2\end{bmatrix} = \begin{bmatrix}1&2&2\\2&4&4\\2&4&4\end{bmatrix}$$
>
> $$P = \frac{1}{9}\begin{bmatrix}1&2&2\\2&4&4\\2&4&4\end{bmatrix}$$
>
> **验证 \(P^2 = P\)**：
>
> $$P^2 = \frac{1}{81}\begin{bmatrix}1&2&2\\2&4&4\\2&4&4\end{bmatrix}\begin{bmatrix}1&2&2\\2&4&4\\2&4&4\end{bmatrix}$$
>
> 算第一行第一列：\(\frac{1}{81}(1\cdot1 + 2\cdot2 + 2\cdot2) = \frac{1+4+4}{81} = \frac{9}{81} = \frac{1}{9} = P_{11}\)。所有元素类似验证 → \(P^2 = P\)。
>
> **验证 \(P^{\mathsf T} = P\)**：矩阵显然对称 ✓。
>
> **数值验证**：\(\mathbf{b} = (3, 0, 0)^{\mathsf T}\)。投影：
>
> $$P\mathbf{b} = \frac{1}{9}\begin{bmatrix}1&2&2\\2&4&4\\2&4&4\end{bmatrix}\begin{bmatrix}3\\0\\0\end{bmatrix} = \frac{1}{9}\begin{bmatrix}3\\6\\6\end{bmatrix} = \begin{bmatrix}1/3\\2/3\\2/3\end{bmatrix}$$
>
> 残差 \(\mathbf{e} = \mathbf{b} - P\mathbf{b} = (8/3, -2/3, -2/3)^{\mathsf T}\)。验证正交性：
>
> $$\mathbf{e} \cdot \mathbf{a} = \frac{8}{3} \cdot 1 + \left(-\frac{2}{3}\right) \cdot 2 + \left(-\frac{2}{3}\right) \cdot 2 = \frac{8-4-4}{3} = 0$$
>
> 残差 \(\perp\) 投影方向 ✓。

---

**A7.** 对下列向量组执行 Gram-Schmidt 正交化：

$$\mathbf{a}_1 = (1, 1, 0)^{\mathsf T}, \quad \mathbf{a}_2 = (0, 1, 1)^{\mathsf T}, \quad \mathbf{a}_3 = (0, 0, 1)^{\mathsf T}.$$

> **标准答案：**
>
> **(1)**
>
> $$\mathbf{q}_1 = \frac{\mathbf{a}_1}{\|\mathbf{a}_1\|} = \frac{1}{\sqrt{2}}(1, 1, 0)^{\mathsf T}$$
>
> $$\mathbf{v}_2 = \mathbf{a}_2 - (\mathbf{a}_2 \cdot \mathbf{q}_1)\mathbf{q}_1$$
>
> $$\mathbf{a}_2 \cdot \mathbf{q}_1 = 0 \cdot \frac{1}{\sqrt{2}} + 1 \cdot \frac{1}{\sqrt{2}} + 1 \cdot 0 = \frac{1}{\sqrt{2}}$$
>
> $$\mathbf{v}_2 = (0,1,1) - \frac{1}{\sqrt{2}} \cdot \frac{1}{\sqrt{2}}(1,1,0) = (0,1,1) - (\tfrac{1}{2},\tfrac{1}{2},0) = (-\tfrac{1}{2}, \tfrac{1}{2}, 1)^{\mathsf T}$$
>
> $$\|\mathbf{v}_2\| = \sqrt{\tfrac{1}{4} + \tfrac{1}{4} + 1} = \sqrt{\tfrac{3}{2}} = \frac{\sqrt{3}}{\sqrt{2}}$$
>
> $$\mathbf{q}_2 = \frac{\sqrt{2}}{\sqrt{3}}(-\tfrac{1}{2}, \tfrac{1}{2}, 1)^{\mathsf T} = \left(-\frac{1}{\sqrt{6}}, \frac{1}{\sqrt{6}}, \frac{\sqrt{2}}{\sqrt{3}}\right)^{\mathsf T}$$
>
> $$\mathbf{v}_3 = \mathbf{a}_3 - (\mathbf{a}_3 \cdot \mathbf{q}_1)\mathbf{q}_1 - (\mathbf{a}_3 \cdot \mathbf{q}_2)\mathbf{q}_2$$
>
> $$\mathbf{a}_3 \cdot \mathbf{q}_1 = 0$$
>
> $$\mathbf{a}_3 \cdot \mathbf{q}_2 = 0 \cdot (-\tfrac{1}{\sqrt{6}}) + 0 \cdot \tfrac{1}{\sqrt{6}} + 1 \cdot \tfrac{\sqrt{2}}{\sqrt{3}} = \frac{\sqrt{2}}{\sqrt{3}}$$
>
> $$\mathbf{v}_3 = (0,0,1) - 0 - \frac{\sqrt{2}}{\sqrt{3}}\left(-\frac{1}{\sqrt{6}}, \frac{1}{\sqrt{6}}, \frac{\sqrt{2}}{\sqrt{3}}\right)^{\mathsf T}$$
>
> 计算：
>
> $$\frac{\sqrt{2}}{\sqrt{3}} \cdot \left(-\frac{1}{\sqrt{6}}\right) = -\frac{\sqrt{2}}{\sqrt{18}} = -\frac{1}{3}$$
>
> $$\frac{\sqrt{2}}{\sqrt{3}} \cdot \frac{1}{\sqrt{6}} = \frac{\sqrt{2}}{\sqrt{18}} = \frac{1}{3}$$
>
> $$\frac{\sqrt{2}}{\sqrt{3}} \cdot \frac{\sqrt{2}}{\sqrt{3}} = \frac{2}{3}$$
>
> $$\mathbf{v}_3 = (0,0,1) - (-\tfrac{1}{3}, \tfrac{1}{3}, \tfrac{2}{3}) = (\tfrac{1}{3}, -\tfrac{1}{3}, \tfrac{1}{3})^{\mathsf T} = \frac{1}{3}(1,-1,1)^{\mathsf T}$$
>
> $$\|\mathbf{v}_3\| = \frac{1}{3}\sqrt{1+1+1} = \frac{1}{\sqrt{3}}$$
>
> $$\mathbf{q}_3 = \left(\frac{1}{\sqrt{3}}, -\frac{1}{\sqrt{3}}, \frac{1}{\sqrt{3}}\right)^{\mathsf T}$$
>
> **(2)** 验证正交归一：
>
> $$\mathbf{q}_1 \cdot \mathbf{q}_2 = \frac{1}{\sqrt{2}}\cdot(-\frac{1}{\sqrt{6}}) + \frac{1}{\sqrt{2}}\cdot\frac{1}{\sqrt{6}} + 0 = 0$$
>
> ✓
>
> $$\mathbf{q}_1 \cdot \mathbf{q}_3 = \frac{1}{\sqrt{2}}\cdot\frac{1}{\sqrt{3}} + \frac{1}{\sqrt{2}}\cdot(-\frac{1}{\sqrt{3}}) + 0 = 0$$
>
> ✓
>

$$\mathbf{q}_2 \cdot \mathbf{q}_3 = -\frac{1}{\sqrt{6}}\cdot\frac{1}{\sqrt{3}} + \frac{1}{\sqrt{6}}\cdot(-\frac{1}{\sqrt{3}}) + \frac{\sqrt{2}}{\sqrt{3}}\cdot\frac{1}{\sqrt{3}} = -\frac{1}{\sqrt{18}} - \frac{1}{\sqrt{18}} + \frac{\sqrt{2}}{3} = -\frac{2}{3\sqrt{2}} + \frac{\sqrt{2}}{3} = -\frac{\sqrt{2}}{3} + \frac{\sqrt{2}}{3} = 0$$
> ✓

> 每个 \(\|\mathbf{q}_i\| = 1\)（手工代公式验证）✓。
>
> **(3)** GS 不改变张成的空间：\(\text{span}\{\mathbf{q}_1,\mathbf{q}_2,\mathbf{q}_3\} = \text{span}\{\mathbf{a}_1,\mathbf{a}_2,\mathbf{a}_3\}\)。因为每一步 \(\mathbf{v}_j\) 都是 \(\mathbf{a}_j\) 与前面 \(\mathbf{q}_i\) 的线性组合 → \(\mathbf{q}_j\) 仍在原来的张成空间内。而三维空间中三个无关向量的 GS 结果必张成整个 \(\mathbb{R}^3\)。

---

**A8.** 判断以下关于正交矩阵 $Q$（$Q^{\mathsf T}Q = I$）的命题的真伪：

(1) $Q$ 的所有列都是单位向量且两两正交。
(2) $Q$ 一定是对称矩阵。
(3) $Q$ 作用于任何向量，保持其长度不变：$\|Q\mathbf{x}\| = \|\mathbf{x}\|$。
(4) $Q$ 的特征值的绝对值一定为 $1$。
(5)（ML）为什么正交权重矩阵可以防止梯度消失/爆炸？

> **标准答案：**
>
> **(1) 对。** 这正是 \(Q^{\mathsf T}Q = I\) 的逐列含义：\((Q^{\mathsf T}Q)_{ij} = \mathbf{q}_i \cdot \mathbf{q}_j = \delta_{ij}\)（Kronecker delta）。
>
> **(2) 错。** 反例：\(Q = \begin{bmatrix}0&-1\\1&0\end{bmatrix}\)（旋转 \(90^\circ\)）。\(Q^{\mathsf T}Q = I\)，但 \(Q^{\mathsf T} = \begin{bmatrix}0&1\\-1&0\end{bmatrix} \neq Q\)。
>
> **(3) 对。**
>
> $$\|Q\mathbf{x}\|^2 = (Q\mathbf{x})^{\mathsf T}(Q\mathbf{x}) = \mathbf{x}^{\mathsf T}Q^{\mathsf T}Q\mathbf{x} = \mathbf{x}^{\mathsf T}I\mathbf{x} = \|\mathbf{x}\|^2$$
>
> **(4) 对。** 若 \(Q\mathbf{v} = \lambda\mathbf{v}\)：
>
> $$|\lambda| \cdot \|\mathbf{v}\| = \|\lambda\mathbf{v}\| = \|Q\mathbf{v}\| = \|\mathbf{v}\| \Rightarrow |\lambda| = 1$$
>
> 特征值可以是复数（如旋转矩阵的特征值 \(e^{\pm i\theta}\)），但绝对值恒为 \(1\)。
>
> **(5)** 前向传播：\(\|W\mathbf{x}\| = \|\mathbf{x}\|\)（等距），激活方差逐层不变。反向传播：梯度乘 \(W^{\mathsf T}\)（也是正交的），\(\|W^{\mathsf T}\mathbf{g}\| = \|\mathbf{g}\|\)，梯度方差也不衰减。这意味着网络深度增加时，信号和梯度的尺度保持恒定——不会出现指数级爆炸或消失。**这正是正交初始化（`torch.nn.init.orthogonal_`）的数学根基。**

---

**A9.** 用 $A = \begin{bmatrix}4&2\\2&5\end{bmatrix}$ 的 LU 分解解两个方程组 $A\mathbf{x} = \mathbf{b}_1$ 和 $A\mathbf{x} = \mathbf{b}_2$，其中 $\mathbf{b}_1 = (6, 7)^{\mathsf T}$，$\mathbf{b}_2 = (2, 9)^{\mathsf T}$。

> **标准答案：**
> **LU 分解**：
> 消去 \((2,1)\)：乘数 \(\ell_{21} = 2/4 = 0.5\)。\(R_2 - 0.5R_1\)：
>
> $$U = \begin{bmatrix}4&2\\0&4\end{bmatrix}, \quad L = \begin{bmatrix}1&0\\0.5&1\end{bmatrix}$$
>
> 验证：\(LU = \begin{bmatrix}1&0\\0.5&1\end{bmatrix}\begin{bmatrix}4&2\\0&4\end{bmatrix} = \begin{bmatrix}4&2\\2&5\end{bmatrix} = A\) ✓。
> 注意 \(LU\) 只需算一次！后续解不同 \(\mathbf{b}\) 时重复使用。
> **解 \(\mathbf{b}_1 = (6,7)^{\mathsf T}\)**：
> 前向代入 \(L\mathbf{c} = \mathbf{b}_1\)：
>
> $$c_1 = 6$$
>
> $$0.5c_1 + c_2 = 7 \Rightarrow c_2 = 7 - 3 = 4$$
>
> 回代 \(U\mathbf{x} = \mathbf{c}\)：
>
> $$4x_2 = 4 \Rightarrow x_2 = 1$$
>
> $$4x_1 + 2x_2 = 6 \Rightarrow 4x_1 + 2 = 6 \Rightarrow x_1 = 1$$
>
> \(\mathbf{x} = (1, 1)^{\mathsf T}\)。验证：\(4(1)+2(1)=6\)，\(2(1)+5(1)=7\) ✓。
> **解 \(\mathbf{b}_2 = (2,9)^{\mathsf T}\)**：
> 前向代入：\(c_1 = 2\)，\(0.5(2) + c_2 = 9 \Rightarrow c_2 = 8\)。
> 回代：\(4x_2 = 8 \Rightarrow x_2 = 2\)，\(4x_1 + 4 = 2 \Rightarrow x_1 = -0.5\)。
> \(\mathbf{x} = (-0.5, 2)^{\mathsf T}\)。验证：\(4(-0.5)+2(2)=2\)，\(2(-0.5)+5(2)=9\) ✓。

---

**A10.** 矩阵 $B = \begin{bmatrix}1&3&0\\2&6&1\\0&0&0\end{bmatrix}$。

(1) 求 $B$ 的四个基本子空间及各自的基和维数。
(2) 验证维数公式：$\dim C(B) + \dim N(B) = n$ 和 $\dim C(B^{\mathsf T}) + \dim N(B^{\mathsf T}) = m$。

> **标准答案：**
>
> **(1)** 行化简 \(B\)：
> \(R_2 - 2R_1\)：\((0, 0, 1)\)。行阶梯形：
>
> $$\begin{bmatrix}1&3&0\\0&0&1\\0&0&0\end{bmatrix}, \quad r=2$$
>
> - **列空间** \(C(B) \subset \mathbb{R}^3\)：主元在第 1 和第 3 列。基：
>
> $$\left\{\begin{bmatrix}1\\2\\0\end{bmatrix},\; \begin{bmatrix}0\\1\\0\end{bmatrix}\right\}$$
>
> 维数 \(=2\)。
> - **零空间** \(N(B) \subset \mathbb{R}^3\)：自由变量 \(x_2\)。由行阶梯形：
>
> $$\begin{cases} x_1 + 3x_2 = 0 \\ x_3 = 0 \end{cases}$$
>
> 令 \(x_2 = 1\)：\(\mathbf{n} = (-3, 1, 0)^{\mathsf T}\)。基础解系 \(\{(-3,1,0)^{\mathsf T}\}\)。维数 \(= 3 - 2 = 1\)。
> - **行空间** \(C(B^{\mathsf T}) \subset \mathbb{R}^3\)：非零行构成基：
>
> $$\{(1, 3, 0),\; (0, 0, 1)\}$$
>
> 维数 \(=2\)。
> - **左零空间** \(N(B^{\mathsf T}) \subset \mathbb{R}^3\)：解 \(B^{\mathsf T}\mathbf{y} = \mathbf{0}\)：
>
> $$B^{\mathsf T} = \begin{bmatrix}1&2&0\\3&6&0\\0&1&0\end{bmatrix}, \quad B^{\mathsf T}\mathbf{y} = \begin{bmatrix}y_1+2y_2\\3y_1+6y_2\\y_2\end{bmatrix} = \mathbf{0}$$
>
> \(y_2 = 0\)，\(y_1 = 0\)，\(y_3\) 自由。基：\(\{(0,0,1)^{\mathsf T}\}\)。维数 \(= 3 - 2 = 1\)。
>
> **(2)** 验证：
>
> $$\dim C(B) + \dim N(B) = 2 + 1 = 3 = n$$
>
> ✓
>
> $$\dim C(B^{\mathsf T}) + \dim N(B^{\mathsf T}) = 2 + 1 = 3 = m$$
>
> ✓

---

**A11.** 求点 $\mathbf{b} = (1, 2, 3)^{\mathsf T}$ 到平面 $\text{span}\{(1, 0, 1)^{\mathsf T}, (0, 1, 1)^{\mathsf T}\}$ 的正交投影和距离。

> **标准答案：**
> 令 \(A = \begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}\)（两列即平面的基）。投影矩阵：
>
> $$A^{\mathsf T}A = \begin{bmatrix}1&0&1\\0&1&1\end{bmatrix}\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix} = \begin{bmatrix}2&1\\1&2\end{bmatrix}$$
>
> $$(A^{\mathsf T}A)^{-1} = \frac{1}{4-1}\begin{bmatrix}2&-1\\-1&2\end{bmatrix} = \frac{1}{3}\begin{bmatrix}2&-1\\-1&2\end{bmatrix}$$
>
> 投影坐标：
>
> $$\hat{\mathbf{x}} = (A^{\mathsf T}A)^{-1}A^{\mathsf T}\mathbf{b}$$
>
> $$A^{\mathsf T}\mathbf{b} = \begin{bmatrix}1&0&1\\0&1&1\end{bmatrix}\begin{bmatrix}1\\2\\3\end{bmatrix} = \begin{bmatrix}4\\5\end{bmatrix}$$
>
> $$\hat{\mathbf{x}} = \frac{1}{3}\begin{bmatrix}2&-1\\-1&2\end{bmatrix}\begin{bmatrix}4\\5\end{bmatrix} = \frac{1}{3}\begin{bmatrix}8-5\\-4+10\end{bmatrix} = \frac{1}{3}\begin{bmatrix}3\\6\end{bmatrix} = \begin{bmatrix}1\\2\end{bmatrix}$$
>
> 投影点：
>
> $$\mathbf{p} = A\hat{\mathbf{x}} = 1 \cdot \begin{bmatrix}1\\0\\1\end{bmatrix} + 2 \cdot \begin{bmatrix}0\\1\\1\end{bmatrix} = \begin{bmatrix}1\\2\\3\end{bmatrix}$$
>
> 恰好 \(\mathbf{p} = \mathbf{b}\)！这意味着 \(\mathbf{b}\) 本身就在平面上。验证：\((1,2,3) \cdot \mathbf{n} = 0\)，其中法向量 \(\mathbf{n} = (1,0,1) \times (0,1,1) = (-1,-1,1)\)。\((1,2,3) \cdot (-1,-1,1) = -1-2+3 = 0\) ✓。距离 \(= 0\)。
> 换一个更一般的 \(\mathbf{b}\)：取 \(\mathbf{b} = (1,0,0)^{\mathsf T}\)。\(A^{\mathsf T}\mathbf{b} = (1,0)^{\mathsf T}\)。\(\hat{\mathbf{x}} = \frac{1}{3}(2,-1)^{\mathsf T} = (2/3, -1/3)^{\mathsf T}\)。\(\mathbf{p} = \frac{2}{3}(1,0,1) - \frac{1}{3}(0,1,1) = (2/3, -1/3, 1/3)^{\mathsf T}\)。距离 \(\|\mathbf{b} - \mathbf{p}\| = \|(1/3, 1/3, -1/3)\| = 1/\sqrt{3}\)。

---

### B 档

**B1.** 为什么 `np.linalg.lstsq` 内部使用 SVD 或 QR，而不直接解正规方程？请从条件数角度给出严格论证。

> **标准答案：**
> 正规方程为 \(A^{\mathsf T}A \mathbf{x} = A^{\mathsf T}\mathbf{b}\)。关键问题：条件数被**平方**。
> **定理**：若 \(A\) 满列秩（\(\text{rank}(A) = n\)），则 \(\kappa_2(A^{\mathsf T}A) = \kappa_2(A)^2\)。
>
> **证明**：\(\kappa_2(A^{\mathsf T}A) = \frac{\sigma_{\max}(A^{\mathsf T}A)}{\sigma_{\min}(A^{\mathsf T}A)} = \frac{\sigma_{\max}(A)^2}{\sigma_{\min}(A)^2} = \left(\frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}\right)^2 = \kappa_2(A)^2\)。
>
> **数值后果**：
> - 若 \(\kappa_2(A) = 10^3\)（在特征尺度差异大的数据中常见 → 各列方差差距大），则 \(\kappa_2(A^{\mathsf T}A) = 10^6\)
> - 双精度浮点有约 16 位有效数字。\(10^6\) 的条件数意味着解会损失 \(\log_{10}(10^6) \approx 6\) 位有效数字 → 解可能完全不可靠
> - **QR 方法**：解 \(R\mathbf{x} = Q^{\mathsf T}\mathbf{b}\)，\(R\) 是上三角矩阵，\(\kappa_2(R) = \kappa_2(A)\)（因为 \(Q\) 是正交矩阵，保持奇异值）。没有平方效应。
> - **SVD 方法**：可以**截断**过小的奇异值（设置阈值 \(\sigma_k/\sigma_1 < \varepsilon\)），显式丢弃噪声子空间，得到 Moore-Penrose 伪逆的截断版——数值上最稳健。
> `np.linalg.lstsq` 底层使用基于 QR（小矩阵）或 SVD（大矩阵/秩亏）的方法，从不直接构造 \(A^{\mathsf T}A\)。

---

**B2.** 深度网络前向传播 $\mathbf{h}_{l+1} = \sigma(W_l \mathbf{h}_l)$。假设所有 $W_l$ 的特征值的绝对值均 $>1$。分析反向传播梯度的行为。当特征值均 $<1$ 时又如何？

> **标准答案：**
> 忽略激活函数的非线性（或考虑 \(\sigma'\) 的尺度），反向传播的梯度为：
>
> $$\frac{\partial L}{\partial \mathbf{h}_l} = \frac{\partial L}{\partial \mathbf{h}_L} \cdot \prod_{k=l}^{L-1} \left( \text{diag}(\sigma'(W_k\mathbf{h}_k)) \cdot W_k^{\mathsf T} \right)$$
>
> 简化分析：假设 \(\sigma' \approx 1\)（如 ReLU 激活区域的导数为 1），梯度近似为 \(\frac{\partial L}{\partial \mathbf{h}_L} \cdot \prod_{k=l}^{L-1} W_k^{\mathsf T}\)。
>
> **情况 1**：\(|\lambda(W_k)| > 1\)。以 \(W_k\) 的最大特征值 \(\lambda_{\max}\) 为例（\(|\lambda_{\max}| > 1\)）。连乘 \(L-l\) 个这样的矩阵 → 梯度范数随层数指数增长：
>
> $$\left\|\frac{\partial L}{\partial \mathbf{h}_l}\right\| \approx |\lambda_{\max}|^{L-l} \cdot \left\|\frac{\partial L}{\partial \mathbf{h}_L}\right\|$$
>
> 当 \(L=100\)，\(|\lambda_{\max}| = 1.5\) 时，放大倍数 \(\approx 1.5^{100} \approx 4 \times 10^{17}\) → **梯度爆炸**。
>
> **情况 2**：\(|\lambda(W_k)| < 1\)。梯度范数随层数指数衰减：
>
> $$\left\|\frac{\partial L}{\partial \mathbf{h}_l}\right\| \approx |\lambda_{\max}|^{L-l} \cdot \left\|\frac{\partial L}{\partial \mathbf{h}_L}\right\|$$
>
> 当 \(|\lambda_{\max}| = 0.5\)，\(L=100\)：\(0.5^{100} \approx 7.9 \times 10^{-31}\) → **梯度消失**。
> **Xavier/He 初始化的动机**：控制 \(W\) 各元素的方差，使 \(W\) 的奇异值集中在 \(1\) 附近 → \(|\lambda| \approx 1\) → 前向和反向传播的信号/梯度尺度保持恒定。\(\kappa(W) \approx 1\) 是最理想的状态。

---

**B3.** 假设某优化问题的 Hessian 特征值为 $\{100,\; 10,\; 0.1\}$。分析：

(1) 普通 SGD 的收敛行为（需要大约多少步？）；
(2) Adam 的表现为什么更好？

> **标准答案：**
>
> **(1) Hessian 的条件数**：\(\kappa = \lambda_{\max} / \lambda_{\min} = 100 / 0.1 = 1000\)。
> 在局部二次近似下，梯度下降沿最大特征值方向的收敛速率为 \((1 - \eta\lambda_{\max})\)，沿最小特征值方向为 \((1 - \eta\lambda_{\min})\)。
> 学习率受最大特征值限制：\(\eta < 2/\lambda_{\max} = 0.02\) 以避免震荡。取 \(\eta = 0.01\)。
> 沿 \(\lambda_{\min}=0.1\) 的方向，每步衰减因子为 \(1 - 0.01 \times 0.1 = 0.999\)。要使误差减少到 \(1/e\)，需要步数 \(\approx 1/(\eta\lambda_{\min}) = 1/(0.01 \times 0.1) = 1000\) 步。沿 \(\lambda_{\max}=100\) 的方向，只需 \(1/(0.01 \times 100) = 1\) 步。
> 结论：SGD 在 \(\lambda_{\max}\) 方向飞速收敛（甚至震荡），在 \(\lambda_{\min}\) 方向几乎寸步难行。总共需要 **数百到上千步**才能使所有方向都充分下降。这就是 ill-conditioned 优化问题的核心困难。
>
> **(2)** Adam 维护梯度平方的指数移动平均 \(v_t \approx \mathbb{E}[g^2]\)。在局部二次近似下，\(g \approx H(\theta - \theta^*)\)。如果 Hessian 近似对角（或各参数独立），\(v_t\) 每个分量近似相应 Hessian 对角元的尺度。
> Adam 更新 \(= \eta \cdot m_t / (\sqrt{v_t} + \varepsilon)\)。除以 \(\sqrt{v_t}\) 等价于对角预处理：
> - 大梯度方向（\(\lambda = 100\)）→ \(v_t\) 大 → 有效步长 \(\eta/\sqrt{100} = \eta/10\)（小步，避免震荡）
> - 小梯度方向（\(\lambda = 0.1\)）→ \(v_t\) 小 → 有效步长 \(\eta/\sqrt{0.1} \approx 3.16\eta\)（大步，加速前进）
> 三个方向的有效步长被拉平到相近尺度，优化景观从「瘦长碗」变为「接近球形」→ 条件数被显著改善 → 各方向同步收敛。

---

**B4.** 考虑 $A = \begin{bmatrix}1&1\\1&1.0001\end{bmatrix}$，$\mathbf{b} = (2, 2.0001)^{\mathsf T}$（精确解 $\mathbf{x} = (1,1)^{\mathsf T}$）。

(1) 估算 $A$ 的条件数。
(2) 若 $\mathbf{b}$ 受微小扰动变为 $\tilde{\mathbf{b}} = (2, 2)^{\mathsf T}$，求解 $\tilde{\mathbf{x}}$，观察误差。
(3) 这和深度学习中 ill-conditioned Hessian 引起的训练不稳定有什么关系？

> **标准答案：**
>
> **(1)** \(A\) 的两列几乎平行——第 2 列仅比第 1 列多 \(10^{-4}\)。\(A\) 近乎奇异。
> 特征值：\(\det(A - \lambda I) = (1-\lambda)(1.0001-\lambda) - 1 = \lambda^2 - 2.0001\lambda + 0.0001 = 0\)。
> \(\lambda_{1,2} = \frac{2.0001 \pm \sqrt{4.00040001 - 0.0004}}{2} \approx \frac{2.0001 \pm 2.0000000025}{2}\)
> \(\lambda_1 \approx 2.00005\)，\(\lambda_2 \approx 0.00005\)。
> 条件数 \(\kappa = \lambda_{\max}/\lambda_{\min} \approx 2.00005/0.00005 \approx 4 \times 10^4\)。极大！
>
> **(2)** 解 \(A\tilde{\mathbf{x}} = (2, 2)^{\mathsf T}\)：
>
> $$\begin{cases} \tilde{x}_1 + \tilde{x}_2 = 2 \\ \tilde{x}_1 + 1.0001\tilde{x}_2 = 2 \end{cases}$$
>
> 两式相减：\(0.0001\tilde{x}_2 = 0 \Rightarrow \tilde{x}_2 = 0\)。代入：\(\tilde{x}_1 = 2\)。
> \(\tilde{\mathbf{x}} = (2, 0)^{\mathsf T}\)。精确解是 \(\mathbf{x} = (1, 1)^{\mathsf T}\)。
>
> **误差分析**：\(\mathbf{b}\) 的相对变化 \(\approx \frac{\|(2,2)-(2,2.0001)\|}{\|(2,2.0001)\|} \approx \frac{0.0001}{2.83} \approx 3.5 \times 10^{-5}\)（极小）。但 \(\mathbf{x}\) 的相对变化 \(\approx \frac{\|(2,0)-(1,1)\|}{\|(1,1)\|} \approx \frac{\sqrt{2}}{\sqrt{2}} = 1\)（100%！）。**输入误差被条件数放大了约 \(3 \times 10^4\) 倍**，完全吻合 \(\kappa\) 的估计。
>
> **(3)** 深度学习中，若 Hessian 的条件数很大，mini-batch SGD 的梯度估计中的统计噪声（类比 \(\mathbf{b}\) 的微小扰动）被 ill-conditioned Hessian 放大为权重更新的巨大方差。表现为：training loss 剧烈震荡、某些参数方向几乎不更新、收敛极慢。**这正是 Adam / BatchNorm / 残差连接共同试图解决的问题**——改善优化景观的条件数。

---

**B5.** 在线性回归 $y = X\beta + \varepsilon$ 中，最小二乘估计满足 $X^{\mathsf T}\mathbf{e} = \mathbf{0}$，其中 $\mathbf{e} = y - X\hat{\beta}$。

(1) 用几何语言解释 $X^{\mathsf T}\mathbf{e} = \mathbf{0}$。
(2) 若 $X$ 有一列全 $1$（截距项），推导残差的统计性质。
(3) 证明 $\|y\|^2 = \|X\hat{\beta}\|^2 + \|\mathbf{e}\|^2$。

> **标准答案：**
>
> **(1)** \(X^{\mathsf T}\mathbf{e} = \mathbf{0}\) 展开：\(\mathbf{x}_j^{\mathsf T}\mathbf{e} = 0\) 对 \(X\) 的每一列 \(j\) 成立。这意味着：
> - 残差向量 \(\mathbf{e}\) 与 \(X\) 的**每一列**正交
> - \(\mathbf{e}\) 正交于 \(X\) 的**整个列空间** \(C(X)\)
> - \(X\hat{\beta}\) 是 \(y\) 在 \(C(X)\) 上的正交投影
> - \(\mathbf{e}\) 是投影的残差，垂直于投影面
>
> **(2)** 若 \(\mathbf{x}_1 = (1,1,\dots,1)^{\mathsf T}\)，则 \(\mathbf{1}^{\mathsf T}\mathbf{e} = \sum_{i=1}^n e_i = 0\) → **残差之和为零** → **残差均值为零**（\(\bar{e} = 0\)）。这是线性回归中 \(\sum_i (y_i - \hat{y}_i) = 0\) 的数学根源。
>
> **(3)**
>
> $$\|y\|^2 = \|X\hat{\beta} + \mathbf{e}\|^2 = \|X\hat{\beta}\|^2 + \|\mathbf{e}\|^2 + 2(X\hat{\beta})^{\mathsf T}\mathbf{e}$$
>
> 由 \((X\hat{\beta})^{\mathsf T}\mathbf{e} = \hat{\beta}^{\mathsf T}(X^{\mathsf T}\mathbf{e}) = \hat{\beta}^{\mathsf T} \cdot \mathbf{0} = 0\)。交叉项消失！
> 所以 \(\|y\|^2 = \|X\hat{\beta}\|^2 + \|\mathbf{e}\|^2\)。这就是方差分析（ANOVA）的核心恒等式：
>
> $$\text{SST} = \text{SSR} + \text{SSE}$$
>
> 总平方和 = 回归平方和 + 残差平方和。几何上，这是 \(\mathbb{R}^n\) 中直角三角形的勾股定理——\(y\) 被分解为列空间分量 \(X\hat{\beta}\) 和正交补分量 \(\mathbf{e}\)。

---

[← 返回教程](elimination-and-subspaces.md)　　　[下一章习题 →](determinant-eigen-svd-exercises.md)