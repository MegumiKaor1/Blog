     1|# 习题与思考：方程与子空间
     2|
     3|A 档巩固基础，B 档培养 ML 直觉。每题附完整标准答案。
     4|
     5|---
     6|
     7|## 二、方程与子空间（§6–§11）
     8|
     9|### A 档
    10|
    11|**A1.** 解方程组 $A\mathbf{x}=\mathbf{b}$，其中 $A=\begin{bmatrix}1&2\\2&4\end{bmatrix}$，$\mathbf{b}=(3,6)^{\mathsf T}$。
    12|
    13|(1) 求通解；
    14|(2) 求零空间 $N(A)$ 的一组基础解系。
    15|
    16|> **标准答案：**
    17|> 观察：\(A\) 的第二行是第一行的 2 倍 → \(\text{rank}(A)=1\)。方程组为：
    18|> \\[\begin{cases} x_1 + 2x_2 = 3 \\ 2x_1 + 4x_2 = 6 \end{cases}\\]
    19|> 第二方程是第一方程的 2 倍，不提供新信息。只有一个独立方程：\(x_1 + 2x_2 = 3\)。
    20|>
    21|> **(1)** 特解：令 \(x_2 = 0\)，则 \(x_1 = 3\) → \(\mathbf{x}_p = (3, 0)^{\mathsf T}\)。也可取 \(x_1 = 0\)，\(x_2 = 1.5\) → 另一个特解 \((0, 1.5)^{\mathsf T}\)。
    22|>
    23|> 零空间（齐次方程 \(x_1 + 2x_2 = 0\)）：基础解系 \(\mathbf{v} = (-2, 1)^{\mathsf T}\)（令 \(x_2 = 1\)，\(x_1 = -2\)）。
    24|>
    25|> 通解：\(\mathbf{x} = \mathbf{x}_p + c \cdot \mathbf{v} = \begin{bmatrix}3\\0\end{bmatrix} + c\begin{bmatrix}-2\\1\end{bmatrix}\)，\(c \in \mathbb{R}\)。
    26|>
    27|> **(2)** 零空间 \(N(A)\) 的基础解系：\(\{(-2, 1)^{\mathsf T}\}\)，维数 \(= n - r = 2 - 1 = 1\)。几何上，解集是 \(\mathbb{R}^2\) 中平行于 \((-2,1)\) 方向的一条直线。
    28|
    29|---
    30|
    31|**A2.** 用 QR 分解求 $A\mathbf{x}=\mathbf{b}$ 的最小二乘解，其中
    32|
    33|\[A=\begin{bmatrix}1&1\\1&2\\1&3\end{bmatrix},\quad \mathbf{b}=\begin{bmatrix}1\\2\\2\end{bmatrix}.\]
    34|
    35|> **标准答案：**
    36|>
    37|> **步骤 1：Gram-Schmidt 正交化。**
    38|> \(\mathbf{a}_1 = (1, 1, 1)^{\mathsf T}\)，\(\|\mathbf{a}_1\| = \sqrt{3}\)。
    39|> \\[\mathbf{q}_1 = \frac{1}{\sqrt{3}}(1, 1, 1)^{\mathsf T}, \quad r_{11} = \sqrt{3}\\]
    40|> \(\mathbf{a}_2 = (1, 2, 3)^{\mathsf T}\)。减去在 \(\mathbf{q}_1\) 上的投影：
    41|> \\[\mathbf{a}_2 \cdot \mathbf{q}_1 = \frac{1+2+3}{\sqrt{3}} = \frac{6}{\sqrt{3}} = 2\sqrt{3}\\]
    42|> \\[\mathbf{v}_2 = \mathbf{a}_2 - (\mathbf{a}_2 \cdot \mathbf{q}_1)\mathbf{q}_1 = \begin{bmatrix}1\\2\\3\end{bmatrix} - 2\sqrt{3} \cdot \frac{1}{\sqrt{3}}\begin{bmatrix}1\\1\\1\end{bmatrix} = \begin{bmatrix}1\\2\\3\end{bmatrix} - \begin{bmatrix}2\\2\\2\end{bmatrix} = \begin{bmatrix}-1\\0\\1\end{bmatrix}\\]
    43|> 注意 \(\mathbf{v}_2\) 与 \(\mathbf{q}_1\) 正交（\((-1,0,1) \cdot (1,1,1) = 0\) ✓）。
    44|> \\[\|\mathbf{v}_2\| = \sqrt{1+0+1} = \sqrt{2}, \quad \mathbf{q}_2 = \frac{1}{\sqrt{2}}(-1, 0, 1)^{\mathsf T}, \quad r_{22} = \sqrt{2}\\]
    45|> \(r_{12} = \mathbf{a}_2 \cdot \mathbf{q}_1 = 2\sqrt{3}\)。
    46|>
    47|> **结果**：
    48|> \\[Q = \begin{bmatrix} 1/\sqrt{3} & -1/\sqrt{2} \\ 1/\sqrt{3} & 0 \\ 1/\sqrt{3} & 1/\sqrt{2} \end{bmatrix}, \quad R = \begin{bmatrix} \sqrt{3} & 2\sqrt{3} \\ 0 & \sqrt{2} \end{bmatrix}\\]
    49|> 验证 \(QR = A\)：
    50|> 第一列：\(\sqrt{3} \cdot \mathbf{q}_1 = (1,1,1)^{\mathsf T}\) ✓。第二列：\(2\sqrt{3}\mathbf{q}_1 + \sqrt{2}\mathbf{q}_2 = (2,2,2) + (-1,0,1) = (1,2,3)^{\mathsf T}\) ✓。
    51|>
    52|> **步骤 2：解 \(R\mathbf{x} = Q^{\mathsf T}\mathbf{b}\)。**
    53|> \\[Q^{\mathsf T}\mathbf{b} = \begin{bmatrix} 1/\sqrt{3} & 1/\sqrt{3} & 1/\sqrt{3} \\ -1/\sqrt{2} & 0 & 1/\sqrt{2} \end{bmatrix} \begin{bmatrix}1\\2\\2\end{bmatrix}\\]
    54|> \\[= \begin{bmatrix}(1+2+2)/\sqrt{3} \\ (-1+0+2)/\sqrt{2}\end{bmatrix} = \begin{bmatrix}5/\sqrt{3} \\ 1/\sqrt{2}\end{bmatrix}\\]
    55|> 解上三角方程：
    56|> \\[\begin{bmatrix}\sqrt{3} & 2\sqrt{3} \\ 0 & \sqrt{2}\end{bmatrix} \begin{bmatrix}x_1\\x_2\end{bmatrix} = \begin{bmatrix}5/\sqrt{3} \\ 1/\sqrt{2}\end{bmatrix}\\]
    57|> \(\sqrt{2} x_2 = 1/\sqrt{2} \Rightarrow x_2 = 1/2\)。
    58|> \(\sqrt{3}x_1 + 2\sqrt{3} \cdot \frac{1}{2} = 5/\sqrt{3} \Rightarrow \sqrt{3}x_1 = 5/\sqrt{3} - \sqrt{3} = \frac{5-3}{\sqrt{3}} = 2/\sqrt{3} \Rightarrow x_1 = 2/3\)。
    59|> 最小二乘解：\(\mathbf{x} = (2/3,\; 1/2)^{\mathsf T}\)。
    60|>
    61|> **注**：QR 解法避免了正规方程中的条件数平方问题。\(A^{\mathsf T}A\) 的直接求解需要 \(\kappa(A)^2\) 的精度，QR 只需要 \(\kappa(A)\)。
    62|
    63|---
    64|
    65|**A3.** 给定协方差矩阵 $\Sigma = \begin{bmatrix}4&2\\2&3\end{bmatrix}$。
    66|
    67|(1) 验证 $\Sigma$ 是正定的；
    68|(2) 求其 Cholesky 分解 $\Sigma = LL^{\mathsf T}$；
    69|(3) 用 $L$ 生成 $3$ 个来自 $\mathcal{N}(\mathbf{0},\Sigma)$ 的随机样本。
    70|
    71|> **标准答案：**
    72|>
    73|> **(1)** 顺序主子式法：
    74|> \\[\Delta_1 = 4 > 0\\]
    75|> \\[\Delta_2 = \det(\Sigma) = 4 \times 3 - 2 \times 2 = 12 - 4 = 8 > 0\\]
    76|> 所有顺序主子式 \(>0\) → \(\Sigma\) 正定 ✓。也可验证特征值：\(\det(\Sigma - \lambda I) = (4-\lambda)(3-\lambda) - 4 = \lambda^2 - 7\lambda + 8 = 0\)，\(\lambda = \frac{7 \pm \sqrt{49-32}}{2} = \frac{7 \pm \sqrt{17}}{2} \approx 5.56, 1.44\)，全部 \(>0\)。
    77|>
    78|> **(2)** Cholesky 分解递推公式：
    79|> \\[L_{11} = \sqrt{\Sigma_{11}} = \sqrt{4} = 2\\]
    80|> \\[L_{21} = \frac{\Sigma_{21}}{L_{11}} = \frac{2}{2} = 1\\]
    81|> \\[L_{22} = \sqrt{\Sigma_{22} - L_{21}^2} = \sqrt{3 - 1} = \sqrt{2}\\]
    82|> \\[L = \begin{bmatrix}2 & 0 \\ 1 & \sqrt{2}\end{bmatrix}\\]
    83|> 验证：\(LL^{\mathsf T} = \begin{bmatrix}2&0\\1&\sqrt{2}\end{bmatrix}\begin{bmatrix}2&1\\0&\sqrt{2}\end{bmatrix} = \begin{bmatrix}4&2\\2&3\end{bmatrix} = \Sigma\) ✓。
    84|>
    85|> **(3)** 采样公式：\(\mathbf{x} = L\mathbf{z}\)，\(\mathbf{z} \sim \mathcal{N}(\mathbf{0}, I_2)\)。
    86|> 验证：\(\mathbb{E}[\mathbf{x}\mathbf{x}^{\mathsf T}] = \mathbb{E}[L\mathbf{z}\mathbf{z}^{\mathsf T}L^{\mathsf T}] = L \cdot I \cdot L^{\mathsf T} = \Sigma\)。
    87|> Python 代码：
    88|> ```python
    89|> import numpy as np
    90|> L = np.array([[2, 0], [1, np.sqrt(2)]])
    91|> z = np.random.randn(2, 3)  # 2维, 3个样本
    92|> x = L @ z
    93|> ```
    94|> 这三个样本的样本协方差矩阵应近似 \(\Sigma\)（样本量小时有随机波动）。
    95|
    96|---
    97|
    98|**A4.** 对矩阵 $A = \begin{bmatrix}2&1&0\\4&3&1\\-2&2&3\end{bmatrix}$，求其 LU 分解 $A = LU$。然后用 LU 解 $A\mathbf{x} = (1,3,7)^{\mathsf T}$。
    99|
   100|> **标准答案：**
   101|>
   102|> **消元步骤：**
   103|> ① 消去 \((2,1)\)：乘数 \(\ell_{21} = 4/2 = 2\)。\(R_2 \leftarrow R_2 - 2R_1\)：
   104|> \\[U_1 = \begin{bmatrix}2&1&0\\0&1&1\\-2&2&3\end{bmatrix}\\]
   105|> 消去 \((3,1)\)：乘数 \(\ell_{31} = -2/2 = -1\)。\(R_3 \leftarrow R_3 - (-1)R_1\)（即 \(R_3 + R_1\)）：
   106|> \\[U_2 = \begin{bmatrix}2&1&0\\0&1&1\\0&3&3\end{bmatrix}\\]
   107|> ② 消去 \((3,2)\)：乘数 \(\ell_{32} = 3/1 = 3\)。\(R_3 \leftarrow R_3 - 3R_2\)：
   108|> \\[U = \begin{bmatrix}2&1&0\\0&1&1\\0&0&0\end{bmatrix}\\]
   109|> \(L\) 由乘数构成（对角线上全为 \(1\)）：
   110|> \\[L = \begin{bmatrix}1&0&0\\\ell_{21}&1&0\\\ell_{31}&\ell_{32}&1\end{bmatrix} = \begin{bmatrix}1&0&0\\2&1&0\\-1&3&1\end{bmatrix}\\]
   111|> 验证：\(LU = \begin{bmatrix}1&0&0\\2&1&0\\-1&3&1\end{bmatrix}\begin{bmatrix}2&1&0\\0&1&1\\0&0&0\end{bmatrix} = \begin{bmatrix}2&1&0\\4&3&1\\-2&2&3\end{bmatrix} = A\) ✓。
   112|> 注意 \(U\) 有零行 → \(A\) 秩亏（秩 \(=2\)）。
   113|>
   114|> **用 LU 解方程**：
   115|> ① 前向代入 \(L\mathbf{c} = \mathbf{b}\)：
   116|> \\[\begin{bmatrix}1&0&0\\2&1&0\\-1&3&1\end{bmatrix}\begin{bmatrix}c_1\\c_2\\c_3\end{bmatrix} = \begin{bmatrix}1\\3\\7\end{bmatrix}\\]
   117|> \\[c_1 = 1\\]
   118|> \\[2c_1 + c_2 = 3 \Rightarrow c_2 = 3 - 2 = 1\\]
   119|> \\[-c_1 + 3c_2 + c_3 = 7 \Rightarrow -1 + 3 + c_3 = 7 \Rightarrow c_3 = 5\\]
   120|> ② 回代 \(U\mathbf{x} = \mathbf{c}\)：
   121|> \\[\begin{bmatrix}2&1&0\\0&1&1\\0&0&0\end{bmatrix}\begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix} = \begin{bmatrix}1\\1\\5\end{bmatrix}\\]
   122|> 第三行：\(0 \cdot x_3 = 5\) → **矛盾！** 方程组无解。\(\mathbf{b} = (1,3,7)^{\mathsf T}\) 不在 \(A\) 的列空间里。
   123|> 验证列空间：\(A\) 的列空间由前两列张成，维数 \(=2\)。\(\mathbf{b}\) 是否在其中？检查 \(\mathbf{b}\) 减去前两列的线性组合后第三分量：\((1,3,7) - c_1(2,4,-2) - c_2(1,3,2)\)。第三分量 \(= 7 + 2c_1 - 2c_2\)。当 \(c_1 = 0, c_2 = -0.5\) 时，第三分量 \(= 7 + 0 + 1 = 8 \neq 0\)。\(\mathbf{b}\) 不在 \(xy\) 平面内 → 不在列空间中。
   124|
   125|---
   126|
   127|**A5.** 设 $A = \begin{bmatrix}1&2&0&0\\0&0&1&0\\0&0&0&0\end{bmatrix}$，$\mathbf{b} = (1,2,0)^{\mathsf T}$。
   128|
   129|(1) 写出 $A$ 的四个基本子空间，每子空间各给一组基和维数。
   130|(2) $\mathbf{b}$ 在列空间里吗？有解吗？若可解，求通解。
   131|
   132|> **标准答案：**
   133|> \(A\) 已是行阶梯形，秩 \(r=2\)（主元在第 1 列和第 3 列）。
   134|>
   135|> **(1)** 四个基本子空间：
   136|> - **列空间** \(C(A) \subset \mathbb{R}^3\)：由 \(A\) 的主元列（第 1 和第 3 列）张成。基：
   137|> \\[\left\{\begin{bmatrix}1\\0\\0\end{bmatrix},\; \begin{bmatrix}0\\1\\0\end{bmatrix}\right\}\\]
   138|> 维数 \(=2\)。这是 \(\mathbb{R}^3\) 中的 \(xy\) 平面。
   139|> - **零空间** \(N(A) \subset \mathbb{R}^4\)：解 \(A\mathbf{x} = \mathbf{0}\)：
   140|> \\[\begin{cases} x_1 + 2x_2 = 0 \\ x_3 = 0 \end{cases}\\]
   141|> 自由变量：\(x_2, x_4\)。令 \(x_2=1, x_4=0\)：\(\mathbf{n}_1 = (-2, 1, 0, 0)^{\mathsf T}\)。令 \(x_2=0, x_4=1\)：\(\mathbf{n}_2 = (0, 0, 0, 1)^{\mathsf T}\)。基础解系：\(\{\mathbf{n}_1, \mathbf{n}_2\}\)。维数 \(=4-2=2\)。
   142|> - **行空间** \(C(A^{\mathsf T}) \subset \mathbb{R}^4\)：\(A\) 的非零行构成基：
   143|> \\[\{(1, 2, 0, 0),\; (0, 0, 1, 0)\}\\]
   144|> 维数 \(=2\)。注意行空间与零空间正交（内积验证：\((1,2,0,0) \cdot (-2,1,0,0) = 0\) ✓）。
   145|> - **左零空间** \(N(A^{\mathsf T}) \subset \mathbb{R}^3\)：解 \(A^{\mathsf T}\mathbf{y} = \mathbf{0}\)：
   146|> \\[A^{\mathsf T} = \begin{bmatrix}1&0&0\\2&0&0\\0&1&0\\0&0&0\end{bmatrix},\quad A^{\mathsf T}\mathbf{y} = \begin{bmatrix}y_1\\2y_1\\y_2\\0\end{bmatrix} = \mathbf{0}\\]
   147|> \(y_1=0\)，\(y_2=0\)，\(y_3\) 自由。基：\(\{(0, 0, 1)^{\mathsf T}\}\)。维数 \(=3-2=1\)。几何上，左零空间是 \(z\) 轴——垂直于 \(xy\) 平面（列空间）。
   148|>
   149|> **(2)** \(\mathbf{b} = (1, 2, 0)^{\mathsf T}\) 的第三分量是 \(0\) → 在 \(xy\) 平面（列空间）内 → 有解。
   150|> 特解：主元列变量 \(x_1, x_3\)，自由变量 \(x_2=x_4=0\)。由方程组：
   151|> \\[x_1 = 1,\quad x_3 = 2\\]
   152|> \(\mathbf{x}_p = (1, 0, 2, 0)^{\mathsf T}\)。
   153|> 通解：\(\mathbf{x} = \mathbf{x}_p + c_1\mathbf{n}_1 + c_2\mathbf{n}_2 = (1, 0, 2, 0)^{\mathsf T} + c_1(-2, 1, 0, 0)^{\mathsf T} + c_2(0, 0, 0, 1)^{\mathsf T}\)。
   154|
   155|---
   156|
   157|**A6.** 设 $\mathbf{a} = (1,2,2)^{\mathsf T}$。求投影到 $\mathbf{a}$ 所在直线上的投影矩阵 $P$，并验证 $P^2 = P$ 和 $P^{\mathsf T} = P$。
   158|
   159|> **标准答案：**
   160|> 一维投影公式：\(P = \frac{\mathbf{a}\mathbf{a}^{\mathsf T}}{\mathbf{a}^{\mathsf T}\mathbf{a}}\)。
   161|> \\[\mathbf{a}^{\mathsf T}\mathbf{a} = 1^2 + 2^2 + 2^2 = 1 + 4 + 4 = 9\\]
   162|> \\[\mathbf{a}\mathbf{a}^{\mathsf T} = \begin{bmatrix}1\\2\\2\end{bmatrix}\begin{bmatrix}1&2&2\end{bmatrix} = \begin{bmatrix}1&2&2\\2&4&4\\2&4&4\end{bmatrix}\\]
   163|> \\[P = \frac{1}{9}\begin{bmatrix}1&2&2\\2&4&4\\2&4&4\end{bmatrix}\\]
   164|>
   165|> **验证 \(P^2 = P\)**：
   166|> \\[P^2 = \frac{1}{81}\begin{bmatrix}1&2&2\\2&4&4\\2&4&4\end{bmatrix}\begin{bmatrix}1&2&2\\2&4&4\\2&4&4\end{bmatrix}\\]
   167|> 算第一行第一列：\(\frac{1}{81}(1\cdot1 + 2\cdot2 + 2\cdot2) = \frac{1+4+4}{81} = \frac{9}{81} = \frac{1}{9} = P_{11}\)。所有元素类似验证 → \(P^2 = P\)。
   168|>
   169|> **验证 \(P^{\mathsf T} = P\)**：矩阵显然对称 ✓。
   170|>
   171|> **数值验证**：\(\mathbf{b} = (3, 0, 0)^{\mathsf T}\)。投影：
   172|> \\[P\mathbf{b} = \frac{1}{9}\begin{bmatrix}1&2&2\\2&4&4\\2&4&4\end{bmatrix}\begin{bmatrix}3\\0\\0\end{bmatrix} = \frac{1}{9}\begin{bmatrix}3\\6\\6\end{bmatrix} = \begin{bmatrix}1/3\\2/3\\2/3\end{bmatrix}\\]
   173|> 残差 \(\mathbf{e} = \mathbf{b} - P\mathbf{b} = (8/3, -2/3, -2/3)^{\mathsf T}\)。验证正交性：
   174|> \\[\mathbf{e} \cdot \mathbf{a} = \frac{8}{3} \cdot 1 + \left(-\frac{2}{3}\right) \cdot 2 + \left(-\frac{2}{3}\right) \cdot 2 = \frac{8-4-4}{3} = 0\\]
   175|> 残差 \(\perp\) 投影方向 ✓。
   176|
   177|---
   178|
   179|**A7.** 对下列向量组执行 Gram-Schmidt 正交化：
   180|
   181|\[\mathbf{a}_1 = (1, 1, 0)^{\mathsf T}, \quad \mathbf{a}_2 = (0, 1, 1)^{\mathsf T}, \quad \mathbf{a}_3 = (0, 0, 1)^{\mathsf T}.\]
   182|
   183|> **标准答案：**
   184|>
   185|> **(1)**
   186|> \\[\mathbf{q}_1 = \frac{\mathbf{a}_1}{\|\mathbf{a}_1\|} = \frac{1}{\sqrt{2}}(1, 1, 0)^{\mathsf T}\\]
   187|> \\[\mathbf{v}_2 = \mathbf{a}_2 - (\mathbf{a}_2 \cdot \mathbf{q}_1)\mathbf{q}_1\\]
   188|> \\[\mathbf{a}_2 \cdot \mathbf{q}_1 = 0 \cdot \frac{1}{\sqrt{2}} + 1 \cdot \frac{1}{\sqrt{2}} + 1 \cdot 0 = \frac{1}{\sqrt{2}}\\]
   189|> \\[\mathbf{v}_2 = (0,1,1) - \frac{1}{\sqrt{2}} \cdot \frac{1}{\sqrt{2}}(1,1,0) = (0,1,1) - (\tfrac{1}{2},\tfrac{1}{2},0) = (-\tfrac{1}{2}, \tfrac{1}{2}, 1)^{\mathsf T}\\]
   190|> \\[\|\mathbf{v}_2\| = \sqrt{\tfrac{1}{4} + \tfrac{1}{4} + 1} = \sqrt{\tfrac{3}{2}} = \frac{\sqrt{3}}{\sqrt{2}}\\]
   191|> \\[\mathbf{q}_2 = \frac{\sqrt{2}}{\sqrt{3}}(-\tfrac{1}{2}, \tfrac{1}{2}, 1)^{\mathsf T} = \left(-\frac{1}{\sqrt{6}}, \frac{1}{\sqrt{6}}, \frac{\sqrt{2}}{\sqrt{3}}\right)^{\mathsf T}\\]
   192|> \\[\mathbf{v}_3 = \mathbf{a}_3 - (\mathbf{a}_3 \cdot \mathbf{q}_1)\mathbf{q}_1 - (\mathbf{a}_3 \cdot \mathbf{q}_2)\mathbf{q}_2\\]
   193|> \\[\mathbf{a}_3 \cdot \mathbf{q}_1 = 0\\]
   194|> \\[\mathbf{a}_3 \cdot \mathbf{q}_2 = 0 \cdot (-\tfrac{1}{\sqrt{6}}) + 0 \cdot \tfrac{1}{\sqrt{6}} + 1 \cdot \tfrac{\sqrt{2}}{\sqrt{3}} = \frac{\sqrt{2}}{\sqrt{3}}\\]
   195|> \\[\mathbf{v}_3 = (0,0,1) - 0 - \frac{\sqrt{2}}{\sqrt{3}}\left(-\frac{1}{\sqrt{6}}, \frac{1}{\sqrt{6}}, \frac{\sqrt{2}}{\sqrt{3}}\right)^{\mathsf T}\\]
   196|> 计算：
   197|> \\[\frac{\sqrt{2}}{\sqrt{3}} \cdot \left(-\frac{1}{\sqrt{6}}\right) = -\frac{\sqrt{2}}{\sqrt{18}} = -\frac{1}{3}\\]
   198|> \\[\frac{\sqrt{2}}{\sqrt{3}} \cdot \frac{1}{\sqrt{6}} = \frac{\sqrt{2}}{\sqrt{18}} = \frac{1}{3}\\]
   199|> \\[\frac{\sqrt{2}}{\sqrt{3}} \cdot \frac{\sqrt{2}}{\sqrt{3}} = \frac{2}{3}\\]
   200|> \\[\mathbf{v}_3 = (0,0,1) - (-\tfrac{1}{3}, \tfrac{1}{3}, \tfrac{2}{3}) = (\tfrac{1}{3}, -\tfrac{1}{3}, \tfrac{1}{3})^{\mathsf T} = \frac{1}{3}(1,-1,1)^{\mathsf T}\\]
   201|> \\[\|\mathbf{v}_3\| = \frac{1}{3}\sqrt{1+1+1} = \frac{1}{\sqrt{3}}\\]
   202|> \\[\mathbf{q}_3 = \left(\frac{1}{\sqrt{3}}, -\frac{1}{\sqrt{3}}, \frac{1}{\sqrt{3}}\right)^{\mathsf T}\\]
   203|>
   204|> **(2)** 验证正交归一：
   205|> \\[\mathbf{q}_1 \cdot \mathbf{q}_2 = \frac{1}{\sqrt{2}}\cdot(-\frac{1}{\sqrt{6}}) + \frac{1}{\sqrt{2}}\cdot\frac{1}{\sqrt{6}} + 0 = 0\\] ✓
   206|> \\[\mathbf{q}_1 \cdot \mathbf{q}_3 = \frac{1}{\sqrt{2}}\cdot\frac{1}{\sqrt{3}} + \frac{1}{\sqrt{2}}\cdot(-\frac{1}{\sqrt{3}}) + 0 = 0\\] ✓
   207|> \\[\mathbf{q}_2 \cdot \mathbf{q}_3 = -\frac{1}{\sqrt{6}}\cdot\frac{1}{\sqrt{3}} + \frac{1}{\sqrt{6}}\cdot(-\frac{1}{\sqrt{3}}) + \frac{\sqrt{2}}{\sqrt{3}}\cdot\frac{1}{\sqrt{3}} = -\frac{1}{\sqrt{18}} - \frac{1}{\sqrt{18}} + \frac{\sqrt{2}}{3}
   208|> = -\frac{2}{3\sqrt{2}} + \frac{\sqrt{2}}{3} = -\frac{\sqrt{2}}{3} + \frac{\sqrt{2}}{3} = 0\\] ✓
   209|> 每个 \(\|\mathbf{q}_i\| = 1\)（手工代公式验证）✓。
   210|>
   211|> **(3)** GS 不改变张成的空间：\(\text{span}\{\mathbf{q}_1,\mathbf{q}_2,\mathbf{q}_3\} = \text{span}\{\mathbf{a}_1,\mathbf{a}_2,\mathbf{a}_3\}\)。因为每一步 \(\mathbf{v}_j\) 都是 \(\mathbf{a}_j\) 与前面 \(\mathbf{q}_i\) 的线性组合 → \(\mathbf{q}_j\) 仍在原来的张成空间内。而三维空间中三个无关向量的 GS 结果必张成整个 \(\mathbb{R}^3\)。
   212|
   213|---
   214|
   215|**A8.** 判断以下关于正交矩阵 $Q$（$Q^{\mathsf T}Q = I$）的命题的真伪：
   216|
   217|(1) $Q$ 的所有列都是单位向量且两两正交。
   218|(2) $Q$ 一定是对称矩阵。
   219|(3) $Q$ 作用于任何向量，保持其长度不变：$\|Q\mathbf{x}\| = \|\mathbf{x}\|$。
   220|(4) $Q$ 的特征值的绝对值一定为 $1$。
   221|(5)（ML）为什么正交权重矩阵可以防止梯度消失/爆炸？
   222|
   223|> **标准答案：**
   224|>
   225|> **(1) 对。** 这正是 \(Q^{\mathsf T}Q = I\) 的逐列含义：\((Q^{\mathsf T}Q)_{ij} = \mathbf{q}_i \cdot \mathbf{q}_j = \delta_{ij}\)（Kronecker delta）。
   226|>
   227|> **(2) 错。** 反例：\(Q = \begin{bmatrix}0&-1\\1&0\end{bmatrix}\)（旋转 \(90^\circ\)）。\(Q^{\mathsf T}Q = I\)，但 \(Q^{\mathsf T} = \begin{bmatrix}0&1\\-1&0\end{bmatrix} \neq Q\)。
   228|>
   229|> **(3) 对。**
   230|> \\[\|Q\mathbf{x}\|^2 = (Q\mathbf{x})^{\mathsf T}(Q\mathbf{x}) = \mathbf{x}^{\mathsf T}Q^{\mathsf T}Q\mathbf{x} = \mathbf{x}^{\mathsf T}I\mathbf{x} = \|\mathbf{x}\|^2\\]
   231|>
   232|> **(4) 对。** 若 \(Q\mathbf{v} = \lambda\mathbf{v}\)：
   233|> \\[|\lambda| \cdot \|\mathbf{v}\| = \|\lambda\mathbf{v}\| = \|Q\mathbf{v}\| = \|\mathbf{v}\| \Rightarrow |\lambda| = 1\\]
   234|> 特征值可以是复数（如旋转矩阵的特征值 \(e^{\pm i\theta}\)），但绝对值恒为 \(1\)。
   235|>
   236|> **(5)** 前向传播：\(\|W\mathbf{x}\| = \|\mathbf{x}\|\)（等距），激活方差逐层不变。反向传播：梯度乘 \(W^{\mathsf T}\)（也是正交的），\(\|W^{\mathsf T}\mathbf{g}\| = \|\mathbf{g}\|\)，梯度方差也不衰减。这意味着网络深度增加时，信号和梯度的尺度保持恒定——不会出现指数级爆炸或消失。**这正是正交初始化（`torch.nn.init.orthogonal_`）的数学根基。**
   237|
   238|---
   239|
   240|**A9.** 用 $A = \begin{bmatrix}4&2\\2&5\end{bmatrix}$ 的 LU 分解解两个方程组 $A\mathbf{x} = \mathbf{b}_1$ 和 $A\mathbf{x} = \mathbf{b}_2$，其中 $\mathbf{b}_1 = (6, 7)^{\mathsf T}$，$\mathbf{b}_2 = (2, 9)^{\mathsf T}$。
   241|
   242|> **标准答案：**
   243|> **LU 分解**：
   244|> 消去 \((2,1)\)：乘数 \(\ell_{21} = 2/4 = 0.5\)。\(R_2 - 0.5R_1\)：
   245|> \\[U = \begin{bmatrix}4&2\\0&4\end{bmatrix}, \quad L = \begin{bmatrix}1&0\\0.5&1\end{bmatrix}\\]
   246|> 验证：\(LU = \begin{bmatrix}1&0\\0.5&1\end{bmatrix}\begin{bmatrix}4&2\\0&4\end{bmatrix} = \begin{bmatrix}4&2\\2&5\end{bmatrix} = A\) ✓。
   247|> 注意 \(LU\) 只需算一次！后续解不同 \(\mathbf{b}\) 时重复使用。
   248|> **解 \(\mathbf{b}_1 = (6,7)^{\mathsf T}\)**：
   249|> 前向代入 \(L\mathbf{c} = \mathbf{b}_1\)：
   250|> \\[c_1 = 6\\]
   251|> \\[0.5c_1 + c_2 = 7 \Rightarrow c_2 = 7 - 3 = 4\\]
   252|> 回代 \(U\mathbf{x} = \mathbf{c}\)：
   253|> \\[4x_2 = 4 \Rightarrow x_2 = 1\\]
   254|> \\[4x_1 + 2x_2 = 6 \Rightarrow 4x_1 + 2 = 6 \Rightarrow x_1 = 1\\]
   255|> \(\mathbf{x} = (1, 1)^{\mathsf T}\)。验证：\(4(1)+2(1)=6\)，\(2(1)+5(1)=7\) ✓。
   256|> **解 \(\mathbf{b}_2 = (2,9)^{\mathsf T}\)**：
   257|> 前向代入：\(c_1 = 2\)，\(0.5(2) + c_2 = 9 \Rightarrow c_2 = 8\)。
   258|> 回代：\(4x_2 = 8 \Rightarrow x_2 = 2\)，\(4x_1 + 4 = 2 \Rightarrow x_1 = -0.5\)。
   259|> \(\mathbf{x} = (-0.5, 2)^{\mathsf T}\)。验证：\(4(-0.5)+2(2)=2\)，\(2(-0.5)+5(2)=9\) ✓。
   260|
   261|---
   262|
   263|**A10.** 矩阵 $B = \begin{bmatrix}1&3&0\\2&6&1\\0&0&0\end{bmatrix}$。
   264|
   265|(1) 求 $B$ 的四个基本子空间及各自的基和维数。
   266|(2) 验证维数公式：$\dim C(B) + \dim N(B) = n$ 和 $\dim C(B^{\mathsf T}) + \dim N(B^{\mathsf T}) = m$。
   267|
   268|> **标准答案：**
   269|>
   270|> **(1)** 行化简 \(B\)：
   271|> \(R_2 - 2R_1\)：\((0, 0, 1)\)。行阶梯形：
   272|> \\[\begin{bmatrix}1&3&0\\0&0&1\\0&0&0\end{bmatrix}, \quad r=2\\]
   273|> - **列空间** \(C(B) \subset \mathbb{R}^3\)：主元在第 1 和第 3 列。基：
   274|> \\[\left\{\begin{bmatrix}1\\2\\0\end{bmatrix},\; \begin{bmatrix}0\\1\\0\end{bmatrix}\right\}\\]
   275|> 维数 \(=2\)。
   276|> - **零空间** \(N(B) \subset \mathbb{R}^3\)：自由变量 \(x_2\)。由行阶梯形：
   277|> \\[\begin{cases} x_1 + 3x_2 = 0 \\ x_3 = 0 \end{cases}\\]
   278|> 令 \(x_2 = 1\)：\(\mathbf{n} = (-3, 1, 0)^{\mathsf T}\)。基础解系 \(\{(-3,1,0)^{\mathsf T}\}\)。维数 \(= 3 - 2 = 1\)。
   279|> - **行空间** \(C(B^{\mathsf T}) \subset \mathbb{R}^3\)：非零行构成基：
   280|> \\[\{(1, 3, 0),\; (0, 0, 1)\}\\]
   281|> 维数 \(=2\)。
   282|> - **左零空间** \(N(B^{\mathsf T}) \subset \mathbb{R}^3\)：解 \(B^{\mathsf T}\mathbf{y} = \mathbf{0}\)：
   283|> \\[B^{\mathsf T} = \begin{bmatrix}1&2&0\\3&6&0\\0&1&0\end{bmatrix}, \quad B^{\mathsf T}\mathbf{y} = \begin{bmatrix}y_1+2y_2\\3y_1+6y_2\\y_2\end{bmatrix} = \mathbf{0}\\]
   284|> \(y_2 = 0\)，\(y_1 = 0\)，\(y_3\) 自由。基：\(\{(0,0,1)^{\mathsf T}\}\)。维数 \(= 3 - 2 = 1\)。
   285|>
   286|> **(2)** 验证：
   287|> \\[\dim C(B) + \dim N(B) = 2 + 1 = 3 = n\\] ✓
   288|> \\[\dim C(B^{\mathsf T}) + \dim N(B^{\mathsf T}) = 2 + 1 = 3 = m\\] ✓
   289|
   290|---
   291|
   292|**A11.** 求点 $\mathbf{b} = (1, 2, 3)^{\mathsf T}$ 到平面 $\text{span}\{(1, 0, 1)^{\mathsf T}, (0, 1, 1)^{\mathsf T}\}$ 的正交投影和距离。
   293|
   294|> **标准答案：**
   295|> 令 \(A = \begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}\)（两列即平面的基）。投影矩阵：
   296|> \\[A^{\mathsf T}A = \begin{bmatrix}1&0&1\\0&1&1\end{bmatrix}\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix} = \begin{bmatrix}2&1\\1&2\end{bmatrix}\\]
   297|> \\[(A^{\mathsf T}A)^{-1} = \frac{1}{4-1}\begin{bmatrix}2&-1\\-1&2\end{bmatrix} = \frac{1}{3}\begin{bmatrix}2&-1\\-1&2\end{bmatrix}\\]
   298|> 投影坐标：
   299|> \\[\hat{\mathbf{x}} = (A^{\mathsf T}A)^{-1}A^{\mathsf T}\mathbf{b}\\]
   300|> \\[A^{\mathsf T}\mathbf{b} = \begin{bmatrix}1&0&1\\0&1&1\end{bmatrix}\begin{bmatrix}1\\2\\3\end{bmatrix} = \begin{bmatrix}4\\5\end{bmatrix}\\]
   301|> \\[\hat{\mathbf{x}} = \frac{1}{3}\begin{bmatrix}2&-1\\-1&2\end{bmatrix}\begin{bmatrix}4\\5\end{bmatrix} = \frac{1}{3}\begin{bmatrix}8-5\\-4+10\end{bmatrix} = \frac{1}{3}\begin{bmatrix}3\\6\end{bmatrix} = \begin{bmatrix}1\\2\end{bmatrix}\\]
   302|> 投影点：
   303|> \\[\mathbf{p} = A\hat{\mathbf{x}} = 1 \cdot \begin{bmatrix}1\\0\\1\end{bmatrix} + 2 \cdot \begin{bmatrix}0\\1\\1\end{bmatrix} = \begin{bmatrix}1\\2\\3\end{bmatrix}\\]
   304|> 恰好 \(\mathbf{p} = \mathbf{b}\)！这意味着 \(\mathbf{b}\) 本身就在平面上。验证：\((1,2,3) \cdot \mathbf{n} = 0\)，其中法向量 \(\mathbf{n} = (1,0,1) \times (0,1,1) = (-1,-1,1)\)。\((1,2,3) \cdot (-1,-1,1) = -1-2+3 = 0\) ✓。距离 \(= 0\)。
   305|> 换一个更一般的 \(\mathbf{b}\)：取 \(\mathbf{b} = (1,0,0)^{\mathsf T}\)。\(A^{\mathsf T}\mathbf{b} = (1,0)^{\mathsf T}\)。\(\hat{\mathbf{x}} = \frac{1}{3}(2,-1)^{\mathsf T} = (2/3, -1/3)^{\mathsf T}\)。\(\mathbf{p} = \frac{2}{3}(1,0,1) - \frac{1}{3}(0,1,1) = (2/3, -1/3, 1/3)^{\mathsf T}\)。距离 \(\|\mathbf{b} - \mathbf{p}\| = \|(1/3, 1/3, -1/3)\| = 1/\sqrt{3}\)。
   306|
   307|---
   308|
   309|### B 档
   310|
   311|**B1.** 为什么 `np.linalg.lstsq` 内部使用 SVD 或 QR，而不直接解正规方程？请从条件数角度给出严格论证。
   312|
   313|> **标准答案：**
   314|> 正规方程为 \(A^{\mathsf T}A \mathbf{x} = A^{\mathsf T}\mathbf{b}\)。关键问题：条件数被**平方**。
   315|> **定理**：若 \(A\) 满列秩（\(\text{rank}(A) = n\)），则 \(\kappa_2(A^{\mathsf T}A) = \kappa_2(A)^2\)。
   316|>
   317|> **证明**：\(\kappa_2(A^{\mathsf T}A) = \frac{\sigma_{\max}(A^{\mathsf T}A)}{\sigma_{\min}(A^{\mathsf T}A)} = \frac{\sigma_{\max}(A)^2}{\sigma_{\min}(A)^2} = \left(\frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}\right)^2 = \kappa_2(A)^2\)。
   318|>
   319|> **数值后果**：
   320|> - 若 \(\kappa_2(A) = 10^3\)（在特征尺度差异大的数据中常见 → 各列方差差距大），则 \(\kappa_2(A^{\mathsf T}A) = 10^6\)
   321|> - 双精度浮点有约 16 位有效数字。\(10^6\) 的条件数意味着解会损失 \(\log_{10}(10^6) \approx 6\) 位有效数字 → 解可能完全不可靠
   322|> - **QR 方法**：解 \(R\mathbf{x} = Q^{\mathsf T}\mathbf{b}\)，\(R\) 是上三角矩阵，\(\kappa_2(R) = \kappa_2(A)\)（因为 \(Q\) 是正交矩阵，保持奇异值）。没有平方效应。
   323|> - **SVD 方法**：可以**截断**过小的奇异值（设置阈值 \(\sigma_k/\sigma_1 < \varepsilon\)），显式丢弃噪声子空间，得到 Moore-Penrose 伪逆的截断版——数值上最稳健。
   324|> `np.linalg.lstsq` 底层使用基于 QR（小矩阵）或 SVD（大矩阵/秩亏）的方法，从不直接构造 \(A^{\mathsf T}A\)。
   325|
   326|---
   327|
   328|**B2.** 深度网络前向传播 $\mathbf{h}_{l+1} = \sigma(W_l \mathbf{h}_l)$。假设所有 $W_l$ 的特征值的绝对值均 $>1$。分析反向传播梯度的行为。当特征值均 $<1$ 时又如何？
   329|
   330|> **标准答案：**
   331|> 忽略激活函数的非线性（或考虑 \(\sigma'\) 的尺度），反向传播的梯度为：
   332|> \\[\frac{\partial L}{\partial \mathbf{h}_l} = \frac{\partial L}{\partial \mathbf{h}_L} \cdot \prod_{k=l}^{L-1} \left( \text{diag}(\sigma'(W_k\mathbf{h}_k)) \cdot W_k^{\mathsf T} \right)\\]
   333|> 简化分析：假设 \(\sigma' \approx 1\)（如 ReLU 激活区域的导数为 1），梯度近似为 \(\frac{\partial L}{\partial \mathbf{h}_L} \cdot \prod_{k=l}^{L-1} W_k^{\mathsf T}\)。
   334|>
   335|> **情况 1**：\(|\lambda(W_k)| > 1\)。以 \(W_k\) 的最大特征值 \(\lambda_{\max}\) 为例（\(|\lambda_{\max}| > 1\)）。连乘 \(L-l\) 个这样的矩阵 → 梯度范数随层数指数增长：
   336|> \\[\left\|\frac{\partial L}{\partial \mathbf{h}_l}\right\| \approx |\lambda_{\max}|^{L-l} \cdot \left\|\frac{\partial L}{\partial \mathbf{h}_L}\right\|\\]
   337|> 当 \(L=100\)，\(|\lambda_{\max}| = 1.5\) 时，放大倍数 \(\approx 1.5^{100} \approx 4 \times 10^{17}\) → **梯度爆炸**。
   338|>
   339|> **情况 2**：\(|\lambda(W_k)| < 1\)。梯度范数随层数指数衰减：
   340|> \\[\left\|\frac{\partial L}{\partial \mathbf{h}_l}\right\| \approx |\lambda_{\max}|^{L-l} \cdot \left\|\frac{\partial L}{\partial \mathbf{h}_L}\right\|\\]
   341|> 当 \(|\lambda_{\max}| = 0.5\)，\(L=100\)：\(0.5^{100} \approx 7.9 \times 10^{-31}\) → **梯度消失**。
   342|> **Xavier/He 初始化的动机**：控制 \(W\) 各元素的方差，使 \(W\) 的奇异值集中在 \(1\) 附近 → \(|\lambda| \approx 1\) → 前向和反向传播的信号/梯度尺度保持恒定。\(\kappa(W) \approx 1\) 是最理想的状态。
   343|
   344|---
   345|
   346|**B3.** 假设某优化问题的 Hessian 特征值为 $\{100,\; 10,\; 0.1\}$。分析：
   347|
   348|(1) 普通 SGD 的收敛行为（需要大约多少步？）；
   349|(2) Adam 的表现为什么更好？
   350|
   351|> **标准答案：**
   352|>
   353|> **(1) Hessian 的条件数**：\(\kappa = \lambda_{\max} / \lambda_{\min} = 100 / 0.1 = 1000\)。
   354|> 在局部二次近似下，梯度下降沿最大特征值方向的收敛速率为 \((1 - \eta\lambda_{\max})\)，沿最小特征值方向为 \((1 - \eta\lambda_{\min})\)。
   355|> 学习率受最大特征值限制：\(\eta < 2/\lambda_{\max} = 0.02\) 以避免震荡。取 \(\eta = 0.01\)。
   356|> 沿 \(\lambda_{\min}=0.1\) 的方向，每步衰减因子为 \(1 - 0.01 \times 0.1 = 0.999\)。要使误差减少到 \(1/e\)，需要步数 \(\approx 1/(\eta\lambda_{\min}) = 1/(0.01 \times 0.1) = 1000\) 步。沿 \(\lambda_{\max}=100\) 的方向，只需 \(1/(0.01 \times 100) = 1\) 步。
   357|> 结论：SGD 在 \(\lambda_{\max}\) 方向飞速收敛（甚至震荡），在 \(\lambda_{\min}\) 方向几乎寸步难行。总共需要 **数百到上千步**才能使所有方向都充分下降。这就是 ill-conditioned 优化问题的核心困难。
   358|>
   359|> **(2)** Adam 维护梯度平方的指数移动平均 \(v_t \approx \mathbb{E}[g^2]\)。在局部二次近似下，\(g \approx H(\theta - \theta^*)\)。如果 Hessian 近似对角（或各参数独立），\(v_t\) 每个分量近似相应 Hessian 对角元的尺度。
   360|> Adam 更新 \(= \eta \cdot m_t / (\sqrt{v_t} + \varepsilon)\)。除以 \(\sqrt{v_t}\) 等价于对角预处理：
   361|> - 大梯度方向（\(\lambda = 100\)）→ \(v_t\) 大 → 有效步长 \(\eta/\sqrt{100} = \eta/10\)（小步，避免震荡）
   362|> - 小梯度方向（\(\lambda = 0.1\)）→ \(v_t\) 小 → 有效步长 \(\eta/\sqrt{0.1} \approx 3.16\eta\)（大步，加速前进）
   363|> 三个方向的有效步长被拉平到相近尺度，优化景观从「瘦长碗」变为「接近球形」→ 条件数被显著改善 → 各方向同步收敛。
   364|
   365|---
   366|
   367|**B4.** 考虑 $A = \begin{bmatrix}1&1\\1&1.0001\end{bmatrix}$，$\mathbf{b} = (2, 2.0001)^{\mathsf T}$（精确解 $\mathbf{x} = (1,1)^{\mathsf T}$）。
   368|
   369|(1) 估算 $A$ 的条件数。
   370|(2) 若 $\mathbf{b}$ 受微小扰动变为 $\tilde{\mathbf{b}} = (2, 2)^{\mathsf T}$，求解 $\tilde{\mathbf{x}}$，观察误差。
   371|(3) 这和深度学习中 ill-conditioned Hessian 引起的训练不稳定有什么关系？
   372|
   373|> **标准答案：**
   374|>
   375|> **(1)** \(A\) 的两列几乎平行——第 2 列仅比第 1 列多 \(10^{-4}\)。\(A\) 近乎奇异。
   376|> 特征值：\(\det(A - \lambda I) = (1-\lambda)(1.0001-\lambda) - 1 = \lambda^2 - 2.0001\lambda + 0.0001 = 0\)。
   377|> \(\lambda_{1,2} = \frac{2.0001 \pm \sqrt{4.00040001 - 0.0004}}{2} \approx \frac{2.0001 \pm 2.0000000025}{2}\)
   378|> \(\lambda_1 \approx 2.00005\)，\(\lambda_2 \approx 0.00005\)。
   379|> 条件数 \(\kappa = \lambda_{\max}/\lambda_{\min} \approx 2.00005/0.00005 \approx 4 \times 10^4\)。极大！
   380|>
   381|> **(2)** 解 \(A\tilde{\mathbf{x}} = (2, 2)^{\mathsf T}\)：
   382|> \\[\begin{cases} \tilde{x}_1 + \tilde{x}_2 = 2 \\ \tilde{x}_1 + 1.0001\tilde{x}_2 = 2 \end{cases}\\]
   383|> 两式相减：\(0.0001\tilde{x}_2 = 0 \Rightarrow \tilde{x}_2 = 0\)。代入：\(\tilde{x}_1 = 2\)。
   384|> \(\tilde{\mathbf{x}} = (2, 0)^{\mathsf T}\)。精确解是 \(\mathbf{x} = (1, 1)^{\mathsf T}\)。
   385|>
   386|> **误差分析**：\(\mathbf{b}\) 的相对变化 \(\approx \frac{\|(2,2)-(2,2.0001)\|}{\|(2,2.0001)\|} \approx \frac{0.0001}{2.83} \approx 3.5 \times 10^{-5}\)（极小）。但 \(\mathbf{x}\) 的相对变化 \(\approx \frac{\|(2,0)-(1,1)\|}{\|(1,1)\|} \approx \frac{\sqrt{2}}{\sqrt{2}} = 1\)（100%！）。**输入误差被条件数放大了约 \(3 \times 10^4\) 倍**，完全吻合 \(\kappa\) 的估计。
   387|>
   388|> **(3)** 深度学习中，若 Hessian 的条件数很大，mini-batch SGD 的梯度估计中的统计噪声（类比 \(\mathbf{b}\) 的微小扰动）被 ill-conditioned Hessian 放大为权重更新的巨大方差。表现为：training loss 剧烈震荡、某些参数方向几乎不更新、收敛极慢。**这正是 Adam / BatchNorm / 残差连接共同试图解决的问题**——改善优化景观的条件数。
   389|
   390|---
   391|
   392|**B5.** 在线性回归 $y = X\beta + \varepsilon$ 中，最小二乘估计满足 $X^{\mathsf T}\mathbf{e} = \mathbf{0}$，其中 $\mathbf{e} = y - X\hat{\beta}$。
   393|
   394|(1) 用几何语言解释 $X^{\mathsf T}\mathbf{e} = \mathbf{0}$。
   395|(2) 若 $X$ 有一列全 $1$（截距项），推导残差的统计性质。
   396|(3) 证明 $\|y\|^2 = \|X\hat{\beta}\|^2 + \|\mathbf{e}\|^2$。
   397|
   398|> **标准答案：**
   399|>
   400|> **(1)** \(X^{\mathsf T}\mathbf{e} = \mathbf{0}\) 展开：\(\mathbf{x}_j^{\mathsf T}\mathbf{e} = 0\) 对 \(X\) 的每一列 \(j\) 成立。这意味着：
   401|> - 残差向量 \(\mathbf{e}\) 与 \(X\) 的**每一列**正交
   402|> - \(\mathbf{e}\) 正交于 \(X\) 的**整个列空间** \(C(X)\)
   403|> - \(X\hat{\beta}\) 是 \(y\) 在 \(C(X)\) 上的正交投影
   404|> - \(\mathbf{e}\) 是投影的残差，垂直于投影面
   405|>
   406|> **(2)** 若 \(\mathbf{x}_1 = (1,1,\dots,1)^{\mathsf T}\)，则 \(\mathbf{1}^{\mathsf T}\mathbf{e} = \sum_{i=1}^n e_i = 0\) → **残差之和为零** → **残差均值为零**（\(\bar{e} = 0\)）。这是线性回归中 \(\sum_i (y_i - \hat{y}_i) = 0\) 的数学根源。
   407|>
   408|> **(3)**
   409|> \\[\|y\|^2 = \|X\hat{\beta} + \mathbf{e}\|^2 = \|X\hat{\beta}\|^2 + \|\mathbf{e}\|^2 + 2(X\hat{\beta})^{\mathsf T}\mathbf{e}\\]
   410|> 由 \((X\hat{\beta})^{\mathsf T}\mathbf{e} = \hat{\beta}^{\mathsf T}(X^{\mathsf T}\mathbf{e}) = \hat{\beta}^{\mathsf T} \cdot \mathbf{0} = 0\)。交叉项消失！
   411|> 所以 \(\|y\|^2 = \|X\hat{\beta}\|^2 + \|\mathbf{e}\|^2\)。这就是方差分析（ANOVA）的核心恒等式：
   412|> \\[\text{SST} = \text{SSR} + \text{SSE}\\]
   413|> 总平方和 = 回归平方和 + 残差平方和。几何上，这是 \(\mathbb{R}^n\) 中直角三角形的勾股定理——\(y\) 被分解为列空间分量 \(X\hat{\beta}\) 和正交补分量 \(\mathbf{e}\)。
   414|
   415|---
   416|
   417|[← 返回教程](elimination-and-subspaces.md)　　　[下一章习题 →](determinant-eigen-svd-exercises.md)