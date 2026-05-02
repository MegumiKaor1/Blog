# 习题与思考：从向量到秩

A 档巩固基础，B 档培养 ML 直觉。每题附完整标准答案。

---

## 一、从向量到秩（§1–§6）

### A 档

**A1.** 给定向量组：$\mathbf{v}_1=(1,2,1)^{\mathsf T}$，$\mathbf{v}_2=(2,4,2)^{\mathsf T}$，$\mathbf{v}_3=(1,3,2)^{\mathsf T}$，$\mathbf{v}_4=(0,1,1)^{\mathsf T}$。求：
- (a) 该向量组的秩；
- (b) 一个极大无关组；
- (c) 将其余向量用该极大无关组线性表出。

> **标准答案：**
> **(a)** 将四个向量按列排成 \(3 \times 4\) 矩阵并做行化简：
> \[A = \begin{bmatrix} 1 & 2 & 1 & 0 \\ 2 & 4 & 3 & 1 \\ 1 & 2 & 2 & 1 \end{bmatrix}\]
> \(R_2 - 2R_1\)：\((0, 0, 1, 1)\)；\(R_3 - R_1\)：\((0, 0, 1, 1)\)；\(R_3 - R_2\)：\((0, 0, 0, 0)\)。
> 行阶梯形有 \(2\) 个非零行 → **秩 \(=2\)**。
> **(b)** 主元在第 1 列和第 3 列 → 极大无关组可取 \(\{\mathbf{v}_1, \mathbf{v}_3\}\)。也可以取 \(\{\mathbf{v}_1, \mathbf{v}_4\}\) 等其他组合——极大无关组不唯一，但向量个数恒为 \(2\)。
> **(c)** \(\mathbf{v}_2 = 2\mathbf{v}_1 + 0\cdot\mathbf{v}_3\)（直接倍数关系）。\(\mathbf{v}_4\)：设 \(\mathbf{v}_4 = c_1\mathbf{v}_1 + c_2\mathbf{v}_3\)，即 \(c_1(1,2,1) + c_2(1,3,2) = (0,1,1)\)。比较分量：
> \[c_1 + c_2 = 0,\quad 2c_1 + 3c_2 = 1,\quad c_1 + 2c_2 = 1\]
> 由第一个方程 \(c_2 = -c_1\)。代入第二个：\(2c_1 - 3c_1 = 1 \Rightarrow -c_1 = 1 \Rightarrow c_1 = -1,\; c_2 = 1\)。验证第三个：\((-1) + 2(1) = 1\) ✓。所以 \(\mathbf{v}_4 = -\mathbf{v}_1 + \mathbf{v}_3\)。

---

**A2.** 判断以下命题的真伪，并简述理由：

(1) 若向量组线性相关，则其中任一向量都可由其余向量线性表出。

(2) $n$ 维向量空间中，任意 $n$ 个线性无关的向量构成一组基。

(3) 秩为 $r$ 的矩阵，其任意 $r$ 个线性无关的列向量构成列空间的一组基。

> **标准答案：**
>
> **(1) 错。** 线性相关只保证**存在**某一个向量可由其余向量线性表出，不保证**每一个**都如此。反例：向量组 \(\{(1,0), (2,0), (0,1)\}\) 是线性相关的（3 个二维向量必然相关），但 \((0,1)\) 不能由前两个表出（前两个只能生成 \(x\) 轴上的向量）。
>
> **(2) 对。** 这是维数的等价定义：\(n\) 维向量空间中，任意 \(n\) 个线性无关的向量必然张成整个空间（否则空间维数 \(>n\)，矛盾），因此构成一组基。
>
> **(3) 对。** \(r\) 个线性无关的列向量都在列空间中。列空间的维数是 \(r\)，而 \(r\) 个线性无关的向量在 \(r\) 维子空间中必为基。

---

**A3.** 求矩阵 $A = \begin{bmatrix}1&2&0&3\\0&1&1&2\\1&4&2&7\end{bmatrix}$ 的秩。

> **标准答案：**
> 初等行变换：
> ① \(R_3 - R_1\)：\((0, 2, 2, 4)\)。矩阵变为：
> \[\begin{bmatrix} 1 & 2 & 0 & 3 \\ 0 & 1 & 1 & 2 \\ 0 & 2 & 2 & 4 \end{bmatrix}\]
> ② \(R_3 - 2R_2\)：\((0, 0, 0, 0)\)。矩阵变为：
> \[\begin{bmatrix} 1 & 2 & 0 & 3 \\ 0 & 1 & 1 & 2 \\ 0 & 0 & 0 & 0 \end{bmatrix}\]
> 行阶梯形有 \(2\) 个非零行 → **秩 \(=2\)**。
> 验证：第 1 列和第 2 列不共线（\(1/0 \neq 2/1\)），第 3 列 \(= 2\times\) 第 2 列 \(-\) 第 1 列（\(2(2,1,4)-(1,0,1) = (3,2,7)\) ✓），确实只有 2 个独立列。

---

**A4.** 计算向量 $\mathbf{x}=(3,-4,0)^{\mathsf T}$ 的 $L_1$、$L_2$、$L_\infty$ 范数。并说明三种范数中，哪一种对该向量的「惩罚」最重，为什么？

> **标准答案：**
> - \(L_1\)：\(\|\mathbf{x}\|_1 = |3| + |-4| + |0| = 3 + 4 + 0 = 7\)
> - \(L_2\)：\(\|\mathbf{x}\|_2 = \sqrt{3^2 + (-4)^2 + 0^2} = \sqrt{9+16} = \sqrt{25} = 5\)
> - \(L_\infty\)：\(\|\mathbf{x}\|_\infty = \max(|3|, |-4|, |0|) = 4\)
> \(L_1\) 最大（\(7\)），因为：
> - \(L_1\) 对每个分量的绝对值**求和**——零分量不稀释总惩罚；
> - \(L_2\) 的平方会缩小各分量的贡献（\(9+16=25\)，开方得 \(5\)，比 \(L_1\) 的 \(7\) 小）；
> - \(L_\infty\) 只关心最大的那个分量（\(4\)），完全忽略其他。
> 在 ML 中，\(L_1\) 正则化（LASSO）产生稀疏解，正是因为 \(L_1\) 对每个非零分量「全价」惩罚，不像 \(L_2\) 那样平方后有递减效应。

---

**A5.** 设 $\mathbf{v}_1=(1,2,1)^{\mathsf T}$，$\mathbf{v}_2=(2,4,2)^{\mathsf T}$，$\mathbf{v}_3=(0,1,0)^{\mathsf T}$。

(1) $\mathbf{v}_1$ 和 $\mathbf{v}_2$ 的所有线性组合构成什么几何对象？
(2) $\mathbf{v}_1$ 和 $\mathbf{v}_3$ 的所有线性组合构成什么几何对象？
(3) 三个向量一起，能张成整个 $\mathbb{R}^3$ 吗？为什么？

> **标准答案：**
>
> **(1)** \(\mathbf{v}_2 = 2\mathbf{v}_1\)，两者共线。所有线性组合：
> \[c_1\mathbf{v}_1 + c_2\mathbf{v}_2 = c_1\mathbf{v}_1 + 2c_2\mathbf{v}_1 = (c_1+2c_2)\mathbf{v}_1\]
> 这是一个标量乘 \(\mathbf{v}_1\) → 构成过原点沿 \(\mathbf{v}_1\) 方向的一条**直线**。
>
> **(2)** \(\mathbf{v}_1 = (1,2,1)\) 和 \(\mathbf{v}_3 = (0,1,0)\) 不共线（不存在 \(c\) 使 \((1,2,1) = c(0,1,0)\)）。两个不共线的三维向量张成一个过原点的**平面**——法向量可用叉积求得：\(\mathbf{v}_1 \times \mathbf{v}_3 = (-1, 0, 1)^{\mathsf T}\)。
>
> **(3)** 不能。\(\mathbf{v}_2\) 没有贡献新方向（它和 \(\mathbf{v}_1\) 共线），所以三个向量实际只张成一个二维平面。要张成 \(\mathbb{R}^3\) 需要 \(3\) 个线性无关的向量。

---

**A6.** 判断以下向量组是否线性无关。若相关，找出一个非平凡的线性组合等于零向量。

(1) $\{(1,2,3)^{\mathsf T},\; (4,5,6)^{\mathsf T},\; (7,8,9)^{\mathsf T}\}$
(2) $\{(1,1,0)^{\mathsf T},\; (0,1,1)^{\mathsf T},\; (1,0,1)^{\mathsf T}\}$

> **标准答案：**
>
> **(1)** 将三列排成矩阵并做行化简：
> \[\begin{bmatrix} 1 & 4 & 7 \\ 2 & 5 & 8 \\ 3 & 6 & 9 \end{bmatrix}\]
> \(R_2 - 2R_1\)：\((0, -3, -6)\)；\(R_3 - 3R_1\)：\((0, -6, -12)\)。\(R_3 - 2R_2\)：\((0, 0, 0)\)。存在零行 → **线性相关**。
> 找关系：观察发现 \(\mathbf{v}_3 = 2\mathbf{v}_2 - \mathbf{v}_1\)。验证：\(2(4,5,6) - (1,2,3) = (8-1, 10-2, 12-3) = (7,8,9)\) ✓。因此 \(-\mathbf{v}_1 + 2\mathbf{v}_2 - \mathbf{v}_3 = \mathbf{0}\)（或等价地 \(\mathbf{v}_1 - 2\mathbf{v}_2 + \mathbf{v}_3 = \mathbf{0}\)）。
>
> **(2)** 矩阵为：
> \[\begin{bmatrix} 1 & 0 & 1 \\ 1 & 1 & 0 \\ 0 & 1 & 1 \end{bmatrix}\]
> 计算行列式：按第一行展开：
> \[\det = 1 \cdot \det\begin{bmatrix}1 & 0 \\ 1 & 1\end{bmatrix} - 0 + 1 \cdot \det\begin{bmatrix}1 & 1 \\ 0 & 1\end{bmatrix}\]
> \[= 1(1-0) + 1(1-0) = 2 \neq 0\]
> 行列式非零 → **线性无关**。三个向量构成 \(\mathbb{R}^3\) 的一组基。

---

**A7.** 设矩阵 $A = \begin{bmatrix}1&2&0&1\\0&0&1&2\\2&4&1&4\end{bmatrix}$。

(1) 求 $A$ 的秩。
(2) 求 $A$ 的列空间的一组基。
(3) 列空间的维数是多少？它和 $\mathbb{R}^3$ 的关系是什么？

> **标准答案：**
>
> **(1)** 行化简：
> ① \(R_3 - 2R_1\)：\((0, 0, 1, 2)\)
> ② \(R_3 - R_2\)：\((0, 0, 0, 0)\)（\(R_2\) 已经是 \((0,0,1,2)\)）
> 行阶梯形：
> \[\begin{bmatrix} 1 & 2 & 0 & 1 \\ 0 & 0 & 1 & 2 \\ 0 & 0 & 0 & 0 \end{bmatrix}\]
> 非零行数 \(=2\) → **秩 \(=2\)**。
>
> **(2)** 主元在第 1 列和第 3 列 → 列空间的一组基为 \(A\) 的第 1 列和第 3 列：
> \[\left\{\begin{bmatrix} 1 \\ 0 \\ 2 \end{bmatrix},\; \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix}\right\}\]
>
> **(3)** \(\dim C(A) = 2\)。它是 \(\mathbb{R}^3\) 中的一个二维平面（过原点），不是整个 \(\mathbb{R}^3\)。任何不在此平面上的 \(\mathbf{b} \in \mathbb{R}^3\) 都无法写成 \(A\) 各列的线性组合——即 \(A\mathbf{x} = \mathbf{b}\) 无解。

---

**A8.** 设 $\mathbf{u} = (1, 2, 2)^{\mathsf T}$，$\mathbf{v} = (2, 1, -2)^{\mathsf T}$。

(1) 计算 $\mathbf{u} \cdot \mathbf{v}$。这两个向量的夹角是锐角、直角还是钝角？
(2) 将 $\mathbf{u}$ 归一化为单位向量 $\hat{\mathbf{u}}$。
(3) 验证 $\hat{\mathbf{u}} \cdot \mathbf{v}$ 等于 $\mathbf{v}$ 在 $\mathbf{u}$ 方向上的投影长度（带符号）。
(4) （ML 应用）两个词向量 $\mathbf{e}_1$ 和 $\mathbf{e}_2$ 的余弦相似度为 $0.95$，而 $\mathbf{e}_1$ 和 $\mathbf{e}_3$ 的余弦相似度为 $-0.8$。这意味着什么？

> **标准答案：**
>
> **(1)**
> \[\mathbf{u} \cdot \mathbf{v} = 1 \cdot 2 + 2 \cdot 1 + 2 \cdot (-2) = 2 + 2 - 4 = 0\]
> 内积为 \(0\) → **直角**（两个向量正交）。由 \(\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta = 0\)，且 \(\|\mathbf{u}\|, \|\mathbf{v}\| > 0\)，故 \(\cos\theta = 0 \Rightarrow \theta = 90^\circ\)。
>
> **(2)**
> \[\|\mathbf{u}\| = \sqrt{1^2 + 2^2 + 2^2} = \sqrt{1 + 4 + 4} = \sqrt{9} = 3\]
> \[\hat{\mathbf{u}} = \frac{\mathbf{u}}{\|\mathbf{u}\|} = \left(\frac{1}{3}, \frac{2}{3}, \frac{2}{3}\right)^{\mathsf T}\]
>
> **(3)** \(\mathbf{v}\) 在 \(\mathbf{u}\) 方向上的投影长度（带符号）为：
> \[\text{proj}_{\mathbf{u}}(\mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|} = \frac{0}{3} = 0\]
> 同时 \(\hat{\mathbf{u}} \cdot \mathbf{v} = (\frac{1}{3})(2) + (\frac{2}{3})(1) + (\frac{2}{3})(-2) = \frac{2+2-4}{3} = 0\)。两者一致。投影长度为 \(0\) 说明 \(\mathbf{v}\) 完全垂直于 \(\mathbf{u}\)——没有「沿 \(\mathbf{u}\) 方向的分量」。
>
> **(4)** 余弦相似度 \(=\) 两向量夹角的余弦。\(0.95\) → \(\theta \approx 18^\circ\)，方向几乎一致——两个词在语义上非常接近（如「猫」和「小猫」）。\(-0.8\) → \(\theta \approx 143^\circ\)，方向几乎相反——语义上强烈对立（如「好」和「坏」）。这正是 Attention 中 \(QK^{\mathsf T}\) 的底层逻辑：内积 = 语义相关性分数。

---

**A9.** 设 $A = \begin{bmatrix} 2 & 0 & 1 \\ 0 & 1 & 3 \end{bmatrix}$，$\mathbf{x} = (1, 2, -1)^{\mathsf T}$。

(1) 用**列图景**计算 $A\mathbf{x}$（即 $A$ 各列以 $\mathbf{x}$ 各分量为系数的线性组合）。
(2) 改写成**行图景**：$A\mathbf{x}$ 的每个分量是什么？
(3) 如果 $A$ 的第 3 列加倍（变成 $(2,6)^{\mathsf T}$），$A\mathbf{x}$ 会怎么变？

> **标准答案：**
>
> **(1)** 列图景——将矩阵乘法视为列的线性组合：
> \[A\mathbf{x} = x_1 \cdot \mathbf{a}_1 + x_2 \cdot \mathbf{a}_2 + x_3 \cdot \mathbf{a}_3\]
> \[= 1 \cdot \begin{bmatrix}2\\0\end{bmatrix} + 2 \cdot \begin{bmatrix}0\\1\end{bmatrix} + (-1) \cdot \begin{bmatrix}1\\3\end{bmatrix}\]
> \[= \begin{bmatrix}2\\0\end{bmatrix} + \begin{bmatrix}0\\2\end{bmatrix} + \begin{bmatrix}-1\\-3\end{bmatrix} = \begin{bmatrix}1\\-1\end{bmatrix}\]
>
> **(2)** 行图景——每个输出分量是 \(A\) 的一行与 \(\mathbf{x}\) 的内积：
> \[(A\mathbf{x})_1 = 2 \cdot 1 + 0 \cdot 2 + 1 \cdot (-1) = 2 + 0 - 1 = 1\]
> \[(A\mathbf{x})_2 = 0 \cdot 1 + 1 \cdot 2 + 3 \cdot (-1) = 0 + 2 - 3 = -1\]
> 结果一致：\(\begin{bmatrix}1\\-1\end{bmatrix}\)。
>
> **(3)** 设 \(A'\) 的第 3 列为 \((2,6)^{\mathsf T}\)，其余不变。按列图景：
> \[A'\mathbf{x} = 1 \cdot \begin{bmatrix}2\\0\end{bmatrix} + 2 \cdot \begin{bmatrix}0\\1\end{bmatrix} + (-1) \cdot \begin{bmatrix}2\\6\end{bmatrix}\]
> \[= \begin{bmatrix}2\\0\end{bmatrix} + \begin{bmatrix}0\\2\end{bmatrix} + \begin{bmatrix}-2\\-6\end{bmatrix} = \begin{bmatrix}0\\-4\end{bmatrix}\]
> 第 3 列的贡献从 \((-1, -3)^{\mathsf T}\) 加倍为 \((-2, -6)^{\mathsf T}\)，总结果从 \((1, -1)^{\mathsf T}\) 变为 \((0, -4)^{\mathsf T}\)。列图景的优势：只需改动一个列向量，无需重新计算整个矩阵乘法。

---

**A10.** 矩阵 $A = \begin{bmatrix} 1 & 0 & 2 \\ 0 & 1 & 3 \\ 0 & 0 & 0 \end{bmatrix}$ 是一个 $3 \times 3$ 矩阵。

(1) $A$ 对标准基向量 $\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3$ 分别做了什么？
(2) $A$ 把整个 $\mathbb{R}^3$ 映射到了哪里？（用几何语言描述）
(3) $A$ 的零空间是什么？给出基础解系。

> **标准答案：**
>
> **(1)** 矩阵的列就是标准基向量的像（这是列图景的核心结论）：
> \[A\mathbf{e}_1 = A\begin{bmatrix}1\\0\\0\end{bmatrix} = \begin{bmatrix}1\\0\\0\end{bmatrix} \quad\text{（第 1 列 = }\mathbf{e}_1\text{，不变）}\]
> \[A\mathbf{e}_2 = \begin{bmatrix}0\\1\\0\end{bmatrix} = \mathbf{e}_2 \quad\text{（第 2 列，不变）}\]
> \[A\mathbf{e}_3 = \begin{bmatrix}2\\3\\0\end{bmatrix} = 2\mathbf{e}_1 + 3\mathbf{e}_2 \quad\text{（第 3 列，投影到 }xy\text{ 平面）}\]
> 所以 \(A\) 保留 \(x\) 和 \(y\) 方向不变，把 \(z\) 方向上的向量「拍平」到 \(xy\) 平面上。
>
> **(2)** 对于任意 \((x, y, z)^{\mathsf T}\)：
> \[A\begin{bmatrix}x\\y\\z\end{bmatrix} = \begin{bmatrix}x+2z\\y+3z\\0\end{bmatrix}\]
> 第三分量始终为 \(0\) → 所有输出都落在 \(xy\) 平面（\(z=0\)）上。\(A\) 把 \(\mathbb{R}^3\) 降维映射到一个二维子空间（\(xy\) 平面），秩 \(=2\)。
>
> **(3)** 零空间 = 所有满足 \(A\mathbf{x} = \mathbf{0}\) 的 \(\mathbf{x}\)：
> \[\begin{bmatrix}x+2z\\y+3z\\0\end{bmatrix} = \begin{bmatrix}0\\0\\0\end{bmatrix} \Rightarrow \begin{cases} x + 2z = 0 \\ y + 3z = 0 \end{cases}\]
> 自由变量 \(z = t\)，则 \(x = -2t\)，\(y = -3t\)。基础解系：
> \[\mathbf{x} = t \begin{bmatrix}-2\\-3\\1\end{bmatrix}\]
> 零空间维数 \(=1 = n - r = 3 - 2\) ✓。几何上，它是垂直于 \(xy\) 平面且指向 \((-2,-3,1)\) 方向的一条直线。
> 验证：\(A(-2,-3,1)^{\mathsf T} = (-2+2, -3+3, 0)^{\mathsf T} = (0,0,0)^{\mathsf T}\) ✓。

---

**A11.** 设 $\mathbf{a} = (3, 0, 4)^{\mathsf T}$，$\mathbf{b} = (0, 5, 0)^{\mathsf T}$。计算：

(1) $\mathbf{a} + \mathbf{b}$ 和 $2\mathbf{a} - 3\mathbf{b}$；
(2) $\|\mathbf{a}\|$、$\|\mathbf{b}\|$、$\mathbf{a} \cdot \mathbf{b}$；
(3) $\mathbf{a}$ 和 $\mathbf{b}$ 的夹角 $\theta$（精确到度）。

> **标准答案：**
>
> **(1)**
> \[\mathbf{a} + \mathbf{b} = (3+0,\; 0+5,\; 4+0)^{\mathsf T} = (3, 5, 4)^{\mathsf T}\]
> \[2\mathbf{a} - 3\mathbf{b} = (6-0,\; 0-15,\; 8-0)^{\mathsf T} = (6, -15, 8)^{\mathsf T}\]
>
> **(2)**
> \[\|\mathbf{a}\| = \sqrt{3^2 + 0^2 + 4^2} = \sqrt{9 + 0 + 16} = \sqrt{25} = 5\]
> \[\|\mathbf{b}\| = \sqrt{0^2 + 5^2 + 0^2} = 5\]
> \[\mathbf{a} \cdot \mathbf{b} = 3 \cdot 0 + 0 \cdot 5 + 4 \cdot 0 = 0\]
>
> **(3)** 由 \(\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta\)：
> \[0 = 5 \cdot 5 \cdot \cos\theta \Rightarrow \cos\theta = 0 \Rightarrow \theta = 90^\circ\]
> 两个向量正交——这在 \(\mathbb{R}^3\) 中很自然：\(\mathbf{a}\) 在 \(xz\) 平面内，\(\mathbf{b}\) 沿 \(y\) 轴。

---

**A12.** 向量 $\mathbf{b} = (5, 7)^{\mathsf T}$ 是否在 $\mathbf{v}_1 = (1, 2)^{\mathsf T}$ 和 $\mathbf{v}_2 = (3, 4)^{\mathsf T}$ 的张成空间里？如果是，求出用 $\mathbf{v}_1, \mathbf{v}_2$ 表示 $\mathbf{b}$ 的系数。

> **标准答案：**
> 问题等价于：是否存在 \(c_1, c_2\) 使得 \(c_1\mathbf{v}_1 + c_2\mathbf{v}_2 = \mathbf{b}\)？即解线性方程组：
> \[\begin{bmatrix}1 & 3 \\ 2 & 4\end{bmatrix} \begin{bmatrix}c_1\\c_2\end{bmatrix} = \begin{bmatrix}5\\7\end{bmatrix}\]
> 增广矩阵：
> \[\begin{bmatrix}1 & 3 & | & 5 \\ 2 & 4 & | & 7\end{bmatrix}\]
> \(R_2 - 2R_1\)：\((0, -2, |, -3)\)。
> 回代：\(-2c_2 = -3 \Rightarrow c_2 = \frac{3}{2}\)。\(c_1 + 3 \cdot \frac{3}{2} = 5 \Rightarrow c_1 = 5 - \frac{9}{2} = \frac{1}{2}\)。
> 验证：\(\frac{1}{2}(1,2) + \frac{3}{2}(3,4) = (\frac{1}{2}+\frac{9}{2}, 1+6) = (5,7)\) ✓。
> **\(\mathbf{b}\) 在张成空间中**（因为 \(\mathbf{v}_1\) 和 \(\mathbf{v}_2\) 线性无关，张成整个 \(\mathbb{R}^2\)，所以任意 \(\mathbf{b} \in \mathbb{R}^2\) 都在其中）。

---

**A13.** 用四种方式计算 $C = AB$，其中 $A = \begin{bmatrix}1&2\\3&4\end{bmatrix}$，$B = \begin{bmatrix}5&6\\7&8\end{bmatrix}$。

> **标准答案：**
>
> **(1) 元素公式**（\(C_{ij} = \sum_k A_{ik}B_{kj}\)）：
> \[C_{11} = 1\cdot5 + 2\cdot7 = 5+14 = 19\]
> \[C_{12} = 1\cdot6 + 2\cdot8 = 6+16 = 22\]
> \[C_{21} = 3\cdot5 + 4\cdot7 = 15+28 = 43\]
> \[C_{22} = 3\cdot6 + 4\cdot8 = 18+32 = 50\]
> \[C = \begin{bmatrix}19&22\\43&50\end{bmatrix}\]
>
> **(2) 列组合**（\(C\) 的第 \(j\) 列 = \(A \times (B\) 的第 \(j\) 列））：
> \[C_{:,1} = A\begin{bmatrix}5\\7\end{bmatrix} = \begin{bmatrix}1\cdot5+2\cdot7\\3\cdot5+4\cdot7\end{bmatrix} = \begin{bmatrix}19\\43\end{bmatrix}\]
> \[C_{:,2} = A\begin{bmatrix}6\\8\end{bmatrix} = \begin{bmatrix}1\cdot6+2\cdot8\\3\cdot6+4\cdot8\end{bmatrix} = \begin{bmatrix}22\\50\end{bmatrix}\]
>
> **(3) 行组合**（\(C\) 的第 \(i\) 行 = \((A\) 的第 \(i\) 行）\(\times B\)）：
> \[C_{1,:} = [1,2]B = [1\cdot5+2\cdot7,\; 1\cdot6+2\cdot8] = [19, 22]\]
> \[C_{2,:} = [3,4]B = [3\cdot5+4\cdot7,\; 3\cdot6+4\cdot8] = [43, 50]\]
>
> **(4) 外积和**（\(AB = \sum_k (A\) 的第 \(k\) 列）\(\times (B\) 的第 \(k\) 行））：
> \[A_{:,1}B_{1,:} = \begin{bmatrix}1\\3\end{bmatrix}[5,6] = \begin{bmatrix}5&6\\15&18\end{bmatrix}\]
> \[A_{:,2}B_{2,:} = \begin{bmatrix}2\\4\end{bmatrix}[7,8] = \begin{bmatrix}14&16\\28&32\end{bmatrix}\]
> \[C = \begin{bmatrix}5&6\\15&18\end{bmatrix} + \begin{bmatrix}14&16\\28&32\end{bmatrix} = \begin{bmatrix}19&22\\43&50\end{bmatrix}\]
> 四种方式，同一结果。代码里用元素公式，理论分析用列图景（理解神经网络前向传播）和外积和（理解秩和 SVD）。

---

**A14.** 在 $\mathbb{R}^3$ 中，子空间 $S = \text{span}\{(1,2,0)^{\mathsf T}, (0,1,1)^{\mathsf T}, (1,3,1)^{\mathsf T}\}$。

(1) 判断向量组是否线性无关。若相关，找出一个极大无关组。
(2) $S$ 的维数是多少？

> **标准答案：**
>
> **(1)** 检查 \((1,3,1)\) 是否可被前两个表出。设 \(c_1(1,2,0) + c_2(0,1,1) = (1,3,1)\)：
> \[\begin{cases} c_1 = 1 \\ 2c_1 + c_2 = 3 \\ 0 + c_2 = 1 \end{cases}\]
> \(c_1 = 1\)，\(c_2 = 1\)。验证第二个：\(2(1) + 1 = 3\) ✓。所以 \((1,3,1) = (1,2,0) + (0,1,1)\)，**线性相关**。
> 极大无关组可取 \(\{(1,2,0)^{\mathsf T}, (0,1,1)^{\mathsf T}\}\)（这两个不共线，线性无关）。
>
> **(2)** 极大无关组合 \(2\) 个向量 → **维数 \(=2\)**。几何上，\(S\) 是 \(\mathbb{R}^3\) 中过原点的一个平面。

---

**A15.** 矩阵 $A = \begin{bmatrix} 1 & 2 & 0 \\ 3 & 6 & 0 \\ 0 & 0 & 5 \end{bmatrix}$。

(1) 求 $A$ 的秩。
(2) $A$ 可逆吗？为什么？

> **标准答案：**
>
> **(1)** 行化简：
> \(R_2 - 3R_1\)：\((0, 0, 0)\)。行阶梯形：
> \[\begin{bmatrix} 1 & 2 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 5 \end{bmatrix} \rightarrow \text{交换 } R_2,R_3 \rightarrow \begin{bmatrix} 1 & 2 & 0 \\ 0 & 0 & 5 \\ 0 & 0 & 0 \end{bmatrix}\]
> 非零行 \(=2\)。或者直接看列：第 2 列 \(= 2\times\) 第 1 列，但第 3 列独立于前两列 → 秩 \(=2\)。
>
> **(2)** \(3 \times 3\) 方阵，秩 \(=2 < 3 = n\) → **不可逆**（不满秩 \(\iff\) 不可逆）。等价地，\(\det(A) = 1 \cdot \det\begin{bmatrix}6&0\\0&5\end{bmatrix} - 2 \cdot \det\begin{bmatrix}3&0\\0&5\end{bmatrix} + 0 = 1(30) - 2(15) = 30 - 30 = 0\) → 行列式为 \(0\) → 不可逆。

---

### B 档

**B1.** 一个 $1000 \times 768$ 的数据矩阵 $X$，其前 $20$ 个奇异值的平方和占总 Frobenius 范数平方的 $95\%$，后 $748$ 个奇异值几乎为零。请解释：

(1) 数据的「真实维度」大约是多少？
(2) PCA 降维到多少维比较合适？
(3) 这对模型容量设计有什么启示？

> **标准答案：**
>
> **(1)** 真实维度 \(\approx 20\)。虽然观测维度是 \(768\)，但前 20 个奇异值就捕获了 \(95\%\) 的总能量（Frobenius 范数平方 \(= \sum \sigma_i^2\)），说明数据的有效自由度只有约 \(20\) 个——数据本质上分布在一个低维线性子空间内。
>
> **(2)** 保留 \(20\)–\(30\) 维（捕获 \(95\%\)–\(98\%\) 方差）即可。752 个维度几乎没有信息（奇异值接近零），砍掉后只丢失约 \(5\%\) 的方差。
>
> **(3)** 模型参数量不必正比于名义输入维度 \(768\)——真正需要拟合的自由度远小于 \(768\)。这直接呼应 **LoRA**（Low-Rank Adaptation）：即使全参数矩阵是 \(d \times d\)，有效的微调只需要一个秩 \(r \ll d\) 的子空间（\(r=8\sim64\)），因为问题本身就是低秩的。同理，Embedding 层可以压缩（如从 \(|V| \times 768\) 降到 \(|V| \times 256\)），因为语义信息也是低秩的。

---

**B2.** LASSO（$L_1$ 正则化）在做特征选择时，为什么一组强相关的特征中 LASSO 往往只保留其中一个？请结合 $L_1$ 单位球的几何形状给出解释。

> **标准答案：**
> 回顾 LASSO 优化目标：\(\min_{\mathbf{w}} \mathcal{L}(\mathbf{w}) + \lambda\|\mathbf{w}\|_1\)。
>
> **几何分析**：无约束损失 \(\mathcal{L}(\mathbf{w})\) 的等高线是 \(\mathbb{R}^n\) 中的一族椭球（在二次近似下）。\(L_1\) 约束区域 \(\|\mathbf{w}\|_1 \le t\) 是一个以原点为中心、顶点在坐标轴上的凸多面体（二维是菱形，高维是交叉多面体/cross-polytope）。
> 最优点出现在等高线族首次与约束区域相切的位置。由于多面体的顶点在坐标轴上（例如二维菱形顶点在 \((t,0)\) 和 \((0,t)\)），切点**大概率落在顶点上**而非边上。顶点意味着只有少数坐标为非零，其余恰好为 \(0\)。
> 对于一组强相关的特征（线性相关系数 \(\approx \pm 1\)）：选择其中任意一个都能几乎等价地拟合数据（因为它们在数据矩阵 \(X\) 中提供了几乎相同的信息）。\(L_1\) 的角点效应会随机（由微小的数值差异决定）选中其中一个，将其余的系数压为精确 \(0\)。这就是 LASSO 的「自动特征选择」机制。
> **弹性网络（Elastic Net）** 通过加入 \(L_2\) 项：\(\lambda_1\|\mathbf{w}\|_1 + \lambda_2\|\mathbf{w}\|_2^2\)，使约束区域从「尖角多面体」变成「圆角多面体」，缓解了这种不稳定性——强相关特征组可能被一起保留（系数分担）。

---

**B3.** Frobenius 范数和核范数分别对奇异值为 $\boldsymbol{\sigma}=(3,3,0)$ 的矩阵惩罚多少？对 $\boldsymbol{\sigma}=(5,1,0)$ 呢？据此说明两种范数在结构偏好上的差异。

> **标准答案：**
> - \((3,3,0)\)：
>   - \(\|A\|_F^2 = 3^2 + 3^2 + 0^2 = 18\)，\(\|A\|_F = \sqrt{18} \approx 4.24\)
>   - \(\|A\|_* = 3 + 3 + 0 = 6\)
> - \((5,1,0)\)：
>   - \(\|A\|_F^2 = 25 + 1 + 0 = 26\)，\(\|A\|_F = \sqrt{26} \approx 5.10\)
>   - \(\|A\|_* = 5 + 1 + 0 = 6\)
>
> **关键观察**：两组奇异值的核范数相同（和都是 \(6\)），但 Frobenius 范数对 \((5,1,0)\) 更大（\(26 > 18\)）。
> **结构偏好解释**：核范数 \(= \sum \sigma_i\) 是奇异值的 \(L_1\) 范数——它偏好「能量集中在少数奇异值」，因为 \(3+3 > 5+1\)（零奇异值不算，\(3+3=6\)，\(5+1=6\) 相同，但更大的极端例子如 \((6,0,0)\) vs \((3,3,0)\) 会更明显：\(6 < 6\) 还是相等，但 \((7,0,0)\) 的核范数 \(7 < (4,3,0)\) 的核范数 \(7\)——等等，让我换个例子）。
> 更准确的直觉：核范数正则化在优化中**倾向于产生低秩解**。原因是核范数是秩的凸松弛——秩是奇异值的 \(L_0\)，核范数是奇异值的 \(L_1\)。正如 \(L_1\) 正则化产生稀疏向量一样，\(\ell_1\) 对奇异值的惩罚倾向于将小奇异值推向 \(0\)，从而降低秩。这就是矩阵补全（matrix completion）中用核范数替代秩函数进行凸优化的原因。
> Frobenius 范数则**不产生低秩结构**——它平等地惩罚所有奇异值的能量，更像是对矩阵「总体大小」的约束。

---

**B4.** 考虑矩阵 $A = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$ 和 $B = \begin{bmatrix} 2 & 0 \\ 0 & 0.5 \end{bmatrix}$。

(1) 用几何语言描述 $A$ 对平面 $\mathbb{R}^2$ 做了什么变换。
(2) $B$ 对平面做了什么？
(3) 求 $\det(A)$ 和 $\det(B)$，据此说明面积变化。
(4) （ML 扩展）为什么在深度学习中，我们不希望权重矩阵像 $B$ 那样特征值悬殊？

> **标准答案：**
>
> **(1)** \(A\) 对任意 \((x, y)^{\mathsf T}\)：
> \[A\begin{bmatrix}x\\y\end{bmatrix} = \begin{bmatrix}-y\\x\end{bmatrix}\]
> 这是**逆时针旋转 \(90^\circ\)**。验证：\((1,0) \to (0,1)\)，\((0,1) \to (-1,0)\)。它不改变任何向量的长度（\(\|A\mathbf{x}\| = \|\mathbf{x}\|\)）且保持面积不变（旋转不改变平行四边形面积）。
>
> **(2)** \(B\) 对任意 \((x, y)^{\mathsf T}\)：
> \[B\begin{bmatrix}x\\y\end{bmatrix} = \begin{bmatrix}2x\\0.5y\end{bmatrix}\]
> 把 \(x\) 轴方向拉伸 \(2\) 倍，\(y\) 轴方向压缩到 \(0.5\) 倍。一个单位圆变成以 \((2,0)\) 和 \((0,0.5)\) 为半轴的椭圆。
>
> **(3)**
> \[\det(A) = 0 \cdot 0 - (-1) \cdot 1 = 1\]
> \[\det(B) = 2 \cdot 0.5 - 0 \cdot 0 = 1\]
> 有趣的是两者行列式都是 \(1\)——都保面积。但 \(A\) 也保长度（正交矩阵），而 \(B\) 在一个方向上极端拉伸、另一个方向上极端压缩。
>
> **(4)** \(B\) 的特征值是 \(2\) 和 \(0.5\)，条件数 \(\kappa = 2/0.5 = 4\)。虽然还不算极端，但如果多层权重矩阵都像这样特征值悬殊，连乘 \(L\) 层后：
> - 前向传播：最大方向信号被放大 \(2^L\) 倍（指数爆炸），最小方向被压缩到 \(0.5^L\)（指数消失）
> - 反向传播：梯度经历相反的缩放——同样爆炸或消失
> Xavier/He 初始化的目标就是让每层权重矩阵的奇异值集中在 \(1\) 附近，消除这种「信号高速公路 + 信号死胡同」的并存局面。

---

**B5.** 矩阵的**谱范数**（$\|A\|_2$）和**Frobenius 范数**（$\|A\|_F$）都衡量矩阵的「大小」，但角度不同。

(1) 对 $A = \begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix}$，计算 $\|A\|_2$ 和 $\|A\|_F$。
(2) 对 $B = \begin{bmatrix} 2 & 2 \\ 0 & 0 \end{bmatrix}$（秩-1），同样计算。
(3) 在 GAN 中，为什么谱归一化（Spectral Normalization）用谱范数而不使用 Frobenius 范数？

> **标准答案：**
>
> **(1)** \(A\) 是对角矩阵，奇异值就是对角元 \(3, 1\)：
> - \(\|A\|_2 = \sigma_{\max} = 3\)
> - \(\|A\|_F = \sqrt{3^2 + 1^2 + 0^2 + 0^2} = \sqrt{10} \approx 3.16\)
>
> **(2)** \(B\) 的奇异值：\(B^{\mathsf T}B = \begin{bmatrix}4&4\\4&4\end{bmatrix}\)。特征方程 \(\det(B^{\mathsf T}B - \lambda I) = (4-\lambda)^2 - 16 = \lambda^2 - 8\lambda = 0\) → \(\lambda_1=8\)，\(\lambda_2=0\)。奇异值 \(\sigma_1 = \sqrt{8} \approx 2.83\)，\(\sigma_2 = 0\)。
> - \(\|B\|_2 = \sigma_1 = \sqrt{8} \approx 2.83\)
> - \(\|B\|_F = \sqrt{2^2+2^2+0^2+0^2} = \sqrt{8} \approx 2.83\)
> 对于秩-1 矩阵，\(\|B\|_2 = \|B\|_F\)（所有非零奇异值只有一个）。
>
> **(3)**
> 谱范数 \(=\) 最大奇异值 \(=\) 矩阵作为线性变换的**最大拉伸因子** \(=\) \(\displaystyle\max_{\|\mathbf{x}\|=1} \|A\mathbf{x}\| =\) **Lipschitz 常数**。
> GAN 中判别器 \(D\) 需要满足 1-Lipschitz 条件（对输入的微小变化，输出变化不超过输入的倍数），否则梯度惩罚无效、训练崩溃。谱归一化强制每层权重矩阵的 \(\|W\|_2 \le 1\)，从而全局 Lipschitz 常数有界化。
> Frobenius 范数不能做这件事——它衡量的是「总能量」，而不直接约束**最大**拉伸因子。一个 Frobenius 范数很小的矩阵仍然可以有一个方向拉伸极大（同时其他方向压缩来保持 Frobenius 小）——这正是 Lipschitz 约束要阻止的。

---

**B6.** 证明：对任意向量 $\mathbf{x} \in \mathbb{R}^n$，$\|\mathbf{x}\|_\infty \le \|\mathbf{x}\|_2 \le \|\mathbf{x}\|_1 \le \sqrt{n}\|\mathbf{x}\|_2$。并说明各等号何时成立。

> **标准答案：**
> **(a)** \(\|\mathbf{x}\|_\infty \le \|\mathbf{x}\|_2\)：
> 设 \(|x_k| = \max_i |x_i| = \|\mathbf{x}\|_\infty\)。则：
> \[\|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2} \ge \sqrt{x_k^2} = |x_k| = \|\mathbf{x}\|_\infty\]
> 等号成立当且仅当只有一个非零分量（其他全为 \(0\)）。
> **(b)** \(\|\mathbf{x}\|_2 \le \|\mathbf{x}\|_1\)：
> \(\|\mathbf{x}\|_1^2 = (\sum_i |x_i|)^2 = \sum_i x_i^2 + \sum_{i \neq j} |x_i||x_j| \ge \sum_i x_i^2 = \|\mathbf{x}\|_2^2\)。开方即得。
> 等号成立当且仅当只有一个非零分量（交叉项全为 \(0\)）。
> **(c)** \(\|\mathbf{x}\|_1 \le \sqrt{n}\|\mathbf{x}\|_2\)：
> 由 Cauchy-Schwarz 不等式：
> \[\|\mathbf{x}\|_1 = \sum_{i=1}^n 1 \cdot |x_i| \le \sqrt{\sum_{i=1}^n 1^2} \cdot \sqrt{\sum_{i=1}^n |x_i|^2} = \sqrt{n} \cdot \|\mathbf{x}\|_2\]
> 等号成立当且仅当所有 \(|x_i|\) 相等（如 \(\mathbf{x} = (1,1,\ldots,1)^{\mathsf T}\)）。
>
> **综合**：\(\|\mathbf{x}\|_\infty \le \|\mathbf{x}\|_2 \le \|\mathbf{x}\|_1 \le \sqrt{n}\|\mathbf{x}\|_2\)。在有限维空间中，所有范数等价——这就是这个不等式链的含义。

---

**B7.** 设 $A$ 是 $m \times n$ 矩阵，$B$ 是 $n \times p$ 矩阵。证明：

(1) $\text{rank}(AB) \le \min(\text{rank}(A), \text{rank}(B))$。
(2) 若 $B$ 是可逆方阵（$n = p$），则 $\text{rank}(AB) = \text{rank}(A)$。

> **标准答案：**
>
> **(1)** 两个方向分开证：
> **\(\text{rank}(AB) \le \text{rank}(A)\)**：\(AB\) 的每一列都是 \(A\) 各列的线性组合（\(AB\) 的第 \(j\) 列 \(= A \cdot (B\) 的第 \(j\) 列\()\)）。所以 \(C(AB) \subseteq C(A)\)，进而 \(\dim C(AB) \le \dim C(A)\)，即 \(\text{rank}(AB) \le \text{rank}(A)\)。
> **\(\text{rank}(AB) \le \text{rank}(B)\)**：\((AB)^{\mathsf T} = B^{\mathsf T}A^{\mathsf T}\)。由上面的结论：\(\text{rank}(B^{\mathsf T}A^{\mathsf T}) \le \text{rank}(B^{\mathsf T}) = \text{rank}(B)\)。而 \(\text{rank}((AB)^{\mathsf T}) = \text{rank}(AB)\)，故得证。
>
> **(2)** 若 \(B\) 可逆，则由 (1)：\(\text{rank}(AB) \le \text{rank}(A)\)。同时 \(A = (AB)B^{-1}\) → \(\text{rank}(A) = \text{rank}((AB)B^{-1}) \le \text{rank}(AB)\)。双向不等式 ⇒ \(\text{rank}(AB) = \text{rank}(A)\)。
> **ML 直觉**：乘以可逆矩阵不改变秩——它只是对列空间做了一个「基变换」。所以数据预处理中的标准化/白化（可逆变换）不会改变数据矩阵的秩，只改变列的尺度。

---

**B8.** 一个 $256 \times 256$ 的 Attention 矩阵 $S = \text{softmax}(QK^{\mathsf T}/\sqrt{d})$，其中 $Q, K \in \mathbb{R}^{256 \times 64}$。

(1) $S$ 的秩的上限是多少？为什么？
(2) 在实际 Transformer 中，$S$ 的实际秩往往远低于这个上限。为什么？这有什么后果？

> **标准答案：**
>
> **(1)** \(QK^{\mathsf T}\) 的秩 \(\le \min(\text{rank}(Q), \text{rank}(K)) \le \min(256, 64) = 64\)。Softmax 是按行（指数 + 归一化）的非线性变换，不改变矩阵的秩（因为指数函数是严格单调的，且行归一化是可逆的）。所以 \(S\) 的秩 \(\le 64\)。虽然 \(S\) 是 \(256 \times 256\)，但实际上内蕴维度最多 \(64\)——它是极度低秩的。
>
> **(2)** 实际秩往往比 \(64\) 更低，因为：
> - 不同 token 的 query/key 向量高度相关 → \(Q\) 和 \(K\) 自身就是低秩的（有效秩 \(\ll 64\)）
> - 很多 token 的 attention 分布趋于相似（比如所有 token 都高度关注 CLS 或分隔符）
> 这导致 **attention collapse**：\(\sigma_1 \gg \sigma_2 \gg \cdots\)，attention 本质上是一维的——所有 token 关注几乎相同的 context。模型浪费了多头注意力的容量。
> **Multi-head attention 的缓解作用**：每个 head 用不同的 \(W_Q, W_K, W_V\) 投影，产生不同的低秩结构（关注不同模式），拼接后获得秩的多样性。这是「低秩分解 + 多组拼接」策略的经典案例。

---

**B9.** 向量 $\mathbf{v} = (1,2,3,4,5)^{\mathsf T}$。

(1) 计算 $\|\mathbf{v}\|_1$、$\|\mathbf{v}\|_2$、$\|\mathbf{v}\|_\infty$。
(2) 验证不等式链 $\|\mathbf{v}\|_\infty \le \|\mathbf{v}\|_2 \le \|\mathbf{v}\|_1 \le \sqrt{5}\|\mathbf{v}\|_2$。
(3) 把 $\mathbf{v}$ 归一化：分别求 $L_1$-归一化（$\mathbf{v}/\|\mathbf{v}\|_1$）和 $L_2$-归一化（$\mathbf{v}/\|\mathbf{v}\|_2$）。两者的分量和分别是多少？

> **标准答案：**
>
> **(1)**
> \[\|\mathbf{v}\|_1 = 1+2+3+4+5 = 15\]
> \[\|\mathbf{v}\|_2 = \sqrt{1+4+9+16+25} = \sqrt{55} \approx 7.42\]
> \[\|\mathbf{v}\|_\infty = \max(1,2,3,4,5) = 5\]
>
> **(2)** 验证不等式：
> \[5 \le 7.42 \le 15 \le \sqrt{5} \cdot 7.42 \approx 16.58\] ✓
>
> **(3)**
> \(L_1\)-归一化：\(\mathbf{v}/15 = (0.0667, 0.1333, 0.2, 0.2667, 0.3333)^{\mathsf T}\)，分量和 \(= 1\)。
> \(L_2\)-归一化：\(\mathbf{v}/\sqrt{55} \approx (0.135, 0.270, 0.405, 0.539, 0.674)^{\mathsf T}\)，分量和 \(= (1+2+3+4+5)/\sqrt{55} = 15/\sqrt{55} \approx 2.02\)。
> \(L_1\)-归一化后的向量分量和恒为 \(1\)（概率单纯形），常用于将向量转为概率分布。\(L_2\)-归一化后向量的 \(L_2\) 范数为 \(1\)（单位球面），常用于 LayerNorm 和 cosine similarity。

---

**B10.** 证明：若矩阵 $A \in \mathbb{R}^{m \times n}$ 有奇异值 $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$，则：

(1) $\|A\|_F^2 = \sum_{i=1}^r \sigma_i^2$。
(2) $\|A\|_2 = \sigma_1$。
(3) $\|A\|_* = \sum_{i=1}^r \sigma_i$。

并比较三种范数在 $A = \text{diag}(10, 0.1, 0, \dots, 0)$（$100 \times 100$ 对角矩阵）时的值。

> **标准答案：**
>
> **(1)** 设 \(A\) 的紧 SVD 为 \(A = U_r \Sigma_r V_r^{\mathsf T}\)。则：
> \[\|A\|_F^2 = \text{tr}(A^{\mathsf T}A) = \text{tr}(V_r\Sigma_r U_r^{\mathsf T} U_r\Sigma_r V_r^{\mathsf T}) = \text{tr}(V_r\Sigma_r^2 V_r^{\mathsf T})\]
> 由迹的循环不变性：\(\text{tr}(V_r\Sigma_r^2 V_r^{\mathsf T}) = \text{tr}(\Sigma_r^2 V_r^{\mathsf T} V_r) = \text{tr}(\Sigma_r^2) = \sum_{i=1}^r \sigma_i^2\)。
>
> **(2)** \(\|A\|_2 = \sup_{\|\mathbf{x}\|=1} \|A\mathbf{x}\|\)。\(U, V\) 是正交矩阵，保持长度：
> \[\|A\mathbf{x}\|^2 = \|U\Sigma V^{\mathsf T}\mathbf{x}\|^2 = \|\Sigma (V^{\mathsf T}\mathbf{x})\|^2\]
> 令 \(\mathbf{y} = V^{\mathsf T}\mathbf{x}\)，\(\|\mathbf{y}\| = 1\)。则 \(\|\Sigma \mathbf{y}\|^2 = \sum_i \sigma_i^2 y_i^2 \le \sigma_1^2 \sum_i y_i^2 = \sigma_1^2\)。在 \(\mathbf{y} = \mathbf{e}_1\)（即 \(\mathbf{x} = V\mathbf{e}_1 = \mathbf{v}_1\)）处取等号。故 \(\|A\|_2 = \sigma_1\)。
>
> **(3)** 核范数定义即为所有奇异值之和。它等于 \(\text{tr}(\sqrt{A^{\mathsf T}A}) = \sum_i \sigma_i\)。
   501|