     1|     1|# 习题与思考：行列式、特征值与 SVD
     2|     2|
     3|     3|A 档巩固基础，B 档培养 ML 直觉。每题附完整标准答案。
     4|     4|
     5|     5|---
     6|     6|
     7|     7|## 三、行列式、特征值与 SVD（§9–§13）
     8|     8|
     9|     9|### A 档
    10|    10|
    11|    11|**A1.** 化二次型 $f(x_1,x_2)=3x_1^2+2x_1x_2+3x_2^2$ 为标准形，并判断其正定性。
    12|    12|
    13|    13|> **标准答案：**
    14|    14|> 写成矩阵形式：\(f(\mathbf{x}) = \mathbf{x}^{\mathsf T}A\mathbf{x}\)，对称矩阵 \(A = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}\)（注意交叉项 \(2x_1x_2\) 拆成 \(x_1x_2 + x_2x_1\)，各分一半）。
    15|    15|> 求特征值：
    16|    16|> \[\det(A - \lambda I) = \det\begin{bmatrix}3-\lambda & 1\\1 & 3-\lambda\end{bmatrix} = (3-\lambda)^2 - 1 = \lambda^2 - 6\lambda + 8 = 0\]
    17|    17|> \[(\lambda - 2)(\lambda - 4) = 0 \Rightarrow \lambda_1 = 4,\; \lambda_2 = 2\]
    18|    18|> 对应特征向量：
    19|    19|> - \(\lambda_1 = 4\)：\((A - 4I)\mathbf{v} = \begin{bmatrix}-1&1\\1&-1\end{bmatrix}\mathbf{v} = \mathbf{0} \Rightarrow \mathbf{v}_1 = \frac{1}{\sqrt{2}}(1, 1)^{\mathsf T}\)
    20|    20|> - \(\lambda_2 = 2\)：\((A - 2I)\mathbf{v} = \begin{bmatrix}1&1\\1&1\end{bmatrix}\mathbf{v} = \mathbf{0} \Rightarrow \mathbf{v}_2 = \frac{1}{\sqrt{2}}(-1, 1)^{\mathsf T}\)
    21|    21|> 正交变换 \(\mathbf{x} = Q\mathbf{y}\)（\(Q = [\mathbf{v}_1, \mathbf{v}_2]\)）：
    22|    22|> \[f = \lambda_1 y_1^2 + \lambda_2 y_2^2 = 4y_1^2 + 2y_2^2\]
    23|    23|> **正定性**：所有特征值 \(> 0\) → 正定。几何上，\(f(\mathbf{x}) = 1\) 是一个倾斜的椭圆，短轴沿 \((1,1)\) 方向（对应 \(\lambda_1=4\)，曲率大），长轴沿 \((-1,1)\) 方向（对应 \(\lambda_2=2\)，曲率小）。
    24|    24|
    25|    25|---
    26|    26|
    27|    27|**A2.** 求 $A = \begin{bmatrix}3&1\\1&3\end{bmatrix}$ 的特征值与特征向量，写出对角化形式 $A = Q\Lambda Q^{\mathsf T}$。
    28|    28|
    29|    29|> **标准答案：**
    30|    30|> 与 A1 同一个矩阵。特征值 \(\lambda_1 = 4\)，\(\lambda_2 = 2\)。
    31|    31|> 特征向量（已标准化）：
    32|    32|> \[\mathbf{v}_1 = \frac{1}{\sqrt{2}}\begin{bmatrix}1\\1\end{bmatrix},\quad \mathbf{v}_2 = \frac{1}{\sqrt{2}}\begin{bmatrix}-1\\1\end{bmatrix}\]
    33|    33|> 验证正交性：\(\mathbf{v}_1 \cdot \mathbf{v}_2 = \frac{1}{2}(-1+1) = 0\) ✓。\(A\) 对称 → 特征向量正交 → \(Q\) 是正交矩阵（\(Q^{\mathsf T} = Q^{-1}\)）。
    34|    34|> 对角化：
    35|    35|> \[A = Q\Lambda Q^{\mathsf T} = \begin{bmatrix}1/\sqrt{2} & -1/\sqrt{2}\\1/\sqrt{2} & 1/\sqrt{2}\end{bmatrix}\begin{bmatrix}4&0\\0&2\end{bmatrix}\begin{bmatrix}1/\sqrt{2} & 1/\sqrt{2}\\-1/\sqrt{2} & 1/\sqrt{2}\end{bmatrix}\]
    36|    36|> 验证：
    37|    37|> \[Q\Lambda Q^{\mathsf T} = \begin{bmatrix}1/\sqrt{2}&-1/\sqrt{2}\\1/\sqrt{2}&1/\sqrt{2}\end{bmatrix}\begin{bmatrix}4/\sqrt{2}&4/\sqrt{2}\\-2/\sqrt{2}&2/\sqrt{2}\end{bmatrix}
    38|    38|> = \frac{1}{2}\begin{bmatrix}4+2&4-2\\4-2&4+2\end{bmatrix} = \begin{bmatrix}3&1\\1&3\end{bmatrix} = A\] ✓
    39|    39|
    40|    40|---
    41|    41|
    42|    42|**A3.** 用矩阵微积分求梯度。设 $L(W) = \frac{1}{2}\|WX - Y\|_F^2$，其中 $W \in \mathbb{R}^{d \times n}$，$X \in \mathbb{R}^{n \times m}$，$Y \in \mathbb{R}^{d \times m}$。求 $\nabla_W L$。
    43|    43|
    44|    44|> **标准答案：**
    45|    45|> 展开 Frobenius 范数平方：
    46|    46|> \[L = \frac{1}{2}\text{tr}\big((WX - Y)^{\mathsf T}(WX - Y)\big)\]
    47|    47|> 微分：
    48|    48|> \[dL = \frac{1}{2}\text{tr}\big(d(WX - Y)^{\mathsf T}(WX - Y) + (WX - Y)^{\mathsf T}d(WX - Y)\big)\]
    49|    49|> 由迹的对称性和 \(d(WX) = (dW)X\)（\(X\) 是常数）：
    50|    50|> \[dL = \text{tr}\big((WX - Y)^{\mathsf T}(dW)X\big)\]
    51|    51|> 利用 \(\text{tr}(AB) = \text{tr}(BA)\)：
    52|    52|> \[dL = \text{tr}\big(X(WX - Y)^{\mathsf T} dW\big)\]
    53|    53|> 标准形式 \(dL = \text{tr}(G^{\mathsf T} dW) \Rightarrow \nabla_W L = G\)。所以：
    54|    54|> \[\nabla_W L = (WX - Y)X^{\mathsf T}\]
    55|    55|> **ML 直觉**：梯度 = 误差矩阵 × 输入的转置。这是线性层反向传播的矩阵形式——每个权重的梯度是「它负责的输出误差」和「它对应的输入」的乘积之和。与 `nn.Linear` 的 `weight.grad` 完全一致。
    56|    56|
    57|    57|---
    58|    58|
    59|    59|**A4.** 求 softmax 函数 $\mathbf{s}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$ 的雅可比矩阵 $J = \partial \mathbf{s} / \partial \mathbf{z}$。
    60|    60|
    61|    61|> **标准答案：**
    62|    62|> 对任意 \(i, j\)：
    63|    63|> \[\frac{\partial s_i}{\partial z_j} = \frac{\partial}{\partial z_j}\left(\frac{e^{z_i}}{\sum_k e^{z_k}}\right)\]
    64|    64|> **当 \(i = j\)**（商数法则，分子的指数在分母中）：
    65|    65|> \[\frac{\partial s_i}{\partial z_i} = \frac{e^{z_i} \cdot \sum_k e^{z_k} - e^{z_i} \cdot e^{z_i}}{(\sum_k e^{z_k})^2} = \frac{e^{z_i}}{\sum_k e^{z_k}} - \left(\frac{e^{z_i}}{\sum_k e^{z_k}}\right)^2 = s_i - s_i^2 = s_i(1 - s_i)\]
    66|    66|> **当 \(i \neq j\)**（分子的指数不在分母中）：
    67|    67|> \[\frac{\partial s_i}{\partial z_j} = \frac{0 \cdot \sum_k e^{z_k} - e^{z_i} \cdot e^{z_j}}{(\sum_k e^{z_k})^2} = -s_i s_j\]
    68|    68|> **矩阵形式**：
    69|    69|> \[J = \text{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^{\mathsf T}\]
    70|    70|> 性质：\(J\) 对称且半正定，\(\text{rank}(J) = K-1\)（零空间由 \(\mathbf{1}\) 张成，因为 \(\sum_i s_i = 1 \Rightarrow \sum_j \frac{\partial s_i}{\partial z_j} = 0\)）。
    71|    71|> 这个 \(J\) 与交叉熵配合使用时会产生极大的简化——见 B2。
    72|    72|
    73|    73|---
    74|    74|
    75|    75|**A5.** 不直接计算行列式，仅用行列式的性质判断以下命题的真伪：
    76|    76|
    77|    77|(1) $A$ 某一行全 $0 \Rightarrow \det(A) = 0$。
    78|    78|(2) $A$ 两行成比例 $\Rightarrow \det(A) = 0$。
    79|    79|(3) $\det(A)=2 \Rightarrow \det(A^{-1}) = -2$。
    80|    80|(4) 初等行变换 $R_2 \leftarrow R_2 - 3R_1$ 使行列式乘以 $-3$。
    81|    81|(5) $\det(A - \lambda I)=0 \iff \lambda$ 是 $A$ 的特征值。
    82|    82|
    83|    83|> **标准答案：**
    84|    84|>
    85|    85|> **(1) 对。** 按该行做 Laplace 展开：每项都含 \(0\) 因子 → 和为 \(0\)。几何上：该行对应的维度被完全压扁 → 体积为 \(0\)。
    86|    86|>
    87|    87|> **(2) 对。** 两行成比例 → 行线性相关 → 高斯消元产生零行 → 行列式为 \(0\)。几何上：两个「边」平行 → 平行多面体退化 → 体积为 \(0\)。
    88|    88|>
    89|    89|> **(3) 错。** \(\det(A^{-1}) = 1/\det(A) = 1/2\)，不是 \(-2\)。行列式乘积法则：\(\det(A)\det(A^{-1}) = \det(AA^{-1}) = \det(I) = 1\)。
    90|    90|>
    91|    91|> **(4) 错。** 把一行的倍数加到另一行**不改变行列式**。验证：\(\det\begin{bmatrix}a&b\\c+3a&d+3b\end{bmatrix} = a(d+3b) - b(c+3a) = ad - bc = \det\begin{bmatrix}a&b\\c&d\end{bmatrix}\)。只有交换两行（变号）和对一行乘以 \(c\)（乘 \(c\)）才会改变行列式值。
    92|    92|>
    93|    93|> **(5) 对。** \(\det(A - \lambda I)=0 \iff A - \lambda I\) 奇异 \(\iff\) 存在 \(\mathbf{x} \neq \mathbf{0}\) 使 \((A - \lambda I)\mathbf{x} = \mathbf{0} \iff A\mathbf{x} = \lambda\mathbf{x}\)。这就是特征方程的标准形式。
    94|    94|
    95|    95|---
    96|    96|
    97|    97|**A6.** 计算矩阵 $A = \begin{bmatrix} 2 & 0 & 1 \\ 0 & 3 & 0 \\ 1 & 0 & 2 \end{bmatrix}$ 的行列式，并解释其几何含义。
    98|    98|
    99|    99|> **标准答案：**
   100|   100|> 按第二行展开（因为有两个 \(0\)，计算最简单）：
   101|   101|> \[\det(A) = 3 \cdot (-1)^{2+2} \cdot \det\begin{bmatrix}2&1\\1&2\end{bmatrix} = 3 \cdot (4 - 1) = 3 \cdot 3 = 9\]
   102|   102|> 也可直接按标准公式计算 \(3 \times 3\) 行列式来验证。
   103|   103|> \(\det(A) = 9 \neq 0\) → **可逆**。
   104|   104|>
   105|   105|> **几何含义**：\(A\) 的列是 \((2,0,1)^{\mathsf T}\)、\((0,3,0)^{\mathsf T}\)、\((1,0,2)^{\mathsf T}\)。\(A\) 把 \(\mathbb{R}^3\) 中的单位立方体变换为一个平行六面体，体积 \(= |\det(A)| = 9\)。具体来说：第二列沿 \(y\) 轴拉伸 \(3\) 倍；第一列和第三列在 \(xz\) 平面内张成一个面积为 \(\det\begin{bmatrix}2&1\\1&2\end{bmatrix} = 3\) 的平行四边形；总体积 \(= 3 \times 3 = 9\)。
   106|   106|
   107|   107|---
   108|   108|
   109|   109|**A7.** 对称矩阵 $B = \begin{bmatrix}2&2&0\\2&5&0\\0&0&3\end{bmatrix}$。
   110|   110|
   111|   111|(1) 求 $B$ 的所有特征值。
   112|   112|(2) 求一组标准正交的特征向量，写出谱分解 $B = Q\Lambda Q^{\mathsf T}$。
   113|   113|(3) 利用特征值判断 $B$ 的正定性。
   114|   114|
   115|   115|> **标准答案：**
   116|   116|>
   117|   117|> **(1)** \(\det(B - \lambda I) = \det\begin{bmatrix}2-\lambda&2&0\\2&5-\lambda&0\\0&0&3-\lambda\end{bmatrix}\)。
   118|   118|> 按第三行/列展开：
   119|   119|> \[= (3-\lambda) \cdot \det\begin{bmatrix}2-\lambda&2\\2&5-\lambda\end{bmatrix}\]
   120|   120|> \[= (3-\lambda)[(2-\lambda)(5-\lambda) - 4] = (3-\lambda)(\lambda^2 - 7\lambda + 6)\]
   121|   121|> \[= (3-\lambda)(\lambda-1)(\lambda-6) = 0\]
   122|   122|> \(\lambda_1 = 6\)，\(\lambda_2 = 3\)，\(\lambda_3 = 1\)。
   123|   123|>
   124|   124|> **(2)** 分别求特征向量：
   125|   125|> - \(\lambda_1 = 6\)：\((B-6I)\mathbf{v} = \begin{bmatrix}-4&2&0\\2&-1&0\\0&0&-3\end{bmatrix}\mathbf{v} = \mathbf{0}\)。前两行给出 \(v_1 = t, v_2 = 2t\)，第三行 \(v_3 = 0\)。取 \(\mathbf{v}_1 = \frac{1}{\sqrt{5}}(1, 2, 0)^{\mathsf T}\)。
   126|   126|> - \(\lambda_2 = 3\)：\((B-3I)\mathbf{v} = \begin{bmatrix}-1&2&0\\2&2&0\\0&0&0\end{bmatrix}\mathbf{v} = \mathbf{0}\)。\(v_1 = 2t, v_2 = t, v_3\) 自由。取 \(\mathbf{v}_2 = (0, 0, 1)^{\mathsf T}\)（与 \(\mathbf{v}_1\) 正交 ✓）。标准化：\(\mathbf{v}_2 = (0, 0, 1)^{\mathsf T}\)（已归一）。
   127|   127|> - \(\lambda_3 = 1\)：\((B-I)\mathbf{v} = \begin{bmatrix}1&2&0\\2&4&0\\0&0&2\end{bmatrix}\mathbf{v} = \mathbf{0}\)。\(v_1 = -2t, v_2 = t, v_3 = 0\)。取 \(\mathbf{v}_3 = \frac{1}{\sqrt{5}}(-2, 1, 0)^{\mathsf T}\)。
   128|   128|> \[Q = \begin{bmatrix}1/\sqrt{5}&0&-2/\sqrt{5}\\2/\sqrt{5}&0&1/\sqrt{5}\\0&1&0\end{bmatrix},\quad \Lambda = \text{diag}(6,3,1)\]
   129|   129|>
   130|   130|> **(3)** 所有特征值 \(>0\) → **正定** ✓。物理直觉：\(x^{\mathsf T}Bx > 0\) 对所有非零 \(\mathbf{x}\) 成立——这是三维空间中的「椭球碗」，任何方向都是上升的。
   131|   131|
   132|   132|---
   133|   133|
   134|   134|### B 档
   135|   135|
   136|   136|**B1.** 某深度网络的 Hessian 特征值为 $\{125,\; 3.2,\; 0.01,\; -0.5,\; -18\}$。判断当前点在优化地形中的位置，预测 SGD 和 Adam 的表现差异。
   137|   137|
   138|   138|> **标准答案：**
   139|   139|> 同时存在正特征值和负特征值 → 当前点是**鞍点**（saddle point），而非局部极小。正惯性指数 \(p = 3\)（\(125, 3.2, 0.01\)），负惯性指数 \(q = 2\)（\(-0.5, -18\)）。
   140|   140|> 条件数（对正特征值部分）：\(\kappa = 125 / 0.01 = 12500\)（极差）。
   141|   141|> **SGD 的表现**：
   142|   142|> - 沿 \(\lambda = 125\) 的方向：曲率极大 → 步长稍大即剧烈震荡（overshoot）
   143|   143|> - 沿 \(\lambda = 0.01\) 的方向：几乎平坦 → 寸步难行（每步进展约 \(1/12500\) 的相对量）
   144|   144|> - 沿 \(\lambda = -18\) 的方向：**负曲率** → 梯度下降实际上在沿此方向「逃离」鞍点（因为负曲率方向上的梯度指向下山方向）
   145|   145|> - 典型现象：loss 在鞍点附近「卡住」很久（等梯度噪声将参数推到负曲率方向），然后突然继续下降
   146|   146|> **Adam 的表现**：
   147|   147|> - 自适应学习率处理各方向的不同尺度：\(\lambda=125\) 方向用小步（\(\sqrt{v_t}\) 大）→ 不震荡；\(\lambda=0.01\) 方向用大步（\(\sqrt{v_t}\) 小）→ 加速前进；\(\lambda=-18\) 方向也以合适步长下行 → 快速逃离鞍点
   148|   148|> - 所有 \(5\) 个方向近乎同步收敛，远优于 SGD
   149|   149|
   150|   150|---
   151|   151|
   152|   152|**B2.** 证明恒等式 $\nabla_{\mathbf{z}}\;\text{CrossEntropy}(\text{softmax}(\mathbf{z}), \mathbf{y}) = \hat{\mathbf{y}} - \mathbf{y}$（使用微分法）。
   153|   153|
   154|   154|> **标准答案：**
   155|   155|> 设 \(\hat{\mathbf{y}} = \text{softmax}(\mathbf{z})\)，损失 \(L = -\sum_{k=1}^K y_k \log \hat{y}_k\)。
   156|   156|> 微分：
   157|   157|> \[dL = -\sum_k \frac{y_k}{\hat{y}_k} d\hat{y}_k = -\mathbf{y}^{\mathsf T} \text{diag}(\hat{\mathbf{y}})^{-1} d\hat{\mathbf{y}}\]
   158|   158|> 由 A4：\(d\hat{\mathbf{y}} = (\text{diag}(\hat{\mathbf{y}}) - \hat{\mathbf{y}}\hat{\mathbf{y}}^{\mathsf T}) d\mathbf{z}\)。
   159|   159|> 代入：
   160|   160|> \[dL = -\mathbf{y}^{\mathsf T} \text{diag}(\hat{\mathbf{y}})^{-1} (\text{diag}(\hat{\mathbf{y}}) - \hat{\mathbf{y}}\hat{\mathbf{y}}^{\mathsf T}) d\mathbf{z}\]
   161|   161|> \[= -\mathbf{y}^{\mathsf T} (I - \mathbf{1}\hat{\mathbf{y}}^{\mathsf T}) d\mathbf{z} \quad \text{（因为 } \text{diag}(\hat{\mathbf{y}})^{-1}\text{diag}(\hat{\mathbf{y}}) = I\text{）}\]
   162|   162|> \[= (-\mathbf{y}^{\mathsf T} + \mathbf{y}^{\mathsf T}\mathbf{1} \cdot \hat{\mathbf{y}}^{\mathsf T}) d\mathbf{z}\]
   163|   163|> \[= (-\mathbf{y}^{\mathsf T} + 1 \cdot \hat{\mathbf{y}}^{\mathsf T}) d\mathbf{z} \quad \text{（因为 } \sum_k y_k = 1\text{）}\]
   164|   164|> \[= (\hat{\mathbf{y}} - \mathbf{y})^{\mathsf T} d\mathbf{z}\]
   165|   165|> 标准形式 \(dL = (\nabla_{\mathbf{z}} L)^{\mathsf T} d\mathbf{z}\) → \(\nabla_{\mathbf{z}} L = \hat{\mathbf{y}} - \mathbf{y}\)。
   166|   166|> **这是 ML 中最优雅的梯度公式之一。** 最后一层反向传播的「误差信号」就是**预测概率减去真实标签**——极度简洁且数值稳定。无论前面的网络多复杂，softmax + CE 的组合永远给出这个形式。
   167|   167|
   168|   168|---
   169|   169|
   170|   170|**B3.** 为什么协方差矩阵必须是半正定的？如果样本协方差矩阵出现负特征值（浮点误差），会有什么实际影响？如何修复？
   171|   171|
   172|   172|> **标准答案：**
   173|   173|>
   174|   174|> **证明半正定性**：对任意向量 \(\mathbf{x}\)：
   175|   175|> \[\mathbf{x}^{\mathsf T}\Sigma \mathbf{x} = \mathbf{x}^{\mathsf T} \mathbb{E}[(\mathbf{X}-\boldsymbol{\mu})(\mathbf{X}-\boldsymbol{\mu})^{\mathsf T}] \mathbf{x} = \mathbb{E}[(\mathbf{x}^{\mathsf T}(\mathbf{X}-\boldsymbol{\mu}))^2] = \text{Var}(\mathbf{x}^{\mathsf T}\mathbf{X}) \ge 0\]
   176|   176|> 方差的期望非负 → \(\Sigma\) 半正定。这是协方差矩阵的数学必然，不是巧合。
   177|   177|> **负特征值的实际影响**（由于样本量不足或浮点舍入误差）：
   178|   178|> - **概率密度函数**：在负特征值对应的方向上，\(\mathcal{N}(\boldsymbol{\mu}, \Sigma)\) 的密度含 \(\exp(-\frac{1}{2} \cdot \text{负数})\) → 指数爆炸 → 无意义
   179|   179|> - **Cholesky 分解失败**：\(\Sigma = LL^{\mathsf T}\) 要求正定；若 \(\Sigma\) 不定 → `np.linalg.cholesky` 抛出 `LinAlgError`
   180|   180|> - **多元正态采样崩溃**：\(\mathbf{x} = L\mathbf{z}\) 的 \(L\) 不存在
   181|   181|> - **马氏距离** \((x-\mu)^{\mathsf T}\Sigma^{-1}(x-\mu)\) 可能为负 → 失去距离度量意义
   182|   182|> **修复方法**：Shrinkage（收缩估计）：
   183|   183|> \[\Sigma_{\text{shrunk}} = (1-\alpha)\Sigma_{\text{sample}} + \alpha I, \quad \alpha \in (0, 1)\]
   184|   184|> 给所有特征值加上 \(\alpha\)：\(\lambda_i^{\text{new}} = (1-\alpha)\lambda_i + \alpha\)。最小特征值被抬升至少 \(\alpha\)，强制正定性。典型的 \(\alpha\) 取 \(0.01 \sim 0.1\)。这正是 `sklearn.covariance.ShrunkCovariance` 和 Ledoit-Wolf 估计的核心思想。
   185|   185|
   186|   186|---
   187|   187|
   188|   188|**B4.** 设 $A$ 的 SVD 为 $A = U\Sigma V^{\mathsf T}$，其中 $\Sigma = \text{diag}(5, 2, 0)$。
   189|   189|
   190|   190|(1) $A$ 的秩是多少？
   191|   191|(2) 描述 $A$ 对 $\mathbb{R}^3$ 中单位球的几何变换（三步拆解）。
   192|   192|(3) $\mathbf{v}_3$ 对应 $\sigma_3=0$。$A\mathbf{v}_3 = ?$ $\mathbf{v}_3$ 属于什么子空间？
   193|   193|(4) （ML）PCA 中若出现很多接近 $0$ 的奇异值，说明什么？
   194|   194|
   195|   195|> **标准答案：**
   196|   196|>
   197|   197|> **(1)** 秩 \(=\) 非零奇异值个数 \(= 2\)。零奇异值对应被完全压扁的方向。
   198|   198|>
   199|   199|> **(2)** 三步拆解：
   200|   200|> ① \(V^{\mathsf T}\)：正交矩阵 → 旋转/反射单位球（形状不变，仍是单位球）。
   201|   201|> ② \(\Sigma\)：沿三个坐标轴各自缩放 \(\sigma_i\) 倍。\(\sigma_1=5\) → \(x\) 方向拉伸 \(5\times\)；\(\sigma_2=2\) → \(y\) 方向拉伸 \(2\times\)；\(\sigma_3=0\) → \(z\) 方向压扁到 \(0\)。球变成一个位于 \(xy\) 平面内的椭圆盘。
   202|   202|> ③ \(U\)：再旋转/反射这个椭圆盘，不改变形状。
   203|   203|> 任何线性变换本质上就是这三步：旋转 → 拉伸/压扁 → 再旋转。没有例外。
   204|   204|>
   205|   205|> **(3)** \(V^{\mathsf T}\mathbf{v}_3 = \mathbf{e}_3\)（因为 \(V\) 的列是 \(\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3\)，\(V^{\mathsf T}\mathbf{v}_i = \mathbf{e}_i\)）。
   206|   206|> \[A\mathbf{v}_3 = U\Sigma (V^{\mathsf T}\mathbf{v}_3) = U\Sigma \mathbf{e}_3 = U \cdot \mathbf{0} = \mathbf{0}\]
   207|   207|> \(\mathbf{v}_3 \in N(A)\)（零空间）。**SVD 的优雅之处：\(V\) 的最后 \(n-r\) 列直接给出零空间的标准正交基。**
   208|   208|>
   209|   209|> **(4)** 奇异值接近 \(0\) → 协方差矩阵有接近 \(0\) 的特征值 → 数据几乎分布在一个低维子空间内：
   210|   210|> - 输入维度假高 → 可以大幅降维而不损失信息
   211|   211|> - 模型容量不必正比于名义维度 → **LoRA** 的核心洞察：权重更新做低秩约束，因为问题本身就是低秩的
   212|   212|> - 若强行在高维空间做最小二乘回归 → 极小奇异值在伪逆中变成极大值（\(1/\sigma\)）→ 系数方差爆炸 → 需要正则化
   213|   213|
   214|   214|---
   215|   215|
   216|   216|**B5.** 设 $A$ 是 $n \times n$ 矩阵，特征值为 $\lambda_1, \dots, \lambda_n$。
   217|   217|
   218|   218|(1) 证明 $\det(A) = \prod_{i=1}^n \lambda_i$ 和 $\text{tr}(A) = \sum_{i=1}^n \lambda_i$。
   219|   219|(2) 推论：$\det(A) = 0$ 暗示了什么？
   220|   220|(3) （ML）协方差矩阵的 $\sum \lambda_i$（总方差）和 $\prod \lambda_i$（广义方差）各有什么意义？
   221|   221|
   222|   222|> **标准答案：**
   223|   223|>
   224|   224|> **(1)** 假设 \(A\) 可对角化：\(A = PDP^{-1}\)。
   225|   225|> 行列式：\(\det(A) = \det(PDP^{-1}) = \det(P)\det(D)\det(P^{-1}) = \det(D) = \prod \lambda_i\)。
   226|   226|> 迹：\(\text{tr}(A) = \text{tr}(PDP^{-1}) = \text{tr}(P^{-1}PD) = \text{tr}(D) = \sum \lambda_i\)。
   227|   227|> （即使 \(A\) 不可对角化，这两个等式对特征值（含代数重数）仍然成立，通过 Jordan 标准型可证。）
   228|   228|>
   229|   229|> **(2)** \(\det(A) = 0 \iff\) 至少一个 \(\lambda_i = 0 \iff A\) 奇异（不可逆）\(\iff\) 秩亏。
   230|   230|>
   231|   231|> **(3)**
   232|   232|> - **总方差** \(= \sum \lambda_i = \text{tr}(\Sigma)\)：衡量数据在所有方向上的总散布量。PCA 中常看「前 \(k\) 个特征值之和占比」来判断降维保留的信息量。
   233|   233|> - **广义方差** \(= \prod \lambda_i = \det(\Sigma)\)：衡量数据占据的「体积」。\(\det(\Sigma) \approx 0\) 意味着至少一个特征值极小 → 数据几乎分布在一个低于 \(n\) 维的子空间内 → 特征之间存在高度共线性。
   234|   234|
   235|   235|---
   236|   236|
   237|   237|**B6.** **Rayleigh 商** $R(\mathbf{x}) = \frac{\mathbf{x}^{\mathsf T} H \mathbf{x}}{\mathbf{x}^{\mathsf T} \mathbf{x}}$，$H$ 对称。
   238|   238|
   239|   239|(1) $H = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}$，计算 $R(1,1)$ 和 $R(1,-1)$。
   240|   240|(2) 它们和 $H$ 的特征值有何关系？
   241|   241|(3) （ML）沿负梯度方向 $\mathbf{d} = -\nabla L$，Rayleigh 商如何决定 SGD 的收敛速度？
   242|   242|
   243|   243|> **标准答案：**
   244|   244|>
   245|   245|> **(1)**
   246|   246|> \[R(1,1) = \frac{[1,1]H[1,1]^{\mathsf T}}{1^2+1^2} = \frac{[1,1]\begin{bmatrix}4\\4\end{bmatrix}}{2} = \frac{8}{2} = 4\]
   247|   247|> \[R(1,-1) = \frac{[1,-1]H[1,-1]^{\mathsf T}}{2} = \frac{[1,-1]\begin{bmatrix}2\\-2\end{bmatrix}}{2} = \frac{4}{2} = 2\]
   248|   248|>
   249|   249|> **(2)** 由 A1/A2，\(H\) 的特征值为 \(\lambda_{\max}=4\)（对应 \((1,1)\)），\(\lambda_{\min}=2\)（对应 \((1,-1)\)）。
   250|   250|> **Rayleigh 商定理**：对任意 \(\mathbf{x} \neq \mathbf{0}\)，\(\lambda_{\min} \le R(\mathbf{x}) \le \lambda_{\max}\)，等号在特征向量方向取得。这里：\(R(1,1) = \lambda_{\max} = 4\)，\(R(1,-1) = \lambda_{\min} = 2\)——恰好命中 ✓。
   251|   251|>
   252|   252|> **(3)** 在局部二次近似下，沿 \(\mathbf{d} = -\nabla L\) 走一步的最优学习率的倒数是该方向的 Rayleigh 商：
   253|   253|> \[R(\mathbf{d}) = \frac{\mathbf{d}^{\mathsf T} H \mathbf{d}}{\mathbf{d}^{\mathsf T} \mathbf{d}}\]
   254|   254|> 这个值度量了「当前梯度方向上 Hessian 的有效曲率」：
   255|   255|> - 若 \(\nabla L\) 落在 \(\lambda_{\max}\) 方向 → \(R(\mathbf{d})\) 大 → 每步下降快但需要小学习率防震荡
   256|   256|> - 若 \(\nabla L\) 落在 \(\lambda_{\min}\) 方向 → \(R(\mathbf{d})\) 小 → 每步进展微乎其微
   257|   257|> Hessian 的**条件数** \(\kappa = \lambda_{\max}/\lambda_{\min}\) 表征了最不利和最有利方向之间 Rayleigh 商的比值——它直接决定了 SGD 在最坏情况下的收敛速度。
   258|   258|
   259|   259|---
   260|   260|
   261|   261|[← 返回教程](determinant-eigen-svd.md)　　　[下一章习题 →](basis-and-neural-networks-exercises.md)