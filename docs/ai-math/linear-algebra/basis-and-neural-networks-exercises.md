     1|     1|# 习题与思考：基变换与神经网络
     2|     2|
     3|     3|A 档巩固基础，B 档培养 ML 直觉。每题附完整标准答案。
     4|     4|
     5|     5|---
     6|     6|
     7|     7|## 四、基变换与神经网络（§12–§14）
     8|     8|
     9|     9|### A 档
    10|    10|
    11|    11|**A1.** 设线性变换 $T$ 在标准基下的矩阵为 $A = \begin{bmatrix}1&2\\3&4\end{bmatrix}$。求 $T$ 在基 $\mathcal{B} = \{(1,1)^{\mathsf T},\; (1,-1)^{\mathsf T}\}$ 下的矩阵表示 $B$。
    12|    12|
    13|    13|> **标准答案：**
    14|    14|> 过渡矩阵 \(P\) 的列就是新基向量（用标准基坐标写出）：
    15|    15|> \[P = \begin{bmatrix}1 & 1\\1 & -1\end{bmatrix}\]
    16|    16|> 求逆：\(\det(P) = -2\)。
    17|    17|> \[P^{-1} = \frac{1}{-2}\begin{bmatrix}-1 & -1\\-1 & 1\end{bmatrix} = \frac{1}{2}\begin{bmatrix}1 & 1\\1 & -1\end{bmatrix}\]
    18|    18|> 注意这里 \(P^{-1} = \frac{1}{2} P\)！因为 \(P\) 是对合矩阵（\(P^2 = 2I\)）。
    19|    19|> 相似变换：
    20|    20|> \[B = P^{-1}AP = \frac{1}{2}\begin{bmatrix}1&1\\1&-1\end{bmatrix} \begin{bmatrix}1&2\\3&4\end{bmatrix} \begin{bmatrix}1&1\\1&-1\end{bmatrix}\]
    21|    21|> 先算 \(AP\)：
    22|    22|> \[AP = \begin{bmatrix}1&2\\3&4\end{bmatrix}\begin{bmatrix}1&1\\1&-1\end{bmatrix} = \begin{bmatrix}3&-1\\7&-1\end{bmatrix}\]
    23|    23|> 再算 \(P^{-1}(AP)\)：
    24|    24|> \[B = \frac{1}{2}\begin{bmatrix}1&1\\1&-1\end{bmatrix}\begin{bmatrix}3&-1\\7&-1\end{bmatrix} = \frac{1}{2}\begin{bmatrix}10&-2\\-4&0\end{bmatrix} = \begin{bmatrix}5&-1\\-2&0\end{bmatrix}\]
    25|    25|> 验证不变量：\(\text{tr}(A) = 1+4 = 5\)，\(\text{tr}(B) = 5+0 = 5\) ✓。\(\det(A) = -2\)，\(\det(B) = 5\cdot0 - (-1)(-2) = -2\) ✓。\(A\) 的特征值：\(\lambda^2 - 5\lambda - 2 = 0\)，\(\lambda = \frac{5 \pm \sqrt{33}}{2}\)。\(B\) 的特征值相同 ✓。相似矩阵有相同的迹、行列式和特征值。
    26|    26|
    27|    27|---
    28|    28|
    29|    29|**A2.** 设 $B = P^{-1}AP$（$P$ 可逆）。证明：
    30|    30|
    31|    31|(1) $\text{tr}(B) = \text{tr}(A)$；
    32|    32|(2) $\det(B) = \det(A)$；
    33|    33|(3) 若 $A\mathbf{x} = \lambda\mathbf{x}$（$\mathbf{x} \neq \mathbf{0}$），则 $\lambda$ 也是 $B$ 的特征值，并写出对应特征向量。
    34|    34|
    35|    35|> **标准答案：**
    36|    36|>
    37|    37|> **(1)** \(\text{tr}(B) = \text{tr}(P^{-1}AP) = \text{tr}(APP^{-1})\)（迹的循环不变性：\(\text{tr}(XYZ) = \text{tr}(YZX)\)）
    38|    38|> \[= \text{tr}(AI) = \text{tr}(A)\]
    39|    39|>
    40|    40|> **(2)** \(\det(B) = \det(P^{-1}AP) = \det(P^{-1})\det(A)\det(P)\)（行列式乘法法则）
    41|    41|> \[= \frac{1}{\det(P)} \cdot \det(A) \cdot \det(P) = \det(A)\]
    42|    42|>
    43|    43|> **(3)** 构造 \(B\) 的特征向量：
    44|    44|> \[B(P^{-1}\mathbf{x}) = P^{-1}AP(P^{-1}\mathbf{x}) = P^{-1}A\mathbf{x} = P^{-1}(\lambda\mathbf{x}) = \lambda(P^{-1}\mathbf{x})\]
    45|    45|> 所以 \(\lambda\) 是 \(B\) 的特征值，\(P^{-1}\mathbf{x}\) 是对应的特征向量。**特征向量被 \(P^{-1}\)「翻译」到了新坐标系中。**
    46|    46|>
    47|    47|> **结论**：相似变换不改变矩阵的谱（特征值）、迹和行列式。这些量是线性变换本身的固有属性，不依赖于「坐标系」的选择。
    48|    48|
    49|    49|---
    50|    50|
    51|    51|**A3.** 考虑纯线性两层网络：$\mathbf{y} = W_2(W_1 \mathbf{x})$，$W_1 \in \mathbb{R}^{3 \times 4}$，$W_2 \in \mathbb{R}^{2 \times 3}$。
    52|    52|
    53|    53|(1) 该网络等效于什么单矩阵变换 $W_{\text{eff}}$？
    54|    54|(2) $W_{\text{eff}}$ 的秩最大是多少？这说明堆纯线性层的本质限制是什么？
    55|    55|(3) 插入 ReLU 后 $\mathbf{y} = W_2 \cdot \text{ReLU}(W_1\mathbf{x})$，还能写成 $W_{\text{eff}}\mathbf{x}$ 吗？为什么？
    56|    56|
    57|    57|> **标准答案：**
    58|    58|>
    59|    59|> **(1)** \(\mathbf{y} = W_2(W_1\mathbf{x}) = (W_2 W_1)\mathbf{x} = W_{\text{eff}}\mathbf{x}\)，其中 \(W_{\text{eff}} \in \mathbb{R}^{2 \times 4}\)。两个线性变换的复合仍是线性变换——这就是「线性」的定义。
    60|    60|>
    61|    61|> **(2)**
    62|    62|> \[\text{rank}(W_{\text{eff}}) = \text{rank}(W_2 W_1) \le \min(\text{rank}(W_1), \text{rank}(W_2))\]
    63|    63|> \(\text{rank}(W_1) \le \min(3,4) = 3\)，\(\text{rank}(W_2) \le \min(2,3) = 2\)。所以 \(\text{rank}(W_{\text{eff}}) \le 2\)。
    64|    64|> 本质限制：无论 \(W_1\) 多大（3×4），输出的「信息瓶颈」被 \(W_2\) 的输出维度 \(2\) 卡死——最窄层决定了整个纯线性网络的最大表达能力。堆 100 层纯线性网络，等效于 1 层——**非线性激活函数是深度网络存在意义的唯一理由。**
    65|    65|>
    66|    66|> **(3) 不能。** ReLU 是非线性函数。具体来说，\(\text{ReLU}(z) = \max(0, z)\) 不满足齐次性：\(\text{ReLU}(c z) \neq c \cdot \text{ReLU}(z)\)（当 \(z < 0\) 时左边 \(=0\)，右边可能是负数）。因此 \(\mathbf{y} = W_2 \cdot \text{ReLU}(W_1\mathbf{x})\) 不能表示为 \(\mathbf{x}\) 的线性函数。但它是**分段线性的**——输入空间被 \(W_1\) 的各行切分成多个区域，每个区域内 \(\text{ReLU}\) 的激活模式固定，\(\mathbf{y}\) 在该区域内是一个线性函数（见 B3）。
    67|    67|
    68|    68|---
    69|    69|
    70|    70|**A4.** 对于 $A = \begin{bmatrix}5&2\\2&5\end{bmatrix}$ 和 $P = \begin{bmatrix}1&1\\1&-1\end{bmatrix}$，验证 $B = P^{-1}AP$ 与 $A$ 有相同的迹、行列式和特征值。
    71|    71|
    72|    72|> **标准答案：**
    73|    73|> \(A\) 的参数：\(\text{tr}(A) = 10\)，\(\det(A) = 25-4 = 21\)。特征方程 \(\lambda^2 - 10\lambda + 21 = 0\) → \((\lambda-3)(\lambda-7) = 0\) → \(\lambda = 3, 7\)。
    74|    74|> 计算 \(B\)：
    75|    75|> \[P^{-1} = \frac{1}{2}\begin{bmatrix}1&1\\1&-1\end{bmatrix}\]
    76|    76|> \[AP = \begin{bmatrix}5&2\\2&5\end{bmatrix}\begin{bmatrix}1&1\\1&-1\end{bmatrix} = \begin{bmatrix}7&3\\7&-3\end{bmatrix}\]
    77|    77|> \[B = P^{-1}AP = \frac{1}{2}\begin{bmatrix}1&1\\1&-1\end{bmatrix}\begin{bmatrix}7&3\\7&-3\end{bmatrix} = \frac{1}{2}\begin{bmatrix}14&0\\0&6\end{bmatrix} = \begin{bmatrix}7&0\\0&3\end{bmatrix}\]
    78|    78|> \(B\) 恰好是对角矩阵！因为 \(P\) 的列恰好是 \(A\) 的特征向量：\((1,1)^{\mathsf T}\) 对应 \(\lambda=7\)，\((1,-1)^{\mathsf T}\) 对应 \(\lambda=3\)。相似变换把 \(A\) 对角化了。
    79|    79|> 验证：\(\text{tr}(B) = 7+3 = 10\) ✓。\(\det(B) = 7\times3 = 21\) ✓。\(B\) 的特征值正是 \(7, 3\) ✓。
    80|    80|> **ML 洞察**：\(P^{-1}AP\) 和 \(A\) 表示相同的线性变换，只是换了「坐标系」。当新基恰好是特征向量时，矩阵变成对角——这就是 PCA 的本质：找到协方差矩阵的特征向量作为新基，使数据在新基下协方差为对角矩阵（各维度独立）。
    81|    81|
    82|    82|---
    83|    83|
    84|    84|**A5.** 对于两层网络（无偏置，MSE）：
    85|    85|\[\mathbf{h} = W_1 \mathbf{x}, \quad \hat{y} = W_2 \mathbf{h}, \quad L = \frac{1}{2}(\hat{y} - y)^2.\]
    86|    86|$W_1 = [1, 2]$，$W_2 = [3]$，$\mathbf{x} = (1, 1)^{\mathsf T}$，$y = 5$。
    87|    87|
    88|    88|(1) 手算前向传播：$\mathbf{h}, \hat{y}, L$。
    89|    89|(2) 手算 $\frac{\partial L}{\partial W_2}$ 和 $\frac{\partial L}{\partial W_1}$（用链式法则）。
    90|    90|
    91|    91|> **标准答案：**
    92|    92|>
    93|    93|> **(1)** 前向：
    94|    94|> \[\mathbf{h} = W_1\mathbf{x} = [1, 2]\begin{bmatrix}1\\1\end{bmatrix} = 1\cdot1 + 2\cdot1 = 3\]
    95|    95|> \[\hat{y} = W_2 \cdot \mathbf{h} = 3 \cdot 3 = 9\]
    96|    96|> \[L = \frac{1}{2}(9 - 5)^2 = \frac{1}{2} \cdot 16 = 8\]
    97|    97|>
    98|    98|> **(2)** 反向传播（链式法则）：
    99|    99|> ① \(\frac{\partial L}{\partial \hat{y}} = \hat{y} - y = 9 - 5 = 4\)
   100|   100|> ② \(\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial W_2}\)。\(\frac{\partial \hat{y}}{\partial W_2} = \mathbf{h} = 3\)。所以：
   101|   101|> \[\frac{\partial L}{\partial W_2} = 4 \cdot 3 = 12\]
   102|   102|> ③ \(\frac{\partial L}{\partial \mathbf{h}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial \mathbf{h}}\)。\(\frac{\partial \hat{y}}{\partial \mathbf{h}} = W_2^{\mathsf T} = 3\)。所以：
   103|   103|> \[\frac{\partial L}{\partial \mathbf{h}} = 4 \cdot 3 = 12\]
   104|   104|> ④ \(\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial \mathbf{h}} \cdot \frac{\partial \mathbf{h}}{\partial W_1}\)。\(\frac{\partial \mathbf{h}}{\partial W_1} = \mathbf{x}^{\mathsf T}\)（列向量变成行向量外积）。所以：
   105|   105|> \[\frac{\partial L}{\partial W_1} = 12 \cdot [1, 1] = [12, 12]\]
   106|   106|> **直观检查**：\(\hat{y}=9\) 比 \(y=5\) 大 \(4\)（误差 \(=4\)）。\(W_2=3\) 把 \(\mathbf{h}\) 放大了 3 倍 → 梯度反向流过时也被放大 3 倍 → \(\frac{\partial L}{\partial \mathbf{h}}=12\)。\(\frac{\partial L}{\partial W_1}\) 的两个分量都是 \(12\)，因为 \(\mathbf{x} = (1,1)\) 的两个输入贡献相同。
   107|   107|> Python 验证：
   108|   108|> ```python
   109|   109|> import torch
   110|   110|> W1 = torch.tensor([[1., 2.]], requires_grad=True)
   111|   111|> W2 = torch.tensor([[3.]], requires_grad=True)
   112|   112|> x = torch.tensor([1., 1.]).unsqueeze(1)
   113|   113|> loss = 0.5 * (W2 @ (W1 @ x) - 5.).pow(2).sum()
   114|   114|> loss.backward()
   115|   115|> print(W1.grad, W2.grad)  # tensor([[12., 12.]])  tensor([[12.]])
   116|   116|> ```
   117|   117|
   118|   118|---
   119|   119|
   120|   120|### B 档
   121|   121|
   122|   122|**B1.** 为什么说 Embedding 层本质上是基变换？预训练学到的 Embedding 矩阵的列向量的几何含义是什么？
   123|   123|
   124|   124|> **标准答案：**
   125|   125|> Embedding 矩阵 \(W \in \mathbb{R}^{d \times |V|}\) 将一个 one-hot 向量 \(\mathbf{e}_i\)（标准基向量）映射为稠密向量 \(\mathbf{w}_i = W\mathbf{e}_i\)（\(W\) 的第 \(i\) 列）。
   126|   126|> **基变换视角**：
   127|   127|> - 标准基（one-hot）中，每个 token 是彼此正交、距离相等的——完全没有语义信息
   128|   128|> - \(W\) 的 \(|V|\) 个列向量定义了一个从 \(\mathbb{R}^{|V|}\)（one-hot 空间）到 \(\mathbb{R}^d\)（语义空间）的线性映射
   129|   129|> - 学到的 \(W\) 的列是「语义基向量」——在这个基下：
   130|   130|>   - 语义相近的词向量夹角小（内积大）
   131|   131|>   - 语义关系的「方向差」是线性的：\(\mathbf{w}_{\text{国王}} - \mathbf{w}_{\text{男人}} + \mathbf{w}_{\text{女人}} \approx \mathbf{w}_{\text{王后}}\)
   132|   132|>
   133|   133|> **几何含义**：每个列向量是该 token 在 \(d\) 维语义空间中的「坐标」。好的 Embedding 找到了一组基，使得**语义方向在该基下线性可分**。这就是分布式表示（distributed representation）的线性代数本质。
   134|   134|> 从压缩角度看：\(W\) 是一个 \(d \times |V|\) 矩阵（\(d \ll |V|\)）。它的列秩 \(\le d \ll |V|\) → 语义信息是低秩的。这也是为什么 Embedding 可以压缩（如从 \(|V| \times 768\) 降到 \(|V| \times 256\)）——真正独立的「语义方向」远少于词汇量。
   135|   135|
   136|   136|---
   137|   137|
   138|   138|**B2.** 为什么 Layer Normalization 放在激活函数**前面**（Pre-LN）比**后面**（Post-LN）效果更好？从投影几何角度解释。
   139|   139|
   140|   140|> **标准答案：**
   141|   141|> 设一层计算为 \(\mathbf{h} = \sigma(W\mathbf{x} + \mathbf{b})\)。
   142|   142|> **Pre-LN**（激活前 LN，标准 Transformer 做法）：
   143|   143|> \[\mathbf{z} = W\mathbf{x} + \mathbf{b}, \quad \tilde{\mathbf{z}} = \frac{\mathbf{z} - \mu}{\sigma} \odot \gamma + \beta, \quad \mathbf{h} = \sigma(\tilde{\mathbf{z}})\]
   144|   144|> LN 在仿射空间中规范化：消除不同样本间均值和方差的偏移（internal covariate shift），让激活函数接收到「规矩」的输入——分布稳定、尺度统一。
   145|   145|> **Post-LN**（激活后 LN）：
   146|   146|> \[\mathbf{h}_{\text{raw}} = \sigma(W\mathbf{x} + \mathbf{b}), \quad \mathbf{h} = \text{LN}(\mathbf{h}_{\text{raw}})\]
   147|   147|> 问题：激活函数（如 ReLU）已经改变了分布的形状（截断负值，正半轴保持）。此时再 LN 试图恢复被激活函数扭曲的统计特性 → 更差的训练稳定性和收敛速度。
   148|   148|> **投影几何视角**：LN 将每个样本的特征向量投影到半径为 \(\sqrt{d}\) 的球面上。**Pre-LN 在进入非线性之前统一了所有样本的「尺度」**，确保无论 \(W\mathbf{x}\) 多大（或经过多少层累积），下一层的输入始终在球面上。这使各层的有效「操作范围」一致——前向信号和反向梯度都不爆炸或消失。
   149|   149|> LN 比 BatchNorm 更适合序列模型：BN 沿 batch 维度归一化（假设样本 i.i.d.），LN 沿 feature 维度归一化（逐样本独立）。序列 token 之间不满足 i.i.d. → LN 天然适用。
   150|   150|
   151|   151|---
   152|   152|
   153|   153|**B3.** 一个两层 ReLU 网络：$\mathbf{h} = \text{ReLU}(W_1 \mathbf{x} + \mathbf{b}_1)$，$\hat{y} = W_2 \mathbf{h} + b_2$。$W_1 \in \mathbb{R}^{3 \times 2}$。
   154|   154|
   155|   155|(1) 第一个隐藏神经元在什么条件下激活？在 $\mathbb{R}^2$ 中对应什么几何对象？
   156|   156|(2) 三个神经元定义的线把平面最多分成几个区域？每区域内 $\hat{y}$ 的形式是什么？
   157|   157|(3) 这如何说明「深度网络 = 分段线性函数」？更深 vs 更宽的区别是什么？
   158|   158|
   159|   159|> **标准答案：**
   160|   160|>
   161|   161|> **(1)** 第 \(i\) 个神经元的激活条件：
   162|   162|> \[\mathbf{w}_i^{\mathsf T}\mathbf{x} + b_i > 0\]
   163|   163|> 其中 \(\mathbf{w}_i^{\mathsf T}\) 是 \(W_1\) 的第 \(i\) 行。在 \(\mathbb{R}^2\) 中，\(\mathbf{w}_i^{\mathsf T}\mathbf{x} + b_i = 0\) 是一条**直线**，激活条件是直线的一侧。神经元 \(i\) 把平面切成两个半平面。
   164|   164|>
   165|   165|> **(2)** 3 条直线最多把平面分成 \(1 + 3 + 3 = 7\) 个区域（一般 \(k\) 条直线最多 \(1 + k(k+1)/2\) 个区域，这里 \(1+3+3=7\) 是因为最多交点数为 3 个时成立）。
   166|   166|> 在每个区域内，哪些神经元激活（输出 \(>0\)）是**固定的**。设激活模式为 \(\mathbf{m} \in \{0,1\}^3\)，则：
   167|   167|> \[\hat{y} = W_2 \cdot \text{diag}(\mathbf{m}) \cdot (W_1\mathbf{x} + \mathbf{b}_1) + b_2 = \tilde{W}\mathbf{x} + \tilde{b}\]
   168|   168|> 即每个区域内 \(\hat{y}\) 是一个不同的**线性函数**。穿过边界（神经元开关翻转）时，线性函数跳变。
   169|   169|>
   170|   170|> **(3)** ReLU 网络的本质：**输入空间上的分段线性函数**。
   171|   171|> - **更深**：每增加一层，新的超平面与已存在折痕相交，在折痕上产生新折痕 → 分段数随深度**指数增长**。深层窄网络可以用很少神经元产生极多分段 → 高效表达复杂函数。
   172|   172|> - **更宽**：单层只是增加超平面数量 → 分段数随宽度**多项式增长**。宽而浅不如深而窄高效。
   173|   173|> 这就是万能逼近定理的构造性直觉：足够多的线性区域 → 可以以任意精度逼近任何连续函数。
   174|   174|
   175|   175|---
   176|   176|
   177|   177|**B4.** 以下场景中 PCA 在做一种「基变换」。为每个场景说明原始基、新基、保留哪些维度：
   178|   178|
   179|   179|(1) 词嵌入矩阵 $N \times 768$ 降到 256 维。
   180|   180|(2) LoRA 对大模型权重做 $W \approx BA$（$B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times d}$）。
   181|   181|(3) 图像去噪：784 维图像 PCA 保留 $k$ 个主成分后重建。
   182|   182|
   183|   183|> **标准答案：**
   184|   184|>
   185|   185|> **(1)**
   186|   186|> - 原始基：标准基（one-hot），每个 token 是正交的 \(|V|\) 维向量
   187|   187|> - 新基：协方差矩阵 \(X^{\mathsf T}X\) 的前 \(k\) 个特征向量——词向量分布方差最大的 \(k\) 个「语义方向」
   188|   188|> - 保留前 256 个主成分（最大特征值），丢弃其余 → 保留 \(>95\%\) 方差
   189|   189|>
   190|   190|> **(2)**
   191|   191|> - 原始基：权重空间 \(\mathbb{R}^{d \times d}\) 的标准基（每个参数一个独立维度）
   192|   192|> - 新基：\(B\) 的 \(r\) 列是 \(r\) 个重要方向的基向量，\(A\) 的行是 \(r\) 个方向的「坐标」
   193|   193|> - LoRA 不做显式 PCA，但思想同源：假设 \(\Delta W\) 低秩 → 只在 \(r\) 个最重要方向上更新（\(r=8\sim64 \ll d\)），丢弃 \(d-r\) 个方向
   194|   194|>
   195|   195|> **(3)**
   196|   196|> - 原始基：784 个像素的标准基（每个像素一个维度）
   197|   197|> - 新基：协方差矩阵的特征向量 = 特征脸（eigenfaces）——像素之间的相关模式。前几个特征脸是低频大结构，后面的越来越精细（皱纹、纹理）
   198|   198|> - 保留前 \(k\) 个特征脸（\(k \ll 784\)），后 \(784-k\) 个丢弃。**噪声集中在高方差方向？不——噪声集中在低方差方向（小特征值方向）。** 保留大方差方向 = 保留结构 + 丢弃噪声（因为噪声在各方向大致等量，结构的信噪比在大方差方向最高）
   199|   199|
   200|   200|---
   201|   201|
   202|   202|**B5.** Adam 优化器可理解为对角预处理。从线性代数角度解释：
   203|   203|
   204|   204|(1) 为什么 SGD 对所有参数用同一个学习率会在 ill-conditioned 问题上失败？
   205|   205|(2) Adam 的 $v_t$ 在做什么？和牛顿法的 $H^{-1}g$ 相比做了什么近似？
   206|   206|
   207|   207|> **标准答案：**
   208|   208|>
   209|   209|> **(1)** SGD：\(\theta_{t+1} = \theta_t - \eta g_t\)。学习率 \(\eta\) 受最大 Hessian 特征值 \(\lambda_{\max}\) 限制（\(\eta < 2/\lambda_{\max}\) 防震荡）。但在 \(\lambda_{\min}\) 方向上，每步衰减因子 \(1 - \eta\lambda_{\min} \approx 1 - \frac{2\lambda_{\min}}{\lambda_{\max}}\)，当 \(\kappa = \lambda_{\max}/\lambda_{\min}\) 很大时，该方向几乎不动。SGD 被「最陡方向」绑架了学习率，在「最平方向」上寸步难行。
   210|   210|>
   211|   211|> **(2)** Adam 维护 \(v_t \approx \mathbb{E}[g^2]\)（梯度平方的移动平均）。在局部二次近似 \(g \approx H(\theta - \theta^*)\) 下，\(v_t\) 每个分量近似对应 Hessian 对角元的尺度。
   212|   212|> Adam 更新：\(\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}\)。除以 \(\sqrt{\hat{v}_t}\) 是对角预处理：
   213|   213|> \[\text{有效更新} \approx \eta \cdot \text{diag}(H)^{-1/2} \cdot g\]
   214|   214|> **与牛顿法对比**：
   215|   215|> - 牛顿法：\(\theta \leftarrow \theta - H^{-1}g\)。完整的二阶预处理——所有方向归一化到相同曲率。代价 \(O(n^3)\)（求逆）。
   216|   216|> - Adam：用 \(\text{diag}(H)^{-1/2}\) 近似 \(H^{-1}\)。捕获了各方向的**尺度差异**（对角元），忽略了方向间的**耦合**（非对角元）。代价 \(O(n)\)。
   217|   217|> 这是计算效率和预处理质量之间的经典权衡。Adam 在实践中效果极好，因为大量 ML 优化问题中 Hessian 是对角占优的（参数之间的耦合相对较弱）。
   218|   218|
   219|   219|---
   220|   220|
   221|   221|**B6.** 残差连接 $h_{l+1} = h_l + F(h_l)$ 为什么能训练深层网络？
   222|   222|
   223|   223|(1) 写出反向传播的雅可比矩阵，指出和普通层的区别。
   224|   224|(2) 为什么特征值从 $\varepsilon$ 变成 $1 + \varepsilon$ 就防止了梯度消失？
   225|   225|(3) 堆叠 $L$ 个残差块后，梯度乘积展开式长什么样？
   226|   226|
   227|   227|> **标准答案：**
   228|   228|>
   229|   229|> **(1)** 对残差块求导（链式法则）：
   230|   230|> \[\frac{\partial h_{l+1}}{\partial h_l} = \frac{\partial}{\partial h_l}(h_l + F(h_l)) = I + \frac{\partial F}{\partial h_l}\]
   231|   231|> 普通层：雅可比 \(= \partial F/\partial h_l\)。残差层：雅可比 \(= I + \partial F/\partial h_l\)。
   232|   232|> 关键：多了 \(I\) 项。
   233|   233|>
   234|   234|> **(2)** 设无残差时雅可比的特征值都 \(< 1\)（如都在 \((0, 0.8)\) 内）。连乘 \(L\) 层后，梯度受最大特征值支配：
   235|   235|> \[\left\|\frac{\partial h_L}{\partial h_1}\right\| \approx \lambda_{\max}^L\]
   236|   236|> 当 \(\lambda_{\max} = 0.8\)，\(L=100\)：\(0.8^{100} \approx 2 \times 10^{-10}\) → 梯度消失。
   237|   237|> 有残差后，雅可比特征值变为 \(1 + \varepsilon\)（\(\varepsilon\) 是 \(\partial F/\partial h_l\) 的特征值）。即使 \(\varepsilon = -0.2\)，特征值仍为 \(0.8\)——但关键是 \(I\) 保证了**至少有一个特征值为 \(1\) 的路径**。乘积不指数衰减。
   238|   238|>
   239|   239|> **(3)** \(L\) 层梯度乘积：
   240|   240|> \[\frac{\partial h_L}{\partial h_1} = \prod_{l=1}^{L-1} \left(I + \frac{\partial F_l}{\partial h_l}\right)\]
   241|   241|> 展开：
   242|   242|> \[= I + \sum_{l=1}^{L-1} \frac{\partial F_l}{\partial h_l} + \sum_{l<m} \frac{\partial F_l}{\partial h_l}\frac{\partial F_m}{\partial h_m} + \cdots\]
   243|   243|> 第一项 \(I\)：梯度的**无损直通路径**（信息从第 \(L\) 层直达第 \(1\) 层）。
   244|   244|> 第二项：每层的独立贡献（一阶）。
   245|   245|> 高阶项：跨层交互。
   246|   246|> **每一层的梯度信号都包含来自所有后续层的直接贡献。** 信息和梯度可以在网络中「无损穿行」。这就是 ResNet-152 和 Transformer 96 层可训练的根本原因。
   247|   247|
   248|   248|---
   249|   249|
   250|   250|**B7.** 推导 Xavier 初始化和 He 初始化的方差公式。
   251|   251|
   252|   252|(1) 对于线性层 $h = Wx$，要使 $\text{Var}(h) = \text{Var}(x)$，推导 $\text{Var}(W)$ 的条件。
   253|   253|(2) 为什么反向传播给出不同的约束？
   254|   254|(3) Xavier 为什么取调和平均？He 为什么改成 $2/n_{\text{in}}$？
   255|   255|
   256|   256|> **标准答案：**
   257|   257|> 假设：\(W_{ij}\) 和 \(x_j\) 独立同分布，均值为 \(0\)。
   258|   258|>
   259|   259|> **(1)** 前向方差：
   260|   260|> \[h_i = \sum_{j=1}^{n_{\text{in}}} W_{ij} x_j\]
   261|   261|> \[\text{Var}(h_i) = \sum_{j=1}^{n_{\text{in}}} \text{Var}(W_{ij}) \cdot \text{Var}(x_j) = n_{\text{in}} \cdot \text{Var}(W) \cdot \text{Var}(x)\]
   262|   262|> 要使 \(\text{Var}(h_i) = \text{Var}(x)\)：\(\text{Var}(W) = 1/n_{\text{in}}\)。
   263|   263|>
   264|   264|> **(2)** 反向传播：\(\frac{\partial L}{\partial x} = W^{\mathsf T} \frac{\partial L}{\partial h}\)。
   265|   265|> \[\frac{\partial L}{\partial x_j} = \sum_{i=1}^{n_{\text{out}}} W_{ij} \cdot \frac{\partial L}{\partial h_i}\]
   266|   266|> \[\text{Var}\left(\frac{\partial L}{\partial x_j}\right) = n_{\text{out}} \cdot \text{Var}(W) \cdot \text{Var}\left(\frac{\partial L}{\partial h}\right)\]
   267|   267|> 保持梯度方差不变：\(\text{Var}(W) = 1/n_{\text{out}}\)。
   268|   268|>
   269|   269|> **(3)** 前向要求 \(1/n_{\text{in}}\)，反向要求 \(1/n_{\text{out}}\)。当 \(n_{\text{in}} \neq n_{\text{out}}\) 时冲突。
   270|   270|> **Xavier**：取两者调和平均 → \(\text{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}}\)。折中方案，同时近似满足前向和反向的方差约束。
   271|   271|> **He**：考虑 ReLU。ReLU 在前向传播中把负激活值置零 → 大约一半的神经元输出为 \(0\) → 前向方差减半：
   272|   272|> \[\text{Var}(h_i) = \frac{1}{2} \cdot n_{\text{in}} \cdot \text{Var}(W) \cdot \text{Var}(x)\]
   273|   273|> 补偿 \(\frac{1}{2}\) 因子 → \(\text{Var}(W) = 2/n_{\text{in}}\)。
   274|   274|> 实践中 PyTorch 的 `nn.Linear` 默认使用 He（Kaiming）uniform 初始化。
   275|   275|
   276|   276|---
   277|   277|
   278|   278|**B8.** 1D 卷积 $y = w * x$ 可写成 $y = Cx$，其中 $C$ 是 Toeplitz 矩阵。
   279|   279|
   280|   280|(1) 写出 $w = (w_0, w_1, w_2)$，输入长 $4$（zero-padding $p=1$）对应的 Toeplitz 矩阵 $C$。
   281|   281|(2) 参数共享如何体现在矩阵结构中？参数量的节省有多大？
   282|   282|(3) 转置卷积等价于什么操作？
   283|   283|
   284|   284|> **标准答案：**
   285|   285|>
   286|   286|> **(1)** Zero-padding 后输入变为 \(\tilde{x} = (0, x_1, x_2, x_3, x_4, 0)^{\mathsf T}\)。\(y = w * x\)（same 模式，输出长 \(4\)）：
   287|   287|> \[y_1 = w_1 \cdot 0 + w_0 \cdot x_1 + w_2 \cdot x_2\]
   288|   288|> 等等——让我理清索引。设 \(w = (w_0, w_1, w_2)\)。带 padding \(p=1\) 的卷积对应：
   289|   289|> \[C = \begin{bmatrix} w_1 & w_0 & 0 & 0 \\ w_2 & w_1 & w_0 & 0 \\ 0 & w_2 & w_1 & w_0 \\ 0 & 0 & w_2 & w_1 \end{bmatrix}\]
   290|   290|> 验证：\(y_1 = w_1x_1 + w_0x_2\)（对应 \(w\) 翻转后滑动：\(w_2\) 超出，\(w_1\) 在 \(x_1\)，\(w_0\) 在 \(x_2\)）。
   291|   291|>
   292|   292|> **(2)** 参数共享：同一条对角线上的元素全相同。例如 \(w_0\) 出现在 \(C_{12}, C_{23}, C_{34}\)——同一个参数被复用了 \(3\) 次。参数量 \(=3\)，对应稠密矩阵需要 \(4 \times 4 = 16\) 个参数 → 节省 \(>5\times\)。
   293|   293|> 图像上：\(3 \times 3\) 卷积核在 \(224 \times 224\) 特征图上，参数 \(= C_{\text{in}} \times C_{\text{out}} \times 9\)，等效稠密参数量 \(= C_{\text{in}} \times C_{\text{out}} \times 224^2\) → 节省 \(\sim 5500\times\)。
   294|   294|>
   295|   295|> **(3)** 转置卷积计算 \(y = C^{\mathsf T} \mathbf{x}\)。\(C^{\mathsf T}\) 也是 Toeplitz，但结构对应「反卷」：
   296|   296|> \[C^{\mathsf T} = \begin{bmatrix} w_1 & w_2 & 0 & 0 \\ w_0 & w_1 & w_2 & 0 \\ 0 & w_0 & w_1 & w_2 \\ 0 & 0 & w_0 & w_1 \end{bmatrix}\]
   297|   297|> 这等价于**用翻转后的卷积核做带空洞的上采样**。`nn.ConvTranspose1d` 底层就是 \(C^{\mathsf T} \mathbf{x}\)。理解和卷积的矩阵对偶关系，是理解 GAN 生成器和语义分割解码器的关键。
   298|   298|
   299|   299|---
   300|   300|
   301|   301|**B9.** Self-Attention $S = \text{softmax}(QK^{\mathsf T}/\sqrt{d})$，$Q,K \in \mathbb{R}^{n \times d}$。
   302|   302|
   303|   303|(1) $S$ 的秩最大是多少？当 $d=64$，$n=512$，这意味着什么？
   304|   304|(2) 为什么实际 Transformer 中 $S$ 常常更低秩（attention collapse）？
   305|   305|(3) Multi-head attention 如何缓解这个问题？
   306|   306|
   307|   307|> **标准答案：**
   308|   308|>
   309|   309|> **(1)** \(QK^{\mathsf T}\) 的秩 \(\le \min(\text{rank}(Q), \text{rank}(K)) \le \min(n, d) = d\)。Softmax 是逐行操作（指数 + 归一化），不改变秩。所以 \(\text{rank}(S) \le 64\)。
   310|   310|> \(S\) 是 \(512 \times 512\)，但内蕴维度仅 \(\le 64\)——它是**极度低秩**的。相当于用 \(64\) 个「注意力模式」去近似 \(512\) 个 token 的所有 pairwise 关系。
   311|   311|>
   312|   312|> **(2) Attention Collapse**：实际秩常远低于 \(64\)，因为：
   313|   313|> - token 的 query/key 向量高度相关 → \(Q\) 和 \(K\) 自身的有效秩 \(< d\)
   314|   314|> - 很多 token 关注相同的 context（如所有 token 都高度关注 CLS token 或分隔符）→ attention 行向量趋于相同
   315|   315|> - 从 SVD 看：\(\sigma_1 \gg \sigma_2, \sigma_3, \dots\)，主导奇异值占了绝大部分能量。\(S \approx \sigma_1 \mathbf{u}_1 \mathbf{v}_1^{\mathsf T}\)（秩-1），所有 token 的关注分布几乎相同
   316|   316|>
   317|   317|> **(3)** Multi-head attention：\(h\) 个 head，每个 head 用不同的投影矩阵 \(W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}\)：
   318|   318|> \[\text{head}_i = \text{Attention}(XW_Q^{(i)}, XW_K^{(i)}, XW_V^{(i)})\]
   319|   319|> 每个 head 的 \(Q_i K_i^{\mathsf T}\) 可能有不同的低秩结构（rank \(\le d/h\)）→ 各自关注不同类型的关系。拼接后 \(\text{rank}(\text{Concat}) \le \sum_i \text{rank}(\text{head}_i)\) → 获得秩的多样性。如果单 head 是秩-1 的，8 heads 组合可达秩-8——大幅提升 attention 模式的丰富度。
   320|   320|
   321|   321|---
   322|   322|
   323|   323|**B10.** 线性回归 $y \approx X\beta$ 的正规方程为 $X^{\mathsf T}X\beta = X^{\mathsf T}y$。
   324|   324|
   325|   325|(1) $X^{\mathsf T}X$ 的秩和 $X$ 的秩有什么关系？什么情况下 $X^{\mathsf T}X$ 奇异？
   326|   326|(2) 当 $X^{\mathsf T}X$ 奇异时，正规方程有无穷多解。为什么实际中往往还加 L2 正则？
   327|   327|(3) 从条件数角度证明：$(X^{\mathsf T}X + \lambda I)^{-1}$ 比 $(X^{\mathsf T}X)^{-1}$ 更稳定。
   328|   328|
   329|   329|> **标准答案：**
   330|   330|>
   331|   331|> **(1)** \(\text{rank}(X^{\mathsf T}X) = \text{rank}(X)\)。证明：\(\text{rank}(X^{\mathsf T}X) \le \text{rank}(X)\)（显然），且 \(X^{\mathsf T}X\mathbf{v} = 0 \Rightarrow \mathbf{v}^{\mathsf T}X^{\mathsf T}X\mathbf{v} = \|X\mathbf{v}\|^2 = 0 \Rightarrow X\mathbf{v} = 0\)，所以零空间相同 → 秩相同。
   332|   332|> \(X^{\mathsf T}X\) 奇异当且仅当 \(X\) 不满列秩——即特征之间存在线性相关（多重共线性），或样本数 \(<\) 特征数（经典 \(p > n\) 问题）。
   333|   333|>
   334|   334|> **(2)** 当 \(X^{\mathsf T}X\) 奇异时，正规方程有无穷多解。L2 正则化（Ridge Regression）：
   335|   335|> \[\hat{\beta} = (X^{\mathsf T}X + \lambda I)^{-1} X^{\mathsf T}y\]
   336|   336|> 给 \(X^{\mathsf T}X\) 的对角线上加 \(\lambda I\) → 所有特征值抬升 \(\lambda\) → 强制正定 → 唯一解。从贝叶斯角度，这等价于给 \(\beta\) 加高斯先验 \(\beta \sim \mathcal{N}(0, \lambda^{-1}I)\)。
   337|   337|>
   338|   338|> **(3)** 设 \(X^{\mathsf T}X\) 的特征值为 \(\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_r > 0\)（可能 \(\lambda_r\) 接近 \(0\)）。
   339|   339|> 无正则化：
   340|   340|> \[\kappa(X^{\mathsf T}X) = \frac{\lambda_1}{\lambda_r}\]
   341|   341|> 若 \(\lambda_r \approx 10^{-6}\)，\(\kappa \approx 10^6 \lambda_1\) → 极度病态。
   342|   342|> 加正则化后：
   343|   343|> \[\kappa(X^{\mathsf T}X + \lambda I) = \frac{\lambda_1 + \lambda}{\lambda_r + \lambda}\]
   344|   344|> \(\lambda_r \approx 0\) 时，分母 \(\approx \lambda\) → \(\kappa \approx \frac{\lambda_1 + \lambda}{\lambda} \le \frac{\lambda_1}{\lambda} + 1\)。条件数被**有界化**，不再随 \(\lambda_r \to 0\) 爆炸。这就是 Ridge Regression 数值稳定的线性代数本质。
   345|   345|
   346|   346|---
   347|   347|
   348|   348|[← 返回教程](basis-and-neural-networks.md)　　　[← 习题总览](exercises.md)