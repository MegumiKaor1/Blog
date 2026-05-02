# 三、行列式、特征值与 SVD

## 9. 行列式：一次变换改变多少体积

### 9.1 核心直觉：体积的缩放因子

矩阵 $A$ 把空间里的单位立方体变成某个平行多面体。$\det(A)$ 就是这个多面体的**有向体积**。

拿二维来看最直观。$A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$ 把单位正方形（顶点 $(0,0), (1,0), (0,1), (1,1)$）变成由列向量 $\begin{bmatrix} a \\ c \end{bmatrix}$ 和 $\begin{bmatrix} b \\ d \end{bmatrix}$ 张成的平行四边形。

$$\det(A) = ad - bc$$

这就是平行四边形的**有向面积**。绝对值 = 面积，符号 = 朝向（正 = 没有翻转，负 = 翻转了）。

三维：$\det(A)$ = 三个列向量张成的平行六面体的有向体积。$n$ 维同理。

### 9.2 行列式 = 0 的含义

$\det(A) = 0$ 意味着多面体被压扁了——体积为 0。

在二维：两列共线，四边形退化成线段 → 面积 = 0 → 矩阵不可逆。
在三维：三列共面（或更糟），六面体退化成平面 → 体积 = 0 → 矩阵不可逆。

> $\det(A) = 0 \iff A$ **不可逆** $\iff$ **秩亏**。这三个条件等价。

### 9.3 核心性质

| 性质 | 直觉 |
|------|------|
| $\det(AB) = \det(A) \det(B)$ | 先 $B$ 后 $A$，体积缩放因子相乘 |
| $\det(A^{-1}) = 1/\det(A)$ | 逆变换把体积放回去 |
| $\det(A^T) = \det(A)$ | 转置不改变体积 |
| $\det(cA) = c^n \det(A)$ | 每个维度各缩放 $c$ 倍 |
| 交换两行 → 行列式变号 | 翻转了空间 |

### 9.4 在 ML 中的影子

你不会经常手算行列式，但它的幽灵一直在：

- **协方差矩阵的行列式（广义方差）**：衡量数据分布占据的「体积」。行列式趋于 0 → 特征高度共线 → 数据冗余严重
- **概率密度里的归一化常数**：多元高斯分布 $\mathcal{N}(\mu, \Sigma)$ 的密度含 $|\Sigma|^{-1/2}$。$\det(\Sigma)$ 极小 → 分布极度「扁平」
- **Hessian 行列式**：$\det(H) \approx 0$ 意味着损失曲面在某个方向几乎平坦 → 梯度下降慢或卡住

---

## 10. 特征值与特征向量：发现矩阵的「天然方向」

### 10.1 核心直觉

矩阵在大多数方向上会把向量扭得面目全非。但在某些**特殊方向**上，矩阵的作用仅仅是拉伸：

$$Ax = \lambda x$$

- $x$ 叫**特征向量**——矩阵的「天然方向」，「无论怎么变换，方向不变」
- $\lambda$ 叫**特征值**——这个方向上的拉伸倍数

### 10.2 一个具体例子

$$A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$$

解 $\det(A - \lambda I) = 0$：

$$\det\begin{bmatrix} 2-\lambda & 1 \\ 1 & 2-\lambda \end{bmatrix} = (2-\lambda)^2 - 1 = \lambda^2 - 4\lambda + 3 = 0$$

$\lambda_1 = 3$，$\lambda_2 = 1$。

对应特征向量：
- $\lambda_1 = 3$：$(A-3I)x = 0 \implies \begin{bmatrix} -1 & 1 \\ 1 & -1 \end{bmatrix}x = 0 \implies x_1 = c\begin{bmatrix} 1 \\ 1 \end{bmatrix}$
- $\lambda_2 = 1$：$(A-I)x = 0 \implies \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}x = 0 \implies x_2 = c\begin{bmatrix} -1 \\ 1 \end{bmatrix}$

**解释**：这个矩阵在 $\begin{bmatrix} 1 \\ 1 \end{bmatrix}$ 方向上拉伸 3 倍，在 $\begin{bmatrix} -1 \\ 1 \end{bmatrix}$ 方向上拉伸 1 倍（不变）。两个方向恰好**正交**——因为 $A$ 是对称矩阵。

### 10.3 对角化：换个坐标系看

把特征向量排成列，组成矩阵 $S$：
$$A = S \Lambda S^{-1}$$

$$\begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} = \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}^{-1}$$

含义：**在特征向量构成的坐标系下，$A$ 就是一个纯拉伸（对角矩阵）**。复杂的变换只是因为观察的「姿势」不对。

### 10.4 对称矩阵：ML 中的常客

对称矩阵（$A = A^T$）有绝佳的性质：
- 所有特征值都是**实数**
- 特征向量可以选为**正交的** → 谱分解 $A = Q \Lambda Q^T$（$Q$ 是正交阵）

协方差矩阵、Gram 矩阵、Hessian、图拉普拉斯——ML 里到处是对称矩阵。

### 10.5 正定性——梯度下降的地形

| 类型 | 特征值 | 几何形状 | 梯度下降体验 |
|------|--------|---------|------------|
| **正定** | 全部 $> 0$ | 碗 🥣 | 一路顺滑到底 |
| **半正定** | 全部 $\ge 0$ | 有平坦方向的碗 | 有些方向不提供梯度信息 |
| **不定** | 有正有负 | 马鞍 🐴 | 有的方向要上升，有的要下降——挣扎 |
| **负定** | 全部 $< 0$ | 倒扣的碗 🙃 | 梯度下降往「上」走（需要梯度上升） |

> 💡 **Hessian 正定 $\implies$ 你到了局部极小值**（碗底）。Hessian 不定 → 你在马鞍点 → 某些方向还没到达最优。

---

## 11. 奇异值分解 SVD：最普适的分解

### 11.1 特征值分解的致命限制——和 SVD 如何突破

特征值分解要求方阵。但现实中的矩阵几乎都是长方形——1000 个样本 × 768 维特征。而且非对称矩阵的特征向量不一定正交，对角化可能不存在。

SVD 没有任何限制。**任何 $m \times n$ 矩阵 $A$**（秩 $r$）：

$$A = U \Sigma V^T$$

- $U$（$m \times m$）：**正交**。前 $r$ 列 = 列空间的标准正交基，后 $m-r$ 列 = 左零空间的基
- $\Sigma$（$m \times n$）：**对角**。$\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$（奇异值），之后全是 0
- $V$（$n \times n$）：**正交**。前 $r$ 列 = 行空间的标准正交基，后 $n-r$ 列 = 零空间的基

**非零奇异值的个数 = 秩 $r$。** SVD 一次性给你：秩、四个子空间的完美基、每个方向的重要性（奇异值就是重要性评分）。

### 11.2 一个具体的 SVD

$$A = \begin{bmatrix} 1 & 2 \\ 2 & 1 \\ 2 & 2 \end{bmatrix}$$

$A^T A = \begin{bmatrix} 9 & 6 \\ 6 & 9 \end{bmatrix}$。特征值 $\lambda_1 = 15, \lambda_2 = 3$。奇异值 $\sigma_1 = \sqrt{15} \approx 3.873, \sigma_2 = \sqrt{3} \approx 1.732$。

- $\sigma_1$ 对应的 $v_1 = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \end{bmatrix}$——**$A$ 拉伸最多的方向**
- $\sigma_2$ 对应的 $v_2 = \frac{1}{\sqrt{2}}\begin{bmatrix} -1 \\ 1 \end{bmatrix}$——第二个重要方向

奇异值 3.873 vs 1.732，前者贡献的信息量是后者的 2 倍多。如果做截断（保留第一个），用秩-1 近似：

$$A_1 = \sigma_1 u_1 v_1^T$$

保留了大部分信息，丢掉了 $\sigma_2$ 方向的细节。

### 11.3 几何含义：旋转 → 拉伸 → 旋转

$A$ 作用于任何向量的三部曲：

1. $V^T$：旋转（转到 $A$ 的天然坐标系——$v_1$ 方向最重要，$v_2$ 次之……）
2. $\Sigma$：沿坐标轴各自拉伸 $\sigma_i$ 倍（$i > r$ 则拉伸 0 → 压扁）
3. $U$：再旋转（转到原始输出坐标系）

**任何线性变换的本质就这三步。没有例外。** 这个视角统一了所有矩阵运算。

### 11.4 截断 SVD 和 Eckart-Young 定理

只留前 $k$ 个奇异值（$k < r$）：

$$A_k = U_k \Sigma_k V_k^T$$

**Eckart-Young 定理**：$A_k$ 是**所有秩-$k$ 矩阵中在 Frobenius 范数下最接近 $A$ 的那一个**。丢掉的能量 = $\sqrt{\sigma_{k+1}^2 + \dots + \sigma_r^2}$。

这定理是 PCA 的数学根基。PCA 就是 SVD 换了个名字——对中心化数据矩阵 $X$ 做 SVD，右奇异向量 $V_k$ 就是前 $k$ 个主成分。

### 11.5 SVD 在 ML 中的三个高光时刻

| 场景 | SVD 的应用 | 怎么理解 |
|------|-----------|---------|
| **PCA 降维** | $X \approx U_k \Sigma_k V_k^T$ | 丢掉小奇异值对应的方向 = 丢掉噪音，保留结构 |
| **LoRA 微调** | $\Delta W = BA$，$B$ 是 $d \times r$，$A$ 是 $r \times k$ | $r$ 就是你决定保留的「几个最重要的奇异值方向」 |
| **协同过滤/推荐** | $R \approx U_k \Sigma_k V_k^T$ | 评分矩阵的低秩近似——用少量的「隐因子」解释用户偏好 |

---

> **下一步**：[四、基变换与神经网络](basis-and-neural-networks.md) —— 这些数学和你写的 PyTorch 代码之间到底什么关系？

---

## 12. 正定性与二次型：矩阵的「形状」分类学

### 12.1 正定的五种等价判定——线性代数最美的定理之一

§10.5 我们初步见了正定性一面。这里我们把它说透。

对于 $n \times n$ **实对称矩阵** $A$，以下五个命题**完全等价**：

| # | 判定条件 | 直觉 |
|---|---------|------|
| ① | 所有特征值 $> 0$ | 在所有「天然方向」上都是拉伸（不翻转、不压扁） |
| ② | $x^T A x > 0,\ \forall x \neq 0$ | 任何非零向量代入二次型，结果恒正 |
| ③ | 所有主元 $> 0$ | 高斯消元不换行时，每个 pivot 都正 |
| ④ | 所有顺序主子式 $> 0$ | 左上角 $1\times 1, 2\times 2, \dots, n\times n$ 子矩阵行列式全正 |
| ⑤ | $A = R^T R$，其中 $R$ 可逆 | $A$ 可分解为某个可逆矩阵的「Gram 矩阵」 |

> 🏆 这五条等价，被我视为线性代数最美的定理之一。从五个完全不同的视角——特征值、二次型、消元、行列式、矩阵分解——看同一个性质，结论完全一致。这是数学「深层结构」的完美展示。

**⑤ 的深入理解**：$A = R^T R$ 意味着 $x^T A x = x^T R^T R x = \|R x\|^2 \ge 0$，且 $R$ 可逆保证了 $x \neq 0$ 时 $Rx \neq 0$，即 $>0$。从这个视角看，正定矩阵本质上就是一个**可逆矩阵跟自己内积得到的矩阵**。所有协方差矩阵 $X^T X$ 都是（半）正定的——这在 §12.2 会展开。

**Cholesky 分解**：⑤ 中如果取 $R$ 为上三角阵，就是 Cholesky 分解 $A = L L^T$（$L$ 是下三角）。它是正定矩阵的「平方根」，在数值计算中比 LU 快一倍，而且天然稳定。

**ML 视角**：Hessian 在极小值点**必须正定**。如果你的 Hessian 有一个非正的特征值，那你不在极小值——要么在鞍点，要么在平坦区域。SGD 能逃离鞍点（§12.5 详谈），但逃离不了特征值为零的平坦方向。

### 12.2 二次型与标准形——一个坐标变换，消去所有交叉项

#### 二次型是什么

$n$ 元二次齐次多项式就是二次型：

$$f(x) = x^T A x = \sum_{i=1}^n \sum_{j=1}^n a_{ij} x_i x_j$$

展开看：有平方项 $a_{ii} x_i^2$，也有交叉项 $a_{ij} x_i x_j\ (i \neq j)$。交叉项让图形「斜着」——椭圆的轴不与坐标轴对齐。

由于 $x_i x_j = x_j x_i$，总可以写成 $a_{ij} = a_{ji}$，即 $A$ 对称。所以：**每一个二次型都唯一对应一个对称矩阵**。

#### 正交变换 → 标准形（主轴定理）

用正交变换 $x = Qy$（$Q$ 的列是 $A$ 的标准正交特征向量）：

$$f(x) = x^T A x = (Qy)^T A (Qy) = y^T (Q^T A Q) y = y^T \Lambda y = \lambda_1 y_1^2 + \lambda_2 y_2^2 + \dots + \lambda_n y_n^2$$

**交叉项全部消失！** 只留下纯粹的平方项，系数就是特征值。这叫做二次型的**标准形**（canonical form）。这是**主轴定理**（Principal Axis Theorem）的核心结论。

#### 几何直觉：斜着的椭圆，换个角度看就正了

在二维：
$$f(x_1, x_2) = 5x_1^2 + 4x_1 x_2 + 5x_2^2 = x^T \underbrace{\begin{bmatrix} 5 & 2 \\ 2 & 5 \end{bmatrix}}_{A} x$$

特征值：$\lambda_1 = 7, \lambda_2 = 3$；特征向量：$v_1 = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \end{bmatrix}$（45° 方向），$v_2 = \frac{1}{\sqrt{2}}\begin{bmatrix} -1 \\ 1 \end{bmatrix}$（135° 方向）。

在原始 $x_1$-$x_2$ 坐标系中，$f(x)=1$ 是一个**倾斜的椭圆**。在 $y_1$-$y_2$ 坐标系（即特征向量方向）中，$f = 7y_1^2 + 3y_2^2 = 1$——**轴对齐的椭圆**，半轴长为 $1/\sqrt{7}$ 和 $1/\sqrt{3}$。

> 特征向量的方向 = 椭圆的主轴方向。特征值 = 主轴方向上的「弯曲程度」。

推广到 $n$ 维：二次型定义了一个 $n$ 维的「超椭球面」，主轴定理告诉你——**任何二次曲面，在它的特征向量坐标系下，都是轴对齐的**。

#### ML 重头戏：Mahalanobis 距离

多元高斯分布 $\mathcal{N}(\mu, \Sigma)$ 的密度：

$$p(x) \propto \exp\left(-\frac{1}{2} (x-\mu)^T \Sigma^{-1} (x-\mu) \right)$$

其中的二次型 $(x-\mu)^T \Sigma^{-1} (x-\mu)$ 就是 **Mahalanobis 距离的平方**。

对协方差矩阵 $\Sigma$ 做谱分解：$\Sigma = Q \Lambda Q^T$，则 $\Sigma^{-1} = Q \Lambda^{-1} Q^T$。

做变换 $y = Q^T (x-\mu)$（转到特征向量坐标系）：

$$(x-\mu)^T \Sigma^{-1} (x-\mu) = y^T \Lambda^{-1} y = \sum_{i=1}^n \frac{y_i^2}{\lambda_i}$$

**这意味着什么？** 在特征向量坐标系中，Mahalanobis 距离退化为沿各轴的独立标准化距离：$y_i / \sqrt{\lambda_i}$。多元高斯在你手里变成了 $n$ 个独立的一元高斯！每个方向上的方差就是对应的特征值。

这就是为什么数据白化（whitening）要做 $x_{\text{white}} = \Lambda^{-1/2} Q^T (x-\mu)$——在各个主轴上独立地缩放到单位方差，消除所有相关性。

### 12.3 规范形与 Sylvester 惯性定律——不变量比你想的更深刻

#### 从标准形到规范形

标准形 $f = \lambda_1 y_1^2 + \dots + \lambda_n y_n^2$ 的系数是特征值。如果我们不限于正交变换，而允许**任何实可逆变换** $x = P z$，可以走得更远：

对正特征值（$\lambda_i > 0$）：令 $z_i = \sqrt{\lambda_i} y_i$ → 系数变为 $+1$
对负特征值（$\lambda_i < 0$）：令 $z_i = \sqrt{|\lambda_i|} y_i$ → 系数变为 $-1$
对零特征值（$\lambda_i = 0$）：保持为 $0$

结果：二次型化为**规范形**（normal form）：

$$f = z_1^2 + \dots + z_p^2 - z_{p+1}^2 - \dots - z_{p+q}^2$$

- $p$ = 正惯性指数（positive inertia index）= 正特征值个数
- $q$ = 负惯性指数（negative inertia index）= 负特征值个数
- $r = p + q$ = 秩（rank）

#### Sylvester 惯性定律

> **不管你用什么可逆变换去化简二次型，$p$ 和 $q$ 都不会变。** 这是二次型的「指纹」——惯性指数是不变量。

**证明直觉**：如果能用两种不同变换分别得到 $(p_1, q_1)$ 和 $(p_2, q_2)$，则存在一个可逆变换把 $p_1$ 个正项变成 $p_2$ 个正项。但正定子空间的最大维数等于正惯性指数——不可能变。严谨证明见丘维声《高等代数》。

#### ML：Hessian 的惯性指数是你逃不掉的「地形签名」

设 $f(\theta)$ 是损失函数，$\theta^*$ 是临界点（$\nabla f(\theta^*) = 0$），Hessian $H = \nabla^2 f(\theta^*)$ 是一个对称矩阵。二次型 $d^T H d$（$d = \theta - \theta^*$）描述了临界点附近的损失曲面。

Sylvester 惯性定律告诉我们的关键事实：

- **$p$（正惯性指数）**= Hessian 正特征值个数 =「往下走不了」的方向数（局部最小的方向维度）
- **$q$（负惯性指数）**= Hessian 负特征值个数 =「还能往下走」的方向数
- $p = n, q = 0$ → **局部极小值**（所有方向都是「上升」）
- $p > 0, q > 0$ → **鞍点**（有些方向能下，有些方向不能）

> **关键洞察**：无论你怎么重参数化模型（对参数做可逆变换 $\phi = g(\theta)$），新 Hessian 的惯性指数 $(p, q)$ 与原来的**完全一致**。鞍点永远是鞍点，极小值永远是极小值。这是几何性质——不随坐标系改变。

在高维优化中，鞍点远比局部极小值常见。Dauphin et al. (2014) 论证了：当维度很高时，临界点几乎全是鞍点。SGD 能通过梯度噪声逃离鞍点（沿负曲率方向滑出），这正是深度学习的「生存之道」。

### 12.4 二次型的完整分类——四张「地形图」

| 类型 | 特征值 | 正惯性指数 | 2D 示意图 | 优化含义 |
|------|--------|-----------|----------|---------|
| **正定** | 全部 $> 0$ | $p=n, q=0$ | 🥣 碗（椭圆抛物面开口向上） | 严格局部极小值，凸函数 |
| **负定** | 全部 $< 0$ | $p=0, q=n$ | 🙃 倒扣碗 | 严格局部极大值 |
| **不定** | 有正有负 | $p>0, q>0$ | 🐴 马鞍面 | 鞍点——某些方向上升，某些下降 |
| **半正定** | 全部 $\ge 0$，有零 | $p< n, q=0$ | 🛝 有平坦底的碗（谷） | 非严格极小值，平坦方向无梯度信号 |

2D 示意图（用二次型的等高线理解）：

- **正定** $x^2 + y^2$：同心圆（各向同性）或同心椭圆（各向异性）→ 只有一个全局极小值
- **负定** $-x^2 - y^2$：倒扣同心圆 → 只有全局极大值
- **不定** $x^2 - y^2$：马鞍面 → 原点 $x$ 方向上升，$y$ 方向下降 → 既是局部最小（沿 $y=0$）又是局部最大（沿 $x=0$）
- **半正定** $x^2 + 0 \cdot y^2$：$x$ 方向是抛物线，$y$ 方向完全平坦 → 沿 $y$ 轴的整条直线都是极小值

### 12.5 ML 实战：Hessian 分析与 Rayleigh 商

#### Hessian 正定性 → 你到了极小值

在临界点 $\theta^*$（$\nabla L(\theta^*) = 0$），Taylor 展开：

$$L(\theta^* + d) \approx L(\theta^*) + \frac{1}{2} d^T H d$$

- $H$ 正定 → 任意方向 $d$ 上 $\frac{1}{2} d^T H d > 0$ → 往任何方向走损失都变大 → 你到了局部极小值
- $H$ 不定 → 存在方向 $d$ 使 $d^T H d < 0$ → 沿该方向损失还能下降 → 你在鞍点，SGD 会逃逸出去
- $H$ 半正定且至少一个零特征值 → 某些方向二阶导为零 → 无法判断（需要三阶或更高阶信息）

#### 条件数决定收敛速度

Hessian 的条件数 $\kappa(H) = \lambda_{\max} / \lambda_{\min}$ 衡量损失曲面「碗」的瘦长程度：

- $\kappa \approx 1$：碗是圆的 → 所有方向的曲率接近 → 梯度下降路径直指底部 → **收敛快**
- $\kappa \gg 1$：碗是瘦长的椭圆形 → 沿最大曲率方向梯度大（震荡），沿最小曲率方向梯度小（蠕行）→ **收敛慢，需要动量或自适应学习率**

> 这就是为什么深度学习要 BatchNorm（改善条件数）、Adam（自适应调整各方向步长）、动量（沿窄谷方向加速）。

#### Rayleigh 商：任意方向的曲率

**Rayleigh 商**定义为：

$$R(x) = \frac{x^T H x}{x^T x}, \quad x \neq 0$$

核心性质：
$$\lambda_{\min}(H) \le R(x) \le \lambda_{\max}(H), \quad \forall x \neq 0$$

- $R(x)$ 的极小值 = $\lambda_{\min}$，在 $x = v_{\min}$（最小特征值对应特征向量）处取得
- $R(x)$ 的极大值 = $\lambda_{\max}$，在 $x = v_{\max}$ 处取得

**物理直觉**：Rayleigh 商 $R(x)$ 给出了 $H$ 在方向 $x$ 上的「有效曲率」。如果你沿 $v_{\min}$ 方向走，曲率最小，损失变化最慢；沿 $v_{\max}$ 方向走，曲率最大，损失变化最快。

**ML 妙用**：Rayleigh 商可以用来分析神经网络的局部光滑性——在输入空间中，沿某些方向（对应大特征值）函数值剧烈变化（对抗样本的温床），沿另一些方向（对应小特征值）则几乎不变。这就是输入空间 Hessian 的谱分析——与对抗鲁棒性密切相关。

---

## 13. 矩阵微积分：写出你的模型听得懂的「梯度语言」

### 13.1 为什么要学矩阵微积分——PyTorch 不会替你思考

PyTorch 的 `loss.backward()` 帮你算梯度。一行代码，完事。为什么还要学矩阵微积分？

**因为你要知道它算的是什么。** 当梯度消失、爆炸、或模型不收敛时，你唯一的调试手段是理解梯度的数学结构。自动微分是工具，不是理解。就像一个外科医生不会说「反正 CT 扫描仪会显示骨头在哪，我不用学解剖学」。

求矩阵/向量标量函数梯度有三种路径：

| 方式 | 做法 | 优缺点 |
|------|------|--------|
| 逐元素法 | 对每个 $x_i$ 求偏导 $\partial f/\partial x_i$，再拼成向量 | 直接但繁琐，高维时容易出错 |
| **微分法** | 求全微分 $df$，凑成内积/迹形式，提出梯度 | **优雅、安全、不易出错**——本文的方法 |
| 直接链式法则 | 记住常见公式，一步到位 | 需要经验，遇到非标准结构就卡住 |

我们主推微分法。它把矩阵微积分变成「识别模式」的游戏，没有神秘感。

### 13.2 标量对向量——微分法入门

#### 核心思路

不直接对函数求导，而是对函数的**微分**「做手术」：把它写成梯度与微变的内积形式。

$$df = \nabla f \cdot dx = (\nabla f)^T dx$$

只要把 $df$ 凑成「某个向量」$^T dx$ 的形式，那个向量就是梯度 $\nabla f$。

#### 例 1：线性型 $f(x) = x^T a$

$$\begin{aligned}
f(x) &= x^T a = \sum_i x_i a_i \\
df &= a^T dx \quad \text{（因为 } d(x^T a) = (dx)^T a = a^T dx\text{）} \\
\implies \nabla f &= a
\end{aligned}$$

**ML 场景**：线性层 $z = Wx$ 的一个输出分量 $z_i = W_{i:} x$，$\nabla_x z_i = W_{i:}^T$。

#### 例 2：二次型 $f(x) = x^T A x$（$A$ 对称）

$$\begin{aligned}
f(x) &= x^T A x \\
df &= (dx)^T A x + x^T A (dx) \\
   &= x^T A^T dx + x^T A dx \\
   &= x^T (A^T + A) dx
\end{aligned}$$

当 $A$ 对称（$A^T = A$）：

$$df = 2 x^T A dx \implies \nabla f = 2Ax$$

当 $A$ 不对称：

$$df = x^T (A^T + A) dx \implies \nabla f = (A + A^T)x$$

**ML 场景**：$A$ 是 Hessian 时，$x^T H x / 2$ 的梯度就是 $Hx$。最小二乘 $\|Ax-b\|^2 = (Ax-b)^T(Ax-b)$ 展开后也是二次型的形式。

#### 微分法口诀

> **先写 $df$，再「凑」成 $(\text{something})^T dx$ 的形式。那个 something 就是 $\nabla f$。**

### 13.3 标量对矩阵——八条黄金恒等式

#### 通用公式

标量函数对矩阵变量的梯度，通过微分和**迹**来求：

$$df = \operatorname{tr}(G^T dX) \quad \implies \quad \nabla_X f = G$$

迹的作用是把「梯度与微变的内积」推广到矩阵。回忆：$\operatorname{tr}(A^T B)$ 是矩阵内积——把矩阵拉成向量后的标准内积。

以下八条恒等式，每条都给推导和 ML 场景。**学会它们，你就覆盖了神经网络里 90% 以上的梯度需求。**

---

#### 恒等式 ①：Frobenius 范数平方的梯度

$$f(W) = \|W\|_F^2 = \operatorname{tr}(W^T W)$$

$$\begin{aligned}
df &= d\operatorname{tr}(W^T W) = \operatorname{tr}\big(d(W^T W)\big) \\
   &= \operatorname{tr}\big((dW)^T W + W^T (dW)\big) \\
   &= \operatorname{tr}(W^T dW) + \operatorname{tr}(W^T dW) \\
   &= \operatorname{tr}(2W^T dW) \\
\implies \nabla_W f &= 2W
\end{aligned}$$

> **ML 场景**：**权重衰减（weight decay）**。Loss $= L_{\text{data}} + \frac{\lambda}{2} \|W\|_F^2$。加完正则项后的梯度 $= \nabla_W L_{\text{data}} + \lambda W$。PyTorch 里 `optimizer = Adam(model.parameters(), weight_decay=1e-4)` 做的就是这件事。

---

#### 恒等式 ②：迹二次型的梯度

$$f(W) = \operatorname{tr}(W^T A W)$$

$$\begin{aligned}
df &= \operatorname{tr}\big((dW)^T A W + W^T A (dW)\big) \\
   &= \operatorname{tr}\big(W^T A^T dW\big) + \operatorname{tr}\big(W^T A dW\big) \\
   &= \operatorname{tr}\big((W^T(A+A^T)) dW\big) \\
\implies \nabla_W f &= (A + A^T) W
\end{aligned}$$

如果 $A$ 对称：$\nabla_W f = 2AW$。

> **ML 场景**：二次损失项 $\operatorname{tr}(W^T \Sigma W)$ 中，$\Sigma$ 是数据协方差矩阵。梯度 $2\Sigma W$ 告诉你：参数更新方向受数据分布结构（$\Sigma$）的调制。

---

#### 恒等式 ③：双线性型的梯度

$$f(W) = a^T W b$$

$$\begin{aligned}
df &= a^T (dW) b = \operatorname{tr}(a^T (dW) b) \\
   &= \operatorname{tr}(b a^T dW) \quad \text{（迹的循环置换：} \operatorname{tr}(ABC) = \operatorname{tr}(CAB)\text{）} \\
   &= \operatorname{tr}\big((ab^T)^T dW\big) \\
\implies \nabla_W f &= a b^T
\end{aligned}$$

> **ML 场景**：**单样本线性层梯度**。$z = Wx$，某输出 $z_k = e_k^T W x$（$e_k$ 是 one-hot 向量，选中第 $k$ 个输出）。$\nabla_W z_k = e_k x^T$——外积！前向是内积（矩阵乘向量），反向是外积（向量乘向量）。这就是 `Linear` 层反向传播的核心。

---

#### 恒等式 ④：对数行列式的梯度

$$f(W) = \log |W| \quad \text{（} W \text{ 可逆）}$$

这里需要 Jacobi 公式：$d|W| = |W| \cdot \operatorname{tr}(W^{-1} dW)$。

$$\begin{aligned}
df &= d(\log|W|) = \frac{1}{|W|} d|W| \\
   &= \frac{1}{|W|} \cdot |W| \cdot \operatorname{tr}(W^{-1} dW) \\
   &= \operatorname{tr}(W^{-1} dW) = \operatorname{tr}\big((W^{-T})^T dW\big) \\
\implies \nabla_W f &= W^{-T}
\end{aligned}$$

> **ML 场景**：多元高斯分布的负对数似然含 $\frac{1}{2}\log|\Sigma|$ 项。求 MLE 估计 $\Sigma$ 时，梯度 $\nabla_\Sigma \log|\Sigma| = \Sigma^{-1}$。这是一切概率图模型中协方差估计的基础。

---

#### 恒等式 ⑤：最小二乘的梯度

$$f(x) = \|Ax - b\|^2 = (Ax-b)^T(Ax-b)$$

$$\begin{aligned}
f(x) &= x^T A^T A x - 2b^T A x + b^T b \\
df &= 2x^T A^T A dx - 2b^T A dx \\
   &= 2(Ax-b)^T A dx \\
\implies \nabla_x f &= 2A^T (Ax-b)
\end{aligned}$$

> **ML 场景**：**最小二乘的正规方程**。设梯度为零：$A^T A x = A^T b$。这就是线性回归的闭式解。梯度下降则每步沿 $-\nabla f$ 方向走。

---

#### 恒等式 ⑥：单隐藏层网络（含激活函数）

$$f(W) = \frac{1}{2} \|\sigma(Wx) - y\|^2$$

令 $a = Wx$，$h = \sigma(a)$。则 $f = \frac{1}{2} \|h - y\|^2$。

$$\begin{aligned}
df &= (h - y)^T dh \\
   &= (h - y)^T (\sigma'(a) \odot da) \\
   &= \big((h-y) \odot \sigma'(a)\big)^T da \\
   &= \big((h-y) \odot \sigma'(a)\big)^T (dW) x
\end{aligned}$$

用恒等式 ③ 的结论（$a^T (dW) b$ 的梯度是 $ab^T$）：

$$\nabla_W f = \big((h-y) \odot \sigma'(a)\big) \cdot x^T$$

> **ML 场景**：**这是反向传播的矩阵形式。** $(h-y) \odot \sigma'(a)$ 就是「从损失传到激活前的梯度信号」——即 PyTorch 里 `a.grad`。然后外积 $x^T$ 就是 `torch.mm(delta.unsqueeze(1), x.unsqueeze(0))` 做的事情。

---

#### 恒等式 ⑦：Softmax 的 Jacobian

Softmax：$s_i = \frac{e^{z_i}}{\sum_k e^{z_k}}$。Jacobian $J_{ij} = \frac{\partial s_i}{\partial z_j}$：

$$J_{ij} = s_i (\delta_{ij} - s_j)$$

写成矩阵形式：

$$J = \operatorname{diag}(s) - s s^T$$

**推导**：当 $i=j$：$\frac{\partial s_i}{\partial z_i} = s_i(1-s_i)$。当 $i \neq j$：$\frac{\partial s_i}{\partial z_j} = -s_i s_j$。合并即得。

> **ML 场景**：这是分类网络最后一层反向传播的关键。Softmax + Cross Entropy 的梯度惊人地简洁（恒等式 ⑧），但如果你单独算 Softmax 梯度再链到 Cross Entropy，Jacobian 是这个矩阵。

---

#### 恒等式 ⑧：Softmax + Cross Entropy——ML 最优雅的梯度

$$L(z) = -\sum_i y_i \log s_i, \quad s = \operatorname{softmax}(z)$$

其中 $y$ 是 one-hot 标签（或软标签，$\sum_i y_i = 1$）。

**奇迹**：

$$\nabla_z L = s - y$$

**推导**：

$$\begin{aligned}
\frac{\partial L}{\partial z_j} &= -\sum_i y_i \frac{1}{s_i} \frac{\partial s_i}{\partial z_j} \\
&= -\sum_i y_i \frac{1}{s_i} \cdot s_i (\delta_{ij} - s_j) \quad \text{（用恒等式 ⑦）} \\
&= -\sum_i y_i (\delta_{ij} - s_j) \\
&= -y_j + s_j \sum_i y_i = -y_j + s_j
\end{aligned}$$

（因为 $\sum_i y_i = 1$）

> 🎉 **「预测值减标签」——这是 ML 最美的梯度公式**。无论 Softmax 多复杂、Cross Entropy 多复杂，组合起来梯度就是 $s - y$。这也是 PyTorch 里 `CrossEntropyLoss` 内部融合 Softmax 的原因——分开算的话数值不稳定，而且浪费计算。

### 13.4 链式法则与自动微分——PyTorch 背后到底在干什么

#### 从标量链式法则到向量-Jacobian 乘积（VJP）

标量链式法则：$y = f(g(x)) \implies \frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$。

向量链式法则（$x \in \mathbb{R}^n, y \in \mathbb{R}^m, L \in \mathbb{R}$）：

$$\nabla_x L = \left( \frac{\partial y}{\partial x} \right)^T \nabla_y L$$

其中 $\frac{\partial y}{\partial x}$ 是 $m \times n$ Jacobian 矩阵。关键点：**我们从来不会显式构造 Jacobian**——内存太大了。我们用 **VJP（Vector-Jacobian Product）**：给定上游梯度 $\nabla_y L$（一个向量），乘以 Jacobian 的转置得到下游梯度 $\nabla_x L$。

#### 完整示例：两层 MLP 的反向传播（纯矩阵微积分语言）

前向传播：

$$\begin{aligned}
z_1 &= W_1 x + b_1 \\
a_1 &= \operatorname{ReLU}(z_1) \\
z_2 &= W_2 a_1 + b_2 \\
\hat{y} &= z_2 \\
L &= \frac{1}{2} \|\hat{y} - y\|^2
\end{aligned}$$

**Step 1**：$L = \frac{1}{2} \|\hat{y} - y\|^2$。由恒等式 ⑤（对标量输出的特化）：
$$\nabla_{\hat{y}} L = \hat{y} - y$$

**Step 2**：$\hat{y} = z_2 = W_2 a_1 + b_2$。
- $\nabla_{z_2} L = \nabla_{\hat{y}} L = \hat{y} - y$（恒等映射的 VJP 就是传递）
- $\nabla_{b_2} L = \nabla_{z_2} L = \hat{y} - y$（偏置的梯度 = 上游梯度）
- $\nabla_{W_2} L = \nabla_{z_2} L \cdot a_1^T$（恒等式 ③：$a^T W b$ 的梯度是 $a b^T$ 的转置视角）
- $\nabla_{a_1} L = W_2^T \nabla_{z_2} L$（线性层的输入梯度）

**Step 3**：$a_1 = \operatorname{ReLU}(z_1)$。ReLU 的 VJP：
$$\nabla_{z_1} L = \nabla_{a_1} L \odot \mathbb{1}[z_1 > 0]$$

即上游梯度穿过 ReLU 时，把 $z_1 \le 0$ 的位置的梯度归零（死了的神经元不传梯度）。

**Step 4**：$z_1 = W_1 x + b_1$。同 Step 2：
- $\nabla_{b_1} L = \nabla_{z_1} L$
- $\nabla_{W_1} L = \nabla_{z_1} L \cdot x^T$
- $\nabla_x L = W_1^T \nabla_{z_1} L$

#### PyTorch 的 `.backward()` 本质

PyTorch 的自动微分引擎做的是同一件事：

1. 前向时，每个算子把操作记录到计算图（computational graph）中
2. 反向时，从损失节点出发，**按拓扑逆序**逐节点调用其 `backward()` 方法
3. 每个 `backward()` 做的事 = 接收上游梯度 → 计算 VJP → 传给下游
4. 梯度累加到 `.grad` 属性中（因为可能有多个路径到达同一参数）

你写的 `loss.backward()` 背后，就是这张 VJP 接力网。

> **学会矩阵微积分，你就懂了 PyTorch 的灵魂。**

---

> **下一步**：[四、基变换与神经网络](basis-and-neural-networks.md) —— 一切矩阵分解和梯度计算，最终都是为了理解基变换。

## 参考文献

- Gilbert Strang. *Introduction to Linear Algebra*, 6th Edition. Wellesley-Cambridge Press, 2023.
- 同济大学数学系. 《线性代数》（第六版）. 高等教育出版社, 2014.
- 丘维声. 《高等代数》（下册）. 科学出版社, 2013.
- 王萼芳, 石生明. 《高等代数》（第四版）. 高等教育出版社, 2013.
- Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong. *Mathematics for Machine Learning*. Cambridge University Press, 2020.
- Ian Goodfellow, Yoshua Bengio, Aaron Courville. *Deep Learning*. MIT Press, 2016.
- Kaare Brandt Petersen, Michael Syskind Pedersen. *The Matrix Cookbook*. Technical University of Denmark, 2012.
- Stephen Boyd, Lieven Vandenberghe. *Convex Optimization*. Cambridge University Press, 2004. 正定的五种等价判定——线性代数最美的定理之一

§10.5 我们初步见了正定性一面。这里我们把它说透。

对于 $n \times n$ **实对称矩阵** $A$，以下五个命题**完全等价**：

| # | 判定条件 | 直觉 |
|---|---------|------|
| ① | 所有特征值 $> 0$ | 在所有「天然方向」上都是拉伸（不翻转、不压扁） |
| ② | $x^T A x > 0,\ \forall x \neq 0$ | 任何非零向量代入二次型，结果恒正 |
| ③ | 所有主元 $> 0$ | 高斯消元不换行时，每个 pivot 都正 |
| ④ | 所有顺序主子式 $> 0$ | 左上角 $1\times 1, 2\times 2, \dots, n\times n$ 子矩阵行列式全正 |
| ⑤ | $A = R^T R$，其中 $R$ 可逆 | $A$ 可分解为某个可逆矩阵的「Gram 矩阵」 |

> 🏆 这五条等价，被我视为线性代数最美的定理之一。从五个完全不同的视角——特征值、二次型、消元、行列式、矩阵分解——看同一个性质，结论完全一致。这是数学「深层结构」的完美展示。

**⑤ 的深入理解**：$A = R^T R$ 意味着 $x^T A x = x^T R^T R x = \|R x\|^2 \ge 0$，且 $R$ 可逆保证了 $x \neq 0$ 时 $Rx \neq 0$，即 $>0$。从这个视角看，正定矩阵本质上就是一个**可逆矩阵跟自己内积得到的矩阵**。所有协方差矩阵 $X^T X$ 都是（半）正定的——这在 §12.2 会展开。

**Cholesky 分解**：⑤ 中如果取 $R$ 为上三角阵，就是 Cholesky 分解 $A = L L^T$（$L$ 是下三角）。它是正定矩阵的「平方根」，在数值计算中比 LU 快一倍，而且天然稳定。

**ML 视角**：Hessian 在极小值点**必须正定**。如果你的 Hessian 有一个非正的特征值，那你不在极小值——要么在鞍点，要么在平坦区域。SGD 能逃离鞍点（§12.5 详谈），但逃离不了特征值为零的平坦方向。

### 12.2 二次型与标准形——一个坐标变换，消去所有交叉项

#### 二次型是什么

$n$ 元二次齐次多项式就是二次型：

$$f(x) = x^T A x = \sum_{i=1}^n \sum_{j=1}^n a_{ij} x_i x_j$$

展开看：有平方项 $a_{ii} x_i^2$，也有交叉项 $a_{ij} x_i x_j\ (i \neq j)$。交叉项让图形「斜着」——椭圆的轴不与坐标轴对齐。

由于 $x_i x_j = x_j x_i$，总可以写成 $a_{ij} = a_{ji}$，即 $A$ 对称。所以：**每一个二次型都唯一对应一个对称矩阵**。

#### 正交变换 → 标准形（主轴定理）

用正交变换 $x = Qy$（$Q$ 的列是 $A$ 的标准正交特征向量）：

$$f(x) = x^T A x = (Qy)^T A (Qy) = y^T (Q^T A Q) y = y^T \Lambda y = \lambda_1 y_1^2 + \lambda_2 y_2^2 + \dots + \lambda_n y_n^2$$

**交叉项全部消失！** 只留下纯粹的平方项，系数就是特征值。这叫做二次型的**标准形**（canonical form）。这是**主轴定理**（Principal Axis Theorem）的核心结论。

#### 几何直觉：斜着的椭圆，换个角度看就正了

在二维：
$$f(x_1, x_2) = 5x_1^2 + 4x_1 x_2 + 5x_2^2 = x^T \underbrace{\begin{bmatrix} 5 & 2 \\ 2 & 5 \end{bmatrix}}_{A} x$$

特征值：$\lambda_1 = 7, \lambda_2 = 3$；特征向量：$v_1 = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \end{bmatrix}$（45° 方向），$v_2 = \frac{1}{\sqrt{2}}\begin{bmatrix} -1 \\ 1 \end{bmatrix}$（135° 方向）。

在原始 $x_1$-$x_2$ 坐标系中，$f(x)=1$ 是一个**倾斜的椭圆**。在 $y_1$-$y_2$ 坐标系（即特征向量方向）中，$f = 7y_1^2 + 3y_2^2 = 1$——**轴对齐的椭圆**，半轴长为 $1/\sqrt{7}$ 和 $1/\sqrt{3}$。

> 特征向量的方向 = 椭圆的主轴方向。特征值 = 主轴方向上的「弯曲程度」。

推广到 $n$ 维：二次型定义了一个 $n$ 维的「超椭球面」，主轴定理告诉你——**任何二次曲面，在它的特征向量坐标系下，都是轴对齐的**。

#### ML 重头戏：Mahalanobis 距离

多元高斯分布 $\mathcal{N}(\mu, \Sigma)$ 的密度：

$$p(x) \propto \exp\left(-\frac{1}{2} (x-\mu)^T \Sigma^{-1} (x-\mu) \right)$$

其中的二次型 $(x-\mu)^T \Sigma^{-1} (x-\mu)$ 就是 **Mahalanobis 距离的平方**。

对协方差矩阵 $\Sigma$ 做谱分解：$\Sigma = Q \Lambda Q^T$，则 $\Sigma^{-1} = Q \Lambda^{-1} Q^T$。

做变换 $y = Q^T (x-\mu)$（转到特征向量坐标系）：

$$(x-\mu)^T \Sigma^{-1} (x-\mu) = y^T \Lambda^{-1} y = \sum_{i=1}^n \frac{y_i^2}{\lambda_i}$$

**这意味着什么？** 在特征向量坐标系中，Mahalanobis 距离退化为沿各轴的独立标准化距离：$y_i / \sqrt{\lambda_i}$。多元高斯在你手里变成了 $n$ 个独立的一元高斯！每个方向上的方差就是对应的特征值。

这就是为什么数据白化（whitening）要做 $x_{\text{white}} = \Lambda^{-1/2} Q^T (x-\mu)$——在各个主轴上独立地缩放到单位方差，消除所有相关性。

### 12.3 规范形与 Sylvester 惯性定律——不变量比你想的更深刻

#### 从标准形到规范形

标准形 $f = \lambda_1 y_1^2 + \dots + \lambda_n y_n^2$ 的系数是特征值。如果我们不限于正交变换，而允许**任何实可逆变换** $x = P z$，可以走得更远：

对正特征值（$\lambda_i > 0$）：令 $z_i = \sqrt{\lambda_i} y_i$ → 系数变为 $+1$
对负特征值（$\lambda_i < 0$）：令 $z_i = \sqrt{|\lambda_i|} y_i$ → 系数变为 $-1$
对零特征值（$\lambda_i = 0$）：保持为 $0$

结果：二次型化为**规范形**（normal form）：

$$f = z_1^2 + \dots + z_p^2 - z_{p+1}^2 - \dots - z_{p+q}^2$$

- $p$ = 正惯性指数（positive inertia index）= 正特征值个数
- $q$ = 负惯性指数（negative inertia index）= 负特征值个数
- $r = p + q$ = 秩（rank）

#### Sylvester 惯性定律

> **不管你用什么可逆变换去化简二次型，$p$ 和 $q$ 都不会变。** 这是二次型的「指纹」——惯性指数是不变量。

**证明直觉**：如果能用两种不同变换分别得到 $(p_1, q_1)$ 和 $(p_2, q_2)$，则存在一个可逆变换把 $p_1$ 个正项变成 $p_2$ 个正项。但正定子空间的最大维数等于正惯性指数——不可能变。严谨证明见丘维声《高等代数》。

#### ML：Hessian 的惯性指数是你逃不掉的「地形签名」

设 $f(\theta)$ 是损失函数，$\theta^*$ 是临界点（$\nabla f(\theta^*) = 0$），Hessian $H = \nabla^2 f(\theta^*)$ 是一个对称矩阵。二次型 $d^T H d$（$d = \theta - \theta^*$）描述了临界点附近的损失曲面。

Sylvester 惯性定律告诉我们的关键事实：

- **$p$（正惯性指数）**= Hessian 正特征值个数 =「往下走不了」的方向数（局部最小的方向维度）
- **$q$（负惯性指数）**= Hessian 负特征值个数 =「还能往下走」的方向数
- $p = n, q = 0$ → **局部极小值**（所有方向都是「上升」）
- $p > 0, q > 0$ → **鞍点**（有些方向能下，有些方向不能）

> **关键洞察**：无论你怎么重参数化模型（对参数做可逆变换 $\phi = g(\theta)$），新 Hessian 的惯性指数 $(p, q)$ 与原来的**完全一致**。鞍点永远是鞍点，极小值永远是极小值。这是几何性质——不随坐标系改变。

在高维优化中，鞍点远比局部极小值常见。Dauphin et al. (2014) 论证了：当维度很高时，临界点几乎全是鞍点。SGD 能通过梯度噪声逃离鞍点（沿负曲率方向滑出），这正是深度学习的「生存之道」。

### 12.4 二次型的完整分类——四张「地形图」

| 类型 | 特征值 | 正惯性指数 | 2D 示意图 | 优化含义 |
|------|--------|-----------|----------|---------|
| **正定** | 全部 $> 0$ | $p=n, q=0$ | 🥣 碗（椭圆抛物面开口向上） | 严格局部极小值，凸函数 |
| **负定** | 全部 $< 0$ | $p=0, q=n$ | 🙃 倒扣碗 | 严格局部极大值 |
| **不定** | 有正有负 | $p>0, q>0$ | 🐴 马鞍面 | 鞍点——某些方向上升，某些下降 |
| **半正定** | 全部 $\ge 0$，有零 | $p< n, q=0$ | 🛝 有平坦底的碗（谷） | 非严格极小值，平坦方向无梯度信号 |

2D 示意图（用二次型的等高线理解）：

- **正定** $x^2 + y^2$：同心圆（各向同性）或同心椭圆（各向异性）→ 只有一个全局极小值
- **负定** $-x^2 - y^2$：倒扣同心圆 → 只有全局极大值
- **不定** $x^2 - y^2$：马鞍面 → 原点 $x$ 方向上升，$y$ 方向下降 → 既是局部最小（沿 $y=0$）又是局部最大（沿 $x=0$）
- **半正定** $x^2 + 0 \cdot y^2$：$x$ 方向是抛物线，$y$ 方向完全平坦 → 沿 $y$ 轴的整条直线都是极小值

### 12.5 ML 实战：Hessian 分析与 Rayleigh 商

#### Hessian 正定性 → 你到了极小值

在临界点 $\theta^*$（$\nabla L(\theta^*) = 0$），Taylor 展开：

$$L(\theta^* + d) \approx L(\theta^*) + \frac{1}{2} d^T H d$$

- $H$ 正定 → 任意方向 $d$ 上 $\frac{1}{2} d^T H d > 0$ → 往任何方向走损失都变大 → 你到了局部极小值
- $H$ 不定 → 存在方向 $d$ 使 $d^T H d < 0$ → 沿该方向损失还能下降 → 你在鞍点，SGD 会逃逸出去
- $H$ 半正定且至少一个零特征值 → 某些方向二阶导为零 → 无法判断（需要三阶或更高阶信息）

#### 条件数决定收敛速度

Hessian 的条件数 $\kappa(H) = \lambda_{\max} / \lambda_{\min}$ 衡量损失曲面「碗」的瘦长程度：

- $\kappa \approx 1$：碗是圆的 → 所有方向的曲率接近 → 梯度下降路径直指底部 → **收敛快**
- $\kappa \gg 1$：碗是瘦长的椭圆形 → 沿最大曲率方向梯度大（震荡），沿最小曲率方向梯度小（蠕行）→ **收敛慢，需要动量或自适应学习率**

> 这就是为什么深度学习要 BatchNorm（改善条件数）、Adam（自适应调整各方向步长）、动量（沿窄谷方向加速）。

#### Rayleigh 商：任意方向的曲率

**Rayleigh 商**定义为：

$$R(x) = \frac{x^T H x}{x^T x}, \quad x \neq 0$$

核心性质：
$$\lambda_{\min}(H) \le R(x) \le \lambda_{\max}(H), \quad \forall x \neq 0$$

- $R(x)$ 的极小值 = $\lambda_{\min}$，在 $x = v_{\min}$（最小特征值对应特征向量）处取得
- $R(x)$ 的极大值 = $\lambda_{\max}$，在 $x = v_{\max}$ 处取得

**物理直觉**：Rayleigh 商 $R(x)$ 给出了 $H$ 在方向 $x$ 上的「有效曲率」。如果你沿 $v_{\min}$ 方向走，曲率最小，损失变化最慢；沿 $v_{\max}$ 方向走，曲率最大，损失变化最快。

**ML 妙用**：Rayleigh 商可以用来分析神经网络的局部光滑性——在输入空间中，沿某些方向（对应大特征值）函数值剧烈变化（对抗样本的温床），沿另一些方向（对应小特征值）则几乎不变。这就是输入空间 Hessian 的谱分析——与对抗鲁棒性密切相关。

---

## 13. 矩阵微积分：写出你的模型听得懂的「梯度语言」

### 13.1 为什么要学矩阵微积分——PyTorch 不会替你思考

PyTorch 的 `loss.backward()` 帮你算梯度。一行代码，完事。为什么还要学矩阵微积分？

**因为你要知道它算的是什么。** 当梯度消失、爆炸、或模型不收敛时，你唯一的调试手段是理解梯度的数学结构。自动微分是工具，不是理解。就像一个外科医生不会说「反正 CT 扫描仪会显示骨头在哪，我不用学解剖学」。

求矩阵/向量标量函数梯度有三种路径：

| 方式 | 做法 | 优缺点 |
|------|------|--------|
| 逐元素法 | 对每个 $x_i$ 求偏导 $\partial f/\partial x_i$，再拼成向量 | 直接但繁琐，高维时容易出错 |
| **微分法** | 求全微分 $df$，凑成内积/迹形式，提出梯度 | **优雅、安全、不易出错**——本文的方法 |
| 直接链式法则 | 记住常见公式，一步到位 | 需要经验，遇到非标准结构就卡住 |

我们主推微分法。它把矩阵微积分变成「识别模式」的游戏，没有神秘感。

### 13.2 标量对向量——微分法入门

#### 核心思路

不直接对函数求导，而是对函数的**微分**「做手术」：把它写成梯度与微变的内积形式。

$$df = \nabla f \cdot dx = (\nabla f)^T dx$$

只要把 $df$ 凑成「某个向量」$^T dx$ 的形式，那个向量就是梯度 $\nabla f$。

#### 例 1：线性型 $f(x) = x^T a$

$$\begin{aligned}
f(x) &= x^T a = \sum_i x_i a_i \\
df &= a^T dx \quad \text{（因为 } d(x^T a) = (dx)^T a = a^T dx\text{）} \\
\implies \nabla f &= a
\end{aligned}$$

**ML 场景**：线性层 $z = Wx$ 的一个输出分量 $z_i = W_{i:} x$，$\nabla_x z_i = W_{i:}^T$。

#### 例 2：二次型 $f(x) = x^T A x$（$A$ 对称）

$$\begin{aligned}
f(x) &= x^T A x \\
df &= (dx)^T A x + x^T A (dx) \\
   &= x^T A^T dx + x^T A dx \\
   &= x^T (A^T + A) dx
\end{aligned}$$

当 $A$ 对称（$A^T = A$）：

$$df = 2 x^T A dx \implies \nabla f = 2Ax$$

当 $A$ 不对称：

$$df = x^T (A^T + A) dx \implies \nabla f = (A + A^T)x$$

**ML 场景**：$A$ 是 Hessian 时，$x^T H x / 2$ 的梯度就是 $Hx$。最小二乘 $\|Ax-b\|^2 = (Ax-b)^T(Ax-b)$ 展开后也是二次型的形式。

#### 微分法口诀

> **先写 $df$，再「凑」成 $(\text{something})^T dx$ 的形式。那个 something 就是 $\nabla f$。**

### 13.3 标量对矩阵——八条黄金恒等式

#### 通用公式

标量函数对矩阵变量的梯度，通过微分和**迹**来求：

$$df = \operatorname{tr}(G^T dX) \quad \implies \quad \nabla_X f = G$$

迹的作用是把「梯度与微变的内积」推广到矩阵。回忆：$\operatorname{tr}(A^T B)$ 是矩阵内积——把矩阵拉成向量后的标准内积。

以下八条恒等式，每条都给推导和 ML 场景。**学会它们，你就覆盖了神经网络里 90% 以上的梯度需求。**

---

#### 恒等式 ①：Frobenius 范数平方的梯度

$$f(W) = \|W\|_F^2 = \operatorname{tr}(W^T W)$$

$$\begin{aligned}
df &= d\operatorname{tr}(W^T W) = \operatorname{tr}\big(d(W^T W)\big) \\
   &= \operatorname{tr}\big((dW)^T W + W^T (dW)\big) \\
   &= \operatorname{tr}(W^T dW) + \operatorname{tr}(W^T dW) \\
   &= \operatorname{tr}(2W^T dW) \\
\implies \nabla_W f &= 2W
\end{aligned}$$

> **ML 场景**：**权重衰减（weight decay）**。Loss $= L_{\text{data}} + \frac{\lambda}{2} \|W\|_F^2$。加完正则项后的梯度 $= \nabla_W L_{\text{data}} + \lambda W$。PyTorch 里 `optimizer = Adam(model.parameters(), weight_decay=1e-4)` 做的就是这件事。

---

#### 恒等式 ②：迹二次型的梯度

$$f(W) = \operatorname{tr}(W^T A W)$$

$$\begin{aligned}
df &= \operatorname{tr}\big((dW)^T A W + W^T A (dW)\big) \\
   &= \operatorname{tr}\big(W^T A^T dW\big) + \operatorname{tr}\big(W^T A dW\big) \\
   &= \operatorname{tr}\big((W^T(A+A^T)) dW\big) \\
\implies \nabla_W f &= (A + A^T) W
\end{aligned}$$

如果 $A$ 对称：$\nabla_W f = 2AW$。

> **ML 场景**：二次损失项 $\operatorname{tr}(W^T \Sigma W)$ 中，$\Sigma$ 是数据协方差矩阵。梯度 $2\Sigma W$ 告诉你：参数更新方向受数据分布结构（$\Sigma$）的调制。

---

#### 恒等式 ③：双线性型的梯度

$$f(W) = a^T W b$$

$$\begin{aligned}
df &= a^T (dW) b = \operatorname{tr}(a^T (dW) b) \\
   &= \operatorname{tr}(b a^T dW) \quad \text{（迹的循环置换：} \operatorname{tr}(ABC) = \operatorname{tr}(CAB)\text{）} \\
   &= \operatorname{tr}\big((ab^T)^T dW\big) \\
\implies \nabla_W f &= a b^T
\end{aligned}$$

> **ML 场景**：**单样本线性层梯度**。$z = Wx$，某输出 $z_k = e_k^T W x$（$e_k$ 是 one-hot 向量，选中第 $k$ 个输出）。$\nabla_W z_k = e_k x^T$——外积！前向是内积（矩阵乘向量），反向是外积（向量乘向量）。这就是 `Linear` 层反向传播的核心。

---

#### 恒等式 ④：对数行列式的梯度

$$f(W) = \log |W| \quad \text{（} W \text{ 可逆）}$$

这里需要 Jacobi 公式：$d|W| = |W| \cdot \operatorname{tr}(W^{-1} dW)$。

$$\begin{aligned}
df &= d(\log|W|) = \frac{1}{|W|} d|W| \\
   &= \frac{1}{|W|} \cdot |W| \cdot \operatorname{tr}(W^{-1} dW) \\
   &= \operatorname{tr}(W^{-1} dW) = \operatorname{tr}\big((W^{-T})^T dW\big) \\
\implies \nabla_W f &= W^{-T}
\end{aligned}$$

> **ML 场景**：多元高斯分布的负对数似然含 $\frac{1}{2}\log|\Sigma|$ 项。求 MLE 估计 $\Sigma$ 时，梯度 $\nabla_\Sigma \log|\Sigma| = \Sigma^{-1}$。这是一切概率图模型中协方差估计的基础。

---

#### 恒等式 ⑤：最小二乘的梯度

$$f(x) = \|Ax - b\|^2 = (Ax-b)^T(Ax-b)$$

$$\begin{aligned}
f(x) &= x^T A^T A x - 2b^T A x + b^T b \\
df &= 2x^T A^T A dx - 2b^T A dx \\
   &= 2(Ax-b)^T A dx \\
\implies \nabla_x f &= 2A^T (Ax-b)
\end{aligned}$$

> **ML 场景**：**最小二乘的正规方程**。设梯度为零：$A^T A x = A^T b$。这就是线性回归的闭式解。梯度下降则每步沿 $-\nabla f$ 方向走。

---

#### 恒等式 ⑥：单隐藏层网络（含激活函数）

$$f(W) = \frac{1}{2} \|\sigma(Wx) - y\|^2$$

令 $a = Wx$，$h = \sigma(a)$。则 $f = \frac{1}{2} \|h - y\|^2$。

$$\begin{aligned}
df &= (h - y)^T dh \\
   &= (h - y)^T (\sigma'(a) \odot da) \\
   &= \big((h-y) \odot \sigma'(a)\big)^T da \\
   &= \big((h-y) \odot \sigma'(a)\big)^T (dW) x
\end{aligned}$$

用恒等式 ③ 的结论（$a^T (dW) b$ 的梯度是 $ab^T$）：

$$\nabla_W f = \big((h-y) \odot \sigma'(a)\big) \cdot x^T$$

> **ML 场景**：**这是反向传播的矩阵形式。** $(h-y) \odot \sigma'(a)$ 就是「从损失传到激活前的梯度信号」——即 PyTorch 里 `a.grad`。然后外积 $x^T$ 就是 `torch.mm(delta.unsqueeze(1), x.unsqueeze(0))` 做的事情。

---

#### 恒等式 ⑦：Softmax 的 Jacobian

Softmax：$s_i = \frac{e^{z_i}}{\sum_k e^{z_k}}$。Jacobian $J_{ij} = \frac{\partial s_i}{\partial z_j}$：

$$J_{ij} = s_i (\delta_{ij} - s_j)$$

写成矩阵形式：

$$J = \operatorname{diag}(s) - s s^T$$

**推导**：当 $i=j$：$\frac{\partial s_i}{\partial z_i} = s_i(1-s_i)$。当 $i \neq j$：$\frac{\partial s_i}{\partial z_j} = -s_i s_j$。合并即得。

> **ML 场景**：这是分类网络最后一层反向传播的关键。Softmax + Cross Entropy 的梯度惊人地简洁（恒等式 ⑧），但如果你单独算 Softmax 梯度再链到 Cross Entropy，Jacobian 是这个矩阵。

---

#### 恒等式 ⑧：Softmax + Cross Entropy——ML 最优雅的梯度

$$L(z) = -\sum_i y_i \log s_i, \quad s = \operatorname{softmax}(z)$$

其中 $y$ 是 one-hot 标签（或软标签，$\sum_i y_i = 1$）。

**奇迹**：

$$\nabla_z L = s - y$$

**推导**：

$$\begin{aligned}
\frac{\partial L}{\partial z_j} &= -\sum_i y_i \frac{1}{s_i} \frac{\partial s_i}{\partial z_j} \\
&= -\sum_i y_i \frac{1}{s_i} \cdot s_i (\delta_{ij} - s_j) \quad \text{（用恒等式 ⑦）} \\
&= -\sum_i y_i (\delta_{ij} - s_j) \\
&= -y_j + s_j \sum_i y_i = -y_j + s_j
\end{aligned}$$

（因为 $\sum_i y_i = 1$）

> 🎉 **「预测值减标签」——这是 ML 最美的梯度公式**。无论 Softmax 多复杂、Cross Entropy 多复杂，组合起来梯度就是 $s - y$。这也是 PyTorch 里 `CrossEntropyLoss` 内部融合 Softmax 的原因——分开算的话数值不稳定，而且浪费计算。

### 13.4 链式法则与自动微分——PyTorch 背后到底在干什么

#### 从标量链式法则到向量-Jacobian 乘积（VJP）

标量链式法则：$y = f(g(x)) \implies \frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$。

向量链式法则（$x \in \mathbb{R}^n, y \in \mathbb{R}^m, L \in \mathbb{R}$）：

$$\nabla_x L = \left( \frac{\partial y}{\partial x} \right)^T \nabla_y L$$

其中 $\frac{\partial y}{\partial x}$ 是 $m \times n$ Jacobian 矩阵。关键点：**我们从来不会显式构造 Jacobian**——内存太大了。我们用 **VJP（Vector-Jacobian Product）**：给定上游梯度 $\nabla_y L$（一个向量），乘以 Jacobian 的转置得到下游梯度 $\nabla_x L$。

#### 完整示例：两层 MLP 的反向传播（纯矩阵微积分语言）

前向传播：

$$\begin{aligned}
z_1 &= W_1 x + b_1 \\
a_1 &= \operatorname{ReLU}(z_1) \\
z_2 &= W_2 a_1 + b_2 \\
\hat{y} &= z_2 \\
L &= \frac{1}{2} \|\hat{y} - y\|^2
\end{aligned}$$

**Step 1**：$L = \frac{1}{2} \|\hat{y} - y\|^2$。由恒等式 ⑤（对标量输出的特化）：
$$\nabla_{\hat{y}} L = \hat{y} - y$$

**Step 2**：$\hat{y} = z_2 = W_2 a_1 + b_2$。
- $\nabla_{z_2} L = \nabla_{\hat{y}} L = \hat{y} - y$（恒等映射的 VJP 就是传递）
- $\nabla_{b_2} L = \nabla_{z_2} L = \hat{y} - y$（偏置的梯度 = 上游梯度）
- $\nabla_{W_2} L = \nabla_{z_2} L \cdot a_1^T$（恒等式 ③：$a^T W b$ 的梯度是 $a b^T$ 的转置视角）
- $\nabla_{a_1} L = W_2^T \nabla_{z_2} L$（线性层的输入梯度）

**Step 3**：$a_1 = \operatorname{ReLU}(z_1)$。ReLU 的 VJP：
$$\nabla_{z_1} L = \nabla_{a_1} L \odot \mathbb{1}[z_1 > 0]$$

即上游梯度穿过 ReLU 时，把 $z_1 \le 0$ 的位置的梯度归零（死了的神经元不传梯度）。

**Step 4**：$z_1 = W_1 x + b_1$。同 Step 2：
- $\nabla_{b_1} L = \nabla_{z_1} L$
- $\nabla_{W_1} L = \nabla_{z_1} L \cdot x^T$
- $\nabla_x L = W_1^T \nabla_{z_1} L$

#### PyTorch 的 `.backward()` 本质

PyTorch 的自动微分引擎做的是同一件事：

1. 前向时，每个算子把操作记录到计算图（computational graph）中
2. 反向时，从损失节点出发，**按拓扑逆序**逐节点调用其 `backward()` 方法
3. 每个 `backward()` 做的事 = 接收上游梯度 → 计算 VJP → 传给下游
4. 梯度累加到 `.grad` 属性中（因为可能有多个路径到达同一参数）

你写的 `loss.backward()` 背后，就是这张 VJP 接力网。

> **学会矩阵微积分，你就懂了 PyTorch 的灵魂。**

---

> 📝 **本章习题**：[行列式、特征值与 SVD · 习题与思考](determinant-eigen-svd-exercises.md)

> **下一步**：[四、基变换与神经网络](basis-and-neural-networks.md) —— 一切矩阵分解和梯度计算，最终都是为了理解基变换。

## 参考文献

- Gilbert Strang. *Introduction to Linear Algebra*, 6th Edition. Wellesley-Cambridge Press, 2023.
- 同济大学数学系. 《线性代数》（第六版）. 高等教育出版社, 2014.
- 丘维声. 《高等代数》（下册）. 科学出版社, 2013.
- 王萼芳, 石生明. 《高等代数》（第四版）. 高等教育出版社, 2013.
- Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong. *Mathematics for Machine Learning*. Cambridge University Press, 2020.
- Ian Goodfellow, Yoshua Bengio, Aaron Courville. *Deep Learning*. MIT Press, 2016.
- Kaare Brandt Petersen, Michael Syskind Pedersen. *The Matrix Cookbook*. Technical University of Denmark, 2012.
- Stephen Boyd, Lieven Vandenberghe. *Convex Optimization*. Cambridge University Press, 2004.
