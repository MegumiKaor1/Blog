# 四、基变换与神经网络

## 12. 基变换：换个角度看同一个变换

### 12.1 矩阵取决于你站的「角度」

同一个线性变换，在不同基下有不同的矩阵表示。

$$B = P^{-1} A P$$

$A$ 是变换 $T$ 在标准基下的矩阵。$B$ 是 **同一个** 变换 $T$ 在另一个基下的矩阵。$P$ 的列就是新基的向量（用标准基坐标写出来）。

**换个基，矩阵的形状变了，但做的事一模一样。** 这就像用不同的语言描述同一个物理现象——英文版和中文版的内容相同，只是「编码方式」不同。

### 12.2 基变换的两步

1. 用 $P^{-1}$ 把输入从「新基坐标」转回「标准基坐标」
2. 用 $A$ 在标准基下做变换
3. 用 $P$ 把输出从「标准基坐标」转成「新基坐标」

$B = P^{-1} A P$ 就是这三步的复合。

### 12.3 Embedding 就是基变换——这是 ML 里的最大应用

**嵌入层（Embedding）** 把离散的 token 映射成向量。本质上，它在做的是：**为语言选一组好的「基」**。

Word2Vec 里经典的类比：

$$\text{vec}(\text{国王}) - \text{vec}(\text{男人}) + \text{vec}(\text{女人}) \approx \text{vec}(\text{王后})$$

这个公式之所以成立，是因为 embedding 空间里这些向量之间的「方向差」恰好编码了语义关系——「国王 − 男人 ≈ 王后 − 女人」（都相差「女性性别」这个方向）。

换言之，预训练学到的 embedding 矩阵找到了一个「好的基」——在这个基下：
- 语义相近的词向量夹角小（内积大）
- 语义关系的方向是线性可分离的
- 降维后依然保留主要语义结构

---

## 13. 神经网络 = 线性变换的嵌套

### 13.1 一层 = 一个仿射变换

$$y = Wx + b$$

$W$ 拉伸/旋转/投影输入空间，$b$ 平移结果。这是**仿射变换**（线性 + 平移）。

### 13.2 没有非线性，多层坍缩成一层

两层线性网络：
$$y = W_2(W_1 x + b_1) + b_2 = (W_2 W_1)x + (W_2 b_1 + b_2) = W' x + b'$$

两个线性变换的复合还是线性变换。堆 100 层也没用——等效于 1 层。**非线性激活函数是深度网络存在的唯一理由。**

### 13.3 ReLU 如何创造复杂性

$$\text{ReLU}(z) = \max(0, z)$$

ReLU 做的事情很粗暴：把负值剪成 0。但这恰好创造了非线性——整个输入空间被切割成不同的**线性区域**。

**举例**：一个两层网络 $y = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$

- 当 $W_1 x + b_1$ 的某个分量 $>0$：那个神经元激活，对应的 $W_2$ 列参与输出
- 当那个分量 $\le 0$：神经元不激活，对应的 $W_2$ 列贡献为 0

输入空间被 $W_1$ 的各行（超平面）切分成多个区域，每个区域内激活的神经元组合不同 → 每个区域内网络都是**不同的线性函数**。足够多的区域可以逼近任何连续函数——这就是万能逼近定理的直觉。

### 13.4 梯度和反向传播：线性代数最壮丽的应用

对于一个简单的两层网络：

- 前向：$h = W_1 x + b_1, \quad a = \text{ReLU}(h), \quad \hat{y} = W_2 a + b_2$
- 损失：$L = \frac{1}{2}\|\hat{y} - y\|^2$

反向传播求梯度：

1. $\frac{\partial L}{\partial \hat{y}} = \hat{y} - y$

> （这就是恒等式⑧——cross-entropy + softmax 的梯度。如果用 MSE 损失则是 $\hat{y} - y$，用 CE 损失同样是 $\hat{y} - y$ 的形式——这不是巧合，是矩阵微积分精心设计的结果。）

2. $\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial \hat{y}} \cdot a^T$（外积）

> （恒等式③的推广——$\nabla_{W_2} L = (\frac{\partial L}{\partial \hat{y}}) \cdot a^{\mathsf{T}}$，外积形式。对权重矩阵求梯度就是「上游梯度 × 该层输入的转置」——这是所有线性层梯度的统一模板。）

3. $\frac{\partial L}{\partial a} = W_2^T \frac{\partial L}{\partial \hat{y}}$（**转置**是关键！）

> （VJP——向量-雅可比乘积。不是算完整雅可比矩阵 $\frac{\partial \hat{y}}{\partial a} \in \mathbb{R}^{m \times n}$，而是算雅可比矩阵的转置乘以一个向量 $W_2^{\mathsf{T}} \cdot \frac{\partial L}{\partial \hat{y}}$。**这就是 PyTorch autograd 的核心操作**——永远只算 $v^{\mathsf{T}} J$，不显式构造 $J$。省内存，省计算。）

4. $\frac{\partial L}{\partial h} = \frac{\partial L}{\partial a} \odot \text{ReLU}'(h)$（逐元素乘）

5. $\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial h} \cdot x^T$

> （同样是恒等式③——$\nabla_{W_1} L = (\frac{\partial L}{\partial h}) \cdot x^{\mathsf{T}}$。无论网络多深，每一层权重的梯度都是「该层的上游梯度 × 该层输入的转置」。这条规则贯穿整个深度学习。）

注意第 3 步：梯度向后传的时候乘的是 $W_2^T$，不是 $W_2$。这就是为什么叫**反向**传播——矩阵在前向时从左乘到右，在反向时**转置后从右乘到左**。

> 反向传播不是黑魔法——它就是链式法则 + 矩阵转置 + 逐元素导数。理解了线性变换的复合和转置，反向传播就一目了然。

### 13.5 完整串联：你写的每一行 PyTorch 背后的线代

```python
# 前向传播
h = x @ W1.T + b1          # 矩阵乘法 = 线性组合
a = torch.relu(h)           # 分段线性 = 切割空间
y = a @ W2.T + b2           # 再一个线性变换
loss = ((y - target) ** 2).mean()  # L2 距离

# 反向传播 (autograd 替你做了)
loss.backward()
# W1.grad = ∂L/∂W1, W2.grad = ∂L/∂W2
# 本质: 链式法则 × 矩阵转置 × 外积
# 然后 optimizer.step() 沿着梯度方向走一步
```

- `@` = 矩阵乘法的列图景
- LayerNorm = 投影到单位球面
- Attention = 内积矩阵 $QK^T$ 的 softmax 加权求和
- LoRA = 低秩修正 $\Delta W = BA$

**一切皆线代。**

> **关于分类损失**：上面我们一直用 MSE 损失。那分类任务呢？现代深度学习最常用的组合是 **softmax + cross-entropy**：
>
> $$z = W_2 a + b_2 \quad\text{(logits)}$$
> $$\hat{y} = \text{softmax}(z) = \frac{\exp(z)}{\sum \exp(z)}$$
> $$L = -\sum_i y_i \log \hat{y}_i \quad\text{(cross-entropy)}$$
>
> 神奇的事发生了：**$\frac{\partial L}{\partial z} = \hat{y} - y$**——跟 MSE 的形式完全一样！这不是巧合，而是精心设计的数学性质。三行推导：
>
> $$\begin{aligned} dL &= -\sum_i y_i \, d(\log \hat{y}_i) = -\sum_i \frac{y_i}{\hat{y}_i} \, d\hat{y}_i \\ &= -\sum_i \frac{y_i}{\hat{y}_i} \left( \hat{y}_i \cdot dz_i - \hat{y}_i \sum_j \hat{y}_j \, dz_j \right) \quad\text{(softmax 的雅可比：}\frac{\partial \hat{y}_i}{\partial z_j} = \hat{y}_i(\delta_{ij} - \hat{y}_j)\text{)} \\ &= -y^{\mathsf{T}}(I - \mathbf{1}\hat{y}^{\mathsf{T}}) \, dz = (\hat{y} - y)^{\mathsf{T}} \, dz \quad\Rightarrow\quad \frac{\partial L}{\partial z} = \hat{y} - y \end{aligned}$$
>
> softmax 的雅可比矩阵是 $\text{diag}(\hat{y}) - \hat{y}\hat{y}^{\mathsf{T}}$，乘上 $y/\hat{y}$ 刚好消掉所有复杂项，只剩 $\hat{y} - y$。用微分法（differential，不写完整雅可比）三步得结果。这也是为什么几乎所有框架都把 softmax 和 cross-entropy 合并成一个算子——分开算数值不稳定，合起来既稳定又简洁。

---

> **下一步**：[💭 个人理解与资源](insights-and-resources.md) —— 学完了，回顾一下这些知识如何改变你看 AI 的方式。

---

## 14. ML 场景全景地图：线性代数统治深度学习的九个瞬间

前面我们用两层的 MLP 讲清楚了神经网络和线性代数的关系。但现代深度学习远不止 MLP——卷积、残差、Transformer、优化器、正则化、二阶方法……每一个看似复杂的组件，拆开来看，核心都是线性代数。下面我们用「**直觉 → 线代公式 → PyTorch 对应 → 关键洞察**」的格式，带你走一遍 ML 里最经典的九个场景。

---

### 14.1 Attention = 内积矩阵 + softmax

**直觉**：Attention 回答一个朴素的问题——「序列里哪些 token 和我最相关？」相关性用内积度量，softmax 把相关性变成概率权重，最后用权重对 value 向量做加权求和。

**线代公式**：

给定 $Q, K, V \in \mathbb{R}^{n \times d}$（$n$ 个 token，每个 $d$ 维），

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^{\mathsf{T}}}{\sqrt{d}}\right) V$$

$QK^{\mathsf{T}} \in \mathbb{R}^{n \times n}$ 是一个**内积矩阵**——第 $(i,j)$ 个元素正是 token $i$ 和 token $j$ 的 query-key 相似度。除以 $\sqrt{d}$ 是为了防止内积值随维度增大而膨胀（因为 $\text{Var}(q \cdot k) = d$，不做缩放的话 softmax 会饱和到 one-hot，梯度消失）。softmax 按行归一化后，每行是一个概率分布，最后左乘 $V$ 得到加权求和。

**PyTorch 对应**：

```python
attn = torch.softmax(Q @ K.T / math.sqrt(d), dim=-1) @ V
```

**关键洞察**：$QK^{\mathsf{T}}$ 的**秩**决定了 attention 模式的多样性。如果 $\text{rank}(QK^{\mathsf{T}})$ 很低（这在很多层中确实如此），多个 token 的 attention 分布几乎相同 → **attention collapse**。从 SVD 的角度看：$QK^{\mathsf{T}} = U\Sigma V^{\mathsf{T}}$。如果 $\sigma_1 \gg \sigma_2, \sigma_3, \dots$，attention 本质上是一维的——所有 token 都关注相同的那几个位置。Multi-head attention 部分缓解了这个问题：每个 head 用不同的投影矩阵 $W_Q, W_K, W_V$，产生不同的低秩结构，拼在一起获得多样性。这也是为什么去掉某些 head 有时不掉点——冗余的低秩结构。

---

### 14.2 LayerNorm = 投影到单位球面

**直觉**：深层网络里每一层的输出分布都在漂移（internal covariate shift）。LayerNorm 把每个样本的特征向量「拉回」单位球面——只保留方向信息，扔掉模长信息。

**线代公式**：

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta, \quad \mu = \frac{1}{d}\sum_i x_i, \quad \sigma = \sqrt{\frac{1}{d}\sum_i (x_i - \mu)^2}$$

分两步看：

- **Step 1**：$x - \mu = x - \frac{1}{n}\mathbf{1}\mathbf{1}^{\mathsf{T}}x = (I - \frac{1}{n}\mathbf{1}\mathbf{1}^{\mathsf{T}})x$。这是把 $x$ 投影到全 1 向量 $\mathbf{1}$ 的正交补空间上——减去均值就是去掉 $\mathbf{1}$ 方向的分量。
- **Step 2**：除以 $\sigma = \frac{\|x - \mu\|}{\sqrt{d}}$，把向量归一化到半径为 $\sqrt{d}$ 的球面上。

**本质**：LayerNorm 让下一层接收到的信号只有「方向」（角度信息），没有「长度」（模长信息），而且范数一致。这极大稳定了训练。

**BatchNorm vs LayerNorm**：BN 沿着 batch 维度归一化（同一个特征，不同样本）→ 假设样本 i.i.d.，适合 CNN。LN 沿着 feature 维度归一化（同一个样本，不同特征）→ 不依赖 batch 统计量，适合 Transformer（序列中 token 之间互相依赖，batch 统计不可靠）。

---

### 14.3 Adam = 对角预处理改善条件数

**直觉**：SGD 对所有参数用同一个学习率，但不同参数的梯度尺度可能差几个数量级。Adam 给每个参数分配**自己的步长**——梯度波动大的参数给小块步，波动小的给大步。

**线代公式**：

SGD：$\theta_{t+1} = \theta_t - \eta \cdot g_t$

Adam：$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}$

其中 $m_t$ 是梯度的指数移动平均（动量），$v_t$ 是梯度平方的指数移动平均。

**线性代数解读**：Adam 隐式地在做**对角预处理**。损失函数的 Hessian $H = \nabla^2 L$ 的特征值往往分布极不均匀——$\lambda_1 \gg \lambda_2 \gg \dots \gg \lambda_n$。SGD 在 $\lambda_1$ 方向震荡剧烈（步长太大），在 $\lambda_n$ 方向几乎不动（步长太小）。Adam 的 $\hat{v}_t \approx \mathbb{E}[g^2]$ 估算每个参数的梯度方差——方差大的参数对应 Hessian 对角元大 → 除以 $\sqrt{\hat{v}_t}$ 相当于缩小该方向的步长。**效果**：优化景观被「拉伸」得更接近各向同性，条件数 $\kappa$ 从 $\sigma_{\max}/\sigma_{\min}$ 被压到接近 1。这就是为什么 Adam 在病态问题上几乎总是比 SGD 收敛得快。

---

### 14.4 L2 正则化 + Dropout 的线代本质

**直觉**：L2 正则化让权重变小，Dropout 随机丢弃神经元——看起来像工程 trick，但线代能解释它们为什么有效。

**L2 正则化**：

$$\text{Loss} = L_{\text{data}} + \lambda \|W\|_F^2$$

梯度更新变成：

$$W \leftarrow (1 - \eta\lambda) W - \eta \cdot \nabla_W L_{\text{data}}$$

每步先把权重缩小 $1 - \eta\lambda$ 倍，再做数据驱动的更新。等价于在 Hessian 对角线上加 $\lambda I$：

$$\kappa_{\text{new}} = \frac{\sigma_{\max} + \lambda}{\sigma_{\min} + \lambda}$$

原始条件数 $\sigma_{\max}/\sigma_{\min}$ 可能很大，加了 $\lambda$ 后分子分母都上移，比值被**近似有界化**——$\kappa_{\text{new}} \leq \sigma_{\max}/\lambda + 1$（当 $\sigma_{\min} \to 0$ 时）。这就是 weight decay 的「数值稳定器」作用。

**Dropout**：

随机把激活值置零。线代视角：每个 dropout mask 从权重矩阵 $W$ 中**随机选择一组基向量**（$W$ 的列）。模型被迫在任何基向量子集上都能正常工作 → $W$ 的各列之间不能过度互相依赖（co-adaptation）→ Gram 矩阵 $W^{\mathsf{T}}W$ 变得更对角占优 → 接近正交 → 条件数更好。换句话说，Dropout 在隐式地迫使权重矩阵「正交化」。

---

### 14.5 卷积的线性代数本质

**直觉**：卷积看起来像一个特殊的「图像操作」——滑动窗口、局部感受野、权重共享。但从线性代数角度看，**卷积就是结构化的稀疏矩阵乘法**。卷积核定义了一个特殊的矩阵：1D 卷积对应 Toeplitz 矩阵，2D 卷积对应 doubly-block-circulant 矩阵。所谓「滑动窗口」，不过是这个特殊矩阵的稀疏结构在物理上的体现。

**线代公式**：

1D 卷积 $y = w * x$ 可以写成矩阵乘法：

$$y = C x, \quad C = \begin{bmatrix}
w_0 & w_{-1} & \cdots & 0 \\
w_1 & w_0 & \ddots & \vdots \\
\vdots & \ddots & \ddots & w_{-1} \\
0 & \cdots & w_1 & w_0
\end{bmatrix}$$

$C$ 是一个 **Toeplitz 矩阵**——每条对角线上的元素相同。2D 卷积则对应 **doubly-block-circulant 矩阵**（块循环矩阵的块循环矩阵）。关键在于：这些矩阵虽然看起来很大，但**自由度极少**——一个 $3 \times 3$ 的卷积核只有 9 个参数，却定义了一个巨大的稀疏矩阵。

**im2col 展开**：实际实现中，卷积通过 `im2col` 把输入图像的重叠 patch 展开成矩阵的列：

$$X_{\text{im2col}} = \text{Unfold}(X) \in \mathbb{R}^{k^2 \times (H' \cdot W')}$$

然后卷积变成一次矩阵乘法：

$$Y = W \cdot X_{\text{im2col}}, \quad W \in \mathbb{R}^{C_{\text{out}} \times (C_{\text{in}} \cdot k^2)}$$

PyTorch 中 `nn.Unfold` 就是 im2col 的官方实现。`nn.Conv2d` 底层调用 `torch.nn.functional.conv2d`，其核心正是 im2col + 矩阵乘法。

**与傅里叶变换的联系**：循环矩阵（circulant matrix）被 DFT 矩阵 $F$ 对角化：

$$C = F^{-1} D F$$

其中 $D$ 的对角线是卷积核的傅里叶变换。这意味着：**空间域的卷积 = 频率域的逐点乘法**。这就是为什么大核卷积可以用 FFT 加速——复杂度从 $O(n^2)$ 降到 $O(n \log n)$。`torch.fft` 可以直接验证这一点。

**关键洞察**：一个 $3 \times 3$ 卷积、64 输入通道、64 输出通道，等效于一个 $64 \times (64 \cdot 9) = 64 \times 576$ 的矩阵乘法。但因为有 Toeplitz 结构，参数只有 $64 \times 64 \times 9 = 36,864$ 个，而对应的稠密矩阵需要 $64 \times 576 = 36,864$……等等，这里参数一样多？不对——Toeplitz 结构约束的是**同一个矩阵里的参数共享**：卷积核在空间上滑动意味着同一个 $3 \times 3$ 权重被应用到所有空间位置，等价于 $W_{\text{conv}}$ 的每一行都由同一个 9 元素向量以不同偏移重复构成。这比稠密矩阵少了 $(H' \cdot W') / 9$ 倍的参数。

**转置卷积（反卷积）**：同样的思想——转置卷积等价于 **Toeplitz 矩阵的转置乘以输入**。前向卷积是 $y = Cx$，转置卷积是 $y = C^{\mathsf{T}}x$。`nn.ConvTranspose2d` 做的事就是矩阵乘法的伴随操作。

---

### 14.6 残差连接：给梯度修一条高速公路

**直觉**：深层网络最大的敌人是梯度消失——反向传播经过几十层后，梯度指数衰减到零。残差连接给了一个简单到极致的解决方案：

$$h_{l+1} = h_l + F(h_l)$$

恒等映射 $h_l$ 不经过任何变换直接传递到下一层，$F(h_l)$ 只学习「残差」——当前层需要**改变**的那部分。如果学什么都不如保持原样好，$F(h_l)$ 可以轻松学成零映射（比学恒等映射容易得多）。

**线代公式**：

考虑一个残差块：$h_{l+1} = h_l + W_2 \cdot \sigma(W_1 \cdot h_l)$。定义 $F(h_l) = W_2 \cdot \sigma(W_1 h_l)$，则反向传播的雅可比矩阵为：

$$\frac{\partial h_{l+1}}{\partial h_l} = I + \frac{\partial F}{\partial h_l}$$

关键在于那个 **$I$ ——单位矩阵**。没有残差连接时，雅可比是 $\partial F / \partial h_l$，其特征值可以全部 $< 1$，导致梯度在经过多层后指数衰减 $\lambda_{\max}^L \to 0$。有残差连接后，雅可比的特征值是 $1 + \varepsilon$，其中 $\varepsilon$ 是 $\partial F / \partial h_l$ 的特征值。即使 $\varepsilon$ 很小，梯度也能通过 $I$ 项无损传播。

**深层残差网络的特征值分析**：

堆叠 $L$ 个残差块，从第 $L$ 层传回第 1 层的梯度为：

$$\frac{\partial h_L}{\partial h_1} = \prod_{l=1}^{L-1} \left(I + \frac{\partial F_l}{\partial h_l}\right)$$

展开这个乘积：$I + \sum_l \frac{\partial F_l}{\partial h_l} + \sum_{l < m} \frac{\partial F_l}{\partial h_l}\frac{\partial F_m}{\partial h_m} + \cdots$。**梯度包含了所有层的直接贡献**（一阶项 $\sum_l$），也包含了不同层的交互项（高阶交叉项）。这意味着信息可以从任意深度直接传递到浅层——梯度信号不会被「压缩」成只有最后几层的信息。

**PyTorch 对应**：

```python
# 经典残差块
def forward(self, x):
    residual = x
    out = self.conv2(self.relu(self.conv1(x)))
    out += residual    # ← 恒等映射的「高速公路」
    return out

# 或更简洁的写法
x = x + self.net(x)
```

`+=` 就是残差连接的全部秘密。

**关键洞察**：残差连接最深刻的线性代数含义是——它确保了反向传播的雅可比矩阵**至少有一个特征值为 1 的路径**。这意味着无论网络多深，梯度范数不会指数衰减到零。ResNet-152、Transformer 的 96 层，之所以可训练，根本原因就在这里。从优化景观看，残差连接让损失曲面更平滑、条件数更接近 1——这跟 Adam 和 L2 正则化的效果是**同一个方向**：改善优化问题的数值条件。

---

### 14.7 权重初始化的数学

**直觉**：训练深度网络就像走钢丝——初始化太大，激活值在前向传播中爆炸，梯度在反向传播中爆炸；初始化太小，激活值和梯度都消失到零。好的初始化让每层的激活方差和梯度方差都保持恒定，信号可以稳定流过整个网络。

**问题形式化**：

设第 $l$ 层的激活为 $h_l \in \mathbb{R}^{n_l}$，权重为 $W_l \in \mathbb{R}^{n_l \times n_{l-1}}$。前向：$h_l = W_l \cdot h_{l-1}$（忽略偏置和激活函数，先分析线性部分）。目标是：

$$\text{Var}(h_l) \approx \text{Var}(h_{l-1}), \quad \text{Var}\left(\frac{\partial L}{\partial h_l}\right) \approx \text{Var}\left(\frac{\partial L}{\partial h_{l+1}}\right)$$

**前向传播的方差分析**：

假设 $W_{ij}$ 和 $h_j$ 独立同分布，均值为 0。则 $h_i = \sum_{j=1}^{n_{\text{in}}} W_{ij} h_j$ 的方差：

$$\text{Var}(h_i) = \sum_{j=1}^{n_{\text{in}}} \text{Var}(W_{ij} h_j) = \sum_{j=1}^{n_{\text{in}}} \text{Var}(W_{ij}) \cdot \text{Var}(h_j) = n_{\text{in}} \cdot \text{Var}(W) \cdot \text{Var}(h)$$

要使 $\text{Var}(h_i) = \text{Var}(h_j)$，需要：

$$\text{Var}(W) = \frac{1}{n_{\text{in}}}$$

**反向传播的方差分析**：

反向传播 $\frac{\partial L}{\partial h_{l-1}} = W_l^{\mathsf{T}} \frac{\partial L}{\partial h_l}$。类似推导：

$$\text{Var}\left(\frac{\partial L}{\partial h_{l-1}}\right) = n_{\text{out}} \cdot \text{Var}(W) \cdot \text{Var}\left(\frac{\partial L}{\partial h_l}\right)$$

要保持梯度方差不变，需要：

$$\text{Var}(W) = \frac{1}{n_{\text{out}}}$$

**Xavier / Glorot 初始化**：前向和反向的约束通常不能同时满足（$n_{\text{in}} \neq n_{\text{out}}$）。折中方案——取两者的调和平均：

$$\text{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}}$$

从这个方差的均匀分布 $U[-\sqrt{6/(n_{\text{in}}+n_{\text{out}})}, \sqrt{6/(n_{\text{in}}+n_{\text{out}})}]$ 或正态分布 $\mathcal{N}(0, 2/(n_{\text{in}}+n_{\text{out}}))$ 中采样。

**He / Kaiming 初始化**：ReLU 把一半的激活值设为零→前向传播中方差减半：

$$\text{Var}(h_l) = \frac{1}{2} n_{\text{in}} \cdot \text{Var}(W) \cdot \text{Var}(h_{l-1})$$

补偿这个 $\frac{1}{2}$ 因子：

$$\text{Var}(W) = \frac{2}{n_{\text{in}}}$$

对于 PReLU（斜率为 $a$）：$\text{Var}(W) = \frac{2}{(1 + a^2) n_{\text{in}}}$。

**线性代数视角**：

这些初始化方案的共同目标：让每一层的权重矩阵 $W$ 在初始化时满足 $\|W\|_2 \approx 1$。这意味着 $W$ 的最大奇异值接近 1 → 条件数 $\kappa(W) \approx 1$ → 前向和反向的信号都不被过度放大或压缩。从优化角度看，这确保了训练初期损失曲面的 Hessian 条件数保持在合理范围内——梯度下降可以稳定进行。**这是为什么从 scratch 训练深层网络之前，初始化方案的选择是最关键的决策之一。**

**PyTorch 对应**：

```python
# nn.Linear 默认使用 Kaiming uniform 初始化
layer = nn.Linear(512, 256)  # 自动初始化好了

# 手动使用 Xavier 初始化
nn.init.xavier_uniform_(layer.weight)
nn.init.xavier_normal_(layer.weight)  # 或正态分布版本

# He 初始化（Kaiming）
nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu', a=0.01)
```

---

### 14.8 Transformer 全景：一切皆线代

**直觉**：2017 年 Transformer 提出以来，它几乎统治了 NLP、CV、语音等所有 AI 领域。很多人觉得 Transformer 很复杂、很「黑盒」。但把变压器拆开来看——**它的每一个组件都是本课程学过的线性代数操作**。没有魔法，只有线性代数的巧妙组合。

**Transformer 块的完整矩阵流**：

输入：$X \in \mathbb{R}^{n \times d}$（$n$ 个 token，每个 $d$ 维）

**(1) Multi-Head Attention**（§14.1）：

$$\text{Attn}(X) = \text{softmax}\left(\frac{XW_Q (XW_K)^{\mathsf{T}}}{\sqrt{d_k}}\right) (XW_V)$$

- $XW_Q (XW_K)^{\mathsf{T}}$：内积矩阵 → token 之间的相似度
- $\text{softmax}$：把相似度变成概率权重
- 乘以 $XW_V$：对 value 做加权求和

**(2) 残差连接 + LayerNorm**（§14.6 + §14.2）：

$$X_1 = \text{LayerNorm}(X + \text{Attn}(X))$$

- $X + \text{Attn}(X)$：残差连接确保梯度畅通
- $\text{LayerNorm}$：投影到单位球面，稳定分布

**(3) Feed-Forward Network (FFN)**（§13.3）：

$$\text{FFN}(X_1) = \text{ReLU}(X_1 W_1 + b_1) W_2 + b_2$$

- $W_1 \in \mathbb{R}^{d \times d_{\text{ff}}}$（通常 $d_{\text{ff}} = 4d$）：升维 → 提供更大的容量
- $\text{ReLU}$：引入非线性
- $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d}$：降维 → 回到原始维度

**(4) 残差连接 + LayerNorm**（再次）：

$$X_2 = \text{LayerNorm}(X_1 + \text{FFN}(X_1))$$

**(5) 最终层**：

$$\text{logits} = X_2 W_{\text{unembed}}$$

- $W_{\text{unembed}} \in \mathbb{R}^{d \times |V|}$：从 embedding 空间投影回词汇空间

**线代组件全景对应**：

| Transformer 组件 | 线代操作 | 本课程章节 |
|---|---|---|
| Embedding | 基变换 | §12 |
| Positional Encoding | 嵌入空间的加性扰动 | — |
| $QK^{\mathsf{T}}$ | 内积矩阵 | §14.1 |
| $\text{softmax}$ | 归一化 + 非线性 | §13.3 |
| Self-Attention | 加权求和（概率 × V） | §14.1 |
| 残差连接 $x + f(x)$ | 恒等映射 + 雅可比的 $I$ 项 | §14.6 |
| LayerNorm | 投影到单位球面 | §14.2 |
| FFN $W_2 \cdot \text{ReLU}(W_1 x)$ | 分段线性变换 | §13.3 |
| 反向传播 | 链式法则 + 矩阵转置 | §13.4 |

**为什么 Transformer 能 scale**：

1. **并行性**：Self-Attention 的 $QK^{\mathsf{T}}$ 虽然复杂度 $O(n^2)$，但没有序列依赖——所有 token 同时计算。这是 RNN 做不到的。
2. **梯度流**：残差连接确保雅可比矩阵有特征值为 1 的路径，梯度可以穿过任意多层。Transformer 的 96 层之所以可训练，不是偶然。
3. **数值稳定**：LayerNorm 持续把激活拉回单位球面 + 残差连接保持恒等路径 → 整个网络的雅可比条件数始终接近 1。训练初期用 Kaiming 初始化（§14.7）+ warmup 学习率 → 平稳起飞。

**关键洞察**：Transformer 没有发明任何新的数学。它的每一个操作——内积、softmax、残差、归一化、矩阵乘法——都是经典线性代数。**Transformer 的创新不在于用了新的数学工具，而在于用已知的工具设计了一种新的组合方式**：用 attention 替代 RNN 的序列传播，用残差保障深度，用 LayerNorm 维护数值稳定。理解了线性代数，Transformer 就不再是黑盒——它是一个优雅的矩阵变换流水线。

---

> **下一步**：学完了 ML 场景全景地图，去 [💭 个人理解与资源](insights-and-resources.md) 回顾这些知识如何改变你看 AI 的方式。

---

### 14.9 K-FAC = Kronecker 结构加速自然梯度

**直觉**：普通梯度下降沿着「最陡方向」走，但这个方向依赖于参数化方式。自然梯度沿着「分布空间的最陡方向」走——参数化无关，理论上更优。代价是需要 Fisher 信息矩阵的逆，$O(n^3)$ 不可接受。K-FAC 利用神经网络层的特殊结构，把大矩阵求逆拆成两个小矩阵的 Kronecker 积求逆。

**线代公式**：

自然梯度：$\theta \leftarrow \theta - \eta \cdot F^{-1} \nabla L$，其中 $F$ 是 Fisher 信息矩阵。

对于一个线性层 $y = Wx$，K-FAC 的关键洞察：

$$F \approx A \otimes G, \quad A = \mathbb{E}[x x^{\mathsf{T}}], \quad G = \mathbb{E}[\nabla_y L \, (\nabla_y L)^{\mathsf{T}}]$$

$A$ 是激活的协方差矩阵（$\mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$），$G$ 是输出梯度的协方差矩阵（$\mathbb{R}^{d_{\text{out}} \times d_{\text{out}}}$）。利用 Kronecker 积的性质：

$$F^{-1} \approx A^{-1} \otimes G^{-1}$$

**复杂度从 $O((d_{\text{in}} d_{\text{out}})^3)$ 降到 $O(d_{\text{in}}^3 + d_{\text{out}}^3)$！**

**PyTorch 对应**：虽无官方实现，但第三方库（如 `kfac-pytorch`）的做法是：前向时收集 $A$ 和 $G$ 的统计量，每 $N$ 步用 Cholesky 或 SVD 求逆，然后对梯度做预处理。

**关联本课程**：K-FAC 几乎用到了我们学过的所有东西——Kronecker 积（矩阵微积分）来分解 Fisher 矩阵，Cholesky / QR 分解来稳定求逆，条件数的概念来判断预处理是否有效。一个算法串起了线性代数的半壁江山。

---

> 📝 **本章习题**：[基变换与神经网络 · 习题与思考](basis-and-neural-networks-exercises.md)

## 参考文献

- Goodfellow, I., Bengio, Y., & Courville, A. *Deep Learning*. MIT Press, 2016.
- Deisenroth, M. P., Faisal, A. A., & Ong, C. S. *Mathematics for Machine Learning*. Cambridge University Press, 2020.
- Martens, J. & Grosse, R. "Optimizing Neural Networks with Kronecker-factored Approximate Curvature." *ICML*, 2015.
