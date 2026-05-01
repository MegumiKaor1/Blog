# 💭 个人理解与资源

## 线性代数是一门「关系」的语言

学完线代回头看 AI，所有东西都是同一个主题的变奏：

| AI 概念 | 线代本质 | 一句话 |
|---------|---------|--------|
| 梯度下降 | 在参数空间中沿负梯度方向移动 | 最陡下山方向 = 负梯度 |
| Attention | $QK^T$ = 内积矩阵，用 softmax 归一化后加权求和 | 谁和谁关系近，就给谁更多权重 |
| PCA | 数据矩阵的截断 SVD | 找到方差最大的方向，丢掉噪音 |
| LoRA | 对权重矩阵做低秩修正 $\Delta W = BA$ | 只动几个最重要的「方向」 |
| Embedding | 为离散对象选一个好的基 | 让语义关系变成向量运算 |
| LayerNorm | 投影到单位球面 | 规范化向量的「方向」，忽略「长度」 |

## 为什么「先看见再算」是对的

线性代数有两种教学极端：一种死算（行列式展开、求逆矩阵，算到怀疑人生），一种纯抽象（从域和公理开始，学到第三章还不知道这和现实有什么关系）。

MIT 的 Gilbert Strang 走了第三条路——他的 18.06 课从第一分钟起就在画图。矩阵乘法不是「行乘列」，而是「列的线性组合」。$2 \times 2$ 的小例子他能讲半小时，因为他要确保你**看见**几列向量是怎么拼出结果的，看见消元在几何上到底改变了什么，看见四个子空间是怎么从同一个矩阵里长出来的。

3Blue1Brown 把这种「看见」做到了极致——动画里一条条向量被扭来扭去，一个正方形被扯成平行四边形，行列式就是面积缩放了多少。有些东西你看十遍公式不一定能穿透，看一遍动画就通了。

这两者结合是我试过的最有效的学习路径：

1. 看 3Blue1Brown 建立**几何直觉**（「噢，原来是这么回事！」）
2. 读 Strang 的教材或看他的视频，建立**系统性的知识结构**（「原来这些概念是这样串起来的」）
3. 翻开 PyTorch 文档，看到 `nn.Linear` 不再是一团黑，而是一个矩阵在等着做线性变换

## 为什么线代让你变成更好的 ML 工程师

不是因为你学会了手算 SVD（你不需要，PyTorch 需要），而是因为你有了**直觉**。

**你不需要推导就能判断的事**：

- Loss 不降了 → Hessian 可能接近奇异，某些方向的梯度信息消失了 → 换个初始化或调学习率
- 权重矩阵的奇异值分布极度不均匀 → 大部分方向没用 → 可以考虑低秩压缩或 LoRA
- 输入特征高度相关 → 数据矩阵秩亏 → PCA 前的协方差矩阵条件数差 → 先做标准化
- LoRA 的 rank 设太大浪费显存，设太小学不到东西 → 就是截断 SVD 的 $k$ 选多少的问题

这些不是调参玄学，是线代告诉你的答案。你会因为理解了底层数学而做出更好的工程判断。

## 如果你想继续深入

线代学到这个程度，已经足够支撑你阅读大多数 ML 论文的数学部分了。但如果还想更深，以下方向值得考虑：

- **数值线性代数**：条件数、浮点误差、迭代法（共轭梯度、GMRES）——理解为什么 `float32` 训练的某些模型会崩溃
- **矩阵微积分**：$\frac{\partial}{\partial W} \text{tr}(W^T A W)$ 这种——反向传播的本质
- **随机矩阵理论**：大随机矩阵的奇异值分布——理解为什么大模型的某些行为是「必然的」而非「玄学的」
- **张量分解**：CP 分解、Tucker 分解——卷积层和多模态模型背后的数学

---

## 相关链接

- [**《Introduction to Linear Algebra》第 6 版**（Gilbert Strang）](https://math.mit.edu/~gs/linearalgebra/) —— 系统性学习的最佳选择
- [**MIT 18.06 公开课**（Strang 亲授，34 讲）](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) —— 不想看书就看视频，他在黑板前如讲故事般展开
- [**3Blue1Brown「线性代数的本质」**](https://www.3blue1brown.com/topics/linear-algebra) —— 几何直觉的巅峰，所有公式都会有画面感
- [**MIT 18.065 — Matrix Methods in Data Analysis**（Strang 的新课）](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/) —— 深度学习时代的矩阵方法，SVD/PCA/随机矩阵/优化
- [**《Matrix Analysis》**（Horn & Johnson）](https://www.cambridge.org/9780521548236) —— 进阶词典：特征值扰动、矩阵不等式、Kronecker 积
- [**《Numerical Linear Algebra》**（Trefethen & Bau）](https://people.maths.ox.ac.uk/trefethen/text.html) —— 数值线性代数的经典，读完后不再写出数值灾难
- [**《Matrix Calculus》**（Magnus & Neudecker）](https://www.janmagnus.nl/misc/matrix-calculus/) —— 矩阵导数百科全书，反向传播的本质
