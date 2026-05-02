# 习题总览

每章习题分 A 档（巩固基础）和 B 档（培养 ML 直觉）。每题附**完整标准答案**（含推导步骤，而非简要提示）。

| 章节 | A 档 | B 档 | 合计 | 覆盖知识点 |
|------|:----:|:----:|:----:|-----------|
| [一、从向量到秩](vectors-to-rank-exercises.md) | 15 | 10 | **25** | 向量运算与内积几何、线性组合与张成、矩阵四种乘法（列/行/外积/元素）、矩阵=线性变换、线性无关/基/维数、极大无关组、秩、六种范数（$L_1$/$L_2$/$L_\infty$/Frobenius/谱/核）、$L_1$ vs $L_2$ 正则化、范数等价性 |
| [二、方程与子空间](elimination-and-subspaces-exercises.md) | 11 | 5 | **16** | LU 分解手算与多右端复用、四个基本子空间识别（含基与维数）、齐次/非齐次解结构、正交投影与最小二乘、QR 分解与 Gram-Schmidt、Cholesky 分解、伪逆与条件数、正交矩阵性质 |
| [三、行列式、特征值与 SVD](determinant-eigen-svd-exercises.md) | 7 | 6 | **13** | 行列式性质真伪+$3\times3$手算、特征值/特征向量/对角化/谱分解、正定性五种等价判定、二次型→标准形、矩阵微积分（线性层/softmax/交叉熵梯度）、SVD 几何三部曲、Rayleigh 商与收敛速度 |
| [四、基变换与神经网络](basis-and-neural-networks-exercises.md) | 5 | 10 | **15** | 相似不变量证明、纯线性层坍缩、ReLU 分段线性区域分析、反向传播手算链式法则、Embedding=基变换、LayerNorm 投影几何、Adam 对角预处理、残差连接雅可比 $I + \partial F/\partial h$、Xavier/He 方差推导、卷积=Toeplitz 矩阵、Attention 秩分析、Ridge 回归条件数 |
| **总计** | **38** | **31** | **69** | |

> 💡 习题覆盖中西主流教材（Strang、Lay、Axler、Deisenroth、同济教材）的核心题型。所有答案均为完整推导，不再只是提示。
