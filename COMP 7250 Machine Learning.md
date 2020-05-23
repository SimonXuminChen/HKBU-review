# COMP 7250 复习笔记

## Chapter 1 Introduction

### 1.1 Statistical Machine Learning 统计机器学习

**学习步骤：**给定带有label的data，给定一个算法，结合先验知识和特征去学习该算法的参数（学习模型），然后通过validation dataset去选择算法的参数，最后通过test sample去验证该算法的效果。



#### 定义

Spaces: 模型的参数空间，比如输入空间和输出空间

Loss function：损失函数$$L:Y*Y \rightarrow R$$，一般用$$L(\hat{y},y)$$ 来表示，其中$$\hat{y}$$是预测的结果

通常损失函数有:

- 二分类问题：0-1loss，记作$$L(\hat{y},y) = 1_{y \neq y'}$$
- 回归问题：平方误差



Hypothesis set：假设集，学习器从模型空间中选取的它的可能假设子集。

- 由feature决定
- 表示关于学习任务的先验知识



#### 监督学习

Training data 独立同分布地从服从D分布的X*Y空间中获取m个样本S。格式为(x,y)元组集合。

目标问题：找到一个在H中的假设h，并使其满足最小泛化误差

- deterministic case：输出结果由输入确定的函数
- stochastic case：输入的输出概率函数

#### Supervised vs. Unsupervised Learning

| 监督学习     | 无监督学习 |
| ------------ | ---------- |
| 概率监督学习 | PCA        |
|     决策树         |    K-means clustering        |
|       SVM       |    深度无监督学习（GAN）        |
|深度监督学习||





#### Errors

Generalization error：泛化误差，$$R(h) = E_{(x,y)~D}[L(h(x),y)]$$

Empirical error：经验误差（期望误差），也就是训练集的平均误差,$$\hat{R}(h)=\frac{1}{m}\sum_{i=1}^mL(h(x_i),y_i)$$

Bayes error:泛化误差的下界,$$R^* = inf R(h)$$,这个也叫做平均噪声。

- 在deterministic case中，Bayes error为0，因为一个label能被一个观测函数唯一确定。
- 在stochastic case中，$$\min R(h)\gt 0$$



#### Noise：

衡量任务难度的特征，如果接近0.5，则称$x$为噪点

在二分类问题中，噪声是$$noise(x) = \min \{Pr[1|x],Pr[0|x]\}$$

可以观测到$$E[noise(x)] = R^*$$，即平均噪声就是Bayes error。



#### Overfitting & Underfitting

fitting拟合表示了模型（学习器）的简易程度或复杂程度。

在样本中学习到最好的h并不一定是对于所有数据来说都是最好的那个。



如何找到一个假设集复杂度与样本大小的trade-off？如果模型过于复杂，而样本数量小，会导致overfitting；如果模型过于简单，样本数量大，会导致underfitting。



#### 模型选择 model selection

$$R(h)-R^*=[R(h)-R(h^*)]+[R(h^*)-R^*]$$，

其中

- $$h^*$$是best in class的假设，假设集H中最好的那个，注意，假设集是基于样本来决定的。
- $$[R(h)-R(h^*)]$$为估计误差，我们希望约束模型的唯一条件
- $$[R(h^*)-R^*]$$为近似误差，不是随机变量，由H决定。



#### 经验误差最小化 Empirical Risk Minimization （ERM）

我们希望可以找到假设集中的一个假设h，使其满足$$h = argmin \hat{R}(h)$$

但是有并不理想的情况会发生：

- 假设集H可能很复杂
- 样本的数量不够多

#### 泛化边界 Generalization Bounds

定义一个规则，即允许泛化误差可以有错误，但是这个泛化误差有最大值即upper bound，如果超过了这个上界就会影响机器学习的可行性。_（PAC可学习）_

Upper bound on $$Pr[\sup |R(h)-\hat{R}(h)|\gt \epsilon]$$

由ERM给出的假设h的估计误差的界。







#### 结构风险最小化 Structural Risk Minimization(SRM)

原理：考虑到有无限假设集的情况，我们要对其进行约束，所以添加一个惩罚项。

$$h = argmin \hat{R}(h)+penality(H_n,m)$$

SRM拥有很强的理论基础保证，但是通常计算很困难。



还有一个是基于正则化的算法结构：$$\lambda \geq 0$$

$$h = argmin \hat{R}(h)+\lambda ||x||^2$$



#### 机器学习的发展顺序

从Rule-based systems到Classic machine learning，再到表示学习，表示学习也包括近代的统计机器学习和深度学习。



## chapter 2 概率机器学习

### 概率和分布

条件概率conditional probability

$$P(Y=y|X=x)=\frac {P(Y=y,X=x)}{P(X=x)}$$

#### Conditional Independence

独立性

$$P(x,y)=P(x)P(y)$$

条件独立

$$P(x,y|z)=P(x|z)P(y|z)$$



#### 伯努利分布Bernoulli Distribution

也叫二项分布



#### 高斯分布 Gaussian Distribution



#### Dirac and Empirical Distribution

#### 混合高斯分布



### Logistics Function

$$\sigma(x)=\frac {1}{1+exp(-x)}$$

图像呈s型，值域为[0,1]



#### Softplus Function

ReLU的优化版本，ReLU的图像呈折页状，小于0的值为0，大于0的值为其本身，此时有很明显的拐角而softplus软化了该拐角，公式为：

$$\xi(x)=\log(1+exp(x))$$



### Bayes' Rule

$$P(x|y)=\frac {P(x)P(y|x)}{P(y)}$$



### Information Theory

- Information	
- Entropy
- KL divergence



#### Shannon Entropy

衡量信息的纯度



#### KL divergence

衡量两个概率分布的匹配程度的指标，两个计算的顺序不同，会导致计算的结果不同，这说明KL散度是非对称的。

$$D_{KL}(P||Q) = \mathbb{E}_{x~P}[\log \frac{P(x)}{Q(x)}] = \mathbb{E}_{x~P}[\log P(x)-\log Q(x)] = \sum_{i=1}^N p_i \log \frac {P(x_i)}{Q(x_i)}$$



#### 概率PCA

加入了一个服从正太分布的噪音，当这个噪音$$\sigma \rightarrow 0$$的时候，则变回原来的PCA

#### 独立成分分析Independent Component Analysis 

#### Sparse Coding

在线性模型的基础上，加上各项同精度噪声。



### Structured Probabilistic Model

因为大多数变量之间互相影响，但是大多数变量之间的影响不是直接的因此我们可以通过图去描述这个影响，图的类别有：

- 有向图-可以直接看出影响关系
- 无向图-不能直接看出影响关系



### Sampling

#### Ancestral Sampling 祖先采样法

针对有向图模型的基础采样方法，指从模型所表示的联合分布中产生样本。



#### Gibbs Sampling 

MCMC的应用方法。

循环所有变量，遵循一定的概率分布，查看会发生什么事件。

迭代到一定次数后，会收敛到p(x)





#### Latent variable model

#### Restrict Boltzmann Machine(RBM)

一起可以实现有效的块吉布斯采样，该采样在同时采样所有h和所有v之间交替进行。

训练模型以推导关于data v的表示h



### Monte Carlo Methods

#### why sampling?

- 可以节省计算开销
- 对于难解的求和和求导问题，提出一个近似解。

无偏:有限的n的期望值等于正确值

低方差O(1/n)



#### why importance sampling

- 降低估计的值的方差
- 对所有的q来说，任然是无偏的



## Chapter 3 Deep Learning

普通的学习器无法对非线性的情况进行处理，所以想了一个办法，就是想样本从原本的空间中投影到隐层空间去处理。

#### Why ReLU?

Nonlinear的mapping，同时可以减少梯度消失的问题。 

### 基于梯度的学习 Gradient-based Learning

有三个特点，分别是损失函数，输出单元以及隐层单元

损失函数根据学习任务来决定如何使用，如果是条件概率分布，则使用Maximum Likelihood；如果是训练集和模型的分布，则用Cross-entropy；如果是条件统计，可以采用均方误差，方差和绝对平均误差，不过绝对平均误差对基于梯度优化的效果并不好。

输出单元采用的方法类型一般有直接线性计算，sigmoid，softmax

隐层单元采用激活函数，如ReLU，Softplus, GELU



#### Universal approximation theorem

一层隐藏层足以表达任意函数的近似值到任意精确程度，因此，神经网络包含了隐藏层和输出层至少可以近似任何想要的非零的可测量函数。

> 更深的网络有更好的泛化能力

### FP & BP

前馈神经网络和回溯神经网络



### CNN

Motivation：

- Sparse Interaction（Sparse Connectivity）-卷积核，不需要所有参数
- Parameter Sharing-采用共同的参数，为了使繁华性能优化
- Equivariant representation-输入变化了，输出同样改变



Max Pooling

找到最大的那个值，而不是所有值。只对最大值敏感。



**2-D Convolution**:采用2*2的卷积核，对输入数据进行卷积。本质是矩阵乘法。

通过卷积和池化去提取特征，可以得到较强的先验知识。

**Stride**:有步长的卷积，可以提高速度，减少计算开销

**zero padding**:对元数据的四周补0，防止特征表示缩小。



### ResNet 残差结构

输入值会与两层后的输出进行相加，结构为单线



### ResNeXt

ResNet的变种，结构不再是单线，而是先将每一层拆分成多个，然后经过三层神经网络后，进行合并再与输入值相加。



### WideResNet

也是一个变种，采用多种层次结构方法，使更加神经网络的层更加广。



### RNN

参考NLP笔记

### LSTM

参考NLP笔记



### sequence2sequence, Attention, GNU



### Clipping Gradients

为了防止梯度爆炸，当梯度的值大于某个阈值时，$$g \leftarrow \frac {gv}{||g||}$$



### Information Flow

为了防止梯度消失



## Chapter 4 Regularization and Optimization

### Regularization正则化

当模型的容量大于样本数量时，我们需要通过正则化来解决overfitting的问题。 

对学习模型做的修改不是为了降低训练误差，而是为了降低泛化误差。

**L1正则可以用来做特征选择**，支持稀疏，相当于服从拉普拉斯先验的最大后验贝叶斯估计

**L2正则也被成为权重衰减**，支持小的权重，相当于服从高斯分布的最大后验贝叶斯估计





### Dataset Augmentation

给数据添加噪音，或旋转，偏移，翻转等操作，增加数据量。



### semi-supervised learning & multi-task learning



### early stopping

当验证集的表现比较好，且满足我们的需求时，提前终止。



### Bagging

集成学习，采用放回抽样去获取数据集，然后多个强学习器一起学习，对学习的结果去平均值，从而使判别器可以更具有鲁棒性。



### Dropout

同样防止过拟合，在训练的过程中，drop掉一些点和权重，在验证集和测试集上，不做drop操作。



### Adversarial Training 对抗学习



### Optimization 优化问题

#### 凸优化和非凸优化问题

凸优化更容易得到最优解，非凸优化可能找不到最优解，或者卡在鞍点上。



优化目标就是经验风险最小化



#### batch 和 mini-batch

batch 梯度下降，一次性对所有w进行更新，很快就能到达最优点，但是计算量大

mini-batch，对一小批数据进行更新

随机梯度下降，收敛较慢，一次只随机更新一个w



#### local Minima

局部最优不等于全局最优，

#### Saddle Points 鞍点

学习器经常会卡在这个点上



### 基础优化算法

#### SGD

随机梯度下降，所有参数中只随机更新一个

#### Momentum

结合之前的权重考虑，需要初始化速度v



#### AdaGrad

自适应学习率，需要设置全局学习率$$\epsilon$$，一个较小的常数$$\delta $$，一般是$$10^{-7}$$，同时还需要一个累积变量$$r$$=0

每次更新，先求g，得到梯度，然后求累计的$$r \leftarrow r=r+g \bigodot g$$，最后用这些值去求一个更新的度量:$$\Delta\theta \leftarrow -\frac{\epsilon}{\delta+\sqrt {r}} \bigodot g$$

最后根据这个结果更新参数$$\theta \leftarrow \theta+\Delta\theta$$

#### RMSProp

在AdaGrad的基础上进行修改，新增一个衰减率$$\rho$$，需要设置全局学习率$$\epsilon$$，一个较小的常数$$\delta $$，一般是$$10^{-6}$$，同时还需要一个累积变量$$r$$=0

每次更新，先求g，得到梯度，然后求累计的$$r \leftarrow r=\rho r+(1-\rho)g \bigodot g$$，最后用这些值去求一个更新的度量:$$\Delta\theta \leftarrow -\frac{\epsilon}{\sqrt {\delta+r}} \bigodot g$$

最后根据这个结果更新参数$$\theta \leftarrow \theta+\Delta\theta$$

#### Adam

一般来说，熟悉优化的人都会采用Momentum SGD，而当不熟悉优化的时候，一般都采用Adam



![image-20200514103542629](/Users/Simonchan/Library/Application Support/typora-user-images/image-20200514103542629.png)

Adam优化了$$\delta$$这个参数



#### 二阶方法牛顿法

构建黑森矩阵，求出所有参数的二阶导去更新



## Chapter5 Bayes Classifier and Likelihood Methods

### Bayes Error

在第一章中也提到过的Bayes Error，它是用来度量学习器的最小误差，在确定性学习中，它的误差为0，但是在不确定学习(随机性学习)中，它不为0。可以用条件概率的形式表达：

$$\forall x \in X,y\in \{0,1\}, h_{Bayes}(x)=argmax \mathbb{P}[y|x]$$

#### Upper bounding 0-1 loss

可以推导



### Entropy

离散型的随机变量可以用$$H(X)=-E[log p(X)]=-\sum_{x\in X}p(x)log p(x)$$计算

其中$$H(X) \geq 0$$，entropy是以2为底的，因为这是用来表示信息的纯度，而在通信中，信息是以0/1字节表示，只有两个数字，所以以2为底。

entropy有一种特殊情况就是Renyi entropy，表示为$$H_a(X)=\frac{1}{1-a} log (\sum_{i=1}^n p_i^a)$$，当$$a\geq0 \& a\neq1 $$的时候成立。。



#### Jensen 不等式 Jensen's inequality

$$H(X)=E[log \frac {1}{p(X)}]\leq log E[\frac{1}{p(X)}]=logN$$



### KL Divergence （Relative Entropy）

之前介绍过，是衡量两个概率分布的匹配程度的指标，两个计算的顺序不同，会导致计算的结果不同，这说明KL散度是非对称的。

$$D_{KL}(P||Q) = \mathbb{E}_{x~P}[\log \frac{P(x)}{Q(x)}] = \mathbb{E}_{x~P}[\log P(x)-\log Q(x)] = \sum_{i=1}^N p_i \log \frac {P(x_i)}{Q(x_i)}$$

另外还有一个特性就是对所有的p和q分布而言，结果总是大于0，如果p和q相等，则值为0.



#### Bregman Divergence

两个不相同的函数，其中一个是凸函数F，衡量在某一点处（非相切点）两个函数之间的距离

$$B_F(x||y)=F(x)-F(y)-<\nabla F(y),x-y>$$

![image-20200514112639934](/Users/Simonchan/Library/Application Support/typora-user-images/image-20200514112639934.png)

 KL散度和Bregman散度不是同一个东西。



#### Renyi Divergence

瑞丽熵，用来衡量两个分布之间的纠缠程度。



#### Conditional Relative Entropy

与kl散度相似，加入了条件概率，考虑到了一个新的分布r



### Maximum Likelihood Solution

从一个分布中找到观测S出现的最大可能性，需要考虑到i.i.d

$$P_{ML}=argmax Pr[S|p] = argmax \prod_{i=1}^m p(x_i)=argmax \sum_{i=1}^mlogp(x_i)$$



#### Maximum a Posteriori (MAP) 最大后验

给出观测样本S和嘉定一个先验分布Pr[p]，求一个我们最想得到的概率分布。

$$P_{MAP}=argmax Pr[p|S] = argmax \frac{Pr[S|p]Pr[p]}{Pr[s]}=argmaxPr[S|p]Pr[p]$$



#### Features

特征工程的应用





#### MaxEnt Model Formulation

最大熵模型，优化的目标函数为带有约束的最小化kl散度。



关于最大熵模型，带有的约束是不同的，当采用L1正则的时候，约束是取绝对值；当采用L2正则的时候，约束是取模。



#### Dual Problems对偶问题

转换了对偶问题后，更好求解。所以在最大熵问题中，可以通过对偶进行正则化约束。





### Conditional MaxEnt Models

问题定义：在多分类问题中，找到sample对应的类最大的概率



二分类问题，逻辑回归，都属于CMM







## Chapter 6 SVM & Kernel

SVM有三宝：间隔、对偶、核技巧。

SVM属于判别模型。

概念：SVM属于二分类模型，目的是寻找到一个超平面可以将样本进行划分，划分的依据是**间隔最大化**，最终将目标函数转化为一个**凸二次规划问题**来求解。

Support Vector支持向量：在求解过程中，只根据部分数据就可以确定分类器，这部分数据称为支持向量。所以支持向量是**距离超平面最近的点**。

**线性可分：**在数据集中，存在一个超平面，可以将数据集中的数据完全划分，则称为该数据集线性可分。

间隔可分为：

- Hard margin 硬间隔
- soft margin 软间隔

目标函数表达式：

硬间隔： $$ min_{w,b} \frac{1}{2} ||w||^2 \qquad st. \quad y^{(i)}(w^Tx^{(i)} + b) \geq 1 $$

软间隔： $$ min_{w,b} \frac{1}{2} ||w||^2 + C \sum_{i=1}^m \xi_i \qquad st. \quad y^{(i)}(w^Tx^{(i)} + b) \geq 1 ,\quad \xi_i \geq 0 $$



损失函数：Hinge Loss



SVM解决的问题：

- 线性分类：对于n维的数据，SVM的目标是找到一个n-1维的最佳超平面来讲数据划分成两部分。

  通过增加一个约束条件：支持向量到超平面的距离是最大的。

- 非线性分类：通过拉格朗日乘子法和KKT条件，结合核函数去对数据进行分类。



SVM的种类：

- 硬间隔 SVM：又称线性可分SVM
- 软间隔 SVM：部分数据点在错误的位置，允许有错误，且错误在可接受的范围。即**软间隔允许部分样本点不满足约束条件**。
- Kernel SVM：训练数据线性不可分，通过核技巧和软间隔最大化去学习非线性SVM



#### 几何间隔和函数间隔

二维空间中点 (x,y) 到超平面的距离公式：$$\frac {|Ax+By+C|}{\sqrt {A^2+B^2}}$$

点到超平面的距离公式：$$\frac {|w^Tx+b|} {||w||}$$



有两点要注意的是：

1. Support Vectors 很稀疏
2. 被去期望误差界定而不是误差的概率。



SVM算法在不可分且**高维**的情况下，依然有一个学习保证，使得超平面在N维空间中至少$$1-\delta$$的概率存在。

$$R(h)\leq \hat{R}(h)+\sqrt{\frac{2(N+1)log\frac{em}{N+1}}{m}}+\sqrt {\frac {log\frac{1}{\delta}}{2m}}$$

当N远大于m时，是不可知的。





#### Confidence Margin Loss

说白了就是几何间隔和函数间隔的问题，找到一个bound去界定它



### Rademacher Complexity of Linear Hypotheses

$$\forall ||w||\leq \Lambda, ||x||\leq R, \hat{R}_S(H)\leq \sqrt {\frac{R^2\Lambda^2}{m}}$$





#### High-Dimensional Feature Space高维特征空间

已观测到泛化边界不依赖与维度，但是依赖margin，这可以在高维度特征空间中找到一个具有large-margin的超平面。如果在高维空间里面做点积操作，计算消耗会非常高，所以提出来Kernel的解决方案，在低维空间中计算好后，映射到高维空间。



### Kernel

kernel可以有效计算高位空间的内积，具有非线性边界，输入的空间的不是向量，对于复杂的特征，可以选择不同的核方法。

优点：Efficiency & Flexibility



#### 正定性PDS condition

要求核方法至少是半正定的，即所有值大于等于0



归一化Kernel，Normalized Kernel，形式与cosine similarity相似





#### 核方法的种类

- Gaussian kernels $$K(x,y)=exp(-\frac{||x-y||^2}{2\sigma^2}),\sigma \neq 0$$
- Sigmoid kernels $$K(x,y)=tanh(a(x\cdot y)+b),a,b\geq0$$





## Chapter 7 PAC Learning Framework & Rademacher Complexity and VC-Dimension

之前的内容其实很多都涉及到了PAC学习结构，目的是找到什么样的学习器（学习任务）是可以被有效学习的



#### 参数和变量定义

- c :concept c，$$X \rightarrow Y$$的映射
- H: hypothesis set
- $$\epsilon$$:错误率
- $$\delta$$:置信度
- S:样本，独立同分布抽样的
- D：样本分布
- size(c):concept c的最大运算表示成本
- poly(·,·,·)多项式
- m:样本数



#### PAC learnable

$$Pr_{S~D^m}[R(h)\leq \epsilon]\geq 1-\delta$$

如果满足此时，即算法的运行复杂度小于$$poly(\frac{1}{\epsilon},\frac{1}{\delta},size(c))$$，则称c可有效学习。



> PAC对分布无限制，而且只关注其中一个C

一致性假设：算法总是可以保证其返回的假设$$h_s$$在训练集下误差为0，我们称假设$$h_s$$一致，称这种问题复合一致性假设。



#### Hoeffding 不等式





#### 样本复杂度界限/泛化界

sample complexity bound/generalization bound都复合一致性假设。

$$R(h_s)\leq \frac{1}{m}(log|H|+log \frac{1}{\delta})$$

当不一致假设时：

$$R(h)\leq \hat{R}(h)+\sqrt{\frac {log|H|+log\frac{2}{\delta}}{2m}}$$



- Bound在large |H|的情况下松弛

- 对于无限的|H|来说是不可知的



### Empirical Rademacher Complexity

表示函数族G在样本集S上的输出与随机噪声的相关性的均值。

$$\hat{R_s}(G)=E_{\sigma}[sup \frac{1}{m}\sum_i^m \sigma_i g(z_i)],\sigma_i\in\{-1,+1\}$$

$$\sigma_i$$为Rademacher variables

### Average Rademacher Complexity

移除对特定样本集的依赖，更加平均地度量了一个函数族的复杂程度

$$R_m(G)=E_{S~D^m}[\hat{R_S}(G)]$$



#### McDiarmid不等式

### Rademacher Complexity Bound

先记着

$$E[g(z)]\leq \frac{1}{m}\sum_i^mg(z_i)+2R_m(G)+\sqrt \frac {log\frac{1}{\delta}}{2m}$$

$$E[g(z)]\leq \frac{1}{m}\sum_i^mg(z_i)+2\hat{R}_s(G)+3\sqrt \frac {log\frac{1}{\delta}}{2m}$$





#### Growth function

用于限制Empirical Rademacher Complexity，假说集合的行知，是用本容量的函数，也就是H中所有假说对S的标记的所有可能性的最大值。

- 不依赖数据的分布
- 同一个m下，如果一个假说集合A的增长函数值大于另一个假说集合的增长函数值，则说明A的拟合能力更强。

#### Massart's Lemma

将Empirical Rademacher Complexity 与Growth Function联系起来

这样将边界变大了，比原本更松弛

$$R_m(G)\leq \sqrt {\frac {2log\prod_G(m)}{m}}$$

#### VC 维

假说集合的属性

证明方法：

1. 存在d个sample可以被H打散
2. 不存在d+1个sample能被H打散

## Chapter 8  Model Selection & Algorithmic Stability

模型选择的一个选择是ERM，ERM是作用于训练集而言的。

### SRM结构风险最小化

在ERM的基础上，添加了惩罚项的约束，这时候的近似误差不能被估计。

随着参数数量k的增加，惩罚项的值会加大，而经验误差会逐渐减小。



#### Cross-validation

交叉验证

1. Holdout 验证

   最简单的验证方法，将原数据集直接分成训练集和验证集，一般是训练集70%，验证集30%。但是该验证方法与分组有关，导致结果会出现随机性。

2. 交叉验证 Cross-Validation

   对Holdout验证法的改进。

   k-fold：交叉验证的一种，首先将样本分为k个相等大小的样本子集，依次遍历这些子集。遍历时，把当前子集作为验证集，其余子集作为训练集，进行模型的训练和评估。最后把k次评估指标的**平均值**作为最终的评估指标。

   留p验证：每次都留下p个样本作为验证集（从n个元素中选择p个元素,总共$C_n^p$种可能。开销大。

   leave-one-out: 留一法，留p验证的特例，每次留下一个样本作为验证集，其余样本作为测试集。同样是遍历所有的样本，并取平均值作为最后评估指标。**样本数量大的时候，留一法的时间开销大。**



与SRM相比，Cross-Validation开销更小。



### Stability

稳定性：当在两个相似的训练集上训练A时，A返回的相应假设所造成的损失相差不应超过beta.

- admissibility
- proposition





## Chapter 9 Multi-Class Classification

用svm处理多分类问题

还有其他工具，比如决策树

#### 分类方法

1. one-vs-one
2. one-vs-all
3. error-correcting codes



## Chapter 10 Robust Deep Learning with Noisy Labels

noise指的是label错误

为什么会错误呢？是因为数据量大，难免会出现误标注。

#### CCN class-conditional noise

label污染，本应该正确的label被标注成别的label



带有noisy labels的鲁棒深度学习的三个方向：

- Data - estimating **noise transition matrix**，就是构造噪声转移矩阵，在数据上进行处理，比如在先在干净的数据集上训练，然后将训练结果在脏数据集上进行训练的时候，对训练好的结果先做个处理，降低loss。
- Training - Training on **selected samples**，就是通过神经网络，将坏label丢掉
- Regularization - designing **implicit regularization**，通过正则的手段，给学习器添加一个隐式的正则子。



#### Estimating noise transition matrix

噪声转移矩阵优点是提供了理论保证，但是缺点是很难估计large-class cases的矩阵。

- pairs n和n+1有叫概率

- sym 对称型，即对角线部分有较高的概率，其他位置比较低的概率




状态转移矩阵内的数值是概率



noise structure:

- Column-diagonal：两个相近的就很容易被标注，比如猫和沙滩，或者沙滩和狗，因为距离相近，所以可以很明显区分，但是狗和猫距离比较远，所以不能很好标注
- Tri-diagonal：就是因为相邻的都比较相似，所以这个也是比较难标注的
- Block-diagonal：因为都属于同一个群组，所以这个标注起来也比较简单



### 基准不足的问题 Deficiency of benchmarks

比如我们检测到一个样本的数据是猫的特征，但是我们不知道真实标签，将其观测时成了狗

1. 





#### Independent framework：对不可知的噪声数据不适用

step：

1. 从$$Y\rightarrow \tilde{Y}$$ 估计出一个T矩阵
2. 然后让这个T矩阵插入到预估的值里面做一个loss

#### Unified framework: 暴力估计容易造成局部最小值

T是不固定的，训练的时候将其加入，一起训练估计



#### Solution

> T矩阵可能非常稀疏

将人类的先验知识输入到一个explicit variable s中，即将不需要估计的地方根据我们输入的先验知识，去约束。





### 如何生成结构化模型

$$x\rightarrow y \rightarrow s \rightarrow \tilde {y}$$

- 如上面的式子，$$y$$是latent的，是从categorical中采样得到的值
- s是离散结构的矩阵，通过神经网络来做参数化
- 而$$\tilde{y}$$就是从$$y \& s$$中得到的迁移矩阵（隐函数）

从未知到已知的过程



### ELBO of Masking

通过人类的知识去将模型结构bound住 （手工处理）



### Training on selected samples

思想是将损失小的sample or instances当做是正确的样本

将大loss的多训练一下，小loss的少训练一下

有点事容易实现和自有假设，但是会积累error，即overfit





#### Co-teaching



## Chapter 11 Clustering



## Chapter 12 Adversarial Training 对抗学习

对抗训练本质上是一种数据增强的方法，即使用对抗样本来训练鲁棒的深度神经网络。该方法的求解可以被归纳为min-max的过程，即InnerMaximization和Outer Minimization两个步骤。Inner maximization用于通过最大化分类损失函数来生成对抗的样本，该过程是一个受限的优化问题，一阶方法PGD（ProjectedGradient Descent）通常会给出好的求解方法。Outer Minimization用于使用Inner Maximization阶段生成的对抗样本来训练一个鲁棒的模型，而且受InnerMaximization阶段的影响非常大。