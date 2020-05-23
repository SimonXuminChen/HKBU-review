# COMP 7210 Intelligence decision support systems

## Regression Analysis

回归分析就是一个用来描述连续型因变量和一个或多个自变量之间的关系的function。

$$Y = f(X_1,X_2,...,X_n)+\xi$$



而关于总体的关系的模型表达如下：

$$Y_i = \beta_0 + \beta_1X_{1_i}+\xi$$_i

但是对我们已有的样本，我们表示如下：

$$\hat {Y_i}=b_0+b_1X_{1_i}$$

这是我们通过已有数据对Y的近似值。

那么如何去决定这个$$b_0,b_1$$呢？我们通过**最小化ESS(Error Sum of Squares)**，即

$$ESS = \sum_{i=1}^n (Y_i-\hat {Y_i})^2 = \sum_{i=1}^n (Y_i-(b_0+b_1X_{1_i}))^2$$

如果ESS为0，说明我们的function对数据fit得很完美！



### $$R^2$$ 统计

$$R^2$$ 统计描述了回归方程对数据的拟合程度，其中$$0\leq R^2\leq1$$;它衡量了Y的方差在均值附近的比例

$$  \sum_{i=1}^n (Y_i-\bar {Y})^2 =\sum_{i=1}^n (Y_i-\hat {Y_i})^2 +\sum_{i=1}^n (\hat{Y_i}-\bar {Y})^2 $$

即$$TSS = ESS+RSS$$, 因此，$$R^2 = \frac {RSS}{TSS} = 1-\frac{ESS}{TSS}$$

**标准差**可以用来衡量实际的点在回归预测线附近的离散程度

$$S_e = \sqrt{\frac{\sum_{i=1}^n(Y_i-\hat{Y_i})^2}{n-k-1}}$$, **k是自变量的个数**

这里的知识点和统计学的很相似，经验法则同样适用，95%的预测区间如下表示：

$$\hat{Y}_h\pm 2S_e$$   where: $$\hat{Y}_h = b_0+b_1X_{1_h}$$

![image-20200521104801670](/Users/Simonchan/Library/Application Support/typora-user-images/image-20200521104801670.png)

### 额外区间 Exact Prediction Interval

其他的预测范围即$$(1-a)%$$, 

$$\hat {Y_h} \pm t_(1-a/2,n-2)S_p$$

where: 

$$\hat{Y_h} = b_0+b_1X_{1_h} \space \space S_p = S_e \sqrt {1+\frac{1}{n}+\frac{(X_{1_h}-\bar{X})^2}{\sum_{i=1}^n(X_{1_i}-\bar{X})^2}}$$ 



### 置信区间 Confidence Intervals for the Mean

$$\hat {Y_h} \pm t_(1-a/2,n-2)S_a$$

where: 

$$\hat{Y_h} = b_0+b_1X_{1_h} \space \space S_a = S_e \sqrt {\frac{1}{n}+\frac{(X_{1_h}-\bar{X})^2}{\sum_{i=1}^n(X_{1_i}-\bar{X})^2}}$$ 



对因变量（使用估计的回归函数）所做的预测对于与样本中不同的自变量值可能几乎没有或没有有效性。





### Multiple Regression Analysis

多变量回归分析，包含多个自变量，方程如下:

$$\hat {Y_i}=b_0+b_1X_{1_i}+b_2X_{2_i}+...+b_kX_{k_i}$$

同样，对参数的优化选择依然能够采用ESS，最终的结果是生成一个超平面去fit我们的样本数据。



### 模型选择

我们想确定一个最简单的模型，该模型足以说明Y变量的变化。 

- 如果随意使用所有的自变量，且这些自变量不能表示population的时候，会导致对样本数据的overfitting
- 因此要避免拟合样本特征，即对sample很靠近或overfitting的数据



### Adjusted $$R^2 $$

在多变量的情况下，用$$R^2$$去衡量的话，会发现新增了变量的情况下，$$R^2$$的值只会上升，因此我们引入$$Adjusted R^2$$，这个值是可以上升或下降的

$$R_a^2 = 1- (\frac{ESS}{TSS})(\frac{n-1}{n-k-1})$$，



- 可以通过向模型添加任何自变量来人为地增大R2统计量。 
- 我们可以比较调整后的R2值作为启发，以检查添加其他自变量是否真的有帮助。



**多重共线性**：两个变量相关，这个不应该在预测模型中出现。



非数字型的因素可以用二进制变量表示



### 多项式回归Polynomial Regression

因为有时候因变量和自变量之间的关系不是线性的





## Discriminant Analysis

DA 是一个统计的工具，用来用已有的自变量去预测一个离散的（有类别的）因变量。

目标是希望可以找到一条规则，可以根据观测到的自变量的值去判断该观测数据属于两个或多个提前定义好的group中的哪一个



DA解决的问题：

- 2 Group Problems
- k-Group Problems

> 2-Group问题可以采用回归分析，k-Group问题则不行

### Discriminant Scores

$$\hat {Y_i} = b_0+b_1X_{1_i}+b_2X_{2_i}$$

其中，$$X_1，X_2$$分别表示其中一个特征的值

#### Step

1. 我们根据在数据中选定的值去计算当前观测数据的Discriminant Score
2. 然后将每个group中的数据求出每个group平均的Discriminant Score
3. 接着再对每个组的Discriminant Score求平均值，即可得到Cutoff Value
4. 最后再用cutoff value和各个观测数据的cutoff value比较，从而分组

$$Cutoff Value = \frac{\hat{\bar{Y_1}}+\hat{\bar{Y_2}}}{2}$$





#### A Refined Cutoff Value

- 错分组的数量可能不一样
- 每个group的成员概率可能也不一样

因此我们需要提炼一下原本的cutoff value

$$Cutoff Value = \frac{\hat{\bar{Y_1}}+\hat{\bar{Y_2}}}{2}+\frac{S_p^2}{\hat{\bar{Y_1}}-\hat{\bar{Y_2}}}Ln(\frac{P_2C(1|2)}{P_1C(2|1)})$$

其中，$$S_p^2 = \frac {(n_1-1)S_{\hat{Y_1}}^2+(n_2-1)S_{\hat{Y_2}}^2}{n_1+n_2-2}$$



之后可以建立一个像confusion matrix的矩阵，来计算accuracy rate



### The k-Group DA Problems

也成为multiple discriminant anaylysis ,or MDA

计算与2-group的相似，不过区分开了多个cutoff-value



#### The Classification Rule

- 通过计算每个点到group中圆心的距离，然后分配给它最近的点

距离计算采用两个点之间的Euclidean Distance

$$Distance = \sqrt {(A_1-A_2)^2+(B_1-B_2)^2}$$

欧几里得距离并不代表方差的不同



Variance-Adjusted Distance:

$$D_ij = \sqrt {\sum_k \frac {(X_{ik}-\bar {X_{jk}})^2} {S_{jk}^2} }$$

还有其他距离可以去测量变量之间的距离





### Classification

定义：给定一堆数据，也就是training set， 然后根据区分训练集去找到一个模型，从而使对没有见过的数据也能精确的区分。



![image-20200520110143661](/Users/Simonchan/Library/Application Support/typora-user-images/image-20200520110143661.png)

![image-20200520110205457](/Users/Simonchan/Library/Application Support/typora-user-images/image-20200520110205457.png)

- Sensitivity 就是Recall值

- Specificity关注negative

### Case-Based Reasoning (CBR)

通过选择之前相似的案例和经验，然后将这些案例的解决方案用到当前问题中。（imitation）



#### Assumption

- 相似的问题拥有相似的解法
- The world is a regular replace: what holds true today will probably hold true tomorrow.
- Situations repeat: The world is repetitive place, similar problems tend to recur.



当之前的案例都能成功解决的时候，而且相似的problem出现了，当某些领域的模型很难解决这个问题，或当系统需要constant maintenance的时候，可以采用CBR



#### Process-$$R^4$$ cycle

- Retrieval：将新进来的数据生成一个target case，然后去寻找最相似的problems
- Reuse：从Retrieval中找到最相似的solution
- Revise：在CBR中修改参数
- Retain：保存

![image-20200520111648122](/Users/Simonchan/Library/Application Support/typora-user-images/image-20200520111648122.png)

#### Case 的表达形式

case = <problem,solution>





关于Similarity的衡量

不同的特征有不同的重要性刻度，通过赋予权重来添加



#### Reuse(adaptation)

适应会改变初始解决方案：考虑新问题和检索到的问题之间的差异。
常见的适应形式：

- Null-Adaption：复制检索到的解决方案
- Substitution:对匹配到的数据的某些部分进行修改
- Transformation:修改solution的结构



#### Revise

根据条件去修改solution





### CBR knowledge container

- Case vocabulary (used features) ：描述案件的特征，包括indexed features & unindexed features

- Case base 
  - Prototype cases(初始化时存在)
  - episodic cases（后面迭代更新存在）

- Similarity assessment 

- Solution adaptation



#### Decision tree





#### Steps for building a case-based reasoning system

1. Collect data for cases. 
2. Design case vocabulary based on data. 
3. Determine the case index structure (especially for *large* case base). 

4. Decide the *similarity/distance* for case retrieval. 
5. 5. Decide whether a case adaptation procedure is appropriate (and, if so, implement it). 

6. Develop the rest of the system (e.g., a user interface is desired but not required).



### Multi-agent task allocation

已知分布式计算场景中某个agent对每个task的处理能力，那么怎么将一个新的problem分配给处理最好的那个agent呢？**Select the agent with the highest suit_score!**





## Time Series Analysis

时间序列是对随时间收集的变量的一组观察结果。

Data Type:

- **Stationary Data** - a time series variable exhibiting no significant upward or downward trend over time. 固定不变

- **Nonstationary Data** - a time series variable exhibiting a significant upward or downward trend over time. 随时间变化

- **Seasonal Data** - a time series variable exhibiting a repeating patterns at regular intervals over time. 随时间周期性变化





测量时间序列技术的方法：

- MAPE
- MAE
- MSE
- RMSE



### Extrapolation Models 

Extrapolation models try to account for the past behavior of a time series variable in an effort to predict the future behavior of the variable.

试图考虑时间序列变量的过去行为，以预测变量的未来行为。











### Stationary Data的Extrapolation Modes

#### Moving Averages

$$\hat {Y_{t+1}} = \frac{Y_t+Y_{t-1}+Y_{t-k+1}}{k}$$

相当于求均值，这样得到的结果并不准确，尤其是当比较久远的数据比较好，而较新的数据比较差的时候，会导致结果并不准确，因此一般都采用最近的数据。

而且，这个k值是不确定的，需要迭代寻找。



#### Weighted Moving Averages

$$\hat {Y_{t+1}} = w_1Y_t+w_2Y_{t-1}+...+w_kY_{t-k+1}$$

分配不同的权重，但是这些权重的和为1。



#### Exponential Smoothing

$$\hat {Y_{t+1}} = \hat {Y_{t}}+a(Y_t-\hat {Y_{t}})= $$

但是假如$$Y_t$$没有数据的话，也就是说当前的$$Y_t$$也是预测来的话，那么接下来的值都等于之前那个有值的的预测结果

**适合用来预测没有明显趋势和季节性的时间序列**

#### Stationary Data WIth Seasonal Effects

- Additive Seasonal Effects
- Multiplicative Seasonal Effects



$$E_t = \alpha(Y_t-S_{t-p})+(1-\alpha)E_{t-1}$$

$$S_t = \beta(Y_t-E_t)+(1-\beta)S_{t-p}$$

其中$$E_t$$当前时刻t的时候的期望等级，而$$S_t$$表示当前时刻的seasonal factor

**Stationary Data With Additive Seasonal Effects**:

$$\hat{Y+n} = E_t+S_{t+n-p}$$



**Stationary Data With Multiplicative Seasonal Effects**:

$$\hat{Y+n} = E_t\times S_{t+n-p}$$



### Nonstationary Data的Extrapolation Modes

#### Double Moving Average



$$\hat {Y}_{t+n} = E_t + nT_t$$

预测未来的n个periods, n表示第几个

$$M_t=\frac {(Y_t+Y_{t-1})+...+Y_{t-k+1}}{k}$$

$$D_t=\frac {(M_t+M_{t-1})+...+M_{t-k+1}}{k}$$

$$E_t=2M_t-D_t$$

$$T_t=\frac{2(M_t-D_t)}{k-1}$$





其中$$E_t$$是在t时刻期望的base level，而$$T_t$$是在时刻t的期望趋势。

#### Double Exponential Smoothing(Holt's Method)

$$\hat{Y}_{t+n} = E_t+nT_t$$

where

$$E_t = \alpha Y_t+(1-\alpha)(E_{t-1}+T_{t-1})$$

$$T_t=\beta(E_t-E_{t-1})+(1-\beta)T_{t-1}$$



#### Holt-Winter's Method 

- Additive Seasonal Effects
- Multiplicative Seasonal Effects



**Holt-Winter’s Method For Additive Seasonal Effects**

$$\hat{Y}_{t+n} = E_t+nT_t+S_{t+n-p}$$

where

$$E_t = \alpha (Y_t-S_{t-p})+(1-\alpha)(E_{t-1}+T_{t-1})$$

$$T_t=\beta(E_t-E_{t-1})+(1-\beta)T_{t-1}$$

$$S_t = \gamma(Y_t-E_t)+(1-\gamma)S_{t-p}$$



**Holt-Winter’s Method For Multiplicative Seasonal Effects**

$$\hat{Y}_{t+n} =( E_t+nT_t)S_{t+n-p}$$

where

$$E_t = \alpha (Y_t/S_{t-p})+(1-\alpha)(E_{t-1}+T_{t-1})$$

$$T_t=\beta(E_t-E_{t-1})+(1-\beta)T_{t-1}$$

$$S_t = \gamma(Y_t/E_t)+(1-\gamma)S_{t-p}$$





#### Modeling trends using Regression: The Linear Trend Model

$$\hat{Y}_t = b_0+b_1X_{1_t}$$, where $$X_{1_t}=t$$

将time period 作为观测变量输入

Quadratic trend model: $$\hat{Y}_t = b_0+b_1X_{1_t}+b_2X_{2_t}$$, where $$X_{1_t}=t,X_{2_t}=t^2$$

#### Computing Multiplicative Seasonal Indices

我们可以计算出周期p的乘性季节性调整指数：

$$\S_p = \frac {\sum_i \frac{Y_i}{\hat{Y}_i}}{n_p}$$, for all i occuring in season p

因此，最终的预测结果:

$$\hat{Y}_i \space adjusted = \hat {Y}_i \times S_p$$

**summary：**

1. 创建一个趋势模型并计算样本中每个观测值的估计值。

2. 对于每个观测值，计算实际值与预测趋势值的比率，对于累加效应，计算差异。

3. 对于每个季节(周期)，计算在步骤2中计算的比率的平均值。这些是季节性的索引。

4. 将趋势模型产生的任何预测值乘以步骤3中计算的适当季节指数。（对于附加的季节性影响，请在预测中添加适当的因子。）

   

### Seasonal Data的Extrapolation Modes

#### Seasonal Regression Models

指标变量也可用于回归模型中以表示季节性影响。

- 如果有p个季节，则需要p -1个指标变量。





## Optimization and Linear Programming

![image-20200520150223539](/Users/Simonchan/Library/Application Support/typora-user-images/image-20200520150223539.png)

决定如何最好地利用对个人或企业可用的有限资源是一个普遍的问题

- 在当今竞争激烈的商业环境中，确保以最有效的方式使用公司有限的资源变得越来越重要
- 这涉及确定如何以最大化利润或最小化成本的方式分配资源
- 数学编程（MP）是管理科学领域，它找到了使用有限资源实现个人或企业目标的最佳或最有效方法
- MP通常被称为优化



**Linear Programming** can be used to solve the problems with linear objective function and linear constraints



### Summary of Graphical Solution to LP Problems

1. Plot the boundary line of each constraint

2. Identify the feasible region

3. Locate the optimal solution by either:

   a. Plotting level curves

   b. Enumerating the extreme points





#### 可能发生在LP问题中的异常现象：

- Alternate Optimal Solutions 

- Redundant Constraints

- Unbounded Solutions

- Infeasibility



LP目标函数是线性的； 这导致以下两个假设：

- 比例性(**proportionality**)：每个决策变量对目标函数的贡献与决策变量的值成比例。 
- 可加性(**additivity**)：任何决策变量对目标函数的贡献均独立于其他决策变量的值。

此外，还有：

- 可分性假设：每个决策变量均允许采用小数值

- 确定性假设：确定性地知道每个参数（目标函数系数cj，每个约束的右侧常数bi和技术系数aij）

  

## Network Modeling

单个问题可以采用Linear Programing去表示，如果是多个问题的话，可以用网络的图形方式去表达。



重点：

-  balance-of-flow rules
- 根据问题描述去画network
- Define the network flow problem with LP models.



解决的问题（network 变种）:

- Transshipment problems：传统网络问题
- shortest path problems：起始点-1,结束点为0，其他点为
- the equipment replacement problem：每个结点之间数据存在递减关系
- generalized network flow problems：输入输出问题，有中间结点
- maximal flow problems：最大问题，需要从终点连接一条线到起始点，目的是最大化，这条线的值
- special model considerations:这类问题一般新增一个或多个节点去解决
  - flow aggregation
  - multiple arcs between nodes
  - capacity restrictions on total supply- 边上会限制大小，使得即便是在supply大于demand的情况下，也不能达到要求，这时候可以新增一个虚拟节点，将supply都指向它。
- the minimum spanning tree problem：先从第一个点开始找，然后找最近的点，以此类推，如果当前点之间的距离不是最短的，则换一个点继续找



In generalized network flow problems, the gains and/or losses associated with flows across each arc effectively increase and/or decrease the supply available in the network.

![image-20200520170744190](/Users/Simonchan/Library/Application Support/typora-user-images/image-20200520170744190.png)

- 当问题尚不清楚时，最安全的做法是首先假定所有需求都可以满足，并且（根据流量平衡规则）使用以下形式的约束：inflow-outflow 大于等于 Supply or Demand。
- 如果导致的问题是不可行的（并且模型中没有错误！），那么我们知道不能满足所有需求，并且（根据流量平衡规则）使用以下形式的约束：inflow-outflow 小于等于 Supply or Demand。

#### 





最大流量问题（或最大流量问题）是一种网络流量问题，其目标是确定网络中可能发生的最大流量。



![image-20200520180316978](/Users/Simonchan/Library/Application Support/typora-user-images/image-20200520180316978.png)

## 





## Integer Linear Programming

当一个或多个变量在NL中要求是整型的时候，这个问题就变成了Integer Lineara Programming 问题



整型条件约束比较好表达，但是会是问题更难解决，甚至有些时候是无解的。



### relaxation

通过假设所有变量都是continuous variable去定义一个标准的LP问题，却对变量进行求解。

ILP问题的LP relaxation的可行区域始终包含原始ILP问题的所有可行整数解。



#### Bounds

ILP问题最优解的目标函数值永远不会比LP松弛最优解的目标函数值更好

在最大化问题中， NL relaxation 提供的是一个upper bound

而在最小化问题中，NL relaxation 提供的是一个lower bound



#### Rounding

四舍五入，但是结果并不可靠，如果用了rounding,会出现以下两种情况：

1. 可能导致不可行的解决方案
2. 就算该值求出来是整型，但是也不能保证这个值就是最优解。



#### Branch-and-Bound (B&B)

需要花费大量时间和资源去计算。

通过查找边界，将大问题切分成小问题



#### Stopping Rules

因为B&B需要大量时间，所以可以降低准确度，采用一个可以接受的，第二优秀容差因子(suboptimality tolerance factor)

- 一旦发现整数解在全局最优解的某个百分比以内，就可以停止计算

- 从LP松弛获得的边界在这里很有帮助



#### Binary Variables

- CRT 投资问题



固定/一次性支出问题

通过定义一个二进制变量和当前变量相乘，决定是否选择





## Goal Programming and Multiple Objective Optimization

### Goal Programming

要解决的问题不再仅仅是一个值，而是一系列的目标。

与LP或ILP不同，LP和ILP都具有一个hard constraints，强制要求结果的范围，而GP则采用一个Soft constraints，针对不同的目标和群体，适应不同的范围。

可以使用目标而不是硬性约束来更准确地建模许多管理决策问题。

此类问题通常没有一个明确的目标功能要求在约束集上最大化或最小化，而是可以表示为也包含硬约束的目标集合。

尽管为GP问题制定约束条件相当容易，但是确定适当的目标函数可能会非常棘手；

GP问题的目标是确定一种解决方案，以尽可能接近地实现所有目标。 解决任何GP问题的理想解决方案是，在其目标值指定的水平上准确实现每个目标。 通常，不可能实现理想的解决方案，因为某些目标可能与其他目标冲突。

![image-20200520213039336](/Users/Simonchan/Library/Application Support/typora-user-images/image-20200520213039336.png)



![image-20200520213714385](/Users/Simonchan/Library/Application Support/typora-user-images/image-20200520213714385.png)



1.确定问题中的决策变量。
2.找出问题中的任何硬性约束，并以通常的方式制定它们。
3.陈述问题的目标及其目标值。
4.使用可以精确实现目标的决策变量创建约束。
5.通过包含偏差变量，将上述约束转化为目标约束。
6.确定哪些偏差变量表示与目标之间的不良偏差。
7.制定一个目标，对不希望的偏差进行惩罚。
8.确定目标的适当权重。
9.解决问题。
10.检查问题的解决方案。 如果解决方案不可接受，请返回步骤8并根据需要修改权重。



### Multiple Objective Optimization

或者说是Multiple Objective Linear Programming

是一个具有多个目标函数的LP问题，同时也能被看作是一个我们必须去为每个goal决定目标值特殊的GP问题。

> 采用MiniMax去客观描述

使用MiniMax目标获得的解决方案是Pareto 最优。

- 偏差变量和MiniMax目标在不涉及MOLP或GP的各种情况下也很有用。
- 对于最小化目标，百分比偏差为：（实际-目标）/目标
- 对于最大化目标，百分比偏差为：（目标-实际）/目标
- 如果目标值为零，请使用加权偏差而不是加权％偏差。

#### Step

1. 确定问题中的决策变量。
2.  确定问题的目标并照常制定。
3. 确定问题中的约束条件并照常制定。
4. 为步骤2中确定的每个目标解决一次问题，以确定每个目标的最佳值。
5. 使用在步骤4中确定的最佳目标值作为目标值，将目标重述为目标。
6. 对于每个目标，创建一个偏差函数，以测量任何给定解决方案未能达到目标的数量（以绝对值或百分比形式）。
7. 对于在步骤6中标识的每个功能，为该功能分配权重并创建一个约束，该约束要求加权偏差函数的值小于MINIMAX变量Q。
8. 解决结果问题，以最小化Q为目标。
9. 检查问题的解决方案。 如果解决方案不可接受，请在步骤7中调整权重并返回到步骤8。



## Nonlinear Programming

NLP问题具有非线性目标函数和/或一个或多个非线性约束。

- NLP问题的制定和实施几乎与线性问题相同。
- 解决NLP问题所涉及的数学与LP问题完全不同。
- 解算器倾向于掩盖这种差异，但重要的是要了解解决NLP时可能遇到的困难。



我们用graph去找最优解的时候，有可能会在local optimal solution处卡住。

在可能的情况下，最好使用与预期最佳值大致相同大小的起始值

#### Location problems

采用距离选择最优值