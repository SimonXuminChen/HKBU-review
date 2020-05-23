

## Data

### what is data?

data is a collection of data objects and their attributes. An attribute is a property or characteristic of an object and a collections of attributes describe an object.



### Types of data sets

- record
- graph
- ordered

### types of attributes and properties

| type     | properties                                  |
| -------- | ------------------------------------------- |
| nominal  | distinctness                                |
| ordinal  | distinctness, order                         |
| interval | distinctness, order, meaningful differences |
| ratio    | above all                                   |



### differences between Discrete and Continuous Attributes

| Discrete                                                  | Continuous                                                   |
| --------------------------------------------------------- | ------------------------------------------------------------ |
| only finite or countably infinite set of values           | real numbers                                                 |
| often represented as integer variables(binary attributes) | can only be measured and represented using a finite number of digits |
|                                                           | typically represented as floating-point variables            |



### Outliers

outliers are data objects with characteristics that are considerably different than most of the other data objects in the data.



### Missing values

#### reason

because information is not collected, or attributes may not be applicable to all cases, or sometimes may caused by human mistakes.



#### handling missing values

- eliminate data objects or variables
- estimate missing values
- ignore the missing value during analysis



- MCAR
- MAR
- MNAR



### Duplicate Data

#### when

Major issue when merging data from heterogeneous sources

#### How to do

process of dealing with duplicate data issues



### 5 num summary

- min
- Q1
- median
- Q3
- max



### Aggregation

生成数据

combine two or more attributes into a single attribute

### Sampling

#### reason

too expensive or time consuming to obtain the entire data of interest, so we want to find a representative sample.



#### type

- simple random sampling
  - sampling without replacement
  - sampling with replacement
- stratified sampling

### Dimensionality Reduction

#### Reason

- to avoid curse of dimensionality
- reduce the time and requirement of data mining alogrithms
- make data easily to visualize
- may help to eliminate irrelevant fetures or reduce noise

#### Techniques

- PCA
- SVD

### Feature subset selection

also is a method to reduce dimensionality of data

we can apply feature subset selection to drop redundants features and ittrlevant features

​	

### feature creation

to create new attributes that can capture the important information in a data set much more efficiently than the original attributes. 

#### methodologies:

- feature extraction
- feature construction
- mapping data to new space(kernel method)



### discretization and binarization

- discretization is the process of converting a continuous attribute into an ordinal attribute
- binarization maps a continuous or categorical attribute into one or more binary variables

### attribute transformation

is a function to map the entire set of valuse of a given attribute to a new range.

#### method:

- normalization
- standardizaiton



## Frequent Itemsets

### what is frequent itemsets?

given a support threshold $$s$$, then sets of items that appear in at least s baskets are called frequent itemsets.



**support:** for itemset I , the number of baskets containing all items in I.

#### Rules

**confidence:**is the probability of $$j$$ given $$I = {i_1,...,i_k}$$，it used to find the significant/interesting ones in the association rules.

$$conf(I\rightarrow j) = \frac {support(I \cup j)}{support(I)}$$

> but not all high-confidence rules are interesting, because sometimes the item j is very often and independent of i



**Interest** of an association rule $$I \rightarrow j$$:difference beetween rules' confidence and the fraction of baskets that contain j

$$Interest(I \rightarrow j)=conf(I \rightarrow j)-Pr[j]$$



#### Rule generation

**Output**: the rules above the confidence threshold

- variant 1: Single pass to compute the rule confidence
- variant 2: Observation, If $$A,B,C \rightarrow D$$ is below confidence, so is $$A,B \rightarrow C,D$$
- also can generate rules with bigger value of confidence from smaller ones. 



#### Maximal frequent itemset 

没有直接超集是frequent的

can gives more pruning



#### Closed frequent itemset

没有直接超集和当前itemset具有相同的frequency

stores not noly frequent information, but exact counts



#### comparison between maximal and closed itemsets:

both of them are to reduce the number of rules, we can post-process them and only store output.

no easy to understand

but maximal frequent itemsets can provide more pruning

closed itemsets not noly store the frequent information, but exact counts.



#### Attention 注意事项

在寻找所有的frequent itemsets中，我们不需要用到confidence、lift和Interest，只需要用到support即可。



given d items, there are $$2^d$$ possible candidate itemsets.



### Naive alogrithm

two approaches:

- triangular matrix, cost 4 bytes per pair
- triples, cost 12 bytes per occurring pair

#### comparison between the two approaches

triangular matrix only count pair of items {i,j} if i<j, and it keep pair counts in lexicographic order. Pair {i,j} is at position (i-1)(n-i/2)+j-1, the total number of pairs $$\frac {n(n-1)}{2}$$, and total bytes is $$2n^2$$

and triples only stores the pairs which count > 0, and it beats approach 1 if less than 1/3 of possible pairs actually occur.



### Apriori

first of all, concentrate on pairs, then extend to larger sets.

- if an itemset is frequent, then all of its subsets must also be frequent



#### support counting 

- hash function
- max leaf size

to reduce number of comparisons, store the candidate itemsets in a hash structure



#### 影响Apriori 复杂度的因素

- choice of minimum support threshold 
  - 较低的threshold会导致更多的frequent itemsets产生，这会导致candidatas数和frequent itemsets的长度增加
- dimensionality(number of items) of the dataset
  - 需要更多的内存空间去存储每个item的support count，这样在scan的时候，需要更多的computation 和I/O costs
- size of database
  - transactions 的次数多了
- average transaction width
  - may increase  max length of frequent itemsets and traversals of hash tree.



### FP-Growth



### Comparsion

**Advantages of Apriori:**

- easy to get frequent sets

**Disadvantages of Apriori:**

- breadth-first search
- candidate generation and test, which means that it might generate a huge number of candidates, and need to scan the transaction many times.



**Advantages of FP-Growth:**

- just scan the database once to store all essential information in a data structure called FP-tree
- divide-and-conquer
  - depth-first search

- Completeness
  - preserve complete information for frequent pattern mining
  - never break a long pattern of any transaction, it grow long patterns from short ones using local frequent items only
- Compactness
  - reducue irrelevant info, and avoid explicit candidate generation
  - items in frequency descending order: the more frequently occurring, the more likely to be shared
  - never be larger than the original database.

**Disadvantages of FP-Growth:**

- if there are too many items, it takes time to calculate and find the frequent itemsets.



## LSH

goal: find the similar items



### Shingling

将documents 向量化

k-shingle 又称为k-gram,

即将document拆分成每个大小为k的表示方法

当k非常大的时候，每一个single都很长，不利于存储，可以hash他们

<img src="/Users/Simonchan/Library/Application Support/typora-user-images/image-20200518165052496.png" alt="image-20200518165052496" style="zoom:50%;" />

#### similarity metric for shingles

- 对每个文件用一个01向量来表示,但是这样的表示方法会导致稀疏
- 可以通过jaccard similarity 来计算文件间的相似度



#### k值的选取

- k=5, 短文件
- k=10, 长文件

### Min-Hashing

每个文件向量两两对比计算量太大，而且浪费时间，所以引入min-hashing解决

通过向量表示两个文本相似度，两个文件中都有的term用1表示，没有的话用0表示。

然后将所有向量整合在一起后，以举证形式表示，其中每一行表示一个term(shingle), 每一列表示一个document



#### signatures

用签名相似度代表文件相似度，从而压缩文件大小

![image-20200518171315802](/Users/Simonchan/Library/Application Support/typora-user-images/image-20200518171315802.png)



具体实现：

- 用$$\pi$$序列获取signature matrix

<img src="/Users/Simonchan/Library/Application Support/typora-user-images/image-20200518171510725.png" alt="image-20200518171510725" style="zoom:50%;" />

这个signature matrix 就是通过每一列的$$permutation\space \pi$$去找到每一个document中第一位不为零的在序列中的编号



- 通过signature matrix去计算similarity

  <img src="/Users/Simonchan/Library/Application Support/typora-user-images/image-20200518172434609.png" alt="image-20200518172434609" style="zoom:33%;" />

可以通过100byte的大小来保存一个permutation过后的文件，每一个permutation对应的最大值是255,即2的8次方,进行100次permutation

#### permutation's implementation trick

用Universal hashing选择随机值代表位置，去搜索一个文件的这行是否有1，若有1就用这个值作为这次min hashing的结果



### Locality-Sensitive Hashing

Goal: find documents with Jaccard similarity at least s



#### Step

1. 首先把每一个文件在signature matrix中分成r*b的格子,**b=band个数，r=每个band的行数**

   <img src="/Users/Simonchan/Library/Application Support/typora-user-images/image-20200518173029305.png" alt="image-20200518173029305" style="zoom:30%;" />

2. 再通过某一种哈希方式把每个文件的同一个band丢进buckets中，依次对每一个band进行该操作

   <img src="/Users/Simonchan/Library/Application Support/typora-user-images/image-20200518173128126.png" alt="image-20200518173128126" style="zoom:33%;" />

3. 只要两个文件的任意一个band在同一个bucket中，这两个文件就成为candidate pair,即阀值超过了s的文件对，反之就不是candidate pair

#### 原理

**simplifying assumption:**有足够多的buckets，以至于只有在完全相同的band才能够掉到同一个bucket

对于不同的b，r和两个document的similarity值t，至少有一个band掉在同一个bucket的概率

<img src="/Users/Simonchan/Library/Application Support/typora-user-images/image-20200518173416451.png" alt="image-20200518173416451" style="zoom:33%;" />

#### examples

- 两个文件的similarity的t=0.8，阀值s=0.8，b=20，r=5。将会有0.99965的概率这个文件将会被正确地预测，即至少一个band会掉入同一个bucket
- 两个文件的similarity的t=0.3，阀值s=0.8，b=20，r=5。将会有0.0474的概率这个文件将会被错误地预测为相似文件对，即至少一个band会掉入同一个bucket

### summary

- b越小，r越大，两个文件至少有一个band掉进同一个bucket的概率就小，false positive的概率就变小，false negative的概率就变大

- 理想状态的LSH,即超过阀值即一定预测为candidate pair



## Clustering

to find groups of objects such that the objects in a group will be similar(or related) to one another and different from(or unrelated to) the objects in other groups.

### Types of clusters:

- well-separated 
- center-based
- contiguity-based(nearest neighbor or transitive)
- density-based
- conceptual(有overlapping的)
- objective function- 由目标函数决定



### K-means

> 将所有点分配好后，再更新中心点

#### limitation

k-means has problems when clusters are of differing:

- sizes
- densities
- non-globular shpaes

it also cannot figure out the problem when data contains outliers

k-means 的初始点选择很重要，一个好的初始点可以更快收敛，且产生更好的cluster；如果初始点选择不好，会使clustering的时间和计算量加大。

#### 选择初始点

如果有k个clusters，那么从其他cluster中选择到一个中心的机会是很小的，尤其是当k很大的时候。假设cluster都具有相同的size=n:

$$P =\frac{K!n^K}{(Kn)^K} = \frac{K!}{K^K}$$



#### Solutions to initial centroids problem

- multiple runs
- sample and use hierarchical clustering to determine initial centroids
- select more than k initial centroids and then select among these initial centroids
- postprocessing
- generate a larger number of clusters and then perform a hierarchical clustering
- bisecting k-means



#### Solutions to Empty Clusters problem

- choose the point that contributes most to SSE
- choose a point from the cluster with the highest SSE
- if there are several empty clusters, the aboce can be repeated several times.



#### Updating Centers Incrementally 

每添加一个点，就更新一次中心



#### Pre-processing & Post-processing

- pre-processing
  - normalization
  - eliminate outliers
- post-preprocessing
  - eliminate small clusters that may represent outliers
  - split 'loose' clusters(拥有较高SSE的)
  - merge clusters(较低SSE和比较接近的)
  - ISODATA



K-means对outliers敏感



**优点：**

1. easy to implement with simple principle 
2.  If the k value and the initial center value are appropriate, the clustering results are better

**坏处：**

1. The number of K cannot be determined
2. It is sensitive to outliers
3. The algorithm complexity is not easy to control, and the number of iterations may be more
4. Local optimal solution rather than global optimal solution (this is related to the initial selection)
5. Unstable results (influenced by input order)
6. Cannot be incrementally calculated
7. Sometimes may lead to empty clusters
8. the result depends on K and initials



### K-medoids（PAM, partition around medoids）

和k-means相似，不过不计算均值，而是采用交换the most centrally located points作为新的中心.

<img src="/Users/Simonchan/Library/Application Support/typora-user-images/image-20200518192735503.png" alt="image-20200518192735503" style="zoom:33%;" />



需要计算每个点之间的距离，然后选择最终clustering后，sse最小的

### Hierachical clustering

层次聚类，两种类型：Agglomerative和Divisive



**优点：**

1. it does not need to make an assumption about the number of clusters

2. They may correspond to meaningful taxonomies

**缺点**：

1. time and space requirements

2. once a decision is made to combine two clusters, it cannot be undone

3. no global objective function is directly minimized

4. different schemes have problems with one or more of the following

5. Breaking large clusters

   

#### Agglomerative

假设每个点都是一个cluster，然后计算cluster之间最小距离的两个，然后将其合并，并更新距离矩阵。



四种选择距离:

- min(single link)
- max(complete linkage)
- group average
- distance between centroids



#### MIN or Single Link

优点：可以处理 non-elliptical shapes

缺点：对noise 和outliers敏感



#### MAX or Complete LInk

优点：noise和outliers的影响少了

缺点：会把大cluster拆分，对globular cluster有偏差





### MST（Minimum Spanning Tree）

![image-20200518200220817](/Users/Simonchan/Library/Application Support/typora-user-images/image-20200518200220817.png)

### DBSCAN

基于密度的算法

优点：

1. clustering fast
2.  can effectively process noise points and find spatial clustering of arbitrary shapes
3. The shape of the cluster is not biased
4. the number of clusters does not depend on initialization requirements

坏处：

1. When the data volume increases, it requires a large amount of memory support and I/O consumption is also large
2.  When the density of spatial clustering is not uniform and the spacing difference between clustering is large, the clustering quality is poor, because the parameters MinPts and Eps are difficult to select in this case
3. Algorithm clustering effect depends on and distance formula selection. In practice, Euclidean distance is often used. For high-dimensional data, there is a "dimension disaster".





## Classification

> 以下三种算法都是选择Gain最大的。

![image-20200518211047204](/Users/Simonchan/Library/Application Support/typora-user-images/image-20200518211047204.png)

### Decision tree based classification

**advantages:**

- inexpensive to construct
- extremely fast at classifying unknown records
- easy to interpret for small-sized trees
- robust to noise, especially when methods to avoid overfitting are employed, pruning can help to reduce  the influence of the noise point
- can easily handle redundant or irrelevant attributes, unless there are some relationship between the attributes

**disadvantages:**

- space of possible decision trees is exponentially large, and greedy approaches can not find the best tree every times.
- does not take into account interactions between attributes
- each decision boundary involves only a single attribute



## Recommendation System

### Content-Based

**Pros:**

1. No need for data on other users(no cold-start or sparsity problems)

2. able to recommend to users with unique tastes

3. able to recommend new & unpopular items(no first-rater problem)

4. able to provide explanations(Can provide explanations of recommended items by 

   listing content-features that caused an item to be 

   recommended)

**Cons:**

1. finding the appropriate features is hard, we have no idea which contents are users' preference.
2. recommendations for new users, we don't know how to construct the user profile
3. overspecialization, the item which recommends to user are common





### Collaborative Filtering

Harnessing quality judgments of other users



关于相似度计算公式的问题：

1. Jaccard similarity measure

   **problems:** ignores the value of the rating

2. Cosine similarity measure

   **problems:** treats missing ratings as "negative", in another word, just consider the item which two rates  exist.

3. Pearson correlation coefficient



如果想让sim(a,b)恒大于sim(a,c),可以让rate减去row_mean，另外，用pearson去算的时候，减去的平均值是已经预测过的平均值，而非算上未预测过的数。



### Rating prediction

对于User-User 来说，看的是与当前用户最相似的并且对item i进行了评分的k个用户，求这些用户对i的平均评分

$$r_{xi}=\frac {\sum_{y\in N}s_{xy}r_{yi}}{\sum_{y\in N}s_{xy}}$$

而对于Item-Item来说，我们看的是和当前item相似的且被当前用户评分了的item，求用户对这些相似产品的平均评分

$$r_{xi}=\frac {\sum_{j\in N(i;x)}s_{ij}r_{xj}}{\sum_{j\in N(i;x)}s_{ij}}$$



**pros:**

1. Works for any kind of item, because no feautre selection needed



**cons:**

1. cold start problem, it need enough users in the system to find a match
2. sparsity, it is hard to find users that have rated the same items
3. first rater, it cannot recommend an item that has not been previously rated, such as new items
4. popularity bias, it cannot recommend items to someone with their own favourite, and it tends to recommend popular items because of high similarity.





### Hybrid methods

1. 混合法，采用两个或多个不同的推荐方法，然后去结合预测(implenment two or more different recommenders and combine predictions)
2. 将content-based methods用到cf上(add content-based methods to collaborative filtering)





### Latent Factor Model

就是采用了SVD降维。

传统的SVD是$$A = U\sum V^T$$

这里令$$P = \sum V^T$$,所以$$A=U \cdot P^T$$

> 注意，这里的svd需要排序计算

然后要预测评分，则把A矩阵中的行和列提取出来，然后再U中提取该行的值，并与$$P^T$$中的列相乘，即是我们的预测值。



SSE就是最小化原矩阵和还原矩阵的差。

但是评分矩阵中，新用户会没有预测值，所以需要一个新的方法去找到U和P



### Evaluating predictions

- 比较已知的ratings：
  - RMSE- might penalize a method that does well for high ratings and badly for others.
  - Precision aat top 10
  - Rank Correlation 

- 0/1 model

  - coverage

  - precision

  - Receiver operatiing characteristic(ROC)

    

### Problems with Error Measure

Narrow focus on accuracy sometimes misses the point such as prediction diversity, prediction context and order of predictions

事实上，我们只关注高分的。

## Big Graph Data Processing

data preprocessing 和 data analysis 的关系，数据量越来越大，分析越来越难，工具和算法等方面受到了限制，所以一个好的预处理方式可以提高数据分析的质量。



### Graph Models

**primary key:** the value of the primary key can identify an entity uniquely. 主键必须具有可识别性和唯一值。 

 Graph 𝐺(𝑉, 𝐸) 

- V is the set of vertices, E is the set of edges.

可以通过邻接矩阵去表示一个图，行表示出发点，列表示到达的点有向图和无向图的表示方法稍微有些不同，无向图是一个对称矩阵，而有向图从$$a \rightarrow b$$的时候，才表示1。

**Advantages of adjacent matrix**

- Amenable to mathematical manipulation.

- Iteration over rows and columns corresponds to computations on outlinks and inlink. 

**Disadvantages of adjacent matrix:**

- Lots of zeros for sparse matrices. Lots of wasted space.



> All the massive graphs are sparse. 



另外的图表示法： **Adjacent lists**，把邻接矩阵中所有非0的点去掉，只保留outlinks。

**Advantages:** 

- Compact representation

- Easy to compute outlinks

**Disadvantages:**

- Cost more to compute inlinks



#### Common friends

两个点之间的共同好友数就是两个点相连的边的公有三角形数。



#### Degree

degree就是图的顶点个数所包含的边数，如果有自己指向自己的循环，那这个degree就是2



### Community Related Algorithms

Community 的特点：dense, connected, possibly additional constraints.



#### K-core

k-core measures whether the subgraph is densely connected.

k-core是子图中的每个点的degree都大于等于k。



**Property of k-core:** 

- Each k’-core is contained in some k-core if k’>k.

- Every vertex in a k-core has degree at least k.



#### A naibe algorithm about finding k-core:

- 多次迭代，将$$degree \lt k$$的点移除，直到不再缩小

time complexity: $$ O(N^2)$$

#### 优化的算法：bucket-based algorithm

Bucket[i]: keeps the node u if degree(u)=i

步骤：

do{:

- Remove the node 𝑢 if degree(u)<k 
  and 𝑢 has the minimum current degree;
- Move the node 𝑣 up by a bin in the 
  bucket if 𝑣 ∈ 𝑛𝑏𝑟(𝑢) 

}while(buckets[0-k-1] is empty)

time complexity: $$O(N)$$



#### Collapsed k-core

即移除某个点后的图，与该点相关的edge都会移除。

Given a graph 𝐺(𝑉, 𝐸) and a set of vertices 𝐴 ⊆ 𝑉, the collapsed k-core, denoted by $$C_k(G_A)$$, is the corresponding k-core of 𝐺 with vertices in 𝐴 removed.

目标是使$$C_k(G_A)$$这个subgraph的size最小



### Link Analysis

#### find single-source shortest paths

algorithm:

1. 初始化所有的点，起始点为0，其他点为$$\infin$$
2. 然后选择距离最近的那个点，并重新计算该点与其他点之间的距离，如果比之前定义的小，则更新；否则不变
3. 重复步骤2，直到Q={}

#### Ranking the results of a query

| Scores                  | Feature    | Example  |
| ----------------------- | ---------- | -------- |
| Query independent score | importance | PageRank |
| Query dependent score   | relevance  | HITS     |



### PageRank

- 给每个页面的超链接的element赋予一个权重，PageRank of E and denoted by PR(E)
- Backlinks as votes, the more votes, the more imporant the page is.

并不是越多人投票就越厉害，而是着重于更有地位的人给你投票。比如一篇论文，被阿猫阿狗引用很多次，不代表这篇论文很重要，但是如果有大佬引用了这篇论文，就说明这篇论文有点重要了。



- 如果有越多的links就表示越重要
- 将in-links当做votes
- 并不是所有links都一样的，越来自越重要的page就拥有更高的值



$$F_u$$-set of pages u points to

$$B_u$$-set of pages pointing to u

$$PR(u)=\sum_{v \in B_u} \frac{PR(v)}{|F_v|}$$



但是PageRank限制只能处理small examples, 因此我们需要更好的方法去处理large web-size graphs.

$$r = M \cdot r$$     #就是WI的迭代方法

M矩阵每一列的值相加为1。



有两个问题：

1. dead ends 即 没有out-links，t'he matrix is not column stochastic so our initial assumptions are not met.
2. spider traps 即一个group中的所有点outlinks是一个死循环，其实不算是problem，不过导致我们pr score不是我们要的。



#### dead ends的解决方法：

teleport（redistribution）：给dead ends的点添加指向其他每个结点的度。



#### Spiders traps 的解决方法：

always-teleport



damping  factor的方法（Random teleport）



#### Topic-Specific PageRank

- Evaluate Web pages not just according to their popularity, but by how close they are to a particular topi

- Allows search queries to be answered based on interests of the user

和传统的PageRank差不多，不过这里更关注的是具有想要topic的Set

$$A_{ij} = \frac{1-d}{|S|} + dM_{ij}$$

在s中的权重都是一样的，不过也能给不同的pages分配不同的权重，最终的结果依然稀疏（sparseness）





### HITS(Hypertext-Induced Topic Selection)

Is a measure of importance of pages or documents, similar to PageRank

- 如果一个站点被引用很多次，那么它是非常权威的，重要站点的引用比不重要站点的引用权重更大
- hubness显示了站点的重要性，一个好的hub是一个链接到许多权威站点的站点。



#### mutually recursive definition

- a good hub links to many good authorities
- a good authority is linked from many good hubs

<img src="/Users/Simonchan/Library/Application Support/typora-user-images/image-20200517171215090.png" alt="image-20200517171215090" style="zoom:50%;" />

当平方差小于一个阈值，即收敛。

$$a = A^T(Aa)=(A^TA)a$$

$$h = A(A^Th)=(AA^T)h$$



### Comparison between PageRank and HITS

共同点： 

PageRank and HITS are two solutions to the same problem:**What is the value of an in-link from u to v**?

- In the PageRank model, the value of the link depends on the links **into** *u* 

- In the HITS model, it depends on the value of the other links **out of** *u* 

不同点：

- PageRank computes authorities only. HITS computes both authorities and hubs.

- The existence of dead ends or spider traps does not affect the solution of HITS.





### Map Reduce System(Distributed System)

遇到的问题：

- storage
- cannot mine one a single server
- network



分布式计算

文档被切分到连续的块中



any distributed system should deal with two task:

- storage
- computation



#### data structures in MapReduce

Key-Value pairs，数值类型可以为integers, float, strings, raw bytes

不过也能为其他任意的数据结构





#### STEP

1. map step:将数据形成k-v键值对形式
2. reduce step：将k-v根据k值group起来，然后排序后reduce



#### Failure

失败的情况分为map阶段失败，reduce阶段失败和master部分失败

- Map worker failure 
  - Map tasks completed or in-progress at worker are reset to idle 
  - Reduce workers are notified when task is rescheduled on another worker 
- Reduce worker failure 
  - Only in-progress tasks are reset to idle 
- Master failure 
  - MapReduce task is aborted and client is notified



### Advanced Topics



![image-20200519204037456](/Users/Simonchan/Library/Application Support/typora-user-images/image-20200519204037456.png)





