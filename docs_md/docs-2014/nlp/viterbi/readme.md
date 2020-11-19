title: 维特比算法
date: 2016-07-11
tags: [NLP,HMM,Viterbi]
---
维特比算法是一种动态规划算法。它用于寻找最有可能产生观测事件序列的隐含状态序列，特别是在马尔可夫信息源上下文和隐马尔可夫模型中。

<!--more-->
## 算法
假设给定隐式马尔科夫模型(HMM)状态空间S，初始状态i的概率$\pi_i$，从状态i到状态j的转移概率为$a_{i,j}$，令观察到的结果为$y_1,..,y_t$，产生观察结果的最有可能的状态序列为$x_1,..,x_t$，由递推关系给出：

\begin{align}
&V_{1,k} = P(y_1|k) \times \pi_k \\
&V_{t,k} = P(y_t|k) \times \max_{x \in S}(a_{x,k} \times V_{t-1,x})
\end{align}

此处$V_{t,k}$是前t个最终状态为k的观察结果最有可能对应的状态序列的概率。通过保存向后指针记住在第二个等式中用到的状态x，可以获得维特比路径。

## 伪代码
首先是一些问题必要的设置：观察值空间为$O=\{o_1,o_2,..,o_n\}$、状态空间为$S=\{s_1,s_2,..,s_k\}$、A为k\*k转移矩阵，$a_{i,j}$为从状态$s_i$转移到$s_j$的概率、B为k\*n放射矩阵，$b_{i,j}$为在状态$s_i$观察到$o_j$的概率、大小为k的数组$\pi$，$\pi_i$为$x_1=s_i$的概率。观察值序列为$Y=\{y_1,y_2,..,y_t\}$，$X=\{x_1,x_2,..,x_t\}$为生成观察值的状态序列。

在这个动态规划问题中，构造两个大小为k\*t的二维表T1和T2，$T_1[i,j]$保存最可能路径$\hat{x_j}=s_i$的概率，$T_2[i,j]$保存最可能路径$\hat{x_{j-1}}$：

\begin{align}
&T_1[i,j] = \max_{k} (T_1[k,j-1] \times A_{k,i} \times B_{i,j}) \\
&T_2[i,j] = arg \, \max_{k} (T_1[k,j-1] \times A_{k,i} \times B_{i,j})
\end{align}

## 用例实践
想象一个乡村，村民要么健康要么发烧，医生相信村民的健康状况如同一个离散马尔可夫链。村民的状态有：健康、发烧，不能直接观察到。医生询问病人的感觉：正常、冷或头晕，观察值。医生知道村民的总体健康状况，还知道健康和发烧的村民通常会抱怨什么症状。换句话说，医生知道隐马尔可夫模型的参数。用Python语言表示如下：
```python
states = ('Healthy', 'Fever')
observations = ('normal', 'cold', 'dizzy')
start_probability = {'Healthy': 0.6, 'Fever': 0.4}
transition_probability = {
    'Healthy' : {'Healthy': 0.7, 'Fever': 0.3},
    'Fever' : {'Healthy': 0.4, 'Fever': 0.6},
    }
emission_probability = {
    'Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
    'Fever' : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
    }
```

病人连续三天看医生，医生发现第一天他感觉正常，第二天感觉冷，第三天感觉头晕。于是医生产生了一个问题：怎样的健康状态序列最能够解释这些观察结果？解答：
```python
def print_dptable(V):
    print "    ",
    for i in range(len(V)): print "%7d" % i,
    print

    for y in V[0].keys():
        print "%.5s: " % y,
        for t in range(len(V)):
            print "%.7s" % ("%f" % V[t][y]),
        print

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}

    # Initialize base cases (t == 0)
    for y in states:
        V[0][y] = start_p[y] * emit_p[y][obs[0]]
        path[y] = [y]

    # Run Viterbi for t > 0
    for t in range(1,len(obs)):
        V.append({})
        newpath = {}

        for y in states:
            (prob, state) = max([(V[t-1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states])
            V[t][y] = prob
            newpath[y] = path[state] + [y]

        # Don't need to remember the old paths
        path = newpath

    print_dptable(V)
    (prob, state) = max([(V[len(obs) - 1][y], y) for y in states])
    return (prob, path[state])
```

函数viterbi具有以下参数：obs为观察结果序列，例如`['normal', 'cold', 'dizzy']`；states为一组隐含状态；start_p为起始状态概率；trans_p为转移概率；而emit_p为放射概率。为了简化代码，我们假设观察序列obs非空且`trans_p[i][j]`和`emit_p[i][j]`对所有状态`i,j`有定义。在运行的例子中正向/维特比算法使用如下：
```python
def example():
    return viterbi(observations,
                   states,
                   start_probability,
                   transition_probability,
                   emission_probability)
print example()
```

维特比算法揭示了观察结果`['normal', 'cold', 'dizzy']`最有可能由状态序列`['Healthy', 'Healthy', 'Fever']`产生。换句话说，对于观察到的活动，病人第一天感到正常，第二天感到冷时都是健康的，而第三天发烧了。在实现维特比算法时需注意许多编程语言使用浮点数计算，当p很小时可能会导致结果下溢。避免这一问题的常用技巧是在整个计算过程中使用对数概率，在对数系统中也使用了同样的技巧。当算法结束时，可以通过适当的幂运算获得精确结果。

## 参考资料：
- [维特比算法](https://www.52ml.net/8002.html)