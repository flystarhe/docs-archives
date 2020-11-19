title: 基于感知器的分词算法
date: 2017-01-05
tags: [NLP,感知器,分词]
---
多类感知器比CRF简单不少，使用SIGHAN2005(icwb2-data)的语料数据进行了分词实验，基于感知器的中文分词算法的性能与基于更复杂模型CRF的分词算法相比相差不到1个百分点。基于感知器的分词实现，其实是用多类感知器代替发射概率的HMM。

<!--more-->
词性标注是一个监督学习，所谓监督学习，指的是给你一张表格，并且告诉你在测试的时候最后一列会被拿掉。你的工作就是通过其他列去预测最后这一列。对于词性标注来讲，缺失的这一列就是词性。其他列（特征）可能是“上个词”的词性，“下个词”的最后三个字符（作者研究的是英语）之类的信息。

## 预测
这是运行时的预测：
```
def predict(self, features):
    '''Dot-product the features and current weights and return the best class.'''
    scores = defaultdict(float)
    for feat in features:
        if feat not in self.weights:
            continue
        weights = self.weights[feat]
        for clas, weight in weights.items():
            scores[clas] += weight
    # Do a secondary alphabetic sort, for stability
    return max(self.classes, key=lambda clas: (scores[clas], clas))
```

前面我将学习问题比作一张表格，该表格在预测的时候缺失了最后一列。对NLP来讲，我们的表格通常特别稀疏。比如`word i-1=Parliament`，这种特征的数量可能接近0。所以我们的`weight vectors`可能从来都不会做成向量形式。Map类型是个好主意，在这里我们使用了Python的dictionary类型。

输入数据（特征），是表格中频次大于0的那些`column`的集合。通常在实现的时候可以写成dictionary类型，这样就可以set值了。但我直接将其写成“命中与否”的二值类型。

`weights`是一个双层的dictionary，最终成为`feature/class/weight`。你想要这种结构，而不是反向，因为字频率分布的方式：大多数字是罕见的，频繁的字是非常频繁的。

## 训练
那么我们如何得到权重的值呢？我们从空`weights`字典开始，并迭代地执行以下操作：

1. 输入一个`(features, tag_truth)`
2. 用当前`weights`预测`tag_guess`
3. `if tag_truth != tag_guess`
    1. 将`features, tag_truth`的`weights`加1；
    2. 将`features, tag_guess`的`weights`减1；

它是最简单的训练算法，每当模型犯错，就将正确`class`对应的权重（注：复数，指的是正确的那些特征函数，下文同理）增加，并惩罚导致预测错误的权重。Python代码如下：
```
def train(self, nr_iter, examples):
    for i in range(nr_iter):
        for features, true_tag in examples:
            guess = self.predict(features)
            if guess != true_tag:
                for f in features:
                    self.weights[f][true_tag] += 1
                    self.weights[f][guess] -= 1
        random.shuffle(examples)
```

如果你在一个训练实例上运行上述代码，该实例的词性相关的权重很快就能算出来。如果是两个训练实例呢？除非它们的特征各不相同，你才可能得到相应的正确结果。通常，只要示例是线性可分的，算法就会收敛。

## 平均权重
我们需要做一件事来使感知器算法具有竞争力。目前算法的问题是，如果你在两个轻微不同的实例集中训练，可能会得到截然不同的模型。模型不能聪明地泛化问题。更大的问题在于，如果你让算法跑到收敛，它会过于关注那些误分类的点，并且调整整个模型以适应它们。

所以，我们要减小特征权重的变化速度（让模型没法在一个迭代中作太多改变，以免前功尽弃）。我们的做法是返回平均权重，而不是最终权重。我猜很多人会说，为什么？但我们不是来做学究的，这个做法已经被长时间的实践证明了。如果你有其他主意，欢迎做做试验然后与大家共享。事实上，我很期待看到更多新发现，因为现在平均感知机在NLP中的地位已经变得非常之高了。

如何平均呢？注意我们并不在每个外层循环中做平均，我们在内层循环中作平均。如果我们有5000个训练实例，我们训练10个迭代，对每个权重我们平均50000次。我们当然不会把这么多中间变量都存起来了，我们只记录每个权重的累加值，然后除以最终的迭代次数，不就平均了吗。我们想要的是每个`feature/class`对应的平均权重，所以关键点在于其累计权重。但如何累计权重也是个技术活。对于大部分训练实例，都只会引起特定的几个权重变化，其他特征函数的权重都不会变化。所以我们不应该傻乎乎地将这些不变的权重也累加起来。

让我们一起实现上述改进算法吧。我们维护一个额外的字典`_timestamps`，记录每个权重上次变化的时间。当我们改变一个特征权重的时候，我们就取出这个值用于更新。下面就是更新时具体的代码了：
```
def update(self, truth, guess, features):
    def upd_feat(c, f, v):
        nr_iters_at_this_weight = self.i - self._timestamps[f][c]
        self._totals[f][c] += nr_iters_at_this_weight * self.weights[f][c]
        self.weights[f][c] += v
        self._timestamps[f][c] = self.i

    self.i += 1
    for f in features:
        upd_feat(truth, f, 1.0)
        upd_feat(guess, f, -1.0)
```

## 特征和预处理
词性标注语料中有成千上万错综复杂的特征，包括大小写和标点符号等。它们对华尔街语料的准确率是有帮助的，但我没见到对其他语料有任何帮助。为了训练一个更通用的模型，我们将在特征提取之前预处理数据：

1. 所有词语都转小写（英文）
2. `1800-2100`之间的数转义为`!YEAR`
3. 其他数字字符串转义为`!DIGITS`
4. 识别日期、电话、邮箱等并转义

我用这些特征，能够取得的97%的成绩、以及较低的内存占用，还是很划算的。特征提取代码如下：
```
def _get_features(self, i, word, context, prev, prev2):
    '''Map tokens-in-contexts into a feature representation, implemented as a
    set. If the features change, a new model must be trained.'''
    def add(name, *args):
        features.add('+'.join((name,) + tuple(args)))

    features = set()
    add('bias') # This acts sort of like a prior
    add('i suffix', word[-3:])
    add('i pref1', word[0])
    add('i-1 tag', prev)
    add('i-2 tag', prev2)
    add('i tag+i-2 tag', prev, prev2)
    add('i word', context[i])
    add('i-1 tag+i word', prev, context[i])
    add('i-1 word', context[i-1])
    add('i-1 suffix', context[i-1][-3:])
    add('i-2 word', context[i-2])
    add('i+1 word', context[i+1])
    add('i+1 suffix', context[i+1][-3:])
    add('i+2 word', context[i+2])
    return features
```

## 关于搜索
推荐模型直接输出当前词的`class`，然后处理下一个。这些预测作为下一个词语的特征。这么做有潜在问题，但问题也不大。用柱搜索很容易解决，但照我说根本不值得。更不用说采用类似如条件随机场那样的又慢又复杂的算法了。

这个潜在的问题是这样的：当前位置假设是3，上个词语和下个词语是2和4，在当前位置决定词性的最佳因素可能是词语本身，但次之就是2和4的词性了。于是就导致了一个“鸡生蛋”的问题：我们想要预测3的词性，可是它会用到未确定的2或4的词性。举个具体的例子：

    Their management plan reforms worked

根据上述算法，你可以发现从左到右处理句子和从右到左会得到完全不同的结果。如果你还没明白，那就这么思考：从右往左看，“worked”很可能是个动词，所以基于此把“reform”标注为名词。但如果你从左往右、从“plan”出发，可能就会觉得“reform”是名词或动词。

只有当你会犯错的时候才需要用搜索。它可以防止错误的传播，或者说你将来的决策会改正这个错误。这就是为什么对词性标注来说，搜索非常重要！你的模型非常棒的时候，你过去做的决定总是对的的话，你就不需要搜索了。贪心搜索的时候注意，模型必须假设测试时历史标签是不完美的。否则它就会过于依赖历史标签。由于感知机是迭代式的，所以这很简单。下面是标注器的训练主循环：
```
def train(self, sentences, save_loc=None, nr_iter=5, quiet=False):
    '''Train a model from sentences, and save it at save_loc. nr_iter
    controls the number of Perceptron training iterations.'''
    self._make_tagdict(sentences, quiet=quiet)
    self.model.classes = self.classes
    prev, prev2 = START
    for iter_ in range(nr_iter):
        c = 0; n = 0
        for words, tags in sentences:
            context = START + [self._normalize(w) for w in words] + END
            for i, word in enumerate(words):
                guess = self.tagdict.get(word)
                if not guess:
                    feats = self._get_features(i, word, context, prev, prev2)
                    guess = self.model.predict(feats)
                    self.model.update(tags[i], guess, feats)
                # Set the history features from the guesses, not the
                # true tags
                prev2 = prev; prev = guess
                c += guess == tags[i]; n += 1
        random.shuffle(sentences)
        if not quiet:
            print("Iter %d: %d/%d=%.3f" % (iter_, c, n, _pc(c, n)))
    self.model.average_weights()
    # Pickle as a binary file
    if save_loc is not None:
        cPickle.dump((self.model.weights, self.tagdict, self.classes), open(save_loc, 'wb'), -1)
```

你可以在这里看到源代码的其余部分：[taggers.py](https://github.com/sloria/textblob-aptagger/blob/master/textblob_aptagger/taggers.py)，[perceptron.py](https://github.com/sloria/textblob-aptagger/blob/master/textblob_aptagger/_perceptron.py)。当然也可以参考[hankcs/AveragedPerceptronPython](https://github.com/hankcs/AveragedPerceptronPython)或[minitools/cws.py](https://github.com/zhangkaixu/minitools/blob/master/cws.py)。

## 参考资料：
- [基于感知器的中文分词算法](http://heshenghuan.github.io/2015/12/21/基于感知器的中文分词算法/)
- [200行Python代码实现感知机词性标注器](http://www.hankcs.com/nlp/averaged-perceptron-tagger.html)
- [基于感知器算法的高效中文分词与词性标注系统设计与实现](http://www.doc88.com/p-3147585302795.html)
- [A Good POS Tagger in about 200 Lines of Python](https://explosion.ai/blog/part-of-speech-pos-tagger-in-python)