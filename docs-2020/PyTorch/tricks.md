# Tricks

## Iteration with Resample
Adaboost的思想，用不同概率分布的训练数据训练一系列弱分类器，并通过投票融合得到强分类器。类似有效的思想在DL中可以如何体现呢？

打破原有样本分布带来了重要的启示。

```
model = Model()
datasets = cvtk.model_selection.coco.KeepPSamplesIn(coco_dataset)

for epoch in range(total_epochs):
    dataset = random_choice(datasets)
    train_detector(model, dataset, **kw)
    evaluation_detector(model, dataset, **kw)
```

## Deformation


## Class-Rebalancing Sampling
假设`n`个类按样本数目降序排列，第`i`类的样本数为`n_i`。对于被预测为`i`的样本，它会被加入下一轮的训练集的概率为：

$$
\mu_{i} = \left( \frac{N_{n+1-i}}{N_1} \right)^{\alpha}
$$

`\alpha > 0`控制采样频率。对10分类问题，被分为第10/1类的样本被选中的概率为：

$$
\mu_{10} = \left( \frac{N_{1}}{N_1} \right)^{\alpha} \\
\mu_{1} = \left( \frac{N_{10}}{N_1} \right)^{\alpha}
$$
