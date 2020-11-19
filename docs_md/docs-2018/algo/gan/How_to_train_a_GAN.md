# How to train a GAN
虽然生成对抗网络(GAN)的研究继续改善这些模型的基本稳定性,但我们使用一系列技巧来训练它们并使它们日常稳定.以下是一些技巧的摘要.[原文](https://github.com/soumith/ganhacks)

## 1 Normalize the inputs
- Normalize the images between -1 and 1
- Tanh as the last layer of the generator output

## 2 A modified loss function
In GAN papers, the loss function to optimize G is `min (log 1-D)`, but in practice folks practically use `max log D`

## 3 BatchNorm
- Construct different mini-batches for real and fake, i.e. each mini-batch needs to contain only all real images or all generated images
- When batchnorm is not an option use instance normalization (for each sample, subtract mean and divide by standard deviation)

## 4 Avoid Sparse Gradients: ReLU, MaxPool
- The stability of the GAN game suffers if you have sparse gradients
- LeakyReLU = good (in both G and D)
- For Downsampling, use: Average Pooling, `Conv2d + stride`
- For Upsampling, use: PixelShuffle, `ConvTranspose2d + stride`

## 5 Use Soft and Noisy Labels
- Label Smoothing, i.e. if you have two target labels: `Real=1` and `Fake=0`, then for each incoming sample, if it is real, then replace the label with a random number between 0.7 and 1.2, and if it is a fake sample, replace it with 0.0 and 0.3
- Make the labels the noisy for the discriminator: occasionally flip the labels when training the discriminator

## 6 DCGAN / Hybrid Models
- Use DCGAN when you can. It works!
- If you cant use DCGANs and no model is stable, use a hybrid model: `KL+GAN` or `VAE+GAN`

## 7 Use the Adam Optimizer
- Use SGD for discriminator and Adam for generator

## 8 Track failures early
- D loss goes to 0: failure mode
- Check norms of gradients: if they are over 100 things are screwing up
- When things are working, D loss has low variance and goes down over time vs having huge variance and spiking
- If loss of generator steadily decreases, then it's fooling D with garbage

## 9 Dont balance loss via statistics (unless you have a good reason to)
- Dont try to find a (number of G / number of D) schedule to uncollapse training
- If you do try it, have a principled approach to it, rather than intuition

For example:
```
while lossD > A:
  train D
while lossG > B:
  train G
```

## 10 If you have labels, use them
If you have labels available, training the discriminator to also classify the samples: auxillary GANs

## 11 Add noise to inputs, decay over time
- Add some artificial noise to inputs to D
- Adding gaussian noise to every layer of generator

## 12 [notsure] Train discriminator more (sometimes)
- Especially when you have noise
- Hard to find a schedule of number of D iterations vs G iterations

## 13 [notsure] Batch Discrimination
Mixed results.

## 14 Discrete variables in Conditional GANs
- Use an Embedding layer
- Add as additional channels to images
- Keep embedding dimensionality low and upsample to match image channel size

## 15 Use Dropouts in G in both train and test phase
- Provide noise in the form of dropout (50%)
- Apply on several layers of our generator at both training and test time

## Generative Models
https://github.com/wiseodd/generative-models