---
title: Batch normlization 原理及其他normalization方法总结
date:  2020-03-13 12:43:13 +0800
category: deep-learning
tags: deep-learning normlization
excerpt: 深度学习基础
mathjax: true
---

BatchNorm作用: **在深度神经网络训练过程中使得每一层神经网络的输入保持相同分布的**。

原始文章：Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

从题目中就可以看出bn是通过减少Internal Covariate Shift来实现训练加速的，所以我们首先介绍什么ICS。

### 1. Internal Covariate Shift
covariate shift：如果ML系统实例集合<X,Y>中的输入值X的分布老是变，这不符合IID假设，网络模型很难稳定的学到规律。

Internal Covariate Shift： 在训练过程中，因为各层参数不停在变化，所以每个隐层都会面临covariate shift的问题，也就是在训练过程中，隐层的输入分布老是变来变去，这就是所谓的“Internal Covariate Shift”，Internal指的是深层网络的隐层，是发生在网络内部的事情，而不是covariate shift问题只发生在输入层。

**总结：** 上述描述说明神经网络训练不稳定的原因主要来自于隐层中分布漂移。所以BN的思想就是如何解决隐层中的分布漂移的问题。

### 2. Motivation
BN不是凭空拍脑袋拍出来的好点子，它是有启发来源的：之前的研究表明如果在图像处理中对输入图像进行白化（Whiten）操作的话——所谓白化，就是对输入数据分布变换到0均值，单位方差的正态分布——那么神经网络会较快收敛，那么BN作者就开始推论了：图像是深度神经网络的输入层，做白化能加快收敛，那么其实对于深度网络来说，其中某个隐层的神经元是下一层的输入，意思是<font color='red'>其实深度神经网络的每一个隐层都是输入层，不过是相对下一层来说而已，那么能不能对每个隐层都做白化呢？这就是启发BN产生的原初想法，而BN也确实就是这么做的，可以理解为对深层神经网络每个隐层神经元的激活值做简化版本的白化操作</font>。

### 3. BN的基本思想
基本思想：通过一定的规范化手段，把每层神经网络任意神经元这个输入值的分布强行拉回到均值为0方差为1的标准正态分布，然后为了保证非线性的获得，对变换后的满足均值为0方差为1的x又进行了scale加上shift操作(y=scale*x+shift)。
[该博客](https://www.cnblogs.com/guoyaohua/p/8724433.html) 简单解释了scale 和 shift 操作的意义：**找到一个线性和非线性的较好平衡点，既能享受非线性的较强表达能力的好处，又避免太靠非线性区两头使得网络收敛速度太慢。**

注意：**按照上述思想，网络结构设计顺序应该是conv-BN-activation。首先通过BN将经过卷积操作的输出归一化分布，使得激活输入的结果能尽可能的落入激活有效区域。** 但实际操作中会发现对于不同的数据，conv-activation-BN实际和上述的差异不是很大。


### 4. BN 实现
#### 4.1. 训练过程
设定$x^k$为BN的输入，$\gamma$ 和 $\beta$ 分别为新数据的方差和均值，首先通过归一化操作将输入分布归一化到[0,1]分布，然后在通过scale操作将数据拉伸到[$\beta$, $\gamma$]分布中。

$$\hat{x}^k = \frac{x^k - E[x^k]}{\sqrt{Var[x^k] +\epsilon}}\\
y^k = \gamma * \hat{x}^k  + \beta$$

其中$\gamma$和$\beta$是需要通过梯度下降训练的。

**手动推导导数：**

$$\frac{y^k}{\gamma} = $$
$$\frac{y^k}{\beta} = $$
$$\frac{y^k}{x^k}$$

#### 4.2. 推断过程
在推断(Inference)过程中，由于输入只有一个实例，无法计算出batch中的均值和方差。解决方案是保存之前在训练过程中的计算出的均值和方差，然后使用这些数据进行估计测试样本的均值方差，概率论中的无偏估计如下：

$$E[x] = E_B(\mu_B) \\
Var[x] = \frac{m}{m-1}E_B(\sigma_B^2)$$

然后使用估计出的均值和方差作用到数据上：

$$y^i = \gamma* \hat{x}^i+\beta \\
=\gamma * \frac{x^i - E[x]}{\sqrt{Var[x] + \epsilon}} + \beta \\
= \frac{\gamma}{\sqrt{Var[x] + \epsilon}} * x^i + (\beta - \frac{\gamma * E[x]}{\sqrt{Var[x] + \epsilon}}) $$

而且$\frac{\gamma}{\sqrt{Var[x] + \epsilon}}$ 和 $(\beta - \frac{\gamma * E[x]}{\sqrt{Var[x] + \epsilon}})$ 均为确定的值，所以就可以在训练过程中使用变量储存在网络中。

#### 4.3. 早期 tensorflow 代码实现
关于tensorflow中代码部分的详解在我[之间的博客](https://blog.csdn.net/selous/article/details/77749776)中详细记录

首先定义bn层：

    import tensorflow as tf
    class ConvolutionalBatchNormalizer(object):
        """Helper class that groups the normalization logic and variables.        

        Use:                                                                      
            ewma = tf.train.ExponentialMovingAverage(decay=0.99)                  
            bn = ConvolutionalBatchNormalizer(depth, 0.001, ewma, True)           
            update_assignments = bn.get_assigner()                                
            x = bn.normalize(y, train=training?)                                  
            (the output x will be batch-normalized).                              
        """

        def __init__(self, depth, epsilon, ewma_trainer, scale_after_norm):
            self.mean = tf.Variable(tf.constant(0.0, shape=[depth]),
                                    trainable=False)
            self.variance = tf.Variable(tf.constant(1.0, shape=[depth]),
                                        trainable=False)
            self.beta = tf.Variable(tf.constant(0.0, shape=[depth]))
            self.gamma = tf.Variable(tf.constant(1.0, shape=[depth]))
            self.ewma_trainer = ewma_trainer
            self.epsilon = epsilon
            self.scale_after_norm = scale_after_norm

        def get_assigner(self):
            """Returns an EWMA apply op that must be invoked after optimization."""
            #在optimization之后必须被调用
            return self.ewma_trainer.apply([self.mean, self.variance])

        def normalize(self, x, train=True):
            """Returns a batch-normalized version of x."""
            if train:
                # 首先计算均值方差
                mean, variance = tf.nn.moments(x, [0, 1, 2])
                assign_mean = self.mean.assign(mean)
                assign_variance = self.variance.assign(variance)
                with tf.control_dependencies([assign_mean, assign_variance]):
                    return tf.nn.batch_normalization(
                        x, mean, variance, self.beta, self.gamma,
                        self.epsilon, self.scale_after_norm)
            else:
                mean = self.ewma_trainer.average(self.mean)
                variance = self.ewma_trainer.average(self.variance)
                local_beta = tf.identity(self.beta)
                local_gamma = tf.identity(self.gamma)
                return tf.nn.batch_normalization(
                    x, mean, variance, local_beta, local_gamma,
                    self.epsilon, self.scale_after_norm)

然后在网络框架中调用：

    #先在层中使用这个类
    ewma = tf.train.ExponentialMovingAverage(decay=0.99)                  
    bn = ConvolutionalBatchNormalizer(depth, 0.001, ewma, True)           
    update_assignments = bn.get_assigner()                                
    x = bn.normalize(y, train=training?)
    #x 就是normalization后的量

    ...

    #定义mean和variance的更新(ExponentialMovingAverage)
    #
    update_assignments = tf.group(bn1.get_assigner(),                         
                                    bn2.get_assigner())                         
    with tf.control_dependencies([optimizer]):                                
        optimizer = tf.group(update_assignments)

当然现在tensorflow已经将bn层集成的很好，一个参数设置就可以在网络中加入BN。

#### 5. 总结
在实际调参过程中，我们发现使用BN层的主要作用是加快了收敛速度，但是在精度上基本不会有提升。至于是否会导致模型表达能力下降，就得考虑到实际问题，实际数据了。

##### 6. 问题
为什么在图像超分领域，从EDSR开始将BN层从基础模块中去除？

个人理解：BN本质还是在解决ICS问题，在分类文中，由于网络输入（图像）和网络输出（类标）之间的分布差异较大，通过BN可以在训练过程中减少错误/过分的分布漂移。但是对于图像超分这种工作，输入和输出之间本身的分布差异就比较小，所以加入BN的影响就不会很大，而且强行的改变分布，也会增加计算复杂度和影响网络的性能。实际调参过程中，去掉BN对模型的不会影响模型性能，当然也没有性能提升。


#### 7. 扩展
基于batch Normalization的改进一般就是在不同数据特征维度上进行。
<center><img src="https://selous123.github.io/assets/img/GN_compare.png" width="700" height="200"/></center>
注: 图片引自论文Group Normlization.
##### 7.1. Layer Normalization
##### 7.2. Weight Normalization
##### 7.3. Instance Normalization
##### 7.4. Group Normalization
##### 7.5. Switchable Normalization
该论文通过大量实验表明，不同的normlization在不同的应用上表现各不相同，所以在实际调参过程中我们应该针对不同的应用选择不同的归一化方式。

#### 8.可能改进的方向
...

Reference:

博客：https://www.cnblogs.com/guoyaohua/p/8724433.html

论文：
[1] Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

[2] Group Normalization

[3] Differentiable learning-to-normalization via switchable normalization.

[4] Layer Normalization

[5] Instance Normalization: The Missing Ingredient for Fast Stylization