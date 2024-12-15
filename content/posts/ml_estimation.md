---
author: Rishab Sharma 
date: '2024-12-15T01:06:38+05:30'
draft: true
title: 'Maximum Likelihood Estimation and Loss Functions'
tags: ["ML Estimation", "KL Divergence", "Loss Functions", "MSE Loss", "Cross Entropy Loss"]
series: ["Themes Guide"]
math: true
font_size: 5
---

When I started learning about loss functions, I could always understand the intuition behind them. For example, the mean squared error (MSE) for regression seemed logical—penalizing large deviations from the ground-truth makes sense. But one thing always bothered me: I could never come up with those loss functions on my own. Where did they come from? Why do we use these specific formulas and not something else? 

This frustration led me to dig deeper into the mathematical and probabilistic foundations of loss functions. It turns out, the answers lie in a concept called Maximum Likelihood Estimation (MLE). In this blog, I’ll take you through this journey, showing how these loss functions are not arbitrary but derive naturally from statistical principles. I'll start by defining what Maximum Likelihood Estimation (MLE) is followed by the intricate connection between Maximum Likelihood Estimation (MLE) and Kullback-Leibler (KL) divergence. 

## Maximum Likelihood estimation (MLE)

Consider a dataset of $ m $ samples $ \mathbb{X} = \\{\mathbf{x^{(1)}}, \dots, \mathbf{x^{(m)}}\\} $ where $ \mathbf{x^{(i)}} \in \mathbb{R}^n $ drawn independently (i.i.d) from the true but unknown data generating distribution $ p_{\text{data}}(\boldsymbol{x}) $. The i.i.d assumption is made in order to simplify the theoretical foundation of many machine learning algorithms. 

Let $ p_{\text{model}}(\boldsymbol{x}; \theta) $ be a parametric family of distributions over a space of parameters $ \theta $. The key idea here is that we are trying to estimate the unknown true distribution $ p_{\text{data}}(\boldsymbol{x}) $ using the model $ p_{\text{model}}(\boldsymbol{x}; \theta) $. 

The goal of any learning algorithm is to estimate $ \theta $ keeping in mind that we only have samples from $ p_{\text{data}}(\boldsymbol{x}) $. Maximum Likelihood Estimation is one way of estimating this $ \theta $. 

The Maximum Likelihood Estimator for $ \theta $ is then defined as 

$$
    \theta_{ML} = \arg\max_\theta p_{\text{model}}(\mathbb{X}; \theta) 
$$

This simplifies to the below expression because of the i.i.d assumption we have made on the data. 

$$
    \theta_{ML} = \arg\max_\theta \prod_{i = 1}^{m} p_{\text{model}}(\boldsymbol{x^{(i)}}; \theta)
$$

This is not the best form to represent this estimator in because of the product over many probabilities. This leads to problems like numerical underflow. We observe that taking the logarithm of the likelihood does not change the solution of the $ \arg\max $. This conveniently changes the product into a sum.

$$
    \theta_{ML} = \arg\max_\theta \sum \log p_{\text{model}}(\boldsymbol{x^{(i)}}; \theta)
$$

One additional observation is that rescaling the cost function does not change the $ \arg\max $. We can divide the above expression by $ m $ to obtain the cost function in terms of an expectation. We arrive at this expectation assuming Monte Carlo Approximation over a large number of samples. 

$$
    \begin{align*}
    \theta_{ML} &= \arg\max_\theta \frac{1}{m} \sum \log p_{\text{model}}(\boldsymbol{x^{(i)}}; \theta) \\\\
    &= \arg\max_\theta \mathbb{E}\_{\mathbf{x} \sim p_{\text{data}}} [\log p_{\text{model}}(\boldsymbol{x^{(i)}}; \theta)] \tag{1} 
    \end{align*}
$$

### Conditional Log Likelihood

Consider a dataset of $ m $ samples $ \\{ (\mathbf{x^{(1)}}, y^{(m)}), \dots, (\mathbf{x^{(m)}}, y^{(m)}) \\} $ drawn independently (i.i.d) from the true but unknown data generating distribution $ p_{\text{data}}(\boldsymbol{x}, y) $. Here $ (\mathbf{x^{(i)}}, y^{(i)}) \in \mathbb{R}^n \times \mathcal{Y} $. In most supervised learning settings, we need to estimate the conditional probability $ P(y | \boldsymbol{x}; \theta) $ in order to predict $ y $ given $ x $. Let's say $ X $ represents all of our inputs and $ Y $ represents all of our targets, then the conditional maximum likelihood estimator is given by: 

$$
    \theta_{ML} = \arg\max_\theta P(Y | X; \theta)
$$

Because of our i.i.d assumption and the property of $ \log $ to not change the solution of $ \arg\max $, the above expression is broken down into

$$
    \begin{align*}
        \theta_{ML} &= \arg\max_\theta \log \prod_{i = 1}^m P(y^{(i)} | \boldsymbol{x^{(i)}}; \theta) \\\\
        &= \arg\max_\theta \sum_{i = 1}^m \log P(y^{(i)} | \boldsymbol{x^{(i)}}; \theta)
    \end{align*}
$$

Many people interpret maximum likelihood estimation as finding the parameters that make the observed data most likely under the assumed  model. I was however not satisfied with this. There is a better interpretation through something called as KL Divergence and that is what we will explore next. 

## What is Kullback-Leibler (KL) divergence

The Kullback-Leibler (KL) divergence is a type of statistical distance between two different distributions. It is used to measure how different two distributions $ P(x) $ and $ Q(x) $ are where both the distributions are over the same random variable $ x $. It is defined as:

$$
    D_{KL}(P||Q) = \mathbb{E}_{\mathbf{x} \sim P} \left[ \log \frac{P(x)}{Q(x)} \right] 
$$

$$
    = \mathbb{E}_{\mathbf{x} \sim P} [\log P(x) - \log Q(x)]
$$

We can notice that KL divergence returns $ 0 $ when $ P $ is the same distribution as $ Q $. KL divergence is also non-negative and can be interpreted as some sort of distance metric between distributions. However, it is not a true distance measure because $ D_{KL}(P||Q) \neq D_{KL}(Q||P) $ for some $ P $ and $ Q $.

The above expectation is equivalent to the below expression for discrete random variables. Here $ \chi $ is the support of $ P $ and $ Q $.

$$
    D_{KL}(P||Q) = \sum_{x \in \chi} P(x) \log \left( \frac{P(x)}{Q(x)} \right) 
$$

However, for the continuous case, the expectation expands into an integral. 

$$
    \int_{x \in \chi} p(x) \log \left( \frac{p(x)}{q(x)} \right) dx
$$

Where $ p $ and $ q $ denote the probability densities of $ P $ and $ Q $ respectively. 

Now that we have seen KL Divergence, it will be easier to go through the connection between maximum likelihood estimation (MLE) and KL Divergence. That is a much more elegant explanation of why MLE is so widespread. 

## Connecting MLE to KL Divergence

So far, we have seen KL Divergence and MLE as two different concepts. However, there is an intricate connection between them. It turns out that minimizing the KL divergence is equivalent to maximizing the ML Estimator. As discussed in this [article](https://researchweb.iiit.ac.in/~md.faizal/articles/like-kl.html), I'll be going through the proof of their equivalence. 

As defined earlier, let $ p_{\text{data}}(\boldsymbol{x}, y) $ be the true data distribution which is unknown and $ p_{\text{model}}(x; \theta) $ be our model distribution which we are trying to learn. We are trying to estimate the true distribution with our model distribution by finding good estimates to its parameters $ \theta $. 

$$
    \begin{align*}
        \arg\min_\theta D_{KL}(p_{\text{data}}||p_{\text{model}}) &= \arg\min_\theta \mathbb{E}\_{\mathbf{x} \sim p_{\text{data}}} \left[ \log \frac{p_{\text{data}}(\boldsymbol{x})}{p_{\text{model}}(\boldsymbol{x}; \theta)} \right] \\\\
        &= \arg\min_\theta \mathbb{E}\_{\mathbf{x} \sim p_{\text{data}}} \left[\log p_{\text{data}}(\boldsymbol{x})\right] - \mathbb{E}\_{\mathbf{x} \sim p_{\text{data}}} \left[p_{\text{model}}(\boldsymbol{x}; \theta)\right] \\\\
        &= \arg\min_\theta - \mathbb{E}\_{\mathbf{x} \sim p_{\text{data}}} \left[p_{\text{model}}(\boldsymbol{x}; \theta)\right] \quad \left( \text{Data distribution doesn't depend on } \theta \right) \\\\
        &= \arg\max_\theta \mathbb{E}\_{\mathbf{x} \sim p_{\text{data}}} \left[p_{\text{model}}(\boldsymbol{x}; \theta)\right]
    \end{align*}
$$

The final expression above is equivalent to the Maximum Likelihood Estimator in $ eq. (1) $. Thus, it is proved that minimizing the KL divergence between $ p_{\text{data}}(\boldsymbol{x}, y) $ and  $ p_{\text{model}}(\boldsymbol{x^{(i)}}; \theta) $ is equivalent to maximizing the likelihood of $ p_{\text{model}}(\boldsymbol{x^{(i)}}; \theta) $. 

## Mean Squared Error as Maximum Likelihood

I follow the proof given in the [deep learning book](http://www.deeplearningbook.org) for this section. 

### Linear Regression

{{< figure src="/images/ml_estimation/linear_regression.png" attr="Fig 1. Linear regression with scalar inputs and outputs" align=center target="_blank" style="width: 30%; height: auto;">}}

First let's define linear regression. The end goal is to take a vector $ x \in \mathbb{R}^d $ and map it into into a scalar $ y \in \mathbb{R} $. The output is a linear function of the input. Let $ \hat{y} $ be the prediction made by our model. We define the model to be

$$
    \hat{y} = \boldsymbol{w^\top} \boldsymbol{x}
$$

Here, $ \boldsymbol{w} \in \mathbb{R}^n $ is a vector of parameters.

Let's vectorize this further. Let's say we have a dataset $ (X^{(train)}, \boldsymbol{y^{(train)}}) $. Here $ X^{(train)} \in \mathbb{R}^{m \times n} $ is a matrix with $ m $ samples of vectors in $ \mathbb{R}^n $. Also, $ \boldsymbol{y^{(train)}} \in \mathbb{R}^m $ is $ m $ samples of scalars written as a vector. Let $ \hat{y}^{\text{(train)}} \in \mathbb{R}^m $ be the predictions from our model. We now define the model to be 

$$
    \boldsymbol{\hat{y}^{\text{(train)}}} = X^{(train)} \boldsymbol{w}
$$

The loss function is defined as

$$
    \begin{align*}
        MSE_{train} &= \frac{1}{m} ||\boldsymbol{\hat{y}^{\text{(train)}}} - \boldsymbol{y^{(train)}}||_2^2 \\\\
        &= \frac{1}{m} ||X^{(train)} \boldsymbol{w} - \boldsymbol{y^{(train)}}||_2^2
    \end{align*}
$$

Intuitively, this can be seen as the penalization of predicted values of $ \boldsymbol{\hat{y}^{(train)}} $ when it deviates from the ground truth targets $ y^{(train)} $. This explanation lacks rigor and fails to be satisfying. For this reason, we will investigate Mean Sqaured Error in the lens of Maximum Likelihood Estimation. 

### Deriving MSE loss with Maximum Likelihood Estimation

Let's get back to the idea of conditional log likelihood. Instead of a single prediction $ \hat{y} $, we now think of the model as producing a conditional distribution $ p(y | \boldsymbol{x}) $. The inherant randomness in $ y $ comes from the fact that a single value of $  \boldsymbol{x} $ input can produce the multiple $ y $ outputs. To derive linear regression, we make an assumption on the distribution of $ p(y | \boldsymbol{x}) $. We define $ p(y | \boldsymbol{x}) = \mathcal{N}(y, \hat{y}(\boldsymbol{x}, \boldsymbol{w}), \sigma^2)$. The mean of this Gaussian distribution is given by $ \hat{y}(\boldsymbol{x}, \boldsymbol{w}) $ which is the model prediction. We also assume a constant variance $ \sigma^2 $. The maximum likelihood estimation of $ p(y | \boldsymbol{x}) $ now gives MSE loss under our given assumption.

Remembering our i.i.d assumption, the conditional log likelihood is given by
$$
    \sum_{i = 1}^m \log (y^{(i)} | \boldsymbol{x}^{(i)}; \theta) \\\\
    = -m \log\sigma - \frac{m}{2}\log(2\pi) - \sum_{i = 1}^m \frac{||\hat{y}^{(i)} - y^{(i)}||_2^2}{2\sigma^2}
$$

We can clearly see the resemblance with the MSE loss term. Here $ m $ is the number of samples. $ \hat{y}^{(i)} $ and $ y^{(i)} $ is the model prediction and the ground truth respectively. 

$$
    MSE_{\text{train}} = \frac{1}{m} \sum_{i = 1}^m ||\hat{y}^{(i)} - y^{(i)}||_2^2
$$

Therefore, maximizing the conditional log likelihood under some given assumptions can be equivalent to minimizing the MSE loss term. Because of the equivalence of MLE with KL Divergence, it can also be seen as the minimization of KL Divergence between $ p_{\text{model}}(y | \boldsymbol{x}; \theta) $ and $ p_{\text{data}}(y | \boldsymbol{x}) $. 

## Cross Entropy as Maximum Likelihood

### Logistic Regression

{{< figure src="/images/ml_estimation/logistic_regression.png" attr="Fig 2. Sigmoid vs Step function. Sigmoid is a better choice for binary classification because it is differentiable at all points in its domain" align=center target="_blank" style="width: 30%; height: auto;">}}

We can generalize linear regression to classification tasks to obtain logistic regression. The end goal of any supervised learning task is to estimate $ \theta $ for the model distribution $ p_{\text{model}}(y | \boldsymbol{x}; \theta) $ by looking at samples from the dataset. In classification, the targets $ y $ can assume binary values $ 0 $ or $ 1 $. This can be solved using the sigmoid function which outputs values in the interval $ (0, 1) $. The outputs of sigmoid can be interpreted as probability values. 

$$
    \begin{align*}
        P(y = 1 | \boldsymbol{x}; \theta) &= \sigma(\theta^\top \boldsymbol{x}) \\\\
        P(y = 0 | \boldsymbol{x}; \theta) &= 1 - P(y = 1 | \boldsymbol{x}; \theta) 
    \end{align*}
$$

As we can clearly observe, $ y $ is a binary random variable. Hence the distribution $ P(y | \boldsymbol{x}; \theta) $ in the case of logistic regression is a Bernoulli distribution. Let $ p := P(y = 1 | \boldsymbol{x}; \theta) $ be the value of the parameter of the Bernoulli distribution. We therefore have

$$
    P(y | \boldsymbol{x}; \theta) = p^{(y)} (1 - p)^{(1 - y)} 
$$

### Deriving Binary Cross Entropy loss with Maximum Likelihood Estimation

In logistic regression too, $ \theta $ can be estimated using maximum likelihood estimation. As we have seen in numerous earlier occasions the conditional log likelihood under our i.i.d assumption is given by

$$
    \begin{align*}
        \theta_{ML} &= \arg\max_\theta \sum_{i = 1}^m \log (y^{(i)} | \boldsymbol{x}^{(i)}; \theta) \\\\
        &= \arg\max_\theta \sum_{i = 1}^m \log p^{y^{(i)}} (1 - p)^{(1 - y^{(i)})} \\\\
        &= \arg\max_\theta \sum_{i = 1}^m (y^{(i)}) \log p + (1 - y^{(i)}) \log (1 - p)\\\\
        &= \arg\min_\theta -\sum_{i = 1}^m (y^{(i)}) \log p + (1 - y^{(i)}) \log (1 - p)
    \end{align*}
$$

And there we have it. The final expression of binary cross entropy is derived from maximum likelihood estimation. Remember that $ p := P(y = 1 | \boldsymbol{x}; \theta) = \sigma(\theta^\top \boldsymbol{x}) $, which is the prediction from our model. Since $ p $ is a function of $ \mathbf{x} $, the term in the summation should technically be $ p(\mathbf{x_i}) $ but I have omitted it for simplicity. 

## Citation

```
@misc{author2024title,
  author       = {Rishab Sharma},
  title        = {Maximum Likelihood Estimation and Loss Functions},
  year         = {2024},
  month        = {Dec},
  howpublished = "https://rish-01.github.io/posts/ml_estimation/",
}

```

## References
[1] Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT Press. Retrieved from [http://www.deeplearningbook.org](http://www.deeplearningbook.org)

[2] Kullback–Leibler divergence. Wikipedia. Retrieved December 15, 2024, from [https://en.wikipedia.org/wiki/Kullback–Leibler_divergence](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence)

[3] Maximzing Likelihood is equivalent to minimizing KL Divergence. Retrieved December 15, 2024, from [https://researchweb.iiit.ac.in/~md.faizal/articles/like-kl.html](https://researchweb.iiit.ac.in/~md.faizal/articles/like-kl.html)

