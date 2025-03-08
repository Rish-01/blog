---
author: Rishab Sharma 
date: '2025-03-06T00:13:00+05:30'
draft: false
title: 'Variational Autoencoders and Maximum Likelihood Estimation'
tags: ["ML Estimation", "KL Divergence", "Evidence Lower Bound", "Variational Autoencoders"]
series: ["Themes Guide"]
math: true
font_size: 5
---

In my [previous blog](https://rish-01.github.io/blog/posts/ml_estimation/), we explored maximum likelihood estimation (MLE) and how it can be used to derive commonly used loss functions. It also turns out that MLE is widely being used in generative models like Variational Autoencoders (VAE) and Diffusion models (DDPM). 

In this blog, we will explore how the loss function of Variational Autoencoders are derived. VAEs are latent variable generative models. They can solve a few tasks:

1. Act as a generative model that mimics the data distribution that it was trained on.
2. Approximate posterior inference of the latent variable $z$ given an observed variable $x$. In other words, it can be used to learn lower dimensional representations of the data it was trained on.

## Preliminary Information
In this section, let us explore the tools necessary to derive a tractable form of the log-likelihood that we need to optimize. 

### Monte Carlo Approximation

Let us consider an expectation of a function $f(x)$ with respect to a probability distribution $p(x)$ where $x$ is our random variable.

$$
    \mathbb{E}\_{x \sim p(x)} [f(x)] = \int_{x} p(x) f(x) dx
$$

However, the above integral is intractable if $x$ is higher dimensional. Instead of computing the expectation exactly, we can approximate it using Monte Carlo sampling. If we draw $N$ independent samples $\set{x_{1}, \cdots, x_{N}} \overset{\text{i.i.d.}}{\sim} p(x)$, then the Monte Carlo estimate of the expectation is:

$$
    \mathbb{E}\_{x \sim p(x)} [f(x)] \approx \frac{1}{N} \sum_{i=1}^N f(x_i) 
$$

This approximation becomes more accurate as the number of samples $N$ increases, following the law of large numbers.

### Jensen's Inequality
I've taken a very crude proof for Jensen's inequality from [Deep Generative Models (Stanford)](https://youtu.be/MAGBUh77bNg?si=pOFOYeWTWk2EFXeJ). Let us consider the term $\log \mathbb{E}\_{q(\boldsymbol{z})}[f(\boldsymbol{z})]$ and get this in terms of an expectation that is computable with sample averages (Monte-Carlo approximation). We also use the fact that $\log$ is a concave function. From the definition of a concave function, we have the below result for any two points $x$ and $y$ in the domain of $\log(\cdot)$. 

$$
    \log(px + (1 - p)y) >= p \log(x) + (1 - p) \log(y) \quad \forall p \in [0, 1] \tag{1}
$$

We can now use the above result to derive Jensen's inequality. The above result is generalizable to any convex combination of more than two points. 

$$
    \begin{align*}
        \log \mathbb{E}\_{q(\boldsymbol{z})}[f(\boldsymbol{z})] &= \log(\sum_{\boldsymbol{z}} q(\boldsymbol{z})f(\boldsymbol{z})) \\\\
        &>= (\sum_{\boldsymbol{z}} q(\boldsymbol{z}) \log f(\boldsymbol{z})) \quad (\text{from eq. (1)}) \\\\
        &= \mathbb{E}\_{q(\boldsymbol{z})}[\log f(\boldsymbol{z})]
    \end{align*}
$$

### Gaussian Reparametrization

Let's say we want to sample from a Gaussian distribution $z \sim \mathcal{N}(\mu, \sigma^{2})$. Instead of directly sampling from our Gaussian with an arbitrary mean and variance, we can instead sample from a standard normal distribution $\boldsymbol{\epsilon} \sim \mathcal{N}(0, 1)$ and use a deterministic transformation.

$$
    z = \mu + \sigma \cdot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0,1)
$$

It turns out that using the reparametrization trick makes the gradient of our loss function tractable in a VAE. 

## Formalizing the Optimization Problem

Consider a dataset of $ m $ images $ \mathbb{X} = \\{\mathbf{x^{(1)}}, \dots, \mathbf{x^{(m)}}\\} $ where $ \mathbf{x^{(i)}} \in \mathbb{R}^{n \times n} $ is drawn independently (i.i.d) from the true but unknown data generating distribution $ p_{\text{data}}(\boldsymbol{x}) $. Our main goal is to model this distribution using $p_{\theta}(\boldsymbol{x})$ and learn it using data samples. 

Recalling our section on [KL-Divergence](https://rish-01.github.io/blog/posts/ml_estimation/#what-is-kullback-leibler-kl-divergence), our objective can be summarized as:

$$
    L(\theta) = \arg\min_\theta D_{KL}(p_{\text{data}}||p_{\theta})
$$

We further recall from our [previous blog](https://rish-01.github.io/blog/posts/ml_estimation/#connecting-mle-to-kl-divergence) -- the connection between minimizing KL-Divergence and maximizing the likelihood. Our optimization problem then converts to maximizing the log-likelihood.

$$
    \begin{align*}
        L(\theta) &= \arg\min_\theta \mathbb{E}\_{\mathbf{x} \sim p_{\text{data}}(\boldsymbol{x})} \left[ \log \frac{p_{\text{data}}(\boldsymbol{x})}{p_{\theta}(\boldsymbol{x})} \right] \\\\
        &= \arg\min_\theta \mathbb{E}\_{\mathbf{x} \sim p_{\text{data}}(\boldsymbol{x})} \left[ \log p_{\text{data}}(\boldsymbol{x}) - \log p_{\theta}(\boldsymbol{x}) \right] \\\\
        &= \arg\min_\theta \mathbb{E}\_{\mathbf{x} \sim p_{\text{data}}(\boldsymbol{x})} \left[ \log p_{\text{data}}(\boldsymbol{x}) \right] - \mathbb{E}\_{\mathbf{x} \sim p_{\text{data}}} \left[ \log p_{\theta}(\boldsymbol{x}) \right] \\\\
        &= \arg\max_\theta \mathbb{E}\_{\mathbf{x} \sim p_{\text{data}}(\boldsymbol{x})} \left[ \log p_{\theta}(\boldsymbol{x}) \right] \quad (\because \text{First term is independent of } \theta) \\\\
        &= \arg\max_\theta \frac{1}{N} \sum_{i=1}^N \log p_{\theta}(x_i) \quad (\text{Monte-Carlo estimation})
    \end{align*} 
$$

## Deriving the Evidence Lower Bound (ELBO)

Let us consider the term $p_{\theta}(\boldsymbol{x})$. Since $p_{\theta}(\boldsymbol{x})$ is a latent variable model, it can be written as,

$$
    p_{\theta}(\boldsymbol{x}) = \int p_{\theta}(\boldsymbol{x}, \boldsymbol{z}) d\boldsymbol{z} \tag{2}
$$

The above equation is just the marginalization of the joint distribution $p_{\theta}(\boldsymbol{x}, \boldsymbol{z})$. We are interested in computing $\log  p_{\theta}(\boldsymbol{x})$. Our main goal is to write the log-likelihood function in terms of an expectation which is computable. 

$$
    \begin{align*}
        \log p_{\theta}(\boldsymbol{x}) &= \log \int_{\boldsymbol{z}} p_{\theta}(\boldsymbol{x}, \boldsymbol{z}) d\boldsymbol{z} \\\\
        &= \log \int_{\boldsymbol{z}} \frac{p_{\theta}(\boldsymbol{x}, \boldsymbol{z})}{q(\boldsymbol{z} | \boldsymbol{x})} q(\boldsymbol{z}|\boldsymbol{x}) d\boldsymbol{z} \\\\
        &= \log \mathbb{E}\_{q(\boldsymbol{z}|\boldsymbol{x})} \left[ \frac{p_{\theta}(\boldsymbol{x}, \boldsymbol{z})}{q(\boldsymbol{z}|\boldsymbol{x})} \right] \\\\
        &>= \mathbb{E}\_{q(\boldsymbol{z}|\boldsymbol{x})} \left[\log \frac{p_{\theta}(\boldsymbol{x}, \boldsymbol{z})}{q(\boldsymbol{z}|\boldsymbol{x})} \right] \quad (\text{From Jensen's inequality}) \\\\
        &= \mathbb{E}\_{q(\boldsymbol{z}|\boldsymbol{x})} \left[\log \frac{p_{\theta}(\boldsymbol{x}|\boldsymbol{z}) p_{\theta}(\boldsymbol{z})}{q(\boldsymbol{z} |\boldsymbol{x})} \right] \\\\
        &= \mathbb{E}\_{q(\boldsymbol{z}|\boldsymbol{x})} \left[\log p_{\theta}(\boldsymbol{x}|\boldsymbol{z}) \right] - \mathbb{E}\_{q(\boldsymbol{z}|\boldsymbol{x})} \left[\log \frac{q(\boldsymbol{z}|\boldsymbol{x})}{p_{\theta}(\boldsymbol{z})} \right] \\\\
        &= \mathbb{E}\_{q(\boldsymbol{z}|\boldsymbol{x})} \left[\log p_{\theta}(\boldsymbol{x}|\boldsymbol{z}) \right] - D_{KL}(q(\boldsymbol{z}|\boldsymbol{x}) || p_{\theta}(\boldsymbol{z})) \\\\
        &:= F_{\theta}(q)
    \end{align*}
$$

We have derived a tractable lower bound $F_{\theta}(q)$ (ELBO) for log-likelihood that we can maximize. The tightness of this lower bound depends on the choice of $q(\boldsymbol{z}|\boldsymbol{x})$ (variational posterior).  

## Analysis on the Variational Posterior $q(\boldsymbol{z}|\boldsymbol{x})$ and Tightness of Lower Bound

Let us consider the term $(\log p_{\theta}(\boldsymbol{x}) - F_{\theta}(q))$. To have a tight lower bound, we need to choose a $q$ that minimizes this difference. 

$$
    \begin{align*}
        \log p_{\theta}(\boldsymbol{x}) - F\_{\theta}(q) &= \log p_{\theta}(\boldsymbol{x})  - \mathbb{E}\_{q(\boldsymbol{z}|\boldsymbol{x})} \left[\log \frac{p_{\theta}(\boldsymbol{x}, \boldsymbol{z})}{q(\boldsymbol{z}|\boldsymbol{x})} \right] \\\\
        &= \log p_{\theta}(\boldsymbol{x}) - \mathbb{E}\_{q(\boldsymbol{z}|\boldsymbol{x})} \left[\log \frac{p_{\theta}(\boldsymbol{z}|\boldsymbol{x}) p_{\theta}(\boldsymbol{x})}{q(\boldsymbol{z} |\boldsymbol{x})} \right] \\\\
        &= \log p_{\theta}(\boldsymbol{x}) - \int_{\boldsymbol{z}} q(\boldsymbol{z} |\boldsymbol{x}) \log \left(\frac{p_{\theta}(\boldsymbol{z}|\boldsymbol{x}) p_{\theta}(\boldsymbol{x})}{q(\boldsymbol{z} |\boldsymbol{x})} \right) d\boldsymbol{z} \\\\
        &= \log p_{\theta}(\boldsymbol{x}) - \int_{\boldsymbol{z}} q(\boldsymbol{z} |\boldsymbol{x}) \log p_{\theta}(\boldsymbol{x})  d\boldsymbol{z} - \int_{\boldsymbol{z}} q(\boldsymbol{z} |\boldsymbol{x}) \log \left(\frac{p_{\theta}(\boldsymbol{z}|\boldsymbol{x})}{{q(\boldsymbol{z} |\boldsymbol{x})}} \right) d\boldsymbol{z} \\\\
        &= \log p_{\theta}(\boldsymbol{x}) - \log p_{\theta}(\boldsymbol{x})  \int_{\boldsymbol{z}} q(\boldsymbol{z} |\boldsymbol{x})  d\boldsymbol{z} - \int_{\boldsymbol{z}} q(\boldsymbol{z} |\boldsymbol{x}) \log \left(\frac{p_{\theta}(\boldsymbol{z}|\boldsymbol{x})}{{q(\boldsymbol{z} |\boldsymbol{x})}} \right) d\boldsymbol{z} \\\\
        &= \cancel{\log p_{\theta}(\boldsymbol{x})} - \cancel{\log p_{\theta}(\boldsymbol{x})} -  \int_{\boldsymbol{z}} q(\boldsymbol{z} |\boldsymbol{x}) \log \left(\frac{p_{\theta}(\boldsymbol{z}|\boldsymbol{x})}{{q(\boldsymbol{z} |\boldsymbol{x})}} \right) d\boldsymbol{z} \quad (\because \int_{\boldsymbol{z}} q(\boldsymbol{z} |\boldsymbol{x}) d\boldsymbol{z} = 1)\\\\
        &= - \int_{\boldsymbol{z}} q(\boldsymbol{z} |\boldsymbol{x}) \log \left(\frac{p_{\theta}(\boldsymbol{z}|\boldsymbol{x})}{{q(\boldsymbol{z} |\boldsymbol{x})}} \right) d\boldsymbol{z} \\\\
        &= \int_{\boldsymbol{z}} q(\boldsymbol{z} |\boldsymbol{x}) \log \left(\frac{{q(\boldsymbol{z} |\boldsymbol{x})}}{p_{\theta}(\boldsymbol{z}|\boldsymbol{x})} \right) d\boldsymbol{z} \\\\
        &= D_{KL}(q(\boldsymbol{z} |\boldsymbol{x}) || p_{\theta}(\boldsymbol{z}|\boldsymbol{x}))
    \end{align*}
$$

We have therefore proved that the error from the lower bound is, 

$$
    \log p_{\theta}(\boldsymbol{x}) - F\_{\theta}(q) = D_{KL}(q(\boldsymbol{z} |\boldsymbol{x}) || p_{\theta}(\boldsymbol{z}|\boldsymbol{x}))
$$

We can find the optimal value for $q(\boldsymbol{z} |\boldsymbol{x})$ by setting this error to zero. 

$$
    D_{KL}(q(\boldsymbol{z} |\boldsymbol{x}) || p_{\theta}(\boldsymbol{z}|\boldsymbol{x})) = 0
$$

Therefore, the optimal value of $q(\boldsymbol{z} |\boldsymbol{x})$ is $p_{\theta}(\boldsymbol{z}|\boldsymbol{x})$ where the lower bound is known to be the tightest. However, $p_{\theta}(\boldsymbol{z}|\boldsymbol{x})$ is not tractable in VAEs which is why we explore alternate forms for $q(\boldsymbol{z} |\boldsymbol{x})$. 

## Modeling VAE using Neural Networks 
{{< figure src="/blog/images/vae/Variational_Autoencoder.png" attr="Fig 1. Variational Autoencoder" align=center target="_blank" style="width: 30%; height: auto;">}}

In a VAE, $q(\boldsymbol{z}|\boldsymbol{x})$ is modeled using a stochastic neural network with parameters $\phi$. We assume a Gaussian form for $q(\boldsymbol{z}|\boldsymbol{x})$ for mathematical convenience. Therefore, the encoder can be used to predict the parameters of our Gaussian distribution. 

$$
    q_{\phi}(\boldsymbol{z}|\boldsymbol{x}) = \mathcal{N}(z; \mu_{\phi}(\boldsymbol{x}), \Sigma_{\phi}(\boldsymbol{x}))
$$

We also assume a diagonal form for the covariance matrix $\Sigma_{\phi}(\boldsymbol{x}) = diag(\sigma_{1}, \cdots, \sigma_{k})$ for further convenience. 

$$
    \therefore q_{\phi}(\boldsymbol{z}|\boldsymbol{x}) = \mathcal{N}(z; \mu_{\phi}(\boldsymbol{x}), diag(\sigma_{1}, \cdots, \sigma_{k}))
$$

Here, our latent variable $\boldsymbol{z} \in \mathbb{R}^k$ and $D = \set{\boldsymbol{x_i}}_{i=1}^n \sim p\_{\text{data}}$ where $x_i \in \mathbb{R}^d \quad \forall i \in \set{1, \cdots, n}$. Since our latent variable is $k$-dimensional, $\mu(\boldsymbol{x}) \in \mathbb{R}^k$, and $ [\sigma_1, \cdots, \sigma_k]^\top \in \mathbb{R}^k $. Our encoder therefore maps our input to $\mathbb{R}^{2k}$ dimensional space to estimate the parameters of our Gaussian distribution, $\mu(\boldsymbol{x})$, and  $diag(\sigma_1, \cdots, \sigma_k)$.  

## Computing the Gradient of the ELBO for Training

Let us go back to the ELBO and try to compute it's gradient with respect to the parameters of our neural network. Let us recall the expression for ELBO. It is given by,

$$
    \begin{align*}
        F\_{\theta}(q_{\phi}) &= \mathbb{E}\_{q_{\phi}(\boldsymbol{z}|\boldsymbol{x})} \left[\log \frac{p_{\theta}(\boldsymbol{x}, \boldsymbol{z})}{q_{\phi}(\boldsymbol{z}|\boldsymbol{x})} \right] \\\\
        &= \mathbb{E}\_{q_{\phi}(\boldsymbol{z}|\boldsymbol{x})} \left[\log p_{\theta}(\boldsymbol{z}, \boldsymbol{x}) - \log q_{\phi}(\boldsymbol{z}|\boldsymbol{x}) \right] 
    \end{align*}
$$

The argument of our expectation is a function of both $x$ and $z$. Therefore, we define our ELBO as $\mathbb{E}\_{z \sim q_{\phi}(\boldsymbol{z}|\boldsymbol{x})} [f_{\theta, \phi}(\boldsymbol{z}, \boldsymbol{x})]$ 

$$
    \begin{align*}
        \mathbb{E}\_{z \sim q_{\phi}(\boldsymbol{z}|\boldsymbol{x})} [f_{\theta, \phi}(\boldsymbol{z}, \boldsymbol{x})] &:= \mathbb{E}\_{z \sim q_{\phi}(\boldsymbol{z}|\boldsymbol{x})} \left[\log p_{\theta}(\boldsymbol{z}, \boldsymbol{x}) - \log q_{\phi}(\boldsymbol{z}|\boldsymbol{x}) \right] 
    \end{align*}
$$

Let us consider the gradient of the ELBO with respect to $\phi$. 

$$
    \begin{align*}
        \nabla_{\phi} \mathbb{E}\_{z \sim q_{\phi}(\boldsymbol{z}|\boldsymbol{x})} [f_{\theta, \phi}(\boldsymbol{z}, \boldsymbol{x})] &= \nabla_{\phi} \int_{\boldsymbol{z}} q_{\phi}(\boldsymbol{z}|\boldsymbol{x}) f_{\theta, \phi}(\boldsymbol{z}, \boldsymbol{x}) d\boldsymbol{z}\\\\
        &= \int_{\boldsymbol{z}} \nabla_{\phi} (q_{\phi}(\boldsymbol{z}|\boldsymbol{x}) f_{\theta, \phi}(\boldsymbol{z}, \boldsymbol{x})) d\boldsymbol{z} \\\\
        &= \int_{\boldsymbol{z}} f_{\theta, \phi}(\boldsymbol{z}, \boldsymbol{x}) \nabla_{\phi} q_{\phi}(\boldsymbol{z}|\boldsymbol{x}) d\boldsymbol{z} + \int_{\boldsymbol{z}} q_{\phi}(\boldsymbol{z}|\boldsymbol{x}) \nabla_{\phi} f_{\theta, \phi}(\boldsymbol{z}, \boldsymbol{x}) d\boldsymbol{z} \quad (\text{product rule})
    \end{align*}
$$

The first term, which is an integral, is intractable. We cannot write it in the form of an expectation that can be estimated using sample averages (Monte-Carlo estimation). 

### The Reparametrization Trick

{{< figure src="/blog/images/vae/Reparametrization.png" attr="Fig 2. Updated forward pass with Gaussian reparametrization" align=center target="_blank" style="width: 30%; height: auto;">}}

Let us recall that $q_{\phi}(\boldsymbol{z}|\boldsymbol{x}) = \mathcal{N}(z; \mu_{\phi}(\boldsymbol{x}), diag(\sigma_{1}, \cdots, \sigma_{k}))$. We can therefore use the Gaussian reparametrization trick to instead sample from $\mathcal{N}(0, I)$ and use a deterministic transformation to get a sample from $q_{\phi}(\boldsymbol{z}|\boldsymbol{x})$. 

Therefore, the sample $z \sim \mathcal{N}(z; \mu_{\phi}(\boldsymbol{x}), diag(\sigma_{1}, \cdots, \sigma_{k}))$ is equivalent to,

$$
    \begin{align*}
        z &= \mu_{\phi}(\boldsymbol{x}) + \sigma_{\phi}(\boldsymbol{x}) \odot \boldsymbol{\epsilon} \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, I) \\\\
        &:= g(\boldsymbol{\epsilon}; \phi)
    \end{align*} 
$$

Here, $\odot$ denotes element-wise multiplication and $\sigma_{\phi}(\boldsymbol{x}) = [\sigma_{1}, \cdots, \sigma_{k}]^\top$.

We can use the reparametrization trick to make the computation of gradient tractable. We use a simplified notation for $f_{\theta, \phi}(\boldsymbol{z}, \boldsymbol{x})$, as $f(\boldsymbol{z}, \boldsymbol{x})$ to avoid clutter. 

$$
    \begin{align*}
        \nabla_{\phi} \mathbb{E}\_{z \sim q_{\phi}(\boldsymbol{z}|\boldsymbol{x})} [f(\boldsymbol{z}, \boldsymbol{x})]  &= \nabla_{\phi} \mathbb{E}\_{\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)} [f({g(\boldsymbol{\epsilon}; \phi)}, \boldsymbol{x})] \\\\
        &= \nabla_{\phi} \int_{\boldsymbol{\epsilon}} \mathcal{N}(0, I) [f({g(\boldsymbol{\epsilon}; \phi)}, \boldsymbol{x})] d \boldsymbol{\epsilon} \\\\
        &= \int_{\boldsymbol{\epsilon}} \mathcal{N}(0, I) \nabla_{\phi}  [f({g(\boldsymbol{\epsilon}; \phi)}, \boldsymbol{x})] d \boldsymbol{\epsilon} \\\\
        &= \mathbb{E}\_{\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)} [\nabla_{\phi} f({g(\boldsymbol{\epsilon}; \phi)}, \boldsymbol{x})] \\\\
        &\approx \frac{1}{N} \sum_{k=1}^{N} \nabla_{\phi} f({g(\boldsymbol{\epsilon}_k; \phi)}, \boldsymbol{x})
    \end{align*}
$$

$\mu(\boldsymbol{x_i}), \Sigma(\boldsymbol{x_i})$

$\boldsymbol{z_i} =\mu_{\phi}(\boldsymbol{x_i}) + \sigma_{\phi}(\boldsymbol{x_i}) \odot \boldsymbol{\epsilon_i} \quad \boldsymbol{\epsilon_i} \sim \mathcal{N}(0, I)$ 

$(\text{Here},\boldsymbol{z_i} \sim q_{\phi}(\boldsymbol{z}|\boldsymbol{x_i}))$

$\boldsymbol{x_i} \sim p_{\theta}(\boldsymbol{x}|\boldsymbol{z_i})$

## Citation

Cited as:

> Rishab Sharma. (March 2025). Variational Autoencoders and Maximum Likelihood Estimation. https://rish-01.github.io/blog/posts/vae/

or

```bibtex
@article{Rishab2025vae,
  author       = "Rishab Sharma",
  title        = "Variational Autoencoders and Maximum Likelihood Estimation",
  journal      = "rish-01.github.io/blog",
  year         = "2025",
  month        = "March",
  howpublished = "https://rish-01.github.io/blog/posts/vae/",
}

```

## References

[1] Diederik P Kingma, Max Welling. [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114). arXiv preprint arXiv:1312.6114 (2022).

[2] [Stanford CS236: Deep Generative Models I 2023 I Lecture 6 - VAEs](https://youtu.be/8cO61e_8oPY?si=4YgMVYen2O0MSz9o). YouTube.

[3] [Lec 9 - Deep Generative Models Variational Auto Encoders](https://youtu.be/c475SLygCK4?si=sIGn-DPwT2LgEMoy). YouTube. 