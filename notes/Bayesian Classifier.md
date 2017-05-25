<center>
# Bayesian Classifier
> Ch.7 *Machine Learning*, Z.H. Zhou
</center>  


### 7.1 Bayesian decision theory
There are N different labels $c_1, c_2, ..., c_N$, and mistaken $c_i$ to $c_j$ would bring a loss of $\lambda_{ij}$. Therefore, conditional loss is $$R(c_i\mid x)=\sum\lambda_{ij}P(c_j\mid x)$$
and expected loss on decision scheme h is $R(h)=\Bbb{E}[R(h(x)\mid x)]$, which leads us to choose decision scheme as $$h(x)=\arg\min_{c} R(c\mid x)$$ 
This is called **Bayes Optimal Classifier**, with related loss R called Bayes risk.  $\lambda_{ij}$ can be simple as $\lambda: 0\ if\ i = j$, which is expected to lower error rate.

>
- Discriminative Models(判别式模型): Modeling P(c|x). e.g. BP NN, SVM, decision tree
- Generative Models(生成式模型): build $P(c\mid x)=\frac{P(c)P(x\mid c)}{P(x)}$

- P(c): Prior probability
- P(x|c): Likelihood probability, class-conditional probability

### 7.2 Maximum Likelihood Estimation (MLE)
By estimation we mean to determine a probabilistic distribution. We assume P(x|c) is determine by parameter $\theta_c$, thus renaming $P(x\mid c)$ as $P(x\mid \theta_c)$. 
>
- Frequentist(频率学派): believes that parameter is unknown yet fixed
- Bayesian(贝叶斯学派): believes that parameter is random and follow a distribution  
By using Maximum Likelihood Estimation (MLE), we have chosen the frequentist.

Define log-likelihood as $$LL(\theta_c)=\log P(D_c\mid \theta_C) = \sum_{x\in D_c} \log P(x\mid \theta_c)$$
where $D_c$ is samples datasets that belongs to category c.
Then the MLE for $\theta$ is $$\hat\theta_c=\arg\max LL(\theta_c)$$.
>For example, assume $p(x\mid c) \leftarrow \scr{N}(\mu_c, \delta_c^2)$, according to MLE, we have $\hat\mu_c=mean(D_c)$, $\hat\delta_c^2=std(D_c)\cdot\frac{N-1}{N}$

### 7.3 Naive Bayesian Classifiers
With **attribute conditional independence assumption**, we infer 
$$P(c\mid \vec{x})=\frac{P(c)}{P(\vec{x})}\prod P(x_i\mid c)$$
$$P(c)=\frac{|D_c|+1}{|D|+N}$$
$$P(x_i\mid c)=\frac{|D_{c,x_i}|+1}{|D_c|+N_i}\ \rm{,\ for\ discrete\ attribute }$$
$$p(x_i\mid c)=\frac{1}{\sqrt{2\pi}\sigma_{c,i}}\exp(-\frac{(x_i\mu_{c,i})^2}{2\sigma^2_{c,i}})\rm{,\ for\ continuous\ attribute\ (with\ normal\ dist)}$$

### 7.4 Semi-naive Bayes Classifiers
TODO


### 7.5 Bayesian Network
In graph B = < G, $\Theta$ >, every edge node1 -> node2 means node2 depends on node1. 
> e.g. G = {1,2,3}, $\Theta$ = {1->3, 2->3}, then p(x1, x2, x3) = p(x1)p(x2)p(x3|x1, x2). In this case, 1, 2 is independent if 3 is unknown, otherwise they are dependent. We call it **marginal independence**.

/// Bishop
### 8.1.1 Example: Polynomial Regression
Model parameters: w, input x, observed truth t, noise $\sigma^2$, hyperparameter $\alpha$ representing precision of Gaussian prior over w  
$$p(t,w|x,\alpha,\sigma^2)=p(w|a)\prod p(t_n|w,x_n,\sigma^2)$$
> $\alpha$ affects w only, $x_n$ affects $t_n$ only, w and $\sigma^2$ affects t_n only.  
> W is not observed, so w is hidden parameter / latent variable.

<img src="img/8-7.png" width="20%" align="right"></img>  
Our final purpose is to predict t with x, with the graph on right hand we infer $$p(\hat t,t,w|\hat x,x,\alpha,\sigma^2)=[\prod p(t_n|x_n,w,\sigma^2)]p(w|\alpha)p(\hat t|\hat x,w,\sigma^2)$$
$$p(\hat t|\hat x,x,t,\alpha,\sigma^2)\propto \int p(\hat t,t,w|\hat x,x,\alpha,\sigma^2) d w$$

### 8.1.2
To draw a sample $\hat x$ from probabilistic graph $x_1,...,x_K$ (ordered from very parent to vary child), it is easy to imagine sample from $p(x_1)$ first and then $p(x_i|parents_i)$.  
Moreover, to sample from $p(x_2,x_4)$ for example, we may also follow through the whole process above and discard values other than $x_2,x_4$.  
Some probabilistic model aims at synthesis an output, which is called **generative models**. It captures causal process. e.g. produce Image from object and illumination. 





