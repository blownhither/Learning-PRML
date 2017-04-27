# Bayesian Classifier
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

### 7.3 Naive Bayesian Classifier
With **attribute conditional independence assumption**, we infer 
$$P(c\mid \vec{x})=\frac{P(c)}{P(\vec{x})}\prod P(x_i\mid c)$$
$$P(c)=\frac{|D_c|+1}{|D|+N}$$
$$P(x_i\mid c)=\frac{|D_{c,x_i}|+1}{|D_c|+N_i}\ \rm{,\ for\ discrete\ attribute }$$
$$p(x_i\mid c)=\frac{1}{\sqrt{2\pi}\sigma_{c,i}}\exp(-\frac{(x_i\mu_{c,i})^2}{2\sigma^2_{c,i}})\rm{,\ for\ continuous\ attribute\ (with\ normal\ dist)}$$






