<center>
# Linear Model
> Ch.3 *Machine Learning*, Z.H. Zhou
</center>  

### 3.2 Linear Regression
On dataset D = {($x_i,y_i$)}, linear regression aims at learning $f(x_i)=wx_i+b$ s.t. $f(x_i)\simeq y_i$. To determine w and b, we minimize **square loss**
$$(w^*,b^*)=\mathop{\arg\min}_{(w,b)}\sum(f(x_i)-y_i)^2$$
This is also called least square method when x is one single feature, where
$$w=\frac{\sum y_i(x_i-\bar{x})}{\sum x_i^2 - \frac{1}{m}(\sum x_i)^2}$$
$$b=\frac{1}{m}\sum(y_i-wx_i)$$
However, when x is multiple features, the targets remain the same while the problem was turned to **multivariate linear regression**. For simplicity, we rewrite $\hat{w}=(w;b),x=(x_1 x_2 ... x_{1d};1)$. Therefore, $$\hat{w}^*=\mathop{\arg\min}_{\hat{w}}(y-X\hat{w})^T(y-X\hat{w})=\mathop{\arg\min}_{\hat{w}}E_\hat{w}$$
to infer gradient $$\frac{\partial E_\hat{w}}{\partial \hat{w}}=2X^T(X\hat{w}-y)=0$$
therefore $$\hat{w}^*=(X^TX)^{-1}X^Ty$$
> If the inverse in the equation does not exist, there will be multiple alternatives that minimizes square loss

A **generalized linear model** would apply this scheme with any differentiable function g() s.t. $y=g^{-1}(w^Tx+b)$


### 3.3 Logistic Regression
From **logistic function** $y=\frac{1}{1+e^{-z}}$, we infer linear model $$y=\frac{1}{1+e^{-w^Tx+b}},\ln\frac{1}{1-y}=w^Tx+b$$
the latter is called **log odds** or **logit**（对数几率）. Here y is conceived as P(x=1).
> Though called regression, Logistic models are applied to classification problems. Its advantages includes:
> 
> - It does not assume any distribution on the data; 
> - It predicts class as well as probability;  
> - It is derivable at any level

Rewrite the equation in a classification problem, we have 
$$\ln\frac{p(y=1|x)}{p(y=0|x)}=w^Tx+b$$
$$p(y=1|x)=\frac{e^{w^Tx+b}}{1+e^{w^Tx+b}}$$
With maximum likelihood method, we infer its log likelihood as 
$$l(w,b)=\sum\ln p(y_i\mid x_i;w,b)$$
Like we did in section 3.2, we rename $\beta=(w;b),\hat{x}=(x;1)$.  
and define $p_1(\hat{x};\beta)=p(y=1|\hat{x};\beta),p_0=1-p_1$.  
therefore $p(y_i|x_i;w,b)=y_ip_1+(1-y_i)p_0$.  
$$l(\beta)=\sum(-y_i\beta^T\hat{x}_i+\ln(1+e^{\beta^T\hat{x}_i}))$$

This convex minimization can be solved with gradient descent method or Newton method. Finally we have $\beta^*=\mathop{\arg\min}_{\beta}l(\beta)$.

>e.g. With Newton method, $\beta$ is updated with
$$\beta^{(t+1)}=\beta^{(t)}-(\frac{\partial^2l(\beta)}{\partial\beta\partial\beta^T})^{-1}\frac{\partial l(\beta)}{\partial\beta}$$
where 
$$\frac{\partial l(\beta)}{\partial\beta}=-\sum\hat{x}_i(y_i-p_1(\hat{x}_i;\beta))$$
$$\frac{\partial^2 l(\beta)}{\partial\beta\partial\beta^T}=\sum\hat{x}_i\hat{x}_i^Tp_1(x_i;\beta)(1-p_1(\hat{x}_i;\beta))$$


### 3.4 LDA
...


