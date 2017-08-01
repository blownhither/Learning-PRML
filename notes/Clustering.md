<center>
# Clustering
> Ch.9 *Machine Learning*, Z.H. Zhou
</center>  

### 9.1 Clustering Task
For dataset $D = {x_1,x_2,...x_m}$, assign class label $\lambda_j$ to each data. This means that $x_j\in D_{\lambda_j}$. 

We also want to increase intra-cluster similarity and lower inter-cluster similarity. 

### 9.2 Measures
To measure clustering performance, we could use external index or internal index. 

External index comes from truth $\lambda^*$ .

$$a=|SS|, SS=\{(x_i, x_j)|\lambda_i=\lambda_j, \lambda_i^*=\lambda_j^*, i< j\}$$
$$b=|SD|, SD=\{(x_i, x_j)|\lambda_i=\lambda_j, \lambda_i^*\neq\lambda_j^*, i< j\}$$
$$c=|DS|, DS=\{(x_i, x_j)|\lambda_i\neq\lambda_j, \lambda_i^*=\lambda_j^*, i< j\}$$
$$d=|DD|, DD=\{(x_i, x_j)|\lambda_i\neq\lambda_j, \lambda_i^*\neq\lambda_j^*, i< j\}$$
$$JC=\frac{a}{a+b+c}$$
$$FMI=\sqrt{\frac{a}{a+b}\frac{a}{a+c}}$$
$$RI=\frac{2(a+d)}{m(m-1)}$$

Internal index does not depend on truth.

$$avg(C)=\frac{2}{|C|(|C|-1)}\sum_{1\leq i< j \leq |C|}dist(x_i,x_j)$$
$$diam(C)=\max dist(x_i,x_j)$$
$$d_{min}(C_i,C_j)=\min_{x_i\in C_i, x_j\in C_j} dist(x_i,x_j)$$
$$d_{cen}(C_i,C_j)=dist(\mu_i,\mu_j)$$
$$DBI=\frac{1}{k}\sum\max(\frac{avg(C_i)+avg(C_j)}{d_{cen}(C_i,C_j)})$$
$$DI=\min_{1\leq i \leq k}\min_{i\neq j}(\frac{d_{min}(C_i,C_j)}{\max diam(C_l)})$$
where $k=|C|$. DB Index wants to be minimized, while Dunn Index wants to be maximized.

### 9.3 Distance measure
Minkovski distance for continuous and ordinal attribute
$$dist_{mk}(x,y)=(\sum_u|x_u-y_u|^p)^\frac{1}{p}$$

For non-ordinal data, there is Value Difference Meric
$$VDM_p(a,b)=\sum_i \lvert\frac{m_{u,a,i}}{m_{u,a}}-\frac{m_{u,b,i}}{m_{u,b}}\rvert^p$$
$m_{u,a}$ - number of samples with $a$ on attribute $u$  
$m_{u,a,i}$ - number of samples in cluster $i$ with $a$ on attribute $u$

Distance measure over attributes of different types combines both and use weight for each attributes.

### 9.4 Prototype clustering
#### k-Means  
k-means minimize $E=\sum_i\sum_{x\in C_i}||x-\mu_i||_2^2$. To fully minimize it is a NP-hard problem. k-means uses greedy instead.   
(Algorithm of k-means is omitted here.)

####Learning Vector Quantization (LVQ)  
When truth data is given, we can use LVQ. LVQ also learns prototype vectors {$p_1,p_2,...,p_q$} stands for each clusters/class. One cluster, one class.
```Python 
def LVQ(D, t, learning_rate):
    p = random_mat                              # prototype vectors
    while not early_stop():
        random_sample(x, y) in D
        d = [distance(x, vec) for vec in p]     # distance array
        i = argmin(d)                           # nearest prototype vector
        if y == t[i]:                           # correct labeling
            p[i] = p[i] + learning_rate * (x - p[i])
        else:                                   # incorrect labeling
            p[i] = p[i] - learning_rate * (x - p[i])
    return p
```




