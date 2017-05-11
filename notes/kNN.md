<center>
# Dimension Reduction and Distance Metric Learning
> Ch.10 *Machine Learning*, Z.H. Zhou
</center>  

### 10.1 k-Nearest Neighbor
A lazy learning (in contrast to eager learning) scheme using nearest samples around an input, either voting, taking mean, etc.  
Given sample x, with its nearest neighbor z, the probability of wrong classification is equal to the probability that x and z belongs to different classes. $$P(err)=1-\sum_{c\in C}P(c|x)P(c|z)$$
It can be proved that $$P(err)\leq 2(1-P(c^*|x))$$ meaning that its error never proceed twice Perfect Bayesian Classifier could do.

### 10.2 Low-dimension embedding
> __Curse of dimensionality__
> 
- (Sparse samples) Real data usually does not provide dense samples so that 'near' neighbors are available. 
- (Distance computation) high-dimension distance computation is difficult.  

To address this, dimension reduction is needed. **Multiple Dimensional Scaling**(MDS) keeps sample distances in lower dimensions.  
Define distance matrix $D_{m\times m}=[dist_{ij}]$ in d-dimension space, and we will infer a d'-dimension representation of samples $Z\in \mathbb{R}^{d'\times m}$ (Z is column-wise vectors), s.t. $||z_i-z_j||=dist_{ij}$ 
Let $B=Z^TZ$ be the inner product matrix after dimension reduction $b_{ij}=z^T_i z_j$. Therefore, 
$$dist_{ij}^2=||z_i||^2 + ||z_j||^2 - 2z_i^T z_j = b_{ii}+b_{jj}-2b_{ij}$$
For convenience, let $\sum z_i=0$, so $\sum_i b{ij}=0,\sum_j b{ij}=0 $

...