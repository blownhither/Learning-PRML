<center> 
#Support Vector Machine
> Ch.6 *Machine Learning*, Z.H. Zhou
</center>

##6.1 Separation and Support Vector
For hyperplane $w^Tx + b = 0$ dividing sample space, any sample x resides $$r=\frac{|w^Tx+b|}{||w||}$$ away from the hyperplane. We want to train the hyperplane so that $$w^Tx_i+b\geq+1, y_i=+1$$ $$w^Tx_i+b\leq-1, y_i=-1$$ (we can always scale w and b to make it 1). We define **margin** as $$\gamma=\frac{2}{||w||}$$. To maximum the margin, we infer basic type of SVM
$$min_{w,b} \frac{1}{2}||w||^2$$ $$s.t.\ \ y_i(w^Tx_i+b)\geq1$$

##6.2 Dual Problem
Using Lagrange method, we infer the dual problem of this optimization:
<!-- $$L(w,b,a)=\frac{1}{2}||w||^2+\sum_i^m a_i(1-y_i(w^Tx_i+b))$$ -->
$$max_{\bf{a}}\sum a_i-\frac{1}{2}\sum_{i=1}^m\sum_{j=1}^ma_ia_jy_iy_jx_i^Tx_j$$
$$s.t.\ \ \sum a_iy_i=0, a_i\geq0$$
and the resulting hyperplane would be $$f(x)=w^Tx+b = \sum a_iy_ix_i^Tx+b$$
> Also, according to KKT condition, the result must follow 
> $$a_i\geq0$$ $$y_if(x_i)-1\geq0$$ $$a_i(y_if(x_i)-1)=0$$
> The third equation indicates $a_i=0$ or $y_if(x_i)=1$. Therefore, you won't find y in the final hyperplane equation.

This can be solved with quadratic programming, yet **Sequential Minimal Optimization(SMO)** is of better efficiency. 
### 6.2.1 SMO Algorithm
SMO updates only 2 parameters among $a_i$. The first chosen parameter is $a_i$ with greatest deviation from KKT conditions. The second chosen parameter is $a_j$ with greatest distance to $a_i$.
> This is a heuristic algorithm because    
>
> - Greatest deviation from KKT conditions might be the most probable item to update the model dramatically.
> - Greatest distance from $a_i$ brings great update, yet is more efficient than finding the gradient of the target function.

After determining which $a_i a_j$ to optimize, there left a single-variable  quadratic programming problem.  
$$a_iy_i+a_jy_j=c=-\sum_{k\neq i,j}a_ky_k$$ $$a_i\geq 0$$ additionally the maximum target above. 
Parameter b is determined with $$b=\frac{1}{|S|}\sum_{s\in S}(1/y_s-\sum_{i\in S}a_iy_ix_i^Tx_s)$$ where S is the index of support vector set (vectors that satisfy $y_i(w^Tx_i+b)=1$ )