<center>
# Decision Tree
> Ch.4 *Machine Learning*, Z.H. Zhou
</center>  

### General Procedures
```python
def TreeGenerate(D, A):
    """ Dataset D=(x,y), Attribute set A """
    node = Node()
    # case 1
    if all_same(D.y):   
        node.leaf = True
        node.c = D.y        # set leaf same class 
        return
    # case 2
    if empty(A) or all_same(D.a):
        node.leaf = True
        node.c = most(D.y)
        return
    # case 3
    a_star = best_division_attr(A)
    for set(D[a_star]) as a_v:      # each value on a_star
        D_v = D[D[a_star] == a_v]   # subset by a_star value
        if empty(D_v):
            branch.leaf = True
            branch.c = most(D_v.c)
        else:
            branch = TreeGenerate(D_v, A - a_star)  
        node.add_branch(branch)
```


### 4.2 Division scheme
Our purpose is to improve **purity** as nodes are divided
#### Information entropy
> Information entropy stands for purity of a set. With ratio $p_k$ of class k in set $D$, information entropy is defined as $$Ent(D)=-\sum_kp_k\log_2 p_k$$

To maximize information Gain
$$Gain(D, a)=Ent(D) - \sum_v \frac{|D^v|}{|D|}Ent(D^v)$$
$$a_*=\arg\max Gain(D, a)$$

But using $Gain$ would prefer attribute which has many classes. An extreme example is index of data, which maximizes purity but leads to little generalization ability. 

#### Gain Ratio
Addressing the problem above, **C4.5** algorithm uses gain ratio instead
$$Gain\_ratio(D, a) = \frac{Gain(D, a)}{IV(a)}$$
$$IV(a) = -\sum_v\frac{|D^v|}{|D|}\log_2\frac{|D^v|}{|D|}$$
where $IV(a)$ is **intrinsic value** of $a$ which increases as number of classes increases. 

Important: intrinsic value prefers attribute with less classes instead, and C4.5 algorithm thus does not maximize $Gain_ratio$. It uses heuristic method instead.

#### CART decision tree
CART uses Gini index of dataset
$$Gini(D)=\sum_k\sum_{k'\neq k}p_kp_{k'}=1-\sum_kp_k^2$$
$Gini$ stands for the possibility that any two samples drawn from D are different.
$$Gini\_index(D,a) = \sum_v\frac{|D^v|}{|D|}Gini(D^v)$$
$$a_*=\arg\max Gini\_index(D,a)$$

### 4.3 Pruning 
Pre-pruning stops at certain point and replace potential subtree with one leaf. Post-pruning substitutes some subtree with leaf. We discuss pruning based on validation set.

Pre-pruning is simple. By not dividing the node, we get a leaf devoted to the most frequent class of D, which gives an accuracy on validation set. By dividing the node, we get a subtree by splitting D on a, which also gives an accuracy on validation set. Compare the accuracies and decide.




