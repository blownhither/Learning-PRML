<center>
# Introduction
> Ch.2 *Introduction*, S. Shalev-Shawartz & S. Ben-David
</center>  

$X$: Domain set / instance set / input set  
$Y$: Label set  
$D$: Instance distribution (unknown)  
$f$: Perfect classifier (unknown)  
$h$: Current classifier  
$L_{D, f}(h)$: loss / Error rate of a prediction criteria 
$$L_{D, f}(h)=\rm{P}_{x\sim D}[h(x)\neq f(x)]=D(\{x:h(x)\neq f(x)\})$$

$S$: Training set  
$L_s(h)$: Training set loss / training error / empirical error / empirical risk  
$ERM$: empirical risk minimization / minimize $L_s(h)$
> ERM would easily cause over-fitting.  

$H$: Hypothesis set, set of $h$ acceptable (not over-fitted)  
$ERM_H$: choose $h$ from $H$   
$h_S$: the best $h$ in $H$ we found  

i.i.d Hypothesis: independent identical distribution $S\sim D^m$  
$\delta$: Probability of get an untypical sample (failure to learn happens by chance)  
**Confidence Parameter**: 1 - $\delta$  
**Accuracy Parameter**: $\epsilon$. We only accept $L_{D,f}(h) > \epsilon$    

Corollary 2.3: With $m = len(S)$, under any $f$ and any $D$, if 
$$m \leq \frac{\log(|H|\ /\ \delta)}{\epsilon}$$  
we have $1-\delta$ chance (confidence) to have  
$$L_{D,f}(h_S)\leq \epsilon$$
in other words, successful learning
> With big enough training set (large m), we have $1-\delta$ chance of getting correct classifier (loss < $\epsilon$) by $ERM_H$. This is called **Probably Approximately Correct** model (PAC).


<center>
# Deviation and complexity
> Ch.5 *Understanding Machine Learning*, S. Shalev-Shawartz & S. Ben-David
</center>  


...


## 10.2 AdaBoost
Concentrate on instances classified wrongly by increasing their chance of being used as input to a WeakLearner  
Input:   
- S: training set   
- WL: weak learner   
- T: number of iterations   

```python 
# Train  
m = len(D)
D = [1 / m] * m     # chance of each sample being used
for t = 1 ... T:
    h = WL(D, S)    # get weak learner's model on S with distribution D
    error = sum(D * (h(x) != y))                    # weighted error rate
    w = 0.5 * log(1 / error - 1)                    # weight for h
    D = [D[i] * exp(- weight * y[i] * h(x[i])) ...] # tune true positive
    D /= sum(D)                                     # sum to 1

    classifiers[t] = h                              # store each weak classifier
    weights[t] = w

# Test
predictions = [classifiers[i].predict(x) ...]
answer = sign(sum(weights * predictions))

```

Or, in each iteration, weight of **true positive instance** are timed with $\sqrt{\frac{\epsilon}{1 - \epsilon}}$ (then normalized)


