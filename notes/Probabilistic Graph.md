<center>
# Probabilistic Graph Model
> Ch.14 *Machine Learning*, Z.H. Zhou
</center>  

### 14.1 Hidden Markov Model
Inference: from observed variables to unknown ones.  
Variable of interest: Y, observed: O, others R  

- generative model（生成模型）: P(Y, R, O)      -> P(Y$\mid$O)
- discriminative model（判别模型）: P(Y, R | O) -> P(Y|O)

Probabilistic Graph Model divides into:

- Bayesian Network: directed acyclic graph
- Markov Network: undirected graph

Our HMM uses simplest dynamic Bayesian network. For Markov chain, on a time sequence (status sequence), hidden status variable $y_i\in S$ depends on $y_{i-1}$ only, while observed variable $x_i\in O$ depends on $y_i$ only.
