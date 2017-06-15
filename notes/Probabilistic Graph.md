<center>
# Probabilistic Graph Model
> Ch.14 *Machine Learning*, Z.H. Zhou
</center>  

### 14.1 Hidden Markov Model (HMM)
Inference: from observed variables to unknown ones.  
Variable of interest: Y, observed: O, others R  

- generative model（生成模型）: P(Y, R, O)      -> P(Y$\mid$O) &nbsp; e.g. HMM, MRF 
- discriminative model（判别模型）: P(Y, R | O) -> P(Y|O) &nbsp; e.g. CRF

Probabilistic Graph Model divides into:

- Bayesian Network: directed acyclic graph
- Markov Network: undirected graph

Our HMM uses simplest dynamic Bayesian network. For Markov chain, on a time sequence (status sequence), hidden status variable $y_i\in S$ depends on $y_{i-1}$ only, while observed variable $x_i\in O$ depends on $y_i$ only. Therefore, 
$$P(x_1, y_1, ..., x_n, y_n)=P(y_1)P(x_1|y_1)\prod_{i=2}^n P(y_i|y_{i-1})P(x_i|y_i)$$

Other variables include, 

- status transition probability matrix $A = [a_{ij}] = [P(y_{t+1}=s_j\mid y_t=s_i)]$
- observation output probability matrix $B=[b_{ij}]=[P(x_t=o_j\mid y_t=s_i)]$
- Status initial probability $\pi=(\pi_1,...,\pi_n)$

HMM can solve 3 problems given $\lambda=[A,B,\pi]$:

- with $\lambda$, predict $P(x|\lambda)$, especially $P(x_n|\lambda)$, or an estimate how an observation matches the model
- with $\lambda,x$, find best match hidden status y (e.g. Speech recognition)
- with $x$, update $\lambda$ so that maximize $P(x|\lambda )$


### 14.2 Markov Random Filed (MRF)
MRF, a typical Markov Network, uses **potential functions (a.k.a. factor)** to define probabilistic distribution on a _subset_ of variables.   
A fully connected sub-graph is called a clique(团) in MRF (e.g. two connected nodes is clique). A clique is called maximal clique if adding any other nodes would disqualify it as a clique. Every clique is in at least one maximal clique. 
Potential function $\psi$ is so defined that, for each clique $Q\in C$ and its corresponding set of variables $x_Q$, joint probability is 
$$P(x)=\frac{1}{Z}\prod_{Q\in C}\psi_Q(x_Q)$$
To avoid explosive calculation cost, we use maximal clique $Q^*$ and set of maximal clique $C^*$ instead. Z functions as normalization factor so that sum of probability is 1.   

If set A, B is separated by C, then C is the separating set of A and B.   

- Global Markov property: Given separating set, then the two sets separated are conditional independent. $P(x_A,x_B|x_C)=P(x_A|x_C)P(x_B|x_C)$  
- Local Markov property: Given every connected node of a node v, then v is independent of any other nodes. Denote $n^*(v)=n(v)\cup v$, and $x_v\bot x_{V - n^*(v)}\mid x_{n(v)}$
- Pairwise Markov property: Given every other nodes, node A and B is conditional independent.  

Potential function $\psi$:  
To maintain non-negativity, we usually use exponential function
$$\psi_Q(x_Q)=e^{-H_Q(x_Q)}$$
$$H_Q(x_Q)=\sum_{u,v\in Q,u \neq v}a_{uv}x_ux_v+\sum_{v\in Q} b_vx_v$$
where a,b are parameters. 


### 14.3 Conditional Random Field (CRF)
...


### 14.4 Learning and inference
Marginalization: to calculate marginal distribution with sum or integral.  
e.g. $P(x_A)=\sum_{x_B}P(x_A,x_B), \ \ where \ V - A = B$







