<center>
# Neural Networks          
> Ch.5 *Machine Learning*, Z.H. Zhou
</center>

## Concepts
- Neuron: n inputs each having weight w, summed by a activation function   
- **Activation function**: (Naive) x>0;  (sigmoid)$\frac{1}{1+e^{-x}}$  
- Perceptron(感知机): A 2-layer network with one output, making decision
    + e.g. $y=f(\sum_i w_i x_i - \theta)$, where f(x) = 1(x>0), can implement AND, OR, NOT
    + practically w is learned from training, also, $\theta$ can be seen as a dummy node with w=-1 that can be learned
    + for sample (x, y), perceptron output $\hat{y}$, then $$w_i = w_i + \delta w = w_i + \eta (y-\hat{y})x_i$$, where $\eta$ is learning rate
    + A linearly separable problem will always come to converge in Perceptron. Others will cause **fluctuation** with unstable w. e.g. XOR
    + Linearly inseparable problems may be solved by multiple layer perceptron.
-   Hidden layer: layers between input and output layer
-   Multi-layer feedforward neural networks (多层前馈神经网络):
    +   A network without in-layer connection or trans-layer connection
    +   This does not mean signal won't be sent backward. But network structure is always forward.

## 5.3 Error Back Propagation (BP, 误差逆传播算法)  
> The most prominent learning rule in practice. Usually used in feedforward networks.   


####def:  

- Training data D = [(x, y), ...], m samples, d input (attributes, and also neurons), l outputs with **threshold** $\theta_j$
- only one hidden layer, containing q hidden neurons with **threshold** $\gamma_h$, output $b_h$
- Connection **weight** between input i and hidden neuron h is $v_{ih}$, those between hidden neuron and output is $w_{hj}$
- Therefore, hidden neuron h accepts $\alpha_h = \sum^d_{i=1} v_{ih}x_i$, output neuron j accepts $\beta_j = \sum^q_{h=1} w_{hj}b_h$
- Learning rate $\eta$ (too big -> fluctuation, too small -> slow)

####proof:
Given input x, output is $\hat{y}_j=f(\beta_j-\theta_j)$, with square error $E_k = \frac{1}{2}\sum(\hat{y}_j - y_j)^2$ (notes 0.5 is merely for convenience).  
Update any parameter v s.t. $v \leftarrow v + \Delta v$  
BP uses **gradient descent** s.t. on the output part$$\Delta w_{hj} = -\eta \frac{\partial E_k}{\partial w_{hj}} = \eta g_j b_h$$
$$\Delta \theta_j = -\eta g_j$$
where output gradient $g_j$ is $$g_j = ... = \hat{y}_j (1-\hat{y}_j) (y_j - \hat{y}_j)$$  
Likewise, on the input side, (could use a $\eta$ different than the two above)
$$\Delta v_{ih} = \eta e_h x_i$$
$$\Delta \gamma_h = -\eta e_h$$
where hidden gradient $e_h$ is $$e_h = ... = b_h(1-b_h)\sum^l_{j=1}w_{hj}g_j$$
> Calculate answer first, modify parameters accordingly, as is called Error Back Propagation

####comments:  
- BP minimizes error on D $E=\frac{1}{m}\sum E_k$
- (Standard) BP calculate E on each sample, **Accumulated Error Back Propagation** works on summed error. ABP need **less** epochs(i.e. rounds) to converge, while Standard BP works better when error is **low** in big training sets.
- With enough hidden neurons, **any continuous function** can be approached. But the number of neurons comes from trail&error in practice.
- BP often encounter **over-fitting**: 
    + **Early Stop**: partition training set into training and validation, stops when validation error increases
    + **Regularization**: punish complex network by $$E=\lambda \frac{1}{m}\sum E_k + (1-\lambda)\sum w_i^2$$ 
    where $\lambda$ comes from cross-validation (?)

> Hint: use one-hot coding for discrete, parallel labels

## 5.4 Local / Global Minimum
solutions:  

- Test more than one random starting point
- **Simulated Annealing(模拟退火)**: Accept _worse_ result at each step with probabilities
- **Random Gradient Descent**
- Genetic Algorithm 遗传算法

## 5.6 Deep Learning
Very deep neural networks.  
However, using Standard BP on deep networks result in diverge.   

- **Unsupervised layer-wise training(无监督逐层训练)** 
    + Pre-training: Use last layer to train a new hidden layer.
    + Fine-tuning: After pre-training, work on the whole network
    This can be seen as partitioning the parameters first and find good configuration in each partition. Then find global optimal with combined local optimal. -> Larger freedom, less cost.
    > e.g. **Deep Belief Network(DBN, 深度信念网络)** pre-train every layer with Boltzmann machine, and then optimize the whole network with BP
- **Weight Sharing (权共享)**
    + A group of neurons using same connection weights.
    > e.g. **Convolutional Neural Network (CNN)**
    > On hand-written numbers [Lecun et al., 1998]


## 5.5 Other common neural networks

- Radial Basis Function RBF（径向基函数）: use Euclid distance and neural center $c_i$ as  activation function
- Competitive learning(竞争学习): unsupervised
    - Adaptive Resonance Theory(ART， 自适应谐振网络): allow incremental learning / online learning
- Self-Organizing Map(SOM, 自组织映射网络)
- Self-adaptive structure network （结构自适应网络）
    + Cascade-Correlation （级联相关网络）
- Recurrent Neural Networks （递归神经网络）
    + Elman Networks
- Boltzmann Machine

## 5.7 Reading

- A good neural network textbook 
    + [Bishop, 1995]
- Tricks 
    + [Reed and Marks, 1998. _Neural Smithing_]
    + [Orr and MUller, 1998. _Tricks of the Trade_]
- Understanding black box 
    + [Tickle et al., 1998. _Extracting knowledge_]
    + [Zhou. 2004. _Rule extraction_]

