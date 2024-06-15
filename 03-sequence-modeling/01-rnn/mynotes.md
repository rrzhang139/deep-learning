# Abstract
Recurrent means the feedback connections or loops. The output of neuron is dependent on the input and the previous state of context

# Introduction
Recurrent networks can deal with time varying inputs. They can be used for sequence modeling, time series prediction, language modeling, etc.

RNNs enjoy generality of backpropogation through time approach AND no memory constraints

# Learning Algos and Variations

Given n output units and m input units, the weight matrix W is of size n x m.
y(t) are the n output units at time t and x(t) are the m input units at time t.
z(t) are the concatenated input and output units at time t.

Let let the net input to the kth unit at time t be:
$$
s_k(t) = \sum_{i \in U \cup I} w_{ki} z_i(t)
$$

We can therefore produce:
$$
y_k(t + 1) = f_k(s_k(t))
$$

Notice how y_k is updated at time t+1, and s_k used y(t) previously. That means the next state will depend on the previous state y_k(t+1)

Calling this a Temporal Supervised Learning Task.

We want to create a target value for some output unit k at time t+1.
d_k(t) should be the target value that the output of the kth unit at time t should match


$$
e_k(t) =
\begin{cases}
  d_k(t) - y_k(t) & \textrm{if } k \in T(t) \\
  0 & \textrm{otherwise}
\end{cases}
$$

Target values can specify which units to calculate error and thus update


$$
J(t) = 1/2 \sum_{k \in U}[e_k(t)]^2
$$

This is the error function, we want to minimize that.

The weight change is:


$$
\Delta w_{ij}(t) = -\alpha\frac{\partial J(t)}{\partial w_{ij}}
$$

$$
\frac{\partial{J(t)}}{\partial{w_{ij}}} = \sum_{k \in U} e_k(t) \frac{\partial{y_k(t)}}{\partial{w_{ij}}}
$$

$$
\frac{\partial{y_k(t+1)}}{\partial{w_{ij}}} = f'_k(s_k(t))[\sum_{i \in U} w_{ki} \frac{\partial{y_i(t)}}{\partial{w_{ij}}} + \delta_{ik}z_j(t)]
$$

# Real-time Recurrent Learning (RTRL)
We need the weights to be changed while running the network. 
Instead of accumulating the weight changes to each respective weights until the end of the epoch, we instead update the weights at each time step t.

# Discussion
Each weight must have access to the