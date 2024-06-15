# Intution


# Abstract
Training deep neural networks: inputs change from the previous layer's parameters changing. Therefore, downstream layers must adapt and be robust to the changes. Must lower learning rate and init, and saturating nonlinearities (ReLU). 
- This is refered to as internal covariate shift. (Internal= inside the network, covariate= input features, shift= change in the distribution of the input features)
Covariate is how one weight changes with respect to another. 
> Our method draws its strength from making normalization a part of the model architecture and performing the normalization for each training mini-batch
# Introduction
Mini-batch SGD offers several advantages: it uses parallelization from modern hardware accelerators. And the gradient of the loss over a minibatch is representative of the gradient of the loss over the whole training set (gradient of loss means the direction in which the loss function decreases the most).

>The training is complicated by the fact that the inputs to each layer are affected by the parameters of all preceding layers – so that small changes to the network parameters amplify as the network becomes deeper
- Deeper networks are harder to control because of this, inputs get progressively out of hand from the original distribution of data. This makes their earlier learning useless. 

>When the input distribution to a learning system changes, it is said to experience covariate shift 

This normalization process is part of the model rather than just a regularizer on the cost function

>Batch Normalization also has a beneficial effect on the gradient flow throughthe network, by reducing the dependence of gradientson the scale of the parameters or of their initial values.
Scaling to same input distribution means that parameter scales dont vary too much. Meaning its scaled so feature importance receives the highest signal, and not the scale of the feature from some weird outlier that was scaled from some previous layers.

# Towards Reducing Internal Covariate Shift
>As each layer observes the inputs produced by the layers below, it would be advantageous to achieve the same whitening of the inputs of each layer.
Whitening refers to normalize inputs to have zero mean and unit variance.

>However, if these modifications are interspersed with the optimization steps, then the gradient descent step may attempt to update the parameters in a way that requires the normalization to be updated, which reduces the effect of the gradient step.
When we normalize, some weights are not accounted for in the gradient step. E.g E[X] will ignore any bias term that influences X. Therefore, as bias parameter is changing, we see the output not changing and thus loss not reducing

>always produces activations with the desired distribution.
Allow the gradient of the loss to account for all weights

$x \hat = \text{Norm} (x, X)$, where x is the input to the layer, X is the mini-batch, and $\hat x$ is the normalized input.

Thus the normalization not only depends on the given training example, but also on all examples
By computing the expected value (mean) over all training examples in the mini-batch, the normalization process takes into account a broader representation of the input distribution.
E.g The bias term is no longer tied to the normalization of just one output x, but rather it influences the normalization of multiple examples in the mini-batch.

# 3 Normalization via Mini-Batch Statistics

![Screenshot 2024-05-13 at 12.11.52 PM.png](../../images/Screenshot_2024-05-13_at_12.11.52_PM.png)

For each neuron, we normalize the input by using the minibatch level statistics (mean and variance)

> Note that simply normalizing each input of a layer may change what the layer can represent
E.g a sigmoid turning linear when inputs are normalized

> To address this, we make sure that the transformation inserted by the normalization can represent the identity transform
![Screenshot 2024-05-13 at 12.16.01 PM.png](../../images/Screenshot_2024-05-13_at_12.16.01_PM.png)
Find two parameters trainable, called the scale and shift parameters. These parameters are applied after the normalization step.
Different layers in a deep neural network may have activations with varying scales and distributions.
The scale and shift parameters allow the network to adapt to these variations by learning the appropriate scaling and shifting for each layer.
This enables the network to handle activations with different magnitudes and distributions effectively.

Each input in a minibatch will by normalized by the mean and variance of the minibatch for that specific layer. After normalizing and passing through the layer, the output is scaled and shifted by the scale and shift parameters.


After training, mini batch normalization and applying statistics are not needed in inference because 

## 3.3 Batch Norm Enables Higher Learning Rates

If we scale a parameter by some factor, it will not affect the outputs

## 3.4 Batch Norm Regularizes the Model

> When training with Batch Normalization, a training ex-ample is seen in conjunction with other examples in themini-batch, and the training network no longer produc-ing deterministic values for a given training example. Inour experiments, we found this effect to be advantageousto the generalization of the network.

# 4.2 ImageNet classification
