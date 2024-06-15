

# Intution
The motivation was from neurobio, in which neurons fire at some rate and are sparse. There exists some saturation threshold where verything before it is not turned on. 
There is a tradeoff on richness of representation (as well as good optimization/convergence and generalizable) and sparsity (interpretable, clear, energy efficient)
The notion of distributiveness is important, because neurons can learn more dynamically and adapt to the data. 

In the brain 1-4% of neurons are active at the same time. In neural networks, sigmoid activation turns on 1/2 of the neurons. This hurts gradient optimization because the gradients are small and network learns slowly. 
An activation function is called asymmetric or symmetric if its response to an opposite excitatory state is respectively inhibitory or excitatory.
- Tangent hyperbolic functions rests at 0, but it is asymmetric (meaning if its opposite then its inhibitory), this is not seen in biological neurons.

## 2.2 Sparsity 
> However, inthe latter, the neurons end up taking small but non-zero activation or firing probability. We show here thatusing a rectifying non-linearity gives rise to real zerosof activations and thus truly sparse representations.From a computational point of view, such representa-tions are appealing
Dense representations are highly entangled because any change in input will affect all the neurons. Sparse representations are more disentangled and easier to learn.
> Different inputs may contain different amounts of information and would be more conveniently represented using a variable-size data-structur
- Different data comes in different dimensionality, so it is better to have sparsity that will allow the network to learn the dimensionality of the data.
>  forcing too much sparsity may hurt pre-dictive performance for an equal number of neurons,because it reduces the effective capacity of the model.

# 3 Deep Rectifier Neural Networks
Rectifier: f(x) = max(0, x)
Cortical neurons can be approximated by rectifiers. 
Does not enforce antisymmetry 
Advantages:
- Given all negative gradients are set to zero, more data is sparse
- Introduces nonlinearities for a subset of neurons that are active. Therefore gradients flow well on this group, and we can interpret the output as a sum of linear parts

### Potential Problems
>One may hypothesize that the hard saturation at 0 may hurt optimization by blocking gradient back-propagation
- gradient flow is essentially blocked when backpropogating, which can slow the learning processs. 
>hypothesize that the hard non-linearities do not hurt so long as the gradient can propagate along some paths, i.e., that some of the hidden units in each layer are non-zero
- 
>Another problem could arise due to the unbounded behavior of the activations; one may thus want to use a regularizer to prevent potential numerical problems. Therefore, we use the L1 penalty on the activation values, which also promotes additional sparsity
- L1 regularizer applied to loss objective will prevent any weight from having too large of a value, thus overpowering the other weights and introducing bias

## 3.2 Unsupervised Pretraining

> difficulties arise when one wants to introduce rectifier activations into stacked denoising auto-encoders (Vincent et al., 2008). First, the hard saturation below the threshold of the rectifier function is not suited for the reconstruction units. Indeed, whenever the network happens to reconstruct a zero in place of a non-zero target, the reconstruction unit can not backpropagate any gradien
- This is called dead neurons, they cannot learn because their weights never update since during backpropogation we multiply by zeros
- Autoencoders suffer the most because they are trying to reconstruct the input, and if the input is zero, then the network will not learn anything.