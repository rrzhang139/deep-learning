# Intution
This seems to have a feeling of passing raw, unfiltered data deeper into the network to remind the inner layers of the data they are transforming. 
This allows optimization 
Convergence is faster for Resnets. This suggests 

# Abstract
Instead, layers learn residual functions. Easier to optimize and gain increased depth

# Introduction
>Driven by the significance of depth, a question arises: Is learning better networks as easy as stacking more layers?
- deeper suggests more rich representations but harder to optimize. Vanishing gradients came from ReLU which hamper convergence from the beginning
> he existence of this constructed solution indicatesthat a deeper model should produce no higher training errorthan its shallower counterpart
- This shows that we are getting higher training errors by adding more layers as opposed to a shallower network. 


# Deep Residual Learning
Solvers might have trouble approximating the identity mappings by multiple nonlinear layers.
Residual learning sort of reminds the nonlinear layers to learn the true data and thus learn the identity.
Solvers may drive the weights to zero

![Screenshot 2024-05-09 at 11.38.41 AM.png](../../images/Screenshot_2024-05-09_at_11.38.41_AM.png)

