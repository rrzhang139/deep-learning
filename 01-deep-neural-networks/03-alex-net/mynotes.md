# Alex Net

- ILSVRC 2012: Test error rate was 15%
- Trained with 60 million parameters and 600,000 neurons

 > Before, image datasets were small, on order of 10s of thousands of images
 There was a data bottleneck, as well as model shortcomings. Models those days needed lots of prior information, and models that took in large swaths of data tended to overfit 
 - The network contained 5 convolutional layers, 3 fully-connected layers, much larger than any CNN previously. 
 - Also saw improved optimization and regularization strategies. 
 > In the end, the network’s size is limited mainly by the amount of memory available on current GPUs and by the amount of training time that we are willing to tolerate. 
 This is important, showing there is a data and compute bottleneck 

 ## ReLu 
 > ReLUs train several times faster than their equivalents with tanh units

 ## Training on Multiple GPUs
 > 3GB of memory llimits the size of the network
 >  the GPUs communicate only in certain layers.
 For example, layer 3 takes all kernel maps from layer 2 (from 2 gpus). This allows higher memory bandwidth

 ## 3.3 Local Response Normalization
 > ReLUs have the desirable property that they do not require input normalization to prevent them from saturating
 Tanh have the problem of saturating, which is the idea of the gradient being very small, so weights are not updating. ReLu does not have this problem

 ## 3.4 Overlapping Pooling
 s = 2, z=3 means it does overlapping pooling
 > We generally observe during training that models with overlapping pooling find it slightly more difficult to overfit.
 - improved regularization and decreases loss

 ## 3.5 Overall Architecture
> The first convolutional layer filters the 224×224×3 input image with 96 kernels of size 11×11×3 with a stride of 4 pixels 
> The second convolutional layer takes as input the (response-normalized and pooled) output of the first convolutional layer and filters it with 256 kernels of size 5 ×5 ×48

 ## 4.1 Data Augmentation
 - Data Augmentation is used as a way to increase dataset size that overcomplicated the model so it would not overfit. The strategies used were also low cost. 
 1. image translations and horizontal reflections. 
 > The second form of data augmentation consists of altering the intensities of the RGB channels in training images. 

 ## 4.2 Dropout
 > The recently-introduced technique, called “dropout” [10], consists of setting to zero the output of each hidden neuron with probability 0.5
 This probability happens using a random seed during training. 
 > This technique reduces complex co-adaptations of neurons, since a neuron cannot rely on the presence of particular other neurons
 Allows individual neurons to adapt and learn more high level features that are not dependent on a particular group of neurons. Then it can adapt with any group of neurons
 Dropout doubles the number of iterations required to converge, since neurons are struggling to reduce the loss function without the help of other neurons

## 5 Details of Learning
- Wegiht decay encourages weights to be small and not overinfluence the loss function

