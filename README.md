# Multi-Layer Perceptron For CIFAR10

 
 In this project, the aim is to classify 10-CIFAR images using the MLP network.

From this set of 60,000 images, we select 50,000 images as training data and 10,000 images as test data. Among the training images, we give the network 5,000 images as validation data and the rest of the images, 45,000 images, as training data.

**We change the batch size, Activation Function, Cost Function, and Optimizer to see the impact of each.**

 ## **Batch Size**
 
* 32, 64, and 256 are chosen for Batch size. 

The graphs show that a smaller batch size usually leads to faster learning and converges to a good value in less time. However, it is a runaway learning process with higher variance in classification accuracy, resulting in high fluctuations in the loss function. Larger batch sizes slow down the learning process, but the final steps lead to convergence to a more stable model that has less variance in classification accuracy. 

**So, We choose Batch size = 64 for the rest of the process.** 
  
 
 ## **Activation Function**
 

We try to use Tanh and ReLu as activation functions.

**Tanh** is also like sigmoid but better than it. The range of function Tanh is from (0 to −0). Tanh is also like an S-shaped sigmoid.

* **Advantages** 
1. Negative entries are strongly negative, and zero entries are close to zero.

2. It is derivable. 

3. It is a monotonic function. 

4. A convergent gradient function is smooth.

* **Disadvantages**

1. Function prone to vanishing gradient. 

2. It is computationally expensive.

3. Its derivative is not a monotonic function.

**ReLU** is currently the most used activation function since it is used in almost all convolutional neural networks or deep learning.

* **Advantages**
1. It is derivable

2. Both function and its derivative are monotonic functions. 

3. Solves Vanishing Gradient problem. 

4. It is computationally efficient.

* **Disadvantages**

The problem is that all negative values immediately become zero, which reduces the ability of the model to fit or train the data correctly. Any negative input given to the activation function ReLU will immediately turn the value in the graph to zero, affecting the resulting graph by appropriately mapping negative values.


**The performance of this network with ReLU is better, so we use this activation function for the rest of the process.** 
 
 ## **Cost Function**

* Cross Entropy
* MSE

First, Cross Entropy is a better measure of classification than MSE because the decision boundary in classification is more significant (compared to regression). MSE does not sufficiently penalize misclassifications but is a suitable measure for regression where the distance between two predictable values ​​is small.

Second, from a probabilistic point of view, if the non-linear sigmoid or softmax function is in the output layer of the network and you want to maximize the likelihood function to classify the input data correctly, Cross Entropy as a cost function is created naturally. If, on the other hand, the target value is continuous and normally distributed, and we want to obtain the maximum likelihood function of the network output under these assumptions, MSE (combined with a linear output layer) is a suitable choice.

For classification, Cross Entropy is better suited than MSE. (The underlying assumptions make more sense for this setup.) That said, you can train a classifier with the MSE cost function, and it will probably do well (although with nonlinearities, sigmoid/softmax does not work very well, a linear output layer would be a better choice in this case). For regression problems, MSE is almost always used.

**The network's performance is better if we use cross-entropy as a cost function.** 
 
 ## **Optimizers**

*SGD+Momentum
*SGD
*ADAM

**Why is SGD+Momentum used?**

Momentum is an extension of the SGD optimization algorithm, often called momentum gradient descent.
They are designed to speed up the optimization process, e.g., by Reducing the number of function evaluations required to reach the optimum or improving the capability of the optimization algorithm, which leads to a better result.
The problem with the gradient descent algorithm is that the search progress can jump around the search space based on the gradient. For example, a search may progress downhill toward a minimum, but during this progress, it may move in another direction, even uphill, depending on the gradient of specific points (a set of parameters) encountered during the search. This can slow search progress, especially for those optimization problems where the trend or broader shape of the search space is more valuable than specific gradients along the path.
One way to solve this problem is to add history to the parameter update equation based on the gradients encountered in previous updates.
Momentum involves adding an additional hyperparameter that controls the amount of history (momentum) to include in the update equation, i.e., the step to a new point in the search space. The hyperparameter value is defined in the range of 0.0 to 1.0 and often has a value close to 1.0, such as 0.8, 0.9, or 0.99. A momentum of 0.0 is the gradient descent without momentum.
So, in short, we can say that Momentum or SGD with momentum is a method that helps to accelerate the gradient vectors in the right directions and thus leads to faster convergence. It is one of the most popular optimization algorithms, and many advanced models have been trained.

**SGD and ADAM**
The optimization algorithm (or optimizer) is today's primary approach to training a machine learning model to minimize its error rate.
There are two criteria to determine the efficiency of an optimizer:
Convergence speed (the process of reaching a global optimum for gradient descent).
Adaptive Moment Estimation (Adam) and Stochastic Gradient Descent (SGD) optimizers can achieve each of these goals.

Gradient Descent is a famous optimizer. This technique can update each model parameter, observe how a change affects the objective function, choose a direction that minimizes the error rate, and continue to iterate until the objective function is minimized. SGD is a type of Gradient Descent. Instead of performing computations on the entire data set—which is redundant and inefficient—SGD performs computations on only a small subset of a random selection of data samples. SGD performs similarly to conventional Gradient Descent when the learning rate is low.

SGD performs frequent updates with high variance, which causes the objective function to fluctuate wildly. On the other hand, in large datasets, SGD can converge faster because it makes more updates.
Unlike gradient descent optimizers, which adapt weights statically with a constant learning rate in all parameters, adaptive optimizers have more internal flexibility. ADAM is an optimizer that has the characteristics of SGD, but if the dataset is small. It is faster. Also, this optimizer requires less memory and has fewer fluctuations.

**Advantages and disadvantages of ADAM Optimizer**
* Advantages :
1. Can handle sparse gradients, especially on Noisy datasets.
2. The default hyperparameter values ​​work well for most problems.
3. It is computationally efficient.
4. It requires little memory, so it is memory efficient.
5. It works well on large datasets.

* Disadvantages:
1. ADAM does not converge to an optimal solution in some contexts (this is the motivation for AMSGrad).
2. ADAM may have weight loss issues (discussed in AdamW).
3. Recent optimization algorithms are faster and better.

The simulation results for the case where the Batch Size is equal to 64 (the best case of part A) and the ReLU activation function (the best case of Part B), and the Cross-Entropy cost function (the best case of Part C) and the SGD optimizer, SGD+Momentum, and ADAM are brought in the code file. 

 ![](https://github.com/Fateme-Azizabadi/Multi-Layer-Perceptron-For-CIFAR10/blob/main/Images/Final Model.png)
 
 
