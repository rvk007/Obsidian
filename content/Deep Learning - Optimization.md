

The goal of optimization is to find parameters θ minimize a loss function L(θ).

Taylor Series: [https://www.youtube.com/watch?v=5Iyah7Qd2Us](https://www.youtube.com/watch?v=5Iyah7Qd2Us)

Aims to approximate a differentiable function. As we add more terms (first order derivative, second order derivative,..) to the series, the accuracy of the approximation increases more and more.

The first-order derivative, tells us the slope of the method. That means, using this slope value we can determine what will be our next position from the current position.

f(x) ≈ f(a) + f’(a)(x-a), this works best when x is only slightly bigger than a.

The second order derivative adds curvature and so on.

**Stochastic Gradient Descent (SGD)**

SGD is the _first-order Taylor Series of the loss function_, used to determine which way to take a step in order to minimize the loss function.

A we want to minimize the loss function, baed on f(x) ≈ f(a) + f’(a)(x-a), we want f(x) < f(a), to guarantee this, we want f’(a)(x-a) to be negative.

f’(a)(x-a) = -z ⇒ x = -(z/f’(a)) + a ⇒ x = a - z/f’(a) This is the update rule of SGD.

To be more aligned with the loss function, L(θ+Δθ) ≈ L(θ) + ∇L(θ)⊤Δθ, the update rule is θ←θ−η∇L(θ), because there is a transpose in the initial Loss update function.

Questions:

1. For optimization given we know that adding more terms to the Taylor Series can lead to more accurate approximation, why did we only use the first-order term in SGD?
    1. Dealing with second-order terms adds complexity, requires more compute and can be noisy. In SGD we take one of a batch of example to optimize the parameters which is usually noisy in comparison to the entire training set, optimizing parameters based on an example or a batch of example can lead to even noisier movements.
2. Does SGD update each parameter equally?
    1. No, SGC does apply the same update rule to each parameter θ←θ−η∇L(θ), however ∇L(θ), derivative of Loss wrt to θ will depend on θ and and hence the parameter will update accordingly.

**Adaptive Moment Estimation (Adam)**


Has a memory of which direction and how big of a step helps minimize the loss, and hence converges faster.

Uses sign of gradient to determine direction and value of gradient to determine the size of the next step.

Maintains a running average of the gradient (Exponential Moving Average (EMA)), by summing all the past gradients, we get the sign of the direction most taken, hence keeping with the momentum.

$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$ , $\beta_1$ is usually 0.9 giving more importance to past gradient than the current gradient.

Also maintains a running average of the square of the gradient (Exponential Moving Average (EMA)), to monitor the size/ magnitude of gradients of the past.

$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$ , $\beta_2$ is usually 0.999 giving more importance to past magnitude of the gradient than the current gradient.

Update rule,

$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$

We have placed $\sqrt{\hat{v}_t}$, in denominator, to ensure when the past magnitude of the gradient is high, we take a smaller step to move slowly and on the flip side when the magnitude of the gradient is low we take a big step to move fast, hence adapting to the gradients.

**Learning Rate Schedules**

Having a constant learning rate throughout the duration of training the model is not an optimal approach. A higher LR will lead to overshoot the minima and a lower LR will take forever to reach the minima. The ideal approach is to start with a higher LR and then after a while move to a lower LR. The decision of when/how to lower the LR introduces many LR schedules.

Step Decay: (Imagine a stair-case), After x epochs drop the LR by y.

Cosine Annealing: Follows a smooth transition to lower LR from higher LR following the cosine graph

Warmup: Starts from 0 learning rate, linearly increase to the higher LR with a few epochs and then starts the decaying slowing using either step decay or cosine annealing method. Helps stabilize the random initialization of the initial weight and so also help stabilize the gradients.

**Weight Decay (L2 Regularization)**

Works with the idea that small weights, generalize better.