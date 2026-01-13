**Variance**

How spread out a set of number is around their mean.

High variance: So a ML model, can often just keep on giving different answer even when we ask the same question again and again. Such model is not reliable. Such model behaves poorly on test data and is said to over fit.

Low variance: On the flip side, if the model continues to give the same answer for a nearly similar question there is a chance that it has a over simplified the test data.

Computed as

![[Pasted image 20260112215414.png]]

Why this formula?

$(xi−μ)$: Gives the distance from the mean.

$(xi−μ)^2$: Squared so that positive and negative distance from the mean don’t cancel each other. Also squaring penalized higher distance from the mean.

$1/N \sum((xi−μ)^2)$: Average over each distance, to get a single number stating amount of spread.

**Bias**

Consistently being wrong in the same direction, by oversimplifying, make an assumption and not considering reality.

High Bias: So a ML model, can make some assumption and always decide on a single answer without considering any other scenarios. These models are generally too simple and miss important pattern of the data, the under fit.

Low Bias: On the flip side, when a ML model has vey little assumption on the dataset, there is chance the model is not confident on the patterns of the data and can be easily swayed.

To improve model’s training so that it doesn’t have variance and bias issue, we employ regularization, early stopping, use simpler models and more data.

**Singular Value Decomposition (SVD)**

SVD says: "Any complex transformation (A) can be broken down into: Rotation → Stretching → Rotation.”

$$A = U . \sigma . V^T$$

In mathematics, rotation means changing your perspective.