#  Formative 3: Part 1 — Gaussian Mixture Models & EM Algorithm Implementation

This repository contains the complete implementation and technical documentation for **Part 1** of Formative 3 (*Mathematics for Machine Learning*). 

Part 1 focuses on probabilistic modeling of unlabeled human height distributions using a custom-built **Expectation-Maximization (EM) Algorithm** implemented entirely from scratch in Python.

---

##  Objective & Overview

We are given an unlabeled dataset combining two distinct human height populations: **Children** and **Basketball Players / Adults** (derived from Galton’s family height records, $N = 1,868$). 

Because the data lacks explicit identity labels, we treat it as an **unlabeled mixture of two Gaussian (bell-curve) distributions**:
1. **Cluster 1 ($\mu_1 \approx 64.81''$):** Shorter cohort (Children)
2. **Cluster 2 ($\mu_2 \approx 69.59''$):** Taller cohort (Basketball Players / Adults)

---

##  Core Conceptual Question: Why Not Split at the Global Mean?

> **Question:** *Should you or should you not simply draw a line at the dataset's global mean ($\approx 67.97''$) to split the data into two piles and calculate the mean of each pile?*

###  Answer: **NO, you should NOT split at the global mean.**

### **Key Justifications:**
1. **Distribution Overlap:** Human height distributions naturally overlap. A height near $67''$–$68''$ could belong to either a tall child or a shorter adult. A hard cutoff forces arbitrary binary decisions on inherently ambiguous data points.
2. **Metric Distortion:** Forcing a hard split clips the natural tails of both Gaussian curves. This artificially reduces the calculated variance ($\sigma^2$) and biases the true group means ($\mu$).
3. **The Soft Assignment Solution:** The EM algorithm uses **soft assignments** (posterior probabilities via Bayes' Rule between $0$ and $1$). Points near the middle contribute proportionally to updating both distributions rather than being forced into a single pile.

---

##  Mathematical Mechanics & Implementation Details

All functions were implemented **from scratch** using basic Python and NumPy (without machine learning libraries like `scikit-learn`):

1. **Gaussian Probability Density Function (PDF):**
   $$P(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

2. **Expectation Step (E-Step):** Calculates soft responsibilities ($r_{ik}$) using Bayes' Rule:
   $$r_{ik} = \frac{\pi_k \cdot \mathcal{N}(x_i \mid \mu_k, \sigma_k^2)}{\sum_{j=1}^{2} \pi_j \cdot \mathcal{N}(x_i \mid \mu_j, \sigma_j^2)}$$

3. **Maximization Step (M-Step):** Updates cluster parameters using weighted responsibilities:
   $$N_k = \sum_{i=1}^N r_{ik}, \quad \mu_k = \frac{\sum_{i=1}^N r_{ik} x_i}{N_k}, \quad \sigma_k^2 = \frac{\sum_{i=1}^N r_{ik} (x_i - \mu_k)^2}{N_k}, \quad \pi_k = \frac{N_k}{N}$$

4. **Log-Likelihood Evaluation:** Tracks total optimization fit until convergence:
   $$\log L(\theta) = \sum_{i=1}^{N} \log \left( \sum_{k=1}^{2} \pi_k \cdot \mathcal{N}(x_i \mid \mu_k, \sigma_k^2) \right)$$

---

##  Optimization Tracking Table

Below is the required optimization tracking table produced by our implementation, displaying parameter updates from initialization through early iterations to full convergence:

| Iteration | $\mu_1$ (Child) | $\mu_2$ (Pro/Adult) | $\sigma_1^2$ | $\sigma_2^2$ | $\pi_1$ | $\pi_2$ | Log-Likelihood |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0 (Init)** | $56.0000$ | $79.0000$ | $10.9639$ | $10.9639$ | $0.5000$ | $0.5000$ | $-11945.2214$ |
| **1** | $64.8279$ | $70.1720$ | $4.6370$ | $3.6331$ | $0.4118$ | $0.5882$ | $-4894.5713$ |
| **2** | $64.9195$ | $70.0463$ | $5.3587$ | $4.1372$ | $0.4047$ | $0.5953$ | $-4875.3084$ |
| **Final (50)** | **$64.8114$** | **$69.5867$** | **$6.4699$** | **$5.5479$** | **$0.3382$** | **$0.6618$** | **$-4861.8828$** |

*Note: The rapid jump in log-likelihood from $-11,945.22$ to $-4,894.57$ on Iteration 1 demonstrates rapid algorithm convergence toward optimal distribution parameters.*

---

##  Live Bayes' Rule Prediction Engine

Our implementation includes an interactive prediction function (`predict`) that evaluates any test height and outputs posterior probabilities:

```python
def predict(height, mu1, mu2, sigma1, sigma2, pi1, pi2):
    p1 = pi1 * gaussian(height, mu1, sigma1)
    p2 = pi2 * gaussian(height, mu2, sigma2)
    total = p1 + p2

    p_child = p1 / total
    p_pro = p2 / total

    verdict = "Child" if p_child > p_pro else "Basketball Player / Pro"
    return {"Child Probability": p_child, "Pro Probability": p_pro, "Classification": verdict}
```



# Part 2: Bayes' Theorem on IMDb Reviews

## What we did
We used Bayes' Theorem to work out how likely an IMDb review is to be positive when it contains a certain word. We used the 50,000-review IMDb dataset and only basic Python (`csv` and `re`), with no external libraries for the calculation. We calculated P(Positive | keyword) only, not P(Negative | keyword).

## Formula
P(Positive | keyword) = P(keyword | Positive) x P(Positive) / P(keyword)

| Term | Name | How we calculated it |
|---|---|---|
| P(Positive) | Prior | positive reviews / all reviews |
| P(keyword given Positive) | Likelihood | positive reviews containing the word / positive reviews |
| P(keyword) | Marginal | reviews containing the word / all reviews |
| P(Positive given keyword) | Posterior | likelihood x prior / marginal |

A review counts as containing a word if the word appears in it at least once. Reviews were changed to lowercase and `<br />` tags were removed before splitting them into words.

## Keywords
- Positive: excellent, wonderful, masterpiece, brilliant
- Negative: worst, awful, boring, waste

Selected Positive and Negative Words

We selected 4 positive words and 4 negative words based on how strongly they indicate whether a reviewer liked or disliked a film. We avoided general words such as “good” and “great” because they can also appear in negative phrases such as “not good” or “not great.”

Positive Words

Excellent: Often used to praise the acting, story, or overall film. 81% of reviews containing this word are positive.

Wonderful: Usually used when a reviewer enjoyed the film. 81% of reviews containing this word are positive.

Masterpiece: A strong expression of praise, describing a film as exceptionally good. Although it is less common, 73% of reviews containing this word are positive.

Brilliant: Used to describe impressive acting, writing, ideas, or the film as a whole. 76% of reviews containing this word are positive.

Negative Words

Worst: Used to strongly express dislike, such as in “the worst movie.” Only 9% of reviews containing this word are positive.

Awful: Used to describe very poor acting, storytelling, or an overall bad film. Only 10% of reviews containing this word are positive.

Boring: A common complaint used when a film is not interesting or engaging. 19% of reviews containing this word are positive. This is the highest percentage among our negative words because phrases such as “not boring” can appear in positive reviews.

Waste: Commonly appears in phrases such as “waste of time” or “waste of money” to express dissatisfaction. Only 7% of reviews containing this word are positive.

## Results

| Keyword | Prior | Likelihood | Marginal | Posterior |
|---|---|---|---|---|
| excellent | 0.5000 | 0.1146 | 0.0710 | 0.8073 |
| wonderful | 0.5000 | 0.0902 | 0.0555 | 0.8120 |
| masterpiece | 0.5000 | 0.0350 | 0.0240 | 0.7300 |
| brilliant | 0.5000 | 0.0634 | 0.0417 | 0.7606 |
| worst | 0.5000 | 0.0164 | 0.0886 | 0.0924 |
| awful | 0.5000 | 0.0112 | 0.0575 | 0.0977 |
| boring | 0.5000 | 0.0236 | 0.0610 | 0.1938 |
| waste | 0.5000 | 0.0070 | 0.0507 | 0.0687 |

## What we found
- The prior is 0.5 because half the reviews are positive and half are negative.
- The positive words give a posterior between 0.73 and 0.81, well above the prior.
- The negative words give a posterior between 0.07 and 0.19, well below the prior, because almost all reviews containing them are negative.
- "masterpiece" is lower than "excellent" because it also appears in some negative reviews, for example "not a masterpiece".
- We checked every posterior by counting directly (positive reviews with the word / all reviews with the word). The answers matched Bayes' Theorem for all 8 words.

## How to run
1. Put `IMDB Dataset.csv` in the same place as the notebook.
2. Open `part2_bayesian_probability(1).ipynb` and run all the cells from top to bottom.
3. The notebook prints the results table, the direct-count check and a bar chart.

## Files
- `part2_bayesian_probability.ipynb(1)`: the code, results and chart for Part 2


## Parts 3 and 4: Gradient Descent

### Part 3 — Manual Calculation
Part 3 shows the gradient descent process worked out by hand for the given linear model, input data, initial weights, and bias. It calculates the predictions, Mean Squared Error (MSE), gradients, and three parameter updates, showing the intermediate values after each update.

### Part 4 — Python Implementation
Part 4 implements the same process in a Jupyter Notebook using NumPy for matrix calculations, SciPy to approximate derivatives, and Matplotlib to plot the results. The notebook updates the parameters over multiple iterations, calculates the final predictions, and produces separate plots for parameter changes and MSE.

### Observations
- The initial predictions are **[6, 17]**, compared with target values **[5, 6]**, giving an initial MSE of **61**.
- The MSE decreases as gradient descent updates the parameters, showing that the predictions generally move closer to the target values.
- The weights and bias change at each iteration as the algorithm follows the gradients to reduce the error.
- The parameter plot shows how the weights and bias change over time, while the error plot shows the reduction in MSE.

**Files:** `Part 3 Gradient Descent Calculation.pdf` contains the manual work, and `Part_4_Gradient_Descent.ipynb` contains the Python implementation and plots.
