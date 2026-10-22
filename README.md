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

(Add your group's reason for choosing each word here.)

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
