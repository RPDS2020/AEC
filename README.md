# AEC

Pytorch Implementation of <a href="https://doi.org/10.1016/j.neunet.2024.106346">High-performance Deep Spiking Neural Networks via At-most-two-spike Exponential Coding</a> 

# Incorrect Formula Correction

We have identified an error in Formula (6) in Section 2.3 of Chapter 2 in the paper. The original formula in the paper is 

$$\begin{align}
e^{l} &= \sum_{i=1}^{M^{l}}\left|a_{i}^{l}-s_{i}^{l}(f)\right|  \\
      &\leq\sum_{j=1}^{L^{l}}\left|s_{j}^{l}(T)-a_{j}^{l}\right| + \sum_{k=1}^{M^{l}-(U^{l} + L^{l})}\left|s_{k}^{l}(f-1)-s_{k}^{l}(f)\right| \\
      &\leq e_{c,min}^{l} + e_{q}^{l} + e_{c,max}^{l}
\end{align}$$


and its correct form should be

$$\begin{align}
e^{l} &= \sum_{i=1}^{M^{l}}\left|a_{i}^{l}-s_{i}^{l}(f)\right|  \\
			&\leq\sum_{j=1}^{L^{l}}\left|s_{j}^{l}(T)-a_{j}^{l}\right| + \sum_{k=1}^{M^{l}-(U^{l} + L^{l})}\left|s_{k}^{l}(f-1)-s_{k}^{l}(f)\right| + \sum_{i=1}^{U^{l}}\left|a_{i}^{l}-s_{i}^{l}(0)\right| \\
			&\leq e_{c,min}^{l} + e_{q}^{l} + e_{c,max}^{l}
\end{align}$$


