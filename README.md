# markov_chains

Two main implementations are made: one regarding Variable-length Markov Chains (VLMC), also regarded as Context Trees, and another regarding Hidden Markov Models (HMM). 

### Variable-length Markov Chains

Folder vlmc/ contains source code for inference of VLMC processes. Three algorithms are implemented:
 - BIC: model estimation via minimization of the Bayesian Information Criterion (BIC), in recursive algorithm as proposed in reference [1].
 - BCT: Bayesian Context Tree algorithm, estimation of maximum a posteriori model via recursive algorithm as proposed in reference [2].
 - Context: non-exact implementation of procedure described in reference [3].

Notebook 1-VLMC_examples.ipynb applies implemented algorithms for a set of toy examples.

### Hidden Markov Models

Folder hmm/ contains source code for inference on HMMs. Implementation follows forward and backward equations, Viterbi algorithm and expectation-maximization as described in reference [4]. Notebook 2-HMM_examples.ipynb applies implemented algorithm to toy examples.


### References
[1] Csiszár, I., & Talata, Z. (2006). Context tree estimation for not necessarily finite memory processes, via BIC and MDL. IEEE Transactions on Information theory, 52(3), 1007-1016.\\
[2] Kontoyiannis, I., Mertzanis, L., Panotopoulou, A., Papageorgiou, I., & Skoularidou, M. (2022). Bayesian context trees: Modelling and exact inference for discrete time series. Journal of the Royal Statistical Society Series B: Statistical Methodology, 84(4), 1287-1323.\\
[3] Rissanen, J. (1983). A universal data compression system. IEEE Transactions on information theory, 29(5), 656-664.\\
[4] Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. Proceedings of the IEEE, 77(2), 257-286.
