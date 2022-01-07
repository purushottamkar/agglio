# AGGLIO: Global Optimization for Locally Convex Functions.
This repository presents an implementation of the AGGLIO algorithm. The accompanying paper can be accessed at [https://arxiv.org/abs/2111.03932]


## Setup
The python packages required to run the code are listed in requirement.txt, with the respective version number at the time of publishing the code. You may use the following command to install the libraries with the given versions if you are a pip user. 

  pip3 install -r requirements.txt

To install the latest version, please drop the version numbers from the requirements.txt file.

## Dataset
Most experiments are performed on synthetic data, which are generated on the fly. One experiment uses Boston Housing regression dataset which is a standard dataset.

## Executing AGGLIO
The application of AGGLIO is demonstrated in the following:

1. **GLM models based on Sigmoid activation:**
The comparative performance of the variants of AGGLIO (GD, SGD, ADAM, SVRG) for Sigmoid activation in the presence of no noise, pre and post activation noise respectively corresponding to figure 4 in the paper are demonstrated by the jupyter notebook files having name suffixed by _4a, _4b and _4c.
2. **GLM models based on Softplus, Leaky-Softplus and SiLU activation:**
The performance of AGGLIO (GD and SGD) for the GLM models with the three above activation functions in the ideal noiseless setting corresponding to figure 3 in the paper are demonstrated by the jupyter notebook files having name suffixed by _3a, _3b and _3c respectively.
3. **Effect of hyperparameter variation on model recovery error (sigmoid activation):** 
  * Model recovery error with respect to the variation in algorithm parameters: gradient stepsize, intial temperature and temperature increment corresponding to figure 5 in the paper are demonstrated by the jupyter notebook files having names suffixed by _5a, _5b and _5c respectively.
  * Model recovery error with respect to the variation in data parameters: dimension and stdev for pre-activation noisy setting corresponding to figure 7 are demonstrated by the jupyter notebook files having names suffixed by _7a and _7b respectively.

4. **Consistent recovery results (sigmoid activation):**
   Improvement in model recovery error, if any, with the increase in sample size for post and pre activation noise settings, also a justification for Theorem 5 in the paper, corresponding to figure 6a and 6b are demonstrated by the jupyter notebook files having names suffixed by _6a and _6b respectively.
5. **Performance on real regression dataset**:
   The performance of AGGLIO with sigmoid activation on Boston Housing regression dataset (normalized to fit logistic regression) corresponding to figure 6c is demonstrated by the jupyter notebook file having name suffixed by _6c.



## Contributing
This repository is released under the MIT license. If you would like to submit a bugfix or an enhancement to AGGLIO, please open an issue on this GitHub repository. We welcome other suggestions and comments too (please mail the corresponding author at debojyot@cse.iitk.ac.in)

## License
This repository is licensed under the MIT license - please see the [LICENSE](LICENSE) file for details.

## Reference
Dey, Debojyoti, Bhaskar Mukhoty, and Purushottam Kar. "AGGLIO: Global Optimization for Locally Convex Functions." *arXiv preprint arXiv:2111.03932* (2021).