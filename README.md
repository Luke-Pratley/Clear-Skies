# Clear-Skies
[![Build Status](https://travis-ci.org/Luke-Pratley/Clear-Skies.svg?branch=master)](https://travis-ci.org/Luke-Pratley/Clear-Skies)
[![codecov](https://codecov.io/gh/Luke-Pratley/Clear-Skies/branch/master/graph/badge.svg)](https://codecov.io/gh/Luke-Pratley/Clear-Skies)

Often images are contaminated by Gaussian noise which makes it difficult to detect signals that are smooth, faint, and generally have low surface brightness.
This is even more difficult when the sensitivity varies over an image. Convex methods such as weiner filtering have proven to be useful in noise reduction.
However, when the true signal has correlated and non Gaussian structure, l1 regularization has proven to be a useful tool.

Clear-Skies aims to bring these methods to astronomical images.

This code is built using [Optimus Primal](https://github.com/Luke-Pratley/Optimus-Primal) to apply convex optimization algorithms in deniosing.
