# Using models penalized models with interaction features

Penalized linear models have been around for a while with formulas like the Elastic-Net:

![equation](https://latex.codecogs.com/svg.latex?%5Carg%5Cmin_%7B%5Cbeta%7D%20%5Cfrac%7B1%7D%7B2n%7D%5C%7Cy-X%5Cbeta%5C%7C_2%5E2%20&plus;%20%5Clambda_1%20%5C%7C%5Cbeta%5C%7C_1%20&plus;%20%5Cfrac%7B%5Clambda_%7B2%7D%7D%7B2%7D%5C%7C%5Cbeta%5C%7C_2%5E2)

But sometimes, the features alone are not enough, and we would like some linear dependance with interactions between the features themselves.
And computing the interactions can be very costly, especially when we don't know exactly which interaction we'll be needing.
Then, attempting feature selection on them might be our only option. With Z the interaction matrix, the model is now:

![equation](https://latex.codecogs.com/svg.latex?%5Carg%5Cmin_%7B%5Cbeta%2C%20%5CTheta%7D%20%5Cfrac%7B1%7D%7B2n%7D%5C%7Cy-X%5Cbeta%20-%20Z%5CTheta%5C%7C_2%5E2%20&plus;%20%5Clambda_%7B%5Cbeta%20l_1%7D%20%5C%7C%5Cbeta%5C%7C_1%20&plus;%20%5Cfrac%7B%5Clambda_%7B%5Cbeta%20l_2%7D%7D%7B2%7D%5C%7C%5Cbeta%5C%7C_2%5E2%20&plus;%20%5Clambda_%7B%5CTheta%20l_1%7D%20%5C%7C%5CTheta%5C%7C_1&plus;%5Cfrac%7B%5Clambda_%7B%5CTheta%20l_2%7D%7D%7B2%7D%5C%7C%5CTheta%5C%7C_2%5E2)

....

# Features included

.....

# Installation
## From git directly

```{bash}
pip install git+https://github.com/josephsalmon/GLMinteractions/InteractionsModel
```

## From source

You need to go where the `setup.py` file is located and install from there, meaning:

```{bash}
$ cd <path>
$ git clone https://github.com/josephsalmon/GLMinteractions <path>
$ pip install -e ./GLMinteractions/InteractionsModel
```
