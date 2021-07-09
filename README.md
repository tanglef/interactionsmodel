# Master's thesis: High dimensional optimization for penalized linearmodels with first order interactions using graphics card computational power
By Tanguy Lefort.
Under the supervision of ![Joseph Salmon](http://josephsalmon.eu/) and ![Benjamin Charlier](https://imag.umontpellier.fr/~charlier/index.php).
The base material is from the ongoing thesis of ![Florent Bascou](https://bascouflorent.github.io/).

## The problem

The problem to minimize is:

<p align="center">
  <img src="https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20%5Cbg_white%20%5Cfrac%7B1%7D%7B2n%7D%20%5C%7Cy-X%5Cbeta%20-%20Z%5CTheta%5C%7C%5E2_2%20&plus;%20%5Clambda_%7B%5Cbeta%2C%20%5Cell_1%7D%20%5C%7C%5Cbeta%5C%7C_1%20&plus;%20%5Cfrac%7B%5Clambda_%7B%5Cbeta%2C%20%5Cell_2%7D%7D%7B2%7D%20%5C%7C%5Cbeta%5C%7C_2%5E2%20&plus;%20%5Clambda_%7B%5CTheta%2C%20%5Cell_1%7D%5C%7C%5CTheta%7C_1%20&plus;%20%5Cfrac%7B%5Clambda_%7B%5CTheta%2C%20%5Cell_2%7D%7D%7B2%7D%20%5C%7C%5CTheta%5C%7C_2%5E2%5C%20,">
</p>
where

- X is the data with n samples and p features,
- Z is the interactions matrix. It has n samples and p(p+1)/2 features. It is defined as
<p align="center">
  <img src="https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20%5Cbg_white%20Z%20%3D%20%5Bx_1%5Codot%20x_1%2C%5Cdots%2C%20x_1%5Codot%20x_p%20%7C%20x_2%5Codot%20x_2%2C%5Cdots%2C%20x_2%5Codot%20x_p%7C%5Cdots%7C%20x_p%5Codot%20x_p%5D%5C%20,">
</p>
<p>
where for two vectors u and v, <img src="https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20%5Cbg_white%20%28u%5Codot%20v%29_i%20%3D%20u_i%20v_i">.
  
- the hyperparameters λ are regularization factors to induce sparsity and handle the multicolinearity of the features.
</p>

We want to see if using GPUs and more complex (in order of magnitude) algorithms can be competitive against a `Numba` accelerated Coordinate Descent.
<p align="center">
<img src="animation_objs.gif" width="500" height="500" />
</p>  
  
## Content

This repository contains:
- a Python package with optimization solvers using interactions. There are:
  -  Proximal Gradient Descent,
  -  Cyclic Block Proximal Gradient Descent,
  -  both versions of these solvers with double interactions (not keeping only unique interactions),
  -  accelerated versions for each (using Nesterov inertial acceleration).
 
In the package, the `/examples` and `/benchmarks` directories provide source codes of the experiments realized during the internship.
The `Data` module provides access to the genomics dataset used in Chapter 4. This dataset is from ![Sophie Lèbre](https://www.univ-montp3.fr/miap/~lebre/).
Continuous integration is present in `/tests`.
- Report TeX files: in the `internship_report` directory. A pre-compiled version of the report is available at the root of the repository.


