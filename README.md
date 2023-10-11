# pyegotaxis

The python code in this repository was developed in the research group of Benjamin M. Friedrich at TU Dresden, Dresden, Germany, for the simulation of chemotactic agents.

## Install

For the compilation the following packages are needed:
*SWIG, g++, python headers and libz dev*

For debian/ubuntu: 
*'apt install build-essential SWIG python3-dev zlib1g-dev'*

Then install with *pip install pyegotaxis*.

## Usage
This is a minimal example to run a simulation with an agent of size a=0.01 using infotaxis in a search domain with absorbing boundary at radius Rmax=0.5.

```python
data = et.simdata( maxTime=10000, N=100, l=.5, Drot=0.01, lambda_=1, agent_size=0.01)
res = et.run_sim(data,et.Infotaxis_ben(data),et.predict_adv_diff_diffrot(data),
                 update_step=et.ben())
```

## Acknowledgements
This version of the code was developed by Julian Rode 
to perform the simulations in [1], 
building on an earlier version developed by Andrea Auconi 
used for the publication [[2](https://iopscience.iop.org/article/10.1209/0295-5075/ac6620)].
It is recommended that both references [1,[2](https://iopscience.iop.org/article/10.1209/0295-5075/ac6620)] are cited if this code is used.
This copyright statement must not be removed. 

[1] J. Rode, M. Novak, B.M. Friedrich: in preparation

[[2](https://iopscience.iop.org/article/10.1209/0295-5075/ac6620)] Andrea Auconi et al 2022 EPL 138 12001
