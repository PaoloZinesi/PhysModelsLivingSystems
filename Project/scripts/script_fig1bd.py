# %% 
import os, sys
from datetime import datetime
import pytz
import pickle

import numpy as np
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt

os.system(f"rm -f events.log")
os.system("mkdir -p ../results/fig1bd")
sys.path.append('../')

import genexpr
from genexpr.twostagemodel import TwoStageModel
from genexpr.utils import KL_div

# %%
# simulation parameters
N_iters_max = [int(1e5), int(1e6)]
model = TwoStageModel()
x0 = np.array([0,0])
t0 = 0.0
tau_max = 10
args_reduced = [
    (1.0, 20.0, 2.5, 5e-4),
    (10.0, 20.0, 2.5, 5e-4),
    (100.0, 20.0, 2.5, 5e-4),
    (1.0, 0.5, 100.0, 5e-4),
    (10.0, 0.5, 100.0, 5e-4),
    (100.0, 0.5, 100.0, 5e-4)
    ]
# gamma, a, b, d1 <---- to be transformed into real args

file_it = 0

# %%
# run for different combination of parameters
for N_iter_max_ in N_iters_max:
    for arg_red_ in args_reduced:
        
        
        # simulation arguments
        gamma, a, b, d1 = arg_red_
        args_ = (a*d1, gamma*d1, b*gamma*d1, d1)
        tmax_ = tau_max / d1


        # start simulation
        (_, n_hist), _ = model.Gillespie_simulation(x0, t0, tmax_, Nsim=N_iter_max_, args=args_, dt=0.01/d1, keep_transient=False)


        res = {"N_iters":N_iter_max_,
                "model":model,
                "x0":x0,
                "t0":t0,
                "tau_max":tau_max,
                "tmax":tmax_,
                "args_gamma_a_b_d1":arg_red_,
                "args":args_,
                "n_hist":n_hist}
        

        file_name = f"fig1bd_fileit{file_it:03}.pkl"
        with open("../results/fig1bd/"+file_name, 'wb') as file:
            pickle.dump(res, file, protocol=pickle.HIGHEST_PROTOCOL)
        file_it += 1

        # update result
        nowtime = datetime.now(tz=pytz.timezone("Europe/Rome")).strftime("%d/%m/%Y %H:%M:%S")
        os.system(f"echo Finished computation of {file_name} at {nowtime} >> events.log")


# %%
