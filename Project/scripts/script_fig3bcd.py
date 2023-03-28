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
os.system("mkdir -p ../results/fig3bcd")
sys.path.append('../')

import genexpr
from genexpr.threestagemodel import ThreeStageModel
from genexpr.utils import KL_div

# %%
# simulation parameters
N_iters_max = [int(1e5), int(1e6)]
model = ThreeStageModel()
x0 = np.array([0,0,0])
t0 = 0.0
tau_max = 10
args_reduced = [
    (1.0, 40.0, 2.0, 5e-4, 6.0, 2.0),
    (10.0, 40.0, 2.0, 5e-4, 6.0, 2.0),
    (100.0, 40.0, 2.0, 5e-4, 6.0, 2.0),
    (1.0, 1.0, 40.0, 5e-4, 6.0, 2.0),
    (10.0, 1.0, 40.0, 5e-4, 6.0, 2.0),
    (100.0, 1.0, 40.0, 5e-4, 6.0, 2.0),
    (1.0, 4.0, 10.0, 5e-4, 0.6, 0.2),
    (10.0, 4.0, 10.0, 5e-4, 0.6, 0.2),
    (100.0, 4.0, 10.0, 5e-4, 0.6, 0.2),
    ]
# gamma, a, b, d1, k0, k1 <---- to be transformed into real args

file_it = 0

# %%
# run for different combination of parameters
for N_iter_max_ in N_iters_max:
    for arg_red_ in args_reduced:
        
        
        # simulation arguments
        gamma, a, b, d1, k0, k1 = arg_red_
        args = (a*d1, gamma*d1, b*gamma*d1, d1, k0*d1, k1*d1)
        tmax_ = tau_max / d1


        # start simulation
        (_, n_hist, _), _ = model.Gillespie_simulation(x0, t0, tmax_, Nsim=N_iter_max_, args=args, dt=0.01/d1, keep_transient=False)


        res = {"N_iters":N_iter_max_,
                "model":model,
                "x0":x0,
                "t0":t0,
                "tau_max":tau_max,
                "tmax":tmax_,
                "args_gamma_a_b_d1_k0_k1":arg_red_,
                "args":args,
                "n_hist":n_hist}
        

        file_name = f"fig3bcd_fileit{file_it:03}.pkl"
        with open("../results/fig3bcd/"+file_name, 'wb') as file:
            pickle.dump(res, file, protocol=pickle.HIGHEST_PROTOCOL)
        file_it += 1

        # update result
        nowtime = datetime.now(tz=pytz.timezone("Europe/Rome")).strftime("%d/%m/%Y %H:%M:%S")
        os.system(f"echo Finished computation of {file_name} at {nowtime} >> events.log")


# %%
