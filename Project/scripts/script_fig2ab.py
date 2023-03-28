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
os.system("mkdir -p ../results/fig2ab")
sys.path.append('../')

import genexpr
from genexpr.twostagemodel import TwoStageModel
from genexpr.utils import KL_div

# %%
# simulation parameters
N_iters_max = [int(1e3), int(1e4), int(1e5)]
model = TwoStageModel()
t0 = 0.0
tau_slices = np.array([0.01, 0.2, 0.8, 2, 10])
x0s_args_reduced = [
    (np.array([0,0]), 10.0, 20.0, 2.5, 5e-4),
    (np.array([0,50]), 10.0, 0.5, 100.0, 5e-4)
    ]
# x0, gamma, a, b, d1 <---- to be transformed into real args

file_it = 0

# %%
# run for different combination of parameters
for N_iter_max_ in N_iters_max:
    for x0_arg_red_ in x0s_args_reduced:
        
        # simulation arguments
        x0_, gamma_, a_, b_, d1_ = x0_arg_red_
        args = (a_*d1_, gamma_*d1_, b_*gamma_*d1_, d1_)
        tslices = tau_slices / d1_


        # start simulation
        (_, n_hist) = model.Gillespie_simulation_transient(x0_, t0, tslices, Nsim=N_iter_max_, args=args)


        res = {"N_iters":N_iter_max_,
                "model":model,
                "x0":x0_,
                "t0":t0,
                "tau_slices":tau_slices,
                "tslices":tslices,
                "args_gamma_a_b_d1":(gamma_, a_, b_, d1_),
                "args":args,
                "n_hist":n_hist}
        

        file_name = f"fig2ab_fileit{file_it:03}.pkl"
        with open("../results/fig2ab/"+file_name, 'wb') as file:
            pickle.dump(res, file, protocol=pickle.HIGHEST_PROTOCOL)
        file_it += 1

        # update result
        nowtime = datetime.now(tz=pytz.timezone("Europe/Rome")).strftime("%d/%m/%Y %H:%M:%S")
        os.system(f"echo Finished computation of {file_name} at {nowtime} >> events.log")


# %%
