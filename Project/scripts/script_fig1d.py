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
os.system("mkdir -p ../results/fig1d")
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
gammas = np.logspace(-2, 2, num=15, base=10)
args_reduced = [
    (20.0, 2.5, 5e-4),
    (0.5, 100.0, 5e-4)
    ]
# a, b, d1 <---- to be transformed into real args


file_it = 0

# %%
# run for different combination of parameters
for N_iter_max_ in N_iters_max:
    for arg_red_ in args_reduced:
        for gamma_ in gammas:
        
            # simulation arguments
            a_, b_, d1_ = arg_red_
            args = (a_*d1_, gamma_*d1_, b_*gamma_*d1_, d1_)
            tmax_ = tau_max / d1_


            # start simulation
            (_, n_hist), _ = model.Gillespie_simulation(x0, t0, tmax_, Nsim=N_iter_max_, args=args, dt=0.01/d1_, keep_transient=False)
            h_, n_, _ = plt.hist(n_hist, bins=np.arange(start=-0.5, stop=max(n_hist)+0.5, step=1), density=True)
            n0 = np.arange(len(h_)+30)
            fn_analytical = model.analytical_stationary(n0, args, normalize=True)
            plt.clf()


            res = {"N_iters":N_iter_max_,
                    "model":model,
                    "x0":x0,
                    "t0":t0,
                    "tau_max":tau_max,
                    "tmax":tmax_,
                    "args_gamma_a_b_d1":[gamma_]+list(arg_red_),
                    "args":args,
                    "D_KL":KL_div(h_, fn_analytical[:len(h_)])}
            

            file_name = f"fig1d_fileit{file_it:03}.pkl"
            with open("../results/fig1d/"+file_name, 'wb') as file:
                pickle.dump(res, file, protocol=pickle.HIGHEST_PROTOCOL)
            file_it += 1

        # update result
        nowtime = datetime.now(tz=pytz.timezone("Europe/Rome")).strftime("%d/%m/%Y %H:%M:%S")
        os.system(f"echo Finished computation of {file_name} at {nowtime} >> events.log")


# %%
