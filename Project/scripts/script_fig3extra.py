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
os.system("mkdir -p ../results/fig3extra")
sys.path.append('../')

import genexpr
from genexpr.threestagemodel import ThreeStageModel
from genexpr.utils import KL_div

# %%
# simulation parameters
N_iters_max = [int(1e5), int(1e6)]
model = ThreeStageModel()
t0 = 0.0
tau_stat = 10.0
gammas = np.logspace(-2, 2, num=15, base=10)
x0s_args_reduced = [
    (np.array([0,0,0]), 40.0, 2.0, 5e-4, 6.0, 2.0),
    (np.array([0,0,0]), 1.0, 40.0, 5e-4, 6.0, 2.0),
    (np.array([0,0,0]), 4.0, 10.0, 5e-4, 0.6, 0.2)
    ]
# x0, a, b, d1, k0, k1 <---- to be transformed into real args


file_it = 0

# %%
# run for different combination of parameters
for N_iter_max_ in N_iters_max:
    for x0_arg_red_ in x0s_args_reduced:
        for gamma_ in gammas:
        
            # simulation arguments
            x0_, a_, b_, d1_, k0_, k1_ = x0_arg_red_
            args = (a_*d1_, gamma_*d1_, b_*gamma_*d1_, d1_, k0_*d1_, k1_*d1_)
            tstat_ = tau_stat / d1_


            # start simulation
            (_, n_hist, _), _ = model.Gillespie_simulation(x0_, t0, tstat_, Nsim=N_iter_max_, args=args, dt=0.01/d1_, keep_transient=False)
            h_, n_, _ = plt.hist(n_hist, bins=np.arange(start=-0.5, stop=max(n_hist)+0.5, step=1), density=True)
            n0 = np.arange(len(h_)+30)
            fn_analytical = model.analytical_stationary(n0, args, normalize=True)
            plt.clf()

            # packe results
            res = {"N_iters":N_iter_max_,
                    "model":model,
                    "x0":x0_,
                    "t0":t0,
                    "tau_stat":tau_stat,
                    "tstat_":tstat_,
                    "args_gamma_a_b_d1_k0_k1":[gamma_, a_, b_, d1_, k0_, k1_],
                    "args":args,
                    "D_KL":KL_div(h_, fn_analytical[:len(h_)])}
        

            file_name = f"fig3extra_fileit{file_it:03}.pkl"
            with open("../results/fig3extra/"+file_name, 'wb') as file:
                pickle.dump(res, file, protocol=pickle.HIGHEST_PROTOCOL)
            file_it += 1

        # update result
        nowtime = datetime.now(tz=pytz.timezone("Europe/Rome")).strftime("%d/%m/%Y %H:%M:%S")
        os.system(f"echo Finished computation of {file_name} at {nowtime} >> events.log")


# %%
