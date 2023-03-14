import numpy as np
from numpy import random



class GeneExpressionModel:
    """ Base class to build specialized gene expression models. """

    def __init__(self, n_vars: int, n_reactions: int) -> None:

        assert n_vars > 0, f"Number of variables greater than 0 expected, got: {n_vars}"
        assert n_reactions > 0, f"Number of reactions greater than 0 expected, got: {n_reactions}"

        self.n_vars = n_vars
        self.n_reactions = n_reactions
        self.var_descr = ""
        self.reactions_descr = ""
        self.updates = self.compute_updates()
        
    

    def compute_propensities(self, x: np.ndarray, t: float, args: tuple) -> np.ndarray:
        """
        Function to compute propensity rates in the Gillespie simulations. Child classes must override this function.
        
        Inputs:
            - x [ndarray, shape=(n_vars,)]: array containing concentration values
            - t [float]: time
            - args [tuple, shape=(n_reactions,)]: parameters (mainly reaction rates) used to compute propensities

        Outputs: 
            - a_vec [ndarray, shape=(n_reactions,)]: array containing propensities

        """

        return np.array([1.0]*self.n_reactions)
    


    def compute_updates(self) -> np.ndarray:
        """
        Function to compute updates in the Gillespie simulations. Child classes must override this function.
        
        Inputs: None

        Outputs: 
            - v_vec [ndarray, shape=(n_reactions,n_vars,)]: array containing updates of the model. If the j-th
                    reaction has been selected by the Gillepsie algorithm, then the concentration x is updated
                    as x <- x + v_vec[j,:]
        """

        return np.zeros((self.n_reactions, self.n_vars))



    def Gillespie_iteration(self, x: np.ndarray, t: float, args: tuple) -> tuple[np.ndarray, float]:
        """
        Function to compute an iteration of Gillespie simulations.
        
        Inputs:
            - x [ndarray, shape=(n_vars,)]: array containing initial concentration values
            - t [float]: starting time 
            - args [tuple, shape=(n_reactions,)]: parameters (mainly reaction rates) used to compute propensities
            
        Outputs: 
            - xf [ndarray, shape=(n_vars,)]: array containing final concentration values
            - tf [float]: final time (after reaction happened)
        """

        assert isinstance(x, np.ndarray),\
            f"x must be an ndarray, but instead is {type(x)}"
        assert len(x)==self.n_vars,\
            f"x must have length {self.n_vars}, but instead has length {len(x)}"
        assert len(args)==self.n_reactions,\
            f"args must have length {self.n_reactions}, but instead has length {len(args)}"
        

        # compute propensities a_vec (which depends on x and t!)
        a_vec = self.compute_propensities(x, t, args)

        # total propensity (rate)
        a0 = np.sum(a_vec)

        # dt of next reaction
        dt = (1./a0)*np.log(1/random.random_sample())

        # choice of next reaction index (j)
        j_next = random.choice(np.arange(self.n_reactions), p=a_vec/a0)

        # perform updates
        xf = x + self.updates[j_next,:]
        tf = t + dt

        return xf, tf
    


    def Gillespie_simulation(self, x0: np.ndarray, t0: float, tstat: float, Nsim: int,
                             args: tuple, dt: float = -1.0, keep_transient: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Function to compute a full Gillespie simulation.
        The algorithm will reach at first the time 'tstat' and then it will start to sample from the converged distribution at each dt.
        By default only the converged distribution is considered, but it is also possible to keep track of the transient part
        by setting 'keep_transient' = True.
        
        Inputs:
            - x0 [ndarray, shape=(n_vars,)]: array containing initial concentration values at time t=0
            - t0 [float]: starting time
            - tstat [float]: simulation time after which the solution has converged
            - Nsim [int]: number of reactions to simulate after distribution has converged
            - args [tuple, shape=(n_reactions,)]: parameters (mainly reaction rates) used to compute propensities
            - dt [float]: time between successive samples of the distribution. If negative, samples are taken only at reaction times.
            - keep_transient [bool]: wheteher to keep track of transient concentrations. Default is False
            
        Outputs: 
            - xf [ndarray, shape=(n_vars,Nsim,) or (n_vars,Nsim+dN,)]: array containing final concentration values
            - tf [ndarray, shape=(Nsim,) or (Nsim+dN,)]: array containing reaction times
        """

        assert isinstance(x0, np.ndarray),\
            f"x must be an ndarray, but instead is {type(x0)}"
        assert len(x0)==self.n_vars,\
            f"x must have length {self.n_vars}, but instead has length {len(x0)}"
        assert len(args)==self.n_reactions,\
            f"args must have length {self.n_reactions}, but instead has length {len(args)}"
        assert t0<tstat,\
            f"starting time t0 must be smaller than stationary time"
        
        x, t = x0.copy(), t0

        # transient simulation (dt here is not used!!)
        if keep_transient:
            xf, tf = [x0], [t0]

            while t < tstat:
                x, t = self.Gillespie_iteration(x, t, args)
                xf.append(x)
                tf.append(t)
        else:
            while t < tstat:
                x, t = self.Gillespie_iteration(x, t, args)

            xf, tf = [x], [t]

        # stationary simulation (dt is used only here!)
        # TO BE IMPROVED!!!
        t_ref = t
        i = 0
        while i < Nsim:
            
            x, t = self.Gillespie_iteration(x, t, args)

            if dt<=0:
                xf.append(x)
                tf.append(t)
                i += 1
            else:
                Nsamp = np.floor((t-t_ref)/dt).astype(int)
                for j in range(Nsamp):
                    xf.append(x)
                    tf.append(t_ref + j*dt)

                t_ref += Nsamp*dt
                i += Nsamp


        return np.asarray(xf).T, np.asarray(tf)
    


    def Gillespie_simulation_transient(self, x0: np.ndarray, t0: float, tslice: np.ndarray, Nsim: int,
                                       args: tuple) -> tuple[np.ndarray, np.ndarray]:
        """
        Function to compute a full Gillespie simulation focusing on the distribution at intermediate times.
        The algorithm will run the Gillespie iterations until time 'tslice[-1]' for 'Nsim' times.
        
        Inputs:
            - x0 [ndarray, shape=(n_vars,)]: array containing initial concentration values at time t=0
            - t0 [float]: starting time
            - tslice [ndarray, shape=(Nslices,)]: intermediate times to construct transient distributions
            - Nsim [int]: number of reactions to simulate
            - args [tuple, shape=(n_reactions,)]: parameters (mainly reaction rates) used to compute propensities
            
        Outputs: 
            - xf [ndarray, shape=(n_vars,Nslices,Nsim,)]: array containing intermediate concentration values
        """

        assert isinstance(x0, np.ndarray),\
            f"x must be an ndarray, but instead is {type(x0)}"
        assert len(x0)==self.n_vars,\
            f"x must have length {self.n_vars}, but instead has length {len(x0)}"
        assert len(args)==self.n_reactions,\
            f"args must have length {self.n_reactions}, but instead has length {len(args)}"

        tslice = np.sort(tslice)
        assert t0 < tslice[-1],\
            f"starting time t0 must be smaller than stationary time"
        
        xf = []

        # transient simulation
        for i in range(Nsim):
            x, t = x0.copy(), t0

            x_sim, t_sim = [], []
            while t < tslice[-1]:
                x, t = self.Gillespie_iteration(x, t, args)
                x_sim.append(x)
                t_sim.append(t)

            # find concentrations at each slice
            x_sim = np.asarray(x_sim)
            t_sim = np.asarray(t_sim)
            slice_mask = np.argmax(t_sim.reshape(-1,1) > tslice.reshape(1,-1), axis=0) - 1 # Nslices indices
            xf.append(x_sim[slice_mask,:]) # appended array has shape = (Nslices,n_vars)

        
        # return distribution of slices ordered correctly
        return np.transpose(np.asarray(xf), (2,1,0))