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
            f"x must be an ndarray, but instead is {str(type(x))}"
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
    


    def Gillespie_simulation(self, x0: np.ndarray, t0: float, tmax: float, args: tuple, keep_track: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Function to compute an iteration of Gillespie simulations.
        
        Inputs:
            - x0 [ndarray, shape=(n_vars,)]: array containing initial concentration values at time t=0
            - t0 [float]: starting time
            - tmax [float]: maximum simulation time 
            - args [tuple, shape=(n_reactions,)]: parameters (mainly reaction rates) used to compute propensities
            - keep_track [bool]: wheteher to keep track of concentration at intermediate. Default is False
            
        Outputs: 
            - xf [ndarray, shape=(n_vars,) or (n_vars,Nt,)]: array containing final concentration values (eventually also at intermediate times)
            - tf [ndarray, shape=(1,) or (Nt,)]: final time or also intermediate times of simulation (depending on keep_track)
        """

        assert isinstance(x0, np.ndarray),\
            f"x must be an ndarray, but instead is {str(type(x0))}"
        assert len(x0)==self.n_vars,\
            f"x must have length {self.n_vars}, but instead has length {len(x0)}"
        assert len(args)==self.n_reactions,\
            f"args must have length {self.n_reactions}, but instead has length {len(args)}"
        assert t0<tmax,\
            f"starting time t0 must be smaller than total simulation time tmax"
        
        # Gillespie iterations
        x, t = x0.copy(), t0

        if keep_track:
            xf, tf = [x0], [t0]

            while t < tmax:
                x, t = self.Gillespie_iteration(x, t, args)
                xf.append(x)
                tf.append(t)

        else:
            while t < tmax:
                x, t = self.Gillespie_iteration(x, t, args)
            xf, tf = x, t

        return np.asarray(xf).squeeze().T, np.asarray(tf).squeeze()