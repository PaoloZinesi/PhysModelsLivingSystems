from genexpr.basemodel import GeneExpressionModel
import numpy as np
from scipy.special import loggamma, hyp2f1



class TwoStageModel(GeneExpressionModel):
    """ Base class to build specialized gene expression models. """

    def __init__(self) -> None:
        super().__init__(n_vars=2, n_reactions=4)
        self.var_descr = ("m: number of mRNA \n"
                          "n: number of proteins\n")
        self.reactions_descr = ("m -> m+1 (rate nu0) \n"
                                "m -> m-1 (rate d0) \n"
                                "n -> n+1 (rate nu1) \n"
                                "n -> n-1 (rate d1) \n")



    def compute_propensities(self, x: np.ndarray, t: float, args: tuple) -> np.ndarray:
        """
        Function to compute propensity rates in the Gillespie simulations of the two-stage model.
        
        Inputs:
            - x [ndarray, shape=(n_vars,)]: array containing concentration values
            - t [float]: time
            - args [tuple, shape=(n_reactions,)]: parameters (mainly reaction rates) used to compute propensities

        Outputs: 
            - a_vec [ndarray, shape=(n_reactions,)]: array containing propensities
        """

        assert isinstance(x, np.ndarray),\
            f"x must be an ndarray, but instead is {str(type(x))}"
        assert len(x)==self.n_vars,\
            f"x must have length {self.n_vars}, but instead has length {len(x)}"
        assert len(args)==self.n_reactions,\
            f"args must have length {self.n_reactions}, but instead has length {len(args)}"
        
        m, n = x # (mRNA, proteins)
        nu0, d0, nu1, d1 = args

        a_vec = np.array([nu0,      # T(m+1,n|m,n) = transcription of DNA into mRNA
                          m*d0,     # T(m-1,n|m,n) = death of mRNA
                          m*nu1,    # T(m,n+1|m,n) = translation of mRNA into protein
                          n*d1      # T(m,n-1|m,n) = death of protein
                          ])
        return a_vec
    


    def compute_updates(self) -> np.ndarray:
        """
        Function to compute updates in the Gillespie simulations of the two-stage model.
        
        Inputs: None

        Outputs: 
            - v_vec [ndarray, shape=(n_reactions,n_vars,)]: array containing updates of the model. If the j-th
                    reaction has been selected by the Gillepsie algorithm, then the concentration x is updated
                    as x <- x + v_vec[j,:]
        """
        
        v_vec = np.array([[+1.0, 0.0],  # m <- m+1 : transcription of DNA into mRNA
                          [-1.0, 0.0],  # m <- m-1 : death of mRNA
                          [0.0, +1.0],  # n <- n+1 :  translation of mRNA into protein
                          [0.0, -1.0]]  # n <- n-1 :  death of protein
                          )
        return v_vec



    def mean_field_prediction(self, x0: np.ndarray, t: np.ndarray, args: tuple) -> np.ndarray:
        """
        Mean field predictions of the two-stage model evaluated at times t with initial concentrations x0 = x(t=0).

        Inputs:
            - x0 [ndarray, shape=(n_vars,)]: array containing concentration values at time t=0
            - t [ndarray, shape=(Nt,)]: array of times where to evaluate the model
            - args [tuple, shape=(n_reactions,)]: parameters (mainly reaction rates) used to compute propensities
            
        Outputs: 
            - x_t [ndarray, shape=(n_vars, Nt,)]: array containing concentration values at time t of variable j in x_t[j,:]
        """

        assert isinstance(x0, np.ndarray),\
            f"x must be an ndarray, but instead is {str(type(x0))}"
        assert len(x0)==self.n_vars,\
            f"x must have length {self.n_vars}, but instead has length {len(x0)}"
        assert len(args)==self.n_reactions,\
            f"args must have length {self.n_reactions}, but instead has length {len(args)}"

        if not isinstance(t, np.ndarray):
            t = np.array(t)


        m0, n0 = x0 # (mRNA, proteins)
        nu0, d0, nu1, d1 = args

        # utility F(t)
        if d0 != d1:
            F_t = (np.exp(-d0*t)-np.exp(-d1*t))/(d1-d0)
        else:
            F_t = t * np.exp(-d1*t)

        m_t = nu0/d0 + (m0 - nu0/d0)*np.exp(-d0*t)
        n_t = (nu0*nu1)/(d0*d1) + (n0 - (nu0*nu1)/(d0*d1))*np.exp(-d1*t) + nu1*(m0 - nu0/d0)*F_t

        return np.array([m_t, n_t])
    


    def analytical_transient(self, n: np.ndarray, t: float, args: tuple, normalize: bool = False) -> np.ndarray:
        """
        Analytical (transient) distribution at time t (notice that in the paper tau = d1*t) of the number of proteins n.

        Inputs:
            - n [ndarray, shape=(Nn,)]: array containing protein concentrations (must be non-negative)
            - t [float]: time where to evaluate the function
            - args [tuple, shape=(n_reactions,)]: parameters (mainly reaction rates) used to compute propensities
            - normalize [bool]: normalize probabilities to have sum = 1 over n. Default is False
            
        Outputs: 
            - P_n_t [ndarray, shape=(Nn,)]: array containing transient probabilities P_n(t)
        """

        assert isinstance(n, np.ndarray),\
            f"x must be an ndarray, but instead is {str(type(n))}"
        assert len(args)==self.n_reactions,\
            f"args must have length {self.n_reactions}, but instead has length {len(args)}"

        # handle non-positive n
        n = n[n>=0]

        # parameters
        nu0, d0, nu1, d1 = args
        a, b, gamma, tau = nu0/d1, nu1/d0, d0/d1, d1*t

        # analytical formula
        P_n_t = np.exp(loggamma(a+n)-loggamma(n+1)-loggamma(a)) * np.power(b/(1+b), n) * ((1+b*np.exp(-tau))/(1+b))**a \
                * hyp2f1(-n, -a, 1-a-n, (1+b)/(np.exp(tau)+b))
        
        if normalize:
            P_n_t /= np.sum(P_n_t)

        return P_n_t.squeeze()