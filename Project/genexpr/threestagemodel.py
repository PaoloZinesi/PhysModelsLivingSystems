from genexpr.basemodel import GeneExpressionModel
import numpy as np
from scipy.special import loggamma
import mpmath as mp



class ThreeStageModel(GeneExpressionModel):
    """ Specialized class to simulate the three-stage gene expression model. """

    def __init__(self) -> None:
        super().__init__(n_vars=3, n_reactions=6)
        self.var_descr = ("m: number of mRNA \n"
                          "n: number of proteins\n"
                          "s: state of the promoter (0=inactive, 1=active) \n")
        self.reactions_descr = ("m -> m+1 (rate nu0) \n"
                                "m -> m-1 (rate d0) \n"
                                "n -> n+1 (rate nu1) \n"
                                "n -> n-1 (rate d1) \n"
                                "s -> s+1 (rate K0) \n"
                                "s -> s-1 (rate K1) \n")



    def compute_propensities(self, x: np.ndarray, t: float, args: tuple) -> np.ndarray:
        """
        Function to compute propensity rates in the Gillespie simulations of the three-stage model.
        
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
        
        m, n, s = x # (mRNA, proteins, state)
        nu0, d0, nu1, d1, K0, K1 = args

        a_vec = np.array([s*nu0,    # T(m+1,n,s|m,n,s) = transcription of DNA into mRNA
                          m*d0,     # T(m-1,n,s|m,n,s) = death of mRNA
                          m*nu1,    # T(m,n+1,s|m,n,s) = translation of mRNA into protein
                          n*d1,     # T(m,n-1,s|m,n,s) = death of protein
                          K0*(1-s), # T(m,n,s=1|m,n,s=0) = activation of promoter
                          K1*s      # T(m,n,s=0|m,n,s=1) = deactivation of promoter
                          ])
        return a_vec
    


    def compute_updates(self) -> np.ndarray:
        """
        Function to compute updates in the Gillespie simulations of the three-stage model.
        
        Inputs: None

        Outputs: 
            - v_vec [ndarray, shape=(n_reactions,n_vars,)]: array containing updates of the model. If the j-th
                    reaction has been selected by the Gillepsie algorithm, then the concentration x is updated
                    as x <- x + v_vec[j,:]
        """
        
        v_vec = np.array([[+1.0, 0.0, 0.0],  # m <- m+1 : transcription of DNA into mRNA
                          [-1.0, 0.0, 0.0],  # m <- m-1 : death of mRNA
                          [0.0, +1.0, 0.0],  # n <- n+1 : translation of mRNA into protein
                          [0.0, -1.0, 0.0],  # n <- n-1 : death of protein
                          [0.0, 0.0, +1.0],   # s <- s+1 : activation of promoter
                          [0.0, 0.0, -1.0]]   # s <- s-1 : deactivation of promoter
                          )
        return v_vec
    


    def analytical_stationary(self, n: np.ndarray, args: tuple, normalize: bool = False) -> np.ndarray:
        """
        Analytical (stationary) distribution of the number of proteins n.

        Inputs:
            - n [ndarray, shape=(Nn,)]: array containing protein concentrations (must be non-negative)
            - args [tuple, shape=(n_reactions,)]: parameters (mainly reaction rates) used to compute propensities
            - normalize [bool]: normalize probabilities to have sum = 1 over n. Default is False
            
        Outputs: 
            - P_n [ndarray, shape=(Nn,)]: array containing transient probabilities P_n(t)
        """

        assert isinstance(n, np.ndarray),\
            f"x must be an ndarray, but instead is {str(type(n))}"
        assert len(args)==self.n_reactions,\
            f"args must have length {self.n_reactions}, but instead has length {len(args)}"

        # handle non-positive n
        n = n[n>=0]

        # parameters
        nu0, d0, nu1, d1, K0, K1 = args
        a, b, _, k0, k1 = nu0/d1, nu1/d0, d0/d1, K0/d1, K1/d1
        phi = np.sqrt((a+k0+k1)**2 - 4*a*k0)
        alpha = 0.5*(a+k0+k1+phi)
        beta = 0.5*(a+k0+k1-phi)

        # analytical formula
        logP_n = loggamma(alpha+n) + loggamma(beta+n) + loggamma(k0+k1) \
                - loggamma(n+1) - loggamma(alpha) - loggamma(beta) - loggamma(k0+k1+n) \
                + n*np.log(b/(1+b)) + alpha*np.log(1/(1+b)) \
                + np.array([float(mp.log(mp.hyp2f1(alpha+n_, k0+k1-beta, k0+k1+n_, b/(1+b)))) for n_ in n])
        P_n = np.exp(logP_n)
        
        if normalize:
            P_n /= np.sum(P_n)

        return P_n.squeeze()