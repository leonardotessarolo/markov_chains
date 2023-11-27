# from .bic_solver import BICSolver
# from .context_algorithm_solver import ContextAlgorithmSolver
# from .bct_solver import BCTSolver
# from .counter import Counter

# from treelib import Node, Tree
import numpy as np
from .em_solver import EMSolver

class HMM:
    
    def __init__(
        self,
        n_hidden_states,
        observed_vocabulary=None
    ):
        
        self.n_hidden_states=n_hidden_states
        self.hidden_states=np.array([s for s in range(self.n_hidden_states)])
        self.observed_vocabulary=np.array(observed_vocabulary) if observed_vocabulary is not None else None
        
        
        
    
    def fit(
        self,
        y,
        thresh=0.0001
    ):
        EMSolver.fit(
            hmm=self,
            y=y,
            thresh=thresh
        )
            
