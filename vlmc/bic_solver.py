import numpy as np

import multiprocessing
from functools import partial
from .utils import count_word_ocurrences, count_subword_ocurrences

class BICSolver:
    
    @classmethod
    def fit(
        cls,
        vlmc,
        X
    ):
        def __get_word_V_chi(parent):
            """
                Recursive method to estimate values for V and indicator function chi. These values will be used
                to estimate tree with minimum BIC. Due to orders of magnitude involved, which may be below computational
                representation for some nodes depending on tree max depth, logarithms are used for V's.
            """
            
            # If not at leaves, recursively move deeper in tree towards leaves
            if parent.children is not None:
                for w in parent.children.values():
                    __get_word_V_chi(parent=w)
            # If at leaves    
            else:
                # V for leaves is calculated from its own transition counts and probabilities.
                parent.V = np.sum([
                    c*np.log(c/parent.n_ocurrences) for c in parent.transition_counts.values() if c >=1
                ]) + cls.adj_factor
                
                # Chi for leaves is zero
                parent.chi = 0
                return
            
            # Calculate candidate for V in parent
            L_parent = np.sum([
                c*np.log(c/parent.n_ocurrences) for c in parent.transition_counts.values() if c >=1
            ]) + cls.adj_factor
            
            # Calculate product (log-sum) of V's in children 
            V_children = np.sum([
                child.V for child in parent.children.values()
            ])
            
            # V in parent is the maximum between candidate value at parent node and
            # product (log-sum) in children
            parent.V = max(
                L_parent,
                V_children
            )
            
            # Define chi according to value which was maximum for obtaining parent V
            parent.chi = 1 if V_children > L_parent else 0
            
            return
        
        def __trim_tree(parent):
            """
                Method that performs trimming according to rule for chi: a node's children will be kept if all children's values for
                chi are zero, and also node and all up to root have chi=1. Movement in tree is done recursively.
            """
            
            #If node has chi=1, move towards leaves
            if parent.chi==1:
                for w in parent.children.values():
                    __trim_tree(parent=w)
            # If node has chi=0, trim all its descendants.
            else:
                parent.children=None
                parent.is_leaf=True
                return
            
            return
        
        cls.vlmc=vlmc
        cls.X=X
        cls.n=len(X)
        cls.adj_factor = np.log(cls.n)*(-(len(cls.vlmc.vocabulary)-1)/2)
        
        # Get values of V and indicator function chi
        __get_word_V_chi(parent=cls.vlmc.tree)
        
        # Trim tree according to rules regarding indicator function chi
        __trim_tree(parent=cls.vlmc.tree)
        
        
        
                
        