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
            
            if parent.children is not None:
                for w in parent.children.values():
                    __get_word_V_chi(parent=w)
                    
            else:
                # import pdb;pdb.set_trace()
                parent.V = np.sum([
                    c*np.log(c/parent.n_ocurrences) for c in parent.transition_counts.values() if c >=1
                ]) + cls.adj_factor
                
                parent.chi = 0
                return
            
            L_parent = np.sum([
                c*np.log(c/parent.n_ocurrences) for c in parent.transition_counts.values() if c >=1
            ]) + cls.adj_factor
            
            V_children = np.sum([
                child.V for child in parent.children.values()
            ])

            parent.V = max(
                L_parent,
                V_children
            )
            
            parent.chi = 1 if V_children > L_parent else 0
            
            return
        
        def __trim_tree(parent):
            
            if parent.chi==1:
                for w in parent.children.values():
                    __trim_tree(parent=w)
                    
            else:
                parent.children=None
                parent.is_leaf=True
                return
            
            return
        
        cls.vlmc=vlmc
        cls.X=X
        cls.n=len(X)
        cls.adj_factor = np.log(cls.n)*(-(len(cls.vlmc.vocabulary)-1)/2)
        
        __get_word_V_chi(parent=cls.vlmc.tree)
        __trim_tree(parent=cls.vlmc.tree)
        
        
        
                
        