import numpy as np

import multiprocessing
from functools import partial
from .utils import count_word_ocurrences, count_subword_ocurrences

class ContextAlgorithmSolver:
    
    @classmethod
    def fit(
        cls,
        vlmc,
        X,
        alpha=1/16,
        beta=1
    ):
        def __get_word_delta(parent):
            
            if parent.children is not None:
                for w in parent.children.values():
                    __get_word_delta(parent=w)
                    
            else:
                parent.delta=None
                return

            L_parent = (parent.n_ocurrences/cls.n)*np.sum([
                (c/parent.n_ocurrences)*np.log(c/parent.n_ocurrences) for b, c in parent.transition_counts.items() if c>0
            ])
            
            L_children = np.sum([
                (child.n_ocurrences/cls.n)*np.sum([
                    (c/child.n_ocurrences)*np.log(c/child.n_ocurrences) for b, c in child.transition_counts.items() if c>0
                ]) for child in parent.children.values() if child.n_ocurrences>0
            ])
            
            parent.delta = L_children-L_parent
            
            return
        
        def __trim_tree(parent):
            
            if parent.children is not None:
                keep_children_children = []
                for w in parent.children.values():
                    keep_children_children.append(__trim_tree(parent=w))
                keep_children=np.any(keep_children_children)
            else:
                return False
            
            if keep_children:
                return True
            if not keep_children:
                word_len_admissible = True if len(parent.word) <= cls.beta*np.log(cls.n) else False
                child_count_admissible = True if min([
                    c_wa for c_wa in parent.transition_counts.values()
                ]) > 2*cls.alpha*cls.n/np.log(cls.n) else False
                
                # child_count_admissible=True
                delta_admissible = True if parent.delta > np.log(cls.n)/cls.n else False
                # import pdb;pdb.set_trace()
                # if not (word_len_admissible and child_count_admissible and delta_admissible):
                if not (delta_admissible and word_len_admissible):
                    parent.children=None
                    parent.is_leaf=True
                    return False
                else:
                    return True
                
        
        cls.vlmc=vlmc
        cls.X=X
        cls.n=vlmc.tree.n_ocurrences
        cls.alpha=alpha
        cls.beta=beta
        
        __get_word_delta(parent=cls.vlmc.tree)
        __trim_tree(parent=cls.vlmc.tree)
        
        
        
                
        