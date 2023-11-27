import numpy as np

import multiprocessing
from functools import partial
from .utils import count_word_ocurrences, count_subword_ocurrences

class BCTSolver:
    
    @classmethod
    def fit(
        cls,
        vlmc,
        X,
        beta=0.5
    ):
        def __get_word_maximal_probs(parent):
            
            # If not at leaves, recursively move deeper in tree towards leaves
            if parent.children is not None:
                for w in parent.children.values():
                    __get_word_maximal_probs(parent=w)
            # If at leaves        
            else:
                if np.any(
                    a=[
                        c==0 for c in parent.transition_counts.values()
                    ]
                ):
                    P_e = 0
                else:
                    
                    prod_vectors = np.array([
                        np.array([
                            i-0.5 for i in range(1,c+1)
                        ]) for c in parent.transition_counts.values()
                    ])
                    
                    div_vector = np.array([
                        i-1+(cls.vocabulary_size/2)for i in range(1, parent.n_ocurrences+1)
                    ])
                    max_size = len(div_vector)
                    
                    prod_vectors = np.array([
                        np.concatenate(
                            [v, np.ones(max_size-len(v))]
                        ) for v in prod_vectors
                        
                    ])
                    
                    
                    P_e=0
                    for i in range(max_size):
                        P_e += np.sum(
                            np.log(prod_vectors[:,i])
                        ) - np.log(div_vector[i])
                        
                    
                parent.P_e=P_e
                parent.P_m=P_e
                parent.is_child_P_m=None
                    
                return
            
            if np.any(
                a=[
                    c==0 for c in parent.transition_counts.values()
                ]
            ):
                P_e = 0
            else:
                prod_vectors = np.array([
                    np.array([
                        i-0.5 for i in range(1,c+1)
                    ]) for c in parent.transition_counts.values()
                ])

                div_vector = np.array([
                    i-1+(cls.vocabulary_size/2)for i in range(1, parent.n_ocurrences+1)
                ])
                max_size = len(div_vector)

                prod_vectors = np.array([
                    np.concatenate(
                        [v, np.ones(max_size-len(v))]
                    ) for v in prod_vectors

                ])


                P_e=0
                for i in range(max_size):
                    P_e += np.sum(
                        np.log(prod_vectors[:,i])
                    ) - np.log(div_vector[i])
                


            parent.P_e=P_e
            
            P_m_parent = np.log(cls.beta)+parent.P_e
            P_m_children = np.log(1-cls.beta) + np.sum(
                [
                    child.P_m for child in parent.children.values()
                ]
            )
            parent.P_m = max(
                P_m_parent,
                P_m_children
            )
            parent.is_child_P_m = True if P_m_children>P_m_parent else False
            
            return
        
        def __trim_tree(parent):
            
            if parent.children is not None:
                for w in parent.children.values():
                    __trim_tree(parent=w)
                    
            else:
                return
            
            if not parent.is_child_P_m:
                # import pdb;pdb.set_trace()
                parent.children=None
                parent.is_leaf=True
                pass
            
            return
        
        cls.vlmc=vlmc
        cls.X=X
        cls.vocabulary_size=len(cls.vlmc.vocabulary)
        cls.beta=beta
        cls.n=len(X)
        
        __get_word_maximal_probs(parent=cls.vlmc.tree)
        __trim_tree(parent=cls.vlmc.tree)
        
        
        
                
        