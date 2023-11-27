import numpy as np

import multiprocessing
from functools import partial
from .utils import count_word_ocurrences, count_subword_ocurrences, count_transitions_ocurrences

class Counter:
    
    @classmethod
    def fit(
        cls,
        vlmc,
        X,
        njobs
    ):
        cls.X=X
        cls.njobs=njobs
        cls.vlmc=vlmc
        
        cls.__get_word_counts()
        cls.__get_transition_probabilities()
        
        
        
    @classmethod
    def __get_word_counts(cls):
        cls.__get_leaves_counts()
        cls.vlmc.tree = cls.__update_tree_counts(parent=cls.vlmc.tree)
     
    @classmethod
    def __get_leaves_counts(cls):
        
        leaves = [
            l.word for l in cls.vlmc.get_leaves()
        ]

        count_fn = partial(
            count_subword_ocurrences, 
            ''.join([str(x) for x in cls.X]),
            [str(s) for s in cls.vlmc.vocabulary]
        )
        
        with multiprocessing.Pool(cls.njobs) as p:
            counts=p.map(count_fn, leaves)


        cls.__leaves_counts = {
            l: c for l, c in zip(leaves, counts)
        }
    
    @classmethod
    def __update_tree_counts(cls,parent):
            
        if parent.children is not None:
            parent_counts = 0
            for w in parent.children.values():
                w = cls.__update_tree_counts(parent=w)
                parent_counts += w.n_ocurrences
        else:
            parent.n_ocurrences = cls.__leaves_counts[parent.word]
            return parent

        parent.n_ocurrences=parent_counts
        return parent
    
        
    @classmethod
    def __get_transition_probabilities(cls):
        
        def __get_node_probabilities(parent):
            
            if parent.children is not None:
                for w in parent.children.values():
                    __get_node_probabilities(parent=w)
                    
            else:
                parent.transition_counts = cls.__transitions_dict[parent.word]
                
                parent.transition_probabilities = {
                    s: c/parent.n_ocurrences for s, c in parent.transition_counts.items()
                }
                
                return
                
            parent.transition_counts = cls.__transitions_dict[parent.word]

            parent.transition_probabilities = {
                s: c/parent.n_ocurrences for s, c in parent.transition_counts.items()
            }
            
            return
        
        transitions_dict = {
            'root':{
                c.word: c.n_ocurrences for c in cls.vlmc.tree.children.values()
            }
        }
    
        words = [
            node.word for node in cls.vlmc.get_all_nodes() if node.word !='root'
        ]
        
        count_transitions_fn = partial(
            count_transitions_ocurrences, 
            ''.join([str(x) for x in cls.X]),
            [str(s) for s in cls.vlmc.vocabulary]
        )
        
        with multiprocessing.Pool(cls.njobs) as p:
            transitions=p.map(count_transitions_fn, words)
        
        
        for t in transitions:
            transitions_dict.update(t)
            
        cls.__transitions_dict = transitions_dict
        
        __get_node_probabilities(cls.vlmc.tree)
            
            
                    
                
        