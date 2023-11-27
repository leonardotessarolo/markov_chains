import numpy as np

import multiprocessing
from functools import partial
from .utils import count_word_ocurrences, count_subword_ocurrences, count_transitions_ocurrences, obtain_associated_context

class Counter:
    """
        Class that performs counting of symbol ocurrences in a given sample X. Developments are intensive on
        class methods and attributes, so instantiation of objects is not needed. Counts and transition probabilities
        associated to symbols are saved in tree object given as input. 
    """
    
    @classmethod
    def fit(
        cls,
        vlmc,
        X,
        njobs
    ):
        """
            High-level method for adjusting counter to sample X, according to given vlmc object.
        """
        # Saves class attributes
        cls.X=X
        cls.njobs=njobs
        cls.vlmc=vlmc
        
        # If admissibility criteria are required, trim tree so that depth is not greater than log(len(X))
        truncate_depth=int(
            np.floor(
                np.log(len(X))
            )
        )
        if cls.vlmc.make_admissible:
            if cls.vlmc.max_order > truncate_depth:
                cls.__truncate_tree(
                    parent=cls.vlmc.tree,
                    k=truncate_depth
                )
                
        # Counts symbols in given sample
        cls.__get_word_counts()
        
        # If admissibility criteria are required, trim non-appearing contexts
        if cls.vlmc.make_admissible:
            # Guarantee that all contexts have non-zero count
            cls.__trim_non_appearing_contexts(
                parent=cls.vlmc.tree
            )
            
            # Guarantee that all words in sample can be associated to context
            cls.__verify_word_association()
        
        # Obtains transition probabilities from word counts
        cls.__get_transition_probabilities()
        
        
            
        
    @classmethod
    def __truncate_tree(
        cls,
        parent,
        k
    ):
        """
            This method truncates tree to depth k. Movements across tree are done by recursion.
        """
        
        #If node has length of word smaller than k, move towards leaves
        if (len(parent.word) < k) or (parent.word == 'root'):
            for w in parent.children.values():
                cls.__truncate_tree(
                    parent=w,
                    k=k
                )
        # If node has length of word k, trim all its descendants.
        else:
            parent.children=None
            parent.is_leaf=True
            return

        return
    
    @classmethod
    def __trim_non_appearing_contexts(
        cls,
        parent
    ):
        """
            This method trims tree by removing contexts for which there are no ocurrences in sample X. 
            Movements across tree are done by recursion.
        """
        def __remove_non_appearing_contexts(
            parent
        ):
            """
                This method removes words that don't appear in sample, as well as all descendants.
            """
            # Move towards leaves
            if parent.children is not None:
                for w in parent.children.values():
                    # If child has appearances, go further down tree
                    if w.n_ocurrences > 0:
                        __remove_non_appearing_contexts(
                            parent=w,
                        )
                    # If child has no appearances, trim child
                    else:
                        parent.children = {
                            a: c for a, c in parent.children.items() if c.word != w.word
                        }
            # If at leaves
            else:
                return

            return
        
        
        def __make_tree_irreducible(
            parent
        ):
            """
                This trims redundancies as to make tree irreducible.
            """
        
            # If at internal node, recursively call method for node's children
            if parent.children is not None:
                for w in parent.children.values():
                    w = __make_tree_irreducible(parent=w)
            # If at leaves, return parent
            else:
                return parent

            # Trims children if parent has only one child
            if len(parent.children.values())==1:
                parent.children=None
                parent.is_leaf=True

            return parent

        # Trim contexts that don't appear
        __remove_non_appearing_contexts(parent=parent)
        
        # Remove redundancies as tom make tree irreducible
        __make_tree_irreducible(parent=parent)
        
        
    @classmethod
    def __verify_word_association(cls):
        
        # Obtains tree leaves
        leaves = [
            l.word for l in cls.vlmc.get_leaves()
        ]
        
        # Get tree max depth and update class object
        max_depth = max([
            len(l) for l in leaves
        ])
        cls.vlmc.max_order=max_depth
        
        # Iteratively window sample using length max_depth 
        windows = [
            ''.join(
                [str(x) for x in cls.X[i-max_depth: i]]
            ) for i in range(max_depth, len(cls.X))
        ]
        
        # Function that will be executed
        verify_fn = partial(
            obtain_associated_context,
            leaves
        )
        
        # Distributedly execute verification, obtaining associated context to windowed words
        with multiprocessing.Pool(cls.njobs) as p:
            associations=p.map(verify_fn, windows)
        
        # Check for words that have no context associated
        no_ctxt = [
            list(d.keys())[0] for d in associations if len(list(d.values())[0])==0
        ]
        if len(no_ctxt) > 0:
            import pdb;pdb.set_trace()
            raise ValueError('Word {} could not be associated to any context in admissible tree.'.format(no_ctxt[0]))
            
        # Check for words that have more than one context associated
        several_ctxts = [
            list(d.keys())[0] for d in associations if len(list(d.values())[0])>1
        ]
        if len(no_ctxt) > 0:
            raise ValueError('Word {} is associated to more than one context in admissible tree.'.format(several_ctxts[0]))
        
        
        
    @classmethod
    def __get_word_counts(cls):
        """
            Obtains counts for all words in tree. Counting is performed on tree leaves and 
            aggregated by summation for subsequent parent nodes.
        """
        
        # Get counts for leaves
        cls.__get_leaves_counts()
        
        # Aggregate counts starting from leaves and moving towards the root
        cls.vlmc.tree = cls.__update_tree_counts(parent=cls.vlmc.tree)
     
    
    @classmethod
    def __get_leaves_counts(cls):
        """
            Obtains counts for leaves in tree. Counting is performed distributedly.
        """
        
        # Obtains tree leaves
        leaves = [
            l.word for l in cls.vlmc.get_leaves()
        ]
        
        # Defines function for executing leaves symbols counts
        # by fixating inputs and vocabulary
        count_fn = partial(
            count_subword_ocurrences, 
            ''.join([str(x) for x in cls.X]),
            [str(s) for s in cls.vlmc.vocabulary]
        )
        
        # Distributedly perform leaf symbol counting
        with multiprocessing.Pool(cls.njobs) as p:
            counts=p.map(count_fn, leaves)

        # Saves leaf counts as class attribute
        cls.__leaves_counts = {
            l: c for l, c in zip(leaves, counts)
        }
    
    @classmethod
    def __update_tree_counts(
        cls,
        parent
    ):
        """
            Given that word counts for leaves are already performed, this method aggregates child ocurrences to obtain counts for
            internal nodes. Summing is performed from the leaves towards the root, and all movement across the tree is performed recursively.
        """
        
        # If at internal node, recursively call method for node's children
        if parent.children is not None:
            parent_counts = 0
            for w in parent.children.values():
                w = cls.__update_tree_counts(parent=w)
                parent_counts += w.n_ocurrences
        # If at leaves, obtain number of ocurrences from count previously performed
        else:
            parent.n_ocurrences = cls.__leaves_counts[parent.word]
            return parent
        
        # After summing child ocurrences, save total count as number of parent ocurrences
        parent.n_ocurrences=parent_counts
        
        return parent
    
        
    @classmethod
    def __get_transition_probabilities(cls):
        """
            After count for leaves and internal nodes has been obtained, this method calculates transition counts and probabilities
            for all symbols in tree. A dictionary (transitions_dict) is calculated with transition counts, in which keys are symbols and
            values are also dictionaries (with keys being symbols in vocabulary and values transition counts). The latter dictionary
            is then used to update transition counts and calculate transition probabilities for all nodes in tree.
        """
        
        def __get_node_probabilities(parent):
            """
                This method uses calculated transitions_dict to update transitions counts for objects in tree. Also, from counts, transition
                probabilities are estimated from counts. Movement in tree is performed recursively.
            """
            
            # If not at leaves, recursively call method to move down tree
            if parent.children is not None:
                for w in parent.children.values():
                    __get_node_probabilities(parent=w)
            
            # If at leaves
            else:
                # Save transition counts to object
                parent.transition_counts = cls.__transitions_dict[parent.word]
                
                # Save estimated transition probabilities to object
                parent.transition_probabilities = {
                    s: c/parent.n_ocurrences for s, c in parent.transition_counts.items()
                }
                
                return
            
            # If returning from leaves to root
            
            # Save transition counts to object
            parent.transition_counts = cls.__transitions_dict[parent.word]
            
            # Save estimated transition probabilities to object
            parent.transition_probabilities = {
                s: c/parent.n_ocurrences for s, c in parent.transition_counts.items()
            }
            
            return
        
        
        # Obtain transition counts for root node
        transitions_dict = {
            'root':{
                c.word: c.n_ocurrences for c in cls.vlmc.tree.children.values()
            }
        }
        
        # Obtain words for all nodes in tree (except the root)
        words = [
            node.word for node in cls.vlmc.get_all_nodes() if node.word !='root'
        ]
        
        # Wrapper function for counting transitions, obtained by fixating sample and vocabulary
        count_transitions_fn = partial(
            count_transitions_ocurrences, 
            ''.join([str(x) for x in cls.X]),
            [str(s) for s in cls.vlmc.vocabulary]
        )
        
        # distributedly execute transition count for all nodes except rood
        with multiprocessing.Pool(cls.njobs) as p:
            transitions=p.map(count_transitions_fn, words)
        
        # Update transition counts in transitions dict
        for t in transitions:
            transitions_dict.update(t)
        
        # Save transitions dict as class argument
        cls.__transitions_dict = transitions_dict
        
        # Execute function to update nodes in tree object with transition counts and probabilities
        __get_node_probabilities(cls.vlmc.tree)
            
            
                    
                
        