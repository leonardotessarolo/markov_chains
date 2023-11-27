from .bic_solver import BICSolver
from .context_algorithm_solver import ContextAlgorithmSolver
from .bct_solver import BCTSolver
from .counter import Counter

from treelib import Node, Tree

class VLMC:
    """
        Objects from this class represent a tree for a variable-length Markov Chain (VLMC) and are endowed with useful methods. Representation 
        is created by using a set of Word class objects. For each node (represented by a Word), children are a list of Word
        objects (each a node or leaf) saved to its "children" attribute.
    """
    def __init__(
        self,
        max_order,
        vocabulary,
        make_admissible=True
    ):
        """
            Class constructor method. Args:
                - max_order (int): max depth for tree.
                - vocabulary (array): array of symbols that are the vocabulary for the tree.
                - make_admissible (boolean): flags for applying admissibility criteria to context tree.
        """
        
        # Checks if vocabulary is not None
        if vocabulary is None:
            raise ValueError("Vocabulary for variable length markov chain must be specified.")
        else:
            self.vocabulary=vocabulary
            self.__vocabulary_str = [str(s) for s in vocabulary]
        
        # Checks if max_order is not None
        if max_order is None:
            raise ValueError("Max order for variable length markov chain must be specified.")
        else:
            self.max_order = int(max_order)
            
        # Saves flag for admissibility
        self.make_admissible = make_admissible
        
        # Initialize full tree with given max depth and vocabulary, and save to tree
        self.tree = self.__initialize_tree()
        
    def __initialize_tree(self):
        """
            This method initializes full tree with given max depth and vocabulary.
        """
        
        def __get_children(parent):
            # Only obtain nodes if given max_order is not 0
            if self.max_order>0:
                
                children={}
                
                # If at root, initialize root children
                if parent.word=='root':                
                    word_len=1
                    for s in self.__vocabulary_str:
                        children[s] = Word(
                            word=s,
                            word_len=word_len,
                            is_leaf=True if word_len==self.max_order else False
                        )

                else:
                    # If not at root, initialize node children
                    word_len=parent.word_len+1
                    for s in self.__vocabulary_str:
                        children[s] = Word(
                            word=s+parent.word,
                            word_len=word_len,
                            is_leaf=True if word_len==self.max_order else False
                        )      
                # If not at depth max_order-1, recursively call method to move down tree
                if word_len < self.max_order:
                    for k, w in children.items():
                        children[k]=__get_children(parent=w)
                # If at depth max_order-1, save children to Word object. Leaves do not have
                # children, so the "children" attribute for them is None.
                else:
                    parent.children=children
                    return parent
                
                # Move back towards the root saving children to parents' "children" attribute
                parent.children=children
                
            return parent
        
        # Initialize Word object for root
        root=Word(word='root')
        
        # Recursively initialize Word objects to represent tree nodes
        tree = __get_children(
            parent=root
        )
        
        return tree
    
    def get_all_nodes(self):
        """
            Get all nodes as Word objects in list.
        """

        def __get_all_nodes(node):
            """
                Recursively move in tree, saving Word objects in list.
            """

            child_nodes = []
            # If not at leaves
            if node.children is not None:
                # Iterate in children, recursively calling method
                for w in node.children.values():
                    child_nodes.extend(__get_all_nodes(node=w))
            # If at leaves
            else:
                # Return node (Word object)
                return [node]
            
            # Return concatenated list of node and child nodes
            return [node] + child_nodes
        
        # Call method to count all nodes
        nodes = __get_all_nodes(node=self.tree)

        return nodes
        
    def get_leaves(self):
        """
            This method obtains all leaves for tree. All nodes are obtained, and then leaves are filtered.
        """

        nodes = self.get_all_nodes()
        leaves = [n for n in nodes if n.is_leaf]

        return leaves
        
    def fit(
        self,
        X,
        solver=None,
        method='bic',
        njobs=1
    ):
        """
           High-level method for performing inference in context tree. Arguments:
               - X (array): input sample.
               - solver: to be implemented, option for giving already initialized BICSolver, ContextAlgorithmSolver or BCTSolver with custom parameters
               as input.
               - method (string): method for estimating context tree. Can be either 'bic', 'context' or 'bct'. Default is 'bic'.
               - njobs (int): number of parallel jobs to instantiate for performing symbol counting and other tasks.
        """
        
        if solver is not None:
            # TODO: make compatible with giving solver object as argument
            pass
        
        else:
            # Solves if specified method is BIC
            if method=='bic':
                Counter.fit(
                    vlmc=self,
                    X=X,
                    njobs=njobs
                )
                BICSolver.fit(
                    vlmc=self,
                    X=X
                )
            # Solves if specified method is Context Algorithm
            elif method=='context':
                Counter.fit(
                    vlmc=self,
                    X=X,
                    njobs=njobs
                )
                ContextAlgorithmSolver.fit(
                    vlmc=self,
                    X=X
                )
            # Solves if specified method is BCT
            elif method=='bct':
                Counter.fit(
                    vlmc=self,
                    X=X,
                    njobs=njobs
                )
                BCTSolver.fit(
                    vlmc=self,
                    X=X
                )
    
    
    def show_tree(self):
        """
            Method for showing tree saved in "tree" attribute.
        """
        
        def __create_plot_node(parent, tree):
            
            if parent.children is not None:
                for w in parent.children.values():
                    tree.create_node(w.word, w.word, parent=parent.word)
                    __create_plot_node(parent=w, tree=tree)
                    
            else:
                return
            return tree
            
        tree_plot = Tree()
        tree_plot.create_node(self.tree.word, self.tree.word)
        tree_plot = __create_plot_node(
            parent=self.tree,
            tree=tree_plot
        )
        
        tree_plot.show()
        


           
        
        
class Word:
    """
        Object that represents a node, corresponding to a specific symbol in context tree.
    """
    
    def __init__(
        self,
        word=None,
        word_len=None,
        is_leaf=None,
        n_ocurrences=None,
        transition_counts=None,
        transition_probabilities=None,
        children=None
    ):
        self.word=word
        self.word_len=word_len
        self.is_leaf=is_leaf
        self.n_ocurrences=n_ocurrences
        self.transition_probabilities=transition_probabilities
        self.children=children

        
