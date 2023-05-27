from .bic_solver import BICSolver
from .context_algorithm_solver import ContextAlgorithmSolver
from .bct_solver import BCTSolver
from .counter import Counter

from treelib import Node, Tree

class VLMC:
    
    def __init__(
        self,
        max_order,
        vocabulary
    ):
        if vocabulary is None:
            raise ValueError("Vocabulary for variable length markov chain must be specified.")
        else:
            self.vocabulary=vocabulary
            self.__vocabulary_str = [str(s) for s in vocabulary]
        
        if max_order is None:
            raise ValueError("Max order for variable length markov chain must be specified.")
        else:
            self.max_order = int(max_order)
            
        self.tree = self.__initialize_tree()
        
    def __initialize_tree(self):
        
        def __get_children(parent):
            
            if self.max_order>0:
                children={}
                if parent.word=='root':                
                    word_len=1
                    for s in self.__vocabulary_str:
                        children[s] = Word(
                            word=s,
                            word_len=word_len,
                            is_leaf=True if word_len==self.max_order else False
                        )

                else:
                    word_len=parent.word_len+1
                    for s in self.__vocabulary_str:
                        children[s] = Word(
                            word=s+parent.word,
                            word_len=word_len,
                            is_leaf=True if word_len==self.max_order else False
                        )      

                if word_len < self.max_order:
                    for k, w in children.items():
                        children[k]=__get_children(parent=w)
                else:
                    parent.children=children
                    return parent

                parent.children=children
            return parent
            
        root=Word(word='root')
        tree = __get_children(
            parent=root
        )
        
        return tree
    
    def get_all_nodes(self):

        def __get_all_nodes(node):

            child_nodes = []
            if node.children is not None:
                for w in node.children.values():
                    child_nodes.extend(__get_all_nodes(node=w))
            else:
                return [node]

            return [node] + child_nodes

        nodes = __get_all_nodes(node=self.tree)

        return nodes
        
    def get_leaves(self):

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
        
        if solver is not None:
            pass
        else:
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
        
        return tree_plot
        


           
        
        
class Word:
    
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

        
