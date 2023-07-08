import numpy as np

def count_word_ocurrences(
    X,
    word
):
    """
        Counts the number of ocurrences of word in sample X.
    """
    
    l=len(word)
    counts=0
    for i in range(l, len(X)+1):
        if X[i-l:i] == word:
            counts += 1
    return counts


def count_subword_ocurrences(
    X,
    vocabulary,
    word
):
    """
        Counts the number of ocurrences of children of word in sample X, then sums to obtain 
        number of ocurrences of word in X.
    """
    
    counts = [
        count_word_ocurrences(
            X=X,
            word=word+s
        ) for s in vocabulary
    ]
    return np.sum(counts)


def count_transitions_ocurrences(
    X,
    vocabulary,
    word
):
    """
        Counts the number of transitions in X from word to each symbol in vocabulary.
    """
    
    counts = [
        count_word_ocurrences(
            X=X,
            word=word+s
        ) for s in vocabulary
    ]
    
    return {
        word: {
            str(s): c for s, c in zip(vocabulary, counts)
        }
    }

def obtain_associated_context(
    leaves,
    word
):
    """
        Associates a member of leaves to word.
    """
    associated_context = {word: []}
    for l in leaves:
        length_leaf = len(l)
        if word[-length_leaf:]==l:
            associated_context[word].append(l)
    # if len(associated_context[word])==0:
    #     import pdb;pdb.set_trace()
    return associated_context