import numpy as np

def count_word_ocurrences(
    X,
    word
):
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