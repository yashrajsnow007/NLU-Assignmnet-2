import numpy as np
import random

def train_scratch_skipgram(sentences, dim=100, window_size=2, neg_k=5, epochs=5):

    words = [w for s in sentences for w in s]
    vocab = list(set(words))
    word_to_idx = {w:i for i,w in enumerate(vocab)}
    idx_to_word = {i:w for w,i in word_to_idx.items()}
    vocab_size = len(vocab)

    pairs = []
    for s in sentences:
        for i, center in enumerate(s):
            for j in range(max(0,i-window_size), min(len(s),i+window_size+1)):
                if i != j:
                    pairs.append((word_to_idx[center], word_to_idx[s[j]]))

    W = np.random.randn(vocab_size, dim)*0.01
    W_out = np.random.randn(vocab_size, dim)*0.01

    def sigmoid(x): return 1/(1+np.exp(-x))

    for epoch in range(epochs):
        for c,o in pairs:

            v_c = W[c]
            v_o = W_out[o]

            score = sigmoid(np.dot(v_c,v_o))
            grad = score - 1

            W[c] -= 0.01*grad*v_o
            W_out[o] -= 0.01*grad*v_c

            for _ in range(neg_k):
                neg = random.randint(0, vocab_size-1)
                if neg == o: continue

                v_n = W_out[neg]
                s = sigmoid(np.dot(v_c,v_n))
                grad_n = s

                W[c] -= 0.01*grad_n*v_n
                W_out[neg] -= 0.01*grad_n*v_c

    return W, word_to_idx, idx_to_word


def train_scratch_cbow(sentences, dim=100, window_size=2, neg_k=5, epochs=5):

    words = [w for s in sentences for w in s]
    vocab = list(set(words))
    word_to_idx = {w:i for i,w in enumerate(vocab)}
    idx_to_word = {i:w for w,i in word_to_idx.items()}
    vocab_size = len(vocab)

    data = []
    for s in sentences:
        for i in range(window_size, len(s)-window_size):
            context = [word_to_idx[s[j]] for j in range(i-window_size,i+window_size+1) if j!=i]
            target = word_to_idx[s[i]]
            data.append((context,target))

    W = np.random.randn(vocab_size, dim)*0.01
    W_out = np.random.randn(vocab_size, dim)*0.01

    def sigmoid(x): return 1/(1+np.exp(-x))

    for epoch in range(epochs):
        for ctx, target in data:

            v_ctx = np.mean([W[i] for i in ctx], axis=0)
            v_t = W_out[target]

            score = sigmoid(np.dot(v_ctx, v_t))
            grad = score - 1

            W_out[target] -= 0.01*grad*v_ctx

            for i in ctx:
                W[i] -= 0.01*grad*v_t/len(ctx)

    return W, word_to_idx, idx_to_word

def nearest_neighbors_scratch(W, word, word_to_idx, idx_to_word):
    if word not in word_to_idx:
        return []

    vec = W[word_to_idx[word]]
    sims = []

    for i in range(len(W)):
        w = idx_to_word[i]
        if w != word:
            sim = np.dot(vec, W[i])/(np.linalg.norm(vec)*np.linalg.norm(W[i]))
            sims.append((w, sim))

    return sorted(sims, key=lambda x:x[1], reverse=True)[:5]


def analogy_scratch(W, word_to_idx, idx_to_word, a, b, c):
    if any(w not in word_to_idx for w in [a,b,c]):
        return []

    vec = W[word_to_idx[b]] + W[word_to_idx[c]] - W[word_to_idx[a]]

    sims = []
    for i in range(len(W)):
        w = idx_to_word[i]
        if w not in [a,b,c]:
            sim = np.dot(vec, W[i])/(np.linalg.norm(vec)*np.linalg.norm(W[i]))
            sims.append((w, sim))

    return sorted(sims, key=lambda x:x[1], reverse=True)[:3]
