from gensim.models import Word2Vec
import pandas as pd

def train_gensim_model(sentences, dim=100, window=5, neg=5, sg=0):
    model = Word2Vec(
        sentences=sentences,
        vector_size=dim,
        window=window,
        min_count=2,
        workers=4,
        sg=sg,
        negative=neg,
        epochs=10
    )
    return model

def evaluate_model(model):
    test_words = ["research", "students", "phd", "engineering"]

    scores = []
    for word in test_words:
        if word in model.wv:
            sims = model.wv.most_similar(word, topn=5)
            scores.append(sum([s for _, s in sims]) / 5)

    return sum(scores)/len(scores) if scores else 0

def run_experiments(sentences):
    dims = [50, 100, 200]
    windows = [3, 5, 8]
    negatives = [5, 10]

    results = []

    for dim in dims:
        for win in windows:
            for neg in negatives:

                cbow = train_gensim_model(sentences, dim, win, neg, sg=0)
                sg = train_gensim_model(sentences, dim, win, neg, sg=1)

                cbow_score = evaluate_model(cbow)
                sg_score = evaluate_model(sg)

                results.append({
                    "dim": dim,
                    "window": win,
                    "negative": neg,
                    "CBOW": round(cbow_score, 4),
                    "SkipGram": round(sg_score, 4)
                })

    df = pd.DataFrame(results)
    return df.sort_values(by="SkipGram", ascending=False)

def nearest_neighbors_gensim(model, word):
    if word in model.wv:
        return model.wv.most_similar(word, topn=5)
    return []


def analogy_gensim(model, a, b, c):
    return model.wv.most_similar(positive=[b, c], negative=[a], topn=3)
