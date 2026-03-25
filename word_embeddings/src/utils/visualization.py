from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plot_gensim(model, words, title, save_path=None):
    # Extract word vectors from Gensim model
    vectors = []
    labels = []

    for w in words:
        if w in model.wv:
            vectors.append(model.wv[w])
            labels.append(w)

    # Reduce to 2D using PCA for visualization
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)

    # Plot words in 2D space
    plt.figure(figsize=(8,6))
    for i, word in enumerate(labels):
        x, y = coords[i]
        plt.scatter(x, y)
        plt.text(x, y, word)

    plt.title(title)

    if save_path:
        plt.savefig(save_path)

    plt.show()

def plot_scratch(W, w2i, i2w, words, title, save_path=None):
    # Extract vectors from scratch-trained embeddings using vocab mappings
    vectors = []
    labels = []

    for w in words:
        if w in w2i:
            vectors.append(W[w2i[w]])
            labels.append(w)

    if len(vectors) < 2:
        print("Not enough words")
        return

    # Reduce to 2D using PCA for visualization
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)

    # Plot words in 2D space
    plt.figure(figsize=(8,6))
    for i, word in enumerate(labels):
        x, y = coords[i]
        plt.scatter(x, y)
        plt.text(x, y, word)

    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()
