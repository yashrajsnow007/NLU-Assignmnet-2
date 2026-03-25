from utils.preprocessing import preprocess
from utils.models import *
from utils.scratch import *
from utils.visualization import *

from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# -------------------------
# PREPROCESS
# -------------------------
tokens = preprocess()

# -------------------------
# WORDCLOUD
# -------------------------
with open("data/corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.imshow(wordcloud)
plt.axis("off")
plt.show()

# -------------------------
# SENTENCES
# -------------------------
words = text.split()

sentences = []
window_size = 20

for i in range(0, len(words), window_size):
    sentences.append(words[i:i+window_size])

print("Total sentences:", len(sentences))

# -------------------------
# EXPERIMENTS
# -------------------------
df = run_experiments(sentences)
print(df)

best = df.iloc[0]

cbow_model = train_gensim_model(sentences, int(best["dim"]), int(best["window"]), int(best["negative"]), 0)
skipgram_model = train_gensim_model(sentences, int(best["dim"]), int(best["window"]), int(best["negative"]), 1)

df.to_csv("results/experiment_results.csv", index=False)

print("\n=== FINAL EXPERIMENT TABLE ===")
print(df.head(10))

# -------------------------
# SCRATCH
# -------------------------
W_sg, w2i_sg, i2w_sg = train_scratch_skipgram(sentences)
W_cb, w2i_cb, i2w_cb = train_scratch_cbow(sentences)

words = ["research", "students", "phd", "exam"]

print("\n===== GENSIM CBOW =====")
for w in words:
    print(w, "->", nearest_neighbors_gensim(cbow_model, w))

print("\n===== GENSIM SKIP-GRAM =====")
for w in words:
    print(w, "->", nearest_neighbors_gensim(skipgram_model, w))

print("\n===== SCRATCH SKIP-GRAM =====")
for w in words:
    print(w, "->", nearest_neighbors_scratch(W_sg, w, w2i_sg, i2w_sg))

print("\n===== SCRATCH CBOW =====")
for w in words:
    print(w, "->", nearest_neighbors_scratch(W_cb, w, w2i_cb, i2w_cb))

# -------------------------
# PLOTS
# -------------------------
words_to_plot = [
    "research", "publication", "project", "labs",
    "students", "phd", "mtech", "undergraduate",
    "engineering", "science", "data",
    "course", "program", "degree"
]

plot_gensim(skipgram_model, words_to_plot, "Skip-gram PCA")
plot_gensim(cbow_model, words_to_plot, "CBOW PCA")

plot_scratch(W_sg, w2i_sg, i2w_sg, words_to_plot, "Scratch Skip-gram PCA")
plot_scratch(W_cb, w2i_cb, i2w_cb, words_to_plot, "Scratch CBOW PCA")

# -------------------------
# VECTOR PRINT
# -------------------------
word = "research"
vector = skipgram_model.wv[word]
vector_str = ", ".join([str(round(v, 4)) for v in vector])
print(f"{word} - {vector_str}")

# -------------------------
# TOP WORDS
# -------------------------
freq = Counter(tokens)
top10 = freq.most_common(10)

print(top10)

result = []
for word, count in top10:
    result.append(f"{word}, {count}")

print(", ".join(result))
