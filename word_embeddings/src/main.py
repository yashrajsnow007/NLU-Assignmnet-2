from utils.preprocessing import preprocess
from utils.models import *
from utils.scratch import *
from utils.visualization import *

from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

# Create results directory
os.makedirs("results", exist_ok=True)

# ========== 1. PREPROCESS ==========
# Clean and prepare raw corpus
tokens = preprocess()

# ========== 2. WORDCLOUD ==========
# Load corpus and generate word cloud visualization
with open("data/corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.imshow(wordcloud)
plt.axis("off")
plt.savefig("results/wordcloud.png")
plt.show()

# ========== 3. CREATE SENTENCES ==========
# Split corpus into fixed-size sentences using sliding window
words = text.split()

sentences = []
window_size = 20

for i in range(0, len(words), window_size):
    sentences.append(words[i:i+window_size])

print("Total sentences:", len(sentences))

# ========== 4. HYPERPARAMETER EXPERIMENTS ==========
# Test different embedding configurations (18 total)
df = run_experiments(sentences)
print(df)

best = df.iloc[0]  # Get best configuration

# ========== 5. TRAIN FINAL MODELS ==========
# Train Gensim models with best hyperparameters
cbow_model = train_gensim_model(sentences, int(best["dim"]), int(best["window"]), int(best["negative"]), 0)
skipgram_model = train_gensim_model(sentences, int(best["dim"]), int(best["window"]), int(best["negative"]), 1)

# Save experiment results to CSV
df.to_csv("results/experiment_results.csv", index=False)

print("\n=== FINAL EXPERIMENT TABLE ===")
print(df.head(10))

# ========== 6. SCRATCH IMPLEMENTATIONS ==========
# Train custom implementations of CBOW and Skip-gram
W_sg, w2i_sg, i2w_sg = train_scratch_skipgram(sentences)
W_cb, w2i_cb, i2w_cb = train_scratch_cbow(sentences)

# ========== 7. TEST NEAREST NEIGHBORS ==========
# Find similar words for test set
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

# ========== 8. VISUALIZE EMBEDDINGS ==========
# Generate PCA plots for embedding visualization
words_to_plot = [
    "research", "publication", "project", "labs",
    "students", "phd", "mtech", "undergraduate",
    "engineering", "science", "data",
    "course", "program", "degree"
]

# Plot Gensim models
plot_gensim(skipgram_model, words_to_plot, "Skip-gram PCA",
            "results/skipgram_pca.png")

plot_gensim(cbow_model, words_to_plot, "CBOW PCA",
            "results/cbow_pca.png")

# Plot scratch implementations
plot_scratch(W_sg, w2i_sg, i2w_sg, words_to_plot, "Scratch Skip-gram PCA",
             "results/scratch_skipgram_pca.png")

plot_scratch(W_cb, w2i_cb, i2w_cb, words_to_plot, "Scratch CBOW PCA",
             "results/scratch_cbow_pca.png")

# ========== 9. SAMPLE EMBEDDING ==========
# Display embedding vector for a sample word
word = "research"
vector = skipgram_model.wv[word]
vector_str = ", ".join([str(round(v, 4)) for v in vector])
print(f"\n{word} embedding: {vector_str}")

# ========== 10. WORD FREQUENCY ANALYSIS ==========
# Find most common words in corpus
freq = Counter(tokens)
top10 = freq.most_common(10)

print("\nTop 10 frequent words:", top10)

result = []
for word, count in top10:
    result.append(f"{word}, {count}")

print(", ".join(result))

# ========== 11. WORD ANALOGIES - GENSIM SKIP-GRAM ==========
print("\n===== ANALOGIES (GENSIM SKIP-GRAM) =====")

print("UG : BTech :: PG :",
      analogy_gensim(skipgram_model, "undergraduate", "btech", "postgraduate"))

print("\nresearch : publication :: teaching :",
      analogy_gensim(skipgram_model, "research", "publication", "teaching"))

print("\nstudent : phd :: undergraduate :",
      analogy_gensim(skipgram_model, "students", "phd", "undergraduate"))

# ========== 12. WORD ANALOGIES - SCRATCH ==========
print("\n===== ANALOGIES (SCRATCH) =====")

print("UG : BTech :: PG :",
      analogy_scratch(W_sg, w2i_sg, i2w_sg, "undergraduate", "btech", "postgraduate"))

print("\nresearch : publication :: teaching :",
      analogy_scratch(W_sg, w2i_sg, i2w_sg, "research", "publication", "teaching"))

print("\nstudent : phd :: undergraduate :",
      analogy_scratch(W_sg, w2i_sg, i2w_sg, "students", "phd", "undergraduate"))

# ========== 13. WORD ANALOGIES - GENSIM CBOW ==========
print("\n===== ANALOGIES (GENSIM CBOW) =====")

print("UG : BTech :: PG :",
      analogy_gensim(cbow_model, "undergraduate", "btech", "postgraduate"))

print("\nresearch : publication :: teaching :",
      analogy_gensim(cbow_model, "research", "publication", "teaching"))

print("\nstudent : phd :: undergraduate :",
      analogy_gensim(cbow_model, "students", "phd", "undergraduate"))

# ========== 14. WORD ANALOGIES - SCRATCH CBOW ==========
print("\n===== ANALOGIES (SCRATCH CBOW) =====")

print("UG : BTech :: PG :",
      analogy_scratch(W_cb, w2i_cb, i2w_cb, "undergraduate", "btech", "postgraduate"))

print("\nresearch : publication :: teaching :",
      analogy_scratch(W_cb, w2i_cb, i2w_cb, "research", "publication", "teaching"))

print("\nstudent : phd :: undergraduate :",
      analogy_scratch(W_cb, w2i_cb, i2w_cb, "students", "phd", "undergraduate"))

print("\n✓ All experiments complete!")
