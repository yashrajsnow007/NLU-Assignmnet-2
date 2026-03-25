# NLU-Assignment-2  
**Natural Language Understanding (Spring 2026) — IIT Jodhpur**  
Author: **Yashraj Chaturvedi**

This repository contains solutions for two assignment problems:

1. **Word Embeddings on IIT Jodhpur Corpus** (CBOW + Skip-gram)
2. **Character-Level Name Generation** (RNN, BLSTM, Attention-style setup)

---

## Project Structure

```text
NLU-Assignmnet-2/
├── word_embeddings/
│   ├── data/
│   │   ├── raw.txt
│   │   └── corpus.txt
│   ├── results/
│   │   ├── experiment_results.csv
│   │   ├── wordcloud.png
│   │   ├── cbow_pca.png
│   │   ├── skipgram_pca.png
│   │   ├── scratch_cbow_pca.png
│   │   └── scratch_skipgram_pca.png
│   └── src/
│       ├── main.py
│       └── utils/
│           ├── preprocessing.py
│           ├── models.py
│           ├── scratch.py
│           └── visualization.py
├── name_generation/
│   ├── data/TrainingNames.txt
│   ├── models/
│   │   ├── rnn.py
│   │   ├── blstm.py
│   │   └── attention.py
│   ├── train/
│   │   ├── train_rnn.py
│   │   ├── train_blstm.py
│   │   └── train_attention.py
│   └── results/
│       ├── rnn.txt
│       ├── blstm.txt
│       └── attention.txt
└── requirements.txt

##Setup

1) Create environment (recommended)
python -m venv .venv
source .venv/bin/activate

2) Install dependencies
pip install -r requirements.txt

Problem 1: Word Embeddings (CBOW + Skip-gram)
Objective
Train and compare Word2Vec models on IIT Jodhpur text after preprocessing.

Preprocessing Summary

Lowercasing
URL/email removal
Non-alphabetic filtering
Boilerplate + noise removal
Tokenization
Stopword filtering
Frequency-based cleanup

Corpus Statistics

Total tokens: 34,035
Vocabulary size: 3,967
Total sentences: 1,702

Run
cd word_embeddings
python src/main.py


Main Outputs
Generated under word_embeddings/results/:

experiment_results.csv
wordcloud.png
cbow_pca.png
skipgram_pca.png
scratch_cbow_pca.png
scratch_skipgram_pca.png


Best Hyperparameter Configuration

Dimension: 200
Window: 3
Negative samples: 5
CBOW score: 0.9946
Skip-gram score: 0.9209

Observations

CBOW performs better overall on this corpus size.
Skip-gram gives more functionally distinct neighborhoods in many cases.
From-scratch Skip-gram is comparatively strong; from-scratch CBOW is weaker.



Problem 2: Character-Level Name Generation

Objective
Generate Indian names character-by-character using recurrent models:

Vanilla RNN
BLSTM
Attention-style model setup


Dataset

Training names: 1000 (LLM-generated)
File: name_generation/data/TrainingNames.txt

Common Hyperparameters

Embedding size = 32
Hidden size = 128
Learning rate = 0.003
Epochs = 25


Run
cd name_generation
python -m train.train_rnn
python -m train.train_blstm
python -m train.train_attention


Quantitative Results

Model	Trainable Params	Novelty	Diversity
RNN	24,633	0.790	0.985
BLSTM	175,314	0.685	0.980
Attention Run	175,314	0.685	0.965



Interpretation

RNN: highest novelty/diversity (more creative outputs)
BLSTM: better structural realism but lower novelty
Attention run: currently close to BLSTM behavior in reported outputs

Dependencies
From requirements.txt:

gensim
matplotlib
scikit-learn
wordcloud
pandas
torch
numpy
Notes
This repository is an academic submission for NLU Assignment 2.
Results depend on data cleaning choices, random initialization, and training seed.