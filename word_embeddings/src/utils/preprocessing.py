import re
from collections import Counter

stop_words = {
    "is","are","was","were","be","been","being","are", "that", "will"
    "have","has","had",
    "do","does","did",
    "the","a","an",
    "in","on","at","of","to","for","with","by",
    "that","this","these","those",
    "and","or","but",
    "not","will","can","should","could","would",
    "as","from","it","its","into","than","then",
    "also","such","their","there","which","who",
    "what","when","where","why","how", "through", "src", "have"
}

def preprocess():
    with open("data/raw.txt", "r", encoding="utf-8") as f:
        text = f.read().lower()

    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)

    junk_phrases = [
        "indian institute technology",
        "iit jodhpur",
        "all rights reserved",
        "important links",
        "redirecttologinpage",
        "sitemap",
        "last updated"
    ]

    for phrase in junk_phrases:
        text = text.replace(phrase, " ")

    text = re.sub(r'digital infrastructure automation.*?india', ' ', text)

    junk_words = [
        "home","about","contact","login","menu","skip","portal",
        "links","click","next","previous","arrow","view","play",
        "copyright","reserved","cccd","nirf","rti","tenders",
        "repository","feedback","committee","intranet","website", "download",
    ]

    for word in junk_words:
        text = re.sub(r'\b' + word + r'\b', ' ', text)

    remove_words = [
        "indian","institute","technology","jodhpur","iitj",
        "india","campus"
    ]

    for word in remove_words:
        text = re.sub(r'\b' + word + r'\b', ' ', text)

    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [w for w in words if len(w) > 2]

    freq = Counter(words)
    filtered_words = [w for w in words if freq[w] < 400]

    tokens = filtered_words

    with open("data/corpus.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(tokens))

    print("Total tokens:", len(tokens))
    print("Vocabulary size:", len(set(tokens)))

    return tokens
