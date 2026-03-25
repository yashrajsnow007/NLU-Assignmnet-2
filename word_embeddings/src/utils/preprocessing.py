import re
from collections import Counter

# Common stop words to remove from corpus
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
    # Load and lowercase raw text
    with open("data/raw.txt", "r", encoding="utf-8") as f:
        text = f.read().lower()

    # Remove URLs, emails, and special characters
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Remove institution-specific junk phrases
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

    # Remove long institutional patterns
    text = re.sub(r'digital infrastructure automation.*?india', ' ', text)

    # Remove website navigation and metadata words
    junk_words = [
        "home","about","contact","login","menu","skip","portal",
        "links","click","next","previous","arrow","view","play",
        "copyright","reserved","cccd","nirf","rti","tenders",
        "repository","feedback","committee","intranet","website", "download",
    ]

    for word in junk_words:
        text = re.sub(r'\b' + word + r'\b', ' ', text)

    # Remove redundant location and institution words
    remove_words = [
        "indian","institute","technology","jodhpur","iitj",
        "india","campus"
    ]

    for word in remove_words:
        text = re.sub(r'\b' + word + r'\b', ' ', text)

    # Split into words and remove stop words and short words (len<=2)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [w for w in words if len(w) > 2]

    # Remove very frequent words (>400 occurrences)
    freq = Counter(words)
    filtered_words = [w for w in words if freq[w] < 400]

    tokens = filtered_words

    # Save preprocessed corpus to file
    with open("data/corpus.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(tokens))

    print("Total tokens:", len(tokens))
    print("Vocabulary size:", len(set(tokens)))

    return tokens
