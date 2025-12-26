# Cyberbullying Text Preprocessing

This repository contains a Python-based project for preprocessing textual data in the context of **cyberbullying detection**. The project demonstrates common NLP techniques including tokenization, stopword removal, frequency analysis, POS tagging, and lemmatization.

---

## Project Overview

The project focuses on **text preprocessing** for a cyberbullying dataset. Preprocessing is a crucial step in NLP pipelines to prepare raw text data for modeling and analysis. The main operations demonstrated here include:

1. **Tokenization** – Splitting text into individual words or tokens.
2. **Stopword Removal** – Filtering out common words that do not contribute meaningful information.
3. **Frequency Analysis** – Counting occurrences of words to understand their importance.
4. **POS (Part-of-Speech) Tagging** – Identifying grammatical categories like nouns, verbs, adjectives.
5. **Lemmatization** – Reducing words to their base or dictionary form.

---

## Tools & Libraries

- **Python 3.x**
- **pandas** – For data manipulation and CSV file handling.
- **NLTK** – For tokenization, stopword filtering, POS tagging, and lemmatization.
- **Jupyter Notebook** – For interactive analysis and experimentation.

---

## Example Workflow

```python
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK packages
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load dataset
df = pd.read_csv("cyberbullying.csv")

# Tokenization
sample_text = df.iloc[6]["Text"]
tokens = word_tokenize(sample_text.lower())

# Stopword removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word not in stop_words]

# Frequency distribution
freq_dist = FreqDist(filtered_tokens)

# POS tagging
pos_tags = nltk.pos_tag(filtered_tokens)

# Lemmatization with POS mapping
lemmatizer = WordNetLemmatizer()
def get_wordnet_pos(nltk_pos):
    from nltk.corpus import wordnet
    if nltk_pos.startswith('J'):
        return wordnet.ADJ
    elif nltk_pos.startswith('V'):
        return wordnet.VERB
    elif nltk_pos.startswith('N'):
        return wordnet.NOUN
    elif nltk_pos.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

lemmatized_words = [(word, lemmatizer.lemmatize(word, get_wordnet_pos(pos))) for word, pos in pos_tags]
