
This repository contains three NLP projects: 
BY/Ahmed Essam 

1. **Arabic POS Tagging**
2. **Semantic Search**
3. **Homonyms Problem in NLP**

---

## 1. Arabic POS Tagging

### Overview:
This project focuses on Part-of-Speech (POS) tagging for Arabic text. POS tagging is the process of labeling each word in a sentence with its corresponding part of speech (e.g., noun, verb, adjective). The goal of this project is to build a POS tagging model that can process Arabic sentences and assign correct POS labels.

### Features:
- Tokenization of Arabic sentences.
- Preprocessing techniques tailored for Arabic (handling diacritics, stemming, and tokenization).
- A trained POS tagging model that works with Arabic text.
  
### Requirements:
- Python 3.x
- Libraries: `nltk`, `keras`, `tensorflow`, `scikit-learn`

Arabic_pos_tagging

https://colab.research.google.com/drive/1lnIun0ZIJ8iiOKZ3KFyAGFgKGLPEOkkZ?usp=sharing


## 2. Semantic Search

### Overview:
This project implements a semantic search system that allows searching for documents or sentences based on meaning, not just keywords. The system uses embeddings from pre-trained models like BERT or Sentence-BERT to understand the meaning of words and sentences in a way that can be compared to other text data. By using semantic similarity, it retrieves the most relevant documents or sentences based on the input query.

### Features:
- Utilizes pre-trained models such as BERT, RoBERTa, or Sentence-BERT for sentence embeddings.
- Retrieval of the most relevant documents or sentences based on semantic meaning.
- Cosine similarity to compare embeddings and rank the relevance of documents.
- Flexible to handle different types of text data for similarity-based retrieval.

### Requirements:
- Python 3.x
- Libraries: `transformers`, `sentence-transformers`, `sklearn`, `numpy`, `torch`

  
Semantic search project

https://colab.research.google.com/drive/1GxzZyQKKDR65COWu3vVArJX0g9ioPQqq?usp=sharing


## 3. Homonyms Problem in NLP

### Overview:
This project addresses the problem of homonyms in Natural Language Processing (NLP). Homonyms are words that have the same spelling or pronunciation but different meanings based on context. This project focuses on building a system that can disambiguate homonyms and correctly identify their meaning based on the surrounding context. The solution uses techniques such as word sense disambiguation (WSD) and contextual embeddings from pre-trained models like BERT to differentiate between the meanings of homonyms.

### Features:
- Uses word embeddings and context-aware models (e.g., BERT) for disambiguating homonyms.
- Applies Word Sense Disambiguation (WSD) methods to identify the correct meaning based on the context.
- Supports multiple homonym types (e.g., same spelling but different meanings, different meanings in different contexts).
- Fine-tuning of pre-trained models for specific homonym disambiguation tasks.

### Requirements:
- Python 3.x
- Libraries: `transformers`, `nltk`, `sklearn`, `torch`, `spacy`



Homonyms Project

https://colab.research.google.com/drive/1KKpSqPcCmb9U7E8s-5taEgiY0y_0HI9F?usp=sharing


