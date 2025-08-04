# Multilingual Event-Aware Content Prioritization
This repository contains the implementation of our multilingual, event-aware content prioritization fraemework applied on X (Twitter) data related to hazardous events. The goal of the project is to identify and prioritize tweets containing multilingual content by combining natural language processing (NLP), topic modeling, deep learning, and contextual feature extraction techniques.

---
## 🔍 Project Objective

With increasing global social media activity during emergencies, prioritizing relevant information across languages is crucial. This project focuses on:
- Translating multilingual tweets
- Preprocessing and feature extraction
- Clustering and topic modeling (LDA, BERT, Autoencoders)
- Extracting semantic features (hashtags, time, location)
- Assigning event-aware labels
- Prioritizing content for emergency response

---

## 📁 Dataset

**Title:** Dataset of tweets, used to detect hazardous events at the Baths of Diocletian site in Rome  
**DOI:** [10.5281/zenodo.3258415](https://doi.org/10.5281/zenodo.3258415)  
**Source:** [Zenodo Repository](https://zenodo.org/record/3258415)  
- Tweets are originally in italian and translated into English using an in-code translation API.
- Due to automatic translation, **minor inconsistencies** may arise; however, trends, conclusions, and feature extraction pipelines remain robust.

---

## 🧠 Project Structure and Execution Flow

| File Name | Description |
|----------|-------------|
| `Required Libraries and Settings.py` | Contains all necessary imports, environment configurations, and global constants. |
| `Dataset Preprocessing and Conversion API.py` | Loads the original dataset and uses translation APIs to convert multilingual tweets into English. |
| `Preprocessing of data and feature extraction.py` | Cleans, tokenizes, removes stopwords, extracts hashtags, and prepares tweet tokens. |
| `Rename Bert Tokenizer and mBERt Embedding for Multiling.ipynb` | Applies multilingual BERT (mBERT) for semantic embeddings of tweets. |
| `LDA Model Fine Tuning.py` | Performs topic modeling using LDA and optimizes the number of topics for better coherence. |
| `Autoencoder Configuration and Tuning.py` | Applies autoencoders for dimensionality reduction and clustering of tweet vectors. |
| `Top 5 words extraction within each cluster.py` | Identifies the most representative words in each cluster for labeling. |
| `Labelling of each Cluster.py` | Assigns meaningful labels to tweet clusters using dominant keywords. |
| `Assign labels to clusters based on keywords.py` | Matches predefined keywords to clusters for supervised label mapping. |
| `Time Slotting and Additional Feature Extraction.py` | Extracts temporal (day, month, hour, year) and spatial (location, city) features from tweet text or metadata. |
| `Event Aware Content Distinguishing Using Combined Feature set.py` | Integrates all extracted features to identify and prioritize event-relevant tweets. |
| `Sample_Executed_Code_For_Guidance.ipynb` | A demonstration notebook with 1,000 sample tweets for ease of understanding and replication. |

---

## 🔄 Execution Guide

1. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2. **Translate tweets** (if raw dataset is used):
    ```bash
    python Dataset Preprocessing and Conversion API.py
    ```

3. **Preprocess tweets**:
    ```bash
    python Preprocessing of data and feature extraction.py
    ```

4. **Generate embeddings**:
    ```bash
    python Rename Bert Tokenizer and mBERt Embedding for Multiling.ipynb
    ```

5. **Apply topic modeling and clustering**:
    ```bash
    python LDA Model Fine Tuning.py
    python Autoencoder Configuration and Tuning.py
    ```

6. **Extract keywords and label clusters**:
    ```bash
    python Top 5 words extraction within each cluster.py
    python Labelling of each Cluster.py
    python Assign labels to clusters based on keywords.py
    ```

7. **Extract additional features**:
    ```bash
    python Time Slotting and Additional Feature Extraction.py
    ```

8. **Run event-aware prioritization**:
    ```bash
    python Event Aware Content Distinguishing Using Combined Feature set.py
    ```

9. **Refer to the sample execution**:
    Open `Sample_Executed_Code_For_Guidance.ipynb` in Google Colab or Jupyter Notebook for end-to-end flow on 1000 tweets.

---

## ✅ Outputs

- Cleaned and tokenized tweet data
- Multilingual embeddings using mBERT
- LDA topics and clustered tweets
- Keyword-based cluster labels
- Time-based slot assignment
- Final tweet classification: **Event-relevant vs Non-relevant**

---

## ⚠️ Notes on Translated Content

Due to automatic translation of tweets using APIs:
- **Word variations and contextual shifts** may occur.
- **Named entities (cities, places)** might get misinterpreted.
- However, **trends, findings, and event detection logic** remain consistent.

---

## 🧪 Sample Execution

To get started, simply run the notebook:
```bash
Sample_Executed_Code_For_Guidance.ipynb

