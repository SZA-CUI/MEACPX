# Basic libraries
import os
import re
import string
import warnings
import itertools
import collections
from pprint import pprint
from collections import Counter
from datetime import datetime

# Data handling
import pandas as pd
import numpy as np
import csv

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# NLP: NLTK
import nltk
from nltk import bigrams
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import RegexpTokenizer, WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer

# NLP: spaCy
import spacy
import en_core_web_sm

# NLP: Gensim
import gensim
from gensim.utils import simple_preprocess
from gensim import corpora
from gensim.models import CoherenceModel
# from gensim.models.wrappers import LdaMallet  # Optional

# LDA visualization
import pyLDAvis
import pyLDAvis.gensim_models

# Tweet streaming
import tweepy

# Transformers (BERT)
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AdamW

# Sklearn utilities
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

# Hierarchical clustering
import scipy.cluster.hierarchy as sch

# Advanced clustering
import hdbscan
