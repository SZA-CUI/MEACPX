 # Preprocessing of data and feature extraction from tweets
 Tweetsdataset=pd.read_csv('FinalDataset3.csv', encoding='Latin-1')
 # REMOVE '@USER'
def remove_users(tweet, pattern1, pattern2):
    r = re.findall(pattern1, tweet)
    for i in r:
        tweet = re.sub(i, '', tweet)

    r = re.findall(pattern2, tweet)
    for i in r:
        tweet = re.sub(i, '', tweet)
    return tweet
Tweetsdataset['tidy_tweet'] = np.vectorize(remove_users)(Tweetsdataset['Tweets'], "@ [\w]*", "@[\w]*")

Tweetsdataset['tidy_tweet'] = Tweetsdataset['tidy_tweet'].str.lower()
# Hashtags Extraction
import re
import numpy as np

def extract_hashtags(tweet, pattern1, pattern2):
    hashtags1 = re.findall(pattern1, tweet)
    hashtags2 = re.findall(pattern2, tweet)

    all_hashtags = hashtags1 + hashtags2
    extracted_pattern = ' '.join(all_hashtags)

    return extracted_pattern

Tweetsdataset['extracted_hashtags'] = np.vectorize(extract_hashtags)(Tweetsdataset['tidy_tweet'], r"#\w*", r"#\w*")
# REMOVE LINKS
def remove_links(tweet):
    tweet_no_link = re.sub(r"http\S+", "", tweet)
    return tweet_no_link
Tweetsdataset['tidy_tweet'] = np.vectorize(remove_links)(Tweetsdataset['tidy_tweet'])
# REMOVE Punctuations, Numbers, and Special Characters
Tweetsdataset['tidy_tweet'] = Tweetsdataset['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")
# REMOVE SHORT WORDS
Tweetsdataset['tidy_tweet'] = Tweetsdataset['tidy_tweet'].apply(lambda x:' '.join([w for w in x.split() if len(w)>3]))
# TOKENIZATION
def tokenize(tweet):
    for word in tweet:
        yield(gensim.utils.simple_preprocess(str(word), deacc=True))
Tweetsdataset['tidy_tweet_tokens'] = list(tokenize(Tweetsdataset['tidy_tweet']))
# Prepare Stop Words
import nltk
nltk.download('stopwords')
stop_words = stopwords.words('english')
stop_words.extend(['from', 'https', 'twitter', 'religions', 'pic','twitt',])
# REMOVE STOPWORDS
def remove_stopwords(tweets):
    return [[word for word in simple_preprocess(str(tweet)) if word    not in stop_words] for tweet in tweets]
Tweetsdataset['tokens_no_stop'] = remove_stopwords(Tweetsdataset['tidy_tweet_tokens'])

import dateutil.parser
import spacy
import re

# Load SpaCy English model
nlp = spacy.load('en_core_web_sm')

# Extract year and time from tweet text
def extract_year_time(tweet):
    time_matches = re.findall(r'\b(?:[01]?\d|2[0-3]):[0-5]\d\b', tweet)  # hh:mm format
    year_matches = re.findall(r'\b(19|20)\d{2}\b', tweet)  # 4-digit year

    time_str = time_matches[0] if time_matches else None
    year_str = year_matches[0] if year_matches else None

    return pd.Series([time_str, year_str])

Tweetsdataset[['extracted_time', 'extracted_year']] = Tweetsdataset['tidy_tweet'].apply(extract_year_time)

# Extract named locations using SpaCy NER
def extract_location(text):
    doc = nlp(text)
    locations = [ent.text for ent in doc.ents if ent.label_ in ['GPE', 'LOC', 'FAC']]  # Cities, countries, places
    return ', '.join(locations) if locations else None

Tweetsdataset['extracted_location'] = Tweetsdataset['tidy_tweet'].apply(extract_location)

# REMOVE TWEETS LESS THAN 2 TOKENS
Tweetsdataset['length'] = Tweetsdataset['tokens_no_stop'].apply(len)
Tweetsdataset = Tweetsdataset.drop(Tweetsdataset[Tweetsdataset['length']<2].index)
Tweetsdataset = Tweetsdataset.drop(['length'], axis=1)
Tweetsdataset.shape
Tweetsdataset.reset_index(drop=True, inplace=True)
Tweetsdataset.to_pickle('gdrive/My Drive/Colab Notebooks/Preprocessed Datasets/pre-processed.pkl')
!pip install pyLDAvis
