#LDA Model Fine Tuning
df = pd.read_pickle('gdrive/My Drive/Colab Notebooks/Preprocessed Datasets/pre-processed.pkl')
# TOKENIZE
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
        # deacc=True removes punctuations
data_words = list(sent_to_words(df['tokens_no_stop']))
# Build the bigram and trigram model
bigram = gensim.models.Phrases(data_words, min_count=10, threshold=100)
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
# Faster way to get a sentence clubbed as a bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]
def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]
# Form Bigrams
data_words_bigrams = make_bigrams(data_words)
# LEMMATIZATION
def lemmatization(tweets, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    tweets_out = []
    for sent in tweets:
        doc = nlp(" ".join(sent))
        tweets_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return tweets_out
# Initialize spacy 'en' model, keeping only tagger component
# python3 -m spacy download en
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
# Lemmatization keeping only noun, adj, vb, adv
df['lemmatized'] = pd.Series(lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']))
# STEMMING
stemmer = PorterStemmer()
df['stemmed'] = df['lemmatized'].apply(lambda x : [stemmer.stem(y) for y in x])
df.to_pickle('gdrive/My Drive/Colab Notebooks/Preprocessed Datasets/Final-pre-processed.pkl')
 = pd.read_pickle('gdrive/My Drive/Colab Notebooks/Preprocessed Datasets/Final-pre-processed.pkl')
 # Create Dictionary
id2word_stemmed = corpora.Dictionary(df['stemmed'])
print(id2word_stemmed)
# Create Corpus

tweets_stemmed = df['stemmed']
tweets_class = df['Classification']
tweets_hashtags = df['extracted_hashtags']
print(tweets_stemmed)
print (tweets_class)
print(df['stemmed'][1])
# Term Document Frequency Word to Vector Model
corpus_stemmed = [id2word_stemmed.doc2bow(tweet) for tweet in tweets_stemmed]
lda_model_stemmed = gensim.models.ldamodel.LdaModel(
    corpus=corpus_stemmed,
    id2word=id2word_stemmed,
    num_topics=30,  # Adjusted number of topics for better representation
    random_state=42,  # Changed random_state for reproducibility
    chunksize=1000,  # Increased chunksize for faster processing
    passes=30,  # Increased the number of passes for better convergence
    alpha='auto',
    eta='auto',  # Updated eta parameter for better results
    per_word_topics=True,
    eval_every=2  # Adjusted evaluation frequency for efficiency
)
lda_train = lda_model_stemmed

import joblib
joblib.dump(lda_train, 'gdrive/My Drive/Colab Notebooks/Preprocessed Datasets/lda_train_model_new.jl')
# then reload it with
import joblib
lda_train = joblib.load('gdrive/My Drive/Colab Notebooks/Preprocessed Datasets/lda_train_model_new.jl')
lda_train.print_topics(23, num_words=5)[:10]
lda_train.show_topic(9)
num_topics= 15
dictionary=id2word_stemmed
corpus=corpus_stemmed
texts=df['stemmed']
limit=15
start=1
step=3

coherence_values = []
perplexity_values = []
alpha_values = []
beta_values = []
model_list = []

# Assuming you have predefined values for alpha and beta, replace them accordingly
alpha = 'auto'
beta = 'auto'

num_topics_range = range(start, limit, step)

for num_topics in num_topics_range:
    model = gensim.models.LdaModel(
        corpus=corpus_stemmed,
        num_topics=num_topics,
        id2word=id2word_stemmed,
        alpha=alpha,
        eta=beta
    )
    model_list.append(model)

    # Compute Coherence
    coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_values.append(coherencemodel.get_coherence())

    # Compute Perplexity
    perplexity_values.append(model.log_perplexity(corpus_stemmed))

    # Append alpha and beta values
    alpha_values.append(model.alpha)
    beta_values.append(model.eta)

    # Print values at each iteration
    print(f"Num Topics: {num_topics}, Coherence Value: {coherencemodel.get_coherence()}, Perplexity: {model.log_perplexity(corpus_stemmed)}, Alpha: {model.alpha}, Beta: {model.eta}")

# Print the results
print("Coherence Values:")
pprint(list(zip(num_topics_range, coherence_values)))

print("\nPerplexity Values:")
pprint(list(zip(num_topics_range, perplexity_values)))

print("\nAlpha Values:")
pprint(list(zip(num_topics_range, alpha_values)))

print("\nBeta Values:")
pprint(list(zip(num_topics_range, beta_values)))
import matplotlib.pyplot as plt

# Assuming you have run the code above and have coherence_values, perplexity_values, alpha_values, and beta_values
# Make sure to replace these variable names if they are different in your code

# Plotting Coherence Values
plt.figure(figsize=(12, 6))
plt.plot(num_topics_range, coherence_values, marker='o', label='Coherence')
plt.title('LDA Model Coherence Values for Different Number of Topics')
plt.xlabel('Number of Topics')
plt.ylabel('Coherence Value')
plt.legend()
plt.show()

# Plotting Perplexity Values
plt.figure(figsize=(12, 6))
plt.plot(num_topics_range, perplexity_values, marker='o', label='Perplexity')
plt.title('LDA Model Perplexity Values for Different Number of Topics')
plt.xlabel('Number of Topics')
plt.ylabel('Perplexity Value')
plt.legend()
plt.show()

alpha_values = []

# Assuming you have the following alpha_values
alpha_values = [
    (1, np.array([1.])),
    (4, np.array([1.9923849, 2.8027625, 1.0267801, 1.3529899])),
    (7, np.array([0.509044  , 0.811627  , 0.98322076, 0.6067799 , 0.4330494 ,
       1.3601582 , 0.24136871])),
    (10, np.array([0.1858202 , 0.35335213, 0.52781767, 0.24209055, 0.3407299 ,
       0.30428106, 0.39656073, 0.14789815, 0.6804027 , 0.34035245])),
    (13, np.array([0.14282241, 0.20257315, 0.14460357, 0.12082244, 0.31781885,
       0.10158791, 0.15768039, 0.14807795, 0.27420977, 0.25447127,
       0.2021856 , 0.32270557, 0.36984152]))
              ]

# Extract alpha values from the tuples
flat_alpha_values = np.concatenate([np.ravel(arr) for _, arr in alpha_values])

# Now, you can plot the alpha values
plt.figure(figsize=(12, 6))
plt.plot(num_topics_range, flat_alpha_values, marker='o', label='Alpha')
plt.title('LDA Model Alpha Values for Different Number of Topics')
plt.xlabel('Number of Topics')
plt.ylabel('Alpha Value')
plt.legend()
plt.show()

beta_values = []
beta_values=[(1, np.array([2.9697707e-01, 1.1561176e+03, 1.5073018e+01, 2.7129743e-01, 2.7129743e-01, 2.7129743e-01])),
        (4, np.array([ 0.13671067, 18.754654  ,  0.19122608,  0.12991083, 0.13006817,  0.1299422 ])),
        (7, np.array([0.092001  , 4.4913797 , 0.11049675,  0.09210596, 0.09204318, 0.09213404])),
        (10, np.array([ 0.0718831 , 15.151722  ,  0.08150688,   0.07194762, 0.0719115 ,  0.07190474])),
        (13, np.array([ 0.05976966, 17.9333    ,  0.06567974,   0.05977086, 0.05977089,  0.05976675]))]

# Plotting Beta Values
plt.figure(figsize=(12, 6))
plt.plot(num_topics_range, beta_values, marker='o', label='Beta')
plt.title('LDA Model Beta Values for Different Number of Topics')
plt.xlabel('Number of Topics')
plt.ylabel('Beta Value')
plt.legend()
plt.show()

pprint(list(zip(num_topics_range, alpha_values)))
