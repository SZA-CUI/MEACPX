df_new=pd.read_csv('gdrive/My Drive/Colab Notebooks/Preprocessed Datasets/new/22-12-2023/LDA_cluster_Data')
# .................................................................................................
from sklearn.feature_extraction.text import TfidfVectorizer
# Function to extract keywords from a cluster using TF-IDF
def extract_keywords(cluster_texts):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(cluster_texts)
    feature_names = vectorizer.get_feature_names_out()

    # Find the top keywords for each document (tweet)
    top_keywords = []
    for i in range(len(cluster_texts)):
        row = tfidf_matrix[i].toarray().flatten()
        top_indices = row.argsort()[-5:][::-1]  # Extract top 3 keywords
        top_keywords.append([feature_names[idx] for idx in top_indices])
    #df_new['Keywords']=top_keywords
    return top_keywords

# Extract keywords for each cluster
cluster_keywords = {}
for cluster_id in df_new['topic'].unique():
    cluster_texts = df_new[df_new['topic'] == cluster_id]['Tweets'].tolist()
    keywords = extract_keywords(cluster_texts)
    cluster_keywords[cluster_id] = keywords
  df_cluster = pd.DataFrame(columns=['cluster_1','cluster_2','cluster_3','cluster_4','cluster_5','cluster_6','cluster_7','cluster_8','cluster_9','cluster_10'])
df_c = pd.DataFrame(cluster_keywords[0], columns=['Word_1', 'Word_2', 'Word_3', 'Word_4', 'Word_5'])
# Merge the three columns into a single column named ''
df_cluster['cluster_1'] = df_c['Word_1'] + ' ' + df_c['Word_2'] + ' ' + df_c['Word_3']+ ' ' + df_c['Word_4']+ ' ' + df_c['Word_5']
df_c = pd.DataFrame(cluster_keywords[1], columns=['Word_1', 'Word_2', 'Word_3', 'Word_4', 'Word_5'])
# Merge the three columns into a single column named ''
df_cluster['cluster_2'] = df_c['Word_1'] + ' ' + df_c['Word_2'] + ' ' + df_c['Word_3']+ ' ' + df_c['Word_4']+ ' ' + df_c['Word_5']
df_c = pd.DataFrame(cluster_keywords[2], columns=['Word_1', 'Word_2', 'Word_3', 'Word_4', 'Word_5'])
# Merge the three columns into a single column named ''
df_cluster['cluster_3'] = df_c['Word_1'] + ' ' + df_c['Word_2'] + ' ' + df_c['Word_3']+ ' ' + df_c['Word_4']+ ' ' + df_c['Word_5']
df_c = pd.DataFrame(cluster_keywords[3], columns=['Word_1', 'Word_2', 'Word_3', 'Word_4', 'Word_5'])
# Merge the three columns into a single column named ''
df_cluster['cluster_4'] = df_c['Word_1'] + ' ' + df_c['Word_2'] + ' ' + df_c['Word_3']+ ' ' + df_c['Word_4']+ ' ' + df_c['Word_5']
df_c = pd.DataFrame(cluster_keywords[4], columns=['Word_1', 'Word_2', 'Word_3', 'Word_4', 'Word_5'])
# Merge the three columns into a single column named ''
df_cluster['cluster_5'] = df_c['Word_1'] + ' ' + df_c['Word_2'] + ' ' + df_c['Word_3']+ ' ' + df_c['Word_4']+ ' ' + df_c['Word_5']
df_c = pd.DataFrame(cluster_keywords[5], columns=['Word_1', 'Word_2', 'Word_3', 'Word_4', 'Word_5'])
# Merge the three columns into a single column named ''
df_cluster['cluster_6'] = df_c['Word_1'] + ' ' + df_c['Word_2'] + ' ' + df_c['Word_3']+ ' ' + df_c['Word_4']+ ' ' + df_c['Word_5']
df_c = pd.DataFrame(cluster_keywords[6], columns=['Word_1', 'Word_2', 'Word_3', 'Word_4', 'Word_5'])
# Merge the three columns into a single column named ''
df_cluster['cluster_7'] = df_c['Word_1'] + ' ' + df_c['Word_2'] + ' ' + df_c['Word_3']+ ' ' + df_c['Word_4']+ ' ' + df_c['Word_5']
df_c = pd.DataFrame(cluster_keywords[7], columns=['Word_1', 'Word_2', 'Word_3', 'Word_4', 'Word_5'])
# Merge the three columns into a single column named ''
df_cluster['cluster_8'] = df_c['Word_1'] + ' ' + df_c['Word_2'] + ' ' + df_c['Word_3']+ ' ' + df_c['Word_4']+ ' ' + df_c['Word_5']
df_c = pd.DataFrame(cluster_keywords[8], columns=['Word_1', 'Word_2', 'Word_3', 'Word_4', 'Word_5'])
# Merge the three columns into a single column named ''
df_cluster['cluster_9'] = df_c['Word_1'] + ' ' + df_c['Word_2'] + ' ' + df_c['Word_3']+ ' ' + df_c['Word_4']+ ' ' + df_c['Word_5']
df_c = pd.DataFrame(cluster_keywords[9], columns=['Word_1', 'Word_2', 'Word_3', 'Word_4', 'Word_5'])
# Merge the three columns into a single column named ''
df_cluster['cluster_10'] = df_c['Word_1'] + ' ' + df_c['Word_2'] + ' ' + df_c['Word_3']+ ' ' + df_c['Word_4']+ ' ' + df_c['Word_5']


# Function to assign labels based on keywords
def assign_labels(df_cluster):
    earthquake = '''earthquake', 'aftershock', 'aftershocks', 'earth vibration', 'quake', 'shoks'''
    tsunami = '''tsunami', 'flood', 'flood waves', 'water'''
    fire = '''wildfire', 'fire', 'burning', 'blast','fir', 'smoke'''
    christmas = '''christmas', 'xmas','christ','christ day', 'christians day'''
    eid = '''eid', 'eid ul fitr', 'eid ul adha', 'fitr', 'adha', 'Eid celebrations'''
    ramadan = '''ramzan', 'ramadan','holy month', 'fasting'''
    independence = '''independence day', 'independence', 'day of independence', 'national day', 'national'''
    transportation='''street', 'road', 'arrive', 'travel', 'bus', 'car', 'vehicle'''
    wind= '''strong wind', 'windstorm', 'gusty winds', 'windy weather', 'air currents', 'breeze'''
    entertainment='''concert','dance', 'gathering', 'audiance'''
    lifeline='''lifeline', 'emergency', 'casualty', 'fatalities', 'injury', 'death', 'rescue', 'emergency services'''
    Vandalism='''vandalism', 'vandal', 'destruction', 'damage', 'graffiti', 'defacement'''
    TelecomFailure='''internet','communication', 'signals', 'network','telecom failure', 'internet disruption', 'communication outage', 'network failure', 'signal loss'''
    Terrorism='''terrorism', 'terrorist', 'attack', 'bombing', 'explosion', 'hostage', 'extremist', 'militant'''
    Thunderstorm='''thunderstorm', 'thunder', 'lightning', 'storm', 'heavy rain', 'rainstorm'''
    nlp = en_core_web_sm.load()
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()
    stop = set(stopwords.words('english'))
    punctuation = list(string.punctuation)
    stop.update(punctuation)
    w_tokenizer = WhitespaceTokenizer()
    # clean the set of words
    def furnished(text):
        final_text = []
        for i in text.split():
            if i.lower() not in stop:
                word = lemmatizer.lemmatize(i)
                final_text.append(word.lower())
        return " ".join(final_text)
    DisasterEvent = ''
    ReligiousEvent = ''
    NationalEvent = ''
    RoadEvents = ''
    OtherEvents = ''
    WeatherEvent= ''
    cluster_label=''
    DisasterEvent += furnished(earthquake)
    DisasterEvent += furnished(tsunami)
    DisasterEvent += furnished(fire)
    DisasterEvent += furnished(Vandalism)
    DisasterEvent += furnished(Terrorism)
    DisasterEvent += furnished(lifeline)
    DisasterEvent += furnished(wind)
    ReligiousEvent += furnished(christmas)
    ReligiousEvent += furnished(eid)
    ReligiousEvent += furnished(ramadan)
    NationalEvent += furnished(independence)
    RoadEvents += furnished(transportation)
    OtherEvents +=  furnished(entertainment)
    OtherEvents +=  furnished(TelecomFailure)
    WeatherEvent+=  furnished(Thunderstorm)
    string1 = DisasterEvent
    words = string1.split()
    DisasterEvent = " ".join(sorted(set(words), key=words.index))
    DisasterEvent
    string1 = ReligiousEvent
    words = string1.split()
    ReligiousEvent = " ".join(sorted(set(words), key=words.index))
    ReligiousEvent
    string1 = NationalEvent
    words = string1.split()
    NationalEvent = " ".join(sorted(set(words), key=words.index))
    NationalEvent
    string1 = RoadEvents
    words = string1.split()
    RoadEvents = " ".join(sorted(set(words), key=words.index))
    RoadEvents
    string1 = OtherEvents
    words = string1.split()
    OtherEvents = " ".join(sorted(set(words), key=words.index))
    OtherEvents
    string1 = WeatherEvent
    words = string1.split()
    WeatherEvent=" ".join(sorted(set(words), key=words.index))
    WeatherEvent
    def get_vectors(*strs):
        text = [t for t in strs]
        vectorizer = TfidfVectorizer()
        vectorizer.fit(text)
        return vectorizer.transform(text).toarray()

    disaster_vector = get_vectors(DisasterEvent)
    religious_vector = get_vectors(ReligiousEvent)
    national_vector = get_vectors(NationalEvent)
    road_vector = get_vectors(RoadEvents)
    other_vector = get_vectors(OtherEvents)
    weather_vector = get_vectors(WeatherEvent)

    def jaccard_similarity(query, document):
        intersection = set(query).intersection(set(document))
        union = set(query).union(set(document))
        return len(intersection)/len(union)

    def get_scores(group,tweets):
        scores = []
        for tweet in tweets:
            s = jaccard_similarity(group, tweet)
            scores.append(s)
        return scores

    d_scores = get_scores(DisasterEvent, df_cluster['cluster_1'])
    r_scores = get_scores(ReligiousEvent, df_cluster['cluster_1'])
    n_scores = get_scores(NationalEvent, df_cluster['cluster_1'])
    o_scores = get_scores(OtherEvents, df_cluster['cluster_1'])
    w_scores = get_scores(WeatherEvent, df_cluster['cluster_1'])
    ro_scores = get_scores(RoadEvents, df_cluster['cluster_1'])
    df_cluster['Disaster_Event_score']=d_scores
    df_cluster['Religious_Event_score']=r_scores
    df_cluster['National_Event_score']=n_scores
    df_cluster['Other_Event_score']=o_scores
    df_cluster['Weather_Event_score']=w_scores
    df_cluster['Road_Event_score']=ro_scores
    def get_clusters(l1, l2, l3, l4, l5, l6):
        disaster = []
        religious = []
        national = []
        road = []
        weather = []
        other = []
        for i, j, k, o, p, q in zip(l1, l2, l3, l4, l5, l6):
            m = max(i, j, k, o, p, q)
            if m == i:
                disaster.append(1)
            else:
                disaster.append(0)
            if m == j:
                religious.append(1)
            else:
                religious.append(0)
            if m == k:
                national.append(1)
            else:
                national.append(0)
            if m == o:
                other.append(1)
            else:
                other.append(0)
            if m == p:
                road.append(1)
            else:
                road.append(0)
            if m == q:
                weather.append(1)
            else:
                weather.append(0)
        return disaster, religious, national, other, road, weather
    l1 = df_cluster.Disaster_Event_score.to_list()
    l2 = df_cluster.Religious_Event_score.to_list()
    l3 = df_cluster.National_Event_score.to_list()
    l4 = df_cluster.Other_Event_score.to_list()
    l5 = df_cluster.Road_Event_score.to_list()
    l6 = df_cluster.Weather_Event_score.to_list()
    disaster, religious, national, other, road, weather = get_clusters(l1, l2, l3, l4, l5, l6)
    df_cluster['Disaster_Event_Propability']=disaster
    df_cluster['Religious_Event_Propability']=religious
    df_cluster['National_Event_Propability']=national
    df_cluster['Other_Event_Propability']=other
    df_cluster['Road_Event_Propability']=road
    df_cluster['Weather_Event_Propability']=weather
    #event_labels = ['Earthquake', 'Fire', 'Lifeline', 'Stromg Wimd', 'Structural Issue','Vandalism','Telecommunication System issue', 'Thunderstorm','Terrorism','Transport System issue']  # Modify based on your expected events
assign_labels(df_cluster)
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
df=df_new
# Sample DataFrame with tweets
# Define individual lists
earthquake = ['earthquake', 'aftershock', 'aftershocks', 'earth vibration', 'quake', 'shoks']
tsunami = ['tsunami', 'flood', 'flood waves', 'water']
fire = ['wildfire', 'fire', 'burning', 'blast', 'fir', 'smoke']
christmas = ['christmas', 'xmas', 'christ', 'christ day', 'christians day']
eid = ['eid', 'eid ul fitr', 'eid ul adha', 'fitr', 'adha', 'Eid celebrations']
ramadan = ['ramzan', 'ramadan', 'holy month', 'fasting']
independence = ['independence day', 'independence', 'day of independence', 'national day', 'national']
transportation = ['street', 'road', 'arrive', 'travel', 'bus', 'car', 'vehicle']
wind = ['strong wind', 'windstorm', 'gusty winds', 'windy weather', 'air currents', 'breeze']
entertainment = ['concert', 'dance', 'gathering', 'audience']
lifeline = ['lifeline', 'emergency', 'casualty', 'fatalities', 'injury', 'death', 'rescue', 'emergency services']
Vandalism = ['vandalism', 'vandal', 'destruction', 'damage', 'graffiti', 'defacement']
TelecomFailure = ['internet', 'communication', 'signals', 'network', 'telecom failure', 'internet disruption', 'communication outage', 'network failure', 'signal loss']
Terrorism = ['terrorism', 'terrorist', 'attack', 'bombing', 'explosion', 'hostage', 'extremist', 'militant']
Thunderstorm = ['thunderstorm', 'thunder', 'lightning', 'storm', 'heavy rain', 'rainstorm']
# Sample keywords for 16 categories
categories_keywords = {
    'Disaster': earthquake+tsunami+fire+lifeline+Vandalism+Terrorism,
    'Religious': christmas+eid+ramadan,
    'National': independence,
    'Road': transportation,
    'entertainment': entertainment,
    'Ramadan': ramadan,
    'Weather': wind+Thunderstorm,

    # Add more categories and keywords as needed
}


# Function to assign labels based on Jaccard similarity with 16 categories of keywords
def assign_labels(tweet, categories_keywords):
    vectorizer = TfidfVectorizer()
    vectorizer.fit([' '.join(keywords) for keywords in categories_keywords.values()])

    tweet_vector = vectorizer.transform([tweet]).toarray()

    label_scores = {}
    for category, keywords in categories_keywords.items():
        category_vector = vectorizer.transform([' '.join(keywords)]).toarray()
        jaccard_similarity = len(set(tweet_vector[0]) & set(category_vector[0])) / len(set(tweet_vector[0]) | set(category_vector[0]))
        label_scores[category] = jaccard_similarity

    assigned_label = max(label_scores, key=label_scores.get)
    return assigned_label

# Apply the function to each tweet in the DataFrame
df['Jaccard Similarity'] = df['Classification'].apply(lambda x: assign_labels(x, categories_keywords))

# Display the result
print(df)

# Function to assign labels based on keywords
def assign_labels(Cluster_keywords):
    earthquake = '''earthquake', 'aftershock', 'aftershocks', 'earth vibration', 'quake', 'shoks'''
    tsunami = '''tsunami', 'flood', 'flood waves', 'water'''
    fire = '''wildfire', 'fire', 'burning', 'blast','fir', 'smoke'''
    christmas = '''christmas', 'xmas','christ','christ day', 'christians day'''
    eid = '''eid', 'eid ul fitr', 'eid ul adha', 'fitr', 'adha', 'Eid celebrations'''
    ramadan = '''ramzan', 'ramadan','holy month', 'fasting'''
    independence = '''independence day', 'independence', 'day of independence', 'national day', 'national'''
    transportation='''street', 'road', 'arrive', 'travel', 'bus', 'car', 'vehicle'''
    wind= '''strong wind', 'windstorm', 'gusty winds', 'windy weather', 'air currents', 'breeze'''
    entertainment='''concert','dance', 'gathering', 'audiance'''
    lifeline='''lifeline', 'emergency', 'casualty', 'fatalities', 'injury', 'death', 'rescue', 'emergency services'''
    Vandalism='''vandalism', 'vandal', 'destruction', 'damage', 'graffiti', 'defacement'''
    TelecomFailure='''internet','communication', 'signals', 'network','telecom failure', 'internet disruption', 'communication outage', 'network failure', 'signal loss'''
    Terrorism='''terrorism', 'terrorist', 'attack', 'bombing', 'explosion', 'hostage', 'extremist', 'militant'''
    Thunderstorm='''thunderstorm', 'thunder', 'lightning', 'storm', 'heavy rain', 'rainstorm'''
    nlp = en_core_web_sm.load()
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()
    stop = set(stopwords.words('english'))
    punctuation = list(string.punctuation)
    stop.update(punctuation)
    w_tokenizer = WhitespaceTokenizer()
    # clean the set of words
    def furnished(text):
        final_text = []
        for i in text.split():
            if i.lower() not in stop:
                word = lemmatizer.lemmatize(i)
                final_text.append(word.lower())
        return " ".join(final_text)
    DisasterEvent = ''
    ReligiousEvent = ''
    NationalEvent = ''
    RoadEvents = ''
    OtherEvents = ''
    WeatherEvent= ''
    cluster_label=''
    DisasterEvent += furnished(earthquake)
    DisasterEvent += furnished(tsunami)
    DisasterEvent += furnished(fire)
    DisasterEvent += furnished(Vandalism)
    DisasterEvent += furnished(Terrorism)
    DisasterEvent += furnished(lifeline)
    DisasterEvent += furnished(wind)
    ReligiousEvent += furnished(christmas)
    ReligiousEvent += furnished(eid)
    ReligiousEvent += furnished(ramadan)
    NationalEvent += furnished(independence)
    RoadEvents += furnished(transportation)
    OtherEvents +=  furnished(entertainment)
    OtherEvents +=  furnished(TelecomFailure)
    WeatherEvent+=  furnished(Thunderstorm)
    string1 = DisasterEvent
    words = string1.split()
    DisasterEvent = " ".join(sorted(set(words), key=words.index))
    DisasterEvent
    string1 = ReligiousEvent
    words = string1.split()
    ReligiousEvent = " ".join(sorted(set(words), key=words.index))
    ReligiousEvent
    string1 = NationalEvent
    words = string1.split()
    NationalEvent = " ".join(sorted(set(words), key=words.index))
    NationalEvent
    string1 = RoadEvents
    words = string1.split()
    RoadEvents = " ".join(sorted(set(words), key=words.index))
    RoadEvents
    string1 = OtherEvents
    words = string1.split()
    OtherEvents = " ".join(sorted(set(words), key=words.index))
    OtherEvents
    string1 = WeatherEvent
    words = string1.split()
    WeatherEvent=" ".join(sorted(set(words), key=words.index))
    WeatherEvent
    def get_vectors(*strs):
        text = [t for t in strs]
        vectorizer = TfidfVectorizer()
        vectorizer.fit(text)
        return vectorizer.transform(text).toarray()

    disaster_vector = get_vectors(DisasterEvent)
    religious_vector = get_vectors(ReligiousEvent)
    national_vector = get_vectors(NationalEvent)
    road_vector = get_vectors(RoadEvents)
    other_vector = get_vectors(OtherEvents)
    weather_vector = get_vectors(WeatherEvent)
    def jaccard_similarity(query, document):
        intersection = set(query).intersection(set(document))
        union = set(query).union(set(document))
        return len(intersection)/len(union)
    #def get_scores(group,tweets):
     #   scores = []
      #  for tweet in tweets:
       #     s = jaccard_similarity(group, tweet)
        #    scores.append(s)
        #return scores

        # Function to assign labels based on Jaccard similarity scores
    def get_scores(keywords, event_labels):
        label_scores = []
        # Calculate Jaccard similarity scores for each label
        label_scores = {label: jaccard_similarity(label, keywords) for label in event_labels}

        # Assign the label with the highest similarity score
        cluster_label = max(label_scores, key=label_scores.get)
        return label_scores

    d_scores = get_scores(DisasterEvent, cluster_keywords)
    r_scores = get_scores(ReligiousEvent, cluster_keywords)
    n_scores = get_scores(NationalEvent, cluster_keywords)
    o_scores = get_scores(OtherEvents, cluster_keywords)
    w_scores = get_scores(WeatherEvent, cluster_keywords)
    ro_scores = get_scores(RoadEvents, cluster_keywords)
    df_new['Disaster_Event_score']=d_scores
    df_new['Religious_Event_score']=r_scores
    df_new['National_Event_score']=n_scores
    df_new['Other_Event_score']=o_scores
    df_new['Weather_Event_score']=w_scores
    df_new['Road_Event_score']=ro_scores
    def get_clusters(l1, l2, l3, l4, l5, l6):
        disaster = []
        religious = []
        national = []
        road = []
        weather = []
        other = []
        for i, j, k, o, p, q in zip(l1, l2, l3, l4, l5, l6):
            m = max(i, j, k, o, p, q)
            if m == i:
                disaster.append(1)
            else:
                disaster.append(0)
            if m == j:
                religious.append(1)
            else:
                religious.append(0)
            if m == k:
                national.append(1)
            else:
                national.append(0)
            if m == o:
                other.append(1)
            else:
                other.append(0)
            if m == p:
                road.append(1)
            else:
                road.append(0)
            if m == q:
                weather.append(1)
            else:
                weather.append(0)
        return disaster, religious, national, other, road, weather
    l1 = df_new.Disaster_Event_score.to_list()
    l2 = df_new.Religious_Event_score.to_list()
    l3 = df_new.National_Event_score.to_list()
    l4 = df_new.Other_Event_score.to_list()
    l5 = df_new.Road_Event_score.to_list()
    l6 = df_new.Weather_Event_score.to_list()
    disaster, religious, national, other, road, weather = get_clusters(l1, l2, l3, l4, l5, l6)
    df_new['Disaster_Event_Propability']=disaster
    df_new['Religious_Event_Propability']=religious
    df_new['National_Event_Propability']=national
    df_new['Other_Event_Propability']=other
    df_new['Road_Event_Propability']=road
    df_new['Weather_Event_Propability']=weather
    #event_labels = ['Earthquake', 'Fire', 'Lifeline', 'Stromg Wimd', 'Structural Issue','Vandalism','Telecommunication System issue', 'Thunderstorm','Terrorism','Transport System issue']  # Modify based on your expected events
assign_labels(keywords)
# Assign labels to clusters based on keywords TFIDF and Jaccard Similarity
df_new = df_new.rename(columns={'stemmed': 'text'})
df_new['topic'].apply(lambda cluster_id: assign_labels(cluster_keywords[cluster_id]))
df_new.head(5)
# write the LDA TFIDF and Jaccard Result to a CSV file
df_new.to_csv('gdrive/My Drive/Colab Notebooks/Preprocessed Datasets/new/22-12-2023/LDA TFIDF and Jaccard Result(1000).csv')
# Assign labels to clusters based on keywords TFIDF and Jaccard Similarity
df_new['Bert_Autoencoder_cluster'].apply(lambda cluster_id: assign_labels(cluster_keywords[cluster_id]))
# write the Bert Autoencoder TFIDF and Jaccard Result to a CSV file
df_new.to_csv('gdrive/My Drive/Colab Notebooks/Preprocessed Datasets/new/22-12-2023/Bert Autoencoder TFIDF and Jaccard Result(1000).csv')
# .................................................................................................
