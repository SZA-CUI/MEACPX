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

    return top_keywords

# Extract keywords for each cluster
cluster_keywords = {}
for cluster_id in df_new['topic'].unique():
    cluster_texts = df_new[df_new['topic'] == cluster_id]['Tweets'].tolist()
    keywords = extract_keywords(cluster_texts)
    cluster_keywords[cluster_id] = keywords

# Function to assign labels based on keywords
def assign_labels(keywords):
    event_labels = ['Earthquake', 'Fire', 'Lifeline', 'Stromg Wimd', 'Structural Issue','Vandalism','Telecommunication System issue', 'Thunderstorm','Terrorism','Transport System issue']  # Modify based on your expected events
    label_scores = {label: sum(keyword.lower() in label.lower() for keyword_list in keywords for keyword in keyword_list) for label in event_labels}
    cluster_label = max(label_scores, key=label_scores.get)
    return cluster_label

# Assign labels to clusters based on keywords
df_new['Event_label'] = df_new['topic'].apply(lambda cluster_id: assign_labels(cluster_keywords[cluster_id]))
# Display the results
print(df_new[['Tweets', 'topic', 'Event_label']])

# write the DataFrame to a CSV file
df_new.to_csv('gdrive/My Drive/Colab Notebooks/Preprocessed Datasets/new/22-12-2023/LDA_cluster_Data(1000).csv')
