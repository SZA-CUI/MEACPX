# Assuming df_new['extract_keywords'] contains preprocessed text data
vectorizer = TfidfVectorizer(stop_words='english')

# Replace NaN values with an empty string
periodic_data['Classification'].fillna('', inplace=True)
periodic_data['Localization'].fillna('', inplace=True)

# Convert 'Classification' and 'Localization' columns to strings and then concatenate
concatenated_text = periodic_data['Classification'].astype(str) + ' ' + periodic_data['Localization'].astype(str)

# Apply TF-IDF vectorization
keywords_matrix = vectorizer.fit_transform(concatenated_text)

# Extract other relevant features
other_features = periodic_data[['day', 'hour','year', 'time_slot']]

# Combine TF-IDF features and other features
X_tfidf = keywords_matrix.toarray()
X_other = other_features.values.astype(np.float64)
X = StandardScaler().fit_transform(np.concatenate((X_tfidf, X_other), axis=1))

# Apply HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
periodic_data['Classification'] = clusterer.fit_predict(X)

# Visualize the clusters in 2D using PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Create a DataFrame for visualization
df_visualize = pd.DataFrame({'PCA1': X_pca[:, 0], 'PCA2': X_pca[:, 1], 'Cluster_Labels': periodic_data['Event_label']})

# Map negative cluster labels to NaN
df_visualize['Cluster_Labels'] = df_visualize['Cluster_Labels'].apply(lambda x: x if x >= 0 else np.nan)

# Create a condensed distance matrix
condensed_distance_matrix = sch.linkage(X_pca, method='ward')

# Plot the cluster dendrogram
plt.figure(figsize=(15, 8))
dendrogram = sch.dendrogram(condensed_distance_matrix, truncate_mode='level', p=10, labels=df_visualize['Cluster_Labels'].to_numpy())
plt.title('HDBSCAN Clustering Dendrogram')
plt.show()
