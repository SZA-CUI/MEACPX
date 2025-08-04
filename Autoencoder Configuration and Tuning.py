
# Autoencoder Configuration
input_size = encoded_tweets.shape[1]
encoding_dim = 20  # You can adjust this based on the desired encoding dimension

# Train Autoencoder
autoencoder = Autoencoder(input_size, encoding_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=1e-3)

# Convert encoded_tweets to PyTorch tensor
encoded_tweets_tensor = torch.tensor(encoded_tweets, dtype=torch.float32)

# DataLoader for autoencoder training
autoencoder_dataloader = DataLoader(encoded_tweets_tensor, batch_size=100, shuffle=True)

# Training loop for autoencoder
num_epochs = 30
for epoch in range(num_epochs):
    for batch in autoencoder_dataloader:
        optimizer.zero_grad()
        outputs = autoencoder(batch)
        loss = criterion(outputs, batch)
        loss.backward()
        optimizer.step()

# Apply K-Means clustering on autoencoder encoded representations
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(encoded_tweets)

# Add cluster information to the DataFrame
df_new['Bert_Autoencoder_cluster'] = clusters
print(df_new[['Tweets','Classification', 'Bert_Autoencoder_cluster']])
# Assuming 'arr1' is your DataFrame with 'stemmed' and 'Classification' columns
# Replace this with your actual DataFrame and column names

plt.figure(figsize=(10, 8))
# Count the occurrences of each class
class_counts = df_new['Bert_Autoencoder_cluster'].value_counts()
bar_plot = sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
# Increase the space between x-axis labels
bar_plot.set_xticklabels(bar_plot.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.title('Distribution of Tweets')
plt.tight_layout()  # Adjust layout for better spacing
plt.show()
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Assuming you have a DataFrame 'df' with a 'cluster' column and a 'text' column
# df = ...

# Generate WordClouds for each cluster
for cluster_id in df_new['mBERT_Autoencoder_cluster'].unique():
    # Extract text for the current cluster
    cluster_text = ' '.join(df_new[df_new['mBERT_Autoencoder_cluster'] == cluster_id]['preprocessed'])

    # Generate WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cluster_text)

    # Plot WordCloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud - Cluster {cluster_id}')
    plt.axis('off')
    plt.show()
