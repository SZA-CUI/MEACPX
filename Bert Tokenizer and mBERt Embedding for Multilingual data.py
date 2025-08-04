#Bert Tokenizer and mBERt Embedding for Multilingual data
# mBERT Tokenizer and Model (mBERT)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')

# Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, input_size, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, encoding_dim)
        self.decoder = nn.Linear(encoding_dim, input_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Custom Dataset class for DataLoader
class CustomDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], return_tensors='pt', truncation=True, padding=True)
        input_ids = encoding['input_ids'].squeeze()
        return input_ids

# Encode tweets using mBERT
df_new=pd.read_pickle('gdrive/My Drive/Colab Notebooks/Preprocessed Datasets/pre-processed.pkl')
encoded_tweets = []
embeddings = []
for tweet in df_new['preprocessed']:
    inputs = tokenizer(tweet, return_tensors='pt', truncation=True, padding=True)
    outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    encoded_tweets.append(embeddings)

encoded_tweets = np.vstack(encoded_tweets)
# create a Pandas DataFrame from the dictionary
df2 = pd.DataFrame(encoded_tweets)

# write the DataFrame to a CSV file
df2.to_csv('gdrive/My Drive/Colab Notebooks/Preprocessed Datasets/new/24-12-2023/mBERT_Tokenized_EMBEDDED_Data.csv')

# Perform PCA for dimensionality reduction
pca = PCA(n_components=2)
pca_result = pca.fit_transform(encoded_tweets)

# Create a DataFrame with PCA results
df_pca = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])

# Plot the PCA results
plt.scatter(df_pca['PC1'], df_pca['PC2'])
plt.title('PCA Visualization of mBERT Embeddings')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
