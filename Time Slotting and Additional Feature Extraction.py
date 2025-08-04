from datetime import datetime
# Assuming df_new['Time'] is a column in your DataFrame
df_new['Formatted_Time'] = df_new['Time'].apply(lambda x: datetime.utcfromtimestamp(x / 1000.0).strftime('%Y-%m-%d %H:%M:%S'))
# Print or display the resulting DataFrame
print(df_new[['Time', 'Formatted_Time']])
import pandas as pd
# Assuming df is your DataFrame with a 'time' column
# Convert 'time' column to datetime format
df_new['Formatted_Time'] = pd.to_datetime(df_new['Formatted_Time'])

# Extract month, day, and hour from the 'time' column
df_new['month'] = df_new['Formatted_Time'].dt.month
df_new['day'] = df_new['Formatted_Time'].dt.day
df_new['hour'] = df_new['Formatted_Time'].dt.hour

# Define time slots (adjust as needed)
time_slots = [(0, 3), (4, 7), (8, 11), (12, 15), (16, 19), (20, 23)]

# Function to assign slots based on the hour
def assign_time_slot(hour):
    for slot, (start, end) in enumerate(time_slots, start=1):
        if start <= hour <= end:
            return slot
    return None

# Apply the function to create a 'time_slot' column
df_new['time_slot'] = df_new['hour'].apply(assign_time_slot)

# Display the resulting DataFrame
print(df_new[['Tweets','Time', 'month', 'day', 'hour', 'time_slot', 'Localization','extracted_hashtags','Event_label','Classification']])


# Assuming df_new['extract_keywords'] contains preprocessed text data
vectorizer = TfidfVectorizer(stop_words='english')

# Replace NaN values with an empty string
df_new['Event_label'].fillna('', inplace=True)
df_new['Localization'].fillna('', inplace=True)

# Concatenate 'Event_Class' and 'Localization' before applying the vectorizer
keywords_matrix = vectorizer.fit_transform(df_new['Classification'] + ' ' + df_new['Localization'])

# Extract other relevant features
other_features = df_new[['day', 'hour', 'time_slot']]

# Combine TF-IDF features and other features
X = StandardScaler().fit_transform(keywords_matrix.toarray())
X = np.concatenate((X, other_features.values), axis=1)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
df_new['Specific Event'] = dbscan.fit_predict(X)

# Replace -1 (outliers) with a specific label or handle them accordingly
df_new['Specific Event'] = np.where(df_new['Specific Event'] == -1, 'Outlier', df_new['Specific Event'])
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Assuming X is your feature matrix
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(X)

plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=df_new['Specific Event'], cmap='viridis')
plt.title('DBSCAN Clusters (PCA)')
plt.show()
# Display the updated DataFrame with the cluster assignments
print(df_new[['Tweets', 'Time', 'month', 'day', 'hour', 'time_slot', 'Localization', 'extracted_hashtags', 'Event_label', 'Specific Event']])
# Load spaCy English model
nlp = spacy.load("en_core_web_sm")
# Sample DataFrame (replace this with your full dataset)
df_new = pd.DataFrame(df_new)
# Step 1: Convert default timestamp
df_new['Formatted_Time'] = pd.to_datetime(df_new['Time'], unit='ms')
df_new['default_year'] = df_new['Formatted_Time'].dt.year
df_new['default_day'] = df_new['Formatted_Time'].dt.day
# Step 2: Extract features from text
def extract_location(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "GPE":
            return ent.text
    return None
def extract_year(text):
    years = re.findall(r'(20\d{2})', text)
    return int(years[0]) if years else None
def extract_day(text):
    try:
        date = dateutil.parser.parse(text, fuzzy=True, default=datetime(1900, 1, 1))
        return date.day
    except Exception:
        return None
# Step 3: Apply extraction functions
df_new['location_extracted'] = df_new['Tweets'].apply(extract_location)
df_new['year_extracted'] = df_new['Tweets'].apply(extract_year)
df_new['day_extracted'] = df_new['Tweets'].apply(extract_day)

# Step 4: Replace values if extracted
df_new['Localization'] = df_new.apply(
    lambda row: row['location_extracted'] if row['location_extracted'] else row['Localization'], axis=1)
df_new['year'] = df_new.apply(
    lambda row: row['year_extracted'] if row['year_extracted'] else row['default_year'], axis=1)
df_new['day'] = df_new.apply(
    lambda row: row['day_extracted'] if row['day_extracted'] else row['default_day'], axis=1)

# Clean up temporary columns
df_new.drop(columns=['location_extracted', 'year_extracted', 'day_extracted', 'default_year', 'default_day'], inplace=True)

# Final output
print(df_new[['Tweets', 'Localization', 'year', 'day','Time']])
