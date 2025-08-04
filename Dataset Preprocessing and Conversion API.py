#datasets Tramslation into english resulted (mix of multilinhual tweets due to incomplete translation of individual tweet text) 
import csv
from googletrans import Translator
import time

# Load your dataset (adjust path if needed)
df = pd.read_csv("FinalDataset3.csv", encoding="latin-1")
# Initialize translator
translator = Translator()
# Function to safely translate text
def safe_translate(text):
    try:
        # Translate and return the translated text
        translated = translator.translate(text).text
        return translated
    except Exception as e:
        print(f"Translation failed for: {text}\nError: {e}")
        return text  # Return original if failed
# Apply translation row-wise (can be slow due to API limits)
df['Translated_Tweet'] = df['Tweets'].apply(lambda x: safe_translate(str(x)))
# Save to a new CSV file
df.to_csv("FinalDataset3.csv", index=False, encoding="utf-8")

print("Translation complete. Saved to FinalDataset3.csv")
