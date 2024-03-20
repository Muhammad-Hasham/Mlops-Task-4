import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the dataset using the correct relative path
df = pd.read_csv('../Data/airlines_reviews.csv')

# Preprocess the data (cleaning, text normalization, etc.)
# For simplicity, let's assume the 'Recommended' column indicates sentiment (1 for positive, 0 for negative)
df['Sentiment'] = df['Recommended'].apply(lambda x: 1 if x == 'yes' else 0)

# Split the data into features (text reviews) and target (sentiment)
X = df['Reviews']
y = df['Sentiment']

# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Limiting to top 1000 features for demonstration
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Initialize and train the model (Logistic Regression as an example)
model = LogisticRegression()
model.fit(X_tfidf, y)

# Save the trained model to the models folder
joblib.dump(model, '../models/model.pkl')  # Assuming the models folder is at the same level as src folder

# Function to predict sentiment based on input text
def predict_sentiment(text):
    text_tfidf = tfidf_vectorizer.transform([text])
    prediction = model.predict(text_tfidf)
    return prediction

# Sample input for testing the Flask API
sample_input = {
    "text": [
        "The flight was amazing! I rate it 9 out of 10."
    ]
}

# Testing the model with the sample input
for text in sample_input['text']:
    prediction = predict_sentiment(text)
    print(f'Text: {text}')
    print(f'Predicted Sentiment: {prediction[0]}')
