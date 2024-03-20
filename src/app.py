from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

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

# Load the trained model
model = joblib.load('../models/model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input text from the request
        text = request.json['text']

        # Vectorize the input text
        text_tfidf = tfidf_vectorizer.transform([text])

        # Make predictions using the trained model
        prediction = model.predict(text_tfidf)

        # Return the prediction as JSON
        return jsonify({'prediction': int(prediction[0])})  # Assuming prediction is binary (0 or 1)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
