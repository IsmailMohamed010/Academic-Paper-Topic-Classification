from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize Flask app
app = Flask(__name__)

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load pre-trained models and preprocessing objects
try:
    vectorizer = joblib.load('vectorizer.pkl')
    lr_model = joblib.load('best_lr_model.pkl')
    nb_model = joblib.load('naive_bayes_model.pkl')
    rnn_model = load_model('rnn_model.keras')
    tokenizer = joblib.load('tokenizer.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
except Exception as e:
    print(f"Error loading models or preprocessors: {e}")
    raise

# Text preprocessing function (same as used during training)
def process_text(text):
    text = re.sub(r'\s+', ' ', text, flags=re.I)  # Remove extra whitespace
    text = re.sub(r'\W', ' ', str(text))  # Remove special characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)  # Remove single characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetical characters
    text = text.lower()

    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    words = [word for word in words if len(word) > 3]

    indices = np.unique(words, return_index=True)[1]
    cleaned_text = np.array(words)[np.sort(indices)].tolist()

    return ' '.join(cleaned_text)

# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        # Extract and preprocess text
        input_text = data['text']
        cleaned_text = process_text(input_text)

        # Prepare data for Logistic Regression and Naive Bayes
        tfidf_text = vectorizer.transform([cleaned_text])

        # Logistic Regression prediction
        lr_pred = lr_model.predict(tfidf_text)
        lr_pred_label = lr_pred[0]

        # Naive Bayes prediction
        nb_pred = nb_model.predict(tfidf_text)
        nb_pred_label = nb_pred[0]

        # Prepare data for RNN
        max_len = 200
        seq = tokenizer.texts_to_sequences([cleaned_text])
        padded_seq = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
        
        # RNN prediction
        rnn_pred_prob = rnn_model.predict(padded_seq, verbose=0)
        rnn_pred_enc = np.argmax(rnn_pred_prob, axis=1)
        rnn_pred_label = label_encoder.inverse_transform(rnn_pred_enc)[0]

        # Return predictions
        response = {
            'logistic_regression': lr_pred_label,
            'naive_bayes': nb_pred_label,
            'rnn': rnn_pred_label
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'API is running'}), 200

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
