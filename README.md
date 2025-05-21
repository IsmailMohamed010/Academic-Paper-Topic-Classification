# ArXiv Abstract Classification

This project involves the classification of scientific paper abstracts from the [ArXiv dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv) into various research categories using traditional machine learning (Logistic Regression, Naive Bayes) and deep learning (RNN with LSTM).

---

## 📚 Dataset

- **Source**: [Kaggle - ArXiv Metadata OAI Snapshot](https://www.kaggle.com/datasets/Cornell-University/arxiv)
- **Format**: JSON lines  
- **Size**: ~2.7 million records  
- **Features Used**: `abstract` (text), `categories` (target)

---

## 🛠️ Features

- Text cleaning and preprocessing using NLTK  
- TF-IDF vectorization for ML models  
- Tokenization + Padding for RNN  
- Logistic Regression and Multinomial Naive Bayes models  
- Deep Learning with LSTM-based RNN  
- Confusion matrices, accuracy, precision, recall, and F1-score visualizations  
- Export of trained models and vectorizers for deployment  
- Flask API available in `app.py` for serving predictions

---

## 🧪 Models Used

| Model                | Input         | Accuracy |
|---------------------|---------------|----------|
| Logistic Regression | TF-IDF        |  78.3%   |
| Naive Bayes         | TF-IDF        |  50.4%   |
| RNN (LSTM)          | Sequences     |  76.7%   |

---

## 🧰 Requirements

- Python 3.8+  
- pandas, numpy  
- scikit-learn  
- nltk  
- seaborn, matplotlib  
- tensorflow / keras  
- joblib  
- flask (for API)

Install requirements:

```bash
pip install -r requirements.txt
```

You may create `requirements.txt` using:

```bash
pip freeze > requirements.txt
```

---

## 🚀 How to Run

### 1. Download and extract the dataset  
From: https://www.kaggle.com/datasets/Cornell-University/arxiv  
Place the file `arxiv-metadata-oai-snapshot.json` into a directory such as `./data`.

> 📦 If you're unable to upload the dataset to GitHub due to size limits, you can host it externally.  
> 🔗 **[Download Dataset from Google Drive](https://drive.google.com/drive/u/0/folders/1d9Dfkptzs_6b3s3Jg0J1dAueQe8skMv8)**

### 2. Run the Notebook  
Use the notebook `arxiv_classification.ipynb` to:

- Load and preprocess data  
- Train and evaluate models  
- Save models and encoders

### 3. Serve Predictions with Flask API (Optional)  
Run the Flask app:

```bash
python app.py
```

Use a REST client or `curl` to POST abstracts and receive predicted categories.

---

## 📈 Visualizations Included

- Distribution of abstract lengths  
- Category frequency plots  
- Confusion matrices for all models  
- Accuracy comparison across models  
- Precision, recall, F1-score bar plots  
- RNN training history (accuracy and loss)

---

## 🧠 Future Work

- Experiment with BERT or other transformer-based models  
- Multi-label classification (papers can belong to multiple categories)  
- Deployment with FastAPI + Docker  
- Add model explainability (e.g., SHAP for LR)

---

## 📂 Saved Files

- `logistic_regression_model.pkl`  
- `naive_bayes_model.pkl`  
- `rnn_model.keras` 
- `best_rnn_model.keras` 
- `tokenizer.pkl`  
- `vectorizer.pkl`  
- `label_encoder.pkl`

---

## 📧 Contact

For questions or collaboration: *ismailalhetimi@gmail.com*
