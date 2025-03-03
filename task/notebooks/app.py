import numpy as np
import pandas as pd
import re
import string
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# 📌 Dataset yükleme
df = pd.read_csv('IMDB_Dataset.csv')
df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# 📌 Metin ön işleme
def preprocess_text(text):
    text = re.sub(r'<[^<>]*>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    tokenizer = WhitespaceTokenizer()
    lemmatizer = WordNetLemmatizer()
    return ' '.join(lemmatizer.lemmatize(word) for word in tokenizer.tokenize(text) if word not in stop_words)

df['review'] = df['review'].apply(preprocess_text)

# 📌 Veriyi eğitim ve test setine ayır
train, test = train_test_split(df, test_size=0.25, random_state=42)
X_train, y_train = train['review'], train['sentiment']
X_test, y_test = test['review'], test['sentiment']

# 📌 Naive Bayes modelini eğit
vocab = set()
word_counts = defaultdict(lambda: [0, 0])
class_doc_counts = [0, 0]

for text, label in zip(X_train, y_train):
    words = text.split()
    class_doc_counts[label] += 1
    for word in words:
        vocab.add(word)
        word_counts[word][label] += 1

vocab_size = len(vocab)
total_words_per_class = {
    0: sum(word_counts[w][0] for w in vocab),
    1: sum(word_counts[w][1] for w in vocab)
}

def compute_prob(word, label):
    word_count = word_counts[word][label]
    total_words_in_class = total_words_per_class[label]
    return (word_count + 1) / (total_words_in_class + vocab_size)

# 📌 Model tahmin fonksiyonu
def predict(text):
    words = text.split()
    log_prob_0 = class_doc_counts[0] / sum(class_doc_counts)
    log_prob_1 = class_doc_counts[1] / sum(class_doc_counts)

    for word in words:
        if word in vocab:
            log_prob_0 += np.log(compute_prob(word, 0))
            log_prob_1 += np.log(compute_prob(word, 1))

    return 1 if log_prob_1 > log_prob_0 else 0

# 📌 API için tahmin endpoint'i
@app.route('/predict', methods=['POST'])
def predict_sentiment():
    try:
        review_text = request.form.get("review")  # Formdan veriyi al
        cleaned_text = preprocess_text(review_text)
        prediction = predict(cleaned_text)
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
        return render_template('index.html', review=review_text, prediction=sentiment)
    except Exception as e:
        return jsonify({"error": str(e)})

# 📌 Flask Sunucusunu Başlat
if __name__ == '__main__':
    app.run(debug=True)
