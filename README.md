# 🎬 IMDb Review Classification with Naive Bayes & Flask API

## 📌 Project Overview
This project classifies IMDb movie reviews as **positive** or **negative** using the **Naive Bayes algorithm**. It implements two different approaches:

1. **Using Scikit-learn's Multinomial Naive Bayes model**
2. **A custom-built Multinomial Naive Bayes algorithm (without libraries)**

Additionally, it includes a **Flask-based API**, which runs inside **Docker**, and a **web interface** for user interaction.

---

## 📂 Project Structure
```
📁 task
│── 📁 notebooks
│   │── MultinomialNB_Sklearn.ipynb  # Naive Bayes with Scikit-learn
│   │── MultinomialNB_FromScratch.ipynb  # Custom Naive Bayes implementation
│   │── 📁 templates
│   │   └── index.html  # HTML web interface
│── app.py  # Flask API
│── Dockerfile  # Docker configuration
│── IMDB_Dataset.csv  # Dataset used for training
│── test_api.py  # API testing script
│── requirements.txt  # Dependencies
│── .gitignore  # Git ignore file
│── LICENSE  # Project license
│── README.md  # Project documentation
```

---

## 📊 Dataset

The dataset used in this project is the **IMDb Dataset of 50K Movie Reviews**, available on Kaggle: 📌 [IMDb Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---

## 🚀 Features
- **Naive Bayes implementation** with and without libraries.
- **IMDb review classification** (positive/negative sentiment analysis).
- **Flask API** for model inference.
- **Docker-ready environment** for easy deployment.
- **Web interface** for manual review classification.

---

## 🛠 Technologies Used
- **Python** (Numpy, Pandas, Scikit-learn, NLTK)
- **Flask** (API development)
- **Docker** (Containerization)
- **HTML, CSS, Bootstrap** (Web interface)
- **Jupyter Notebook** (Model training and evaluation)

---

## ⚙️ Installation & Setup
### 1️⃣ Clone the repository
```sh
git clone gh repo clone berkaykaratas07/IMDB-NB-Comparison
cd IMDB-NB-Comparison
```

### 2️⃣ Create a virtual environment (optional but recommended)
```sh
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3️⃣ Install dependencies
```sh
pip install -r requirements.txt
```

### 4️⃣ Run Flask API
```sh
python app.py
```

### 5️⃣ Access the web interface
Open your browser and go to:
```sh
http://localhost:5000
```

---

## 🐳 Running with Docker
1. **Build the Docker image**
   ```sh
   docker build -t imdb-flask-api .
   ```
2. **Run the container**
   ```sh
   docker run -p 5000:5000 imdb-flask-api
   ```
3. **Access the web interface**
   ```sh
   http://localhost:5000
   ```

---

## 📝 API Endpoints
### **1️⃣ Classify a review (POST /predict)**
#### **Request:**
```sh
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"review": "This movie was amazing!"}'
```
#### **Response:**
```json
{
    "review": "This movie was amazing!",
    "predicted_sentiment": "positive"
}
```

---

## 🔍 Results & Evaluation
- The **Scikit-learn model** achieves an accuracy of **86%**.
- The **custom-built model** achieves an accuracy of **86%**.
- Performance comparison and insights are available in the **notebooks** directory.

---

## 📜 License
This project is licensed under the **MIT License**.

---

## 🙌 Acknowledgments
- IMDb dataset is publicly available and used for research purposes.
- Inspired by machine learning and text classification techniques.

---

## 📬 Contact
For questions or suggestions, feel free to reach out:
- **GitHub:** [berkaykaratas07](https://github.com/berkaykaratas07)
- **Email:** berkaykaratas054@gmail.com

🚀 Happy coding!


