README.md
# SMS Spam Classifier (Naive Bayes)

A simple machine learning project that classifies SMS messages as **spam** or **ham** using Python and the Naive Bayes algorithm.
# SPAM CLASSIFIER USING MACHINE LEARNING

## 📌 Overview of the Project
The Spam Classifier is an intelligent machine learning–based system designed to automatically detect and categorize incoming text messages as either **Spam** or **Ham (non-spam)**. With a rapid increase in online communication, individuals and organizations receive huge volumes of unwanted messages every day. Manually filtering spam is inefficient, time-consuming, and unsafe. This project aims to create an automated solution using artificial intelligence that reads input text, analyzes patterns using trained datasets, and predicts whether the message is safe or spam with high accuracy.

The model uses Natural Language Processing (NLP) to analyze message content, converts text into numeric vector representation using TF-IDF and Bag-of-Words techniques, and trains a predictive classifier model using supervised learning algorithms such as **Multinomial Naive Bayes** and **Logistic Regression**. It evaluates accuracy using advanced evaluation metrics including Confusion Matrix, Accuracy Score, Precision, Recall, and F1-Score.

---

## ✨ Key Features
1. **Automated Spam Detection**
   - Classifies messages in real-time without manual intervention.
   - Prevents unwanted promotions, fraud and phishing attempts.

2. **Text Preprocessing**
   - Converts text to lowercase, removes punctuation, symbols, stopwords and unnecessary whitespace.
   - Improves model learning by standardizing input text.

3. **Vectorization (TF-IDF / Bag of Words)**
   - Converts raw text into numerical features understandable by ML algorithms.

4. **Model Training & Prediction**
   - Uses Naive Bayes or Logistic Regression for classification
   - Provides output label as *Spam* or *Not Spam* (Ham)

5. **Performance Visualization**
   - Confusion Matrix heatmap & accuracy score plotting using Matplotlib.

6. **Dataset Handling**
   - CSV dataset with thousands of labeled text samples for better learning.

---

## 🛠 Technology Stack & Tools Used
| Tool / Library | Purpose |
|----------------|---------|
| **Python** | Main programming language |
| **Scikit-Learn** | ML model training & evaluation |
| **Pandas, NumPy** | Dataset handling & numeric operations |
| **Matplotlib / Seaborn** | Result visualization |
| **NLTK** | NLP preprocessing & stopword removal |
| **TF-IDF Vectorizer** | Text-to-numeric conversion |

---

## 🚀 Steps to Install and Run the Project
### Step 1: Install Required Libraries

pip install numpy pandas scikit-learn matplotlib nltk seaborn

### Step 2: Place dataset `spam.csv` in the project folder

### Step 3: Run the classifier script

python spam_classifier.py

### Step 4: Enter a message to check spam manually

---

## 🧪 Testing Instructions
| Test Scenario | Expected Output |
|---------------|-----------------|
| Message with offers, discounts, free prizes | Spam |
| Friendly or academic messages | Ham |
| Empty / Single word input | Classified based on context |
| Unknown new messages | Predict based on learned patterns |

Verify:
- Accuracy Score
- Confusion Matrix
- Precision & Recall Score
- Re-train model with new dataset entries

---

## 🔮 Future Enhancements
- Deep learning model using LSTM / BERT for high-level accuracy
- SMS / Email live filtering using API
- Mobile app version & cloud deployment
- Voice-to-text spam detection system

