import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

# Function to read reviews from files in a directory and assign sentiments
def load_reviews_from_directory(directory, sentiment):
    reviews = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                reviews.append(file.read())
    return pd.DataFrame({'review': reviews, 'sentiment': [sentiment] * len(reviews)})

# Load the training data
train_pos_dir = 'imdb/aclImdb/train/pos'  
train_neg_dir = 'imdb/aclImdb/train/neg' 

train_pos_reviews = load_reviews_from_directory(train_pos_dir, 'positive')
train_neg_reviews = load_reviews_from_directory(train_neg_dir, 'negative')

# Combine positive and negative reviews into a single training dataframe
train_df = pd.concat([train_pos_reviews, train_neg_reviews], ignore_index=True)

# Load the test data
test_pos_dir = 'imdb/aclImdb/test/pos'  
test_neg_dir = 'imdb/aclImdb/test/neg' 

test_pos_reviews = load_reviews_from_directory(test_pos_dir, 'positive')
test_neg_reviews = load_reviews_from_directory(test_neg_dir, 'negative')

# Combine positive and negative reviews into a single test dataframe
test_df = pd.concat([test_pos_reviews, test_neg_reviews], ignore_index=True)

# Show the first few rows of the training dataset
print(train_df.head())
print(test_df.head())

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=10000)
X_train = tfidf_vectorizer.fit_transform(train_df['review'])
X_test = tfidf_vectorizer.transform(test_df['review'])

# Encode sentiment labels to binary (positive = 1, negative = 0)
y_train = train_df['sentiment'].map({'positive': 1, 'negative': 0})
y_test = test_df['sentiment'].map({'positive': 1, 'negative': 0})

#Create and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict the sentiment for test data
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))

# #logistic

# Create and train the Logistic Regression model
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_lr = lr_model.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {accuracy_lr}")
print(classification_report(y_test, y_pred_lr))

#naive bayes

# Create and train the Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_nb = nb_model.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"Naive Bayes Accuracy: {accuracy_nb}")



# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['negative', 'positive'], yticklabels=['negative', 'positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()



# Save the trained model
joblib.dump(lr_model, 'lr_model.pkl')

# Save the TF-IDF vectorizer
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

print("Model and vectorizer saved successfully!")


#Project deployement on website
from flask import Flask, render_template, request, jsonify
import requests
from bs4 import BeautifulSoup
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
lr_model = joblib.load('lr_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# OMDb API Key
OMDB_API_KEY = "ba45b4ce"

def predict_sentiment(review):
    review_tfidf = tfidf_vectorizer.transform([review])
    sentiment = lr_model.predict(review_tfidf)
    sentiment_label = 'positive' if sentiment == 1 else 'negative'
    return sentiment_label

def get_imdb_id(movie_title):
    url = f"http://www.omdbapi.com/?t={movie_title}&apikey={OMDB_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:#this checks if api sends back an response
        data = response.json()
        if data.get("Response") == "True":#"Response"- is a variable which is present in json file recieved, if movie not found it will be false and an error 
            return data.get("imdbID"), data
        else:
            return None, {"error": data.get('Error')}
    return None, {"error": "Failed to fetch IMDb ID"}

def scrape_imdb_reviews(imdb_id):
    imdb_url = f"https://www.imdb.com/title/{imdb_id}/reviews"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(imdb_url, headers=headers)
    
    reviews_list = []
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        reviews = soup.find_all('div', {'class': 'ipc-html-content-inner-div'})
        
        for review in reviews:
            review_text = review.get_text(strip=True)
            reviews_list.append(review_text)
    return reviews_list

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    movie_title = request.form.get('movie_title')
    if not movie_title:
        return jsonify({"error": "Please enter a movie title"})

    imdb_id, movie_data = get_imdb_id(movie_title)
    if not imdb_id:
        return jsonify(movie_data)

    reviews = scrape_imdb_reviews(imdb_id)
    if not reviews:
        return jsonify({"error": "No reviews found"})

    results = []
    positive_count = 0
    negative_count = 0

    for review in reviews:
        sentiment = predict_sentiment(review)
        if sentiment == 'positive':
            positive_count += 1
        else:
            negative_count += 1
            
        results.append({
            'review': review[:500] + "..." if len(review) > 500 else review,
            'sentiment': sentiment
        })

    return jsonify({
        "movie_info": movie_data,
        "reviews": results,
        "stats": {
            "total": len(results),
            "positive": positive_count,
            "negative": negative_count,
            "positive_percentage": (positive_count / len(results)) * 100 if results else 0
        }
    })

@app.route('/test_model', methods=['POST'])
def test_model():
    review_text = request.form.get('review_text')
    if not review_text:
        return jsonify({"error": "Please enter a review to test."})

    sentiment = predict_sentiment(review_text)
    return jsonify({"sentiment": sentiment})


if __name__ == '__main__':
    app.run(debug=True)