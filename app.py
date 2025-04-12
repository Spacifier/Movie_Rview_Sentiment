from flask import Flask, render_template, request, jsonify
import requests
from bs4 import BeautifulSoup
import joblib
import os

app = Flask(__name__)

# Load the trained model and vectorizer
rf_model = joblib.load('lr_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# OMDb API Key
OMDB_API_KEY = os.environ.get("OMDB_API_KEY", "ba45b4ce")

def predict_sentiment(review):
    review_tfidf = tfidf_vectorizer.transform([review])
    sentiment = rf_model.predict(review_tfidf)
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
    # Use this for local development
    # app.run(debug=True)
    
    # Use this for production
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)