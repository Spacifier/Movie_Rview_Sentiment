import requests
from bs4 import BeautifulSoup
import joblib
import time

# Load the trained model and vectorizer
rf_model = joblib.load('sentiment_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# OMDb API Key (replace with your API key)
OMDB_API_KEY = "ba45b4ce"

# Function to predict sentiment of a review
def predict_sentiment(review):
    review_tfidf = tfidf_vectorizer.transform([review])
    sentiment = rf_model.predict(review_tfidf)
    sentiment_label = 'positive' if sentiment == 1 else 'negative'
    return sentiment_label

# Function to get IMDb ID using OMDb API
def get_imdb_id(movie_title):
    url = f"http://www.omdbapi.com/?t={movie_title}&apikey={OMDB_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data.get("Response") == "True":
            return data.get("imdbID")
        else:
            print(f"Error: {data.get('Error')}")
    else:
        print("Failed to fetch IMDb ID.")
    return None

# Function to scrape reviews from IMDb
def scrape_imdb_reviews(imdb_id):
    imdb_url = f"https://www.imdb.com/title/{imdb_id}/reviews"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(imdb_url, headers=headers)

    print(f"Status Code: {response.status_code}")
    reviews_list = []
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        reviews = soup.find_all('div', {'class': 'ipc-html-content-inner-div'})
        
        for review in reviews:
            review_text = review.get_text(strip=True)
            reviews_list.append(review_text)
    else:
        print(f"Failed to retrieve data from {imdb_url}")
    return reviews_list

# Function to process reviews and predict sentiments
def process_reviews(movie_title):
    imdb_id = get_imdb_id(movie_title)
    if not imdb_id:
        print("Could not find IMDb ID for the movie.")
        return

    reviews = scrape_imdb_reviews(imdb_id)
    if not reviews:
        print("No reviews found.")
        return

    print(f"Total reviews found: {len(reviews)}")
    for i, review in enumerate(reviews, 1):
        sentiment = predict_sentiment(review)
        print(f"Review {i}: {review[:100]}...")  # Display first 100 characters for readability
        print(f"Predicted Sentiment: {sentiment}")
        print("-" * 80)

# User Input for Movie Title
movie_title = input("Enter the movie title: ")
process_reviews(movie_title)
