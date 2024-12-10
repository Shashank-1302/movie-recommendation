from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def preprocess_text(text):
    import re
    return re.sub(r'\W+', ' ', text).lower()

class RecommendationEngine:
    def __init__(self, data_path):
        self.movies = pd.read_csv(data_path)
        
        self.movies['combined_features'] = (
            self.movies['Genre'].fillna('') + ' ' +
            self.movies['Description'].fillna('')
        ).apply(preprocess_text)
        
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movies['combined_features'])

    def recommend_movies(self, user_input, top_n=5):
        user_input_processed = preprocess_text(user_input)
        user_vec = self.tfidf_vectorizer.transform([user_input_processed])
        
        similarity_scores = cosine_similarity(user_vec, self.tfidf_matrix)
        
        top_indices = similarity_scores.argsort()[0][-top_n:][::-1]
        
        return self.movies.iloc[top_indices][['Title', 'Genre', 'Description']]
