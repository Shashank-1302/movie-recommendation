from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def preprocess_text(text):
    import re
    return re.sub(r'\W+', ' ', text).lower().strip()

class RecommendationEngine:
    def __init__(self, data_path):

        self.movies = pd.read_csv(data_path)
        self.movies['combined_features'] = (
            self.movies['Genre'].fillna('') + ' ' * 3 +  
            self.movies['Description'].fillna('')
        ).apply(preprocess_text)

        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),  
            max_df=0.8  
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movies['combined_features'])

    def recommend_movies(self, user_input, top_n=5):

        user_input_processed = preprocess_text(user_input)

        self.movies['boosted_features'] = (
            self.movies['Genre'].fillna('') + ' ' * 3 +  
            self.movies['Description'].fillna('')
        ).apply(preprocess_text)

        boosted_tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movies['boosted_features'])
        user_vec = self.tfidf_vectorizer.transform([user_input_processed])

        similarity_scores = cosine_similarity(user_vec, boosted_tfidf_matrix)

        similarity_scores = similarity_scores.flatten()
        top_indices = similarity_scores.argsort()[-top_n:][::-1]

        recommendations = self.movies.iloc[top_indices][['Title', 'Genre', 'Description']].reset_index(drop=True)
        recommendations['Similarity'] = similarity_scores[top_indices]
        return recommendations