from flask import Flask, request, jsonify
from recommendation.recommendation import RecommendationEngine

app = Flask(__name__)

DATA_PATH = '../Dataset/imdb_movie_dataset.csv'
engine = RecommendationEngine(DATA_PATH)

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        user_input = data.get('preferences', '')

        if not user_input:
            return jsonify({"error": "Preferences field is required!"}), 400

        recommendations = engine.recommend_movies(user_input, top_n=5)
        recommendations_list = recommendations.to_dict(orient='records')

        return jsonify({
            "status": "success",
            "recommendations": recommendations_list
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
